#ifndef FFX_AI_OPERATOR_FTREE_RECONSTRUCTOR_HH
#define FFX_AI_OPERATOR_FTREE_RECONSTRUCTOR_HH

#include "factorized_ftree/factorized_tree_element.hpp"
#include "factorized_ftree/ftree_ancestor_finder.hpp"
#include "factorized_ftree/ftree_batch_iterator.hpp"
#include "vector/state.hpp"
#include "vector/vector.hpp"
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ffx {

struct Schema;

// Lightweight container to pass LLM results from Map to FTreeReconstructor.
// One result per logical tuple, in DFS order matching count_logical_tuples().
struct LLMResultBatch {
    const uint64_t* results;
    size_t count;
    enum class Granularity : uint8_t { PER_TUPLE = 0, PER_ROOT = 1 };
    Granularity granularity = Granularity::PER_TUPLE;
};

/// Reconstructs a factorized tree by linearizing the Steiner set into a chain
/// and modifying the original ftree in-place.
///
/// The Steiner set is read from the batch iterator (not computed here).
/// Steiner nodes are rewired into a linear chain with new vectors/states.
/// Non-Steiner nodes keep their original topology but get new vectors/states.
/// Same-state sharing from the original tree is preserved in new allocations.
class FTreeReconstructor {
public:
    FTreeReconstructor() = delete;

    /// \param output_attr Name of the output attribute (e.g. "_llm")
    /// \param attribute_names Context column attribute names (for serialization ordering)
    FTreeReconstructor(std::string output_attr, std::vector<std::string> attribute_names);
    ~FTreeReconstructor() = default;

    /// Initialize: reads Steiner set from batch iterator, linearizes into chain,
    /// modifies original ftree in-place (new vectors, rewired Steiner children).
    /// Does NOT mutate schema->root or schema->column_ordering directly.
    /// The owning operator should call get_new_root() and get_visible_ordering()
    /// to apply schema mutations before calling next_op->init().
    void init(Schema* schema, FTreeBatchIterator* batch_iterator, std::function<void()> flush_callback);

    /// Append a batch of LLM results and reconstruct. May call flush_callback
    /// multiple times if the data exceeds MAX_VECTOR_SIZE.
    void append(const LLMResultBatch& batch);

    uint64_t get_num_output_tuples() const { return _num_output_tuples; }

    /// Returns the (potentially modified) original root.
    std::shared_ptr<FactorizedTreeElement> get_new_root() const { return _root; }

    /// Get the visible column ordering (query ordering + output attr).
    const std::vector<std::string>& get_visible_ordering() const { return _visible_ordering; }

private:
    std::string _output_attr;
    std::vector<std::string> _attribute_names;
    Schema* _schema;
    FTreeBatchIterator* _batch_iterator;
    std::function<void()> _flush_callback;
    uint32_t _num_attr;
    uint64_t _num_output_tuples;

    // The original root (modified in-place).
    std::shared_ptr<FactorizedTreeElement> _root;

    // Linearized Steiner chain + _llm output at end.
    std::vector<std::string> _chain_attrs;
    std::vector<size_t> _chain_full_idx;
    std::vector<bool> _chain_is_projected;

    // For each chain node i, the chain index of its tree-parent in the original ftree.
    // -1 for root. Used to detect direct-parent vs cross-branch edges.
    std::vector<int32_t> _chain_tree_parent_idx;

    // True if chain[i] shares state with chain[i-1] AND chain[i]'s tree parent
    // is chain[i-1] (direct parent with identity offset). For these nodes,
    // offset writing is skipped during fill.
    std::vector<bool> _chain_shares_state_with_prev;

    // Per-chain-node State pointers. May be shared for adjacent same-state nodes.
    // _chain_states[i] may equal _chain_states[i-1] when _chain_shares_state_with_prev[i].
    std::vector<State*> _chain_states;

    // Vectors owned by the reconstructor. Original ftree nodes' _value pointers
    // are updated to point to these.
    std::vector<std::unique_ptr<Vector<uint64_t>>> _owned_vectors;
    std::vector<Vector<uint64_t>*> _map_vectors;

    // Shared_ptrs held for lifetime management of reparented Steiner nodes.
    std::vector<std::shared_ptr<FactorizedTreeElement>> _held_nodes;
    // The new _llm output node (created fresh, added as child of last chain node).
    std::shared_ptr<FactorizedTreeElement> _llm_node;

    std::unordered_map<std::string, uint32_t> _attr_to_col;
    std::vector<std::string> _visible_ordering;

    int32_t compute_steiner_height(size_t full_idx, const FTreeBatchIterator::FullTreeInfo& info,
                                   const std::unordered_set<size_t>& steiner,
                                   std::unordered_map<size_t, int32_t>& memo) const;
    void linearize_steiner(size_t full_idx, const FTreeBatchIterator::FullTreeInfo& info,
                           const std::unordered_set<size_t>& steiner, std::unordered_map<size_t, int32_t>& height_memo);
    // Persistent chunking state across fill calls within one append()
    std::vector<uint32_t> _write_counts;
    // Precomputed ancestor positions for expand_cross_branch (avoids repeated binary search).
    std::vector<uint32_t> _ancestor_map;
    // Precomputed state paths [tree_parent_chain .. chain_idx-1] for each chain node.
    std::vector<std::vector<const State*>> _ancestor_state_paths;
    std::unique_ptr<FtreeAncestorFinder> _ancestor_finder;
    size_t _llm_read_idx;

    // Cached batch pointer for use during append/fill
    const LLMResultBatch* _llm_batch;

    // Non-Steiner nodes: projected full-tree nodes NOT in the Steiner chain.
    // Original topology is preserved (no parent/children changes).
    // New vectors/states are allocated; values filled relative to actual parent.
    struct NonSteinerNode {
        size_t full_idx;     // Index in the batch iterator's full tree
        int32_t proj_col;    // Projected column index in the batch iterator
        bool parent_is_chain;// true if parent is a chain node, false if parent is another NS node
        size_t parent_idx;   // Index into _write_counts (chain) or _ns_write_counts (NS)
    };
    std::vector<NonSteinerNode> _ns_info;
    std::vector<State*> _ns_states;
    std::vector<std::unique_ptr<Vector<uint64_t>>> _ns_owned_vectors;
    std::vector<Vector<uint64_t>*> _ns_map_vectors;
    std::vector<uint32_t> _ns_write_counts;

    void finalize_state_for_count(State* state, uint32_t count);
    void fill_chain_from_iterator();
    void fill_non_steiner_from_iterator(const FTreeBatchIterator::FullTreeInfo& info);
    void fill_downstream(size_t start_idx, const FTreeBatchIterator::FullTreeInfo& info);
    void expand_cross_branch(size_t chain_idx, const FTreeBatchIterator::FullTreeInfo& info);
    void place_llm_outputs(const FTreeBatchIterator::FullTreeInfo& info);
    void flush_at_node(size_t chain_idx, uint32_t parent_pos, uint32_t chunk_start,
                       const FTreeBatchIterator::FullTreeInfo& info);
    uint32_t trace_to_ancestor(size_t from_chain_idx, size_t to_chain_idx, uint32_t pos) const;
    void debug_print_constructed_tree() const;
    void debug_print_chain_values() const;
};

}// namespace ffx

#endif
