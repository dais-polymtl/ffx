#ifndef VFENGINE_FTREE_BATCH_ITERATOR_HH
#define VFENGINE_FTREE_BATCH_ITERATOR_HH

#include "factorized_tree_element.hpp"
#include "ftree_iterator.hpp"
#include "operator.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <cstdlib>
#include <vector>

namespace ffx {

/// Lightweight span for child indices (replaces std::vector for get_children)
struct ChildSpan {
    const size_t* data = nullptr;
    size_t _size = 0;
    const size_t* begin() const { return data; }
    const size_t* end() const { return data + _size; }
    bool empty() const { return _size == 0; }
    size_t size() const { return _size; }
    size_t operator[](size_t i) const { return data[i]; }
};

class FTreeBatchIterator final : public Operator {
public:
    struct FullTreeInfo {
        size_t num_nodes = 0;
        const FactorizedTreeElement* const* nodes = nullptr;
        const int32_t* parent_idx = nullptr;
        const int32_t* full_to_projected = nullptr;
        const int32_t* projected_to_full = nullptr;
        const size_t* children_flat = nullptr;
        const size_t* children_start = nullptr;
        const size_t* children_counts = nullptr;
    };

    explicit FTreeBatchIterator(const std::unordered_map<std::string, size_t>& required_attributes);
    ~FTreeBatchIterator() override = default;

    void init(Schema* schema) override;
    void execute() override {}
    uint64_t get_num_output_tuples() override { return _num_output_tuples; }

    bool next();
    void reset();
    void initialize_iterators();
    bool is_valid() const { return _is_valid; }

    size_t num_attributes() const { return _num_attributes; }
    size_t tuple_size() const { return _num_attributes; }

    const uint64_t* get_buffer(size_t attr_idx) const { return _node_buffer[attr_idx].get(); }
    const uint32_t* get_node_offset(size_t attr_idx) const { return _node_offset[attr_idx].get(); }
    size_t get_count(size_t attr_idx) const { return _output_counts[attr_idx]; }
    bool is_leaf_attr(size_t attr_idx) const { return _is_leaf[attr_idx]; }
    const int32_t* get_positions(size_t attr_idx) const { return _node_pos[attr_idx].get(); }
    ChildSpan get_children(size_t attr_idx) const {
        ChildSpan s;
        s.data = _children_indices[attr_idx].get();
        s._size = _children_counts[attr_idx];
        return s;
    }

    const size_t* counts() const { return _output_counts.get(); }
    const bool* leaf_flags() const { return _is_leaf.get(); }

    /// Returns the number of logical (fully expanded) tuples in the current batch.
    /// Must be called after next() returns true, when buffers are filled.
    size_t count_logical_tuples() const;

    /// Zero-copy view over the iterator's full-tree internals.
    FullTreeInfo get_full_tree_info() const;

    /// Returns the Steiner set (full-tree indices) computed during init.
    /// Includes context_column attributes traced to root, plus same-state children.
    const std::unordered_set<size_t>& get_steiner_set() const { return _steiner_full_indices; }

public:
    static void find_leaf_attributes(FactorizedTreeElement* node, std::vector<std::string>& out);
    __attribute__((always_inline)) inline bool reset_iterator_to_start(size_t idx);
    __attribute__((always_inline)) inline bool try_advance(size_t idx);
    __attribute__((always_inline)) inline int32_t find_next_set_bit_root(const SimpleLocalIterator& itr) const;
    __attribute__((always_inline)) inline int32_t find_next_set_bit_child(const SimpleLocalIterator& itr,
                                                                          int32_t parent_pos) const;

    void append_node_value(size_t idx);
    void append_full_tree_positions(size_t full_idx, int32_t pos);
    void append_full_tree_block(size_t full_idx, size_t block_idx, uint64_t block_mask);
    void clear_full_tree_subtree_counts(size_t full_idx);
    void build_full_tree_structure();
    enum class FillStatus { SUCCESS_EXHAUSTED = 0, BLOCKED_FULL = 1, BLOCKED_PARTIAL = 2 };
    FillStatus fill_leaf_window(size_t node_idx);

    FillStatus greedy_fill_subtree(size_t parent_idx);
    void greedy_fill_all();
    void fill_remaining_nodes(size_t start_idx);
    void cascade_down_from(size_t idx);
    bool yield_batch();

    int find_rightmost_remaining_leaf() const;
    bool check_any_leaf_has_data() const;
    void finalize_batch_offsets();
    size_t count_logical_tuples_rec(const int32_t* flat_pos, const size_t* pos_starts,
                                    const size_t* pos_counts, size_t full_idx,
                                    size_t pos_start, size_t pos_end) const;
    bool compute_has_projected(size_t full_idx);
    void debug_print_projected_ftree() const;
    void debug_print_internal_state() const;

    std::unordered_map<std::string, size_t> _required_attributes;
    std::unique_ptr<std::unique_ptr<uint16_t[]>[]> _custom_offsets;
    std::unique_ptr<size_t[]> _custom_offset_sizes;
    size_t _num_custom_offsets = 0;
    Schema* _schema = nullptr;
    std::unique_ptr<FactorizedTreeElement*[]> _nodes;

    // Full tree: all nodes for internal position tracking (not exposed in output)
    std::unique_ptr<FactorizedTreeElement*[]> _full_tree_nodes;
    size_t _num_full_tree_nodes = 0;
    std::unique_ptr<int32_t[]> _full_tree_parent_idx;
    std::unique_ptr<size_t[]> _full_tree_children_flat;    // flat array of all children indices
    std::unique_ptr<size_t[]> _full_tree_children_start;   // start index per node into _full_tree_children_flat
    std::unique_ptr<size_t[]> _full_tree_children_counts;  // number of children per node
    std::unique_ptr<int32_t[]> _full_to_projected;  // full_idx -> projected idx, -1 if not required
    std::unique_ptr<int32_t[]> _projected_to_full;  // projected idx -> full_idx
    std::unique_ptr<bool[]> _full_tree_has_projected;  // true if subtree contains any projected node
    // Reverse offset: child_pos → parent_pos, O(1) lookup replacing binary search.
    // nullptr for root nodes. Indexed by child position within state range.
    std::unique_ptr<std::unique_ptr<int16_t[]>[]> _full_tree_reverse_offset;
    // True if child→parent mapping is identity (shared state).
    std::unique_ptr<bool[]> _full_tree_identity_parent;
    // Per-node bitsets tracking which positions were visited during batch filling.
    // Flat layout: _full_tree_node_bits[i * NUM_BLOCKS + b] = block b of node i.
    static constexpr size_t NUM_BLOCKS = State::MAX_VECTOR_SIZE / 64;
    std::unique_ptr<uint64_t[]> _full_tree_node_bits;

    size_t _num_attributes = 0;
    std::unique_ptr<std::unique_ptr<size_t[]>[]> _children_indices;
    std::unique_ptr<size_t[]> _children_counts;
    std::unique_ptr<bool[]> _is_leaf;
    std::unique_ptr<size_t[]> _node_capacity;

    std::unique_ptr<SimpleLocalIterator[]> _iterators;
    std::unique_ptr<std::unique_ptr<uint64_t[]>[]> _node_buffer;
    std::unique_ptr<std::unique_ptr<int32_t[]>[]> _node_pos;
    std::unique_ptr<std::unique_ptr<uint32_t[]>[]> _node_offset;
    std::unique_ptr<size_t[]> _output_counts;

    std::shared_ptr<FactorizedTreeElement> _root;
    bool _is_valid = false;
    bool _first_call = true;
    uint64_t _num_output_tuples = 0;

    // Cached original state/values for projected nodes. Captured during init()
    // so that the batch iterator remains functional even if the reconstructor
    // modifies _value on the original FactorizedTreeElement nodes.
    std::unique_ptr<const State*[]> _cached_states;
    std::unique_ptr<uint64_t*[]> _cached_values;
    // Cached states for full tree nodes (used in count_logical_tuples_rec).
    std::unique_ptr<const State*[]> _cached_full_tree_states;

    // Steiner set: full-tree indices of context_column attrs traced to root,
    // expanded with same-state children.
    std::unordered_set<size_t> _steiner_full_indices;
    void compute_steiner_set();
};

}// namespace ffx

#endif
