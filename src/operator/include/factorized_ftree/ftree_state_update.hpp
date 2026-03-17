#ifndef VFENGINE_FTREE_STATE_UPDATE_TREE_HH
#define VFENGINE_FTREE_STATE_UPDATE_TREE_HH

#include "factorized_tree_element.hpp"
#include "vector/vector.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ffx {
enum slice_update_type { NONE, FORWARD, BACKWARD };
struct CascadePropResult;

struct FtreeStateUpdateNode {
    FtreeStateUpdateNode(const Vector<uint64_t>* vec, slice_update_type t, const std::string& attr);
    void fill_bwd(FactorizedTreeElement* ftree, const std::string& output_key);
    void fill_bwd_join_key(FactorizedTreeElement* ftree, const std::string& output_key);
    bool start_propagation();
    bool update_range(bool& is_vector_empty);
    void print_tree(int depth = 0) const;
    /// Prints the compact propagation graph (follows effective_children only; call after precompute_effective_children).
    void print_effective_tree(int depth = 0) const;
    bool start_propagation_fwd_cascade(const uint32_t* gp_invalidated_indices, const int32_t& gp_invalidated_count);
    bool start_propagation_cascade(const uint32_t* invalidated_indices, const int32_t& invalidated_count);
    bool update_range_cascade(bool& is_vector_empty, const CascadePropResult& cascade_result);

    // Must be called once after the full tree is built (after fill_bwd / fill_bwd_join_key).
    // Post-order: children first, then this node. For each direct child subtree, collects the
    // first nodes along each branch whose State* differs from this node's State* (same-state
    // segments from this node toward leaves are compacted away).
    void precompute_effective_children();

    const Vector<uint64_t>* vector;
    slice_update_type type;
    FtreeStateUpdateNode* parent;
    std::vector<std::unique_ptr<FtreeStateUpdateNode>> children;
    std::string attribute;

    // Precomputed during init: first State* boundary per branch below this node (same-state
    // chains from this node toward leaves are omitted). Raw pointers — lifetime from children.
    std::vector<FtreeStateUpdateNode*> effective_children;
};
}// namespace ffx
#endif
