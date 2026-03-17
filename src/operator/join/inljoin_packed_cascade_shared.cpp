#include "join/inljoin_packed_cascade_shared.hpp"

#include "../../table/include/cardinality.hpp"
#include "../../table/include/ffx_str_t.hpp"
#include <cassert>
#include <cstring>
#include <iostream>
#include <type_traits>
#include <unordered_map>
#include <vector/bitmask.hpp>
#include <vector>

namespace ffx {

template<typename T>
static void register_node_in_saved_data_cascade_shared(FtreeStateUpdateNode* node,
                                                       VectorSavedData<T>* vector_saved_data,
                                                       uint32_t& saved_data_index);
template<typename T>
static void register_backward_nodes_in_saved_data_cascade_shared(FtreeStateUpdateNode* parent_node,
                                                                 VectorSavedData<T>* vector_saved_data,
                                                                 uint32_t& saved_data_index);

template<typename T>
void INLJoinPackedCascadeShared<T>::create_slice_update_infrastructure(FactorizedTreeElement* ftree_leaf) {
    _vector_saved_data = std::make_unique<VectorSavedData<T>[]>(_vector_saved_data_count);
    uint32_t saved_data_index = 0;
    const auto join_key_fnode = ftree_leaf->_parent;
    _range_update_tree->fill_bwd_join_key(/* parent node */ join_key_fnode, /* child to exclude */ _output_key);
    register_backward_nodes_in_saved_data_cascade_shared<T>(_range_update_tree.get(), _vector_saved_data.get(),
                                                            saved_data_index);

    FactorizedTreeElement* ftreenode = join_key_fnode->_parent;
    FtreeStateUpdateNode* current_node = _range_update_tree.get();
    while (ftreenode != nullptr) {
        auto child = std::make_unique<FtreeStateUpdateNode>(ftreenode->_value, FORWARD, ftreenode->_attribute);
        auto child_ptr = child.get();
        register_node_in_saved_data_cascade_shared<T>(child_ptr, _vector_saved_data.get(), saved_data_index);
        child_ptr->parent = current_node;
        current_node->children.push_back(std::move(child));
        // If the current node has children, we need to fill it with backward updates
        child_ptr->fill_bwd(ftreenode, _output_key);
        register_backward_nodes_in_saved_data_cascade_shared<T>(child_ptr, _vector_saved_data.get(), saved_data_index);
        current_node = current_node->children.back().get();
        ftreenode = ftreenode->_parent;
    }
}

template<typename T>
void INLJoinPackedCascadeShared<T>::init(Schema* schema) {
    static_assert(std::is_same_v<T, uint64_t>, "INLJoinPackedCascadeShared currently only supports uint64_t. String "
                                               "support requires hash-based adjacency list indexing.");
    auto& map = *schema->map;
    auto root = schema->root;

    _in_vec = map.get_vector(_join_key);

    // cascade bookkeeping
    {
        _invalidated_indices = std::make_unique<uint32_t[]>(State::MAX_VECTOR_SIZE);
        _invalidated_count = 0;
    }

    ResolvedJoinAdjList resolved;
    if (!schema->try_resolve_join_adj_list(_join_key, _output_key, resolved)) {
        throw std::runtime_error("No table found for join " + _join_key + " -> " + _output_key);
    }
    _adj_lists = reinterpret_cast<AdjList<T>*>(resolved.adj_list);
    _is_join_index_fwd = resolved.is_fwd;
    std::cout << "INLJoinPackedCascadeShared " << _join_key << "->" << _output_key
              << (resolved.from_schema_map ? " using Schema adj_list" : " using fallback table adj_list")
              << " (" << (resolved.is_fwd ? "fwd" : "bwd") << ") [SHARED STATE]" << std::endl;

    // Always allocate output vector with shared state
    _out_vec = map.allocate_vector_shared_state<T>(_output_key, _join_key);
    const auto ftree_leaf = root->add_leaf(_join_key, _output_key, _in_vec, _out_vec);
    _range_update_tree = std::make_unique<FtreeStateUpdateNode>(_in_vec, NONE, _join_key);

    const auto unique_datachunks = root->count_unique_datachunks();
    _vector_saved_data_count = root->get_num_nodes() - 2;// Exclude join_key and output_key nodes
    if (_vector_saved_data_count > 0 && unique_datachunks > 2) { create_slice_update_infrastructure(ftree_leaf); }
    else { _vector_saved_data_count = 0; }
    _active_mask_uptr = std::make_unique<BitMask<State::MAX_VECTOR_SIZE>>();
    _range_update_tree->precompute_effective_children();
    _next_op->init(schema);
}

template<typename T>
void INLJoinPackedCascadeShared<T>::store_slices() {
    for (std::size_t i = 0; i < _vector_saved_data_count; i++) {
        auto& vec_data = _vector_saved_data[i];
        const Vector<T>* vec = vec_data.vector;
        auto& [start_pos, end_pos] = vec_data.backup_state;
        auto& state = *vec->state;
        start_pos = GET_START_POS(state);
        end_pos = GET_END_POS(state);
        assert(start_pos < State::MAX_VECTOR_SIZE);
        assert(end_pos < State::MAX_VECTOR_SIZE);
        assert(start_pos <= end_pos);
    }
}

template<typename T>
void INLJoinPackedCascadeShared<T>::print_upstream_vectors() {
    for (std::size_t i = 0; i < _vector_saved_data_count; i++) {
        // auto& vec_data = _vector_saved_data[i];
        // const Vector* vec = vec_data.vector;
        // auto& state = *vec->state;
        // auto start_pos = GET_START_POS(state);
        // auto end_pos = GET_END_POS(state);
    }
}

template<typename T>
void INLJoinPackedCascadeShared<T>::restore_slices() {
    // Restore all vectors from unified backup store
    // This handles both forward and backward slice updates

    for (std::size_t i = 0; i < _vector_saved_data_count; i++) {
        auto& vec_data = _vector_saved_data[i];
        const Vector<T>* vec = vec_data.vector;
        const auto& [start_pos, end_pos] = vec_data.backup_state;
        auto& state = *vec->state;
        assert(start_pos < State::MAX_VECTOR_SIZE);
        assert(end_pos < State::MAX_VECTOR_SIZE);
        assert(start_pos <= end_pos);
        SET_START_POS(state, start_pos);
        SET_END_POS(state, end_pos);
    }
}

template<typename T>
void INLJoinPackedCascadeShared<T>::execute() {
    num_exec_call++;

    // Store ancestor slices before executing next operator
    store_slices();

    State* RESTRICT in_state = _in_vec->state;
    const T* RESTRICT in_vals = _in_vec->values;
    T* RESTRICT out_vals = _out_vec->values;

    // Save the original input selection mask pointer directly on the stack
    COPY_BITMASK(State::MAX_VECTOR_SIZE, _in_selector_backup, in_state->selector);

    // Copy the original mask into our active mask (single deep copy)
    COPY_BITMASK(State::MAX_VECTOR_SIZE, *_active_mask_uptr, _in_selector_backup);

    // Use our pre-allocated active mask as the temporary selection mask
    COPY_BITMASK(State::MAX_VECTOR_SIZE, in_state->selector, *_active_mask_uptr);

    // Get the active bit range from the bitmask
    const auto in_vec_start_idx = GET_START_POS(*in_state);
    const auto in_vec_end_idx = GET_END_POS(*in_state);

    // Shared-state path: one output per input pos, no dense packing, no out_offset writes.
    // Use temporary buffer for better cache performance
    T temp_out_vals[State::MAX_VECTOR_SIZE];
    _invalidated_count = 0;
    for (auto pidx = static_cast<int32_t>(next_set_bit_in_range(*_active_mask_uptr, in_vec_start_idx,
                                                                 in_vec_end_idx));
         pidx <= in_vec_end_idx;
         pidx = static_cast<int32_t>(next_set_bit_in_range(*_active_mask_uptr, static_cast<uint32_t>(pidx + 1),
                                                            in_vec_end_idx))) {
        const auto& adj_list = _adj_lists[in_vals[pidx]];
        if (adj_list.size == 0) {
            CLEAR_BIT(*_active_mask_uptr, pidx);
            _invalidated_indices[_invalidated_count++] = static_cast<uint32_t>(pidx);
            continue;
        }
        assert(adj_list.size == 1 && "State sharing requires exactly one output per input position");
        temp_out_vals[pidx] = adj_list.values[0];
    }
    // Copy all values in one sweep (garbage values don't matter, bitmask handles validity)
    std::memcpy(&out_vals[in_vec_start_idx], &temp_out_vals[in_vec_start_idx],
                (in_vec_end_idx - in_vec_start_idx + 1) * sizeof(T));

    // Shrink start/end to first/last set bit
    int32_t new_start = in_vec_end_idx + 1;
    int32_t new_end = in_vec_start_idx - 1;
    new_start = static_cast<int32_t>(next_set_bit_in_range(*_active_mask_uptr, in_vec_start_idx, in_vec_end_idx));
    for (int32_t p = new_start; p <= in_vec_end_idx;
         p = static_cast<int32_t>(next_set_bit_in_range(*_active_mask_uptr, static_cast<uint32_t>(p + 1),
                                                        in_vec_end_idx))) {
        new_end = p;
    }

    if (new_start <= new_end) {
        SET_START_POS(*in_state, new_start);
        SET_END_POS(*in_state, new_end);
        COPY_BITMASK(State::MAX_VECTOR_SIZE, in_state->selector, *_active_mask_uptr);

        auto is_vector_empty = _range_update_tree->start_propagation();
        if (!is_vector_empty && (_invalidated_count > 0)) {
            is_vector_empty =
                    _range_update_tree->start_propagation_cascade(_invalidated_indices.get(), _invalidated_count);
        }
        _invalidated_count = 0;
        if (!is_vector_empty) { _next_op->execute(); }

        restore_slices();

        // When state is shared, restore_slices() may have restored the input/output state's start_pos/end_pos
        // to backup values. We need to re-apply the updated values because the output vector needs them
        // for downstream operators (e.g., sink) to read correctly.
        // Also ensure the selector stays as the active mask (not restored to backup).
        SET_START_POS(*in_state, new_start);
        SET_END_POS(*in_state, new_end);
        COPY_BITMASK(State::MAX_VECTOR_SIZE, in_state->selector,
                     *_active_mask_uptr);// Ensure selector is active mask, not backup
    } else {
        _invalidated_count = 0;
    }

    // When state is shared, do NOT restore input state because output vector shares the same state
    // and needs to keep the updated state (selector, start_pos, end_pos) for downstream operators (e.g., sink) to read correctly
    // The active mask contains the filtered bits that the output vector needs
    // Do NOT restore selector, start_pos, or end_pos - they are shared with output vector
}

template<typename T>
static void register_node_in_saved_data_cascade_shared(FtreeStateUpdateNode* node,
                                                       VectorSavedData<T>* vector_saved_data,
                                                       uint32_t& saved_data_index) {
    // Get current start and end positions from the vector's selector
    int32_t current_start = GET_START_POS(*node->vector->state);
    int32_t current_end = GET_END_POS(*node->vector->state);

    // Create VectorSavedData entry
    vector_saved_data[saved_data_index++] =
            VectorSavedData<T>(node->attribute, const_cast<Vector<T>*>(node->vector), current_start, current_end);
}

template<typename T>
static void register_backward_nodes_in_saved_data_cascade_shared(FtreeStateUpdateNode* parent_node,
                                                                 VectorSavedData<T>* vector_saved_data,
                                                                 uint32_t& saved_data_index) {
    for (const auto& child: parent_node->children) {
        // Register this backward node
        register_node_in_saved_data_cascade_shared<T>(child.get(), vector_saved_data, saved_data_index);

        // Recursively register its children
        register_backward_nodes_in_saved_data_cascade_shared<T>(child.get(), vector_saved_data, saved_data_index);
    }
}

// Explicit template instantiations
template class INLJoinPackedCascadeShared<uint64_t>;
// template class INLJoinPackedCascadeShared<ffx_str_t>; // TODO: Enable when hash-based adjacency list indexing is implemented

}// namespace ffx
