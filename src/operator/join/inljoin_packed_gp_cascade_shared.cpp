#include "join/inljoin_packed_gp_cascade_shared.hpp"

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
static void register_node_in_saved_data_gp_shared(FtreeStateUpdateNode* node, VectorGPSavedData<T>* vector_saved_data,
                                                  uint32_t& saved_data_index);
template<typename T>
static void register_backward_nodes_in_saved_data_gp_shared(FtreeStateUpdateNode* parent_node,
                                                            VectorGPSavedData<T>* vector_saved_data,
                                                            uint32_t& saved_data_index);

template<typename T>
void INLJoinPackedGPCascadeShared<T>::create_slice_update_infrastructure(const FactorizedTreeElement* ftree_leaf) {
    _vector_saved_data = std::make_unique<VectorGPSavedData<T>[]>(_vector_saved_data_count);
    uint32_t saved_data_index = 0;
    const auto join_key_fnode = ftree_leaf->_parent;
    _range_update_tree->fill_bwd_join_key(/* parent node */ join_key_fnode, /* child to exclude */ _output_key);
    register_backward_nodes_in_saved_data_gp_shared<T>(_range_update_tree.get(), _vector_saved_data.get(),
                                                       saved_data_index);

    FactorizedTreeElement* ftreenode = join_key_fnode->_parent;
    FtreeStateUpdateNode* current_node = _range_update_tree.get();
    while (ftreenode != nullptr) {
        auto child = std::make_unique<FtreeStateUpdateNode>(ftreenode->_value, FORWARD, ftreenode->_attribute);
        auto child_ptr = child.get();
        register_node_in_saved_data_gp_shared<T>(child_ptr, _vector_saved_data.get(), saved_data_index);
        child_ptr->parent = current_node;
        current_node->children.push_back(std::move(child));
        // If the current node has children, we need to fill it with backward updates
        child_ptr->fill_bwd(ftreenode, _output_key);
        register_backward_nodes_in_saved_data_gp_shared<T>(child_ptr, _vector_saved_data.get(), saved_data_index);
        current_node = current_node->children.back().get();
        ftreenode = ftreenode->_parent;
    }
}

template<typename T>
void INLJoinPackedGPCascadeShared<T>::init(Schema* schema) {
    static_assert(std::is_same_v<T, uint64_t>, "INLJoinPackedGPCascadeShared currently only supports uint64_t. String "
                                               "support requires hash-based adjacency list indexing.");
    auto& map = *schema->map;
    auto root = schema->root;

    _in_vec = map.get_vector(_join_key);

    ResolvedJoinAdjList resolved;
    if (!schema->try_resolve_join_adj_list(_join_key, _output_key, resolved)) {
        throw std::runtime_error("No table found for join " + _join_key + " -> " + _output_key);
    }
    _adj_lists = reinterpret_cast<AdjList<T>*>(resolved.adj_list);
    _is_join_index_fwd = resolved.is_fwd;
    std::cout << "INLJoinPackedGPCascadeShared " << _join_key << "->" << _output_key
              << (resolved.from_schema_map ? " using Schema adj_list" : " using fallback table adj_list")
              << " (" << (resolved.is_fwd ? "fwd" : "bwd") << ") [SHARED STATE]" << std::endl;

    // Always allocate output vector with shared state
    _out_vec = map.allocate_vector_shared_state<T>(_output_key, _join_key);
    const auto ftree_leaf = root->add_leaf(_join_key, _output_key, _in_vec, _out_vec);
    _range_update_tree = std::make_unique<FtreeStateUpdateNode>(_in_vec, NONE, _join_key);

    // cascade bookkeeping (needs ftree_leaf)
    {
        const auto& join_key_ftree_node = ftree_leaf->_parent;
        const auto& grandparent_ftree_node = join_key_ftree_node->_parent;
        assert(grandparent_ftree_node != nullptr);
        _grandparent_state = grandparent_ftree_node->_value->state;
        assert(_grandparent_state != nullptr);
        _invalid_gp_indices = std::make_unique<uint32_t[]>(State::MAX_VECTOR_SIZE);
        _invalid_gp_indices_cnt = 0;
    }

    const auto unique_datachunks = root->count_unique_datachunks();
    _vector_saved_data_count = root->get_num_nodes() - 2;// Exclude join_key and output_key nodes
    if (_vector_saved_data_count > 0 && unique_datachunks > 3) { create_slice_update_infrastructure(ftree_leaf); }
    else { _vector_saved_data_count = 0; }
    _active_mask_uptr = std::make_unique<BitMask<State::MAX_VECTOR_SIZE>>();
    _range_update_tree->precompute_effective_children();
    _next_op->init(schema);
}

template<typename T>
void INLJoinPackedGPCascadeShared<T>::store_slices() {
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
void INLJoinPackedGPCascadeShared<T>::print_upstream_vectors() {
    for (std::size_t i = 0; i < _vector_saved_data_count; i++) {
        // auto& vec_data = _vector_saved_data[i];
        // const Vector* vec = vec_data.vector;
        // auto& state = *vec->state;
        // auto start_pos = GET_START_POS(state);
        // auto end_pos = GET_END_POS(state);
    }
}

template<typename T>
void INLJoinPackedGPCascadeShared<T>::restore_slices() {
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
void INLJoinPackedGPCascadeShared<T>::execute() {
    num_exec_call++;

    // Store ancestor slices before executing next operator
    store_slices();

    State* RESTRICT in_state = _in_vec->state;
    const T* RESTRICT in_vals = _in_vec->values;
    T* RESTRICT out_vals = _out_vec->values;

    // Check if input vector has identity RLE (shares state with grandparent)
    const bool has_identity_rle = _in_vec->has_identity_rle();
    uint16_t* RESTRICT in_offset = has_identity_rle ? nullptr : in_state->offset;

    // Save the original input selection mask pointer directly on the stack
    COPY_BITMASK(State::MAX_VECTOR_SIZE, _in_selector_backup, in_state->selector);

    // Copy the original mask into our active mask (single deep copy)
    COPY_BITMASK(State::MAX_VECTOR_SIZE, *_active_mask_uptr, _in_selector_backup);

    // Use our pre-allocated active mask as the temporary selection mask
    COPY_BITMASK(State::MAX_VECTOR_SIZE, in_state->selector, *_active_mask_uptr);

    // For cascade, get the grandparent start and end positions
    const auto gp_start_pos = GET_START_POS(*_grandparent_state);
    const auto gp_end_pos = GET_END_POS(*_grandparent_state);

    // Get the active bit range from the bitmask
    const auto in_vec_start_idx = GET_START_POS(*in_state);
    const auto in_vec_end_idx = GET_END_POS(*in_state);

    // Shared-state path: one output per input pos, no dense packing / no out_offset writes.
    // Use temporary buffer for better cache performance
    T temp_out_vals[State::MAX_VECTOR_SIZE];
    _invalid_gp_indices_cnt = 0;

    // Iterate gp buckets and fill output values at the same indices as input positions.
    for (auto gp_idx = static_cast<int32_t>(
                 next_set_bit_in_range(_grandparent_state->selector, gp_start_pos, gp_end_pos));
         gp_idx <= gp_end_pos;
         gp_idx = static_cast<int32_t>(
                 next_set_bit_in_range(_grandparent_state->selector, static_cast<uint32_t>(gp_idx + 1), gp_end_pos))) {

        // Handle identity RLE: if input shares state with grandparent, mapping is 1:1
        int32_t pstart_idx, pend_idx;
        if (has_identity_rle) {
            // Identity RLE: each grandparent position maps to the same input position
            pstart_idx = std::max(static_cast<int32_t>(gp_idx), static_cast<int32_t>(in_vec_start_idx));
            pend_idx = std::min(static_cast<int32_t>(gp_idx), static_cast<int32_t>(in_vec_end_idx));
        } else {
            pstart_idx = std::max(static_cast<int32_t>(in_offset[gp_idx]), static_cast<int32_t>(in_vec_start_idx));
            pend_idx = std::min(static_cast<int32_t>(in_offset[gp_idx + 1] - 1), static_cast<int32_t>(in_vec_end_idx));
        }

        uint32_t parent_count = 0;
        if (bool is_first_or_last = (gp_idx == gp_start_pos) || (gp_idx == gp_end_pos)) {
            const std::size_t start_block = pstart_idx >> 6;
            const std::size_t end_block = pend_idx >> 6;
            const std::size_t start_bit = pstart_idx & 63;
            const std::size_t end_bit = pend_idx & 63;

            const uint64_t first_mask = ~0ULL << start_bit;
            const uint64_t first_block_mask =
                    (start_block == end_block) ? (first_mask & ((1ULL << (end_bit + 1)) - 1)) : first_mask;
            parent_count = __builtin_popcountll(_in_selector_backup.bits[start_block] & first_block_mask);

            for (std::size_t block = start_block + 1; block < end_block; ++block) {
                parent_count += __builtin_popcountll(_in_selector_backup.bits[block]);
            }
            if (start_block < end_block) {
                const uint64_t last_mask = (1ULL << (end_bit + 1)) - 1;
                parent_count += __builtin_popcountll(_in_selector_backup.bits[end_block] & last_mask);
            }
        } else {
            if (has_identity_rle) {
                // Identity RLE: exactly one input position per grandparent position
                parent_count = (pstart_idx <= pend_idx && TEST_BIT(_in_selector_backup, gp_idx)) ? 1 : 0;
            } else {
                parent_count = in_offset[gp_idx + 1] - in_offset[gp_idx];
            }
        }

        const uint32_t range_len = (pstart_idx <= pend_idx) ? static_cast<uint32_t>(pend_idx - pstart_idx + 1) : 0u;
        uint32_t active_in_range = 0;
        for (auto in_vector_idx = static_cast<int32_t>(next_set_bit_in_range(*_active_mask_uptr, pstart_idx,
                                                                              pend_idx));
             in_vector_idx <= pend_idx;
             in_vector_idx = static_cast<int32_t>(
                     next_set_bit_in_range(*_active_mask_uptr, static_cast<uint32_t>(in_vector_idx + 1),
                                           pend_idx))) {
            active_in_range++;
            const auto& adj_list = _adj_lists[in_vals[in_vector_idx]];
            if (adj_list.size == 0) {
                CLEAR_BIT(*_active_mask_uptr, in_vector_idx);
                parent_count--;
            } else {
                assert(adj_list.size == 1 && "State sharing requires exactly one output per input position");
                temp_out_vals[in_vector_idx] = adj_list.values[0];
            }
        }
        parent_count -= (range_len - active_in_range);

        _invalid_gp_indices[_invalid_gp_indices_cnt] = gp_idx;
        _invalid_gp_indices_cnt += parent_count == 0;
    }
    // Copy all values in one sweep (garbage values don't matter, bitmask handles validity)
    std::memcpy(&out_vals[in_vec_start_idx], &temp_out_vals[in_vec_start_idx],
                (in_vec_end_idx - in_vec_start_idx + 1) * sizeof(T));

    // Shrink start/end to first/last valid for the input/output shared state
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
    }

    COPY_BITMASK(State::MAX_VECTOR_SIZE, in_state->selector, *_active_mask_uptr);
    auto is_vector_empty = _range_update_tree->start_propagation();
    if (!is_vector_empty && (_invalid_gp_indices_cnt > 0)) {
        is_vector_empty =
                _range_update_tree->start_propagation_fwd_cascade(_invalid_gp_indices.get(), _invalid_gp_indices_cnt);
    }
    if (!is_vector_empty && (new_start <= new_end)) { _next_op->execute(); }

    restore_slices();

    COPY_BITMASK(State::MAX_VECTOR_SIZE, in_state->selector, _in_selector_backup);
    in_state->start_pos = in_vec_start_idx;
    in_state->end_pos = in_vec_end_idx;
    _invalid_gp_indices_cnt = 0;
}

template<typename T>
static void register_node_in_saved_data_gp_shared(FtreeStateUpdateNode* node, VectorGPSavedData<T>* vector_saved_data,
                                                  uint32_t& saved_data_index) {
    // Get current start and end positions from the vector's selector
    const int32_t current_start = GET_START_POS(*node->vector->state);
    const int32_t current_end = GET_END_POS(*node->vector->state);

    // Create VectorSavedData entry
    vector_saved_data[saved_data_index++] =
            VectorGPSavedData<T>(node->attribute, const_cast<Vector<T>*>(node->vector), current_start, current_end);
}

template<typename T>
static void register_backward_nodes_in_saved_data_gp_shared(FtreeStateUpdateNode* parent_node,
                                                            VectorGPSavedData<T>* vector_saved_data,
                                                            uint32_t& saved_data_index) {
    for (const auto& child: parent_node->children) {
        // Register this backward node
        register_node_in_saved_data_gp_shared<T>(child.get(), vector_saved_data, saved_data_index);

        // Recursively register its children
        register_backward_nodes_in_saved_data_gp_shared<T>(child.get(), vector_saved_data, saved_data_index);
    }
}

// Explicit template instantiations
template class INLJoinPackedGPCascadeShared<uint64_t>;
// template class INLJoinPackedGPCascadeShared<ffx_str_t>; // TODO: Enable when hash-based adjacency list indexing is implemented

}// namespace ffx
