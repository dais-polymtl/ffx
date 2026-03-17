#include "join/flat_join.hpp"

#include "../../table/include/ffx_str_t.hpp"
#include "ancestor_finder_utils.hpp"
#include <cassert>
#include <cstring>
#include <iostream>
#include <type_traits>
#include <vector/bitmask.hpp>

namespace ffx {

template<typename T>
static void register_node_in_saved_data(FtreeStateUpdateNode* node,
                                        FlatJoinVectorSliceUpdateSavedData<T>* vector_saved_data,
                                        uint32_t& saved_data_index, uint32_t max_size);
template<typename T>
static void register_backward_nodes_in_saved_data(FtreeStateUpdateNode* parent_node,
                                                  FlatJoinVectorSliceUpdateSavedData<T>* vector_saved_data,
                                                  uint32_t& saved_data_index, uint32_t max_size);

template<typename T>
void FlatJoin<T>::create_slice_update_infrastructure(FactorizedTreeElement* ftree_leaf) {
    _vector_saved_data = std::make_unique<FlatJoinVectorSliceUpdateSavedData<T>[]>(_vector_saved_data_count);
    uint32_t saved_data_index = 0;
    const auto parent_fnode = ftree_leaf->_parent;
    _range_update_tree->fill_bwd_join_key(/* parent node */ parent_fnode, /* child to exclude */ _output_attr);
    register_backward_nodes_in_saved_data<T>(_range_update_tree.get(), _vector_saved_data.get(), saved_data_index,
                                             _vector_saved_data_count);

    FactorizedTreeElement* ftreenode = parent_fnode->_parent;
    FtreeStateUpdateNode* current_node = _range_update_tree.get();
    while (ftreenode != nullptr) {
        auto child = std::make_unique<FtreeStateUpdateNode>(ftreenode->_value, FORWARD, ftreenode->_attribute);
        auto child_ptr = child.get();
        register_node_in_saved_data<T>(child_ptr, _vector_saved_data.get(), saved_data_index, _vector_saved_data_count);
        child_ptr->parent = current_node;
        current_node->children.push_back(std::move(child));
        // If the current node has children, we need to fill it with backward updates
        child_ptr->fill_bwd(ftreenode, _output_attr);
        register_backward_nodes_in_saved_data(child_ptr, _vector_saved_data.get(), saved_data_index,
                                              _vector_saved_data_count);
        current_node = current_node->children.back().get();
        ftreenode = ftreenode->_parent;
    }
}

template<typename T>
void FlatJoin<T>::store_slices() {
    for (auto i = 0; i < _vector_saved_data_count; i++) {
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
void FlatJoin<T>::restore_slices() {
    // Restore all vectors from unified backup store
    for (auto i = 0; i < _vector_saved_data_count; i++) {
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
void FlatJoin<T>::init(Schema* schema) {
    static_assert(
            std::is_same_v<T, uint64_t>,
            "FlatJoin currently only supports uint64_t. String support requires hash-based adjacency list indexing.");
    auto& map = *schema->map;
    auto root = schema->root;

    _parent_vec = map.get_vector(_parent_attr);
    _lca_vec = map.get_vector(_lca_attr);

    // Ensure output vector is allocated as a child of the parent_attr in the DataChunk tree
    map.set_current_parent_chunk(_parent_attr);
    _out_vec = map.allocate_vector(_output_attr);

    _parent_offset = _parent_vec->state->offset;

    ResolvedJoinAdjList resolved;
    if (!schema->try_resolve_join_adj_list(_lca_attr, _output_attr, resolved)) {
        throw std::runtime_error("FlatJoin: No table found for LCA " + _lca_attr + " and output " + _output_attr);
    }
    _adj_lists = reinterpret_cast<AdjList<T>*>(resolved.adj_list);
    std::cout << "FlatJoin " << _lca_attr << "->" << _output_attr
              << (resolved.from_schema_map ? " using Schema adj_list" : " using fallback table adj_list")
              << " (" << (_is_join_index_fwd ? "fwd" : "bwd") << ")" << std::endl;

    const auto ftree_leaf = root->add_leaf(_parent_attr, _output_attr, _parent_vec, _out_vec);
    _range_update_tree = std::make_unique<FtreeStateUpdateNode>(_parent_vec, NONE, _parent_attr);

    const auto unique_datachunks = root->count_unique_datachunks();
    _vector_saved_data_count = root->get_num_nodes() - 2;// Exclude parent and output_attr
    if (_vector_saved_data_count > 0 && unique_datachunks > 2) { create_slice_update_infrastructure(ftree_leaf); }
    else { _vector_saved_data_count = 0; }

    // Create ancestor finder to map parent (descendant) indices to LCA (ancestor) indices
    // Build state path using the utility function
    auto path_info = internal::build_ancestor_finder_path(map, _lca_attr, _parent_attr);
    _same_data_chunk = path_info.same_data_chunk;

    // Create FtreeAncestorFinder only if not in the same DataChunk
    if (!_same_data_chunk) {
        _ancestor_finder = std::make_unique<FtreeAncestorFinder>(path_info.state_path.data(),
                                      path_info.state_path.size());
    }

    _active_mask_uptr = std::make_unique<BitMask<State::MAX_VECTOR_SIZE>>();
    _range_update_tree->precompute_effective_children();

    _next_op->init(schema);
}

template<typename T>
void FlatJoin<T>::execute() {
    num_exec_call++;

    // Get input state and values
    State* parent_state = _parent_vec->state;
    // const T* parent_vals = _parent_vec->values;
    //
    // State* lca_state = _lca_vec->state;
    const T* lca_vals = _lca_vec->values;

    State* out_state = _out_vec->state;
    T* out_vals = _out_vec->values;
    uint16_t* out_offset = out_state->offset;

    // Save the original selector pointer
    COPY_BITMASK(State::MAX_VECTOR_SIZE, _parent_selector_backup, parent_state->selector);

    // Copy parent selector to active mask
    COPY_BITMASK(State::MAX_VECTOR_SIZE, *_active_mask_uptr, _parent_selector_backup);

    // Update parent selector pointer to point to active mask
    COPY_BITMASK(State::MAX_VECTOR_SIZE, _parent_vec->state->selector, *_active_mask_uptr);

    // Get active range
    const int32_t parent_start = GET_START_POS(*parent_state);
    const int32_t parent_end = GET_END_POS(*parent_state);

    // Initialize output write index
    int32_t out_write_idx = 0;


    // Pre-calculate b_idx to a_idx mapping
    uint32_t b_idx_to_a_idx[State::MAX_VECTOR_SIZE];

    const int32_t lca_start = GET_START_POS(*_lca_vec->state);
    const int32_t lca_end = GET_END_POS(*_lca_vec->state);

    if (_same_data_chunk) {
        // Same DataChunk: identity mapping (each position maps to itself)
        for (int32_t idx = parent_start; idx <= parent_end; idx++) {
            b_idx_to_a_idx[idx] = static_cast<uint32_t>(idx);
        }
    } else {
        // Different DataChunks: use FtreeAncestorFinder
        _ancestor_finder->process(b_idx_to_a_idx, lca_start, lca_end, parent_start, parent_end);
    }

    // If the parent idx is completely exhausted,
    // the new start pos is the next valid index processed,
    // instead of the previous end idx. This flag tells us
    // whether we need to set the start pos for the current
    // input vector idx
    bool should_set_start_pos = true;

    // For each parent (b) position
    for (int32_t b_idx = static_cast<int32_t>(next_set_bit_in_range(*_active_mask_uptr, parent_start, parent_end));
         b_idx <= parent_end;
         b_idx = static_cast<int32_t>(
                 next_set_bit_in_range(*_active_mask_uptr, static_cast<uint32_t>(b_idx + 1), parent_end))) {

        // Step 1: Get the lca (a) value for this b using pre-calculated mapping
        const uint32_t a_idx = b_idx_to_a_idx[b_idx];
        assert(a_idx != UINT32_MAX && "b_idx must belong to some a_idx");

        // Get the a value for this a_idx
        const T a_val = lca_vals[a_idx];

        // Step 2: Get adjacency list for a->c
        const auto& adj_list = _adj_lists[a_val];
        const auto num_c_values = static_cast<int32_t>(adj_list.size);

        if (num_c_values == 0) {
            CLEAR_BIT(*_active_mask_uptr, b_idx);
            continue;
        }

        // Step 3: Write c values to output vector
        auto out_values_read_idx = 0;
        const int32_t remaining_space = State::MAX_VECTOR_SIZE - out_write_idx;
        const int32_t elements_to_copy_in_first_stage = std::min(remaining_space, num_c_values);

        // Set end pos for current parent position
        SET_END_POS(*parent_state, b_idx);
        if (should_set_start_pos) {
            SET_START_POS(*parent_state, b_idx);
            should_set_start_pos = false;
        }

        // First stage: copy the initial elements into the output vector and update RLE
        if constexpr (std::is_same_v<T, uint64_t>) {
            std::memcpy(&out_vals[out_write_idx], &adj_list.values[out_values_read_idx],
                        elements_to_copy_in_first_stage * sizeof(T));
        } else {
            for (int32_t i = 0; i < elements_to_copy_in_first_stage; ++i) {
                out_vals[out_write_idx + i] = adj_list.values[out_values_read_idx + i];
            }
        }
        out_offset[b_idx] = out_write_idx;
        out_offset[b_idx + 1] = out_write_idx + elements_to_copy_in_first_stage;
        out_write_idx += elements_to_copy_in_first_stage;
        out_values_read_idx += elements_to_copy_in_first_stage;

        // Step 4: Handle chunking if output vector is full
        if (out_write_idx == State::MAX_VECTOR_SIZE) {
            process_data_chunk(&_current_ip_mask, out_write_idx - 1);
            out_write_idx = 0;

            // Calculate remaining elements after the first stage
            const int32_t remaining_adj_list_elements = num_c_values - elements_to_copy_in_first_stage;
            const int32_t num_second_stage_count = remaining_adj_list_elements / State::MAX_VECTOR_SIZE;
            const int32_t elements_to_copy_in_last_stage = remaining_adj_list_elements & (State::MAX_VECTOR_SIZE - 1);

            // If no remaining elements, set a new start pos
            should_set_start_pos = (remaining_adj_list_elements == 0);

            // Second stage: process middle elements that completely fill the output vector
            if (num_second_stage_count > 0) {
                // These values don't change, so set them once
                SET_START_POS(*parent_state, b_idx);
                SET_END_POS(*parent_state, b_idx);
                for (auto i = 0; i < num_second_stage_count; ++i) {
                    if constexpr (std::is_same_v<T, uint64_t>) {
                        std::memcpy(&out_vals[0], &adj_list.values[out_values_read_idx],
                                    State::MAX_VECTOR_SIZE * sizeof(T));
                    } else {
                        for (int32_t j = 0; j < State::MAX_VECTOR_SIZE; ++j) {
                            out_vals[j] = adj_list.values[out_values_read_idx + j];
                        }
                    }
                    out_offset[b_idx] = 0;
                    out_offset[b_idx + 1] = State::MAX_VECTOR_SIZE;
                    process_data_chunk(&_current_ip_mask, State::MAX_VECTOR_SIZE - 1);
                    out_values_read_idx += State::MAX_VECTOR_SIZE;
                }
                // If no remaining elements, set a new start pos
                should_set_start_pos = (elements_to_copy_in_last_stage == 0);
            }

            // Third stage: process remaining elements that don't fill the output vector
            if (elements_to_copy_in_last_stage > 0) {
                assert(out_write_idx == 0);
                if constexpr (std::is_same_v<T, uint64_t>) {
                    std::memcpy(&out_vals[0], &adj_list.values[out_values_read_idx],
                                elements_to_copy_in_last_stage * sizeof(T));
                } else {
                    for (int32_t i = 0; i < elements_to_copy_in_last_stage; ++i) {
                        out_vals[i] = adj_list.values[out_values_read_idx + i];
                    }
                }
                out_offset[b_idx] = 0;
                out_offset[b_idx + 1] = elements_to_copy_in_last_stage;
                SET_START_POS(*parent_state, b_idx);
                out_write_idx += elements_to_copy_in_last_stage;
            }
        }
    }

    // Process final chunk if any
    if (out_write_idx > 0) { process_data_chunk(&_current_ip_mask, out_write_idx - 1); }

    // Restore the original selector pointer
    COPY_BITMASK(State::MAX_VECTOR_SIZE, _parent_vec->state->selector, _parent_selector_backup);
}

template<typename T>
__attribute__((always_inline)) inline void
FlatJoin<T>::process_data_chunk(BitMask<State::MAX_VECTOR_SIZE>* _current_ip_mask, const int32_t op_filled_idx) {

    SET_ALL_BITS(_out_vec->state->selector);
    SET_START_POS(*_out_vec->state, 0);
    SET_END_POS(*_out_vec->state, op_filled_idx);

    // Record the current input mask state
    COPY_BITMASK(State::MAX_VECTOR_SIZE, *_current_ip_mask, *_active_mask_uptr);
    COPY_BITMASK(State::MAX_VECTOR_SIZE, _parent_vec->state->selector, *_active_mask_uptr);
    const auto backup_start_pos = GET_START_POS(*_parent_vec->state);
    const auto backup_end_pos = GET_END_POS(*_parent_vec->state);

    assert(TEST_BIT(*_active_mask_uptr, backup_start_pos));
    assert(TEST_BIT(*_active_mask_uptr, backup_end_pos));
    assert(backup_start_pos <= backup_end_pos);
    assert(backup_end_pos < State::MAX_VECTOR_SIZE);

    // Save slices before propagation
    store_slices();

    // Update ancestor slices before executing the next operator
    auto is_vector_empty = _range_update_tree->start_propagation();

    // Execute next operator
    if (!is_vector_empty) { _next_op->execute(); }

    // Restore ancestor slices after executing next operator
    restore_slices();
    COPY_BITMASK(State::MAX_VECTOR_SIZE, *_active_mask_uptr, *_current_ip_mask);
    SET_START_POS(*_parent_vec->state, backup_start_pos);
    SET_END_POS(*_parent_vec->state, backup_end_pos);
}

template<typename T>
static void register_node_in_saved_data(FtreeStateUpdateNode* node,
                                        FlatJoinVectorSliceUpdateSavedData<T>* vector_saved_data,
                                        uint32_t& saved_data_index, uint32_t max_size) {
    // Get current start and end positions from the vector's selector
    int32_t current_start = GET_START_POS(*node->vector->state);
    int32_t current_end = GET_END_POS(*node->vector->state);

    // Create VectorSavedData entry
    vector_saved_data[saved_data_index++] = FlatJoinVectorSliceUpdateSavedData<T>(
            node->attribute, const_cast<Vector<T>*>(node->vector), current_start, current_end);
}

template<typename T>
static void register_backward_nodes_in_saved_data(FtreeStateUpdateNode* parent_node,
                                                  FlatJoinVectorSliceUpdateSavedData<T>* vector_saved_data,
                                                  uint32_t& saved_data_index, uint32_t max_size) {
    for (const auto& child: parent_node->children) {
        // Register this backward node
        register_node_in_saved_data<T>(child.get(), vector_saved_data, saved_data_index, max_size);

        // Recursively register its children
        register_backward_nodes_in_saved_data<T>(child.get(), vector_saved_data, saved_data_index, max_size);
    }
}

// Explicit template instantiations
template class FlatJoin<uint64_t>;
// template class FlatJoin<ffx_str_t>; // TODO: Enable when hash-based adjacency list indexing is implemented

}// namespace ffx
