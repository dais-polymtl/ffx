#include "join/inljoin_packed.hpp"

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

// Helper function to copy values from adjacency list to output vector
template<typename T>
inline void copy_values(T* dest, const T* src, size_t count) {
    if constexpr (std::is_same_v<T, uint64_t>) {
        // For uint64_t, use memcpy for efficiency
        std::memcpy(dest, src, count * sizeof(T));
    } else {
        // For ffx_str_t or other types, copy element-wise
        for (size_t i = 0; i < count; ++i) {
            dest[i] = src[i];
        }
    }
}

template<typename T>
static void register_node_in_saved_data(FtreeStateUpdateNode* node, VectorSliceUpdateSavedData<T>* vector_saved_data,
                                        uint32_t& saved_data_index);
template<typename T>
static void register_backward_nodes_in_saved_data(FtreeStateUpdateNode* parent_node,
                                                  VectorSliceUpdateSavedData<T>* vector_saved_data,
                                                  uint32_t& saved_data_index);

template<typename T>
void INLJoinPacked<T>::create_slice_update_infrastructure(FactorizedTreeElement* ftree_leaf) {
    _vector_saved_data = std::make_unique<VectorSliceUpdateSavedData<T>[]>(_vector_saved_data_count);
    uint32_t saved_data_index = 0;
    const auto join_key_fnode = ftree_leaf->_parent;
    _range_update_tree->fill_bwd_join_key(/* parent node */ join_key_fnode, /* child to exclude */ _output_key);
    register_backward_nodes_in_saved_data<T>(_range_update_tree.get(), _vector_saved_data.get(), saved_data_index);

    FactorizedTreeElement* ftreenode = join_key_fnode->_parent;
    FtreeStateUpdateNode* current_node = _range_update_tree.get();
    while (ftreenode != nullptr) {
        auto child = std::make_unique<FtreeStateUpdateNode>(ftreenode->_value, FORWARD, ftreenode->_attribute);
        auto child_ptr = child.get();
        register_node_in_saved_data<T>(child_ptr, _vector_saved_data.get(), saved_data_index);
        child_ptr->parent = current_node;
        current_node->children.push_back(std::move(child));
        // If the current node has children, we need to fill it with backward updates
        child_ptr->fill_bwd(ftreenode, _output_key);
        register_backward_nodes_in_saved_data<T>(child_ptr, _vector_saved_data.get(), saved_data_index);
        current_node = current_node->children.back().get();
        ftreenode = ftreenode->_parent;
    }
}

// one less bitmask refilling inside your scan operator / can be due to the restore inside the first join.
// join works this way, 1s in the bitmask are set inside the init() and the first operator consuming or processing over
// this vector is responsible in restoring back the 1s and so no need to reset.

template<typename T>
void INLJoinPacked<T>::init(Schema* schema) {
    static_assert(std::is_same_v<T, uint64_t>, "INLJoinPacked currently only supports uint64_t. String support "
                                               "requires hash-based adjacency list indexing.");
    auto& map = *schema->map;
    auto root = schema->root;

    // Input vector must already exist (created by upstream operator).
    // After CartesianProduct rewrites, schema->map will return the new vectors
    // via the override mechanism; operators should keep using get_vector().
    _in_vec = map.get_vector(_join_key);

    ResolvedJoinAdjList resolved;
    if (!schema->try_resolve_join_adj_list(_join_key, _output_key, resolved)) {
        throw std::runtime_error("No table found for join " + _join_key + " -> " + _output_key);
    }

    _adj_lists = reinterpret_cast<AdjList<T>*>(resolved.adj_list);
    _is_join_index_fwd = resolved.is_fwd;

    map.set_current_parent_chunk(_join_key);
    _out_vec = map.allocate_vector(_output_key);
    if (resolved.source_table) {
        std::cout << "Join operator " << _join_key << "->" << _output_key
                  << (resolved.from_schema_map ? " using Schema adj_list from table " : " selected fallback table ")
                  << resolved.source_table->name << " (" << (resolved.is_fwd ? "fwd" : "bwd") << ")"
                  << std::endl;
    } else {
        std::cout << "Join operator " << _join_key << "->" << _output_key << " using Schema adj_list"
                  << std::endl;
    }

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
void INLJoinPacked<T>::store_slices() {
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
void INLJoinPacked<T>::print_upstream_vectors() {
    for (std::size_t i = 0; i < _vector_saved_data_count; i++) {
        // auto& vec_data = _vector_saved_data[i];
        // const Vector* vec = vec_data.vector;
        // auto& state = *vec->state;
        // auto start_pos = GET_START_POS(state);
        // auto end_pos = GET_END_POS(state);
    }
}

template<typename T>
void INLJoinPacked<T>::restore_slices() {
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
void INLJoinPacked<T>::execute() {
    num_exec_call++;

    // Store ancestor slices before executing next operator
    store_slices();

    State* RESTRICT in_state = _in_vec->state;
    State* RESTRICT out_state = _out_vec->state;
    const T* RESTRICT in_vals = _in_vec->values;
    T* RESTRICT out_vals = _out_vec->values;

    // Save the original input selection mask (deep copy since selector is inline)
    COPY_BITMASK(State::MAX_VECTOR_SIZE, _in_selector_backup, in_state->selector);

    // Copy the original mask into our active mask (single deep copy)
    COPY_BITMASK(State::MAX_VECTOR_SIZE, *_active_mask_uptr, _in_selector_backup);

    // Use our pre-allocated active mask as the working selection mask
    COPY_BITMASK(State::MAX_VECTOR_SIZE, in_state->selector, *_active_mask_uptr);

    // Regular packed path: uses output offset + dense output chunks
    uint16_t* RESTRICT out_offset = out_state->offset;


    // Get the active bit range from the bitmask
    const auto in_vec_start_idx = GET_START_POS(*in_state);
    const auto in_vec_end_idx = GET_END_POS(*in_state);

    int32_t out_vector_write_idx = 0;

    // If the parent idx is completely exhausted,
    // the new start pos is the next valid index processed,
    // instead of the previous end idx. This flag tells us
    // whether we need to set the start pos for the current
    // input vector idx
    bool should_set_start_pos = true;
    for (auto in_vector_idx = static_cast<int32_t>(next_set_bit_in_range(*_active_mask_uptr, in_vec_start_idx,
                                                                          in_vec_end_idx));
         in_vector_idx <= in_vec_end_idx;
         in_vector_idx = static_cast<int32_t>(
                 next_set_bit_in_range(*_active_mask_uptr, static_cast<uint32_t>(in_vector_idx + 1),
                                       in_vec_end_idx))) {
            // Get the adjacency list for current value
            const auto& adj_list = _adj_lists[in_vals[in_vector_idx]];
            const auto output_elems_produced = static_cast<int32_t>(adj_list.size);
            if (output_elems_produced == 0) {
                CLEAR_BIT(*_active_mask_uptr, in_vector_idx);
            } else {
                // Might not be worth it long term, but I was thinking about a notion of full vectors packed first
                // followed by a set of partial vectors but that would usually just be one.

                // We divide the processing of adj list elements into 3 stages
                // First, the initial chunk required to fill the output vector
                // Second, the middle elements which completely fill the output
                // vector 0 or more times, and finally the last chunk which
                // contains the remaining elements that don't fill the output vector,
                // The first stage always exists, the other two may or may not exist.

                // To get the first stage, we need to find min of remaining space
                // in output vector and elements in the adjacency list
                auto out_values_read_idx = 0;// idx from where we will copy values from the adjacency list
                SET_END_POS(*in_state, in_vector_idx);
                if (should_set_start_pos) {
                    SET_START_POS(*in_state, in_vector_idx);
                    should_set_start_pos = false;
                }

                const auto remaining_space_in_output_vector = State::MAX_VECTOR_SIZE - out_vector_write_idx;
                const auto elements_to_copy_in_first_stage =
                        std::min(remaining_space_in_output_vector, output_elems_produced);

                // First stage: copy the initial elements into the output vector and update RLE
                copy_values<T>(&out_vals[out_vector_write_idx], &adj_list.values[out_values_read_idx],
                               elements_to_copy_in_first_stage);
                out_offset[in_vector_idx] = out_vector_write_idx;
                out_offset[in_vector_idx + 1] = out_vector_write_idx + elements_to_copy_in_first_stage;
                out_vector_write_idx += elements_to_copy_in_first_stage;
                out_values_read_idx += elements_to_copy_in_first_stage;


                if (out_vector_write_idx == State::MAX_VECTOR_SIZE) {
                    process_data_chunk(&_current_ip_mask, out_vector_write_idx - 1);
                    out_vector_write_idx = 0;

                    // Now we can calculate the remaining elements after the first stage
                    // This will help us determine if we have a second and third stage
                    const auto remaining_adj_list_elements = output_elems_produced - elements_to_copy_in_first_stage;
                    const auto num_second_stage_count = remaining_adj_list_elements / State::MAX_VECTOR_SIZE;
                    const auto elements_to_copy_in_last_stage =
                            remaining_adj_list_elements & (State::MAX_VECTOR_SIZE - 1);

                    // If no remaining elements, set a new start pos
                    should_set_start_pos = (remaining_adj_list_elements == 0);


                    // We have a second stage, so we can copy the middle elements
                    // These elements will completely fill the output vector 0 or more times, so we can skip
                    // updating the op_filled_idx. new_start_pos will not change since we are still working
                    // on the same end_idx
                    if (num_second_stage_count > 0) {
                        // These values don't change, so set them once
                        SET_START_POS(*in_state, in_vector_idx);
                        SET_END_POS(*in_state, in_vector_idx);
                        for (auto i = 0; i < num_second_stage_count; ++i) {
                            copy_values<T>(&out_vals[0], &adj_list.values[out_values_read_idx], State::MAX_VECTOR_SIZE);
                            out_offset[in_vector_idx] = 0;
                            out_offset[in_vector_idx + 1] = State::MAX_VECTOR_SIZE;
                            process_data_chunk(&_current_ip_mask, State::MAX_VECTOR_SIZE - 1);
                            out_values_read_idx += State::MAX_VECTOR_SIZE;
                        }
                        // If no remaining elements, set a new start pos
                        should_set_start_pos = (elements_to_copy_in_last_stage == 0);
                    }

                    // Finally, we have the last stage which may contain the remaining elements
                    if (elements_to_copy_in_last_stage > 0) {
                        assert(out_vector_write_idx == 0);
                        copy_values<T>(&out_vals[0], &adj_list.values[out_values_read_idx],
                                       elements_to_copy_in_last_stage);
                        out_offset[in_vector_idx] = 0;
                        out_offset[in_vector_idx + 1] = elements_to_copy_in_last_stage;
                        out_vector_write_idx += elements_to_copy_in_last_stage;
                        SET_START_POS(*in_state, in_vector_idx);
                    }
                }
            }
    }
    if (out_vector_write_idx > 0) { process_data_chunk(&_current_ip_mask, out_vector_write_idx - 1); }

    // Restore the original input selection mask (deep copy back)
    COPY_BITMASK(State::MAX_VECTOR_SIZE, in_state->selector, _in_selector_backup);
    in_state->start_pos = in_vec_start_idx;
    in_state->end_pos = in_vec_end_idx;
}

template<typename T>
void INLJoinPacked<T>::process_data_chunk(BitMask<State::MAX_VECTOR_SIZE>* _current_ip_mask,
                                          const int32_t op_filled_idx) {

    SET_ALL_BITS(_out_vec->state->selector);
    SET_START_POS(*_out_vec->state, 0);
    SET_END_POS(*_out_vec->state, op_filled_idx);

    // Record the current input mask state
    COPY_BITMASK(State::MAX_VECTOR_SIZE, *_current_ip_mask, *_active_mask_uptr);
    // Also sync active_mask into in_state->selector
    COPY_BITMASK(State::MAX_VECTOR_SIZE, _in_vec->state->selector, *_active_mask_uptr);
    const auto backup_start_pos = GET_START_POS(*_in_vec->state);
    const auto backup_end_pos = GET_END_POS(*_in_vec->state);

    assert(TEST_BIT(*_active_mask_uptr, backup_start_pos));
    assert(TEST_BIT(*_active_mask_uptr, backup_end_pos));
    assert(backup_start_pos <= backup_end_pos);
    assert(backup_end_pos < State::MAX_VECTOR_SIZE);

    // Update ancestor slices before executing the next operator
    auto is_vector_empty = _range_update_tree->start_propagation();

    // Execute next operator
    if (!is_vector_empty) { _next_op->execute(); }

    // Restore ancestor slices after executing next operator
    restore_slices();
    COPY_BITMASK(State::MAX_VECTOR_SIZE, *_active_mask_uptr, *_current_ip_mask);
    SET_START_POS(*_in_vec->state, backup_start_pos);
    SET_END_POS(*_in_vec->state, backup_end_pos);
}


template<typename T>
static void register_node_in_saved_data(FtreeStateUpdateNode* node, VectorSliceUpdateSavedData<T>* vector_saved_data,
                                        uint32_t& saved_data_index) {
    //assert (saved_data_index < max_size);
    // Get current start and end positions from the vector's selector
    int32_t current_start = GET_START_POS(*node->vector->state);
    int32_t current_end = GET_END_POS(*node->vector->state);

    // Create VectorSavedData entry
    vector_saved_data[saved_data_index++] = VectorSliceUpdateSavedData<T>(
            node->attribute, const_cast<Vector<T>*>(node->vector), current_start, current_end);
}

template<typename T>
static void register_backward_nodes_in_saved_data(FtreeStateUpdateNode* parent_node,
                                                  VectorSliceUpdateSavedData<T>* vector_saved_data,
                                                  uint32_t& saved_data_index) {
    for (const auto& child: parent_node->children) {
        // Register this backward node
        register_node_in_saved_data<T>(child.get(), vector_saved_data, saved_data_index);

        // Recursively register its children
        register_backward_nodes_in_saved_data<T>(child.get(), vector_saved_data, saved_data_index);
    }
}

// Explicit template instantiations
template class INLJoinPacked<uint64_t>;
// template class INLJoinPacked<ffx_str_t>; // TODO: Enable when hash-based adjacency list indexing is implemented

}// namespace ffx