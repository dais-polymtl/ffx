#include "join/nway_intersection.hpp"

#include "../../table/include/ffx_str_t.hpp"
#include <cassert>
#include <cstring>
#include <iostream>
#include <type_traits>
#include <vector/bitmask.hpp>
#include <vector>

namespace ffx {

template<typename T>
static void register_node_in_saved_data(FtreeStateUpdateNode* node,
                                        NWayIntersectionVectorSliceUpdateSavedData<T>* vector_saved_data,
                                        uint32_t& saved_data_index, uint32_t max_size);
template<typename T>
static void register_backward_nodes_in_saved_data(FtreeStateUpdateNode* parent_node,
                                                  NWayIntersectionVectorSliceUpdateSavedData<T>* vector_saved_data,
                                                  uint32_t& saved_data_index, uint32_t max_size);

template<typename T>
NWayIntersection<T>::NWayIntersection(std::string output_attr,
                                      std::vector<std::pair<std::string, bool>> input_attrs_and_directions)
    : Operator(), _output_attr(std::move(output_attr)),
      _input_attrs_and_directions(std::move(input_attrs_and_directions)), _out_vec(nullptr),
      _range_update_tree(nullptr), _vector_saved_data(nullptr) {

    // Validate at least 2 inputs
    if (_input_attrs_and_directions.size() < 2) {
        throw std::runtime_error("NWayIntersection: requires at least 2 input attributes");
    }
}

template<typename T>
void NWayIntersection<T>::create_slice_update_infrastructure(const FactorizedTreeElement* ftree_output_node) {
    _vector_saved_data = std::make_unique<NWayIntersectionVectorSliceUpdateSavedData<T>[]>(_vector_saved_data_count);
    uint32_t saved_data_index = 0;
    const auto last_input_fnode = ftree_output_node->_parent;
    _range_update_tree->fill_bwd_join_key(/* parent node */ last_input_fnode, /* child to exclude */ _output_attr);
    register_backward_nodes_in_saved_data<T>(_range_update_tree.get(), _vector_saved_data.get(), saved_data_index,
                                             _vector_saved_data_count);

    FactorizedTreeElement* ftreenode = last_input_fnode->_parent;// ancestor node (first input)
    FtreeStateUpdateNode* current_node = _range_update_tree.get();
    while (ftreenode != nullptr) {
        auto child = std::make_unique<FtreeStateUpdateNode>(ftreenode->_value, FORWARD, ftreenode->_attribute);
        auto child_ptr = child.get();
        register_node_in_saved_data<T>(child_ptr, _vector_saved_data.get(), saved_data_index, _vector_saved_data_count);
        child_ptr->parent = current_node;
        current_node->children.push_back(std::move(child));
        // If the current node has children, we need to fill it with backward updates
        child_ptr->fill_bwd(ftreenode, _output_attr);
        register_backward_nodes_in_saved_data<T>(child_ptr, _vector_saved_data.get(), saved_data_index,
                                                 _vector_saved_data_count);
        current_node = current_node->children.back().get();
        ftreenode = ftreenode->_parent;
    }
}

template<typename T>
void NWayIntersection<T>::store_slices() {
    for (auto i = 0; i < _vector_saved_data_count; i++) {
        auto& vec_data = _vector_saved_data[i];
        Vector<T>* vec = vec_data.vector;
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
void NWayIntersection<T>::restore_slices() {
    for (auto i = 0; i < _vector_saved_data_count; i++) {
        auto& vec_data = _vector_saved_data[i];
        Vector<T>* vec = vec_data.vector;
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
void NWayIntersection<T>::init(Schema* schema) {
    static_assert(std::is_same_v<T, uint64_t>, "NWayIntersection currently only supports uint64_t. String support "
                                               "requires hash-based adjacency list indexing.");
    auto& map = *schema->map;
    const auto& tables = schema->tables;
    auto root = schema->root;

    // Get vectors for all input attributes
    _input_vecs.clear();
    for (const auto& [attr, _]: _input_attrs_and_directions) {
        _input_vecs.push_back(map.get_vector(attr));
    }

    // NOTE: Do NOT call set_current_parent_chunk here to preserve DataChunk ancestry
    // for transitive theta joins like EQ(a, b) AND EQ(b, c).
    // Allocate output vector
    _out_vec = map.allocate_vector(_output_attr);

    // Get RLE arrays for all input vectors (last input's RLE is used for output mapping)
    _offset_arrays.clear();
    for (auto* vec: _input_vecs) {
        _offset_arrays.push_back(vec->state->offset);
    }

    // Get first and last input attributes
    const std::string& first_input_attr = _input_attrs_and_directions[0].first;
    const std::string& last_input_attr = _input_attrs_and_directions.back().first;

    // Find first input node in ftree
    FactorizedTreeElement* first_input_node = root->find_node_by_attribute(first_input_attr);
    if (!first_input_node) { throw std::runtime_error("NWayIntersection: first input attribute not found in tree"); }

    // Set up adjacency lists using the direction flags from constructor (query edge directions)
    // Don't recalculate direction based on table column order - use the query edge direction
    _adj_lists.clear();
    for (size_t i = 0; i < _input_attrs_and_directions.size(); i++) {
        const auto& [input_attr, query_edge_is_fwd] = _input_attrs_and_directions[i];

        ResolvedJoinAdjList resolved;
        if (!schema->try_resolve_join_adj_list(input_attr, _output_attr, resolved)) {
            throw std::runtime_error("NWayIntersection: No table found for input " + input_attr + " and output " +
                                     _output_attr);
        }

        if (!resolved.from_schema_map && resolved.source_table) {
            auto* table_adj_lists = query_edge_is_fwd ? resolved.source_table->fwd_adj_lists
                                                      : resolved.source_table->bwd_adj_lists;
            _adj_lists.push_back(reinterpret_cast<AdjList<T>*>(table_adj_lists));
        } else {
            _adj_lists.push_back(reinterpret_cast<AdjList<T>*>(resolved.adj_list));
        }
    }

    // Add to factorized tree
    root->add_leaf(last_input_attr, _output_attr, _input_vecs.back(), _out_vec);

    // Verify all input attributes are encountered in reverse order going up from output
    FactorizedTreeElement* output_node = root->find_node_by_attribute(_output_attr);
    FactorizedTreeElement* current = output_node->_parent;                            // Start from parent of output
    int expected_input_idx = static_cast<int>(_input_attrs_and_directions.size()) - 1;// Start from last input

    while (current != nullptr && expected_input_idx >= 0) {
        const std::string& expected_attr = _input_attrs_and_directions[expected_input_idx].first;
        if (current->_attribute == expected_attr) { expected_input_idx--; }
        current = current->_parent;
    }

    if (expected_input_idx >= 0) {
        throw std::runtime_error(
                "NWayIntersection: input attributes do not form a valid path from output to first input");
    }

    // Set up range update infrastructure
    _range_update_tree = std::make_unique<FtreeStateUpdateNode>(_input_vecs.back(), NONE, last_input_attr);

    const auto unique_datachunks = root->count_unique_datachunks();
    _vector_saved_data_count = root->get_num_nodes() - 2;// Exclude last input, output_attr
    if (_vector_saved_data_count > 0 && unique_datachunks > 2) { create_slice_update_infrastructure(output_node); }
    else { _vector_saved_data_count = 0; }

    // Initialize bitmask
    _last_input_valid_mask_uptr = std::make_unique<BitMask<State::MAX_VECTOR_SIZE>>();
    _range_update_tree->precompute_effective_children();

    // Initialize next operator
    _next_op->init(schema);
}

template<typename T>
void NWayIntersection<T>::process_data_chunk(BitMask<State::MAX_VECTOR_SIZE>* _current_ip_mask, int32_t op_filled_idx) {
    SET_ALL_BITS(_out_vec->state->selector);
    SET_START_POS(*_out_vec->state, 0);
    SET_END_POS(*_out_vec->state, op_filled_idx);

    COPY_BITMASK(State::MAX_VECTOR_SIZE, *_current_ip_mask, *_last_input_valid_mask_uptr);
    const auto backup_start_pos = GET_START_POS(*_input_vecs.back()->state);
    const auto backup_end_pos = GET_END_POS(*_input_vecs.back()->state);

    COPY_BITMASK(State::MAX_VECTOR_SIZE, _input_vecs.back()->state->selector, *_last_input_valid_mask_uptr);
    store_slices();
    _range_update_tree->start_propagation();
    if (_next_op) { _next_op->execute(); }
    restore_slices();

    COPY_BITMASK(State::MAX_VECTOR_SIZE, *_last_input_valid_mask_uptr, *_current_ip_mask);
    SET_START_POS(*_input_vecs.back()->state, backup_start_pos);
    SET_END_POS(*_input_vecs.back()->state, backup_end_pos);
}

// Helper functions for ftree state update
template<typename T>
static void register_node_in_saved_data(FtreeStateUpdateNode* node,
                                        NWayIntersectionVectorSliceUpdateSavedData<T>* vector_saved_data,
                                        uint32_t& saved_data_index, uint32_t max_size) {
    // Get current start and end positions from the vector's selector
    int32_t current_start = GET_START_POS(*node->vector->state);
    int32_t current_end = GET_END_POS(*node->vector->state);

    // Create VectorSavedData entry
    vector_saved_data[saved_data_index++] = NWayIntersectionVectorSliceUpdateSavedData<T>(
            node->attribute, const_cast<Vector<T>*>(node->vector), current_start, current_end);
}

template<typename T>
static void register_backward_nodes_in_saved_data(FtreeStateUpdateNode* parent_node,
                                                  NWayIntersectionVectorSliceUpdateSavedData<T>* vector_saved_data,
                                                  uint32_t& saved_data_index, uint32_t max_size) {
    for (auto& child: parent_node->children) {
        register_node_in_saved_data<T>(child.get(), vector_saved_data, saved_data_index, max_size);
        register_backward_nodes_in_saved_data<T>(child.get(), vector_saved_data, saved_data_index, max_size);
    }
}

// Helper function: compute intersection of two sorted arrays
template<typename T>
int32_t NWayIntersection<T>::compute_sorted_intersection(const T* arr1, int32_t size1, const T* arr2, int32_t size2,
                                                         T* dest) {

    int32_t i = 0, j = 0, count = 0;

    while (i < size1 && j < size2) {
        if (arr1[i] == arr2[j]) {
            dest[count++] = arr1[i];
            i++;
            j++;
        } else if (arr1[i] < arr2[j]) {
            i++;
        } else {
            j++;
        }
    }

    return count;
}

template<typename T>
void NWayIntersection<T>::execute() {
    num_exec_call++;

    // Get output state
    State* out_state = _out_vec->state;
    T* out_vals = _out_vec->values;
    uint16_t* out_offset = out_state->offset;

    // Get the last input attribute (the one we'll modify the bitmask for)
    const size_t last_input_idx = _input_vecs.size() - 1;
    State* last_input_state = _input_vecs[last_input_idx]->state;
    const T* last_input_vals = _input_vecs[last_input_idx]->values;

    // Save the original selector pointer for last input
    COPY_BITMASK(State::MAX_VECTOR_SIZE, _last_input_selector_backup, last_input_state->selector);

    // Copy last input selector to valid mask
    COPY_BITMASK(State::MAX_VECTOR_SIZE, *_last_input_valid_mask_uptr, _last_input_selector_backup);

    // Update last input selector to point to valid mask
    COPY_BITMASK(State::MAX_VECTOR_SIZE, last_input_state->selector, *_last_input_valid_mask_uptr);

    // Get active range for first input attribute
    State* first_input_state = _input_vecs[0]->state;
    const T* first_input_vals = _input_vecs[0]->values;
    const int32_t first_start = GET_START_POS(*first_input_state);
    const int32_t first_end = GET_END_POS(*first_input_state);

    // Initialize output write index
    int32_t out_write_idx = 0;

    bool should_set_last_start_pos = true;

    // Allocate stack arrays for tracking iteration state
    const size_t num_levels = _input_vecs.size();

    // Allocate buffers for levels 1 onwards (level 0 doesn't need a buffer)
    // temp_buffers[1] = buffer for level 1
    // temp_buffers[i] = buffer for level i (for i >= 1)
    auto** temp_buffers = static_cast<T**>(alloca(sizeof(T*) * num_levels));

    // We don't allocate temp_buffers[0] since we'll use first_adj_list.values directly
    temp_buffers[0] = nullptr;

    for (size_t i = 1; i < num_levels; i++) {
        temp_buffers[i] = static_cast<T*>(alloca(sizeof(T) * State::MAX_VECTOR_SIZE));
    }

    // positions[i] tracks current position at level i
    auto* positions = static_cast<int32_t*>(alloca(sizeof(int32_t) * num_levels));

    // range_starts[i] and range_ends[i] track the range for level i
    auto* range_starts = static_cast<uint32_t*>(alloca(sizeof(uint32_t) * num_levels));
    auto* range_ends = static_cast<uint32_t*>(alloca(sizeof(uint32_t) * num_levels));

    // intersection_sizes[i] tracks the size of intersection up to level i
    auto* intersection_sizes = static_cast<int32_t*>(alloca(sizeof(int32_t) * num_levels));

    // Start iterating through first input attribute
    for (int32_t pos_0 = static_cast<int32_t>(next_set_bit_in_range(first_input_state->selector, first_start,
                                                                     first_end));
         pos_0 <= first_end;
         pos_0 = static_cast<int32_t>(
                 next_set_bit_in_range(first_input_state->selector, static_cast<uint32_t>(pos_0 + 1), first_end))) {

        positions[0] = pos_0;
        const T val_0 = first_input_vals[pos_0];

        // Get initial adjacency list (first attribute -> output)
        const auto& first_adj_list = _adj_lists[0][val_0];
        intersection_sizes[0] = static_cast<int32_t>(first_adj_list.size);

        // DON'T copy to buffer - we'll use first_adj_list.values directly

        // Reset current_level
        size_t current_level = 1;

        // Initialize range for level 1
        if (current_level < num_levels) {
            uint16_t* offset_arr = _offset_arrays[current_level];
            range_starts[current_level] = offset_arr[pos_0];
            range_ends[current_level] = offset_arr[pos_0 + 1];
            positions[current_level] = range_starts[current_level];
        }

        // Iterative traversal through all levels
        while (current_level > 0 && current_level < num_levels) {
            uint32_t range_start = range_starts[current_level];
            uint32_t range_end = range_ends[current_level];
            int32_t pos = positions[current_level];

            // Check if we've exhausted this level
            if (pos >= range_end) {
                // Backtrack to previous level
                current_level--;
                if (current_level > 0) { positions[current_level]++; }
                continue;
            }

            // Get current level info
            State* current_state = _input_vecs[current_level]->state;

            // Check selector
            bool is_valid = (current_level == last_input_idx) ? TEST_BIT(*_last_input_valid_mask_uptr, pos)
                                                              : TEST_BIT(current_state->selector, pos);

            if (!is_valid) {
                positions[current_level]++;
                continue;
            }

            const T* current_vals = _input_vecs[current_level]->values;
            const T current_val = current_vals[pos];

            // Get adjacency list for current attribute -> output
            const auto& current_adj_list = _adj_lists[current_level][current_val];
            const int32_t adj_size = static_cast<int32_t>(current_adj_list.size);

            // Get previous buffer/data
            // If current_level == 1, read from first_adj_list.values directly
            // Otherwise, read from temp_buffers[current_level - 1]
            const T* prev_buffer = (current_level == 1) ? first_adj_list.values : temp_buffers[current_level - 1];

            T* curr_buffer = temp_buffers[current_level];
            int32_t prev_size = intersection_sizes[current_level - 1];

            int32_t new_size = 0;

            // Compute intersection only if both previous buffer and current adj list are non-empty
            if (prev_size > 0 && adj_size > 0) {
                new_size = compute_sorted_intersection(prev_buffer, prev_size, current_adj_list.values, adj_size,
                                                       curr_buffer);
            }

            intersection_sizes[current_level] = new_size;

            // If this is the last input attribute, handle the result
            if (current_level == last_input_idx) {
                if (new_size == 0) {
                    // Empty intersection, mark this position as invalid
                    CLEAR_BIT(*_last_input_valid_mask_uptr, pos);
                    positions[current_level]++;
                    continue;
                }

                // Update last input start/end positions
                SET_END_POS(*current_state, pos);
                if (should_set_last_start_pos) {
                    SET_START_POS(*current_state, pos);
                    should_set_last_start_pos = false;
                }

                // Write intersection to output vector (from curr_buffer)
                int32_t intersection_read_idx = 0;

                while (intersection_read_idx < new_size) {
                    const uint32_t remaining_space = State::MAX_VECTOR_SIZE - out_write_idx;
                    const auto elements_to_copy =
                            std::min(remaining_space, static_cast<uint32_t>(new_size - intersection_read_idx));

                    // Copy intersection values from curr_buffer
                    if constexpr (std::is_same_v<T, uint64_t>) {
                        std::memcpy(&out_vals[out_write_idx], &curr_buffer[intersection_read_idx],
                                    elements_to_copy * sizeof(T));
                    } else {
                        for (uint32_t k = 0; k < elements_to_copy; ++k) {
                            out_vals[out_write_idx + k] = curr_buffer[intersection_read_idx + k];
                        }
                    }

                    // Set RLE for last input position
                    out_offset[pos] = out_write_idx;
                    out_offset[pos + 1] = out_write_idx + elements_to_copy;

                    assert(out_offset[pos] < out_offset[pos + 1]);

                    out_write_idx += elements_to_copy;
                    intersection_read_idx += elements_to_copy;

                    // Handle chunking if output vector is full
                    if (out_write_idx == State::MAX_VECTOR_SIZE) {
                        process_data_chunk(&_current_ip_mask, out_write_idx - 1);
                        out_write_idx = 0;

                        SET_START_POS(*current_state, pos);
                        SET_END_POS(*current_state, pos);
                        should_set_last_start_pos = (intersection_read_idx == new_size);
                    }
                }

                // Move to next position at this level
                positions[current_level]++;
            } else {
                // Not the last level, advance to next level
                current_level++;

                // Initialize range for next level using current position as parent
                uint16_t* next_offset = _offset_arrays[current_level];
                range_starts[current_level] = next_offset[pos];
                range_ends[current_level] = next_offset[pos + 1];
                positions[current_level] = range_starts[current_level];
            }
        }
    }

    // Process final chunk if any
    if (out_write_idx > 0) { process_data_chunk(&_current_ip_mask, out_write_idx - 1); }

    // Restore the original selector pointer for last input
    COPY_BITMASK(State::MAX_VECTOR_SIZE, last_input_state->selector, _last_input_selector_backup);
}

// Explicit template instantiations
template class NWayIntersection<uint64_t>;
// template class NWayIntersection<ffx_str_t>; // TODO: Enable when hash-based adjacency list indexing is implemented

}// namespace ffx