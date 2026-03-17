#include "join/nway_intersection_predicated.hpp"

#include "../../table/include/ffx_str_t.hpp"
#include "ancestor_finder_utils.hpp"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <type_traits>
#include <vector/bitmask.hpp>
#include <vector>

namespace ffx {

template<typename T>
static void register_node_in_saved_data_nwip(FtreeStateUpdateNode* node,
                                             NWayIntersectionPredicatedVectorSliceUpdateSavedData<T>* vector_saved_data,
                                             uint32_t& saved_data_index);
template<typename T>
static void
register_backward_nodes_in_saved_data_nwip(FtreeStateUpdateNode* parent_node,
                                           NWayIntersectionPredicatedVectorSliceUpdateSavedData<T>* vector_saved_data,
                                           uint32_t& saved_data_index);

template<typename T>
NWayIntersectionPredicated<T>::NWayIntersectionPredicated(
        std::string output_attr, std::vector<std::pair<std::string, bool>> input_attrs_and_directions,
        PredicateExpression predicate_expr)
    : Operator(), _output_attr(std::move(output_attr)),
      _input_attrs_and_directions(std::move(input_attrs_and_directions)), _out_vec(nullptr),
      _range_update_tree(nullptr), _vector_saved_data(nullptr), _predicate_expr_raw(std::move(predicate_expr)) {

    // Validate at least 2 inputs
    if (_input_attrs_and_directions.size() < 2) {
        throw std::runtime_error("NWayIntersectionPredicated: requires at least 2 input attributes");
    }
}

template<typename T>
NWayIntersectionPredicated<T>::NWayIntersectionPredicated(
        std::string output_attr, std::vector<std::pair<std::string, bool>> input_attrs_and_directions)
    : Operator(), _output_attr(std::move(output_attr)),
      _input_attrs_and_directions(std::move(input_attrs_and_directions)), _out_vec(nullptr),
      _range_update_tree(nullptr), _vector_saved_data(nullptr) {

    // Validate at least 2 inputs
    if (_input_attrs_and_directions.size() < 2) {
        throw std::runtime_error("NWayIntersectionPredicated: requires at least 2 input attributes");
    }
}

template<typename T>
void NWayIntersectionPredicated<T>::create_slice_update_infrastructure(const FactorizedTreeElement* ftree_output_node) {
    _vector_saved_data =
            std::make_unique<NWayIntersectionPredicatedVectorSliceUpdateSavedData<T>[]>(_vector_saved_data_count);
    uint32_t saved_data_index = 0;
    const auto last_input_fnode = ftree_output_node->_parent;
    _range_update_tree->fill_bwd_join_key(/* parent node */ last_input_fnode, /* child to exclude */ _output_attr);
    register_backward_nodes_in_saved_data_nwip<T>(_range_update_tree.get(), _vector_saved_data.get(), saved_data_index);

    FactorizedTreeElement* ftreenode = last_input_fnode->_parent;// ancestor node (first input)
    FtreeStateUpdateNode* current_node = _range_update_tree.get();
    while (ftreenode != nullptr) {
        auto child = std::make_unique<FtreeStateUpdateNode>(ftreenode->_value, FORWARD, ftreenode->_attribute);
        auto child_ptr = child.get();
        register_node_in_saved_data_nwip<T>(child_ptr, _vector_saved_data.get(), saved_data_index);
        child_ptr->parent = current_node;
        current_node->children.push_back(std::move(child));
        // If the current node has children, we need to fill it with backward updates
        child_ptr->fill_bwd(ftreenode, _output_attr);
        register_backward_nodes_in_saved_data_nwip<T>(child_ptr, _vector_saved_data.get(), saved_data_index);
        current_node = current_node->children.back().get();
        ftreenode = ftreenode->_parent;
    }
}

template<typename T>
void NWayIntersectionPredicated<T>::store_slices() {
    for (std::size_t i = 0; i < _vector_saved_data_count; i++) {
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
void NWayIntersectionPredicated<T>::restore_slices() {
    for (std::size_t i = 0; i < _vector_saved_data_count; i++) {
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
void NWayIntersectionPredicated<T>::init(Schema* schema) {
    static_assert(std::is_same_v<T, uint64_t>, "NWayIntersectionPredicated currently only supports uint64_t. String "
                                               "support requires hash-based adjacency list indexing.");
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
    if (!first_input_node) {
        throw std::runtime_error("NWayIntersectionPredicated: first input attribute not found in tree");
    }

    // Set up adjacency lists using the direction flags from constructor (query edge directions)
    _adj_lists.clear();
    for (size_t i = 0; i < _input_attrs_and_directions.size(); i++) {
        const auto& [input_attr, query_edge_is_fwd] = _input_attrs_and_directions[i];

        ResolvedJoinAdjList resolved;
        if (!schema->try_resolve_join_adj_list(input_attr, _output_attr, resolved)) {
            throw std::runtime_error("NWayIntersectionPredicated: No table found for input " + input_attr +
                                     " and output " + _output_attr);
        }

        if (!resolved.from_schema_map && resolved.source_table) {
            auto* table_adj_lists = query_edge_is_fwd ? resolved.source_table->fwd_adj_lists
                                                      : resolved.source_table->bwd_adj_lists;
            _adj_lists.push_back(reinterpret_cast<AdjList<T>*>(table_adj_lists));
        } else {
            _adj_lists.push_back(reinterpret_cast<AdjList<T>*>(resolved.adj_list));
        }
    }

    // Build the scalar predicate expression for this attribute
    _is_string_predicate = (schema->string_attributes && schema->string_attributes->count(_output_attr) > 0);
    if (_is_string_predicate) {
        _predicate_expr_string = build_scalar_predicate_expr<ffx_str_t>(_predicate_expr_raw, _output_attr,
                                                                        schema->predicate_pool, schema->dictionary);
    } else {
        _predicate_expr_numeric = build_scalar_predicate_expr<T>(_predicate_expr_raw, _output_attr,
                                                                 schema->predicate_pool, schema->dictionary);
    }

    // Print operator info with predicates
    std::cout << "NWayIntersectionPredicated(";
    for (size_t i = 0; i < _input_attrs_and_directions.size(); ++i) {
        if (i > 0) std::cout << ",";
        std::cout << _input_attrs_and_directions[i].first;
    }
    std::cout << "->" << _output_attr << ")";
    if ((_is_string_predicate && _predicate_expr_string.has_predicates()) ||
        (!_is_string_predicate && _predicate_expr_numeric.has_predicates())) {
        std::cout << " [predicates: "
                  << (_is_string_predicate ? _predicate_expr_string.to_string() : _predicate_expr_numeric.to_string())
                  << "]";
    }
    std::cout << std::endl;

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
                "NWayIntersectionPredicated: input attributes do not form a valid path from output to first input");
    }

    // Set up range update infrastructure
    _range_update_tree = std::make_unique<FtreeStateUpdateNode>(_input_vecs.back(), NONE, last_input_attr);

    const auto unique_datachunks = root->count_unique_datachunks();
    _vector_saved_data_count = root->get_num_nodes() - 2;// Exclude last input, output_attr
    if (_vector_saved_data_count > 0 && unique_datachunks > 2) { create_slice_update_infrastructure(output_node); }
    else { _vector_saved_data_count = 0; }

    // Initialize bitmask
    _last_input_valid_mask_uptr = std::make_unique<BitMask<State::MAX_VECTOR_SIZE>>();
    _invalidated_indices = std::make_unique<uint32_t[]>(State::MAX_VECTOR_SIZE);
    _invalidated_count = 0;

    // Build the FULL path from shallowest input to deepest input using utility
    // Collect all input attribute names for building the path
    std::vector<std::string> path_attrs;
    for (const auto& [attr, _]: _input_attrs_and_directions) {
        path_attrs.push_back(attr);
    }

    auto path_info = internal::build_multi_ancestor_finder_path(map, path_attrs);
    _all_same_data_chunk = path_info.all_same_data_chunk;

    // Create multi-ancestor finder only if not all in the same DataChunk
    // Important: save state_path before moving it, as we need it for input_level_indices mapping
    std::vector<const State*> saved_state_path;
    if (!_all_same_data_chunk) {
        saved_state_path = path_info.state_path;// Copy before moving
        _multi_ancestor_finder = std::make_unique<FtreeMultiAncestorFinder>(std::move(path_info.state_path));

        // Allocate ancestor index buffers (one per level in the state path, excluding the last)
        _ancestor_index_buffers.clear();
        _ancestor_index_buffer_ptrs.clear();
        for (size_t i = 0; i < _multi_ancestor_finder->num_ancestor_levels(); i++) {
            _ancestor_index_buffers.push_back(std::make_unique<uint32_t[]>(State::MAX_VECTOR_SIZE));
            _ancestor_index_buffer_ptrs.push_back(_ancestor_index_buffers.back().get());
        }
    }

    // Build mapping from input index to path level index (for state path)
    _input_level_indices.clear();
    if (!_all_same_data_chunk) {
        // Build mapping based on DataChunk boundaries
        // Each input attribute maps to a level in state_path
        // Multiple inputs may map to the same level if they share a state
        for (size_t i = 0; i < _input_attrs_and_directions.size(); i++) {
            const std::string& input_attr = _input_attrs_and_directions[i].first;
            internal::DataChunk* chunk = map.get_chunk_for_attr(input_attr);
            const State* input_state = chunk->get_state();

            // Find which level in saved_state_path this corresponds to
            bool found = false;
            for (size_t j = 0; j < saved_state_path.size(); j++) {
                if (saved_state_path[j] == input_state) {
                    _input_level_indices.push_back(j);
                    found = true;
                    break;
                }
            }

            // If not found in state_path, this input shares state with a previous input
            // In this case, use the same level index as the previous input with the same state
            if (!found) {
                // Find a previous input with the same state
                for (size_t prev_i = 0; prev_i < i; prev_i++) {
                    const std::string& prev_attr = _input_attrs_and_directions[prev_i].first;
                    internal::DataChunk* prev_chunk = map.get_chunk_for_attr(prev_attr);
                    if (prev_chunk->get_state() == input_state) {
                        _input_level_indices.push_back(_input_level_indices[prev_i]);
                        found = true;
                        break;
                    }
                }
            }

            // Should always find a match
            if (!found) {
                throw std::runtime_error("NWayIntersectionPredicated: could not map input " + input_attr +
                                         " to state path level");
            }
        }
    }

    _range_update_tree->precompute_effective_children();

    // Initialize next operator
    _next_op->init(schema);
}

template<typename T>
void NWayIntersectionPredicated<T>::process_data_chunk(BitMask<State::MAX_VECTOR_SIZE>* _current_ip_mask,
                                                       int32_t op_filled_idx) {

    SET_ALL_BITS(_out_vec->state->selector);
    SET_START_POS(*_out_vec->state, 0);
    SET_END_POS(*_out_vec->state, op_filled_idx);

    COPY_BITMASK(State::MAX_VECTOR_SIZE, *_current_ip_mask, *_last_input_valid_mask_uptr);
    COPY_BITMASK(State::MAX_VECTOR_SIZE, _input_vecs.back()->state->selector, *_last_input_valid_mask_uptr);
    const auto backup_start_pos = GET_START_POS(*_input_vecs.back()->state);
    const auto backup_end_pos = GET_END_POS(*_input_vecs.back()->state);

    store_slices();
    auto is_vector_empty = _range_update_tree->start_propagation();

    if (!is_vector_empty && (_invalidated_count > 0)) {
        is_vector_empty = _range_update_tree->start_propagation_cascade(_invalidated_indices.get(), _invalidated_count);
    }
    _invalidated_count = 0;

    if (!is_vector_empty) { _next_op->execute(); }

    restore_slices();

    COPY_BITMASK(State::MAX_VECTOR_SIZE, *_last_input_valid_mask_uptr, *_current_ip_mask);
    SET_START_POS(*_input_vecs.back()->state, backup_start_pos);
    SET_END_POS(*_input_vecs.back()->state, backup_end_pos);
}

// Helper functions for ftree state update
template<typename T>
static void register_node_in_saved_data_nwip(FtreeStateUpdateNode* node,
                                             NWayIntersectionPredicatedVectorSliceUpdateSavedData<T>* vector_saved_data,
                                             uint32_t& saved_data_index) {
    // Get current start and end positions from the vector's selector
    int32_t current_start = GET_START_POS(*node->vector->state);
    int32_t current_end = GET_END_POS(*node->vector->state);

    // Create VectorSavedData entry
    vector_saved_data[saved_data_index++] = NWayIntersectionPredicatedVectorSliceUpdateSavedData<T>(
            node->attribute, const_cast<Vector<T>*>(node->vector), current_start, current_end);
}

template<typename T>
static void
register_backward_nodes_in_saved_data_nwip(FtreeStateUpdateNode* parent_node,
                                           NWayIntersectionPredicatedVectorSliceUpdateSavedData<T>* vector_saved_data,
                                           uint32_t& saved_data_index) {
    for (auto& child: parent_node->children) {
        register_node_in_saved_data_nwip<T>(child.get(), vector_saved_data, saved_data_index);
        register_backward_nodes_in_saved_data_nwip<T>(child.get(), vector_saved_data, saved_data_index);
    }
}

// Helper function: compute intersection of two sorted arrays
template<typename T>
int32_t NWayIntersectionPredicated<T>::compute_sorted_intersection(const T* arr1, int32_t size1, const T* arr2,
                                                                   int32_t size2, T* dest) {

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
void NWayIntersectionPredicated<T>::execute() {
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

    // Get active range for last input attribute
    const int32_t last_start = GET_START_POS(*last_input_state);
    const int32_t last_end = GET_END_POS(*last_input_state);

    // Build all ancestor index buffers
    if (_all_same_data_chunk) {
        // All in same DataChunk: identity mapping - no processing needed
        // Each position maps to itself, handled inline in the loop below
    } else {
        // Different DataChunks: use FtreeMultiAncestorFinder
        _multi_ancestor_finder->process(_ancestor_index_buffer_ptrs.data(), last_start, last_end);
    }

    // Initialize output write index
    int32_t out_write_idx = 0;

    bool should_set_last_start_pos = true;

    // Allocate stack arrays
    const size_t num_levels = _input_vecs.size();
    auto** temp_buffers = static_cast<T**>(alloca(sizeof(T*) * num_levels));
    temp_buffers[0] = nullptr;// First level uses adj_list.values directly
    for (size_t i = 1; i < num_levels; i++) {
        temp_buffers[i] = static_cast<T*>(alloca(sizeof(T) * State::MAX_VECTOR_SIZE));
    }
    auto* intersection_sizes = static_cast<int32_t*>(alloca(sizeof(int32_t) * num_levels));

    // Filtered buffer for predicate filtering
    T* filtered_buf = static_cast<T*>(alloca(sizeof(T) * State::MAX_VECTOR_SIZE));

    // Iterate over last input level (deepest level)
    for (int32_t last_pos = static_cast<int32_t>(next_set_bit_in_range(*_last_input_valid_mask_uptr, last_start,
                                                                        last_end));
         last_pos <= last_end;
         last_pos = static_cast<int32_t>(next_set_bit_in_range(*_last_input_valid_mask_uptr,
                                                                static_cast<uint32_t>(last_pos + 1), last_end))) {

        // Look up ancestor indices at all input levels (excluding last)
        // For same DataChunk, use identity mapping (index == position)
        for (size_t i = 0; i < _input_vecs.size() - 1; i++) {
            uint32_t ancestor_idx;
            if (_all_same_data_chunk) {
                ancestor_idx = static_cast<uint32_t>(last_pos);// Identity mapping
            } else {
                size_t path_level = _input_level_indices[i];
                ancestor_idx = _ancestor_index_buffers[path_level][last_pos];
            }
            State* ancestor_state = _input_vecs[i]->state;
            assert(ancestor_idx != UINT32_MAX);
            assert(TEST_BIT(ancestor_state->selector, ancestor_idx));
        }

        // Get value at last position
        const T last_val = last_input_vals[last_pos];

        // Compute N-way intersection
        // Start with first input's adjacency list
        uint32_t first_ancestor_idx;
        if (_all_same_data_chunk) {
            first_ancestor_idx = static_cast<uint32_t>(last_pos);// Identity mapping
        } else {
            size_t first_path_level = _input_level_indices[0];
            first_ancestor_idx = _ancestor_index_buffers[first_path_level][last_pos];
        }
        const T first_val = _input_vecs[0]->values[first_ancestor_idx];
        const auto& first_adj_list = _adj_lists[0][first_val];
        intersection_sizes[0] = static_cast<int32_t>(first_adj_list.size);

        // Progressively intersect with each subsequent input level
        bool empty_intersection = (intersection_sizes[0] == 0);

        for (size_t input_level = 1; input_level < num_levels && !empty_intersection; input_level++) {
            // Get value at this input level
            T level_val;
            if (input_level == last_input_idx) {
                level_val = last_val;
            } else {
                uint32_t level_idx;
                if (_all_same_data_chunk) {
                    level_idx = static_cast<uint32_t>(last_pos);// Identity mapping
                } else {
                    size_t path_level = _input_level_indices[input_level];
                    level_idx = _ancestor_index_buffers[path_level][last_pos];
                }
                level_val = _input_vecs[input_level]->values[level_idx];
            }

            const auto& level_adj_list = _adj_lists[input_level][level_val];
            const int32_t adj_size = static_cast<int32_t>(level_adj_list.size);

            // Get previous buffer
            const T* prev_buffer = (input_level == 1) ? first_adj_list.values : temp_buffers[input_level - 1];
            int32_t prev_size = intersection_sizes[input_level - 1];

            if (prev_size > 0 && adj_size > 0) {
                intersection_sizes[input_level] = compute_sorted_intersection(
                        prev_buffer, prev_size, level_adj_list.values, adj_size, temp_buffers[input_level]);
            } else {
                intersection_sizes[input_level] = 0;
            }

            if (intersection_sizes[input_level] == 0) { empty_intersection = true; }
        }

        if (intersection_sizes[num_levels - 1] == 0) {
            CLEAR_BIT(*_last_input_valid_mask_uptr, last_pos);
            _invalidated_indices[_invalidated_count++] = last_pos;
            continue;
        }

        // Get final intersection result
        int32_t final_size = intersection_sizes[last_input_idx];
        T* final_buffer = temp_buffers[last_input_idx];

        // Apply predicate filtering
        int32_t filtered_count = 0;
        if (_is_string_predicate) {
            for (int32_t i = 0; i < final_size; ++i) {
                const bool pass = _predicate_expr_string.evaluate_id(static_cast<uint64_t>(final_buffer[i]));
                filtered_buf[filtered_count] = final_buffer[i];
                filtered_count += pass ? 1 : 0;
            }
        } else {
            for (int32_t i = 0; i < final_size; ++i) {
                const bool pass = _predicate_expr_numeric.evaluate(final_buffer[i]);
                filtered_buf[filtered_count] = final_buffer[i];
                filtered_count += pass ? 1 : 0;
            }
        }

        if (filtered_count == 0) {
            CLEAR_BIT(*_last_input_valid_mask_uptr, last_pos);
            _invalidated_indices[_invalidated_count++] = last_pos;
            continue;
        }

        // Update last input start/end positions
        SET_END_POS(*last_input_state, last_pos);
        if (should_set_last_start_pos) {
            SET_START_POS(*last_input_state, last_pos);
            should_set_last_start_pos = false;
        }

        // Write filtered values to output vector
        int32_t read_idx = 0;
        while (read_idx < filtered_count) {
            const uint32_t remaining_space = State::MAX_VECTOR_SIZE - out_write_idx;
            const auto elements_to_copy = std::min(remaining_space, static_cast<uint32_t>(filtered_count - read_idx));

            if constexpr (std::is_same_v<T, uint64_t>) {
                std::memcpy(&out_vals[out_write_idx], &filtered_buf[read_idx], elements_to_copy * sizeof(T));
            } else {
                for (uint32_t k = 0; k < elements_to_copy; ++k) {
                    out_vals[out_write_idx + k] = filtered_buf[read_idx + k];
                }
            }

            out_offset[last_pos] = out_write_idx;
            out_offset[last_pos + 1] = out_write_idx + elements_to_copy;

            assert(out_offset[last_pos] < out_offset[last_pos + 1]);

            out_write_idx += elements_to_copy;
            read_idx += elements_to_copy;

            // Handle chunking if output vector is full
            if (out_write_idx == State::MAX_VECTOR_SIZE) {
                process_data_chunk(&_current_ip_mask, out_write_idx - 1);
                out_write_idx = 0;

                SET_START_POS(*last_input_state, last_pos);
                SET_END_POS(*last_input_state, last_pos);
                should_set_last_start_pos = (read_idx == filtered_count);
            }
        }
    }

    // Process final chunk if any
    if (out_write_idx > 0) { process_data_chunk(&_current_ip_mask, out_write_idx - 1); }

    // Restore the original selector pointer for last input
    COPY_BITMASK(State::MAX_VECTOR_SIZE, last_input_state->selector, _last_input_selector_backup);
    _invalidated_count = 0;
}

// Explicit template instantiations
template class NWayIntersectionPredicated<uint64_t>;
// template class NWayIntersectionPredicated<ffx_str_t>; // TODO: Enable when hash-based adjacency list indexing is implemented

}// namespace ffx
