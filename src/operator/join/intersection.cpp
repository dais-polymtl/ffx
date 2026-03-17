#include "join/intersection.hpp"

#include "ancestor_finder_utils.hpp"
#include <cassert>
#include <cstring>
#include <iostream>
#include <vector/bitmask.hpp>
#include <vector>

namespace ffx {

template<typename T>
static void register_node_in_saved_data(FtreeStateUpdateNode* node,
                                        IntersectionVectorSliceUpdateSavedData<T>* vector_saved_data,
                                        uint32_t& saved_data_index, uint32_t max_size);
template<typename T>
static void register_backward_nodes_in_saved_data(FtreeStateUpdateNode* parent_node,
                                                  IntersectionVectorSliceUpdateSavedData<T>* vector_saved_data,
                                                  uint32_t& saved_data_index, uint32_t max_size);

template<typename T>
void Intersection<T>::create_slice_update_infrastructure(const FactorizedTreeElement* ftree_output_node) {
    _vector_saved_data = std::make_unique<IntersectionVectorSliceUpdateSavedData<T>[]>(_vector_saved_data_count);
    uint32_t saved_data_index = 0;
    const auto descendant_fnode = ftree_output_node->_parent;
    _range_update_tree->fill_bwd_join_key(/* parent node */ descendant_fnode, /* child to exclude */ _output_attr);
    register_backward_nodes_in_saved_data<T>(_range_update_tree.get(), _vector_saved_data.get(), saved_data_index,
                                             _vector_saved_data_count);

    FactorizedTreeElement* ftreenode = descendant_fnode->_parent;//ancestor node
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
void Intersection<T>::store_slices() {
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
void Intersection<T>::restore_slices() {
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
void Intersection<T>::init(Schema* schema) {
    static_assert(std::is_same_v<T, uint64_t>, "Intersection currently only supports uint64_t. String support requires "
                                               "hash-based adjacency list indexing.");
    auto& map = *schema->map;
    const auto& tables = schema->tables;
    auto root = schema->root;

    // Get vectors from schema
    _ancestor_vec = map.get_vector(_ancestor_attr);
    _descendant_vec = map.get_vector(_descendant_attr);
    // NOTE: Do NOT call set_current_parent_chunk here. The parent chunk should already
    // be set correctly by preceding operators (e.g., theta joins). If we override it
    // here, we break the DataChunk ancestor-descendant relationships needed for
    // transitive theta joins like EQ(a, b) AND EQ(b, c).
    _out_vec = map.allocate_vector(_output_attr);

    // Get RLE array
    _descendant_offset = _descendant_vec->state->offset;

    // Build state path using the utility function
    auto path_info = internal::build_ancestor_finder_path(map, _ancestor_attr, _descendant_attr);
    _same_data_chunk = path_info.same_data_chunk;

    // Set up ancestor index buffer
    _ancestor_index_buffer = std::make_unique<uint32_t[]>(State::MAX_VECTOR_SIZE);

    // Create FtreeAncestorFinder only if not in the same DataChunk
    if (!_same_data_chunk) {
        _ancestor_finder = std::make_unique<FtreeAncestorFinder>(path_info.state_path.data(),
                                      path_info.state_path.size());
    }

    // Find nodes in ftree for verification and infrastructure setup
    FactorizedTreeElement* ancestor_node = root->find_node_by_attribute(_ancestor_attr);
    FactorizedTreeElement* descendant_node = root->find_node_by_attribute(_descendant_attr);

    if (!ancestor_node || !descendant_node) {
        throw std::runtime_error("Intersection: nodes not found in tree for " + _ancestor_attr + " or " +
                                 _descendant_attr);
    }

    ResolvedJoinAdjList ancestor_resolved;
    if (!schema->try_resolve_join_adj_list(_ancestor_attr, _output_attr, ancestor_resolved)) {
        throw std::runtime_error("Intersection: No table found for ancestor " + _ancestor_attr + " and output " +
                                 _output_attr);
    }
    _ancestor_adj_lists = reinterpret_cast<AdjList<T>*>(ancestor_resolved.adj_list);

    ResolvedJoinAdjList descendant_resolved;
    if (!schema->try_resolve_join_adj_list(_descendant_attr, _output_attr, descendant_resolved)) {
        throw std::runtime_error("Intersection: No table found for descendant " + _descendant_attr +
                                 " and output " + _output_attr);
    }
    _descendant_adj_lists = reinterpret_cast<AdjList<T>*>(descendant_resolved.adj_list);

    // Add to factorized tree
    root->add_leaf(_descendant_attr, _output_attr, _descendant_vec, _out_vec);

    // Set up range update infrastructure
    FactorizedTreeElement* output_node = root->find_node_by_attribute(_output_attr);
    _range_update_tree = std::make_unique<FtreeStateUpdateNode>(_descendant_vec, NONE, _descendant_attr);


    const auto unique_datachunks = root->count_unique_datachunks();
    _vector_saved_data_count = root->get_num_nodes() - 2;// Exclude descendant, output_attr
    if (_vector_saved_data_count > 0 && unique_datachunks > 2) { create_slice_update_infrastructure(output_node); }
    else { _vector_saved_data_count = 0; }

    // Initialize bitmasks
    _descendant_valid_mask_uptr = std::make_unique<BitMask<State::MAX_VECTOR_SIZE>>();
    _range_update_tree->precompute_effective_children();

    // Initialize next operator
    _next_op->init(schema);
}

template<typename T>
uint32_t Intersection<T>::get_intersection(T a_val, T b_val, const AdjList<T>& ancestor_adj_list,
                                           const AdjList<T>& descendant_adj_list, T* dest_buffer,
                                           int32_t max_dest_size) {

    // Compute intersection without caching, writing directly into destination buffer
    const auto ancestor_output_size = static_cast<int32_t>(ancestor_adj_list.size);
    const auto descendant_output_size = static_cast<int32_t>(descendant_adj_list.size);

    uint32_t intersection_count = 0;

    // Sorted merge intersection, writing directly to destination buffer
    int32_t ancestor_idx = 0;
    int32_t descendant_idx = 0;

    while (ancestor_idx < ancestor_output_size && descendant_idx < descendant_output_size &&
           intersection_count < max_dest_size) {
        const T ancestor_val = ancestor_adj_list.values[ancestor_idx];
        const T descendant_val = descendant_adj_list.values[descendant_idx];

        if (ancestor_val == descendant_val) {
            dest_buffer[intersection_count++] = ancestor_val;
            ancestor_idx++;
            descendant_idx++;
        } else if (ancestor_val < descendant_val) {
            ancestor_idx++;
        } else {
            descendant_idx++;
        }
    }

    return intersection_count;
}

template<typename T>
const IntersectionCacheEntry<T>& Intersection<T>::get_or_compute_intersection(T a_val, T b_val,
                                                                              const AdjList<T>& ancestor_adj_list,
                                                                              const AdjList<T>& descendant_adj_list) {

    // Create cache key
    auto cache_key = std::make_pair(a_val, b_val);

    // Check if already cached
    auto it = _intersection_cache.find(cache_key);
    if (it != _intersection_cache.end()) { return it->second; }

    // Not cached, compute intersection
    const int32_t ancestor_output_size = static_cast<int32_t>(ancestor_adj_list.size);
    const int32_t descendant_output_size = static_cast<int32_t>(descendant_adj_list.size);
    const int32_t max_intersection_size = std::min(ancestor_output_size, descendant_output_size);

    // Use alloca for temporary computation buffer
    auto* temp_buffer = static_cast<T*>(alloca(sizeof(T) * max_intersection_size));
    int32_t intersection_count = 0;

    // Sorted merge intersection
    int32_t ancestor_idx = 0;
    int32_t descendant_idx = 0;

    while (ancestor_idx < ancestor_output_size && descendant_idx < descendant_output_size) {
        const T ancestor_val = ancestor_adj_list.values[ancestor_idx];
        const T descendant_val = descendant_adj_list.values[descendant_idx];

        if (ancestor_val == descendant_val) {
            temp_buffer[intersection_count++] = ancestor_val;
            ancestor_idx++;
            descendant_idx++;
        } else if (ancestor_val < descendant_val) {
            ancestor_idx++;
        } else {
            descendant_idx++;
        }
    }

    // Create cache entry
    IntersectionCacheEntry<T> entry;
    entry.count = intersection_count;

    if (intersection_count <= State::MAX_VECTOR_SIZE) {
        // Use stack storage
        entry.is_heap = false;
        if constexpr (std::is_trivially_copyable_v<T>) {
            std::memcpy(entry.stack_values, temp_buffer, intersection_count * sizeof(T));
        } else {
            for (int32_t i = 0; i < intersection_count; ++i) {
                new (&entry.stack_values[i]) T(temp_buffer[i]);
            }
        }
    } else {
        // Use heap storage
        entry.is_heap = true;
        new (&entry.heap_values) std::unique_ptr<T[]>(new T[intersection_count]);
        if constexpr (std::is_trivially_copyable_v<T>) {
            std::memcpy(entry.heap_values.get(), temp_buffer, intersection_count * sizeof(T));
        } else {
            for (int32_t i = 0; i < intersection_count; ++i) {
                entry.heap_values[i] = temp_buffer[i];
            }
        }
    }

    // Insert into cache and return reference
    auto [inserted_it, _] = _intersection_cache.emplace(cache_key, std::move(entry));
    return inserted_it->second;
}

template<typename T>
void Intersection<T>::execute() {
    num_exec_call++;

    // Get input state and values
    State* ancestor_state = _ancestor_vec->state;
    const T* ancestor_vals = _ancestor_vec->values;

    State* descendant_state = _descendant_vec->state;
    const T* descendant_vals = _descendant_vec->values;

    State* out_state = _out_vec->state;
    T* out_vals = _out_vec->values;
    uint16_t* out_offset = out_state->offset;

    // Save the original selector pointers
    COPY_BITMASK(State::MAX_VECTOR_SIZE, _descendant_selector_backup, descendant_state->selector);

    // Copy ancestor selector to ancestor active mask
    COPY_BITMASK(State::MAX_VECTOR_SIZE, *_descendant_valid_mask_uptr, _descendant_selector_backup);

    // Update descendant selector pointer to point to valid mask
    COPY_BITMASK(State::MAX_VECTOR_SIZE, descendant_state->selector, *_descendant_valid_mask_uptr);

    // Get active range
    const int32_t ancestor_start = GET_START_POS(*ancestor_state);
    const int32_t ancestor_end = GET_END_POS(*ancestor_state);

    // Get descendant range
    const int32_t descendant_start = GET_START_POS(*descendant_state);
    const int32_t descendant_end = GET_END_POS(*descendant_state);

    // Initialize output write index
    int32_t out_write_idx = 0;

    // Build the ancestor index buffer for this morsel
    if (_same_data_chunk) {
        // Same DataChunk: identity mapping (each position maps to itself)
        for (int32_t idx = descendant_start; idx <= descendant_end; idx++) {
            _ancestor_index_buffer[idx] = static_cast<uint32_t>(idx);
        }
    } else {
        // Different DataChunks: use FtreeAncestorFinder
        _ancestor_finder->process(_ancestor_index_buffer.get(), ancestor_start, ancestor_end, descendant_start,
                                  descendant_end);
    }

    // For each descendant position
    bool should_set_desc_start_pos = true;
    for (int32_t b_pos = static_cast<int32_t>(
                 next_set_bit_in_range(*_descendant_valid_mask_uptr, descendant_start, descendant_end));
         b_pos <= descendant_end;
         b_pos = static_cast<int32_t>(next_set_bit_in_range(*_descendant_valid_mask_uptr,
                                                             static_cast<uint32_t>(b_pos + 1), descendant_end))) {

        // Get ancestor index from the finder
        const uint32_t a_idx = _ancestor_index_buffer[b_pos];
        assert(a_idx != UINT32_MAX && "Valid descendant cannot have invalid ancestor in Intersection::execute()");
        // if (a_idx == UINT32_MAX) {
        //     // No valid ancestor for this descendant
        //     CLEAR_BIT(*_descendant_valid_mask_uptr, b_pos);
        //     continue;
        // }

        // Check if ancestor is valid
        assert(TEST_BIT(ancestor_state->selector, a_idx) &&
               "Valid descendant cannot have invalid ancestor in Intersection::execute()");
        // if (!TEST_BIT(ancestor_state->selector, a_idx)) {
        //     CLEAR_BIT(*_descendant_valid_mask_uptr, b_pos);
        //     continue;
        // }

        // Get values
        const T a_val = ancestor_vals[a_idx];
        const T b_val = descendant_vals[b_pos];

        // Get adjacency lists
        const auto& ancestor_adj_list = _ancestor_adj_lists[a_val];
        const auto& descendant_adj_list = _descendant_adj_lists[b_val];
        const auto ancestor_output_size = static_cast<int32_t>(ancestor_adj_list.size);
        const auto descendant_output_size = static_cast<int32_t>(descendant_adj_list.size);

        if (ancestor_output_size == 0 || descendant_output_size == 0) {
            CLEAR_BIT(*_descendant_valid_mask_uptr, b_pos);
            continue;
        }

        const int32_t max_intersection_size = std::min(ancestor_output_size, descendant_output_size);
        auto* intersection_buffer = static_cast<T*>(alloca(sizeof(T) * max_intersection_size));
        const uint32_t intersection_count = get_intersection(a_val, b_val, ancestor_adj_list, descendant_adj_list,
                                                             intersection_buffer, max_intersection_size);

        if (intersection_count == 0) {
            CLEAR_BIT(*_descendant_valid_mask_uptr, b_pos);
            continue;
        }


        // Update descendant range
        SET_END_POS(*descendant_state, b_pos);
        if (should_set_desc_start_pos) {
            SET_START_POS(*descendant_state, b_pos);
            should_set_desc_start_pos = false;
        }

        // Write intersection to output
        int32_t intersection_read_idx = 0;
        while (intersection_read_idx < static_cast<int32_t>(intersection_count)) {
            const uint32_t remaining_space = State::MAX_VECTOR_SIZE - out_write_idx;
            const auto elements_to_copy = std::min(remaining_space, intersection_count - intersection_read_idx);

            std::memcpy(&out_vals[out_write_idx], &intersection_buffer[intersection_read_idx],
                        elements_to_copy * sizeof(uint64_t));

            out_offset[b_pos] = out_write_idx;
            out_offset[b_pos + 1] = out_write_idx + elements_to_copy;

            out_write_idx += elements_to_copy;
            intersection_read_idx += elements_to_copy;

            if (out_write_idx == State::MAX_VECTOR_SIZE) {
                process_data_chunk(&_current_ip_mask, out_write_idx - 1);
                out_write_idx = 0;
                SET_START_POS(*descendant_state, b_pos);
                SET_END_POS(*descendant_state, b_pos);
                should_set_desc_start_pos = (intersection_read_idx == static_cast<int32_t>(intersection_count));
            }
        }
    }

    // Process final chunk if any
    if (out_write_idx > 0) { process_data_chunk(&_current_ip_mask, out_write_idx - 1); }

    // Restore the original selector pointers
    COPY_BITMASK(State::MAX_VECTOR_SIZE, descendant_state->selector, _descendant_selector_backup);

    // Clear the intersection cache to free memory
    _intersection_cache.clear();
}

template<typename T>
__attribute__((always_inline)) inline void
Intersection<T>::process_data_chunk(BitMask<State::MAX_VECTOR_SIZE>* _current_ip_mask, const int32_t op_filled_idx) {

    SET_ALL_BITS(_out_vec->state->selector);
    SET_START_POS(*_out_vec->state, 0);
    SET_END_POS(*_out_vec->state, op_filled_idx);

    // Backup descendant valid mask state
    COPY_BITMASK(State::MAX_VECTOR_SIZE, *_current_ip_mask, *_descendant_valid_mask_uptr);
    COPY_BITMASK(State::MAX_VECTOR_SIZE, _descendant_vec->state->selector, *_descendant_valid_mask_uptr);
    const auto descendant_backup_start_pos = GET_START_POS(*_descendant_vec->state);
    const auto descendant_backup_end_pos = GET_END_POS(*_descendant_vec->state);

    // Save slices before propagation
    store_slices();

    // Update ancestor slices before executing the next operator
    const auto is_vector_empty = _range_update_tree->start_propagation();

    // Execute next operator
    if (!is_vector_empty) { _next_op->execute(); }

    // Restore ancestor slices after executing next operator
    restore_slices();

    // Restore descendant valid mask state
    COPY_BITMASK(State::MAX_VECTOR_SIZE, *_descendant_valid_mask_uptr, *_current_ip_mask);
    SET_START_POS(*_descendant_vec->state, descendant_backup_start_pos);
    SET_END_POS(*_descendant_vec->state, descendant_backup_end_pos);
}

template<typename T>
static void register_node_in_saved_data(FtreeStateUpdateNode* node,
                                        IntersectionVectorSliceUpdateSavedData<T>* vector_saved_data,
                                        uint32_t& saved_data_index, uint32_t max_size) {
    // Get current start and end positions from the vector's selector
    int32_t current_start = GET_START_POS(*node->vector->state);
    int32_t current_end = GET_END_POS(*node->vector->state);

    // Create VectorSavedData entry
    vector_saved_data[saved_data_index++] = IntersectionVectorSliceUpdateSavedData<T>(
            node->attribute, const_cast<Vector<T>*>(node->vector), current_start, current_end);
}

template<typename T>
static void register_backward_nodes_in_saved_data(FtreeStateUpdateNode* parent_node,
                                                  IntersectionVectorSliceUpdateSavedData<T>* vector_saved_data,
                                                  uint32_t& saved_data_index, uint32_t max_size) {
    for (const auto& child: parent_node->children) {
        // Register this backward node
        register_node_in_saved_data<T>(child.get(), vector_saved_data, saved_data_index, max_size);

        // Recursively register its children
        register_backward_nodes_in_saved_data<T>(child.get(), vector_saved_data, saved_data_index, max_size);
    }
}

// Explicit template instantiations
template class Intersection<uint64_t>;
// template class Intersection<ffx_str_t>; // TODO: Enable when hash-based adjacency list indexing is implemented

}// namespace ffx
