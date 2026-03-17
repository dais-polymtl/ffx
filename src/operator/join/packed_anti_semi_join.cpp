#include "join/packed_anti_semi_join.hpp"

#include "ancestor_finder_utils.hpp"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <type_traits>
#include <vector/bitmask.hpp>

namespace ffx {

template<typename T>
static void register_node_in_saved_data_asj(FtreeStateUpdateNode* node,
                                            PackedAntiSemiJoinVectorSliceUpdateSavedData<T>* vector_saved_data,
                                            uint32_t& saved_data_index);

template<typename T>
static void
register_backward_nodes_in_saved_data_asj(FtreeStateUpdateNode* parent_node,
                                          PackedAntiSemiJoinVectorSliceUpdateSavedData<T>* vector_saved_data,
                                          uint32_t& saved_data_index);

template<typename T>
void PackedAntiSemiJoin<T>::create_slice_update_infrastructure(FactorizedTreeElement* ftree_right_node) {
    _vector_saved_data = std::make_unique<PackedAntiSemiJoinVectorSliceUpdateSavedData<T>[]>(_vector_saved_data_count);
    uint32_t saved_data_index = 0;

    // Add all ancestor nodes (excluding filtered node) to the range_update_tree
    // Start from the immediate parent and traverse up
    FactorizedTreeElement* ftreenode = ftree_right_node->_parent;
    FtreeStateUpdateNode* current_node = _range_update_tree.get();

    while (ftreenode != nullptr) {
        auto child = std::make_unique<FtreeStateUpdateNode>(ftreenode->_value, FORWARD, ftreenode->_attribute);
        auto child_ptr = child.get();
        register_node_in_saved_data_asj<T>(child_ptr, _vector_saved_data.get(), saved_data_index);
        child_ptr->parent = current_node;
        current_node->children.push_back(std::move(child));
        child_ptr->fill_bwd(ftreenode, _right_attr);
        register_backward_nodes_in_saved_data_asj<T>(child_ptr, _vector_saved_data.get(), saved_data_index);
        current_node = current_node->children.back().get();
        ftreenode = ftreenode->_parent;
    }
}

template<typename T>
void PackedAntiSemiJoin<T>::store_slices() {
    for (std::size_t i = 0; i < _vector_saved_data_count; i++) {
        auto& vec_data = _vector_saved_data[i];
        const Vector<T>* vec = vec_data.vector;
        auto& [start_pos, end_pos] = vec_data.backup_state;
        auto& state = *vec->state;
        start_pos = GET_START_POS(state);
        end_pos = GET_END_POS(state);
    }
}

template<typename T>
void PackedAntiSemiJoin<T>::restore_slices() {
    for (std::size_t i = 0; i < _vector_saved_data_count; i++) {
        auto& vec_data = _vector_saved_data[i];
        const Vector<T>* vec = vec_data.vector;
        const auto& [start_pos, end_pos] = vec_data.backup_state;
        auto& state = *vec->state;
        SET_START_POS(state, start_pos);
        SET_END_POS(state, end_pos);
    }
}

template<typename T>
void PackedAntiSemiJoin<T>::process_slice(T source_val, const T* right_vals, int32_t slice_start, int32_t slice_end,
                                          int32_t& new_right_start, int32_t& new_right_end) {

    // Get the adjacency list for the source node
    const AdjList<T>& adj = _adj_lists[source_val];

    // If adjacency list is empty, no edges exist - keep all valid targets
    if (adj.size == 0) {
        for (int32_t r_idx = static_cast<int32_t>(
                     next_set_bit_in_range(*_right_valid_mask_uptr, slice_start, slice_end));
             r_idx <= slice_end;
             r_idx = static_cast<int32_t>(next_set_bit_in_range(*_right_valid_mask_uptr,
                                                                 static_cast<uint32_t>(r_idx + 1), slice_end))) {
            if (new_right_start == -1) new_right_start = r_idx;
            new_right_end = r_idx;
        }
        return;
    }

    const T* adj_ptr = adj.values;
    const T* adj_end = adj.values + adj.size;

    // Linear merge: both adjacency list and right_vals (within valid bits) are sorted
    // For anti-semi-join: KEEP if NOT in adjacency list, REMOVE if in adjacency list
    for (int32_t r_idx = static_cast<int32_t>(next_set_bit_in_range(*_right_valid_mask_uptr, slice_start,
                                                                     slice_end));
         r_idx <= slice_end;
         r_idx = static_cast<int32_t>(
                 next_set_bit_in_range(*_right_valid_mask_uptr, static_cast<uint32_t>(r_idx + 1), slice_end))) {

        const T target_val = right_vals[r_idx];

        // Advance adj_ptr to catch up to target_val
        while (adj_ptr < adj_end && *adj_ptr < target_val) {
            adj_ptr++;
        }

        // Check if edge exists
        if (adj_ptr < adj_end && *adj_ptr == target_val) {
            // Edge exists - REMOVE this tuple
            CLEAR_BIT(*_right_valid_mask_uptr, r_idx);
        } else {
            // No edge exists - KEEP this tuple
            if (new_right_start == -1) new_right_start = r_idx;
            new_right_end = r_idx;
        }
    }
}

template<typename T>
void PackedAntiSemiJoin<T>::init(Schema* schema) {
    auto& map = *schema->map;
    auto root = schema->root;
    const auto& tables = schema->tables;

    // Get vectors for both attributes
    _left_vec = map.get_vector(_left_attr);
    _right_vec = map.get_vector(_right_attr);

    if (!_left_vec || !_right_vec) {
        throw std::runtime_error("PackedAntiSemiJoin: vectors not found for " + _left_attr + " or " + _right_attr);
    }

    // NEW: Try Schema-based adj_list lookup first (using left_attr -> right_attr)
    if (schema->has_adj_list(_left_attr, _right_attr)) {
        _adj_lists = reinterpret_cast<AdjList<T>*>(schema->get_adj_list(_left_attr, _right_attr));
        std::cout << "PackedAntiSemiJoin " << _left_attr << "->" << _right_attr << " using Schema adj_list"
                  << std::endl;
    } else {
        // FALLBACK: Legacy table-based lookup
        const Table* edge_table = nullptr;
        bool is_fwd = true;
        for (const auto* table: tables) {
            int left_idx = -1, right_idx = -1;
            for (size_t i = 0; i < table->columns.size(); ++i) {
                if (table->columns[i] == _left_attr) left_idx = static_cast<int>(i);
                if (table->columns[i] == _right_attr) right_idx = static_cast<int>(i);
            }
            if (left_idx != -1 && right_idx != -1) {
                edge_table = table;
                is_fwd = (left_idx == 0);
                break;
            }
        }

        if (!edge_table) {
            throw std::runtime_error("PackedAntiSemiJoin: no table found with columns " + _left_attr + " and " +
                                     _right_attr);
        }

        if (is_fwd) {
            _adj_lists = reinterpret_cast<AdjList<T>*>(edge_table->fwd_adj_lists);
        } else {
            _adj_lists = reinterpret_cast<AdjList<T>*>(edge_table->bwd_adj_lists);
        }

        if (!_adj_lists) { throw std::runtime_error("PackedAntiSemiJoin: adjacency lists not found in edge table"); }

        std::cout << "Anti-semi-join table selected: " << edge_table->name << "(" << _left_attr << " -> " << _right_attr
                  << " (" << (is_fwd ? "fwd" : "bwd") << ")" << std::endl;
    }

    // Build state path for ancestor finder using utility function
    auto path_info = internal::build_ancestor_finder_path(map, _left_attr, _right_attr);
    _same_data_chunk = path_info.same_data_chunk;

    // Create FtreeAncestorFinder only if not in same DataChunk
    // (same DataChunk means identity mapping - handled in execute())
    if (!_same_data_chunk) {
        _ancestor_finder = std::make_unique<FtreeAncestorFinder>(path_info.state_path.data(),
                                      path_info.state_path.size());
    }

    // Set up range update tree for ftree state propagation
    _range_update_tree = std::make_unique<FtreeStateUpdateNode>(_right_vec, NONE, _right_attr);

    // Find right_node for slice update infrastructure (still needed)
    FactorizedTreeElement* right_node = root->find_node_by_attribute(_right_attr);

    // Set up slice update infrastructure
    _vector_saved_data_count = root->get_num_nodes() - 1;
    if (_vector_saved_data_count > 0) { create_slice_update_infrastructure(right_node); }

    // Initialize bitmask
    _right_valid_mask_uptr = std::make_unique<BitMask<State::MAX_VECTOR_SIZE>>();

    // Print operator info
    std::cout << "PackedAntiSemiJoin(NOT " << _left_attr << "->" << _right_attr << ")" << std::endl;
    _range_update_tree->precompute_effective_children();

    // Initialize next operator
    _next_op->init(schema);
}

template<typename T>
void PackedAntiSemiJoin<T>::execute() {
    num_exec_call++;

    // Save slices before execute
    store_slices();

    // Get states and values
    State* left_state = _left_vec->state;
    const T* left_vals = _left_vec->values;

    State* right_state = _right_vec->state;
    const T* right_vals = _right_vec->values;

    // Save the original selector pointer for right vector (descendant)
    COPY_BITMASK(State::MAX_VECTOR_SIZE, _right_selector_backup, right_state->selector);

    // Copy right selector to valid mask
    COPY_BITMASK(State::MAX_VECTOR_SIZE, *_right_valid_mask_uptr, _right_selector_backup);

    // Update right selector to point to our valid mask
    COPY_BITMASK(State::MAX_VECTOR_SIZE, right_state->selector, *_right_valid_mask_uptr);

    // Get active ranges
    const int32_t left_start = GET_START_POS(*left_state);
    const int32_t left_end = GET_END_POS(*left_state);
    const int32_t right_start = GET_START_POS(*right_state);
    const int32_t right_end = GET_END_POS(*right_state);

    // Step 1: Get ancestor indices for each descendant
    uint32_t ancestor_indices[State::MAX_VECTOR_SIZE];
    if (_same_data_chunk) {
        // Identity mapping: descendant and ancestor share state
        // Each position maps to itself (filtered by validity)
        for (int32_t i = right_start; i <= right_end; i++) {
            if (i >= left_start && i <= left_end && TEST_BIT(_right_selector_backup, i)) {
                ancestor_indices[i] = static_cast<uint32_t>(i);
            } else {
                ancestor_indices[i] = UINT32_MAX;
            }
        }
    } else {
        _ancestor_finder->process(ancestor_indices, left_start, left_end, right_start, right_end);
    }

    // Step 2 & 3: Build slice mapping - ancestor_indices has contiguous repeated values
    uint32_t slice_ancestors[State::MAX_VECTOR_SIZE];
    int32_t slice_starts[State::MAX_VECTOR_SIZE];
    int32_t slice_ends[State::MAX_VECTOR_SIZE];
    uint32_t num_slices = 0;

    int32_t curr_slice_start = right_start;
    while (curr_slice_start <= right_end) {
        const uint32_t current_ancestor = ancestor_indices[curr_slice_start];

        // Find end of this ancestor's slice
        int32_t curr_slice_end = curr_slice_start;
        while (curr_slice_end < right_end && ancestor_indices[curr_slice_end + 1] == current_ancestor) {
            curr_slice_end++;
        }

        // Store slice info
        slice_ancestors[num_slices] = current_ancestor;
        slice_starts[num_slices] = curr_slice_start;
        slice_ends[num_slices] = curr_slice_end;
        num_slices++;

        curr_slice_start = curr_slice_end + 1;
    }

    // Step 4: Process each slice using linear merge
    // Anti-semi-join: KEEP tuples where edge does NOT exist, REMOVE where it exists
    int32_t new_right_start = -1;
    int32_t new_right_end = -1;

    for (uint32_t s = 0; s < num_slices; s++) {
        const uint32_t ancestor_idx = slice_ancestors[s];
        const int32_t s_start = slice_starts[s];
        const int32_t s_end = slice_ends[s];

        assert(ancestor_idx != UINT32_MAX && "Invalid ancestor index in slice mapping");
        const T left_val = left_vals[ancestor_idx];// Source node value

        // Process entire slice at once with linear merge
        process_slice(left_val, right_vals, s_start, s_end, new_right_start, new_right_end);
    }

    // Check if all bits were filtered out
    bool is_vector_empty = (new_right_start == -1);

    if (!is_vector_empty) {
        SET_START_POS(*right_state, new_right_start);
        SET_END_POS(*right_state, new_right_end);
    }

    // Sync valid mask to state->selector before propagation
    if (!is_vector_empty) { COPY_BITMASK(State::MAX_VECTOR_SIZE, right_state->selector, *_right_valid_mask_uptr); }

    // Propagate ftree state updates
    if (!is_vector_empty) { is_vector_empty = _range_update_tree->start_propagation(); }

    // Execute next operator only if there are valid tuples
    if (!is_vector_empty) { _next_op->execute(); }

    // Restore slices after executing next operator
    restore_slices();

    // Restore the original selector pointer
    COPY_BITMASK(State::MAX_VECTOR_SIZE, right_state->selector, _right_selector_backup);
}

template<typename T>
static void register_node_in_saved_data_asj(FtreeStateUpdateNode* node,
                                            PackedAntiSemiJoinVectorSliceUpdateSavedData<T>* vector_saved_data,
                                            uint32_t& saved_data_index) {
    int32_t current_start = GET_START_POS(*node->vector->state);
    int32_t current_end = GET_END_POS(*node->vector->state);
    vector_saved_data[saved_data_index++] = PackedAntiSemiJoinVectorSliceUpdateSavedData<T>(
            node->attribute, const_cast<Vector<T>*>(node->vector), current_start, current_end);
}

template<typename T>
static void
register_backward_nodes_in_saved_data_asj(FtreeStateUpdateNode* parent_node,
                                          PackedAntiSemiJoinVectorSliceUpdateSavedData<T>* vector_saved_data,
                                          uint32_t& saved_data_index) {
    for (const auto& child: parent_node->children) {
        register_node_in_saved_data_asj<T>(child.get(), vector_saved_data, saved_data_index);
        register_backward_nodes_in_saved_data_asj<T>(child.get(), vector_saved_data, saved_data_index);
    }
}

// Explicit template instantiations
template class PackedAntiSemiJoin<uint64_t>;
// template class PackedAntiSemiJoin<ffx_str_t>; // TODO: Enable when needed

}// namespace ffx
