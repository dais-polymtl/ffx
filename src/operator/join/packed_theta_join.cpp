#include "join/packed_theta_join.hpp"

#include "ancestor_finder_utils.hpp"
#include <cassert>
#include <cstring>
#include <iostream>
#include <type_traits>
#include <vector/bitmask.hpp>

namespace ffx {

template<typename T>
static void register_node_in_saved_data_ptj(FtreeStateUpdateNode* node,
                                            PackedThetaJoinVectorSliceUpdateSavedData<T>* vector_saved_data,
                                            uint32_t& saved_data_index);

template<typename T>
static void register_backward_nodes_in_saved_data_ptj(FtreeStateUpdateNode* parent_node,
                                                      PackedThetaJoinVectorSliceUpdateSavedData<T>* vector_saved_data,
                                                      uint32_t& saved_data_index);

template<typename T>
void PackedThetaJoin<T>::create_slice_update_infrastructure(FactorizedTreeElement* ftree_right_node) {
    _vector_saved_data = std::make_unique<PackedThetaJoinVectorSliceUpdateSavedData<T>[]>(_vector_saved_data_count);
    uint32_t saved_data_index = 0;

    // Add all ancestor nodes (excluding filtered node) to the range_update_tree
    // Start from the immediate parent and traverse up
    FactorizedTreeElement* ftreenode = ftree_right_node->_parent;
    FtreeStateUpdateNode* current_node = _range_update_tree.get();

    while (ftreenode != nullptr) {
        auto child = std::make_unique<FtreeStateUpdateNode>(ftreenode->_value, FORWARD, ftreenode->_attribute);
        auto child_ptr = child.get();
        register_node_in_saved_data_ptj<T>(child_ptr, _vector_saved_data.get(), saved_data_index);
        child_ptr->parent = current_node;
        current_node->children.push_back(std::move(child));
        child_ptr->fill_bwd(ftreenode, _right_attr);
        register_backward_nodes_in_saved_data_ptj<T>(child_ptr, _vector_saved_data.get(), saved_data_index);
        current_node = current_node->children.back().get();
        ftreenode = ftreenode->_parent;
    }
}

template<typename T>
void PackedThetaJoin<T>::store_slices() {
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
void PackedThetaJoin<T>::restore_slices() {
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
void PackedThetaJoin<T>::init(Schema* schema) {
    auto& map = *schema->map;
    auto root = schema->root;

    // Get vectors for both attributes
    _left_vec = map.get_vector(_left_attr);
    _right_vec = map.get_vector(_right_attr);

    if (!_left_vec || !_right_vec) {
        throw std::runtime_error("PackedThetaJoin: vectors not found for " + _left_attr + " or " + _right_attr);
    }

    // Build state path using the utility function
    auto path_info = internal::build_ancestor_finder_path(map, _left_attr, _right_attr);
    _same_data_chunk = path_info.same_data_chunk;


    // Create FtreeAncestorFinder only if not in the same DataChunk
    if (!_same_data_chunk) {
        _ancestor_finder = std::make_unique<FtreeAncestorFinder>(path_info.state_path.data(),
                                      path_info.state_path.size());
    }

    // Find right_node in ftree for slice update infrastructure
    FactorizedTreeElement* right_node = root->find_node_by_attribute(_right_attr);
    if (!right_node) { throw std::runtime_error("PackedThetaJoin: right_attr node not found in tree"); }

    // Set up range update tree for ftree state propagation
    // The right_attr (descendant) is the one being filtered, so use it as the root
    _range_update_tree = std::make_unique<FtreeStateUpdateNode>(_right_vec, NONE, _right_attr);

    // Set up slice update infrastructure
    // Count excludes: right_attr (filtered node)
    _vector_saved_data_count = root->get_num_nodes() - 1;
    if (_vector_saved_data_count > 0) { create_slice_update_infrastructure(right_node); }

    // Initialize bitmask
    _right_valid_mask_uptr = std::make_unique<BitMask<State::MAX_VECTOR_SIZE>>();

    // Initialize cascade propagation tracking
    _invalidated_indices = std::make_unique<uint32_t[]>(State::MAX_VECTOR_SIZE);
    _invalidated_count = 0;

    // Ensure downstream operators (that might be added to this branch) see right_attr as the parent
    map.set_current_parent_chunk(_right_attr);

    // Print operator info
    std::cout << "PackedThetaJoin(" << _left_attr << " " << predicate_op_to_string(_op) << " " << _right_attr << ")"
              << std::endl;

    _range_update_tree->precompute_effective_children();

    // Initialize next operator
    _next_op->init(schema);
}

template<typename T>
void PackedThetaJoin<T>::execute() {
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
        // Same DataChunk: identity mapping (each position maps to itself)
        for (int32_t idx = right_start; idx <= right_end; idx++) {
            ancestor_indices[idx] = static_cast<uint32_t>(idx);
        }
    } else {
        // Different DataChunks: use FtreeAncestorFinder
        _ancestor_finder->process(ancestor_indices, left_start, left_end, right_start, right_end);
    }

    // Step 2 & 3: Build slice mapping - ancestor_indices has contiguous repeated values
    // Store (ancestor_idx, slice_start, slice_end) for each slice.
    // Some positions may have no valid ancestor (marked as UINT32_MAX) – we treat those
    // descendants as invalid and clear their bits before building slices.
    uint32_t slice_ancestors[State::MAX_VECTOR_SIZE];
    int32_t slice_starts[State::MAX_VECTOR_SIZE];
    int32_t slice_ends[State::MAX_VECTOR_SIZE];
    uint32_t num_slices = 0;

    int32_t curr_idx = right_start;
    while (curr_idx <= right_end) {
        uint32_t current_ancestor = ancestor_indices[curr_idx];

        // If no valid ancestor, clear this position and move on
        if (current_ancestor == UINT32_MAX) {
            CLEAR_BIT(*_right_valid_mask_uptr, curr_idx);
            curr_idx++;
            continue;
        }

        // Find end of this ancestor's slice (contiguous range with same ancestor)
        int32_t slice_start = curr_idx;
        int32_t slice_end = curr_idx;
        while (slice_end < right_end && ancestor_indices[slice_end + 1] == current_ancestor) {
            slice_end++;
        }

        // Store slice info
        slice_ancestors[num_slices] = current_ancestor;
        slice_starts[num_slices] = slice_start;
        slice_ends[num_slices] = slice_end;
        num_slices++;

        curr_idx = slice_end + 1;
    }

    // Step 4: Process each slice - apply predicate for ancestor value against its descendants
    int32_t new_right_start = -1;
    int32_t new_right_end = -1;

    for (uint32_t s = 0; s < num_slices; s++) {
        const uint32_t ancestor_idx = slice_ancestors[s];
        const int32_t s_start = slice_starts[s];
        const int32_t s_end = slice_ends[s];

        assert(ancestor_idx != UINT32_MAX && "Invalid ancestor index in slice mapping");
        const T left_val = left_vals[ancestor_idx];

        // Apply predicate to entire descendant slice
        for (int32_t r_idx = static_cast<int32_t>(next_set_bit_in_range(*_right_valid_mask_uptr, s_start, s_end));
             r_idx <= s_end;
             r_idx = static_cast<int32_t>(
                 next_set_bit_in_range(*_right_valid_mask_uptr, static_cast<uint32_t>(r_idx + 1), s_end))) {

            const T right_val = right_vals[r_idx];
            const bool passes = _pred_fn(left_val, right_val);

            if (passes) {
                if (new_right_start == -1) { new_right_start = r_idx; }
                new_right_end = r_idx;
            } else {
                CLEAR_BIT(*_right_valid_mask_uptr, r_idx);
                _invalidated_indices[_invalidated_count++] = r_idx;
            }
        }
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

    // Cascade propagation for sibling nodes (handles interior bit changes)
    if (!is_vector_empty && _invalidated_count > 0) {
        is_vector_empty = _range_update_tree->start_propagation_cascade(_invalidated_indices.get(), _invalidated_count);
    }
    _invalidated_count = 0;

    // Execute next operator only if there are valid tuples
    if (!is_vector_empty) { _next_op->execute(); }

    // Restore slices after executing next operator
    restore_slices();

    // Restore the original selector pointer
    COPY_BITMASK(State::MAX_VECTOR_SIZE, right_state->selector, _right_selector_backup);
}

template<typename T>
static void register_node_in_saved_data_ptj(FtreeStateUpdateNode* node,
                                            PackedThetaJoinVectorSliceUpdateSavedData<T>* vector_saved_data,
                                            uint32_t& saved_data_index) {
    int32_t current_start = GET_START_POS(*node->vector->state);
    int32_t current_end = GET_END_POS(*node->vector->state);
    vector_saved_data[saved_data_index++] = PackedThetaJoinVectorSliceUpdateSavedData<T>(
            node->attribute, const_cast<Vector<T>*>(node->vector), current_start, current_end);
}

template<typename T>
static void register_backward_nodes_in_saved_data_ptj(FtreeStateUpdateNode* parent_node,
                                                      PackedThetaJoinVectorSliceUpdateSavedData<T>* vector_saved_data,
                                                      uint32_t& saved_data_index) {
    for (const auto& child: parent_node->children) {
        register_node_in_saved_data_ptj<T>(child.get(), vector_saved_data, saved_data_index);
        register_backward_nodes_in_saved_data_ptj<T>(child.get(), vector_saved_data, saved_data_index);
    }
}

// Explicit template instantiations
template class PackedThetaJoin<uint64_t>;
// template class PackedThetaJoin<ffx_str_t>; // TODO: Enable when needed

}// namespace ffx
