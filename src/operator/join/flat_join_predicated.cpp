#include "join/flat_join_predicated.hpp"

#include "ancestor_finder_utils.hpp"
#include <cassert>
#include <cstring>
#include <iostream>
#include <type_traits>
#include <vector/bitmask.hpp>

namespace ffx {

template<typename T>
static void register_node_in_saved_data_fj(FtreeStateUpdateNode* node,
                                           FlatJoinPredicatedVectorSliceUpdateSavedData<T>* vector_saved_data,
                                           uint32_t& saved_data_index);
template<typename T>
static void register_backward_nodes_in_saved_data_fj(FtreeStateUpdateNode* parent_node,
                                                     FlatJoinPredicatedVectorSliceUpdateSavedData<T>* vector_saved_data,
                                                     uint32_t& saved_data_index);

template<typename T>
void FlatJoinPredicated<T>::create_slice_update_infrastructure(FactorizedTreeElement* ftree_leaf) {
    _vector_saved_data = std::make_unique<FlatJoinPredicatedVectorSliceUpdateSavedData<T>[]>(_vector_saved_data_count);
    uint32_t saved_data_index = 0;
    const auto parent_fnode = ftree_leaf->_parent;
    _range_update_tree->fill_bwd_join_key(parent_fnode, _output_attr);
    register_backward_nodes_in_saved_data_fj<T>(_range_update_tree.get(), _vector_saved_data.get(), saved_data_index);

    FactorizedTreeElement* ftreenode = parent_fnode->_parent;
    FtreeStateUpdateNode* current_node = _range_update_tree.get();
    while (ftreenode != nullptr) {
        auto child = std::make_unique<FtreeStateUpdateNode>(ftreenode->_value, FORWARD, ftreenode->_attribute);
        auto child_ptr = child.get();
        register_node_in_saved_data_fj<T>(child_ptr, _vector_saved_data.get(), saved_data_index);
        child_ptr->parent = current_node;
        current_node->children.push_back(std::move(child));
        child_ptr->fill_bwd(ftreenode, _output_attr);
        register_backward_nodes_in_saved_data_fj<T>(child_ptr, _vector_saved_data.get(), saved_data_index);
        current_node = current_node->children.back().get();
        ftreenode = ftreenode->_parent;
    }
}

template<typename T>
void FlatJoinPredicated<T>::store_slices() {
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
void FlatJoinPredicated<T>::restore_slices() {
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
void FlatJoinPredicated<T>::init(Schema* schema) {
    static_assert(std::is_same_v<T, uint64_t>, "FlatJoinPredicated currently only supports uint64_t.");
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
        throw std::runtime_error("FlatJoinPredicated: No table found for LCA " + _lca_attr + " and output " +
                                 _output_attr);
    }
    _adj_lists = reinterpret_cast<AdjList<T>*>(resolved.adj_list);
    std::cout << "FlatJoinPredicated(" << _parent_attr << "," << _lca_attr << "->" << _output_attr << ")"
              << (resolved.from_schema_map ? " using Schema adj_list" : " using fallback table adj_list")
              << " (" << (_is_join_index_fwd ? "fwd" : "bwd") << ")";

    // Build the scalar predicate expression for this attribute
    _is_string_predicate = (schema->string_attributes && schema->string_attributes->count(_output_attr) > 0);
    if (_is_string_predicate) {
        _predicate_expr_string = build_scalar_predicate_expr<ffx_str_t>(_predicate_expr_raw, _output_attr,
                                                                        schema->predicate_pool, schema->dictionary);
    } else {
        _predicate_expr_numeric = build_scalar_predicate_expr<T>(_predicate_expr_raw, _output_attr,
                                                                 schema->predicate_pool, schema->dictionary);
    }

    if ((_is_string_predicate && _predicate_expr_string.has_predicates()) ||
        (!_is_string_predicate && _predicate_expr_numeric.has_predicates())) {
        std::cout << " [predicates: "
                  << (_is_string_predicate ? _predicate_expr_string.to_string() : _predicate_expr_numeric.to_string())
                  << "]";
    }
    std::cout << std::endl;

    const auto ftree_leaf = root->add_leaf(_parent_attr, _output_attr, _parent_vec, _out_vec);
    _range_update_tree = std::make_unique<FtreeStateUpdateNode>(_parent_vec, NONE, _parent_attr);

    const auto unique_datachunks = root->count_unique_datachunks();
    _vector_saved_data_count = root->get_num_nodes() - 2;
    if (_vector_saved_data_count > 0 && unique_datachunks > 2) { create_slice_update_infrastructure(ftree_leaf); }
    else { _vector_saved_data_count = 0; }

    // Build state path using the utility function
    auto path_info = internal::build_ancestor_finder_path(map, _lca_attr, _parent_attr);
    _same_data_chunk = path_info.same_data_chunk;

    // Create FtreeAncestorFinder only if not in the same DataChunk
    if (!_same_data_chunk) {
        _ancestor_finder = std::make_unique<FtreeAncestorFinder>(path_info.state_path.data(),
                                      path_info.state_path.size());
    }

    _invalidated_indices = std::make_unique<uint32_t[]>(State::MAX_VECTOR_SIZE);
    _invalidated_count = 0;
    _active_mask_uptr = std::make_unique<BitMask<State::MAX_VECTOR_SIZE>>();
    _range_update_tree->precompute_effective_children();
    _next_op->init(schema);
}

template<typename T>
void FlatJoinPredicated<T>::execute() {
    num_exec_call++;

    State* parent_state = _parent_vec->state;
    const T* lca_vals = _lca_vec->values;

    State* out_state = _out_vec->state;
    T* out_vals = _out_vec->values;
    uint16_t* out_offset = out_state->offset;

    COPY_BITMASK(State::MAX_VECTOR_SIZE, _parent_selector_backup, parent_state->selector);
    COPY_BITMASK(State::MAX_VECTOR_SIZE, *_active_mask_uptr, _parent_selector_backup);
    COPY_BITMASK(State::MAX_VECTOR_SIZE, _parent_vec->state->selector, *_active_mask_uptr);

    const int32_t parent_start = GET_START_POS(*parent_state);
    const int32_t parent_end = GET_END_POS(*parent_state);

    int32_t out_write_idx = 0;

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

    // Temp buffer for branchless predicate filtering
    T temp_buf[State::MAX_VECTOR_SIZE];

    bool should_set_start_pos = true;

    for (int32_t b_idx = static_cast<int32_t>(next_set_bit_in_range(*_active_mask_uptr, parent_start, parent_end));
         b_idx <= parent_end;
         b_idx = static_cast<int32_t>(
                 next_set_bit_in_range(*_active_mask_uptr, static_cast<uint32_t>(b_idx + 1), parent_end))) {

        const uint32_t a_idx = b_idx_to_a_idx[b_idx];
        assert(a_idx != UINT32_MAX && "b_idx must belong to some a_idx");

        const T a_val = lca_vals[a_idx];
        const auto& adj_list = _adj_lists[a_val];
        const auto adj_list_size = static_cast<int32_t>(adj_list.size);

        if (adj_list_size == 0) {
            CLEAR_BIT(*_active_mask_uptr, b_idx);
            _invalidated_indices[_invalidated_count++] = b_idx;
            continue;
        }

        bool any_passed = false;
        for (int32_t adj_offset = 0; adj_offset < adj_list_size; adj_offset += State::MAX_VECTOR_SIZE) {
            const int32_t chunk_size = std::min(adj_list_size - adj_offset, (int32_t) State::MAX_VECTOR_SIZE);

            // Step 1: Copy chunk of adjacency list values to temp buffer
            std::memcpy(temp_buf, &adj_list.values[adj_offset], chunk_size * sizeof(T));

            // Step 2: Branchless predicate compaction for this chunk
            int32_t filtered_count = 0;
            if (_is_string_predicate) {
                for (int32_t i = 0; i < chunk_size; ++i) {
                    const bool pass = _predicate_expr_string.evaluate_id(static_cast<uint64_t>(temp_buf[i]));
                    temp_buf[filtered_count] = temp_buf[i];
                    filtered_count += pass ? 1 : 0;
                }
            } else {
                for (int32_t i = 0; i < chunk_size; ++i) {
                    const bool pass = _predicate_expr_numeric.evaluate(temp_buf[i]);
                    temp_buf[filtered_count] = temp_buf[i];
                    filtered_count += pass ? 1 : 0;
                }
            }

            if (filtered_count == 0) continue;
            any_passed = true;

            SET_END_POS(*parent_state, b_idx);
            if (should_set_start_pos) {
                SET_START_POS(*parent_state, b_idx);
                should_set_start_pos = false;
            }

            // Process filtered values in chunks
            int32_t read_idx = 0;
            while (read_idx < filtered_count) {
                const auto remaining_space = State::MAX_VECTOR_SIZE - out_write_idx;
                const auto remaining_values = filtered_count - read_idx;
                const auto to_copy = std::min(remaining_space, remaining_values);

                std::memcpy(&out_vals[out_write_idx], &temp_buf[read_idx], to_copy * sizeof(T));

                out_offset[b_idx] = out_write_idx;
                out_offset[b_idx + 1] = out_write_idx + to_copy;

                out_write_idx += to_copy;
                read_idx += to_copy;

                if (out_write_idx == State::MAX_VECTOR_SIZE) {
                    process_data_chunk(&_current_ip_mask, out_write_idx - 1);
                    out_write_idx = 0;

                    if (read_idx < filtered_count || (adj_offset + State::MAX_VECTOR_SIZE < adj_list_size)) {
                        SET_START_POS(*parent_state, b_idx);
                        should_set_start_pos = false;
                    } else {
                        should_set_start_pos = true;
                    }
                }
            }
        }

        if (!any_passed) {
            CLEAR_BIT(*_active_mask_uptr, b_idx);
            _invalidated_indices[_invalidated_count++] = b_idx;
            continue;
        }
    }

    if (out_write_idx > 0) { process_data_chunk(&_current_ip_mask, out_write_idx - 1); }

    COPY_BITMASK(State::MAX_VECTOR_SIZE, _parent_vec->state->selector, _parent_selector_backup);
    _invalidated_count = 0;
}

template<typename T>
__attribute__((always_inline)) inline void
FlatJoinPredicated<T>::process_data_chunk(BitMask<State::MAX_VECTOR_SIZE>* _current_ip_mask,
                                          const int32_t op_filled_idx) {

    SET_ALL_BITS(_out_vec->state->selector);
    SET_START_POS(*_out_vec->state, 0);
    SET_END_POS(*_out_vec->state, op_filled_idx);

    COPY_BITMASK(State::MAX_VECTOR_SIZE, *_current_ip_mask, *_active_mask_uptr);
    COPY_BITMASK(State::MAX_VECTOR_SIZE, _parent_vec->state->selector, *_active_mask_uptr);
    const auto backup_start_pos = GET_START_POS(*_parent_vec->state);
    const auto backup_end_pos = GET_END_POS(*_parent_vec->state);

    auto is_vector_empty = _range_update_tree->start_propagation();

    if (!is_vector_empty && (_invalidated_count > 0)) {
        is_vector_empty = _range_update_tree->start_propagation_cascade(_invalidated_indices.get(), _invalidated_count);
    }
    _invalidated_count = 0;

    if (!is_vector_empty) { _next_op->execute(); }

    restore_slices();
    COPY_BITMASK(State::MAX_VECTOR_SIZE, *_active_mask_uptr, *_current_ip_mask);
    SET_START_POS(*_parent_vec->state, backup_start_pos);
    SET_END_POS(*_parent_vec->state, backup_end_pos);
}

template<typename T>
static void register_node_in_saved_data_fj(FtreeStateUpdateNode* node,
                                           FlatJoinPredicatedVectorSliceUpdateSavedData<T>* vector_saved_data,
                                           uint32_t& saved_data_index) {
    int32_t current_start = GET_START_POS(*node->vector->state);
    int32_t current_end = GET_END_POS(*node->vector->state);
    vector_saved_data[saved_data_index++] = FlatJoinPredicatedVectorSliceUpdateSavedData<T>(
            node->attribute, const_cast<Vector<T>*>(node->vector), current_start, current_end);
}

template<typename T>
static void register_backward_nodes_in_saved_data_fj(FtreeStateUpdateNode* parent_node,
                                                     FlatJoinPredicatedVectorSliceUpdateSavedData<T>* vector_saved_data,
                                                     uint32_t& saved_data_index) {
    for (const auto& child: parent_node->children) {
        register_node_in_saved_data_fj<T>(child.get(), vector_saved_data, saved_data_index);
        register_backward_nodes_in_saved_data_fj<T>(child.get(), vector_saved_data, saved_data_index);
    }
}

template class FlatJoinPredicated<uint64_t>;

}// namespace ffx
