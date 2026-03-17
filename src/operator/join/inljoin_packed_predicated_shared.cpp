#include "join/inljoin_packed_predicated_shared.hpp"

#include "../../table/include/cardinality.hpp"
#include <cassert>
#include <cstring>
#include <iostream>
#include <type_traits>
#include <vector/bitmask.hpp>
#include <vector>

namespace ffx {

template<typename T>
static void register_node_in_saved_data_pred_shared(FtreeStateUpdateNode* node,
                                                    VectorSliceUpdateSavedData<T>* vector_saved_data,
                                                    uint32_t& saved_data_index);
template<typename T>
static void register_backward_nodes_in_saved_data_pred_shared(FtreeStateUpdateNode* parent_node,
                                                              VectorSliceUpdateSavedData<T>* vector_saved_data,
                                                              uint32_t& saved_data_index);

template<typename T>
void INLJoinPackedPredicatedShared<T>::create_slice_update_infrastructure(FactorizedTreeElement* ftree_leaf) {
    _vector_saved_data = std::make_unique<VectorSliceUpdateSavedData<T>[]>(_vector_saved_data_count);
    uint32_t saved_data_index = 0;
    const auto join_key_fnode = ftree_leaf->_parent;
    _range_update_tree->fill_bwd_join_key(join_key_fnode, _output_key);
    register_backward_nodes_in_saved_data_pred_shared<T>(_range_update_tree.get(), _vector_saved_data.get(),
                                                         saved_data_index);

    FactorizedTreeElement* ftreenode = join_key_fnode->_parent;
    FtreeStateUpdateNode* current_node = _range_update_tree.get();
    while (ftreenode != nullptr) {
        auto child = std::make_unique<FtreeStateUpdateNode>(ftreenode->_value, FORWARD, ftreenode->_attribute);
        auto child_ptr = child.get();
        register_node_in_saved_data_pred_shared<T>(child_ptr, _vector_saved_data.get(), saved_data_index);
        child_ptr->parent = current_node;
        current_node->children.push_back(std::move(child));
        child_ptr->fill_bwd(ftreenode, _output_key);
        register_backward_nodes_in_saved_data_pred_shared<T>(child_ptr, _vector_saved_data.get(), saved_data_index);
        current_node = current_node->children.back().get();
        ftreenode = ftreenode->_parent;
    }
}

template<typename T>
void INLJoinPackedPredicatedShared<T>::init(Schema* schema) {
    static_assert(std::is_same_v<T, uint64_t>, "INLJoinPackedPredicatedShared currently only supports uint64_t.");

    auto& map = *schema->map;
    auto root = schema->root;

    _in_vec = map.get_vector(_join_key);

    ResolvedJoinAdjList resolved;
    if (!schema->try_resolve_join_adj_list(_join_key, _output_key, resolved)) {
        throw std::runtime_error("No table found for join " + _join_key + " -> " + _output_key);
    }
    _adj_lists = reinterpret_cast<AdjList<T>*>(resolved.adj_list);
    _is_join_index_fwd = resolved.is_fwd;
    std::cout << "INLJoinPackedPredicatedShared(" << _join_key << "->" << _output_key << ")"
              << (resolved.from_schema_map ? " using Schema adj_list" : " using fallback table adj_list")
              << " (" << (resolved.is_fwd ? "fwd" : "bwd") << ") [SHARED STATE]";

    // Always allocate output vector with shared state
    _out_vec = map.allocate_vector_shared_state<T>(_output_key, _join_key);
    const auto ftree_leaf = root->add_leaf(_join_key, _output_key, _in_vec, _out_vec);
    _range_update_tree = std::make_unique<FtreeStateUpdateNode>(_in_vec, NONE, _join_key);

    // Build the scalar predicate expression for this attribute
    _is_string_predicate = (schema->string_attributes && schema->string_attributes->count(_output_key) > 0);
    if (_is_string_predicate) {
        _predicate_expr_string = build_scalar_predicate_expr<ffx_str_t>(_predicate_expr_raw, _output_key,
                                                                        schema->predicate_pool, schema->dictionary);
    } else {
        _predicate_expr_numeric = build_scalar_predicate_expr<T>(_predicate_expr_raw, _output_key,
                                                                 schema->predicate_pool, schema->dictionary);
    }

    if ((_is_string_predicate && _predicate_expr_string.has_predicates()) ||
        (!_is_string_predicate && _predicate_expr_numeric.has_predicates())) {
        std::cout << " [predicates: "
                  << (_is_string_predicate ? _predicate_expr_string.to_string() : _predicate_expr_numeric.to_string())
                  << "]";
    }
    std::cout << std::endl;

    const auto unique_datachunks = root->count_unique_datachunks();
    _vector_saved_data_count = root->get_num_nodes() - 2;
    if (_vector_saved_data_count > 0 && unique_datachunks > 2) { create_slice_update_infrastructure(ftree_leaf); }
    else { _vector_saved_data_count = 0; }
    _invalidated_indices = std::make_unique<uint32_t[]>(State::MAX_VECTOR_SIZE);
    _invalidated_count = 0;
    _active_mask_uptr = std::make_unique<BitMask<State::MAX_VECTOR_SIZE>>();
    _range_update_tree->precompute_effective_children();
    _next_op->init(schema);
}

template<typename T>
void INLJoinPackedPredicatedShared<T>::store_slices() {
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
void INLJoinPackedPredicatedShared<T>::restore_slices() {
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
void INLJoinPackedPredicatedShared<T>::execute() {
    num_exec_call++;
    store_slices();

    State* RESTRICT in_state = _in_vec->state;
    const T* RESTRICT in_vals = _in_vec->values;
    T* RESTRICT out_vals = _out_vec->values;

    COPY_BITMASK(State::MAX_VECTOR_SIZE, _in_selector_backup, in_state->selector);
    COPY_BITMASK(State::MAX_VECTOR_SIZE, *_active_mask_uptr, _in_selector_backup);
    COPY_BITMASK(State::MAX_VECTOR_SIZE, in_state->selector, *_active_mask_uptr);

    const auto in_vec_start_idx = GET_START_POS(*in_state);
    const auto in_vec_end_idx = GET_END_POS(*in_state);

    // Shared-state path: exactly 1 output per input position, written at same index (no dense packing / no out_offset).
    // Use temporary buffer for better cache performance
    T temp_out_vals[State::MAX_VECTOR_SIZE];
    for (auto pidx = static_cast<int32_t>(next_set_bit_in_range(*_active_mask_uptr, in_vec_start_idx,
                                                                 in_vec_end_idx));
         pidx <= in_vec_end_idx;
         pidx = static_cast<int32_t>(next_set_bit_in_range(*_active_mask_uptr, static_cast<uint32_t>(pidx + 1),
                                                            in_vec_end_idx))) {
        const auto& adj_list = _adj_lists[in_vals[pidx]];
        const auto adj_list_size = static_cast<int32_t>(adj_list.size);
        if (adj_list_size == 0) {
            CLEAR_BIT(*_active_mask_uptr, pidx);
            _invalidated_indices[_invalidated_count++] = static_cast<uint32_t>(pidx);
            continue;
        }
        assert(adj_list_size == 1 && "State sharing requires exactly one output per input position");
        const T v = adj_list.values[0];
        const bool pass = _is_string_predicate ? _predicate_expr_string.evaluate_id(static_cast<uint64_t>(v))
                                               : _predicate_expr_numeric.evaluate(v);
        if (!pass) {
            CLEAR_BIT(*_active_mask_uptr, pidx);
            _invalidated_indices[_invalidated_count++] = static_cast<uint32_t>(pidx);
            continue;
        }
        temp_out_vals[pidx] = v;
    }
    // Copy all values in one sweep (garbage values don't matter, bitmask handles validity)
    std::memcpy(&out_vals[in_vec_start_idx], &temp_out_vals[in_vec_start_idx],
                (in_vec_end_idx - in_vec_start_idx + 1) * sizeof(T));

    // Shrink start/end to first/last valid
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

        // Maintain updated state for shared-state downstream operators
        SET_START_POS(*in_state, new_start);
        SET_END_POS(*in_state, new_end);
        COPY_BITMASK(State::MAX_VECTOR_SIZE, in_state->selector, *_active_mask_uptr);
    } else {
        _invalidated_count = 0;
    }

    COPY_BITMASK(State::MAX_VECTOR_SIZE, in_state->selector, _in_selector_backup);
    in_state->start_pos = in_vec_start_idx;
    in_state->end_pos = in_vec_end_idx;
}

template<typename T>
static void register_node_in_saved_data_pred_shared(FtreeStateUpdateNode* node,
                                                    VectorSliceUpdateSavedData<T>* vector_saved_data,
                                                    uint32_t& saved_data_index) {
    int32_t current_start = GET_START_POS(*node->vector->state);
    int32_t current_end = GET_END_POS(*node->vector->state);
    vector_saved_data[saved_data_index++] = VectorSliceUpdateSavedData<T>(
            node->attribute, const_cast<Vector<T>*>(node->vector), current_start, current_end);
}

template<typename T>
static void register_backward_nodes_in_saved_data_pred_shared(FtreeStateUpdateNode* parent_node,
                                                              VectorSliceUpdateSavedData<T>* vector_saved_data,
                                                              uint32_t& saved_data_index) {
    for (const auto& child: parent_node->children) {
        register_node_in_saved_data_pred_shared<T>(child.get(), vector_saved_data, saved_data_index);
        register_backward_nodes_in_saved_data_pred_shared<T>(child.get(), vector_saved_data, saved_data_index);
    }
}

template class INLJoinPackedPredicatedShared<uint64_t>;

}// namespace ffx
