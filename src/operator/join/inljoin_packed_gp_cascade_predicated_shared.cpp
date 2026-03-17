#include "join/inljoin_packed_gp_cascade_predicated_shared.hpp"

#include "../../table/include/cardinality.hpp"
#include <cassert>
#include <cstring>
#include <iostream>
#include <type_traits>
#include <vector/bitmask.hpp>
#include <vector>

namespace ffx {

template<typename T>
static void register_node_in_saved_data_gp_pred_shared(FtreeStateUpdateNode* node,
                                                       VectorGPSavedData<T>* vector_saved_data,
                                                       uint32_t& saved_data_index);
template<typename T>
static void register_backward_nodes_in_saved_data_gp_pred_shared(FtreeStateUpdateNode* parent_node,
                                                                 VectorGPSavedData<T>* vector_saved_data,
                                                                 uint32_t& saved_data_index);

template<typename T>
void INLJoinPackedGPCascadePredicatedShared<T>::create_slice_update_infrastructure(
        const FactorizedTreeElement* ftree_leaf) {
    _vector_saved_data = std::make_unique<VectorGPSavedData<T>[]>(_vector_saved_data_count);
    uint32_t saved_data_index = 0;
    const auto join_key_fnode = ftree_leaf->_parent;
    _range_update_tree->fill_bwd_join_key(join_key_fnode, _output_key);
    register_backward_nodes_in_saved_data_gp_pred_shared<T>(_range_update_tree.get(), _vector_saved_data.get(),
                                                            saved_data_index);

    FactorizedTreeElement* ftreenode = join_key_fnode->_parent;
    FtreeStateUpdateNode* current_node = _range_update_tree.get();
    while (ftreenode != nullptr) {
        auto child = std::make_unique<FtreeStateUpdateNode>(ftreenode->_value, FORWARD, ftreenode->_attribute);
        auto child_ptr = child.get();
        register_node_in_saved_data_gp_pred_shared<T>(child_ptr, _vector_saved_data.get(), saved_data_index);
        child_ptr->parent = current_node;
        current_node->children.push_back(std::move(child));
        child_ptr->fill_bwd(ftreenode, _output_key);
        register_backward_nodes_in_saved_data_gp_pred_shared<T>(child_ptr, _vector_saved_data.get(), saved_data_index);
        current_node = current_node->children.back().get();
        ftreenode = ftreenode->_parent;
    }
}

template<typename T>
void INLJoinPackedGPCascadePredicatedShared<T>::init(Schema* schema) {
    static_assert(std::is_same_v<T, uint64_t>,
                  "INLJoinPackedGPCascadePredicatedShared currently only supports uint64_t.");
    auto& map = *schema->map;
    auto root = schema->root;

    _in_vec = map.get_vector(_join_key);

    ResolvedJoinAdjList resolved;
    if (!schema->try_resolve_join_adj_list(_join_key, _output_key, resolved)) {
        throw std::runtime_error("No table found for join " + _join_key + " -> " + _output_key);
    }
    _adj_lists = reinterpret_cast<AdjList<T>*>(resolved.adj_list);
    _is_join_index_fwd = resolved.is_fwd;
    std::cout << "INLJoinPackedGPCascadePredicatedShared(" << _join_key << "->" << _output_key << ")"
              << (resolved.from_schema_map ? " using Schema adj_list" : " using fallback table adj_list")
              << " (" << (resolved.is_fwd ? "fwd" : "bwd") << ") [SHARED STATE]";

    // Always allocate output vector with shared state
    _out_vec = map.allocate_vector_shared_state<T>(_output_key, _join_key);
    const auto ftree_leaf = root->add_leaf(_join_key, _output_key, _in_vec, _out_vec);
    _range_update_tree = std::make_unique<FtreeStateUpdateNode>(_in_vec, NONE, _join_key);

    {
        const auto& join_key_ftree_node = ftree_leaf->_parent;
        const auto& grandparent_ftree_node = join_key_ftree_node->_parent;
        assert(grandparent_ftree_node != nullptr);
        _grandparent_state = grandparent_ftree_node->_value->state;
        assert(_grandparent_state != nullptr);
        _invalid_gp_indices = std::make_unique<uint32_t[]>(State::MAX_VECTOR_SIZE);
        _invalid_gp_indices_cnt = 0;
    }

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
    if (_vector_saved_data_count > 0 && unique_datachunks > 3) { create_slice_update_infrastructure(ftree_leaf); }
    else { _vector_saved_data_count = 0; }
    _active_mask_uptr = std::make_unique<BitMask<State::MAX_VECTOR_SIZE>>();
    _range_update_tree->precompute_effective_children();
    _next_op->init(schema);
}

template<typename T>
void INLJoinPackedGPCascadePredicatedShared<T>::store_slices() {
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
void INLJoinPackedGPCascadePredicatedShared<T>::restore_slices() {
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
void INLJoinPackedGPCascadePredicatedShared<T>::execute() {
    num_exec_call++;
    store_slices();

    State* RESTRICT in_state = _in_vec->state;
    const T* RESTRICT in_vals = _in_vec->values;
    T* RESTRICT out_vals = _out_vec->values;

    // Check if input vector has identity RLE (shares state with grandparent)
    const bool has_identity_rle = _in_vec->has_identity_rle();
    uint16_t* RESTRICT in_offset = has_identity_rle ? nullptr : in_state->offset;

    COPY_BITMASK(State::MAX_VECTOR_SIZE, _in_selector_backup, in_state->selector);
    COPY_BITMASK(State::MAX_VECTOR_SIZE, *_active_mask_uptr, _in_selector_backup);
    COPY_BITMASK(State::MAX_VECTOR_SIZE, in_state->selector, *_active_mask_uptr);

    const auto gp_start_pos = GET_START_POS(*_grandparent_state);
    const auto gp_end_pos = GET_END_POS(*_grandparent_state);

    const auto in_vec_start_idx = GET_START_POS(*in_state);
    const auto in_vec_end_idx = GET_END_POS(*in_state);

    // Shared-state path: one output per input pos, no dense packing / no out_offset writes.
    // Use temporary buffer for better cache performance
    T temp_out_vals[State::MAX_VECTOR_SIZE];
    _invalid_gp_indices_cnt = 0;

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
            const auto sz = static_cast<int32_t>(adj_list.size);
            if (sz == 0) {
                CLEAR_BIT(*_active_mask_uptr, in_vector_idx);
                parent_count--;
                continue;
            }
            const T v = adj_list.values[0];
            const bool pass = _is_string_predicate ? _predicate_expr_string.evaluate_id(static_cast<uint64_t>(v))
                                                   : _predicate_expr_numeric.evaluate(v);
            if (!pass) {
                CLEAR_BIT(*_active_mask_uptr, in_vector_idx);
                parent_count--;
                continue;
            }
            temp_out_vals[in_vector_idx] = v;
        }
        parent_count -= (range_len - active_in_range);

        _invalid_gp_indices[_invalid_gp_indices_cnt] = gp_idx;
        _invalid_gp_indices_cnt += parent_count == 0;
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
    }

    COPY_BITMASK(State::MAX_VECTOR_SIZE, in_state->selector, *_active_mask_uptr);
    auto is_vector_empty = _range_update_tree->start_propagation();
    if (!is_vector_empty && (_invalid_gp_indices_cnt > 0)) {
        is_vector_empty =
                _range_update_tree->start_propagation_fwd_cascade(_invalid_gp_indices.get(), _invalid_gp_indices_cnt);
    }
    if (!is_vector_empty && (new_start <= new_end)) { _next_op->execute(); }

    restore_slices();

    if (new_start <= new_end) {
        // Maintain updated state for shared-state downstream operators
        SET_START_POS(*in_state, new_start);
        SET_END_POS(*in_state, new_end);
        COPY_BITMASK(State::MAX_VECTOR_SIZE, in_state->selector, *_active_mask_uptr);
    }
    _invalid_gp_indices_cnt = 0;
}

template<typename T>
static void register_node_in_saved_data_gp_pred_shared(FtreeStateUpdateNode* node,
                                                       VectorGPSavedData<T>* vector_saved_data,
                                                       uint32_t& saved_data_index) {
    int32_t current_start = GET_START_POS(*node->vector->state);
    int32_t current_end = GET_END_POS(*node->vector->state);
    vector_saved_data[saved_data_index++] =
            VectorGPSavedData<T>(node->attribute, const_cast<Vector<T>*>(node->vector), current_start, current_end);
}

template<typename T>
static void register_backward_nodes_in_saved_data_gp_pred_shared(FtreeStateUpdateNode* parent_node,
                                                                 VectorGPSavedData<T>* vector_saved_data,
                                                                 uint32_t& saved_data_index) {
    for (const auto& child: parent_node->children) {
        register_node_in_saved_data_gp_pred_shared<T>(child.get(), vector_saved_data, saved_data_index);
        register_backward_nodes_in_saved_data_gp_pred_shared<T>(child.get(), vector_saved_data, saved_data_index);
    }
}

template class INLJoinPackedGPCascadePredicatedShared<uint64_t>;

}// namespace ffx
