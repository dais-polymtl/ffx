#include "join/property_filter.hpp"

#include "../../table/include/table.hpp"
#include <cassert>
#include <cstring>
#include <iostream>
#include <vector/bitmask.hpp>

namespace ffx {

static void register_node_in_saved_data_pf(FtreeStateUpdateNode* node, PropertyFilterSavedData* vector_saved_data,
                                           uint32_t& saved_data_index) {
    int32_t current_start = GET_START_POS(*node->vector->state);
    int32_t current_end = GET_END_POS(*node->vector->state);
    vector_saved_data[saved_data_index++] =
            PropertyFilterSavedData(node->attribute, const_cast<Vector<uint64_t>*>(node->vector), current_start,
                                    current_end);
}

static void register_backward_nodes_in_saved_data_pf(FtreeStateUpdateNode* parent_node,
                                                      PropertyFilterSavedData* vector_saved_data,
                                                      uint32_t& saved_data_index) {
    for (const auto& child: parent_node->children) {
        register_node_in_saved_data_pf(child.get(), vector_saved_data, saved_data_index);
        register_backward_nodes_in_saved_data_pf(child.get(), vector_saved_data, saved_data_index);
    }
}

PropertyFilter::PropertyFilter(std::string entity_attr, std::vector<PropertyLookupInfo> properties,
                               PredicateExpression predicate_expr)
    : Operator(), _entity_attr(std::move(entity_attr)), _properties(std::move(properties)),
      _predicate_expr_raw(std::move(predicate_expr)), _entity_vec(nullptr) {}

void PropertyFilter::create_slice_update_infrastructure(FactorizedTreeElement* ftree_node) {
    _vector_saved_data = std::make_unique<PropertyFilterSavedData[]>(_vector_saved_data_count);
    uint32_t saved_data_index = 0;

    // Traverse up from entity node's parent to root, adding each to the range_update_tree
    FactorizedTreeElement* ftreenode = ftree_node->_parent;
    FtreeStateUpdateNode* current_node = _range_update_tree.get();

    while (ftreenode != nullptr) {
        auto child = std::make_unique<FtreeStateUpdateNode>(ftreenode->_value, FORWARD, ftreenode->_attribute);
        auto child_ptr = child.get();
        register_node_in_saved_data_pf(child_ptr, _vector_saved_data.get(), saved_data_index);
        child_ptr->parent = current_node;
        current_node->children.push_back(std::move(child));
        child_ptr->fill_bwd(ftreenode, _entity_attr);
        register_backward_nodes_in_saved_data_pf(child_ptr, _vector_saved_data.get(), saved_data_index);
        current_node = current_node->children.back().get();
        ftreenode = ftreenode->_parent;
    }
}

void PropertyFilter::init(Schema* schema) {
    auto& map = *schema->map;
    auto root = schema->root;

    _entity_vec = map.get_vector(_entity_attr);
    if (!_entity_vec) { throw std::runtime_error("PropertyFilter: vector not found for " + _entity_attr); }

    // Resolve adjacency lists for each property
    _resolved_properties.reserve(_properties.size());
    for (const auto& prop: _properties) {
        ResolvedProperty rp;
        rp.attr_name = prop.property_attr;
        rp.is_string = (schema->string_attributes && schema->string_attributes->count(prop.property_attr) > 0);

        ResolvedJoinAdjList resolved;
        if (!schema->try_resolve_join_adj_list(prop.entity_attr, prop.property_attr, resolved)) {
            throw std::runtime_error("PropertyFilter: No table found for " + prop.entity_attr + " -> " +
                                     prop.property_attr);
        }
        rp.adj_lists = reinterpret_cast<AdjList<uint64_t>*>(resolved.adj_list);

        _resolved_properties.push_back(std::move(rp));
    }

    // Build compiled predicates for each group/predicate in the expression
    // Map property_attr -> index in _resolved_properties
    for (size_t i = 0; i < _resolved_properties.size(); ++i) {
        _prop_idx_map[_resolved_properties[i].attr_name] = i;
    }
    const auto& prop_idx_map = _prop_idx_map;

    // Helper lambda to compile a single predicate
    auto compile_one = [&](const Predicate& pred) -> CompiledPredicate {
        CompiledPredicate cp;
        if (!pred.is_scalar()) return cp;

        auto it = prop_idx_map.find(pred.left_attr);
        if (it == prop_idx_map.end()) return cp;

        const auto& rp = _resolved_properties[it->second];
        cp.is_string = rp.is_string;

        PredicateExpression single_expr;
        single_expr.top_level_op = LogicalOp::AND;
        PredicateGroup single_group;
        single_group.op = LogicalOp::AND;
        single_group.predicates.push_back(pred);
        single_expr.groups.push_back(std::move(single_group));

        if (rp.is_string) {
            auto compiled_expr = build_scalar_predicate_expr<ffx_str_t>(single_expr, pred.left_attr,
                                                                        schema->predicate_pool, schema->dictionary);
            if (!compiled_expr.groups.empty() && !compiled_expr.groups[0].predicates.empty()) {
                cp.string = std::move(compiled_expr.groups[0].predicates[0]);
            }
        } else {
            auto compiled_expr = build_scalar_predicate_expr<uint64_t>(single_expr, pred.left_attr,
                                                                       schema->predicate_pool);
            if (!compiled_expr.groups.empty() && !compiled_expr.groups[0].predicates.empty()) {
                cp.numeric = std::move(compiled_expr.groups[0].predicates[0]);
            }
        }
        return cp;
    };

    _compiled.resize(_predicate_expr_raw.groups.size());
    _compiled_branches.resize(_predicate_expr_raw.groups.size());
    for (size_t g = 0; g < _predicate_expr_raw.groups.size(); ++g) {
        const auto& group = _predicate_expr_raw.groups[g];

        if (group.has_branches()) {
            // Compile predicates within each branch
            _compiled_branches[g].resize(group.branches.size());
            for (size_t b = 0; b < group.branches.size(); ++b) {
                const auto& branch = group.branches[b];
                _compiled_branches[g][b].resize(branch.predicates.size());
                for (size_t p = 0; p < branch.predicates.size(); ++p) {
                    _compiled_branches[g][b][p] = compile_one(branch.predicates[p]);
                }
            }
        } else {
            // Flat predicates (original path)
            _compiled[g].resize(group.predicates.size());
            for (size_t p = 0; p < group.predicates.size(); ++p) {
                _compiled[g][p] = compile_one(group.predicates[p]);
            }
        }
    }

    // Set up ftree state propagation (same pattern as PackedThetaJoin)
    FactorizedTreeElement* entity_node = root->find_node_by_attribute(_entity_attr);
    if (!entity_node) { throw std::runtime_error("PropertyFilter: entity node not found in ftree: " + _entity_attr); }

    _range_update_tree = std::make_unique<FtreeStateUpdateNode>(_entity_vec, NONE, _entity_attr);

    // Count nodes for saved data (all nodes except the entity itself)
    _vector_saved_data_count = root->get_num_nodes() - 1;
    if (_vector_saved_data_count > 0) { create_slice_update_infrastructure(entity_node); }

    _valid_mask_uptr = std::make_unique<BitMask<State::MAX_VECTOR_SIZE>>();
    _invalidated_indices = std::make_unique<uint32_t[]>(State::MAX_VECTOR_SIZE);
    _invalidated_count = 0;

    _range_update_tree->precompute_effective_children();

    // Print operator info
    std::cout << "PropertyFilter(" << _entity_attr << ") properties: [";
    for (size_t i = 0; i < _resolved_properties.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << _resolved_properties[i].attr_name;
    }
    std::cout << "] predicates: " << _predicate_expr_raw.to_string() << std::endl;

    map.set_current_parent_chunk(_entity_attr);
    _next_op->init(schema);
}

void PropertyFilter::store_slices() {
    for (std::size_t i = 0; i < _vector_saved_data_count; i++) {
        auto& vec_data = _vector_saved_data[i];
        const Vector<uint64_t>* vec = vec_data.vector;
        auto& [start_pos, end_pos] = vec_data.backup_state;
        auto& state = *vec->state;
        start_pos = GET_START_POS(state);
        end_pos = GET_END_POS(state);
    }
}

void PropertyFilter::restore_slices() {
    for (std::size_t i = 0; i < _vector_saved_data_count; i++) {
        auto& vec_data = _vector_saved_data[i];
        const Vector<uint64_t>* vec = vec_data.vector;
        const auto& [start_pos, end_pos] = vec_data.backup_state;
        auto& state = *vec->state;
        SET_START_POS(state, start_pos);
        SET_END_POS(state, end_pos);
    }
}

bool PropertyFilter::evaluate_predicate_for_entity(uint64_t entity_val) const {
    const auto& groups = _predicate_expr_raw.groups;
    if (groups.empty()) return true;

    // Helper: evaluate a single compiled predicate against entity value
    auto eval_one = [&](const Predicate& pred, const CompiledPredicate& cp) -> bool {
        if (!pred.is_scalar()) return true;

        auto it = _prop_idx_map.find(pred.left_attr);
        if (it == _prop_idx_map.end()) return true;

        const auto& rp = _resolved_properties[it->second];
        const auto& adj = rp.adj_lists[entity_val];
        if (adj.size == 0) return false;

        if (cp.is_string) {
            return cp.string.evaluate_id(static_cast<uint64_t>(adj.values[0]));
        } else {
            return cp.numeric.evaluate(adj.values[0]);
        }
    };

    // Helper: evaluate a flat list of predicates with AND/OR
    auto eval_flat = [&](const std::vector<Predicate>& preds,
                         const std::vector<CompiledPredicate>& compiled_preds,
                         LogicalOp op) -> bool {
        if (preds.empty()) return true;
        if (op == LogicalOp::AND) {
            for (size_t p = 0; p < preds.size(); ++p) {
                if (!eval_one(preds[p], compiled_preds[p])) return false;
            }
            return true;
        } else {
            for (size_t p = 0; p < preds.size(); ++p) {
                if (eval_one(preds[p], compiled_preds[p])) return true;
            }
            return false;
        }
    };

    for (size_t g = 0; g < groups.size(); ++g) {
        const auto& group = groups[g];
        bool group_result;

        if (group.has_branches()) {
            // Group with branches: each branch is an AND group, branches combined with OR
            group_result = false;
            for (size_t b = 0; b < group.branches.size(); ++b) {
                const auto& branch = group.branches[b];
                const auto& compiled_branch = _compiled_branches[g][b];
                // Each branch is AND
                bool branch_result = eval_flat(branch.predicates, compiled_branch, branch.op);
                if (branch_result) {
                    group_result = true;
                    break;
                }
            }
        } else if (group.predicates.empty()) {
            group_result = true;
        } else {
            group_result = eval_flat(group.predicates, _compiled[g], group.op);
        }

        // Combine with top-level op
        if (_predicate_expr_raw.top_level_op == LogicalOp::AND) {
            if (!group_result) return false;
        } else {
            if (group_result) return true;
        }
    }

    return (_predicate_expr_raw.top_level_op == LogicalOp::AND);
}

void PropertyFilter::execute() {
    num_exec_call++;
    store_slices();

    State* RESTRICT entity_state = _entity_vec->state;
    const uint64_t* RESTRICT entity_vals = _entity_vec->values;

    COPY_BITMASK(State::MAX_VECTOR_SIZE, _selector_backup, entity_state->selector);
    COPY_BITMASK(State::MAX_VECTOR_SIZE, *_valid_mask_uptr, _selector_backup);
    COPY_BITMASK(State::MAX_VECTOR_SIZE, entity_state->selector, *_valid_mask_uptr);

    const auto start_idx = GET_START_POS(*entity_state);
    const auto end_idx = GET_END_POS(*entity_state);

    int32_t new_start = -1;
    int32_t new_end = -1;

    for (auto idx = static_cast<int32_t>(next_set_bit_in_range(*_valid_mask_uptr, start_idx, end_idx)); idx <= end_idx;
         idx = static_cast<int32_t>(next_set_bit_in_range(*_valid_mask_uptr, static_cast<uint32_t>(idx + 1),
                                                          end_idx))) {

        const uint64_t entity_val = entity_vals[idx];

        // Look up all property values for this entity
        // Evaluate the full predicate expression
        bool passes = evaluate_predicate_for_entity(entity_val);

        if (passes) {
            if (new_start == -1) new_start = idx;
            new_end = idx;
        } else {
            CLEAR_BIT(*_valid_mask_uptr, idx);
            _invalidated_indices[_invalidated_count++] = idx;
        }
    }

    bool is_vector_empty = (new_start == -1);

    if (!is_vector_empty) {
        SET_START_POS(*entity_state, new_start);
        SET_END_POS(*entity_state, new_end);
    }

    if (!is_vector_empty) { COPY_BITMASK(State::MAX_VECTOR_SIZE, entity_state->selector, *_valid_mask_uptr); }

    // Propagate ftree state updates
    if (!is_vector_empty) { is_vector_empty = _range_update_tree->start_propagation(); }

    if (!is_vector_empty && _invalidated_count > 0) {
        is_vector_empty = _range_update_tree->start_propagation_cascade(_invalidated_indices.get(), _invalidated_count);
    }
    _invalidated_count = 0;

    if (!is_vector_empty) { _next_op->execute(); }

    restore_slices();
    COPY_BITMASK(State::MAX_VECTOR_SIZE, entity_state->selector, _selector_backup);
}

}// namespace ffx
