#ifndef VFENGINE_PROPERTY_FILTER_OPERATOR_HH
#define VFENGINE_PROPERTY_FILTER_OPERATOR_HH

#include "../../table/include/adj_list.hpp"
#include "../../table/include/ffx_str_t.hpp"
#include "../factorized_ftree/ftree_state_update.hpp"
#include "../predicate/predicate_eval.hpp"
#include "operator.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace ffx {

struct PropertyLookupInfo {
    std::string property_attr;
    std::string entity_attr;
    bool is_fwd;
};

struct PropertyFilterSavedData {
    std::string attribute;
    Vector<uint64_t>* vector;
    std::pair<int32_t, int32_t> backup_state;
    PropertyFilterSavedData() : vector(nullptr), backup_state(-1, -1) {}
    PropertyFilterSavedData(const std::string& attr, Vector<uint64_t>* vec, int32_t start, int32_t end)
        : attribute(attr), vector(vec), backup_state(start, end) {}
};

class PropertyFilter final : public Operator {
public:
    PropertyFilter() = delete;
    PropertyFilter(const PropertyFilter&) = delete;

    PropertyFilter(std::string entity_attr, std::vector<PropertyLookupInfo> properties,
                   PredicateExpression predicate_expr);

    void init(Schema* schema) override;
    void execute() override;
    uint64_t get_num_output_tuples() override { return 0; }

    const std::string& entity_attr() const { return _entity_attr; }

private:
    void create_slice_update_infrastructure(FactorizedTreeElement* ftree_node);
    void store_slices();
    void restore_slices();

    std::string _entity_attr;
    std::vector<PropertyLookupInfo> _properties;
    PredicateExpression _predicate_expr_raw;

    Vector<uint64_t>* _entity_vec;

    struct ResolvedProperty {
        std::string attr_name;
        AdjList<uint64_t>* adj_lists;
        bool is_string;
    };
    std::vector<ResolvedProperty> _resolved_properties;

    // Per-property compiled predicates (indexed by position in _resolved_properties)
    // We store per-attribute ScalarPredicate vectors so we can evaluate the full
    // PredicateExpression by walking groups and looking up per-attribute evaluators.
    struct CompiledPredicate {
        bool is_string;
        ScalarPredicate<uint64_t> numeric;
        ScalarPredicate<ffx_str_t> string;
    };

    // For each group in _predicate_expr_raw, for each predicate in the group,
    // we store a compiled evaluator. Indexed as _compiled[group_idx][pred_idx].
    std::vector<std::vector<CompiledPredicate>> _compiled;

    // For groups with branches (nested AND within OR):
    // _compiled_branches[group_idx][branch_idx][pred_idx]
    std::vector<std::vector<std::vector<CompiledPredicate>>> _compiled_branches;

    // Map from property attribute name to index in _resolved_properties
    std::unordered_map<std::string, size_t> _prop_idx_map;

    // Bitmask for tracking valid positions
    std::unique_ptr<BitMask<State::MAX_VECTOR_SIZE>> _valid_mask_uptr;
    BitMask<State::MAX_VECTOR_SIZE> _selector_backup;

    // Ftree state update infrastructure
    std::unique_ptr<FtreeStateUpdateNode> _range_update_tree;
    std::unique_ptr<PropertyFilterSavedData[]> _vector_saved_data;
    std::size_t _vector_saved_data_count = 0;

    // Cascade propagation
    std::unique_ptr<uint32_t[]> _invalidated_indices;
    int32_t _invalidated_count = 0;

    bool evaluate_predicate_for_entity(uint64_t entity_val) const;
};

}// namespace ffx

#endif// VFENGINE_PROPERTY_FILTER_OPERATOR_HH
