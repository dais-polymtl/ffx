#ifndef VFENGINE_SINK_MIN_ITR_OPERATOR_HH
#define VFENGINE_SINK_MIN_ITR_OPERATOR_HH

#include "factorized_ftree/ftree_iterator.hpp"
#include "operator.hpp"
#include "../../../query/include/predicate.hpp"
#include "../../../table/include/adj_list.hpp"
#include <re2/re2.h>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace ffx {

class StringDictionary;// Forward declaration

class SinkMinItr final : public Operator {
public:
    SinkMinItr() : Operator(), _num_output_tuples(0), _min_values(nullptr), _min_values_size(0), _dictionary(nullptr) {}

    void init(Schema* schema) override;
    void execute() override;
    uint64_t get_num_output_tuples() override { return _num_output_tuples; }

    // Get pointer to the local min_values array for merging
    const uint64_t* get_min_values() const { return _min_values; }
    size_t get_min_values_size() const { return _min_values_size; }

private:
    struct CompiledPredicate {
        PredicateOp op = PredicateOp::EQ;
        size_t tuple_idx = 0;
        bool is_string = false;
        uint64_t value_num = 0;
        uint64_t value_num2 = 0;
        std::vector<uint64_t> in_values_num;
        std::string value_str;
        std::string value_str2;
        std::vector<std::string> in_values_str;
        std::shared_ptr<re2::RE2> like_regex;
    };

    struct CompiledPredicateGroup {
        LogicalOp op = LogicalOp::AND;
        std::vector<CompiledPredicate> predicates;
    };

    bool evaluate_final_predicates(const uint64_t* tuple_values) const;
    bool evaluate_group(const CompiledPredicateGroup& group, const uint64_t* tuple_values) const;
    bool evaluate_predicate(const CompiledPredicate& pred, const uint64_t* tuple_values) const;

    uint64_t _num_output_tuples;
    std::unique_ptr<FTreeIterator> _iterator;
    uint64_t* _min_values;
    size_t _min_values_size;

    // Pre-allocated buffer for reading tuples
    // std::unique_ptr<uint64_t[]> _tmp_buffer;
    size_t _tuple_size;

    // String attribute support
    StringDictionary* _dictionary;
    std::vector<bool> _is_string_attr;
    std::vector<size_t> _mapped_indices;
    bool _has_final_predicates = false;
    LogicalOp _final_top_level_op = LogicalOp::AND;
    std::vector<CompiledPredicateGroup> _compiled_groups;

    // Property attribute support (for min attrs not in column ordering)
    struct PropertyMinAttr {
        size_t entity_col_idx;      // Index of entity attr in column ordering
        AdjList<uint64_t>* adj_lists;// Adjacency list for property lookup
        bool is_string;             // Whether property values are strings
    };
    // For each min_attr position: -1 if direct column, or index into _property_min_attrs
    std::vector<int> _min_attr_type;          // -1 = direct, >=0 = property index
    std::vector<PropertyMinAttr> _property_min_attrs;
    bool _has_property_attrs = false;
};

}// namespace ffx

#endif
