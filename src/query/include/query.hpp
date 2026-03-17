#ifndef VFENGINE_QUERY_HPP
#define VFENGINE_QUERY_HPP

#include "predicate.hpp"
#include "query_rule_parser.hpp"
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <unordered_map>

namespace ffx {

/**
 * QueryRelation represents one relation atom in the query graph.
 * 
 * Datalog format: "TableName(a, b)" where a and b are positional aliases
 *    - Position 0 maps to fromVariable
 *    - Position 1 maps to toVariable
 *    - "_" means wildcard/don't care (skipped)
 */
struct QueryRelation {
    QueryRelation() = default;
    
    explicit QueryRelation(std::string fromVariable, std::string toVariable)
        : fromVariable(std::move(fromVariable)), toVariable(std::move(toVariable)), tableName("") {}
    
    // Datalog constructor (with table name)
    QueryRelation(std::string tableName, std::string fromVariable, std::string toVariable)
        : fromVariable(std::move(fromVariable)), toVariable(std::move(toVariable)), 
          tableName(std::move(tableName)) {}
    
    std::string fromVariable;   // Position 0 attribute (source)
    std::string toVariable;     // Position 1 attribute (destination)
    std::string tableName;      // Table name (empty only for internal/test use)

    bool isFwd(const std::string& joinKey, const std::string& outputKey) const {
        return fromVariable == joinKey && toVariable == outputKey;
    }
    
    bool hasTableName() const { return !tableName.empty(); }
};

class Query {
public:
    Query() = delete;
    Query(const Query&) = delete;
    Query(Query&&) = delete;
    Query& operator=(const Query&) = delete;
    Query& operator=(Query&&) = delete;

    /** Rule syntax only: Q(head) := body [WHERE ...] (see query_rule_parser.cpp). */
    explicit Query(const std::string& query_as_str);

    std::set<std::string> get_unique_query_variables() const;
    QueryRelation* get_query_relation(const std::string& var1, const std::string& var2) const;
    
    // Get table name for a relation
    std::string get_table_name(const std::string& var1, const std::string& var2) const;
    
    // Check if query uses Datalog format (has table names)
    bool is_datalog_format() const { return _is_datalog_format; }
    
    // Predicate accessors
    const PredicateExpression& get_predicates() const { return _predicates; }
    bool has_predicates() const { return _predicates.has_predicates(); }
    bool has_scalar_predicates_for(const std::string& attr) const {
        return _predicates.has_scalar_predicates_for(attr);
    }
    bool has_attribute_predicates() const {
        return _predicates.has_attribute_predicates();
    }

    // LLM_MAP accessors (only populated for Q(...) := ... rules with LLM_MAP in the body)
    bool has_llm_map() const { return !_llm_map_config.empty() && !_llm_map_output_attr.empty(); }
    const std::string& get_llm_map_config() const { return _llm_map_config; }
    const std::string& get_llm_map_output_attr() const { return _llm_map_output_attr; }

    /// True when input used `Q(...) := ...` rule syntax.
    bool is_rule_syntax() const { return _is_rule_syntax; }
    QueryHeadKind head_kind() const { return _head_kind; }
    const std::vector<std::string>& head_attributes() const { return _head_attributes; }
    /// Rule head is Q(MIN(...)) — pair with sink `min`; `evaluate_query` rejects `min` without this head.
    bool requires_min_sink() const { return _is_rule_syntax && _head_kind == QueryHeadKind::Min; }

    uint64_t num_rels;
    QueryRelation* rels;

private:
    std::unique_ptr<QueryRelation[]> _rels;
    std::unordered_map<std::string, QueryRelation*> _relation_lookup;
    PredicateExpression _predicates;
    bool _is_datalog_format = false;
    std::string _llm_map_config;
    std::string _llm_map_output_attr;
    bool _is_rule_syntax = false;
    QueryHeadKind _head_kind = QueryHeadKind::Projection;
    std::vector<std::string> _head_attributes;
    
    void parse_relations(const std::string& relations_str);
    void parse_datalog_relation(const std::string& relation_str, QueryRelation& out_rel);
    void parse_predicates(const std::string& predicates_str);
    void build_lookup_maps();
};

}// namespace ffx

#endif// VFENGINE_QUERY_HPP
