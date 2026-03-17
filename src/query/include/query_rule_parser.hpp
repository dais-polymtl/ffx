#ifndef FFX_QUERY_RULE_PARSER_HPP
#define FFX_QUERY_RULE_PARSER_HPP

#include <string>
#include <vector>

namespace ffx {

enum class QueryHeadKind { Projection, Min, CountStar, Noop };

/** Result of parsing Q(head) := body (optional WHERE, optional LLM_MAP assignment). */
struct ParsedQueryRule {
    QueryHeadKind head_kind = QueryHeadKind::Projection;
    std::vector<std::string> head_attributes;
    /// Comma-separated datalog atoms only (no WHERE, no LLM assign).
    std::string relational_body;
    std::string where_clause;
    /// Empty if no `name = LLM_MAP(...)` in body.
    std::string llm_assign_name;
    /// JSON inside LLM_MAP(...); empty if no LLM clause.
    std::string llm_json;
};

/**
 * Parse and validate a rule-shaped query. Throws std::invalid_argument on error.
 */
ParsedQueryRule parse_and_validate_query_rule(const std::string& input);

/// True if trimmed input is rule-shaped (starts with Q after whitespace and contains :=).
bool looks_like_query_rule(const std::string& input);

} // namespace ffx

#endif
