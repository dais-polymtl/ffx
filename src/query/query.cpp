#include "query.hpp"
#include "predicate_parser.hpp"
#include "query_rule_parser.hpp"
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace ffx {

static std::vector<std::string> split(const std::string& str, const std::string& delimiter);
static std::string make_key(const std::string& var1, const std::string& var2);
static std::string trim(const std::string& str);

Query::Query(const std::string& query_as_str) {
    if (!looks_like_query_rule(query_as_str)) {
        throw std::invalid_argument(
                "Query: expected Q(head) := body (optional WHERE ...), e.g. Q(a,b) := R(a,b),R(b,c).");
    }

    ParsedQueryRule rule = parse_and_validate_query_rule(query_as_str);
    _is_rule_syntax = true;
    _head_kind = rule.head_kind;
    _head_attributes = std::move(rule.head_attributes);
    if (!rule.llm_json.empty()) {
        if (rule.llm_assign_name.empty()) {
            throw std::invalid_argument(
                    "LLM_MAP requires 'name = LLM_MAP({...})' with a non-empty name before '='.");
        }
        _llm_map_config = std::move(rule.llm_json);
        _llm_map_output_attr = std::move(rule.llm_assign_name);
    }
    parse_relations(trim(rule.relational_body));
    parse_predicates(trim(rule.where_clause));
    build_lookup_maps();
}

void Query::parse_relations(const std::string& relations_str) {
    _is_datalog_format = true;

    // Split by comma, but handle parentheses properly so commas inside Table(a,b) are preserved
    std::vector<std::string> query_relations;
    int paren_depth = 0;
    std::string current;
    for (char c : relations_str) {
        if (c == '(') {
            paren_depth++;
            current += c;
        } else if (c == ')') {
            paren_depth--;
            current += c;
        } else if (c == ',' && paren_depth == 0) {
            query_relations.push_back(trim(current));
            current.clear();
        } else {
            current += c;
        }
    }
    if (!current.empty()) {
        query_relations.push_back(trim(current));
    }

    num_rels = query_relations.size();
    _rels = std::make_unique<QueryRelation[]>(query_relations.size());
    rels = _rels.get();

    for (size_t i = 0; i < query_relations.size(); i++) {
        std::string rel_str = trim(query_relations[i]);
        parse_datalog_relation(rel_str, rels[i]);
    }
}

void Query::parse_datalog_relation(const std::string& relation_str, QueryRelation& out_rel) {
    // Format: TableName(attr1, attr2, ...)
    // attr1 = fromVariable (position 0)
    // attr2 = toVariable (position 1)
    // Additional attrs are ignored for now
    // "_" means wildcard/don't care

    size_t paren_open = relation_str.find('(');
    size_t paren_close = relation_str.find(')');

    if (paren_open == std::string::npos || paren_close == std::string::npos) {
        throw std::runtime_error("Invalid Datalog relation format (missing parentheses): " + relation_str);
    }

    std::string table_name = trim(relation_str.substr(0, paren_open));
    std::string attrs_str = relation_str.substr(paren_open + 1, paren_close - paren_open - 1);

    // Split attributes by comma
    std::vector<std::string> attrs = split(attrs_str, ",");

    if (attrs.size() < 2) {
        throw std::runtime_error("Datalog relation must have at least 2 attributes: " + relation_str);
    }

    // Position 0 -> fromVariable, Position 1 -> toVariable (or first non-_ after pos 0 if pos 1 is _)
    std::string attr0 = trim(attrs[0]);
    std::string attr1 = trim(attrs[1]);

    if (attr0 == "_") {
        throw std::runtime_error("First attribute in a relation cannot be a wildcard: " + relation_str);
    }
    // If position 1 is _, use the next non-_ attribute (e.g. AAbs(a,_,aabs) binds a to aabs)
    if (attr1 == "_") {
        attr1.clear();
        for (size_t i = 2; i < attrs.size(); ++i) {
            std::string a = trim(attrs[i]);
            if (a != "_") {
                attr1 = a;
                break;
            }
        }
        if (attr1.empty()) {
            throw std::runtime_error("No valid toVariable when position 1 is wildcard: " + relation_str);
        }
    }

    out_rel.tableName = table_name;
    out_rel.fromVariable = attr0;
    out_rel.toVariable = attr1;

    // Debug output
    std::cout << "Parsed Datalog relation: " << table_name << "(" << attr0 << ", " << attr1 << ")" << std::endl;
}

void Query::parse_predicates(const std::string& predicates_str) {
    if (!predicates_str.empty()) {
        _predicates = PredicateParser::parse_predicates(predicates_str);
    }
}

std::set<std::string> Query::get_unique_query_variables() const {
    std::set<std::string> query_variables;
    for (size_t i = 0; i < num_rels; i++) {
        query_variables.insert(rels[i].fromVariable);
        query_variables.insert(rels[i].toVariable);
    }
    return query_variables;
}

static std::vector<std::string> split(const std::string& str, const std::string& delimiter) {
    std::vector<std::string> result;
    size_t start = 0, end;

    while ((end = str.find(delimiter, start)) != std::string::npos) {
        result.push_back(str.substr(start, end - start));
        start = end + delimiter.length();// Move past the delimiter
    }

    result.push_back(str.substr(start));// Add the last segment
    return result;
}

QueryRelation* Query::get_query_relation(const std::string& var1, const std::string& var2) const {
    std::string key = make_key(var1, var2);
    auto it = _relation_lookup.find(key);
    if (it != _relation_lookup.end()) {
        return it->second;
    }
    return nullptr; // Or throw an exception if relation must exist
}

std::string Query::get_table_name(const std::string& var1, const std::string& var2) const {
    QueryRelation* rel = get_query_relation(var1, var2);
    if (rel) {
        return rel->tableName;
    }
    return "";
}

void Query::build_lookup_maps() {
    _relation_lookup.clear();
    for (uint64_t i = 0; i < num_rels; ++i) {
        QueryRelation* rel = &rels[i];
        std::string key = make_key(rel->fromVariable, rel->toVariable);
        _relation_lookup[key] = rel;
    }
}

static std::string make_key(const std::string& var1, const std::string& var2) {
    // Create a normalized key by ensuring consistent ordering
    if (var1 < var2) {
        return var1 + "|" + var2;
    } else {
        return var2 + "|" + var1;
    }
}

static std::string trim(const std::string& str) {
    size_t start = str.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    size_t end = str.find_last_not_of(" \t\n\r");
    return str.substr(start, end - start + 1);
}

}// namespace ffx
