#ifndef VFENGINE_PREDICATE_HPP
#define VFENGINE_PREDICATE_HPP

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace ffx {

// Comparison operators for predicates
enum class PredicateOp : uint8_t {
    EQ,      // ==
    NEQ,     // !=
    LT,      // <
    GT,      // >
    LTE,     // <=
    GTE,     // >=
    IN,      // IN (...)
    NOT_IN,  // NOT IN (...)
    IS_NULL, // value is NULL
    IS_NOT_NULL, // value is NOT NULL
    LIKE,    // LIKE pattern (supports % and _)
    NOT_LIKE,// NOT LIKE pattern
    BETWEEN  // BETWEEN min AND max
};

// Logical operators for combining predicates
enum class LogicalOp : uint8_t {
    AND,// All predicates must be true
    OR  // At least one predicate must be true
};

// Type of predicate
enum class PredicateType : uint8_t {
    SCALAR,   // Compare to constant value
    ATTRIBUTE,// Compare to another attribute
    NOT_EXISTS// Anti-semi-join: NOT (a)-[edge]->(b)
};

// A single predicate
struct Predicate {
    PredicateOp op;
    PredicateType type;
    std::string left_attr;// First attribute (always an attribute name)

    // For SCALAR type
    std::string scalar_value;              // The constant value (as string) for non-list scalars
    std::string scalar_value2;             // Second constant value for BETWEEN
    std::vector<std::string> scalar_values;// For IN(...) list values (as strings)

    // For ATTRIBUTE type
    std::string right_attr;// Second attribute name

    // For NOT_EXISTS type
    std::string edge_table; // The edge table to check
    bool is_edge_fwd = true;// Direction of the edge

    Predicate() = default;

    // Constructor for scalar predicate
    Predicate(PredicateOp op, std::string left, std::string scalar)
        : op(op), type(PredicateType::SCALAR), left_attr(std::move(left)), scalar_value(std::move(scalar)) {}

    // Constructor for BETWEEN scalar predicate
    Predicate(PredicateOp op, std::string left, std::string scalar, std::string scalar2)
        : op(op), type(PredicateType::SCALAR), left_attr(std::move(left)), scalar_value(std::move(scalar)),
          scalar_value2(std::move(scalar2)) {
        if (op != PredicateOp::BETWEEN) { throw std::invalid_argument("BETWEEN constructor used with non-BETWEEN op"); }
    }

    // Constructor for attribute predicate (e.g. attr1 = attr2)
    static Predicate Attribute(PredicateOp op, std::string left, std::string right) {
        Predicate p;
        p.op = op;
        p.type = PredicateType::ATTRIBUTE;
        p.left_attr = std::move(left);
        p.right_attr = std::move(right);
        return p;
    }

    // Constructor for NOT EXISTS predicate
    Predicate(std::string left, std::string right, std::string edge, bool edge_fwd = true)
        : op(PredicateOp::NEQ), type(PredicateType::NOT_EXISTS), left_attr(std::move(left)),
          right_attr(std::move(right)), edge_table(std::move(edge)), is_edge_fwd(edge_fwd) {}

    bool is_scalar() const { return type == PredicateType::SCALAR; }
    bool is_attribute() const { return type == PredicateType::ATTRIBUTE; }
    bool is_not_exists() const { return type == PredicateType::NOT_EXISTS; }
};

// A group of predicates combined with a logical operator
struct PredicateGroup {
    LogicalOp op = LogicalOp::AND;
    std::vector<Predicate> predicates;
    // For OR groups where some branches are multi-predicate AND expressions.
    // E.g., (EQ(a,1) OR (EQ(b,2) AND EQ(c,3))) has two branches:
    //   branch 0: AND [EQ(a,1)]
    //   branch 1: AND [EQ(b,2), EQ(c,3)]
    // When non-empty, branches are used instead of predicates.
    std::vector<PredicateGroup> branches;

    bool empty() const { return predicates.empty() && branches.empty(); }
    size_t size() const { return branches.empty() ? predicates.size() : branches.size(); }
    bool has_branches() const { return !branches.empty(); }
};

// Utility functions (declared before PredicateExpression so to_string() can use them)
inline const char* predicate_op_to_string(PredicateOp op) {
    switch (op) {
        case PredicateOp::EQ:
            return "EQ";
        case PredicateOp::NEQ:
            return "NEQ";
        case PredicateOp::LT:
            return "LT";
        case PredicateOp::GT:
            return "GT";
        case PredicateOp::LTE:
            return "LTE";
        case PredicateOp::GTE:
            return "GTE";
        case PredicateOp::IN:
            return "IN";
        case PredicateOp::NOT_IN:
            return "NOT_IN";
        case PredicateOp::IS_NULL:
            return "IS_NULL";
        case PredicateOp::IS_NOT_NULL:
            return "IS_NOT_NULL";
        case PredicateOp::LIKE:
            return "LIKE";
        case PredicateOp::NOT_LIKE:
            return "NOT_LIKE";
        case PredicateOp::BETWEEN:
            return "BETWEEN";
    }
    return "UNKNOWN";
}

inline const char* logical_op_to_string(LogicalOp op) {
    switch (op) {
        case LogicalOp::AND:
            return "AND";
        case LogicalOp::OR:
            return "OR";
    }
    return "UNKNOWN";
}

// Top-level predicate expression (supports nested AND/OR)
// Structure: top_level_op applied to groups
// Example: "(A AND B) OR (C AND D)" has top_level_op=OR, with two AND groups
struct PredicateExpression {
    LogicalOp top_level_op = LogicalOp::AND;
    std::vector<PredicateGroup> groups;

    bool empty() const { return groups.empty(); }
    bool has_predicates() const {
        for (const auto& g: groups) {
            if (!g.empty()) return true;
        }
        return false;
    }

    // Get all predicates (flattened)
    std::vector<Predicate> all_predicates() const {
        std::vector<Predicate> result;
        for (const auto& g: groups) {
            for (const auto& p: g.predicates) {
                result.push_back(p);
            }
        }
        return result;
    }

    // Get all scalar predicates for a specific attribute
    std::vector<const Predicate*> get_scalar_predicates_for(const std::string& attr) const {
        std::vector<const Predicate*> result;
        for (const auto& g: groups) {
            for (const auto& p: g.predicates) {
                if (p.is_scalar() && p.left_attr == attr) { result.push_back(&p); }
            }
        }
        return result;
    }

    // Get all attribute predicates
    std::vector<const Predicate*> get_attribute_predicates() const {
        std::vector<const Predicate*> result;
        for (const auto& g: groups) {
            for (const auto& p: g.predicates) {
                if (p.is_attribute()) { result.push_back(&p); }
            }
        }
        return result;
    }

    // Check if attribute has any scalar predicates
    bool has_scalar_predicates_for(const std::string& attr) const { return !get_scalar_predicates_for(attr).empty(); }

    // Check if there are any attribute predicates
    bool has_attribute_predicates() const { return !get_attribute_predicates().empty(); }

    // Get all NOT EXISTS predicates
    std::vector<const Predicate*> get_not_exists_predicates() const {
        std::vector<const Predicate*> result;
        for (const auto& g: groups) {
            for (const auto& p: g.predicates) {
                if (p.is_not_exists()) { result.push_back(&p); }
            }
        }
        return result;
    }

    // Check if there are any NOT EXISTS predicates
    bool has_not_exists_predicates() const { return !get_not_exists_predicates().empty(); }

    // String representation of predicates
    std::string to_string() const {
        if (groups.empty()) return "";

        std::string result;
        bool first_group = true;
        for (const auto& g: groups) {
            if (g.empty()) continue;

            if (!first_group) { result += (top_level_op == LogicalOp::AND) ? " AND " : " OR "; }
            first_group = false;

            auto format_pred = [](const Predicate& p) -> std::string {
                std::string s = std::string(predicate_op_to_string(p.op)) + "(" + p.left_attr;
                if (p.is_scalar()) {
                    switch (p.op) {
                        case PredicateOp::IN:
                        case PredicateOp::NOT_IN:
                            for (const auto& v: p.scalar_values) { s += ", " + v; }
                            break;
                        case PredicateOp::BETWEEN:
                            s += ", " + p.scalar_value + ", " + p.scalar_value2;
                            break;
                        case PredicateOp::IS_NULL:
                        case PredicateOp::IS_NOT_NULL:
                            break;
                        default:
                            s += ", " + p.scalar_value;
                            break;
                    }
                } else if (p.is_attribute()) {
                    s += ", " + p.right_attr;
                }
                s += ")";
                return s;
            };

            if (g.has_branches()) {
                // OR group with AND sub-branches
                result += "(";
                for (size_t bi = 0; bi < g.branches.size(); ++bi) {
                    if (bi > 0) result += " OR ";
                    const auto& branch = g.branches[bi];
                    if (branch.predicates.size() > 1) result += "(";
                    for (size_t pi = 0; pi < branch.predicates.size(); ++pi) {
                        if (pi > 0) result += " AND ";
                        result += format_pred(branch.predicates[pi]);
                    }
                    if (branch.predicates.size() > 1) result += ")";
                }
                result += ")";
            } else {
                if (g.predicates.size() > 1) result += "(";
                bool first_pred = true;
                for (const auto& p: g.predicates) {
                    if (!first_pred) { result += (g.op == LogicalOp::AND) ? " AND " : " OR "; }
                    first_pred = false;
                    result += format_pred(p);
                }
                if (g.predicates.size() > 1) result += ")";
            }
        }
        return result;
    }
};

// Utility functions are defined above, before PredicateExpression

}// namespace ffx

#endif// VFENGINE_PREDICATE_HPP
