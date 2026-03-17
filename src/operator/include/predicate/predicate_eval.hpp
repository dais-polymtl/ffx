#ifndef VFENGINE_PREDICATE_EVAL_HPP
#define VFENGINE_PREDICATE_EVAL_HPP

#include "../../../query/include/like_util.hpp"
#include "../../../query/include/predicate.hpp"
#include "../../../table/include/ffx_str_t.hpp"
#include "../../../table/include/string_dictionary.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <memory>
#include <re2/re2.h>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace ffx {

// Function pointer type for predicate evaluation
template<typename T>
using PredicateFn = bool (*)(const T& value, const T& scalar);

// Predicate implementations for any comparable type
template<typename T>
inline bool pred_eq(const T& a, const T& b) {
    return a == b;
}

template<typename T>
inline bool pred_neq(const T& a, const T& b) {
    return a != b;
}

template<typename T>
inline bool pred_lt(const T& a, const T& b) {
    return a < b;
}

template<typename T>
inline bool pred_gt(const T& a, const T& b) {
    return a > b;
}

template<typename T>
inline bool pred_lte(const T& a, const T& b) {
    return a <= b;
}

template<typename T>
inline bool pred_gte(const T& a, const T& b) {
    return a >= b;
}

// Always true predicate (no filter)
template<typename T>
inline bool pred_true(const T&, const T&) {
    return true;
}

// Compare ffx_str_t with std::string lexicographically
// Returns: <0 if ffx < str, 0 if equal, >0 if ffx > str
inline int ffx_str_compare(const ffx_str_t& ffx, const std::string& str) {
    uint32_t ffx_len = ffx.get_size();
    uint32_t str_len = static_cast<uint32_t>(str.size());
    uint32_t min_len = std::min(ffx_len, str_len);

    // First compare up to INLINE_THRESHOLD (prefix)
    uint32_t prefix_cmp_len = std::min(min_len, ffx_str_t::INLINE_THRESHOLD);
    int cmp = std::memcmp(ffx.prefix, str.data(), prefix_cmp_len);
    if (cmp != 0) return cmp;

    // If tied on prefix and there's more to compare
    if (min_len > ffx_str_t::INLINE_THRESHOLD) {
        // Get full string pointers
        const char* ffx_ptr = (ffx_len > ffx_str_t::INLINE_THRESHOLD) ? ffx.data.ptr : ffx.prefix;
        const char* str_ptr = str.data();

        // Compare remaining bytes after prefix
        cmp = std::memcmp(ffx_ptr + ffx_str_t::INLINE_THRESHOLD, str_ptr + ffx_str_t::INLINE_THRESHOLD,
                          min_len - ffx_str_t::INLINE_THRESHOLD);
        if (cmp != 0) return cmp;
    }

    // If all compared bytes are equal, shorter string is "less"
    if (ffx_len < str_len) return -1;
    if (ffx_len > str_len) return 1;
    return 0;
}

// Equality check: ffx_str_t == std::string
inline bool ffx_str_eq(const ffx_str_t& ffx, const std::string& str) { return ffx_str_compare(ffx, str) == 0; }

// Less than: ffx_str_t < std::string
inline bool ffx_str_lt(const ffx_str_t& ffx, const std::string& str) { return ffx_str_compare(ffx, str) < 0; }

// Greater than: ffx_str_t > std::string
inline bool ffx_str_gt(const ffx_str_t& ffx, const std::string& str) { return ffx_str_compare(ffx, str) > 0; }

// Less than or equal: ffx_str_t <= std::string
inline bool ffx_str_lte(const ffx_str_t& ffx, const std::string& str) { return ffx_str_compare(ffx, str) <= 0; }

// Greater than or equal: ffx_str_t >= std::string
inline bool ffx_str_gte(const ffx_str_t& ffx, const std::string& str) { return ffx_str_compare(ffx, str) >= 0; }

// A single scalar predicate with its function and comparison value
// Base template (unimplemented to catch unsupported types at compile time)
template<typename T>
struct ScalarPredicate;

// Specialization for uint64_t - Pure numeric predicate
template<>
struct ScalarPredicate<uint64_t> {
    PredicateFn<uint64_t> fn = pred_true<uint64_t>;
    uint64_t value = 0;
    uint64_t value2 = 0;             // Second value for BETWEEN
    PredicateOp op = PredicateOp::EQ;// Store for string representation
    std::vector<uint64_t> in_values; // For IN(...) predicates (sorted)

    ScalarPredicate() = default;
    ScalarPredicate(PredicateFn<uint64_t> f, uint64_t v, PredicateOp o = PredicateOp::EQ)
        : fn(f), value(v), value2(0), op(o) {}
    ScalarPredicate(uint64_t v1, uint64_t v2, PredicateOp o = PredicateOp::BETWEEN)
        : fn(nullptr), value(v1), value2(v2), op(o) {}

    bool evaluate(const uint64_t& input) const {
        switch (op) {
            case PredicateOp::IN:
                return std::find(in_values.begin(), in_values.end(), input) != in_values.end();
            case PredicateOp::NOT_IN:
                return std::find(in_values.begin(), in_values.end(), input) == in_values.end();
            case PredicateOp::IS_NULL:
                return input == std::numeric_limits<uint64_t>::max();
            case PredicateOp::IS_NOT_NULL:
                return input != std::numeric_limits<uint64_t>::max();
            case PredicateOp::BETWEEN:
                return input >= value && input <= value2;
            case PredicateOp::LIKE:
                return fn(input, value);
            case PredicateOp::NOT_LIKE:
                return !fn(input, value);
            default:
                return fn(input, value);
        }
    }
};

// Specialization for ffx_str_t - String dictionary predicate
template<>
struct ScalarPredicate<ffx_str_t> {
    PredicateFn<ffx_str_t> fn = pred_true<ffx_str_t>;
    ffx_str_t value;                         // Reference value (constructed via pool)
    ffx_str_t value2;                        // Second value for BETWEEN
    PredicateOp op = PredicateOp::EQ;        // Store for string representation
    std::vector<ffx_str_t> in_values;        // For IN(...) predicates (constructed via pool)
    std::shared_ptr<re2::RE2> compiled_regex;// Pre-compiled regex for LIKE

    StringDictionary* dictionary = nullptr;
    StringPool* pool = nullptr;

    ScalarPredicate() = default;
    ScalarPredicate(PredicateFn<ffx_str_t> f, ffx_str_t v, PredicateOp o = PredicateOp::EQ)
        : fn(f), value(std::move(v)), op(o) {}
    ScalarPredicate(ffx_str_t v1, ffx_str_t v2, PredicateOp o = PredicateOp::BETWEEN)
        : fn(nullptr), value(std::move(v1)), value2(std::move(v2)), op(o) {}

    // Evaluation via dictionary ID (operators always pass IDs)
    bool evaluate_id(uint64_t dict_id) const {
        assert(dictionary != nullptr && "String predicate requires dictionary");

        // Handle NULL values (UINT64_MAX represents NULL string)
        if (dict_id == UINT64_MAX) {
            if (op == PredicateOp::IS_NULL) return true;
            if (op == PredicateOp::IS_NOT_NULL) return false;
            return false;
        }

        // Validate ID before lookup
        assert(dictionary->has_id(dict_id));

        const ffx_str_t& input = dictionary->get_string(dict_id);
        return evaluate(input);
    }

    // Direct string evaluation (used by specialization and tests)
    bool evaluate(const ffx_str_t& input) const;
};

// Define evaluate outside to avoid circular dependencies if any, but MUST NOT use template<> for member of specialization
inline bool ScalarPredicate<ffx_str_t>::evaluate(const ffx_str_t& input) const {
    if (op == PredicateOp::IN) { return std::find(in_values.begin(), in_values.end(), input) != in_values.end(); }
    if (op == PredicateOp::NOT_IN) { return std::find(in_values.begin(), in_values.end(), input) == in_values.end(); }
    if (op == PredicateOp::IS_NULL) { return input.is_null(); }
    if (op == PredicateOp::IS_NOT_NULL) { return !input.is_null(); }
    if (op == PredicateOp::BETWEEN) { return input >= value && input <= value2; }
    if (op == PredicateOp::LIKE) {
        // Optimized LIKE using RE2
        assert(compiled_regex != nullptr);

        uint32_t s_len = input.get_size();
        const char* s_ptr;
        if (s_len <= ffx_str_t::INLINE_THRESHOLD) {
            s_ptr = input.prefix;
        } else {
            assert(input.data.ptr != nullptr);
            s_ptr = input.data.ptr;
        }

        return re2::RE2::FullMatch(re2::StringPiece(s_ptr, s_len), *compiled_regex);
    }
    if (op == PredicateOp::NOT_LIKE) {
        // Optimized NOT LIKE using RE2
        assert(compiled_regex != nullptr);

        uint32_t s_len = input.get_size();
        const char* s_ptr;
        if (s_len <= ffx_str_t::INLINE_THRESHOLD) {
            s_ptr = input.prefix;
        } else {
            assert(input.data.ptr != nullptr);
            s_ptr = input.data.ptr;
        }

        return !re2::RE2::FullMatch(re2::StringPiece(s_ptr, s_len), *compiled_regex);
    }
    // Handle comparison operators explicitly using ffx_str_t's comparison operators
    // This ensures proper lexicographical comparison even when fn is pred_true
    switch (op) {
        case PredicateOp::EQ:
            return input == value;
        case PredicateOp::NEQ:
            return input != value;
        case PredicateOp::LT:
            return input < value;
        case PredicateOp::GT:
            return input > value;
        case PredicateOp::LTE:
            return input <= value;
        case PredicateOp::GTE:
            return input >= value;
        default:
            return fn(input, value);
    }
}

// Group of scalar predicates combined with AND or OR
template<typename T>
struct ScalarPredicateGroup {
    LogicalOp op = LogicalOp::AND;
    std::vector<ScalarPredicate<T>> predicates;

    bool empty() const { return predicates.empty(); }

    bool evaluate(const T& value) const {
        if (predicates.empty()) return true;

        if (op == LogicalOp::AND) {
            // All must pass (short-circuit on first failure)
            for (const auto& p: predicates) {
                if (!p.evaluate(value)) return false;
            }
            return true;
        } else {// OR
            // At least one must pass (short-circuit on first success)
            for (const auto& p: predicates) {
                if (p.evaluate(value)) return true;
            }
            return false;
        }
    }

    // Specialization for uint64_t dictionary ID evaluation
    bool evaluate_id(uint64_t) const {
        assert(false && "evaluate_id only supported for string predicates");
        return false;
    }
};

template<>
inline bool ScalarPredicateGroup<ffx_str_t>::evaluate_id(uint64_t dict_id) const {
    if (predicates.empty()) return true;
    if (op == LogicalOp::AND) {
        for (const auto& p: predicates) {
            if (!p.evaluate_id(dict_id)) return false;
        }
        return true;
    } else {
        for (const auto& p: predicates) {
            if (p.evaluate_id(dict_id)) return true;
        }
        return false;
    }
}

// Top-level predicate expression for an attribute
// Supports: (A AND B) OR (C AND D) structure
template<typename T>
struct ScalarPredicateExpression {
    LogicalOp top_level_op = LogicalOp::AND;
    std::vector<ScalarPredicateGroup<T>> groups;

    bool empty() const { return groups.empty(); }
    bool has_predicates() const {
        for (const auto& g: groups) {
            if (!g.empty()) return true;
        }
        return false;
    }

    bool evaluate(const T& value) const {
        if (groups.empty()) return true;

        if (top_level_op == LogicalOp::AND) {
            // All groups must pass
            for (const auto& g: groups) {
                if (!g.evaluate(value)) return false;
            }
            return true;
        } else {// OR
            // At least one group must pass
            for (const auto& g: groups) {
                if (g.evaluate(value)) return true;
            }
            return false;
        }
    }

    // Specialization for uint64_t dictionary ID evaluation
    bool evaluate_id(uint64_t) const {
        assert(false && "evaluate_id only supported for string predicates");
        return false;
    }

    std::string to_string() const {
        if (groups.empty()) return "";

        std::string result;
        bool first_group = true;
        for (const auto& g: groups) {
            if (g.empty()) continue;

            if (!first_group) { result += (top_level_op == LogicalOp::AND) ? " AND " : " OR "; }
            first_group = false;

            if (g.predicates.size() > 1) result += "(";
            bool first_pred = true;
            for (const auto& p: g.predicates) {
                if (!first_pred) { result += (g.op == LogicalOp::AND) ? " AND " : " OR "; }
                first_pred = false;
                result += std::string(predicate_op_to_string(p.op)) + "(";
                auto append_scalar = [&](const T& v) {
                    if constexpr (std::is_same_v<T, ffx_str_t>) {
                        result += v.to_string();
                    } else {
                        result += std::to_string(v);
                    }
                };

                switch (p.op) {
                    case PredicateOp::IN:
                    case PredicateOp::NOT_IN:
                        for (size_t i = 0; i < p.in_values.size(); ++i) {
                            if (i > 0) result += ", ";
                            append_scalar(p.in_values[i]);
                        }
                        break;
                    case PredicateOp::IS_NULL:
                    case PredicateOp::IS_NOT_NULL:
                        break;
                    case PredicateOp::BETWEEN:
                        append_scalar(p.value);
                        result += ", ";
                        append_scalar(p.value2);
                        break;
                    default:
                        append_scalar(p.value);
                        break;
                }
                result += ")";
            }
            if (g.predicates.size() > 1) result += ")";
        }
        return result;
    }
};

template<>
inline bool ScalarPredicateExpression<ffx_str_t>::evaluate_id(uint64_t dict_id) const {
    if (groups.empty()) return true;
    if (top_level_op == LogicalOp::AND) {
        for (const auto& g: groups) {
            if (!g.evaluate_id(dict_id)) return false;
        }
        return true;
    } else {
        for (const auto& g: groups) {
            if (g.evaluate_id(dict_id)) return true;
        }
        return false;
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

// Get predicate function from PredicateOp enum
template<typename T>
inline PredicateFn<T> get_predicate_fn(PredicateOp op) {
    switch (op) {
        case PredicateOp::EQ:
            return pred_eq<T>;
        case PredicateOp::NEQ:
            return pred_neq<T>;
        case PredicateOp::LT:
            return pred_lt<T>;
        case PredicateOp::GT:
            return pred_gt<T>;
        case PredicateOp::LTE:
            return pred_lte<T>;
        case PredicateOp::GTE:
            return pred_gte<T>;
        case PredicateOp::IN:
        case PredicateOp::NOT_IN:
        case PredicateOp::IS_NULL:
        case PredicateOp::IS_NOT_NULL:
            return pred_true<T>;// handled in ScalarPredicate::evaluate
        case PredicateOp::LIKE:
        case PredicateOp::NOT_LIKE:
            return pred_true<T>;// handled via specialization / custom evaluation
        case PredicateOp::BETWEEN:
            return pred_true<T>;// handled in ScalarPredicate::evaluate
    }
    return pred_true<T>;// Default: no filter
}

// Parse scalar value from string
template<typename T>
inline T parse_scalar_value(const std::string& str, StringPool* pool);

template<>
inline uint64_t parse_scalar_value<uint64_t>(const std::string& str, StringPool*) {
    return std::stoull(str);
}

template<>
inline int64_t parse_scalar_value<int64_t>(const std::string& str, StringPool*) {
    return std::stoll(str);
}

template<>
inline double parse_scalar_value<double>(const std::string& str, StringPool*) {
    return std::stod(str);
}

template<>
inline ffx_str_t parse_scalar_value<ffx_str_t>(const std::string& str, StringPool* pool) {
    assert(pool != nullptr);
    return ffx_str_t(str, pool);
}

// Build ScalarPredicateExpression from PredicateExpression for a specific attribute
template<typename T>
inline ScalarPredicateExpression<T> build_scalar_predicate_expr(const PredicateExpression& expr,
                                                                const std::string& attr, StringPool* pool,
                                                                StringDictionary* dict = nullptr) {
    ScalarPredicateExpression<T> result;
    result.top_level_op = expr.top_level_op;

    for (const auto& group: expr.groups) {
        ScalarPredicateGroup<T> scalar_group;
        scalar_group.op = group.op;

        for (const auto& pred: group.predicates) {
            // Only include scalar predicates for this attribute
            if (pred.is_scalar() && pred.left_attr == attr) {
                ScalarPredicate<T> sp;
                sp.op = pred.op;

                if constexpr (std::is_same_v<T, uint64_t>) {
                    // uint64_t specialization: purely numeric
                    switch (pred.op) {
                        case PredicateOp::IN:
                        case PredicateOp::NOT_IN:
                            sp.in_values.reserve(pred.scalar_values.size());
                            for (const auto& v: pred.scalar_values) {
                                sp.in_values.push_back(parse_scalar_value<T>(v, pool));
                            }
                            break;
                        case PredicateOp::IS_NULL:
                        case PredicateOp::IS_NOT_NULL:
                            // Unary predicate; value fields are unused.
                            break;
                        case PredicateOp::BETWEEN:
                            sp.value = parse_scalar_value<T>(pred.scalar_value, pool);
                            sp.value2 = parse_scalar_value<T>(pred.scalar_value2, pool);
                            break;
                        default:
                            sp.value = parse_scalar_value<T>(pred.scalar_value, pool);
                            sp.fn = get_predicate_fn<T>(pred.op);
                            break;
                    }
                } else if constexpr (std::is_same_v<T, ffx_str_t>) {
                    // ffx_str_t specialization: dictionary-based string predicate
                    sp.dictionary = dict;
                    sp.pool = pool;

                    switch (pred.op) {
                        case PredicateOp::IN:
                        case PredicateOp::NOT_IN:
                            sp.in_values.reserve(pred.scalar_values.size());
                            for (const auto& v: pred.scalar_values) {
                                sp.in_values.push_back(ffx_str_t(v, pool));
                            }
                            break;
                        case PredicateOp::IS_NULL:
                        case PredicateOp::IS_NOT_NULL:
                            // Unary predicate; value fields are unused.
                            break;
                        case PredicateOp::BETWEEN:
                            sp.value = ffx_str_t(pred.scalar_value, pool);
                            sp.value2 = ffx_str_t(pred.scalar_value2, pool);
                            break;
                        case PredicateOp::LIKE:
                        case PredicateOp::NOT_LIKE:
                            sp.value = ffx_str_t(pred.scalar_value, pool);
                            sp.compiled_regex = std::make_shared<re2::RE2>(sql_like_to_regex(pred.scalar_value));
                            if (!sp.compiled_regex->ok()) {
                                throw std::runtime_error("Failed to compile LIKE regex: " + pred.scalar_value);
                            }
                            break;
                        default:
                            sp.value = ffx_str_t(pred.scalar_value, pool);
                            sp.fn = get_predicate_fn<T>(pred.op);
                            break;
                    }
                }

                // IMPORTANT: move, don't copy.
                scalar_group.predicates.push_back(std::move(sp));
            }
        }

        // Preserve empty groups for OR expressions.
        // Example: (A) OR (B AND C) projected to attribute C should become
        // (TRUE) OR (C), not just (C), otherwise semantics become too restrictive.
        if (!scalar_group.empty() || expr.top_level_op == LogicalOp::OR) {
            result.groups.push_back(std::move(scalar_group));
        }
    }

    return result;
}

// Check if an expression has scalar predicates for a specific attribute
inline bool has_scalar_predicates_for(const PredicateExpression& expr, const std::string& attr) {
    for (const auto& group: expr.groups) {
        for (const auto& pred: group.predicates) {
            if (pred.is_scalar() && pred.left_attr == attr) { return true; }
        }
    }
    return false;
}

}// namespace ffx

#endif// VFENGINE_PREDICATE_EVAL_HPP
