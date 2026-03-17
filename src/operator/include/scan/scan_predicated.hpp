#ifndef VFENGINE_SCAN_PREDICATED_OPERATOR_HH
#define VFENGINE_SCAN_PREDICATED_OPERATOR_HH

#include "../predicate/predicate_eval.hpp"
#include "operator.hpp"
#include <string>
#include <vector>

namespace ffx {

/**
 * ScanPredicated - Scan operator with scalar predicate support
 * 
 * Scans values from 0 to max_id and applies a predicate filter.
 * Only values that pass the predicate are included in the output.
 * 
 * Example: ScanPredicated with GTE(a,10) AND LT(a,100) will only
 * output values in range [10, 100).
 */
template<typename T = uint64_t>
class ScanPredicated final : public Operator {
public:
    ScanPredicated() = delete;
    ScanPredicated(const ScanPredicated&) = delete;

    // Constructor with predicate expression
    explicit ScanPredicated(std::string attribute, PredicateExpression predicate_expr)
        : _attribute(std::move(attribute)), _max_id_value(State::MAX_VECTOR_SIZE - 1), _output_vector(nullptr),
          _output_selection_mask(nullptr), _predicate_expr_raw(std::move(predicate_expr)) {}

    // Constructor without predicate (behaves like regular Scan)
    explicit ScanPredicated(std::string attribute)
        : _attribute(std::move(attribute)), _max_id_value(State::MAX_VECTOR_SIZE - 1), _output_vector(nullptr),
          _output_selection_mask(nullptr) {}

    void init(Schema* schema) override;
    void execute() override;

    // Getters for testing
    const std::string& attribute() const { return _attribute; }
    bool has_predicate() const { return _predicate_expr_raw.has_predicates(); }
    std::string predicate_string() const { return _predicate_expr_raw.to_string(); }

private:
    const std::string _attribute;
    uint64_t _max_id_value;
    Vector<T>* _output_vector;
    BitMask<State::MAX_VECTOR_SIZE>* _output_selection_mask;
    PredicateExpression _predicate_expr_raw;

    bool _is_string_predicate = false;
    ScalarPredicateExpression<T> _predicate_expr_numeric;
    ScalarPredicateExpression<ffx_str_t> _predicate_expr_string;
};

// Type aliases for convenience
using ScanPredicatedUint64 = ScanPredicated<uint64_t>;
using ScanPredicatedString = ScanPredicated<ffx_str_t>;

}// namespace ffx

#endif// VFENGINE_SCAN_PREDICATED_OPERATOR_HH
