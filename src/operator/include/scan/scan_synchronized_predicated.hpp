#ifndef VFENGINE_SCAN_SYNCHRONIZED_PREDICATED_OPERATOR_HH
#define VFENGINE_SCAN_SYNCHRONIZED_PREDICATED_OPERATOR_HH

#include "../predicate/predicate_eval.hpp"
#include "operator.hpp"
#include <string>
#include <vector>

namespace ffx {

template<typename T = uint64_t>
class ScanSynchronizedPredicated final : public Operator {
public:
    ScanSynchronizedPredicated() = delete;
    ScanSynchronizedPredicated(const ScanSynchronizedPredicated&) = delete;

    // Constructor with predicate expression
    explicit ScanSynchronizedPredicated(std::string attribute, uint64_t start_id, PredicateExpression predicate_expr)
        : _attribute(std::move(attribute)), _max_id_value(State::MAX_VECTOR_SIZE - 1), _output_vector(nullptr),
          _output_selection_mask(nullptr), _start_id(start_id), _predicate_expr_raw(std::move(predicate_expr)) {}

    void init(Schema* schema) override;
    void execute() override;

    // Setters for multithreaded execution
    void set_start_id(uint64_t start_id) { _start_id = start_id; }
    void set_max_id(uint64_t max_id) { _max_id_value = max_id; }

    // Getters for testing
    const std::string& attribute() const { return _attribute; }
    bool has_predicate() const { return _predicate_expr_raw.has_predicates(); }
    std::string predicate_string() const { return _predicate_expr_raw.to_string(); }

private:
    const std::string _attribute;
    uint64_t _max_id_value;
    Vector<T>* _output_vector;
    BitMask<State::MAX_VECTOR_SIZE>* _output_selection_mask;
    uint64_t _start_id;
    PredicateExpression _predicate_expr_raw;

    bool _is_string_predicate = false;
    ScalarPredicateExpression<T> _predicate_expr_numeric;
    ScalarPredicateExpression<ffx_str_t> _predicate_expr_string;
};

// Type alias for convenience
using ScanSynchronizedPredicatedUint64 = ScanSynchronizedPredicated<uint64_t>;

}// namespace ffx

#endif// VFENGINE_SCAN_SYNCHRONIZED_PREDICATED_OPERATOR_HH
