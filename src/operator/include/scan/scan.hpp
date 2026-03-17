#ifndef VFENGINE_SCAN_OPERATOR_HH
#define VFENGINE_SCAN_OPERATOR_HH

#include "operator.hpp"
#include <string>
#include <vector>


namespace ffx {

template<typename T = uint64_t>
class Scan final : public Operator {
public:
    Scan() = delete;
    Scan(const Scan&) = delete;
    explicit Scan(const std::vector<std::string>&) = delete;
    explicit Scan(std::string attribute)
        : _attribute(std::move(attribute)), _max_id_value(State::MAX_VECTOR_SIZE - 1), _output_vector(nullptr) {
        _output_selection_mask = nullptr;
    }
    void init(Schema* schema) override;
    void execute() override;

    // Getter for testing
    const std::string& attribute() const { return _attribute; }

private:
    const std::string _attribute;
    uint64_t _max_id_value;
    Vector<T>* _output_vector;
    BitMask<State::MAX_VECTOR_SIZE>* _output_selection_mask;
};

// Type aliases for convenience
using ScanUint64 = Scan<uint64_t>;
using ScanString = Scan<ffx_str_t>;

}// namespace ffx

#endif
