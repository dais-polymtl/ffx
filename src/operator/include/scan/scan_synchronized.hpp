#ifndef VFENGINE_SCAN_SYNCHRONIZED_OPERATOR_HH
#define VFENGINE_SCAN_SYNCHRONIZED_OPERATOR_HH

#include "operator.hpp"
#include <string>
#include <vector>


namespace ffx {

class ScanSynchronized final : public Operator {
public:
    ScanSynchronized() = delete;
    ScanSynchronized(const ScanSynchronized&) = delete;
    explicit ScanSynchronized(const std::vector<std::string>&) = delete;
    explicit ScanSynchronized(std::string attribute, uint64_t start_id)
        : _attribute(std::move(attribute)), _max_id_value(State::MAX_VECTOR_SIZE - 1), 
          _output_vector(nullptr), _start_id(start_id) {
        _output_selection_mask = nullptr;
    }
    void init(Schema* schema) override;
    void execute() override;
    void set_start_id(uint64_t start_id) { _start_id = start_id; }
    void set_max_id(uint64_t max_id) { _max_id_value = max_id; }

    // Getter for testing
    const std::string& attribute() const { return _attribute; }

private:
    const std::string _attribute;
    uint64_t _max_id_value;
    Vector<uint64_t>* _output_vector;
    BitMask<State::MAX_VECTOR_SIZE>* _output_selection_mask;
    uint64_t _start_id;
};

}// namespace ffx

#endif
