#ifndef VFENGINE_SCAN_UNPACKED_SYNCHRONIZED_OPERATOR_HH
#define VFENGINE_SCAN_UNPACKED_SYNCHRONIZED_OPERATOR_HH

#include "operator.hpp"
#include "vector/unpacked_vector.hpp"
#include <string>

namespace ffx {

class ScanUnpackedSynchronized final : public Operator {
public:
    ScanUnpackedSynchronized() = delete;
    ScanUnpackedSynchronized(const ScanUnpackedSynchronized&) = delete;
    explicit ScanUnpackedSynchronized(std::string attribute, uint64_t start_id)
        : _attribute(std::move(attribute)), _max_id_value(UnpackedState::MAX_VECTOR_SIZE - 1), _output_vector(nullptr),
          _start_id(start_id) {}
    void init(Schema* schema) override;
    void execute() override;
    void set_start_id(uint64_t start_id) { _start_id = start_id; }
    void set_max_id(uint64_t max_id) { _max_id_value = max_id; }

    // Getter for testing/naming
    const std::string& attribute() const { return _attribute; }

private:
    const std::string _attribute;
    uint64_t _max_id_value;
    UnpackedVector<uint64_t>* _output_vector;
    uint64_t _start_id;
};

}// namespace ffx

#endif
