#ifndef VFENGINE_SCAN_UNPACKED_OPERATOR_HH
#define VFENGINE_SCAN_UNPACKED_OPERATOR_HH

#include "operator.hpp"
#include "vector/unpacked_vector.hpp"
#include <string>

namespace ffx {

template<typename T = uint64_t>
class ScanUnpacked final : public Operator {
public:
    ScanUnpacked() = delete;
    ScanUnpacked(const ScanUnpacked&) = delete;
    explicit ScanUnpacked(std::string attribute)
        : _attribute(std::move(attribute)), _max_id_value(UnpackedState::MAX_VECTOR_SIZE - 1), _output_vector(nullptr) {
    }
    void init(Schema* schema) override;
    void execute() override;

    // Getter for testing/naming
    const std::string& attribute() const { return _attribute; }

private:
    const std::string _attribute;
    uint64_t _max_id_value;
    UnpackedVector<T>* _output_vector;
};

// Type aliases for convenience
using ScanUnpackedUint64 = ScanUnpacked<uint64_t>;

}// namespace ffx

#endif
