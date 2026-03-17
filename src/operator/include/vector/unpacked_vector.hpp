#ifndef VFENGINE_UNPACKED_VECTOR_HH
#define VFENGINE_UNPACKED_VECTOR_HH

#include "../../table/include/ffx_str_t.hpp"
#include "unpacked_state.hpp"
#include <memory>

namespace ffx {

template<typename T = uint64_t>
class UnpackedVector {
public:
    UnpackedVector();
    ~UnpackedVector();

    T* values;
    UnpackedState* state;

private:
    std::unique_ptr<uint8_t[]> _memory_block;
};

// Type aliases for convenience
using UnpackedVectorUint64 = UnpackedVector<uint64_t>;
}// namespace ffx

#endif
