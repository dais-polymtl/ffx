#ifndef VFENGINE_UNPACKED_STATE_HH
#define VFENGINE_UNPACKED_STATE_HH

#include <cstdint>

namespace ffx {

class UnpackedState {
public:
    static constexpr int32_t MAX_VECTOR_SIZE = 2048;
    UnpackedState() : pos(-1), size(MAX_VECTOR_SIZE) {}
    int32_t pos, size;
};

}// namespace ffx

#endif
