#ifndef VFENGINE_STATE_HH
#define VFENGINE_STATE_HH

#ifdef ENABLE_SELECTION_POS_VECTOR
#include "selection_vector.hpp"
#else
#include "bitmask.hpp"
#endif
#include <cstdint>
#include <cstring>

namespace ffx {

class State {
public:
    static constexpr int32_t MAX_VECTOR_SIZE = 2048;
    static constexpr int32_t SELECTOR_BYTES = MAX_VECTOR_SIZE / 8;// W = 8 bits per byte

    State() : start_pos(0), end_pos(MAX_VECTOR_SIZE - 1) { std::memset(offset, 0, sizeof(offset)); }

    [[nodiscard]] uint16_t getStartPos() const { return start_pos; }
    [[nodiscard]] uint16_t getEndPos() const { return end_pos; }
    void setStartPos(uint16_t idx_value) { start_pos = idx_value; }
    void setEndPos(uint16_t idx_value) { end_pos = idx_value; }

    uint16_t start_pos, end_pos;
#ifdef ENABLE_SELECTION_POS_VECTOR
    alignas(64) SelectionVector<MAX_VECTOR_SIZE> selector;
#else
    alignas(64) BitMask<MAX_VECTOR_SIZE> selector;
#endif
    alignas(64) uint16_t offset[MAX_VECTOR_SIZE + 1];
};

#define GET_START_POS(state) ((state).getStartPos())
#define GET_END_POS(state) ((state).getEndPos())
#define SET_START_POS(state, index) ((state).setStartPos(index))
#define SET_END_POS(state, index) ((state).setEndPos(index))

}// namespace ffx
#endif