#ifndef VFENGINE_BITMASK_HH
#define VFENGINE_BITMASK_HH

#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <cstring>
#include <algorithm>

namespace ffx {

#ifdef BIT_ARRAY_AS_FILTER
// BitArrayMask implementation (one byte per bit)
template<std::size_t N>
class BitMask {
public:
    static_assert((N & (N - 1)) == 0, "Size must be a power of 2");

    BitMask();
    BitMask(void* pre_allocated_buffer);
    ~BitMask() = default;  // unique_ptr handles cleanup when owns_memory=true
    BitMask(const BitMask& other);
    BitMask& operator=(const BitMask& other);

    void setBit(std::size_t index) const;
    void clearBit(std::size_t index) const;
    bool testBit(std::size_t index) const;
    void toggleBit(std::size_t index) const;
    void clearBitsTillIdx(size_t index) const;
    void setBitsTillIdx(size_t index) const;
    void andWith(const BitMask& other);
    void copyFrom(const BitMask& other);

    static constexpr std::size_t size() { return N; }
    static constexpr std::size_t getRequiredBytes() { return N * sizeof(uint8_t); }

private:
    std::unique_ptr<uint8_t[]> bits_uptr;
    uint8_t* bits;
    bool owns_memory;
};

#else
// BitMask implementation (bit packing in uint64_t)
constexpr std::size_t BITS_PER_UINT64 = 64;
constexpr uint64_t ALL_ONES = ~0ULL;
constexpr uint64_t BIT_MASK_63 = BITS_PER_UINT64 - 1;// 63 = 0b111111

template<std::size_t N>
constexpr std::size_t REQUIRED_UINT64 = (N + BITS_PER_UINT64 - 1) / BITS_PER_UINT64;

template<std::size_t N>
class BitMask {
public:
    static_assert((N & (N - 1)) == 0, "Size must be a power of 2");

    BitMask();
    explicit BitMask(void* pre_allocated_buffer);
    ~BitMask() = default;  // unique_ptr handles cleanup when owns_memory=true
    BitMask(const BitMask& other);
    BitMask& operator=(const BitMask& other);

    void setBit(std::size_t index) const;
    void clearBit(std::size_t index) const;
    bool testBit(std::size_t index) const;
    void toggleBit(std::size_t index) const;
    void clearBitsTillIdx(size_t index) const;
    void setBitsTillIdx(size_t index) const;
    void andWith(const BitMask& other);
    void copyFrom(const BitMask& other);

    static constexpr std::size_t getBitPosition(const std::size_t index) { return index & BIT_MASK_63; }
    static constexpr std::size_t getUint64Index(const std::size_t index) { return index >> 6; }
    static constexpr uint64_t getBitMask(const std::size_t index) { return 1ULL << getBitPosition(index); }
    static constexpr std::size_t size() { return N; }
    static constexpr std::size_t getRequiredBytes() { return REQUIRED_UINT64<N> * sizeof(uint64_t); }

    uint64_t* bits;
private:
    std::unique_ptr<uint64_t[]> bits_uptr;
    bool owns_memory;
};
#endif

template<std::size_t N>
inline uint32_t next_set_bit_in_range(const BitMask<N>& mask, uint32_t from, uint32_t end) {
    if (from > end || from >= static_cast<uint32_t>(N)) {
        return end + 1;
    }

#ifdef BIT_ARRAY_AS_FILTER
    for (uint32_t idx = from; idx <= end; ++idx) {
        if (mask.testBit(idx)) {
            return idx;
        }
    }
    return end + 1;
#else
    const uint32_t capped_end = std::min<uint32_t>(end, static_cast<uint32_t>(N - 1));
    uint32_t word_idx = from >> 6;
    const uint32_t end_word_idx = capped_end >> 6;

    uint64_t word = mask.bits[word_idx];
    word &= (~0ULL << (from & 63U));

    while (word_idx <= end_word_idx) {
        if (word_idx == end_word_idx) {
            const uint32_t bit_in_word_end = capped_end & 63U;
            const uint64_t end_mask = (bit_in_word_end == 63U)
                                      ? ~0ULL
                                      : ((1ULL << (bit_in_word_end + 1U)) - 1ULL);
            word &= end_mask;
        }

        if (word != 0ULL) {
            const uint32_t bit_pos = static_cast<uint32_t>(__builtin_ctzll(word));
            return (word_idx << 6U) + bit_pos;
        }

        ++word_idx;
        if (word_idx > end_word_idx) {
            break;
        }
        word = mask.bits[word_idx];
    }

    return capped_end + 1;
#endif
}

}// namespace ffx

#define SET_BIT(bitmask, index) ((bitmask).setBit(index))
#define CLEAR_BIT(bitmask, index) ((bitmask).clearBit(index))
#define TEST_BIT(bitmask, index) ((bitmask).testBit(index))
#define TOGGLE_BIT(bitmask, index) ((bitmask).toggleBit(index))
#define CLEAR_ALL_BITS(bitmask) ((bitmask).clearBitsTillIdx((bitmask).size() - 1))
#define SET_ALL_BITS(bitmask) ((bitmask).setBitsTillIdx((bitmask).size() - 1))
#define SET_BITS_TILL_IDX(bitmask, idx) ((bitmask).setBitsTillIdx(idx))
#define CLEAR_BITS_TILL_IDX(bitmask, idx) ((bitmask).clearBitsTillIdx(idx))
#define AND_BITMASKS(N, first, second) ((first).andWith(second))
#define COPY_BITMASK(N, first, second) ((first).copyFrom(second))

#endif