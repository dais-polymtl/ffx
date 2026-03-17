#include "../include/vector/bitmask.hpp"
#include <cstring>

namespace ffx {

#ifdef BIT_ARRAY_AS_FILTER
// Standard allocator implementation with unique_ptr
template<std::size_t N>
BitMask<N>::BitMask() : owns_memory(true) {
    bits_uptr = std::make_unique<uint8_t[]>(N);
    bits = bits_uptr.get();
    setBitsTillIdx(N - 1);
}

template<std::size_t N>
BitMask<N>::BitMask(void* pre_allocated_buffer) : bits_uptr(nullptr), owns_memory(false) {
    bits = reinterpret_cast<uint8_t*>(pre_allocated_buffer);
    std::memset(bits, 0, N * sizeof(uint8_t));
    setBitsTillIdx(N - 1);
}

template<std::size_t N>
BitMask<N>::BitMask(const BitMask& other) : owns_memory(true) {
    bits_uptr = std::make_unique<uint8_t[]>(N);
    bits = bits_uptr.get();
    copyFrom(other);
}

template<std::size_t N>
BitMask<N>& BitMask<N>::operator=(const BitMask& other) {
    if (this != &other) { copyFrom(other); }
    return *this;
}

template<std::size_t N>
void BitMask<N>::setBit(const std::size_t index) const {
    bits[index] = 1;
}

template<std::size_t N>
void BitMask<N>::clearBit(const std::size_t index) const {
    bits[index] = 0;
}

template<std::size_t N>
bool BitMask<N>::testBit(const std::size_t index) const {
    return bits[index] != 0;
}

template<std::size_t N>
void BitMask<N>::toggleBit(const std::size_t index) const {
    bits[index] = !bits[index];
}

template<std::size_t N>
void BitMask<N>::setBitsTillIdx(const std::size_t idx) const {
    for (auto i = 0; i <= idx; i++)
        bits[i] = 1;
}

template<std::size_t N>
void BitMask<N>::clearBitsTillIdx(const std::size_t idx) const {
    for (auto i = 0; i <= idx; i++)
        bits[i] = 0;
}

template<std::size_t N>
void BitMask<N>::andWith(const BitMask& other) {
    for (std::size_t i = 0; i < N; ++i) {
        bits[i] = bits[i] && other.bits[i];
    }
}

template<std::size_t N>
void BitMask<N>::copyFrom(const BitMask& other) {
    for (auto i = 0; i < N; i++)
        bits[i] = other.bits[i];
}

#else

template<std::size_t N>
BitMask<N>::BitMask() : owns_memory(true) {
    const std::size_t numBlocks = REQUIRED_UINT64<N>;
    bits_uptr = std::make_unique<uint64_t[]>(numBlocks);
    bits = bits_uptr.get();
    setBitsTillIdx(N - 1);
}

template<std::size_t N>
BitMask<N>::BitMask(void* pre_allocated_buffer) : bits_uptr(nullptr), owns_memory(false) {
    const std::size_t numBlocks = REQUIRED_UINT64<N>;
    bits = reinterpret_cast<uint64_t*>(pre_allocated_buffer);
    std::memset(bits, 0, numBlocks * sizeof(uint64_t));
    setBitsTillIdx(N - 1);
}

template<std::size_t N>
BitMask<N>::BitMask(const BitMask& other) : owns_memory(true) {
    const std::size_t numBlocks = REQUIRED_UINT64<N>;
    bits_uptr = std::make_unique<uint64_t[]>(numBlocks);
    bits = bits_uptr.get();
    copyFrom(other);
}

template<std::size_t N>
BitMask<N>& BitMask<N>::operator=(const BitMask& other) {
    if (this != &other) { copyFrom(other); }
    return *this;
}

template<std::size_t N>
void BitMask<N>::setBit(const std::size_t index) const {
    bits[getUint64Index(index)] |= getBitMask(index);
}

template<std::size_t N>
void BitMask<N>::clearBit(const std::size_t index) const {
    bits[getUint64Index(index)] &= ~getBitMask(index);
}

template<std::size_t N>
bool BitMask<N>::testBit(const std::size_t index) const {
    return (bits[getUint64Index(index)] & getBitMask(index)) != 0;
}

template<std::size_t N>
void BitMask<N>::toggleBit(const std::size_t index) const {
    bits[getUint64Index(index)] ^= getBitMask(index);
}

template<std::size_t N>
void BitMask<N>::setBitsTillIdx(const std::size_t index) const {
    const std::size_t completeBlocks = getUint64Index(index);
    const std::size_t remainingBits = getBitPosition(index);

    // Set complete blocks
    for (std::size_t i = 0; i < completeBlocks; ++i) {
        bits[i] = ALL_ONES;
    }

    // Set remaining bits in the partial block (including the idx bit)
    const uint64_t mask = remainingBits == 63 ? ~0ULL : (1ULL << (remainingBits + 1)) - 1;
    bits[completeBlocks] |= mask;
}

template<std::size_t N>
void BitMask<N>::clearBitsTillIdx(const std::size_t index) const {
    const std::size_t completeBlocks = getUint64Index(index);
    const std::size_t remainingBits = getBitPosition(index);

    // Clear complete blocks
    for (std::size_t i = 0; i < completeBlocks; ++i) {
        bits[i] = 0;
    }

    // Clear remaining bits in the partial block (including the idx bit)
    const uint64_t mask = remainingBits == 63 ? ~0ULL : (1ULL << (remainingBits + 1)) - 1;
    bits[completeBlocks] &= ~mask;
}

template<std::size_t N>
void BitMask<N>::andWith(const BitMask& other) {
    const std::size_t numBlocks = REQUIRED_UINT64<N>;
    for (std::size_t i = 0; i < numBlocks; ++i) {
        bits[i] &= other.bits[i];
    }
}

template<std::size_t N>
void BitMask<N>::copyFrom(const BitMask& other) {
    const std::size_t numBlocks = REQUIRED_UINT64<N>;
    std::memcpy(bits, other.bits, numBlocks * sizeof(uint64_t));
}

#endif

// Explicit instantiations
template class BitMask<2>;
template class BitMask<4>;
template class BitMask<8>;
template class BitMask<16>;
template class BitMask<32>;
template class BitMask<64>;
template class BitMask<128>;
template class BitMask<256>;
template class BitMask<512>;
template class BitMask<1024>;
template class BitMask<2048>;

}// namespace ffx