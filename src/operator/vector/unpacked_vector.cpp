#include "../include/vector/unpacked_vector.hpp"
#include <cstddef>
#include <cstring>

namespace ffx {

template<typename T>
UnpackedVector<T>::UnpackedVector() : state(nullptr) {
    const int32_t size = UnpackedState::MAX_VECTOR_SIZE;

    size_t values_size = size * sizeof(T);
    size_t state_size = sizeof(UnpackedState);

    // Total size with alignment padding
    size_t total_size = values_size + state_size + alignof(std::max_align_t) + alignof(T);

    _memory_block = std::make_unique<uint8_t[]>(total_size);
    uint8_t* ptr = _memory_block.get();

    // Values
    values = reinterpret_cast<T*>(ptr);
    ptr += values_size;

    // UnpackedState (aligned, placement new)
    ptr = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(ptr) + alignof(UnpackedState) - 1) &
                                     ~(alignof(UnpackedState) - 1));
    state = new (ptr) UnpackedState();
}

template<typename T>
UnpackedVector<T>::~UnpackedVector() {
    if constexpr (std::is_same_v<T, ffx_str_t>) {
        if (values && state) {
            const int32_t sz = state->size;
            for (int32_t i = 0; i < sz; ++i) {
                values[i].~ffx_str_t();
            }
        }
    }
    if (state) { state->~UnpackedState(); }
}

// Explicit template instantiations
template class UnpackedVector<uint64_t>;
template class UnpackedVector<ffx_str_t>;

}// namespace ffx
