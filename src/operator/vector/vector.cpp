#include "../include/vector/vector.hpp"
#include <cstddef>
#include <cstring>
namespace ffx {

template<typename T>
Vector<T>::Vector(bool unpacked_mode)
    : state(nullptr), _unpacked_mode(unpacked_mode), _owns_state(true), _has_identity_rle(false) {
    const int32_t size = State::MAX_VECTOR_SIZE;

    // Calculate sizes for different components
    size_t values_size = size * sizeof(T);
    size_t state_size = sizeof(State);

    // Total size: values + State (selector and offset are now inline in State)
    size_t total_size = values_size + state_size;

    // Add extra padding for alignment
    total_size += alignof(State) + alignof(T);

    _memory_block = std::make_unique<uint8_t[]>(total_size);
    uint8_t* ptr = _memory_block.get();

    // Values
    values = reinterpret_cast<T*>(ptr);
    ptr += values_size;

    // State (selector and offset are inline — no separate allocation needed)
    ptr = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(ptr) + alignof(State) - 1) & ~(alignof(State) - 1));
    state = new (ptr) State();
}

template<typename T>
Vector<T>::Vector(State* shared_state, bool has_identity_rle)
    : state(shared_state), _unpacked_mode(false), _owns_state(false), _has_identity_rle(has_identity_rle) {
    // Only allocate memory for values - state is shared
    const int32_t size = State::MAX_VECTOR_SIZE;
    size_t values_size = size * sizeof(T);

    // Add padding for alignment
    size_t total_size = values_size + alignof(T);

    _memory_block = std::make_unique<uint8_t[]>(total_size);
    uint8_t* ptr = _memory_block.get();

    // Values only
    values = reinterpret_cast<T*>(ptr);
}

template<typename T>
Vector<T>::~Vector() {
    // For string types, we need to call destructors on the values array
    if constexpr (std::is_same_v<T, ffx_str_t>) {
        if (values && state) {
            // Use end_pos as an upper bound for constructed elements
            const uint16_t ep = state->end_pos;
            for (uint16_t i = 0; i <= ep && i < State::MAX_VECTOR_SIZE; ++i) {
                values[i].~ffx_str_t();
            }
        }
    }

    // Only destroy state if we own it (not shared)
    if (_owns_state && state) { state->~State(); }
}

// Explicit template instantiations
template class Vector<uint64_t>;
template class Vector<ffx_str_t>;

}// namespace ffx
