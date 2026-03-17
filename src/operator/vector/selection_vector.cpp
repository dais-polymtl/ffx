#include "../include/vector/selection_vector.hpp"
#include <cstring>
namespace ffx {
template<std::size_t N>
SelectionVector<N>::SelectionVector() {
    _bits_uptr = std::make_unique<uint32_t[]>(N);
    bits = _bits_uptr.get();
    size = 0;
}

template<std::size_t N>
SelectionVector<N>::SelectionVector(const SelectionVector<N>& other) {
    _bits_uptr = std::make_unique<uint32_t[]>(N);
    bits = _bits_uptr.get();
    size = other.size;
    std::memcpy(_bits_uptr.get(), other._bits_uptr.get(), size * sizeof(uint32_t));
}

template<std::size_t N>
SelectionVector<N>& SelectionVector<N>::operator=(const SelectionVector<N>& other) {
    if (this != &other) {
        _bits_uptr = std::make_unique<uint32_t[]>(N);
        bits = _bits_uptr.get();
        size = other.size;
        std::memcpy(_bits_uptr.get(), other._bits_uptr.get(), size * sizeof(uint32_t));
    }
    return *this;
}

template class SelectionVector<2048>;
}// namespace ffx