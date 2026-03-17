#ifndef VFENGINE_SELECTION_VECTOR_HPP
#define VFENGINE_SELECTION_VECTOR_HPP
#include <memory>
#include <string>
namespace ffx {
template<std::size_t N>
class SelectionVector {
public:
    SelectionVector();
    SelectionVector(const SelectionVector& other);
    SelectionVector& operator=(const SelectionVector& other);
    uint32_t* bits;
    int32_t size;

private:
    std::unique_ptr<uint32_t[]> _bits_uptr;
};
}// namespace ffx
#endif//VFENGINE_SELECTION_VECTOR_HPP
