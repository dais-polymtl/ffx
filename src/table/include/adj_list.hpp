#ifndef VFENGINE_ADJLIST_HH
#define VFENGINE_ADJLIST_HH

#include <memory>
#include "ffx_str_t.hpp"

namespace ffx {

template<typename T = uint64_t>
class AdjList {
public:
    AdjList() : size(0), values(nullptr), _values_uptr(nullptr) {}
    explicit AdjList(size_t size);
    std::size_t size;
    T* values;

private:
    std::unique_ptr<T[]> _values_uptr;
};

// Type aliases for convenience
using AdjListUint64 = AdjList<uint64_t>;
using AdjListString = AdjList<ffx_str_t>;

}// namespace ffx

#endif
