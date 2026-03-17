#include "include/adj_list.hpp"
#include "ffx_str_t.hpp"

namespace ffx {

template<typename T>
AdjList<T>::AdjList(size_t size) : size(size), _values_uptr(std::make_unique<T[]>(size)) {
    values = _values_uptr.get();
}

// Explicit template instantiations
template class AdjList<uint64_t>;
template class AdjList<ffx_str_t>;

}// namespace ffx
