#include "scan/scan_unpacked.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

namespace ffx {

template<typename T>
void ScanUnpacked<T>::init(Schema* schema) {
    static_assert(std::is_same_v<T, uint64_t>, "ScanUnpacked currently only supports uint64_t.");
    auto& map = *schema->map;
    _output_vector = map.allocate_unpacked_vector(_attribute);
    const auto table_size = schema->resolve_scan_domain_size(_attribute);
    std::cout << "ScanUnpacked operator for attribute " << _attribute << " selected domain size " << table_size
              << std::endl;

    _max_id_value = table_size - 1;
    _next_op->init(schema);
}

template<typename T>
void ScanUnpacked<T>::execute() {
    num_exec_call++;
    constexpr auto chunk_size = UnpackedState::MAX_VECTOR_SIZE;
    const size_t number_chunks = std::ceil(static_cast<double>(_max_id_value + 1) / chunk_size);

    for (size_t chunk = 1; chunk <= number_chunks; ++chunk) {
        auto start = (chunk - 1) * chunk_size;

        auto end = start + chunk_size * ((_max_id_value - start) >= chunk_size) +
                   (_max_id_value - start + 1) * ((_max_id_value - start) < chunk_size);
        const size_t _curr_chunk_size = end - start;
        for (size_t i = 0; i < _curr_chunk_size; ++i) {
            _output_vector->values[i] = start + i;
        }

        _output_vector->state->size = static_cast<int32_t>(_curr_chunk_size);
        _output_vector->state->pos = -1;

        // call next operator
        _next_op->execute();

        // backward compatibility reset
        _output_vector->state->size = 0;
        _output_vector->state->pos = -1;
    }
}

// Explicit template instantiations
template class ScanUnpacked<uint64_t>;

}// namespace ffx
