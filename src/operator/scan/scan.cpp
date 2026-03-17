#include "scan/scan.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
namespace ffx {

template<typename T>
void Scan<T>::init(Schema* schema) {
    static_assert(
            std::is_same_v<T, uint64_t>,
            "Scan currently only supports uint64_t. String support requires templating QueryVariableToVectorMap.");
    auto& map = *schema->map;
    _output_vector = map.allocate_vector(_attribute);
    _output_selection_mask = &_output_vector->state->selector;
    const auto table_size = schema->resolve_scan_domain_size(_attribute);
    std::cout << "Scan operator for attribute " << _attribute << " selected domain size " << table_size
              << std::endl;

    _max_id_value = table_size - 1;
    _next_op->init(schema);
}

template<typename T>
void Scan<T>::execute() {
    num_exec_call++;
    constexpr auto chunk_size = State::MAX_VECTOR_SIZE;
    const size_t number_chunks = std::ceil(static_cast<double>(_max_id_value + 1) / chunk_size);

    for (size_t chunk = 1; chunk <= number_chunks; ++chunk) {
        auto start = (chunk - 1) * chunk_size;

        // Branch less calculation of end index to handle leftover elements
        // If (max_id - start) >= chunk_size, then end = start + chunk_size
        // Else, end = max_id (remaining elements are less than chunk size)

        auto end = start + chunk_size * ((_max_id_value - start) >= chunk_size) +
                   (_max_id_value - start + 1) * ((_max_id_value - start) < chunk_size);
        const size_t _curr_chunk_size = end - start;
        for (size_t i = 0; i < _curr_chunk_size; ++i) {
            _output_vector->values[i] = start + i;
        }

        // _output_vector->state->size = static_cast<int32_t>(_curr_chunk_size);
        // _output_vector->state->pos = -1;

        // only needed for INLJ Packed
        // Set start and end position in the selection mask
        // Mark all bits as valid
        SET_ALL_BITS(*_output_selection_mask);
        SET_START_POS(*_output_vector->state, 0);
        SET_END_POS(*_output_vector->state, _curr_chunk_size - 1);
        // Initialize output offset
        std::memset(_output_vector->state->offset, 0, State::MAX_VECTOR_SIZE * sizeof(uint16_t));

        // call next operator
        _next_op->execute();

        // these two only needed for backward compatibility with Base INLJ
        // _output_vector->state->size = 0;
        // _output_vector->state->pos = -1;
    }
}

// Explicit template instantiations
template class Scan<uint64_t>;
// template class Scan<ffx_str_t>; // TODO: Enable when QueryVariableToVectorMap is templated

}// namespace ffx
