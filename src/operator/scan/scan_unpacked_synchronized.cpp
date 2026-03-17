#include "scan/scan_unpacked_synchronized.hpp"
#include <algorithm>
#include <cstddef>
#include <iostream>
#include <string>

namespace ffx {

void ScanUnpackedSynchronized::init(Schema* schema) {
    auto& map = *schema->map;

    // For multithreaded execution, max_id is provided beforehand via set_max_id()
    _output_vector = map.allocate_unpacked_vector(_attribute);

    _next_op->init(schema);
}

void ScanUnpackedSynchronized::execute() {
    num_exec_call++;
    constexpr auto chunk_size = UnpackedState::MAX_VECTOR_SIZE;

    // Process only one morsel starting from _start_id
    auto start = _start_id;

    auto end = std::min(start + chunk_size - 1, _max_id_value);
    const size_t _curr_chunk_size = end - start + 1;

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

}// namespace ffx
