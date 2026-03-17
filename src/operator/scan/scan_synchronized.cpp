#include "scan/scan_synchronized.hpp"
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace ffx {

void ScanSynchronized::init(Schema* schema) {
    auto& map = *schema->map;

    // For multithreaded execution, max_id is provided beforehand via set_max_id()
    // No need to search for table or calculate max_id here
    _output_vector = map.allocate_vector(_attribute);
    _output_selection_mask = &_output_vector->state->selector;

    _next_op->init(schema);
}

void ScanSynchronized::execute() {
    num_exec_call++;
    constexpr auto chunk_size = State::MAX_VECTOR_SIZE;

    // Process only one morsel starting from _start_id
    auto start = _start_id;

    auto end = std::min(start + chunk_size - 1, _max_id_value);
    const size_t _curr_chunk_size = end - start + 1;

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

}// namespace ffx
