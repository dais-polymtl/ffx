#include "scan/scan_synchronized_predicated.hpp"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace ffx {

template<typename T>
void ScanSynchronizedPredicated<T>::init(Schema* schema) {
    static_assert(std::is_same_v<T, uint64_t>, "ScanSynchronizedPredicated currently only supports uint64_t.");

    auto& map = *schema->map;

    // For multithreaded execution, max_id is provided beforehand via set_max_id()
    // No need to search for table or calculate max_id here
    _output_vector = map.allocate_vector(_attribute);
    _output_selection_mask = &_output_vector->state->selector;

    // Build the scalar predicate expression for this attribute
    _is_string_predicate = (schema->string_attributes && schema->string_attributes->count(_attribute) > 0);
    if (_is_string_predicate) {
        _predicate_expr_string = build_scalar_predicate_expr<ffx_str_t>(_predicate_expr_raw, _attribute,
                                                                        schema->predicate_pool, schema->dictionary);
    } else {
        _predicate_expr_numeric = build_scalar_predicate_expr<T>(_predicate_expr_raw, _attribute,
                                                                 schema->predicate_pool, schema->dictionary);
    }

    _next_op->init(schema);
}

template<typename T>
void ScanSynchronizedPredicated<T>::execute() {
    num_exec_call++;
    constexpr auto chunk_size = State::MAX_VECTOR_SIZE;

    // Process only one morsel starting from _start_id
    auto start = _start_id;
    auto end = std::min(start + chunk_size - 1, _max_id_value);
    const size_t curr_chunk_size = end - start + 1;

    // Temp buffer for branchless predicate evaluation
    T temp_buf[State::MAX_VECTOR_SIZE];

    // Step 1: Fill temp buffer with all values in this morsel
    for (size_t i = 0; i < curr_chunk_size; ++i) {
        temp_buf[i] = static_cast<T>(start + i);
    }

    // Step 2: Branchless compaction - keep only values passing predicate
    int32_t out_write_idx = 0;
    if (_is_string_predicate) {
        for (size_t i = 0; i < curr_chunk_size; ++i) {
            const bool pass = _predicate_expr_string.evaluate_id(static_cast<uint64_t>(temp_buf[i]));
            temp_buf[out_write_idx] = temp_buf[i];
            out_write_idx += pass ? 1 : 0;
        }
    } else {
        for (size_t i = 0; i < curr_chunk_size; ++i) {
            const bool pass = _predicate_expr_numeric.evaluate(temp_buf[i]);
            temp_buf[out_write_idx] = temp_buf[i];
            out_write_idx += pass ? 1 : 0;
        }
    }

    // Skip calling next operator if no values pass the predicate
    if (out_write_idx == 0) { return; }

    // Step 3: Copy from temp buffer to output vector using memcpy
    std::memcpy(_output_vector->values, temp_buf, out_write_idx * sizeof(T));

    // _output_vector->state->size = out_write_idx;
    // _output_vector->state->pos = -1;

    // Set start and end positions in the selection mask
    // Mark all bits as valid
    SET_ALL_BITS(*_output_selection_mask);
    SET_START_POS(*_output_vector->state, 0);
    SET_END_POS(*_output_vector->state, out_write_idx - 1);

    // Initialize output offset
    std::memset(_output_vector->state->offset, 0, State::MAX_VECTOR_SIZE * sizeof(uint16_t));

    // Call next operator
    _next_op->execute();

    // Reset for backward compatibility
    // _output_vector->state->size = 0;
    // _output_vector->state->pos = -1;
}

// Explicit template instantiation
template class ScanSynchronizedPredicated<uint64_t>;

}// namespace ffx
