#include "scan/scan_predicated.hpp"

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
void ScanPredicated<T>::init(Schema* schema) {
    auto& map = *schema->map;
    _output_vector = map.allocate_vector<T>(_attribute);
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

    const auto table_size = schema->resolve_scan_domain_size(_attribute);

    std::cout << "ScanPredicated(" << _attribute << ") domain size=" << table_size;
    if ((_is_string_predicate && _predicate_expr_string.has_predicates()) ||
        (!_is_string_predicate && _predicate_expr_numeric.has_predicates())) {
        std::cout << " [predicates: "
                  << (_is_string_predicate ? _predicate_expr_string.to_string() : _predicate_expr_numeric.to_string())
                  << "]";
    }
    std::cout << std::endl;

    _max_id_value = table_size - 1;
    _next_op->init(schema);
}

template<typename T>
void ScanPredicated<T>::execute() {
    num_exec_call++;
    constexpr auto chunk_size = State::MAX_VECTOR_SIZE;
    const size_t number_chunks = std::ceil(static_cast<double>(_max_id_value + 1) / chunk_size);

    // Temp buffer for branchless predicate evaluation
    T temp_buf[State::MAX_VECTOR_SIZE];

    for (size_t chunk = 1; chunk <= number_chunks; ++chunk) {
        auto start = (chunk - 1) * chunk_size;
        auto end = start + chunk_size * ((_max_id_value - start) >= chunk_size) +
                   (_max_id_value - start + 1) * ((_max_id_value - start) < chunk_size);
        const size_t curr_chunk_size = end - start;

        // Step 1: Fill temp buffer with all values (like normal scan)
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

        // Skip if no values pass the predicate
        if (out_write_idx == 0) { continue; }

        // Step 3: Copy from temp buffer to output vector using memcpy
        std::memcpy(_output_vector->values, temp_buf, out_write_idx * sizeof(T));

        // _output_vector->state->size = out_write_idx;
        // _output_vector->state->pos = -1;

        // Set start and end positions (these signify the correct slice)
        SET_START_POS(*_output_vector->state, 0);
        SET_END_POS(*_output_vector->state, out_write_idx - 1);
        SET_ALL_BITS(*_output_selection_mask);

        // Initialize output offset
        std::memset(_output_vector->state->offset, 0, State::MAX_VECTOR_SIZE * sizeof(uint16_t));

        // Call next operator
        _next_op->execute();

        // Reset for next chunk
        // _output_vector->state->size = 0;
        // _output_vector->state->pos = -1;
    }
}

// Explicit template instantiations
template class ScanPredicated<uint64_t>;
// Note: ScanPredicated<ffx_str_t> not instantiated - scans work with integer node IDs

}// namespace ffx
