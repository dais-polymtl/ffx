#include "join/inljoin_unpacked.hpp"
#include <cmath>
#include <iostream>
#include <type_traits>

namespace ffx {

template<typename T>
void INLJoinUnpacked<T>::init(Schema* schema) {
    static_assert(std::is_same_v<T, uint64_t>, "INLJoinUnpacked currently only supports uint64_t. String support "
                                               "requires hash-based adjacency list indexing.");
    auto& map = *schema->map;

    _join_key_vector = map.get_unpacked_vector(_join_key);
    _output_key_vector = map.allocate_unpacked_vector(_output_key);

    ResolvedJoinAdjList resolved;
    if (!schema->try_resolve_join_adj_list(_join_key, _output_key, resolved)) {
        throw std::runtime_error("No table found for join " + _join_key + " -> " + _output_key);
    }
    _adj_lists = reinterpret_cast<AdjList<T>*>(resolved.adj_list);
    std::cout << "INLJoinUnpacked " << _join_key << "->" << _output_key
              << (resolved.from_schema_map ? " using Schema adj_list" : " selected fallback table adj_list")
              << " (" << (resolved.is_fwd ? "fwd" : "bwd") << ")" << std::endl;

    _next_op->init(schema);
}

template<typename T>
void INLJoinUnpacked<T>::execute() {
    num_exec_call++;
    if (_join_key_vector->state->pos == -1) {
        loop_and_process_join_keys();
    } else {
        process_join_key();
    }
}

template<typename T>
void INLJoinUnpacked<T>::loop_and_process_join_keys() {
#ifdef STORAGE_TO_VECTOR_MEMCPY_PTR_ALIAS
    auto* __restrict__ state = _join_key_vector->state;
#else
    UnpackedState* state = _join_key_vector->state;
#endif

    state->pos++;
    while (state->pos < state->size) {
        process_join_key();
        state->pos++;
    }
    state->pos = -1;
}

template<typename T>
void INLJoinUnpacked<T>::process_join_key() {
    auto join_key_value = _join_key_vector->values[_join_key_vector->state->pos];
    const auto& adj_list = _adj_lists[join_key_value];

    constexpr auto chunk_size = UnpackedState::MAX_VECTOR_SIZE;

    const int32_t number_chunks = std::ceil(static_cast<double>(adj_list.size) / chunk_size);

    UnpackedState* RESTRICT output_state = _output_key_vector->state;
    uint64_t* RESTRICT _op_vector_values = _output_key_vector->values;

    for (int32_t chunk = 1; chunk <= number_chunks; ++chunk) {
        auto start = (chunk - 1) * chunk_size;

        // Branch less calculation of end index to handle leftover elements
        // If (max_id - start) >= chunk_size, then end = start + chunk_size
        // Else, end = max_id (remaining elements are less than chunk size)

        auto end = start + chunk_size * ((adj_list.size - start) >= chunk_size) +
                   (adj_list.size - start) * ((adj_list.size - start) < chunk_size);

        const size_t op_vector_size = end - start;
        for (size_t idx = 0; idx < op_vector_size; idx++) {
            _op_vector_values[idx] = adj_list.values[start + idx];
        }

        output_state->size = static_cast<int32_t>(op_vector_size);
        output_state->pos = -1;

        _next_op->execute();
    }
}

// Explicit template instantiations
template class INLJoinUnpacked<uint64_t>;
// template class INLJoinUnpacked<ffx_str_t>; // TODO: Enable when hash-based adjacency list indexing is implemented

}// namespace ffx
