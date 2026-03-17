#include "include/string_pool.hpp"
#include <algorithm>
#include <cstring>

namespace ffx {

StringPool::StringPool(size_t chunk_size) : _current_chunk_offset(0), _chunk_size(chunk_size), _current_chunk_idx(0) {
    allocate_new_chunk();
}

const char* StringPool::allocate_string(const char* str, size_t len) {
    if (len == 0) { return nullptr; }

    // Check if we need a new chunk
    if (_current_chunk_offset + len + 1 > _chunk_size) {
        // If the request is larger than the default chunk size, create a custom-sized chunk
        if (len + 1 > _chunk_size) {
            auto new_chunk = std::make_unique<char[]>(len + 1);
            char* dest = new_chunk.get();
            if (str) { std::memcpy(dest, str, len); }
            dest[len] = '\0';
            _chunks.push_back(std::move(new_chunk));

            // Note: we don't update _current_chunk_idx or _current_chunk_offset
            // for the current chunk, as this one is fully occupied.
            return dest;
        }
        allocate_new_chunk();
    }

    // Allocate in current chunk
    char* dest = _chunks[_current_chunk_idx].get() + _current_chunk_offset;

    // Copy string data
    if (str) { std::memcpy(dest, str, len); }

    // Null-terminate
    dest[len] = '\0';

    // Update offset
    const char* result = dest;
    _current_chunk_offset += len + 1;

    return result;
}

void StringPool::clear() {
    _current_chunk_offset = 0;
    _current_chunk_idx = 0;
    // Keep chunks allocated for reuse
    // Only clear if we want to free memory (could add a method for that)
}

size_t StringPool::total_allocated() const {
    size_t total = 0;
    for (const auto& chunk: _chunks) {
        total += _chunk_size;
    }
    return total;
}

void StringPool::allocate_new_chunk() {
    auto new_chunk = std::make_unique<char[]>(_chunk_size);
    _chunks.push_back(std::move(new_chunk));
    _current_chunk_idx = _chunks.size() - 1;
    _current_chunk_offset = 0;
}

}// namespace ffx
