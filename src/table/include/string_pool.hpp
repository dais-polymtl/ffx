#ifndef VFENGINE_STRING_POOL_HH
#define VFENGINE_STRING_POOL_HH

#include <cstddef>
#include <memory>
#include <vector>

namespace ffx {

class StringPool {
public:
    // Constructor with configurable chunk size (default 1MB)
    explicit StringPool(size_t chunk_size = 1024 * 1024);

    // Destructor
    ~StringPool() = default;

    // Allocate string in the pool
    // Returns pointer to allocated string data (null-terminated)
    const char* allocate_string(const char* str, size_t len);

    // Clear all allocated strings (reuse pool without deallocation)
    void clear();

    // Get total allocated memory
    size_t total_allocated() const;

    // Get number of chunks
    size_t num_chunks() const { return _chunks.size(); }

    // Get current chunk index
    size_t current_chunk_idx() const { return _current_chunk_idx; }

    // Get current chunk offset
    size_t current_chunk_offset() const { return _current_chunk_offset; }

private:
    std::vector<std::unique_ptr<char[]>> _chunks;
    size_t _current_chunk_offset;
    size_t _chunk_size;
    size_t _current_chunk_idx;

    // Allocate a new chunk
    void allocate_new_chunk();
};

}// namespace ffx

#endif

