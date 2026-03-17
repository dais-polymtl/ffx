#ifndef VFENGINE_ARENA_ALLOCATOR_HH
#define VFENGINE_ARENA_ALLOCATOR_HH

#include <atomic>
#include <cstdlib>
#include <memory>

namespace ffx {

class ArenaAllocator {
public:
    ArenaAllocator(const ArenaAllocator&) = delete;
    ArenaAllocator(ArenaAllocator&&) = delete;
    ArenaAllocator& operator=(const ArenaAllocator&) = delete;
    ArenaAllocator& operator=(ArenaAllocator&&) = delete;

    static ArenaAllocator& getInstance();
    void initialize(size_t total_size);
    void* allocate(size_t size);

private:
    ArenaAllocator() : _total_size(0), _current_offset(0) {}
    ~ArenaAllocator() = default;

    size_t _total_size;
    std::atomic<size_t> _current_offset;
    bool _is_initialized = false;
    std::unique_ptr<uint8_t[]> _memory_pool;
};

}// namespace ffx

#endif
