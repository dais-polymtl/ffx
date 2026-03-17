#include "include/arena_allocator.hpp"

namespace ffx {
ArenaAllocator& ArenaAllocator::getInstance() {
    static ArenaAllocator instance;
    return instance;
}

void ArenaAllocator::initialize(const size_t total_size) {
    _memory_pool.reset(new uint8_t[total_size]);
    _total_size = total_size;
    _current_offset = 0;
    _is_initialized = true;
}

void* ArenaAllocator::allocate(size_t size) {
    if (!_is_initialized) {
        throw std::runtime_error("ArenaAllocator not initialized before allocation");
    }

    std::atomic<size_t>& offset = _current_offset;
    size_t current = offset.load(std::memory_order_relaxed);
    size_t desired;

    do {
        desired = current + size;
        if (desired > _total_size) {
            throw std::bad_alloc();
        }
    } while (!offset.compare_exchange_weak(current, desired, std::memory_order_release, std::memory_order_relaxed));

    return _memory_pool.get() + current;
}
}// namespace ffx
