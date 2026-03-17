#include "include/ffx_str_t.hpp"
#include "include/string_pool.hpp"
#include <algorithm>
#include <cassert>
#include <iostream>

namespace ffx {

ffx_str_t::ffx_str_t(const char* str, StringPool* pool) : size(0) {
    std::memset(prefix, 0, 4);
    data.ptr = nullptr;

    if (!str) { return; }

    size_t len = std::strlen(str);
    if (len > FFX_STR_MAX_LENGTH) { len = FFX_STR_MAX_LENGTH; }

    copy_from(str, len, pool);
}

ffx_str_t::ffx_str_t(const std::string& str, StringPool* pool) : size(0) {
    std::memset(prefix, 0, 4);
    data.ptr = nullptr;

    size_t len = str.length();
    if (len > FFX_STR_MAX_LENGTH) { len = FFX_STR_MAX_LENGTH; }

    copy_from(str.c_str(), len, pool);
}

ffx_str_t::ffx_str_t(const ffx_str_t& other, StringPool* pool) : size(other.size) {
    std::memcpy(prefix, other.prefix, 4);

    if (other.is_null()) {
        data.ptr = nullptr;
        return;
    }

    const uint32_t actual_size = other.get_size();
    if (actual_size <= INLINE_THRESHOLD) {
        data.ptr = nullptr;
    } else {
        assert(pool != nullptr);
        assert(other.data.ptr != nullptr);
        data.ptr = pool->allocate_string(other.data.ptr, actual_size);
    }
}

ffx_str_t::ffx_str_t(ffx_str_t&& other) noexcept : size(other.size) {
    std::memcpy(prefix, other.prefix, 4);
    data.ptr = other.data.ptr;// Works for inlined too since it's a union

    other.size = 0;
    other.data.ptr = nullptr;
    std::memset(other.prefix, 0, 4);
}

void ffx_str_t::assign(const ffx_str_t& other, StringPool* pool) {
    if (this != &other) {
        size = other.size;
        std::memcpy(prefix, other.prefix, 4);

        if (other.is_null()) {
            data.ptr = nullptr;
            return;
        }

        const uint32_t actual_size = other.get_size();
        if (actual_size <= INLINE_THRESHOLD) {
            data.ptr = nullptr;
        } else {
            assert(pool != nullptr);
            assert(other.data.ptr != nullptr);
            data.ptr = pool->allocate_string(other.data.ptr, actual_size);
        }
    }
}

ffx_str_t& ffx_str_t::operator=(ffx_str_t&& other) noexcept {
    if (this != &other) {
        size = other.size;
        std::memcpy(prefix, other.prefix, 4);
        data.ptr = other.data.ptr;

        other.size = 0;
        other.data.ptr = nullptr;
        std::memset(other.prefix, 0, 4);
    }
    return *this;
}

void ffx_str_t::copy_from(const char* str, size_t len, StringPool* pool) {
    // Clear null flag and set size
    size = static_cast<uint32_t>(len) & SIZE_MASK;
    std::memset(prefix, 0, 4);

    if (len == 0) {
        data.ptr = nullptr;
        return;
    }

    // Copy prefix (first 4 chars or all if len <= 4)
    size_t prefix_len = std::min(len, size_t(4));
    std::memcpy(prefix, str, prefix_len);

    // Handle remaining characters
    if (len > INLINE_THRESHOLD) {
        assert(pool != nullptr);
        data.ptr = pool->allocate_string(str, len);
    } else {
        data.ptr = nullptr;
    }
}

int ffx_str_t::compare(const ffx_str_t& other) const {
    const bool this_null = is_null();
    const bool other_null = other.is_null();

    if (this_null && other_null) return 0;
    if (this_null) return -1;
    if (other_null) return 1;

    const uint32_t this_size = get_size();
    const uint32_t other_size = other.get_size();

    if (this_size == 0 || other_size == 0) { return static_cast<int>(this_size) - static_cast<int>(other_size); }

    const uint32_t min_len = std::min(this_size, other_size);

    // Compare prefix first
    const uint32_t prefix_len = std::min(min_len, 4u);
    const int prefix_cmp = std::memcmp(prefix, other.prefix, prefix_len);
    if (prefix_cmp != 0) return prefix_cmp;

    // Compare beyond prefix
    if (min_len > 4) {
        // Since ptr stores the entire string, we compare from byte 4
        // If one is inlined (size=4) and other is ptr, this block won't be entered for min_len=4
        if (this_size > 4 && other_size > 4) {
            assert(data.ptr != nullptr);
            assert(other.data.ptr != nullptr);
            const int tail_cmp = std::memcmp(data.ptr + 4, other.data.ptr + 4, min_len - 4);
            if (tail_cmp != 0) return tail_cmp;
        }
    }

    return static_cast<int>(this_size) - static_cast<int>(other_size);
}

bool ffx_str_t::operator==(const ffx_str_t& other) const { return compare(other) == 0; }
bool ffx_str_t::operator!=(const ffx_str_t& other) const { return compare(other) != 0; }
bool ffx_str_t::operator<(const ffx_str_t& other) const { return compare(other) < 0; }
bool ffx_str_t::operator>(const ffx_str_t& other) const { return compare(other) > 0; }
bool ffx_str_t::operator<=(const ffx_str_t& other) const { return compare(other) <= 0; }
bool ffx_str_t::operator>=(const ffx_str_t& other) const { return compare(other) >= 0; }

std::string ffx_str_t::to_string() const {
    if (is_null()) return std::string();

    const uint32_t actual_size = get_size();
    if (actual_size == 0) return std::string();

    if (actual_size <= INLINE_THRESHOLD) {
        return std::string(prefix, actual_size);
    } else {
        assert(data.ptr != nullptr);
        return std::string(data.ptr, actual_size);
    }
}

std::string ffx_str_t::to_display_string() const {
    if (is_null()) return std::string(FFX_NULL_PLACEHOLDER);
    return to_string();
}

const char* ffx_str_t::c_str() const {
    if (is_null()) return "";
    if (get_size() > INLINE_THRESHOLD) { assert(data.ptr != nullptr); }
    static thread_local std::string buffer;
    buffer = to_string();
    return buffer.c_str();
}

uint64_t ffx_str_t::hash() const {
    if (is_null()) return static_cast<uint64_t>(std::hash<uint32_t>{}(NULL_FLAG));

    const uint64_t FNV_OFFSET_BASIS = 14695981039346656037ULL;
    const uint64_t FNV_PRIME = 1099511628211ULL;

    uint64_t h = FNV_OFFSET_BASIS;
    const uint32_t actual_size = get_size();

    if (actual_size <= INLINE_THRESHOLD) {
        for (uint32_t i = 0; i < actual_size; ++i) {
            h ^= static_cast<unsigned char>(prefix[i]);
            h *= FNV_PRIME;
        }
    } else {
        assert(data.ptr != nullptr);
        for (uint32_t i = 0; i < actual_size; ++i) {
            h ^= static_cast<unsigned char>(data.ptr[i]);
            h *= FNV_PRIME;
        }
    }

    h ^= static_cast<uint64_t>(actual_size);
    h *= FNV_PRIME;

    return h;
}

// Hash function implementation for ffx_str_hash (uses ffx_str_t::hash())
std::size_t ffx_str_hash::operator()(const ffx_str_t& str) const { return static_cast<std::size_t>(str.hash()); }

void ffx_str_t::serialize(std::ostream& out) const {
    // Write size (including null flag)
    out.write(reinterpret_cast<const char*>(&size), sizeof(uint32_t));

    // Write prefix (always 4 bytes)
    out.write(prefix, 4);

    // Write remaining data if exists
    uint32_t actual_size = get_size();
    if (actual_size > 4) {
        assert(data.ptr != nullptr);
        // ptr stores the entire string, write from byte 4
        out.write(data.ptr + 4, actual_size - 4);
    }
}

ffx_str_t ffx_str_t::deserialize(std::istream& in, StringPool* pool) {
    ffx_str_t result;

    // Read size (including null flag)
    in.read(reinterpret_cast<char*>(&result.size), sizeof(uint32_t));

    // Read prefix
    in.read(result.prefix, 4);

    // Handle remaining data
    uint32_t actual_size = result.get_size();
    if (actual_size > 4) {
        assert(pool != nullptr);
        char* allocated = const_cast<char*>(pool->allocate_string(nullptr, actual_size));
        std::memcpy(allocated, result.prefix, 4);
        in.read(allocated + 4, actual_size - 4);
        result.data.ptr = allocated;
    } else {
        result.data.ptr = nullptr;
    }

    return result;
}

}// namespace ffx
