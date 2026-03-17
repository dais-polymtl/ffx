#ifndef VFENGINE_FFX_STR_T_HH
#define VFENGINE_FFX_STR_T_HH

#include <cstdint>
#include <cstring>
#include <functional>
#include <iosfwd>
#include <string>

namespace ffx {

// Forward declaration
class StringPool;

struct ffx_str_t {
    // Bit 31: null flag, Bits 0-30: actual string length
    uint32_t size;
    char prefix[4];// First 4 characters (always present)
    union {
        const char* ptr;// Pointer to the ENTIRE string for strings > 4 bytes
    } data;

    // Thresholds and flags
    static constexpr uint32_t INLINE_THRESHOLD = 4;
    static constexpr uint32_t NULL_FLAG = 0x80000000U;
    static constexpr uint32_t SIZE_MASK = 0x7FFFFFFFU;

    // Default constructor (empty string, not null)
    ffx_str_t() : size(0) {
        std::memset(prefix, 0, 4);
        data.ptr = nullptr;
    }

    // Constructor from const char* (requires StringPool for strings > 4 chars)
    // nullptr creates empty string, use null_value() for null
    ffx_str_t(const char* str, StringPool* pool);

    // Constructor from std::string
    ffx_str_t(const std::string& str, StringPool* pool);

    // Deep copy constructor (requires pool for long strings)
    ffx_str_t(const ffx_str_t& other, StringPool* pool);

    // Deleted implicit copy operations to prevent accidental interning without a pool
    ffx_str_t(const ffx_str_t& other) = delete;
    ffx_str_t& operator=(const ffx_str_t& other) = delete;

    // Move constructor
    ffx_str_t(ffx_str_t&& other) noexcept;

    // Destructor
    ~ffx_str_t() = default;// StringPool manages memory

    // Deep copy assignment
    void assign(const ffx_str_t& other, StringPool* pool);

    // Move assignment
    ffx_str_t& operator=(ffx_str_t&& other) noexcept;

    // Comparison operators
    bool operator==(const ffx_str_t& other) const;
    bool operator!=(const ffx_str_t& other) const;
    bool operator<(const ffx_str_t& other) const;
    bool operator>(const ffx_str_t& other) const;
    bool operator<=(const ffx_str_t& other) const;
    bool operator>=(const ffx_str_t& other) const;

    // Convert to std::string (for debugging/output)
    // Returns empty string for null values
    std::string to_string() const;

    // Convert to display string (shows "<NULL>" for null values)
    std::string to_display_string() const;

    // Get full string as const char* (for compatibility)
    const char* c_str() const;

    // ==================== Null Flag Methods ====================

    // Check if this represents a null value
    bool is_null() const { return (size & NULL_FLAG) != 0; }

    // Set as null value (clears all data)
    void set_null() {
        size = NULL_FLAG;
        data.ptr = nullptr;
        std::memset(prefix, 0, 4);
    }

    // Clear the null flag only
    void clear_null() { size &= SIZE_MASK; }

    // Get actual string length (without null flag)
    uint32_t get_size() const { return size & SIZE_MASK; }

    // Set size preserving the null flag
    void set_size(uint32_t s) { size = (size & NULL_FLAG) | (s & SIZE_MASK); }

    // Static factory for creating a null value
    static ffx_str_t null_value() {
        ffx_str_t result;
        result.set_null();
        return result;
    }

    // Check if string is empty (size 0 and NOT null)
    bool empty() const { return get_size() == 0 && !is_null(); }

    // Check if truly an empty string (not null, size 0)
    bool is_empty_string() const { return get_size() == 0 && !is_null(); }

    uint64_t hash() const;

    // ==================== Serialization ====================

    // Standardized serialization for ffx_str_t
    void serialize(std::ostream& out) const;

    // Standardized deserialization for ffx_str_t
    // Note: requires StringPool for strings > 4 chars
    static ffx_str_t deserialize(std::istream& in, StringPool* pool);

private:
    // Helper to copy string data
    void copy_from(const char* str, size_t len, StringPool* pool);

    // Helper to compare strings
    int compare(const ffx_str_t& other) const;
};

// Hash function for use in unordered_map/unordered_set
struct ffx_str_hash {
    std::size_t operator()(const ffx_str_t& str) const;
};

// Maximum string length (using 31 bits, but practically limited)
constexpr uint32_t FFX_STR_MAX_LENGTH = 1024 * 1024 * 1024;// 1GB

// Placeholder string for null values (used in display/debugging)
constexpr const char* FFX_NULL_PLACEHOLDER = "<NULL>";

}// namespace ffx

#endif
