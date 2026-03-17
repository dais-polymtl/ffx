#ifndef FFX_STRING_HASH_TABLE_HPP
#define FFX_STRING_HASH_TABLE_HPP

#include "ffx_str_t.hpp"
#include "string_pool.hpp"
#include <cstdint>
#include <fstream>
#include <vector>

namespace ffx {

/**
 * Hash table for fast ffx_str_t->ID lookup.
 * Uses open addressing with linear probing.
 * Supports build-time and incremental inserts.
 */
class StringHashTable {
public:
    struct Entry {
        ffx_str_t key;
        uint64_t value;
        bool occupied;

        Entry() : value(0), occupied(false) {}
    };

    StringHashTable(StringPool* pool = nullptr) : _pool(pool) {}

    // Build hash table from ffx_str_t->ID pairs (called once during construction)
    void build(const std::vector<std::pair<ffx_str_t, uint64_t>>& entries);


    // Lookup ffx_str_t->ID (O(1) average case, no mutex needed)
    uint64_t lookup(const ffx_str_t& str) const;

    // Insert or update one entry incrementally.
    // Safe to call after build(); returns the stored value's ID.
    uint64_t insert(const ffx_str_t& str, uint64_t value);


    // Check if string exists
    bool contains(const ffx_str_t& str) const;

    size_t size() const { return _size; }
    size_t capacity() const { return _table.size(); }

    // Serialization
    void serialize(std::ofstream& out) const;
    void deserialize(std::ifstream& in, StringPool* pool = nullptr);

private:
    std::vector<Entry> _table;
    size_t _size = 0;
    StringPool* _pool;// For managing ffx_str_t memory
    static constexpr double LOAD_FACTOR = 0.75;

    // Find slot for insertion/lookup (uses ffx_str_t::hash() directly)
    size_t find_slot(const ffx_str_t& str) const;

    // Resize table to maintain load factor
    void resize(size_t new_capacity);
};

}// namespace ffx

#endif
