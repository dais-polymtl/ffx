#ifndef FFX_STRING_DICTIONARY_HPP
#define FFX_STRING_DICTIONARY_HPP

#include "ffx_str_t.hpp"
#include "string_hash_table.hpp"
#include "string_pool.hpp"
#include <cstdint>
#include <fstream>
#include <unordered_map>
#include <vector>

namespace ffx {

/**
 * String dictionary for dictionary encoding.
 * 
 * Design principles:
 * 1. Built during table loading/serialization
 * 2. Can be extended incrementally during execution when needed
 * 3. Fast lookups: O(1) ID->String (array), O(1) String->ID (hash table)
 * 4. No std::map/unordered_map in hot path
 */
class StringDictionary {
public:
    StringDictionary(StringPool* pool = nullptr) : _hash_table(pool), _pool(pool) {}
    ~StringDictionary() = default;

    /**
     * Build dictionary from a vector of strings.
     * Assigns IDs sequentially (0, 1, 2, ...).
     * Must be called before using the dictionary.
     */
    void build(const std::vector<ffx_str_t>& strings);

    /**
     * Build dictionary from string->ID pairs (for merging dictionaries).
     */
    void build(const std::vector<std::pair<ffx_str_t, uint64_t>>& entries);

    /**
    * Add a single string and get its ID.
     * Returns existing ID if string already exists.
     */
    uint64_t add_string(const ffx_str_t& str);

    /**
    * Finalize dictionary (builds hash table).
     * Must be called after all strings are added.
     */
    void finalize();

    // ========================================================================
    // Query Phase (hot path - no mutex, no map lookups)
    // ========================================================================

    /**
     * Get string from dictionary ID (O(1) array access).
     * Throws if ID doesn't exist.
     */
    const ffx_str_t& get_string(uint64_t id) const {
        if (id >= _id_to_string.size()) { throw std::runtime_error("StringDictionary: Invalid ID "); }
        return _id_to_string[id];
    }

    /**
     * Get dictionary ID for ffx_str_t (O(1) hash table lookup).
     * Returns UINT64_MAX if not found.
     */
    uint64_t get_id(const ffx_str_t& str) const {
        if (!_finalized) { throw std::runtime_error("StringDictionary: Not finalized, cannot lookup"); }
        return _hash_table.lookup(str);
    }


    /**
     * Check if ID exists (O(1) array bounds check).
     */
    bool has_id(uint64_t id) const { return id < _id_to_string.size(); }

    /**
     * Check if ffx_str_t exists (O(1) hash table lookup).
     */
    bool has_string(const ffx_str_t& str) const { return _finalized && _hash_table.contains(str); }

    /**
     * Get total number of entries.
     */
    size_t size() const { return _id_to_string.size(); }

    /**
    * Check if dictionary hash table is initialized.
     */
    bool is_finalized() const { return _finalized; }

    // ========================================================================
    // Serialization
    // ========================================================================

    void serialize(std::ofstream& out) const;
    void deserialize(std::ifstream& in, StringPool* pool = nullptr);

    // ========================================================================
    // Merging (for multi-table scenarios)
    // ========================================================================

    /**
     * Merge another dictionary into this one.
     * Creates new IDs for strings not in this dictionary.
     * Must be called before finalize().
     */
    void merge(const StringDictionary& other);

private:
    // ID -> String mapping (O(1) access by index)
    std::vector<ffx_str_t> _id_to_string;

    // String -> ID mapping (O(1) hash table lookup)
    StringHashTable _hash_table;

    // StringPool for managing ffx_str_t memory
    StringPool* _pool;

    // Temporary map for building phase (only used before finalize())
    std::unordered_map<ffx_str_t, uint64_t, ffx_str_hash> _build_map;

    // Flag indicating if dictionary hash table is initialized
    bool _finalized = false;

    // Next ID to assign (during building phase)
    uint64_t _next_id = 0;
};

}// namespace ffx

#endif
