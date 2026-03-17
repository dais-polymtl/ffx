#include "include/string_hash_table.hpp"
#include <algorithm>
#include <climits>
#include <cmath>
#include <fstream>

namespace ffx {

size_t StringHashTable::find_slot(const ffx_str_t& str) const {
    if (_table.empty()) return SIZE_MAX;

    // Use ffx_str_t::hash() directly - no wrapper method needed
    uint64_t h = str.hash();
    size_t capacity = _table.size();
    size_t slot = h % capacity;

    // Linear probing
    size_t start_slot = slot;
    do {
        if (!_table[slot].occupied) {
            return slot;// Empty slot found
        }
        if (_table[slot].key == str) {// Uses ffx_str_t::operator==
            return slot;              // Key found
        }
        slot = (slot + 1) % capacity;
    } while (slot != start_slot);

    return SIZE_MAX;// Table full (should not happen with proper load factor)
}

void StringHashTable::resize(size_t new_capacity) {
    if (new_capacity == 0) {
        new_capacity = 16;// Minimum capacity
    }

    std::vector<Entry> old_table = std::move(_table);
    _table.resize(new_capacity);

    // Rehash all entries
    for (auto& entry: old_table) {
        if (entry.occupied) {
            size_t slot = find_slot(entry.key);
            if (slot != SIZE_MAX) {
                _table[slot].key = std::move(entry.key);
                _table[slot].value = entry.value;
                _table[slot].occupied = true;
            }
        }
    }
}

void StringHashTable::build(const std::vector<std::pair<ffx_str_t, uint64_t>>& entries) {
    _size = entries.size();

    // Calculate capacity based on load factor
    size_t capacity = static_cast<size_t>(std::ceil(_size / LOAD_FACTOR));
    // Round up to power of 2 for better hash distribution
    capacity = std::max(size_t(16), static_cast<size_t>(std::pow(2, std::ceil(std::log2(capacity)))));

    _table.clear();
    _table.resize(capacity);

    // Insert all entries
    for (const auto& pair: entries) {
        // Copy ffx_str_t using copy constructor with pool
        // This properly copies string data into our pool
        ffx_str_t key_copy(pair.first, _pool);

        size_t slot = find_slot(key_copy);
        if (slot != SIZE_MAX) {
            _table[slot].key = std::move(key_copy);
            _table[slot].value = pair.second;
            _table[slot].occupied = true;
        } else {
            // Table full, resize and retry
            resize(capacity * 2);
            slot = find_slot(key_copy);
            if (slot != SIZE_MAX) {
                _table[slot].key = std::move(key_copy);
                _table[slot].value = pair.second;
                _table[slot].occupied = true;
            }
        }
    }
}


uint64_t StringHashTable::lookup(const ffx_str_t& str) const {
    if (_table.empty()) return UINT64_MAX;

    size_t slot = find_slot(str);
    if (slot != SIZE_MAX && _table[slot].occupied) { return _table[slot].value; }
    return UINT64_MAX;
}

uint64_t StringHashTable::insert(const ffx_str_t& str, uint64_t value) {
    if (_table.empty()) {
        _table.resize(16);
    }

    if (static_cast<double>(_size + 1) > static_cast<double>(_table.size()) * LOAD_FACTOR) {
        resize(_table.size() * 2);
    }

    size_t slot = find_slot(str);
    if (slot == SIZE_MAX) {
        resize(std::max<size_t>(16, _table.size() * 2));
        slot = find_slot(str);
    }

    if (slot == SIZE_MAX) {
        throw std::runtime_error("StringHashTable: failed to find insertion slot");
    }

    if (_table[slot].occupied) {
        _table[slot].value = value;
        return _table[slot].value;
    }

    ffx_str_t key_copy(str, _pool);
    _table[slot].key = std::move(key_copy);
    _table[slot].value = value;
    _table[slot].occupied = true;
    _size++;
    return _table[slot].value;
}


bool StringHashTable::contains(const ffx_str_t& str) const { return lookup(str) != UINT64_MAX; }


void StringHashTable::serialize(std::ofstream& out) const {
    // Write capacity
    uint64_t capacity = _table.size();
    out.write(reinterpret_cast<const char*>(&capacity), sizeof(uint64_t));

    // Write size
    out.write(reinterpret_cast<const char*>(&_size), sizeof(uint64_t));

    // Write entries (only occupied ones)
    for (const auto& entry: _table) {
        if (entry.occupied) {
            uint8_t occupied_flag = 1;
            out.write(reinterpret_cast<const char*>(&occupied_flag), sizeof(uint8_t));

            // Use ffx_str_t's own serialization
            entry.key.serialize(out);

            out.write(reinterpret_cast<const char*>(&entry.value), sizeof(uint64_t));
        } else {
            uint8_t occupied_flag = 0;
            out.write(reinterpret_cast<const char*>(&occupied_flag), sizeof(uint8_t));
        }
    }
}

void StringHashTable::deserialize(std::ifstream& in, StringPool* pool) {
    _pool = pool;

    // Read capacity
    uint64_t capacity;
    in.read(reinterpret_cast<char*>(&capacity), sizeof(uint64_t));

    // Read size
    in.read(reinterpret_cast<char*>(&_size), sizeof(uint64_t));

    // Read entries
    _table.clear();
    _table.resize(capacity);

    for (size_t i = 0; i < capacity; i++) {
        uint8_t occupied_flag;
        in.read(reinterpret_cast<char*>(&occupied_flag), sizeof(uint8_t));

        if (occupied_flag) {
            _table[i].occupied = true;

            // Use ffx_str_t's own deserialization
            _table[i].key = ffx_str_t::deserialize(in, _pool);

            in.read(reinterpret_cast<char*>(&_table[i].value), sizeof(uint64_t));
        } else {
            _table[i].occupied = false;
        }
    }
}

}// namespace ffx
