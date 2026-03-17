#include "include/string_dictionary.hpp"
#include <algorithm>
#include <fstream>
#include <stdexcept>

namespace ffx {

void StringDictionary::build(const std::vector<ffx_str_t>& strings) {
    if (_finalized) { throw std::runtime_error("StringDictionary: Cannot build after finalization"); }

    _build_map.clear();
    _id_to_string.clear();
    _next_id = 0;

    // Reserve space to avoid reallocations during build
    _id_to_string.reserve(strings.size());
    _build_map.reserve(strings.size());

    // Add all strings
    for (const auto& str: strings) {
        if (_build_map.find(str) == _build_map.end()) {
            uint64_t id = _next_id++;

            // Ensure ffx_str_t is properly stored
            ffx_str_t str_for_id_to_string(str, _pool);
            _id_to_string.push_back(std::move(str_for_id_to_string));

            // Deep copy again for the build map to satisfy mandatory pool requirement
            ffx_str_t str_for_build_map(_id_to_string.back(), _pool);
            _build_map[std::move(str_for_build_map)] = id;
        }
    }

    // Finalize automatically
    finalize();
}

void StringDictionary::build(const std::vector<std::pair<ffx_str_t, uint64_t>>& entries) {
    if (_finalized) { throw std::runtime_error("StringDictionary: Cannot build after finalization"); }

    _build_map.clear();
    _id_to_string.clear();

    // Find max ID
    uint64_t max_id = 0;
    for (const auto& pair: entries) {
        max_id = std::max(max_id, pair.second);
    }

    // Resize vector
    _id_to_string.resize(max_id + 1);

    // Add entries
    for (const auto& pair: entries) {
        ffx_str_t str_for_id_to_string(pair.first, _pool);
        _id_to_string[pair.second] = std::move(str_for_id_to_string);

        ffx_str_t str_for_build_map(_id_to_string[pair.second], _pool);
        _build_map[std::move(str_for_build_map)] = pair.second;
    }

    _next_id = max_id + 1;

    // Finalize automatically
    finalize();
}

uint64_t StringDictionary::add_string(const ffx_str_t& str) {
    if (_finalized) {
        uint64_t existing = _hash_table.lookup(str);
        if (existing != UINT64_MAX) {
            return existing;
        }

        uint64_t id = _next_id++;
        ffx_str_t str_for_id_to_string(str, _pool);
        _id_to_string.push_back(std::move(str_for_id_to_string));
        _hash_table.insert(_id_to_string.back(), id);
        return id;
    }

    auto it = _build_map.find(str);
    if (it != _build_map.end()) { return it->second; }

    uint64_t id = _next_id++;
    ffx_str_t str_for_id_to_string(str, _pool);
    _id_to_string.push_back(std::move(str_for_id_to_string));

    ffx_str_t str_for_build_map(_id_to_string.back(), _pool);
    _build_map[std::move(str_for_build_map)] = id;

    return id;
}

void StringDictionary::finalize() {
    if (_finalized) {
        return;// Already finalized
    }

    // Build hash table from build_map or _id_to_string
    std::vector<std::pair<ffx_str_t, uint64_t>> entries;

    if (!_build_map.empty()) {
        entries.reserve(_build_map.size());
        for (const auto& pair: _build_map) {
            entries.emplace_back(ffx_str_t(pair.first, _pool), pair.second);
        }
    } else {
        // Rebuilding from _id_to_string (e.g. after deserialization)
        entries.reserve(_id_to_string.size());
        for (uint64_t i = 0; i < _id_to_string.size(); ++i) {
            // Add everything; if we have gaps, the last one wins in the hash table
            // In a valid dictionary, all entries in _id_to_string are valid mappings
            entries.emplace_back(ffx_str_t(_id_to_string[i], _pool), i);
        }
    }

    _hash_table.build(entries);

    // Clear build_map (no longer needed)
    _build_map.clear();

    _finalized = true;
}

void StringDictionary::merge(const StringDictionary& other) {
    if (_finalized) { throw std::runtime_error("StringDictionary: Cannot merge after finalization"); }

    // Add all strings from other dictionary
    for (uint64_t i = 0; i < other._id_to_string.size(); i++) {
        const ffx_str_t& str = other._id_to_string[i];
        if (!str.is_empty_string() && _build_map.find(str) == _build_map.end()) {
            uint64_t new_id = _next_id++;
            ffx_str_t str_for_id_to_string(str, _pool);
            _id_to_string.push_back(std::move(str_for_id_to_string));

            ffx_str_t str_for_build_map(_id_to_string.back(), _pool);
            _build_map[std::move(str_for_build_map)] = new_id;
        }
    }
}

void StringDictionary::serialize(std::ofstream& out) const {
    // Write finalized flag
    out.write(reinterpret_cast<const char*>(&_finalized), sizeof(bool));

    // Write number of entries
    uint64_t num_entries = _id_to_string.size();
    out.write(reinterpret_cast<const char*>(&num_entries), sizeof(uint64_t));

    // Write next_id
    out.write(reinterpret_cast<const char*>(&_next_id), sizeof(uint64_t));

    // Write ID->String mapping
    for (size_t i = 0; i < _id_to_string.size(); i++) {
        _id_to_string[i].serialize(out);
    }
}

void StringDictionary::deserialize(std::ifstream& in, StringPool* pool) {
    if (pool) { _pool = pool; }
    _hash_table = StringHashTable(_pool);

    // Read finalized flag
    in.read(reinterpret_cast<char*>(&_finalized), sizeof(bool));

    // Read number of entries
    uint64_t num_entries;
    in.read(reinterpret_cast<char*>(&num_entries), sizeof(uint64_t));

    // Read next_id
    in.read(reinterpret_cast<char*>(&_next_id), sizeof(uint64_t));

    // Read ID->String mapping
    _id_to_string.clear();
    _id_to_string.reserve(num_entries);

    for (uint64_t i = 0; i < num_entries; i++) {
        _id_to_string.push_back(ffx_str_t::deserialize(in, _pool));
    }

    // Rebuild hash table
    _finalized = false;// Ensure finalize() runs
    finalize();
}

}// namespace ffx
