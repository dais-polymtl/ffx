#include "query_variable_to_vector.hpp"

namespace ffx {

// ============================================================================
// uint64_t specializations
// ============================================================================

template<>
Vector<uint64_t>* QueryVariableToVectorMap::allocate_vector<uint64_t>(const std::string& query_variable) {
    if (_uint64_vector_map.find(query_variable) != _uint64_vector_map.end()) { throw; }
    // Create new DataChunk and use its state for the vector
    auto chunk = std::make_unique<internal::DataChunk>(_current_parent_chunk, query_variable);
    auto* chunk_ptr = chunk.get();
    State* shared_state = chunk_ptr->get_state();

    _uint64_vector_map.emplace(std::piecewise_construct, std::forward_as_tuple(query_variable),
                               std::forward_as_tuple(shared_state, false /* has_identity_rle */));
    _variable_to_is_list_map.try_emplace(query_variable, true /* is_list */);

    _chunks.push_back(std::move(chunk));
    _attr_to_chunk[query_variable] = chunk_ptr;
    _current_parent_chunk = chunk_ptr;// New allocations will be children of this chunk
    return &_uint64_vector_map[query_variable];
}

template<>
Vector<uint64_t>* QueryVariableToVectorMap::allocate_vector<uint64_t>(const std::string& query_variable,
                                                                      bool unpacked_mode) {
    if (_uint64_vector_map.find(query_variable) != _uint64_vector_map.end()) { throw; }
    // Create new DataChunk and use its state for the vector (even if unpacked_mode)
    auto chunk = std::make_unique<internal::DataChunk>(_current_parent_chunk, query_variable);
    auto* chunk_ptr = chunk.get();
    State* shared_state = chunk_ptr->get_state();

    _uint64_vector_map.emplace(std::piecewise_construct, std::forward_as_tuple(query_variable),
                               std::forward_as_tuple(shared_state, false /* has_identity_rle */));
    _variable_to_is_list_map.try_emplace(query_variable, true /* is_list */);

    _chunks.push_back(std::move(chunk));
    _attr_to_chunk[query_variable] = chunk_ptr;
    _current_parent_chunk = chunk_ptr;
    return &_uint64_vector_map[query_variable];
}

template<>
Vector<uint64_t>* QueryVariableToVectorMap::allocate_vector_shared_state<uint64_t>(const std::string& query_variable,
                                                                                   const std::string& share_with_attr) {

    if (_uint64_vector_map.find(query_variable) != _uint64_vector_map.end()) { throw; }

    // Find the DataChunk to share with
    auto it = _attr_to_chunk.find(share_with_attr);
    if (it == _attr_to_chunk.end()) {
        throw std::runtime_error("Cannot share state with non-existent attribute: " + share_with_attr);
    }

    internal::DataChunk* chunk = it->second;
    State* shared_state = chunk->get_state();

    // Create vector with shared state and identity RLE flag
    _uint64_vector_map.emplace(std::piecewise_construct, std::forward_as_tuple(query_variable),
                               std::forward_as_tuple(shared_state, true /* has_identity_rle */));
    _variable_to_is_list_map.try_emplace(query_variable, false);

    // Add as secondary attribute to the existing chunk
    chunk->add_secondary_attr(query_variable);
    _attr_to_chunk[query_variable] = chunk;

    // Update current parent to this chunk (since state is shared, we're at same level)
    _current_parent_chunk = chunk;

    return &_uint64_vector_map[query_variable];
}

template<>
Vector<uint64_t>* QueryVariableToVectorMap::allocate_vector<uint64_t>(const std::string& query_variable,
                                                                      State* shared_state) {
    // Legacy: allocate with explicit shared state (no DataChunk tracking)
    if (_uint64_vector_map.find(query_variable) != _uint64_vector_map.end()) { throw; }
    _uint64_vector_map.emplace(std::piecewise_construct, std::forward_as_tuple(query_variable),
                               std::forward_as_tuple(shared_state, true /* has_identity_rle */));
    _variable_to_is_list_map.try_emplace(query_variable, false);
    return &_uint64_vector_map[query_variable];
}

template<>
Vector<uint64_t>* QueryVariableToVectorMap::get_vector<uint64_t>(const std::string& query_variable) {
    auto oit = _uint64_vector_overrides.find(query_variable);
    if (oit != _uint64_vector_overrides.end() && oit->second != nullptr) {
        _variable_to_is_list_map[query_variable] = false;
        return oit->second;
    }
    auto& vec = _uint64_vector_map[query_variable];
    _variable_to_is_list_map[query_variable] = false;
    return &vec;
}

Vector<uint64_t>* QueryVariableToVectorMap::set_vector(const std::string& query_variable, State* shared_state,
                                                       bool has_identity_rle) {
    if (shared_state == nullptr) { throw std::runtime_error("set_vector: shared_state cannot be null"); }

    auto chunk_it = _attr_to_chunk.find(query_variable);

    _uint64_vector_map.erase(query_variable);
    _uint64_vector_map.emplace(std::piecewise_construct, std::forward_as_tuple(query_variable),
                               std::forward_as_tuple(shared_state, has_identity_rle));

    _variable_to_is_list_map[query_variable] = false;
    if (chunk_it != _attr_to_chunk.end()) { _current_parent_chunk = chunk_it->second; }
    return &_uint64_vector_map.at(query_variable);
}

void QueryVariableToVectorMap::set_vector_override(const std::string& query_variable, Vector<uint64_t>* vec) {
    if (vec == nullptr) { throw std::runtime_error("set_vector_override: vec cannot be null"); }
    _uint64_vector_overrides[query_variable] = vec;
    _variable_to_is_list_map[query_variable] = false;
}

void QueryVariableToVectorMap::clear_vector_override(const std::string& query_variable) {
    _uint64_vector_overrides.erase(query_variable);
}

// ============================================================================
// ffx_str_t specializations
// ============================================================================

template<>
Vector<ffx_str_t>* QueryVariableToVectorMap::allocate_vector<ffx_str_t>(const std::string& query_variable) {
    if (_string_vector_map.find(query_variable) != _string_vector_map.end()) { throw; }
    // Create new DataChunk and use its state
    auto chunk = std::make_unique<internal::DataChunk>(_current_parent_chunk, query_variable);
    auto* chunk_ptr = chunk.get();
    State* shared_state = chunk_ptr->get_state();

    _string_vector_map[query_variable] =
            std::make_unique<Vector<ffx_str_t>>(shared_state, false /* has_identity_rle */);
    _variable_to_is_list_map.try_emplace(query_variable, true /* is_list */);

    _chunks.push_back(std::move(chunk));
    _attr_to_chunk[query_variable] = chunk_ptr;
    _current_parent_chunk = chunk_ptr;
    return _string_vector_map[query_variable].get();
}

template<>
Vector<ffx_str_t>* QueryVariableToVectorMap::allocate_vector<ffx_str_t>(const std::string& query_variable,
                                                                        bool unpacked_mode) {
    if (_string_vector_map.find(query_variable) != _string_vector_map.end()) { throw; }
    // Create new DataChunk and use its state (even if unpacked_mode)
    auto chunk = std::make_unique<internal::DataChunk>(_current_parent_chunk, query_variable);
    auto* chunk_ptr = chunk.get();
    State* shared_state = chunk_ptr->get_state();

    _string_vector_map[query_variable] =
            std::make_unique<Vector<ffx_str_t>>(shared_state, false /* has_identity_rle */);
    _variable_to_is_list_map.try_emplace(query_variable, true /* is_list */);

    _chunks.push_back(std::move(chunk));
    _attr_to_chunk[query_variable] = chunk_ptr;
    _current_parent_chunk = chunk_ptr;
    return _string_vector_map[query_variable].get();
}

template<>
Vector<ffx_str_t>*
QueryVariableToVectorMap::allocate_vector_shared_state<ffx_str_t>(const std::string& query_variable,
                                                                  const std::string& share_with_attr) {

    if (_string_vector_map.find(query_variable) != _string_vector_map.end()) { throw; }

    // Find the DataChunk to share with
    auto it = _attr_to_chunk.find(share_with_attr);
    if (it == _attr_to_chunk.end()) {
        throw std::runtime_error("Cannot share state with non-existent attribute: " + share_with_attr);
    }

    internal::DataChunk* chunk = it->second;
    State* shared_state = chunk->get_state();

    // Create vector with shared state and identity RLE flag
    _string_vector_map[query_variable] = std::make_unique<Vector<ffx_str_t>>(shared_state, true);
    _variable_to_is_list_map.try_emplace(query_variable, false);

    // Add as secondary attribute to the existing chunk
    chunk->add_secondary_attr(query_variable);
    _attr_to_chunk[query_variable] = chunk;
    _current_parent_chunk = chunk;

    return _string_vector_map[query_variable].get();
}

template<>
Vector<ffx_str_t>* QueryVariableToVectorMap::allocate_vector<ffx_str_t>(const std::string& query_variable,
                                                                        State* shared_state) {
    // Legacy: allocate with explicit shared state (no DataChunk tracking)
    if (_string_vector_map.find(query_variable) != _string_vector_map.end()) { throw; }
    _string_vector_map[query_variable] = std::make_unique<Vector<ffx_str_t>>(shared_state, true);
    _variable_to_is_list_map.try_emplace(query_variable, false);
    return _string_vector_map[query_variable].get();
}

template<>
Vector<ffx_str_t>* QueryVariableToVectorMap::get_vector<ffx_str_t>(const std::string& query_variable) {
    auto it = _string_vector_map.find(query_variable);
    if (it == _string_vector_map.end()) { return nullptr; }
    _variable_to_is_list_map[query_variable] = false;
    return it->second.get();
}

void QueryVariableToVectorMap::set_current_parent_chunk(const std::string& attr) {
    auto it = _attr_to_chunk.find(attr);
    if (it == _attr_to_chunk.end()) {
        throw std::runtime_error("set_current_parent_chunk: Attribute '" + attr + "' not found");
    }
    _current_parent_chunk = it->second;
}

State* QueryVariableToVectorMap::reset_chunk_state(const std::string& attr) {
    auto it = _attr_to_chunk.find(attr);
    if (it == _attr_to_chunk.end()) {
        // If DataChunk doesn't exist (e.g., in manual test setups without joins),
        // create a new one hooked to the current parent chunk.
        auto chunk = std::make_unique<internal::DataChunk>(_current_parent_chunk, attr);
        auto* chunk_ptr = chunk.get();
        _chunks.push_back(std::move(chunk));
        _attr_to_chunk[attr] = chunk_ptr;

        // Ensure there is a vector allocated for this attribute and it points to the new chunk's state.
        if (_uint64_vector_map.find(attr) == _uint64_vector_map.end()) {
            _uint64_vector_map.emplace(std::piecewise_construct, std::forward_as_tuple(attr),
                                       std::forward_as_tuple(chunk_ptr->get_state(), false));
        } else {
            _uint64_vector_map.at(attr).state = chunk_ptr->get_state();
        }
        _variable_to_is_list_map[attr] = true;

        return chunk_ptr->get_state();
    }

    // For existing DataChunks, we reset the internal state.
    // We do NOT update existing vector state pointers here, because those vectors
    // might still be needed as read-only inputs by the FTreeBatchIterator.
    // FTreeReconstructor will provide new vectors via set_vector_override for downstream.
    return it->second->reset_state();
}

void QueryVariableToVectorMap::set_chunk_for_attr(const std::string& attr, const std::string& share_with_attr) {
    auto it = _attr_to_chunk.find(share_with_attr);
    if (it == _attr_to_chunk.end()) {
        throw std::runtime_error("set_chunk_for_attr: share_with_attr '" + share_with_attr + "' not found");
    }
    _attr_to_chunk[attr] = it->second;
}

void QueryVariableToVectorMap::reparent_chunk(const std::string& attr, const std::string& new_parent_attr) {
    auto child_it = _attr_to_chunk.find(attr);
    if (child_it == _attr_to_chunk.end()) {
        throw std::runtime_error("reparent_chunk: child attr '" + attr + "' not found");
    }
    auto parent_it = _attr_to_chunk.find(new_parent_attr);
    if (parent_it == _attr_to_chunk.end()) {
        throw std::runtime_error("reparent_chunk: new parent attr '" + new_parent_attr + "' not found");
    }

    // Set the new parent. Only change the parent if it's different to avoid loops/unnecessary work.
    if (child_it->second->get_parent() != parent_it->second) { child_it->second->set_parent(parent_it->second); }
}


// ============================================================================
// Non-templated methods
// ============================================================================

void QueryVariableToVectorMap::get_states_of_list_vectors(State** states) {
    auto i = 0;
    for (const auto& pair: _variable_to_is_list_map) {
        if (pair.second) {
            // Check uint64 map first
            auto uint_it = _uint64_vector_map.find(pair.first);
            if (uint_it != _uint64_vector_map.end()) {
                states[i++] = uint_it->second.state;
            } else {
                // Check string map
                auto str_it = _string_vector_map.find(pair.first);
                if (str_it != _string_vector_map.end()) { states[i++] = str_it->second->state; }
            }
        }
    }
}

uint64_t QueryVariableToVectorMap::get_num_list_vectors() const {
    uint64_t num_vectors = 0u;
    for (const auto& [variable, is_list]: _variable_to_is_list_map) {
        num_vectors += is_list ? 1 : 0;
    }
    return num_vectors;
}

uint64_t QueryVariableToVectorMap::get_populated_vectors_count() const {
    return _uint64_vector_map.size() + _string_vector_map.size();
}

std::vector<std::string> QueryVariableToVectorMap::get_populated_vectors() const {
    std::vector<std::string> populated_vectors;
    populated_vectors.reserve(_uint64_vector_map.size() + _string_vector_map.size());
    for (const auto& [key, _]: _uint64_vector_map) {
        populated_vectors.push_back(key);
    }
    for (const auto& [key, _]: _string_vector_map) {
        populated_vectors.push_back(key);
    }
    return populated_vectors;
}

// ============================================================================
// DataChunk query methods
// ============================================================================

bool QueryVariableToVectorMap::has_identity_rle(const std::string& attr) const {
    auto it = _attr_to_chunk.find(attr);
    if (it == _attr_to_chunk.end()) {
        return false;// Unknown attribute - assume no identity RLE
    }
    return it->second->has_identity_rle(attr);
}

std::vector<internal::DataChunk*> QueryVariableToVectorMap::get_chunk_tree() const {
    std::vector<internal::DataChunk*> result;
    result.reserve(_chunks.size());
    for (const auto& chunk: _chunks) {
        result.push_back(chunk.get());
    }
    return result;
}

internal::DataChunk* QueryVariableToVectorMap::get_chunk_for_attr(const std::string& attr) const {
    auto it = _attr_to_chunk.find(attr);
    if (it == _attr_to_chunk.end()) { return nullptr; }
    return it->second;
}

// ============================================================================
// Unpacked vector methods
// ============================================================================

template<>
UnpackedVector<uint64_t>*
QueryVariableToVectorMap::allocate_unpacked_vector<uint64_t>(const std::string& query_variable) {
    if (_unpacked_uint64_vector_map.find(query_variable) != _unpacked_uint64_vector_map.end()) { throw; }
    _unpacked_uint64_vector_map.emplace(std::piecewise_construct, std::forward_as_tuple(query_variable),
                                        std::forward_as_tuple());
    _unpacked_variable_to_is_list_map.try_emplace(query_variable, true /* is_list */);
    return &_unpacked_uint64_vector_map[query_variable];
}

template<>
UnpackedVector<uint64_t>* QueryVariableToVectorMap::get_unpacked_vector<uint64_t>(const std::string& query_variable) {
    auto& vec = _unpacked_uint64_vector_map[query_variable];
    _unpacked_variable_to_is_list_map[query_variable] = false;
    return &vec;
}

void QueryVariableToVectorMap::get_unpacked_states_of_list_vectors(UnpackedState** states) {
    auto i = 0;
    for (const auto& pair: _unpacked_variable_to_is_list_map) {
        if (pair.second) {
            auto it = _unpacked_uint64_vector_map.find(pair.first);
            if (it != _unpacked_uint64_vector_map.end()) { states[i++] = it->second.state; }
        }
    }
}

uint64_t QueryVariableToVectorMap::get_num_unpacked_list_vectors() const {
    uint64_t num_vectors = 0u;
    for (const auto& [variable, is_list]: _unpacked_variable_to_is_list_map) {
        num_vectors += is_list ? 1 : 0;
    }
    return num_vectors;
}

}// namespace ffx
