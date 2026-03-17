#ifndef FFX_DATA_CHUNK_INTERNAL_HPP
#define FFX_DATA_CHUNK_INTERNAL_HPP

/**
 * DataChunk - Internal class for managing groups of vectors that share state.
 * 
 * This class is NOT part of the public API. It is managed by QueryVariableToVectorMap
 * and operators should never interact with DataChunks directly.
 * 
 * Key concepts:
 * - A DataChunk groups vectors that share the same State (bitmask + offset)
 * - DataChunks form a tree structure mirroring the factorized tree
 * - The "primary" attribute owns the offset; secondary attributes have identity offset
 * 
 * Example:
 *   Query: a -> b -> c -> d
 *   Cardinalities: a->b (m:n), b->c (n:1), c->d (m:n)
 *   
 *   DataChunk tree:
 *     Chunk 0: [a] State_0, primary="a"
 *         └── Chunk 1: [b, c] State_1, primary="b"  (c shares state due to n:1)
 *                 └── Chunk 2: [d] State_2, primary="d"
 */

#include "../include/vector/state.hpp"
#include <memory>
#include <string>
#include <vector>

namespace ffx {
namespace internal {

class DataChunk {
public:
    /**
     * Create a new DataChunk.
     * @param parent Parent chunk in the tree (nullptr for root)
     * @param primary_attr The first attribute added to this chunk (owns offset)
     */
    DataChunk(DataChunk* parent, const std::string& primary_attr)
        : _shared_state(std::make_shared<State>()), _parent(parent), _primary_attr(primary_attr) {
        _attr_names.push_back(primary_attr);
        // State constructor already initializes selector and offset inline
        // Initialize full valid range
        _shared_state->start_pos = 0;
        _shared_state->end_pos = State::MAX_VECTOR_SIZE - 1;
    }

    // Get the shared state (includes bitmask + offset)
    // NOTE: offset is only valid for the primary_attr. Secondary attrs have identity offset.
    State* get_state() { return _shared_state.get(); }
    const State* get_state() const { return _shared_state.get(); }

    /// Create a fresh State, replace the old one, return pointer.
    /// The old State is preserved in history so existing raw pointers remain valid.
    State* reset_state() {
        _historical_states.push_back(std::move(_shared_state));
        _shared_state = std::make_shared<State>();
        _shared_state->start_pos = 0;
        _shared_state->end_pos = State::MAX_VECTOR_SIZE - 1;
        return _shared_state.get();
    }

    // Tree structure
    DataChunk* get_parent() { return _parent; }
    const DataChunk* get_parent() const { return _parent; }
    void set_parent(DataChunk* new_parent) { _parent = new_parent; }
    bool is_root() const { return _parent == nullptr; }

    // Attribute tracking
    const std::string& get_primary_attr() const { return _primary_attr; }

    /**
     * Add a secondary attribute to this chunk.
     * Secondary attributes share state with the primary but have identity offset.
     */
    void add_secondary_attr(const std::string& attr_name) { _attr_names.push_back(attr_name); }

    /**
     * Check if an attribute has identity offset (is secondary in this chunk).
     */
    bool has_identity_rle(const std::string& attr_name) const { return attr_name != _primary_attr; }

    /**
     * Check if this chunk contains a specific attribute.
     */
    bool contains_attr(const std::string& attr_name) const {
        for (const auto& a: _attr_names) {
            if (a == attr_name) return true;
        }
        return false;
    }

    // Get all attribute names in this chunk
    const std::vector<std::string>& get_attr_names() const { return _attr_names; }

    // Get number of attributes in this chunk
    size_t size() const { return _attr_names.size(); }

private:
    std::shared_ptr<State> _shared_state;                  // Current active state
    std::vector<std::shared_ptr<State>> _historical_states;// Keep old states alive for upstream operators
    DataChunk* _parent;                                    // Parent chunk (nullptr for root)
    std::string _primary_attr;                             // First attribute (owns offset)
    std::vector<std::string> _attr_names;                  // All attributes (primary first, then secondary)
};

}// namespace internal
}// namespace ffx

#endif// FFX_DATA_CHUNK_INTERNAL_HPP
