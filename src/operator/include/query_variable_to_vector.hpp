#ifndef VFENGINE_CONTEXT_MEMORY_HH
#define VFENGINE_CONTEXT_MEMORY_HH

#include "../vector/data_chunk.hpp"
#include "vector/unpacked_vector.hpp"
#include "vector/vector.hpp"
#include <memory>
#include <unordered_map>
#include <vector>

namespace ffx {

class QueryVariableToVectorMap {
public:
    QueryVariableToVectorMap() = default;
    QueryVariableToVectorMap(QueryVariableToVectorMap&&) = delete;
    QueryVariableToVectorMap(const QueryVariableToVectorMap&) = delete;

    // ========================================================================
    // Standard allocation - creates new DataChunk
    // ========================================================================

    // Templated allocate_vector - owns its state, creates new DataChunk
    template<typename T = uint64_t>
    Vector<T>* allocate_vector(const std::string& query_variable);

    template<typename T = uint64_t>
    Vector<T>* allocate_vector(const std::string& query_variable, bool unpacked_mode);

    // ========================================================================
    // Shared state allocation - adds to existing DataChunk
    // ========================================================================

    // Allocate with shared state - adds to existing DataChunk, has identity RLE
    // @param query_variable The attribute name for the new vector
    // @param share_with_attr The attribute whose state to share (must exist)
    // @return The newly created vector with shared state
    template<typename T = uint64_t>
    Vector<T>* allocate_vector_shared_state(const std::string& query_variable, const std::string& share_with_attr);

    // Legacy: Allocate with explicit shared state pointer
    template<typename T = uint64_t>
    Vector<T>* allocate_vector(const std::string& query_variable, State* shared_state);

    // Set (upsert) uint64 vector for an attribute during operator rewrites.
    // If the attribute already belongs to a DataChunk, that membership is kept.
    // If no DataChunk mapping exists (legacy path), vector map is still updated.
    Vector<uint64_t>* set_vector(const std::string& query_variable, State* shared_state, bool has_identity_rle = false);

    // Register an externally-owned vector pointer for this variable name.
    // This is used by operator rewrites (e.g., CartesianProduct) to make
    // downstream operators read a new vector via get_vector(), while keeping
    // the original map-owned vectors alive for upstream iterators.
    //
    // The external vector must outlive all downstream operator usage.
    void set_vector_override(const std::string& query_variable, Vector<uint64_t>* vec);
    void clear_vector_override(const std::string& query_variable);

    // Explicitly set the parent chunk for subsequent allocations
    // @param attr The attribute whose DataChunk should become the parent
    void set_current_parent_chunk(const std::string& attr);

    /// Reset the state inside the DataChunk backing `attr`.
    /// Creates a new State, replaces the old one, returns pointer to the new State.
    /// Returns nullptr if `attr` has no DataChunk.
    State* reset_chunk_state(const std::string& attr);

    /// Re-map `attr` to the same DataChunk that backs `share_with_attr`.
    /// Does NOT allocate a new state — uses the one already in the target DataChunk.
    void set_chunk_for_attr(const std::string& attr, const std::string& share_with_attr);

    /// Reparent `attr`'s DataChunk under `new_parent_attr`'s DataChunk.
    /// This is needed when operators (like FTreeReconstructor) restructure the tree mid-pipeline.
    void reparent_chunk(const std::string& attr, const std::string& new_parent_attr);

    // ========================================================================
    // Query methods
    // ========================================================================

    // Templated get_vector
    template<typename T = uint64_t>
    Vector<T>* get_vector(const std::string& query_variable);

    /**
     * Check if an attribute has identity RLE (shares state with parent).
     * Returns true if the attribute is a secondary in its DataChunk.
     */
    bool has_identity_rle(const std::string& attr) const;

    /**
     * Get the DataChunk tree for sink traversal.
     * Returns chunks in creation order (root first).
     */
    std::vector<internal::DataChunk*> get_chunk_tree() const;

    /**
     * Get the DataChunk containing a specific attribute.
     * Returns nullptr if not found.
     */
    internal::DataChunk* get_chunk_for_attr(const std::string& attr) const;

    void get_states_of_list_vectors(State** states);
    uint64_t get_num_list_vectors() const;
    uint64_t get_populated_vectors_count() const;
    std::vector<std::string> get_populated_vectors() const;

    // ========================================================================
    // Unpacked vector allocation and retrieval
    // ========================================================================

    template<typename T = uint64_t>
    UnpackedVector<T>* allocate_unpacked_vector(const std::string& query_variable);

    template<typename T = uint64_t>
    UnpackedVector<T>* get_unpacked_vector(const std::string& query_variable);

    void get_unpacked_states_of_list_vectors(UnpackedState** states);
    uint64_t get_num_unpacked_list_vectors() const;

private:
    // Storage for uint64_t vectors (most common)
    std::unordered_map<std::string, Vector<uint64_t>> _uint64_vector_map;
    // Optional overrides for uint64_t vectors (externally owned)
    std::unordered_map<std::string, Vector<uint64_t>*> _uint64_vector_overrides;
    // Storage for string vectors
    std::unordered_map<std::string, std::unique_ptr<Vector<ffx_str_t>>> _string_vector_map;
    // Track which variables are "list" vectors (own their state)
    std::unordered_map<std::string, bool> _variable_to_is_list_map;

    // Storage for unpacked uint64_t vectors
    std::unordered_map<std::string, UnpackedVector<uint64_t>> _unpacked_uint64_vector_map;
    // Track which unpacked variables are "list" vectors
    std::unordered_map<std::string, bool> _unpacked_variable_to_is_list_map;

    // ========================================================================
    // DataChunk management (internal)
    // ========================================================================

    // All DataChunks in creation order
    std::vector<std::unique_ptr<internal::DataChunk>> _chunks;

    // Map attribute name -> DataChunk containing it
    std::unordered_map<std::string, internal::DataChunk*> _attr_to_chunk;

    // Current parent chunk for new chunk creation (tracks ftree depth)
    internal::DataChunk* _current_parent_chunk = nullptr;
};

}// namespace ffx

#endif
