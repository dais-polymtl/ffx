#ifndef FFX_ANCESTOR_FINDER_UTILS_HPP
#define FFX_ANCESTOR_FINDER_UTILS_HPP

#include "query_variable_to_vector.hpp"
#include "vector/state.hpp"
#include "../vector/data_chunk.hpp"
#include <vector>
#include <string>
#include <stdexcept>

namespace ffx {
namespace internal {

/**
 * Result of building a state path for ancestor finding.
 */
struct AncestorFinderPathInfo {
    bool same_data_chunk;                   // true if ancestor and descendant share DataChunk
    std::vector<const State*> state_path;   // ordered states [ancestor...descendant], empty if same_data_chunk
};

/**
 * Build the state path from ancestor to descendant for FtreeAncestorFinder.
 * 
 * - If ancestor and descendant are in the same DataChunk: returns {same_data_chunk=true, state_path={}}
 * - If different DataChunks: validates linear path and returns ordered states (skipping duplicates)
 * 
 * @param map QueryVariableToVectorMap to get DataChunks and states
 * @param ancestor_attr The ancestor attribute name
 * @param descendant_attr The descendant attribute name
 * @return AncestorFinderPathInfo with same_data_chunk flag and state_path
 * @throws std::runtime_error if DataChunks are not in a linear ancestor-descendant path
 */
inline AncestorFinderPathInfo build_ancestor_finder_path(
    QueryVariableToVectorMap& map,
    const std::string& ancestor_attr,
    const std::string& descendant_attr
) {
    AncestorFinderPathInfo result;
    result.same_data_chunk = false;
    
    // Get DataChunks for both attributes
    DataChunk* ancestor_chunk = map.get_chunk_for_attr(ancestor_attr);
    DataChunk* descendant_chunk = map.get_chunk_for_attr(descendant_attr);
    
    if (!ancestor_chunk || !descendant_chunk) {
        throw std::runtime_error(
            "build_ancestor_finder_path: DataChunks not found for " + 
            ancestor_attr + " or " + descendant_attr
        );
    }
    
    // Case 1: Same DataChunk - identity mapping
    if (ancestor_chunk == descendant_chunk) {
        result.same_data_chunk = true;
        return result;
    }
    
    // Case 2: Different DataChunks - build state path
    // Verify descendant_chunk is a descendant of ancestor_chunk in DataChunk tree
    // and collect the path of DataChunks
    std::vector<DataChunk*> chunk_path;  // from descendant to ancestor
    DataChunk* current = descendant_chunk;
    
    while (current != nullptr) {
        chunk_path.push_back(current);
        if (current == ancestor_chunk) {
            break;
        }
        current = current->get_parent();
    }
    
    if (current != ancestor_chunk) {
        throw std::runtime_error(
            "build_ancestor_finder_path: " + descendant_attr + 
            " is not a descendant of " + ancestor_attr + " in DataChunk tree"
        );
    }
    
    // Reverse to get [ancestor_chunk, ..., descendant_chunk]
    std::reverse(chunk_path.begin(), chunk_path.end());
    
    // Build state path - collect unique states only (skip duplicates from shared state)
    result.state_path.reserve(chunk_path.size());
    
    const State* prev_state = nullptr;
    for (DataChunk* chunk : chunk_path) {
        const State* chunk_state = chunk->get_state();
        
        // Skip if this state is the same as previous (shared state within same DataChunk)
        // Note: This shouldn't happen in the chunk path since we're iterating over distinct chunks
        // But we check anyway for safety
        if (chunk_state != prev_state) {
            result.state_path.push_back(chunk_state);
            prev_state = chunk_state;
        }
    }
    
    // Validate we have at least 2 states
    if (result.state_path.size() < 2) {
        // This can happen if all chunks share the same state pointer
        // In this case, treat as same DataChunk (identity mapping)
        result.same_data_chunk = true;
        result.state_path.clear();
    }
    
    return result;
}

/**
 * Result of building state paths for FtreeMultiAncestorFinder.
 */
struct MultiAncestorFinderPathInfo {
    bool all_same_data_chunk;               // true if ALL attributes are in the same DataChunk
    std::vector<const State*> state_path;   // single path from root ancestor to descendant
};

/**
 * Build state path for FtreeMultiAncestorFinder (multiple ancestors to single descendant).
 * Used by NWayIntersection operators.
 * 
 * @param map QueryVariableToVectorMap to get DataChunks and states
 * @param path_attrs List of attribute names, ordered from root ancestor to descendant
 * @return MultiAncestorFinderPathInfo with state path
 * @throws std::runtime_error if path is invalid
 */
inline MultiAncestorFinderPathInfo build_multi_ancestor_finder_path(
    QueryVariableToVectorMap& map,
    const std::vector<std::string>& path_attrs
) {
    if (path_attrs.size() < 2) {
        throw std::runtime_error(
            "build_multi_ancestor_finder_path: path must have at least 2 attributes"
        );
    }
    
    MultiAncestorFinderPathInfo result;
    result.all_same_data_chunk = true;
    
    // Get first chunk to compare
    DataChunk* first_chunk = map.get_chunk_for_attr(path_attrs[0]);
    if (!first_chunk) {
        throw std::runtime_error(
            "build_multi_ancestor_finder_path: DataChunk not found for " + path_attrs[0]
        );
    }
    
    // Build state path by collecting unique states along the attribute path
    std::vector<const State*> state_path;
    const State* prev_state = nullptr;
    
    for (const auto& attr : path_attrs) {
        DataChunk* chunk = map.get_chunk_for_attr(attr);
        if (!chunk) {
            throw std::runtime_error(
                "build_multi_ancestor_finder_path: DataChunk not found for " + attr
            );
        }
        
        // Check if all in same DataChunk
        if (chunk != first_chunk) {
            result.all_same_data_chunk = false;
        }
        
        const State* chunk_state = chunk->get_state();
        
        // Only add if different from previous (skip shared states)
        if (chunk_state != prev_state) {
            state_path.push_back(chunk_state);
            prev_state = chunk_state;
        }
    }
    
    // If all same DataChunk or only one unique state
    if (result.all_same_data_chunk || state_path.size() < 2) {
        result.all_same_data_chunk = true;
        result.state_path.clear();
    } else {
        result.state_path = std::move(state_path);
    }
    
    return result;
}

} // namespace internal
} // namespace ffx

#endif // FFX_ANCESTOR_FINDER_UTILS_HPP

