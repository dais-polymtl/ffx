#ifndef FFX_OPERATOR_UTILS_HPP
#define FFX_OPERATOR_UTILS_HPP

#include "query_variable_to_vector.hpp"
#include "factorized_ftree/factorized_tree_element.hpp"
#include "factorized_ftree/ftree_ancestor_finder.hpp"
#include "../vector/data_chunk.hpp"
#include <memory>
#include <set>
#include <string>

namespace ffx {
namespace internal {

/**
 * Verify that descendant_attr is a descendant of ancestor_attr using DataChunks.
 * If the initial FactorizedTreeElement check fails, try to find a node in the
 * ancestor's DataChunk that is directly reachable by the descendant.
  */
inline std::pair<FactorizedTreeElement*, FactorizedTreeElement*>
verify_ancestor_descendant_relationship(
    QueryVariableToVectorMap& map,
    std::shared_ptr<FactorizedTreeElement> root,
    const std::string& ancestor_attr,
    const std::string& descendant_attr
) {
    // Find nodes in the ftree
    FactorizedTreeElement* ancestor_node = root->find_node_by_attribute(ancestor_attr);
    FactorizedTreeElement* descendant_node = root->find_node_by_attribute(descendant_attr);

    if (!ancestor_node || !descendant_node) {
        throw std::runtime_error("Nodes not found in tree for " + ancestor_attr + " or " + descendant_attr);
    }

    // Verify that descendant_attr is a descendant of ancestor_attr using DataChunks
    // Check the DataChunk tree structure instead of FactorizedTreeElement nodes
    DataChunk* ancestor_chunk = map.get_chunk_for_attr(ancestor_attr);
    DataChunk* descendant_chunk = map.get_chunk_for_attr(descendant_attr);

    if (!ancestor_chunk || !descendant_chunk) {
        throw std::runtime_error("DataChunks not found for " + ancestor_attr + " or " + descendant_attr);
    }

    // Traverse from descendant_chunk up to find ancestor_chunk
    DataChunk* current = descendant_chunk;
    bool found_ancestor = false;
    while (current != nullptr) {
        if (current == ancestor_chunk) {
            found_ancestor = true;
            break;
        }
        current = current->get_parent();
    }

    if (!found_ancestor) {
        throw std::runtime_error(descendant_attr + " must be a descendant of " + ancestor_attr);
    }

    // Check if ancestor_node is directly reachable from descendant_node
    FactorizedTreeElement* check_node = descendant_node;
    bool is_reachable = false;
    while (check_node != nullptr) {
        if (check_node == ancestor_node) {
            is_reachable = true;
            break;
        }
        check_node = check_node->_parent;
    }

    if (is_reachable) {
        // Direct path exists - use original nodes
        return {ancestor_node, descendant_node};
    }

    // Initial check failed - try to find a node in ancestor_chunk that is reachable by descendant_node
    FactorizedTreeElement* actual_ancestor_node = nullptr;
    
    // Get all attributes in the ancestor_chunk
    const auto& ancestor_chunk_attrs = ancestor_chunk->get_attr_names();
    
    // Check each attribute in ancestor_chunk to see if it's an ancestor of descendant_node
    for (const auto& attr : ancestor_chunk_attrs) {
        FactorizedTreeElement* candidate_node = root->find_node_by_attribute(attr);
        if (!candidate_node) continue;
        
        // Check if descendant_node is reachable from candidate_node
        check_node = descendant_node;
        bool candidate_reachable = false;
        while (check_node != nullptr) {
            if (check_node == candidate_node) {
                candidate_reachable = true;
                break;
            }
            check_node = check_node->_parent;
        }
        
        if (candidate_reachable) {
            actual_ancestor_node = candidate_node;
            break;
        }
    }
    
    if (actual_ancestor_node) {
        // Found a node in ancestor_chunk that is an ancestor of descendant_node
        return {actual_ancestor_node, descendant_node};
    }
    
    // Fallback 2: Find the common ftree ancestor of ancestor_node and descendant_node
    // This handles the case where they're in different branches of the ftree
    // but share a common ancestor (e.g., country2 and country3 both under person2)
    std::set<FactorizedTreeElement*> ancestor_path_set;
    check_node = ancestor_node;
    while (check_node != nullptr) {
        ancestor_path_set.insert(check_node);
        check_node = check_node->_parent;
    }
    
    FactorizedTreeElement* common_ancestor = nullptr;
    check_node = descendant_node;
    while (check_node != nullptr) {
        if (ancestor_path_set.count(check_node)) {
            common_ancestor = check_node;
            break;
        }
        check_node = check_node->_parent;
    }
    
    if (common_ancestor) {
        // Use the common ftree ancestor for FtreeAncestorFinder
        return {common_ancestor, descendant_node};
    }
    
    // Still can't find a path - throw error
    throw std::runtime_error("No valid path found from " + ancestor_attr + " to " + descendant_attr);
}

} // namespace internal
} // namespace ffx

#endif // FFX_OPERATOR_UTILS_HPP

