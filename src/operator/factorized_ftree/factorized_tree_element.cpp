#include "../include/factorized_ftree/factorized_tree_element.hpp"
#include <algorithm>
#include <functional>
#include <iostream>
#include <unordered_set>

namespace ffx {

void FactorizedTreeElement::set_value_ptr(const Vector<uint64_t>* value) { _value = value; }

bool FactorizedTreeElement::is_leaf() const { return _children.empty(); }

void print_tree_level(const std::vector<std::shared_ptr<FactorizedTreeElement>>& nodes, const int depth) {
    for (int i = 0; i < depth; ++i) {
        std::cout << "  ";
    }

    for (const auto& node: nodes) {
        std::cout << node->_attribute << " ";
    }
    std::cout << std::endl;

    // Collect all children for the next level
    std::vector<std::shared_ptr<FactorizedTreeElement>> next_level;
    for (const auto& node: nodes) {
        next_level.insert(next_level.end(), node->_children.begin(), node->_children.end());
    }

    if (!next_level.empty()) { print_tree_level(next_level, depth + 1); }
}

void print_tree_level(const FactorizedTreeElement* root) {
    std::cout << root->_attribute << std::endl;
    std::vector<std::shared_ptr<FactorizedTreeElement>> next_level;
    next_level.insert(next_level.end(), root->_children.begin(), root->_children.end());
    print_tree_level(next_level, 1);
}

void FactorizedTreeElement::print_tree() const { print_tree_level(this); }

// Converted from static to member function
FactorizedTreeElement* FactorizedTreeElement::add_leaf(const std::string& parent, const std::string& child,
                                                       const Vector<uint64_t>* parent_vector, const Vector<uint64_t>* child_vector) {
    // Find the parent node in current tree
    FactorizedTreeElement* parent_node = find_node_by_attribute(parent);
    if (!parent_node) { throw std::runtime_error("Parent node '" + parent + "' not found in tree"); }

    // Create new child node and add edge
    auto child_node = std::make_shared<FactorizedTreeElement>(child);
    child_node->_parent = parent_node;
    parent_node->_children.push_back(child_node);
    parent_node->_value = parent_vector;
    child_node->_value = child_vector;
    return child_node.get();
}

// Converted from static to member function
std::vector<std::string> FactorizedTreeElement::get_ancestors_to_root(const std::string& attribute) {
    // Find the target node in current tree
    FactorizedTreeElement* target_node = find_node_by_attribute(attribute);
    if (!target_node) { throw std::runtime_error("Attribute '" + attribute + "' not found in tree"); }

    // Get ancestors (path from target back to root)
    return trace_path_to_root_from_node(target_node->_parent);
}

// Converted from static to member function
std::vector<std::vector<std::string>> FactorizedTreeElement::get_prior_branches(const std::string& attribute) {
    // Collect all root-to-leaf branches that don't contain the target attribute
    std::vector<std::vector<std::string>> prior_branches;
    std::vector<std::string> current_path;
    collect_branches_without_attribute(attribute, current_path, prior_branches);

    return prior_branches;
}

// Converted from static to member function
std::size_t FactorizedTreeElement::count_edges_in_tree() { return count_edges_from_node(this); }

// New function
std::size_t FactorizedTreeElement::get_num_nodes() { return count_nodes_from_node(this); }

FactorizedTreeElement* FactorizedTreeElement::find_node_by_attribute(const std::string& attribute) {
    if (_attribute == attribute) { return this; }

    // Search in children recursively
    for (const auto& child: _children) {
        if (FactorizedTreeElement* found = child->find_node_by_attribute(attribute)) { return found; }
    }

    return nullptr;
}

// Private helper functions
std::vector<std::string> FactorizedTreeElement::trace_path_to_root_from_node(FactorizedTreeElement* node) {
    std::vector<std::string> ancestors;

    FactorizedTreeElement* current = node;
    while (current != nullptr) {
        ancestors.push_back(current->_attribute);
        current = current->_parent;
    }

    // ancestors now contains [target, parent, grandparent, ..., root]
    // Reverse to get [root, ..., grandparent, parent, target]
    std::reverse(ancestors.begin(), ancestors.end());

    return ancestors;
}

void FactorizedTreeElement::collect_branches_without_attribute(const std::string& target_attr,
                                                               std::vector<std::string>& current_path,
                                                               std::vector<std::vector<std::string>>& all_branches) {
    // Add current node to path
    current_path.push_back(_attribute);

    // If we encounter the target attribute, abandon this path (backtrack without adding to results)
    if (_attribute == target_attr) {
        current_path.pop_back();
        return;
    }

    // If this is a leaf, add this branch to results (since it doesn't contain target attribute)
    if (is_leaf()) {
        all_branches.push_back(current_path);
    } else {
        // Continue exploring children
        for (const auto& child: _children) {
            child->collect_branches_without_attribute(target_attr, current_path, all_branches);
        }
    }

    // Backtrack
    current_path.pop_back();
}

std::size_t FactorizedTreeElement::count_edges_from_node(FactorizedTreeElement* node) {
    if (!node) return 0;

    auto edges = node->_children.size();
    for (const auto& child: node->_children) {
        edges += count_edges_from_node(child.get());
    }
    return edges;
}

std::size_t FactorizedTreeElement::count_nodes_from_node(FactorizedTreeElement* node) {
    if (!node) return 0;

    std::size_t nodes = 1;// Count current node
    for (const auto& child: node->_children) {
        nodes += count_nodes_from_node(child.get());
    }
    return nodes;
}

std::size_t FactorizedTreeElement::count_unique_datachunks() {
    std::unordered_set<State*> seen;
    std::function<void(FactorizedTreeElement*)> collect = [&](FactorizedTreeElement* node) {
        if (!node || !node->_value) return;
        seen.insert(node->_value->state);
        for (const auto& child: node->_children) {
            collect(child.get());
        }
    };
    collect(this);
    return seen.size();
}

}// namespace ffx