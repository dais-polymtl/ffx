#include "../include/factorized_ftree/factorized_tree.hpp"
#include <iostream>
#include <unordered_map>

namespace ffx {
FactorizedTree::FactorizedTree(const std::vector<std::pair<std::string, std::string>>& logical_plan) : _logical_plan(logical_plan) {}

int32_t FactorizedTree::get_max_depth(const std::shared_ptr<FactorizedTreeElement>& node) const {
    if (!node || node->_children.empty())
        return 0;

    auto maxDepth = 0;
    for (auto& child: node->_children) {
        maxDepth = std::max(maxDepth, get_max_depth(child));
    }
    return 1 + maxDepth;
}

FactorizedTreeElement*
FactorizedTree::find_last_non_leaf_node(const std::shared_ptr<FactorizedTreeElement>& node) const {
    if (!node || node->_children.empty())
        return node.get();// Node is a leaf

    auto current = node;
    while (!current->_children.empty()) {
        // Find the child with the maximum depth
        std::shared_ptr<FactorizedTreeElement> next = nullptr;
        int maxDepth = -1;

        for (const auto& child: current->_children) {
            int depth = get_max_depth(child);
            if (depth > maxDepth) {
                maxDepth = depth;
                next = child;
            }
        }
        current = next;

        if (!next || next->_children.empty()) {
            // Stop if the next node is a leaf
            break;
        }
    }
    return current ? current->_parent : nullptr;// this function finds the leaf, so we return
                                                // one step above for the last non leaf
}

std::shared_ptr<FactorizedTreeElement> FactorizedTree::create_node(const std::pair<std::string, std::string>& elem,
                                                                   const bool is_first) const {
    auto attribute = is_first ? elem.first : elem.second;
    auto new_node = std::make_shared<FactorizedTreeElement>(attribute);
    return new_node;
}

std::shared_ptr<FactorizedTreeElement> FactorizedTree::insert(const std::shared_ptr<FactorizedTreeElement>& root,
                                                              const std::pair<std::string, std::string>& elem) const {
    if (!root)
        return nullptr;

    // Attach the new child to this node
    if (const auto target_node = find_last_non_leaf_node(root)) {
        auto new_node = create_node(elem, false);
        target_node->_children.push_back(new_node);
        new_node->_parent = target_node;

        return new_node;
    }

    return nullptr;
}

std::shared_ptr<FactorizedTreeElement>
FactorizedTree::insert_packed(const std::shared_ptr<FactorizedTreeElement>& root,
                              const std::pair<std::string, std::string>& elem) const {
    /* Simply add the node to the root*/
    if (!root)
        return nullptr;

    auto new_node = create_node(elem, false);
    root->_children.push_back(new_node);
    new_node->_parent = root.get();

    return new_node;
}


std::shared_ptr<FactorizedTreeElement> FactorizedTree::build_tree(const bool is_packed) const {
    /*
         * create a simple tree while traversing the logical plan
         * N ary tree to be created
         * Follow the pattern defined in the doc -> start at the parent, traverse down the longest path till you find
         * the last non leaf node, and attach the child there
         */

    std::unordered_map<std::string, std::shared_ptr<FactorizedTreeElement>> cache;
    const auto scan_op = _logical_plan.front();
    auto root = create_node(scan_op, true);
    root->_parent = nullptr;
    cache[scan_op.first] = root;

    for (size_t i = 1; i < _logical_plan.size(); ++i) {
        const auto& elem = _logical_plan[i];

        const auto& parent = elem.first;
        const auto& child = elem.second;

        // Ensure parent node exists
        if (cache.find(parent) == cache.end()) {
            const auto err_msg = "FactorizedTree::build_tree(): " + child + " attribute does not have a parent";
            throw std::runtime_error(err_msg);
        }

        const auto& parent_node = cache[parent];

        // Insert the child into the tree
        if (is_packed) {
            cache[child] = insert_packed(parent_node, elem);
        } else {
            cache[child] = insert(parent_node, elem);
        }
    }

    return root;
}
}// namespace ffx