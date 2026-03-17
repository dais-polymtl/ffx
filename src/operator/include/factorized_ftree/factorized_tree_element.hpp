#ifndef VFENGINE_FACTORIZED_TREE_ELEMENT_HPP
#define VFENGINE_FACTORIZED_TREE_ELEMENT_HPP
#include "../../include/vector/vector.hpp"
#include <memory>
#include <string>
#include <vector>

namespace ffx {
class FactorizedTreeElement {
public:
    FactorizedTreeElement(const std::string& attribute, const Vector<uint64_t>* value)
        : _parent(nullptr), _attribute(attribute), _value(value){};

    explicit FactorizedTreeElement(const std::string& attribute)
        : _parent(nullptr), _attribute(attribute), _value(nullptr){};

        // Basic utility functions
    void set_value_ptr(const Vector<uint64_t>* value);
    bool is_leaf() const;
    void print_tree() const;

    FactorizedTreeElement* add_leaf(const std::string& parent, const std::string& child, const Vector<uint64_t>* parent_vector,
                                    const Vector<uint64_t>* child_vector);
    std::vector<std::string> get_ancestors_to_root(const std::string& attribute);
    std::vector<std::vector<std::string>> get_prior_branches(const std::string& attribute);
    std::size_t count_edges_in_tree();
    std::size_t get_num_nodes();
    std::size_t count_unique_datachunks();

    // Helper functions for finding nodes
    FactorizedTreeElement* find_node_by_attribute(const std::string& attribute);

    // Public member variables
    FactorizedTreeElement* _parent;
    std::string _attribute;
    const Vector<uint64_t>* _value;
    std::vector<std::shared_ptr<FactorizedTreeElement>> _children;

    ~FactorizedTreeElement() {
        _parent = nullptr;
        _value = nullptr;
    }
    FactorizedTreeElement(const FactorizedTreeElement&) = delete;
    FactorizedTreeElement& operator=(const FactorizedTreeElement&) = delete;

private:
    // Private helper functions
    std::vector<std::string> trace_path_to_root_from_node(FactorizedTreeElement* node);
    void collect_branches_without_attribute(const std::string& target_attr, std::vector<std::string>& current_path,
                                            std::vector<std::vector<std::string>>& all_branches);
    std::size_t count_edges_from_node(FactorizedTreeElement* node);
    std::size_t count_nodes_from_node(FactorizedTreeElement* node);
};

}// namespace ffx

#endif// VFENGINE_FACTORIZED_TREE_ELEMENT_HPP