#ifndef VFENGINE_FACTORIZED_TREE_HPP
#define VFENGINE_FACTORIZED_TREE_HPP

#include "factorized_tree_element.hpp"
#include <vector>

namespace ffx {
class FactorizedTree {
public:
    explicit FactorizedTree(const std::vector<std::pair<std::string, std::string>>& logical_plan);
    std::shared_ptr<FactorizedTreeElement> build_tree(bool is_packed) const;

private:
    std::vector<std::pair<std::string, std::string>> _logical_plan;
    std::shared_ptr<FactorizedTreeElement>
    insert(const std::shared_ptr<FactorizedTreeElement>& root,
           const std::pair<std::string, std::string>& node) const;
    std::shared_ptr<FactorizedTreeElement>
    insert_packed(const std::shared_ptr<FactorizedTreeElement>& root,
                  const std::pair<std::string, std::string>& node) const;
    int32_t get_max_depth(const std::shared_ptr<FactorizedTreeElement>& node) const;
    FactorizedTreeElement*
    find_last_non_leaf_node(const std::shared_ptr<FactorizedTreeElement>& node) const;
    std::shared_ptr<FactorizedTreeElement> create_node(const std::pair<std::string, std::string>& ele,
                                                       bool is_first) const;
};
}// namespace ffx

#endif