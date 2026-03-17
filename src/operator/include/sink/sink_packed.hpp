#ifndef FFX_SINK_PACKED_OPERATOR_HH
#define FFX_SINK_PACKED_OPERATOR_HH

#include "../factorized_ftree/factorized_tree_element.hpp"
#include "operator.hpp"
#include <memory>
#include <vector/bitmask.hpp>
#include "../factorized_ftree/ftree_iterator.hpp"

namespace ffx {

// Struct for parent-leaf child pairs
struct ParentLeafPair {
    std::string parent_attr, child_attr;
    FactorizedTreeElement* parent;
    FactorizedTreeElement* child;
    uint64_t* parent_output;
};

// Struct for parent-non-leaf child pairs
struct ParentNonLeafPair {
    std::string parent_attr, child_attr;
    FactorizedTreeElement* parent;
    FactorizedTreeElement* child;
    uint64_t* parent_output;
    uint64_t* child_output;
};

class SinkPacked final : public Operator {
public:
    SinkPacked();
    SinkPacked(const SinkPacked&) = delete;
    void execute() override;
    void init(Schema* schema) override;
    uint64_t get_num_output_tuples() override { return _num_output_tuples; }
    std::vector<std::vector<uint64_t>> get_itr_values();
    uint64_t get_itr_values_size();

private:
    void print_debug_counts();  // Debug: print per-position counts matching iterator format
    
    Schema* _schema;  // Store schema for debug analysis
    std::shared_ptr<FactorizedTreeElement> _ftree;
    uint64_t _num_output_tuples;
    std::size_t _total_node_count;
    std::vector<FactorizedTreeElement*> _ordered_nodes;

    // Output registry using linear array of pairs
    std::unique_ptr<std::pair<std::string, std::unique_ptr<uint64_t[]>>[]> _output_registry;
    size_t _output_registry_size;

    // Parent-child pair arrays
    std::unique_ptr<ParentLeafPair[]> _leaf_pairs;
    std::unique_ptr<ParentNonLeafPair[]> _non_leaf_pairs;
    size_t _leaf_pair_count;
    size_t _non_leaf_pair_count;
    uint64_t* _root_output;
    State* _root_state;
    int32_t _root_start_idx, _root_end_idx;
    std::unique_ptr<ffx::FTreeIterator> _ftree_iterator;
    std::unique_ptr<uint64_t[]> _ftree_output;
};

}// namespace ffx

#endif