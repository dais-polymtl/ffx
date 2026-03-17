#ifndef VFENGINE_SINK_LINEAR_OPERATOR_HH
#define VFENGINE_SINK_LINEAR_OPERATOR_HH
#include "factorized_ftree/factorized_tree_element.hpp"
#include "operator.hpp"

namespace ffx {
class SinkLinear final : public Operator {
public:
    SinkLinear()
        : Operator(), _num_output_tuples(0), _ftree(nullptr), _leaf_parent(nullptr), _leaf(nullptr) {}
    void init(Schema* schema) override;
    void execute() override;
    uint64_t get_num_output_tuples() override { return _num_output_tuples; }

private:
    uint64_t _num_output_tuples;
    std::shared_ptr<FactorizedTreeElement> _ftree;
    const Vector<uint64_t>* _leaf_parent;
    const Vector<uint64_t>* _leaf;
};
}// namespace ffx

#endif
