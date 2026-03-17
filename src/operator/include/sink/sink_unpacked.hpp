#ifndef VFENGINE_SINK_OPERATOR_HH
#define VFENGINE_SINK_OPERATOR_HH

#include "operator.hpp"
#include "vector/unpacked_state.hpp"

namespace ffx {

class SinkUnpacked final : public Operator {
public:
    SinkUnpacked() : Operator(), _states_of_list_vectors(nullptr){};
    SinkUnpacked(const SinkUnpacked&) = delete;

    void init(Schema* schema) override;
    void execute() override;

    uint64_t get_num_output_tuples() override { return _num_output_tuples; }

private:
    std::unique_ptr<UnpackedState*[]> _states_of_list_vectors;
    uint64_t _num_list_vectors;
    uint64_t _num_output_tuples;
};

}// namespace ffx

#endif
