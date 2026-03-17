#ifndef VFENGINE_SINK_NO_OP_OPERATOR_HH
#define VFENGINE_SINK_NO_OP_OPERATOR_HH

#include "../operator.hpp"

namespace ffx {

class SinkNoOp final : public Operator {
public:
    SinkNoOp() : Operator() {}
    SinkNoOp(const SinkNoOp&) = delete;

    void init(Schema* /*schema*/) override {}
    void execute() override { num_exec_call++; }
    uint64_t get_num_output_tuples() override { return 0; }
};

}// namespace ffx

#endif
