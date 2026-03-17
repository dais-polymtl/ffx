// AI operator: Map — enumerates ftree tuples, applies a mapping (e.g. LLM stub),
// writes one result per tuple into a new vector so any sink can consume it.
#ifndef FFX_AI_OPERATOR_MAP_HH
#define FFX_AI_OPERATOR_MAP_HH

#include "ai_operator/ai_base.hpp"

#include "operator.hpp"
#include "vector/vector.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>


namespace ffx {

class Map final : public AIOperator {
public:
    Map();
    ~Map() override = default;

    void init_internal() override;
    void execute_internal() override;

    uint64_t get_num_output_tuples() override { return _num_output_tuples; }
};

}// namespace ffx

#endif
