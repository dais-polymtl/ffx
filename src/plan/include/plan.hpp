#ifndef VFENGINE_PIPELINE_HH
#define VFENGINE_PIPELINE_HH

#include "../../table/include/string_pool.hpp"
#include "factorized_ftree/factorized_tree_element.hpp"
#include "operator.hpp"
#include <chrono>

namespace ffx {

class Operator;

enum sink_type {
    SINK_UNPACKED,
    SINK_PACKED,
    SINK_PACKED_NOOP,
    SINK_UNPACKED_NOOP,
    SINK_MIN,
    SINK_LINEAR,
    // Export final result table
    SINK_EXPORT_CSV,
    SINK_EXPORT_JSON,
    SINK_EXPORT_MARKDOWN
};

class Plan {
public:
    Plan() = delete;
    Plan(const Plan&) = delete;
    Plan& operator=(const Plan&) = delete;
    Plan(std::unique_ptr<Operator> first_operator, const std::vector<std::string>& column_ordering);
    void init(const std::vector<const Table*>& tables, const std::shared_ptr<FactorizedTreeElement>& root,
              Schema* schema) const;
    std::chrono::milliseconds execute() const;
    uint64_t get_num_output_tuples() const;
    Operator* get_first_op() const;
    std::vector<std::string> get_operator_names(sink_type sink) const;
    void print_pipeline() const;

    StringPool* get_predicate_pool() const { return _predicate_pool.get(); }

private:
    std::unique_ptr<Operator> _first_op;
    std::unique_ptr<QueryVariableToVectorMap> map;
    const std::vector<std::string> _column_ordering;
    std::unique_ptr<StringPool> _predicate_pool;
};

// Standalone utility function to print operator chain from any starting operator
void print_operator_chain(Operator* first_op);

}// namespace ffx

#endif
