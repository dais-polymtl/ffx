#ifndef VFENGINE_OPERATOR_DEFINITION_HH
#define VFENGINE_OPERATOR_DEFINITION_HH

#include "../../table/include/table.hpp"
#include "factorized_ftree/factorized_tree_element.hpp"
#include "query_variable_to_vector.hpp"
#include "schema/schema.hpp"
#include <stdexcept>

#ifdef STORAGE_TO_VECTOR_MEMCPY_PTR_ALIAS
#define RESTRICT __restrict__
#else
#define RESTRICT
#endif

namespace ffx {

class Operator {
public:
    Operator(const Operator&) = delete;
    Operator& operator=(const Operator&) = delete;

    Operator() : next_op(nullptr), num_exec_call(0u) {}

    virtual ~Operator() = default;

    virtual void init(Schema* schema) = 0;
    virtual void execute() = 0;

    void set_next_operator(std::unique_ptr<Operator> next_operator) {
        _next_op = std::move(next_operator);
        next_op = _next_op.get();
    }

    virtual uint64_t get_num_output_tuples() { throw; }
    uint64_t get_num_exec_call() const { return num_exec_call; }

    Operator* next_op;

protected:
    uint64_t num_exec_call;
    std::unique_ptr<Operator> _next_op;

    static std::pair<const Table*, bool> select_join_table(const std::vector<const Table*>& tables,
                                                           const std::string& join_key, const std::string& output_key) {
        const Table* selected_table = nullptr;
        bool is_fwd = true;

        for (const auto* table: tables) {
            int join_key_idx = -1;
            int output_key_idx = -1;

            for (size_t i = 0; i < table->columns.size(); ++i) {
                if (table->columns[i] == join_key) join_key_idx = i;
                if (table->columns[i] == output_key) output_key_idx = i;
            }

            if (join_key_idx != -1 && output_key_idx != -1) {
                selected_table = table;
                if (join_key_idx < output_key_idx) {
                    is_fwd = true;
                } else {
                    is_fwd = false;
                }
                break;// Found a valid table (first match logic)
            }
        }

        if (!selected_table) { throw std::runtime_error("No table found for join " + join_key + " -> " + output_key); }

        return {selected_table, is_fwd};
    }
};

}// namespace ffx

#endif
