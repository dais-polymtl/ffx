#ifndef VFENGINE_SINK_MIN_OPERATOR_HH
#define VFENGINE_SINK_MIN_OPERATOR_HH
#include "factorized_ftree/factorized_tree_element.hpp"
#include "operator.hpp"
#include <limits>
#include <string>
#include <vector>

namespace ffx {

class StringDictionary;// Forward declaration

// Group of attributes sharing the same State (allows single bitmap scan)
struct StateGroup {
    const State* state;
    std::vector<size_t> attr_indices;  // indices into _attribute_vectors
};

class SinkMin final : public Operator {
public:
    SinkMin()
        : Operator(), _num_output_tuples(0), _ftree(nullptr), _min_values(nullptr), _min_values_size(0),
          _dictionary(nullptr) {}

    void init(Schema* schema) override;
    void execute() override;
    uint64_t get_num_output_tuples() override { return _num_output_tuples; }

    // Get pointer to the local min_values array for merging
    const uint64_t* get_min_values() const { return _min_values; }
    size_t get_min_values_size() const { return _min_values_size; }

private:
    uint64_t _num_output_tuples;
    std::shared_ptr<FactorizedTreeElement> _ftree;
    uint64_t* _min_values;
    size_t _min_values_size;

    // Store Vector* pointers for each attribute
    std::vector<const Vector<uint64_t>*> _attribute_vectors;

    // String attribute support
    StringDictionary* _dictionary;
    std::vector<bool> _is_string_attr;

    // Optimization: group attributes by shared State for single-pass bitmap scanning
    std::vector<StateGroup> _state_groups;
};
}// namespace ffx

#endif