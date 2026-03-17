#include "sink/sink_linear.hpp"
#include "operator_utils.hpp"
#include <cassert>
#include <stdexcept>
#include <unordered_set>

namespace ffx {

void SinkLinear::init(Schema* schema) {
    _ftree = schema->root;
    _num_output_tuples = 0;

    // Get leaf attribute from column ordering (last attribute in ordering)
    const auto& ordering = *schema->column_ordering;
    if (ordering.empty()) { throw std::runtime_error("SinkLinear: column ordering is empty"); }
    const std::string& leaf_attr = ordering.back();

    // Get leaf vector and its DataChunk
    auto& map = *schema->map;
    _leaf = map.get_vector(leaf_attr);
    internal::DataChunk* leaf_chunk = map.get_chunk_for_attr(leaf_attr);

    if (!_leaf || !leaf_chunk) { throw std::runtime_error("SinkLinear: leaf vector/chunk not found for " + leaf_attr); }

    // Assert: leaf_attr must be the last attribute of the last DataChunk
    // The last DataChunk's last attribute is the final attribute in the linear chain
    const auto& chunk_attrs = leaf_chunk->get_attr_names();
    assert(!chunk_attrs.empty() && "DataChunk must have at least one attribute");
    assert(chunk_attrs.back() == leaf_attr && "SinkLinear: leaf_attr must be the last attribute of its DataChunk");

    // Get parent DataChunk
    internal::DataChunk* parent_chunk = leaf_chunk->get_parent();
    if (!parent_chunk) {
        // Leaf is root - no parent, just count leaf positions directly
        // This is a special case for single-attribute queries
        _leaf_parent = _leaf;// Use leaf as its own parent (identity)
    } else {
        // Get any vector from parent chunk to access parent state
        // The primary attribute is guaranteed to exist
        const std::string& parent_primary = parent_chunk->get_primary_attr();
        _leaf_parent = map.get_vector(parent_primary);

        if (!_leaf_parent) { throw std::runtime_error("SinkLinear: parent vector not found for " + parent_primary); }
    }
}

void SinkLinear::execute() {
    num_exec_call++;
    // RLE in leaf maps parent indices to child ranges
    // rle[p] = start index in leaf for parent index p
    // rle[p+1] = end index in leaf for parent index p
    // So we iterate over PARENT's valid indices and count valid children
    const auto* const RESTRICT parent_state = _leaf_parent->state;
    const auto* const RESTRICT parent_selector = &parent_state->selector;
    const auto* const RESTRICT leaf_selector = &_leaf->state->selector;

    const auto parent_start = GET_START_POS(*parent_state);
    const auto parent_end = GET_END_POS(*parent_state);

    for (auto pidx = parent_start; pidx <= parent_end; pidx++) {
        if (TEST_BIT(*parent_selector, pidx)) {
            // For each valid parent, count valid children in the leaf
            uint32_t child_start;
            uint32_t child_end;
            if (_leaf->state == parent_state) {
                // Identity RLE: leaf positions == parent positions
                child_start = static_cast<uint32_t>(pidx);
                child_end = static_cast<uint32_t>(pidx + 1);
            } else {
                const uint16_t* const RESTRICT offset_arr = _leaf->state->offset;
                child_start = offset_arr[pidx];
                child_end = offset_arr[pidx + 1];
            }
            for (auto cidx = child_start; cidx < child_end; cidx++) {
                if (TEST_BIT(*leaf_selector, cidx)) { _num_output_tuples++; }
            }
        }
    }
}

}// namespace ffx
