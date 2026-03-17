#include "sink/sink_min.hpp"
#include "schema/schema.hpp"
#include "string_dictionary.hpp"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
namespace ffx {

static void collect_all_nodes(FactorizedTreeElement* node,
                              std::unordered_map<std::string, const Vector<uint64_t>*>& vectors);

void SinkMin::init(Schema* schema) {
    auto root = schema->root;

    _ftree = root;
    _num_output_tuples = 0;

    // Get min_values array and size from schema
    assert(schema);
    assert(schema->min_values);
    assert(schema->min_values_size > 0);

    _min_values = schema->min_values;
    _min_values_size = schema->min_values_size;

    // Initialize all min values to maximum possible value
    for (size_t i = 0; i < _min_values_size; i++) {
        _min_values[i] = std::numeric_limits<uint64_t>::max();
    }

    // Initialize string attribute support
    _dictionary = schema->dictionary;

    if (schema->required_min_attrs.empty()) {
        throw std::runtime_error(
                "SinkMin: schema.required_min_attrs must be non-empty; use Q(MIN(a,b,...)) := ... with the min sink");
    }
    const std::vector<std::string>& target_attrs = schema->required_min_attrs;

    // Find vectors for each target attribute
    std::unordered_map<std::string, const Vector<uint64_t>*> all_vectors;
    collect_all_nodes(_ftree.get(), all_vectors);

    for (const auto& attr: target_attrs) {
        if (attr == "_cd") continue;

        auto it = all_vectors.find(attr);
        if (it != all_vectors.end()) {
            _attribute_vectors.push_back(it->second);
            bool is_string = false;
            if (schema->string_attributes) { is_string = schema->string_attributes->count(attr) > 0; }
            _is_string_attr.push_back(is_string);
        }
    }

    // Ensure we have the expected number of attributes
    assert(_attribute_vectors.size() == _min_values_size);

    // Build state groups: group attributes that share the same State* (single bitmap scan)
    std::unordered_map<const State*, std::vector<size_t>> state_to_attrs;
    for (size_t i = 0; i < _attribute_vectors.size(); i++) {
        const State* state = _attribute_vectors[i]->state;
        state_to_attrs[state].push_back(i);
    }

    // Convert to vector for cache-friendly iteration
    _state_groups.clear();
    _state_groups.reserve(state_to_attrs.size());
    for (auto& [state, attr_indices] : state_to_attrs) {
        _state_groups.push_back({state, std::move(attr_indices)});
    }
}

void SinkMin::execute() {
    num_exec_call++;

    // Process attributes grouped by shared State (single bitmap scan per group)
    for (const auto& group : _state_groups) {
        const State* state = group.state;
        const auto* const selector = &state->selector;
        const int32_t start = GET_START_POS(*state);
        const int32_t end = GET_END_POS(*state);
        const uint64_t* __restrict__ bits = selector->bits;

        const size_t start_block = start >> 6;
        const size_t end_block = end >> 6;
        const size_t start_bit = start & 63;
        const size_t end_bit = end & 63;

        // Single pass over the bitmap for all attributes in this group
        for (size_t block = start_block; block <= end_block; ++block) {
            uint64_t block_val = bits[block];

            // Skip empty blocks early
            if (__builtin_expect(block_val == 0, 0)) { continue; }

            // Apply masks for boundary blocks
            if (block == start_block) { block_val &= (~0ULL << start_bit); }
            if (block == end_block) {
                const uint64_t end_mask = (end_bit == 63) ? ~0ULL : ((1ULL << (end_bit + 1)) - 1);
                block_val &= end_mask;
            }

            // Process all set bits in this block
            while (block_val != 0) {
                const int bit_pos = __builtin_ctzll(block_val);
                const size_t idx = (block << 6) | bit_pos;

                // Update min for ALL attributes in this group at once
                for (const size_t attr_idx : group.attr_indices) {
                    const uint64_t val = _attribute_vectors[attr_idx]->values[idx];
                    const bool is_string = !_is_string_attr.empty() && _is_string_attr[attr_idx];

                    if (is_string) {
                        // String comparison
                        if (val != std::numeric_limits<uint64_t>::max()) {
                            uint64_t& curr_min = _min_values[attr_idx];
                            if (curr_min == std::numeric_limits<uint64_t>::max()) {
                                curr_min = val;
                            } else {
                                const auto& val_str = _dictionary->get_string(val);
                                const auto& curr_str = _dictionary->get_string(curr_min);
                                if (val_str < curr_str) { curr_min = val; }
                            }
                        }
                    } else {
                        // Numeric comparison (branchless)
                        uint64_t& curr_min = _min_values[attr_idx];
                        curr_min = (val < curr_min) ? val : curr_min;
                    }
                }

                block_val &= (block_val - 1);  // Clear lowest set bit
            }
        }
    }
}

static void collect_all_nodes(FactorizedTreeElement* node,
                              std::unordered_map<std::string, const Vector<uint64_t>*>& vectors) {
    if (!node) return;

    // Add this node's vector if it exists and is not "_cd"
    if (node->_value && node->_attribute != "_cd") { vectors[node->_attribute] = node->_value; }

    // Recursively collect from children in order
    for (const auto& child: node->_children) {
        collect_all_nodes(child.get(), vectors);
    }
}

}// namespace ffx
