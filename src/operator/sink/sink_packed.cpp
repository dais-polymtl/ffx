#include "sink/sink_packed.hpp"
#include "factorized_ftree/ftree_iterator.hpp"
#include "sink/sink_itr_merged.hpp"
#include <cassert>
#include <climits>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <queue>
#include <unordered_set>
#include <vector/bitmask.hpp>
#include <vector>

namespace ffx {

static void process_parent_leaf_pair(const ParentLeafPair& pair);
static void process_parent_non_leaf_pair(const ParentNonLeafPair& pair);
static void collect_nodes_by_level(FactorizedTreeElement* node, int depth,
                                   std::vector<std::vector<FactorizedTreeElement*>>& levels);
static std::vector<FactorizedTreeElement*> get_topological_order(const std::shared_ptr<FactorizedTreeElement>& root);

SinkPacked::SinkPacked()
    : _ftree(nullptr), _num_output_tuples(0), _total_node_count(0), _ordered_nodes({}), _output_registry(nullptr),
      _output_registry_size(0), _leaf_pairs(nullptr), _non_leaf_pairs(nullptr), _leaf_pair_count(0),
      _non_leaf_pair_count(0), _root_output(nullptr), _root_state(nullptr) {
    num_exec_call = 0;
}

void SinkPacked::init(Schema* schema) {
    _schema = schema;// Store schema for debug analysis
    auto& map = *schema->map;
    auto root = schema->root;

    _ftree = root;
    _num_output_tuples = 0;
    _total_node_count = map.get_populated_vectors_count();
    _ordered_nodes = get_topological_order(_ftree);

    // Count non-leaf nodes and parent-child pairs
    const size_t non_leaf_count = _ordered_nodes.size();
    size_t total_leaf_pairs = 0;
    size_t total_non_leaf_pairs = 0;

    for (auto* node: _ordered_nodes) {
        if (!node->_children.empty()) {
            for (const auto& child: node->_children) {
                if (child->_children.empty()) {
                    total_leaf_pairs++;
                } else {
                    total_non_leaf_pairs++;
                }
            }
        }
    }

    // Create output registry for non-leaf nodes
    _output_registry = std::make_unique<std::pair<std::string, std::unique_ptr<uint64_t[]>>[]>(non_leaf_count);
    _output_registry_size = 0;

    for (auto* node: _ordered_nodes) {
        if (!node->_children.empty()) {
            auto output_array = std::make_unique<uint64_t[]>(State::MAX_VECTOR_SIZE);
            std::fill_n(output_array.get(), State::MAX_VECTOR_SIZE, 1);
            _output_registry[_output_registry_size++] = {node->_attribute, std::move(output_array)};
        }
    }

    // Create parent-child pair arrays
    _leaf_pairs = std::make_unique<ParentLeafPair[]>(total_leaf_pairs);
    _non_leaf_pairs = std::make_unique<ParentNonLeafPair[]>(total_non_leaf_pairs);
    _leaf_pair_count = 0;
    _non_leaf_pair_count = 0;

    // Populate parent-child pairs
    for (auto* parent_node: _ordered_nodes) {
        if (!parent_node->_children.empty()) {
            uint64_t* parent_output = nullptr;
            for (auto i = 0; i < _output_registry_size; i++) {
                auto& [attr, output] = _output_registry[i];
                if (attr == parent_node->_attribute) {
                    parent_output = output.get();
                    break;
                }
            }
            assert(parent_output != nullptr);
            for (const auto& child: parent_node->_children) {
                if (child->_children.empty()) {
                    auto& pair = _leaf_pairs[_leaf_pair_count++];
                    pair.parent_attr = parent_node->_attribute;
                    pair.child_attr = child->_attribute;
                    pair.parent = parent_node;
                    pair.child = child.get();
                    pair.parent_output = parent_output;
                } else {
                    auto& pair = _non_leaf_pairs[_non_leaf_pair_count++];
                    pair.parent_attr = parent_node->_attribute;
                    pair.child_attr = child->_attribute;
                    pair.parent = parent_node;
                    pair.child = child.get();
                    pair.parent_output = parent_output;
                    for (auto i = 0; i < _output_registry_size; i++) {
                        auto& [attr, output] = _output_registry[i];
                        if (strcmp(attr.c_str(), child->_attribute.c_str()) == 0) {
                            pair.child_output = output.get();
                            break;
                        }
                    }
                    assert(pair.child != nullptr);
                    assert(pair.child_output != nullptr);
                }
            }
        }
    }

    // Set the root output pointer
    for (auto i = 0; i < _output_registry_size; i++) {
        auto& [attr, output] = _output_registry[i];
        if (strcmp(attr.c_str(), _ftree->_attribute.c_str()) == 0) {
            _root_output = output.get();
            break;
        }
    }

    _ftree_iterator = std::make_unique<FTreeIterator>();
    _ftree_iterator->init(schema);
    _ftree_output = std::make_unique<uint64_t[]>(_ftree_iterator->tuple_size());
}

std::vector<std::vector<uint64_t>> SinkPacked::get_itr_values() { return _ftree_iterator->get_values(); }
uint64_t SinkPacked::get_itr_values_size() { return _ftree_iterator->get_num_output_tuples(); }

void SinkPacked::execute() {
    // static uint64_t itr_cnt = -1, sink_cnt = -1;
    // if (itr_cnt == -1) {
    //     itr_cnt = 0;
    // }
    // if (sink_cnt == -1) {
    //     sink_cnt = 0;
    // }
    //
    // // need to set the correct state for the iterators before we start processing
    // _ftree_iterator->initialize_iterators();
    // auto prev_itr_cnt = itr_cnt;
    // while (_ftree_iterator->next(_ftree_output.get())){}
    // auto new_itr_cnt = _ftree_iterator->_num_output_tuples;
    // auto curr_itr_cnt = new_itr_cnt - prev_itr_cnt;
    // itr_cnt = curr_itr_cnt;

    num_exec_call++;
    for (auto idx = 0; idx < _output_registry_size; idx++) {
        auto& [_, output] = _output_registry[idx];
        std::fill_n(output.get(), State::MAX_VECTOR_SIZE, 1);
    }
    for (size_t i = 0; i < _leaf_pair_count; i++) {
        process_parent_leaf_pair(_leaf_pairs[i]);
    }
    for (size_t i = 0; i < _non_leaf_pair_count; i++) {
        process_parent_non_leaf_pair(_non_leaf_pairs[i]);
    }

    // Calculate final output tuples
    _root_state = _ftree->_value->state;
    _root_start_idx = GET_START_POS(*_root_state);
    _root_end_idx = GET_END_POS(*_root_state);
    // auto prev_sink_cnt = sink_cnt;
    for (auto i = _root_start_idx; i <= _root_end_idx; i++) {
        _num_output_tuples += _root_output[i];
    }
    // auto new_sink_cnt = _num_output_tuples;
    // auto curr_sink_cnt = new_sink_cnt - prev_sink_cnt;
    // sink_cnt = curr_sink_cnt;

    // Only print debug info when counts mismatch
    // if (curr_itr_cnt != curr_sink_cnt)
    // {
    //     // Print iterator per-position counts
    //     _ftree_iterator->print_debug_counts();
    //
    //     // Print sink per-vector tuple counts using top-down propagation
    //     print_debug_counts();
    //
    //     // Run merged analysis for detailed comparison
    //     SinkItrMerged merged_analyzer;
    //     std::cout << "\n Sink output count: " << curr_sink_cnt << ", Iterator output count: " << curr_itr_cnt << std::endl;
    //     merged_analyzer.init(_schema, _ftree, _ordered_nodes, _ftree_iterator.get());
    //     merged_analyzer.run_merged_analysis();
    //
    //     // Reset and replay with debug output
    //     // std::cout << "\n=== Replaying with debug... ===" << std::endl;
    //     _ftree_iterator->reset();
    //     uint64_t debug_count = 0;
    //     const uint64_t max_debug_tuples = 100;
    //
    //     while (_ftree_iterator->next_debug(_ftree_output.get())) {
    //         // debug_count++;
    //         // if (debug_count >= max_debug_tuples) {
    //         //     std::cout << "... (stopping debug output after " << max_debug_tuples << " tuples)" << std::endl;
    //         //     break;
    //         // }
    //     }
    //
    //     throw std::runtime_error("ERROR: Iterator and Sink tuple counts do not match!");
    // }

    for (auto idx = 0; idx < _output_registry_size; idx++) {
        auto& [_, output] = _output_registry[idx];
        std::fill_n(output.get(), State::MAX_VECTOR_SIZE, 1);
    }
}

void SinkPacked::print_debug_counts() {
    std::cout << "\n=== SINK per-vector tuple counts ===" << std::endl;

    // Get column ordering from iterator
    auto column_ordering = _ftree_iterator->get_attribute_ordering();
    size_t num_nodes = column_ordering.size();

    // Create a debug count array for each node (indexed by column ordering)
    std::vector<std::unique_ptr<uint64_t[]>> debug_counts(num_nodes);
    std::vector<FactorizedTreeElement*> nodes(num_nodes);

    for (size_t i = 0; i < num_nodes; i++) {
        debug_counts[i] = std::make_unique<uint64_t[]>(State::MAX_VECTOR_SIZE);
        std::fill_n(debug_counts[i].get(), State::MAX_VECTOR_SIZE, 0);
        nodes[i] = _ftree->find_node_by_attribute(column_ordering[i]);
    }

    // Step 1: Initialize root with its output registry values
    {
        FactorizedTreeElement* root = nodes[0];
        const auto* state = root->_value->state;
        const auto start = GET_START_POS(*state);
        const auto end = GET_END_POS(*state);
        const auto* selector = &state->selector;

        uint64_t* root_output = nullptr;
        for (size_t idx = 0; idx < _output_registry_size; idx++) {
            if (_output_registry[idx].first == root->_attribute) {
                root_output = _output_registry[idx].second.get();
                break;
            }
        }

        if (root_output) {
            for (auto pos = start; pos <= end; pos++) {
                if (TEST_BIT(*selector, pos)) { debug_counts[0][pos] = root_output[pos]; }
            }
        }
    }

    // Step 2: Propagate counts top-down through the tree
    // For each non-leaf node, propagate to its children
    for (size_t i = 0; i < num_nodes; i++) {
        FactorizedTreeElement* parent = nodes[i];
        if (parent->_children.empty()) continue;

        const auto* parent_state = parent->_value->state;
        const auto* parent_selector = &parent_state->selector;
        const auto parent_start = GET_START_POS(*parent_state);
        const auto parent_end = GET_END_POS(*parent_state);

        // Find parent's output registry
        uint64_t* parent_output = nullptr;
        for (size_t idx = 0; idx < _output_registry_size; idx++) {
            if (_output_registry[idx].first == parent->_attribute) {
                parent_output = _output_registry[idx].second.get();
                break;
            }
        }

        for (const auto& child_ptr: parent->_children) {
            FactorizedTreeElement* child = child_ptr.get();

            size_t child_idx = SIZE_MAX;
            for (size_t j = 0; j < num_nodes; j++) {
                if (nodes[j] == child) {
                    child_idx = j;
                    break;
                }
            }
            if (child_idx == SIZE_MAX) continue;

            const auto* child_state = child->_value->state;
            const auto* child_selector = &child_state->selector;
            const auto* child_offset = child_state->offset;
            bool is_leaf = child->_children.empty();

            // For each parent position
            for (auto pidx = parent_start; pidx <= parent_end; pidx++) {
                if (!TEST_BIT(*parent_selector, pidx)) continue;
                if (parent_output == nullptr || parent_output[pidx] == 0) continue;

                auto child_range_start = child_offset[pidx];
                auto child_range_end = child_offset[pidx + 1];
                uint64_t range_size = child_range_end - child_range_start;

                if (range_size == 0) continue;

                if (is_leaf) {
                    // For leaf nodes: each position's count = parent_output / range_size
                    // because parent_output includes this child's range_size as a factor
                    uint64_t count_per_pos = parent_output[pidx] / range_size;
                    for (auto cidx = child_range_start; cidx < child_range_end; cidx++) {
                        debug_counts[child_idx][cidx] += count_per_pos;
                    }
                } else {
                    // For non-leaf nodes: get from child's output registry
                    uint64_t* child_output = nullptr;
                    for (size_t idx = 0; idx < _output_registry_size; idx++) {
                        if (_output_registry[idx].first == child->_attribute) {
                            child_output = _output_registry[idx].second.get();
                            break;
                        }
                    }

                    if (child_output) {
                        // Sum child outputs in range to get the divisor
                        uint64_t child_sum = 0;
                        for (auto cidx = child_range_start; cidx < child_range_end; cidx++) {
                            if (TEST_BIT(*child_selector, cidx)) { child_sum += child_output[cidx]; }
                        }

                        if (child_sum > 0) {
                            // Each child position gets: parent_output * (child_output[cidx] / child_sum)
                            for (auto cidx = child_range_start; cidx < child_range_end; cidx++) {
                                if (TEST_BIT(*child_selector, cidx)) {
                                    debug_counts[child_idx][cidx] +=
                                            (parent_output[pidx] * child_output[cidx]) / child_sum;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Step 3: Print the counts for each node
    for (size_t i = 0; i < num_nodes; i++) {
        FactorizedTreeElement* node = nodes[i];
        const auto* state = node->_value->state;
        const auto start = GET_START_POS(*state);
        const auto end = GET_END_POS(*state);
        const auto* selector = &state->selector;

        uint64_t total = 0;
        std::cout << "  " << node->_attribute << " [" << start << "," << end << "]: ";

        for (auto pos = start; pos <= end; pos++) {
            if (TEST_BIT(*selector, pos) && debug_counts[i][pos] > 0) {
                std::cout << "pos" << pos << "=" << debug_counts[i][pos] << " ";
                total += debug_counts[i][pos];
            }
        }
        std::cout << " | total=" << total << std::endl;
    }
}

void process_parent_leaf_pair(const ParentLeafPair& pair) {
    const State* parent_state = pair.parent->_value->state;
    const State* child_state = pair.child->_value->state;
    const BitMask<State::MAX_VECTOR_SIZE>* parent_selector = &parent_state->selector;
    const BitMask<State::MAX_VECTOR_SIZE>* child_selector = &child_state->selector;

    const int32_t parent_start = GET_START_POS(*parent_state);
    const int32_t parent_end = GET_END_POS(*parent_state);
    const int32_t child_start = GET_START_POS(*child_state);
    const int32_t child_end = GET_END_POS(*child_state);
    constexpr auto u64_max = std::numeric_limits<uint64_t>::max();

    for (int32_t pidx = parent_start; pidx <= parent_end; pidx++) {
        // Skip if parent selector bit is not set
        if (!TEST_BIT(*parent_selector, pidx)) {
            pair.parent_output[pidx] = 0;
            continue;
        }

        // Get RLE range for this parent position
        int32_t range_start;
        int32_t range_end;
        if (child_state == parent_state) {
            // Identity RLE: child positions == parent positions
            range_start = pidx;
            range_end = pidx;
        } else {
            const uint16_t* child_offset = child_state->offset;
            range_start = static_cast<int32_t>(child_offset[pidx]);
            range_end = static_cast<int32_t>(child_offset[pidx + 1]) - 1;
        }

        // Intersect with child's state range
        const int32_t effective_start = std::max(child_start, range_start);
        const int32_t effective_end = std::min(child_end, range_end);

        // Count valid positions by checking child selector bits
        uint64_t child_count = 0;
        for (int32_t cidx = effective_start; cidx <= effective_end; cidx++) {
            child_count += TEST_BIT(*child_selector, cidx) ? 1 : 0;
        }

        // Check for overflow before multiplication
        const uint64_t current_val = pair.parent_output[pidx];
        assert(current_val * child_count <= u64_max);

        pair.parent_output[pidx] = current_val * child_count;
    }
}


void process_parent_non_leaf_pair(const ParentNonLeafPair& pair) {
    const State* parent_state = pair.parent->_value->state;
    const State* child_state = pair.child->_value->state;
    const BitMask<State::MAX_VECTOR_SIZE>* parent_selector = &parent_state->selector;
    const BitMask<State::MAX_VECTOR_SIZE>* child_selector = &child_state->selector;
    const uint16_t* child_offset = child_state->offset;

    const int32_t parent_start = GET_START_POS(*parent_state);
    const int32_t parent_end = GET_END_POS(*parent_state);
    const int32_t child_start = GET_START_POS(*child_state);
    const int32_t child_end = GET_END_POS(*child_state);
    constexpr auto u64_max = std::numeric_limits<uint64_t>::max();

    // Process first position (parent_start) - needs intersection with child state range
    {
        const int32_t pidx = parent_start;
        int32_t range_start;
        int32_t range_end;
        if (child_state == parent_state) {
            // Identity RLE: child positions == parent positions
            range_start = pidx;
            range_end = pidx;
        } else {
            const uint16_t* child_offset = child_state->offset;
            range_start = static_cast<int32_t>(child_offset[pidx]);
            range_end = static_cast<int32_t>(child_offset[pidx + 1]) - 1;
        }

        const int32_t effective_start = std::max(child_start, range_start);
        const int32_t effective_end = std::min(child_end, range_end);

        uint64_t child_sum = 0;
        for (int32_t cidx = effective_start; cidx <= effective_end; cidx++) {
            if (TEST_BIT(*child_selector, cidx)) {
                child_sum += pair.child_output[cidx];
                assert(child_sum <= u64_max);
            }
        }

        const uint64_t current_val = pair.parent_output[pidx];
        assert(current_val * child_sum <= u64_max);
        pair.parent_output[pidx] = current_val * child_sum;
    }

    // Process middle positions - RLE range is fully within child state range
    for (int32_t pidx = parent_start + 1; pidx < parent_end; pidx++) {
        if (!TEST_BIT(*parent_selector, pidx)) {
            pair.parent_output[pidx] = 0;
            continue;
        }

        int32_t range_start;
        int32_t range_end;
        if (child_state == parent_state) {
            // Identity RLE: child positions == parent positions
            range_start = pidx;
            range_end = pidx;
        } else {
            const uint16_t* child_offset = child_state->offset;
            range_start = static_cast<int32_t>(child_offset[pidx]);
            range_end = static_cast<int32_t>(child_offset[pidx + 1]) - 1;
        }

        uint64_t child_sum = 0;
        for (int32_t cidx = range_start; cidx <= range_end; cidx++) {
            if (TEST_BIT(*child_selector, cidx)) {
                child_sum += pair.child_output[cidx];
                assert(child_sum <= u64_max);
            }
        }

        const uint64_t current_val = pair.parent_output[pidx];
        assert(current_val * child_sum <= u64_max);
        pair.parent_output[pidx] = current_val * child_sum;
    }

    // Process last position (parent_end) - needs intersection with child state range
    if (parent_start != parent_end) {
        const int32_t pidx = parent_end;
        int32_t range_start;
        int32_t range_end;
        if (child_state == parent_state) {
            // Identity RLE: child positions == parent positions
            range_start = pidx;
            range_end = pidx;
        } else {
            const uint16_t* child_offset = child_state->offset;
            range_start = static_cast<int32_t>(child_offset[pidx]);
            range_end = static_cast<int32_t>(child_offset[pidx + 1]) - 1;
        }

        const int32_t effective_start = std::max(child_start, range_start);
        const int32_t effective_end = std::min(child_end, range_end);

        uint64_t child_sum = 0;
        for (int32_t cidx = effective_start; cidx <= effective_end; cidx++) {
            if (TEST_BIT(*child_selector, cidx)) {
                child_sum += pair.child_output[cidx];
                assert(child_sum <= u64_max);
            }
        }

        const uint64_t current_val = pair.parent_output[pidx];
        assert(current_val * child_sum <= u64_max);
        pair.parent_output[pidx] = current_val * child_sum;
    }
}

static void collect_nodes_by_level(FactorizedTreeElement* node, int depth,
                                   std::vector<std::vector<FactorizedTreeElement*>>& levels) {
    if (!node) return;

    if (depth >= levels.size()) { levels.resize(depth + 1); }

    if (!node->_children.empty()) { levels[depth].push_back(node); }

    for (const auto& child: node->_children) {
        collect_nodes_by_level(child.get(), depth + 1, levels);
    }
}

static std::vector<FactorizedTreeElement*> get_topological_order(const std::shared_ptr<FactorizedTreeElement>& root) {
    std::vector<FactorizedTreeElement*> result;
    std::vector<std::vector<FactorizedTreeElement*>> levels;

    collect_nodes_by_level(root.get(), 0, levels);

    for (int i = levels.size() - 1; i >= 0; i--) {
        for (auto& node: levels[i]) {
            result.push_back(node);
        }
    }

    return result;
}

}// namespace ffx