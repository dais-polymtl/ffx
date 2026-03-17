#include "../include/factorized_ftree/ftree_batch_iterator.hpp"
#include "../include/factorized_ftree/ftree_ancestor_finder.hpp"
#include "../include/vector/bitmask.hpp"
#include "../include/vector/state.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ffx {

// Bitmask-jump helper: find first set bit in [range_start, range_end] of a selector.
// Returns the bit position, or -1 if no set bit exists in that range.
static inline int32_t find_first_set_bit_in_range(const BitMask<State::MAX_VECTOR_SIZE>* __restrict__ selector,
                                                  int32_t range_start, int32_t range_end) {
    if (__builtin_expect(range_start > range_end, 0)) return -1;
    const uint64_t* __restrict__ bits = selector->bits;
    const size_t start_block = static_cast<size_t>(range_start) >> 6;
    const size_t end_block = static_cast<size_t>(range_end) >> 6;
    const size_t start_bit = static_cast<size_t>(range_start) & 63;
    const size_t end_bit = static_cast<size_t>(range_end) & 63;
    const uint64_t end_mask = (end_bit == 63) ? ~0ULL : ((1ULL << (end_bit + 1)) - 1);

    for (size_t block = start_block; block <= end_block; ++block) {
        uint64_t block_val = bits[block];
        if (__builtin_expect(block_val == 0, 0)) continue;
        const uint64_t smask = (block == start_block) ? (~0ULL << start_bit) : ~0ULL;
        const uint64_t emask = (block == end_block) ? end_mask : ~0ULL;
        block_val &= smask & emask;
        if (__builtin_expect(block_val != 0, 1)) {
            return static_cast<int32_t>((block << 6) | static_cast<size_t>(__builtin_ctzll(block_val)));
        }
    }
    return -1;
}

namespace {
// Identity offset: offset[i]=i, offset[i+1]=i+1 => range [i,i]. Used when child shares parent state.
alignas(64) uint16_t g_identity_offset[State::MAX_VECTOR_SIZE + 1];
bool g_identity_offset_init = false;
void init_identity_offset() {
    if (g_identity_offset_init) return;
    for (int i = 0; i <= State::MAX_VECTOR_SIZE; ++i)
        g_identity_offset[i] = static_cast<uint16_t>(i);
    g_identity_offset_init = true;
}
}// namespace

FTreeBatchIterator::FTreeBatchIterator(const std::unordered_map<std::string, size_t>& required_attributes)
    : Operator(), _required_attributes(required_attributes) {
    assert(!_required_attributes.empty() && "required_attributes map cannot be empty");
    for (const auto& [attr, cap]: _required_attributes) {
        (void)attr;
        assert(cap != 0 && "Buffer size for required attribute cannot be 0");
    }
}

void FTreeBatchIterator::find_leaf_attributes(FactorizedTreeElement* node, std::vector<std::string>& out) {
    if (node == nullptr) return;
    if (node->_children.empty()) {
        out.push_back(node->_attribute);
        return;
    }
    for (auto& child: node->_children) {
        find_leaf_attributes(child.get(), out);
    }
}

namespace {
void collect_nodes_dfs(FactorizedTreeElement* node, std::vector<FactorizedTreeElement*>& out,
                      std::unordered_map<FactorizedTreeElement*, size_t>& node_to_idx) {
    if (node == nullptr) return;
    size_t idx = out.size();
    node_to_idx[node] = idx;
    out.push_back(node);
    for (auto& child: node->_children) {
        collect_nodes_dfs(child.get(), out, node_to_idx);
    }
}
}// namespace

void FTreeBatchIterator::init(Schema* schema) {
    assert(schema != nullptr);
    assert(schema->root != nullptr);
    assert(schema->column_ordering != nullptr);

    _root = schema->root;
    _is_valid = false;

    const auto& column_ordering = *schema->column_ordering;

    // Collect all attributes present in the current ftree.
    std::unordered_set<std::string> ftree_attrs;
    {
        std::vector<FactorizedTreeElement*> stack = {_root.get()};
        while (!stack.empty()) {
            auto* n = stack.back();
            stack.pop_back();
            ftree_attrs.insert(n->_attribute);
            for (auto& child : n->_children) stack.push_back(child.get());
        }
    }

    // Project the ENTIRE ftree — all nodes present at this point in the pipeline.
    // Downstream operators may add more nodes later, but we capture everything
    // that exists now so the reconstructor can preserve non-Steiner branches.
    std::vector<std::string> actual_columns;
    std::vector<FactorizedTreeElement*> projected_nodes;
    for (const auto& col: column_ordering) {
        if (col == "_cd") continue;
        if (ftree_attrs.count(col)) {
            auto* node = _root->find_node_by_attribute(col);
            assert(node != nullptr);
            actual_columns.push_back(col);
            projected_nodes.push_back(node);
        }
    }
    // Also add ftree nodes not in column_ordering (e.g., structural ancestors).
    for (const auto& attr : ftree_attrs) {
        bool already = false;
        for (const auto& col : actual_columns) {
            if (col == attr) { already = true; break; }
        }
        if (!already) {
            auto* node = _root->find_node_by_attribute(attr);
            assert(node != nullptr);
            actual_columns.push_back(attr);
            projected_nodes.push_back(node);
        }
    }
    assert(!actual_columns.empty());

    _num_attributes = actual_columns.size();
    _iterators = std::make_unique<SimpleLocalIterator[]>(_num_attributes);
    _nodes = std::make_unique<FactorizedTreeElement*[]>(_num_attributes);
    _is_leaf = std::make_unique<bool[]>(_num_attributes);
    _node_buffer = std::make_unique<std::unique_ptr<uint64_t[]>[]>(_num_attributes);
    _node_pos = std::make_unique<std::unique_ptr<int32_t[]>[]>(_num_attributes);
    _node_capacity = std::make_unique<size_t[]>(_num_attributes);
    _output_counts = std::make_unique<size_t[]>(_num_attributes);
    _children_indices = std::make_unique<std::unique_ptr<size_t[]>[]>(_num_attributes);
    _children_counts = std::make_unique<size_t[]>(_num_attributes);

    for (size_t i = 0; i < _num_attributes; ++i) {
        _nodes[i] = projected_nodes[i];
        _children_indices[i] = std::make_unique<size_t[]>(_num_attributes);
        _children_counts[i] = 0;
    }

    size_t num_bridged = 0;
    for (size_t i = 0; i < _num_attributes; ++i) {
        auto* node = _nodes[i];
        if (node->_parent == nullptr) continue;
        // Check if direct parent is in projected tree
        bool direct = false;
        for (size_t j = 0; j < _num_attributes; ++j) {
            if (_nodes[j] == node->_parent) {
                direct = true;
                break;
            }
        }
        if (!direct) num_bridged++;
    }
    _num_custom_offsets = num_bridged;
    _custom_offsets = std::make_unique<std::unique_ptr<uint16_t[]>[]>(num_bridged);
    _custom_offset_sizes = std::make_unique<size_t[]>(num_bridged);

    size_t bridged_idx = 0;
    for (size_t i = 0; i < _num_attributes; ++i) {
        auto& itr = _iterators[i];
        auto* node = _nodes[i];

        itr.selector = nullptr;
        itr.values = nullptr;
        itr.offset = nullptr;
        itr.parent_idx = -1;

        // Walk up node->_parent to find nearest projected ancestor
        auto* curr_parent = node->_parent;
        FactorizedTreeElement* projected_parent = nullptr;
        while (curr_parent != nullptr) {
            for (size_t j = 0; j < _num_attributes; ++j) {
                if (_nodes[j] == curr_parent) {
                    itr.parent_idx = static_cast<int32_t>(j);
                    projected_parent = curr_parent;
                    break;
                }
            }
            if (itr.parent_idx != -1) break;
            curr_parent = curr_parent->_parent;
        }

        if (itr.parent_idx == -1) {
            itr.offset = nullptr;// Root in projected tree
        } else {
            // Build state path from projected parent to this node
            std::vector<const State*> path;
            auto* path_node = node;
            while (path_node != projected_parent) {
                path.push_back(path_node->_value->state);
                path_node = path_node->_parent;
                assert(path_node != nullptr);
            }
            path.push_back(projected_parent->_value->state);
            std::reverse(path.begin(), path.end());
            // path = [parent_state, ..., child_state]

            // Remove duplicate (shared) states
            std::vector<const State*> unique_path;
            unique_path.push_back(path[0]);
            for (size_t k = 1; k < path.size(); ++k) {
                if (path[k] != path[k - 1]) unique_path.push_back(path[k]);
            }

            if (unique_path.size() == 1) {
                init_identity_offset();
                itr.offset = g_identity_offset;
            } else if (unique_path.size() == 2 && node->_parent == projected_parent) {
                // Direct parent-child in original tree
                init_identity_offset();
                itr.offset = (unique_path[0] == unique_path[1]) ? g_identity_offset : node->_value->state->offset;
            } else {
                // Indirect — use FtreeAncestorFinder
                const State* anc_state = unique_path.front();
                const State* desc_state = unique_path.back();

                FtreeAncestorFinder finder(unique_path.data(), unique_path.size());
                std::vector<uint32_t> temp_map(State::MAX_VECTOR_SIZE, 0xFFFFFFFF);

                const int32_t start_a = GET_START_POS(*anc_state);
                const int32_t end_a = GET_END_POS(*anc_state);
                const int32_t start_d = GET_START_POS(*desc_state);
                const int32_t end_d = GET_END_POS(*desc_state);

                finder.process(temp_map.data(), start_a, end_a, start_d, end_d);

                std::vector<uint16_t> bridged(static_cast<size_t>(end_a) + 2, 0);
                uint32_t current_d = static_cast<uint32_t>(start_d);

                for (uint32_t a = static_cast<uint32_t>(start_a); a <= static_cast<uint32_t>(end_a); ++a) {
                    while (current_d <= static_cast<uint32_t>(end_d) &&
                           (temp_map[current_d] == 0xFFFFFFFF || temp_map[current_d] < a))
                        current_d++;
                    bridged[a] = static_cast<uint16_t>(current_d);
                    while (current_d <= static_cast<uint32_t>(end_d) && temp_map[current_d] == a)
                        current_d++;
                }
                bridged[static_cast<size_t>(end_a) + 1] = static_cast<uint16_t>(current_d);

                const size_t bridged_size = static_cast<size_t>(end_a) + 2;
                _custom_offsets[bridged_idx] = std::make_unique<uint16_t[]>(bridged_size);
                std::memcpy(_custom_offsets[bridged_idx].get(), bridged.data(), bridged_size * sizeof(uint16_t));
                _custom_offset_sizes[bridged_idx] = bridged_size;
                itr.offset = _custom_offsets[bridged_idx].get();
                bridged_idx++;
            }
        }
    }

    for (size_t i = 0; i < _num_attributes; ++i)
        _children_counts[i] = 0;
    for (size_t i = 0; i < _num_attributes; ++i) {
        int32_t p = _iterators[i].parent_idx;
        if (p >= 0) {
            _children_indices[p][_children_counts[p]] = i;
            _children_counts[p]++;
        }
    }

    for (size_t i = 0; i < _num_attributes; ++i) {
        _is_leaf[i] = (_children_counts[i] == 0);
        // Non-required Steiner nodes (e.g., ancestors) are internal routing
        // attributes; use a safe default capacity for them.
        size_t cap = State::MAX_VECTOR_SIZE;
        if (auto it = _required_attributes.find(actual_columns[i]); it != _required_attributes.end()) {
            cap = it->second;
        }

        _node_buffer[i] = std::make_unique<uint64_t[]>(cap);
        _node_pos[i] = std::make_unique<int32_t[]>(cap);
        _node_capacity[i] = cap;
        _output_counts[i] = 0;
    }

    _node_offset = std::make_unique<std::unique_ptr<uint32_t[]>[]>(_num_attributes);
    for (size_t i = 0; i < _num_attributes; ++i) {
        int32_t p = _iterators[i].parent_idx;
        size_t p_cap = (p >= 0) ? _node_capacity[p] : 1;
        _node_offset[i] = std::make_unique<uint32_t[]>(p_cap + 1);
    }

    build_full_tree_structure();

    // Cache original state/values pointers so the iterator works even if the
    // reconstructor later modifies _value on the original FactorizedTreeElement nodes.
    _cached_states = std::make_unique<const State*[]>(_num_attributes);
    _cached_values = std::make_unique<uint64_t*[]>(_num_attributes);
    for (size_t i = 0; i < _num_attributes; ++i) {
        _cached_states[i] = _nodes[i]->_value->state;
        _cached_values[i] = _nodes[i]->_value->values;
    }
    _cached_full_tree_states = std::make_unique<const State*[]>(_num_full_tree_nodes);
    for (size_t i = 0; i < _num_full_tree_nodes; ++i) {
        _cached_full_tree_states[i] = _full_tree_nodes[i]->_value->state;
    }
}

void FTreeBatchIterator::build_full_tree_structure() {
    std::unordered_map<FactorizedTreeElement*, size_t> node_to_idx;
    std::vector<FactorizedTreeElement*> tmp_nodes;
    collect_nodes_dfs(_root.get(), tmp_nodes, node_to_idx);
    _num_full_tree_nodes = tmp_nodes.size();

    _full_tree_nodes = std::make_unique<FactorizedTreeElement*[]>(_num_full_tree_nodes);
    _full_tree_parent_idx = std::make_unique<int32_t[]>(_num_full_tree_nodes);
    _full_tree_children_counts = std::make_unique<size_t[]>(_num_full_tree_nodes);
    _full_tree_children_start = std::make_unique<size_t[]>(_num_full_tree_nodes);
    _full_to_projected = std::make_unique<int32_t[]>(_num_full_tree_nodes);
    _projected_to_full = std::make_unique<int32_t[]>(_num_attributes);
    _full_tree_node_bits = std::make_unique<uint64_t[]>(_num_full_tree_nodes * NUM_BLOCKS);
    std::memset(_full_tree_node_bits.get(), 0, _num_full_tree_nodes * NUM_BLOCKS * sizeof(uint64_t));

    for (size_t i = 0; i < _num_full_tree_nodes; ++i) {
        _full_tree_nodes[i] = tmp_nodes[i];
        _full_to_projected[i] = -1;
        _full_tree_children_counts[i] = 0;
    }

    for (size_t i = 0; i < _num_attributes; ++i) {
        _projected_to_full[i] = -1;
    }

    // First pass: set parent indices, count children, map projected nodes.
    for (size_t i = 0; i < _num_full_tree_nodes; ++i) {
        auto* node = _full_tree_nodes[i];
        if (node->_parent == nullptr) {
            _full_tree_parent_idx[i] = -1;
        } else {
            auto it = node_to_idx.find(node->_parent);
            assert(it != node_to_idx.end());
            const size_t parent_idx = it->second;
            _full_tree_parent_idx[i] = static_cast<int32_t>(parent_idx);
            _full_tree_children_counts[parent_idx]++;
        }
        for (size_t j = 0; j < _num_attributes; ++j) {
            if (_nodes[j] == node) {
                _full_to_projected[i] = static_cast<int32_t>(j);
                break;
            }
        }
    }

    // Compute start offsets for flat children array (prefix sum).
    const size_t total_edges = _num_full_tree_nodes > 0 ? _num_full_tree_nodes - 1 : 0;
    _full_tree_children_flat = std::make_unique<size_t[]>(total_edges);
    size_t offset_accum = 0;
    for (size_t i = 0; i < _num_full_tree_nodes; ++i) {
        _full_tree_children_start[i] = offset_accum;
        offset_accum += _full_tree_children_counts[i];
        _full_tree_children_counts[i] = 0;// Reset for second pass
    }

    // Second pass: fill flat children array.
    for (size_t i = 0; i < _num_full_tree_nodes; ++i) {
        const int32_t p = _full_tree_parent_idx[i];
        if (p >= 0) {
            const size_t pi = static_cast<size_t>(p);
            _full_tree_children_flat[_full_tree_children_start[pi] + _full_tree_children_counts[pi]] = i;
            _full_tree_children_counts[pi]++;
        }
    }

    for (size_t i = 0; i < _num_full_tree_nodes; ++i) {
        if (_full_to_projected[i] >= 0) {
            _projected_to_full[static_cast<size_t>(_full_to_projected[i])] = static_cast<int32_t>(i);
        }
    }

    _full_tree_has_projected = std::make_unique<bool[]>(_num_full_tree_nodes);
    for (size_t i = 0; i < _num_full_tree_nodes; ++i)
        _full_tree_has_projected[i] = false;
    compute_has_projected(0);

    // Compute the Steiner set from required_attributes + same-state expansion.
    compute_steiner_set();

    // Build reverse offset tables: child_pos → parent_pos (O(1) lookup).
    _full_tree_reverse_offset = std::make_unique<std::unique_ptr<int16_t[]>[]>(_num_full_tree_nodes);
    _full_tree_identity_parent = std::make_unique<bool[]>(_num_full_tree_nodes);
    for (size_t i = 0; i < _num_full_tree_nodes; ++i) {
        if (_full_tree_parent_idx[i] < 0) {
            _full_tree_reverse_offset[i] = nullptr;
            _full_tree_identity_parent[i] = false;
            continue;
        }
        auto* child_node = _full_tree_nodes[i];
        auto* parent_node = _full_tree_nodes[static_cast<size_t>(_full_tree_parent_idx[i])];
        const auto* child_state = child_node->_value->state;
        const auto* parent_state = parent_node->_value->state;
        const int32_t child_end = GET_END_POS(*child_state);
        _full_tree_reverse_offset[i] = std::make_unique<int16_t[]>(static_cast<size_t>(child_end) + 1);
        int16_t* rev = _full_tree_reverse_offset[i].get();

        _full_tree_identity_parent[i] = (parent_state == child_state);
        if (parent_state == child_state) {
            // Identity: child_pos == parent_pos
            for (int32_t p = 0; p <= child_end; ++p)
                rev[p] = static_cast<int16_t>(p);
        } else {
            const uint16_t* offset = child_state->offset;
            const int32_t parent_start = GET_START_POS(*parent_state);
            const int32_t parent_end = GET_END_POS(*parent_state);
            std::memset(rev, -1, (static_cast<size_t>(child_end) + 1) * sizeof(int16_t));
            for (int32_t p = parent_start; p <= parent_end; ++p) {
                const uint32_t rs = offset[p];
                const uint32_t re = offset[p + 1];
                for (uint32_t c = rs; c < re; ++c) {
                    if (static_cast<int32_t>(c) <= child_end)
                        rev[c] = static_cast<int16_t>(p);
                }
            }
        }
    }
}

void FTreeBatchIterator::compute_steiner_set() {
    _steiner_full_indices.clear();

    // Step 1: Trace each required attribute to the root.
    for (const auto& [attr, _] : _required_attributes) {
        int32_t found = -1;
        for (size_t i = 0; i < _num_full_tree_nodes; ++i) {
            if (_full_tree_nodes[i]->_attribute == attr) {
                found = static_cast<int32_t>(i);
                break;
            }
        }
        if (found < 0) continue;
        int32_t cur = found;
        while (cur >= 0) {
            _steiner_full_indices.insert(static_cast<size_t>(cur));
            cur = _full_tree_parent_idx[cur];
        }
    }

    // Step 2: Expand with same-state children.
    // For each Steiner node, if a child shares the same state (datachunk),
    // add it to the Steiner set. Repeat until no new nodes are added.
    bool changed = true;
    while (changed) {
        changed = false;
        for (size_t fi : std::vector<size_t>(_steiner_full_indices.begin(), _steiner_full_indices.end())) {
            const auto* node_state = _full_tree_nodes[fi]->_value->state;
            const size_t start = _full_tree_children_start[fi];
            const size_t count = _full_tree_children_counts[fi];
            for (size_t k = 0; k < count; ++k) {
                const size_t child = _full_tree_children_flat[start + k];
                if (_steiner_full_indices.count(child)) continue;
                if (_full_tree_nodes[child]->_value->state == node_state) {
                    _steiner_full_indices.insert(child);
                    changed = true;
                }
            }
        }
    }
}

bool FTreeBatchIterator::compute_has_projected(size_t full_idx) {
    bool has = (_full_to_projected[full_idx] >= 0);
    for (size_t k = 0; k < _full_tree_children_counts[full_idx]; ++k) {
        has |= compute_has_projected(_full_tree_children_flat[_full_tree_children_start[full_idx] + k]);
    }
    _full_tree_has_projected[full_idx] = has;
    return has;
}


void FTreeBatchIterator::clear_full_tree_subtree_counts(size_t full_idx) {
    std::memset(&_full_tree_node_bits[full_idx * NUM_BLOCKS], 0, NUM_BLOCKS * sizeof(uint64_t));
    for (size_t k = 0; k < _full_tree_children_counts[full_idx]; ++k) {
        clear_full_tree_subtree_counts(_full_tree_children_flat[_full_tree_children_start[full_idx] + k]);
    }
}

void FTreeBatchIterator::append_full_tree_positions(size_t full_idx, int32_t pos) {
    int32_t current_idx = static_cast<int32_t>(full_idx);
    int32_t current_pos = pos;
    while (current_idx >= 0) {
        const size_t idx = static_cast<size_t>(current_idx);
        if (!_full_tree_has_projected[idx]) break;
        _full_tree_node_bits[idx * NUM_BLOCKS + (static_cast<size_t>(current_pos) >> 6)] |=
                1ULL << (current_pos & 63);
        const int32_t parent_idx = _full_tree_parent_idx[idx];
        if (parent_idx < 0) break;
        const int16_t* rev = _full_tree_reverse_offset[idx].get();
        current_pos = static_cast<int32_t>(rev[current_pos]);
        if (current_pos < 0) break;
        current_idx = parent_idx;
    }
}

void FTreeBatchIterator::append_full_tree_block(size_t full_idx, size_t block_idx, uint64_t block_mask) {
    int32_t current_idx = static_cast<int32_t>(full_idx);
    size_t cur_block = block_idx;
    uint64_t cur_mask = block_mask;

    while (current_idx >= 0) {
        const size_t idx = static_cast<size_t>(current_idx);
        if (!_full_tree_has_projected[idx]) return;
        _full_tree_node_bits[idx * NUM_BLOCKS + cur_block] |= cur_mask;

        const int32_t parent_idx = _full_tree_parent_idx[idx];
        if (parent_idx < 0) return;

        if (_full_tree_identity_parent[idx]) {
            current_idx = parent_idx;
            continue;
        }

        // Non-identity edge: map child bits to parent bits via reverse offset,
        // then recursively propagate each non-zero parent block.
        const int16_t* rev = _full_tree_reverse_offset[idx].get();
        uint64_t parent_blocks[NUM_BLOCKS] = {};
        uint64_t tmp = cur_mask;
        while (tmp) {
            const int bit = __builtin_ctzll(tmp);
            const int32_t child_pos = static_cast<int32_t>((cur_block << 6) | static_cast<size_t>(bit));
            const int16_t parent_pos = rev[child_pos];
            if (parent_pos >= 0) {
                parent_blocks[static_cast<size_t>(parent_pos) >> 6] |=
                        1ULL << (parent_pos & 63);
            }
            tmp &= tmp - 1;
        }
        for (size_t b = 0; b < NUM_BLOCKS; ++b) {
            if (parent_blocks[b] != 0) {
                append_full_tree_block(static_cast<size_t>(parent_idx), b, parent_blocks[b]);
            }
        }
        return;
    }
}

void FTreeBatchIterator::initialize_iterators() {
    _is_valid = true;

    auto& root_itr = _iterators[0];
    const auto* root_state = _cached_states[0];
    root_itr.selector = &root_state->selector;
    root_itr.values = _cached_values[0];
    root_itr.valid_start = GET_START_POS(*root_state);
    root_itr.valid_end = GET_END_POS(*root_state);

    const int32_t first_root = find_first_set_bit_in_range(root_itr.selector, root_itr.valid_start, root_itr.valid_end);
    root_itr.current_pos = (first_root >= 0) ? first_root : root_itr.valid_end + 1;
    _is_valid = (first_root >= 0);

    for (size_t i = 1; i < _num_attributes && _is_valid; ++i) {
        auto& itr = _iterators[i];
        const auto* __restrict__ state = _cached_states[i];
        itr.selector = &state->selector;
        itr.values = _cached_values[i];

        const int32_t parent_pos = _iterators[itr.parent_idx].current_pos;
        const auto* __restrict__ rle = itr.offset;
        const uint32_t range_start = rle[parent_pos];
        const uint32_t range_end = rle[parent_pos + 1] - 1;
        const int32_t state_start = GET_START_POS(*state);
        const int32_t state_end = GET_END_POS(*state);

        itr.valid_start = std::max(state_start, static_cast<int32_t>(range_start));
        itr.valid_end = std::min(state_end, static_cast<int32_t>(range_end));
        itr.cached_range_start = itr.valid_start;
        itr.cached_range_end = itr.valid_end;
        itr.cached_parent_pos = parent_pos;
        const int32_t first_child = find_first_set_bit_in_range(itr.selector, itr.valid_start, itr.valid_end);
        itr.current_pos = (first_child >= 0) ? first_child : itr.valid_end + 1;
        _is_valid = (first_child >= 0);
    }

    if (!_is_valid) {
        auto& ritr = _iterators[0];
        while (!_is_valid && ritr.current_pos <= ritr.valid_end) {
            const int32_t next_root = find_first_set_bit_in_range(ritr.selector, ritr.current_pos + 1, ritr.valid_end);
            if (next_root < 0) break;
            ritr.current_pos = next_root;
            _is_valid = true;
            for (size_t i = 1; i < _num_attributes && _is_valid; ++i) {
                _is_valid = reset_iterator_to_start(i);
            }
        }
    }
}

void FTreeBatchIterator::append_node_value(size_t idx) {
    const size_t count = _output_counts[idx];
    _node_buffer[idx][count] = _iterators[idx].values[_iterators[idx].current_pos];
    _node_pos[idx][count] = _iterators[idx].current_pos;

    for (size_t k = 0; k < _children_counts[idx]; ++k) {
        const size_t c = _children_indices[idx][k];
        _node_offset[c][count] = static_cast<uint32_t>(_output_counts[c]);
    }

    _output_counts[idx] = count + 1;

    // Append positions for all full tree nodes on path (required and non-required)
    if (_projected_to_full) {
        const int32_t full_idx = _projected_to_full[idx];
        if (full_idx >= 0) {
            append_full_tree_positions(static_cast<size_t>(full_idx), _iterators[idx].current_pos);
        }
    }
}

__attribute__((always_inline)) inline bool FTreeBatchIterator::reset_iterator_to_start(size_t idx) {
    assert(idx > 0);
    auto& itr = _iterators[idx];
    const int32_t parent_pos = _iterators[itr.parent_idx].current_pos;

    if (__builtin_expect(parent_pos == itr.cached_parent_pos && itr.cached_range_start >= 0, 1)) {
        itr.valid_start = itr.cached_range_start;
        itr.valid_end = itr.cached_range_end;
    } else {
        const auto* __restrict__ state = _cached_states[idx];
        const auto* __restrict__ rle = itr.offset;
        const uint32_t range_start = rle[parent_pos];
        const uint32_t range_end = rle[parent_pos + 1] - 1;
        const int32_t state_start = GET_START_POS(*state);
        const int32_t state_end = GET_END_POS(*state);
        itr.valid_start = std::max(state_start, static_cast<int32_t>(range_start));
        itr.valid_end = std::min(state_end, static_cast<int32_t>(range_end));
        itr.cached_range_start = itr.valid_start;
        itr.cached_range_end = itr.valid_end;
        itr.cached_parent_pos = parent_pos;
    }

    const int32_t first_pos = find_first_set_bit_in_range(itr.selector, itr.valid_start, itr.valid_end);
    itr.current_pos = (first_pos >= 0) ? first_pos : itr.valid_end + 1;
    return (first_pos >= 0);
}

__attribute__((always_inline)) inline bool FTreeBatchIterator::try_advance(size_t idx) {
    auto& itr = _iterators[idx];
    int32_t next_pos;
    if (__builtin_expect(idx == 0, 0)) {
        next_pos = find_next_set_bit_root(itr);
    } else {
        const int32_t parent_pos = _iterators[itr.parent_idx].current_pos;
        next_pos = find_next_set_bit_child(itr, parent_pos);
    }
    if (__builtin_expect(next_pos >= 0, 1)) {
        itr.current_pos = next_pos;
        return true;
    }
    return false;
}

__attribute__((always_inline)) inline int32_t
FTreeBatchIterator::find_next_set_bit_root(const SimpleLocalIterator& itr) const {
    const auto* __restrict__ selector = itr.selector;
    const uint64_t* __restrict__ bits = selector->bits;
    const int32_t pos = (itr.current_pos < itr.valid_start) ? itr.valid_start : itr.current_pos + 1;
    const size_t start_block = pos >> 6;
    const size_t end_block = itr.valid_end >> 6;
    const size_t start_bit = pos & 63;
    const size_t end_bit = itr.valid_end & 63;
    const uint64_t end_mask = (end_bit == 63) ? ~0ULL : ((1ULL << (end_bit + 1)) - 1);

    for (size_t block = start_block; block <= end_block; ++block) {
        uint64_t block_val = bits[block];
        if (__builtin_expect(block_val == 0, 0)) continue;
        const bool is_start = (block == start_block);
        const bool is_end = (block == end_block);
        const uint64_t start_mask = is_start ? (~0ULL << start_bit) : ~0ULL;
        const uint64_t end_mask_val = is_end ? end_mask : ~0ULL;
        block_val &= start_mask & end_mask_val;
        if (__builtin_expect(block_val != 0, 1)) {
            const int bit_pos = __builtin_ctzll(block_val);
            return (block << 6) | bit_pos;
        }
    }
    return -1;
}

__attribute__((always_inline)) inline int32_t
FTreeBatchIterator::find_next_set_bit_child(const SimpleLocalIterator& itr, int32_t parent_pos) const {
    const uint16_t* __restrict__ rle = itr.offset;
    const uint32_t parent_start = rle[parent_pos];
    const uint32_t parent_end = rle[parent_pos + 1] - 1;
    const int32_t allowed_start = std::max<int32_t>(itr.valid_start, static_cast<int32_t>(parent_start));
    const int32_t allowed_end = std::min<int32_t>(itr.valid_end, static_cast<int32_t>(parent_end));
    if (__builtin_expect(allowed_start > allowed_end, 0)) return -1;

    const auto* __restrict__ selector = itr.selector;
    const uint64_t* __restrict__ bits = selector->bits;
    const int32_t pos = (itr.current_pos < allowed_start) ? allowed_start : itr.current_pos + 1;
    if (__builtin_expect(pos > allowed_end, 0)) return -1;

    const size_t start_block = pos >> 6;
    const size_t end_block = allowed_end >> 6;
    const size_t start_bit = pos & 63;
    const size_t end_bit = allowed_end & 63;
    const uint64_t end_mask = (end_bit == 63) ? ~0ULL : ((1ULL << (end_bit + 1)) - 1);

    for (size_t block = start_block; block <= end_block; ++block) {
        uint64_t block_val = bits[block];
        if (__builtin_expect(block_val == 0, 0)) continue;
        const bool is_start = (block == start_block);
        const bool is_end = (block == end_block);
        const uint64_t start_mask = is_start ? (~0ULL << start_bit) : ~0ULL;
        const uint64_t end_mask_val = is_end ? end_mask : ~0ULL;
        block_val &= start_mask & end_mask_val;
        if (__builtin_expect(block_val != 0, 1)) {
            const int bit_pos = __builtin_ctzll(block_val);
            return (block << 6) | bit_pos;
        }
    }
    return -1;
}

void FTreeBatchIterator::finalize_batch_offsets() {
    for (size_t i = 1; i < _num_attributes; ++i) {
        int32_t p = _iterators[i].parent_idx;
        if (p < 0) continue;

        size_t p_count = _output_counts[p];
        if (p_count > 0) {
            _node_offset[i][p_count] = static_cast<uint32_t>(_output_counts[i]);
        } else {
            _node_offset[i][0] = 0;
        }
    }
}

size_t FTreeBatchIterator::count_logical_tuples_rec(const int32_t* flat_pos, const size_t* pos_starts,
                                                    const size_t* pos_counts, size_t full_idx,
                                                    size_t pos_start, size_t pos_end) const {
    if (!_full_tree_has_projected[full_idx]) return 1;
    if (pos_start >= pos_end) return 0;

    const int32_t* positions = flat_pos + pos_starts[full_idx];
    const size_t num_children = _full_tree_children_counts[full_idx];

    if (num_children == 0) return pos_end - pos_start;

    // Per-child cursors for monotonic linear scan.
    // As parent positions increase, child offset ranges are non-overlapping and
    // increasing, so each cursor only moves forward.
    static constexpr size_t MAX_CHILDREN = 64;
    size_t child_cursors[MAX_CHILDREN];
    assert(num_children <= MAX_CHILDREN);
    for (size_t k = 0; k < num_children; ++k)
        child_cursors[k] = 0;

    size_t total = 0;
    for (size_t i = pos_start; i < pos_end; ++i) {
        const int32_t pos = positions[i];
        size_t product = 1;
        for (size_t k = 0; k < num_children; ++k) {
            const size_t c = _full_tree_children_flat[_full_tree_children_start[full_idx] + k];
            if (!_full_tree_has_projected[c]) continue;

            const int32_t* child_pos_arr = flat_pos + pos_starts[c];
            const size_t child_count = pos_counts[c];

            const auto* child_state = _cached_full_tree_states[c];
            int32_t range_lo, range_hi;
            if (_cached_full_tree_states[full_idx] == child_state) {
                range_lo = pos;
                range_hi = pos;
            } else {
                range_lo = static_cast<int32_t>(child_state->offset[pos]);
                range_hi = static_cast<int32_t>(child_state->offset[pos + 1]) - 1;
            }

            while (child_cursors[k] < child_count && child_pos_arr[child_cursors[k]] < range_lo)
                child_cursors[k]++;
            const size_t c_start = child_cursors[k];

            while (child_cursors[k] < child_count && child_pos_arr[child_cursors[k]] <= range_hi)
                child_cursors[k]++;
            const size_t c_end = child_cursors[k];

            product *= count_logical_tuples_rec(flat_pos, pos_starts, pos_counts, c, c_start, c_end);
            if (product == 0) break;
        }
        total += product;
    }
    return total;
}

size_t FTreeBatchIterator::count_logical_tuples() const {
    // Bitsets are already populated during batch filling — just count and extract.
    const size_t n = _num_full_tree_nodes;

    size_t* pos_counts = static_cast<size_t*>(alloca(n * sizeof(size_t)));
    size_t total_positions = 0;
    for (size_t i = 0; i < n; ++i) {
        const uint64_t* bits = &_full_tree_node_bits[i * NUM_BLOCKS];
        size_t unique_count = 0;
        for (size_t b = 0; b < NUM_BLOCKS; ++b)
            unique_count += static_cast<size_t>(__builtin_popcountll(bits[b]));
        pos_counts[i] = unique_count;
        total_positions += unique_count;
    }

    if (total_positions == 0) return 0;

    // Extract sorted unique positions from bitsets.
    int32_t* flat_pos = static_cast<int32_t*>(alloca(total_positions * sizeof(int32_t)));
    size_t* pos_starts = static_cast<size_t*>(alloca(n * sizeof(size_t)));

    size_t offset = 0;
    for (size_t i = 0; i < n; ++i) {
        pos_starts[i] = offset;
        if (pos_counts[i] == 0) continue;
        const uint64_t* bits = &_full_tree_node_bits[i * NUM_BLOCKS];
        for (size_t b = 0; b < NUM_BLOCKS; ++b) {
            uint64_t val = bits[b];
            while (val) {
                flat_pos[offset++] = static_cast<int32_t>((b << 6) | __builtin_ctzll(val));
                val &= val - 1;
            }
        }
    }

    return count_logical_tuples_rec(flat_pos, pos_starts, pos_counts, 0, 0, pos_counts[0]);
}

FTreeBatchIterator::FullTreeInfo FTreeBatchIterator::get_full_tree_info() const {
    FullTreeInfo info;
    info.num_nodes = _num_full_tree_nodes;
    info.nodes = reinterpret_cast<const FactorizedTreeElement* const*>(_full_tree_nodes.get());
    info.parent_idx = _full_tree_parent_idx.get();
    info.full_to_projected = _full_to_projected.get();
    info.projected_to_full = _projected_to_full.get();
    info.children_flat = _full_tree_children_flat.get();
    info.children_start = _full_tree_children_start.get();
    info.children_counts = _full_tree_children_counts.get();
    return info;
}

FTreeBatchIterator::FillStatus FTreeBatchIterator::fill_leaf_window(size_t node_idx) {
    auto& itr = _iterators[node_idx];
    uint64_t* __restrict__ buf = _node_buffer[node_idx].get();
    int32_t* __restrict__ pos_buf = _node_pos[node_idx].get();
    const size_t count_before = _output_counts[node_idx];
    size_t count = count_before;
    const size_t cap = _node_capacity[node_idx];

    if (count >= cap || itr.current_pos > itr.valid_end) {
        if (count >= cap) {
            return (find_first_set_bit_in_range(itr.selector, itr.current_pos, itr.valid_end) >= 0)
                           ? FillStatus::BLOCKED_PARTIAL
                           : FillStatus::BLOCKED_FULL;
        }
        return FillStatus::SUCCESS_EXHAUSTED;
    }

    const uint64_t* __restrict__ values = itr.values;
    const uint64_t* __restrict__ bits = itr.selector->bits;
    int32_t pos = itr.current_pos;
    const int32_t end_pos = itr.valid_end;

    size_t block = pos >> 6;
    const size_t end_block = end_pos >> 6;
    uint64_t mask = static_cast<uint64_t>(~0ULL) << (pos & 63);

    while (count < cap && block <= end_block) {
        uint64_t bval = bits[block] & mask;
        if (block == end_block) {
            size_t end_bit = end_pos & 63;
            if (end_bit < 63) bval &= (1ULL << (end_bit + 1)) - 1;
        }

        // FAST PATH: entire block is all 1s and we have room for 64 values
        if (bval == ~0ULL && count + 64 <= cap) {
            int32_t base_pos = static_cast<int32_t>(block << 6);
            std::memcpy(buf + count, values + base_pos, 64 * sizeof(uint64_t));
            std::iota(pos_buf + count, pos_buf + count + 64, base_pos);
            count += 64;
            pos = base_pos + 64;
            block++;
            mask = ~0ULL;
            continue;
        }

        // SLOW PATH: sparse block, scan bits individually
        while (bval && count < cap) {
            int bit_pos = __builtin_ctzll(bval);
            int32_t global_pos = (block << 6) | bit_pos;

            buf[count] = values[global_pos];
            pos_buf[count] = global_pos;
            count++;

            bval &= (bval - 1);
            pos = global_pos + 1;
        }

        if (count == cap) break;

        block++;
        mask = ~0ULL;
        pos = block << 6;
    }

    itr.current_pos = (pos <= end_pos) ? pos : end_pos + 1;
    _output_counts[node_idx] = count;

    const int32_t full_idx = _projected_to_full[node_idx];
    if (full_idx >= 0) {
        // Batch positions by block for efficient ancestor propagation.
        // pos_buf[count_before..count) contains ascending positions.
        size_t j = count_before;
        while (j < count) {
            const size_t blk = static_cast<size_t>(pos_buf[j]) >> 6;
            uint64_t mask = 0;
            while (j < count && (static_cast<size_t>(pos_buf[j]) >> 6) == blk) {
                mask |= 1ULL << (pos_buf[j] & 63);
                j++;
            }
            append_full_tree_block(static_cast<size_t>(full_idx), blk, mask);
        }
    }

    if (count >= cap) {
        return (find_first_set_bit_in_range(itr.selector, itr.current_pos, itr.valid_end) >= 0)
                       ? FillStatus::BLOCKED_PARTIAL
                       : FillStatus::BLOCKED_FULL;
    }
    return FillStatus::SUCCESS_EXHAUSTED;
}

FTreeBatchIterator::FillStatus FTreeBatchIterator::greedy_fill_subtree(size_t root_node) {
    struct Frame {
        size_t node;
        size_t child_k;
        FillStatus agg;
    };
    // Stack depth bounded by projected tree depth (typically < 10).
    std::array<Frame, 64> stack{};
    int top = 0;
    stack[0] = {root_node, 0, FillStatus::SUCCESS_EXHAUSTED};
    // Respect internal-node capacities (context_column can cap non-leaf nodes like the root).
    if (_output_counts[root_node] >= _node_capacity[root_node]) {
        return FillStatus::BLOCKED_FULL;
    }
    append_node_value(root_node);

    while (top >= 0) {
        auto& f = stack[top];

        // Process next child
        bool child_pushed = false;
        while (f.child_k < _children_counts[f.node]) {
            const size_t c = _children_indices[f.node][f.child_k];
            f.child_k++;
            if (!reset_iterator_to_start(c)) continue;
            if (_is_leaf[c]) {
                f.agg = std::max(f.agg, fill_leaf_window(c));
            } else {
                // Push new frame for non-leaf child
                assert(top + 1 < 64);
                if (_output_counts[c] >= _node_capacity[c]) {
                    f.agg = std::max(f.agg, FillStatus::BLOCKED_FULL);
                    continue;
                }
                stack[++top] = {c, 0, FillStatus::SUCCESS_EXHAUSTED};
                append_node_value(c);
                child_pushed = true;
                break;
            }
        }
        if (child_pushed) continue;

        // All children processed — check status
        if (f.agg == FillStatus::BLOCKED_PARTIAL || f.agg == FillStatus::BLOCKED_FULL) {
            FillStatus result = f.agg;
            top--;
            if (top >= 0) {
                stack[top].agg = std::max(stack[top].agg, result);
                // Continue parent's child loop — already incremented child_k
            }
            continue;
        }

        // Try to advance this node
        if (try_advance(f.node)) {
            if (_output_counts[f.node] >= _node_capacity[f.node]) {
                // Can't append any more values for this internal node in this batch.
                FillStatus result = FillStatus::BLOCKED_FULL;
                top--;
                if (top >= 0) {
                    stack[top].agg = std::max(stack[top].agg, result);
                }
                continue;
            }
            append_node_value(f.node);
            f.child_k = 0;
            f.agg = FillStatus::SUCCESS_EXHAUSTED;
        } else {
            // Exhausted — pop and propagate SUCCESS_EXHAUSTED to parent
            top--;
            // Parent's agg stays as-is (SUCCESS_EXHAUSTED doesn't raise it)
        }
    }

    return FillStatus::SUCCESS_EXHAUSTED;
}

void FTreeBatchIterator::greedy_fill_all() {
    if (_is_leaf[0]) {
        fill_leaf_window(0);
        return;
    }
    greedy_fill_subtree(0);
}

void FTreeBatchIterator::fill_remaining_nodes(size_t start_idx) {
    for (size_t i = start_idx; i < _num_attributes; ++i) {
        if (_output_counts[i] > 0) continue;
        const int32_t p = _iterators[i].parent_idx;
        if (p < 0 || _output_counts[static_cast<size_t>(p)] == 0) continue;
        if (i > 0 && !reset_iterator_to_start(i)) continue;
        if (_is_leaf[i]) {
            fill_leaf_window(i);
        } else {
            greedy_fill_subtree(i);
        }
    }
}

void FTreeBatchIterator::reset() {
    _first_call = true;
    for (size_t i = 0; i < _num_attributes; ++i) {
        _output_counts[i] = 0;
    }
    std::memset(_full_tree_node_bits.get(), 0, _num_full_tree_nodes * NUM_BLOCKS * sizeof(uint64_t));
}

int FTreeBatchIterator::find_rightmost_remaining_leaf() const {
    for (int i = static_cast<int>(_num_attributes) - 1; i >= 0; --i) {
        if (!_is_leaf[i]) continue;
        if (_output_counts[i] < _node_capacity[i]) continue;
        const auto& itr = _iterators[i];
        if (itr.current_pos <= itr.valid_end) return i;
    }
    return -1;
}

bool FTreeBatchIterator::check_any_leaf_has_data() const {
    for (size_t i = 0; i < _num_attributes; ++i) {
        if (_is_leaf[i] && _output_counts[i] > 0) return true;
    }
    return false;
}

void FTreeBatchIterator::cascade_down_from(size_t idx) {
    const int32_t full_idx = _projected_to_full[idx];
    if (full_idx >= 0) {
        clear_full_tree_subtree_counts(static_cast<size_t>(full_idx));
    }
    for (size_t i = idx; i < _num_attributes; ++i) {
        _output_counts[i] = 0;
        int32_t p = _iterators[i].parent_idx;
        if (p >= 0 && _output_counts[p] > 0) {
            for (size_t j = 0; j < _output_counts[p]; ++j) {
                _node_offset[i][j] = 0;
            }
        }
    }

    if (_is_leaf[idx]) {
        fill_leaf_window(idx);
    } else {
        greedy_fill_subtree(idx);
    }
    fill_remaining_nodes(idx + 1);
}

bool FTreeBatchIterator::yield_batch() {
    _num_output_tuples++;
    finalize_batch_offsets();
    return true;
}

bool FTreeBatchIterator::next() {
    if (!_is_valid) return false;

    if (_first_call) {
        _first_call = false;
        greedy_fill_all();
        if (check_any_leaf_has_data()) return yield_batch();
        _is_valid = false;
        return false;
    }

    int skew_idx = find_rightmost_remaining_leaf();
    if (skew_idx >= 0) {
        cascade_down_from(static_cast<size_t>(skew_idx));
        if (check_any_leaf_has_data()) return yield_batch();
    }

    for (int idx = static_cast<int>(_num_attributes) - 1; idx >= 0; idx--) {
        if (_is_leaf[idx]) continue;
        if (!try_advance(idx)) continue;

        cascade_down_from(static_cast<size_t>(idx));
        if (check_any_leaf_has_data()) return yield_batch();
    }

    _is_valid = false;
    return false;
}

void FTreeBatchIterator::debug_print_projected_ftree() const {
    std::cerr << "\n[DEBUG][FTreeBatchIterator] Projected FTree\n";
    std::cerr << "  projected_nodes=" << _num_attributes << "\n";
    for (size_t i = 0; i < _num_attributes; ++i) {
        const auto* node = _nodes[i];
        const int32_t p = _iterators[i].parent_idx;
        std::cerr << "  [" << i << "] attr='" << (node ? node->_attribute : std::string("<null>")) << "'"
                  << " parent=";
        if (p < 0) {
            std::cerr << "<root>";
        } else {
            std::cerr << p << "('" << _nodes[static_cast<size_t>(p)]->_attribute << "')";
        }
        std::cerr << " children={";
        for (size_t k = 0; k < _children_counts[i]; ++k) {
            const size_t c = _children_indices[i][k];
            std::cerr << c;
            if (k + 1 < _children_counts[i]) std::cerr << ",";
        }
        std::cerr << "} leaf=" << (_is_leaf[i] ? "true" : "false")
                  << " cap=" << _node_capacity[i]
                  << " count=" << _output_counts[i] << "\n";
    }
}

void FTreeBatchIterator::debug_print_internal_state() const {
    std::cerr << "\n[DEBUG][FTreeBatchIterator] Internal Full-Tree State\n";
    std::cerr << "  full_nodes=" << _num_full_tree_nodes << " first_call=" << (_first_call ? "true" : "false")
              << " valid=" << (_is_valid ? "true" : "false") << "\n";
    for (size_t i = 0; i < _num_full_tree_nodes; ++i) {
        const auto* node = _full_tree_nodes[i];
        std::cerr << "  [full " << i << "] attr='" << (node ? node->_attribute : std::string("<null>"))
                  << "' parent=" << _full_tree_parent_idx[i]
                  << " projected_idx=" << _full_to_projected[i]
                  << " has_projected=" << (_full_tree_has_projected[i] ? "true" : "false")
                  << " children={";
        const size_t start = _full_tree_children_start[i];
        const size_t cnt = _full_tree_children_counts[i];
        for (size_t k = 0; k < cnt; ++k) {
            std::cerr << _full_tree_children_flat[start + k];
            if (k + 1 < cnt) std::cerr << ",";
        }
        std::cerr << "}";

        const uint64_t* bits = &_full_tree_node_bits[i * NUM_BLOCKS];
        size_t bit_count = 0;
        for (size_t b = 0; b < NUM_BLOCKS; ++b)
            bit_count += static_cast<size_t>(__builtin_popcountll(bits[b]));
        if (bit_count > 0) {
            std::cerr << " pos=[";
            size_t printed = 0;
            for (size_t b = 0; b < NUM_BLOCKS; ++b) {
                uint64_t val = bits[b];
                while (val) {
                    if (printed > 0) std::cerr << ",";
                    std::cerr << ((b << 6) | __builtin_ctzll(val));
                    val &= val - 1;
                    printed++;
                }
            }
            std::cerr << "]";
        }
        std::cerr << "\n";
    }

    std::cerr << "  projected buffers:\n";
    for (size_t i = 0; i < _num_attributes; ++i) {
        std::cerr << "    [" << i << "] attr='" << _nodes[i]->_attribute << "' count=" << _output_counts[i]
                  << " cap=" << _node_capacity[i] << " values=[";
        for (size_t j = 0; j < _output_counts[i]; ++j) {
            std::cerr << _node_buffer[i][j] << "@" << _node_pos[i][j];
            if (j + 1 < _output_counts[i]) std::cerr << ",";
        }
        std::cerr << "]";
        if (_iterators[i].parent_idx >= 0) {
            const size_t p = static_cast<size_t>(_iterators[i].parent_idx);
            const size_t pcount = _output_counts[p];
            std::cerr << " offset=[";
            for (size_t k = 0; k <= pcount; ++k) {
                std::cerr << _node_offset[i][k];
                if (k < pcount) std::cerr << ",";
            }
            std::cerr << "]";
        }
        std::cerr << "\n";
    }
}

}// namespace ffx
