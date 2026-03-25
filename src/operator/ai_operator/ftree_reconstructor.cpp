#include "ai_operator/ftree_reconstructor.hpp"
#include "factorized_ftree/ftree_ancestor_finder.hpp"
#include "query_variable_to_vector.hpp"
#include "schema/schema.hpp"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace ffx {

FTreeReconstructor::FTreeReconstructor(std::string output_attr, std::vector<std::string> attribute_names)
    : _output_attr(std::move(output_attr)), _attribute_names(std::move(attribute_names)), _schema(nullptr),
      _batch_iterator(nullptr), _flush_callback(nullptr), _num_attr(0), _num_output_tuples(0),
      _ancestor_finder(nullptr), _llm_batch(nullptr) {
    assert(!_attribute_names.empty() && "FTreeReconstructor: attribute_names cannot be empty");
}

int32_t FTreeReconstructor::compute_steiner_height(size_t full_idx, const FTreeBatchIterator::FullTreeInfo& info,
                                                   const std::unordered_set<size_t>& steiner,
                                                   std::unordered_map<size_t, int32_t>& memo) const {
    auto it = memo.find(full_idx);
    if (it != memo.end()) return it->second;

    int32_t best_child_height = -1;
    const size_t start = info.children_start[full_idx];
    const size_t count = info.children_counts[full_idx];
    for (size_t k = 0; k < count; ++k) {
        const size_t child = info.children_flat[start + k];
        if (!steiner.count(child)) continue;
        best_child_height = std::max(best_child_height, compute_steiner_height(child, info, steiner, memo));
    }

    const int32_t h = 1 + std::max(0, best_child_height);
    memo[full_idx] = h;
    return h;
}

void FTreeReconstructor::linearize_steiner(size_t full_idx, const FTreeBatchIterator::FullTreeInfo& info,
                                           const std::unordered_set<size_t>& steiner,
                                           std::unordered_map<size_t, int32_t>& height_memo) {
    _chain_full_idx.push_back(full_idx);
    _chain_attrs.push_back(info.nodes[full_idx]->_attribute);

    std::vector<size_t> steiner_children;
    const size_t start = info.children_start[full_idx];
    const size_t count = info.children_counts[full_idx];
    steiner_children.reserve(count);
    for (size_t k = 0; k < count; ++k) {
        const size_t child = info.children_flat[start + k];
        if (steiner.count(child)) steiner_children.push_back(child);
    }

    std::stable_sort(steiner_children.begin(), steiner_children.end(), [&](size_t lhs, size_t rhs) {
        const int32_t hl = compute_steiner_height(lhs, info, steiner, height_memo);
        const int32_t hr = compute_steiner_height(rhs, info, steiner, height_memo);
        if (hl != hr) return hl > hr;
        return lhs < rhs;
    });

    for (size_t child: steiner_children) {
        linearize_steiner(child, info, steiner, height_memo);
    }
}

void FTreeReconstructor::init(Schema* schema, FTreeBatchIterator* batch_iterator,
                              std::function<void()> flush_callback) {
    _schema = schema;
    _batch_iterator = batch_iterator;
    _flush_callback = std::move(flush_callback);
    assert(_schema != nullptr && "FTreeReconstructor: schema is null");
    assert(_schema->map != nullptr && "FTreeReconstructor: schema map is null");
    assert(_schema->root != nullptr && "FTreeReconstructor: schema root is null");
    assert(_schema->column_ordering != nullptr && "FTreeReconstructor: schema column_ordering is null");
    assert(_batch_iterator != nullptr && "FTreeReconstructor: batch_iterator is null");

    _root = _schema->root;
    _num_attr = static_cast<uint32_t>(_attribute_names.size());
    assert(std::find(_attribute_names.begin(), _attribute_names.end(), _output_attr) == _attribute_names.end() &&
           "FTreeReconstructor: output_attr must not be part of input attribute names");

    const auto info = _batch_iterator->get_full_tree_info();
    assert(info.num_nodes > 0 && "FTreeReconstructor: full tree info is empty");

    // Filter _attribute_names against batch iterator's projected set.
    {
        std::vector<std::string> valid_attrs;
        for (const auto& attr: _attribute_names) {
            for (size_t i = 0; i < info.num_nodes; ++i) {
                if (info.nodes[i]->_attribute == attr && info.full_to_projected[i] >= 0) {
                    valid_attrs.push_back(attr);
                    break;
                }
            }
        }
        _attribute_names = valid_attrs;
        _num_attr = static_cast<uint32_t>(_attribute_names.size());
    }

    // ---- Step 1: Read Steiner set from batch iterator ----
    const auto& steiner = _batch_iterator->get_steiner_set();
    assert(!steiner.empty() && "FTreeReconstructor: batch iterator Steiner set is empty");

    size_t steiner_root = 0;
    bool found_root = false;
    for (size_t i = 0; i < info.num_nodes; ++i) {
        if (info.parent_idx[i] < 0 && steiner.count(i)) {
            steiner_root = i;
            found_root = true;
            break;
        }
    }
    assert(found_root && "FTreeReconstructor: Steiner root not found");

    // ---- Step 2: Linearize Steiner set into chain ----
    _chain_attrs.clear();
    _chain_full_idx.clear();

    std::unordered_map<size_t, int32_t> height_memo;
    linearize_steiner(steiner_root, info, steiner, height_memo);

    // Append _llm output at end of chain.
    _chain_attrs.push_back(_output_attr);
    _chain_full_idx.push_back(std::numeric_limits<size_t>::max());

    const size_t M = _chain_attrs.size();
    _chain_is_projected.assign(M, false);
    _attr_to_col.clear();

    for (uint32_t col = 0; col < _num_attr; ++col) {
        _attr_to_col[_attribute_names[col]] = col;
    }

    for (size_t i = 0; i + 1 < M; ++i) {
        const int32_t projected_idx = info.full_to_projected[_chain_full_idx[i]];
        _chain_is_projected[i] = (projected_idx >= 0);
    }

    // Build _chain_tree_parent_idx
    _chain_tree_parent_idx.assign(M, -1);
    std::unordered_map<size_t, int32_t> full_idx_to_chain;
    for (size_t i = 0; i + 1 < M; ++i) {
        full_idx_to_chain[_chain_full_idx[i]] = static_cast<int32_t>(i);
    }
    for (size_t i = 0; i + 1 < M; ++i) {
        int32_t parent_full = info.parent_idx[_chain_full_idx[i]];
        if (parent_full >= 0) {
            auto it = full_idx_to_chain.find(static_cast<size_t>(parent_full));
            if (it != full_idx_to_chain.end()) { _chain_tree_parent_idx[i] = it->second; }
        }
    }

    // ---- Step 3: Determine state sharing ----
    // chain[i] shares state with chain[i-1] iff they shared state in the original
    // tree AND chain[i]'s tree parent is chain[i-1] (direct parent, identity offset).
    _chain_shares_state_with_prev.assign(M, false);
    for (size_t i = 1; i + 1 < M; ++i) {// skip root (i=0) and _llm (i=M-1)
        if (_chain_tree_parent_idx[i] != static_cast<int32_t>(i - 1)) continue;
        const auto* state_i = info.nodes[_chain_full_idx[i]]->_value->state;
        const auto* state_prev = info.nodes[_chain_full_idx[i - 1]]->_value->state;
        if (state_i == state_prev) { _chain_shares_state_with_prev[i] = true; }
    }

    // The newly created _llm node is always attached as a child to the last data node (M-2)
    // and naturally shares its execution structure. We share its state so it inherits
    // proper ranges, avoiding naive offset copying which corrupts the parent's actual RLE.
    _chain_tree_parent_idx[M - 1] = static_cast<int32_t>(M - 2);
    _chain_shares_state_with_prev[M - 1] = true;

    // ---- Step 4: Allocate states and vectors ----
    _chain_states.resize(M, nullptr);

    for (size_t i = 0; i + 1 < M; ++i) {
        if (i > 0 && _chain_shares_state_with_prev[i]) {
            // Share state with predecessor — re-map this attr to predecessor's DataChunk.
            _chain_states[i] = _chain_states[i - 1];
            _schema->map->set_chunk_for_attr(_chain_attrs[i], _chain_attrs[i - 1]);
        } else {
            // Request new state from DataChunk.
            _chain_states[i] = _schema->map->reset_chunk_state(_chain_attrs[i]);
        }
    }
    // LLM output node (M-1): shares state with last data node (M-2), no DataChunk exists.
    _chain_states[M - 1] = _chain_states[M - 2];

    // Precompute ancestor state paths (for cross-branch expansion).
    _ancestor_state_paths.clear();
    _ancestor_state_paths.resize(M);
    for (size_t i = 1; i + 1 < M; ++i) {
        const int32_t tree_parent_chain = _chain_tree_parent_idx[i];
        if (tree_parent_chain < 0 || tree_parent_chain == static_cast<int32_t>(i - 1)) { continue; }
        const size_t k = static_cast<size_t>(tree_parent_chain);
        auto& state_path = _ancestor_state_paths[i];
        state_path.reserve(i - k);
        for (size_t j = k; j < i; ++j) {
            // Skip consecutive duplicate states (shared-state chain nodes).
            if (!state_path.empty() && _chain_states[j] == state_path.back()) continue;
            state_path.push_back(_chain_states[j]);
        }
    }

    if (M >= 2 && _chain_states[0] != _chain_states[1]) {
        const State* initial_state_path[2] = {_chain_states[0], _chain_states[1]};
        _ancestor_finder = std::make_unique<FtreeAncestorFinder>(initial_state_path, 2);
    } else {
        _ancestor_finder.reset();
    }

    _map_vectors.assign(M, nullptr);
    _owned_vectors.clear();
    _owned_vectors.reserve(M);

    for (size_t i = 0; i < M; ++i) {
        bool identity_rle = (i > 0 && _chain_shares_state_with_prev[i]);
        _owned_vectors.push_back(std::make_unique<Vector<uint64_t>>(_chain_states[i], identity_rle));
        _map_vectors[i] = _owned_vectors.back().get();
    }

    // ---- Step 5: Modify original ftree in-place ----

    // Build a map from full_idx to the shared_ptr held in the parent's _children.
    // We need these to keep reparented Steiner nodes alive.
    _held_nodes.clear();
    std::unordered_map<const FactorizedTreeElement*, std::shared_ptr<FactorizedTreeElement>> node_to_shared;
    {
        // Walk the original tree and record shared_ptrs.
        std::vector<FactorizedTreeElement*> stack = {_root.get()};
        // Root's shared_ptr is _root itself.
        node_to_shared[_root.get()] = _root;
        while (!stack.empty()) {
            auto* n = stack.back();
            stack.pop_back();
            for (auto& child_sp: n->_children) {
                node_to_shared[child_sp.get()] = child_sp;
                stack.push_back(child_sp.get());
            }
        }
    }

    // Hold shared_ptrs for all Steiner nodes (so they survive removal from parents).
    for (size_t i = 0; i + 1 < M; ++i) {
        auto* node = const_cast<FactorizedTreeElement*>(info.nodes[_chain_full_idx[i]]);
        auto it = node_to_shared.find(node);
        assert(it != node_to_shared.end());
        _held_nodes.push_back(it->second);
    }

    // For each Steiner node: remove Steiner children from _children.
    // We need to build a set of Steiner FactorizedTreeElement* for fast lookup.
    std::unordered_set<const FactorizedTreeElement*> steiner_node_ptrs;
    for (size_t fi: steiner) {
        steiner_node_ptrs.insert(info.nodes[fi]);
    }

    for (size_t fi: steiner) {
        auto* node = const_cast<FactorizedTreeElement*>(info.nodes[fi]);
        auto it = std::remove_if(node->_children.begin(), node->_children.end(),
                                 [&](const std::shared_ptr<FactorizedTreeElement>& child) {
                                     return steiner_node_ptrs.count(child.get()) > 0;
                                 });
        node->_children.erase(it, node->_children.end());
    }

    // Re-add chain edges: chain[i]'s node gets chain[i+1]'s node as a child.
    // Create the _llm output node.
    _llm_node = std::make_shared<FactorizedTreeElement>(_output_attr, _map_vectors[M - 1]);

    for (size_t i = 0; i + 1 < M; ++i) {
        auto* parent_node = const_cast<FactorizedTreeElement*>(info.nodes[_chain_full_idx[i]]);
        std::shared_ptr<FactorizedTreeElement> child_sp;
        if (i + 1 == M - 1) {
            // Last chain edge: connect to _llm node.
            child_sp = _llm_node;
            // The _llm node's parent attribute is parent_node->_attribute
            // Since _llm_node didn't exist before, it might not have a DataChunk yet, or it shares with M-2.
            // But we don't need to reparent the llm_out chunk, because it's already properly tied.
        } else {
            auto* child_raw = const_cast<FactorizedTreeElement*>(info.nodes[_chain_full_idx[i + 1]]);
            auto sp_it = node_to_shared.find(child_raw);
            assert(sp_it != node_to_shared.end());
            child_sp = sp_it->second;
        }
        parent_node->_children.push_back(child_sp);
        child_sp->_parent = parent_node;

        // Ensure the DataChunk tree mirrors the new parent-child relationship.
        if (child_sp->_attribute != _output_attr) {
            _schema->map->reparent_chunk(child_sp->_attribute, parent_node->_attribute);
        }
    }

    // Update _value pointers on Steiner chain nodes to point to new vectors.
    for (size_t i = 0; i + 1 < M; ++i) {
        auto* node = const_cast<FactorizedTreeElement*>(info.nodes[_chain_full_idx[i]]);
        node->set_value_ptr(_map_vectors[i]);
    }

    // Publish chain vectors to the schema's variable map.
    for (size_t i = 0; i < M; ++i) {
        _schema->map->set_vector_override(_chain_attrs[i], _map_vectors[i]);
    }

    // ---- Step 6: Non-Steiner nodes ----
    // Allocate new vectors, update _value, but do NOT change parent/children.
    _ns_info.clear();
    _ns_states.clear();
    _ns_owned_vectors.clear();
    _ns_map_vectors.clear();

    // Traverse full tree in DFS order (parents before children).
    for (size_t fi = 0; fi < info.num_nodes; ++fi) {
        if (steiner.count(fi)) continue;             // In chain
        if (info.full_to_projected[fi] < 0) continue;// Not projected

        // Find nearest ancestor that is in the Steiner chain or a previously added NS node.
        int32_t ancestor_full = info.parent_idx[fi];
        // The parent should be in the chain or another NS node (since we project the entire ftree).
        // Determine parent type and index.
        bool parent_is_chain = false;
        size_t parent_idx = SIZE_MAX;

        // Check if direct parent is a chain node.
        if (ancestor_full >= 0) {
            auto chain_it = full_idx_to_chain.find(static_cast<size_t>(ancestor_full));
            if (chain_it != full_idx_to_chain.end()) {
                parent_is_chain = true;
                parent_idx = static_cast<size_t>(chain_it->second);
            } else {
                // Parent should be a previously-added non-Steiner node.
                for (size_t nsi = 0; nsi < _ns_info.size(); ++nsi) {
                    if (_ns_info[nsi].full_idx == static_cast<size_t>(ancestor_full)) {
                        parent_is_chain = false;
                        parent_idx = nsi;
                        break;
                    }
                }
            }
        }
        if (parent_idx == SIZE_MAX) continue;// No reachable parent — skip.

        // Get new state from DataChunk for this non-Steiner node.
        auto* node = const_cast<FactorizedTreeElement*>(info.nodes[fi]);
        State* ns_state = _schema->map->reset_chunk_state(node->_attribute);
        auto ns_vec = std::make_unique<Vector<uint64_t>>(ns_state, false);

        // Update the original ftree node's _value to point to the new vector.
        node->set_value_ptr(ns_vec.get());

        // Publish to schema's variable map.
        _schema->map->set_vector_override(node->_attribute, ns_vec.get());

        _ns_info.push_back({fi, info.full_to_projected[fi], parent_is_chain, parent_idx});
        _ns_map_vectors.push_back(ns_vec.get());
        _ns_states.push_back(ns_state);
        _ns_owned_vectors.push_back(std::move(ns_vec));
    }

    // ---- Step 7: Build visible ordering ----
    _visible_ordering.clear();
    if (_schema->column_ordering) {
        bool has_output_attr = false;
        for (const auto& c: *_schema->column_ordering) {
            if (c != "_cd") {
                _visible_ordering.push_back(c);
                if (c == _output_attr) has_output_attr = true;
            }
        }
        if (!has_output_attr) { _visible_ordering.push_back(_output_attr); }
    } else {
        _visible_ordering.push_back(_output_attr);
    }
}

void FTreeReconstructor::finalize_state_for_count(State* state, uint32_t count) {
    state->selector.clearBitsTillIdx(State::MAX_VECTOR_SIZE - 1);
    state->start_pos = 0;
    if (count == 0) {
        state->end_pos = 0;
        return;
    }
    state->end_pos = static_cast<uint16_t>(count - 1);
    state->selector.setBitsTillIdx(count - 1);
}

uint32_t FTreeReconstructor::trace_to_ancestor(size_t from_chain_idx, size_t to_chain_idx, uint32_t pos) const {
    uint32_t current_pos = pos;
    for (size_t j = from_chain_idx; j > to_chain_idx; --j) {
        if (_chain_shares_state_with_prev[j]) {
            // Identity offset: position unchanged.
            continue;
        }
        const uint16_t* off = _chain_states[j]->offset;
        uint32_t parent_count = _write_counts[j - 1];
        uint32_t lo = 0, hi = parent_count;
        while (lo < hi) {
            uint32_t mid = lo + (hi - lo) / 2;
            if (off[mid + 1] <= current_pos) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        current_pos = lo;
    }
    return current_pos;
}

void FTreeReconstructor::expand_cross_branch(size_t chain_idx, const FTreeBatchIterator::FullTreeInfo& info) {
    int32_t tree_parent_chain = _chain_tree_parent_idx[chain_idx];
    assert(tree_parent_chain >= 0 && "FTreeReconstructor: cross-branch node has no tree parent in chain");
    size_t k = static_cast<size_t>(tree_parent_chain);

    size_t full_idx = _chain_full_idx[chain_idx];
    int32_t proj_col = info.full_to_projected[full_idx];
    assert(proj_col >= 0 && "FTreeReconstructor: cross-branch node is not projected");
    const uint64_t* itr_values = _batch_iterator->get_buffer(static_cast<size_t>(proj_col));
    const uint32_t* itr_offset = _batch_iterator->get_node_offset(static_cast<size_t>(proj_col));

    size_t parent_full_idx = _chain_full_idx[k];
    int32_t parent_proj_col = info.full_to_projected[parent_full_idx];
    assert(parent_proj_col >= 0 && "FTreeReconstructor: cross-branch parent is not projected");
    const uint32_t parent_pos_count =
            static_cast<uint32_t>(_batch_iterator->get_count(static_cast<size_t>(parent_proj_col)));

    uint32_t prev_count = _write_counts[chain_idx - 1];
    uint64_t* out_values = _map_vectors[chain_idx]->values;
    uint16_t* out_offset = _chain_states[chain_idx]->offset;

    // Precompute ancestor positions
    _ancestor_map.resize(prev_count);
    if (k == chain_idx - 1) {
        for (uint32_t p = 0; p < prev_count; ++p)
            _ancestor_map[p] = p;
    } else if (prev_count == 0 || parent_pos_count == 0) {
        std::fill(_ancestor_map.begin(), _ancestor_map.end(), 0u);
    } else {
        for (size_t j = k; j <= chain_idx - 1; ++j) {
            finalize_state_for_count(_chain_states[j], _write_counts[j]);
        }

        const auto& state_path = _ancestor_state_paths[chain_idx];
        assert(!state_path.empty() && "FTreeReconstructor: ancestor state path was not precomputed");

        if (state_path.size() == 1) {
            // All intermediate states are shared (identity) — positions map 1:1.
            for (uint32_t p = 0; p < prev_count; ++p)
                _ancestor_map[p] = p;
        } else {
            std::fill(_ancestor_map.begin(), _ancestor_map.end(), UINT32_MAX);
            if (_ancestor_finder == nullptr) {
                _ancestor_finder = std::make_unique<FtreeAncestorFinder>(state_path.data(), state_path.size());
            } else {
                _ancestor_finder->reset(state_path.data(), state_path.size());
            }
            _ancestor_finder->process(_ancestor_map.data(), 0, static_cast<int32_t>(parent_pos_count - 1), 0,
                                      static_cast<int32_t>(prev_count - 1));

            for (uint32_t p = 0; p < prev_count; ++p) {
                assert(_ancestor_map[p] != UINT32_MAX &&
                       "FTreeReconstructor: ancestor finder failed to map a parent position");
            }
        }
    }

    for (uint32_t p = 0; p < prev_count; ++p) {
        uint32_t ancestor_pos = _ancestor_map[p];
        if (parent_pos_count == 0) {
            out_offset[p] = static_cast<uint16_t>(_write_counts[chain_idx]);
            out_offset[p + 1] = static_cast<uint16_t>(_write_counts[chain_idx]);
            continue;
        }
        if (ancestor_pos >= parent_pos_count) { ancestor_pos = parent_pos_count - 1; }
        uint32_t range_start = itr_offset[ancestor_pos];
        uint32_t range_end = itr_offset[ancestor_pos + 1];
        uint32_t num_values = range_end - range_start;

        out_offset[p] = static_cast<uint16_t>(_write_counts[chain_idx]);

        // Stage 1: fill remaining space in current vector
        uint32_t remaining_space = static_cast<uint32_t>(State::MAX_VECTOR_SIZE) - _write_counts[chain_idx];
        uint32_t to_copy = std::min(remaining_space, num_values);
        std::memcpy(&out_values[_write_counts[chain_idx]], &itr_values[range_start], to_copy * sizeof(uint64_t));
        _write_counts[chain_idx] += to_copy;
        uint32_t read_idx = to_copy;

        if (_write_counts[chain_idx] == static_cast<uint32_t>(State::MAX_VECTOR_SIZE)) {
            out_offset[p + 1] = static_cast<uint16_t>(State::MAX_VECTOR_SIZE);
            flush_at_node(chain_idx, p, 0, info);

            std::memset(out_offset, 0, (p + 1) * sizeof(uint16_t));
            out_offset[p + 1] = 0;

            uint32_t remaining_values = num_values - to_copy;

            // Stage 2: full MAX_VECTOR_SIZE chunks
            while (remaining_values >= static_cast<uint32_t>(State::MAX_VECTOR_SIZE)) {
                std::memcpy(&out_values[0], &itr_values[range_start + read_idx],
                            static_cast<size_t>(State::MAX_VECTOR_SIZE) * sizeof(uint64_t));
                _write_counts[chain_idx] = static_cast<uint32_t>(State::MAX_VECTOR_SIZE);
                out_offset[p] = 0;
                out_offset[p + 1] = static_cast<uint16_t>(State::MAX_VECTOR_SIZE);
                flush_at_node(chain_idx, p, 0, info);
                out_offset[p + 1] = 0;
                read_idx += static_cast<uint32_t>(State::MAX_VECTOR_SIZE);
                remaining_values -= static_cast<uint32_t>(State::MAX_VECTOR_SIZE);
            }

            // Stage 3: leftover partial chunk
            if (remaining_values > 0) {
                std::memcpy(&out_values[0], &itr_values[range_start + read_idx], remaining_values * sizeof(uint64_t));
                _write_counts[chain_idx] = remaining_values;
                out_offset[p] = 0;
            }
        }

        out_offset[p + 1] = static_cast<uint16_t>(_write_counts[chain_idx]);
    }
}

void FTreeReconstructor::place_llm_outputs(const FTreeBatchIterator::FullTreeInfo& info) {
    const size_t M = _chain_attrs.size();
    const size_t llm_idx = M - 1;
    const size_t last_data_idx = M - 2;

    const uint32_t prev_count = _write_counts[last_data_idx];
    uint64_t* out_values = _map_vectors[llm_idx]->values;
    uint16_t* out_offset = _chain_states[llm_idx]->offset;
    const uint64_t* llm_results = _llm_batch->results;
    const size_t llm_count = _llm_batch->count;

    uint32_t p = 0;
    while (p < prev_count) {
        const uint32_t space = static_cast<uint32_t>(State::MAX_VECTOR_SIZE) - _write_counts[llm_idx];
        const uint32_t batch = std::min(space, prev_count - p);

        if (!_chain_shares_state_with_prev[llm_idx]) {
            const uint16_t base = static_cast<uint16_t>(_write_counts[llm_idx]);
#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
            for (uint32_t i = 0; i < batch; ++i) {
                out_offset[p + i] = static_cast<uint16_t>(base + i);
            }
            out_offset[p + batch] = static_cast<uint16_t>(base + batch);
        }

        std::vector<uint32_t> root_by_parent;
        if (llm_count > 0) {
            if (_llm_read_idx + batch <= llm_count) {
                std::memcpy(&out_values[_write_counts[llm_idx]], &llm_results[_llm_read_idx],
                             batch * sizeof(uint64_t));
                _llm_read_idx += batch;
            } else {
                for (uint32_t i = 0; i < batch; ++i) {
                    size_t safe_idx = (_llm_read_idx < llm_count) ? _llm_read_idx : (llm_count - 1);
                    out_values[_write_counts[llm_idx] + i] = llm_results[safe_idx];
                    if (_llm_read_idx < llm_count) _llm_read_idx++;
                }
            }
        } else {
            std::memset(&out_values[_write_counts[llm_idx]], 0, batch * sizeof(uint64_t));
        }

        _write_counts[llm_idx] += batch;
        p += batch;

        if (_write_counts[llm_idx] == static_cast<uint32_t>(State::MAX_VECTOR_SIZE) && p < prev_count) {
            flush_at_node(llm_idx, p - 1, 0, info);
            if (!_chain_shares_state_with_prev[llm_idx]) {
                std::memset(out_offset, 0, p * sizeof(uint16_t));
                out_offset[p] = 0;
            }
        }
    }
}

void FTreeReconstructor::fill_downstream(size_t start_idx, const FTreeBatchIterator::FullTreeInfo& info) {
    const size_t M = _chain_attrs.size();
    if (start_idx >= M) return;
    bool needs_expansion = false;

    for (size_t i = start_idx; i + 1 < M; ++i) {
        // Shared-state chain nodes: values only, offsets are identity (implicit from shared State).
        if (_chain_shares_state_with_prev[i]) {
            size_t full_idx = _chain_full_idx[i];
            int32_t proj_col = info.full_to_projected[full_idx];
            assert(proj_col >= 0 && "FTreeReconstructor: shared-state chain node not projected");
            const uint64_t* itr_values = _batch_iterator->get_buffer(static_cast<size_t>(proj_col));
            size_t itr_count = _batch_iterator->get_count(static_cast<size_t>(proj_col));
            std::memcpy(_map_vectors[i]->values, itr_values, itr_count * sizeof(uint64_t));
            _write_counts[i] = static_cast<uint32_t>(itr_count);
            // No offset writing — identity is implicit from shared state.
            // needs_expansion stays as is (shared-state doesn't change expansion status).
            continue;
        }

        size_t full_idx = _chain_full_idx[i];
        int32_t proj_col = info.full_to_projected[full_idx];
        assert(proj_col >= 0 && "FTreeReconstructor: chain node not projected");

        const bool direct_parent = (_chain_tree_parent_idx[i] == static_cast<int32_t>(i - 1));
        if (direct_parent && !needs_expansion) {
            const uint64_t* itr_values = _batch_iterator->get_buffer(static_cast<size_t>(proj_col));
            size_t itr_count = _batch_iterator->get_count(static_cast<size_t>(proj_col));
            std::memcpy(_map_vectors[i]->values, itr_values, itr_count * sizeof(uint64_t));
            _write_counts[i] = static_cast<uint32_t>(itr_count);

            const uint32_t* itr_offset = _batch_iterator->get_node_offset(static_cast<size_t>(proj_col));
            uint32_t parent_count = _write_counts[i - 1];
            uint16_t* out_offset = _chain_states[i]->offset;
            for (uint32_t p = 0; p <= parent_count; ++p) {
                out_offset[p] = static_cast<uint16_t>(itr_offset[p]);
            }
        } else {
            expand_cross_branch(i, info);
            needs_expansion = true;
        }
    }

    place_llm_outputs(info);
}

void FTreeReconstructor::fill_non_steiner_from_iterator(const FTreeBatchIterator::FullTreeInfo& info) {
    _ns_write_counts.assign(_ns_info.size(), 0u);

    // NS nodes are in DFS order (parents before children), so when we process
    // ns[i], its parent (if also NS) has already been filled with correct _ns_write_counts.
    for (size_t i = 0; i < _ns_info.size(); ++i) {
        const auto& ns = _ns_info[i];
        const int32_t proj_col = ns.proj_col;
        assert(proj_col >= 0);

        const uint64_t* itr_values = _batch_iterator->get_buffer(static_cast<size_t>(proj_col));
        const size_t itr_count = _batch_iterator->get_count(static_cast<size_t>(proj_col));
        const uint32_t copy_count =
                static_cast<uint32_t>(std::min(itr_count, static_cast<size_t>(State::MAX_VECTOR_SIZE)));

        std::memcpy(_ns_map_vectors[i]->values, itr_values, copy_count * sizeof(uint64_t));
        _ns_write_counts[i] = copy_count;

        // Copy offsets relative to actual parent (chain node or another NS node).
        const uint32_t* itr_offset = _batch_iterator->get_node_offset(static_cast<size_t>(proj_col));
        const uint32_t parent_count =
                ns.parent_is_chain ? _write_counts[ns.parent_idx] : _ns_write_counts[ns.parent_idx];
        uint16_t* out_offset = _ns_states[i]->offset;
        for (uint32_t p = 0; p <= parent_count; ++p) {
            out_offset[p] = static_cast<uint16_t>(itr_offset[p]);
        }
    }
}

void FTreeReconstructor::flush_at_node(size_t chain_idx, uint32_t /*parent_pos*/, uint32_t /*chunk_start*/,
                                       const FTreeBatchIterator::FullTreeInfo& info) {
    const size_t M = _chain_attrs.size();

    // 1. Fill downstream chain[chain_idx+1..M-1]
    fill_downstream(chain_idx + 1, info);

    // 2. Fill non-Steiner branches
    fill_non_steiner_from_iterator(info);

    // 3. Finalize all states (chain + non-Steiner).
    //    Skip shared-state chain nodes (their state is finalized by the owning node).
    std::unordered_set<State*> finalized_states;
    for (size_t j = 0; j < M; ++j) {
        if (finalized_states.insert(_chain_states[j]).second) {
            finalize_state_for_count(_chain_states[j], _write_counts[j]);
        }
    }
    for (size_t j = 0; j < _ns_info.size(); ++j) {
        finalize_state_for_count(_ns_states[j], _ns_write_counts[j]);
    }

    // 4. Invoke flush callback (replaces next_op->execute())
    if (_flush_callback) _flush_callback();

    // 5. Reset write counts for flushed nodes
    for (size_t j = chain_idx; j < M; ++j) {
        _write_counts[j] = 0;
    }
    std::fill(_ns_write_counts.begin(), _ns_write_counts.end(), 0u);
}

void FTreeReconstructor::fill_chain_from_iterator() {
    const auto info = _batch_iterator->get_full_tree_info();
    const size_t M = _chain_attrs.size();
    if (M < 2) return;

    _write_counts.resize(M);
    std::fill(_write_counts.begin(), _write_counts.end(), 0u);
    _llm_read_idx = 0;

    // Fill root (chain[0]): always direct copy from iterator
    {
        size_t full_idx = _chain_full_idx[0];
        int32_t proj_col = info.full_to_projected[full_idx];
        assert(proj_col >= 0 && "FTreeReconstructor: root chain node not projected");
        const uint64_t* itr_values = _batch_iterator->get_buffer(static_cast<size_t>(proj_col));
        size_t itr_count = _batch_iterator->get_count(static_cast<size_t>(proj_col));
        std::memcpy(_map_vectors[0]->values, itr_values, itr_count * sizeof(uint64_t));
        _write_counts[0] = static_cast<uint32_t>(itr_count);
    }

    // Fill chain[1..M-1] (data nodes + LLM) with chunking
    fill_downstream(1, info);

    // Fill non-Steiner branches from the batch iterator
    fill_non_steiner_from_iterator(info);

    // Final flush for remaining data
    if (_write_counts[0] > 0) {
        std::unordered_set<State*> finalized_states;
        for (size_t j = 0; j < M; ++j) {
            if (finalized_states.insert(_chain_states[j]).second) {
                finalize_state_for_count(_chain_states[j], _write_counts[j]);
            }
        }
        for (size_t j = 0; j < _ns_info.size(); ++j) {
            finalize_state_for_count(_ns_states[j], _ns_write_counts[j]);
        }
        if (_flush_callback) _flush_callback();
    }
}

void FTreeReconstructor::append(const LLMResultBatch& batch) {
    _llm_batch = &batch;
    fill_chain_from_iterator();
    assert(_llm_read_idx <= _llm_batch->count && "FTreeReconstructor: invalid llm_result_batch read index");
    _num_output_tuples += _llm_batch->count;
}


void FTreeReconstructor::debug_print_constructed_tree() const {
    std::cerr << "\n[DEBUG][FTreeReconstructor] Constructed Tree\n";
    std::cerr << "  chain_attrs=[";
    for (size_t i = 0; i < _chain_attrs.size(); ++i) {
        std::cerr << _chain_attrs[i];
        if (i + 1 < _chain_attrs.size()) std::cerr << ",";
    }
    std::cerr << "]\n";

    std::cerr << "  chain_projected=[";
    for (size_t i = 0; i < _chain_is_projected.size(); ++i) {
        std::cerr << (_chain_is_projected[i] ? 1 : 0);
        if (i + 1 < _chain_is_projected.size()) std::cerr << ",";
    }
    std::cerr << "]\n";

    std::cerr << "  chain_tree_parent_idx=[";
    for (size_t i = 0; i < _chain_tree_parent_idx.size(); ++i) {
        std::cerr << _chain_tree_parent_idx[i];
        if (i + 1 < _chain_tree_parent_idx.size()) std::cerr << ",";
    }
    std::cerr << "]\n";

    std::cerr << "  chain_shares_state=[";
    for (size_t i = 0; i < _chain_shares_state_with_prev.size(); ++i) {
        std::cerr << (_chain_shares_state_with_prev[i] ? 1 : 0);
        if (i + 1 < _chain_shares_state_with_prev.size()) std::cerr << ",";
    }
    std::cerr << "]\n";

    std::cerr << "  visible_ordering=[";
    for (size_t i = 0; i < _visible_ordering.size(); ++i) {
        std::cerr << _visible_ordering[i];
        if (i + 1 < _visible_ordering.size()) std::cerr << ",";
    }
    std::cerr << "]\n";

    if (_root) {
        std::function<void(const FactorizedTreeElement*, int)> print_tree;
        print_tree = [&](const FactorizedTreeElement* n, int depth) {
            for (int d = 0; d < depth; ++d)
                std::cerr << "  ";
            std::cerr << "node attr='" << n->_attribute << "' children=" << n->_children.size() << "\n";
            for (const auto& c: n->_children)
                print_tree(c.get(), depth + 1);
        };
        print_tree(_root.get(), 1);
    }
}

void FTreeReconstructor::debug_print_chain_values() const {
    std::cerr << "\n[DEBUG][FTreeReconstructor] Chain values after fill_chain_from_iterator\n";

    const size_t M = _chain_attrs.size();
    for (size_t i = 0; i < M; ++i) {
        uint32_t count = _write_counts.size() > i ? _write_counts[i] : 0;
        std::cerr << "  chain[" << i << "] attr='" << _chain_attrs[i] << "' count=" << count;
        if (_chain_shares_state_with_prev[i]) std::cerr << " [shared-state]";
        if (_map_vectors[i] && count > 0) {
            std::cerr << " values=[";
            for (uint32_t j = 0; j < std::min(count, 16u); ++j) {
                std::cerr << _map_vectors[i]->values[j];
                if (j + 1 < count) std::cerr << ",";
            }
            if (count > 16) std::cerr << "...";
            std::cerr << "]";
        }
        std::cerr << "\n";

        if (i > 0 && !_chain_shares_state_with_prev[i] && _chain_states[i]) {
            uint32_t parent_count = _write_counts.size() > (i - 1) ? _write_counts[i - 1] : 0;
            std::cerr << "    offset=[";
            for (uint32_t p = 0; p <= std::min(parent_count, 16u); ++p) {
                std::cerr << _chain_states[i]->offset[p];
                if (p < parent_count) std::cerr << ",";
            }
            if (parent_count > 16) std::cerr << "...";
            std::cerr << "]\n";
        }
    }
}

}// namespace ffx
