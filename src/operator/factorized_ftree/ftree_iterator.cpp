#include "factorized_ftree/ftree_iterator.hpp"
#include "schema/schema.hpp"
#include "vector/state.hpp"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

namespace ffx {
namespace {
// Identity offset: offset[i]=i, offset[i+1]=i+1 => range [i,i]. Used when child shares parent state.
alignas(64) uint16_t g_identity_offset[State::MAX_VECTOR_SIZE + 1];
bool g_identity_offset_init = false;
void init_identity_offset() {
    if (g_identity_offset_init) return;
    for (int i = 0; i <= State::MAX_VECTOR_SIZE; ++i) g_identity_offset[i] = static_cast<uint16_t>(i);
    g_identity_offset_init = true;
}
}// namespace

FTreeIterator::FTreeIterator()
    : Operator(), _root(nullptr), _iterators(nullptr), _nodes(nullptr), _current_tuple(nullptr), _num_attributes(0),
      _is_valid(false) {}

void FTreeIterator::init(Schema* schema) {
    if (!schema) {
        throw std::runtime_error("FTreeIterator::init: schema is null");
    }
    if (!schema->root) {
        throw std::runtime_error("FTreeIterator::init: schema->root is null");
    }
    if (!schema->column_ordering) {
        throw std::runtime_error("FTreeIterator::init: schema->column_ordering is null");
    }

    _root = schema->root;
    _is_valid = false;
    _synthetic_nodes.clear();

    // Use the provided column ordering directly (filtering out "_cd" markers)
    const auto& column_ordering = *schema->column_ordering;
    std::vector<std::string> actual_columns;
    for (const auto& col: column_ordering) {
        if (col != "_cd") { actual_columns.push_back(col); }
    }

    _num_attributes = actual_columns.size();
    assert(_num_attributes > 0);

    _iterators = std::make_unique<SimpleLocalIterator[]>(_num_attributes);
    _nodes = std::make_unique<FactorizedTreeElement*[]>(_num_attributes);
    _current_tuple = std::make_unique<uint64_t[]>(_num_attributes);

    // Debug: allocate per-position counters
    // _debug_pos_counts = std::make_unique<std::unique_ptr<uint64_t[]>[]>(_num_attributes);
    // for (size_t i = 0; i < _num_attributes; ++i) {
    //     _debug_pos_counts[i] = std::make_unique<uint64_t[]>(State::MAX_VECTOR_SIZE);
    //     std::fill_n(_debug_pos_counts[i].get(), State::MAX_VECTOR_SIZE, 0);
    // }

    // Find each node in the tree by attribute name and populate in column order
    for (size_t i = 0; i < _num_attributes; ++i) {
        const std::string& attr = actual_columns[i];
        auto* node = _root->find_node_by_attribute(attr);
        if (!node) {
            // Fallback: some attributes (e.g., 1:1 properties) may be materialized
            // as vectors in the Schema map without being embedded as nodes in the
            // factorized tree. In that case, synthesize a node so iteration works.
            if (!schema->map) {
                throw std::runtime_error("FTreeIterator::init: schema->map is null (cannot resolve '" + attr + "')");
            }
            auto* vec = schema->map->get_vector<uint64_t>(attr);
            if (!vec) {
                throw std::runtime_error("FTreeIterator::init: attribute '" + attr +
                                         "' not found in tree and no vector exists in schema map");
            }
            auto owned = std::make_shared<FactorizedTreeElement>(attr, vec);
            // Heuristic parent assignment: attach to the most recent prior node that
            // shares the same State (typical for shared-state 1:1 properties).
            for (int64_t j = static_cast<int64_t>(i) - 1; j >= 0; --j) {
                if (_nodes[static_cast<size_t>(j)] && _nodes[static_cast<size_t>(j)]->_value &&
                    _nodes[static_cast<size_t>(j)]->_value->state == vec->state) {
                    owned->_parent = _nodes[static_cast<size_t>(j)];
                    break;
                }
            }
            _synthetic_nodes.push_back(owned);
            node = owned.get();
        }
        _nodes[i] = node;
    }

    // Set up iterators: for each node in column order, find its parent in the array
    for (size_t i = 0; i < _num_attributes; ++i) {
        auto& itr = _iterators[i];
        auto* node = _nodes[i];

        // These are set during initialize_iterators
        itr.selector = nullptr;
        itr.values = nullptr;
        itr.offset = nullptr;

        // Find the tree parent index by searching for parent node in _nodes array
        if (node->_parent != nullptr) {
            itr.parent_idx = -1;
            for (size_t j = 0; j < _num_attributes; ++j) {
                if (_nodes[j] == node->_parent) {
                    itr.parent_idx = static_cast<int32_t>(j);
                    break;
                }
            }
            assert(itr.parent_idx >= 0);
            const bool shares_parent_state = (_nodes[itr.parent_idx]->_value->state == node->_value->state);
            init_identity_offset();
            itr.offset = shares_parent_state ? g_identity_offset : node->_value->state->offset;
        } else {
            // Root node
            itr.parent_idx = -1;
            itr.offset = nullptr;
        }
    }
}

void FTreeIterator::initialize_iterators() {
    // Debug: reset position counters
    // for (size_t i = 0; i < _num_attributes; ++i) {
    //     std::fill_n(_debug_pos_counts[i].get(), State::MAX_VECTOR_SIZE, 0);
    // }

    _is_valid = true;

    // Root node - initialize selector, values, and range
    auto& root_itr = _iterators[0];
    if (!_nodes[0] || !_nodes[0]->_value || !_nodes[0]->_value->state) {
        std::cout << "Root node is null or has null value/state!\n";
    }
    const auto* root_state = _nodes[0]->_value->state;
    root_itr.selector = &root_state->selector;
    root_itr.values = _nodes[0]->_value->values;
    root_itr.valid_start = GET_START_POS(*root_state);
    root_itr.valid_end = GET_END_POS(*root_state);
    root_itr.current_pos = root_itr.valid_start;

    // Find first valid bit for root
    while (root_itr.current_pos <= root_itr.valid_end && !TEST_BIT(*root_itr.selector, root_itr.current_pos)) {
        root_itr.current_pos++;
    }
    _is_valid = (root_itr.current_pos <= root_itr.valid_end);

    // Initialize all other iterators based on their parent's position
    // Since parent always comes before child in ordering, we can iterate sequentially
    for (size_t i = 1; i < _num_attributes && _is_valid; ++i) {
        auto& itr = _iterators[i];
        if (!_nodes[i] || !_nodes[i]->_value || !_nodes[i]->_value->state) {
            std::cout << "Node " << i << " is null or has null value/state!\n";
        }
        const auto* __restrict__ state = _nodes[i]->_value->state;
        itr.selector = &state->selector;
        // itr.offset set in init() (identity or state->offset)
        itr.values = _nodes[i]->_value->values;

        const auto* __restrict__ selector = itr.selector;

        // Compute valid range based on parent position
        const int32_t parent_pos = _iterators[itr.parent_idx].current_pos;
        const auto* __restrict__ rle = itr.offset;
        const uint32_t range_start = rle[parent_pos];
        const uint32_t range_end = rle[parent_pos + 1] - 1;
        const int32_t state_start = GET_START_POS(*state);
        const int32_t state_end = GET_END_POS(*state);

        itr.valid_start = std::max(state_start, static_cast<int32_t>(range_start));
        itr.valid_end = std::min(state_end, static_cast<int32_t>(range_end));

        // Cache the computed range
        itr.cached_range_start = itr.valid_start;
        itr.cached_range_end = itr.valid_end;
        itr.cached_parent_pos = parent_pos;

        itr.current_pos = itr.valid_start;

        // Find first valid bit
        while (itr.current_pos <= itr.valid_end && !TEST_BIT(*itr.selector, itr.current_pos)) {
            itr.current_pos++;
        }
        _is_valid = (itr.current_pos <= itr.valid_end);
    }

    // If initialization failed, try to find a valid configuration
    if (!_is_valid) {
        // Try advancing from the root to find a valid tuple
        auto& root_itr = _iterators[0];
        while (!_is_valid && root_itr.current_pos <= root_itr.valid_end) {
            root_itr.current_pos++;
            while (root_itr.current_pos <= root_itr.valid_end && !TEST_BIT(*root_itr.selector, root_itr.current_pos)) {
                root_itr.current_pos++;
            }

            if (root_itr.current_pos > root_itr.valid_end) { break; }

            _is_valid = true;
            for (size_t i = 1; i < _num_attributes && _is_valid; ++i) {
                _is_valid = reset_iterator_to_start(i);
            }
        }
    }
}

__attribute__((always_inline)) inline bool FTreeIterator::reset_iterator_to_start(size_t idx) {
    assert(idx > 0);

    auto& itr = _iterators[idx];
    const int32_t parent_pos = _iterators[itr.parent_idx].current_pos;

    // Check if we can reuse cached range
    if (__builtin_expect(parent_pos == itr.cached_parent_pos && itr.cached_range_start >= 0, 1)) {
        // Use cached values - no computation needed!
        itr.valid_start = itr.cached_range_start;
        itr.valid_end = itr.cached_range_end;
    } else {
        // Recompute and cache
        const auto* __restrict__ state = _nodes[idx]->_value->state;
        const auto* __restrict__ rle = itr.offset;
        const uint32_t range_start = rle[parent_pos];
        const uint32_t range_end = rle[parent_pos + 1] - 1;
        const int32_t state_start = GET_START_POS(*state);
        const int32_t state_end = GET_END_POS(*state);

        itr.valid_start = std::max(state_start, static_cast<int32_t>(range_start));
        itr.valid_end = std::min(state_end, static_cast<int32_t>(range_end));

        // Cache the computed values
        itr.cached_range_start = itr.valid_start;
        itr.cached_range_end = itr.valid_end;
        itr.cached_parent_pos = parent_pos;
    }

    itr.current_pos = itr.valid_start;

    // Find first valid bit
    while (__builtin_expect(itr.current_pos <= itr.valid_end, 1) && !TEST_BIT(*itr.selector, itr.current_pos)) {
        itr.current_pos++;
    }
    
    return (itr.current_pos <= itr.valid_end);
}

__attribute__((always_inline)) inline bool FTreeIterator::try_advance(size_t idx) {
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
FTreeIterator::find_next_set_bit_root(const SimpleLocalIterator& itr) const {
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

        // Skip empty blocks early (cheap check)
        if (__builtin_expect(block_val == 0, 0)) { continue; }

        // Branchless mask application
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

__attribute__((always_inline)) inline int32_t FTreeIterator::find_next_set_bit_child(const SimpleLocalIterator& itr,
                                                                                     int32_t parent_pos) const {
    const uint16_t* __restrict__ rle = itr.offset;
    const uint32_t parent_start = rle[parent_pos];
    const uint32_t parent_end = rle[parent_pos + 1] - 1;
    const int32_t allowed_start = std::max<int32_t>(itr.valid_start, static_cast<int32_t>(parent_start));
    const int32_t allowed_end = std::min<int32_t>(itr.valid_end, static_cast<int32_t>(parent_end));

    if (__builtin_expect(allowed_start > allowed_end, 0)) { return -1; }

    const auto* __restrict__ selector = itr.selector;
    const uint64_t* __restrict__ bits = selector->bits;
    const int32_t pos = (itr.current_pos < allowed_start) ? allowed_start : itr.current_pos + 1;

    if (__builtin_expect(pos > allowed_end, 0)) { return -1; }

    const size_t start_block = pos >> 6;
    const size_t end_block = allowed_end >> 6;
    const size_t start_bit = pos & 63;
    const size_t end_bit = allowed_end & 63;
    const uint64_t end_mask = (end_bit == 63) ? ~0ULL : ((1ULL << (end_bit + 1)) - 1);

    for (size_t block = start_block; block <= end_block; ++block) {
        uint64_t block_val = bits[block];

        // Skip empty blocks early (cheap check)
        if (__builtin_expect(block_val == 0, 0)) { continue; }

        // Branchless mask application
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

void FTreeIterator::read_current_tuple(uint64_t* buffer) const {
    assert(buffer != nullptr);
    for (size_t i = 0; i < _num_attributes; ++i) {
        const auto& itr = _iterators[i];
        assert(itr.values != nullptr);
        const auto* __restrict__ values = itr.values;
        buffer[i] = values[itr.current_pos];
    }
    _num_output_tuples++;
}

void FTreeIterator::update_min_vals_current_tuple(uint64_t* buffer) const {
    assert(buffer != nullptr);

    // Hoist the empty check outside the loop
    const bool has_string_attrs = !_is_string_attr.empty();

    for (size_t i = 0; i < _num_attributes; ++i) {
        const auto& itr = _iterators[i];
        assert(itr.values != nullptr);
        const auto* __restrict__ values = itr.values;
        const uint64_t val = values[itr.current_pos];
        const uint64_t cur = buffer[i];

        // Check if this is a string attribute and needs lexicographic comparison
        if (has_string_attrs && _is_string_attr[i]) {
            assert(_dictionary);
            // Skip NULL values (UINT64_MAX represents NULL string)
            if (val == std::numeric_limits<uint64_t>::max()) { continue; }

            // String comparison - handle uninitialized case
            if (cur == std::numeric_limits<uint64_t>::max()) {
                buffer[i] = val;
            } else {
                // Only do dictionary lookups when actually comparing
                const auto& val_str = _dictionary->get_string(val);
                const auto& cur_str = _dictionary->get_string(cur);
                if (val_str < cur_str) { buffer[i] = val; }
            }
        } else {
            // Numeric comparison (branchless min - compiler generates cmov)
            buffer[i] = val < cur ? val : cur;
        }
    }
}

void FTreeIterator::update_min_vals_current_tuple_2(uint64_t* buffer, size_t start_idx) const {
    assert(buffer != nullptr);
    assert(start_idx < _num_attributes);

    // Hoist the empty check outside the loop
    const bool has_string_attrs = !_is_string_attr.empty();

    // Only update iterators from start_idx to the end
    for (size_t i = start_idx; i < _num_attributes; ++i) {
        const auto& itr = _iterators[i];
        assert(itr.values != nullptr);
        const auto* __restrict__ values = itr.values;
        const uint64_t val = values[itr.current_pos];
        const uint64_t cur = buffer[i];

        // Check if this is a string attribute and needs lexicographic comparison
        if (has_string_attrs && _is_string_attr[i]) {
            assert(_dictionary);
            // Skip NULL values (UINT64_MAX represents NULL string)
            if (val == std::numeric_limits<uint64_t>::max()) { continue; }

            // String comparison - handle uninitialized case
            if (cur == std::numeric_limits<uint64_t>::max()) {
                buffer[i] = val;
            } else {
                // Only do dictionary lookups when actually comparing
                const auto& val_str = _dictionary->get_string(val);
                const auto& cur_str = _dictionary->get_string(cur);
                if (val_str < cur_str) { buffer[i] = val; }
            }
        } else {
            // Numeric comparison (branchless min - compiler generates cmov)
            buffer[i] = val < cur ? val : cur;
        }
    }
}


// void FTreeIterator::print_debug_counts() const {
//     std::cout << "\n=== SIMPLIFIED ITERATOR per-vector tuple counts ===" << std::endl;
//     for (size_t i = 0; i < _num_attributes; ++i) {
//         const auto* state = _nodes[i]->_value->state;
//         const auto start = GET_START_POS(*state);
//         const auto end = GET_END_POS(*state);
//
//         uint64_t total = 0;
//         std::cout << "  " << _nodes[i]->_attribute << " [" << start << "," << end << "]: ";
//         for (auto pos = start; pos <= end; pos++) {
//             if (_debug_pos_counts[i][pos] > 0) {
//                 std::cout << "pos" << pos << "=" << _debug_pos_counts[i][pos] << " ";
//                 total += _debug_pos_counts[i][pos];
//             }
//         }
//         std::cout << " | total=" << total << std::endl;
//     }
// }

// void FTreeIterator::print_iterator_state() const {
//     const auto& root_itr = _iterators[0];
//     if (root_itr.current_pos != 45) return;
//     const auto& a_itr = _iterators[1];
//     if (a_itr.current_pos != 961) return;
//     std::cout << "Iterator state: ";
//     for (size_t i = 0; i < _num_attributes; ++i) {
//         const auto& itr = _iterators[i];
//         std::cout << _nodes[i]->_attribute << "={pos=" << itr.current_pos
//                   << ",range=[" << itr.valid_start << "," << itr.valid_end << "]}";
//         if (i < _num_attributes - 1) std::cout << ", ";
//     }
//     std::cout << std::endl;
// }

// bool FTreeIterator::next_debug(uint64_t* output_buffer) {
//     assert(_is_valid);
//
//     // Print state before outputting tuple
//     std::cout << "OUTPUT TUPLE: ";
//     print_iterator_state();
//
//     read_current_tuple(output_buffer);
//
//     // Start from the rightmost iterator and try to advance
//     int start_idx = static_cast<int>(_num_attributes) - 1;
//
//     while (true) {
//         // Try to advance from start_idx backwards until we find one that can advance
//         int updated_itr_idx = start_idx;
//         for (; updated_itr_idx >= 0; updated_itr_idx--) {
//             if (try_advance(static_cast<size_t>(updated_itr_idx))) {
//                 break;
//             }
//         }
//
//         // Could not advance any iterator - all exhausted
//         if (updated_itr_idx < 0) {
//             _is_valid = false;
//             return false;
//         }
//
//         std::cout << "  ADVANCED " << _nodes[updated_itr_idx]->_attribute
//                   << " (idx=" << updated_itr_idx << ") to pos="
//                   << _iterators[updated_itr_idx].current_pos << std::endl;
//
//         // Reset all iterators after updated_itr_idx
//         // Since parent always comes before child in ordering, we can iterate sequentially
//         bool all_valid = true;
//         for (size_t i = static_cast<size_t>(updated_itr_idx) + 1; i < _num_attributes && all_valid; ++i) {
//             std::cout << "    RESET " << _nodes[i]->_attribute << " (idx=" << i << ")" << std::endl;
//             all_valid = reset_iterator_to_start(i);
//         }
//
//         if (all_valid) {
//             return true;
//         }
//
//         std::cout << "    RETRY (empty range)" << std::endl;
//         start_idx = updated_itr_idx;
//     }
// }

bool FTreeIterator::next(uint64_t* __restrict__ output_buffer) {
    assert(_is_valid);
    if (__builtin_expect(!_is_valid, 0)) { return false; }

    read_current_tuple(output_buffer);

    // Start from the rightmost iterator and try to advance
    int start_idx = static_cast<int>(_num_attributes) - 1;
    while (true) {
        // Try to advance from start_idx backwards until we find one that can advance
        int updated_itr_idx = start_idx;
        for (; updated_itr_idx >= 0; updated_itr_idx--) {
            if (try_advance(static_cast<size_t>(updated_itr_idx))) { break; }
        }

        // Could not advance any iterator - all exhausted
        if (__builtin_expect(updated_itr_idx < 0, 0)) {
            _is_valid = false;
            return false;
        }

        // Reset all iterators after updated_itr_idx
        // Since parent always comes before child in ordering, we can iterate sequentially
        bool all_valid = true;
        const size_t reset_start = static_cast<size_t>(updated_itr_idx) + 1;
        const size_t reset_end = _num_attributes;

        for (size_t i = reset_start; i < reset_end && all_valid; ++i) {
            all_valid = reset_iterator_to_start(i);
        }

        if (__builtin_expect(all_valid, 1)) { return true; }

        // Some iterator had empty range, continue trying from the updated iterator
        start_idx = updated_itr_idx;
    }
}

bool FTreeIterator::next_min_tuple(uint64_t* __restrict__ output_buffer) {
    if (__builtin_expect(!_is_valid, 0)) { return false; }

    // Always process current tuple first (same contract as next())
    update_min_vals_current_tuple(output_buffer);

    // Start from the rightmost iterator and try to advance
    int start_idx = static_cast<int>(_num_attributes) - 1;
    while (true) {
        // Try to advance from start_idx backwards until we find one that can advance
        int updated_itr_idx = start_idx;
        for (; updated_itr_idx >= 0; updated_itr_idx--) {
            if (try_advance(static_cast<size_t>(updated_itr_idx))) { break; }
        }

        // Could not advance any iterator - current tuple was the last valid one
        if (__builtin_expect(updated_itr_idx < 0, 0)) {
            _is_valid = false;
            return true;
        }

        // Reset all iterators after updated_itr_idx
        // Since parent always comes before child in ordering, we can iterate sequentially
        bool all_valid = true;
        const size_t reset_start = static_cast<size_t>(updated_itr_idx) + 1;
        const size_t reset_end = _num_attributes;

        for (size_t i = reset_start; i < reset_end && all_valid; ++i) {
            all_valid = reset_iterator_to_start(i);
        }

        if (__builtin_expect(all_valid, 1)) { return true; }

        // Some iterator had empty range, continue trying from the updated iterator
        start_idx = updated_itr_idx;
    }
}


bool FTreeIterator::current(uint64_t* output_buffer) const {
    assert(_is_valid);
    read_current_tuple(output_buffer);
    return true;
}

void FTreeIterator::reset() { initialize_iterators(); }

}// namespace ffx
