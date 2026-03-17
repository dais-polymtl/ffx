#include "../include/factorized_ftree/ftree_ancestor_finder.hpp"
#include "../include/vector/bitmask.hpp"

#include <cassert>
#include <cstring>

namespace ffx {
namespace {

inline void validate_state_path(const std::vector<const State*>& state_path, const char* prefix) {
    if (state_path.size() < 2) {
        throw std::runtime_error(std::string(prefix) + ": state_path must have at least 2 states");
    }

    for (size_t i = 1; i < state_path.size(); i++) {
        if (state_path[i] == state_path[i - 1]) {
            throw std::runtime_error(std::string(prefix) +
                                     ": state_path contains duplicate/shared states - "
                                     "caller should filter these out");
        }
    }
}

inline uint64_t block_range_mask(size_t block, size_t start_block, size_t end_block, size_t start_bit, size_t end_bit) {
    uint64_t mask = ~0ULL;
    if (block == start_block) {
        mask &= (~0ULL << start_bit);
    }
    if (block == end_block) {
        const uint64_t end_mask = (end_bit == 63) ? ~0ULL : ((1ULL << (end_bit + 1)) - 1);
        mask &= end_mask;
    }
    return mask;
}

// Resolve a parent index for a child index using a monotonic cursor.
// Skips parents whose selector bit is cleared (stale offsets).
// For increasing child_idx values, parent cursor only moves forward.
inline uint32_t find_parent_from_child_linear(const State* parent_state, const State* child_state, uint32_t child_idx,
                                              int32_t& parent_cursor) {
    const int32_t parent_start = GET_START_POS(*parent_state);
    const int32_t parent_end = GET_END_POS(*parent_state);
    if (parent_start > parent_end) return UINT32_MAX;

    if (parent_cursor < parent_start) parent_cursor = parent_start;
    if (parent_cursor > parent_end) return UINT32_MAX;

    const uint16_t* offsets = child_state->offset;
    const auto& parent_selector = parent_state->selector;

    while (parent_cursor <= parent_end) {
        if (!TEST_BIT(parent_selector, parent_cursor)) {
            parent_cursor++;
            continue;
        }
        if (child_idx < offsets[parent_cursor + 1]) break;
        parent_cursor++;
    }
    if (parent_cursor > parent_end) return UINT32_MAX;

    const int32_t parent_idx = parent_cursor;
    if (offsets[parent_idx] > child_idx || child_idx >= offsets[parent_idx + 1]) return UINT32_MAX;
    return static_cast<uint32_t>(parent_idx);
}

}// namespace

// ============================================================================
// FtreeAncestorFinder Implementation (State-based)
// ============================================================================

FtreeAncestorFinder::FtreeAncestorFinder(const State* const* state_path, size_t path_size)
    : _state_path(state_path, state_path + path_size) {
    validate_state_path(_state_path, "FtreeAncestorFinder");
}

void FtreeAncestorFinder::reset(const State* const* state_path, size_t path_size) {
    _state_path.assign(state_path, state_path + path_size);
    validate_state_path(_state_path, "FtreeAncestorFinder");
}

void FtreeAncestorFinder::process(uint32_t* output_buffer, int32_t ancestor_start, int32_t ancestor_end,
                                  int32_t descendant_start, int32_t descendant_end) {
    std::memset(output_buffer + descendant_start, 0xFF, sizeof(uint32_t) * (descendant_end - descendant_start + 1));

    const size_t num_levels = _state_path.size();
    auto* parent_cursors = static_cast<int32_t*>(alloca(sizeof(int32_t) * (num_levels - 1)));
    for (size_t level = 1; level < num_levels; ++level) {
        parent_cursors[level - 1] = GET_START_POS(*_state_path[level - 1]);
    }

    const auto& descendant_selector = _state_path.back()->selector;
    const size_t start_block = static_cast<size_t>(descendant_start) >> 6;
    const size_t end_block = static_cast<size_t>(descendant_end) >> 6;
    const size_t start_bit = static_cast<size_t>(descendant_start) & 63U;
    const size_t end_bit = static_cast<size_t>(descendant_end) & 63U;

    for (size_t block = start_block; block <= end_block; ++block) {
        uint64_t block_val = descendant_selector.bits[block] &
                             block_range_mask(block, start_block, end_block, start_bit, end_bit);
        while (block_val != 0) {
            const int bit_pos = __builtin_ctzll(block_val);
            const uint32_t d_idx = static_cast<uint32_t>((block << 6) | static_cast<size_t>(bit_pos));
            block_val &= (block_val - 1);

            uint32_t current_idx = d_idx;
            bool valid_path = true;
            for (int level = static_cast<int>(num_levels) - 1; level > 0; --level) {
                const State* parent_state = _state_path[static_cast<size_t>(level - 1)];
                const State* child_state = _state_path[static_cast<size_t>(level)];
                int32_t& cursor = parent_cursors[static_cast<size_t>(level - 1)];
                const uint32_t parent_idx = find_parent_from_child_linear(parent_state, child_state, current_idx, cursor);
                if (parent_idx == UINT32_MAX) {
                    valid_path = false;
                    break;
                }
                current_idx = parent_idx;
            }
            if (!valid_path ||
                static_cast<int32_t>(current_idx) < ancestor_start ||
                static_cast<int32_t>(current_idx) > ancestor_end) {
                continue;
            }
            output_buffer[d_idx] = current_idx;
        }
    }
}

// ============================================================================
// FtreeMultiAncestorFinder Implementation (State-based)
// ============================================================================

FtreeMultiAncestorFinder::FtreeMultiAncestorFinder(std::vector<const State*> state_path)
    : _state_path(std::move(state_path)) {
    if (_state_path.size() < 2) {
        throw std::runtime_error("FtreeMultiAncestorFinder: state_path must have at least 2 states");
    }

    for (size_t i = 1; i < _state_path.size(); i++) {
        if (_state_path[i] == _state_path[i - 1]) {
            throw std::runtime_error("FtreeMultiAncestorFinder: state_path contains duplicate/shared states");
        }
    }
}

void FtreeMultiAncestorFinder::process(uint32_t** output_buffers, int32_t descendant_start, int32_t descendant_end) {
    const size_t num_levels = _state_path.size();
    const size_t num_ancestors = num_levels - 1;

    for (size_t i = 0; i < num_ancestors; i++) {
        std::memset(output_buffers[i] + descendant_start, 0xFF,
                    sizeof(uint32_t) * (descendant_end - descendant_start + 1));
    }

    auto* parent_cursors = static_cast<int32_t*>(alloca(sizeof(int32_t) * (num_levels - 1)));
    for (size_t level = 1; level < num_levels; ++level) {
        parent_cursors[level - 1] = GET_START_POS(*_state_path[level - 1]);
    }

    const auto& descendant_selector = _state_path.back()->selector;
    const size_t start_block = static_cast<size_t>(descendant_start) >> 6;
    const size_t end_block = static_cast<size_t>(descendant_end) >> 6;
    const size_t start_bit = static_cast<size_t>(descendant_start) & 63U;
    const size_t end_bit = static_cast<size_t>(descendant_end) & 63U;

    for (size_t block = start_block; block <= end_block; ++block) {
        uint64_t block_val = descendant_selector.bits[block] &
                             block_range_mask(block, start_block, end_block, start_bit, end_bit);
        while (block_val != 0) {
            const int bit_pos = __builtin_ctzll(block_val);
            const uint32_t d_idx = static_cast<uint32_t>((block << 6) | static_cast<size_t>(bit_pos));
            block_val &= (block_val - 1);

            uint32_t current_idx = d_idx;
            for (int level = static_cast<int>(num_levels) - 1; level > 0; --level) {
                const State* parent_state = _state_path[static_cast<size_t>(level - 1)];
                const State* child_state = _state_path[static_cast<size_t>(level)];
                int32_t& cursor = parent_cursors[static_cast<size_t>(level - 1)];
                const uint32_t parent_idx =
                    find_parent_from_child_linear(parent_state, child_state, current_idx, cursor);
                if (parent_idx == UINT32_MAX) break;
                output_buffers[level - 1][d_idx] = parent_idx;
                current_idx = parent_idx;
            }
        }
    }
}

}// namespace ffx
