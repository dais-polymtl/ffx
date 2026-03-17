#include "factorized_ftree/ftree_state_update.hpp"
#include <array>
#include <cassert>
#include <iostream>
namespace ffx {

struct CascadePropResult {
    std::array<uint32_t, State::MAX_VECTOR_SIZE> invalidated_indices{};// indices of vector that were invalidated
    int32_t start = 0, end = 0;// start and end (inclusive) indices for entries in invalidated_indices
    bool result = false;       // true if any updates were made
};

inline bool has_any_bit_set_in_range(const uint64_t* bits, uint32_t cstart, uint32_t cend) {
    const std::size_t start_block = cstart >> 6;
    const std::size_t end_block = (cend - 1) >> 6;

    const std::size_t start_bit = cstart & 63;
    const std::size_t end_bit = (cend - 1) & 63;

    // Create masks for first and last blocks
    const uint64_t first_mask = ~0ULL << start_bit;
    const uint64_t last_mask = (end_bit == 63) ? ~0ULL : ((1ULL << (end_bit + 1)) - 1);

    // If single block, combine both masks
    if (start_block == end_block) { return (bits[start_block] & first_mask & last_mask) != 0; }

    // Check first block
    if ((bits[start_block] & first_mask) != 0) { return true; }

    // Check middle blocks
    for (std::size_t block = start_block + 1; block < end_block; ++block) {
        if (bits[block] != 0) { return true; }
    }

    // Check last block
    return (bits[end_block] & last_mask) != 0;
}


template<typename Selector>
inline int32_t clear_bits_in_range_and_collect(const uint64_t* selector_bits, Selector& child_selector,
                                               uint32_t cstart, uint32_t cend,
                                               std::array<uint32_t, State::MAX_VECTOR_SIZE>& invalidated_indices,
                                               int32_t& current_end) {
    int32_t num_cleared = 0;
    const std::size_t start_block = cstart >> 6;
    const std::size_t end_block = (cend - 1) >> 6;

    const std::size_t start_bit = cstart & 63;
    const std::size_t end_bit = (cend - 1) & 63;

    // Create masks for first and last blocks
    const uint64_t first_mask = ~0ULL << start_bit;
    const uint64_t last_mask = (end_bit == 63) ? ~0ULL : ((1ULL << (end_bit + 1)) - 1);

    // Process all blocks in a single loop
    for (std::size_t block = start_block; block <= end_block; ++block) {
        // Determine which mask to apply
        uint64_t mask = ~0ULL;
        if (block == start_block) mask &= first_mask;
        if (block == end_block) mask &= last_mask;

        // Get bits that are both set and within range
        uint64_t block_val = selector_bits[block] & mask;

        if (block_val == 0) continue;

        // Clear all these bits at once in the child_selector
        child_selector.bits[block] &= ~(block_val);

        // Process each set bit to collect indices
        while (block_val != 0) {
            int bit_pos = __builtin_ctzll(block_val);
            const auto cidx = (block << 6) | bit_pos;

            invalidated_indices[current_end++] = cidx;
            num_cleared++;
            // Clear the lowest set bit from our working copy
            block_val &= (block_val - 1);
        }
    }

    return num_cleared;
}

template<typename Selector>
inline void find_first_last_set_bits(const Selector& selector, const int32_t start_pos, const int32_t end_pos,
                                     int32_t& new_start, int32_t& new_end) {
    // Find first set bit from start_pos
    new_start = end_pos + 1;// Default to invalid if not found
    for (std::size_t block = start_pos >> 6; block <= (std::size_t)(end_pos >> 6); ++block) {
        uint64_t block_val = selector.bits[block];

        // Mask off bits before start_pos in first block
        if (block == (std::size_t)(start_pos >> 6)) { block_val &= ~0ULL << (start_pos & 63); }

        // Mask off bits after end_pos in last block
        if (block == (std::size_t)(end_pos >> 6)) {
            const std::size_t end_bit = end_pos & 63;
            block_val &= (end_bit == 63) ? ~0ULL : ((1ULL << (end_bit + 1)) - 1);
        }

        if (block_val != 0) {
            new_start = (block << 6) | __builtin_ctzll(block_val);
            break;
        }
    }

    // Find last set bit from end_pos
    new_end = start_pos - 1;// Default to invalid if not found
    for (int64_t block = end_pos >> 6; block >= (int64_t) (start_pos >> 6); --block) {
        uint64_t block_val = selector.bits[block];

        // Mask off bits after end_pos in last block
        if (block == (int64_t) (end_pos >> 6)) {
            const std::size_t end_bit = end_pos & 63;
            block_val &= (end_bit == 63) ? ~0ULL : ((1ULL << (end_bit + 1)) - 1);
        }

        // Mask off bits before start_pos in first block
        if (block == (int64_t) (start_pos >> 6)) { block_val &= ~0ULL << (start_pos & 63); }

        if (block_val != 0) {
            new_end = (block << 6) | (63 - __builtin_clzll(block_val));
            break;
        }
    }
}

static CascadePropResult forward_update_cascade(const Vector<uint64_t>* to, const Vector<uint64_t>* from,
                                                bool& is_vector_empty) {
    auto& parent_state = *to->state;
    auto& parent_selector = parent_state.selector;
    const auto& child_selector = from->state->selector;
    const int32_t parent_active_start = GET_START_POS(parent_state);
    const int32_t parent_active_end = GET_END_POS(parent_state);
    const int32_t child_active_start = GET_START_POS(*from->state);
    const int32_t child_active_end = GET_END_POS(*from->state);
    assert(TEST_BIT(parent_selector, parent_active_start));
    assert(TEST_BIT(parent_selector, parent_active_end));
    assert(TEST_BIT(from->state->selector, child_active_start));
    assert(TEST_BIT(from->state->selector, child_active_end));

    CascadePropResult cascade_result;
    bool& result = cascade_result.result;
    auto& invalidated_indices = cascade_result.invalidated_indices;
    auto& end = cascade_result.end;
    auto& start = cascade_result.start;

    for (auto parent_idx = parent_active_start; parent_idx <= parent_active_end; parent_idx++) {
        if (!TEST_BIT(parent_selector, parent_idx)) { continue; }

        uint32_t cstart;
        uint32_t cend;
        if (from->state == to->state) {
            // Identity RLE: child positions == parent positions
            cstart = static_cast<uint32_t>(parent_idx);
            cend = static_cast<uint32_t>(parent_idx + 1);
        } else {
            const uint16_t* child_offset = from->state->offset;
            cstart = child_offset[parent_idx];
            cend = child_offset[parent_idx + 1];
        }
        if (!has_any_bit_set_in_range(child_selector.bits, cstart, cend)) {
            //result = true;
            CLEAR_BIT(parent_selector, parent_idx);
            invalidated_indices[end++] = parent_idx;
        }
    }

    // // Find new parent start
    // int32_t new_parent_start;
    // for (new_parent_start = parent_active_start; new_parent_start <= parent_active_end; new_parent_start++) {
    //     if (TEST_BIT(parent_selector, new_parent_start)) { break; }
    // }
    //
    // // Find new parent end
    // int32_t new_parent_end;
    // for (new_parent_end = parent_active_end; new_parent_end >= parent_active_start; new_parent_end--) {
    //     if (TEST_BIT(parent_selector, new_parent_end)) { break; }
    // }

    int32_t new_parent_start, new_parent_end;
    find_first_last_set_bits(parent_selector, parent_active_start, parent_active_end, new_parent_start, new_parent_end);

    assert(TEST_BIT(parent_selector, new_parent_start));
    assert(TEST_BIT(parent_selector, new_parent_end));

    // Apply new parent state positions
    SET_START_POS(parent_state, new_parent_start);
    SET_END_POS(parent_state, new_parent_end);

    if (new_parent_start > new_parent_end) {
        is_vector_empty = true;
        result = false;
        return cascade_result;
    }

    // We need to shrink the invalidated indices array to only contain the valid entries after the new start and end pos
    // are selected for the parent vector
    for (; start < end; start++) {
        assert(static_cast<int32_t>(invalidated_indices[start]) != new_parent_start);
        if (static_cast<int32_t>(invalidated_indices[start]) >
            new_parent_start) {// can never be equal since new_parent_start is always valid
            break;
        }
    }

    for (; end > start; end--) {
        assert(static_cast<int32_t>(invalidated_indices[end - 1]) != new_parent_end);
        if (static_cast<int32_t>(invalidated_indices[end - 1]) <
            new_parent_end) {// can never be equal since new_parent_end is always valid
            break;
        }
    }

    result = (start < end);
    return cascade_result;
}

static CascadePropResult
backward_update_cascade(const Vector<uint64_t>* to, const Vector<uint64_t>* from,
                        const std::array<uint32_t, State::MAX_VECTOR_SIZE>& invalidated_indices,
                        const std::size_t& start, const std::size_t& end, bool& is_vector_empty) {

    const auto& parent_state = *from->state;
    const auto& parent_selector = parent_state.selector;
    auto& child_state = *to->state;
    auto& child_selector = child_state.selector;

    const int32_t parent_active_start = GET_START_POS(parent_state);
    const int32_t parent_active_end = GET_END_POS(parent_state);
    const int32_t child_active_start = GET_START_POS(child_state);
    const int32_t child_active_end = GET_END_POS(child_state);

    assert(TEST_BIT(parent_selector, parent_active_start));
    assert(TEST_BIT(parent_selector, parent_active_end));
    assert(TEST_BIT(child_selector, child_active_start));
    assert(TEST_BIT(child_selector, child_active_end));

    CascadePropResult cascade_result;
    bool& result = cascade_result.result;
    auto& new_invalidated_indices = cascade_result.invalidated_indices;
    auto& new_start = cascade_result.start;
    auto& new_end = cascade_result.end;

    for (auto i = start; i < end; i++) {
        const auto pidx = invalidated_indices[i];
        // These asserts are not valid, since the parent start/end pos can be modified after the range update
        // assert (pidx > parent_active_start);
        // assert (pidx < parent_active_end);
        uint32_t cstart;
        uint32_t cend;
        if (to->state == from->state) {
            // Identity RLE: child positions == parent positions
            cstart = pidx;
            cend = pidx + 1;
        } else {
            const uint16_t* rle = to->state->offset;
            cstart = rle[pidx];
            cend = rle[pidx + 1];
        }

        // we dont have to check if cidx lie in the child active range since we can turn all the values off
        clear_bits_in_range_and_collect(child_selector.bits, child_selector, cstart, cend, new_invalidated_indices,
                                        new_end);

        // if (num_cleared > 0) {
        //     result = true;
        // }
    }

    // for (auto i = start; i < end; i++) {
    //     const auto pidx = invalidated_indices[i];
    //     assert (pidx > parent_active_start);
    //     assert (pidx < parent_active_end);
    //     const auto cstart = rle[pidx];
    //     const auto cend = rle[pidx + 1]; // we dont have to check if cidx lie in the child active range since we can turn all the values off
    //     for (auto cidx = cstart; cidx < cend; cidx++) {
    //         if (!TEST_BIT(child_selector, cidx)) { continue; }// already invalid
    //         CLEAR_BIT(child_selector, cidx);
    //         result = true;
    //         new_invalidated_indices[new_end++] = cidx;
    //     }
    // }


    // int32_t new_child_start;
    // for (new_child_start = child_active_start; new_child_start <= child_active_end; new_child_start++) {
    //     if (TEST_BIT(child_selector, new_child_start)) { break; }
    // }
    //
    // int32_t new_child_end;
    // for (new_child_end = child_active_end; new_child_end >= child_active_start; new_child_end--) {
    //     if (TEST_BIT(child_selector, new_child_end)) { break; }
    // }

    int32_t new_child_start, new_child_end;
    find_first_last_set_bits(child_selector, child_active_start, child_active_end, new_child_start, new_child_end);

    if (new_child_start > new_child_end) {
        is_vector_empty = true;
        result = false;
        return cascade_result;
    }

    assert(TEST_BIT(child_selector, new_child_start));
    assert(TEST_BIT(child_selector, new_child_end));

    SET_START_POS(child_state, new_child_start);
    SET_END_POS(child_state, new_child_end);

    // We need to shrink the invalidated indices array to only contain the valid entries after the new start and end pos
    // are selected for the child vector
    for (; new_start < new_end; new_start++) {
        assert(static_cast<int32_t>(new_invalidated_indices[new_start]) !=
               new_child_start);// can never be equal since new_child_start is always valid
        if (static_cast<int32_t>(new_invalidated_indices[new_start]) > new_child_start) { break; }
    }
    for (; new_end > new_start; new_end--) {
        assert(static_cast<int32_t>(new_invalidated_indices[new_end - 1]) !=
               new_child_end);// can never be equal since new_child_end is always valid
        if (static_cast<int32_t>(new_invalidated_indices[new_end - 1]) < new_child_end) { break; }
    }

    result = (new_start < new_end);// new_end is exclusive
    return cascade_result;
}

bool forward_update(const Vector<uint64_t>* to, const Vector<uint64_t>* from, bool& is_vector_empty) {
    const auto& child_state = *from->state;
    const auto& child_selector = child_state.selector;
    auto& parent_state = *to->state;
    const auto& parent_selector = parent_state.selector;
    const int32_t parent_active_start = GET_START_POS(parent_state);
    const int32_t parent_active_end = GET_END_POS(parent_state);
    const int32_t child_active_start = GET_START_POS(child_state);
    const int32_t child_active_end = GET_END_POS(child_state);
    assert(TEST_BIT(parent_selector, parent_active_start));
    assert(TEST_BIT(parent_selector, parent_active_end));
    assert(TEST_BIT(child_selector, child_active_start));
    assert(TEST_BIT(child_selector, child_active_end));

    // Initialize to current values (fallback case)
    auto prev_parent_start = parent_active_start;
    int32_t new_parent_start;
    for (new_parent_start = parent_active_start; new_parent_start <= parent_active_end; new_parent_start++) {
        if (TEST_BIT(parent_selector, new_parent_start)) {
            uint32_t parent_child_start;
            if (from->state == to->state) {
                // Identity RLE: child positions == parent positions
                parent_child_start = static_cast<uint32_t>(new_parent_start);
            } else {
                const uint16_t* rle = from->state->offset;
                parent_child_start = rle[new_parent_start];
            }
            if (static_cast<int32_t>(parent_child_start) > child_active_start) { break; }
            prev_parent_start = new_parent_start;
        }
    }
    new_parent_start = prev_parent_start;

    auto prev_parent_end = parent_active_end;
    int32_t new_parent_end;
    for (new_parent_end = new_parent_start; new_parent_end <= parent_active_end; new_parent_end++) {
        if (TEST_BIT(parent_selector, new_parent_end)) {
            uint32_t parent_child_start;
            if (from->state == to->state) {
                parent_child_start = static_cast<uint32_t>(new_parent_end);
            } else {
                const uint16_t* rle = from->state->offset;
                parent_child_start = rle[new_parent_end];
            }
            if (static_cast<int32_t>(parent_child_start) > child_active_end) { break; }
            prev_parent_end = new_parent_end;
        }
    }
    new_parent_end = prev_parent_end;

    // Check if the new range is valid
    if (new_parent_start > new_parent_end) {
        is_vector_empty = true;
        return false;
    }

    // Apply new parent slice positions
    SET_START_POS(parent_state, new_parent_start);
    SET_END_POS(parent_state, new_parent_end);


    assert(TEST_BIT(parent_selector, new_parent_start));
    assert(TEST_BIT(parent_selector, new_parent_end));

    // We don't assert on the end idx,
    // rather we assert on the number of elements since it can produce 0 elements as well.
    // The idea is a valid start or end idx might produce 0 elements,
    // due to how we handle the child start/end pos inside backward prop
    uint32_t range_start, range_end;
    if (from->state == to->state) {
        range_start = static_cast<uint32_t>(new_parent_start);
        range_end = static_cast<uint32_t>(new_parent_start + 1);
    } else {
        const uint16_t* rle = from->state->offset;
        range_start = rle[new_parent_start];
        range_end = rle[new_parent_start + 1];
    }
    assert(range_start < State::MAX_VECTOR_SIZE);
    assert(range_end <= State::MAX_VECTOR_SIZE);
    assert(range_start <= range_end);

    // Asserts go here for everything between new_parent_start and new_parent_end
    for (auto pidx = new_parent_start + 1; pidx < new_parent_end; pidx++) {
        if (!TEST_BIT(parent_selector, pidx)) { continue; }
        uint32_t range_start2, range_end2;
        const bool is_identity = (from->state == to->state);
        if (is_identity) {
            range_start2 = static_cast<uint32_t>(pidx);
            range_end2 = static_cast<uint32_t>(pidx + 1);
        } else {
            const uint16_t* rle = from->state->offset;
            range_start2 = rle[pidx];
            range_end2 = rle[pidx + 1] - 1;
        }
        assert(range_start2 < State::MAX_VECTOR_SIZE);
        assert(range_end2 <= State::MAX_VECTOR_SIZE);
        assert(range_start2 <= range_end2);
        // These strict inequalities do not hold for identity-RLE.
        if (!is_identity) {
            assert(static_cast<int32_t>(range_start2) > child_active_start);
            assert(static_cast<int32_t>(range_end2) < child_active_end);
        }
    }

    uint32_t range_start1, range_end1;
    if (from->state == to->state) {
        range_start1 = static_cast<uint32_t>(new_parent_end);
        range_end1 = static_cast<uint32_t>(new_parent_end + 1);
    } else {
        const uint16_t* rle = from->state->offset;
        range_start1 = rle[new_parent_end];
        range_end1 = rle[new_parent_end + 1];
    }
    assert(range_start1 < State::MAX_VECTOR_SIZE);
    assert(range_end1 <= State::MAX_VECTOR_SIZE);
    assert(range_start1 <= range_end1);


    return (parent_active_start != new_parent_start) || (parent_active_end != new_parent_end);
}

bool backward_update(const Vector<uint64_t>* to, const Vector<uint64_t>* from, bool& is_vector_empty) {
    const auto& parent_state = *from->state;
    const auto& parent_selector = parent_state.selector;
    const auto& child_selector = to->state->selector;
    uint16_t* rle = to->state->offset;
    auto& child_state = *to->state;
    const int32_t child_active_start = GET_START_POS(child_state);
    const int32_t child_active_end = GET_END_POS(child_state);

    assert(TEST_BIT(parent_selector, GET_START_POS(parent_state)));
    assert(TEST_BIT(parent_selector, GET_END_POS(parent_state)));
    assert(TEST_BIT(child_selector, child_active_start));
    assert(TEST_BIT(child_selector, child_active_end));

    // Initialize to current values (fallback case)
    auto new_child_start_op = std::max(static_cast<int32_t>(rle[GET_START_POS(parent_state)]),
                                       static_cast<int32_t>(GET_START_POS(child_state)));
    auto new_child_end_op = std::min(static_cast<int32_t>(rle[GET_END_POS(parent_state) + 1] - 1),
                                     static_cast<int32_t>(GET_END_POS(child_state)));

    // Early exit if parent and child ranges don't overlap
    if (new_child_start_op > new_child_end_op) {
        is_vector_empty = true;
        return false;
    }

    int32_t new_child_start, new_child_end;
    find_first_last_set_bits(child_selector, new_child_start_op, new_child_end_op, new_child_start, new_child_end);

    if (new_child_start > new_child_end) {
        is_vector_empty = true;
        return false;
    }

    // Initially, the parent range determines the new child start and end.
    // However, we need to ensure the new child start and end are also valid positions in the child selector.
    // If they are not valid, we need to move them to the next valid position.
    // This is done in the loops above.
    // This implies that the parent start and end are now not the correct positions.
    // Therefore, we modify the rle entries of the parent to ensure that they point to valid positions in the child selector.
    // Update the parent selector to remove entries that no longer have any children after moving the child start/end to
    // valid positions
    auto actuaL_parent_start = GET_START_POS(parent_state);
    for (auto pidx = GET_START_POS(parent_state) + 1; pidx < GET_END_POS(parent_state); pidx++) {
        if (TEST_BIT(parent_selector, pidx)) {
            const auto parent_child_start = rle[pidx];
            if (static_cast<int32_t>(parent_child_start) > new_child_start) { break; }
            actuaL_parent_start = pidx;
        }
    }
    auto actual_parent_end = GET_END_POS(parent_state);
    for (auto pidx = GET_END_POS(parent_state) - 1; pidx > GET_START_POS(parent_state); pidx--) {
        if (TEST_BIT(parent_selector, pidx)) {
            const auto parent_child_end = rle[pidx + 1] - 1;
            if (static_cast<int32_t>(parent_child_end) < new_child_end) { break; }
            actual_parent_end = pidx;
        }
    }
    for (auto pidx = GET_START_POS(parent_state); pidx < actuaL_parent_start; pidx++) {
        if (TEST_BIT(parent_selector, pidx)) { rle[pidx + 1] = rle[pidx]; }
    }

    for (auto pidx = actual_parent_end + 1; pidx <= GET_END_POS(parent_state); pidx++) {
        if (TEST_BIT(parent_selector, pidx)) { rle[pidx + 1] = rle[pidx]; }
    }

    SET_START_POS(child_state, new_child_start);
    SET_END_POS(child_state, new_child_end);

    assert(TEST_BIT(parent_selector, GET_START_POS(parent_state)));
    assert(TEST_BIT(parent_selector, GET_END_POS(parent_state)));
    assert(TEST_BIT(child_selector, new_child_start));
    assert(TEST_BIT(child_selector, new_child_end));

    //Asserts go here for everything between new_parent_start and new_parent_end

    // 1) All parent indices between parent_start (inclusive) and actual_parent_start
    // (exclusive) must not produce any children.
    // 2) All parent indices between actual_parent_end (exclusive) and parent_end (inclusive)
    // must not produce any children.

    for (auto pidx = GET_START_POS(parent_state); pidx < actuaL_parent_start; pidx++) {
        if (!TEST_BIT(parent_selector, pidx)) { continue; }
        const auto range_start = rle[pidx];
        const auto range_end = rle[pidx + 1];
        assert(range_start == range_end);
    }

    for (auto pidx = actual_parent_end + 1; pidx <= GET_END_POS(parent_state); pidx++) {
        if (!TEST_BIT(parent_selector, pidx)) { continue; }
        const auto range_start = rle[pidx];
        const auto range_end = rle[pidx + 1];
        assert(range_start == range_end);
    }

    //for (auto pidx = GET_START_POS(parent_state) + 1; pidx < GET_END_POS(parent_state); pidx++) {
    for (auto pidx = actuaL_parent_start + 1; pidx < actual_parent_end; pidx++) {
        if (!TEST_BIT(parent_selector, pidx)) { continue; }
        const auto range_start = rle[pidx];
        const auto range_end = rle[pidx + 1] - 1;
        // auto effective_start = std::max(child_active_start, (int32_t) range_start);
        // auto effective_end = std::min(child_active_end, (int32_t) range_end);
        assert(static_cast<int32_t>(range_start) > new_child_start);
        assert(static_cast<int32_t>(range_end) < new_child_end);
        assert(range_start < State::MAX_VECTOR_SIZE);
        assert(range_end < State::MAX_VECTOR_SIZE);
    }

    return (new_child_start != child_active_start) || (new_child_end != child_active_end);
}


FtreeStateUpdateNode::FtreeStateUpdateNode(const Vector<uint64_t>* vec, slice_update_type t, const std::string& attr)
    : vector(vec), type(t), parent(nullptr), attribute(attr) {}

bool FtreeStateUpdateNode::start_propagation() {
    bool is_vector_empty = false;
    for (auto* child: effective_children) {
        child->update_range(is_vector_empty);
        if (is_vector_empty) return true;
    }
    return false;
}

bool FtreeStateUpdateNode::update_range(bool& is_vector_empty) {
    auto is_updated = false;
    switch (type) {
        case FORWARD: {
            is_updated = forward_update(this->vector /*vector to update*/, parent->vector /*last updated vector*/,
                                        is_vector_empty);
            break;
        }
        case BACKWARD: {
            is_updated = backward_update(this->vector /*vector to update*/, parent->vector /*last updated vector*/,
                                         is_vector_empty);
            break;
        }
        default:
            return false;
    }

    // If empty vector encountered, suspend the update process, and return immediately
    if (is_vector_empty) { return false; }

    if (is_updated) {
        for (auto* child: effective_children) {
            child->update_range(is_vector_empty);
            if (is_vector_empty) return false;
        }
    }
    return is_updated;
}

bool FtreeStateUpdateNode::start_propagation_fwd_cascade(const uint32_t* gp_invalidated_indices,
                                                         const int32_t& gp_invalidated_count) {
    // Nothing to propagate if tree has no children
    if (children.empty()) { return false; }

    // We already know which parent indices were invalidated
    // So we just need to invalidate the gp indices.
    // Therefore, it is a simplified fwd cascade

    // First, set up the required info for next propagation
    CascadePropResult result;// will hold exactly gp_invalidated_indices
    auto& new_invalid_indices = result.invalidated_indices;
    auto& start = result.start;
    auto& end = result.end;
    start = 0;
    end = 0;

    // Second, invalidate the indices in the current vector's ftree parent (update tree child)
    assert(children.size() == 1);
    const auto& cnode = children.back();
    const auto& cstate = cnode->vector->state;
    auto& cselector = cstate->selector;
    const auto start_pos = GET_START_POS(*cstate);
    const auto end_pos = GET_END_POS(*cstate);
    assert(TEST_BIT(cselector, start_pos));
    assert(TEST_BIT(cselector, end_pos));

    uint32_t cidx = 0;
    for (auto i = 0; i < gp_invalidated_count; i++) {
        // since range update might have changed the gp start and end pos
        if (static_cast<int32_t>(gp_invalidated_indices[i]) > start_pos &&
            static_cast<int32_t>(gp_invalidated_indices[i]) < end_pos) {
            cidx = gp_invalidated_indices[i];
            new_invalid_indices[end++] = cidx;
            CLEAR_BIT(cselector, cidx);
        }
    }

    // auto new_start_pos = start_pos;
    // for (; new_start_pos <= end_pos; new_start_pos++) {
    //     if (TEST_BIT(*cselector, new_start_pos)) { break; }
    // }
    // auto new_end_pos = end_pos;
    // for (; new_end_pos >= start_pos; new_end_pos--) {
    //     if (TEST_BIT(*cselector, new_end_pos)) { break; }
    // }

    int32_t new_start_pos, new_end_pos;
    find_first_last_set_bits(cselector, start_pos, end_pos, new_start_pos, new_end_pos);


    bool is_vector_empty = false;
    if (new_start_pos > new_end_pos) { is_vector_empty = true; }

    if (is_vector_empty) {
        // vector became empty, so no need to propagate further
        return is_vector_empty;
    }

    assert(TEST_BIT(cselector, new_start_pos));
    assert(TEST_BIT(cselector, new_end_pos));

    SET_START_POS(*cstate, new_start_pos);
    SET_END_POS(*cstate, new_end_pos);

    for (; start < end; start++) {
        if (static_cast<int32_t>(new_invalid_indices[start]) > new_start_pos) { break; }
    }

    for (; end > start; end--) {
        if (static_cast<int32_t>(new_invalid_indices[end - 1]) < new_end_pos) { break; }
    }

    result.result = end > start;

    if (result.result) {
        for (auto* child: cnode->effective_children) {
            child->update_range_cascade(is_vector_empty, result);
            if (is_vector_empty) return true;
        }
    }

    return is_vector_empty;
}


bool FtreeStateUpdateNode::start_propagation_cascade(const uint32_t* invalidated_indices,
                                                     const int32_t& invalidated_count) {
    // Nothing to propagate if tree has no effective children
    if (effective_children.empty()) { return false; }

    bool is_vector_empty = false;
    CascadePropResult result;// dummy initialization in case first propagation that happens is a backward one
    auto& invalid_indices = result.invalidated_indices;
    auto& start = result.start;
    auto& end = result.end;
    const auto& selector = vector->state->selector;
    const auto start_pos = GET_START_POS(*vector->state);
    const auto end_pos = GET_END_POS(*vector->state);
    assert(TEST_BIT(selector, start_pos));
    assert(TEST_BIT(selector, end_pos));
    assert(start == end);
    assert(start == 0);
    auto idx = 0;
    for (auto i = 0; i < invalidated_count; i++) {
        // Below check is required, since in the join,
        // we set the input vector state as the first valid idx we encounter,
        // but if the first indices are invalid, then our invalidated indices will contain
        // outside valid range indices as well, which we need to filter out here
        const auto inc = (static_cast<int32_t>(invalidated_indices[i]) > start_pos) &&
                         (static_cast<int32_t>(invalidated_indices[i]) < end_pos);
        invalid_indices[idx] = invalidated_indices[i];
        idx += inc;
    }

    result.result = idx > 0;
    end = idx;

    for (auto* child: effective_children) {
        child->update_range_cascade(is_vector_empty, result);
        if (is_vector_empty) return true;
    }
    return is_vector_empty;
}

bool FtreeStateUpdateNode::update_range_cascade(bool& is_vector_empty, const CascadePropResult& prev_result) {
    CascadePropResult cascade_result;
    switch (type) {
        case FORWARD: {
            cascade_result = forward_update_cascade(this->vector /*vector to update*/,
                                                    parent->vector /*last updated vector*/, is_vector_empty);
            break;
        }
        case BACKWARD: {
            cascade_result = backward_update_cascade(
                    this->vector /*vector to update*/, parent->vector /*last updated vector*/,
                    prev_result.invalidated_indices, prev_result.start, prev_result.end, is_vector_empty);

            break;
        }
        default:
            return false;
    }

    // If empty vector encountered, suspend the update process, and return immediately
    if (is_vector_empty) { return false; }

    const auto is_updated = cascade_result.result;
    if (is_updated) {
        for (auto* child: effective_children) {
            child->update_range_cascade(is_vector_empty, cascade_result);
            if (is_vector_empty) return false;
        }
    }
    return is_updated;
}


void FtreeStateUpdateNode::precompute_effective_children() {
    // Post-order: compact children first
    for (auto& child: children) {
        child->precompute_effective_children();
    }
    // For each direct child: if it shares State* with this node, absorb its
    // effective_children (they already point to the first boundary nodes below it).
    // If it has a different State*, it is itself an effective child.
    effective_children.clear();
    State* my_state = this->vector->state;
    for (auto& child: children) {
        if (child->vector->state == my_state) {
            // Same datachunk — absorb grandchildren
            for (auto* gc: child->effective_children) {
                effective_children.push_back(gc);
            }
        } else {
            effective_children.push_back(child.get());
        }
    }
}

static void fill_bwd_helper(FtreeStateUpdateNode* u_subtree, FactorizedTreeElement* f_subtree) {
    for (const auto& f_child: f_subtree->_children) {
        auto u_child = std::make_unique<FtreeStateUpdateNode>(f_child->_value, BACKWARD, f_child->_attribute);
        auto u_child_ptr = u_child.get();
        u_child_ptr->parent = u_subtree;
        u_subtree->children.push_back(std::move(u_child));
        fill_bwd_helper(u_child_ptr, f_child.get());
    }
}

void FtreeStateUpdateNode::fill_bwd(FactorizedTreeElement* ftree, const std::string& /*output_key*/) {
    auto curr_parent = this->parent;
    for (const auto& ftree_child: ftree->_children) {
        if (ftree_child->_attribute == curr_parent->attribute) continue;
        auto subtree = std::make_unique<FtreeStateUpdateNode>(ftree_child->_value, BACKWARD, ftree_child->_attribute);
        auto subtree_ptr = subtree.get();
        subtree_ptr->parent = this;
        this->children.push_back(std::move(subtree));
        fill_bwd_helper(subtree_ptr, ftree_child.get());
    }
}

/* For join_key node, we dont need to check for parent condition since its parent is null */
void FtreeStateUpdateNode::fill_bwd_join_key(FactorizedTreeElement* ftree, const std::string& output_key) {
    for (const auto& ftree_child: ftree->_children) {
        if (ftree_child->_attribute == output_key) continue;
        auto subtree = std::make_unique<FtreeStateUpdateNode>(ftree_child->_value, BACKWARD, ftree_child->_attribute);
        auto subtree_ptr = subtree.get();
        subtree_ptr->parent = this;
        this->children.push_back(std::move(subtree));
        fill_bwd_helper(subtree_ptr, ftree_child.get());
    }
}


void FtreeStateUpdateNode::print_tree(int depth) const {
    for (int i = 0; i < depth; ++i) {
        std::cout << "  ";
    }
    std::string type_str;
    switch (type) {
        case NONE:
            type_str = "NONE";
            break;
        case FORWARD:
            type_str = "FORWARD";
            break;
        case BACKWARD:
            type_str = "BACKWARD";
            break;
    }
    std::cout << attribute << " (" << type_str << ")";
    if (children.empty()) { std::cout << " [leaf]"; }
    std::cout << std::endl;
    for (const auto& child: children) {
        child->print_tree(depth + 1);
    }
}

void FtreeStateUpdateNode::print_effective_tree(int depth) const {
    for (int i = 0; i < depth; ++i) {
        std::cout << "  ";
    }
    std::string type_str;
    switch (type) {
        case NONE:
            type_str = "NONE";
            break;
        case FORWARD:
            type_str = "FORWARD";
            break;
        case BACKWARD:
            type_str = "BACKWARD";
            break;
    }
    std::cout << attribute << " (" << type_str << ")";
    if (effective_children.empty()) { std::cout << " [eff-leaf]"; }
    std::cout << std::endl;
    for (const FtreeStateUpdateNode* ec: effective_children) {
        ec->print_effective_tree(depth + 1);
    }
}

}// namespace ffx