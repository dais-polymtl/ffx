#include "../src/operator/include/factorized_ftree/factorized_tree_element.hpp"
#include "../src/operator/include/factorized_ftree/ftree_ancestor_finder.hpp"
#include "../src/operator/include/vector/bitmask.hpp"
#include "../src/operator/include/vector/state.hpp"
#include "../src/operator/include/vector/vector.hpp"
#include <gtest/gtest.h>

#include <memory>
#include <vector>

namespace {

class FtreeAncestorFinderTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create vectors for the test tree (packed mode = false, so they have rle and selector)
        vec_a = std::make_unique<ffx::Vector<uint64_t>>(false);
        vec_b = std::make_unique<ffx::Vector<uint64_t>>(false);
        vec_c = std::make_unique<ffx::Vector<uint64_t>>(false);
        vec_d = std::make_unique<ffx::Vector<uint64_t>>(false);

        // Initialize all selectors to all valid
        SET_ALL_BITS(vec_a->state->selector);
        SET_ALL_BITS(vec_b->state->selector);
        SET_ALL_BITS(vec_c->state->selector);
        SET_ALL_BITS(vec_d->state->selector);
    }

    void TearDown() override {
        // Clean up
    }

    std::unique_ptr<ffx::Vector<uint64_t>> vec_a, vec_b, vec_c, vec_d;
};

// Test 1: Direct parent-child relationship (two states)
TEST_F(FtreeAncestorFinderTest, DirectParentChild) {
    // Set up state for a: indices 0, 1 valid
    SET_START_POS(*vec_a->state, 0);
    SET_END_POS(*vec_a->state, 1);

    // Set up state for b: indices 0, 1, 2, 3 valid
    SET_START_POS(*vec_b->state, 0);
    SET_END_POS(*vec_b->state, 3);

    // Set up RLE: a[0] -> b[0,1], a[1] -> b[2,3]
    // NOTE: RLE is stored on the CHILD state (b), indexed by PARENT index (a_idx)
    vec_b->state->offset[0] = 0;
    vec_b->state->offset[1] = 2;// a[0] produces b[0], b[1]
    vec_b->state->offset[2] = 4;// a[1] produces b[2], b[3]

    // Create finder with state path [a, b]
    std::vector<const ffx::State*> state_path = {vec_a->state, vec_b->state};
    ffx::FtreeAncestorFinder finder(state_path.data(), state_path.size());

    EXPECT_EQ(finder.path_length(), 2);

    // Process
    uint32_t output[2048];
    finder.process(output, 0, 1, 0, 3);

    // b[0], b[1] should map to a[0]
    EXPECT_EQ(output[0], 0u);
    EXPECT_EQ(output[1], 0u);
    // b[2], b[3] should map to a[1]
    EXPECT_EQ(output[2], 1u);
    EXPECT_EQ(output[3], 1u);
}

// Test 2: Multi-level path (a -> b -> c)
TEST_F(FtreeAncestorFinderTest, MultiLevelPath) {
    // Set up state for a: indices 0, 1 valid
    SET_START_POS(*vec_a->state, 0);
    SET_END_POS(*vec_a->state, 1);

    // Set up state for b: indices 0, 1, 2, 3 valid
    SET_START_POS(*vec_b->state, 0);
    SET_END_POS(*vec_b->state, 3);

    // Set up state for c: indices 0-7 valid
    SET_START_POS(*vec_c->state, 0);
    SET_END_POS(*vec_c->state, 7);

    // Set up RLE for a -> b: a[0] -> b[0,1], a[1] -> b[2,3]
    vec_b->state->offset[0] = 0;
    vec_b->state->offset[1] = 2;
    vec_b->state->offset[2] = 4;

    // Set up RLE for b -> c: b[0] -> c[0,1], b[1] -> c[2,3], b[2] -> c[4,5], b[3] -> c[6,7]
    vec_c->state->offset[0] = 0;
    vec_c->state->offset[1] = 2;
    vec_c->state->offset[2] = 4;
    vec_c->state->offset[3] = 6;
    vec_c->state->offset[4] = 8;

    // Create finder with state path [a, b, c]
    std::vector<const ffx::State*> state_path = {vec_a->state, vec_b->state, vec_c->state};
    ffx::FtreeAncestorFinder finder(state_path.data(), state_path.size());

    EXPECT_EQ(finder.path_length(), 3);

    // Process
    uint32_t output[2048];
    finder.process(output, 0, 1, 0, 7);

    // c[0-3] should map to a[0] (through b[0] and b[1])
    EXPECT_EQ(output[0], 0u);
    EXPECT_EQ(output[1], 0u);
    EXPECT_EQ(output[2], 0u);
    EXPECT_EQ(output[3], 0u);

    // c[4-7] should map to a[1] (through b[2] and b[3])
    EXPECT_EQ(output[4], 1u);
    EXPECT_EQ(output[5], 1u);
    EXPECT_EQ(output[6], 1u);
    EXPECT_EQ(output[7], 1u);
}

// Test 3: Four-level path (a -> b -> c -> d)
TEST_F(FtreeAncestorFinderTest, FourLevelPath) {
    // Set up state for a: index 0 valid
    SET_START_POS(*vec_a->state, 0);
    SET_END_POS(*vec_a->state, 0);

    // Set up state for b: indices 0, 1 valid
    SET_START_POS(*vec_b->state, 0);
    SET_END_POS(*vec_b->state, 1);

    // Set up state for c: indices 0-3 valid
    SET_START_POS(*vec_c->state, 0);
    SET_END_POS(*vec_c->state, 3);

    // Set up state for d: indices 0-7 valid
    SET_START_POS(*vec_d->state, 0);
    SET_END_POS(*vec_d->state, 7);

    // Set up RLE for a -> b: a[0] -> b[0,1]
    vec_b->state->offset[0] = 0;
    vec_b->state->offset[1] = 2;

    // Set up RLE for b -> c: b[0] -> c[0,1], b[1] -> c[2,3]
    vec_c->state->offset[0] = 0;
    vec_c->state->offset[1] = 2;
    vec_c->state->offset[2] = 4;

    // Set up RLE for c -> d: c[0] -> d[0,1], c[1] -> d[2,3], c[2] -> d[4,5], c[3] -> d[6,7]
    vec_d->state->offset[0] = 0;
    vec_d->state->offset[1] = 2;
    vec_d->state->offset[2] = 4;
    vec_d->state->offset[3] = 6;
    vec_d->state->offset[4] = 8;

    // Create finder with state path [a, b, c, d]
    std::vector<const ffx::State*> state_path = {vec_a->state, vec_b->state, vec_c->state, vec_d->state};
    ffx::FtreeAncestorFinder finder(state_path.data(), state_path.size());

    EXPECT_EQ(finder.path_length(), 4);

    // Process
    uint32_t output[2048];
    finder.process(output, 0, 0, 0, 7);

    // All d indices should map to a[0]
    for (int i = 0; i <= 7; i++) {
        EXPECT_EQ(output[i], 0u) << "Failed at index " << i;
    }
}

// Test 4: Invalid indices are skipped
TEST_F(FtreeAncestorFinderTest, InvalidIndicesSkipped) {
    // Set up state for a: indices 0, 1, 2 valid but mask out index 1
    SET_START_POS(*vec_a->state, 0);
    SET_END_POS(*vec_a->state, 2);
    CLEAR_BIT(vec_a->state->selector, 1);// a[1] is invalid

    // Set up state for b: indices 0-5 valid
    SET_START_POS(*vec_b->state, 0);
    SET_END_POS(*vec_b->state, 5);

    // Set up RLE: a[0] -> b[0,1], a[1] -> b[2,3], a[2] -> b[4,5]
    vec_b->state->offset[0] = 0;
    vec_b->state->offset[1] = 2;
    vec_b->state->offset[2] = 4;
    vec_b->state->offset[3] = 6;

    // Create finder
    std::vector<const ffx::State*> state_path = {vec_a->state, vec_b->state};
    ffx::FtreeAncestorFinder finder(state_path.data(), state_path.size());

    // Process
    uint32_t output[2048];
    finder.process(output, 0, 2, 0, 5);

    // b[0], b[1] should map to a[0]
    EXPECT_EQ(output[0], 0u);
    EXPECT_EQ(output[1], 0u);

    // b[2], b[3] should be UINT32_MAX because a[1] is invalid
    EXPECT_EQ(output[2], UINT32_MAX);
    EXPECT_EQ(output[3], UINT32_MAX);

    // b[4], b[5] should map to a[2]
    EXPECT_EQ(output[4], 2u);
    EXPECT_EQ(output[5], 2u);
}

// Test 5: Path with only one state throws
TEST_F(FtreeAncestorFinderTest, SingleStateThrows) {
    std::vector<const ffx::State*> state_path = {vec_a->state};
    EXPECT_THROW(ffx::FtreeAncestorFinder(state_path.data(), state_path.size()), std::runtime_error);
}

// Test 6: Path with duplicate states throws
TEST_F(FtreeAncestorFinderTest, DuplicateStatesThrows) {
    // Same state repeated
    std::vector<const ffx::State*> state_path = {vec_a->state, vec_a->state};
    EXPECT_THROW(ffx::FtreeAncestorFinder(state_path.data(), state_path.size()), std::runtime_error);
}

// Test 7: Partial range processing
TEST_F(FtreeAncestorFinderTest, PartialRangeProcessing) {
    // Set up state for a: indices 0, 1, 2 valid
    SET_START_POS(*vec_a->state, 0);
    SET_END_POS(*vec_a->state, 2);

    // Set up state for b: indices 0-5 valid
    SET_START_POS(*vec_b->state, 0);
    SET_END_POS(*vec_b->state, 5);

    // Set up RLE: a[0] -> b[0,1], a[1] -> b[2,3], a[2] -> b[4,5]
    vec_b->state->offset[0] = 0;
    vec_b->state->offset[1] = 2;
    vec_b->state->offset[2] = 4;
    vec_b->state->offset[3] = 6;

    // Create finder
    std::vector<const ffx::State*> state_path = {vec_a->state, vec_b->state};
    ffx::FtreeAncestorFinder finder(state_path.data(), state_path.size());

    // Process only partial range: ancestor [1, 2], descendant [2, 5]
    uint32_t output[2048];
    for (int i = 0; i < 2048; i++) {
        output[i] = UINT32_MAX;
    }
    finder.process(output, 1, 2, 2, 5);

    // b[0], b[1] should still be UINT32_MAX (outside descendant range)
    EXPECT_EQ(output[0], UINT32_MAX);
    EXPECT_EQ(output[1], UINT32_MAX);

    // b[2], b[3] should map to a[1]
    EXPECT_EQ(output[2], 1u);
    EXPECT_EQ(output[3], 1u);

    // b[4], b[5] should map to a[2]
    EXPECT_EQ(output[4], 2u);
    EXPECT_EQ(output[5], 2u);
}

// ============================================================================
// FtreeMultiAncestorFinder Tests
// ============================================================================

class FtreeMultiAncestorFinderTest : public ::testing::Test {
protected:
    void SetUp() override {
        vec_a = std::make_unique<ffx::Vector<uint64_t>>(false);
        vec_b = std::make_unique<ffx::Vector<uint64_t>>(false);
        vec_c = std::make_unique<ffx::Vector<uint64_t>>(false);
        vec_d = std::make_unique<ffx::Vector<uint64_t>>(false);

        SET_ALL_BITS(vec_a->state->selector);
        SET_ALL_BITS(vec_b->state->selector);
        SET_ALL_BITS(vec_c->state->selector);
        SET_ALL_BITS(vec_d->state->selector);
    }

    std::unique_ptr<ffx::Vector<uint64_t>> vec_a, vec_b, vec_c, vec_d;
};

// Test: Two states - direct parent child
TEST_F(FtreeMultiAncestorFinderTest, TwoStates) {
    // Set up RLE: a[0] -> b[0,1], a[1] -> b[2,3]
    SET_START_POS(*vec_a->state, 0);
    SET_END_POS(*vec_a->state, 1);
    SET_START_POS(*vec_b->state, 0);
    SET_END_POS(*vec_b->state, 3);

    vec_b->state->offset[0] = 0;
    vec_b->state->offset[1] = 2;
    vec_b->state->offset[2] = 4;

    // Create multi-finder with state path [a, b]
    std::vector<const ffx::State*> state_path = {vec_a->state, vec_b->state};
    ffx::FtreeMultiAncestorFinder multi_finder(std::move(state_path));

    EXPECT_EQ(multi_finder.path_length(), 2);
    EXPECT_EQ(multi_finder.num_ancestor_levels(), 1);

    // Allocate output buffers (one per ancestor level)
    uint32_t output_0[2048];
    uint32_t* output_buffers[1] = {output_0};

    multi_finder.process(output_buffers, 0, 3);

    // b[0,1] -> a[0], b[2,3] -> a[1]
    EXPECT_EQ(output_0[0], 0u);
    EXPECT_EQ(output_0[1], 0u);
    EXPECT_EQ(output_0[2], 1u);
    EXPECT_EQ(output_0[3], 1u);
}

// Test: Three states (a -> b -> c)
TEST_F(FtreeMultiAncestorFinderTest, ThreeStates) {
    // Set up state
    SET_START_POS(*vec_a->state, 0);
    SET_END_POS(*vec_a->state, 1);
    SET_START_POS(*vec_b->state, 0);
    SET_END_POS(*vec_b->state, 3);
    SET_START_POS(*vec_c->state, 0);
    SET_END_POS(*vec_c->state, 7);

    // RLE: a[0] -> b[0,1], a[1] -> b[2,3]
    vec_b->state->offset[0] = 0;
    vec_b->state->offset[1] = 2;
    vec_b->state->offset[2] = 4;

    // RLE: b[0] -> c[0,1], b[1] -> c[2,3], b[2] -> c[4,5], b[3] -> c[6,7]
    vec_c->state->offset[0] = 0;
    vec_c->state->offset[1] = 2;
    vec_c->state->offset[2] = 4;
    vec_c->state->offset[3] = 6;
    vec_c->state->offset[4] = 8;

    // Create multi-finder with state path [a, b, c]
    std::vector<const ffx::State*> state_path = {vec_a->state, vec_b->state, vec_c->state};
    ffx::FtreeMultiAncestorFinder multi_finder(std::move(state_path));

    EXPECT_EQ(multi_finder.path_length(), 3);
    EXPECT_EQ(multi_finder.num_ancestor_levels(), 2);

    // Allocate output buffers
    uint32_t output_0[2048];// maps c -> a
    uint32_t output_1[2048];// maps c -> b
    uint32_t* output_buffers[2] = {output_0, output_1};

    multi_finder.process(output_buffers, 0, 7);

    // c[0-3] -> a[0], c[4-7] -> a[1]
    EXPECT_EQ(output_0[0], 0u);
    EXPECT_EQ(output_0[1], 0u);
    EXPECT_EQ(output_0[2], 0u);
    EXPECT_EQ(output_0[3], 0u);
    EXPECT_EQ(output_0[4], 1u);
    EXPECT_EQ(output_0[5], 1u);
    EXPECT_EQ(output_0[6], 1u);
    EXPECT_EQ(output_0[7], 1u);

    // c[0,1] -> b[0], c[2,3] -> b[1], c[4,5] -> b[2], c[6,7] -> b[3]
    EXPECT_EQ(output_1[0], 0u);
    EXPECT_EQ(output_1[1], 0u);
    EXPECT_EQ(output_1[2], 1u);
    EXPECT_EQ(output_1[3], 1u);
    EXPECT_EQ(output_1[4], 2u);
    EXPECT_EQ(output_1[5], 2u);
    EXPECT_EQ(output_1[6], 3u);
    EXPECT_EQ(output_1[7], 3u);
}

// Test: Four states (a -> b -> c -> d)
TEST_F(FtreeMultiAncestorFinderTest, FourStates) {
    // Set up state
    SET_START_POS(*vec_a->state, 0);
    SET_END_POS(*vec_a->state, 0);
    SET_START_POS(*vec_b->state, 0);
    SET_END_POS(*vec_b->state, 1);
    SET_START_POS(*vec_c->state, 0);
    SET_END_POS(*vec_c->state, 3);
    SET_START_POS(*vec_d->state, 0);
    SET_END_POS(*vec_d->state, 7);

    // RLE: a[0] -> b[0,1]
    vec_b->state->offset[0] = 0;
    vec_b->state->offset[1] = 2;

    // RLE: b[0] -> c[0,1], b[1] -> c[2,3]
    vec_c->state->offset[0] = 0;
    vec_c->state->offset[1] = 2;
    vec_c->state->offset[2] = 4;

    // RLE: c[0] -> d[0,1], c[1] -> d[2,3], c[2] -> d[4,5], c[3] -> d[6,7]
    vec_d->state->offset[0] = 0;
    vec_d->state->offset[1] = 2;
    vec_d->state->offset[2] = 4;
    vec_d->state->offset[3] = 6;
    vec_d->state->offset[4] = 8;

    // Create multi-finder with state path [a, b, c, d]
    std::vector<const ffx::State*> state_path = {vec_a->state, vec_b->state, vec_c->state, vec_d->state};
    ffx::FtreeMultiAncestorFinder multi_finder(std::move(state_path));

    EXPECT_EQ(multi_finder.path_length(), 4);
    EXPECT_EQ(multi_finder.num_ancestor_levels(), 3);

    // Allocate output buffers
    uint32_t output_0[2048];// maps d -> a
    uint32_t output_1[2048];// maps d -> b
    uint32_t output_2[2048];// maps d -> c
    uint32_t* output_buffers[3] = {output_0, output_1, output_2};

    multi_finder.process(output_buffers, 0, 7);

    // All d values map to a[0]
    for (int i = 0; i <= 7; i++) {
        EXPECT_EQ(output_0[i], 0u) << "d[" << i << "] should map to a[0]";
    }

    // d[0-3] -> b[0], d[4-7] -> b[1]
    EXPECT_EQ(output_1[0], 0u);
    EXPECT_EQ(output_1[1], 0u);
    EXPECT_EQ(output_1[2], 0u);
    EXPECT_EQ(output_1[3], 0u);
    EXPECT_EQ(output_1[4], 1u);
    EXPECT_EQ(output_1[5], 1u);
    EXPECT_EQ(output_1[6], 1u);
    EXPECT_EQ(output_1[7], 1u);

    // d[0,1] -> c[0], d[2,3] -> c[1], d[4,5] -> c[2], d[6,7] -> c[3]
    EXPECT_EQ(output_2[0], 0u);
    EXPECT_EQ(output_2[1], 0u);
    EXPECT_EQ(output_2[2], 1u);
    EXPECT_EQ(output_2[3], 1u);
    EXPECT_EQ(output_2[4], 2u);
    EXPECT_EQ(output_2[5], 2u);
    EXPECT_EQ(output_2[6], 3u);
    EXPECT_EQ(output_2[7], 3u);
}

// Test: Path with only one state throws
TEST_F(FtreeMultiAncestorFinderTest, SingleStateThrows) {
    std::vector<const ffx::State*> state_path = {vec_a->state};
    EXPECT_THROW(ffx::FtreeMultiAncestorFinder(std::move(state_path)), std::runtime_error);
}

// Test: Path with duplicate states throws
TEST_F(FtreeMultiAncestorFinderTest, DuplicateStatesThrows) {
    std::vector<const ffx::State*> state_path = {vec_a->state, vec_a->state, vec_b->state};
    EXPECT_THROW(ffx::FtreeMultiAncestorFinder(std::move(state_path)), std::runtime_error);
}

// Test: Verify single traversal gives same result as multiple FtreeAncestorFinder
TEST_F(FtreeMultiAncestorFinderTest, ConsistentWithSingleFinder) {
    // Set up state
    SET_START_POS(*vec_a->state, 0);
    SET_END_POS(*vec_a->state, 1);
    SET_START_POS(*vec_b->state, 0);
    SET_END_POS(*vec_b->state, 3);
    SET_START_POS(*vec_c->state, 0);
    SET_END_POS(*vec_c->state, 7);

    vec_b->state->offset[0] = 0;
    vec_b->state->offset[1] = 2;
    vec_b->state->offset[2] = 4;

    vec_c->state->offset[0] = 0;
    vec_c->state->offset[1] = 2;
    vec_c->state->offset[2] = 4;
    vec_c->state->offset[3] = 6;
    vec_c->state->offset[4] = 8;

    // Use single FtreeAncestorFinder instances
    std::vector<const ffx::State*> path_a_c = {vec_a->state, vec_b->state, vec_c->state};
    std::vector<const ffx::State*> path_b_c = {vec_b->state, vec_c->state};
    ffx::FtreeAncestorFinder finder_a_c(path_a_c.data(), path_a_c.size());
    ffx::FtreeAncestorFinder finder_b_c(path_b_c.data(), path_b_c.size());

    uint32_t single_output_a[2048];
    uint32_t single_output_b[2048];
    finder_a_c.process(single_output_a, 0, 1, 0, 7);
    finder_b_c.process(single_output_b, 0, 3, 0, 7);

    // Use FtreeMultiAncestorFinder
    std::vector<const ffx::State*> state_path = {vec_a->state, vec_b->state, vec_c->state};
    ffx::FtreeMultiAncestorFinder multi_finder(std::move(state_path));

    uint32_t multi_output_a[2048];
    uint32_t multi_output_b[2048];
    uint32_t* output_buffers[2] = {multi_output_a, multi_output_b};
    multi_finder.process(output_buffers, 0, 7);

    // Compare results
    for (int i = 0; i <= 7; i++) {
        EXPECT_EQ(multi_output_a[i], single_output_a[i]) << "Mismatch at c[" << i << "] -> a mapping";
        EXPECT_EQ(multi_output_b[i], single_output_b[i]) << "Mismatch at c[" << i << "] -> b mapping";
    }
}

}// namespace
