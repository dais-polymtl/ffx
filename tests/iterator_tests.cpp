#include "../src/operator/include/factorized_ftree/ftree_iterator.hpp"
#include "../src/operator/include/schema/schema.hpp"
#include "../src/operator/include/vector/state.hpp"
#include "../src/operator/include/vector/vector.hpp"
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

// Test fixture for iterator tests
class IteratorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create vectors for the tree nodes
        // Root: A with values [0, 1, 2]
        a_vector = std::make_unique<ffx::Vector<uint64_t>>();
        a_vector->values[0] = 0;
        a_vector->values[1] = 1;
        a_vector->values[2] = 2;
        SET_ALL_BITS(a_vector->state->selector);
        SET_START_POS(*a_vector->state, 0);
        SET_END_POS(*a_vector->state, 2);

        // Child B: values [10, 11, 12, 13, 14, 15]
        // RLE: for parent A pos 0: 0-1, pos 1: 2-3, pos 2: 4-5
        b_vector = std::make_unique<ffx::Vector<uint64_t>>();
        b_vector->values[0] = 10;
        b_vector->values[1] = 11;
        b_vector->values[2] = 12;
        b_vector->values[3] = 13;
        b_vector->values[4] = 14;
        b_vector->values[5] = 15;
        SET_ALL_BITS(b_vector->state->selector);
        SET_START_POS(*b_vector->state, 0);
        SET_END_POS(*b_vector->state, 5);
        b_vector->state->offset[0] = 0;
        b_vector->state->offset[1] = 2;
        b_vector->state->offset[2] = 4;
        b_vector->state->offset[3] = 6;

        // Child C: values [20, 21, 22, 23, 24, 25]
        // RLE: for parent A pos 0: 0-1, pos 1: 2-3, pos 2: 4-5
        c_vector = std::make_unique<ffx::Vector<uint64_t>>();
        c_vector->values[0] = 20;
        c_vector->values[1] = 21;
        c_vector->values[2] = 22;
        c_vector->values[3] = 23;
        c_vector->values[4] = 24;
        c_vector->values[5] = 25;
        SET_ALL_BITS(c_vector->state->selector);
        SET_START_POS(*c_vector->state, 0);
        SET_END_POS(*c_vector->state, 5);
        c_vector->state->offset[0] = 0;
        c_vector->state->offset[1] = 2;
        c_vector->state->offset[2] = 4;
        c_vector->state->offset[3] = 6;

        // Create tree: A -> B, A -> C
        root = std::make_shared<ffx::FactorizedTreeElement>("A", a_vector.get());
        b_node = root->add_leaf("A", "B", a_vector.get(), b_vector.get());
        c_node = root->add_leaf("A", "C", a_vector.get(), c_vector.get());

        // Column ordering
        column_ordering = {"A", "B", "C"};

        // Create schema
        schema.root = root;
        schema.column_ordering = &column_ordering;
    }

    void TearDown() override {
        // Cleanup if needed
    }

    std::unique_ptr<ffx::Vector<uint64_t>> a_vector, b_vector, c_vector;
    std::shared_ptr<ffx::FactorizedTreeElement> root;
    ffx::FactorizedTreeElement *b_node, *c_node;
    std::vector<std::string> column_ordering;
    ffx::Schema schema;
};

// Recursive helper to compute expected tuple count for subtree rooted at node for a given parent position
static uint64_t count_tuples_for_node_at_pos(ffx::FactorizedTreeElement* node, int32_t pos) {
    // If node is leaf, one tuple for a given position
    if (node->_children.empty()) return 1;

    uint64_t prod = 1;
    for (const auto& child_ptr: node->_children) {
        auto* child = child_ptr.get();
        const ffx::State* child_state = child->_value->state;
        const uint16_t* child_rle = child_state->offset;

        // Get the child's RLE range for parent position 'pos'
        const int32_t cstart = static_cast<int32_t>(child_rle[pos]);
        const int32_t cend = static_cast<int32_t>(child_rle[pos + 1]) - 1;
        const int32_t cs = std::max(static_cast<int32_t>(GET_START_POS(*child_state)), cstart);
        const int32_t ce = std::min(static_cast<int32_t>(GET_END_POS(*child_state)), cend);
        uint64_t child_sum = 0;
        if (cs <= ce) {
            for (int32_t cpos = cs; cpos <= ce; ++cpos) {
                if (!TEST_BIT(child_state->selector, cpos)) continue;
                child_sum += count_tuples_for_node_at_pos(child, cpos);
            }
        }
        prod *= child_sum;
    }
    return prod;
}

static uint64_t count_tuples_for_tree(ffx::FactorizedTreeElement* root) {
    const ffx::State* root_state = root->_value->state;
    uint64_t total = 0;
    for (int32_t rpos = GET_START_POS(*root_state); rpos <= GET_END_POS(*root_state); ++rpos) {
        if (!TEST_BIT(root_state->selector, rpos)) continue;
        // For each root position, product across children ranges
        uint64_t prod = 1;
        for (const auto& child_ptr: root->_children) {
            auto* child = child_ptr.get();
            const ffx::State* child_state = child->_value->state;
            const uint16_t* child_rle = child_state->offset;
            const int32_t cstart = static_cast<int32_t>(child_rle[rpos]);
            const int32_t cend = static_cast<int32_t>(child_rle[rpos + 1]) - 1;
            const int32_t cs = std::max(static_cast<int32_t>(GET_START_POS(*child_state)), cstart);
            const int32_t ce = std::min(static_cast<int32_t>(GET_END_POS(*child_state)), cend);
            uint64_t child_sum = 0;
            if (cs <= ce) {
                for (int32_t cpos = cs; cpos <= ce; ++cpos) {
                    if (!TEST_BIT(child_state->selector, cpos)) continue;
                    child_sum += count_tuples_for_node_at_pos(child, cpos);
                }
            }
            prod *= child_sum;
        }
        total += prod;
    }
    return total;
}

// Test basic iterator functionality
TEST_F(IteratorTest, BasicIteration) {
    ffx::FTreeIterator iterator;
    iterator.init(&schema);
    iterator.initialize_iterators();

    std::vector<std::vector<uint64_t>> tuples;
    uint64_t buffer[3];

    // Use iterator.next() to write current tuple each call; next() returns false after writing the last
    while (true) {
        bool ok = iterator.next(buffer);
        tuples.push_back({buffer[0], buffer[1], buffer[2]});
        if (!ok) break;
    }

    // Expected tuples based on the setup
    // A has 3 positions: 0,1,2
    // For each A, B and C have 2 positions each
    // Total tuples: 3 * 2 * 2 = 12
    ASSERT_EQ(tuples.size(), 12u);

    // Check some specific tuples
    // First tuple: A=0, B=10, C=20
    ASSERT_EQ(tuples[0][0], 0u);
    ASSERT_EQ(tuples[0][1], 10u);
    ASSERT_EQ(tuples[0][2], 20u);

    // Second tuple: A=0, B=10, C=21
    ASSERT_EQ(tuples[1][0], 0u);
    ASSERT_EQ(tuples[1][1], 10u);
    ASSERT_EQ(tuples[1][2], 21u);

    // And so on...
    // Last tuple: A=2, B=15, C=25
    ASSERT_EQ(tuples.back()[0], 2u);
    ASSERT_EQ(tuples.back()[1], 15u);
    ASSERT_EQ(tuples.back()[2], 25u);
}

// Test with some bits cleared in selectors
TEST_F(IteratorTest, FilteredIteration) {
    // Clear some bits in B selector
    CLEAR_BIT(b_vector->state->selector, 1);// B pos 1
    CLEAR_BIT(b_vector->state->selector, 4);// B pos 4

    ffx::FTreeIterator iterator;
    iterator.init(&schema);
    iterator.initialize_iterators();

    std::vector<std::vector<uint64_t>> tuples;
    uint64_t buffer[3];

    while (true) {
        bool ok = iterator.next(buffer);
        tuples.push_back({buffer[0], buffer[1], buffer[2]});
        if (!ok) break;
    }

    // Now, for each A, B has fewer positions
    // A=0: B positions 0 (valid), 1 (cleared) -> only 1 for B, C has 2 -> 2 tuples
    // A=1: B positions 2,3 (both valid) -> 2, C has 2 -> 4 tuples
    // A=2: B positions 4 (cleared),5 (valid) -> 1, C has 2 -> 2 tuples
    // Total: 2 + 4 + 2 = 8 tuples
    ASSERT_EQ(tuples.size(), 8u);

    // Check first tuple: A=0, B=10, C=20
    ASSERT_EQ(tuples[0][0], 0u);
    ASSERT_EQ(tuples[0][1], 10u);
    ASSERT_EQ(tuples[0][2], 20u);

    // Next: A=0, B=10, C=21
    ASSERT_EQ(tuples[1][0], 0u);
    ASSERT_EQ(tuples[1][1], 10u);
    ASSERT_EQ(tuples[1][2], 21u);

    // Then A=1, B=12, C=20 etc.
}

TEST_F(IteratorTest, NextMinTupleProcessesAllTuplesIncludingLast) {
    ffx::FTreeIterator iterator;
    iterator.init(&schema);
    iterator.initialize_iterators();

    uint64_t min_buffer[3] = {std::numeric_limits<uint64_t>::max(),
                              std::numeric_limits<uint64_t>::max(),
                              std::numeric_limits<uint64_t>::max()};

    uint64_t tuples_processed = 0;
    while (iterator.next_min_tuple(min_buffer)) {
        tuples_processed++;
    }

    // 3 root positions * 2 B positions * 2 C positions
    EXPECT_EQ(tuples_processed, 12u);
    EXPECT_EQ(min_buffer[0], 0u);
    EXPECT_EQ(min_buffer[1], 10u);
    EXPECT_EQ(min_buffer[2], 20u);
}

TEST_F(IteratorTest, NextMinTupleReturnsFalseAfterExhaustion) {
    ffx::FTreeIterator iterator;
    iterator.init(&schema);
    iterator.initialize_iterators();

    uint64_t min_buffer[3] = {std::numeric_limits<uint64_t>::max(),
                              std::numeric_limits<uint64_t>::max(),
                              std::numeric_limits<uint64_t>::max()};

    uint64_t tuples_processed = 0;
    while (iterator.next_min_tuple(min_buffer)) {
        tuples_processed++;
    }

    EXPECT_EQ(tuples_processed, 12u);
    EXPECT_FALSE(iterator.next_min_tuple(min_buffer));
}

TEST_F(IteratorTest, EmptyVectorIteration) {
    // Create a root with empty selector
    auto root_vec = std::make_unique<ffx::Vector<uint64_t>>();
    SET_ALL_BITS(root_vec->state->selector);
    // Now clear all bits to make it empty
    CLEAR_ALL_BITS(root_vec->state->selector);
    SET_START_POS(*root_vec->state, 1);
    SET_END_POS(*root_vec->state, 0);
    auto root_local = std::make_shared<ffx::FactorizedTreeElement>("root", root_vec.get());
    std::vector<std::string> col_order = {"root"};
    ffx::Schema s;
    s.root = root_local;
    s.column_ordering = &col_order;

    ffx::FTreeIterator itr;
    itr.init(&s);
    itr.initialize_iterators();
    // Iterator should be invalid and no tuples should be produced
    ASSERT_FALSE(itr.is_valid());
}

TEST_F(IteratorTest, DeepComplexTreeIteration) {
    // Build deeper tree: A->B->C->D->E->F
    const int depth = 6;
    const int root_positions = 2;
    std::vector<std::unique_ptr<ffx::Vector<uint64_t>>> vecs;
    std::vector<ffx::FactorizedTreeElement*> nodes;
    vecs.reserve(depth);
    nodes.reserve(depth);

    // Create vectors
    for (int i = 0; i < depth; ++i) {
        vecs.push_back(std::make_unique<ffx::Vector<uint64_t>>());
        auto st = vecs.back()->state;
        SET_ALL_BITS(st->selector);
        SET_START_POS(*st, 0);
        SET_END_POS(*st, 3);// child positions 0..3 to allow rle ranges
        // RLE size: for parent positions (we'll set rle later), initialize default
    }

    // Fill values and simple RLE such that each parent pos maps to two child positions
    // root has root_positions positions
    auto root_vec2 = std::make_unique<ffx::Vector<uint64_t>>();
    for (int i = 0; i < root_positions; ++i)
        root_vec2->values[i] = i;
    SET_ALL_BITS(root_vec2->state->selector);
    SET_START_POS(*root_vec2->state, 0);
    SET_END_POS(*root_vec2->state, root_positions - 1);

    auto root_node2 = std::make_shared<ffx::FactorizedTreeElement>("A", root_vec2.get());
    ffx::FactorizedTreeElement* parent = root_node2.get();
    // chain depth-1 children
    for (int i = 0; i < depth; ++i) {
        parent->add_leaf(parent->_attribute, std::string(1, 'a' + i + 1), parent->_value, vecs[i].get());
        parent = parent->_children.back().get();
    }

    // Setup RLE: each parent position p maps to child positions 0 and 1
    for (int i = 0; i < depth; ++i) {
        auto st = vecs[i]->state;
        st->offset[0] = 0;
        st->offset[1] = 2;
        st->offset[2] = 4;// for parent positions 0 and 1
        // set some values
        for (int v = 0; v < 4; ++v)
            vecs[i]->values[v] = 100 + i * 10 + v;
    }

    std::vector<std::string> col_order2;
    col_order2.reserve(depth + 1);
    col_order2.push_back("A");
    for (int i = 0; i < depth; ++i)
        col_order2.push_back(std::string(1, 'a' + i + 1));
    ffx::Schema s2;
    s2.root = root_node2;
    s2.column_ordering = &col_order2;

    // compute expected count via helper
    uint64_t expected = count_tuples_for_tree(root_node2.get());

    ffx::FTreeIterator itr2;
    itr2.init(&s2);
    itr2.initialize_iterators();
    uint64_t buf[depth + 1];
    uint64_t num = 0;
    while (true) {
        bool ok = itr2.next(buf);
        num++;
        if (!ok) break;
    }
    EXPECT_EQ(num, expected);
}

TEST_F(IteratorTest, RandomMaskingIteration) {
    // create a modest tree A->B->C with rle sizes 0..n
    auto a = std::make_unique<ffx::Vector<uint64_t>>();
    auto b = std::make_unique<ffx::Vector<uint64_t>>();
    auto c = std::make_unique<ffx::Vector<uint64_t>>();
    const int A_size = 4;
    const int B_size = 8;
    const int C_size = 16;
    for (int i = 0; i < A_size; ++i)
        a->values[i] = i;
    SET_ALL_BITS(a->state->selector);
    SET_START_POS(*a->state, 0);
    SET_END_POS(*a->state, A_size - 1);
    for (int i = 0; i < B_size; ++i)
        b->values[i] = 100 + i;
    SET_ALL_BITS(b->state->selector);
    SET_START_POS(*b->state, 0);
    SET_END_POS(*b->state, B_size - 1);
    for (int i = 0; i < C_size; ++i)
        c->values[i] = 200 + i;
    SET_ALL_BITS(c->state->selector);
    SET_START_POS(*c->state, 0);
    SET_END_POS(*c->state, C_size - 1);

    // RLE: each A position maps to two B positions, and each B to two C positions
    for (int i = 0; i < A_size; ++i)
        b->state->offset[i] = i * (B_size / A_size);
    b->state->offset[A_size] = B_size;
    for (int i = 0; i <= B_size; ++i)
        c->state->offset[i] = i * (C_size / B_size);

    auto root_local = std::make_shared<ffx::FactorizedTreeElement>("A", a.get());
    auto b_node_local = root_local->add_leaf("A", "B", a.get(), b.get());
    b_node_local->add_leaf("B", "C", b.get(), c.get());
    std::vector<std::string> ord = {"A", "B", "C"};
    ffx::Schema s3;
    s3.root = root_local;
    s3.column_ordering = &ord;

    // Randomly clear some bits in B and C selectors
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> distb(0, B_size - 1);
    std::uniform_int_distribution<int> distc(0, C_size - 1);
    for (int i = 0; i < 5; ++i)
        CLEAR_BIT(b->state->selector, distb(rng));
    for (int i = 0; i < 7; ++i)
        CLEAR_BIT(c->state->selector, distc(rng));

    // Compute expected tuples via explicit enumeration (match iterator's traversal semantics)
    std::vector<std::vector<uint64_t>> expected_tuples;
    const auto* a_state_local = root_local->_value->state;
    const auto* b_state_local = b->state;
    const auto* c_state_local = c->state;
    for (int32_t aidx = GET_START_POS(*a_state_local); aidx <= GET_END_POS(*a_state_local); ++aidx) {
        if (!TEST_BIT(a_state_local->selector, aidx)) continue;
        const int32_t b_start = static_cast<int32_t>(b_state_local->offset[aidx]);
        const int32_t b_end = static_cast<int32_t>(b_state_local->offset[aidx + 1]) - 1;
        for (int32_t bidx = b_start; bidx <= b_end; ++bidx) {
            if (!TEST_BIT(b_state_local->selector, bidx)) continue;
            const int32_t c_start = static_cast<int32_t>(c_state_local->offset[bidx]);
            const int32_t c_end = static_cast<int32_t>(c_state_local->offset[bidx + 1]) - 1;
            for (int32_t cidx = c_start; cidx <= c_end; ++cidx) {
                if (!TEST_BIT(c_state_local->selector, cidx)) continue;
                expected_tuples.push_back(
                        {(uint64_t) a->values[aidx], (uint64_t) b->values[bidx], (uint64_t) c->values[cidx]});
            }
        }
    }

    ffx::FTreeIterator itr3;
    itr3.init(&s3);
    itr3.initialize_iterators();
    uint64_t buf3[3];
    std::vector<std::vector<uint64_t>> actual_tuples;
    while (true) {
        bool ok = itr3.next(buf3);
        actual_tuples.push_back({buf3[0], buf3[1], buf3[2]});
        if (!ok) break;
    }

    // Compare sizes and tuples; if mismatch, show both for debugging
    EXPECT_EQ(actual_tuples.size(), expected_tuples.size());
    if (actual_tuples.size() != expected_tuples.size()) {
        std::cerr << "Expected tuples:\n";
        for (auto& t: expected_tuples)
            std::cerr << "  (" << t[0] << "," << t[1] << "," << t[2] << ")\n";
        std::cerr << "Actual tuples:\n";
        for (auto& t: actual_tuples)
            std::cerr << "  (" << t[0] << "," << t[1] << "," << t[2] << ")\n";
    }
    EXPECT_EQ(actual_tuples, expected_tuples);
}

// NOTE: Buffered iterator parity test removed — tests use canonical FTreeIterator

//------------------------------------------------------------------------------
// Complex Iterator Tests
//------------------------------------------------------------------------------

// Test wide tree with sibling children
// Note: Using smaller number of children to keep cartesian product manageable
TEST_F(IteratorTest, WideTreeManySiblings) {
    // Create a root with 4 leaf children (wide tree)
    const int NUM_CHILDREN = 4;// Reduced from 8 to keep test fast
    const int ROOT_SIZE = 2;
    const int CHILD_SIZE = 4;

    std::vector<std::unique_ptr<ffx::Vector<uint64_t>>> child_vectors;
    child_vectors.reserve(NUM_CHILDREN);

    auto root_vec = std::make_unique<ffx::Vector<uint64_t>>();
    for (int i = 0; i < ROOT_SIZE; ++i)
        root_vec->values[i] = i;
    SET_ALL_BITS(root_vec->state->selector);
    SET_START_POS(*root_vec->state, 0);
    SET_END_POS(*root_vec->state, ROOT_SIZE - 1);

    auto root_node = std::make_shared<ffx::FactorizedTreeElement>("R", root_vec.get());

    std::vector<std::string> col_order = {"R"};

    for (int c = 0; c < NUM_CHILDREN; ++c) {
        auto cv = std::make_unique<ffx::Vector<uint64_t>>();
        for (int i = 0; i < CHILD_SIZE; ++i)
            cv->values[i] = 100 * (c + 1) + i;
        SET_ALL_BITS(cv->state->selector);
        SET_START_POS(*cv->state, 0);
        SET_END_POS(*cv->state, CHILD_SIZE - 1);

        // RLE: each root position maps to 2 child positions
        cv->state->offset[0] = 0;
        cv->state->offset[1] = 2;
        cv->state->offset[2] = 4;

        std::string name = "C" + std::to_string(c);
        root_node->add_leaf("R", name, root_vec.get(), cv.get());
        col_order.push_back(name);
        child_vectors.push_back(std::move(cv));
    }

    ffx::Schema s;
    s.root = root_node;
    s.column_ordering = &col_order;

    ffx::FTreeIterator itr;
    itr.init(&s);
    itr.initialize_iterators();

    // Expected: ROOT_SIZE * (2^NUM_CHILDREN) = 2 * 16 = 32 tuples
    // Each root pos has 2 positions per child, so 2^4 = 16 combinations per root
    uint64_t expected = ROOT_SIZE;
    for (int c = 0; c < NUM_CHILDREN; ++c)
        expected *= 2;

    uint64_t count = 0;
    uint64_t buffer[NUM_CHILDREN + 1];
    while (true) {
        bool ok = itr.next(buffer);
        count++;
        if (!ok) break;
    }
    EXPECT_EQ(count, expected);
}

// Test asymmetric RLE distributions
TEST_F(IteratorTest, AsymmetricRLEDistribution) {
    // Root has 4 positions, child has varying number of elements per parent
    auto root_vec = std::make_unique<ffx::Vector<uint64_t>>();
    auto child_vec = std::make_unique<ffx::Vector<uint64_t>>();

    const int ROOT_SIZE = 4;
    // Child distribution: pos0->1, pos1->3, pos2->0, pos3->5 = 9 total child positions
    const int CHILD_SIZE = 9;

    for (int i = 0; i < ROOT_SIZE; ++i)
        root_vec->values[i] = i;
    SET_ALL_BITS(root_vec->state->selector);
    SET_START_POS(*root_vec->state, 0);
    SET_END_POS(*root_vec->state, ROOT_SIZE - 1);

    for (int i = 0; i < CHILD_SIZE; ++i)
        child_vec->values[i] = 100 + i;
    SET_ALL_BITS(child_vec->state->selector);
    SET_START_POS(*child_vec->state, 0);
    SET_END_POS(*child_vec->state, CHILD_SIZE - 1);

    // Asymmetric RLE: pos0->1 element, pos1->3 elements, pos2->0 elements, pos3->5 elements
    child_vec->state->offset[0] = 0;// pos0: [0, 1)
    child_vec->state->offset[1] = 1;// pos1: [1, 4)
    child_vec->state->offset[2] = 4;// pos2: [4, 4) empty
    child_vec->state->offset[3] = 4;// pos3: [4, 9)
    child_vec->state->offset[4] = 9;

    auto root_node = std::make_shared<ffx::FactorizedTreeElement>("R", root_vec.get());
    root_node->add_leaf("R", "C", root_vec.get(), child_vec.get());
    std::vector<std::string> col_order = {"R", "C"};
    ffx::Schema s;
    s.root = root_node;
    s.column_ordering = &col_order;

    ffx::FTreeIterator itr;
    itr.init(&s);
    itr.initialize_iterators();

    // Expected: 1 + 3 + 0 + 5 = 9 tuples (pos2 produces 0 tuples)
    uint64_t expected = 9;

    uint64_t count = 0;
    uint64_t buffer[2];
    while (itr.is_valid()) {
        bool ok = itr.next(buffer);
        count++;
        if (!ok) break;
    }
    EXPECT_EQ(count, expected);
}

// Test with sparse bitmask patterns across multi-block boundaries
TEST_F(IteratorTest, SparseBitmaskMultiBlock) {
    // Create vectors that span multiple 64-bit blocks with sparse bit patterns
    auto root_vec = std::make_unique<ffx::Vector<uint64_t>>();
    auto child_vec = std::make_unique<ffx::Vector<uint64_t>>();

    const int ROOT_SIZE = 128;// Spans 2 64-bit blocks
    const int CHILD_SIZE = 128;

    for (int i = 0; i < ROOT_SIZE; ++i)
        root_vec->values[i] = i;
    CLEAR_ALL_BITS(root_vec->state->selector);
    // Set only every 17th bit (prime number for sparse pattern)
    for (int i = 0; i < ROOT_SIZE; i += 17) {
        SET_BIT(root_vec->state->selector, i);
    }
    SET_START_POS(*root_vec->state, 0);
    SET_END_POS(*root_vec->state, ROOT_SIZE - 1);

    for (int i = 0; i < CHILD_SIZE; ++i)
        child_vec->values[i] = 1000 + i;
    CLEAR_ALL_BITS(child_vec->state->selector);
    // Set only every 13th bit
    for (int i = 0; i < CHILD_SIZE; i += 13) {
        SET_BIT(child_vec->state->selector, i);
    }
    SET_START_POS(*child_vec->state, 0);
    SET_END_POS(*child_vec->state, CHILD_SIZE - 1);

    // RLE: each root position maps to 1 child position (1-to-1)
    for (int i = 0; i <= ROOT_SIZE; ++i) {
        child_vec->state->offset[i] = i;
    }

    auto root_node = std::make_shared<ffx::FactorizedTreeElement>("R", root_vec.get());
    root_node->add_leaf("R", "C", root_vec.get(), child_vec.get());
    std::vector<std::string> col_order = {"R", "C"};
    ffx::Schema s;
    s.root = root_node;
    s.column_ordering = &col_order;

    ffx::FTreeIterator itr;
    itr.init(&s);
    itr.initialize_iterators();

    // Count expected tuples: positions where both root AND child bits are set
    // Root positions: 0, 17, 34, 51, 68, 85, 102, 119
    // Child positions: 0, 13, 26, 39, 52, 65, 78, 91, 104, 117
    // With 1-to-1 mapping, only position 0 has both set (0 % 17 == 0 AND 0 % 13 == 0)
    uint64_t expected = 0;
    for (int i = 0; i < ROOT_SIZE; ++i) {
        if (i % 17 == 0 && i % 13 == 0) expected++;
    }

    uint64_t count = 0;
    uint64_t buffer[2];
    while (itr.is_valid()) {
        bool ok = itr.next(buffer);
        count++;
        if (!ok) break;
    }
    EXPECT_EQ(count, expected);
}

// Test deep chain with alternating bit patterns
TEST_F(IteratorTest, DeepChainAlternatingPatterns) {
    // Create a chain: A -> B -> C -> D with alternating checkerboard patterns
    const int DEPTH = 4;
    const int SIZE = 16;

    std::vector<std::unique_ptr<ffx::Vector<uint64_t>>> vecs;
    vecs.reserve(DEPTH);

    for (int d = 0; d < DEPTH; ++d) {
        auto v = std::make_unique<ffx::Vector<uint64_t>>();
        for (int i = 0; i < SIZE; ++i)
            v->values[i] = d * 100 + i;
        CLEAR_ALL_BITS(v->state->selector);
        // Alternating pattern: even depth gets even positions, odd depth gets odd positions
        for (int i = (d % 2); i < SIZE; i += 2) {
            SET_BIT(v->state->selector, i);
        }
        SET_START_POS(*v->state, 0);
        SET_END_POS(*v->state, SIZE - 1);

        // RLE: 2 children per parent position
        for (int p = 0; p <= SIZE / 2; ++p) {
            v->state->offset[p] = p * 2;
        }

        vecs.push_back(std::move(v));
    }

    // First vec is root, no RLE needed for root itself
    auto root_node = std::make_shared<ffx::FactorizedTreeElement>("A", vecs[0].get());
    ffx::FactorizedTreeElement* parent = root_node.get();

    std::vector<std::string> col_order = {"A"};
    for (int d = 1; d < DEPTH; ++d) {
        std::string name(1, 'A' + d);
        parent->add_leaf(parent->_attribute, name, parent->_value, vecs[d].get());
        parent = parent->_children.back().get();
        col_order.push_back(name);
    }

    ffx::Schema s;
    s.root = root_node;
    s.column_ordering = &col_order;

    ffx::FTreeIterator itr;
    itr.init(&s);
    itr.initialize_iterators();

    // Count tuples manually
    uint64_t manual_count = count_tuples_for_tree(root_node.get());

    uint64_t count = 0;
    uint64_t buffer[DEPTH];
    while (itr.is_valid()) {
        bool ok = itr.next(buffer);
        count++;
        if (!ok) break;
    }
    EXPECT_EQ(count, manual_count);
}

// Test single element vectors (edge case)
TEST_F(IteratorTest, SingleElementVectors) {
    auto root_vec = std::make_unique<ffx::Vector<uint64_t>>();
    auto child_vec = std::make_unique<ffx::Vector<uint64_t>>();

    root_vec->values[0] = 42;
    SET_ALL_BITS(root_vec->state->selector);
    SET_START_POS(*root_vec->state, 0);
    SET_END_POS(*root_vec->state, 0);

    child_vec->values[0] = 99;
    SET_ALL_BITS(child_vec->state->selector);
    SET_START_POS(*child_vec->state, 0);
    SET_END_POS(*child_vec->state, 0);
    child_vec->state->offset[0] = 0;
    child_vec->state->offset[1] = 1;

    auto root_node = std::make_shared<ffx::FactorizedTreeElement>("R", root_vec.get());
    root_node->add_leaf("R", "C", root_vec.get(), child_vec.get());
    std::vector<std::string> col_order = {"R", "C"};
    ffx::Schema s;
    s.root = root_node;
    s.column_ordering = &col_order;

    ffx::FTreeIterator itr;
    itr.init(&s);
    itr.initialize_iterators();

    EXPECT_TRUE(itr.is_valid());

    uint64_t buffer[2];
    bool ok = itr.next(buffer);
    EXPECT_EQ(buffer[0], 42u);
    EXPECT_EQ(buffer[1], 99u);
    EXPECT_FALSE(ok);// Should be the last tuple
}

// Test with maximum vector size stress test
TEST_F(IteratorTest, MaxVectorSizeStress) {
    // Use maximum vector size with sparse pattern to keep iteration count manageable
    auto root_vec = std::make_unique<ffx::Vector<uint64_t>>();
    auto child_vec = std::make_unique<ffx::Vector<uint64_t>>();

    const int ROOT_SIZE = 4;    // Small root to keep iteration tractable
    const int CHILD_SIZE = 2048;// use a fixed size larger than MAX_VECTOR_SIZE for tests

    for (int i = 0; i < ROOT_SIZE; ++i)
        root_vec->values[i] = i;
    SET_ALL_BITS(root_vec->state->selector);
    SET_START_POS(*root_vec->state, 0);
    SET_END_POS(*root_vec->state, ROOT_SIZE - 1);

    for (int i = 0; i < CHILD_SIZE; ++i)
        child_vec->values[i] = i;
    CLEAR_ALL_BITS(child_vec->state->selector);
    // Only set first 4 bits per root position range (16 total active)
    for (int r = 0; r < ROOT_SIZE; ++r) {
        int start = r * (CHILD_SIZE / ROOT_SIZE);
        for (int i = 0; i < 4; ++i) {
            SET_BIT(child_vec->state->selector, start + i);
        }
    }
    SET_START_POS(*child_vec->state, 0);
    SET_END_POS(*child_vec->state, CHILD_SIZE - 1);

    // RLE: partition child evenly across root
    int chunk = CHILD_SIZE / ROOT_SIZE;
    for (int r = 0; r < ROOT_SIZE; ++r) {
        child_vec->state->offset[r] = r * chunk;
    }
    child_vec->state->offset[ROOT_SIZE] = CHILD_SIZE;

    auto root_node = std::make_shared<ffx::FactorizedTreeElement>("R", root_vec.get());
    root_node->add_leaf("R", "C", root_vec.get(), child_vec.get());
    std::vector<std::string> col_order = {"R", "C"};
    ffx::Schema s;
    s.root = root_node;
    s.column_ordering = &col_order;

    ffx::FTreeIterator itr;
    itr.init(&s);
    itr.initialize_iterators();

    // Expected: ROOT_SIZE * 4 = 16 tuples
    uint64_t expected = ROOT_SIZE * 4;

    uint64_t count = 0;
    uint64_t buffer[2];
    while (itr.is_valid()) {
        bool ok = itr.next(buffer);
        count++;
        if (!ok) break;
    }
    EXPECT_EQ(count, expected);
}

// Test overlapping RLE ranges with complex masking
TEST_F(IteratorTest, ComplexRLEWithMasking) {
    // Create a tree where child ranges overlap at block boundaries
    auto root_vec = std::make_unique<ffx::Vector<uint64_t>>();
    auto child1_vec = std::make_unique<ffx::Vector<uint64_t>>();
    auto child2_vec = std::make_unique<ffx::Vector<uint64_t>>();

    const int ROOT_SIZE = 8;
    const int CHILD_SIZE = 32;

    for (int i = 0; i < ROOT_SIZE; ++i)
        root_vec->values[i] = i;
    SET_ALL_BITS(root_vec->state->selector);
    CLEAR_BIT(root_vec->state->selector, 3);// Clear middle position
    CLEAR_BIT(root_vec->state->selector, 6);
    SET_START_POS(*root_vec->state, 0);
    SET_END_POS(*root_vec->state, ROOT_SIZE - 1);

    for (int i = 0; i < CHILD_SIZE; ++i)
        child1_vec->values[i] = 100 + i;
    SET_ALL_BITS(child1_vec->state->selector);
    // Clear bits in checkerboard pattern
    for (int i = 0; i < CHILD_SIZE; i += 2)
        CLEAR_BIT(child1_vec->state->selector, i);
    SET_START_POS(*child1_vec->state, 0);
    SET_END_POS(*child1_vec->state, CHILD_SIZE - 1);

    for (int i = 0; i < CHILD_SIZE; ++i)
        child2_vec->values[i] = 200 + i;
    SET_ALL_BITS(child2_vec->state->selector);
    // Clear bits in inverse checkerboard pattern
    for (int i = 1; i < CHILD_SIZE; i += 2)
        CLEAR_BIT(child2_vec->state->selector, i);
    SET_START_POS(*child2_vec->state, 0);
    SET_END_POS(*child2_vec->state, CHILD_SIZE - 1);

    // RLE: 4 children per parent
    for (int r = 0; r <= ROOT_SIZE; ++r) {
        child1_vec->state->offset[r] = r * 4;
        child2_vec->state->offset[r] = r * 4;
    }

    auto root_node = std::make_shared<ffx::FactorizedTreeElement>("R", root_vec.get());
    root_node->add_leaf("R", "C1", root_vec.get(), child1_vec.get());
    root_node->add_leaf("R", "C2", root_vec.get(), child2_vec.get());
    std::vector<std::string> col_order = {"R", "C1", "C2"};
    ffx::Schema s;
    s.root = root_node;
    s.column_ordering = &col_order;

    ffx::FTreeIterator itr;
    itr.init(&s);
    itr.initialize_iterators();

    uint64_t manual_count = count_tuples_for_tree(root_node.get());

    uint64_t count = 0;
    uint64_t buffer[3];
    while (itr.is_valid()) {
        bool ok = itr.next(buffer);
        count++;
        if (!ok) break;
    }
    EXPECT_EQ(count, manual_count);
}

// Test boundary positions at start and end
TEST_F(IteratorTest, BoundaryStartEndPositions) {
    // Test when start_pos and end_pos don't cover entire range
    auto root_vec = std::make_unique<ffx::Vector<uint64_t>>();
    auto child_vec = std::make_unique<ffx::Vector<uint64_t>>();

    const int VEC_SIZE = 64;

    for (int i = 0; i < VEC_SIZE; ++i)
        root_vec->values[i] = i;
    SET_ALL_BITS(root_vec->state->selector);
    SET_START_POS(*root_vec->state, 10);// Start at 10, not 0
    SET_END_POS(*root_vec->state, 50);  // End at 50, not 63

    for (int i = 0; i < VEC_SIZE; ++i)
        child_vec->values[i] = 100 + i;
    SET_ALL_BITS(child_vec->state->selector);
    SET_START_POS(*child_vec->state, 5);
    SET_END_POS(*child_vec->state, 55);
    // 1-to-1 mapping
    for (int i = 0; i <= VEC_SIZE; ++i)
        child_vec->state->offset[i] = i;

    auto root_node = std::make_shared<ffx::FactorizedTreeElement>("R", root_vec.get());
    root_node->add_leaf("R", "C", root_vec.get(), child_vec.get());
    std::vector<std::string> col_order = {"R", "C"};
    ffx::Schema s;
    s.root = root_node;
    s.column_ordering = &col_order;

    ffx::FTreeIterator itr;
    itr.init(&s);
    itr.initialize_iterators();

    // Expected: positions 10-50 in root, but child effective range is [max(5, rle[i]), min(55, rle[i+1]-1)]
    // With 1-to-1 mapping: root pos i maps to child pos i
    // Valid pairs: root 10-50, child must also be in [5, 55] -> all 10-50 are valid
    uint64_t expected = 41;// 50 - 10 + 1

    uint64_t count = 0;
    uint64_t buffer[2];
    while (itr.is_valid()) {
        bool ok = itr.next(buffer);
        count++;
        if (!ok) break;
    }
    EXPECT_EQ(count, expected);
}

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}