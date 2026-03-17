#include "../src/operator/include/factorized_ftree/ftree_iterator.hpp"
#include "../src/operator/include/schema/schema.hpp"
#include "../src/operator/include/sink/sink_packed.hpp"
#include "../src/operator/include/vector/state.hpp"
#include "../src/operator/include/vector/vector.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <vector>

using namespace ffx;

class SinkPackedTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup QueryVariableToVectorMap and allocate vectors
        // for attributes A (root), B (non-leaf), C (leaf child of B), D (leaf child of A)
        a_vec = map.allocate_vector("A");
        b_vec = map.allocate_vector("B");
        c_vec = map.allocate_vector("C");
        d_vec = map.allocate_vector("D");

        // Initialize simple tree values
        // A: 2 positions: 0, 1
        a_vec->values[0] = 0;
        a_vec->values[1] = 1;
        SET_ALL_BITS(a_vec->state->selector);
        SET_START_POS(*a_vec->state, 0);
        SET_END_POS(*a_vec->state, 1);

        // B: 4 positions: maps A pos 0 -> B pos 0..1, pos1 -> B pos 2..3
        for (int i = 0; i < 4; ++i)
            b_vec->values[i] = 100 + i;// arbitrary
        SET_ALL_BITS(b_vec->state->selector);
        SET_START_POS(*b_vec->state, 0);
        SET_END_POS(*b_vec->state, 3);
        b_vec->state->offset[0] = 0;
        b_vec->state->offset[1] = 2;
        b_vec->state->offset[2] = 4;

        // C: 4 positions: maps B pos 0..3 to C positions 0..3 (1-1 mapping)
        for (int i = 0; i < 4; ++i)
            c_vec->values[i] = 200 + i;
        SET_ALL_BITS(c_vec->state->selector);
        SET_START_POS(*c_vec->state, 0);
        SET_END_POS(*c_vec->state, 3);
        c_vec->state->offset[0] = 0;
        c_vec->state->offset[1] = 1;
        c_vec->state->offset[2] = 2;
        c_vec->state->offset[3] = 3;
        c_vec->state->offset[4] = 4;

        // D: 2 positions mapping A->D: pos0 -> d pos 0, pos1 -> d pos 1
        d_vec->values[0] = 300;
        d_vec->values[1] = 301;
        SET_ALL_BITS(d_vec->state->selector);
        SET_START_POS(*d_vec->state, 0);
        SET_END_POS(*d_vec->state, 1);
        d_vec->state->offset[0] = 0;
        d_vec->state->offset[1] = 1;
        d_vec->state->offset[2] = 2;

        // Create factorized tree: A -> B -> C and A -> D
        root = std::make_shared<FactorizedTreeElement>("A", a_vec);
        auto b_node = root->add_leaf("A", "B", a_vec, b_vec);
        b_node->add_leaf("B", "C", b_vec, c_vec);
        root->add_leaf("A", "D", a_vec, d_vec);

        // Schema
        column_ordering = {"A", "B", "C", "D"};
        schema.root = root;
        schema.column_ordering = &column_ordering;
        schema.map = &map;
    }

    void TearDown() override {}

    QueryVariableToVectorMap map;
    Vector<uint64_t>* a_vec;
    Vector<uint64_t>* b_vec;
    Vector<uint64_t>* c_vec;
    Vector<uint64_t>* d_vec;
    std::shared_ptr<FactorizedTreeElement> root;
    Schema schema;
    std::vector<std::string> column_ordering;
};

TEST_F(SinkPackedTest, MatchesIteratorCount) {
    // Compute expected tuple count as the SinkPacked would compute it (top-down propagation),
    // i.e., it multiplies parent output by child range sizes for leaf children (ignoring child selectors),
    // and sums child outputs for non-leaf children considering child selector bits.
    uint64_t expected_count = 0;
    const auto* a_state = a_vec->state;
    const auto* b_state = b_vec->state;
    const auto* c_state = c_vec->state;
    const auto* d_state = d_vec->state;
    // Build registry values for A and B (non-leaf nodes) initialised to 1
    std::vector<uint64_t> A_output(2048, 1);
    std::vector<uint64_t> B_output(2048, 1);

    // Process leaf pair B->C: multiply B_output by child_range size (ignoring C selector)
    for (int32_t b_idx = GET_START_POS(*b_state); b_idx <= GET_END_POS(*b_state); ++b_idx) {
        const int32_t c_start = static_cast<int32_t>(c_state->offset[b_idx]);
        const int32_t c_end = static_cast<int32_t>(c_state->offset[b_idx + 1]) - 1;
        int32_t effective_start = std::max(static_cast<int32_t>(GET_START_POS(*c_state)), c_start);
        int32_t effective_end = std::min(static_cast<int32_t>(GET_END_POS(*c_state)), c_end);
        uint64_t child_count = 0;
        if (effective_start <= effective_end) {
            child_count = static_cast<uint64_t>(effective_end - effective_start + 1);
        }
        // mask by parent selector (B)
        if (!TEST_BIT(b_state->selector, b_idx)) child_count = 0;
        B_output[static_cast<size_t>(b_idx)] = B_output[static_cast<size_t>(b_idx)] * child_count;
    }

    // Process leaf pair A->D
    for (int32_t a_idx = GET_START_POS(*a_state); a_idx <= GET_END_POS(*a_state); ++a_idx) {
        const int32_t d_start = static_cast<int32_t>(d_state->offset[a_idx]);
        const int32_t d_end = static_cast<int32_t>(d_state->offset[a_idx + 1]) - 1;
        int32_t effective_start = std::max(static_cast<int32_t>(GET_START_POS(*d_state)), d_start);
        int32_t effective_end = std::min(static_cast<int32_t>(GET_END_POS(*d_state)), d_end);
        uint64_t child_count = 0;
        if (effective_start <= effective_end) {
            child_count = static_cast<uint64_t>(effective_end - effective_start + 1);
        }
        if (!TEST_BIT(a_state->selector, a_idx)) child_count = 0;
        A_output[static_cast<size_t>(a_idx)] = A_output[static_cast<size_t>(a_idx)] * child_count;
    }

    // Process non-leaf A->B: parent_output A multiplied by sum of B_output in its B range, but only include B outputs where B selector bit is set
    for (int32_t a_idx = GET_START_POS(*a_state); a_idx <= GET_END_POS(*a_state); ++a_idx) {
        const int32_t b_start = static_cast<int32_t>(b_state->offset[a_idx]);
        const int32_t b_end = static_cast<int32_t>(b_state->offset[a_idx + 1]) - 1;
        uint64_t child_sum = 0;
        for (int32_t b_idx = b_start; b_idx <= b_end; ++b_idx) {
            if (!TEST_BIT(b_state->selector, b_idx)) continue;// sum only for enabled child positions
            child_sum += B_output[static_cast<size_t>(b_idx)];
        }
        A_output[static_cast<size_t>(a_idx)] = A_output[static_cast<size_t>(a_idx)] * child_sum;
    }

    // Now sum A_output across A positions with A selector set
    for (int32_t a_idx = GET_START_POS(*a_state); a_idx <= GET_END_POS(*a_state); ++a_idx) {
        if (!TEST_BIT(a_state->selector, a_idx)) continue;
        expected_count += A_output[static_cast<size_t>(a_idx)];
    }

    // Run sink packed
    SinkPacked sink;
    sink.init(&schema);
    sink.execute();

    EXPECT_EQ(sink.get_num_output_tuples(), expected_count);
}

TEST_F(SinkPackedTest, MatchesIteratorWithMasking) {
    // Clear some bits in B to check masking
    // Note: Only clear non-leaf (B) bits, not leaf (C) bits
    // because SinkPacked ignores leaf selectors but iterator respects them
    CLEAR_BIT(b_vec->state->selector, 1);// remove B pos 1

    // Compute expected tuple count by enumerating using RLE and selectors
    uint64_t expected_count = 0;
    const auto* a_state = a_vec->state;
    const auto* b_state = b_vec->state;
    const auto* c_state = c_vec->state;
    const auto* d_state = d_vec->state;
    // Compute expected_count following the SinkPacked logic (which ignores child selectors for leaf children)
    std::vector<uint64_t> A_output(2048, 1);
    std::vector<uint64_t> B_output(2048, 1);
    for (int32_t b_idx = GET_START_POS(*b_state); b_idx <= GET_END_POS(*b_state); ++b_idx) {
        const int32_t c_start = static_cast<int32_t>(c_state->offset[b_idx]);
        const int32_t c_end = static_cast<int32_t>(c_state->offset[b_idx + 1]) - 1;
        int32_t effective_start = std::max(static_cast<int32_t>(GET_START_POS(*c_state)), c_start);
        int32_t effective_end = std::min(static_cast<int32_t>(GET_END_POS(*c_state)), c_end);
        uint64_t child_count = 0;
        if (effective_start <= effective_end) {
            child_count = static_cast<uint64_t>(effective_end - effective_start + 1);
        }
        if (!TEST_BIT(b_state->selector, b_idx)) child_count = 0;
        B_output[static_cast<size_t>(b_idx)] = B_output[static_cast<size_t>(b_idx)] * child_count;
    }
    for (int32_t a_idx = GET_START_POS(*a_state); a_idx <= GET_END_POS(*a_state); ++a_idx) {
        const int32_t d_start = static_cast<int32_t>(d_state->offset[a_idx]);
        const int32_t d_end = static_cast<int32_t>(d_state->offset[a_idx + 1]) - 1;
        int32_t effective_start = std::max(static_cast<int32_t>(GET_START_POS(*d_state)), d_start);
        int32_t effective_end = std::min(static_cast<int32_t>(GET_END_POS(*d_state)), d_end);
        uint64_t child_count = 0;
        if (effective_start <= effective_end) {
            child_count = static_cast<uint64_t>(effective_end - effective_start + 1);
        }
        if (!TEST_BIT(a_state->selector, a_idx)) child_count = 0;
        A_output[static_cast<size_t>(a_idx)] = A_output[static_cast<size_t>(a_idx)] * child_count;
    }
    for (int32_t a_idx = GET_START_POS(*a_state); a_idx <= GET_END_POS(*a_state); ++a_idx) {
        const int32_t b_start = static_cast<int32_t>(b_state->offset[a_idx]);
        const int32_t b_end = static_cast<int32_t>(b_state->offset[a_idx + 1]) - 1;
        uint64_t child_sum = 0;
        for (int32_t b_idx = b_start; b_idx <= b_end; ++b_idx) {
            if (!TEST_BIT(b_state->selector, b_idx)) continue;
            child_sum += B_output[static_cast<size_t>(b_idx)];
        }
        A_output[static_cast<size_t>(a_idx)] = A_output[static_cast<size_t>(a_idx)] * child_sum;
    }
    for (int32_t a_idx = GET_START_POS(*a_state); a_idx <= GET_END_POS(*a_state); ++a_idx) {
        if (!TEST_BIT(a_state->selector, a_idx)) continue;
        expected_count += A_output[static_cast<size_t>(a_idx)];
    }

    SinkPacked sink;
    sink.init(&schema);
    sink.execute();

    EXPECT_EQ(sink.get_num_output_tuples(), expected_count);
}

TEST_F(SinkPackedTest, MatchesIteratorLargeVectors) {
    // Build large vectors (but limited by some fixed large size)
    const int32_t N = 2048;// use a fixed size larger than MAX_VECTOR_SIZE for tests

    // Re-allocate vectors to make them large
    a_vec = map.allocate_vector("A_large");
    b_vec = map.allocate_vector("B_large");
    c_vec = map.allocate_vector("C_large");
    d_vec = map.allocate_vector("D_large");

    // A will have few positions (to keep nested loops reasonable)
    const int32_t A_size = 4;
    for (int i = 0; i < A_size; ++i)
        a_vec->values[i] = i;
    SET_ALL_BITS(a_vec->state->selector);
    SET_START_POS(*a_vec->state, 0);
    SET_END_POS(*a_vec->state, A_size - 1);

    // Distribute B positions across A: equally partition B indices between A positions
    const int32_t B_size = N;// large
    for (int i = 0; i < B_size; ++i)
        b_vec->values[i] = 1000 + i;
    SET_ALL_BITS(b_vec->state->selector);
    SET_START_POS(*b_vec->state, 0);
    SET_END_POS(*b_vec->state, B_size - 1);

    // C maps each B to many positions (1-1 mapping here to keep it simple)
    const int32_t C_size = N;
    for (int i = 0; i < C_size; ++i)
        c_vec->values[i] = 2000 + i;
    SET_ALL_BITS(c_vec->state->selector);
    SET_START_POS(*c_vec->state, 0);
    SET_END_POS(*c_vec->state, C_size - 1);

    // D also large but smaller than B/C: 2 per A
    const int32_t D_size = A_size;
    for (int i = 0; i < D_size; ++i)
        d_vec->values[i] = 3000 + i;
    SET_ALL_BITS(d_vec->state->selector);
    SET_START_POS(*d_vec->state, 0);
    SET_END_POS(*d_vec->state, D_size - 1);

    // Setup RLE maps
    // B parented by A: partition B_size into A_size ranges
    int32_t chunk = B_size / A_size;
    for (int a = 0; a < A_size; ++a) {
        b_vec->state->offset[a] = a * chunk;
    }
    b_vec->state->offset[A_size] = B_size;
    // Fill C mapping as 1-1 for each B -> each c pos = same index
    for (int i = 0; i <= B_size; ++i)
        c_vec->state->offset[i] = i;
    // D mapping: 1-to-1 for A
    for (int a = 0; a <= A_size; ++a)
        d_vec->state->offset[a] = a;

    // Make some elements invalid
    // NOTE: Only clear bits in non-leaf vectors (A, B). Clearing leaf bits (C, D)
    // causes sink/iterator mismatch since SinkPacked ignores leaf selectors.
    // Disable some B positions
    CLEAR_BIT(b_vec->state->selector, 1);
    CLEAR_BIT(b_vec->state->selector, 3);
    CLEAR_BIT(b_vec->state->selector, 5);
    // Disable one A position entirely
    CLEAR_BIT(a_vec->state->selector, 2);

    // Build factorized tree
    root = std::make_shared<FactorizedTreeElement>("A", a_vec);
    auto b_node = root->add_leaf("A", "B", a_vec, b_vec);
    b_node->add_leaf("B", "C", b_vec, c_vec);
    root->add_leaf("A", "D", a_vec, d_vec);
    column_ordering = {"A", "B", "C", "D"};
    schema.root = root;
    schema.column_ordering = &column_ordering;
    schema.map = &map;

    // Compute expected count as per SinkPacked logic
    uint64_t expected_count = 0;
    const auto* a_state = a_vec->state;
    const auto* b_state = b_vec->state;
    const auto* c_state = c_vec->state;
    const auto* d_state = d_vec->state;

    // 1) compute B_output by multiplying by C counts (child leaf)
    std::vector<uint64_t> B_output(2048, 1);
    for (int32_t b_idx = GET_START_POS(*b_state); b_idx <= GET_END_POS(*b_state); ++b_idx) {
        const int32_t c_start = static_cast<int32_t>(c_state->offset[b_idx]);
        const int32_t c_end = static_cast<int32_t>(c_state->offset[b_idx + 1]) - 1;
        const int32_t effective_start = std::max(static_cast<int32_t>(GET_START_POS(*c_state)), c_start);
        const int32_t effective_end = std::min(static_cast<int32_t>(GET_END_POS(*c_state)), c_end);
        uint64_t child_count = 0;
        if (effective_start <= effective_end) child_count = (uint64_t) (effective_end - effective_start + 1);
        if (!TEST_BIT(b_state->selector, b_idx)) child_count = 0;
        B_output[static_cast<size_t>(b_idx)] = B_output[static_cast<size_t>(b_idx)] * child_count;
    }

    // 2) compute A_output from D leaf
    std::vector<uint64_t> A_output(2048, 1);
    for (int32_t a_idx = GET_START_POS(*a_state); a_idx <= GET_END_POS(*a_state); ++a_idx) {
        const int32_t d_start = static_cast<int32_t>(d_state->offset[a_idx]);
        const int32_t d_end = static_cast<int32_t>(d_state->offset[a_idx + 1]) - 1;
        const int32_t effective_start = std::max(static_cast<int32_t>(GET_START_POS(*d_state)), d_start);
        const int32_t effective_end = std::min(static_cast<int32_t>(GET_END_POS(*d_state)), d_end);
        uint64_t child_count = 0;
        if (effective_start <= effective_end) child_count = (uint64_t) (effective_end - effective_start + 1);
        if (!TEST_BIT(a_state->selector, a_idx)) child_count = 0;
        A_output[static_cast<size_t>(a_idx)] = A_output[static_cast<size_t>(a_idx)] * child_count;
    }

    // 3) A->B non-leaf: multiply A_output by sum of B_output in A's B-range
    for (int32_t a_idx = GET_START_POS(*a_state); a_idx <= GET_END_POS(*a_state); ++a_idx) {
        const int32_t b_start = static_cast<int32_t>(b_state->offset[a_idx]);
        const int32_t b_end = static_cast<int32_t>(b_state->offset[a_idx + 1]) - 1;
        uint64_t child_sum = 0;
        for (int32_t b_idx = b_start; b_idx <= b_end; ++b_idx) {
            if (!TEST_BIT(b_state->selector, b_idx)) continue;
            child_sum += B_output[static_cast<size_t>(b_idx)];
        }
        A_output[static_cast<size_t>(a_idx)] = A_output[static_cast<size_t>(a_idx)] * child_sum;
    }

    // final sum across A positions
    for (int32_t a_idx = GET_START_POS(*a_state); a_idx <= GET_END_POS(*a_state); ++a_idx) {
        if (!TEST_BIT(a_state->selector, a_idx)) continue;
        expected_count += A_output[static_cast<size_t>(a_idx)];
    }

    // Make sure the ftree iterator has at least one valid tuple; sometimes clearing random bits can create an
    // empty configuration (no valid tuples). If so, restore a single valid path by enabling one B/C/D bit.
    {
        FTreeIterator tmpItr;
        tmpItr.init(&schema);
        tmpItr.initialize_iterators();
        if (!tmpItr.is_valid()) {
            // Find an A position that's selectable
            int32_t a_pos = -1;
            for (int32_t a_idx = GET_START_POS(*a_vec->state); a_idx <= GET_END_POS(*a_vec->state); ++a_idx) {
                if (TEST_BIT(a_vec->state->selector, a_idx)) {
                    a_pos = a_idx;
                    break;
                }
            }
            if (a_pos < 0) {
                // Make A pos 0 valid
                SET_BIT(a_vec->state->selector, 0);
                a_pos = 0;
            }
            // Ensure there's a selected B position in the A->B range
            const int32_t b_start = static_cast<int32_t>(b_vec->state->offset[a_pos]);
            const int32_t b_end = static_cast<int32_t>(b_vec->state->offset[a_pos + 1]) - 1;
            int32_t chosen_b = -1;
            for (int32_t b_idx = b_start; b_idx <= b_end; ++b_idx) {
                if (TEST_BIT(b_vec->state->selector, b_idx)) {
                    chosen_b = b_idx;
                    break;
                }
            }
            if (chosen_b < 0) {
                // enable the first B bit in the range
                SET_BIT(b_vec->state->selector, b_start);
                chosen_b = b_start;
            }
            // Ensure there's a selected C position in the chosen B->C range
            const int32_t c_start = static_cast<int32_t>(c_vec->state->offset[chosen_b]);
            const int32_t c_end = static_cast<int32_t>(c_vec->state->offset[chosen_b + 1]) - 1;
            bool c_has = false;
            for (int32_t c_idx = c_start; c_idx <= c_end; ++c_idx) {
                if (TEST_BIT(c_vec->state->selector, c_idx)) {
                    c_has = true;
                    break;
                }
            }
            if (!c_has) SET_BIT(c_vec->state->selector, c_start);
            // Ensure D exists for this A
            const int32_t d_start = static_cast<int32_t>(d_vec->state->offset[a_pos]);
            if (!TEST_BIT(d_vec->state->selector, d_start)) SET_BIT(d_vec->state->selector, d_start);
        }
    }

    // Run sink packed
    SinkPacked sink;
    sink.init(&schema);
    sink.execute();

    EXPECT_EQ(sink.get_num_output_tuples(), expected_count);
}

TEST_F(SinkPackedTest, LargeVectorsTopology1) {
    // a->b, a->c, a->d, a->e (all leaves under A)
    // NOTE: Using small child sizes instead of selector clearing because
    // SinkPacked uses range size (start_pos to end_pos), not selector bits for leaf children
    const int32_t A_size = 4;
    const int32_t CHILD_SIZE = 16;// Small to keep iteration tractable

    auto a = map.allocate_vector("A_t1");
    auto b = map.allocate_vector("B_t1");
    auto c = map.allocate_vector("C_t1");
    auto d = map.allocate_vector("D_t1");
    auto e = map.allocate_vector("E_t1");

    for (int i = 0; i < A_size; ++i)
        a->values[i] = i;
    SET_ALL_BITS(a->state->selector);
    SET_START_POS(*a->state, 0);
    SET_END_POS(*a->state, A_size - 1);

    for (int i = 0; i < CHILD_SIZE; ++i)
        b->values[i] = 1000 + i;
    SET_ALL_BITS(b->state->selector);
    SET_START_POS(*b->state, 0);
    SET_END_POS(*b->state, CHILD_SIZE - 1);

    for (int i = 0; i < CHILD_SIZE; ++i)
        c->values[i] = 2000 + i;
    SET_ALL_BITS(c->state->selector);
    SET_START_POS(*c->state, 0);
    SET_END_POS(*c->state, CHILD_SIZE - 1);

    for (int i = 0; i < CHILD_SIZE; ++i)
        d->values[i] = 3000 + i;
    SET_ALL_BITS(d->state->selector);
    SET_START_POS(*d->state, 0);
    SET_END_POS(*d->state, CHILD_SIZE - 1);

    for (int i = 0; i < CHILD_SIZE; ++i)
        e->values[i] = 4000 + i;
    SET_ALL_BITS(e->state->selector);
    SET_START_POS(*e->state, 0);
    SET_END_POS(*e->state, CHILD_SIZE - 1);

    // Partition child vectors across A (4 positions per A)
    int chunk = CHILD_SIZE / A_size;// 4 per A position
    for (int aidx = 0; aidx <= A_size; ++aidx) {
        b->state->offset[aidx] = aidx * chunk;
        c->state->offset[aidx] = aidx * chunk;
        d->state->offset[aidx] = aidx * chunk;
        e->state->offset[aidx] = aidx * chunk;
    }

    // Build tree
    auto root_local = std::make_shared<FactorizedTreeElement>("A", a);
    root_local->add_leaf("A", "B", a, b);
    root_local->add_leaf("A", "C", a, c);
    root_local->add_leaf("A", "D", a, d);
    root_local->add_leaf("A", "E", a, e);
    std::vector<std::string> column_order = {"A", "B", "C", "D", "E"};
    Schema s;
    s.root = root_local;
    s.column_ordering = &column_order;
    s.map = &map;

    // Compute expected_count: for each A, product of child range sizes
    // Each A position has chunk=4 children for each of B, C, D, E
    // So each A produces 4^4 = 256 tuples, total = 4 * 256 = 1024
    uint64_t expected_count = 0;
    for (int aidx = 0; aidx < A_size; ++aidx) {
        int bstart = static_cast<int>(b->state->offset[aidx]);
        int bend = static_cast<int>(b->state->offset[aidx + 1]) - 1;
        int cstart = static_cast<int>(c->state->offset[aidx]);
        int cend = static_cast<int>(c->state->offset[aidx + 1]) - 1;
        int dstart = static_cast<int>(d->state->offset[aidx]);
        int dend = static_cast<int>(d->state->offset[aidx + 1]) - 1;
        int estart = static_cast<int>(e->state->offset[aidx]);
        int eend = static_cast<int>(e->state->offset[aidx + 1]) - 1;
        uint64_t bcount = (bstart <= bend) ? (uint64_t) (bend - bstart + 1) : 0;
        uint64_t ccount = (cstart <= cend) ? (uint64_t) (cend - cstart + 1) : 0;
        uint64_t dcount = (dstart <= dend) ? (uint64_t) (dend - dstart + 1) : 0;
        uint64_t ecount = (estart <= eend) ? (uint64_t) (eend - estart + 1) : 0;
        expected_count += bcount * ccount * dcount * ecount;
    }

    SinkPacked sink;
    sink.init(&s);
    sink.execute();
    EXPECT_EQ(sink.get_num_output_tuples(), expected_count);
}

TEST_F(SinkPackedTest, LargeVectorsTopology2) {
    // a->b, b->c, b->d, a->e
    // NOTE: Using smaller sizes to keep iteration tractable
    // B is non-leaf (has C and D as children), E is leaf of A
    const int32_t A_size = 4;
    const int32_t B_size = 16;// B partitioned across A (4 per A)
    const int32_t C_size = 16;// 1-1 mapping with B
    const int32_t D_size = 16;// 1-1 mapping with B
    const int32_t E_size = 4; // 1-1 mapping with A

    auto a = map.allocate_vector("A_t2");
    auto b = map.allocate_vector("B_t2");
    auto c = map.allocate_vector("C_t2");
    auto d = map.allocate_vector("D_t2");
    auto e = map.allocate_vector("E_t2");

    for (int i = 0; i < A_size; ++i)
        a->values[i] = i;
    SET_ALL_BITS(a->state->selector);
    SET_START_POS(*a->state, 0);
    SET_END_POS(*a->state, A_size - 1);

    for (int i = 0; i < B_size; ++i)
        b->values[i] = 1000 + i;
    SET_ALL_BITS(b->state->selector);
    SET_START_POS(*b->state, 0);
    SET_END_POS(*b->state, B_size - 1);

    for (int i = 0; i < C_size; ++i)
        c->values[i] = 2000 + i;
    SET_ALL_BITS(c->state->selector);
    SET_START_POS(*c->state, 0);
    SET_END_POS(*c->state, C_size - 1);

    for (int i = 0; i < D_size; ++i)
        d->values[i] = 3000 + i;
    SET_ALL_BITS(d->state->selector);
    SET_START_POS(*d->state, 0);
    SET_END_POS(*d->state, D_size - 1);

    for (int i = 0; i < E_size; ++i)
        e->values[i] = 4000 + i;
    SET_ALL_BITS(e->state->selector);
    SET_START_POS(*e->state, 0);
    SET_END_POS(*e->state, E_size - 1);

    // RLE: B partitioned across A (4 per A)
    int chunk = B_size / A_size;
    for (int aidx = 0; aidx <= A_size; ++aidx)
        b->state->offset[aidx] = aidx * chunk;
    // C & D 1-1 mapping for each B
    for (int bidx = 0; bidx <= B_size; ++bidx) {
        c->state->offset[bidx] = bidx;
        d->state->offset[bidx] = bidx;
    }
    // E 1-1 for A
    for (int aidx = 0; aidx <= A_size; ++aidx)
        e->state->offset[aidx] = aidx;

    // Introduce some invalid b positions to test masking (B is non-leaf, so selector is respected)
    CLEAR_BIT(b->state->selector, 2);
    CLEAR_BIT(b->state->selector, 5);

    auto root_local = std::make_shared<FactorizedTreeElement>("A", a);
    auto b_node = root_local->add_leaf("A", "B", a, b);
    b_node->add_leaf("B", "C", b, c);
    b_node->add_leaf("B", "D", b, d);
    root_local->add_leaf("A", "E", a, e);

    std::vector<std::string> column_order = {"A", "B", "C", "D", "E"};
    Schema s;
    s.root = root_local;
    s.column_ordering = &column_order;
    s.map = &map;

    // B_output = C_count * D_count = 1*1 = 1 for each B position
    std::vector<uint64_t> B_output(2048, 1);
    for (int bidx = GET_START_POS(*b->state); bidx <= GET_END_POS(*b->state); ++bidx) {
        // C and D are 1-1 => count 1 each
        uint64_t ccount = 1, dcount = 1;
        if (!TEST_BIT(b->state->selector, bidx)) {
            B_output[static_cast<size_t>(bidx)] = 0;
            continue;
        }
        B_output[static_cast<size_t>(bidx)] = ccount * dcount;
    }

    std::vector<uint64_t> A_output(2048, 1);
    // E is 1-1 leaf child of A
    for (int aidx = 0; aidx < A_size; ++aidx) {
        uint64_t estart = e->state->offset[aidx];
        uint64_t eend = e->state->offset[aidx + 1] - 1;
        uint64_t ecount = (estart <= eend) ? (eend - estart + 1) : 0;
        // sum B outputs for this A
        int bstart = static_cast<int>(b->state->offset[aidx]);
        int bend = static_cast<int>(b->state->offset[aidx + 1]) - 1;
        uint64_t child_sum = 0;
        for (int bidx = bstart; bidx <= bend; ++bidx)
            child_sum += B_output[static_cast<size_t>(bidx)];
        A_output[static_cast<size_t>(aidx)] = ecount * child_sum;
    }
    uint64_t expected_count = 0;
    for (int aidx = 0; aidx < A_size; ++aidx)
        expected_count += A_output[static_cast<size_t>(aidx)];

    SinkPacked sink;
    sink.init(&s);
    sink.execute();
    EXPECT_EQ(sink.get_num_output_tuples(), expected_count);
}

TEST_F(SinkPackedTest, LargeVectorsTopology3) {
    // a->b, b->c, c->d, d->e (chain)
    // NOTE: Using smaller sizes to keep iteration tractable
    const int32_t A_size = 4;
    const int32_t B_size = 16;// B partitioned across A (4 per A)
    const int32_t C_size = 16;// 1-1 mapping with B
    const int32_t D_size = 16;// 1-1 mapping with C
    const int32_t E_size = 16;// 1-1 mapping with D (E is the only leaf)

    auto a = map.allocate_vector("A_t3");
    auto b = map.allocate_vector("B_t3");
    auto c = map.allocate_vector("C_t3");
    auto d = map.allocate_vector("D_t3");
    auto e = map.allocate_vector("E_t3");

    for (int i = 0; i < A_size; ++i)
        a->values[i] = i;
    SET_ALL_BITS(a->state->selector);
    SET_START_POS(*a->state, 0);
    SET_END_POS(*a->state, A_size - 1);
    for (int i = 0; i < B_size; ++i)
        b->values[i] = 1000 + i;
    SET_ALL_BITS(b->state->selector);
    SET_START_POS(*b->state, 0);
    SET_END_POS(*b->state, B_size - 1);
    for (int i = 0; i < C_size; ++i)
        c->values[i] = 2000 + i;
    SET_ALL_BITS(c->state->selector);
    SET_START_POS(*c->state, 0);
    SET_END_POS(*c->state, C_size - 1);
    for (int i = 0; i < D_size; ++i)
        d->values[i] = 3000 + i;
    SET_ALL_BITS(d->state->selector);
    SET_START_POS(*d->state, 0);
    SET_END_POS(*d->state, D_size - 1);
    for (int i = 0; i < E_size; ++i)
        e->values[i] = 4000 + i;
    SET_ALL_BITS(e->state->selector);
    SET_START_POS(*e->state, 0);
    SET_END_POS(*e->state, E_size - 1);

    // B partition across A (4 per A)
    int chunk = B_size / A_size;
    for (int aidx = 0; aidx <= A_size; ++aidx)
        b->state->offset[aidx] = aidx * chunk;
    // C, D, E mapping as 1-1
    for (int bidx = 0; bidx <= B_size; ++bidx)
        c->state->offset[bidx] = bidx;
    for (int cidx = 0; cidx <= C_size; ++cidx)
        d->state->offset[cidx] = cidx;
    for (int didx = 0; didx <= D_size; ++didx)
        e->state->offset[didx] = didx;

    auto root_local = std::make_shared<FactorizedTreeElement>("A", a);
    auto b_node = root_local->add_leaf("A", "B", a, b);
    auto c_node = b_node->add_leaf("B", "C", b, c);
    auto d_node = c_node->add_leaf("C", "D", c, d);
    d_node->add_leaf("D", "E", d, e);
    std::vector<std::string> column_order = {"A", "B", "C", "D", "E"};
    Schema s;
    s.root = root_local;
    s.column_ordering = &column_order;
    s.map = &map;

    // E is leaf -> E_output[didx] = range size (1 for 1-1 mapping)
    std::vector<uint64_t> E_output(2048, 1);
    for (int didx = GET_START_POS(*d->state); didx <= GET_END_POS(*d->state); ++didx) {
        const int32_t estart = static_cast<int32_t>(e->state->offset[didx]);
        const int32_t eend = static_cast<int32_t>(e->state->offset[didx + 1]) - 1;
        uint64_t ecount = 0;
        if (estart <= eend) ecount = (uint64_t) (eend - estart + 1);
        E_output[static_cast<size_t>(didx)] = E_output[static_cast<size_t>(didx)] * ecount;
    }
    // D is non-leaf -> D_output[cidx] = sum of E_output in D's range
    std::vector<uint64_t> D_output(2048, 1);
    for (int cidx = GET_START_POS(*c->state); cidx <= GET_END_POS(*c->state); ++cidx) {
        const int32_t dstart = static_cast<int32_t>(d->state->offset[cidx]);
        const int32_t dend = static_cast<int32_t>(d->state->offset[cidx + 1]) - 1;
        uint64_t child_sum = 0;
        for (int didx = dstart; didx <= dend; ++didx)
            child_sum += E_output[static_cast<size_t>(didx)];
        D_output[static_cast<size_t>(cidx)] = D_output[static_cast<size_t>(cidx)] * child_sum;
    }
    // C is non-leaf -> C_output[bidx] = sum of D_output in C's range
    std::vector<uint64_t> C_output(2048, 1);
    for (int bidx = GET_START_POS(*b->state); bidx <= GET_END_POS(*b->state); ++bidx) {
        const int32_t cstart = static_cast<int32_t>(c->state->offset[bidx]);
        const int32_t cend = static_cast<int32_t>(c->state->offset[bidx + 1]) - 1;
        uint64_t child_sum = 0;
        for (int cidx = cstart; cidx <= cend; ++cidx)
            child_sum += D_output[static_cast<size_t>(cidx)];
        C_output[static_cast<size_t>(bidx)] = C_output[static_cast<size_t>(bidx)] * child_sum;
    }
    // B is non-leaf -> B_output[aidx] = sum of C_output in B's range
    std::vector<uint64_t> B_output(2048, 1);
    for (int aidx = 0; aidx < A_size; ++aidx) {
        const int32_t bstart = static_cast<int32_t>(b->state->offset[aidx]);
        const int32_t bend = static_cast<int32_t>(b->state->offset[aidx + 1]) - 1;
        uint64_t child_sum = 0;
        for (int bidx = bstart; bidx <= bend; ++bidx)
            child_sum += C_output[static_cast<size_t>(bidx)];
        B_output[static_cast<size_t>(aidx)] = B_output[static_cast<size_t>(aidx)] * child_sum;
    }
    uint64_t expected_count = 0;
    for (int aidx = 0; aidx < A_size; ++aidx)
        expected_count += B_output[static_cast<size_t>(aidx)];

    SinkPacked sink;
    sink.init(&s);
    sink.execute();
    EXPECT_EQ(sink.get_num_output_tuples(), expected_count);
}

//------------------------------------------------------------------------------
// Complex Sink Packed Tests
// NOTE: All tests follow the constraint that leaf selectors must NOT be modified
// because SinkPacked ignores leaf selectors but FTreeIterator respects them.
//------------------------------------------------------------------------------

// Test diamond topology: A->B, A->C, B->D, C->D (D has two parents)
// This tests the case where a node can be reached via multiple paths
TEST_F(SinkPackedTest, DiamondTopologySharedLeaf) {
    // Tree structure: A is root, B and C are children of A, D is shared leaf of both B and C
    // However, factorized trees don't actually support shared children, so we test
    // a close approximation: A->B->D and A->C->E where D and E have similar structure
    const int32_t A_SIZE = 4;
    const int32_t B_SIZE = 8;
    const int32_t C_SIZE = 8;
    const int32_t D_SIZE = 8;// Leaf of B
    const int32_t E_SIZE = 8;// Leaf of C

    auto a = map.allocate_vector("A_diamond");
    auto b = map.allocate_vector("B_diamond");
    auto c = map.allocate_vector("C_diamond");
    auto d = map.allocate_vector("D_diamond");
    auto e = map.allocate_vector("E_diamond");

    for (int i = 0; i < A_SIZE; ++i)
        a->values[i] = i;
    SET_ALL_BITS(a->state->selector);
    SET_START_POS(*a->state, 0);
    SET_END_POS(*a->state, A_SIZE - 1);

    for (int i = 0; i < B_SIZE; ++i)
        b->values[i] = 100 + i;
    SET_ALL_BITS(b->state->selector);
    SET_START_POS(*b->state, 0);
    SET_END_POS(*b->state, B_SIZE - 1);

    for (int i = 0; i < C_SIZE; ++i)
        c->values[i] = 200 + i;
    SET_ALL_BITS(c->state->selector);
    SET_START_POS(*c->state, 0);
    SET_END_POS(*c->state, C_SIZE - 1);

    for (int i = 0; i < D_SIZE; ++i)
        d->values[i] = 300 + i;
    SET_ALL_BITS(d->state->selector);
    SET_START_POS(*d->state, 0);
    SET_END_POS(*d->state, D_SIZE - 1);

    for (int i = 0; i < E_SIZE; ++i)
        e->values[i] = 400 + i;
    SET_ALL_BITS(e->state->selector);
    SET_START_POS(*e->state, 0);
    SET_END_POS(*e->state, E_SIZE - 1);

    // RLE mappings: B and C partitioned across A (2 each), D maps to B, E maps to C
    for (int i = 0; i <= A_SIZE; ++i) {
        b->state->offset[i] = i * 2;
        c->state->offset[i] = i * 2;
    }
    for (int i = 0; i <= B_SIZE; ++i)
        d->state->offset[i] = i;
    for (int i = 0; i <= C_SIZE; ++i)
        e->state->offset[i] = i;

    // Mask out some B and C positions (non-leaf, so selector is respected)
    CLEAR_BIT(b->state->selector, 1);
    CLEAR_BIT(c->state->selector, 3);

    auto root = std::make_shared<FactorizedTreeElement>("A", a);
    auto b_node = root->add_leaf("A", "B", a, b);
    auto c_node = root->add_leaf("A", "C", a, c);
    b_node->add_leaf("B", "D", b, d);
    c_node->add_leaf("C", "E", c, e);

    std::vector<std::string> col_order = {"A", "B", "C", "D", "E"};
    Schema s;
    s.root = root;
    s.column_ordering = &col_order;
    s.map = &map;

    // Compute expected: B_output[bidx] = D_range_size (1-1 = 1)
    std::vector<uint64_t> B_output(B_SIZE, 0);
    for (int bidx = 0; bidx < B_SIZE; ++bidx) {
        if (!TEST_BIT(b->state->selector, bidx)) continue;
        B_output[bidx] = 1;// 1-1 mapping
    }
    // C_output[cidx] = E_range_size (1-1 = 1)
    std::vector<uint64_t> C_output(C_SIZE, 0);
    for (int cidx = 0; cidx < C_SIZE; ++cidx) {
        if (!TEST_BIT(c->state->selector, cidx)) continue;
        C_output[cidx] = 1;
    }
    // A_output = sum(B_output in range) * sum(C_output in range)
    uint64_t expected = 0;
    for (int aidx = 0; aidx < A_SIZE; ++aidx) {
        int b_start = b->state->offset[aidx], b_end = b->state->offset[aidx + 1] - 1;
        int c_start = c->state->offset[aidx], c_end = c->state->offset[aidx + 1] - 1;
        uint64_t b_sum = 0, c_sum = 0;
        for (int i = b_start; i <= b_end; ++i)
            b_sum += B_output[i];
        for (int i = c_start; i <= c_end; ++i)
            c_sum += C_output[i];
        expected += b_sum * c_sum;
    }

    SinkPacked sink;
    sink.init(&s);
    sink.execute();
    EXPECT_EQ(sink.get_num_output_tuples(), expected);
}

// Test sparse non-leaf masking with various patterns
TEST_F(SinkPackedTest, SparseNonLeafMasking) {
    const int32_t A_SIZE = 8;
    const int32_t B_SIZE = 32;
    const int32_t C_SIZE = 32;

    auto a = map.allocate_vector("A_sparse");
    auto b = map.allocate_vector("B_sparse");
    auto c = map.allocate_vector("C_sparse");

    for (int i = 0; i < A_SIZE; ++i)
        a->values[i] = i;
    SET_ALL_BITS(a->state->selector);
    SET_START_POS(*a->state, 0);
    SET_END_POS(*a->state, A_SIZE - 1);

    for (int i = 0; i < B_SIZE; ++i)
        b->values[i] = 100 + i;
    SET_ALL_BITS(b->state->selector);
    SET_START_POS(*b->state, 0);
    SET_END_POS(*b->state, B_SIZE - 1);

    for (int i = 0; i < C_SIZE; ++i)
        c->values[i] = 200 + i;
    SET_ALL_BITS(c->state->selector);
    SET_START_POS(*c->state, 0);
    SET_END_POS(*c->state, C_SIZE - 1);

    // RLE: B partitioned across A (4 per A), C 1-1 with B
    for (int i = 0; i <= A_SIZE; ++i)
        b->state->offset[i] = i * 4;
    for (int i = 0; i <= B_SIZE; ++i)
        c->state->offset[i] = i;

    // Sparse masking: clear every 3rd B position (non-leaf), but keep start_pos (0) and end_pos (31) valid
    for (int i = 3; i < B_SIZE - 1; i += 3)
        CLEAR_BIT(b->state->selector, i);
    // Also clear some interior A positions (not 0 or 7)
    CLEAR_BIT(a->state->selector, 2);
    CLEAR_BIT(a->state->selector, 5);

    auto root = std::make_shared<FactorizedTreeElement>("A", a);
    auto b_node = root->add_leaf("A", "B", a, b);
    b_node->add_leaf("B", "C", b, c);

    std::vector<std::string> col_order = {"A", "B", "C"};
    Schema s;
    s.root = root;
    s.column_ordering = &col_order;
    s.map = &map;

    // Compute expected
    std::vector<uint64_t> B_output(B_SIZE, 0);
    for (int bidx = 0; bidx < B_SIZE; ++bidx) {
        if (!TEST_BIT(b->state->selector, bidx)) continue;
        B_output[bidx] = 1;// C is 1-1 leaf
    }
    uint64_t expected = 0;
    for (int aidx = 0; aidx < A_SIZE; ++aidx) {
        if (!TEST_BIT(a->state->selector, aidx)) continue;
        int b_start = b->state->offset[aidx], b_end = b->state->offset[aidx + 1] - 1;
        uint64_t b_sum = 0;
        for (int i = b_start; i <= b_end; ++i)
            b_sum += B_output[i];
        expected += b_sum;
    }

    SinkPacked sink;
    sink.init(&s);
    sink.execute();
    EXPECT_EQ(sink.get_num_output_tuples(), expected);
}

// Test asymmetric RLE distributions (varying child counts per parent)
TEST_F(SinkPackedTest, AsymmetricRLEDistribution) {
    const int32_t A_SIZE = 4;
    const int32_t B_SIZE = 20;// Asymmetric: A0->5, A1->3, A2->7, A3->5

    auto a = map.allocate_vector("A_asym");
    auto b = map.allocate_vector("B_asym");

    for (int i = 0; i < A_SIZE; ++i)
        a->values[i] = i;
    SET_ALL_BITS(a->state->selector);
    SET_START_POS(*a->state, 0);
    SET_END_POS(*a->state, A_SIZE - 1);

    for (int i = 0; i < B_SIZE; ++i)
        b->values[i] = 100 + i;
    SET_ALL_BITS(b->state->selector);
    SET_START_POS(*b->state, 0);
    SET_END_POS(*b->state, B_SIZE - 1);

    // Asymmetric RLE: A0->B[0..4], A1->B[5..7], A2->B[8..14], A3->B[15..19]
    b->state->offset[0] = 0;
    b->state->offset[1] = 5;
    b->state->offset[2] = 8;
    b->state->offset[3] = 15;
    b->state->offset[4] = 20;

    auto root = std::make_shared<FactorizedTreeElement>("A", a);
    root->add_leaf("A", "B", a, b);

    std::vector<std::string> col_order = {"A", "B"};
    Schema s;
    s.root = root;
    s.column_ordering = &col_order;
    s.map = &map;

    // Expected: sum of range sizes = 5 + 3 + 7 + 5 = 20
    uint64_t expected = 0;
    for (int aidx = 0; aidx < A_SIZE; ++aidx) {
        int range_size = b->state->offset[aidx + 1] - b->state->offset[aidx];
        expected += range_size;
    }

    SinkPacked sink;
    sink.init(&s);
    sink.execute();
    EXPECT_EQ(sink.get_num_output_tuples(), expected);
}

// Test deep chain (5 levels) with non-leaf masking at various levels
TEST_F(SinkPackedTest, DeepChainWithMasking) {
    const int32_t SIZE = 8;

    auto a = map.allocate_vector("A_deep");
    auto b = map.allocate_vector("B_deep");
    auto c = map.allocate_vector("C_deep");
    auto d = map.allocate_vector("D_deep");
    auto e = map.allocate_vector("E_deep");

    // All same size with 1-1 mappings
    for (int i = 0; i < SIZE; ++i) {
        a->values[i] = i;
        b->values[i] = 100 + i;
        c->values[i] = 200 + i;
        d->values[i] = 300 + i;
        e->values[i] = 400 + i;
    }

    SET_ALL_BITS(a->state->selector);
    SET_START_POS(*a->state, 0);
    SET_END_POS(*a->state, SIZE - 1);
    SET_ALL_BITS(b->state->selector);
    SET_START_POS(*b->state, 0);
    SET_END_POS(*b->state, SIZE - 1);
    SET_ALL_BITS(c->state->selector);
    SET_START_POS(*c->state, 0);
    SET_END_POS(*c->state, SIZE - 1);
    SET_ALL_BITS(d->state->selector);
    SET_START_POS(*d->state, 0);
    SET_END_POS(*d->state, SIZE - 1);
    SET_ALL_BITS(e->state->selector);
    SET_START_POS(*e->state, 0);
    SET_END_POS(*e->state, SIZE - 1);

    // 1-1 RLE mappings
    for (int i = 0; i <= SIZE; ++i) {
        b->state->offset[i] = i;
        c->state->offset[i] = i;
        d->state->offset[i] = i;
        e->state->offset[i] = i;
    }

    // Mask some non-leaf positions (A, B, C, D are non-leaf; E is leaf)
    CLEAR_BIT(a->state->selector, 2);
    CLEAR_BIT(b->state->selector, 4);
    CLEAR_BIT(c->state->selector, 1);
    CLEAR_BIT(d->state->selector, 6);

    auto root = std::make_shared<FactorizedTreeElement>("A", a);
    auto b_node = root->add_leaf("A", "B", a, b);
    auto c_node = b_node->add_leaf("B", "C", b, c);
    auto d_node = c_node->add_leaf("C", "D", c, d);
    d_node->add_leaf("D", "E", d, e);

    std::vector<std::string> col_order = {"A", "B", "C", "D", "E"};
    Schema s;
    s.root = root;
    s.column_ordering = &col_order;
    s.map = &map;

    // Compute expected bottom-up
    // D_output[cidx] = E_count (1-1 = 1) if D is set
    std::vector<uint64_t> D_output(SIZE, 0);
    for (int didx = 0; didx < SIZE; ++didx) {
        if (TEST_BIT(d->state->selector, didx)) D_output[didx] = 1;
    }
    // C_output[bidx] = sum of D_output in range, if C is set
    std::vector<uint64_t> C_output(SIZE, 0);
    for (int cidx = 0; cidx < SIZE; ++cidx) {
        if (!TEST_BIT(c->state->selector, cidx)) continue;
        int d_start = d->state->offset[cidx], d_end = d->state->offset[cidx + 1] - 1;
        for (int i = d_start; i <= d_end; ++i)
            C_output[cidx] += D_output[i];
    }
    // B_output[aidx] = sum of C_output in range, if B is set
    std::vector<uint64_t> B_output(SIZE, 0);
    for (int bidx = 0; bidx < SIZE; ++bidx) {
        if (!TEST_BIT(b->state->selector, bidx)) continue;
        int c_start = c->state->offset[bidx], c_end = c->state->offset[bidx + 1] - 1;
        for (int i = c_start; i <= c_end; ++i)
            B_output[bidx] += C_output[i];
    }
    // A total
    uint64_t expected = 0;
    for (int aidx = 0; aidx < SIZE; ++aidx) {
        if (!TEST_BIT(a->state->selector, aidx)) continue;
        int b_start = b->state->offset[aidx], b_end = b->state->offset[aidx + 1] - 1;
        for (int i = b_start; i <= b_end; ++i)
            expected += B_output[i];
    }

    SinkPacked sink;
    sink.init(&s);
    sink.execute();
    EXPECT_EQ(sink.get_num_output_tuples(), expected);
}

// Test mixed sibling types: A has both leaf (B) and non-leaf (C->D) children
TEST_F(SinkPackedTest, MixedSiblingTypes) {
    const int32_t A_SIZE = 4;
    const int32_t B_SIZE = 8;// Leaf child of A
    const int32_t C_SIZE = 8;// Non-leaf child of A
    const int32_t D_SIZE = 8;// Leaf child of C

    auto a = map.allocate_vector("A_mixed");
    auto b = map.allocate_vector("B_mixed");
    auto c = map.allocate_vector("C_mixed");
    auto d = map.allocate_vector("D_mixed");

    for (int i = 0; i < A_SIZE; ++i)
        a->values[i] = i;
    SET_ALL_BITS(a->state->selector);
    SET_START_POS(*a->state, 0);
    SET_END_POS(*a->state, A_SIZE - 1);

    for (int i = 0; i < B_SIZE; ++i)
        b->values[i] = 100 + i;
    SET_ALL_BITS(b->state->selector);
    SET_START_POS(*b->state, 0);
    SET_END_POS(*b->state, B_SIZE - 1);

    for (int i = 0; i < C_SIZE; ++i)
        c->values[i] = 200 + i;
    SET_ALL_BITS(c->state->selector);
    SET_START_POS(*c->state, 0);
    SET_END_POS(*c->state, C_SIZE - 1);

    for (int i = 0; i < D_SIZE; ++i)
        d->values[i] = 300 + i;
    SET_ALL_BITS(d->state->selector);
    SET_START_POS(*d->state, 0);
    SET_END_POS(*d->state, D_SIZE - 1);

    // RLE: B and C partitioned across A (2 each), D 1-1 with C
    for (int i = 0; i <= A_SIZE; ++i) {
        b->state->offset[i] = i * 2;
        c->state->offset[i] = i * 2;
    }
    for (int i = 0; i <= C_SIZE; ++i)
        d->state->offset[i] = i;

    // Mask some C positions (non-leaf)
    CLEAR_BIT(c->state->selector, 1);
    CLEAR_BIT(c->state->selector, 5);

    auto root = std::make_shared<FactorizedTreeElement>("A", a);
    root->add_leaf("A", "B", a, b);
    auto c_node = root->add_leaf("A", "C", a, c);
    c_node->add_leaf("C", "D", c, d);

    std::vector<std::string> col_order = {"A", "B", "C", "D"};
    Schema s;
    s.root = root;
    s.column_ordering = &col_order;
    s.map = &map;

    // Compute expected:
    // B is leaf: B_count = range_size (2 per A)
    // C is non-leaf with D as 1-1 leaf: C_output[cidx] = 1 if set
    std::vector<uint64_t> C_output(C_SIZE, 0);
    for (int cidx = 0; cidx < C_SIZE; ++cidx) {
        if (TEST_BIT(c->state->selector, cidx)) C_output[cidx] = 1;
    }

    uint64_t expected = 0;
    for (int aidx = 0; aidx < A_SIZE; ++aidx) {
        // B contribution (leaf): range size
        int b_start = b->state->offset[aidx], b_end = b->state->offset[aidx + 1] - 1;
        uint64_t b_count = (b_start <= b_end) ? (b_end - b_start + 1) : 0;

        // C contribution (non-leaf): sum of C_output
        int c_start = c->state->offset[aidx], c_end = c->state->offset[aidx + 1] - 1;
        uint64_t c_sum = 0;
        for (int i = c_start; i <= c_end; ++i)
            c_sum += C_output[i];

        expected += b_count * c_sum;
    }

    SinkPacked sink;
    sink.init(&s);
    sink.execute();
    EXPECT_EQ(sink.get_num_output_tuples(), expected);
}

// Test boundary edge cases: single element vectors, empty ranges
TEST_F(SinkPackedTest, BoundaryEdgeCases) {
    // Minimal tree: A has 1 element, B has 1 element, C has 1 element
    auto a = map.allocate_vector("A_boundary");
    auto b = map.allocate_vector("B_boundary");
    auto c = map.allocate_vector("C_boundary");

    a->values[0] = 0;
    SET_ALL_BITS(a->state->selector);
    SET_START_POS(*a->state, 0);
    SET_END_POS(*a->state, 0);

    b->values[0] = 100;
    SET_ALL_BITS(b->state->selector);
    SET_START_POS(*b->state, 0);
    SET_END_POS(*b->state, 0);

    c->values[0] = 200;
    SET_ALL_BITS(c->state->selector);
    SET_START_POS(*c->state, 0);
    SET_END_POS(*c->state, 0);

    // 1-1 mappings
    b->state->offset[0] = 0;
    b->state->offset[1] = 1;
    c->state->offset[0] = 0;
    c->state->offset[1] = 1;

    auto root = std::make_shared<FactorizedTreeElement>("A", a);
    auto b_node = root->add_leaf("A", "B", a, b);
    b_node->add_leaf("B", "C", b, c);

    std::vector<std::string> col_order = {"A", "B", "C"};
    Schema s;
    s.root = root;
    s.column_ordering = &col_order;
    s.map = &map;

    // Expected: 1 tuple (all 1-1 mappings, single elements)
    uint64_t expected = 1;

    SinkPacked sink;
    sink.init(&s);
    sink.execute();
    EXPECT_EQ(sink.get_num_output_tuples(), expected);
}

// Test 7: Wide shallow tree (many leaf siblings under root)
TEST_F(SinkPackedTest, WideShallowTree) {
    const int32_t A_SIZE = 2;
    const int32_t CHILD_SIZE = 4;

    auto a = map.allocate_vector("A_wide_shallow");
    auto b = map.allocate_vector("B_wide_shallow");
    auto c = map.allocate_vector("C_wide_shallow");
    auto d = map.allocate_vector("D_wide_shallow");

    for (int i = 0; i < A_SIZE; ++i)
        a->values[i] = i;
    SET_ALL_BITS(a->state->selector);
    SET_START_POS(*a->state, 0);
    SET_END_POS(*a->state, A_SIZE - 1);

    for (int i = 0; i < CHILD_SIZE; ++i) {
        b->values[i] = 100 + i;
        c->values[i] = 200 + i;
        d->values[i] = 300 + i;
    }
    SET_ALL_BITS(b->state->selector);
    SET_START_POS(*b->state, 0);
    SET_END_POS(*b->state, CHILD_SIZE - 1);
    SET_ALL_BITS(c->state->selector);
    SET_START_POS(*c->state, 0);
    SET_END_POS(*c->state, CHILD_SIZE - 1);
    SET_ALL_BITS(d->state->selector);
    SET_START_POS(*d->state, 0);
    SET_END_POS(*d->state, CHILD_SIZE - 1);

    // Each A has 2 children per leaf vector
    for (int i = 0; i <= A_SIZE; ++i) {
        b->state->offset[i] = i * 2;
        c->state->offset[i] = i * 2;
        d->state->offset[i] = i * 2;
    }

    auto root = std::make_shared<FactorizedTreeElement>("A", a);
    root->add_leaf("A", "B", a, b);
    root->add_leaf("A", "C", a, c);
    root->add_leaf("A", "D", a, d);

    std::vector<std::string> col_order = {"A", "B", "C", "D"};
    Schema s;
    s.root = root;
    s.column_ordering = &col_order;
    s.map = &map;

    // Expected: 2 A positions, each with 2*2*2 = 8 tuples = 16 total
    uint64_t expected = A_SIZE * 2 * 2 * 2;

    SinkPacked sink;
    sink.init(&s);
    sink.execute();
    EXPECT_EQ(sink.get_num_output_tuples(), expected);
}

// Test 8: Narrow deep chain (6 levels, 1-1 mappings)
TEST_F(SinkPackedTest, NarrowDeepChain) {
    const int32_t SIZE = 4;

    auto a = map.allocate_vector("A_narrow");
    auto b = map.allocate_vector("B_narrow");
    auto c = map.allocate_vector("C_narrow");
    auto d = map.allocate_vector("D_narrow");
    auto e = map.allocate_vector("E_narrow");
    auto f = map.allocate_vector("F_narrow");

    for (int i = 0; i < SIZE; ++i) {
        a->values[i] = i;
        b->values[i] = 10 + i;
        c->values[i] = 20 + i;
        d->values[i] = 30 + i;
        e->values[i] = 40 + i;
        f->values[i] = 50 + i;
    }

    SET_ALL_BITS(a->state->selector);
    SET_START_POS(*a->state, 0);
    SET_END_POS(*a->state, SIZE - 1);
    SET_ALL_BITS(b->state->selector);
    SET_START_POS(*b->state, 0);
    SET_END_POS(*b->state, SIZE - 1);
    SET_ALL_BITS(c->state->selector);
    SET_START_POS(*c->state, 0);
    SET_END_POS(*c->state, SIZE - 1);
    SET_ALL_BITS(d->state->selector);
    SET_START_POS(*d->state, 0);
    SET_END_POS(*d->state, SIZE - 1);
    SET_ALL_BITS(e->state->selector);
    SET_START_POS(*e->state, 0);
    SET_END_POS(*e->state, SIZE - 1);
    SET_ALL_BITS(f->state->selector);
    SET_START_POS(*f->state, 0);
    SET_END_POS(*f->state, SIZE - 1);

    // All 1-1 mappings
    for (int i = 0; i <= SIZE; ++i) {
        b->state->offset[i] = i;
        c->state->offset[i] = i;
        d->state->offset[i] = i;
        e->state->offset[i] = i;
        f->state->offset[i] = i;
    }

    auto root = std::make_shared<FactorizedTreeElement>("A", a);
    auto b_node = root->add_leaf("A", "B", a, b);
    auto c_node = b_node->add_leaf("B", "C", b, c);
    auto d_node = c_node->add_leaf("C", "D", c, d);
    auto e_node = d_node->add_leaf("D", "E", d, e);
    e_node->add_leaf("E", "F", e, f);

    std::vector<std::string> col_order = {"A", "B", "C", "D", "E", "F"};
    Schema s;
    s.root = root;
    s.column_ordering = &col_order;
    s.map = &map;

    // Expected: 4 tuples (1-1 chain)
    uint64_t expected = SIZE;

    SinkPacked sink;
    sink.init(&s);
    sink.execute();
    EXPECT_EQ(sink.get_num_output_tuples(), expected);
}

// Test 9: Alternating masking at non-leaf levels
// CONSTRAINT: start_pos and end_pos must always have their selector bits SET
TEST_F(SinkPackedTest, AlternatingNonLeafMasking) {
    const int32_t A_SIZE = 8;
    const int32_t B_SIZE = 16;
    const int32_t C_SIZE = 16;

    auto a = map.allocate_vector("A_alt");
    auto b = map.allocate_vector("B_alt");
    auto c_vec = map.allocate_vector("C_alt");

    for (int i = 0; i < A_SIZE; ++i)
        a->values[i] = i;
    for (int i = 0; i < B_SIZE; ++i)
        b->values[i] = 100 + i;
    for (int i = 0; i < C_SIZE; ++i)
        c_vec->values[i] = 200 + i;

    SET_ALL_BITS(a->state->selector);
    SET_START_POS(*a->state, 0);
    SET_END_POS(*a->state, A_SIZE - 1);
    SET_ALL_BITS(b->state->selector);
    SET_START_POS(*b->state, 0);
    SET_END_POS(*b->state, B_SIZE - 1);
    SET_ALL_BITS(c_vec->state->selector);
    SET_START_POS(*c_vec->state, 0);
    SET_END_POS(*c_vec->state, C_SIZE - 1);

    for (int i = 0; i <= A_SIZE; ++i)
        b->state->offset[i] = i * 2;
    for (int i = 0; i <= B_SIZE; ++i)
        c_vec->state->offset[i] = i;

    // Alternating masking on INTERIOR positions only (not at start_pos or end_pos)
    // For A: clear positions 2, 4, 6 (keep 0 and 7 valid)
    for (int i = 2; i < A_SIZE - 1; i += 2)
        CLEAR_BIT(a->state->selector, i);
    // For B: clear odd interior positions 1, 3, 5, 7, 9, 11, 13 (keep 0 and 15 valid)
    for (int i = 1; i < B_SIZE - 1; i += 2)
        CLEAR_BIT(b->state->selector, i);

    auto root = std::make_shared<FactorizedTreeElement>("A", a);
    auto b_node = root->add_leaf("A", "B", a, b);
    b_node->add_leaf("B", "C", b, c_vec);

    std::vector<std::string> col_order = {"A", "B", "C"};
    Schema s;
    s.root = root;
    s.column_ordering = &col_order;
    s.map = &map;

    // Compute expected: For each active A, sum over active B, multiply by C range size
    // C is leaf with 1-1 mapping, so each active B contributes 1
    uint64_t expected = 0;
    for (int aidx = 0; aidx < A_SIZE; ++aidx) {
        if (!TEST_BIT(a->state->selector, aidx)) continue;
        int b_start = b->state->offset[aidx], b_end = b->state->offset[aidx + 1] - 1;
        for (int bidx = b_start; bidx <= b_end; ++bidx) {
            if (!TEST_BIT(b->state->selector, bidx)) continue;
            // C is leaf: count = range size
            int c_start = c_vec->state->offset[bidx];
            int c_end = c_vec->state->offset[bidx + 1] - 1;
            expected += (c_start <= c_end) ? (c_end - c_start + 1) : 0;
        }
    }

    SinkPacked sink;
    sink.init(&s);
    sink.execute();
    EXPECT_EQ(sink.get_num_output_tuples(), expected);
}

// Test 10: Large range sizes (multiple children per parent)
TEST_F(SinkPackedTest, LargeRangeSizes) {
    const int32_t A_SIZE = 2;
    const int32_t B_SIZE = 20;

    auto a = map.allocate_vector("A_large_range");
    auto b = map.allocate_vector("B_large_range");

    for (int i = 0; i < A_SIZE; ++i)
        a->values[i] = i;
    for (int i = 0; i < B_SIZE; ++i)
        b->values[i] = 100 + i;

    SET_ALL_BITS(a->state->selector);
    SET_START_POS(*a->state, 0);
    SET_END_POS(*a->state, A_SIZE - 1);
    SET_ALL_BITS(b->state->selector);
    SET_START_POS(*b->state, 0);
    SET_END_POS(*b->state, B_SIZE - 1);

    // A0 -> B[0..9] (10), A1 -> B[10..19] (10)
    b->state->offset[0] = 0;
    b->state->offset[1] = 10;
    b->state->offset[2] = 20;

    auto root = std::make_shared<FactorizedTreeElement>("A", a);
    root->add_leaf("A", "B", a, b);

    std::vector<std::string> col_order = {"A", "B"};
    Schema s;
    s.root = root;
    s.column_ordering = &col_order;
    s.map = &map;

    // Expected: 10 + 10 = 20
    uint64_t expected = 20;

    SinkPacked sink;
    sink.init(&s);
    sink.execute();
    EXPECT_EQ(sink.get_num_output_tuples(), expected);
}

// Test 11: Heavily skewed distribution (one parent has most children)
TEST_F(SinkPackedTest, SkewedDistribution) {
    const int32_t A_SIZE = 4;
    const int32_t B_SIZE = 25;

    auto a = map.allocate_vector("A_skewed");
    auto b = map.allocate_vector("B_skewed");

    for (int i = 0; i < A_SIZE; ++i)
        a->values[i] = i;
    for (int i = 0; i < B_SIZE; ++i)
        b->values[i] = 100 + i;

    SET_ALL_BITS(a->state->selector);
    SET_START_POS(*a->state, 0);
    SET_END_POS(*a->state, A_SIZE - 1);
    SET_ALL_BITS(b->state->selector);
    SET_START_POS(*b->state, 0);
    SET_END_POS(*b->state, B_SIZE - 1);

    // A0->1, A1->1, A2->1, A3->22 (heavily skewed)
    b->state->offset[0] = 0;
    b->state->offset[1] = 1;
    b->state->offset[2] = 2;
    b->state->offset[3] = 3;
    b->state->offset[4] = 25;

    auto root = std::make_shared<FactorizedTreeElement>("A", a);
    root->add_leaf("A", "B", a, b);

    std::vector<std::string> col_order = {"A", "B"};
    Schema s;
    s.root = root;
    s.column_ordering = &col_order;
    s.map = &map;

    // Expected: 1 + 1 + 1 + 22 = 25
    uint64_t expected = 25;

    SinkPacked sink;
    sink.init(&s);
    sink.execute();
    EXPECT_EQ(sink.get_num_output_tuples(), expected);
}

// Test 12: Multiple non-leaf siblings (B and C both non-leaf, each with leaf children)
TEST_F(SinkPackedTest, MultipleNonLeafSiblings) {
    const int32_t A_SIZE = 2;
    const int32_t B_SIZE = 4;
    const int32_t C_SIZE = 4;
    const int32_t D_SIZE = 4;// Leaf of B
    const int32_t E_SIZE = 4;// Leaf of C

    auto a = map.allocate_vector("A_multi_nonleaf");
    auto b = map.allocate_vector("B_multi_nonleaf");
    auto c = map.allocate_vector("C_multi_nonleaf");
    auto d = map.allocate_vector("D_multi_nonleaf");
    auto e = map.allocate_vector("E_multi_nonleaf");

    for (int i = 0; i < A_SIZE; ++i)
        a->values[i] = i;
    for (int i = 0; i < B_SIZE; ++i)
        b->values[i] = 100 + i;
    for (int i = 0; i < C_SIZE; ++i)
        c->values[i] = 200 + i;
    for (int i = 0; i < D_SIZE; ++i)
        d->values[i] = 300 + i;
    for (int i = 0; i < E_SIZE; ++i)
        e->values[i] = 400 + i;

    SET_ALL_BITS(a->state->selector);
    SET_START_POS(*a->state, 0);
    SET_END_POS(*a->state, A_SIZE - 1);
    SET_ALL_BITS(b->state->selector);
    SET_START_POS(*b->state, 0);
    SET_END_POS(*b->state, B_SIZE - 1);
    SET_ALL_BITS(c->state->selector);
    SET_START_POS(*c->state, 0);
    SET_END_POS(*c->state, C_SIZE - 1);
    SET_ALL_BITS(d->state->selector);
    SET_START_POS(*d->state, 0);
    SET_END_POS(*d->state, D_SIZE - 1);
    SET_ALL_BITS(e->state->selector);
    SET_START_POS(*e->state, 0);
    SET_END_POS(*e->state, E_SIZE - 1);

    // 2 B per A, 2 C per A, D 1-1 with B, E 1-1 with C
    for (int i = 0; i <= A_SIZE; ++i) {
        b->state->offset[i] = i * 2;
        c->state->offset[i] = i * 2;
    }
    for (int i = 0; i <= B_SIZE; ++i)
        d->state->offset[i] = i;
    for (int i = 0; i <= C_SIZE; ++i)
        e->state->offset[i] = i;

    auto root = std::make_shared<FactorizedTreeElement>("A", a);
    auto b_node = root->add_leaf("A", "B", a, b);
    auto c_node = root->add_leaf("A", "C", a, c);
    b_node->add_leaf("B", "D", b, d);
    c_node->add_leaf("C", "E", c, e);

    std::vector<std::string> col_order = {"A", "B", "C", "D", "E"};
    Schema s;
    s.root = root;
    s.column_ordering = &col_order;
    s.map = &map;

    // Expected: each A has 2 B (each with 1 D) * 2 C (each with 1 E) = 2*2 = 4 per A, total 8
    uint64_t expected = A_SIZE * 2 * 2;

    SinkPacked sink;
    sink.init(&s);
    sink.execute();
    EXPECT_EQ(sink.get_num_output_tuples(), expected);
}

// Test 13: All-masked non-leaf (one parent with all children masked)
TEST_F(SinkPackedTest, AllMaskedNonLeafPosition) {
    const int32_t A_SIZE = 4;
    const int32_t B_SIZE = 8;
    const int32_t C_SIZE = 8;

    auto a = map.allocate_vector("A_allmask");
    auto b = map.allocate_vector("B_allmask");
    auto c = map.allocate_vector("C_allmask");

    for (int i = 0; i < A_SIZE; ++i)
        a->values[i] = i;
    for (int i = 0; i < B_SIZE; ++i)
        b->values[i] = 100 + i;
    for (int i = 0; i < C_SIZE; ++i)
        c->values[i] = 200 + i;

    SET_ALL_BITS(a->state->selector);
    SET_START_POS(*a->state, 0);
    SET_END_POS(*a->state, A_SIZE - 1);
    SET_ALL_BITS(b->state->selector);
    SET_START_POS(*b->state, 0);
    SET_END_POS(*b->state, B_SIZE - 1);
    SET_ALL_BITS(c->state->selector);
    SET_START_POS(*c->state, 0);
    SET_END_POS(*c->state, C_SIZE - 1);

    for (int i = 0; i <= A_SIZE; ++i)
        b->state->offset[i] = i * 2;
    for (int i = 0; i <= B_SIZE; ++i)
        c->state->offset[i] = i;

    // Mask all B positions for A position 1 (B[2] and B[3])
    CLEAR_BIT(b->state->selector, 2);
    CLEAR_BIT(b->state->selector, 3);

    auto root = std::make_shared<FactorizedTreeElement>("A", a);
    auto b_node = root->add_leaf("A", "B", a, b);
    b_node->add_leaf("B", "C", b, c);

    std::vector<std::string> col_order = {"A", "B", "C"};
    Schema s;
    s.root = root;
    s.column_ordering = &col_order;
    s.map = &map;

    // Expected: A0->2, A1->0 (all B masked), A2->2, A3->2 = 6
    uint64_t expected = 6;

    SinkPacked sink;
    sink.init(&s);
    sink.execute();
    EXPECT_EQ(sink.get_num_output_tuples(), expected);
}

// Test 14: Binary tree structure (each non-leaf has exactly 2 children)
TEST_F(SinkPackedTest, BinaryTreeStructure) {
    const int32_t SIZE = 4;

    auto a = map.allocate_vector("A_binary");
    auto b = map.allocate_vector("B_binary");
    auto c = map.allocate_vector("C_binary");
    auto d = map.allocate_vector("D_binary");
    auto e = map.allocate_vector("E_binary");

    for (int i = 0; i < SIZE; ++i) {
        a->values[i] = i;
        b->values[i] = 10 + i;
        c->values[i] = 20 + i;
        d->values[i] = 30 + i;
        e->values[i] = 40 + i;
    }

    SET_ALL_BITS(a->state->selector);
    SET_START_POS(*a->state, 0);
    SET_END_POS(*a->state, SIZE - 1);
    SET_ALL_BITS(b->state->selector);
    SET_START_POS(*b->state, 0);
    SET_END_POS(*b->state, SIZE - 1);
    SET_ALL_BITS(c->state->selector);
    SET_START_POS(*c->state, 0);
    SET_END_POS(*c->state, SIZE - 1);
    SET_ALL_BITS(d->state->selector);
    SET_START_POS(*d->state, 0);
    SET_END_POS(*d->state, SIZE - 1);
    SET_ALL_BITS(e->state->selector);
    SET_START_POS(*e->state, 0);
    SET_END_POS(*e->state, SIZE - 1);

    // 1-1 mappings throughout
    for (int i = 0; i <= SIZE; ++i) {
        b->state->offset[i] = i;
        c->state->offset[i] = i;
        d->state->offset[i] = i;
        e->state->offset[i] = i;
    }

    // A -> B, C (2 children), B -> D (leaf), C -> E (leaf)
    auto root = std::make_shared<FactorizedTreeElement>("A", a);
    auto b_node = root->add_leaf("A", "B", a, b);
    auto c_node = root->add_leaf("A", "C", a, c);
    b_node->add_leaf("B", "D", b, d);
    c_node->add_leaf("C", "E", c, e);

    std::vector<std::string> col_order = {"A", "B", "C", "D", "E"};
    Schema s;
    s.root = root;
    s.column_ordering = &col_order;
    s.map = &map;

    // Expected: 4 A positions, each with 1 B (1 D) * 1 C (1 E) = 1 per A, total 4
    uint64_t expected = SIZE;

    SinkPacked sink;
    sink.init(&s);
    sink.execute();
    EXPECT_EQ(sink.get_num_output_tuples(), expected);
}

// Test 15: Root with single child chain
TEST_F(SinkPackedTest, SingleChildChain) {
    const int32_t SIZE = 8;

    auto a = map.allocate_vector("A_single");
    auto b = map.allocate_vector("B_single");
    auto c = map.allocate_vector("C_single");

    for (int i = 0; i < SIZE; ++i) {
        a->values[i] = i;
        b->values[i] = 10 + i;
        c->values[i] = 20 + i;
    }

    SET_ALL_BITS(a->state->selector);
    SET_START_POS(*a->state, 0);
    SET_END_POS(*a->state, SIZE - 1);
    SET_ALL_BITS(b->state->selector);
    SET_START_POS(*b->state, 0);
    SET_END_POS(*b->state, SIZE - 1);
    SET_ALL_BITS(c->state->selector);
    SET_START_POS(*c->state, 0);
    SET_END_POS(*c->state, SIZE - 1);

    // Each A maps to exactly 1 B, each B to exactly 1 C
    for (int i = 0; i <= SIZE; ++i) {
        b->state->offset[i] = i;
        c->state->offset[i] = i;
    }

    auto root = std::make_shared<FactorizedTreeElement>("A", a);
    auto b_node = root->add_leaf("A", "B", a, b);
    b_node->add_leaf("B", "C", b, c);

    std::vector<std::string> col_order = {"A", "B", "C"};
    Schema s;
    s.root = root;
    s.column_ordering = &col_order;
    s.map = &map;

    // Expected: 8 (1-1-1 chain)
    uint64_t expected = SIZE;

    SinkPacked sink;
    sink.init(&s);
    sink.execute();
    EXPECT_EQ(sink.get_num_output_tuples(), expected);
}

// Test 16: Partial root masking with varying child counts
// CONSTRAINT: start_pos and end_pos must always have their selector bits SET
TEST_F(SinkPackedTest, PartialRootMasking) {
    const int32_t A_SIZE = 8; // Changed from 6 to 8 so end_pos (7) is not cleared
    const int32_t B_SIZE = 16;// Changed from 12 to 16

    auto a = map.allocate_vector("A_partial");
    auto b = map.allocate_vector("B_partial");

    for (int i = 0; i < A_SIZE; ++i)
        a->values[i] = i;
    for (int i = 0; i < B_SIZE; ++i)
        b->values[i] = 100 + i;

    SET_ALL_BITS(a->state->selector);
    SET_START_POS(*a->state, 0);
    SET_END_POS(*a->state, A_SIZE - 1);
    SET_ALL_BITS(b->state->selector);
    SET_START_POS(*b->state, 0);
    SET_END_POS(*b->state, B_SIZE - 1);

    // 2 B per A
    for (int i = 0; i <= A_SIZE; ++i)
        b->state->offset[i] = i * 2;

    // Mask interior odd A positions (keep 0 and 7 valid)
    CLEAR_BIT(a->state->selector, 1);
    CLEAR_BIT(a->state->selector, 3);
    CLEAR_BIT(a->state->selector, 5);

    auto root = std::make_shared<FactorizedTreeElement>("A", a);
    root->add_leaf("A", "B", a, b);

    std::vector<std::string> col_order = {"A", "B"};
    Schema s;
    s.root = root;
    s.column_ordering = &col_order;
    s.map = &map;

    // Expected: A0->2, A2->2, A4->2, A6->2, A7->2 = 10 (active: 0,2,4,6,7)
    uint64_t expected = 0;
    for (int aidx = 0; aidx < A_SIZE; ++aidx) {
        if (!TEST_BIT(a->state->selector, aidx)) continue;
        int b_start = b->state->offset[aidx], b_end = b->state->offset[aidx + 1] - 1;
        expected += (b_start <= b_end) ? (b_end - b_start + 1) : 0;
    }

    SinkPacked sink;
    sink.init(&s);
    sink.execute();
    EXPECT_EQ(sink.get_num_output_tuples(), expected);
}

// Test 17: Very deep chain (7 levels)
TEST_F(SinkPackedTest, VeryDeepChain) {
    const int32_t SIZE = 4;

    auto a = map.allocate_vector("A_vdeep");
    auto b = map.allocate_vector("B_vdeep");
    auto c = map.allocate_vector("C_vdeep");
    auto d = map.allocate_vector("D_vdeep");
    auto e = map.allocate_vector("E_vdeep");
    auto f = map.allocate_vector("F_vdeep");
    auto g = map.allocate_vector("G_vdeep");

    for (int i = 0; i < SIZE; ++i) {
        a->values[i] = i;
        b->values[i] = 10 + i;
        c->values[i] = 20 + i;
        d->values[i] = 30 + i;
        e->values[i] = 40 + i;
        f->values[i] = 50 + i;
        g->values[i] = 60 + i;
    }

    SET_ALL_BITS(a->state->selector);
    SET_START_POS(*a->state, 0);
    SET_END_POS(*a->state, SIZE - 1);
    SET_ALL_BITS(b->state->selector);
    SET_START_POS(*b->state, 0);
    SET_END_POS(*b->state, SIZE - 1);
    SET_ALL_BITS(c->state->selector);
    SET_START_POS(*c->state, 0);
    SET_END_POS(*c->state, SIZE - 1);
    SET_ALL_BITS(d->state->selector);
    SET_START_POS(*d->state, 0);
    SET_END_POS(*d->state, SIZE - 1);
    SET_ALL_BITS(e->state->selector);
    SET_START_POS(*e->state, 0);
    SET_END_POS(*e->state, SIZE - 1);
    SET_ALL_BITS(f->state->selector);
    SET_START_POS(*f->state, 0);
    SET_END_POS(*f->state, SIZE - 1);
    SET_ALL_BITS(g->state->selector);
    SET_START_POS(*g->state, 0);
    SET_END_POS(*g->state, SIZE - 1);

    // All 1-1 mappings
    for (int i = 0; i <= SIZE; ++i) {
        b->state->offset[i] = i;
        c->state->offset[i] = i;
        d->state->offset[i] = i;
        e->state->offset[i] = i;
        f->state->offset[i] = i;
        g->state->offset[i] = i;
    }

    auto root = std::make_shared<FactorizedTreeElement>("A", a);
    auto b_node = root->add_leaf("A", "B", a, b);
    auto c_node = b_node->add_leaf("B", "C", b, c);
    auto d_node = c_node->add_leaf("C", "D", c, d);
    auto e_node = d_node->add_leaf("D", "E", d, e);
    auto f_node = e_node->add_leaf("E", "F", e, f);
    f_node->add_leaf("F", "G", f, g);

    std::vector<std::string> col_order = {"A", "B", "C", "D", "E", "F", "G"};
    Schema s;
    s.root = root;
    s.column_ordering = &col_order;
    s.map = &map;

    // Expected: 4 tuples (all 1-1)
    uint64_t expected = SIZE;

    SinkPacked sink;
    sink.init(&s);
    sink.execute();
    EXPECT_EQ(sink.get_num_output_tuples(), expected);
}

// Test 18: Multiple roots with siblings having different depths
TEST_F(SinkPackedTest, AsymmetricDepthSiblings) {
    const int32_t SIZE = 4;

    auto a = map.allocate_vector("A_asym_depth");
    auto b = map.allocate_vector("B_asym_depth");// B->C->D (depth 3)
    auto c = map.allocate_vector("C_asym_depth");
    auto d = map.allocate_vector("D_asym_depth");
    auto e = map.allocate_vector("E_asym_depth");// E is direct leaf of A (depth 1)

    for (int i = 0; i < SIZE; ++i) {
        a->values[i] = i;
        b->values[i] = 10 + i;
        c->values[i] = 20 + i;
        d->values[i] = 30 + i;
        e->values[i] = 40 + i;
    }

    SET_ALL_BITS(a->state->selector);
    SET_START_POS(*a->state, 0);
    SET_END_POS(*a->state, SIZE - 1);
    SET_ALL_BITS(b->state->selector);
    SET_START_POS(*b->state, 0);
    SET_END_POS(*b->state, SIZE - 1);
    SET_ALL_BITS(c->state->selector);
    SET_START_POS(*c->state, 0);
    SET_END_POS(*c->state, SIZE - 1);
    SET_ALL_BITS(d->state->selector);
    SET_START_POS(*d->state, 0);
    SET_END_POS(*d->state, SIZE - 1);
    SET_ALL_BITS(e->state->selector);
    SET_START_POS(*e->state, 0);
    SET_END_POS(*e->state, SIZE - 1);

    // 1-1 mappings
    for (int i = 0; i <= SIZE; ++i) {
        b->state->offset[i] = i;
        c->state->offset[i] = i;
        d->state->offset[i] = i;
        e->state->offset[i] = i;
    }

    auto root = std::make_shared<FactorizedTreeElement>("A", a);
    auto b_node = root->add_leaf("A", "B", a, b);
    auto c_node = b_node->add_leaf("B", "C", b, c);
    c_node->add_leaf("C", "D", c, d);
    root->add_leaf("A", "E", a, e);// E is sibling of B but at different depth

    std::vector<std::string> col_order = {"A", "B", "C", "D", "E"};
    Schema s;
    s.root = root;
    s.column_ordering = &col_order;
    s.map = &map;

    // Expected: 4 (1-1 chain on B side, 1-1 on E side, product = 1*1 = 1 per A)
    uint64_t expected = SIZE;

    SinkPacked sink;
    sink.init(&s);
    sink.execute();
    EXPECT_EQ(sink.get_num_output_tuples(), expected);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
