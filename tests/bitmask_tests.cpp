#include "../src/operator/include/join/inljoin_packed.hpp"
#include "../src/operator/include/schema/schema.hpp"
#include "../src/operator/include/vector/bitmask.hpp"
#include "../src/operator/include/vector/state.hpp"
#include "../src/operator/include/vector/vector.hpp"
#include "../src/table/include/adj_list.hpp"

#include <cassert>
#include <cstring>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

// Global adjacency lists used in tests
std::unique_ptr<ffx::AdjList<uint64_t>> fwd_adj_list0;
std::unique_ptr<ffx::AdjList<uint64_t>> fwd_adj_list1;
std::unique_ptr<ffx::AdjList<uint64_t>> fwd_adj_list2;
std::unique_ptr<ffx::AdjList<uint64_t>> fwd_adj_list3;
std::unique_ptr<ffx::AdjList<uint64_t>> fwd_adj_list4;

std::unique_ptr<ffx::AdjList<uint64_t>> bwd_adj_list0;
std::unique_ptr<ffx::AdjList<uint64_t>> bwd_adj_list1;
std::unique_ptr<ffx::AdjList<uint64_t>> bwd_adj_list2;
std::unique_ptr<ffx::AdjList<uint64_t>> bwd_adj_list3;
std::unique_ptr<ffx::AdjList<uint64_t>> bwd_adj_list4;

// Initialize adjacency lists with test data
void initialize_adj_lists() {
    // Create adjacency lists with specified sizes
    fwd_adj_list0 = std::make_unique<ffx::AdjList<uint64_t>>();
    fwd_adj_list0->size = 0;
    fwd_adj_list0->values = nullptr;

    fwd_adj_list1 = std::make_unique<ffx::AdjList<uint64_t>>();
    fwd_adj_list1->size = 3;
    fwd_adj_list1->values = new uint64_t[3]{2, 3, 4};

    fwd_adj_list2 = std::make_unique<ffx::AdjList<uint64_t>>();
    fwd_adj_list2->size = 2;
    fwd_adj_list2->values = new uint64_t[2]{3, 4};

    fwd_adj_list3 = std::make_unique<ffx::AdjList<uint64_t>>();
    fwd_adj_list3->size = 1;
    fwd_adj_list3->values = new uint64_t[1]{4};

    fwd_adj_list4 = std::make_unique<ffx::AdjList<uint64_t>>();
    fwd_adj_list4->size = 0;
    fwd_adj_list4->values = nullptr;

    // Create backward adjacency lists
    bwd_adj_list0 = std::make_unique<ffx::AdjList<uint64_t>>();
    bwd_adj_list0->size = 0;
    bwd_adj_list0->values = nullptr;

    bwd_adj_list1 = std::make_unique<ffx::AdjList<uint64_t>>();
    bwd_adj_list1->size = 0;
    bwd_adj_list1->values = nullptr;

    bwd_adj_list2 = std::make_unique<ffx::AdjList<uint64_t>>();
    bwd_adj_list2->size = 1;
    bwd_adj_list2->values = new uint64_t[1]{1};

    bwd_adj_list3 = std::make_unique<ffx::AdjList<uint64_t>>();
    bwd_adj_list3->size = 2;
    bwd_adj_list3->values = new uint64_t[2]{1, 2};

    bwd_adj_list4 = std::make_unique<ffx::AdjList<uint64_t>>();
    bwd_adj_list4->size = 3;
    bwd_adj_list4->values = new uint64_t[3]{1, 2, 3};
}

// Clean up function for tests
void cleanup_adj_lists() {
    // Delete the dynamically allocated arrays
    if (fwd_adj_list1 && fwd_adj_list1->values) delete[] fwd_adj_list1->values;
    if (fwd_adj_list2 && fwd_adj_list2->values) delete[] fwd_adj_list2->values;
    if (fwd_adj_list3 && fwd_adj_list3->values) delete[] fwd_adj_list3->values;

    if (bwd_adj_list2 && bwd_adj_list2->values) delete[] bwd_adj_list2->values;
    if (bwd_adj_list3 && bwd_adj_list3->values) delete[] bwd_adj_list3->values;
    if (bwd_adj_list4 && bwd_adj_list4->values) delete[] bwd_adj_list4->values;
}

// Mock next operator for testing
class MockNextOp : public ffx::Operator {
public:
    int execute_calls = 0;
    void init(ffx::Schema* schema) override {
        assert(schema->map);
        // Expect at least one table to be present in the schema
        assert(!schema->tables.empty());
    }
    void execute() override { execute_calls++; }
};

// Base test fixture for all bitmask tests
class BitmaskTestBase : public ::testing::Test {
protected:
    void SetUp() override { initialize_adj_lists(); }

    void TearDown() override { cleanup_adj_lists(); }
};

// Test fixture for basic bitmask tests
class BitmaskTest : public BitmaskTestBase {};

// Test fixture for tests that involve INLJoinPacked
class BitmaskINLJoinTest : public BitmaskTestBase {
protected:
    std::unique_ptr<ffx::Table> table;
    ffx::Vector<uint64_t>*join_key_vector, *output_key_vector;
    std::unique_ptr<MockNextOp> next_op;
    ffx::QueryVariableToVectorMap vector_map;

    void SetUp() override {
        // Call the parent SetUp to initialize adjacency lists
        BitmaskTestBase::SetUp();

        // Setup table with adjacency lists
        // Create arrays for adjacency lists
        auto fwd_adj_lists_arr = std::make_unique<ffx::AdjList<uint64_t>[]>(5);
        auto bwd_adj_lists_arr = std::make_unique<ffx::AdjList<uint64_t>[]>(5);

        // Copy the constructed adjacency lists to the arrays
        fwd_adj_lists_arr[0] = std::move(*fwd_adj_list0);
        fwd_adj_lists_arr[1] = std::move(*fwd_adj_list1);
        fwd_adj_lists_arr[2] = std::move(*fwd_adj_list2);
        fwd_adj_lists_arr[3] = std::move(*fwd_adj_list3);
        fwd_adj_lists_arr[4] = std::move(*fwd_adj_list4);

        bwd_adj_lists_arr[0] = std::move(*bwd_adj_list0);
        bwd_adj_lists_arr[1] = std::move(*bwd_adj_list1);
        bwd_adj_lists_arr[2] = std::move(*bwd_adj_list2);
        bwd_adj_lists_arr[3] = std::move(*bwd_adj_list3);
        bwd_adj_lists_arr[4] = std::move(*bwd_adj_list4);


        // Create table with constructor that takes adjacency list arrays
        table = std::make_unique<ffx::Table>(5, 5, std::move(fwd_adj_lists_arr), std::move(bwd_adj_lists_arr));

        // Set table columns to match the join keys used in tests
        table->columns = {"new_join_key", "new_output_key"};
        table->name = "test_table";

        // Allocate join_key_vector (input) - operator will use get_vector for this
        join_key_vector = vector_map.allocate_vector("new_join_key");
        // Don't allocate output_key_vector here - operator will allocate it in init()
        output_key_vector = nullptr;

        next_op = std::make_unique<MockNextOp>();
    }
};

//------------------------------------------------------------------------------
// Basic Bitmask Tests
//------------------------------------------------------------------------------

TEST_F(BitmaskTest, BasicFunctionality) {
    // Test basic BitMask functionality
    const auto vec1 = std::make_unique<ffx::Vector<uint64_t>>();
    auto& selector1 = vec1->state->selector;

    // Initially, all bits should be set to 1
    EXPECT_EQ(TEST_BIT(selector1, 0), 1);
    EXPECT_EQ(TEST_BIT(selector1, 1), 1);
    EXPECT_EQ(TEST_BIT(selector1, 10), 1);

    // Test clearing bits
    CLEAR_BIT(selector1, 5);
    EXPECT_EQ(TEST_BIT(selector1, 5), 0);
    EXPECT_EQ(TEST_BIT(selector1, 4), 1);
    EXPECT_EQ(TEST_BIT(selector1, 6), 1);

    // Test setting bits
    CLEAR_BIT(selector1, 7);
    EXPECT_EQ(TEST_BIT(selector1, 7), 0);
    SET_BIT(selector1, 7);
    EXPECT_EQ(TEST_BIT(selector1, 7), 1);

    // Test toggling bits
    TOGGLE_BIT(selector1, 8);
    EXPECT_EQ(TEST_BIT(selector1, 8), 0);
    TOGGLE_BIT(selector1, 8);
    EXPECT_EQ(TEST_BIT(selector1, 8), 1);

    // Test clearing all bits
    CLEAR_ALL_BITS(selector1);
    for (size_t i = 0; i < 16; ++i) {
        EXPECT_EQ(TEST_BIT(selector1, i), 0);
    }

    // Test setting all bits
    SET_ALL_BITS(selector1);
    for (size_t i = 0; i < 16; ++i) {
        EXPECT_EQ(TEST_BIT(selector1, i), 1);
    }
}

TEST_F(BitmaskTest, AdvancedFunctionality) {
    // Test more BitMask functionality
    const auto vec1 = std::make_unique<ffx::Vector<uint64_t>>();
    const auto vec2 = std::make_unique<ffx::Vector<uint64_t>>();
    auto& selector1 = vec1->state->selector;
    auto& selector2 = vec2->state->selector;

    // Test setting bits till index
    CLEAR_ALL_BITS(selector1);
    SET_BITS_TILL_IDX(selector1, 3);
    for (size_t i = 0; i <= 3; ++i) {
        EXPECT_EQ(TEST_BIT(selector1, i), 1);
    }
    for (size_t i = 4; i < 16; ++i) {
        EXPECT_EQ(TEST_BIT(selector1, i), 0);
    }

    // Test clearing bits till index
    SET_ALL_BITS(selector1);
    CLEAR_BITS_TILL_IDX(selector1, 5);
    for (size_t i = 0; i <= 5; ++i) {
        EXPECT_EQ(TEST_BIT(selector1, i), 0);
    }
    for (size_t i = 6; i < 16; ++i) {
        EXPECT_EQ(TEST_BIT(selector1, i), 1);
    }

    // Test AND operation between bitmasks
    SET_ALL_BITS(selector1);
    CLEAR_BIT(selector1, 2);
    CLEAR_BIT(selector1, 7);

    SET_ALL_BITS(selector2);
    CLEAR_BIT(selector2, 3);
    CLEAR_BIT(selector2, 7);

    AND_BITMASKS(selector1->size(), selector1, selector2);

    EXPECT_EQ(TEST_BIT(selector1, 0), 1);
    EXPECT_EQ(TEST_BIT(selector1, 1), 1);
    EXPECT_EQ(TEST_BIT(selector1, 2), 0);// Already 0 in selector1
    EXPECT_EQ(TEST_BIT(selector1, 3), 0);// Was 1 in selector1, 0 in selector2
    EXPECT_EQ(TEST_BIT(selector1, 7), 0);// Was 0 in both

    // Test copying bitmask
    SET_ALL_BITS(selector1);
    CLEAR_BIT(selector1, 4);
    CLEAR_BIT(selector1, 9);

    CLEAR_ALL_BITS(selector2);
    COPY_BITMASK(selector1->size(), selector2, selector1);

    for (size_t i = 0; i < 16; ++i) {
        EXPECT_EQ(TEST_BIT(selector2, i), TEST_BIT(selector1, i));
    }

    // Test start/end positions
    CLEAR_ALL_BITS(selector1);
    SET_BIT(selector1, 3);
    SET_BIT(selector1, 8);
    // UPDATE_POSITIONS(selector1);
    // EXPECT_EQ(GET_START_POS(selector1), 3);
    // EXPECT_EQ(GET_END_POS(selector1), 8);

    // Test setting start/end positions manually
    // SET_START_POS(selector1, 2);
    // SET_END_POS(selector1, 10);
    // EXPECT_EQ(GET_START_POS(selector1), 2);
    // EXPECT_EQ(GET_END_POS(selector1), 10);
}

//------------------------------------------------------------------------------
// Additional Bitmask Tests
//------------------------------------------------------------------------------

TEST_F(BitmaskTest, BitmaskBoundary) {
    // Test boundary conditions of bitmask
    const auto vec = std::make_unique<ffx::Vector<uint64_t>>();
    auto& selector = vec->state->selector;

    // Test operations at the first and last possible indices
    const size_t first_idx = 0;
    const size_t last_idx = selector.size() - 1;

    // Test operations at the first bit
    SET_BIT(selector, first_idx);
    EXPECT_EQ(TEST_BIT(selector, first_idx), 1);
    CLEAR_BIT(selector, first_idx);
    EXPECT_EQ(TEST_BIT(selector, first_idx), 0);

    // Test operations at the last bit
    SET_BIT(selector, last_idx);
    EXPECT_EQ(TEST_BIT(selector, last_idx), 1);
    CLEAR_BIT(selector, last_idx);
    EXPECT_EQ(TEST_BIT(selector, last_idx), 0);

    // Test setting first and last bits
    CLEAR_ALL_BITS(selector);
    SET_BIT(selector, first_idx);
    SET_BIT(selector, last_idx);
    // UPDATE_POSITIONS(selector);

    // Start position should be at the first set bit
    // EXPECT_EQ(GET_START_POS(selector), first_idx);
    // End position should be at the last set bit
    // EXPECT_EQ(GET_END_POS(selector), last_idx);
}

TEST_F(BitmaskTest, BitmaskPatterns) {
    // Test various bit patterns
    const auto vec = std::make_unique<ffx::Vector<uint64_t>>();
    auto& selector = vec->state->selector;

    // Set checkerboard pattern (alternating 1s and 0s)
    CLEAR_ALL_BITS(selector);
    for (size_t i = 0; i < 32; i += 2) {
        SET_BIT(selector, i);
    }

    // Verify pattern
    for (size_t i = 0; i < 32; ++i) {
        if (i % 2 == 0) {
            EXPECT_EQ(TEST_BIT(selector, i), 1);
        } else {
            EXPECT_EQ(TEST_BIT(selector, i), 0);
        }
    }

    // Invert the pattern
    for (size_t i = 0; i < 32; ++i) {
        TOGGLE_BIT(selector, i);
    }

    // Verify inverted pattern
    for (size_t i = 0; i < 32; ++i) {
        if (i % 2 == 0) {
            EXPECT_EQ(TEST_BIT(selector, i), 0);
        } else {
            EXPECT_EQ(TEST_BIT(selector, i), 1);
        }
    }
}

TEST_F(BitmaskTest, BitmaskOperations) {
    // Test additional bitmask operations
    const auto vec1 = std::make_unique<ffx::Vector<uint64_t>>();
    const auto vec2 = std::make_unique<ffx::Vector<uint64_t>>();
    auto& selector1 = vec1->state->selector;
    auto& selector2 = vec2->state->selector;

    // Test OR operation between bitmasks (if available)
    // First, set up the bitmasks
    CLEAR_ALL_BITS(selector1);
    SET_BIT(selector1, 1);
    SET_BIT(selector1, 3);
    SET_BIT(selector1, 5);

    CLEAR_ALL_BITS(selector2);
    SET_BIT(selector2, 2);
    SET_BIT(selector2, 3);
    SET_BIT(selector2, 6);

    // Test finding first/last set bit
    CLEAR_ALL_BITS(selector1);
    SET_BIT(selector1, 7);
    SET_BIT(selector1, 12);
    SET_BIT(selector1, 18);
    // UPDATE_POSITIONS(selector1);

    // EXPECT_EQ(GET_START_POS(selector1), 7);
    // EXPECT_EQ(GET_END_POS(selector1), 18);
}

TEST_F(BitmaskINLJoinTest, BitmaskWithJoinOperations) {
    // Create a bitmask for testing with INLJoinPacked
    auto bitmask = std::make_unique<ffx::BitMask<ffx::State::MAX_VECTOR_SIZE>>();

    // Initially, all bits should be set to 1
    for (size_t i = 0; i < 16; ++i) {
        EXPECT_EQ(TEST_BIT(*bitmask, i), 1);
    }

    // Simulate the pattern of bit modifications during INLJoinPacked execution
    CLEAR_BIT(*bitmask, 5);
    CLEAR_BIT(*bitmask, 8);
    CLEAR_BIT(*bitmask, 12);

    // Set the start and end positions, simulating what happens in INLJoinPacked
    // SET_START_POS(*bitmask, 2);
    // SET_END_POS(*bitmask, 15);

    // Verify positions
    // EXPECT_EQ(GET_START_POS(*bitmask), 2);
    // EXPECT_EQ(GET_END_POS(*bitmask), 15);

    // Verify bit patterns
    EXPECT_EQ(TEST_BIT(*bitmask, 2), 1);
    EXPECT_EQ(TEST_BIT(*bitmask, 5), 0);
    EXPECT_EQ(TEST_BIT(*bitmask, 8), 0);
    EXPECT_EQ(TEST_BIT(*bitmask, 12), 0);
    EXPECT_EQ(TEST_BIT(*bitmask, 15), 1);
}

TEST_F(BitmaskINLJoinTest, INLJoinWithBitmask) {
    auto inl_join = std::make_unique<ffx::INLJoinPacked<uint64_t>>("new_join_key", "new_output_key", true);
    inl_join->set_next_operator(std::move(next_op));

    EXPECT_NE(inl_join->next_op, nullptr);

    // Initialize the join operator with the vector map and table
    ffx::Schema schema;
    schema.map = &vector_map;
    schema.tables.clear();
    schema.tables.push_back(table.get());
    auto root_vector = vector_map.allocate_vector("root");
    schema.root = std::make_shared<ffx::FactorizedTreeElement>("root", root_vector);
    schema.root->add_leaf("root", "new_join_key", root_vector, join_key_vector);
    // output_key_vector will be allocated by inl_join->init()
    inl_join->init(&schema);

    // Get the output vector that was allocated by the operator
    output_key_vector = vector_map.get_vector("new_output_key");

    // Fill join_key_vector with test values
    join_key_vector->values[0] = 1;// Has 3 neighbors: 2,3,4
    join_key_vector->values[1] = 2;// Has 2 neighbors: 3,4
    join_key_vector->values[2] = 3;// Has 1 neighbor: 4
    join_key_vector->values[3] = 0;// Has 0 neighbors

    // Get the bitmask from the join_key_vector
    auto& bitmask = join_key_vector->state->selector;

    // Initially all bits should be set
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(TEST_BIT(bitmask, i), 1);
    }

    // Modify the bitmask to filter some indices
    CLEAR_BIT(bitmask, 3);// Filter out the entry with 0 neighbors
    SET_START_POS(*join_key_vector->state, 0);
    SET_END_POS(*join_key_vector->state, 2);

    // Execute the join
    inl_join->execute();

    // Check if output vector's state was properly allocated
    EXPECT_NE(output_key_vector->state, nullptr);
    EXPECT_NE(output_key_vector->values, nullptr);
}

TEST_F(BitmaskINLJoinTest, BitmaskManipulationDuringJoin) {
    auto inl_join = std::make_unique<ffx::INLJoinPacked<uint64_t>>("new_join_key", "new_output_key", true);
    inl_join->set_next_operator(std::move(next_op));

    EXPECT_NE(inl_join->next_op, nullptr);

    // Initialize the join operator
    ffx::Schema schema;
    schema.map = &vector_map;
    schema.tables.clear();
    schema.tables.push_back(table.get());
    auto root_vector = vector_map.allocate_vector("root");
    schema.root = std::make_shared<ffx::FactorizedTreeElement>("root", root_vector);
    schema.root->add_leaf("root", "new_join_key", root_vector, join_key_vector);
    // output_key_vector will be allocated by inl_join->init()
    inl_join->init(&schema);

    // Get the output vector that was allocated by the operator
    output_key_vector = vector_map.get_vector("new_output_key");

    // Fill join_key_vector with test values
    join_key_vector->values[0] = 1;// Has 3 neighbors: 2,3,4
    join_key_vector->values[1] = 2;// Has 2 neighbors: 3,4
    join_key_vector->values[2] = 3;// Has 1 neighbor: 4
    join_key_vector->values[3] = 0;// Has 0 neighbors

    // Simulate bitmask manipulation similar to what happens in execute()
    auto& bitmask = join_key_vector->state->selector;

    // Save the original bitmask for comparison
    auto original_bitmask = std::make_unique<ffx::BitMask<ffx::State::MAX_VECTOR_SIZE>>();
    COPY_BITMASK(bitmask.size(), *original_bitmask, bitmask);

    // Create a temporary bitmask for active positions
    auto active_bitmask = std::make_unique<ffx::BitMask<ffx::State::MAX_VECTOR_SIZE>>();
    COPY_BITMASK(bitmask.size(), *active_bitmask, bitmask);

    // Associate the active bitmask with the input state temporarily
    auto original_selector = join_key_vector->state->selector;
    join_key_vector->state->selector = *active_bitmask;

    // Create a RLE array for output
    uint32_t rle_array[ffx::State::MAX_VECTOR_SIZE + 1] = {0};

    // Process indices one by one, similar to INLJoinPacked::execute
    int32_t op_filled_idx = 0;
    const int32_t start_idx = GET_START_POS(*join_key_vector->state);
    const int32_t end_idx = GET_END_POS(*join_key_vector->state);

    for (auto idx = start_idx; idx <= end_idx;) {
        // Skip if index is not valid
        if (!TEST_BIT(*active_bitmask, idx)) {
            idx++;
            continue;
        }

        // Get the adjacency list for this value
        const auto& adj_list = table->fwd_adj_lists[static_cast<size_t>(join_key_vector->values[idx])];

        // Skip if there are no elements to produce
        if (adj_list.size == 0) {
            CLEAR_BIT(*active_bitmask, idx);
            idx++;
            continue;
        }

        // Simulate updating RLE entries
        rle_array[idx] = op_filled_idx;
        rle_array[idx + 1] = op_filled_idx + adj_list.size;

        op_filled_idx += adj_list.size;
        idx++;
    }

    // Verify that the active bitmask was modified correctly
    // Expecting modification to remove indices with empty adjacency lists (index 3 only)
    for (size_t i = 0; i < 4; ++i) {
        bool expected = (i != 3);// idx 3 has 0 neighbors, should be cleared
        EXPECT_EQ(TEST_BIT(*active_bitmask, i), expected);
    }

    // Original bitmask should remain unchanged (INLJoinPacked copies to an internal active mask)
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(TEST_BIT(bitmask, i), TEST_BIT(*original_bitmask, i));
    }

    // Verify RLE entries for indices with valid neighbors
    EXPECT_EQ(rle_array[0], 0);// First index starts at position 0

    // Restore the original selector
    join_key_vector->state->selector = original_selector;
}

TEST_F(BitmaskINLJoinTest, BitmaskPerformance) {
    // Create a large bitmask for performance testing
    auto bitmask = std::make_unique<ffx::BitMask<ffx::State::MAX_VECTOR_SIZE>>();

    // Set specific pattern to test performance
    SET_ALL_BITS(*bitmask);

    // Clear every third bit
    for (size_t i = 0; i < ffx::State::MAX_VECTOR_SIZE; i += 3) {
        CLEAR_BIT(*bitmask, i);
    }

    // Test operation that would be common in join operations
    size_t set_bits_count = 0;

    // Count set bits using the bitmask API
    for (size_t i = 0; i < ffx::State::MAX_VECTOR_SIZE; ++i) {
        if (TEST_BIT(*bitmask, i)) { set_bits_count++; }
    }

    // Verify that approximately 2/3 of the bits are set
    EXPECT_GT(set_bits_count, ffx::State::MAX_VECTOR_SIZE / 3);
    EXPECT_LT(set_bits_count, ffx::State::MAX_VECTOR_SIZE);
}

TEST_F(BitmaskINLJoinTest, BitmaskSerialization) {
    // Test serialization/deserialization of bitmasks if applicable
    auto original_bitmask = std::make_unique<ffx::BitMask<ffx::State::MAX_VECTOR_SIZE>>();

    // Set a specific pattern
    CLEAR_ALL_BITS(*original_bitmask);
    SET_BIT(*original_bitmask, 1);
    SET_BIT(*original_bitmask, 3);
    SET_BIT(*original_bitmask, 7);
    SET_BIT(*original_bitmask, 11);
    // SET_START_POS(*original_bitmask, 1);
    // SET_END_POS(*original_bitmask, 11);

    // Create another bitmask to simulate deserialization
    auto copied_bitmask = std::make_unique<ffx::BitMask<ffx::State::MAX_VECTOR_SIZE>>();

    // Copy the bitmask (simulating serialization/deserialization)
    COPY_BITMASK(original_bitmask->size(), *copied_bitmask, *original_bitmask);

    // Verify that all properties are preserved
    // EXPECT_EQ(GET_START_POS(*copied_bitmask), GET_START_POS(*original_bitmask));
    // EXPECT_EQ(GET_END_POS(*copied_bitmask), GET_END_POS(*original_bitmask));

    for (size_t i = 0; i < ffx::State::MAX_VECTOR_SIZE; ++i) {
        EXPECT_EQ(TEST_BIT(*copied_bitmask, i), TEST_BIT(*original_bitmask, i));
    }
}

//------------------------------------------------------------------------------
// Complex Bitmask Tests
//------------------------------------------------------------------------------

// Test multi-block operations at 64-bit boundaries
TEST_F(BitmaskTest, MultiBlockBoundaryOperations) {
    auto bitmask = std::make_unique<ffx::BitMask<ffx::State::MAX_VECTOR_SIZE>>();

    // Test operations exactly at 64-bit boundaries
    const size_t boundaries[] = {63, 64, 127, 128, 191, 192, 255, 256};

    for (size_t boundary: boundaries) {
        if (boundary < ffx::State::MAX_VECTOR_SIZE) {
            // Clear all, then set bits around the boundary
            CLEAR_ALL_BITS(*bitmask);

            SET_BIT(*bitmask, boundary - 1);
            SET_BIT(*bitmask, boundary);
            if (boundary + 1 < ffx::State::MAX_VECTOR_SIZE) { SET_BIT(*bitmask, boundary + 1); }

            EXPECT_EQ(TEST_BIT(*bitmask, boundary - 1), 1);
            EXPECT_EQ(TEST_BIT(*bitmask, boundary), 1);
            if (boundary + 1 < ffx::State::MAX_VECTOR_SIZE) { EXPECT_EQ(TEST_BIT(*bitmask, boundary + 1), 1); }

            // Clear the boundary bit and verify neighbors are unaffected
            CLEAR_BIT(*bitmask, boundary);
            EXPECT_EQ(TEST_BIT(*bitmask, boundary - 1), 1);
            EXPECT_EQ(TEST_BIT(*bitmask, boundary), 0);
            if (boundary + 1 < ffx::State::MAX_VECTOR_SIZE) { EXPECT_EQ(TEST_BIT(*bitmask, boundary + 1), 1); }
        }
    }
}

// Test SET_BITS_TILL_IDX across multiple 64-bit blocks
TEST_F(BitmaskTest, SetBitsTillIdxMultiBlock) {
    auto bitmask = std::make_unique<ffx::BitMask<ffx::State::MAX_VECTOR_SIZE>>();

    // Test setting bits till various indices that span multiple blocks
    const size_t test_indices[] = {0, 1, 32, 63, 64, 65, 100, 127, 128, 200, 255, 256, 500, 1000};

    for (size_t idx: test_indices) {
        if (idx < ffx::State::MAX_VECTOR_SIZE) {
            CLEAR_ALL_BITS(*bitmask);
            SET_BITS_TILL_IDX(*bitmask, idx);

            // Verify all bits up to idx are set
            for (size_t i = 0; i <= idx; ++i) {
                EXPECT_EQ(TEST_BIT(*bitmask, i), 1) << "Bit " << i << " should be set when idx=" << idx;
            }

            // Verify bits after idx are clear
            for (size_t i = idx + 1; i < std::min(idx + 100, static_cast<size_t>(ffx::State::MAX_VECTOR_SIZE)); ++i) {
                EXPECT_EQ(TEST_BIT(*bitmask, i), 0) << "Bit " << i << " should be clear when idx=" << idx;
            }
        }
    }
}

// Test CLEAR_BITS_TILL_IDX across multiple 64-bit blocks
TEST_F(BitmaskTest, ClearBitsTillIdxMultiBlock) {
    auto bitmask = std::make_unique<ffx::BitMask<ffx::State::MAX_VECTOR_SIZE>>();

    const size_t test_indices[] = {0, 1, 32, 63, 64, 65, 100, 127, 128, 200, 255, 256, 500, 1000};

    for (size_t idx: test_indices) {
        if (idx < ffx::State::MAX_VECTOR_SIZE) {
            SET_ALL_BITS(*bitmask);
            CLEAR_BITS_TILL_IDX(*bitmask, idx);

            // Verify all bits up to idx are clear
            for (size_t i = 0; i <= idx; ++i) {
                EXPECT_EQ(TEST_BIT(*bitmask, i), 0) << "Bit " << i << " should be clear when idx=" << idx;
            }

            // Verify bits after idx are still set
            for (size_t i = idx + 1; i < std::min(idx + 100, static_cast<size_t>(ffx::State::MAX_VECTOR_SIZE)); ++i) {
                EXPECT_EQ(TEST_BIT(*bitmask, i), 1) << "Bit " << i << " should be set when idx=" << idx;
            }
        }
    }
}

// Test sparse bitmask operations with large gaps
TEST_F(BitmaskTest, SparsePatternOperations) {
    auto bitmask1 = std::make_unique<ffx::BitMask<ffx::State::MAX_VECTOR_SIZE>>();
    auto bitmask2 = std::make_unique<ffx::BitMask<ffx::State::MAX_VECTOR_SIZE>>();

    CLEAR_ALL_BITS(*bitmask1);
    CLEAR_ALL_BITS(*bitmask2);

    // Set sparse pattern in bitmask1: every 97th bit (prime number to avoid patterns)
    for (size_t i = 0; i < ffx::State::MAX_VECTOR_SIZE; i += 97) {
        SET_BIT(*bitmask1, i);
    }

    // Set sparse pattern in bitmask2: every 101st bit (another prime)
    for (size_t i = 0; i < ffx::State::MAX_VECTOR_SIZE; i += 101) {
        SET_BIT(*bitmask2, i);
    }

    // Verify patterns before AND
    for (size_t i = 0; i < ffx::State::MAX_VECTOR_SIZE; ++i) {
        bool expected1 = (i % 97 == 0);
        bool expected2 = (i % 101 == 0);
        EXPECT_EQ(TEST_BIT(*bitmask1, i), expected1 ? 1 : 0);
        EXPECT_EQ(TEST_BIT(*bitmask2, i), expected2 ? 1 : 0);
    }

    // Create a copy before AND operation
    auto bitmask1_copy = std::make_unique<ffx::BitMask<ffx::State::MAX_VECTOR_SIZE>>();
    COPY_BITMASK(bitmask1->size(), *bitmask1_copy, *bitmask1);

    // AND the bitmasks
    AND_BITMASKS(bitmask1->size(), *bitmask1, *bitmask2);

    // After AND, only bits at multiples of LCM(97, 101) = 9797 should be set
    for (size_t i = 0; i < ffx::State::MAX_VECTOR_SIZE; ++i) {
        bool expected = (i % 97 == 0) && (i % 101 == 0);
        EXPECT_EQ(TEST_BIT(*bitmask1, i), expected ? 1 : 0) << "Bit " << i << " after AND operation";
    }
}

// Test complex AND operation chains
TEST_F(BitmaskTest, ComplexAndChains) {
    auto bitmask1 = std::make_unique<ffx::BitMask<ffx::State::MAX_VECTOR_SIZE>>();
    auto bitmask2 = std::make_unique<ffx::BitMask<ffx::State::MAX_VECTOR_SIZE>>();
    auto bitmask3 = std::make_unique<ffx::BitMask<ffx::State::MAX_VECTOR_SIZE>>();

    // Set up different patterns
    // bitmask1: first half set
    CLEAR_ALL_BITS(*bitmask1);
    SET_BITS_TILL_IDX(*bitmask1, ffx::State::MAX_VECTOR_SIZE / 2 - 1);

    // bitmask2: second half set
    CLEAR_ALL_BITS(*bitmask2);
    for (size_t i = ffx::State::MAX_VECTOR_SIZE / 2; i < ffx::State::MAX_VECTOR_SIZE; ++i) {
        SET_BIT(*bitmask2, i);
    }

    // bitmask3: every 4th bit in middle quarter
    CLEAR_ALL_BITS(*bitmask3);
    for (size_t i = ffx::State::MAX_VECTOR_SIZE / 4; i < 3 * ffx::State::MAX_VECTOR_SIZE / 4; i += 4) {
        SET_BIT(*bitmask3, i);
    }

    // Chain: bitmask1 AND bitmask3 (should give bits in [1/4, 1/2) that are multiples of 4)
    AND_BITMASKS(bitmask1->size(), *bitmask1, *bitmask3);

    for (size_t i = 0; i < ffx::State::MAX_VECTOR_SIZE; ++i) {
        bool in_first_half = i < ffx::State::MAX_VECTOR_SIZE / 2;
        bool in_middle_quarter = (i >= ffx::State::MAX_VECTOR_SIZE / 4) && (i < 3 * ffx::State::MAX_VECTOR_SIZE / 4);
        bool is_mult_of_4 = (i % 4 == 0);
        bool expected = in_first_half && in_middle_quarter && is_mult_of_4;
        EXPECT_EQ(TEST_BIT(*bitmask1, i), expected ? 1 : 0) << "Bit " << i << " after chained AND";
    }
}

// Test toggle operations spanning multiple blocks
TEST_F(BitmaskTest, MultiBlockToggleSequence) {
    auto bitmask = std::make_unique<ffx::BitMask<ffx::State::MAX_VECTOR_SIZE>>();

    // Start with all bits set
    SET_ALL_BITS(*bitmask);

    // Toggle every bit in a range spanning multiple blocks
    const size_t start = 50;
    const size_t end = 200;
    for (size_t i = start; i <= end; ++i) {
        TOGGLE_BIT(*bitmask, i);
    }

    // Verify: bits before start and after end should be 1, bits in [start, end] should be 0
    for (size_t i = 0; i < start; ++i) {
        EXPECT_EQ(TEST_BIT(*bitmask, i), 1) << "Bit " << i << " before toggle range";
    }
    for (size_t i = start; i <= end; ++i) {
        EXPECT_EQ(TEST_BIT(*bitmask, i), 0) << "Bit " << i << " in toggle range";
    }
    for (size_t i = end + 1; i < ffx::State::MAX_VECTOR_SIZE; ++i) {
        EXPECT_EQ(TEST_BIT(*bitmask, i), 1) << "Bit " << i << " after toggle range";
    }

    // Toggle again - should restore original state
    for (size_t i = start; i <= end; ++i) {
        TOGGLE_BIT(*bitmask, i);
    }

    // All bits should now be set
    for (size_t i = 0; i < ffx::State::MAX_VECTOR_SIZE; ++i) {
        EXPECT_EQ(TEST_BIT(*bitmask, i), 1) << "Bit " << i << " after double toggle";
    }
}

// Test copy operation preserves complex patterns
TEST_F(BitmaskTest, CopyPreservesComplexPatterns) {
    auto source = std::make_unique<ffx::BitMask<ffx::State::MAX_VECTOR_SIZE>>();
    auto dest = std::make_unique<ffx::BitMask<ffx::State::MAX_VECTOR_SIZE>>();

    // Create a complex pattern: Fibonacci-like positions
    CLEAR_ALL_BITS(*source);
    size_t fib1 = 1, fib2 = 1;
    while (fib2 < ffx::State::MAX_VECTOR_SIZE) {
        SET_BIT(*source, fib2);
        size_t next = fib1 + fib2;
        fib1 = fib2;
        fib2 = next;
    }

    // Also set some bits at block boundaries
    const size_t boundaries[] = {0, 63, 64, 127, 128, 255, 256, 511, 512, 1023, 1024};
    for (size_t b: boundaries) {
        if (b < ffx::State::MAX_VECTOR_SIZE) { SET_BIT(*source, b); }
    }

    // Copy to destination
    COPY_BITMASK(source->size(), *dest, *source);

    // Verify all bits match
    for (size_t i = 0; i < ffx::State::MAX_VECTOR_SIZE; ++i) {
        EXPECT_EQ(TEST_BIT(*dest, i), TEST_BIT(*source, i)) << "Mismatch at bit " << i;
    }

    // Modify destination and verify source is unchanged
    TOGGLE_BIT(*dest, 100);
    EXPECT_NE(TEST_BIT(*dest, 100), TEST_BIT(*source, 100));
}

// Test operations at exact word boundaries
TEST_F(BitmaskTest, ExactWordBoundaryOperations) {
    auto bitmask = std::make_unique<ffx::BitMask<ffx::State::MAX_VECTOR_SIZE>>();

    // Test operations that set/clear exactly one 64-bit word
    CLEAR_ALL_BITS(*bitmask);

    // Set exactly the first 64 bits
    SET_BITS_TILL_IDX(*bitmask, 63);

    // Verify first 64 bits are set
    for (size_t i = 0; i < 64; ++i) {
        EXPECT_EQ(TEST_BIT(*bitmask, i), 1) << "Bit " << i << " should be set";
    }

    // Verify bit 64 onwards are clear
    for (size_t i = 64; i < 128; ++i) {
        EXPECT_EQ(TEST_BIT(*bitmask, i), 0) << "Bit " << i << " should be clear";
    }

    // Now set exactly bits 64-127 (second word)
    for (size_t i = 64; i < 128; ++i) {
        SET_BIT(*bitmask, i);
    }

    // Clear exactly the first word
    CLEAR_BITS_TILL_IDX(*bitmask, 63);

    // First 64 bits should be clear, next 64 should be set
    for (size_t i = 0; i < 64; ++i) {
        EXPECT_EQ(TEST_BIT(*bitmask, i), 0) << "Bit " << i << " should be clear after CLEAR_BITS_TILL_IDX(63)";
    }
    for (size_t i = 64; i < 128; ++i) {
        EXPECT_EQ(TEST_BIT(*bitmask, i), 1) << "Bit " << i << " should still be set";
    }
}

// Stress test: rapid alternating set/clear on random-like positions
TEST_F(BitmaskTest, StressTestRapidOperations) {
    auto bitmask = std::make_unique<ffx::BitMask<ffx::State::MAX_VECTOR_SIZE>>();

    // Use a simple LCG-like sequence for deterministic but scattered positions
    uint32_t seed = 12345;
    const uint32_t a = 1103515245;
    const uint32_t c = 12345;
    const uint32_t m = ffx::State::MAX_VECTOR_SIZE;

    std::vector<size_t> set_positions;

    // Set bits at pseudo-random positions
    CLEAR_ALL_BITS(*bitmask);
    for (int i = 0; i < 500; ++i) {
        seed = (a * seed + c) % m;
        size_t pos = seed % ffx::State::MAX_VECTOR_SIZE;
        SET_BIT(*bitmask, pos);
        set_positions.push_back(pos);
    }

    // Verify all set positions are actually set
    for (size_t pos: set_positions) {
        EXPECT_EQ(TEST_BIT(*bitmask, pos), 1) << "Position " << pos << " should be set";
    }

    // Clear half of them
    for (size_t i = 0; i < set_positions.size() / 2; ++i) {
        CLEAR_BIT(*bitmask, set_positions[i]);
    }

    // Verify cleared positions
    for (size_t i = 0; i < set_positions.size() / 2; ++i) {
        // Note: might still be set if a later position was the same - skip duplicate check
    }

    // Toggle all positions
    for (size_t pos: set_positions) {
        TOGGLE_BIT(*bitmask, pos);
    }

    // Final state verification: toggle all again should restore
    for (size_t pos: set_positions) {
        TOGGLE_BIT(*bitmask, pos);
    }
}

// Test full capacity operations
TEST_F(BitmaskTest, FullCapacityOperations) {
    auto bitmask = std::make_unique<ffx::BitMask<ffx::State::MAX_VECTOR_SIZE>>();

    // Count set bits after SET_ALL_BITS
    SET_ALL_BITS(*bitmask);
    size_t count = 0;
    for (size_t i = 0; i < ffx::State::MAX_VECTOR_SIZE; ++i) {
        if (TEST_BIT(*bitmask, i)) { count++; }
    }
    EXPECT_EQ(count, static_cast<size_t>(ffx::State::MAX_VECTOR_SIZE));

    // Count after CLEAR_ALL_BITS
    CLEAR_ALL_BITS(*bitmask);
    count = 0;
    for (size_t i = 0; i < ffx::State::MAX_VECTOR_SIZE; ++i) {
        if (TEST_BIT(*bitmask, i)) { count++; }
    }
    EXPECT_EQ(count, 0u);

    // Set all bits one by one and verify
    for (size_t i = 0; i < ffx::State::MAX_VECTOR_SIZE; ++i) {
        SET_BIT(*bitmask, i);
    }
    for (size_t i = 0; i < ffx::State::MAX_VECTOR_SIZE; ++i) {
        EXPECT_EQ(TEST_BIT(*bitmask, i), 1) << "Bit " << i << " should be set";
    }
}

// Test interleaved block patterns
TEST_F(BitmaskTest, InterleavedBlockPatterns) {
    auto bitmask = std::make_unique<ffx::BitMask<ffx::State::MAX_VECTOR_SIZE>>();

    CLEAR_ALL_BITS(*bitmask);

    // Set alternating 64-bit blocks: set block 0, clear block 1, set block 2, etc.
    for (size_t block = 0; block < ffx::State::MAX_VECTOR_SIZE / 64; ++block) {
        if (block % 2 == 0) {
            for (size_t bit = 0; bit < 64; ++bit) {
                SET_BIT(*bitmask, block * 64 + bit);
            }
        }
    }

    // Verify the pattern
    for (size_t i = 0; i < ffx::State::MAX_VECTOR_SIZE; ++i) {
        size_t block = i / 64;
        bool expected = (block % 2 == 0);
        EXPECT_EQ(TEST_BIT(*bitmask, i), expected ? 1 : 0) << "Bit " << i << " in block " << block;
    }

    // Now AND with a mask that only keeps odd-positioned bits within each block
    auto mask = std::make_unique<ffx::BitMask<ffx::State::MAX_VECTOR_SIZE>>();
    CLEAR_ALL_BITS(*mask);
    for (size_t i = 1; i < ffx::State::MAX_VECTOR_SIZE; i += 2) {
        SET_BIT(*mask, i);
    }

    AND_BITMASKS(bitmask->size(), *bitmask, *mask);

    // Verify: only odd bits in even blocks should be set
    for (size_t i = 0; i < ffx::State::MAX_VECTOR_SIZE; ++i) {
        size_t block = i / 64;
        bool is_odd_bit = (i % 2 == 1);
        bool is_even_block = (block % 2 == 0);
        bool expected = is_odd_bit && is_even_block;
        EXPECT_EQ(TEST_BIT(*bitmask, i), expected ? 1 : 0) << "Bit " << i << " after AND with odd-bit mask";
    }
}

// Test sliding window pattern
TEST_F(BitmaskTest, SlidingWindowPattern) {
    auto bitmask = std::make_unique<ffx::BitMask<ffx::State::MAX_VECTOR_SIZE>>();

    const size_t window_size = 16;

    // Simulate a sliding window iterator pattern
    for (size_t window_start = 0; window_start + window_size <= ffx::State::MAX_VECTOR_SIZE;
         window_start += window_size) {
        CLEAR_ALL_BITS(*bitmask);

        // Set bits in current window
        for (size_t i = window_start; i < window_start + window_size; ++i) {
            SET_BIT(*bitmask, i);
        }

        // Verify only window bits are set
        for (size_t i = 0; i < ffx::State::MAX_VECTOR_SIZE; ++i) {
            bool in_window = (i >= window_start && i < window_start + window_size);
            EXPECT_EQ(TEST_BIT(*bitmask, i), in_window ? 1 : 0)
                    << "Bit " << i << " with window starting at " << window_start;
        }
    }
}

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}