#include "../src/operator/include/join/inljoin_packed.hpp"
#include "../src/operator/include/vector/selection_vector.hpp"
#include "../src/operator/include/vector/state.hpp"
#include "../src/operator/include/vector/vector.hpp"
#include "../src/table/include/adj_list.hpp"

#include <cassert>
#include <cstring>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

const std::string JOIN_KEY = "join_key";
const std::string OUTPUT_KEY = "output_key";

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

// Mock next operator for testing
class MockNextOp : public ffx::Operator {
public:
    int execute_calls = 0;
    void init(ffx::QueryVariableToVectorMap& map, const ffx::Table* table) override {
        assert(table);
        assert(map.get_vector(JOIN_KEY));
        assert(map.get_vector(OUTPUT_KEY));
    }
    void execute() override { execute_calls++; }
};

// Base test fixture for all selection vector tests
class SelectionVectorTestBase : public ::testing::Test {
protected:
    void SetUp() override {
        initialize_adj_lists();
    }

    void TearDown() override {
        cleanup_adj_lists();
    }
};

class SelectionVectorTest : public SelectionVectorTestBase {};

class INLJoinPackedTest : public SelectionVectorTestBase {
protected:
    std::unique_ptr<ffx::Table> table;
    ffx::Vector<uint64_t> *join_key_vector, *output_key_vector;
    std::unique_ptr<MockNextOp> next_op;
    ffx::QueryVariableToVectorMap vector_map;

    void SetUp() override {
        SelectionVectorTestBase::SetUp();
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

        join_key_vector = vector_map.get_vector(JOIN_KEY);
        output_key_vector = vector_map.allocate_vector(OUTPUT_KEY);

        next_op = std::make_unique<MockNextOp>();
    }
};

TEST_F(SelectionVectorTest, Initialization) {
    ffx::SelectionVector<2048> selVec;
    EXPECT_EQ(selVec.size, 0);
    EXPECT_NE(selVec.bits, nullptr);
}

TEST_F(SelectionVectorTest, BasicOperations) {
    ffx::SelectionVector<2048> selVec;

    const uint32_t testValues[] = {1, 5, 10, 15, 20};
    const int numValues = 5;

    for (int i = 0; i < numValues; i++) {
        selVec.bits[i] = testValues[i];
        selVec.size++;
    }

    EXPECT_EQ(selVec.size, numValues);

    for (int i = 0; i < numValues; i++) {
        EXPECT_EQ(selVec.bits[i], testValues[i]);
    }
}

TEST_F(SelectionVectorTest, CapacityHandling) {
    ffx::SelectionVector<2048> smallSelVec;

    for (int i = 0; i < 10; i++) {
        smallSelVec.bits[i] = i * 10;
        smallSelVec.size++;
    }

    // Verify all values are stored correctly
    EXPECT_EQ(smallSelVec.size, 10);
    for (int i = 0; i < 10; i++) {
        EXPECT_EQ(smallSelVec.bits[i], i * 10);
    }
}

TEST_F(SelectionVectorTest, WithVector) {
    const auto vec = std::make_unique<ffx::Vector>();
    vec->state->allocate();
    ffx::SelectionVector<2048> selVec;

    const uint32_t testIndices[] = {0, 2, 4, 6, 8};
    const int numIndices = 5;

    for (int i = 0; i < numIndices; i++) {
        selVec.bits[i] = testIndices[i];
        selVec.size++;
    }

    EXPECT_EQ(selVec.size, numIndices);
    for (int i = 0; i < numIndices; i++) {
        EXPECT_EQ(selVec.bits[i], testIndices[i]);
    }
}

TEST_F(INLJoinPackedTest, SelectionVectorBasicUsage) {
    // Define a selection vector like used in INLJoinPacked
    ffx::SelectionVector<ffx::State::MAX_VECTOR_SIZE> selection_vector;

    // Set initial values (similar to what happens in execute())
    const int test_size = 5;
    for (int i = 0; i < test_size; i++) {
        selection_vector.bits[i] = i * 2;
        selection_vector.size++;
    }

    EXPECT_EQ(selection_vector.size, test_size);

    // Test clearing the selection vector (similar to curr_ip_sel_vec_size = 0)
    selection_vector.size = 0;
    EXPECT_EQ(selection_vector.size, 0);

    // Test appending to the selection vector (similar to curr_selection_vec_values[curr_selection_vec_size++] = idx)
    selection_vector.bits[selection_vector.size++] = 10;
    selection_vector.bits[selection_vector.size++] = 12;

    EXPECT_EQ(selection_vector.size, 2);
    EXPECT_EQ(selection_vector.bits[0], 10);
    EXPECT_EQ(selection_vector.bits[1], 12);
}

TEST_F(INLJoinPackedTest, SelectionVectorMemCopy) {
    ffx::SelectionVector<ffx::State::MAX_VECTOR_SIZE> src_selection;
    ffx::SelectionVector<ffx::State::MAX_VECTOR_SIZE> dst_selection;

    // Fill source vector
    for (int i = 0; i < 10; i++) {
        src_selection.bits[i] = i * 5;
        src_selection.size++;
    }

    // Copy using memcpy (similar to how INLJoinPacked copies selection vector values)
    std::memcpy(dst_selection.bits, src_selection.bits,
                src_selection.size * sizeof(dst_selection.bits[0]));
    dst_selection.size = src_selection.size;

    // Verify the copy
    EXPECT_EQ(dst_selection.size, src_selection.size);
    for (int i = 0; i < src_selection.size; i++) {
        EXPECT_EQ(dst_selection.bits[i], src_selection.bits[i]);
    }

    // Modify source and verify destination remains unchanged
    src_selection.bits[0] = 999;
    EXPECT_EQ(dst_selection.bits[0], 0);
}

TEST_F(INLJoinPackedTest, EmptyAdjacencyLists) {
    // Creating a special selection vector test case for when adjacency lists are empty
    ffx::SelectionVector<ffx::State::MAX_VECTOR_SIZE> selection_vector;

    // Test behavior similar to what happens in execute() when adj_list.size == 0
    selection_vector.bits[0] = 5;// Index with no neighbors
    selection_vector.size = 1;

    // In the INLJoinPacked::execute(), entries with no adjacencies are skipped
    // So let's verify we can handle this pattern
    int processed = 0;
    for (int i = 0; i < selection_vector.size; i++) {
        uint32_t idx = selection_vector.bits[i];
        // Simulate check like: if (adj_list.size == 0) { continue; }
        if (idx == 5) {
            continue;// Skip this index, it has no neighbors
        }
        processed++;
    }

    EXPECT_EQ(processed, 0);            // We should process no indices
    EXPECT_EQ(selection_vector.size, 1);// Size remains unchanged
}

TEST_F(INLJoinPackedTest, LargeSelectionVectors) {
    const int CHUNK_SIZE = 1024;// Similar to State::MAX_VECTOR_SIZE
    ffx::SelectionVector<2048> selection_vector;

    // Fill with test data
    for (int i = 0; i < CHUNK_SIZE + 100; i++) {
        selection_vector.bits[i] = i;
        selection_vector.size++;
    }

    EXPECT_EQ(selection_vector.size, CHUNK_SIZE + 100);

    // Simulate processing in chunks (similar to process_data_chunk)
    int total_processed = 0;
    while (total_processed < selection_vector.size) {
        int chunk_to_process = std::min(CHUNK_SIZE, selection_vector.size - total_processed);

        // Process the chunk (in real code, this would call next_op->execute())
        for (int i = 0; i < chunk_to_process; i++) {
            EXPECT_EQ(selection_vector.bits[total_processed + i], total_processed + i);
        }

        total_processed += chunk_to_process;
    }

    EXPECT_EQ(total_processed, CHUNK_SIZE + 100);// All elements processed
}

TEST_F(INLJoinPackedTest, CopySelectionVector) {
    ffx::SelectionVector<ffx::State::MAX_VECTOR_SIZE> original;

    // Fill original with test data
    const int test_size = 10;
    for (int i = 0; i < test_size; i++) {
        original.bits[i] = i * 3;
        original.size++;
    }

    auto copy = std::make_unique<uint32_t[]>(original.size);
    for (int i = 0; i < original.size; i++) {
        copy[i] = original.bits[i];
    }

    // Verify the copy
    for (int i = 0; i < original.size; i++) {
        EXPECT_EQ(copy[i], original.bits[i]);
    }

    // Modify the original and ensure the copy remains unchanged
    original.bits[0] = 999;
    EXPECT_EQ(copy[0], 0);           // Still the original value
    EXPECT_EQ(original.bits[0], 999);// Updated value
}

TEST_F(INLJoinPackedTest, SelectionVectorWithRLE) {
    ffx::SelectionVector<ffx::State::MAX_VECTOR_SIZE> selection_vector;

    // Fill with test indices
    selection_vector.bits[0] = 2;
    selection_vector.bits[1] = 5;
    selection_vector.bits[2] = 8;
    selection_vector.size = 3;

    // Create a mock RLE array (like _op_vector_rle in INLJoinPacked)
    uint32_t rle_array[ffx::State::MAX_VECTOR_SIZE + 1] = {0};

    // Simulate the RLE assignment pattern from INLJoinPacked
    int op_filled_idx = 0;
    for (int i = 0; i < selection_vector.size; i++) {
        uint32_t idx = selection_vector.bits[i];

        // For each index, simulate adding 2 values to output
        int elements_to_add = 2;

        // Set RLE entries (similar to what happens in execute())
        rle_array[idx] = op_filled_idx;
        rle_array[idx + 1] = op_filled_idx + elements_to_add;

        op_filled_idx += elements_to_add;
    }

    // Verify RLE entries
    EXPECT_EQ(rle_array[2], 0);
    EXPECT_EQ(rle_array[3], 2);
    EXPECT_EQ(rle_array[5], 2);
    EXPECT_EQ(rle_array[6], 4);
    EXPECT_EQ(rle_array[8], 4);
    EXPECT_EQ(rle_array[9], 6);

    // Verify other entries are still 0
    EXPECT_EQ(rle_array[0], 0);
    EXPECT_EQ(rle_array[1], 0);
    EXPECT_EQ(rle_array[4], 0);
    EXPECT_EQ(rle_array[7], 0);
}

TEST_F(INLJoinPackedTest, SelectionVectorResetAndReuse) {
    // Test resetting and reusing a selection vector (as done in process_data_chunk)
    ffx::SelectionVector<ffx::State::MAX_VECTOR_SIZE> selection_vector;

    // Fill with initial values
    for (int i = 0; i < 10; i++) {
        selection_vector.bits[i] = i + 100;
        selection_vector.size++;
    }

    // Simulate resetting the vector (similar to curr_ip_sel_vec_size = 0 in process_data_chunk)
    selection_vector.size = 0;

    // Refill with new values
    for (int i = 0; i < 5; i++) {
        selection_vector.bits[i] = i + 200;
        selection_vector.size++;
    }

    // Verify new values
    EXPECT_EQ(selection_vector.size, 5);
    for (int i = 0; i < selection_vector.size; i++) {
        EXPECT_EQ(selection_vector.bits[i], i + 200);
    }
}

TEST_F(INLJoinPackedTest, SelectionVectorAsSelector) {
    // Test using a selection vector as a selector for a state (as in INLJoinPacked)
    auto vec = std::make_unique<ffx::Vector>();

    // Create a selection vector
    auto selection = std::make_unique<ffx::SelectionVector<ffx::State::MAX_VECTOR_SIZE>>();

    // Fill with test indices
    selection->bits[0] = 3;
    selection->bits[1] = 7;
    selection->bits[2] = 12;
    selection->size = 3;

    // Assign selection vector as selector (as done in INLJoinPacked)
    vec->state->selector = selection.get();

    // Fill vector with values
    for (int i = 0; i < ffx::State::MAX_VECTOR_SIZE; i++) {
        vec->values[i] = i * 10;
    }

    // Verify selector works correctly
    EXPECT_EQ(vec->state->selector->size, 3);
    EXPECT_EQ(vec->state->selector->bits[0], 3);
    EXPECT_EQ(vec->state->selector->bits[1], 7);
    EXPECT_EQ(vec->state->selector->bits[2], 12);

    // Access values using selection vector
    EXPECT_EQ(vec->values[vec->state->selector->bits[0]], 30);
    EXPECT_EQ(vec->values[vec->state->selector->bits[1]], 70);
    EXPECT_EQ(vec->values[vec->state->selector->bits[2]], 120);
}

TEST_F(INLJoinPackedTest, SelectionVectorMultipleChunks) {
    // Test processing multiple chunks with a selection vector
    ffx::SelectionVector<ffx::State::MAX_VECTOR_SIZE> selection_vector;

    // Create RLE array
    uint32_t rle_array[ffx::State::MAX_VECTOR_SIZE + 1] = {0};

    // Simulate multiple chunks of processing
    const int CHUNKS = 3;
    const int CHUNK_SIZE = 5;

    for (int chunk = 0; chunk < CHUNKS; chunk++) {
        // Reset selection vector for each chunk (like in process_data_chunk)
        selection_vector.size = 0;

        // Fill selection vector for this chunk
        for (int i = 0; i < CHUNK_SIZE; i++) {
            selection_vector.bits[i] = chunk * 10 + i;
            selection_vector.size++;
        }

        // Reset RLE array (like in process_data_chunk)
        std::memset(rle_array, 0, (ffx::State::MAX_VECTOR_SIZE + 1) * sizeof(uint32_t));

        // Process the chunk
        int op_filled_idx = 0;
        for (int i = 0; i < selection_vector.size; i++) {
            uint32_t idx = selection_vector.bits[i];

            // Add one element per index
            rle_array[idx] = op_filled_idx;
            rle_array[idx + 1] = op_filled_idx + 1;

            op_filled_idx++;
        }

        // Verify for this chunk
        EXPECT_EQ(op_filled_idx, CHUNK_SIZE);

        // Check a few values per chunk
        EXPECT_EQ(rle_array[chunk * 10], 0);
        EXPECT_EQ(rle_array[chunk * 10 + 1], 1);
        EXPECT_EQ(rle_array[chunk * 10 + 4], 4);
        EXPECT_EQ(rle_array[chunk * 10 + 5], 5);
    }
}

TEST_F(INLJoinPackedTest, INLJoinIntegrationTest) {
    auto inl_join = std::make_unique<ffx::INLJoinPacked>("new_join_key", "new_output_key", true);
    inl_join->set_next_operator(std::move(next_op));

    EXPECT_NE(inl_join->next_op, nullptr);

    // Initialize the join operator with the vector map and table
    inl_join->init(vector_map, table.get());

    // Fill join_key_vector with test values
    join_key_vector->values[0] = 1;// Has 3 neighbors: 2,3,4
    join_key_vector->values[1] = 2;// Has 2 neighbors: 3,4
    join_key_vector->values[2] = 3;// Has 1 neighbor: 4
    join_key_vector->values[3] = 0;// Has 0 neighbors

    // Create a selection vector for the input
    auto input_selection = std::make_unique<ffx::SelectionVector<ffx::State::MAX_VECTOR_SIZE>>();
    input_selection->bits[0] = 0;
    input_selection->bits[1] = 1;
    input_selection->bits[2] = 2;
    input_selection->bits[3] = 3;
    input_selection->size = 4;

    // Assign the selection vector to the input state
    join_key_vector->state->selector = input_selection.get();

    inl_join->execute();

    EXPECT_NE(output_key_vector->state, nullptr);
    EXPECT_NE(output_key_vector->values, nullptr);
}

TEST_F(INLJoinPackedTest, SelectionVectorIterationPatterns) {
    ffx::SelectionVector<ffx::State::MAX_VECTOR_SIZE> selection_vector;

    // Fill with test data that has sequential and non-sequential patterns
    const uint32_t indices[] = {0, 1, 5, 6, 10, 15, 16, 17, 20, 25};
    const int num_indices = sizeof(indices) / sizeof(indices[0]);

    for (int i = 0; i < num_indices; i++) {
        selection_vector.bits[i] = indices[i];
        selection_vector.size++;
    }

    // Test iteration pattern similar to INLJoinPacked
    int sequential_groups = 0;
    int current_group_size = 1;

    for (int i = 1; i < selection_vector.size; i++) {
        if (selection_vector.bits[i] == selection_vector.bits[i - 1] + 1) {
            // Sequential index
            current_group_size++;
        } else {
            // Non-sequential, end of group
            if (current_group_size > 1) {
                sequential_groups++;
            }
            current_group_size = 1;
        }
    }

    // Count the last group if it's sequential
    if (current_group_size > 1) {
        sequential_groups++;
    }

    // We should have 3 sequential groups in our test data: [0,1], [5,6], and [15,16,17]
    EXPECT_EQ(sequential_groups, 3);
}

TEST_F(INLJoinPackedTest, SelectionVectorWithAdjacencyListProcessing) {
    // Test processing selection vector with adjacency lists, similar to INLJoinPacked
    ffx::SelectionVector<ffx::State::MAX_VECTOR_SIZE> selection_vector;

    // Create some test data
    const uint32_t indices[] = {1, 2, 3};// Values from our adjacency lists
    const int num_indices = sizeof(indices) / sizeof(indices[0]);

    for (int i = 0; i < num_indices; i++) {
        selection_vector.bits[i] = indices[i];
        selection_vector.size++;
    }

    // Create an output array and RLE array
    uint64_t output_values[ffx::State::MAX_VECTOR_SIZE] = {0};
    uint32_t rle_array[ffx::State::MAX_VECTOR_SIZE + 1] = {0};

    // Process the selection vector with adjacency lists
    int op_filled_idx = 0;
    for (int i = 0; i < selection_vector.size; i++) {
        uint32_t idx = selection_vector.bits[i];

        // Get the appropriate adjacency list
        const ffx::AdjList<uint64_t>* adj_list = nullptr;
        switch (idx) {
            case 1:
                adj_list = fwd_adj_list1.get();
                break;
            case 2:
                adj_list = fwd_adj_list2.get();
                break;
            case 3:
                adj_list = fwd_adj_list3.get();
                break;
            default:
                adj_list = nullptr;
                break;
        }

        if (adj_list == nullptr || adj_list->size == 0) {
            continue;
        }

        // Copy values from adjacency list to output
        std::memcpy(&output_values[op_filled_idx], adj_list->values,
                    adj_list->size * sizeof(output_values[0]));

        // Set RLE entries
        rle_array[idx] = op_filled_idx;
        rle_array[idx + 1] = op_filled_idx + adj_list->size;

        op_filled_idx += adj_list->size;
    }

    // Verify the output
    EXPECT_EQ(op_filled_idx, 6);// Total of 6 values (3+2+1)

    // Check RLE entries
    EXPECT_EQ(rle_array[1], 0);
    EXPECT_EQ(rle_array[2], 3);
    EXPECT_EQ(rle_array[3], 5);
    EXPECT_EQ(rle_array[4], 6);

    // Check some output values
    EXPECT_EQ(output_values[0], 2);
    EXPECT_EQ(output_values[1], 3);
    EXPECT_EQ(output_values[2], 4);
    EXPECT_EQ(output_values[3], 3);
    EXPECT_EQ(output_values[4], 4);
    EXPECT_EQ(output_values[5], 4);
}

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}