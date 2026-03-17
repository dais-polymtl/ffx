#include <gtest/gtest.h>
#include "../src/operator/include/schema/adj_list_manager.hpp"
#include "../src/table/include/adj_list_builder.hpp"
#include <memory>

class AdjListManagerExtendedTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(AdjListManagerExtendedTest, RegisterAdjLists) {
    ffx::AdjListManager manager;
    
    // Create test adjacency lists
    uint64_t src_col[] = {0, 0, 1};
    uint64_t dest_col[] = {1, 2, 2};
    uint64_t num_rows = 3;
    uint64_t num_fwd_ids = 3;
    uint64_t num_bwd_ids = 3;
    
    std::unique_ptr<ffx::AdjList<uint64_t>[]> fwd_adj, bwd_adj;
    ffx::build_adj_lists_from_columns(
        src_col, dest_col, num_rows, num_fwd_ids, num_bwd_ids,
        fwd_adj, bwd_adj
    );
    
    // Register forward adjacency list
    manager.register_adj_lists(
        "test_table",
        "src",
        "dest",
        true,
        std::move(fwd_adj),
        num_fwd_ids
    );
    
    // Register backward adjacency list
    manager.register_adj_lists(
        "test_table",
        "src",
        "dest",
        false,
        std::move(bwd_adj),
        num_bwd_ids
    );
    
    // Verify registration
    EXPECT_TRUE(manager.is_loaded("test_table", "src", "dest", true));
    EXPECT_TRUE(manager.is_loaded("test_table", "src", "dest", false));
    
    // Verify retrieval
    auto* fwd = manager.get_or_load("test_table", "src", "dest", true);
    ASSERT_NE(fwd, nullptr);
    EXPECT_EQ(fwd[0].size, 2);
    
    auto* bwd = manager.get_or_load("test_table", "src", "dest", false);
    ASSERT_NE(bwd, nullptr);
    EXPECT_EQ(bwd[2].size, 2);
    
    // Verify num_adj_lists
    EXPECT_EQ(manager.get_num_adj_lists("test_table", "src", "dest", true), num_fwd_ids);
    EXPECT_EQ(manager.get_num_adj_lists("test_table", "src", "dest", false), num_bwd_ids);
}

TEST_F(AdjListManagerExtendedTest, RegisterMultipleAdjLists) {
    ffx::AdjListManager manager;
    
    // Register first pair
    uint64_t src1[] = {0, 1};
    uint64_t dest1[] = {1, 2};
    std::unique_ptr<ffx::AdjList<uint64_t>[]> fwd1, bwd1;
    ffx::build_adj_lists_from_columns(src1, dest1, 2, 3, 3, fwd1, bwd1);
    
    manager.register_adj_lists("table1", "a", "b", true, std::move(fwd1), 3);
    
    // Register second pair
    uint64_t src2[] = {0, 2};
    uint64_t dest2[] = {1, 3};
    std::unique_ptr<ffx::AdjList<uint64_t>[]> fwd2, bwd2;
    ffx::build_adj_lists_from_columns(src2, dest2, 2, 3, 4, fwd2, bwd2);
    
    manager.register_adj_lists("table2", "x", "y", true, std::move(fwd2), 3);
    
    // Verify both registered
    EXPECT_TRUE(manager.is_loaded("table1", "a", "b", true));
    EXPECT_TRUE(manager.is_loaded("table2", "x", "y", true));
    
    // Verify independent
    auto* adj1 = manager.get_or_load("table1", "a", "b", true);
    auto* adj2 = manager.get_or_load("table2", "x", "y", true);
    EXPECT_NE(adj1, adj2);
}

TEST_F(AdjListManagerExtendedTest, RegisterOverwrite) {
    ffx::AdjListManager manager;
    
    // Register first
    uint64_t src1[] = {0};
    uint64_t dest1[] = {1};
    std::unique_ptr<ffx::AdjList<uint64_t>[]> fwd1, bwd1;
    ffx::build_adj_lists_from_columns(src1, dest1, 1, 2, 2, fwd1, bwd1);
    
    manager.register_adj_lists("table", "a", "b", true, std::move(fwd1), 2);
    
    // Register again with different data (should overwrite)
    uint64_t src2[] = {0, 0};
    uint64_t dest2[] = {1, 2};
    std::unique_ptr<ffx::AdjList<uint64_t>[]> fwd2, bwd2;
    ffx::build_adj_lists_from_columns(src2, dest2, 2, 3, 3, fwd2, bwd2);
    
    manager.register_adj_lists("table", "a", "b", true, std::move(fwd2), 3);
    
    // Verify overwritten
    auto* adj = manager.get_or_load("table", "a", "b", true);
    EXPECT_EQ(adj[0].size, 2); // Should have 2 edges now
    EXPECT_EQ(manager.get_num_adj_lists("table", "a", "b", true), 3);
}

TEST_F(AdjListManagerExtendedTest, ClearCache) {
    ffx::AdjListManager manager;
    
    uint64_t src[] = {0};
    uint64_t dest[] = {1};
    std::unique_ptr<ffx::AdjList<uint64_t>[]> fwd, bwd;
    ffx::build_adj_lists_from_columns(src, dest, 1, 2, 2, fwd, bwd);
    
    manager.register_adj_lists("table", "a", "b", true, std::move(fwd), 2);
    EXPECT_TRUE(manager.is_loaded("table", "a", "b", true));
    
    manager.clear();
    EXPECT_FALSE(manager.is_loaded("table", "a", "b", true));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

