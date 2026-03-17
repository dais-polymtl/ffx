#include <gtest/gtest.h>
#include <fstream>
#include <filesystem>

#include "../src/operator/include/schema/adj_list_manager.hpp"
#include "../src/operator/include/schema/schema.hpp"
#include "../src/table/include/table.hpp"
#include "../src/ser_der/include/serializer.hpp"

namespace fs = std::filesystem;

class AdjListManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create temp directory
        temp_dir = "/tmp/adj_list_manager_test_" + std::to_string(getpid());
        fs::create_directories(temp_dir);
        
        // Create a simple edge CSV file
        csv_path = temp_dir + "/edges.csv";
        {
            std::ofstream ofs(csv_path);
            ofs << "0 1\n";
            ofs << "0 2\n";
            ofs << "1 2\n";
            ofs << "2 3\n";
        }
        
        // Create and serialize a table
        // Table(num_fwd_ids, num_bwd_ids, csv_path)
        ffx::Table table(4, 5, csv_path);  // num_fwd=4 (0..3), num_bwd=5 (0..4 with 0 empty)
        ffx::serialize(table, temp_dir, "person", "friend");
    }
    
    void TearDown() override {
        if (!temp_dir.empty()) {
            fs::remove_all(temp_dir);
        }
    }
    
    std::string temp_dir;
    std::string csv_path;
};

// Test AdjListManager loading
TEST_F(AdjListManagerTest, LoadForwardAdjList) {
    ffx::AdjListManager manager;
    
    auto* adj_list = manager.load_adj_list(temp_dir, "person", "friend", true);
    ASSERT_NE(adj_list, nullptr);
    
    // Verify node 0 has neighbors 1 and 2
    EXPECT_EQ(adj_list[0].size, 2);
    EXPECT_EQ(adj_list[0].values[0], 1);
    EXPECT_EQ(adj_list[0].values[1], 2);
    
    // Verify node 1 has neighbor 2
    EXPECT_EQ(adj_list[1].size, 1);
    EXPECT_EQ(adj_list[1].values[0], 2);
    
    // Verify node 2 has neighbor 3
    EXPECT_EQ(adj_list[2].size, 1);
    EXPECT_EQ(adj_list[2].values[0], 3);
}

TEST_F(AdjListManagerTest, LoadBackwardAdjList) {
    ffx::AdjListManager manager;
    
    auto* adj_list = manager.load_adj_list(temp_dir, "person", "friend", false);
    ASSERT_NE(adj_list, nullptr);
    
    // Verify node 1 has incoming edge from 0
    EXPECT_EQ(adj_list[1].size, 1);
    EXPECT_EQ(adj_list[1].values[0], 0);
    
    // Verify node 2 has incoming edges from 0 and 1
    EXPECT_EQ(adj_list[2].size, 2);
    
    // Verify node 3 has incoming edge from 2
    EXPECT_EQ(adj_list[3].size, 1);
    EXPECT_EQ(adj_list[3].values[0], 2);
}

TEST_F(AdjListManagerTest, GetOrLoadCaching) {
    ffx::AdjListManager manager;
    
    EXPECT_FALSE(manager.is_loaded(temp_dir, "person", "friend", true));
    
    auto* adj_list1 = manager.get_or_load(temp_dir, "person", "friend", true);
    EXPECT_TRUE(manager.is_loaded(temp_dir, "person", "friend", true));
    
    // Second call should return cached pointer
    auto* adj_list2 = manager.get_or_load(temp_dir, "person", "friend", true);
    EXPECT_EQ(adj_list1, adj_list2);
}

TEST_F(AdjListManagerTest, GetNumAdjLists) {
    ffx::AdjListManager manager;
    
    manager.load_adj_list(temp_dir, "person", "friend", true);
    
    // num_fwd was 4 (0..3)
    uint64_t num = manager.get_num_adj_lists(temp_dir, "person", "friend", true);
    EXPECT_EQ(num, 4);
}

TEST_F(AdjListManagerTest, Clear) {
    ffx::AdjListManager manager;
    
    manager.load_adj_list(temp_dir, "person", "friend", true);
    EXPECT_TRUE(manager.is_loaded(temp_dir, "person", "friend", true));
    
    manager.clear();
    EXPECT_FALSE(manager.is_loaded(temp_dir, "person", "friend", true));
}

// Test Schema adj_list_map functionality
class SchemaAdjListTest : public AdjListManagerTest {
protected:
    void SetUp() override {
        AdjListManagerTest::SetUp();
    }
};

TEST_F(SchemaAdjListTest, RegisterAndGetAdjList) {
    ffx::AdjListManager manager;
    ffx::Schema schema;
    
    auto* adj_list = manager.load_adj_list(temp_dir, "person", "friend", true);
    uint64_t num_adj_lists = manager.get_num_adj_lists(temp_dir, "person", "friend", true);
    
    schema.register_adj_list("person", "friend", adj_list, num_adj_lists);
    
    EXPECT_TRUE(schema.has_adj_list("person", "friend"));
    EXPECT_EQ(schema.get_adj_list("person", "friend"), adj_list);
    EXPECT_EQ(schema.get_adj_list_size("person", "friend"), num_adj_lists);
}

TEST_F(SchemaAdjListTest, GetAdjListThrowsIfNotFound) {
    ffx::Schema schema;
    
    EXPECT_THROW(schema.get_adj_list("a", "b"), std::runtime_error);
    EXPECT_THROW(schema.get_adj_list_size("a", "b"), std::runtime_error);
}

TEST_F(SchemaAdjListTest, HasAdjListReturnsFalseIfNotRegistered) {
    ffx::Schema schema;
    
    EXPECT_FALSE(schema.has_adj_list("x", "y"));
}

TEST_F(SchemaAdjListTest, MultipleAdjListsInSchema) {
    ffx::AdjListManager manager;
    ffx::Schema schema;
    
    auto* fwd_adj_list = manager.load_adj_list(temp_dir, "person", "friend", true);
    auto* bwd_adj_list = manager.load_adj_list(temp_dir, "person", "friend", false);
    
    schema.register_adj_list("person", "friend", fwd_adj_list, 4);
    schema.register_adj_list("friend", "person", bwd_adj_list, 5);
    
    EXPECT_EQ(schema.get_adj_list("person", "friend"), fwd_adj_list);
    EXPECT_EQ(schema.get_adj_list("friend", "person"), bwd_adj_list);
    EXPECT_NE(fwd_adj_list, bwd_adj_list);
}
