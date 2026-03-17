#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include "../src/ser_der/include/serializer.hpp"
#include "../src/table/include/table.hpp"

namespace fs = std::filesystem;

class IndexCreatorTest : public ::testing::Test {
protected:
    std::string test_dir;
    std::string csv_path;
    std::string output_dir;

    void SetUp() override {
        test_dir = "/tmp/index_creator_test_" + std::to_string(::testing::UnitTest::GetInstance()->random_seed());
        csv_path = test_dir + "/edges.csv";
        output_dir = test_dir + "/output";
        
        fs::create_directories(test_dir);
        fs::create_directories(output_dir);
        
        // Create test CSV: simple graph with 4 edges (space-separated)
        std::ofstream csv(csv_path);
        csv << "0 1\n0 2\n1 2\n2 0\n";
        csv.close();
    }

    void TearDown() override {
        fs::remove_all(test_dir);
    }
};

TEST_F(IndexCreatorTest, SerializeWithDefaultAttributes) {
    // Create table and serialize with default attributes
    ffx::Table table(3, 3, csv_path);
    ffx::serialize(table, output_dir);  // Uses defaults: "src", "dest"
    
    // Check files exist with default attribute names
    EXPECT_TRUE(fs::exists(output_dir + "/num_adj_lists_src_dest.bin"));
    EXPECT_TRUE(fs::exists(output_dir + "/fwd_src_dest_0_3.bin"));
    EXPECT_TRUE(fs::exists(output_dir + "/bwd_src_dest_0_3.bin"));
}

TEST_F(IndexCreatorTest, SerializeWithCustomAttributes) {
    // Create table and serialize with custom attributes
    ffx::Table table(3, 3, csv_path);
    ffx::serialize(table, output_dir, "person", "friend");
    
    // Check files exist with custom attribute names
    EXPECT_TRUE(fs::exists(output_dir + "/num_adj_lists_person_friend.bin"));
    EXPECT_TRUE(fs::exists(output_dir + "/fwd_person_friend_0_3.bin"));
    EXPECT_TRUE(fs::exists(output_dir + "/bwd_person_friend_0_3.bin"));
}

TEST_F(IndexCreatorTest, DeserializeWithCustomAttributes) {
    // Create table and serialize
    ffx::Table table(3, 3, csv_path);
    ffx::serialize(table, output_dir, "person", "friend");
    
    // Deserialize with same attribute names
    auto loaded = ffx::deserialize(output_dir, "person", "friend");
    
    ASSERT_NE(loaded, nullptr);
    EXPECT_EQ(loaded->num_fwd_ids, 3);
    EXPECT_EQ(loaded->num_bwd_ids, 3);
    
    // Verify adjacency lists
    // Node 0 -> [1, 2]
    EXPECT_EQ(loaded->fwd_adj_lists[0].size, 2);
    // Node 1 -> [2]
    EXPECT_EQ(loaded->fwd_adj_lists[1].size, 1);
    // Node 2 -> [0]
    EXPECT_EQ(loaded->fwd_adj_lists[2].size, 1);
}

TEST_F(IndexCreatorTest, AttributeNamesInFilenames) {
    // Test various attribute name combinations
    ffx::Table table(3, 3, csv_path);
    
    // Test with underscores in names
    std::string out1 = output_dir + "/test1";
    fs::create_directories(out1);
    ffx::serialize(table, out1, "src_node", "dest_node");
    EXPECT_TRUE(fs::exists(out1 + "/num_adj_lists_src_node_dest_node.bin"));
    
    // Test with longer names
    std::string out2 = output_dir + "/test2";
    fs::create_directories(out2);
    ffx::serialize(table, out2, "person_id", "friend_id");
    EXPECT_TRUE(fs::exists(out2 + "/num_adj_lists_person_id_friend_id.bin"));
}
