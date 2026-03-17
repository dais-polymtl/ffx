#include <gtest/gtest.h>
#include "../src/ser_der/include/serializer.hpp"
#include "../src/ser_der/include/table_metadata.hpp"
#include "../src/table/include/table_loader.hpp"
#include "../src/table/include/adj_list_builder.hpp"
#include "../src/operator/include/schema/adj_list_manager.hpp"
#include "../src/query/include/query.hpp"
#include <filesystem>
#include <limits>

namespace fs = std::filesystem;

class EdgeCaseTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_dir = fs::temp_directory_path() / "edge_case_test";
        fs::create_directories(test_dir);
    }

    void TearDown() override {
        if (fs::exists(test_dir)) {
            fs::remove_all(test_dir);
        }
    }

    std::string test_dir;
};

TEST_F(EdgeCaseTest, EmptyDataset) {
    // Serialize empty dataset
    std::vector<ffx::ColumnConfig> configs;
    configs.emplace_back("id", 0, ffx::ColumnType::UINT64);
    std::vector<uint64_t> max_values = {0};
    
    ffx::write_metadata_binary(test_dir, configs, 0, max_values);
    
    auto metadata = ffx::read_metadata_binary(test_dir);
    EXPECT_EQ(metadata.num_rows, 0);
    EXPECT_EQ(metadata.columns.size(), 1);
}

TEST_F(EdgeCaseTest, SingleRowDataset) {
    std::vector<uint64_t> data = {42};
    std::vector<ffx::ColumnConfig> configs;
    configs.emplace_back("id", 0, ffx::ColumnType::UINT64);
    std::vector<uint64_t> max_values = {42};
    
    ffx::serialize_uint64_column(data, test_dir + "/id_uint64.bin", &max_values[0]);
    ffx::write_metadata_binary(test_dir, configs, 1, max_values);
    
    auto metadata = ffx::read_metadata_binary(test_dir);
    EXPECT_EQ(metadata.num_rows, 1);
    
    uint64_t num_rows;
    auto loaded = ffx::deserialize_uint64_column(test_dir + "/id_uint64.bin", num_rows);
    EXPECT_EQ(num_rows, 1);
    EXPECT_EQ(loaded[0], 42);
}

TEST_F(EdgeCaseTest, MaximumValues) {
    std::vector<uint64_t> data = {UINT64_MAX};
    std::vector<ffx::ColumnConfig> configs;
    configs.emplace_back("id", 0, ffx::ColumnType::UINT64);
    std::vector<uint64_t> max_values = {UINT64_MAX};
    
    ffx::serialize_uint64_column(data, test_dir + "/id_uint64.bin", &max_values[0]);
    ffx::write_metadata_binary(test_dir, configs, 1, max_values);
    
    auto metadata = ffx::read_metadata_binary(test_dir);
    EXPECT_EQ(metadata.columns[0].max_value, UINT64_MAX);
    
    uint64_t num_rows;
    auto loaded = ffx::deserialize_uint64_column(test_dir + "/id_uint64.bin", num_rows);
    EXPECT_EQ(loaded[0], UINT64_MAX);
}

TEST_F(EdgeCaseTest, AllSameValue) {
    // All rows have same value
    std::vector<uint64_t> data(100, 5);
    std::vector<ffx::ColumnConfig> configs;
    configs.emplace_back("id", 0, ffx::ColumnType::UINT64);
    std::vector<uint64_t> max_values = {5};
    
    ffx::serialize_uint64_column(data, test_dir + "/id_uint64.bin", &max_values[0]);
    ffx::write_metadata_binary(test_dir, configs, 100, max_values);
    
    uint64_t num_rows;
    auto loaded = ffx::deserialize_uint64_column(test_dir + "/id_uint64.bin", num_rows);
    EXPECT_EQ(num_rows, 100);
    for (uint64_t i = 0; i < 100; i++) {
        EXPECT_EQ(loaded[i], 5);
    }
}

TEST_F(EdgeCaseTest, AllSameString) {
    // All strings identical
    std::vector<std::string> data(10, "same");
    std::vector<bool> nulls(10, false);
    
    ffx::serialize_string_column(data, nulls, test_dir + "/name_string.bin");
    
    auto pool = std::make_unique<ffx::StringPool>();
    uint64_t num_rows;
    auto loaded = ffx::deserialize_string_column(test_dir + "/name_string.bin", num_rows, pool.get());
    
    EXPECT_EQ(num_rows, 10);
    for (uint64_t i = 0; i < 10; i++) {
        EXPECT_EQ(loaded[i].to_string(), "same");
    }
}

TEST_F(EdgeCaseTest, AllEdgesToSameVertex) {
    // All edges go to same vertex
    uint64_t src_col[] = {0, 1, 2, 3};
    uint64_t dest_col[] = {5, 5, 5, 5};
    uint64_t num_rows = 4;
    uint64_t num_fwd_ids = 6;
    uint64_t num_bwd_ids = 6;
    
    std::unique_ptr<ffx::AdjList<uint64_t>[]> fwd_adj, bwd_adj;
    ffx::build_adj_lists_from_columns(
        src_col, dest_col, num_rows, num_fwd_ids, num_bwd_ids,
        fwd_adj, bwd_adj
    );
    
    // All forward lists should have 1 edge to vertex 5
    for (uint64_t i = 0; i < 4; i++) {
        EXPECT_EQ(fwd_adj[i].size, 1);
        EXPECT_EQ(fwd_adj[i].values[0], 5);
    }
    
    // Vertex 5 should have 4 incoming edges
    EXPECT_EQ(bwd_adj[5].size, 4);
}

TEST_F(EdgeCaseTest, VeryLongAttributeName) {
    std::string long_name(200, 'a');
    std::vector<ffx::ColumnConfig> configs;
    configs.emplace_back(long_name, 0, ffx::ColumnType::UINT64);
    std::vector<uint64_t> max_values = {100};
    
    ffx::write_metadata_binary(test_dir, configs, 1, max_values);
    
    auto metadata = ffx::read_metadata_binary(test_dir);
    EXPECT_EQ(metadata.columns[0].attr_name, long_name);
    
    const auto* found = metadata.find_column(long_name);
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(found->attr_name, long_name);
}

TEST_F(EdgeCaseTest, SpecialCharactersInAttributeName) {
    std::string special_name = "attr_with_underscores_and-dashes";
    std::vector<ffx::ColumnConfig> configs;
    configs.emplace_back(special_name, 0, ffx::ColumnType::UINT64);
    std::vector<uint64_t> max_values = {100};
    
    ffx::write_metadata_binary(test_dir, configs, 1, max_values);
    
    auto metadata = ffx::read_metadata_binary(test_dir);
    const auto* found = metadata.find_column(special_name);
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(found->attr_name, special_name);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

