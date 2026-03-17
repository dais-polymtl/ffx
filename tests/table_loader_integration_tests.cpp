#include <gtest/gtest.h>
#include "../src/table/include/table_loader.hpp"
#include "../src/table/include/table_loader_utils.hpp"
#include "../src/operator/include/schema/adj_list_manager.hpp"
#include "../src/operator/include/schema/schema.hpp"
#include "../src/query/include/query.hpp"
#include "../src/ser_der/include/serializer.hpp"
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

class TableLoaderIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_dir = fs::temp_directory_path() / "table_loader_integration_test";
        fs::create_directories(test_dir);
        
        // Create test data: simple graph 0->1, 0->2, 1->2
        std::vector<uint64_t> src_data = {0, 0, 1};
        std::vector<uint64_t> dest_data = {1, 2, 2};
        
        // Serialize columns
        uint64_t max_src = 1, max_dest = 2;
        ffx::serialize_uint64_column(src_data, test_dir + "/src_uint64.bin", &max_src);
        ffx::serialize_uint64_column(dest_data, test_dir + "/dest_uint64.bin", &max_dest);
        
        // Create metadata
        std::vector<ffx::ColumnConfig> configs;
        configs.emplace_back("src", 0, ffx::ColumnType::UINT64);
        configs.emplace_back("dest", 1, ffx::ColumnType::UINT64);
        std::vector<uint64_t> max_values = {max_src, max_dest};
        ffx::write_metadata_binary(test_dir, configs, 3, max_values);
    }

    void TearDown() override {
        if (fs::exists(test_dir)) {
            fs::remove_all(test_dir);
        }
    }

    std::string test_dir;
};

TEST_F(TableLoaderIntegrationTest, LoadTableAndBuildAdjLists) {
    ffx::Query query{"Q(a,b,c) := R(a,b),R(b,c)"};
    ffx::AdjListManager adj_list_manager;
    
    // Load table from columns
    ffx::LoadedTable loaded_table = ffx::load_table_from_columns(
        test_dir,
        "src",
        "dest",
        query,
        adj_list_manager
    );
    
    // Verify columns loaded
    EXPECT_EQ(loaded_table.num_rows, 3);
    EXPECT_NE(loaded_table.get_uint64_column("src"), nullptr);
    EXPECT_NE(loaded_table.get_uint64_column("dest"), nullptr);
    
    // Verify adjacency lists registered
    EXPECT_TRUE(adj_list_manager.is_loaded(test_dir, "src", "dest", true));
    EXPECT_TRUE(adj_list_manager.is_loaded(test_dir, "src", "dest", false));
    
    // Get adjacency lists and verify
    auto* fwd_adj = adj_list_manager.get_or_load(test_dir, "src", "dest", true);
    EXPECT_NE(fwd_adj, nullptr);
    EXPECT_EQ(fwd_adj[0].size, 2); // 0 -> [1, 2]
    EXPECT_EQ(fwd_adj[1].size, 1); // 1 -> [2]
    
    auto* bwd_adj = adj_list_manager.get_or_load(test_dir, "src", "dest", false);
    EXPECT_NE(bwd_adj, nullptr);
    EXPECT_EQ(bwd_adj[1].size, 1); // 1 <- [0]
    EXPECT_EQ(bwd_adj[2].size, 2); // 2 <- [0, 1]
}

TEST_F(TableLoaderIntegrationTest, PopulateSchemaFromAdjListManager) {
    ffx::Query query{"Q(a,b) := R(a,b)"};
    ffx::AdjListManager adj_list_manager;
    
    // Load table and register adjlists
    ffx::load_table_from_columns(test_dir, "src", "dest", query, adj_list_manager);
    
    // Populate schema
    ffx::Schema schema;
    ffx::populate_schema_from_adj_list_manager(
        schema, query, test_dir, "src", "dest", adj_list_manager
    );
    
    // Verify schema populated
    EXPECT_TRUE(schema.has_adj_list("src", "dest"));
    EXPECT_TRUE(schema.has_adj_list("dest", "src"));
    
    auto* fwd_adj = schema.get_adj_list("src", "dest");
    EXPECT_NE(fwd_adj, nullptr);
    EXPECT_EQ(schema.get_adj_list_size("src", "dest"), 2); // num_fwd_ids = 2
}

TEST_F(TableLoaderIntegrationTest, StringJoinIntegration) {
    // Create test data with string columns
    std::vector<std::string> src_str = {"alice", "bob", "alice"};
    std::vector<std::string> dest_str = {"bob", "charlie", "charlie"};
    std::vector<bool> nulls = {false, false, false};
    
    ffx::serialize_string_column(src_str, nulls, test_dir + "/src_string.bin");
    ffx::serialize_string_column(dest_str, nulls, test_dir + "/dest_string.bin");
    
    std::vector<ffx::ColumnConfig> configs;
    configs.emplace_back("src", 0, ffx::ColumnType::STRING);
    configs.emplace_back("dest", 1, ffx::ColumnType::STRING);
    std::vector<uint64_t> max_values = {0, 0};
    ffx::write_metadata_binary(test_dir, configs, 3, max_values);
    
    ffx::Query query{"Q(a,b) := R(a,b)"};
    ffx::AdjListManager adj_list_manager;
    
    // Load table
    ffx::LoadedTable loaded_table = ffx::load_table_from_columns(
        test_dir, "src", "dest", query, adj_list_manager
    );
    
    // Verify string dictionary built
    EXPECT_NE(loaded_table.global_string_dict.get(), nullptr);
    EXPECT_GT(loaded_table.global_string_dict->size(), 0);
    
    // Verify ID columns created
    const uint64_t* src_ids = loaded_table.get_string_id_column("src");
    ASSERT_NE(src_ids, nullptr);
    
    // "alice" should have same ID in rows 0 and 2
    EXPECT_EQ(src_ids[0], src_ids[2]);
    
    // Verify adjacency lists built from IDs
    EXPECT_TRUE(adj_list_manager.is_loaded(test_dir, "src", "dest", true));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

