#include <gtest/gtest.h>
#include "../src/table/include/table_loader.hpp"
#include "../src/operator/include/schema/adj_list_manager.hpp"
#include "../src/query/include/query.hpp"
#include "../src/ser_der/include/serializer.hpp"
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

class DatalogTableLoadingTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_dir = fs::temp_directory_path() / "datalog_table_loading_test";
        fs::create_directories(test_dir);
        
        // Create test data: table T with columns a1, a2, a3
        // Data: 0->1, 0->2, 1->2
        std::vector<uint64_t> a1_data = {0, 0, 1};
        std::vector<uint64_t> a2_data = {1, 2, 2};
        std::vector<uint64_t> a3_data = {10, 20, 30}; // Extra column
        
        // Serialize columns
        uint64_t max_a1 = 1, max_a2 = 2, max_a3 = 30;
        ffx::serialize_uint64_column(a1_data, test_dir + "/a1_uint64.bin", &max_a1);
        ffx::serialize_uint64_column(a2_data, test_dir + "/a2_uint64.bin", &max_a2);
        ffx::serialize_uint64_column(a3_data, test_dir + "/a3_uint64.bin", &max_a3);
        
        // Create metadata with table T having columns a1, a2, a3
        std::vector<ffx::ColumnConfig> configs;
        configs.emplace_back("a1", 0, ffx::ColumnType::UINT64);
        configs.emplace_back("a2", 1, ffx::ColumnType::UINT64);
        configs.emplace_back("a3", 2, ffx::ColumnType::UINT64);
        std::vector<uint64_t> max_values = {max_a1, max_a2, max_a3};
        ffx::write_metadata_binary(test_dir, configs, 3, max_values);
    }

    void TearDown() override {
        if (fs::exists(test_dir)) {
            fs::remove_all(test_dir);
        }
    }

    std::string test_dir;
};

TEST_F(DatalogTableLoadingTest, LoadTableWithDatalogMapping) {
    // Query: T(q1, q2) - Datalog format
    // Table: T(a1, a2, a3)
    // Mapping: q1 -> a1, q2 -> a2 (by position)
    
    ffx::Query query{"Q(q1, q2) := T(q1, q2)"};
    ASSERT_TRUE(query.is_datalog_format());
    
    // Get table columns from metadata
    auto metadata = ffx::read_metadata_binary(test_dir);
    std::vector<std::string> table_columns;
    for (const auto& col : metadata.columns) {
        table_columns.push_back(col.attr_name);
    }
    
    // Map Datalog query to table columns
    auto [src_attr, dest_attr] = ffx::map_datalog_to_table_columns(
        query.rels[0],
        table_columns,
        "T"
    );
    
    EXPECT_EQ(src_attr, "a1");
    EXPECT_EQ(dest_attr, "a2");
    
    // Load table using mapped columns
    ffx::AdjListManager adj_list_manager;
    ffx::LoadedTable loaded_table = ffx::load_table_from_columns(
        test_dir,
        src_attr,
        dest_attr,
        query,
        adj_list_manager
    );
    
    // Verify columns loaded
    EXPECT_EQ(loaded_table.num_rows, 3);
    EXPECT_NE(loaded_table.get_uint64_column("a1"), nullptr);
    EXPECT_NE(loaded_table.get_uint64_column("a2"), nullptr);
    
    // Verify adjacency lists built correctly
    EXPECT_TRUE(adj_list_manager.is_loaded(test_dir, "a1", "a2", true));
    auto* fwd_adj = adj_list_manager.get_or_load(test_dir, "a1", "a2", true);
    EXPECT_EQ(fwd_adj[0].size, 2); // 0 -> [1, 2]
    EXPECT_EQ(fwd_adj[1].size, 1); // 1 -> [2]
}

TEST_F(DatalogTableLoadingTest, QueryVariablesDontMatchColumnNames) {
    // Query: T(x, y) - query variables are x, y
    // Table: T(a1, a2, a3) - actual column names are a1, a2, a3
    // This should work because we map by position, not by name
    
    ffx::Query query{"Q(x, y) := T(x, y)"};
    
    auto metadata = ffx::read_metadata_binary(test_dir);
    std::vector<std::string> table_columns;
    for (const auto& col : metadata.columns) {
        table_columns.push_back(col.attr_name);
    }
    
    // Map should work even though variable names don't match
    auto [src_attr, dest_attr] = ffx::map_datalog_to_table_columns(
        query.rels[0],
        table_columns,
        "T"
    );
    
    EXPECT_EQ(src_attr, "a1"); // Position 0
    EXPECT_EQ(dest_attr, "a2"); // Position 1
    
    // Load should succeed
    ffx::AdjListManager adj_list_manager;
    ffx::LoadedTable loaded_table = ffx::load_table_from_columns(
        test_dir,
        src_attr,
        dest_attr,
        query,
        adj_list_manager
    );
    
    EXPECT_EQ(loaded_table.num_rows, 3);
    EXPECT_NE(loaded_table.get_uint64_column("a1"), nullptr);
    EXPECT_NE(loaded_table.get_uint64_column("a2"), nullptr);
}

TEST_F(DatalogTableLoadingTest, MultipleTableQuery) {
    // Query: T1(q1, q2), T2(q2, q3)
    // For this test, we'll use the same table data for both
    // In practice, T1 and T2 would be different tables
    
    ffx::Query query{"Q(q1, q2, q3) := T1(q1, q2), T2(q2, q3)"};
    ASSERT_EQ(query.num_rels, 2);
    
    auto metadata = ffx::read_metadata_binary(test_dir);
    std::vector<std::string> table_columns;
    for (const auto& col : metadata.columns) {
        table_columns.push_back(col.attr_name);
    }
    
    // Map first relation T1
    auto [src1, dest1] = ffx::map_datalog_to_table_columns(
        query.rels[0],
        table_columns,
        "T1"
    );
    EXPECT_EQ(src1, "a1");
    EXPECT_EQ(dest1, "a2");
    
    // Map second relation T2 (using same table for test)
    auto [src2, dest2] = ffx::map_datalog_to_table_columns(
        query.rels[1],
        table_columns,
        "T2"
    );
    EXPECT_EQ(src2, "a1");
    EXPECT_EQ(dest2, "a2");
}

TEST_F(DatalogTableLoadingTest, TableNameMismatchError) {
    // Query: T1(q1, q2)
    // Table: T (different name)
    // Should throw error
    
    ffx::Query query{"Q(q1, q2) := T1(q1, q2)"};
    
    auto metadata = ffx::read_metadata_binary(test_dir);
    std::vector<std::string> table_columns;
    for (const auto& col : metadata.columns) {
        table_columns.push_back(col.attr_name);
    }
    
    EXPECT_THROW(
        ffx::map_datalog_to_table_columns(query.rels[0], table_columns, "T"),
        std::runtime_error
    );
}

TEST_F(DatalogTableLoadingTest, AutoLoadWithDatalogMapping) {
    // Test the auto-loading function that automatically does Datalog mapping
    // Query: T(q1, q2) - Datalog format
    // Table: T(a1, a2, a3)
    
    ffx::Query query{"Q(q1, q2) := T(q1, q2)"};
    ffx::AdjListManager adj_list_manager;
    
    auto metadata = ffx::read_metadata_binary(test_dir);
    std::vector<std::string> table_columns;
    for (const auto& col : metadata.columns) {
        table_columns.push_back(col.attr_name);
    }
    
    // Use auto-loading function
    ffx::LoadedTable loaded_table = ffx::load_table_from_columns_auto(
        test_dir,
        "T",
        table_columns,
        query,
        adj_list_manager
    );
    
    // Verify it automatically mapped q1->a1, q2->a2
    EXPECT_EQ(loaded_table.num_rows, 3);
    EXPECT_NE(loaded_table.get_uint64_column("a1"), nullptr);
    EXPECT_NE(loaded_table.get_uint64_column("a2"), nullptr);
    
    // Verify adjacency lists built with correct columns
    EXPECT_TRUE(adj_list_manager.is_loaded(test_dir, "a1", "a2", true));
}

TEST_F(DatalogTableLoadingTest, AutoLoadWithDatalogFormat) {
    // Auto-loading uses explicit src_attr/dest_attr when mapping from query + table "T"
    ffx::Query query{"Q(a, b) := T(a,b)"};
    ffx::AdjListManager adj_list_manager;
    
    auto metadata = ffx::read_metadata_binary(test_dir);
    std::vector<std::string> table_columns;
    for (const auto& col : metadata.columns) {
        table_columns.push_back(col.attr_name);
    }
    
    // Use auto-loading with explicit attributes (for arrow format)
    ffx::LoadedTable loaded_table = ffx::load_table_from_columns_auto(
        test_dir,
        "T",
        table_columns,
        query,
        adj_list_manager,
        "a1",  // src_attr
        "a2"   // dest_attr
    );
    
    // Verify it used the provided attributes
    EXPECT_EQ(loaded_table.num_rows, 3);
    EXPECT_NE(loaded_table.get_uint64_column("a1"), nullptr);
    EXPECT_NE(loaded_table.get_uint64_column("a2"), nullptr);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

