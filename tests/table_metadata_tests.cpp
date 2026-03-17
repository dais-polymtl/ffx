#include <gtest/gtest.h>
#include "../src/ser_der/include/table_metadata.hpp"
#include "../src/ser_der/include/serializer.hpp"
#include <filesystem>
#include <fstream>
#include <cstdio>

namespace fs = std::filesystem;

class TableMetadataTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create temporary directory for tests
        test_dir = fs::temp_directory_path() / "table_metadata_test";
        fs::create_directories(test_dir);
    }

    void TearDown() override {
        // Clean up temporary directory
        if (fs::exists(test_dir)) {
            fs::remove_all(test_dir);
        }
    }

    std::string test_dir;
};

TEST_F(TableMetadataTest, RoundTripTest) {
    // Create test metadata
    std::vector<ffx::ColumnConfig> columns;
    columns.emplace_back("src", 0, ffx::ColumnType::UINT64);
    columns.emplace_back("dest", 1, ffx::ColumnType::UINT64);
    columns.emplace_back("name", 2, ffx::ColumnType::STRING);
    
    uint64_t num_rows = 1000;
    std::vector<uint64_t> max_values = {500, 600, 0}; // max_value for strings is 0
    
    // Write metadata
    ffx::write_metadata_binary(test_dir, columns, num_rows, max_values);
    
    // Read metadata back
    ffx::SerializedTableMetadata metadata = ffx::read_metadata_binary(test_dir);
    
    // Verify
    EXPECT_EQ(metadata.num_rows, num_rows);
    EXPECT_EQ(metadata.columns.size(), 3);
    
    EXPECT_EQ(metadata.columns[0].attr_name, "src");
    EXPECT_EQ(metadata.columns[0].type, ffx::ColumnType::UINT64);
    EXPECT_EQ(metadata.columns[0].max_value, 500);
    
    EXPECT_EQ(metadata.columns[1].attr_name, "dest");
    EXPECT_EQ(metadata.columns[1].type, ffx::ColumnType::UINT64);
    EXPECT_EQ(metadata.columns[1].max_value, 600);
    
    EXPECT_EQ(metadata.columns[2].attr_name, "name");
    EXPECT_EQ(metadata.columns[2].type, ffx::ColumnType::STRING);
    EXPECT_EQ(metadata.columns[2].max_value, 0);
    
    // Test find_column
    const auto* src_col = metadata.find_column("src");
    ASSERT_NE(src_col, nullptr);
    EXPECT_EQ(src_col->attr_name, "src");
    
    const auto* name_col = metadata.find_column("name");
    ASSERT_NE(name_col, nullptr);
    EXPECT_EQ(name_col->type, ffx::ColumnType::STRING);
    
    const auto* missing_col = metadata.find_column("nonexistent");
    EXPECT_EQ(missing_col, nullptr);
}

TEST_F(TableMetadataTest, EmptyMetadataTest) {
    // Test with empty metadata
    std::vector<ffx::ColumnConfig> columns;
    uint64_t num_rows = 0;
    std::vector<uint64_t> max_values;
    
    ffx::write_metadata_binary(test_dir, columns, num_rows, max_values);
    
    ffx::SerializedTableMetadata metadata = ffx::read_metadata_binary(test_dir);
    
    EXPECT_EQ(metadata.num_rows, 0);
    EXPECT_EQ(metadata.columns.size(), 0);
}

TEST_F(TableMetadataTest, SingleColumnTest) {
    std::vector<ffx::ColumnConfig> columns;
    columns.emplace_back("id", 0, ffx::ColumnType::UINT64);
    
    uint64_t num_rows = 42;
    std::vector<uint64_t> max_values = {100};
    
    ffx::write_metadata_binary(test_dir, columns, num_rows, max_values);
    
    ffx::SerializedTableMetadata metadata = ffx::read_metadata_binary(test_dir);
    
    EXPECT_EQ(metadata.num_rows, 42);
    EXPECT_EQ(metadata.columns.size(), 1);
    EXPECT_EQ(metadata.columns[0].attr_name, "id");
    EXPECT_EQ(metadata.columns[0].max_value, 100);
}

TEST_F(TableMetadataTest, ErrorHandlingTest) {
    // Test reading from non-existent directory
    EXPECT_THROW(
        ffx::read_metadata_binary("/nonexistent/directory"),
        std::runtime_error
    );
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

