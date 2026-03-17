#include <gtest/gtest.h>
#include "../src/ser_der/include/serializer.hpp"
#include "../src/ser_der/include/table_metadata.hpp"
#include "../src/table/include/string_pool.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

class SerDesIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_dir = fs::temp_directory_path() / "ser_des_integration_test";
        fs::create_directories(test_dir);
        
        // Create test CSV
        csv_path = fs::path(test_dir) / "test.csv";
        std::ofstream csv(csv_path);
        csv << "0,1\n";
        csv << "0,2\n";
        csv << "1,2\n";
        csv << "2,3\n";
        csv.close();
        
        // Create config file
        config_path = fs::path(test_dir) / "config.txt";
        std::ofstream config(config_path);
        config << "COLUMN,src,0,uint64_t\n";
        config << "COLUMN,dest,1,uint64_t\n";
        config << "COLUMN,name,2,string\n";
        config << "PROJECTION,default,0,1,m:n\n";
        config.close();
    }

    void TearDown() override {
        if (fs::exists(test_dir)) {
            fs::remove_all(test_dir);
        }
    }

    std::string test_dir;
    fs::path csv_path;
    fs::path config_path;
};

TEST_F(SerDesIntegrationTest, FullRoundTrip) {
    // Parse config
    auto configs = ffx::parse_column_config(config_path.string());
    EXPECT_EQ(configs.size(), 3);
    
    // Read CSV and collect data
    std::vector<std::vector<uint64_t>> uint64_columns(2);
    std::vector<std::vector<std::string>> string_columns(1);
    std::vector<std::vector<bool>> string_nulls(1);
    
    std::ifstream csv_file(csv_path);
    std::string line;
    uint64_t row_count = 0;
    
    while (std::getline(csv_file, line)) {
        if (line.empty()) continue;
        
        std::stringstream ss(line);
        std::string field;
        std::vector<std::string> fields;
        
        while (std::getline(ss, field, ',')) {
            fields.push_back(field);
        }
        
        if (fields.size() >= 2) {
            uint64_columns[0].push_back(std::stoull(fields[0])); // src
            uint64_columns[1].push_back(std::stoull(fields[1])); // dest
            string_columns[0].push_back("name_" + fields[0]); // name
            string_nulls[0].push_back(false);
        }
        row_count++;
    }
    
    // Serialize columns
    std::vector<uint64_t> max_values(2, 0);
    ffx::serialize_uint64_column(uint64_columns[0], test_dir + "/src_uint64.bin", &max_values[0]);
    ffx::serialize_uint64_column(uint64_columns[1], test_dir + "/dest_uint64.bin", &max_values[1]);
    ffx::serialize_string_column(string_columns[0], string_nulls[0], test_dir + "/name_string.bin");
    
    // Write metadata
    ffx::write_metadata_binary(test_dir, configs, row_count, max_values);
    
    // Deserialize and verify
    auto metadata = ffx::read_metadata_binary(test_dir);
    EXPECT_EQ(metadata.num_rows, row_count);
    EXPECT_EQ(metadata.columns.size(), 3);
    
    uint64_t num_rows;
    auto src_data = ffx::deserialize_uint64_column(test_dir + "/src_uint64.bin", num_rows);
    EXPECT_EQ(num_rows, row_count);
    EXPECT_EQ(src_data[0], 0);
    EXPECT_EQ(src_data[1], 0);
    
    auto dest_data = ffx::deserialize_uint64_column(test_dir + "/dest_uint64.bin", num_rows);
    EXPECT_EQ(num_rows, row_count);
    EXPECT_EQ(dest_data[0], 1);
    EXPECT_EQ(dest_data[1], 2);
    
    auto pool = std::make_unique<ffx::StringPool>();
    auto name_data = ffx::deserialize_string_column(test_dir + "/name_string.bin", num_rows, pool.get());
    EXPECT_EQ(num_rows, row_count);
    EXPECT_EQ(name_data[0].to_string(), "name_0");
    EXPECT_EQ(name_data[1].to_string(), "name_0");
}

TEST_F(SerDesIntegrationTest, SelectiveDeserialization) {
    // Serialize 3 columns
    std::vector<ffx::ColumnConfig> configs;
    configs.emplace_back("col1", 0, ffx::ColumnType::UINT64);
    configs.emplace_back("col2", 1, ffx::ColumnType::UINT64);
    configs.emplace_back("col3", 2, ffx::ColumnType::STRING);
    
    std::vector<uint64_t> data1 = {1, 2, 3};
    std::vector<uint64_t> data2 = {10, 20, 30};
    std::vector<std::string> data3 = {"a", "b", "c"};
    std::vector<bool> nulls3 = {false, false, false};
    
    std::vector<uint64_t> max_values = {3, 30, 0};
    ffx::serialize_uint64_column(data1, test_dir + "/col1_uint64.bin", &max_values[0]);
    ffx::serialize_uint64_column(data2, test_dir + "/col2_uint64.bin", &max_values[1]);
    ffx::serialize_string_column(data3, nulls3, test_dir + "/col3_string.bin");
    ffx::write_metadata_binary(test_dir, configs, 3, max_values);
    
    // Deserialize only col1 and col3
    auto metadata = ffx::read_metadata_binary(test_dir);
    
    const auto* col1_info = metadata.find_column("col1");
    ASSERT_NE(col1_info, nullptr);
    
    const auto* col3_info = metadata.find_column("col3");
    ASSERT_NE(col3_info, nullptr);
    
    uint64_t num_rows;
    auto col1_data = ffx::deserialize_uint64_column(test_dir + "/col1_uint64.bin", num_rows);
    EXPECT_EQ(num_rows, 3);
    EXPECT_EQ(col1_data[0], 1);
    
    auto pool = std::make_unique<ffx::StringPool>();
    auto col3_data = ffx::deserialize_string_column(test_dir + "/col3_string.bin", num_rows, pool.get());
    EXPECT_EQ(num_rows, 3);
    EXPECT_EQ(col3_data[0].to_string(), "a");
    
    // col2 should not be loaded
    EXPECT_FALSE(fs::exists(test_dir + "/col2_uint64.bin") == false); // File exists, but we didn't load it
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

