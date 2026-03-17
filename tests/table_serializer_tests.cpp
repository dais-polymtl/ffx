#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include "../src/ser_der/include/serializer.hpp"
#include "../src/table/include/string_pool.hpp"

namespace fs = std::filesystem;

class TableSerializerTest : public ::testing::Test {
protected:
    std::string test_dir;
    std::string csv_path;
    std::string config_path;
    std::string output_dir;

    void SetUp() override {
        test_dir = "/tmp/table_ser_test_" + std::to_string(::testing::UnitTest::GetInstance()->random_seed());
        csv_path = test_dir + "/data.csv";
        config_path = test_dir + "/config.txt";
        output_dir = test_dir + "/output";
        
        fs::create_directories(test_dir);
        fs::create_directories(output_dir);
    }

    void TearDown() override {
        fs::remove_all(test_dir);
    }
};

TEST_F(TableSerializerTest, ParseColumnConfig) {
    // Create config file
    std::ofstream cfg(config_path);
    cfg << "COLUMN,person_id,0,uint64_t\n";
    cfg << "COLUMN,person_name,1,string\n";
    cfg << "# This is a comment\n";
    cfg << "COLUMN,person_age,2,uint64\n";  // Alternative type name
    cfg << "PROJECTION,PersonName,0,1,1:n\n";
    cfg.close();
    
    auto configs = ffx::parse_column_config(config_path);
    
    ASSERT_EQ(configs.size(), 3);
    
    EXPECT_EQ(configs[0].attr_name, "person_id");
    EXPECT_EQ(configs[0].column_idx, 0);
    EXPECT_EQ(configs[0].type, ffx::ColumnType::UINT64);
    
    EXPECT_EQ(configs[1].attr_name, "person_name");
    EXPECT_EQ(configs[1].column_idx, 1);
    EXPECT_EQ(configs[1].type, ffx::ColumnType::STRING);
    
    EXPECT_EQ(configs[2].attr_name, "person_age");
    EXPECT_EQ(configs[2].column_idx, 2);
    EXPECT_EQ(configs[2].type, ffx::ColumnType::UINT64);

    ASSERT_EQ(configs.projections.size(), 1);
    EXPECT_EQ(configs.projections[0].relation_name, "PersonName");
    EXPECT_EQ(configs.projections[0].source_column_idx, 0);
    EXPECT_EQ(configs.projections[0].target_column_idx, 1);
}

TEST_F(TableSerializerTest, SerializeDeserializeUint64Column) {
    std::vector<uint64_t> data = {100, 200, 300, 400, 500};
    std::string output_path = output_dir + "/test_uint64.bin";
    
    uint64_t max_val = 0;
    ffx::serialize_uint64_column(data, output_path, &max_val);
    
    EXPECT_EQ(max_val, 500);
    EXPECT_TRUE(fs::exists(output_path));
    
    // Deserialize and verify
    uint64_t num_rows;
    auto loaded = ffx::deserialize_uint64_column(output_path, num_rows);
    
    EXPECT_EQ(num_rows, 5);
    for (size_t i = 0; i < 5; i++) {
        EXPECT_EQ(loaded[i], data[i]);
    }
}

TEST_F(TableSerializerTest, SerializeDeserializeStringColumn) {
    std::vector<std::string> data = {"Alice", "Bob", "", "Charlie", "Diana"};
    std::vector<bool> is_null = {false, false, true, false, false};  // Third is NULL
    std::string output_path = output_dir + "/test_string.bin";
    
    ffx::serialize_string_column(data, is_null, output_path);
    EXPECT_TRUE(fs::exists(output_path));
    
    // Deserialize and verify
    uint64_t num_rows;
    ffx::StringPool pool;
    auto loaded = ffx::deserialize_string_column(output_path, num_rows, &pool);
    
    EXPECT_EQ(num_rows, 5);
    EXPECT_EQ(loaded[0].to_string(), "Alice");
    EXPECT_EQ(loaded[1].to_string(), "Bob");
    EXPECT_TRUE(loaded[2].is_null());  // NULL
    EXPECT_EQ(loaded[3].to_string(), "Charlie");
    EXPECT_EQ(loaded[4].to_string(), "Diana");
}

TEST_F(TableSerializerTest, MetadataFiles) {
    std::vector<ffx::ColumnConfig> configs = {
        {"person_id", 0, ffx::ColumnType::UINT64},
        {"person_name", 1, ffx::ColumnType::STRING},
        {"person_age", 2, ffx::ColumnType::UINT64}
    };
    std::vector<uint64_t> max_values = {100, 0, 50};
    
    ffx::write_metadata_json(output_dir, configs, 1000, max_values);
    ffx::write_metadata_binary(output_dir, configs, 1000, max_values);
    
    EXPECT_TRUE(fs::exists(output_dir + "/table_metadata.json"));
    EXPECT_TRUE(fs::exists(output_dir + "/table_metadata.bin"));
    
    // Verify JSON content
    std::ifstream json_file(output_dir + "/table_metadata.json");
    std::string content((std::istreambuf_iterator<char>(json_file)),
                        std::istreambuf_iterator<char>());
    EXPECT_TRUE(content.find("\"num_rows\": 1000") != std::string::npos);
    EXPECT_TRUE(content.find("\"person_id\"") != std::string::npos);
    EXPECT_TRUE(content.find("\"max_value\": 100") != std::string::npos);
}

TEST_F(TableSerializerTest, EmptyStringHandling) {
    std::vector<std::string> data = {"", "test", ""};
    std::vector<bool> is_null = {true, false, true};
    std::string output_path = output_dir + "/empty_string.bin";
    
    ffx::serialize_string_column(data, is_null, output_path);
    
    uint64_t num_rows;
    ffx::StringPool pool;
    auto loaded = ffx::deserialize_string_column(output_path, num_rows, &pool);
    
    EXPECT_EQ(num_rows, 3);
    EXPECT_TRUE(loaded[0].is_null());
    EXPECT_FALSE(loaded[1].is_null());
    EXPECT_EQ(loaded[1].to_string(), "test");
    EXPECT_TRUE(loaded[2].is_null());
}
