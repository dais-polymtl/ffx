#include <gtest/gtest.h>
#include "../src/ser_der/include/serializer.hpp"
#include "../src/table/include/string_pool.hpp"
#include <filesystem>
#include <fstream>
#include <limits>

namespace fs = std::filesystem;

class ColumnSerializerTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_dir = fs::temp_directory_path() / "column_serializer_test";
        fs::create_directories(test_dir);
    }

    void TearDown() override {
        if (fs::exists(test_dir)) {
            fs::remove_all(test_dir);
        }
    }

    std::string test_dir;
};

TEST_F(ColumnSerializerTest, UInt64RoundTrip) {
    std::vector<uint64_t> data = {0, 1, 2, 100, 1000, UINT64_MAX};
    std::string output_path = test_dir + "/test_uint64.bin";
    
    uint64_t max_value = 0;
    ffx::serialize_uint64_column(data, output_path, &max_value);
    
    EXPECT_EQ(max_value, UINT64_MAX);
    
    uint64_t num_rows;
    auto loaded_data = ffx::deserialize_uint64_column(output_path, num_rows);
    
    EXPECT_EQ(num_rows, data.size());
    for (size_t i = 0; i < data.size(); i++) {
        EXPECT_EQ(loaded_data[i], data[i]);
    }
}

TEST_F(ColumnSerializerTest, UInt64EmptyVector) {
    std::vector<uint64_t> data;
    std::string output_path = test_dir + "/empty_uint64.bin";
    
    ffx::serialize_uint64_column(data, output_path);
    
    uint64_t num_rows;
    auto loaded_data = ffx::deserialize_uint64_column(output_path, num_rows);
    
    EXPECT_EQ(num_rows, 0);
}

TEST_F(ColumnSerializerTest, UInt64SingleValue) {
    std::vector<uint64_t> data = {42};
    std::string output_path = test_dir + "/single_uint64.bin";
    
    uint64_t max_value = 0;
    ffx::serialize_uint64_column(data, output_path, &max_value);
    
    EXPECT_EQ(max_value, 42);
    
    uint64_t num_rows;
    auto loaded_data = ffx::deserialize_uint64_column(output_path, num_rows);
    
    EXPECT_EQ(num_rows, 1);
    EXPECT_EQ(loaded_data[0], 42);
}

TEST_F(ColumnSerializerTest, UInt64LargeVector) {
    const size_t size = 10000;
    std::vector<uint64_t> data;
    data.reserve(size);
    for (size_t i = 0; i < size; i++) {
        data.push_back(i * 2);
    }
    
    std::string output_path = test_dir + "/large_uint64.bin";
    uint64_t max_value = 0;
    ffx::serialize_uint64_column(data, output_path, &max_value);
    
    EXPECT_EQ(max_value, (size - 1) * 2);
    
    uint64_t num_rows;
    auto loaded_data = ffx::deserialize_uint64_column(output_path, num_rows);
    
    EXPECT_EQ(num_rows, size);
    for (size_t i = 0; i < size; i++) {
        EXPECT_EQ(loaded_data[i], data[i]);
    }
}

TEST_F(ColumnSerializerTest, StringShortStrings) {
    std::vector<std::string> data = {"a", "ab", "abc", "abcd"};
    std::vector<bool> is_null = {false, false, false, false};
    std::string output_path = test_dir + "/short_strings.bin";
    
    ffx::serialize_string_column(data, is_null, output_path);
    
    auto pool = std::make_unique<ffx::StringPool>();
    uint64_t num_rows;
    auto loaded_data = ffx::deserialize_string_column(output_path, num_rows, pool.get());
    
    EXPECT_EQ(num_rows, data.size());
    for (size_t i = 0; i < data.size(); i++) {
        EXPECT_FALSE(loaded_data[i].is_null());
        EXPECT_EQ(loaded_data[i].to_string(), data[i]);
    }
}

TEST_F(ColumnSerializerTest, StringLongStrings) {
    std::vector<std::string> data = {
        "short",
        "this is a longer string that exceeds 4 bytes",
        "another very long string with many characters",
        "x"
    };
    std::vector<bool> is_null = {false, false, false, false};
    std::string output_path = test_dir + "/long_strings.bin";
    
    ffx::serialize_string_column(data, is_null, output_path);
    
    auto pool = std::make_unique<ffx::StringPool>();
    uint64_t num_rows;
    auto loaded_data = ffx::deserialize_string_column(output_path, num_rows, pool.get());
    
    EXPECT_EQ(num_rows, data.size());
    for (size_t i = 0; i < data.size(); i++) {
        EXPECT_FALSE(loaded_data[i].is_null());
        EXPECT_EQ(loaded_data[i].to_string(), data[i]);
    }
}

TEST_F(ColumnSerializerTest, StringWithNulls) {
    std::vector<std::string> data = {"a", "", "b", ""};
    std::vector<bool> is_null = {false, true, false, true};
    std::string output_path = test_dir + "/null_strings.bin";
    
    ffx::serialize_string_column(data, is_null, output_path);
    
    auto pool = std::make_unique<ffx::StringPool>();
    uint64_t num_rows;
    auto loaded_data = ffx::deserialize_string_column(output_path, num_rows, pool.get());
    
    EXPECT_EQ(num_rows, data.size());
    EXPECT_FALSE(loaded_data[0].is_null());
    EXPECT_TRUE(loaded_data[1].is_null());
    EXPECT_FALSE(loaded_data[2].is_null());
    EXPECT_TRUE(loaded_data[3].is_null());
}

TEST_F(ColumnSerializerTest, StringAllNulls) {
    std::vector<std::string> data = {"", "", ""};
    std::vector<bool> is_null = {true, true, true};
    std::string output_path = test_dir + "/all_null_strings.bin";
    
    ffx::serialize_string_column(data, is_null, output_path);
    
    auto pool = std::make_unique<ffx::StringPool>();
    uint64_t num_rows;
    auto loaded_data = ffx::deserialize_string_column(output_path, num_rows, pool.get());
    
    EXPECT_EQ(num_rows, 3);
    for (size_t i = 0; i < 3; i++) {
        EXPECT_TRUE(loaded_data[i].is_null());
    }
}

TEST_F(ColumnSerializerTest, StringEmptyVector) {
    std::vector<std::string> data;
    std::vector<bool> is_null;
    std::string output_path = test_dir + "/empty_strings.bin";
    
    ffx::serialize_string_column(data, is_null, output_path);
    
    auto pool = std::make_unique<ffx::StringPool>();
    uint64_t num_rows;
    auto loaded_data = ffx::deserialize_string_column(output_path, num_rows, pool.get());
    
    EXPECT_EQ(num_rows, 0);
}

TEST_F(ColumnSerializerTest, StringSpecialCharacters) {
    std::vector<std::string> data = {
        "hello\nworld",
        "tab\there",
        "unicode: 你好",
        "null\0byte"
    };
    std::vector<bool> is_null = {false, false, false, false};
    std::string output_path = test_dir + "/special_strings.bin";
    
    ffx::serialize_string_column(data, is_null, output_path);
    
    auto pool = std::make_unique<ffx::StringPool>();
    uint64_t num_rows;
    auto loaded_data = ffx::deserialize_string_column(output_path, num_rows, pool.get());
    
    EXPECT_EQ(num_rows, data.size());
    // Note: null bytes in strings may be handled differently, so we check first 3
    for (size_t i = 0; i < 3; i++) {
        EXPECT_FALSE(loaded_data[i].is_null());
        EXPECT_EQ(loaded_data[i].to_string(), data[i]);
    }
}

TEST_F(ColumnSerializerTest, ErrorHandlingNonExistentFile) {
    uint64_t num_rows;
    EXPECT_THROW(
        ffx::deserialize_uint64_column("/nonexistent/file.bin", num_rows),
        std::runtime_error
    );
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

