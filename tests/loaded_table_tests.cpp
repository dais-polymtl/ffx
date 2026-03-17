#include <gtest/gtest.h>
#include "../src/table/include/loaded_table.hpp"
#include "../src/ser_der/include/serializer.hpp"
#include "../src/table/include/string_pool.hpp"
#include <filesystem>
#include <memory>

namespace fs = std::filesystem;

class LoadedTableTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_dir = fs::temp_directory_path() / "loaded_table_test";
        fs::create_directories(test_dir);
        
        // Create test data
        pool = std::make_unique<ffx::StringPool>();
        dict = std::make_unique<ffx::StringDictionary>(pool.get());
    }

    void TearDown() override {
        if (fs::exists(test_dir)) {
            fs::remove_all(test_dir);
        }
    }

    std::string test_dir;
    std::unique_ptr<ffx::StringPool> pool;
    std::unique_ptr<ffx::StringDictionary> dict;
};

TEST_F(LoadedTableTest, UInt64ColumnAccess) {
    ffx::LoadedTable table;
    table.num_rows = 5;
    
    auto data = std::make_unique<uint64_t[]>(5);
    for (uint64_t i = 0; i < 5; i++) {
        data[i] = i * 10;
    }
    
    ffx::LoadedColumnUInt64 col;
    col.name = "id";
    col.data = std::move(data);
    col.num_rows = 5;
    col.max_value = 40;
    table.uint64_columns["id"] = std::move(col);
    
    const uint64_t* col_data = table.get_uint64_column("id");
    ASSERT_NE(col_data, nullptr);
    EXPECT_EQ(col_data[0], 0);
    EXPECT_EQ(col_data[1], 10);
    EXPECT_EQ(col_data[4], 40);
    
    EXPECT_EQ(table.get_uint64_column("nonexistent"), nullptr);
}

TEST_F(LoadedTableTest, StringColumnAccess) {
    ffx::LoadedTable table;
    table.num_rows = 3;
    table.global_string_pool = std::make_unique<ffx::StringPool>();
    table.global_string_dict = std::make_unique<ffx::StringDictionary>(table.global_string_pool.get());
    
    auto data = std::make_unique<ffx::ffx_str_t[]>(3);
    data[0] = ffx::ffx_str_t("hello", table.global_string_pool.get());
    data[1] = ffx::ffx_str_t("world", table.global_string_pool.get());
    data[2] = ffx::ffx_str_t("test", table.global_string_pool.get());
    
    auto id_column = std::make_unique<uint64_t[]>(3);
    for (uint64_t i = 0; i < 3; i++) {
        id_column[i] = table.global_string_dict->add_string(data[i]);
    }
    table.global_string_dict->finalize();
    
    ffx::LoadedColumnString col_str;
    col_str.name = "name";
    col_str.data = std::move(data);
    col_str.num_rows = 3;
    col_str.id_column = std::move(id_column);
    table.string_columns["name"] = std::move(col_str);
    
    const ffx::ffx_str_t* col = table.get_string_column("name");
    ASSERT_NE(col, nullptr);
    EXPECT_EQ(col[0].to_string(), "hello");
    EXPECT_EQ(col[1].to_string(), "world");
    EXPECT_EQ(col[2].to_string(), "test");
    
    EXPECT_EQ(table.get_string_column("nonexistent"), nullptr);
}

TEST_F(LoadedTableTest, StringIdColumnAccess) {
    ffx::LoadedTable table;
    table.num_rows = 3;
    table.global_string_pool = std::make_unique<ffx::StringPool>();
    table.global_string_dict = std::make_unique<ffx::StringDictionary>(table.global_string_pool.get());
    
    auto data = std::make_unique<ffx::ffx_str_t[]>(3);
    data[0] = ffx::ffx_str_t("a", table.global_string_pool.get());
    data[1] = ffx::ffx_str_t("b", table.global_string_pool.get());
    data[2] = ffx::ffx_str_t("a", table.global_string_pool.get()); // Duplicate
    
    auto id_column = std::make_unique<uint64_t[]>(3);
    for (uint64_t i = 0; i < 3; i++) {
        id_column[i] = table.global_string_dict->add_string(data[i]);
    }
    table.global_string_dict->finalize();
    
    ffx::LoadedColumnString col2;
    col2.name = "name";
    col2.data = std::move(data);
    col2.num_rows = 3;
    col2.id_column = std::move(id_column);
    table.string_columns["name"] = std::move(col2);
    
    const uint64_t* id_col = table.get_string_id_column("name");
    ASSERT_NE(id_col, nullptr);
    // "a" should get ID 0, "b" should get ID 1
    // Both "a" strings should have same ID
    EXPECT_EQ(id_col[0], id_col[2]); // Both "a" should have same ID
    EXPECT_NE(id_col[0], id_col[1]);  // "a" and "b" should have different IDs
}

TEST_F(LoadedTableTest, GlobalStringDictionary) {
    ffx::LoadedTable table;
    table.num_rows = 5;
    table.global_string_pool = std::make_unique<ffx::StringPool>();
    table.global_string_dict = std::make_unique<ffx::StringDictionary>(table.global_string_pool.get());
    
    // Add strings from multiple columns
    auto data1 = std::make_unique<ffx::ffx_str_t[]>(2);
    data1[0] = ffx::ffx_str_t("hello", table.global_string_pool.get());
    data1[1] = ffx::ffx_str_t("world", table.global_string_pool.get());
    
    auto id_column1 = std::make_unique<uint64_t[]>(2);
    for (uint64_t i = 0; i < 2; i++) {
        id_column1[i] = table.global_string_dict->add_string(data1[i]);
    }
    
    auto data2 = std::make_unique<ffx::ffx_str_t[]>(2);
    data2[0] = ffx::ffx_str_t("hello", table.global_string_pool.get()); // Duplicate
    data2[1] = ffx::ffx_str_t("test", table.global_string_pool.get());
    
    auto id_column2 = std::make_unique<uint64_t[]>(2);
    for (uint64_t i = 0; i < 2; i++) {
        id_column2[i] = table.global_string_dict->add_string(data2[i]);
    }
    
    table.global_string_dict->finalize();
    
    ffx::LoadedColumnString col1;
    col1.name = "col1";
    col1.data = std::move(data1);
    col1.num_rows = 2;
    col1.id_column = std::move(id_column1);
    table.string_columns["col1"] = std::move(col1);
    
    ffx::LoadedColumnString col2;
    col2.name = "col2";
    col2.data = std::move(data2);
    col2.num_rows = 2;
    col2.id_column = std::move(id_column2);
    table.string_columns["col2"] = std::move(col2);
    
    // Verify dictionary has 3 unique strings: "hello", "world", "test"
    EXPECT_EQ(table.global_string_dict->size(), 3);
    
    // Verify "hello" from both columns gets same ID
    EXPECT_EQ(table.get_string_id_column("col1")[0], 
              table.get_string_id_column("col2")[0]);
    
    // Verify get_string_dict returns global dictionary
    EXPECT_EQ(table.get_string_dict(), table.global_string_dict.get());
}

TEST_F(LoadedTableTest, EmptyTable) {
    ffx::LoadedTable table;
    table.num_rows = 0;
    
    EXPECT_EQ(table.get_uint64_column("any"), nullptr);
    EXPECT_EQ(table.get_string_column("any"), nullptr);
    EXPECT_EQ(table.get_string_id_column("any"), nullptr);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

