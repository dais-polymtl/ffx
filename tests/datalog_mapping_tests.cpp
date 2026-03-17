#include <gtest/gtest.h>
#include "../src/table/include/table_loader.hpp"
#include "../src/query/include/query.hpp"
#include <vector>

class DatalogMappingTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(DatalogMappingTest, BasicPositionalMapping) {
    // Query: T(q1, q2) where T is table name
    // Table: T(a1, a2, a3)
    // Expected: q1 -> a1 (pos 0), q2 -> a2 (pos 1)
    
    ffx::Query query{"Q(q1, q2) := T(q1, q2)"};
    ASSERT_TRUE(query.is_datalog_format());
    ASSERT_EQ(query.num_rels, 1);
    
    std::vector<std::string> table_columns = {"a1", "a2", "a3"};
    std::string table_name = "T";
    
    auto [src_attr, dest_attr] = ffx::map_datalog_to_table_columns(
        query.rels[0],
        table_columns,
        table_name
    );
    
    EXPECT_EQ(src_attr, "a1");
    EXPECT_EQ(dest_attr, "a2");
}

TEST_F(DatalogMappingTest, ThreeColumnTable) {
    // Query: T(x, y)
    // Table: T(col1, col2, col3)
    // Expected: x -> col1, y -> col2
    
    ffx::Query query{"Q(x, y) := T(x, y)"};
    std::vector<std::string> table_columns = {"col1", "col2", "col3"};
    std::string table_name = "T";
    
    auto [src_attr, dest_attr] = ffx::map_datalog_to_table_columns(
        query.rels[0],
        table_columns,
        table_name
    );
    
    EXPECT_EQ(src_attr, "col1");
    EXPECT_EQ(dest_attr, "col2");
}

TEST_F(DatalogMappingTest, TwoColumnTable) {
    // Query: T(a, b)
    // Table: T(src, dest)
    // Expected: a -> src, b -> dest
    
    ffx::Query query{"Q(a, b) := T(a, b)"};
    std::vector<std::string> table_columns = {"src", "dest"};
    std::string table_name = "T";
    
    auto [src_attr, dest_attr] = ffx::map_datalog_to_table_columns(
        query.rels[0],
        table_columns,
        table_name
    );
    
    EXPECT_EQ(src_attr, "src");
    EXPECT_EQ(dest_attr, "dest");
}

TEST_F(DatalogMappingTest, TableNameMismatch) {
    // Query: T1(a, b)
    // Table: T2(col1, col2)
    // Should throw error
    
    ffx::Query query{"Q(a, b) := T1(a, b)"};
    std::vector<std::string> table_columns = {"col1", "col2"};
    std::string table_name = "T2";
    
    EXPECT_THROW(
        ffx::map_datalog_to_table_columns(query.rels[0], table_columns, table_name),
        std::runtime_error
    );
}

TEST_F(DatalogMappingTest, InsufficientColumns) {
    // Query: T(a, b)
    // Table: T(col1) - only 1 column
    // Should throw error
    
    ffx::Query query{"Q(a, b) := T(a, b)"};
    std::vector<std::string> table_columns = {"col1"};
    std::string table_name = "T";
    
    EXPECT_THROW(
        ffx::map_datalog_to_table_columns(query.rels[0], table_columns, table_name),
        std::runtime_error
    );
}

TEST_F(DatalogMappingTest, MultipleRelations) {
    // Query: T1(a, b), T2(b, c)
    // Test mapping for each relation
    
    ffx::Query query{"Q(a, b, c) := T1(a, b), T2(b, c)"};
    ASSERT_EQ(query.num_rels, 2);
    
    // Map first relation
    std::vector<std::string> table1_columns = {"col1", "col2"};
    auto [src1, dest1] = ffx::map_datalog_to_table_columns(
        query.rels[0],
        table1_columns,
        "T1"
    );
    EXPECT_EQ(src1, "col1");
    EXPECT_EQ(dest1, "col2");
    
    // Map second relation
    std::vector<std::string> table2_columns = {"attr1", "attr2", "attr3"};
    auto [src2, dest2] = ffx::map_datalog_to_table_columns(
        query.rels[1],
        table2_columns,
        "T2"
    );
    EXPECT_EQ(src2, "attr1");
    EXPECT_EQ(dest2, "attr2");
}

TEST_F(DatalogMappingTest, LongColumnNames) {
    // Query: T(var1, var2)
    // Table: T(very_long_column_name_1, very_long_column_name_2)
    
    ffx::Query query{"Q(var1, var2) := T(var1, var2)"};
    std::vector<std::string> table_columns = {
        "very_long_column_name_1",
        "very_long_column_name_2"
    };
    std::string table_name = "T";
    
    auto [src_attr, dest_attr] = ffx::map_datalog_to_table_columns(
        query.rels[0],
        table_columns,
        table_name
    );
    
    EXPECT_EQ(src_attr, "very_long_column_name_1");
    EXPECT_EQ(dest_attr, "very_long_column_name_2");
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

