#include "../src/table/include/string_dictionary.hpp"
#include "../src/table/include/string_pool.hpp"
#include <gtest/gtest.h>
#include <vector>

namespace ffx {

class StringDictionaryTests : public ::testing::Test {
protected:
    void SetUp() override { pool = std::make_unique<StringPool>(); }

    void TearDown() override { pool.reset(); }

    std::unique_ptr<StringPool> pool;
};

TEST_F(StringDictionaryTests, BuildFromStrings) {
    StringDictionary dict(pool.get());

    std::vector<ffx_str_t> strings;
    strings.emplace_back("hello", pool.get());
    strings.emplace_back("world", pool.get());
    strings.emplace_back("test", pool.get());
    dict.build(strings);

    EXPECT_EQ(dict.size(), 3);
    EXPECT_TRUE(dict.is_finalized());
}

TEST_F(StringDictionaryTests, GetStringById) {
    StringDictionary dict(pool.get());

    std::vector<ffx_str_t> strings;
    strings.emplace_back("hello", pool.get());
    strings.emplace_back("world", pool.get());
    strings.emplace_back("test", pool.get());
    dict.build(strings);

    EXPECT_EQ(dict.get_string(0).to_string(), "hello");
    EXPECT_EQ(dict.get_string(1).to_string(), "world");
    EXPECT_EQ(dict.get_string(2).to_string(), "test");
}

TEST_F(StringDictionaryTests, GetIdByFfxStrT) {
    StringDictionary dict(pool.get());

    std::vector<ffx_str_t> strings;
    strings.emplace_back("hello", pool.get());
    strings.emplace_back("world", pool.get());
    strings.emplace_back("test", pool.get());
    dict.build(strings);

    ffx_str_t hello_str("hello", pool.get());
    ffx_str_t world_str("world", pool.get());
    ffx_str_t test_str("test", pool.get());
    ffx_str_t nonexistent_str("nonexistent", pool.get());

    EXPECT_EQ(dict.get_id(hello_str), 0);
    EXPECT_EQ(dict.get_id(world_str), 1);
    EXPECT_EQ(dict.get_id(test_str), 2);
    EXPECT_EQ(dict.get_id(nonexistent_str), UINT64_MAX);
}

TEST_F(StringDictionaryTests, HasId) {
    StringDictionary dict(pool.get());

    std::vector<ffx_str_t> strings;
    strings.emplace_back("hello", pool.get());
    strings.emplace_back("world", pool.get());
    dict.build(strings);

    EXPECT_TRUE(dict.has_id(0));
    EXPECT_TRUE(dict.has_id(1));
    EXPECT_FALSE(dict.has_id(2));
    EXPECT_FALSE(dict.has_id(100));
}

TEST_F(StringDictionaryTests, HasString) {
    StringDictionary dict(pool.get());

    std::vector<ffx_str_t> strings;
    strings.emplace_back("hello", pool.get());
    strings.emplace_back("world", pool.get());
    dict.build(strings);

    ffx_str_t hello_str("hello", pool.get());
    ffx_str_t world_str("world", pool.get());
    ffx_str_t test_str("test", pool.get());

    EXPECT_TRUE(dict.has_string(hello_str));
    EXPECT_TRUE(dict.has_string(world_str));
    EXPECT_FALSE(dict.has_string(test_str));
}

TEST_F(StringDictionaryTests, AddString) {
    StringDictionary dict(pool.get());

    uint64_t id1 = dict.add_string(ffx_str_t("hello", pool.get()));
    uint64_t id2 = dict.add_string(ffx_str_t("world", pool.get()));
    uint64_t id3 = dict.add_string(ffx_str_t("hello", pool.get()));// Duplicate

    EXPECT_EQ(id1, 0);
    EXPECT_EQ(id2, 1);
    EXPECT_EQ(id3, 0);// Should return existing ID

    EXPECT_FALSE(dict.is_finalized());

    dict.finalize();

    ffx_str_t hello = ffx_str_t("hello", pool.get());
    ffx_str_t world = ffx_str_t("world", pool.get());

    EXPECT_TRUE(dict.is_finalized());
    EXPECT_EQ(dict.get_id(hello), 0);
    EXPECT_EQ(dict.get_id(world), 1);
}

TEST_F(StringDictionaryTests, BuildFromPairs) {
    StringDictionary dict(pool.get());

    std::vector<std::pair<ffx_str_t, uint64_t>> entries;
    entries.emplace_back(ffx_str_t("hello", pool.get()), 5);
    entries.emplace_back(ffx_str_t("world", pool.get()), 10);
    entries.emplace_back(ffx_str_t("test", pool.get()), 15);

    dict.build(entries);

    ffx_str_t hello = ffx_str_t("hello", pool.get());
    ffx_str_t world = ffx_str_t("world", pool.get());
    ffx_str_t test = ffx_str_t("test", pool.get());

    EXPECT_EQ(dict.size(), 16);// Max ID + 1
    EXPECT_EQ(dict.get_string(5).to_string(), "hello");
    EXPECT_EQ(dict.get_string(10).to_string(), "world");
    EXPECT_EQ(dict.get_string(15).to_string(), "test");
    EXPECT_EQ(dict.get_id(hello), 5);
    EXPECT_EQ(dict.get_id(world), 10);
    EXPECT_EQ(dict.get_id(test), 15);
}

TEST_F(StringDictionaryTests, Merge) {
    StringDictionary dict1(pool.get());
    dict1.add_string(ffx_str_t("hello", pool.get()));
    dict1.add_string(ffx_str_t("world", pool.get()));

    StringDictionary dict2(pool.get());
    dict2.add_string(ffx_str_t("test", pool.get()));
    dict2.add_string(ffx_str_t("foo", pool.get()));

    dict1.merge(dict2);
    dict1.finalize();

    ffx_str_t hello = ffx_str_t("hello", pool.get());
    ffx_str_t world = ffx_str_t("world", pool.get());
    ffx_str_t test = ffx_str_t("test", pool.get());
    ffx_str_t foo = ffx_str_t("foo", pool.get());

    EXPECT_EQ(dict1.size(), 4);
    EXPECT_EQ(dict1.get_id(hello), 0);
    EXPECT_EQ(dict1.get_id(world), 1);
    EXPECT_EQ(dict1.get_id(test), 2);
    EXPECT_EQ(dict1.get_id(foo), 3);
}

TEST_F(StringDictionaryTests, MergeWithDuplicates) {
    StringDictionary dict1(pool.get());
    dict1.add_string(ffx_str_t("hello", pool.get()));
    dict1.add_string(ffx_str_t("world", pool.get()));

    StringDictionary dict2(pool.get());
    dict2.add_string(ffx_str_t("hello", pool.get()));// Duplicate
    dict2.add_string(ffx_str_t("test", pool.get()));

    dict1.merge(dict2);
    dict1.finalize();

    ffx_str_t hello = ffx_str_t("hello", pool.get());
    ffx_str_t world = ffx_str_t("world", pool.get());
    ffx_str_t test = ffx_str_t("test", pool.get());

    // Should have 3 unique strings
    EXPECT_EQ(dict1.size(), 3);
    EXPECT_EQ(dict1.get_id(hello), 0);
    EXPECT_EQ(dict1.get_id(world), 1);
    EXPECT_EQ(dict1.get_id(test), 2);
}

TEST_F(StringDictionaryTests, InvalidId) {
    StringDictionary dict(pool.get());

    std::vector<ffx_str_t> strings;
    strings.emplace_back("hello", pool.get());
    dict.build(strings);

    EXPECT_THROW(dict.get_string(100), std::runtime_error);
}

TEST_F(StringDictionaryTests, LookupBeforeFinalize) {
    StringDictionary dict(pool.get());

    dict.add_string(ffx_str_t("hello", pool.get()));

    ffx_str_t hello = ffx_str_t("hello", pool.get());

    // Should throw if not finalized
    EXPECT_THROW(dict.get_id(hello), std::runtime_error);

    dict.finalize();

    // Should work after finalize
    EXPECT_EQ(dict.get_id(hello), 0);
}

TEST_F(StringDictionaryTests, AddAfterFinalize) {
    StringDictionary dict(pool.get());

    dict.add_string(ffx_str_t("hello", pool.get()));
    dict.finalize();

    uint64_t world_id = dict.add_string(ffx_str_t("world", pool.get()));
    EXPECT_EQ(world_id, 1u);
    EXPECT_EQ(dict.get_id(ffx_str_t("world", pool.get())), world_id);

    // Re-adding should return the same ID.
    EXPECT_EQ(dict.add_string(ffx_str_t("world", pool.get())), world_id);
}

TEST_F(StringDictionaryTests, LargeDictionary) {
    StringDictionary dict(pool.get());

    std::vector<ffx_str_t> strings;
    for (int i = 0; i < 1000; i++) {
        std::string s = "string" + std::to_string(i);
        strings.push_back(ffx_str_t(s, pool.get()));
    }

    dict.build(strings);

    EXPECT_EQ(dict.size(), 1000);

    // Test lookups
    for (int i = 0; i < 1000; i++) {
        std::string s = "string" + std::to_string(i);
        ffx_str_t str = ffx_str_t(s, pool.get());
        EXPECT_EQ(dict.get_id(str), static_cast<uint64_t>(i));
        EXPECT_EQ(dict.get_string(static_cast<uint64_t>(i)).to_string(), s);
    }
}

TEST_F(StringDictionaryTests, EmptyDictionary) {
    StringDictionary dict(pool.get());

    std::vector<ffx_str_t> strings;
    dict.build(strings);

    ffx_str_t anything = ffx_str_t("anything", pool.get());

    EXPECT_EQ(dict.size(), 0);
    EXPECT_TRUE(dict.is_finalized());
    EXPECT_EQ(dict.get_id(anything), UINT64_MAX);
}

TEST_F(StringDictionaryTests, DuplicateStringsInBuild) {
    StringDictionary dict(pool.get());

    // Building with duplicates should deduplicate
    std::vector<ffx_str_t> strings;
    strings.emplace_back("hello", pool.get());
    strings.emplace_back("world", pool.get());
    strings.emplace_back("hello", pool.get());
    strings.emplace_back("test", pool.get());
    dict.build(strings);

    ffx_str_t hello = ffx_str_t("hello", pool.get());
    ffx_str_t world = ffx_str_t("world", pool.get());
    ffx_str_t test = ffx_str_t("test", pool.get());

    // Should have 3 unique strings
    EXPECT_EQ(dict.size(), 3);
    EXPECT_EQ(dict.get_id(hello), 0);
    EXPECT_EQ(dict.get_id(world), 1);
    EXPECT_EQ(dict.get_id(test), 2);
}

}// namespace ffx
