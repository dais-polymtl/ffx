#include "../src/table/include/string_hash_table.hpp"
#include "../src/table/include/string_pool.hpp"
#include <gtest/gtest.h>
#include <unordered_set>
#include <vector>

namespace ffx {

class StringHashTableTests : public ::testing::Test {
protected:
    void SetUp() override { pool = std::make_unique<StringPool>(); }

    void TearDown() override { pool.reset(); }

    std::unique_ptr<StringPool> pool;
};

TEST_F(StringHashTableTests, BuildFromFfxStrTPairs) {
    StringHashTable table(pool.get());

    std::vector<std::pair<ffx_str_t, uint64_t>> entries;
    entries.emplace_back(ffx_str_t("hello", pool.get()), 0);
    entries.emplace_back(ffx_str_t("world", pool.get()), 1);
    entries.emplace_back(ffx_str_t("test", pool.get()), 2);

    table.build(entries);

    EXPECT_EQ(table.size(), 3);
    EXPECT_GE(table.capacity(), 3);
}

TEST_F(StringHashTableTests, LookupFfxStrT) {
    StringHashTable table(pool.get());

    std::vector<std::pair<ffx_str_t, uint64_t>> entries;
    entries.emplace_back(ffx_str_t("hello", pool.get()), 0);
    entries.emplace_back(ffx_str_t("world", pool.get()), 1);
    entries.emplace_back(ffx_str_t("test", pool.get()), 2);

    table.build(entries);

    ffx_str_t hello = ffx_str_t("hello", pool.get());
    ffx_str_t world = ffx_str_t("world", pool.get());
    ffx_str_t test = ffx_str_t("test", pool.get());
    ffx_str_t nonexistent = ffx_str_t("nonexistent", pool.get());

    EXPECT_EQ(table.lookup(hello), 0);
    EXPECT_EQ(table.lookup(world), 1);
    EXPECT_EQ(table.lookup(test), 2);
    EXPECT_EQ(table.lookup(nonexistent), UINT64_MAX);
}

TEST_F(StringHashTableTests, Contains) {
    StringHashTable table(pool.get());

    std::vector<std::pair<ffx_str_t, uint64_t>> entries;
    entries.emplace_back(ffx_str_t("hello", pool.get()), 0);
    entries.emplace_back(ffx_str_t("world", pool.get()), 1);

    table.build(entries);

    ffx_str_t hello = ffx_str_t("hello", pool.get());
    ffx_str_t world = ffx_str_t("world", pool.get());
    ffx_str_t test = ffx_str_t("test", pool.get());

    EXPECT_TRUE(table.contains(hello));
    EXPECT_TRUE(table.contains(world));
    EXPECT_FALSE(table.contains(test));
}

TEST_F(StringHashTableTests, EmptyTable) {
    StringHashTable table(pool.get());

    ffx_str_t anything = ffx_str_t("anything", pool.get());

    EXPECT_EQ(table.size(), 0);
    EXPECT_EQ(table.capacity(), 0);
    EXPECT_EQ(table.lookup(anything), UINT64_MAX);
    EXPECT_FALSE(table.contains(anything));
}

TEST_F(StringHashTableTests, LargeTable) {
    StringHashTable table(pool.get());

    std::vector<std::pair<ffx_str_t, uint64_t>> entries;
    for (uint64_t i = 0; i < 1000; i++) {
        std::string s = "string" + std::to_string(i);
        entries.emplace_back(ffx_str_t(s, pool.get()), i);
    }

    table.build(entries);

    EXPECT_EQ(table.size(), 1000);

    // Test lookups
    for (uint64_t i = 0; i < 1000; i++) {
        std::string s = "string" + std::to_string(i);
        ffx_str_t str = ffx_str_t(s, pool.get());
        EXPECT_EQ(table.lookup(str), i);
    }

    ffx_str_t str1000 = ffx_str_t("string1000", pool.get());
    EXPECT_EQ(table.lookup(str1000), UINT64_MAX);
}

TEST_F(StringHashTableTests, CollisionHandling) {
    StringHashTable table(pool.get());

    // Create many entries to force collisions
    std::vector<std::pair<ffx_str_t, uint64_t>> entries;
    for (uint64_t i = 0; i < 100; i++) {
        std::string s = "str" + std::to_string(i);
        entries.emplace_back(ffx_str_t(s, pool.get()), i);
    }

    table.build(entries);

    // All should be retrievable
    for (uint64_t i = 0; i < 100; i++) {
        std::string s = "str" + std::to_string(i);
        ffx_str_t str = ffx_str_t(s, pool.get());
        EXPECT_EQ(table.lookup(str), i) << "Failed lookup for " << s;
    }
}

TEST_F(StringHashTableTests, DuplicateKeys) {
    StringHashTable table(pool.get());

    // Building with duplicate keys should use the last value
    std::vector<std::pair<ffx_str_t, uint64_t>> entries;
    entries.emplace_back(ffx_str_t("hello", pool.get()), 0);
    entries.emplace_back(ffx_str_t("hello", pool.get()), 1);
    entries.emplace_back(ffx_str_t("world", pool.get()), 2);

    table.build(entries);

    // Should find one of the values (implementation dependent)
    ffx_str_t hello_str = ffx_str_t("hello", pool.get());
    ffx_str_t world_str = ffx_str_t("world", pool.get());
    uint64_t id = table.lookup(hello_str);
    EXPECT_TRUE(id == 0 || id == 1);
    EXPECT_EQ(table.lookup(world_str), 2);
}

TEST_F(StringHashTableTests, LongStrings) {
    StringHashTable table(pool.get());

    std::string long_s(100, 'x');
    std::vector<std::pair<ffx_str_t, uint64_t>> entries;
    entries.emplace_back(ffx_str_t(long_s, pool.get()), 0);
    entries.emplace_back(ffx_str_t("short", pool.get()), 1);

    table.build(entries);

    ffx_str_t long_str(long_s, pool.get());
    ffx_str_t short_str("short", pool.get());
    EXPECT_EQ(table.lookup(long_str), 0);
    EXPECT_EQ(table.lookup(short_str), 1);
}

TEST_F(StringHashTableTests, EmptyString) {
    StringHashTable table(pool.get());

    std::vector<std::pair<ffx_str_t, uint64_t>> entries;
    entries.emplace_back(ffx_str_t("", pool.get()), 0);
    entries.emplace_back(ffx_str_t("hello", pool.get()), 1);

    table.build(entries);

    ffx_str_t empty("", pool.get());
    ffx_str_t hello("hello", pool.get());
    EXPECT_EQ(table.lookup(empty), 0);
    EXPECT_EQ(table.lookup(hello), 1);
}

}// namespace ffx
