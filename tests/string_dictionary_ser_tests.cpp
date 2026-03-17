#include "../src/table/include/string_dictionary.hpp"
#include "../src/table/include/string_pool.hpp"
#include <filesystem>
#include <gtest/gtest.h>

namespace ffx {

class StringDictionarySerTests : public ::testing::Test {
protected:
    void SetUp() override {
        pool = std::make_unique<StringPool>();
        test_dir = "/tmp/ffx_string_dict_ser";
        std::filesystem::create_directories(test_dir);
    }

    void TearDown() override {
        if (std::filesystem::exists(test_dir)) { std::filesystem::remove_all(test_dir); }
        pool.reset();
    }

    std::unique_ptr<StringPool> pool;
    std::string test_dir;
};

TEST_F(StringDictionarySerTests, RoundTripSmallDictionary) {
    // Build dictionary
    StringDictionary dict(pool.get());
    dict.add_string(ffx_str_t("hello", pool.get()));
    dict.add_string(ffx_str_t("world", pool.get()));
    dict.add_string(ffx_str_t("test", pool.get()));
    dict.finalize();

    // Serialize to file
    std::string filename = test_dir + "/dictionary.bin";
    {
        std::ofstream out(filename, std::ios::binary);
        ASSERT_TRUE(out.is_open());
        dict.serialize(out);
    }

    ASSERT_TRUE(std::filesystem::exists(filename));

    // Deserialize
    StringDictionary loaded(pool.get());
    {
        std::ifstream in(filename, std::ios::binary);
        ASSERT_TRUE(in.is_open());
        loaded.deserialize(in, pool.get());
    }

    EXPECT_TRUE(loaded.is_finalized());
    EXPECT_EQ(loaded.size(), dict.size());

    // Verify mappings
    ffx_str_t hello("hello", pool.get());
    ffx_str_t world("world", pool.get());
    ffx_str_t test("test", pool.get());

    auto id_hello = loaded.get_id(hello);
    auto id_world = loaded.get_id(world);
    auto id_test = loaded.get_id(test);

    EXPECT_NE(id_hello, UINT64_MAX);
    EXPECT_NE(id_world, UINT64_MAX);
    EXPECT_NE(id_test, UINT64_MAX);

    EXPECT_EQ(loaded.get_string(id_hello).to_string(), "hello");
    EXPECT_EQ(loaded.get_string(id_world).to_string(), "world");
    EXPECT_EQ(loaded.get_string(id_test).to_string(), "test");
}

TEST_F(StringDictionarySerTests, LargeDictionaryRoundTrip) {
    StringDictionary dict(pool.get());
    const int num_strings = 5000;

    for (int i = 0; i < num_strings; ++i) {
        std::string s = "string_value_" + std::to_string(i);
        dict.add_string(ffx_str_t(s, pool.get()));
    }
    dict.finalize();

    std::string filename = test_dir + "/large_dict.bin";
    {
        std::ofstream out(filename, std::ios::binary);
        dict.serialize(out);
    }

    StringDictionary loaded(pool.get());
    {
        std::ifstream in(filename, std::ios::binary);
        loaded.deserialize(in, pool.get());
    }

    EXPECT_EQ(loaded.size(), num_strings);
    for (int i = 0; i < num_strings; ++i) {
        std::string s = "string_value_" + std::to_string(i);
        ffx_str_t str(s, pool.get());
        EXPECT_EQ(loaded.get_id(str), static_cast<uint64_t>(i));
    }
}

TEST_F(StringDictionarySerTests, LongStringRoundTrip) {
    StringDictionary dict(pool.get());

    std::string long_s1(100, 'A');
    std::string long_s2(500, 'B');
    std::string long_s3(1000, 'C');

    dict.add_string(ffx_str_t(long_s1, pool.get()));
    dict.add_string(ffx_str_t(long_s2, pool.get()));
    dict.add_string(ffx_str_t(long_s3, pool.get()));
    dict.finalize();

    std::string filename = test_dir + "/long_strings.bin";
    {
        std::ofstream out(filename, std::ios::binary);
        dict.serialize(out);
    }

    StringDictionary loaded(pool.get());
    {
        std::ifstream in(filename, std::ios::binary);
        loaded.deserialize(in, pool.get());
    }

    EXPECT_EQ(loaded.size(), 3);
    EXPECT_EQ(loaded.get_string(0).to_string(), long_s1);
    EXPECT_EQ(loaded.get_string(1).to_string(), long_s2);
    EXPECT_EQ(loaded.get_string(2).to_string(), long_s3);
}

TEST_F(StringDictionarySerTests, EdgeCaseStrings) {
    StringDictionary dict(pool.get());

    dict.add_string(ffx_str_t("", pool.get()));     // Empty
    dict.add_string(ffx_str_t::null_value());       // Null
    dict.add_string(ffx_str_t("four", pool.get())); // Exactly 4 chars
    dict.add_string(ffx_str_t("five!", pool.get()));// Exactly 5 chars
    dict.finalize();

    std::string filename = test_dir + "/edge_cases.bin";
    {
        std::ofstream out(filename, std::ios::binary);
        dict.serialize(out);
    }

    StringDictionary loaded(pool.get());
    {
        std::ifstream in(filename, std::ios::binary);
        loaded.deserialize(in, pool.get());
    }

    EXPECT_EQ(loaded.size(), 4);
    EXPECT_EQ(loaded.get_string(0).to_string(), "");
    EXPECT_TRUE(loaded.get_string(1).is_null());
    EXPECT_EQ(loaded.get_string(2).to_string(), "four");
    EXPECT_EQ(loaded.get_string(3).to_string(), "five!");
}

}// namespace ffx
