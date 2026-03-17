#include "../src/ser_der/include/serializer.hpp"
#include "../src/table/include/ffx_str_t.hpp"
#include "../src/table/include/string_pool.hpp"
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

namespace ffx {

class SerDesStrTests : public ::testing::Test {
protected:
    void SetUp() override {
        pool = std::make_unique<StringPool>();
        test_dir = "/tmp/ffx_test_serialization";
        // Create test directory
        std::filesystem::create_directories(test_dir);
    }

    void TearDown() override {
        // Clean up test directory
        if (std::filesystem::exists(test_dir)) { std::filesystem::remove_all(test_dir); }
        pool.reset();
    }

    std::unique_ptr<StringPool> pool;
    std::string test_dir;
};

// Basic Serialization Tests
TEST_F(SerDesStrTests, SerializeEmptyString) {
    ffx_str_t str("", pool.get());
    std::string filename = test_dir + "/test_empty.bin";
    std::ofstream out(filename, std::ios::binary);

    serialize_value(str, out);
    out.close();

    // Verify file was created
    EXPECT_TRUE(std::filesystem::exists(filename));

    // Verify file size (4 bytes for size + 4 bytes for prefix = 8 bytes)
    auto file_size = std::filesystem::file_size(filename);
    EXPECT_EQ(file_size, 8);// size (4) + prefix (4)
}

TEST_F(SerDesStrTests, SerializeShortString) {
    ffx_str_t str("abc", pool.get());
    std::string filename = test_dir + "/test_short.bin";
    std::ofstream out(filename, std::ios::binary);

    serialize_value(str, out);
    out.close();

    // Verify file was created
    EXPECT_TRUE(std::filesystem::exists(filename));

    // Verify file size
    auto file_size = std::filesystem::file_size(filename);
    EXPECT_EQ(file_size, 8);// size (4) + prefix (4)
}

TEST_F(SerDesStrTests, SerializeLongString) {
    ffx_str_t str("abcdefghijklmnop", pool.get());
    std::string filename = test_dir + "/test_long.bin";
    std::ofstream out(filename, std::ios::binary);

    serialize_value(str, out);
    out.close();

    // Verify file was created
    EXPECT_TRUE(std::filesystem::exists(filename));

    // Verify file size (4 bytes size + 4 bytes prefix + 12 bytes remaining = 20 bytes)
    auto file_size = std::filesystem::file_size(filename);
    EXPECT_EQ(file_size, 20);// size (4) + prefix (4) + remaining (12)
}

TEST_F(SerDesStrTests, SerializeMaxLengthString) {
    std::string max_str(FFX_STR_MAX_LENGTH, 'x');
    ffx_str_t str(max_str.c_str(), pool.get());
    std::string filename = test_dir + "/test_max.bin";
    std::ofstream out(filename, std::ios::binary);

    serialize_value(str, out);
    out.close();

    // Verify file was created
    EXPECT_TRUE(std::filesystem::exists(filename));

    // Verify file size
    auto file_size = std::filesystem::file_size(filename);
    EXPECT_EQ(file_size, 4 + 4 + (FFX_STR_MAX_LENGTH - 4));// size + prefix + remaining
}

// Basic Deserialization Tests
TEST_F(SerDesStrTests, DeserializeEmptyString) {
    // Serialize first
    ffx_str_t original("", pool.get());
    std::string filename = test_dir + "/test_deser_empty.bin";
    {
        std::ofstream out(filename, std::ios::binary);
        serialize_value(original, out);
    }

    // Deserialize
    std::ifstream in(filename, std::ios::binary);
    StringPool deser_pool;
    ffx_str_t deserialized;
    deserialize_value(deserialized, in, &deser_pool);
    in.close();

    EXPECT_EQ(deserialized.size, 0);
    EXPECT_EQ(deserialized.data.ptr, nullptr);
    EXPECT_TRUE(deserialized.empty());
    EXPECT_EQ(deserialized.to_string(), "");
}

TEST_F(SerDesStrTests, DeserializeShortString) {
    // Serialize first
    ffx_str_t original("test", pool.get());
    std::string filename = test_dir + "/test_deser_short.bin";
    {
        std::ofstream out(filename, std::ios::binary);
        serialize_value(original, out);
    }

    // Deserialize
    std::ifstream in(filename, std::ios::binary);
    StringPool deser_pool;
    ffx_str_t deserialized;
    deserialize_value(deserialized, in, &deser_pool);
    in.close();

    EXPECT_EQ(deserialized.size, 4);
    EXPECT_EQ(deserialized.data.ptr, nullptr);
    EXPECT_EQ(deserialized.to_string(), "test");
    EXPECT_EQ(deserialized.prefix[0], 't');
    EXPECT_EQ(deserialized.prefix[1], 'e');
    EXPECT_EQ(deserialized.prefix[2], 's');
    EXPECT_EQ(deserialized.prefix[3], 't');
}

TEST_F(SerDesStrTests, DeserializeLongString) {
    // Serialize first
    ffx_str_t original("abcdefghijklmnop", pool.get());
    std::string filename = test_dir + "/test_deser_long.bin";
    {
        std::ofstream out(filename, std::ios::binary);
        serialize_value(original, out);
    }

    // Deserialize
    std::ifstream in(filename, std::ios::binary);
    StringPool deser_pool;
    ffx_str_t deserialized;
    deserialize_value(deserialized, in, &deser_pool);
    in.close();

    EXPECT_EQ(deserialized.size, 16);
    EXPECT_NE(deserialized.data.ptr, nullptr);
    EXPECT_EQ(deserialized.to_string(), "abcdefghijklmnop");
    EXPECT_EQ(deserialized.prefix[0], 'a');
    EXPECT_EQ(deserialized.prefix[1], 'b');
    EXPECT_EQ(deserialized.prefix[2], 'c');
    EXPECT_EQ(deserialized.prefix[3], 'd');
}

TEST_F(SerDesStrTests, DeserializeMaxLengthString) {
    // Serialize first
    std::string max_str(FFX_STR_MAX_LENGTH, 'x');
    ffx_str_t original(max_str.c_str(), pool.get());
    std::string filename = test_dir + "/test_deser_max.bin";
    {
        std::ofstream out(filename, std::ios::binary);
        serialize_value(original, out);
    }

    // Deserialize
    std::ifstream in(filename, std::ios::binary);
    StringPool deser_pool;
    ffx_str_t deserialized;
    deserialize_value(deserialized, in, &deser_pool);
    in.close();

    EXPECT_EQ(deserialized.size, FFX_STR_MAX_LENGTH);
    EXPECT_NE(deserialized.data.ptr, nullptr);
    EXPECT_EQ(deserialized.to_string(), max_str);
}

// Round-Trip Tests
TEST_F(SerDesStrTests, RoundTripEmptyString) {
    ffx_str_t original("", pool.get());
    std::string filename = test_dir + "/test_roundtrip_empty.bin";

    // Serialize
    {
        std::ofstream out(filename, std::ios::binary);
        serialize_value(original, out);
    }

    // Deserialize
    StringPool deser_pool;
    ffx_str_t deserialized;
    {
        std::ifstream in(filename, std::ios::binary);
        deserialize_value(deserialized, in, &deser_pool);
    }

    EXPECT_EQ(deserialized, original);
    EXPECT_EQ(deserialized.to_string(), original.to_string());
}

TEST_F(SerDesStrTests, RoundTripShortString) {
    ffx_str_t original("test", pool.get());
    std::string filename = test_dir + "/test_roundtrip_short.bin";

    // Serialize
    {
        std::ofstream out(filename, std::ios::binary);
        serialize_value(original, out);
    }

    // Deserialize
    StringPool deser_pool;
    ffx_str_t deserialized;
    {
        std::ifstream in(filename, std::ios::binary);
        deserialize_value(deserialized, in, &deser_pool);
    }

    EXPECT_EQ(deserialized, original);
    EXPECT_EQ(deserialized.to_string(), original.to_string());
}

TEST_F(SerDesStrTests, SerializeDeserializeAndCompareSingleValue) {
    // Simple sanity check: single value round-trip and direct field comparison
    const char* text = "simple_string";
    ffx_str_t original(text, pool.get());
    std::string filename = test_dir + "/test_single_roundtrip.bin";

    // Serialize
    {
        std::ofstream out(filename, std::ios::binary);
        serialize_value(original, out);
    }

    // Deserialize
    StringPool deser_pool;
    ffx_str_t deserialized;
    {
        std::ifstream in(filename, std::ios::binary);
        deserialize_value(deserialized, in, &deser_pool);
    }

    // Compare values
    EXPECT_EQ(deserialized.size, original.size);
    EXPECT_EQ(deserialized.to_string(), original.to_string());
    EXPECT_TRUE(deserialized == original);
}

TEST_F(SerDesStrTests, RoundTripLongString) {
    ffx_str_t original("abcdefghijklmnopqrstuvwxyz", pool.get());
    std::string filename = test_dir + "/test_roundtrip_long.bin";

    // Serialize
    {
        std::ofstream out(filename, std::ios::binary);
        serialize_value(original, out);
    }

    // Deserialize
    StringPool deser_pool;
    ffx_str_t deserialized;
    {
        std::ifstream in(filename, std::ios::binary);
        deserialize_value(deserialized, in, &deser_pool);
    }

    EXPECT_EQ(deserialized, original);
    EXPECT_EQ(deserialized.to_string(), original.to_string());
}

TEST_F(SerDesStrTests, RoundTripMultipleStrings) {
    std::vector<ffx_str_t> originals;
    originals.push_back(ffx_str_t("short", pool.get()));
    originals.push_back(ffx_str_t("a", pool.get()));
    originals.push_back(ffx_str_t("verylongstringthatexceedsfourchars", pool.get()));
    originals.push_back(ffx_str_t("", pool.get()));

    std::string filename = test_dir + "/test_roundtrip_multiple.bin";

    // Serialize all
    {
        std::ofstream out(filename, std::ios::binary);
        for (const auto& str: originals) {
            serialize_value(str, out);
        }
    }

    // Deserialize all
    StringPool deser_pool;
    std::vector<ffx_str_t> deserialized;
    {
        std::ifstream in(filename, std::ios::binary);
        for (size_t i = 0; i < originals.size(); ++i) {
            ffx_str_t str;
            deserialize_value(str, in, &deser_pool);
            deserialized.push_back(std::move(str));
        }
    }

    EXPECT_EQ(deserialized.size(), originals.size());
    for (size_t i = 0; i < originals.size(); ++i) {
        EXPECT_EQ(deserialized[i], originals[i]);
        EXPECT_EQ(deserialized[i].to_string(), originals[i].to_string());
    }
}

// Type Detection Tests
TEST_F(SerDesStrTests, TypeDetectionNumeric) {
    // Create a numeric file (num_adj_lists.bin without _str suffix)
    std::string filename = test_dir + "/num_adj_lists.bin";
    {
        std::ofstream out(filename, std::ios::binary);
        uint64_t num = 5;
        out.write(reinterpret_cast<const char*>(&num), sizeof(num));
    }

    EXPECT_FALSE(is_string_type(test_dir));
}

TEST_F(SerDesStrTests, TypeDetectionString) {
    // Create a string file (num_adj_lists_str.bin with _str suffix)
    std::string filename = test_dir + "/num_adj_lists_str.bin";
    {
        std::ofstream out(filename, std::ios::binary);
        uint64_t num = 5;
        out.write(reinterpret_cast<const char*>(&num), sizeof(num));
    }

    EXPECT_TRUE(is_string_type(test_dir));
}

TEST_F(SerDesStrTests, TypeDetectionMissingFile) {
    // No files exist
    std::string empty_dir = "/tmp/ffx_test_empty";
    std::filesystem::create_directories(empty_dir);

    EXPECT_FALSE(is_string_type(empty_dir));

    std::filesystem::remove_all(empty_dir);
}

// Edge Cases
TEST_F(SerDesStrTests, SerializeWithDifferentPools) {
    StringPool pool1;
    StringPool pool2;

    ffx_str_t str1("longstring", &pool1);
    ffx_str_t str2("longstring", &pool2);

    std::string filename1 = test_dir + "/test_pool1.bin";
    std::string filename2 = test_dir + "/test_pool2.bin";

    {
        std::ofstream out1(filename1, std::ios::binary);
        serialize_value(str1, out1);
    }

    {
        std::ofstream out2(filename2, std::ios::binary);
        serialize_value(str2, out2);
    }

    // Both should serialize the same
    EXPECT_EQ(std::filesystem::file_size(filename1), std::filesystem::file_size(filename2));
}

TEST_F(SerDesStrTests, DeserializeWithDifferentPool) {
    ffx_str_t original("teststring", pool.get());
    std::string filename = test_dir + "/test_different_pool.bin";

    // Serialize with original pool
    {
        std::ofstream out(filename, std::ios::binary);
        serialize_value(original, out);
    }

    // Deserialize with different pool
    StringPool new_pool;
    ffx_str_t deserialized;
    {
        std::ifstream in(filename, std::ios::binary);
        deserialize_value(deserialized, in, &new_pool);
    }

    EXPECT_EQ(deserialized, original);
    EXPECT_EQ(deserialized.to_string(), original.to_string());
}

TEST_F(SerDesStrTests, FourCharacterBoundary) {
    // Test exactly 4 characters (fits in prefix)
    ffx_str_t str4("abcd", pool.get());
    std::string filename = test_dir + "/test_4chars.bin";

    {
        std::ofstream out(filename, std::ios::binary);
        serialize_value(str4, out);
    }

    StringPool deser_pool;
    ffx_str_t deserialized;
    {
        std::ifstream in(filename, std::ios::binary);
        deserialize_value(deserialized, in, &deser_pool);
    }

    EXPECT_EQ(deserialized, str4);
    EXPECT_EQ(deserialized.size, 4);
    EXPECT_EQ(deserialized.data.ptr, nullptr);
}

TEST_F(SerDesStrTests, FiveCharacterBoundary) {
    // Test exactly 5 characters (needs ptr)
    ffx_str_t str5("abcde", pool.get());
    std::string filename = test_dir + "/test_5chars.bin";

    {
        std::ofstream out(filename, std::ios::binary);
        serialize_value(str5, out);
    }

    StringPool deser_pool;
    ffx_str_t deserialized;
    {
        std::ifstream in(filename, std::ios::binary);
        deserialize_value(deserialized, in, &deser_pool);
    }

    EXPECT_EQ(deserialized, str5);
    EXPECT_EQ(deserialized.size, 5);
    EXPECT_NE(deserialized.data.ptr, nullptr);// > 4 chars
}


TEST_F(SerDesStrTests, TwelveCharacterBoundary) {
    // Test exactly 12 characters (fits in SSO)
    ffx_str_t str12("1234567890ab", pool.get());
    std::string filename = test_dir + "/test_12chars.bin";

    {
        std::ofstream out(filename, std::ios::binary);
        serialize_value(str12, out);
    }

    StringPool deser_pool;
    ffx_str_t deserialized;
    {
        std::ifstream in(filename, std::ios::binary);
        deserialize_value(deserialized, in, &deser_pool);
    }

    EXPECT_EQ(deserialized, str12);
    EXPECT_EQ(deserialized.get_size(), 12);
    EXPECT_NE(deserialized.data.ptr, nullptr);// > 4 chars
}

TEST_F(SerDesStrTests, ThirteenCharacterBoundary) {
    // Test exactly 13 characters (needs ptr)
    ffx_str_t str13("1234567890abc", pool.get());
    std::string filename = test_dir + "/test_13chars.bin";

    {
        std::ofstream out(filename, std::ios::binary);
        serialize_value(str13, out);
    }

    StringPool deser_pool;
    ffx_str_t deserialized;
    {
        std::ifstream in(filename, std::ios::binary);
        deserialize_value(deserialized, in, &deser_pool);
    }

    EXPECT_EQ(deserialized, str13);
    EXPECT_EQ(deserialized.get_size(), 13);
    EXPECT_NE(deserialized.data.ptr, nullptr);
    // Verify ptr stores entire string
    EXPECT_EQ(std::string(deserialized.data.ptr, 13), "1234567890abc");
}

}// namespace ffx
