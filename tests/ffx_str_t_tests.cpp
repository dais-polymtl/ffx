#include "../src/table/include/ffx_str_t.hpp"
#include "../src/table/include/string_pool.hpp"
#include <cstring>
#include <gtest/gtest.h>

namespace ffx {

class FfxStrTTests : public ::testing::Test {
protected:
    void SetUp() override { pool = std::make_unique<StringPool>(); }

    void TearDown() override { pool.reset(); }

    std::unique_ptr<StringPool> pool;
};

// Construction Tests
TEST_F(FfxStrTTests, DefaultConstructor) {
    ffx_str_t str;
    EXPECT_EQ(str.size, 0);
    EXPECT_EQ(str.data.ptr, nullptr);
    EXPECT_TRUE(str.empty());
    EXPECT_EQ(str.to_string(), "");
}

TEST_F(FfxStrTTests, ConstructorFromShortString) {
    ffx_str_t str("abc", pool.get());
    EXPECT_EQ(str.size, 3);
    EXPECT_EQ(str.data.ptr, nullptr);
    EXPECT_FALSE(str.empty());
    EXPECT_EQ(str.to_string(), "abc");
    // Compare prefix bytes directly
    EXPECT_EQ(str.prefix[0], 'a');
    EXPECT_EQ(str.prefix[1], 'b');
    EXPECT_EQ(str.prefix[2], 'c');
    EXPECT_EQ(str.prefix[3], '\0');
}

TEST_F(FfxStrTTests, ConstructorFromLongString) {
    ffx_str_t str("abcdefghijklmnop", pool.get());
    EXPECT_EQ(str.size, 16);
    EXPECT_NE(str.data.ptr, nullptr);
    EXPECT_FALSE(str.empty());
    EXPECT_EQ(str.to_string(), "abcdefghijklmnop");
    // Compare prefix bytes directly (prefix may not be null-terminated)
    EXPECT_EQ(str.prefix[0], 'a');
    EXPECT_EQ(str.prefix[1], 'b');
    EXPECT_EQ(str.prefix[2], 'c');
    EXPECT_EQ(str.prefix[3], 'd');
}

TEST_F(FfxStrTTests, ConstructorFromStdString) {
    std::string input = "hello";
    ffx_str_t str(input, pool.get());
    EXPECT_EQ(str.get_size(), 5);
    EXPECT_EQ(str.to_string(), "hello");
}

TEST_F(FfxStrTTests, ConstructorFromEmptyString) {
    ffx_str_t str("", pool.get());
    EXPECT_EQ(str.size, 0);
    EXPECT_EQ(str.data.ptr, nullptr);
    EXPECT_TRUE(str.empty());
}

TEST_F(FfxStrTTests, ConstructorFromNullPointer) {
    ffx_str_t str(nullptr, pool.get());
    EXPECT_EQ(str.size, 0);
    EXPECT_EQ(str.data.ptr, nullptr);
    EXPECT_TRUE(str.empty());
}


TEST_F(FfxStrTTests, ConstructorMaxLength) {
    std::string max_str(FFX_STR_MAX_LENGTH, 'x');
    ffx_str_t str(max_str.c_str(), pool.get());
    EXPECT_EQ(str.size, FFX_STR_MAX_LENGTH);
    EXPECT_NE(str.data.ptr, nullptr);
    EXPECT_EQ(str.to_string(), max_str);
}

// Copy Constructor Tests
TEST_F(FfxStrTTests, DeepCopyConstructorShortString) {
    ffx_str_t original("test", pool.get());
    ffx_str_t copy(original, pool.get());
    EXPECT_EQ(copy.size, original.size);
    EXPECT_EQ(copy.to_string(), original.to_string());
    // Compare prefix bytes directly using memcmp
    EXPECT_EQ(std::memcmp(copy.prefix, original.prefix, 4), 0);
}

TEST_F(FfxStrTTests, DeepCopyConstructorLongString) {
    ffx_str_t original("abcdefghijklmnop", pool.get());
    ffx_str_t copy(original, pool.get());
    EXPECT_EQ(copy.size, original.size);
    EXPECT_EQ(copy.to_string(), original.to_string());
}


// Move Constructor Tests
TEST_F(FfxStrTTests, MoveConstructor) {
    ffx_str_t original("hello", pool.get());
    ffx_str_t moved(std::move(original));
    EXPECT_EQ(moved.size, 5);
    EXPECT_EQ(moved.to_string(), "hello");
    EXPECT_EQ(original.size, 0);
    EXPECT_TRUE(original.empty());
}

// Assignment Operator Tests
TEST_F(FfxStrTTests, CopyAssignment) {
    ffx_str_t str1("first", pool.get());
    ffx_str_t str2("second", pool.get());
    str2.assign(str1, pool.get());// Use assign with pool
    EXPECT_EQ(str2.size, str1.size);
    EXPECT_EQ(str2.to_string(), str1.to_string());
}

// assignment without pool is deleted
#if 0
TEST_F(FfxStrTTests, CopyAssignmentWithoutPool) {
    ffx_str_t str1("first", pool.get());
    ffx_str_t str2("second", pool.get());
    str2 = str1;// Default assignment without pool
    // Should NOT truncate as it's a shallow copy of another ffx_str_t
    EXPECT_EQ(str2.size, 5);
    EXPECT_EQ(str2.to_string(), "first");
}

TEST_F(FfxStrTTests, CopyAssignmentShortString) {
    ffx_str_t str1("test", pool.get());
    ffx_str_t str2("abcd", pool.get());
    str2 = str1;// Short string - should work without pool
    EXPECT_EQ(str2.size, str1.size);
    EXPECT_EQ(str2.to_string(), str1.to_string());
}
#endif

TEST_F(FfxStrTTests, MoveAssignment) {
    ffx_str_t str1("source", pool.get());
    ffx_str_t str2("target", pool.get());
    str2 = std::move(str1);
    EXPECT_EQ(str2.size, 6);
    EXPECT_EQ(str2.to_string(), "source");
    EXPECT_EQ(str1.size, 0);
}

// Self-assignment via reference is deleted
#if 0
TEST_F(FfxStrTTests, SelfAssignment) {
    ffx_str_t str("test", pool.get());
    ffx_str_t& str_ref = str;
    str = str_ref;// Self-assignment via reference
    EXPECT_EQ(str.size, 4);
    EXPECT_EQ(str.to_string(), "test");
}
#endif

// Comparison Operator Tests
TEST_F(FfxStrTTests, EqualityOperator) {
    ffx_str_t str1("hello", pool.get());
    ffx_str_t str2("hello", pool.get());
    EXPECT_TRUE(str1 == str2);
}

TEST_F(FfxStrTTests, InequalityOperator) {
    ffx_str_t str1("hello", pool.get());
    ffx_str_t str2("world", pool.get());
    EXPECT_TRUE(str1 != str2);
}

TEST_F(FfxStrTTests, LessThanOperator) {
    ffx_str_t str1("apple", pool.get());
    ffx_str_t str2("banana", pool.get());
    EXPECT_TRUE(str1 < str2);
    EXPECT_FALSE(str2 < str1);
}

TEST_F(FfxStrTTests, GreaterThanOperator) {
    ffx_str_t str1("zebra", pool.get());
    ffx_str_t str2("apple", pool.get());
    EXPECT_TRUE(str1 > str2);
    EXPECT_FALSE(str2 > str1);
}

TEST_F(FfxStrTTests, LessThanOrEqualOperator) {
    ffx_str_t str1("apple", pool.get());
    ffx_str_t str2("banana", pool.get());
    ffx_str_t str3("apple", pool.get());
    EXPECT_TRUE(str1 <= str2);
    EXPECT_TRUE(str1 <= str3);
    EXPECT_FALSE(str2 <= str1);
}

TEST_F(FfxStrTTests, GreaterThanOrEqualOperator) {
    ffx_str_t str1("zebra", pool.get());
    ffx_str_t str2("apple", pool.get());
    ffx_str_t str3("zebra", pool.get());
    EXPECT_TRUE(str1 >= str2);
    EXPECT_TRUE(str1 >= str3);
    EXPECT_FALSE(str2 >= str1);
}

TEST_F(FfxStrTTests, ComparisonWithEmptyString) {
    ffx_str_t empty("", pool.get());
    ffx_str_t non_empty("test", pool.get());
    EXPECT_TRUE(empty < non_empty);
    EXPECT_TRUE(non_empty > empty);
    EXPECT_TRUE(empty == empty);
}

TEST_F(FfxStrTTests, ComparisonDifferentLengths) {
    ffx_str_t short_str("a", pool.get());
    ffx_str_t long_str("aa", pool.get());
    EXPECT_TRUE(short_str < long_str);
}

TEST_F(FfxStrTTests, ComparisonLexicographicOverridesLength) {
    // Lexicographic: 'x' > 'a', regardless of length
    ffx_str_t xyz("xyz", pool.get());
    ffx_str_t abcd("abcd", pool.get());
    EXPECT_TRUE(xyz > abcd);
    EXPECT_TRUE(abcd < xyz);
}

TEST_F(FfxStrTTests, ComparisonPrefixEqualShorterIsSmaller) {
    ffx_str_t ab("ab", pool.get());
    ffx_str_t abc("abc", pool.get());
    EXPECT_TRUE(ab < abc);
    EXPECT_TRUE(abc > ab);
}

TEST_F(FfxStrTTests, ComparisonLongStringDiffersInTailAfterPrefix) {
    // Same prefix[4] = "abcd", differ after that
    ffx_str_t s1("abcdx", pool.get());
    ffx_str_t s2("abcdy", pool.get());
    EXPECT_TRUE(s1 < s2);
    EXPECT_TRUE(s2 > s1);
}

TEST_F(FfxStrTTests, ComparisonLongStringPrefixEqualDifferentLengths) {
    // One is a strict prefix of the other (after 4-byte prefix)
    ffx_str_t s1("abcdef", pool.get());
    ffx_str_t s2("abcdefg", pool.get());
    EXPECT_TRUE(s1 < s2);
    EXPECT_TRUE(s2 > s1);
}

TEST_F(FfxStrTTests, ComparisonShortDifferentAtThirdChar) {
    // "xyx" vs "xyza" – differ at position 3 vs 4; lexicographically "xyx" < "xyza"
    ffx_str_t s1("xyx", pool.get());
    ffx_str_t s2("xyza", pool.get());
    EXPECT_TRUE(s1 < s2);
    EXPECT_TRUE(s2 > s1);
}

TEST_F(FfxStrTTests, ComparisonInlineVsNonInlineSamePrefix) {
    // 'abcd' is fully in prefix, 'abcde' uses ptr; lexicographically 'abcd' < 'abcde'
    ffx_str_t inline_str("abcd", pool.get());     // inline only
    ffx_str_t non_inline_str("abcde", pool.get());// prefix + ptr
    EXPECT_TRUE(inline_str < non_inline_str);
    EXPECT_TRUE(non_inline_str > inline_str);
}

TEST_F(FfxStrTTests, ComparisonInlineVsNonInlineDifferentAtFirstExtraChar) {
    // Same first 4 chars, differ in the 5th
    ffx_str_t inline_str("abcd", pool.get());
    ffx_str_t non_inline_str("abcex", pool.get());
    // At first differing position: 'd' (100) < 'e' (101), so "abcd" < "abcex"
    EXPECT_TRUE(inline_str < non_inline_str);
    EXPECT_TRUE(non_inline_str > inline_str);
}

TEST_F(FfxStrTTests, ComparisonCaseSensitive) {
    ffx_str_t lower("apple", pool.get());
    ffx_str_t upper("APPLE", pool.get());
    EXPECT_TRUE(upper < lower);// 'A' (65) < 'a' (97) in ASCII
    EXPECT_FALSE(lower < upper);
}

// Hash Function Tests - Testing ffx_str_hash wrapper
TEST_F(FfxStrTTests, HashFunction) {
    ffx_str_hash hasher;
    ffx_str_t str1("test", pool.get());
    ffx_str_t str2("test", pool.get());
    ffx_str_t str3("different", pool.get());

    std::size_t hash1 = hasher(str1);
    std::size_t hash2 = hasher(str2);
    std::size_t hash3 = hasher(str3);

    EXPECT_EQ(hash1, hash2);// Same string should have same hash
    EXPECT_NE(hash1, hash3);// Different strings should have different hashes
}

// Hash Function Tests - Testing direct hash() method
TEST_F(FfxStrTTests, HashMethodDirect) {
    ffx_str_t str1("test", pool.get());
    ffx_str_t str2("test", pool.get());
    ffx_str_t str3("different", pool.get());

    uint64_t hash1 = str1.hash();
    uint64_t hash2 = str2.hash();
    uint64_t hash3 = str3.hash();

    EXPECT_EQ(hash1, hash2);// Same string should have same hash
    EXPECT_NE(hash1, hash3);// Different strings should have different hashes
    EXPECT_NE(hash1, 0ULL); // Hash should be non-zero
    EXPECT_NE(hash3, 0ULL); // Hash should be non-zero
}

TEST_F(FfxStrTTests, HashMethodConsistency) {
    // Test that hash() returns consistent values
    ffx_str_t str("hello", pool.get());
    uint64_t hash1 = str.hash();
    uint64_t hash2 = str.hash();
    uint64_t hash3 = str.hash();

    EXPECT_EQ(hash1, hash2);
    EXPECT_EQ(hash2, hash3);
}

TEST_F(FfxStrTTests, HashMethodShortString) {
    // Test hash for short strings (fits in prefix)
    ffx_str_t str1("abc", pool.get());
    ffx_str_t str2("abc", pool.get());
    ffx_str_t str3("abcd", pool.get());

    EXPECT_EQ(str1.hash(), str2.hash());// Same string
    EXPECT_NE(str1.hash(), str3.hash());// Different strings
}

TEST_F(FfxStrTTests, HashMethodLongString) {
    // Test hash for long strings (uses ptr)
    std::string long_str = "abcdefghijklmnopqrstuvwxyz";
    ffx_str_t str1(long_str.c_str(), pool.get());
    ffx_str_t str2(long_str.c_str(), pool.get());
    std::string different_str = "abcdefghijklmnopqrstuvwxyZ";// Different last char
    ffx_str_t str3(different_str.c_str(), pool.get());

    EXPECT_EQ(str1.hash(), str2.hash());// Same string
    EXPECT_NE(str1.hash(), str3.hash());// Different strings
}

TEST_F(FfxStrTTests, HashMethodEmptyString) {
    ffx_str_t empty("", pool.get());
    uint64_t hash = empty.hash();

    // Hash should be consistent
    EXPECT_EQ(hash, empty.hash());
    EXPECT_NE(hash, 0ULL);// Empty string should have non-zero hash
}

TEST_F(FfxStrTTests, HashMethodNullValue) {
    ffx_str_t null1 = ffx_str_t::null_value();
    ffx_str_t null2 = ffx_str_t::null_value();
    ffx_str_t empty("", pool.get());

    uint64_t null_hash1 = null1.hash();
    uint64_t null_hash2 = null2.hash();
    uint64_t empty_hash = empty.hash();

    EXPECT_EQ(null_hash1, null_hash2);// Same null values should have same hash
    EXPECT_NE(null_hash1, empty_hash);// Null and empty should have different hashes
    EXPECT_NE(null_hash1, 0ULL);      // Null hash should be non-zero
}

TEST_F(FfxStrTTests, HashMethodCollisionResistance) {
    // Test that different strings produce different hashes (basic collision resistance)
    std::vector<std::string> test_strings = {"a",     "b",     "c",     "ab",   "ba",        "abc",   "abcd",
                                             "abcde", "hello", "world", "test", "different", "string"};

    std::vector<uint64_t> hashes;
    for (const auto& s: test_strings) {
        ffx_str_t str(s.c_str(), pool.get());
        hashes.push_back(str.hash());
    }

    // Check for collisions (all hashes should be unique)
    for (size_t i = 0; i < hashes.size(); i++) {
        for (size_t j = i + 1; j < hashes.size(); j++) {
            EXPECT_NE(hashes[i], hashes[j])
                    << "Hash collision detected between '" << test_strings[i] << "' and '" << test_strings[j] << "'";
        }
    }
}

TEST_F(FfxStrTTests, HashFunctionEmptyString) {
    ffx_str_hash hasher;
    ffx_str_t empty("", pool.get());
    std::size_t hash = hasher(empty);
    // Hash should be consistent
    EXPECT_EQ(hash, hasher(empty));
}

TEST_F(FfxStrTTests, HashFunctionLongString) {
    ffx_str_hash hasher;
    std::string long_str(FFX_STR_MAX_LENGTH, 'x');
    ffx_str_t str(long_str.c_str(), pool.get());
    std::size_t hash = hasher(str);
    EXPECT_NE(hash, 0);// Should produce a hash
}

// ToString Tests
TEST_F(FfxStrTTests, ToStringShort) {
    ffx_str_t str("abc", pool.get());
    EXPECT_EQ(str.to_string(), "abc");
}

TEST_F(FfxStrTTests, ToStringLong) {
    std::string input = "abcdefghijklmnop";
    ffx_str_t str(input.c_str(), pool.get());
    EXPECT_EQ(str.to_string(), input);
}

TEST_F(FfxStrTTests, ToStringEmpty) {
    ffx_str_t str("", pool.get());
    EXPECT_EQ(str.to_string(), "");
}

// Edge Cases
TEST_F(FfxStrTTests, StringWithNullBytes) {
    // Note: C strings are null-terminated, so "a\0b" will be truncated at first null byte
    // This is expected behavior
    ffx_str_t str("a\0b", pool.get());
    EXPECT_EQ(str.size, 1);// Truncated at null byte
}

TEST_F(FfxStrTTests, FourCharacterString) {
    // Boundary case: exactly 4 characters
    ffx_str_t str("abcd", pool.get());
    EXPECT_EQ(str.size, 4);
    EXPECT_EQ(str.data.ptr, nullptr);// Should fit in prefix
    EXPECT_EQ(str.to_string(), "abcd");
}

TEST_F(FfxStrTTests, TwelveCharacterString) {
    ffx_str_t str("1234567890ab", pool.get());
    EXPECT_EQ(str.size, 12);
    EXPECT_NE(str.data.ptr, nullptr);// > 4 chars
    EXPECT_EQ(str.to_string(), "1234567890ab");
}

TEST_F(FfxStrTTests, ThirteenCharacterString) {
    ffx_str_t str("1234567890abc", pool.get());
    EXPECT_EQ(str.size, 13);
    EXPECT_NE(str.data.ptr, nullptr);// Should use ptr
    EXPECT_EQ(str.to_string(), "1234567890abc");
    // Verify ptr stores entire string
    EXPECT_EQ(std::string(str.data.ptr, 13), "1234567890abc");
}

TEST_F(FfxStrTTests, FiveCharacterString) {
    // Boundary case: 5 characters (now uses ptr)
    ffx_str_t str("abcde", pool.get());
    EXPECT_EQ(str.size, 5);
    EXPECT_NE(str.data.ptr, nullptr);// > 4 chars
    EXPECT_EQ(str.to_string(), "abcde");
}

TEST_F(FfxStrTTests, MemoryLayout) {
    // Verify memory layout size
    // 4 (size as uint32_t) + 4 (prefix) + 8 (ptr) = 16
    EXPECT_GE(sizeof(ffx_str_t), 16);
    EXPECT_LE(sizeof(ffx_str_t), 24);// Likely padded for alignment
}

TEST_F(FfxStrTTests, PrefixStorage) {
    ffx_str_t str("xyz", pool.get());
    EXPECT_EQ(str.prefix[0], 'x');
    EXPECT_EQ(str.prefix[1], 'y');
    EXPECT_EQ(str.prefix[2], 'z');
    EXPECT_EQ(str.prefix[3], '\0');
}

TEST_F(FfxStrTTests, LongStringPrefix) {
    ffx_str_t str("abcdefgh", pool.get());
    EXPECT_EQ(str.prefix[0], 'a');
    EXPECT_EQ(str.prefix[1], 'b');
    EXPECT_EQ(str.prefix[2], 'c');
    EXPECT_EQ(str.prefix[3], 'd');
    EXPECT_NE(str.data.ptr, nullptr);// size 8 > 4
    EXPECT_EQ(str.to_string(), "abcdefgh");
}

// ==================== Null Value Tests ====================

TEST_F(FfxStrTTests, NullValueFactory) {
    ffx_str_t null_str = ffx_str_t::null_value();
    EXPECT_TRUE(null_str.is_null());
    EXPECT_EQ(null_str.get_size(), 0);
    EXPECT_EQ(null_str.data.ptr, nullptr);
    EXPECT_EQ(null_str.to_string(), "");
}

TEST_F(FfxStrTTests, NullVsEmpty) {
    ffx_str_t null_str = ffx_str_t::null_value();
    ffx_str_t empty_str("", pool.get());

    EXPECT_TRUE(null_str.is_null());
    EXPECT_FALSE(empty_str.is_null());

    EXPECT_FALSE(null_str.empty());// null is NOT considered "empty"
    EXPECT_TRUE(empty_str.empty());// empty string is "empty"

    EXPECT_FALSE(null_str.is_empty_string());
    EXPECT_TRUE(empty_str.is_empty_string());

    // null != empty
    EXPECT_NE(null_str, empty_str);
}

TEST_F(FfxStrTTests, NullComparison) {
    ffx_str_t null1 = ffx_str_t::null_value();
    ffx_str_t null2 = ffx_str_t::null_value();
    ffx_str_t str("test", pool.get());
    ffx_str_t empty_str("", pool.get());

    // null == null
    EXPECT_TRUE(null1 == null2);
    EXPECT_FALSE(null1 != null2);

    // null < non-null (including empty string)
    EXPECT_TRUE(null1 < str);
    EXPECT_TRUE(null1 < empty_str);

    // non-null > null
    EXPECT_TRUE(str > null1);
    EXPECT_TRUE(empty_str > null1);

    // null <= null
    EXPECT_TRUE(null1 <= null2);

    // null >= null
    EXPECT_TRUE(null1 >= null2);

    // null != non-null
    EXPECT_TRUE(null1 != str);
    EXPECT_FALSE(null1 == str);
}

TEST_F(FfxStrTTests, SetAndClearNull) {
    ffx_str_t str("test", pool.get());
    EXPECT_FALSE(str.is_null());
    EXPECT_EQ(str.get_size(), 4);

    // Set null
    str.set_null();
    EXPECT_TRUE(str.is_null());
    EXPECT_EQ(str.get_size(), 0);
    EXPECT_EQ(str.data.ptr, nullptr);

    // Clear null (back to empty, not original value)
    str.clear_null();
    EXPECT_FALSE(str.is_null());
    EXPECT_EQ(str.get_size(), 0);// Size was cleared when set_null was called
}

TEST_F(FfxStrTTests, NullCopyConstructor) {
    ffx_str_t null_str = ffx_str_t::null_value();
    ffx_str_t copy(null_str, pool.get());

    EXPECT_TRUE(copy.is_null());
    EXPECT_EQ(copy, null_str);
}

TEST_F(FfxStrTTests, NullCopyAssignment) {
    ffx_str_t null_str = ffx_str_t::null_value();
    ffx_str_t str("test", pool.get());

    str.assign(null_str, pool.get());
    EXPECT_TRUE(str.is_null());
    EXPECT_EQ(str, null_str);
}

TEST_F(FfxStrTTests, NullMoveConstructor) {
    ffx_str_t null_str = ffx_str_t::null_value();
    ffx_str_t moved(std::move(null_str));

    EXPECT_TRUE(moved.is_null());
    EXPECT_FALSE(null_str.is_null());// Moved-from is reset to empty (not null)
    EXPECT_TRUE(null_str.empty());
}

TEST_F(FfxStrTTests, NullMoveAssignment) {
    ffx_str_t null_str = ffx_str_t::null_value();
    ffx_str_t str("test", pool.get());

    str = std::move(null_str);
    EXPECT_TRUE(str.is_null());
    EXPECT_FALSE(null_str.is_null());// Moved-from is reset to empty
}

TEST_F(FfxStrTTests, NullHash) {
    ffx_str_hash hasher;
    ffx_str_t null1 = ffx_str_t::null_value();
    ffx_str_t null2 = ffx_str_t::null_value();
    ffx_str_t empty_str("", pool.get());

    // Same null values should have same hash
    EXPECT_EQ(hasher(null1), hasher(null2));

    // Null and empty should have different hashes
    EXPECT_NE(hasher(null1), hasher(empty_str));
}

TEST_F(FfxStrTTests, NullFlagPreservedInSize) {
    ffx_str_t null_str = ffx_str_t::null_value();

    // Verify the null flag is set in the size field
    EXPECT_TRUE((null_str.size & ffx_str_t::NULL_FLAG) != 0);
    EXPECT_EQ(null_str.size & ffx_str_t::SIZE_MASK, 0);
}

TEST_F(FfxStrTTests, GetSizeReturnsActualSize) {
    ffx_str_t str("hello", pool.get());
    EXPECT_EQ(str.get_size(), 5);
    EXPECT_FALSE(str.is_null());

    ffx_str_t null_str = ffx_str_t::null_value();
    EXPECT_EQ(null_str.get_size(), 0);
    EXPECT_TRUE(null_str.is_null());
}

TEST_F(FfxStrTTests, SetSizePreservesNullFlag) {
    ffx_str_t str("test", pool.get());

    // Set null, then try to set size
    str.set_null();
    EXPECT_TRUE(str.is_null());

    str.set_size(10);
    EXPECT_TRUE(str.is_null());// Null flag should be preserved
    EXPECT_EQ(str.get_size(), 10);
}

TEST_F(FfxStrTTests, NullptrConstructorCreatesEmptyNotNull) {
    // Passing nullptr to constructor should create empty string, not null
    ffx_str_t str(nullptr, pool.get());
    EXPECT_FALSE(str.is_null());
    EXPECT_TRUE(str.empty());
    EXPECT_TRUE(str.is_empty_string());
    EXPECT_EQ(str.get_size(), 0);
}

TEST_F(FfxStrTTests, NullAssignWithPool) {
    ffx_str_t null_str = ffx_str_t::null_value();
    ffx_str_t str("test", pool.get());

    str.assign(null_str, pool.get());
    EXPECT_TRUE(str.is_null());
}

TEST_F(FfxStrTTests, NullPlaceholder) {
    // Verify the placeholder constant exists
    EXPECT_STREQ(FFX_NULL_PLACEHOLDER, "<NULL>");
}

TEST_F(FfxStrTTests, ToDisplayStringNull) {
    ffx_str_t null_str = ffx_str_t::null_value();

    // to_string() returns empty for null
    EXPECT_EQ(null_str.to_string(), "");

    // to_display_string() returns placeholder for null
    EXPECT_EQ(null_str.to_display_string(), "<NULL>");
}

TEST_F(FfxStrTTests, ToDisplayStringNonNull) {
    ffx_str_t str("hello", pool.get());
    ffx_str_t empty_str("", pool.get());

    // For non-null strings, to_display_string() == to_string()
    EXPECT_EQ(str.to_display_string(), "hello");
    EXPECT_EQ(str.to_display_string(), str.to_string());

    // Empty string (not null) shows empty
    EXPECT_EQ(empty_str.to_display_string(), "");
    EXPECT_EQ(empty_str.to_display_string(), empty_str.to_string());
}

}// namespace ffx
