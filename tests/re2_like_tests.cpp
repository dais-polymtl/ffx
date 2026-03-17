#include "../src/query/include/like_util.hpp"
#include <gtest/gtest.h>
#include <re2/re2.h>

namespace ffx {

TEST(RE2LikeTests, SQLToRegexConversion) {
    EXPECT_EQ(sql_like_to_regex("abc"), "^abc$");
    EXPECT_EQ(sql_like_to_regex("a%b"), "^a.*b$");
    EXPECT_EQ(sql_like_to_regex("a_b"), "^a.b$");
    EXPECT_EQ(sql_like_to_regex("%abc%"), "^.*abc.*$");
    EXPECT_EQ(sql_like_to_regex("a.b+c*"), "^a\\.b\\+c\\*$");
}

TEST(RE2LikeTests, BasicMatching) {
    re2::RE2 re(sql_like_to_regex("abc"));
    EXPECT_TRUE(re2::RE2::FullMatch("abc", re));
    EXPECT_FALSE(re2::RE2::FullMatch("abcd", re));
    EXPECT_FALSE(re2::RE2::FullMatch("ab", re));
}

TEST(RE2LikeTests, WildcardPercent) {
    re2::RE2 re(sql_like_to_regex("a%b"));
    EXPECT_TRUE(re2::RE2::FullMatch("ab", re));
    EXPECT_TRUE(re2::RE2::FullMatch("axb", re));// Corrected from sp
    EXPECT_TRUE(re2::RE2::FullMatch("axxxb", re));
    EXPECT_FALSE(re2::RE2::FullMatch("abc", re));
}

TEST(RE2LikeTests, WildcardUnderscore) {
    re2::RE2 re(sql_like_to_regex("a_b"));
    EXPECT_TRUE(re2::RE2::FullMatch("axb", re));
    EXPECT_FALSE(re2::RE2::FullMatch("ab", re));
    EXPECT_FALSE(re2::RE2::FullMatch("axxb", re));
}

TEST(RE2LikeTests, EscapedChars) {
    re2::RE2 re(sql_like_to_regex("a.b"));
    EXPECT_TRUE(re2::RE2::FullMatch("a.b", re));
    EXPECT_FALSE(re2::RE2::FullMatch("axb", re));
}

}// namespace ffx
