#include "../src/operator/include/predicate/predicate_eval.hpp"
#include "../src/table/include/ffx_str_t.hpp"
#include "../src/table/include/string_dictionary.hpp"
#include "../src/table/include/string_pool.hpp"
#include <gtest/gtest.h>

using namespace ffx;

class PredicateEvalStringTests : public ::testing::Test {
protected:
    void SetUp() override { pool = std::make_unique<StringPool>(); }

    std::unique_ptr<StringPool> pool;

    ffx_str_t make_str(const std::string& s) { return ffx_str_t(s, pool.get()); }
};

TEST_F(PredicateEvalStringTests, StringEQ) {
    ScalarPredicate<ffx_str_t> sp(pred_eq<ffx_str_t>, make_str("hello"), PredicateOp::EQ);
    EXPECT_TRUE(sp.evaluate(make_str("hello")));
    EXPECT_FALSE(sp.evaluate(make_str("world")));
}

TEST_F(PredicateEvalStringTests, StringNEQ) {
    ScalarPredicate<ffx_str_t> sp(pred_neq<ffx_str_t>, make_str("hello"), PredicateOp::NEQ);
    EXPECT_TRUE(sp.evaluate(make_str("world")));
    EXPECT_FALSE(sp.evaluate(make_str("hello")));
}

TEST_F(PredicateEvalStringTests, StringLexicographicComparisons) {
    // GT - lexicographic comparison
    ScalarPredicate<ffx_str_t> gt_sp(pred_true<ffx_str_t>, make_str("50"), PredicateOp::GT);
    EXPECT_TRUE(gt_sp.evaluate(make_str("60")));  // "60" > "50" lexicographically
    EXPECT_TRUE(gt_sp.evaluate(make_str("9")));   // "9" > "50" lexicographically ('9' > '5')
    EXPECT_FALSE(gt_sp.evaluate(make_str("100")));// "100" < "50" lexicographically ('1' < '5')
    EXPECT_FALSE(gt_sp.evaluate(make_str("20"))); // "20" < "50" lexicographically ('2' < '5')

    // LT - lexicographic comparison
    ScalarPredicate<ffx_str_t> lt_sp(pred_true<ffx_str_t>, make_str("50"), PredicateOp::LT);
    EXPECT_TRUE(lt_sp.evaluate(make_str("20"))); // "20" < "50" lexicographically
    EXPECT_TRUE(lt_sp.evaluate(make_str("100")));// "100" < "50" lexicographically ('1' < '5')
    EXPECT_FALSE(lt_sp.evaluate(make_str("60")));// "60" > "50" lexicographically

    // GTE - lexicographic comparison
    ScalarPredicate<ffx_str_t> gte_sp(pred_true<ffx_str_t>, make_str("50"), PredicateOp::GTE);
    EXPECT_TRUE(gte_sp.evaluate(make_str("50"))); // Equal
    EXPECT_TRUE(gte_sp.evaluate(make_str("51"))); // "51" > "50"
    EXPECT_TRUE(gte_sp.evaluate(make_str("9")));  // "9" > "50" ('9' > '5')
    EXPECT_FALSE(gte_sp.evaluate(make_str("49")));// "49" < "50" ('4' < '5')

    // LTE - lexicographic comparison
    ScalarPredicate<ffx_str_t> lte_sp(pred_true<ffx_str_t>, make_str("50"), PredicateOp::LTE);
    EXPECT_TRUE(lte_sp.evaluate(make_str("50"))); // Equal
    EXPECT_TRUE(lte_sp.evaluate(make_str("49"))); // "49" < "50"
    EXPECT_TRUE(lte_sp.evaluate(make_str("100")));// "100" < "50" ('1' < '5')
    EXPECT_FALSE(lte_sp.evaluate(make_str("51")));// "51" > "50"
}

TEST_F(PredicateEvalStringTests, StringIN) {
    ScalarPredicate<ffx_str_t> sp;
    sp.op = PredicateOp::IN;
    // Avoid initializer_list here: it forces copies from const elements which call
    // ffx_str_t(copy, pool=nullptr) for long strings and will assert.
    sp.in_values.clear();
    sp.in_values.reserve(3);
    sp.in_values.push_back(make_str("apple"));
    sp.in_values.push_back(make_str("banana"));
    sp.in_values.push_back(make_str("cherry"));
    std::sort(sp.in_values.begin(), sp.in_values.end());

    EXPECT_TRUE(sp.evaluate(make_str("banana")));
    EXPECT_TRUE(sp.evaluate(make_str("apple")));
    EXPECT_FALSE(sp.evaluate(make_str("orange")));
}

TEST_F(PredicateEvalStringTests, StringLIKE) {
    ScalarPredicate<ffx_str_t> sp;
    sp.op = PredicateOp::LIKE;
    auto compile_like = [&](const ffx_str_t& pattern) {
        sp.compiled_regex = std::make_shared<re2::RE2>(sql_like_to_regex(pattern.to_string()));
        ASSERT_TRUE(sp.compiled_regex->ok());
    };

    // Pattern: %world%
    sp.value = make_str("%world%");
    compile_like(sp.value);
    EXPECT_TRUE(sp.evaluate(make_str("hello world")));
    EXPECT_TRUE(sp.evaluate(make_str("world is big")));
    EXPECT_FALSE(sp.evaluate(make_str("hello")));

    // Pattern: a_c
    sp.value = make_str("a_c");
    compile_like(sp.value);
    EXPECT_TRUE(sp.evaluate(make_str("abc")));
    EXPECT_TRUE(sp.evaluate(make_str("axc")));
    EXPECT_FALSE(sp.evaluate(make_str("ac")));
    EXPECT_FALSE(sp.evaluate(make_str("abbc")));
}


TEST_F(PredicateEvalStringTests, StringBETWEEN) {
    ScalarPredicate<ffx_str_t> sp(make_str("apple"), make_str("cherry"), PredicateOp::BETWEEN);

    EXPECT_TRUE(sp.evaluate(make_str("banana")));
    EXPECT_TRUE(sp.evaluate(make_str("apple")));
    EXPECT_TRUE(sp.evaluate(make_str("cherry")));
    EXPECT_FALSE(sp.evaluate(make_str("aardvark")));
    EXPECT_FALSE(sp.evaluate(make_str("date")));
}

TEST_F(PredicateEvalStringTests, NumericBETWEEN) {
    ScalarPredicate<uint64_t> sp(10, 20, PredicateOp::BETWEEN);

    EXPECT_TRUE(sp.evaluate(15));
    EXPECT_TRUE(sp.evaluate(10));
    EXPECT_TRUE(sp.evaluate(20));
    EXPECT_FALSE(sp.evaluate(5));
    EXPECT_FALSE(sp.evaluate(21));
}

TEST_F(PredicateEvalStringTests, BuildScalarPredicateExpr) {
    PredicateExpression expr;
    PredicateGroup group;
    group.op = LogicalOp::AND;

    // attr1 BETWEEN "10" AND "20"
    Predicate p1(PredicateOp::BETWEEN, "attr1", "10", "20");
    group.predicates.push_back(p1);

    // attr1 IN ("15", "25")
    Predicate p2;
    p2.op = PredicateOp::IN;
    p2.type = PredicateType::SCALAR;
    p2.left_attr = "attr1";
    p2.scalar_values = {"15", "25"};
    group.predicates.push_back(p2);

    expr.groups.push_back(std::move(group));

    auto scalar_expr = build_scalar_predicate_expr<uint64_t>(expr, "attr1", pool.get());
    ASSERT_EQ(scalar_expr.groups.size(), 1);
    ASSERT_EQ(scalar_expr.groups[0].predicates.size(), 2);

    // 15 is BETWEEN 10 and 20 AND is IN (15, 25) -> TRUE
    EXPECT_TRUE(scalar_expr.evaluate(15));
    // 25 is NOT BETWEEN 10 and 20 -> FALSE
    EXPECT_FALSE(scalar_expr.evaluate(25));
    // 12 is BETWEEN 10 and 20 but NOT IN (15, 25) -> FALSE
    EXPECT_FALSE(scalar_expr.evaluate(12));

    // Test with ffx_str_t
    PredicateExpression str_expr;
    PredicateGroup str_group;
    str_group.predicates.push_back(Predicate(PredicateOp::LIKE, "attr2", "%test%"));
    str_expr.groups.push_back(std::move(str_group));

    auto scalar_str_expr = build_scalar_predicate_expr<ffx_str_t>(str_expr, "attr2", pool.get());
    EXPECT_TRUE(scalar_str_expr.evaluate(make_str("this is a test string")));
    EXPECT_FALSE(scalar_str_expr.evaluate(make_str("nothing here")));
}

// =============================================================================
// Tests for completely inlined strings (≤4 chars, no heap pointer)
// =============================================================================

TEST_F(PredicateEvalStringTests, InlinedStringEQ) {
    // Test with 1-4 character strings (fully inlined)
    ScalarPredicate<ffx_str_t> sp1(pred_eq<ffx_str_t>, make_str("a"), PredicateOp::EQ);
    EXPECT_TRUE(sp1.evaluate(make_str("a")));
    EXPECT_FALSE(sp1.evaluate(make_str("b")));

    ScalarPredicate<ffx_str_t> sp2(pred_eq<ffx_str_t>, make_str("ab"), PredicateOp::EQ);
    EXPECT_TRUE(sp2.evaluate(make_str("ab")));
    EXPECT_FALSE(sp2.evaluate(make_str("ba")));

    ScalarPredicate<ffx_str_t> sp3(pred_eq<ffx_str_t>, make_str("abc"), PredicateOp::EQ);
    EXPECT_TRUE(sp3.evaluate(make_str("abc")));
    EXPECT_FALSE(sp3.evaluate(make_str("abd")));

    ScalarPredicate<ffx_str_t> sp4(pred_eq<ffx_str_t>, make_str("abcd"), PredicateOp::EQ);
    EXPECT_TRUE(sp4.evaluate(make_str("abcd")));
    EXPECT_FALSE(sp4.evaluate(make_str("abce")));

    // Empty string
    ScalarPredicate<ffx_str_t> sp0(pred_eq<ffx_str_t>, make_str(""), PredicateOp::EQ);
    EXPECT_TRUE(sp0.evaluate(make_str("")));
    EXPECT_FALSE(sp0.evaluate(make_str("x")));
}

TEST_F(PredicateEvalStringTests, InlinedStringNEQ) {
    ScalarPredicate<ffx_str_t> sp(pred_neq<ffx_str_t>, make_str("abc"), PredicateOp::NEQ);
    EXPECT_TRUE(sp.evaluate(make_str("xyz")));
    EXPECT_TRUE(sp.evaluate(make_str("ab")));// Different length
    EXPECT_FALSE(sp.evaluate(make_str("abc")));
}

TEST_F(PredicateEvalStringTests, InlinedStringIN) {
    ScalarPredicate<ffx_str_t> sp;
    sp.op = PredicateOp::IN;
    // All inlined strings (≤4 chars)
    sp.in_values.clear();
    sp.in_values.reserve(4);
    sp.in_values.push_back(make_str("a"));
    sp.in_values.push_back(make_str("bb"));
    sp.in_values.push_back(make_str("ccc"));
    sp.in_values.push_back(make_str("dddd"));

    EXPECT_TRUE(sp.evaluate(make_str("a")));
    EXPECT_TRUE(sp.evaluate(make_str("bb")));
    EXPECT_TRUE(sp.evaluate(make_str("ccc")));
    EXPECT_TRUE(sp.evaluate(make_str("dddd")));
    EXPECT_FALSE(sp.evaluate(make_str("e")));
    EXPECT_FALSE(sp.evaluate(make_str("ddddd")));// 5 chars - not inlined
}

TEST_F(PredicateEvalStringTests, InlinedStringLIKE) {
    ScalarPredicate<ffx_str_t> sp;
    sp.op = PredicateOp::LIKE;
    auto compile_like = [&](const ffx_str_t& pattern) {
        sp.compiled_regex = std::make_shared<re2::RE2>(sql_like_to_regex(pattern.to_string()));
        ASSERT_TRUE(sp.compiled_regex->ok());
    };

    // Short pattern matching short strings
    sp.value = make_str("a%");
    compile_like(sp.value);
    EXPECT_TRUE(sp.evaluate(make_str("a")));
    EXPECT_TRUE(sp.evaluate(make_str("ab")));
    EXPECT_TRUE(sp.evaluate(make_str("abc")));
    EXPECT_FALSE(sp.evaluate(make_str("ba")));

    // Single character wildcard
    sp.value = make_str("a_c");
    compile_like(sp.value);
    EXPECT_TRUE(sp.evaluate(make_str("abc")));
    EXPECT_TRUE(sp.evaluate(make_str("axc")));
    EXPECT_FALSE(sp.evaluate(make_str("ac")));

    // Exact 4-char match pattern
    sp.value = make_str("test");
    compile_like(sp.value);
    EXPECT_TRUE(sp.evaluate(make_str("test")));
    EXPECT_FALSE(sp.evaluate(make_str("tes")));
}

TEST_F(PredicateEvalStringTests, InlinedStringBETWEEN) {
    // Using short strings for lexicographic BETWEEN
    ScalarPredicate<ffx_str_t> sp(make_str("b"), make_str("d"), PredicateOp::BETWEEN);

    EXPECT_TRUE(sp.evaluate(make_str("b")));
    EXPECT_TRUE(sp.evaluate(make_str("c")));
    EXPECT_TRUE(sp.evaluate(make_str("d")));
    EXPECT_TRUE(sp.evaluate(make_str("cat")));
    EXPECT_FALSE(sp.evaluate(make_str("a")));
    EXPECT_FALSE(sp.evaluate(make_str("e")));
}

TEST_F(PredicateEvalStringTests, InlinedStringComparisons) {
    // GT with inlined strings - lexicographic comparison
    ScalarPredicate<ffx_str_t> gt_sp(pred_true<ffx_str_t>, make_str("5"), PredicateOp::GT);
    EXPECT_TRUE(gt_sp.evaluate(make_str("9"))); // "9" > "5" lexicographically
    EXPECT_TRUE(gt_sp.evaluate(make_str("50")));// "50" > "5" (longer string with same prefix)
    EXPECT_FALSE(gt_sp.evaluate(make_str("3")));// "3" < "5"
    EXPECT_FALSE(gt_sp.evaluate(make_str("4")));// "4" < "5"

    // LT with inlined strings - lexicographic comparison
    ScalarPredicate<ffx_str_t> lt_sp(pred_true<ffx_str_t>, make_str("5"), PredicateOp::LT);
    EXPECT_TRUE(lt_sp.evaluate(make_str("3"))); // "3" < "5"
    EXPECT_TRUE(lt_sp.evaluate(make_str("4"))); // "4" < "5"
    EXPECT_FALSE(lt_sp.evaluate(make_str("7")));// "7" > "5"
    EXPECT_FALSE(lt_sp.evaluate(make_str("9")));// "9" > "5"

    // GTE with inlined strings - lexicographic comparison
    ScalarPredicate<ffx_str_t> gte_sp(pred_true<ffx_str_t>, make_str("5"), PredicateOp::GTE);
    EXPECT_TRUE(gte_sp.evaluate(make_str("5"))); // Equal
    EXPECT_TRUE(gte_sp.evaluate(make_str("6"))); // "6" > "5"
    EXPECT_TRUE(gte_sp.evaluate(make_str("50")));// "50" > "5" (longer with same prefix)
    EXPECT_FALSE(gte_sp.evaluate(make_str("4")));// "4" < "5"
}

TEST_F(PredicateEvalStringTests, MixedInlinedAndHeapStrings) {
    // Test mixing inlined (≤4) and heap-allocated (>4) strings
    ScalarPredicate<ffx_str_t> sp;
    sp.op = PredicateOp::IN;
    sp.in_values.clear();
    sp.in_values.reserve(4);
    sp.in_values.push_back(make_str("ab"));
    sp.in_values.push_back(make_str("abcde"));
    sp.in_values.push_back(make_str("xy"));
    sp.in_values.push_back(make_str("xyz123"));

    // Inlined matches
    EXPECT_TRUE(sp.evaluate(make_str("ab")));
    EXPECT_TRUE(sp.evaluate(make_str("xy")));
    // Heap-allocated matches
    EXPECT_TRUE(sp.evaluate(make_str("abcde")));
    EXPECT_TRUE(sp.evaluate(make_str("xyz123")));
    // No match
    EXPECT_FALSE(sp.evaluate(make_str("abc")));
    EXPECT_FALSE(sp.evaluate(make_str("zzz")));
}

// =============================================================================
// Tests where predicate value is inlined but evaluated value is not (and vice versa)
// =============================================================================

TEST_F(PredicateEvalStringTests, InlinedPredicateHeapValue_EQ) {
    // Predicate has inlined string (≤4), evaluate against heap string (>4)
    ScalarPredicate<ffx_str_t> sp(pred_eq<ffx_str_t>, make_str("abc"), PredicateOp::EQ);

    // Heap value does NOT match inlined predicate
    EXPECT_FALSE(sp.evaluate(make_str("abcdef")));
    EXPECT_FALSE(sp.evaluate(make_str("abcde")));
    // Different prefix entirely
    EXPECT_FALSE(sp.evaluate(make_str("xyz123")));
}

TEST_F(PredicateEvalStringTests, HeapPredicateInlinedValue_EQ) {
    // Predicate has heap string (>4), evaluate against inlined string (≤4)
    ScalarPredicate<ffx_str_t> sp(pred_eq<ffx_str_t>, make_str("hello"), PredicateOp::EQ);

    // Inlined value does NOT match heap predicate
    EXPECT_FALSE(sp.evaluate(make_str("hel")));
    EXPECT_FALSE(sp.evaluate(make_str("hell")));
    EXPECT_FALSE(sp.evaluate(make_str("h")));
    // Heap value matches
    EXPECT_TRUE(sp.evaluate(make_str("hello")));
}

TEST_F(PredicateEvalStringTests, InlinedPredicateHeapValue_NEQ) {
    // Predicate has inlined string, evaluate against heap string
    ScalarPredicate<ffx_str_t> sp(pred_neq<ffx_str_t>, make_str("ab"), PredicateOp::NEQ);

    // All heap strings are != inlined predicate
    EXPECT_TRUE(sp.evaluate(make_str("abcdef")));
    EXPECT_TRUE(sp.evaluate(make_str("different")));
}

TEST_F(PredicateEvalStringTests, InlinedPredicateHeapValue_BETWEEN) {
    // Inlined bounds, heap value
    ScalarPredicate<ffx_str_t> sp(make_str("b"), make_str("d"), PredicateOp::BETWEEN);

    // Heap strings in range
    EXPECT_TRUE(sp.evaluate(make_str("banana")));
    EXPECT_TRUE(sp.evaluate(make_str("cherry")));
    // Heap strings outside range
    EXPECT_FALSE(sp.evaluate(make_str("apple")));// "apple" < "b"
    EXPECT_FALSE(sp.evaluate(make_str("zebra")));// "zebra" > "d"
}

TEST_F(PredicateEvalStringTests, HeapPredicateInlinedValue_BETWEEN) {
    // Heap bounds, inlined value
    ScalarPredicate<ffx_str_t> sp(make_str("apple"), make_str("cherry"), PredicateOp::BETWEEN);

    // Inlined strings in range
    EXPECT_TRUE(sp.evaluate(make_str("b")));
    EXPECT_TRUE(sp.evaluate(make_str("c")));
    EXPECT_TRUE(sp.evaluate(make_str("ban")));
    // Inlined strings outside range
    EXPECT_FALSE(sp.evaluate(make_str("a")));
    EXPECT_FALSE(sp.evaluate(make_str("z")));
}

TEST_F(PredicateEvalStringTests, InlinedPredicateHeapValue_LT_GT) {
    // Inlined predicate, heap value comparison - lexicographic
    ScalarPredicate<ffx_str_t> gt_sp(pred_true<ffx_str_t>, make_str("50"), PredicateOp::GT);

    // Heap value > inlined predicate (lexicographically)
    EXPECT_TRUE(gt_sp.evaluate(make_str("60")));   // "60" > "50"
    EXPECT_TRUE(gt_sp.evaluate(make_str("9")));    // "9" > "50" ('9' > '5')
    EXPECT_TRUE(gt_sp.evaluate(make_str("99999")));// "99999" > "50" ('9' > '5')
    // Heap value < inlined predicate (lexicographically)
    EXPECT_FALSE(gt_sp.evaluate(make_str("100")));// "100" < "50" ('1' < '5')
    EXPECT_FALSE(gt_sp.evaluate(make_str("49"))); // "49" < "50"

    ScalarPredicate<ffx_str_t> lt_sp(pred_true<ffx_str_t>, make_str("50"), PredicateOp::LT);
    // Heap value < inlined predicate
    EXPECT_TRUE(lt_sp.evaluate(make_str("10"))); // "10" < "50" ('1' < '5')
    EXPECT_TRUE(lt_sp.evaluate(make_str("100")));// "100" < "50" ('1' < '5')
    // Heap value > inlined predicate
    EXPECT_FALSE(lt_sp.evaluate(make_str("60")));// "60" > "50"
}

TEST_F(PredicateEvalStringTests, HeapPredicateInlinedValue_LT_GT) {
    // Heap predicate, inlined value comparison - lexicographic
    ScalarPredicate<ffx_str_t> gt_sp(pred_true<ffx_str_t>, make_str("50000"), PredicateOp::GT);

    // Inlined value vs heap predicate (lexicographically)
    EXPECT_TRUE(gt_sp.evaluate(make_str("6")));   // "6" > "50000" ('6' > '5')
    EXPECT_TRUE(gt_sp.evaluate(make_str("9")));   // "9" > "50000" ('9' > '5')
    EXPECT_FALSE(gt_sp.evaluate(make_str("100")));// "100" < "50000" ('1' < '5')
    EXPECT_FALSE(gt_sp.evaluate(make_str("50"))); // "50" < "50000" (prefix, shorter)

    ScalarPredicate<ffx_str_t> lt_sp(pred_true<ffx_str_t>, make_str("50000"), PredicateOp::LT);
    // Inlined value < heap predicate
    EXPECT_TRUE(lt_sp.evaluate(make_str("100")));// "100" < "50000" ('1' < '5')
    EXPECT_TRUE(lt_sp.evaluate(make_str("50"))); // "50" < "50000" (prefix, shorter)
    EXPECT_TRUE(lt_sp.evaluate(make_str("4")));  // "4" < "50000" ('4' < '5')
    EXPECT_FALSE(lt_sp.evaluate(make_str("6"))); // "6" > "50000" ('6' > '5')
}

TEST_F(PredicateEvalStringTests, InlinedPredicateHeapValue_LIKE) {
    // Short pattern (inlined) matching long string (heap)
    ScalarPredicate<ffx_str_t> sp;
    sp.op = PredicateOp::LIKE;
    auto compile_like = [&](const ffx_str_t& pattern) {
        sp.compiled_regex = std::make_shared<re2::RE2>(sql_like_to_regex(pattern.to_string()));
        ASSERT_TRUE(sp.compiled_regex->ok());
    };

    // Pattern: "a%" (3 chars, inlined)
    sp.value = make_str("a%");
    compile_like(sp.value);
    EXPECT_TRUE(sp.evaluate(make_str("abcdefghij")));// Heap, starts with 'a'
    EXPECT_FALSE(sp.evaluate(make_str("xyzabcdef")));// Heap, doesn't start with 'a'

    // Pattern: "%z" (2 chars, inlined)
    sp.value = make_str("%z");
    compile_like(sp.value);
    EXPECT_TRUE(sp.evaluate(make_str("the quick brown fox jumps over the lazy dog z")));
    EXPECT_FALSE(sp.evaluate(make_str("the quick brown fox")));
}

TEST_F(PredicateEvalStringTests, HeapPredicateInlinedValue_LIKE) {
    // Long pattern (heap) matching short string (inlined)
    ScalarPredicate<ffx_str_t> sp;
    sp.op = PredicateOp::LIKE;
    auto compile_like = [&](const ffx_str_t& pattern) {
        sp.compiled_regex = std::make_shared<re2::RE2>(sql_like_to_regex(pattern.to_string()));
        ASSERT_TRUE(sp.compiled_regex->ok());
    };

    // Pattern: "%abc%" (6 chars, heap)
    sp.value = make_str("%abc%");
    compile_like(sp.value);
    EXPECT_TRUE(sp.evaluate(make_str("abc"))); // Inlined, contains "abc"
    EXPECT_TRUE(sp.evaluate(make_str("xabc")));// Inlined, contains "abc"
    EXPECT_FALSE(sp.evaluate(make_str("ab"))); // Inlined, does NOT contain "abc"
    EXPECT_FALSE(sp.evaluate(make_str("xyz")));

    // Long exact pattern should NOT match short string
    sp.value = make_str("abcdef");
    compile_like(sp.value);
    EXPECT_FALSE(sp.evaluate(make_str("abc")));// Short string can't match longer pattern
    EXPECT_FALSE(sp.evaluate(make_str("abcd")));
}

TEST_F(PredicateEvalStringTests, InlinedPredicateHeapValue_IN) {
    // IN list has inlined strings, evaluate heap string
    ScalarPredicate<ffx_str_t> sp;
    sp.op = PredicateOp::IN;
    sp.in_values.clear();
    sp.in_values.reserve(3);
    sp.in_values.push_back(make_str("a"));
    sp.in_values.push_back(make_str("bb"));
    sp.in_values.push_back(make_str("ccc"));

    // Heap string NOT in inlined list
    EXPECT_FALSE(sp.evaluate(make_str("hello")));
    EXPECT_FALSE(sp.evaluate(make_str("world")));
}

TEST_F(PredicateEvalStringTests, HeapPredicateInlinedValue_IN) {
    // IN list has heap strings, evaluate inlined string
    ScalarPredicate<ffx_str_t> sp;
    sp.op = PredicateOp::IN;
    sp.in_values.clear();
    sp.in_values.reserve(3);
    sp.in_values.push_back(make_str("hello"));
    sp.in_values.push_back(make_str("world"));
    sp.in_values.push_back(make_str("testing"));

    // Inlined string NOT in heap list
    EXPECT_FALSE(sp.evaluate(make_str("abc")));
    EXPECT_FALSE(sp.evaluate(make_str("x")));
}

// =============================================================================
// Tests for Dictionary-based String Predicates (uint64_t with is_string_attr)
// =============================================================================

class StringDictPredicateTests : public ::testing::Test {
protected:
    void SetUp() override {
        pool = std::make_unique<StringPool>();
        dict = std::make_unique<StringDictionary>(pool.get());

        // Add strings to dictionary
        id_hello = dict->add_string(ffx_str_t("hello", pool.get()));
        id_world = dict->add_string(ffx_str_t("world", pool.get()));
        id_production = dict->add_string(ffx_str_t("production companies", pool.get()));
        id_distributor = dict->add_string(ffx_str_t("distributors", pool.get()));
        id_test_note = dict->add_string(ffx_str_t("(as Metro-Goldwyn-Mayer Pictures)", pool.get()));
        id_other_note = dict->add_string(ffx_str_t("(co-production)", pool.get()));

        dict->finalize();
    }

    std::unique_ptr<StringPool> pool;
    std::unique_ptr<StringDictionary> dict;
    uint64_t id_hello, id_world, id_production, id_distributor;
    uint64_t id_test_note, id_other_note;
};

TEST_F(StringDictPredicateTests, StringEQ_DictionaryLookup) {
    ScalarPredicate<ffx_str_t> sp;
    sp.op = PredicateOp::EQ;
    sp.dictionary = dict.get();
    sp.pool = pool.get();
    sp.value = ffx_str_t("hello", pool.get());
    sp.fn = pred_eq<ffx_str_t>;

    EXPECT_TRUE(sp.evaluate_id(id_hello));
    EXPECT_FALSE(sp.evaluate_id(id_world));
    EXPECT_FALSE(sp.evaluate_id(id_production));
}

TEST_F(StringDictPredicateTests, StringNEQ_DictionaryLookup) {
    ScalarPredicate<ffx_str_t> sp;
    sp.op = PredicateOp::NEQ;
    sp.dictionary = dict.get();
    sp.pool = pool.get();
    sp.value = ffx_str_t("hello", pool.get());
    sp.fn = pred_neq<ffx_str_t>;

    EXPECT_FALSE(sp.evaluate_id(id_hello));
    EXPECT_TRUE(sp.evaluate_id(id_world));
    EXPECT_TRUE(sp.evaluate_id(id_production));
}

TEST_F(StringDictPredicateTests, StringIN_DictionaryLookup) {
    ScalarPredicate<ffx_str_t> sp;
    sp.op = PredicateOp::IN;
    sp.dictionary = dict.get();
    sp.pool = pool.get();
    sp.in_values.push_back(ffx_str_t("hello", pool.get()));
    sp.in_values.push_back(ffx_str_t("world", pool.get()));
    std::sort(sp.in_values.begin(), sp.in_values.end());

    EXPECT_TRUE(sp.evaluate_id(id_hello));
    EXPECT_TRUE(sp.evaluate_id(id_world));
    EXPECT_FALSE(sp.evaluate_id(id_production));
    EXPECT_FALSE(sp.evaluate_id(id_distributor));
}

TEST_F(StringDictPredicateTests, StringNOT_IN_DictionaryLookup) {
    ScalarPredicate<ffx_str_t> sp;
    sp.op = PredicateOp::NOT_IN;
    sp.dictionary = dict.get();
    sp.pool = pool.get();
    sp.in_values.push_back(ffx_str_t("hello", pool.get()));
    sp.in_values.push_back(ffx_str_t("world", pool.get()));
    std::sort(sp.in_values.begin(), sp.in_values.end());

    EXPECT_FALSE(sp.evaluate_id(id_hello));
    EXPECT_FALSE(sp.evaluate_id(id_world));
    EXPECT_TRUE(sp.evaluate_id(id_production));
    EXPECT_TRUE(sp.evaluate_id(id_distributor));
}

TEST_F(StringDictPredicateTests, StringLIKE_DictionaryLookup) {
    ScalarPredicate<ffx_str_t> sp;
    sp.op = PredicateOp::LIKE;
    sp.dictionary = dict.get();
    sp.pool = pool.get();
    sp.value = ffx_str_t("%production%", pool.get());

    // Compile the regex
    std::string regex_pattern = sql_like_to_regex("%production%");
    sp.compiled_regex = std::make_shared<re2::RE2>(regex_pattern);

    EXPECT_FALSE(sp.evaluate_id(id_hello));
    EXPECT_TRUE(sp.evaluate_id(id_production));// "production companies" contains "production"
    EXPECT_FALSE(sp.evaluate_id(id_distributor));
}

TEST_F(StringDictPredicateTests, StringNOT_LIKE_DictionaryLookup) {
    ScalarPredicate<ffx_str_t> sp;
    sp.op = PredicateOp::NOT_LIKE;
    sp.dictionary = dict.get();
    sp.pool = pool.get();
    sp.value = ffx_str_t("%(as Metro-Goldwyn-Mayer Pictures)%", pool.get());

    // Compile the regex
    std::string regex_pattern = sql_like_to_regex("%(as Metro-Goldwyn-Mayer Pictures)%");
    sp.compiled_regex = std::make_shared<re2::RE2>(regex_pattern);

    EXPECT_TRUE(sp.evaluate_id(id_hello));     // Doesn't contain the pattern
    EXPECT_TRUE(sp.evaluate_id(id_production));// Doesn't contain the pattern
    EXPECT_FALSE(sp.evaluate_id(id_test_note));// Contains the pattern
    EXPECT_TRUE(sp.evaluate_id(id_other_note));// Doesn't contain the pattern
}

TEST_F(StringDictPredicateTests, NumericPredicateNotAffected) {
    // Basic numeric comparison
    ScalarPredicate<uint64_t> sp;
    sp.op = PredicateOp::EQ;
    sp.value = 42;
    sp.fn = pred_eq<uint64_t>;

    EXPECT_TRUE(sp.evaluate(42));
    EXPECT_FALSE(sp.evaluate(43));
    EXPECT_FALSE(sp.evaluate(0));
}
