#include <gtest/gtest.h>
#include "../src/table/include/cardinality.hpp"
#include "../src/operator/include/query_variable_to_vector.hpp"

using namespace ffx;

// ============================================================================
// Cardinality Parsing Tests
// ============================================================================

TEST(CardinalityTests, ParseOneToOne) {
    EXPECT_EQ(parse_cardinality("1:1"), Cardinality::ONE_TO_ONE);
    EXPECT_EQ(parse_cardinality("[1:1]"), Cardinality::ONE_TO_ONE);
}

TEST(CardinalityTests, ParseManyToOne) {
    EXPECT_EQ(parse_cardinality("n:1"), Cardinality::MANY_TO_ONE);
    EXPECT_EQ(parse_cardinality("[n:1]"), Cardinality::MANY_TO_ONE);
}

TEST(CardinalityTests, ParseOneToMany) {
    EXPECT_EQ(parse_cardinality("1:n"), Cardinality::ONE_TO_MANY);
    EXPECT_EQ(parse_cardinality("[1:n]"), Cardinality::ONE_TO_MANY);
}

TEST(CardinalityTests, ParseManyToMany) {
    EXPECT_EQ(parse_cardinality("m:n"), Cardinality::MANY_TO_MANY);
    EXPECT_EQ(parse_cardinality("[m:n]"), Cardinality::MANY_TO_MANY);
}

TEST(CardinalityTests, ParseEmpty) {
    EXPECT_EQ(parse_cardinality(""), Cardinality::MANY_TO_MANY);
}

TEST(CardinalityTests, ParseInvalid) {
    // Invalid formats should default to MANY_TO_MANY
    EXPECT_EQ(parse_cardinality("invalid"), Cardinality::MANY_TO_MANY);
    EXPECT_EQ(parse_cardinality("1:m"), Cardinality::MANY_TO_MANY);
    EXPECT_EQ(parse_cardinality("n:n"), Cardinality::MANY_TO_MANY);
}

// ============================================================================
// should_share_state Tests
// ============================================================================

TEST(CardinalityTests, ShouldShareState_OneToOne) {
    // 1:1: Always share in both directions
    EXPECT_TRUE(should_share_state(Cardinality::ONE_TO_ONE, true));
    EXPECT_TRUE(should_share_state(Cardinality::ONE_TO_ONE, false));
}

TEST(CardinalityTests, ShouldShareState_ManyToOne) {
    // n:1: Share when forward (each src has exactly 1 dest)
    EXPECT_TRUE(should_share_state(Cardinality::MANY_TO_ONE, true));
    // n:1: Don't share when backward (each dest has many srcs)
    EXPECT_FALSE(should_share_state(Cardinality::MANY_TO_ONE, false));
}

TEST(CardinalityTests, ShouldShareState_OneToMany) {
    // 1:n: Don't share when forward (each src has many dests)
    EXPECT_FALSE(should_share_state(Cardinality::ONE_TO_MANY, true));
    // 1:n: Share when backward (each dest has exactly 1 src)
    EXPECT_TRUE(should_share_state(Cardinality::ONE_TO_MANY, false));
}

TEST(CardinalityTests, ShouldShareState_ManyToMany) {
    // m:n: Never share in either direction
    EXPECT_FALSE(should_share_state(Cardinality::MANY_TO_MANY, true));
    EXPECT_FALSE(should_share_state(Cardinality::MANY_TO_MANY, false));
}

// ============================================================================
// QueryVariableToVectorMap shared-state allocation tests
// ============================================================================

TEST(CardinalityTests, SharedStateSetsIdentityFlag) {
    QueryVariableToVectorMap map;

    auto* a = map.allocate_vector<uint64_t>("a");
    ASSERT_NE(a, nullptr);
    // Chunk owns the state; vectors do not
    EXPECT_FALSE(a->owns_state());
    EXPECT_FALSE(a->has_identity_rle());

    // Allocate b sharing state with a
    auto* b = map.allocate_vector_shared_state<uint64_t>("b", "a");
    ASSERT_NE(b, nullptr);

    // Both vectors point to the same state
    EXPECT_EQ(a->state, b->state);
    // Shared vector must flag identity RLE
    EXPECT_TRUE(b->has_identity_rle());
    // Original owner keeps normal RLE flag
    EXPECT_FALSE(a->has_identity_rle());
}

TEST(CardinalityTests, SharedStateAddsSecondaryToChunk) {
    QueryVariableToVectorMap map;

    auto* a = map.allocate_vector<uint64_t>("a");
    auto* b = map.allocate_vector_shared_state<uint64_t>("b", "a");
    (void)a; (void)b;

    // Both attributes should resolve to the same chunk
    auto* chunk_a = map.get_chunk_for_attr("a");
    auto* chunk_b = map.get_chunk_for_attr("b");
    ASSERT_NE(chunk_a, nullptr);
    ASSERT_NE(chunk_b, nullptr);
    EXPECT_EQ(chunk_a, chunk_b);

    // Primary attr is "a"; "b" should be identity-RLE within the same chunk
    EXPECT_EQ(chunk_a->get_primary_attr(), "a");
    EXPECT_TRUE(chunk_a->has_identity_rle("b"));
    EXPECT_FALSE(chunk_a->has_identity_rle("a"));
}

// ============================================================================
// cardinality_to_string Tests
// ============================================================================

TEST(CardinalityTests, ToString) {
    EXPECT_EQ(cardinality_to_string(Cardinality::ONE_TO_ONE), "1:1");
    EXPECT_EQ(cardinality_to_string(Cardinality::MANY_TO_ONE), "n:1");
    EXPECT_EQ(cardinality_to_string(Cardinality::ONE_TO_MANY), "1:n");
    EXPECT_EQ(cardinality_to_string(Cardinality::MANY_TO_MANY), "m:n");
}

// ============================================================================
// Table Integration Tests
// ============================================================================

TEST(CardinalityTests, TableDefaultCardinality) {
    // When not specified, tables should default to m:n
    // This would require creating a table, which we'll skip for now
    // as it requires file I/O. The default is set in table.hpp.
}

TEST(CardinalityTests, TableShouldShareState) {
    // Test the Table::should_share_state method
    // We would need to create a Table instance, but the constructor
    // requires file paths. For now, we test the standalone function.
    
    // Example scenario: Person -> City (n:1 - many people live in one city)
    // Forward join (person -> city): Each person has exactly 1 city -> SHARE
    // Backward join (city -> person): Each city has many people -> DON'T SHARE
    
    Cardinality person_city = Cardinality::MANY_TO_ONE;
    EXPECT_TRUE(should_share_state(person_city, true));   // person -> city
    EXPECT_FALSE(should_share_state(person_city, false)); // city -> person
    
    // Example scenario: Person -> SSN (1:1 - one person has one SSN)
    // Both directions: SHARE
    Cardinality person_ssn = Cardinality::ONE_TO_ONE;
    EXPECT_TRUE(should_share_state(person_ssn, true));
    EXPECT_TRUE(should_share_state(person_ssn, false));
}

