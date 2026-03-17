#include <gtest/gtest.h>
#include "predicate.hpp"
#include "predicate_parser.hpp"
#include "query.hpp"

using namespace ffx;

// =============================================================================
// Lexer Tests
// =============================================================================

class PredicateLexerTests : public ::testing::Test {};

TEST_F(PredicateLexerTests, EmptyInput) {
    PredicateLexer lexer("");
    auto tokens = lexer.tokenize();
    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0].type, TokenType::END_OF_INPUT);
}

TEST_F(PredicateLexerTests, SinglePredicate) {
    PredicateLexer lexer("EQ(a,5)");
    auto tokens = lexer.tokenize();
    
    ASSERT_EQ(tokens.size(), 7); // EQ ( a , 5 ) END
    EXPECT_EQ(tokens[0].type, TokenType::EQ);
    EXPECT_EQ(tokens[1].type, TokenType::LPAREN);
    EXPECT_EQ(tokens[2].type, TokenType::IDENTIFIER);
    EXPECT_EQ(tokens[2].value, "a");
    EXPECT_EQ(tokens[3].type, TokenType::COMMA);
    EXPECT_EQ(tokens[4].type, TokenType::NUMBER);
    EXPECT_EQ(tokens[4].value, "5");
    EXPECT_EQ(tokens[5].type, TokenType::RPAREN);
    EXPECT_EQ(tokens[6].type, TokenType::END_OF_INPUT);
}

TEST_F(PredicateLexerTests, AllOperators) {
    PredicateLexer lexer("EQ NEQ LT GT LTE GTE");
    auto tokens = lexer.tokenize();
    
    ASSERT_EQ(tokens.size(), 7); // 6 operators + END
    EXPECT_EQ(tokens[0].type, TokenType::EQ);
    EXPECT_EQ(tokens[1].type, TokenType::NEQ);
    EXPECT_EQ(tokens[2].type, TokenType::LT);
    EXPECT_EQ(tokens[3].type, TokenType::GT);
    EXPECT_EQ(tokens[4].type, TokenType::LTE);
    EXPECT_EQ(tokens[5].type, TokenType::GTE);
}

TEST_F(PredicateLexerTests, LowercaseOperators) {
    PredicateLexer lexer("eq neq lt gt lte gte");
    auto tokens = lexer.tokenize();
    
    ASSERT_EQ(tokens.size(), 7);
    EXPECT_EQ(tokens[0].type, TokenType::EQ);
    EXPECT_EQ(tokens[1].type, TokenType::NEQ);
    EXPECT_EQ(tokens[2].type, TokenType::LT);
    EXPECT_EQ(tokens[3].type, TokenType::GT);
    EXPECT_EQ(tokens[4].type, TokenType::LTE);
    EXPECT_EQ(tokens[5].type, TokenType::GTE);
}

TEST_F(PredicateLexerTests, LogicalOperators) {
    PredicateLexer lexer("AND OR and or");
    auto tokens = lexer.tokenize();
    
    ASSERT_EQ(tokens.size(), 5);
    EXPECT_EQ(tokens[0].type, TokenType::AND);
    EXPECT_EQ(tokens[1].type, TokenType::OR);
    EXPECT_EQ(tokens[2].type, TokenType::AND);
    EXPECT_EQ(tokens[3].type, TokenType::OR);
}

TEST_F(PredicateLexerTests, StringLiteral) {
    PredicateLexer lexer("EQ(name,\"hello world\")");
    auto tokens = lexer.tokenize();
    
    ASSERT_EQ(tokens.size(), 7);
    EXPECT_EQ(tokens[4].type, TokenType::STRING);
    EXPECT_EQ(tokens[4].value, "hello world");
}

TEST_F(PredicateLexerTests, StringLiteralWithEscapes) {
    PredicateLexer lexer("EQ(a,\"line1\\nline2\")");
    auto tokens = lexer.tokenize();
    
    EXPECT_EQ(tokens[4].type, TokenType::STRING);
    EXPECT_EQ(tokens[4].value, "line1\nline2");
}

TEST_F(PredicateLexerTests, NegativeNumber) {
    PredicateLexer lexer("GTE(x,-100)");
    auto tokens = lexer.tokenize();
    
    EXPECT_EQ(tokens[4].type, TokenType::NUMBER);
    EXPECT_EQ(tokens[4].value, "-100");
}

TEST_F(PredicateLexerTests, DecimalNumber) {
    PredicateLexer lexer("LT(price,99.99)");
    auto tokens = lexer.tokenize();
    
    EXPECT_EQ(tokens[4].type, TokenType::NUMBER);
    EXPECT_EQ(tokens[4].value, "99.99");
}

TEST_F(PredicateLexerTests, WhitespaceHandling) {
    PredicateLexer lexer("  EQ ( a , 5 )  ");
    auto tokens = lexer.tokenize();
    
    ASSERT_EQ(tokens.size(), 7);
    EXPECT_EQ(tokens[0].type, TokenType::EQ);
}

TEST_F(PredicateLexerTests, UnterminatedString) {
    PredicateLexer lexer("EQ(a,\"unterminated)");
    EXPECT_THROW(lexer.tokenize(), PredicateParseError);
}

TEST_F(PredicateLexerTests, UnexpectedCharacter) {
    PredicateLexer lexer("EQ(a,5) @");
    EXPECT_THROW(lexer.tokenize(), PredicateParseError);
}

// =============================================================================
// Parser Tests - Single Predicates
// =============================================================================

class PredicateParserTests : public ::testing::Test {};

TEST_F(PredicateParserTests, EmptyInput) {
    auto expr = PredicateParser::parse_predicates("");
    EXPECT_FALSE(expr.has_predicates());
    EXPECT_TRUE(expr.empty());
}

TEST_F(PredicateParserTests, SingleScalarPredicate) {
    auto expr = PredicateParser::parse_predicates("EQ(a,5)");
    
    ASSERT_TRUE(expr.has_predicates());
    ASSERT_EQ(expr.groups.size(), 1);
    ASSERT_EQ(expr.groups[0].predicates.size(), 1);
    
    const auto& pred = expr.groups[0].predicates[0];
    EXPECT_EQ(pred.op, PredicateOp::EQ);
    EXPECT_EQ(pred.type, PredicateType::SCALAR);
    EXPECT_EQ(pred.left_attr, "a");
    EXPECT_EQ(pred.scalar_value, "5");
}

TEST_F(PredicateParserTests, SingleAttributePredicate) {
    auto expr = PredicateParser::parse_predicates("NEQ(a,b)");
    
    ASSERT_TRUE(expr.has_predicates());
    ASSERT_EQ(expr.groups.size(), 1);
    ASSERT_EQ(expr.groups[0].predicates.size(), 1);
    
    const auto& pred = expr.groups[0].predicates[0];
    EXPECT_EQ(pred.op, PredicateOp::NEQ);
    EXPECT_EQ(pred.type, PredicateType::ATTRIBUTE);
    EXPECT_EQ(pred.left_attr, "a");
    EXPECT_EQ(pred.right_attr, "b");
}

TEST_F(PredicateParserTests, AllPredicateOperators) {
    std::vector<std::pair<std::string, PredicateOp>> test_cases = {
        {"EQ(x,1)", PredicateOp::EQ},
        {"NEQ(x,1)", PredicateOp::NEQ},
        {"LT(x,1)", PredicateOp::LT},
        {"GT(x,1)", PredicateOp::GT},
        {"LTE(x,1)", PredicateOp::LTE},
        {"GTE(x,1)", PredicateOp::GTE},
    };
    
    for (const auto& [input, expected_op] : test_cases) {
        auto expr = PredicateParser::parse_predicates(input);
        ASSERT_TRUE(expr.has_predicates()) << "Failed for: " << input;
        EXPECT_EQ(expr.groups[0].predicates[0].op, expected_op) << "Failed for: " << input;
    }
}

TEST_F(PredicateParserTests, StringScalarValue) {
    auto expr = PredicateParser::parse_predicates("EQ(name,\"John Doe\")");
    
    ASSERT_TRUE(expr.has_predicates());
    const auto& pred = expr.groups[0].predicates[0];
    EXPECT_EQ(pred.type, PredicateType::SCALAR);
    EXPECT_EQ(pred.scalar_value, "John Doe");
}

TEST_F(PredicateParserTests, LikePredicateParses) {
    auto expr = PredicateParser::parse_predicates("LIKE(name,\"Jo%\")");
    ASSERT_TRUE(expr.has_predicates());
    const auto& pred = expr.groups[0].predicates[0];
    EXPECT_EQ(pred.type, PredicateType::SCALAR);
    EXPECT_EQ(pred.op, PredicateOp::LIKE);
    EXPECT_EQ(pred.left_attr, "name");
    EXPECT_EQ(pred.scalar_value, "Jo%");
}

TEST_F(PredicateParserTests, InPredicateParsesStrings) {
    auto expr = PredicateParser::parse_predicates("IN(tag,\"a\",\"b\",\"c\")");
    ASSERT_TRUE(expr.has_predicates());
    const auto& pred = expr.groups[0].predicates[0];
    EXPECT_EQ(pred.type, PredicateType::SCALAR);
    EXPECT_EQ(pred.op, PredicateOp::IN);
    EXPECT_EQ(pred.left_attr, "tag");
    ASSERT_EQ(pred.scalar_values.size(), 3u);
    EXPECT_EQ(pred.scalar_values[0], "a");
    EXPECT_EQ(pred.scalar_values[1], "b");
    EXPECT_EQ(pred.scalar_values[2], "c");
}

TEST_F(PredicateParserTests, InPredicateParsesNumbers) {
    auto expr = PredicateParser::parse_predicates("IN(x,1,2,3,10)");
    ASSERT_TRUE(expr.has_predicates());
    const auto& pred = expr.groups[0].predicates[0];
    EXPECT_EQ(pred.type, PredicateType::SCALAR);
    EXPECT_EQ(pred.op, PredicateOp::IN);
    EXPECT_EQ(pred.left_attr, "x");
    ASSERT_EQ(pred.scalar_values.size(), 4u);
    EXPECT_EQ(pred.scalar_values[0], "1");
    EXPECT_EQ(pred.scalar_values[1], "2");
    EXPECT_EQ(pred.scalar_values[2], "3");
    EXPECT_EQ(pred.scalar_values[3], "10");
}

// =============================================================================
// Parser Tests - AND Predicates
// =============================================================================

TEST_F(PredicateParserTests, TwoPredicatesWithAND) {
    auto expr = PredicateParser::parse_predicates("GTE(b,10) AND LT(b,100)");
    
    ASSERT_TRUE(expr.has_predicates());
    EXPECT_EQ(expr.top_level_op, LogicalOp::AND);
    ASSERT_EQ(expr.groups.size(), 1);
    ASSERT_EQ(expr.groups[0].predicates.size(), 2);
    
    const auto& pred1 = expr.groups[0].predicates[0];
    EXPECT_EQ(pred1.op, PredicateOp::GTE);
    EXPECT_EQ(pred1.left_attr, "b");
    EXPECT_EQ(pred1.scalar_value, "10");
    
    const auto& pred2 = expr.groups[0].predicates[1];
    EXPECT_EQ(pred2.op, PredicateOp::LT);
    EXPECT_EQ(pred2.left_attr, "b");
    EXPECT_EQ(pred2.scalar_value, "100");
}

TEST_F(PredicateParserTests, ThreePredicatesWithAND) {
    auto expr = PredicateParser::parse_predicates("EQ(a,1) AND EQ(b,2) AND EQ(c,3)");
    
    ASSERT_TRUE(expr.has_predicates());
    EXPECT_EQ(expr.top_level_op, LogicalOp::AND);
    ASSERT_EQ(expr.groups.size(), 1);
    ASSERT_EQ(expr.groups[0].predicates.size(), 3);
}

// =============================================================================
// Parser Tests - OR Predicates
// =============================================================================

TEST_F(PredicateParserTests, TwoPredicatesWithOR) {
    auto expr = PredicateParser::parse_predicates("EQ(a,5) OR EQ(a,10)");
    
    ASSERT_TRUE(expr.has_predicates());
    EXPECT_EQ(expr.top_level_op, LogicalOp::OR);
    ASSERT_EQ(expr.groups.size(), 2);
    
    ASSERT_EQ(expr.groups[0].predicates.size(), 1);
    EXPECT_EQ(expr.groups[0].predicates[0].scalar_value, "5");
    
    ASSERT_EQ(expr.groups[1].predicates.size(), 1);
    EXPECT_EQ(expr.groups[1].predicates[0].scalar_value, "10");
}

TEST_F(PredicateParserTests, ThreePredicatesWithOR) {
    auto expr = PredicateParser::parse_predicates("EQ(x,1) OR EQ(x,2) OR EQ(x,3)");
    
    EXPECT_EQ(expr.top_level_op, LogicalOp::OR);
    ASSERT_EQ(expr.groups.size(), 3);
}

// =============================================================================
// Parser Tests - Mixed AND/OR
// =============================================================================

TEST_F(PredicateParserTests, ANDThenOR) {
    // "A AND B OR C" should be parsed as "(A AND B) OR C"
    auto expr = PredicateParser::parse_predicates("EQ(a,1) AND EQ(b,2) OR EQ(c,3)");
    
    EXPECT_EQ(expr.top_level_op, LogicalOp::OR);
    ASSERT_EQ(expr.groups.size(), 2);
    
    // First group: A AND B
    EXPECT_EQ(expr.groups[0].op, LogicalOp::AND);
    ASSERT_EQ(expr.groups[0].predicates.size(), 2);
    EXPECT_EQ(expr.groups[0].predicates[0].scalar_value, "1");
    EXPECT_EQ(expr.groups[0].predicates[1].scalar_value, "2");
    
    // Second group: C
    ASSERT_EQ(expr.groups[1].predicates.size(), 1);
    EXPECT_EQ(expr.groups[1].predicates[0].scalar_value, "3");
}

TEST_F(PredicateParserTests, ORThenAND) {
    // "A OR B AND C" should be parsed as "A OR (B AND C)"
    auto expr = PredicateParser::parse_predicates("EQ(a,1) OR EQ(b,2) AND EQ(c,3)");
    
    EXPECT_EQ(expr.top_level_op, LogicalOp::OR);
    ASSERT_EQ(expr.groups.size(), 2);
    
    // First group: A
    ASSERT_EQ(expr.groups[0].predicates.size(), 1);
    EXPECT_EQ(expr.groups[0].predicates[0].scalar_value, "1");
    
    // Second group: B AND C
    EXPECT_EQ(expr.groups[1].op, LogicalOp::AND);
    ASSERT_EQ(expr.groups[1].predicates.size(), 2);
}

// =============================================================================
// Parser Tests - Parentheses
// =============================================================================

TEST_F(PredicateParserTests, SimpleParentheses) {
    auto expr = PredicateParser::parse_predicates("(EQ(a,5))");
    
    ASSERT_TRUE(expr.has_predicates());
    ASSERT_EQ(expr.groups[0].predicates.size(), 1);
    EXPECT_EQ(expr.groups[0].predicates[0].scalar_value, "5");
}

// =============================================================================
// Parser Tests - Error Cases
// =============================================================================

TEST_F(PredicateParserTests, MissingLeftParen) {
    EXPECT_THROW(PredicateParser::parse_predicates("EQ a,5)"), PredicateParseError);
}

TEST_F(PredicateParserTests, MissingRightParen) {
    EXPECT_THROW(PredicateParser::parse_predicates("EQ(a,5"), PredicateParseError);
}

TEST_F(PredicateParserTests, MissingComma) {
    EXPECT_THROW(PredicateParser::parse_predicates("EQ(a 5)"), PredicateParseError);
}

TEST_F(PredicateParserTests, MissingSecondArgument) {
    EXPECT_THROW(PredicateParser::parse_predicates("EQ(a,)"), PredicateParseError);
}

TEST_F(PredicateParserTests, InvalidOperator) {
    EXPECT_THROW(PredicateParser::parse_predicates("INVALID(a,5)"), PredicateParseError);
}

TEST_F(PredicateParserTests, UnexpectedToken) {
    EXPECT_THROW(PredicateParser::parse_predicates("EQ(a,5) extra"), PredicateParseError);
}

// =============================================================================
// Parser Tests - NOT EXISTS Predicates
// =============================================================================

TEST_F(PredicateParserTests, NotExistsPredicate) {
    auto expr = PredicateParser::parse_predicates("NOTEXISTS(person1, person3)");
    
    ASSERT_TRUE(expr.has_predicates());
    ASSERT_EQ(expr.groups.size(), 1);
    ASSERT_EQ(expr.groups[0].predicates.size(), 1);
    
    const auto& pred = expr.groups[0].predicates[0];
    EXPECT_EQ(pred.type, PredicateType::NOT_EXISTS);
    EXPECT_EQ(pred.left_attr, "person1");
    EXPECT_EQ(pred.right_attr, "person3");
}

TEST_F(PredicateParserTests, NotExistsAlternativeKeywords) {
    // Test NOT_EXISTS (with underscore)
    auto expr1 = PredicateParser::parse_predicates("NOT_EXISTS(a, b)");
    ASSERT_EQ(expr1.groups[0].predicates[0].type, PredicateType::NOT_EXISTS);
    
    // Test ANTI shorthand
    auto expr2 = PredicateParser::parse_predicates("ANTI(a, b)");
    ASSERT_EQ(expr2.groups[0].predicates[0].type, PredicateType::NOT_EXISTS);
    
    // Test lowercase
    auto expr3 = PredicateParser::parse_predicates("notexists(a, b)");
    ASSERT_EQ(expr3.groups[0].predicates[0].type, PredicateType::NOT_EXISTS);
}

TEST_F(PredicateParserTests, NotExistsWithOtherPredicates) {
    auto expr = PredicateParser::parse_predicates("NEQ(a, 10) AND NOTEXISTS(a, b)");
    
    ASSERT_TRUE(expr.has_predicates());
    ASSERT_EQ(expr.groups.size(), 1);
    ASSERT_EQ(expr.groups[0].predicates.size(), 2);
    
    EXPECT_EQ(expr.groups[0].predicates[0].type, PredicateType::SCALAR);
    EXPECT_EQ(expr.groups[0].predicates[1].type, PredicateType::NOT_EXISTS);
}

// =============================================================================
// Query Tests - Predicates Integration
// =============================================================================

class QueryPredicateTests : public ::testing::Test {};

TEST_F(QueryPredicateTests, QueryWithoutPredicates) {
    Query query("Q(a,b,c) := R(a,b),R(b,c)");
    
    EXPECT_FALSE(query.has_predicates());
    EXPECT_EQ(query.num_rels, 2);
}

TEST_F(QueryPredicateTests, QueryWithSimplePredicate) {
    Query query("Q(a,b) := R(a,b) WHERE EQ(a,5)");
    
    EXPECT_TRUE(query.has_predicates());
    EXPECT_EQ(query.num_rels, 1);
    
    const auto& preds = query.get_predicates();
    ASSERT_EQ(preds.groups.size(), 1);
    ASSERT_EQ(preds.groups[0].predicates.size(), 1);
    EXPECT_EQ(preds.groups[0].predicates[0].left_attr, "a");
}

TEST_F(QueryPredicateTests, QueryWithMultiplePredicates) {
    Query query("Q(a,b,c) := R(a,b),R(b,c) WHERE GTE(a,10) AND LT(b,100)");
    
    EXPECT_TRUE(query.has_predicates());
    EXPECT_EQ(query.num_rels, 2);
    
    const auto& preds = query.get_predicates();
    ASSERT_EQ(preds.groups.size(), 1);
    ASSERT_EQ(preds.groups[0].predicates.size(), 2);
}

TEST_F(QueryPredicateTests, QueryWithORPredicates) {
    Query query("Q(a,b) := R(a,b) WHERE EQ(a,1) OR EQ(a,2)");
    
    EXPECT_TRUE(query.has_predicates());
    const auto& preds = query.get_predicates();
    EXPECT_EQ(preds.top_level_op, LogicalOp::OR);
    ASSERT_EQ(preds.groups.size(), 2);
}

TEST_F(QueryPredicateTests, QueryWithAttributePredicate) {
    Query query("Q(a,b,c) := R(a,b),R(b,c) WHERE NEQ(a,b)");
    
    EXPECT_TRUE(query.has_predicates());
    EXPECT_TRUE(query.has_attribute_predicates());
    
    auto attr_preds = query.get_predicates().get_attribute_predicates();
    ASSERT_EQ(attr_preds.size(), 1);
    EXPECT_EQ(attr_preds[0]->left_attr, "a");
    EXPECT_EQ(attr_preds[0]->right_attr, "b");
}

TEST_F(QueryPredicateTests, QueryWithMixedPredicates) {
    Query query("Q(a,b,c) := R(a,b),R(b,c) WHERE GTE(a,5) AND NEQ(a,b)");
    
    EXPECT_TRUE(query.has_predicates());
    EXPECT_TRUE(query.has_scalar_predicates_for("a"));
    EXPECT_TRUE(query.has_attribute_predicates());
}

TEST_F(QueryPredicateTests, CaseInsensitiveWHERE) {
    Query query1("Q(a,b) := R(a,b) where EQ(a,5)");
    Query query2("Q(a,b) := R(a,b) WHERE EQ(a,5)");
    Query query3("Q(a,b) := R(a,b) Where EQ(a,5)");
    
    EXPECT_TRUE(query1.has_predicates());
    EXPECT_TRUE(query2.has_predicates());
    EXPECT_TRUE(query3.has_predicates());
}

TEST_F(QueryPredicateTests, PredicateStringMergedIntoRule) {
    Query query("Q(a,b,c) := R(a,b),R(b,c) WHERE EQ(a,5) AND GTE(b,10)");
    
    EXPECT_TRUE(query.has_predicates());
    EXPECT_EQ(query.num_rels, 2);
    
    const auto& preds = query.get_predicates();
    ASSERT_EQ(preds.groups[0].predicates.size(), 2);
}

// =============================================================================
// PredicateExpression Helper Methods Tests
// =============================================================================

class PredicateExpressionTests : public ::testing::Test {};

TEST_F(PredicateExpressionTests, GetScalarPredicatesFor) {
    auto expr = PredicateParser::parse_predicates("GTE(a,10) AND LT(a,100) AND EQ(b,5)");
    
    auto a_preds = expr.get_scalar_predicates_for("a");
    EXPECT_EQ(a_preds.size(), 2);
    
    auto b_preds = expr.get_scalar_predicates_for("b");
    EXPECT_EQ(b_preds.size(), 1);
    
    auto c_preds = expr.get_scalar_predicates_for("c");
    EXPECT_EQ(c_preds.size(), 0);
}

TEST_F(PredicateExpressionTests, GetAttributePredicates) {
    auto expr = PredicateParser::parse_predicates("EQ(a,5) AND NEQ(a,b) AND LT(b,c)");
    
    auto attr_preds = expr.get_attribute_predicates();
    ASSERT_EQ(attr_preds.size(), 2);
    
    EXPECT_EQ(attr_preds[0]->left_attr, "a");
    EXPECT_EQ(attr_preds[0]->right_attr, "b");
    
    EXPECT_EQ(attr_preds[1]->left_attr, "b");
    EXPECT_EQ(attr_preds[1]->right_attr, "c");
}

TEST_F(PredicateExpressionTests, AllPredicates) {
    auto expr = PredicateParser::parse_predicates("EQ(a,1) AND NEQ(b,c) OR GTE(d,10)");
    
    auto all = expr.all_predicates();
    EXPECT_EQ(all.size(), 3);
}


// =============================================================================
// Parser Tests - Parenthesized OR within AND (JOB pattern)
// =============================================================================

TEST_F(PredicateParserTests, ANDWithParenthesizedOR) {
    // Pattern from JOB queries: A AND B AND (C OR D)
    auto expr = PredicateParser::parse_predicates(
        "EQ(a, \"x\") AND NOT_LIKE(b, \"%y%\") AND (LIKE(b, \"%p%\") OR LIKE(b, \"%q%\"))");

    ASSERT_TRUE(expr.has_predicates());
    EXPECT_EQ(expr.top_level_op, LogicalOp::AND);
    ASSERT_EQ(expr.groups.size(), 2);

    // Group 0: AND group with EQ and NOT_LIKE
    EXPECT_EQ(expr.groups[0].op, LogicalOp::AND);
    ASSERT_EQ(expr.groups[0].predicates.size(), 2);
    EXPECT_EQ(expr.groups[0].predicates[0].op, PredicateOp::EQ);
    EXPECT_EQ(expr.groups[0].predicates[0].left_attr, "a");
    EXPECT_EQ(expr.groups[0].predicates[0].scalar_value, "x");
    EXPECT_EQ(expr.groups[0].predicates[1].op, PredicateOp::NOT_LIKE);
    EXPECT_EQ(expr.groups[0].predicates[1].left_attr, "b");

    // Group 1: OR group with two LIKEs
    EXPECT_EQ(expr.groups[1].op, LogicalOp::OR);
    ASSERT_EQ(expr.groups[1].predicates.size(), 2);
    EXPECT_EQ(expr.groups[1].predicates[0].op, PredicateOp::LIKE);
    EXPECT_EQ(expr.groups[1].predicates[0].left_attr, "b");
    EXPECT_EQ(expr.groups[1].predicates[0].scalar_value, "%p%");
    EXPECT_EQ(expr.groups[1].predicates[1].op, PredicateOp::LIKE);
    EXPECT_EQ(expr.groups[1].predicates[1].left_attr, "b");
    EXPECT_EQ(expr.groups[1].predicates[1].scalar_value, "%q%");
}

TEST_F(PredicateParserTests, ParenthesizedORAtStart) {
    // (A OR B) AND C
    auto expr = PredicateParser::parse_predicates(
        "(LIKE(x, \"%a%\") OR LIKE(x, \"%b%\")) AND EQ(y, 1)");

    EXPECT_EQ(expr.top_level_op, LogicalOp::AND);
    ASSERT_EQ(expr.groups.size(), 2);

    EXPECT_EQ(expr.groups[0].op, LogicalOp::OR);
    ASSERT_EQ(expr.groups[0].predicates.size(), 2);

    EXPECT_EQ(expr.groups[1].op, LogicalOp::AND);
    ASSERT_EQ(expr.groups[1].predicates.size(), 1);
    EXPECT_EQ(expr.groups[1].predicates[0].op, PredicateOp::EQ);
}

TEST_F(PredicateParserTests, MultipleParenthesizedORGroups) {
    // A AND (B OR C) AND (D OR E)
    auto expr = PredicateParser::parse_predicates(
        "EQ(a, 1) AND (EQ(b, 2) OR EQ(b, 3)) AND (EQ(c, 4) OR EQ(c, 5))");

    EXPECT_EQ(expr.top_level_op, LogicalOp::AND);
    ASSERT_EQ(expr.groups.size(), 3);

    EXPECT_EQ(expr.groups[0].op, LogicalOp::AND);
    ASSERT_EQ(expr.groups[0].predicates.size(), 1);

    EXPECT_EQ(expr.groups[1].op, LogicalOp::OR);
    ASSERT_EQ(expr.groups[1].predicates.size(), 2);

    EXPECT_EQ(expr.groups[2].op, LogicalOp::OR);
    ASSERT_EQ(expr.groups[2].predicates.size(), 2);
}


// =============================================================================
// Lexer Tests - Symbolic Operators
// =============================================================================

TEST_F(PredicateLexerTests, SymbolicLessThan) {
    PredicateLexer lexer("x < 5");
    auto tokens = lexer.tokenize();
    ASSERT_EQ(tokens.size(), 4); // IDENTIFIER LT NUMBER END
    EXPECT_EQ(tokens[0].type, TokenType::IDENTIFIER);
    EXPECT_EQ(tokens[1].type, TokenType::LT);
    EXPECT_EQ(tokens[2].type, TokenType::NUMBER);
}

TEST_F(PredicateLexerTests, SymbolicGreaterThan) {
    PredicateLexer lexer("x > 10");
    auto tokens = lexer.tokenize();
    ASSERT_EQ(tokens.size(), 4);
    EXPECT_EQ(tokens[1].type, TokenType::GT);
}

TEST_F(PredicateLexerTests, SymbolicLessThanOrEqual) {
    PredicateLexer lexer("x <= 5");
    auto tokens = lexer.tokenize();
    ASSERT_EQ(tokens.size(), 4);
    EXPECT_EQ(tokens[1].type, TokenType::LTE);
}

TEST_F(PredicateLexerTests, SymbolicGreaterThanOrEqual) {
    PredicateLexer lexer("x >= 5");
    auto tokens = lexer.tokenize();
    ASSERT_EQ(tokens.size(), 4);
    EXPECT_EQ(tokens[1].type, TokenType::GTE);
}

TEST_F(PredicateLexerTests, SymbolicEquals) {
    PredicateLexer lexer("x = 5");
    auto tokens = lexer.tokenize();
    ASSERT_EQ(tokens.size(), 4);
    EXPECT_EQ(tokens[1].type, TokenType::EQ);
}

TEST_F(PredicateLexerTests, SymbolicDoubleEquals) {
    PredicateLexer lexer("x == 5");
    auto tokens = lexer.tokenize();
    ASSERT_EQ(tokens.size(), 4);
    EXPECT_EQ(tokens[1].type, TokenType::EQ);
}

TEST_F(PredicateLexerTests, SymbolicNotEquals) {
    PredicateLexer lexer("x != 5");
    auto tokens = lexer.tokenize();
    ASSERT_EQ(tokens.size(), 4);
    EXPECT_EQ(tokens[1].type, TokenType::NEQ);
}

TEST_F(PredicateLexerTests, SymbolicDiamondNotEquals) {
    PredicateLexer lexer("x <> 5");
    auto tokens = lexer.tokenize();
    ASSERT_EQ(tokens.size(), 4);
    EXPECT_EQ(tokens[1].type, TokenType::NEQ);
}

TEST_F(PredicateLexerTests, BangWithoutEquals) {
    PredicateLexer lexer("x ! 5");
    EXPECT_THROW(lexer.tokenize(), PredicateParseError);
}

// =============================================================================
// Parser Tests - Symbolic Infix Predicates
// =============================================================================

TEST_F(PredicateParserTests, InfixScalarLT) {
    auto expr = PredicateParser::parse_predicates("x < 5");
    ASSERT_TRUE(expr.has_predicates());
    const auto& pred = expr.groups[0].predicates[0];
    EXPECT_EQ(pred.op, PredicateOp::LT);
    EXPECT_EQ(pred.type, PredicateType::SCALAR);
    EXPECT_EQ(pred.left_attr, "x");
    EXPECT_EQ(pred.scalar_value, "5");
}

TEST_F(PredicateParserTests, InfixScalarGT) {
    auto expr = PredicateParser::parse_predicates("price > 100");
    ASSERT_TRUE(expr.has_predicates());
    const auto& pred = expr.groups[0].predicates[0];
    EXPECT_EQ(pred.op, PredicateOp::GT);
    EXPECT_EQ(pred.left_attr, "price");
    EXPECT_EQ(pred.scalar_value, "100");
}

TEST_F(PredicateParserTests, InfixScalarLTE) {
    auto expr = PredicateParser::parse_predicates("age <= 30");
    ASSERT_TRUE(expr.has_predicates());
    const auto& pred = expr.groups[0].predicates[0];
    EXPECT_EQ(pred.op, PredicateOp::LTE);
    EXPECT_EQ(pred.scalar_value, "30");
}

TEST_F(PredicateParserTests, InfixScalarGTE) {
    auto expr = PredicateParser::parse_predicates("score >= 90");
    ASSERT_TRUE(expr.has_predicates());
    const auto& pred = expr.groups[0].predicates[0];
    EXPECT_EQ(pred.op, PredicateOp::GTE);
    EXPECT_EQ(pred.scalar_value, "90");
}

TEST_F(PredicateParserTests, InfixScalarEQ) {
    auto expr = PredicateParser::parse_predicates("status = 1");
    ASSERT_TRUE(expr.has_predicates());
    const auto& pred = expr.groups[0].predicates[0];
    EXPECT_EQ(pred.op, PredicateOp::EQ);
    EXPECT_EQ(pred.scalar_value, "1");
}

TEST_F(PredicateParserTests, InfixScalarDoubleEQ) {
    auto expr = PredicateParser::parse_predicates("status == 1");
    ASSERT_TRUE(expr.has_predicates());
    const auto& pred = expr.groups[0].predicates[0];
    EXPECT_EQ(pred.op, PredicateOp::EQ);
    EXPECT_EQ(pred.scalar_value, "1");
}

TEST_F(PredicateParserTests, InfixScalarNEQ) {
    auto expr = PredicateParser::parse_predicates("x != 0");
    ASSERT_TRUE(expr.has_predicates());
    const auto& pred = expr.groups[0].predicates[0];
    EXPECT_EQ(pred.op, PredicateOp::NEQ);
    EXPECT_EQ(pred.scalar_value, "0");
}

TEST_F(PredicateParserTests, InfixScalarDiamondNEQ) {
    auto expr = PredicateParser::parse_predicates("x <> 0");
    ASSERT_TRUE(expr.has_predicates());
    const auto& pred = expr.groups[0].predicates[0];
    EXPECT_EQ(pred.op, PredicateOp::NEQ);
    EXPECT_EQ(pred.scalar_value, "0");
}

TEST_F(PredicateParserTests, InfixAttributePredicate) {
    auto expr = PredicateParser::parse_predicates("a != b");
    ASSERT_TRUE(expr.has_predicates());
    const auto& pred = expr.groups[0].predicates[0];
    EXPECT_EQ(pred.op, PredicateOp::NEQ);
    EXPECT_EQ(pred.type, PredicateType::ATTRIBUTE);
    EXPECT_EQ(pred.left_attr, "a");
    EXPECT_EQ(pred.right_attr, "b");
}

TEST_F(PredicateParserTests, InfixStringValue) {
    auto expr = PredicateParser::parse_predicates("name = \"Alice\"");
    ASSERT_TRUE(expr.has_predicates());
    const auto& pred = expr.groups[0].predicates[0];
    EXPECT_EQ(pred.op, PredicateOp::EQ);
    EXPECT_EQ(pred.type, PredicateType::SCALAR);
    EXPECT_EQ(pred.scalar_value, "Alice");
}

TEST_F(PredicateParserTests, InfixWithAND) {
    auto expr = PredicateParser::parse_predicates("x >= 10 AND x < 100");
    ASSERT_TRUE(expr.has_predicates());
    EXPECT_EQ(expr.top_level_op, LogicalOp::AND);
    ASSERT_EQ(expr.groups.size(), 1);
    ASSERT_EQ(expr.groups[0].predicates.size(), 2);
    EXPECT_EQ(expr.groups[0].predicates[0].op, PredicateOp::GTE);
    EXPECT_EQ(expr.groups[0].predicates[0].scalar_value, "10");
    EXPECT_EQ(expr.groups[0].predicates[1].op, PredicateOp::LT);
    EXPECT_EQ(expr.groups[0].predicates[1].scalar_value, "100");
}

TEST_F(PredicateParserTests, InfixWithOR) {
    auto expr = PredicateParser::parse_predicates("x = 1 OR x = 2");
    EXPECT_EQ(expr.top_level_op, LogicalOp::OR);
    ASSERT_EQ(expr.groups.size(), 2);
    EXPECT_EQ(expr.groups[0].predicates[0].scalar_value, "1");
    EXPECT_EQ(expr.groups[1].predicates[0].scalar_value, "2");
}

TEST_F(PredicateParserTests, MixedInfixAndFunctionCall) {
    auto expr = PredicateParser::parse_predicates("x > 5 AND EQ(y, 10)");
    ASSERT_TRUE(expr.has_predicates());
    ASSERT_EQ(expr.groups.size(), 1);
    ASSERT_EQ(expr.groups[0].predicates.size(), 2);
    EXPECT_EQ(expr.groups[0].predicates[0].op, PredicateOp::GT);
    EXPECT_EQ(expr.groups[0].predicates[0].left_attr, "x");
    EXPECT_EQ(expr.groups[0].predicates[1].op, PredicateOp::EQ);
    EXPECT_EQ(expr.groups[0].predicates[1].left_attr, "y");
}

TEST_F(PredicateParserTests, InfixNegativeNumber) {
    auto expr = PredicateParser::parse_predicates("temp > -10");
    ASSERT_TRUE(expr.has_predicates());
    const auto& pred = expr.groups[0].predicates[0];
    EXPECT_EQ(pred.op, PredicateOp::GT);
    EXPECT_EQ(pred.scalar_value, "-10");
}

TEST_F(PredicateParserTests, InfixDecimalNumber) {
    auto expr = PredicateParser::parse_predicates("price <= 99.99");
    ASSERT_TRUE(expr.has_predicates());
    const auto& pred = expr.groups[0].predicates[0];
    EXPECT_EQ(pred.op, PredicateOp::LTE);
    EXPECT_EQ(pred.scalar_value, "99.99");
}

TEST_F(PredicateParserTests, InfixAllSymbolicOps) {
    std::vector<std::pair<std::string, PredicateOp>> test_cases = {
        {"x < 1", PredicateOp::LT},
        {"x > 1", PredicateOp::GT},
        {"x <= 1", PredicateOp::LTE},
        {"x >= 1", PredicateOp::GTE},
        {"x = 1", PredicateOp::EQ},
        {"x == 1", PredicateOp::EQ},
        {"x != 1", PredicateOp::NEQ},
        {"x <> 1", PredicateOp::NEQ},
    };
    for (const auto& [input, expected_op] : test_cases) {
        auto expr = PredicateParser::parse_predicates(input);
        ASSERT_TRUE(expr.has_predicates()) << "Failed for: " << input;
        EXPECT_EQ(expr.groups[0].predicates[0].op, expected_op) << "Failed for: " << input;
    }
}

TEST_F(PredicateParserTests, InfixWithParenthesizedOR) {
    auto expr = PredicateParser::parse_predicates("a > 5 AND (b = 1 OR b = 2)");
    EXPECT_EQ(expr.top_level_op, LogicalOp::AND);
    ASSERT_EQ(expr.groups.size(), 2);
    EXPECT_EQ(expr.groups[0].predicates[0].op, PredicateOp::GT);
    EXPECT_EQ(expr.groups[1].op, LogicalOp::OR);
    ASSERT_EQ(expr.groups[1].predicates.size(), 2);
}

// Query integration test with symbolic syntax
TEST_F(QueryPredicateTests, QueryWithSymbolicPredicates) {
    Query query("Q(a,b) := R(a,b) WHERE a > 5");
    EXPECT_TRUE(query.has_predicates());
    const auto& preds = query.get_predicates();
    ASSERT_EQ(preds.groups[0].predicates.size(), 1);
    EXPECT_EQ(preds.groups[0].predicates[0].op, PredicateOp::GT);
    EXPECT_EQ(preds.groups[0].predicates[0].left_attr, "a");
    EXPECT_EQ(preds.groups[0].predicates[0].scalar_value, "5");
}

TEST_F(QueryPredicateTests, QueryWithMixedSymbolicAndKeyword) {
    Query query("Q(a,b,c) := R(a,b),R(b,c) WHERE a >= 10 AND LT(b, 100)");
    EXPECT_TRUE(query.has_predicates());
    const auto& preds = query.get_predicates();
    ASSERT_EQ(preds.groups[0].predicates.size(), 2);
    EXPECT_EQ(preds.groups[0].predicates[0].op, PredicateOp::GTE);
    EXPECT_EQ(preds.groups[0].predicates[1].op, PredicateOp::LT);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

