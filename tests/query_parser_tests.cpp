#include <gtest/gtest.h>
#include "../src/query/include/query.hpp"

class QueryParserTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// ========== Rule syntax required ==========

TEST_F(QueryParserTest, NonRuleQueryRejected) {
    EXPECT_THROW((void)ffx::Query("a->b"), std::invalid_argument);
    EXPECT_THROW((void)ffx::Query("R(a,b),R(b,c)"), std::invalid_argument);
}

TEST_F(QueryParserTest, InvalidRelationMissingParensThrows) {
    EXPECT_THROW((void)ffx::Query("Q(a) := a->b"), std::invalid_argument);
}

TEST_F(QueryParserTest, InvalidRelationMissingParensMultipleThrows) {
    EXPECT_THROW((void)ffx::Query("Q(a,b,c) := a->b,b->c,c->d"), std::invalid_argument);
}

TEST_F(QueryParserTest, InvalidRelationMissingParensWithWhereThrows) {
    EXPECT_THROW((void)ffx::Query("Q(a,b,c) := a->b,b->c WHERE EQ(a,5)"), std::invalid_argument);
}

// ========== Datalog body under projection head ==========

TEST_F(QueryParserTest, DatalogFormatBasic) {
    ffx::Query query("Q(person1, person2) := Person_knows_Person(person1, person2)");

    EXPECT_TRUE(query.is_datalog_format());
    EXPECT_EQ(query.num_rels, 1);
    EXPECT_EQ(query.rels[0].tableName, "Person_knows_Person");
    EXPECT_EQ(query.rels[0].fromVariable, "person1");
    EXPECT_EQ(query.rels[0].toVariable, "person2");
}

TEST_F(QueryParserTest, DatalogFormatMultipleRelations) {
    ffx::Query query("Q(a, b, c) := Table1(a, b), Table2(b, c)");

    EXPECT_TRUE(query.is_datalog_format());
    EXPECT_EQ(query.num_rels, 2);

    EXPECT_EQ(query.rels[0].tableName, "Table1");
    EXPECT_EQ(query.rels[0].fromVariable, "a");
    EXPECT_EQ(query.rels[0].toVariable, "b");

    EXPECT_EQ(query.rels[1].tableName, "Table2");
    EXPECT_EQ(query.rels[1].fromVariable, "b");
    EXPECT_EQ(query.rels[1].toVariable, "c");
}

TEST_F(QueryParserTest, DatalogFormatTriangle) {
    ffx::Query query("Q(p1, p2, p3) := Knows_12(p1, p2), Knows_23(p2, p3), Knows_31(p3, p1)");

    EXPECT_TRUE(query.is_datalog_format());
    EXPECT_EQ(query.num_rels, 3);

    EXPECT_EQ(query.rels[0].tableName, "Knows_12");
    EXPECT_EQ(query.rels[0].fromVariable, "p1");
    EXPECT_EQ(query.rels[0].toVariable, "p2");

    EXPECT_EQ(query.rels[1].tableName, "Knows_23");
    EXPECT_EQ(query.rels[1].fromVariable, "p2");
    EXPECT_EQ(query.rels[1].toVariable, "p3");

    EXPECT_EQ(query.rels[2].tableName, "Knows_31");
    EXPECT_EQ(query.rels[2].fromVariable, "p3");
    EXPECT_EQ(query.rels[2].toVariable, "p1");
}

TEST_F(QueryParserTest, DatalogFormatWithWhere) {
    ffx::Query query("Q(a, b) := Person_knows_Person(a, b) WHERE EQ(a, 5)");

    EXPECT_TRUE(query.is_datalog_format());
    EXPECT_EQ(query.num_rels, 1);
    EXPECT_TRUE(query.has_predicates());
    EXPECT_EQ(query.rels[0].tableName, "Person_knows_Person");
}

TEST_F(QueryParserTest, DatalogFormatWithAttributePredicate) {
    ffx::Query query("Q(a, b, c) := Table1(a, b), Table2(b, c) WHERE NEQ(a, c)");

    EXPECT_TRUE(query.is_datalog_format());
    EXPECT_EQ(query.num_rels, 2);
    EXPECT_TRUE(query.has_predicates());
    EXPECT_TRUE(query.has_attribute_predicates());
}

// ========== Query relation lookup ==========

TEST_F(QueryParserTest, GetQueryRelationDatalogFormat) {
    ffx::Query query("Q(a,b,c) := R(a,b),R(b,c)");

    auto* rel = query.get_query_relation("a", "b");
    ASSERT_NE(rel, nullptr);
    EXPECT_EQ(rel->fromVariable, "a");
    EXPECT_EQ(rel->toVariable, "b");

    auto* rel2 = query.get_query_relation("b", "a");
    ASSERT_NE(rel2, nullptr);
    EXPECT_EQ(rel2->fromVariable, "a");
    EXPECT_EQ(rel2->toVariable, "b");
}

TEST_F(QueryParserTest, GetTableNameDatalogFormat) {
    ffx::Query query("Q(a, b, c) := Person_knows_Person(a, b), Comment_replyOf_Post(b, c)");

    std::string table1 = query.get_table_name("a", "b");
    EXPECT_EQ(table1, "Person_knows_Person");

    std::string table2 = query.get_table_name("b", "c");
    EXPECT_EQ(table2, "Comment_replyOf_Post");
}

TEST_F(QueryParserTest, GetUniqueVariables) {
    ffx::Query query("Q(a, b, c) := Table1(a, b), Table2(b, c), Table3(c, a)");

    auto vars = query.get_unique_query_variables();
    EXPECT_EQ(vars.size(), 3);
    EXPECT_TRUE(vars.count("a") > 0);
    EXPECT_TRUE(vars.count("b") > 0);
    EXPECT_TRUE(vars.count("c") > 0);
}

// ========== IsFwd tests ==========

TEST_F(QueryParserTest, IsFwdTest) {
    ffx::Query query("Q(a,b) := Table1(a, b)");

    EXPECT_TRUE(query.rels[0].isFwd("a", "b"));
    EXPECT_FALSE(query.rels[0].isFwd("b", "a"));
}

// ========== Error cases ==========

TEST_F(QueryParserTest, DatalogMissingParenthesesThrows) {
    EXPECT_THROW(ffx::Query query("Q(a,b) := TableName a, b"), std::invalid_argument);
}

TEST_F(QueryParserTest, DatalogWildcardInvalidThrows) {
    EXPECT_THROW(ffx::Query query("Q(b) := Table(_, b)"), std::runtime_error);
    // Head matches body; relation parser rejects wildcard in position 1 with no further column.
    EXPECT_THROW(ffx::Query query("Q(a) := Table(a, _)"), std::runtime_error);
}

// ========== Query rule syntax Q(...) := ... ==========

TEST_F(QueryParserTest, RuleSyntaxMinHead) {
    ffx::Query q("Q(MIN(a,b,c)) := R(a,b), R(b,c), R(c,d)");
    EXPECT_TRUE(q.is_rule_syntax());
    EXPECT_EQ(q.head_kind(), ffx::QueryHeadKind::Min);
    ASSERT_EQ(q.head_attributes().size(), 3u);
    EXPECT_EQ(q.head_attributes()[0], "a");
    EXPECT_EQ(q.head_attributes()[2], "c");
    EXPECT_TRUE(q.is_datalog_format());
    EXPECT_EQ(q.num_rels, 3u);
    EXPECT_FALSE(q.has_llm_map());
}

TEST_F(QueryParserTest, RuleSyntaxCountStar) {
    ffx::Query q("Q(COUNT(*)) := R(a,b), R(b,c)");
    EXPECT_TRUE(q.is_rule_syntax());
    EXPECT_EQ(q.head_kind(), ffx::QueryHeadKind::CountStar);
    EXPECT_TRUE(q.head_attributes().empty());
    EXPECT_EQ(q.num_rels, 2u);
}

TEST_F(QueryParserTest, RuleSyntaxNoop) {
    ffx::Query q("Q(NOOP) := R(a,b)");
    EXPECT_TRUE(q.is_rule_syntax());
    EXPECT_EQ(q.head_kind(), ffx::QueryHeadKind::Noop);
    EXPECT_EQ(q.num_rels, 1u);
}

TEST_F(QueryParserTest, RuleSyntaxProjectionWithLlm) {
    const char* s = "Q(a,b,c,llm_resp) := R(a,b), R(b,c), llm_resp = LLM_MAP({\"model\":\"x\"})";
    ffx::Query q(s);
    EXPECT_TRUE(q.is_rule_syntax());
    EXPECT_EQ(q.head_kind(), ffx::QueryHeadKind::Projection);
    ASSERT_EQ(q.head_attributes().size(), 4u);
    EXPECT_EQ(q.head_attributes().back(), "llm_resp");
    EXPECT_TRUE(q.has_llm_map());
    EXPECT_EQ(q.get_llm_map_output_attr(), "llm_resp");
    EXPECT_EQ(q.get_llm_map_config(), "{\"model\":\"x\"}");
    EXPECT_EQ(q.num_rels, 2u);
}

TEST_F(QueryParserTest, RuleSyntaxWhereClause) {
    ffx::Query q("Q(MIN(x,y)) := T(x,y), T(y,z) WHERE EQ(x, 1)");
    EXPECT_TRUE(q.has_predicates());
    EXPECT_EQ(q.head_kind(), ffx::QueryHeadKind::Min);
}

TEST_F(QueryParserTest, RuleSyntaxWildcardInBody) {
    ffx::Query q("Q(MIN(a,c)) := T(a,_,c)");
    EXPECT_EQ(q.num_rels, 1u);
    EXPECT_EQ(q.rels[0].fromVariable, "a");
    EXPECT_EQ(q.rels[0].toVariable, "c");
}

TEST_F(QueryParserTest, RuleSyntaxInvalidColonEqualsWithoutQ) {
    EXPECT_THROW(ffx::Query q("a->b := foo"), std::invalid_argument);
}

TEST_F(QueryParserTest, RuleSyntaxHeadVarNotInBody) {
    EXPECT_THROW(ffx::Query q("Q(x) := R(a,b)"), std::invalid_argument);
}

TEST_F(QueryParserTest, RuleSyntaxLlmMissingFromHead) {
    EXPECT_THROW(ffx::Query q("Q(a,b) := R(a,b), llm_resp = LLM_MAP({})"), std::invalid_argument);
}

TEST_F(QueryParserTest, LegacyDatalogWithLlmMapRejected) {
    EXPECT_THROW(ffx::Query q("R(a,b), LLM_MAP({\"model\":\"x\"})"), std::invalid_argument);
}

TEST_F(QueryParserTest, RuleSyntaxTwoLlmRejected) {
    EXPECT_THROW(ffx::Query q("Q(a,l1,l2) := R(a,b), l1 = LLM_MAP({}), l2 = LLM_MAP({})"), std::invalid_argument);
}

TEST_F(QueryParserTest, RuleSyntaxMinEmptyRejected) {
    EXPECT_THROW(ffx::Query q("Q(MIN()) := R(a,b)"), std::invalid_argument);
}

TEST_F(QueryParserTest, RuleSyntaxBodyNotTableFormRejected) {
    EXPECT_THROW(ffx::Query q("Q(a,b) := a->b"), std::invalid_argument);
}

TEST_F(QueryParserTest, RuleSyntaxMissingHead) {
    EXPECT_THROW(ffx::Query q(":= R(a,b)"), std::invalid_argument);
}

TEST_F(QueryParserTest, RuleSyntaxSemicolonRejected) {
    EXPECT_THROW(ffx::Query q("Q(a) := R(a,b); Q(c) := R(c,d)"), std::invalid_argument);
}

TEST_F(QueryParserTest, RuleSyntaxMinWithLlmRejected) {
    EXPECT_THROW(ffx::Query q("Q(MIN(a)) := R(a,b), x = LLM_MAP({})"), std::invalid_argument);
}

TEST_F(QueryParserTest, RequiresMinSink) {
    ffx::Query q("Q(MIN(a)) := R(a,b)");
    EXPECT_TRUE(q.requires_min_sink());
    ffx::Query q2("Q(a,b) := R(a,b)");
    EXPECT_FALSE(q2.requires_min_sink());
}
