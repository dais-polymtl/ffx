#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>

#include "../src/operator/include/factorized_ftree/factorized_tree_element.hpp"
#include "../src/operator/include/join/flat_join.hpp"
#include "../src/operator/include/join/flat_join_predicated.hpp"
#include "../src/operator/include/join/inljoin_packed.hpp"
#include "../src/operator/include/join/inljoin_packed_cascade.hpp"
#include "../src/operator/include/join/inljoin_packed_cascade_predicated.hpp"
#include "../src/operator/include/join/inljoin_packed_cascade_predicated_shared.hpp"
#include "../src/operator/include/join/inljoin_packed_cascade_shared.hpp"
#include "../src/operator/include/join/inljoin_packed_gp_cascade.hpp"
#include "../src/operator/include/join/inljoin_packed_gp_cascade_predicated.hpp"
#include "../src/operator/include/join/inljoin_packed_gp_cascade_predicated_shared.hpp"
#include "../src/operator/include/join/inljoin_packed_gp_cascade_shared.hpp"
#include "../src/operator/include/join/inljoin_packed_predicated.hpp"
#include "../src/operator/include/join/inljoin_packed_predicated_shared.hpp"
#include "../src/operator/include/join/inljoin_packed_shared.hpp"
#include "../src/operator/include/join/intersection.hpp"
#include "../src/operator/include/join/intersection_predicated.hpp"
#include "../src/operator/include/join/nway_intersection.hpp"
#include "../src/operator/include/join/nway_intersection_predicated.hpp"
#include "../src/operator/include/join/packed_theta_join.hpp"
#include "../src/operator/include/predicate/predicate_eval.hpp"
#include "../src/operator/include/query_variable_to_vector.hpp"
#include "../src/operator/include/scan/scan.hpp"
#include "../src/operator/include/scan/scan_predicated.hpp"
#include "../src/operator/include/schema/schema.hpp"
#include "../src/operator/include/sink/sink_packed.hpp"
#include "../src/plan/include/plan_tree.hpp"
#include "../src/query/include/predicate.hpp"
#include "../src/query/include/predicate_parser.hpp"
#include "../src/table/include/adj_list.hpp"
#include "../src/table/include/cardinality.hpp"
#include "../src/table/include/table.hpp"

using namespace ffx;

// =============================================================================
// Test helpers (pipeline introspection)
// =============================================================================

static void assert_theta_joins_come_after_attrs_are_produced(Operator* first_op) {
    std::set<std::string> produced;

    for (Operator* op = first_op; op != nullptr; op = op->next_op) {
        // "Produce" attribute when we see an operator that materializes/creates it.
        if (auto* scan = dynamic_cast<Scan<uint64_t>*>(op)) {
            produced.insert(scan->attribute());
        } else if (auto* scanp = dynamic_cast<ScanPredicated<uint64_t>*>(op)) {
            produced.insert(scanp->attribute());
        } else if (auto* inl = dynamic_cast<INLJoinPacked<uint64_t>*>(op)) {
            produced.insert(inl->output_key());
            produced.insert(inl->join_key());
        } else if (auto* inlp = dynamic_cast<INLJoinPackedPredicated<uint64_t>*>(op)) {
            produced.insert(inlp->output_key());
            produced.insert(inlp->join_key());
        } else if (auto* inls = dynamic_cast<INLJoinPackedShared<uint64_t>*>(op)) {
            produced.insert(inls->output_key());
            produced.insert(inls->join_key());
        } else if (auto* inlps = dynamic_cast<INLJoinPackedPredicatedShared<uint64_t>*>(op)) {
            produced.insert(inlps->output_key());
            produced.insert(inlps->join_key());
        } else if (auto* casc = dynamic_cast<INLJoinPackedCascade<uint64_t>*>(op)) {
            produced.insert(casc->output_key());
            produced.insert(casc->join_key());
        } else if (auto* cascp = dynamic_cast<INLJoinPackedCascadePredicated<uint64_t>*>(op)) {
            produced.insert(cascp->output_key());
            produced.insert(cascp->join_key());
        } else if (auto* cascs = dynamic_cast<INLJoinPackedCascadeShared<uint64_t>*>(op)) {
            produced.insert(cascs->output_key());
            produced.insert(cascs->join_key());
        } else if (auto* cascps = dynamic_cast<INLJoinPackedCascadePredicatedShared<uint64_t>*>(op)) {
            produced.insert(cascps->output_key());
            produced.insert(cascps->join_key());
        } else if (auto* gpc = dynamic_cast<INLJoinPackedGPCascade<uint64_t>*>(op)) {
            produced.insert(gpc->output_key());
            produced.insert(gpc->join_key());
        } else if (auto* gpcp = dynamic_cast<INLJoinPackedGPCascadePredicated<uint64_t>*>(op)) {
            produced.insert(gpcp->output_key());
            produced.insert(gpcp->join_key());
        } else if (auto* gpcs = dynamic_cast<INLJoinPackedGPCascadeShared<uint64_t>*>(op)) {
            produced.insert(gpcs->output_key());
            produced.insert(gpcs->join_key());
        } else if (auto* gpcps = dynamic_cast<INLJoinPackedGPCascadePredicatedShared<uint64_t>*>(op)) {
            produced.insert(gpcps->output_key());
            produced.insert(gpcps->join_key());
        } else if (auto* fj = dynamic_cast<FlatJoin<uint64_t>*>(op)) {
            produced.insert(fj->output_attr());
            produced.insert(fj->parent_attr());
            produced.insert(fj->lca_attr());
        } else if (auto* fjp = dynamic_cast<FlatJoinPredicated<uint64_t>*>(op)) {
            produced.insert(fjp->output_attr());
            produced.insert(fjp->parent_attr());
            produced.insert(fjp->lca_attr());
        } else if (auto* isec = dynamic_cast<Intersection<uint64_t>*>(op)) {
            produced.insert(isec->output_attr());
            produced.insert(isec->ancestor_attr());
            produced.insert(isec->descendant_attr());
        } else if (auto* isecp = dynamic_cast<IntersectionPredicated<uint64_t>*>(op)) {
            produced.insert(isecp->output_attr());
            produced.insert(isecp->ancestor_attr());
            produced.insert(isecp->descendant_attr());
        } else if (auto* nwi = dynamic_cast<NWayIntersection<uint64_t>*>(op)) {
            produced.insert(nwi->output_attr());
            for (const auto& [attr, _dir]: nwi->input_attrs_and_directions()) {
                produced.insert(attr);
            }
        } else if (auto* nwip = dynamic_cast<NWayIntersectionPredicated<uint64_t>*>(op)) {
            produced.insert(nwip->output_attr());
            for (const auto& [attr, _dir]: nwip->input_attrs_and_directions()) {
                produced.insert(attr);
            }
        }

        // Validate theta joins: both attrs must already be produced
        if (auto* theta = dynamic_cast<PackedThetaJoin<uint64_t>*>(op)) {
            ASSERT_TRUE(produced.count(theta->left_attr()) > 0)
                    << "PackedThetaJoin encountered before ancestor attr was produced: " << theta->left_attr();
            ASSERT_TRUE(produced.count(theta->right_attr()) > 0)
                    << "PackedThetaJoin encountered before descendant attr was produced: " << theta->right_attr();
        }
    }
}

static bool is_join_operator(Operator* op) {
    return dynamic_cast<INLJoinPacked<uint64_t>*>(op) != nullptr ||
           dynamic_cast<INLJoinPackedPredicated<uint64_t>*>(op) != nullptr ||
           dynamic_cast<INLJoinPackedShared<uint64_t>*>(op) != nullptr ||
           dynamic_cast<INLJoinPackedPredicatedShared<uint64_t>*>(op) != nullptr ||
           dynamic_cast<INLJoinPackedCascade<uint64_t>*>(op) != nullptr ||
           dynamic_cast<INLJoinPackedCascadePredicated<uint64_t>*>(op) != nullptr ||
           dynamic_cast<INLJoinPackedCascadeShared<uint64_t>*>(op) != nullptr ||
           dynamic_cast<INLJoinPackedCascadePredicatedShared<uint64_t>*>(op) != nullptr ||
           dynamic_cast<INLJoinPackedGPCascade<uint64_t>*>(op) != nullptr ||
           dynamic_cast<INLJoinPackedGPCascadePredicated<uint64_t>*>(op) != nullptr ||
           dynamic_cast<INLJoinPackedGPCascadeShared<uint64_t>*>(op) != nullptr ||
           dynamic_cast<INLJoinPackedGPCascadePredicatedShared<uint64_t>*>(op) != nullptr ||
           dynamic_cast<FlatJoin<uint64_t>*>(op) != nullptr ||
           dynamic_cast<FlatJoinPredicated<uint64_t>*>(op) != nullptr ||
           dynamic_cast<Intersection<uint64_t>*>(op) != nullptr ||
           dynamic_cast<IntersectionPredicated<uint64_t>*>(op) != nullptr ||
           dynamic_cast<NWayIntersection<uint64_t>*>(op) != nullptr ||
           dynamic_cast<NWayIntersectionPredicated<uint64_t>*>(op) != nullptr;
}

static bool is_cascade_join_operator(Operator* op) {
    return dynamic_cast<INLJoinPackedCascade<uint64_t>*>(op) != nullptr ||
           dynamic_cast<INLJoinPackedCascadePredicated<uint64_t>*>(op) != nullptr ||
           dynamic_cast<INLJoinPackedCascadeShared<uint64_t>*>(op) != nullptr ||
           dynamic_cast<INLJoinPackedCascadePredicatedShared<uint64_t>*>(op) != nullptr ||
           dynamic_cast<INLJoinPackedGPCascade<uint64_t>*>(op) != nullptr ||
           dynamic_cast<INLJoinPackedGPCascadePredicated<uint64_t>*>(op) != nullptr ||
           dynamic_cast<INLJoinPackedGPCascadeShared<uint64_t>*>(op) != nullptr ||
           dynamic_cast<INLJoinPackedGPCascadePredicatedShared<uint64_t>*>(op) != nullptr;
}

// =============================================================================
// Scalar Predicate Tests
// =============================================================================

TEST(ScalarPredicateTest, EqPredicateEvaluatesCorrectly) {
    ScalarPredicate<uint64_t> eq_pred(pred_eq<uint64_t>, 5);
    EXPECT_TRUE(eq_pred.evaluate(5));
    EXPECT_FALSE(eq_pred.evaluate(4));
    EXPECT_FALSE(eq_pred.evaluate(6));
}

TEST(ScalarPredicateTest, NeqPredicateEvaluatesCorrectly) {
    ScalarPredicate<uint64_t> neq_pred(pred_neq<uint64_t>, 5);
    EXPECT_TRUE(neq_pred.evaluate(4));
    EXPECT_FALSE(neq_pred.evaluate(5));
    EXPECT_TRUE(neq_pred.evaluate(6));
}

TEST(ScalarPredicateTest, LtPredicateEvaluatesCorrectly) {
    ScalarPredicate<uint64_t> lt_pred(pred_lt<uint64_t>, 5);
    EXPECT_TRUE(lt_pred.evaluate(4));
    EXPECT_FALSE(lt_pred.evaluate(5));
    EXPECT_FALSE(lt_pred.evaluate(6));
}

TEST(ScalarPredicateTest, GtPredicateEvaluatesCorrectly) {
    ScalarPredicate<uint64_t> gt_pred(pred_gt<uint64_t>, 5);
    EXPECT_FALSE(gt_pred.evaluate(4));
    EXPECT_FALSE(gt_pred.evaluate(5));
    EXPECT_TRUE(gt_pred.evaluate(6));
}

TEST(ScalarPredicateTest, LtePredicateEvaluatesCorrectly) {
    ScalarPredicate<uint64_t> lte_pred(pred_lte<uint64_t>, 5);
    EXPECT_TRUE(lte_pred.evaluate(4));
    EXPECT_TRUE(lte_pred.evaluate(5));
    EXPECT_FALSE(lte_pred.evaluate(6));
}

TEST(ScalarPredicateTest, GtePredicateEvaluatesCorrectly) {
    ScalarPredicate<uint64_t> gte_pred(pred_gte<uint64_t>, 5);
    EXPECT_FALSE(gte_pred.evaluate(4));
    EXPECT_TRUE(gte_pred.evaluate(5));
    EXPECT_TRUE(gte_pred.evaluate(6));
}

TEST(ScalarPredicateTest, InPredicateEvaluatesCorrectly) {
    ScalarPredicate<uint64_t> in_pred;
    in_pred.op = PredicateOp::IN;
    in_pred.in_values = {1, 3, 5};
    EXPECT_TRUE(in_pred.evaluate(1));
    EXPECT_TRUE(in_pred.evaluate(3));
    EXPECT_TRUE(in_pred.evaluate(5));
    EXPECT_FALSE(in_pred.evaluate(2));
    EXPECT_FALSE(in_pred.evaluate(4));
    EXPECT_FALSE(in_pred.evaluate(6));
}

TEST(ScalarPredicateTest, NotInPredicateEvaluatesCorrectly) {
    ScalarPredicate<uint64_t> not_in_pred;
    not_in_pred.op = PredicateOp::NOT_IN;
    not_in_pred.in_values = {1, 3, 5};
    EXPECT_FALSE(not_in_pred.evaluate(1));
    EXPECT_FALSE(not_in_pred.evaluate(3));
    EXPECT_FALSE(not_in_pred.evaluate(5));
    EXPECT_TRUE(not_in_pred.evaluate(2));
    EXPECT_TRUE(not_in_pred.evaluate(4));
    EXPECT_TRUE(not_in_pred.evaluate(6));
}

TEST(ScalarPredicateTest, LikePredicateEvaluatesCorrectlyFfxStrT) {
    StringPool pool;
    ScalarPredicate<ffx_str_t> like_pred;
    like_pred.op = PredicateOp::LIKE;
    like_pred.value = ffx_str_t("a%b", &pool);
    like_pred.compiled_regex = std::make_shared<re2::RE2>("^a.*b$");

    EXPECT_TRUE(like_pred.evaluate(ffx_str_t("ab", &pool)));
    EXPECT_TRUE(like_pred.evaluate(ffx_str_t("axb", &pool)));
    EXPECT_FALSE(like_pred.evaluate(ffx_str_t("abc", &pool)));
}

TEST(ScalarPredicateTest, NotLikePredicateEvaluatesCorrectlyFfxStrT) {
    StringPool pool;
    ScalarPredicate<ffx_str_t> not_like_pred;
    not_like_pred.op = PredicateOp::NOT_LIKE;
    not_like_pred.value = ffx_str_t("a%b", &pool);
    not_like_pred.compiled_regex = std::make_shared<re2::RE2>("^a.*b$");

    EXPECT_FALSE(not_like_pred.evaluate(ffx_str_t("ab", &pool)));
    EXPECT_FALSE(not_like_pred.evaluate(ffx_str_t("axb", &pool)));
    EXPECT_TRUE(not_like_pred.evaluate(ffx_str_t("abc", &pool)));
}

TEST(ScalarPredicateTest, NotInPredicateEvaluatesCorrectlyFfxStrT) {
    StringPool pool;
    ScalarPredicate<ffx_str_t> not_in_pred;
    not_in_pred.op = PredicateOp::NOT_IN;
    not_in_pred.in_values.push_back(ffx_str_t("apple", &pool));
    not_in_pred.in_values.push_back(ffx_str_t("banana", &pool));
    std::sort(not_in_pred.in_values.begin(), not_in_pred.in_values.end());

    EXPECT_FALSE(not_in_pred.evaluate(ffx_str_t("apple", &pool)));
    EXPECT_FALSE(not_in_pred.evaluate(ffx_str_t("banana", &pool)));
    EXPECT_TRUE(not_in_pred.evaluate(ffx_str_t("cherry", &pool)));
}

TEST(ScalarPredicateTest, DefaultPredicateAlwaysTrue) {
    ScalarPredicate<uint64_t> default_pred;
    EXPECT_TRUE(default_pred.evaluate(0));
    EXPECT_TRUE(default_pred.evaluate(100));
    EXPECT_TRUE(default_pred.evaluate(UINT64_MAX));
}

// =============================================================================
// Scalar Predicate Group Tests
// =============================================================================

TEST(ScalarPredicateGroupTest, EmptyGroupReturnsTrue) {
    ScalarPredicateGroup<uint64_t> group;
    EXPECT_TRUE(group.evaluate(0));
    EXPECT_TRUE(group.evaluate(100));
}

TEST(ScalarPredicateGroupTest, AndLogicAllMustPass) {
    ScalarPredicateGroup<uint64_t> group;
    group.op = LogicalOp::AND;
    group.predicates.push_back(ScalarPredicate<uint64_t>(pred_gte<uint64_t>, 2));
    group.predicates.push_back(ScalarPredicate<uint64_t>(pred_lte<uint64_t>, 4));

    EXPECT_FALSE(group.evaluate(1));
    EXPECT_TRUE(group.evaluate(2));
    EXPECT_TRUE(group.evaluate(3));
    EXPECT_TRUE(group.evaluate(4));
    EXPECT_FALSE(group.evaluate(5));
}

TEST(ScalarPredicateGroupTest, OrLogicAtLeastOneMustPass) {
    ScalarPredicateGroup<uint64_t> group;
    group.op = LogicalOp::OR;
    group.predicates.push_back(ScalarPredicate<uint64_t>(pred_eq<uint64_t>, 1));
    group.predicates.push_back(ScalarPredicate<uint64_t>(pred_eq<uint64_t>, 5));

    EXPECT_TRUE(group.evaluate(1));
    EXPECT_FALSE(group.evaluate(2));
    EXPECT_FALSE(group.evaluate(3));
    EXPECT_FALSE(group.evaluate(4));
    EXPECT_TRUE(group.evaluate(5));
}

TEST(ScalarPredicateGroupTest, SinglePredicateAnd) {
    ScalarPredicateGroup<uint64_t> group;
    group.op = LogicalOp::AND;
    group.predicates.push_back(ScalarPredicate<uint64_t>(pred_gt<uint64_t>, 10));

    EXPECT_FALSE(group.evaluate(5));
    EXPECT_FALSE(group.evaluate(10));
    EXPECT_TRUE(group.evaluate(15));
}

TEST(ScalarPredicateGroupTest, SinglePredicateOr) {
    ScalarPredicateGroup<uint64_t> group;
    group.op = LogicalOp::OR;
    group.predicates.push_back(ScalarPredicate<uint64_t>(pred_lt<uint64_t>, 5));

    EXPECT_TRUE(group.evaluate(0));
    EXPECT_TRUE(group.evaluate(4));
    EXPECT_FALSE(group.evaluate(5));
    EXPECT_FALSE(group.evaluate(10));
}

// =============================================================================
// Scalar Predicate Expression Tests
// =============================================================================

TEST(ScalarPredicateExpressionTest, EmptyExpressionAlwaysTrue) {
    ScalarPredicateExpression<uint64_t> expr;
    EXPECT_TRUE(expr.evaluate(0));
    EXPECT_TRUE(expr.evaluate(100));
    EXPECT_TRUE(expr.evaluate(UINT64_MAX));
}

TEST(ScalarPredicateExpressionTest, SingleGroupAndLogic) {
    ScalarPredicateExpression<uint64_t> expr;
    expr.top_level_op = LogicalOp::AND;

    ScalarPredicateGroup<uint64_t> group;
    group.op = LogicalOp::AND;
    group.predicates.push_back(ScalarPredicate<uint64_t>(pred_gte<uint64_t>, 5));
    group.predicates.push_back(ScalarPredicate<uint64_t>(pred_lte<uint64_t>, 10));
    expr.groups.push_back(std::move(group));

    EXPECT_FALSE(expr.evaluate(4));
    EXPECT_TRUE(expr.evaluate(5));
    EXPECT_TRUE(expr.evaluate(7));
    EXPECT_TRUE(expr.evaluate(10));
    EXPECT_FALSE(expr.evaluate(11));
}

TEST(ScalarPredicateExpressionTest, MultipleGroupsOrLogic) {
    // Expression: (x >= 2 AND x <= 4) OR (x == 7)
    ScalarPredicateExpression<uint64_t> expr;
    expr.top_level_op = LogicalOp::OR;

    ScalarPredicateGroup<uint64_t> group1;
    group1.op = LogicalOp::AND;
    group1.predicates.push_back(ScalarPredicate<uint64_t>(pred_gte<uint64_t>, 2));
    group1.predicates.push_back(ScalarPredicate<uint64_t>(pred_lte<uint64_t>, 4));
    expr.groups.push_back(std::move(group1));

    ScalarPredicateGroup<uint64_t> group2;
    group2.op = LogicalOp::AND;
    group2.predicates.push_back(ScalarPredicate<uint64_t>(pred_eq<uint64_t>, 7));
    expr.groups.push_back(std::move(group2));

    EXPECT_FALSE(expr.evaluate(1));
    EXPECT_TRUE(expr.evaluate(2));
    EXPECT_TRUE(expr.evaluate(3));
    EXPECT_TRUE(expr.evaluate(4));
    EXPECT_FALSE(expr.evaluate(5));
    EXPECT_FALSE(expr.evaluate(6));
    EXPECT_TRUE(expr.evaluate(7));
    EXPECT_FALSE(expr.evaluate(8));
}

TEST(ScalarPredicateExpressionTest, MultipleGroupsAndLogic) {
    // Expression: (x > 0) AND (x < 10) AND (x != 5)
    ScalarPredicateExpression<uint64_t> expr;
    expr.top_level_op = LogicalOp::AND;

    ScalarPredicateGroup<uint64_t> group1;
    group1.op = LogicalOp::AND;
    group1.predicates.push_back(ScalarPredicate<uint64_t>(pred_gt<uint64_t>, 0));
    expr.groups.push_back(std::move(group1));

    ScalarPredicateGroup<uint64_t> group2;
    group2.op = LogicalOp::AND;
    group2.predicates.push_back(ScalarPredicate<uint64_t>(pred_lt<uint64_t>, 10));
    expr.groups.push_back(std::move(group2));

    ScalarPredicateGroup<uint64_t> group3;
    group3.op = LogicalOp::AND;
    group3.predicates.push_back(ScalarPredicate<uint64_t>(pred_neq<uint64_t>, 5));
    expr.groups.push_back(std::move(group3));

    EXPECT_FALSE(expr.evaluate(0));
    EXPECT_TRUE(expr.evaluate(1));
    EXPECT_TRUE(expr.evaluate(4));
    EXPECT_FALSE(expr.evaluate(5));// x == 5 fails neq condition
    EXPECT_TRUE(expr.evaluate(6));
    EXPECT_TRUE(expr.evaluate(9));
    EXPECT_FALSE(expr.evaluate(10));// x >= 10 fails lt condition
}

TEST(ScalarPredicateExpressionTest, HasPredicates) {
    ScalarPredicateExpression<uint64_t> empty_expr;
    EXPECT_FALSE(empty_expr.has_predicates());

    ScalarPredicateExpression<uint64_t> non_empty_expr;
    ScalarPredicateGroup<uint64_t> group;
    group.predicates.push_back(ScalarPredicate<uint64_t>(pred_eq<uint64_t>, 1));
    non_empty_expr.groups.push_back(std::move(group));
    EXPECT_TRUE(non_empty_expr.has_predicates());
}

// =============================================================================
// Helper Function Tests
// =============================================================================

TEST(PredicateHelperTest, GetPredicateFn) {
    auto eq_fn = get_predicate_fn<uint64_t>(PredicateOp::EQ);
    EXPECT_TRUE(eq_fn(5, 5));
    EXPECT_FALSE(eq_fn(5, 6));

    auto neq_fn = get_predicate_fn<uint64_t>(PredicateOp::NEQ);
    EXPECT_FALSE(neq_fn(5, 5));
    EXPECT_TRUE(neq_fn(5, 6));

    auto lt_fn = get_predicate_fn<uint64_t>(PredicateOp::LT);
    EXPECT_TRUE(lt_fn(4, 5));
    EXPECT_FALSE(lt_fn(5, 5));

    auto gt_fn = get_predicate_fn<uint64_t>(PredicateOp::GT);
    EXPECT_TRUE(gt_fn(6, 5));
    EXPECT_FALSE(gt_fn(5, 5));

    auto lte_fn = get_predicate_fn<uint64_t>(PredicateOp::LTE);
    EXPECT_TRUE(lte_fn(5, 5));
    EXPECT_FALSE(lte_fn(6, 5));

    auto gte_fn = get_predicate_fn<uint64_t>(PredicateOp::GTE);
    EXPECT_TRUE(gte_fn(5, 5));
    EXPECT_FALSE(gte_fn(4, 5));
}

TEST(PredicateHelperTest, ParseScalarValueUint64) {
    EXPECT_EQ(parse_scalar_value<uint64_t>("0", nullptr), 0ULL);
    EXPECT_EQ(parse_scalar_value<uint64_t>("42", nullptr), 42ULL);
    EXPECT_EQ(parse_scalar_value<uint64_t>("18446744073709551615", nullptr), UINT64_MAX);
}

TEST(PredicateHelperTest, ParseScalarValueInt64) {
    EXPECT_EQ(parse_scalar_value<int64_t>("0", nullptr), 0LL);
    EXPECT_EQ(parse_scalar_value<int64_t>("42", nullptr), 42LL);
    EXPECT_EQ(parse_scalar_value<int64_t>("-42", nullptr), -42LL);
}

TEST(PredicateHelperTest, ParseScalarValueDouble) {
    EXPECT_DOUBLE_EQ(parse_scalar_value<double>("0.0", nullptr), 0.0);
    EXPECT_DOUBLE_EQ(parse_scalar_value<double>("3.14", nullptr), 3.14);
    EXPECT_DOUBLE_EQ(parse_scalar_value<double>("-2.5", nullptr), -2.5);
}

TEST(PredicateHelperTest, BuildScalarPredicateExprForAttribute) {
    PredicateExpression expr;
    PredicateGroup group;
    group.op = LogicalOp::AND;

    // Scalar predicate for attribute "a"
    group.predicates.push_back(Predicate(PredicateOp::GT, "a", "5"));
    // Scalar predicate for attribute "b"
    group.predicates.push_back(Predicate(PredicateOp::LT, "b", "10"));
    // Attribute predicate (should be ignored for scalar extraction)
    group.predicates.push_back(Predicate::Attribute(PredicateOp::EQ, "a", "b"));

    expr.groups.push_back(std::move(group));

    // Build scalar expression for "a"
    auto a_expr = build_scalar_predicate_expr<uint64_t>(expr, "a", nullptr);
    EXPECT_TRUE(a_expr.has_predicates());
    ASSERT_EQ(a_expr.groups.size(), 1);
    EXPECT_EQ(a_expr.groups[0].predicates.size(), 1);
    // Verify the predicate works: a > 5
    EXPECT_FALSE(a_expr.evaluate(5));
    EXPECT_TRUE(a_expr.evaluate(6));

    // Build scalar expression for "b"
    auto b_expr = build_scalar_predicate_expr<uint64_t>(expr, "b", nullptr);
    EXPECT_TRUE(b_expr.has_predicates());
    ASSERT_EQ(b_expr.groups.size(), 1);
    EXPECT_EQ(b_expr.groups[0].predicates.size(), 1);
    // Verify the predicate works: b < 10
    EXPECT_TRUE(b_expr.evaluate(9));
    EXPECT_FALSE(b_expr.evaluate(10));

    // Test NOT IN construction
    PredicateExpression expr2;
    PredicateGroup group2;
    Predicate not_in_p;
    not_in_p.op = PredicateOp::NOT_IN;
    not_in_p.type = PredicateType::SCALAR;
    not_in_p.left_attr = "x";
    not_in_p.scalar_values = {"10", "20", "30"};
    group2.predicates.push_back(std::move(not_in_p));
    expr2.groups.push_back(std::move(group2));

    auto x_expr = build_scalar_predicate_expr<uint64_t>(expr2, "x", nullptr);
    EXPECT_TRUE(x_expr.has_predicates());
    EXPECT_FALSE(x_expr.evaluate(10));
    EXPECT_FALSE(x_expr.evaluate(20));
    EXPECT_TRUE(x_expr.evaluate(15));

    // Test NOT LIKE construction
    StringPool pool;
    PredicateExpression expr3;
    PredicateGroup group3;
    group3.predicates.push_back(Predicate(PredicateOp::NOT_LIKE, "s", "abc%"));
    expr3.groups.push_back(std::move(group3));

    auto s_expr = build_scalar_predicate_expr<ffx_str_t>(expr3, "s", &pool);
    EXPECT_TRUE(s_expr.has_predicates());
    EXPECT_FALSE(s_expr.evaluate(ffx_str_t("abcd", &pool)));
    EXPECT_TRUE(s_expr.evaluate(ffx_str_t("axyz", &pool)));

    // Build scalar expression for "c" (no predicates)
    auto c_expr = build_scalar_predicate_expr<uint64_t>(expr, "c", nullptr);
    EXPECT_FALSE(c_expr.has_predicates());
    EXPECT_TRUE(c_expr.evaluate(100));// Should always return true
}

TEST(PredicateHelperTest, HasScalarPredicatesFor) {
    PredicateExpression expr;
    PredicateGroup group;
    group.predicates.push_back(Predicate(PredicateOp::EQ, "x", "42"));
    group.predicates.push_back(Predicate(PredicateOp::GT, "y", "0"));
    expr.groups.push_back(std::move(group));

    EXPECT_TRUE(has_scalar_predicates_for(expr, "x"));
    EXPECT_TRUE(has_scalar_predicates_for(expr, "y"));
    EXPECT_FALSE(has_scalar_predicates_for(expr, "z"));
}

// =============================================================================
// Edge Case Tests
// =============================================================================

TEST(PredicateEdgeCaseTest, ZeroValue) {
    ScalarPredicate<uint64_t> eq_zero(pred_eq<uint64_t>, 0);
    EXPECT_TRUE(eq_zero.evaluate(0));
    EXPECT_FALSE(eq_zero.evaluate(1));

    ScalarPredicate<uint64_t> gt_zero(pred_gt<uint64_t>, 0);
    EXPECT_FALSE(gt_zero.evaluate(0));
    EXPECT_TRUE(gt_zero.evaluate(1));
}

TEST(PredicateEdgeCaseTest, MaxValue) {
    ScalarPredicate<uint64_t> eq_max(pred_eq<uint64_t>, UINT64_MAX);
    EXPECT_TRUE(eq_max.evaluate(UINT64_MAX));
    EXPECT_FALSE(eq_max.evaluate(UINT64_MAX - 1));

    ScalarPredicate<uint64_t> lt_max(pred_lt<uint64_t>, UINT64_MAX);
    EXPECT_TRUE(lt_max.evaluate(UINT64_MAX - 1));
    EXPECT_FALSE(lt_max.evaluate(UINT64_MAX));
}

TEST(PredicateEdgeCaseTest, ChainedRangePredicates) {
    // Testing range: 10 <= x < 20
    ScalarPredicateGroup<uint64_t> group;
    group.op = LogicalOp::AND;
    group.predicates.push_back(ScalarPredicate<uint64_t>(pred_gte<uint64_t>, 10));
    group.predicates.push_back(ScalarPredicate<uint64_t>(pred_lt<uint64_t>, 20));

    EXPECT_FALSE(group.evaluate(9));
    EXPECT_TRUE(group.evaluate(10));
    EXPECT_TRUE(group.evaluate(15));
    EXPECT_TRUE(group.evaluate(19));
    EXPECT_FALSE(group.evaluate(20));
}

TEST(PredicateEdgeCaseTest, DisjointRangesOr) {
    // Testing: x < 5 OR x > 10
    ScalarPredicateExpression<uint64_t> expr;
    expr.top_level_op = LogicalOp::OR;

    ScalarPredicateGroup<uint64_t> group1;
    group1.op = LogicalOp::AND;
    group1.predicates.push_back(ScalarPredicate<uint64_t>(pred_lt<uint64_t>, 5));
    expr.groups.push_back(std::move(group1));

    ScalarPredicateGroup<uint64_t> group2;
    group2.op = LogicalOp::AND;
    group2.predicates.push_back(ScalarPredicate<uint64_t>(pred_gt<uint64_t>, 10));
    expr.groups.push_back(std::move(group2));

    EXPECT_TRUE(expr.evaluate(0));
    EXPECT_TRUE(expr.evaluate(4));
    EXPECT_FALSE(expr.evaluate(5));
    EXPECT_FALSE(expr.evaluate(7));
    EXPECT_FALSE(expr.evaluate(10));
    EXPECT_TRUE(expr.evaluate(11));
    EXPECT_TRUE(expr.evaluate(100));
}

// =============================================================================
// PackedThetaJoin Tests
// =============================================================================

class PackedThetaJoinTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create adjacency lists for testing
        // Forward: 0->{}, 1->{2,3,4}, 2->{3,4}, 3->{4}, 4->{}
        // Backward: 0->{}, 1->{}, 2->{1}, 3->{1,2}, 4->{1,2,3}

        auto fwd_adj_lists_arr = std::make_unique<AdjList<uint64_t>[]>(5);
        auto bwd_adj_lists_arr = std::make_unique<AdjList<uint64_t>[]>(5);

        // Forward adjacency lists
        fwd_adj_lists_arr[0].size = 0;
        fwd_adj_lists_arr[0].values = nullptr;

        fwd_adj_lists_arr[1].size = 3;
        fwd_adj_lists_arr[1].values = new uint64_t[3]{2, 3, 4};

        fwd_adj_lists_arr[2].size = 2;
        fwd_adj_lists_arr[2].values = new uint64_t[2]{3, 4};

        fwd_adj_lists_arr[3].size = 1;
        fwd_adj_lists_arr[3].values = new uint64_t[1]{4};

        fwd_adj_lists_arr[4].size = 0;
        fwd_adj_lists_arr[4].values = nullptr;

        // Backward adjacency lists
        bwd_adj_lists_arr[0].size = 0;
        bwd_adj_lists_arr[0].values = nullptr;

        bwd_adj_lists_arr[1].size = 0;
        bwd_adj_lists_arr[1].values = nullptr;

        bwd_adj_lists_arr[2].size = 1;
        bwd_adj_lists_arr[2].values = new uint64_t[1]{1};

        bwd_adj_lists_arr[3].size = 2;
        bwd_adj_lists_arr[3].values = new uint64_t[2]{1, 2};

        bwd_adj_lists_arr[4].size = 3;
        bwd_adj_lists_arr[4].values = new uint64_t[3]{1, 2, 3};

        table = std::make_unique<Table>(5, 5, std::move(fwd_adj_lists_arr), std::move(bwd_adj_lists_arr));
        table->columns = {"a", "b", "c"};
        table->name = "test_table";
    }

    void TearDown() override {
        if (table && table->fwd_adj_lists) {
            for (uint64_t i = 0; i < 5; i++) {
                if (table->fwd_adj_lists[i].values) { delete[] table->fwd_adj_lists[i].values; }
            }
        }
        if (table && table->bwd_adj_lists) {
            for (uint64_t i = 0; i < 5; i++) {
                if (table->bwd_adj_lists[i].values) { delete[] table->bwd_adj_lists[i].values; }
            }
        }
    }

    std::unique_ptr<Table> table;
    QueryVariableToVectorMap map;
    Schema schema;
    std::shared_ptr<FactorizedTreeElement> root;
    Vector<uint64_t>* a_vec;
    std::vector<std::string> column_ordering;
};

TEST_F(PackedThetaJoinTest, AncestorLtDescendant) {
    // Test a < b filter: left_attr=a (ancestor), right_attr=b (descendant)
    // For a=1 -> b={2,3,4}: check if a=1 < b values
    // 1 < 2 -> true, 1 < 3 -> true, 1 < 4 -> true => all pass

    schema.tables.clear();
    schema.tables.push_back(table.get());
    schema.map = &map;

    a_vec = nullptr;
    root = std::make_shared<FactorizedTreeElement>("a", a_vec);
    schema.root = root;

    column_ordering = {"a", "b"};
    schema.column_ordering = &column_ordering;

    // Pipeline: scan(a) -> join(a,b) -> theta_join(a < b) -> sink
    auto scan_op = std::make_unique<Scan<>>("a");
    auto join_op = std::make_unique<INLJoinPacked<>>("a", "b", true);
    scan_op->set_next_operator(std::move(join_op));

    // left_attr=a (ancestor), right_attr=b (descendant), predicate: a < b
    auto theta_op = std::make_unique<PackedThetaJoin<>>("a", "b", PredicateOp::LT);
    scan_op->next_op->set_next_operator(std::move(theta_op));

    auto sink_op = std::make_unique<SinkPacked>();
    scan_op->next_op->next_op->set_next_operator(std::move(sink_op));

    scan_op->init(&schema);
    scan_op->execute();

    // a=1 -> b={2,3,4}, a < b means 1 < {2,3,4} => all pass: 3 tuples
    // a=2 -> b={3,4}, 2 < {3,4} => all pass: 2 tuples
    // a=3 -> b={4}, 3 < 4 => pass: 1 tuple
    // Total: 6 tuples
    SinkPacked* sink = dynamic_cast<SinkPacked*>(scan_op->next_op->next_op->next_op);
    ASSERT_NE(sink, nullptr);
    EXPECT_EQ(sink->get_num_output_tuples(), 6);
}

TEST_F(PackedThetaJoinTest, AncestorGtDescendant) {
    // Test a > b filter: left_attr=a (ancestor), right_attr=b (descendant)
    // For a=1 -> b={2,3,4}: check if a=1 > b values
    // 1 > 2 -> false, 1 > 3 -> false, 1 > 4 -> false => none pass

    schema.tables.clear();
    schema.tables.push_back(table.get());
    schema.map = &map;

    a_vec = nullptr;
    root = std::make_shared<FactorizedTreeElement>("a", a_vec);
    schema.root = root;

    column_ordering = {"a", "b"};
    schema.column_ordering = &column_ordering;

    // Pipeline: scan(a) -> join(a,b) -> theta_join(a > b) -> sink
    auto scan_op = std::make_unique<Scan<>>("a");
    auto join_op = std::make_unique<INLJoinPacked<>>("a", "b", true);
    scan_op->set_next_operator(std::move(join_op));

    // left_attr=a (ancestor), right_attr=b (descendant), predicate: a > b
    auto theta_op = std::make_unique<PackedThetaJoin<>>("a", "b", PredicateOp::GT);
    scan_op->next_op->set_next_operator(std::move(theta_op));

    auto sink_op = std::make_unique<SinkPacked>();
    scan_op->next_op->next_op->set_next_operator(std::move(sink_op));

    scan_op->init(&schema);
    scan_op->execute();

    // a=1 -> b={2,3,4}, a > b means 1 > {2,3,4} => none pass
    // a=2 -> b={3,4}, 2 > {3,4} => none pass
    // a=3 -> b={4}, 3 > 4 => false
    // Total: 0 tuples
    SinkPacked* sink = dynamic_cast<SinkPacked*>(scan_op->next_op->next_op->next_op);
    ASSERT_NE(sink, nullptr);
    EXPECT_EQ(sink->get_num_output_tuples(), 0);
}

TEST_F(PackedThetaJoinTest, AncestorLteDescendant) {
    // Test a <= b filter

    schema.tables.clear();
    schema.tables.push_back(table.get());
    schema.map = &map;

    a_vec = nullptr;
    root = std::make_shared<FactorizedTreeElement>("a", a_vec);
    schema.root = root;

    column_ordering = {"a", "b"};
    schema.column_ordering = &column_ordering;

    auto scan_op = std::make_unique<Scan<>>("a");
    auto join_op = std::make_unique<INLJoinPacked<>>("a", "b", true);
    scan_op->set_next_operator(std::move(join_op));

    auto theta_op = std::make_unique<PackedThetaJoin<>>("a", "b", PredicateOp::LTE);
    scan_op->next_op->set_next_operator(std::move(theta_op));

    auto sink_op = std::make_unique<SinkPacked>();
    scan_op->next_op->next_op->set_next_operator(std::move(sink_op));

    scan_op->init(&schema);
    scan_op->execute();

    // Same as LT since a < b for all pairs (no equality cases in test data)
    SinkPacked* sink = dynamic_cast<SinkPacked*>(scan_op->next_op->next_op->next_op);
    ASSERT_NE(sink, nullptr);
    EXPECT_EQ(sink->get_num_output_tuples(), 6);
}

TEST_F(PackedThetaJoinTest, AncestorEqDescendant) {
    // Test a == b filter - should produce no results since a < b for all pairs

    schema.tables.clear();
    schema.tables.push_back(table.get());
    schema.map = &map;

    a_vec = nullptr;
    root = std::make_shared<FactorizedTreeElement>("a", a_vec);
    schema.root = root;

    column_ordering = {"a", "b"};
    schema.column_ordering = &column_ordering;

    auto scan_op = std::make_unique<Scan<>>("a");
    auto join_op = std::make_unique<INLJoinPacked<>>("a", "b", true);
    scan_op->set_next_operator(std::move(join_op));

    auto theta_op = std::make_unique<PackedThetaJoin<>>("a", "b", PredicateOp::EQ);
    scan_op->next_op->set_next_operator(std::move(theta_op));

    auto sink_op = std::make_unique<SinkPacked>();
    scan_op->next_op->next_op->set_next_operator(std::move(sink_op));

    scan_op->init(&schema);
    scan_op->execute();

    // No a values equal to b values in our test data
    SinkPacked* sink = dynamic_cast<SinkPacked*>(scan_op->next_op->next_op->next_op);
    ASSERT_NE(sink, nullptr);
    EXPECT_EQ(sink->get_num_output_tuples(), 0);
}

TEST_F(PackedThetaJoinTest, AncestorNeqDescendant) {
    // Test a != b filter - should produce all results since a != b for all pairs

    schema.tables.clear();
    schema.tables.push_back(table.get());
    schema.map = &map;

    a_vec = nullptr;
    root = std::make_shared<FactorizedTreeElement>("a", a_vec);
    schema.root = root;

    column_ordering = {"a", "b"};
    schema.column_ordering = &column_ordering;

    auto scan_op = std::make_unique<Scan<>>("a");
    auto join_op = std::make_unique<INLJoinPacked<>>("a", "b", true);
    scan_op->set_next_operator(std::move(join_op));

    auto theta_op = std::make_unique<PackedThetaJoin<>>("a", "b", PredicateOp::NEQ);
    scan_op->next_op->set_next_operator(std::move(theta_op));

    auto sink_op = std::make_unique<SinkPacked>();
    scan_op->next_op->next_op->set_next_operator(std::move(sink_op));

    scan_op->init(&schema);
    scan_op->execute();

    // All a values are != b values
    SinkPacked* sink = dynamic_cast<SinkPacked*>(scan_op->next_op->next_op->next_op);
    ASSERT_NE(sink, nullptr);
    EXPECT_EQ(sink->get_num_output_tuples(), 6);
}

// =============================================================================
// End-to-End Integration Test
// =============================================================================

/*
 * Query:
 *   SELECT R1.src AS a, R1.dest AS b, R2.dest AS c, R3.dest AS e, R4.dest AS d
 *   FROM R AS R1
 *   JOIN R AS R2 ON R1.src = R2.src      -- a -> c
 *   JOIN R AS R3 ON R1.src = R3.src      -- a -> e
 *   JOIN R AS R4 ON R1.dest = R4.src     -- b -> d
 *   JOIN R AS R5 ON R2.dest = R5.src     -- c -> d
 *   JOIN R AS R6 ON R3.dest = R6.src     -- e -> d
 *   WHERE R4.dest = R5.dest AND R5.dest = R6.dest  -- 3-way intersection on d
 *     AND R1.src <> 2                              -- scalar predicate: a != 2
 *     AND R1.dest <> R3.dest                       -- attribute predicate: b != e
 *
 * Factorized Tree (linear ordering: a, b, c, e, d):
 *   a (root)
 *   └── b (INLJoinPacked: a -> b)
 *       └── c (FlatJoin: b, a -> c)
 *           └── e (FlatJoin: c, a -> e)
 *               └── d (NWayIntersection: b, c, e -> d)
 *
 * Pipeline:
 *   1. ScanPredicated(a) with scalar predicate: a != 2
 *   2. INLJoinPacked(a -> b)
 *   3. FlatJoin(b, a -> c)
 *   4. FlatJoin(c, a -> e)
 *   5. PackedThetaJoin(b, e, NEQ) - attribute predicate: b != e
 *   6. NWayIntersection(b, c, e -> d)
 *   7. SinkPacked
 */

#include "../src/operator/include/join/flat_join.hpp"
#include "../src/operator/include/join/nway_intersection.hpp"
#include "../src/operator/include/scan/scan_predicated.hpp"
#include "../src/operator/include/scan/scan_synchronized.hpp"
#include "../src/operator/include/scan/scan_synchronized_predicated.hpp"

class PredicatedE2ETest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create adjacency lists for table R
        // Forward: 0->{}, 1->{2,3,4}, 2->{3,4}, 3->{4}, 4->{}
        // Backward: 0->{}, 1->{}, 2->{1}, 3->{1,2}, 4->{1,2,3}

        auto fwd_adj_lists_arr = std::make_unique<AdjList<uint64_t>[]>(5);
        auto bwd_adj_lists_arr = std::make_unique<AdjList<uint64_t>[]>(5);

        // Forward adjacency lists
        fwd_adj_lists_arr[0].size = 0;
        fwd_adj_lists_arr[0].values = nullptr;

        fwd_adj_lists_arr[1].size = 3;
        fwd_adj_lists_arr[1].values = new uint64_t[3]{2, 3, 4};

        fwd_adj_lists_arr[2].size = 2;
        fwd_adj_lists_arr[2].values = new uint64_t[2]{3, 4};

        fwd_adj_lists_arr[3].size = 1;
        fwd_adj_lists_arr[3].values = new uint64_t[1]{4};

        fwd_adj_lists_arr[4].size = 0;
        fwd_adj_lists_arr[4].values = nullptr;

        // Backward adjacency lists
        bwd_adj_lists_arr[0].size = 0;
        bwd_adj_lists_arr[0].values = nullptr;

        bwd_adj_lists_arr[1].size = 0;
        bwd_adj_lists_arr[1].values = nullptr;

        bwd_adj_lists_arr[2].size = 1;
        bwd_adj_lists_arr[2].values = new uint64_t[1]{1};

        bwd_adj_lists_arr[3].size = 2;
        bwd_adj_lists_arr[3].values = new uint64_t[2]{1, 2};

        bwd_adj_lists_arr[4].size = 3;
        bwd_adj_lists_arr[4].values = new uint64_t[3]{1, 2, 3};

        table = std::make_unique<Table>(5, 5, std::move(fwd_adj_lists_arr), std::move(bwd_adj_lists_arr));
        table->columns = {"a", "b", "c", "d", "e"};
        table->name = "R";
    }

    void TearDown() override {
        if (table && table->fwd_adj_lists) {
            for (uint64_t i = 0; i < 5; i++) {
                if (table->fwd_adj_lists[i].values) { delete[] table->fwd_adj_lists[i].values; }
            }
        }
        if (table && table->bwd_adj_lists) {
            for (uint64_t i = 0; i < 5; i++) {
                if (table->bwd_adj_lists[i].values) { delete[] table->bwd_adj_lists[i].values; }
            }
        }
    }

    std::unique_ptr<Table> table;
    QueryVariableToVectorMap map;
    Schema schema;
    std::shared_ptr<FactorizedTreeElement> root;
    Vector<uint64_t>* a_vec;
    std::vector<std::string> column_ordering;
};

TEST_F(PredicatedE2ETest, FullPipelineWithPredicates) {
    /*
     * Test data analysis:
     * - a values: 1, 2, 3 have outgoing edges
     * - After filtering a != 2: only a=1 and a=3 remain
     * 
     * For a=1:
     *   b = {2,3,4}, c = {2,3,4}, e = {2,3,4}
     *   After b != e filter and 3-way intersection (b->d ∩ c->d ∩ e->d):
     *   Valid tuples where intersection is non-empty:
     *     (b=2, c=2, e=3, d=4), (b=2, c=3, e=3, d=4),
     *     (b=3, c=2, e=2, d=4), (b=3, c=3, e=2, d=4)
     *   Total: 4 tuples
     * 
     * For a=3:
     *   b = {4}, c = {4}, e = {4}
     *   b != e filter: 4 != 4 is false, filtered out
     *   Total: 0 tuples
     * 
     * Expected total: 4 tuples
     */

    // Setup schema
    schema.tables.clear();
    schema.tables.push_back(table.get());
    schema.map = &map;

    // Create factorized tree with root "a"
    a_vec = nullptr;
    root = std::make_shared<FactorizedTreeElement>("a", a_vec);
    schema.root = root;

    column_ordering = {"a", "b", "c", "e", "d"};
    schema.column_ordering = &column_ordering;

    // Build predicate expression for a != 2
    PredicateExpression scan_pred_expr = PredicateParser::parse_predicates("NEQ(a,2)");

    // Create operator pipeline:
    // 1. ScanPredicated(a) with a != 2
    auto scan_op = std::make_unique<ScanPredicated<uint64_t>>("a", scan_pred_expr);

    // 2. INLJoinPacked(a -> b)
    auto join_ab = std::make_unique<INLJoinPacked<uint64_t>>("a", "b", true);
    scan_op->set_next_operator(std::move(join_ab));

    // 3. FlatJoin(b, a -> c) - produces c under b using a as LCA
    auto flat_join_c = std::make_unique<FlatJoin<uint64_t>>("b", "a", "c", true);
    scan_op->next_op->set_next_operator(std::move(flat_join_c));

    // 4. FlatJoin(c, a -> e) - produces e under c using a as LCA
    auto flat_join_e = std::make_unique<FlatJoin<uint64_t>>("c", "a", "e", true);
    scan_op->next_op->next_op->set_next_operator(std::move(flat_join_e));

    // 5. PackedThetaJoin(b, e, NEQ) - attribute predicate: b != e
    auto theta_join = std::make_unique<PackedThetaJoin<uint64_t>>("b", "e", PredicateOp::NEQ);
    scan_op->next_op->next_op->next_op->set_next_operator(std::move(theta_join));

    // 6. NWayIntersection(b, c, e -> d)
    std::vector<std::pair<std::string, bool>> input_attrs_and_dirs = {
            {"b", true},// b -> d (forward)
            {"c", true},// c -> d (forward)
            {"e", true} // e -> d (forward)
    };
    auto nway_intersection = std::make_unique<NWayIntersection<uint64_t>>("d", input_attrs_and_dirs);
    scan_op->next_op->next_op->next_op->next_op->set_next_operator(std::move(nway_intersection));

    // 7. SinkPacked
    auto sink_op = std::make_unique<SinkPacked>();
    scan_op->next_op->next_op->next_op->next_op->next_op->set_next_operator(std::move(sink_op));

    // Initialize and execute
    scan_op->init(&schema);
    scan_op->execute();

    // Get vectors for debug output
    auto* vec_a = map.get_vector("a");
    auto* vec_b = map.get_vector("b");
    auto* vec_c = map.get_vector("c");
    auto* vec_e = map.get_vector("e");
    auto* vec_d = map.get_vector("d");

    std::cout << "=== FullPipelineWithPredicates Output ===" << std::endl;
    std::cout << "a range: [" << GET_START_POS(*vec_a->state) << ", " << GET_END_POS(*vec_a->state) << "]" << std::endl;
    std::cout << "b range: [" << GET_START_POS(*vec_b->state) << ", " << GET_END_POS(*vec_b->state) << "]" << std::endl;
    std::cout << "c range: [" << GET_START_POS(*vec_c->state) << ", " << GET_END_POS(*vec_c->state) << "]" << std::endl;
    std::cout << "e range: [" << GET_START_POS(*vec_e->state) << ", " << GET_END_POS(*vec_e->state) << "]" << std::endl;
    std::cout << "d range: [" << GET_START_POS(*vec_d->state) << ", " << GET_END_POS(*vec_d->state) << "]" << std::endl;

    // Print tuples - traverse factorized tree
    int tuple_count = 0;
    for (int a_pos = GET_START_POS(*vec_a->state); a_pos <= GET_END_POS(*vec_a->state); a_pos++) {
        if (!TEST_BIT(vec_a->state->selector, a_pos)) continue;
        uint64_t a_val = vec_a->values[a_pos];

        for (uint32_t b_pos = vec_a->state->offset[a_pos]; b_pos < vec_a->state->offset[a_pos + 1]; b_pos++) {
            if (!TEST_BIT(vec_b->state->selector, b_pos)) continue;
            uint64_t b_val = vec_b->values[b_pos];

            for (uint32_t c_pos = vec_b->state->offset[b_pos]; c_pos < vec_b->state->offset[b_pos + 1]; c_pos++) {
                if (!TEST_BIT(vec_c->state->selector, c_pos)) continue;
                uint64_t c_val = vec_c->values[c_pos];

                for (uint32_t e_pos = vec_c->state->offset[c_pos]; e_pos < vec_c->state->offset[c_pos + 1]; e_pos++) {
                    if (!TEST_BIT(vec_e->state->selector, e_pos)) continue;
                    uint64_t e_val = vec_e->values[e_pos];

                    for (uint32_t d_pos = vec_e->state->offset[e_pos]; d_pos < vec_e->state->offset[e_pos + 1]; d_pos++) {
                        if (!TEST_BIT(vec_d->state->selector, d_pos)) continue;
                        uint64_t d_val = vec_d->values[d_pos];

                        std::cout << "  Tuple " << tuple_count++ << ": a=" << a_val << " b=" << b_val << " c=" << c_val
                                  << " e=" << e_val << " d=" << d_val << std::endl;
                    }
                }
            }
        }
    }
    std::cout << "Total tuples enumerated: " << tuple_count << std::endl;

    // Verify result: Based on analysis, should be 4 tuples
    SinkPacked* sink = dynamic_cast<SinkPacked*>(scan_op->next_op->next_op->next_op->next_op->next_op->next_op);
    ASSERT_NE(sink, nullptr);
    std::cout << "Sink reported: " << sink->get_num_output_tuples() << " tuples" << std::endl;
    EXPECT_EQ(sink->get_num_output_tuples(), 4);// Corrected from 6 to 4 based on analysis
}

TEST_F(PredicatedE2ETest, FullPipelineWithTwoAttributePredicates) {
    /*
     * Same query as above but with additional attribute predicate: b != c
     * 
     * WHERE R4.dest = R5.dest AND R5.dest = R6.dest  -- 3-way intersection
     *   AND R1.src <> 2                              -- a != 2
     *   AND R1.dest <> R3.dest                       -- b != e
     *   AND R1.dest <> R2.dest                       -- b != c (NEW)
     *
     * Pipeline:
     *   1. ScanPredicated(a) with a != 2
     *   2. INLJoinPacked(a -> b)
     *   3. FlatJoin(b, a -> c)
     *   4. PackedThetaJoin(b, c, NEQ) - b != c (NEW)
     *   5. FlatJoin(c, a -> e)
     *   6. PackedThetaJoin(b, e, NEQ) - b != e
     *   7. NWayIntersection(b, c, e -> d)
     *   8. SinkPacked
     *
     * Expected: 2 tuples
     *   - (a=1, b=2, c=3, e=3, d=4): intersection({3,4}, {4}, {4}) = {4}
     *   - (a=1, b=3, c=2, e=2, d=4): intersection({4}, {3,4}, {3,4}) = {4}
     */

    // Setup schema
    schema.tables.clear();
    schema.tables.push_back(table.get());
    schema.map = &map;

    // Create factorized tree with root "a"
    a_vec = nullptr;
    root = std::make_shared<FactorizedTreeElement>("a", a_vec);
    schema.root = root;

    column_ordering = {"a", "b", "c", "e", "d"};
    schema.column_ordering = &column_ordering;

    // Build scalar predicate expression for a != 2
    PredicateExpression scan_pred_expr = PredicateParser::parse_predicates("NEQ(a,2)");

    // Create operator pipeline:
    // 1. ScanPredicated(a) with a != 2
    auto scan_op = std::make_unique<ScanPredicated<uint64_t>>("a", scan_pred_expr);

    // 2. INLJoinPacked(a -> b)
    auto join_ab = std::make_unique<INLJoinPacked<uint64_t>>("a", "b", true);
    scan_op->set_next_operator(std::move(join_ab));

    // 3. FlatJoin(b, a -> c) - produces c under b using a as LCA
    auto flat_join_c = std::make_unique<FlatJoin<uint64_t>>("b", "a", "c", true);
    scan_op->next_op->set_next_operator(std::move(flat_join_c));

    // 4. PackedThetaJoin(b, c, NEQ) - attribute predicate: b != c (NEW)
    auto theta_join_bc = std::make_unique<PackedThetaJoin<uint64_t>>("b", "c", PredicateOp::NEQ);
    scan_op->next_op->next_op->set_next_operator(std::move(theta_join_bc));

    // 5. FlatJoin(c, a -> e) - produces e under c using a as LCA
    auto flat_join_e = std::make_unique<FlatJoin<uint64_t>>("c", "a", "e", true);
    scan_op->next_op->next_op->next_op->set_next_operator(std::move(flat_join_e));

    // 6. PackedThetaJoin(b, e, NEQ) - attribute predicate: b != e
    auto theta_join_be = std::make_unique<PackedThetaJoin<uint64_t>>("b", "e", PredicateOp::NEQ);
    scan_op->next_op->next_op->next_op->next_op->set_next_operator(std::move(theta_join_be));

    // 7. NWayIntersection(b, c, e -> d)
    std::vector<std::pair<std::string, bool>> input_attrs_and_dirs = {
            {"b", true},// b -> d (forward)
            {"c", true},// c -> d (forward)
            {"e", true} // e -> d (forward)
    };
    auto nway_intersection = std::make_unique<NWayIntersection<uint64_t>>("d", input_attrs_and_dirs);
    scan_op->next_op->next_op->next_op->next_op->next_op->set_next_operator(std::move(nway_intersection));

    // 8. SinkPacked
    auto sink_op = std::make_unique<SinkPacked>();
    scan_op->next_op->next_op->next_op->next_op->next_op->next_op->set_next_operator(std::move(sink_op));

    // Initialize and execute
    scan_op->init(&schema);
    scan_op->execute();

    // Get vectors for debug output
    auto* vec_a = map.get_vector("a");
    auto* vec_b = map.get_vector("b");
    auto* vec_c = map.get_vector("c");
    auto* vec_e = map.get_vector("e");
    auto* vec_d = map.get_vector("d");

    std::cout << "=== FullPipelineWithTwoAttributePredicates Output ===" << std::endl;
    std::cout << "a range: [" << GET_START_POS(*vec_a->state) << ", " << GET_END_POS(*vec_a->state) << "]" << std::endl;
    std::cout << "b range: [" << GET_START_POS(*vec_b->state) << ", " << GET_END_POS(*vec_b->state) << "]" << std::endl;
    std::cout << "c range: [" << GET_START_POS(*vec_c->state) << ", " << GET_END_POS(*vec_c->state) << "]" << std::endl;
    std::cout << "e range: [" << GET_START_POS(*vec_e->state) << ", " << GET_END_POS(*vec_e->state) << "]" << std::endl;
    std::cout << "d range: [" << GET_START_POS(*vec_d->state) << ", " << GET_END_POS(*vec_d->state) << "]" << std::endl;

    // Print tuples - traverse factorized tree
    int tuple_count = 0;
    for (int a_pos = GET_START_POS(*vec_a->state); a_pos <= GET_END_POS(*vec_a->state); a_pos++) {
        if (!TEST_BIT(vec_a->state->selector, a_pos)) continue;
        uint64_t a_val = vec_a->values[a_pos];

        for (uint32_t b_pos = vec_a->state->offset[a_pos]; b_pos < vec_a->state->offset[a_pos + 1]; b_pos++) {
            if (!TEST_BIT(vec_b->state->selector, b_pos)) continue;
            uint64_t b_val = vec_b->values[b_pos];

            for (uint32_t c_pos = vec_b->state->offset[b_pos]; c_pos < vec_b->state->offset[b_pos + 1]; c_pos++) {
                if (!TEST_BIT(vec_c->state->selector, c_pos)) continue;
                uint64_t c_val = vec_c->values[c_pos];

                for (uint32_t e_pos = vec_c->state->offset[c_pos]; e_pos < vec_c->state->offset[c_pos + 1]; e_pos++) {
                    if (!TEST_BIT(vec_e->state->selector, e_pos)) continue;
                    uint64_t e_val = vec_e->values[e_pos];

                    for (uint32_t d_pos = vec_e->state->offset[e_pos]; d_pos < vec_e->state->offset[e_pos + 1]; d_pos++) {
                        if (!TEST_BIT(vec_d->state->selector, d_pos)) continue;
                        uint64_t d_val = vec_d->values[d_pos];

                        std::cout << "  Tuple " << tuple_count++ << ": a=" << a_val << " b=" << b_val << " c=" << c_val
                                  << " e=" << e_val << " d=" << d_val << std::endl;
                    }
                }
            }
        }
    }
    std::cout << "Total tuples enumerated: " << tuple_count << std::endl;

    // Verify result: Based on analysis, should be 2 tuples (b != c AND b != e)
    SinkPacked* sink =
            dynamic_cast<SinkPacked*>(scan_op->next_op->next_op->next_op->next_op->next_op->next_op->next_op);
    ASSERT_NE(sink, nullptr);
    std::cout << "Sink reported: " << sink->get_num_output_tuples() << " tuples" << std::endl;
    EXPECT_EQ(sink->get_num_output_tuples(), 2);// Corrected from 3 to 2 based on analysis
}

// =============================================================================
// Two-Phase Plan Creation Tests
// =============================================================================

TEST_F(PredicatedE2ETest, TwoPhaseSimpleQuery) {
    /**
     * Test simple two-phase plan creation for query: a -> b
     */

    // Setup schema
    schema.tables.clear();
    schema.tables.push_back(table.get());
    schema.map = &map;

    root = std::make_shared<FactorizedTreeElement>("a", a_vec);
    schema.root = root;

    column_ordering = {"a", "b"};
    schema.column_ordering = &column_ordering;

    // Create query (no predicates)
    Query query("Q(a,b) := R(a,b)");
    std::vector<std::string> ordering = {"a", "b"};

    // Build plan tree (Phase 1)
    auto plan_tree = build_plan_tree(query, ordering, SINK_PACKED);
    ASSERT_NE(plan_tree, nullptr);
    ASSERT_NE(plan_tree->get_root(), nullptr);
    EXPECT_EQ(plan_tree->get_root()->attribute, "a");
    EXPECT_EQ(plan_tree->get_root()->type, PlanNodeType::SCAN);

    // Create operators from plan tree (Phase 2)
    std::vector<const Table*> tables_vec = {table.get()};
    auto [plan, ftree] = create_operators_from_plan_tree(*plan_tree, query, ordering, tables_vec);
    ASSERT_NE(plan, nullptr);

    // Initialize and execute
    plan->init(tables_vec, ftree, &schema);
    plan->get_first_op()->execute();

    // Verify result - traverse to find sink
    Operator* op = plan->get_first_op();
    while (op->next_op != nullptr) {
        op = op->next_op;
    }
    // Could be SinkPacked or SinkLinear depending on tree structure
    EXPECT_GT(op->get_num_output_tuples(), 0);
}

TEST_F(PredicatedE2ETest, TwoPhaseQueryWithScalarPredicate) {
    /**
     * Test two-phase plan creation for query: a -> b with predicate a != 2
     */

    // Setup schema
    schema.tables.clear();
    schema.tables.push_back(table.get());
    schema.map = &map;

    root = std::make_shared<FactorizedTreeElement>("a", a_vec);
    schema.root = root;

    column_ordering = {"a", "b"};
    schema.column_ordering = &column_ordering;

    // Create query with predicate
    Query query("Q(a,b) := R(a,b) WHERE NEQ(a,2)");
    std::vector<std::string> ordering = {"a", "b"};

    // Use combined two-phase function
    auto [plan, ftree] = map_ordering_to_plan(query, ordering, SINK_PACKED);
    ASSERT_NE(plan, nullptr);

    // Initialize and execute
    std::vector<const Table*> tables_vec = {table.get()};
    plan->init(tables_vec, ftree, &schema);
    plan->get_first_op()->execute();

    // Verify result - traverse to find sink
    Operator* op = plan->get_first_op();
    while (op->next_op != nullptr) {
        op = op->next_op;
    }
    // Expected: all tuples where a != 2
    // Could be SinkPacked or SinkLinear depending on tree structure
    std::cout << "TwoPhaseQueryWithScalarPredicate result: " << op->get_num_output_tuples() << " tuples" << std::endl;
    EXPECT_GT(op->get_num_output_tuples(), 0);
}

TEST_F(PredicatedE2ETest, V2LastJoinBeforeSinkIsCascade) {
    schema.tables.clear();
    schema.tables.push_back(table.get());
    schema.map = &map;

    root = std::make_shared<FactorizedTreeElement>("a", a_vec);
    schema.root = root;

    column_ordering = {"a", "b", "c"};
    schema.column_ordering = &column_ordering;

    Query query("Q(a,b,c) := R(a,b),R(b,c)");
    std::vector<std::string> ordering = {"a", "b", "c"};

    auto [plan, ftree] = map_ordering_to_plan(query, ordering, SINK_PACKED);
    ASSERT_NE(plan, nullptr);

    std::vector<const Table*> tables_vec = {table.get()};
    plan->init(tables_vec, ftree, &schema);

    Operator* last_join = nullptr;
    for (Operator* op = plan->get_first_op(); op != nullptr; op = op->next_op) {
        if (is_join_operator(op)) {
            last_join = op;
        }
    }

    ASSERT_NE(last_join, nullptr);
    EXPECT_TRUE(is_cascade_join_operator(last_join)) << "Expected the last join before sink to be cascaded";
}

TEST_F(PredicatedE2ETest, V2SynchronizedLastJoinBeforeSinkIsCascade) {
    schema.tables.clear();
    schema.tables.push_back(table.get());
    schema.map = &map;

    root = std::make_shared<FactorizedTreeElement>("a", a_vec);
    schema.root = root;

    column_ordering = {"a", "b", "c"};
    schema.column_ordering = &column_ordering;

    Query query("Q(a,b,c) := R(a,b),R(b,c)");
    std::vector<std::string> ordering = {"a", "b", "c"};

    auto [plan, ftree] = map_ordering_to_plan_synchronized(query, ordering, SINK_PACKED, 0);
    ASSERT_NE(plan, nullptr);

    std::vector<const Table*> tables_vec = {table.get()};
    plan->init(tables_vec, ftree, &schema);

    Operator* last_join = nullptr;
    for (Operator* op = plan->get_first_op(); op != nullptr; op = op->next_op) {
        if (is_join_operator(op)) {
            last_join = op;
        }
    }

    ASSERT_NE(last_join, nullptr);
    EXPECT_TRUE(is_cascade_join_operator(last_join))
            << "Expected the last join before sink to be cascaded in synchronized two-phase planner";
}

TEST_F(PredicatedE2ETest, PrintPipelineTest) {
    /**
     * Test print_pipeline function
     */

    // Setup schema
    schema.tables.clear();
    schema.tables.push_back(table.get());
    schema.map = &map;

    root = std::make_shared<FactorizedTreeElement>("a", a_vec);
    schema.root = root;

    column_ordering = {"a", "b", "c"};
    schema.column_ordering = &column_ordering;

    // Create query with predicates
    Query query("Q(a,b,c) := R(a,b),R(b,c) WHERE NEQ(a,2) AND NEQ(a,c)");
    std::vector<std::string> ordering = {"a", "b", "c"};

    // Use combined two-phase function
    auto [plan, ftree] = map_ordering_to_plan(query, ordering, SINK_PACKED);
    ASSERT_NE(plan, nullptr);

    // Print the pipeline
    std::cout << "\n--- Test: PrintPipelineTest ---" << std::endl;
    plan->print_pipeline();

    // Also test standalone function
    std::cout << "--- Using standalone print_operator_chain: ---" << std::endl;
    print_operator_chain(plan->get_first_op());

    SUCCEED();
}

TEST_F(PredicatedE2ETest, TwoPhaseQueryWithAttributePredicate) {
    /**
     * Test two-phase plan creation for query: a -> b -> c with predicate a != c
     * 
     * Note: PackedThetaJoin requires ancestor-descendant relationship.
     * a is ancestor of c (a -> b -> c), so a != c is a valid theta join.
     */

    // Setup schema
    schema.tables.clear();
    schema.tables.push_back(table.get());
    schema.map = &map;

    root = std::make_shared<FactorizedTreeElement>("a", a_vec);
    schema.root = root;

    column_ordering = {"a", "b", "c"};
    schema.column_ordering = &column_ordering;

    // Create query with attribute predicate
    // a -> b -> c with predicate a != c
    Query query("Q(a,b,c) := R(a,b),R(b,c) WHERE NEQ(a,c)");
    std::vector<std::string> ordering = {"a", "b", "c"};

    // Use combined two-phase function
    auto [plan, ftree] = map_ordering_to_plan(query, ordering, SINK_PACKED);
    ASSERT_NE(plan, nullptr);

    // Initialize and execute
    std::vector<const Table*> tables_vec = {table.get()};
    plan->init(tables_vec, ftree, &schema);
    plan->get_first_op()->execute();

    // Verify result - traverse to find sink
    Operator* op = plan->get_first_op();
    while (op->next_op != nullptr) {
        op = op->next_op;
    }
    // Could be SinkPacked or SinkLinear depending on tree structure
    std::cout << "TwoPhaseQueryWithAttributePredicate result: " << op->get_num_output_tuples() << " tuples"
              << std::endl;
    EXPECT_GT(op->get_num_output_tuples(), 0);
}

TEST_F(PredicatedE2ETest, ThetaJoinOrderingWithSiblings) {
    /**
     * Regression test for theta-join planning when predicate attributes are siblings
     * in the plan tree and the middle node is chosen as root.
     *
     * Query pattern (logical):
     *   person1 -> person2 -> person3 -> tag
     * Predicate:
     *   person1 != person3
     *
     * Ordering:
     *   {person2, person1, person3, tag}
     *
     * In this case, person1 and person3 are siblings under root person2.
     * PackedThetaJoin requires a true ancestor-descendant relationship, so sibling
     * predicates cannot be handled and should be skipped with a warning.
     */

    Query query("Q(person1,person2,person3,tag) := "
                 "R(person1,person2),R(person2,person3),R(person3,tag) WHERE NEQ(person1,person3)");
    std::vector<std::string> ordering = {"person2", "person1", "person3", "tag"};

    auto plan_tree = build_plan_tree(query, ordering, SINK_PACKED);
    ASSERT_NE(plan_tree, nullptr);

    const PlanNode* root_node = plan_tree->get_root();
    ASSERT_NE(root_node, nullptr);
    EXPECT_EQ(root_node->attribute, "person2");

    // Find the node for person3 (would be the descendant if they weren't siblings)
    PlanNode* person3_node = plan_tree->find_node("person3");
    ASSERT_NE(person3_node, nullptr);

    // Since person1 and person3 are siblings, the tree should be restructured
    // to make person3 a descendant of person1 using FlatJoin
    // Then PackedThetaJoin should be created
    EXPECT_TRUE(person3_node->has_theta_joins());
    EXPECT_EQ(person3_node->theta_joins_after.size(), 1u);

    const auto& tj = person3_node->theta_joins_after[0];
    EXPECT_EQ(tj.ancestor_attr, "person1");
    EXPECT_EQ(tj.descendant_attr, "person3");
    EXPECT_EQ(tj.op, PredicateOp::NEQ);

    // Verify that person3 node is now a FlatJoin (restructured)
    EXPECT_EQ(person3_node->type, PlanNodeType::FLAT_JOIN);
    EXPECT_EQ(person3_node->join_from_attr, "person1");
    EXPECT_EQ(person3_node->lca_attr, "person2");
}

TEST_F(PredicatedE2ETest, ThetaJoinOrderingWithSiblings_MultipleOrderings) {
    /**
     * Same query as ThetaJoinOrderingWithSiblings but with multiple orderings.
     * We assert that for every generated plan:
     *  - PackedThetaJoin(NEQ(person1,person3)) appears only after both attrs
     *    have been produced by previous join operators in the pipeline.
     *  - If person1 and person3 are siblings in the tree, the tree is restructured
     *    using FlatJoin to make one a descendant of the other.
     */
    Query query("Q(person1,person2,person3,tag) := "
                 "R(person1,person2),R(person2,person3),R(person3,tag) WHERE NEQ(person1,person3)");

    const std::vector<std::vector<std::string>> orderings = {
            {"person2", "person1", "person3", "tag"},// person1 and person3 are siblings -> predicate skipped
            {"person2", "person3", "person1", "tag"},// person1 and person3 are siblings -> predicate skipped
            {"person1", "person2", "person3", "tag"},// person1 is ancestor of person3 -> predicate created
    };

    for (const auto& ordering: orderings) {
        auto [plan, ftree] = map_ordering_to_plan(query, ordering, SINK_PACKED);
        ASSERT_NE(plan, nullptr);
        // Only assert theta join ordering if theta joins exist
        // (they won't exist if attributes are siblings)
        assert_theta_joins_come_after_attrs_are_produced(plan->get_first_op());
    }
}

TEST_F(PredicatedE2ETest, MultipleAttributeNEQPredicates_ThetaJoinsAfterPopulation) {
    /**
     * Multi-theta-join regression: ensure every PackedThetaJoin is placed only after
     * both referenced attributes have been joined/produced in the operator pipeline.
     *
     * Query:
     *   a->b, b->c, c->d, c->e
     * Predicates:
     *   NEQ(a,d) AND NEQ(a,e) AND NEQ(b,e)
     *
     * All pairs here are ancestor-descendant w.r.t the tree:
     *   a -> b -> c -> {d,e}
     */
    Query query("Q(a,b,c,d,e) := R(a,b),R(b,c),R(c,d),R(c,e) WHERE NEQ(a,d) AND NEQ(a,e) AND NEQ(b,e)");

    const std::vector<std::vector<std::string>> orderings = {
            {"a", "b", "c", "d", "e"},
            {"a", "b", "c", "e", "d"},// swap siblings under c
    };

    for (const auto& ordering: orderings) {
        auto [plan, ftree] = map_ordering_to_plan(query, ordering, SINK_PACKED);
        ASSERT_NE(plan, nullptr);
        assert_theta_joins_come_after_attrs_are_produced(plan->get_first_op());
    }
}

// =============================================================================
// ScanSynchronized and ScanSynchronizedPredicated Tests
// =============================================================================

class ScanSynchronizedTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple table with edges for testing
        // Forward: 0->{}, 1->{2,3,4}, 2->{3,4}, 3->{4}, 4->{}
        auto fwd_adj_lists_arr = std::make_unique<AdjList<uint64_t>[]>(5);
        auto bwd_adj_lists_arr = std::make_unique<AdjList<uint64_t>[]>(5);

        fwd_adj_lists_arr[0].size = 0;
        fwd_adj_lists_arr[0].values = nullptr;

        fwd_adj_lists_arr[1].size = 3;
        fwd_adj_lists_arr[1].values = new uint64_t[3]{2, 3, 4};

        fwd_adj_lists_arr[2].size = 2;
        fwd_adj_lists_arr[2].values = new uint64_t[2]{3, 4};

        fwd_adj_lists_arr[3].size = 1;
        fwd_adj_lists_arr[3].values = new uint64_t[1]{4};

        fwd_adj_lists_arr[4].size = 0;
        fwd_adj_lists_arr[4].values = nullptr;

        bwd_adj_lists_arr[0].size = 0;
        bwd_adj_lists_arr[0].values = nullptr;
        bwd_adj_lists_arr[1].size = 0;
        bwd_adj_lists_arr[1].values = nullptr;
        bwd_adj_lists_arr[2].size = 1;
        bwd_adj_lists_arr[2].values = new uint64_t[1]{1};
        bwd_adj_lists_arr[3].size = 2;
        bwd_adj_lists_arr[3].values = new uint64_t[2]{1, 2};
        bwd_adj_lists_arr[4].size = 3;
        bwd_adj_lists_arr[4].values = new uint64_t[3]{1, 2, 3};

        table = std::make_unique<Table>(5, 5, std::move(fwd_adj_lists_arr), std::move(bwd_adj_lists_arr));
        table->columns = {"a", "b"};
        table->name = "test_table";
    }

    void TearDown() override {
        if (table && table->fwd_adj_lists) {
            for (uint64_t i = 0; i < 5; i++) {
                if (table->fwd_adj_lists[i].values) { delete[] table->fwd_adj_lists[i].values; }
            }
        }
        if (table && table->bwd_adj_lists) {
            for (uint64_t i = 0; i < 5; i++) {
                if (table->bwd_adj_lists[i].values) { delete[] table->bwd_adj_lists[i].values; }
            }
        }
    }

    std::unique_ptr<Table> table;
    QueryVariableToVectorMap map;
    Schema schema;
    std::shared_ptr<FactorizedTreeElement> root;
    Vector<uint64_t>* a_vec;
    std::vector<std::string> column_ordering;
};

TEST_F(ScanSynchronizedTest, ScanSynchronizedBasic) {
    /**
     * Test basic ScanSynchronized functionality
     * Scans a single morsel starting from start_id
     */
    schema.tables.clear();
    schema.tables.push_back(table.get());
    schema.map = &map;

    a_vec = nullptr;
    root = std::make_shared<FactorizedTreeElement>("a", a_vec);
    schema.root = root;

    column_ordering = {"a", "b"};
    schema.column_ordering = &column_ordering;

    // Create pipeline: ScanSynchronized(a, start_id=1) -> INLJoin(a->b) -> Sink
    auto scan_op = std::make_unique<ScanSynchronized>("a", 1);
    scan_op->set_max_id(4);// max_id = 4 (values 0-4)

    auto join_op = std::make_unique<INLJoinPacked<uint64_t>>("a", "b", true);
    scan_op->set_next_operator(std::move(join_op));

    auto sink_op = std::make_unique<SinkPacked>();
    scan_op->next_op->set_next_operator(std::move(sink_op));

    scan_op->init(&schema);
    scan_op->execute();

    SinkPacked* sink = dynamic_cast<SinkPacked*>(scan_op->next_op->next_op);
    ASSERT_NE(sink, nullptr);
    // Starting from id=1, processing morsel [1, 4]
    // a=1 has 3 edges, a=2 has 2, a=3 has 1, a=4 has 0 = 6 total
    EXPECT_EQ(sink->get_num_output_tuples(), 6);
}

TEST_F(ScanSynchronizedTest, ScanSynchronizedPredicatedBasic) {
    /**
     * Test ScanSynchronizedPredicated with a simple predicate
     * Scans a morsel starting from start_id and applies predicate
     */
    schema.tables.clear();
    schema.tables.push_back(table.get());
    schema.map = &map;

    a_vec = nullptr;
    root = std::make_shared<FactorizedTreeElement>("a", a_vec);
    schema.root = root;

    column_ordering = {"a", "b"};
    schema.column_ordering = &column_ordering;

    // Build predicate: a != 2
    PredicateExpression pred_expr = PredicateParser::parse_predicates("NEQ(a,2)");

    // Create pipeline: ScanSynchronizedPredicated(a, start_id=1, a != 2) -> INLJoin(a->b) -> Sink
    auto scan_op = std::make_unique<ScanSynchronizedPredicated<uint64_t>>("a", 1, pred_expr);
    scan_op->set_max_id(4);

    auto join_op = std::make_unique<INLJoinPacked<uint64_t>>("a", "b", true);
    scan_op->set_next_operator(std::move(join_op));

    auto sink_op = std::make_unique<SinkPacked>();
    scan_op->next_op->set_next_operator(std::move(sink_op));

    scan_op->init(&schema);
    scan_op->execute();

    SinkPacked* sink = dynamic_cast<SinkPacked*>(scan_op->next_op->next_op);
    ASSERT_NE(sink, nullptr);
    // Starting from id=1, morsel [1, 4], filtering a != 2
    // a=1 (pass, 3 edges), a=2 (fail), a=3 (pass, 1 edge), a=4 (pass, 0 edges) = 4 total
    EXPECT_EQ(sink->get_num_output_tuples(), 4);
}

TEST_F(ScanSynchronizedTest, ScanSynchronizedPredicatedRangePredicate) {
    /**
     * Test ScanSynchronizedPredicated with range predicate
     * Predicate: a >= 2 AND a <= 3
     */
    schema.tables.clear();
    schema.tables.push_back(table.get());
    schema.map = &map;

    a_vec = nullptr;
    root = std::make_shared<FactorizedTreeElement>("a", a_vec);
    schema.root = root;

    column_ordering = {"a", "b"};
    schema.column_ordering = &column_ordering;

    // Build predicate: a >= 2 AND a <= 3
    PredicateExpression pred_expr = PredicateParser::parse_predicates("GTE(a,2) AND LTE(a,3)");

    // Create pipeline
    auto scan_op = std::make_unique<ScanSynchronizedPredicated<uint64_t>>("a", 0, pred_expr);
    scan_op->set_max_id(4);

    auto join_op = std::make_unique<INLJoinPacked<uint64_t>>("a", "b", true);
    scan_op->set_next_operator(std::move(join_op));

    auto sink_op = std::make_unique<SinkPacked>();
    scan_op->next_op->set_next_operator(std::move(sink_op));

    scan_op->init(&schema);
    scan_op->execute();

    SinkPacked* sink = dynamic_cast<SinkPacked*>(scan_op->next_op->next_op);
    ASSERT_NE(sink, nullptr);
    // a in [2, 3]: a=2 (2 edges), a=3 (1 edge) = 3 total
    EXPECT_EQ(sink->get_num_output_tuples(), 3);
}

TEST_F(ScanSynchronizedTest, ScanSynchronizedPredicatedNoPassing) {
    /**
     * Test ScanSynchronizedPredicated where no values pass the predicate
     * Predicate: a > 10 (no values pass)
     */
    schema.tables.clear();
    schema.tables.push_back(table.get());
    schema.map = &map;

    a_vec = nullptr;
    root = std::make_shared<FactorizedTreeElement>("a", a_vec);
    schema.root = root;

    column_ordering = {"a", "b"};
    schema.column_ordering = &column_ordering;

    // Build predicate: a > 10 (no values pass)
    PredicateExpression pred_expr = PredicateParser::parse_predicates("GT(a,10)");

    // Create pipeline
    auto scan_op = std::make_unique<ScanSynchronizedPredicated<uint64_t>>("a", 0, pred_expr);
    scan_op->set_max_id(4);

    auto join_op = std::make_unique<INLJoinPacked<uint64_t>>("a", "b", true);
    scan_op->set_next_operator(std::move(join_op));

    auto sink_op = std::make_unique<SinkPacked>();
    scan_op->next_op->set_next_operator(std::move(sink_op));

    scan_op->init(&schema);
    scan_op->execute();

    SinkPacked* sink = dynamic_cast<SinkPacked*>(scan_op->next_op->next_op);
    ASSERT_NE(sink, nullptr);
    // No values pass predicate a > 10
    EXPECT_EQ(sink->get_num_output_tuples(), 0);
}

TEST_F(ScanSynchronizedTest, ScanSynchronizedPredicatedGetters) {
    /**
     * Test getters for ScanSynchronizedPredicated
     */
    // Build predicate
    PredicateExpression pred_expr = PredicateParser::parse_predicates("EQ(test_attr,5)");

    auto scan_op = std::make_unique<ScanSynchronizedPredicated<uint64_t>>("test_attr", 100, pred_expr);

    EXPECT_EQ(scan_op->attribute(), "test_attr");
    EXPECT_TRUE(scan_op->has_predicate());

    // Without predicate
    PredicateExpression empty_expr;
    auto scan_op_no_pred = std::make_unique<ScanSynchronizedPredicated<uint64_t>>("test_attr2", 0, empty_expr);

    EXPECT_EQ(scan_op_no_pred->attribute(), "test_attr2");
    EXPECT_FALSE(scan_op_no_pred->has_predicate());
}

// =============================================================================
// Shared State Operator Tests
// =============================================================================

class SharedStateOperatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create adjacency lists for n:1 relationship (each source has exactly 1 destination)
        // Forward: 0->{10}, 1->{11}, 2->{12}, 3->{13}, 4->{14}
        // This simulates a many-to-one relationship (e.g., person -> city)

        auto fwd_adj_lists_arr = std::make_unique<AdjList<uint64_t>[]>(5);
        auto bwd_adj_lists_arr = std::make_unique<AdjList<uint64_t>[]>(15);

        // Forward: each source maps to exactly one destination
        for (uint64_t i = 0; i < 5; i++) {
            fwd_adj_lists_arr[i].size = 1;
            fwd_adj_lists_arr[i].values = new uint64_t[1]{10 + i};
        }

        // Backward: each destination can have multiple sources (for testing)
        for (uint64_t i = 0; i < 15; i++) {
            if (i >= 10 && i < 15) {
                // Destinations 10-14 have sources 0-4
                bwd_adj_lists_arr[i].size = 1;
                bwd_adj_lists_arr[i].values = new uint64_t[1]{i - 10};
            } else {
                bwd_adj_lists_arr[i].size = 0;
                bwd_adj_lists_arr[i].values = nullptr;
            }
        }

        table = std::make_unique<Table>(5, 15, std::move(fwd_adj_lists_arr), std::move(bwd_adj_lists_arr));
        table->columns = {"person", "city"};
        table->name = "PersonCity";
        table->cardinality = Cardinality::MANY_TO_ONE;// n:1 relationship
    }

    void TearDown() override {
        if (table && table->fwd_adj_lists) {
            for (uint64_t i = 0; i < 5; i++) {
                if (table->fwd_adj_lists[i].values) { delete[] table->fwd_adj_lists[i].values; }
            }
        }
        if (table && table->bwd_adj_lists) {
            for (uint64_t i = 0; i < 15; i++) {
                if (table->bwd_adj_lists[i].values) { delete[] table->bwd_adj_lists[i].values; }
            }
        }
    }

    std::unique_ptr<Table> table;
    QueryVariableToVectorMap map;
    Schema schema;
    std::shared_ptr<FactorizedTreeElement> root;
    Vector<uint64_t>* person_vec;
    std::vector<std::string> column_ordering;
};

TEST_F(SharedStateOperatorTest, INLJoinPackedShared_Basic) {
    schema.tables.clear();
    schema.tables.push_back(table.get());
    schema.map = &map;

    person_vec = nullptr;
    root = std::make_shared<FactorizedTreeElement>("person", person_vec);
    schema.root = root;

    column_ordering = {"person", "city"};
    schema.column_ordering = &column_ordering;

    // Pipeline: scan(person) -> INLJoinPackedShared(person->city) -> sink
    auto scan_op = std::make_unique<Scan<uint64_t>>("person");
    auto join_op = std::make_unique<INLJoinPackedShared<uint64_t>>("person", "city", true);
    scan_op->set_next_operator(std::move(join_op));

    auto sink_op = std::make_unique<SinkPacked>();
    scan_op->next_op->set_next_operator(std::move(sink_op));

    scan_op->init(&schema);

    // Verify shared state was used
    auto* city_vec = map.get_vector("city");
    ASSERT_NE(city_vec, nullptr);
    auto* person_vec_check = map.get_vector("person");
    ASSERT_NE(person_vec_check, nullptr);

    // City vector should share state with person vector (n:1 forward)
    EXPECT_EQ(city_vec->state, person_vec_check->state);
    EXPECT_TRUE(city_vec->has_identity_rle());

    scan_op->execute();

    // Verify output: 5 persons -> 5 cities (1:1 mapping)
    SinkPacked* sink = dynamic_cast<SinkPacked*>(scan_op->next_op->next_op);
    ASSERT_NE(sink, nullptr);
    EXPECT_EQ(sink->get_num_output_tuples(), 5);
}

TEST_F(SharedStateOperatorTest, INLJoinPackedPredicatedShared_WithPredicate) {
    schema.tables.clear();
    schema.tables.push_back(table.get());
    schema.map = &map;

    person_vec = nullptr;
    root = std::make_shared<FactorizedTreeElement>("person", person_vec);
    schema.root = root;

    column_ordering = {"person", "city"};
    schema.column_ordering = &column_ordering;

    // Build predicate: city < 12
    PredicateExpression pred_expr = PredicateParser::parse_predicates("LT(city,12)");

    // Pipeline: scan(person) -> INLJoinPackedPredicatedShared(person->city, city < 12) -> sink
    auto scan_op = std::make_unique<Scan<uint64_t>>("person");
    auto join_op = std::make_unique<INLJoinPackedPredicatedShared<uint64_t>>("person", "city", true, pred_expr);
    scan_op->set_next_operator(std::move(join_op));

    auto sink_op = std::make_unique<SinkPacked>();
    scan_op->next_op->set_next_operator(std::move(sink_op));

    scan_op->init(&schema);

    // Verify shared state
    auto* city_vec = map.get_vector("city");
    ASSERT_NE(city_vec, nullptr);
    auto* person_vec_check = map.get_vector("person");
    ASSERT_NE(person_vec_check, nullptr);
    EXPECT_EQ(city_vec->state, person_vec_check->state);
    EXPECT_TRUE(city_vec->has_identity_rle());

    scan_op->execute();

    // Verify output: persons 0,1 -> cities 10,11 (city < 12 filters out 12,13,14)
    SinkPacked* sink = dynamic_cast<SinkPacked*>(scan_op->next_op->next_op);
    ASSERT_NE(sink, nullptr);
    EXPECT_EQ(sink->get_num_output_tuples(), 2);
}

TEST_F(SharedStateOperatorTest, SharedStateNotUsedForManyToMany) {
    // Change table to m:n relationship
    table->cardinality = Cardinality::MANY_TO_MANY;

    schema.tables.clear();
    schema.tables.push_back(table.get());
    schema.map = &map;

    person_vec = nullptr;
    root = std::make_shared<FactorizedTreeElement>("person", person_vec);
    schema.root = root;

    column_ordering = {"person", "city"};
    schema.column_ordering = &column_ordering;

    // Pipeline: scan(person) -> INLJoinPacked(person->city) -> sink
    // Should use regular operator, not shared
    auto scan_op = std::make_unique<Scan<uint64_t>>("person");
    auto join_op = std::make_unique<INLJoinPacked<uint64_t>>("person", "city", true);
    scan_op->set_next_operator(std::move(join_op));

    auto sink_op = std::make_unique<SinkPacked>();
    scan_op->next_op->set_next_operator(std::move(sink_op));

    scan_op->init(&schema);

    // Verify state is NOT shared for m:n
    auto* city_vec = map.get_vector("city");
    ASSERT_NE(city_vec, nullptr);
    auto* person_vec_check = map.get_vector("person");
    ASSERT_NE(person_vec_check, nullptr);

    // For m:n, states should be different
    EXPECT_NE(city_vec->state, person_vec_check->state);
    EXPECT_FALSE(city_vec->has_identity_rle());

    scan_op->execute();

    // Verify output still works
    SinkPacked* sink = dynamic_cast<SinkPacked*>(scan_op->next_op->next_op);
    ASSERT_NE(sink, nullptr);
    EXPECT_EQ(sink->get_num_output_tuples(), 5);
}

TEST_F(SharedStateOperatorTest, INLJoinPackedCascadeShared_Basic) {
    // Test cascade join with shared state
    // Setup: person -> city (first child), person -> country (second child, cascade)
    // Create a table for person -> country with n:1 cardinality
    auto fwd_adj_lists_arr2 = std::make_unique<AdjList<uint64_t>[]>(5);
    auto bwd_adj_lists_arr2 = std::make_unique<AdjList<uint64_t>[]>(20);

    // Forward: each person maps to exactly one country
    for (uint64_t i = 0; i < 5; i++) {
        fwd_adj_lists_arr2[i].size = 1;
        fwd_adj_lists_arr2[i].values = new uint64_t[1]{100 + i};
    }

    // Backward: each country has one person
    for (uint64_t i = 0; i < 20; i++) {
        if (i >= 100 && i < 105) {
            bwd_adj_lists_arr2[i].size = 1;
            bwd_adj_lists_arr2[i].values = new uint64_t[1]{i - 100};
        } else {
            bwd_adj_lists_arr2[i].size = 0;
            bwd_adj_lists_arr2[i].values = nullptr;
        }
    }

    auto table2 = std::make_unique<Table>(5, 20, std::move(fwd_adj_lists_arr2), std::move(bwd_adj_lists_arr2));
    table2->columns = {"person", "country"};
    table2->name = "PersonCountry";
    table2->cardinality = Cardinality::MANY_TO_ONE;

    schema.tables.clear();
    schema.tables.push_back(table.get()); // PersonCity
    schema.tables.push_back(table2.get());// PersonCountry
    schema.map = &map;

    person_vec = nullptr;
    root = std::make_shared<FactorizedTreeElement>("person", person_vec);
    schema.root = root;

    column_ordering = {"person", "city", "country"};
    schema.column_ordering = &column_ordering;

    // Pipeline: scan(person) -> INLJoinPackedShared(person->city) -> INLJoinPackedCascadeShared(person->country) -> sink
    auto scan_op = std::make_unique<Scan<uint64_t>>("person");
    auto join_city = std::make_unique<INLJoinPackedShared<uint64_t>>("person", "city", true);
    scan_op->set_next_operator(std::move(join_city));

    auto join_country = std::make_unique<INLJoinPackedCascadeShared<uint64_t>>("person", "country", true);
    scan_op->next_op->set_next_operator(std::move(join_country));

    auto sink_op = std::make_unique<SinkPacked>();
    scan_op->next_op->next_op->set_next_operator(std::move(sink_op));

    scan_op->init(&schema);

    // Verify shared state
    auto* city_vec = map.get_vector("city");
    auto* country_vec = map.get_vector("country");
    auto* person_vec_check = map.get_vector("person");

    ASSERT_NE(city_vec, nullptr);
    ASSERT_NE(country_vec, nullptr);
    ASSERT_NE(person_vec_check, nullptr);

    // Both city and country should share state with person (n:1 forward)
    EXPECT_EQ(city_vec->state, person_vec_check->state);
    EXPECT_EQ(country_vec->state, person_vec_check->state);
    EXPECT_TRUE(city_vec->has_identity_rle());
    EXPECT_TRUE(country_vec->has_identity_rle());

    scan_op->execute();

    // Verify output: 5 persons -> 5 cities -> 5 countries
    SinkPacked* sink = dynamic_cast<SinkPacked*>(scan_op->next_op->next_op->next_op);
    ASSERT_NE(sink, nullptr);
    EXPECT_EQ(sink->get_num_output_tuples(), 5);

    // Cleanup
    if (table2 && table2->fwd_adj_lists) {
        for (uint64_t i = 0; i < 5; i++) {
            if (table2->fwd_adj_lists[i].values) { delete[] table2->fwd_adj_lists[i].values; }
        }
    }
    if (table2 && table2->bwd_adj_lists) {
        for (uint64_t i = 0; i < 20; i++) {
            if (table2->bwd_adj_lists[i].values) { delete[] table2->bwd_adj_lists[i].values; }
        }
    }
}

TEST_F(SharedStateOperatorTest, INLJoinPackedCascadePredicatedShared_WithPredicate) {
    // Test cascade predicated join with shared state
    // Setup similar to above but with predicate: country < 103
    auto fwd_adj_lists_arr2 = std::make_unique<AdjList<uint64_t>[]>(5);
    auto bwd_adj_lists_arr2 = std::make_unique<AdjList<uint64_t>[]>(20);

    for (uint64_t i = 0; i < 5; i++) {
        fwd_adj_lists_arr2[i].size = 1;
        fwd_adj_lists_arr2[i].values = new uint64_t[1]{100 + i};
    }

    for (uint64_t i = 0; i < 20; i++) {
        if (i >= 100 && i < 105) {
            bwd_adj_lists_arr2[i].size = 1;
            bwd_adj_lists_arr2[i].values = new uint64_t[1]{i - 100};
        } else {
            bwd_adj_lists_arr2[i].size = 0;
            bwd_adj_lists_arr2[i].values = nullptr;
        }
    }

    auto table2 = std::make_unique<Table>(5, 20, std::move(fwd_adj_lists_arr2), std::move(bwd_adj_lists_arr2));
    table2->columns = {"person", "country"};
    table2->name = "PersonCountry";
    table2->cardinality = Cardinality::MANY_TO_ONE;

    schema.tables.clear();
    schema.tables.push_back(table.get());
    schema.tables.push_back(table2.get());
    schema.map = &map;

    person_vec = nullptr;
    root = std::make_shared<FactorizedTreeElement>("person", person_vec);
    schema.root = root;

    column_ordering = {"person", "city", "country"};
    schema.column_ordering = &column_ordering;

    // Build predicate: country < 103
    PredicateExpression pred_expr_raw = PredicateParser::parse_predicates("LT(country,103)");

    // Pipeline: scan(person) -> INLJoinPackedShared(person->city) -> INLJoinPackedCascadePredicatedShared(person->country, country < 103) -> sink
    auto scan_op = std::make_unique<Scan<uint64_t>>("person");
    auto join_city = std::make_unique<INLJoinPackedShared<uint64_t>>("person", "city", true);
    scan_op->set_next_operator(std::move(join_city));

    auto join_country =
            std::make_unique<INLJoinPackedCascadePredicatedShared<uint64_t>>("person", "country", true, pred_expr_raw);
    scan_op->next_op->set_next_operator(std::move(join_country));

    auto sink_op = std::make_unique<SinkPacked>();
    scan_op->next_op->next_op->set_next_operator(std::move(sink_op));

    scan_op->init(&schema);

    // Verify shared state
    auto* city_vec = map.get_vector("city");
    auto* country_vec = map.get_vector("country");
    auto* person_vec_check = map.get_vector("person");

    ASSERT_NE(city_vec, nullptr);
    ASSERT_NE(country_vec, nullptr);
    ASSERT_NE(person_vec_check, nullptr);

    EXPECT_EQ(city_vec->state, person_vec_check->state);
    EXPECT_EQ(country_vec->state, person_vec_check->state);
    EXPECT_TRUE(city_vec->has_identity_rle());
    EXPECT_TRUE(country_vec->has_identity_rle());

    scan_op->execute();

    // Verify output: persons 0,1,2 -> countries 100,101,102 (country < 103 filters out 103,104)
    SinkPacked* sink = dynamic_cast<SinkPacked*>(scan_op->next_op->next_op->next_op);
    ASSERT_NE(sink, nullptr);
    EXPECT_EQ(sink->get_num_output_tuples(), 3);

    // Cleanup
    if (table2 && table2->fwd_adj_lists) {
        for (uint64_t i = 0; i < 5; i++) {
            if (table2->fwd_adj_lists[i].values) { delete[] table2->fwd_adj_lists[i].values; }
        }
    }
    if (table2 && table2->bwd_adj_lists) {
        for (uint64_t i = 0; i < 20; i++) {
            if (table2->bwd_adj_lists[i].values) { delete[] table2->bwd_adj_lists[i].values; }
        }
    }
}

TEST_F(SharedStateOperatorTest, INLJoinPackedGPCascadeShared_Basic) {
    // Test GP cascade join with shared state
    // Setup: person -> city -> country (GP cascade: person -> country via city)
    // Create a table for city -> country with n:1 cardinality
    auto fwd_adj_lists_arr2 = std::make_unique<AdjList<uint64_t>[]>(15);
    auto bwd_adj_lists_arr2 = std::make_unique<AdjList<uint64_t>[]>(20);

    // Forward: each city maps to exactly one country
    for (uint64_t i = 0; i < 15; i++) {
        if (i >= 10 && i < 15) {
            fwd_adj_lists_arr2[i].size = 1;
            fwd_adj_lists_arr2[i].values = new uint64_t[1]{100 + (i - 10)};
        } else {
            fwd_adj_lists_arr2[i].size = 0;
            fwd_adj_lists_arr2[i].values = nullptr;
        }
    }

    // Backward: each country has one city
    for (uint64_t i = 0; i < 20; i++) {
        if (i >= 100 && i < 105) {
            bwd_adj_lists_arr2[i].size = 1;
            bwd_adj_lists_arr2[i].values = new uint64_t[1]{10 + (i - 100)};
        } else {
            bwd_adj_lists_arr2[i].size = 0;
            bwd_adj_lists_arr2[i].values = nullptr;
        }
    }

    auto table2 = std::make_unique<Table>(15, 20, std::move(fwd_adj_lists_arr2), std::move(bwd_adj_lists_arr2));
    table2->columns = {"city", "country"};
    table2->name = "CityCountry";
    table2->cardinality = Cardinality::MANY_TO_ONE;

    schema.tables.clear();
    schema.tables.push_back(table.get()); // PersonCity
    schema.tables.push_back(table2.get());// CityCountry
    schema.map = &map;

    person_vec = nullptr;
    root = std::make_shared<FactorizedTreeElement>("person", person_vec);
    schema.root = root;

    column_ordering = {"person", "city", "country"};
    schema.column_ordering = &column_ordering;

    // Pipeline: scan(person) -> INLJoinPackedShared(person->city) -> INLJoinPackedGPCascadeShared(city->country) -> sink
    auto scan_op = std::make_unique<Scan<uint64_t>>("person");
    auto join_city = std::make_unique<INLJoinPackedShared<uint64_t>>("person", "city", true);
    scan_op->set_next_operator(std::move(join_city));

    auto join_country = std::make_unique<INLJoinPackedGPCascadeShared<uint64_t>>("city", "country", true);
    scan_op->next_op->set_next_operator(std::move(join_country));

    auto sink_op = std::make_unique<SinkPacked>();
    scan_op->next_op->next_op->set_next_operator(std::move(sink_op));

    scan_op->init(&schema);

    // Verify shared state
    auto* city_vec = map.get_vector("city");
    auto* country_vec = map.get_vector("country");
    auto* person_vec_check = map.get_vector("person");

    ASSERT_NE(city_vec, nullptr);
    ASSERT_NE(country_vec, nullptr);
    ASSERT_NE(person_vec_check, nullptr);

    // All should share state (n:1 forward chain)
    EXPECT_EQ(city_vec->state, person_vec_check->state);
    EXPECT_EQ(country_vec->state, person_vec_check->state);
    EXPECT_TRUE(city_vec->has_identity_rle());
    EXPECT_TRUE(country_vec->has_identity_rle());

    scan_op->execute();

    // Verify output: 5 persons -> 5 cities -> 5 countries
    SinkPacked* sink = dynamic_cast<SinkPacked*>(scan_op->next_op->next_op->next_op);
    ASSERT_NE(sink, nullptr);
    EXPECT_EQ(sink->get_num_output_tuples(), 5);

    // Cleanup
    if (table2 && table2->fwd_adj_lists) {
        for (uint64_t i = 0; i < 15; i++) {
            if (table2->fwd_adj_lists[i].values) { delete[] table2->fwd_adj_lists[i].values; }
        }
    }
    if (table2 && table2->bwd_adj_lists) {
        for (uint64_t i = 0; i < 20; i++) {
            if (table2->bwd_adj_lists[i].values) { delete[] table2->bwd_adj_lists[i].values; }
        }
    }
}

TEST_F(SharedStateOperatorTest, INLJoinPackedGPCascadePredicatedShared_WithPredicate) {
    // Test GP cascade predicated join with shared state
    // Setup similar to above but with predicate: country < 103
    auto fwd_adj_lists_arr2 = std::make_unique<AdjList<uint64_t>[]>(15);
    auto bwd_adj_lists_arr2 = std::make_unique<AdjList<uint64_t>[]>(20);

    for (uint64_t i = 0; i < 15; i++) {
        if (i >= 10 && i < 15) {
            fwd_adj_lists_arr2[i].size = 1;
            fwd_adj_lists_arr2[i].values = new uint64_t[1]{100 + (i - 10)};
        } else {
            fwd_adj_lists_arr2[i].size = 0;
            fwd_adj_lists_arr2[i].values = nullptr;
        }
    }

    for (uint64_t i = 0; i < 20; i++) {
        if (i >= 100 && i < 105) {
            bwd_adj_lists_arr2[i].size = 1;
            bwd_adj_lists_arr2[i].values = new uint64_t[1]{10 + (i - 100)};
        } else {
            bwd_adj_lists_arr2[i].size = 0;
            bwd_adj_lists_arr2[i].values = nullptr;
        }
    }

    auto table2 = std::make_unique<Table>(15, 20, std::move(fwd_adj_lists_arr2), std::move(bwd_adj_lists_arr2));
    table2->columns = {"city", "country"};
    table2->name = "CityCountry";
    table2->cardinality = Cardinality::MANY_TO_ONE;

    schema.tables.clear();
    schema.tables.push_back(table.get());
    schema.tables.push_back(table2.get());
    schema.map = &map;

    person_vec = nullptr;
    root = std::make_shared<FactorizedTreeElement>("person", person_vec);
    schema.root = root;

    column_ordering = {"person", "city", "country"};
    schema.column_ordering = &column_ordering;

    // Build predicate: country < 103
    PredicateExpression pred_expr_raw = PredicateParser::parse_predicates("LT(country,103)");

    // Pipeline: scan(person) -> INLJoinPackedShared(person->city) -> INLJoinPackedGPCascadePredicatedShared(city->country, country < 103) -> sink
    auto scan_op = std::make_unique<Scan<uint64_t>>("person");
    auto join_city = std::make_unique<INLJoinPackedShared<uint64_t>>("person", "city", true);
    scan_op->set_next_operator(std::move(join_city));

    auto join_country =
            std::make_unique<INLJoinPackedGPCascadePredicatedShared<uint64_t>>("city", "country", true, pred_expr_raw);
    scan_op->next_op->set_next_operator(std::move(join_country));

    auto sink_op = std::make_unique<SinkPacked>();
    scan_op->next_op->next_op->set_next_operator(std::move(sink_op));

    scan_op->init(&schema);

    // Verify shared state
    auto* city_vec = map.get_vector("city");
    auto* country_vec = map.get_vector("country");
    auto* person_vec_check = map.get_vector("person");

    ASSERT_NE(city_vec, nullptr);
    ASSERT_NE(country_vec, nullptr);
    ASSERT_NE(person_vec_check, nullptr);

    EXPECT_EQ(city_vec->state, person_vec_check->state);
    EXPECT_EQ(country_vec->state, person_vec_check->state);
    EXPECT_TRUE(city_vec->has_identity_rle());
    EXPECT_TRUE(country_vec->has_identity_rle());

    scan_op->execute();

    // Verify output: persons 0,1,2 -> cities 10,11,12 -> countries 100,101,102 (country < 103 filters out 103,104)
    SinkPacked* sink = dynamic_cast<SinkPacked*>(scan_op->next_op->next_op->next_op);
    ASSERT_NE(sink, nullptr);
    EXPECT_EQ(sink->get_num_output_tuples(), 3);

    // Cleanup
    if (table2 && table2->fwd_adj_lists) {
        for (uint64_t i = 0; i < 15; i++) {
            if (table2->fwd_adj_lists[i].values) { delete[] table2->fwd_adj_lists[i].values; }
        }
    }
    if (table2 && table2->bwd_adj_lists) {
        for (uint64_t i = 0; i < 20; i++) {
            if (table2->bwd_adj_lists[i].values) { delete[] table2->bwd_adj_lists[i].values; }
        }
    }
}
