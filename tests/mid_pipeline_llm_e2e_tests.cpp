/// End-to-end integration test: LLM Map operator inserted mid-pipeline.
///
/// Scenario (4 attributes, 3 binary relations):
///   R1(a, b),  R2(b, c),  R3(c, d)
///
/// With context_column:{a, b} the plan builder inserts Map after processing
/// a and b, BEFORE the join that produces c:
///
///   Scan(a) → Join(a→b) → Map(a,b → llm_out) → Join(b→c) → Join(c→d) → Sink
///
/// This exercises the mid-pipeline insertion path that previously crashed
/// because CartesianProduct's init() destroyed the tree structure that
/// downstream joins depend on.

#include <filesystem>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>

#include "../src/operator/include/schema/schema.hpp"
#include "../src/plan/include/plan.hpp"
#include "../src/plan/include/plan_tree.hpp"
#include "../src/query/include/query.hpp"
#include "../src/ser_der/include/serializer.hpp"
#include "../src/table/include/string_dictionary.hpp"
#include "../src/table/include/string_pool.hpp"
#include "../src/table/include/table.hpp"

namespace ffx {

class MidPipelineLLMTest : public ::testing::Test {
protected:
    void SetUp() override {
        pool_ = std::make_unique<StringPool>();
        dict_ = std::make_unique<StringDictionary>(pool_.get());
        dict_->finalize();
    }

    /// Build a Table for a binary relation src→dest.
    ///
    /// @param fwd  fwd[i] = list of dest IDs reachable from src ID i
    /// @param num_src  number of distinct source IDs (0..num_src-1)
    /// @param num_dest number of distinct dest IDs (0..num_dest-1)
    static std::unique_ptr<Table> make_table(
            const std::vector<std::vector<uint64_t>>& fwd,
            uint64_t num_src, uint64_t num_dest,
            const std::string& name,
            const std::string& src_col, const std::string& dest_col) {

        // Forward adjacency lists.
        auto fwd_adj = std::make_unique<AdjList<uint64_t>[]>(num_src);
        for (uint64_t i = 0; i < num_src; ++i) {
            const auto& nbrs = fwd[i];
            fwd_adj[i].values = new uint64_t[nbrs.size()];
            fwd_adj[i].size = nbrs.size();
            for (size_t j = 0; j < nbrs.size(); ++j) {
                fwd_adj[i].values[j] = nbrs[j];
            }
        }

        // Backward adjacency lists (invert forward).
        std::vector<std::vector<uint64_t>> bwd_vecs(num_dest);
        for (uint64_t i = 0; i < num_src; ++i) {
            for (auto d : fwd[i]) bwd_vecs[d].push_back(i);
        }
        auto bwd_adj = std::make_unique<AdjList<uint64_t>[]>(num_dest);
        for (uint64_t i = 0; i < num_dest; ++i) {
            bwd_adj[i].values = new uint64_t[bwd_vecs[i].size()];
            bwd_adj[i].size = bwd_vecs[i].size();
            for (size_t j = 0; j < bwd_vecs[i].size(); ++j) {
                bwd_adj[i].values[j] = bwd_vecs[i][j];
            }
        }

        auto table = std::make_unique<Table>(num_src, num_dest, std::move(fwd_adj), std::move(bwd_adj));
        table->name = name;
        table->columns = {src_col, dest_col};
        return table;
    }

    std::unique_ptr<StringPool> pool_;
    std::unique_ptr<StringDictionary> dict_;
};

// -----------------------------------------------------------------------
// Test: Mid-pipeline Map with 4 attributes and downstream joins.
//
// Data:
//   R1(a, b):  a=0→{0,1}, a=1→{1,2}
//   R2(b, c):  b=0→{0},   b=1→{0,1}, b=2→{1}
//   R3(c, d):  c=0→{0,1}, c=1→{1}
//
// Ordering: a, b, c, d  (plus synthetic llm_out)
// context_column: {a, b}  → Map inserted after b, before c
//
// Join results (a,b,c,d):
//   a=0,b=0: c∈{0}:     d∈{0,1}  → (0,0,0,0),(0,0,0,1)
//   a=0,b=1: c∈{0,1}:   d∈{0,1},{1} → (0,1,0,0),(0,1,0,1),(0,1,1,1)
//   a=1,b=1: c∈{0,1}:   d∈{0,1},{1} → (1,1,0,0),(1,1,0,1),(1,1,1,1)
//   a=1,b=2: c∈{1}:     d∈{1}       → (1,2,1,1)
// Total: 2 + 3 + 3 + 1 = 9 tuples
// -----------------------------------------------------------------------
TEST_F(MidPipelineLLMTest, FourAttributeMidPipelineMap) {
    // -- Build tables --
    // R1(a, b): a ∈ {0,1}, b ∈ {0,1,2}
    auto r1 = make_table(
        {{0, 1}, {1, 2}},   // a=0→{0,1}, a=1→{1,2}
        2, 3, "R1", "a", "b");

    // R2(b, c): b ∈ {0,1,2}, c ∈ {0,1}
    auto r2 = make_table(
        {{0}, {0, 1}, {1}}, // b=0→{0}, b=1→{0,1}, b=2→{1}
        3, 2, "R2", "b", "c");

    // R3(c, d): c ∈ {0,1}, d ∈ {0,1}
    auto r3 = make_table(
        {{0, 1}, {1}},      // c=0→{0,1}, c=1→{1}
        2, 2, "R3", "c", "d");

    // -- Build query with LLM_MAP depending on a and b --
    // context_column:{a,b} ensures Map is pushed down after a and b are processed.
    std::string llm_cfg =
        R"({"provider":"demo","model":"demo","model_name":"demo_model",)"
        R"("tuple_format":"FTREE","context_column":{"a":64,"b":64},"prompt":"mid-pipeline test"})";

    Query query("Q(a, b, c, d, llm_out) := R1(a, b), R2(b, c), R3(c, d), llm_out = LLM_MAP(" + llm_cfg + ")");
    ASSERT_TRUE(query.is_rule_syntax());
    ASSERT_TRUE(query.has_llm_map());
    EXPECT_EQ(query.get_llm_map_output_attr(), "llm_out");

    // -- Build plan --
    std::vector<std::string> ordering = {"a", "b", "llm_out", "c", "d"};
    auto [plan, ftree] = map_ordering_to_plan(query, ordering, SINK_PACKED);
    ASSERT_NE(plan, nullptr);

    // -- Register adjacency lists and set up schema --
    Schema schema;
    schema.dictionary = dict_.get();

    std::vector<const Table*> tables_vec = {r1.get(), r2.get(), r3.get()};

    // Register adj lists for each relation.
    schema.register_adj_list("a", "b", r1->fwd_adj_lists, r1->num_fwd_ids);
    schema.register_adj_list("b", "a", r1->bwd_adj_lists, r1->num_bwd_ids);
    schema.register_adj_list("b", "c", r2->fwd_adj_lists, r2->num_fwd_ids);
    schema.register_adj_list("c", "b", r2->bwd_adj_lists, r2->num_bwd_ids);
    schema.register_adj_list("c", "d", r3->fwd_adj_lists, r3->num_fwd_ids);
    schema.register_adj_list("d", "c", r3->bwd_adj_lists, r3->num_bwd_ids);

    // Wire LLM config into schema (normally done by benchmark harness).
    std::string llm_config_str = query.get_llm_map_config();
    std::string llm_output_attr = query.get_llm_map_output_attr();
    schema.llm_config_str = &llm_config_str;
    schema.llm_output_attr = &llm_output_attr;

    // -- Init & Execute --
    plan->init(tables_vec, ftree, &schema);
    plan->execute();

    // -- Verify --
    // 9 join result tuples, each gets an LLM output, so 9 output tuples.
    uint64_t output = plan->get_num_output_tuples();
    EXPECT_EQ(output, 9u) << "Expected 9 output tuples from the 4-attribute mid-pipeline join + LLM";
}

// -----------------------------------------------------------------------
// Test: Mid-pipeline Map does not crash with empty join results.
//
// R1(a, b):  a=0→{0}
// R2(b, c):  (empty — b=0 has no neighbours)
// R3(c, d):  c=0→{0}
//
// After Map processes (a,b), the downstream join b→c produces nothing.
// The pipeline should complete without crashing and report 0 output tuples.
// -----------------------------------------------------------------------
TEST_F(MidPipelineLLMTest, MidPipelineMapWithEmptyDownstream) {
    auto r1 = make_table({{0}}, 1, 1, "R1", "a", "b");

    // R2: b has 1 source ID but no forward edges (empty adj list).
    auto r2_fwd = std::make_unique<AdjList<uint64_t>[]>(1);
    r2_fwd[0].values = nullptr;
    r2_fwd[0].size = 0;
    auto r2_bwd = std::make_unique<AdjList<uint64_t>[]>(1);
    r2_bwd[0].values = nullptr;
    r2_bwd[0].size = 0;
    auto r2 = std::make_unique<Table>(1, 1, std::move(r2_fwd), std::move(r2_bwd));
    r2->name = "R2";
    r2->columns = {"b", "c"};

    auto r3 = make_table({{0}}, 1, 1, "R3", "c", "d");

    std::string llm_cfg =
        R"({"provider":"demo","model":"demo","model_name":"demo_model",)"
        R"("tuple_format":"FTREE","context_column":{"a":64,"b":64},"prompt":"empty test"})";

    Query query("Q(a, b, c, d, llm_out) := R1(a, b), R2(b, c), R3(c, d), llm_out = LLM_MAP(" + llm_cfg + ")");

    std::vector<std::string> ordering = {"a", "b", "llm_out", "c", "d"};
    auto [plan, ftree] = map_ordering_to_plan(query, ordering, SINK_PACKED);

    Schema schema;
    schema.dictionary = dict_.get();

    std::vector<const Table*> tables_vec = {r1.get(), r2.get(), r3.get()};

    schema.register_adj_list("a", "b", r1->fwd_adj_lists, r1->num_fwd_ids);
    schema.register_adj_list("b", "a", r1->bwd_adj_lists, r1->num_bwd_ids);
    schema.register_adj_list("b", "c", r2->fwd_adj_lists, r2->num_fwd_ids);
    schema.register_adj_list("c", "b", r2->bwd_adj_lists, r2->num_bwd_ids);
    schema.register_adj_list("c", "d", r3->fwd_adj_lists, r3->num_fwd_ids);
    schema.register_adj_list("d", "c", r3->bwd_adj_lists, r3->num_bwd_ids);

    std::string llm_config_str = query.get_llm_map_config();
    std::string llm_output_attr = query.get_llm_map_output_attr();
    schema.llm_config_str = &llm_config_str;
    schema.llm_output_attr = &llm_output_attr;

    EXPECT_NO_THROW({
        plan->init(tables_vec, ftree, &schema);
        plan->execute();
    });

    EXPECT_EQ(plan->get_num_output_tuples(), 0u);
}

// -----------------------------------------------------------------------
// Test: LLM at the END of pipeline (no downstream joins).
//
// R1(a, b), R2(b, c), R3(c, d)
// context_column covers all attributes → Map placed after all joins.
//
//   Scan(a) → Join(a→b) → Join(b→c) → Join(c→d) → Map → Sink
//
// This verifies end-of-pipeline placement still works after the refactoring.
// -----------------------------------------------------------------------
TEST_F(MidPipelineLLMTest, EndOfPipelineMap) {
    auto r1 = make_table({{0, 1}, {1}}, 2, 2, "R1", "a", "b");
    auto r2 = make_table({{0}, {0}},    2, 1, "R2", "b", "c");
    auto r3 = make_table({{0}},         1, 1, "R3", "c", "d");

    // context_column includes c and d — Map cannot be inserted until after d.
    std::string llm_cfg =
        R"({"provider":"demo","model":"demo","model_name":"demo_model",)"
        R"("tuple_format":"FTREE","context_column":{"a":64,"b":64,"c":64,"d":64},"prompt":"end test"})";

    Query query("Q(a, b, c, d, llm_out) := R1(a, b), R2(b, c), R3(c, d), llm_out = LLM_MAP(" + llm_cfg + ")");

    std::vector<std::string> ordering = {"a", "b", "c", "d", "llm_out"};
    auto [plan, ftree] = map_ordering_to_plan(query, ordering, SINK_PACKED);

    Schema schema;
    schema.dictionary = dict_.get();
    std::vector<const Table*> tables_vec = {r1.get(), r2.get(), r3.get()};

    schema.register_adj_list("a", "b", r1->fwd_adj_lists, r1->num_fwd_ids);
    schema.register_adj_list("b", "a", r1->bwd_adj_lists, r1->num_bwd_ids);
    schema.register_adj_list("b", "c", r2->fwd_adj_lists, r2->num_fwd_ids);
    schema.register_adj_list("c", "b", r2->bwd_adj_lists, r2->num_bwd_ids);
    schema.register_adj_list("c", "d", r3->fwd_adj_lists, r3->num_fwd_ids);
    schema.register_adj_list("d", "c", r3->bwd_adj_lists, r3->num_bwd_ids);

    std::string llm_config_str = query.get_llm_map_config();
    std::string llm_output_attr = query.get_llm_map_output_attr();
    schema.llm_config_str = &llm_config_str;
    schema.llm_output_attr = &llm_output_attr;

    plan->init(tables_vec, ftree, &schema);
    plan->execute();

    // a=0→{0,1}: b∈{0,1}: c∈{0},{0} → c=0: d∈{0} → (0,0,0,0),(0,1,0,0)
    // a=1→{1}:   b=1: c∈{0}         → c=0: d∈{0} → (1,1,0,0)
    // Total: 3 tuples
    EXPECT_EQ(plan->get_num_output_tuples(), 3u);
}

}// namespace ffx
