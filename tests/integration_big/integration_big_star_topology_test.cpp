/// Integration test: Star topology query with LLM at various positions.
///
/// Query: Q(ATitle, BTitle, CTitle, llm_out) := AB(ATitle, BTitle), AC(ATitle, CTitle), llm_out = LLM_MAP(...)
///
/// Both AB and AC use the same edges table (src→dst).
/// The ftree has ATitle as root with two children: BTitle and CTitle (star/branching topology).
///
/// Tests LLM placed at different positions in the star:
///   1. Right after ATitle (before both branches)
///   2. After ATitle and BTitle (between branches)
///   3. At the end (after all joins)

#include <filesystem>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "../../src/operator/include/schema/adj_list_manager.hpp"
#include "../../src/operator/include/schema/schema.hpp"
#include "../../src/plan/include/plan.hpp"
#include "../../src/plan/include/plan_tree.hpp"
#include "../../src/query/include/query.hpp"
#include "../../src/ser_der/include/serializer.hpp"
#include "../../src/table/include/string_dictionary.hpp"
#include "../../src/table/include/string_pool.hpp"
#include "../../src/table/include/table.hpp"
#include "../../src/table/include/table_loader.hpp"

namespace ffx {

class BigIntegrationStarTest : public ::testing::Test {
public:
    static std::string g_data_base;

protected:
    void SetUp() override {
        pool_ = std::make_unique<StringPool>();
        dict_ = std::make_unique<StringDictionary>(pool_.get());
        dict_->finalize();

        if (!g_data_base.empty()) {
            data_base = g_data_base;
        } else {
            data_base = "/home/sunny/work/SampleDB2/tests/integration_big/data/serialized";
        }
    }

    void run_star_test(const std::string& context_column_json,
                       const std::vector<std::string>& llm_ordering,
                       const std::string& label) {
        if (!std::filesystem::exists(data_base)) { GTEST_SKIP() << "Test data not found at " << data_base; }

        std::string edges_dir = data_base + "/edges";
        uint64_t num_ids = 3000;

        AdjListManager manager;

        // Star query: ATitle→BTitle, ATitle→CTitle
        std::string llm_cfg = R"({"provider":"demo","model":"demo","model_name":"demo_model",)"
                              R"("tuple_format":"FTREE","context_column":{)" + context_column_json + R"(},"prompt":"star test"})";
        std::string q_llm_str = "Q(ATitle, BTitle, CTitle, llm_out) := "
                                "AB(ATitle, BTitle), AC(ATitle, CTitle), "
                                "llm_out = LLM_MAP(" + llm_cfg + ")";
        Query q_llm(q_llm_str);

        std::string q_plain_str = "Q(ATitle, BTitle, CTitle) := "
                                  "AB(ATitle, BTitle), AC(ATitle, CTitle)";
        Query q_plain(q_plain_str);

        load_table_from_columns(edges_dir, "src", "dst", q_llm, manager);

        auto tab_edges = std::make_unique<Table>(num_ids, num_ids, nullptr, nullptr);
        tab_edges->name = "Edges";
        tab_edges->columns = {"src", "dst"};
        std::vector<const Table*> tables = {tab_edges.get()};

        auto setup_schema = [&](Schema& schema, const Query& q) {
            schema.dictionary = dict_.get();
            schema.adj_list_manager = &manager;
            // AB(ATitle, BTitle) -> edges(src, dst)
            schema.register_adj_list("ATitle", "BTitle", manager.get_or_load(edges_dir, "src", "dst", true), num_ids);
            schema.register_adj_list("BTitle", "ATitle", manager.get_or_load(edges_dir, "src", "dst", false), num_ids);
            // AC(ATitle, CTitle) -> edges(src, dst)
            schema.register_adj_list("ATitle", "CTitle", manager.get_or_load(edges_dir, "src", "dst", true), num_ids);
            schema.register_adj_list("CTitle", "ATitle", manager.get_or_load(edges_dir, "src", "dst", false), num_ids);

            if (q.has_llm_map()) {
                static std::string llm_cfg_str;
                llm_cfg_str = q.get_llm_map_config();
                static std::string llm_out_attr;
                llm_out_attr = q.get_llm_map_output_attr();
                schema.llm_config_str = &llm_cfg_str;
                schema.llm_output_attr = &llm_out_attr;
            }
        };

        // Plain query
        std::vector<std::string> ord_plain = {"ATitle", "BTitle", "CTitle"};
        auto [plan_plain, ftree_plain] = map_ordering_to_plan(q_plain, ord_plain, SINK_PACKED);
        Schema schema_plain;
        setup_schema(schema_plain, q_plain);
        plan_plain->init(tables, ftree_plain, &schema_plain);
        plan_plain->execute();
        uint64_t count_plain = plan_plain->get_num_output_tuples();

        // LLM query
        auto [plan_llm, ftree_llm] = map_ordering_to_plan(q_llm, llm_ordering, SINK_PACKED);
        Schema schema_llm;
        setup_schema(schema_llm, q_llm);
        plan_llm->init(tables, ftree_llm, &schema_llm);
        plan_llm->execute();
        uint64_t count_llm = plan_llm->get_num_output_tuples();

        std::cout << label << " - Plain: " << count_plain << " LLM: " << count_llm << std::endl;
        EXPECT_GT(count_plain, 0u) << "Plain query should produce results";
        EXPECT_EQ(count_llm, count_plain);
    }

    std::unique_ptr<StringPool> pool_;
    std::unique_ptr<StringDictionary> dict_;
    std::string data_base;
};

std::string BigIntegrationStarTest::g_data_base = "";

// LLM right after ATitle (before both branches)
// context_column:{ATitle} → llm_out placed after scan, before any join
TEST_F(BigIntegrationStarTest, LLMAfterScanBeforeBothBranches) {
    run_star_test(
        R"("ATitle":64)",
        {"ATitle", "llm_out", "BTitle", "CTitle"},
        "Star LLM after scan");
}

// LLM after ATitle and BTitle (between branches)
// context_column:{ATitle, BTitle} → llm_out placed after first branch join
TEST_F(BigIntegrationStarTest, LLMBetweenBranches) {
    run_star_test(
        R"("ATitle":64,"BTitle":64)",
        {"ATitle", "BTitle", "llm_out", "CTitle"},
        "Star LLM between branches");
}

// LLM at the end (after all joins)
// context_column:{ATitle, BTitle, CTitle} → llm_out placed last
TEST_F(BigIntegrationStarTest, LLMAtEnd) {
    run_star_test(
        R"("ATitle":64,"BTitle":64,"CTitle":64)",
        {"ATitle", "BTitle", "CTitle", "llm_out"},
        "Star LLM at end");
}

}// namespace ffx

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.find("--data_dir=") == 0) { ffx::BigIntegrationStarTest::g_data_base = arg.substr(11); }
    }

    return RUN_ALL_TESTS();
}
