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

class BigIntegrationLLMPos3Test : public ::testing::Test {
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

    std::unique_ptr<StringPool> pool_;
    std::unique_ptr<StringDictionary> dict_;
    std::string data_base;
};

std::string BigIntegrationLLMPos3Test::g_data_base = "";

TEST_F(BigIntegrationLLMPos3Test, LLMAfterFirstCrossEdgeJoin) {
    if (!std::filesystem::exists(data_base)) { GTEST_SKIP() << "Test data not found at " << data_base; }

    std::string papers_dir = data_base + "/papers";
    std::string edges_dir = data_base + "/edges";
    uint64_t num_ids = 3000;

    AdjListManager manager;

    std::string llm_cfg = R"({"provider":"demo","model":"demo","model_name":"demo_model",)"
                          R"("tuple_format":"FTREE","context_column":{"ATitle":64,"AAbs":64,"BTitle":64},"prompt":"PROMPT"})";
    std::string q_llm_str = "Q(ATitle, AAbs, BTitle, BAbs, CTitle, CAbs, llm_out) := "
                            "AB(ATitle, BTitle), BC(BTitle, CTitle), "
                            "ATitle(ATitle, AAbs), BAbs(BTitle, BAbs), CTitle(CTitle, CAbs), "
                            "llm_out = LLM_MAP(" + llm_cfg + ")";
    Query q_llm(q_llm_str);

    std::string q_plain_str = "Q(ATitle, AAbs, BTitle, BAbs, CTitle, CAbs) := "
                              "AB(ATitle, BTitle), BC(BTitle, CTitle), "
                              "ATitle(ATitle, AAbs), BAbs(BTitle, BAbs), CTitle(CTitle, CAbs)";
    Query q_plain(q_plain_str);

    load_table_from_columns(edges_dir, "src", "dst", q_llm, manager);
    load_table_from_columns(papers_dir, "pid", "abstract", q_llm, manager);

    auto tab_edges = std::make_unique<Table>(num_ids, num_ids, nullptr, nullptr);
    tab_edges->name = "Edges";
    tab_edges->columns = {"src", "dst"};
    auto tab_papers = std::make_unique<Table>(num_ids, num_ids, nullptr, nullptr);
    tab_papers->name = "Papers";
    tab_papers->columns = {"pid", "abstract"};
    std::vector<const Table*> tables = {tab_edges.get(), tab_papers.get()};

    auto setup_schema = [&](Schema& schema, const Query& q) {
        schema.dictionary = dict_.get();
        schema.adj_list_manager = &manager;
        schema.register_adj_list("ATitle", "BTitle", manager.get_or_load(edges_dir, "src", "dst", true), num_ids);
        schema.register_adj_list("BTitle", "ATitle", manager.get_or_load(edges_dir, "src", "dst", false), num_ids);
        schema.register_adj_list("BTitle", "CTitle", manager.get_or_load(edges_dir, "src", "dst", true), num_ids);
        schema.register_adj_list("CTitle", "BTitle", manager.get_or_load(edges_dir, "src", "dst", false), num_ids);
        schema.register_adj_list("ATitle", "AAbs", manager.get_or_load(papers_dir, "pid", "abstract", true), num_ids);
        schema.register_adj_list("AAbs", "ATitle", manager.get_or_load(papers_dir, "pid", "abstract", false), num_ids);
        schema.register_adj_list("BTitle", "BAbs", manager.get_or_load(papers_dir, "pid", "abstract", true), num_ids);
        schema.register_adj_list("BAbs", "BTitle", manager.get_or_load(papers_dir, "pid", "abstract", false), num_ids);
        schema.register_adj_list("CTitle", "CAbs", manager.get_or_load(papers_dir, "pid", "abstract", true), num_ids);
        schema.register_adj_list("CAbs", "CTitle", manager.get_or_load(papers_dir, "pid", "abstract", false), num_ids);

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
    std::vector<std::string> ord_plain = {"ATitle", "AAbs", "BTitle", "BAbs", "CTitle", "CAbs"};
    auto [plan_plain, ftree_plain] = map_ordering_to_plan(q_plain, ord_plain, SINK_PACKED);
    Schema schema_plain;
    setup_schema(schema_plain, q_plain);
    plan_plain->init(tables, ftree_plain, &schema_plain);
    plan_plain->execute();
    uint64_t count_plain = plan_plain->get_num_output_tuples();

    // LLM query
    std::vector<std::string> ord_llm = {"ATitle", "AAbs", "BTitle", "llm_out", "BAbs", "CTitle", "CAbs"};
    auto [plan_llm, ftree_llm] = map_ordering_to_plan(q_llm, ord_llm, SINK_PACKED);
    Schema schema_llm;
    setup_schema(schema_llm, q_llm);
    plan_llm->init(tables, ftree_llm, &schema_llm);
    plan_llm->execute();
    uint64_t count_llm = plan_llm->get_num_output_tuples();

    std::cout << "LLM Pos3 - Plain: " << count_plain << " LLM: " << count_llm << std::endl;
    EXPECT_EQ(count_plain, 2998u);
    EXPECT_EQ(count_llm, count_plain);
}

}// namespace ffx

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.find("--data_dir=") == 0) { ffx::BigIntegrationLLMPos3Test::g_data_base = arg.substr(11); }
    }

    return RUN_ALL_TESTS();
}
