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

class BigIntegrationTest : public ::testing::Test {
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

std::string BigIntegrationTest::g_data_base = "";

TEST_F(BigIntegrationTest, CompareLLMAndPlainCount) {
    if (!std::filesystem::exists(data_base)) { GTEST_SKIP() << "Test data not found at " << data_base; }

    std::string papers_dir = data_base + "/papers";
    std::string edges_dir = data_base + "/edges";

    // Queries
    std::string llm_cfg = R"({"provider":"demo","model":"demo","model_name":"demo_model",)"
                          R"("tuple_format":"FTREE","context_column":{"ATitle":64},"prompt":"test"})";
    std::string q1_str = "Q(ATitle, AAbs, BTitle, BAbs, CTitle, CAbs, llm_out) := "
                         "AB(ATitle, BTitle), BC(BTitle, CTitle), "
                         "ATitle(ATitle,  AAbs), BAbs(BTitle, BAbs), CTitle(CTitle, CAbs), "
                         "llm_out = LLM_MAP(" +
                         llm_cfg + ")";
    Query q1(q1_str);

    std::string q2_str = "Q(ATitle, AAbs, BTitle, BAbs, CTitle, CAbs) := "
                         "AB(ATitle, BTitle), BC(BTitle, CTitle), "
                         "ATitle(ATitle,  AAbs), BAbs(BTitle, BAbs), CTitle(CTitle, CAbs)";
    Query q2(q2_str);

    // Load tables using AdjListManager
    std::cout << "Loading tables via AdjListManager..." << std::endl;
    AdjListManager manager;

    // Explicitly load required columns and build adj lists
    // Note: We use "src"->"dst" for edges and "pid"->"abstract" for papers
    load_table_from_columns(edges_dir, "src", "dst", q1, manager);
    load_table_from_columns(papers_dir, "pid", "abstract", q1, manager);

    uint64_t num_ids = 3000;

    // Create dummy Table objects for Plan::init (metadata only)
    auto tab_edges = std::make_unique<Table>(num_ids, num_ids, nullptr, nullptr);
    tab_edges->name = "Edges";
    tab_edges->columns = {"src", "dst"};

    auto tab_papers = std::make_unique<Table>(num_ids, num_ids, nullptr, nullptr);
    tab_papers->name = "Papers";
    tab_papers->columns = {"pid", "abstract"};

    std::vector<const Table*> tables = {tab_edges.get(), tab_papers.get()};

    // Common Schema setup helper
    auto setup_schema = [&](Schema& schema, const Query& q) {
        schema.dictionary = dict_.get();
        schema.adj_list_manager = &manager;

        // Register adjacency lists manually using the ones built by manager
        // AB(ATitle, BTitle) -> edges(src, dst)
        schema.register_adj_list("ATitle", "BTitle", manager.get_or_load(edges_dir, "src", "dst", true), num_ids);
        schema.register_adj_list("BTitle", "ATitle", manager.get_or_load(edges_dir, "src", "dst", false), num_ids);

        // BC(BTitle, CTitle) -> edges(src, dst)
        schema.register_adj_list("BTitle", "CTitle", manager.get_or_load(edges_dir, "src", "dst", true), num_ids);
        schema.register_adj_list("CTitle", "BTitle", manager.get_or_load(edges_dir, "src", "dst", false), num_ids);

        // ATitle(ATitle, AAbs) -> papers(pid, abstract)
        schema.register_adj_list("ATitle", "AAbs", manager.get_or_load(papers_dir, "pid", "abstract", true), num_ids);
        schema.register_adj_list("AAbs", "ATitle", manager.get_or_load(papers_dir, "pid", "abstract", false), num_ids);

        // BAbs(BTitle, BAbs) -> papers(pid, abstract)
        schema.register_adj_list("BTitle", "BAbs", manager.get_or_load(papers_dir, "pid", "abstract", true), num_ids);
        schema.register_adj_list("BAbs", "BTitle", manager.get_or_load(papers_dir, "pid", "abstract", false), num_ids);

        // CTitle(CTitle, CAbs) -> papers(pid, abstract)
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

    // 1. Plain Query
    std::cout << "Setting up Plain Query..." << std::endl;
    std::vector<std::string> ord2 = {"ATitle", "AAbs", "BTitle", "BAbs", "CTitle", "CAbs"};
    auto [plan2, ftree2] = map_ordering_to_plan(q2, ord2, SINK_PACKED);
    Schema schema2;
    setup_schema(schema2, q2);
    plan2->init(tables, ftree2, &schema2);

    std::cout << "Executing Plain Query..." << std::endl;
    plan2->execute();
    uint64_t count_plain = plan2->get_num_output_tuples();
    std::cout << "Plain Query Count: " << count_plain << std::endl;

    // 2. LLM Query
    std::cout << "Setting up LLM Query..." << std::endl;
    std::vector<std::string> ord1 = {"ATitle", "llm_out", "AAbs", "BTitle", "BAbs", "CTitle", "CAbs"};
    auto [plan1, ftree1] = map_ordering_to_plan(q1, ord1, SINK_PACKED);
    Schema schema1;
    setup_schema(schema1, q1);
    plan1->init(tables, ftree1, &schema1);

    std::cout << "Executing LLM Query..." << std::endl;
    plan1->execute();
    uint64_t count_llm = plan1->get_num_output_tuples();
    std::cout << "LLM Query Count: " << count_llm << std::endl;

    EXPECT_EQ(count_plain, 2998u);
    EXPECT_EQ(count_llm, count_plain);
}

}// namespace ffx

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.find("--data_dir=") == 0) { ffx::BigIntegrationTest::g_data_base = arg.substr(11); }
    }

    return RUN_ALL_TESTS();
}
