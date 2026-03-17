#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "../src/operator/include/join/inljoin_packed.hpp"
#include "../src/operator/include/query_variable_to_vector.hpp"
#include "../src/operator/include/scan/scan.hpp"
#include "../src/operator/include/schema/schema.hpp"
#include "../src/operator/include/sink/sink_packed.hpp"
#include "../src/plan/include/plan_tree.hpp"
#include "../src/plan/include/plan.hpp"
#include "../src/query/include/query.hpp"
#include "../src/ser_der/include/serializer.hpp"
#include "../src/table/include/string_dictionary.hpp"
#include "../src/table/include/string_pool.hpp"
#include "../src/table/include/table.hpp"

namespace ffx {

class DictionaryPipelineE2ETests : public ::testing::Test {
protected:
    void SetUp() override {
        ser_dir = "/tmp/ffx_e2e_dict_test";
        std::filesystem::remove_all(ser_dir);
        std::filesystem::create_directories(ser_dir);
        pool = std::make_unique<StringPool>();
    }

    void TearDown() override { std::filesystem::remove_all(ser_dir); }

    std::string ser_dir;
    std::unique_ptr<StringPool> pool;
};

TEST_F(DictionaryPipelineE2ETests, PipelineE2ERoundTrip) {
    const int num_rows = 1000;
    const int num_unique_strings = 100;

    std::cout << "[E2E] 1. Setup Data" << std::endl;
    StringDictionary dict(pool.get());
    std::vector<std::string> raw_strings;
    for (int i = 0; i < num_unique_strings; ++i) {
        raw_strings.push_back("user_name_" + std::to_string(i));
        dict.add_string(ffx_str_t(raw_strings.back(), pool.get()));
    }
    dict.finalize();

    auto fwd_adj = std::make_unique<AdjList<uint64_t>[]>(num_rows);
    auto bwd_adj = std::make_unique<AdjList<uint64_t>[]>(num_unique_strings);

    for (int i = 0; i < num_rows; ++i) {
        uint64_t str_id = i % num_unique_strings;
        fwd_adj[i].values = new uint64_t[1];
        fwd_adj[i].values[0] = str_id;
        fwd_adj[i].size = 1;
    }

    for (uint64_t str_id = 0; str_id < num_unique_strings; ++str_id) {
        bwd_adj[str_id].values = new uint64_t[num_rows / num_unique_strings];
        bwd_adj[str_id].size = 0;
    }
    for (int i = 0; i < num_rows; ++i) {
        uint64_t str_id = i % num_unique_strings;
        bwd_adj[str_id].values[bwd_adj[str_id].size++] = i;
    }

    Table table(num_rows, num_unique_strings, std::move(fwd_adj), std::move(bwd_adj));
    table.columns = {"user_id", "name_id"};
    table.name = "UsersTable";

    std::cout << "[E2E] 2. Serialize" << std::endl;
    serialize(table, ser_dir, "user_id", "name_id", &dict);

    std::cout << "[E2E] 3. Deserialize" << std::endl;
    StringPool loaded_pool;
    StringDictionary loaded_dict(&loaded_pool);
    auto loaded_table = deserialize(ser_dir, "user_id", "name_id", &loaded_dict);
    ASSERT_NE(loaded_table, nullptr);
    loaded_table->name = "UsersTable";
    loaded_table->columns = {"user_id", "name_id"};

    std::cout << "[E2E] 4. Build Pipeline via map_ordering_to_plan" << std::endl;
    Query query("Q(user_id, name_id) := UsersTable(user_id, name_id)");
    std::vector<std::string> ordering = {"user_id", "name_id"};
    std::vector<const Table*> tables_vec = {loaded_table.get()};

    auto [plan, ftree] = map_ordering_to_plan(query, ordering, SINK_PACKED);

    Schema schema;
    schema.dictionary = &loaded_dict;
    schema.tables = tables_vec;

    // AdjList registration must happen correctly
    schema.register_adj_list("user_id", "name_id", loaded_table->fwd_adj_lists, loaded_table->num_fwd_ids);

    std::cout << "[E2E] 5. Init Plan" << std::endl;
    plan->init(tables_vec, ftree, &schema);

    std::cout << "[E2E] 6. Execute Plan" << std::endl;
    plan->execute();

    std::cout << "[E2E] 7. Verify Results" << std::endl;
    EXPECT_EQ(plan->get_num_output_tuples(), num_rows);

    for (int i = 0; i < num_unique_strings; ++i) {
        uint64_t id = static_cast<uint64_t>(i);
        EXPECT_EQ(schema.dictionary->get_string(id).to_string(), "user_name_" + std::to_string(i));
    }
    std::cout << "[E2E] Done!" << std::endl;
}

}// namespace ffx
