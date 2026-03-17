#include "../src/ser_der/include/serializer.hpp"
#include "../src/table/include/string_dictionary.hpp"
#include "../src/table/include/string_pool.hpp"
#include <filesystem>
#include <gtest/gtest.h>

namespace ffx {

class StringDictionaryIntegrationTests : public ::testing::Test {
protected:
    void SetUp() override {
        ser_dir = "/tmp/ffx_dict_integration";
        std::filesystem::create_directories(ser_dir);
        pool = std::make_unique<StringPool>();
    }

    void TearDown() override {
        if (std::filesystem::exists(ser_dir)) { std::filesystem::remove_all(ser_dir); }
        pool.reset();
    }

    std::string ser_dir;
    std::unique_ptr<StringPool> pool;
};

TEST_F(StringDictionaryIntegrationTests, SerializeDeserializeTableWithDictionary) {
    // Prepare a tiny table: 2 forward IDs, 2 backward IDs, one edge 0->1
    uint64_t num_fwd_ids = 2;
    uint64_t num_bwd_ids = 2;

    auto fwd_adj = std::make_unique<AdjList<uint64_t>[]>(num_fwd_ids);
    auto bwd_adj = std::make_unique<AdjList<uint64_t>[]>(num_bwd_ids);

    fwd_adj[0] = AdjList<uint64_t>(1);
    fwd_adj[0].values[0] = 1;
    fwd_adj[0].size = 1;

    bwd_adj[1] = AdjList<uint64_t>(1);
    bwd_adj[1].values[0] = 0;
    bwd_adj[1].size = 1;

    Table table(num_fwd_ids, num_bwd_ids, std::move(fwd_adj), std::move(bwd_adj));

    // Build a small dictionary mapping 0->"a", 1->"b"
    StringDictionary dict(pool.get());
    dict.add_string(ffx_str_t("a", pool.get()));
    dict.add_string(ffx_str_t("b", pool.get()));
    dict.finalize();

    // Serialize both table and dictionary via single API
    serialize(table, ser_dir, "src", "dest", &dict);

    // Deserialize table and dictionary
    StringDictionary loaded_dict(pool.get());
    auto loaded_table = deserialize(ser_dir, "src", "dest", &loaded_dict);

    ASSERT_NE(loaded_table, nullptr);
    EXPECT_EQ(loaded_table->num_fwd_ids, num_fwd_ids);
    EXPECT_EQ(loaded_table->num_bwd_ids, num_bwd_ids);

    // Check adjacency lists
    EXPECT_EQ(loaded_table->fwd_adj_lists[0].size, 1u);
    EXPECT_EQ(loaded_table->fwd_adj_lists[0].values[0], 1u);
    EXPECT_EQ(loaded_table->bwd_adj_lists[1].size, 1u);
    EXPECT_EQ(loaded_table->bwd_adj_lists[1].values[0], 0u);

    // Check dictionary round-trip
    EXPECT_TRUE(loaded_dict.is_finalized());
    EXPECT_EQ(loaded_dict.size(), dict.size());

    ffx_str_t a("a", pool.get());
    ffx_str_t b("b", pool.get());

    auto id_a = loaded_dict.get_id(a);
    auto id_b = loaded_dict.get_id(b);

    EXPECT_NE(id_a, UINT64_MAX);
    EXPECT_NE(id_b, UINT64_MAX);
    EXPECT_EQ(loaded_dict.get_string(id_a).to_string(), "a");
    EXPECT_EQ(loaded_dict.get_string(id_b).to_string(), "b");
}

TEST_F(StringDictionaryIntegrationTests, MultiColumnWithDictRoundTrip) {
    const int num_rows = 3000;
    std::string out_dir = ser_dir + "/multi_col_test";
    std::filesystem::create_directories(out_dir);

    // 1. Generate data: 2 uint64 columns + 1 string column
    std::vector<uint64_t> col0(num_rows);
    std::vector<uint64_t> col1(num_rows);
    std::vector<std::string> col2_raw(num_rows);

    StringDictionary dict(pool.get());
    for (int i = 0; i < num_rows; ++i) {
        col0[i] = static_cast<uint64_t>(i) * 10;
        col1[i] = static_cast<uint64_t>(i) * 100;
        // 100 unique strings repeated
        col2_raw[i] = "string_value_" + std::to_string(i % 100);
        dict.add_string(ffx_str_t(col2_raw[i], pool.get()));
    }
    dict.finalize();

    // Convert strings to IDs for serialization
    std::vector<uint64_t> col2_ids(num_rows);
    for (int i = 0; i < num_rows; ++i) {
        col2_ids[i] = dict.get_id(ffx_str_t(col2_raw[i], pool.get()));
    }

    // 2. Serialize columns
    serialize_uint64_column(col0, out_dir + "/col0.bin");
    serialize_uint64_column(col1, out_dir + "/col1.bin");
    serialize_uint64_column(col2_ids, out_dir + "/col2_ids.bin");

    // Serialize dictionary
    std::string dict_path = out_dir + "/dict.bin";
    std::ofstream dict_out(dict_path, std::ios::binary);
    ASSERT_TRUE(dict_out.is_open());
    dict.serialize(dict_out);
    dict_out.close();

    // 3. Deserialize
    uint64_t loaded_rows_0, loaded_rows_1, loaded_rows_2;
    auto loaded_col0 = deserialize_uint64_column(out_dir + "/col0.bin", loaded_rows_0);
    auto loaded_col1 = deserialize_uint64_column(out_dir + "/col1.bin", loaded_rows_1);
    auto loaded_col2_ids = deserialize_uint64_column(out_dir + "/col2_ids.bin", loaded_rows_2);

    StringDictionary loaded_dict(pool.get());
    std::ifstream dict_in(dict_path, std::ios::binary);
    ASSERT_TRUE(dict_in.is_open());
    loaded_dict.deserialize(dict_in, pool.get());
    dict_in.close();

    // 4. Verify match
    EXPECT_EQ(loaded_rows_0, num_rows);
    EXPECT_EQ(loaded_rows_1, num_rows);
    EXPECT_EQ(loaded_rows_2, num_rows);
    EXPECT_EQ(loaded_dict.size(), 100u);// Only 100 unique strings

    for (int i = 0; i < num_rows; ++i) {
        EXPECT_EQ(loaded_col0[i], col0[i]);
        EXPECT_EQ(loaded_col1[i], col1[i]);

        uint64_t id = loaded_col2_ids[i];
        EXPECT_EQ(loaded_dict.get_string(id).to_string(), col2_raw[i]);
    }
}

TEST_F(StringDictionaryIntegrationTests, LargeIntegrationTest) {
    const int num_entries = 2000;

    // Create a large dictionary
    StringDictionary dict(pool.get());
    for (int i = 0; i < num_entries; ++i) {
        std::string s = "integration_string_" + std::to_string(i);
        dict.add_string(ffx_str_t(s, pool.get()));
    }
    dict.finalize();

    // Create a table where edges correspond to dictionary IDs
    auto fwd_adj = std::make_unique<AdjList<uint64_t>[]>(num_entries);
    auto bwd_adj = std::make_unique<AdjList<uint64_t>[]>(num_entries);

    for (int i = 0; i < num_entries - 1; ++i) {
        fwd_adj[i] = AdjList<uint64_t>(1);
        fwd_adj[i].values[0] = i + 1;
        fwd_adj[i].size = 1;

        bwd_adj[i + 1] = AdjList<uint64_t>(1);
        bwd_adj[i + 1].values[0] = i;
        bwd_adj[i + 1].size = 1;
    }

    Table table(num_entries, num_entries, std::move(fwd_adj), std::move(bwd_adj));

    // Serialize
    serialize(table, ser_dir, "s", "d", &dict);

    // Deserialize
    StringDictionary loaded_dict(pool.get());
    auto loaded_table = deserialize(ser_dir, "s", "d", &loaded_dict);

    ASSERT_NE(loaded_table, nullptr);
    EXPECT_EQ(loaded_table->num_fwd_ids, num_entries);
    EXPECT_EQ(loaded_dict.size(), num_entries);

    // Check random samples
    for (int i = 0; i < num_entries; i += 100) {
        std::string expected = "integration_string_" + std::to_string(i);
        EXPECT_EQ(loaded_dict.get_string(i).to_string(), expected);

        ffx_str_t lookup_str(expected, pool.get());
        EXPECT_EQ(loaded_dict.get_id(lookup_str), static_cast<uint64_t>(i));
    }
}

TEST_F(StringDictionaryIntegrationTests, AdjListWithDictRoundTrip) {
    const int num_rows = 3000;
    std::string out_dir = ser_dir + "/adj_list_dict_test";
    std::filesystem::create_directories(out_dir);

    // 1. Generate data
    StringDictionary dict(pool.get());
    std::vector<std::string> strings(num_rows);
    for (int i = 0; i < num_rows; ++i) {
        strings[i] = "long_string_entry_" + std::to_string(i % 50);// 50 unique strings
        dict.add_string(ffx_str_t(strings[i], pool.get()));
    }
    dict.finalize();

    // 2. Create Adjacency Lists
    // Let's say we have 3000 source nodes (col0), each pointing to one target string ID (col2)
    uint64_t num_fwd_ids = num_rows;
    uint64_t num_bwd_ids = 50;// Unique string IDs

    auto fwd_adj = std::make_unique<AdjList<uint64_t>[]>(num_fwd_ids);
    auto bwd_adj = std::make_unique<AdjList<uint64_t>[]>(num_bwd_ids);

    for (int i = 0; i < num_rows; ++i) {
        uint64_t str_id = dict.get_id(ffx_str_t(strings[i], pool.get()));

        fwd_adj[i] = AdjList<uint64_t>(1);
        fwd_adj[i].values[0] = str_id;
        fwd_adj[i].size = 1;

        // Add back-link to make it more realistic
        if (bwd_adj[str_id].size == 0) {
            bwd_adj[str_id] = AdjList<uint64_t>(num_rows);// Over-allocate for simplicity
            bwd_adj[str_id].size = 0;
        }
        bwd_adj[str_id].values[bwd_adj[str_id].size++] = i;
    }

    Table table(num_fwd_ids, num_bwd_ids, std::move(fwd_adj), std::move(bwd_adj));

    // 3. Serialize Table and Dictionary
    serialize(table, out_dir, "node", "string_id", &dict);

    // 4. Deserialize
    StringDictionary loaded_dict(pool.get());
    auto loaded_table = deserialize(out_dir, "node", "string_id", &loaded_dict);

    ASSERT_NE(loaded_table, nullptr);
    EXPECT_EQ(loaded_table->num_fwd_ids, num_fwd_ids);
    EXPECT_EQ(loaded_dict.size(), 50u);

    // 5. Verify the data match by looking up strings in the dictionary
    for (int i = 0; i < num_rows; ++i) {
        ASSERT_EQ(loaded_table->fwd_adj_lists[i].size, 1u);
        uint64_t loaded_id = loaded_table->fwd_adj_lists[i].values[0];

        std::string expected_str = strings[i];
        std::string actual_str = loaded_dict.get_string(loaded_id).to_string();

        EXPECT_EQ(actual_str, expected_str);
    }

    // Check back-links
    for (uint64_t id = 0; id < 50; ++id) {
        uint64_t expected_count = 0;
        for (int i = 0; i < num_rows; ++i) {
            if (i % 50 == static_cast<int>(id)) expected_count++;
        }
        EXPECT_EQ(loaded_table->bwd_adj_lists[id].size, expected_count);
    }
}

}// namespace ffx
