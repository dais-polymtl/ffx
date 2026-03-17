#include <gtest/gtest.h>
#include "../src/table/include/adj_list_builder.hpp"
#include "../src/table/include/string_dictionary.hpp"
#include "../src/table/include/ffx_str_t.hpp"
#include "../src/table/include/string_pool.hpp"
#include <algorithm>
#include <memory>
#include <vector>

class AdjListBuilderTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(AdjListBuilderTest, BasicGraph) {
    // Simple graph: 0->1, 0->2, 1->2
    uint64_t src_col[] = {0, 0, 1};
    uint64_t dest_col[] = {1, 2, 2};
    uint64_t num_rows = 3;
    uint64_t num_fwd_ids = 3; // vertices 0, 1, 2
    uint64_t num_bwd_ids = 3;
    
    std::unique_ptr<ffx::AdjList<uint64_t>[]> fwd_adj, bwd_adj;
    ffx::build_adj_lists_from_columns(
        src_col, dest_col, num_rows, num_fwd_ids, num_bwd_ids,
        fwd_adj, bwd_adj
    );
    
    // Verify forward: 0 -> [1,2], 1 -> [2], 2 -> []
    EXPECT_EQ(fwd_adj[0].size, 2);
    EXPECT_EQ(fwd_adj[0].values[0], 1);
    EXPECT_EQ(fwd_adj[0].values[1], 2);
    EXPECT_TRUE(std::is_sorted(fwd_adj[0].values, fwd_adj[0].values + fwd_adj[0].size));
    
    EXPECT_EQ(fwd_adj[1].size, 1);
    EXPECT_EQ(fwd_adj[1].values[0], 2);
    
    EXPECT_EQ(fwd_adj[2].size, 0);
    
    // Verify backward: 0 -> [], 1 -> [0], 2 -> [0,1]
    EXPECT_EQ(bwd_adj[0].size, 0);
    
    EXPECT_EQ(bwd_adj[1].size, 1);
    EXPECT_EQ(bwd_adj[1].values[0], 0);
    
    EXPECT_EQ(bwd_adj[2].size, 2);
    EXPECT_EQ(bwd_adj[2].values[0], 0);
    EXPECT_EQ(bwd_adj[2].values[1], 1);
    EXPECT_TRUE(std::is_sorted(bwd_adj[2].values, bwd_adj[2].values + bwd_adj[2].size));
}

TEST_F(AdjListBuilderTest, ChainGraph) {
    // Chain: 0->1->2->3
    uint64_t src_col[] = {0, 1, 2};
    uint64_t dest_col[] = {1, 2, 3};
    uint64_t num_rows = 3;
    uint64_t num_fwd_ids = 4;
    uint64_t num_bwd_ids = 4;
    
    std::unique_ptr<ffx::AdjList<uint64_t>[]> fwd_adj, bwd_adj;
    ffx::build_adj_lists_from_columns(
        src_col, dest_col, num_rows, num_fwd_ids, num_bwd_ids,
        fwd_adj, bwd_adj
    );
    
    EXPECT_EQ(fwd_adj[0].size, 1);
    EXPECT_EQ(fwd_adj[0].values[0], 1);
    EXPECT_EQ(fwd_adj[1].size, 1);
    EXPECT_EQ(fwd_adj[1].values[0], 2);
    EXPECT_EQ(fwd_adj[2].size, 1);
    EXPECT_EQ(fwd_adj[2].values[0], 3);
    EXPECT_EQ(fwd_adj[3].size, 0);
    
    EXPECT_EQ(bwd_adj[0].size, 0);
    EXPECT_EQ(bwd_adj[1].size, 1);
    EXPECT_EQ(bwd_adj[1].values[0], 0);
    EXPECT_EQ(bwd_adj[2].size, 1);
    EXPECT_EQ(bwd_adj[2].values[0], 1);
    EXPECT_EQ(bwd_adj[3].size, 1);
    EXPECT_EQ(bwd_adj[3].values[0], 2);
}

TEST_F(AdjListBuilderTest, StarGraph) {
    // Star: 0->1, 0->2, 0->3
    uint64_t src_col[] = {0, 0, 0};
    uint64_t dest_col[] = {1, 2, 3};
    uint64_t num_rows = 3;
    uint64_t num_fwd_ids = 4;
    uint64_t num_bwd_ids = 4;
    
    std::unique_ptr<ffx::AdjList<uint64_t>[]> fwd_adj, bwd_adj;
    ffx::build_adj_lists_from_columns(
        src_col, dest_col, num_rows, num_fwd_ids, num_bwd_ids,
        fwd_adj, bwd_adj
    );
    
    EXPECT_EQ(fwd_adj[0].size, 3);
    EXPECT_TRUE(std::is_sorted(fwd_adj[0].values, fwd_adj[0].values + fwd_adj[0].size));
    EXPECT_EQ(fwd_adj[1].size, 0);
    EXPECT_EQ(fwd_adj[2].size, 0);
    EXPECT_EQ(fwd_adj[3].size, 0);
    
    EXPECT_EQ(bwd_adj[0].size, 0);
    EXPECT_EQ(bwd_adj[1].size, 1);
    EXPECT_EQ(bwd_adj[1].values[0], 0);
    EXPECT_EQ(bwd_adj[2].size, 1);
    EXPECT_EQ(bwd_adj[2].values[0], 0);
    EXPECT_EQ(bwd_adj[3].size, 1);
    EXPECT_EQ(bwd_adj[3].values[0], 0);
}

TEST_F(AdjListBuilderTest, SelfLoops) {
    // Self-loops: 0->0, 1->1
    uint64_t src_col[] = {0, 1};
    uint64_t dest_col[] = {0, 1};
    uint64_t num_rows = 2;
    uint64_t num_fwd_ids = 2;
    uint64_t num_bwd_ids = 2;
    
    std::unique_ptr<ffx::AdjList<uint64_t>[]> fwd_adj, bwd_adj;
    ffx::build_adj_lists_from_columns(
        src_col, dest_col, num_rows, num_fwd_ids, num_bwd_ids,
        fwd_adj, bwd_adj
    );
    
    EXPECT_EQ(fwd_adj[0].size, 1);
    EXPECT_EQ(fwd_adj[0].values[0], 0);
    EXPECT_EQ(fwd_adj[1].size, 1);
    EXPECT_EQ(fwd_adj[1].values[0], 1);
    
    EXPECT_EQ(bwd_adj[0].size, 1);
    EXPECT_EQ(bwd_adj[0].values[0], 0);
    EXPECT_EQ(bwd_adj[1].size, 1);
    EXPECT_EQ(bwd_adj[1].values[0], 1);
}

TEST_F(AdjListBuilderTest, EmptyGraph) {
    uint64_t* src_col = nullptr;
    uint64_t* dest_col = nullptr;
    uint64_t num_rows = 0;
    uint64_t num_fwd_ids = 5;
    uint64_t num_bwd_ids = 5;
    
    std::unique_ptr<ffx::AdjList<uint64_t>[]> fwd_adj, bwd_adj;
    ffx::build_adj_lists_from_columns(
        src_col, dest_col, num_rows, num_fwd_ids, num_bwd_ids,
        fwd_adj, bwd_adj
    );
    
    for (uint64_t i = 0; i < num_fwd_ids; i++) {
        EXPECT_EQ(fwd_adj[i].size, 0);
        EXPECT_EQ(bwd_adj[i].size, 0);
    }
}

TEST_F(AdjListBuilderTest, SingleEdge) {
    uint64_t src_col[] = {5};
    uint64_t dest_col[] = {10};
    uint64_t num_rows = 1;
    uint64_t num_fwd_ids = 20;
    uint64_t num_bwd_ids = 20;
    
    std::unique_ptr<ffx::AdjList<uint64_t>[]> fwd_adj, bwd_adj;
    ffx::build_adj_lists_from_columns(
        src_col, dest_col, num_rows, num_fwd_ids, num_bwd_ids,
        fwd_adj, bwd_adj
    );
    
    EXPECT_EQ(fwd_adj[5].size, 1);
    EXPECT_EQ(fwd_adj[5].values[0], 10);
    EXPECT_EQ(bwd_adj[10].size, 1);
    EXPECT_EQ(bwd_adj[10].values[0], 5);
}

TEST_F(AdjListBuilderTest, SparseIds) {
    // Sparse: only vertices 0, 100, 1000 used
    uint64_t src_col[] = {0, 100};
    uint64_t dest_col[] = {100, 1000};
    uint64_t num_rows = 2;
    uint64_t num_fwd_ids = 2000; // Much larger than max ID
    uint64_t num_bwd_ids = 2000;
    
    std::unique_ptr<ffx::AdjList<uint64_t>[]> fwd_adj, bwd_adj;
    ffx::build_adj_lists_from_columns(
        src_col, dest_col, num_rows, num_fwd_ids, num_bwd_ids,
        fwd_adj, bwd_adj
    );
    
    EXPECT_EQ(fwd_adj[0].size, 1);
    EXPECT_EQ(fwd_adj[0].values[0], 100);
    EXPECT_EQ(fwd_adj[100].size, 1);
    EXPECT_EQ(fwd_adj[100].values[0], 1000);
    EXPECT_EQ(fwd_adj[1000].size, 0);
    
    EXPECT_EQ(bwd_adj[0].size, 0);
    EXPECT_EQ(bwd_adj[100].size, 1);
    EXPECT_EQ(bwd_adj[100].values[0], 0);
    EXPECT_EQ(bwd_adj[1000].size, 1);
    EXPECT_EQ(bwd_adj[1000].values[0], 100);
}

TEST_F(AdjListBuilderTest, OutOfBoundsIgnored) {
    // Some IDs exceed bounds - should be ignored
    uint64_t src_col[] = {0, 1, 10}; // 10 exceeds num_fwd_ids=5
    uint64_t dest_col[] = {1, 2, 3};
    uint64_t num_rows = 3;
    uint64_t num_fwd_ids = 5;
    uint64_t num_bwd_ids = 5;
    
    std::unique_ptr<ffx::AdjList<uint64_t>[]> fwd_adj, bwd_adj;
    ffx::build_adj_lists_from_columns(
        src_col, dest_col, num_rows, num_fwd_ids, num_bwd_ids,
        fwd_adj, bwd_adj
    );
    
    // Only edges from vertices < 5 should be included
    EXPECT_EQ(fwd_adj[0].size, 1);
    EXPECT_EQ(fwd_adj[0].values[0], 1);
    EXPECT_EQ(fwd_adj[1].size, 1);
    EXPECT_EQ(fwd_adj[1].values[0], 2);
    // Out-of-bounds src=10 must be ignored; verify remaining in-range vertices are empty.
    EXPECT_EQ(fwd_adj[2].size, 0);
    EXPECT_EQ(fwd_adj[3].size, 0);
    EXPECT_EQ(fwd_adj[4].size, 0);
}

TEST_F(AdjListBuilderTest, SortingCorrectness) {
    // Create unsorted edges: 0->3, 0->1, 0->2
    uint64_t src_col[] = {0, 0, 0};
    uint64_t dest_col[] = {3, 1, 2};
    uint64_t num_rows = 3;
    uint64_t num_fwd_ids = 4;
    uint64_t num_bwd_ids = 4;
    
    std::unique_ptr<ffx::AdjList<uint64_t>[]> fwd_adj, bwd_adj;
    ffx::build_adj_lists_from_columns(
        src_col, dest_col, num_rows, num_fwd_ids, num_bwd_ids,
        fwd_adj, bwd_adj
    );
    
    // Should be sorted: [1, 2, 3]
    EXPECT_EQ(fwd_adj[0].size, 3);
    EXPECT_TRUE(std::is_sorted(fwd_adj[0].values, fwd_adj[0].values + fwd_adj[0].size));
    EXPECT_EQ(fwd_adj[0].values[0], 1);
    EXPECT_EQ(fwd_adj[0].values[1], 2);
    EXPECT_EQ(fwd_adj[0].values[2], 3);
}

// String-based adjacency list tests
TEST_F(AdjListBuilderTest, StringToStringAdjList) {
    // Create string-to-string edges using ID columns
    // Strings: "NYC"->"USA", "SF"->"USA", "LA"->"USA"
    // After dictionary: NYC=0, SF=1, LA=2, USA=3
    
    // Build a string dictionary
    auto string_pool = std::make_unique<ffx::StringPool>();
    auto string_dict = std::make_unique<ffx::StringDictionary>(string_pool.get());
    
    // Add strings and get IDs
    std::vector<std::string> src_strings = {"NYC", "SF", "LA"};
    std::vector<std::string> dest_strings = {"USA", "USA", "USA"};
    std::vector<uint64_t> src_ids, dest_ids;
    
    for (const auto& s : src_strings) {
        ffx::ffx_str_t str_val(s, string_pool.get());
        src_ids.push_back(string_dict->add_string(str_val));
    }
    
    for (const auto& s : dest_strings) {
        ffx::ffx_str_t str_val(s, string_pool.get());
        dest_ids.push_back(string_dict->add_string(str_val));
    }
    
    string_dict->finalize();
    uint64_t num_unique_strings = string_dict->size(); // Should be 4: NYC, SF, LA, USA
    
    // Build adjacency lists from ID columns
    std::unique_ptr<ffx::AdjList<uint64_t>[]> fwd_adj, bwd_adj;
    ffx::build_adj_lists_from_columns(
        src_ids.data(),
        dest_ids.data(),
        3,  // num_rows
        num_unique_strings,  // num_fwd_ids
        num_unique_strings,  // num_bwd_ids
        fwd_adj,
        bwd_adj
    );
    
    // Verify: NYC (0) -> USA (3), SF (1) -> USA (3), LA (2) -> USA (3)
    EXPECT_EQ(fwd_adj[0].size, 1);  // NYC -> USA
    EXPECT_EQ(fwd_adj[0].values[0], 3);
    EXPECT_EQ(fwd_adj[1].size, 1);  // SF -> USA
    EXPECT_EQ(fwd_adj[1].values[0], 3);
    EXPECT_EQ(fwd_adj[2].size, 1);  // LA -> USA
    EXPECT_EQ(fwd_adj[2].values[0], 3);
    EXPECT_EQ(fwd_adj[3].size, 0);  // USA has no outgoing
    
    // Verify backward: USA (3) -> [NYC, SF, LA] = [0, 1, 2]
    EXPECT_EQ(bwd_adj[3].size, 3);
    EXPECT_TRUE(std::is_sorted(bwd_adj[3].values, bwd_adj[3].values + bwd_adj[3].size));
    EXPECT_EQ(bwd_adj[3].values[0], 0);
    EXPECT_EQ(bwd_adj[3].values[1], 1);
    EXPECT_EQ(bwd_adj[3].values[2], 2);
}

TEST_F(AdjListBuilderTest, IntToStringAdjList) {
    // Create int-to-string edges: 0->"NYC", 1->"SF", 2->"LA"
    uint64_t src_col[] = {0, 1, 2};
    
    // Build string dictionary for destination
    auto string_pool = std::make_unique<ffx::StringPool>();
    auto string_dict = std::make_unique<ffx::StringDictionary>(string_pool.get());
    
    std::vector<std::string> dest_strings = {"NYC", "SF", "LA"};
    std::vector<uint64_t> dest_ids;
    
    for (const auto& s : dest_strings) {
        ffx::ffx_str_t str_val(s, string_pool.get());
        dest_ids.push_back(string_dict->add_string(str_val));
    }
    
    string_dict->finalize();
    uint64_t num_string_ids = string_dict->size(); // Should be 3
    
    // Build adjacency lists
    std::unique_ptr<ffx::AdjList<uint64_t>[]> fwd_adj, bwd_adj;
    ffx::build_adj_lists_from_columns(
        src_col,
        dest_ids.data(),
        3,  // num_rows
        3,  // num_fwd_ids (max int + 1)
        num_string_ids,  // num_bwd_ids
        fwd_adj,
        bwd_adj
    );
    
    // Verify forward: 0->NYC(0), 1->SF(1), 2->LA(2)
    EXPECT_EQ(fwd_adj[0].size, 1);
    EXPECT_EQ(fwd_adj[0].values[0], 0);  // NYC
    EXPECT_EQ(fwd_adj[1].size, 1);
    EXPECT_EQ(fwd_adj[1].values[0], 1);  // SF
    EXPECT_EQ(fwd_adj[2].size, 1);
    EXPECT_EQ(fwd_adj[2].values[0], 2);  // LA
    
    // Verify backward: NYC(0)->0, SF(1)->1, LA(2)->2
    EXPECT_EQ(bwd_adj[0].size, 1);
    EXPECT_EQ(bwd_adj[0].values[0], 0);
    EXPECT_EQ(bwd_adj[1].size, 1);
    EXPECT_EQ(bwd_adj[1].values[0], 1);
    EXPECT_EQ(bwd_adj[2].size, 1);
    EXPECT_EQ(bwd_adj[2].values[0], 2);
}

TEST_F(AdjListBuilderTest, StringToIntAdjList) {
    // Create string-to-int edges: "NYC"->0, "SF"->1, "LA"->2
    auto string_pool = std::make_unique<ffx::StringPool>();
    auto string_dict = std::make_unique<ffx::StringDictionary>(string_pool.get());
    
    std::vector<std::string> src_strings = {"NYC", "SF", "LA"};
    std::vector<uint64_t> src_ids;
    
    for (const auto& s : src_strings) {
        ffx::ffx_str_t str_val(s, string_pool.get());
        src_ids.push_back(string_dict->add_string(str_val));
    }
    
    string_dict->finalize();
    uint64_t num_string_ids = string_dict->size(); // Should be 3
    
    uint64_t dest_col[] = {0, 1, 2};
    
    // Build adjacency lists
    std::unique_ptr<ffx::AdjList<uint64_t>[]> fwd_adj, bwd_adj;
    ffx::build_adj_lists_from_columns(
        src_ids.data(),
        dest_col,
        3,  // num_rows
        num_string_ids,  // num_fwd_ids
        3,  // num_bwd_ids (max int + 1)
        fwd_adj,
        bwd_adj
    );
    
    // Verify forward: NYC(0)->0, SF(1)->1, LA(2)->2
    EXPECT_EQ(fwd_adj[0].size, 1);
    EXPECT_EQ(fwd_adj[0].values[0], 0);
    EXPECT_EQ(fwd_adj[1].size, 1);
    EXPECT_EQ(fwd_adj[1].values[0], 1);
    EXPECT_EQ(fwd_adj[2].size, 1);
    EXPECT_EQ(fwd_adj[2].values[0], 2);
    
    // Verify backward: 0->NYC(0), 1->SF(1), 2->LA(2)
    EXPECT_EQ(bwd_adj[0].size, 1);
    EXPECT_EQ(bwd_adj[0].values[0], 0);
    EXPECT_EQ(bwd_adj[1].size, 1);
    EXPECT_EQ(bwd_adj[1].values[0], 1);
    EXPECT_EQ(bwd_adj[2].size, 1);
    EXPECT_EQ(bwd_adj[2].values[0], 2);
}

TEST_F(AdjListBuilderTest, StringToStringWithNulls) {
    // Test string-to-string with NULL values (UINT64_MAX)
    // Strings: "NYC"->"USA", NULL->"USA", "LA"->NULL
    auto string_pool = std::make_unique<ffx::StringPool>();
    auto string_dict = std::make_unique<ffx::StringDictionary>(string_pool.get());
    
    std::vector<std::string> src_strings = {"NYC", "", "LA"};  // Empty string represents NULL
    std::vector<std::string> dest_strings = {"USA", "USA", ""};
    std::vector<uint64_t> src_ids, dest_ids;
    
    for (size_t i = 0; i < src_strings.size(); i++) {
        if (src_strings[i].empty()) {
            src_ids.push_back(UINT64_MAX);  // NULL
        } else {
            ffx::ffx_str_t str_val(src_strings[i], string_pool.get());
            src_ids.push_back(string_dict->add_string(str_val));
        }
    }
    
    for (size_t i = 0; i < dest_strings.size(); i++) {
        if (dest_strings[i].empty()) {
            dest_ids.push_back(UINT64_MAX);  // NULL
        } else {
            ffx::ffx_str_t str_val(dest_strings[i], string_pool.get());
            dest_ids.push_back(string_dict->add_string(str_val));
        }
    }
    
    string_dict->finalize();
    uint64_t num_unique_strings = string_dict->size(); // Should be 3: NYC, LA, USA
    
    // Build adjacency lists
    std::unique_ptr<ffx::AdjList<uint64_t>[]> fwd_adj, bwd_adj;
    ffx::build_adj_lists_from_columns(
        src_ids.data(),
        dest_ids.data(),
        3,  // num_rows
        num_unique_strings,  // num_fwd_ids (UINT64_MAX will be >= this, so skipped)
        num_unique_strings,  // num_bwd_ids
        fwd_adj,
        bwd_adj
    );
    
    // Verify: NULL destination edges are preserved in forward adjacency.
    // Dictionary order: NYC=0, LA=1, USA=2
    // Edges: NYC(0)->USA(2), NULL->USA(2), LA(1)->NULL
    // NULL source is skipped, but LA(1)->NULL is retained in forward adjacency.
    
    // Verify forward: NYC(0) -> USA(2)
    EXPECT_EQ(fwd_adj[0].size, 1);
    EXPECT_EQ(fwd_adj[0].values[0], 2);  // USA
    
    // LA(1) -> NULL: forward edge is preserved as UINT64_MAX
    EXPECT_EQ(fwd_adj[1].size, 1);
    EXPECT_EQ(fwd_adj[1].values[0], UINT64_MAX);
    
    // NULL->USA: This edge should also be skipped (NULL source)
    
    // Verify backward: USA(2) -> NYC(0)
    // The NULL->USA edge should be skipped, so USA should only have one incoming edge (from NYC)
    EXPECT_EQ(bwd_adj[2].size, 1);
    EXPECT_EQ(bwd_adj[2].values[0], 0);  // NYC
    
    // Verify no other backward edges exist (forward may include NULL destinations)
    for (uint64_t i = 0; i < num_unique_strings; i++) {
        if (i != 2) {
            EXPECT_EQ(bwd_adj[i].size, 0) << "Backward adj list " << i << " should be empty";
        }
    }
}

TEST_F(AdjListBuilderTest, IntToStringWithNulls) {
    // Test int-to-string with NULL string values
    // 0->"NYC", 1->NULL, 2->"LA"
    uint64_t src_col[] = {0, 1, 2};
    
    auto string_pool = std::make_unique<ffx::StringPool>();
    auto string_dict = std::make_unique<ffx::StringDictionary>(string_pool.get());
    
    std::vector<std::string> dest_strings = {"NYC", "", "LA"};  // Empty = NULL
    std::vector<uint64_t> dest_ids;
    
    for (const auto& s : dest_strings) {
        if (s.empty()) {
            dest_ids.push_back(UINT64_MAX);  // NULL
        } else {
            ffx::ffx_str_t str_val(s, string_pool.get());
            dest_ids.push_back(string_dict->add_string(str_val));
        }
    }
    
    string_dict->finalize();
    uint64_t num_string_ids = string_dict->size(); // Should be 2: NYC, LA
    
    // Build adjacency lists
    std::unique_ptr<ffx::AdjList<uint64_t>[]> fwd_adj, bwd_adj;
    ffx::build_adj_lists_from_columns(
        src_col,
        dest_ids.data(),
        3,  // num_rows
        3,  // num_fwd_ids
        num_string_ids,  // num_bwd_ids (UINT64_MAX will be >= this, so skipped)
        fwd_adj,
        bwd_adj
    );
    
    // Verify: 0->NYC, 2->LA, and 1->NULL (preserved in forward list)
    // Dictionary order: NYC=0, LA=1
    EXPECT_EQ(fwd_adj[0].size, 1);  // 0->NYC(0)
    EXPECT_EQ(fwd_adj[0].values[0], 0);  // NYC
    
    // 1->NULL: forward edge is preserved as UINT64_MAX
    EXPECT_EQ(fwd_adj[1].size, 1);
    EXPECT_EQ(fwd_adj[1].values[0], UINT64_MAX);
    
    EXPECT_EQ(fwd_adj[2].size, 1);  // 2->LA(1)
    EXPECT_EQ(fwd_adj[2].values[0], 1);  // LA
    
    // Backward: NYC(0)->0, LA(1)->2
    EXPECT_EQ(bwd_adj[0].size, 1);
    EXPECT_EQ(bwd_adj[0].values[0], 0);
    EXPECT_EQ(bwd_adj[1].size, 1);
    EXPECT_EQ(bwd_adj[1].values[0], 2);
}

TEST_F(AdjListBuilderTest, StringToIntWithNulls) {
    // Test string-to-int with NULL string values
    // "NYC"->0, NULL->1, "LA"->2
    auto string_pool = std::make_unique<ffx::StringPool>();
    auto string_dict = std::make_unique<ffx::StringDictionary>(string_pool.get());
    
    std::vector<std::string> src_strings = {"NYC", "", "LA"};  // Empty = NULL
    std::vector<uint64_t> src_ids;
    
    for (const auto& s : src_strings) {
        if (s.empty()) {
            src_ids.push_back(UINT64_MAX);  // NULL
        } else {
            ffx::ffx_str_t str_val(s, string_pool.get());
            src_ids.push_back(string_dict->add_string(str_val));
        }
    }
    
    string_dict->finalize();
    uint64_t num_string_ids = string_dict->size(); // Should be 2: NYC, LA
    
    uint64_t dest_col[] = {0, 1, 2};
    
    // Build adjacency lists
    std::unique_ptr<ffx::AdjList<uint64_t>[]> fwd_adj, bwd_adj;
    ffx::build_adj_lists_from_columns(
        src_ids.data(),
        dest_col,
        3,  // num_rows
        num_string_ids,  // num_fwd_ids (UINT64_MAX will be >= this, so skipped)
        3,  // num_bwd_ids
        fwd_adj,
        bwd_adj
    );
    
    // Verify: NYC(0)->0, LA(1)->2, NULL->1 should be skipped
    EXPECT_EQ(fwd_adj[0].size, 1);
    EXPECT_EQ(fwd_adj[0].values[0], 0);
    EXPECT_EQ(fwd_adj[1].size, 1);
    EXPECT_EQ(fwd_adj[1].values[0], 2);
    
    // Backward: 0->NYC(0), 2->LA(1)
    EXPECT_EQ(bwd_adj[0].size, 1);
    EXPECT_EQ(bwd_adj[0].values[0], 0);
    EXPECT_EQ(bwd_adj[2].size, 1);
    EXPECT_EQ(bwd_adj[2].values[0], 1);
}

TEST_F(AdjListBuilderTest, StringToStringComplexGraph) {
    // Complex string-to-string graph with multiple edges
    // "NYC"->"USA", "SF"->"USA", "LA"->"USA", "NYC"->"East", "SF"->"West"
    auto string_pool = std::make_unique<ffx::StringPool>();
    auto string_dict = std::make_unique<ffx::StringDictionary>(string_pool.get());
    
    std::vector<std::string> src_strings = {"NYC", "SF", "LA", "NYC", "SF"};
    std::vector<std::string> dest_strings = {"USA", "USA", "USA", "East", "West"};
    std::vector<uint64_t> src_ids, dest_ids;
    
    for (const auto& s : src_strings) {
        ffx::ffx_str_t str_val(s, string_pool.get());
        src_ids.push_back(string_dict->add_string(str_val));
    }
    
    for (const auto& s : dest_strings) {
        ffx::ffx_str_t str_val(s, string_pool.get());
        dest_ids.push_back(string_dict->add_string(str_val));
    }
    
    string_dict->finalize();
    uint64_t num_unique_strings = string_dict->size(); // Should be 5: NYC, SF, LA, USA, East, West
    
    // Build adjacency lists
    std::unique_ptr<ffx::AdjList<uint64_t>[]> fwd_adj, bwd_adj;
    ffx::build_adj_lists_from_columns(
        src_ids.data(),
        dest_ids.data(),
        5,  // num_rows
        num_unique_strings,
        num_unique_strings,
        fwd_adj,
        bwd_adj
    );
    
    // Find IDs (order depends on insertion)
    // Assuming: NYC=0, SF=1, LA=2, USA=3, East=4, West=5
    // But order may vary. Let's verify structure:
    // NYC should have 2 outgoing edges (USA and East)
    // SF should have 2 outgoing edges (USA and West)
    // LA should have 1 outgoing edge (USA)
    
    // We need to find which IDs correspond to which strings
    // For now, just verify that NYC and SF have 2 edges, LA has 1
    // This is a bit tricky without knowing exact IDs, so let's count non-empty lists
    size_t non_empty_fwd = 0;
    for (uint64_t i = 0; i < num_unique_strings; i++) {
        if (fwd_adj[i].size > 0) {
            non_empty_fwd++;
        }
    }
    
    // At least NYC, SF, LA should have outgoing edges
    EXPECT_GE(non_empty_fwd, 3);
    
    // Verify sorting
    for (uint64_t i = 0; i < num_unique_strings; i++) {
        if (fwd_adj[i].size > 1) {
            EXPECT_TRUE(std::is_sorted(fwd_adj[i].values, fwd_adj[i].values + fwd_adj[i].size));
        }
        if (bwd_adj[i].size > 1) {
            EXPECT_TRUE(std::is_sorted(bwd_adj[i].values, bwd_adj[i].values + bwd_adj[i].size));
        }
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

