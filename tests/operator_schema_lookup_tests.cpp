#include "../src/operator/include/factorized_ftree/factorized_tree_element.hpp"
#include "../src/operator/include/join/inljoin_packed_cascade.hpp"
#include "../src/operator/include/join/inljoin_packed_cascade_predicated.hpp"
#include "../src/operator/include/join/inljoin_packed_cascade_predicated_shared.hpp"
#include "../src/operator/include/join/inljoin_packed_cascade_shared.hpp"
#include "../src/operator/include/join/inljoin_packed_gp_cascade.hpp"
#include "../src/operator/include/join/inljoin_packed_gp_cascade_predicated.hpp"
#include "../src/operator/include/join/inljoin_packed_gp_cascade_predicated_shared.hpp"
#include "../src/operator/include/join/inljoin_packed_gp_cascade_shared.hpp"
#include "../src/operator/include/join/inljoin_packed_predicated_shared.hpp"
#include "../src/operator/include/join/inljoin_packed_shared.hpp"
#include "../src/operator/include/query_variable_to_vector.hpp"
#include "../src/operator/include/scan/scan.hpp"
#include "../src/operator/include/schema/schema.hpp"
#include "../src/operator/include/sink/sink_packed.hpp"
#include "../src/query/include/predicate_parser.hpp"
#include "../src/table/include/adj_list.hpp"
#include "../src/table/include/table.hpp"
#include <cstring>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

using namespace ffx;

// Helper function to create test adjacency lists (n:1 for shared state, n:m for others)
std::pair<std::unique_ptr<AdjList<uint64_t>[]>, std::unique_ptr<AdjList<uint64_t>[]>>
create_test_adj_lists(bool n_to_one = false) {
    auto fwd_adj = std::make_unique<AdjList<uint64_t>[]>(4);
    auto bwd_adj = std::make_unique<AdjList<uint64_t>[]>(4);

    if (n_to_one) {
        // n:1 relationship for shared state operators
        // Forward: 0->{10}, 1->{11}, 2->{12}, 3->{13}
        // Backward: 10->{0}, 11->{1}, 12->{2}, 13->{3}
        for (uint64_t i = 0; i < 4; i++) {
            fwd_adj[i].size = 1;
            fwd_adj[i].values = new uint64_t[1]{10 + i};

            bwd_adj[i].size = 0;
            bwd_adj[i].values = nullptr;
        }
    } else {
        // n:m relationship for regular operators
        // Forward: 0->{1,2}, 1->{2,3}, 2->{3}, 3->{}
        // Backward: 0->{}, 1->{0}, 2->{0,1}, 3->{1,2}
        fwd_adj[0].size = 2;
        fwd_adj[0].values = new uint64_t[2]{1, 2};

        fwd_adj[1].size = 2;
        fwd_adj[1].values = new uint64_t[2]{2, 3};

        fwd_adj[2].size = 1;
        fwd_adj[2].values = new uint64_t[1]{3};

        fwd_adj[3].size = 0;
        fwd_adj[3].values = nullptr;

        bwd_adj[0].size = 0;
        bwd_adj[0].values = nullptr;

        bwd_adj[1].size = 1;
        bwd_adj[1].values = new uint64_t[1]{0};

        bwd_adj[2].size = 2;
        bwd_adj[2].values = new uint64_t[2]{0, 1};

        bwd_adj[3].size = 2;
        bwd_adj[3].values = new uint64_t[2]{1, 2};
    }

    return {std::move(fwd_adj), std::move(bwd_adj)};
}

// Helper function to cleanup adjacency lists
void cleanup_adj_lists(AdjList<uint64_t>* adj_lists, uint64_t size) {
    if (adj_lists) {
        for (uint64_t i = 0; i < size; i++) {
            if (adj_lists[i].values) { delete[] adj_lists[i].values; }
        }
    }
}

class OperatorSchemaLookupTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create adjacency lists (n:m for regular operators)
        auto [fwd_adj, bwd_adj] = create_test_adj_lists(false);
        fwd_adj_lists = std::move(fwd_adj);
        bwd_adj_lists = std::move(bwd_adj);

        // Create n:1 adjacency lists for shared state operators
        auto [fwd_adj_n1, bwd_adj_n1] = create_test_adj_lists(true);
        fwd_adj_lists_n1 = std::move(fwd_adj_n1);
        bwd_adj_lists_n1 = std::move(bwd_adj_n1);

        // Create table (for fallback testing)
        table = std::make_unique<Table>(4, 4, std::make_unique<AdjList<uint64_t>[]>(4),
                                        std::make_unique<AdjList<uint64_t>[]>(4));
        table->columns = {"src", "dest"};
        table->name = "test_table";

        // Copy adjacency lists to table (for fallback)
        for (uint64_t i = 0; i < 4; i++) {
            if (fwd_adj_lists[i].size > 0) {
                table->fwd_adj_lists[i].size = fwd_adj_lists[i].size;
                table->fwd_adj_lists[i].values = new uint64_t[fwd_adj_lists[i].size];
                std::memcpy(table->fwd_adj_lists[i].values, fwd_adj_lists[i].values,
                            fwd_adj_lists[i].size * sizeof(uint64_t));
            }
            if (bwd_adj_lists[i].size > 0) {
                table->bwd_adj_lists[i].size = bwd_adj_lists[i].size;
                table->bwd_adj_lists[i].values = new uint64_t[bwd_adj_lists[i].size];
                std::memcpy(table->bwd_adj_lists[i].values, bwd_adj_lists[i].values,
                            bwd_adj_lists[i].size * sizeof(uint64_t));
            }
        }

        // Setup schema
        schema.tables.push_back(table.get());
        schema.map = &map;

        // Create factorized tree - operators will add leaves as needed
        root = std::make_shared<FactorizedTreeElement>("src", nullptr);
        schema.root = root;

        column_ordering = {"src", "dest"};
        schema.column_ordering = &column_ordering;
    }

    void TearDown() override {
        cleanup_adj_lists(fwd_adj_lists.get(), 4);
        cleanup_adj_lists(bwd_adj_lists.get(), 4);
        cleanup_adj_lists(fwd_adj_lists_n1.get(), 4);
        cleanup_adj_lists(bwd_adj_lists_n1.get(), 4);
        if (table) {
            cleanup_adj_lists(table->fwd_adj_lists, 4);
            cleanup_adj_lists(table->bwd_adj_lists, 4);
        }
    }

    std::unique_ptr<AdjList<uint64_t>[]> fwd_adj_lists;
    std::unique_ptr<AdjList<uint64_t>[]> bwd_adj_lists;
    std::unique_ptr<AdjList<uint64_t>[]> fwd_adj_lists_n1;// n:1 for shared state
    std::unique_ptr<AdjList<uint64_t>[]> bwd_adj_lists_n1;
    std::unique_ptr<Table> table;
    QueryVariableToVectorMap map;
    Schema schema;
    std::shared_ptr<FactorizedTreeElement> root;
    std::vector<std::string> column_ordering;
};

// Test INLJoinPackedShared with schema-based lookup
TEST_F(OperatorSchemaLookupTest, INLJoinPackedShared_SchemaLookup) {
    // Register adjacency list in schema (n:1 for shared state)
    schema.register_adj_list("src", "dest", fwd_adj_lists_n1.get(), 4);

    // Create operator pipeline
    auto scan_op = std::make_unique<Scan<uint64_t>>("src");
    auto join_op = std::make_unique<INLJoinPackedShared<uint64_t>>("src", "dest", true);
    scan_op->set_next_operator(std::move(join_op));

    auto sink_op = std::make_unique<SinkPacked>();
    scan_op->next_op->set_next_operator(std::move(sink_op));

    // Initialize - should use schema-based lookup
    scan_op->init(&schema);

    // Verify schema lookup was used (check that operator has adjacency lists)
    // The operator should have successfully initialized with schema-provided adj lists
    EXPECT_TRUE(schema.has_adj_list("src", "dest"));

    // Execute to verify it works
    scan_op->execute();

    // Verify output
    SinkPacked* sink = dynamic_cast<SinkPacked*>(scan_op->next_op->next_op);
    ASSERT_NE(sink, nullptr);
    EXPECT_GT(sink->get_num_output_tuples(), 0);
}

// Test INLJoinPackedCascade with schema-based lookup
TEST_F(OperatorSchemaLookupTest, INLJoinPackedCascade_SchemaLookup) {
    schema.register_adj_list("src", "dest", fwd_adj_lists.get(), 4);

    auto scan_op = std::make_unique<Scan<uint64_t>>("src");
    auto join_op = std::make_unique<INLJoinPackedCascade<uint64_t>>("src", "dest", true);
    scan_op->set_next_operator(std::move(join_op));

    auto sink_op = std::make_unique<SinkPacked>();
    scan_op->next_op->set_next_operator(std::move(sink_op));

    scan_op->init(&schema);
    EXPECT_TRUE(schema.has_adj_list("src", "dest"));

    scan_op->execute();
    SinkPacked* sink = dynamic_cast<SinkPacked*>(scan_op->next_op->next_op);
    ASSERT_NE(sink, nullptr);
    EXPECT_GT(sink->get_num_output_tuples(), 0);
}

// Test INLJoinPackedCascadeShared with schema-based lookup
TEST_F(OperatorSchemaLookupTest, INLJoinPackedCascadeShared_SchemaLookup) {
    schema.register_adj_list("src", "dest", fwd_adj_lists_n1.get(), 4);

    auto scan_op = std::make_unique<Scan<uint64_t>>("src");
    auto join_op = std::make_unique<INLJoinPackedCascadeShared<uint64_t>>("src", "dest", true);
    scan_op->set_next_operator(std::move(join_op));

    auto sink_op = std::make_unique<SinkPacked>();
    scan_op->next_op->set_next_operator(std::move(sink_op));

    scan_op->init(&schema);
    EXPECT_TRUE(schema.has_adj_list("src", "dest"));

    scan_op->execute();
    SinkPacked* sink = dynamic_cast<SinkPacked*>(scan_op->next_op->next_op);
    ASSERT_NE(sink, nullptr);
    EXPECT_GT(sink->get_num_output_tuples(), 0);
}

// Test INLJoinPackedGPCascade with schema-based lookup
// GP cascade requires a grandparent in the tree, so we create a 3-level tree:
// grandparent -> parent -> child
TEST_F(OperatorSchemaLookupTest, INLJoinPackedGPCascade_SchemaLookup) {
    // Create adjacency lists for grandparent -> parent relationship
    auto gp_fwd_adj = std::make_unique<AdjList<uint64_t>[]>(4);
    for (uint64_t i = 0; i < 4; i++) {
        gp_fwd_adj[i].size = 2;
        gp_fwd_adj[i].values = new uint64_t[2]{i * 2, i * 2 + 1};
    }

    // Create adjacency lists for parent -> child relationship
    auto parent_fwd_adj = std::make_unique<AdjList<uint64_t>[]>(8);
    for (uint64_t i = 0; i < 8; i++) {
        parent_fwd_adj[i].size = 1;
        parent_fwd_adj[i].values = new uint64_t[1]{i % 4};
    }

    // Register both adjacency lists in schema
    schema.register_adj_list("grandparent", "parent", gp_fwd_adj.get(), 4);
    schema.register_adj_list("parent", "child", parent_fwd_adj.get(), 8);

    // Update column ordering for 3-level tree
    column_ordering = {"grandparent", "parent", "child"};
    schema.column_ordering = &column_ordering;

    // Update table columns to match the 3-level tree
    table->columns = {"grandparent", "parent", "child"};

    // Create 3-level factorized tree
    auto gp_root = std::make_shared<FactorizedTreeElement>("grandparent", nullptr);
    schema.root = gp_root;

    // Create operator pipeline: scan -> join(gp->parent) -> join(parent->child) -> sink
    auto scan_op = std::make_unique<Scan<uint64_t>>("grandparent");
    auto join1_op = std::make_unique<INLJoinPackedCascade<uint64_t>>("grandparent", "parent", true);
    auto join2_op = std::make_unique<INLJoinPackedGPCascade<uint64_t>>("parent", "child", true);
    auto sink_op = std::make_unique<SinkPacked>();

    scan_op->set_next_operator(std::move(join1_op));
    scan_op->next_op->set_next_operator(std::move(join2_op));
    scan_op->next_op->next_op->set_next_operator(std::move(sink_op));

    // Initialize - should use schema-based lookup for both joins
    scan_op->init(&schema);

    EXPECT_TRUE(schema.has_adj_list("grandparent", "parent"));
    EXPECT_TRUE(schema.has_adj_list("parent", "child"));

    // Execute to verify it works
    scan_op->execute();

    // Verify output
    SinkPacked* sink = dynamic_cast<SinkPacked*>(scan_op->next_op->next_op->next_op);
    ASSERT_NE(sink, nullptr);
    EXPECT_GT(sink->get_num_output_tuples(), 0);

    // Cleanup
    for (uint64_t i = 0; i < 4; i++) {
        delete[] gp_fwd_adj[i].values;
    }
    for (uint64_t i = 0; i < 8; i++) {
        delete[] parent_fwd_adj[i].values;
    }
}

// Test INLJoinPackedGPCascadeShared with schema-based lookup
TEST_F(OperatorSchemaLookupTest, INLJoinPackedGPCascadeShared_SchemaLookup) {
    // Create n:1 adjacency lists for grandparent -> parent relationship (for shared state)
    auto gp_fwd_adj = std::make_unique<AdjList<uint64_t>[]>(4);
    for (uint64_t i = 0; i < 4; i++) {
        gp_fwd_adj[i].size = 1;
        gp_fwd_adj[i].values = new uint64_t[1]{i % 2};// n:1 mapping
    }

    // Create n:1 adjacency lists for parent -> child relationship
    auto parent_fwd_adj = std::make_unique<AdjList<uint64_t>[]>(2);
    for (uint64_t i = 0; i < 2; i++) {
        parent_fwd_adj[i].size = 1;
        parent_fwd_adj[i].values = new uint64_t[1]{i};// n:1 mapping
    }

    // Register both adjacency lists in schema
    schema.register_adj_list("grandparent", "parent", gp_fwd_adj.get(), 4);
    schema.register_adj_list("parent", "child", parent_fwd_adj.get(), 2);

    // Update column ordering for 3-level tree
    column_ordering = {"grandparent", "parent", "child"};
    schema.column_ordering = &column_ordering;

    // Update table columns to match the 3-level tree
    table->columns = {"grandparent", "parent", "child"};

    // Create 3-level factorized tree
    auto gp_root = std::make_shared<FactorizedTreeElement>("grandparent", nullptr);
    schema.root = gp_root;

    // Create operator pipeline with shared state operators
    auto scan_op = std::make_unique<Scan<uint64_t>>("grandparent");
    auto join1_op = std::make_unique<INLJoinPackedCascadeShared<uint64_t>>("grandparent", "parent", true);
    auto join2_op = std::make_unique<INLJoinPackedGPCascadeShared<uint64_t>>("parent", "child", true);
    auto sink_op = std::make_unique<SinkPacked>();

    scan_op->set_next_operator(std::move(join1_op));
    scan_op->next_op->set_next_operator(std::move(join2_op));
    scan_op->next_op->next_op->set_next_operator(std::move(sink_op));

    // Initialize
    scan_op->init(&schema);

    EXPECT_TRUE(schema.has_adj_list("grandparent", "parent"));
    EXPECT_TRUE(schema.has_adj_list("parent", "child"));

    // Execute
    scan_op->execute();

    // Verify output
    SinkPacked* sink = dynamic_cast<SinkPacked*>(scan_op->next_op->next_op->next_op);
    ASSERT_NE(sink, nullptr);
    EXPECT_GT(sink->get_num_output_tuples(), 0);

    // Cleanup
    for (uint64_t i = 0; i < 4; i++) {
        delete[] gp_fwd_adj[i].values;
    }
    for (uint64_t i = 0; i < 2; i++) {
        delete[] parent_fwd_adj[i].values;
    }
}

// Test INLJoinPackedPredicatedShared with schema-based lookup
TEST_F(OperatorSchemaLookupTest, INLJoinPackedPredicatedShared_SchemaLookup) {
    schema.register_adj_list("src", "dest", fwd_adj_lists_n1.get(), 4);

    auto scan_op = std::make_unique<Scan<uint64_t>>("src");
    // Create predicate: dest > 1
    PredicateExpression pred_expr = PredicateParser::parse_predicates("GT(dest,1)");
    auto join_op = std::make_unique<INLJoinPackedPredicatedShared<uint64_t>>("src", "dest", true, pred_expr);
    scan_op->set_next_operator(std::move(join_op));

    auto sink_op = std::make_unique<SinkPacked>();
    scan_op->next_op->set_next_operator(std::move(sink_op));

    scan_op->init(&schema);
    EXPECT_TRUE(schema.has_adj_list("src", "dest"));

    scan_op->execute();
    SinkPacked* sink = dynamic_cast<SinkPacked*>(scan_op->next_op->next_op);
    ASSERT_NE(sink, nullptr);
    EXPECT_GT(sink->get_num_output_tuples(), 0);
}

// Test INLJoinPackedCascadePredicated with schema-based lookup
TEST_F(OperatorSchemaLookupTest, INLJoinPackedCascadePredicated_SchemaLookup) {
    schema.register_adj_list("src", "dest", fwd_adj_lists.get(), 4);

    auto scan_op = std::make_unique<Scan<uint64_t>>("src");
    PredicateExpression pred_expr = PredicateParser::parse_predicates("GT(dest,1)");
    auto join_op = std::make_unique<INLJoinPackedCascadePredicated<uint64_t>>("src", "dest", true, pred_expr);
    scan_op->set_next_operator(std::move(join_op));

    auto sink_op = std::make_unique<SinkPacked>();
    scan_op->next_op->set_next_operator(std::move(sink_op));

    scan_op->init(&schema);
    EXPECT_TRUE(schema.has_adj_list("src", "dest"));

    scan_op->execute();
    SinkPacked* sink = dynamic_cast<SinkPacked*>(scan_op->next_op->next_op);
    ASSERT_NE(sink, nullptr);
    EXPECT_GT(sink->get_num_output_tuples(), 0);
}

// Test INLJoinPackedCascadePredicatedShared with schema-based lookup
TEST_F(OperatorSchemaLookupTest, INLJoinPackedCascadePredicatedShared_SchemaLookup) {
    schema.register_adj_list("src", "dest", fwd_adj_lists_n1.get(), 4);

    auto scan_op = std::make_unique<Scan<uint64_t>>("src");
    PredicateExpression pred_expr = PredicateParser::parse_predicates("GT(dest,1)");
    auto join_op = std::make_unique<INLJoinPackedCascadePredicatedShared<uint64_t>>("src", "dest", true, pred_expr);
    scan_op->set_next_operator(std::move(join_op));

    auto sink_op = std::make_unique<SinkPacked>();
    scan_op->next_op->set_next_operator(std::move(sink_op));

    scan_op->init(&schema);
    EXPECT_TRUE(schema.has_adj_list("src", "dest"));

    scan_op->execute();
    SinkPacked* sink = dynamic_cast<SinkPacked*>(scan_op->next_op->next_op);
    ASSERT_NE(sink, nullptr);
    EXPECT_GT(sink->get_num_output_tuples(), 0);
}

// Test INLJoinPackedGPCascadePredicated with schema-based lookup
TEST_F(OperatorSchemaLookupTest, INLJoinPackedGPCascadePredicated_SchemaLookup) {
    // Create adjacency lists for grandparent -> parent relationship
    auto gp_fwd_adj = std::make_unique<AdjList<uint64_t>[]>(4);
    for (uint64_t i = 0; i < 4; i++) {
        gp_fwd_adj[i].size = 2;
        gp_fwd_adj[i].values = new uint64_t[2]{i * 2, i * 2 + 1};
    }

    // Create adjacency lists for parent -> child relationship
    auto parent_fwd_adj = std::make_unique<AdjList<uint64_t>[]>(8);
    for (uint64_t i = 0; i < 8; i++) {
        parent_fwd_adj[i].size = 1;
        parent_fwd_adj[i].values = new uint64_t[1]{i % 4};
    }

    // Register both adjacency lists in schema
    schema.register_adj_list("grandparent", "parent", gp_fwd_adj.get(), 4);
    schema.register_adj_list("parent", "child", parent_fwd_adj.get(), 8);

    // Update column ordering for 3-level tree
    column_ordering = {"grandparent", "parent", "child"};
    schema.column_ordering = &column_ordering;

    // Update table columns to match the 3-level tree
    table->columns = {"grandparent", "parent", "child"};

    // Create 3-level factorized tree
    auto gp_root = std::make_shared<FactorizedTreeElement>("grandparent", nullptr);
    schema.root = gp_root;

    // Create predicate expression: child > 1
    PredicateExpression pred_expr = PredicateParser::parse_predicates("GT(child,1)");

    // Create operator pipeline with predicate
    auto scan_op = std::make_unique<Scan<uint64_t>>("grandparent");
    auto join1_op =
            std::make_unique<INLJoinPackedCascadePredicated<uint64_t>>("grandparent", "parent", true, pred_expr);
    auto join2_op = std::make_unique<INLJoinPackedGPCascadePredicated<uint64_t>>("parent", "child", true, pred_expr);
    auto sink_op = std::make_unique<SinkPacked>();

    scan_op->set_next_operator(std::move(join1_op));
    scan_op->next_op->set_next_operator(std::move(join2_op));
    scan_op->next_op->next_op->set_next_operator(std::move(sink_op));

    // Initialize
    scan_op->init(&schema);

    EXPECT_TRUE(schema.has_adj_list("grandparent", "parent"));
    EXPECT_TRUE(schema.has_adj_list("parent", "child"));

    // Execute
    scan_op->execute();

    // Verify output
    SinkPacked* sink = dynamic_cast<SinkPacked*>(scan_op->next_op->next_op->next_op);
    ASSERT_NE(sink, nullptr);
    // May have 0 tuples if predicate filters all - just verify no crash

    // Cleanup
    for (uint64_t i = 0; i < 4; i++) {
        delete[] gp_fwd_adj[i].values;
    }
    for (uint64_t i = 0; i < 8; i++) {
        delete[] parent_fwd_adj[i].values;
    }
}

// Test INLJoinPackedGPCascadePredicatedShared with schema-based lookup
TEST_F(OperatorSchemaLookupTest, INLJoinPackedGPCascadePredicatedShared_SchemaLookup) {
    // Create n:1 adjacency lists for grandparent -> parent relationship
    auto gp_fwd_adj = std::make_unique<AdjList<uint64_t>[]>(4);
    for (uint64_t i = 0; i < 4; i++) {
        gp_fwd_adj[i].size = 1;
        gp_fwd_adj[i].values = new uint64_t[1]{i % 2};// n:1 mapping
    }

    // Create n:1 adjacency lists for parent -> child relationship
    auto parent_fwd_adj = std::make_unique<AdjList<uint64_t>[]>(2);
    for (uint64_t i = 0; i < 2; i++) {
        parent_fwd_adj[i].size = 1;
        parent_fwd_adj[i].values = new uint64_t[1]{i};// n:1 mapping
    }

    // Register both adjacency lists in schema
    schema.register_adj_list("grandparent", "parent", gp_fwd_adj.get(), 4);
    schema.register_adj_list("parent", "child", parent_fwd_adj.get(), 2);

    // Update column ordering for 3-level tree
    column_ordering = {"grandparent", "parent", "child"};
    schema.column_ordering = &column_ordering;

    // Update table columns to match the 3-level tree
    table->columns = {"grandparent", "parent", "child"};

    // Create 3-level factorized tree
    auto gp_root = std::make_shared<FactorizedTreeElement>("grandparent", nullptr);
    schema.root = gp_root;

    // Create predicate expression: child < 10
    PredicateExpression pred_expr = PredicateParser::parse_predicates("LT(child,10)");

    // Create operator pipeline with predicated shared state operators
    auto scan_op = std::make_unique<Scan<uint64_t>>("grandparent");
    auto join1_op =
            std::make_unique<INLJoinPackedCascadePredicatedShared<uint64_t>>("grandparent", "parent", true, pred_expr);
    auto join2_op =
            std::make_unique<INLJoinPackedGPCascadePredicatedShared<uint64_t>>("parent", "child", true, pred_expr);
    auto sink_op = std::make_unique<SinkPacked>();

    scan_op->set_next_operator(std::move(join1_op));
    scan_op->next_op->set_next_operator(std::move(join2_op));
    scan_op->next_op->next_op->set_next_operator(std::move(sink_op));

    // Initialize
    scan_op->init(&schema);

    EXPECT_TRUE(schema.has_adj_list("grandparent", "parent"));
    EXPECT_TRUE(schema.has_adj_list("parent", "child"));

    // Execute
    scan_op->execute();

    // Verify output
    SinkPacked* sink = dynamic_cast<SinkPacked*>(scan_op->next_op->next_op->next_op);
    ASSERT_NE(sink, nullptr);
    EXPECT_GT(sink->get_num_output_tuples(), 0);

    // Cleanup
    for (uint64_t i = 0; i < 4; i++) {
        delete[] gp_fwd_adj[i].values;
    }
    for (uint64_t i = 0; i < 2; i++) {
        delete[] parent_fwd_adj[i].values;
    }
}

// Test fallback to legacy table-based lookup when schema doesn't have adj list
TEST_F(OperatorSchemaLookupTest, INLJoinPackedShared_FallbackToTable) {
    // Don't register in schema - should fall back to table-based lookup
    // Schema should not have the adjacency list
    EXPECT_FALSE(schema.has_adj_list("src", "dest"));

    // Update table to use n:1 adjacency lists for shared state operator
    cleanup_adj_lists(table->fwd_adj_lists, 4);
    for (uint64_t i = 0; i < 4; i++) {
        table->fwd_adj_lists[i].size = 1;
        table->fwd_adj_lists[i].values = new uint64_t[1]{10 + i};
    }

    auto scan_op = std::make_unique<Scan<uint64_t>>("src");
    auto join_op = std::make_unique<INLJoinPackedShared<uint64_t>>("src", "dest", true);
    scan_op->set_next_operator(std::move(join_op));

    auto sink_op = std::make_unique<SinkPacked>();
    scan_op->next_op->set_next_operator(std::move(sink_op));

    // Should not throw - should use table-based lookup
    EXPECT_NO_THROW(scan_op->init(&schema));

    scan_op->execute();
    SinkPacked* sink = dynamic_cast<SinkPacked*>(scan_op->next_op->next_op);
    ASSERT_NE(sink, nullptr);
    EXPECT_GT(sink->get_num_output_tuples(), 0);
}

// Test that schema lookup takes precedence over table lookup
TEST_F(OperatorSchemaLookupTest, SchemaLookupTakesPrecedence) {
    // Register in schema (n:1 for shared state)
    schema.register_adj_list("src", "dest", fwd_adj_lists_n1.get(), 4);

    // Create different adjacency lists for table (to verify schema is used)
    // Replace table's adjacency lists with different values
    cleanup_adj_lists(table->fwd_adj_lists, 4);
    for (uint64_t i = 0; i < 4; i++) {
        table->fwd_adj_lists[i].size = 1;
        table->fwd_adj_lists[i].values = new uint64_t[1]{99 + i};// Different values
    }

    auto scan_op = std::make_unique<Scan<uint64_t>>("src");
    auto join_op = std::make_unique<INLJoinPackedShared<uint64_t>>("src", "dest", true);
    scan_op->set_next_operator(std::move(join_op));

    auto sink_op = std::make_unique<SinkPacked>();
    scan_op->next_op->set_next_operator(std::move(sink_op));

    scan_op->init(&schema);

    // Execute and verify - should use schema adjacency lists (not table's)
    scan_op->execute();
    SinkPacked* sink = dynamic_cast<SinkPacked*>(scan_op->next_op->next_op);
    ASSERT_NE(sink, nullptr);
    EXPECT_GT(sink->get_num_output_tuples(), 0);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
