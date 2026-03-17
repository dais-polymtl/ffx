#include "../src/operator/include/ai_operator/map.hpp"
#include "../src/operator/include/ai_operator/ftree_reconstructor.hpp"
#include "../src/operator/include/factorized_ftree/ftree_batch_iterator.hpp"
#include "../src/operator/include/schema/schema.hpp"
#include "../src/operator/include/vector/state.hpp"
#include "../src/operator/include/vector/vector.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

using namespace ffx;

class StreamingMapTest : public ::testing::Test {
protected:
    std::unique_ptr<Vector<uint64_t>> vecs_[20];
    std::shared_ptr<FactorizedTreeElement> root_;
    std::vector<std::string> ordering_;
    Schema schema_;
    QueryVariableToVectorMap map_;
    std::unique_ptr<StringPool> pool_;
    std::unique_ptr<StringDictionary> dictionary_;
    std::string llm_config_;
    std::string llm_output_name_;

    Vector<uint64_t>* V(int slot) { return vecs_[slot].get(); }

    /// Build FTREE `context_column` from `ordering_` and wire output column pointer.
    void sync_llm_config_from_ordering() {
        std::string ctx = "{";
        bool first = true;
        for (const auto& name : ordering_) {
            if (!first) ctx += ",";
            ctx += "\"" + name + "\":64";
            first = false;
        }
        ctx += "}";
        llm_config_ =
            R"({"provider":"demo","model":"demo","model_name":"demo_model","tuple_format":"FTREE","context_column":)" +
            ctx + R"(,"prompt":"test"})";
        schema_.llm_config_str = &llm_config_;
        llm_output_name_ = "_llm";
        schema_.llm_output_attr = &llm_output_name_;
    }

    void SetUp() override {
        schema_.map = &map_;
        pool_ = std::make_unique<StringPool>();
        dictionary_ = std::make_unique<StringDictionary>(pool_.get());
        dictionary_->finalize();
        schema_.predicate_pool = pool_.get();
        schema_.dictionary = dictionary_.get();
        llm_config_ = "{}";
        schema_.llm_config_str = &llm_config_;
        llm_output_name_.clear();
        schema_.llm_output_attr = nullptr;
    }

    // ---- Balanced 3-node tree: a->(b,c) ----
    //   a: [1,2]
    //   b: [10,11 | 12,13]  (2 per a)
    //   c: [20,21 | 22,23]  (2 per a)
    void build_balanced_tree() {
        vecs_[0] = std::make_unique<Vector<uint64_t>>();
        V(0)->values[0] = 1;
        V(0)->values[1] = 2;
        SET_ALL_BITS(V(0)->state->selector);
        SET_START_POS(*V(0)->state, 0);
        SET_END_POS(*V(0)->state, 1);

        vecs_[1] = std::make_unique<Vector<uint64_t>>();
        V(1)->values[0] = 10;
        V(1)->values[1] = 11;
        V(1)->values[2] = 12;
        V(1)->values[3] = 13;
        SET_ALL_BITS(V(1)->state->selector);
        SET_START_POS(*V(1)->state, 0);
        SET_END_POS(*V(1)->state, 3);
        V(1)->state->offset[0] = 0;
        V(1)->state->offset[1] = 2;
        V(1)->state->offset[2] = 4;

        vecs_[2] = std::make_unique<Vector<uint64_t>>();
        V(2)->values[0] = 20;
        V(2)->values[1] = 21;
        V(2)->values[2] = 22;
        V(2)->values[3] = 23;
        SET_ALL_BITS(V(2)->state->selector);
        SET_START_POS(*V(2)->state, 0);
        SET_END_POS(*V(2)->state, 3);
        V(2)->state->offset[0] = 0;
        V(2)->state->offset[1] = 2;
        V(2)->state->offset[2] = 4;

        root_ = std::make_shared<FactorizedTreeElement>("a", V(0));
        root_->add_leaf("a", "b", V(0), V(1));
        root_->add_leaf("a", "c", V(0), V(2));

        ordering_ = {"a", "b", "c"};
        schema_.root = root_;
        schema_.column_ordering = &ordering_;
        sync_llm_config_from_ordering();
    }

    // ---- Deep+wide topology ----
    // r: [1,2]
    // r -> a(2 each), b(1 each)
    // a -> c(1 each), d(2 each)
    // b -> e(3 each)
    void build_deep_wide_tree() {
        vecs_[0] = std::make_unique<Vector<uint64_t>>();
        V(0)->values[0] = 1;
        V(0)->values[1] = 2;
        SET_ALL_BITS(V(0)->state->selector);
        SET_START_POS(*V(0)->state, 0);
        SET_END_POS(*V(0)->state, 1);

        vecs_[1] = std::make_unique<Vector<uint64_t>>();
        for (int i = 0; i < 4; ++i) V(1)->values[i] = 10 + i;
        SET_ALL_BITS(V(1)->state->selector);
        SET_START_POS(*V(1)->state, 0);
        SET_END_POS(*V(1)->state, 3);
        V(1)->state->offset[0] = 0;
        V(1)->state->offset[1] = 2;
        V(1)->state->offset[2] = 4;

        vecs_[2] = std::make_unique<Vector<uint64_t>>();
        V(2)->values[0] = 30;
        V(2)->values[1] = 31;
        SET_ALL_BITS(V(2)->state->selector);
        SET_START_POS(*V(2)->state, 0);
        SET_END_POS(*V(2)->state, 1);
        V(2)->state->offset[0] = 0;
        V(2)->state->offset[1] = 1;
        V(2)->state->offset[2] = 2;

        vecs_[3] = std::make_unique<Vector<uint64_t>>();
        for (int i = 0; i < 4; ++i) V(3)->values[i] = 100 + i;
        SET_ALL_BITS(V(3)->state->selector);
        SET_START_POS(*V(3)->state, 0);
        SET_END_POS(*V(3)->state, 3);
        V(3)->state->offset[0] = 0;
        V(3)->state->offset[1] = 1;
        V(3)->state->offset[2] = 2;
        V(3)->state->offset[3] = 3;
        V(3)->state->offset[4] = 4;

        vecs_[4] = std::make_unique<Vector<uint64_t>>();
        for (int i = 0; i < 8; ++i) V(4)->values[i] = 200 + i;
        SET_ALL_BITS(V(4)->state->selector);
        SET_START_POS(*V(4)->state, 0);
        SET_END_POS(*V(4)->state, 7);
        V(4)->state->offset[0] = 0;
        V(4)->state->offset[1] = 2;
        V(4)->state->offset[2] = 4;
        V(4)->state->offset[3] = 6;
        V(4)->state->offset[4] = 8;

        vecs_[5] = std::make_unique<Vector<uint64_t>>();
        for (int i = 0; i < 6; ++i) V(5)->values[i] = 300 + i;
        SET_ALL_BITS(V(5)->state->selector);
        SET_START_POS(*V(5)->state, 0);
        SET_END_POS(*V(5)->state, 5);
        V(5)->state->offset[0] = 0;
        V(5)->state->offset[1] = 3;
        V(5)->state->offset[2] = 6;

        root_ = std::make_shared<FactorizedTreeElement>("r", V(0));
        root_->add_leaf("r", "a", V(0), V(1));
        root_->add_leaf("r", "b", V(0), V(2));
        root_->add_leaf("a", "c", V(1), V(3));
        root_->add_leaf("a", "d", V(1), V(4));
        root_->add_leaf("b", "e", V(2), V(5));

        ordering_ = {"r", "a", "b", "c", "d", "e"};
        schema_.root = root_;
        schema_.column_ordering = &ordering_;
        sync_llm_config_from_ordering();
    }
};

TEST(LLMResultBatchUnitTest, BasicStructure) {
    std::vector<uint64_t> results = {1000, 1001, 1002};
    LLMResultBatch batch{results.data(), results.size()};
    EXPECT_EQ(batch.count, 3u);
    EXPECT_EQ(batch.results[0], 1000u);
    EXPECT_EQ(batch.results[1], 1001u);
    EXPECT_EQ(batch.results[2], 1002u);
}

TEST_F(StreamingMapTest, MultiBranchOutputMapping) {
    build_balanced_tree();

    Map map_op;
    map_op.init(&schema_);
    map_op.execute();

    EXPECT_EQ(map_op.get_num_output_tuples(), 8u);
}

TEST_F(StreamingMapTest, BufferOverflowTesting) {
    vecs_[0] = std::make_unique<Vector<uint64_t>>();
    V(0)->values[0] = 7;
    SET_ALL_BITS(V(0)->state->selector);
    SET_START_POS(*V(0)->state, 0);
    SET_END_POS(*V(0)->state, 0);

    vecs_[1] = std::make_unique<Vector<uint64_t>>();
    for (int i = 0; i < 10; ++i)
        V(1)->values[i] = 100 + i;
    SET_ALL_BITS(V(1)->state->selector);
    SET_START_POS(*V(1)->state, 0);
    SET_END_POS(*V(1)->state, 9);
    V(1)->state->offset[0] = 0;
    V(1)->state->offset[1] = 10;

    vecs_[2] = std::make_unique<Vector<uint64_t>>();
    for (int i = 0; i < 10; ++i)
        V(2)->values[i] = 200 + i;
    SET_ALL_BITS(V(2)->state->selector);
    SET_START_POS(*V(2)->state, 0);
    SET_END_POS(*V(2)->state, 9);
    V(2)->state->offset[0] = 0;
    V(2)->state->offset[1] = 10;

    root_ = std::make_shared<FactorizedTreeElement>("a", V(0));
    root_->add_leaf("a", "b", V(0), V(1));
    root_->add_leaf("a", "c", V(0), V(2));

    ordering_ = {"a", "b", "c"};
    schema_.root = root_;
    schema_.column_ordering = &ordering_;
    sync_llm_config_from_ordering();

    Map map_op;
    map_op.init(&schema_);
    map_op.execute();

    EXPECT_EQ(map_op.get_num_output_tuples(), 100);
}

TEST_F(StreamingMapTest, EmptyBoundaryTest) {
    vecs_[0] = std::make_unique<Vector<uint64_t>>();
    SET_START_POS(*V(0)->state, 1);
    SET_END_POS(*V(0)->state, 0);

    root_ = std::make_shared<FactorizedTreeElement>("a", V(0));
    ordering_ = {"a"};
    schema_.root = root_;
    schema_.column_ordering = &ordering_;
    sync_llm_config_from_ordering();

    Map map_op;
    map_op.init(&schema_);
    map_op.execute();

    EXPECT_EQ(map_op.get_num_output_tuples(), 0u);
}

TEST_F(StreamingMapTest, SingleTupleTest) {
    vecs_[0] = std::make_unique<Vector<uint64_t>>();
    V(0)->values[0] = 1;
    SET_ALL_BITS(V(0)->state->selector);
    SET_START_POS(*V(0)->state, 0);
    SET_END_POS(*V(0)->state, 0);

    vecs_[1] = std::make_unique<Vector<uint64_t>>();
    V(1)->values[0] = 10;
    SET_ALL_BITS(V(1)->state->selector);
    SET_START_POS(*V(1)->state, 0);
    SET_END_POS(*V(1)->state, 0);
    V(1)->state->offset[0] = 0;
    V(1)->state->offset[1] = 1;

    vecs_[2] = std::make_unique<Vector<uint64_t>>();
    V(2)->values[0] = 100;
    SET_ALL_BITS(V(2)->state->selector);
    SET_START_POS(*V(2)->state, 0);
    SET_END_POS(*V(2)->state, 0);
    V(2)->state->offset[0] = 0;
    V(2)->state->offset[1] = 1;

    root_ = std::make_shared<FactorizedTreeElement>("a", V(0));
    root_->add_leaf("a", "b", V(0), V(1));
    root_->add_leaf("b", "c", V(1), V(2));

    ordering_ = {"a", "b", "c"};
    schema_.root = root_;
    schema_.column_ordering = &ordering_;
    sync_llm_config_from_ordering();

    Map map_op;
    map_op.init(&schema_);
    map_op.execute();

    EXPECT_EQ(map_op.get_num_output_tuples(), 1u);
}

TEST_F(StreamingMapTest, SkewedTreeTest) {
    vecs_[0] = std::make_unique<Vector<uint64_t>>();
    V(0)->values[0] = 7;
    SET_ALL_BITS(V(0)->state->selector);
    SET_START_POS(*V(0)->state, 0);
    SET_END_POS(*V(0)->state, 0);

    vecs_[1] = std::make_unique<Vector<uint64_t>>();
    V(1)->values[0] = 100;
    V(1)->values[1] = 101;
    SET_ALL_BITS(V(1)->state->selector);
    SET_START_POS(*V(1)->state, 0);
    SET_END_POS(*V(1)->state, 1);
    V(1)->state->offset[0] = 0;
    V(1)->state->offset[1] = 2;

    vecs_[2] = std::make_unique<Vector<uint64_t>>();
    for (int i = 0; i < 8; ++i)
        V(2)->values[i] = 200 + i;
    SET_ALL_BITS(V(2)->state->selector);
    SET_START_POS(*V(2)->state, 0);
    SET_END_POS(*V(2)->state, 7);
    V(2)->state->offset[0] = 0;
    V(2)->state->offset[1] = 8;

    root_ = std::make_shared<FactorizedTreeElement>("a", V(0));
    root_->add_leaf("a", "b", V(0), V(1));
    root_->add_leaf("a", "c", V(0), V(2));

    ordering_ = {"a", "b", "c"};
    schema_.root = root_;
    schema_.column_ordering = &ordering_;
    sync_llm_config_from_ordering();

    Map map_op;
    map_op.init(&schema_);
    map_op.execute();

    EXPECT_EQ(map_op.get_num_output_tuples(), 16u);
}

TEST_F(StreamingMapTest, SingleLeafTreeTest) {
    vecs_[0] = std::make_unique<Vector<uint64_t>>();
    for (int i = 0; i < 5; ++i)
        V(0)->values[i] = 10 + i * 10;
    SET_ALL_BITS(V(0)->state->selector);
    SET_START_POS(*V(0)->state, 0);
    SET_END_POS(*V(0)->state, 4);

    root_ = std::make_shared<FactorizedTreeElement>("x", V(0));
    ordering_ = {"x"};
    schema_.root = root_;
    schema_.column_ordering = &ordering_;
    sync_llm_config_from_ordering();

    Map map_op;
    map_op.init(&schema_);
    map_op.execute();

    EXPECT_EQ(map_op.get_num_output_tuples(), 5u);
}

TEST_F(StreamingMapTest, StarTreeTest) {
    vecs_[0] = std::make_unique<Vector<uint64_t>>();
    for (int i = 0; i < 5; ++i)
        V(0)->values[i] = 1 + i;
    SET_ALL_BITS(V(0)->state->selector);
    SET_START_POS(*V(0)->state, 0);
    SET_END_POS(*V(0)->state, 4);

    vecs_[1] = std::make_unique<Vector<uint64_t>>();
    for (int i = 0; i < 15; ++i)
        V(1)->values[i] = 10 + i;
    SET_ALL_BITS(V(1)->state->selector);
    SET_START_POS(*V(1)->state, 0);
    SET_END_POS(*V(1)->state, 14);
    for (int i = 0; i <= 5; ++i)
        V(1)->state->offset[i] = i * 3;

    vecs_[2] = std::make_unique<Vector<uint64_t>>();
    for (int i = 0; i < 10; ++i)
        V(2)->values[i] = 30 + i;
    SET_ALL_BITS(V(2)->state->selector);
    SET_START_POS(*V(2)->state, 0);
    SET_END_POS(*V(2)->state, 9);
    for (int i = 0; i <= 5; ++i)
        V(2)->state->offset[i] = i * 2;

    vecs_[3] = std::make_unique<Vector<uint64_t>>();
    for (int i = 0; i < 20; ++i)
        V(3)->values[i] = 50 + i;
    SET_ALL_BITS(V(3)->state->selector);
    SET_START_POS(*V(3)->state, 0);
    SET_END_POS(*V(3)->state, 19);
    for (int i = 0; i <= 5; ++i)
        V(3)->state->offset[i] = i * 4;

    vecs_[4] = std::make_unique<Vector<uint64_t>>();
    for (int i = 0; i < 10; ++i)
        V(4)->values[i] = 80 + i;
    SET_ALL_BITS(V(4)->state->selector);
    SET_START_POS(*V(4)->state, 0);
    SET_END_POS(*V(4)->state, 9);
    for (int i = 0; i <= 5; ++i)
        V(4)->state->offset[i] = i * 2;

    root_ = std::make_shared<FactorizedTreeElement>("r", V(0));
    root_->add_leaf("r", "a", V(0), V(1));
    root_->add_leaf("r", "b", V(0), V(2));
    root_->add_leaf("r", "c", V(0), V(3));
    root_->add_leaf("r", "d", V(0), V(4));

    ordering_ = {"r", "a", "b", "c", "d"};
    schema_.root = root_;
    schema_.column_ordering = &ordering_;
    sync_llm_config_from_ordering();

    Map map_op;
    map_op.init(&schema_);
    map_op.execute();

    EXPECT_EQ(map_op.get_num_output_tuples(), 240u);
}

TEST_F(StreamingMapTest, MapConfiguredProjectionOnDeepWideTopology) {
    build_deep_wide_tree();

    // Use context_column to control per-attribute capacities.
    std::string cfg =
        R"({"provider":"demo","model":"demo","model_name":"demo_model","tuple_format":"FTREE",)"
        R"("context_column":{"r":8,"a":16,"b":8,"c":8,"d":32,"e":16},"prompt":"test"})";
    schema_.llm_config_str = &cfg;
    llm_output_name_ = "_llm";
    schema_.llm_output_attr = &llm_output_name_;

    Map map_op;
    map_op.init(&schema_);
    map_op.execute();

    EXPECT_GT(map_op.get_num_output_tuples(), 0u);
}

TEST_F(StreamingMapTest, ConfiguredMapWithDifferentOrderingStillExecutes) {
    build_balanced_tree();
    ordering_ = {"a", "c", "b"};
    schema_.column_ordering = &ordering_;
    sync_llm_config_from_ordering();

    Map map_op;
    map_op.init(&schema_);
    map_op.execute();

    EXPECT_EQ(map_op.get_num_output_tuples(), 8u);
}

class CountingSink final : public Operator {
public:
    uint32_t init_calls = 0;
    uint32_t execute_calls = 0;

    void init(Schema* /*schema*/) override { init_calls++; }
    void execute() override { execute_calls++; }
    uint64_t get_num_output_tuples() override { return 0u; }
};

TEST_F(StreamingMapTest, FTreeVsFlatJSONParity_OnTupleCount) {
    build_balanced_tree();

    // FTREE mode
    std::string ftree_cfg =
        R"({"provider":"demo","model":"demo","model_name":"demo_model","tuple_format":"FTREE",)"
        R"("context_column":{"a":64,"b":64,"c":64},"prompt":"test"})";
    schema_.llm_config_str = &ftree_cfg;
    llm_output_name_ = "_llm";
    schema_.llm_output_attr = &llm_output_name_;

    Map tree_mode;
    auto tree_sink = std::make_unique<CountingSink>();
    auto tree_sink_raw = tree_sink.get();
    tree_mode.set_next_operator(std::move(tree_sink));
    tree_mode.init(&schema_);
    tree_mode.execute();

    // JSON (flat) mode — use a large batch_size so the whole tree fits in one batch.
    std::string json_cfg =
        R"({"provider":"demo","model":"demo","model_name":"demo_model","tuple_format":"JSON","batch_size":128,)"
        R"("context_column":["a","b","c"],"prompt":"test"})";
    schema_.llm_config_str = &json_cfg;

    Map flat_mode;
    auto flat_sink = std::make_unique<CountingSink>();
    auto flat_sink_raw = flat_sink.get();
    flat_mode.set_next_operator(std::move(flat_sink));
    flat_mode.init(&schema_);
    flat_mode.execute();

    EXPECT_EQ(tree_mode.get_num_output_tuples(), flat_mode.get_num_output_tuples());
    EXPECT_EQ(tree_mode.get_num_output_tuples(), 8u);
    EXPECT_EQ(tree_sink_raw->init_calls, 1u);
    EXPECT_EQ(flat_sink_raw->init_calls, 1u);
    EXPECT_GE(tree_sink_raw->execute_calls, 1u);
    EXPECT_GE(flat_sink_raw->execute_calls, 1u);
}

TEST_F(StreamingMapTest, RepeatedExecuteIsStablePerRun) {
    build_balanced_tree();

    Map map_op;
    map_op.init(&schema_);

    map_op.execute();
    uint64_t first_run = map_op.get_num_output_tuples();

    map_op.execute();
    uint64_t second_run = map_op.get_num_output_tuples();

    EXPECT_EQ(first_run, 8u);
    // Map accumulates across execute() calls
    EXPECT_EQ(second_run, first_run * 2u);
}

TEST_F(StreamingMapTest, SteinerAugmentWithContextColumnSubset) {
    build_deep_wide_tree();

    // Only request d and e — Steiner ancestors (r, a, b) must be added automatically.
    std::string cfg =
        R"({"provider":"demo","model":"demo","model_name":"demo_model","tuple_format":"FTREE",)"
        R"("context_column":{"d":64,"e":64},"prompt":"test"})";
    schema_.llm_config_str = &cfg;
    llm_output_name_ = "_llm";
    schema_.llm_output_attr = &llm_output_name_;

    Map map_op;
    EXPECT_NO_THROW(map_op.init(&schema_));
    map_op.execute();
    EXPECT_GT(map_op.get_num_output_tuples(), 0u);
}

TEST_F(StreamingMapTest, LargeCrossBranchInputStress) {
    vecs_[0] = std::make_unique<Vector<uint64_t>>();
    V(0)->values[0] = 1;
    SET_ALL_BITS(V(0)->state->selector);
    SET_START_POS(*V(0)->state, 0);
    SET_END_POS(*V(0)->state, 0);

    vecs_[1] = std::make_unique<Vector<uint64_t>>();
    vecs_[2] = std::make_unique<Vector<uint64_t>>();
    for (int i = 0; i < 32; ++i) {
        V(1)->values[i] = static_cast<uint64_t>(100 + i);
        V(2)->values[i] = static_cast<uint64_t>(1000 + i);
    }
    SET_ALL_BITS(V(1)->state->selector);
    SET_ALL_BITS(V(2)->state->selector);
    SET_START_POS(*V(1)->state, 0);
    SET_END_POS(*V(1)->state, 31);
    SET_START_POS(*V(2)->state, 0);
    SET_END_POS(*V(2)->state, 31);
    V(1)->state->offset[0] = 0;
    V(1)->state->offset[1] = 32;
    V(2)->state->offset[0] = 0;
    V(2)->state->offset[1] = 32;

    root_ = std::make_shared<FactorizedTreeElement>("r", V(0));
    root_->add_leaf("r", "b", V(0), V(1));
    root_->add_leaf("r", "c", V(0), V(2));
    ordering_ = {"r", "b", "c"};
    schema_.root = root_;
    schema_.column_ordering = &ordering_;
    sync_llm_config_from_ordering();

    Map map_op;
    map_op.init(&schema_);
    map_op.execute();

    EXPECT_EQ(map_op.get_num_output_tuples(), 1024u);
}
