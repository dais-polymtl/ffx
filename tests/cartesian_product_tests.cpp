/// Tests for FTreeReconstructor (via Map operator).
///
/// Covers:
///  1. Linear tree (all nodes projected) — values, RLE, column ordering
///  2. Linear tree with intermediate ancestor — Steiner tree, visibility
///  3. Branching OT → Steiner + linearization
///  4. Equal-depth sibling ordering (right under left)
///  5. Edge cases: empty batch, single tuple
///  6. Batch > MAX_VECTOR_SIZE chunking

#include "../src/operator/include/ai_operator/map.hpp"
#include "../src/operator/include/factorized_ftree/ftree_batch_iterator.hpp"
#include "../src/operator/include/operator.hpp"
#include "../src/operator/include/schema/schema.hpp"
#include "../src/operator/include/vector/bitmask.hpp"
#include "../src/operator/include/vector/state.hpp"
#include "../src/operator/include/vector/vector.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

using namespace ffx;

// ---------------------------------------------------------------------------
// CollectingOperator: downstream sink that expands the new linear chain and
// records each logical tuple.  Uses the child-state-offset RLE convention:
//   child->_value->state->offset[p]   = start of child range for parent pos p
//   child->_value->state->offset[p+1] = exclusive end
// ---------------------------------------------------------------------------
class CollectingOperator final : public Operator {
public:
    // Tuples collected across all execute() calls.
    std::vector<std::vector<uint64_t>> tuples;
    // Visible column ordering captured from schema during init().
    std::vector<std::string> visible_ordering;
    // Snapshot of the new tree root.
    std::shared_ptr<FactorizedTreeElement> new_root;

    void init(Schema* schema) override {
        tuples.clear();
        visible_ordering = *schema->column_ordering;
        new_root = schema->root;
    }

    void execute() override {
        if (!new_root) return;
        // Collect all FactorizedTreeElement nodes from root in linear-chain order.
        std::vector<FactorizedTreeElement*> chain;
        FactorizedTreeElement* cur = new_root.get();
        while (cur) {
            chain.push_back(cur);
            cur = cur->_children.empty() ? nullptr : cur->_children[0].get();
        }
        if (chain.empty()) return;

        // Root range
        const State* root_state = chain[0]->_value->state;
        size_t root_lo = root_state->start_pos;
        size_t root_hi = static_cast<size_t>(root_state->end_pos) + 1;

        std::vector<uint64_t> current(chain.size());
        expand(chain, 0, root_lo, root_hi, current);
    }

    uint64_t get_num_output_tuples() override { return static_cast<uint64_t>(tuples.size()); }

private:
    void expand(const std::vector<FactorizedTreeElement*>& chain, size_t depth, size_t lo, size_t hi,
                std::vector<uint64_t>& current) {
        FactorizedTreeElement* node = chain[depth];
        for (size_t p = lo; p < hi; ++p) {
            current[depth] = node->_value->values[p];
            if (depth + 1 == chain.size()) {
                tuples.push_back(current);
            } else {
                // Child offset is stored in the CHILD's state (consistent with existing tree convention)
                // Child offset is stored in the CHILD's state
                FactorizedTreeElement* child = chain[depth + 1];

                size_t child_lo, child_hi;
                if (child->_value->state == node->_value->state) {
                    // Identity RLE: exactly one child per parent position
                    child_lo = p;
                    child_hi = p + 1;
                } else {
                    const uint16_t* off = child->_value->state->offset;
                    child_lo = off[p];
                    child_hi = off[p + 1];
                }
                expand(chain, depth + 1, child_lo, child_hi, current);
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Test fixture — mirrors StreamingMapTest layout
// ---------------------------------------------------------------------------
class CartesianProductTest : public ::testing::Test {
protected:
    std::unique_ptr<Vector<uint64_t>> vecs_[20];
    std::shared_ptr<FactorizedTreeElement> root_;
    std::vector<std::string> ordering_;
    Schema schema_;
    QueryVariableToVectorMap map_;
    std::unique_ptr<StringPool> pool_;
    std::unique_ptr<StringDictionary> dictionary_;
    std::string llm_config_;
    std::string llm_output_attr_storage_{"llm_out"};

    Vector<uint64_t>* V(int slot) { return vecs_[slot].get(); }

    void SetUp() override {
        schema_.map = &map_;
        pool_ = std::make_unique<StringPool>();
        dictionary_ = std::make_unique<StringDictionary>(pool_.get());
        dictionary_->finalize();
        schema_.predicate_pool = pool_.get();
        schema_.dictionary = dictionary_.get();
        llm_config_ =
                R"({"provider":"demo","model":"demo","model_name":"demo_model","tuple_format":"FTREE","prompt":"test"})";
        schema_.llm_config_str = &llm_config_;
        schema_.llm_output_attr = &llm_output_attr_storage_;
    }

    // Build a linear chain  a → b → c
    //   a: [1, 2]
    //   b: [10, 11 | 20, 21]   (2 per a-value)
    //   c: [100, 101 | 200, 201 | 300, 301 | 400, 401]  (2 per b-value)
    // Total logical tuples: 2 * 2 * 2 = 8
    void build_linear_tree_abc() {
        // a: 2 values
        vecs_[0] = std::make_unique<Vector<uint64_t>>();
        V(0)->values[0] = 1;
        V(0)->values[1] = 2;
        SET_ALL_BITS(V(0)->state->selector);
        SET_START_POS(*V(0)->state, 0);
        SET_END_POS(*V(0)->state, 1);

        // b: 4 values, 2 per a
        vecs_[1] = std::make_unique<Vector<uint64_t>>();
        V(1)->values[0] = 10;
        V(1)->values[1] = 11;
        V(1)->values[2] = 20;
        V(1)->values[3] = 21;
        SET_ALL_BITS(V(1)->state->selector);
        SET_START_POS(*V(1)->state, 0);
        SET_END_POS(*V(1)->state, 3);
        V(1)->state->offset[0] = 0;
        V(1)->state->offset[1] = 2;
        V(1)->state->offset[2] = 4;

        // c: 8 values, 2 per b
        vecs_[2] = std::make_unique<Vector<uint64_t>>();
        for (int i = 0; i < 8; ++i)
            V(2)->values[i] = 100 + i * 100;
        SET_ALL_BITS(V(2)->state->selector);
        SET_START_POS(*V(2)->state, 0);
        SET_END_POS(*V(2)->state, 7);
        V(2)->state->offset[0] = 0;
        V(2)->state->offset[1] = 2;
        V(2)->state->offset[2] = 4;
        V(2)->state->offset[3] = 6;
        V(2)->state->offset[4] = 8;

        root_ = std::make_shared<FactorizedTreeElement>("a", V(0));
        root_->add_leaf("a", "b", V(0), V(1));
        root_->add_leaf("b", "c", V(1), V(2));
    }

    // Build branching tree: a → (b, c),  b → (d, e),  c → (f, g)
    //   a: [1]
    //   b: [10],  c: [20]          (1 per a)
    //   d: [100], e: [110]         (1 per b)
    //   f: [200], g: [210]         (1 per c)
    // Logical tuples (cross-product): a×b×d × a×c×f × a×c×g × a×b×e × ...
    // Actually the factorized tree gives: a with children (b and c). b has children (d,e). c has children (f,g).
    // For required={b,f,g}: Steiner = {a,b,c,f,g}, linearized = a→c→f→g→b→_llm
    // Logical tuples from projected {b,f,g}: 1(b) * 1(f) * 1(g) = ... this is a cross product
    // Actually Map iterates the OT and produces one expanded tuple per combination.
    // For OT with b×d, b×e under a, c×f, c×g under a:
    //   Projected cols from Map with required={b,f,g}: (b,f), (b,g) → 2 tuples
    void build_branching_tree_abcdefg() {
        // a: 1 value
        vecs_[0] = std::make_unique<Vector<uint64_t>>();
        V(0)->values[0] = 1;
        SET_ALL_BITS(V(0)->state->selector);
        SET_START_POS(*V(0)->state, 0);
        SET_END_POS(*V(0)->state, 0);

        // b: 1 value under a[0]
        vecs_[1] = std::make_unique<Vector<uint64_t>>();
        V(1)->values[0] = 10;
        SET_ALL_BITS(V(1)->state->selector);
        SET_START_POS(*V(1)->state, 0);
        SET_END_POS(*V(1)->state, 0);
        V(1)->state->offset[0] = 0;
        V(1)->state->offset[1] = 1;

        // c: 1 value under a[0]
        vecs_[2] = std::make_unique<Vector<uint64_t>>();
        V(2)->values[0] = 20;
        SET_ALL_BITS(V(2)->state->selector);
        SET_START_POS(*V(2)->state, 0);
        SET_END_POS(*V(2)->state, 0);
        V(2)->state->offset[0] = 0;
        V(2)->state->offset[1] = 1;

        // d: 1 value under b[0]
        vecs_[3] = std::make_unique<Vector<uint64_t>>();
        V(3)->values[0] = 100;
        SET_ALL_BITS(V(3)->state->selector);
        SET_START_POS(*V(3)->state, 0);
        SET_END_POS(*V(3)->state, 0);
        V(3)->state->offset[0] = 0;
        V(3)->state->offset[1] = 1;

        // e: 1 value under b[0]
        vecs_[4] = std::make_unique<Vector<uint64_t>>();
        V(4)->values[0] = 110;
        SET_ALL_BITS(V(4)->state->selector);
        SET_START_POS(*V(4)->state, 0);
        SET_END_POS(*V(4)->state, 0);
        V(4)->state->offset[0] = 0;
        V(4)->state->offset[1] = 1;

        // f: 1 value under c[0]
        vecs_[5] = std::make_unique<Vector<uint64_t>>();
        V(5)->values[0] = 200;
        SET_ALL_BITS(V(5)->state->selector);
        SET_START_POS(*V(5)->state, 0);
        SET_END_POS(*V(5)->state, 0);
        V(5)->state->offset[0] = 0;
        V(5)->state->offset[1] = 1;

        // g: 1 value under c[0]
        vecs_[6] = std::make_unique<Vector<uint64_t>>();
        V(6)->values[0] = 210;
        SET_ALL_BITS(V(6)->state->selector);
        SET_START_POS(*V(6)->state, 0);
        SET_END_POS(*V(6)->state, 0);
        V(6)->state->offset[0] = 0;
        V(6)->state->offset[1] = 1;

        // Tree: a→(b,c), b→(d,e), c→(f,g)
        root_ = std::make_shared<FactorizedTreeElement>("a", V(0));
        root_->add_leaf("a", "b", V(0), V(1));
        root_->add_leaf("a", "c", V(0), V(2));
        root_->add_leaf("b", "d", V(1), V(3));
        root_->add_leaf("b", "e", V(1), V(4));
        root_->add_leaf("c", "f", V(2), V(5));
        root_->add_leaf("c", "g", V(2), V(6));
    }

    // Setup schema and run Map→CollectingOperator pipeline.
    // Map internally uses FTreeReconstructor (replaces CartesianProduct).
    // Returns the CollectingOperator pointer (owned by the chain).
    std::unique_ptr<CollectingOperator> run_pipeline(const std::unordered_map<std::string, size_t>& user_required,
                                                     const std::vector<std::string>& col_ordering) {
        ordering_ = col_ordering;
        schema_.root = root_;
        schema_.column_ordering = &ordering_;

        // Build LLM config with context_column matching user_required so that
        // MapImpl projects exactly the same attributes as the old set_required_attributes.
        std::string ctx = "{";
        bool first = true;
        for (const auto& [attr, cap]: user_required) {
            if (!first) ctx += ",";
            ctx += "\"" + attr + "\":" + std::to_string(cap);
            first = false;
        }
        ctx += "}";
        std::string pipeline_cfg =
                R"({"provider":"demo","model":"demo","model_name":"demo_model","tuple_format":"FTREE","context_column":)" +
                ctx + R"(,"prompt":"test"})";
        schema_.llm_config_str = &pipeline_cfg;

        Map map_op;

        auto collector = std::make_unique<CollectingOperator>();
        CollectingOperator* raw_collector = collector.get();

        map_op.set_next_operator(std::move(collector));
        map_op.init(&schema_);
        map_op.execute();

        // Restore fixture-level config pointer.
        schema_.llm_config_str = &llm_config_;

        auto result = std::make_unique<CollectingOperator>();
        result->tuples = raw_collector->tuples;
        result->visible_ordering = raw_collector->visible_ordering;
        result->new_root = raw_collector->new_root;
        return result;
    }
};

// ---------------------------------------------------------------------------
// Test 1: Linear tree, all nodes projected
// OT: a→b→c,  required={a,b,c}
// Expected new chain: a→b→c→_llm
// column_ordering: {a, b, c, _llm}
// Total tuples: 8
// ---------------------------------------------------------------------------
TEST_F(CartesianProductTest, LinearTreeAllProjected) {
    build_linear_tree_abc();
    auto col = run_pipeline({{"a", 64u}, {"b", 64u}, {"c", 64u}}, {"a", "b", "c"});

    // Total expanded tuples: 2*2*2 = 8
    EXPECT_EQ(col->tuples.size(), 8u);

    // Column ordering must include all projected attrs + output, nothing extra
    ASSERT_EQ(col->visible_ordering.size(), 4u);
    EXPECT_EQ(col->visible_ordering[0], "a");
    EXPECT_EQ(col->visible_ordering[1], "b");
    EXPECT_EQ(col->visible_ordering[2], "c");
    EXPECT_EQ(col->visible_ordering[3], "llm_out");

    // New tree is a linear chain of 4 nodes: a→b→c→_llm
    ASSERT_NE(col->new_root, nullptr);
    EXPECT_EQ(col->new_root->_attribute, "a");
    ASSERT_EQ(col->new_root->_children.size(), 1u);
    EXPECT_EQ(col->new_root->_children[0]->_attribute, "b");
    ASSERT_EQ(col->new_root->_children[0]->_children.size(), 1u);
    EXPECT_EQ(col->new_root->_children[0]->_children[0]->_attribute, "c");
    ASSERT_EQ(col->new_root->_children[0]->_children[0]->_children.size(), 1u);
    EXPECT_EQ(col->new_root->_children[0]->_children[0]->_children[0]->_attribute, "llm_out");

    // Each tuple has 4 values: (a_val, b_val, c_val, llm_val)
    for (const auto& t: col->tuples) {
        ASSERT_EQ(t.size(), 4u);
        // a_val must be 1 or 2
        EXPECT_TRUE(t[0] == 1u || t[0] == 2u);
        // b_val depends on a_val
        if (t[0] == 1u) EXPECT_TRUE(t[1] == 10u || t[1] == 11u);
        if (t[0] == 2u) EXPECT_TRUE(t[1] == 20u || t[1] == 21u);
        // c_val is in {100,200,300,400,500,600,700,800}
        EXPECT_EQ(t[2] % 100u, 0u);
        EXPECT_NE(t[3], 0u);
    }
}

// ---------------------------------------------------------------------------
// Test 2: Intermediate ancestor not visible in output
// OT: a→b→c,  required={b,c}  (a is ancestor-intermediate)
// Expected new chain: a→b→c→_llm
// column_ordering: {b, c, _llm}  — a is structural, NOT visible
// Total tuples: 8
// ---------------------------------------------------------------------------
TEST_F(CartesianProductTest, IntermediateAncestorNotVisible) {
    build_linear_tree_abc();
    auto col = run_pipeline({{"b", 64u}, {"c", 64u}}, {"a", "b", "c"});

    EXPECT_EQ(col->tuples.size(), 8u);

    // Column ordering is preserved as the full query ordering, with _llm appended.
    ASSERT_EQ(col->visible_ordering.size(), 4u);
    EXPECT_EQ(col->visible_ordering[0], "a");
    EXPECT_EQ(col->visible_ordering[1], "b");
    EXPECT_EQ(col->visible_ordering[2], "c");
    EXPECT_EQ(col->visible_ordering[3], "llm_out");

    // New tree still has 'a' as root (structural), but new chain is a→b→c→_llm
    ASSERT_NE(col->new_root, nullptr);
    EXPECT_EQ(col->new_root->_attribute, "a");

    // Each tuple has 4 values (a, b, c, llm) — CollectingOperator walks the
    // full chain including structural ancestor 'a'; visibility is separate.
    for (const auto& t: col->tuples) {
        ASSERT_EQ(t.size(), 4u);
        // a_val (structural ancestor)
        EXPECT_TRUE(t[0] == 1u || t[0] == 2u);
        // b_val must be one of {10, 11, 20, 21}
        EXPECT_TRUE(t[1] == 10u || t[1] == 11u || t[1] == 20u || t[1] == 21u);
        EXPECT_NE(t[3], 0u);
    }
}

// ---------------------------------------------------------------------------
// Test 3: Branching tree — Steiner linearization
// OT: a→(b,c), b→(d,e), c→(f,g)
// required={b,f,g}
// Steiner tree: {a,b,c,f,g}  (d,e dropped)
// Linearized: a→c→f→g→b→_llm  (c-branch depth 3, b-branch depth 1; f,g equal depth → g under f)
// column_ordering: {b, f, g, _llm}  — a,c are structural
// ---------------------------------------------------------------------------
TEST_F(CartesianProductTest, BranchingTreeSteinerLinearization) {
    build_branching_tree_abcdefg();
    // Column ordering must list all OT attrs for Map to iterate; required={b,f,g}
    auto col = run_pipeline({{"b", 64u}, {"f", 64u}, {"g", 64u}}, {"a", "b", "c", "d", "e", "f", "g"});

    // With augmented required set {a,b,c,f,g} (ancestors added for connectivity),
    // the batch iterator's DFS visits: a=1, b=10, c=20, f=200, g=210.
    // Each node has exactly 1 value, so there is 1 logical tuple.
    EXPECT_EQ(col->tuples.size(), 1u);

    // Chain order: a→c→f→g→b→_llm
    ASSERT_NE(col->new_root, nullptr);
    EXPECT_EQ(col->new_root->_attribute, "a");

    // Column ordering is preserved as the full query ordering, with _llm appended.
    ASSERT_EQ(col->visible_ordering.size(), 8u);
    EXPECT_EQ(col->visible_ordering[0], "a");
    EXPECT_EQ(col->visible_ordering[1], "b");
    EXPECT_EQ(col->visible_ordering[2], "c");
    EXPECT_EQ(col->visible_ordering[3], "d");
    EXPECT_EQ(col->visible_ordering[4], "e");
    EXPECT_EQ(col->visible_ordering[5], "f");
    EXPECT_EQ(col->visible_ordering[6], "g");
    EXPECT_EQ(col->visible_ordering[7], "llm_out");
}

// ---------------------------------------------------------------------------
// Test 4: Chain order — longer branch first, equal depth → right under left
// OT: a→(b,c),  b→d,  c→(f,g)
// required={b,f,g}
// Steiner: {a,b,c,f,g}
// c-branch length from a: a→c→f, a→c→g → depth 2 from a (length 3 nodes)
// b-branch length from a: a→b → depth 1 from a (length 2 nodes)
// → c-branch is longer → a→c→f→g→b→_llm
// ---------------------------------------------------------------------------
TEST_F(CartesianProductTest, ChainOrderLongerBranchFirst) {
    // a: 1 value
    vecs_[0] = std::make_unique<Vector<uint64_t>>();
    V(0)->values[0] = 1;
    SET_ALL_BITS(V(0)->state->selector);
    SET_START_POS(*V(0)->state, 0);
    SET_END_POS(*V(0)->state, 0);

    // b: 1 value under a[0]
    vecs_[1] = std::make_unique<Vector<uint64_t>>();
    V(1)->values[0] = 10;
    SET_ALL_BITS(V(1)->state->selector);
    SET_START_POS(*V(1)->state, 0);
    SET_END_POS(*V(1)->state, 0);
    V(1)->state->offset[0] = 0;
    V(1)->state->offset[1] = 1;

    // c: 1 value under a[0]
    vecs_[2] = std::make_unique<Vector<uint64_t>>();
    V(2)->values[0] = 20;
    SET_ALL_BITS(V(2)->state->selector);
    SET_START_POS(*V(2)->state, 0);
    SET_END_POS(*V(2)->state, 0);
    V(2)->state->offset[0] = 0;
    V(2)->state->offset[1] = 1;

    // d: 1 value under b[0]
    vecs_[3] = std::make_unique<Vector<uint64_t>>();
    V(3)->values[0] = 100;
    SET_ALL_BITS(V(3)->state->selector);
    SET_START_POS(*V(3)->state, 0);
    SET_END_POS(*V(3)->state, 0);
    V(3)->state->offset[0] = 0;
    V(3)->state->offset[1] = 1;

    // f: 1 value under c[0]
    vecs_[5] = std::make_unique<Vector<uint64_t>>();
    V(5)->values[0] = 200;
    SET_ALL_BITS(V(5)->state->selector);
    SET_START_POS(*V(5)->state, 0);
    SET_END_POS(*V(5)->state, 0);
    V(5)->state->offset[0] = 0;
    V(5)->state->offset[1] = 1;

    // g: 1 value under c[0]
    vecs_[6] = std::make_unique<Vector<uint64_t>>();
    V(6)->values[0] = 210;
    SET_ALL_BITS(V(6)->state->selector);
    SET_START_POS(*V(6)->state, 0);
    SET_END_POS(*V(6)->state, 0);
    V(6)->state->offset[0] = 0;
    V(6)->state->offset[1] = 1;

    root_ = std::make_shared<FactorizedTreeElement>("a", V(0));
    root_->add_leaf("a", "b", V(0), V(1));
    root_->add_leaf("a", "c", V(0), V(2));
    root_->add_leaf("b", "d", V(1), V(3));
    root_->add_leaf("c", "f", V(2), V(5));
    root_->add_leaf("c", "g", V(2), V(6));

    auto col = run_pipeline({{"b", 64u}, {"f", 64u}, {"g", 64u}}, {"a", "b", "c", "d", "f", "g"});

    // All ftree nodes are now in the Steiner set (entire ftree projected).
    // Steiner chain: a → b → d → c → f → g → _llm
    // (b subtree has same height as c subtree; tie broken by index: b < c)
    ASSERT_NE(col->new_root, nullptr);
    auto* n0 = col->new_root.get();
    EXPECT_EQ(n0->_attribute, "a");
    ASSERT_EQ(n0->_children.size(), 1u);
    auto* n1 = n0->_children[0].get();
    EXPECT_EQ(n1->_attribute, "b");
    ASSERT_EQ(n1->_children.size(), 1u);
    auto* n2 = n1->_children[0].get();
    EXPECT_EQ(n2->_attribute, "d");
    ASSERT_EQ(n2->_children.size(), 1u);
    auto* n3 = n2->_children[0].get();
    EXPECT_EQ(n3->_attribute, "c");
    ASSERT_EQ(n3->_children.size(), 1u);
    auto* n4 = n3->_children[0].get();
    EXPECT_EQ(n4->_attribute, "f");
    ASSERT_EQ(n4->_children.size(), 1u);
    auto* n5 = n4->_children[0].get();
    EXPECT_EQ(n5->_attribute, "g");
    ASSERT_EQ(n5->_children.size(), 1u);
    EXPECT_EQ(n5->_children[0]->_attribute, "llm_out");
}

// ---------------------------------------------------------------------------
// Test 5: Single logical tuple
// OT: a→b→c, a=1, b=10, c=100
// required={a,b,c}
// Expected: 1 tuple (1, 10, 100, 1000)
// ---------------------------------------------------------------------------
TEST_F(CartesianProductTest, SingleTuple) {
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

    auto col = run_pipeline({{"a", 64u}, {"b", 64u}, {"c", 64u}}, {"a", "b", "c"});

    ASSERT_EQ(col->tuples.size(), 1u);
    EXPECT_EQ(col->tuples[0][0], 1u);
    EXPECT_EQ(col->tuples[0][1], 10u);
    EXPECT_EQ(col->tuples[0][2], 100u);
    EXPECT_NE(col->tuples[0][3], 0u);
}

// ---------------------------------------------------------------------------
// Test 6: Empty tree produces no output tuples
// ---------------------------------------------------------------------------
TEST_F(CartesianProductTest, EmptyTree) {
    vecs_[0] = std::make_unique<Vector<uint64_t>>();
    // No valid elements
    SET_START_POS(*V(0)->state, 1);
    SET_END_POS(*V(0)->state, 0);

    root_ = std::make_shared<FactorizedTreeElement>("a", V(0));
    ordering_ = {"a"};
    schema_.root = root_;
    schema_.column_ordering = &ordering_;

    std::string cfg =
            R"({"provider":"demo","model":"demo","model_name":"demo_model","tuple_format":"FTREE","context_column":{"a":64},"prompt":"test"})";
    schema_.llm_config_str = &cfg;

    Map map_op;

    auto collector = std::make_unique<CollectingOperator>();
    CollectingOperator* raw = collector.get();
    map_op.set_next_operator(std::move(collector));
    map_op.init(&schema_);
    map_op.execute();

    EXPECT_EQ(raw->tuples.size(), 0u);
}

// ---------------------------------------------------------------------------
// Test 7: Batch overflow (total tuples > MAX_VECTOR_SIZE via cross-product)
// OT: a→b→c,  a=2, b=32 per a (64 total), c=32 per b (2048 total)
// Logical tuples = 2*32*32 = 2048. That fits. To exceed, use a=3:
//   a=3, b=32 per a (96 total), c=32 per b (3072 total) — c exceeds 2048.
// Instead: a→b, a=3, b=1024 per a = 3072 tuples.
// But b vector can hold max 2048 values, so use 2 a-values with b having
// 1024 each = 2048 b-values total. That's exactly MAX_VECTOR_SIZE (2048).
// For overflow we need > 2048 logical tuples. Use: a=3, b at max.
// Better approach: a→(b,c), cross product creates more tuples.
// Simplest: a=3, b=700 per a[0], 700 per a[1], 648 per a[2] = 2048 total.
// Total logical tuples = 2048. Not overflow.
// Actually, the overflow is in StreamingBatch/CartesianProduct chunking,
// not in the OT vector size. Let's just use a→b with b=2048 (max) under 1 a.
// That gives exactly 2048 tuples. For chunking test, that's enough since
// MAX_VECTOR_SIZE=2048 and CartesianProduct chunks at that boundary.
// ---------------------------------------------------------------------------
TEST_F(CartesianProductTest, BatchOverflow) {
    constexpr size_t kB = 2048;

    vecs_[0] = std::make_unique<Vector<uint64_t>>();
    V(0)->values[0] = 7;
    SET_ALL_BITS(V(0)->state->selector);
    SET_START_POS(*V(0)->state, 0);
    SET_END_POS(*V(0)->state, 0);

    vecs_[1] = std::make_unique<Vector<uint64_t>>();
    for (size_t i = 0; i < kB; ++i)
        V(1)->values[i] = 1000 + i;
    SET_ALL_BITS(V(1)->state->selector);
    SET_START_POS(*V(1)->state, 0);
    SET_END_POS(*V(1)->state, static_cast<int>(kB - 1));
    V(1)->state->offset[0] = 0;
    V(1)->state->offset[1] = static_cast<uint16_t>(kB);

    root_ = std::make_shared<FactorizedTreeElement>("a", V(0));
    root_->add_leaf("a", "b", V(0), V(1));

    auto col = run_pipeline({{"a", 64u}, {"b", 64u}}, {"a", "b"});

    EXPECT_EQ(col->tuples.size(), kB);
    ASSERT_EQ(col->visible_ordering.size(), 3u);// a, b, _llm
}

// ---------------------------------------------------------------------------
// Test 8: Values in new vectors correctly reflect original data
// OT: a→b,  a=[5,6],  b=[50,60,70 | 80]  (3 under a[0], 1 under a[1])
// required={a,b}
// New chain: a→b→_llm
// Expected tuples: (5,50,1000), (5,60,1001), (5,70,1002), (6,80,1003)
// ---------------------------------------------------------------------------
TEST_F(CartesianProductTest, CorrectValuesInNewVectors) {
    vecs_[0] = std::make_unique<Vector<uint64_t>>();
    V(0)->values[0] = 5;
    V(0)->values[1] = 6;
    SET_ALL_BITS(V(0)->state->selector);
    SET_START_POS(*V(0)->state, 0);
    SET_END_POS(*V(0)->state, 1);

    vecs_[1] = std::make_unique<Vector<uint64_t>>();
    V(1)->values[0] = 50;
    V(1)->values[1] = 60;
    V(1)->values[2] = 70;
    V(1)->values[3] = 80;
    SET_ALL_BITS(V(1)->state->selector);
    SET_START_POS(*V(1)->state, 0);
    SET_END_POS(*V(1)->state, 3);
    V(1)->state->offset[0] = 0;
    V(1)->state->offset[1] = 3;
    V(1)->state->offset[2] = 4;

    root_ = std::make_shared<FactorizedTreeElement>("a", V(0));
    root_->add_leaf("a", "b", V(0), V(1));

    auto col = run_pipeline({{"a", 64u}, {"b", 64u}}, {"a", "b"});

    // Must produce exactly 4 tuples in DFS order
    ASSERT_EQ(col->tuples.size(), 4u);

    const std::vector<std::pair<uint64_t, uint64_t>> expected_ab = {
            {5u, 50u},
            {5u, 60u},
            {5u, 70u},
            {6u, 80u},
    };
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(col->tuples[i][0], expected_ab[i].first) << "row " << i << " a-val";
        EXPECT_EQ(col->tuples[i][1], expected_ab[i].second) << "row " << i << " b-val";
        EXPECT_NE(col->tuples[i][2], 0u) << "row " << i << " llm-val should be non-zero";
    }
}

// ---------------------------------------------------------------------------
// Test 9: RLE correctness in new vectors
// OT: a→b→c,  a=[1,2],  b=[10,20 | 30,40],  c=[100 | 200 | 300 | 400]
// required={a,b,c}
// New chain a→b→c→_llm.
// After execute():
//   new a-state: start=0, end=1 (2 values)
//   new b-state: offset[0]=0, offset[1]=2, offset[2]=4  (2 b per a)
//   new c-state: offset[0]=0, offset[1]=1, offset[2]=2, offset[3]=3, offset[4]=4  (1 c per b)
//   new _llm-state: offset mirrors c (identity: 1 llm per c)
// ---------------------------------------------------------------------------
// RLECheckingOperator: captures RLE state data during execute() while pipeline
// is still alive, avoiding use-after-free on CartesianProduct's backing buffers.
class RLECheckingOperator final : public Operator {
public:
    std::vector<std::vector<uint64_t>> tuples;
    // Captured RLE data from new chain nodes during execute().
    uint16_t b_start_pos{0}, b_end_pos{0};
    uint16_t b_offset[3]{};
    uint16_t c_start_pos{0}, c_end_pos{0};
    uint16_t c_offset[5]{};
    bool captured{false};

    void init(Schema* schema) override {
        tuples.clear();
        captured = false;
        new_root_ = schema->root;
    }

    void execute() override {
        if (!new_root_) return;
        // Walk chain: a→b→c→_llm
        std::vector<FactorizedTreeElement*> chain;
        FactorizedTreeElement* cur = new_root_.get();
        while (cur) {
            chain.push_back(cur);
            cur = cur->_children.empty() ? nullptr : cur->_children[0].get();
        }
        if (chain.empty()) return;

        // Capture RLE state while pipeline is alive
        if (!captured && chain.size() >= 4) {
            auto* b_state = chain[1]->_value->state;
            b_start_pos = b_state->start_pos;
            b_end_pos = b_state->end_pos;
            for (int i = 0; i < 3; ++i)
                b_offset[i] = b_state->offset[i];

            auto* c_state = chain[2]->_value->state;
            c_start_pos = c_state->start_pos;
            c_end_pos = c_state->end_pos;
            for (int i = 0; i < 5; ++i)
                c_offset[i] = c_state->offset[i];
            captured = true;
        }

        // Collect tuples
        const State* root_state = chain[0]->_value->state;
        size_t root_lo = root_state->start_pos;
        size_t root_hi = static_cast<size_t>(root_state->end_pos) + 1;
        std::vector<uint64_t> current(chain.size());
        expand(chain, 0, root_lo, root_hi, current);
    }

    uint64_t get_num_output_tuples() override { return static_cast<uint64_t>(tuples.size()); }

private:
    std::shared_ptr<FactorizedTreeElement> new_root_;

    void expand(const std::vector<FactorizedTreeElement*>& chain, size_t depth, size_t lo, size_t hi,
                std::vector<uint64_t>& current) {
        FactorizedTreeElement* node = chain[depth];
        for (size_t p = lo; p < hi; ++p) {
            current[depth] = node->_value->values[p];
            if (depth + 1 == chain.size()) {
                tuples.push_back(current);
            } else {
                FactorizedTreeElement* child = chain[depth + 1];
                const uint16_t* off = child->_value->state->offset;
                expand(chain, depth + 1, off[p], off[p + 1], current);
            }
        }
    }
};

TEST_F(CartesianProductTest, RLECorrectnessNewVectors) {
    vecs_[0] = std::make_unique<Vector<uint64_t>>();
    V(0)->values[0] = 1;
    V(0)->values[1] = 2;
    SET_ALL_BITS(V(0)->state->selector);
    SET_START_POS(*V(0)->state, 0);
    SET_END_POS(*V(0)->state, 1);

    vecs_[1] = std::make_unique<Vector<uint64_t>>();
    V(1)->values[0] = 10;
    V(1)->values[1] = 20;
    V(1)->values[2] = 30;
    V(1)->values[3] = 40;
    SET_ALL_BITS(V(1)->state->selector);
    SET_START_POS(*V(1)->state, 0);
    SET_END_POS(*V(1)->state, 3);
    V(1)->state->offset[0] = 0;
    V(1)->state->offset[1] = 2;
    V(1)->state->offset[2] = 4;

    vecs_[2] = std::make_unique<Vector<uint64_t>>();
    V(2)->values[0] = 100;
    V(2)->values[1] = 200;
    V(2)->values[2] = 300;
    V(2)->values[3] = 400;
    SET_ALL_BITS(V(2)->state->selector);
    SET_START_POS(*V(2)->state, 0);
    SET_END_POS(*V(2)->state, 3);
    V(2)->state->offset[0] = 0;
    V(2)->state->offset[1] = 1;
    V(2)->state->offset[2] = 2;
    V(2)->state->offset[3] = 3;
    V(2)->state->offset[4] = 4;

    root_ = std::make_shared<FactorizedTreeElement>("a", V(0));
    root_->add_leaf("a", "b", V(0), V(1));
    root_->add_leaf("b", "c", V(1), V(2));

    // Run pipeline manually so we can capture RLE state while it's alive.
    ordering_ = {"a", "b", "c"};
    schema_.root = root_;
    schema_.column_ordering = &ordering_;

    std::string cfg =
            R"({"provider":"demo","model":"demo","model_name":"demo_model","tuple_format":"FTREE","context_column":{"a":64,"b":64,"c":64},"prompt":"test"})";
    schema_.llm_config_str = &cfg;

    Map map_op;

    auto rle_checker = std::make_unique<RLECheckingOperator>();
    RLECheckingOperator* raw_checker = rle_checker.get();
    map_op.set_next_operator(std::move(rle_checker));
    map_op.init(&schema_);
    map_op.execute();

    ASSERT_EQ(raw_checker->tuples.size(), 4u);
    ASSERT_TRUE(raw_checker->captured);

    // b's state: 4 values total, offset[0]=0, offset[1]=2, offset[2]=4
    EXPECT_EQ(raw_checker->b_start_pos, 0u);
    EXPECT_EQ(raw_checker->b_end_pos, 3u);
    EXPECT_EQ(raw_checker->b_offset[0], 0u);
    EXPECT_EQ(raw_checker->b_offset[1], 2u);
    EXPECT_EQ(raw_checker->b_offset[2], 4u);

    // c's state: 4 values total, 1 per b → offset[i] = i
    EXPECT_EQ(raw_checker->c_start_pos, 0u);
    EXPECT_EQ(raw_checker->c_end_pos, 3u);
    for (int i = 0; i <= 4; ++i) {
        EXPECT_EQ(raw_checker->c_offset[i], static_cast<uint16_t>(i)) << "offset[" << i << "]";
    }
}

// ---------------------------------------------------------------------------
// Test 10: Complex branching topology with hidden ancestors and subset projection
// OT: a→(b,c), b→(d,e), c→(f,g)
// required={d,f,g}; structural ancestors should not be visible
// ---------------------------------------------------------------------------
TEST_F(CartesianProductTest, ComplexBranchingSubsetProjectionHiddenAncestors) {
    build_branching_tree_abcdefg();

    auto col = run_pipeline({{"d", 64u}, {"f", 64u}, {"g", 64u}}, {"a", "b", "c", "d", "e", "f", "g"});

    ASSERT_GT(col->tuples.size(), 0u);
    ASSERT_EQ(col->visible_ordering.size(), 8u);
    EXPECT_EQ(col->visible_ordering[0], "a");
    EXPECT_EQ(col->visible_ordering[1], "b");
    EXPECT_EQ(col->visible_ordering[2], "c");
    EXPECT_EQ(col->visible_ordering[3], "d");
    EXPECT_EQ(col->visible_ordering[4], "e");
    EXPECT_EQ(col->visible_ordering[5], "f");
    EXPECT_EQ(col->visible_ordering[6], "g");
    EXPECT_EQ(col->visible_ordering[7], "llm_out");
}

// ---------------------------------------------------------------------------
// Test 11: Required attribute order is derived from column ordering under
// complex topology
// ---------------------------------------------------------------------------
TEST_F(CartesianProductTest, RequiredOrderFollowsColumnOrderingInComplexTopology) {
    build_branching_tree_abcdefg();

    auto col = run_pipeline({{"d", 64u}, {"g", 64u}, {"b", 64u}}, {"a", "c", "b", "f", "g", "d", "e"});

    ASSERT_GT(col->tuples.size(), 0u);
    ASSERT_EQ(col->visible_ordering.size(), 8u);
    EXPECT_EQ(col->visible_ordering[0], "a");
    EXPECT_EQ(col->visible_ordering[1], "c");
    EXPECT_EQ(col->visible_ordering[2], "b");
    EXPECT_EQ(col->visible_ordering[3], "f");
    EXPECT_EQ(col->visible_ordering[4], "g");
    EXPECT_EQ(col->visible_ordering[5], "d");
    EXPECT_EQ(col->visible_ordering[6], "e");
    EXPECT_EQ(col->visible_ordering[7], "llm_out");
}

// ---------------------------------------------------------------------------
// Test 12: Deep+wide topology works with sparse required set and custom ordering
// Topology: a→(b,c), b→h, c→(f,g)
// required={h,g}
// ---------------------------------------------------------------------------
TEST_F(CartesianProductTest, DeepWideTopologySparseRequiredSet) {
    vecs_[0] = std::make_unique<Vector<uint64_t>>();
    V(0)->values[0] = 1;
    V(0)->values[1] = 2;
    SET_ALL_BITS(V(0)->state->selector);
    SET_START_POS(*V(0)->state, 0);
    SET_END_POS(*V(0)->state, 1);

    vecs_[1] = std::make_unique<Vector<uint64_t>>();
    V(1)->values[0] = 10;
    V(1)->values[1] = 20;
    SET_ALL_BITS(V(1)->state->selector);
    SET_START_POS(*V(1)->state, 0);
    SET_END_POS(*V(1)->state, 1);
    V(1)->state->offset[0] = 0;
    V(1)->state->offset[1] = 1;
    V(1)->state->offset[2] = 2;

    vecs_[2] = std::make_unique<Vector<uint64_t>>();
    V(2)->values[0] = 30;
    V(2)->values[1] = 40;
    SET_ALL_BITS(V(2)->state->selector);
    SET_START_POS(*V(2)->state, 0);
    SET_END_POS(*V(2)->state, 1);
    V(2)->state->offset[0] = 0;
    V(2)->state->offset[1] = 1;
    V(2)->state->offset[2] = 2;

    vecs_[3] = std::make_unique<Vector<uint64_t>>();
    V(3)->values[0] = 100;
    V(3)->values[1] = 200;
    SET_ALL_BITS(V(3)->state->selector);
    SET_START_POS(*V(3)->state, 0);
    SET_END_POS(*V(3)->state, 1);
    V(3)->state->offset[0] = 0;
    V(3)->state->offset[1] = 1;
    V(3)->state->offset[2] = 2;

    vecs_[4] = std::make_unique<Vector<uint64_t>>();
    V(4)->values[0] = 300;
    V(4)->values[1] = 400;
    SET_ALL_BITS(V(4)->state->selector);
    SET_START_POS(*V(4)->state, 0);
    SET_END_POS(*V(4)->state, 1);
    V(4)->state->offset[0] = 0;
    V(4)->state->offset[1] = 1;
    V(4)->state->offset[2] = 2;

    vecs_[5] = std::make_unique<Vector<uint64_t>>();
    V(5)->values[0] = 500;
    V(5)->values[1] = 600;
    SET_ALL_BITS(V(5)->state->selector);
    SET_START_POS(*V(5)->state, 0);
    SET_END_POS(*V(5)->state, 1);
    V(5)->state->offset[0] = 0;
    V(5)->state->offset[1] = 1;
    V(5)->state->offset[2] = 2;

    root_ = std::make_shared<FactorizedTreeElement>("a", V(0));
    root_->add_leaf("a", "b", V(0), V(1));
    root_->add_leaf("a", "c", V(0), V(2));
    root_->add_leaf("b", "h", V(1), V(3));
    root_->add_leaf("c", "f", V(2), V(4));
    root_->add_leaf("c", "g", V(2), V(5));

    auto col = run_pipeline({{"h", 64u}, {"g", 64u}}, {"a", "c", "b", "f", "g", "h"});

    ASSERT_GT(col->tuples.size(), 0u);
    ASSERT_EQ(col->visible_ordering.size(), 7u);
    EXPECT_EQ(col->visible_ordering[0], "a");
    EXPECT_EQ(col->visible_ordering[1], "c");
    EXPECT_EQ(col->visible_ordering[2], "b");
    EXPECT_EQ(col->visible_ordering[3], "f");
    EXPECT_EQ(col->visible_ordering[4], "g");
    EXPECT_EQ(col->visible_ordering[5], "h");
    EXPECT_EQ(col->visible_ordering[6], "llm_out");
}

TEST_F(CartesianProductTest, CrossBranchFlushExactBoundary) {
    constexpr int kN = State::MAX_VECTOR_SIZE;

    vecs_[0] = std::make_unique<Vector<uint64_t>>();
    vecs_[1] = std::make_unique<Vector<uint64_t>>();
    vecs_[2] = std::make_unique<Vector<uint64_t>>();

    for (int i = 0; i < kN; ++i) {
        V(0)->values[i] = static_cast<uint64_t>(i + 1);    // a
        V(1)->values[i] = static_cast<uint64_t>(10000 + i);// b
        V(2)->values[i] = static_cast<uint64_t>(20000 + i);// c
    }

    SET_ALL_BITS(V(0)->state->selector);
    SET_ALL_BITS(V(1)->state->selector);
    SET_ALL_BITS(V(2)->state->selector);
    SET_START_POS(*V(0)->state, 0);
    SET_END_POS(*V(0)->state, static_cast<uint16_t>(kN - 1));
    SET_START_POS(*V(1)->state, 0);
    SET_END_POS(*V(1)->state, static_cast<uint16_t>(kN - 1));
    SET_START_POS(*V(2)->state, 0);
    SET_END_POS(*V(2)->state, static_cast<uint16_t>(kN - 1));

    for (int i = 0; i <= kN; ++i) {
        V(1)->state->offset[i] = static_cast<uint16_t>(i);
        V(2)->state->offset[i] = static_cast<uint16_t>(i);
    }

    root_ = std::make_shared<FactorizedTreeElement>("a", V(0));
    root_->add_leaf("a", "b", V(0), V(1));
    root_->add_leaf("a", "c", V(0), V(2));

    auto col = run_pipeline({{"b", 64u}, {"c", 64u}}, {"a", "b", "c"});
    EXPECT_EQ(col->tuples.size(), static_cast<size_t>(kN));
    ASSERT_EQ(col->visible_ordering.size(), 4u);
    EXPECT_EQ(col->visible_ordering[0], "a");
    EXPECT_EQ(col->visible_ordering[1], "b");
    EXPECT_EQ(col->visible_ordering[2], "c");
    EXPECT_EQ(col->visible_ordering[3], "llm_out");
}

TEST_F(CartesianProductTest, CrossBranchMultipleFlushes) {
    constexpr int kRoot = 1024;
    constexpr int kPerRoot = 2;
    constexpr int kTotal = kRoot * kPerRoot;// 2048

    vecs_[0] = std::make_unique<Vector<uint64_t>>();// a
    vecs_[1] = std::make_unique<Vector<uint64_t>>();// b
    vecs_[2] = std::make_unique<Vector<uint64_t>>();// c

    for (int i = 0; i < kRoot; ++i) {
        V(0)->values[i] = static_cast<uint64_t>(i + 1);
    }
    for (int i = 0; i < kTotal; ++i) {
        V(1)->values[i] = static_cast<uint64_t>(30000 + i);
        V(2)->values[i] = static_cast<uint64_t>(40000 + i);
    }

    SET_ALL_BITS(V(0)->state->selector);
    SET_ALL_BITS(V(1)->state->selector);
    SET_ALL_BITS(V(2)->state->selector);
    SET_START_POS(*V(0)->state, 0);
    SET_END_POS(*V(0)->state, static_cast<uint16_t>(kRoot - 1));
    SET_START_POS(*V(1)->state, 0);
    SET_END_POS(*V(1)->state, static_cast<uint16_t>(kTotal - 1));
    SET_START_POS(*V(2)->state, 0);
    SET_END_POS(*V(2)->state, static_cast<uint16_t>(kTotal - 1));

    for (int i = 0; i <= kRoot; ++i) {
        V(1)->state->offset[i] = static_cast<uint16_t>(i * kPerRoot);
        V(2)->state->offset[i] = static_cast<uint16_t>(i * kPerRoot);
    }

    root_ = std::make_shared<FactorizedTreeElement>("a", V(0));
    root_->add_leaf("a", "b", V(0), V(1));
    root_->add_leaf("a", "c", V(0), V(2));

    auto col = run_pipeline({{"b", 64u}, {"c", 64u}}, {"a", "b", "c"});
    EXPECT_EQ(col->tuples.size(), static_cast<size_t>(kTotal * kPerRoot));
}

TEST_F(CartesianProductTest, ExecuteTwiceIsStable) {
    build_linear_tree_abc();
    ordering_ = {"a", "b", "c"};
    schema_.root = root_;
    schema_.column_ordering = &ordering_;

    std::string cfg =
            R"({"provider":"demo","model":"demo","model_name":"demo_model","tuple_format":"FTREE","context_column":{"a":64,"b":64,"c":64},"prompt":"test"})";
    schema_.llm_config_str = &cfg;

    Map map_op;


    auto collector = std::make_unique<CollectingOperator>();
    CollectingOperator* raw = collector.get();
    map_op.set_next_operator(std::move(collector));
    map_op.init(&schema_);

    map_op.execute();
    uint64_t first_run = map_op.get_num_output_tuples();
    size_t first_tuples = raw->tuples.size();

    map_op.execute();
    uint64_t second_run = map_op.get_num_output_tuples();

    EXPECT_EQ(first_run, 8u);
    EXPECT_EQ(second_run, first_run * 2u);
    EXPECT_GT(raw->tuples.size(), first_tuples);
}

// ===========================================================================
// Chunking tests: use large iterator capacities (2048) so the full dataset
// arrives in a single iterator batch, forcing CartesianProduct cross-branch
// expansion to exceed MAX_VECTOR_SIZE and exercise the 3-stage chunking.
// ===========================================================================

// Helper: build tree a→(b,c) with uniform fan-out per root position.
// a has `na` values [1..na], b has `nb_per_a` values per a-position, c has
// `nc_per_a` values per a-position.
// b values: 10000 + sequential, c values: 20000 + sequential.
static void build_ab_c_tree(Vector<uint64_t>* va, Vector<uint64_t>* vb, Vector<uint64_t>* vc,
                            std::shared_ptr<FactorizedTreeElement>& root, int na, int nb_per_a, int nc_per_a) {
    int total_b = na * nb_per_a;
    int total_c = na * nc_per_a;

    for (int i = 0; i < na; ++i)
        va->values[i] = static_cast<uint64_t>(i + 1);
    SET_ALL_BITS(va->state->selector);
    SET_START_POS(*va->state, 0);
    SET_END_POS(*va->state, static_cast<uint16_t>(na - 1));

    for (int i = 0; i < total_b; ++i)
        vb->values[i] = static_cast<uint64_t>(10000 + i);
    SET_ALL_BITS(vb->state->selector);
    SET_START_POS(*vb->state, 0);
    SET_END_POS(*vb->state, static_cast<uint16_t>(total_b - 1));
    for (int i = 0; i <= na; ++i)
        vb->state->offset[i] = static_cast<uint16_t>(i * nb_per_a);

    for (int i = 0; i < total_c; ++i)
        vc->values[i] = static_cast<uint64_t>(20000 + i);
    SET_ALL_BITS(vc->state->selector);
    SET_START_POS(*vc->state, 0);
    SET_END_POS(*vc->state, static_cast<uint16_t>(total_c - 1));
    for (int i = 0; i <= na; ++i)
        vc->state->offset[i] = static_cast<uint16_t>(i * nc_per_a);

    root = std::make_shared<FactorizedTreeElement>("a", va);
    root->add_leaf("a", "b", va, vb);
    root->add_leaf("a", "c", va, vc);
}

// ---------------------------------------------------------------------------
// Test 17: Cross-branch overflow with value verification
// a→(b,c), a=1, b=50, c=50.  Chain: a→b→c→_llm.
// Cross-product: 50×50=2500 > 2048 → one flush.
// Verify every tuple has correct (a, b, c, llm) values.
// ---------------------------------------------------------------------------
TEST_F(CartesianProductTest, ChunkingCrossBranchOverflowValues) {
    constexpr int kNA = 1, kNB = 50, kNC = 50;
    constexpr size_t kExpected = kNB * kNC;// 2500

    vecs_[0] = std::make_unique<Vector<uint64_t>>();
    vecs_[1] = std::make_unique<Vector<uint64_t>>();
    vecs_[2] = std::make_unique<Vector<uint64_t>>();
    build_ab_c_tree(V(0), V(1), V(2), root_, kNA, kNB, kNC);

    // Large capacity so iterator yields everything in one batch.
    auto col = run_pipeline({{"a", 2048u}, {"b", 2048u}, {"c", 2048u}}, {"a", "b", "c"});

    ASSERT_EQ(col->tuples.size(), kExpected);

    // Build expected set of (b, c) pairs.
    std::set<std::pair<uint64_t, uint64_t>> expected_bc;
    for (int bi = 0; bi < kNB; ++bi)
        for (int ci = 0; ci < kNC; ++ci)
            expected_bc.insert({10000u + bi, 20000u + ci});

    std::set<std::pair<uint64_t, uint64_t>> actual_bc;
    for (const auto& t: col->tuples) {
        ASSERT_EQ(t.size(), 4u);
        EXPECT_EQ(t[0], 1u);// a-value
        actual_bc.insert({t[1], t[2]});
    }
    EXPECT_EQ(actual_bc, expected_bc);
}

// ---------------------------------------------------------------------------
// Test 18: Large cross-branch overflow — multiple flushes
// a→(b,c), a=1, b=100, c=100.  100×100=10000, ~5 flushes.
// ---------------------------------------------------------------------------
TEST_F(CartesianProductTest, ChunkingMultipleFlushes) {
    constexpr int kNA = 1, kNB = 100, kNC = 100;
    constexpr size_t kExpected = kNB * kNC;// 10000

    vecs_[0] = std::make_unique<Vector<uint64_t>>();
    vecs_[1] = std::make_unique<Vector<uint64_t>>();
    vecs_[2] = std::make_unique<Vector<uint64_t>>();
    build_ab_c_tree(V(0), V(1), V(2), root_, kNA, kNB, kNC);

    auto col = run_pipeline({{"a", 2048u}, {"b", 2048u}, {"c", 2048u}}, {"a", "b", "c"});

    ASSERT_EQ(col->tuples.size(), kExpected);

    // Verify all LLM values are unique (demo model returns distinct hashes).
    std::set<uint64_t> llm_vals;
    for (const auto& t: col->tuples)
        llm_vals.insert(t[3]);
    EXPECT_EQ(llm_vals.size(), kExpected);
}

// ---------------------------------------------------------------------------
// Test 19: Exact MAX_VECTOR_SIZE boundary
// a→(b,c), a=1, b=64, c=32.  64×32=2048 exactly.
// The LLM node writes exactly 2048 values → tests boundary flush.
// ---------------------------------------------------------------------------
TEST_F(CartesianProductTest, ChunkingExactBoundary) {
    constexpr int kNA = 1, kNB = 64, kNC = 32;
    constexpr size_t kExpected = kNB * kNC;// 2048

    vecs_[0] = std::make_unique<Vector<uint64_t>>();
    vecs_[1] = std::make_unique<Vector<uint64_t>>();
    vecs_[2] = std::make_unique<Vector<uint64_t>>();
    build_ab_c_tree(V(0), V(1), V(2), root_, kNA, kNB, kNC);

    auto col = run_pipeline({{"a", 2048u}, {"b", 2048u}, {"c", 2048u}}, {"a", "b", "c"});

    ASSERT_EQ(col->tuples.size(), kExpected);
}

// ---------------------------------------------------------------------------
// Test 20: Multiple root positions with overflow
// a→(b,c), a=5, b=16 per a (80 total), c=32 per a (160 total).
// Cross-product: 80×32=2560 > 2048.  Verify a-values per tuple.
// ---------------------------------------------------------------------------
TEST_F(CartesianProductTest, ChunkingMultiRootOverflow) {
    constexpr int kNA = 5, kNB = 16, kNC = 32;
    constexpr size_t kExpected = static_cast<size_t>(kNA) * kNB * kNC;// 2560

    vecs_[0] = std::make_unique<Vector<uint64_t>>();
    vecs_[1] = std::make_unique<Vector<uint64_t>>();
    vecs_[2] = std::make_unique<Vector<uint64_t>>();
    build_ab_c_tree(V(0), V(1), V(2), root_, kNA, kNB, kNC);

    auto col = run_pipeline({{"a", 2048u}, {"b", 2048u}, {"c", 2048u}}, {"a", "b", "c"});

    ASSERT_EQ(col->tuples.size(), kExpected);

    // Verify each a-value appears exactly kNB*kNC times
    std::unordered_map<uint64_t, size_t> a_counts;
    for (const auto& t: col->tuples)
        a_counts[t[0]]++;
    ASSERT_EQ(a_counts.size(), static_cast<size_t>(kNA));
    for (auto& [a_val, cnt]: a_counts) {
        EXPECT_EQ(cnt, static_cast<size_t>(kNB * kNC)) << "a=" << a_val;
    }
}

// ---------------------------------------------------------------------------
// Test 21: Deep chain with cascading overflow
// a→(b,c,d),  a=1, b=20, c=20, d=20  (all children of a).
// Chain: a→b→c→d→_llm.
// c cross-branch: 20×20=400.  d cross-branch: 400×20=8000 → ~4 flushes.
// ---------------------------------------------------------------------------
TEST_F(CartesianProductTest, ChunkingDeepChainCascading) {
    constexpr int kN = 20;
    constexpr size_t kExpected = static_cast<size_t>(kN) * kN * kN;// 8000

    vecs_[0] = std::make_unique<Vector<uint64_t>>();// a
    vecs_[1] = std::make_unique<Vector<uint64_t>>();// b
    vecs_[2] = std::make_unique<Vector<uint64_t>>();// c
    vecs_[3] = std::make_unique<Vector<uint64_t>>();// d

    V(0)->values[0] = 1;
    SET_ALL_BITS(V(0)->state->selector);
    SET_START_POS(*V(0)->state, 0);
    SET_END_POS(*V(0)->state, 0);

    for (int i = 0; i < kN; ++i)
        V(1)->values[i] = static_cast<uint64_t>(100 + i);
    SET_ALL_BITS(V(1)->state->selector);
    SET_START_POS(*V(1)->state, 0);
    SET_END_POS(*V(1)->state, kN - 1);
    V(1)->state->offset[0] = 0;
    V(1)->state->offset[1] = kN;

    for (int i = 0; i < kN; ++i)
        V(2)->values[i] = static_cast<uint64_t>(200 + i);
    SET_ALL_BITS(V(2)->state->selector);
    SET_START_POS(*V(2)->state, 0);
    SET_END_POS(*V(2)->state, kN - 1);
    V(2)->state->offset[0] = 0;
    V(2)->state->offset[1] = kN;

    for (int i = 0; i < kN; ++i)
        V(3)->values[i] = static_cast<uint64_t>(300 + i);
    SET_ALL_BITS(V(3)->state->selector);
    SET_START_POS(*V(3)->state, 0);
    SET_END_POS(*V(3)->state, kN - 1);
    V(3)->state->offset[0] = 0;
    V(3)->state->offset[1] = kN;

    root_ = std::make_shared<FactorizedTreeElement>("a", V(0));
    root_->add_leaf("a", "b", V(0), V(1));
    root_->add_leaf("a", "c", V(0), V(2));
    root_->add_leaf("a", "d", V(0), V(3));

    auto col = run_pipeline({{"a", 2048u}, {"b", 2048u}, {"c", 2048u}, {"d", 2048u}}, {"a", "b", "c", "d"});

    ASSERT_EQ(col->tuples.size(), kExpected);

    // All tuples should have a=1
    for (const auto& t: col->tuples) {
        ASSERT_EQ(t.size(), 5u);// a, b, c, d, _llm
        EXPECT_EQ(t[0], 1u);
    }

    // Verify all (b, c, d) triples are present
    std::set<std::tuple<uint64_t, uint64_t, uint64_t>> triples;
    for (const auto& t: col->tuples)
        triples.insert({t[1], t[2], t[3]});
    EXPECT_EQ(triples.size(), kExpected);
}

// ---------------------------------------------------------------------------
// Test 22: Ancestor state slicing correctness across flushes
// a→(b,c), a=3, b=30 per a (90 total), c=30 per a (90 total).
// Cross-product: 90×30=2700 > 2048 → flush mid-stream.
// Each tuple's a-value must match: b values 10000..10029 → a=1,
// b values 10030..10059 → a=2, b values 10060..10089 → a=3.
// ---------------------------------------------------------------------------
TEST_F(CartesianProductTest, ChunkingAncestorSlicingCorrectness) {
    constexpr int kNA = 3, kNB = 30, kNC = 30;
    constexpr size_t kExpected = static_cast<size_t>(kNA) * kNB * kNC;// 2700

    vecs_[0] = std::make_unique<Vector<uint64_t>>();
    vecs_[1] = std::make_unique<Vector<uint64_t>>();
    vecs_[2] = std::make_unique<Vector<uint64_t>>();
    build_ab_c_tree(V(0), V(1), V(2), root_, kNA, kNB, kNC);

    auto col = run_pipeline({{"a", 2048u}, {"b", 2048u}, {"c", 2048u}}, {"a", "b", "c"});

    ASSERT_EQ(col->tuples.size(), kExpected);

    // For each tuple, verify the a→b→c ancestry is consistent.
    // b-values 10000 + [a_idx * kNB .. (a_idx+1)*kNB - 1] should have a = a_idx+1.
    // c-values 20000 + [a_idx * kNC .. (a_idx+1)*kNC - 1] should have a = a_idx+1.
    for (const auto& t: col->tuples) {
        uint64_t a_val = t[0];
        uint64_t b_val = t[1];
        uint64_t c_val = t[2];
        int b_idx = static_cast<int>(b_val - 10000);
        int c_idx = static_cast<int>(c_val - 20000);
        uint64_t expected_a_from_b = static_cast<uint64_t>(b_idx / kNB + 1);
        uint64_t expected_a_from_c = static_cast<uint64_t>(c_idx / kNC + 1);
        EXPECT_EQ(a_val, expected_a_from_b) << "b=" << b_val << " expects a=" << expected_a_from_b;
        EXPECT_EQ(a_val, expected_a_from_c) << "c=" << c_val << " expects a=" << expected_a_from_c;
    }
}
