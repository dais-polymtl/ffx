#include "../../src/operator/include/ai_operator/ai_serializer.hpp"
#include "../../src/operator/include/factorized_ftree/ftree_batch_iterator.hpp"
#include "../../src/operator/include/schema/schema.hpp"
#include "../../src/operator/include/vector/state.hpp"
#include "../../src/operator/include/vector/vector.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace ffx;

class AISerializerTest : public ::testing::Test {
protected:
    std::vector<std::unique_ptr<Vector<uint64_t>>> _vecs;
    std::shared_ptr<FactorizedTreeElement> _root;
    std::vector<std::string> _ordering;
    Schema _schema;

    std::unique_ptr<StringPool> _pool;
    std::unique_ptr<StringDictionary> _dict;
    std::unordered_set<std::string> _string_attrs;

    void SetUp() override {
        _pool = std::make_unique<StringPool>();
        _dict = std::make_unique<StringDictionary>(_pool.get());
        _string_attrs = {"atitle", "aabs"};
        _schema.dictionary = _dict.get();
        _schema.string_attributes = &_string_attrs;
    }

    Vector<uint64_t>* make_vec() {
        _vecs.push_back(std::make_unique<Vector<uint64_t>>());
        return _vecs.back().get();
    }

    static void init_state(Vector<uint64_t>* v, int start, int end) {
        SET_ALL_BITS(v->state->selector);
        SET_START_POS(*v->state, start);
        SET_END_POS(*v->state, end);
    }

    void reset_schema() {
        _vecs.clear();
        _ordering.clear();
        _root.reset();
    }

    void build_two_leaf_string_tree() {
        reset_schema();
        _string_attrs = {"name", "desc"};

        const uint64_t n0 = _dict->add_string(ffx_str_t("node-0", _pool.get()));
        const uint64_t n1 = _dict->add_string(ffx_str_t("node-1", _pool.get()));
        const uint64_t n2 = _dict->add_string(ffx_str_t("node-2", _pool.get()));

        const uint64_t d0 = _dict->add_string(ffx_str_t("desc-0", _pool.get()));
        const uint64_t d1 = _dict->add_string(ffx_str_t("desc-1", _pool.get()));
        const uint64_t d2 = _dict->add_string(ffx_str_t("desc-2", _pool.get()));
        _dict->finalize();

        auto* a = make_vec();
        auto* name = make_vec();
        auto* desc = make_vec();

        a->values[0] = 0; a->values[1] = 1; a->values[2] = 2;
        init_state(a, 0, 2);

        name->values[0] = n0; name->values[1] = n1; name->values[2] = n2;
        init_state(name, 0, 2);
        name->state->offset[0] = 0; name->state->offset[1] = 1; name->state->offset[2] = 2; name->state->offset[3] = 3;

        desc->values[0] = d0; desc->values[1] = d1; desc->values[2] = d2;
        init_state(desc, 0, 2);
        desc->state->offset[0] = 0; desc->state->offset[1] = 1; desc->state->offset[2] = 2; desc->state->offset[3] = 3;

        _root = std::make_shared<FactorizedTreeElement>("a", a);
        _root->add_leaf("a", "name", a, name);
        _root->add_leaf("a", "desc", a, desc);
        _ordering = {"a", "name", "desc"};
        _schema.root = _root;
        _schema.column_ordering = &_ordering;
    }

    // r -> p -> q -> tag, with variable fan-out.
    void build_deep_chain_tree() {
        reset_schema();
        _string_attrs = {"tag"};

        const uint64_t t0 = _dict->add_string(ffx_str_t("tag-a", _pool.get()));
        const uint64_t t1 = _dict->add_string(ffx_str_t("tag-b", _pool.get()));
        const uint64_t t2 = _dict->add_string(ffx_str_t("tag-c", _pool.get()));
        _dict->finalize();

        auto* r = make_vec();
        auto* p = make_vec();
        auto* q = make_vec();
        auto* tag = make_vec();

        r->values[0] = 1; r->values[1] = 2;
        init_state(r, 0, 1);

        p->values[0] = 10; p->values[1] = 20; p->values[2] = 30;
        init_state(p, 0, 2);
        p->state->offset[0] = 0; p->state->offset[1] = 1; p->state->offset[2] = 3;

        q->values[0] = 100; q->values[1] = 200; q->values[2] = 300; q->values[3] = 400;
        init_state(q, 0, 3);
        q->state->offset[0] = 0; q->state->offset[1] = 1; q->state->offset[2] = 3; q->state->offset[3] = 4;

        tag->values[0] = t0; tag->values[1] = t1; tag->values[2] = t2; tag->values[3] = t0;
        init_state(tag, 0, 3);
        tag->state->offset[0] = 0; tag->state->offset[1] = 1; tag->state->offset[2] = 2; tag->state->offset[3] = 3; tag->state->offset[4] = 4;

        _root = std::make_shared<FactorizedTreeElement>("r", r);
        _root->add_leaf("r", "p", r, p);
        _root->add_leaf("p", "q", p, q);
        _root->add_leaf("q", "tag", q, tag);
        _ordering = {"r", "p", "q", "tag"};
        _schema.root = _root;
        _schema.column_ordering = &_ordering;
    }

    // Root with mixed branches: r -> (x, y), x -> z
    void build_branching_tree() {
        reset_schema();
        _string_attrs = {"yname"};
        const uint64_t ys0 = _dict->add_string(ffx_str_t("left", _pool.get()));
        const uint64_t ys1 = _dict->add_string(ffx_str_t("right", _pool.get()));
        _dict->finalize();

        auto* r = make_vec();
        auto* x = make_vec();
        auto* yname = make_vec();
        auto* z = make_vec();

        r->values[0] = 5; r->values[1] = 6;
        init_state(r, 0, 1);

        x->values[0] = 50; x->values[1] = 60; x->values[2] = 70;
        init_state(x, 0, 2);
        x->state->offset[0] = 0; x->state->offset[1] = 2; x->state->offset[2] = 3;

        yname->values[0] = ys0; yname->values[1] = ys1;
        init_state(yname, 0, 1);
        yname->state->offset[0] = 0; yname->state->offset[1] = 1; yname->state->offset[2] = 2;

        z->values[0] = 500; z->values[1] = 600; z->values[2] = 700; z->values[3] = 800;
        init_state(z, 0, 3);
        z->state->offset[0] = 0; z->state->offset[1] = 2; z->state->offset[2] = 3; z->state->offset[3] = 4;

        _root = std::make_shared<FactorizedTreeElement>("r", r);
        _root->add_leaf("r", "x", r, x);
        _root->add_leaf("r", "yname", r, yname);
        _root->add_leaf("x", "z", x, z);
        _ordering = {"r", "x", "z", "yname"};
        _schema.root = _root;
        _schema.column_ordering = &_ordering;
    }

    // Multi-branch shape: u -> (v, w), v -> (x, y), w -> z
    void build_double_branching_tree() {
        reset_schema();
        _string_attrs = {"label"};
        const uint64_t l0 = _dict->add_string(ffx_str_t("alpha", _pool.get()));
        const uint64_t l1 = _dict->add_string(ffx_str_t("beta", _pool.get()));
        _dict->finalize();

        auto* u = make_vec();
        auto* v = make_vec();
        auto* w = make_vec();
        auto* x = make_vec();
        auto* y = make_vec();
        auto* z = make_vec();
        auto* label = make_vec();

        u->values[0] = 1; u->values[1] = 2;
        init_state(u, 0, 1);

        v->values[0] = 10; v->values[1] = 11; v->values[2] = 20;
        init_state(v, 0, 2);
        v->state->offset[0] = 0; v->state->offset[1] = 2; v->state->offset[2] = 3;

        w->values[0] = 100; w->values[1] = 200;
        init_state(w, 0, 1);
        w->state->offset[0] = 0; w->state->offset[1] = 1; w->state->offset[2] = 2;

        x->values[0] = 1000; x->values[1] = 1100; x->values[2] = 2000;
        init_state(x, 0, 2);
        x->state->offset[0] = 0; x->state->offset[1] = 1; x->state->offset[2] = 2; x->state->offset[3] = 3;

        y->values[0] = 3000; y->values[1] = 3100; y->values[2] = 4000;
        init_state(y, 0, 2);
        y->state->offset[0] = 0; y->state->offset[1] = 1; y->state->offset[2] = 2; y->state->offset[3] = 3;

        z->values[0] = 5000; z->values[1] = 6000;
        init_state(z, 0, 1);
        z->state->offset[0] = 0; z->state->offset[1] = 1; z->state->offset[2] = 2;

        label->values[0] = l0; label->values[1] = l1;
        init_state(label, 0, 1);
        label->state->offset[0] = 0; label->state->offset[1] = 1; label->state->offset[2] = 2;

        _root = std::make_shared<FactorizedTreeElement>("u", u);
        _root->add_leaf("u", "v", u, v);
        _root->add_leaf("u", "w", u, w);
        _root->add_leaf("v", "x", v, x);
        _root->add_leaf("v", "y", v, y);
        _root->add_leaf("w", "z", w, z);
        _root->add_leaf("u", "label", u, label);
        _ordering = {"u", "v", "x", "y", "w", "z", "label"};
        _schema.root = _root;
        _schema.column_ordering = &_ordering;
    }

    static std::set<std::string> root_child_keys(const nlohmann::json& root_object) {
        std::set<std::string> out;
        if (!root_object.is_object()) return out;
        for (auto it = root_object.begin(); it != root_object.end(); ++it) {
            out.insert(it.key());
        }
        return out;
    }

    static std::set<std::string> node_child_keys(const nlohmann::json& node) {
        std::set<std::string> out;
        if (!node.is_object() || !node.contains("children") || !node["children"].is_object()) return out;
        for (auto it = node["children"].begin(); it != node["children"].end(); ++it) out.insert(it.key());
        return out;
    }

    static void prime_iterator(FTreeBatchIterator& itr, Schema* schema) {
        itr.init(schema);
        itr.reset();
        itr.initialize_iterators();
    }
};

TEST_F(AISerializerTest, FlatColumnsRespectsRequestedAttributeOrderAndDecoding) {
    build_two_leaf_string_tree();
    FTreeBatchIterator itr({{"name", 16}, {"desc", 16}});
    prime_iterator(itr, &_schema);
    ASSERT_TRUE(itr.next());

    AISerializer serializer("JSON");
    auto columns = serializer.build_columns(itr, &_schema, {"desc", "name"}, itr.count_logical_tuples());

    ASSERT_EQ(columns.size(), 2u);
    EXPECT_EQ(columns[0]["name"], "desc");
    EXPECT_EQ(columns[1]["name"], "name");
    EXPECT_EQ(columns[0]["data"][0], "desc-0");
    EXPECT_EQ(columns[0]["data"][1], "desc-1");
    EXPECT_EQ(columns[0]["data"][2], "desc-2");
    EXPECT_EQ(columns[1]["data"][0], "node-0");
    EXPECT_EQ(columns[1]["data"][1], "node-1");
    EXPECT_EQ(columns[1]["data"][2], "node-2");
}

TEST_F(AISerializerTest, FlatColumnsUseProjectedOrderingOnComplexDeepTree) {
    build_deep_chain_tree();
    FTreeBatchIterator itr({{"tag", 64}});
    prime_iterator(itr, &_schema);
    ASSERT_TRUE(itr.next());

    AISerializer serializer("JSON");
    auto columns = serializer.build_columns(itr, &_schema, {}, itr.count_logical_tuples());

    ASSERT_EQ(columns.size(), 4u);
    EXPECT_EQ(columns[0]["name"], "r");
    EXPECT_EQ(columns[1]["name"], "p");
    EXPECT_EQ(columns[2]["name"], "q");
    EXPECT_EQ(columns[3]["name"], "tag");
    const auto logical_count = itr.count_logical_tuples();
    for (const auto& c : columns) {
        ASSERT_EQ(c["data"].size(), logical_count);
    }
    EXPECT_TRUE(columns[3]["data"][0].is_string());
}

TEST_F(AISerializerTest, FTreeColumnsAreStableAcrossMultipleComplexTopologies) {
    struct FTreeCase {
        std::string name;
        std::unordered_map<std::string, size_t> required;
        std::string expected_root;
        std::set<std::string> expected_root_children;
        std::string nested_parent;
        std::set<std::string> expected_first_child_children;
    };
    const std::vector<FTreeCase> cases = {
        {"branching", {{"z", 64}, {"yname", 64}}, "r", {"x", "yname"}, "x", {"z"}},
        {"deep", {{"tag", 64}}, "r", {"p"}, "p", {"q"}},
        {"two_leaf", {{"name", 64}, {"desc", 64}}, "a", {"desc", "name"}, "", {}},
        {"double_branching", {{"x", 64}, {"y", 64}, {"z", 64}, {"label", 64}}, "u", {"label", "v", "w"}, "v", {"x", "y"}},
    };

    for (const auto& tc : cases) {
        if (tc.name == "branching") build_branching_tree();
        if (tc.name == "deep") build_deep_chain_tree();
        if (tc.name == "two_leaf") build_two_leaf_string_tree();
        if (tc.name == "double_branching") build_double_branching_tree();

        FTreeBatchIterator itr(tc.required);
        prime_iterator(itr, &_schema);
        ASSERT_TRUE(itr.next()) << tc.name;

        AISerializer serializer("FTREE");
        const size_t logical_count = itr.count_logical_tuples();
        auto columns = serializer.build_columns(itr, &_schema, {}, logical_count);

        ASSERT_EQ(columns.size(), 1u) << tc.name;
        EXPECT_EQ(columns[0]["name"], "ftree") << tc.name;
        ASSERT_EQ(columns[0]["data"].size(), logical_count) << tc.name;
        ASSERT_TRUE(columns[0]["data"][0].contains("root")) << tc.name;
        ASSERT_TRUE(columns[0]["data"][0].contains("tree")) << tc.name;
        ASSERT_TRUE(columns[0]["data"][0]["tree"].is_object()) << tc.name;
        EXPECT_EQ(columns[0]["data"][0]["root"], tc.expected_root) << tc.name;

        const auto& tree = columns[0]["data"][0]["tree"];
        ASSERT_FALSE(tree.empty()) << tc.name;
        auto first_root_it = tree.begin();
        ASSERT_TRUE(first_root_it.value().is_object()) << tc.name;

        EXPECT_EQ(root_child_keys(first_root_it.value()), tc.expected_root_children) << tc.name;
        if (!tc.expected_first_child_children.empty()) {
            ASSERT_FALSE(tc.nested_parent.empty()) << tc.name;
            ASSERT_TRUE(first_root_it.value().contains(tc.nested_parent)) << tc.name;
            EXPECT_EQ(node_child_keys(first_root_it.value()[tc.nested_parent]), tc.expected_first_child_children) << tc.name;
        }
    }
}

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
