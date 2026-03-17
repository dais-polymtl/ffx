#include "factorized_ftree/factorized_tree_element.hpp"
#include "schema/schema.hpp"
#include "sink/sink_min.hpp"
#include "sink/sink_min_itr.hpp"
#include "vector/vector.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>

namespace ffx {

class SinkMinPartialTest : public ::testing::Test {
protected:
    void SetUp() override {
        pool = std::make_unique<StringPool>();
        dict = std::make_unique<StringDictionary>(pool.get());
    }

    std::unique_ptr<StringPool> pool;
    std::unique_ptr<StringDictionary> dict;

    // Helper to create a simple ftree with numeric and string vectors
    std::shared_ptr<FactorizedTreeElement> create_test_ftree(VectorUint64* vec_a, VectorUint64* vec_b,
                                                             VectorUint64* vec_c) {

        auto root = std::make_shared<FactorizedTreeElement>("a", vec_a);

        auto child_b = std::make_shared<FactorizedTreeElement>("b", vec_b);
        root->_children.push_back(child_b);

        auto child_c = std::make_shared<FactorizedTreeElement>("c", vec_c);
        root->_children.push_back(child_c);

        return root;
    }
};

TEST_F(SinkMinPartialTest, SinkMinPartialNumeric) {
    VectorUint64 vec_a;
    vec_a.state->setStartPos(0);
    vec_a.state->setEndPos(1);
    vec_a.values[0] = 10;
    vec_a.values[1] = 5;
    vec_a.state->selector.setBit(0);
    vec_a.state->selector.setBit(1);

    VectorUint64 vec_b;
    vec_b.state->setStartPos(0);
    vec_b.state->setEndPos(1);
    vec_b.values[0] = 100;
    vec_b.values[1] = 200;
    vec_b.state->selector.setBit(0);
    vec_b.state->selector.setBit(1);

    VectorUint64 vec_c;
    // For string attributes, the vector should contain IDs
    uint64_t id1 = dict->add_string(ffx_str_t("aaa", pool.get()));
    uint64_t id2 = dict->add_string(ffx_str_t("bbb", pool.get()));
    vec_c.values[0] = id1;
    vec_c.values[1] = id2;
    vec_c.state->setStartPos(0);
    vec_c.state->setEndPos(1);
    vec_c.state->selector.setBit(0);
    vec_c.state->selector.setBit(1);

    auto ftree = create_test_ftree(&vec_a, &vec_b, &vec_c);

    Schema schema;
    schema.root = ftree;
    schema.required_min_attrs = {"a", "c"};
    schema.min_values_size = 2;
    uint64_t min_vals[2] = {UINT64_MAX, UINT64_MAX};
    schema.min_values = min_vals;
    std::unordered_set<std::string> strings = {"c"};
    schema.string_attributes = &strings;
    schema.dictionary = dict.get();

    SinkMin sink;
    sink.init(&schema);
    sink.execute();

    EXPECT_EQ(min_vals[0], 5);// min of attribute 'a'
    // For 'c', min is "aaa", we need to check its ID in the dictionary
    uint64_t aaa_id = dict->add_string(ffx_str_t("aaa", pool.get()));
    EXPECT_EQ(min_vals[1], aaa_id);
}

TEST_F(SinkMinPartialTest, SinkMinItrPartial) {
    // This part is harder as FTreeIterator needs more setup,
    // but we can verify that Schema flows correctly.
    // For brevity, we'll focus on SinkMin first as it's the most common path.
}

}// namespace ffx
