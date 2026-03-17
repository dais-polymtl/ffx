#include "../src/operator/include/factorized_ftree/ftree_iterator.hpp"
#include "../src/operator/include/join/inljoin_packed.hpp"
#include "../src/operator/include/join/inljoin_packed_cascade.hpp"
#include "../src/operator/include/scan/scan.hpp"
#include "../src/operator/include/scan/scan_synchronized.hpp"
#include "../src/operator/include/schema/schema.hpp"
#include "../src/operator/include/sink/sink_min.hpp"
#include "../src/operator/include/sink/sink_min_itr.hpp"
#include "../src/operator/include/vector/state.hpp"
#include "../src/operator/include/vector/vector.hpp"
#include "../src/table/include/adj_list.hpp"
#include "../src/table/include/string_dictionary.hpp"
#include "../src/table/include/string_pool.hpp"
#include "../src/table/include/table.hpp"
#include <atomic>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <vector>

using namespace ffx;

//------------------------------------------------------------------------------
// Unit Tests for SinkMin with Integer Values (Regression Tests)
//------------------------------------------------------------------------------

class SinkMinIntegerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup QueryVariableToVectorMap and allocate vectors
        // for attributes A (root), B (child)
        a_vec = map.allocate_vector("A");
        b_vec = map.allocate_vector("B");

        // A: 4 positions with values 10, 5, 20, 3
        a_vec->values[0] = 10;
        a_vec->values[1] = 5;
        a_vec->values[2] = 20;
        a_vec->values[3] = 3;
        SET_ALL_BITS(a_vec->state->selector);
        SET_START_POS(*a_vec->state, 0);
        SET_END_POS(*a_vec->state, 3);

        // B: 4 positions with values 100, 50, 200, 25, mapped 1:1 to A
        b_vec->values[0] = 100;
        b_vec->values[1] = 50;
        b_vec->values[2] = 200;
        b_vec->values[3] = 25;
        SET_ALL_BITS(b_vec->state->selector);
        SET_START_POS(*b_vec->state, 0);
        SET_END_POS(*b_vec->state, 3);
        b_vec->state->offset[0] = 0;
        b_vec->state->offset[1] = 1;
        b_vec->state->offset[2] = 2;
        b_vec->state->offset[3] = 3;
        b_vec->state->offset[4] = 4;

        // Create factorized tree: A -> B
        root = std::make_shared<FactorizedTreeElement>("A", a_vec);
        root->add_leaf("A", "B", a_vec, b_vec);

        // Schema
        column_ordering = {"A", "B"};
        schema.root = root;
        schema.column_ordering = &column_ordering;
        schema.map = &map;

        // Allocate min_values
        min_values = std::make_unique<uint64_t[]>(2);
        schema.min_values = min_values.get();
        schema.min_values_size = 2;
        schema.required_min_attrs = {"A", "B"};
    }

    QueryVariableToVectorMap map;
    Vector<uint64_t>* a_vec;
    Vector<uint64_t>* b_vec;
    std::shared_ptr<FactorizedTreeElement> root;
    Schema schema;
    std::vector<std::string> column_ordering;
    std::unique_ptr<uint64_t[]> min_values;
};

TEST_F(SinkMinIntegerTest, FindsMinIntegerValues) {
    SinkMin sink;
    sink.init(&schema);
    sink.execute();

    // A min should be 3 (at position 3)
    EXPECT_EQ(min_values[0], 3);
    // B min should be 25 (at position 3)
    EXPECT_EQ(min_values[1], 25);
}

TEST_F(SinkMinIntegerTest, HandlesPartialSelection) {
    // Clear some bits - only positions 0 and 1 are selected
    // Must clear in both A and B since they have 1:1 mapping
    CLEAR_BIT(a_vec->state->selector, 2);
    CLEAR_BIT(a_vec->state->selector, 3);
    CLEAR_BIT(b_vec->state->selector, 2);
    CLEAR_BIT(b_vec->state->selector, 3);

    SinkMin sink;
    sink.init(&schema);
    sink.execute();

    // A min should be 5 (at position 1)
    EXPECT_EQ(min_values[0], 5);
    // B min should be 50 (at position 1)
    EXPECT_EQ(min_values[1], 50);
}

//------------------------------------------------------------------------------
// Unit Tests for SinkMin with String Values
//------------------------------------------------------------------------------

class SinkMinStringTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create string pool and dictionary
        string_pool = std::make_unique<StringPool>();
        string_dict = std::make_unique<StringDictionary>(string_pool.get());

        // Add strings to dictionary in NON-lexicographic order
        // IDs will be: "zebra"=0, "apple"=1, "mango"=2, "banana"=3
        id_zebra = string_dict->add_string(ffx_str_t("zebra", string_pool.get()));
        id_apple = string_dict->add_string(ffx_str_t("apple", string_pool.get()));
        id_mango = string_dict->add_string(ffx_str_t("mango", string_pool.get()));
        id_banana = string_dict->add_string(ffx_str_t("banana", string_pool.get()));

        // Setup vectors
        a_vec = map.allocate_vector("Name");

        // A: 4 positions with string IDs (in insertion order, not lexicographic)
        a_vec->values[0] = id_zebra; // "zebra" - should NOT be min
        a_vec->values[1] = id_apple; // "apple" - should be min lexicographically
        a_vec->values[2] = id_mango; // "mango"
        a_vec->values[3] = id_banana;// "banana"
        SET_ALL_BITS(a_vec->state->selector);
        SET_START_POS(*a_vec->state, 0);
        SET_END_POS(*a_vec->state, 3);

        // Create factorized tree with single attribute
        root = std::make_shared<FactorizedTreeElement>("Name", a_vec);

        // Schema with string attributes
        column_ordering = {"Name"};
        string_attributes = {"Name"};

        schema.root = root;
        schema.column_ordering = &column_ordering;
        schema.map = &map;
        schema.dictionary = string_dict.get();
        schema.string_attributes = &string_attributes;

        // Allocate min_values
        min_values = std::make_unique<uint64_t[]>(1);
        schema.min_values = min_values.get();
        schema.min_values_size = 1;
        schema.required_min_attrs = {"Name"};
    }

    std::unique_ptr<StringPool> string_pool;
    std::unique_ptr<StringDictionary> string_dict;
    QueryVariableToVectorMap map;
    Vector<uint64_t>* a_vec;
    std::shared_ptr<FactorizedTreeElement> root;
    Schema schema;
    std::vector<std::string> column_ordering;
    std::unordered_set<std::string> string_attributes;
    std::unique_ptr<uint64_t[]> min_values;
    uint64_t id_zebra, id_apple, id_mango, id_banana;
};

TEST_F(SinkMinStringTest, FindsLexicographicMinString) {
    SinkMin sink;
    sink.init(&schema);
    sink.execute();

    // Min should be "apple" (id_apple), not "zebra" (id_zebra which has lower numeric ID=0)
    EXPECT_EQ(min_values[0], id_apple);

    // Verify the string is actually "apple"
    const auto& min_str = string_dict->get_string(min_values[0]);
    EXPECT_EQ(min_str.to_string(), "apple");
}

TEST_F(SinkMinStringTest, NumericIdComparisonWouldBeWrong) {
    // This test verifies that without string support, numeric ID comparison would give wrong result
    // "zebra" has ID=0 (lowest numeric), but "apple" (ID=1) is lexicographically smaller
    EXPECT_LT(id_zebra, id_apple);// Numeric: zebra < apple

    // But lexicographically: apple < zebra
    const auto& zebra_str = string_dict->get_string(id_zebra);
    const auto& apple_str = string_dict->get_string(id_apple);
    EXPECT_LT(apple_str, zebra_str);// Lexicographic: apple < zebra
}

//------------------------------------------------------------------------------
// Unit Tests for SinkMinItr with String Values
//------------------------------------------------------------------------------

class SinkMinItrStringTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create string pool and dictionary
        string_pool = std::make_unique<StringPool>();
        string_dict = std::make_unique<StringDictionary>(string_pool.get());

        // Add strings in non-lexicographic order
        id_charlie = string_dict->add_string(ffx_str_t("charlie", string_pool.get()));
        id_alice = string_dict->add_string(ffx_str_t("alice", string_pool.get()));
        id_bob = string_dict->add_string(ffx_str_t("bob", string_pool.get()));

        // Setup vectors
        a_vec = map.allocate_vector("PersonA");
        b_vec = map.allocate_vector("PersonB");

        // A: 3 positions
        a_vec->values[0] = id_charlie;
        a_vec->values[1] = id_alice;
        a_vec->values[2] = id_bob;
        SET_ALL_BITS(a_vec->state->selector);
        SET_START_POS(*a_vec->state, 0);
        SET_END_POS(*a_vec->state, 2);

        // B: 3 positions, 1:1 mapping with A
        b_vec->values[0] = id_bob;
        b_vec->values[1] = id_charlie;
        b_vec->values[2] = id_alice;
        SET_ALL_BITS(b_vec->state->selector);
        SET_START_POS(*b_vec->state, 0);
        SET_END_POS(*b_vec->state, 2);
        b_vec->state->offset[0] = 0;
        b_vec->state->offset[1] = 1;
        b_vec->state->offset[2] = 2;
        b_vec->state->offset[3] = 3;

        // Create factorized tree: A -> B
        root = std::make_shared<FactorizedTreeElement>("PersonA", a_vec);
        root->add_leaf("PersonA", "PersonB", a_vec, b_vec);

        // Schema with both attributes as strings
        column_ordering = {"PersonA", "PersonB"};
        string_attributes = {"PersonA", "PersonB"};

        schema.root = root;
        schema.column_ordering = &column_ordering;
        schema.map = &map;
        schema.dictionary = string_dict.get();
        schema.string_attributes = &string_attributes;

        // Allocate min_values
        min_values = std::make_unique<uint64_t[]>(2);
        schema.min_values = min_values.get();
        schema.min_values_size = 2;
        schema.required_min_attrs = {"PersonA", "PersonB"};
    }

    std::unique_ptr<StringPool> string_pool;
    std::unique_ptr<StringDictionary> string_dict;
    QueryVariableToVectorMap map;
    Vector<uint64_t>* a_vec;
    Vector<uint64_t>* b_vec;
    std::shared_ptr<FactorizedTreeElement> root;
    Schema schema;
    std::vector<std::string> column_ordering;
    std::unordered_set<std::string> string_attributes;
    std::unique_ptr<uint64_t[]> min_values;
    uint64_t id_alice, id_bob, id_charlie;
};

TEST_F(SinkMinItrStringTest, FindsLexicographicMinStringsViaIterator) {
    SinkMinItr sink;
    sink.init(&schema);
    sink.execute();

    // PersonA min should be "alice"
    const auto& min_a_str = string_dict->get_string(min_values[0]);
    EXPECT_EQ(min_a_str.to_string(), "alice");

    // PersonB min should be "alice" (appears at position 2)
    const auto& min_b_str = string_dict->get_string(min_values[1]);
    EXPECT_EQ(min_b_str.to_string(), "alice");
}

//------------------------------------------------------------------------------
// Integration Tests: Mixed Integer and String Columns
//------------------------------------------------------------------------------

class SinkMinMixedTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create string pool and dictionary
        string_pool = std::make_unique<StringPool>();
        string_dict = std::make_unique<StringDictionary>(string_pool.get());

        // Add strings
        id_zulu = string_dict->add_string(ffx_str_t("zulu", string_pool.get()));
        id_alpha = string_dict->add_string(ffx_str_t("alpha", string_pool.get()));
        id_bravo = string_dict->add_string(ffx_str_t("bravo", string_pool.get()));

        // Setup vectors
        age_vec = map.allocate_vector("Age");  // Integer column
        name_vec = map.allocate_vector("Name");// String column

        // Age: 3 numeric values
        age_vec->values[0] = 30;
        age_vec->values[1] = 25;
        age_vec->values[2] = 35;
        SET_ALL_BITS(age_vec->state->selector);
        SET_START_POS(*age_vec->state, 0);
        SET_END_POS(*age_vec->state, 2);

        // Name: 3 string IDs, 1:1 mapping with Age
        name_vec->values[0] = id_zulu;
        name_vec->values[1] = id_alpha;
        name_vec->values[2] = id_bravo;
        SET_ALL_BITS(name_vec->state->selector);
        SET_START_POS(*name_vec->state, 0);
        SET_END_POS(*name_vec->state, 2);
        name_vec->state->offset[0] = 0;
        name_vec->state->offset[1] = 1;
        name_vec->state->offset[2] = 2;
        name_vec->state->offset[3] = 3;

        // Create factorized tree: Age -> Name
        root = std::make_shared<FactorizedTreeElement>("Age", age_vec);
        root->add_leaf("Age", "Name", age_vec, name_vec);

        // Schema with only Name as string attribute
        column_ordering = {"Age", "Name"};
        string_attributes = {"Name"};// Only Name is a string

        schema.root = root;
        schema.column_ordering = &column_ordering;
        schema.map = &map;
        schema.dictionary = string_dict.get();
        schema.string_attributes = &string_attributes;

        // Allocate min_values
        min_values = std::make_unique<uint64_t[]>(2);
        schema.min_values = min_values.get();
        schema.min_values_size = 2;
        schema.required_min_attrs = {"Age", "Name"};
    }

    std::unique_ptr<StringPool> string_pool;
    std::unique_ptr<StringDictionary> string_dict;
    QueryVariableToVectorMap map;
    Vector<uint64_t>* age_vec;
    Vector<uint64_t>* name_vec;
    std::shared_ptr<FactorizedTreeElement> root;
    Schema schema;
    std::vector<std::string> column_ordering;
    std::unordered_set<std::string> string_attributes;
    std::unique_ptr<uint64_t[]> min_values;
    uint64_t id_zulu, id_alpha, id_bravo;
};

TEST_F(SinkMinMixedTest, FindsCorrectMinsForMixedTypes) {
    SinkMin sink;
    sink.init(&schema);
    sink.execute();

    // Age (integer) min should be 25
    EXPECT_EQ(min_values[0], 25);

    // Name (string) min should be "alpha"
    const auto& min_name_str = string_dict->get_string(min_values[1]);
    EXPECT_EQ(min_name_str.to_string(), "alpha");
}

TEST_F(SinkMinMixedTest, SinkMinItrAlsoHandlesMixedTypes) {
    SinkMinItr sink;
    sink.init(&schema);
    sink.execute();

    // Age (integer) min should be 25
    EXPECT_EQ(min_values[0], 25);

    // Name (string) min should be "alpha"
    const auto& min_name_str = string_dict->get_string(min_values[1]);
    EXPECT_EQ(min_name_str.to_string(), "alpha");
}

//------------------------------------------------------------------------------
// Edge Case Tests
//------------------------------------------------------------------------------

class SinkMinEdgeCaseTest : public ::testing::Test {
protected:
    void SetUp() override {
        string_pool = std::make_unique<StringPool>();
        string_dict = std::make_unique<StringDictionary>(string_pool.get());
    }

    std::unique_ptr<StringPool> string_pool;
    std::unique_ptr<StringDictionary> string_dict;
    QueryVariableToVectorMap map;
};

TEST_F(SinkMinEdgeCaseTest, SingleValueString) {
    // Single string value test
    uint64_t id_only = string_dict->add_string(ffx_str_t("only_value", string_pool.get()));

    auto a_vec = map.allocate_vector("SingleAttr");
    a_vec->values[0] = id_only;
    SET_ALL_BITS(a_vec->state->selector);
    SET_START_POS(*a_vec->state, 0);
    SET_END_POS(*a_vec->state, 0);

    auto root = std::make_shared<FactorizedTreeElement>("SingleAttr", a_vec);

    std::vector<std::string> column_ordering = {"SingleAttr"};
    std::unordered_set<std::string> string_attributes = {"SingleAttr"};

    Schema schema;
    schema.root = root;
    schema.column_ordering = &column_ordering;
    schema.map = &map;
    schema.dictionary = string_dict.get();
    schema.string_attributes = &string_attributes;

    auto min_values = std::make_unique<uint64_t[]>(1);
    schema.min_values = min_values.get();
    schema.min_values_size = 1;
    schema.required_min_attrs = {"SingleAttr"};

    SinkMin sink;
    sink.init(&schema);
    sink.execute();

    EXPECT_EQ(min_values[0], id_only);
}

TEST_F(SinkMinEdgeCaseTest, IdenticalStrings) {
    // All identical strings - any ID should be valid as min
    uint64_t id_same1 = string_dict->add_string(ffx_str_t("same", string_pool.get()));
    // Note: add_string should return same ID for identical string
    uint64_t id_same2 = string_dict->add_string(ffx_str_t("same", string_pool.get()));
    EXPECT_EQ(id_same1, id_same2);// Dictionary should deduplicate

    auto a_vec = map.allocate_vector("SameAttr");
    a_vec->values[0] = id_same1;
    a_vec->values[1] = id_same1;
    a_vec->values[2] = id_same1;
    SET_ALL_BITS(a_vec->state->selector);
    SET_START_POS(*a_vec->state, 0);
    SET_END_POS(*a_vec->state, 2);

    auto root = std::make_shared<FactorizedTreeElement>("SameAttr", a_vec);

    std::vector<std::string> column_ordering = {"SameAttr"};
    std::unordered_set<std::string> string_attributes = {"SameAttr"};

    Schema schema;
    schema.root = root;
    schema.column_ordering = &column_ordering;
    schema.map = &map;
    schema.dictionary = string_dict.get();
    schema.string_attributes = &string_attributes;

    auto min_values = std::make_unique<uint64_t[]>(1);
    schema.min_values = min_values.get();
    schema.min_values_size = 1;
    schema.required_min_attrs = {"SameAttr"};

    SinkMin sink;
    sink.init(&schema);
    sink.execute();

    // Min should be the same ID
    EXPECT_EQ(min_values[0], id_same1);
}

//------------------------------------------------------------------------------
// End-to-End Test: Scan -> Join -> Join -> SinkMin vs SinkMinItr
// One of the join attributes (City) is a string
//------------------------------------------------------------------------------

class SinkMinE2ETest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create string pool and dictionary
        string_pool = std::make_unique<StringPool>();
        string_dict = std::make_unique<StringDictionary>(string_pool.get());

        // Add city strings (in non-lexicographic order of IDs)
        // "NewYork"=0, "Atlanta"=1, "Boston"=2, "Chicago"=3
        id_newyork = string_dict->add_string(ffx_str_t("NewYork", string_pool.get()));
        id_atlanta = string_dict->add_string(ffx_str_t("Atlanta", string_pool.get()));
        id_boston = string_dict->add_string(ffx_str_t("Boston", string_pool.get()));
        id_chicago = string_dict->add_string(ffx_str_t("Chicago", string_pool.get()));

        // Create adjacency lists for Person -> City relationship
        person_to_city_adj = std::make_unique<AdjList<uint64_t>[]>(4);
        for (uint64_t i = 0; i < 4; i++) {
            person_to_city_adj[i].size = 0;
            person_to_city_adj[i].values = nullptr;
        }
        person_to_city_adj[0].size = 1;
        person_to_city_adj[0].values = new uint64_t[1]{id_newyork};
        person_to_city_adj[1].size = 1;
        person_to_city_adj[1].values = new uint64_t[1]{id_atlanta};
        person_to_city_adj[2].size = 1;
        person_to_city_adj[2].values = new uint64_t[1]{id_boston};
        person_to_city_adj[3].size = 1;
        person_to_city_adj[3].values = new uint64_t[1]{id_chicago};

        // Create adjacency lists for City -> Country relationship
        city_to_country_adj = std::make_unique<AdjList<uint64_t>[]>(4);
        for (uint64_t i = 0; i < 4; i++) {
            city_to_country_adj[i].size = 0;
            city_to_country_adj[i].values = nullptr;
        }
        city_to_country_adj[id_newyork].size = 1;
        city_to_country_adj[id_newyork].values = new uint64_t[1]{100};
        city_to_country_adj[id_atlanta].size = 1;
        city_to_country_adj[id_atlanta].values = new uint64_t[1]{101};
        city_to_country_adj[id_boston].size = 1;
        city_to_country_adj[id_boston].values = new uint64_t[1]{102};
        city_to_country_adj[id_chicago].size = 1;
        city_to_country_adj[id_chicago].values = new uint64_t[1]{103};

        // Create table for scan
        auto fwd_adj = std::make_unique<AdjList<uint64_t>[]>(4);
        auto bwd_adj = std::make_unique<AdjList<uint64_t>[]>(4);
        for (uint64_t i = 0; i < 4; i++) {
            fwd_adj[i].size = 1;
            fwd_adj[i].values = new uint64_t[1]{i};
            bwd_adj[i].size = 0;
            bwd_adj[i].values = nullptr;
        }

        table = std::make_unique<Table>(4, 4, std::move(fwd_adj), std::move(bwd_adj));
        table->name = "PersonCityCountry";
        table->columns = {"Person", "City", "Country"};
    }

    void TearDown() override {
        for (uint64_t i = 0; i < 4; i++) {
            if (person_to_city_adj && person_to_city_adj[i].values) delete[] person_to_city_adj[i].values;
            if (city_to_country_adj && city_to_country_adj[i].values) delete[] city_to_country_adj[i].values;
            if (table && table->fwd_adj_lists && table->fwd_adj_lists[i].values)
                delete[] table->fwd_adj_lists[i].values;
        }
    }

    // Helper to run pipeline with SinkMin
    uint64_t runWithSinkMin() {
        QueryVariableToVectorMap local_map;
        Schema local_schema;
        std::vector<std::string> local_col_ordering = {"Person", "City", "Country"};
        std::unordered_set<std::string> local_string_attrs = {"City"};

        auto min_vals = std::make_unique<uint64_t[]>(3);
        for (size_t i = 0; i < 3; i++) {
            min_vals[i] = std::numeric_limits<uint64_t>::max();
        }

        local_schema.map = &local_map;
        local_schema.dictionary = string_dict.get();
        local_schema.string_attributes = &local_string_attrs;
        local_schema.column_ordering = &local_col_ordering;
        local_schema.tables.push_back(table.get());
        local_schema.min_values = min_vals.get();
        local_schema.min_values_size = 3;
        local_schema.required_min_attrs = {"Person", "City", "Country"};

        local_schema.register_adj_list("Person", "City", person_to_city_adj.get(), 4);
        local_schema.register_adj_list("City", "Country", city_to_country_adj.get(), 4);

        // Create root before init - operators will add leaves during init
        auto root = std::make_shared<FactorizedTreeElement>("Person", nullptr);
        local_schema.root = root;

        // Create pipeline - operators will allocate vectors during init
        auto scan_op = std::make_unique<Scan<uint64_t>>("Person");
        auto join1_op = std::make_unique<INLJoinPacked<uint64_t>>("Person", "City", true);
        auto join2_op = std::make_unique<INLJoinPackedCascade<uint64_t>>("City", "Country", true);
        auto sink_op = std::make_unique<SinkMin>();

        scan_op->set_next_operator(std::move(join1_op));
        scan_op->next_op->set_next_operator(std::move(join2_op));
        scan_op->next_op->next_op->set_next_operator(std::move(sink_op));

        scan_op->init(&local_schema);
        scan_op->execute();

        // Return City min value (the string attribute)
        return min_vals[1];
    }

    // Helper to run pipeline with SinkMinItr
    uint64_t runWithSinkMinItr() {
        QueryVariableToVectorMap local_map;
        Schema local_schema;
        std::vector<std::string> local_col_ordering = {"Person", "City", "Country"};
        std::unordered_set<std::string> local_string_attrs = {"City"};

        auto min_vals = std::make_unique<uint64_t[]>(3);
        for (size_t i = 0; i < 3; i++) {
            min_vals[i] = std::numeric_limits<uint64_t>::max();
        }

        local_schema.map = &local_map;
        local_schema.dictionary = string_dict.get();
        local_schema.string_attributes = &local_string_attrs;
        local_schema.column_ordering = &local_col_ordering;
        local_schema.tables.push_back(table.get());
        local_schema.min_values = min_vals.get();
        local_schema.min_values_size = 3;
        local_schema.required_min_attrs = {"Person", "City", "Country"};

        local_schema.register_adj_list("Person", "City", person_to_city_adj.get(), 4);
        local_schema.register_adj_list("City", "Country", city_to_country_adj.get(), 4);

        // Create root before init - operators will add leaves during init
        auto root = std::make_shared<FactorizedTreeElement>("Person", nullptr);
        local_schema.root = root;

        // Create pipeline - operators will allocate vectors during init
        auto scan_op = std::make_unique<Scan<uint64_t>>("Person");
        auto join1_op = std::make_unique<INLJoinPacked<uint64_t>>("Person", "City", true);
        auto join2_op = std::make_unique<INLJoinPackedCascade<uint64_t>>("City", "Country", true);
        auto sink_op = std::make_unique<SinkMinItr>();

        scan_op->set_next_operator(std::move(join1_op));
        scan_op->next_op->set_next_operator(std::move(join2_op));
        scan_op->next_op->next_op->set_next_operator(std::move(sink_op));

        scan_op->init(&local_schema);
        scan_op->execute();

        // Return City min value (the string attribute)
        return min_vals[1];
    }

    std::unique_ptr<StringPool> string_pool;
    std::unique_ptr<StringDictionary> string_dict;
    std::unique_ptr<AdjList<uint64_t>[]> person_to_city_adj;
    std::unique_ptr<AdjList<uint64_t>[]> city_to_country_adj;
    std::unique_ptr<Table> table;

    uint64_t id_newyork, id_atlanta, id_boston, id_chicago;
};

TEST_F(SinkMinE2ETest, SinkMinAndSinkMinItrProduceSameResults) {
    // Run with SinkMin
    uint64_t city_min_from_sink_min = runWithSinkMin();

    // Run with SinkMinItr
    uint64_t city_min_from_sink_min_itr = runWithSinkMinItr();

    // Both should produce the same result
    EXPECT_EQ(city_min_from_sink_min, city_min_from_sink_min_itr);

    // City min should be "Atlanta" (lexicographically smallest city name)
    EXPECT_EQ(city_min_from_sink_min, id_atlanta);

    // Verify string content
    const auto& city_str = string_dict->get_string(city_min_from_sink_min);
    EXPECT_EQ(city_str.to_string(), "Atlanta");
}

TEST_F(SinkMinE2ETest, StringMinIsLexicographicNotNumeric) {
    // Run pipeline
    uint64_t city_min = runWithSinkMin();

    // Verify lexicographic vs numeric ordering
    // NewYork has ID 0 (lowest numeric), but is NOT the min
    EXPECT_LT(id_newyork, id_atlanta);// Numeric: NewYork < Atlanta

    // The actual min should be Atlanta (lexicographically smallest)
    EXPECT_EQ(city_min, id_atlanta);
    EXPECT_NE(city_min, id_newyork);// NOT the numerically smallest

    // Verify the string
    EXPECT_EQ(string_dict->get_string(city_min).to_string(), "Atlanta");
}

//------------------------------------------------------------------------------
// Multithreaded End-to-End Test: Multiple threads running Scan -> Join -> SinkMin
//------------------------------------------------------------------------------

class SinkMinMTTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create string pool and dictionary
        string_pool = std::make_unique<StringPool>();
        string_dict = std::make_unique<StringDictionary>(string_pool.get());

        // Add city strings (8 cities for MT distribution)
        // Lexicographic order: Atlanta, Boston, Chicago, Denver, Houston, Miami, NewYork, Seattle
        id_newyork = string_dict->add_string(ffx_str_t("NewYork", string_pool.get()));// ID 0
        id_atlanta = string_dict->add_string(ffx_str_t("Atlanta", string_pool.get()));// ID 1
        id_boston = string_dict->add_string(ffx_str_t("Boston", string_pool.get()));  // ID 2
        id_chicago = string_dict->add_string(ffx_str_t("Chicago", string_pool.get()));// ID 3
        id_denver = string_dict->add_string(ffx_str_t("Denver", string_pool.get()));  // ID 4
        id_houston = string_dict->add_string(ffx_str_t("Houston", string_pool.get()));// ID 5
        id_miami = string_dict->add_string(ffx_str_t("Miami", string_pool.get()));    // ID 6
        id_seattle = string_dict->add_string(ffx_str_t("Seattle", string_pool.get()));// ID 7

        // Create adjacency lists for Person -> City (8 persons, each to a different city)
        person_to_city_adj = std::make_unique<AdjList<uint64_t>[]>(8);
        for (uint64_t i = 0; i < 8; i++) {
            person_to_city_adj[i].size = 0;
            person_to_city_adj[i].values = nullptr;
        }
        person_to_city_adj[0].size = 1;
        person_to_city_adj[0].values = new uint64_t[1]{id_newyork};
        person_to_city_adj[1].size = 1;
        person_to_city_adj[1].values = new uint64_t[1]{id_atlanta};
        person_to_city_adj[2].size = 1;
        person_to_city_adj[2].values = new uint64_t[1]{id_boston};
        person_to_city_adj[3].size = 1;
        person_to_city_adj[3].values = new uint64_t[1]{id_chicago};
        person_to_city_adj[4].size = 1;
        person_to_city_adj[4].values = new uint64_t[1]{id_denver};
        person_to_city_adj[5].size = 1;
        person_to_city_adj[5].values = new uint64_t[1]{id_houston};
        person_to_city_adj[6].size = 1;
        person_to_city_adj[6].values = new uint64_t[1]{id_miami};
        person_to_city_adj[7].size = 1;
        person_to_city_adj[7].values = new uint64_t[1]{id_seattle};

        // Create table for scan
        auto fwd_adj = std::make_unique<AdjList<uint64_t>[]>(8);
        auto bwd_adj = std::make_unique<AdjList<uint64_t>[]>(8);
        for (uint64_t i = 0; i < 8; i++) {
            fwd_adj[i].size = 1;
            fwd_adj[i].values = new uint64_t[1]{i};
            bwd_adj[i].size = 0;
            bwd_adj[i].values = nullptr;
        }

        table = std::make_unique<Table>(8, 8, std::move(fwd_adj), std::move(bwd_adj));
        table->name = "PersonCity";
        table->columns = {"Person", "City"};
    }

    void TearDown() override {
        for (uint64_t i = 0; i < 8; i++) {
            if (person_to_city_adj && person_to_city_adj[i].values) delete[] person_to_city_adj[i].values;
            if (table && table->fwd_adj_lists && table->fwd_adj_lists[i].values)
                delete[] table->fwd_adj_lists[i].values;
        }
    }

    std::unique_ptr<StringPool> string_pool;
    std::unique_ptr<StringDictionary> string_dict;
    std::unique_ptr<AdjList<uint64_t>[]> person_to_city_adj;
    std::unique_ptr<Table> table;

    uint64_t id_newyork, id_atlanta, id_boston, id_chicago;
    uint64_t id_denver, id_houston, id_miami, id_seattle;
};

TEST_F(SinkMinMTTest, MultithreadedSinkMinProducesCorrectResults) {
    /**
     * Test that multiple threads running SinkMin independently produce correct results.
     * Each thread processes a portion of the scan range and computes local min values.
     * We then aggregate the results to verify global correctness.
     */
    const size_t num_threads = 4;
    const uint64_t total_persons = 8;
    const uint64_t persons_per_thread = total_persons / num_threads;

    // Shared min value (atomic for thread-safe aggregation)
    std::atomic<uint64_t> global_city_min{std::numeric_limits<uint64_t>::max()};
    std::mutex dict_mutex;// Protect dictionary access for comparison

    std::vector<std::thread> workers;
    std::vector<uint64_t> thread_local_mins(num_threads, std::numeric_limits<uint64_t>::max());

    for (size_t t = 0; t < num_threads; t++) {
        workers.emplace_back([&, t]() {
            // Each thread has its own local schema and map
            QueryVariableToVectorMap local_map;
            Schema local_schema;
            std::vector<std::string> local_col_ordering = {"Person", "City"};
            std::unordered_set<std::string> local_string_attrs = {"City"};

            auto min_vals = std::make_unique<uint64_t[]>(2);
            min_vals[0] = std::numeric_limits<uint64_t>::max();
            min_vals[1] = std::numeric_limits<uint64_t>::max();

            local_schema.map = &local_map;
            local_schema.dictionary = string_dict.get();// Shared dictionary
            local_schema.string_attributes = &local_string_attrs;
            local_schema.column_ordering = &local_col_ordering;
            local_schema.tables.push_back(table.get());// Shared table (read-only)
            local_schema.min_values = min_vals.get();
            local_schema.min_values_size = 2;
            local_schema.required_min_attrs = {"Person", "City"};

            local_schema.register_adj_list("Person", "City", person_to_city_adj.get(), 8);

            // Create root before init
            auto root = std::make_shared<FactorizedTreeElement>("Person", nullptr);
            local_schema.root = root;

            // Calculate morsel range for this thread
            uint64_t start_id = t * persons_per_thread;
            uint64_t end_id = (t + 1) * persons_per_thread - 1;

            // Create pipeline with ScanSynchronized
            auto scan_op = std::make_unique<ScanSynchronized>("Person", start_id);
            scan_op->set_max_id(end_id);

            auto join_op = std::make_unique<INLJoinPacked<uint64_t>>("Person", "City", true);
            auto sink_op = std::make_unique<SinkMin>();

            scan_op->set_next_operator(std::move(join_op));
            scan_op->next_op->set_next_operator(std::move(sink_op));

            scan_op->init(&local_schema);
            scan_op->execute();

            // Store local min
            thread_local_mins[t] = min_vals[1];// City min

            // Atomically update global min using string comparison
            uint64_t local_min = min_vals[1];
            if (local_min < std::numeric_limits<uint64_t>::max()) {
                uint64_t expected = global_city_min.load();
                while (true) {
                    if (expected >= std::numeric_limits<uint64_t>::max()) {
                        // First valid update
                        if (global_city_min.compare_exchange_weak(expected, local_min)) break;
                    } else {
                        // Compare strings
                        std::lock_guard<std::mutex> lock(dict_mutex);
                        const auto& local_str = string_dict->get_string(local_min);
                        const auto& global_str = string_dict->get_string(expected);
                        if (local_str.to_string() < global_str.to_string()) {
                            if (global_city_min.compare_exchange_weak(expected, local_min)) break;
                        } else {
                            break;// Current global is smaller or equal
                        }
                    }
                }
            }
        });
    }

    // Wait for all threads
    for (auto& w: workers) {
        w.join();
    }

    // Verify: The global min should be "Atlanta" (lexicographically smallest)
    uint64_t final_min = global_city_min.load();
    EXPECT_EQ(final_min, id_atlanta);

    const auto& city_str = string_dict->get_string(final_min);
    EXPECT_EQ(city_str.to_string(), "Atlanta");
}

TEST_F(SinkMinMTTest, SingleThreadedMatchesMultithreaded) {
    /**
     * Verify that single-threaded result matches the expected min
     * (sanity check for MT test correctness).
     */
    QueryVariableToVectorMap local_map;
    Schema local_schema;
    std::vector<std::string> local_col_ordering = {"Person", "City"};
    std::unordered_set<std::string> local_string_attrs = {"City"};

    auto min_vals = std::make_unique<uint64_t[]>(2);
    min_vals[0] = std::numeric_limits<uint64_t>::max();
    min_vals[1] = std::numeric_limits<uint64_t>::max();

    local_schema.map = &local_map;
    local_schema.dictionary = string_dict.get();
    local_schema.string_attributes = &local_string_attrs;
    local_schema.column_ordering = &local_col_ordering;
    local_schema.tables.push_back(table.get());
    local_schema.min_values = min_vals.get();
    local_schema.min_values_size = 2;
    local_schema.required_min_attrs = {"Person", "City"};

    local_schema.register_adj_list("Person", "City", person_to_city_adj.get(), 8);

    auto root = std::make_shared<FactorizedTreeElement>("Person", nullptr);
    local_schema.root = root;

    // Full scan (single thread)
    auto scan_op = std::make_unique<Scan<uint64_t>>("Person");
    auto join_op = std::make_unique<INLJoinPacked<uint64_t>>("Person", "City", true);
    auto sink_op = std::make_unique<SinkMin>();

    scan_op->set_next_operator(std::move(join_op));
    scan_op->next_op->set_next_operator(std::move(sink_op));

    scan_op->init(&local_schema);
    scan_op->execute();

    // Verify single-threaded min is also Atlanta
    EXPECT_EQ(min_vals[1], id_atlanta);
    EXPECT_EQ(string_dict->get_string(min_vals[1]).to_string(), "Atlanta");
}
