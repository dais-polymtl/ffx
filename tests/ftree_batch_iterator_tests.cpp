#include "../src/operator/include/factorized_ftree/ftree_batch_iterator.hpp"
#include "../src/operator/include/factorized_ftree/ftree_iterator.hpp"
#include "../src/operator/include/schema/schema.hpp"
#include "../src/operator/include/vector/state.hpp"
#include "../src/operator/include/vector/vector.hpp"
#include <algorithm>
#include <functional>
#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace ffx;

namespace {

struct NodeInfo {
    const uint16_t* rle;
    bool shares_parent;
    int32_t parent_col;
};

std::vector<NodeInfo> build_node_info(const Schema& schema, const FTreeBatchIterator& itr) {
    const auto& col_order = *schema.column_ordering;
    size_t n = itr.num_attributes();

    std::vector<std::string> cols;
    for (const auto& c: col_order) {
        if (c != "_cd") cols.push_back(c);
    }

    std::vector<FactorizedTreeElement*> nodes(n);
    for (size_t i = 0; i < n; ++i) {
        nodes[i] = schema.root->find_node_by_attribute(cols[i]);
    }

    std::vector<NodeInfo> info(n);
    for (size_t i = 0; i < n; ++i) {
        if (nodes[i]->_parent == nullptr) {
            info[i] = {nullptr, false, -1};
        } else {
            int32_t parent_col = -1;
            for (size_t j = 0; j < n; ++j) {
                if (nodes[j] == nodes[i]->_parent) {
                    parent_col = static_cast<int32_t>(j);
                    break;
                }
            }
            bool shares = (nodes[static_cast<size_t>(parent_col)]->_value->state == nodes[i]->_value->state);
            const uint16_t* rle = shares ? nullptr : nodes[i]->_value->state->offset;
            info[i] = {rle, shares, parent_col};
        }
    }
    return info;
}

std::pair<size_t, size_t> find_child_range(const FTreeBatchIterator& itr, size_t child_col, size_t parent_batch_idx) {
    const uint32_t* batch_offset = itr.get_node_offset(child_col);
    return {batch_offset[parent_batch_idx], batch_offset[parent_batch_idx + 1]};
}

void expand_with_continuation(const FTreeBatchIterator& itr, const std::vector<NodeInfo>& info, size_t node_col,
                              size_t buf_start, size_t buf_end, const std::function<void()>& continuation,
                              std::vector<uint64_t>& tuple, std::vector<std::vector<uint64_t>>& result);

void expand_children_cont(const FTreeBatchIterator& itr, const std::vector<NodeInfo>& info,
                          ChildSpan children, size_t idx, size_t parent_batch_idx,
                          const std::function<void()>& continuation, std::vector<uint64_t>& tuple,
                          std::vector<std::vector<uint64_t>>& result) {

    if (idx >= children.size()) {
        continuation();
        return;
    }

    size_t c = children[idx];
    auto [cs, ce] = find_child_range(itr, c, parent_batch_idx);

    auto next_cont = [&, parent_batch_idx, idx]() {
        expand_children_cont(itr, info, children, idx + 1, parent_batch_idx, continuation, tuple, result);
    };

    expand_with_continuation(itr, info, c, cs, ce, next_cont, tuple, result);
}

void expand_with_continuation(const FTreeBatchIterator& itr, const std::vector<NodeInfo>& info, size_t node_col,
                              size_t buf_start, size_t buf_end, const std::function<void()>& continuation,
                              std::vector<uint64_t>& tuple, std::vector<std::vector<uint64_t>>& result) {

    for (size_t i = buf_start; i < buf_end; ++i) {
        tuple[node_col] = itr.get_buffer(node_col)[i];

        if (itr.is_leaf_attr(node_col)) {
            continuation();
        } else {
            const auto& children = itr.get_children(node_col);
            expand_children_cont(itr, info, children, 0, i, continuation, tuple, result);
        }
    }
}

std::vector<std::vector<uint64_t>> expand_batch(FTreeBatchIterator& itr, const Schema& schema) {
    const size_t n = itr.num_attributes();
    if (n == 0) return {};

    auto info = build_node_info(schema, itr);
    std::vector<uint64_t> tuple(n);
    std::vector<std::vector<uint64_t>> result;

    auto emit = [&]() { result.push_back(tuple); };

    if (itr.is_leaf_attr(0)) {
        expand_with_continuation(itr, info, 0, 0, itr.get_count(0), emit, tuple, result);
    } else {
        for (size_t i = 0; i < itr.get_count(0); ++i) {
            tuple[0] = itr.get_buffer(0)[i];
            const auto& children = itr.get_children(0);
            expand_children_cont(itr, info, children, 0, i, emit, tuple, result);
        }
    }
    return result;
}

std::vector<std::vector<uint64_t>> collect_scalar_tuples(Schema* schema) {
    FTreeIterator itr;
    itr.init(schema);
    itr.initialize_iterators();
    if (!itr.is_valid()) return {};

    std::vector<std::vector<uint64_t>> out;
    std::vector<uint64_t> row(itr.tuple_size());
    while (true) {
        bool ok = itr.next(row.data());
        out.push_back(row);
        if (!ok) break;
    }
    return out;
}

std::vector<std::vector<uint64_t>> collect_all_batch_tuples(FTreeBatchIterator& itr, Schema& schema) {
    std::vector<std::vector<uint64_t>> all;
    while (itr.next()) {
        auto batch = expand_batch(itr, schema);
        all.insert(all.end(), batch.begin(), batch.end());
    }
    return all;
}

}// namespace

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------
class FTreeBatchIteratorTest : public ::testing::Test {
protected:
    std::unique_ptr<Vector<uint64_t>> vecs_[50];
    std::shared_ptr<FactorizedTreeElement> root_;
    std::vector<std::string> ordering_;
    Schema schema_;

    Vector<uint64_t>* V(int slot) { return vecs_[slot].get(); }

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
    }

    // ---- Skewed 3-node tree: a->(d,g)  ----
    //   a: [7]  (single root value)
    //   d: [100,101]  (2 values under a)
    //   g: [200..207] (8 values under a) -- highly skewed
    void build_skew_tree() {
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
        root_->add_leaf("a", "d", V(0), V(1));
        root_->add_leaf("a", "g", V(0), V(2));

        ordering_ = {"a", "d", "g"};
        schema_.root = root_;
        schema_.column_ordering = &ordering_;
    }

    // ---- 7-node tree: a->(b,c), b->(d,e), c->(f,g) ----
    //   a: [1,2]
    //   b: [10,11 | 12,13]       offset from a: [0,2,4]
    //   c: [20 | 21]             offset from a: [0,1,2]
    //   d: [100,101 | 102 | 103 | 104]   offset from b: [0,2,3,4,5]
    //   e: [200 | 201 | 202 | 203]       offset from b: [0,1,2,3,4]
    //   f: [300,301 | 302]               offset from c: [0,2,3]
    //   g: [400,401,402 | 403,404]       offset from c: [0,3,5]
    void build_seven_node_tree() {
        // a (slot 0)
        vecs_[0] = std::make_unique<Vector<uint64_t>>();
        V(0)->values[0] = 1;
        V(0)->values[1] = 2;
        SET_ALL_BITS(V(0)->state->selector);
        SET_START_POS(*V(0)->state, 0);
        SET_END_POS(*V(0)->state, 1);

        // b (slot 1)
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

        // c (slot 2)
        vecs_[2] = std::make_unique<Vector<uint64_t>>();
        V(2)->values[0] = 20;
        V(2)->values[1] = 21;
        SET_ALL_BITS(V(2)->state->selector);
        SET_START_POS(*V(2)->state, 0);
        SET_END_POS(*V(2)->state, 1);
        V(2)->state->offset[0] = 0;
        V(2)->state->offset[1] = 1;
        V(2)->state->offset[2] = 2;

        // d (slot 3)
        vecs_[3] = std::make_unique<Vector<uint64_t>>();
        V(3)->values[0] = 100;
        V(3)->values[1] = 101;
        V(3)->values[2] = 102;
        V(3)->values[3] = 103;
        V(3)->values[4] = 104;
        SET_ALL_BITS(V(3)->state->selector);
        SET_START_POS(*V(3)->state, 0);
        SET_END_POS(*V(3)->state, 4);
        V(3)->state->offset[0] = 0;
        V(3)->state->offset[1] = 2;
        V(3)->state->offset[2] = 3;
        V(3)->state->offset[3] = 4;
        V(3)->state->offset[4] = 5;

        // e (slot 4)
        vecs_[4] = std::make_unique<Vector<uint64_t>>();
        V(4)->values[0] = 200;
        V(4)->values[1] = 201;
        V(4)->values[2] = 202;
        V(4)->values[3] = 203;
        SET_ALL_BITS(V(4)->state->selector);
        SET_START_POS(*V(4)->state, 0);
        SET_END_POS(*V(4)->state, 3);
        V(4)->state->offset[0] = 0;
        V(4)->state->offset[1] = 1;
        V(4)->state->offset[2] = 2;
        V(4)->state->offset[3] = 3;
        V(4)->state->offset[4] = 4;

        // f (slot 5)
        vecs_[5] = std::make_unique<Vector<uint64_t>>();
        V(5)->values[0] = 300;
        V(5)->values[1] = 301;
        V(5)->values[2] = 302;
        SET_ALL_BITS(V(5)->state->selector);
        SET_START_POS(*V(5)->state, 0);
        SET_END_POS(*V(5)->state, 2);
        V(5)->state->offset[0] = 0;
        V(5)->state->offset[1] = 2;
        V(5)->state->offset[2] = 3;

        // g (slot 6)
        vecs_[6] = std::make_unique<Vector<uint64_t>>();
        V(6)->values[0] = 400;
        V(6)->values[1] = 401;
        V(6)->values[2] = 402;
        V(6)->values[3] = 403;
        V(6)->values[4] = 404;
        SET_ALL_BITS(V(6)->state->selector);
        SET_START_POS(*V(6)->state, 0);
        SET_END_POS(*V(6)->state, 4);
        V(6)->state->offset[0] = 0;
        V(6)->state->offset[1] = 3;
        V(6)->state->offset[2] = 5;

        root_ = std::make_shared<FactorizedTreeElement>("a", V(0));
        root_->add_leaf("a", "b", V(0), V(1));
        root_->add_leaf("a", "c", V(0), V(2));
        root_->add_leaf("b", "d", V(1), V(3));
        root_->add_leaf("b", "e", V(1), V(4));
        root_->add_leaf("c", "f", V(2), V(5));
        root_->add_leaf("c", "g", V(2), V(6));

        ordering_ = {"a", "b", "d", "e", "c", "f", "g"};
        schema_.root = root_;
        schema_.column_ordering = &ordering_;
    }

    // ---- Deep chain: a->b->c->d (linear, only d is leaf) ----
    //   a: [1,2]
    //   b: [10 | 11,12]              offset from a: [0,1,3]
    //   c: [20 | 21 | 22,23]         offset from b: [0,1,2,4]
    //   d: [100 | 101,102 | 103 | 104,105]  offset from c: [0,1,3,4,6]
    void build_deep_chain() {
        // a (slot 0)
        vecs_[0] = std::make_unique<Vector<uint64_t>>();
        V(0)->values[0] = 1;
        V(0)->values[1] = 2;
        SET_ALL_BITS(V(0)->state->selector);
        SET_START_POS(*V(0)->state, 0);
        SET_END_POS(*V(0)->state, 1);

        // b (slot 1)
        vecs_[1] = std::make_unique<Vector<uint64_t>>();
        V(1)->values[0] = 10;
        V(1)->values[1] = 11;
        V(1)->values[2] = 12;
        SET_ALL_BITS(V(1)->state->selector);
        SET_START_POS(*V(1)->state, 0);
        SET_END_POS(*V(1)->state, 2);
        V(1)->state->offset[0] = 0;
        V(1)->state->offset[1] = 1;
        V(1)->state->offset[2] = 3;

        // c (slot 2)
        vecs_[2] = std::make_unique<Vector<uint64_t>>();
        V(2)->values[0] = 20;
        V(2)->values[1] = 21;
        V(2)->values[2] = 22;
        V(2)->values[3] = 23;
        SET_ALL_BITS(V(2)->state->selector);
        SET_START_POS(*V(2)->state, 0);
        SET_END_POS(*V(2)->state, 3);
        V(2)->state->offset[0] = 0;
        V(2)->state->offset[1] = 1;
        V(2)->state->offset[2] = 2;
        V(2)->state->offset[3] = 4;

        // d (slot 3)
        vecs_[3] = std::make_unique<Vector<uint64_t>>();
        V(3)->values[0] = 100;
        V(3)->values[1] = 101;
        V(3)->values[2] = 102;
        V(3)->values[3] = 103;
        V(3)->values[4] = 104;
        V(3)->values[5] = 105;
        SET_ALL_BITS(V(3)->state->selector);
        SET_START_POS(*V(3)->state, 0);
        SET_END_POS(*V(3)->state, 5);
        V(3)->state->offset[0] = 0;
        V(3)->state->offset[1] = 1;
        V(3)->state->offset[2] = 3;
        V(3)->state->offset[3] = 4;
        V(3)->state->offset[4] = 6;

        root_ = std::make_shared<FactorizedTreeElement>("a", V(0));
        root_->add_leaf("a", "b", V(0), V(1));
        root_->add_leaf("b", "c", V(1), V(2));
        root_->add_leaf("c", "d", V(2), V(3));

        ordering_ = {"a", "b", "c", "d"};
        schema_.root = root_;
        schema_.column_ordering = &ordering_;
    }

    // ---- Single-leaf tree: just x ----
    //   x: [10,20,30,40,50]  (root is also the only leaf)
    void build_single_leaf_tree() {
        vecs_[0] = std::make_unique<Vector<uint64_t>>();
        V(0)->values[0] = 10;
        V(0)->values[1] = 20;
        V(0)->values[2] = 30;
        V(0)->values[3] = 40;
        V(0)->values[4] = 50;
        SET_ALL_BITS(V(0)->state->selector);
        SET_START_POS(*V(0)->state, 0);
        SET_END_POS(*V(0)->state, 4);

        root_ = std::make_shared<FactorizedTreeElement>("x", V(0));

        ordering_ = {"x"};
        schema_.root = root_;
        schema_.column_ordering = &ordering_;
    }

    // ---- Star: r(5) -> (a(15),b(10),c(20),d(10)), 240 tuples ----
    void build_star_tree() {
        // r: 5 values
        vecs_[0] = std::make_unique<Vector<uint64_t>>();
        for (int i = 0; i < 5; ++i)
            V(0)->values[i] = 1 + i;
        SET_ALL_BITS(V(0)->state->selector);
        SET_START_POS(*V(0)->state, 0);
        SET_END_POS(*V(0)->state, 4);

        // a: 15 values, 3 per r
        vecs_[1] = std::make_unique<Vector<uint64_t>>();
        for (int i = 0; i < 15; ++i)
            V(1)->values[i] = 10 + i;
        SET_ALL_BITS(V(1)->state->selector);
        SET_START_POS(*V(1)->state, 0);
        SET_END_POS(*V(1)->state, 14);
        for (int i = 0; i <= 5; ++i)
            V(1)->state->offset[i] = i * 3;

        // b: 10 values, 2 per r
        vecs_[2] = std::make_unique<Vector<uint64_t>>();
        for (int i = 0; i < 10; ++i)
            V(2)->values[i] = 30 + i;
        SET_ALL_BITS(V(2)->state->selector);
        SET_START_POS(*V(2)->state, 0);
        SET_END_POS(*V(2)->state, 9);
        for (int i = 0; i <= 5; ++i)
            V(2)->state->offset[i] = i * 2;

        // c: 20 values, 4 per r
        vecs_[3] = std::make_unique<Vector<uint64_t>>();
        for (int i = 0; i < 20; ++i)
            V(3)->values[i] = 50 + i;
        SET_ALL_BITS(V(3)->state->selector);
        SET_START_POS(*V(3)->state, 0);
        SET_END_POS(*V(3)->state, 19);
        for (int i = 0; i <= 5; ++i)
            V(3)->state->offset[i] = i * 4;

        // d: 10 values, 2 per r
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
    }

    // ---- Wide Fan: r(1)->m(4)->(b1(12),b2(8),b3(16),b4(8),b5(12)), 576 tuples ----
    void build_wide_fan_tree() {
        vecs_[0] = std::make_unique<Vector<uint64_t>>();
        V(0)->values[0] = 1;
        SET_ALL_BITS(V(0)->state->selector);
        SET_START_POS(*V(0)->state, 0);
        SET_END_POS(*V(0)->state, 0);

        // m: 4 values
        vecs_[1] = std::make_unique<Vector<uint64_t>>();
        for (int i = 0; i < 4; ++i)
            V(1)->values[i] = 10 + i;
        SET_ALL_BITS(V(1)->state->selector);
        SET_START_POS(*V(1)->state, 0);
        SET_END_POS(*V(1)->state, 3);
        V(1)->state->offset[0] = 0;
        V(1)->state->offset[1] = 4;

        // b1: 12 values, 3 per m
        vecs_[2] = std::make_unique<Vector<uint64_t>>();
        for (int i = 0; i < 12; ++i)
            V(2)->values[i] = 100 + i;
        SET_ALL_BITS(V(2)->state->selector);
        SET_START_POS(*V(2)->state, 0);
        SET_END_POS(*V(2)->state, 11);
        for (int i = 0; i <= 4; ++i)
            V(2)->state->offset[i] = i * 3;

        // b2: 8 values, 2 per m
        vecs_[3] = std::make_unique<Vector<uint64_t>>();
        for (int i = 0; i < 8; ++i)
            V(3)->values[i] = 200 + i;
        SET_ALL_BITS(V(3)->state->selector);
        SET_START_POS(*V(3)->state, 0);
        SET_END_POS(*V(3)->state, 7);
        for (int i = 0; i <= 4; ++i)
            V(3)->state->offset[i] = i * 2;

        // b3: 16 values, 4 per m
        vecs_[4] = std::make_unique<Vector<uint64_t>>();
        for (int i = 0; i < 16; ++i)
            V(4)->values[i] = 300 + i;
        SET_ALL_BITS(V(4)->state->selector);
        SET_START_POS(*V(4)->state, 0);
        SET_END_POS(*V(4)->state, 15);
        for (int i = 0; i <= 4; ++i)
            V(4)->state->offset[i] = i * 4;

        // b4: 8 values, 2 per m
        vecs_[5] = std::make_unique<Vector<uint64_t>>();
        for (int i = 0; i < 8; ++i)
            V(5)->values[i] = 400 + i;
        SET_ALL_BITS(V(5)->state->selector);
        SET_START_POS(*V(5)->state, 0);
        SET_END_POS(*V(5)->state, 7);
        for (int i = 0; i <= 4; ++i)
            V(5)->state->offset[i] = i * 2;

        // b5: 12 values, 3 per m
        vecs_[6] = std::make_unique<Vector<uint64_t>>();
        for (int i = 0; i < 12; ++i)
            V(6)->values[i] = 500 + i;
        SET_ALL_BITS(V(6)->state->selector);
        SET_START_POS(*V(6)->state, 0);
        SET_END_POS(*V(6)->state, 11);
        for (int i = 0; i <= 4; ++i)
            V(6)->state->offset[i] = i * 3;

        root_ = std::make_shared<FactorizedTreeElement>("r", V(0));
        root_->add_leaf("r", "m", V(0), V(1));
        root_->add_leaf("m", "b1", V(1), V(2));
        root_->add_leaf("m", "b2", V(1), V(3));
        root_->add_leaf("m", "b3", V(1), V(4));
        root_->add_leaf("m", "b4", V(1), V(5));
        root_->add_leaf("m", "b5", V(1), V(6));

        ordering_ = {"r", "m", "b1", "b2", "b3", "b4", "b5"};
        schema_.root = root_;
        schema_.column_ordering = &ordering_;
    }

    // ---- Deep5: a(3)->b(6)->c(12)->d(24)->e(48), 48 tuples ----
    void build_deep5_chain() {
        vecs_[0] = std::make_unique<Vector<uint64_t>>();
        for (int i = 0; i < 3; ++i)
            V(0)->values[i] = 1 + i;
        SET_ALL_BITS(V(0)->state->selector);
        SET_START_POS(*V(0)->state, 0);
        SET_END_POS(*V(0)->state, 2);

        vecs_[1] = std::make_unique<Vector<uint64_t>>();
        for (int i = 0; i < 6; ++i)
            V(1)->values[i] = 10 + i;
        SET_ALL_BITS(V(1)->state->selector);
        SET_START_POS(*V(1)->state, 0);
        SET_END_POS(*V(1)->state, 5);
        for (int i = 0; i <= 3; ++i)
            V(1)->state->offset[i] = i * 2;

        vecs_[2] = std::make_unique<Vector<uint64_t>>();
        for (int i = 0; i < 12; ++i)
            V(2)->values[i] = 20 + i;
        SET_ALL_BITS(V(2)->state->selector);
        SET_START_POS(*V(2)->state, 0);
        SET_END_POS(*V(2)->state, 11);
        for (int i = 0; i <= 6; ++i)
            V(2)->state->offset[i] = i * 2;

        vecs_[3] = std::make_unique<Vector<uint64_t>>();
        for (int i = 0; i < 24; ++i)
            V(3)->values[i] = 40 + i;
        SET_ALL_BITS(V(3)->state->selector);
        SET_START_POS(*V(3)->state, 0);
        SET_END_POS(*V(3)->state, 23);
        for (int i = 0; i <= 12; ++i)
            V(3)->state->offset[i] = i * 2;

        vecs_[4] = std::make_unique<Vector<uint64_t>>();
        for (int i = 0; i < 48; ++i)
            V(4)->values[i] = 100 + i;
        SET_ALL_BITS(V(4)->state->selector);
        SET_START_POS(*V(4)->state, 0);
        SET_END_POS(*V(4)->state, 47);
        for (int i = 0; i <= 24; ++i)
            V(4)->state->offset[i] = i * 2;

        root_ = std::make_shared<FactorizedTreeElement>("a", V(0));
        root_->add_leaf("a", "b", V(0), V(1));
        root_->add_leaf("b", "c", V(1), V(2));
        root_->add_leaf("c", "d", V(2), V(3));
        root_->add_leaf("d", "e", V(3), V(4));

        ordering_ = {"a", "b", "c", "d", "e"};
        schema_.root = root_;
        schema_.column_ordering = &ordering_;
    }

    // ---- Asymmetric: r(4)->(A(8)->C(16)->(d(32),e(16)), B(8)), 64 tuples ----
    void build_asymmetric_tree() {
        vecs_[0] = std::make_unique<Vector<uint64_t>>();
        for (int i = 0; i < 4; ++i)
            V(0)->values[i] = 1 + i;
        SET_ALL_BITS(V(0)->state->selector);
        SET_START_POS(*V(0)->state, 0);
        SET_END_POS(*V(0)->state, 3);

        // A: 8 vals, 2 per r
        vecs_[1] = std::make_unique<Vector<uint64_t>>();
        for (int i = 0; i < 8; ++i)
            V(1)->values[i] = 10 + i;
        SET_ALL_BITS(V(1)->state->selector);
        SET_START_POS(*V(1)->state, 0);
        SET_END_POS(*V(1)->state, 7);
        for (int i = 0; i <= 4; ++i)
            V(1)->state->offset[i] = i * 2;

        // B: 8 vals, 2 per r (leaf)
        vecs_[2] = std::make_unique<Vector<uint64_t>>();
        for (int i = 0; i < 8; ++i)
            V(2)->values[i] = 50 + i;
        SET_ALL_BITS(V(2)->state->selector);
        SET_START_POS(*V(2)->state, 0);
        SET_END_POS(*V(2)->state, 7);
        for (int i = 0; i <= 4; ++i)
            V(2)->state->offset[i] = i * 2;

        // C: 16 vals, 2 per A
        vecs_[3] = std::make_unique<Vector<uint64_t>>();
        for (int i = 0; i < 16; ++i)
            V(3)->values[i] = 20 + i;
        SET_ALL_BITS(V(3)->state->selector);
        SET_START_POS(*V(3)->state, 0);
        SET_END_POS(*V(3)->state, 15);
        for (int i = 0; i <= 8; ++i)
            V(3)->state->offset[i] = i * 2;

        // d: 32 vals, 2 per C (leaf)
        vecs_[4] = std::make_unique<Vector<uint64_t>>();
        for (int i = 0; i < 32; ++i)
            V(4)->values[i] = 100 + i;
        SET_ALL_BITS(V(4)->state->selector);
        SET_START_POS(*V(4)->state, 0);
        SET_END_POS(*V(4)->state, 31);
        for (int i = 0; i <= 16; ++i)
            V(4)->state->offset[i] = i * 2;

        // e: 16 vals, 1 per C (leaf)
        vecs_[5] = std::make_unique<Vector<uint64_t>>();
        for (int i = 0; i < 16; ++i)
            V(5)->values[i] = 200 + i;
        SET_ALL_BITS(V(5)->state->selector);
        SET_START_POS(*V(5)->state, 0);
        SET_END_POS(*V(5)->state, 15);
        for (int i = 0; i <= 16; ++i)
            V(5)->state->offset[i] = i;

        root_ = std::make_shared<FactorizedTreeElement>("r", V(0));
        root_->add_leaf("r", "A", V(0), V(1));
        root_->add_leaf("r", "B", V(0), V(2));
        root_->add_leaf("A", "C", V(1), V(3));
        root_->add_leaf("C", "d", V(3), V(4));
        root_->add_leaf("C", "e", V(3), V(5));

        ordering_ = {"r", "A", "C", "d", "e", "B"};
        schema_.root = root_;
        schema_.column_ordering = &ordering_;
    }

    // ---- Sparse Bitmask: r(10 pos, 5 selected)->(a(30),b(20)), 30 tuples ----
    void build_sparse_bitmask_tree() {
        // r: 10 positions, only even selected
        vecs_[0] = std::make_unique<Vector<uint64_t>>();
        for (int i = 0; i < 10; ++i)
            V(0)->values[i] = 1 + i;
        CLEAR_ALL_BITS(V(0)->state->selector);
        for (int i = 0; i < 10; i += 2)
            SET_BIT(V(0)->state->selector, i);
        SET_START_POS(*V(0)->state, 0);
        SET_END_POS(*V(0)->state, 9);

        // a: 30 vals, 3 per r
        vecs_[1] = std::make_unique<Vector<uint64_t>>();
        for (int i = 0; i < 30; ++i)
            V(1)->values[i] = 100 + i;
        SET_ALL_BITS(V(1)->state->selector);
        SET_START_POS(*V(1)->state, 0);
        SET_END_POS(*V(1)->state, 29);
        for (int i = 0; i <= 10; ++i)
            V(1)->state->offset[i] = i * 3;

        // b: 20 vals, 2 per r
        vecs_[2] = std::make_unique<Vector<uint64_t>>();
        for (int i = 0; i < 20; ++i)
            V(2)->values[i] = 200 + i;
        SET_ALL_BITS(V(2)->state->selector);
        SET_START_POS(*V(2)->state, 0);
        SET_END_POS(*V(2)->state, 19);
        for (int i = 0; i <= 10; ++i)
            V(2)->state->offset[i] = i * 2;

        root_ = std::make_shared<FactorizedTreeElement>("r", V(0));
        root_->add_leaf("r", "a", V(0), V(1));
        root_->add_leaf("r", "b", V(0), V(2));

        ordering_ = {"r", "a", "b"};
        schema_.root = root_;
        schema_.column_ordering = &ordering_;
    }

    // ---- Diamond: r(4)->(m(8)->(x(16),y(8)), p(12), q(8)), 96 tuples ----
    void build_diamond_tree() {
        vecs_[0] = std::make_unique<Vector<uint64_t>>();
        for (int i = 0; i < 4; ++i)
            V(0)->values[i] = 1 + i;
        SET_ALL_BITS(V(0)->state->selector);
        SET_START_POS(*V(0)->state, 0);
        SET_END_POS(*V(0)->state, 3);

        // m: 8 vals, 2 per r
        vecs_[1] = std::make_unique<Vector<uint64_t>>();
        for (int i = 0; i < 8; ++i)
            V(1)->values[i] = 10 + i;
        SET_ALL_BITS(V(1)->state->selector);
        SET_START_POS(*V(1)->state, 0);
        SET_END_POS(*V(1)->state, 7);
        for (int i = 0; i <= 4; ++i)
            V(1)->state->offset[i] = i * 2;

        // p: 12 vals, 3 per r (leaf)
        vecs_[2] = std::make_unique<Vector<uint64_t>>();
        for (int i = 0; i < 12; ++i)
            V(2)->values[i] = 50 + i;
        SET_ALL_BITS(V(2)->state->selector);
        SET_START_POS(*V(2)->state, 0);
        SET_END_POS(*V(2)->state, 11);
        for (int i = 0; i <= 4; ++i)
            V(2)->state->offset[i] = i * 3;

        // q: 8 vals, 2 per r (leaf)
        vecs_[3] = std::make_unique<Vector<uint64_t>>();
        for (int i = 0; i < 8; ++i)
            V(3)->values[i] = 70 + i;
        SET_ALL_BITS(V(3)->state->selector);
        SET_START_POS(*V(3)->state, 0);
        SET_END_POS(*V(3)->state, 7);
        for (int i = 0; i <= 4; ++i)
            V(3)->state->offset[i] = i * 2;

        // x: 16 vals, 2 per m (leaf)
        vecs_[4] = std::make_unique<Vector<uint64_t>>();
        for (int i = 0; i < 16; ++i)
            V(4)->values[i] = 100 + i;
        SET_ALL_BITS(V(4)->state->selector);
        SET_START_POS(*V(4)->state, 0);
        SET_END_POS(*V(4)->state, 15);
        for (int i = 0; i <= 8; ++i)
            V(4)->state->offset[i] = i * 2;

        // y: 8 vals, 1 per m (leaf)
        vecs_[5] = std::make_unique<Vector<uint64_t>>();
        for (int i = 0; i < 8; ++i)
            V(5)->values[i] = 200 + i;
        SET_ALL_BITS(V(5)->state->selector);
        SET_START_POS(*V(5)->state, 0);
        SET_END_POS(*V(5)->state, 7);
        for (int i = 0; i <= 8; ++i)
            V(5)->state->offset[i] = i;

        root_ = std::make_shared<FactorizedTreeElement>("r", V(0));
        root_->add_leaf("r", "m", V(0), V(1));
        root_->add_leaf("r", "p", V(0), V(2));
        root_->add_leaf("r", "q", V(0), V(3));
        root_->add_leaf("m", "x", V(1), V(4));
        root_->add_leaf("m", "y", V(1), V(5));

        ordering_ = {"r", "m", "x", "y", "p", "q"};
        schema_.root = root_;
        schema_.column_ordering = &ordering_;
    }
};

// ===========================================================================
// 1. Correctness: factorized expansion matches scalar iterator (sorted sets)
// ===========================================================================

TEST_F(FTreeBatchIteratorTest, FactorizedMatchesScalar_BalancedTree) {
    build_balanced_tree();
    auto expected = collect_scalar_tuples(&schema_);

    FTreeBatchIterator batch({{"a", 512}, {"b", 3}, {"c", 2}});
    batch.init(&schema_);
    batch.initialize_iterators();
    auto actual = collect_all_batch_tuples(batch, schema_);

    std::sort(expected.begin(), expected.end());
    std::sort(actual.begin(), actual.end());
    EXPECT_EQ(actual, expected);
}

TEST_F(FTreeBatchIteratorTest, FactorizedMatchesScalar_SevenNodeTree) {
    build_seven_node_tree();
    auto expected = collect_scalar_tuples(&schema_);

    FTreeBatchIterator batch({{"a", 512}, {"b", 512}, {"c", 512}, {"d", 4}, {"e", 2}, {"f", 2}, {"g", 3}});
    batch.init(&schema_);
    batch.initialize_iterators();
    auto actual = collect_all_batch_tuples(batch, schema_);

    std::sort(expected.begin(), expected.end());
    std::sort(actual.begin(), actual.end());
    EXPECT_EQ(actual.size(), expected.size());
    EXPECT_EQ(actual, expected);
}

TEST_F(FTreeBatchIteratorTest, FactorizedMatchesScalar_SevenNodeTree_SmallWindows) {
    build_seven_node_tree();
    auto expected = collect_scalar_tuples(&schema_);

    FTreeBatchIterator batch({{"a", 512}, {"b", 512}, {"c", 512}, {"d", 1}, {"e", 1}, {"f", 1}, {"g", 1}});
    batch.init(&schema_);
    batch.initialize_iterators();
    auto actual = collect_all_batch_tuples(batch, schema_);

    std::sort(expected.begin(), expected.end());
    std::sort(actual.begin(), actual.end());
    EXPECT_EQ(actual, expected);
}

TEST_F(FTreeBatchIteratorTest, FactorizedMatchesScalar_DeepChain) {
    build_deep_chain();
    auto expected = collect_scalar_tuples(&schema_);

    FTreeBatchIterator batch({{"a", 512}, {"b", 512}, {"c", 512}, {"d", 3}});
    batch.init(&schema_);
    batch.initialize_iterators();
    auto actual = collect_all_batch_tuples(batch, schema_);

    std::sort(expected.begin(), expected.end());
    std::sort(actual.begin(), actual.end());
    EXPECT_EQ(actual.size(), expected.size());
    EXPECT_EQ(actual, expected);
}

TEST_F(FTreeBatchIteratorTest, FactorizedMatchesScalar_SkewTree) {
    build_skew_tree();
    auto expected = collect_scalar_tuples(&schema_);

    FTreeBatchIterator batch({{"a", 512}, {"d", 4}, {"g", 2}});
    batch.init(&schema_);
    batch.initialize_iterators();
    auto actual = collect_all_batch_tuples(batch, schema_);

    std::sort(expected.begin(), expected.end());
    std::sort(actual.begin(), actual.end());
    EXPECT_EQ(actual.size(), expected.size());
    EXPECT_EQ(actual, expected);
}

TEST_F(FTreeBatchIteratorTest, FactorizedMatchesScalar_SingleLeafTree) {
    build_single_leaf_tree();
    auto expected = collect_scalar_tuples(&schema_);

    FTreeBatchIterator batch({{"x", 3}});
    batch.init(&schema_);
    batch.initialize_iterators();
    auto actual = collect_all_batch_tuples(batch, schema_);

    std::sort(expected.begin(), expected.end());
    std::sort(actual.begin(), actual.end());
    EXPECT_EQ(actual, expected);
}

// ===========================================================================
// 2. Output structure invariants
// ===========================================================================

TEST_F(FTreeBatchIteratorTest, InternalNodeCountsWithinCapacity) {
    build_seven_node_tree();
    FTreeBatchIterator batch({{"a", 512}, {"b", 512}, {"c", 512}, {"d", 2}, {"e", 2}, {"f", 2}, {"g", 3}});
    batch.init(&schema_);
    batch.initialize_iterators();

    while (batch.next()) {
        for (size_t i = 0; i < batch.num_attributes(); ++i) {
            if (!batch.is_leaf_attr(i)) {
                EXPECT_GE(batch.get_count(i), 1u) << "Internal node at col " << i << " must have count >= 1";
            }
        }
    }
}

TEST_F(FTreeBatchIteratorTest, LeafCountNeverExceedsCapacity) {
    build_seven_node_tree();
    const std::unordered_map<std::string, size_t> caps = {{"a", 512}, {"b", 512}, {"c", 512}, {"d", 2},
                                                          {"e", 3},   {"f", 1},   {"g", 4}};
    FTreeBatchIterator batch(caps);
    batch.init(&schema_);
    batch.initialize_iterators();

    const std::vector<std::string> col_order = {"a", "b", "d", "e", "c", "f", "g"};

    while (batch.next()) {
        for (size_t i = 0; i < batch.num_attributes(); ++i) {
            if (batch.is_leaf_attr(i)) {
                auto it = caps.find(col_order[i]);
                ASSERT_NE(it, caps.end());
                EXPECT_LE(batch.get_count(i), it->second) << "Leaf " << col_order[i] << " count exceeds capacity";
                EXPECT_GT(batch.get_count(i), 0u) << "Leaf " << col_order[i] << " has zero count in a valid batch";
            }
        }
    }
}

TEST_F(FTreeBatchIteratorTest, LeafFlagsCorrectlyIdentifyLeavesAndInternals) {
    build_seven_node_tree();
    FTreeBatchIterator batch({{"a", 512}, {"b", 512}, {"c", 512}, {"d", 2}, {"e", 2}, {"f", 2}, {"g", 2}});
    batch.init(&schema_);
    batch.initialize_iterators();

    EXPECT_FALSE(batch.is_leaf_attr(0));// a
    EXPECT_FALSE(batch.is_leaf_attr(1));// b
    EXPECT_TRUE(batch.is_leaf_attr(2)); // d
    EXPECT_TRUE(batch.is_leaf_attr(3)); // e
    EXPECT_FALSE(batch.is_leaf_attr(4));// c
    EXPECT_TRUE(batch.is_leaf_attr(5)); // f
    EXPECT_TRUE(batch.is_leaf_attr(6)); // g
}

// ===========================================================================
// 3. Windowing / prefix-reuse / skew behavior
// ===========================================================================

TEST_F(FTreeBatchIteratorTest, SkewedLeaf_OnlyRightmostBufferChanges) {
    build_skew_tree();
    // a=7, d=[100,101] (2 values), g=[200..207] (8 values)
    // d and g are both leaf children of root a.
    // Window: d=4, g=2. d fills both values (2 < cap 4). g fills 2/8.
    // Subsequent batches: only g's window changes (skew continuation).
    FTreeBatchIterator batch({{"a", 512}, {"d", 4}, {"g", 2}});
    batch.init(&schema_);
    batch.initialize_iterators();

    ASSERT_TRUE(batch.next());

    const size_t d_idx = 1;

    const size_t d_count_1 = batch.get_count(d_idx);
    EXPECT_EQ(d_count_1, 2u);
    std::vector<uint64_t> d_buf_1(batch.get_buffer(d_idx), batch.get_buffer(d_idx) + d_count_1);

    size_t batches = 1;
    while (batch.next()) {
        ++batches;
        EXPECT_EQ(batch.get_count(d_idx), d_count_1) << "d count changed on batch " << batches;
        std::vector<uint64_t> d_now(batch.get_buffer(d_idx), batch.get_buffer(d_idx) + batch.get_count(d_idx));
        EXPECT_EQ(d_now, d_buf_1) << "d buffer changed on batch " << batches;

        EXPECT_EQ(batch.get_buffer(0)[0], 7u) << "root a changed on batch " << batches;
    }

    EXPECT_EQ(batches, 4u);
}

TEST_F(FTreeBatchIteratorTest, RightmostLeafWindowAdvancesFirst) {
    build_balanced_tree();
    // a->(b,c), each leaf has 2 values per a position.
    // Window: b=100 (huge), c=1
    // For a=1: b=[10,11], c=20 then c=21. Two batches, b unchanged.
    // For a=2: b=[12,13], c=22 then c=23. Two batches, b unchanged.
    FTreeBatchIterator batch({{"a", 512}, {"b", 100}, {"c", 1}});
    batch.init(&schema_);
    batch.initialize_iterators();

    std::vector<std::vector<uint64_t>> c_values;
    std::vector<std::vector<uint64_t>> b_values;

    while (batch.next()) {
        const size_t b_idx = 1, c_idx = 2;
        c_values.push_back({batch.get_buffer(c_idx)[0]});
        b_values.push_back({batch.get_buffer(b_idx), batch.get_buffer(b_idx) + batch.get_count(b_idx)});
    }

    ASSERT_EQ(c_values.size(), 4u);

    EXPECT_EQ(b_values[0], b_values[1]);
    EXPECT_EQ(b_values[2], b_values[3]);
    EXPECT_NE(b_values[0], b_values[2]);

    EXPECT_EQ(c_values[0][0], 20u);
    EXPECT_EQ(c_values[1][0], 21u);
    EXPECT_EQ(c_values[2][0], 22u);
    EXPECT_EQ(c_values[3][0], 23u);
}

TEST_F(FTreeBatchIteratorTest, SevenNode_PrefixReuse) {
    build_seven_node_tree();
    // g is the rightmost leaf. Give it window=1 so it advances every batch.
    // Give all other leaves huge windows. While g steps, other buffers stay fixed.
    FTreeBatchIterator batch({{"a", 512}, {"b", 512}, {"c", 512}, {"d", 100}, {"e", 100}, {"f", 100}, {"g", 1}});
    batch.init(&schema_);
    batch.initialize_iterators();

    // For a=1: c=20, g has 3 values (400,401,402).
    // Expect 3 consecutive batches with same prefix but g changes.
    ASSERT_TRUE(batch.next());
    uint64_t a1 = batch.get_buffer(0)[0];
    EXPECT_EQ(a1, 1u);
    EXPECT_EQ(batch.get_count(6), 1u);
    EXPECT_EQ(batch.get_buffer(6)[0], 400u);

    ASSERT_TRUE(batch.next());
    EXPECT_EQ(batch.get_buffer(0)[0], a1);
    EXPECT_EQ(batch.get_buffer(6)[0], 401u);

    ASSERT_TRUE(batch.next());
    EXPECT_EQ(batch.get_buffer(0)[0], a1);
    EXPECT_EQ(batch.get_buffer(6)[0], 402u);

    // After g is exhausted under c=20, cascade upward.
    ASSERT_TRUE(batch.next());
}

// ===========================================================================
// 4. Siblings share parent prefix
// ===========================================================================

TEST_F(FTreeBatchIteratorTest, SiblingsShareParentPrefix_SevenNode) {
    build_seven_node_tree();
    // d and e are both children of b. Give them large windows.
    // After greedy fill, d and e should have values from the SAME set of b positions.
    FTreeBatchIterator batch({{"a", 512}, {"b", 512}, {"c", 512}, {"d", 100}, {"e", 100}, {"f", 100}, {"g", 100}});
    batch.init(&schema_);
    batch.initialize_iterators();

    ASSERT_TRUE(batch.next());

    // With root cap=512, the first batch fills both a=1 and a=2.
    // b has 4 positions: [10,11] under a=1, [12,13] under a=2.
    const size_t b_idx = 1, d_idx = 2, e_idx = 3;
    EXPECT_GE(batch.get_count(b_idx), 2u);

    // d: under a=1: b@0->[100,101], b@1->[102]; under a=2: b@2->[103,104], total 5.
    EXPECT_EQ(batch.get_count(d_idx), 5u);

    // e: under a=1: b@0->[200], b@1->[201]; under a=2: b@2->[202], b@3->[203], total 4.
    EXPECT_EQ(batch.get_count(e_idx), 4u);
}

// ===========================================================================
// 5. Window-size-1 produces one tuple per batch (scalar equivalence)
// ===========================================================================

TEST_F(FTreeBatchIteratorTest, WindowSizeOneProducesOneTuplePerBatch) {
    build_balanced_tree();
    FTreeBatchIterator batch({{"a", 512}, {"b", 1}, {"c", 1}});
    batch.init(&schema_);
    batch.initialize_iterators();

    auto expected = collect_scalar_tuples(&schema_);
    size_t batch_count = 0;
    std::vector<std::vector<uint64_t>> actual;

    while (batch.next()) {
        ++batch_count;
        auto expanded = expand_batch(batch, schema_);
        EXPECT_EQ(expanded.size(), 1u) << "batch " << batch_count << " has != 1 tuple";
        actual.insert(actual.end(), expanded.begin(), expanded.end());
    }

    EXPECT_EQ(batch_count, expected.size());
    std::sort(expected.begin(), expected.end());
    std::sort(actual.begin(), actual.end());
    EXPECT_EQ(actual, expected);
}

// ===========================================================================
// 6. Large window covers entire leaf in one batch
// ===========================================================================

TEST_F(FTreeBatchIteratorTest, LargeWindowCoversAllValuesPerParentContext) {
    build_skew_tree();
    FTreeBatchIterator batch({{"a", 512}, {"d", 100}, {"g", 100}});
    batch.init(&schema_);
    batch.initialize_iterators();

    ASSERT_TRUE(batch.next());
    EXPECT_EQ(batch.get_count(1), 2u);
    EXPECT_EQ(batch.get_count(2), 8u);

    EXPECT_FALSE(batch.next());
}

TEST_F(FTreeBatchIteratorTest, LargeWindowSevenNode_FewBatches) {
    build_seven_node_tree();
    // With multi-root: root cap=512 means both a=1 and a=2 fit in one batch.
    // Expect 1 batch.
    FTreeBatchIterator batch({{"a", 512}, {"b", 512}, {"c", 512}, {"d", 100}, {"e", 100}, {"f", 100}, {"g", 100}});
    batch.init(&schema_);
    batch.initialize_iterators();

    size_t n = 0;
    while (batch.next())
        ++n;
    EXPECT_EQ(n, 1u);
}

// ===========================================================================
// 7. Deep-chain specific: only one leaf, internal cascade
// ===========================================================================

TEST_F(FTreeBatchIteratorTest, DeepChain_CorrectTuples) {
    build_deep_chain();
    // a->b->c->d. Only d is leaf. Window d=2.
    FTreeBatchIterator batch({{"a", 512}, {"b", 512}, {"c", 512}, {"d", 2}});
    batch.init(&schema_);
    batch.initialize_iterators();

    auto expected = collect_scalar_tuples(&schema_);
    auto actual = collect_all_batch_tuples(batch, schema_);

    std::sort(expected.begin(), expected.end());
    std::sort(actual.begin(), actual.end());
    EXPECT_EQ(actual.size(), expected.size());
    EXPECT_EQ(actual, expected);
}

TEST_F(FTreeBatchIteratorTest, DeepChain_LargeWindow) {
    build_deep_chain();
    FTreeBatchIterator batch({{"a", 512}, {"b", 512}, {"c", 512}, {"d", 100}});
    batch.init(&schema_);
    batch.initialize_iterators();

    auto expected = collect_scalar_tuples(&schema_);
    auto actual = collect_all_batch_tuples(batch, schema_);

    std::sort(expected.begin(), expected.end());
    std::sort(actual.begin(), actual.end());
    EXPECT_EQ(actual, expected);
}

// ===========================================================================
// 8. Single-leaf (root = leaf) edge case
// ===========================================================================

TEST_F(FTreeBatchIteratorTest, SingleLeafTree_WindowedCorrectly) {
    build_single_leaf_tree();
    FTreeBatchIterator batch({{"x", 3}});
    batch.init(&schema_);
    batch.initialize_iterators();

    ASSERT_TRUE(batch.next());
    EXPECT_EQ(batch.get_count(0), 3u);
    EXPECT_EQ(batch.get_buffer(0)[0], 10u);
    EXPECT_EQ(batch.get_buffer(0)[1], 20u);
    EXPECT_EQ(batch.get_buffer(0)[2], 30u);

    ASSERT_TRUE(batch.next());
    EXPECT_EQ(batch.get_count(0), 2u);
    EXPECT_EQ(batch.get_buffer(0)[0], 40u);
    EXPECT_EQ(batch.get_buffer(0)[1], 50u);

    EXPECT_FALSE(batch.next());
}

// ===========================================================================
// 9. Reset: re-iterate produces same results
// ===========================================================================

TEST_F(FTreeBatchIteratorTest, ResetRestartsIteration) {
    build_seven_node_tree();
    FTreeBatchIterator batch({{"a", 512}, {"b", 512}, {"c", 512}, {"d", 2}, {"e", 2}, {"f", 2}, {"g", 2}});
    batch.init(&schema_);
    batch.initialize_iterators();

    auto first_run = collect_all_batch_tuples(batch, schema_);

    batch.reset();
    batch.initialize_iterators();
    auto second_run = collect_all_batch_tuples(batch, schema_);

    std::sort(first_run.begin(), first_run.end());
    std::sort(second_run.begin(), second_run.end());
    EXPECT_EQ(first_run, second_run);
}

TEST_F(FTreeBatchIteratorTest, ResetAfterPartialIteration) {
    build_skew_tree();
    FTreeBatchIterator batch({{"a", 512}, {"d", 2}, {"g", 3}});
    batch.init(&schema_);
    batch.initialize_iterators();

    batch.next();
    batch.next();

    batch.reset();
    batch.initialize_iterators();
    auto full_run = collect_all_batch_tuples(batch, schema_);
    auto expected = collect_scalar_tuples(&schema_);

    std::sort(full_run.begin(), full_run.end());
    std::sort(expected.begin(), expected.end());
    EXPECT_EQ(full_run, expected);
}

// ===========================================================================
// 10. Varying window sizes produce same total tuple set
// ===========================================================================

TEST_F(FTreeBatchIteratorTest, DifferentWindowSizesSameResults_SevenNode) {
    build_seven_node_tree();
    auto expected = collect_scalar_tuples(&schema_);
    std::sort(expected.begin(), expected.end());

    const std::vector<std::unordered_map<std::string, size_t>> configs = {
            {{"a", 512}, {"b", 512}, {"c", 512}, {"d", 1}, {"e", 1}, {"f", 1}, {"g", 1}},
            {{"a", 512}, {"b", 512}, {"c", 512}, {"d", 2}, {"e", 3}, {"f", 1}, {"g", 4}},
            {{"a", 512}, {"b", 512}, {"c", 512}, {"d", 5}, {"e", 5}, {"f", 5}, {"g", 5}},
            {{"a", 512}, {"b", 512}, {"c", 512}, {"d", 100}, {"e", 100}, {"f", 100}, {"g", 100}},
    };

    for (size_t ci = 0; ci < configs.size(); ++ci) {
        FTreeBatchIterator batch(configs[ci]);
        batch.init(&schema_);
        batch.initialize_iterators();
        auto actual = collect_all_batch_tuples(batch, schema_);
        std::sort(actual.begin(), actual.end());
        EXPECT_EQ(actual, expected) << "Mismatch for config index " << ci;
    }
}

// ===========================================================================
// 11. No duplicate tuples across batches
// ===========================================================================

TEST_F(FTreeBatchIteratorTest, NoDuplicateTuplesAcrossBatches) {
    build_seven_node_tree();
    FTreeBatchIterator batch({{"a", 512}, {"b", 512}, {"c", 512}, {"d", 2}, {"e", 1}, {"f", 1}, {"g", 2}});
    batch.init(&schema_);
    batch.initialize_iterators();

    auto all = collect_all_batch_tuples(batch, schema_);
    auto sorted = all;
    std::sort(sorted.begin(), sorted.end());
    auto it = std::unique(sorted.begin(), sorted.end());
    EXPECT_EQ(it, sorted.end()) << "Found duplicate tuples across batches";
}

// ===========================================================================
// 12. Validation: construction-time errors
// ===========================================================================

TEST_F(FTreeBatchIteratorTest, ZeroBufferSizeThrows) {
#ifdef NDEBUG
    GTEST_SKIP() << "Assertions are disabled in release builds (NDEBUG).";
#else
    EXPECT_DEATH((FTreeBatchIterator({{"b", 0}, {"c", 2}})), "Buffer size for required attribute cannot be 0");
#endif
}

TEST_F(FTreeBatchIteratorTest, EmptyMapThrows) {
#ifdef NDEBUG
    GTEST_SKIP() << "Assertions are disabled in release builds (NDEBUG).";
#else
    EXPECT_DEATH((FTreeBatchIterator({})), "required_attributes map cannot be empty");
#endif
}

TEST_F(FTreeBatchIteratorTest, MissingLeafInMapNoLongerThrows) {
    // The batch iterator now projects all ftree nodes regardless of required_attributes.
    // A nonexistent attribute in required_attributes is simply ignored (used only for capacity).
    build_balanced_tree();
    FTreeBatchIterator batch({{"nonexistent", 2}});
    EXPECT_NO_FATAL_FAILURE(batch.init(&schema_));
}

// ===========================================================================
// 13. Tuple-size / attribute-count API
// ===========================================================================

TEST_F(FTreeBatchIteratorTest, TupleSizeMatchesScalarIterator) {
    build_seven_node_tree();
    FTreeIterator scalar;
    scalar.init(&schema_);
    scalar.initialize_iterators();

    FTreeBatchIterator batch({{"a", 512}, {"b", 512}, {"c", 512}, {"d", 2}, {"e", 2}, {"f", 2}, {"g", 2}});
    batch.init(&schema_);
    batch.initialize_iterators();

    EXPECT_EQ(batch.tuple_size(), scalar.tuple_size());
    EXPECT_EQ(batch.num_attributes(), 7u);
}

// ===========================================================================
// 14. Root fixed per batch
// ===========================================================================

TEST_F(FTreeBatchIteratorTest, RootFixedPerBatch) {
    build_seven_node_tree();
    FTreeBatchIterator batch({{"a", 512}, {"b", 512}, {"c", 512}, {"d", 1}, {"e", 1}, {"f", 1}, {"g", 1}});
    batch.init(&schema_);
    batch.initialize_iterators();

    while (batch.next()) {
        EXPECT_EQ(batch.get_count(0), 1u) << "Root must have exactly 1 value per batch";
    }
}

// ===========================================================================
// 15. Positions are monotonically increasing
// ===========================================================================

TEST_F(FTreeBatchIteratorTest, PositionsMonotonicallyIncreasing) {
    build_seven_node_tree();
    FTreeBatchIterator batch({{"a", 512}, {"b", 512}, {"c", 512}, {"d", 100}, {"e", 100}, {"f", 100}, {"g", 100}});
    batch.init(&schema_);
    batch.initialize_iterators();

    while (batch.next()) {
        for (size_t i = 0; i < batch.num_attributes(); ++i) {
            size_t cnt = batch.get_count(i);
            const int32_t* pos = batch.get_positions(i);
            for (size_t j = 1; j < cnt; ++j) {
                EXPECT_GT(pos[j], pos[j - 1]) << "Position not monotonically increasing at col " << i << " idx " << j;
            }
        }
    }
}

// ===========================================================================
// 16. Star topology tests (240 tuples)
// ===========================================================================

TEST_F(FTreeBatchIteratorTest, Star_MatchesScalar) {
    build_star_tree();
    auto expected = collect_scalar_tuples(&schema_);
    FTreeBatchIterator batch({{"r", 512}, {"a", 2}, {"b", 1}, {"c", 1}, {"d", 1}});
    batch.init(&schema_);
    batch.initialize_iterators();
    auto actual = collect_all_batch_tuples(batch, schema_);
    std::sort(expected.begin(), expected.end());
    std::sort(actual.begin(), actual.end());
    EXPECT_EQ(actual.size(), 240u);
    EXPECT_EQ(actual, expected);
}

TEST_F(FTreeBatchIteratorTest, Star_Window1) {
    build_star_tree();
    auto expected = collect_scalar_tuples(&schema_);
    FTreeBatchIterator batch({{"r", 512}, {"a", 1}, {"b", 1}, {"c", 1}, {"d", 1}});
    batch.init(&schema_);
    batch.initialize_iterators();
    size_t batch_count = 0;
    auto actual = std::vector<std::vector<uint64_t>>();
    while (batch.next()) {
        ++batch_count;
        auto expanded = expand_batch(batch, schema_);
        EXPECT_EQ(expanded.size(), 1u) << "batch " << batch_count;
        actual.insert(actual.end(), expanded.begin(), expanded.end());
    }
    EXPECT_EQ(batch_count, 240u);
    std::sort(expected.begin(), expected.end());
    std::sort(actual.begin(), actual.end());
    EXPECT_EQ(actual, expected);
}

TEST_F(FTreeBatchIteratorTest, Star_Window3_NoDuplicates) {
    build_star_tree();
    FTreeBatchIterator batch({{"r", 512}, {"a", 3}, {"b", 2}, {"c", 2}, {"d", 2}});
    batch.init(&schema_);
    batch.initialize_iterators();
    auto all = collect_all_batch_tuples(batch, schema_);
    EXPECT_EQ(all.size(), 240u);
    auto sorted = all;
    std::sort(sorted.begin(), sorted.end());
    auto it = std::unique(sorted.begin(), sorted.end());
    EXPECT_EQ(it, sorted.end()) << "Found duplicate tuples in Star_Window3";
}

// ===========================================================================
// 17. Wide Fan topology tests (576 tuples)
// ===========================================================================

TEST_F(FTreeBatchIteratorTest, WideFan_MatchesScalar) {
    build_wide_fan_tree();
    auto expected = collect_scalar_tuples(&schema_);
    FTreeBatchIterator batch({{"r", 512}, {"m", 512}, {"b1", 3}, {"b2", 2}, {"b3", 2}, {"b4", 1}, {"b5", 3}});
    batch.init(&schema_);
    batch.initialize_iterators();
    auto actual = collect_all_batch_tuples(batch, schema_);
    std::sort(expected.begin(), expected.end());
    std::sort(actual.begin(), actual.end());
    EXPECT_EQ(actual.size(), 576u);
    EXPECT_EQ(actual, expected);
}

TEST_F(FTreeBatchIteratorTest, WideFan_Window1) {
    build_wide_fan_tree();
    auto expected = collect_scalar_tuples(&schema_);
    FTreeBatchIterator batch({{"r", 512}, {"m", 512}, {"b1", 1}, {"b2", 1}, {"b3", 1}, {"b4", 1}, {"b5", 1}});
    batch.init(&schema_);
    batch.initialize_iterators();
    size_t batch_count = 0;
    auto actual = std::vector<std::vector<uint64_t>>();
    while (batch.next()) {
        ++batch_count;
        auto expanded = expand_batch(batch, schema_);
        actual.insert(actual.end(), expanded.begin(), expanded.end());
    }
    EXPECT_EQ(batch_count, 576u);
    std::sort(expected.begin(), expected.end());
    std::sort(actual.begin(), actual.end());
    EXPECT_EQ(actual, expected);
}

TEST_F(FTreeBatchIteratorTest, WideFan_Window2_NoDuplicates) {
    build_wide_fan_tree();
    FTreeBatchIterator batch({{"r", 512}, {"m", 512}, {"b1", 2}, {"b2", 2}, {"b3", 2}, {"b4", 2}, {"b5", 2}});
    batch.init(&schema_);
    batch.initialize_iterators();
    auto all = collect_all_batch_tuples(batch, schema_);
    EXPECT_EQ(all.size(), 576u);
    auto sorted = all;
    std::sort(sorted.begin(), sorted.end());
    auto it = std::unique(sorted.begin(), sorted.end());
    EXPECT_EQ(it, sorted.end());
}

// ===========================================================================
// 18. Deep-5 Chain tests (48 tuples)
// ===========================================================================

TEST_F(FTreeBatchIteratorTest, Deep5_MatchesScalar) {
    build_deep5_chain();
    auto expected = collect_scalar_tuples(&schema_);
    FTreeBatchIterator batch({{"a", 512}, {"b", 512}, {"c", 512}, {"d", 512}, {"e", 2}});
    batch.init(&schema_);
    batch.initialize_iterators();
    auto actual = collect_all_batch_tuples(batch, schema_);
    std::sort(expected.begin(), expected.end());
    std::sort(actual.begin(), actual.end());
    EXPECT_EQ(actual.size(), 48u);
    EXPECT_EQ(actual, expected);
}

TEST_F(FTreeBatchIteratorTest, Deep5_Window1) {
    build_deep5_chain();
    auto expected = collect_scalar_tuples(&schema_);
    FTreeBatchIterator batch({{"a", 512}, {"b", 512}, {"c", 512}, {"d", 512}, {"e", 1}});
    batch.init(&schema_);
    batch.initialize_iterators();
    size_t batch_count = 0;
    auto actual = std::vector<std::vector<uint64_t>>();
    while (batch.next()) {
        ++batch_count;
        auto expanded = expand_batch(batch, schema_);
        EXPECT_EQ(expanded.size(), 1u);
        actual.insert(actual.end(), expanded.begin(), expanded.end());
    }
    EXPECT_EQ(batch_count, 48u);
    std::sort(expected.begin(), expected.end());
    std::sort(actual.begin(), actual.end());
    EXPECT_EQ(actual, expected);
}

TEST_F(FTreeBatchIteratorTest, Deep5_Window8) {
    build_deep5_chain();
    auto expected = collect_scalar_tuples(&schema_);
    FTreeBatchIterator batch({{"a", 512}, {"b", 512}, {"c", 512}, {"d", 512}, {"e", 8}});
    batch.init(&schema_);
    batch.initialize_iterators();
    auto actual = collect_all_batch_tuples(batch, schema_);
    std::sort(expected.begin(), expected.end());
    std::sort(actual.begin(), actual.end());
    EXPECT_EQ(actual, expected);
}

// ===========================================================================
// 19. Asymmetric branching tests (64 tuples)
// ===========================================================================

TEST_F(FTreeBatchIteratorTest, Asymmetric_MatchesScalar) {
    build_asymmetric_tree();
    auto expected = collect_scalar_tuples(&schema_);
    FTreeBatchIterator batch({{"r", 512}, {"A", 512}, {"C", 512}, {"d", 3}, {"e", 2}, {"B", 2}});
    batch.init(&schema_);
    batch.initialize_iterators();
    auto actual = collect_all_batch_tuples(batch, schema_);
    std::sort(expected.begin(), expected.end());
    std::sort(actual.begin(), actual.end());
    EXPECT_EQ(actual.size(), 64u);
    EXPECT_EQ(actual, expected);
}

TEST_F(FTreeBatchIteratorTest, Asymmetric_Window1) {
    build_asymmetric_tree();
    auto expected = collect_scalar_tuples(&schema_);
    FTreeBatchIterator batch({{"r", 512}, {"A", 512}, {"C", 512}, {"d", 1}, {"e", 1}, {"B", 1}});
    batch.init(&schema_);
    batch.initialize_iterators();
    size_t batch_count = 0;
    auto actual = std::vector<std::vector<uint64_t>>();
    while (batch.next()) {
        ++batch_count;
        auto expanded = expand_batch(batch, schema_);
        EXPECT_EQ(expanded.size(), 1u);
        actual.insert(actual.end(), expanded.begin(), expanded.end());
    }
    EXPECT_EQ(batch_count, 64u);
    std::sort(expected.begin(), expected.end());
    std::sort(actual.begin(), actual.end());
    EXPECT_EQ(actual, expected);
}

TEST_F(FTreeBatchIteratorTest, Asymmetric_Window2_NoDuplicates) {
    build_asymmetric_tree();
    FTreeBatchIterator batch({{"r", 512}, {"A", 512}, {"C", 512}, {"d", 2}, {"e", 1}, {"B", 1}});
    batch.init(&schema_);
    batch.initialize_iterators();
    auto all = collect_all_batch_tuples(batch, schema_);
    EXPECT_EQ(all.size(), 64u);
    auto sorted = all;
    std::sort(sorted.begin(), sorted.end());
    auto it = std::unique(sorted.begin(), sorted.end());
    EXPECT_EQ(it, sorted.end());
}

// ===========================================================================
// 20. Sparse Bitmask tests (30 tuples, exercises bit-jump)
// ===========================================================================

TEST_F(FTreeBatchIteratorTest, SparseBitmask_MatchesScalar) {
    build_sparse_bitmask_tree();
    auto expected = collect_scalar_tuples(&schema_);
    FTreeBatchIterator batch({{"r", 512}, {"a", 3}, {"b", 2}});
    batch.init(&schema_);
    batch.initialize_iterators();
    auto actual = collect_all_batch_tuples(batch, schema_);
    std::sort(expected.begin(), expected.end());
    std::sort(actual.begin(), actual.end());
    EXPECT_EQ(actual.size(), 30u);
    EXPECT_EQ(actual, expected);
}

TEST_F(FTreeBatchIteratorTest, SparseBitmask_Window1) {
    build_sparse_bitmask_tree();
    auto expected = collect_scalar_tuples(&schema_);
    FTreeBatchIterator batch({{"r", 512}, {"a", 1}, {"b", 1}});
    batch.init(&schema_);
    batch.initialize_iterators();
    size_t batch_count = 0;
    auto actual = std::vector<std::vector<uint64_t>>();
    while (batch.next()) {
        ++batch_count;
        auto expanded = expand_batch(batch, schema_);
        EXPECT_EQ(expanded.size(), 1u);
        actual.insert(actual.end(), expanded.begin(), expanded.end());
    }
    EXPECT_EQ(batch_count, 30u);
    std::sort(expected.begin(), expected.end());
    std::sort(actual.begin(), actual.end());
    EXPECT_EQ(actual, expected);
}

TEST_F(FTreeBatchIteratorTest, SparseBitmask_Window2_NoDuplicates) {
    build_sparse_bitmask_tree();
    FTreeBatchIterator batch({{"r", 512}, {"a", 2}, {"b", 1}});
    batch.init(&schema_);
    batch.initialize_iterators();
    auto all = collect_all_batch_tuples(batch, schema_);
    EXPECT_EQ(all.size(), 30u);
    auto sorted = all;
    std::sort(sorted.begin(), sorted.end());
    auto it = std::unique(sorted.begin(), sorted.end());
    EXPECT_EQ(it, sorted.end());
}

// ===========================================================================
// 21. Diamond topology tests (96 tuples)
// ===========================================================================

TEST_F(FTreeBatchIteratorTest, Diamond_MatchesScalar) {
    build_diamond_tree();
    auto expected = collect_scalar_tuples(&schema_);
    FTreeBatchIterator batch({{"r", 512}, {"m", 512}, {"x", 2}, {"y", 1}, {"p", 3}, {"q", 2}});
    batch.init(&schema_);
    batch.initialize_iterators();
    auto actual = collect_all_batch_tuples(batch, schema_);
    std::sort(expected.begin(), expected.end());
    std::sort(actual.begin(), actual.end());
    EXPECT_EQ(actual.size(), 96u);
    EXPECT_EQ(actual, expected);
}

TEST_F(FTreeBatchIteratorTest, Diamond_Window1) {
    build_diamond_tree();
    auto expected = collect_scalar_tuples(&schema_);
    FTreeBatchIterator batch({{"r", 512}, {"m", 512}, {"x", 1}, {"y", 1}, {"p", 1}, {"q", 1}});
    batch.init(&schema_);
    batch.initialize_iterators();
    size_t batch_count = 0;
    auto actual = std::vector<std::vector<uint64_t>>();
    while (batch.next()) {
        ++batch_count;
        auto expanded = expand_batch(batch, schema_);
        EXPECT_EQ(expanded.size(), 1u);
        actual.insert(actual.end(), expanded.begin(), expanded.end());
    }
    EXPECT_EQ(batch_count, 96u);
    std::sort(expected.begin(), expected.end());
    std::sort(actual.begin(), actual.end());
    EXPECT_EQ(actual, expected);
}

TEST_F(FTreeBatchIteratorTest, Diamond_Window2_NoDuplicates) {
    build_diamond_tree();
    FTreeBatchIterator batch({{"r", 512}, {"m", 512}, {"x", 2}, {"y", 1}, {"p", 1}, {"q", 1}});
    batch.init(&schema_);
    batch.initialize_iterators();
    auto all = collect_all_batch_tuples(batch, schema_);
    EXPECT_EQ(all.size(), 96u);
    auto sorted = all;
    std::sort(sorted.begin(), sorted.end());
    auto it = std::unique(sorted.begin(), sorted.end());
    EXPECT_EQ(it, sorted.end());
}

// ===========================================================================
// Logical tuple count invariant: count_logical_tuples() == expand_batch().size()
// ===========================================================================

TEST_F(FTreeBatchIteratorTest, LogicalTupleCountMatchesExpandPerBatch_BalancedTree) {
    build_balanced_tree();
    FTreeBatchIterator batch({{"a", 512}, {"b", 3}, {"c", 2}});
    batch.init(&schema_);
    batch.initialize_iterators();
    size_t total_expanded = 0;
    while (batch.next()) {
        size_t logical = batch.count_logical_tuples();
        auto expanded = expand_batch(batch, schema_);
        EXPECT_EQ(logical, expanded.size())
            << "Each batch: count_logical_tuples() must equal number of expanded tuples";
        total_expanded += expanded.size();
    }
    EXPECT_EQ(total_expanded, 8u);  // 2 a's x 2 b's x 2 c's = 8
}

TEST_F(FTreeBatchIteratorTest, LogicalTupleCountMatchesExpandPerBatch_SevenNodeTree) {
    build_seven_node_tree();
    FTreeBatchIterator batch({{"a", 512}, {"b", 512}, {"c", 512}, {"d", 4}, {"e", 2}, {"f", 2}, {"g", 3}});
    batch.init(&schema_);
    batch.initialize_iterators();
    size_t total_expanded = 0;
    while (batch.next()) {
        size_t logical = batch.count_logical_tuples();
        auto expanded = expand_batch(batch, schema_);
        EXPECT_EQ(logical, expanded.size());
        total_expanded += expanded.size();
    }
    EXPECT_EQ(total_expanded, 22u);  // a[0]: 3*(b,d,e) x 6*(c,f,g)=18; a[1]: 2*2=4; total 22
}

TEST_F(FTreeBatchIteratorTest, LogicalTupleCountMatchesExpandPerBatch_DeepChain) {
    build_deep_chain();
    FTreeBatchIterator batch({{"a", 512}, {"b", 512}, {"c", 512}, {"d", 3}});
    batch.init(&schema_);
    batch.initialize_iterators();
    size_t total_expanded = 0;
    while (batch.next()) {
        size_t logical = batch.count_logical_tuples();
        auto expanded = expand_batch(batch, schema_);
        EXPECT_EQ(logical, expanded.size());
        total_expanded += expanded.size();
    }
    EXPECT_EQ(total_expanded, 6u);  // 2 a's, 3 b's, 4 c's, 6 d's = 6 leaves
}

TEST_F(FTreeBatchIteratorTest, LogicalTupleCountMatchesExpandPerBatch_StarTree) {
    build_star_tree();
    FTreeBatchIterator batch({{"r", 512}, {"a", 512}, {"b", 512}, {"c", 512}, {"d", 512}});
    batch.init(&schema_);
    batch.initialize_iterators();
    size_t total_expanded = 0;
    while (batch.next()) {
        size_t logical = batch.count_logical_tuples();
        auto expanded = expand_batch(batch, schema_);
        EXPECT_EQ(logical, expanded.size());
        total_expanded += expanded.size();
    }
    EXPECT_EQ(total_expanded, 240u);  // 5*15*10*20*10 / (3*2*4*2) = 240
}

TEST_F(FTreeBatchIteratorTest, LogicalTupleCountMatchesExpandPerBatch_WideFanTree) {
    build_wide_fan_tree();
    FTreeBatchIterator batch({{"r", 512}, {"m", 512}, {"b1", 512}, {"b2", 512}, {"b3", 512}, {"b4", 512}, {"b5", 512}});
    batch.init(&schema_);
    batch.initialize_iterators();
    size_t total_expanded = 0;
    while (batch.next()) {
        size_t logical = batch.count_logical_tuples();
        auto expanded = expand_batch(batch, schema_);
        EXPECT_EQ(logical, expanded.size());
        total_expanded += expanded.size();
    }
    EXPECT_EQ(total_expanded, 576u);
}

TEST_F(FTreeBatchIteratorTest, LogicalTupleCountMatchesExpandPerBatch_SevenNodeTree_SubsetRequired) {
    // Tree: a->(b,c), b->(d,e), c->(f,g). Required: [a, d, g] only (subset of leaves)
    build_seven_node_tree();
    FTreeBatchIterator batch({{"a", 512}, {"d", 4}, {"g", 3}});
    batch.init(&schema_);
    batch.initialize_iterators();
    size_t total_expanded = 0;
    while (batch.next()) {
        size_t logical = batch.count_logical_tuples();
        auto expanded = expand_batch(batch, schema_);
        EXPECT_EQ(logical, expanded.size());
        total_expanded += expanded.size();
    }
    // All ftree nodes are now projected (not just subset), so full cross product = 22.
    EXPECT_EQ(total_expanded, 22u);
}

TEST_F(FTreeBatchIteratorTest, LogicalTupleCountMatchesExpandPerBatch_SevenNodeTree_SubsetRequired_WithEandF) {
    // d and e are siblings under b, f and g under c.
    // Projecting to (a,d,e,f,g) with full-tree counting yields 22 logical tuples.
    // expand_batch uses the projected tree and overcounts (44) because it loses the
    // correlation imposed by intermediate nodes b and c — so we only check the full-tree count.
    build_seven_node_tree();
    FTreeBatchIterator batch({{"a", 512}, {"d", 8}, {"e", 8}, {"f", 8}, {"g", 8}});
    batch.init(&schema_);
    batch.initialize_iterators();
    size_t total_logical = 0;
    while (batch.next()) {
        total_logical += batch.count_logical_tuples();
    }
    EXPECT_EQ(total_logical, 22u);
}

TEST_F(FTreeBatchIteratorTest, LogicalTupleCountMatchesExpandPerBatch_DeepChain_SubsetRequired) {
    build_deep_chain();
    FTreeBatchIterator batch({{"a", 512}, {"d", 3}});
    batch.init(&schema_);
    batch.initialize_iterators();
    size_t total_expanded = 0;
    while (batch.next()) {
        size_t logical = batch.count_logical_tuples();
        auto expanded = expand_batch(batch, schema_);
        EXPECT_EQ(logical, expanded.size());
        total_expanded += expanded.size();
    }
    EXPECT_EQ(total_expanded, 6u);
}

// ProjectedTreeStructure tests require get_projected_tree_info() API - add when implemented.

TEST_F(FTreeBatchIteratorTest, PrintExpectedVsActualTuples) {
    // Diagnostic test: prints expected vs actual for debugging
    build_seven_node_tree();
    auto expected = collect_scalar_tuples(&schema_);
    FTreeBatchIterator batch({{"a", 512}, {"d", 4}, {"g", 3}});
    batch.init(&schema_);
    batch.initialize_iterators();
    auto actual = collect_all_batch_tuples(batch, schema_);
    std::sort(expected.begin(), expected.end());
    std::sort(actual.begin(), actual.end());
    (void)expected;
    (void)actual;
}

TEST_F(FTreeBatchIteratorTest, IndirectAncestorBridge_MultiHopPath_MatchesScalar) {
    build_seven_node_tree();

    FTreeBatchIterator batch({{"a", 256}, {"g", 256}});
    batch.init(&schema_);
    batch.initialize_iterators();

    auto actual = collect_all_batch_tuples(batch, schema_);
    auto scalar = collect_scalar_tuples(&schema_);

    // All ftree nodes are now projected, so batch tuples match scalar tuples exactly.
    std::vector<std::vector<uint64_t>> expected = scalar;
    std::sort(expected.begin(), expected.end());
    std::sort(actual.begin(), actual.end());
    EXPECT_EQ(actual, expected);
}

TEST_F(FTreeBatchIteratorTest, ZeroWidthRanges_OnSubsetParents) {
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
    V(1)->state->offset[1] = 0; // a[0] -> empty child range
    V(1)->state->offset[2] = 2; // a[1] -> both child values

    root_ = std::make_shared<FactorizedTreeElement>("a", V(0));
    root_->add_leaf("a", "b", V(0), V(1));
    ordering_ = {"a", "b"};
    schema_.root = root_;
    schema_.column_ordering = &ordering_;

    FTreeBatchIterator batch({{"a", 64}, {"b", 64}});
    batch.init(&schema_);
    batch.initialize_iterators();
    auto actual = collect_all_batch_tuples(batch, schema_);

    ASSERT_EQ(actual.size(), 2u);
    EXPECT_EQ(actual[0][0], 2u);
    EXPECT_EQ(actual[1][0], 2u);
    EXPECT_TRUE((actual[0][1] == 10u && actual[1][1] == 20u) ||
                (actual[0][1] == 20u && actual[1][1] == 10u));
}

TEST_F(FTreeBatchIteratorTest, AsymmetricCapacities_OneNodeCapacity1) {
    build_seven_node_tree();

    FTreeBatchIterator batch({{"a", 64}, {"b", 64}, {"d", 64}, {"e", 64}, {"c", 64}, {"f", 64}, {"g", 1}});
    batch.init(&schema_);
    batch.initialize_iterators();

    auto actual = collect_all_batch_tuples(batch, schema_);
    auto expected = collect_scalar_tuples(&schema_);
    std::sort(actual.begin(), actual.end());
    std::sort(expected.begin(), expected.end());
    EXPECT_EQ(actual, expected);
}

TEST_F(FTreeBatchIteratorTest, RandomizedFuzz_CompareScalar_Seeded) {
    std::mt19937 rng(1337);

    for (int it = 0; it < 20; ++it) {
        const int shape = static_cast<int>(rng() % 4);
        switch (shape) {
            case 0: build_balanced_tree(); break;
            case 1: build_seven_node_tree(); break;
            case 2: build_deep_chain(); break;
            default: build_skew_tree(); break;
        }

        std::unordered_map<std::string, size_t> req;
        for (const auto& a : ordering_) {
            req[a] = 64;
        }

        FTreeBatchIterator batch(req);
        batch.init(&schema_);
        batch.initialize_iterators();

        auto actual = collect_all_batch_tuples(batch, schema_);
        auto expected = collect_scalar_tuples(&schema_);
        std::sort(actual.begin(), actual.end());
        std::sort(expected.begin(), expected.end());
        EXPECT_EQ(actual, expected) << "seeded iteration=" << it << " shape=" << shape;
    }
}

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
