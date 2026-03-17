#include "../src/plan/include/plan_tree.hpp"
#include "../src/query/include/query.hpp"
#include "../src/operator/include/operator.hpp"
#include "../src/operator/include/scan/scan.hpp"
#include "../src/operator/include/scan/scan_synchronized.hpp"
#include "../src/operator/include/join/inljoin_packed.hpp"
#include "../src/operator/include/join/flat_join.hpp"
#include "../src/operator/include/join/intersection.hpp"
#include "../src/operator/include/join/nway_intersection.hpp"
#include "../src/operator/include/sink/sink_linear.hpp"
#include "../src/operator/include/sink/sink_packed.hpp"
#include "join/inljoin_packed_cascade.hpp"
#include "join/inljoin_packed_gp_cascade.hpp"

#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace ffx;

// Helper to extract operator chain info with attributes
struct OperatorInfo {
    std::string type;
    std::vector<std::string> attributes;
    
    std::string to_string() const {
        std::string result = type + "(";
        for (size_t i = 0; i < attributes.size(); i++) {
            if (i > 0) result += ", ";
            result += attributes[i];
        }
        result += ")";
        return result;
    }
};

std::vector<OperatorInfo> extract_operator_chain(Operator* op) {
    std::vector<OperatorInfo> chain;
    while (op) {
        OperatorInfo info;
        if (auto* scan = dynamic_cast<Scan<>*>(op)) {
            info.type = "Scan";
            info.attributes.push_back(scan->attribute());
        } else if (auto* scan_sync = dynamic_cast<ScanSynchronized*>(op)) {
            info.type = "ScanSynchronized";
            info.attributes.push_back(scan_sync->attribute());
        } else if (auto* join = dynamic_cast<INLJoinPackedGPCascade<>*>(op)) {
            info.type = "INLJoinPackedGPCascade";
            info.attributes.push_back(join->join_key());
            info.attributes.push_back(join->output_key());
            info.attributes.push_back(join->is_join_index_fwd() ? "fwd" : "bwd");
        } else if (auto* join = dynamic_cast<INLJoinPackedCascade<>*>(op)) {
            info.type = "INLJoinPackedCascade";
            info.attributes.push_back(join->join_key());
            info.attributes.push_back(join->output_key());
            info.attributes.push_back(join->is_join_index_fwd() ? "fwd" : "bwd");
        } else if (auto* join = dynamic_cast<INLJoinPacked<>*>(op)) {
            info.type = "INLJoinPacked";
            info.attributes.push_back(join->join_key());
            info.attributes.push_back(join->output_key());
            info.attributes.push_back(join->is_join_index_fwd() ? "fwd" : "bwd");
        } else if (auto* flatjoin = dynamic_cast<FlatJoin<>*>(op)) {
            info.type = "FlatJoin";
            info.attributes.push_back(flatjoin->parent_attr());
            info.attributes.push_back(flatjoin->lca_attr());
            info.attributes.push_back(flatjoin->output_attr());
            info.attributes.push_back(flatjoin->is_join_index_fwd() ? "fwd" : "bwd");
        } else if (auto* nway_intersection = dynamic_cast<NWayIntersection<>*>(op)) {
            info.type = "NWayIntersection";
            info.attributes.push_back(nway_intersection->output_attr());
            const auto& inputs = nway_intersection->input_attrs_and_directions();
            for (const auto& [attr, dir] : inputs) {
                info.attributes.push_back(attr);
                info.attributes.push_back(dir ? "fwd" : "bwd");
            }
        } else if (auto* intersection = dynamic_cast<Intersection<>*>(op)) {
            info.type = "Intersection";
            info.attributes.push_back(intersection->ancestor_attr());
            info.attributes.push_back(intersection->descendant_attr());
            info.attributes.push_back(intersection->output_attr());
            info.attributes.push_back(intersection->is_ancestor_join_fwd() ? "fwd" : "bwd");
            info.attributes.push_back(intersection->is_descendant_join_fwd() ? "fwd" : "bwd");
        } else if (dynamic_cast<SinkPacked*>(op)) {
            info.type = "SinkPacked";
        } else if (dynamic_cast<SinkLinear*>(op)) {
            info.type = "SinkLinear";
        } else {
            info.type = "Unknown";
        }
        chain.push_back(info);
        op = op->next_op;
    }
    return chain;
}

TEST(CycleHandlingTest, SimpleCycle) {
    // Query: a->b,b->c,c->d,a->e,e->f,f->d,a->k
    // Expected: scan(a) -> join(a,k) -> join(a,b) -> flat_join(b,a,e) -> flat_join(e,b,c) -> flat_join(c,e,f) -> nway_intersection(d, [c,f]) -> sink
    std::string query_str =
            "Q(a,k,b,e,c,f,d) := R(a,b),R(b,c),R(c,d),R(a,e),R(e,f),R(f,d),R(a,k)";
    Query query{query_str};
    
    // Ordering: a,k,b,c,d,e,f (acyclic part first: a,k)
    std::vector<std::string> ordering = {"a", "k", "b", "e", "c", "f", "d"};
    
    auto [plan, ftree] = map_ordering_to_plan(query, ordering, SINK_PACKED);
    
    // Verify plan was created
    ASSERT_NE(plan, nullptr);
    ASSERT_NE(ftree, nullptr);
    
    // Extract operator chain with attributes
    auto chain = extract_operator_chain(plan->get_first_op());

    // Verify exact operator sequence and attributes
    ASSERT_EQ(chain.size(), 8) << "Expected exactly 8 operators (Scan, 2 joins, 3 FlatJoins, Intersection, Sink)";
    
    // 0: Scan(a)
    EXPECT_EQ(chain[0].type, "Scan");
    ASSERT_EQ(chain[0].attributes.size(), 1);
    EXPECT_EQ(chain[0].attributes[0], "a");
    
    // 1: INLJoinPacked(a, k, fwd) - acyclic join
    EXPECT_EQ(chain[1].type, "INLJoinPacked");
    ASSERT_EQ(chain[1].attributes.size(), 3);
    EXPECT_EQ(chain[1].attributes[0], "a");
    EXPECT_EQ(chain[1].attributes[1], "k");
    
    // 2: INLJoinPacked(a, b, fwd) - entry to cycle
    EXPECT_EQ(chain[2].type, "INLJoinPacked");
    ASSERT_EQ(chain[2].attributes.size(), 3);
    EXPECT_EQ(chain[2].attributes[0], "a");
    EXPECT_EQ(chain[2].attributes[1], "b");
    
    // 3: FlatJoin(b, a, e, fwd) - cycle edge a->e
    EXPECT_EQ(chain[3].type, "FlatJoin");
    ASSERT_EQ(chain[3].attributes.size(), 4);
    EXPECT_EQ(chain[3].attributes[0], "b");  // parent_attr
    EXPECT_EQ(chain[3].attributes[1], "a");   // lca_attr
    EXPECT_EQ(chain[3].attributes[2], "e");   // output_attr
    
    // 4: FlatJoin(e, b, c, fwd) - cycle edge b->c
    EXPECT_EQ(chain[4].type, "FlatJoin");
    ASSERT_EQ(chain[4].attributes.size(), 4);
    EXPECT_EQ(chain[4].attributes[0], "e");  // parent_attr
    EXPECT_EQ(chain[4].attributes[1], "b");   // lca_attr
    EXPECT_EQ(chain[4].attributes[2], "c");   // output_attr
    
    // 5: FlatJoin(c, e, f, fwd) - cycle edge e->f
    EXPECT_EQ(chain[5].type, "FlatJoin");
    ASSERT_EQ(chain[5].attributes.size(), 4);
    EXPECT_EQ(chain[5].attributes[0], "c");  // parent_attr
    EXPECT_EQ(chain[5].attributes[1], "e");   // lca_attr
    EXPECT_EQ(chain[5].attributes[2], "f");   // output_attr
    
    // 6: Intersection(c, f, d) closes multi-parent node d (two-phase planner uses binary Intersection)
    EXPECT_EQ(chain[6].type, "Intersection");
    ASSERT_EQ(chain[6].attributes.size(), 5);
    EXPECT_EQ(chain[6].attributes[0], "c");   // ancestor_attr
    EXPECT_EQ(chain[6].attributes[1], "f");   // descendant_attr
    EXPECT_EQ(chain[6].attributes[2], "d");   // output_attr
    
    // 7: SinkPacked
    EXPECT_EQ(chain[7].type, "SinkPacked");
}

TEST(CycleHandlingTest, ComplexAcyclicBranching) {
    // Query: a->b,b->c,a->e,c->d,e->d,a->x,x->y,x->z
    // Acyclic part: a->x,x->y,x->z (ONLY these are acyclic)
    // Cycle part: a->b,b->c,a->e,c->d,e->d (d has multiple parents: c and e)
    std::string query_str = "Q(a,x,y,z,b,e,c,d) := "
                            "R(a,b),R(b,c),R(a,e),R(c,d),R(e,d),R(a,x),R(x,y),R(x,z)";
    Query query{query_str};
    
    // Test with different orderings for acyclic part
    // Ordering 1: a,x,y,z,b,c,d,e (acyclic first: a,x,y,z)
    std::vector<std::string> ordering1 = {"a", "x", "y", "z", "b", "e", "c", "d"};
    auto [plan1, ftree1] = map_ordering_to_plan(query, ordering1, SINK_PACKED);
    
    // Ordering 2: a,x,z,y,b,c,d,e (different acyclic ordering: a,x,z,y)
    std::vector<std::string> ordering2 = {"a", "x", "z", "y", "b", "e", "c", "d"};
    auto [plan2, ftree2] = map_ordering_to_plan(query, ordering2, SINK_PACKED);
    
    // Verify both plans were created
    ASSERT_NE(plan1, nullptr);
    ASSERT_NE(plan2, nullptr);
    
    // Extract operator chains with attributes
    auto chain1 = extract_operator_chain(plan1->get_first_op());
    auto chain2 = extract_operator_chain(plan2->get_first_op());

    // Both should start with Scan(a)
    ASSERT_GE(chain1.size(), 1);
    ASSERT_GE(chain2.size(), 1);
    EXPECT_EQ(chain1[0].type, "Scan");
    EXPECT_EQ(chain1[0].attributes[0], "a");
    EXPECT_EQ(chain2[0].type, "Scan");
    EXPECT_EQ(chain2[0].attributes[0], "a");
    
    // Both should end with SinkPacked
    EXPECT_EQ(chain1.back().type, "SinkPacked");
    EXPECT_EQ(chain2.back().type, "SinkPacked");
    
    // Find acyclic joins (should be a->x, x->y, x->z)
    std::vector<OperatorInfo> acyclic_joins1, acyclic_joins2;
    for (const auto& op : chain1) {
        if (op.type == "INLJoinPacked") {
            acyclic_joins1.push_back(op);
        }
    }
    for (const auto& op : chain2) {
        if (op.type == "INLJoinPacked") {
            acyclic_joins2.push_back(op);
        }
    }
    
    // Should have at least 3 acyclic joins (a->x, x->y, x->z)
    EXPECT_GE(acyclic_joins1.size(), 3) << "Plan1 should have at least 3 acyclic joins";
    EXPECT_GE(acyclic_joins2.size(), 3) << "Plan2 should have at least 3 acyclic joins";
    
    // Verify acyclic joins match ordering
    // Plan1: a->x, x->y, x->z
    EXPECT_EQ(acyclic_joins1[0].attributes[0], "a");
    EXPECT_EQ(acyclic_joins1[0].attributes[1], "x");
    EXPECT_EQ(acyclic_joins1[1].attributes[0], "x");
    EXPECT_EQ(acyclic_joins1[1].attributes[1], "y");
    EXPECT_EQ(acyclic_joins1[2].attributes[0], "x");
    EXPECT_EQ(acyclic_joins1[2].attributes[1], "z");
    
    // Plan2: a->x, x->z, x->y (different ordering)
    EXPECT_EQ(acyclic_joins2[0].attributes[0], "a");
    EXPECT_EQ(acyclic_joins2[0].attributes[1], "x");
    EXPECT_EQ(acyclic_joins2[1].attributes[0], "x");
    EXPECT_EQ(acyclic_joins2[1].attributes[1], "z");
    EXPECT_EQ(acyclic_joins2[2].attributes[0], "x");
    EXPECT_EQ(acyclic_joins2[2].attributes[1], "y");
    
    // Find FlatJoin and Intersection operators (for cycle handling; two-phase planner uses binary Intersection)
    std::vector<OperatorInfo> flatjoins1, intersections1;
    std::vector<OperatorInfo> flatjoins2, intersections2;
    
    for (const auto& op : chain1) {
        if (op.type == "FlatJoin") flatjoins1.push_back(op);
        if (op.type == "Intersection") intersections1.push_back(op);
    }
    for (const auto& op : chain2) {
        if (op.type == "FlatJoin") flatjoins2.push_back(op);
        if (op.type == "Intersection") intersections2.push_back(op);
    }
    
    EXPECT_GE(flatjoins1.size(), 1) << "Plan1 should have FlatJoin operators";
    EXPECT_EQ(intersections1.size(), 1) << "Plan1 should have Intersection for multi-parent d";
    EXPECT_GE(flatjoins2.size(), 1) << "Plan2 should have FlatJoin operators";
    EXPECT_EQ(intersections2.size(), 1) << "Plan2 should have Intersection for multi-parent d";
    
    EXPECT_EQ(intersections1[0].attributes[2], "d") << "Intersection should output d";
    EXPECT_EQ(intersections2[0].attributes[2], "d") << "Intersection should output d";
    
    EXPECT_EQ(flatjoins1.size(), flatjoins2.size()) << "Both plans should have same number of FlatJoins";
    EXPECT_EQ(intersections1.size(), intersections2.size()) << "Both plans should have same number of Intersections";
}

TEST(CycleHandlingTest, OddLengthCycle) {
    // Query: a->b, b->c, c->d, a->e, e->d
    // Left path: a->b->c->d (3 edges, odd)
    // Right path: a->e->d (2 edges, even)
    // d has multiple parents: c and e
    std::string query_str = "Q(a,b,e,c,d) := R(a,b),R(b,c),R(c,d),R(a,e),R(e,d)";
    Query query{query_str};
    
    // Ordering: a,b,c,d,e
    std::vector<std::string> ordering = {"a", "b", "e", "c", "d"};
    
    auto [plan, ftree] = map_ordering_to_plan(query, ordering, SINK_PACKED);
    
    ASSERT_NE(plan, nullptr);
    ASSERT_NE(ftree, nullptr);
    
    auto chain = extract_operator_chain(plan->get_first_op());
    
    // Expected: Scan(a) -> INLJoinPacked(a,b) -> FlatJoin(b,a,e) -> FlatJoin(e,b,c) -> Intersection(e,c,d) -> SinkLinear
    EXPECT_GE(chain.size(), 6) << "Chain size: " << chain.size();
    
    if (chain.size() < 6) {
        std::cerr << "ERROR: Chain too short!" << std::endl;
        return;
    }
    
    // 0: Scan(a)
    EXPECT_EQ(chain[0].type, "Scan");
    EXPECT_GE(chain[0].attributes.size(), 1);
    if (chain[0].attributes.size() >= 1) {
        EXPECT_EQ(chain[0].attributes[0], "a");
    }
    
    // 1: INLJoinPacked(a, b, fwd) - entry to cycle
    EXPECT_EQ(chain[1].type, "INLJoinPacked");
    EXPECT_GE(chain[1].attributes.size(), 3);
    if (chain[1].attributes.size() >= 3) {
        EXPECT_EQ(chain[1].attributes[0], "a");
        EXPECT_EQ(chain[1].attributes[1], "b");
    }
    
    // 2: FlatJoin(b, a, e, fwd) - cycle edge a->e
    EXPECT_EQ(chain[2].type, "FlatJoin");
    EXPECT_GE(chain[2].attributes.size(), 4);
    if (chain[2].attributes.size() >= 4) {
        EXPECT_EQ(chain[2].attributes[0], "b");  // parent_attr
        EXPECT_EQ(chain[2].attributes[1], "a");   // lca_attr
        EXPECT_EQ(chain[2].attributes[2], "e");   // output_attr
    }
    
    // 3: FlatJoin(e, b, c, fwd) - cycle edge b->c
    EXPECT_EQ(chain[3].type, "FlatJoin");
    EXPECT_GE(chain[3].attributes.size(), 4);
    if (chain[3].attributes.size() >= 4) {
        EXPECT_EQ(chain[3].attributes[0], "e");  // parent_attr
        EXPECT_EQ(chain[3].attributes[1], "b");   // lca_attr
        EXPECT_EQ(chain[3].attributes[2], "c");   // output_attr
    }
    
    // 4: Intersection(e, c, d) - multi-parent node d
    EXPECT_EQ(chain[4].type, "Intersection");
    ASSERT_EQ(chain[4].attributes.size(), 5);
    EXPECT_EQ(chain[4].attributes[0], "e");
    EXPECT_EQ(chain[4].attributes[1], "c");
    EXPECT_EQ(chain[4].attributes[2], "d");
    
    // Last: SinkLinear (linear packed plan)
    EXPECT_EQ(chain.back().type, "SinkLinear");
}

TEST(CycleHandlingTest, CompletelyCyclic) {
    // Query: a->b, b->c, c->a (simple triangle cycle)
    // No acyclic parts, everything is part of the cycle
    std::string query_str = "Q(a,b,c) := R(a,b),R(b,c),R(c,a)";
    Query query{query_str};
    
    // Ordering: a,b,c
    std::vector<std::string> ordering = {"a", "b", "c"};
    
    auto [plan, ftree] = map_ordering_to_plan(query, ordering, SINK_PACKED);
    
    ASSERT_NE(plan, nullptr);
    ASSERT_NE(ftree, nullptr);
    
    auto chain = extract_operator_chain(plan->get_first_op());
    
    // Expected: Scan(a) -> INLJoinPacked(a,b) -> Intersection(a,b,c) -> SinkPacked
    // Note: c->a edge closes the cycle but may not need explicit operator if handled differently
    ASSERT_GE(chain.size(), 4);
    
    // 0: Scan(a)
    EXPECT_EQ(chain[0].type, "Scan");
    ASSERT_EQ(chain[0].attributes.size(), 1);
    EXPECT_EQ(chain[0].attributes[0], "a");
    
    // 1: INLJoinPacked(a, b, fwd) - entry to cycle
    EXPECT_EQ(chain[1].type, "INLJoinPacked");
    ASSERT_EQ(chain[1].attributes.size(), 3);
    EXPECT_EQ(chain[1].attributes[0], "a");
    EXPECT_EQ(chain[1].attributes[1], "b");
    
    // 2: Intersection(a, b, c) closes triangle (two-phase planner)
    EXPECT_EQ(chain[2].type, "Intersection");
    ASSERT_EQ(chain[2].attributes.size(), 5);
    EXPECT_EQ(chain[2].attributes[0], "a");
    EXPECT_EQ(chain[2].attributes[1], "b");
    EXPECT_EQ(chain[2].attributes[2], "c");
    
    // Linear packed pipeline uses SinkLinear
    EXPECT_EQ(chain.back().type, "SinkLinear");
}

TEST(CycleHandlingTest, CompletelyAcyclic) {
    // Query: a->b, b->c, a->d, d->e (pure tree, no cycles)
    // This should use only Scan and INLJoinPacked, no FlatJoin or Intersection
    std::string query_str = "Q(a,b,c,d,e) := R(a,b),R(b,c),R(a,d),R(d,e)";
    Query query{query_str};
    
    // Ordering: a,b,c,d,e
    std::vector<std::string> ordering = {"a", "b", "c", "d", "e"};
    
    auto [plan, ftree] = map_ordering_to_plan(query, ordering, SINK_PACKED);
    
    ASSERT_NE(plan, nullptr);
    ASSERT_NE(ftree, nullptr);
    
    auto chain = extract_operator_chain(plan->get_first_op());
    
    // Expected: Scan(a) -> INLJoinPacked(a,b) -> INLJoinPacked(b,c) -> INLJoinPacked(a,d) -> INLJoinPacked(d,e) -> SinkPacked
    ASSERT_EQ(chain.size(), 6) << "Should have Scan, 4 joins, and Sink";
    
    // 0: Scan(a)
    EXPECT_EQ(chain[0].type, "Scan");
    ASSERT_EQ(chain[0].attributes.size(), 1);
    EXPECT_EQ(chain[0].attributes[0], "a");
    
    // 1: INLJoinPacked(a, b, fwd) - first child of a
    EXPECT_EQ(chain[1].type, "INLJoinPacked");
    ASSERT_EQ(chain[1].attributes.size(), 3);
    EXPECT_EQ(chain[1].attributes[0], "a");
    EXPECT_EQ(chain[1].attributes[1], "b");
    
    // 2: INLJoinPacked(b, c, fwd)
    EXPECT_EQ(chain[2].type, "INLJoinPacked");
    ASSERT_EQ(chain[2].attributes.size(), 3);
    EXPECT_EQ(chain[2].attributes[0], "b");
    EXPECT_EQ(chain[2].attributes[1], "c");
    
    // 3: INLJoinPacked(a d, fwd)
    EXPECT_EQ(chain[3].type, "INLJoinPacked");
    ASSERT_EQ(chain[3].attributes.size(), 3);
    EXPECT_EQ(chain[3].attributes[0], "a");
    EXPECT_EQ(chain[3].attributes[1], "d");
    
    // 4: Last join before sink is cascaded (two-phase planner)
    EXPECT_EQ(chain[4].type, "INLJoinPackedCascade");
    ASSERT_EQ(chain[4].attributes.size(), 3);
    EXPECT_EQ(chain[4].attributes[0], "d");
    EXPECT_EQ(chain[4].attributes[1], "e");
    
    // Verify no FlatJoin or Intersection operators
    for (const auto& op : chain) {
        EXPECT_NE(op.type, "FlatJoin") << "Acyclic query should not have FlatJoin operators";
        EXPECT_NE(op.type, "NWayIntersection") << "Acyclic query should not have NWayIntersection operators";
        EXPECT_NE(op.type, "Intersection") << "Acyclic query should not have Intersection operators";
    }
    
    // Last: SinkPacked
    EXPECT_EQ(chain[5].type, "SinkPacked");
}

TEST(CycleHandlingTest, CompletelyAcyclicWithCascade) {
    // Query: a->b, b->c, a->d, d->e (pure tree, no cycles)
    // This should use only Scan and INLJoinPacked (with Cascades), no FlatJoin or Intersection
    std::string query_str = "Q(a,b,c,d,e) := R(a,b),R(b,c),R(a,d),R(d,e)";
    Query query{query_str};
    std::vector<std::string> ordering = {"a", "b", "c", "_cd", "d", "e", "_cd"};

    auto [plan, ftree] = map_ordering_to_plan(query, ordering, SINK_PACKED);

    ASSERT_NE(plan, nullptr);
    ASSERT_NE(ftree, nullptr);

    auto chain = extract_operator_chain(plan->get_first_op());

    // Expected: Scan(a) -> INLJoinPacked(a,b) -> INLJoinPacked(b,c) -> INLJoinPacked(a,d) -> INLJoinPacked(d,e) -> SinkPacked
    ASSERT_EQ(chain.size(), 6) << "Should have Scan, 4 joins, and Sink";

    // 0: Scan(a)
    EXPECT_EQ(chain[0].type, "Scan");
    ASSERT_EQ(chain[0].attributes.size(), 1);
    EXPECT_EQ(chain[0].attributes[0], "a");

    // 1: INLJoinPacked(a, b, fwd) - first child of a
    EXPECT_EQ(chain[1].type, "INLJoinPacked");
    ASSERT_EQ(chain[1].attributes.size(), 3);
    EXPECT_EQ(chain[1].attributes[0], "a");
    EXPECT_EQ(chain[1].attributes[1], "b");

    // 2: INLJoinPacked(b, c, fwd)
    EXPECT_EQ(chain[2].type, "INLJoinPackedGPCascade");
    ASSERT_EQ(chain[2].attributes.size(), 3);
    EXPECT_EQ(chain[2].attributes[0], "b");
    EXPECT_EQ(chain[2].attributes[1], "c");

    // 3: INLJoinPacked(a d, fwd)
    EXPECT_EQ(chain[3].type, "INLJoinPacked");
    ASSERT_EQ(chain[3].attributes.size(), 3);
    EXPECT_EQ(chain[3].attributes[0], "a");
    EXPECT_EQ(chain[3].attributes[1], "d");

    // 4: INLJoinPacked(d, e, fwd) - child of d
    EXPECT_EQ(chain[4].type, "INLJoinPackedGPCascade");
    ASSERT_EQ(chain[4].attributes.size(), 3);
    EXPECT_EQ(chain[4].attributes[0], "d");
    EXPECT_EQ(chain[4].attributes[1], "e");

    // Verify no FlatJoin or Intersection operators
    for (const auto& op : chain) {
        EXPECT_NE(op.type, "FlatJoin") << "Acyclic query should not have FlatJoin operators";
        EXPECT_NE(op.type, "NWayIntersection") << "Acyclic query should not have NWayIntersection operators";
        EXPECT_NE(op.type, "Intersection") << "Acyclic query should not have Intersection operators";
    }

    // Last: SinkPacked
    EXPECT_EQ(chain[5].type, "SinkPacked");
}

TEST(CycleHandlingTest, SimpleCycleSynchronized) {
    // Test synchronized version of SimpleCycle
    // Query: a->b,b->c,c->d,a->e,e->f,f->d,a->k
    std::string query_str =
            "Q(a,k,b,e,c,f,d) := R(a,b),R(b,c),R(c,d),R(a,e),R(e,f),R(f,d),R(a,k)";
    Query query{query_str};
    
    // Ordering: a,k,b,e,c,f,d (acyclic part first: a,k)
    std::vector<std::string> ordering = {"a", "k", "b", "e", "c", "f", "d"};
    
    uint64_t start_id = 0;
    auto [plan, ftree] = map_ordering_to_plan_synchronized(query, ordering, SINK_PACKED, start_id);
    
    // Verify plan was created
    ASSERT_NE(plan, nullptr);
    ASSERT_NE(ftree, nullptr);
    
    // Extract operator chain with attributes
    auto chain = extract_operator_chain(plan->get_first_op());
    
    // Verify exact operator sequence and attributes
    ASSERT_EQ(chain.size(), 8) << "Expected exactly 8 operators (ScanSynchronized, 2 joins, 3 FlatJoins, Intersection, Sink)";
    
    // 0: ScanSynchronized(a)
    EXPECT_EQ(chain[0].type, "ScanSynchronized");
    ASSERT_EQ(chain[0].attributes.size(), 1);
    EXPECT_EQ(chain[0].attributes[0], "a");
    
    // 1: INLJoinPacked(a, k, fwd) - acyclic join
    EXPECT_EQ(chain[1].type, "INLJoinPacked");
    ASSERT_EQ(chain[1].attributes.size(), 3);
    EXPECT_EQ(chain[1].attributes[0], "a");
    EXPECT_EQ(chain[1].attributes[1], "k");
    EXPECT_EQ(chain[1].attributes[2], "fwd");
    
    // 2: INLJoinPacked(a, b, fwd) - entry join for cyclic attribute
    EXPECT_EQ(chain[2].type, "INLJoinPacked");
    ASSERT_EQ(chain[2].attributes.size(), 3);
    EXPECT_EQ(chain[2].attributes[0], "a");
    EXPECT_EQ(chain[2].attributes[1], "b");
    EXPECT_EQ(chain[2].attributes[2], "fwd");
    
    // 3: FlatJoin(b, a, e) - cyclic join
    EXPECT_EQ(chain[3].type, "FlatJoin");
    ASSERT_EQ(chain[3].attributes.size(), 4);
    EXPECT_EQ(chain[3].attributes[0], "b");
    EXPECT_EQ(chain[3].attributes[1], "a");
    EXPECT_EQ(chain[3].attributes[2], "e");
    
    // 4: FlatJoin(e, b, c) - cyclic join
    EXPECT_EQ(chain[4].type, "FlatJoin");
    ASSERT_EQ(chain[4].attributes.size(), 4);
    EXPECT_EQ(chain[4].attributes[0], "e");
    EXPECT_EQ(chain[4].attributes[1], "b");
    EXPECT_EQ(chain[4].attributes[2], "c");
    
    // 5: FlatJoin(c, e, f) - cyclic join
    EXPECT_EQ(chain[5].type, "FlatJoin");
    ASSERT_EQ(chain[5].attributes.size(), 4);
    EXPECT_EQ(chain[5].attributes[0], "c");
    EXPECT_EQ(chain[5].attributes[1], "e");
    EXPECT_EQ(chain[5].attributes[2], "f");
    
    // 6: Intersection(c, f, d) - last cyclic attribute
    EXPECT_EQ(chain[6].type, "Intersection");
    ASSERT_EQ(chain[6].attributes.size(), 5);
    EXPECT_EQ(chain[6].attributes[0], "c");
    EXPECT_EQ(chain[6].attributes[1], "f");
    EXPECT_EQ(chain[6].attributes[2], "d");
    
    // 7: SinkPacked
    EXPECT_EQ(chain[7].type, "SinkPacked");
}

TEST(CycleHandlingTest, SimpleCycleSynchronizedWithStartId) {
    // Test synchronized version with different start_id
    std::string query_str =
            "Q(a,k,b,e,c,f,d) := R(a,b),R(b,c),R(c,d),R(a,e),R(e,f),R(f,d),R(a,k)";
    Query query{query_str};
    
    std::vector<std::string> ordering = {"a", "k", "b", "e", "c", "f", "d"};
    
    // Test with start_id=1
    uint64_t start_id = 1;
    auto [plan, ftree] = map_ordering_to_plan_synchronized(query, ordering, SINK_PACKED, start_id);
    
    // Verify plan was created
    ASSERT_NE(plan, nullptr);
    ASSERT_NE(ftree, nullptr);
    
    // Verify that the first operator is ScanSynchronized
    auto* scan_sync = dynamic_cast<ScanSynchronized*>(plan->get_first_op());
    ASSERT_NE(scan_sync, nullptr);
    EXPECT_EQ(scan_sync->attribute(), "a");
    
    // Extract operator chain
    auto chain = extract_operator_chain(plan->get_first_op());
    
    // Verify we have ScanSynchronized instead of Scan
    EXPECT_EQ(chain[0].type, "ScanSynchronized");
    EXPECT_EQ(chain[0].attributes[0], "a");
    
    // Verify the rest of the chain is the same
    ASSERT_EQ(chain.size(), 8);
    EXPECT_EQ(chain[1].type, "INLJoinPacked");
    EXPECT_EQ(chain[2].type, "INLJoinPacked");
    EXPECT_EQ(chain[3].type, "FlatJoin");
    EXPECT_EQ(chain[4].type, "FlatJoin");
    EXPECT_EQ(chain[5].type, "FlatJoin");
    EXPECT_EQ(chain[6].type, "Intersection");
    EXPECT_EQ(chain[7].type, "SinkPacked");
}

TEST(CycleHandlingTest, AcyclicQuerySynchronized) {
    // Test synchronized version of acyclic query
    std::string query_str = "Q(a,b,c,d) := R(a,b),R(b,c),R(c,d)";
    Query query{query_str};
    
    std::vector<std::string> ordering = {"a", "b", "c", "d"};
    
    uint64_t start_id = 0;
    auto [plan, ftree] = map_ordering_to_plan_synchronized(query, ordering, SINK_PACKED, start_id);
    
    // Verify plan was created
    ASSERT_NE(plan, nullptr);
    ASSERT_NE(ftree, nullptr);
    
    // Extract operator chain
    auto chain = extract_operator_chain(plan->get_first_op());
    
    // Verify we have ScanSynchronized
    EXPECT_EQ(chain[0].type, "ScanSynchronized");
    EXPECT_EQ(chain[0].attributes[0], "a");
    
    // Verify no cyclic operators
    for (const auto& op : chain) {
        EXPECT_NE(op.type, "FlatJoin") << "Acyclic query should not have FlatJoin operators";
        EXPECT_NE(op.type, "NWayIntersection") << "Acyclic query should not have NWayIntersection operators";
        EXPECT_NE(op.type, "Intersection") << "Acyclic query should not have Intersection operators";
    }
    
    // Verify we have joins and sink
    ASSERT_GE(chain.size(), 3);
    EXPECT_EQ(chain.back().type, "SinkLinear");
}


