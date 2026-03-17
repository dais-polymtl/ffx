#ifndef VFENGINE_PLAN_TREE_HPP
#define VFENGINE_PLAN_TREE_HPP

#include "../../operator/include/predicate/predicate_eval.hpp"
#include "../../query/include/predicate.hpp"
#include "../../query/include/query.hpp"
#include "../../table/include/cardinality.hpp"
#include "factorized_ftree/factorized_tree_element.hpp"
#include "plan.hpp"
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace ffx {

// =============================================================================
// Plan Node Types
// =============================================================================

enum class PlanNodeType {
    // Scans
    SCAN,
    SCAN_PREDICATED,
    SCAN_SYNCHRONIZED,
    SCAN_SYNCHRONIZED_PREDICATED,
    SCAN_UNPACKED,
    SCAN_UNPACKED_SYNCHRONIZED,

    // INL Joins
    INL_JOIN_PACKED,
    INL_JOIN_UNPACKED,
    INL_JOIN_PACKED_PREDICATED,

    // INL Joins (Shared State)
    INL_JOIN_PACKED_SHARED,
    INL_JOIN_PACKED_PREDICATED_SHARED,

    // Cascade Joins (for cyclic queries)
    INL_JOIN_PACKED_CASCADE,
    INL_JOIN_PACKED_CASCADE_PREDICATED,
    INL_JOIN_PACKED_GP_CASCADE,
    INL_JOIN_PACKED_GP_CASCADE_PREDICATED,

    // Cascade Joins (Shared State)
    INL_JOIN_PACKED_CASCADE_SHARED,
    INL_JOIN_PACKED_CASCADE_PREDICATED_SHARED,
    INL_JOIN_PACKED_GP_CASCADE_SHARED,
    INL_JOIN_PACKED_GP_CASCADE_PREDICATED_SHARED,

    // Flat Joins
    FLAT_JOIN,
    FLAT_JOIN_PREDICATED,

    // Intersections
    INTERSECTION,
    INTERSECTION_PREDICATED,
    NWAY_INTERSECTION,
    NWAY_INTERSECTION_PREDICATED,

    // Filters (attribute-to-attribute predicates)
    PACKED_THETA_JOIN,
    PACKED_ANTI_SEMI_JOIN,// NOT EXISTS filter

    // Sinks
    SINK_PACKED,
    SINK_LINEAR,
    SINK_MIN,
    SINK_NOOP
};

// Convert PlanNodeType to string for debugging
std::string plan_node_type_to_string(PlanNodeType type);

// =============================================================================
// Theta Join Info (attribute-to-attribute predicate)
// =============================================================================

struct ThetaJoinInfo {
    std::string ancestor_attr;  // Left side (must be ancestor in ftree)
    std::string descendant_attr;// Right side (must be descendant in ftree)
    PredicateOp op;             // Comparison operator

    ThetaJoinInfo(std::string ancestor, std::string descendant, PredicateOp op)
        : ancestor_attr(std::move(ancestor)), descendant_attr(std::move(descendant)), op(op) {}
};

// =============================================================================
// Anti-Semi-Join Info (NOT EXISTS predicate)
// =============================================================================

struct AntiSemiJoinInfo {
    std::string ancestor_attr;  // Source node (must be ancestor in ftree)
    std::string descendant_attr;// Target node (must be descendant in ftree)

    AntiSemiJoinInfo(std::string ancestor, std::string descendant)
        : ancestor_attr(std::move(ancestor)), descendant_attr(std::move(descendant)) {}
};

// =============================================================================
// Property Filter Info (property attribute predicate via adj_list lookup)
// =============================================================================

struct PropertyFilterInfo {
    std::string property_attr; // The property attribute (e.g., "kind")
    std::string entity_attr;   // The entity to look up from (e.g., "ct")
    bool is_fwd;               // Direction of adjacency list lookup
};

// =============================================================================
// Plan Node - Temporary tree structure for plan building
// =============================================================================

struct PlanNode {
    // Node identification
    std::string attribute;
    PlanNodeType type;

    // Join info
    std::string join_from_attr;// For joins: the attribute we're joining from
    std::string lca_attr;      // For flat joins: the LCA attribute
    bool is_fwd = true;        // Direction of the join

    // For N-way intersection
    std::vector<std::pair<std::string, bool>> intersection_inputs;// (attr, is_fwd)

    // Scalar predicates to apply at this node (attr <op> value)
    PredicateExpression scalar_predicates;

    // Theta joins to insert AFTER this node (ancestor <op> descendant)
    // The descendant_attr should match this node's attribute
    std::vector<ThetaJoinInfo> theta_joins_after;

    // Anti-semi-joins to insert AFTER this node (NOT EXISTS)
    // The descendant_attr should match this node's attribute
    std::vector<AntiSemiJoinInfo> anti_semi_joins_after;

    // Property filters: property attributes looked up via adj_list with predicates
    std::vector<PropertyFilterInfo> property_filter_infos;
    PredicateExpression property_predicate_expr;

    // Tree structure
    std::vector<std::unique_ptr<PlanNode>> children;
    PlanNode* parent = nullptr;

    // Metadata
    uint32_t depth = 0;
    bool is_cyclic = false;

    // Cardinality info for state sharing decision
    Cardinality table_cardinality = Cardinality::MANY_TO_MANY;// Default to m:n

    // Constructors
    PlanNode() = default;
    PlanNode(std::string attr, PlanNodeType t) : attribute(std::move(attr)), type(t) {}

    // Check if this node has scalar predicates
    bool has_scalar_predicates() const { return scalar_predicates.has_predicates(); }

    // Check if this node needs theta joins after it
    bool has_theta_joins() const { return !theta_joins_after.empty(); }

    // Check if this node needs anti-semi-joins after it
    bool has_anti_semi_joins() const { return !anti_semi_joins_after.empty(); }

    // Check if this node has property filters
    bool has_property_filters() const { return !property_filter_infos.empty(); }

    // Add a child node
    PlanNode* add_child(std::unique_ptr<PlanNode> child);

    // Find a node by attribute name (recursive search)
    PlanNode* find_node(const std::string& attr);
    const PlanNode* find_node(const std::string& attr) const;

    // Print the tree for debugging
    void print(int indent = 0) const;
};

// =============================================================================
// Plan Tree - Container for the entire plan
// =============================================================================

class PlanTree {
public:
    PlanTree() = default;

    // Set the root node
    void set_root(std::unique_ptr<PlanNode> root);

    // Get the root node
    PlanNode* get_root() { return _root.get(); }
    const PlanNode* get_root() const { return _root.get(); }

    // Find a node by attribute name
    PlanNode* find_node(const std::string& attr);
    const PlanNode* find_node(const std::string& attr) const;

    // Add a theta join constraint (will be placed optimally in the tree)
    void add_theta_join(const std::string& ancestor_attr, const std::string& descendant_attr, PredicateOp op);

    // Add an anti-semi-join constraint (NOT EXISTS)
    void add_anti_semi_join(const std::string& ancestor_attr, const std::string& descendant_attr);

    // Add a scalar predicate for an attribute
    void add_scalar_predicate(const std::string& attr, PredicateOp op, uint64_t value);

    // Finalize the tree - update node types based on predicates
    void finalize();

    // Print the tree for debugging
    void print() const;

    // Set sink type
    void set_sink_type(sink_type sink) { _sink_type = sink; }
    sink_type get_sink_type() const { return _sink_type; }

    // Get sorted theta joins (sorted by first operand position in ordering)
    const std::vector<ThetaJoinInfo>& get_sorted_theta_joins() const { return _sorted_theta_joins; }
    std::vector<ThetaJoinInfo>& get_sorted_theta_joins() { return _sorted_theta_joins; }

private:
    std::unique_ptr<PlanNode> _root;
    sink_type _sink_type = SINK_PACKED;
    std::vector<ThetaJoinInfo> _sorted_theta_joins;// Theta joins sorted by first operand position

    // Helper: Check if attr1 is an ancestor of attr2 in the tree
    bool is_ancestor_of(const std::string& ancestor, const std::string& descendant) const;

    // Helper: Update node types to predicated versions where needed
    void update_node_types(PlanNode* node);
};

// =============================================================================
// Phase 1: Build Plan Tree from Query
// =============================================================================

/**
 * Build a plan tree from a query and ordering.
 * This is Phase 1 of the two-phase plan creation.
 * 
 * @param query The parsed query
 * @param ordering The attribute ordering
 * @param sink The sink type
 * @return The plan tree (temporary structure)
 */
std::unique_ptr<PlanTree> build_plan_tree(const Query& query, const std::vector<std::string>& ordering, sink_type sink);

// =============================================================================
// Phase 2: Create Operators from Plan Tree
// =============================================================================

/**
 * Create the actual operator pipeline from a plan tree.
 * This is Phase 2 of the two-phase plan creation.
 * 
 * @param plan_tree The finalized plan tree
 * @return Pair of (Plan, FactorizedTreeElement root)
 */
std::pair<std::unique_ptr<Plan>, std::shared_ptr<FactorizedTreeElement>>
create_operators_from_plan_tree(const PlanTree& plan_tree, const Query& query,
                                const std::vector<std::string>& original_ordering,
                                const std::vector<const Table*>& tables);

// =============================================================================
// Ordering → plan (two-phase: build_plan_tree + create_operators_from_plan_tree)
// =============================================================================

class Table;

std::vector<std::string> enumerate_orderings(const std::string& query_str);

std::pair<std::unique_ptr<Plan>, std::shared_ptr<FactorizedTreeElement>> map_ordering_to_plan(const Query& query,
                                                                const std::vector<std::string>& ordering, sink_type sink,
                                                                const std::vector<const Table*>& tables = {});

std::pair<std::unique_ptr<Plan>, std::shared_ptr<FactorizedTreeElement>> map_ordering_to_plan_synchronized(const Query&
                                                        query, const std::vector<std::string>& ordering, sink_type sink,
                                                        uint64_t start_id,
                                                        const std::vector<const Table*>& tables = {});

}// namespace ffx

#endif// VFENGINE_PLAN_TREE_HPP
