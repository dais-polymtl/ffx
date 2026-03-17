#include "include/plan_tree.hpp"
#include "../operator/include/factorized_ftree/factorized_tree.hpp"
#include "join/flat_join.hpp"
#include "join/flat_join_predicated.hpp"
#include "join/inljoin_packed.hpp"
#include "join/inljoin_packed_cascade.hpp"
#include "join/inljoin_packed_cascade_predicated.hpp"
#include "join/inljoin_packed_cascade_predicated_shared.hpp"
#include "join/inljoin_packed_cascade_shared.hpp"
#include "join/inljoin_packed_gp_cascade.hpp"
#include "join/inljoin_packed_gp_cascade_predicated.hpp"
#include "join/inljoin_packed_gp_cascade_predicated_shared.hpp"
#include "join/inljoin_packed_gp_cascade_shared.hpp"
#include "join/inljoin_packed_predicated.hpp"
#include "join/inljoin_packed_predicated_shared.hpp"
#include "join/inljoin_packed_shared.hpp"
#include "join/inljoin_unpacked.hpp"
#include "join/intersection.hpp"
#include "join/intersection_predicated.hpp"
#include "join/nway_intersection.hpp"
#include "join/nway_intersection_predicated.hpp"
#include "join/packed_anti_semi_join.hpp"
#include "join/packed_theta_join.hpp"
#include "join/property_filter.hpp"
#include "scan/scan.hpp"
#include "scan/scan_predicated.hpp"
#include "scan/scan_synchronized.hpp"
#include "scan/scan_synchronized_predicated.hpp"
#include "scan/scan_unpacked.hpp"
#include "scan/scan_unpacked_synchronized.hpp"
#include "sink/sink_linear.hpp"
#include "sink/sink_export.hpp"
#include "sink/sink_min.hpp"
#include "sink/sink_min_itr.hpp"
#include "sink/sink_no_op.hpp"
#include "ai_operator/map.hpp"
#include "sink/sink_packed.hpp"
#include "sink/sink_unpacked.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace ffx {

std::string plan_node_type_to_string(PlanNodeType type) {
    switch (type) {
        case PlanNodeType::SCAN:
            return "SCAN";
        case PlanNodeType::SCAN_PREDICATED:
            return "SCAN_PREDICATED";
        case PlanNodeType::SCAN_SYNCHRONIZED:
            return "SCAN_SYNCHRONIZED";
        case PlanNodeType::SCAN_SYNCHRONIZED_PREDICATED:
            return "SCAN_SYNCHRONIZED_PREDICATED";
        case PlanNodeType::SCAN_UNPACKED:
            return "SCAN_UNPACKED";
        case PlanNodeType::SCAN_UNPACKED_SYNCHRONIZED:
            return "SCAN_UNPACKED_SYNCHRONIZED";
        case PlanNodeType::INL_JOIN_PACKED:
            return "INL_JOIN_PACKED";
        case PlanNodeType::INL_JOIN_UNPACKED:
            return "INL_JOIN_UNPACKED";
        case PlanNodeType::INL_JOIN_PACKED_PREDICATED:
            return "INL_JOIN_PACKED_PREDICATED";
        case PlanNodeType::INL_JOIN_PACKED_SHARED:
            return "INL_JOIN_PACKED_SHARED";
        case PlanNodeType::INL_JOIN_PACKED_PREDICATED_SHARED:
            return "INL_JOIN_PACKED_PREDICATED_SHARED";
        case PlanNodeType::INL_JOIN_PACKED_CASCADE:
            return "INL_JOIN_PACKED_CASCADE";
        case PlanNodeType::INL_JOIN_PACKED_CASCADE_PREDICATED:
            return "INL_JOIN_PACKED_CASCADE_PREDICATED";
        case PlanNodeType::INL_JOIN_PACKED_CASCADE_SHARED:
            return "INL_JOIN_PACKED_CASCADE_SHARED";
        case PlanNodeType::INL_JOIN_PACKED_CASCADE_PREDICATED_SHARED:
            return "INL_JOIN_PACKED_CASCADE_PREDICATED_SHARED";
        case PlanNodeType::INL_JOIN_PACKED_GP_CASCADE:
            return "INL_JOIN_PACKED_GP_CASCADE";
        case PlanNodeType::INL_JOIN_PACKED_GP_CASCADE_PREDICATED:
            return "INL_JOIN_PACKED_GP_CASCADE_PREDICATED";
        case PlanNodeType::INL_JOIN_PACKED_GP_CASCADE_SHARED:
            return "INL_JOIN_PACKED_GP_CASCADE_SHARED";
        case PlanNodeType::INL_JOIN_PACKED_GP_CASCADE_PREDICATED_SHARED:
            return "INL_JOIN_PACKED_GP_CASCADE_PREDICATED_SHARED";
        case PlanNodeType::FLAT_JOIN:
            return "FLAT_JOIN";
        case PlanNodeType::FLAT_JOIN_PREDICATED:
            return "FLAT_JOIN_PREDICATED";
        case PlanNodeType::INTERSECTION:
            return "INTERSECTION";
        case PlanNodeType::INTERSECTION_PREDICATED:
            return "INTERSECTION_PREDICATED";
        case PlanNodeType::NWAY_INTERSECTION:
            return "NWAY_INTERSECTION";
        case PlanNodeType::NWAY_INTERSECTION_PREDICATED:
            return "NWAY_INTERSECTION_PREDICATED";
        case PlanNodeType::PACKED_THETA_JOIN:
            return "PACKED_THETA_JOIN";
        case PlanNodeType::PACKED_ANTI_SEMI_JOIN:
            return "PACKED_ANTI_SEMI_JOIN";
        case PlanNodeType::SINK_PACKED:
            return "SINK_PACKED";
        case PlanNodeType::SINK_LINEAR:
            return "SINK_LINEAR";
        case PlanNodeType::SINK_MIN:
            return "SINK_MIN";
        case PlanNodeType::SINK_NOOP:
            return "SINK_NOOP";
        default:
            return "UNKNOWN";
    }
}

PlanNode* PlanNode::add_child(std::unique_ptr<PlanNode> child) {
    child->parent = this;
    child->depth = this->depth + 1;
    children.push_back(std::move(child));
    return children.back().get();
}

PlanNode* PlanNode::find_node(const std::string& attr) {
    if (attribute == attr) return this;
    for (auto& child: children) {
        if (auto* found = child->find_node(attr)) { return found; }
    }
    return nullptr;
}

const PlanNode* PlanNode::find_node(const std::string& attr) const {
    if (attribute == attr) return this;
    for (const auto& child: children) {
        if (const auto* found = child->find_node(attr)) { return found; }
    }
    return nullptr;
}

void PlanNode::print(int indent) const {
    const std::string ind(indent * 2, ' ');
    std::cout << ind << "- " << attribute << " [" << plan_node_type_to_string(type) << "]";
    if (!join_from_attr.empty()) { std::cout << " join_from=" << join_from_attr; }
    if (!lca_attr.empty()) { std::cout << " lca=" << lca_attr; }
    if (!intersection_inputs.empty()) {
        std::cout << " inputs=[";
        for (const auto& [attr, fwd]: intersection_inputs) {
            std::cout << attr << "(" << (fwd ? "fwd" : "bwd") << ") ";
        }
        std::cout << "]";
    }
    if (has_scalar_predicates()) { std::cout << " [has_scalar_preds]"; }
    if (has_theta_joins()) { std::cout << " [theta_joins=" << theta_joins_after.size() << "]"; }
    std::cout << std::endl;

    for (const auto& child: children) {
        child->print(indent + 1);
    }
}

void PlanTree::set_root(std::unique_ptr<PlanNode> root) { _root = std::move(root); }

PlanNode* PlanTree::find_node(const std::string& attr) { return _root ? _root->find_node(attr) : nullptr; }

const PlanNode* PlanTree::find_node(const std::string& attr) const { return _root ? _root->find_node(attr) : nullptr; }

bool PlanTree::is_ancestor_of(const std::string& ancestor, const std::string& descendant) const {
    const PlanNode* desc_node = find_node(descendant);
    if (!desc_node) return false;

    // Walk up from descendant to see if we reach ancestor
    const PlanNode* current = desc_node->parent;
    while (current) {
        if (current->attribute == ancestor) return true;
        current = current->parent;
    }
    return false;
}

void PlanTree::add_theta_join(const std::string& ancestor_attr, const std::string& descendant_attr, PredicateOp op) {
    // Find the descendant node - theta join is placed AFTER the descendant
    PlanNode* desc_node = find_node(descendant_attr);
    if (!desc_node) {
        std::cerr << "Warning: Cannot add theta join - descendant attr '" << descendant_attr
                  << "' not found in plan tree" << std::endl;
        return;
    }

    // Verify ancestor exists and is actually an ancestor
    if (!is_ancestor_of(ancestor_attr, descendant_attr)) {
        std::cerr << "Warning: Cannot add theta join - '" << ancestor_attr << "' is not an ancestor of '"
                  << descendant_attr << "'" << std::endl;
        return;
    }

    desc_node->theta_joins_after.emplace_back(ancestor_attr, descendant_attr, op);
}

void PlanTree::add_anti_semi_join(const std::string& ancestor_attr, const std::string& descendant_attr) {
    // Find the descendant node - anti-semi-join is placed AFTER the descendant
    PlanNode* desc_node = find_node(descendant_attr);
    if (!desc_node) {
        std::cerr << "Warning: Cannot add anti-semi-join - descendant attr '" << descendant_attr
                  << "' not found in plan tree" << std::endl;
        return;
    }

    // Verify ancestor exists and is actually an ancestor
    if (!is_ancestor_of(ancestor_attr, descendant_attr)) {
        std::cerr << "Warning: Cannot add anti-semi-join - '" << ancestor_attr << "' is not an ancestor of '"
                  << descendant_attr << "'" << std::endl;
        return;
    }

    desc_node->anti_semi_joins_after.emplace_back(ancestor_attr, descendant_attr);
}

void PlanTree::add_scalar_predicate(const std::string& attr, PredicateOp op, uint64_t value) {
    PlanNode* node = find_node(attr);
    if (!node) {
        std::cerr << "Warning: Cannot add scalar predicate - attr '" << attr << "' not found in plan tree" << std::endl;
        return;
    }

    // Add to the node's scalar predicates
    Predicate pred(op, attr, std::to_string(value));

    if (node->scalar_predicates.groups.empty()) {
        // Create first group (AND group)
        PredicateGroup group;
        group.predicates.push_back(pred);
        group.op = LogicalOp::AND;
        node->scalar_predicates.groups.push_back(group);
    } else {
        // Add to first group (assuming AND logic for now)
        node->scalar_predicates.groups[0].predicates.push_back(pred);
    }
}

void PlanTree::update_node_types(PlanNode* node) {
    if (!node) return;

    // Update type based on scalar predicates
    if (node->has_scalar_predicates()) {
        switch (node->type) {
            case PlanNodeType::SCAN:
                node->type = PlanNodeType::SCAN_PREDICATED;
                break;
            case PlanNodeType::SCAN_SYNCHRONIZED:
                node->type = PlanNodeType::SCAN_SYNCHRONIZED_PREDICATED;
                break;
            case PlanNodeType::INL_JOIN_PACKED:
                node->type = PlanNodeType::INL_JOIN_PACKED_PREDICATED;
                break;
            case PlanNodeType::INL_JOIN_PACKED_SHARED:
                node->type = PlanNodeType::INL_JOIN_PACKED_PREDICATED_SHARED;
                break;
            case PlanNodeType::INL_JOIN_PACKED_CASCADE:
                node->type = PlanNodeType::INL_JOIN_PACKED_CASCADE_PREDICATED;
                break;
            case PlanNodeType::INL_JOIN_PACKED_CASCADE_SHARED:
                node->type = PlanNodeType::INL_JOIN_PACKED_CASCADE_PREDICATED_SHARED;
                break;
            case PlanNodeType::INL_JOIN_PACKED_GP_CASCADE:
                node->type = PlanNodeType::INL_JOIN_PACKED_GP_CASCADE_PREDICATED;
                break;
            case PlanNodeType::INL_JOIN_PACKED_GP_CASCADE_SHARED:
                node->type = PlanNodeType::INL_JOIN_PACKED_GP_CASCADE_PREDICATED_SHARED;
                break;
            case PlanNodeType::FLAT_JOIN:
                node->type = PlanNodeType::FLAT_JOIN_PREDICATED;
                break;
            case PlanNodeType::INTERSECTION:
                node->type = PlanNodeType::INTERSECTION_PREDICATED;
                break;
            case PlanNodeType::NWAY_INTERSECTION:
                node->type = PlanNodeType::NWAY_INTERSECTION_PREDICATED;
                break;
            default:
                throw std::runtime_error("Operator type not found for predication update: " +
                                         plan_node_type_to_string(node->type));
                break;
        }
    }

    // Recursively update children
    for (auto& child: node->children) {
        update_node_types(child.get());
    }
}

void PlanTree::finalize() {
    if (_root) { update_node_types(_root.get()); }
}

void PlanTree::print() const {
    std::cout << "=== Plan Tree ===" << std::endl;
    if (_root) {
        _root->print();
    } else {
        std::cout << "(empty)" << std::endl;
    }
    std::cout << "=================" << std::endl;
}

// =============================================================================
// Helper structures for plan-tree construction
// =============================================================================

struct JoinEdge {
    std::string from_attr;
    std::string to_attr;
    bool is_forward;
    QueryRelation* relation;
    size_t edge_index;
};

struct AttributeMetadata {
    std::string name;
    bool is_cyclic;
    std::set<std::string> neighbors;
    int degree;
};

// Build join graph from query
static std::vector<JoinEdge> build_join_graph(const Query& query) {
    std::vector<JoinEdge> edges;
    for (size_t i = 0; i < query.num_rels; i++) {
        JoinEdge edge;
        edge.from_attr = query.rels[i].fromVariable;
        edge.to_attr = query.rels[i].toVariable;
        edge.is_forward = query.rels[i].isFwd(edge.from_attr, edge.to_attr);
        edge.relation = &query.rels[i];
        edge.edge_index = i;
        edges.push_back(edge);
    }
    return edges;
}

// Helper to create sorted edge pair for undirected comparison
static std::pair<std::string, std::string> make_edge_pair(const std::string& u, const std::string& v) {
    return (u < v) ? std::make_pair(u, v) : std::make_pair(v, u);
}

// Bridge detection using Tarjan's algorithm
class CycleDetectorLocal {
public:
    CycleDetectorLocal(const std::vector<JoinEdge>& edges) : edges(edges) { build_undirected_graph(); }

    struct CycleDetectionResult {
        std::vector<std::vector<std::string>> cyclic_components;
        std::vector<std::string> acyclic_nodes;
        std::set<std::pair<std::string, std::string>> bridge_edges;
        std::set<std::pair<std::string, std::string>> cycle_edges;
    };

    CycleDetectionResult find_cyclic_and_acyclic() {
        CycleDetectionResult result;
        result.bridge_edges = find_bridges();
        auto cycle_graph = build_cycle_graph(result.bridge_edges);
        auto components = find_connected_components(cycle_graph);

        for (const auto& component: components) {
            if (component.size() > 1) {
                result.cyclic_components.push_back(component);
                for (size_t i = 0; i < component.size(); i++) {
                    for (size_t j = i + 1; j < component.size(); j++) {
                        for (const auto& edge: edges) {
                            if ((edge.from_attr == component[i] && edge.to_attr == component[j]) ||
                                (edge.from_attr == component[j] && edge.to_attr == component[i])) {
                                result.cycle_edges.insert(make_edge_pair(component[i], component[j]));
                                break;
                            }
                        }
                    }
                }
            } else {
                result.acyclic_nodes.push_back(component[0]);
            }
        }
        std::sort(result.acyclic_nodes.begin(), result.acyclic_nodes.end());
        return result;
    }

private:
    void build_undirected_graph() {
        for (const auto& edge: edges) {
            graph[edge.from_attr].push_back(edge.to_attr);
            graph[edge.to_attr].push_back(edge.from_attr);
            all_nodes.insert(edge.from_attr);
            all_nodes.insert(edge.to_attr);
        }
    }

    std::set<std::pair<std::string, std::string>> find_bridges() {
        std::set<std::pair<std::string, std::string>> bridge_edges;
        std::unordered_map<std::string, int> discovery_time;
        std::unordered_map<std::string, int> low_link;
        std::unordered_map<std::string, std::string> parent;
        std::set<std::string> visited;
        int time = 0;

        std::function<void(const std::string&)> dfs_bridges = [&](const std::string& u) {
            visited.insert(u);
            discovery_time[u] = time;
            low_link[u] = time;
            time++;

            for (const auto& v: graph[u]) {
                if (visited.find(v) == visited.end()) {
                    parent[v] = u;
                    dfs_bridges(v);
                    low_link[u] = std::min(low_link[u], low_link[v]);
                    if (low_link[v] > discovery_time[u]) { bridge_edges.insert(make_edge_pair(u, v)); }
                } else if (v != parent[u]) {
                    low_link[u] = std::min(low_link[u], discovery_time[v]);
                }
            }
        };

        for (const auto& node: all_nodes) {
            if (visited.find(node) == visited.end()) { dfs_bridges(node); }
        }
        return bridge_edges;
    }

    std::unordered_map<std::string, std::vector<std::string>>
    build_cycle_graph(const std::set<std::pair<std::string, std::string>>& bridge_edges) {
        std::unordered_map<std::string, std::vector<std::string>> cycle_graph;
        for (const auto& [u, neighbors]: graph) {
            cycle_graph[u] = {};
            for (const auto& v: neighbors) {
                auto edge_pair = make_edge_pair(u, v);
                if (bridge_edges.find(edge_pair) == bridge_edges.end()) { cycle_graph[u].push_back(v); }
            }
        }
        return cycle_graph;
    }

    std::vector<std::vector<std::string>>
    find_connected_components(const std::unordered_map<std::string, std::vector<std::string>>& graph) {
        std::set<std::string> visited;
        std::vector<std::vector<std::string>> components;

        std::function<void(const std::string&, std::vector<std::string>&)> dfs_collect =
                [&](const std::string& node, std::vector<std::string>& component) {
                    visited.insert(node);
                    component.push_back(node);
                    auto it = graph.find(node);
                    if (it != graph.end()) {
                        for (const auto& neighbor: it->second) {
                            if (visited.find(neighbor) == visited.end()) { dfs_collect(neighbor, component); }
                        }
                    }
                };

        for (const auto& node: all_nodes) {
            if (visited.find(node) == visited.end()) {
                std::vector<std::string> component;
                dfs_collect(node, component);
                components.push_back(component);
            }
        }
        return components;
    }

    const std::vector<JoinEdge>& edges;
    std::unordered_map<std::string, std::vector<std::string>> graph;
    std::set<std::string> all_nodes;
};

// Build metadata for all attributes
static std::unordered_map<std::string, AttributeMetadata>
build_attribute_metadata(const std::vector<JoinEdge>& edges, const std::set<std::string>& cyclic_attrs,
                         const std::set<std::string>& acyclic_attrs) {
    std::unordered_map<std::string, AttributeMetadata> metadata;

    for (const auto& edge: edges) {
        // Initialize metadata if not exists
        if (metadata.find(edge.from_attr) == metadata.end()) {
            metadata[edge.from_attr] = {edge.from_attr, false, {}, 0};
        }
        if (metadata.find(edge.to_attr) == metadata.end()) { metadata[edge.to_attr] = {edge.to_attr, false, {}, 0}; }

        // Add neighbors
        metadata[edge.from_attr].neighbors.insert(edge.to_attr);
        metadata[edge.to_attr].neighbors.insert(edge.from_attr);

        // Update degrees
        metadata[edge.from_attr].degree++;
        metadata[edge.to_attr].degree++;
    }

    // Mark cyclic attributes
    for (const auto& attr: cyclic_attrs) {
        if (metadata.find(attr) != metadata.end()) { metadata[attr].is_cyclic = true; }
    }

    return metadata;
}

// Get processed neighbors ordered by depth (shallowest first)
static std::vector<std::string>
get_processed_neighbors_ordered(const std::string& attr, const AttributeMetadata& attr_meta,
                                const std::set<std::string>& processed_attrs,
                                const std::unordered_map<std::string, uint32_t>& ftree_depth_map) {
    std::vector<std::pair<std::string, int>> neighbors_with_depth;

    for (const auto& nbr: attr_meta.neighbors) {
        if (processed_attrs.find(nbr) != processed_attrs.end()) {
            int depth = ftree_depth_map.count(nbr) ? ftree_depth_map.at(nbr) : 0;
            neighbors_with_depth.push_back({nbr, depth});
        }
    }

    std::sort(neighbors_with_depth.begin(), neighbors_with_depth.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    std::vector<std::string> result;
    for (const auto& [nbr, depth]: neighbors_with_depth) {
        result.push_back(nbr);
    }
    return result;
}

// Get edge direction between two attributes
static bool get_edge_direction(const std::string& from_attr, const std::string& to_attr,
                               const std::vector<JoinEdge>& edges) {
    for (const auto& edge: edges) {
        if (edge.from_attr == from_attr && edge.to_attr == to_attr) {
            return edge.is_forward;
        } else if (edge.to_attr == from_attr && edge.from_attr == to_attr) {
            return !edge.is_forward;
        }
    }
    return true;
}

// Check if we should use grandparent cascade
static bool should_use_linear_cascade(const Query& query, const std::vector<std::string>& ordering, size_t current_idx,
                                      const std::string& join_key) {
    std::unordered_map<std::string, std::vector<std::string>> parent_to_children;

    for (size_t i = 1; i < current_idx; i++) {
        if (ordering[i] == "_cd") continue;
        std::string current = ordering[i];
        for (int j = i - 1; j >= 0; j--) {
            if (ordering[j] == "_cd") continue;
            if (query.get_query_relation(ordering[j], current) != nullptr) {
                parent_to_children[ordering[j]].push_back(current);
                break;
            }
        }
    }
    return parent_to_children[join_key].empty();
}

// Get join key for an attribute
static std::string get_join_key(const Query& query, const std::vector<std::string>& ordering, size_t idx,
                                const std::string& next_output_attr) {
    for (size_t i = 0; i < idx; i++) {
        std::string join_key = ordering[i];
        if (join_key == "_cd") continue;
        if (query.get_query_relation(join_key, next_output_attr) != nullptr) { return join_key; }
    }
    throw std::runtime_error("No join key found for attribute: " + next_output_attr);
}

// =============================================================================
// Phase 1: Build Plan Tree
// =============================================================================

// Helper: Build ordering index map
static std::unordered_map<std::string, size_t> build_ordering_index(const std::vector<std::string>& ordering) {
    std::unordered_map<std::string, size_t> idx_map;
    for (size_t i = 0; i < ordering.size(); i++) {
        if (ordering[i] != "_cd") { idx_map[ordering[i]] = i; }
    }
    return idx_map;
}

// Helper: Extract theta join constraints from predicates
// Returns pairs of (ancestor_attr, descendant_attr) based on ordering
static std::vector<std::pair<std::string, std::string>>
extract_theta_constraints(const Query& query, const std::unordered_map<std::string, size_t>& ordering_idx) {
    std::vector<std::pair<std::string, std::string>> constraints;

    if (!query.has_predicates()) return constraints;

    const auto& pred_expr = query.get_predicates();
    for (const auto& group: pred_expr.groups) {
        for (const auto& pred: group.predicates) {
            if (pred.type == PredicateType::ATTRIBUTE) {
                auto left_it = ordering_idx.find(pred.left_attr);
                auto right_it = ordering_idx.find(pred.right_attr);

                if (left_it != ordering_idx.end() && right_it != ordering_idx.end()) {
                    if (left_it->second < right_it->second) {
                        constraints.push_back({pred.left_attr, pred.right_attr});
                    } else {
                        constraints.push_back({pred.right_attr, pred.left_attr});
                    }
                }
            }
        }
    }
    return constraints;
}

// Helper: Check if an attribute must be on a theta join path
// Returns the theta_ancestor if this attr is the actual theta descendant
// or if it's between ancestor and descendant in the ordering
static std::string get_required_theta_ancestor(
        size_t current_idx, const std::vector<std::pair<std::string, std::string>>& theta_constraints,
        const std::unordered_map<std::string, size_t>& ordering_idx, const std::string& current_attr) {
    // Check if current_attr is a descendant in any theta join
    for (const auto& [ancestor, descendant]: theta_constraints) {
        if (descendant == current_attr) {
            // This attribute is a theta join descendant
            // Return the ancestor it must be on the same path with
            return ancestor;
        }

        // Also check if current_attr is between ancestor and descendant
        // If so, it must also be on the path
        auto curr_it = ordering_idx.find(current_attr);
        auto anc_it = ordering_idx.find(ancestor);
        auto desc_it = ordering_idx.find(descendant);

        if (curr_it != ordering_idx.end() && anc_it != ordering_idx.end() && desc_it != ordering_idx.end()) {
            // If current is between ancestor and descendant in ordering
            if (anc_it->second < curr_it->second && curr_it->second < desc_it->second) { return ancestor; }
        }
    }
    return "";
}

// LLM_MAP: names in context_column (object keys for FTREE, or array of strings for flat formats).
static std::set<std::string> parse_llm_context_column_attrs(const Query& query) {
    nlohmann::json j;
    try {
        j = nlohmann::json::parse(query.get_llm_map_config());
    } catch (const nlohmann::json::exception& ex) {
        throw std::runtime_error(std::string("invalid JSON in LLM_MAP config: ") + ex.what());
    }
    if (!j.contains("context_column")) {
        throw std::runtime_error("LLM_MAP config is missing required key 'context_column'.");
    }
    const auto& cc = j["context_column"];
    std::set<std::string> out;
    if (cc.is_object()) {
        for (auto it = cc.begin(); it != cc.end(); ++it) { out.insert(it.key()); }
    } else if (cc.is_array()) {
        for (const auto& el : cc) {
            if (!el.is_string()) {
                throw std::runtime_error(
                        "LLM_MAP context_column array must contain only attribute names (strings).");
            }
            out.insert(el.get<std::string>());
        }
    } else {
        throw std::runtime_error(
                "LLM_MAP context_column must be a JSON object (FTREE) or an array of strings (flat formats).");
    }
    if (out.empty()) { throw std::runtime_error("LLM_MAP context_column cannot be empty."); }
    return out;
}

static size_t column_ordering_position_of(const std::vector<std::string>& ordering, const std::string& attr) {
    for (size_t i = 0; i < ordering.size(); ++i) {
        if (ordering[i] != "_cd" && ordering[i] == attr) return i;
    }
    return ordering.size();
}

// Every context input must appear in column ordering strictly before the LLM output attribute.
static void validate_llm_context_vs_ordering(const std::vector<std::string>& ordering,
                                             const std::string& llm_out, const std::set<std::string>& ctx_attrs) {
    const size_t llm_pos = column_ordering_position_of(ordering, llm_out);
    if (llm_pos >= ordering.size()) {
        throw std::runtime_error("LLM_MAP output attribute '" + llm_out + "' must appear in the column ordering.");
    }
    for (const auto& req : ctx_attrs) {
        const size_t p = column_ordering_position_of(ordering, req);
        if (p >= ordering.size()) {
            throw std::runtime_error("LLM_MAP context_column attribute '" + req +
                                     "' must appear in the column ordering.");
        }
        if (p >= llm_pos) {
            throw std::runtime_error("LLM_MAP: context_column attribute '" + req +
                                     "' must appear before output attribute '" + llm_out +
                                     "' in column ordering.");
        }
    }
}

std::unique_ptr<PlanTree> build_plan_tree(const Query& query, const std::vector<std::string>& ordering,
                                          sink_type sink) {
    const bool is_export_sink =
            (sink == SINK_EXPORT_CSV) || (sink == SINK_EXPORT_JSON) || (sink == SINK_EXPORT_MARKDOWN);
    // Export sinks rely on factorized-tree iteration (FTreeIterator), so we must build a packed plan.
    const bool is_packed = (sink == SINK_PACKED) || (sink == SINK_MIN) || (sink == SINK_PACKED_NOOP) || is_export_sink;

    auto plan_tree = std::make_unique<PlanTree>();
    plan_tree->set_sink_type(sink);

    // ==================================================================
    // STEP 0: Build ordering index and extract theta constraints
    // ==================================================================
    auto ordering_idx = build_ordering_index(ordering);
    auto theta_constraints = extract_theta_constraints(query, ordering_idx);

    // ==================================================================
    // STEP 1: Build join graph and classify edges
    // ==================================================================
    std::vector<JoinEdge> edges = build_join_graph(query);
    std::string root_attr = ordering[0];

    CycleDetectorLocal detector(edges);
    auto detection_result = detector.find_cyclic_and_acyclic();

    std::set<std::string> cyclic_attrs;
    for (const auto& component: detection_result.cyclic_components) {
        for (const auto& attr: component) {
            cyclic_attrs.insert(attr);
        }
    }

    std::set<std::string> acyclic_attrs;
    for (const auto& node: detection_result.acyclic_nodes) {
        acyclic_attrs.insert(node);
    }

    // ==================================================================
    // STEP 2: Build attribute metadata
    // ==================================================================
    auto metadata = build_attribute_metadata(edges, cyclic_attrs, acyclic_attrs);

    // ==================================================================
    // STEP 3: Create root (scan) node
    // ==================================================================
    auto root_node =
            std::make_unique<PlanNode>(root_attr, is_packed ? PlanNodeType::SCAN : PlanNodeType::SCAN_UNPACKED);
    root_node->depth = 0;
    root_node->is_cyclic = cyclic_attrs.count(root_attr) > 0;

    // Track state
    std::set<std::string> processed_attrs;
    processed_attrs.insert(root_attr);
    std::unordered_map<std::string, uint32_t> ftree_depth_map;
    ftree_depth_map[root_attr] = 0;
    std::unordered_map<std::string, PlanNode*> attr_to_node;
    attr_to_node[root_attr] = root_node.get();

    std::string last_attribute_processed = root_attr;
    PlanNode* last_node = root_node.get();

    // ==================================================================
    // STEP 4: Process each attribute in ordering
    // ==================================================================
    uint32_t cyclic_joins_cnt = 0;
    const std::string llm_output_attr =
            query.has_llm_map() ? query.get_llm_map_output_attr() : std::string();
    std::set<std::string> llm_context_attrs;
    if (!llm_output_attr.empty()) {
        llm_context_attrs = parse_llm_context_column_attrs(query);
        validate_llm_context_vs_ordering(ordering, llm_output_attr, llm_context_attrs);
    }
    for (size_t i = 1; i < ordering.size(); i++) {
        if (ordering[i] == "_cd") continue;

        const auto current_attr = ordering[i];

        // LLM output is synthetic (no base relation): do not create a PlanNode. When we reach
        // it in column ordering, require every context_column attribute to already be in the tree.
        if (!llm_output_attr.empty() && current_attr == llm_output_attr) {
            for (const auto& req : llm_context_attrs) {
                if (!processed_attrs.count(req)) {
                    throw std::runtime_error(
                            "LLM_MAP: column ordering places '" + llm_output_attr +
                            "' before all context_column inputs are available in the plan. Missing '" + req +
                            "'. Put '" + llm_output_attr +
                            "' after every attribute listed in context_column once joins are possible.");
                }
            }
            continue;
        }
        const bool is_cascaded = (i + 1 < ordering.size() && ordering[i + 1] == "_cd");
        const bool is_last_attr = (i == ordering.size() - 1) || (i == ordering.size() - 2 && ordering.back() == "_cd");

        auto& current_meta = metadata[current_attr];

        // ==================================================================
        // Handle non-last attributes
        // ==================================================================
        if (!is_last_attr) {
            // Check if this attribute must be on a theta join path
            std::string required_theta_ancestor =
                    get_required_theta_ancestor(i, theta_constraints, ordering_idx, current_attr);

            if (!current_meta.is_cyclic) {
                // Acyclic attribute - use regular INLJoin from its actual join key
                // Theta join constraints are applied after the join, not as part of join type decision
                std::string join_key = get_join_key(query, ordering, i, current_attr);
                bool is_fwd = get_edge_direction(join_key, current_attr, edges);

                PlanNodeType node_type;
                if (is_packed) {
                    if (is_cascaded) {
                        node_type = should_use_linear_cascade(query, ordering, i, join_key)
                                            ? PlanNodeType::INL_JOIN_PACKED_GP_CASCADE
                                            : PlanNodeType::INL_JOIN_PACKED_CASCADE;
                    } else {
                        node_type = PlanNodeType::INL_JOIN_PACKED;
                    }
                } else {
                    node_type = PlanNodeType::INL_JOIN_UNPACKED;
                }

                auto new_node = std::make_unique<PlanNode>(current_attr, node_type);
                new_node->join_from_attr = join_key;
                new_node->is_fwd = is_fwd;
                new_node->is_cyclic = false;

                // Add as child of join_key (the actual parent in the join graph)
                // This ensures the plan tree reflects the actual join relationships
                // If join_key is the same as last_attribute_processed, we still add it as a child
                // to maintain the tree structure that matches the join graph
                PlanNode* parent_node = attr_to_node[join_key];
                PlanNode* added_node = parent_node->add_child(std::move(new_node));

                ftree_depth_map[current_attr] = ftree_depth_map[join_key] + 1;
                processed_attrs.insert(current_attr);
                attr_to_node[current_attr] = added_node;
                last_attribute_processed = current_attr;
                last_node = added_node;
            } else {
                // Cyclic attribute or attribute on theta join path
                if (cyclic_joins_cnt == 0) {
                    // Entry join -> normal join
                    std::string join_key = get_join_key(query, ordering, i, current_attr);
                    bool is_fwd = get_edge_direction(join_key, current_attr, edges);

                    PlanNodeType node_type;
                    if (is_packed) {
                        if (is_cascaded) {
                            node_type = should_use_linear_cascade(query, ordering, i, join_key)
                                                ? PlanNodeType::INL_JOIN_PACKED_GP_CASCADE
                                                : PlanNodeType::INL_JOIN_PACKED_CASCADE;
                        } else {
                            node_type = PlanNodeType::INL_JOIN_PACKED;
                        }
                    } else {
                        node_type = PlanNodeType::INL_JOIN_UNPACKED;
                    }

                    auto new_node = std::make_unique<PlanNode>(current_attr, node_type);
                    new_node->join_from_attr = join_key;
                    new_node->is_fwd = is_fwd;
                    new_node->is_cyclic = true;

                    // If this attribute is on a theta join path, add as child of required_theta_ancestor
                    // Otherwise, add as child of last_attribute_processed to maintain ordering order
                    PlanNode* parent_node;
                    if (!required_theta_ancestor.empty() &&
                        attr_to_node.find(required_theta_ancestor) != attr_to_node.end()) {
                        // Add as child of required_theta_ancestor to maintain theta join path
                        parent_node = attr_to_node[required_theta_ancestor];
                        // If join_key differs from required_theta_ancestor, we need FlatJoin
                        if (join_key != required_theta_ancestor) {
                            new_node->type = PlanNodeType::FLAT_JOIN;
                            new_node->join_from_attr = required_theta_ancestor;
                            new_node->lca_attr = join_key;
                        }
                    } else {
                        // Add as child of join_key to reflect actual join relationship
                        parent_node = attr_to_node[join_key];
                    }
                    PlanNode* added_node = parent_node->add_child(std::move(new_node));

                    // Update depth based on the actual parent
                    ftree_depth_map[current_attr] = ftree_depth_map[parent_node->attribute] + 1;
                    processed_attrs.insert(current_attr);
                    attr_to_node[current_attr] = added_node;
                    last_attribute_processed = current_attr;
                    last_node = added_node;
                    cyclic_joins_cnt++;
                } else {
                    // Check if current attr has multiple processed neighbors -> use Intersection
                    std::string parent = last_attribute_processed;
                    auto ordered_neighbors = get_processed_neighbors_ordered(current_attr, current_meta,
                                                                             processed_attrs, ftree_depth_map);

                    if (ordered_neighbors.size() >= 2) {
                        // Multiple processed neighbors -> use Intersection
                        std::vector<std::pair<std::string, bool>> input_attrs_and_directions;
                        for (const auto& nbr: ordered_neighbors) {
                            bool nbr_is_fwd = get_edge_direction(nbr, current_attr, edges);
                            input_attrs_and_directions.push_back({nbr, nbr_is_fwd});
                        }

                        PlanNodeType node_type = (ordered_neighbors.size() == 2) ? PlanNodeType::INTERSECTION
                                                                                 : PlanNodeType::NWAY_INTERSECTION;

                        auto new_node = std::make_unique<PlanNode>(current_attr, node_type);
                        new_node->intersection_inputs = input_attrs_and_directions;
                        new_node->is_cyclic = true;

                        // Add as child of last_attribute_processed to maintain ordering order
                        PlanNode* parent_node = attr_to_node[last_attribute_processed];
                        PlanNode* added_node = parent_node->add_child(std::move(new_node));

                        ftree_depth_map[current_attr] = ftree_depth_map[last_attribute_processed] + 1;
                        processed_attrs.insert(current_attr);
                        attr_to_node[current_attr] = added_node;
                        last_attribute_processed = current_attr;
                        last_node = added_node;
                    } else {
                        // Single processed neighbor - use INLJoin or FlatJoin
                        std::string lca = ordered_neighbors.back();
                        bool is_fwd = get_edge_direction(lca, current_attr, edges);

                        // If LCA == parent, we have a direct edge -> use INLJoin
                        // If LCA != parent, we need to jump back to ancestor -> use FlatJoin
                        if (lca == parent) {
                            // Direct edge to parent - use INLJoin
                            PlanNodeType node_type;
                            if (is_packed) {
                                if (is_cascaded) {
                                    node_type = should_use_linear_cascade(query, ordering, i, lca)
                                                        ? PlanNodeType::INL_JOIN_PACKED_GP_CASCADE
                                                        : PlanNodeType::INL_JOIN_PACKED_CASCADE;
                                } else {
                                    node_type = PlanNodeType::INL_JOIN_PACKED;
                                }
                            } else {
                                node_type = PlanNodeType::INL_JOIN_UNPACKED;
                            }

                            auto new_node = std::make_unique<PlanNode>(current_attr, node_type);
                            new_node->join_from_attr = lca;
                            new_node->is_fwd = is_fwd;
                            new_node->is_cyclic = true;

                            // Add as child of last_attribute_processed to maintain ordering order
                            PlanNode* parent_node = attr_to_node[last_attribute_processed];
                            PlanNode* added_node = parent_node->add_child(std::move(new_node));

                            ftree_depth_map[current_attr] = ftree_depth_map[last_attribute_processed] + 1;
                            processed_attrs.insert(current_attr);
                            attr_to_node[current_attr] = added_node;
                            last_attribute_processed = current_attr;
                            last_node = added_node;
                        } else {
                            // Need to jump back to ancestor - use FlatJoin
                            auto new_node = std::make_unique<PlanNode>(current_attr, PlanNodeType::FLAT_JOIN);
                            new_node->join_from_attr = parent;
                            new_node->lca_attr = lca;
                            new_node->is_fwd = is_fwd;
                            new_node->is_cyclic = true;

                            // Add as child of last_attribute_processed to maintain ordering order
                            PlanNode* parent_node = attr_to_node[last_attribute_processed];
                            PlanNode* added_node = parent_node->add_child(std::move(new_node));

                            ftree_depth_map[current_attr] = ftree_depth_map[last_attribute_processed] + 1;
                            processed_attrs.insert(current_attr);
                            attr_to_node[current_attr] = added_node;
                            last_attribute_processed = current_attr;
                            last_node = added_node;
                        }
                    }
                    cyclic_joins_cnt++;
                }
            }
        } else {
            // ==================================================================
            // Handle LAST attribute
            // ==================================================================
            if (!current_meta.is_cyclic) {
                // Simple acyclic join
                std::string join_key = get_join_key(query, ordering, i, current_attr);
                bool is_fwd = get_edge_direction(join_key, current_attr, edges);

                PlanNodeType node_type;
                if (is_packed) {
                    if (is_cascaded) {
                        node_type = should_use_linear_cascade(query, ordering, i, join_key)
                                            ? PlanNodeType::INL_JOIN_PACKED_GP_CASCADE
                                            : PlanNodeType::INL_JOIN_PACKED_CASCADE;
                    } else {
                        node_type = PlanNodeType::INL_JOIN_PACKED;
                    }
                } else {
                    node_type = PlanNodeType::INL_JOIN_UNPACKED;
                }

                auto new_node = std::make_unique<PlanNode>(current_attr, node_type);
                new_node->join_from_attr = join_key;
                new_node->is_fwd = is_fwd;
                new_node->is_cyclic = false;

                // Add as child of join_key to reflect actual join relationship
                PlanNode* parent_node = attr_to_node[join_key];
                PlanNode* added_node = parent_node->add_child(std::move(new_node));

                ftree_depth_map[current_attr] = ftree_depth_map[join_key] + 1;
                processed_attrs.insert(current_attr);
                attr_to_node[current_attr] = added_node;
                last_node = added_node;
            } else {
                // N-way intersection for cyclic last attribute
                auto ordered_neighbors =
                        get_processed_neighbors_ordered(current_attr, current_meta, processed_attrs, ftree_depth_map);

                assert(ordered_neighbors.size() >= 2 && "Need at least 2 processed neighbors for intersection");

                // Build input attributes
                std::vector<std::pair<std::string, bool>> input_attrs_and_directions;
                for (const auto& nbr: ordered_neighbors) {
                    bool is_fwd = get_edge_direction(nbr, current_attr, edges);
                    input_attrs_and_directions.push_back({nbr, is_fwd});
                }

                PlanNodeType node_type =
                        (ordered_neighbors.size() == 2) ? PlanNodeType::INTERSECTION : PlanNodeType::NWAY_INTERSECTION;

                auto new_node = std::make_unique<PlanNode>(current_attr, node_type);
                new_node->intersection_inputs = input_attrs_and_directions;
                new_node->is_cyclic = true;

                // Add as child of last_attribute_processed to maintain ordering order
                PlanNode* parent_node = attr_to_node[last_attribute_processed];
                PlanNode* added_node = parent_node->add_child(std::move(new_node));

                ftree_depth_map[current_attr] = ftree_depth_map[last_attribute_processed] + 1;
                processed_attrs.insert(current_attr);
                attr_to_node[current_attr] = added_node;
                last_node = added_node;
            }
        }
    }

    // ==================================================================
    // STEP 4.5: Identify property attributes (in query but not in ordering)
    // ==================================================================
    // Property attributes are those that appear in query relations but not in
    // the ordering. They are 1:1 lookups (e.g., Company_type(ct, kind)) where
    // the entity (ct) is in the ordering but the property (kind) is not.
    // Property predicates are evaluated via PropertyFilter operators instead
    // of being distributed to individual join nodes.
    std::unordered_set<std::string> property_attrs;
    // Map: property_attr -> (entity_attr, is_fwd)
    std::unordered_map<std::string, std::pair<std::string, bool>> property_to_entity;

    if (query.is_datalog_format()) {
        for (uint64_t r = 0; r < query.num_rels; ++r) {
            const auto& rel = query.rels[r];
            bool from_in_ordering = (processed_attrs.count(rel.fromVariable) > 0);
            bool to_in_ordering = (processed_attrs.count(rel.toVariable) > 0);

            if (from_in_ordering && !to_in_ordering && !rel.toVariable.empty() && rel.toVariable != "_") {
                // toVariable is a property of fromVariable
                property_attrs.insert(rel.toVariable);
                property_to_entity[rel.toVariable] = {rel.fromVariable, true};
            } else if (!from_in_ordering && to_in_ordering && !rel.fromVariable.empty() && rel.fromVariable != "_") {
                // fromVariable is a property of toVariable
                property_attrs.insert(rel.fromVariable);
                property_to_entity[rel.fromVariable] = {rel.toVariable, false};
            }
        }
    }

    // ==================================================================
    // STEP 5: Process predicates from query
    // ==================================================================
    if (query.has_predicates()) {
        const auto& pred_expr = query.get_predicates();

        // Collect all scalar-predicate attributes once.
        // For top-level OR, attributes not constrained in a branch must still
        // receive an empty TRUE group to preserve branch semantics.
        std::unordered_set<std::string> scalar_attrs;
        // Also track which scalar attrs are property attrs vs join attrs
        std::unordered_set<std::string> property_scalar_attrs;
        // Helper to classify a predicate as property or join scalar
        auto classify_pred = [&](const Predicate& pred) {
            if (pred.type == PredicateType::SCALAR) {
                if (property_attrs.count(pred.left_attr) > 0) {
                    property_scalar_attrs.insert(pred.left_attr);
                } else {
                    scalar_attrs.insert(pred.left_attr);
                }
            }
        };
        for (const auto& group: pred_expr.groups) {
            for (const auto& pred: group.predicates) {
                classify_pred(pred);
            }
            // Also scan predicates inside branches
            for (const auto& branch: group.branches) {
                for (const auto& pred: branch.predicates) {
                    classify_pred(pred);
                }
            }
        }

        // Collect property predicates by entity, preserving group structure
        // For each entity, we build a PredicateExpression with the same group structure
        // but only containing predicates on that entity's properties.
        std::unordered_map<std::string, PredicateExpression> entity_property_preds;
        std::unordered_set<std::string> entities_with_properties;

        if (!property_scalar_attrs.empty()) {
            // Identify which entities have property predicates
            for (const auto& prop_attr : property_scalar_attrs) {
                auto it = property_to_entity.find(prop_attr);
                if (it != property_to_entity.end()) {
                    entities_with_properties.insert(it->second.first);
                }
            }

            // For each entity, build the property predicate expression
            for (const auto& entity : entities_with_properties) {
                PredicateExpression& entity_expr = entity_property_preds[entity];
                entity_expr.top_level_op = pred_expr.top_level_op;

                for (const auto& group : pred_expr.groups) {
                    PredicateGroup pg;
                    pg.op = group.op;

                    if (group.has_branches()) {
                        // OR group with AND branches — filter each branch
                        for (const auto& branch : group.branches) {
                            PredicateGroup filtered_branch;
                            filtered_branch.op = branch.op;
                            for (const auto& pred : branch.predicates) {
                                if (pred.type == PredicateType::SCALAR && property_attrs.count(pred.left_attr) > 0) {
                                    auto pit = property_to_entity.find(pred.left_attr);
                                    if (pit != property_to_entity.end() && pit->second.first == entity) {
                                        filtered_branch.predicates.push_back(pred);
                                    }
                                }
                            }
                            pg.branches.push_back(std::move(filtered_branch));
                        }
                    } else {
                        for (const auto& pred : group.predicates) {
                            if (pred.type == PredicateType::SCALAR && property_attrs.count(pred.left_attr) > 0) {
                                auto pit = property_to_entity.find(pred.left_attr);
                                if (pit != property_to_entity.end() && pit->second.first == entity) {
                                    pg.predicates.push_back(pred);
                                }
                            }
                        }
                    }

                    // For OR top-level, always add the group (empty = TRUE)
                    if (!pg.empty() || pred_expr.top_level_op == LogicalOp::OR) {
                        entity_expr.groups.push_back(std::move(pg));
                    }
                }

                // Attach PropertyFilterInfo to the entity's plan node
                PlanNode* entity_node = attr_to_node[entity];
                if (entity_node) {
                    entity_node->property_predicate_expr = entity_expr;
                    for (const auto& prop_attr : property_scalar_attrs) {
                        auto pit = property_to_entity.find(prop_attr);
                        if (pit != property_to_entity.end() && pit->second.first == entity) {
                            PropertyFilterInfo pfi;
                            pfi.property_attr = prop_attr;
                            pfi.entity_attr = entity;
                            pfi.is_fwd = pit->second.second;
                            entity_node->property_filter_infos.push_back(std::move(pfi));
                        }
                    }
                }
            }
        }

        // Distribute each source group to plan nodes, preserving group op.
        // Each source group becomes a separate group at the target node so
        // that AND/OR semantics between groups are retained.
        // NOTE: Skip predicates on property attributes — they're handled by PropertyFilter.
        for (const auto& group: pred_expr.groups) {
            // Collect scalar predicates by target attribute for this group
            // Only include join attributes (not property attributes)
            std::unordered_map<std::string, std::vector<const Predicate*>> attr_preds;

            for (const auto& pred: group.predicates) {
                if (pred.type == PredicateType::SCALAR && property_attrs.count(pred.left_attr) == 0) {
                    attr_preds[pred.left_attr].push_back(&pred);
                }
            }

            // For OR expressions, emit groups for every scalar attr even when
            // this branch has no predicate for that attr (empty group => TRUE).
            if (pred_expr.top_level_op == LogicalOp::OR) {
                for (const auto& attr : scalar_attrs) {
                    PlanNode* node = attr_to_node[attr];
                    if (!node) continue;
                    node->scalar_predicates.top_level_op = pred_expr.top_level_op;

                    PredicateGroup sg;
                    sg.op = group.op;
                    auto it = attr_preds.find(attr);
                    if (it != attr_preds.end()) {
                        for (auto* p : it->second) {
                            sg.predicates.push_back(*p);
                        }
                    }
                    node->scalar_predicates.groups.push_back(std::move(sg));
                }
            } else {
                for (auto& [attr, preds] : attr_preds) {
                    PlanNode* node = attr_to_node[attr];
                    if (!node) continue;
                    node->scalar_predicates.top_level_op = pred_expr.top_level_op;
                    PredicateGroup sg;
                    sg.op = group.op;
                    for (auto* p : preds) {
                        sg.predicates.push_back(*p);
                    }
                    node->scalar_predicates.groups.push_back(std::move(sg));
                }
            }

            for (const auto& pred: group.predicates) {
                if (pred.type == PredicateType::ATTRIBUTE) {
                    // Attribute predicate: attr1 <op> attr2
                    // Determine which is ancestor and which is descendant
                    PlanNode* left_node = attr_to_node[pred.left_attr];
                    PlanNode* right_node = attr_to_node[pred.right_attr];

                    if (!left_node || !right_node) {
                        std::cerr << "Warning: Cannot find nodes for attribute predicate: " << pred.left_attr
                                  << " <op> " << pred.right_attr << std::endl;
                        continue;
                    }

                    // Determine ancestor/descendant relationship
                    // First check tree structure: if one is an ancestor of the other, use that
                    // Otherwise, use depth comparison, and if depths are equal, use ordering index
                    std::string ancestor_attr, descendant_attr;
                    PredicateOp adjusted_op = pred.op;

                    // Check if left is ancestor of right in the tree
                    bool left_is_ancestor = false;
                    const PlanNode* current = right_node;
                    while (current != nullptr) {
                        if (current == left_node) {
                            left_is_ancestor = true;
                            break;
                        }
                        current = current->parent;
                    }

                    // Check if right is ancestor of left in the tree
                    bool right_is_ancestor = false;
                    current = left_node;
                    while (current != nullptr) {
                        if (current == right_node) {
                            right_is_ancestor = true;
                            break;
                        }
                        current = current->parent;
                    }

                    // PackedThetaJoin requires a true ancestor-descendant relationship in the tree
                    // If neither attribute is an ancestor of the other, we cannot use PackedThetaJoin
                    if (left_is_ancestor) {
                        // left is ancestor of right
                        ancestor_attr = pred.left_attr;
                        descendant_attr = pred.right_attr;
                        // Keep original op

                        // Add to sorted theta joins list (will be sorted later)
                        plan_tree->get_sorted_theta_joins().emplace_back(ancestor_attr, descendant_attr, adjusted_op);

                        // Also add to descendant node's theta_joins_after for backward compatibility
                        PlanNode* desc_node = attr_to_node[descendant_attr];
                        if (desc_node) {
                            desc_node->theta_joins_after.emplace_back(ancestor_attr, descendant_attr, adjusted_op);
                        }
                    } else if (right_is_ancestor) {
                        // right is ancestor of left
                        ancestor_attr = pred.right_attr;
                        descendant_attr = pred.left_attr;
                        // Flip the operator since we swapped operands
                        switch (pred.op) {
                            case PredicateOp::LT:
                                adjusted_op = PredicateOp::GT;
                                break;
                            case PredicateOp::GT:
                                adjusted_op = PredicateOp::LT;
                                break;
                            case PredicateOp::LTE:
                                adjusted_op = PredicateOp::GTE;
                                break;
                            case PredicateOp::GTE:
                                adjusted_op = PredicateOp::LTE;
                                break;
                            default:
                                break;// EQ, NEQ don't need adjustment
                        }

                        // Add to sorted theta joins list (will be sorted later)
                        plan_tree->get_sorted_theta_joins().emplace_back(ancestor_attr, descendant_attr, adjusted_op);

                        // Also add to descendant node's theta_joins_after for backward compatibility
                        PlanNode* desc_node2 = attr_to_node[descendant_attr];
                        if (desc_node2) {
                            desc_node2->theta_joins_after.emplace_back(ancestor_attr, descendant_attr, adjusted_op);
                        }
                    } else {
                        // Neither is an ancestor of the other - they are siblings or unrelated
                        // Force a linear path by restructuring the tree using FlatJoin
                        // Find the LCA (lowest common ancestor) of the two attributes
                        std::set<const PlanNode*> left_ancestors;
                        const PlanNode* current = left_node;
                        while (current != nullptr) {
                            left_ancestors.insert(current);
                            current = current->parent;
                        }

                        // Find LCA by walking up from right_node
                        current = right_node;
                        const PlanNode* lca_node = nullptr;
                        while (current != nullptr) {
                            if (left_ancestors.count(current)) {
                                lca_node = current;
                                break;
                            }
                            current = current->parent;
                        }

                        if (!lca_node) {
                            std::cerr << "Warning: Cannot find LCA for predicate " << pred.left_attr << " "
                                      << predicate_op_to_string(pred.op) << " " << pred.right_attr
                                      << ". Skipping this predicate." << std::endl;
                            continue;
                        }

                        // Determine ancestor/descendant based on ordering index
                        // Earlier in ordering becomes the ancestor
                        auto left_it = ordering_idx.find(pred.left_attr);
                        auto right_it = ordering_idx.find(pred.right_attr);

                        PlanNode* ancestor_node;
                        PlanNode* descendant_node;
                        std::string lca_attr = lca_node->attribute;

                        if (left_it != ordering_idx.end() && right_it != ordering_idx.end() &&
                            left_it->second < right_it->second) {
                            // left comes before right in ordering -> left is ancestor
                            ancestor_attr = pred.left_attr;
                            descendant_attr = pred.right_attr;
                            ancestor_node = left_node;
                            descendant_node = right_node;
                        } else {
                            // right comes before left in ordering -> right is ancestor
                            ancestor_attr = pred.right_attr;
                            descendant_attr = pred.left_attr;
                            ancestor_node = right_node;
                            descendant_node = left_node;
                            // Flip the operator since we swapped operands
                            switch (pred.op) {
                                case PredicateOp::LT:
                                    adjusted_op = PredicateOp::GT;
                                    break;
                                case PredicateOp::GT:
                                    adjusted_op = PredicateOp::LT;
                                    break;
                                case PredicateOp::LTE:
                                    adjusted_op = PredicateOp::GTE;
                                    break;
                                case PredicateOp::GTE:
                                    adjusted_op = PredicateOp::LTE;
                                    break;
                                default:
                                    break;// EQ, NEQ don't need adjustment
                            }
                        }

                        // Check if descendant is already a child of ancestor
                        bool already_descendant = false;
                        current = descendant_node;
                        while (current != nullptr) {
                            if (current == ancestor_node) {
                                already_descendant = true;
                                break;
                            }
                            current = current->parent;
                        }

                        // Check if intermediate attributes between ancestor and descendant need to be FlatJoin
                        auto ancestor_it = ordering_idx.find(ancestor_attr);
                        auto descendant_it = ordering_idx.find(descendant_attr);

                        if (ancestor_it != ordering_idx.end() && descendant_it != ordering_idx.end()) {
                            size_t ancestor_pos = ancestor_it->second;
                            size_t descendant_pos = descendant_it->second;

                            // Check intermediate attributes between ancestor and descendant
                            for (size_t pos = ancestor_pos + 1; pos < descendant_pos; ++pos) {
                                std::string intermediate_attr = ordering[pos];
                                PlanNode* intermediate_node = attr_to_node[intermediate_attr];

                                if (!intermediate_node) continue;

                                // Get the join_key for this intermediate attribute
                                std::string join_key = get_join_key(query, ordering, pos, intermediate_attr);
                                PlanNode* tree_parent = intermediate_node->parent;

                                // Always convert intermediate attributes on the path to FlatJoin if needed
                                // This ensures the path from ancestor to descendant is maintained
                                if (tree_parent && join_key != tree_parent->attribute) {
                                    // Convert to FlatJoin to jump from tree_parent to join_key
                                    // join_from_attr is the tree parent (where we are in the tree)
                                    // lca_attr is the actual join key (where we need to join from)
                                    intermediate_node->type = PlanNodeType::FLAT_JOIN;
                                    intermediate_node->join_from_attr = tree_parent->attribute;
                                    intermediate_node->lca_attr = join_key;
                                    intermediate_node->is_fwd = get_edge_direction(join_key, intermediate_attr, edges);
                                }
                            }
                        }

                        if (!already_descendant) {
                            // Restructure: need to make descendant_node a descendant of ancestor_node
                            //
                            // IMPORTANT: We must respect the column ordering!
                            // If moving descendant's branch would place an earlier-ordered attribute
                            // under a later-ordered one, we should move the ancestor instead.

                            // Step 1: Find common ancestor of ancestor_node and descendant_node
                            std::set<PlanNode*> ancestor_path_set;
                            PlanNode* path_node = ancestor_node;
                            while (path_node != nullptr) {
                                ancestor_path_set.insert(path_node);
                                path_node = path_node->parent;
                            }

                            PlanNode* common_ancestor = nullptr;
                            path_node = descendant_node;
                            while (path_node != nullptr) {
                                if (ancestor_path_set.count(path_node)) {
                                    common_ancestor = path_node;
                                    break;
                                }
                                path_node = path_node->parent;
                            }

                            if (!common_ancestor) { common_ancestor = root_node.get(); }

                            // Step 2: Find branch roots (direct children of common_ancestor)
                            // Branch containing ancestor_node
                            PlanNode* ancestor_branch_root = nullptr;
                            path_node = ancestor_node;
                            while (path_node != nullptr && path_node->parent != common_ancestor) {
                                path_node = path_node->parent;
                            }
                            if (path_node && path_node->parent == common_ancestor) { ancestor_branch_root = path_node; }

                            // Branch containing descendant_node
                            PlanNode* descendant_branch_root = nullptr;
                            path_node = descendant_node;
                            while (path_node != nullptr && path_node->parent != common_ancestor) {
                                path_node = path_node->parent;
                            }
                            if (path_node && path_node->parent == common_ancestor) {
                                descendant_branch_root = path_node;
                            }

                            // Step 3: Determine which branch to move based on ordering
                            // We should move the branch whose root comes LATER in ordering
                            // This ensures we don't violate the column ordering
                            size_t ancestor_branch_pos = SIZE_MAX;
                            size_t descendant_branch_pos = SIZE_MAX;

                            if (ancestor_branch_root) {
                                auto it = ordering_idx.find(ancestor_branch_root->attribute);
                                if (it != ordering_idx.end()) ancestor_branch_pos = it->second;
                            }
                            if (descendant_branch_root) {
                                auto it = ordering_idx.find(descendant_branch_root->attribute);
                                if (it != ordering_idx.end()) descendant_branch_pos = it->second;
                            }

                            // If descendant's branch root comes BEFORE ancestor's branch root in ordering,
                            // we should move the ANCESTOR branch under descendant's path instead
                            bool move_ancestor_instead = (descendant_branch_root && ancestor_branch_root &&
                                                          descendant_branch_pos < ancestor_branch_pos);

                            if (move_ancestor_instead) {
                                // Move ancestor_node (or its branch) to be under the appropriate node
                                // in descendant's path, just before the descendant
                                //
                                // Find the node in descendant's path that comes just before ancestor in ordering
                                // This will be the new parent for ancestor_node
                                PlanNode* new_parent_for_ancestor = nullptr;
                                path_node = descendant_node;
                                while (path_node != nullptr && path_node != common_ancestor) {
                                    auto pos_it = ordering_idx.find(path_node->attribute);
                                    if (pos_it != ordering_idx.end() && pos_it->second < ancestor_branch_pos) {
                                        // This node comes before the ancestor in ordering
                                        // It could be the new parent, but keep looking for one closer to ancestor's position
                                        if (!new_parent_for_ancestor) {
                                            new_parent_for_ancestor = path_node;
                                        } else {
                                            auto curr_parent_pos =
                                                    ordering_idx.find(new_parent_for_ancestor->attribute);
                                            if (curr_parent_pos != ordering_idx.end() &&
                                                pos_it->second > curr_parent_pos->second) {
                                                new_parent_for_ancestor = path_node;
                                            }
                                        }
                                    }
                                    path_node = path_node->parent;
                                }

                                if (!new_parent_for_ancestor) {
                                    // Use descendant's parent as fallback
                                    new_parent_for_ancestor = descendant_node->parent;
                                }

                                if (new_parent_for_ancestor && ancestor_branch_root) {
                                    // Remove ancestor's branch from common_ancestor
                                    PlanNode* old_parent = ancestor_branch_root->parent;
                                    std::string original_join_from = ancestor_branch_root->join_from_attr;

                                    auto& children = old_parent->children;
                                    auto it = std::find_if(
                                            children.begin(), children.end(),
                                            [ancestor_branch_root](const std::unique_ptr<PlanNode>& child) {
                                                return child.get() == ancestor_branch_root;
                                            });
                                    if (it != children.end()) {
                                        std::unique_ptr<PlanNode> moved_branch = std::move(*it);
                                        children.erase(it);

                                        // Convert to FlatJoin
                                        moved_branch->type = PlanNodeType::FLAT_JOIN;
                                        moved_branch->lca_attr = original_join_from;
                                        moved_branch->join_from_attr = new_parent_for_ancestor->attribute;

                                        // Add under new parent
                                        PlanNode* added_branch =
                                                new_parent_for_ancestor->add_child(std::move(moved_branch));
                                        added_branch->parent = new_parent_for_ancestor;

                                        // Update depths
                                        std::function<void(PlanNode*, uint32_t)> update_depths = [&](PlanNode* node,
                                                                                                     uint32_t depth) {
                                            ftree_depth_map[node->attribute] = depth;
                                            for (auto& child: node->children) {
                                                update_depths(child.get(), depth + 1);
                                            }
                                        };
                                        update_depths(added_branch,
                                                      ftree_depth_map[new_parent_for_ancestor->attribute] + 1);

                                        attr_to_node[added_branch->attribute] = added_branch;

                                        // Update ancestor_node reference
                                        ancestor_node = attr_to_node[ancestor_attr];
                                    }
                                }

                                // Now move descendant under ancestor (if not already)
                                if (descendant_node->parent != ancestor_node) {
                                    PlanNode* desc_old_parent = descendant_node->parent;
                                    if (desc_old_parent) {
                                        auto& desc_children = desc_old_parent->children;
                                        auto desc_it =
                                                std::find_if(desc_children.begin(), desc_children.end(),
                                                             [descendant_node](const std::unique_ptr<PlanNode>& child) {
                                                                 return child.get() == descendant_node;
                                                             });
                                        if (desc_it != desc_children.end()) {
                                            std::unique_ptr<PlanNode> moved_desc = std::move(*desc_it);
                                            desc_children.erase(desc_it);

                                            std::string desc_original_join_from = moved_desc->join_from_attr;
                                            moved_desc->type = PlanNodeType::FLAT_JOIN;
                                            moved_desc->lca_attr = desc_original_join_from;
                                            moved_desc->join_from_attr = ancestor_attr;

                                            PlanNode* added_desc = ancestor_node->add_child(std::move(moved_desc));
                                            added_desc->parent = ancestor_node;
                                            ftree_depth_map[descendant_attr] = ftree_depth_map[ancestor_attr] + 1;
                                            attr_to_node[descendant_attr] = added_desc;
                                        }
                                    }
                                }
                            } else if (descendant_branch_root && descendant_branch_root != descendant_node) {
                                // Original logic: move descendant's branch under ancestor
                                PlanNode* old_parent = descendant_branch_root->parent;
                                std::string branch_original_join_from = descendant_branch_root->join_from_attr;

                                auto& children = old_parent->children;
                                auto it =
                                        std::find_if(children.begin(), children.end(),
                                                     [descendant_branch_root](const std::unique_ptr<PlanNode>& child) {
                                                         return child.get() == descendant_branch_root;
                                                     });
                                if (it != children.end()) {
                                    std::unique_ptr<PlanNode> moved_branch = std::move(*it);
                                    children.erase(it);

                                    moved_branch->type = PlanNodeType::FLAT_JOIN;
                                    moved_branch->lca_attr = branch_original_join_from;
                                    moved_branch->join_from_attr = ancestor_attr;

                                    PlanNode* added_branch = ancestor_node->add_child(std::move(moved_branch));
                                    added_branch->parent = ancestor_node;

                                    std::function<void(PlanNode*, uint32_t)> update_depths = [&](PlanNode* node,
                                                                                                 uint32_t depth) {
                                        ftree_depth_map[node->attribute] = depth;
                                        for (auto& child: node->children) {
                                            update_depths(child.get(), depth + 1);
                                        }
                                    };
                                    update_depths(added_branch, ftree_depth_map[ancestor_attr] + 1);

                                    attr_to_node[added_branch->attribute] = added_branch;
                                }
                            } else {
                                // Direct case: just move descendant_node
                                PlanNode* old_parent = descendant_node->parent;
                                if (old_parent) {
                                    auto& children = old_parent->children;
                                    auto it = std::find_if(children.begin(), children.end(),
                                                           [descendant_node](const std::unique_ptr<PlanNode>& child) {
                                                               return child.get() == descendant_node;
                                                           });
                                    if (it != children.end()) {
                                        std::unique_ptr<PlanNode> moved_child = std::move(*it);
                                        children.erase(it);

                                        moved_child->type = PlanNodeType::FLAT_JOIN;
                                        moved_child->join_from_attr = ancestor_attr;
                                        moved_child->lca_attr = lca_attr;
                                        moved_child->is_fwd = get_edge_direction(lca_attr, descendant_attr, edges);

                                        PlanNode* added_node = ancestor_node->add_child(std::move(moved_child));
                                        added_node->parent = ancestor_node;
                                        ftree_depth_map[descendant_attr] = ftree_depth_map[ancestor_attr] + 1;
                                        attr_to_node[descendant_attr] = added_node;
                                    }
                                }
                            }
                        }

                        // Add to sorted theta joins list (will be sorted later)
                        plan_tree->get_sorted_theta_joins().emplace_back(ancestor_attr, descendant_attr, adjusted_op);

                        // Also add to descendant node's theta_joins_after for backward compatibility
                        PlanNode* desc_node = attr_to_node[descendant_attr];
                        if (desc_node) {
                            desc_node->theta_joins_after.emplace_back(ancestor_attr, descendant_attr, adjusted_op);
                        }
                    }
                } else if (pred.type == PredicateType::NOT_EXISTS) {
                    // NOT EXISTS predicate: NOT (left_attr)->(right_attr)
                    PlanNode* left_node = attr_to_node[pred.left_attr];
                    PlanNode* right_node = attr_to_node[pred.right_attr];

                    if (!left_node || !right_node) {
                        std::cerr << "Warning: Cannot find nodes for NOT EXISTS predicate: " << pred.left_attr << " -> "
                                  << pred.right_attr << std::endl;
                        continue;
                    }

                    // Determine ancestor/descendant relationship
                    std::string ancestor_attr, descendant_attr;

                    if (left_node->depth < right_node->depth) {
                        ancestor_attr = pred.left_attr;
                        descendant_attr = pred.right_attr;
                    } else {
                        ancestor_attr = pred.right_attr;
                        descendant_attr = pred.left_attr;
                    }

                    // Add anti-semi-join to descendant node
                    PlanNode* desc_node = attr_to_node[descendant_attr];
                    desc_node->anti_semi_joins_after.emplace_back(ancestor_attr, descendant_attr);
                }
            }
        }
    }

    // Sort theta joins by first operand position in ordering
    auto& theta_joins = plan_tree->get_sorted_theta_joins();
    std::sort(theta_joins.begin(), theta_joins.end(), [&ordering_idx](const ThetaJoinInfo& a, const ThetaJoinInfo& b) {
        auto a_pos = ordering_idx.find(a.ancestor_attr);
        auto b_pos = ordering_idx.find(b.ancestor_attr);
        if (a_pos == ordering_idx.end() || b_pos == ordering_idx.end()) {
            return false;// Keep original order if not found
        }
        return a_pos->second < b_pos->second;
    });

    plan_tree->set_root(std::move(root_node));

    // Finalize: update node types based on predicates
    plan_tree->finalize();

    return plan_tree;
}

// =============================================================================
// Phase 2: Create Operators from Plan Tree
// =============================================================================

// Helper: Get table cardinality for a join
// IMPORTANT: is_fwd indicates the direction of the join relative to the table's definition
// - For a table T(A, B) with cardinality c:
//   - Forward (A->B): returns c
//   - Backward (B->A): returns inverse of c
static Cardinality get_table_cardinality(const std::string& join_from_attr, const std::string& output_attr,
                                         const std::vector<const Table*>& tables, bool is_fwd) {
    // Find matching table
    for (const auto* table: tables) {
        if (table->columns.size() < 2) continue;

        // Table is defined as T(col0, col1) with some cardinality
        // Forward direction: col0 -> col1 uses the defined cardinality
        // Backward direction: col1 -> col0 uses the inverse cardinality

        if (is_fwd) {
            // Forward: join_from should be col0, output should be col1
            if (table->columns[0] == join_from_attr && table->columns[1] == output_attr) { return table->cardinality; }
        } else {
            // Backward: join_from should be col1, output should be col0
            if (table->columns[1] == join_from_attr && table->columns[0] == output_attr) {
                // Return the cardinality as-is - the caller will handle direction
                return table->cardinality;
            }
        }
    }
    return Cardinality::MANY_TO_MANY;// Default if not found
}

// Helper: Determine if state should be shared
static bool should_use_shared_operator(const PlanNode* node, const std::vector<const Table*>& tables) {
    if (node->join_from_attr.empty() || node->attribute.empty()) {
        return false;// Not a join node
    }

    Cardinality card = get_table_cardinality(node->join_from_attr, node->attribute, tables, node->is_fwd);
    return should_share_state(card, node->is_fwd);
}

// Helper: Check if the path from lca_attr to join_from_attr in the plan tree consists of n:1 edges
// If so, they share state (same DataChunk) and FlatJoin is unnecessary
static bool path_shares_state(const std::string& lca_attr, const std::string& join_from_attr,
                              const std::unordered_map<std::string, const PlanNode*>& attr_to_node,
                              const std::vector<const Table*>& tables) {
    if (lca_attr == join_from_attr) {
        return true;// Same attribute, trivially shares state
    }

    // Find the node for join_from_attr
    auto join_from_it = attr_to_node.find(join_from_attr);
    if (join_from_it == attr_to_node.end()) {
        return false;// Node not found
    }

    // Traverse up from join_from_attr to lca_attr, checking each edge's cardinality
    const PlanNode* current = join_from_it->second;

    // We need to traverse the tree to find the path from lca to join_from
    // The join_from_attr is the tree parent, lca_attr is higher up
    // So we need to find lca_attr by traversing the join chain

    // Build the path by going up via join_from_attr links
    std::vector<std::pair<std::string, std::string>> edges;// (from, to) pairs

    std::string curr_attr = join_from_attr;
    while (curr_attr != lca_attr) {
        auto it = attr_to_node.find(curr_attr);
        if (it == attr_to_node.end()) {
            return false;// Path broken
        }

        const PlanNode* node = it->second;
        if (node->join_from_attr.empty()) {
            return false;// No more parent to check
        }

        edges.push_back({node->join_from_attr, curr_attr});
        curr_attr = node->join_from_attr;

        // Safety: prevent infinite loop
        if (edges.size() > 100) { return false; }
    }

    // Check if all edges share state
    for (const auto& [from_attr, to_attr]: edges) {
        // Determine direction: we're checking if from_attr -> to_attr is n:1
        // Find the table and check cardinality
        Cardinality card = get_table_cardinality(from_attr, to_attr, tables, true);
        if (!should_share_state(card, true)) {
            // Try backward direction
            card = get_table_cardinality(to_attr, from_attr, tables, false);
            if (!should_share_state(card, false)) {
                return false;// This edge doesn't share state
            }
        }
    }

    return true;// All edges share state
}

// Helper: Create operator from plan node
static std::unique_ptr<Operator>
create_operator_from_node(const PlanNode* node, const std::vector<const Table*>& tables,
                          const std::unordered_map<std::string, const PlanNode*>& attr_to_node = {},
                          bool use_synchronized = false, uint64_t start_id = 0) {
    switch (node->type) {
        case PlanNodeType::SCAN:
            if (use_synchronized) { return std::make_unique<ScanSynchronized>(node->attribute, start_id); }
            return std::make_unique<Scan<uint64_t>>(node->attribute);

        case PlanNodeType::SCAN_UNPACKED:
            if (use_synchronized) { return std::make_unique<ScanUnpackedSynchronized>(node->attribute, start_id); }
            return std::make_unique<ScanUnpacked<uint64_t>>(node->attribute);

        case PlanNodeType::SCAN_UNPACKED_SYNCHRONIZED:
            return std::make_unique<ScanUnpackedSynchronized>(node->attribute, start_id);

        case PlanNodeType::SCAN_PREDICATED:
            if (use_synchronized) {
                return std::make_unique<ScanSynchronizedPredicated<uint64_t>>(node->attribute, start_id,
                                                                              node->scalar_predicates);
            }
            return std::make_unique<ScanPredicated<uint64_t>>(node->attribute, node->scalar_predicates);

        case PlanNodeType::SCAN_SYNCHRONIZED:
            return std::make_unique<ScanSynchronized>(node->attribute, start_id);

        case PlanNodeType::SCAN_SYNCHRONIZED_PREDICATED:
            return std::make_unique<ScanSynchronizedPredicated<uint64_t>>(node->attribute, start_id,
                                                                          node->scalar_predicates);

        case PlanNodeType::INL_JOIN_PACKED: {
            if (should_use_shared_operator(node, tables)) {
                return std::make_unique<INLJoinPackedShared<uint64_t>>(node->join_from_attr, node->attribute,
                                                                       node->is_fwd);
            } else {
                return std::make_unique<INLJoinPacked<uint64_t>>(node->join_from_attr, node->attribute, node->is_fwd);
            }
        }

        case PlanNodeType::INL_JOIN_PACKED_PREDICATED: {
            if (should_use_shared_operator(node, tables)) {
                return std::make_unique<INLJoinPackedPredicatedShared<uint64_t>>(node->join_from_attr, node->attribute,
                                                                                 node->is_fwd, node->scalar_predicates);
            } else {
                return std::make_unique<INLJoinPackedPredicated<uint64_t>>(node->join_from_attr, node->attribute,
                                                                           node->is_fwd, node->scalar_predicates);
            }
        }

        case PlanNodeType::INL_JOIN_UNPACKED:
            return std::make_unique<INLJoinUnpacked<uint64_t>>(node->join_from_attr, node->attribute, node->is_fwd);

        case PlanNodeType::INL_JOIN_PACKED_CASCADE: {
            if (should_use_shared_operator(node, tables)) {
                return std::make_unique<INLJoinPackedCascadeShared<uint64_t>>(node->join_from_attr, node->attribute,
                                                                              node->is_fwd);
            } else {
                return std::make_unique<INLJoinPackedCascade<uint64_t>>(node->join_from_attr, node->attribute,
                                                                        node->is_fwd);
            }
        }

        case PlanNodeType::INL_JOIN_PACKED_CASCADE_PREDICATED: {
            if (should_use_shared_operator(node, tables)) {
                return std::make_unique<INLJoinPackedCascadePredicatedShared<uint64_t>>(
                        node->join_from_attr, node->attribute, node->is_fwd, node->scalar_predicates);
            } else {
                return std::make_unique<INLJoinPackedCascadePredicated<uint64_t>>(
                        node->join_from_attr, node->attribute, node->is_fwd, node->scalar_predicates);
            }
        }

        case PlanNodeType::INL_JOIN_PACKED_GP_CASCADE: {
            if (should_use_shared_operator(node, tables)) {
                return std::make_unique<INLJoinPackedGPCascadeShared<uint64_t>>(node->join_from_attr, node->attribute,
                                                                                node->is_fwd);
            } else {
                return std::make_unique<INLJoinPackedGPCascade<uint64_t>>(node->join_from_attr, node->attribute,
                                                                          node->is_fwd);
            }
        }

        case PlanNodeType::INL_JOIN_PACKED_GP_CASCADE_PREDICATED: {
            if (should_use_shared_operator(node, tables)) {
                return std::make_unique<INLJoinPackedGPCascadePredicatedShared<uint64_t>>(
                        node->join_from_attr, node->attribute, node->is_fwd, node->scalar_predicates);
            } else {
                return std::make_unique<INLJoinPackedGPCascadePredicated<uint64_t>>(
                        node->join_from_attr, node->attribute, node->is_fwd, node->scalar_predicates);
            }
        }

        case PlanNodeType::INL_JOIN_PACKED_SHARED:
            return std::make_unique<INLJoinPackedShared<uint64_t>>(node->join_from_attr, node->attribute, node->is_fwd);

        case PlanNodeType::INL_JOIN_PACKED_PREDICATED_SHARED:
            return std::make_unique<INLJoinPackedPredicatedShared<uint64_t>>(node->join_from_attr, node->attribute,
                                                                             node->is_fwd, node->scalar_predicates);

        case PlanNodeType::INL_JOIN_PACKED_CASCADE_SHARED:
            return std::make_unique<INLJoinPackedCascadeShared<uint64_t>>(node->join_from_attr, node->attribute,
                                                                          node->is_fwd);

        case PlanNodeType::INL_JOIN_PACKED_CASCADE_PREDICATED_SHARED:
            return std::make_unique<INLJoinPackedCascadePredicatedShared<uint64_t>>(
                    node->join_from_attr, node->attribute, node->is_fwd, node->scalar_predicates);

        case PlanNodeType::INL_JOIN_PACKED_GP_CASCADE_SHARED:
            return std::make_unique<INLJoinPackedGPCascadeShared<uint64_t>>(node->join_from_attr, node->attribute,
                                                                            node->is_fwd);

        case PlanNodeType::INL_JOIN_PACKED_GP_CASCADE_PREDICATED_SHARED:
            return std::make_unique<INLJoinPackedGPCascadePredicatedShared<uint64_t>>(
                    node->join_from_attr, node->attribute, node->is_fwd, node->scalar_predicates);

        case PlanNodeType::FLAT_JOIN:
            // Check if lca_attr and join_from_attr are in the same DataChunk (share state)
            // If so, FlatJoin is unnecessary - use INLJoinPacked directly from lca_attr
            if (!attr_to_node.empty() &&
                path_shares_state(node->lca_attr, node->join_from_attr, attr_to_node, tables)) {
                // They share state - use regular INLJoinPacked from lca_attr
                return std::make_unique<INLJoinPacked<uint64_t>>(node->lca_attr, node->attribute, node->is_fwd);
            }
            return std::make_unique<FlatJoin<uint64_t>>(node->join_from_attr, node->lca_attr, node->attribute,
                                                        node->is_fwd);

        case PlanNodeType::FLAT_JOIN_PREDICATED:
            // Check if lca_attr and join_from_attr are in the same DataChunk (share state)
            if (!attr_to_node.empty() &&
                path_shares_state(node->lca_attr, node->join_from_attr, attr_to_node, tables)) {
                // They share state - use regular INLJoinPackedPredicated from lca_attr
                return std::make_unique<INLJoinPackedPredicated<uint64_t>>(node->lca_attr, node->attribute,
                                                                           node->is_fwd, node->scalar_predicates);
            }
            return std::make_unique<FlatJoinPredicated<uint64_t>>(node->join_from_attr, node->lca_attr, node->attribute,
                                                                  node->is_fwd, node->scalar_predicates);

        case PlanNodeType::INTERSECTION:
            return std::make_unique<Intersection<uint64_t>>(
                    node->intersection_inputs[0].first,  // ancestor_attr
                    node->intersection_inputs[1].first,  // descendant_attr
                    node->attribute,                     // output_attr
                    node->intersection_inputs[0].second, // is_ancestor_join_fwd
                    node->intersection_inputs[1].second);// is_descendant_join_fwd

        case PlanNodeType::INTERSECTION_PREDICATED:
            return std::make_unique<IntersectionPredicated<uint64_t>>(
                    node->intersection_inputs[0].first, // ancestor_attr
                    node->intersection_inputs[1].first, // descendant_attr
                    node->attribute,                    // output_attr
                    node->intersection_inputs[0].second,// is_ancestor_join_fwd
                    node->intersection_inputs[1].second,// is_descendant_join_fwd
                    node->scalar_predicates);

        case PlanNodeType::NWAY_INTERSECTION:
            return std::make_unique<NWayIntersection<uint64_t>>(node->attribute, node->intersection_inputs);

        case PlanNodeType::NWAY_INTERSECTION_PREDICATED:
            return std::make_unique<NWayIntersectionPredicated<uint64_t>>(node->attribute, node->intersection_inputs,
                                                                          node->scalar_predicates);

        default:
            throw std::runtime_error("Unknown plan node type: " + plan_node_type_to_string(node->type));
    }
}

// INL packed joins with *_SHARED reuse the parent key's DataChunk; they are not a second
// independent branch for pipeline linearity (SinkLinear) purposes.
static bool is_inl_join_shared_state_type(PlanNodeType type) {
    switch (type) {
        case PlanNodeType::INL_JOIN_PACKED_SHARED:
        case PlanNodeType::INL_JOIN_PACKED_PREDICATED_SHARED:
        case PlanNodeType::INL_JOIN_PACKED_CASCADE_SHARED:
        case PlanNodeType::INL_JOIN_PACKED_CASCADE_PREDICATED_SHARED:
        case PlanNodeType::INL_JOIN_PACKED_GP_CASCADE_SHARED:
        case PlanNodeType::INL_JOIN_PACKED_GP_CASCADE_PREDICATED_SHARED:
            return true;
        default:
            return false;
    }
}

static bool shares_datachunk_with_parent(const PlanNode* child, const PlanNode* parent) {
    if (!child || !parent) return false;
    if (!is_inl_join_shared_state_type(child->type)) return false;
    return child->join_from_attr == parent->attribute;
}

// Linear if each node has at most one structural child; shared-state siblings are skipped.
static bool is_plan_tree_linear(const PlanNode* node) {
    if (!node) return true;
    std::vector<const PlanNode*> structural_children;
    structural_children.reserve(node->children.size());
    for (const auto& ch : node->children) {
        const PlanNode* c = ch.get();
        if (shares_datachunk_with_parent(c, node)) {
            if (!is_plan_tree_linear(c)) return false;
            continue;
        }
        structural_children.push_back(c);
    }
    if (structural_children.size() > 1) return false;
    if (structural_children.empty()) return true;
    return is_plan_tree_linear(structural_children[0]);
}

static std::unique_ptr<Operator> create_sink_operator(sink_type sink) {
    switch (sink) {
        case SINK_PACKED:
            return std::make_unique<SinkPacked>();
        case SINK_LINEAR:
            return std::make_unique<SinkLinear>();
        case SINK_EXPORT_CSV:
            return std::make_unique<SinkExport>(SinkExport::Format::CSV);
        case SINK_EXPORT_JSON:
            return std::make_unique<SinkExport>(SinkExport::Format::JSON);
        case SINK_EXPORT_MARKDOWN:
            return std::make_unique<SinkExport>(SinkExport::Format::MARKDOWN);
        case SINK_PACKED_NOOP:
        case SINK_UNPACKED_NOOP:
            return std::make_unique<SinkNoOp>();
        case SINK_MIN:
            return std::make_unique<SinkMinItr>();
        default:
            return std::make_unique<SinkUnpacked>();
    }
}

// Helper: Create theta join operators and chain them
static Operator* add_theta_joins(Operator* last_op, const std::vector<ThetaJoinInfo>& theta_joins) {
    for (const auto& tj: theta_joins) {
        auto theta_op = std::make_unique<PackedThetaJoin<uint64_t>>(tj.ancestor_attr, tj.descendant_attr, tj.op);
        last_op->set_next_operator(std::move(theta_op));
        last_op = last_op->next_op;
    }
    return last_op;
}

// Helper: Create anti-semi-join operators and chain them
static Operator* add_anti_semi_joins(Operator* last_op, const std::vector<AntiSemiJoinInfo>& anti_semi_joins) {
    for (const auto& asj: anti_semi_joins) {
        auto asj_op = std::make_unique<PackedAntiSemiJoin<uint64_t>>(asj.ancestor_attr, asj.descendant_attr);
        last_op->set_next_operator(std::move(asj_op));
        last_op = last_op->next_op;
    }
    return last_op;
}

// Force the last eligible INL packed join in DFS order to be a cascaded variant.
// This affects the join immediately before sink in the linearized operator chain.
static bool is_cascadable_inl_packed_join_type(PlanNodeType type) {
    switch (type) {
        case PlanNodeType::INL_JOIN_PACKED:
        case PlanNodeType::INL_JOIN_PACKED_PREDICATED:
        case PlanNodeType::INL_JOIN_PACKED_SHARED:
        case PlanNodeType::INL_JOIN_PACKED_PREDICATED_SHARED:
        case PlanNodeType::INL_JOIN_PACKED_CASCADE:
        case PlanNodeType::INL_JOIN_PACKED_CASCADE_PREDICATED:
        case PlanNodeType::INL_JOIN_PACKED_CASCADE_SHARED:
        case PlanNodeType::INL_JOIN_PACKED_CASCADE_PREDICATED_SHARED:
        case PlanNodeType::INL_JOIN_PACKED_GP_CASCADE:
        case PlanNodeType::INL_JOIN_PACKED_GP_CASCADE_PREDICATED:
        case PlanNodeType::INL_JOIN_PACKED_GP_CASCADE_SHARED:
        case PlanNodeType::INL_JOIN_PACKED_GP_CASCADE_PREDICATED_SHARED:
            return true;
        default:
            return false;
    }
}

static PlanNodeType to_cascade_variant(PlanNodeType type) {
    switch (type) {
        case PlanNodeType::INL_JOIN_PACKED:
            return PlanNodeType::INL_JOIN_PACKED_CASCADE;
        case PlanNodeType::INL_JOIN_PACKED_PREDICATED:
            return PlanNodeType::INL_JOIN_PACKED_CASCADE_PREDICATED;
        case PlanNodeType::INL_JOIN_PACKED_SHARED:
            return PlanNodeType::INL_JOIN_PACKED_CASCADE_SHARED;
        case PlanNodeType::INL_JOIN_PACKED_PREDICATED_SHARED:
            return PlanNodeType::INL_JOIN_PACKED_CASCADE_PREDICATED_SHARED;
        case PlanNodeType::INL_JOIN_PACKED_CASCADE:
        case PlanNodeType::INL_JOIN_PACKED_CASCADE_PREDICATED:
        case PlanNodeType::INL_JOIN_PACKED_CASCADE_SHARED:
        case PlanNodeType::INL_JOIN_PACKED_CASCADE_PREDICATED_SHARED:
        case PlanNodeType::INL_JOIN_PACKED_GP_CASCADE:
        case PlanNodeType::INL_JOIN_PACKED_GP_CASCADE_PREDICATED:
        case PlanNodeType::INL_JOIN_PACKED_GP_CASCADE_SHARED:
        case PlanNodeType::INL_JOIN_PACKED_GP_CASCADE_PREDICATED_SHARED:
            return type;
        default:
            return type;
    }
}

static void force_last_join_cascaded(PlanNode* root) {
    if (!root) return;

    // Sink predecessor in this linearized execution model is the rightmost leaf,
    // unless that leaf has post-join operators (theta/anti-semi/property-filter).
    PlanNode* sink_predecessor_node = root;
    while (!sink_predecessor_node->children.empty()) {
        sink_predecessor_node = sink_predecessor_node->children.back().get();
    }

    if (sink_predecessor_node->has_theta_joins() || sink_predecessor_node->has_anti_semi_joins() ||
        sink_predecessor_node->has_property_filters()) {
        return;
    }

    if (is_cascadable_inl_packed_join_type(sink_predecessor_node->type)) {
        sink_predecessor_node->type = to_cascade_variant(sink_predecessor_node->type);
    }
}

// Advance through original_ordering from order_i. For each slot: skip _cd; skip attributes already
// joined. When the slot is the LLM output name: require every context_column attribute in processed_attrs,
// append Map, or throw. If the slot is a relation attribute not yet joined, stop (more operators coming).
// (build_plan_tree already validated that llm_output_attr appears in ordering and context attrs precede it.)
static void insert_map(const Query& query, const std::vector<std::string>& original_ordering,
                       const std::set<std::string>& llm_context_attrs, const std::string& llm_output_attr,
                       std::set<std::string>& processed_attrs, size_t& order_i, Operator*& last_op,
                       bool& map_inserted) {
    if (!query.has_llm_map()) return;
    for (; order_i < original_ordering.size(); ++order_i) {
        const std::string& slot = original_ordering[order_i];
        if (slot == "_cd") { continue; }
        if (slot == llm_output_attr) {
            if (map_inserted) { continue; }
            for (const auto& req : llm_context_attrs) {
                if (processed_attrs.count(req) == 0) {
                    throw std::runtime_error(
                            "LLM_MAP: column ordering reaches '" + llm_output_attr +
                            "' before all context_column attributes are joined; missing '" + req + "'.");
                }
            }
            last_op->set_next_operator(std::make_unique<Map>());
            last_op = last_op->next_op;
            map_inserted = true;
            continue;
        }
        if (processed_attrs.count(slot) > 0) { continue; }
        break;
    }
}

std::pair<std::unique_ptr<Plan>, std::shared_ptr<FactorizedTreeElement>>
create_operators_from_plan_tree(const PlanTree& plan_tree, const Query& query,
                                const std::vector<std::string>& original_ordering,
                                const std::vector<const Table*>& tables) {
    const PlanNode* root = plan_tree.get_root();
    if (!root) { throw std::runtime_error("Plan tree has no root"); }

    // attr_to_node from plan tree DFS (no PlanNode for synthetic LLM output; CLI ordering is Plan column order).
    std::unordered_map<std::string, const PlanNode*> attr_to_node;
    std::function<void(const PlanNode*)> collect_attr_to_node = [&](const PlanNode* node) {
        attr_to_node[node->attribute] = node;
        for (const auto& child: node->children) {
            collect_attr_to_node(child.get());
        }
    };
    collect_attr_to_node(root);

    // Create ftree
    auto ftree = std::make_shared<FactorizedTreeElement>(root->attribute);

    // Create operator pipeline using DFS
    std::unique_ptr<Operator> first_op = create_operator_from_node(root, tables, attr_to_node);
    Operator* last_op = first_op.get();

    // Add anti-semi-joins after root if any
    if (root->has_anti_semi_joins()) { last_op = add_anti_semi_joins(last_op, root->anti_semi_joins_after); }

    // Add property filter after root if any
    if (root->has_property_filters()) {
        std::vector<PropertyLookupInfo> lookups;
        for (const auto& pfi : root->property_filter_infos) {
            lookups.push_back({pfi.property_attr, pfi.entity_attr, pfi.is_fwd});
        }
        auto pf_op = std::make_unique<PropertyFilter>(root->attribute, std::move(lookups),
                                                       root->property_predicate_expr);
        last_op->set_next_operator(std::move(pf_op));
        last_op = last_op->next_op;
    }

    // Track which attributes have been processed (have operators created)
    std::set<std::string> processed_attrs;
    processed_attrs.insert(root->attribute);

    // Track which theta joins have been inserted
    std::set<size_t> inserted_theta_joins;
    const auto& sorted_theta_joins = plan_tree.get_sorted_theta_joins();

    bool map_inserted = false;
    std::set<std::string> llm_context_attrs;
    std::string llm_output_attr;
    size_t order_i = 0;
    if (query.has_llm_map()) {
        llm_context_attrs = parse_llm_context_column_attrs(query);
        llm_output_attr = query.get_llm_map_output_attr();
    }

    // Check if any theta joins can be inserted after root
    for (size_t i = 0; i < sorted_theta_joins.size(); ++i) {
        if (inserted_theta_joins.count(i) > 0) continue;

        const auto& tj = sorted_theta_joins[i];
        // Check if both operands are processed and descendant matches root attribute
        if (processed_attrs.count(tj.ancestor_attr) > 0 && processed_attrs.count(tj.descendant_attr) > 0 &&
            tj.descendant_attr == root->attribute) {
            // Insert this theta join now
            auto theta_op = std::make_unique<PackedThetaJoin<uint64_t>>(tj.ancestor_attr, tj.descendant_attr, tj.op);
            last_op->set_next_operator(std::move(theta_op));
            last_op = last_op->next_op;
            inserted_theta_joins.insert(i);
        }
    }

    insert_map(query, original_ordering, llm_context_attrs, llm_output_attr, processed_attrs, order_i, last_op,
               map_inserted);

    // Process children recursively
    // Theta joins are inserted immediately after the descendant node is created,
    // as long as both operands have been processed
    std::function<void(const PlanNode*)> process_children = [&](const PlanNode* node) {
        for (const auto& child: node->children) {
            // Create operator for child
            auto child_op = create_operator_from_node(child.get(), tables, attr_to_node);
            last_op->set_next_operator(std::move(child_op));
            last_op = last_op->next_op;

            // Mark this attribute as processed
            processed_attrs.insert(child->attribute);

            // Check if any theta joins can be inserted now
            // A theta join can be inserted if:
            // 1. Both ancestor and descendant attributes are processed
            // 2. The descendant attribute matches the current node's attribute
            // 3. The theta join hasn't been inserted yet
            for (size_t i = 0; i < sorted_theta_joins.size(); ++i) {
                if (inserted_theta_joins.count(i) > 0) continue;// Already inserted

                const auto& tj = sorted_theta_joins[i];
                // Check if both operands are processed and descendant matches current attribute
                if (processed_attrs.count(tj.ancestor_attr) > 0 && processed_attrs.count(tj.descendant_attr) > 0 &&
                    tj.descendant_attr == child->attribute) {
                    // Insert this theta join now
                    auto theta_op =
                            std::make_unique<PackedThetaJoin<uint64_t>>(tj.ancestor_attr, tj.descendant_attr, tj.op);
                    last_op->set_next_operator(std::move(theta_op));
                    last_op = last_op->next_op;
                    inserted_theta_joins.insert(i);
                }
            }

            // Add anti-semi-joins after this node (similar logic)
            if (child->has_anti_semi_joins()) {
                for (const auto& asj: child->anti_semi_joins_after) {
                    if (processed_attrs.count(asj.ancestor_attr) > 0) {
                        auto asj_op =
                                std::make_unique<PackedAntiSemiJoin<uint64_t>>(asj.ancestor_attr, asj.descendant_attr);
                        last_op->set_next_operator(std::move(asj_op));
                        last_op = last_op->next_op;
                    }
                }
            }

            // Add property filter after this node if any
            if (child->has_property_filters()) {
                std::vector<PropertyLookupInfo> lookups;
                for (const auto& pfi : child->property_filter_infos) {
                    lookups.push_back({pfi.property_attr, pfi.entity_attr, pfi.is_fwd});
                }
                auto pf_op = std::make_unique<PropertyFilter>(child->attribute, std::move(lookups),
                                                               child->property_predicate_expr);
                last_op->set_next_operator(std::move(pf_op));
                last_op = last_op->next_op;
            }

            insert_map(query, original_ordering, llm_context_attrs, llm_output_attr, processed_attrs, order_i, last_op,
                       map_inserted);

            // Process grandchildren
            process_children(child.get());
        }
    };
    process_children(root);

    insert_map(query, original_ordering, llm_context_attrs, llm_output_attr, processed_attrs, order_i, last_op,
               map_inserted);

    // Determine final sink type
    // If tree is linear and is_packed=true (not SINK_MIN), use SINK_LINEAR
    sink_type final_sink = plan_tree.get_sink_type();
    bool is_linear = is_plan_tree_linear(root);
    bool is_packed = (final_sink == SINK_PACKED);

    // SinkLinear expects the last column_ordering attribute to have a
    // DataChunk-backed vector in the schema map. LLM_MAP adds a synthetic
    // output attribute that SinkPacked supports but SinkLinear does not.
    if (is_linear && is_packed && !query.has_llm_map()) { final_sink = SINK_LINEAR; }

    // Create and add sink
    auto sink_op = create_sink_operator(final_sink);
    last_op->set_next_operator(std::move(sink_op));

    return std::make_pair(std::make_unique<Plan>(std::move(first_op), original_ordering), ftree);
}

// =============================================================================
// Combined: Two-Phase Plan Creation
// =============================================================================

std::pair<std::unique_ptr<Plan>, std::shared_ptr<FactorizedTreeElement>>
map_ordering_to_plan(const Query& query, const std::vector<std::string>& ordering, sink_type sink,
                     const std::vector<const Table*>& tables) {
    // Phase 1: Build plan tree
    auto plan_tree = build_plan_tree(query, ordering, sink);

    // Force to cascade only when the sink predecessor is a packed INL join.
    force_last_join_cascaded(plan_tree->get_root());

    // Phase 2: Create operators (including optional AI pipeline)
    return create_operators_from_plan_tree(*plan_tree, query, ordering, tables);
}

// =============================================================================
// Synchronized Version: Create Operators with ScanSynchronized at root
// =============================================================================

std::pair<std::unique_ptr<Plan>, std::shared_ptr<FactorizedTreeElement>>
create_operators_from_plan_tree_synchronized(const PlanTree& plan_tree, const Query& query,
                                             const std::vector<std::string>& original_ordering,
                                             const std::vector<const Table*>& tables, uint64_t start_id) {
    const PlanNode* root = plan_tree.get_root();
    if (!root) { throw std::runtime_error("Plan tree has no root"); }

    std::unordered_map<std::string, const PlanNode*> attr_to_node;
    std::function<void(const PlanNode*)> collect_attr_to_node = [&](const PlanNode* node) {
        attr_to_node[node->attribute] = node;
        for (const auto& child: node->children) {
            collect_attr_to_node(child.get());
        }
    };
    collect_attr_to_node(root);

    // Create ftree
    auto ftree = std::make_shared<FactorizedTreeElement>(root->attribute);

    // Create operator pipeline using DFS
    // Use synchronized scan for root (pass use_synchronized=true, start_id)
    std::unique_ptr<Operator> first_op =
            create_operator_from_node(root, tables, attr_to_node, true, start_id);
    Operator* last_op = first_op.get();

    // Add anti-semi-joins after root if any
    if (root->has_anti_semi_joins()) { last_op = add_anti_semi_joins(last_op, root->anti_semi_joins_after); }

    // Add property filter after root if any
    if (root->has_property_filters()) {
        std::vector<PropertyLookupInfo> lookups;
        for (const auto& pfi : root->property_filter_infos) {
            lookups.push_back({pfi.property_attr, pfi.entity_attr, pfi.is_fwd});
        }
        auto pf_op = std::make_unique<PropertyFilter>(root->attribute, std::move(lookups),
                                                       root->property_predicate_expr);
        last_op->set_next_operator(std::move(pf_op));
        last_op = last_op->next_op;
    }

    // Track which attributes have been processed (have operators created)
    std::set<std::string> processed_attrs;
    processed_attrs.insert(root->attribute);

    // Track which theta joins have been inserted
    std::set<size_t> inserted_theta_joins;
    const auto& sorted_theta_joins = plan_tree.get_sorted_theta_joins();

    bool map_inserted = false;
    std::set<std::string> llm_context_attrs;
    std::string llm_output_attr;
    size_t order_i = 0;
    if (query.has_llm_map()) {
        llm_context_attrs = parse_llm_context_column_attrs(query);
        llm_output_attr = query.get_llm_map_output_attr();
    }

    // Check if any theta joins can be inserted after root
    for (size_t i = 0; i < sorted_theta_joins.size(); ++i) {
        if (inserted_theta_joins.count(i) > 0) continue;

        const auto& tj = sorted_theta_joins[i];
        // Check if both operands are processed and descendant matches root attribute
        if (processed_attrs.count(tj.ancestor_attr) > 0 && processed_attrs.count(tj.descendant_attr) > 0 &&
            tj.descendant_attr == root->attribute) {
            // Insert this theta join now
            auto theta_op = std::make_unique<PackedThetaJoin<uint64_t>>(tj.ancestor_attr, tj.descendant_attr, tj.op);
            last_op->set_next_operator(std::move(theta_op));
            last_op = last_op->next_op;
            inserted_theta_joins.insert(i);
        }
    }

    insert_map(query, original_ordering, llm_context_attrs, llm_output_attr, processed_attrs, order_i, last_op,
               map_inserted);

    // Process children recursively (these use regular operators, not synchronized)
    std::function<void(const PlanNode*)> process_children = [&](const PlanNode* node) {
        for (const auto& child: node->children) {
            // Create operator for child (regular, not synchronized)
            auto child_op = create_operator_from_node(child.get(), tables, attr_to_node, false, 0);
            last_op->set_next_operator(std::move(child_op));
            last_op = last_op->next_op;

            // Mark this attribute as processed
            processed_attrs.insert(child->attribute);

            // Check if any theta joins can be inserted now
            for (size_t i = 0; i < sorted_theta_joins.size(); ++i) {
                if (inserted_theta_joins.count(i) > 0) continue;

                const auto& tj = sorted_theta_joins[i];
                // Check if both operands are processed and descendant matches current attribute
                if (processed_attrs.count(tj.ancestor_attr) > 0 && processed_attrs.count(tj.descendant_attr) > 0 &&
                    tj.descendant_attr == child->attribute) {
                    // Insert this theta join now
                    auto theta_op =
                            std::make_unique<PackedThetaJoin<uint64_t>>(tj.ancestor_attr, tj.descendant_attr, tj.op);
                    last_op->set_next_operator(std::move(theta_op));
                    last_op = last_op->next_op;
                    inserted_theta_joins.insert(i);
                }
            }

            // Add anti-semi-joins after this node if any
            if (child->has_anti_semi_joins()) { last_op = add_anti_semi_joins(last_op, child->anti_semi_joins_after); }

            // Add property filter after this node if any
            if (child->has_property_filters()) {
                std::vector<PropertyLookupInfo> lookups;
                for (const auto& pfi : child->property_filter_infos) {
                    lookups.push_back({pfi.property_attr, pfi.entity_attr, pfi.is_fwd});
                }
                auto pf_op = std::make_unique<PropertyFilter>(child->attribute, std::move(lookups),
                                                               child->property_predicate_expr);
                last_op->set_next_operator(std::move(pf_op));
                last_op = last_op->next_op;
            }

            insert_map(query, original_ordering, llm_context_attrs, llm_output_attr, processed_attrs, order_i, last_op,
                       map_inserted);

            // Process grandchildren
            process_children(child.get());
        }
    };
    process_children(root);

    insert_map(query, original_ordering, llm_context_attrs, llm_output_attr, processed_attrs, order_i, last_op,
               map_inserted);

    // Determine final sink type
    // If tree is linear and is_packed=true (not SINK_MIN), use SINK_LINEAR
    sink_type final_sink = plan_tree.get_sink_type();
    bool is_linear = is_plan_tree_linear(root);
    bool is_packed = (final_sink == SINK_PACKED);

    if (is_linear && is_packed && !query.has_llm_map()) { final_sink = SINK_LINEAR; }

    // Create and add sink
    auto sink_op = create_sink_operator(final_sink);
    last_op->set_next_operator(std::move(sink_op));

    return std::make_pair(std::make_unique<Plan>(std::move(first_op), original_ordering), ftree);
}

std::pair<std::unique_ptr<Plan>, std::shared_ptr<FactorizedTreeElement>>
map_ordering_to_plan_synchronized(const Query& query, const std::vector<std::string>& ordering, sink_type sink,
                                    uint64_t start_id, const std::vector<const Table*>& tables) {
    // Phase 1: Build plan tree
    auto plan_tree = build_plan_tree(query, ordering, sink);

    // Force to cascade only when the sink predecessor is a packed INL join.
    force_last_join_cascaded(plan_tree->get_root());

    // Phase 2: Create operators with synchronized scan at root (including AI pipeline)
    return create_operators_from_plan_tree_synchronized(*plan_tree, query, ordering, tables, start_id);
}

std::vector<std::string> enumerate_orderings(const std::string& query_str) {
    Query query{query_str};
    auto query_variables = query.get_unique_query_variables();
    std::vector<std::string> vars(query_variables.begin(), query_variables.end());
    std::sort(vars.begin(), vars.end());

    std::vector<std::string> valid_orders;
    do {
        bool valid = true;
        std::unordered_set<std::string> seen{vars[0]};
        for (size_t i = 1; i < vars.size(); ++i) {
            const std::string& current = vars[i];
            bool connected = false;
            for (size_t rel_idx = 0; rel_idx < query.num_rels; ++rel_idx) {
                const auto& rel = query.rels[rel_idx];
                if ((rel.fromVariable == current && seen.count(rel.toVariable)) ||
                    (rel.toVariable == current && seen.count(rel.fromVariable))) {
                    connected = true;
                    break;
                }
            }
            if (!connected) {
                valid = false;
                break;
            }
            seen.insert(current);
        }
        if (valid) {
            std::string o = vars[0];
            for (size_t i = 1; i < vars.size(); i++) {
                o += "," + vars[i];
            }
            valid_orders.push_back(o);
        }
    } while (std::next_permutation(vars.begin(), vars.end()));
    return valid_orders;
}

}// namespace ffx
