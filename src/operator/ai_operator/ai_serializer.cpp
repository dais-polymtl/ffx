#include "ai_operator/ai_serializer.hpp"

#include "factorized_ftree/ftree_batch_iterator.hpp"
#include "schema/schema.hpp"

#include <algorithm>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ffx {

namespace {

static std::vector<std::string> get_projected_attributes(const FTreeBatchIterator& itr) {
    const auto info = itr.get_full_tree_info();
    std::vector<std::string> attrs(itr.num_attributes());
    for (size_t i = 0; i < itr.num_attributes(); ++i) {
        attrs[i] = info.nodes[static_cast<size_t>(info.projected_to_full[i])]->_attribute;
    }
    return attrs;
}

static std::vector<std::vector<uint64_t>> flatten_tuples(const FTreeBatchIterator& itr) {
    const size_t n = itr.num_attributes();
    std::vector<std::vector<uint64_t>> out;
    std::vector<uint64_t> row(n, 0);
    std::vector<std::pair<size_t, int32_t>> pending;
    pending.emplace_back(0u, -1);

    auto dfs = [&](auto& self, std::vector<std::pair<size_t, int32_t>>& nodes) -> void {
        if (nodes.empty()) {
            out.push_back(row);
            return;
        }

        auto [node_col, parent_idx] = nodes.back();
        nodes.pop_back();

        size_t start = 0;
        size_t end = itr.get_count(node_col);
        if (parent_idx >= 0) {
            const uint32_t* offset = itr.get_node_offset(node_col);
            start = offset[parent_idx];
            end = offset[parent_idx + 1];
        }

        const uint64_t* buf = itr.get_buffer(node_col);
        const auto children = itr.get_children(node_col);
        for (size_t i = start; i < end; ++i) {
            row[node_col] = buf[i];
            const size_t prev_size = nodes.size();
            for (size_t k = children.size(); k > 0; --k) {
                nodes.emplace_back(children[k - 1], static_cast<int32_t>(i));
            }
            self(self, nodes);
            nodes.resize(prev_size);
        }

        nodes.emplace_back(node_col, parent_idx);
    };

    dfs(dfs, pending);
    return out;
}

static nlohmann::json decode_value(const Schema* schema, const std::string& attr, uint64_t v) {
    if (schema && schema->string_attributes && schema->string_attributes->count(attr)) {
        return schema->dictionary->get_string(v).to_string();
    }
    return v;
}

// True if this node should appear in the factorized JSON / FTREE prompt (skip empty branches).
static bool nonempty_ftree_node(const nlohmann::json& node) {
    if (!node.is_object()) return false;
    if (node.contains("values") && node["values"].is_array() && !node["values"].empty()) return true;
    if (node.contains("children") && node["children"].is_object() && !node["children"].empty()) return true;
    return false;
}

static std::vector<int32_t> get_projected_parent(const FTreeBatchIterator& itr) {
    const auto info = itr.get_full_tree_info();
    const size_t n = itr.num_attributes();
    std::vector<int32_t> parent_proj(n, -1);
    for (size_t p = 0; p < n; ++p) {
        int32_t full_idx = info.projected_to_full[p];
        int32_t cur = info.parent_idx[static_cast<size_t>(full_idx)];
        while (cur >= 0 && info.full_to_projected[static_cast<size_t>(cur)] < 0) {
            cur = info.parent_idx[static_cast<size_t>(cur)];
        }
        parent_proj[p] = (cur >= 0) ? info.full_to_projected[static_cast<size_t>(cur)] : -1;
    }
    return parent_proj;
}

static nlohmann::json build_factorized_node(const FTreeBatchIterator& itr,
                                            const Schema& schema,
                                            const std::vector<std::string>& projected_attrs,
                                            const std::vector<std::vector<size_t>>& children_proj,
                                            size_t node_col,
                                            const std::vector<size_t>& parent_indices) {
    const uint64_t* buf = itr.get_buffer(node_col);

    // Rows under this node are always defined by parent positions in the parent column's buffer.
    // An empty parent list means zero rows in this subtree — never "full column" (that was a bug:
    // empty siblings like `c` under some roots could wrongly pull all c values, or emit empty `c:`).
    std::vector<size_t> reachable;
    const uint32_t* offsets = itr.get_node_offset(node_col);
    for (size_t p : parent_indices) {
        for (size_t i = offsets[p]; i < offsets[p + 1]; ++i) reachable.push_back(i);
    }

    nlohmann::json node = nlohmann::json::object();
    nlohmann::json values = nlohmann::json::array();
    std::unordered_set<uint64_t> seen_vals;
    seen_vals.reserve(reachable.size());
    for (size_t idx : reachable) {
        const uint64_t v = buf[idx];
        if (!seen_vals.insert(v).second) continue;
        values.push_back(decode_value(&schema, projected_attrs[node_col], v));
    }
    node["values"] = std::move(values);

    nlohmann::json children = nlohmann::json::object();
    for (size_t child_col : children_proj[node_col]) {
        nlohmann::json child_json =
                build_factorized_node(itr, schema, projected_attrs, children_proj, child_col, reachable);
        if (nonempty_ftree_node(child_json)) {
            children[projected_attrs[child_col]] = std::move(child_json);
        }
    }
    node["children"] = std::move(children);
    return node;
}

} // namespace

AISerializer::AISerializer(const std::string& tuple_format) : _tuple_format(tuple_format) {}

nlohmann::json AISerializer::build_flat_columns(const FTreeBatchIterator& itr, const Schema* schema,
                                                const std::vector<std::string>& attrs, size_t num_tuples) const {
    const std::vector<std::string> projected = get_projected_attributes(itr);
    const std::vector<std::string>& selected = attrs.empty() ? projected : attrs;
    const auto tuples = flatten_tuples(itr);
    const size_t limit = std::min(num_tuples, tuples.size());

    std::unordered_map<std::string, size_t> attr_to_col;
    for (size_t i = 0; i < projected.size(); ++i) {
        attr_to_col[projected[i]] = i;
    }

    nlohmann::json columns = nlohmann::json::array();
    for (const auto& attr : selected) {
        const size_t col_idx = attr_to_col.at(attr);
        nlohmann::json data = nlohmann::json::array();
        for (size_t i = 0; i < limit; ++i) {
            data.push_back(decode_value(schema, attr, tuples[i][col_idx]));
        }
        columns.push_back({{"name", attr}, {"data", std::move(data)}});
    }
    return columns;
}

nlohmann::json AISerializer::build_tree_columns(const FTreeBatchIterator& itr, const Schema& schema,
                                                const std::vector<std::string>& attrs, size_t num_tuples) const {
    (void)attrs;
    const auto projected = get_projected_attributes(itr);
    const size_t n = projected.size();
    const auto parent_proj = get_projected_parent(itr);
    std::vector<std::vector<size_t>> children_proj(n);
    for (size_t i = 0; i < n; ++i) {
        if (parent_proj[i] >= 0) children_proj[static_cast<size_t>(parent_proj[i])].push_back(i);
    }

    size_t root_col = 0;
    for (size_t i = 0; i < n; ++i) {
        if (parent_proj[i] == -1) { root_col = i; break; }
    }

    nlohmann::json tree = nlohmann::json::object();
    const uint64_t* root_buf = itr.get_buffer(root_col);
    const size_t root_count = itr.get_count(root_col);
    std::unordered_map<uint64_t, std::vector<size_t>> root_to_positions;
    for (size_t i = 0; i < root_count; ++i) root_to_positions[root_buf[i]].push_back(i);
    for (const auto& [root_val, pos] : root_to_positions) {
        nlohmann::json root_children = nlohmann::json::object();
        for (size_t child_col : children_proj[root_col]) {
            nlohmann::json child_json =
                    build_factorized_node(itr, schema, projected, children_proj, child_col, pos);
            if (nonempty_ftree_node(child_json)) {
                root_children[projected[child_col]] = std::move(child_json);
            }
        }
        const nlohmann::json root_key = decode_value(&schema, projected[root_col], root_val);
        tree[root_key.is_string() ? root_key.get<std::string>() : root_key.dump()] = std::move(root_children);
    }
    nlohmann::json tree_json = {{"root", projected[root_col]}, {"tree", std::move(tree)}};

    nlohmann::json data = nlohmann::json::array();
    for (size_t i = 0; i < num_tuples; ++i) {
        data.push_back(tree_json);
    }

    return nlohmann::json::array({{{"name", "ftree"}, {"data", std::move(data)}}});
}

nlohmann::json AISerializer::build_columns(const FTreeBatchIterator& itr, const Schema* schema,
                                           const std::vector<std::string>& attrs, size_t num_tuples) const {
    if (_tuple_format == "FTREE") {
        return build_tree_columns(itr, *schema, attrs, num_tuples);
    }
    return build_flat_columns(itr, schema, attrs, num_tuples);
}

} // namespace ffx
