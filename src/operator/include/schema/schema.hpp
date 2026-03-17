#ifndef VFENGINE_SCHEMA_HH
#define VFENGINE_SCHEMA_HH
#include "../../../query/include/predicate.hpp"
#include "../../table/include/adj_list.hpp"
#include "../../table/include/table.hpp"
#include "../../table/include/string_dictionary.hpp"
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ffx {

// Forward declarations
class QueryVariableToVectorMap;
class FactorizedTreeElement;
class AdjListManager;

struct ResolvedJoinAdjList {
    AdjList<uint64_t>* adj_list = nullptr;
    uint64_t num_adj_lists = 0;
    bool is_fwd = true;
    const Table* source_table = nullptr;
    bool from_schema_map = false;
};

struct Schema {
    uint32_t* invalidated_indices = nullptr;
    uint32_t* invalidated_count = nullptr;
    uint64_t* min_values = nullptr;
    size_t min_values_size = 0;

    // Core query execution objects
    QueryVariableToVectorMap* map = nullptr;
    std::vector<const Table*> tables;
    std::shared_ptr<FactorizedTreeElement> root = nullptr;
    const std::vector<std::string>* column_ordering = nullptr;
    StringDictionary* dictionary = nullptr;
    const std::unordered_set<std::string>* string_attributes = nullptr;
    StringPool* predicate_pool = nullptr;

    // Dedicated dictionary for LLM output strings. Owned by the LLM operator,
    // exposed here so downstream sinks can decode LLM column values.
    StringDictionary* llm_dictionary = nullptr;
    std::vector<std::string> required_min_attrs;
    const PredicateExpression* query_predicates = nullptr;

    // Map from "join_key|output_key" -> pre-loaded AdjList*
    // Key format: "a|b" for join from a to b
    std::unordered_map<std::string, AdjList<uint64_t>*> adj_list_map;

    // Map from "join_key|output_key" -> number of adjacency lists
    std::unordered_map<std::string, uint64_t> adj_list_size_map;

    // Generic execution context for passing shared memory objects downstream
    std::unordered_map<std::string, void*> execution_context;

    // LLM configuration JSON string (populated from LLM_MAP clause in the query).
    // When non-null, AIOperator reads this instead of the environment variable.
    const std::string* llm_config_str = nullptr;

    // Output attribute name from the rule head (required when using Map / LLM_MAP).
    const std::string* llm_output_attr = nullptr;

    // Optional export sink output path (e.g. provided via sink_csv:<path>).
    // When non-null, SinkExport writes to this path instead of stdout/env.
    const std::string* export_output_path = nullptr;

    // AdjListManager for loading (optional, ownership managed externally)
    AdjListManager* adj_list_manager = nullptr;

    // Helper to get adjacency list for a join
    AdjList<uint64_t>* get_adj_list(const std::string& join_key, const std::string& output_key) const {
        std::string key = join_key + "|" + output_key;
        auto it = adj_list_map.find(key);
        if (it == adj_list_map.end()) { throw std::runtime_error("Schema: No adj list found for: " + key); }
        return it->second;
    }

    // Helper to get number of adjacency lists for a join
    uint64_t get_adj_list_size(const std::string& join_key, const std::string& output_key) const {
        std::string key = join_key + "|" + output_key;
        auto it = adj_list_size_map.find(key);
        if (it == adj_list_size_map.end()) { throw std::runtime_error("Schema: No adj list size found for: " + key); }
        return it->second;
    }

    // Helper to register an adjacency list
    void register_adj_list(const std::string& join_key, const std::string& output_key, AdjList<uint64_t>* adj_list,
                           uint64_t num_adj_lists) {
        std::string key = join_key + "|" + output_key;
        adj_list_map[key] = adj_list;
        adj_list_size_map[key] = num_adj_lists;
    }

    // Check if adjacency list is registered
    bool has_adj_list(const std::string& join_key, const std::string& output_key) const {
        std::string key = join_key + "|" + output_key;
        return adj_list_map.find(key) != adj_list_map.end();
    }

    bool try_resolve_join_adj_list(const std::string& join_key, const std::string& output_key,
                                   ResolvedJoinAdjList& out) const {
        if (has_adj_list(join_key, output_key)) {
            out.adj_list = get_adj_list(join_key, output_key);
            out.num_adj_lists = get_adj_list_size(join_key, output_key);
            out.from_schema_map = true;

            for (const auto* table: tables) {
                int join_key_idx = -1;
                int output_key_idx = -1;
                for (size_t i = 0; i < table->columns.size(); ++i) {
                    if (table->columns[i] == join_key) join_key_idx = static_cast<int>(i);
                    if (table->columns[i] == output_key) output_key_idx = static_cast<int>(i);
                }
                if (join_key_idx != -1 && output_key_idx != -1) {
                    out.source_table = table;
                    out.is_fwd = (join_key_idx < output_key_idx);
                    return true;
                }
            }

            out.source_table = nullptr;
            out.is_fwd = true;
            return true;
        }

        for (const auto* table: tables) {
            int join_key_idx = -1;
            int output_key_idx = -1;
            for (size_t i = 0; i < table->columns.size(); ++i) {
                if (table->columns[i] == join_key) join_key_idx = static_cast<int>(i);
                if (table->columns[i] == output_key) output_key_idx = static_cast<int>(i);
            }

            if (join_key_idx != -1 && output_key_idx != -1) {
                out.source_table = table;
                out.is_fwd = (join_key_idx < output_key_idx);
                out.adj_list = out.is_fwd ? reinterpret_cast<AdjList<uint64_t>*>(table->fwd_adj_lists)
                                          : reinterpret_cast<AdjList<uint64_t>*>(table->bwd_adj_lists);
                out.num_adj_lists = out.is_fwd ? table->num_fwd_ids : table->num_bwd_ids;
                out.from_schema_map = false;
                return true;
            }
        }

        return false;
    }

    uint64_t resolve_scan_domain_size(const std::string& attribute) const {
        const std::string prefix = attribute + "|";
        for (const auto& [key, size]: adj_list_size_map) {
            if (key.rfind(prefix, 0) == 0) return size;
        }

        if (!tables.empty() && tables[0]) {
            const Table* root_table = tables[0];
            for (size_t i = 0; i < root_table->columns.size(); ++i) {
                if (root_table->columns[i] == attribute) {
                    if (i == 0) return root_table->num_fwd_ids;
                    if (i == 1) return root_table->num_bwd_ids;
                }
            }
        }

        for (const auto* table: tables) {
            for (size_t i = 0; i < table->columns.size(); ++i) {
                if (table->columns[i] == attribute) {
                    if (i == 0) return table->num_fwd_ids;
                    if (i == 1) return table->num_bwd_ids;
                }
            }
        }

        throw std::runtime_error("Schema: Unable to resolve scan domain size for attribute: " + attribute);
    }
};

}// namespace ffx

#endif
