#include "include/benchmark.hpp"

#include "../operator/include/scan/scan_synchronized.hpp"
#include "../operator/include/scan/scan_unpacked_synchronized.hpp"
#include "../operator/include/vector/state.hpp"
#include "../plan/include/plan.hpp"
#include "../plan/include/plan_tree.hpp"
#include "../query/include/query.hpp"
#include "../ser_der/include/storage_api.hpp"
#include "../ser_der/include/table_metadata.hpp"
#include "../table/include/adj_list_builder.hpp"
#include "sink/sink_min.hpp"
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

#include <cassert>

namespace ffx {


static std::unordered_map<std::string, std::vector<std::pair<std::string, std::string>>>
get_dataset_to_query_and_output(const std::string& query_test_suite_filename);

static std::unordered_map<std::string, sink_type> sink_map = {{"pnoop", SINK_PACKED_NOOP},
                                                              {"upnoop", SINK_UNPACKED_NOOP},
                                                              {"packed", SINK_PACKED},
                                                              {"unpacked", SINK_UNPACKED},
                                                              {"min", SINK_MIN},
                                                              {"sink_csv", SINK_EXPORT_CSV},
                                                              {"sink_json", SINK_EXPORT_JSON},
                                                              {"sink_markdown", SINK_EXPORT_MARKDOWN}};

static std::pair<std::string, std::optional<std::string>> parse_sink_spec(const std::string& sink_spec) {
    const auto pos = sink_spec.find(':');
    if (pos == std::string::npos) {
        return {sink_spec, std::nullopt};
    }
    std::string base = sink_spec.substr(0, pos);
    std::string path = sink_spec.substr(pos + 1);
    if (path.empty()) {
        return {base, std::nullopt};
    }
    return {base, path};
}

static bool valid_sink_type_str(const std::string& sink_spec) {
    const auto parsed = parse_sink_spec(sink_spec);
    return sink_map.find(parsed.first) != sink_map.end();
}

static std::vector<std::string> split(const std::string& str, char delimiter = ',');

// Helper to trim check if string is a file path and read it
static std::vector<std::string> read_config(const std::string& config_path_or_content) {
    std::vector<std::string> lines;
    std::ifstream file(config_path_or_content);
    if (file) {
        std::cout << "Reading config from file: " << config_path_or_content << std::endl;
        std::string line;
        while (std::getline(file, line)) {
            std::cout << "line: " << line << std::endl;
            if (!line.empty()) lines.push_back(line);
        }

    } else {
        throw std::runtime_error("Could not open config file: " + config_path_or_content);
    }
    return lines;
}

// Check if directory contains column-based serialization format
static bool is_column_format(const std::string& path) {
    return std::filesystem::exists(path + "/table_metadata.bin") ||
           std::filesystem::exists(path + "/table_metadata.json");
}

// Global attribute info: tracks the maximum ID value for each logical attribute across all tables
// This ensures adjacency lists are sized to accommodate ALL possible values from ANY table
struct GlobalAttributeInfo {
    std::unordered_map<std::string, uint64_t> max_values;// logical_attr -> global_max_value

    void update(const std::string& attr, uint64_t max_val) {
        auto it = max_values.find(attr);
        if (it == max_values.end()) {
            max_values[attr] = max_val;
        } else {
            it->second = std::max(it->second, max_val);
        }
    }

    uint64_t get_max(const std::string& attr, uint64_t default_val) const {
        auto it = max_values.find(attr);
        return (it != max_values.end()) ? it->second : default_val;
    }
};

// Parse config line to extract path and column info (without loading data)
static void parse_config_line_info(const std::string& line, std::string& path, std::vector<std::string>& all_columns,
                                   std::vector<std::string>& logical_columns, std::vector<size_t>& column_positions) {
    auto comma_pos = line.find(',');
    if (comma_pos == std::string::npos) throw std::runtime_error("Invalid config line format: " + line);

    path = line.substr(0, comma_pos);
    std::string remaining = line.substr(comma_pos + 1);

    auto paren_start = remaining.find('(');
    auto paren_end = remaining.find(')');
    if (paren_start == std::string::npos || paren_end == std::string::npos)
        throw std::runtime_error("Invalid table definition in config: " + remaining);

    std::string cols_str = remaining.substr(paren_start + 1, paren_end - paren_start - 1);

    std::stringstream ss(cols_str);
    std::string col;
    while (std::getline(ss, col, ',')) {
        all_columns.push_back(col);
    }

    for (size_t i = 0; i < all_columns.size(); i++) {
        if (all_columns[i] != "_") {
            logical_columns.push_back(all_columns[i]);
            column_positions.push_back(i);
        }
    }
}

// First pass: scan all tables to determine global max values for each attribute
static GlobalAttributeInfo scan_global_attribute_info(const std::vector<std::string>& config_lines) {
    GlobalAttributeInfo global_info;

    std::cout << "=== First pass: scanning for global attribute max values ===" << std::endl;

    for (const auto& line: config_lines) {
        std::string path;
        std::vector<std::string> all_columns, logical_columns;
        std::vector<size_t> column_positions;

        parse_config_line_info(line, path, all_columns, logical_columns, column_positions);

        if (logical_columns.size() < 2 || !is_column_format(path)) continue;

        // Load metadata only (not actual data)
        SerializedTableMetadata metadata = read_metadata_binary(path);

        // For each logical column, track its global max value
        for (size_t i = 0; i < logical_columns.size() && i < column_positions.size(); i++) {
            size_t pos = column_positions[i];
            if (pos >= metadata.columns.size()) continue;

            const std::string& logical_attr = logical_columns[i];
            const SerializedColumnInfo& col_info = metadata.columns[pos];

            if (col_info.type == ColumnType::UINT64) { global_info.update(logical_attr, col_info.max_value); }
        }
    }

    std::cout << "=== Global attribute max values ===" << std::endl;
    for (const auto& [attr, max_val]: global_info.max_values) {
        std::cout << "  " << attr << ": " << max_val << std::endl;
    }

    return global_info;
}

// Load table from column-based serialization format
// Supports underscore wildcards for column position skipping:
//   RELATION(var1, var2) -> columns 0, 1
//   RELATION(var1, _, var2) -> columns 0, 2 (skip column 1)
//   RELATION(var1, _, _, var2) -> columns 0, 3 (skip columns 1 and 2)
static std::unique_ptr<Table>
load_table_from_columns_format(const std::string& path, const std::string& name,
                               const std::vector<std::string>& all_columns,// All variables including underscores
                               StringDictionary* dict, StringPool* pool,
                               const GlobalAttributeInfo* global_info = nullptr) {

    std::cout << "  Loading from column-based format..." << std::endl;

    // Step 1: Load metadata
    SerializedTableMetadata metadata = read_metadata_binary(path);
    std::cout << "  Found " << metadata.columns.size() << " columns, " << metadata.num_rows << " rows" << std::endl;

    // Step 2: Find non-underscore variables and their positions
    // Underscore "_" means skip this column position
    std::vector<std::string> logical_columns;// Non-underscore variable names
    std::vector<size_t> column_positions;    // Their positions in the relation

    for (size_t i = 0; i < all_columns.size(); i++) {
        if (all_columns[i] != "_") {
            logical_columns.push_back(all_columns[i]);
            column_positions.push_back(i);
        }
    }

    if (logical_columns.size() < 2) {
        throw std::runtime_error("Table must have at least 2 non-underscore columns: " + path);
    }

    // Step 3: Map positions to physical columns from metadata
    size_t src_pos = column_positions[0];
    size_t dest_pos = column_positions[1];

    if (src_pos >= metadata.columns.size() || dest_pos >= metadata.columns.size()) {
        throw std::runtime_error("Column position out of range for table: " + path + " (positions " +
                                 std::to_string(src_pos) + "," + std::to_string(dest_pos) + " but table has " +
                                 std::to_string(metadata.columns.size()) + " columns)");
    }

    std::string src_attr = metadata.columns[src_pos].attr_name;
    std::string dest_attr = metadata.columns[dest_pos].attr_name;

    std::cout << "  Mapping: logical[" << logical_columns[0] << "," << logical_columns[1] << "] at positions["
              << src_pos << "," << dest_pos << "] -> physical[" << src_attr << "," << dest_attr << "]" << std::endl;

    // Step 3: Load the ID columns for join attributes
    const uint64_t* src_id_col = nullptr;
    const uint64_t* dest_id_col = nullptr;
    uint64_t num_fwd_ids = 0;
    uint64_t num_bwd_ids = 0;

    std::unique_ptr<uint64_t[]> src_col_data;
    std::unique_ptr<uint64_t[]> dest_col_data;
    std::unique_ptr<ffx_str_t[]> src_str_data;
    std::unique_ptr<ffx_str_t[]> dest_str_data;
    std::unique_ptr<uint64_t[]> src_str_ids;
    std::unique_ptr<uint64_t[]> dest_str_ids;

    // Load source column
    const SerializedColumnInfo* src_col_info = metadata.find_column(src_attr);
    if (!src_col_info) { throw std::runtime_error("Source attribute not found in metadata: " + src_attr); }

    if (src_col_info->type == ColumnType::UINT64) {
        uint64_t num_rows;
        src_col_data = deserialize_uint64_column(path + "/" + src_attr + "_uint64.bin", num_rows);
        src_id_col = src_col_data.get();
        // Use global max if available, otherwise use local max
        uint64_t effective_max = src_col_info->max_value;
        if (global_info) { effective_max = global_info->get_max(logical_columns[0], src_col_info->max_value); }
        num_fwd_ids = effective_max + 1;
        std::cout << "  Loaded uint64 source column: " << src_attr << " (local_max=" << src_col_info->max_value
                  << ", effective_max=" << effective_max << ")" << std::endl;
    } else if (src_col_info->type == ColumnType::STRING) {
        uint64_t num_rows;
        src_str_data = deserialize_string_column(path + "/" + src_attr + "_string.bin", num_rows, pool);
        // Build ID column using dictionary
        src_str_ids = std::make_unique<uint64_t[]>(num_rows);
        for (uint64_t i = 0; i < num_rows; i++) {
            if (!src_str_data[i].is_null()) {
                src_str_ids[i] = dict->add_string(src_str_data[i]);
            } else {
                src_str_ids[i] = UINT64_MAX;
            }
        }
        src_id_col = src_str_ids.get();
        num_fwd_ids = dict->size() + 1;// +1 to accommodate new strings
        std::cout << "  Loaded string source column: " << src_attr << std::endl;
    }

    // Load destination column
    const SerializedColumnInfo* dest_col_info = metadata.find_column(dest_attr);
    if (!dest_col_info) { throw std::runtime_error("Destination attribute not found in metadata: " + dest_attr); }

    if (dest_col_info->type == ColumnType::UINT64) {
        uint64_t num_rows;
        dest_col_data = deserialize_uint64_column(path + "/" + dest_attr + "_uint64.bin", num_rows);
        dest_id_col = dest_col_data.get();
        // Use global max if available, otherwise use local max
        uint64_t effective_max = dest_col_info->max_value;
        if (global_info) { effective_max = global_info->get_max(logical_columns[1], dest_col_info->max_value); }
        num_bwd_ids = effective_max + 1;
        std::cout << "  Loaded uint64 dest column: " << dest_attr << " (local_max=" << dest_col_info->max_value
                  << ", effective_max=" << effective_max << ")" << std::endl;
    } else if (dest_col_info->type == ColumnType::STRING) {
        uint64_t num_rows;
        dest_str_data = deserialize_string_column(path + "/" + dest_attr + "_string.bin", num_rows, pool);
        // Build ID column using dictionary
        dest_str_ids = std::make_unique<uint64_t[]>(num_rows);
        for (uint64_t i = 0; i < num_rows; i++) {
            if (!dest_str_data[i].is_null()) {
                dest_str_ids[i] = dict->add_string(dest_str_data[i]);
            } else {
                dest_str_ids[i] = UINT64_MAX;
            }
        }
        dest_id_col = dest_str_ids.get();
        num_bwd_ids = dict->size() + 1;
        std::cout << "  Loaded string dest column: " << dest_attr << std::endl;
    }

    // Update string counts based on current dictionary size
    // Note: We don't finalize here - finalization happens once after all tables are loaded
    if (src_col_info->type == ColumnType::STRING || dest_col_info->type == ColumnType::STRING) {
        // Update counts based on current dictionary size
        if (src_col_info->type == ColumnType::STRING) { num_fwd_ids = dict->size(); }
        if (dest_col_info->type == ColumnType::STRING) { num_bwd_ids = dict->size(); }
        std::cout << "  Dictionary has " << dict->size() << " strings (will finalize after all tables loaded)"
                  << std::endl;
    }

    // Step 4: Build adjacency lists
    std::unique_ptr<AdjList<uint64_t>[]> fwd_adj, bwd_adj;
    build_adj_lists_from_columns(src_id_col, dest_id_col, metadata.num_rows, num_fwd_ids, num_bwd_ids, fwd_adj,
                                 bwd_adj);

    std::cout << "  Built adjacency lists: fwd(" << num_fwd_ids << "), bwd(" << num_bwd_ids << ")" << std::endl;

    // Step 5: Create Table
    auto table = std::make_unique<Table>(num_fwd_ids, num_bwd_ids, std::move(fwd_adj), std::move(bwd_adj));
    table->name = name;
    // Use only the non-underscore logical column names for query matching
    for (const auto& c: logical_columns) {
        table->columns.push_back(c);
    }

    // Track string columns - use logical column names for query matching
    if (src_col_info->type == ColumnType::STRING) { table->string_columns.insert(logical_columns[0]); }
    if (dest_col_info->type == ColumnType::STRING) { table->string_columns.insert(logical_columns[1]); }

    return table;
}

static std::unique_ptr<Table> load_table_from_config_line(const std::string& line, StringDictionary* dict = nullptr,
                                                          StringPool* pool = nullptr,
                                                          const GlobalAttributeInfo* global_info = nullptr) {
    // Format: /path/to/table,Name(var1,var2),[cardinality]
    // Use underscore _ to skip columns: Name(var1,_,var2) uses columns 0 and 2
    // Cardinality is optional, defaults to m:n
    // Examples:
    //   /path/to/knows,Knows(person1,person2)              -> columns 0,1
    //   /path/to/movie_info_idx,IS_INFO_TYPE(mi_idx,_,it)  -> columns 0,2
    //   /path/to/livesIn,LivesIn(person,city),[n:1]        -> columns 0,1 with n:1 cardinality
    std::cout << "Loading table from config line: " << line << std::endl;
    auto comma_pos = line.find(',');
    if (comma_pos == std::string::npos) throw std::runtime_error("Invalid config line format: " + line);

    std::string path = line.substr(0, comma_pos);
    std::string remaining = line.substr(comma_pos + 1);

    auto paren_start = remaining.find('(');
    auto paren_end = remaining.find(')');
    if (paren_start == std::string::npos || paren_end == std::string::npos)
        throw std::runtime_error("Invalid table definition in config: " + remaining);

    std::string name = remaining.substr(0, paren_start);
    std::string cols_str = remaining.substr(paren_start + 1, paren_end - paren_start - 1);

    // Split column variables (including underscores)
    std::vector<std::string> all_columns;

    std::stringstream ss(cols_str);
    std::string col;
    while (std::getline(ss, col, ',')) {
        all_columns.push_back(col);
    }

    // Count non-underscore columns
    size_t non_underscore_count = 0;
    for (const auto& c: all_columns) {
        if (c != "_") non_underscore_count++;
    }
    if (non_underscore_count < 2) {
        throw std::runtime_error("Table must have at least 2 non-underscore columns: " + line);
    }

    // Parse optional cardinality suffix: ,[1:1] or ,[n:1] or ,[1:n] or ,[m:n]
    Cardinality cardinality = Cardinality::MANY_TO_MANY;
    std::string after_paren = remaining.substr(paren_end + 1);
    if (!after_paren.empty()) {
        auto bracket_start = after_paren.find('[');
        auto bracket_end = after_paren.find(']');
        if (bracket_start != std::string::npos && bracket_end != std::string::npos) {
            std::string content = after_paren.substr(bracket_start, bracket_end - bracket_start + 1);
            if (content.find(':') != std::string::npos) {
                cardinality = parse_cardinality(content);
                std::cout << "  Parsed cardinality: " << cardinality_to_string(cardinality) << std::endl;
            }
        }
    }

    if (!is_column_format(path)) {
        throw std::runtime_error(
                "Unsupported table format at path: " + path +
                ". Only column format is supported (expected table_metadata.bin/json).");
    }

    // Column-based format (table_metadata.bin + *_uint64.bin/*_string.bin)
    std::unique_ptr<Table> table = load_table_from_columns_format(path, name, all_columns, dict, pool, global_info);

    table->cardinality = cardinality;

    assert(table != nullptr);
    assert(table->columns.size() > 0);
    return table;
}

struct RelationBinding {
    std::string table_dir;
    std::string relation_name;
    std::string from_variable;
    std::string to_variable;
    size_t source_column_idx;
    size_t target_column_idx;
    ProjectionCardinality projection_cardinality;
};

static Cardinality to_table_cardinality(ProjectionCardinality card) {
    switch (card) {
        case ProjectionCardinality::ONE_TO_ONE:
            return Cardinality::ONE_TO_ONE;
        case ProjectionCardinality::ONE_TO_MANY:
            return Cardinality::ONE_TO_MANY;
        case ProjectionCardinality::MANY_TO_ONE:
            return Cardinality::MANY_TO_ONE;
        case ProjectionCardinality::MANY_TO_MANY:
        default:
            return Cardinality::MANY_TO_MANY;
    }
}

static std::vector<std::string> list_serialized_table_dirs(const std::string& serialized_root_dir) {
    std::vector<std::string> table_dirs;
    const std::filesystem::path root(serialized_root_dir);
    if (!std::filesystem::exists(root) || !std::filesystem::is_directory(root)) {
        throw std::runtime_error("Serialized root directory does not exist or is not a directory: " +
                                 serialized_root_dir);
    }

    for (const auto& entry: std::filesystem::directory_iterator(root)) {
        if (!entry.is_directory()) continue;
        const auto candidate = entry.path();
        if (std::filesystem::exists(candidate / "table_metadata.bin")) {
            table_dirs.push_back(candidate.string());
        }
    }

    if (table_dirs.empty()) {
        throw std::runtime_error("No serialized table directories found under: " + serialized_root_dir);
    }
    return table_dirs;
}

static std::vector<RelationBinding> resolve_relation_bindings(const Query& query,
                                                              const std::string& serialized_root_dir) {
    const auto table_dirs = list_serialized_table_dirs(serialized_root_dir);

    struct TableCatalogEntry {
        std::string table_dir;
        SerializedTableMetadata metadata;
    };

    std::vector<TableCatalogEntry> catalog;
    catalog.reserve(table_dirs.size());
    for (const auto& table_dir: table_dirs) {
        catalog.push_back({table_dir, read_metadata_binary(table_dir)});
    }

    std::vector<RelationBinding> bindings;
    bindings.reserve(query.num_rels);

    for (uint64_t rel_idx = 0; rel_idx < query.num_rels; ++rel_idx) {
        const auto& rel = query.rels[rel_idx];
        bool matched = false;

        if (rel.hasTableName()) {
            for (const auto& entry: catalog) {
                const auto* projection = entry.metadata.find_projection(rel.tableName);
                if (!projection) continue;
                bindings.push_back(RelationBinding{entry.table_dir,
                                                   projection->relation_name,
                                                   rel.fromVariable,
                                                   rel.toVariable,
                                                   projection->source_column_idx,
                                                   projection->target_column_idx,
                                                   projection->cardinality});
                matched = true;
                break;
            }
            if (!matched) {
                throw std::runtime_error("No serialized projection found for relation '" + rel.tableName + "'");
            }
            continue;
        }

        for (const auto& entry: catalog) {
            const auto* src_col = entry.metadata.find_column(rel.fromVariable);
            const auto* dst_col = entry.metadata.find_column(rel.toVariable);
            if (!src_col || !dst_col) continue;

            bindings.push_back(RelationBinding{entry.table_dir,
                                               rel.fromVariable + "->" + rel.toVariable,
                                               rel.fromVariable,
                                               rel.toVariable,
                                               src_col->column_idx,
                                               dst_col->column_idx,
                                               ProjectionCardinality::MANY_TO_MANY});
            matched = true;
            break;
        }

        if (!matched) {
            throw std::runtime_error("No serialized table/projection found for relation '" + rel.fromVariable +
                                     "->" + rel.toVariable + "'");
        }
    }

    return bindings;
}

static GlobalAttributeInfo scan_global_attribute_info(const std::vector<RelationBinding>& bindings,
                                                      StringDictionary* dict = nullptr) {
    GlobalAttributeInfo global_info;

    std::cout << "=== First pass: scanning serialized metadata for global attribute max values ===" << std::endl;
    for (const auto& binding: bindings) {
        const SerializedTableMetadata metadata = read_metadata_binary(binding.table_dir);

        if (binding.source_column_idx >= metadata.columns.size() ||
            binding.target_column_idx >= metadata.columns.size()) {
            throw std::runtime_error("Projection column index out of range in table: " + binding.table_dir);
        }

        const auto& src_col = metadata.columns[binding.source_column_idx];
        const auto& dst_col = metadata.columns[binding.target_column_idx];

        if (src_col.type == ColumnType::UINT64) {
            global_info.update(binding.from_variable, src_col.max_value);
        } else if (dict) {
            global_info.update(binding.from_variable, dict->size());
        }

        if (dst_col.type == ColumnType::UINT64) {
            global_info.update(binding.to_variable, dst_col.max_value);
        } else if (dict) {
            global_info.update(binding.to_variable, dict->size());
        }
    }

    std::cout << "=== Global attribute max values ===" << std::endl;
    for (const auto& [attr, max_val]: global_info.max_values) {
        std::cout << "  " << attr << ": " << max_val << std::endl;
    }

    return global_info;
}

static std::vector<std::unique_ptr<Table>> load_tables(const std::string& serialized_root_dir, const Query& query,
                                                       StringDictionary* dict = nullptr, StringPool* pool = nullptr) {
    const auto bindings = resolve_relation_bindings(query, serialized_root_dir);
    const auto global_info = scan_global_attribute_info(bindings, dict);

    std::vector<std::unique_ptr<Table>> loaded_tables;
    loaded_tables.reserve(bindings.size());

    for (const auto& binding: bindings) {
        const SerializedTableMetadata metadata = read_metadata_binary(binding.table_dir);
        if (binding.source_column_idx >= metadata.columns.size() ||
            binding.target_column_idx >= metadata.columns.size()) {
            throw std::runtime_error("Projection column index out of range in table: " + binding.table_dir);
        }

        const size_t max_idx = std::max(binding.source_column_idx, binding.target_column_idx);
        std::vector<std::string> all_columns(max_idx + 1, "_");
        all_columns[binding.source_column_idx] = binding.from_variable;
        all_columns[binding.target_column_idx] = binding.to_variable;

        auto table = load_table_from_columns_format(binding.table_dir, binding.relation_name, all_columns, dict, pool,
                                                    &global_info);
        table->cardinality = to_table_cardinality(binding.projection_cardinality);
        loaded_tables.push_back(std::move(table));
    }

    if (dict && !dict->is_finalized()) {
        dict->finalize();
        std::cout << "Dictionary finalized with " << dict->size() << " total strings" << std::endl;
    }

    assert(!loaded_tables.empty());
    return loaded_tables;
}

static std::vector<const Table*> filter_tables(const std::vector<std::unique_ptr<Table>>& loaded_tables,
                                               const Query& query) {
    std::vector<const Table*> tables;
    for (const auto& tbl: loaded_tables) {
        bool include = true;
        // Filter out tables that do not correspond to a query relation
        // We assume tables represent edges if they have 2 columns.
        if (tbl->columns.size() == 2) {
            if (query.get_query_relation(tbl->columns[0], tbl->columns[1]) == nullptr) { include = false; }
        }
        if (include) { tables.push_back(tbl.get()); }
    }
    return tables;
}

static void populate_schema_adj_lists(Schema& schema, const std::vector<const Table*>& tables, const Query& query) {
    for (const auto* table: tables) {
        if (table->columns.size() != 2) continue;

        const std::string& col0 = table->columns[0];// source column
        const std::string& col1 = table->columns[1];// dest column

        // Check if this table corresponds to a query relation
        auto* rel = query.get_query_relation(col0, col1);
        if (!rel) continue;

        // Determine direction based on query relation
        bool is_fwd = rel->isFwd(col0, col1);

        // Register forward adj_list: col0 -> col1
        if (table->fwd_adj_lists) {
            schema.register_adj_list(col0, col1, table->fwd_adj_lists, table->num_fwd_ids);
            std::cout << "Schema: Registered adj_list " << col0 << "->" << col1 << " (fwd, " << table->num_fwd_ids
                      << " entries) from table " << table->name << std::endl;
        }

        // Register backward adj_list: col1 -> col0
        if (table->bwd_adj_lists) {
            schema.register_adj_list(col1, col0, table->bwd_adj_lists, table->num_bwd_ids);
            std::cout << "Schema: Registered adj_list " << col1 << "->" << col0 << " (bwd, " << table->num_bwd_ids
                      << " entries) from table " << table->name << std::endl;
        }
    }
}

static bool evaluate_query_min_sink(const std::string& serialized_root_dir, const Query& query,
                                    const std::string& query_as_str, const std::string& column_ordering,
                                    const std::string& sink_type_str,
                                    const std::vector<std::string>& expected_min_values) {
    const auto column_ordering_vec = split(column_ordering);
    assert(sink_type_str == "min");
    constexpr auto sink = SINK_MIN;

    const std::vector<std::string>& required_min_attrs = query.head_attributes();

    // Load tables first so we can pass them to plan creation
    auto pool = std::make_unique<StringPool>();
    auto dict = std::make_unique<StringDictionary>(pool.get());

    auto loaded_tables = load_tables(serialized_root_dir, query, dict.get(), pool.get());
    auto tables = filter_tables(loaded_tables, query);
    assert(tables.size() > 0 && "No tables loaded for the query!");

    const auto [plan, ftree] = map_ordering_to_plan(query, column_ordering_vec, sink, tables);

    const auto min_values_size = required_min_attrs.size();
    if (!expected_min_values.empty() && expected_min_values.size() != min_values_size) {
        throw std::invalid_argument("expected_min_values size must match required_min_attrs size");
    }

    Schema schema;
    schema.dictionary = dict.get();
    schema.query_predicates = &query.get_predicates();
    schema.required_min_attrs = required_min_attrs;
    schema.min_values_size = min_values_size;
    auto min_values_owner = std::make_unique<uint64_t[]>(schema.min_values_size);
    schema.min_values = min_values_owner.get();
    assert(schema.min_values_size);
    std::fill_n(schema.min_values, schema.min_values_size, std::numeric_limits<uint64_t>::max());

    // Collect string attributes from all loaded tables for min operator string comparison
    std::unordered_set<std::string> string_attrs;
    for (const auto& table: tables) {
        for (const auto& col: table->string_columns) {
            string_attrs.insert(col);
        }
    }
    schema.string_attributes = string_attrs.empty() ? nullptr : &string_attrs;

    // Populate Schema's adj_list_map for Schema-based lookup in operators
    populate_schema_adj_lists(schema, tables, query);

    plan->init(tables, ftree, &schema);

    const auto duration = plan->execute();
    auto first_op = plan->get_first_op();
    const auto operator_names = plan->get_operator_names(sink);

    std::cout << std::endl;
    std::cout << "-------------- Report ----------------" << std::endl;
    std::cout << "Query: " << query_as_str << std::endl;
    std::cout << "Column Ordering: [" << column_ordering << "]" << std::endl;
    std::cout << "------------- Operators --------------" << std::endl;

    int idx = 0;
    while (first_op && idx < static_cast<int>(operator_names.size())) {
        if (operator_names[idx] == "CASCADE") {
            std::cout << operator_names[idx] << " : " << -1 << std::endl;
        } else {
            std::cout << operator_names[idx] << " : " << first_op->get_num_exec_call() << std::endl;
        }
        idx++;
        first_op = first_op->next_op;
    }

    std::cout << "-------------- Results ---------------" << std::endl;
    std::cout << "Execution Duration: " << duration.count() << " msecs" << std::endl;

    const auto& actual_min_values = schema.min_values;
    const auto& actual_min_values_size = schema.min_values_size;

    // Print min values, converting string IDs to actual strings
    std::cout << "Actual Min Values: ";
    for (size_t i = 0; i < actual_min_values_size; ++i) {
        const std::string& attr_name = required_min_attrs[i];
        bool is_string_attr = string_attrs.count(attr_name) > 0;

        if (is_string_attr && actual_min_values[i] != std::numeric_limits<uint64_t>::max()) {
            // Convert string ID to actual string value
            const ffx_str_t& str_value = dict->get_string(actual_min_values[i]);
            std::cout << "'" << str_value.to_string() << "' ";
        } else {
            std::cout << actual_min_values[i] << " ";
        }
    }
    std::cout << std::endl;

    // Also print attribute names for clarity
    std::cout << "Attribute Names:   ";
    for (size_t i = 0; i < actual_min_values_size; ++i) {
        const std::string& attr_name = required_min_attrs[i];
        bool is_string_attr = string_attrs.count(attr_name) > 0;
        std::cout << attr_name << (is_string_attr ? "(str) " : " ");
    }
    std::cout << std::endl;

    if (!expected_min_values.empty()) {
        std::cout << "Expected Min Values: ";
        for (const auto& val: expected_min_values) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    bool result = true;
    if (!expected_min_values.empty()) {
        for (size_t i = 0; i < actual_min_values_size; ++i) {
            if (actual_min_values[i] != std::stoull(expected_min_values[i])) {
                result = false;
                break;
            }
        }
    }
    std::cout << "--------------------------------------" << std::endl;
    return result;
}

bool evaluate_query(const std::string& serialized_root_dir, const std::string& query_as_str,
                    const std::string& column_ordering, const std::string& sink_type_str,
                    const std::vector<std::string>& expected_values) {

    if (!valid_sink_type_str(sink_type_str)) { throw std::invalid_argument("Invalid sink type: " + sink_type_str); }

    const auto [sink_base, sink_path] = parse_sink_spec(sink_type_str);
    const auto sink = sink_map[sink_base];

    Query query{query_as_str};
    if (query.requires_min_sink() && sink != SINK_MIN) {
        throw std::invalid_argument("Q(MIN(...)) requires sink type 'min'");
    }

    // Check if sink type is SINK_MIN
    if (sink == SINK_MIN) {
        if (!query.requires_min_sink() || query.head_attributes().empty()) {
            throw std::invalid_argument(
                    "sink type 'min' requires rule syntax Q(MIN(attr1,attr2,...)) := <body> with at least one "
                    "attribute in MIN(...)");
        }
        return evaluate_query_min_sink(serialized_root_dir, query, query_as_str, column_ordering, sink_type_str,
                                       expected_values);
    }

    // Non-SINK_MIN processing
    const auto column_ordering_vec = split(column_ordering);

    const bool is_export = (sink == SINK_EXPORT_CSV || sink == SINK_EXPORT_JSON || sink == SINK_EXPORT_MARKDOWN);
    if (sink_path.has_value() && !is_export) {
        throw std::invalid_argument("Sink output path is only supported for export sinks: " + sink_type_str);
    }
    auto effective_sink = sink;

    // Load tables first so we can pass them to plan creation
    auto pool = std::make_unique<StringPool>();
    auto dict = std::make_unique<StringDictionary>(pool.get());

    auto loaded_tables = load_tables(serialized_root_dir, query, dict.get(), pool.get());
    auto tables = filter_tables(loaded_tables, query);
    assert(tables.size() > 0 && "No tables loaded for the query!");

    std::pair<std::unique_ptr<Plan>, std::shared_ptr<FactorizedTreeElement>> plan_and_tree =
            map_ordering_to_plan(query, column_ordering_vec, effective_sink, tables);

    auto& plan = plan_and_tree.first;
    auto& ftree = plan_and_tree.second;

    // Print operator pipeline before init
    plan->print_pipeline();
    assert(tables.size() > 0 && "No tables loaded for the query!");

    // Wire LLM config from query to schema so AIOperator can read it.
    std::string llm_config_str = query.get_llm_map_config();
    std::string llm_output_attr = query.get_llm_map_output_attr();
    std::string export_output_path;

    Schema schema;
    schema.dictionary = dict.get();
    schema.query_predicates = &query.get_predicates();
    if (query.has_llm_map()) {
        schema.llm_config_str = &llm_config_str;
        schema.llm_output_attr = &llm_output_attr;
    }
    if (sink_path.has_value()) {
        export_output_path = *sink_path;
        schema.export_output_path = &export_output_path;
    }

    // Collect string attributes from all loaded tables for predicate evaluation
    std::unordered_set<std::string> string_attrs;
    for (const auto& table: tables) {
        for (const auto& col: table->string_columns) {
            string_attrs.insert(col);
        }
    }
    schema.string_attributes = string_attrs.empty() ? nullptr : &string_attrs;

    // Populate Schema's adj_list_map for Schema-based lookup in operators
    populate_schema_adj_lists(schema, tables, query);

    plan->init(tables, ftree, &schema);
    const auto duration = plan->execute();
    auto* first_op = plan->get_first_op();
    const auto operator_names = plan->get_operator_names(effective_sink);
    const auto actual_num_output_tuples = plan->get_num_output_tuples();

    // Snapshot exec counts before destroying the plan (destroying flushes SinkExport output).
    std::vector<std::pair<std::string, int64_t>> op_exec_counts;
    op_exec_counts.reserve(operator_names.size());
    {
        int idx = 0;
        auto* cur = first_op;
        while (cur && idx < static_cast<int>(operator_names.size())) {
            if (operator_names[idx] == "CASCADE") {
                op_exec_counts.emplace_back(operator_names[idx], -1);
                idx++;
                continue;
            }
            op_exec_counts.emplace_back(operator_names[idx], static_cast<int64_t>(cur->get_num_exec_call()));
            cur = cur->next_op;
            idx++;
        }
    }

    // Destroy operators before printing report so SinkExport's finalized output
    // appears before the report section (especially for MARKDOWN buffering).
    plan.reset();

    std::cout << std::endl;
    std::cout << "-------------- Report ----------------" << std::endl;
    std::cout << "Query: " << query_as_str << std::endl;
    std::cout << "Column Ordering: [" << column_ordering << "]" << std::endl;
    std::cout << "------------- Operators --------------" << std::endl;

    for (const auto& [name, cnt] : op_exec_counts) {
        std::cout << name << " : " << cnt << std::endl;
    }

    std::cout << "-------------- Results ---------------" << std::endl;
    std::cout << "Actual Output: " << actual_num_output_tuples << std::endl;
    if (!expected_values.empty()) { std::cout << "Expected Output: " << expected_values.front() << std::endl; }
    std::cout << "Execution Duration: " << duration.count() << " msecs" << std::endl;
    std::cout << "--------------------------------------" << std::endl;

    bool result = true;
    if (!is_export && !query.has_llm_map() && !expected_values.empty()) {
        result = (actual_num_output_tuples == std::stoull(expected_values.front()));
    }
    return result;
}


static std::string trim_whitespace(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\n\r");
    return s.substr(start, end - start + 1);
}

static std::vector<std::string> split(const std::string& str, char delimiter) {
    std::vector<std::string> result;
    size_t start = 0, end;

    while ((end = str.find(delimiter, start)) != std::string::npos) {
        result.push_back(trim_whitespace(str.substr(start, end - start)));
        start = end + 1;
    }
    result.push_back(trim_whitespace(str.substr(start)));

    return result;
}

// Padded counter to avoid false sharing
struct alignas(64) PaddedCounter {
    uint64_t count = 0;
};

static bool evaluate_query_multithreaded_min_sink(const std::string& serialized_root_dir,
                                                  const std::string& query_as_str,
                                                  const std::string& column_ordering, const std::string& sink_type_str,
                                                  const std::vector<std::string>& expected_min_values,
                                                  uint32_t num_threads) {
    Query query{query_as_str};
    const auto column_ordering_vec = split(column_ordering);
    if (!query.requires_min_sink() || query.head_attributes().empty()) {
        throw std::invalid_argument(
                "sink type 'min' requires rule syntax Q(MIN(attr1,attr2,...)) := <body> with at least one attribute "
                "in MIN(...)");
    }
    const std::vector<std::string>& eff_min_attrs = query.head_attributes();

    const auto [sink_base, sink_path] = parse_sink_spec(sink_type_str);
    if (sink_path.has_value()) {
        throw std::invalid_argument("Sink output path is not supported for min sink: " + sink_type_str);
    }
    const auto sink = sink_map[sink_base];

    auto pool = std::make_unique<StringPool>();
    auto dict = std::make_unique<StringDictionary>(pool.get());

    auto loaded_tables = load_tables(serialized_root_dir, query, dict.get(), pool.get());
    auto tables = filter_tables(loaded_tables, query);
    assert(tables.size() > 0 && "No tables loaded for the query!");

    uint64_t total_ids = 0;
    const std::string& root_attr = column_ordering_vec[0];

    // Root table is always at index 0
    if (tables.empty()) { throw std::runtime_error("No tables available"); }
    const Table* root_table = tables[0];

    // Find the index of the root attribute in the table's columns
    // First column (index 0) -> use num_fwd_ids
    // Second column (index 1) -> use num_bwd_ids
    size_t attr_index = SIZE_MAX;
    for (size_t i = 0; i < root_table->columns.size(); i++) {
        if (root_table->columns[i] == root_attr) {
            attr_index = i;
            break;
        }
    }

    if (attr_index == SIZE_MAX) {
        throw std::runtime_error("Root attribute " + root_attr + " not found in table " + root_table->name);
    }

    if (attr_index == 0) {
        // First column -> forward adjacency lists
        total_ids = root_table->num_fwd_ids;
    } else if (attr_index == 1) {
        // Second column -> backward adjacency lists
        total_ids = root_table->num_bwd_ids;
    } else {
        throw std::runtime_error("Root attribute " + root_attr + " is at unexpected column index " +
                                 std::to_string(attr_index));
    }

    constexpr auto chunk_size = State::MAX_VECTOR_SIZE;
    // const uint64_t total_ids = table->num_ids;
    const uint64_t total_chunks = (total_ids + chunk_size - 1) / chunk_size;

    const auto num_attributes = eff_min_attrs.size();
    assert(num_attributes);
    if (!expected_min_values.empty() && expected_min_values.size() != num_attributes) {
        throw std::invalid_argument("expected_min_values size must match number of min-aggregated attributes");
    }

    std::atomic<uint64_t> next_chunk_idx{0};

    struct alignas(64) ThreadLocalData {
        std::unique_ptr<uint64_t[]> min_values;
    };
    std::vector<ThreadLocalData> thread_local_data(num_threads);

    // Collect string attributes from all loaded tables for min operator string comparison
    std::unordered_set<std::string> string_attrs;
    for (const auto& table: tables) {
        for (const auto& col: table->string_columns) {
            string_attrs.insert(col);
        }
    }

    Schema schema_template;
    schema_template.dictionary = dict.get();
    schema_template.tables = tables;// Pass vector
    schema_template.min_values_size = num_attributes;
    schema_template.required_min_attrs = eff_min_attrs;
    schema_template.string_attributes = string_attrs.empty() ? nullptr : &string_attrs;

    // Populate Schema's adj_list_map for Schema-based lookup in operators
    populate_schema_adj_lists(schema_template, tables, query);

    // Initialize thread-local min_values arrays
    for (uint32_t tid = 0; tid < num_threads; ++tid) {
        thread_local_data[tid].min_values = std::make_unique<uint64_t[]>(num_attributes);
        std::fill_n(thread_local_data[tid].min_values.get(), num_attributes, std::numeric_limits<uint64_t>::max());
    }

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    // Benchmark barrier
    auto benchmark_barrier = []() {
        std::atomic_thread_fence(std::memory_order_seq_cst);
#if defined(__arm__) || defined(__aarch64__)
        __asm__ volatile("dmb ish" ::: "memory");
#else
        __asm__ volatile("" ::: "memory");
#endif
    };

    benchmark_barrier();
    const auto exec_start_time = std::chrono::steady_clock::now();

    // Launch worker threads
    for (uint32_t tid = 0; tid < num_threads; ++tid) {
        threads.emplace_back([&, tid]() {
            auto [plan, ftree] = map_ordering_to_plan_synchronized(query, column_ordering_vec, sink, 0, tables);
            Schema thread_schema = schema_template;
            thread_schema.min_values = thread_local_data[tid].min_values.get();
            plan->init(tables, ftree, &thread_schema);
            auto* scan_sync = dynamic_cast<ScanSynchronized*>(plan->get_first_op());
            auto* scan_sync_unpacked = dynamic_cast<ScanUnpackedSynchronized*>(plan->get_first_op());
            // Set max_id for multithreaded execution (total_ids - 1 since IDs are 0-indexed)
            if (scan_sync) {
                scan_sync->set_max_id(total_ids > 0 ? total_ids - 1 : 0);
            } else if (scan_sync_unpacked) {
                scan_sync_unpacked->set_max_id(total_ids > 0 ? total_ids - 1 : 0);
            }

            constexpr uint64_t batch_size = 4;

            for (uint64_t start_chunk = next_chunk_idx.fetch_add(batch_size, std::memory_order_relaxed);
                 start_chunk < total_chunks;
                 start_chunk = next_chunk_idx.fetch_add(batch_size, std::memory_order_relaxed)) {

                uint64_t end_chunk = std::min(start_chunk + batch_size, total_chunks);
                uint64_t last_valid_chunk = (total_ids > 0) ? ((total_ids - 1) / chunk_size) : 0;
                end_chunk = std::min(end_chunk, last_valid_chunk + 1);

                for (uint64_t chunk_idx = start_chunk; chunk_idx < end_chunk; ++chunk_idx) {
                    uint64_t start_id = chunk_idx * chunk_size;
                    if (scan_sync) {
                        scan_sync->set_start_id(start_id);
                    } else if (scan_sync_unpacked) {
                        scan_sync_unpacked->set_start_id(start_id);
                    }
                    plan->get_first_op()->execute();
                }
            }
        });
    }

    for (auto& thread: threads) {
        thread.join();
    }
    benchmark_barrier();

    const auto exec_end_time = std::chrono::steady_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(exec_end_time - exec_start_time);

    // Merge min_values from all threads
    std::vector<uint64_t> global_min_values(num_attributes, std::numeric_limits<uint64_t>::max());
    for (uint32_t tid = 0; tid < num_threads; ++tid) {
        for (size_t i = 0; i < num_attributes; ++i) {
            global_min_values[i] = std::min(global_min_values[i], thread_local_data[tid].min_values[i]);
        }
    }

    std::cout << std::endl;
    std::cout << "-------------- Report ----------------" << std::endl;
    std::cout << "Query: " << query_as_str << std::endl;
    std::cout << "Column Ordering: [" << column_ordering << "]" << std::endl;
    std::cout << "Number of Threads: " << num_threads << std::endl;
    std::cout << "------------- Operators --------------" << std::endl;
    std::cout << "(Operator statistics not available in multithreaded mode)" << std::endl;
    std::cout << "-------------- Results ---------------" << std::endl;
    std::cout << "Execution Duration: " << duration.count() << " msecs" << std::endl;

    // Print min values, converting string IDs to actual strings
    std::cout << "Actual Min Values: ";
    for (size_t i = 0; i < num_attributes; ++i) {
        const std::string& attr_name = eff_min_attrs[i];
        bool is_string_attr = string_attrs.count(attr_name) > 0;

        if (is_string_attr && global_min_values[i] != std::numeric_limits<uint64_t>::max()) {
            // Convert string ID to actual string value
            const ffx_str_t& str_value = dict->get_string(global_min_values[i]);
            std::cout << "'" << str_value.to_string() << "' ";
        } else {
            std::cout << global_min_values[i] << " ";
        }
    }
    std::cout << std::endl;

    // Also print attribute names for clarity
    std::cout << "Attribute Names:   ";
    for (size_t i = 0; i < num_attributes; ++i) {
        const std::string& attr_name = eff_min_attrs[i];
        bool is_string_attr = string_attrs.count(attr_name) > 0;
        std::cout << attr_name << (is_string_attr ? "(str) " : " ");
    }
    std::cout << std::endl;

    if (!expected_min_values.empty()) {
        std::cout << "Expected Min Values: ";
        for (const auto& val: expected_min_values) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    bool result = true;
    if (!expected_min_values.empty()) {
        for (size_t i = 0; i < num_attributes; ++i) {
            if (global_min_values[i] != std::stoull(expected_min_values[i])) {
                result = false;
                break;
            }
        }
    }
    std::cout << "--------------------------------------" << std::endl;

    return result;
}

// Main multi-threaded evaluation function with routing
bool evaluate_query_multithreaded(const std::string& serialized_root_dir, const std::string& query_as_str,
                                  const std::string& column_ordering, const std::string& sink_type_str,
                                  const std::vector<std::string>& expected_values, uint32_t num_threads) {

    if (!valid_sink_type_str(sink_type_str)) { throw std::invalid_argument("Invalid sink type: " + sink_type_str); }
    const auto [sink_base, sink_path] = parse_sink_spec(sink_type_str);
    if (sink_path.has_value()) {
        throw std::invalid_argument("Sink output path is not supported for multithreaded runner: " + sink_type_str);
    }
    const auto sink = sink_map[sink_base];
    num_threads = std::min(num_threads, std::thread::hardware_concurrency());
    if (num_threads == 0) { num_threads = 1; }

    Query query{query_as_str};
    if (query.requires_min_sink() && sink != SINK_MIN) {
        throw std::invalid_argument("Q(MIN(...)) requires sink type 'min'");
    }

    // Check if sink type is SINK_MIN
    if (sink == SINK_MIN) {
        return evaluate_query_multithreaded_min_sink(serialized_root_dir, query_as_str, column_ordering, sink_type_str,
                                                     expected_values, num_threads);
    }

    // If provided, only a single expected value is supported for count(*) sink
    if (!expected_values.empty() && expected_values.size() != 1) {
        throw std::invalid_argument("At most one expected count value is supported");
    }

    const auto column_ordering_vec = split(column_ordering);
    assert(sink == SINK_PACKED_NOOP || sink == SINK_PACKED || sink == SINK_UNPACKED_NOOP || sink == SINK_UNPACKED);

    // Determine total_ids
    uint64_t total_ids = 0;
    const std::string& root_attr = column_ordering_vec[0];

    auto pool = std::make_unique<StringPool>();
    auto dict = std::make_unique<StringDictionary>(pool.get());

    auto loaded_tables = load_tables(serialized_root_dir, query, dict.get(), pool.get());
    auto tables = filter_tables(loaded_tables, query);

    // Root table is always at index 0
    if (tables.empty()) { throw std::runtime_error("No tables available"); }
    const Table* root_table = tables[0];

    // Find the index of the root attribute in the table's columns
    // First column (index 0) -> use num_fwd_ids
    // Second column (index 1) -> use num_bwd_ids
    size_t attr_index = SIZE_MAX;
    for (size_t i = 0; i < root_table->columns.size(); i++) {
        if (root_table->columns[i] == root_attr) {
            attr_index = i;
            break;
        }
    }

    if (attr_index == SIZE_MAX) {
        throw std::runtime_error("Root attribute " + root_attr + " not found in table " + root_table->name);
    }

    if (attr_index == 0) {
        // First column -> forward adjacency lists
        total_ids = root_table->num_fwd_ids;
    } else if (attr_index == 1) {
        // Second column -> backward adjacency lists
        total_ids = root_table->num_bwd_ids;
    } else {
        throw std::runtime_error("Root attribute " + root_attr + " is at unexpected column index " +
                                 std::to_string(attr_index));
    }

    constexpr auto chunk_size = State::MAX_VECTOR_SIZE;
    // const uint64_t total_ids = table->num_ids;
    const uint64_t total_chunks = (total_ids + chunk_size - 1) / chunk_size;

    std::atomic<uint64_t> next_chunk_idx{0};
    std::vector<PaddedCounter> thread_output_counts(num_threads);
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    // Create schema template with adj_list_map populated
    Schema schema_template;
    schema_template.dictionary = dict.get();
    schema_template.tables = tables;
    populate_schema_adj_lists(schema_template, tables, query);

    auto benchmark_barrier = []() {
        std::atomic_thread_fence(std::memory_order_seq_cst);
#if defined(__arm__) || defined(__aarch64__)
        __asm__ volatile("dmb ish" ::: "memory");
#else
        __asm__ volatile("" ::: "memory");
#endif
    };

    benchmark_barrier();
    const auto exec_start_time = std::chrono::steady_clock::now();

    for (uint32_t tid = 0; tid < num_threads; ++tid) {
        threads.emplace_back([&, tid]() {
            auto [plan, ftree] = map_ordering_to_plan_synchronized(query, column_ordering_vec, sink, 0, tables);
            Schema schema = schema_template;// Copy the template with adj_list_map
            plan->init(tables, ftree, &schema);
            auto* scan_sync = dynamic_cast<ScanSynchronized*>(plan->get_first_op());
            auto* scan_sync_unpacked = dynamic_cast<ScanUnpackedSynchronized*>(plan->get_first_op());
            // Set max_id for multithreaded execution (total_ids - 1 since IDs are 0-indexed)
            if (scan_sync) {
                scan_sync->set_max_id(total_ids > 0 ? total_ids - 1 : 0);
            } else if (scan_sync_unpacked) {
                scan_sync_unpacked->set_max_id(total_ids > 0 ? total_ids - 1 : 0);
            }

            uint64_t local_output_count = 0;
            uint64_t prev_output_count = 0;
            constexpr uint64_t batch_size = 4;

            for (uint64_t start_chunk = next_chunk_idx.fetch_add(batch_size, std::memory_order_relaxed);
                 start_chunk < total_chunks;
                 start_chunk = next_chunk_idx.fetch_add(batch_size, std::memory_order_relaxed)) {

                uint64_t end_chunk = std::min(start_chunk + batch_size, total_chunks);
                uint64_t last_valid_chunk = (total_ids > 0) ? ((total_ids - 1) / chunk_size) : 0;
                end_chunk = std::min(end_chunk, last_valid_chunk + 1);

                for (uint64_t chunk_idx = start_chunk; chunk_idx < end_chunk; ++chunk_idx) {
                    uint64_t start_id = chunk_idx * chunk_size;
                    if (scan_sync) {
                        scan_sync->set_start_id(start_id);
                    } else if (scan_sync_unpacked) {
                        scan_sync_unpacked->set_start_id(start_id);
                    }
                    plan->get_first_op()->execute();
                    uint64_t curr_output_count = plan->get_num_output_tuples();
                    local_output_count += (curr_output_count - prev_output_count);
                    prev_output_count = curr_output_count;
                }
            }
            thread_output_counts[tid].count = local_output_count;
        });
    }

    for (auto& thread: threads) {
        thread.join();
    }
    benchmark_barrier();

    uint64_t total_output_tuples = 0;
    for (uint32_t i = 0; i < num_threads; ++i) {
        total_output_tuples += thread_output_counts[i].count;
    }

    const auto exec_end_time = std::chrono::steady_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(exec_end_time - exec_start_time);

    std::cout << std::endl;
    std::cout << "-------------- Report ----------------" << std::endl;
    std::cout << "Query: " << query_as_str << std::endl;
    std::cout << "Column Ordering: [" << column_ordering << "]" << std::endl;
    std::cout << "Number of Threads: " << num_threads << std::endl;
    std::cout << "------------- Operators --------------" << std::endl;
    std::cout << "(Operator statistics not available in multithreaded mode)" << std::endl;
    std::cout << "-------------- Results ---------------" << std::endl;
    std::cout << "Actual Output: " << total_output_tuples << std::endl;
    if (!expected_values.empty()) { std::cout << "Expected Output: " << expected_values.front() << std::endl; }
    std::cout << "Execution Duration: " << duration.count() << " msecs" << std::endl;
    std::cout << "--------------------------------------" << std::endl;

    bool result = true;
    if (!expected_values.empty()) { result = (total_output_tuples == std::stoull(expected_values.front())); }
    return result;
}

}// namespace ffx
