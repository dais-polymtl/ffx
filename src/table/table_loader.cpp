#include "include/table_loader.hpp"
#include "include/adj_list_builder.hpp"
#include "../../query/include/query.hpp"

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <climits>

namespace ffx {

std::set<std::string> determine_required_attributes(
    const Query& query,
    const std::string& src_attr,
    const std::string& dest_attr) {
    
    std::set<std::string> required;
    
    // Join attributes are always required
    required.insert(src_attr);
    required.insert(dest_attr);
    
    // Add attributes used in predicates (only if they match table attributes)
    // Note: Query variables (like "a", "b") are query aliases, not table attributes.
    // For predicates, we check if they reference the table's actual attributes.
    if (query.has_predicates()) {
        auto all_predicates = query.get_predicates().all_predicates();
        for (const auto& pred : all_predicates) {
            // Only add if it matches one of our table attributes
            if (pred.left_attr == src_attr || pred.left_attr == dest_attr) {
                required.insert(pred.left_attr);
            }
            if (pred.is_attribute() && 
                (pred.right_attr == src_attr || pred.right_attr == dest_attr)) {
                required.insert(pred.right_attr);
            }
        }
    }
    
    // Note: We don't add query variables here because they are query aliases,
    // not table attribute names. The table only has src_attr and dest_attr.
    
    return required;
}

std::pair<std::string, std::string> map_datalog_to_table_columns(
    const QueryRelation& query_rel,
    const std::vector<std::string>& table_columns,
    const std::string& table_name) {
    
    // Validate table name matches
    if (query_rel.tableName != table_name) {
        throw std::runtime_error("Table name mismatch: query has " + query_rel.tableName + 
                                " but table config has " + table_name);
    }
    
    // Validate we have at least 2 columns
    if (table_columns.size() < 2) {
        throw std::runtime_error("Table must have at least 2 columns for mapping");
    }
    
    // Map by position:
    // Query position 0 (fromVariable) -> Table column position 0 -> src_attr
    // Query position 1 (toVariable) -> Table column position 1 -> dest_attr
    std::string src_attr = table_columns[0];
    std::string dest_attr = table_columns[1];
    
    // Note: query_rel.fromVariable and query_rel.toVariable are query variable names
    // (like "q1", "q2"), not table column names. We ignore them and use position.
    
    return std::make_pair(src_attr, dest_attr);
}

LoadedTable load_table_from_columns_auto(
    const std::string& table_dir,
    const std::string& table_name,
    const std::vector<std::string>& table_columns,
    const Query& query,
    AdjListManager& adj_list_manager,
    const std::string& src_attr,
    const std::string& dest_attr) {
    
    std::string actual_src_attr = src_attr;
    std::string actual_dest_attr = dest_attr;
    
    // If query uses Datalog format, try to map by position
    if (query.is_datalog_format()) {
        // Find query relation that matches this table
        for (uint64_t i = 0; i < query.num_rels; i++) {
            const auto& rel = query.rels[i];
            if (rel.hasTableName() && rel.tableName == table_name) {
                try {
                    // Map Datalog query to table columns
                    auto [mapped_src, mapped_dest] = map_datalog_to_table_columns(
                        rel,
                        table_columns,
                        table_name
                    );
                    actual_src_attr = mapped_src;
                    actual_dest_attr = mapped_dest;
                    std::cout << "  Mapped Datalog query " << rel.tableName 
                              << "(" << rel.fromVariable << ", " << rel.toVariable << ")"
                              << " to table columns: " << actual_src_attr << " -> " << actual_dest_attr << std::endl;
                    break;
                } catch (const std::exception& e) {
                    // If mapping fails, fall back to provided src_attr/dest_attr
                    std::cerr << "  Warning: Datalog mapping failed: " << e.what() 
                              << ", using provided attributes" << std::endl;
                }
            }
        }
    }
    
    // Validate we have src and dest attributes
    if (actual_src_attr.empty() || actual_dest_attr.empty()) {
        throw std::runtime_error("Cannot determine src/dest attributes for table " + table_name);
    }
    
    // Call the regular load function
    return load_table_from_columns(
        table_dir,
        actual_src_attr,
        actual_dest_attr,
        query,
        adj_list_manager
    );
}

LoadedTable load_table_from_columns(
    const std::string& table_dir,
    const std::string& src_attr,
    const std::string& dest_attr,
    const Query& query,
    AdjListManager& adj_list_manager) {
    
    std::cout << "Loading table from columns in: " << table_dir << std::endl;
    std::cout << "  Join attributes: " << src_attr << " -> " << dest_attr << std::endl;
    
    // Step 1: Load metadata
    SerializedTableMetadata metadata = read_metadata_binary(table_dir);
    std::cout << "  Found " << metadata.columns.size() << " columns, " << metadata.num_rows << " rows" << std::endl;
    
    // Step 2: Determine required attributes
    std::set<std::string> required_attrs = determine_required_attributes(query, src_attr, dest_attr);
    std::cout << "  Required attributes: ";
    for (const auto& attr : required_attrs) {
        std::cout << attr << " ";
    }
    std::cout << std::endl;
    
    // Step 3: Create LoadedTable and initialize global string pool/dictionary
    LoadedTable loaded_table;
    loaded_table.num_rows = metadata.num_rows;
    loaded_table.global_string_pool = std::make_unique<StringPool>();
    loaded_table.global_string_dict = std::make_unique<StringDictionary>(loaded_table.global_string_pool.get());
    
    // Step 4: Deserialize required columns
    for (const auto& attr : required_attrs) {
        const SerializedColumnInfo* col_info = metadata.find_column(attr);
        if (!col_info) {
            throw std::runtime_error("Required attribute not found in metadata: " + attr);
        }
        
        if (col_info->type == ColumnType::UINT64) {
            // Deserialize uint64 column
            uint64_t num_rows;
            auto data = deserialize_uint64_column(
                table_dir + "/" + attr + "_uint64.bin",
                num_rows
            );
            
            if (num_rows != metadata.num_rows) {
                throw std::runtime_error("Row count mismatch for column " + attr + 
                    ": expected " + std::to_string(metadata.num_rows) + 
                    ", got " + std::to_string(num_rows));
            }
            
            LoadedColumnUInt64 col;
            col.name = attr;
            col.data = std::move(data);
            col.num_rows = num_rows;
            col.max_value = col_info->max_value;
            loaded_table.uint64_columns[attr] = std::move(col);
            
            std::cout << "  Loaded uint64 column: " << attr << " (max=" << col_info->max_value << ")" << std::endl;
            
        } else if (col_info->type == ColumnType::STRING) {
            // Deserialize string column
            uint64_t num_rows;
            auto data = deserialize_string_column(
                table_dir + "/" + attr + "_string.bin",
                num_rows,
                loaded_table.global_string_pool.get()
            );
            
            if (num_rows != metadata.num_rows) {
                throw std::runtime_error("Row count mismatch for column " + attr + 
                    ": expected " + std::to_string(metadata.num_rows) + 
                    ", got " + std::to_string(num_rows));
            }
            
            // Build ID column using global dictionary
            auto id_column = std::make_unique<uint64_t[]>(num_rows);
            for (uint64_t i = 0; i < num_rows; i++) {
                if (!data[i].is_null()) {
                    id_column[i] = loaded_table.global_string_dict->add_string(data[i]);
                } else {
                    id_column[i] = UINT64_MAX; // NULL ID representation
                }
            }
            
            LoadedColumnString col;
            col.name = attr;
            col.data = std::move(data);
            col.num_rows = num_rows;
            col.id_column = std::move(id_column);
            loaded_table.string_columns[attr] = std::move(col);
            
            std::cout << "  Loaded string column: " << attr << std::endl;
        }
    }
    
    // Step 5: Finalize global dictionary after processing all string columns
    if (loaded_table.global_string_dict) {
        loaded_table.global_string_dict->finalize();
        std::cout << "  Global string dictionary finalized with " 
                  << loaded_table.global_string_dict->size() << " unique strings" << std::endl;
    }
    
    // Step 6: Build adjacency lists for join attributes
    // Determine ID columns (numeric or string IDs)
    const uint64_t* src_id_col = nullptr;
    const uint64_t* dest_id_col = nullptr;
    uint64_t num_fwd_ids = 0;
    uint64_t num_bwd_ids = 0;
    
    if (loaded_table.uint64_columns.count(src_attr)) {
        // Numeric source attribute
        src_id_col = loaded_table.uint64_columns[src_attr].data.get();
        num_fwd_ids = loaded_table.uint64_columns[src_attr].max_value + 1;
    } else if (loaded_table.string_columns.count(src_attr)) {
        // String source attribute - use ID column
        src_id_col = loaded_table.string_columns[src_attr].id_column.get();
        num_fwd_ids = loaded_table.global_string_dict->size();
    } else {
        throw std::runtime_error("Source attribute not found in loaded columns: " + src_attr);
    }
    
    if (loaded_table.uint64_columns.count(dest_attr)) {
        // Numeric destination attribute
        dest_id_col = loaded_table.uint64_columns[dest_attr].data.get();
        num_bwd_ids = loaded_table.uint64_columns[dest_attr].max_value + 1;
    } else if (loaded_table.string_columns.count(dest_attr)) {
        // String destination attribute - use ID column
        dest_id_col = loaded_table.string_columns[dest_attr].id_column.get();
        num_bwd_ids = loaded_table.global_string_dict->size();
    } else {
        throw std::runtime_error("Destination attribute not found in loaded columns: " + dest_attr);
    }
    
    // Build adjacency lists
    std::unique_ptr<AdjList<uint64_t>[]> fwd_adj, bwd_adj;
    build_adj_lists_from_columns(
        src_id_col,
        dest_id_col,
        loaded_table.num_rows,
        num_fwd_ids,
        num_bwd_ids,
        fwd_adj,
        bwd_adj
    );
    
    std::cout << "  Built adjacency lists: fwd(" << num_fwd_ids << "), bwd(" << num_bwd_ids << ")" << std::endl;
    
    // Step 7: Register adjacency lists with AdjListManager
    std::string logical_table_id = table_dir; // Use table_dir as logical ID
    
    adj_list_manager.register_adj_lists(
        logical_table_id,
        src_attr,
        dest_attr,
        true,  // is_fwd
        std::move(fwd_adj),
        num_fwd_ids
    );
    
    adj_list_manager.register_adj_lists(
        logical_table_id,
        src_attr,
        dest_attr,
        false,  // is_fwd
        std::move(bwd_adj),
        num_bwd_ids
    );
    
    std::cout << "  Registered adjacency lists with AdjListManager" << std::endl;
    
    return loaded_table;
}

} // namespace ffx

