#ifndef FFX_TABLE_LOADER_HPP
#define FFX_TABLE_LOADER_HPP

#include "loaded_table.hpp"
#include "../../ser_der/include/table_metadata.hpp"
#include "../../ser_der/include/serializer.hpp"
#include "../../operator/include/schema/adj_list_manager.hpp"
#include "../../query/include/query.hpp"
#include <memory>
#include <string>
#include <set>
#include <vector>
#include <utility>

namespace ffx {

/**
 * Load table from column-serialized format.
 * 
 * This function:
 * 1. Reads metadata to determine available columns
 * 2. Determines which columns are needed based on query
 * 3. Deserializes only required columns
 * 4. Builds global string dictionary for all string columns
 * 5. Builds adjacency lists for join attributes
 * 6. Registers adjacency lists with AdjListManager
 * 
 * @param table_dir Directory containing serialized column files
 * @param src_attr Source attribute name for joins
 * @param dest_attr Destination attribute name for joins
 * @param query Query object to determine required attributes
 * @param adj_list_manager AdjListManager to register adjacency lists
 * @return LoadedTable with deserialized columns
 */
LoadedTable load_table_from_columns(
    const std::string& table_dir,
    const std::string& src_attr,
    const std::string& dest_attr,
    const Query& query,
    AdjListManager& adj_list_manager
);

/**
 * Determine which attributes are required for a query.
 * 
 * @param query Query object
 * @param src_attr Source attribute (always needed for joins)
 * @param dest_attr Destination attribute (always needed for joins)
 * @return Set of required attribute names
 */
std::set<std::string> determine_required_attributes(
    const Query& query,
    const std::string& src_attr,
    const std::string& dest_attr
);

/**
 * Map Datalog query relation to table columns by position.
 * 
 * For a Datalog query like "T(q1, q2)" and table "T(a1, a2, a3)":
 * - q1 (position 0) maps to a1 (position 0) -> src_attr
 * - q2 (position 1) maps to a2 (position 1) -> dest_attr
 * 
 * @param query_rel QueryRelation from Datalog query
 * @param table_columns Vector of table column names (in order)
 * @param table_name Table name from config (for validation)
 * @return Pair of (src_attr, dest_attr) column names
 */
std::pair<std::string, std::string> map_datalog_to_table_columns(
    const QueryRelation& query_rel,
    const std::vector<std::string>& table_columns,
    const std::string& table_name
);

/**
 * Load table from column-serialized format with automatic Datalog mapping.
 * 
 * This is a convenience function that:
 * 1. If query uses Datalog format, automatically maps query variables to table columns by position
 * 2. Otherwise, uses the provided src_attr and dest_attr directly
 * 
 * @param table_dir Directory containing serialized column files
 * @param table_name Table name from config (for Datalog matching)
 * @param table_columns Vector of table column names (in order, from config)
 * @param query Query object
 * @param adj_list_manager AdjListManager to register adjacency lists
 * @param src_attr Source attribute (used if not Datalog format, or as fallback)
 * @param dest_attr Destination attribute (used if not Datalog format, or as fallback)
 * @return LoadedTable with deserialized columns
 */
LoadedTable load_table_from_columns_auto(
    const std::string& table_dir,
    const std::string& table_name,
    const std::vector<std::string>& table_columns,
    const Query& query,
    AdjListManager& adj_list_manager,
    const std::string& src_attr = "",
    const std::string& dest_attr = ""
);

} // namespace ffx

#endif // FFX_TABLE_LOADER_HPP

