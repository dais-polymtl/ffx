#ifndef FFX_TABLE_LOADER_UTILS_HPP
#define FFX_TABLE_LOADER_UTILS_HPP

#include "../../operator/include/schema/schema.hpp"
#include "../../operator/include/schema/adj_list_manager.hpp"
#include "../../query/include/query.hpp"
#include <string>

namespace ffx {

/**
 * Populate Schema's adj_list_map from AdjListManager for a query.
 * 
 * This function queries AdjListManager for adjacency lists needed by the query
 * and registers them in the Schema.
 * 
 * @param schema Schema to populate
 * @param query Query object
 * @param table_dir Directory containing serialized data (used as logical_table_id)
 * @param src_attr Source attribute name
 * @param dest_attr Destination attribute name
 */
void populate_schema_from_adj_list_manager(
    Schema& schema,
    const Query& query,
    const std::string& table_dir,
    const std::string& src_attr,
    const std::string& dest_attr,
    AdjListManager& adj_list_manager
);

} // namespace ffx

#endif // FFX_TABLE_LOADER_UTILS_HPP

