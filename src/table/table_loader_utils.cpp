#include "include/table_loader_utils.hpp"

#include <iostream>

namespace ffx {

void populate_schema_from_adj_list_manager(
    Schema& schema,
    const Query& query,
    const std::string& table_dir,
    const std::string& src_attr,
    const std::string& dest_attr,
    AdjListManager& adj_list_manager) {
    
    // Set AdjListManager in schema
    schema.adj_list_manager = &adj_list_manager;
    
    // Check if adjacency lists are registered/available
    // Note: We populate schema if adjacency lists exist, regardless of query variable names.
    // Query variables (like "a", "b") are aliases; table attributes (like "src", "dest") are the actual columns.
    
    // Register forward adj_list: src_attr -> dest_attr
    try {
        AdjList<uint64_t>* fwd_adj = adj_list_manager.get_or_load(
            table_dir, src_attr, dest_attr, true
        );
        uint64_t num_fwd = adj_list_manager.get_num_adj_lists(
            table_dir, src_attr, dest_attr, true
        );
        
        schema.register_adj_list(src_attr, dest_attr, fwd_adj, num_fwd);
        std::cout << "Schema: Registered adj_list " << src_attr << "->" << dest_attr 
                  << " (fwd, " << num_fwd << " entries) from AdjListManager" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "  Warning: Could not load forward adj list: " << e.what() << std::endl;
    }
    
    // Register backward adj_list: dest_attr -> src_attr
    try {
        AdjList<uint64_t>* bwd_adj = adj_list_manager.get_or_load(
            table_dir, src_attr, dest_attr, false
        );
        uint64_t num_bwd = adj_list_manager.get_num_adj_lists(
            table_dir, src_attr, dest_attr, false
        );
        
        schema.register_adj_list(dest_attr, src_attr, bwd_adj, num_bwd);
        std::cout << "Schema: Registered adj_list " << dest_attr << "->" << src_attr 
                  << " (bwd, " << num_bwd << " entries) from AdjListManager" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "  Warning: Could not load backward adj list: " << e.what() << std::endl;
    }
}

} // namespace ffx

