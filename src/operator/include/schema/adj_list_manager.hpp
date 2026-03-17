#ifndef FFX_ADJ_LIST_MANAGER_HPP
#define FFX_ADJ_LIST_MANAGER_HPP

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "../../table/include/adj_list.hpp"

namespace ffx {

/**
 * AdjListManager - Manages loading and caching of adjacency lists.
 * 
 * Adjacency lists are loaded from serialized files created by index_creator.
 * Files use the format: fwd_<src>_<dest>_<start>_<end>.bin
 * 
 * The manager caches loaded adjacency lists to avoid reloading.
 */
class AdjListManager {
public:
    AdjListManager() = default;
    ~AdjListManager() = default;

    // Non-copyable
    AdjListManager(const AdjListManager&) = delete;
    AdjListManager& operator=(const AdjListManager&) = delete;

    /**
     * Load adjacency list from serialized directory.
     * 
     * @param data_dir   Directory containing serialized adj list files
     * @param src_attr   Source attribute name (as used in index_creator)
     * @param dest_attr  Destination attribute name (as used in index_creator)
     * @param is_fwd     True for forward adj list, false for backward
     * @return Pointer to loaded AdjList array (owned by manager)
     */
    AdjList<uint64_t>* load_adj_list(const std::string& data_dir,
                                      const std::string& src_attr,
                                      const std::string& dest_attr,
                                      bool is_fwd);

    /**
     * Get cached adjacency list, or load if not cached.
     * 
     * @param data_dir   Directory containing serialized adj list files
     * @param src_attr   Source attribute name
     * @param dest_attr  Destination attribute name
     * @param is_fwd     True for forward, false for backward
     * @return Pointer to AdjList array (owned by manager)
     */
    AdjList<uint64_t>* get_or_load(const std::string& data_dir,
                                    const std::string& src_attr,
                                    const std::string& dest_attr,
                                    bool is_fwd);

    /**
     * Get number of adjacency lists (max ID + 1) for a loaded adj list.
     */
    uint64_t get_num_adj_lists(const std::string& data_dir,
                                const std::string& src_attr,
                                const std::string& dest_attr,
                                bool is_fwd) const;

    /**
     * Clear all cached adjacency lists.
     */
    void clear();

    /**
     * Check if adjacency list is already loaded/cached.
     */
    bool is_loaded(const std::string& data_dir,
                   const std::string& src_attr,
                   const std::string& dest_attr,
                   bool is_fwd) const;

    /**
     * Register in-memory adjacency lists (built from vectors).
     * This allows adjacency lists created from deserialized columns to be
     * registered with the manager for use by operators.
     * 
     * @param logical_table_id  Logical identifier (e.g., table_dir or query_id)
     * @param src_attr         Source attribute name
     * @param dest_attr        Destination attribute name
     * @param is_fwd           True for forward, false for backward
     * @param adj_lists        Adjacency list array (ownership transferred)
     * @param num_adj_lists    Number of adjacency lists (max ID + 1)
     */
    void register_adj_lists(const std::string& logical_table_id,
                           const std::string& src_attr,
                           const std::string& dest_attr,
                           bool is_fwd,
                           std::unique_ptr<AdjList<uint64_t>[]>&& adj_lists,
                           uint64_t num_adj_lists);

private:
    // Cache key: "data_dir|src_attr|dest_attr|fwd" or "data_dir|src_attr|dest_attr|bwd"
    std::string make_cache_key(const std::string& data_dir,
                                const std::string& src_attr,
                                const std::string& dest_attr,
                                bool is_fwd) const;

    // Cached adjacency lists: key -> (adj_list_array, num_adj_lists)
    struct CacheEntry {
        std::unique_ptr<AdjList<uint64_t>[]> adj_lists;
        uint64_t num_adj_lists;
    };
    std::unordered_map<std::string, CacheEntry> _cache;
};

}  // namespace ffx

#endif  // FFX_ADJ_LIST_MANAGER_HPP
