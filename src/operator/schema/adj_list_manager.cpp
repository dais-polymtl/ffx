#include "../include/schema/adj_list_manager.hpp"
#include "../../ser_der/include/storage_api.hpp"

#include <fstream>
#include <iostream>
#include <stdexcept>

namespace ffx {

std::string AdjListManager::make_cache_key(const std::string& data_dir,
                                            const std::string& src_attr,
                                            const std::string& dest_attr,
                                            bool is_fwd) const {
    return data_dir + "|" + src_attr + "|" + dest_attr + "|" + (is_fwd ? "fwd" : "bwd");
}

bool AdjListManager::is_loaded(const std::string& data_dir,
                                const std::string& src_attr,
                                const std::string& dest_attr,
                                bool is_fwd) const {
    std::string key = make_cache_key(data_dir, src_attr, dest_attr, is_fwd);
    return _cache.find(key) != _cache.end();
}

AdjList<uint64_t>* AdjListManager::get_or_load(const std::string& data_dir,
                                                const std::string& src_attr,
                                                const std::string& dest_attr,
                                                bool is_fwd) {
    std::string key = make_cache_key(data_dir, src_attr, dest_attr, is_fwd);
    
    auto it = _cache.find(key);
    if (it != _cache.end()) {
        return it->second.adj_lists.get();
    }
    
    return load_adj_list(data_dir, src_attr, dest_attr, is_fwd);
}

AdjList<uint64_t>* AdjListManager::load_adj_list(const std::string& data_dir,
                                                  const std::string& src_attr,
                                                  const std::string& dest_attr,
                                                  bool is_fwd) {
    std::string key = make_cache_key(data_dir, src_attr, dest_attr, is_fwd);
    
    // Read num_adj_lists file - try new format first, fall back to old format
    std::string num_file = data_dir + "/num_adj_lists_" + src_attr + "_" + dest_attr + ".bin";
    std::ifstream ifs(num_file, std::ios::binary);
    if (!ifs) {
        // Fall back to old format for backward compatibility
        num_file = data_dir + "/num_adj_lists.bin";
        ifs.open(num_file, std::ios::binary);
        if (!ifs) {
            throw std::runtime_error("AdjListManager: Cannot open " + num_file);
        }
    }
    
    uint64_t num_fwd, num_bwd;
    ifs.read(reinterpret_cast<char*>(&num_fwd), sizeof(uint64_t));
    ifs.read(reinterpret_cast<char*>(&num_bwd), sizeof(uint64_t));
    ifs.close();
    
    uint64_t num_adj_lists = is_fwd ? num_fwd : num_bwd;
    
    // Deserialize the adjacency lists using existing infrastructure
    auto adj_lists = std::make_unique<AdjList<uint64_t>[]>(num_adj_lists);
    
    // Use the deserialize function from ser_des
    // Note: This expects files in format: fwd_src_dest_<start>_<end>.bin
    auto loaded_table = deserialize(data_dir, src_attr, dest_attr);
    
    // Copy the appropriate adjacency lists
    if (is_fwd) {
        for (uint64_t i = 0; i < num_adj_lists; i++) {
            adj_lists[i] = std::move(loaded_table->fwd_adj_lists[i]);
        }
    } else {
        for (uint64_t i = 0; i < num_adj_lists; i++) {
            adj_lists[i] = std::move(loaded_table->bwd_adj_lists[i]);
        }
    }
    
    // Store in cache
    CacheEntry entry;
    entry.adj_lists = std::move(adj_lists);
    entry.num_adj_lists = num_adj_lists;
    
    AdjList<uint64_t>* result = entry.adj_lists.get();
    _cache[key] = std::move(entry);
    
    std::cout << "AdjListManager: Loaded " << (is_fwd ? "fwd" : "bwd") 
              << " adj list for " << src_attr << "->" << dest_attr 
              << " (" << num_adj_lists << " entries)" << std::endl;
    
    return result;
}

uint64_t AdjListManager::get_num_adj_lists(const std::string& data_dir,
                                            const std::string& src_attr,
                                            const std::string& dest_attr,
                                            bool is_fwd) const {
    std::string key = make_cache_key(data_dir, src_attr, dest_attr, is_fwd);
    
    auto it = _cache.find(key);
    if (it != _cache.end()) {
        return it->second.num_adj_lists;
    }
    
    // Not loaded yet - read from file - try new format first, fall back to old format
    std::string num_file = data_dir + "/num_adj_lists_" + src_attr + "_" + dest_attr + ".bin";
    std::ifstream ifs(num_file, std::ios::binary);
    if (!ifs) {
        // Fall back to old format for backward compatibility
        num_file = data_dir + "/num_adj_lists.bin";
        ifs.open(num_file, std::ios::binary);
        if (!ifs) {
            throw std::runtime_error("AdjListManager: Cannot open " + num_file);
        }
    }
    
    uint64_t num_fwd, num_bwd;
    ifs.read(reinterpret_cast<char*>(&num_fwd), sizeof(uint64_t));
    ifs.read(reinterpret_cast<char*>(&num_bwd), sizeof(uint64_t));
    
    return is_fwd ? num_fwd : num_bwd;
}

void AdjListManager::register_adj_lists(const std::string& logical_table_id,
                                         const std::string& src_attr,
                                         const std::string& dest_attr,
                                         bool is_fwd,
                                         std::unique_ptr<AdjList<uint64_t>[]>&& adj_lists,
                                         uint64_t num_adj_lists) {
    std::string key = make_cache_key(logical_table_id, src_attr, dest_attr, is_fwd);
    
    CacheEntry entry;
    entry.adj_lists = std::move(adj_lists);
    entry.num_adj_lists = num_adj_lists;
    
    _cache[key] = std::move(entry);
    
    std::cout << "AdjListManager: Registered " << (is_fwd ? "fwd" : "bwd")
              << " adj list for " << src_attr << "->" << dest_attr
              << " (" << num_adj_lists << " entries)" << std::endl;
}

void AdjListManager::clear() {
    _cache.clear();
}

}  // namespace ffx
