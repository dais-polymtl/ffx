#ifndef FFX_LOADED_TABLE_HPP
#define FFX_LOADED_TABLE_HPP

#include "string_dictionary.hpp"
#include "ffx_str_t.hpp"
#include <memory>
#include <string>
#include <unordered_map>

namespace ffx {

struct LoadedColumnUInt64 {
    std::string name;
    std::unique_ptr<uint64_t[]> data;
    uint64_t num_rows;
    uint64_t max_value;
    
    LoadedColumnUInt64() : num_rows(0), max_value(0) {}
};

struct LoadedColumnString {
    std::string name;
    std::unique_ptr<ffx_str_t[]> data;
    uint64_t num_rows;
    
    // ID column for joins (strings converted to IDs using global dictionary)
    std::unique_ptr<uint64_t[]> id_column;
    
    LoadedColumnString() : num_rows(0) {}
};

struct LoadedTable {
    uint64_t num_rows;
    std::unordered_map<std::string, LoadedColumnUInt64> uint64_columns;
    std::unordered_map<std::string, LoadedColumnString> string_columns;
    
    // Global string dictionary and pool shared across all string attributes
    std::unique_ptr<StringPool> global_string_pool;
    std::unique_ptr<StringDictionary> global_string_dict;
    
    LoadedTable() : num_rows(0) {}
    
    // Helper methods
    const uint64_t* get_uint64_column(const std::string& attr) const {
        auto it = uint64_columns.find(attr);
        if (it == uint64_columns.end()) return nullptr;
        return it->second.data.get();
    }
    
    const ffx_str_t* get_string_column(const std::string& attr) const {
        auto it = string_columns.find(attr);
        if (it == string_columns.end()) return nullptr;
        return it->second.data.get();
    }
    
    const uint64_t* get_string_id_column(const std::string& attr) const {
        auto it = string_columns.find(attr);
        if (it == string_columns.end()) return nullptr;
        return it->second.id_column.get();
    }
    
    StringDictionary* get_string_dict() const { 
        return global_string_dict.get(); 
    }
};

} // namespace ffx

#endif // FFX_LOADED_TABLE_HPP

