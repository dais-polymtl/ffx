#ifndef FFX_TABLE_METADATA_HPP
#define FFX_TABLE_METADATA_HPP

#include "serializer.hpp"
#include <string>
#include <vector>
#include <unordered_map>

namespace ffx {

struct SerializedColumnInfo {
    std::string attr_name;
    ColumnType type;
    size_t column_idx;
    uint64_t max_value;
    
    SerializedColumnInfo() 
        : type(ColumnType::UINT64), column_idx(0), max_value(0) {}
    
    SerializedColumnInfo(std::string name, ColumnType t, size_t idx, uint64_t max_val)
        : attr_name(std::move(name)), type(t), column_idx(idx), max_value(max_val) {}
};

struct SerializedProjectionInfo {
    std::string relation_name;
    size_t source_column_idx;
    size_t target_column_idx;
    ProjectionCardinality cardinality;

    SerializedProjectionInfo()
        : source_column_idx(0), target_column_idx(1), cardinality(ProjectionCardinality::MANY_TO_MANY) {}

    SerializedProjectionInfo(std::string name, size_t src_idx, size_t dst_idx, ProjectionCardinality card)
        : relation_name(std::move(name)), source_column_idx(src_idx), target_column_idx(dst_idx), cardinality(card) {}
};

struct SerializedTableMetadata {
    uint64_t num_rows;
    std::vector<SerializedColumnInfo> columns;
    std::unordered_map<std::string, size_t> attr_to_index; // For fast lookup
    std::vector<SerializedProjectionInfo> projections;
    std::unordered_map<std::string, size_t> projection_to_index;
    
    SerializedTableMetadata() : num_rows(0) {}
    
    const SerializedColumnInfo* find_column(const std::string& attr_name) const {
        auto it = attr_to_index.find(attr_name);
        if (it == attr_to_index.end()) return nullptr;
        return &columns[it->second];
    }

    const SerializedProjectionInfo* find_projection(const std::string& relation_name) const {
        auto it = projection_to_index.find(relation_name);
        if (it == projection_to_index.end()) return nullptr;
        return &projections[it->second];
    }
};

// Read metadata from binary file
SerializedTableMetadata read_metadata_binary(const std::string& output_dir);

} // namespace ffx

#endif // FFX_TABLE_METADATA_HPP

