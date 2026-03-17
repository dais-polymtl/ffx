#include "include/table_metadata.hpp"

#include <fstream>
#include <stdexcept>
#include <iostream>

namespace ffx {

SerializedTableMetadata read_metadata_binary(const std::string& output_dir) {
    std::string path = output_dir + "/table_metadata.bin";
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Cannot open metadata file: " + path);
    }
    
    SerializedTableMetadata metadata;
    
    // Read num_rows
    ifs.read(reinterpret_cast<char*>(&metadata.num_rows), sizeof(uint64_t));
    if (!ifs) {
        throw std::runtime_error("Failed to read num_rows from metadata file");
    }
    
    // Read num_columns
    uint64_t num_cols;
    ifs.read(reinterpret_cast<char*>(&num_cols), sizeof(uint64_t));
    if (!ifs) {
        throw std::runtime_error("Failed to read num_columns from metadata file");
    }
    
    // Read each column info
    metadata.columns.reserve(num_cols);
    for (uint64_t i = 0; i < num_cols; i++) {
        SerializedColumnInfo col_info;
        
        // Read column name length and name
        uint64_t name_len;
        ifs.read(reinterpret_cast<char*>(&name_len), sizeof(uint64_t));
        if (!ifs) {
            throw std::runtime_error("Failed to read column name length for column " + std::to_string(i));
        }
        
        std::string name(name_len, '\0');
        ifs.read(&name[0], name_len);
        if (!ifs) {
            throw std::runtime_error("Failed to read column name for column " + std::to_string(i));
        }
        col_info.attr_name = name;
        
        // Read type (0 = uint64, 1 = string)
        uint8_t type_byte;
        ifs.read(reinterpret_cast<char*>(&type_byte), sizeof(uint8_t));
        if (!ifs) {
            throw std::runtime_error("Failed to read column type for column " + std::to_string(i));
        }
        col_info.type = (type_byte == 0) ? ColumnType::UINT64 : ColumnType::STRING;
        
        // Read column_idx
        uint64_t col_idx;
        ifs.read(reinterpret_cast<char*>(&col_idx), sizeof(uint64_t));
        if (!ifs) {
            throw std::runtime_error("Failed to read column_idx for column " + std::to_string(i));
        }
        col_info.column_idx = static_cast<size_t>(col_idx);
        
        // Read max_value
        ifs.read(reinterpret_cast<char*>(&col_info.max_value), sizeof(uint64_t));
        if (!ifs) {
            throw std::runtime_error("Failed to read max_value for column " + std::to_string(i));
        }
        
        // Populate map before moving (use name variable, not col_info.attr_name after move)
        metadata.attr_to_index[name] = i;
        metadata.columns.push_back(std::move(col_info));
    }

    uint64_t num_projections = 0;
    ifs.read(reinterpret_cast<char*>(&num_projections), sizeof(uint64_t));
    if (!ifs) {
        throw std::runtime_error("Failed to read num_projections from metadata file");
    }

    metadata.projections.reserve(num_projections);
    for (uint64_t i = 0; i < num_projections; i++) {
        SerializedProjectionInfo projection_info;

        uint64_t name_len;
        ifs.read(reinterpret_cast<char*>(&name_len), sizeof(uint64_t));
        if (!ifs) {
            throw std::runtime_error("Failed to read projection name length for projection " + std::to_string(i));
        }

        std::string relation_name(name_len, '\0');
        ifs.read(&relation_name[0], name_len);
        if (!ifs) {
            throw std::runtime_error("Failed to read projection name for projection " + std::to_string(i));
        }
        projection_info.relation_name = relation_name;

        uint64_t src_idx = 0;
        uint64_t dst_idx = 1;
        ifs.read(reinterpret_cast<char*>(&src_idx), sizeof(uint64_t));
        ifs.read(reinterpret_cast<char*>(&dst_idx), sizeof(uint64_t));
        if (!ifs) {
            throw std::runtime_error("Failed to read projection column indices for projection " + std::to_string(i));
        }
        projection_info.source_column_idx = static_cast<size_t>(src_idx);
        projection_info.target_column_idx = static_cast<size_t>(dst_idx);

        uint8_t card_byte = 3;
        ifs.read(reinterpret_cast<char*>(&card_byte), sizeof(uint8_t));
        if (!ifs) {
            throw std::runtime_error("Failed to read projection cardinality for projection " + std::to_string(i));
        }
        switch (card_byte) {
            case 0:
                projection_info.cardinality = ProjectionCardinality::ONE_TO_ONE;
                break;
            case 1:
                projection_info.cardinality = ProjectionCardinality::ONE_TO_MANY;
                break;
            case 2:
                projection_info.cardinality = ProjectionCardinality::MANY_TO_ONE;
                break;
            case 3:
            default:
                projection_info.cardinality = ProjectionCardinality::MANY_TO_MANY;
                break;
        }

        metadata.projection_to_index[relation_name] = i;
        metadata.projections.push_back(std::move(projection_info));
    }
    
    ifs.close();
    return metadata;
}

} // namespace ffx

