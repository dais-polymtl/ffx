#ifndef FFX_STORAGE_API_HPP
#define FFX_STORAGE_API_HPP

#include "../../table/include/ffx_str_t.hpp"
#include "../../table/include/string_dictionary.hpp"
#include "../../table/include/string_pool.hpp"
#include "../../table/include/table.hpp"

#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace ffx {

enum class ColumnType { UINT64, STRING };

enum class ProjectionCardinality { ONE_TO_ONE, ONE_TO_MANY, MANY_TO_ONE, MANY_TO_MANY };

struct ColumnConfig {
    std::string attr_name;
    size_t column_idx;
    ColumnType type;

    ColumnConfig(std::string name, size_t idx, ColumnType t)
        : attr_name(std::move(name)), column_idx(idx), type(t) {}
};

struct ProjectionConfig {
    std::string relation_name;
    size_t source_column_idx;
    size_t target_column_idx;
    ProjectionCardinality cardinality;

    ProjectionConfig(std::string name, size_t src_idx, size_t dst_idx, ProjectionCardinality card)
        : relation_name(std::move(name)), source_column_idx(src_idx), target_column_idx(dst_idx), cardinality(card) {}
};

struct ParsedTableConfig {
    std::vector<ColumnConfig> columns;
    std::vector<ProjectionConfig> projections;

    size_t size() const { return columns.size(); }
    const ColumnConfig& operator[](size_t idx) const { return columns[idx]; }
    ColumnConfig& operator[](size_t idx) { return columns[idx]; }
    auto begin() { return columns.begin(); }
    auto end() { return columns.end(); }
    auto begin() const { return columns.begin(); }
    auto end() const { return columns.end(); }
    operator const std::vector<ColumnConfig>&() const { return columns; }
    operator std::vector<ColumnConfig>&() { return columns; }
};

ParsedTableConfig parse_column_config(const std::string& config_file);

void serialize_uint64_column(const std::vector<uint64_t>& data, const std::string& output_path,
                             uint64_t* out_max_value = nullptr);

void serialize_string_column(const std::vector<std::string>& data, const std::vector<bool>& is_null,
                             const std::string& output_path);

std::unique_ptr<uint64_t[]> deserialize_uint64_column(const std::string& input_path, uint64_t& num_rows);

std::unique_ptr<ffx_str_t[]> deserialize_string_column(const std::string& input_path, uint64_t& num_rows,
                                                       StringPool* pool);

void write_metadata_json(const std::string& output_dir, const std::vector<ColumnConfig>& columns, uint64_t num_rows,
                                                 const std::vector<uint64_t>& max_values,
                                                 const std::vector<ProjectionConfig>& projections = {});

void write_metadata_binary(const std::string& output_dir, const std::vector<ColumnConfig>& columns, uint64_t num_rows,
                                                     const std::vector<uint64_t>& max_values,
                                                     const std::vector<ProjectionConfig>& projections = {});

void serialize(const Table& table, const std::string& output_dir, const std::string& src_attr = "src",
               const std::string& dest_attr = "dest", const StringDictionary* dict = nullptr);

std::unique_ptr<Table> deserialize(const std::string& ser_dir, const std::string& src_attr = "src",
                                   const std::string& dest_attr = "dest", StringDictionary* dict = nullptr);

template<typename T>
inline void serialize_value(const T& value, std::ofstream& out) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template<typename T>
inline void deserialize_value(T& value, std::ifstream& in, StringPool* pool = nullptr) {
    (void) pool;
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
}

template<>
inline void serialize_value<ffx_str_t>(const ffx_str_t& str, std::ofstream& out) {
    str.serialize(out);
}

template<>
inline void deserialize_value<ffx_str_t>(ffx_str_t& str, std::ifstream& in, StringPool* pool) {
    str = ffx_str_t::deserialize(in, pool);
}

bool is_string_type(const std::string& ser_dir);

} // namespace ffx

#endif // FFX_STORAGE_API_HPP