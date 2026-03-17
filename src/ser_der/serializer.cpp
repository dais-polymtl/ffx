#include "include/serializer.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

namespace ffx {

constexpr size_t CHUNK_SIZE = 500000;

static std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\n\r");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\n\r");
    return str.substr(first, last - first + 1);
}

static ProjectionCardinality parse_projection_cardinality(const std::string& value) {
    std::string v = trim(value);
    std::transform(v.begin(), v.end(), v.begin(), ::tolower);
    if (v == "1:1") return ProjectionCardinality::ONE_TO_ONE;
    if (v == "1:n") return ProjectionCardinality::ONE_TO_MANY;
    if (v == "n:1") return ProjectionCardinality::MANY_TO_ONE;
    if (v == "m:n" || v == "n:n") return ProjectionCardinality::MANY_TO_MANY;
    throw std::runtime_error("Unknown projection cardinality: " + value);
}

static std::string cardinality_to_string(const ProjectionCardinality card) {
    switch (card) {
        case ProjectionCardinality::ONE_TO_ONE:
            return "1:1";
        case ProjectionCardinality::ONE_TO_MANY:
            return "1:n";
        case ProjectionCardinality::MANY_TO_ONE:
            return "n:1";
        case ProjectionCardinality::MANY_TO_MANY:
            return "m:n";
        default:
            return "m:n";
    }
}

static void ser_num_adj_lists(const std::string& ser_dir, uint64_t num_fwd_adj_lists, uint64_t num_bwd_adj_lists,
                              const std::string& src_attr, const std::string& dest_attr);

template<typename T>
static void ser_adj_lists(const std::string& ser_dir, const AdjList<T>* adj_lists, uint64_t num_adj_lists, bool is_fwd,
                          const std::string& src_attr, const std::string& dest_attr);

static void deser_num_adj_lists(const std::string& ser_dir, uint64_t& num_fwd_adj_lists, uint64_t& num_bwd_adj_lists,
                                const std::string& src_attr, const std::string& dest_attr);

template<typename T>
static void deser_adj_lists(const std::string& ser_dir, AdjList<T>* adj_lists, uint64_t num_adj_lists, bool is_fwd,
                            const std::string& src_attr, const std::string& dest_attr, StringPool* pool = nullptr);

ParsedTableConfig parse_column_config(const std::string& config_file) {
    ParsedTableConfig parsed;
    std::ifstream file(config_file);
    if (!file) { throw std::runtime_error("Cannot open config file: " + config_file); }

    std::string line;
    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;

        std::stringstream ss(line);
        std::string kind;
        if (!std::getline(ss, kind, ',')) continue;
        kind = trim(kind);
        std::transform(kind.begin(), kind.end(), kind.begin(), ::toupper);

        if (kind == "COLUMN") {
            std::string attr_name, col_idx_str, type_str;
            if (!std::getline(ss, attr_name, ',')) continue;
            if (!std::getline(ss, col_idx_str, ',')) continue;
            if (!std::getline(ss, type_str, ',')) continue;

            attr_name = trim(attr_name);
            col_idx_str = trim(col_idx_str);
            type_str = trim(type_str);

            size_t col_idx = std::stoull(col_idx_str);
            ColumnType type;
            if (type_str == "uint64_t" || type_str == "uint64" || type_str == "int") {
                type = ColumnType::UINT64;
            } else if (type_str == "string" || type_str == "str") {
                type = ColumnType::STRING;
            } else {
                throw std::runtime_error("Unknown column type: " + type_str);
            }

            parsed.columns.emplace_back(attr_name, col_idx, type);
        } else if (kind == "PROJECTION") {
            std::string relation_name, src_idx_str, dst_idx_str, card_str;
            if (!std::getline(ss, relation_name, ',')) continue;
            if (!std::getline(ss, src_idx_str, ',')) continue;
            if (!std::getline(ss, dst_idx_str, ',')) continue;
            if (!std::getline(ss, card_str, ',')) {
                card_str = "m:n";
            }

            relation_name = trim(relation_name);
            src_idx_str = trim(src_idx_str);
            dst_idx_str = trim(dst_idx_str);
            card_str = trim(card_str);

            const size_t src_idx = std::stoull(src_idx_str);
            const size_t dst_idx = std::stoull(dst_idx_str);
            const auto card = parse_projection_cardinality(card_str.empty() ? "m:n" : card_str);
            parsed.projections.emplace_back(relation_name, src_idx, dst_idx, card);
        } else {
            throw std::runtime_error("Unknown config record kind: " + kind + " in file " + config_file);
        }
    }

    if (parsed.columns.empty()) {
        throw std::runtime_error("No COLUMN entries found in config file: " + config_file);
    }

    if (parsed.projections.empty()) {
        parsed.projections.emplace_back("default", 0, 1, ProjectionCardinality::MANY_TO_MANY);
    }

    return parsed;
}

void serialize_uint64_column(const std::vector<uint64_t>& data, const std::string& output_path,
                             uint64_t* out_max_value) {
    std::ofstream ofs(output_path, std::ios::binary);
    if (!ofs) { throw std::runtime_error("Cannot open file for writing: " + output_path); }

    uint64_t num_rows = data.size();
    ofs.write(reinterpret_cast<const char*>(&num_rows), sizeof(uint64_t));

    uint64_t max_val = 0;
    for (const auto& val: data) {
        ofs.write(reinterpret_cast<const char*>(&val), sizeof(uint64_t));
        if (val > max_val) max_val = val;
    }

    if (out_max_value) *out_max_value = max_val;

    std::cout << "  Written uint64 column: " << output_path << " (" << num_rows << " rows, max=" << max_val << ")"
              << std::endl;
}

void serialize_string_column(const std::vector<std::string>& data, const std::vector<bool>& is_null,
                             const std::string& output_path) {
    std::ofstream ofs(output_path, std::ios::binary);
    if (!ofs) { throw std::runtime_error("Cannot open file for writing: " + output_path); }

    uint64_t num_rows = data.size();
    ofs.write(reinterpret_cast<const char*>(&num_rows), sizeof(uint64_t));

    uint64_t null_count = 0;
    uint64_t truncated_count = 0;
    char prefix_buf[4] = {0, 0, 0, 0};

    for (size_t i = 0; i < data.size(); i++) {
        if (is_null.size() > i && is_null[i]) {
            uint32_t null_size = ffx_str_t::NULL_FLAG;
            ofs.write(reinterpret_cast<const char*>(&null_size), sizeof(uint32_t));
            std::memset(prefix_buf, 0, 4);
            ofs.write(prefix_buf, 4);
            null_count++;
        } else {
            const std::string& str = data[i];
            uint32_t size = static_cast<uint32_t>(std::min(str.size(), static_cast<size_t>(FFX_STR_MAX_LENGTH)));
            if (str.size() > FFX_STR_MAX_LENGTH) { truncated_count++; }

            ofs.write(reinterpret_cast<const char*>(&size), sizeof(uint32_t));

            std::memset(prefix_buf, 0, 4);
            size_t prefix_len = std::min(static_cast<size_t>(4), str.size());
            std::memcpy(prefix_buf, str.data(), prefix_len);
            ofs.write(prefix_buf, 4);

            if (size > 4) { ofs.write(str.data() + 4, size - 4); }
        }
    }

    std::cout << "  Written string column (ffx_str_t): " << output_path << " (" << num_rows << " rows, " << null_count
              << " nulls";
    if (truncated_count > 0) {
        std::cout << ", " << truncated_count << " truncated to " << FFX_STR_MAX_LENGTH << " chars";
    }
    std::cout << ")" << std::endl;
}

std::unique_ptr<uint64_t[]> deserialize_uint64_column(const std::string& input_path, uint64_t& num_rows) {
    std::ifstream ifs(input_path, std::ios::binary);
    if (!ifs) { throw std::runtime_error("Cannot open file for reading: " + input_path); }

    ifs.read(reinterpret_cast<char*>(&num_rows), sizeof(uint64_t));
    auto data = std::make_unique<uint64_t[]>(num_rows);
    ifs.read(reinterpret_cast<char*>(data.get()), num_rows * sizeof(uint64_t));

    return data;
}

std::unique_ptr<ffx_str_t[]> deserialize_string_column(const std::string& input_path, uint64_t& num_rows,
                                                       StringPool* pool) {
    std::ifstream ifs(input_path, std::ios::binary);
    if (!ifs) { throw std::runtime_error("Cannot open file for reading: " + input_path); }

    ifs.read(reinterpret_cast<char*>(&num_rows), sizeof(uint64_t));
    auto data = std::make_unique<ffx_str_t[]>(num_rows);

    for (uint64_t i = 0; i < num_rows; i++) {
        data[i] = ffx_str_t::deserialize(ifs, pool);
    }

    return data;
}

void write_metadata_json(const std::string& output_dir, const std::vector<ColumnConfig>& columns, uint64_t num_rows,
                         const std::vector<uint64_t>& max_values,
                         const std::vector<ProjectionConfig>& projections) {
    std::string path = output_dir + "/table_metadata.json";
    std::ofstream ofs(path);
    if (!ofs) { throw std::runtime_error("Cannot open metadata file: " + path); }

    ofs << "{\n";
    ofs << "  \"num_rows\": " << num_rows << ",\n";
    ofs << "  \"columns\": [\n";

    for (size_t i = 0; i < columns.size(); i++) {
        ofs << "    {\"name\": \"" << columns[i].attr_name << "\", ";
        ofs << "\"type\": \"" << (columns[i].type == ColumnType::UINT64 ? "uint64_t" : "string") << "\"";
        if (columns[i].type == ColumnType::UINT64 && i < max_values.size()) {
            ofs << ", \"max_value\": " << max_values[i];
        }
        ofs << "}";
        if (i < columns.size() - 1) ofs << ",";
        ofs << "\n";
    }

    ofs << "  ],\n";
    ofs << "  \"projections\": [\n";
    for (size_t i = 0; i < projections.size(); i++) {
        ofs << "    {\"name\": \"" << projections[i].relation_name << "\", ";
        ofs << "\"source_column_idx\": " << projections[i].source_column_idx << ", ";
        ofs << "\"target_column_idx\": " << projections[i].target_column_idx << ", ";
        ofs << "\"cardinality\": \"" << cardinality_to_string(projections[i].cardinality) << "\"}";
        if (i < projections.size() - 1) ofs << ",";
        ofs << "\n";
    }
    ofs << "  ]\n";
    ofs << "}\n";

    std::cout << "  Written metadata: " << path << std::endl;
}

void write_metadata_binary(const std::string& output_dir, const std::vector<ColumnConfig>& columns, uint64_t num_rows,
                           const std::vector<uint64_t>& max_values,
                           const std::vector<ProjectionConfig>& projections) {
    std::string path = output_dir + "/table_metadata.bin";
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) { throw std::runtime_error("Cannot open binary metadata file: " + path); }

    ofs.write(reinterpret_cast<const char*>(&num_rows), sizeof(uint64_t));

    uint64_t num_cols = columns.size();
    ofs.write(reinterpret_cast<const char*>(&num_cols), sizeof(uint64_t));

    for (size_t i = 0; i < columns.size(); i++) {
        uint64_t name_len = columns[i].attr_name.size();
        ofs.write(reinterpret_cast<const char*>(&name_len), sizeof(uint64_t));
        ofs.write(columns[i].attr_name.data(), name_len);

        uint8_t type_byte = (columns[i].type == ColumnType::UINT64) ? 0 : 1;
        ofs.write(reinterpret_cast<const char*>(&type_byte), sizeof(uint8_t));

        uint64_t col_idx = columns[i].column_idx;
        ofs.write(reinterpret_cast<const char*>(&col_idx), sizeof(uint64_t));

        uint64_t max_val = (i < max_values.size()) ? max_values[i] : 0;
        ofs.write(reinterpret_cast<const char*>(&max_val), sizeof(uint64_t));
    }

    uint64_t num_projections = projections.size();
    ofs.write(reinterpret_cast<const char*>(&num_projections), sizeof(uint64_t));
    for (const auto& projection: projections) {
        uint64_t name_len = projection.relation_name.size();
        ofs.write(reinterpret_cast<const char*>(&name_len), sizeof(uint64_t));
        ofs.write(projection.relation_name.data(), name_len);

        uint64_t src_idx = projection.source_column_idx;
        uint64_t dst_idx = projection.target_column_idx;
        ofs.write(reinterpret_cast<const char*>(&src_idx), sizeof(uint64_t));
        ofs.write(reinterpret_cast<const char*>(&dst_idx), sizeof(uint64_t));

        uint8_t card_byte = 3;
        switch (projection.cardinality) {
            case ProjectionCardinality::ONE_TO_ONE:
                card_byte = 0;
                break;
            case ProjectionCardinality::ONE_TO_MANY:
                card_byte = 1;
                break;
            case ProjectionCardinality::MANY_TO_ONE:
                card_byte = 2;
                break;
            case ProjectionCardinality::MANY_TO_MANY:
                card_byte = 3;
                break;
        }
        ofs.write(reinterpret_cast<const char*>(&card_byte), sizeof(uint8_t));
    }

    std::cout << "  Written binary metadata: " << path << std::endl;
}

void serialize(const Table& table, const std::string& ser_dir, const std::string& src_attr, const std::string& dest_attr,
               const StringDictionary* dict) {
    if (ser_dir.empty()) {
        std::cerr << "Failed to read path for serializing the dataset." << std::endl;
        return;
    }
    std::cout << "Starting serialization to " << ser_dir << std::endl;
    std::cout << "Attribute names: " << src_attr << " -> " << dest_attr << std::endl;

    std::cout << "Serializing the number of adjacency lists..." << std::endl;
    ser_num_adj_lists(ser_dir, table.num_fwd_ids, table.num_bwd_ids, src_attr, dest_attr);

    std::cout << "Serializing forward adjacency lists..." << std::endl;
    ser_adj_lists<uint64_t>(ser_dir, table.fwd_adj_lists, table.num_fwd_ids, true, src_attr, dest_attr);

    std::cout << "Serializing backward adjacency lists..." << std::endl;
    ser_adj_lists<uint64_t>(ser_dir, table.bwd_adj_lists, table.num_bwd_ids, false, src_attr, dest_attr);

    if (dict && dict->size() > 0) {
        const std::string dict_filename = ser_dir + "/string_dictionary_" + src_attr + "_" + dest_attr + ".bin";
        std::ofstream dict_out(dict_filename, std::ios::binary);
        if (!dict_out) {
            std::cerr << "Error: Unable to open file for writing dictionary: " << dict_filename << std::endl;
            throw std::runtime_error("Cannot open dictionary file: " + dict_filename);
        }
        dict->serialize(dict_out);
        dict_out.close();
        std::cout << "  Written dictionary: " << dict_filename << std::endl;
    }
}

std::unique_ptr<Table> deserialize(const std::string& ser_dir, const std::string& src_attr, const std::string& dest_attr,
                                   StringDictionary* dict) {
    if (ser_dir.empty()) {
        std::cerr << "Failed to read path for deserializing the dataset" << std::endl;
        throw std::runtime_error("Empty serialization directory");
    }

    uint64_t num_fwd_adj_lists, num_bwd_adj_lists;
    deser_num_adj_lists(ser_dir, num_fwd_adj_lists, num_bwd_adj_lists, src_attr, dest_attr);

    auto fwd_adj_lists = std::make_unique<AdjList<uint64_t>[]>(num_fwd_adj_lists);
    auto bwd_adj_lists = std::make_unique<AdjList<uint64_t>[]>(num_bwd_adj_lists);

    deser_adj_lists<uint64_t>(ser_dir, fwd_adj_lists.get(), num_fwd_adj_lists, true, src_attr, dest_attr);
    deser_adj_lists<uint64_t>(ser_dir, bwd_adj_lists.get(), num_bwd_adj_lists, false, src_attr, dest_attr);

    if (dict) {
        const std::string dict_filename = ser_dir + "/string_dictionary_" + src_attr + "_" + dest_attr + ".bin";
        std::ifstream dict_in(dict_filename, std::ios::binary);
        if (dict_in.is_open()) {
            dict->deserialize(dict_in, nullptr);
            dict_in.close();
            std::cout << "Loaded dictionary from: " << dict_filename << std::endl;
        }
    }

    return std::make_unique<Table>(num_fwd_adj_lists, num_bwd_adj_lists, std::move(fwd_adj_lists), std::move(bwd_adj_lists));
}

static void ser_num_adj_lists(const std::string& ser_dir, uint64_t num_fwd_adj_lists, uint64_t num_bwd_adj_lists,
                              const std::string& src_attr, const std::string& dest_attr) {
    std::string filename = ser_dir + "/num_adj_lists_" + src_attr + "_" + dest_attr + ".bin";
    std::ofstream out_file(filename, std::ios::binary);
    if (!out_file) {
        std::cerr << "Error: Unable to open file for writing: " << filename << std::endl;
        throw std::runtime_error("Cannot open file: " + filename);
    }
    out_file.write(reinterpret_cast<const char*>(&num_fwd_adj_lists), sizeof(num_fwd_adj_lists));
    out_file.write(reinterpret_cast<const char*>(&num_bwd_adj_lists), sizeof(num_bwd_adj_lists));
    out_file.close();
    std::cout << "  Written: " << filename << std::endl;
}

static void deser_num_adj_lists(const std::string& ser_dir, uint64_t& num_fwd_adj_lists, uint64_t& num_bwd_adj_lists,
                                const std::string& src_attr, const std::string& dest_attr) {
    std::string filename = ser_dir + "/num_adj_lists_" + src_attr + "_" + dest_attr + ".bin";
    std::ifstream in_file(filename, std::ios::binary);

    if (!in_file) {
        filename = ser_dir + "/num_adj_lists.bin";
        in_file.open(filename, std::ios::binary);
        if (!in_file) {
            std::cerr << "Error: Unable to open file for reading: " << filename << std::endl;
            throw std::runtime_error("Cannot open file: " + filename);
        }
    }

    in_file.read(reinterpret_cast<char*>(&num_fwd_adj_lists), sizeof(num_fwd_adj_lists));
    in_file.read(reinterpret_cast<char*>(&num_bwd_adj_lists), sizeof(num_bwd_adj_lists));
    in_file.close();
}

template<typename T>
void ser_chunk(const AdjList<T>* adj_lists, size_t start_row, size_t end_row, const std::string& filename);

template<typename T>
void ser_worker(const AdjList<T>* adj_lists, uint64_t num_adj_lists, const std::string& ser_dir,
                std::atomic<size_t>& next_start_idx, bool is_fwd, const std::string& src_attr, const std::string& dest_attr) {
    while (true) {
        const size_t start_idx = next_start_idx.fetch_add(CHUNK_SIZE);
        if (start_idx >= num_adj_lists) {
            break;
        }

        const size_t end_idx = std::min(start_idx + CHUNK_SIZE, static_cast<size_t>(num_adj_lists));

        std::string base = ser_dir + "/" + (is_fwd ? "fwd" : "bwd") + "_" + src_attr + "_" + dest_attr + "_" +
                           std::to_string(start_idx) + "_" + std::to_string(end_idx);

        std::string filename;
        if constexpr (std::is_same_v<T, ffx_str_t>) {
            filename = base + ".str.bin";
        } else {
            filename = base + ".bin";
        }

        ser_chunk<T>(adj_lists, start_idx, end_idx, filename);
    }
}

template<typename T>
static void ser_adj_lists(const std::string& ser_dir, const AdjList<T>* adj_lists, uint64_t num_adj_lists, bool is_fwd,
                          const std::string& src_attr, const std::string& dest_attr) {
    size_t num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    std::atomic<size_t> next_start_idx(0);

    for (size_t i = 0; i < num_threads; ++i) {
        threads.emplace_back(ser_worker<T>, std::ref(adj_lists), num_adj_lists, std::ref(ser_dir), std::ref(next_start_idx),
                             is_fwd, std::ref(src_attr), std::ref(dest_attr));
    }

    for (auto& t: threads) {
        t.join();
    }
}

template<typename T>
void ser_chunk(const AdjList<T>* adj_lists, const size_t start_row, const size_t end_row, const std::string& filename) {
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    for (size_t i = start_row; i < end_row; ++i) {
        outFile.write(reinterpret_cast<const char*>(&adj_lists[i].size), sizeof(size_t));
        for (size_t j = 0; j < adj_lists[i].size; ++j) {
            serialize_value(adj_lists[i].values[j], outFile);
        }
    }

    outFile.close();
}

template<typename T>
void deser_chunk(AdjList<T>* adj_lists, size_t start_id, size_t end_id, const std::string& filename, StringPool* pool = nullptr);

template<typename T>
void deser_worker(AdjList<T>* adj_lists, uint64_t num_adj_lists, const std::string& ser_dir,
                  std::atomic<size_t>& next_start_idx, bool is_fwd, const std::string& src_attr,
                  const std::string& dest_attr, StringPool* pool = nullptr) {
    while (true) {
        const size_t start_idx = next_start_idx.fetch_add(CHUNK_SIZE);
        if (start_idx >= num_adj_lists) {
            break;
        }

        const size_t end_idx = std::min(start_idx + CHUNK_SIZE, static_cast<size_t>(num_adj_lists));

        std::string base = ser_dir + "/" + (is_fwd ? "fwd" : "bwd") + "_" + src_attr + "_" + dest_attr + "_" +
                           std::to_string(start_idx) + "_" + std::to_string(end_idx);

        std::string filename;
        if constexpr (std::is_same_v<T, ffx_str_t>) {
            filename = base + ".str.bin";
        } else {
            filename = base + ".bin";
        }

        std::ifstream test_file(filename, std::ios::binary);
        if (!test_file) {
            base = ser_dir + "/" + (is_fwd ? "fwd" : "bwd") + "_" + std::to_string(start_idx) + "_" + std::to_string(end_idx);
            if constexpr (std::is_same_v<T, ffx_str_t>) {
                filename = base + ".str.bin";
            } else {
                filename = base + ".bin";
            }
        } else {
            test_file.close();
        }

        deser_chunk<T>(adj_lists, start_idx, end_idx, filename, pool);
    }
}

template<typename T>
static void deser_adj_lists(const std::string& ser_dir, AdjList<T>* adj_lists, uint64_t num_adj_lists, bool is_fwd,
                            const std::string& src_attr, const std::string& dest_attr, StringPool* pool) {
    size_t num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    std::atomic<size_t> next_start_idx(0);

    for (size_t i = 0; i < num_threads; ++i) {
        threads.emplace_back(deser_worker<T>, std::ref(adj_lists), num_adj_lists, std::ref(ser_dir), std::ref(next_start_idx),
                             is_fwd, std::ref(src_attr), std::ref(dest_attr), pool);
    }

    for (auto& t: threads) {
        t.join();
    }
}

template<typename T>
void deser_chunk(AdjList<T>* adj_lists, const size_t start_id, const size_t end_id, const std::string& filename, StringPool* pool) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) {
        std::cerr << "Error opening file for reading: " << filename << std::endl;
        return;
    }

    for (size_t i = start_id; i < end_id; ++i) {
        size_t list_size;
        ifs.read(reinterpret_cast<char*>(&list_size), sizeof(std::size_t));
        AdjList<T> new_list(list_size);
        for (size_t j = 0; j < list_size; ++j) {
            deserialize_value(new_list.values[j], ifs, pool);
        }
        adj_lists[i] = std::move(new_list);
    }

    ifs.close();
}

template void ser_adj_lists<uint64_t>(const std::string&, const AdjList<uint64_t>*, uint64_t, bool, const std::string&,
                                      const std::string&);
template void deser_adj_lists<uint64_t>(const std::string&, AdjList<uint64_t>*, uint64_t, bool, const std::string&,
                                        const std::string&, StringPool*);

template void ser_adj_lists<ffx_str_t>(const std::string&, const AdjList<ffx_str_t>*, uint64_t, bool, const std::string&,
                                       const std::string&);
template void deser_adj_lists<ffx_str_t>(const std::string&, AdjList<ffx_str_t>*, uint64_t, bool, const std::string&,
                                         const std::string&, StringPool*);

bool is_string_type(const std::string& ser_dir) {
    std::ifstream test(ser_dir + "/num_adj_lists_str.bin");
    return test.good();
}

} // namespace ffx
