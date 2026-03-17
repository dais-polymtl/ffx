#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>

#include "ser_der/include/storage_api.hpp"

void print_usage(const char* prog_name) {
    std::cerr << "Usage: " << prog_name << " <input_csv> <output_dir> <config_file>" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Config file format:" << std::endl;
    std::cerr << "  COLUMN,<attribute_name>,<column_idx>,<type>" << std::endl;
    std::cerr << "  PROJECTION,<relation_name>,<source_column_idx>,<target_column_idx>,<cardinality>" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Types: uint64_t (or uint64, int), string (or str)" << std::endl;
    std::cerr << "Cardinality: 1:1, 1:n, n:1, m:n" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Output files:" << std::endl;
    std::cerr << "  <attribute_name>_uint64.bin   - for uint64_t columns" << std::endl;
    std::cerr << "  <attribute_name>_string.bin   - for string columns" << std::endl;
    std::cerr << "  table_metadata.json           - human-readable metadata" << std::endl;
    std::cerr << "  table_metadata.bin            - binary metadata" << std::endl;
}

static std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\n\r");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\n\r");
    return str.substr(first, last - first + 1);
}

static std::vector<std::string> parse_csv_line_quoted(const std::string& line, char delimiter) {
    std::vector<std::string> fields;
    std::string cur;
    cur.reserve(line.size());
    bool in_quotes = false;
    bool at_field_start = true;

    for (size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (c == '"') {
            if (in_quotes) {
                if (i + 1 < line.size() && line[i + 1] == '"') {
                    // Escaped quote inside quoted field.
                    cur.push_back('"');
                    ++i;
                } else if (i + 1 == line.size() || line[i + 1] == delimiter) {
                    // Closing quote at field boundary.
                    in_quotes = false;
                } else {
                    // Stray quote inside quoted text: keep it literally.
                    cur.push_back('"');
                }
            } else if (at_field_start) {
                // Opening quote for a quoted field.
                in_quotes = true;
            } else {
                // Stray quote in unquoted text: keep it literally.
                cur.push_back('"');
            }
            continue;
        }
        if (in_quotes && c == '\\' && i + 1 < line.size()) {
            // Handle backslash escapes used by some JOB rows.
            cur.push_back(line[i + 1]);
            ++i;
            at_field_start = false;
            continue;
        }
        if (c == delimiter && !in_quotes) {
            fields.push_back(trim(cur));
            cur.clear();
            at_field_start = true;
            continue;
        }
        cur.push_back(c);
        at_field_start = false;
    }

    fields.push_back(trim(cur));
    if (!line.empty() && line.back() == delimiter) { fields.push_back(""); }
    return fields;
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        print_usage(argv[0]);
        return 1;
    }

    const std::string input_csv = argv[1];
    const std::string output_dir = argv[2];
    const std::string config_file = argv[3];

    // Parse config
    auto t1 = std::chrono::steady_clock::now();
    auto parsed = ffx::parse_column_config(config_file);
    const auto& configs = parsed.columns;

    // Find max column index needed
    size_t max_col_idx = 0;
    for (const auto& cfg : configs) {
        max_col_idx = std::max(max_col_idx, cfg.column_idx);
    }

    // Read CSV and collect column data
    std::vector<std::vector<uint64_t>> uint64_columns(configs.size());
    std::vector<std::vector<std::string>> string_columns(configs.size());
    std::vector<std::vector<bool>> string_nulls(configs.size());

    std::ifstream csv_file(input_csv);
    if (!csv_file) {
        std::cerr << "Error: Cannot open CSV file: " << input_csv << std::endl;
        return 1;
    }

    std::string line;
    uint64_t row_count = 0;
    
    uint64_t skipped_rows = 0;
    uint64_t uint64_parse_failures = 0;
    while (std::getline(csv_file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        // Parse CSV line (comma, pipe, or tab separated) with quote-awareness.
        char delimiter = '|';
        if (line.find(',') != std::string::npos) {
            delimiter = ',';
        } else if (line.find('\t') != std::string::npos) {
            delimiter = '\t';
        }
        std::vector<std::string> fields = parse_csv_line_quoted(line, delimiter);
        
        if (fields.size() <= max_col_idx) {
            skipped_rows++;
            continue;
        }
        
        // Extract values for each configured column
        for (size_t i = 0; i < configs.size(); i++) {
            const auto& cfg = configs[i];
            const std::string& value = fields[cfg.column_idx];
            
            if (cfg.type == ffx::ColumnType::UINT64) {
                try {
                    uint64_columns[i].push_back(std::stoull(value));
                } catch (...) {
                    uint64_parse_failures++;
                    uint64_columns[i].push_back(0);
                }
            } else {
                // String column
                bool is_null = value.empty() || value == "NULL" || value == "null";
                string_columns[i].push_back(is_null ? "" : value);
                string_nulls[i].push_back(is_null);
            }
        }
        
        row_count++;
    }
    
    auto t2 = std::chrono::steady_clock::now();
    std::cout << "Serialized " << row_count << " rows from " << input_csv
              << " in " << std::chrono::duration<double>(t2 - t1).count() << "s";
    if (skipped_rows > 0 || uint64_parse_failures > 0) {
        std::cout << " (skipped_rows=" << skipped_rows
                  << ", uint64_parse_fallbacks=" << uint64_parse_failures << ")";
    }
    std::cout << std::endl;

    // Serialize each column
    std::vector<uint64_t> max_values(configs.size(), 0);
    
    for (size_t i = 0; i < configs.size(); i++) {
        const auto& cfg = configs[i];
        std::string type_suffix = (cfg.type == ffx::ColumnType::UINT64) ? "_uint64.bin" : "_string.bin";
        std::string output_path = output_dir + "/" + cfg.attr_name + type_suffix;
        
        if (cfg.type == ffx::ColumnType::UINT64) {
            ffx::serialize_uint64_column(uint64_columns[i], output_path, &max_values[i]);
        } else {
            ffx::serialize_string_column(string_columns[i], string_nulls[i], output_path);
        }
    }

    // Write metadata files
    ffx::write_metadata_json(output_dir, configs, row_count, max_values, parsed.projections);
    ffx::write_metadata_binary(output_dir, configs, row_count, max_values, parsed.projections);

    auto t3 = std::chrono::steady_clock::now();
    std::cout << "Done: " << output_dir << " (" 
              << std::chrono::duration<double>(t3 - t1).count() << "s)" << std::endl;

    return 0;
}
