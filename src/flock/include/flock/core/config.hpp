#pragma once

// FFX: stub config (no DuckDB). Model must be constructed with full resolved model_json.
#include "flock/core/common.hpp"
#include <filesystem>
#include <string>

namespace flock {

enum ConfigType { LOCAL, GLOBAL };

class Config {
public:
    static std::string get_schema_name() { return "flock_config"; }
    static std::filesystem::path get_global_storage_path() { return std::filesystem::temp_directory_path(); }
    static std::string get_default_models_table_name() { return ""; }
    static std::string get_user_defined_models_table_name() { return ""; }
    static std::string get_prompts_table_name() { return ""; }
};

}// namespace flock
