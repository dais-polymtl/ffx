#pragma once

// FFX: stub secret manager - reads from environment (e.g. OPENAI_API_KEY).
#include "flock/core/common.hpp"
#include <string>
#include <unordered_map>

namespace flock {

class SecretManager {
public:
    // Returns secret key-value map from environment. E.g. secret_name "__default_openai" -> {"api_key": getenv("OPENAI_API_KEY")}.
    static std::unordered_map<std::string, std::string> GetSecret(const std::string& secret_name);
};

}// namespace flock
