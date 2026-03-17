// FFX: stub implementation - read secrets from environment.
#include "flock/secret_manager/secret_manager.hpp"
#include <cstdlib>
#include <unordered_map>

namespace flock {

std::unordered_map<std::string, std::string> SecretManager::GetSecret(const std::string& secret_name) {
    std::unordered_map<std::string, std::string> out;
    // Map common secret names to env vars
    if (secret_name.find("openai") != std::string::npos) {
        if (const char* key = std::getenv("OPENAI_API_KEY")) out["api_key"] = key;
        if (const char* base = std::getenv("OPENAI_BASE_URL")) out["base_url"] = base;
    } else if (secret_name.find("anthropic") != std::string::npos) {
        if (const char* key = std::getenv("ANTHROPIC_API_KEY")) out["api_key"] = key;
    } else if (secret_name.find("ollama") != std::string::npos) {
        if (const char* url = std::getenv("OLLAMA_API_URL")) out["api_url"] = url;
        else out["api_url"] = "http://localhost:11434";
    } else if (secret_name.find("azure") != std::string::npos) {
        if (const char* key = std::getenv("AZURE_API_KEY")) out["api_key"] = key;
        if (const char* r = std::getenv("AZURE_RESOURCE_NAME")) out["resource_name"] = r;
        if (const char* v = std::getenv("AZURE_API_VERSION")) out["api_version"] = v;
    }
    return out;
}

}// namespace flock
