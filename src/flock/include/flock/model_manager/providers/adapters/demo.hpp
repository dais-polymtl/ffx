#pragma once

#include <vector>
#include <string>

#include "flock/model_manager/providers/provider.hpp"

namespace flock {

// Demo provider that performs no external API calls.
// For each completion request it returns a deterministic string "Tuple <idx>".
class DemoProvider : public IProvider {
public:
    explicit DemoProvider(const ModelDetails& model_details)
        : IProvider(model_details), next_index_(0) {}

    void AddCompletionRequest(const std::string& prompt,
                              const int num_output_tuples,
                              OutputType output_type,
                              const nlohmann::json& media_data) override;

    void AddEmbeddingRequest(const std::vector<std::string>& inputs) override;
    void AddTranscriptionRequest(const nlohmann::json& audio_files) override;

    std::vector<nlohmann::json> CollectCompletions(const std::string& contentType = "application/json") override;
    std::vector<nlohmann::json> CollectEmbeddings(const std::string& contentType = "application/json") override;
    std::vector<nlohmann::json> CollectTranscriptions(const std::string& contentType = "multipart/form-data") override;

private:
    std::vector<nlohmann::json> completion_results_;
    std::vector<nlohmann::json> embedding_results_;
    std::vector<nlohmann::json> transcription_results_;
    std::int64_t next_index_;
};

}// namespace flock

