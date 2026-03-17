#include "flock/model_manager/providers/adapters/demo.hpp"

namespace flock {

void DemoProvider::AddCompletionRequest(const std::string& /*prompt*/,
                                        const int num_output_tuples,
                                        OutputType /*output_type*/,
                                        const nlohmann::json& /*media_data*/) {
    // For demo purposes, we ignore the prompt and media data and just enqueue
    // deterministic string results: "Tuple <idx>".
    const int count = num_output_tuples > 0 ? num_output_tuples : 1;
    completion_results_.push_back({{"items", {}}});
    for (int i = 0; i < count; ++i) {
        completion_results_[0]["items"].push_back(
                nlohmann::json("Tuple " + std::to_string(next_index_++)));
    }
}

void DemoProvider::AddEmbeddingRequest(const std::vector<std::string>& /*inputs*/) {
    // No-op: demo provider does not generate embeddings.
}

void DemoProvider::AddTranscriptionRequest(const nlohmann::json& /*audio_files*/) {
    // No-op: demo provider does not generate transcriptions.
}

std::vector<nlohmann::json> DemoProvider::CollectCompletions(const std::string& /*contentType*/) {
    auto out = completion_results_;
    completion_results_.clear();
    return out;
}

std::vector<nlohmann::json> DemoProvider::CollectEmbeddings(const std::string& /*contentType*/) {
    auto out = embedding_results_;
    embedding_results_.clear();
    return out;
}

std::vector<nlohmann::json> DemoProvider::CollectTranscriptions(const std::string& /*contentType*/) {
    auto out = transcription_results_;
    transcription_results_.clear();
    return out;
}

}// namespace flock

