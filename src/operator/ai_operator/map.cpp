#include "ai_operator/map.hpp"
#include "ai_operator/ai_base.hpp"
#include "ai_operator/ai_serializer.hpp"
#include "ai_operator/ftree_reconstructor.hpp"

#include "factorized_ftree/ftree_batch_iterator.hpp"
#include "schema/schema.hpp"
#include "string_dictionary.hpp"
#include "string_pool.hpp"

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <flock/prompt_manager/prompt_manager.hpp>

namespace ffx {

Map::Map() = default;

void Map::init_internal() {
    // Base class handles all common init (batch iterator, reconstructor, serializer, model).
    // Map-specific post-init can go here if needed.
}

void Map::execute_internal() {
    auto encode_to_dict_id = [this](const std::string& s) -> uint64_t {
        ffx_str_t key(s, _llm_pool.get());
        uint64_t id = _llm_dict->get_id(key);
        if (id == UINT64_MAX) { id = _llm_dict->add_string(key); }
        return id + 1;
    };

    _batch_iterator->reset();
    _batch_iterator->initialize_iterators();

    if (!_batch_iterator->is_valid()) { return; }

    while (_batch_iterator->next()) {
        size_t num_logical_tuples = _batch_iterator->count_logical_tuples();
        if (num_logical_tuples == 0) { continue; }

        // How many model outputs to request for this batch.
        size_t num_requested_outputs = num_logical_tuples;
        LLMResultBatch::Granularity granularity = LLMResultBatch::Granularity::PER_TUPLE;
        if (_tuple_format == "FTREE") {
            const std::string per = _model_config_json.value("llm_per", std::string("tuple"));
            if (per == "root") {
                num_requested_outputs = _batch_iterator->get_count(0);
                granularity = LLMResultBatch::Granularity::PER_ROOT;
            }
        }
        if (num_requested_outputs == 0) { continue; }

        // Serialize the current batch into a tabular representation.
        nlohmann::json columns =
                _serializer->build_columns(*_batch_iterator, _schema, _required_attributes, num_logical_tuples);

        if (_debug) {
            std::cout << "[AI-Map][debug] tuple_format=" << _tuple_format << " logical_tuples=" << num_logical_tuples
                      << "\n";
            std::cout << "[AI-Map][debug] columns JSON:\n" << columns.dump(2) << "\n";
        }

        // Construct the final prompt for the model.
        std::string user_prompt = _prompt;
        auto [prompt, media_data] =
                flock::PromptManager::Render(user_prompt, columns, flock::ScalarFunctionType::COMPLETE, _tuple_format);

        if (_debug) { std::cout << "[AI-Map][debug] Final prompt:\n" << prompt << "\n---\n"; }

        // Execute the model for this batch.
        if (_llm_buffer_capacity < num_requested_outputs) {
            _llm_buffer = std::make_unique<uint64_t[]>(num_requested_outputs);
            _llm_buffer_capacity = num_requested_outputs;
        }
        std::fill_n(_llm_buffer.get(), num_requested_outputs, 0u);

        _model->AddCompletionRequest(prompt, static_cast<int>(num_requested_outputs), flock::OutputType::STRING,
                                     media_data);
        auto results = _model->CollectCompletions("application/json");

        if (_debug) {
            std::cout << "[AI-Map][debug] Raw model responses (" << results.size() << "):";
            for (size_t i = 0; i < results.size(); ++i) {
                std::string r_str = results[i].is_string() ? results[i].get<std::string>() : results[i].dump();
                std::cout << " [" << i << "]=\"" << r_str << "\"";
            }
            std::cout << "\n";
        }

        if (!results.empty()) {
            bool parsed = false;
            if (results.size() == 1) {
                const auto& first = results.front();
                if (first.is_string() || first.is_object()) {
                    nlohmann::json j =
                            first.is_string() ? nlohmann::json::parse(first.get<std::string>(), nullptr, false) : first;
                    if (!j.is_discarded() && j.contains("items") && j["items"].is_array()) {
                        const auto& arr = j["items"];
                        const size_t take = std::min(arr.size(), num_requested_outputs);
                        for (size_t i = 0; i < take; ++i) {
                            std::string s = arr[i].is_string() ? arr[i].get<std::string>() : arr[i].dump();
                            _llm_buffer[i] = encode_to_dict_id(s);
                        }
                        parsed = true;
                    }
                }
            }
            if (!parsed) {
                const size_t take = std::min(results.size(), num_requested_outputs);
                for (size_t i = 0; i < take; ++i) {
                    std::string s = results[i].is_string() ? results[i].get<std::string>() : results[i].dump();
                    _llm_buffer[i] = encode_to_dict_id(s);
                }
            }
        }

        // Append results to reconstructor (handles tree reconstruction + flushing).
        LLMResultBatch batch{_llm_buffer.get(), num_requested_outputs, granularity};
        _num_output_tuples += batch.count;
        _reconstructor->append(batch);
    }
}

}// namespace ffx
