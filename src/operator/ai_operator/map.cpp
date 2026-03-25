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
    _batch_iterator->reset();
    _batch_iterator->initialize_iterators();

    if (!_batch_iterator->is_valid()) { return; }

    while (_batch_iterator->next()) {
        size_t num_tuples = _batch_iterator->count_logical_tuples();
        if (num_tuples == 0) { continue; }

        // Serialize the current batch into a tabular representation.
        nlohmann::json columns =
                _serializer->build_columns(*_batch_iterator, _schema, _required_attributes, num_tuples);

        if (_debug) {
            std::cout << "[AI-Map][debug] tuple_format=" << _tuple_format
                      << " num_tuples=" << num_tuples << "\n";
            std::cout << "[AI-Map][debug] columns JSON:\n" << columns.dump(2) << "\n";
        }

        // Construct the final prompt for the model.
        std::string user_prompt = _prompt;
        auto [prompt, media_data] =
                flock::PromptManager::Render(user_prompt, columns, flock::ScalarFunctionType::COMPLETE, _tuple_format);

        if (_debug) {
            std::cout << "[AI-Map][debug] Final prompt:\n" << prompt << "\n---\n";
        }

        // Execute the model for this batch.
        size_t num_requested_outputs = num_tuples;
        if (_llm_buffer_capacity < num_requested_outputs) {
            _llm_buffer = std::make_unique<uint64_t[]>(num_requested_outputs);
            _llm_buffer_capacity = num_requested_outputs;
        }
        std::fill_n(_llm_buffer.get(), num_requested_outputs, 0u);

        _model->AddCompletionRequest(prompt, static_cast<int>(num_requested_outputs), flock::OutputType::STRING,
                                     media_data);
        auto results = _model->CollectCompletions("application/json")[0]["items"];

        if (_debug) {
            std::cout << "[AI-Map][debug] Raw model responses (" << results.size() << "):";
            for (size_t i = 0; i < results.size(); ++i) {
                std::string r_str = results[i].is_string() ? results[i].get<std::string>() : results[i].dump();
                std::cout << " [" << i << "]=\"" << r_str << "\"";
            }
            std::cout << "\n";
        }

        for (size_t i = 0; i < num_requested_outputs; ++i) {
            _llm_buffer[i] = encode_to_dict_id(results[i].is_string() ? results[i].get<std::string>() : results[i].dump());
        }

        // Append results to reconstructor (handles tree reconstruction + flushing).
        LLMResultBatch batch{_llm_buffer.get(), num_requested_outputs};
        _num_output_tuples += batch.count;
        _reconstructor->append(batch);
    }
}

}// namespace ffx
