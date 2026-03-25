// Base interface for AI operators (Map, Reduce, Rerank, ...)
// Provides a common lifecycle around environment configuration and model setup.
#ifndef FFX_AI_OPERATOR_BASE_HH
#define FFX_AI_OPERATOR_BASE_HH

#include "operator.hpp"

#include <nlohmann/json.hpp>

#include <memory>
#include <string>

#include <flock/model_manager/model.hpp>

#include "ai_operator/ai_serializer.hpp"
#include "ai_operator/ftree_reconstructor.hpp"

#include "factorized_ftree/ftree_batch_iterator.hpp"

namespace ffx {

struct Schema;

// Abstract base class for all AI operators.
// It is responsible for:
//  - Reading configuration from the environment
//  - Initializing the LLM model manager
//  - Providing helpers for derived operators to build prompts and execute the model
class AIOperator : public Operator {
public:
    AIOperator();
    ~AIOperator() override = default;

    // Final overrides for the standard operator lifecycle.
    void init(Schema* schema) final;
    void execute() final;

protected:
    Schema* _schema;

    bool _debug{false};
    uint64_t _num_output_tuples;

    // Raw JSON configuration for the model / prompts, typically sourced from FFX_LLM_CONFIG.
    nlohmann::json _model_config_json;
    nlohmann::json _context_columns;
    std::vector<std::string> _required_attributes;

    // Cached common fields from _model_config_json.
    std::string _prompt;
    std::string _tuple_format;
    std::unique_ptr<flock::Model> _model;
    enum class SerializationMode { Flat, Tree };
    SerializationMode _mode;

    size_t _llm_buffer_capacity;
    std::unique_ptr<uint64_t[]> _llm_buffer;

    std::unique_ptr<FTreeBatchIterator> _batch_iterator;
    std::unique_ptr<FTreeReconstructor> _reconstructor;
    std::unique_ptr<AISerializer> _serializer;
    std::unique_ptr<StringPool> _llm_pool;
    std::unique_ptr<StringDictionary> _llm_dict;

    // Hook for subclasses to perform additional initialization after the base
    // configuration and model objects are ready.
    virtual void init_internal() = 0;

    // Hook for subclasses to execute their operator-specific logic.
    virtual void execute_internal() = 0;

    // Load LLM configuration.  Reads from Schema::llm_config_str (set by the
    // LLM_MAP query clause).  Subclasses may override to add sources.
    virtual void load_config();

    // Helper to (re)create the underlying model from _model_config_json.
    void rebuild_model();

    // Intern a model output string into the LLM dictionary and return 1-based id.
    uint64_t encode_to_dict_id(const std::string& s);
};

} // namespace ffx

#endif // FFX_AI_OPERATOR_BASE_HH

