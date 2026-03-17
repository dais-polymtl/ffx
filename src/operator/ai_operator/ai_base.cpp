#include "ai_operator/ai_base.hpp"

#include "factorized_ftree/factorized_tree_element.hpp"
#include "schema/schema.hpp"
#include "vector/state.hpp"

#include <algorithm>
#include <unordered_set>
#include <vector>

namespace ffx {
AIOperator::AIOperator()
    : _schema(nullptr), _model_config_json(nlohmann::json::object()), _tuple_format("JSON"),
      _model(nullptr), _num_output_tuples(0), _llm_buffer(nullptr), _llm_buffer_capacity(0) {}

void AIOperator::load_config() {
    _model_config_json = nlohmann::json::object();
    _debug = false;

    // Primary source: Schema::llm_config_str (populated from LLM_MAP clause).
    if (_schema && _schema->llm_config_str && !_schema->llm_config_str->empty()) {
        try {
            _model_config_json = nlohmann::json::parse(*_schema->llm_config_str);
        } catch (const std::exception& ex) {
            throw std::runtime_error(std::string("AIOperator: failed to parse the config: ") + ex.what());
        }
    }

    // Fill sensible defaults if missing.
    if (!_model_config_json.contains("model_name")) { _model_config_json["model_name"] = "demo_model"; }
    if (!_model_config_json.contains("model")) { _model_config_json["model"] = "demo"; }
    if (!_model_config_json.contains("provider")) { _model_config_json["provider"] = "demo"; }

    // Normalize tuple_format to upper case and validate against the supported set.
    _tuple_format = _model_config_json.value("tuple_format", std::string("JSON"));
    std::transform(_tuple_format.begin(), _tuple_format.end(), _tuple_format.begin(), ::toupper);

    if (_tuple_format != "JSON" && _tuple_format != "XML" && _tuple_format != "MARKDOWN" &&
        _tuple_format != "COLUMN_ENCODED" && _tuple_format != "FTREE") {
        throw std::runtime_error("AIOperator: unsupported tuple_format '" + _tuple_format +
                                 "'. Supported formats are JSON, XML, MARKDOWN, COLUMN_ENCODED, FTREE.");
        }

    if (!_model_config_json.contains("prompt")) {
        throw std::runtime_error(
            "AIOperator: Config is missing required key 'prompt'.");
    }

    if (_tuple_format == "FTREE" && _model_config_json.contains("batch_size")) {
        throw std::runtime_error(
            "AIOperator: 'batch_size' is not supported with tuple_format='FTREE'. "
            "FTREE batching is determined by the factorized iterator.");
    }

    if (!_model_config_json.contains("context_column")) {
        throw std::runtime_error("AIOperator: Config is missing required key 'context_column'.");
    }

    if (_tuple_format == "FTREE") {
        if (!_model_config_json["context_column"].is_object()) {
            throw std::runtime_error(
                "AIOperator: for tuple_format='FTREE', 'context_column' must be an object "
                "(attribute -> capacity).");
        }
    } else {
        if (!_model_config_json["context_column"].is_array()) {
            throw std::runtime_error(
                "AIOperator: for flat tuple formats, 'context_column' must be a list of attributes.");
        }
    }

    _context_columns = _model_config_json["context_column"];
    _prompt = _model_config_json["prompt"].get<std::string>();
    _debug = _model_config_json.value("debug", false);
}

void AIOperator::rebuild_model() {
    try {
        _model = std::make_unique<flock::Model>(_model_config_json);
    } catch (const std::exception& ex) {
        throw std::runtime_error(std::string("AIOperator: failed to construct Model: ") + ex.what());
    }
}

void AIOperator::init(Schema* schema) {
    _schema = schema;
    load_config();
    rebuild_model();

    _num_output_tuples = 0;
    _required_attributes.clear();
    _batch_iterator.reset();
    _reconstructor.reset();

    // Create dedicated LLM string pool + dictionary.
    _llm_pool = std::make_unique<StringPool>();
    _llm_dict = std::make_unique<StringDictionary>(_llm_pool.get());
    _llm_dict->finalize();
    // Expose the LLM dictionary to downstream sinks (e.g., SinkExport).
    _schema->llm_dictionary = _llm_dict.get();

    // Decide iterator filling policy from LLM config:
    // - tuple_format == FTREE        => tree mode (factorized batches)
    // - otherwise (JSON/Markdown/..) => flat mode (flat tuples)
    _mode = (_tuple_format == "FTREE") ? SerializationMode::Tree : SerializationMode::Flat;

    // Build context_column capacities and _required_attributes (for serialization).
    std::unordered_map<std::string, size_t> context_col_capacities;
    if (_mode == SerializationMode::Tree) {
        for (const auto& [attr, _] : _context_columns.items()) {
            context_col_capacities[attr] = static_cast<size_t>(_context_columns[attr].get<int>());
            _required_attributes.push_back(attr);
        }
    } else {
        for (const auto& attr_json : _context_columns) {
            if (!attr_json.is_string()) {
                throw std::runtime_error(
                    "AIOperator: 'context_column' list must contain only attribute strings.");
            }
            const std::string attr = attr_json.get<std::string>();
            context_col_capacities[attr] = State::MAX_VECTOR_SIZE;
            _required_attributes.push_back(attr);
        }
    }

    // Pass ALL ftree-present attributes to the batch iterator so the full tree
    // is projected. Context column attrs use their configured capacity;
    // other ftree attrs get MAX_VECTOR_SIZE.
    assert(_schema->column_ordering != nullptr &&
           "AIOperator: schema column_ordering must exist at MAP init time");
    assert(_schema->root != nullptr && "AIOperator: schema root must exist");

    // Collect all attributes present in the current ftree.
    std::unordered_set<std::string> ftree_attrs;
    {
        std::vector<FactorizedTreeElement*> stack = {_schema->root.get()};
        while (!stack.empty()) {
            auto* n = stack.back();
            stack.pop_back();
            ftree_attrs.insert(n->_attribute);
            for (auto& child : n->_children) stack.push_back(child.get());
        }
    }

    std::unordered_map<std::string, size_t> required_attributes_map;
    for (const auto& attr : ftree_attrs) {
        auto it = context_col_capacities.find(attr);
        if (it != context_col_capacities.end()) {
            required_attributes_map[attr] = it->second;
        } else {
            required_attributes_map[attr] = State::MAX_VECTOR_SIZE;
        }
    }

    // Ensure all ftree nodes have _value wired from the schema map.
    // When MAP is early in the pipeline (before joins), Scan only allocates the
    // vector in the map but never calls add_leaf(), leaving _value == nullptr.
    {
        std::vector<FactorizedTreeElement*> stack = {_schema->root.get()};
        while (!stack.empty()) {
            auto* n = stack.back();
            stack.pop_back();
            if (n->_value == nullptr) {
                auto* vec = _schema->map->get_vector(n->_attribute);
                if (vec) n->set_value_ptr(vec);
            }
            for (auto& child : n->_children) stack.push_back(child.get());
        }
    }

    _batch_iterator = std::make_unique<FTreeBatchIterator>(required_attributes_map);
    _batch_iterator->init(_schema);

    // Create and initialize the FTreeReconstructor.
    const std::string out_attr = *_schema->llm_output_attr;
    _reconstructor = std::make_unique<FTreeReconstructor>(out_attr, _required_attributes);

    // Serializer for this operator instance.
    _serializer = std::make_unique<AISerializer>(_tuple_format);

    _reconstructor->init(_schema, _batch_iterator.get(), [this]() {
        if (next_op) next_op->execute();
    });

    // Apply schema mutations from reconstructor so downstream sees the new tree.
    _schema->root = _reconstructor->get_new_root();
    _schema->column_ordering = &_reconstructor->get_visible_ordering();

    // Allow subclasses to perform additional initialization (e.g., serializer, batch iterator).
    init_internal();

    if (next_op) { next_op->init(schema); }
}

void AIOperator::execute() {
    num_exec_call++;
    execute_internal();
}

} // namespace ffx

