#include "flock/prompt_manager/prompt_manager.hpp"
#include <sstream>

namespace flock {
template<>
std::string PromptManager::ToString<PromptSection>(const PromptSection section) {
    switch (section) {
        case PromptSection::USER_PROMPT:
            return "{{USER_PROMPT}}";
        case PromptSection::TUPLES:
            return "{{TUPLES}}";
        case PromptSection::RESPONSE_FORMAT:
            return "{{RESPONSE_FORMAT}}";
        case PromptSection::INSTRUCTIONS:
            return "{{INSTRUCTIONS}}";
        default:
            return "";
    }
}

std::string PromptManager::ReplaceSection(const std::string& prompt_template, const PromptSection section,
                                          const std::string& section_content) {
    auto replace_string = PromptManager::ToString(section);
    return PromptManager::ReplaceSection(prompt_template, replace_string, section_content);
}

std::string PromptManager::ReplaceSection(const std::string& prompt_template, const std::string& replace_string,
                                          const std::string& section_content) {
    auto prompt = prompt_template;
    auto replace_string_size = replace_string.size();
    auto replace_pos = prompt.find(replace_string);

    while (replace_pos != std::string::npos) {
        prompt.replace(replace_pos, replace_string_size, section_content);
        replace_pos = prompt.find(replace_string, replace_pos + section_content.size());
    }

    return prompt;
}

std::string PromptManager::ConstructInputTuplesHeader(const nlohmann::json& columns, const std::string& tuple_format) {
    switch (stringToTupleFormat(tuple_format)) {
        case TupleFormat::XML:
            return ConstructInputTuplesHeaderXML(columns);
        case TupleFormat::Markdown:
            return ConstructInputTuplesHeaderMarkdown(columns);
        case TupleFormat::JSON:
            return "";
        case TupleFormat::ColumnEncoded:
            return "";
        case TupleFormat::CompactRepFormat:
            return "";
        default:
            throw std::runtime_error("Invalid tuple format provided `" + tuple_format + "`");
    }
}

std::string PromptManager::ConstructInputTuplesHeaderXML(const nlohmann::json& columns) {
    if (columns.empty()) { return "<header></header>\n"; }
    auto header = std::string("<header>");
    auto column_idx = 1u;
    for (const auto& column: columns) {
        std::string column_name;
        if (column.contains("name") && column["name"].is_string()) {
            column_name = column["name"].get<std::string>();
        } else {
            column_name = "COLUMN " + std::to_string(column_idx++);
        }
        header += "<column>" + column_name + "</column>";
    }
    header += "</header>\n";
    return header;
}

std::string PromptManager::ConstructInputTuplesHeaderMarkdown(const nlohmann::json& columns) {
    if (columns.empty()) { return " | Empty | \n | ----- | \n"; }
    auto header = std::string(" | ");
    auto column_idx = 1u;
    for (const auto& column: columns) {
        if (column.contains("name") && column["name"].is_string()) {
            header += "COLUMN_" + column["name"].get<std::string>() + " | ";
        } else {
            header += "COLUMN " + std::to_string(column_idx++) + " | ";
        }
    }
    header += "\n | ";
    column_idx = 1u;
    for (const auto& column: columns) {
        std::string column_name;
        if (column.contains("name") && column["name"].is_string()) {
            column_name = column["name"].get<std::string>();
        } else {
            column_name = "COLUMN " + std::to_string(column_idx++);
        }
        header += std::string(column_name.length(), '-') + " | ";
    }
    header += "\n";
    return header;
}

std::string PromptManager::ConstructInputTuplesXML(const nlohmann::json& columns) {
    if (columns.empty() || columns[0]["data"].empty()) { return "<row></row>\n"; }

    auto tuples_str = std::string("");
    for (auto i = 0; i < static_cast<int>(columns[0]["data"].size()); i++) {
        tuples_str += "<row>";
        for (const auto& column: columns) {
            std::string value_str;
            const auto& data_item = column["data"][i];
            if (data_item.is_null()) {
                value_str = "";
            } else if (data_item.is_string()) {
                value_str = data_item.get<std::string>();
            } else {
                value_str = data_item.dump();
            }
            tuples_str += "<column>" + value_str + "</column>";
        }
        tuples_str += "</row>\n";
    }
    return tuples_str;
}

std::string PromptManager::ConstructInputTuplesMarkdown(const nlohmann::json& columns) {
    if (columns.empty() || columns[0]["data"].empty()) { return ""; }

    auto tuples_str = std::string("");
    for (auto i = 0; i < static_cast<int>(columns[0]["data"].size()); i++) {
        tuples_str += " | ";
        for (const auto& column: columns) {
            tuples_str += column["data"][i].dump() + " | ";
        }
        tuples_str += "\n";
    }
    return tuples_str;
}

std::string PromptManager::ConstructInputTuplesJSON(const nlohmann::json& columns) {
    auto tuples_json = nlohmann::json::object();
    auto column_idx = 1u;
    for (const auto& column: columns) {
        std::string column_name;
        if (column.contains("name") && column["name"].is_string()) {
            column_name = column["name"].get<std::string>();
        } else {
            column_name = "COLUMN " + std::to_string(column_idx++);
        }
        tuples_json[column_name] = column["data"];
    }
    auto tuples_str = tuples_json.dump(4);
    tuples_str += "\n";
    return tuples_str;
}

std::string PromptManager::ConstructInputTuplesColumnEncoded(const nlohmann::json& columns) {
    if (columns.empty()) { return ""; }

    std::ostringstream oss;
    auto column_idx = 1u;

    for (const auto& column: columns) {
        std::string column_name;
        if (column.contains("name") && column["name"].is_string()) {
            column_name = column["name"].get<std::string>();
        } else {
            column_name = "COLUMN " + std::to_string(column_idx++);
        }

        oss << "- " << column_name << ": ";
        if (!column.contains("data") || !column["data"].is_array() || column["data"].empty()) {
            oss << "\n";
            continue;
        }

        const auto& data = column["data"];
        bool first = true;
        for (const auto& v: data) {
            if (!first) {
                oss << " | ";
            }
            first = false;
            if (v.is_null()) {
                oss << "null";
            } else if (v.is_string()) {
                oss << v.get<std::string>();
            } else {
                oss << v.dump();
            }
        }
        oss << "\n";
    }

    return oss.str();
}

namespace {
static std::string indent(int n) { return std::string(static_cast<size_t>(n), ' '); }

// Attempt to extract optional metadata fields for nicer debug/prompts.
static void append_optional_metadata(std::ostringstream& oss, const nlohmann::json& obj) {
    // Support a few common keys; if not present, do nothing.
    auto append_if = [&](const char* key, const char* label) {
        if (obj.contains(key) && obj[key].is_string()) {
            oss << label << ": \"" << obj[key].get<std::string>() << "\" ";
        }
    };
    append_if("title", "Title");
    append_if("Title", "Title");
    append_if("abstract", "Abstract");
    append_if("Abstract", "Abstract");
}

// Skip factorized nodes with no values and no non-empty children (avoids bare `c:` lines in FTREE prompts).
static bool ftree_compact_node_is_empty(const nlohmann::json& node) {
    if (!node.is_object()) return true;
    const bool has_vals = node.contains("values") && node["values"].is_array() && !node["values"].empty();
    if (has_vals) return false;
    if (!node.contains("children") || !node["children"].is_object() || node["children"].empty()) return true;
    for (auto it = node["children"].begin(); it != node["children"].end(); ++it) {
        if (!ftree_compact_node_is_empty(it.value())) return false;
    }
    return true;
}

// Render a node-centric subtree of the form:
// { "values": [...], "children": { "<childAttr>": { ... }, ... } }
static void render_node(std::ostringstream& oss, const std::string& attr_name, const nlohmann::json& node,
                        int base_indent) {
    if (!node.is_object()) return;

    if (node.contains("values") && node["values"].is_array()) {
        for (const auto& v: node["values"]) {
            std::string vstr = v.is_string() ? v.get<std::string>() : v.dump();
            oss << indent(base_indent) << "- " << vstr << "\n";
        }
    }

    if (!node.contains("children") || !node["children"].is_object()) return;
    const auto& children = node["children"];
    for (auto it = children.begin(); it != children.end(); ++it) {
        const std::string child_attr = it.key();
        const nlohmann::json& child_node = it.value();
        if (ftree_compact_node_is_empty(child_node)) continue;
        oss << indent(base_indent) << child_attr << ":\n";
        render_node(oss, child_attr, child_node, base_indent + 2);
    }
}

}// namespace

std::string PromptManager::ConstructInputTuplesCompactRepFormat(const nlohmann::json& columns) {
    // Expect a single row containing the ftree JSON (either as object or as a JSON string).
    if (columns.empty() || !columns[0].contains("data") || columns[0]["data"].empty()) { return ""; }

    nlohmann::json tree_json = columns[0]["data"][0];
    if (tree_json.is_string()) { tree_json = nlohmann::json::parse(tree_json.get<std::string>()); }
    if (!tree_json.is_object() || !tree_json.contains("root") || !tree_json.contains("tree")) {
        return tree_json.dump(2) + "\n";
    }

    const std::string root_attr = tree_json["root"].get<std::string>();
    const nlohmann::json& tree = tree_json["tree"];
    if (!tree.is_object()) { return tree_json.dump(2) + "\n"; }

    std::ostringstream oss;
    oss << "[FTREE]\n";
    for (auto it = tree.begin(); it != tree.end(); ++it) {
        const std::string root_val = it.key();
        const nlohmann::json& children_obj = it.value();// object mapping childAttr -> node
        oss << "[ROOT " << root_attr << "=" << root_val << "] ";
        append_optional_metadata(oss, children_obj);
        oss << "\n";

        if (!children_obj.is_object()) continue;
        for (auto child_it = children_obj.begin(); child_it != children_obj.end(); ++child_it) {
            const std::string child_attr = child_it.key();
            const nlohmann::json& child_node = child_it.value();
            if (ftree_compact_node_is_empty(child_node)) continue;
            oss << indent(0) << child_attr << ":\n";
            render_node(oss, child_attr, child_node, 2);
        }
    }
    return oss.str();
}

std::string PromptManager::ConstructNumTuples(const int num_tuples) {
    return "- The Number of Tuples to Generate Responses for: " + std::to_string(num_tuples) + "\n\n";
}

std::string PromptManager::ConstructInputTuples(const nlohmann::json& columns, const std::string& tuple_format) {
    auto tuples_str = std::string("");
    const auto num_tuples = columns.size() > 0 ? static_cast<int>(columns[0]["data"].size()) : 0;

    tuples_str += PromptManager::ConstructNumTuples(num_tuples);
    tuples_str += PromptManager::ConstructInputTuplesHeader(columns, tuple_format);
    switch (const auto format = stringToTupleFormat(tuple_format)) {
        case TupleFormat::XML:
            return tuples_str + ConstructInputTuplesXML(columns);
        case TupleFormat::Markdown:
            return tuples_str + ConstructInputTuplesMarkdown(columns);
        case TupleFormat::JSON:
            return tuples_str + ConstructInputTuplesJSON(columns);
        case TupleFormat::ColumnEncoded:
            return tuples_str + ConstructInputTuplesColumnEncoded(columns);
        case TupleFormat::CompactRepFormat:
            return tuples_str + ConstructInputTuplesCompactRepFormat(columns);
        default:
            throw std::runtime_error("Invalid tuple format provided `" + tuple_format + "`");
    }
}

PromptDetails PromptManager::CreatePromptDetails(const nlohmann::json& prompt_details_json) {
    PromptDetails prompt_details;

    try {
        if (prompt_details_json.contains("prompt_name")) {
            // FFX: no DuckDB — require inline prompt text; do not resolve prompt_name from DB.
            throw std::runtime_error("In ffx, provide 'prompt' directly in JSON instead of 'prompt_name'.");
        } else if (prompt_details_json.contains("prompt")) {
            if (prompt_details_json.size() > 1) { throw std::runtime_error(""); }
            if (prompt_details_json["prompt"].get<std::string>().empty()) {
                throw std::runtime_error("The prompt cannot be empty");
            }
            prompt_details.prompt = prompt_details_json["prompt"];
        } else {
            throw std::runtime_error("");
        }
    } catch (const std::exception& e) {
        if (e.what() == std::string("")) {
            throw std::runtime_error("The prompt details struct should contain a single key value pair of prompt or "
                                     "prompt_name with prompt version");
        }
        throw std::runtime_error(e.what());
    }
    return prompt_details;
}

nlohmann::json PromptManager::TranscribeAudioColumn(const nlohmann::json& audio_column) {
    auto transcription_model_name = audio_column["transcription_model"].get<std::string>();

    // Look up the transcription model
    nlohmann::json transcription_model_json;
    transcription_model_json["model_name"] = transcription_model_name;
    Model transcription_model(transcription_model_json);

    // Add transcription requests to batch
    transcription_model.AddTranscriptionRequest(audio_column["data"]);

    // Collect transcriptions
    auto transcription_results = transcription_model.CollectTranscriptions();

    // Convert vector<nlohmann::json> to nlohmann::json array
    nlohmann::json transcriptions = nlohmann::json::array();
    for (const auto& result: transcription_results) {
        transcriptions.push_back(result);
    }

    // Create transcription column with proper naming
    auto transcription_column = nlohmann::json::object();
    std::string original_name;
    if (audio_column.contains("name") && audio_column["name"].is_string()) {
        original_name = audio_column["name"].get<std::string>();
    }
    auto transcription_name = original_name.empty() ? "transcription" : "transcription_of_" + original_name;
    transcription_column["name"] = transcription_name;
    transcription_column["data"] = transcriptions;

    return transcription_column;
}

}// namespace flock
