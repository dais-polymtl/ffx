#include "sink/sink_export.hpp"

#include "schema/schema.hpp"
#include "string_dictionary.hpp"

#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <flock/metrics/manager.hpp>

namespace ffx {

static std::string csv_escape(const std::string& s) {
    bool need_quotes = false;
    for (char c : s) {
        if (c == '"' || c == ',' || c == '\n' || c == '\r') {
            need_quotes = true;
            break;
        }
    }
    if (!need_quotes) return s;

    std::string out;
    out.reserve(s.size() + 2);
    out.push_back('"');
    for (char c : s) {
        if (c == '"') out.push_back('"');
        out.push_back(c);
    }
    out.push_back('"');
    return out;
}

static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '\"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out.push_back(c); break;
        }
    }
    return out;
}

SinkExport::SinkExport(Format fmt) : _fmt(fmt), _schema(nullptr), _itr() {}

SinkExport::SinkExport(Format fmt, std::string output_path)
    : _fmt(fmt), _schema(nullptr), _itr(), _output_path(std::move(output_path)), _output_path_explicit(true) {}

static void emit_markdown_output(std::ostream& out,
                                 const std::vector<std::string>& cols,
                                 const std::vector<std::vector<std::string>>& rows) {
    const size_t ncols = cols.size();
    std::vector<size_t> widths(ncols);
    for (size_t i = 0; i < ncols; ++i) widths[i] = cols[i].size();
    for (const auto& row : rows) {
        for (size_t i = 0; i < ncols; ++i) {
            widths[i] = std::max(widths[i], row[i].size());
        }
    }

    out << "-------------- Output ----------------\n\n";
    // Header row
    out << "|";
    for (size_t i = 0; i < ncols; ++i) {
        out << " ";
        out << cols[i];
        for (size_t p = cols[i].size(); p < widths[i]; ++p) out << ' ';
        out << " |";
    }
    out << "\n";

    // Separator
    out << "|";
    for (size_t i = 0; i < ncols; ++i) {
        out << "-";
        for (size_t p = 0; p < widths[i]; ++p) out << '-';
        out << "-|";
    }
    out << "\n";

    // Data rows
    for (const auto& row : rows) {
        out << "|";
        for (size_t i = 0; i < ncols; ++i) {
            out << " ";
            out << row[i];
            for (size_t p = row[i].size(); p < widths[i]; ++p) out << ' ';
            out << " |";
        }
        out << "\n";
    }
    out.flush();
}

SinkExport::~SinkExport() {
    if (_header_written) {
        // Always print a nicely formatted output block to stdout for export sinks,
        // regardless of the on-disk format (CSV/JSON/MARKDOWN).
        emit_markdown_output(std::cout, _cols, _md_rows);

        // Finalize on-disk format if we were writing to a file.
        if (_file_stream.is_open()) {
            write_footer(_file_stream);
        }

        // If a file was explicitly specified, print the path (matches CLI style).
        if (_file_stream.is_open() && _output_path_explicit && _output_path.has_value()) {
            std::cout << "\n- Writing output to: " << *_output_path << std::endl;
        }

        // If we have an on-disk output path, also persist aggregated LLM metrics
        // next to the data file (metrics.json).
        if (_output_path.has_value()) {
            const std::string& data_path = *_output_path;
            const auto pos = data_path.find_last_of("/\\");
            const std::string dir =
                (pos == std::string::npos) ? std::string(".") : data_path.substr(0, pos);
            const std::string metrics_path = dir + "/metrics.json";
            flock::MetricsManager::WriteToFile(metrics_path);
            std::cout << "- Metrics saved to: " << metrics_path << std::endl;
        }
    }
}

std::optional<std::string> SinkExport::read_env(const char* key) {
    if (const char* v = std::getenv(key)) {
        std::string s(v);
        if (!s.empty()) return s;
    }
    return std::nullopt;
}

void SinkExport::init(Schema* schema) {
    _schema = schema;
    _num_output_tuples = 0;
    _cols.clear();
    _dict = nullptr;
    _header_written = false;
    _first_json_row = true;

    if (_schema == nullptr) {
        throw std::runtime_error("SinkExport: schema is null");
    }
    if (_schema->root == nullptr) {
        throw std::runtime_error("SinkExport: schema root is null");
    }
    if (_schema->column_ordering == nullptr) {
        throw std::runtime_error("SinkExport: schema column_ordering is null");
    }

    _dict = _schema->dictionary;
    _llm_dict = _schema->llm_dictionary;

    // Start this query with clean LLM usage metrics.
    flock::MetricsManager::Reset();

    for (const auto& c : *_schema->column_ordering) {
        if (c != "_cd") _cols.push_back(c);
    }
    if (_cols.empty()) {
        throw std::runtime_error("SinkExport: empty column ordering");
    }

    if (!_output_path.has_value() && _schema && _schema->export_output_path) {
        if (!_schema->export_output_path->empty()) {
            _output_path = *_schema->export_output_path;
            _output_path_explicit = true;
        }
    }

    if (!_output_path.has_value()) {
        if (auto p = read_env("FFX_SINK_OUTPUT_PATH")) {
            _output_path = std::move(*p);
            // Env-provided path is not considered "specified" by the CLI.
            _output_path_explicit = false;
        }
    }

    if (_output_path.has_value()) {
        _file_stream.open(*_output_path, std::ios::out | std::ios::trunc);
        if (!_file_stream.is_open()) {
            throw std::runtime_error("SinkExport: failed to open output file: " + *_output_path);
        }
    }

    // FTreeIterator assumes parent attributes precede children in the provided ordering.
    // After CartesianProduct, the tree is linearized in chain order, but the visible
    // output ordering may be the original query ordering (which can violate that
    // parent-before-child constraint). We therefore iterate in chain order and
    // then permute into the visible ordering when writing rows.
    _itr_ordering.clear();
    {
        // DFS traversal to collect all attributes in the ftree (handles branching
        // topologies, e.g., when an LLM Map output is a leaf sibling of join children).
        std::function<void(FactorizedTreeElement*)> collect = [&](FactorizedTreeElement* node) {
            if (!node) return;
            _itr_ordering.push_back(node->_attribute);
            for (auto& child : node->_children) {
                collect(child.get());
            }
        };
        collect(_schema->root.get());
    }
    if (_itr_ordering.empty()) {
        throw std::runtime_error("SinkExport: empty chain ordering");
    }
    std::unordered_map<std::string, size_t> itr_idx;
    itr_idx.reserve(_itr_ordering.size());
    for (size_t i = 0; i < _itr_ordering.size(); ++i) {
        itr_idx[_itr_ordering[i]] = i;
    }

    // Keep only columns present in the reconstructed chain.
    // Map may output a projected subset (e.g., context attrs + _llm).
    std::vector<std::string> filtered_cols;
    std::vector<size_t> filtered_cols_to_itr_idx;
    filtered_cols.reserve(_cols.size());
    filtered_cols_to_itr_idx.reserve(_cols.size());
    for (const auto& col : _cols) {
        auto it = itr_idx.find(col);
        if (it == itr_idx.end()) continue;
        filtered_cols.push_back(col);
        filtered_cols_to_itr_idx.push_back(it->second);
    }
    _cols = std::move(filtered_cols);
    _cols_to_itr_idx = std::move(filtered_cols_to_itr_idx);
    if (_cols.empty()) {
        throw std::runtime_error("SinkExport: no output columns present in chain ordering");
    }

    _is_string_col.assign(_cols.size(), false);
    for (size_t i = 0; i < _cols.size(); ++i) {
        if (_cols[i] == "_llm" || (_schema->llm_output_attr && _cols[i] == *_schema->llm_output_attr)) {
            _is_string_col[i] = true;
        } else if (_schema->string_attributes && _schema->string_attributes->count(_cols[i])) {
            _is_string_col[i] = true;
        }
    }

    // Temporarily override schema ordering for iterator initialization.
    const auto* original_ordering = _schema->column_ordering;
    _schema->column_ordering = &_itr_ordering;
    _itr.init(_schema);
    _schema->column_ordering = original_ordering;
}

std::string SinkExport::format_value(size_t col_idx, uint64_t val) const {
    if (!_is_string_col[col_idx] || !_dict) {
        return std::to_string(val);
    }

    // LLM columns use a special encoding: dict_id + 1 (so 0 means "unset").
    const bool is_llm_col =
            (_cols[col_idx] == "_llm") || (_schema && _schema->llm_output_attr && _cols[col_idx] == *_schema->llm_output_attr);

    if (is_llm_col) {
        if (val == 0) return "";
        const uint64_t dict_id = val - 1;
        // Prefer the dedicated LLM dictionary; fall back to the global one.
        if (_llm_dict && _llm_dict->has_id(dict_id)) return _llm_dict->get_string(dict_id).to_string();
        if (_dict && _dict->has_id(dict_id)) return _dict->get_string(dict_id).to_string();
        return std::to_string(val);
    }

    // Regular string columns from table_serializer are stored as 0-based dictionary IDs.
    if (_dict->has_id(val)) return _dict->get_string(val).to_string();
    return std::to_string(val);
}

void SinkExport::execute() {
    num_exec_call++;

    std::ostream& out = _file_stream.is_open() ? _file_stream : std::cout;

    if (!_header_written) {
        // For CSV/JSON to stdout, render markdown preview only (no raw format).
        const bool write_raw_header = (_fmt == Format::MARKDOWN) || _file_stream.is_open();
        if (write_raw_header) write_header(out);
        _header_written = true;
    }

    write_rows(out);
}

// ---- Header / footer per format ----

void SinkExport::write_header(std::ostream& out) {
    switch (_fmt) {
        case Format::CSV:
            for (size_t i = 0; i < _cols.size(); ++i) {
                if (i) out << ",";
                out << csv_escape(_cols[i]);
            }
            out << "\n";
            break;
        case Format::JSON:
            out << "[\n";
            break;
        case Format::MARKDOWN:
            _md_rows.clear();
            break;
    }
}

void SinkExport::write_footer(std::ostream& out) {
    switch (_fmt) {
        case Format::JSON:
            out << "\n]\n";
            break;
        case Format::MARKDOWN:
            // stdout rendering is handled in ~SinkExport() for consistent formatting
            // across all export sinks; only flush here for file outputs.
            if (_file_stream.is_open()) emit_markdown_output(out, _cols, _md_rows);
            break;
        default:
            break;
    }
}

// ---- Data rows (called per flush) ----

void SinkExport::write_rows(std::ostream& out) {
    _itr.reset();
    _itr.initialize_iterators();
    if (!_itr.is_valid()) { return; }

    std::vector<uint64_t> itr_buf(_itr.tuple_size());
    std::vector<uint64_t> out_buf(_cols.size());
    bool more = true;
    while (more) {
        more = _itr.next(itr_buf.data());
        for (size_t i = 0; i < _cols.size(); ++i) {
            out_buf[i] = itr_buf[_cols_to_itr_idx[i]];
        }
        std::vector<std::string> row(out_buf.size());
        for (size_t i = 0; i < out_buf.size(); ++i) {
            row[i] = format_value(i, out_buf[i]);
        }

        switch (_fmt) {
            case Format::CSV:
                if (!_file_stream.is_open()) break;
                for (size_t i = 0; i < out_buf.size(); ++i) {
                    if (i) out << ",";
                    out << csv_escape(row[i]);
                }
                out << "\n";
                break;

            case Format::JSON:
                if (!_file_stream.is_open()) break;
                if (!_first_json_row) out << ",\n";
                _first_json_row = false;
                out << "  {";
                for (size_t i = 0; i < _cols.size(); ++i) {
                    if (i) out << ",";
                    if (_is_string_col[i]) {
                        out << "\"" << _cols[i] << "\":\"" << json_escape(row[i]) << "\"";
                    } else {
                        out << "\"" << _cols[i] << "\":" << out_buf[i];
                    }
                }
                out << "}";
                break;

            case Format::MARKDOWN:
                _md_rows.push_back(std::move(row));
                break;
        }
        // Always buffer a markdown-friendly version so stdout can show a
        // well-formatted preview at the end (consistent formatting).
        if (_fmt != Format::MARKDOWN) {
            _md_rows.push_back(std::move(row));
        }
        _num_output_tuples++;
    }
    out.flush();
}

} // namespace ffx
