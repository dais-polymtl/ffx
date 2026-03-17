#ifndef FFX_SINK_EXPORT_OPERATOR_HH
#define FFX_SINK_EXPORT_OPERATOR_HH

#include "factorized_ftree/ftree_iterator.hpp"
#include "operator.hpp"
#include <cstddef>
#include <fstream>
#include <unordered_map>
#include <optional>
#include <string>
#include <vector>

namespace ffx {

class StringDictionary;

class SinkExport final : public Operator {
public:
    enum class Format { CSV, JSON, MARKDOWN };

    explicit SinkExport(Format fmt);
    SinkExport(Format fmt, std::string output_path);
    ~SinkExport() override;

    void init(Schema* schema) override;
    void execute() override;
    uint64_t get_num_output_tuples() override { return _num_output_tuples; }

private:
    static std::optional<std::string> read_env(const char* key);

    std::string format_value(size_t col_idx, uint64_t val) const;

    void write_header(std::ostream& out);
    void write_rows(std::ostream& out);
    void write_footer(std::ostream& out);

    Format _fmt;
    Schema* _schema;
    FTreeIterator _itr;
    std::vector<std::string> _cols;
    std::vector<std::string> _itr_ordering;
    std::vector<size_t> _cols_to_itr_idx;
    std::vector<bool> _is_string_col;
    StringDictionary* _dict{nullptr};
    StringDictionary* _llm_dict{nullptr};
    uint64_t _num_output_tuples{0};
    std::optional<std::string> _output_path;
    bool _output_path_explicit{false};
    bool _header_written{false};
    bool _first_json_row{true};
    std::ofstream _file_stream;

    // Markdown: buffer all rows so we can compute column widths for alignment.
    std::vector<std::vector<std::string>> _md_rows;
};

} // namespace ffx

#endif

