// Serializer utilities for AI operators: turn database tuples into textual
// representations suitable for LLM prompts (flat, column-encoded, FTREE, ...).
#ifndef FFX_AI_OPERATOR_SERIALIZER_HH
#define FFX_AI_OPERATOR_SERIALIZER_HH

#include <nlohmann/json.hpp>

#include <string>
#include <unordered_set>
#include <vector>

namespace ffx {

class FTreeBatchIterator;
struct Schema;

class AISerializer {
public:
    explicit AISerializer(const std::string& tuple_format);

    // Build a "columns" JSON structure understood by PromptManager for the
    // current iterator window. The attributes vector is the logical projection
    // order (e.g., ["a","b","c"]).
    nlohmann::json build_columns(const FTreeBatchIterator& itr, const Schema* schema,
                                 const std::vector<std::string>& attrs, size_t num_tuples) const;

private:
    std::string _tuple_format;

    // Flat serializers (JSON / Markdown / COLUMN_ENCODED) materialize tuples.
    nlohmann::json build_flat_columns(const FTreeBatchIterator& itr, const Schema* schema,
                                      const std::vector<std::string>& attrs, size_t num_tuples) const;

    // FTREE serializer keeps the factorized tree representation.
    nlohmann::json build_tree_columns(const FTreeBatchIterator& itr, const Schema& schema,
                                      const std::vector<std::string>& attrs, size_t num_tuples) const;
};

} // namespace ffx

#endif // FFX_AI_OPERATOR_SERIALIZER_HH

