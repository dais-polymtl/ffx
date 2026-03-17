#ifndef VFENGINE_DATASOURCE_H
#define VFENGINE_DATASOURCE_H

#include "adj_list.hpp"
#include "cardinality.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace ffx {

class Table {
public:
    Table() = delete;
    Table(const Table&) = delete;
    Table& operator=(const Table&) /* noexcept */ = delete;

    explicit Table(uint64_t num_fwd_ids, uint64_t num_bwd_ids, const std::string& csv_absolute_filename);
    explicit Table(uint64_t num_fwd_ids, uint64_t num_bwd_ids, std::unique_ptr<AdjList<uint64_t>[]> fwd_adj_lists,
                   std::unique_ptr<AdjList<uint64_t>[]> bwd_adj_lists);

    AdjList<uint64_t>* fwd_adj_lists;
    AdjList<uint64_t>* bwd_adj_lists;
    const uint64_t num_fwd_ids;
    const uint64_t num_bwd_ids;

    std::string name;
    std::vector<std::string> columns;

    // Cardinality of the relation (default: m:n)
    Cardinality cardinality = Cardinality::MANY_TO_MANY;

    // Set of column names that are string type (for min operator string comparison)
    std::unordered_set<std::string> string_columns;

    bool should_share_state(bool is_fwd) const { return ffx::should_share_state(cardinality, is_fwd); }


private:
    std::unique_ptr<AdjList<uint64_t>[]> _fwd_adj_lists;
    std::unique_ptr<AdjList<uint64_t>[]> _bwd_adj_lists;
};

}// namespace ffx

#endif
