#ifndef FFX_ADJ_LIST_BUILDER_HPP
#define FFX_ADJ_LIST_BUILDER_HPP

#include "adj_list.hpp"
#include <memory>
#include <cstdint>

namespace ffx {

// Build adjacency lists from column vectors (no CSV needed)
// Similar to ingest_csv_and_build_indexes but operates on in-memory column data
void build_adj_lists_from_columns(
    const uint64_t* src_col,
    const uint64_t* dest_col,
    uint64_t num_rows,
    uint64_t num_fwd_ids,   // e.g., max(src_col) + 1
    uint64_t num_bwd_ids,   // e.g., max(dest_col) + 1
    std::unique_ptr<AdjList<uint64_t>[]>& out_fwd_adj,
    std::unique_ptr<AdjList<uint64_t>[]>& out_bwd_adj
);

} // namespace ffx

#endif // FFX_ADJ_LIST_BUILDER_HPP

