#include "include/adj_list_builder.hpp"

#include <algorithm>
#include <cstring>
#include <memory>

namespace ffx {

void build_adj_lists_from_columns(
    const uint64_t* src_col,
    const uint64_t* dest_col,
    uint64_t num_rows,
    uint64_t num_fwd_ids,
    uint64_t num_bwd_ids,
    std::unique_ptr<AdjList<uint64_t>[]>& out_fwd_adj,
    std::unique_ptr<AdjList<uint64_t>[]>& out_bwd_adj) {
    
    // First pass: Count degrees
    auto fwd_counts = std::make_unique<size_t[]>(num_fwd_ids);
    std::memset(fwd_counts.get(), 0, num_fwd_ids * sizeof(size_t));
    auto bwd_counts = std::make_unique<size_t[]>(num_bwd_ids);
    std::memset(bwd_counts.get(), 0, num_bwd_ids * sizeof(size_t));
    
    // NULL value representation
    constexpr uint64_t NULL_ID = UINT64_MAX;
    
    for (uint64_t i = 0; i < num_rows; i++) {
        uint64_t src = src_col[i];
        uint64_t dest = dest_col[i];

        // Preserve forward edges with NULL destinations so predicates like
        // IS_NULL(attr) can be evaluated correctly later in the pipeline.
        // Backward lists still skip NULL destinations because there is no
        // concrete destination node/index to attach.
        if (src == NULL_ID) {
            continue;
        }

        if (src < num_fwd_ids) {
            fwd_counts[src]++;
        }
        if (dest != NULL_ID && dest < num_bwd_ids) {
            bwd_counts[dest]++;
        }
    }
    
    // Allocate adjacency lists
    out_fwd_adj = std::make_unique<AdjList<uint64_t>[]>(num_fwd_ids);
    out_bwd_adj = std::make_unique<AdjList<uint64_t>[]>(num_bwd_ids);
    
    // Pre-allocate space
    for (size_t i = 0; i < num_fwd_ids; i++) {
        if (fwd_counts[i] > 0) {
            new (&out_fwd_adj[i]) AdjList<uint64_t>(fwd_counts[i]);
            out_fwd_adj[i].size = 0;
        }
    }
    
    for (size_t i = 0; i < num_bwd_ids; i++) {
        if (bwd_counts[i] > 0) {
            new (&out_bwd_adj[i]) AdjList<uint64_t>(bwd_counts[i]);
            out_bwd_adj[i].size = 0;
        }
    }
    
    // Second pass: Fill lists
    for (uint64_t i = 0; i < num_rows; i++) {
        uint64_t src = src_col[i];
        uint64_t dest = dest_col[i];

        if (src == NULL_ID) {
            continue;
        }

        // Forward (A->B): keep NULL destinations (UINT64_MAX) to preserve
        // nullable attribute semantics for downstream predicates.
        if (src < num_fwd_ids) {
            auto& fwd_list = out_fwd_adj[src];
            fwd_list.values[fwd_list.size++] = dest;
        }

        // Backward (B->A): only materialize concrete destination IDs.
        if (dest != NULL_ID && dest < num_bwd_ids && src < num_fwd_ids) {
            auto& bwd_list = out_bwd_adj[dest];
            bwd_list.values[bwd_list.size++] = src;
        }
    }
    
    // Sort adjacency lists
    for (size_t i = 0; i < num_fwd_ids; i++) {
        auto& fwd_list = out_fwd_adj[i];
        if (fwd_list.size > 1 && 
            !std::is_sorted(fwd_list.values, fwd_list.values + fwd_list.size)) {
            std::sort(fwd_list.values, fwd_list.values + fwd_list.size);
        }
    }
    
    for (size_t i = 0; i < num_bwd_ids; i++) {
        auto& bwd_list = out_bwd_adj[i];
        if (bwd_list.size > 1 && 
            !std::is_sorted(bwd_list.values, bwd_list.values + bwd_list.size)) {
            std::sort(bwd_list.values, bwd_list.values + bwd_list.size);
        }
    }
}

} // namespace ffx

