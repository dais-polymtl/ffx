#include "include/table.hpp"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

namespace ffx {

static void ingest_csv_and_build_indexes(const std::string& csv_absolute_filename, 
                                         uint64_t num_fwd_ids, uint64_t num_bwd_ids,
                                         AdjList<uint64_t>* fwd_adj_lists, AdjList<uint64_t>* bwd_adj_lists);

Table::Table(uint64_t num_fwd_ids, uint64_t num_bwd_ids, const std::string& csv_absolute_filename) 
    : num_fwd_ids(num_fwd_ids), num_bwd_ids(num_bwd_ids) {
    _fwd_adj_lists = std::make_unique<AdjList<uint64_t>[]>(num_fwd_ids);
    _bwd_adj_lists = std::make_unique<AdjList<uint64_t>[]>(num_bwd_ids);
    fwd_adj_lists = _fwd_adj_lists.get();
    bwd_adj_lists = _bwd_adj_lists.get();
    ingest_csv_and_build_indexes(csv_absolute_filename, num_fwd_ids, num_bwd_ids, fwd_adj_lists, bwd_adj_lists);
}

Table::Table(uint64_t num_fwd_ids, uint64_t num_bwd_ids, 
             std::unique_ptr<AdjList<uint64_t>[]> fwd_adj_lists, std::unique_ptr<AdjList<uint64_t>[]> bwd_adj_lists)
    : num_fwd_ids(num_fwd_ids), num_bwd_ids(num_bwd_ids), 
      _fwd_adj_lists(std::move(fwd_adj_lists)), _bwd_adj_lists(std::move(bwd_adj_lists)) {
    this->fwd_adj_lists = _fwd_adj_lists.get();
    this->bwd_adj_lists = _bwd_adj_lists.get();
}

static bool can_open_file(const std::string& filename);

void ingest_csv_and_build_indexes(const std::string& csv_absolute_filename, 
                                  uint64_t num_fwd_ids, uint64_t num_bwd_ids,
                                  AdjList<uint64_t>* fwd_adj_lists, AdjList<uint64_t>* bwd_adj_lists) {
    if (csv_absolute_filename.empty() or !can_open_file(csv_absolute_filename)) {
        throw std::runtime_error("Error opening file for reading CSV : " + csv_absolute_filename);
    }

    // First pass: Count edges per vertex to pre-allocate
    auto fwd_counts = std::make_unique<size_t[]>(num_fwd_ids);
    std::memset(fwd_counts.get(), 0, num_fwd_ids * sizeof(size_t));
    auto bwd_counts = std::make_unique<size_t[]>(num_bwd_ids);
    std::memset(bwd_counts.get(), 0, num_bwd_ids * sizeof(size_t));

    {
        std::ifstream file;
        file.open(csv_absolute_filename);
        std::string line;
        uint64_t src, dest;

        while (std::getline(file, line)) {
            if (line[0] == '#') { continue; }
            std::istringstream iss(line);
            if (iss >> src >> dest) {
                if (src < num_fwd_ids) {
                    fwd_counts[src]++;
                }
                if (dest < num_bwd_ids) {
                    bwd_counts[dest]++;
                }
            }
        }
    }

    // Pre-allocate space for forward adjacency lists
    for (size_t i = 0; i < num_fwd_ids; i++) {
        if (fwd_counts[i] > 0) {
            new (&fwd_adj_lists[i]) AdjList<uint64_t>(fwd_counts[i]);
            fwd_adj_lists[i].size = 0;
        }
    }

    // Pre-allocate space for backward adjacency lists
    for (size_t i = 0; i < num_bwd_ids; i++) {
        if (bwd_counts[i] > 0) {
            new (&bwd_adj_lists[i]) AdjList<uint64_t>(bwd_counts[i]);
            bwd_adj_lists[i].size = 0;
        }
    }

    // Second pass: Fill adjacency lists
    std::ifstream file;
    file.open(csv_absolute_filename);
    std::string line;
    while (std::getline(file, line)) {
        if (line[0] == '#') { continue; }
        std::istringstream iss(line);
        uint64_t src, dest;
        if (iss >> src >> dest) {
            if (src < num_fwd_ids) {
                auto& fwd_adj_list = fwd_adj_lists[src];
                fwd_adj_list.values[fwd_adj_list.size++] = dest;
            }
            if (dest < num_bwd_ids) {
                auto& bwd_adj_list = bwd_adj_lists[dest];
                bwd_adj_list.values[bwd_adj_list.size++] = src;
            }
        }
    }

    // Ensure adjacency lists are sorted (both forward and backward)
    for (size_t i = 0; i < num_fwd_ids; i++) {
        auto& fwd_adj_list = fwd_adj_lists[i];
        if (fwd_adj_list.size > 1 &&
            !std::is_sorted(fwd_adj_list.values, fwd_adj_list.values + fwd_adj_list.size)) {
            std::sort(fwd_adj_list.values, fwd_adj_list.values + fwd_adj_list.size);
            }
    }
    for (size_t i = 0; i < num_bwd_ids; i++) {
        auto& bwd_adj_list = bwd_adj_lists[i];
        if (bwd_adj_list.size > 1 &&
            !std::is_sorted(bwd_adj_list.values, bwd_adj_list.values + bwd_adj_list.size)) {
            std::sort(bwd_adj_list.values, bwd_adj_list.values + bwd_adj_list.size);
        }
    }
}

static bool can_open_file(const std::string& filename) {
    std::ifstream file(filename);
    auto success = file.is_open();
    if (success) { file.close(); }
    return success;
}

}// namespace ffx
