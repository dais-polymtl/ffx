#include "sink/sink_itr_merged.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cassert>

namespace ffx {

SinkItrMerged::SinkItrMerged() 
    : _ftree(nullptr), _schema(nullptr) {
}

void SinkItrMerged::init(Schema* schema, 
                          std::shared_ptr<FactorizedTreeElement> ftree,
                          const std::vector<FactorizedTreeElement*>& ordered_nodes,
                          FTreeIterator* existing_iterator) {
    _schema = schema;
    _ftree = ftree;
    _ordered_nodes = ordered_nodes;
    
    // Use the existing iterator (already ran, has debug counts)
    _iterator = existing_iterator;
    
    // Get column ordering from iterator
    _column_ordering = _iterator->get_attribute_ordering();
    for (size_t i = 0; i < _column_ordering.size(); ++i) {
        _attr_to_col_idx[_column_ordering[i]] = i;
    }
    
    // Initialize node analyses
    _node_analyses.clear();
    _attr_to_idx.clear();
    
    // Create analysis entries for all nodes in column ordering
    for (size_t i = 0; i < _column_ordering.size(); ++i) {
        const auto& attr = _column_ordering[i];
        auto* node = _ftree->find_node_by_attribute(attr);
        assert(node != nullptr);
        
        NodeAnalysis analysis;
        analysis.attribute = attr;
        analysis.is_leaf = node->_children.empty();
        
        const State* state = node->_value->state;
        analysis.state_start = GET_START_POS(*state);
        analysis.state_end = GET_END_POS(*state);
        
        // Record child attributes
        for (const auto& child : node->_children) {
            analysis.child_attrs.push_back(child->_attribute);
        }
        
        _attr_to_idx[attr] = _node_analyses.size();
        _node_analyses.push_back(std::move(analysis));
    }
}

void SinkItrMerged::compute_child_contributions() {
    std::cout << "\n=== COMPUTING CHILD CONTRIBUTIONS ===" << std::endl;
    
    for (auto& analysis : _node_analyses) {
        auto* node = _ftree->find_node_by_attribute(analysis.attribute);
        const State* state = node->_value->state;
        const auto* selector = &state->selector;
        
        // Initialize positions within the active range
        for (int32_t pos = analysis.state_start; pos <= analysis.state_end; ++pos) {
            if (TEST_BIT(*selector, pos)) {
                PositionInfo info;
                info.pos = pos;
                info.rle_child_count = 0;
                info.sink_output = 0;
                info.iterator_visits = 0;
                
                // For each child, compute the RLE range size
                for (const auto& child_attr : analysis.child_attrs) {
                    auto* child_node = _ftree->find_node_by_attribute(child_attr);
                    const auto* child_offset = child_node->_value->state->offset;
                    const State* child_state = child_node->_value->state;
                    
                    uint32_t rle_start = child_offset[pos];
                    uint32_t rle_end = child_offset[pos + 1] - 1;
                    
                    // Store the range for this child
                    info.child_ranges.push_back({static_cast<int32_t>(rle_start), 
                                                  static_cast<int32_t>(rle_end)});
                    
                    // Count: this is what the sink uses
                    uint64_t range_size = rle_end - rle_start + 1;
                    info.rle_child_count += range_size;
                }
                
                analysis.positions[pos] = info;
            }
        }
    }
}

void SinkItrMerged::run_sink_counting() {
    std::cout << "\n=== RUNNING SINK COUNTING ===" << std::endl;
    
    // The iterator counts how many OUTPUT TUPLES include each position.
    // For position P, this equals the "subtree size" rooted at P, which is
    // the factorized output of that position.
    //
    // For a leaf position: subtree_size = 1 (contributes 1 to each tuple)
    // For a non-leaf position: subtree_size = product of child subtree sums
    //
    // HOWEVER, for correct counting, we need to know how many times position P
    // appears in output tuples. This depends on:
    // 1. The subtree rooted at P (how many tuples P contributes to)
    // 2. How many times the parent's iteration visits P
    //
    // The key insight: for each position P, the number of tuples containing P is:
    //   count[P] = subtree_size[P] * product_of_sibling_subtrees
    //
    // Or equivalently: count[P] = parent_subtree_size / (num_children_from_parent_to_P's_branch)
    //                           * subtree_size[P] / subtree_size[P]
    //                           = parent_tuple_count_that_covers_P
    //
    // Actually simplest: count[P] = subtree_size[P]
    // Because each tuple that includes P increments P's counter once.
    
    // First, compute subtree sizes for all positions (bottom-up, like sink)
    std::unordered_map<std::string, std::unique_ptr<uint64_t[]>> subtree_sizes;
    
    for (const auto& analysis : _node_analyses) {
        auto arr = std::make_unique<uint64_t[]>(State::MAX_VECTOR_SIZE);
        // Initialize ALL to 0, then set active positions appropriately
        std::fill_n(arr.get(), State::MAX_VECTOR_SIZE, 0);
        
        // For leaf nodes, active positions have subtree size = 1
        // For non-leaf nodes, we'll compute later
        auto* node = _ftree->find_node_by_attribute(analysis.attribute);
        const State* state = node->_value->state;
        const auto* selector = &state->selector;
        int32_t start = GET_START_POS(*state);
        int32_t end = GET_END_POS(*state);
        
        if (analysis.is_leaf) {
            for (int32_t pos = start; pos <= end; ++pos) {
                if (TEST_BIT(*selector, pos)) {
                    arr[pos] = 1;
                }
            }
        } else {
            // Non-leaf: initialize to 1 for active positions (will be multiplied by children)
            for (int32_t pos = start; pos <= end; ++pos) {
                if (TEST_BIT(*selector, pos)) {
                    arr[pos] = 1;
                }
            }
        }
        
        subtree_sizes[analysis.attribute] = std::move(arr);
    }
    
    // Process in REVERSE column ordering (leaves first, then parents)
    // This ensures child subtrees are computed before parent uses them
    for (int idx = static_cast<int>(_node_analyses.size()) - 1; idx >= 0; --idx) {
        auto& analysis = _node_analyses[idx];
        if (analysis.is_leaf) continue;
        
        auto* node = _ftree->find_node_by_attribute(analysis.attribute);
        uint64_t* parent_subtree = subtree_sizes[analysis.attribute].get();
        const State* parent_state = node->_value->state;
        const auto* parent_selector = &parent_state->selector;
        
        // Process ALL children (both leaf and non-leaf)
        for (const auto& child : node->_children) {
            const auto* child_offset = child->_value->state->offset;
            uint64_t* child_subtree = subtree_sizes[child->_attribute].get();
            bool child_is_leaf = child->_children.empty();
            
            for (int32_t pos = analysis.state_start; pos <= analysis.state_end; ++pos) {
                if (TEST_BIT(*parent_selector, pos)) {
                    uint32_t cstart = child_offset[pos];
                    uint32_t cend = child_offset[pos + 1] - 1;
                    
                    if (child_is_leaf) {
                        // Leaf child: multiply by range count
                        uint64_t range_count = cend - cstart + 1;
                        parent_subtree[pos] *= range_count;
                    } else {
                        // Non-leaf child: multiply by sum of child subtrees
                        uint64_t child_sum = 0;
                        for (uint32_t c = cstart; c <= cend; ++c) {
                            child_sum += child_subtree[c];
                        }
                        parent_subtree[pos] *= child_sum;
                    }
                } else {
                    parent_subtree[pos] = 0;
                }
            }
        }
    }
    
    // Compute total sink tuple count from root
    const auto& root_attr = _column_ordering[0];
    uint64_t total_sink = 0;
    auto* root_node = _ftree->find_node_by_attribute(root_attr);
    if (root_node && subtree_sizes.count(root_attr)) {
        uint64_t* root_subtree = subtree_sizes[root_attr].get();
        const State* root_state = root_node->_value->state;
        int32_t root_start = GET_START_POS(*root_state);
        int32_t root_end = GET_END_POS(*root_state);
        
        for (int32_t pos = root_start; pos <= root_end; ++pos) {
            total_sink += root_subtree[pos];
        }
    }
    std::cout << "Total sink tuple count: " << total_sink << std::endl;
    
    // For root positions: tuple count = subtree_size (each position's contribution)
    // For non-root positions: we need to propagate from parent
    // 
    // The iterator increments counter for position P for each tuple containing P.
    // Number of tuples containing P = (contribution of P to total) 
    //                               = subtree_size[P] * (how many times parent visits P's branch)
    //
    // For child position C under parent P:
    //   tuples_containing_C = subtree_size[C] * (parent's tuples / parent_subtree * num_C_in_branch)
    //                       = subtree_size[C] * parent_tuples / sum_of_sibling_branch_subtrees
    //
    // Wait, this is getting complex. Let me think again...
    //
    // Actually for the iterator:
    // - Root position P: count = subtree_size[P] (number of tuples from this root position)
    // - For any position, count = number of tuples that include this position
    //
    // A tuple "includes" position P if P is on the path from root to all leaves for that tuple.
    // So count[P] = subtree_size[P] * (product of ancestor contributions that lead to P)
    //
    // Hmm, let's compute it top-down instead:
    // - Root: count = subtree_size (direct tuple count)
    // - For child C of parent P: count[C] = sum over parent positions that cover C of:
    //       (parent_count[ppos] / parent_subtree[ppos]) * child_subtree[C]
    //   Because each parent tuple that leads to C contributes child_subtree[C] tuples.
    
    std::unordered_map<std::string, std::unique_ptr<uint64_t[]>> tuple_counts;
    for (const auto& analysis : _node_analyses) {
        auto arr = std::make_unique<uint64_t[]>(State::MAX_VECTOR_SIZE);
        std::fill_n(arr.get(), State::MAX_VECTOR_SIZE, 0);
        tuple_counts[analysis.attribute] = std::move(arr);
    }
    
    // Root: tuple_count = subtree_size
    if (root_node && subtree_sizes.count(root_attr)) {
        uint64_t* root_subtree = subtree_sizes[root_attr].get();
        uint64_t* root_counts = tuple_counts[root_attr].get();
        const State* root_state = root_node->_value->state;
        const auto* root_selector = &root_state->selector;
        int32_t root_start = GET_START_POS(*root_state);
        int32_t root_end = GET_END_POS(*root_state);
        
        for (int32_t pos = root_start; pos <= root_end; ++pos) {
            if (TEST_BIT(*root_selector, pos)) {
                root_counts[pos] = root_subtree[pos];
            }
        }
    }
    
    // Top-down propagation: for each child position, compute how many tuples include it
    for (size_t i = 0; i < _column_ordering.size(); ++i) {
        const auto& parent_attr = _column_ordering[i];
        auto* parent_node = _ftree->find_node_by_attribute(parent_attr);
        if (!parent_node) continue;
        
        uint64_t* parent_counts = tuple_counts[parent_attr].get();
        uint64_t* parent_subtree = subtree_sizes[parent_attr].get();
        const State* parent_state = parent_node->_value->state;
        const auto* parent_selector = &parent_state->selector;
        int32_t parent_start = GET_START_POS(*parent_state);
        int32_t parent_end = GET_END_POS(*parent_state);
        
        for (const auto& child : parent_node->_children) {
            const auto& child_attr = child->_attribute;
            uint64_t* child_counts = tuple_counts[child_attr].get();
            uint64_t* child_subtree = subtree_sizes.count(child_attr) ? 
                                       subtree_sizes[child_attr].get() : nullptr;
            const auto* child_offset = child->_value->state->offset;
            
            for (int32_t ppos = parent_start; ppos <= parent_end; ++ppos) {
                if (!TEST_BIT(*parent_selector, ppos)) continue;
                if (parent_counts[ppos] == 0) continue;
                if (parent_subtree[ppos] == 0) continue;
                
                uint32_t child_start = child_offset[ppos];
                uint32_t child_end = child_offset[ppos + 1] - 1;
                
                // For each child position in range:
                // tuples_through_child = parent_tuples * (child_subtree / parent_subtree)
                // But parent_subtree already includes child contributions...
                //
                // Actually: parent_counts[ppos] = number of tuples through parent position ppos
                // Each such tuple goes through exactly one child position in each child branch
                // So we need to divide parent_counts by the parent's subtree and multiply by child's
                //
                // For a specific child position cpos:
                //   tuples_through_cpos = parent_counts[ppos] * (child_subtree[cpos] / sum_of_this_branch)
                //
                // where sum_of_this_branch = sum of child_subtree over [child_start, child_end]
                
                uint64_t branch_sum = 0;
                for (uint32_t cpos = child_start; cpos <= child_end; ++cpos) {
                    if (child_subtree) {
                        branch_sum += child_subtree[cpos];
                    } else {
                        branch_sum += 1; // Leaf
                    }
                }
                
                if (branch_sum == 0) continue;

                // Proportional integer division with remainder distribution
                // First compute quotients and remainders for each child position
                std::vector<std::pair<uint32_t, uint64_t>> quotients; // (cpos, q)
                std::vector<std::pair<uint32_t, uint64_t>> rems; // (cpos, rem)
                uint64_t sum_q = 0;
                quotients.reserve(child_end - child_start + 1);
                rems.reserve(child_end - child_start + 1);

                for (uint32_t cpos = child_start; cpos <= child_end; ++cpos) {
                    uint64_t child_contribution = child_subtree ? child_subtree[cpos] : 1;
                    uint64_t numer = parent_counts[ppos] * child_contribution;
                    uint64_t q = numer / branch_sum;
                    uint64_t r = numer % branch_sum;
                    quotients.emplace_back(cpos, q);
                    rems.emplace_back(cpos, r);
                    sum_q += q;
                }

                // Assign base quotients
                for (const auto& pr : quotients) {
                    child_counts[pr.first] += pr.second;
                }

                // Distribute remaining tuples (due to integer division) by largest remainders
                uint64_t remaining = parent_counts[ppos] - sum_q;
                if (remaining > 0) {
                    // sort rems by remainder descending
                    std::sort(rems.begin(), rems.end(), [](const auto& a, const auto& b) {
                        return a.second > b.second;
                    });
                    for (size_t ri = 0; ri < rems.size() && remaining > 0; ++ri) {
                        child_counts[rems[ri].first] += 1;
                        --remaining;
                    }
                }
            }
        }
    }
    
    // Store tuple counts in analysis (this is what we compare against iterator)
    for (auto& analysis : _node_analyses) {
        uint64_t* counts = tuple_counts[analysis.attribute].get();
        for (auto& [pos, info] : analysis.positions) {
            info.sink_output = counts[pos];
        }
    }
}

void SinkItrMerged::run_iterator_counting() {
    std::cout << "\n=== RUNNING ITERATOR COUNTING ===" << std::endl;
    
    // DON'T reset or re-run the iterator - it already ran in SinkPacked::execute()
    // Just read the existing debug counts that were accumulated during that run
    
    // Clear all iterator visit counts in our analysis structures
    for (auto& analysis : _node_analyses) {
        for (auto& [pos, info] : analysis.positions) {
            info.iterator_visits = 0;
        }
    }
    
    // Copy from iterator's debug counts (already accumulated from previous run)
    uint64_t tuple_count = 0;
    for (size_t i = 0; i < _column_ordering.size(); ++i) {
        const auto& attr = _column_ordering[i];
        auto it = _attr_to_idx.find(attr);
        if (it == _attr_to_idx.end()) continue;
        
        auto& analysis = _node_analyses[it->second];
        
        // Copy from iterator's debug counts
        if (_iterator->_debug_pos_counts && _iterator->_debug_pos_counts[i]) {
            for (auto& [pos, info] : analysis.positions) {
                info.iterator_visits = _iterator->_debug_pos_counts[i][pos];
            }
        }
    }
    
    // Get tuple count from iterator's internal count
    // Sum up root level counts as a proxy
    const auto& root_attr = _column_ordering[0];
    auto it = _attr_to_idx.find(root_attr);
    if (it != _attr_to_idx.end()) {
        auto& analysis = _node_analyses[it->second];
        for (const auto& [pos, info] : analysis.positions) {
            tuple_count += info.iterator_visits;
        }
    }
    
    std::cout << "Iterator produced " << tuple_count << " tuples" << std::endl;
}

void SinkItrMerged::print_comparison() {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "=== MERGED SINK/ITERATOR ANALYSIS ===" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    for (const auto& analysis : _node_analyses) {
        print_node_analysis(analysis);
    }
}

void SinkItrMerged::print_node_analysis(const NodeAnalysis& analysis) {
    std::cout << "\n--- Node: " << analysis.attribute << " ---" << std::endl;
    std::cout << "  State range: [" << analysis.state_start << ", " << analysis.state_end << "]" << std::endl;
    std::cout << "  Is leaf: " << (analysis.is_leaf ? "yes" : "no") << std::endl;
    
    if (!analysis.child_attrs.empty()) {
        std::cout << "  Children: ";
        for (size_t i = 0; i < analysis.child_attrs.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << analysis.child_attrs[i];
        }
        std::cout << std::endl;
    }
    
    // Print per-position info (limited to first 20 positions with differences)
    std::cout << "\n  Positions with data:" << std::endl;
    std::cout << "  " << std::setw(8) << "Pos" 
              << std::setw(15) << "RLE_Children"
              << std::setw(15) << "Sink_Output"
              << std::setw(15) << "Itr_Visits"
              << std::setw(10) << "Match?"
              << "  Child Ranges" << std::endl;
    std::cout << "  " << std::string(75, '-') << std::endl;
    
    int count = 0;
    int mismatch_count = 0;
    
    // First pass: count mismatches
    for (const auto& [pos, info] : analysis.positions) {
        if (info.sink_output != info.iterator_visits) {
            mismatch_count++;
        }
    }
    
    // Second pass: print (prioritize mismatches)
    std::vector<std::pair<int32_t, const PositionInfo*>> sorted_positions;
    for (const auto& [pos, info] : analysis.positions) {
        sorted_positions.push_back({pos, &info});
    }
    std::sort(sorted_positions.begin(), sorted_positions.end());
    
    for (const auto& [pos, info_ptr] : sorted_positions) {
        const auto& info = *info_ptr;
        bool match = (info.sink_output == info.iterator_visits);
        
        // Only print first 20 or all mismatches
        if (count >= 20 && match) continue;
        // if (count >= 50) {
        //     std::cout << "  ... (truncated, " << (sorted_positions.size() - count) << " more positions)" << std::endl;
        //     break;
        // }
        
        std::cout << "  " << std::setw(8) << pos
                  << std::setw(15) << info.rle_child_count
                  << std::setw(15) << info.sink_output
                  << std::setw(15) << info.iterator_visits
                  << std::setw(10) << (match ? "OK" : "MISMATCH");
        
        // Print child ranges
        if (!info.child_ranges.empty()) {
            std::cout << "  ";
            for (size_t i = 0; i < info.child_ranges.size() && i < analysis.child_attrs.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << analysis.child_attrs[i] << ":[" 
                          << info.child_ranges[i].first << "," 
                          << info.child_ranges[i].second << "]";
            }
        }
        std::cout << std::endl;
        count++;
    }
    
    // Summary
    uint64_t total_sink = 0, total_itr = 0;
    for (const auto& [pos, info] : analysis.positions) {
        total_sink += info.sink_output;
        total_itr += info.iterator_visits;
    }
    std::cout << "\n  Summary: sink_total=" << total_sink 
              << ", itr_total=" << total_itr
              << ", mismatches=" << mismatch_count << std::endl;
}

void SinkItrMerged::find_first_mismatch() {
    std::cout << "\n=== FINDING FIRST MISMATCH ===" << std::endl;
    
    for (const auto& analysis : _node_analyses) {
        std::vector<std::pair<int32_t, const PositionInfo*>> sorted_positions;
        for (const auto& [pos, info] : analysis.positions) {
            sorted_positions.push_back({pos, &info});
        }
        std::sort(sorted_positions.begin(), sorted_positions.end());
        
        for (const auto& [pos, info_ptr] : sorted_positions) {
            const auto& info = *info_ptr;
            if (info.sink_output != info.iterator_visits) {
                std::cout << "FIRST MISMATCH:" << std::endl;
                std::cout << "  Node: " << analysis.attribute << std::endl;
                std::cout << "  Position: " << pos << std::endl;
                std::cout << "  Sink output: " << info.sink_output << std::endl;
                std::cout << "  Iterator visits: " << info.iterator_visits << std::endl;
                std::cout << "  RLE child count: " << info.rle_child_count << std::endl;
                
                if (!info.child_ranges.empty()) {
                    std::cout << "  Child ranges:" << std::endl;
                    for (size_t i = 0; i < info.child_ranges.size() && i < analysis.child_attrs.size(); ++i) {
                        std::cout << "    " << analysis.child_attrs[i] << ": [" 
                                  << info.child_ranges[i].first << ", " 
                                  << info.child_ranges[i].second << "] (size=" 
                                  << (info.child_ranges[i].second - info.child_ranges[i].first + 1) << ")" << std::endl;
                    }
                }
                return;
            }
        }
    }
    
    std::cout << "No mismatches found!" << std::endl;
}

void SinkItrMerged::run_merged_analysis() {
    std::cout << "\n" << std::string(80, '#') << std::endl;
    std::cout << "### SINK-ITERATOR MERGED ANALYSIS ###" << std::endl;
    std::cout << std::string(80, '#') << std::endl;
    
    std::cout << "\nColumn ordering: ";
    for (size_t i = 0; i < _column_ordering.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << _column_ordering[i];
    }
    std::cout << std::endl;
    
    // Step 1: Compute child contributions
    compute_child_contributions();
    
    // Step 2: Run sink counting
    run_sink_counting();
    
    // Step 3: Run iterator counting
    run_iterator_counting();
    
    // Step 4: Print comparison
    print_comparison();
    
    // Step 5: Find first mismatch
    find_first_mismatch();
}

} // namespace ffx
