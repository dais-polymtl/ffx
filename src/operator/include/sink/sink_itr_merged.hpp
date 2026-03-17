#ifndef FFX_SINK_ITR_MERGED_HH
#define FFX_SINK_ITR_MERGED_HH

#include "../factorized_ftree/factorized_tree_element.hpp"
#include "../factorized_ftree/ftree_iterator.hpp"
#include "sink_packed.hpp"
#include <memory>
#include <vector>
#include <unordered_map>

namespace ffx {

/**
 * SinkItrMerged: A debugging class that runs the sink counting and iterator
 * together step-by-step, printing detailed logs for each state.
 * 
 * For each position in each vector, it tracks:
 * - How many children that position contributes to (from RLE)
 * - The actual count from sink's output registry
 * - The number of times the iterator visits that position
 */
class SinkItrMerged {
public:
    SinkItrMerged();
    ~SinkItrMerged() = default;
    
    /**
     * Initialize with the same schema as the sink
     * @param schema The query schema
     * @param ftree The factorized tree root
     * @param ordered_nodes Topologically ordered nodes (non-leaf first)
     * @param existing_iterator The iterator that was already run (to get debug counts from)
     */
    void init(Schema* schema, 
              std::shared_ptr<FactorizedTreeElement> ftree,
              const std::vector<FactorizedTreeElement*>& ordered_nodes,
              FTreeIterator* existing_iterator);
    
    /**
     * Run merged analysis and print detailed logs
     * Compares sink counting vs iterator traversal step by step
     */
    void run_merged_analysis();

private:
    // Structure to track per-position contribution info
    struct PositionInfo {
        int32_t pos;
        uint64_t rle_child_count;     // Number of children this pos contributes to (from RLE)
        uint64_t sink_output;         // Final sink output count for this position
        uint64_t iterator_visits;     // Number of times iterator visited this position
        std::vector<std::pair<int32_t, int32_t>> child_ranges; // (start, end) for each child's RLE range
    };
    
    // Structure to hold per-node analysis
    struct NodeAnalysis {
        std::string attribute;
        int32_t state_start;
        int32_t state_end;
        bool is_leaf;
        std::vector<std::string> child_attrs;
        std::unordered_map<int32_t, PositionInfo> positions; // pos -> info
    };
    
    /**
     * Compute child contribution counts for each position in each node
     * For each parent position, how many positions does each child have in its RLE range
     */
    void compute_child_contributions();
    
    /**
     * Run the sink counting logic and capture per-position outputs
     */
    void run_sink_counting();
    
    /**
     * Run the iterator and count visits per position
     */
    void run_iterator_counting();
    
    /**
     * Print detailed comparison for all nodes
     */
    void print_comparison();
    
    /**
     * Print a single node's detailed analysis
     */
    void print_node_analysis(const NodeAnalysis& analysis);
    
    /**
     * Find first position where sink and iterator disagree
     */
    void find_first_mismatch();

    // Member variables
    std::shared_ptr<FactorizedTreeElement> _ftree;
    std::vector<FactorizedTreeElement*> _ordered_nodes;
    std::vector<NodeAnalysis> _node_analyses;
    
    // Map from attribute name to node analysis index
    std::unordered_map<std::string, size_t> _attr_to_idx;
    
    // Iterator for traversal (pointer to existing, not owned)
    FTreeIterator* _iterator;
    Schema* _schema;
    
    // Column ordering (BFS order from iterator)
    std::vector<std::string> _column_ordering;
    std::unordered_map<std::string, size_t> _attr_to_col_idx;
};

} // namespace ffx

#endif // FFX_SINK_ITR_MERGED_HH
