#ifndef VFENGINE_PACKED_ANTI_SEMI_JOIN_OPERATOR_HH
#define VFENGINE_PACKED_ANTI_SEMI_JOIN_OPERATOR_HH

#include "../../table/include/adj_list.hpp"
#include "../../table/include/ffx_str_t.hpp"
#include "../factorized_ftree/ftree_ancestor_finder.hpp"
#include "../factorized_ftree/ftree_state_update.hpp"
#include "operator.hpp"
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

namespace ffx {

template<typename T = uint64_t>
struct PackedAntiSemiJoinVectorSliceUpdateSavedData {
    std::string attribute;
    Vector<T>* vector;
    std::pair<int32_t, int32_t> backup_state;
    PackedAntiSemiJoinVectorSliceUpdateSavedData() : vector(nullptr), backup_state(-1, -1) {}
    PackedAntiSemiJoinVectorSliceUpdateSavedData(const std::string& attr, Vector<T>* vec, int32_t start, int32_t end)
        : attribute(attr), vector(vec), backup_state(start, end) {}
};

/**
 * PackedAntiSemiJoin - Anti-Semi-Join Filter (NOT EXISTS check)
 * 
 * This operator filters out tuples where an edge EXISTS between two attributes.
 * It is used for patterns like:
 *   WHERE NOT (person1)-[:KNOWS]->(person3)
 * 
 * For each (left_val, right_val) pair:
 *   - Look up left_val's adjacency list in the specified edge table
 *   - Binary search for right_val in the adjacency list
 *   - If FOUND (edge exists): remove the tuple (clear bit)
 *   - If NOT FOUND (no edge): keep the tuple
 * 
 * The left_attr is the ANCESTOR and right_attr is the DESCENDANT in the factorized tree.
 * The operator filters based on the right_attr's positions.
 * 
 * Example: For "NOT (person1)-[:KNOWS]->(person3)" where person1 is ancestor of person3,
 * the operator will check if person3's value exists in person1's adjacency list,
 * and filter out rows where such an edge exists.
 */
template<typename T = uint64_t>
class PackedAntiSemiJoin final : public Operator {
public:
    PackedAntiSemiJoin() = delete;
    PackedAntiSemiJoin(const PackedAntiSemiJoin&) = delete;

    /**
     * Constructor
     * @param left_attr The source attribute (ancestor in ftree)
     * @param right_attr The target attribute (descendant in ftree, will be filtered)
     */
    PackedAntiSemiJoin(std::string left_attr, std::string right_attr)
        : Operator(), _left_attr(std::move(left_attr)), _right_attr(std::move(right_attr)), _left_vec(nullptr),
          _right_vec(nullptr), _adj_lists(nullptr), _ancestor_finder(nullptr), _range_update_tree(nullptr),
          _vector_saved_data(nullptr) {}

    void init(Schema* schema) override;
    void execute() override;

    // Getters for testing/printing
    const std::string& left_attr() const { return _left_attr; }
    const std::string& right_attr() const { return _right_attr; }

private:
    void create_slice_update_infrastructure(FactorizedTreeElement* ftree_right_node);
    void store_slices();
    void restore_slices();

    // Process a slice of descendants for a given source value using linear merge
    // Updates bitmask and tracks new_right_start/end
    void process_slice(T source_val, const T* right_vals, int32_t slice_start, int32_t slice_end,
                       int32_t& new_right_start, int32_t& new_right_end);

    // Data members
    const std::string _left_attr; // Source attribute (ancestor)
    const std::string _right_attr;// Target attribute (descendant, filtered)

    Vector<T>* _left_vec;  // Source (left) attribute vector
    Vector<T>* _right_vec; // Target (right) attribute vector (the one being filtered)
    AdjList<T>* _adj_lists;// Adjacency lists from the edge table

    // Bitmask for tracking valid right_attr positions
    std::unique_ptr<BitMask<State::MAX_VECTOR_SIZE>> _right_valid_mask_uptr;
    BitMask<State::MAX_VECTOR_SIZE> _right_selector_backup;

    // Flag: true if left and right are in the same DataChunk (identity mapping)
    bool _same_data_chunk = false;
    // FtreeAncestorFinder to map right_attr indices to left_attr indices
    // Only created if _same_data_chunk is false
    std::unique_ptr<FtreeAncestorFinder> _ancestor_finder;

    // Ftree state update infrastructure
    std::unique_ptr<FtreeStateUpdateNode> _range_update_tree;
    std::unique_ptr<PackedAntiSemiJoinVectorSliceUpdateSavedData<T>[]> _vector_saved_data;
    std::size_t _vector_saved_data_count = 0;
};

// Type aliases for convenience
using PackedAntiSemiJoinUint64 = PackedAntiSemiJoin<uint64_t>;
using PackedAntiSemiJoinString = PackedAntiSemiJoin<ffx_str_t>;

}// namespace ffx

#endif// VFENGINE_PACKED_ANTI_SEMI_JOIN_OPERATOR_HH
