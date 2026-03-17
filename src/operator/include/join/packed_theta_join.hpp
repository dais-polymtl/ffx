#ifndef VFENGINE_PACKED_THETA_JOIN_OPERATOR_HH
#define VFENGINE_PACKED_THETA_JOIN_OPERATOR_HH

#include "../../table/include/ffx_str_t.hpp"
#include "../factorized_ftree/ftree_ancestor_finder.hpp"
#include "../factorized_ftree/ftree_state_update.hpp"
#include "../predicate/predicate_eval.hpp"
#include "operator.hpp"
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

namespace ffx {

template<typename T = uint64_t>
struct PackedThetaJoinVectorSliceUpdateSavedData {
    std::string attribute;
    Vector<T>* vector;
    std::pair<int32_t, int32_t> backup_state;
    PackedThetaJoinVectorSliceUpdateSavedData() : vector(nullptr), backup_state(-1, -1) {}
    PackedThetaJoinVectorSliceUpdateSavedData(const std::string& attr, Vector<T>* vec, int32_t start, int32_t end)
        : attribute(attr), vector(vec), backup_state(start, end) {}
};

/**
 * PackedThetaJoin - Attribute-to-Attribute Predicate Filter
 * 
 * This operator compares values from two attributes and filters rows
 * where the predicate evaluates to false. The predicate is of the form:
 *   left_attr <op> right_attr
 * where <op> is one of: ==, !=, <, >, <=, >=
 * 
 * The left_attr is the ANCESTOR and right_attr is the DESCENDANT in the factorized tree.
 * The operator filters based on the right_attr's values, marking positions as invalid 
 * where the predicate fails.
 * 
 * Example: For predicate "a < b" where a is an ancestor of b,
 * the operator will iterate through each b index, find the corresponding a index
 * using FtreeAncestorFinder, and filter out b values where a >= b.
 */
template<typename T = uint64_t>
class PackedThetaJoin final : public Operator {
public:
    PackedThetaJoin() = delete;
    PackedThetaJoin(const PackedThetaJoin&) = delete;

    /**
     * Constructor
     * @param left_attr The ancestor attribute (on the left side of the predicate)
     * @param right_attr The descendant attribute (on the right side, will be filtered)
     * @param op The comparison operator
     */
    PackedThetaJoin(std::string left_attr, std::string right_attr, PredicateOp op)
        : Operator(), _left_attr(std::move(left_attr)), _right_attr(std::move(right_attr)), _op(op),
          _pred_fn(get_predicate_fn<T>(op)), _left_vec(nullptr), _right_vec(nullptr), _ancestor_finder(nullptr),
          _range_update_tree(nullptr), _vector_saved_data(nullptr) {}

    void init(Schema* schema) override;
    void execute() override;

    // Getters for testing
    const std::string& left_attr() const { return _left_attr; }
    const std::string& right_attr() const { return _right_attr; }
    PredicateOp op() const { return _op; }

private:
    void create_slice_update_infrastructure(FactorizedTreeElement* ftree_right_node);
    void store_slices();
    void restore_slices();

    // Data members
    const std::string _left_attr; // Ancestor attribute
    const std::string _right_attr;// Descendant attribute (the one being filtered)
    PredicateOp _op;
    PredicateFn<T> _pred_fn;

    Vector<T>* _left_vec; // Ancestor (left) attribute vector
    Vector<T>* _right_vec;// Descendant (right) attribute vector (the one being filtered)

    // Bitmask for tracking valid right_attr positions
    std::unique_ptr<BitMask<State::MAX_VECTOR_SIZE>> _right_valid_mask_uptr;
    BitMask<State::MAX_VECTOR_SIZE> _right_selector_backup;

    // Flag: true if ancestor and descendant are in the same DataChunk (identity mapping)
    bool _same_data_chunk = false;

    // FtreeAncestorFinder to map right_attr (descendant) indices to left_attr (ancestor) indices
    // Only created if _same_data_chunk is false (different DataChunks)
    std::unique_ptr<FtreeAncestorFinder> _ancestor_finder;

    // Ftree state update infrastructure
    std::unique_ptr<FtreeStateUpdateNode> _range_update_tree;
    std::unique_ptr<PackedThetaJoinVectorSliceUpdateSavedData<T>[]> _vector_saved_data;
    std::size_t _vector_saved_data_count = 0;

    // Cascade propagation: track invalidated indices for sibling propagation
    std::unique_ptr<uint32_t[]> _invalidated_indices;
    int32_t _invalidated_count = 0;
};

// Type aliases for convenience
using PackedThetaJoinUint64 = PackedThetaJoin<uint64_t>;
using PackedThetaJoinString = PackedThetaJoin<ffx_str_t>;

}// namespace ffx

#endif// VFENGINE_PACKED_THETA_JOIN_OPERATOR_HH
