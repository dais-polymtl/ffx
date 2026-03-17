#ifndef VFENGINE_NWAY_INTERSECTION_PREDICATED_OPERATOR_HH
#define VFENGINE_NWAY_INTERSECTION_PREDICATED_OPERATOR_HH

#include "../../table/include/adj_list.hpp"
#include "../../table/include/ffx_str_t.hpp"
#include "../factorized_ftree/ftree_ancestor_finder.hpp"
#include "../factorized_ftree/ftree_state_update.hpp"
#include "../predicate/predicate_eval.hpp"
#include "../vector/vector.hpp"
#include "operator.hpp"
#include <cstring>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace ffx {

template<typename T = uint64_t>
struct NWayIntersectionPredicatedVectorSliceUpdateSavedData {
    std::string attribute;
    Vector<T>* vector;
    std::pair<int32_t, int32_t> backup_state;// (start, end) positions
    NWayIntersectionPredicatedVectorSliceUpdateSavedData() : vector(nullptr), backup_state(-1, -1) {}
    NWayIntersectionPredicatedVectorSliceUpdateSavedData(const std::string& attr, Vector<T>* vec, int32_t start,
                                                         int32_t end)
        : attribute(attr), vector(vec), backup_state(start, end) {}
};

template<typename T = uint64_t>
class NWayIntersectionPredicated final : public Operator {
public:
    NWayIntersectionPredicated() = delete;
    NWayIntersectionPredicated(const NWayIntersectionPredicated&) = delete;

    // Constructor with predicate
    NWayIntersectionPredicated(std::string output_attr,
                               std::vector<std::pair<std::string, bool>> input_attrs_and_directions,
                               PredicateExpression predicate_expr);

    // Constructor without predicate
    NWayIntersectionPredicated(std::string output_attr,
                               std::vector<std::pair<std::string, bool>> input_attrs_and_directions);

    void init(Schema* schema) override;
    void execute() override;
    int32_t compute_sorted_intersection(const T* arr1, int32_t size1, const T* arr2, int32_t size2, T* dest);

    // Getters for testing
    const std::string& output_attr() const { return _output_attr; }
    const std::vector<std::pair<std::string, bool>>& input_attrs_and_directions() const {
        return _input_attrs_and_directions;
    }
    bool has_predicate() const { return _predicate_expr_raw.has_predicates(); }
    std::string predicate_string() const { return _predicate_expr_raw.to_string(); }

private:
    void create_slice_update_infrastructure(const FactorizedTreeElement* ftree_output_node);
    void store_slices();
    void restore_slices();

    __attribute__((always_inline)) inline void process_data_chunk(BitMask<State::MAX_VECTOR_SIZE>* _current_ip_mask,
                                                                  int32_t op_filled_idx);

    // Data members
    std::string _output_attr;
    std::vector<std::pair<std::string, bool>> _input_attrs_and_directions;// (attribute, is_fwd)

    // Vectors: one per input attribute + output
    std::vector<Vector<T>*> _input_vecs;
    Vector<T>* _out_vec;

    // Adjacency lists: one per input attribute
    std::vector<AdjList<T>*> _adj_lists;

    // RLE arrays: one per input attribute (for chunking)
    std::vector<uint16_t*> _offset_arrays;

    // Bitmask: only for the last input attribute (since only it is modified)
    std::unique_ptr<BitMask<State::MAX_VECTOR_SIZE>> _last_input_valid_mask_uptr;
    BitMask<State::MAX_VECTOR_SIZE> _last_input_selector_backup;
    BitMask<State::MAX_VECTOR_SIZE> _current_ip_mask;

    // Ftree state update infrastructure
    std::unique_ptr<FtreeStateUpdateNode> _range_update_tree;
    std::unique_ptr<NWayIntersectionPredicatedVectorSliceUpdateSavedData<T>[]> _vector_saved_data;
    std::size_t _vector_saved_data_count = 0;

    // Predicate expression for output attribute
    PredicateExpression _predicate_expr_raw;

    bool _is_string_predicate = false;
    ScalarPredicateExpression<T> _predicate_expr_numeric;
    ScalarPredicateExpression<ffx_str_t> _predicate_expr_string;

    // Flag: true if all inputs are in the same DataChunk (identity mapping)
    bool _all_same_data_chunk = false;
    // Multi-ancestor finder: maps from last input to ALL levels in the full path
    // Only created if _all_same_data_chunk is false
    std::unique_ptr<FtreeMultiAncestorFinder> _multi_ancestor_finder;
    // Ancestor index buffers: one per level in the full path (excluding last)
    std::vector<std::unique_ptr<uint32_t[]>> _ancestor_index_buffers;
    // Raw pointers for passing to process()
    std::vector<uint32_t*> _ancestor_index_buffer_ptrs;
    // Maps input index (0..num_inputs-1) to level index in full path
    std::vector<size_t> _input_level_indices;

    // Cascade propagation: track invalidated indices for sibling propagation
    std::unique_ptr<uint32_t[]> _invalidated_indices;
    int32_t _invalidated_count = 0;
};

// Type aliases for convenience
using NWayIntersectionPredicatedUint64 = NWayIntersectionPredicated<uint64_t>;
using NWayIntersectionPredicatedString = NWayIntersectionPredicated<ffx_str_t>;

}// namespace ffx

#endif
