#ifndef VFENGINE_INTERSECTION_PREDICATED_OPERATOR_HH
#define VFENGINE_INTERSECTION_PREDICATED_OPERATOR_HH

#include "../../table/include/ffx_str_t.hpp"
#include "../factorized_ftree/ftree_ancestor_finder.hpp"
#include "../factorized_ftree/ftree_state_update.hpp"
#include "../predicate/predicate_eval.hpp"
#include "operator.hpp"
#include <cstring>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace ffx {

template<typename T = uint64_t>
struct IntersectionPredicatedVectorSliceUpdateSavedData {
    std::string attribute;
    Vector<T>* vector;
    std::pair<int32_t, int32_t> backup_state;
    IntersectionPredicatedVectorSliceUpdateSavedData() : vector(nullptr), backup_state(-1, -1) {}
    IntersectionPredicatedVectorSliceUpdateSavedData(const std::string& attr, Vector<T>* vec, int32_t start,
                                                     int32_t end)
        : attribute(attr), vector(vec), backup_state(start, end) {}
};

template<typename T = uint64_t>
class IntersectionPredicated final : public Operator {
public:
    IntersectionPredicated() = delete;
    IntersectionPredicated(const IntersectionPredicated&) = delete;

    IntersectionPredicated(std::string ancestor_attr, std::string descendant_attr, std::string output_attr,
                           bool is_ancestor_join_fwd, bool is_descendant_join_fwd, PredicateExpression predicate_expr)
        : Operator(), _ancestor_attr(std::move(ancestor_attr)), _descendant_attr(std::move(descendant_attr)),
          _output_attr(std::move(output_attr)), _is_ancestor_join_fwd(is_ancestor_join_fwd),
          _is_descendant_join_fwd(is_descendant_join_fwd), _ancestor_vec(nullptr), _descendant_vec(nullptr),
          _out_vec(nullptr), _descendant_offset(nullptr), _ancestor_adj_lists(nullptr), _descendant_adj_lists(nullptr),
          _range_update_tree(nullptr), _vector_saved_data(nullptr), _predicate_expr_raw(std::move(predicate_expr)) {}

    IntersectionPredicated(std::string ancestor_attr, std::string descendant_attr, std::string output_attr,
                           bool is_ancestor_join_fwd, bool is_descendant_join_fwd)
        : Operator(), _ancestor_attr(std::move(ancestor_attr)), _descendant_attr(std::move(descendant_attr)),
          _output_attr(std::move(output_attr)), _is_ancestor_join_fwd(is_ancestor_join_fwd),
          _is_descendant_join_fwd(is_descendant_join_fwd), _ancestor_vec(nullptr), _descendant_vec(nullptr),
          _out_vec(nullptr), _descendant_offset(nullptr), _ancestor_adj_lists(nullptr), _descendant_adj_lists(nullptr),
          _range_update_tree(nullptr), _vector_saved_data(nullptr) {}

    void init(Schema* schema) override;
    void execute() override;

    const std::string& ancestor_attr() const { return _ancestor_attr; }
    const std::string& descendant_attr() const { return _descendant_attr; }
    const std::string& output_attr() const { return _output_attr; }
    bool is_ancestor_join_fwd() const { return _is_ancestor_join_fwd; }
    bool is_descendant_join_fwd() const { return _is_descendant_join_fwd; }
    bool has_predicate() const { return _predicate_expr_raw.has_predicates(); }
    std::string predicate_string() const { return _predicate_expr_raw.to_string(); }

private:
    void create_slice_update_infrastructure(const FactorizedTreeElement* ftree_output_node);
    void store_slices();
    void restore_slices();

    __attribute__((always_inline)) inline void process_data_chunk(BitMask<State::MAX_VECTOR_SIZE>* current_ip_mask,
                                                                  int32_t op_filled_idx);

    // Compute intersection without caching
    uint32_t get_intersection(T a_val, T b_val, const AdjList<T>& ancestor_adj_list,
                              const AdjList<T>& descendant_adj_list, T* dest_buffer, int32_t max_dest_size);

    const std::string _ancestor_attr, _descendant_attr, _output_attr;
    bool _is_ancestor_join_fwd, _is_descendant_join_fwd;
    Vector<T>* _ancestor_vec;
    Vector<T>* _descendant_vec;
    Vector<T>* _out_vec;
    uint16_t* _descendant_offset;
    AdjList<T>* _ancestor_adj_lists;
    AdjList<T>* _descendant_adj_lists;
    std::unique_ptr<BitMask<State::MAX_VECTOR_SIZE>> _descendant_valid_mask_uptr;
    BitMask<State::MAX_VECTOR_SIZE> _descendant_selector_backup;
    BitMask<State::MAX_VECTOR_SIZE> _current_ip_mask;
    std::unique_ptr<FtreeStateUpdateNode> _range_update_tree;
    std::unique_ptr<IntersectionPredicatedVectorSliceUpdateSavedData<T>[]> _vector_saved_data;
    std::size_t _vector_saved_data_count = 0;
    // Ancestor index buffer - maps descendant indices to ancestor indices
    std::unique_ptr<uint32_t[]> _ancestor_index_buffer;
    // Flag: true if ancestor and descendant are in the same DataChunk (identity mapping)
    bool _same_data_chunk = false;
    // Ancestor finder - maps descendant indices to ancestor indices (handles both direct and multi-hop)
    // Only created if _same_data_chunk is false
    std::unique_ptr<FtreeAncestorFinder> _ancestor_finder;
    PredicateExpression _predicate_expr_raw;

    bool _is_string_predicate = false;
    ScalarPredicateExpression<T> _predicate_expr_numeric;
    ScalarPredicateExpression<ffx_str_t> _predicate_expr_string;

    // Cascade propagation: track invalidated indices for sibling propagation
    std::unique_ptr<uint32_t[]> _invalidated_indices;
    int32_t _invalidated_count = 0;
};

using IntersectionPredicatedUint64 = IntersectionPredicated<uint64_t>;
using IntersectionPredicatedString = IntersectionPredicated<ffx_str_t>;

}// namespace ffx

#endif
