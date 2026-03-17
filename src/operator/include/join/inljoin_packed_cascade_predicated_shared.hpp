#ifndef VFENGINE_INDEX_NESTED_LOOP_JOIN_PACKED_CASCADE_PREDICATED_SHARED_OPERATOR_HH
#define VFENGINE_INDEX_NESTED_LOOP_JOIN_PACKED_CASCADE_PREDICATED_SHARED_OPERATOR_HH

#include "../../table/include/ffx_str_t.hpp"
#include "../factorized_ftree/ftree_state_update.hpp"
#include "../predicate/predicate_eval.hpp"
#include "inljoin_packed_cascade_shared.hpp"// For VectorSavedData
#include "operator.hpp"
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace ffx {

template<typename T = uint64_t>
class INLJoinPackedCascadePredicatedShared final : public Operator {
public:
    INLJoinPackedCascadePredicatedShared() = delete;
    INLJoinPackedCascadePredicatedShared(const INLJoinPackedCascadePredicatedShared&) = delete;

    // Constructor with predicate expression
    INLJoinPackedCascadePredicatedShared(std::string join_key, std::string output_key, bool is_join_index_fwd,
                                         PredicateExpression predicate_expr)
        : Operator(), _join_key(std::move(join_key)), _output_key(std::move(output_key)),
          _is_join_index_fwd(is_join_index_fwd), _in_vec(nullptr), _out_vec(nullptr), _adj_lists(nullptr),
          _range_update_tree(nullptr), _vector_saved_data(nullptr), _invalidated_indices(nullptr),
          _invalidated_count(0), _predicate_expr_raw(std::move(predicate_expr)) {}

    // Constructor without predicate
    INLJoinPackedCascadePredicatedShared(std::string join_key, std::string output_key, bool is_join_index_fwd)
        : Operator(), _join_key(std::move(join_key)), _output_key(std::move(output_key)),
          _is_join_index_fwd(is_join_index_fwd), _in_vec(nullptr), _out_vec(nullptr), _adj_lists(nullptr),
          _range_update_tree(nullptr), _vector_saved_data(nullptr), _invalidated_indices(nullptr),
          _invalidated_count(0) {}

    void init(Schema* schema) override;
    void execute() override;

    const std::string& join_key() const { return _join_key; }
    const std::string& output_key() const { return _output_key; }
    bool is_join_index_fwd() const { return _is_join_index_fwd; }
    bool has_predicate() const { return _predicate_expr_raw.has_predicates(); }
    std::string predicate_string() const { return _predicate_expr_raw.to_string(); }

private:
    void create_slice_update_infrastructure(FactorizedTreeElement* ftree_leaf);
    void store_slices();
    void restore_slices();

    const std::string _join_key, _output_key;
    bool _is_join_index_fwd;
    Vector<T>* _in_vec;
    Vector<T>* _out_vec;
    AdjList<T>* _adj_lists;
    std::unique_ptr<BitMask<State::MAX_VECTOR_SIZE>> _active_mask_uptr;
    BitMask<State::MAX_VECTOR_SIZE> _in_selector_backup;
    std::unique_ptr<FtreeStateUpdateNode> _range_update_tree;
    std::unique_ptr<VectorSavedData<T>[]> _vector_saved_data;
    std::size_t _vector_saved_data_count = 0;
    std::unique_ptr<uint32_t[]> _invalidated_indices;
    int32_t _invalidated_count = 0;
    PredicateExpression _predicate_expr_raw;

    bool _is_string_predicate = false;
    ScalarPredicateExpression<T> _predicate_expr_numeric;
    ScalarPredicateExpression<ffx_str_t> _predicate_expr_string;
};

using INLJoinPackedCascadePredicatedSharedUint64 = INLJoinPackedCascadePredicatedShared<uint64_t>;
using INLJoinPackedCascadePredicatedSharedString = INLJoinPackedCascadePredicatedShared<ffx_str_t>;

}// namespace ffx

#endif
