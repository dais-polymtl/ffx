#ifndef VFENGINE_INDEX_NESTED_LOOP_JOIN_PACKED_PREDICATED_SHARED_OPERATOR_HH
#define VFENGINE_INDEX_NESTED_LOOP_JOIN_PACKED_PREDICATED_SHARED_OPERATOR_HH

#include "../../table/include/ffx_str_t.hpp"
#include "../factorized_ftree/ftree_state_update.hpp"
#include "../predicate/predicate_eval.hpp"
#include "inljoin_packed.hpp"// For VectorSliceUpdateSavedData
#include "operator.hpp"
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace ffx {

template<typename T>
class INLJoinPackedPredicatedShared final : public Operator {
public:
    INLJoinPackedPredicatedShared() = delete;
    INLJoinPackedPredicatedShared(const INLJoinPackedPredicatedShared&) = delete;

    // Constructor with predicate expression
    INLJoinPackedPredicatedShared(std::string join_key, std::string output_key, bool is_join_index_fwd,
                                  PredicateExpression predicate_expr)
        : Operator(), _join_key(std::move(join_key)), _output_key(std::move(output_key)), _in_vec(nullptr),
          _out_vec(nullptr), _is_join_index_fwd(is_join_index_fwd), _adj_lists(nullptr), _range_update_tree(nullptr),
          _vector_saved_data(nullptr), _predicate_expr_raw(std::move(predicate_expr)) {}

    // Constructor without predicate
    INLJoinPackedPredicatedShared(std::string join_key, std::string output_key, bool is_join_index_fwd)
        : Operator(), _join_key(std::move(join_key)), _output_key(std::move(output_key)), _in_vec(nullptr),
          _out_vec(nullptr), _is_join_index_fwd(is_join_index_fwd), _adj_lists(nullptr), _range_update_tree(nullptr),
          _vector_saved_data(nullptr) {}

    void init(Schema* schema) override;
    void execute() override;

    // Getters for testing
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
    Vector<T>* _in_vec;
    Vector<T>* _out_vec;
    bool _is_join_index_fwd;
    AdjList<T>* _adj_lists;
    std::unique_ptr<BitMask<State::MAX_VECTOR_SIZE>> _active_mask_uptr;
    BitMask<State::MAX_VECTOR_SIZE> _in_selector_backup;
    std::unique_ptr<FtreeStateUpdateNode> _range_update_tree;
    std::unique_ptr<VectorSliceUpdateSavedData<T>[]> _vector_saved_data;
    std::size_t _vector_saved_data_count = 0;
    PredicateExpression _predicate_expr_raw;

    bool _is_string_predicate = false;
    ScalarPredicateExpression<T> _predicate_expr_numeric;
    ScalarPredicateExpression<ffx_str_t> _predicate_expr_string;

    // Cascade propagation: track invalidated indices for sibling propagation
    std::unique_ptr<uint32_t[]> _invalidated_indices;
    int32_t _invalidated_count = 0;
};

// Type aliases for convenience
using INLJoinPackedPredicatedSharedUint64 = INLJoinPackedPredicatedShared<uint64_t>;
using INLJoinPackedPredicatedSharedString = INLJoinPackedPredicatedShared<ffx_str_t>;

}// namespace ffx

#endif// VFENGINE_INDEX_NESTED_LOOP_JOIN_PACKED_PREDICATED_SHARED_OPERATOR_HH
