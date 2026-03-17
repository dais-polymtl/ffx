#ifndef VFENGINE_FLAT_JOIN_OPERATOR_HH
#define VFENGINE_FLAT_JOIN_OPERATOR_HH

#include "../../table/include/ffx_str_t.hpp"
#include "../factorized_ftree/ftree_ancestor_finder.hpp"
#include "../factorized_ftree/ftree_state_update.hpp"
#include "operator.hpp"
#include <memory>
#include <string>
#include <type_traits>

namespace ffx {
template<typename T = uint64_t>
struct FlatJoinVectorSliceUpdateSavedData {
    std::string attribute;
    Vector<T>* vector;
    std::pair<int32_t, int32_t> backup_state;// (start, end) positions
    FlatJoinVectorSliceUpdateSavedData() : vector(nullptr), backup_state(-1, -1) {}
    FlatJoinVectorSliceUpdateSavedData(const std::string& attr, Vector<T>* vec, int32_t start, int32_t end)
        : attribute(attr), vector(vec), backup_state(start, end) {}
};

template<typename T = uint64_t>
class FlatJoin final : public Operator {
public:
    FlatJoin() = delete;
    FlatJoin(const FlatJoin&) = delete;

    FlatJoin(std::string parent_attr, std::string lca_attr, std::string output_attr, bool is_join_index_fwd)
        : Operator(), _parent_attr(std::move(parent_attr)), _lca_attr(std::move(lca_attr)),
          _output_attr(std::move(output_attr)), _is_join_index_fwd(is_join_index_fwd), _parent_vec(nullptr),
          _lca_vec(nullptr), _out_vec(nullptr), _parent_offset(nullptr), _adj_lists(nullptr),
          _range_update_tree(nullptr), _vector_saved_data(nullptr) {}

    void init(Schema* schema) override;
    void execute() override;

    // Getters for testing
    const std::string& parent_attr() const { return _parent_attr; }
    const std::string& lca_attr() const { return _lca_attr; }
    const std::string& output_attr() const { return _output_attr; }
    bool is_join_index_fwd() const { return _is_join_index_fwd; }

private:
    void create_slice_update_infrastructure(FactorizedTreeElement* ftree_leaf);
    void store_slices();
    void restore_slices();

    __attribute__((always_inline)) inline void process_data_chunk(BitMask<State::MAX_VECTOR_SIZE>* _current_ip_mask,
                                                                  int32_t op_filled_idx);

    const std::string _parent_attr, _lca_attr, _output_attr;
    bool _is_join_index_fwd;
    Vector<T>* _parent_vec;
    Vector<T>* _lca_vec;
    Vector<T>* _out_vec;
    uint16_t* _parent_offset;
    AdjList<T>* _adj_lists;
    std::unique_ptr<BitMask<State::MAX_VECTOR_SIZE>> _active_mask_uptr;
    std::unique_ptr<FtreeStateUpdateNode> _range_update_tree;
    std::unique_ptr<FlatJoinVectorSliceUpdateSavedData<T>[]> _vector_saved_data;
    std::size_t _vector_saved_data_count = 0;
    BitMask<State::MAX_VECTOR_SIZE> _parent_selector_backup;
    BitMask<State::MAX_VECTOR_SIZE> _current_ip_mask;
    // Flag: true if lca and parent are in the same DataChunk (identity mapping)
    bool _same_data_chunk = false;
    // Ancestor finder - only created if _same_data_chunk is false
    std::unique_ptr<FtreeAncestorFinder> _ancestor_finder;
};

// Type aliases for convenience
using FlatJoinUint64 = FlatJoin<uint64_t>;
using FlatJoinString = FlatJoin<ffx_str_t>;
}// namespace ffx

#endif
