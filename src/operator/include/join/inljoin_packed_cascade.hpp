#ifndef VFENGINE_INDEX_NESTED_LOOP_JOIN_PACKED_CASCADE_OPERATOR_HH
#define VFENGINE_INDEX_NESTED_LOOP_JOIN_PACKED_CASCADE_OPERATOR_HH

#include "../factorized_ftree/ftree_state_update.hpp"
#include "operator.hpp"
#include <memory>
#include <string>
#include <vector>

namespace ffx {
template<typename T = uint64_t>
struct VectorSavedData {
    std::string attribute;
    Vector<T>* vector;
    std::pair<int32_t, int32_t> backup_state;// (start, end) positions
    VectorSavedData() : vector(nullptr), backup_state(-1, -1) {}
    VectorSavedData(const std::string& attr, Vector<T>* vec, int32_t start, int32_t end)
        : attribute(attr), vector(vec), backup_state(start, end) {}
};

template<typename T = uint64_t>
class INLJoinPackedCascade final : public Operator {
public:
    INLJoinPackedCascade() = delete;

    INLJoinPackedCascade(const INLJoinPackedCascade&) = delete;

    INLJoinPackedCascade(std::string join_key, std::string output_key, bool is_join_index_fwd)
        : Operator(), _join_key(std::move(join_key)), _output_key(std::move(output_key)),
          _is_join_index_fwd(is_join_index_fwd), _in_vec(nullptr), _out_vec(nullptr), _adj_lists(nullptr),
          _range_update_tree(nullptr), _vector_saved_data(nullptr), _invalidated_indices(nullptr),
          _invalidated_count(0) {}

    void init(Schema* schema) override;

    void execute() override;

    // Getters for testing
    const std::string& join_key() const { return _join_key; }
    const std::string& output_key() const { return _output_key; }
    bool is_join_index_fwd() const { return _is_join_index_fwd; }

private:
    void create_slice_update_infrastructure(FactorizedTreeElement* ftree_leaf);
    void store_slices();
    void print_upstream_vectors();
    void restore_slices();

    __attribute__((always_inline)) inline void process_data_chunk(BitMask<State::MAX_VECTOR_SIZE>* _current_ip_mask,
                                                                  int32_t op_filled_idx);

    const std::string _join_key, _output_key;
    bool _is_join_index_fwd;
    Vector<T>* _in_vec;
    Vector<T>* _out_vec;
    AdjList<T>* _adj_lists;
    std::unique_ptr<BitMask<State::MAX_VECTOR_SIZE>> _active_mask_uptr;
    std::unique_ptr<FtreeStateUpdateNode> _range_update_tree;
    std::unique_ptr<VectorSavedData<T>[]> _vector_saved_data;
    std::size_t _vector_saved_data_count = 0;
    std::unique_ptr<uint32_t[]> _invalidated_indices;
    int32_t _invalidated_count = 0;
    BitMask<State::MAX_VECTOR_SIZE> _in_selector_backup;
    BitMask<State::MAX_VECTOR_SIZE> _current_ip_mask;
};

// Type aliases for convenience
using INLJoinPackedCascadeUint64 = INLJoinPackedCascade<uint64_t>;
using INLJoinPackedCascadeString = INLJoinPackedCascade<ffx_str_t>;
}// namespace ffx

#endif