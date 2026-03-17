#ifndef VFENGINE_NWAY_INTERSECTION_OPERATOR_HH
#define VFENGINE_NWAY_INTERSECTION_OPERATOR_HH

#include "../../table/include/adj_list.hpp"
#include "../../table/include/ffx_str_t.hpp"
#include "../factorized_ftree/ftree_ancestor_finder.hpp"
#include "../factorized_ftree/ftree_state_update.hpp"
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
struct NWayIntersectionVectorSliceUpdateSavedData {
    std::string attribute;
    Vector<T>* vector;
    std::pair<int32_t, int32_t> backup_state;// (start, end) positions
    NWayIntersectionVectorSliceUpdateSavedData() : vector(nullptr), backup_state(-1, -1) {}
    NWayIntersectionVectorSliceUpdateSavedData(const std::string& attr, Vector<T>* vec, int32_t start, int32_t end)
        : attribute(attr), vector(vec), backup_state(start, end) {}
};

template<typename T = uint64_t>
class NWayIntersection final : public Operator {
public:
    NWayIntersection() = delete;
    NWayIntersection(const NWayIntersection&) = delete;

    // Constructor
    NWayIntersection(std::string output_attr, std::vector<std::pair<std::string, bool>> input_attrs_and_directions);

    void init(Schema* schema) override;
    void execute() override;
    int32_t compute_sorted_intersection(const T* arr1, int32_t size1, const T* arr2, int32_t size2, T* dest);

    // Getters for testing
    const std::string& output_attr() const { return _output_attr; }
    const std::vector<std::pair<std::string, bool>>& input_attrs_and_directions() const {
        return _input_attrs_and_directions;
    }

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
    std::unique_ptr<NWayIntersectionVectorSliceUpdateSavedData<T>[]> _vector_saved_data;
    std::size_t _vector_saved_data_count = 0;

    // Pair ancestor finders: one for each consecutive pair of inputs
    // _pair_ancestor_finders[i] maps from input[i+1] to input[i]
    // Only created if they are in different DataChunks
    std::vector<std::unique_ptr<FtreeAncestorFinder>> _pair_ancestor_finders;
    // Same data chunk flags: true if input[i] and input[i+1] share same DataChunk
    std::vector<bool> _pair_same_data_chunk;
    // Ancestor index buffers: one per consecutive pair
    // _pair_ancestor_buffers[i][pos] gives input[i] index for input[i+1] position pos
    std::vector<std::unique_ptr<uint32_t[]>> _pair_ancestor_buffers;
};

// Type aliases for convenience
using NWayIntersectionUint64 = NWayIntersection<uint64_t>;
using NWayIntersectionString = NWayIntersection<ffx_str_t>;

}// namespace ffx

#endif
