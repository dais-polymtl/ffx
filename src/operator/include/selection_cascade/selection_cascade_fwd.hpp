#ifndef FFX_CASCADE_SELECTION_FWD_OPERATOR_HH
#define FFX_CASCADE_SELECTION_FWD_OPERATOR_HH

#include "../factorized_ftree/factorized_tree_element.hpp"
#include "../factorized_ftree/ftree_state_update.hpp"
#include "operator.hpp"
#include <memory>

namespace ffx {

struct VectorBitMaskUpdateSavedData {
    Vector<uint64_t>* vector;
    BitMask<State::MAX_VECTOR_SIZE> backup_state{};
    VectorBitMaskUpdateSavedData() : vector(nullptr), backup_state() {}
    VectorBitMaskUpdateSavedData(Vector<uint64_t>* vec, const BitMask<State::MAX_VECTOR_SIZE>& other) :
        vector(vec), backup_state(other) {}
};

class CascadeSelection final : public Operator {
public:
    explicit CascadeSelection(const std::string& attribute);
    void execute() override;
    void init(Schema* schema) override;

private:
    void create_bitmask_update_infrastructure(FactorizedTreeElement* ftree_leaf);
    void store_bitmask();
    void restore_bitmask();

    std::string _attribute;
    std::unique_ptr<FtreeStateUpdateNode> _range_update_tree;
    std::unique_ptr<VectorBitMaskUpdateSavedData[]> _vector_saved_data;
    std::size_t _vector_saved_data_count = 0;
};
}// namespace ffx

#endif