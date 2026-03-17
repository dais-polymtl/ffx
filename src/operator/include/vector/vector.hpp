#ifndef VFENGINE_VECTOR_HH
#define VFENGINE_VECTOR_HH

#include "state.hpp"
#include "../../table/include/ffx_str_t.hpp"

namespace ffx {
template<typename T = uint64_t>
class Vector {
public:
    // Standard constructor - owns its own state
    Vector(bool unpacked_mode = false);
    
    // Shared state constructor - uses existing state from another vector
    // The shared_state must outlive this vector
    // When has_identity_rle is true, this vector's RLE is implicitly identity
    // (i.e., rle[i] = i for all i) and the actual RLE in state should not be used.
    Vector(State* shared_state, bool has_identity_rle = false);
    
    ~Vector();
    
    T* values;
    State* state;
    
    // Check if this vector owns its state
    bool owns_state() const { return _owns_state; }
    
    // Check if this vector has identity RLE (shares state with parent)
    // When true, don't use state->rle - use identity mapping instead
    bool has_identity_rle() const { return _has_identity_rle; }

private:
    std::unique_ptr<uint8_t[]> _memory_block;
    bool _unpacked_mode;
    bool _owns_state;  // True if this vector allocated and owns the state
    bool _has_identity_rle;  // True if RLE is identity (shares state with parent)
};

// Type aliases for convenience
using VectorUint64 = Vector<uint64_t>;
using VectorString = Vector<ffx_str_t>;

}// namespace ffx

#endif
