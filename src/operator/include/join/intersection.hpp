#ifndef VFENGINE_INTERSECTION_OPERATOR_HH
#define VFENGINE_INTERSECTION_OPERATOR_HH

#include "../../table/include/ffx_str_t.hpp"
#include "../factorized_ftree/ftree_ancestor_finder.hpp"
#include "../factorized_ftree/ftree_state_update.hpp"
#include "operator.hpp"
#include <cstring>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace ffx {
template<typename T = uint64_t>
struct IntersectionVectorSliceUpdateSavedData {
    std::string attribute;
    Vector<T>* vector;
    std::pair<int32_t, int32_t> backup_state;// (start, end) positions
    IntersectionVectorSliceUpdateSavedData() : vector(nullptr), backup_state(-1, -1) {}
    IntersectionVectorSliceUpdateSavedData(const std::string& attr, Vector<T>* vec, int32_t start, int32_t end)
        : attribute(attr), vector(vec), backup_state(start, end) {}
};

// Cache entry for intersection results
template<typename T = uint64_t>
struct IntersectionCacheEntry {
    int32_t count;
    union {
        T stack_values[State::MAX_VECTOR_SIZE];
        std::unique_ptr<T[]> heap_values;
    };
    bool is_heap;

    IntersectionCacheEntry() : count(0), is_heap(false) {
        // Initialize stack_values array (POD type, no constructor needed)
        // Values will be set when intersection is computed
    }

    ~IntersectionCacheEntry() {
        if (is_heap) {
            heap_values.~unique_ptr();
        } else if constexpr (!std::is_trivially_destructible_v<T>) {
            // For non-trivial types, call destructors
            for (int32_t i = 0; i < count; ++i) {
                stack_values[i].~T();
            }
        }
    }

    // Non-copyable, but movable
    IntersectionCacheEntry(const IntersectionCacheEntry&) = delete;
    IntersectionCacheEntry& operator=(const IntersectionCacheEntry&) = delete;

    IntersectionCacheEntry(IntersectionCacheEntry&& other) noexcept : count(other.count), is_heap(other.is_heap) {
        if (is_heap) {
            new (&heap_values) std::unique_ptr<T[]>(std::move(other.heap_values));
        } else {
            if constexpr (std::is_trivially_copyable_v<T>) {
                std::memcpy(stack_values, other.stack_values, count * sizeof(T));
            } else {
                for (int32_t i = 0; i < count; ++i) {
                    new (&stack_values[i]) T(std::move(other.stack_values[i]));
                }
            }
        }
    }

    IntersectionCacheEntry& operator=(IntersectionCacheEntry&& other) noexcept {
        if (this != &other) {
            // Clean up current state
            if (is_heap) {
                heap_values.~unique_ptr();
            } else if constexpr (!std::is_trivially_destructible_v<T>) {
                for (int32_t i = 0; i < count; ++i) {
                    stack_values[i].~T();
                }
            }

            count = other.count;
            is_heap = other.is_heap;

            if (is_heap) {
                new (&heap_values) std::unique_ptr<T[]>(std::move(other.heap_values));
            } else {
                if constexpr (std::is_trivially_copyable_v<T>) {
                    std::memcpy(stack_values, other.stack_values, count * sizeof(T));
                } else {
                    for (int32_t i = 0; i < count; ++i) {
                        new (&stack_values[i]) T(std::move(other.stack_values[i]));
                    }
                }
            }
        }
        return *this;
    }

    // Get pointer to values array
    const T* get_values() const { return is_heap ? heap_values.get() : stack_values; }
};

// Hash function for pair<T, T>
template<typename T = uint64_t>
struct PairHash {
    size_t operator()(const std::pair<T, T>& p) const {
        if constexpr (std::is_same_v<T, uint64_t>) {
            return std::hash<T>{}(p.first) ^ (std::hash<T>{}(p.second) << 1);
        } else {
            // For ffx_str_t, use the hash function
            return ffx_str_hash{}(p.first) ^ (ffx_str_hash{}(p.second) << 1);
        }
    }
};

template<typename T = uint64_t>
class Intersection final : public Operator {
public:
    Intersection() = delete;
    Intersection(const Intersection&) = delete;

    Intersection(std::string ancestor_attr, std::string descendant_attr, std::string output_attr,
                 bool is_ancestor_join_fwd, bool is_descendant_join_fwd)
        : Operator(), _ancestor_attr(std::move(ancestor_attr)), _descendant_attr(std::move(descendant_attr)),
          _output_attr(std::move(output_attr)), _is_ancestor_join_fwd(is_ancestor_join_fwd),
          _is_descendant_join_fwd(is_descendant_join_fwd), _ancestor_vec(nullptr), _descendant_vec(nullptr),
          _out_vec(nullptr), _descendant_offset(nullptr), _ancestor_adj_lists(nullptr), _descendant_adj_lists(nullptr),
          _range_update_tree(nullptr), _vector_saved_data(nullptr) {}

    void init(Schema* schema) override;
    void execute() override;

    // Getters for testing
    const std::string& ancestor_attr() const { return _ancestor_attr; }
    const std::string& descendant_attr() const { return _descendant_attr; }
    const std::string& output_attr() const { return _output_attr; }
    bool is_ancestor_join_fwd() const { return _is_ancestor_join_fwd; }
    bool is_descendant_join_fwd() const { return _is_descendant_join_fwd; }

private:
    void create_slice_update_infrastructure(const FactorizedTreeElement* ftree_output_node);
    void store_slices();
    void restore_slices();

    __attribute__((always_inline)) inline void process_data_chunk(BitMask<State::MAX_VECTOR_SIZE>* _current_ip_mask,
                                                                  int32_t op_filled_idx);

    // Compute intersection and cache it (lazy caching)
    const IntersectionCacheEntry<T>& get_or_compute_intersection(T a_val, T b_val, const AdjList<T>& ancestor_adj_list,
                                                                 const AdjList<T>& descendant_adj_list);

    // Compute intersection without caching (writes directly into destination buffer, returns count)
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
    std::unique_ptr<BitMask<State::MAX_VECTOR_SIZE>>
            _descendant_valid_mask_uptr;// Tracks which descendant positions are valid
    BitMask<State::MAX_VECTOR_SIZE> _descendant_selector_backup;
    BitMask<State::MAX_VECTOR_SIZE> _current_ip_mask;
    std::unique_ptr<FtreeStateUpdateNode> _range_update_tree;
    std::unique_ptr<IntersectionVectorSliceUpdateSavedData<T>[]> _vector_saved_data;
    std::size_t _vector_saved_data_count = 0;
    // Ancestor index buffer - maps descendant indices to ancestor indices
    std::unique_ptr<uint32_t[]> _ancestor_index_buffer;
    // Flag: true if ancestor and descendant are in the same DataChunk (identity mapping)
    bool _same_data_chunk = false;
    // Ancestor finder - maps descendant indices to ancestor indices (handles both direct and multi-hop)
    // Only created if _same_data_chunk is false
    std::unique_ptr<FtreeAncestorFinder> _ancestor_finder;
    // Cache for intersection results: (a_val, b_val) -> IntersectionCacheEntry
    std::unordered_map<std::pair<T, T>, IntersectionCacheEntry<T>, PairHash<T>> _intersection_cache;
};

// Type aliases for convenience
using IntersectionUint64 = Intersection<uint64_t>;
using IntersectionString = Intersection<ffx_str_t>;

}// namespace ffx

#endif
