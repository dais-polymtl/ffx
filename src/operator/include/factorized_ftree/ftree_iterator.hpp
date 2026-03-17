#ifndef VFENGINE_SIMPLIFIED_FTREE_ITERATOR_HH
#define VFENGINE_SIMPLIFIED_FTREE_ITERATOR_HH

#include "factorized_tree_element.hpp"
#include "operator.hpp"
#include "vector/vector.hpp"
#include <cstdint>
#include <memory>
#include <vector>

namespace ffx {

class StringDictionary;// Forward declaration

struct SimpleLocalIterator {
    // Hot path data (frequently accessed together) - first cache line
    int32_t current_pos;
    int32_t valid_start;
    int32_t valid_end;
    int32_t parent_idx;// Index of parent in ordering array (-1 for root)

    // Cached range calculations to avoid redundant computation
    int32_t cached_range_start;// Cache last computed range
    int32_t cached_range_end;
    int32_t cached_parent_pos;// Track parent pos when range was computed
    bool shares_parent_state;

    // Pointers (less frequently changed) - may be in separate cache line
    const uint16_t* __restrict__ offset;
    const BitMask<State::MAX_VECTOR_SIZE>* __restrict__ selector;
    const uint64_t* __restrict__ values;

    SimpleLocalIterator()
        : current_pos(-1), valid_start(-1), valid_end(-1), parent_idx(-1), cached_range_start(-1), cached_range_end(-1),
          cached_parent_pos(-1), shares_parent_state(false), offset(nullptr), selector(nullptr), values(nullptr) {}
} __attribute__((aligned(64)));// Align to cache line for better performance

class FTreeIterator : public Operator {
public:
    FTreeIterator();
    ~FTreeIterator() override = default;
    void init(Schema* schema) override;
    void execute() override {}

    inline bool is_valid() const { return _is_valid; }

    // Move to next tuple and write to output buffer
    // Returns true if successful, false if exhausted
    bool next(uint64_t* output_buffer);
    bool next_min_tuple(uint64_t* output_buffer);

    // Debug version of next() that prints each iterator state
    bool next_debug(uint64_t* output_buffer);

    // Get current tuple values and write to output buffer
    // Returns true if valid, false if exhausted
    bool current(uint64_t* output_buffer) const;

    void reset();

    inline size_t tuple_size() const { return _num_attributes; }

    // Get attribute names in column ordering (for debug output)
    std::vector<std::string> get_attribute_ordering() const {
        std::vector<std::string> ordering;
        ordering.reserve(_num_attributes);
        for (size_t i = 0; i < _num_attributes; ++i) {
            ordering.push_back(_nodes[i]->_attribute);
        }
        return ordering;
    }

    // Initialize all local iterators to their starting positions
    void initialize_iterators();

    // Print current state of all iterators
    void print_iterator_state() const;

    mutable std::vector<std::vector<uint64_t>> _values;
    mutable uint64_t _num_output_tuples = 0;

    // Debug: per-position visit counts for each attribute
    mutable std::unique_ptr<std::unique_ptr<uint64_t[]>[]> _debug_pos_counts;
    void print_debug_counts() const;

    // Interface helpers
    std::vector<std::vector<uint64_t>> get_values() const { return _values; }
    uint64_t get_num_output_tuples() override { return _num_output_tuples; }

    // Update min values for all iterators (public for sink_min_itr to initialize buffer)
    void update_min_vals_current_tuple(uint64_t* buffer) const;

    // Set string attribute support for min comparison
    void set_string_support(StringDictionary* dictionary, const std::vector<bool>& is_string_attr) {
        _dictionary = dictionary;
        _is_string_attr = is_string_attr;
    }

private:
    // Try to advance iterator at position idx in the ordering
    // Returns true if successfully advanced, false if need to backtrack
    __attribute__((always_inline)) inline bool try_advance(size_t idx);

    // Reset iterator at idx to start of its valid range based on parent's current position
    // Returns true if valid position found, false if empty range
    __attribute__((always_inline)) inline bool reset_iterator_to_start(size_t idx);

    // Find next set bit for the root-level iterator
    __attribute__((always_inline)) inline int32_t find_next_set_bit_root(const SimpleLocalIterator& itr) const;

    // Find next set bit for non-root iterators
    __attribute__((always_inline)) inline int32_t find_next_set_bit_child(const SimpleLocalIterator& itr,
                                                                          int32_t parent_pos) const;

    void read_current_tuple(uint64_t* buffer) const;
    void update_min_vals_current_tuple_2(uint64_t* buffer, size_t start_idx) const;

    std::shared_ptr<FactorizedTreeElement> _root;
    std::unique_ptr<SimpleLocalIterator[]> _iterators;// Ordered by column ordering
    std::unique_ptr<FactorizedTreeElement*[]> _nodes; // Corresponding tree nodes in same order
    std::unique_ptr<uint64_t[]> _current_tuple;       // Current tuple values (owned)

    size_t _num_attributes;
    bool _is_valid;

    // Synthetic nodes for attributes that exist as vectors in Schema::map but
    // are not present in the factorized tree (e.g., 1:1 property vectors with
    // shared state). Owned here to keep pointers in _nodes valid.
    std::vector<std::shared_ptr<FactorizedTreeElement>> _synthetic_nodes;

    // String attribute support for min comparison
    StringDictionary* _dictionary = nullptr;
    std::vector<bool> _is_string_attr;
};

}// namespace ffx

#endif
