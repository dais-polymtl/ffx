#ifndef VFENGINE_FTREE_ANCESTOR_FINDER_HH
#define VFENGINE_FTREE_ANCESTOR_FINDER_HH

#include "factorized_tree_element.hpp"
#include "../vector/vector.hpp"
#include "../vector/state.hpp"
#include <cstdint>
#include <vector>
#include <stdexcept>

namespace ffx {

/**
 * FtreeAncestorFinder: Maps descendant indices to a single ancestor's indices.
 * Used when you only need the mapping to one specific ancestor level.
 * 
 * NEW API (preferred): Construct with ordered state path.
 * The state path should contain unique states (no duplicates from shared state).
 * This is typically built by ancestor_finder_utils.hpp:build_ancestor_finder_path().
 */
class FtreeAncestorFinder {
public:
    /**
     * Construct from ordered state path.
     * @param state_path Ordered states from ancestor to descendant [anc_state, ..., desc_state]
     *                   Must have at least 2 states. No duplicate (shared) states allowed.
     */
    explicit FtreeAncestorFinder(const State* const* state_path, size_t path_size);

    void reset(const State* const* state_path, size_t path_size);

    void process(
        uint32_t* output_buffer,
        int32_t ancestor_start,
        int32_t ancestor_end,
        int32_t descendant_start,
        int32_t descendant_end
    );

    size_t path_length() const { return _state_path.size(); }

private:
    std::vector<const State*> _state_path;  // Ordered states from ancestor to descendant
};

/**
 * FtreeMultiAncestorFinder: Maps descendant indices to ALL ancestor levels' indices
 * in a single traversal. More efficient when you need indices at multiple levels.
 * 
 * Example: For state path [S_A, S_B, S_C, S_D] (A is root, D is deepest):
 *   - process() fills output_buffers[0], output_buffers[1], output_buffers[2]
 *   - output_buffers[i][d_idx] = index at level i corresponding to descendant index d_idx
 *   - Note: output_buffers has size (path_length - 1), one for each ancestor level
 * 
 * NEW API (preferred): Construct with ordered state path.
 * The state path should contain unique states (no duplicates from shared state).
 * This is typically built by ancestor_finder_utils.hpp:build_multi_ancestor_finder_path().
 */
class FtreeMultiAncestorFinder {
public:
    /**
     * Construct from ordered state path.
     * @param state_path Ordered states from root ancestor to descendant.
     *                   Must have at least 2 states. No duplicate states allowed.
     */
    explicit FtreeMultiAncestorFinder(std::vector<const State*> state_path);

    /**
     * Process and fill all ancestor index buffers in a single traversal.
     * @param output_buffers Array of buffers, one per ancestor level (size = path_length - 1).
     *                       output_buffers[i] maps descendant indices to level i indices.
     * @param descendant_start Start index in the descendant (deepest) level.
     * @param descendant_end End index in the descendant (deepest) level.
     */
    void process(
        uint32_t** output_buffers,
        int32_t descendant_start,
        int32_t descendant_end
    );

    size_t path_length() const { return _state_path.size(); }
    size_t num_ancestor_levels() const { return _state_path.size() - 1; }

private:
    std::vector<const State*> _state_path;  // Ordered states from root ancestor to descendant
};

} // namespace ffx

#endif // VFENGINE_FTREE_ANCESTOR_FINDER_HH

