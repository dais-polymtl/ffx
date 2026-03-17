#ifndef FFX_CARDINALITY_HPP
#define FFX_CARDINALITY_HPP

#include <string>
#include <stdexcept>

namespace ffx {

/**
 * Cardinality of a binary relation (edge table).
 * Describes how many outputs each input produces.
 */
enum class Cardinality {
    ONE_TO_ONE,   // 1:1 - each src maps to exactly one dest, and vice versa
    MANY_TO_ONE,  // n:1 - many srcs map to one dest (forward gives 1 output per input)
    ONE_TO_MANY,  // 1:n - one src maps to many dests (backward gives 1 output per input)
    MANY_TO_MANY  // m:n - general case, variable output count (default)
};

/**
 * Determine if state should be shared based on cardinality and traversal direction.
 * 
 * State can be shared when each input position produces exactly ONE output position:
 * - 1:1: Always share (both directions give 1:1 mapping)
 * - n:1 + forward: Share (each source has exactly 1 dest)
 * - 1:n + backward: Share (each dest has exactly 1 source)
 * - m:n: Never share (variable output count)
 * 
 * @param card The cardinality of the relation
 * @param is_fwd True if traversing forward (src -> dest), false if backward
 * @return True if state can be shared, false otherwise
 */
inline bool should_share_state(Cardinality card, bool is_fwd) {
    switch (card) {
        case Cardinality::ONE_TO_ONE:
            return true;  // Always 1:1 mapping in both directions
        case Cardinality::MANY_TO_ONE:
            return is_fwd;  // n sources → 1 dest: forward gives 1 output per input
        case Cardinality::ONE_TO_MANY:
            return !is_fwd; // 1 source → n dests: backward gives 1 output per input
        case Cardinality::MANY_TO_MANY:
        default:
            return false;   // Variable output count
    }
}

/**
 * Parse cardinality from string format.
 * 
 * Supported formats: "1:1", "n:1", "1:n", "m:n"
 * Empty string or unrecognized format defaults to MANY_TO_MANY.
 * 
 * @param str The cardinality string (e.g., "[n:1]" or "n:1")
 * @return The parsed Cardinality enum value
 */
inline Cardinality parse_cardinality(const std::string& str) {
    // Remove brackets if present
    std::string s = str;
    if (!s.empty() && s.front() == '[') {
        s = s.substr(1);
    }
    if (!s.empty() && s.back() == ']') {
        s = s.substr(0, s.size() - 1);
    }
    
    if (s == "1:1") return Cardinality::ONE_TO_ONE;
    if (s == "n:1") return Cardinality::MANY_TO_ONE;
    if (s == "1:n") return Cardinality::ONE_TO_MANY;
    if (s == "m:n" || s.empty()) return Cardinality::MANY_TO_MANY;
    
    // Default to m:n for unrecognized formats
    return Cardinality::MANY_TO_MANY;
}

/**
 * Convert cardinality to string for display/debugging.
 */
inline std::string cardinality_to_string(Cardinality card) {
    switch (card) {
        case Cardinality::ONE_TO_ONE: return "1:1";
        case Cardinality::MANY_TO_ONE: return "n:1";
        case Cardinality::ONE_TO_MANY: return "1:n";
        case Cardinality::MANY_TO_MANY: return "m:n";
        default: return "m:n";
    }
}

} // namespace ffx

#endif // FFX_CARDINALITY_HPP

