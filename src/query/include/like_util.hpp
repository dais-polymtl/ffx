#ifndef VFENGINE_LIKE_UTIL_HPP
#define VFENGINE_LIKE_UTIL_HPP

#include <string>

namespace ffx {

/**
 * Converts a SQL LIKE pattern (e.g., "a%_b") to an RE2 compatible regex string (e.g., "^a.*.b$").
 */
inline std::string sql_like_to_regex(const std::string& pattern) {
    std::string regex = "^";
    for (size_t i = 0; i < pattern.length(); ++i) {
        char c = pattern[i];
        switch (c) {
            case '%':
                regex += ".*";
                break;
            case '_':
                regex += ".";
                break;
            // Escape regex special characters
            case '.':
            case '^':
            case '$':
            case '*':
            case '+':
            case '?':
            case '(':
            case ')':
            case '[':
            case ']':
            case '{':
            case '}':
            case '|':
            case '\\':
                regex += "\\";
                regex += c;
                break;
            default:
                regex += c;
                break;
        }
    }
    regex += "$";
    return regex;
}

}// namespace ffx

#endif// VFENGINE_LIKE_UTIL_HPP
