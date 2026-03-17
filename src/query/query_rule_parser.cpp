#include "query_rule_parser.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace ffx {

namespace {

std::string trim(const std::string& str) {
    size_t start = str.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) { return ""; }
    size_t end = str.find_last_not_of(" \t\n\r");
    return str.substr(start, end - start + 1);
}

void to_upper_inplace(std::string& s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
}

std::string to_upper_copy(std::string s) {
    to_upper_inplace(s);
    return s;
}

bool starts_with_ci(const std::string& s, const char* prefix) {
    for (size_t i = 0; prefix[i] != '\0'; ++i) {
        if (i >= s.size()) { return false; }
        if (std::tolower(static_cast<unsigned char>(s[i])) != std::tolower(static_cast<unsigned char>(prefix[i]))) {
            return false;
        }
    }
    return true;
}

std::vector<std::string> split_comma_paren_depth(const std::string& s) {
    std::vector<std::string> out;
    int depth = 0;
    std::string cur;
    for (char c : s) {
        if (c == '(') {
            ++depth;
            cur += c;
        } else if (c == ')') {
            --depth;
            cur += c;
        } else if (c == ',' && depth == 0) {
            out.push_back(trim(cur));
            cur.clear();
        } else {
            cur += c;
        }
    }
    if (!trim(cur).empty() || !cur.empty()) {
        out.push_back(trim(cur));
    }
    return out;
}

std::vector<std::string> split_comma_top_level(const std::string& s) { return split_comma_paren_depth(s); }

/// Variables inside Rel(a,b,_) — all trimmed tokens that are not "_"
void collect_vars_from_atom(const std::string& atom, std::unordered_set<std::string>& vars) {
    size_t o = atom.find('(');
    size_t c = atom.rfind(')');
    if (o == std::string::npos || c == std::string::npos || c <= o) { return; }
    std::string inner = atom.substr(o + 1, c - o - 1);
    for (const auto& part : split_comma_top_level(inner)) {
        std::string t = trim(part);
        if (!t.empty() && t != "_") { vars.insert(t); }
    }
}

/// Explains invalid Q(...) attribute names; suggests a missing comma when a digit runs into an identifier (e.g. `1llm_out`).
std::string invalid_q_head_attribute_message(const std::string& t) {
    std::string msg = "In Q(...), '" + t
            + "' is not a valid attribute name. Use a letter or underscore first, then only letters, digits, or "
              "underscores.";
    if (t.size() >= 2 && std::isdigit(static_cast<unsigned char>(t[0]))) {
        size_t j = 1;
        while (j < t.size() && std::isdigit(static_cast<unsigned char>(t[j]))) { ++j; }
        if (j < t.size()) {
            const std::string rest = t.substr(j);
            if (!rest.empty() &&
                (std::isalpha(static_cast<unsigned char>(rest[0])) || rest[0] == '_')) {
                msg += " If you meant two separate attributes in Q(...), there may be a missing comma before '" + rest
                        + "'.";
            }
        }
    }
    return msg;
}

void parse_head_inner(const std::string& inner_raw, ParsedQueryRule& out) {
    std::string inner = trim(inner_raw);
    if (inner.empty()) {
        throw std::invalid_argument("Q(...) head is empty.");
    }
    std::string u = to_upper_copy(inner);

    if (u == "NOOP") {
        out.head_kind = QueryHeadKind::Noop;
        return;
    }

    if (starts_with_ci(inner, "COUNT(")) {
        size_t close = inner.rfind(')');
        if (close == std::string::npos || close <= 5) {
            throw std::invalid_argument("Malformed Q(COUNT(...)) head.");
        }
        std::string inside = trim(inner.substr(6, close - 6));
        // Between COUNT( and ) we expect only `*` (optional spaces), not nested parens.
        std::string compact;
        for (char ch : inside) {
            if (!std::isspace(static_cast<unsigned char>(ch))) { compact += ch; }
        }
        if (compact != "*" && compact != "(*)") {
            throw std::invalid_argument("Only Q(COUNT(*)) is supported for this head form.");
        }
        out.head_kind = QueryHeadKind::CountStar;
        return;
    }

    if (starts_with_ci(inner, "MIN(")) {
        size_t close = inner.rfind(')');
        if (close == std::string::npos || close <= 4) {
            throw std::invalid_argument("Malformed Q(MIN(...)) head.");
        }
        std::string inside = trim(inner.substr(4, close - 4));
        if (inside.empty()) {
            throw std::invalid_argument("Q(MIN(...)) requires at least one attribute.");
        }
        for (const auto& tok : split_comma_top_level(inside)) {
            std::string t = trim(tok);
            if (t.empty() || t == "_") {
                throw std::invalid_argument("Invalid attribute name inside Q(MIN(...)).");
            }
            out.head_attributes.push_back(t);
        }
        if (out.head_attributes.empty()) {
            throw std::invalid_argument("Q(MIN(...)) requires at least one attribute.");
        }
        out.head_kind = QueryHeadKind::Min;
        return;
    }

    // Q(a,b,c,...) head: comma-separated attribute names
    for (const auto& tok : split_comma_top_level(inner)) {
        std::string t = trim(tok);
        if (t.empty()) {
            throw std::invalid_argument("Q(...) lists an empty attribute name.");
        }
        if (!std::isalpha(static_cast<unsigned char>(t[0])) && t[0] != '_') {
            throw std::invalid_argument(invalid_q_head_attribute_message(t));
        }
        for (size_t i = 1; i < t.size(); ++i) {
            if (!std::isalnum(static_cast<unsigned char>(t[i])) && t[i] != '_') {
                throw std::invalid_argument(invalid_q_head_attribute_message(t));
            }
        }
        out.head_attributes.push_back(t);
    }
    if (out.head_attributes.empty()) {
        throw std::invalid_argument("Q(...) must name at least one attribute.");
    }
    out.head_kind = QueryHeadKind::Projection;
}

/// Extract `name = LLM_MAP({...})` from segments; returns joined relational atoms string.
std::string extract_llm_assign_from_segments(const std::vector<std::string>& segments, std::string& assign_name,
                                             std::string& llm_json) {
    assign_name.clear();
    llm_json.clear();
    std::vector<std::string> rel_parts;
    int llm_count = 0;

    for (const auto& seg_raw : segments) {
        std::string seg = trim(seg_raw);
        if (seg.empty()) { continue; }

        std::string u = to_upper_copy(seg);
        size_t lm_pos = u.find("LLM_MAP(");
        if (lm_pos == std::string::npos) {
            rel_parts.push_back(seg);
            continue;
        }

        size_t eq = seg.rfind('=', lm_pos);
        if (eq == std::string::npos) {
            throw std::invalid_argument("LLM_MAP in the body must use 'name = LLM_MAP({...})'.");
        }
        std::string name = trim(seg.substr(0, eq));
        if (name.empty()) {
            throw std::invalid_argument("Missing attribute name before '= LLM_MAP(...)'.");
        }

        size_t json_start = lm_pos + 8;
        int depth = 1;
        size_t i = json_start;
        for (; i < seg.size() && depth > 0; ++i) {
            if (seg[i] == '(') { ++depth; }
            else if (seg[i] == ')') { --depth; }
        }
        if (depth != 0) {
            throw std::invalid_argument("Unbalanced parentheses in LLM_MAP(...).");
        }
        llm_json = trim(seg.substr(json_start, i - 1 - json_start));

        if (i < seg.size() && seg[i] == ':') {
            throw std::invalid_argument(
                    "A ':attr' suffix after LLM_MAP is not supported; put the output attribute name in Q(...) and use "
                    "'name = LLM_MAP({...})' in the body.");
        }

        if (llm_count > 0) {
            throw std::invalid_argument("At most one LLM_MAP assignment is allowed in the body.");
        }
        assign_name = name;
        ++llm_count;
    }

    std::ostringstream oss;
    for (size_t k = 0; k < rel_parts.size(); ++k) {
        if (k) { oss << ", "; }
        oss << rel_parts[k];
    }
    return oss.str();
}

} // namespace

bool looks_like_query_rule(const std::string& input) {
    std::string t = trim(input);
    if (t.find(":=") == std::string::npos) { return false; }
    // Must begin with Q (after whitespace) to avoid treating accidental :=
    size_t i = 0;
    while (i < t.size() && std::isspace(static_cast<unsigned char>(t[i]))) { ++i; }
    return i < t.size() && (t[i] == 'Q' || t[i] == 'q');
}

ParsedQueryRule parse_and_validate_query_rule(const std::string& input) {
    std::string t = trim(input);
    if (t.find(';') != std::string::npos) {
        throw std::invalid_argument("Multiple queries in one string (';') are not supported.");
    }

    size_t def_pos = t.find(":=");
    if (def_pos == std::string::npos) {
        throw std::invalid_argument("Missing ':=' between head and body.");
    }

    std::string left = trim(t.substr(0, def_pos));
    std::string right = trim(t.substr(def_pos + 2));

    if (left.empty()) {
        throw std::invalid_argument("Missing head before ':='.");
    }
    if (right.empty()) {
        throw std::invalid_argument("Missing body after ':='.");
    }

    if (!starts_with_ci(left, "Q(")) {
        throw std::invalid_argument("Head must start with Q(...).");
    }

    size_t open = left.find('(');
    if (open == std::string::npos) {
        throw std::invalid_argument("Malformed Q(...).");
    }
    int depth = 0;
    size_t close = std::string::npos;
    for (size_t i = open; i < left.size(); ++i) {
        if (left[i] == '(') { ++depth; }
        else if (left[i] == ')') {
            --depth;
            if (depth == 0) {
                close = i;
                break;
            }
        }
    }
    if (close == std::string::npos || close != left.size() - 1) {
        throw std::invalid_argument("Q(...) must be fully closed with no text after the closing ')'.");
    }

    std::string head_inner = trim(left.substr(open + 1, close - open - 1));

    ParsedQueryRule out;
    parse_head_inner(head_inner, out);

    // Body: WHERE
    std::string body_work = right;
    std::string upper_body = to_upper_copy(body_work);
    size_t where_pos = upper_body.find(" WHERE ");
    if (where_pos != std::string::npos) {
        out.where_clause = trim(body_work.substr(where_pos + 7));
        body_work = trim(body_work.substr(0, where_pos));
    }

    if (trim(body_work).empty()) {
        throw std::invalid_argument("Body has no relational part (only WHERE, or body is empty).");
    }

    const std::vector<std::string> segments = split_comma_top_level(body_work);
    out.relational_body = extract_llm_assign_from_segments(segments, out.llm_assign_name, out.llm_json);

    if (!out.llm_json.empty() && out.llm_assign_name.empty()) {
        throw std::invalid_argument(
                "LLM_MAP must use 'name = LLM_MAP({...})' with a non-empty attribute name before '='.");
    }

    if (trim(out.relational_body).empty()) {
        throw std::invalid_argument("Body must contain at least one relation atom.");
    }

    if (out.relational_body.find('(') == std::string::npos) {
        throw std::invalid_argument("Each body relation must use Table(attribute_list) form (not arrow syntax).");
    }

    // --- Validation ---
    std::unordered_set<std::string> body_vars;
    for (const auto& atom : split_comma_top_level(out.relational_body)) {
        if (trim(atom).empty()) { continue; }
        collect_vars_from_atom(trim(atom), body_vars);
    }
    if (!out.llm_assign_name.empty()) {
        body_vars.insert(out.llm_assign_name);
    }

    auto check_head_vars_in_body = [&](const std::vector<std::string>& attrs) {
        for (const auto& h : attrs) {
            if (body_vars.count(h) == 0) {
                throw std::invalid_argument(
                        "Attribute '" + h + "' is listed in Q(...) but is not bound in the body: it does not appear "
                        "in any relation atom, and it is not the left-hand side of `name = LLM_MAP(...)`.");
            }
        }
    };

    switch (out.head_kind) {
    case QueryHeadKind::Projection: {
        std::unordered_set<std::string> seen;
        for (const auto& h : out.head_attributes) {
            if (!seen.insert(h).second) {
                throw std::invalid_argument("Attribute '" + h + "' appears more than once in Q(...).");
            }
        }
        check_head_vars_in_body(out.head_attributes);
        if (!out.llm_assign_name.empty()) {
            bool found = false;
            for (const auto& h : out.head_attributes) {
                if (h == out.llm_assign_name) { found = true; }
            }
            if (!found) {
                throw std::invalid_argument(
                        "The body assigns the LLM output to '" + out.llm_assign_name
                        + "', but that name is missing from Q(...). Add '" + out.llm_assign_name
                        + "' to Q(...) so head and body declare the same output attributes.");
            }
        }
        break;
    }
    case QueryHeadKind::Min:
        check_head_vars_in_body(out.head_attributes);
        if (!out.llm_json.empty()) {
            throw std::invalid_argument("LLM_MAP is not allowed with Q(MIN(...)).");
        }
        break;
    case QueryHeadKind::CountStar:
        if (!out.llm_json.empty()) {
            throw std::invalid_argument("LLM_MAP is not allowed with Q(COUNT(*)).");
        }
        if (!out.head_attributes.empty()) {
            throw std::invalid_argument("Internal error: COUNT head has attributes.");
        }
        break;
    case QueryHeadKind::Noop:
        if (!out.llm_json.empty()) {
            throw std::invalid_argument("LLM_MAP is not allowed with Q(NOOP).");
        }
        if (!out.head_attributes.empty()) {
            throw std::invalid_argument("Internal error: NOOP head has attributes.");
        }
        break;
    default:
        break;
    }

    return out;
}

} // namespace ffx
