#include "include/predicate_parser.hpp"
#include <cctype>
#include <unordered_map>

namespace ffx {

PredicateLexer::PredicateLexer(const std::string& input) : _input(input) {}

char PredicateLexer::peek() const {
    if (is_at_end()) return '\0';
    return _input[_pos];
}

char PredicateLexer::advance() {
    if (is_at_end()) return '\0';
    return _input[_pos++];
}

void PredicateLexer::skip_whitespace() {
    while (!is_at_end() && std::isspace(peek())) {
        advance();
    }
}

Token PredicateLexer::read_identifier_or_keyword() {
    size_t start = _pos;
    while (!is_at_end() && is_alnum(peek())) {
        advance();
    }
    
    std::string value = _input.substr(start, _pos - start);
    
    // Check for keywords
    static const std::unordered_map<std::string, TokenType> keywords = {
        {"EQ", TokenType::EQ},
        {"NEQ", TokenType::NEQ},
        {"LT", TokenType::LT},
        {"GT", TokenType::GT},
        {"LTE", TokenType::LTE},
        {"GTE", TokenType::GTE},
        {"IN", TokenType::IN},
        {"LIKE", TokenType::LIKE},
        {"NOT_LIKE", TokenType::NOT_LIKE},
        {"NOT_IN", TokenType::NOT_IN},
        {"BETWEEN", TokenType::BETWEEN},
        {"IS_NULL", TokenType::IS_NULL},
        {"IS_NOT_NULL", TokenType::IS_NOT_NULL},
        {"AND", TokenType::AND},
        {"OR", TokenType::OR},
        {"NOTEXISTS", TokenType::NOTEXISTS},
        {"NOT_EXISTS", TokenType::NOTEXISTS},
        {"ANTI", TokenType::NOTEXISTS},  // Shorthand alias
        // Also support lowercase
        {"eq", TokenType::EQ},
        {"neq", TokenType::NEQ},
        {"lt", TokenType::LT},
        {"gt", TokenType::GT},
        {"lte", TokenType::LTE},
        {"gte", TokenType::GTE},
        {"in", TokenType::IN},
        {"like", TokenType::LIKE},
        {"not_like", TokenType::NOT_LIKE},
        {"not_in", TokenType::NOT_IN},
        {"between", TokenType::BETWEEN},
        {"is_null", TokenType::IS_NULL},
        {"is_not_null", TokenType::IS_NOT_NULL},
        {"and", TokenType::AND},
        {"or", TokenType::OR},
        {"notexists", TokenType::NOTEXISTS},
        {"not_exists", TokenType::NOTEXISTS},
        {"anti", TokenType::NOTEXISTS},
    };
    
    auto it = keywords.find(value);
    if (it != keywords.end()) {
        return Token(it->second, value, start);
    }
    
    return Token(TokenType::IDENTIFIER, value, start);
}

Token PredicateLexer::read_number() {
    size_t start = _pos;
    
    // Handle negative numbers
    if (peek() == '-') {
        advance();
    }
    
    while (!is_at_end() && is_digit(peek())) {
        advance();
    }
    
    // Handle decimals
    if (peek() == '.' && _pos + 1 < _input.size() && is_digit(_input[_pos + 1])) {
        advance(); // consume '.'
        while (!is_at_end() && is_digit(peek())) {
            advance();
        }
    }
    
    return Token(TokenType::NUMBER, _input.substr(start, _pos - start), start);
}

Token PredicateLexer::read_string() {
    size_t start = _pos;
    char quote = advance(); // consume opening quote
    
    std::string value;
    while (!is_at_end() && peek() != quote) {
        if (peek() == '\\' && _pos + 1 < _input.size()) {
            advance(); // skip backslash
            char escaped = advance();
            switch (escaped) {
                case 'n': value += '\n'; break;
                case 't': value += '\t'; break;
                case '\\': value += '\\'; break;
                case '"': value += '"'; break;
                case '\'': value += '\''; break;
                default: value += escaped; break;
            }
        } else {
            value += advance();
        }
    }
    
    if (is_at_end()) {
        throw PredicateParseError("Unterminated string literal", start);
    }
    
    advance(); // consume closing quote
    return Token(TokenType::STRING, value, start);
}

std::vector<Token> PredicateLexer::tokenize() {
    std::vector<Token> tokens;
    
    while (!is_at_end()) {
        skip_whitespace();
        if (is_at_end()) break;
        
        size_t start = _pos;
        char c = peek();
        
        if (is_alpha(c)) {
            tokens.push_back(read_identifier_or_keyword());
        } else if (is_digit(c) || (c == '-' && _pos + 1 < _input.size() && is_digit(_input[_pos + 1]))) {
            tokens.push_back(read_number());
        } else if (c == '"' || c == '\'') {
            tokens.push_back(read_string());
        } else if (c == '(') {
            advance();
            tokens.emplace_back(TokenType::LPAREN, "(", start);
        } else if (c == ')') {
            advance();
            tokens.emplace_back(TokenType::RPAREN, ")", start);
        } else if (c == ',') {
            advance();
            tokens.emplace_back(TokenType::COMMA, ",", start);
        } else if (c == '<') {
            advance();
            if (peek() == '=') {
                advance();
                tokens.emplace_back(TokenType::LTE, "<=", start);
            } else if (peek() == '>') {
                advance();
                tokens.emplace_back(TokenType::NEQ, "<>", start);
            } else {
                tokens.emplace_back(TokenType::LT, "<", start);
            }
        } else if (c == '>') {
            advance();
            if (peek() == '=') {
                advance();
                tokens.emplace_back(TokenType::GTE, ">=", start);
            } else {
                tokens.emplace_back(TokenType::GT, ">", start);
            }
        } else if (c == '=') {
            advance();
            if (peek() == '=') {
                advance();
            }
            tokens.emplace_back(TokenType::EQ, "=", start);
        } else if (c == '!') {
            advance();
            if (peek() == '=') {
                advance();
                tokens.emplace_back(TokenType::NEQ, "!=", start);
            } else {
                throw PredicateParseError(std::string("Expected '=' after '!'"), start);
            }
        } else {
            throw PredicateParseError(std::string("Unexpected character: ") + c, start);
        }
    }
    
    tokens.emplace_back(TokenType::END_OF_INPUT, "", _pos);
    return tokens;
}

PredicateParser::PredicateParser(const std::string& predicate_str) {
    PredicateLexer lexer(predicate_str);
    _tokens = lexer.tokenize();
}

PredicateExpression PredicateParser::parse() {
    if (_tokens.empty() || (_tokens.size() == 1 && _tokens[0].type == TokenType::END_OF_INPUT)) {
        return PredicateExpression();
    }
    
    PredicateExpression expr = parse_expression();
    
    // Verify we consumed all tokens at the top level
    if (!check(TokenType::END_OF_INPUT)) {
        error("Unexpected token after predicate expression");
    }
    
    return expr;
}

PredicateExpression PredicateParser::parse_predicates(const std::string& predicate_str) {
    if (predicate_str.empty()) {
        return PredicateExpression();
    }
    PredicateParser parser(predicate_str);
    return parser.parse();
}

Token& PredicateParser::current() {
    return _tokens[_pos];
}

Token& PredicateParser::peek() {
    return _tokens[_pos];
}

Token PredicateParser::advance() {
    if (!check(TokenType::END_OF_INPUT)) {
        _pos++;
    }
    return _tokens[_pos - 1];
}

bool PredicateParser::check(TokenType type) const {
    return _tokens[_pos].type == type;
}

bool PredicateParser::match(TokenType type) {
    if (check(type)) {
        advance();
        return true;
    }
    return false;
}

void PredicateParser::expect(TokenType type, const std::string& message) {
    if (!check(type)) {
        error(message);
    }
    advance();
}

bool PredicateParser::is_predicate_op() const {
    TokenType t = _tokens[_pos].type;
    return t == TokenType::EQ || t == TokenType::NEQ ||
           t == TokenType::LT || t == TokenType::GT ||
           t == TokenType::LTE || t == TokenType::GTE ||
           t == TokenType::IN || t == TokenType::LIKE ||
           t == TokenType::NOT_LIKE || t == TokenType::NOT_IN ||
           t == TokenType::BETWEEN || t == TokenType::IS_NULL ||
           t == TokenType::IS_NOT_NULL || t == TokenType::NOTEXISTS;
}

bool PredicateParser::is_comparison_op(TokenType t) {
    return t == TokenType::EQ || t == TokenType::NEQ ||
           t == TokenType::LT || t == TokenType::GT ||
           t == TokenType::LTE || t == TokenType::GTE;
}

PredicateOp PredicateParser::token_to_predicate_op(TokenType type) const {
    switch (type) {
        case TokenType::EQ:       return PredicateOp::EQ;
        case TokenType::NEQ:      return PredicateOp::NEQ;
        case TokenType::LT:       return PredicateOp::LT;
        case TokenType::GT:       return PredicateOp::GT;
        case TokenType::LTE:      return PredicateOp::LTE;
        case TokenType::GTE:      return PredicateOp::GTE;
        case TokenType::IN:       return PredicateOp::IN;
        case TokenType::LIKE:     return PredicateOp::LIKE;
        case TokenType::NOT_LIKE: return PredicateOp::NOT_LIKE;
        case TokenType::NOT_IN:   return PredicateOp::NOT_IN;
        case TokenType::BETWEEN:  return PredicateOp::BETWEEN;
        case TokenType::IS_NULL: return PredicateOp::IS_NULL;
        case TokenType::IS_NOT_NULL: return PredicateOp::IS_NOT_NULL;
        default:
            throw PredicateParseError("Expected predicate operator", _tokens[_pos].position);
    }
}

void PredicateParser::error(const std::string& message) {
    throw PredicateParseError(message, current().position);
}

namespace {
static std::vector<PredicateGroup> expand_and_expression_to_or_groups(const PredicateExpression& expr) {
    // Convert an AND expression (which may contain OR groups) into OR-of-AND groups (DNF branches).
    std::vector<PredicateGroup> branches(1);
    branches[0].op = LogicalOp::AND;

    for (const auto& g : expr.groups) {
        if (g.op == LogicalOp::AND) {
            for (auto& b : branches) {
                b.predicates.insert(b.predicates.end(), g.predicates.begin(), g.predicates.end());
            }
            continue;
        }

        // OR group in an AND context -> distribute predicates across branches.
        std::vector<PredicateGroup> next;
        next.reserve(branches.size() * std::max<size_t>(1, g.predicates.size()));
        for (const auto& b : branches) {
            if (g.predicates.empty()) {
                next.push_back(b);
                continue;
            }
            for (const auto& p : g.predicates) {
                PredicateGroup nb = b;
                nb.predicates.push_back(p);
                next.push_back(std::move(nb));
            }
        }
        branches = std::move(next);
    }
    return branches;
}
} // namespace

// expression ::= and_term ( "OR" and_term )*
PredicateExpression PredicateParser::parse_expression() {
    PredicateExpression first = parse_and_term();
    if (!check(TokenType::OR)) {
        return first;
    }

    std::vector<PredicateGroup> disjuncts = expand_and_expression_to_or_groups(first);
    while (match(TokenType::OR)) {
        PredicateExpression rhs = parse_and_term();
        auto rhs_groups = expand_and_expression_to_or_groups(rhs);
        disjuncts.insert(disjuncts.end(), rhs_groups.begin(), rhs_groups.end());
    }

    PredicateExpression out;
    out.top_level_op = LogicalOp::OR;
    out.groups = std::move(disjuncts);
    return out;
}

// or_term ::= and_factor ( "AND" and_factor )*
// Kept for compatibility with existing declarations.
PredicateExpression PredicateParser::parse_or_term() {
    return parse_and_term();
}

// and_term ::= and_factor ( "AND" and_factor )*
// and_factor ::= predicate | "(" expression ")"
PredicateExpression PredicateParser::parse_and_term() {
    auto parse_factor_group = [&]() -> PredicateGroup {
        if (check(TokenType::LPAREN) && !is_predicate_op()) {
            advance();// consume '('
            PredicateExpression nested = parse_expression();
            expect(TokenType::RPAREN, "Expected ')' after grouped expression");

            PredicateGroup g;
            if (nested.top_level_op == LogicalOp::OR) {
                g.op = LogicalOp::OR;
                // Check if any branch has multiple predicates (AND sub-expression)
                bool has_complex_branch = false;
                for (const auto& ng : nested.groups) {
                    if (ng.predicates.size() > 1) { has_complex_branch = true; break; }
                }

                if (has_complex_branch) {
                    // Preserve AND branches using the branches field
                    for (const auto& ng : nested.groups) {
                        g.branches.push_back(ng);
                    }
                } else {
                    // Simple OR: all branches are single predicates, flatten as before
                    for (const auto& ng : nested.groups) {
                        if (!ng.predicates.empty()) {
                            g.predicates.push_back(ng.predicates[0]);
                        }
                    }
                }
                return g;
            }

            // AND-expression in parentheses: flatten all AND predicates.
            g.op = LogicalOp::AND;
            for (const auto& ng : nested.groups) {
                g.predicates.insert(g.predicates.end(), ng.predicates.begin(), ng.predicates.end());
            }
            return g;
        }

        Predicate pred = parse_predicate();
        PredicateGroup g;
        g.op = LogicalOp::AND;
        g.predicates.push_back(std::move(pred));
        return g;
    };

    PredicateExpression result;
    result.top_level_op = LogicalOp::AND;

    PredicateGroup merged_and;
    merged_and.op = LogicalOp::AND;
    auto flush_and = [&]() {
        if (!merged_and.predicates.empty()) {
            result.groups.push_back(std::move(merged_and));
            merged_and = PredicateGroup();
            merged_and.op = LogicalOp::AND;
        }
    };

    PredicateGroup first = parse_factor_group();
    if (first.op == LogicalOp::OR) {
        result.groups.push_back(std::move(first));
    } else {
        merged_and.predicates.insert(merged_and.predicates.end(), first.predicates.begin(), first.predicates.end());
    }

    while (match(TokenType::AND)) {
        PredicateGroup rhs = parse_factor_group();
        if (rhs.op == LogicalOp::OR) {
            flush_and();
            result.groups.push_back(std::move(rhs));
        } else {
            merged_and.predicates.insert(merged_and.predicates.end(), rhs.predicates.begin(), rhs.predicates.end());
        }
    }

    flush_and();
    if (result.groups.empty()) {
        PredicateGroup empty_group;
        empty_group.op = LogicalOp::AND;
        result.groups.push_back(std::move(empty_group));
    }
    return result;
}

// predicate ::= OP "(" identifier "," value ")"
//             | identifier COMP_OP value          (infix symbolic syntax)
//             | NOTEXISTS "(" identifier "," identifier "," identifier ["," direction] ")"
// value ::= identifier | number | string
// direction ::= "fwd" | "bwd"
Predicate PredicateParser::parse_predicate() {
    // Infix syntax: identifier <op> value  (e.g. "x < 5", "a != b", "price >= 100")
    if (check(TokenType::IDENTIFIER) && _pos + 1 < _tokens.size() && is_comparison_op(_tokens[_pos + 1].type)) {
        Token id_token = advance();
        Token op_token = advance();
        PredicateOp op = token_to_predicate_op(op_token.type);

        Predicate pred;
        pred.op = op;
        pred.left_attr = id_token.value;

        if (check(TokenType::IDENTIFIER)) {
            pred.type = PredicateType::ATTRIBUTE;
            pred.right_attr = advance().value;
        } else if (check(TokenType::NUMBER)) {
            pred.type = PredicateType::SCALAR;
            pred.scalar_value = advance().value;
        } else if (check(TokenType::STRING)) {
            pred.type = PredicateType::SCALAR;
            pred.scalar_value = advance().value;
        } else {
            error("Expected attribute name, number, or string after comparison operator");
        }
        return pred;
    }

    if (!is_predicate_op()) {
        error("Expected predicate operator (EQ, NEQ, LT, GT, LTE, GTE, IN, NOT_IN, IS_NULL, IS_NOT_NULL, LIKE, NOT_LIKE, BETWEEN, NOTEXISTS) or infix comparison (e.g. x < 5)");
    }
    
    Token op_token = advance();
    
    // Handle NOTEXISTS separately
    if (op_token.type == TokenType::NOTEXISTS) {
        return parse_not_exists_predicate();
    }
    
    PredicateOp op = token_to_predicate_op(op_token.type);
    
    expect(TokenType::LPAREN, "Expected '(' after predicate operator");
    
    // First argument: attribute name
    if (!check(TokenType::IDENTIFIER)) {
        error("Expected attribute name as first argument");
    }
    std::string left_attr = advance().value;
    
    // IS_NULL(attr) / IS_NOT_NULL(attr): unary scalar predicates
    if (op_token.type == TokenType::IS_NULL || op_token.type == TokenType::IS_NOT_NULL) {
        Predicate pred;
        pred.op = (op_token.type == TokenType::IS_NULL) ? PredicateOp::IS_NULL : PredicateOp::IS_NOT_NULL;
        pred.type = PredicateType::SCALAR;
        pred.left_attr = left_attr;
        expect(TokenType::RPAREN, "Expected ')' after IS_NULL/IS_NOT_NULL argument");
        return pred;
    }

    expect(TokenType::COMMA, "Expected ',' after first argument");
    
    // IN(...) and NOT_IN(...) have a variable number of scalar arguments
    if (op_token.type == TokenType::IN || op_token.type == TokenType::NOT_IN) {
        Predicate pred;
        pred.op = (op_token.type == TokenType::IN) ? PredicateOp::IN : PredicateOp::NOT_IN;
        pred.type = PredicateType::SCALAR;
        pred.left_attr = left_attr;

        // Parse 1+ values until ')'
        while (!check(TokenType::RPAREN)) {
            if (check(TokenType::NUMBER) || check(TokenType::STRING)) {
                pred.scalar_values.push_back(advance().value);
            } else {
                error("Expected number or string inside IN/NOT_IN(...)");
            }
            if (match(TokenType::COMMA)) {
                continue;
            }
            break;
        }

        expect(TokenType::RPAREN, "Expected ')' after IN/NOT_IN predicate arguments");
        return pred;
    }
    
    // BETWEEN(attr, min, max) has exactly 3 arguments
    if (op_token.type == TokenType::BETWEEN) {
        Predicate pred;
        pred.op = PredicateOp::BETWEEN;
        pred.type = PredicateType::SCALAR;
        pred.left_attr = left_attr;
        
        // Parse min value
        if (check(TokenType::NUMBER) || check(TokenType::STRING)) {
            pred.scalar_value = advance().value;
        } else {
            error("Expected min value (number or string) as second argument in BETWEEN");
        }
        
        expect(TokenType::COMMA, "Expected ',' after min value in BETWEEN");
        
        // Parse max value
        if (check(TokenType::NUMBER) || check(TokenType::STRING)) {
            pred.scalar_value2 = advance().value;
        } else {
            error("Expected max value (number or string) as third argument in BETWEEN");
        }
        
        expect(TokenType::RPAREN, "Expected ')' after BETWEEN predicate arguments");
        return pred;
    }

    // Second argument: can be identifier (attribute), number, or string
    Predicate pred;
    pred.op = op;
    pred.left_attr = left_attr;
    
    if (check(TokenType::IDENTIFIER)) {
        // Could be another attribute (attribute predicate)
        std::string second_value = advance().value;
        pred.type = PredicateType::ATTRIBUTE;
        pred.right_attr = second_value;
    } else if (check(TokenType::NUMBER)) {
        // Scalar numeric predicate
        pred.type = PredicateType::SCALAR;
        pred.scalar_value = advance().value;
    } else if (check(TokenType::STRING)) {
        // Scalar string predicate
        pred.type = PredicateType::SCALAR;
        pred.scalar_value = advance().value;
    } else {
        error("Expected attribute name, number, or string as second argument");
    }
    
    expect(TokenType::RPAREN, "Expected ')' after predicate arguments");
    
    return pred;
}

// NOTEXISTS "(" left_attr "," right_attr ")"
Predicate PredicateParser::parse_not_exists_predicate() {
    expect(TokenType::LPAREN, "Expected '(' after NOTEXISTS");
    
    // First argument: left attribute (source)
    if (!check(TokenType::IDENTIFIER)) {
        error("Expected source attribute name as first argument");
    }
    std::string left_attr = advance().value;
    
    expect(TokenType::COMMA, "Expected ',' after first argument");
    
    // Second argument: right attribute (target)
    if (!check(TokenType::IDENTIFIER)) {
        error("Expected target attribute name as second argument");
    }
    std::string right_attr = advance().value;
    
    expect(TokenType::RPAREN, "Expected ')' after NOTEXISTS arguments");
    
    // Create NOT_EXISTS predicate
    Predicate pred;
    pred.type = PredicateType::NOT_EXISTS;
    pred.op = PredicateOp::NEQ;  // Not used, but set for consistency
    pred.left_attr = left_attr;
    pred.right_attr = right_attr;
    
    return pred;
}

} // namespace ffx

