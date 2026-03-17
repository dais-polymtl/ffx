#ifndef VFENGINE_PREDICATE_PARSER_HPP
#define VFENGINE_PREDICATE_PARSER_HPP

#include "predicate.hpp"
#include <stdexcept>
#include <string>
#include <vector>

namespace ffx {

// Token types for lexer
enum class TokenType {
    // Predicate operators
    EQ, NEQ, LT, GT, LTE, GTE,
    IN, LIKE, NOT_LIKE, NOT_IN, BETWEEN, IS_NULL, IS_NOT_NULL,
    // Special predicates
    NOTEXISTS,  // NOT EXISTS anti-semi-join
    // Logical operators
    AND, OR,
    // Punctuation
    LPAREN,   // (
    RPAREN,   // )
    COMMA,    // ,
    // Values
    IDENTIFIER,  // Attribute name
    NUMBER,      // Numeric literal
    STRING,      // Quoted string literal
    // End
    END_OF_INPUT
};

struct Token {
    TokenType type;
    std::string value;
    size_t position;  // Position in input string for error reporting
    
    Token(TokenType t, std::string v, size_t pos)
        : type(t), value(std::move(v)), position(pos) {}
};

// Lexer: converts predicate string to tokens
class PredicateLexer {
public:
    explicit PredicateLexer(const std::string& input);
    
    std::vector<Token> tokenize();

private:
    std::string _input;  // Store by value to avoid dangling reference
    size_t _pos = 0;
    
    char peek() const;
    char advance();
    void skip_whitespace();
    
    Token read_identifier_or_keyword();
    Token read_number();
    Token read_string();
    
    bool is_at_end() const { return _pos >= _input.size(); }
    bool is_digit(char c) const { return c >= '0' && c <= '9'; }
    bool is_alpha(char c) const { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'; }
    bool is_alnum(char c) const { return is_alpha(c) || is_digit(c); }
};

// Parser: converts tokens to PredicateExpression (recursive descent)
class PredicateParser {
public:
    explicit PredicateParser(const std::string& predicate_str);
    
    // Parse the predicate expression
    PredicateExpression parse();
    
    // Static utility to parse a predicate string
    static PredicateExpression parse_predicates(const std::string& predicate_str);

private:
    std::vector<Token> _tokens;
    size_t _pos = 0;
    
    // Recursive descent parsing methods
    // Grammar:
    //   expression  ::= or_term ( "OR" or_term )*
    //   or_term     ::= and_factor ( "AND" and_factor )*
    //   and_factor  ::= predicate | "(" expression ")"
    //   predicate   ::= OP "(" identifier "," value ")"
    //   value       ::= identifier | number | string
    
    PredicateExpression parse_expression();
    PredicateExpression parse_or_term();
    PredicateExpression parse_and_term();
    Predicate parse_predicate();
    Predicate parse_not_exists_predicate();
    
    // Helper methods
    Token& current();
    Token& peek();
    Token advance();
    bool check(TokenType type) const;
    bool match(TokenType type);
    void expect(TokenType type, const std::string& message);
    
    bool is_predicate_op() const;
    static bool is_comparison_op(TokenType type);
    PredicateOp token_to_predicate_op(TokenType type) const;
    
    [[noreturn]] void error(const std::string& message);
};

// Exception for parsing errors
class PredicateParseError : public std::runtime_error {
public:
    PredicateParseError(const std::string& message, size_t position)
        : std::runtime_error(message + " at position " + std::to_string(position)),
          _position(position) {}
    
    size_t position() const { return _position; }

private:
    size_t _position;
};

} // namespace ffx

#endif // VFENGINE_PREDICATE_PARSER_HPP

