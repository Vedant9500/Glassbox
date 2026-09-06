#pragma once

#include "ast.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace sr {

// Safe ctype wrappers (plain char may be signed; high-bit values are UB otherwise).
inline bool is_space_char(unsigned char c) { return std::isspace(c) != 0; }
inline bool is_digit_char(unsigned char c) { return std::isdigit(c) != 0; }
inline bool is_alpha_char(unsigned char c) { return std::isalpha(c) != 0; }
inline bool is_alnum_char(unsigned char c) { return std::isalnum(c) != 0; }


enum class TokenType {
    Number,
    Identifier,
    Plus,
    Minus,
    Mul,
    Div,
    Pow,
    LParen,
    RParen,
    Comma,
    End
};

struct Token {
    TokenType type;
    std::string text;
    double value = 0.0;
};

inline std::vector<Token> tokenize(const std::string& input) {
    std::vector<Token> tokens;
    size_t i = 0;
    while (i < input.size()) {
        char c = input[i];
        if (is_space_char(static_cast<unsigned char>(c))) {
            i++;
            continue;
        }
        if (c == '+') {
            tokens.push_back({TokenType::Plus, "+"});
            i++;
        } else if (c == '-') {
            tokens.push_back({TokenType::Minus, "-"});
            i++;
        } else if (c == '*') {
            if (i + 1 < input.size() && input[i+1] == '*') {
                tokens.push_back({TokenType::Pow, "**"});
                i += 2;
            } else {
                tokens.push_back({TokenType::Mul, "*"});
                i++;
            }
        } else if (c == '/') {
            tokens.push_back({TokenType::Div, "/"});
            i++;
        } else if (c == '^') {
            tokens.push_back({TokenType::Pow, "^"});
            i++;
        } else if (c == '(') {
            tokens.push_back({TokenType::LParen, "("});
            i++;
        } else if (c == ')') {
            tokens.push_back({TokenType::RParen, ")"});
            i++;
        } else if (c == ',') {
            tokens.push_back({TokenType::Comma, ","});
            i++;
        } else if (is_digit_char(static_cast<unsigned char>(c)) || c == '.') {
            std::string s;
            bool has_dot = false;
            bool has_exp = false;
            while (i < input.size()) {
                char curr = input[i];
                if (is_digit_char(static_cast<unsigned char>(curr))) {
                    s += curr;
                    i++;
                } else if (curr == '.' && !has_dot && !has_exp) {
                    s += curr;
                    has_dot = true;
                    i++;
                } else if ((curr == 'e' || curr == 'E') && !has_exp) {
                    s += curr;
                    has_exp = true;
                    i++;
                    if (i < input.size() && (input[i] == '+' || input[i] == '-')) {
                        s += input[i];
                        i++;
                    }
                } else {
                    break;
                }
            }
            tokens.push_back({TokenType::Number, s, std::stod(s)});
        } else if (is_alpha_char(static_cast<unsigned char>(c)) || c == '_') {
            std::string s;
            while (i < input.size() && (is_alnum_char(static_cast<unsigned char>(input[i])) || input[i] == '_')) {
                s += input[i];
                i++;
            }
            tokens.push_back({TokenType::Identifier, s});
        } else {
            throw std::runtime_error("Unexpected character in formula: " + std::string(1, c));
        }
    }
    tokens.push_back({TokenType::End, ""});
    return tokens;
}

enum class ParseNodeType {
    Input,
    Constant,
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Sin,
    Cos,
    Exp,
    Log,
    Abs,
    Sqrt
};

struct ParseNode {
    ParseNodeType type = ParseNodeType::Constant;
    double value = 0.0;
    int feature_idx = 0;
    std::shared_ptr<ParseNode> left = nullptr;
    std::shared_ptr<ParseNode> right = nullptr;
};

class Parser {
public:
    explicit Parser(const std::vector<Token>& tokens) : tokens_(tokens), pos_(0) {}

    std::shared_ptr<ParseNode> parse() {
        auto node = parse_expression();
        if (peek().type != TokenType::End) {
            throw std::runtime_error(
                "Unexpected token at end of formula: " + peek().text);
        }
        return node;
    }

private:
    std::vector<Token> tokens_;
    size_t pos_ = 0;

    Token peek() {
        if (pos_ >= tokens_.size()) return {TokenType::End, ""};
        return tokens_[pos_];
    }

    Token consume(TokenType expected) {
        Token t = peek();
        if (t.type != expected) {
            throw std::runtime_error("Expected token but got " + t.text);
        }
        pos_++;
        return t;
    }

    void advance() {
        pos_++;
    }

    std::shared_ptr<ParseNode> parse_expression() {
        auto node = parse_term();
        while (peek().type == TokenType::Plus || peek().type == TokenType::Minus) {
            Token op = peek();
            advance();
            auto right = parse_term();
            auto parent = std::make_shared<ParseNode>();
            parent->type = (op.type == TokenType::Plus) ? ParseNodeType::Add : ParseNodeType::Sub;
            parent->left = node;
            parent->right = right;
            node = parent;
        }
        return node;
    }

    std::shared_ptr<ParseNode> parse_term() {
        auto node = parse_unary();
        while (peek().type == TokenType::Mul || peek().type == TokenType::Div) {
            Token op = peek();
            advance();
            auto right = parse_unary();
            auto parent = std::make_shared<ParseNode>();
            parent->type = (op.type == TokenType::Mul) ? ParseNodeType::Mul : ParseNodeType::Div;
            parent->left = node;
            parent->right = right;
            node = parent;
        }
        return node;
    }

    std::shared_ptr<ParseNode> parse_unary() {
        Token t = peek();
        if (t.type == TokenType::Minus) {
            advance();
            auto child = parse_unary();
            if (child->type == ParseNodeType::Constant) {
                child->value = -child->value;
                return child;
            }
            auto parent = std::make_shared<ParseNode>();
            parent->type = ParseNodeType::Constant;
            parent->value = -1.0;
            auto mul = std::make_shared<ParseNode>();
            mul->type = ParseNodeType::Mul;
            mul->left = parent;
            mul->right = child;
            return mul;
        }
        if (t.type == TokenType::Plus) {
            advance();
            return parse_unary();
        }
        return parse_power();
    }

    std::shared_ptr<ParseNode> parse_power() {
        auto node = parse_primary();
        if (peek().type == TokenType::Pow) {
            advance();
            auto right = parse_unary(); // right-associative
            auto parent = std::make_shared<ParseNode>();
            parent->type = ParseNodeType::Pow;
            parent->left = node;
            parent->right = right;
            node = parent;
        }
        return node;
    }

    std::shared_ptr<ParseNode> parse_primary() {
        Token t = peek();
        if (t.type == TokenType::Number) {
            advance();
            auto node = std::make_shared<ParseNode>();
            node->type = ParseNodeType::Constant;
            node->value = t.value;
            return node;
        }
        if (t.type == TokenType::LParen) {
            advance();
            auto node = parse_expression();
            consume(TokenType::RParen);
            return node;
        }
        if (t.type == TokenType::Identifier) {
            advance();
            if (peek().type == TokenType::LParen) {
                advance();
                auto arg = parse_expression();
                consume(TokenType::RParen);
                
                auto node = std::make_shared<ParseNode>();
                if (t.text == "sin") node->type = ParseNodeType::Sin;
                else if (t.text == "cos") node->type = ParseNodeType::Cos;
                else if (t.text == "exp") node->type = ParseNodeType::Exp;
                else if (t.text == "log") node->type = ParseNodeType::Log;
                else if (t.text == "abs") node->type = ParseNodeType::Abs;
                else if (t.text == "sqrt") node->type = ParseNodeType::Sqrt;
                else {
                    throw std::runtime_error("Unsupported function name: " + t.text);
                }
                node->left = arg;
                return node;
            }
            
            if (t.text == "pi") {
                auto node = std::make_shared<ParseNode>();
                node->type = ParseNodeType::Constant;
                node->value = kPi;
                return node;
            }
            if (t.text == "E" || t.text == "e") {
                auto node = std::make_shared<ParseNode>();
                node->type = ParseNodeType::Constant;
                node->value = kE;
                return node;
            }

            if (t.text == "x" || (t.text.size() > 1 && t.text[0] == 'x' &&
                                   std::all_of(t.text.begin() + 1, t.text.end(),
                                               [](unsigned char ch) { return is_digit_char(ch); }))) {
                auto node = std::make_shared<ParseNode>();
                node->type = ParseNodeType::Input;
                node->feature_idx = (t.text == "x") ? 0 : std::stoi(t.text.substr(1));
                return node;
            }

            throw std::runtime_error("Unsupported symbol: " + t.text);
        }
        throw std::runtime_error("Unexpected token: " + t.text);
    }
};


inline std::string normalize_formula_string(std::string formula) {
    // Intentionally Unicode: rewrite common math glyphs (x, pi, sqrt, ...) to ASCII ops.
    auto replace_all = [](std::string& str, const std::string& from, const std::string& to) {
        size_t start_pos = 0;
        while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
            str.replace(start_pos, from.length(), to);
            start_pos += to.length();
        }
    };
    
    replace_all(formula, "·", "*");
    replace_all(formula, "⋅", "*");
    replace_all(formula, "•", "*");
    replace_all(formula, "×", "*");
    replace_all(formula, "÷", "/");
    replace_all(formula, "−", "-");
    replace_all(formula, "–", "-");
    replace_all(formula, "—", "-");

    
    replace_all(formula, "π²", "(pi^2)");
    replace_all(formula, "e²", "(e^2)");
    // §3.16: general superscript parity with benchmark normalizer so Phase-2
    // feature names (x²/x³) parse natively without external normalization.
    replace_all(formula, "²", "^2");
    replace_all(formula, "³", "^3");
    replace_all(formula, "2π", "(2*pi)");
    replace_all(formula, "π/2", "(pi/2)");
    replace_all(formula, "π/3", "(pi/3)");
    replace_all(formula, "π/4", "(pi/4)");
    replace_all(formula, "π/6", "(pi/6)");
    replace_all(formula, "1/π", "(1/pi)");
    replace_all(formula, "2/π", "(2/pi)");
    replace_all(formula, "√2", "sqrt(2)");
    replace_all(formula, "√3", "sqrt(3)");
    replace_all(formula, "√5", "sqrt(5)");
    replace_all(formula, "φ", "((1+sqrt(5))/2)");
    replace_all(formula, "log₂(e)", "(log(E)/log(2))");
    replace_all(formula, "log₁₀(e)", "(log(E)/log(10))");
    replace_all(formula, "π", "pi");
    replace_all(formula, "ln(", "log(");
    
    size_t bar_pos;
    bool is_opening = true;
    while ((bar_pos = formula.find('|')) != std::string::npos) {
        if (is_opening) {
            formula.replace(bar_pos, 1, "abs(");
            is_opening = false;
        } else {
            formula.replace(bar_pos, 1, ")");
            is_opening = true;
        }
    }
    
    size_t e_pow_pos;
    while ((e_pow_pos = formula.find("e^(")) != std::string::npos) {
        formula.replace(e_pow_pos, 3, "exp(");
    }
    
    replace_all(formula, "**", "^");

    // S5-14: "sin x" / "sin 2" / "abs (x)" → function(arg) for known unaries.
    {
        const char* funcs[] = {
            "sin", "cos", "tan", "exp", "log", "ln", "abs", "sqrt", "sign", "asin", "acos", "atan"
        };
        for (const char* fn : funcs) {
            const std::string f(fn);
            size_t pos = 0;
            while ((pos = formula.find(f, pos)) != std::string::npos) {
                if (pos > 0) {
                    char prev = formula[pos - 1];
                    if (std::isalnum(static_cast<unsigned char>(prev)) || prev == '_') {
                        pos += f.size();
                        continue;
                    }
                }
                size_t j = pos + f.size();
                while (j < formula.size() && std::isspace(static_cast<unsigned char>(formula[j]))) ++j;
                if (j >= formula.size() || formula[j] == '(') {
                    pos = j < formula.size() ? j + 1 : formula.size();
                    continue;
                }
                size_t start = j;
                size_t end = start;
                if (formula[start] == '(') {
                    int depth = 0;
                    for (; end < formula.size(); ++end) {
                        if (formula[end] == '(') ++depth;
                        else if (formula[end] == ')') {
                            --depth;
                            if (depth == 0) { ++end; break; }
                        }
                    }
                } else if (std::isdigit(static_cast<unsigned char>(formula[start])) || formula[start] == '.') {
                    bool has_dot = false, has_exp = false;
                    while (end < formula.size()) {
                        char c = formula[end];
                        if (std::isdigit(static_cast<unsigned char>(c))) { ++end; continue; }
                        if (c == '.' && !has_dot && !has_exp) { has_dot = true; ++end; continue; }
                        if ((c == 'e' || c == 'E') && !has_exp) {
                            has_exp = true; ++end;
                            if (end < formula.size() && (formula[end] == '+' || formula[end] == '-')) ++end;
                            continue;
                        }
                        break;
                    }
                } else if (std::isalpha(static_cast<unsigned char>(formula[start])) || formula[start] == '_') {
                    while (end < formula.size() &&
                           (std::isalnum(static_cast<unsigned char>(formula[end])) || formula[end] == '_')) {
                        ++end;
                    }
                } else {
                    pos = j + 1;
                    continue;
                }
                if (end <= start) { pos = j + 1; continue; }
                const std::string atom = formula.substr(start, end - start);
                const std::string repl = f + "(" + atom + ")";
                formula.replace(pos, end - pos, repl);
                pos += repl.size();
            }
        }
    }
    return formula;
}

// S5-14: insert explicit '*' for juxtaposition: 2x, 2(x+1), (x)(y), x(
inline std::vector<Token> insert_implicit_multiplication(std::vector<Token> tokens) {
    if (tokens.size() < 2) return tokens;
    auto is_value_like = [](const Token& t) {
        return t.type == TokenType::Number
            || t.type == TokenType::Identifier
            || t.type == TokenType::RParen;
    };
    auto is_atom_start = [](const Token& t) {
        return t.type == TokenType::Number
            || t.type == TokenType::Identifier
            || t.type == TokenType::LParen;
    };
    auto is_func_name = [](const Token& t) {
        if (t.type != TokenType::Identifier) return false;
        const std::string& s = t.text;
        return s == "sin" || s == "cos" || s == "tan" || s == "exp" || s == "log"
            || s == "ln" || s == "abs" || s == "sqrt" || s == "sign"
            || s == "asin" || s == "acos" || s == "atan";
    };
    std::vector<Token> out;
    out.reserve(tokens.size() * 2);
    for (size_t i = 0; i < tokens.size(); ++i) {
        out.push_back(tokens[i]);
        if (i + 1 >= tokens.size()) break;
        const Token& a = tokens[i];
        const Token& b = tokens[i + 1];
        if (!is_value_like(a) || !is_atom_start(b)) continue;
        // function call: sin (
        if (is_func_name(a) && b.type == TokenType::LParen) continue;
        // do not insert between two bare identifiers that look like multi-char vars only
        // (still insert for 2x via Number+Identifier, and x( via Identifier+LParen)
        out.push_back(Token{TokenType::Mul, "*"});
    }
    return out;
}

inline void collect_additive_terms(
    const std::shared_ptr<ParseNode>& node,
    double coeff,
    std::vector<std::pair<std::shared_ptr<ParseNode>, double>>& terms,
    double& bias
) {
    if (!node) return;

    if (node->type == ParseNodeType::Add) {
        collect_additive_terms(node->left, coeff, terms, bias);
        collect_additive_terms(node->right, coeff, terms, bias);
    } else if (node->type == ParseNodeType::Sub) {
        collect_additive_terms(node->left, coeff, terms, bias);
        collect_additive_terms(node->right, -coeff, terms, bias);
    } else if (node->type == ParseNodeType::Mul) {
        if (node->left->type == ParseNodeType::Constant) {
            collect_additive_terms(node->right, coeff * node->left->value, terms, bias);
        } else if (node->right->type == ParseNodeType::Constant) {
            collect_additive_terms(node->left, coeff * node->right->value, terms, bias);
        } else {
            terms.push_back({node, coeff});
        }
    } else if (node->type == ParseNodeType::Constant) {
        bias += coeff * node->value;
    } else {
        terms.push_back({node, coeff});
    }
}

class GraphCompiler {
public:
    IndividualGraph graph;
    std::vector<uint64_t> node_hashes;
    std::unordered_map<uint64_t, int> node_index_cache;
    int n_inputs = 1;

    // Constant-fold parse subtrees (e.g. 1/2, 2+3) so power exponents become
    // unary Power/IntPow instead of being discarded (S5-3).
    bool try_const_value(const std::shared_ptr<ParseNode>& pnode, double& out) const {
        if (!pnode) return false;
        switch (pnode->type) {
            case ParseNodeType::Constant:
                out = pnode->value;
                return std::isfinite(out);
            case ParseNodeType::Add:
            case ParseNodeType::Sub:
            case ParseNodeType::Mul:
            case ParseNodeType::Div:
            case ParseNodeType::Pow: {
                double a = 0.0, b = 0.0;
                if (!try_const_value(pnode->left, a) || !try_const_value(pnode->right, b)) return false;
                if (pnode->type == ParseNodeType::Add) out = a + b;
                else if (pnode->type == ParseNodeType::Sub) out = a - b;
                else if (pnode->type == ParseNodeType::Mul) out = a * b;
                else if (pnode->type == ParseNodeType::Div) {
                    if (std::abs(b) < 1e-15) return false;
                    out = a / b;
                } else {  // Pow
                    out = std::pow(a, b);
                }
                return std::isfinite(out);
            }
            case ParseNodeType::Abs: {
                double a = 0.0;
                if (!try_const_value(pnode->left, a)) return false;
                out = std::abs(a);
                return std::isfinite(out);
            }
            case ParseNodeType::Sqrt: {
                double a = 0.0;
                if (!try_const_value(pnode->left, a) || a < 0.0) return false;
                out = std::sqrt(a);
                return std::isfinite(out);
            }
            case ParseNodeType::Log: {
                double a = 0.0;
                if (!try_const_value(pnode->left, a)) return false;
                out = std::log(std::abs(a) + 1e-300);
                return std::isfinite(out);
            }
            case ParseNodeType::Exp: {
                double a = 0.0;
                if (!try_const_value(pnode->left, a)) return false;
                out = std::exp(a);
                return std::isfinite(out);
            }
            case ParseNodeType::Sin: {
                double a = 0.0;
                if (!try_const_value(pnode->left, a)) return false;
                out = std::sin(a);
                return std::isfinite(out);
            }
            case ParseNodeType::Cos: {
                double a = 0.0;
                if (!try_const_value(pnode->left, a)) return false;
                out = std::cos(a);
                return std::isfinite(out);
            }
            default:
                return false;
        }
    }

    int intern_node(OpNode node) {
        int temp_idx = static_cast<int>(graph.nodes.size());
        graph.nodes.push_back(node);
        node_hashes.push_back(0);
        uint64_t hash = compute_node_hash(graph, temp_idx, node_hashes);
        node_hashes[temp_idx] = hash;

        auto it = node_index_cache.find(hash);
        if (it != node_index_cache.end()) {
            graph.nodes.pop_back();
            node_hashes.pop_back();
            return it->second;
        }

        node_index_cache[hash] = temp_idx;
        return temp_idx;
    }

    int append_const(double value) {
        OpNode node;
        node.type = NodeType::Constant;
        node.value = value;
        return intern_node(node);
    }

    int append_unary_power(int base_idx, double p_val) {
        OpNode node;
        node.type = NodeType::Unary;
        node.left_child = base_idx;
        node.right_child = -1;
        // §3.330: non-finite exponents (e.g. x^(1/0)) are caller errors, not
        // seed material — fail loud instead of admitting poisoned Power nodes.
        // Finite out-of-domain exponents stay admitted by design: parsed seeds
        // are exempt from the evolution domain and the engine clamps them on
        // mutation/Adam (H-03), so clamping here would silently move seeds.
        if (!std::isfinite(p_val)) {
            throw std::runtime_error("Power exponent must be finite");
        }
        double p_round = std::round(p_val);
        if (std::abs(p_val - p_round) < 1e-9 && p_round >= 2.0 && p_round <= 6.0) {
            node.unary_op = UnaryOp::IntPow;
            node.p = p_round;
        } else {
            node.unary_op = UnaryOp::Power;
            node.p = p_val;
        }
        return intern_node(node);
    }

    // Variable exponent: sign(base) * exp(exp * log(|base|)), matching Unary Power domain.
    // §3.8 note: approximation, not exact Unary Power. No parity blend, 5 extra
    // nodes per variable power, and exp/log saturation (±1e6 out / log eps)
    // differs from the direct Power path (±1e8). Documented, not silently equal.
    int append_variable_power(int base_idx, int exp_idx) {
        // log(|base|) — Log already applies abs inside eval.
        OpNode log_n;
        log_n.type = NodeType::Unary;
        log_n.unary_op = UnaryOp::Log;
        log_n.left_child = base_idx;
        int log_idx = intern_node(log_n);

        // exp * log(|base|)
        OpNode mul_n;
        mul_n.type = NodeType::Binary;
        mul_n.binary_op = BinaryOp::Arithmetic;
        mul_n.beta = 2.0;
        mul_n.gamma = 1.0;
        mul_n.left_child = exp_idx;
        mul_n.right_child = log_idx;
        int mul_idx = intern_node(mul_n);

        // exp(exp * log(|base|)) = |base|^exp
        OpNode exp_n;
        exp_n.type = NodeType::Unary;
        exp_n.unary_op = UnaryOp::Exp;
        exp_n.omega = 1.0;
        exp_n.phi = 0.0;
        exp_n.left_child = mul_idx;
        int mag_idx = intern_node(exp_n);

        // abs(base)
        OpNode abs_n;
        abs_n.type = NodeType::Unary;
        abs_n.unary_op = UnaryOp::Abs;
        abs_n.left_child = base_idx;
        int abs_idx = intern_node(abs_n);

        // sign(base) ≈ base / abs(base) via protected Division
        OpNode div_n;
        div_n.type = NodeType::Binary;
        div_n.binary_op = BinaryOp::Division;
        div_n.left_child = base_idx;
        div_n.right_child = abs_idx;
        int sign_idx = intern_node(div_n);

        // sign(base) * |base|^exp
        OpNode out_n;
        out_n.type = NodeType::Binary;
        out_n.binary_op = BinaryOp::Arithmetic;
        out_n.beta = 2.0;
        out_n.gamma = 1.0;
        out_n.left_child = sign_idx;
        out_n.right_child = mag_idx;
        return intern_node(out_n);
    }

    int compile_node(const std::shared_ptr<ParseNode>& pnode) {
        if (!pnode) return -1;

        // Power: constant exponent (including folded 1/2) or variable rewrite (S5-3).
        if (pnode->type == ParseNodeType::Pow) {
            int base_idx = compile_node(pnode->left);
            if (base_idx < 0) return append_const(0.0);
            double p_val = 0.0;
            if (pnode->right && try_const_value(pnode->right, p_val)) {
                return append_unary_power(base_idx, p_val);
            }
            int exp_idx = compile_node(pnode->right);
            if (exp_idx < 0) return append_unary_power(base_idx, 1.0);
            return append_variable_power(base_idx, exp_idx);
        }

        int left_idx = compile_node(pnode->left);
        int right_idx = compile_node(pnode->right);

        OpNode node;
        node.left_child = left_idx;
        node.right_child = right_idx;

        switch (pnode->type) {
            case ParseNodeType::Input:
                node.type = NodeType::Input;
                node.feature_idx = pnode->feature_idx;
                if (pnode->feature_idx + 1 > n_inputs) {
                    n_inputs = pnode->feature_idx + 1;
                }
                break;
            case ParseNodeType::Constant:
                node.type = NodeType::Constant;
                node.value = pnode->value;
                break;
            case ParseNodeType::Sin:
                node.type = NodeType::Unary;
                node.unary_op = UnaryOp::Periodic;
                node.omega = 1.0;
                node.phi = 0.0;
                node.amplitude = 1.0;
                break;
            case ParseNodeType::Cos:
                node.type = NodeType::Unary;
                node.unary_op = UnaryOp::Periodic;
                node.omega = 1.0;
                node.phi = kPi / 2.0;
                node.amplitude = 1.0;
                break;
            case ParseNodeType::Exp:
                node.type = NodeType::Unary;
                node.unary_op = UnaryOp::Exp;
                node.omega = 1.0;
                node.phi = 0.0;
                break;
            case ParseNodeType::Log:
                node.type = NodeType::Unary;
                node.unary_op = UnaryOp::Log;
                break;
            case ParseNodeType::Abs:
                node.type = NodeType::Unary;
                node.unary_op = UnaryOp::Abs;
                break;
            case ParseNodeType::Sqrt:
                node.type = NodeType::Unary;
                node.unary_op = UnaryOp::Power;
                node.p = 0.5;
                break;
            case ParseNodeType::Add:
                node.type = NodeType::Binary;
                node.binary_op = BinaryOp::Arithmetic;
                node.beta = 1.0;
                node.gamma = 1.0;
                break;
            case ParseNodeType::Sub:
                node.type = NodeType::Binary;
                node.binary_op = BinaryOp::Arithmetic;
                node.beta = 1.0;
                node.gamma = -1.0;
                break;
            case ParseNodeType::Mul:
                node.type = NodeType::Binary;
                node.binary_op = BinaryOp::Arithmetic;
                node.beta = 2.0;
                node.gamma = 1.0;
                break;
            case ParseNodeType::Div:
                node.type = NodeType::Binary;
                node.binary_op = BinaryOp::Division;
                break;
            case ParseNodeType::Pow:
                // Handled above.
                node.type = NodeType::Constant;
                node.value = 0.0;
                break;
        }

        return intern_node(node);
    }
};

inline IndividualGraph formula_to_graph(const std::string& formula_str) {
    std::string norm = normalize_formula_string(formula_str);
    auto tokens = insert_implicit_multiplication(tokenize(norm));
    Parser parser(tokens);
    auto root = parser.parse();

    std::vector<std::pair<std::shared_ptr<ParseNode>, double>> terms;
    double bias = 0.0;
    collect_additive_terms(root, 1.0, terms, bias);

    GraphCompiler compiler;
    for (const auto& term_pair : terms) {
        int idx = compiler.compile_node(term_pair.first);
        while (compiler.graph.output_weights.size() <= idx) {
            compiler.graph.output_weights.push_back(0.0);
        }
        compiler.graph.output_weights[idx] += term_pair.second;
    }
    compiler.graph.output_bias = bias;

    compact_graph(compiler.graph);
    return compiler.graph;
}

} // namespace sr
