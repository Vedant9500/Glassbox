#pragma once

#define _USE_MATH_DEFINES
#include "ast.h"
#include "simplify.h"
#include <string>
#include <vector>
#include <cctype>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <memory>
#include <unordered_map>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_E
#define M_E 2.71828182845904523536
#endif

namespace sr {

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
        if (std::isspace(c)) {
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
        } else if (std::isdigit(c) || c == '.') {
            std::string s;
            bool has_dot = false;
            bool has_exp = false;
            while (i < input.size()) {
                char curr = input[i];
                if (std::isdigit(curr)) {
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
        } else if (std::isalpha(c) || c == '_') {
            std::string s;
            while (i < input.size() && (std::isalnum(input[i]) || input[i] == '_')) {
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
    ParseNodeType type;
    double value = 0.0;
    int feature_idx = 0;
    std::shared_ptr<ParseNode> left = nullptr;
    std::shared_ptr<ParseNode> right = nullptr;
};

class Parser {
    std::vector<Token> tokens;
    size_t pos = 0;

    Token peek() {
        if (pos >= tokens.size()) return {TokenType::End, ""};
        return tokens[pos];
    }

    Token consume(TokenType expected) {
        Token t = peek();
        if (t.type != expected) {
            throw std::runtime_error("Expected token but got " + t.text);
        }
        pos++;
        return t;
    }

    void advance() {
        pos++;
    }

public:
    Parser(const std::vector<Token>& tokens) : tokens(tokens) {}

    std::shared_ptr<ParseNode> parse() {
        auto node = parse_expression();
        if (peek().type != TokenType::End) {
            throw std::runtime_error("Unexpected token at end of formula: " + peek().text);
        }
        return node;
    }

private:
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
                node->value = M_PI;
                return node;
            }
            if (t.text == "E" || t.text == "e") {
                auto node = std::make_shared<ParseNode>();
                node->type = ParseNodeType::Constant;
                node->value = M_E;
                return node;
            }

            if (t.text == "x" || (t.text.size() > 1 && t.text[0] == 'x' &&
                                   std::all_of(t.text.begin() + 1, t.text.end(), ::isdigit))) {
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
    return formula;
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

struct GraphCompiler {
    IndividualGraph graph;
    std::vector<uint64_t> node_hashes;
    std::unordered_map<uint64_t, int> node_index_cache;
    int n_inputs = 1;

    int compile_node(const std::shared_ptr<ParseNode>& pnode) {
        if (!pnode) return -1;

        int left_idx = -1;
        int right_idx = -1;
        if (pnode->type == ParseNodeType::Pow) {
            left_idx = compile_node(pnode->left);
        } else {
            left_idx = compile_node(pnode->left);
            right_idx = compile_node(pnode->right);
        }

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
                node.phi = M_PI / 2.0;
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
                if (left_idx < 0) {
                    node.type = NodeType::Constant;
                    node.value = 0.0;
                    break;
                }
                if (pnode->right && pnode->right->type == ParseNodeType::Constant) {
                    double p_val = pnode->right->value;
                    double p_round = std::round(p_val);
                    if (std::abs(p_val - p_round) < 1e-9 && p_round >= 2.0 && p_round <= 6.0) {
                        node.type = NodeType::Unary;
                        node.unary_op = UnaryOp::IntPow;
                        node.p = p_round;
                        node.right_child = -1;
                    } else {
                        node.type = NodeType::Unary;
                        node.unary_op = UnaryOp::Power;
                        node.p = p_val;
                        node.right_child = -1;
                    }
                } else {
                    node.type = NodeType::Unary;
                    node.unary_op = UnaryOp::Power;
                    node.p = 1.0;
                    node.right_child = -1;
                }
                break;
        }

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
};

inline IndividualGraph formula_to_graph(const std::string& formula_str) {
    std::string norm = normalize_formula_string(formula_str);
    auto tokens = tokenize(norm);
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
