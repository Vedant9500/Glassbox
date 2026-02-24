#pragma once

#define _USE_MATH_DEFINES
#include "ast.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace sr {

// Evaluates the output of a single graph given feature columns X
inline Eigen::ArrayXd evaluate_graph(const IndividualGraph& graph, const std::vector<Eigen::ArrayXd>& X, int num_samples) {
    if (graph.nodes.empty()) {
        return Eigen::ArrayXd::Zero(num_samples);
    }
    
    // Cache for node values
    std::vector<Eigen::ArrayXd> cache(graph.nodes.size());
    
    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        const auto& node = graph.nodes[i];
        
        switch (node.type) {
            case NodeType::Input: {
                if (node.feature_idx >= 0 && node.feature_idx < X.size()) {
                    cache[i] = X[node.feature_idx];
                } else {
                    cache[i] = Eigen::ArrayXd::Zero(num_samples);
                }
                break;
            }
            case NodeType::Constant: {
                cache[i] = Eigen::ArrayXd::Constant(num_samples, node.value);
                break;
            }
            case NodeType::Unary: {
                // Assume left_child is valid
                const auto& x = cache[node.left_child];
                switch (node.unary_op) {
                    case UnaryOp::Periodic: {
                        cache[i] = node.amplitude * (node.omega * x + node.phi).sin();
                        break;
                    }
                    case UnaryOp::Power: {
                        // sign(x) * |x|^p as defined in MetaPower
                        auto abs_x = x.abs() + 1e-6;
                        auto sign_x = x.sign();
                        auto abs_pow = abs_x.pow(node.p);
                        
                        // MSVC sometimes misses M_PI even with _USE_MATH_DEFINES, define it explicitly
                        const double pi = 3.14159265358979323846;
                        double is_even = 0.5 * (1.0 + std::cos(node.p * pi));
                        cache[i] = (1.0 - is_even) * (sign_x * abs_pow) + is_even * abs_pow;
                        // Clamp
                        cache[i] = cache[i].max(-100.0).min(100.0);
                        break;
                    }
                    case UnaryOp::Exp: {
                        // We'll simplify MetaExp here: base^x
                        cache[i] = x.exp().max(-100.0).min(100.0);
                        break;
                    }
                    case UnaryOp::Log: {
                        cache[i] = (x.abs() + 1e-6).log().max(-100.0).min(100.0);
                        break;
                    }
                }
                break;
            }
            case NodeType::Binary: {
                const auto& x = cache[node.left_child];
                const auto& y = cache[node.right_child];
                
                switch (node.binary_op) {
                    case BinaryOp::Arithmetic: {
                        // Equivalent to MetaArithmeticExtended
                        double d_add = (node.beta - 1.0)*(node.beta - 1.0) + (node.gamma - 1.0)*(node.gamma - 1.0);
                        double d_mul = (node.beta - 2.0)*(node.beta - 2.0) + (node.gamma - 1.0)*(node.gamma - 1.0);
                        double d_div = (node.beta - 2.0)*(node.beta - 2.0) + (node.gamma + 1.0)*(node.gamma + 1.0);
                        double d_sub = (node.beta - 1.0)*(node.beta - 1.0) + (node.gamma + 1.0)*(node.gamma + 1.0);
                        
                        // Softmax over distances
                        double t = 5.0;
                        double max_logit = std::max({-d_add*t, -d_mul*t, -d_div*t, -d_sub*t});
                        double w_add = std::exp(-d_add*t - max_logit);
                        double w_mul = std::exp(-d_mul*t - max_logit);
                        double w_div = std::exp(-d_div*t - max_logit);
                        double w_sub = std::exp(-d_sub*t - max_logit);
                        double sum_w = w_add + w_mul + w_div + w_sub;
                        
                        w_add /= sum_w; w_mul /= sum_w; w_div /= sum_w; w_sub /= sum_w;
                        
                        auto res_add = x + y;
                        auto res_sub = x - y;
                        auto res_mul = x * y;
                        auto res_div = x / (y.abs() + 1e-6) * y.sign();
                        
                        cache[i] = (w_add * res_add + w_mul * res_mul + w_div * res_div + w_sub * res_sub).max(-100.0).min(100.0);
                        break;
                    }
                    case BinaryOp::Aggregation: {
                        // Simplification for MetaAggregation (just simple sum here to avoid complexity of tau-softmax)
                        // In full implementation we might need full softmax over stacked (x,y)
                        double local_tau = std::abs(node.tau) < 0.01 ? (node.tau > 0 ? 0.01 : -0.01) : node.tau;
                        auto max_val = x.max(y);
                        auto exp_x = ((x - max_val) / local_tau).exp();
                        auto exp_y = ((y - max_val) / local_tau).exp();
                        auto sum_exp = exp_x + exp_y;
                        cache[i] = (x * exp_x / sum_exp) + (y * exp_y / sum_exp);
                        break;
                    }
                }
                break;
            }
        }
    }
    return cache.back(); // Dummy return for original evaluate_graph before closing
}

// Output layer computation (weighted sum of nodes)
// Returns intermediate node outputs as well for least-squares solving if needed
inline Eigen::ArrayXd evaluate_graph(const IndividualGraph& graph, const std::vector<Eigen::ArrayXd>& X, int num_samples, std::vector<Eigen::ArrayXd>& cache_out) {
    if (graph.nodes.empty()) {
        cache_out.clear();
        return Eigen::ArrayXd::Zero(num_samples);
    }
    
    // Cache for node values
    cache_out.resize(graph.nodes.size());
    
    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        const auto& node = graph.nodes[i];
        
        switch (node.type) {
            case NodeType::Input: {
                if (node.feature_idx >= 0 && node.feature_idx < X.size()) {
                    cache_out[i] = X[node.feature_idx];
                } else {
                    cache_out[i] = Eigen::ArrayXd::Zero(num_samples);
                }
                break;
            }
            case NodeType::Constant: {
                cache_out[i] = Eigen::ArrayXd::Constant(num_samples, node.value);
                break;
            }
            case NodeType::Unary: {
                // Assume left_child is valid
                const auto& x = cache_out[node.left_child];
                switch (node.unary_op) {
                    case UnaryOp::Periodic: {
                        cache_out[i] = node.amplitude * (node.omega * x + node.phi).sin();
                        break;
                    }
                    case UnaryOp::Power: {
                        // sign(x) * |x|^p as defined in MetaPower
                        auto abs_x = x.abs() + 1e-6;
                        auto sign_x = x.sign();
                        auto abs_pow = abs_x.pow(node.p);
                        
                        // MSVC sometimes misses M_PI even with _USE_MATH_DEFINES, define it explicitly
                        const double pi = 3.14159265358979323846;
                        double is_even = 0.5 * (1.0 + std::cos(node.p * pi));
                        cache_out[i] = (1.0 - is_even) * (sign_x * abs_pow) + is_even * abs_pow;
                        // Clamp
                        cache_out[i] = cache_out[i].max(-100.0).min(100.0);
                        break;
                    }
                    case UnaryOp::Exp: {
                        // We'll simplify MetaExp here: base^x
                        cache_out[i] = x.exp().max(-100.0).min(100.0);
                        break;
                    }
                    case UnaryOp::Log: {
                        cache_out[i] = (x.abs() + 1e-6).log().max(-100.0).min(100.0);
                        break;
                    }
                }
                break;
            }
            case NodeType::Binary: {
                const auto& x = cache_out[node.left_child];
                const auto& y = cache_out[node.right_child];
                
                switch (node.binary_op) {
                    case BinaryOp::Arithmetic: {
                        // Equivalent to MetaArithmeticExtended
                        double d_add = (node.beta - 1.0)*(node.beta - 1.0) + (node.gamma - 1.0)*(node.gamma - 1.0);
                        double d_mul = (node.beta - 2.0)*(node.beta - 2.0) + (node.gamma - 1.0)*(node.gamma - 1.0);
                        double d_div = (node.beta - 2.0)*(node.beta - 2.0) + (node.gamma + 1.0)*(node.gamma + 1.0);
                        double d_sub = (node.beta - 1.0)*(node.beta - 1.0) + (node.gamma + 1.0)*(node.gamma + 1.0);
                        
                        // Softmax over distances
                        double t = 5.0;
                        double max_logit = std::max({-d_add*t, -d_mul*t, -d_div*t, -d_sub*t});
                        double w_add = std::exp(-d_add*t - max_logit);
                        double w_mul = std::exp(-d_mul*t - max_logit);
                        double w_div = std::exp(-d_div*t - max_logit);
                        double w_sub = std::exp(-d_sub*t - max_logit);
                        double sum_w = w_add + w_mul + w_div + w_sub;
                        
                        w_add /= sum_w; w_mul /= sum_w; w_div /= sum_w; w_sub /= sum_w;
                        
                        auto res_add = x + y;
                        auto res_sub = x - y;
                        auto res_mul = x * y;
                        auto res_div = x / (y.abs() + 1e-6) * y.sign();
                        
                        cache_out[i] = (w_add * res_add + w_mul * res_mul + w_div * res_div + w_sub * res_sub).max(-100.0).min(100.0);
                        break;
                    }
                    case BinaryOp::Aggregation: {
                        // Simplification for MetaAggregation (just simple sum here to avoid complexity of tau-softmax)
                        // In full implementation we might need full softmax over stacked (x,y)
                        double local_tau = std::abs(node.tau) < 0.01 ? (node.tau > 0 ? 0.01 : -0.01) : node.tau;
                        auto max_val = x.max(y);
                        auto exp_x = ((x - max_val) / local_tau).exp();
                        auto exp_y = ((y - max_val) / local_tau).exp();
                        auto sum_exp = exp_x + exp_y;
                        cache_out[i] = (x * exp_x / sum_exp) + (y * exp_y / sum_exp);
                        break;
                    }
                }
                break;
            }
        }
    }
    
    // Output layer computation (weighted sum of nodes)
    Eigen::ArrayXd final_output = Eigen::ArrayXd::Constant(num_samples, graph.output_bias);
    for (size_t i = 0; i < graph.output_weights.size() && i < graph.nodes.size(); ++i) {
        if (std::abs(graph.output_weights[i]) > 1e-6) {
            final_output += graph.output_weights[i] * cache_out[i];
        }
    }
    
    return final_output;
}

inline Eigen::ArrayXd evaluate_graph_simple(const IndividualGraph& graph, const std::vector<Eigen::ArrayXd>& X, int num_samples) {
    std::vector<Eigen::ArrayXd> cache;
    return evaluate_graph(graph, X, num_samples, cache);
}

// ── Cache-Aware Evaluation ──────────────────────────────────────────────
// Uses a shared SubtreeCache to skip recomputation of structurally identical
// subtrees across the population. Call with a shared cache per generation.
inline Eigen::ArrayXd evaluate_graph_cached(const IndividualGraph& graph,
                                             const std::vector<Eigen::ArrayXd>& X,
                                             int num_samples,
                                             std::vector<Eigen::ArrayXd>& cache_out,
                                             SubtreeCache& shared_cache) {
    if (graph.nodes.empty()) {
        cache_out.clear();
        return Eigen::ArrayXd::Zero(num_samples);
    }

    cache_out.resize(graph.nodes.size());
    std::vector<uint64_t> node_hashes(graph.nodes.size(), 0);

    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        // Compute structural hash for this node
        node_hashes[i] = compute_node_hash(graph, static_cast<int>(i), node_hashes);

        // Check shared cache
        auto it = shared_cache.find(node_hashes[i]);
        if (it != shared_cache.end() && it->second.size() == num_samples) {
            cache_out[i] = it->second; // Cache hit — skip computation
            continue;
        }

        // Cache miss — compute normally
        const auto& node = graph.nodes[i];
        switch (node.type) {
            case NodeType::Input: {
                if (node.feature_idx >= 0 && node.feature_idx < static_cast<int>(X.size())) {
                    cache_out[i] = X[node.feature_idx];
                } else {
                    cache_out[i] = Eigen::ArrayXd::Zero(num_samples);
                }
                break;
            }
            case NodeType::Constant: {
                cache_out[i] = Eigen::ArrayXd::Constant(num_samples, node.value);
                break;
            }
            case NodeType::Unary: {
                const auto& x = cache_out[node.left_child];
                switch (node.unary_op) {
                    case UnaryOp::Periodic: {
                        cache_out[i] = node.amplitude * (node.omega * x + node.phi).sin();
                        break;
                    }
                    case UnaryOp::Power: {
                        auto abs_x = x.abs() + 1e-6;
                        auto sign_x = x.sign();
                        auto abs_pow = abs_x.pow(node.p);
                        const double pi = 3.14159265358979323846;
                        double is_even = 0.5 * (1.0 + std::cos(node.p * pi));
                        cache_out[i] = (1.0 - is_even) * (sign_x * abs_pow) + is_even * abs_pow;
                        cache_out[i] = cache_out[i].max(-100.0).min(100.0);
                        break;
                    }
                    case UnaryOp::Exp: {
                        cache_out[i] = x.exp().max(-100.0).min(100.0);
                        break;
                    }
                    case UnaryOp::Log: {
                        cache_out[i] = (x.abs() + 1e-6).log().max(-100.0).min(100.0);
                        break;
                    }
                }
                break;
            }
            case NodeType::Binary: {
                const auto& x = cache_out[node.left_child];
                const auto& y = cache_out[node.right_child];
                switch (node.binary_op) {
                    case BinaryOp::Arithmetic: {
                        double d_add = (node.beta - 1.0)*(node.beta - 1.0) + (node.gamma - 1.0)*(node.gamma - 1.0);
                        double d_mul = (node.beta - 2.0)*(node.beta - 2.0) + (node.gamma - 1.0)*(node.gamma - 1.0);
                        double d_div = (node.beta - 2.0)*(node.beta - 2.0) + (node.gamma + 1.0)*(node.gamma + 1.0);
                        double d_sub = (node.beta - 1.0)*(node.beta - 1.0) + (node.gamma + 1.0)*(node.gamma + 1.0);
                        double t = 5.0;
                        double max_logit = std::max({-d_add*t, -d_mul*t, -d_div*t, -d_sub*t});
                        double w_add = std::exp(-d_add*t - max_logit);
                        double w_mul = std::exp(-d_mul*t - max_logit);
                        double w_div = std::exp(-d_div*t - max_logit);
                        double w_sub = std::exp(-d_sub*t - max_logit);
                        double sum_w = w_add + w_mul + w_div + w_sub;
                        w_add /= sum_w; w_mul /= sum_w; w_div /= sum_w; w_sub /= sum_w;
                        auto res_add = x + y;
                        auto res_sub = x - y;
                        auto res_mul = x * y;
                        auto res_div = x / (y.abs() + 1e-6) * y.sign();
                        cache_out[i] = (w_add * res_add + w_mul * res_mul + w_div * res_div + w_sub * res_sub).max(-100.0).min(100.0);
                        break;
                    }
                    case BinaryOp::Aggregation: {
                        double local_tau = std::abs(node.tau) < 0.01 ? (node.tau > 0 ? 0.01 : -0.01) : node.tau;
                        auto max_val = x.max(y);
                        auto exp_x = ((x - max_val) / local_tau).exp();
                        auto exp_y = ((y - max_val) / local_tau).exp();
                        auto sum_exp = exp_x + exp_y;
                        cache_out[i] = (x * exp_x / sum_exp) + (y * exp_y / sum_exp);
                        break;
                    }
                }
                break;
            }
        }

        // Store in shared cache (only for non-trivial nodes)
        if (node.type == NodeType::Unary || node.type == NodeType::Binary) {
            shared_cache[node_hashes[i]] = cache_out[i];
        }
    }

    // Output layer computation
    Eigen::ArrayXd final_output = Eigen::ArrayXd::Constant(num_samples, graph.output_bias);
    for (size_t i = 0; i < graph.output_weights.size() && i < graph.nodes.size(); ++i) {
        if (std::abs(graph.output_weights[i]) > 1e-6) {
            final_output += graph.output_weights[i] * cache_out[i];
        }
    }
    return final_output;
}

// Compute MSE fitness
inline double evaluate_fitness(IndividualGraph& graph, const std::vector<Eigen::ArrayXd>& X, const Eigen::ArrayXd& y, int num_samples) {
    Eigen::ArrayXd pred = evaluate_graph_simple(graph, X, num_samples);
    double mse = (pred - y).square().mean();
    graph.fitness = mse;
    return mse;
}

// Convert a node subtree to string
inline std::string format_node_to_string(const IndividualGraph& graph, int node_idx, int n_inputs) {
    if (node_idx < 0 || node_idx >= graph.nodes.size()) return "0";
    const auto& node = graph.nodes[node_idx];
    
    char buf[256];
    
    switch (node.type) {
        case NodeType::Input:
            if (n_inputs > 1) {
                return "x" + std::to_string(node.feature_idx);
            } else {
                return "x";
            }
        case NodeType::Constant:
            snprintf(buf, sizeof(buf), "%.4g", node.value);
            return std::string(buf);
        case NodeType::Unary: {
            std::string child_str = format_node_to_string(graph, node.left_child, n_inputs);
            switch (node.unary_op) {
                case UnaryOp::Periodic:
                    snprintf(buf, sizeof(buf), "%.4g*sin(%.4g*(%s) + %.4g)", node.amplitude, node.omega, child_str.c_str(), node.phi);
                    return std::string(buf);
                case UnaryOp::Power:
                    snprintf(buf, sizeof(buf), "(%s)^%.4g", child_str.c_str(), node.p);
                    return std::string(buf);
                case UnaryOp::Exp:
                    return "exp(" + child_str + ")";
                case UnaryOp::Log:
                    return "log(|" + child_str + "|)";
            }
            break;
        }
        case NodeType::Binary: {
            std::string l_str = format_node_to_string(graph, node.left_child, n_inputs);
            std::string r_str = format_node_to_string(graph, node.right_child, n_inputs);
            
            switch (node.binary_op) {
                case BinaryOp::Arithmetic: {
                    // Find largest weight
                    double d_add = (node.beta - 1.0)*(node.beta - 1.0) + (node.gamma - 1.0)*(node.gamma - 1.0);
                    double d_mul = (node.beta - 2.0)*(node.beta - 2.0) + (node.gamma - 1.0)*(node.gamma - 1.0);
                    double d_div = (node.beta - 2.0)*(node.beta - 2.0) + (node.gamma + 1.0)*(node.gamma + 1.0);
                    double d_sub = (node.beta - 1.0)*(node.beta - 1.0) + (node.gamma + 1.0)*(node.gamma + 1.0);
                    
                    double min_dist = std::min({d_add, d_mul, d_div, d_sub});
                    if (min_dist == d_add) return "(" + l_str + " + " + r_str + ")";
                    if (min_dist == d_sub) return "(" + l_str + " - " + r_str + ")";
                    if (min_dist == d_mul) return "(" + l_str + " * " + r_str + ")";
                    if (min_dist == d_div) return "(" + l_str + " / " + r_str + ")";
                    break;
                }
                case BinaryOp::Aggregation:
                    return "(" + l_str + " + " + r_str + ")/2"; // Simplified aggregation display
            }
            break;
        }
    }
    return "?";
}

// Convert entire graph to formula string
inline std::string get_formula_string(const IndividualGraph& graph, int n_inputs) {
    if (graph.nodes.empty()) return "0";
    
    std::string final_formula = "";
    bool first = true;
    
    char buf[256];
    for (size_t i = 0; i < graph.output_weights.size() && i < graph.nodes.size(); ++i) {
        double w = graph.output_weights[i];
        if (std::abs(w) > 1e-4) {
            std::string sub_formula = format_node_to_string(graph, static_cast<int>(i), n_inputs);
            
            if (!first) {
                final_formula += (w > 0) ? " + " : " - ";
            } else if (w < 0) {
                final_formula += "-";
            }
            first = false;
            
            if (std::abs(std::abs(w) - 1.0) > 1e-4) {
                snprintf(buf, sizeof(buf), "%.4g*", std::abs(w));
                final_formula += std::string(buf) + sub_formula;
            } else {
                final_formula += sub_formula;
            }
        }
    }
    
    if (std::abs(graph.output_bias) > 1e-4) {
        if (!first) {
            final_formula += (graph.output_bias > 0) ? " + " : " - ";
        } else if (graph.output_bias < 0) {
            final_formula += "-";
        }
        snprintf(buf, sizeof(buf), "%.4g", std::abs(graph.output_bias));
        final_formula += std::string(buf);
    }
    
    if (final_formula.empty()) return "0";
    return final_formula;
}

} // namespace sr
