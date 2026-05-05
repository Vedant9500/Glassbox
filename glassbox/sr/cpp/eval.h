#pragma once

#define _USE_MATH_DEFINES
#include "ast.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <array>

// MSVC fallbacks for M_PI and M_E
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_E
#define M_E 2.71828182845904523536
#endif

namespace sr {

inline double& arithmetic_temperature_ref() {
    static double t = 5.0;
    return t;
}

inline void set_arithmetic_temperature(double t) {
    // Keep temperature in a numerically stable range.
    arithmetic_temperature_ref() = std::clamp(t, 0.1, 100.0);
}

inline double get_arithmetic_temperature() {
    return arithmetic_temperature_ref();
}

inline double stabilized_tau(double tau) {
    constexpr double kMinAbsTau = 1e-3;
    if (std::abs(tau) >= kMinAbsTau) return tau;
    return (tau >= 0.0) ? kMinAbsTau : -kMinAbsTau;
}

inline double power_sign_blend(double p) {
    // Only treat near-integers as parity-sensitive; otherwise use sign-preserving power.
    double p_round = std::round(p);
    if (std::abs(p - p_round) < 1e-6) {
        long long p_int = static_cast<long long>(p_round);
        return (p_int % 2 == 0) ? 1.0 : 0.0;
    }
    return 0.0;
}

inline std::array<double, 4> arithmetic_soft_weights(const OpNode& node) {
    double d_add = (node.beta - 1.0) * (node.beta - 1.0) + (node.gamma - 1.0) * (node.gamma - 1.0);
    double d_mul = (node.beta - 2.0) * (node.beta - 2.0) + (node.gamma - 1.0) * (node.gamma - 1.0);
    double d_div = (node.beta - 2.0) * (node.beta - 2.0) + (node.gamma + 1.0) * (node.gamma + 1.0);
    double d_sub = (node.beta - 1.0) * (node.beta - 1.0) + (node.gamma + 1.0) * (node.gamma + 1.0);

    double t = get_arithmetic_temperature();
    double max_logit = std::max({-d_add * t, -d_mul * t, -d_div * t, -d_sub * t});
    double w_add = std::exp(-d_add * t - max_logit);
    double w_mul = std::exp(-d_mul * t - max_logit);
    double w_div = std::exp(-d_div * t - max_logit);
    double w_sub = std::exp(-d_sub * t - max_logit);
    double sum_w = w_add + w_mul + w_div + w_sub;
    if (sum_w <= 0.0 || !std::isfinite(sum_w)) {
        return {0.25, 0.25, 0.25, 0.25};
    }
    return {w_add / sum_w, w_mul / sum_w, w_div / sum_w, w_sub / sum_w};
}

// Evaluates the output of a single graph given feature columns X
inline Eigen::ArrayXd evaluate_graph(const IndividualGraph& graph, const std::vector<Eigen::ArrayXd>& X, int num_samples) {
    if (graph.nodes.empty()) {
        return Eigen::ArrayXd::Zero(num_samples);
    }
    
    // 1. THE REGISTER FILE PATTERN (Zero-Allocation)
    // Pre-allocate a contiguous matrix instead of dynamic arrays for every node.
    // Thread-local ensures we don't re-allocate across thousands of evaluations per thread.
    thread_local Eigen::ArrayXXd arena;
    if (arena.rows() != num_samples || arena.cols() < static_cast<int>(graph.nodes.size())) {
        // Over-allocate columns to avoid resizing frequently
        arena.resize(num_samples, std::max(static_cast<int>(graph.nodes.size()), 64));
    }
    
    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        const auto& node = graph.nodes[i];
        
        switch (node.type) {
            case NodeType::Input: {
                if (node.feature_idx >= 0 && node.feature_idx < static_cast<int>(X.size())) {
                    arena.col(i) = X[node.feature_idx];
                } else {
                    arena.col(i).setZero();
                }
                break;
            }
            case NodeType::Constant: {
                arena.col(i).setConstant(node.value);
                break;
            }
            case NodeType::Unary: {
                const auto x = arena.col(node.left_child);
                switch (node.unary_op) {
                    case UnaryOp::Periodic: {
                        arena.col(i) = node.amplitude * (node.omega * x + node.phi).sin();
                        break;
                    }
                    case UnaryOp::Power: {
                        auto abs_x = x.abs() + 1e-10;
                        auto sign_x = x.sign();
                        auto abs_pow = abs_x.pow(node.p);

                        double is_even = power_sign_blend(node.p);
                        arena.col(i) = (1.0 - is_even) * (sign_x * abs_pow) + is_even * abs_pow;
                        arena.col(i) = arena.col(i).max(-1e8).min(1e8);
                        break;
                    }
                    case UnaryOp::IntPow: {
                        int n = static_cast<int>(std::round(node.p));
                        n = std::clamp(n, 2, 6);
                        arena.col(i) = x.pow(n).max(-1e8).min(1e8);
                        break;
                    }
                    case UnaryOp::Exp: {
                        arena.col(i) = (node.omega * x + node.phi).exp().max(-1e6).min(1e6);
                        break;
                    }
                    case UnaryOp::Log: {
                        arena.col(i) = (x.abs() + 1e-6).log().max(-1e6).min(1e6);
                        break;
                    }
                }
                break;
            }
            case NodeType::Binary: {
                const auto x = arena.col(node.left_child);
                const auto y = arena.col(node.right_child);
                
                switch (node.binary_op) {
                    case BinaryOp::Arithmetic: {
                        auto w = arithmetic_soft_weights(node);
                        auto res_add = x + y;
                        auto res_sub = x - y;
                        auto res_mul = x * y;
                        auto res_div = x / (1.0 + y.square()).sqrt();

                        arena.col(i) = (w[0] * res_add + w[1] * res_mul + w[2] * res_div + w[3] * res_sub).max(-1e6).min(1e6);
                        break;
                    }
                    case BinaryOp::Division: {
                        arena.col(i) = (x / (y.abs() + 1e-6) * y.sign()).max(-1e6).min(1e6);
                        break;
                    }
                    case BinaryOp::Aggregation: {
                        double local_tau = stabilized_tau(node.tau);
                        auto max_val = x.max(y);
                        auto exp_x = ((x - max_val) / local_tau).exp();
                        auto exp_y = ((y - max_val) / local_tau).exp();
                        auto sum_exp = exp_x + exp_y;
                        arena.col(i) = (x * exp_x / sum_exp) + (y * exp_y / sum_exp);
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
            final_output += graph.output_weights[i] * arena.col(i);
        }
    }
    
    return final_output;
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
                        auto abs_x = x.abs() + 1e-10;
                        auto sign_x = x.sign();
                        auto abs_pow = abs_x.pow(node.p);

                        double is_even = power_sign_blend(node.p);
                        cache_out[i] = (1.0 - is_even) * (sign_x * abs_pow) + is_even * abs_pow;
                        // Clamp
                        cache_out[i] = cache_out[i].max(-1e8).min(1e8);
                        break;
                    }
                    case UnaryOp::IntPow: {
                        int n = static_cast<int>(std::round(node.p));
                        n = std::clamp(n, 2, 6);
                        cache_out[i] = x.pow(n).max(-1e8).min(1e8);
                        break;
                    }
                    case UnaryOp::Exp: {
                        // exp(omega*x + phi) — omega enables sign (exp(-x)), phi enables shift
                        cache_out[i] = (node.omega * x + node.phi).exp().max(-1e6).min(1e6);
                        break;
                    }
                    case UnaryOp::Log: {
                        cache_out[i] = (x.abs() + 1e-6).log().max(-1e6).min(1e6);
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
                        auto w = arithmetic_soft_weights(node);
                        auto res_add = x + y;
                        auto res_sub = x - y;
                        auto res_mul = x * y;
                        auto res_div = x / (1.0 + y.square()).sqrt();

                        cache_out[i] = (w[0] * res_add + w[1] * res_mul + w[2] * res_div + w[3] * res_sub).max(-1e6).min(1e6);
                        break;
                    }
                    case BinaryOp::Division: {
                        cache_out[i] = (x / (y.abs() + 1e-6) * y.sign()).max(-1e6).min(1e6);
                        break;
                    }
                    case BinaryOp::Aggregation: {
                        // Simplification for MetaAggregation (just simple sum here to avoid complexity of tau-softmax)
                        // In full implementation we might need full softmax over stacked (x,y)
                        double local_tau = stabilized_tau(node.tau);
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

inline void evaluate_graph_partial(const IndividualGraph& graph, 
                                   int perturbed_node_idx,
                                   const std::vector<Eigen::ArrayXd>& old_cache,
                                   std::vector<Eigen::ArrayXd>& new_cache_out,
                                   std::vector<int>& changed_indices_out) {
    if (graph.nodes.empty()) return;
    
    new_cache_out = old_cache;
    changed_indices_out.clear();
    
    std::vector<bool> changed(graph.nodes.size(), false);
    changed[perturbed_node_idx] = true;
    changed_indices_out.push_back(perturbed_node_idx);

    for (size_t i = perturbed_node_idx; i < graph.nodes.size(); ++i) {
        const auto& node = graph.nodes[i];
        bool needs_eval = false;

        if (i == static_cast<size_t>(perturbed_node_idx)) {
            needs_eval = true;
        } else if (node.type == NodeType::Unary && changed[node.left_child]) {
            needs_eval = true;
            changed[i] = true;
            changed_indices_out.push_back(i);
        } else if (node.type == NodeType::Binary && (changed[node.left_child] || changed[node.right_child])) {
            needs_eval = true;
            changed[i] = true;
            changed_indices_out.push_back(i);
        }

        if (needs_eval) {
            switch (node.type) {
                case NodeType::Unary: {
                    const auto& x = new_cache_out[node.left_child];
                    switch (node.unary_op) {
                        case UnaryOp::Periodic:
                            new_cache_out[i] = node.amplitude * (node.omega * x + node.phi).sin();
                            break;
                        case UnaryOp::Power: {
                            auto abs_x = x.abs() + 1e-10;
                            auto sign_x = x.sign();
                            auto abs_pow = abs_x.pow(node.p);
                            double is_even = power_sign_blend(node.p);
                            new_cache_out[i] = (1.0 - is_even) * (sign_x * abs_pow) + is_even * abs_pow;
                            new_cache_out[i] = new_cache_out[i].max(-1e8).min(1e8);
                            break;
                        }
                        case UnaryOp::IntPow: {
                            int n = static_cast<int>(std::round(node.p));
                            n = std::clamp(n, 2, 6);
                            new_cache_out[i] = x.pow(n).max(-1e8).min(1e8);
                            break;
                        }
                        case UnaryOp::Exp: {
                            new_cache_out[i] = (node.omega * x + node.phi).exp().max(-1e6).min(1e6);
                            break;
                        }
                        case UnaryOp::Log: {
                            new_cache_out[i] = (x.abs() + 1e-6).log().max(-1e6).min(1e6);
                            break;
                        }
                    }
                    break;
                }
                case NodeType::Binary: {
                    const auto& x = new_cache_out[node.left_child];
                    const auto& y = new_cache_out[node.right_child];
                    switch (node.binary_op) {
                        case BinaryOp::Arithmetic: {
                            auto w = arithmetic_soft_weights(node);
                            auto res_add = x + y;
                            auto res_sub = x - y;
                            auto res_mul = x * y;
                            auto res_div = x / (1.0 + y.square()).sqrt();
                            new_cache_out[i] = (w[0] * res_add + w[1] * res_mul + w[2] * res_div + w[3] * res_sub).max(-1e6).min(1e6);
                            break;
                        }
                        case BinaryOp::Division: {
                            new_cache_out[i] = (x / (y.square() + 1e-12).sqrt()).max(-1e6).min(1e6);
                            break;
                        }
                        case BinaryOp::Aggregation: {
                            double local_tau = stabilized_tau(node.tau);
                            auto max_val = x.max(y);
                            auto exp_x = ((x - max_val) / local_tau).exp();
                            auto exp_y = ((y - max_val) / local_tau).exp();
                            auto sum_exp = exp_x + exp_y;
                            new_cache_out[i] = (x * exp_x / sum_exp) + (y * exp_y / sum_exp);
                            break;
                        }
                    }
                    break;
                }
                default: break;
            }
        }
    }
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
                        auto abs_x = x.abs() + 1e-10;
                        auto sign_x = x.sign();
                        auto abs_pow = abs_x.pow(node.p);
                        double is_even = power_sign_blend(node.p);
                        cache_out[i] = (1.0 - is_even) * (sign_x * abs_pow) + is_even * abs_pow;
                        cache_out[i] = cache_out[i].max(-1e8).min(1e8);
                        break;
                    }
                    case UnaryOp::IntPow: {
                        int n = static_cast<int>(std::round(node.p));
                        n = std::clamp(n, 2, 6);
                        cache_out[i] = x.pow(n).max(-1e8).min(1e8);
                        break;
                    }
                    case UnaryOp::Exp: {
                        cache_out[i] = (node.omega * x + node.phi).exp().max(-1e6).min(1e6);
                        break;
                    }
                    case UnaryOp::Log: {
                        cache_out[i] = (x.abs() + 1e-6).log().max(-1e6).min(1e6);
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
                        auto w = arithmetic_soft_weights(node);
                        auto res_add = x + y;
                        auto res_sub = x - y;
                        auto res_mul = x * y;
                        auto res_div = x / (1.0 + y.square()).sqrt();
                        cache_out[i] = (w[0] * res_add + w[1] * res_mul + w[2] * res_div + w[3] * res_sub).max(-1e6).min(1e6);
                        break;
                    }
                    case BinaryOp::Division: {
                        cache_out[i] = (x / (y.abs() + 1e-6) * y.sign()).max(-1e6).min(1e6);
                        break;
                    }
                    case BinaryOp::Aggregation: {
                        double local_tau = stabilized_tau(node.tau);
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
                case UnaryOp::Periodic: {
                    // Build clean periodic string: [amp*]sin([omega*]child[ + phi])
                    std::string result = "";
                    
                    // Amplitude: omit if ~1.0
                    bool has_amp = std::abs(node.amplitude - 1.0) > 1e-4;
                    if (has_amp) {
                        if (std::abs(node.amplitude - std::round(node.amplitude)) < 1e-6) {
                            snprintf(buf, sizeof(buf), "%d*", (int)std::round(node.amplitude));
                        } else {
                            snprintf(buf, sizeof(buf), "%.4g*", node.amplitude);
                        }
                        result += std::string(buf);
                    }
                    
                    result += "sin(";
                    
                    // Omega: omit if ~1.0, use integer if whole number
                    bool has_omega = std::abs(node.omega - 1.0) > 1e-4;
                    if (has_omega) {
                        if (std::abs(node.omega - std::round(node.omega)) < 1e-6) {
                            snprintf(buf, sizeof(buf), "%d*", (int)std::round(node.omega));
                        } else {
                            snprintf(buf, sizeof(buf), "%.4g*", node.omega);
                        }
                        result += std::string(buf);
                    }
                    result += child_str;
                    
                    // Phase: omit if ~0.0
                    bool has_phi = std::abs(node.phi) > 1e-4;
                    if (has_phi) {
                        if (std::abs(node.phi - std::round(node.phi)) < 1e-6) {
                            snprintf(buf, sizeof(buf), " + %d", (int)std::round(node.phi));
                        } else {
                            snprintf(buf, sizeof(buf), " + %.4g", node.phi);
                        }
                        result += std::string(buf);
                    }
                    
                    result += ")";
                    return result;
                }
                case UnaryOp::Power: {
                    // Use integer if p is a whole number
                    if (std::abs(node.p - std::round(node.p)) < 1e-6) {
                        snprintf(buf, sizeof(buf), "(%s)^%d", child_str.c_str(), (int)std::round(node.p));
                    } else {
                        snprintf(buf, sizeof(buf), "sign(%s)*(abs(%s))^%.4g", child_str.c_str(), child_str.c_str(), node.p);
                    }
                    return std::string(buf);
                }
                case UnaryOp::IntPow: {
                    int n = static_cast<int>(std::round(node.p));
                    n = std::clamp(n, 2, 6);
                    snprintf(buf, sizeof(buf), "(%s)^%d", child_str.c_str(), n);
                    return std::string(buf);
                }
                case UnaryOp::Exp: {
                    // Build exp string: exp([omega*]child[ + phi])
                    std::string exp_arg = "";
                    bool has_omega_e = std::abs(node.omega - 1.0) > 1e-4;
                    if (has_omega_e) {
                        if (std::abs(node.omega - (-1.0)) < 1e-6) {
                            exp_arg += "-";
                        } else if (std::abs(node.omega - std::round(node.omega)) < 1e-6) {
                            snprintf(buf, sizeof(buf), "%d*", (int)std::round(node.omega));
                            exp_arg += std::string(buf);
                        } else {
                            snprintf(buf, sizeof(buf), "%.4g*", node.omega);
                            exp_arg += std::string(buf);
                        }
                    }
                    exp_arg += child_str;
                    bool has_phi_e = std::abs(node.phi) > 1e-4;
                    if (has_phi_e) {
                        if (std::abs(node.phi - std::round(node.phi)) < 1e-6) {
                            snprintf(buf, sizeof(buf), " + %d", (int)std::round(node.phi));
                        } else {
                            snprintf(buf, sizeof(buf), " + %.4g", node.phi);
                        }
                        exp_arg += std::string(buf);
                    }
                    return "exp(" + exp_arg + ")";
                }
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
                    auto w = arithmetic_soft_weights(node);
                    constexpr double kNearDiscrete = 0.98;
                    double max_w = std::max({w[0], w[1], w[2], w[3]});
                    if (max_w >= kNearDiscrete) {
                        if (max_w == w[0]) return "(" + l_str + " + " + r_str + ")";
                        if (max_w == w[3]) return "(" + l_str + " - " + r_str + ")";
                        if (max_w == w[1]) return "(" + l_str + " * " + r_str + ")";
                        return "(" + l_str + " / sqrt(1.0 + (" + r_str + ")^2))";
                    }

                    std::string blend = "(";
                    bool first = true;
                    auto append_term = [&](double ww, const std::string& expr) {
                        if (ww < 1e-3) return;
                        char wbuf[64];
                        snprintf(wbuf, sizeof(wbuf), "%.3g", ww);
                        if (!first) blend += " + ";
                        blend += std::string(wbuf) + "*" + expr;
                        first = false;
                    };
                    append_term(w[0], "(" + l_str + " + " + r_str + ")");
                    append_term(w[1], "(" + l_str + " * " + r_str + ")");
                    append_term(w[2], "(" + l_str + " / " + r_str + ")");
                    append_term(w[3], "(" + l_str + " - " + r_str + ")");
                    if (first) return "(" + l_str + " + " + r_str + ")";
                    blend += ")";
                    return blend;
                    break;
                }
                case BinaryOp::Division:
                    return "(" + l_str + " / " + r_str + ")";
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
                double abs_w = std::abs(w);
                if (std::abs(abs_w - std::round(abs_w)) < 1e-6) {
                    snprintf(buf, sizeof(buf), "%d*", (int)std::round(abs_w));
                } else {
                    snprintf(buf, sizeof(buf), "%.4g*", abs_w);
                }
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
        double abs_bias = std::abs(graph.output_bias);
        if (std::abs(abs_bias - std::round(abs_bias)) < 1e-6) {
            snprintf(buf, sizeof(buf), "%d", (int)std::round(abs_bias));
        } else {
            snprintf(buf, sizeof(buf), "%.4g", abs_bias);
        }
        final_formula += std::string(buf);
    }
    
    if (final_formula.empty()) return "0";
    return final_formula;
}

} // namespace sr

