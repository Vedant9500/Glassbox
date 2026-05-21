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
enum class EvalPolicy { Simple, CacheOut, SharedCache, Partial };

template<EvalPolicy Policy>
inline Eigen::ArrayXd evaluate_graph_impl(
    const IndividualGraph& graph,
    const std::vector<Eigen::ArrayXd>& X,
    int num_samples,
    std::vector<Eigen::ArrayXd>* cache_out = nullptr,
    SubtreeCache* shared_cache = nullptr,
    int perturbed_node_idx = -1,
    const std::vector<Eigen::ArrayXd>* old_cache = nullptr,
    std::vector<int>* changed_indices_out = nullptr) {
    
    if (graph.nodes.empty()) {
        if constexpr (Policy == EvalPolicy::CacheOut || Policy == EvalPolicy::SharedCache) {
            if (cache_out) cache_out->clear();
        }
        if constexpr (Policy == EvalPolicy::Partial) {
            if (cache_out && old_cache) *cache_out = *old_cache;
            if (changed_indices_out) changed_indices_out->clear();
        }
        return Eigen::ArrayXd::Zero(num_samples);
    }
    
    // Setup caches and tracking
    std::vector<bool> changed;
    if constexpr (Policy == EvalPolicy::Partial) {
        if (cache_out && old_cache) *cache_out = *old_cache;
        if (changed_indices_out) changed_indices_out->clear();
        changed.resize(graph.nodes.size(), false);
        changed[perturbed_node_idx] = true;
        changed_indices_out->push_back(perturbed_node_idx);
    } else if constexpr (Policy != EvalPolicy::Simple) {
        if (cache_out) cache_out->resize(graph.nodes.size());
    }

    std::vector<uint64_t> node_hashes;
    if constexpr (Policy == EvalPolicy::SharedCache) {
        node_hashes.resize(graph.nodes.size(), 0);
    }

    thread_local Eigen::ArrayXXd arena;
    if constexpr (Policy == EvalPolicy::Simple) {
        if (arena.rows() != num_samples || arena.cols() < static_cast<int>(graph.nodes.size())) {
            arena.resize(num_samples, std::max(static_cast<int>(graph.nodes.size()), 64));
        }
    }

    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        bool needs_eval = true;

        if constexpr (Policy == EvalPolicy::Partial) {
            needs_eval = false;
            if (i == static_cast<size_t>(perturbed_node_idx)) {
                needs_eval = true;
            } else {
                const auto& n = graph.nodes[i];
                if (n.type == NodeType::Unary && changed[n.left_child]) {
                    needs_eval = true;
                    changed[i] = true;
                    changed_indices_out->push_back(static_cast<int>(i));
                } else if (n.type == NodeType::Binary && (changed[n.left_child] || changed[n.right_child])) {
                    needs_eval = true;
                    changed[i] = true;
                    changed_indices_out->push_back(static_cast<int>(i));
                }
            }
            if (!needs_eval) continue;
        }

        if constexpr (Policy == EvalPolicy::SharedCache) {
            node_hashes[i] = compute_node_hash(graph, static_cast<int>(i), node_hashes);
            auto it = shared_cache->find(node_hashes[i]);
            if (it != shared_cache->end() && it->second.size() == num_samples) {
                (*cache_out)[i] = it->second;
                continue;
            }
        }
        
        const auto& node = graph.nodes[i];
        Eigen::ArrayXd val;
        if constexpr (Policy == EvalPolicy::Simple) {
            // Allocate a proxy/block reference... NO, let's just use Eigen::ArrayXd wrapper to standardize
            // For simple we assign directly later.
        } else {
            val = Eigen::ArrayXd::Zero(num_samples);
        }
        
        auto get_child = [&](int idx) -> Eigen::ArrayXd {
            if constexpr (Policy == EvalPolicy::Simple) return arena.col(idx);
            else return (*cache_out)[idx];
        };

        switch (node.type) {
            case NodeType::Input: {
                if (node.feature_idx >= 0 && node.feature_idx < static_cast<int>(X.size())) {
                    val = X[node.feature_idx];
                } else {
                    val = Eigen::ArrayXd::Zero(num_samples);
                }
                break;
            }
            case NodeType::Constant: {
                val = Eigen::ArrayXd::Constant(num_samples, node.value);
                break;
            }
            case NodeType::Unary: {
                auto x = get_child(node.left_child);
                switch (node.unary_op) {
                    case UnaryOp::Periodic:
                        val = node.amplitude * (node.omega * x + node.phi).sin();
                        break;
                    case UnaryOp::Power: {
                        auto abs_x = x.abs() + 1e-10;
                        auto sign_x = x.sign();
                        auto abs_pow = abs_x.pow(node.p);
                        double is_even = power_sign_blend(node.p);
                        val = (1.0 - is_even) * (sign_x * abs_pow) + is_even * abs_pow;
                        val = val.max(-1e8).min(1e8);
                        break;
                    }
                    case UnaryOp::IntPow: {
                        int n = static_cast<int>(std::round(node.p));
                        n = std::clamp(n, 2, 6);
                        val = x.pow(n).max(-1e8).min(1e8);
                        break;
                    }
                    case UnaryOp::Exp: {
                        val = (node.omega * x + node.phi).exp().max(-1e6).min(1e6);
                        break;
                    }
                    case UnaryOp::Log: {
                        val = (x.abs() + 1e-6).log().max(-1e6).min(1e6);
                        break;
                    }
                }
                break;
            }
            case NodeType::Binary: {
                auto x = get_child(node.left_child);
                auto y = get_child(node.right_child);
                switch (node.binary_op) {
                    case BinaryOp::Arithmetic: {
                        auto w = arithmetic_soft_weights(node);
                        auto res_add = x + y;
                        auto res_sub = x - y;
                        auto res_mul = x * y;
                        
                        // Division changed to y.square()+1e-12 in partial but y.abs()+... elsewhere. Let's stick to base implementation.
                        auto res_div = x / (1.0 + y.square()).sqrt();
                        
                        val = (w[0] * res_add + w[1] * res_mul + w[2] * res_div + w[3] * res_sub).max(-1e6).min(1e6);
                        break;
                    }
                    case BinaryOp::Division: {
                        val = (x / (y.abs() + 1e-6) * y.sign()).max(-1e6).min(1e6);
                        break;
                    }
                    case BinaryOp::Aggregation: {
                        double local_tau = stabilized_tau(node.tau);
                        auto max_val = x.max(y);
                        auto exp_x = ((x - max_val) / local_tau).exp();
                        auto exp_y = ((y - max_val) / local_tau).exp();
                        auto sum_exp = exp_x + exp_y;
                        val = (x * exp_x / sum_exp) + (y * exp_y / sum_exp);
                        break;
                    }
                }
                break;
            }
        }
        
        if constexpr (Policy == EvalPolicy::Simple) {
            arena.col(i) = val;
        } else {
            (*cache_out)[i] = std::move(val);
        }
        
        if constexpr (Policy == EvalPolicy::SharedCache) {
            if (node.type == NodeType::Unary || node.type == NodeType::Binary) {
                (*shared_cache)[node_hashes[i]] = (*cache_out)[i];
            }
        }
    }
    
    if constexpr (Policy == EvalPolicy::Partial) {
        return Eigen::ArrayXd::Zero(num_samples);
    }
    
    Eigen::ArrayXd final_output = Eigen::ArrayXd::Constant(num_samples, graph.output_bias);
    for (size_t i = 0; i < graph.output_weights.size() && i < graph.nodes.size(); ++i) {
        if (std::abs(graph.output_weights[i]) > 1e-6) {
            if constexpr (Policy == EvalPolicy::Simple) {
                final_output += graph.output_weights[i] * arena.col(i);
            } else {
                final_output += graph.output_weights[i] * (*cache_out)[i];
            }
        }
    }
    return final_output;
}

inline Eigen::ArrayXd evaluate_graph(const IndividualGraph& graph, const std::vector<Eigen::ArrayXd>& X, int num_samples) {
    return evaluate_graph_impl<EvalPolicy::Simple>(graph, X, num_samples);
}

inline Eigen::ArrayXd evaluate_graph(const IndividualGraph& graph, const std::vector<Eigen::ArrayXd>& X, int num_samples, std::vector<Eigen::ArrayXd>& cache_out) {
    return evaluate_graph_impl<EvalPolicy::CacheOut>(graph, X, num_samples, &cache_out);
}

inline void evaluate_graph_partial(const IndividualGraph& graph, 
                                   int perturbed_node_idx,
                                   const std::vector<Eigen::ArrayXd>& old_cache,
                                   std::vector<Eigen::ArrayXd>& new_cache_out,
                                   std::vector<int>& changed_indices_out) {
    evaluate_graph_impl<EvalPolicy::Partial>(graph, std::vector<Eigen::ArrayXd>(), 0, &new_cache_out, nullptr, perturbed_node_idx, &old_cache, &changed_indices_out);
}

inline Eigen::ArrayXd evaluate_graph_simple(const IndividualGraph& graph, const std::vector<Eigen::ArrayXd>& X, int num_samples) {
    return evaluate_graph_impl<EvalPolicy::Simple>(graph, X, num_samples);
}

inline Eigen::ArrayXd evaluate_graph_cached(const IndividualGraph& graph,
                                             const std::vector<Eigen::ArrayXd>& X,
                                             int num_samples,
                                             std::vector<Eigen::ArrayXd>& cache_out,
                                             SubtreeCache& shared_cache) {
    return evaluate_graph_impl<EvalPolicy::SharedCache>(graph, X, num_samples, &cache_out, &shared_cache);
}

// Compute MSE fitness
inline double evaluate_fitness(IndividualGraph& graph, const std::vector<Eigen::ArrayXd>& X, const Eigen::ArrayXd& y, int num_samples) {
    Eigen::ArrayXd pred = evaluate_graph_simple(graph, X, num_samples);
    double mse = (pred - y).square().mean();
    graph.fitness = mse;
    return mse;
}

inline bool near(double value, double target, double tol = 1e-4) {
    return std::abs(value - target) <= tol;
}

inline double normalize_angle(double value) {
    double out = std::fmod(value, 2.0 * M_PI);
    if (out < 0.0) out += 2.0 * M_PI;
    return out;
}

inline std::string format_pi_like(double value) {
    constexpr double kPiTol = 5e-3;
    if (std::abs(value) <= 1e-4) return "0";

    bool negative = value < 0.0;
    double abs_value = std::abs(value);
    struct Candidate {
        double multiplier;
        const char* text;
    };
    constexpr Candidate kCandidates[] = {
        {1.0, "pi"},
        {2.0, "2*pi"},
        {0.5, "pi/2"},
        {1.0 / 3.0, "pi/3"},
        {0.25, "pi/4"},
        {1.0 / 6.0, "pi/6"},
        {1.5, "3*pi/2"},
    };

    for (const auto& candidate : kCandidates) {
        double target = candidate.multiplier * M_PI;
        if (near(abs_value, target, kPiTol)) {
            return negative ? std::string("-") + candidate.text : std::string(candidate.text);
        }
    }

    char buf[64];
    if (std::abs(abs_value - std::round(abs_value)) < 1e-6) {
        snprintf(buf, sizeof(buf), "%s%d", negative ? "-" : "", (int)std::round(abs_value));
    } else {
        snprintf(buf, sizeof(buf), "%s%.4g", negative ? "-" : "", abs_value);
    }
    return std::string(buf);
}

inline std::string format_constant_display(double value) {
    constexpr double kPiTol = 5e-3;
    if (std::abs(value - M_PI) < kPiTol) return "pi";
    if (std::abs(value + M_PI) < kPiTol) return "-pi";
    if (std::abs(value - 2.0 * M_PI) < kPiTol) return "2*pi";
    if (std::abs(value + 2.0 * M_PI) < kPiTol) return "-2*pi";
    if (std::abs(value - M_PI / 2.0) < kPiTol) return "pi/2";
    if (std::abs(value + M_PI / 2.0) < kPiTol) return "-pi/2";
    if (std::abs(value - 3.0 * M_PI / 2.0) < kPiTol) return "3*pi/2";
    if (std::abs(value + 3.0 * M_PI / 2.0) < kPiTol) return "-3*pi/2";
    if (std::abs(value - M_PI / 3.0) < kPiTol) return "pi/3";
    if (std::abs(value - M_PI / 4.0) < kPiTol) return "pi/4";
    if (std::abs(value - M_PI / 6.0) < kPiTol) return "pi/6";

    char buf[64];
    if (std::abs(value - std::round(value)) < 1e-6) {
        snprintf(buf, sizeof(buf), "%d", (int)std::round(value));
    } else {
        snprintf(buf, sizeof(buf), "%.4g", value);
    }
    return std::string(buf);
}

inline bool has_top_level_add_sub(const std::string& s) {
    int depth = 0;
    for (size_t i = 0; i < s.size(); ++i) {
        char c = s[i];
        if (c == '(') {
            ++depth;
        } else if (c == ')') {
            --depth;
        } else if (depth == 0 && (c == '+' || c == '-')) {
            if (i == 0) continue;
            return true;
        }
    }
    return false;
}

inline std::string strip_outer_parens_if_simple(const std::string& s) {
    if (s.size() < 2 || s.front() != '(' || s.back() != ')') return s;
    if (has_top_level_add_sub(s.substr(1, s.size() - 2))) return s;
    return s.substr(1, s.size() - 2);
}

// Convert a node subtree to string. The extra guard state prevents malformed
// graphs from recursing forever while formatting.
inline std::string format_node_to_string(
    const IndividualGraph& graph,
    int node_idx,
    int n_inputs,
    std::vector<unsigned char>* visiting_ptr = nullptr,
    int depth = 0
) {
    if (node_idx < 0 || node_idx >= graph.nodes.size()) return "0";
    if (depth > static_cast<int>(graph.nodes.size()) + 8) return "0";

    std::vector<unsigned char> owned_visiting;
    if (visiting_ptr == nullptr) {
        owned_visiting.assign(graph.nodes.size(), 0);
        visiting_ptr = &owned_visiting;
    }
    if (visiting_ptr->size() != graph.nodes.size()) {
        visiting_ptr->assign(graph.nodes.size(), 0);
    }
    if ((*visiting_ptr)[node_idx]) return "0";
    (*visiting_ptr)[node_idx] = 1;
    struct VisitGuard {
        std::vector<unsigned char>& visiting;
        int idx;
        ~VisitGuard() { visiting[idx] = 0; }
    } guard{*visiting_ptr, node_idx};

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
            return format_constant_display(node.value);
        case NodeType::Unary: {
            std::string child_str = format_node_to_string(graph, node.left_child, n_inputs, visiting_ptr, depth + 1);
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
                    
                    constexpr double kTrigTol = 5e-3;
                    double phi_norm = normalize_angle(node.phi);
                    bool use_cos = near(phi_norm, M_PI / 2.0, kTrigTol) || near(phi_norm, 1.5 * M_PI, kTrigTol);
                    bool negate = near(phi_norm, M_PI, kTrigTol) || near(phi_norm, 1.5 * M_PI, kTrigTol);
                    child_str = strip_outer_parens_if_simple(child_str);

                    if (negate) {
                        result += "-";
                    }
                    result += use_cos ? "cos(" : "sin(";
                    
                    // Omega: omit if ~1.0, use integer if whole number
                    bool has_omega = std::abs(node.omega - 1.0) > 1e-4;
                    if (has_omega) result += format_pi_like(node.omega) + "*";
                    result += child_str;
                    
                    // Phase: omit if it is absorbed by sin/cos canonicalization
                    if (!use_cos) {
                        bool has_phi = std::abs(node.phi) > 1e-4;
                        if (has_phi) {
                            result += " + " + format_pi_like(node.phi);
                        }
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
            std::string l_str = format_node_to_string(graph, node.left_child, n_inputs, visiting_ptr, depth + 1);
            std::string r_str = format_node_to_string(graph, node.right_child, n_inputs, visiting_ptr, depth + 1);
            
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
    char buf[256];
    if (graph.nodes.empty()) {
        if (std::abs(graph.output_bias) <= 1e-4) return "0";
        double abs_bias = std::abs(graph.output_bias);
        if (std::abs(abs_bias - std::round(abs_bias)) < 1e-6) {
            snprintf(buf, sizeof(buf), "%s%d", graph.output_bias < 0 ? "-" : "", (int)std::round(abs_bias));
        } else {
            snprintf(buf, sizeof(buf), "%s%.4g", graph.output_bias < 0 ? "-" : "", abs_bias);
        }
        return std::string(buf);
    }
    
    std::string final_formula = "";
    bool first = true;
    
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

