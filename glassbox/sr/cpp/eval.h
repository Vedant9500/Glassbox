#pragma once

#include "ast.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

namespace sr {

// kPi/kE live in ast.h (shared with parser/simplify).

// E4 note: soft Arithmetic is a continuous relaxation during search (kitchen-sink
// outer linear combo of node outputs + soft op blend). Higher temperature sharpens
// toward discrete +/*/div/-. Final cleanup snaps near-discrete gates (see evolution).
//
// H-06 / E10: temperature is primarily **thread_local** so concurrent engines /
// scoring jobs on different threads do not stomp each other. A process-wide
// atomic default seeds newly spawned OpenMP workers that have not set TLS yet.
// Always re-apply engine config temperature on fitness eval (and at the start of
// OMP worker bodies that evaluate) so workers pick up the intended value.
// ScopedArithmeticTemperature restores only the calling thread's TLS.
inline std::atomic<double>& arithmetic_temperature_default_ref() {
    static std::atomic<double> t{5.0};
    return t;
}

inline double& arithmetic_temperature_tls_ref() {
    // NaN => "unset"; fall back to process default (for fresh OMP workers).
    thread_local double t = std::numeric_limits<double>::quiet_NaN();
    return t;
}

// Backward-compatible alias used by older call sites / docs.
inline std::atomic<double>& arithmetic_temperature_ref() {
    return arithmetic_temperature_default_ref();
}

inline void set_arithmetic_temperature(double t) {
    // Keep temperature in a numerically stable range.
    t = std::clamp(t, 0.1, 100.0);
    arithmetic_temperature_tls_ref() = t;
    // Publish default so newly spawned workers inherit until they set TLS.
    arithmetic_temperature_default_ref().store(t, std::memory_order_relaxed);
}

inline double get_arithmetic_temperature() {
    const double tls = arithmetic_temperature_tls_ref();
    if (std::isfinite(tls) && tls > 0.0) {
        return tls;
    }
    return arithmetic_temperature_default_ref().load(std::memory_order_relaxed);
}

// RAII restore for nested API entry points on the **same thread**.
// Only TLS is restored; process default is left alone so concurrent engines
// on other threads keep their published default.
class ScopedArithmeticTemperature {
public:
    explicit ScopedArithmeticTemperature(double t)
        : prev_tls_(arithmetic_temperature_tls_ref())
        , had_tls_(std::isfinite(prev_tls_) && prev_tls_ > 0.0) {
        set_arithmetic_temperature(t);
    }
    ~ScopedArithmeticTemperature() {
        if (had_tls_) {
            arithmetic_temperature_tls_ref() = prev_tls_;
        } else {
            arithmetic_temperature_tls_ref() = std::numeric_limits<double>::quiet_NaN();
        }
    }
    ScopedArithmeticTemperature(const ScopedArithmeticTemperature&) = delete;
    ScopedArithmeticTemperature& operator=(const ScopedArithmeticTemperature&) = delete;

private:
    double prev_tls_;
    bool had_tls_;
};

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

    // Per-thread scratch for Simple policy (TLS; not process-global).
    thread_local Eigen::ArrayXXd arena;
    if constexpr (Policy == EvalPolicy::Simple) {
        const int n_nodes = static_cast<int>(graph.nodes.size());
        const int need_cols = std::max(n_nodes, 64);
        // S5-11: grow as needed; shrink when sample count drops a lot (avoid sticky huge arena).
        if (arena.rows() != num_samples || arena.cols() < need_cols) {
            arena.resize(num_samples, need_cols);
        } else if (arena.rows() > 0 && num_samples > 0 && arena.rows() > num_samples * 4) {
            arena.resize(num_samples, arena.cols());
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
                // S5-15: bounds-check child indices before indexing changed[].
                const int n_nodes_i = static_cast<int>(graph.nodes.size());
                if (n.type == NodeType::Unary
                    && n.left_child >= 0 && n.left_child < n_nodes_i
                    && changed[static_cast<size_t>(n.left_child)]) {
                    needs_eval = true;
                    changed[i] = true;
                    if (changed_indices_out) changed_indices_out->push_back(static_cast<int>(i));
                } else if (n.type == NodeType::Binary) {
                    bool ch = false;
                    if (n.left_child >= 0 && n.left_child < n_nodes_i
                        && changed[static_cast<size_t>(n.left_child)]) ch = true;
                    if (n.right_child >= 0 && n.right_child < n_nodes_i
                        && changed[static_cast<size_t>(n.right_child)]) ch = true;
                    if (ch) {
                        needs_eval = true;
                        changed[i] = true;
                        if (changed_indices_out) changed_indices_out->push_back(static_cast<int>(i));
                    }
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
        if constexpr (Policy != EvalPolicy::Simple) {
            val = Eigen::ArrayXd::Zero(num_samples);
        }
        
        // S5-11: do not force ArrayXd return type (that copied every child).
        // Simple path: arena column expression; cache paths: const ArrayXd&.
        auto get_child = [&](int idx) -> decltype(auto) {
            if constexpr (Policy == EvalPolicy::Simple) {
                return arena.col(idx);
            } else {
                return (*cache_out)[static_cast<size_t>(idx)];
            }
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
                const int n_nodes = static_cast<int>(graph.nodes.size());
                if (node.left_child < 0 || node.left_child >= n_nodes) {
                    val = Eigen::ArrayXd::Zero(num_samples);
                    break;
                }
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
                    case UnaryOp::Abs: {
                        val = x.abs();
                        break;
                    }
                }
                break;
            }
            case NodeType::Binary: {
                const int n_nodes = static_cast<int>(graph.nodes.size());
                if (node.left_child < 0 || node.left_child >= n_nodes ||
                    node.right_child < 0 || node.right_child >= n_nodes) {
                    val = Eigen::ArrayXd::Zero(num_samples);
                    break;
                }
                auto x = get_child(node.left_child);
                auto y = get_child(node.right_child);
                switch (node.binary_op) {
                    case BinaryOp::Arithmetic: {
                        auto w = arithmetic_soft_weights(node);
                        auto res_add = x + y;
                        auto res_sub = x - y;
                        auto res_mul = x * y;
                        
                        // Soft division: x / sqrt(1 + y^2) (protected; matches display S5-4).
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
            // P-01: only cache mid/deep operator subtrees. Depth-1 ops over
            // Input/Constant are cheap to recompute and dominate entry count.
            if (node.type == NodeType::Unary || node.type == NodeType::Binary) {
                auto is_op_child = [&](int child_idx) -> bool {
                    return child_idx >= 0
                        && child_idx < static_cast<int>(graph.nodes.size())
                        && (graph.nodes[static_cast<size_t>(child_idx)].type == NodeType::Unary
                            || graph.nodes[static_cast<size_t>(child_idx)].type == NodeType::Binary);
                };
                const bool worth_caching =
                    (node.type == NodeType::Unary && is_op_child(node.left_child))
                    || (node.type == NodeType::Binary
                        && (is_op_child(node.left_child) || is_op_child(node.right_child)));
                if (worth_caching) {
                    shared_cache->insert_or_assign(node_hashes[i], (*cache_out)[i]);
                }
            }
        }
    }
    
    if constexpr (Policy == EvalPolicy::Partial) {
        return Eigen::ArrayXd::Zero(num_samples);
    }
    
    Eigen::ArrayXd final_output = Eigen::ArrayXd::Constant(num_samples, graph.output_bias);
    for (size_t i = 0; i < graph.output_weights.size() && i < graph.nodes.size(); ++i) {
        if (std::abs(graph.output_weights[i]) > kOutputWeightActive) {
            if constexpr (Policy == EvalPolicy::Simple) {
                final_output += graph.output_weights[i] * arena.col(i);
            } else {
                final_output += graph.output_weights[i] * (*cache_out)[i];
            }
        }
    }
    return final_output;
}

inline Eigen::ArrayXd evaluate_graph(
    const IndividualGraph& graph,
    const std::vector<Eigen::ArrayXd>& X,
    int num_samples) {
    return evaluate_graph_impl<EvalPolicy::Simple>(graph, X, num_samples);
}

inline Eigen::ArrayXd evaluate_graph(
    const IndividualGraph& graph,
    const std::vector<Eigen::ArrayXd>& X,
    int num_samples,
    std::vector<Eigen::ArrayXd>& cache_out) {
    return evaluate_graph_impl<EvalPolicy::CacheOut>(
        graph, X, num_samples, &cache_out);
}

inline void evaluate_graph_partial(
    const IndividualGraph& graph,
    int perturbed_node_idx,
    const std::vector<Eigen::ArrayXd>& old_cache,
    std::vector<Eigen::ArrayXd>& new_cache_out,
    std::vector<int>& changed_indices_out) {
    evaluate_graph_impl<EvalPolicy::Partial>(
        graph,
        std::vector<Eigen::ArrayXd>(),
        0,
        &new_cache_out,
        nullptr,
        perturbed_node_idx,
        &old_cache,
        &changed_indices_out);
}

// Alias of 3-arg evaluate_graph (kept for call-site compatibility).
inline Eigen::ArrayXd evaluate_graph_simple(
    const IndividualGraph& graph,
    const std::vector<Eigen::ArrayXd>& X,
    int num_samples) {
    return evaluate_graph(graph, X, num_samples);
}

inline Eigen::ArrayXd evaluate_graph_cached(const IndividualGraph& graph,
                                             const std::vector<Eigen::ArrayXd>& X,
                                             int num_samples,
                                             std::vector<Eigen::ArrayXd>& cache_out,
                                             SubtreeCache& shared_cache) {
    return evaluate_graph_impl<EvalPolicy::SharedCache>(graph, X, num_samples, &cache_out, &shared_cache);
}

// Unweighted MSE only (legacy / tests). Search scoring uses
// EvolutionEngine::evaluate_fitness_with_penalty (weights + complexity penalties).
// S5-16: keep raw_mse/weighted_mse/fitness_valid in sync with that path's fields;
// without sample weights, weighted_mse mirrors unweighted MSE.
inline double evaluate_fitness(
    IndividualGraph& graph,
    const std::vector<Eigen::ArrayXd>& X,
    const Eigen::ArrayXd& y,
    int num_samples) {
    Eigen::ArrayXd pred = evaluate_graph(graph, X, num_samples);
    double mse = (pred - y).square().mean();
    graph.fitness = mse;
    graph.raw_mse = mse;
    graph.weighted_mse = mse;
    graph.fitness_valid = true;
    return mse;
}

inline bool near(double value, double target, double tol = 1e-4) {
    return std::abs(value - target) <= tol;
}

inline double normalize_angle(double value) {
    double out = std::fmod(value, 2.0 * kPi);
    if (out < 0.0) out += 2.0 * kPi;
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
        double target = candidate.multiplier * kPi;
        if (near(abs_value, target, kPiTol)) {
            return negative ? std::string("-") + candidate.text : std::string(candidate.text);
        }
    }

    char buf[64];
    if (std::abs(abs_value - std::round(abs_value)) < 1e-6) {
        snprintf(buf, sizeof(buf), "%s%d", negative ? "-" : "", static_cast<int>(std::round(abs_value)));
    } else {
        snprintf(buf, sizeof(buf), "%s%.4g", negative ? "-" : "", abs_value);
    }
    return std::string(buf);
}

inline std::string format_constant_display(double value) {
    constexpr double kPiTol = 5e-3;
    if (std::abs(value - kPi) < kPiTol) return "pi";
    if (std::abs(value + kPi) < kPiTol) return "-pi";
    if (std::abs(value - 2.0 * kPi) < kPiTol) return "2*pi";
    if (std::abs(value + 2.0 * kPi) < kPiTol) return "-2*pi";
    if (std::abs(value - kPi / 2.0) < kPiTol) return "pi/2";
    if (std::abs(value + kPi / 2.0) < kPiTol) return "-pi/2";
    if (std::abs(value - 3.0 * kPi / 2.0) < kPiTol) return "3*pi/2";
    if (std::abs(value + 3.0 * kPi / 2.0) < kPiTol) return "-3*pi/2";
    if (std::abs(value - kPi / 3.0) < kPiTol) return "pi/3";
    if (std::abs(value - kPi / 4.0) < kPiTol) return "pi/4";
    if (std::abs(value - kPi / 6.0) < kPiTol) return "pi/6";

    char buf[64];
    if (std::abs(value - std::round(value)) < 1e-6) {
        snprintf(buf, sizeof(buf), "%d", static_cast<int>(std::round(value)));
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
                    std::string result;
                    
                    // Amplitude: omit if ~1.0
                    bool has_amp = std::abs(node.amplitude - 1.0) > 1e-4;
                    if (has_amp) {
                        if (std::abs(node.amplitude - std::round(node.amplitude)) < 1e-6) {
                            snprintf(buf, sizeof(buf), "%d*", static_cast<int>(std::round(node.amplitude)));
                        } else {
                            snprintf(buf, sizeof(buf), "%.4g*", node.amplitude);
                        }
                        result += std::string(buf);
                    }
                    
                    constexpr double kTrigTol = 5e-3;
                    double phi_norm = normalize_angle(node.phi);
                    bool use_cos = near(phi_norm, kPi / 2.0, kTrigTol) || near(phi_norm, 1.5 * kPi, kTrigTol);
                    bool negate = near(phi_norm, kPi, kTrigTol) || near(phi_norm, 1.5 * kPi, kTrigTol);
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
                    // S5-14: match eval parity for near-integer p.
                    // Even integers use abs-base in evaluate_graph (power_sign_blend);
                    // print abs form so external string eval cannot diverge on negatives.
                    // Odd integers keep signed base; non-integers keep sign*(abs)^p.
                    if (std::abs(node.p - std::round(node.p)) < 1e-6) {
                        int n = static_cast<int>(std::round(node.p));
                        if ((n % 2) == 0) {
                            snprintf(buf, sizeof(buf), "(abs(%s))^%d", child_str.c_str(), n);
                        } else {
                            snprintf(buf, sizeof(buf), "(%s)^%d", child_str.c_str(), n);
                        }
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
                    std::string exp_arg;
                    bool has_omega_e = std::abs(node.omega - 1.0) > 1e-4;
                    if (has_omega_e) {
                        if (std::abs(node.omega - (-1.0)) < 1e-6) {
                            exp_arg += "-";
                        } else if (std::abs(node.omega - std::round(node.omega)) < 1e-6) {
                            snprintf(buf, sizeof(buf), "%d*", static_cast<int>(std::round(node.omega)));
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
                            snprintf(buf, sizeof(buf), " + %d", static_cast<int>(std::round(node.phi)));
                        } else {
                            snprintf(buf, sizeof(buf), " + %.4g", node.phi);
                        }
                        exp_arg += std::string(buf);
                    }
                    return "exp(" + exp_arg + ")";
                }
                case UnaryOp::Log:
                    return "log(|" + child_str + "|)";
                case UnaryOp::Abs:
                    return "abs(" + child_str + ")";
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
                    // Match eval soft-div: x / sqrt(1 + y^2), not true x/y (S5-4).
                    append_term(w[2], "(" + l_str + " / sqrt(1.0 + (" + r_str + ")^2))");
                    append_term(w[3], "(" + l_str + " - " + r_str + ")");
                    if (first) return "(" + l_str + " + " + r_str + ")";
                    blend += ")";
                    return blend;
                }
                case BinaryOp::Division:
                    // Match protected Division eval: x * sign(y) / (|y| + eps) (S5-4).
                    return "((" + l_str + ") * sign(" + r_str + ") / (abs(" + r_str + ") + 1e-6))";
                case BinaryOp::Aggregation: {
                    // Softmax-weighted mean of children (eval.h). Print forms that
                    // Python display eval can execute with abs/exp only (S5-4).
                    double tau = stabilized_tau(node.tau);
                    if (std::abs(tau) >= 10.0) {
                        return "((" + l_str + " + " + r_str + ")/2)";
                    }
                    // max(l,r) = 0.5*(l+r+abs(l-r)); min uses minus abs.
                    std::string max_lr = "(0.5*((" + l_str + ")+(" + r_str + ")+abs((" + l_str + ")-(" + r_str + "))))";
                    if (std::abs(tau) <= 1e-2) {
                        if (tau >= 0.0) return max_lr;
                        return "(0.5*((" + l_str + ")+(" + r_str + ")-abs((" + l_str + ")-(" + r_str + "))))";
                    }
                    char tbuf[64];
                    snprintf(tbuf, sizeof(tbuf), "%.6g", tau);
                    std::string t_str(tbuf);
                    std::string el = "exp(((" + l_str + ")-(" + max_lr + "))/" + t_str + ")";
                    std::string er = "exp(((" + r_str + ")-(" + max_lr + "))/" + t_str + ")";
                    return "(((" + l_str + ")*(" + el + ")+(" + r_str + ")*(" + er + "))/((" + el + ")+(" + er + ")))";
                }
            }
            break;
        }
    }
    return "?";
}

// Prefer explicit n_inputs, but never collapse multi-feature graphs to bare "x".
inline int effective_n_inputs(const IndividualGraph& graph, int n_inputs) {
    int max_f = -1;
    for (const auto& n : graph.nodes) {
        if (n.type == NodeType::Input) {
            max_f = std::max(max_f, n.feature_idx);
        }
    }
    int inferred = max_f + 1;
    if (inferred <= 1) return std::max(1, n_inputs);
    return std::max(std::max(1, n_inputs), inferred);
}

// Convert entire graph to formula string
inline std::string get_formula_string(const IndividualGraph& graph, int n_inputs) {
    n_inputs = effective_n_inputs(graph, n_inputs);
    char buf[256];
    if (graph.nodes.empty()) {
        if (std::abs(graph.output_bias) <= 1e-4) return "0";
        double abs_bias = std::abs(graph.output_bias);
        if (std::abs(abs_bias - std::round(abs_bias)) < 1e-6) {
            snprintf(buf, sizeof(buf), "%s%d", graph.output_bias < 0 ? "-" : "", static_cast<int>(std::round(abs_bias)));
        } else {
            snprintf(buf, sizeof(buf), "%s%.4g", graph.output_bias < 0 ? "-" : "", abs_bias);
        }
        return std::string(buf);
    }
    
    std::string final_formula;
    bool first = true;
    
    for (size_t i = 0; i < graph.output_weights.size() && i < graph.nodes.size(); ++i) {
        double w = graph.output_weights[i];
        // S5-6: match eval activity threshold (was 1e-4 -> silent dropped terms).
        if (std::abs(w) > kOutputWeightActive) {
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
                    snprintf(buf, sizeof(buf), "%d*", static_cast<int>(std::round(abs_w)));
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
            snprintf(buf, sizeof(buf), "%d", static_cast<int>(std::round(abs_bias)));
        } else {
            snprintf(buf, sizeof(buf), "%.4g", abs_bias);
        }
        final_formula += std::string(buf);
    }
    
    if (final_formula.empty()) return "0";
    return final_formula;
}

} // namespace sr

