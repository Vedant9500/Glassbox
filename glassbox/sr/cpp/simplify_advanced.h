#pragma once

#include "ast.h"
#include "eval.h"
#include "formula_parser.h"
#include "simplify.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

namespace sr {

inline double snap_val(double val, double int_tol, double zero_tol) {
    if (std::abs(val) <= zero_tol) return 0.0;
    double r = std::round(val);
    if (std::abs(val - r) <= int_tol) return r;
    return val;
}

inline bool is_valid_graph_topology(const IndividualGraph& graph) {
    if (graph.output_weights.size() > graph.nodes.size()) return false;
    for (size_t i = 0; i < graph.output_weights.size(); ++i) {
        if (!std::isfinite(graph.output_weights[i])) return false;
    }
    if (!std::isfinite(graph.output_bias)) return false;

    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        const auto& n = graph.nodes[i];
        auto valid_child = [&](int child) {
            return child >= 0 && child < static_cast<int>(i);
        };
        if (n.type == NodeType::Unary) {
            if (!valid_child(n.left_child)) return false;
        } else if (n.type == NodeType::Binary) {
            if (!valid_child(n.left_child) || !valid_child(n.right_child)) return false;
        }
    }
    return true;
}

inline bool is_trig_sq(const IndividualGraph& graph, int idx, const std::vector<uint64_t>& node_hashes, uint64_t& arg_hash, bool& is_cos) {
    if (idx < 0 || idx >= static_cast<int>(graph.nodes.size())) return false;
    const auto& n = graph.nodes[idx];
    if (n.type == NodeType::Unary && (n.unary_op == UnaryOp::Power || n.unary_op == UnaryOp::IntPow) && std::abs(n.p - 2.0) < 1e-5) {
        int child_idx = n.left_child;
        if (child_idx < 0 || child_idx >= static_cast<int>(graph.nodes.size())) return false;
        const auto& child = graph.nodes[child_idx];
        if (child.type == NodeType::Unary && child.unary_op == UnaryOp::Periodic && std::abs(child.amplitude - 1.0) < 1e-4 && std::abs(child.omega - 1.0) < 1e-4) {
            int arg_idx = child.left_child;
            if (arg_idx < 0 || arg_idx >= static_cast<int>(node_hashes.size())) return false;
            arg_hash = node_hashes[arg_idx];
            
            double phi_norm = std::fmod(child.phi, 2.0 * kPi);
            if (phi_norm < 0.0) phi_norm += 2.0 * kPi;
            
            // cos has phi = PI/2.0
            is_cos = std::abs(phi_norm - kPi / 2.0) < 1e-4 || std::abs(phi_norm - 1.5 * kPi) < 1e-4;
            bool is_sin = std::abs(phi_norm) < 1e-4 || std::abs(phi_norm - kPi) < 1e-4 || std::abs(phi_norm - 2.0 * kPi) < 1e-4;
            
            return is_cos || is_sin;
        }
    }
    return false;
}

inline void simplify_node_advanced(IndividualGraph& graph, int i, std::vector<int>& redirect, const std::vector<uint64_t>& node_hashes, double int_tol, double zero_tol) {
    OpNode& node = graph.nodes[i];
    
    if (node.type == NodeType::Unary) {
        if (node.left_child >= 0 && node.left_child < static_cast<int>(redirect.size())) {
            node.left_child = redirect[node.left_child];
        }
        if (node.left_child < 0 || node.left_child >= i) {
            node.type = NodeType::Constant;
            node.value = 0.0;
            node.left_child = -1;
            node.right_child = -1;
            return;
        }
        const auto& child = graph.nodes[node.left_child];
        
        // Constant folding
        if (child.type == NodeType::Constant) {
            double v = child.value;
            double res = 0.0;
            switch (node.unary_op) {
                case UnaryOp::Periodic: res = node.amplitude * std::sin(node.omega * v + node.phi); break;
                case UnaryOp::Power: {
                    double abs_v = std::abs(v) + 1e-10;
                    double sign_v = (v >= 0) ? 1.0 : -1.0;
                    double p_round = std::round(node.p);
                    bool is_even = (std::abs(node.p - p_round) < 1e-6) && (static_cast<long long>(p_round) % 2 == 0);
                    double abs_pow = std::pow(abs_v, node.p);
                    res = is_even ? abs_pow : sign_v * abs_pow;
                    break;
                }
                case UnaryOp::IntPow: res = std::pow(v, std::clamp(static_cast<int>(std::round(node.p)), 2, 6)); break;
                case UnaryOp::Exp:
                    // H-02: match eval clamp; bare exp can yield Inf.
                    res = std::exp(std::clamp(node.omega * v + node.phi, -50.0, 50.0));
                    res = std::clamp(res, -1e6, 1e6);
                    break;
                case UnaryOp::Log: res = std::log(std::abs(v) + 1e-6); break;
                case UnaryOp::Abs: res = std::abs(v); break;
            }
            // H-02: never store Inf/NaN constants from folds.
            if (!std::isfinite(res)) res = 0.0;
            res = std::clamp(res, -1e8, 1e8);
            node.type = NodeType::Constant;
            node.value = snap_val(res, int_tol, zero_tol);
            return;
        }
        
        // Identities
        if (node.unary_op == UnaryOp::Periodic) {
            node.omega = snap_val(node.omega, int_tol, zero_tol);
            node.amplitude = snap_val(node.amplitude, int_tol, zero_tol);
            node.phi = snap_val(node.phi, int_tol, zero_tol);
            
            if (node.amplitude == 0.0) {
                node.type = NodeType::Constant;
                node.value = 0.0;
            } else if (node.omega == 0.0) {
                node.type = NodeType::Constant;
                node.value = snap_val(node.amplitude * std::sin(node.phi), int_tol, zero_tol);
            }
        } else if (node.unary_op == UnaryOp::Power || node.unary_op == UnaryOp::IntPow) {
            node.p = snap_val(node.p, int_tol, zero_tol);
            if (node.p == 0.0) {
                node.type = NodeType::Constant;
                node.value = 1.0;
            } else if (node.p == 1.0) {
                redirect[i] = node.left_child;
            } else if (child.type == NodeType::Unary && (child.unary_op == UnaryOp::Power || child.unary_op == UnaryOp::IntPow)) {
                // §3.108: (x^a)^b = x^(a*b) changes signed-power domain
                // semantics for fractional b over negative x (exact gives
                // (x^0.5)^2=|x|^2-like at x=-4 while x^1 is -4). Collapse
                // only when domain-safe: both near-integers, or outer
                // integer (IntPow path is exact integer pow).
                const double pa = child.p;
                const double pb = node.p;
                const bool a_int = std::abs(pa - std::round(pa)) < 1e-9;
                const bool b_int = std::abs(pb - std::round(pb)) < 1e-9;
                const bool outer_intpow = (node.unary_op == UnaryOp::IntPow);
                if (a_int || b_int || outer_intpow) {
                    // (x^a)^b = x^(a*b)
                    node.p = snap_val(node.p * child.p, int_tol, zero_tol);
                    node.left_child = child.left_child;
                }
            }
        } else if (node.unary_op == UnaryOp::Exp) {
            node.omega = snap_val(node.omega, int_tol, zero_tol);
            node.phi = snap_val(node.phi, int_tol, zero_tol);
            if (node.omega == 0.0) {
                node.type = NodeType::Constant;
                // H-02: clamp exp(phi) fold like live eval (§3.7/§3.112:
                // arg ±500 then output ±1e6, not early ±50 saturation).
                double folded = std::exp(std::clamp(node.phi, -500.0, 500.0));
                if (!std::isfinite(folded)) folded = 0.0;
                folded = std::clamp(folded, -1e6, 1e6);
                node.value = snap_val(folded, int_tol, zero_tol);
            } else if (child.type == NodeType::Unary && child.unary_op == UnaryOp::Log && std::abs(node.omega - 1.0) < 1e-4 && std::abs(node.phi) < 1e-4) {
                // §3.109: exp(log(|y|)) = |y| holds only because Log eval
                // uses log(|y|+eps); omega/phi generality already gated above.
                node.unary_op = UnaryOp::Abs;
                node.left_child = child.left_child;
            }
        } else if (node.unary_op == UnaryOp::Log) {
            if (child.type == NodeType::Constant) {
                node.type = NodeType::Constant;
                node.value = snap_val(std::log(std::abs(child.value) + 1e-6), int_tol, zero_tol);
            } else if (child.type == NodeType::Unary && child.unary_op == UnaryOp::Exp && std::abs(child.omega - 1.0) < 1e-4 && std::abs(child.phi) < 1e-4) {
                // §3.109: log(exp(y)) = y valid only for omega~1, phi~0
                // (gated above); exp output is positive so log(|.|) is safe.
                // No epsilon/range collapse beyond the existing gate.
                redirect[i] = child.left_child;
            }
        } else if (node.unary_op == UnaryOp::Abs) {
            // abs(abs(y)) = abs(y); abs of non-negative-ish Power even integer may stay
            if (child.type == NodeType::Unary && child.unary_op == UnaryOp::Abs) {
                redirect[i] = node.left_child;
            }
        }
        
    } else if (node.type == NodeType::Binary) {
        if (node.left_child >= 0 && node.left_child < static_cast<int>(redirect.size())) {
            node.left_child = redirect[node.left_child];
        }
        if (node.right_child >= 0 && node.right_child < static_cast<int>(redirect.size())) {
            node.right_child = redirect[node.right_child];
        }
        if (node.left_child < 0 || node.left_child >= i || node.right_child < 0 || node.right_child >= i) {
            node.type = NodeType::Constant;
            node.value = 0.0;
            node.left_child = -1;
            node.right_child = -1;
            return;
        }
        const auto& left = graph.nodes[node.left_child];
        const auto& right = graph.nodes[node.right_child];
        
        // Constant folding
        if (left.type == NodeType::Constant && right.type == NodeType::Constant) {
            double l = left.value;
            double r = right.value;
            double res = 0.0;
            
            if (node.binary_op == BinaryOp::Arithmetic) {
                auto w = arithmetic_soft_weights(node);
                double max_w = std::max({w[0], w[1], w[2], w[3]});
                if (max_w == w[0]) res = l + r;
                else if (max_w == w[3]) res = l - r;
                else if (max_w == w[1]) res = l * r;
                else res = l / std::sqrt(1.0 + r*r);
            } else if (node.binary_op == BinaryOp::Division) {
                res = (l / (std::abs(r) + 1e-6)) * ((r >= 0) ? 1.0 : -1.0);
            }
            
            // H-02: sanitize folded binary constants.
            if (!std::isfinite(res)) res = 0.0;
            res = std::clamp(res, -1e8, 1e8);
            node.type = NodeType::Constant;
            node.value = snap_val(res, int_tol, zero_tol);
            return;
        }
        
        // Algebraic identities
        if (node.binary_op == BinaryOp::Arithmetic) {
            auto w = arithmetic_soft_weights(node);
            // §3.111: live eval blends all branches; identity rewrite must
            // only fire when near-discrete. 0.99 (not 0.95) avoids
            // contradicting live soft evaluation.
            constexpr double kNearDiscrete = 0.99;
            double max_w = std::max({w[0], w[1], w[2], w[3]});
            
            if (max_w >= kNearDiscrete) {
                if (max_w == w[0]) { // Add
                    if (left.type == NodeType::Constant && left.value == 0.0) {
                        redirect[i] = node.right_child;
                    } else if (right.type == NodeType::Constant && right.value == 0.0) {
                        redirect[i] = node.left_child;
                    } else {
                        // Check for sin^2(A) + cos^2(A) = 1
                        uint64_t left_arg_hash = 0, right_arg_hash = 0;
                        bool left_cos = false, right_cos = false;
                        if (is_trig_sq(graph, node.left_child, node_hashes, left_arg_hash, left_cos) &&
                            is_trig_sq(graph, node.right_child, node_hashes, right_arg_hash, right_cos)) {
                            if (left_arg_hash == right_arg_hash && left_cos != right_cos) {
                                node.type = NodeType::Constant;
                                node.value = 1.0;
                            }
                        }
                    }
                } else if (max_w == w[3]) { // Sub
                    if (right.type == NodeType::Constant && right.value == 0.0) {
                        redirect[i] = node.left_child;
                    } else if (node.left_child == node.right_child) { // same index or same hash
                        node.type = NodeType::Constant;
                        node.value = 0.0;
                    } else if (node.left_child >= 0 && node.right_child >= 0 &&
                               node.left_child < static_cast<int>(node_hashes.size()) &&
                               node.right_child < static_cast<int>(node_hashes.size()) &&
                               node_hashes[node.left_child] == node_hashes[node.right_child]) {
                        node.type = NodeType::Constant;
                        node.value = 0.0;
                    }
                } else if (max_w == w[1]) { // Mul
                    if ((left.type == NodeType::Constant && left.value == 0.0) ||
                        (right.type == NodeType::Constant && right.value == 0.0)) {
                        node.type = NodeType::Constant;
                        node.value = 0.0;
                    } else if (left.type == NodeType::Constant && left.value == 1.0) {
                        redirect[i] = node.right_child;
                    } else if (right.type == NodeType::Constant && right.value == 1.0) {
                        redirect[i] = node.left_child;
                    } else if (node.left_child >= 0 && node.right_child >= 0 &&
                               node.left_child < static_cast<int>(node_hashes.size()) &&
                               node.right_child < static_cast<int>(node_hashes.size()) &&
                               (node.left_child == node.right_child || 
                                node_hashes[node.left_child] == node_hashes[node.right_child])) {
                        // x * x = x^2
                        node.type = NodeType::Unary;
                        node.unary_op = UnaryOp::IntPow;
                        node.p = 2.0;
                        node.right_child = -1;
                    }
                }
            }
        } else if (node.binary_op == BinaryOp::Division) {
            if (left.type == NodeType::Constant && left.value == 0.0) {
                node.type = NodeType::Constant;
                node.value = 0.0;
            } else if (right.type == NodeType::Constant && right.value == 1.0) {
                redirect[i] = node.left_child;
            } else if (node.left_child >= 0 && node.right_child >= 0 &&
                       node.left_child < static_cast<int>(node_hashes.size()) &&
                       node.right_child < static_cast<int>(node_hashes.size()) &&
                       (node.left_child == node.right_child ||
                        node_hashes[node.left_child] == node_hashes[node.right_child])) {
                node.type = NodeType::Constant;
                node.value = 1.0;
            }
        }
    }
}

inline void simplify_ast_advanced(IndividualGraph& graph, double int_tol = 1e-5, double zero_tol = 1e-8) {
    if (graph.nodes.empty()) return;
    if (graph.output_weights.size() > graph.nodes.size()) {
        graph.output_weights.resize(graph.nodes.size());
    }
    if (!is_valid_graph_topology(graph)) {
        throw std::runtime_error("Invalid graph topology before advanced simplification");
    }
    
    std::vector<int> redirect(graph.nodes.size());
    for (size_t i = 0; i < redirect.size(); ++i) redirect[i] = static_cast<int>(i);
    
    // We need structural hashes to check identity of subtrees
    std::vector<uint64_t> node_hashes(graph.nodes.size(), 0);
    
    for (int i = 0; i < static_cast<int>(graph.nodes.size()); ++i) {
        // Pre-compute structural hashes for the un-simplified node
        node_hashes[i] = compute_node_hash(graph, i, node_hashes);
        simplify_node_advanced(graph, i, redirect, node_hashes, int_tol, zero_tol);
    }
    
    // Output weights redirection & snapping
    std::vector<double> new_weights(graph.nodes.size(), 0.0);
    for (size_t i = 0; i < graph.output_weights.size() && i < graph.nodes.size(); ++i) {
        int target = redirect[i];
        if (target >= 0 && target < static_cast<int>(new_weights.size())) {
            new_weights[target] += graph.output_weights[i];
        }
    }
    
    // Merge identical output weight nodes
    // Compute final structural hashes of the simplified graph
    std::vector<uint64_t> final_hashes(graph.nodes.size(), 0);
    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        final_hashes[i] = compute_node_hash(graph, static_cast<int>(i), final_hashes);
    }
    
    std::unordered_map<uint64_t, int> hash_to_first_node;
    std::vector<int> final_redirect(graph.nodes.size());
    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        final_redirect[i] = static_cast<int>(i);
    }
    
    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        if (std::abs(new_weights[i]) > 1e-8) {
            uint64_t h = final_hashes[i];
            auto it = hash_to_first_node.find(h);
            if (it != hash_to_first_node.end()) {
                final_redirect[i] = it->second;
                new_weights[it->second] += new_weights[i];
                new_weights[i] = 0.0;
            } else {
                hash_to_first_node[h] = static_cast<int>(i);
            }
        }
    }
    
    // Propagate final redirection
    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        if (graph.nodes[i].type == NodeType::Unary) {
            if (graph.nodes[i].left_child >= 0 && graph.nodes[i].left_child < static_cast<int>(final_redirect.size())) {
                graph.nodes[i].left_child = final_redirect[graph.nodes[i].left_child];
            }
        } else if (graph.nodes[i].type == NodeType::Binary) {
            if (graph.nodes[i].left_child >= 0 && graph.nodes[i].left_child < static_cast<int>(final_redirect.size())) {
                graph.nodes[i].left_child = final_redirect[graph.nodes[i].left_child];
            }
            if (graph.nodes[i].right_child >= 0 && graph.nodes[i].right_child < static_cast<int>(final_redirect.size())) {
                graph.nodes[i].right_child = final_redirect[graph.nodes[i].right_child];
            }
        }
    }
    
    // Snap output weights
    for (size_t i = 0; i < new_weights.size(); ++i) {
        new_weights[i] = snap_val(new_weights[i], int_tol, zero_tol);
        if (std::abs(new_weights[i]) <= zero_tol) {
            new_weights[i] = 0.0;
        }
    }
    
    // Absorb constant weights to bias
    for (size_t i = 0; i < new_weights.size() && i < graph.nodes.size(); ++i) {
        if (std::abs(new_weights[i]) > 1e-8) {
            if (graph.nodes[i].type == NodeType::Constant) {
                graph.output_bias += new_weights[i] * graph.nodes[i].value;
                new_weights[i] = 0.0;
            }
        }
    }

    // Check for top-level sin^2(A) + cos^2(A) = 1 in output weights
    std::vector<uint64_t> post_redirect_hashes(graph.nodes.size(), 0);
    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        post_redirect_hashes[i] = compute_node_hash(graph, static_cast<int>(i), post_redirect_hashes);
    }

    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        if (std::abs(new_weights[i]) > 1e-8) {
            uint64_t i_arg_hash = 0;
            bool i_cos = false;
            if (is_trig_sq(graph, static_cast<int>(i), post_redirect_hashes, i_arg_hash, i_cos)) {
                // Find a matching trig term
                for (size_t j = i + 1; j < graph.nodes.size(); ++j) {
                    if (std::abs(new_weights[j] - new_weights[i]) < 1e-6) {
                        uint64_t j_arg_hash = 0;
                        bool j_cos = false;
                        if (is_trig_sq(graph, static_cast<int>(j), post_redirect_hashes, j_arg_hash, j_cos)) {
                            if (i_arg_hash == j_arg_hash && i_cos != j_cos) {
                                // Collapse: new_weights[i] * (sin^2 + cos^2) -> new_weights[i] * 1
                                graph.output_bias += new_weights[i];
                                new_weights[i] = 0.0;
                                new_weights[j] = 0.0;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
    
    graph.output_bias = snap_val(graph.output_bias, int_tol, zero_tol);
    graph.output_weights = new_weights;
    
    compact_graph(graph);
    if (!is_valid_graph_topology(graph)) {
        throw std::runtime_error("Invalid graph topology after advanced simplification");
    }
}

// Core simplify: only int_tol/zero_tol/max_passes/n_features affect behavior.
// Extra kwargs (use_nsimplify, identities, trig approx) are accepted by the
// pybind wrapper for API compatibility with the Python sympy path, but are
// intentionally unused here (graph identities always run via simplify_ast_advanced).
inline std::string simplify_formula_cpp(
    const std::string& formula_str,
    double int_tol = 1e-5,
    double zero_tol = 1e-8,
    int max_passes = 6,
    int n_features = 1
) {
    if (formula_str.empty() || formula_str == "0") return formula_str;
    IndividualGraph graph = formula_to_graph(formula_str);
    if (!is_valid_graph_topology(graph)) {
        throw std::runtime_error("Invalid graph generated from formula");
    }

    // Multi-pass simplification
    int pass_count = std::clamp(max_passes, 0, 8);
    for (int p = 0; p < pass_count; ++p) {
        simplify_ast_advanced(graph, int_tol, zero_tol);
    }

    return get_formula_string(graph, n_features);
}

// Phase 6: optional y_weights for BIC term pruning (uniform when empty/null-length).
// holdout_fraction in [0, 0.45]: if >0 and N large enough, reject term drops that
// worsen unweighted holdout MSE beyond relative_slack (noise-aware fidelity guard).
inline std::string reduce_formula_noise_cpp(
    const std::string& formula_str,
    const std::vector<Eigen::ArrayXd>& X,
    const Eigen::ArrayXd& y,
    const Eigen::ArrayXd* y_weights = nullptr,
    double holdout_fraction = 0.0,
    double relative_slack = 0.10
) {
    if (formula_str.empty() || formula_str == "0") return formula_str;

    IndividualGraph graph = formula_to_graph(formula_str);
    if (graph.nodes.empty()) return "0";

    std::vector<int> candidate_nodes;
    for (size_t i = 0; i < graph.output_weights.size() && i < graph.nodes.size(); ++i) {
        if (std::abs(graph.output_weights[i]) > 1e-8) {
            candidate_nodes.push_back(static_cast<int>(i));
        }
    }

    int K = candidate_nodes.size();
    int N = static_cast<int>(y.size());
    if (K <= 1 || K > 20 || N == 0) {
        simplify_ast_advanced(graph);
        return get_formula_string(graph, X.size());
    }

    // Validate optional weights (must match N if provided and non-empty).
    Eigen::ArrayXd w = Eigen::ArrayXd::Ones(N);
    bool use_weights = false;
    if (y_weights != nullptr && y_weights->size() == N) {
        double wsum = 0.0;
        bool ok = true;
        for (int i = 0; i < N; ++i) {
            double wi = (*y_weights)(i);
            if (!std::isfinite(wi) || wi < 0.0) { ok = false; break; }
            w(i) = wi;
            wsum += wi;
        }
        if (ok && wsum > 0.0) {
            // Normalize to mean ~1 so BIC scale stays comparable.
            w *= (static_cast<double>(N) / wsum);
            use_weights = true;
        }
    }

    Eigen::MatrixXd Z(N, K);
    std::vector<Eigen::ArrayXd> cache;
    evaluate_graph(graph, X, N, cache);

    for (int j = 0; j < K; ++j) {
        int idx = candidate_nodes[j];
        if (idx >= 0 && idx < static_cast<int>(cache.size())) {
            Z.col(j) = cache[idx].matrix();
        } else {
            Z.col(j) = Eigen::VectorXd::Zero(N);
        }
    }

    Eigen::VectorXd y_vec = (y - graph.output_bias).matrix();
    std::vector<bool> current_mask(K, true);

    // Optional holdout indices for fidelity guard (last floor(frac*N) rows).
    int n_hold = 0;
    double frac = holdout_fraction;
    if (frac < 0.0) frac = 0.0;
    if (frac > 0.45) frac = 0.45;
    if (frac > 0.0 && N >= 20) {
        n_hold = static_cast<int>(std::floor(frac * N));
        if (n_hold < 4) n_hold = 0;
        if (n_hold >= N - 4) n_hold = 0;
    }
    const int n_fit = N - n_hold;
    const double slack = std::max(0.0, relative_slack);

    auto solve_weighted = [&](const Eigen::MatrixXd& Z_sub, const Eigen::VectorXd& y_sub,
                              const Eigen::ArrayXd& w_sub, Eigen::VectorXd& coefs) {
        const int n = static_cast<int>(Z_sub.rows());
        const int k = static_cast<int>(Z_sub.cols());
        if (!use_weights) {
            coefs = Z_sub.colPivHouseholderQr().solve(y_sub);
            return;
        }
        // Weighted least squares via sqrt(w) row scaling.
        Eigen::MatrixXd Zw(n, k);
        Eigen::VectorXd yw(n);
        for (int i = 0; i < n; ++i) {
            double s = std::sqrt(std::max(w_sub(i), 0.0));
            Zw.row(i) = Z_sub.row(i) * s;
            yw(i) = y_sub(i) * s;
        }
        coefs = Zw.colPivHouseholderQr().solve(yw);
    };

    auto get_bic = [&](const std::vector<bool>& mask, Eigen::VectorXd& coefs) -> double {
        int k = 0;
        for (bool m : mask) if (m) k++;
        if (k == 0) return 1e15;

        // Fit on fit-slice only when holdout is active.
        Eigen::MatrixXd Z_fit(n_fit, k);
        Eigen::VectorXd y_fit(n_fit);
        Eigen::ArrayXd w_fit(n_fit);
        int col_idx = 0;
        for (int j = 0; j < K; ++j) {
            if (mask[j]) {
                Z_fit.col(col_idx) = Z.block(0, j, n_fit, 1);
                ++col_idx;
            }
        }
        y_fit = y_vec.head(n_fit);
        w_fit = w.head(n_fit);

        solve_weighted(Z_fit, y_fit, w_fit, coefs);

        // Weighted MSE on fit rows for BIC.
        double wsum = 0.0;
        double sse = 0.0;
        for (int i = 0; i < n_fit; ++i) {
            double pred = 0.0;
            int c = 0;
            for (int j = 0; j < K; ++j) {
                if (mask[j]) pred += Z(i, j) * coefs(c++);
            }
            double r = pred - y_vec(i);
            double wi = use_weights ? w(i) : 1.0;
            sse += wi * r * r;
            wsum += wi;
        }
        double mse = sse / std::max(wsum, 1e-15);
        if (mse < 1e-15) mse = 1e-15;

        // Effective N for BIC: Kish ESS when weighted, else n_fit.
        double n_eff = static_cast<double>(n_fit);
        if (use_weights) {
            double s1 = 0.0, s2 = 0.0;
            for (int i = 0; i < n_fit; ++i) {
                s1 += w(i);
                s2 += w(i) * w(i);
            }
            if (s2 > 0.0) n_eff = (s1 * s1) / s2;
            n_eff = std::max(n_eff, 2.0);
        }
        return n_eff * std::log(mse) + k * std::log(n_eff);
    };

    auto holdout_mse = [&](const std::vector<bool>& mask, const Eigen::VectorXd& coefs) -> double {
        if (n_hold <= 0) return 0.0;
        double sse = 0.0;
        for (int i = n_fit; i < N; ++i) {
            double pred = 0.0;
            int c = 0;
            for (int j = 0; j < K; ++j) {
                if (mask[j]) pred += Z(i, j) * coefs(c++);
            }
            double r = pred - y_vec(i);
            sse += r * r;
        }
        return sse / static_cast<double>(n_hold);
    };

    Eigen::VectorXd best_coef;
    double best_bic = get_bic(current_mask, best_coef);
    double base_hold = holdout_mse(current_mask, best_coef);


    while (true) {
        int num_active = 0;
        for (bool m : current_mask) if (m) num_active++;
        if (num_active <= 1) break;

        int best_drop_idx = -1;
        double best_drop_bic = best_bic;
        Eigen::VectorXd best_drop_coef;

        for (int j = 0; j < K; ++j) {
            if (current_mask[j]) {
                std::vector<bool> test_mask = current_mask;
                test_mask[j] = false;

                Eigen::VectorXd coefs;
                double bic = get_bic(test_mask, coefs);
                if (bic >= best_drop_bic) continue;

                // Holdout fidelity: do not drop terms that blow up unweighted holdout.
                if (n_hold > 0) {
                    double h = holdout_mse(test_mask, coefs);
                    double allowed = base_hold * (1.0 + slack) + 1e-12;
                    if (std::isfinite(base_hold) && std::isfinite(h) && h > allowed) {
                        continue;
                    }
                }

                best_drop_bic = bic;
                best_drop_idx = j;
                best_drop_coef = coefs;
            }
        }

        if (best_drop_idx != -1) {
            current_mask[best_drop_idx] = false;
            best_bic = best_drop_bic;
            best_coef = best_drop_coef;
            // Refresh base holdout after accepted drop (allow gradual simplification).
            if (n_hold > 0) {
                base_hold = holdout_mse(current_mask, best_coef);
            }
        } else {
            break;
        }
    }

    std::fill(graph.output_weights.begin(), graph.output_weights.end(), 0.0);
    int coef_idx = 0;
    for (int j = 0; j < K; ++j) {
        if (current_mask[j]) {
            int idx = candidate_nodes[j];
            double c = best_coef(coef_idx++);
            if (std::abs(c) > 1e-8) {
                if (idx >= static_cast<int>(graph.output_weights.size())) {
                    graph.output_weights.resize(idx + 1, 0.0);
                }
                graph.output_weights[idx] = c;
            }
        }
    }

    simplify_ast_advanced(graph);
    return get_formula_string(graph, X.size());
}

} // namespace sr
