#pragma once

#define _USE_MATH_DEFINES
#include "ast.h"
#include "simplify.h"
#include "eval.h"
#include "formula_parser.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <iostream>
#include <stdexcept>
#include <Eigen/Dense>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_E
#define M_E 2.71828182845904523536
#endif

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
            
            double phi_norm = std::fmod(child.phi, 2.0 * M_PI);
            if (phi_norm < 0.0) phi_norm += 2.0 * M_PI;
            
            // cos has phi = PI/2.0
            is_cos = std::abs(phi_norm - M_PI / 2.0) < 1e-4 || std::abs(phi_norm - 1.5 * M_PI) < 1e-4;
            bool is_sin = std::abs(phi_norm) < 1e-4 || std::abs(phi_norm - M_PI) < 1e-4 || std::abs(phi_norm - 2.0 * M_PI) < 1e-4;
            
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
                case UnaryOp::Exp: res = std::exp(node.omega * v + node.phi); break;
                case UnaryOp::Log: res = std::log(std::abs(v) + 1e-6); break;
            }
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
                // (x^a)^b = x^(a*b)
                node.p = snap_val(node.p * child.p, int_tol, zero_tol);
                node.left_child = child.left_child;
            }
        } else if (node.unary_op == UnaryOp::Exp) {
            node.omega = snap_val(node.omega, int_tol, zero_tol);
            node.phi = snap_val(node.phi, int_tol, zero_tol);
            if (node.omega == 0.0) {
                node.type = NodeType::Constant;
                node.value = snap_val(std::exp(node.phi), int_tol, zero_tol);
            } else if (child.type == NodeType::Unary && child.unary_op == UnaryOp::Log && std::abs(node.omega - 1.0) < 1e-4 && std::abs(node.phi) < 1e-4) {
                // exp(log(|y|)) = |y|
                node.unary_op = UnaryOp::Power;
                node.p = 1.0;
                node.left_child = child.left_child;
            }
        } else if (node.unary_op == UnaryOp::Log) {
            if (child.type == NodeType::Constant) {
                node.type = NodeType::Constant;
                node.value = snap_val(std::log(std::abs(child.value) + 1e-6), int_tol, zero_tol);
            } else if (child.type == NodeType::Unary && child.unary_op == UnaryOp::Exp && std::abs(child.omega - 1.0) < 1e-4 && std::abs(child.phi) < 1e-4) {
                // log(exp(y)) = y
                redirect[i] = child.left_child;
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
            
            node.type = NodeType::Constant;
            node.value = snap_val(res, int_tol, zero_tol);
            return;
        }
        
        // Algebraic identities
        if (node.binary_op == BinaryOp::Arithmetic) {
            auto w = arithmetic_soft_weights(node);
            constexpr double kNearDiscrete = 0.95;
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

inline std::string simplify_formula_cpp(
    const std::string& formula_str,
    double int_tol = 1e-5,
    double zero_tol = 1e-8,
    int max_passes = 6,
    bool use_nsimplify = true,
    bool use_identities = true,
    bool approximate_trig = false,
    double dominant_trig_ratio = 0.9,
    double small_term_ratio = 0.08,
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

inline std::string reduce_formula_noise_cpp(
    const std::string& formula_str,
    const std::vector<Eigen::ArrayXd>& X,
    const Eigen::ArrayXd& y
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
    int N = y.size();
    if (K <= 1 || K > 20 || N == 0) {
        simplify_ast_advanced(graph);
        return get_formula_string(graph, X.size());
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

    auto get_bic = [&](const std::vector<bool>& mask, Eigen::VectorXd& coefs) -> double {
        int k = 0;
        for (bool m : mask) if (m) k++;
        if (k == 0) return 1e15;

        Eigen::MatrixXd Z_sub(N, k);
        int col_idx = 0;
        for (int j = 0; j < K; ++j) {
            if (mask[j]) {
                Z_sub.col(col_idx++) = Z.col(j);
            }
        }

        coefs = Z_sub.colPivHouseholderQr().solve(y_vec);
        double mse = (Z_sub * coefs - y_vec).squaredNorm() / N;
        if (mse < 1e-15) mse = 1e-15;

        return N * std::log(mse) + k * std::log(N);
    };

    Eigen::VectorXd best_coef;
    double best_bic = get_bic(current_mask, best_coef);

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
                if (bic < best_drop_bic) {
                    best_drop_bic = bic;
                    best_drop_idx = j;
                    best_drop_coef = coefs;
                }
            }
        }

        if (best_drop_idx != -1) {
            current_mask[best_drop_idx] = false;
            best_bic = best_drop_bic;
            best_coef = best_drop_coef;
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
