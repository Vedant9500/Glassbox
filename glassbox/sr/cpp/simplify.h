#pragma once

#include "ast.h"
#include "eval.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace sr {

// Forward declare for recursive simplifier
inline void simplify_node(IndividualGraph& graph, int node_idx);

// compact_graph is defined in ast.h (shared with formula_to_graph post-compile).

inline void simplify_ast(IndividualGraph& graph) {
    if (graph.nodes.empty()) return;

    // Bottom-up simplification
    for (int i = 0; i < static_cast<int>(graph.nodes.size()); ++i) {
        simplify_node(graph, i);
    }

    // Output layer simplification: Fold constants into bias
    for (size_t i = 0; i < graph.output_weights.size() && i < graph.nodes.size(); ++i) {
        if (std::abs(graph.output_weights[i]) > kOutputWeightDead) {
            if (graph.nodes[i].type == NodeType::Constant) {
                graph.output_bias += graph.output_weights[i] * graph.nodes[i].value;
                graph.output_weights[i] = 0.0;
            }
        }
    }

    compact_graph(graph);
}

inline void simplify_node(IndividualGraph& graph, int node_idx) {
    if (node_idx < 0 || node_idx >= static_cast<int>(graph.nodes.size())) return;
    OpNode& node = graph.nodes[static_cast<size_t>(node_idx)];
    const int n = static_cast<int>(graph.nodes.size());

    if (node.type == NodeType::Unary) {
        if (node.left_child < 0 || node.left_child >= n) return;
        const OpNode& child = graph.nodes[static_cast<size_t>(node.left_child)];

        // Constant folding
        if (child.type == NodeType::Constant) {
            double v = child.value;
            double res = 0.0;
            switch (node.unary_op) {
                case UnaryOp::Periodic:
                    res = node.amplitude * std::sin(node.omega * v + node.phi);
                    break;
                case UnaryOp::Power: {
                    // §3.1 canonical parity tol 1e-9 (matches eval.h).
                    double abs_v = std::abs(v) + 1e-10;
                    double sign_v = (v >= 0) ? 1.0 : -1.0;
                    double p_round = std::round(node.p);
                    bool is_even = (std::abs(node.p - p_round) < 1e-9) &&
                                   (static_cast<long long>(p_round) % 2 == 0);
                    double abs_pow = std::pow(abs_v, node.p);
                    res = is_even ? abs_pow : sign_v * abs_pow;
                    break;
                }
                case UnaryOp::IntPow:
                    res = std::pow(
                        v, std::clamp(static_cast<int>(std::round(node.p)), 2, 6));
                    break;
                case UnaryOp::Exp:
                    // §3.7/§3.112: exact path clamps arg ±500, graph clamps
                    // output ±1e6. Fold must not saturate early at ±50: use
                    // ±500 arg clamp then output clamp, matching exact+graph.
                    res = std::exp(std::clamp(node.omega * v + node.phi, -500.0, 500.0));
                    res = std::clamp(res, -1e6, 1e6);
                    break;
                case UnaryOp::Log:
                    res = std::log(std::abs(v) + 1e-6);
                    break;
                case UnaryOp::Abs:
                    res = std::abs(v);
                    break;
            }
            // H-02: never store Inf/NaN constants from folds (eval clamps live path).
            if (!std::isfinite(res)) res = 0.0;
            res = std::clamp(res, -1e8, 1e8);
            node.type = NodeType::Constant;
            node.value = res;
        }

    } else if (node.type == NodeType::Binary) {
        if (node.left_child < 0 || node.left_child >= n ||
            node.right_child < 0 || node.right_child >= n) {
            return;
        }
        const OpNode& left = graph.nodes[static_cast<size_t>(node.left_child)];
        const OpNode& right = graph.nodes[static_cast<size_t>(node.right_child)];

        // Constant folding for binary ops
        if (left.type == NodeType::Constant && right.type == NodeType::Constant) {
            double l = left.value;
            double r = right.value;
            double res = 0.0;

            if (node.binary_op == BinaryOp::Arithmetic) {
                // S5-12 / P3-017: share eval soft-arithmetic weights (max-logit stable).
                // §3.334/§3.336: at/above the shared near-discrete threshold the
                // argmax pick is exact; below it, fold the true weighted mixture
                // (w0*(l+r)+w1*(l*r)+w2*softdiv+w3*(l-r), live-eval form incl.
                // its ±1e6 output clamp) instead of silently switching to argmax.
                auto w = arithmetic_soft_weights(node);
                double m = std::max({w[0], w[1], w[2], w[3]});
                if (m >= kArithmeticNearDiscrete) {
                    if (m == w[0]) res = l + r;
                    else if (m == w[3]) res = l - r;
                    else if (m == w[1]) res = l * r;
                    else res = l / std::sqrt(1.0 + r * r);
                } else {
                    res = w[0] * (l + r) + w[1] * (l * r)
                        + w[2] * (l / std::sqrt(1.0 + r * r)) + w[3] * (l - r);
                    res = std::clamp(res, -1e6, 1e6);
                }
            } else if (node.binary_op == BinaryOp::Division) {
                res = (l / (std::abs(r) + 1e-6)) * ((r >= 0) ? 1.0 : -1.0);
            } else if (node.binary_op == BinaryOp::Aggregation) {
                // Soft-mean / soft-max fold at tau (S5-12).
                double tau = stabilized_tau(node.tau);
                double max_val = std::max(l, r);
                double ex = std::exp((l - max_val) / tau);
                double ey = std::exp((r - max_val) / tau);
                double den = ex + ey;
                res = (den > 0.0) ? (l * ex + r * ey) / den : 0.5 * (l + r);
            }

            // H-02: sanitize folded binary constants.
            if (!std::isfinite(res)) res = 0.0;
            res = std::clamp(res, -1e8, 1e8);
            node.type = NodeType::Constant;
            node.value = res;
        }
    }
}

} // namespace sr
