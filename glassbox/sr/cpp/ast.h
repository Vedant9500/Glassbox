#pragma once

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <functional>
#include <cmath>
#include <cstring>
#include <limits>
#include <cstdint>
#include <Eigen/Dense>

namespace sr {

// Unified output-weight activity threshold (S5-6).
// Eval, formula export, and active_complexity must agree so terms that affect
// predictions are not silently dropped from displayed formulas.
inline constexpr double kOutputWeightActive = 1e-6;
// Compact/simplify may still drop weights far below activity (dead columns).
inline constexpr double kOutputWeightDead = 1e-8;


enum class NodeType {
    Input,
    Constant,
    Unary,
    Binary
};

// Operator enums for discrete ops
enum class UnaryOp {
    Periodic,
    Power,
    IntPow,
    Exp,
    Log,
    Abs
};

enum class BinaryOp {
    Arithmetic,
    Division,
    Aggregation  // Sum, Mean, Max
};

// Represents a node in the computational graph
struct OpNode {
    NodeType type;
    
    // For Input nodes
    int feature_idx = 0;
    
    // For Constant nodes
    double value = 0.0;
    
    // For Unary/Binary nodes
    UnaryOp unary_op;
    BinaryOp binary_op;
    
    // Meta-operation parameters
    double p = 1.0;          // Power / IntPow exponent
    double omega = 1.0;      // Periodic frequency
    double phi = 0.0;        // Periodic phase
    double amplitude = 1.0;  // Periodic amplitude
    double beta = 1.5;       // Arithmetic (1.0 = add, 2.0 = mul)
    double gamma = 1.0;      // Arithmetic sign (for sub/div)
    double tau = 1.0;        // Aggregation temperature
    
    // Child pointers (indices in the layer)
    int left_child = -1;
    int right_child = -1;
};

// Pre-allocated array representing a formula's structure
struct IndividualGraph {
    std::vector<OpNode> nodes;
    std::vector<double> output_weights; // Linear combination of top nodes
    double output_bias = 0.0;
    
    double fitness = 1e9; // Penalized fitness (uses weighted MSE when weights set)
    double raw_mse = 1e9; // Unweighted mathematical MSE (diagnostics / back-compat)
    double weighted_mse = 1e9; // Weighted MSE when y_weights provided; else == raw_mse

    // NSGA-II fields (P5)
    int pareto_rank = 0;           // Non-domination rank (0 = Pareto front)
    double crowding_distance = 0.0; // Crowding distance within the same rank
    int age = 0;                    // AFPO: generations survived (0 = newly created)
    int complexity() const { return static_cast<int>(nodes.size()); } // AST node count as 2nd objective
    int active_complexity() const {
        if (nodes.empty()) return 0;
        std::vector<char> active(nodes.size(), 0);
        for (int i = 0; i < static_cast<int>(nodes.size()) && i < static_cast<int>(output_weights.size()); ++i) {
            if (std::abs(output_weights[i]) <= kOutputWeightActive) continue;
            std::vector<int> stack = {i};
            while (!stack.empty()) {
                int idx = stack.back();
                stack.pop_back();
                if (idx < 0 || idx >= static_cast<int>(nodes.size()) || active[idx]) continue;
                active[idx] = 1;
                const auto& node = nodes[idx];
                if ((node.type == NodeType::Unary || node.type == NodeType::Binary) && node.left_child >= 0) {
                    stack.push_back(node.left_child);
                }
                if (node.type == NodeType::Binary && node.right_child >= 0) {
                    stack.push_back(node.right_child);
                }
            }
        }

        int total = 0;
        for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
            if (!active[i]) continue;
            const auto& node = nodes[i];
            switch (node.type) {
                case NodeType::Input:
                    total += 1;
                    break;
                case NodeType::Constant:
                    total += 1;
                    break;
                case NodeType::Unary:
                    if (node.unary_op == UnaryOp::IntPow) total += 2;
                    else if (node.unary_op == UnaryOp::Abs) total += 2;
                    else if (node.unary_op == UnaryOp::Power) total += 3;
                    else if (node.unary_op == UnaryOp::Periodic) total += 3;
                    else total += 4;  // Exp, Log
                    break;
                case NodeType::Binary:
                    if (node.binary_op == BinaryOp::Arithmetic) total += 2;
                    else if (node.binary_op == BinaryOp::Aggregation) total += 3;
                    else total += 5;
                    break;
            }
        }
        return std::max(1, total);
    }

    // Count distinct active nodes (S5-13): unit-matched with nodes.size().
    int active_node_count() const {
        if (nodes.empty()) return 0;
        std::vector<char> active(nodes.size(), 0);
        for (int i = 0; i < static_cast<int>(nodes.size()) && i < static_cast<int>(output_weights.size()); ++i) {
            if (std::abs(output_weights[i]) <= kOutputWeightActive) continue;
            std::vector<int> stack = {i};
            while (!stack.empty()) {
                int idx = stack.back();
                stack.pop_back();
                if (idx < 0 || idx >= static_cast<int>(nodes.size()) || active[idx]) continue;
                active[idx] = 1;
                const auto& node = nodes[idx];
                if ((node.type == NodeType::Unary || node.type == NodeType::Binary) && node.left_child >= 0) {
                    stack.push_back(node.left_child);
                }
                if (node.type == NodeType::Binary && node.right_child >= 0) {
                    stack.push_back(node.right_child);
                }
            }
        }
        int n = 0;
        for (char a : active) if (a) ++n;
        return n;
    }
};

// ── Structural Hashing ──────────────────────────────────────────────────
// Combines node type, op, quantized parameters, and children hashes into
// a 64-bit fingerprint. Two subtrees with the same hash produce identical
// outputs and can share cached Eigen::ArrayXd results.

inline uint64_t hash_combine(uint64_t seed, uint64_t v) {
    seed ^= v + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
    return seed;
}

// S5-5: default was 2 decimals → SharedCache collisions (ω=1.004 vs 1.006).
// Use 8 decimals for eval cache keys. decimals < 0 ⇒ bit-exact double bits.
// Coarse CSE (if desired) can call quantize(v, 2) explicitly.
inline uint64_t quantize(double v, int decimals = 8) {
    if (decimals < 0) {
        uint64_t u = 0;
        static_assert(sizeof(double) == sizeof(uint64_t), "double must be 64-bit");
        std::memcpy(&u, &v, sizeof(u));
        return u;
    }
    // Quantize to N decimal places for near-match dedup
    double scale = 1.0;
    for (int i = 0; i < decimals; ++i) scale *= 10.0;
    // Clamp huge values so round stays in int64 range.
    double scaled = v * scale;
    if (!std::isfinite(scaled)) {
        uint64_t u = 0;
        std::memcpy(&u, &v, sizeof(u));
        return u;
    }
    if (scaled > static_cast<double>(std::numeric_limits<int64_t>::max())) scaled = static_cast<double>(std::numeric_limits<int64_t>::max());
    if (scaled < static_cast<double>(std::numeric_limits<int64_t>::min())) scaled = static_cast<double>(std::numeric_limits<int64_t>::min());
    int64_t q = static_cast<int64_t>(std::round(scaled));
    uint64_t u;
    std::memcpy(&u, &q, sizeof(u));
    return u;
}

// Compute structural hash for node at index `idx` in the graph.
// `node_hashes` must be pre-allocated to graph.nodes.size() and filled
// bottom-up (lower indices first, which is the natural DAG order).
inline uint64_t compute_node_hash(const IndividualGraph& graph, int idx,
                                   std::vector<uint64_t>& node_hashes) {
    const auto& node = graph.nodes[idx];
    uint64_t h = static_cast<uint64_t>(node.type);

    switch (node.type) {
        case NodeType::Input:
            h = hash_combine(h, static_cast<uint64_t>(node.feature_idx));
            break;
        case NodeType::Constant:
            h = hash_combine(h, quantize(node.value));
            break;
        case NodeType::Unary:
            h = hash_combine(h, static_cast<uint64_t>(node.unary_op));
            h = hash_combine(h, quantize(node.p));
            h = hash_combine(h, quantize(node.omega));
            h = hash_combine(h, quantize(node.phi));
            h = hash_combine(h, quantize(node.amplitude));
            if (node.left_child >= 0 && node.left_child < idx) {
                h = hash_combine(h, node_hashes[node.left_child]);
            }
            break;
        case NodeType::Binary:
            h = hash_combine(h, static_cast<uint64_t>(node.binary_op));
            h = hash_combine(h, quantize(node.beta));
            h = hash_combine(h, quantize(node.gamma));
            h = hash_combine(h, quantize(node.tau));
            if (node.left_child >= 0 && node.left_child < idx) {
                h = hash_combine(h, node_hashes[node.left_child]);
            }
            if (node.right_child >= 0 && node.right_child < idx) {
                h = hash_combine(h, node_hashes[node.right_child]);
            }
            break;
    }
    return h;
}

// Cache type: maps subtree hash → evaluated ArrayXd
using SubtreeCache = std::unordered_map<uint64_t, Eigen::ArrayXd>;

} // namespace sr
