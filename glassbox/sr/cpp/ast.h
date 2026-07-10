#pragma once

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <functional>
#include <cmath>
#include <Eigen/Dense>

namespace sr {

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
    Log
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
    
    double fitness = 1e9; // Penalized fitness
    double raw_mse = 1e9; // Actual mathematical MSE

    // NSGA-II fields (P5)
    int pareto_rank = 0;           // Non-domination rank (0 = Pareto front)
    double crowding_distance = 0.0; // Crowding distance within the same rank
    int age = 0;                    // AFPO: generations survived (0 = newly created)
    int complexity() const { return static_cast<int>(nodes.size()); } // AST node count as 2nd objective
    int active_complexity() const {
        if (nodes.empty()) return 0;
        std::vector<char> active(nodes.size(), 0);
        for (int i = 0; i < static_cast<int>(nodes.size()) && i < static_cast<int>(output_weights.size()); ++i) {
            if (std::abs(output_weights[i]) <= 1e-4) continue;
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
                    else if (node.unary_op == UnaryOp::Power) total += 3;
                    else if (node.unary_op == UnaryOp::Periodic) total += 3;
                    else total += 4;
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
};

// ── Structural Hashing ──────────────────────────────────────────────────
// Combines node type, op, quantized parameters, and children hashes into
// a 64-bit fingerprint. Two subtrees with the same hash produce identical
// outputs and can share cached Eigen::ArrayXd results.

inline uint64_t hash_combine(uint64_t seed, uint64_t v) {
    seed ^= v + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
    return seed;
}

inline uint64_t quantize(double v, int decimals = 2) {
    // Quantize to N decimal places for near-match dedup
    double scale = 1.0;
    for (int i = 0; i < decimals; ++i) scale *= 10.0;
    int64_t q = static_cast<int64_t>(std::round(v * scale));
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
