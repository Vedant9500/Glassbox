#pragma once

#include "ast.h"
#include <cmath>
#include <vector>
#include <iostream>

namespace sr {

// Forward declare for recursive simplifier
inline void simplify_node(IndividualGraph& graph, int node_idx);

// Remove dead nodes and compact the graph
inline void compact_graph(IndividualGraph& graph) {
    if (graph.nodes.empty()) return;
    
    std::vector<int> new_indices(graph.nodes.size(), -1);
    std::vector<OpNode> new_nodes;
    std::vector<double> new_weights;
    
    // Mark used nodes from output weights
    std::vector<bool> used(graph.nodes.size(), false);
    for (size_t i = 0; i < graph.output_weights.size() && i < graph.nodes.size(); ++i) {
        if (std::abs(graph.output_weights[i]) > 1e-8) {
            used[i] = true;
        }
    }
    
    // Propagate used flags downwards
    for (int i = static_cast<int>(graph.nodes.size()) - 1; i >= 0; --i) {
        if (used[i]) {
            if (graph.nodes[i].type == NodeType::Unary) {
                if (graph.nodes[i].left_child >= 0) used[graph.nodes[i].left_child] = true;
            } else if (graph.nodes[i].type == NodeType::Binary) {
                if (graph.nodes[i].left_child >= 0) used[graph.nodes[i].left_child] = true;
                if (graph.nodes[i].right_child >= 0) used[graph.nodes[i].right_child] = true;
            }
        }
    }
    
    // Build compacted nodes
    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        if (used[i]) {
            new_indices[i] = static_cast<int>(new_nodes.size());
            OpNode n = graph.nodes[i];
            if (n.type == NodeType::Unary) {
                if (n.left_child >= 0) n.left_child = new_indices[n.left_child];
            } else if (n.type == NodeType::Binary) {
                if (n.left_child >= 0) n.left_child = new_indices[n.left_child];
                if (n.right_child >= 0) n.right_child = new_indices[n.right_child];
            }
            new_nodes.push_back(n);
            
            // Map weights
            if (i < graph.output_weights.size() && std::abs(graph.output_weights[i]) > 1e-8) {
                // Ensure new_weights is large enough
                while (new_weights.size() <= new_indices[i]) {
                    new_weights.push_back(0.0);
                }
                new_weights[new_indices[i]] = graph.output_weights[i];
            }
        }
    }
    
    graph.nodes = new_nodes;
    graph.output_weights = new_weights;
}

inline void simplify_ast(IndividualGraph& graph) {
    if (graph.nodes.empty()) return;
    
    // Bottom-up simplification
    for (int i = 0; i < static_cast<int>(graph.nodes.size()); ++i) {
        simplify_node(graph, i);
    }
    
    // Output layer simplification: Fold constants into bias
    for (size_t i = 0; i < graph.output_weights.size() && i < graph.nodes.size(); ++i) {
        if (std::abs(graph.output_weights[i]) > 1e-8) {
            if (graph.nodes[i].type == NodeType::Constant) {
                graph.output_bias += graph.output_weights[i] * graph.nodes[i].value;
                graph.output_weights[i] = 0.0;
            }
        }
    }
    
    compact_graph(graph);
}

inline void simplify_node(IndividualGraph& graph, int node_idx) {
    OpNode& node = graph.nodes[node_idx];
    
    if (node.type == NodeType::Unary) {
        OpNode& child = graph.nodes[node.left_child];
        
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
            node.value = res;
        }
        
    } else if (node.type == NodeType::Binary) {
        OpNode& left = graph.nodes[node.left_child];
        OpNode& right = graph.nodes[node.right_child];
        
        // Constant folding for binary ops
        if (left.type == NodeType::Constant && right.type == NodeType::Constant) {
            double l = left.value;
            double r = right.value;
            double res = 0.0;
            
            if (node.binary_op == BinaryOp::Arithmetic) {
                // Determine dominant op from softmax weights
                double t = 5.0; // Assume default temp
                double d_add = (node.beta - 1.0) * (node.beta - 1.0) + (node.gamma - 1.0) * (node.gamma - 1.0);
                double d_mul = (node.beta - 2.0) * (node.beta - 2.0) + (node.gamma - 1.0) * (node.gamma - 1.0);
                double d_div = (node.beta - 2.0) * (node.beta - 2.0) + (node.gamma + 1.0) * (node.gamma + 1.0);
                double d_sub = (node.beta - 1.0) * (node.beta - 1.0) + (node.gamma + 1.0) * (node.gamma + 1.0);
                
                double w_add = std::exp(-d_add * t);
                double w_mul = std::exp(-d_mul * t);
                double w_div = std::exp(-d_div * t);
                double w_sub = std::exp(-d_sub * t);
                
                double m = std::max({w_add, w_mul, w_div, w_sub});
                if (m == w_add) res = l + r;
                else if (m == w_sub) res = l - r;
                else if (m == w_mul) res = l * r;
                else res = l / std::sqrt(1.0 + r*r);
            } else if (node.binary_op == BinaryOp::Division) {
                res = (l / (std::abs(r) + 1e-6)) * ((r >= 0) ? 1.0 : -1.0);
            }
            
            node.type = NodeType::Constant;
            node.value = res;
        }
    }
}

} // namespace sr