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
    
    // Output layer computation (weighted sum of nodes)
    Eigen::ArrayXd final_output = Eigen::ArrayXd::Constant(num_samples, graph.output_bias);
    for (size_t i = 0; i < graph.output_weights.size() && i < graph.nodes.size(); ++i) {
        if (std::abs(graph.output_weights[i]) > 1e-6) {
            final_output += graph.output_weights[i] * cache[i];
        }
    }
    
    return final_output;
}

// Compute MSE fitness
inline double evaluate_fitness(IndividualGraph& graph, const std::vector<Eigen::ArrayXd>& X, const Eigen::ArrayXd& y, int num_samples) {
    Eigen::ArrayXd pred = evaluate_graph(graph, X, num_samples);
    double mse = (pred - y).square().mean();
    graph.fitness = mse;
    return mse;
}

} // namespace sr
