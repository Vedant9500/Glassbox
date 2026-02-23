#pragma once

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
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
    Exp,
    Log
};

enum class BinaryOp {
    Arithmetic,
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
    double p = 1.0;          // Power
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
    
    double fitness = 1e9; // MSE
};

} // namespace sr
