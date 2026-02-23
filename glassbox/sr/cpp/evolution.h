#pragma once

#include "ast.h"
#include "eval.h"
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>

namespace sr {

// Configuration for evolution
struct EvolutionConfig {
    int pop_size = 50;
    int elite_size = 10;
    int generations = 1000;
    
    double mutation_rate_structural = 0.3;
    double mutation_rate_parametric = 0.5;
    
    // Bounds
    double p_min = -2.0, p_max = 3.0;
    double omega_min = 0.1, omega_max = 8.0;
    
    bool use_early_stop = true;
    double early_stop_mse = 1e-6;
};

class EvolutionEngine {
public:
    EvolutionEngine(const EvolutionConfig& config, 
                    const std::vector<Eigen::ArrayXd>& X, 
                    const Eigen::ArrayXd& y)
        : config_(config), X_(X), y_(y), rng_(std::random_device{}()) {}

    // Main run loop
    void run() {
        initialize_population();
        
        for (int gen = 0; gen < config_.generations; ++gen) {
            evaluate_population();
            
            // Sort by fitness (lowest MSE first)
            std::sort(population_.begin(), population_.end(), 
                      [](const IndividualGraph& a, const IndividualGraph& b) {
                          return a.fitness < b.fitness;
                      });
            
            // Track best
            if (population_[0].fitness < best_overall_.fitness) {
                best_overall_ = population_[0];
            }
            
            if (config_.use_early_stop && best_overall_.fitness < config_.early_stop_mse) {
                break; // Exact algebraic match found
            }
            
            // Create next generation
            std::vector<IndividualGraph> next_gen;
            next_gen.reserve(config_.pop_size);
            
            // Elitism ensures top survivors pass verbatim
            for (int i = 0; i < config_.elite_size; ++i) {
                next_gen.push_back(population_[i]);
            }
            
            // Fill remainder with mutated offspring
            std::uniform_int_distribution<int> parent_dist(0, config_.elite_size - 1);
            while (next_gen.size() < config_.pop_size) {
                int parent_idx = parent_dist(rng_);
                next_gen.push_back(mutate(population_[parent_idx]));
            }
            
            population_ = std::move(next_gen);
        }
    }
    
    IndividualGraph get_best() const {
        return best_overall_;
    }

private:
    EvolutionConfig config_;
    std::vector<Eigen::ArrayXd> X_;
    Eigen::ArrayXd y_;
    
    std::vector<IndividualGraph> population_;
    IndividualGraph best_overall_;
    
    std::mt19937 rng_;
    
    void initialize_population() {
        population_.resize(config_.pop_size);
        int n_inputs = static_cast<int>(X_.size());
        
        for (auto& ind : population_) {
            // Random DAG generator
            std::uniform_int_distribution<int> num_nodes_dist(3, 15);
            int num_nodes = num_nodes_dist(rng_);
            ind.nodes.resize(num_nodes);
            
            std::uniform_real_distribution<double> runif(0.0, 1.0);
            std::normal_distribution<double> rnorm(0.0, 1.0);
            
            for (int i = 0; i < num_nodes; ++i) {
                auto& node = ind.nodes[i];
                node.p = 1.0 + rnorm(rng_)*0.5;
                node.omega = 1.0 + rnorm(rng_);
                node.phi = rnorm(rng_);
                node.amplitude = 1.0 + rnorm(rng_)*0.5;
                node.beta = 1.5 + rnorm(rng_)*0.5;
                node.gamma = 1.0 + rnorm(rng_)*0.5;
                node.tau = 1.0;
                
                if (i == 0 || runif(rng_) < 0.2) {
                    if (runif(rng_) < 0.5 && n_inputs > 0) {
                        node.type = NodeType::Input;
                        std::uniform_int_distribution<int> feat_dist(0, n_inputs - 1);
                        node.feature_idx = feat_dist(rng_);
                    } else {
                        node.type = NodeType::Constant;
                        node.value = rnorm(rng_);
                    }
                } else {
                    if (runif(rng_) < 0.6 || i < 2) {
                        node.type = NodeType::Unary;
                        std::uniform_int_distribution<int> op_dist(0, 3);
                        node.unary_op = static_cast<UnaryOp>(op_dist(rng_));
                        std::uniform_int_distribution<int> child_dist(0, i - 1);
                        node.left_child = child_dist(rng_);
                    } else {
                        node.type = NodeType::Binary;
                        std::uniform_int_distribution<int> op_dist(0, 1);
                        node.binary_op = static_cast<BinaryOp>(op_dist(rng_));
                        std::uniform_int_distribution<int> child_dist(0, i - 1);
                        node.left_child = child_dist(rng_);
                        node.right_child = child_dist(rng_);
                    }
                }
            }
            
            ind.output_weights.resize(num_nodes);
            for (int i = 0; i < num_nodes; ++i) {
                ind.output_weights[i] = rnorm(rng_) * 0.1;
            }
            ind.output_bias = rnorm(rng_) * 0.1;
        }
    }
    
    void evaluate_population() {
        int samples = static_cast<int>(y_.size());
        for (auto& ind : population_) {
            evaluate_fitness(ind, X_, y_, samples);
        }
    }
    
    IndividualGraph mutate(IndividualGraph parent) {
        // Shallow/deep copy happens automatically in C++
        IndividualGraph child = parent;
        
        std::uniform_real_distribution<double> runif(0.0, 1.0);
        
        // Parametric mutation (continuous params)
        for (auto& node : child.nodes) {
            if (runif(rng_) < config_.mutation_rate_parametric) {
                std::normal_distribution<double> rnorm(0.0, 0.5);
                node.p += rnorm(rng_);
                node.omega += rnorm(rng_);
                node.phi += rnorm(rng_);
            }
            
            // Structural mutation (discrete choices)
            if (runif(rng_) < config_.mutation_rate_structural) {
                if (node.type == NodeType::Unary) {
                    std::uniform_int_distribution<int> op_dist(0, 3);
                    node.unary_op = static_cast<UnaryOp>(op_dist(rng_));
                }
            }
        }
        
        return child;
    }
};

} // namespace sr
