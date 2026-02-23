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
        
        // Initial Refinement
        for (auto& ind : population_) {
            refine_constants(ind);
        }
        
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
                IndividualGraph child = mutate_lamarckian(population_[parent_idx]);
                refine_constants(child); // Gradient refinement on constants
                next_gen.push_back(std::move(child));
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
    
    IndividualGraph mutate_lamarckian(IndividualGraph parent) {
        // Deep copy
        IndividualGraph child = parent;
        
        std::uniform_real_distribution<double> runif(0.0, 1.0);
        
        // ONLY mutate discrete structure choices (operation types, connections)
        // Continuous parameters (p, omega, phi, etc) are inherited identically!
        for (auto& node : child.nodes) {
            if (runif(rng_) < config_.mutation_rate_structural) {
                if (node.type == NodeType::Unary) {
                    std::uniform_int_distribution<int> op_dist(0, 3);
                    node.unary_op = static_cast<UnaryOp>(op_dist(rng_));
                } else if (node.type == NodeType::Binary) {
                    std::uniform_int_distribution<int> op_dist(0, 1);
                    node.binary_op = static_cast<BinaryOp>(op_dist(rng_));
                }
            }
        }
        
        return child;
    }
    
    // Custom native C++ gradient descent (Adam-like) for continuous parameters
    void refine_constants(IndividualGraph& ind) {
        int steps = 15; // Fast local steps
        double lr = 0.05;
        double epsilon = 1e-4; // Step size for finite differences
        
        int n_samples = static_cast<int>(y_.size());
        evaluate_fitness(ind, X_, y_, n_samples);
        double best_fitness = ind.fitness;
        
        std::vector<double> m_omega(ind.nodes.size(), 0.0);
        std::vector<double> v_omega(ind.nodes.size(), 0.0);
        std::vector<double> m_p(ind.nodes.size(), 0.0);
        std::vector<double> v_p(ind.nodes.size(), 0.0);
        
        double beta1 = 0.9;
        double beta2 = 0.999;
        double eps = 1e-8;
        
        IndividualGraph best_ind = ind;
        IndividualGraph working_ind = ind;

        for (int step = 1; step <= steps; ++step) {
            bool improved = false;
            
            // Loop over all nodes
            for (size_t i = 0; i < working_ind.nodes.size(); ++i) {
                auto& node = working_ind.nodes[i];
                if (node.type != NodeType::Unary) continue; 
                
                // 1. Gradient w.r.t Omega
                double orig_omega = node.omega;
                node.omega = orig_omega + epsilon;
                evaluate_fitness(working_ind, X_, y_, n_samples);
                double f_plus = working_ind.fitness;
                
                node.omega = orig_omega - epsilon;
                evaluate_fitness(working_ind, X_, y_, n_samples);
                double f_minus = working_ind.fitness;
                
                double grad_omega = (f_plus - f_minus) / (2.0 * epsilon);
                node.omega = orig_omega; // Restore
                
                // 2. Gradient w.r.t P (power operations)
                double orig_p = node.p;
                node.p = orig_p + epsilon;
                evaluate_fitness(working_ind, X_, y_, n_samples);
                double fp_plus = working_ind.fitness;
                
                node.p = orig_p - epsilon;
                evaluate_fitness(working_ind, X_, y_, n_samples);
                double fp_minus = working_ind.fitness;
                
                double grad_p = (fp_plus - fp_minus) / (2.0 * epsilon);
                node.p = orig_p; // Restore
                
                // Adam update
                m_omega[i] = beta1 * m_omega[i] + (1.0 - beta1) * grad_omega;
                v_omega[i] = beta2 * v_omega[i] + (1.0 - beta2) * grad_omega * grad_omega;
                double m_hat_omega = m_omega[i] / (1.0 - std::pow(beta1, step));
                double v_hat_omega = v_omega[i] / (1.0 - std::pow(beta2, step));
                
                m_p[i] = beta1 * m_p[i] + (1.0 - beta1) * grad_p;
                v_p[i] = beta2 * v_p[i] + (1.0 - beta2) * grad_p * grad_p;
                double m_hat_p = m_p[i] / (1.0 - std::pow(beta1, step));
                double v_hat_p = v_p[i] / (1.0 - std::pow(beta2, step));
                
                node.omega -= lr * m_hat_omega / (std::sqrt(v_hat_omega) + eps);
                node.p -= lr * m_hat_p / (std::sqrt(v_hat_p) + eps);
                
                // Bound clamping
                node.omega = std::clamp(node.omega, config_.omega_min, config_.omega_max);
                node.p = std::clamp(node.p, config_.p_min, config_.p_max);
            }
            
            evaluate_fitness(working_ind, X_, y_, n_samples);
            if (working_ind.fitness < best_fitness) {
                best_fitness = working_ind.fitness;
                best_ind = working_ind;
                improved = true;
            }
        }
        
        // Final Analytical Least Squares for Output Weights
        // Now that internal constants (p, omega) are tuned, we solve exactly for the linear output combination
        std::vector<Eigen::ArrayXd> cache;
        evaluate_graph(best_ind, X_, n_samples, cache);
        
        if (!best_ind.nodes.empty()) {
            int num_features = static_cast<int>(best_ind.nodes.size());
            // Build Design Matrix A: [nodes | 1 (bias)]
            Eigen::MatrixXd A(n_samples, num_features + 1);
            for (int i = 0; i < num_features; ++i) {
                A.col(i) = cache[i].matrix();
            }
            A.col(num_features).setOnes();
            
            Eigen::VectorXd b = y_.matrix();
            
            // Solve A * w = b using SVD for numerical stability
            // bdcSvd is the recommended solver in Eigen for general least squares
            Eigen::VectorXd w = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
            
            for (int i = 0; i < num_features; ++i) {
                best_ind.output_weights[i] = w(i);
            }
            best_ind.output_bias = w(num_features);
            
            // Re-evaluate to lock in the new least-squares fitness
            evaluate_fitness(best_ind, X_, y_, n_samples);
        }
        
        ind = best_ind;
    }
};

} // namespace sr
