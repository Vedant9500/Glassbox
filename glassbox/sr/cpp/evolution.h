#pragma once

#include "ast.h"
#include "eval.h"
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>

#include <omp.h>

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

    // Pruning and rounding
    double prune_threshold = 0.05;
    double round_penalty_weight = 0.01;

    // Search Dynamics
    double explorer_fraction = 0.2; // 20% of population are explorers
    double explorer_mutation_multiplier = 3.0; 
};

class EvolutionEngine {
public:
    EvolutionEngine(const EvolutionConfig& config, 
                    const std::vector<Eigen::ArrayXd>& X, 
                    const Eigen::ArrayXd& y,
                    const std::vector<double>& seed_omegas = {})
        : config_(config), X_(X), y_(y), seed_omegas_(seed_omegas), rng_(std::random_device{}()) {}

    // Main run loop
    void run() {
        initialize_population();
        
        // Initial Refinement
        for (auto& ind : population_) {
            refine_constants(ind);
        }
        
        double current_structural_mutation_rate = config_.mutation_rate_structural;
        double best_mse_history = 1e9;
        int plateau_counter = 0;

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
            
            // Dynamic mutation decay based on plateauing
            if (best_overall_.fitness >= best_mse_history * 0.99) {
                plateau_counter++;
            } else {
                plateau_counter = 0;
                best_mse_history = best_overall_.fitness;
            }

            if (plateau_counter > 50) {
                // If stuck for 50 gens, lower mutation rate to exploit
                current_structural_mutation_rate = std::max(0.05, current_structural_mutation_rate * 0.9);
                plateau_counter = 0; // Reset
            }

            if (config_.use_early_stop && best_overall_.raw_mse < config_.early_stop_mse && best_overall_.nodes.size() <= 8) {
                break; // Exact algebraic match found that is simple
            }
            
            // Create next generation
            std::vector<IndividualGraph> next_gen;
            next_gen.reserve(config_.pop_size);
            
            // Elitism ensures top survivors pass verbatim
            for (int i = 0; i < config_.elite_size; ++i) {
                next_gen.push_back(population_[i]);
            }
            
            int num_explorers = static_cast<int>(config_.pop_size * config_.explorer_fraction);
            int main_pop_target = config_.pop_size - num_explorers;

            // Fill remainder of main population with mutated offspring
            std::uniform_int_distribution<int> parent_dist(0, config_.elite_size - 1);
            while (next_gen.size() < main_pop_target) {
                int parent_idx = parent_dist(rng_);
                IndividualGraph child = mutate_lamarckian(population_[parent_idx], current_structural_mutation_rate);
                if (gen % 5 == 0 || child.fitness < best_overall_.fitness * 1.5) {
                    refine_constants(child); // Gradient refinement on constants
                } else {
                    // Fast re-evaluation
                    evaluate_fitness_with_penalty(child, X_, y_, y_.size());
                }
                next_gen.push_back(std::move(child));
            }

            // Fill explorer population
            while (next_gen.size() < config_.pop_size) {
                 int parent_idx = parent_dist(rng_); // Use elite parents, but mutate more aggressively
                 double explorer_rate = std::min(1.0, current_structural_mutation_rate * config_.explorer_mutation_multiplier);
                 IndividualGraph explorer = mutate_lamarckian(population_[parent_idx], explorer_rate);
                 if (gen % 10 == 0) {
                     refine_constants(explorer); 
                 } else {
                     evaluate_fitness_with_penalty(explorer, X_, y_, y_.size());
                 }
                 next_gen.push_back(std::move(explorer));
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
    std::vector<double> seed_omegas_;
    
    std::vector<IndividualGraph> population_;
    IndividualGraph best_overall_;
    
    std::mt19937 rng_;
    
    void initialize_population() {
        population_.resize(config_.pop_size);
        int n_inputs = static_cast<int>(X_.size());
        
        for (auto& ind : population_) {
            // Random DAG generator
            std::uniform_int_distribution<int> num_nodes_dist(5, 20); // allow slightly deeper
            int num_nodes = num_nodes_dist(rng_);
            ind.nodes.resize(num_nodes);
            
            std::uniform_real_distribution<double> runif(0.0, 1.0);
            std::normal_distribution<double> rnorm(0.0, 1.0);
            
            for (int i = 0; i < num_nodes; ++i) {
                auto& node = ind.nodes[i];
                node.p = 1.0 + rnorm(rng_)*0.5;
                node.omega = 1.0 + rnorm(rng_);

                // 🌟 Inject seeded omegas if available 🌟
                if (!seed_omegas_.empty() && runif(rng_) < 0.6) { // Boosted omega seeding probability
                    std::uniform_int_distribution<int> seed_dist(0, seed_omegas_.size() - 1);
                    node.omega = seed_omegas_[seed_dist(rng_)];
                }

                node.phi = rnorm(rng_);
                node.amplitude = 1.0 + rnorm(rng_)*0.5;
                node.beta = 1.5 + rnorm(rng_)*0.5;
                node.gamma = 1.0 + rnorm(rng_)*0.5;
                node.tau = 1.0;
                
                if (i == 0 || runif(rng_) < 0.1) { // Lower terminal proability for deeper trees
                    if (runif(rng_) < 0.8 && n_inputs > 0) { // Mostly inputs rather than constants
                        node.type = NodeType::Input;
                        std::uniform_int_distribution<int> feat_dist(0, n_inputs - 1);
                        node.feature_idx = feat_dist(rng_);
                    } else {
                        node.type = NodeType::Constant;
                        node.value = rnorm(rng_);
                    }
                } else {
                    if (runif(rng_) < 0.5 || i < 2) {
                        node.type = NodeType::Unary;
                        // 0=Periodic, 1=Power, 2=Exp, 3=Log
                        // Uniform distribution
                        double op_choice = runif(rng_);
                        if (op_choice < 0.25) node.unary_op = UnaryOp::Periodic; 
                        else if (op_choice < 0.5) node.unary_op = UnaryOp::Power; 
                        else if (op_choice < 0.75) node.unary_op = UnaryOp::Exp;
                        else node.unary_op = UnaryOp::Log;
                        
                        std::uniform_int_distribution<int> child_dist(0, i - 1);
                        node.left_child = child_dist(rng_);
                    } else {
                        node.type = NodeType::Binary;
                        double op_choice = runif(rng_);
                        if (op_choice < 0.5) node.binary_op = BinaryOp::Arithmetic; 
                        else node.binary_op = BinaryOp::Aggregation;
                        
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
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < static_cast<int>(population_.size()); ++i) {
            evaluate_fitness_with_penalty(population_[i], X_, y_, samples);
        }
    }
    
    // Evaluate fitness including complexity and soft rounding penalty
    double evaluate_fitness_with_penalty(IndividualGraph& graph, const std::vector<Eigen::ArrayXd>& X, const Eigen::ArrayXd& y, int num_samples) {
        Eigen::ArrayXd pred = evaluate_graph_simple(graph, X, num_samples);
        double mse = (pred - y).square().mean();
        graph.raw_mse = mse;
        
        double penalty = 0.0;
        if (config_.round_penalty_weight > 0) {
            for (const auto& node : graph.nodes) {
                if (node.type == NodeType::Unary) {
                    double frac_p = node.p - std::floor(node.p);
                    double dist_p = std::min(frac_p, 1.0 - frac_p);
                    penalty += dist_p * dist_p;
                    
                    double frac_o = node.omega - std::floor(node.omega);
                    double dist_o = std::min(frac_o, 1.0 - frac_o);
                    penalty += dist_o * dist_o;
                }
            }
        }
        
        // Parsimony pressure: penalize graph size to prevent overfitting
        double complexity_penalty = 1e-4 * graph.nodes.size();
        
        graph.fitness = mse + complexity_penalty + config_.round_penalty_weight * penalty / std::max(1.0, (double)graph.nodes.size());
        return graph.fitness;
    }
    
    IndividualGraph mutate_lamarckian(IndividualGraph parent, double structural_rate) {
        IndividualGraph child = parent;
        
        std::uniform_real_distribution<double> runif(0.0, 1.0);
        std::normal_distribution<double> rnorm(0.0, 0.5); 
        
        int n_inputs = static_cast<int>(X_.size());
        
        for (int i = 0; i < child.nodes.size(); ++i) {
            auto& node = child.nodes[i];
            
            if (runif(rng_) < structural_rate) {
                // Structural mutation - change node type or connections
                if (i == 0 || runif(rng_) < 0.2) {
                    if (runif(rng_) < 0.5 && n_inputs > 0) {
                        node.type = NodeType::Input;
                        std::uniform_int_distribution<int> feat_dist(0, n_inputs - 1);
                        node.feature_idx = feat_dist(rng_);
                    } else {
                        node.type = NodeType::Constant;
                        node.value += rnorm(rng_); 
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
            } else {
                // Continuous Parameter Mutation
                if (runif(rng_) < 0.3) {
                    node.p += rnorm(rng_);
                    node.omega += rnorm(rng_);
                    node.phi += rnorm(rng_);
                    node.amplitude += rnorm(rng_) * 0.5;
                    if (node.type == NodeType::Constant) node.value += rnorm(rng_);
                    
                    node.omega = std::clamp(node.omega, config_.omega_min, config_.omega_max);
                    node.p = std::clamp(node.p, config_.p_min, config_.p_max);
                }
            }
        }
        
        return child;
    }
    
    // Fast analytical solver for linear weights
    void refine_constants(IndividualGraph& ind) {
        int n_samples = static_cast<int>(y_.size());
        
        std::vector<Eigen::ArrayXd> cache;
        evaluate_graph(ind, X_, n_samples, cache);
        
        if (!ind.nodes.empty()) {
            int num_features = static_cast<int>(ind.nodes.size());
            // Build Design Matrix A: [nodes | 1 (bias)]
            Eigen::MatrixXd A(n_samples, num_features + 1);
            for (int i = 0; i < num_features; ++i) {
                if (!cache[i].isFinite().all()) {
                    cache[i] = Eigen::ArrayXd::Zero(n_samples); 
                }
                A.col(i) = cache[i].matrix();
            }
            A.col(num_features).setOnes();
            
            Eigen::VectorXd b = y_.matrix();
            Eigen::VectorXd w;
            try {
                w = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
            } catch (...) {
                w = Eigen::VectorXd::Zero(num_features + 1);
            }
            
            // Post-SVD Hard Thresholding (Coefficient Pruning)
            double max_w = 0.0;
            for (int i = 0; i < num_features; ++i) {
                if (std::isfinite(w(i))) {
                    max_w = std::max(max_w, std::abs(w(i)));
                } else {
                    w(i) = 0.0;
                }
            }
            
            for (int i = 0; i < num_features; ++i) {
                if (std::abs(w(i)) < config_.prune_threshold * max_w || std::abs(w(i)) < 1e-4) {
                    ind.output_weights[i] = 0.0;
                } else {
                    ind.output_weights[i] = w(i);
                }
            }
            
            if (std::isfinite(w(num_features)) && std::abs(w(num_features)) >= 1e-4) {
                ind.output_bias = w(num_features);
            } else {
                ind.output_bias = 0.0;
            }
            
            evaluate_fitness_with_penalty(ind, X_, y_, n_samples);
        }
    }
};

} // namespace sr
