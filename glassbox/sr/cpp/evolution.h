#pragma once

#define _USE_MATH_DEFINES
#include "ast.h"
#include "eval.h"
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <cmath>

#include <omp.h>

// MSVC fallbacks for M_PI and M_E
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_E
#define M_E 2.71828182845904523536
#endif

namespace sr {

// Configuration for evolution
struct EvolutionConfig {
    int pop_size = 50;
    int elite_size = 10;
    int generations = 1000;
    
    double mutation_rate_structural = 0.3;
    double mutation_rate_parametric = 0.5;
    double crossover_rate = 0.3; // Fraction of offspring produced via crossover
    
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

    // Classifier priors: probability weights for [Periodic, Power, Exp, Log]
    // Empty = uniform sampling. Non-empty = sample proportionally.
    std::vector<double> op_priors; // e.g. {0.9, 0.03, 0.03, 0.04}

    // P5: NSGA-II multi-objective
    bool use_nsga2 = false;

    // P6: Island Model
    int num_islands = 1;          // 1 = single population (default)
    int migration_interval = 25;  // Exchange elites every N generations
    int migration_size = 2;       // Number of elites migrated per exchange

    // P7: Dimensional Analysis
    std::vector<std::vector<double>> input_units;  // Per-feature unit exponents
    std::vector<double> output_units;              // Target variable units
    double dim_penalty_weight = 0.1;
};

class EvolutionEngine {
public:
    EvolutionEngine(const EvolutionConfig& config, 
                    const std::vector<Eigen::ArrayXd>& X, 
                    const Eigen::ArrayXd& y,
                    const std::vector<double>& seed_omegas = {})
        : config_(config), X_(X), y_(y), seed_omegas_(seed_omegas), rng_(std::random_device{}()) {
    sanitize_config();

        // Normalize op_priors if provided
        if (!config_.op_priors.empty()) {
            double sum = 0.0;
            for (double p : config_.op_priors) sum += p;
            if (sum > 0) {
                for (double& p : config_.op_priors) p /= sum;
            }
            // Build CDF for weighted sampling
            op_cdf_.resize(config_.op_priors.size());
            op_cdf_[0] = config_.op_priors[0];
            for (size_t i = 1; i < config_.op_priors.size(); ++i) {
                op_cdf_[i] = op_cdf_[i-1] + config_.op_priors[i];
            }
        }
    }

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
            
            // P5: NSGA-II selection or standard sort
            if (config_.use_nsga2) {
                // Create offspring pool
                std::vector<IndividualGraph> combined;
                combined.reserve(population_.size() * 2);
                for (auto& ind : population_) {
                    ind.age++;  // AFPO: existing population ages
                    combined.push_back(ind);
                }
                // Generate offspring
                std::uniform_int_distribution<int> p_dist(0, std::max(0, (int)population_.size() - 1));
                std::uniform_real_distribution<double> co(0.0, 1.0);
                for (int i = 0; i < config_.pop_size; ++i) {
                    IndividualGraph child;
                    double roll = co(rng_);
                    if (roll < 0.15) {
                        child = macro_mutate(population_[p_dist(rng_)]);
                    } else if (config_.elite_size >= 2 && roll < 0.15 + config_.crossover_rate) {
                        int p1 = p_dist(rng_), p2 = p_dist(rng_);
                        while (p2 == p1) p2 = p_dist(rng_);
                        child = crossover(population_[p1], population_[p2]);
                        child = mutate_lamarckian(child, current_structural_mutation_rate * 0.3);
                    } else {
                        child = mutate_lamarckian(population_[p_dist(rng_)], current_structural_mutation_rate);
                    }
                    child.age = 0;  // AFPO: new children start young
                    if (gen % 5 == 0) refine_constants(child);
                    else evaluate_fitness_with_penalty(child, X_, y_, y_.size());
                    combined.push_back(std::move(child));
                }
                // NSGA-II selection from combined pool
                population_ = nsga2_select(combined, config_.pop_size);
                // Track best (min MSE from front)
                for (auto& ind : population_) {
                    if (ind.fitness < best_overall_.fitness) best_overall_ = ind;
                }
            } else {
                // Standard single-objective sort
                std::sort(population_.begin(), population_.end(), 
                          [](const IndividualGraph& a, const IndividualGraph& b) {
                              return a.fitness < b.fitness;
                          });
                
                // Track best
                if (population_[0].fitness < best_overall_.fitness) {
                    best_overall_ = population_[0];
                }
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
            
            // Elitism ensures top survivors pass verbatim (age incremented)
            for (int i = 0; i < config_.elite_size; ++i) {
                IndividualGraph elite = population_[i];
                elite.age++;  // AFPO: survivors age
                next_gen.push_back(std::move(elite));
            }
            
            int num_explorers = static_cast<int>(config_.pop_size * config_.explorer_fraction);
            int main_pop_target = config_.pop_size - num_explorers;

            // Fill remainder of main population with crossover + mutated offspring
            std::uniform_int_distribution<int> parent_dist(0, config_.elite_size - 1);
            std::uniform_real_distribution<double> coin(0.0, 1.0);
            while (next_gen.size() < main_pop_target) {
                IndividualGraph child;
                double roll = coin(rng_);

                if (roll < 0.15) {
                    // --- Macro-Mutation (15%) ---
                    int parent_idx = parent_dist(rng_);
                    child = macro_mutate(population_[parent_idx]);
                } else if (config_.elite_size >= 2 && roll < 0.15 + config_.crossover_rate) {
                    // --- Subtree Crossover ---
                    int p1 = parent_dist(rng_);
                    int p2 = parent_dist(rng_);
                    while (p2 == p1) p2 = parent_dist(rng_); // ensure distinct parents
                    child = crossover(population_[p1], population_[p2]);
                    // Light mutation on crossover children for diversity
                    child = mutate_lamarckian(child, current_structural_mutation_rate * 0.3);
                } else {
                    // --- Mutation-only ---
                    int parent_idx = parent_dist(rng_);
                    child = mutate_lamarckian(population_[parent_idx], current_structural_mutation_rate);
                }

                child.age = 0;  // AFPO: new children start young

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
                 IndividualGraph explorer;
                 if (coin(rng_) < 0.2) {
                     explorer = macro_mutate(population_[parent_idx]);
                 } else {
                     explorer = mutate_lamarckian(population_[parent_idx], explorer_rate);
                 }
                 explorer.age = 0;  // AFPO: explorers start young
                 if (gen % 10 == 0) {
                     refine_constants(explorer); 
                 } else {
                     evaluate_fitness_with_penalty(explorer, X_, y_, y_.size());
                 }
                 next_gen.push_back(std::move(explorer));
            }
            
            population_ = std::move(next_gen);
            
            // Periodic inner-param refinement on top elite only (every 20 gens)
            // This is where Adam refines omega/p/phi — crucial for sin(3x) etc.
            if (gen % 20 == 19) {
                for (int i = 0; i < std::min(5, config_.elite_size); ++i) {
                    refine_inner_params(population_[i]);
                }
            }
        }
        
        // Post-evolution cleanup: deduplicate + prune the best graph
        cleanup_graph(best_overall_);
    }
    
    IndividualGraph get_best() const {
        return best_overall_;
    }

    // P5: Return entire Pareto front (rank-0 individuals)
    // Re-runs non_dominated_sort on the current population to get clean ranks.
    std::vector<IndividualGraph> get_pareto_front() {
        if (population_.empty()) {
            return {best_overall_};
        }
        // Fresh sort on current population
        non_dominated_sort(population_);

        std::vector<IndividualGraph> front;
        for (const auto& ind : population_) {
            if (ind.pareto_rank == 0) front.push_back(ind);
        }
        if (front.empty()) front.push_back(best_overall_);

        // Sort by MSE
        std::sort(front.begin(), front.end(),
                  [](const IndividualGraph& a, const IndividualGraph& b) {
                      return a.raw_mse < b.raw_mse;
                  });

        // Deduplicate: remove solutions with identical (mse, complexity)
        auto last = std::unique(front.begin(), front.end(),
                                [](const IndividualGraph& a, const IndividualGraph& b) {
                                    return std::abs(a.raw_mse - b.raw_mse) < 1e-12 &&
                                           a.complexity() == b.complexity();
                                });
        front.erase(last, front.end());

        // 2-objective domination filter (MSE + complexity only).
        // Internal NSGA-II uses 3 objectives (including age) for selection,
        // but the reported Pareto front should be clean on user-visible axes.
        std::vector<IndividualGraph> clean_front;
        for (size_t i = 0; i < front.size(); ++i) {
            bool dominated = false;
            for (size_t j = 0; j < front.size(); ++j) {
                if (i == j) continue;
                bool j_leq = (front[j].raw_mse <= front[i].raw_mse) &&
                              (front[j].complexity() <= front[i].complexity());
                bool j_lt  = (front[j].raw_mse < front[i].raw_mse) ||
                              (front[j].complexity() < front[i].complexity());
                if (j_leq && j_lt) { dominated = true; break; }
            }
            if (!dominated) clean_front.push_back(front[i]);
        }
        if (clean_front.empty()) clean_front.push_back(front[0]);

        return clean_front;
    }

    // P6: Island Model run
    void run_islands() {
        if (config_.num_islands <= 1) { run(); return; }

        int island_size = config_.pop_size / config_.num_islands;
        if (island_size < 4) { run(); return; } // Too small for islands

        // Create per-island engines with split configs
        std::vector<EvolutionEngine> islands;
        islands.reserve(config_.num_islands);
        EvolutionConfig island_cfg = config_;
        island_cfg.pop_size = island_size;
        island_cfg.elite_size = std::max(2, config_.elite_size / config_.num_islands);
        island_cfg.num_islands = 1; // Each island is single-population

        for (int i = 0; i < config_.num_islands; ++i) {
            islands.emplace_back(island_cfg, X_, y_, seed_omegas_);
        }

        // Initialize all islands
        for (auto& island : islands) {
            island.initialize_population();
            for (auto& ind : island.population_) {
                island.refine_constants(ind);
            }
        }

        // Run generations with periodic migration
        for (int gen = 0; gen < config_.generations; ++gen) {
            // Evolve each island one generation
            for (auto& island : islands) {
                island.evolve_one_generation(gen);
            }

            // Migration: ring topology (island i → island i+1)
            if (gen > 0 && gen % config_.migration_interval == 0) {
                for (int i = 0; i < config_.num_islands; ++i) {
                    int next = (i + 1) % config_.num_islands;
                    auto& src = islands[i].population_;
                    auto& dst = islands[next].population_;

                    // Sort source by fitness to get top elites
                    std::sort(src.begin(), src.end(),
                              [](const IndividualGraph& a, const IndividualGraph& b) {
                                  return a.fitness < b.fitness;
                              });

                    // Replace worst in destination with source elites
                    std::sort(dst.begin(), dst.end(),
                              [](const IndividualGraph& a, const IndividualGraph& b) {
                                  return a.fitness < b.fitness;
                              });

                    int n_migrate = std::min(config_.migration_size, (int)src.size() / 2);
                    for (int m = 0; m < n_migrate; ++m) {
                        dst[dst.size() - 1 - m] = src[m];
                    }
                }
            }

            // Check early stop across all islands
            bool should_stop = false;
            for (auto& island : islands) {
                auto best = island.get_best();
                if (best.raw_mse < config_.early_stop_mse && best.nodes.size() <= 8) {
                    should_stop = true;
                }
                if (best.fitness < best_overall_.fitness) {
                    best_overall_ = best;
                }
            }
            if (config_.use_early_stop && should_stop) break;
        }

        // Collect the best overall across all islands and run cleanup
        for (auto& island : islands) {
            auto best = island.get_best();
            if (best.fitness < best_overall_.fitness) {
                best_overall_ = best;
            }
        }

        // Merge all island populations for Pareto front (if NSGA-II)
        population_.clear();
        for (auto& island : islands) {
            for (auto& ind : island.population_) {
                population_.push_back(std::move(ind));
            }
        }

        cleanup_graph(best_overall_);
    }

private:
    EvolutionConfig config_;
    std::vector<Eigen::ArrayXd> X_;
    Eigen::ArrayXd y_;
    std::vector<double> seed_omegas_;
    
    std::vector<IndividualGraph> population_;
    IndividualGraph best_overall_;
    SubtreeCache gen_cache_; // Per-generation subtree cache
    std::vector<double> op_cdf_; // CDF for prior-weighted op sampling
    
    std::mt19937 rng_;

    void sanitize_config() {
        config_.pop_size = std::max(1, config_.pop_size);
        config_.elite_size = std::max(1, std::min(config_.elite_size, config_.pop_size));
        config_.num_islands = std::max(1, config_.num_islands);
        config_.migration_size = std::max(1, config_.migration_size);
    }
    
    // Sample a UnaryOp using classifier priors (if available) or uniform
    UnaryOp sample_unary_op() {
        std::uniform_real_distribution<double> u(0.0, 1.0);
        if (!op_cdf_.empty()) {
            double r = u(rng_);
            for (size_t i = 0; i < op_cdf_.size(); ++i) {
                if (r <= op_cdf_[i]) return static_cast<UnaryOp>(i);
            }
            return UnaryOp::Log; // Fallback to last
        }
        // Uniform: 0=Periodic, 1=Power, 2=Exp, 3=Log
        double op_choice = u(rng_);
        if (op_choice < 0.25) return UnaryOp::Periodic;
        if (op_choice < 0.5) return UnaryOp::Power;
        if (op_choice < 0.75) return UnaryOp::Exp;
        return UnaryOp::Log;
    }
    
    void initialize_population() {
        population_.resize(config_.pop_size);
        int n_inputs = static_cast<int>(X_.size());
        
        for (auto& ind : population_) {
            // Random DAG generator
            std::uniform_int_distribution<int> num_nodes_dist(3, 8); // compact graphs
            int num_nodes = num_nodes_dist(rng_);
            ind.nodes.resize(num_nodes);
            
            std::uniform_real_distribution<double> runif(0.0, 1.0);
            std::normal_distribution<double> rnorm(0.0, 1.0);
            
            for (int i = 0; i < num_nodes; ++i) {
                auto& node = ind.nodes[i];
                node.p = 1.0 + rnorm(rng_)*0.5;
                node.omega = 1.0 + rnorm(rng_);

                // Inject seeded omegas if available
                if (!seed_omegas_.empty() && runif(rng_) < 0.6) {
                    std::uniform_int_distribution<int> seed_dist(0, seed_omegas_.size() - 1);
                    node.omega = seed_omegas_[seed_dist(rng_)];
                }

                node.phi = rnorm(rng_);
                node.amplitude = 1.0;  // Fixed — SVD handles scaling via output_weights
                node.beta = 1.5 + rnorm(rng_)*0.5;
                node.gamma = 1.0 + rnorm(rng_)*0.5;
                node.tau = 1.0;
                
                // ── FIX: Only node 0 is Input. All others are Unary/Binary. ──
                // This prevents multiple collinear 'x' columns in the SVD.
                if (i < n_inputs) {
                    node.type = NodeType::Input;
                    node.feature_idx = i;
                } else {
                    if (runif(rng_) < 0.6 || i < n_inputs + 1) {
                        node.type = NodeType::Unary;
                        node.unary_op = sample_unary_op();
                        std::uniform_int_distribution<int> child_dist(0, i - 1);
                        node.left_child = child_dist(rng_);
                    } else {
                        node.type = NodeType::Binary;
                        double op_choice = runif(rng_);
                        if (op_choice < 0.5) node.binary_op = BinaryOp::Arithmetic; 
                        else node.binary_op = BinaryOp::Aggregation;
                        
                        std::uniform_int_distribution<int> child_dist(0, i - 1);
                        node.left_child = child_dist(rng_);
                        // Ensure different children for Binary to avoid (x+x)/2 == x
                        if (i >= 2) {
                            node.right_child = child_dist(rng_);
                            int tries = 0;
                            while (node.right_child == node.left_child && tries < 5) {
                                node.right_child = child_dist(rng_);
                                tries++;
                            }
                        } else {
                            node.right_child = node.left_child;
                        }
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
        gen_cache_.clear(); // Fresh cache per generation
        // Note: not using cache in parallel eval to avoid thread-safety issues.
        // Each thread evaluates independently; cache is used in serial paths.
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
        
        // Parsimony pressure: penalize graph size to prevent bloat
        // Count active nodes (non-zero output weight) for stronger penalty
        int active_nodes = 0;
        for (size_t i = 0; i < graph.nodes.size() && i < graph.output_weights.size(); ++i) {
            if (std::abs(graph.output_weights[i]) > 1e-4) active_nodes++;
        }
        double complexity_penalty = 5e-3 * active_nodes + 1e-4 * graph.nodes.size();
        
        graph.fitness = mse + complexity_penalty + config_.round_penalty_weight * penalty / std::max(1.0, (double)graph.nodes.size());

        // P7: Dimensional analysis penalty (only active when input_units provided)
        if (!config_.input_units.empty()) {
            graph.fitness += config_.dim_penalty_weight * dimensional_penalty(graph);
        }

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
                        node.unary_op = sample_unary_op();
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
                    // amplitude is fixed at 1.0 — SVD handles scaling
                    if (node.type == NodeType::Constant) node.value += rnorm(rng_);
                    
                    node.omega = std::clamp(node.omega, config_.omega_min, config_.omega_max);
                    node.p = std::clamp(node.p, config_.p_min, config_.p_max);
                }
            }
        }
        
        return child;
    }

    // ── Macro-Mutations ────────────────────────────────────────────────────
    // Structural mutations that preserve building blocks:
    //   - Wrap:     f(x) → sin(f(x)) or exp(f(x)) or |f(x)|^p
    //   - Multiply: f(x), g(x) → f(x) * g(x)
    //   - Nest:     f(x), g(x) → f(g(x))
    IndividualGraph macro_mutate(const IndividualGraph& parent) {
        IndividualGraph child = parent;
        std::uniform_real_distribution<double> runif(0.0, 1.0);
        
        int n = static_cast<int>(child.nodes.size());
        if (n < 2) return mutate_lamarckian(child, 0.3); // Too small for macro
        
        double roll = runif(rng_);
        
        if (roll < 0.5) {
            // ── Wrap Mutation ──
            // Pick a random non-input node and wrap it with a new unary op
            std::uniform_int_distribution<int> node_dist(1, n - 1);
            int target = node_dist(rng_);
            
            // Create new unary node that takes 'target' as input
            OpNode wrap_node;
            wrap_node.type = NodeType::Unary;
            wrap_node.unary_op = sample_unary_op();
            wrap_node.left_child = target;
            wrap_node.p = 1.0;
            wrap_node.omega = 1.0;
            wrap_node.phi = 0.0;
            wrap_node.amplitude = 1.0;
            
            // If wrapping with Power, use interesting exponents
            if (wrap_node.unary_op == UnaryOp::Power) {
                double powers[] = {0.5, 2.0, 3.0, -1.0, 1.5};
                std::uniform_int_distribution<int> pow_dist(0, 4);
                wrap_node.p = powers[pow_dist(rng_)];
            }
            // If wrapping with Periodic, seed useful frequencies
            if (wrap_node.unary_op == UnaryOp::Periodic) {
                double freqs[] = {1.0, 2.0, 3.0, 0.5};
                std::uniform_int_distribution<int> freq_dist(0, 3);
                wrap_node.omega = freqs[freq_dist(rng_)];
            }
            
            child.nodes.push_back(wrap_node);
            child.output_weights.push_back(0.5); // Small initial weight
            
        } else if (roll < 0.8) {
            // ── Multiply Mutation ──
            // Pick two existing nodes and create f(x) * g(x)
            std::uniform_int_distribution<int> node_dist(0, n - 1);
            int left = node_dist(rng_);
            int right = node_dist(rng_);
            while (right == left && n > 1) right = node_dist(rng_);
            
            OpNode mul_node;
            mul_node.type = NodeType::Binary;
            mul_node.binary_op = BinaryOp::Arithmetic;
            mul_node.beta = 2.0;  // 2.0 = multiply mode
            mul_node.gamma = 1.0;
            mul_node.left_child = left;
            mul_node.right_child = right;
            
            child.nodes.push_back(mul_node);
            child.output_weights.push_back(1.0);
            
        } else {
            // ── Nest Mutation ──
            // Pick a unary node f and change its input to another node g
            // Creating f(g(x)) from f(old_input) and g(x)
            std::vector<int> unary_indices;
            for (int i = 0; i < n; ++i) {
                if (child.nodes[i].type == NodeType::Unary) {
                    unary_indices.push_back(i);
                }
            }
            
            if (unary_indices.empty()) {
                // No unary nodes, fall back to wrap
                return macro_mutate(parent); // Retry (will likely hit wrap/multiply)
            }
            
            std::uniform_int_distribution<int> uni_dist(0, static_cast<int>(unary_indices.size()) - 1);
            int f_idx = unary_indices[uni_dist(rng_)];
            
            // Pick a different node to be the new input
            std::uniform_int_distribution<int> node_dist(0, f_idx > 0 ? f_idx - 1 : 0);
            int g_idx = node_dist(rng_);
            
            // Rewire: f's input becomes g (creating f(g(x)) composition)
            child.nodes[f_idx].left_child = g_idx;
        }
        
        return child;
    }
    
    // ── Subtree Crossover ──────────────────────────────────────────────────
    // Swaps a contiguous subtree between two parents to produce one child.
    // A "subtree" here is all nodes reachable from a selected crossover point.
    IndividualGraph crossover(const IndividualGraph& parent_a, const IndividualGraph& parent_b) {
        IndividualGraph child = parent_a; // Start from parent A

        if (parent_a.nodes.size() < 3 || parent_b.nodes.size() < 3) {
            return child; // Too small for meaningful crossover
        }

        // Pick a non-terminal crossover point in each parent (skip node 0 to preserve an input root)
        std::uniform_int_distribution<int> dist_a(1, static_cast<int>(parent_a.nodes.size()) - 1);
        std::uniform_int_distribution<int> dist_b(1, static_cast<int>(parent_b.nodes.size()) - 1);

        int xo_a = dist_a(rng_); // Point in parent A to replace
        int xo_b = dist_b(rng_); // Point in parent B to donate

        // Collect the subtree rooted at xo_b in parent B
        // (all nodes whose index >= xo_b that are reachable from xo_b)
        std::vector<int> subtree_b = collect_subtree(parent_b, xo_b);
        if (subtree_b.empty()) {
            return child; // Degenerate, just return parent A
        }

        // Collect subtree rooted at xo_a in parent A (to remove)
        std::vector<int> subtree_a = collect_subtree(parent_a, xo_a);

        // Strategy: replace node at xo_a with the donated subtree from B.
        // To keep things simple and avoid complex re-indexing, we do a
        // "graft" approach: copy the subtree_b nodes into the child,
        // adjusting child pointers by offset.

        // Remove subtree_a nodes from child (replace with donated subtree_b)
        // Build new node list: [nodes before xo_a] + [donated subtree] + [nodes after subtree_a]
        std::vector<OpNode> new_nodes;
        new_nodes.reserve(child.nodes.size());

        // 1. Copy nodes before the crossover point
        for (int i = 0; i < xo_a; ++i) {
            new_nodes.push_back(child.nodes[i]);
        }

        // 2. Insert donated subtree from parent B, adjusting child pointers
        int offset = xo_a - xo_b; // Index shift: donated node i maps to i + offset
        for (int idx : subtree_b) {
            OpNode donated = parent_b.nodes[idx];
            // Adjust child pointers by the offset
            if (donated.left_child >= 0) {
                int new_left = donated.left_child + offset;
                donated.left_child = std::clamp(new_left, 0, static_cast<int>(xo_a + subtree_b.size()) - 1);
            }
            if (donated.right_child >= 0) {
                int new_right = donated.right_child + offset;
                donated.right_child = std::clamp(new_right, 0, static_cast<int>(xo_a + subtree_b.size()) - 1);
            }
            new_nodes.push_back(donated);
        }

        // 3. Copy remaining nodes from parent A after the subtree_a region
        int end_of_subtree_a = subtree_a.empty() ? xo_a + 1 : subtree_a.back() + 1;
        int size_diff = static_cast<int>(subtree_b.size()) - (end_of_subtree_a - xo_a);
        for (int i = end_of_subtree_a; i < static_cast<int>(parent_a.nodes.size()); ++i) {
            OpNode n = parent_a.nodes[i];
            // Adjust child pointers for the size change
            if (n.left_child >= xo_a) {
                n.left_child = std::clamp(n.left_child + size_diff, 0, static_cast<int>(new_nodes.size()) + static_cast<int>(parent_a.nodes.size()) - end_of_subtree_a - 1);
            }
            if (n.right_child >= xo_a) {
                n.right_child = std::clamp(n.right_child + size_diff, 0, static_cast<int>(new_nodes.size()) + static_cast<int>(parent_a.nodes.size()) - end_of_subtree_a - 1);
            }
            new_nodes.push_back(n);
        }

        // Safety: cap graph size to prevent bloat
        if (new_nodes.size() > 30) {
            new_nodes.resize(30);
        }

        // Fix any dangling child pointers
        int total = static_cast<int>(new_nodes.size());
        for (int i = 0; i < total; ++i) {
            auto& n = new_nodes[i];
            if (n.left_child >= total) n.left_child = std::max(0, i - 1);
            if (n.right_child >= total) n.right_child = std::max(0, i - 1);
            if (n.left_child >= i && (n.type == NodeType::Unary || n.type == NodeType::Binary)) n.left_child = std::max(0, i - 1);
            if (n.right_child >= i && n.type == NodeType::Binary) n.right_child = std::max(0, i - 1);
        }

        child.nodes = std::move(new_nodes);

        // Resize output weights to match new node count
        std::normal_distribution<double> rnorm(0.0, 0.1);
        child.output_weights.resize(child.nodes.size());
        for (size_t i = parent_a.output_weights.size(); i < child.output_weights.size(); ++i) {
            child.output_weights[i] = rnorm(rng_);
        }

        child.fitness = 1e9; // Mark for re-evaluation
        child.raw_mse = 1e9;
        return child;
    }

    // Collect all node indices reachable from `root` via child pointers (DFS)
    std::vector<int> collect_subtree(const IndividualGraph& graph, int root) {
        std::vector<int> result;
        if (root < 0 || root >= static_cast<int>(graph.nodes.size())) return result;

        std::vector<bool> visited(graph.nodes.size(), false);
        std::vector<int> stack = {root};

        while (!stack.empty()) {
            int idx = stack.back();
            stack.pop_back();
            if (idx < 0 || idx >= static_cast<int>(graph.nodes.size()) || visited[idx]) continue;
            visited[idx] = true;
            result.push_back(idx);

            const auto& n = graph.nodes[idx];
            // Children are at lower indices (DAG invariant), but we collect them anyway
            if (n.type == NodeType::Unary || n.type == NodeType::Binary) {
                if (n.left_child >= 0 && n.left_child < static_cast<int>(graph.nodes.size())) {
                    stack.push_back(n.left_child);
                }
            }
            if (n.type == NodeType::Binary) {
                if (n.right_child >= 0 && n.right_child < static_cast<int>(graph.nodes.size())) {
                    stack.push_back(n.right_child);
                }
            }
        }

        std::sort(result.begin(), result.end());
        return result;
    }
    
    // ── Ridge Regression solver for output weights ──────────────────────
    // Replaces bare SVD with (A^T*A + λI)^{-1} A^T b to prevent
    // multicollinearity from producing massive cancelling coefficients.
    // Returns true if solve succeeded.
    bool solve_output_weights(IndividualGraph& ind, const std::vector<Eigen::ArrayXd>& cache) {
        int n_samples = static_cast<int>(y_.size());
        int num_features = static_cast<int>(ind.nodes.size());
        if (num_features == 0) return false;
        
        // Build Design Matrix A: [nodes | 1 (bias)]
        Eigen::MatrixXd A(n_samples, num_features + 1);
        for (int i = 0; i < num_features; ++i) {
            if (cache[i].size() == n_samples && cache[i].isFinite().all()) {
                A.col(i) = cache[i].matrix();
            } else {
                A.col(i).setZero();
            }
        }
        A.col(num_features).setOnes();
        
        Eigen::VectorXd b = y_.matrix();
        Eigen::VectorXd w;
        
        try {
            // Ridge regression: w = (A^T A + λI)^{-1} A^T b
            double lambda = 0.01;
            Eigen::MatrixXd AtA = A.transpose() * A;
            AtA.diagonal() += Eigen::VectorXd::Constant(num_features + 1, lambda);
            w = AtA.ldlt().solve(A.transpose() * b);
        } catch (...) {
            return false;
        }
        
        // Coefficient pruning: zero out weak weights
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
        return true;
    }
    
    // Fast analytical solver for linear weights (used during evolution)
    void refine_constants(IndividualGraph& ind) {
        int n_samples = static_cast<int>(y_.size());
        
        std::vector<Eigen::ArrayXd> cache;
        evaluate_graph_cached(ind, X_, n_samples, cache, gen_cache_);
        
        if (!ind.nodes.empty()) {
            solve_output_weights(ind, cache);
            evaluate_fitness_with_penalty(ind, X_, y_, n_samples);
            // NOTE: refine_inner_params is NOT called here — too expensive
            // for per-child use. It runs once on the best graph in cleanup_graph.
        }
    }
    
    // ── Finite-Difference Adam Optimizer for Inner Parameters ────────────
    // Alternates between: (1) Adam steps on {p, omega, phi},
    // then (2) SVD refit of output weights. This ensures linear weights
    // stay in sync with the refined inner parameters.
    void refine_inner_params(IndividualGraph& ind) {
        if (ind.nodes.empty()) return;
        int n_samples = static_cast<int>(y_.size());
        
        // Collect indices of active unary nodes (non-zero output weight)
        std::vector<int> active_unary;
        for (int i = 0; i < static_cast<int>(ind.nodes.size()); ++i) {
            if (ind.nodes[i].type == NodeType::Unary && 
                std::abs(ind.output_weights[i]) > 1e-4) {
                active_unary.push_back(i);
            }
        }
        if (active_unary.empty()) return;
        
        // Adam hyperparameters
        const double lr = 0.02;
        const double beta1 = 0.9, beta2 = 0.999, eps_adam = 1e-8;
        const double epsilon = 1e-4; // Finite difference step
        const int adam_steps_per_round = 10;
        const int num_rounds = 3; // Alternate Adam → SVD this many times
        
        int n_params = static_cast<int>(active_unary.size()) * 3; // {p, omega, phi} — NOT amplitude (redundant with SVD output_weight)
        std::vector<double> m(n_params, 0.0), v(n_params, 0.0);
        
        double best_mse = ind.raw_mse;
        IndividualGraph best_snapshot = ind; // Keep best seen
        int global_step = 0;
        
        for (int round = 0; round < num_rounds; ++round) {
            // ── Phase 1: Adam steps on inner params ──
            for (int step = 0; step < adam_steps_per_round; ++step) {
                std::vector<double> grads(n_params, 0.0);
                
                for (int ai = 0; ai < static_cast<int>(active_unary.size()); ++ai) {
                    int node_idx = active_unary[ai];
                    auto& node = ind.nodes[node_idx];
                    // Only optimize {p, omega, phi} — amplitude handled by SVD
                    double* params[3] = {&node.p, &node.omega, &node.phi};
                    
                    for (int pi = 0; pi < 3; ++pi) {
                        double original = *params[pi];
                        
                        *params[pi] = original + epsilon;
                        Eigen::ArrayXd pred_plus = evaluate_graph_simple(ind, X_, n_samples);
                        double mse_plus = (pred_plus - y_).square().mean();
                        
                        *params[pi] = original - epsilon;
                        Eigen::ArrayXd pred_minus = evaluate_graph_simple(ind, X_, n_samples);
                        double mse_minus = (pred_minus - y_).square().mean();
                        
                        *params[pi] = original;
                        
                        double grad = (mse_plus - mse_minus) / (2.0 * epsilon);
                        if (!std::isfinite(grad)) grad = 0.0;
                        grads[ai * 3 + pi] = grad;
                    }
                }
                
                // Adam update
                for (int ai = 0; ai < static_cast<int>(active_unary.size()); ++ai) {
                    int node_idx = active_unary[ai];
                    auto& node = ind.nodes[node_idx];
                    double* params[3] = {&node.p, &node.omega, &node.phi};
                    
                    for (int pi = 0; pi < 3; ++pi) {
                        int idx = ai * 3 + pi;
                        double g = grads[idx];
                        
                        m[idx] = beta1 * m[idx] + (1.0 - beta1) * g;
                        v[idx] = beta2 * v[idx] + (1.0 - beta2) * g * g;
                        
                        double m_hat = m[idx] / (1.0 - std::pow(beta1, global_step + 1));
                        double v_hat = v[idx] / (1.0 - std::pow(beta2, global_step + 1));
                        
                        double update = lr * m_hat / (std::sqrt(v_hat) + eps_adam);
                        *params[pi] -= update;
                    }
                    
                    node.p = std::clamp(node.p, config_.p_min, config_.p_max);
                    node.omega = std::clamp(node.omega, config_.omega_min, config_.omega_max);
                }
                global_step++;
            }
            
            // ── Phase 2: Ridge refit of output weights ──
            // Inner params changed, so re-solve the linear layer analytically
            {
                std::vector<Eigen::ArrayXd> cache;
                evaluate_graph(ind, X_, n_samples, cache);
                if (!solve_output_weights(ind, cache)) continue;
            }
            
            // Evaluate and track best
            evaluate_fitness_with_penalty(ind, X_, y_, n_samples);
            if (ind.raw_mse < best_mse) {
                best_mse = ind.raw_mse;
                best_snapshot = ind;
            }
            
            // Early exit if already excellent
            if (ind.raw_mse < 1e-10) break;
        }
        
        // Restore best seen (in case later rounds degraded)
        if (best_snapshot.raw_mse < ind.raw_mse) {
            ind = best_snapshot;
        }
    }
    
    // ── Post-Evolution Graph Cleanup ─────────────────────────────────────
    // Uses output-correlation-based deduplication (like PyTorch pruning.py):
    // Nodes producing identical outputs get merged regardless of structure.
    // This catches x == (x+x)/2 == (x+x+x)/3 etc.
    void cleanup_graph(IndividualGraph& ind) {
        if (ind.nodes.empty()) return;
        int n_samples = static_cast<int>(y_.size());
        
        // ── Step 1: Evaluate all nodes to get their actual output vectors ──
        std::vector<Eigen::ArrayXd> cache;
        evaluate_graph(ind, X_, n_samples, cache);
        
        // ── Step 2: Correlation-based deduplication ──
        // Group nodes that produce (nearly) identical outputs
        int n_nodes = static_cast<int>(ind.nodes.size());
        std::vector<int> canonical(n_nodes); // Maps each node to its canonical representative
        for (int i = 0; i < n_nodes; ++i) canonical[i] = i;
        
        for (int i = 0; i < n_nodes; ++i) {
            if (canonical[i] != i) continue; // Already merged
            if (i >= static_cast<int>(ind.output_weights.size())) continue;
            if (std::abs(ind.output_weights[i]) < 1e-8) continue; // Skip dead
            
            // Skip if output is all zeros or constant NaN
            if (!cache[i].isFinite().all()) continue;
            double var_i = (cache[i] - cache[i].mean()).square().mean();
            
            for (int j = i + 1; j < n_nodes; ++j) {
                if (canonical[j] != j) continue;
                if (j >= static_cast<int>(ind.output_weights.size())) continue;
                if (std::abs(ind.output_weights[j]) < 1e-8) continue;
                if (!cache[j].isFinite().all()) continue;
                
                // Check if outputs are identical (or proportional)
                // Use normalized correlation: sum((a-mean(a)) * (b-mean(b))) / (std(a)*std(b)*N)
                double var_j = (cache[j] - cache[j].mean()).square().mean();
                
                bool is_duplicate = false;
                
                if (var_i < 1e-12 && var_j < 1e-12) {
                    // Both constant — check if same constant
                    is_duplicate = std::abs(cache[i].mean() - cache[j].mean()) < 1e-6;
                } else if (var_i > 1e-12 && var_j > 1e-12) {
                    // Both non-constant — check correlation AND scale
                    Eigen::ArrayXd diff = cache[i] - cache[j];
                    double max_abs_diff = diff.abs().maxCoeff();
                    double max_abs_val = cache[i].abs().maxCoeff();
                    
                    if (max_abs_val > 1e-10) {
                        // Relative error check: are they the same output?
                        double rel_err = max_abs_diff / max_abs_val;
                        is_duplicate = (rel_err < 1e-4);
                    } else {
                        is_duplicate = (max_abs_diff < 1e-10);
                    }
                    
                    // Also check for proportional outputs (a = k*b)
                    // These can be merged since SVD handles the scaling
                    if (!is_duplicate) {
                        Eigen::ArrayXd a_norm = cache[i] - cache[i].mean();
                        Eigen::ArrayXd b_norm = cache[j] - cache[j].mean();
                        double corr = (a_norm * b_norm).mean() / 
                                     (std::sqrt(var_i * var_j) + 1e-15);
                        is_duplicate = (std::abs(corr) > 0.9999);
                    }
                }
                
                if (is_duplicate) {
                    canonical[j] = i; // j is a duplicate of i
                }
            }
        }
        
        // Merge: for each group, keep canonical node, zero out duplicates
        // Don't sum weights — let SVD refit handle optimal weights
        for (int j = 0; j < n_nodes; ++j) {
            if (canonical[j] != j && j < static_cast<int>(ind.output_weights.size())) {
                ind.output_weights[j] = 0.0;
            }
        }
        
        // ── Step 3: Remove dead nodes (zero output weight, not a dependency) ──
        std::vector<bool> keep(n_nodes, false);
        
        // First pass: mark nodes with non-zero weight
        for (int i = 0; i < n_nodes; ++i) {
            if (i < static_cast<int>(ind.output_weights.size()) && 
                std::abs(ind.output_weights[i]) > 1e-8) {
                keep[i] = true;
            }
        }
        
        // Second pass: mark dependencies of kept nodes
        for (int i = n_nodes - 1; i >= 0; --i) {
            if (keep[i]) {
                const auto& n = ind.nodes[i];
                if (n.left_child >= 0 && n.left_child < n_nodes) keep[n.left_child] = true;
                if (n.right_child >= 0 && n.right_child < n_nodes) keep[n.right_child] = true;
            }
        }
        
        // Build compacted graph
        std::vector<OpNode> clean_nodes;
        std::vector<double> clean_weights;
        std::vector<int> old_to_new(n_nodes, -1);
        
        for (int i = 0; i < n_nodes; ++i) {
            if (keep[i]) {
                old_to_new[i] = static_cast<int>(clean_nodes.size());
                clean_nodes.push_back(ind.nodes[i]);
                if (i < static_cast<int>(ind.output_weights.size())) {
                    clean_weights.push_back(ind.output_weights[i]);
                } else {
                    clean_weights.push_back(0.0);
                }
            }
        }
        
        // Remap child pointers
        for (auto& node : clean_nodes) {
            if (node.left_child >= 0 && node.left_child < n_nodes) {
                node.left_child = old_to_new[node.left_child];
            }
            if (node.right_child >= 0 && node.right_child < n_nodes) {
                node.right_child = old_to_new[node.right_child];
            }
            if (node.left_child < 0 && (node.type == NodeType::Unary || node.type == NodeType::Binary)) {
                node.type = NodeType::Constant;
                node.value = 0.0;
            }
            if (node.right_child < 0 && node.type == NodeType::Binary) {
                node.type = NodeType::Unary;
            }
        }
        
        // Only accept the cleanup if graph got smaller
        if (clean_nodes.size() < ind.nodes.size()) {
            ind.nodes = std::move(clean_nodes);
            ind.output_weights = std::move(clean_weights);
        }
        
        // ── Step 4: Ridge refit on clean graph ──
        if (!ind.nodes.empty()) {
            std::vector<Eigen::ArrayXd> new_cache;
            evaluate_graph(ind, X_, n_samples, new_cache);
            solve_output_weights(ind, new_cache);
        }
        
        evaluate_fitness_with_penalty(ind, X_, y_, n_samples);
        double baseline_mse = ind.raw_mse;
        
        // ── Step 5: Iterative Backward Elimination ──
        // Greedily remove least-important node, re-solve Ridge, repeat
        // until removing any more node degrades MSE too much.
        for (int elim_iter = 0; elim_iter < 10; ++elim_iter) {
            
            // Find least important node (smallest non-zero |output_weight|, non-Input)
            int weakest = -1;
            double weakest_weight = 1e18;
            for (int i = 0; i < static_cast<int>(ind.nodes.size()); ++i) {
                if (ind.nodes[i].type == NodeType::Input) continue;
                if (i >= static_cast<int>(ind.output_weights.size())) continue;
                double w = std::abs(ind.output_weights[i]);
                if (w < 1e-6) continue; // Skip already-dead nodes
                if (w < weakest_weight) {
                    weakest_weight = w;
                    weakest = i;
                }
            }
            
            if (weakest < 0) break; // No more removable nodes
            
            // Try removing it
            IndividualGraph candidate = ind;
            candidate.output_weights[weakest] = 0.0;
            
            // Re-evaluate without that node and re-solve
            std::vector<Eigen::ArrayXd> trial_cache;
            evaluate_graph(candidate, X_, n_samples, trial_cache);
            solve_output_weights(candidate, trial_cache);
            evaluate_fitness_with_penalty(candidate, X_, y_, n_samples);
            
            // Accept if MSE is still acceptable (within 2x baseline)
            if (candidate.raw_mse < baseline_mse * 2.0 + 1e-8) {
                ind = candidate;
                baseline_mse = std::max(baseline_mse, ind.raw_mse);
            } else {
                break; // Can't remove any more without hurting accuracy
            }
        }
        
        // ── Step 6: Parameter & Coefficient Snapping ─────────────────────────
        // Try rounding inner parameters and output weights to clean values.
        // This converts 0.9997*sin(2.998*x + 0.0012) → sin(3*x).
        {
            evaluate_fitness_with_penalty(ind, X_, y_, n_samples);
            double snap_baseline_mse = ind.raw_mse;
            
            // 6a. Inner parameter snapping (p, omega, phi)
            const double snap_candidates_p[] = {-2, -1.5, -1, -0.5, 0, 0.25, 1.0/3.0, 0.5, 2.0/3.0, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5};
            const int n_snap_p = sizeof(snap_candidates_p) / sizeof(snap_candidates_p[0]);
            
            const double snap_candidates_omega[] = {0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, M_PI, 2*M_PI, M_PI/2};
            const int n_snap_omega = sizeof(snap_candidates_omega) / sizeof(snap_candidates_omega[0]);
            
            const double snap_candidates_phi[] = {0.0, M_PI/4, M_PI/2, M_PI, 3*M_PI/2, -M_PI/4, -M_PI/2, -M_PI};
            const int n_snap_phi = sizeof(snap_candidates_phi) / sizeof(snap_candidates_phi[0]);
            
            for (int i = 0; i < static_cast<int>(ind.nodes.size()); ++i) {
                if (ind.nodes[i].type != NodeType::Unary) continue;
                if (i >= static_cast<int>(ind.output_weights.size())) continue;
                if (std::abs(ind.output_weights[i]) < 1e-6) continue; // Skip dead nodes
                
                auto& node = ind.nodes[i];
                
                // Try snapping p
                {
                    double original_p = node.p;
                    double best_snap_p = original_p;
                    double best_snap_mse = snap_baseline_mse;
                    
                    for (int si = 0; si < n_snap_p; ++si) {
                        double candidate = snap_candidates_p[si];
                        if (std::abs(original_p - candidate) > 0.3) continue; // Only snap if close
                        
                        node.p = candidate;
                        // Re-solve output weights with snapped parameter
                        std::vector<Eigen::ArrayXd> snap_cache;
                        evaluate_graph(ind, X_, n_samples, snap_cache);
                        solve_output_weights(ind, snap_cache);
                        evaluate_fitness_with_penalty(ind, X_, y_, n_samples);
                        
                        if (ind.raw_mse < best_snap_mse * 1.5 + 1e-8) {
                            best_snap_p = candidate;
                            best_snap_mse = ind.raw_mse;
                        }
                    }
                    node.p = best_snap_p;
                    if (best_snap_p != original_p) {
                        // Re-solve with accepted snap
                        std::vector<Eigen::ArrayXd> snap_cache;
                        evaluate_graph(ind, X_, n_samples, snap_cache);
                        solve_output_weights(ind, snap_cache);
                        evaluate_fitness_with_penalty(ind, X_, y_, n_samples);
                        snap_baseline_mse = ind.raw_mse;
                    }
                }
                
                // Try snapping omega
                if (node.unary_op == UnaryOp::Periodic) {
                    double original_omega = node.omega;
                    double best_snap_omega = original_omega;
                    double best_snap_mse = snap_baseline_mse;
                    
                    // Also try snapping to nearest integer
                    double nearest_int = std::round(original_omega);
                    
                    for (int si = 0; si < n_snap_omega; ++si) {
                        double candidate = snap_candidates_omega[si];
                        if (std::abs(original_omega - candidate) > 0.3) continue;
                        
                        node.omega = candidate;
                        std::vector<Eigen::ArrayXd> snap_cache;
                        evaluate_graph(ind, X_, n_samples, snap_cache);
                        solve_output_weights(ind, snap_cache);
                        evaluate_fitness_with_penalty(ind, X_, y_, n_samples);
                        
                        if (ind.raw_mse < best_snap_mse * 1.5 + 1e-8) {
                            best_snap_omega = candidate;
                            best_snap_mse = ind.raw_mse;
                        }
                    }
                    // Try nearest integer if not already in candidates
                    if (nearest_int >= 1.0 && nearest_int <= 10.0 && std::abs(original_omega - nearest_int) <= 0.3) {
                        node.omega = nearest_int;
                        std::vector<Eigen::ArrayXd> snap_cache;
                        evaluate_graph(ind, X_, n_samples, snap_cache);
                        solve_output_weights(ind, snap_cache);
                        evaluate_fitness_with_penalty(ind, X_, y_, n_samples);
                        
                        if (ind.raw_mse < best_snap_mse * 1.5 + 1e-8) {
                            best_snap_omega = nearest_int;
                            best_snap_mse = ind.raw_mse;
                        }
                    }
                    
                    node.omega = best_snap_omega;
                    if (best_snap_omega != original_omega) {
                        std::vector<Eigen::ArrayXd> snap_cache;
                        evaluate_graph(ind, X_, n_samples, snap_cache);
                        solve_output_weights(ind, snap_cache);
                        evaluate_fitness_with_penalty(ind, X_, y_, n_samples);
                        snap_baseline_mse = ind.raw_mse;
                    }
                    
                    // Try snapping phi
                    double original_phi = node.phi;
                    double best_snap_phi = original_phi;
                    best_snap_mse = snap_baseline_mse;
                    
                    for (int si = 0; si < n_snap_phi; ++si) {
                        double candidate = snap_candidates_phi[si];
                        if (std::abs(original_phi - candidate) > 0.3) continue;
                        
                        node.phi = candidate;
                        std::vector<Eigen::ArrayXd> snap_cache;
                        evaluate_graph(ind, X_, n_samples, snap_cache);
                        solve_output_weights(ind, snap_cache);
                        evaluate_fitness_with_penalty(ind, X_, y_, n_samples);
                        
                        if (ind.raw_mse < best_snap_mse * 1.5 + 1e-8) {
                            best_snap_phi = candidate;
                            best_snap_mse = ind.raw_mse;
                        }
                    }
                    node.phi = best_snap_phi;
                    if (best_snap_phi != original_phi) {
                        std::vector<Eigen::ArrayXd> snap_cache;
                        evaluate_graph(ind, X_, n_samples, snap_cache);
                        solve_output_weights(ind, snap_cache);
                        evaluate_fitness_with_penalty(ind, X_, y_, n_samples);
                        snap_baseline_mse = ind.raw_mse;
                    }
                }
            }
            
            // 6a.5 Trigonometric identity simplification
            // -sin(x + pi) = sin(x), -sin(x - pi) = sin(x)
            // sin(x + 2*pi) = sin(x), etc.
            for (int i = 0; i < static_cast<int>(ind.nodes.size()); ++i) {
                if (ind.nodes[i].type != NodeType::Unary) continue;
                if (ind.nodes[i].unary_op != UnaryOp::Periodic) continue;
                if (i >= static_cast<int>(ind.output_weights.size())) continue;
                if (std::abs(ind.output_weights[i]) < 1e-6) continue;
                
                auto& node = ind.nodes[i];
                double w = ind.output_weights[i];
                
                // Remove full 2*pi multiples from phi
                if (std::abs(node.phi) > M_PI) {
                    double reduced = std::fmod(node.phi, 2.0 * M_PI);
                    if (reduced > M_PI) reduced -= 2.0 * M_PI;
                    if (reduced < -M_PI) reduced += 2.0 * M_PI;
                    node.phi = reduced;
                }
                
                // -sin(x + pi) = sin(x): if phi ≈ ±π and weight is negative,
                // flip weight sign and zero out phi
                if (std::abs(std::abs(node.phi) - M_PI) < 0.05 && w < 0) {
                    node.phi = 0.0;
                    ind.output_weights[i] = -w;  // Flip sign
                }
                // sin(x + pi) with positive weight → -sin(x)
                else if (std::abs(std::abs(node.phi) - M_PI) < 0.05 && w > 0) {
                    node.phi = 0.0;
                    ind.output_weights[i] = -w;  // Flip sign
                }
                
                // -sin(x + pi/2) = -cos(x) ... leave as-is (no simplification needed)
                
                // If phi is now ~0, finalize it
                if (std::abs(node.phi) < 0.05) {
                    node.phi = 0.0;
                }
                
                // If amplitude is negative, absorb into output weight
                if (node.amplitude < 0) {
                    ind.output_weights[i] = -ind.output_weights[i];
                    node.amplitude = -node.amplitude;
                }
            }
            
            // Re-evaluate after trig simplification
            evaluate_fitness_with_penalty(ind, X_, y_, n_samples);
            snap_baseline_mse = ind.raw_mse;
            
            // 6b. Output weight snapping
            {
                const double snap_weight_values[] = {
                    0.0, 0.25, 1.0/3.0, 0.5, 2.0/3.0, 0.75,
                    1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                    M_PI, M_E, std::sqrt(2.0), std::sqrt(3.0)
                };
                const int n_snap_w = sizeof(snap_weight_values) / sizeof(snap_weight_values[0]);
                
                for (int i = 0; i < static_cast<int>(ind.output_weights.size()); ++i) {
                    double w = ind.output_weights[i];
                    if (std::abs(w) < 1e-6) continue; // Already zero
                    
                    double abs_w = std::abs(w);
                    double sign_w = (w >= 0) ? 1.0 : -1.0;
                    
                    double best_snap_w = w;
                    double best_snap_mse = snap_baseline_mse;
                    
                    for (int si = 0; si < n_snap_w; ++si) {
                        double candidate = sign_w * snap_weight_values[si];
                        // Also try the opposite sign for zero candidate
                        double rel_dist = (abs_w > 1e-8) ? std::abs(abs_w - snap_weight_values[si]) / abs_w : 1.0;
                        if (rel_dist > 0.15 && std::abs(w - candidate) > 0.3) continue; // Within 15% or 0.3 absolute
                        
                        double original_w = ind.output_weights[i];
                        ind.output_weights[i] = candidate;
                        evaluate_fitness_with_penalty(ind, X_, y_, n_samples);
                        
                        if (ind.raw_mse < best_snap_mse * 1.01 + 1e-8) {
                            best_snap_w = candidate;
                            best_snap_mse = ind.raw_mse;
                        }
                        ind.output_weights[i] = original_w; // Restore
                    }
                    
                    if (best_snap_w != w) {
                        ind.output_weights[i] = best_snap_w;
                        evaluate_fitness_with_penalty(ind, X_, y_, n_samples);
                        snap_baseline_mse = ind.raw_mse;
                    }
                }
            }
            
            // 6c. Output bias snapping
            {
                double bias = ind.output_bias;
                if (std::abs(bias) > 1e-6) {
                    const double snap_bias_values[] = {
                        0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0,
                        -0.25, -0.5, -0.75, -1.0, -1.5, -2.0, -3.0, -4.0, -5.0,
                        M_PI, -M_PI, M_E, -M_E
                    };
                    const int n_snap_b = sizeof(snap_bias_values) / sizeof(snap_bias_values[0]);
                    
                    double best_snap_b = bias;
                    double best_snap_mse = snap_baseline_mse;
                    
                    for (int si = 0; si < n_snap_b; ++si) {
                        double candidate = snap_bias_values[si];
                        if (std::abs(bias - candidate) > 0.5) continue;
                        
                        ind.output_bias = candidate;
                        evaluate_fitness_with_penalty(ind, X_, y_, n_samples);
                        
                        if (ind.raw_mse < best_snap_mse * 1.01 + 1e-8) {
                            best_snap_b = candidate;
                            best_snap_mse = ind.raw_mse;
                        }
                    }
                    // Also try nearest integer
                    double nearest_int = std::round(bias);
                    if (std::abs(bias - nearest_int) <= 0.3) {
                        ind.output_bias = nearest_int;
                        evaluate_fitness_with_penalty(ind, X_, y_, n_samples);
                        if (ind.raw_mse < best_snap_mse * 1.01 + 1e-8) {
                            best_snap_b = nearest_int;
                            best_snap_mse = ind.raw_mse;
                        }
                    }
                    
                    ind.output_bias = best_snap_b;
                }
            }
            
            // Final Ridge refit after all snapping
            if (!ind.nodes.empty()) {
                std::vector<Eigen::ArrayXd> final_cache;
                evaluate_graph(ind, X_, n_samples, final_cache);
                solve_output_weights(ind, final_cache);
            }
        }
        
        // Final inner param refinement on the clean graph
        refine_inner_params(ind);
    }

    // ── P5: NSGA-II Non-Dominated Sort ────────────────────────────────────
    // Assigns pareto_rank to each individual in the population.
    // Objectives: minimize raw_mse, minimize complexity(), minimize age (AFPO).
    void non_dominated_sort(std::vector<IndividualGraph>& pop) {
        int n = static_cast<int>(pop.size());
        if (n == 0) return;

        // domination_count[i] = how many solutions dominate i
        std::vector<int> domination_count(n, 0);
        // dominated_set[i] = list of solutions that i dominates
        std::vector<std::vector<int>> dominated_set(n);

        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                // 3-objective dominance: minimize raw_mse, complexity, age
                double mse_i = pop[i].raw_mse, mse_j = pop[j].raw_mse;
                int comp_i = pop[i].complexity(), comp_j = pop[j].complexity();
                int age_i = pop[i].age, age_j = pop[j].age;
                
                bool i_leq_j = (mse_i <= mse_j) && (comp_i <= comp_j) && (age_i <= age_j);
                bool i_lt_j  = (mse_i < mse_j) || (comp_i < comp_j) || (age_i < age_j);
                bool i_dom_j = i_leq_j && i_lt_j;
                
                bool j_leq_i = (mse_j <= mse_i) && (comp_j <= comp_i) && (age_j <= age_i);
                bool j_lt_i  = (mse_j < mse_i) || (comp_j < comp_i) || (age_j < age_i);
                bool j_dom_i = j_leq_i && j_lt_i;

                if (i_dom_j) {
                    dominated_set[i].push_back(j);
                    domination_count[j]++;
                } else if (j_dom_i) {
                    dominated_set[j].push_back(i);
                    domination_count[i]++;
                }
            }
        }

        // Assign ranks front-by-front
        std::vector<int> current_front;
        for (int i = 0; i < n; ++i) {
            if (domination_count[i] == 0) {
                pop[i].pareto_rank = 0;
                current_front.push_back(i);
            }
        }

        int rank = 0;
        while (!current_front.empty()) {
            std::vector<int> next_front;
            for (int i : current_front) {
                for (int j : dominated_set[i]) {
                    domination_count[j]--;
                    if (domination_count[j] == 0) {
                        pop[j].pareto_rank = rank + 1;
                        next_front.push_back(j);
                    }
                }
            }
            rank++;
            current_front = std::move(next_front);
        }
    }

    // ── P5: Crowding Distance Assignment ──────────────────────────────────
    // Assigns crowding_distance to individuals within the same Pareto front.
    // 3 objectives: raw_mse, complexity, age (AFPO).
    void crowding_distance_assignment(std::vector<IndividualGraph*>& front) {
        int n = static_cast<int>(front.size());
        if (n == 0) return;

        for (auto* ind : front) ind->crowding_distance = 0.0;
        if (n <= 2) {
            for (auto* ind : front) ind->crowding_distance = 1e18;
            return;
        }

        // For each objective, sort and compute distance
        // Objective 0: raw_mse
        std::sort(front.begin(), front.end(),
                  [](const IndividualGraph* a, const IndividualGraph* b) {
                      return a->raw_mse < b->raw_mse;
                  });
        front.front()->crowding_distance = 1e18;
        front.back()->crowding_distance = 1e18;
        double mse_range = front.back()->raw_mse - front.front()->raw_mse;
        if (mse_range > 1e-15) {
            for (int i = 1; i < n - 1; ++i) {
                front[i]->crowding_distance += (front[i+1]->raw_mse - front[i-1]->raw_mse) / mse_range;
            }
        }

        // Objective 1: complexity
        std::sort(front.begin(), front.end(),
                  [](const IndividualGraph* a, const IndividualGraph* b) {
                      return a->complexity() < b->complexity();
                  });
        front.front()->crowding_distance = 1e18;
        front.back()->crowding_distance = 1e18;
        double comp_range = front.back()->complexity() - front.front()->complexity();
        if (comp_range > 1e-15) {
            for (int i = 1; i < n - 1; ++i) {
                front[i]->crowding_distance += (double)(front[i+1]->complexity() - front[i-1]->complexity()) / comp_range;
            }
        }

        // Objective 2: age (AFPO)
        std::sort(front.begin(), front.end(),
                  [](const IndividualGraph* a, const IndividualGraph* b) {
                      return a->age < b->age;
                  });
        front.front()->crowding_distance = 1e18;
        front.back()->crowding_distance = 1e18;
        double age_range = static_cast<double>(front.back()->age - front.front()->age);
        if (age_range > 0.5) {
            for (int i = 1; i < n - 1; ++i) {
                front[i]->crowding_distance += (double)(front[i+1]->age - front[i-1]->age) / age_range;
            }
        }
    }

    // ── P5: NSGA-II Selection ─────────────────────────────────────────────
    // Select pop_size individuals from a combined pool using NSGA-II ranking.
    std::vector<IndividualGraph> nsga2_select(std::vector<IndividualGraph>& combined, int target_size) {
        non_dominated_sort(combined);

        // Group by rank
        int max_rank = 0;
        for (auto& ind : combined) max_rank = std::max(max_rank, ind.pareto_rank);

        std::vector<IndividualGraph> selected;
        selected.reserve(target_size);

        for (int r = 0; r <= max_rank && static_cast<int>(selected.size()) < target_size; ++r) {
            std::vector<IndividualGraph*> front;
            for (auto& ind : combined) {
                if (ind.pareto_rank == r) front.push_back(&ind);
            }
            crowding_distance_assignment(front);

            // Sort this front by crowding distance (descending)
            std::sort(front.begin(), front.end(),
                      [](const IndividualGraph* a, const IndividualGraph* b) {
                          return a->crowding_distance > b->crowding_distance;
                      });

            for (auto* ind : front) {
                if (static_cast<int>(selected.size()) >= target_size) break;
                selected.push_back(*ind);
            }
        }

        return selected;
    }

    // ── P6: Single-generation evolution step (for island model) ───────────
    void evolve_one_generation(int gen) {
        evaluate_population();

        std::sort(population_.begin(), population_.end(),
                  [](const IndividualGraph& a, const IndividualGraph& b) {
                      return a.fitness < b.fitness;
                  });

        if (population_[0].fitness < best_overall_.fitness) {
            best_overall_ = population_[0];
        }

        // Create next generation (same logic as run() loop body)
        std::vector<IndividualGraph> next_gen;
        next_gen.reserve(config_.pop_size);

        for (int i = 0; i < config_.elite_size && i < static_cast<int>(population_.size()); ++i) {
            next_gen.push_back(population_[i]);
        }

        double current_structural_mutation_rate = config_.mutation_rate_structural;
        std::uniform_int_distribution<int> parent_dist(0, std::max(0, config_.elite_size - 1));
        std::uniform_real_distribution<double> coin(0.0, 1.0);

        int num_explorers = static_cast<int>(config_.pop_size * config_.explorer_fraction);
        int main_pop_target = config_.pop_size - num_explorers;

        while (static_cast<int>(next_gen.size()) < main_pop_target) {
            IndividualGraph child;
            if (config_.elite_size >= 2 && coin(rng_) < config_.crossover_rate) {
                int p1 = parent_dist(rng_);
                int p2 = parent_dist(rng_);
                while (p2 == p1) p2 = parent_dist(rng_);
                child = crossover(population_[p1], population_[p2]);
                child = mutate_lamarckian(child, current_structural_mutation_rate * 0.3);
            } else {
                int parent_idx = parent_dist(rng_);
                child = mutate_lamarckian(population_[parent_idx], current_structural_mutation_rate);
            }
            if (gen % 5 == 0) {
                refine_constants(child);
            } else {
                evaluate_fitness_with_penalty(child, X_, y_, y_.size());
            }
            next_gen.push_back(std::move(child));
        }

        while (static_cast<int>(next_gen.size()) < config_.pop_size) {
            int parent_idx = parent_dist(rng_);
            double explorer_rate = std::min(1.0, current_structural_mutation_rate * config_.explorer_mutation_multiplier);
            IndividualGraph explorer = mutate_lamarckian(population_[parent_idx], explorer_rate);
            evaluate_fitness_with_penalty(explorer, X_, y_, y_.size());
            next_gen.push_back(std::move(explorer));
        }

        population_ = std::move(next_gen);

        if (gen % 20 == 19) {
            for (int i = 0; i < std::min(3, config_.elite_size); ++i) {
                refine_inner_params(population_[i]);
            }
        }
    }

    // ── P7: Dimensional Analysis Penalty ──────────────────────────────────
    double dimensional_penalty(const IndividualGraph& graph) {
        if (config_.input_units.empty()) return 0.0;

        int n_dims = static_cast<int>(config_.input_units[0].size());
        int n_nodes = static_cast<int>(graph.nodes.size());
        std::vector<std::vector<double>> node_units(n_nodes, std::vector<double>(n_dims, 0.0));

        // Propagate units bottom-up
        for (int i = 0; i < n_nodes; ++i) {
            const auto& node = graph.nodes[i];
            switch (node.type) {
                case NodeType::Input:
                    if (node.feature_idx < static_cast<int>(config_.input_units.size())) {
                        node_units[i] = config_.input_units[node.feature_idx];
                    }
                    break;
                case NodeType::Constant:
                    // Constants are dimensionless
                    break;
                case NodeType::Unary: {
                    std::vector<double> child_u(n_dims, 0.0);
                    if (node.left_child >= 0 && node.left_child < n_nodes)
                        child_u = node_units[node.left_child];

                    if (node.unary_op == UnaryOp::Power) {
                        // x^p: multiply units by p
                        for (int d = 0; d < n_dims; ++d)
                            node_units[i][d] = child_u[d] * node.p;
                    } else {
                        // sin, exp, log: argument must be dimensionless
                        // Result is dimensionless too
                        // Penalty for non-zero child units
                        // (units stay as zero — result is dimensionless)
                    }
                    break;
                }
                case NodeType::Binary: {
                    std::vector<double> left_u(n_dims, 0.0), right_u(n_dims, 0.0);
                    if (node.left_child >= 0 && node.left_child < n_nodes)
                        left_u = node_units[node.left_child];
                    if (node.right_child >= 0 && node.right_child < n_nodes)
                        right_u = node_units[node.right_child];

                    if (node.binary_op == BinaryOp::Arithmetic) {
                        if (node.beta < 1.5) {
                            // Addition: units must match, result = same
                            node_units[i] = left_u;
                        } else {
                            // Multiplication: add exponents
                            for (int d = 0; d < n_dims; ++d)
                                node_units[i][d] = left_u[d] + right_u[d];
                        }
                    } else {
                        node_units[i] = left_u; // Aggregation keeps units
                    }
                    break;
                }
            }
        }

        // Compute penalty as sum of squared unit mismatches
        double penalty = 0.0;
        for (int i = 0; i < n_nodes; ++i) {
            const auto& node = graph.nodes[i];
            if (node.type == NodeType::Unary && node.unary_op != UnaryOp::Power) {
                // sin/exp/log argument must be dimensionless
                if (node.left_child >= 0 && node.left_child < n_nodes) {
                    for (int d = 0; d < n_dims; ++d)
                        penalty += node_units[node.left_child][d] * node_units[node.left_child][d];
                }
            }
            if (node.type == NodeType::Binary && node.binary_op == BinaryOp::Arithmetic && node.beta < 1.5) {
                // Addition: left and right units must match
                if (node.left_child >= 0 && node.right_child >= 0 &&
                    node.left_child < n_nodes && node.right_child < n_nodes) {
                    for (int d = 0; d < n_dims; ++d) {
                        double diff = node_units[node.left_child][d] - node_units[node.right_child][d];
                        penalty += diff * diff;
                    }
                }
            }
        }

        // Output unit check: compare weighted sum units against target
        if (!config_.output_units.empty()) {
            // Check each active node's units against output units
            for (int i = 0; i < n_nodes && i < static_cast<int>(graph.output_weights.size()); ++i) {
                if (std::abs(graph.output_weights[i]) > 1e-4) {
                    for (int d = 0; d < n_dims && d < static_cast<int>(config_.output_units.size()); ++d) {
                        double diff = node_units[i][d] - config_.output_units[d];
                        penalty += diff * diff;
                    }
                }
            }
        }

        return penalty;
    }
};

} // namespace sr
