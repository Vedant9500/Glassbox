#pragma once

#include "ast.h"
#include "eval.h"
#include "execution.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <mutex>
#include <random>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <omp.h>

namespace sr {
// kPi/kE live in ast.h (shared with eval/parser).

// Phase 4 robust search losses (display / best_mse stay plain MSE).
enum class LossMode {
    Mse = 0,
    Huber = 1,
    TrimmedMse = 2,
    StudentT = 3,
};

// FC-1: process-global OpenMP active-level state must not be mutated
// concurrently from GIL-released threads. Serialize configuration with a
// process-wide mutex and restore via RAII (exception-safe).
inline std::mutex& island_openmp_state_mutex() {
    static std::mutex m;
    return m;
}

#if defined(_OPENMP)
class MaxActiveLevelsGuard {
public:
    explicit MaxActiveLevelsGuard(int desired) : locked_(false), previous_(1) {
        island_openmp_state_mutex().lock();
        locked_ = true;
        previous_ = omp_get_max_active_levels();
        omp_set_max_active_levels(desired);
    }
    ~MaxActiveLevelsGuard() {
        if (locked_) {
            omp_set_max_active_levels(previous_);
            island_openmp_state_mutex().unlock();
        }
    }
    MaxActiveLevelsGuard(const MaxActiveLevelsGuard&) = delete;
    MaxActiveLevelsGuard& operator=(const MaxActiveLevelsGuard&) = delete;
private:
    bool locked_;
    int previous_;
};
#endif

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
    double omega_min = -8.0, omega_max = 8.0;
    // Phase shift (Periodic/Exp). ±4π covers multi-period wraps; Exp uses a
    // tighter clamp in optimizers to avoid exp-argument blow-ups (H-03).
    double phi_min = -4.0 * kPi;
    double phi_max =  4.0 * kPi;
    double exp_phi_min = -4.0, exp_phi_max = 4.0;
    double exp_omega_min = -4.0, exp_omega_max = 4.0;
    int max_nodes = 24;  // Hard structural bloat cap for mutation/crossover

    // Phase 4: search fitness loss (default mse = legacy behaviour)
    LossMode loss_mode = LossMode::Mse;
    double huber_delta = -1.0;      // <=0 => MAD scale from residuals
    double trim_fraction = 0.1;     // fraction of largest residuals dropped
    
    bool use_early_stop = true;
    double early_stop_mse = 1e-6;
    int early_stop_max_nodes = 8;  // Max graph nodes for early-stop eligibility

    // Inner-parameter optimizer (nonlinear constants like p/omega/phi)
    bool use_lm_inner_optimizer = true;
    bool lm_fallback_to_adam = true;
    // P-04: skip FD probes for parameters that are analytically inert for the
    // node's unary op (p on Periodic/Exp, omega/phi on Power, all on Log/Abs).
    bool fd_skip_inert_params = true;
    int lm_max_iterations = 15;
    double lm_lambda_init = 1e-2;
    int timeout_seconds = 120;

    // Pruning and rounding
    double prune_threshold = 0.05;
    double round_penalty_weight = 0.01;

    // Search Dynamics
    double explorer_fraction = 0.2; // 20% of population are explorers
    double explorer_mutation_multiplier = 3.0;

    // Phase 4: reliability-first rollout controls
    bool use_staged_schedule = true;
    int topology_phase_generations = 40;
    double topology_phase_mutation_boost = 1.5;
    int topology_refine_interval = 20;

    bool use_adaptive_restart = true;
    int stagnation_window = 40;
    double stagnation_min_improvement = 1e-5;
    double diversity_floor = 0.25;
    double restart_fraction = 0.2;
    double post_restart_mutation_boost = 1.25;

    // Classifier priors: probability weights for
    // [Periodic, Power, IntPow, Exp, Log].
    // Legacy 4-slot priors [Periodic, Power, Exp, Log] are supported.
    // Empty = uniform sampling. Non-empty = sample proportionally.
    std::vector<double> op_priors; // e.g. {0.8, 0.08, 0.02, 0.05, 0.05}
    std::vector<int> allowed_unary_ops;
    // Binary priors: [Arithmetic, Division, Aggregation].
    // Empty = use the built-in defaults.
    std::vector<double> binary_op_priors;
    std::vector<int> allowed_binary_ops;

    // P5: NSGA-II multi-objective
    bool use_nsga2 = false;

    // P6: Island Model
    int num_islands = 1;          // 1 = single population (default)
    int migration_interval = 25;  // Exchange elites every N generations
    int migration_size = 2;       // Number of elites migrated per exchange
    
    // Diverse Islands Support
    std::vector<std::vector<double>> multi_op_priors;
    std::vector<std::vector<int>> multi_allowed_unary_ops;
    std::vector<std::vector<double>> multi_binary_op_priors;
    std::vector<std::vector<int>> multi_allowed_binary_ops;
    std::vector<std::vector<double>> multi_seed_omegas;

    // P7: Dimensional Analysis
    std::vector<std::vector<double>> input_units;  // Per-feature unit exponents
    std::vector<double> output_units;              // Target variable units
    double dim_penalty_weight = 0.1;

    // Reproducibility
    int random_seed = -1; // <0 => nondeterministic random_device

    // Soft-arithmetic blend sharpness (synced to eval.h during fitness eval).
    // Higher => closer to discrete +/x///- (E4 / E10).
    double arithmetic_temperature = 5.0;

    // Parallel pop evaluation thread budget. 0 => auto (max threads, or 1 if nested).
    int eval_num_threads = 0;

    // Seed injection cap fraction of population (E3). Historical default was 0.25.
    // Islands use small island_size; higher fraction avoids seed starvation.
    double seed_fraction = 0.5;

    // Macro mutation rate among offspring (historical hard-coded 0.15).
    // Raise when product/rational structure is expected (e.g. x^2*sin(x)).
    double macro_mutation_rate = 0.15;
    // Relative weights for macro modes: [wrap, multiply, divide, nest].
    // Empty => historical defaults {0.4, 0.2, 0.2, 0.2}.
    std::vector<double> macro_mode_weights;

    // Phase 0 evaluation hooks
    double acceptable_mse = 1e-3;
    int acceptable_complexity = 20;

    // Trace logging (JSONL)
    bool enable_trace = false;
    std::string trace_path;
    bool trace_include_formulas = false;
};

class DifferentialGramian {
public:
    int n_features() const { return n_features_; }
    int n_samples() const { return n_samples_; }

    void initialize(
        const std::vector<Eigen::ArrayXd>& cache,
        const Eigen::ArrayXd& y,
        const Eigen::ArrayXd& sample_weights = Eigen::ArrayXd()
    ) {
        n_features_ = static_cast<int>(cache.size());
        n_samples_ = static_cast<int>(y.size());
        y_ = y;
        use_weights_ = (sample_weights.size() == n_samples_ && n_samples_ > 0);
        if (use_weights_) {
            w_ = sample_weights;
            w_sum_ = w_.sum();
            if (!(w_sum_ > 0.0) || !std::isfinite(w_sum_)) {
                use_weights_ = false;
                w_ = Eigen::ArrayXd();
                w_sum_ = static_cast<double>(n_samples_);
            }
        } else {
            w_ = Eigen::ArrayXd();
            w_sum_ = static_cast<double>(n_samples_);
        }

        A_.resize(n_samples_, n_features_ + 1);
        for (int i = 0; i < n_features_; ++i) {
            A_.col(i) = cache[i].matrix();
        }
        A_.col(n_features_).setOnes(); // Bias column

        if (use_weights_) {
            Eigen::MatrixXd WA = w_.matrix().asDiagonal() * A_;
            G_ = A_.transpose() * WA;
            c_ = A_.transpose() * (w_ * y_).matrix();
        } else {
            G_ = A_.transpose() * A_;
            c_ = A_.transpose() * y.matrix();
        }
    }

    void update_nodes(const std::vector<int>& changed_indices,
                      const std::vector<Eigen::ArrayXd>& /*old_cache*/,
                      const std::vector<Eigen::ArrayXd>& new_cache) {
        if (changed_indices.empty()) return;

        // §3.4: update all changed design-matrix columns first, then
        // recompute cross terms together. Row-by-row updates using a
        // partially refreshed A_ can leave G_ inconsistent with A^T W A
        // when several nonlinear-descendant columns change at once.
        for (int idx : changed_indices) {
            if (idx < 0 || idx >= n_features_) continue;
            if (idx >= static_cast<int>(new_cache.size())) continue;
            if (new_cache[static_cast<size_t>(idx)].size() != n_samples_) continue;
            A_.col(idx) = new_cache[static_cast<size_t>(idx)].matrix();
        }
        if (use_weights_) {
            Eigen::MatrixXd WA = w_.matrix().asDiagonal() * A_;
            G_ = A_.transpose() * WA;
            c_ = A_.transpose() * (w_ * y_).matrix();
        } else {
            G_ = A_.transpose() * A_;
            c_ = A_.transpose() * y_.matrix();
        }
    }

    bool solve_ridge(double lambda, Eigen::VectorXd& w_out) const {
        // ColPivHouseholderQR for robustness on near-singular bases (no LLT cache).
        Eigen::MatrixXd G_ridge = G_;
        G_ridge.diagonal().array() += lambda;
        w_out = G_ridge.colPivHouseholderQr().solve(c_);
        return w_out.allFinite();
    }

    // Compute MSE directly from weights and a cache (avoids full graph re-eval).
    // When sample weights are set, returns weighted MSE used for selection.
    double compute_mse(const Eigen::VectorXd& w, const std::vector<Eigen::ArrayXd>& cache) const {
        Eigen::ArrayXd pred = Eigen::ArrayXd::Constant(n_samples_, w(n_features_)); // bias
        for (int f = 0; f < n_features_; ++f) {
            if (f < static_cast<int>(cache.size()) && cache[f].size() == n_samples_) {
                pred += w(f) * cache[f];
            }
        }
        Eigen::ArrayXd err2 = (pred - y_).square();
        double mse = use_weights_
            ? (w_ * err2).sum() / w_sum_
            : err2.mean();
        return std::isfinite(mse) ? mse : std::numeric_limits<double>::infinity();
    }

private:
    Eigen::MatrixXd A_; // Full design matrix: n_samples x (n_features + 1)
    Eigen::MatrixXd G_; // A^T W A : size (F+1) x (F+1)
    Eigen::VectorXd c_; // A^T W y : size (F+1)
    Eigen::ArrayXd y_;  // Target vector (owned copy, not a reference)
    Eigen::ArrayXd w_;  // Optional per-point weights (empty => uniform)
    double w_sum_ = 0.0;
    bool use_weights_ = false;
    int n_samples_ = 0;
    int n_features_ = 0;

};


// Shared anti-trig-bloat predicates (soft penalty + hard ban share the same rules).
inline bool is_low_omega_periodic(const OpNode& node) {
    return node.type == NodeType::Unary &&
           node.unary_op == UnaryOp::Periodic &&
           std::abs(node.omega) < 0.1;
}

inline bool is_nested_periodic(const IndividualGraph& graph, const OpNode& node) {
    if (node.type != NodeType::Unary || node.unary_op != UnaryOp::Periodic) return false;
    const int n = static_cast<int>(graph.nodes.size());
    if (node.left_child < 0 || node.left_child >= n) return false;
    const auto& child = graph.nodes[static_cast<size_t>(node.left_child)];
    return child.type == NodeType::Unary && child.unary_op == UnaryOp::Periodic;
}

class EvolutionEngine {
public:
    EvolutionEngine(const EvolutionConfig& config, 
                    const std::vector<Eigen::ArrayXd>& X, 
                    const Eigen::ArrayXd& y,
                    const std::vector<double>& seed_omegas = {},
                    const std::vector<IndividualGraph>& seed_graphs = {},
                    const Eigen::ArrayXd& y_weights = Eigen::ArrayXd())
                : config_(config), X_(X), y_(y), seed_omegas_(seed_omegas), seed_graphs_(seed_graphs),
                    rng_(config.random_seed >= 0
                            ? static_cast<unsigned int>(config.random_seed)
                            : std::random_device{}()) {
        set_y_weights(y_weights);
        sanitize_config();

        if (config_.enable_trace && !config_.trace_path.empty()) {
            trace_stream_.open(config_.trace_path, std::ios::out | std::ios::trunc);
            if (trace_stream_.is_open()) {
                trace_enabled_ = true;
                trace_event("run.start", -1);
            }
        }

        // Normalize op_priors if provided
        if (!config_.op_priors.empty()) {
            // Backward compatibility:
            //   4 slots [Periodic, Power, Exp, Log] -> 5 (+IntPow split)
            //   5 slots [Periodic, Power, IntPow, Exp, Log] -> 6 (+Abs=0 by default)
            if (config_.op_priors.size() == 4) {
                std::vector<double> expanded(5, 0.0);
                expanded[0] = config_.op_priors[0];               // Periodic
                expanded[1] = config_.op_priors[1] * 0.75;        // Power
                expanded[2] = config_.op_priors[1] * 0.25;        // IntPow
                expanded[3] = config_.op_priors[2];               // Exp
                expanded[4] = config_.op_priors[3];               // Log
                config_.op_priors = expanded;
            }
            if (config_.op_priors.size() == 5) {
                // Append Abs prior (0): seeds/simplify can still introduce Abs.
                config_.op_priors.push_back(0.0);
            }
            normalize_prior_vector(config_.op_priors);
            op_cdf_ = build_cdf(config_.op_priors);
        }

        if (!config_.binary_op_priors.empty()) {
            normalize_prior_vector(config_.binary_op_priors);
            binary_op_cdf_ = build_cdf(config_.binary_op_priors);
        }

        allowed_unary_ops_ = sanitize_allowed_ops<UnaryOp>(
            config_.allowed_unary_ops,
            static_cast<int>(UnaryOp::Abs)
        );
        allowed_binary_ops_ = sanitize_allowed_ops<BinaryOp>(
            config_.allowed_binary_ops,
            static_cast<int>(BinaryOp::Aggregation)
        );
        current_structural_mutation_rate_ = config_.mutation_rate_structural;
        best_mse_history_ = 1e9;
        plateau_counter_ = 0;
        recent_best_.reserve(config_.stagnation_window + 2);
    }

    // Main run loop
    void run() {
        auto start_time = std::chrono::steady_clock::now();
        initialize_population();
        
        // Initial Refinement
        for (auto& ind : population_) {
            refine_constants(ind);
        }
        trace_event("init.refined", -1);

        for (int gen = 0; gen < config_.generations; ++gen) {
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count() > config_.timeout_seconds) {
                break;
            }

            evolve_one_generation(gen);

            update_discovery_metrics(gen, start_time);

            if (config_.use_early_stop && early_stop_metric(best_overall_) < config_.early_stop_mse && best_overall_.nodes.size() <= static_cast<size_t>(config_.early_stop_max_nodes)) {
                trace_event("run.early_stop", gen);
                break;
            }
        }
        
        // Post-evolution cleanup: deduplicate + prune; then export may prefer raw champion.
        cleanup_graph(best_overall_);
        if (!best_raw_overall_.nodes.empty()) {
            cleanup_graph(best_raw_overall_);
        }
        // Re-evaluate export choice after cleanup (raw_mse may change).
        // consider_champion is non-const; re-pick via select after cleanup.
        update_discovery_metrics(config_.generations, start_time);
        run_wall_time_sec_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
        trace_event("run.end", -1);
    }

    IndividualGraph get_best() const {
        return select_export_champion();
    }

    void set_eval_num_threads(int n) { config_.eval_num_threads = std::max(0, n); }
    void set_arithmetic_temperature_config(double t) { config_.arithmetic_temperature = t; }
    int max_seed_capacity_public() const { return max_seed_capacity(); }


    int get_first_exact_generation() const { return first_exact_generation_; }
    double get_first_exact_time_sec() const { return first_exact_time_sec_; }
    int get_first_acceptable_generation() const { return first_acceptable_generation_; }
    double get_first_acceptable_time_sec() const { return first_acceptable_time_sec_; }
    double get_run_wall_time_sec() const { return run_wall_time_sec_; }
    int get_random_seed() const { return config_.random_seed; }
    int get_last_island_outer_threads() const { return last_island_outer_threads_; }
    int get_last_island_inner_threads() const { return last_island_inner_threads_; }
    bool get_last_crossover_valid() const { return last_crossover_valid_; }
    int get_crossover_attempts() const { return crossover_attempts_; }
    // P-04 diagnostics.
    long long get_fd_probes_total() const { return fd_probes_total_; }
    long long get_fd_probes_skipped_inert() const {
        return fd_probes_skipped_inert_;
    }
    int get_crossover_successes() const { return crossover_successes_; }
    double get_crossover_valid_rate() const {
        if (crossover_attempts_ <= 0) return 0.0;
        return static_cast<double>(crossover_successes_) / static_cast<double>(crossover_attempts_);
    }
    std::size_t get_subtree_cache_entries() const { return gen_cache_.size(); }
    std::size_t get_subtree_cache_bytes() const { return gen_cache_.bytes_used(); }
    std::size_t get_subtree_cache_evictions() const { return gen_cache_.evictions(); }
    std::size_t get_subtree_cache_max_entries() const { return gen_cache_.max_entries(); }
    std::size_t get_subtree_cache_max_bytes() const { return gen_cache_.max_bytes(); }

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
                                           a.active_complexity() == b.active_complexity();
                                });
        front.erase(last, front.end());

        // §3.104: export only finite-MSE individuals. Non-finite (inf/NaN)
        // models can hold rank 0 on complexity/age alone (see §3.102 guard
        // for selection); they must not appear in the reported
        // MSE/complexity Pareto front. NaN in particular is never dominated
        // by plain comparisons and would otherwise leak into the export.
        {
            std::vector<IndividualGraph> finite_front;
            for (auto& ind : front) {
                if (std::isfinite(ind.raw_mse)) finite_front.push_back(ind);
            }
            if (!finite_front.empty()) front.swap(finite_front);
        }

        // 2-objective domination filter (MSE + complexity only).
        // Internal NSGA-II uses 3 objectives (including age) for selection,
        // but the reported Pareto front should be clean on user-visible axes.
        std::vector<IndividualGraph> clean_front;
        for (size_t i = 0; i < front.size(); ++i) {
            bool dominated = false;
            for (size_t j = 0; j < front.size(); ++j) {
                if (i == j) continue;
                bool j_leq = (front[j].raw_mse <= front[i].raw_mse) &&
                              (front[j].active_complexity() <= front[i].active_complexity());
                bool j_lt  = (front[j].raw_mse < front[i].raw_mse) ||
                              (front[j].active_complexity() < front[i].active_complexity());
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
        auto start_time = std::chrono::steady_clock::now();
        auto timed_out = [&]() {
            return std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start_time
            ).count() >= static_cast<double>(config_.timeout_seconds);
        };

        // Create per-island engines with split configs
        std::vector<EvolutionEngine> islands;
        islands.reserve(config_.num_islands);
        EvolutionConfig island_cfg = config_;
        island_cfg.pop_size = island_size;
        island_cfg.elite_size = std::max(2, config_.elite_size / config_.num_islands);
        island_cfg.num_islands = 1; // Each island is single-population

        for (int i = 0; i < config_.num_islands; ++i) {
            EvolutionConfig current_island_cfg = island_cfg;
            if (i < config_.multi_op_priors.size() && !config_.multi_op_priors[i].empty()) {
                current_island_cfg.op_priors = config_.multi_op_priors[i];
            }
            if (i < config_.multi_allowed_unary_ops.size() && !config_.multi_allowed_unary_ops[i].empty()) {
                current_island_cfg.allowed_unary_ops = config_.multi_allowed_unary_ops[i];
            }
            if (i < config_.multi_binary_op_priors.size() && !config_.multi_binary_op_priors[i].empty()) {
                current_island_cfg.binary_op_priors = config_.multi_binary_op_priors[i];
            }
            if (i < config_.multi_allowed_binary_ops.size() && !config_.multi_allowed_binary_ops[i].empty()) {
                current_island_cfg.allowed_binary_ops = config_.multi_allowed_binary_ops[i];
            }
            
            std::vector<double> current_seed_omegas = seed_omegas_;
            if (i < config_.multi_seed_omegas.size() && !config_.multi_seed_omegas[i].empty()) {
                current_seed_omegas = config_.multi_seed_omegas[i];
            }

            // Distinct RNG stream per island under a fixed parent seed (E1).
            // Without this, all islands clone the same trajectories.
            if (config_.random_seed >= 0) {
                // Large coprime-ish stride keeps streams far apart.
                long long offset = static_cast<long long>(i) * 1000003LL;
                long long seed = static_cast<long long>(config_.random_seed) + offset + static_cast<long long>(i);
                // Keep in unsigned 32-bit range for mt19937.
                current_island_cfg.random_seed = static_cast<int>(
                    static_cast<unsigned int>(seed & 0xffffffffLL)
                );
            } else {
                current_island_cfg.random_seed = -1;
            }

            // Optional light sharding of seed graphs across islands to increase diversity.
            std::vector<IndividualGraph> island_seeds = seed_graphs_;
            if (seed_graphs_.size() > 1 && config_.num_islands > 1) {
                island_seeds.clear();
                for (size_t s = 0; s < seed_graphs_.size(); ++s) {
                    if (static_cast<int>(s % static_cast<size_t>(config_.num_islands)) == i) {
                        island_seeds.push_back(seed_graphs_[s]);
                    }
                }
                // Ensure every island gets at least one seed when possible.
                if (island_seeds.empty()) {
                    island_seeds.push_back(seed_graphs_[static_cast<size_t>(i % static_cast<int>(seed_graphs_.size()))]);
                }
            }
            
            // Pass seed_graphs_ + y_weights_ to all islands
            islands.emplace_back(
                current_island_cfg, X_, y_, current_seed_omegas, island_seeds,
                has_y_weights_ ? y_weights_ : Eigen::ArrayXd()
            );
        }

        // Initialize all islands. This includes constant refinement and can be
        // expensive, so run islands concurrently and keep inner OpenMP regions
        // small to avoid oversubscription.
        // E7: do NOT call omp_set_num_threads inside parallel regions (process-wide
        // race). Budget inner eval via config.eval_num_threads + num_threads clause.
        // FC-1: prefer per-engine eval_num_threads over process-global
        // omp_get_max_threads() so concurrent runs do not observe each other.
        int requested_threads = (config_.eval_num_threads > 0)
            ? config_.eval_num_threads
            : std::max(1, omp_get_max_threads());
        int outer_threads = std::min(config_.num_islands, requested_threads);
        int inner_threads = std::max(1, requested_threads / std::max(1, outer_threads));
        last_island_outer_threads_ = outer_threads;
        last_island_inner_threads_ = inner_threads;
        for (auto& island : islands) {
            island.set_eval_num_threads(inner_threads);
            island.set_arithmetic_temperature_config(config_.arithmetic_temperature);
        }
#if defined(_OPENMP)
        // Prefer active levels over deprecated omp_set_nested when available.
        // RAII + process mutex: exception-safe and race-free (FC-1/M-323).
        MaxActiveLevelsGuard active_levels_guard(2);
#endif
        std::atomic<bool> init_timed_out(false);
        #pragma omp parallel for schedule(dynamic) num_threads(outer_threads)
        for (int i = 0; i < static_cast<int>(islands.size()); ++i) {
            auto& island = islands[i];
            island.initialize_population();
            for (auto& ind : island.population_) {
                if (init_timed_out.load(std::memory_order_relaxed) || timed_out()) {
                    init_timed_out.store(true, std::memory_order_relaxed);
                    break;
                }
                island.refine_constants(ind);
            }
            if (init_timed_out.load(std::memory_order_relaxed)) {
                island.evaluate_population();
            }
        }

        // Run generations with periodic migration
        for (int gen = 0; gen < config_.generations; ++gen) {
            if (init_timed_out.load(std::memory_order_relaxed) || timed_out()) {
                break;
            }

            // Evolve islands independently in parallel. Each island uses a
            // bounded inner OpenMP team for population evaluation (eval_num_threads).
            #pragma omp parallel for schedule(dynamic) num_threads(outer_threads)
            for (int i = 0; i < static_cast<int>(islands.size()); ++i) {
                islands[i].evolve_one_generation(gen);
            }

            // Migration: ring topology (island i -> island i+1)
            if (gen > 0 && gen % config_.migration_interval == 0) {
                for (int i = 0; i < config_.num_islands; ++i) {
                    int next = (i + 1) % config_.num_islands;
                    auto& src = islands[i].population_;
                    auto& dst = islands[next].population_;

                    // Sort source by fitness (raw_mse tie-break) to get top elites
                    std::sort(src.begin(), src.end(),
                              [](const IndividualGraph& a, const IndividualGraph& b) {
                                  return is_better_champion(a, b);
                              });

                    // Replace worst in destination with source elites
                    std::sort(dst.begin(), dst.end(),
                              [](const IndividualGraph& a, const IndividualGraph& b) {
                                  return is_better_champion(a, b);
                              });

                    int n_migrate = std::min(config_.migration_size, static_cast<int>(src.size()) / 2);
                    for (int m = 0; m < n_migrate; ++m) {
                        dst[dst.size() - 1 - m] = src[m];
                    }
                }
            }

            // Check early stop across all islands
            bool should_stop = false;
            for (auto& island : islands) {
                auto best = island.get_best();
                if (early_stop_metric(best) < config_.early_stop_mse && best.nodes.size() <= static_cast<size_t>(config_.early_stop_max_nodes)) {
                    should_stop = true;
                }
                consider_champion(best);
            }
            update_discovery_metrics(gen, start_time);

            if (config_.use_early_stop && should_stop) break;
        }
        // MaxActiveLevelsGuard destructor restores previous levels here
        // (exception-safe; replaces manual omp_set_max_active_levels restore).

        // Collect the best overall across all islands and run cleanup
        for (auto& island : islands) {
            auto best = island.get_best();
            consider_champion(best);
            crossover_attempts_ += island.get_crossover_attempts();
            crossover_successes_ += island.get_crossover_successes();
            last_crossover_valid_ = island.get_last_crossover_valid();
            // P-04 diagnostics: aggregate island FD probe counters.
            fd_probes_total_ += island.get_fd_probes_total();
            fd_probes_skipped_inert_ += island.get_fd_probes_skipped_inert();
            // P-01 diagnostics: sum island cache pressure into the parent engine.
            gen_cache_.add_evictions(island.get_subtree_cache_evictions());
        }

        // Merge all island populations for Pareto front (if NSGA-II)
        population_.clear();
        for (auto& island : islands) {
            for (auto& ind : island.population_) {
                population_.push_back(std::move(ind));
            }
        }

        cleanup_graph(best_overall_);
        if (!best_raw_overall_.nodes.empty()) {
            cleanup_graph(best_raw_overall_);
        }
        update_discovery_metrics(config_.generations, start_time);
        run_wall_time_sec_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
    }

private:
    EvolutionConfig config_;
    std::vector<Eigen::ArrayXd> X_;
    Eigen::ArrayXd y_;
    Eigen::ArrayXd y_weights_; // empty => uniform; else length == y_.size()
    bool has_y_weights_ = false;
    double y_weight_sum_ = 0.0;
    std::vector<double> seed_omegas_;
    std::vector<IndividualGraph> seed_graphs_;
    
    std::vector<IndividualGraph> population_;
    IndividualGraph best_overall_;
    IndividualGraph best_raw_overall_; // dual archive by raw_mse (E5)
    SubtreeCache gen_cache_; // Per-generation subtree cache
    std::vector<double> op_cdf_; // CDF for prior-weighted op sampling
    std::vector<double> binary_op_cdf_; // CDF for prior-weighted binary-op sampling
    std::vector<int> allowed_unary_ops_;
    std::vector<int> allowed_binary_ops_;
    
    std::mt19937 rng_;
    std::ofstream trace_stream_;
    bool trace_enabled_ = false;
    bool last_crossover_valid_ = false;
    int crossover_attempts_ = 0;
    int crossover_successes_ = 0;
    // P-04 diagnostics: FD probe volume + inert-param probes avoided.
    // Plain counters: each island engine is confined to one OpenMP thread
    // (parallel-for over islands) and the parent aggregates post-join.
    // NOTE: must stay copyable/movable — islands.reserve() moves engines.
    long long fd_probes_total_ = 0;
    long long fd_probes_skipped_inert_ = 0;
    int first_exact_generation_ = -1;
    double first_exact_time_sec_ = -1.0;
    int first_acceptable_generation_ = -1;
    double first_acceptable_time_sec_ = -1.0;
    int last_island_outer_threads_ = 1;
    int last_island_inner_threads_ = 1;

    double current_structural_mutation_rate_ = 0.05;
    double best_mse_history_ = 1e9;
    int plateau_counter_ = 0;
    std::vector<double> recent_best_;

    void set_y_weights(const Eigen::ArrayXd& y_weights) {
        has_y_weights_ = false;
        y_weights_ = Eigen::ArrayXd();
        y_weight_sum_ = static_cast<double>(y_.size());
        if (y_weights.size() == 0) return;
        if (y_weights.size() != y_.size()) {
            throw std::runtime_error("y_weights must be 1D with length matching y");
        }
        for (int i = 0; i < static_cast<int>(y_weights.size()); ++i) {
            if (!std::isfinite(y_weights(i)) || y_weights(i) < 0.0) {
                throw std::runtime_error("y_weights must be finite and non-negative");
            }
        }
        double total = y_weights.sum();
        if (!(total > 0.0) || !std::isfinite(total)) {
            throw std::runtime_error("y_weights must have positive total weight");
        }
        y_weights_ = y_weights;
        y_weight_sum_ = total;
        has_y_weights_ = true;
    }

    // Objective used for selection/fitness (weighted and/or robust).
    double objective_mse(const IndividualGraph& ind) const {
        // weighted_mse holds the search objective after evaluate_fitness_with_penalty
        // (plain weighted MSE, huber, trimmed, or student_t). Fall back to raw_mse
        // only if it was never evaluated.
        if (std::isfinite(ind.weighted_mse) && ind.weighted_mse < 1e90) {
            return ind.weighted_mse;
        }
        return ind.raw_mse;
    }

    // Early-stop / "exact" claims must use unweighted raw MSE so robust search
    // losses cannot declare success while true MSE is still large (E2/N5).
    double early_stop_metric(const IndividualGraph& ind) const {
        if (std::isfinite(ind.raw_mse) && ind.raw_mse < 1e90) {
            return ind.raw_mse;
        }
        return objective_mse(ind);
    }

    // Weighted median of vals using optional per-point weights (same length).
    // Falls back to unweighted median when weights are absent/mismatched.
    // §3.133: unweighted fallback averages the two middle values for even n
    // (NumPy convention); the old lower-median pick ([1,2,3,4] -> 3 vs 2.5)
    // silently disagreed with every NumPy-side statistic.
    static double unweighted_median_of(std::vector<double>& vals) {
        if (vals.empty()) return 0.0;
        std::sort(vals.begin(), vals.end());
        const size_t n = vals.size();
        if (n % 2 == 1) return vals[n / 2];
        return 0.5 * (vals[n / 2 - 1] + vals[n / 2]);
    }
    static double weighted_median_of(std::vector<double>& vals,
                                     const Eigen::ArrayXd* weights,
                                     const std::vector<int>* weight_idx) {
        if (vals.empty()) return 0.0;
        if (weights == nullptr || weight_idx == nullptr ||
            weight_idx->size() != vals.size()) {
            return unweighted_median_of(vals);
        }
        std::vector<size_t> order(vals.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::sort(order.begin(), order.end(),
                  [&](size_t a, size_t b) { return vals[a] < vals[b]; });
        double total = 0.0;
        for (size_t k = 0; k < order.size(); ++k) {
            const int wi = (*weight_idx)[order[k]];
            double w = (wi >= 0 && wi < static_cast<int>(weights->size()))
                           ? (*weights)(wi) : 0.0;
            if (!std::isfinite(w) || w < 0.0) w = 0.0;
            total += w;
        }
        if (!(total > 0.0)) {
            return unweighted_median_of(vals);
        }
        double half = 0.5 * total;
        double cum = 0.0;
        for (size_t k = 0; k < order.size(); ++k) {
            const int wi = (*weight_idx)[order[k]];
            double w = (wi >= 0 && wi < static_cast<int>(weights->size()))
                           ? (*weights)(wi) : 0.0;
            if (!std::isfinite(w) || w < 0.0) w = 0.0;
            cum += w;
            if (cum >= half) return vals[order[k]];
        }
        return vals[order.back()];
    }

    // Robust residual scale via MAD (≈ σ for Gaussian).
    // N7: when y_weights_ are set, use weighted median/MAD to match Python _mad_scale.
    double mad_scale(const Eigen::ArrayXd& resid) const {
        const int n = static_cast<int>(resid.size());
        if (n <= 0) return 1.0;
        std::vector<double> vals;
        std::vector<int> idx;
        vals.reserve(static_cast<size_t>(n));
        idx.reserve(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            if (std::isfinite(resid(i))) {
                vals.push_back(resid(i));
                idx.push_back(i);
            }
        }
        if (vals.empty()) return 1.0;

        const bool use_w = has_y_weights_ && y_weights_.size() == resid.size();
        const Eigen::ArrayXd* wptr = use_w ? &y_weights_ : nullptr;
        const std::vector<int>* iptr = use_w ? &idx : nullptr;

        // Copy for median; weighted_median_of may reorder via index sort.
        std::vector<double> vals_for_med = vals;
        double med = weighted_median_of(vals_for_med, wptr, iptr);

        std::vector<double> abs_dev;
        abs_dev.reserve(vals.size());
        for (double v : vals) abs_dev.push_back(std::abs(v - med));
        double mad = weighted_median_of(abs_dev, wptr, iptr);
        double scale = 1.4826 * mad;
        if (!std::isfinite(scale) || scale < 1e-12) {
            // Weighted RMSE fallback when MAD collapses.
            if (use_w) {
                double acc = 0.0, wsum = 0.0;
                for (size_t k = 0; k < vals.size(); ++k) {
                    double w = y_weights_(idx[k]);
                    if (!std::isfinite(w) || w < 0.0) continue;
                    acc += w * vals[k] * vals[k];
                    wsum += w;
                }
                if (wsum > 0.0) {
                    scale = std::sqrt(acc / wsum);
                }
            } else {
                double acc = 0.0;
                int c = 0;
                for (int i = 0; i < n; ++i) {
                    if (std::isfinite(resid(i))) {
                        acc += resid(i) * resid(i);
                        ++c;
                    }
                }
                if (c > 0) {
                    scale = std::sqrt(acc / static_cast<double>(c));
                }
            }
            if (!std::isfinite(scale) || scale < 1e-12) scale = 1.0;
        }
        return scale;
    }

    double residual_mse(const Eigen::ArrayXd& pred, const Eigen::ArrayXd& y) const {
        const int n = static_cast<int>(pred.size());
        if (n <= 0 || pred.size() != y.size()) {
            return std::numeric_limits<double>::infinity();
        }
        // H-01: reject non-finite predictions before any loss aggregation so
        // NaN never enters fitness / std::sort comparisons.
        if (!pred.isFinite().all() || !y.isFinite().all()) {
            return std::numeric_limits<double>::infinity();
        }
        Eigen::ArrayXd resid = pred - y;

        // Plain (optionally weighted) MSE path - also used for diagnostics.
        auto weighted_mean_of = [&](const Eigen::ArrayXd& vals) -> double {
            if (has_y_weights_ && y_weights_.size() == vals.size()) {
                const double m = (y_weights_ * vals).sum() / y_weight_sum_;
                return std::isfinite(m) ? m : std::numeric_limits<double>::infinity();
            }
            const double m = vals.mean();
            return std::isfinite(m) ? m : std::numeric_limits<double>::infinity();
        };

        if (config_.loss_mode == LossMode::Mse) {
            return weighted_mean_of(resid.square());
        }

        if (config_.loss_mode == LossMode::Huber) {
            double d = config_.huber_delta;
            if (!(d > 0.0) || !std::isfinite(d)) d = mad_scale(resid);
            d = std::max(d, 1e-12);
            Eigen::ArrayXd loss(n);
            for (int i = 0; i < n; ++i) {
                double a = std::abs(resid(i));
                if (a <= d) loss(i) = 0.5 * resid(i) * resid(i);
                else loss(i) = d * (a - 0.5 * d);
            }
            return weighted_mean_of(loss);
        }

        if (config_.loss_mode == LossMode::TrimmedMse) {
            Eigen::ArrayXd sq = resid.square();
            double frac = std::clamp(config_.trim_fraction, 0.0, 0.45);
            std::vector<int> order(static_cast<size_t>(n));
            for (int i = 0; i < n; ++i) order[static_cast<size_t>(i)] = i;
            // §3.135: trim by WEIGHT MASS, not by fixed residual count. The
            // old code kept round(n*(1-frac)) rows even with sample weights,
            // so zero-weight rows consumed the keep budget while heavy rows
            // could be trimmed. Unweighted path keeps the count behavior.
            if (has_y_weights_ && y_weights_.size() == n) {
                std::sort(order.begin(), order.end(),
                    [&](int a, int b) { return sq(a) < sq(b); });
                double w_total = 0.0;
                for (int i = 0; i < n; ++i) {
                    double w = y_weights_(i);
                    if (std::isfinite(w) && w > 0.0) w_total += w;
                }
                if (!(w_total > 0.0)) {
                    return std::numeric_limits<double>::infinity();
                }
                const double keep_mass = w_total * (1.0 - frac);
                double num = 0.0, den = 0.0;
                for (int k = 0; k < n; ++k) {
                    int i = order[static_cast<size_t>(k)];
                    double w = y_weights_(i);
                    if (!std::isfinite(w) || w <= 0.0) continue;
                    if (den + w > keep_mass && den > 0.0) break;
                    num += w * sq(i);
                    den += w;
                }
                return (den > 0.0) ? (num / den) : std::numeric_limits<double>::infinity();
            }
            int keep = std::max(1, static_cast<int>(std::llround(n * (1.0 - frac))));
            std::partial_sort(order.begin(), order.begin() + keep, order.end(),
                [&](int a, int b) { return sq(a) < sq(b); });
            double acc = 0.0;
            for (int k = 0; k < keep; ++k) acc += sq(order[static_cast<size_t>(k)]);
            return acc / static_cast<double>(keep);
        }

        // Student-t style heavy-tail loss: log(1 + (r/s)^2)
        double s = config_.huber_delta;
        if (!(s > 0.0) || !std::isfinite(s)) s = mad_scale(resid);
        s = std::max(s, 1e-12);
        Eigen::ArrayXd loss = (resid / s).square().log1p();
        return weighted_mean_of(loss);
    }

    double residual_mse_unweighted(const Eigen::ArrayXd& pred, const Eigen::ArrayXd& y) const {
        // Always plain MSE for raw_mse diagnostics / back-compat.
        // H-01: non-finite predictions must not yield NaN (ill-defined sort).
        if (pred.size() != y.size() || pred.size() == 0) {
            return std::numeric_limits<double>::infinity();
        }
        if (!pred.isFinite().all() || !y.isFinite().all()) {
            return std::numeric_limits<double>::infinity();
        }
        const double mse = (pred - y).square().mean();
        return std::isfinite(mse) ? mse : std::numeric_limits<double>::infinity();
    }

    static void normalize_prior_vector(std::vector<double>& priors) {
        double sum = 0.0;
        for (double& p : priors) {
            if (!std::isfinite(p) || p < 0.0) {
                p = 0.0;
            }
            sum += p;
        }
        if (sum > 0.0) {
            for (double& p : priors) {
                p /= sum;
            }
        }
    }

    static std::vector<double> build_cdf(const std::vector<double>& priors) {
        std::vector<double> cdf(priors.size(), 0.0);
        if (priors.empty()) {
            return cdf;
        }
        cdf[0] = priors[0];
        for (size_t i = 1; i < priors.size(); ++i) {
            cdf[i] = cdf[i - 1] + priors[i];
        }
        cdf.back() = 1.0;
        return cdf;
    }

    template <typename EnumT>
    static std::vector<int> sanitize_allowed_ops(const std::vector<int>& raw, int max_value) {
        std::vector<int> allowed;
        allowed.reserve(raw.size());
        for (int v : raw) {
            if (v >= 0 && v <= max_value && std::find(allowed.begin(), allowed.end(), v) == allowed.end()) {
                allowed.push_back(v);
            }
        }
        return allowed;
    }

    bool unary_op_allowed(UnaryOp op) const {
        if (allowed_unary_ops_.empty()) return true;
        int value = static_cast<int>(op);
        return std::find(allowed_unary_ops_.begin(), allowed_unary_ops_.end(), value) != allowed_unary_ops_.end();
    }

    bool binary_op_allowed(BinaryOp op) const {
        if (allowed_binary_ops_.empty()) return true;
        int value = static_cast<int>(op);
        return std::find(allowed_binary_ops_.begin(), allowed_binary_ops_.end(), value) != allowed_binary_ops_.end();
    }

    int tournament_select(int k = 5) {
        if (population_.empty()) return 0;
        std::uniform_int_distribution<int> dist(0, static_cast<int>(population_.size()) - 1);
        int best_idx = dist(rng_);
        for (int i = 1; i < k; ++i) {
            int idx = dist(rng_);
            if (is_better_champion(population_[idx], population_[best_idx])) {
                best_idx = idx;
            }
        }
        return best_idx;
    }
    double run_wall_time_sec_ = 0.0;

    void update_discovery_metrics(int generation, const std::chrono::steady_clock::time_point& start_time) {
        const bool is_exact = (early_stop_metric(best_overall_) < config_.early_stop_mse && best_overall_.nodes.size() <= static_cast<size_t>(config_.early_stop_max_nodes));
        // Acceptable band also uses raw MSE so robust objectives do not over-claim.
        const bool is_acceptable =
            (early_stop_metric(best_overall_) < config_.acceptable_mse &&
             static_cast<int>(best_overall_.nodes.size()) <= config_.acceptable_complexity);

        if (is_exact && first_exact_generation_ < 0) {
            first_exact_generation_ = generation;
            first_exact_time_sec_ = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start_time
            ).count();
        }
        if (is_acceptable && first_acceptable_generation_ < 0) {
            first_acceptable_generation_ = generation;
            first_acceptable_time_sec_ = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start_time
            ).count();
        }
    }

    static std::string json_escape(const std::string& s) {
        std::string out;
        out.reserve(s.size());
        for (char c : s) {
            switch (c) {
                case '"': out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\b': out += "\\b"; break;
                case '\f': out += "\\f"; break;
                case '\n': out += "\\n"; break;
                case '\r': out += "\\r"; break;
                case '\t': out += "\\t"; break;
                default:
                    if (static_cast<unsigned char>(c) < 0x20) out += "?";
                    else out += c;
            }
        }
        return out;
    }

    void trace_event(const char* event, int generation) {
        if (!trace_enabled_) return;

        trace_stream_ << "{\"event\":\"" << event << "\"";
        if (generation >= 0) trace_stream_ << ",\"generation\":" << generation;

        if (!population_.empty()) {
            const auto best_it = std::min_element(
                population_.begin(), population_.end(),
                [](const IndividualGraph& a, const IndividualGraph& b) { return a.fitness < b.fitness; }
            );
            trace_stream_ << ",\"best_fitness\":" << best_it->fitness;
            trace_stream_ << ",\"best_raw_mse\":" << best_it->raw_mse;
            trace_stream_ << ",\"best_complexity\":" << best_it->complexity();
            trace_stream_ << ",\"best_nodes\":" << best_it->nodes.size();
        }

        trace_stream_ << ",\"population\":[";
        for (size_t i = 0; i < population_.size(); ++i) {
            const auto& ind = population_[i];
            if (i) trace_stream_ << ",";
            trace_stream_ << "{\"idx\":" << i
                          << ",\"fitness\":" << ind.fitness
                          << ",\"raw_mse\":" << ind.raw_mse
                          << ",\"complexity\":" << ind.complexity()
                          << ",\"nodes\":" << ind.nodes.size()
                          << ",\"age\":" << ind.age;
            if (config_.trace_include_formulas) {
                std::string formula = get_formula_string(ind, static_cast<int>(X_.size()));
                trace_stream_ << ",\"formula\":\"" << json_escape(formula) << "\"";
            }
            trace_stream_ << "}";
        }
        trace_stream_ << "]";

        trace_stream_ << ",\"best_overall_fitness\":" << best_overall_.fitness;
        trace_stream_ << ",\"best_overall_raw_mse\":" << best_overall_.raw_mse;
        if (config_.trace_include_formulas && !best_overall_.nodes.empty()) {
            std::string best_formula = get_formula_string(best_overall_, static_cast<int>(X_.size()));
            trace_stream_ << ",\"best_overall_formula\":\"" << json_escape(best_formula) << "\"";
        }
        trace_stream_ << "}\n";
        trace_stream_.flush();
    }

    void sanitize_config() {
        config_.pop_size = std::max(1, config_.pop_size);
        config_.elite_size = std::max(1, std::min(config_.elite_size, config_.pop_size));
        config_.num_islands = std::max(1, config_.num_islands);
        config_.max_nodes = std::max(4, config_.max_nodes);
        config_.migration_size = std::max(1, config_.migration_size);
        config_.topology_phase_generations = std::max(0, config_.topology_phase_generations);
        config_.topology_refine_interval = std::max(1, config_.topology_refine_interval);
        config_.stagnation_window = std::max(5, config_.stagnation_window);
        config_.diversity_floor = std::clamp(config_.diversity_floor, 0.0, 1.0);
        config_.restart_fraction = std::clamp(config_.restart_fraction, 0.0, 0.8);
        config_.topology_phase_mutation_boost = std::max(1.0, config_.topology_phase_mutation_boost);
        config_.post_restart_mutation_boost = std::max(1.0, config_.post_restart_mutation_boost);
        config_.lm_max_iterations = std::max(1, config_.lm_max_iterations);
        config_.lm_lambda_init = std::max(1e-8, config_.lm_lambda_init);
    }
    
    // Sample a UnaryOp using classifier priors (if available) or uniform
    UnaryOp sample_unary_op() {
        std::uniform_real_distribution<double> u(0.0, 1.0);
        if (!op_cdf_.empty()) {
            for (int attempt = 0; attempt < 16; ++attempt) {
                double r = u(rng_);
                for (size_t i = 0; i < op_cdf_.size(); ++i) {
                    UnaryOp op = static_cast<UnaryOp>(i);
                    if (r <= op_cdf_[i] && unary_op_allowed(op)) return op;
                }
            }
            for (int op = 0; op <= static_cast<int>(UnaryOp::Abs); ++op) {
                if (unary_op_allowed(static_cast<UnaryOp>(op))) {
                    return static_cast<UnaryOp>(op);
                }
            }
            return UnaryOp::Log;
        }
        // Uniform: 0=Periodic, 1=Power, 2=IntPow, 3=Exp, 4=Log
        const UnaryOp defaults[] = {
            UnaryOp::Periodic, UnaryOp::Power, UnaryOp::IntPow, UnaryOp::Exp, UnaryOp::Log
        };
        for (int attempt = 0; attempt < 16; ++attempt) {
            double op_choice = u(rng_);
            UnaryOp op = UnaryOp::Log;
            if (op_choice < 0.25) op = UnaryOp::Periodic;
            else if (op_choice < 0.50) op = UnaryOp::Power;
            else if (op_choice < 0.70) op = UnaryOp::IntPow;
            else if (op_choice < 0.85) op = UnaryOp::Exp;
            if (unary_op_allowed(op)) return op;
        }
        for (UnaryOp op : defaults) {
            if (unary_op_allowed(op)) return op;
        }
        return UnaryOp::Log;
    }

    BinaryOp sample_binary_op() {
        std::uniform_real_distribution<double> u(0.0, 1.0);
        if (!binary_op_cdf_.empty()) {
            for (int attempt = 0; attempt < 16; ++attempt) {
                double r = u(rng_);
                for (size_t i = 0; i < binary_op_cdf_.size(); ++i) {
                    BinaryOp op = static_cast<BinaryOp>(i);
                    if (r <= binary_op_cdf_[i] && binary_op_allowed(op)) return op;
                }
            }
            for (int op = 0; op <= static_cast<int>(BinaryOp::Aggregation); ++op) {
                if (binary_op_allowed(static_cast<BinaryOp>(op))) {
                    return static_cast<BinaryOp>(op);
                }
            }
            return BinaryOp::Aggregation;
        }
        const BinaryOp defaults[] = {
            BinaryOp::Arithmetic, BinaryOp::Division, BinaryOp::Aggregation
        };
        for (int attempt = 0; attempt < 16; ++attempt) {
            double r = u(rng_);
            BinaryOp op = BinaryOp::Aggregation;
            if (r < 0.45) op = BinaryOp::Arithmetic;
            else if (r < 0.75) op = BinaryOp::Division;
            if (binary_op_allowed(op)) return op;
        }
        for (BinaryOp op : defaults) {
            if (binary_op_allowed(op)) return op;
        }
        return BinaryOp::Aggregation;
    }

    bool unary_wrap_allowed(const IndividualGraph& graph, int child_idx, UnaryOp op) const {
        if (child_idx < 0 || child_idx >= static_cast<int>(graph.nodes.size())) {
            return true;
        }
        const auto& child = graph.nodes[child_idx];
        if (child.type != NodeType::Unary) {
            return true;
        }
        UnaryOp child_op = child.unary_op;
        if (op == UnaryOp::Periodic && child_op == UnaryOp::Periodic) {
            return false;
        }
        if (op == UnaryOp::Log && (child_op == UnaryOp::Periodic || child_op == UnaryOp::Exp || child_op == UnaryOp::Log)) {
            return false;
        }
        if (op == UnaryOp::Exp && (child_op == UnaryOp::Exp || child_op == UnaryOp::Log)) {
            return false;
        }
        if (op == UnaryOp::Abs && child_op == UnaryOp::Abs) {
            return false;
        }
        if ((op == UnaryOp::Power || op == UnaryOp::IntPow) &&
            (child_op == UnaryOp::Power || child_op == UnaryOp::IntPow || child_op == UnaryOp::Exp)) {
            return false;
        }
        return true;
    }

    UnaryOp sample_unary_op_for_child(const IndividualGraph& graph, int child_idx) {
        for (int attempt = 0; attempt < 24; ++attempt) {
            UnaryOp op = sample_unary_op();
            if (unary_wrap_allowed(graph, child_idx, op)) {
                return op;
            }
        }
        const UnaryOp fallbacks[] = {
            UnaryOp::Periodic, UnaryOp::IntPow, UnaryOp::Power, UnaryOp::Exp, UnaryOp::Log
        };
        for (UnaryOp op : fallbacks) {
            if (unary_op_allowed(op) && unary_wrap_allowed(graph, child_idx, op)) {
                return op;
            }
        }
        return sample_unary_op();
    }

    void seed_arithmetic_gate(OpNode& node) {
        std::uniform_int_distribution<int> mode_dist(0, 3);
        switch (mode_dist(rng_)) {
            case 0:
                node.beta = 1.0;
                node.gamma = 1.0;
                break;
            case 1:
                node.beta = 2.0;
                node.gamma = 1.0;
                break;
            case 2:
                node.beta = 2.0;
                node.gamma = -1.0;
                break;
            default:
                node.beta = 1.0;
                node.gamma = -1.0;
                break;
        }
        node.tau = 1.0;
    }

    // H-03: keep unary inner params in safe ranges after Adam/LM/mutation.
    // Exp uses tighter omega/phi so omega*x+phi cannot explode into Inf.
    void clamp_unary_inner_params(OpNode& node) const {
        node.p = std::clamp(node.p, config_.p_min, config_.p_max);
        if (node.type == NodeType::Unary && node.unary_op == UnaryOp::Exp) {
            node.omega = std::clamp(node.omega, config_.exp_omega_min, config_.exp_omega_max);
            node.phi = std::clamp(node.phi, config_.exp_phi_min, config_.exp_phi_max);
        } else {
            node.omega = std::clamp(node.omega, config_.omega_min, config_.omega_max);
            node.phi = std::clamp(node.phi, config_.phi_min, config_.phi_max);
        }
        if (!std::isfinite(node.p)) node.p = 1.0;
        if (!std::isfinite(node.omega)) node.omega = 1.0;
        if (!std::isfinite(node.phi)) node.phi = 0.0;
    }
    
    IndividualGraph create_random_individual(int n_inputs) {
        IndividualGraph ind;
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
                std::uniform_int_distribution<int> seed_dist(0, static_cast<int>(seed_omegas_.size()) - 1);
                node.omega = seed_omegas_[seed_dist(rng_)];
            }

            node.phi = rnorm(rng_);
            node.amplitude = 1.0;  // Fixed - SVD handles scaling via output_weights
            node.beta = 1.5 + rnorm(rng_)*0.5;
            node.gamma = 1.0 + rnorm(rng_)*0.5;
            node.tau = 1.0;

            // -- FIX: Only node 0 is Input. All others are Unary/Binary. --
            // This prevents multiple collinear 'x' columns in the SVD.
            if (i < n_inputs) {
                node.type = NodeType::Input;
                node.feature_idx = i;
            } else {
                if (runif(rng_) < 0.6 || i < n_inputs + 1) {
                    node.type = NodeType::Unary;
                    std::uniform_int_distribution<int> child_dist(0, i - 1);
                    node.left_child = child_dist(rng_);
                    node.unary_op = sample_unary_op_for_child(ind, node.left_child);
                    if (node.unary_op == UnaryOp::Power && runif(rng_) < 0.7) {
                        const double power_candidates[] = {-1.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0};
                        std::vector<double> valid_powers;
                        for (double candidate : power_candidates) {
                            if (candidate >= config_.p_min && candidate <= config_.p_max) {
                                valid_powers.push_back(candidate);
                            }
                        }
                        if (!valid_powers.empty()) {
                            std::uniform_int_distribution<int> p_dist(0, static_cast<int>(valid_powers.size()) - 1);
                            node.p = valid_powers[p_dist(rng_)];
                        }
                    }
                    if (node.unary_op == UnaryOp::IntPow) {
                        const int intpow_candidates[] = {2, 3, 4, 5, 6};
                        std::uniform_int_distribution<int> ip_dist(0, 4);
                        node.p = static_cast<double>(intpow_candidates[ip_dist(rng_)]);
                    }
                    // Seed Exp nodes with useful omega values (including negative for exp(-x))
                    if (node.unary_op == UnaryOp::Exp) {
                        const double exp_omega_seeds[] = {-2.0, -1.0, -0.5, 0.5, 1.0, 2.0};
                        std::uniform_int_distribution<int> eo_dist(0, 5);
                        node.omega = exp_omega_seeds[eo_dist(rng_)];
                        node.phi = 0.0; // Exp shift is rarely needed at init
                    }
                } else {
                    node.type = NodeType::Binary;
                    node.binary_op = sample_binary_op();

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
        return ind;
    }

    // E3: seed capacity - allow up to seed_fraction of pop (default 50%),
    // and on tiny island pops keep almost all slots for seeds while leaving
    // at least one random individual when pop > 1.
    int max_seed_capacity() const {
        const int pop = std::max(1, config_.pop_size);
        const int n_seeds = static_cast<int>(seed_graphs_.size());
        if (n_seeds <= 0) return 0;
        double frac = config_.seed_fraction;
        if (!(frac > 0.0) || !std::isfinite(frac)) frac = 0.5;
        frac = std::clamp(frac, 0.1, 0.9);
        int by_frac = std::max(1, static_cast<int>(std::ceil(frac * static_cast<double>(pop))));
        int cap;
        if (pop <= 12) {
            // Small island/pop: avoid historical pop/4 starvation (e.g. 12->3).
            cap = std::max(by_frac, pop - 1);
        } else {
            cap = by_frac;
        }
        cap = std::min(cap, pop);
        return std::min(cap, n_seeds);
    }

    // E5/E7(tracker): champion compare - lower penalized fitness wins; on near-ties
    // prefer better unweighted raw_mse (export/protocol accuracy).
    static bool is_better_champion(const IndividualGraph& cand, const IndividualGraph& best) {
        constexpr double kFitEps = 1e-12;
        if (!std::isfinite(cand.fitness)) return false;
        if (!std::isfinite(best.fitness)) return true;
        if (cand.fitness < best.fitness - kFitEps) return true;
        if (cand.fitness > best.fitness + kFitEps) return false;
        // Fitness tie (or numerically equal): prefer raw_mse, then fewer nodes.
        if (std::isfinite(cand.raw_mse) && std::isfinite(best.raw_mse)) {
            if (cand.raw_mse < best.raw_mse - kFitEps) return true;
            if (cand.raw_mse > best.raw_mse + kFitEps) return false;
        }
        return cand.nodes.size() < best.nodes.size();
    }

    void consider_champion(const IndividualGraph& cand) {
        if (cand.nodes.empty()) return;
        if (best_overall_.nodes.empty() || is_better_champion(cand, best_overall_)) {
            best_overall_ = cand;
        }
        // Dual archive: best plain MSE regardless of complexity penalty (export aid).
        if (best_raw_overall_.nodes.empty()
            || (std::isfinite(cand.raw_mse)
                && (!std::isfinite(best_raw_overall_.raw_mse)
                    || cand.raw_mse < best_raw_overall_.raw_mse))) {
            best_raw_overall_ = cand;
        }
    }

    // Prefer fitness champion; if dual raw archive is much better on raw_mse and
    // not much worse on fitness, export the raw champion (protocol accuracy).
    IndividualGraph select_export_champion() const {
        if (best_raw_overall_.nodes.empty()) return best_overall_;
        if (best_overall_.nodes.empty()) return best_raw_overall_;
        constexpr double kFitSlack = 0.05; // 5% fitness slack
        const double fit_lim = best_overall_.fitness * (1.0 + kFitSlack) + 1e-12;
        if (std::isfinite(best_raw_overall_.fitness)
            && best_raw_overall_.fitness <= fit_lim
            && std::isfinite(best_raw_overall_.raw_mse)
            && std::isfinite(best_overall_.raw_mse)
            && best_raw_overall_.raw_mse < best_overall_.raw_mse - 1e-15) {
            return best_raw_overall_;
        }
        return best_overall_;
    }

    void initialize_population() {
        current_structural_mutation_rate_ = config_.mutation_rate_structural;
        best_mse_history_ = 1e9;
        plateau_counter_ = 0;
        recent_best_.clear();
        recent_best_.reserve(config_.stagnation_window + 2);
        population_.resize(config_.pop_size);
        int n_inputs = static_cast<int>(X_.size());
        
        int seeded = 0;
        // E3: seed first max_seed graphs (capacity raised vs historical pop/4).
        int max_seed = max_seed_capacity();
        // Light shuffle of seed order under RNG so unused tails rotate across runs
        // when seeds > capacity (single-pop). Islands already shard seed lists.
        std::vector<int> seed_order(static_cast<size_t>(seed_graphs_.size()));
        for (int i = 0; i < static_cast<int>(seed_graphs_.size()); ++i) seed_order[static_cast<size_t>(i)] = i;
        if (seed_order.size() > 1) {
            std::shuffle(seed_order.begin(), seed_order.end(), rng_);
        }
        for (int i = 0; i < max_seed; ++i) {
            population_[i] = seed_graphs_[static_cast<size_t>(seed_order[static_cast<size_t>(i)])];
            seeded++;
        }
        
        // Fill rest with random individuals
        for (int i = seeded; i < config_.pop_size; ++i) {
            population_[i] = create_random_individual(n_inputs);
        }
    }
    
    void evaluate_population() {
        int samples = static_cast<int>(y_.size());
        // Engine does not own X/y; eval lambda closes over EvolutionEngine data.
        ParallelExecutionEngine executor;

        // E6: skip re-eval of individuals already scored (elites carried over,
        // children scored at birth). Mutate/crossover clear fitness_valid.
        executor.evaluate_population(population_, [&](IndividualGraph& ind, SubtreeCache& tc) {
            if (ind.fitness_valid) return;
            evaluate_fitness_with_penalty(ind, X_, y_, samples, &tc);
        }, config_.eval_num_threads);

        // Pull the global merged cache out of executor
        gen_cache_ = executor.take_gen_cache();
    }

    uint64_t graph_signature(const IndividualGraph& graph) const {
        if (graph.nodes.empty()) return 0ULL;
        std::vector<uint64_t> node_hashes(graph.nodes.size(), 0ULL);
        for (int i = 0; i < static_cast<int>(graph.nodes.size()); ++i) {
            node_hashes[i] = compute_node_hash(graph, i, node_hashes);
        }

        uint64_t h = node_hashes.back();
        for (double w : graph.output_weights) {
            h = hash_combine(h, quantize(w));
        }
        h = hash_combine(h, quantize(graph.output_bias));
        return h;
    }

    double population_diversity_ratio(const std::vector<IndividualGraph>& pop) const {
        if (pop.empty()) return 0.0;
        std::unordered_set<uint64_t> signatures;
        signatures.reserve(pop.size());
        for (const auto& ind : pop) {
            signatures.insert(graph_signature(ind));
        }
        return static_cast<double>(signatures.size()) / static_cast<double>(pop.size());
    }

    void inject_restarts(std::vector<IndividualGraph>& pop) {
        int budget = static_cast<int>(std::floor(config_.restart_fraction * static_cast<double>(pop.size())));
        budget = std::max(1, std::min(budget, std::max(0, config_.pop_size - config_.elite_size)));
        if (budget <= 0) return;

        int n_inputs = static_cast<int>(X_.size());
        std::uniform_int_distribution<int> elite_dist(0, std::max(0, config_.elite_size - 1));
        for (int i = 0; i < budget; ++i) {
            int idx = static_cast<int>(pop.size()) - 1 - i;
            if (idx < config_.elite_size || idx < 0) break;

            if (i % 2 == 0 && config_.elite_size > 0) {
                int parent_idx = elite_dist(rng_);
                pop[idx] = macro_mutate(pop[parent_idx]);
            } else {
                // Random immigrant for diversity reset.
                IndividualGraph immigrant = create_random_individual(n_inputs);
                immigrant.age = 0;
                pop[idx] = std::move(immigrant);
            }
        }
    }
    
    // Evaluate fitness including complexity and soft rounding penalty.
    // raw_mse: always plain unweighted MSE (diagnostics / back-compat).
    // weighted_mse: search objective (weights and/or robust loss from residual_mse).
    // fitness: search objective * complexity penalty.
    double evaluate_fitness_with_penalty(IndividualGraph& graph, const std::vector<Eigen::ArrayXd>& X, const Eigen::ArrayXd& y, int num_samples, SubtreeCache* tc = nullptr) {
        // H-06: re-apply engine temperature on every fitness eval so this thread's
        // TLS (and, via set_arithmetic_temperature, the process default for fresh
        // OMP workers) matches config_. Concurrent engines keep independent TLS.
        set_arithmetic_temperature(config_.arithmetic_temperature);
        Eigen::ArrayXd pred;
        if (tc != nullptr) {
            std::vector<Eigen::ArrayXd> cache_out;
            pred = evaluate_graph_cached(graph, X, num_samples, cache_out, *tc);
        } else {
            pred = evaluate_graph_simple(graph, X, num_samples);
        }
        // H-01: non-finite pred/MSE → worst finite fitness so sort stays total-ordered.
        if (pred.size() != y.size() || !pred.isFinite().all()) {
            graph.raw_mse = std::numeric_limits<double>::infinity();
            graph.weighted_mse = std::numeric_limits<double>::infinity();
            graph.fitness = 1e30;
            graph.fitness_valid = true;
            return graph.fitness;
        }
        const double unweighted_mse = residual_mse_unweighted(pred, y);
        const double search_obj = residual_mse(pred, y);
        graph.raw_mse = unweighted_mse;
        graph.weighted_mse = search_obj;
        if (!std::isfinite(unweighted_mse) || !std::isfinite(search_obj)) {
            graph.raw_mse = std::numeric_limits<double>::infinity();
            graph.weighted_mse = std::numeric_limits<double>::infinity();
            graph.fitness = 1e30;
            graph.fitness_valid = true;
            return graph.fitness;
        }
        // Selection always uses search objective (weights + robust loss when set).
        const double mse = search_obj;
        
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

                    // P8: soft anti-trig-bloat (shared predicates with hard ban below).
                    if (is_low_omega_periodic(node)) {
                        penalty += 5.0 * (0.1 - std::abs(node.omega));
                    }
                    if (is_nested_periodic(graph, node)) {
                        penalty += 5.0;
                    }
                }
                // P2: Arithmetic entropy penalty - push soft binary ops toward discrete
                // selection. Lower entropy = more committed to one operation.
                if (node.type == NodeType::Binary && node.binary_op == BinaryOp::Arithmetic) {
                    auto w = arithmetic_soft_weights(node);
                    double entropy = 0.0;
                    for (int k = 0; k < 4; ++k) {
                        if (w[k] > 1e-10) entropy -= w[k] * std::log(w[k]);
                    }
                    penalty += 0.1 * entropy;  // max entropy = ln(4) ≈ 1.39
                }
            }
        }
        
        // Parsimony pressure: penalize active weighted complexity, not just
        // graph length. This keeps dead columns cheap to prune while making
        // risky active operators pay their way.
        int active_complexity = graph.active_complexity();
        // S5-13: inactive count must use node count, not weighted complexity units.
        int inactive_nodes = std::max(0, static_cast<int>(graph.nodes.size()) - graph.active_node_count());
        
        // Scale-invariant parsimony: a graph must improve MSE by roughly 1.2%
        // per active complexity unit to justify added structure.
        double complexity_penalty_factor = 1.2e-2 * active_complexity + 5e-4 * inactive_nodes;
        if (static_cast<int>(graph.nodes.size()) > config_.max_nodes) {
            complexity_penalty_factor += 0.25 * (static_cast<int>(graph.nodes.size()) - config_.max_nodes);
        }

        // E8: under robust loss or sample weights, robust objectives compress outliers
        // so weak extra basis nodes can look cheap. Amplify parsimony so structure
        // must pay a larger MSE improvement (~2.1% per active unit) to survive.
        const bool robust_or_weighted =
            (config_.loss_mode != LossMode::Mse) || has_y_weights_;
        if (robust_or_weighted) {
            complexity_penalty_factor *= 1.75;
        }

        // Relax penalty if we have discovered an exact physical law
        if (mse < 1e-6) {
            complexity_penalty_factor *= 1e-4;
        }

        // Apply multiplicative complexity penalty to guarantee scale invariance
        graph.fitness = mse * (1.0 + complexity_penalty_factor) + config_.round_penalty_weight * penalty / std::max(1.0, static_cast<double>(graph.nodes.size()));

        // P8: hard anti-trig-bloat ban (same predicates as soft penalty above).
        for (const auto& node : graph.nodes) {
            if (is_low_omega_periodic(node)) {
                graph.fitness += 100.0;
            }
            if (is_nested_periodic(graph, node)) {
                graph.fitness += 100.0;
            }
        }

        // P7: Dimensional analysis penalty (only active when input_units provided)
        if (!config_.input_units.empty()) {
            graph.fitness += config_.dim_penalty_weight * dimensional_penalty(graph);
        }

        graph.fitness_valid = true;  // E6
        return graph.fitness;
    }
    
    IndividualGraph mutate_lamarckian(IndividualGraph parent, double structural_rate) {
        IndividualGraph child = parent;
        child.fitness_valid = false;  // E6: structure/params may change
        
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
                        std::uniform_int_distribution<int> child_dist(0, i - 1);
                        node.left_child = child_dist(rng_);
                        node.unary_op = sample_unary_op_for_child(child, node.left_child);
                        if (node.unary_op == UnaryOp::IntPow) {
                            const int intpow_candidates[] = {2, 3, 4, 5, 6};
                            std::uniform_int_distribution<int> ip_dist(0, 4);
                            node.p = static_cast<double>(intpow_candidates[ip_dist(rng_)]);
                        }
                    } else {
                        node.type = NodeType::Binary;
                        node.binary_op = sample_binary_op();
                        if (node.binary_op == BinaryOp::Arithmetic) {
                            seed_arithmetic_gate(node);
                        }
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
                    // amplitude is fixed at 1.0 - SVD handles scaling
                    if (node.type == NodeType::Constant) node.value += rnorm(rng_);
                    
                    // H-03: clamp p/omega/phi (Exp tighter) after parametric mutation.
                    clamp_unary_inner_params(node);
                    if (node.type == NodeType::Unary && node.unary_op == UnaryOp::IntPow) {
                        node.p = static_cast<double>(std::clamp(static_cast<int>(std::round(node.p)), 2, 6));
                    }
                    if (node.type == NodeType::Binary && node.binary_op == BinaryOp::Arithmetic) {
                        node.beta = std::clamp(node.beta + rnorm(rng_) * 0.15, 0.5, 2.5);
                        node.gamma = std::clamp(node.gamma + rnorm(rng_) * 0.15, -1.5, 1.5);
                    }
                    if (node.type == NodeType::Binary && node.binary_op == BinaryOp::Aggregation) {
                        node.tau = std::clamp(node.tau + rnorm(rng_) * 0.1, 0.1, 10.0);
                    }
                }
            }
        }
        
        return child;
    }

    // -- Macro-Mutations ----------------------------------------------------
    // Structural mutations that preserve building blocks:
    //   - Wrap:     f(x) -> sin(f(x)) or exp(f(x)) or |f(x)|^p
    //   - Multiply: f(x), g(x) -> f(x) * g(x)
    //   - Nest:     f(x), g(x) -> f(g(x))
    IndividualGraph macro_mutate(const IndividualGraph& parent) {
        IndividualGraph child = parent;
        child.fitness_valid = false;  // E6
        std::uniform_real_distribution<double> runif(0.0, 1.0);
        
        int n = static_cast<int>(child.nodes.size());
        if (n < 2) return mutate_lamarckian(child, 0.3); // Too small for macro
        if (n >= config_.max_nodes) {
            return mutate_lamarckian(child, 0.15);
        }

        // Mode thresholds from [wrap, multiply, divide, nest] weights (default historical).
        double w_wrap = 0.4, w_mul = 0.2, w_div = 0.2, w_nest = 0.2;
        if (config_.macro_mode_weights.size() >= 4) {
            w_wrap = std::max(0.0, config_.macro_mode_weights[0]);
            w_mul = std::max(0.0, config_.macro_mode_weights[1]);
            w_div = std::max(0.0, config_.macro_mode_weights[2]);
            w_nest = std::max(0.0, config_.macro_mode_weights[3]);
            double wsum = w_wrap + w_mul + w_div + w_nest;
            if (wsum > 1e-12) {
                w_wrap /= wsum; w_mul /= wsum; w_div /= wsum; w_nest /= wsum;
            } else {
                w_wrap = 0.4; w_mul = 0.2; w_div = 0.2; w_nest = 0.2;
            }
        }
        // E8: nest creates f(g(x)) compositions that bloat under noise; down-weight
        // nest when robust/weighted search can absorb weak structure into fitness.
        if ((config_.loss_mode != LossMode::Mse) || has_y_weights_) {
            w_nest *= 0.35;
            double wsum = w_wrap + w_mul + w_div + w_nest;
            if (wsum > 1e-12) {
                w_wrap /= wsum; w_mul /= wsum; w_div /= wsum; w_nest /= wsum;
            }
        }
        const double t_wrap = w_wrap;
        const double t_mul = t_wrap + w_mul;
        const double t_div = t_mul + w_div;
        
        double roll = runif(rng_);
        
        if (roll < t_wrap) {
            // -- Wrap Mutation --
            // Prefer active building blocks, falling back to random nodes.
            int target = sample_active_node(child, 1, n - 1);
            
            // Create new unary node that takes 'target' as input
            OpNode wrap_node;
            wrap_node.type = NodeType::Unary;
            wrap_node.left_child = target;
            wrap_node.unary_op = sample_unary_op_for_child(child, target);
            wrap_node.p = 1.0;
            wrap_node.omega = 1.0;
            wrap_node.phi = 0.0;
            wrap_node.amplitude = 1.0;
            
            // If wrapping with Power, use interesting exponents
            if (wrap_node.unary_op == UnaryOp::Power) {
                const double power_candidates[] = {-1.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0};
                std::vector<double> valid_powers;
                for (double candidate : power_candidates) {
                    if (candidate >= config_.p_min && candidate <= config_.p_max) {
                        valid_powers.push_back(candidate);
                    }
                }
                if (!valid_powers.empty()) {
                    std::uniform_int_distribution<int> pow_dist(0, static_cast<int>(valid_powers.size()) - 1);
                    wrap_node.p = valid_powers[pow_dist(rng_)];
                }
            }
            if (wrap_node.unary_op == UnaryOp::IntPow) {
                const int intpow_candidates[] = {2, 3, 4, 5, 6};
                std::uniform_int_distribution<int> ip_dist(0, 4);
                wrap_node.p = static_cast<double>(intpow_candidates[ip_dist(rng_)]);
            }
            // If wrapping with Periodic, seed useful frequencies
            if (wrap_node.unary_op == UnaryOp::Periodic) {
                double freqs[] = {1.0, 2.0, 3.0, 0.5};
                std::uniform_int_distribution<int> freq_dist(0, 3);
                wrap_node.omega = freqs[freq_dist(rng_)];
            }
            
            child.nodes.push_back(wrap_node);
            child.output_weights.push_back(0.5); // Small initial weight
            
        } else if (roll < t_mul) {
            // -- Multiply Mutation --
            // Recombine active modules into a product candidate.
            int left = sample_active_node(child, 0, n - 1);
            int right = sample_active_node(child, 0, n - 1);
            int attempts = 0;
            while (right == left && n > 1 && attempts++ < 8) {
                right = sample_active_node(child, 0, n - 1);
            }
            if (right == left && n > 1) {
                right = (left + 1) % n;
            }
            
            OpNode mul_node;
            mul_node.type = NodeType::Binary;
            mul_node.binary_op = BinaryOp::Arithmetic;
            seed_arithmetic_gate(mul_node);
            mul_node.beta = 2.0;  // 2.0 = multiply mode
            mul_node.gamma = 1.0;
            mul_node.left_child = left;
            mul_node.right_child = right;
            
            child.nodes.push_back(mul_node);
            child.output_weights.push_back(1.0);
            
            // P3: Zero out children's additive contribution
            if (left < static_cast<int>(child.output_weights.size())) {
                child.output_weights[left] = 0.0;
            }
            if (right < static_cast<int>(child.output_weights.size())) {
                child.output_weights[right] = 0.0;
            }
            
        } else if (roll < t_div) {
            // -- Divide / Rational Mutation --
            // Prefer useful numerator/denominator modules (Analytic Quotient).
            int left = sample_active_node(child, 0, n - 1);
            int right = sample_active_node(child, 0, n - 1);
            int attempts = 0;
            while (right == left && n > 1 && attempts++ < 8) {
                right = sample_active_node(child, 0, n - 1);
            }
            if (right == left && n > 1) {
                right = (left + 1) % n;
            }
            
            OpNode div_node;
            div_node.type = NodeType::Binary;
            div_node.binary_op = sample_binary_op();
            if (div_node.binary_op == BinaryOp::Arithmetic) {
                seed_arithmetic_gate(div_node);
                div_node.beta = 2.0;
                div_node.gamma = 1.0;
            } else if (div_node.binary_op == BinaryOp::Division) {
                div_node.beta = 2.0;
                div_node.gamma = -1.0;
            } else {
                div_node.tau = 1.0;
            }
            div_node.left_child = left;
            div_node.right_child = right;
            
            child.nodes.push_back(div_node);
            child.output_weights.push_back(1.0);
            
            if (left < static_cast<int>(child.output_weights.size())) {
                child.output_weights[left] = 0.0;
            }
            if (right < static_cast<int>(child.output_weights.size())) {
                child.output_weights[right] = 0.0;
            }
            
        } else {
            // -- Nest Mutation --
            // Pick a unary node f and change its input to another node g
            // Creating f(g(x)) from f(old_input) and g(x)
            (void)w_nest;
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
            
            int f_idx = sample_active_from_candidates(child, unary_indices);
            
            // Pick an active lower-index module as the new input.
            int g_idx = sample_active_node(child, 0, f_idx > 0 ? f_idx - 1 : 0);
            
            // Rewire: f's input becomes g (creating f(g(x)) composition)
            child.nodes[f_idx].left_child = g_idx;
        }
        
        if (static_cast<int>(child.nodes.size()) > config_.max_nodes) {
            return parent;
        }
        return child;
    }
    
    // -- Subtree Crossover --------------------------------------------------
    // Swaps a contiguous subtree between two parents to produce one child.
    // A "subtree" here is all nodes reachable from a selected crossover point.
    IndividualGraph crossover(const IndividualGraph& parent_a, const IndividualGraph& parent_b) {
        ++crossover_attempts_;
        last_crossover_valid_ = false;
        IndividualGraph child = parent_a; // Start from parent A

        if (parent_a.nodes.size() < 3 || parent_b.nodes.size() < 3) {
            return child; // Too small for meaningful crossover
        }

        // Prefer active crossover points so recombination exchanges useful modules.
        int xo_a = sample_active_node(parent_a, 1, static_cast<int>(parent_a.nodes.size()) - 1);
        int xo_b = sample_active_node(parent_b, 1, static_cast<int>(parent_b.nodes.size()) - 1);

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
                if (new_left < 0 || new_left >= static_cast<int>(xo_a + subtree_b.size())) {
                    return parent_a;
                }
                donated.left_child = new_left;
            }
            if (donated.right_child >= 0) {
                int new_right = donated.right_child + offset;
                if (new_right < 0 || new_right >= static_cast<int>(xo_a + subtree_b.size())) {
                    return parent_a;
                }
                donated.right_child = new_right;
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
                int shifted = n.left_child + size_diff;
                if (shifted < 0) return parent_a;
                n.left_child = shifted;
            }
            if (n.right_child >= xo_a) {
                int shifted = n.right_child + size_diff;
                if (shifted < 0) return parent_a;
                n.right_child = shifted;
            }
            new_nodes.push_back(n);
        }

        // Safety: cap graph size to prevent bloat
        if (new_nodes.size() > static_cast<size_t>(config_.max_nodes)) {
            return parent_a;
        }

        // Reject offspring that violate DAG invariants; do not silently repair.
        int total = static_cast<int>(new_nodes.size());
        for (int i = 0; i < total; ++i) {
            const auto& n = new_nodes[i];
            if ((n.type == NodeType::Unary || n.type == NodeType::Binary)) {
                if (n.left_child < 0 || n.left_child >= i) return parent_a;
            }
            if (n.type == NodeType::Binary) {
                if (n.right_child < 0 || n.right_child >= i) return parent_a;
            }
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
        child.weighted_mse = 1e9;
        child.fitness_valid = false;  // E6
        last_crossover_valid_ = true;
        ++crossover_successes_;
        return child;
    }

    IndividualGraph crossover_with_retry(const IndividualGraph& parent_a,
                                         const IndividualGraph& parent_b,
                                         int max_attempts = 3) {
        IndividualGraph child = parent_a;
        int attempts = std::max(1, max_attempts);
        for (int i = 0; i < attempts; ++i) {
            child = crossover(parent_a, parent_b);
            if (last_crossover_valid_) {
                return child;
            }
        }
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

    std::vector<double> compute_activity_scores(const IndividualGraph& graph) {
        int n = static_cast<int>(graph.nodes.size());
        std::vector<double> scores(n, 1e-3);
        int w_count = static_cast<int>(graph.output_weights.size());

        for (int i = 0; i < n; ++i) {
            if (i < w_count) {
                scores[i] += std::abs(graph.output_weights[i]);
            }
        }

        // Active output terms make their dependencies valuable modules too.
        for (int root = 0; root < n && root < w_count; ++root) {
            double root_weight = std::abs(graph.output_weights[root]);
            if (root_weight < 1e-8) continue;

            std::vector<bool> visited(n, false);
            std::vector<std::pair<int, double>> stack;
            stack.emplace_back(root, root_weight * 0.5);

            while (!stack.empty()) {
                int idx = stack.back().first;
                double contribution = stack.back().second;
                stack.pop_back();
                if (idx < 0 || idx >= n || visited[idx] || contribution < 1e-6) continue;
                visited[idx] = true;
                scores[idx] += contribution;

                const auto& node = graph.nodes[idx];
                double child_contribution = contribution * 0.65;
                if ((node.type == NodeType::Unary || node.type == NodeType::Binary) &&
                    node.left_child >= 0 && node.left_child < idx) {
                    stack.emplace_back(node.left_child, child_contribution);
                }
                if (node.type == NodeType::Binary &&
                    node.right_child >= 0 && node.right_child < idx) {
                    stack.emplace_back(node.right_child, child_contribution);
                }
            }
        }

        for (int i = 0; i < n; ++i) {
            if (graph.nodes[i].type == NodeType::Constant) {
                scores[i] *= 0.25;
            }
            if (!std::isfinite(scores[i]) || scores[i] <= 0.0) {
                scores[i] = 1e-3;
            }
        }
        return scores;
    }

    int sample_active_node(const IndividualGraph& graph, int min_idx, int max_idx) {
        int n = static_cast<int>(graph.nodes.size());
        if (n == 0) return 0;
        min_idx = std::max(0, min_idx);
        max_idx = std::min(n - 1, max_idx);
        if (min_idx >= max_idx) return min_idx;

        auto scores = compute_activity_scores(graph);
        std::vector<double> weights;
        weights.reserve(max_idx - min_idx + 1);
        for (int i = min_idx; i <= max_idx; ++i) {
            double weight = scores[i];
            if (graph.nodes[i].type == NodeType::Input && min_idx > 0) {
                weight *= 0.2;
            }
            weights.push_back(std::max(1e-6, weight));
        }

        std::discrete_distribution<int> dist(weights.begin(), weights.end());
        return min_idx + dist(rng_);
    }

    int sample_active_from_candidates(const IndividualGraph& graph,
                                      const std::vector<int>& candidates) {
        if (candidates.empty()) return 0;
        if (candidates.size() == 1) return candidates.front();

        auto scores = compute_activity_scores(graph);
        std::vector<double> weights;
        weights.reserve(candidates.size());
        for (int idx : candidates) {
            double weight = (idx >= 0 && idx < static_cast<int>(scores.size())) ? scores[idx] : 1e-3;
            weights.push_back(std::max(1e-6, weight));
        }

        std::discrete_distribution<int> dist(weights.begin(), weights.end());
        return candidates[dist(rng_)];
    }
    
    // -- Ridge Regression solver for output weights ----------------------
    // Replaces bare SVD with (A^T W A + λI)^{-1} A^T W b to prevent
    // multicollinearity from producing massive cancelling coefficients.
    // When y_weights_ are set, W is diag(weights); otherwise W = I.
    // Phase 4 tighten: Huber / trimmed / student_t use a few IRLS iterations
    // so linear output coeffs are fit under the same robust objective as fitness.
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

        // Base sample weights (Phase 3) or uniform.
        Eigen::ArrayXd base_w = Eigen::ArrayXd::Ones(n_samples);
        if (has_y_weights_ && y_weights_.size() == n_samples) {
            base_w = y_weights_;
        }

        const bool robust =
            config_.loss_mode == LossMode::Huber
            || config_.loss_mode == LossMode::TrimmedMse
            || config_.loss_mode == LossMode::StudentT;
        const int irls_iters = robust ? 4 : 1;
        const double lambda = 1e-4;

        auto ridge_solve = [&](const Eigen::ArrayXd& ww, Eigen::VectorXd& w_out) -> bool {
            try {
                double wsum = ww.sum();
                if (!(wsum > 0.0) || !std::isfinite(wsum)) return false;
                Eigen::MatrixXd WA = ww.matrix().asDiagonal() * A;
                Eigen::MatrixXd AtWA = A.transpose() * WA;
                Eigen::VectorXd AtWb = A.transpose() * (ww * y_).matrix();
                AtWA.diagonal() += Eigen::VectorXd::Constant(num_features + 1, lambda);
                w_out = AtWA.ldlt().solve(AtWb);
                return w_out.allFinite();
            } catch (...) {
                return false;
            }
        };

        auto pred_from_w = [&](const Eigen::VectorXd& coef) -> Eigen::ArrayXd {
            Eigen::ArrayXd pred = Eigen::ArrayXd::Constant(n_samples, coef(num_features));
            for (int f = 0; f < num_features; ++f) {
                pred += coef(f) * A.col(f).array();
            }
            return pred;
        };

        auto irls_weights = [&](const Eigen::ArrayXd& resid) -> Eigen::ArrayXd {
            Eigen::ArrayXd rw = Eigen::ArrayXd::Ones(n_samples);
            if (config_.loss_mode == LossMode::Huber) {
                double d = config_.huber_delta;
                if (!(d > 0.0) || !std::isfinite(d)) d = mad_scale(resid);
                d = std::max(d, 1e-12);
                for (int i = 0; i < n_samples; ++i) {
                    double a = std::abs(resid(i));
                    // Standard Huber IRLS: w = 1 if |r|<=d else d/|r|
                    rw(i) = (a <= d || a < 1e-15) ? 1.0 : (d / a);
                }
            } else if (config_.loss_mode == LossMode::TrimmedMse) {
                Eigen::ArrayXd sq = resid.square();
                double frac = std::clamp(config_.trim_fraction, 0.0, 0.45);
                int drop = static_cast<int>(std::llround(n_samples * frac));
                drop = std::clamp(drop, 0, std::max(0, n_samples - 1));
                if (drop > 0) {
                    std::vector<int> order(static_cast<size_t>(n_samples));
                    for (int i = 0; i < n_samples; ++i) order[static_cast<size_t>(i)] = i;
                    std::partial_sort(
                        order.begin(),
                        order.begin() + drop,
                        order.end(),
                        [&](int a, int b) { return sq(a) > sq(b); }
                    );
                    for (int k = 0; k < drop; ++k) {
                        rw(order[static_cast<size_t>(k)]) = 1e-6; // soft zero (keep matrix invertible)
                    }
                }
            } else if (config_.loss_mode == LossMode::StudentT) {
                double s = config_.huber_delta;
                if (!(s > 0.0) || !std::isfinite(s)) s = mad_scale(resid);
                s = std::max(s, 1e-12);
                // IRLS for log(1+(r/s)^2): weight ∝ 1 / (1 + (r/s)^2)
                for (int i = 0; i < n_samples; ++i) {
                    double z = resid(i) / s;
                    rw(i) = 1.0 / (1.0 + z * z);
                }
            }
            return rw;
        };

        Eigen::ArrayXd cur_w = base_w;
        if (!ridge_solve(cur_w, w)) {
            return false;
        }

        for (int it = 1; it < irls_iters; ++it) {
            Eigen::ArrayXd resid = pred_from_w(w) - y_;
            if (!resid.isFinite().all()) break;
            Eigen::ArrayXd rw = irls_weights(resid);
            cur_w = base_w * rw;
            // Renormalize mean ~1 for numerical stability with ridge scale.
            double mean_w = cur_w.mean();
            if (mean_w > 0.0 && std::isfinite(mean_w)) cur_w /= mean_w;
            Eigen::VectorXd w_next;
            if (!ridge_solve(cur_w, w_next)) break;
            // §3.137: stop when the IRLS fixed point settles instead of
            // always running the full fixed iteration budget.
            double w_change = (w_next - w).cwiseAbs().maxCoeff();
            w = w_next;
            if (std::isfinite(w_change) && w_change < 1e-10) break;
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
        // E6: output layer changed - cached fitness is stale until re-scored.
        ind.fitness_valid = false;
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
            // NOTE: refine_inner_params is NOT called here - too expensive
            // for per-child use. It runs once on the best graph in cleanup_graph.
        }
    }
    
    // -- Finite-Difference Adam Optimizer for Inner Parameters ------------
    // Alternates between: (1) Adam steps on {p, omega, phi},
    // then (2) SVD refit of output weights. This ensures linear weights
    // stay in sync with the refined inner parameters.

    // S5-8: unaries that participate in any active output basis (including nested
    // ancestors), not only those with nonzero *output* weight.
    // Reachable nodes under active output weights (direct basis + nested).
    // Used for unary refine and for binary Arithmetic snap (H-05).
    std::vector<char> collect_active_reachable_mask(const IndividualGraph& ind) const {
        const int n = static_cast<int>(ind.nodes.size());
        std::vector<char> reachable(static_cast<size_t>(std::max(0, n)), 0);
        if (n <= 0) return reachable;
        std::vector<int> stack;
        stack.reserve(static_cast<size_t>(n));
        for (int i = 0; i < n && i < static_cast<int>(ind.output_weights.size()); ++i) {
            if (std::abs(ind.output_weights[static_cast<size_t>(i)]) > kOutputWeightActive) {
                stack.push_back(i);
            }
        }
        while (!stack.empty()) {
            int idx = stack.back();
            stack.pop_back();
            if (idx < 0 || idx >= n || reachable[static_cast<size_t>(idx)]) continue;
            reachable[static_cast<size_t>(idx)] = 1;
            const auto& node = ind.nodes[static_cast<size_t>(idx)];
            if ((node.type == NodeType::Unary || node.type == NodeType::Binary) && node.left_child >= 0) {
                stack.push_back(node.left_child);
            }
            if (node.type == NodeType::Binary && node.right_child >= 0) {
                stack.push_back(node.right_child);
            }
        }
        return reachable;
    }

    std::vector<int> collect_active_unary_indices(const IndividualGraph& ind) const {
        std::vector<int> out;
        if (ind.nodes.empty()) return out;
        const int n = static_cast<int>(ind.nodes.size());
        const std::vector<char> reachable = collect_active_reachable_mask(ind);
        for (int i = 0; i < n; ++i) {
            if (!reachable[static_cast<size_t>(i)]) continue;
            if (ind.nodes[static_cast<size_t>(i)].type == NodeType::Unary) {
                out.push_back(i);
            }
        }
        return out;
    }

    // §3.3: analytical LM Jacobian multiplies the local node derivative by
    // the direct output weight wi, omitting downstream nonlinear ancestors.
    // Detect whether any *other* reachable op node transitively consumes
    // node_idx; when true the chain term is missing and FD is required.
    bool has_nonlinear_downstream(const IndividualGraph& g, int node_idx,
                                  const std::vector<char>& reachable) const {
        const int n = static_cast<int>(g.nodes.size());
        if (node_idx < 0 || node_idx >= n) return false;
        for (int j = 0; j < n; ++j) {
            if (j == node_idx) continue;
            if (j < static_cast<int>(reachable.size()) && !reachable[static_cast<size_t>(j)]) continue;
            const auto& cand = g.nodes[static_cast<size_t>(j)];
            const bool is_op = (cand.type == NodeType::Unary || cand.type == NodeType::Binary);
            if (!is_op) continue;
            // DFS from candidate down through children seeking node_idx.
            std::vector<int> stack;
            stack.push_back(j);
            std::vector<char> seen(static_cast<size_t>(n), 0);
            while (!stack.empty()) {
                int cur = stack.back();
                stack.pop_back();
                if (cur == node_idx) return true;
                if (cur < 0 || cur >= n || seen[static_cast<size_t>(cur)]) continue;
                seen[static_cast<size_t>(cur)] = 1;
                const auto& nd = g.nodes[static_cast<size_t>(cur)];
                if (nd.type == NodeType::Unary && nd.left_child >= 0) {
                    if (nd.left_child == node_idx) return true;
                    stack.push_back(nd.left_child);
                } else if (nd.type == NodeType::Binary) {
                    if (nd.left_child == node_idx || nd.right_child == node_idx) return true;
                    if (nd.left_child >= 0) stack.push_back(nd.left_child);
                    if (nd.right_child >= 0) stack.push_back(nd.right_child);
                }
            }
        }
        return false;
    }

    void refine_inner_params_adam(IndividualGraph& ind) {
        if (ind.nodes.empty()) return;
        int n_samples = static_cast<int>(y_.size());
        
        // Collect unaries in active output subtrees (S5-8 nested refine).
        std::vector<int> active_unary = collect_active_unary_indices(ind);
        if (active_unary.empty()) return;
        // E6: about to mutate inner params - force re-score if interrupted/early-exit.
        ind.fitness_valid = false;
        
        // Adam hyperparameters
        const double lr = 0.02;
        const double beta1 = 0.9, beta2 = 0.999, eps_adam = 1e-8;
        const double epsilon = 1e-4; // Finite difference step
        const int adam_steps_per_round = 25;  // P2: boosted from 10 for better constant discovery
        const int num_rounds = 3; // Alternate Adam -> SVD this many times

        // P-04: which of {p, omega, phi} actually enter the eval for each op.
        // p feeds only Power/IntPow; omega/phi feed only Periodic/Exp;
        // Log/Abs have no inner parameters at all. Probing inert slots cost
        // two full-graph evals per slot per step for pure finite-difference
        // noise (analytically zero gradient).
        static constexpr bool kParamLive[6][3] = {
            /*Periodic*/ {false, true,  true},
            /*Power*/    {true,  false, false},
            /*IntPow*/   {true,  false, false},
            /*Exp*/      {false, true,  true},
            /*Log*/      {false, false, false},
            /*Abs*/      {false, false, false},
        };
        auto slot_live = [&](const OpNode& node, int pi) -> bool {
            if (!config_.fd_skip_inert_params) return true;
            const int op = static_cast<int>(node.unary_op);
            return op >= 0 && op < 6 && kParamLive[op][pi];
        };

        int n_params = static_cast<int>(active_unary.size()) * 3; // {p, omega, phi} - NOT amplitude (redundant with SVD output_weight)
        std::vector<double> m(n_params, 0.0), v(n_params, 0.0);

        double best_mse = objective_mse(ind);
        IndividualGraph best_snapshot = ind; // Keep best seen
        int global_step = 0;

        for (int round = 0; round < num_rounds; ++round) {
            // -- Phase 1: Adam steps on inner params --
            for (int step = 0; step < adam_steps_per_round; ++step) {
                // P-04: one base cache + prediction per step; every FD probe
                // below recomputes only the perturbed subtree
                // (evaluate_graph_partial + value deltas onto base_pred)
                // instead of re-evaluating the whole graph twice per parameter.
                std::vector<Eigen::ArrayXd> base_cache;
                evaluate_graph(ind, X_, n_samples, base_cache);
                Eigen::ArrayXd base_pred = assemble_prediction(ind, base_cache, n_samples);

                std::vector<double> grads(n_params, 0.0);

                for (int ai = 0; ai < static_cast<int>(active_unary.size()); ++ai) {
                    int node_idx = active_unary[ai];
                    auto& node = ind.nodes[node_idx];
                    // Only optimize {p, omega, phi} - amplitude handled by SVD
                    double* params[3] = {&node.p, &node.omega, &node.phi};

                    for (int pi = 0; pi < 3; ++pi) {
                        fd_probes_total_ += 2;  // P-04: plus/minus probe pair
                        if (!slot_live(node, pi)) {
                            fd_probes_skipped_inert_ += 2;
                            continue;
                        }

                        double original = *params[pi];

                        *params[pi] = original + epsilon;
                        Eigen::ArrayXd pred_plus = evaluate_perturbed_pred(
                            ind, node_idx, base_cache, base_pred, n_samples);
                        double mse_plus = residual_mse(pred_plus, y_);

                        *params[pi] = original - epsilon;
                        Eigen::ArrayXd pred_minus = evaluate_perturbed_pred(
                            ind, node_idx, base_cache, base_pred, n_samples);
                        double mse_minus = residual_mse(pred_minus, y_);

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
                    
                    clamp_unary_inner_params(node);
                }
                global_step++;
            }
            
            // -- Phase 2: Ridge refit of output weights --
            // Inner params changed, so re-solve the linear layer analytically
            {
                std::vector<Eigen::ArrayXd> cache;
                evaluate_graph(ind, X_, n_samples, cache);
                if (!solve_output_weights(ind, cache)) continue;
            }
            
            // Evaluate and track best
            evaluate_fitness_with_penalty(ind, X_, y_, n_samples);
            if (objective_mse(ind) < best_mse) {
                best_mse = objective_mse(ind);
                best_snapshot = ind;
            }
            
            // Early exit if already excellent
            if (objective_mse(ind) < 1e-10) break;
        }
        
        // Restore best seen (in case later rounds degraded)
        if (objective_mse(best_snapshot) < objective_mse(ind)) {
            ind = best_snapshot;
        }
    }

    // -- Levenberg-Marquardt-style optimizer for inner nonlinear params --─
    // Optimizes unary {p, omega, phi} while analytically refitting output
    // weights for each trial point (variable projection style).
    bool refine_inner_params_lm(IndividualGraph& ind) {
        if (ind.nodes.empty()) return false;
        const int n_samples = static_cast<int>(y_.size());
        if (n_samples <= 0) return false;

        std::vector<int> active_unary = collect_active_unary_indices(ind);
        if (active_unary.empty()) return false;

        const int n_params = static_cast<int>(active_unary.size()) * 3;
        if (n_params <= 0) return false;

        auto pack_params = [&](const IndividualGraph& g) {
            Eigen::VectorXd theta(n_params);
            for (int ai = 0; ai < static_cast<int>(active_unary.size()); ++ai) {
                const auto& node = g.nodes[active_unary[ai]];
                theta(ai * 3 + 0) = node.p;
                theta(ai * 3 + 1) = node.omega;
                theta(ai * 3 + 2) = node.phi;
            }
            return theta;
        };

        auto unpack_params = [&](IndividualGraph& g, const Eigen::VectorXd& theta) {
            for (int ai = 0; ai < static_cast<int>(active_unary.size()); ++ai) {
                auto& node = g.nodes[active_unary[ai]];
                node.p = theta(ai * 3 + 0);
                node.omega = theta(ai * 3 + 1);
                node.phi = theta(ai * 3 + 2);
                // H-03: clamp p/omega/phi (Exp gets tighter omega/phi bounds).
                clamp_unary_inner_params(node);
            }
        };

        auto evaluate_residual = [&](IndividualGraph& g, Eigen::VectorXd* residual_out = nullptr) {
            std::vector<Eigen::ArrayXd> cache;
            evaluate_graph(g, X_, n_samples, cache);
            if (!solve_output_weights(g, cache)) {
                return std::numeric_limits<double>::infinity();
            }

            Eigen::ArrayXd pred = Eigen::ArrayXd::Constant(n_samples, g.output_bias);
            for (size_t i = 0; i < g.output_weights.size() && i < cache.size(); ++i) {
                if (std::abs(g.output_weights[i]) > 1e-12) pred += g.output_weights[i] * cache[i];
            }

            Eigen::ArrayXd residual = pred - y_;
            double mse = residual_mse(pred, y_);
            if (!std::isfinite(mse)) return std::numeric_limits<double>::infinity();

            if (residual_out != nullptr) {
                *residual_out = residual.matrix();
            }
            return mse;
        };

        Eigen::VectorXd theta = pack_params(ind);
        double lambda = config_.lm_lambda_init;
        IndividualGraph best_snapshot = ind;
        Eigen::VectorXd residual;
        double best_mse = evaluate_residual(best_snapshot, &residual);
        if (!std::isfinite(best_mse)) return false;

        bool improved_any = false;
        const double fd_eps = 1e-4;

        for (int iter = 0; iter < config_.lm_max_iterations; ++iter) {
            IndividualGraph base_graph = best_snapshot;
            unpack_params(base_graph, theta);
            
            std::vector<Eigen::ArrayXd> base_cache;
            evaluate_graph(base_graph, X_, n_samples, base_cache);
            
            DifferentialGramian dg;
            dg.initialize(base_cache, y_, has_y_weights_ ? y_weights_ : Eigen::ArrayXd());
            
            Eigen::VectorXd r;
            double base_mse = evaluate_residual(base_graph, &r);
            if (!std::isfinite(base_mse)) {
                lambda *= 10.0;
                continue;
            }
            
            int n_features = static_cast<int>(base_cache.size());

            // Get base weights explicitly to use in analytical jacobian
            Eigen::VectorXd base_w;
            if (!dg.solve_ridge(1e-4, base_w)) {
                lambda *= 10.0;
                continue;
            }

            Eigen::MatrixXd J(n_samples, n_params);
            J.setZero();
            
            // Build A matrix for the VARPRO projection step
            Eigen::MatrixXd A(n_samples, n_features + 1);
            for (int i = 0; i < n_features; ++i) A.col(i) = base_cache[i].matrix();
            A.col(n_features).setOnes();
            
            // Precompute QR decomposition of A for the projection P_perp
            // We want to compute v - A * (A^T A + lambda I)^{-1} A^T * v
            Eigen::MatrixXd G_ridge = A.transpose() * A;
            G_ridge.diagonal().array() += 1e-4;
            Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr;
            qr.compute(G_ridge);
            
            for (int ai = 0; ai < static_cast<int>(active_unary.size()); ++ai) {
                int node_idx = active_unary[ai];
                if (node_idx < 0 || node_idx >= static_cast<int>(base_graph.nodes.size())) continue;
                const auto& node = base_graph.nodes[static_cast<size_t>(node_idx)];
                // H-04: bounds-check left_child before indexing cache.
                const int n_nodes = static_cast<int>(base_graph.nodes.size());
                const int child_idx = node.left_child;
                const bool child_ok = (child_idx >= 0 && child_idx < n_nodes
                    && child_idx < static_cast<int>(base_cache.size())
                    && base_cache[static_cast<size_t>(child_idx)].size() == n_samples);
                double wi = (node_idx < base_w.size()) ? base_w(node_idx) : 0.0;
                // Nested unaries often have near-zero direct output weight; the
                // analytical path multiplies by wi and starves gradients. Fall
                // back to residual finite differences when |wi| is tiny (H-04).
                // §3.3: even with large |wi|, a downstream nonlinear ancestor
                // contributes a chain term the wi-only product omits, so nested
                // nodes must also take FD (exact up to FD error).
                constexpr double kDirectWeightEps = 1e-6;
                bool use_fd = !child_ok || !(std::abs(wi) > kDirectWeightEps)
                    || node_idx >= static_cast<int>(base_cache.size())
                    || base_cache[static_cast<size_t>(node_idx)].size() != n_samples;
                if (!use_fd) {
                    const std::vector<char> reach_mask =
                        collect_active_reachable_mask(base_graph);
                    if (has_nonlinear_downstream(base_graph, node_idx, reach_mask)) {
                        use_fd = true;
                    }
                }

                // P-04: Log/Abs have no inner parameters — all three Jacobian
                // columns are analytically zero. Skip entirely; count the six
                // FD residual evals avoided only when this node would have
                // taken the finite-difference fallback (the analytical branch
                // merely wasted zero-column Gramian work, not evals).
                if (config_.fd_skip_inert_params
                        && (node.unary_op == UnaryOp::Log || node.unary_op == UnaryOp::Abs)) {
                    if (use_fd) {
                        fd_probes_total_ += 6;
                        fd_probes_skipped_inert_ += 6;
                    }
                    continue;
                }

                if (use_fd) {
                    auto& mut_node = base_graph.nodes[static_cast<size_t>(node_idx)];
                    double* params[3] = {&mut_node.p, &mut_node.omega, &mut_node.phi};
                    for (int pi = 0; pi < 3; ++pi) {
                        const double original = *params[pi];
                        *params[pi] = original + fd_eps;
                        Eigen::VectorXd r_plus;
                        evaluate_residual(base_graph, &r_plus);
                        *params[pi] = original - fd_eps;
                        Eigen::VectorXd r_minus;
                        evaluate_residual(base_graph, &r_minus);
                        *params[pi] = original;
                        int pidx = ai * 3 + pi;
                        if (r_plus.size() == n_samples && r_minus.size() == n_samples) {
                            J.col(pidx) = (r_plus - r_minus) / (2.0 * fd_eps);
                            if (!J.col(pidx).allFinite()) J.col(pidx).setZero();
                        }
                    }
                    // Restore graph params from theta (FD may have left clamps).
                    unpack_params(base_graph, theta);
                    continue;
                }

                const auto& child_out = base_cache[static_cast<size_t>(child_idx)];

                Eigen::ArrayXd df_dp, df_domega, df_dphi;

                switch (node.unary_op) {
                    case UnaryOp::Periodic: {
                        Eigen::ArrayXd arg = node.omega * child_out + node.phi;
                        Eigen::ArrayXd cos_arg = arg.cos();
                        df_domega = node.amplitude * child_out * cos_arg;
                        df_dphi   = node.amplitude * cos_arg;
                        df_dp     = Eigen::ArrayXd::Zero(n_samples);
                        break;
                    }
                    case UnaryOp::Exp: {
                        // base_cache[node_idx] is exactly exp(omega*x + phi) * amplitude
                        Eigen::ArrayXd val = base_cache[static_cast<size_t>(node_idx)];
                        df_domega = child_out * val;
                        df_dphi   = val;
                        df_dp     = Eigen::ArrayXd::Zero(n_samples);
                        break;
                    }
                    case UnaryOp::Power: {
                        // derivative of |x|^p w.r.t p is |x|^p * ln(|x|)
                        Eigen::ArrayXd abs_x = child_out.abs() + 1e-10;
                        df_dp     = base_cache[static_cast<size_t>(node_idx)] * abs_x.log();
                        df_domega = Eigen::ArrayXd::Zero(n_samples);
                        df_dphi   = Eigen::ArrayXd::Zero(n_samples);
                        break;
                    }
                    default:
                        df_dp = df_domega = df_dphi = Eigen::ArrayXd::Zero(n_samples);
                }

                Eigen::ArrayXd* derivs[3] = {&df_dp, &df_domega, &df_dphi};
                for (int pi = 0; pi < 3; ++pi) {
                    Eigen::VectorXd dAw = ((*derivs[pi]) * wi).matrix();
                    
                    // Varpro correction projection v = dAw - A * (A^T A + \lambda I)^{-1} A^T * dAw
                    Eigen::VectorXd AtdAw = A.transpose() * dAw;
                    Eigen::VectorXd correction = A * qr.solve(AtdAw);
                    
                    int pidx = ai * 3 + pi;
                    J.col(pidx) = dAw - correction;
                    // §3.410: same finite sanitization as the FD branch —
                    // clamped node outputs times large log terms can still
                    // yield non-finite entries that would poison H/g_grad.
                    if (!J.col(pidx).allFinite()) J.col(pidx).setZero();
                }
            }

            Eigen::MatrixXd H = J.transpose() * J;
            Eigen::VectorXd g_grad = J.transpose() * r;

            H.diagonal().array() += lambda;

            Eigen::VectorXd delta = H.ldlt().solve(-g_grad);
            if (!delta.allFinite()) {
                lambda *= 10.0;
                continue;
            }

            if (delta.norm() < 1e-8) break;

            Eigen::VectorXd theta_trial = theta + delta;
            IndividualGraph trial_graph = base_graph;
            unpack_params(trial_graph, theta_trial);
            
            // Trial step evaluation uses full pruned path to ensure validity
            double trial_mse = evaluate_residual(trial_graph, nullptr);

            if (std::isfinite(trial_mse) && trial_mse < base_mse) {
                theta = theta_trial;
                best_snapshot = trial_graph;
                best_mse = trial_mse;
                improved_any = true;
                lambda = std::max(1e-8, lambda * 0.5);
                if (best_mse < 1e-12) break;
            } else {
                lambda = std::min(1e6, lambda * 2.0);
            }
        }

        if (improved_any && std::isfinite(best_mse)) {
            evaluate_fitness_with_penalty(best_snapshot, X_, y_, n_samples);
            ind = std::move(best_snapshot);
            return true;
        }
        return false;
    }

    // Entry point: prefer LM-style optimizer, fallback to Adam if requested.
    void refine_inner_params(IndividualGraph& ind) {
        if (config_.use_lm_inner_optimizer) {
            const bool lm_ok = refine_inner_params_lm(ind);
            if (!lm_ok && config_.lm_fallback_to_adam) {
                refine_inner_params_adam(ind);
            }
            return;
        }
        refine_inner_params_adam(ind);
    }
    
    // -- Post-Evolution Graph Cleanup ------------------------------------─
    // Uses output-correlation-based deduplication (like PyTorch pruning.py):
    // Nodes producing identical outputs get merged regardless of structure.
    // This catches x == (x+x)/2 == (x+x+x)/3 etc.
    void cleanup_graph(IndividualGraph& ind) {
        if (ind.nodes.empty()) return;
        int n_samples = static_cast<int>(y_.size());
        
        // -- Step 1: Evaluate all nodes to get their actual output vectors --
        std::vector<Eigen::ArrayXd> cache;
        evaluate_graph(ind, X_, n_samples, cache);
        
        // -- Step 2: Correlation-based deduplication --
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
                    // Both constant - check if same constant
                    is_duplicate = std::abs(cache[i].mean() - cache[j].mean()) < 1e-6;
                } else if (var_i > 1e-12 && var_j > 1e-12) {
                    // Both non-constant - check correlation AND scale
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
        // Don't sum weights - let SVD refit handle optimal weights
        for (int j = 0; j < n_nodes; ++j) {
            if (canonical[j] != j && j < static_cast<int>(ind.output_weights.size())) {
                ind.output_weights[j] = 0.0;
            }
        }
        
        // -- Step 3: Remove dead nodes (zero output weight, not a dependency) --
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
        
        // -- Step 4: Ridge refit on clean graph --
        if (!ind.nodes.empty()) {
            std::vector<Eigen::ArrayXd> new_cache;
            evaluate_graph(ind, X_, n_samples, new_cache);
            solve_output_weights(ind, new_cache);
        }
        
        evaluate_fitness_with_penalty(ind, X_, y_, n_samples);
        double baseline_mse = objective_mse(ind);
        
        // -- Step 5: Iterative Backward Elimination --
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
            
            // Accept if MSE is still acceptable (within 5% of baseline)
            if (objective_mse(candidate) < baseline_mse * 1.05 + 1e-8) {
                ind = candidate;
                baseline_mse = objective_mse(ind);
            } else {
                break; // Can't remove any more without hurting accuracy
            }
        }
        
        // -- Step 6: Parameter & Coefficient Snapping ------------------------─
        // Try rounding inner parameters and output weights to clean values.
        // This converts 0.9997*sin(2.998*x + 0.0012) -> sin(3*x).
        {
            evaluate_fitness_with_penalty(ind, X_, y_, n_samples);
            double snap_baseline_mse = objective_mse(ind);

            // Finding 3: Initialize base cache and Gramian for batch snapping
            int n_feat = static_cast<int>(ind.nodes.size());
            std::vector<Eigen::ArrayXd> base_cache;
            evaluate_graph(ind, X_, n_samples, base_cache);
            DifferentialGramian dg;
            dg.initialize(base_cache, y_, has_y_weights_ ? y_weights_ : Eigen::ArrayXd());
            double ridge_lambda = 1e-8; // Low lambda for precision during snapping

            auto compute_mse_from_trial = [&](const Eigen::VectorXd& w, 
                                              const std::vector<Eigen::ArrayXd>& cache) -> double {
                int ns = static_cast<int>(y_.size());
                // solve_ridge returns [weights | bias]
                Eigen::ArrayXd pred = Eigen::ArrayXd::Constant(ns, w(n_feat));
                for (int f = 0; f < n_feat; ++f) {
                    if (std::abs(w(f)) > 1e-15) {
                        pred += w(f) * cache[f];
                    }
                }
                double mse = residual_mse(pred, y_);
                return std::isfinite(mse) ? mse : std::numeric_limits<double>::infinity();
            };

            enum class SnapTier {
                Integer,
                Fraction,
                Special,
            };

            auto snap_accept_ratio = [](SnapTier tier) -> double {
                switch (tier) {
                    case SnapTier::Integer:
                        return 1.01;
                    case SnapTier::Fraction:
                        return 1.02;
                    case SnapTier::Special:
                        return 1.05;
                }
                return 1.02;
            };

            auto snap_accepts = [&](double baseline_mse, double candidate_mse, SnapTier tier) -> bool {
                // Add a small constant slack (1e-7) to allow exact matches even if noisy baseline is slightly better.
                return candidate_mse < baseline_mse * snap_accept_ratio(tier) + 1e-7;
            };

            auto is_near = [](double a, double b) -> bool {
                return std::abs(a - b) < 1e-9;
            };

            auto classify_p_tier = [&](double candidate) -> SnapTier {
                return std::abs(candidate - std::round(candidate)) < 1e-9 ? SnapTier::Integer : SnapTier::Fraction;
            };

            auto classify_omega_tier = [&](double candidate) -> SnapTier {
                if (is_near(candidate, kPi) || is_near(candidate, 2.0 * kPi) || is_near(candidate, kPi / 2.0)) {
                    return SnapTier::Special;
                }
                return std::abs(candidate - std::round(candidate)) < 1e-9 ? SnapTier::Integer : SnapTier::Fraction;
            };

            auto classify_phi_tier = [&](double candidate) -> SnapTier {
                return std::abs(candidate - std::round(candidate)) < 1e-9 ? SnapTier::Integer : SnapTier::Special;
            };
            
            // 6a. Inner parameter snapping (p, omega, phi)
            // S5-8: allow nested unaries under active outputs, not only direct basis terms.
            // H-05: full reachable mask (Unary + Binary) so Arithmetic gate snap can run.
            std::vector<char> snap_active_nodes = collect_active_reachable_mask(ind);
            // Alias kept for unary snap loops (same mask; Binary bits are simply unused there).
            const std::vector<char>& snap_active_unary = snap_active_nodes;
            const double snap_candidates_p[] = {-2, -1.5, -1, -0.5, 0, 0.25, 1.0/3.0, 0.5, 2.0/3.0, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5};
            const int n_snap_p = sizeof(snap_candidates_p) / sizeof(snap_candidates_p[0]);
            
            const double snap_candidates_omega[] = {0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, kPi, 2*kPi, kPi/2};
            const int n_snap_omega = sizeof(snap_candidates_omega) / sizeof(snap_candidates_omega[0]);
            
            const double snap_candidates_phi[] = {0.0, kPi/4, kPi/2, kPi, 3*kPi/2, -kPi/4, -kPi/2, -kPi};
            const int n_snap_phi = sizeof(snap_candidates_phi) / sizeof(snap_candidates_phi[0]);
            
            for (int i = 0; i < static_cast<int>(ind.nodes.size()); ++i) {
                if (ind.nodes[i].type != NodeType::Unary) continue;
                if (i < 0 || i >= static_cast<int>(snap_active_unary.size()) || !snap_active_unary[static_cast<size_t>(i)]) continue; // S5-8 nested-active
                
                auto& node = ind.nodes[i];
                
                // Try snapping p
                {
                    double original_p = node.p;
                    double best_snap_p = original_p;
                    double best_snap_mse = std::numeric_limits<double>::infinity();
                    
                    for (int si = 0; si < n_snap_p; ++si) {
                        double candidate = snap_candidates_p[si];
                        if (std::abs(original_p - candidate) > 0.3) continue; 
                        
                        node.p = candidate;
                        
                        // Finding 3: Incremental Trial
                        std::vector<Eigen::ArrayXd> trial_cache;
                        std::vector<int> changed;
                        evaluate_graph_partial(ind, i, base_cache, trial_cache, changed);
                        
                        dg.update_nodes(changed, base_cache, trial_cache);
                        Eigen::VectorXd w_trial;
                        if (dg.solve_ridge(ridge_lambda, w_trial)) {
                            double trial_mse = compute_mse_from_trial(w_trial, trial_cache);
                            if (snap_accepts(snap_baseline_mse, trial_mse, classify_p_tier(candidate)) &&
                                trial_mse < best_snap_mse) {
                                best_snap_p = candidate;
                                best_snap_mse = trial_mse;
                            }
                        }
                        // REVERT Gramian to base state for next candidate
                        dg.update_nodes(changed, trial_cache, base_cache);
                    }
                    
                    node.p = best_snap_p;
                    if (best_snap_p != original_p) {
                        // Permanent update of base_cache and dg
                        std::vector<int> changed;
                        std::vector<Eigen::ArrayXd> new_cache;
                        evaluate_graph_partial(ind, i, base_cache, new_cache, changed);
                        dg.update_nodes(changed, base_cache, new_cache);
                        base_cache = std::move(new_cache);
                        
                        // Sync weights/bias
                        Eigen::VectorXd w_final;
                        if (dg.solve_ridge(ridge_lambda, w_final)) {
                            for (int f = 0; f < n_feat; ++f) ind.output_weights[f] = w_final(f);
                            ind.output_bias = w_final(n_feat);
                        }
                        evaluate_fitness_with_penalty(ind, X_, y_, n_samples);
                        snap_baseline_mse = objective_mse(ind);
                    }
                }
                
                // Try snapping omega
                if (node.unary_op == UnaryOp::Periodic) {
                    double original_omega = node.omega;
                    double best_snap_omega = original_omega;
                    double best_snap_mse = std::numeric_limits<double>::infinity();
                    
                    double nearest_int = std::round(original_omega);
                    
                    for (int si = 0; si < n_snap_omega; ++si) {
                        double candidate = snap_candidates_omega[si];
                        if (std::abs(original_omega - candidate) > 0.3) continue;
                        
                        node.omega = candidate;
                        
                        // Finding 3: Incremental Trial
                        std::vector<Eigen::ArrayXd> trial_cache;
                        std::vector<int> changed;
                        evaluate_graph_partial(ind, i, base_cache, trial_cache, changed);
                        
                        dg.update_nodes(changed, base_cache, trial_cache);
                        Eigen::VectorXd w_trial;
                        if (dg.solve_ridge(ridge_lambda, w_trial)) {
                            double trial_mse = compute_mse_from_trial(w_trial, trial_cache);
                            if (snap_accepts(snap_baseline_mse, trial_mse, classify_omega_tier(candidate)) &&
                                trial_mse < best_snap_mse) {
                                best_snap_omega = candidate;
                                best_snap_mse = trial_mse;
                            }
                        }
                        // REVERT
                        dg.update_nodes(changed, trial_cache, base_cache);
                    }
                    // Try nearest integer
                    if (nearest_int >= 1.0 && nearest_int <= 10.0 && std::abs(original_omega - nearest_int) <= 0.3) {
                        node.omega = nearest_int;
                        
                        // Finding 3: Incremental Trial
                        std::vector<Eigen::ArrayXd> trial_cache;
                        std::vector<int> changed;
                        evaluate_graph_partial(ind, i, base_cache, trial_cache, changed);
                        
                        dg.update_nodes(changed, base_cache, trial_cache);
                        Eigen::VectorXd w_trial;
                        if (dg.solve_ridge(ridge_lambda, w_trial)) {
                            double trial_mse = compute_mse_from_trial(w_trial, trial_cache);
                            if (snap_accepts(snap_baseline_mse, trial_mse, SnapTier::Integer) &&
                                trial_mse < best_snap_mse) {
                                best_snap_omega = nearest_int;
                                best_snap_mse = trial_mse;
                            }
                        }
                        // REVERT
                        dg.update_nodes(changed, trial_cache, base_cache);
                    }
                    
                    node.omega = best_snap_omega;
                    if (best_snap_omega != original_omega) {
                        // Permanent update
                        std::vector<int> changed;
                        std::vector<Eigen::ArrayXd> new_cache;
                        evaluate_graph_partial(ind, i, base_cache, new_cache, changed);
                        dg.update_nodes(changed, base_cache, new_cache);
                        base_cache = std::move(new_cache);
                        
                        // Sync weights/bias
                        Eigen::VectorXd w_final;
                        if (dg.solve_ridge(ridge_lambda, w_final)) {
                            for (int f = 0; f < n_feat; ++f) ind.output_weights[f] = w_final(f);
                            ind.output_bias = w_final(n_feat);
                        }
                        evaluate_fitness_with_penalty(ind, X_, y_, n_samples);
                        snap_baseline_mse = objective_mse(ind);
                    }
                    
                    // Try snapping phi
                    double original_phi = node.phi;
                    double best_snap_phi = original_phi;
                    best_snap_mse = std::numeric_limits<double>::infinity();
                    
                    for (int si = 0; si < n_snap_phi; ++si) {
                        double candidate = snap_candidates_phi[si];
                        if (std::abs(original_phi - candidate) > 0.3) continue;
                        
                        node.phi = candidate;
                        
                        // Finding 3: Incremental Trial
                        std::vector<Eigen::ArrayXd> trial_cache;
                        std::vector<int> changed;
                        evaluate_graph_partial(ind, i, base_cache, trial_cache, changed);
                        
                        dg.update_nodes(changed, base_cache, trial_cache);
                        Eigen::VectorXd w_trial;
                        if (dg.solve_ridge(ridge_lambda, w_trial)) {
                            double trial_mse = compute_mse_from_trial(w_trial, trial_cache);
                            if (snap_accepts(snap_baseline_mse, trial_mse, classify_phi_tier(candidate)) &&
                                trial_mse < best_snap_mse) {
                                best_snap_phi = candidate;
                                best_snap_mse = trial_mse;
                            }
                        }
                        // REVERT
                        dg.update_nodes(changed, trial_cache, base_cache);
                    }
                    node.phi = best_snap_phi;
                    if (best_snap_phi != original_phi) {
                        // Permanent update
                        std::vector<int> changed;
                        std::vector<Eigen::ArrayXd> new_cache;
                        evaluate_graph_partial(ind, i, base_cache, new_cache, changed);
                        dg.update_nodes(changed, base_cache, new_cache);
                        base_cache = std::move(new_cache);
                        
                        // Sync weights/bias
                        Eigen::VectorXd w_final;
                        if (dg.solve_ridge(ridge_lambda, w_final)) {
                            for (int f = 0; f < n_feat; ++f) ind.output_weights[f] = w_final(f);
                            ind.output_bias = w_final(n_feat);
                        }
                        evaluate_fitness_with_penalty(ind, X_, y_, n_samples);
                        snap_baseline_mse = objective_mse(ind);
                    }
                }
            }
            
            // 6a.2 Exp omega/phi snapping: exp(omega*x + phi)
            // Key snaps: omega ∈ {-3, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 3}
            for (int i = 0; i < static_cast<int>(ind.nodes.size()); ++i) {
                if (ind.nodes[i].type != NodeType::Unary) continue;
                if (ind.nodes[i].unary_op != UnaryOp::Exp) continue;
                if (i < 0 || i >= static_cast<int>(snap_active_unary.size()) || !snap_active_unary[static_cast<size_t>(i)]) continue;
                
                auto& node = ind.nodes[i];
                
                // Snap omega for Exp
                const double snap_exp_omega[] = {-3.0, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 3.0};
                const int n_snap_exp_omega = sizeof(snap_exp_omega) / sizeof(snap_exp_omega[0]);
                
                double original_omega = node.omega;
                double best_snap_omega = original_omega;
                double best_snap_mse = std::numeric_limits<double>::infinity();
                
                    for (int si = 0; si < n_snap_exp_omega; ++si) {
                        double candidate = snap_exp_omega[si];
                        if (std::abs(original_omega - candidate) > 0.3) continue;
                        
                        node.omega = candidate;
                        
                        // Finding 3: Incremental Trial
                        std::vector<Eigen::ArrayXd> trial_cache;
                        std::vector<int> changed;
                        evaluate_graph_partial(ind, i, base_cache, trial_cache, changed);
                        
                        dg.update_nodes(changed, base_cache, trial_cache);
                        Eigen::VectorXd w_trial;
                        if (dg.solve_ridge(ridge_lambda, w_trial)) {
                            double trial_mse = compute_mse_from_trial(w_trial, trial_cache);
                            if (snap_accepts(snap_baseline_mse, trial_mse, classify_omega_tier(candidate)) &&
                                trial_mse < best_snap_mse) {
                                best_snap_omega = candidate;
                                best_snap_mse = trial_mse;
                            }
                        }
                        // REVERT
                        dg.update_nodes(changed, trial_cache, base_cache);
                    }
                    double nearest_int_e = std::round(original_omega);
                    // Also try nearest integer
                    if (std::abs(nearest_int_e) >= 1.0 && std::abs(nearest_int_e) <= 5.0 && 
                        std::abs(original_omega - nearest_int_e) <= 0.3) {
                        node.omega = nearest_int_e;
                        
                        // Finding 3: Incremental Trial
                        std::vector<Eigen::ArrayXd> trial_cache;
                        std::vector<int> changed;
                        evaluate_graph_partial(ind, i, base_cache, trial_cache, changed);
                        
                        dg.update_nodes(changed, base_cache, trial_cache);
                        Eigen::VectorXd w_trial;
                        if (dg.solve_ridge(ridge_lambda, w_trial)) {
                            double trial_mse = compute_mse_from_trial(w_trial, trial_cache);
                            if (snap_accepts(snap_baseline_mse, trial_mse, SnapTier::Integer) &&
                                trial_mse < best_snap_mse) {
                                best_snap_omega = nearest_int_e;
                                best_snap_mse = trial_mse;
                            }
                        }
                        // REVERT
                        dg.update_nodes(changed, trial_cache, base_cache);
                    }
                    
                    node.omega = best_snap_omega;
                    if (best_snap_omega != original_omega) {
                        // Permanent update
                        std::vector<int> changed;
                        std::vector<Eigen::ArrayXd> new_cache;
                        evaluate_graph_partial(ind, i, base_cache, new_cache, changed);
                        dg.update_nodes(changed, base_cache, new_cache);
                        base_cache = std::move(new_cache);
                        
                        // Sync weights/bias
                        Eigen::VectorXd w_final;
                        if (dg.solve_ridge(ridge_lambda, w_final)) {
                            for (int f = 0; f < n_feat; ++f) ind.output_weights[f] = w_final(f);
                            ind.output_bias = w_final(n_feat);
                        }
                        evaluate_fitness_with_penalty(ind, X_, y_, n_samples);
                        snap_baseline_mse = objective_mse(ind);
                    }
                
                // Snap phi for Exp
                const double snap_exp_phi[] = {0.0, -1.0, -0.5, 0.5, 1.0};
                const int n_snap_exp_phi = sizeof(snap_exp_phi) / sizeof(snap_exp_phi[0]);
                
                double original_phi = node.phi;
                double best_snap_phi = original_phi;
                best_snap_mse = std::numeric_limits<double>::infinity();
                
                    for (int si = 0; si < n_snap_exp_phi; ++si) {
                        double candidate = snap_exp_phi[si];
                        if (std::abs(original_phi - candidate) > 0.3) continue;
                        
                        node.phi = candidate;
                        
                        // Finding 3: Incremental Trial
                        std::vector<Eigen::ArrayXd> trial_cache;
                        std::vector<int> changed;
                        evaluate_graph_partial(ind, i, base_cache, trial_cache, changed);
                        
                        dg.update_nodes(changed, base_cache, trial_cache);
                        Eigen::VectorXd w_trial;
                        if (dg.solve_ridge(ridge_lambda, w_trial)) {
                            double trial_mse = compute_mse_from_trial(w_trial, trial_cache);
                            if (snap_accepts(snap_baseline_mse, trial_mse, classify_phi_tier(candidate)) &&
                                trial_mse < best_snap_mse) {
                                best_snap_phi = candidate;
                                best_snap_mse = trial_mse;
                            }
                        }
                        // REVERT
                        dg.update_nodes(changed, trial_cache, base_cache);
                    }
                    node.phi = best_snap_phi;
                    if (best_snap_phi != original_phi) {
                        // Permanent update
                        std::vector<int> changed;
                        std::vector<Eigen::ArrayXd> new_cache;
                        evaluate_graph_partial(ind, i, base_cache, new_cache, changed);
                        dg.update_nodes(changed, base_cache, new_cache);
                        base_cache = std::move(new_cache);
                        
                        // Sync weights/bias
                        Eigen::VectorXd w_final;
                        if (dg.solve_ridge(ridge_lambda, w_final)) {
                            for (int f = 0; f < n_feat; ++f) ind.output_weights[f] = w_final(f);
                            ind.output_bias = w_final(n_feat);
                        }
                        evaluate_fitness_with_penalty(ind, X_, y_, n_samples);
                        snap_baseline_mse = objective_mse(ind);
                    }
            }
            
            // 6a.5 Trigonometric identity simplification
            // -sin(x + pi) = sin(x), -sin(x - pi) = sin(x)
            // sin(x + 2*pi) = sin(x), etc.
            for (int i = 0; i < static_cast<int>(ind.nodes.size()); ++i) {
                if (ind.nodes[i].type != NodeType::Unary) continue;
                if (ind.nodes[i].unary_op != UnaryOp::Periodic) continue;
                if (i < 0 || i >= static_cast<int>(snap_active_unary.size()) || !snap_active_unary[static_cast<size_t>(i)]) continue;
                
                auto& node = ind.nodes[i];
                double w = ind.output_weights[i];
                
                // Remove full 2*pi multiples from phi
                if (std::abs(node.phi) > kPi) {
                    double reduced = std::fmod(node.phi, 2.0 * kPi);
                    if (reduced > kPi) reduced -= 2.0 * kPi;
                    if (reduced < -kPi) reduced += 2.0 * kPi;
                    node.phi = reduced;
                }
                
                // -sin(x + pi) = sin(x): if phi ≈ ±π and weight is negative,
                // flip weight sign and zero out phi
                if (std::abs(std::abs(node.phi) - kPi) < 0.05 && w < 0) {
                    node.phi = 0.0;
                    ind.output_weights[i] = -w;  // Flip sign
                }
                // sin(x + pi) with positive weight -> -sin(x)
                else if (std::abs(std::abs(node.phi) - kPi) < 0.05 && w > 0) {
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
            snap_baseline_mse = objective_mse(ind);

            // 6a.6 Arithmetic gate snapping
            // If an arithmetic blend is already close to a discrete operator,
            // snap it only when the fitted MSE stays effectively unchanged.
            // H-05: gate on full active-subtree mask (Binary nodes are reachable
            // there; the old unary-only mask made this entire loop dead code).
            for (int i = 0; i < static_cast<int>(ind.nodes.size()); ++i) {
                if (ind.nodes[i].type != NodeType::Binary) continue;
                if (ind.nodes[i].binary_op != BinaryOp::Arithmetic) continue;
                if (i < 0 || i >= static_cast<int>(snap_active_nodes.size()) || !snap_active_nodes[static_cast<size_t>(i)]) continue;

                auto& node = ind.nodes[i];
                const double original_beta = node.beta;
                const double original_gamma = node.gamma;

                struct ArithmeticSnapCandidate {
                    double beta;
                    double gamma;
                    SnapTier tier;
                };

                const ArithmeticSnapCandidate candidates[] = {
                    {1.0, 1.0, SnapTier::Integer},   // add
                    {2.0, 1.0, SnapTier::Integer},   // mul
                    {2.0, -1.0, SnapTier::Integer},  // div-like arithmetic gate
                    {1.0, -1.0, SnapTier::Integer},  // sub
                };

                double best_beta = original_beta;
                double best_gamma = original_gamma;
                double best_snap_mse = std::numeric_limits<double>::infinity();

                for (const auto& candidate : candidates) {
                    double dist = std::sqrt(
                        (original_beta - candidate.beta) * (original_beta - candidate.beta) +
                        (original_gamma - candidate.gamma) * (original_gamma - candidate.gamma)
                    );
                    if (dist > 0.35) continue;

                    node.beta = candidate.beta;
                    node.gamma = candidate.gamma;

                    std::vector<Eigen::ArrayXd> trial_cache;
                    evaluate_graph(ind, X_, n_samples, trial_cache);
                    solve_output_weights(ind, trial_cache);
                    evaluate_fitness_with_penalty(ind, X_, y_, n_samples);

                    if (snap_accepts(snap_baseline_mse, objective_mse(ind), candidate.tier) &&
                        objective_mse(ind) < best_snap_mse) {
                        best_beta = candidate.beta;
                        best_gamma = candidate.gamma;
                        best_snap_mse = objective_mse(ind);
                    }
                }

                node.beta = best_beta;
                node.gamma = best_gamma;
                if (best_beta != original_beta || best_gamma != original_gamma) {
                    std::vector<Eigen::ArrayXd> updated_cache;
                    evaluate_graph(ind, X_, n_samples, updated_cache);
                    solve_output_weights(ind, updated_cache);
                    evaluate_fitness_with_penalty(ind, X_, y_, n_samples);
                    snap_baseline_mse = objective_mse(ind);
                } else {
                    node.beta = original_beta;
                    node.gamma = original_gamma;
                }
            }
            
            // Build node output cache for 6b/6c - weight/bias snapping only
            // changes output coefficients, not node params, so we can compute
            // MSE from cached outputs instead of re-evaluating the full graph.
            std::vector<Eigen::ArrayXd> snap_base_cache;
            evaluate_graph(ind, X_, n_samples, snap_base_cache);
            
            // Helper: compute MSE from cached node outputs + current weights/bias.
            // Used for 6b/6c where only output weights change, not node parameters.
            auto mse_from_cache = [&](const IndividualGraph& graph,
                                      const std::vector<Eigen::ArrayXd>& cache) -> double {
                int ns = static_cast<int>(y_.size());
                Eigen::ArrayXd pred = Eigen::ArrayXd::Constant(ns, graph.output_bias);
                for (int f = 0; f < static_cast<int>(graph.output_weights.size()); ++f) {
                    if (std::abs(graph.output_weights[f]) > 1e-15 &&
                        f < static_cast<int>(cache.size()) && cache[f].size() == ns) {
                        pred += graph.output_weights[f] * cache[f];
                    }
                }
                double mse = residual_mse(pred, y_);
                return std::isfinite(mse) ? mse : std::numeric_limits<double>::infinity();
            };

            // 6b. Output weight snapping - using cached node outputs (no graph re-eval)
            {
                const double snap_weight_values[] = {
                    0.0, 0.25, 1.0/3.0, 0.5, 2.0/3.0, 0.75,
                    1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                    kPi, kE, std::sqrt(2.0), std::sqrt(3.0)
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
                        double rel_dist = (abs_w > 1e-8) ? std::abs(abs_w - snap_weight_values[si]) / abs_w : 1.0;
                        if (rel_dist > 0.15 && std::abs(w - candidate) > 0.3) continue;
                        
                        double original_w = ind.output_weights[i];
                        ind.output_weights[i] = candidate;
                        double trial_mse = mse_from_cache(ind, snap_base_cache);
                        
                        if (trial_mse < snap_baseline_mse * 1.05 + 1e-8 && trial_mse < best_snap_mse) {
                            best_snap_w = candidate;
                            best_snap_mse = trial_mse;
                        }
                        ind.output_weights[i] = original_w; // Restore
                    }
                    
                    if (best_snap_w != w) {
                        ind.output_weights[i] = best_snap_w;
                        evaluate_fitness_with_penalty(ind, X_, y_, n_samples); // Update fitness/raw_mse
                        snap_baseline_mse = objective_mse(ind);
                    }
                }
            }
            
            // 6c. Output bias snapping - using cached node outputs (no graph re-eval)
            {
                double bias = ind.output_bias;
                if (std::abs(bias) > 1e-6) {
                    const double snap_bias_values[] = {
                        0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0,
                        -0.25, -0.5, -0.75, -1.0, -1.5, -2.0, -3.0, -4.0, -5.0,
                        kPi, -kPi, kE, -kE
                    };
                    const int n_snap_b = sizeof(snap_bias_values) / sizeof(snap_bias_values[0]);
                    
                    double best_snap_b = bias;
                    double best_snap_mse = snap_baseline_mse;
                    
                    for (int si = 0; si < n_snap_b; ++si) {
                        double candidate = snap_bias_values[si];
                        if (std::abs(bias - candidate) > 0.5) continue;
                        
                        ind.output_bias = candidate;
                        double trial_mse = mse_from_cache(ind, snap_base_cache);
                        
                        if (trial_mse < snap_baseline_mse * 1.05 + 1e-8 && trial_mse < best_snap_mse) {
                            best_snap_b = candidate;
                            best_snap_mse = trial_mse;
                        }
                    }
                    // Also try nearest integer
                    double nearest_int = std::round(bias);
                    if (std::abs(bias - nearest_int) <= 0.3) {
                        ind.output_bias = nearest_int;
                        double trial_mse = mse_from_cache(ind, snap_base_cache);
                        if (trial_mse < snap_baseline_mse * 1.05 + 1e-8 && trial_mse < best_snap_mse) {
                            best_snap_b = nearest_int;
                            best_snap_mse = trial_mse;
                        }
                    }
                    
                    if (best_snap_b != bias) {
                        ind.output_bias = best_snap_b;
                        evaluate_fitness_with_penalty(ind, X_, y_, n_samples);
                    }
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

    // -- P5: NSGA-II Non-Dominated Sort ------------------------------------
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
                // §3.102: non-finite raw_mse never earns rank 0 on
                // complexity/age alone; use (is_finite, mse) as the key.
                double mse_i = pop[i].raw_mse, mse_j = pop[j].raw_mse;
                const bool fin_i = std::isfinite(mse_i);
                const bool fin_j = std::isfinite(mse_j);
                int comp_i = pop[i].active_complexity(), comp_j = pop[j].active_complexity();
                int age_i = pop[i].age, age_j = pop[j].age;

                bool i_leq_j;
                bool i_lt_j;
                bool j_leq_i;
                bool j_lt_i;
                if (fin_i && fin_j) {
                    i_leq_j = (mse_i <= mse_j) && (comp_i <= comp_j) && (age_i <= age_j);
                    i_lt_j  = (mse_i < mse_j) || (comp_i < comp_j) || (age_i < age_j);
                    j_leq_i = (mse_j <= mse_i) && (comp_j <= comp_i) && (age_j <= age_i);
                    j_lt_i  = (mse_j < mse_i) || (comp_j < comp_i) || (age_j < age_i);
                } else if (fin_i && !fin_j) {
                    // Finite always dominates non-finite (strict on mse).
                    i_leq_j = true; i_lt_j = true;
                    j_leq_i = false; j_lt_i = false;
                } else if (!fin_i && fin_j) {
                    i_leq_j = false; i_lt_j = false;
                    j_leq_i = true; j_lt_i = true;
                } else {
                    // Both non-finite: neither dominates on mse; fall back to
                    // complexity/age only so invalid models cannot reach rank 0
                    // unless nothing finite exists, and crowding keeps them grouped.
                    i_leq_j = (comp_i <= comp_j) && (age_i <= age_j);
                    i_lt_j  = (comp_i < comp_j) || (age_i < age_j);
                    j_leq_i = (comp_j <= comp_i) && (age_j <= age_i);
                    j_lt_i  = (comp_j < comp_i) || (age_j < age_i);
                    // Demote both: require strict finite win for rank 0.
                    // Handled by rank reassignment below; keep pairwise neutral
                    // when equal so they share the worst front.
                }
                bool i_dom_j = i_leq_j && i_lt_j;
                
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

    // -- P5: Crowding Distance Assignment ----------------------------------
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
                      return a->active_complexity() < b->active_complexity();
                  });
        front.front()->crowding_distance = 1e18;
        front.back()->crowding_distance = 1e18;
        double comp_range = front.back()->active_complexity() - front.front()->active_complexity();
        if (comp_range > 1e-15) {
            for (int i = 1; i < n - 1; ++i) {
                front[i]->crowding_distance += static_cast<double>(
                    front[i+1]->active_complexity() - front[i-1]->active_complexity()) / comp_range;
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
                front[i]->crowding_distance += static_cast<double>(
                    front[i+1]->age - front[i-1]->age) / age_range;
            }
        }
    }

    // -- P5: NSGA-II Selection --------------------------------------------─
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

    // -- P6: Single-generation evolution step (for island model) ----------─
    // -- P6: Single-generation evolution step (used by island model and run()) ---
    void evolve_one_generation(int gen) {
        evaluate_population();
        trace_event("generation.post_eval", gen);

        bool in_topology_phase = config_.use_staged_schedule && (gen < config_.topology_phase_generations);
        double scheduled_mutation_rate = current_structural_mutation_rate_;
        if (in_topology_phase) {
            scheduled_mutation_rate = std::min(1.0, current_structural_mutation_rate_ * config_.topology_phase_mutation_boost);
        }

        // P5: NSGA-II selection or standard sort
        if (config_.use_nsga2) {
            // Create offspring pool
            std::vector<IndividualGraph> combined;
            combined.reserve(population_.size() + static_cast<size_t>(config_.pop_size));
            for (auto& ind : population_) {
                ind.age++;  // AFPO: existing population ages
                combined.push_back(ind);
            }

            std::uniform_int_distribution<int> p_dist(0, std::max(0, static_cast<int>(population_.size()) - 1));
            std::uniform_real_distribution<double> co(0.0, 1.0);
            const double macro_rate = std::clamp(config_.macro_mutation_rate, 0.0, 0.9);

            for (int i = 0; i < config_.pop_size; ++i) {
                IndividualGraph child;
                double roll = co(rng_);
                if (roll < macro_rate) {
                    child = macro_mutate(population_[p_dist(rng_)]);
                } else if (config_.elite_size >= 2 && roll < macro_rate + config_.crossover_rate) {
                    int p1 = p_dist(rng_), p2 = p_dist(rng_);
                    while (p2 == p1) p2 = p_dist(rng_);
                    child = crossover_with_retry(population_[p1], population_[p2], 3);
                    child = mutate_lamarckian(child, scheduled_mutation_rate * 0.3);
                } else {
                    child = mutate_lamarckian(population_[p_dist(rng_)], scheduled_mutation_rate);
                }
                child.age = 0;  // AFPO: new children start young

                bool do_refine = in_topology_phase
                    ? (gen % config_.topology_refine_interval == 0)
                    : (gen % 5 == 0);
                if (do_refine) {
                    refine_constants(child);
                } else {
                    evaluate_fitness_with_penalty(child, X_, y_, static_cast<int>(y_.size()));
                }
                combined.push_back(std::move(child));
            }

            // NSGA-II selection from combined pool
            population_ = nsga2_select(combined, config_.pop_size);

            // Track best (fitness primary, raw_mse secondary)
            for (auto& ind : population_) {
                consider_champion(ind);
            }

            if (gen % 10 == 9) {
                for (int i = 0; i < std::min(5, config_.elite_size) && i < static_cast<int>(population_.size()); ++i) {
                    refine_inner_params(population_[i]);
                }
            }

            trace_event("generation.post_reproduce", gen);
            return;
        }

        // Standard single-objective sort (raw_mse tie-break)
        std::sort(population_.begin(), population_.end(),
                  [](const IndividualGraph& a, const IndividualGraph& b) {
                      return is_better_champion(a, b);
                  });

        // Track best (fitness primary, raw_mse secondary)
        if (!population_.empty()) {
            consider_champion(population_[0]);
        }

        // Dynamic mutation decay based on plateauing
        if (best_overall_.fitness >= best_mse_history_ * 0.99) {
            plateau_counter_++;
        } else {
            plateau_counter_ = 0;
            best_mse_history_ = best_overall_.fitness;
        }

        if (plateau_counter_ > 50) {
            current_structural_mutation_rate_ = std::max(0.05, current_structural_mutation_rate_ * 0.9);
            plateau_counter_ = 0;
        }

        recent_best_.push_back(best_overall_.fitness);
        if (recent_best_.size() > static_cast<size_t>(config_.stagnation_window)) {
            recent_best_.erase(recent_best_.begin());
        }

        bool should_restart = false;
        if (config_.use_adaptive_restart && recent_best_.size() >= static_cast<size_t>(config_.stagnation_window)) {
            double window_improvement = recent_best_.front() - recent_best_.back();
            double diversity = population_diversity_ratio(population_);
            should_restart = (window_improvement < config_.stagnation_min_improvement) && (diversity < config_.diversity_floor);
        }

        // Create next generation
        std::vector<IndividualGraph> next_gen;
        next_gen.reserve(config_.pop_size);

        // Elitism ensures top survivors pass verbatim (age incremented)
        int elite_count = std::min(config_.elite_size, static_cast<int>(population_.size()));
        for (int i = 0; i < elite_count; ++i) {
            IndividualGraph elite = population_[i];
            elite.age++;  // AFPO: survivors age
            next_gen.push_back(std::move(elite));
        }

        int num_explorers = static_cast<int>(config_.pop_size * config_.explorer_fraction);
        int main_pop_target = std::max(elite_count, config_.pop_size - num_explorers);

        std::uniform_int_distribution<int> parent_dist(0, std::max(0, elite_count - 1));
        std::uniform_real_distribution<double> coin(0.0, 1.0);
        const double macro_rate = std::clamp(config_.macro_mutation_rate, 0.0, 0.9);

        // Fill remainder of main population with crossover + mutated offspring
        while (static_cast<int>(next_gen.size()) < main_pop_target) {
            IndividualGraph child;
            double roll = coin(rng_);

            if (roll < macro_rate) {
                int parent_idx = parent_dist(rng_);
                child = macro_mutate(population_[parent_idx]);
            } else if (elite_count >= 2 && roll < macro_rate + config_.crossover_rate) {
                int p1 = parent_dist(rng_);
                int p2 = parent_dist(rng_);
                while (p2 == p1) p2 = parent_dist(rng_);
                child = crossover_with_retry(population_[p1], population_[p2], 3);
                child = mutate_lamarckian(child, scheduled_mutation_rate * 0.3);
            } else {
                int parent_idx = tournament_select();
                child = mutate_lamarckian(population_[parent_idx], scheduled_mutation_rate);
            }

            child.age = 0;  // AFPO: new children start young

            bool do_refine = in_topology_phase
                ? (gen % config_.topology_refine_interval == 0)
                : (gen % 5 == 0);
            if (do_refine) {
                refine_constants(child);
            } else {
                evaluate_fitness_with_penalty(child, X_, y_, static_cast<int>(y_.size()));
            }
            next_gen.push_back(std::move(child));
        }

        // Fill explorer population
        while (static_cast<int>(next_gen.size()) < config_.pop_size) {
            int parent_idx = tournament_select();
            double explorer_rate = std::min(1.0, scheduled_mutation_rate * config_.explorer_mutation_multiplier);
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
                evaluate_fitness_with_penalty(explorer, X_, y_, static_cast<int>(y_.size()));
            }
            next_gen.push_back(std::move(explorer));
        }

        population_ = std::move(next_gen);

        if (should_restart) {
            inject_restarts(population_);
            current_structural_mutation_rate_ = std::min(1.0, current_structural_mutation_rate_ * config_.post_restart_mutation_boost);
            trace_event("population.restart", gen);
        }

        // Periodic inner-param refinement on top elite only (every 10 gens)
        if (gen % 10 == 9) {
            for (int i = 0; i < std::min(5, config_.elite_size) && i < static_cast<int>(population_.size()); ++i) {
                refine_inner_params(population_[i]);
            }
        }

        trace_event("generation.post_reproduce", gen);
    }

    // -- P7: Dimensional Analysis Penalty ----------------------------------
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

                    if (node.unary_op == UnaryOp::Power || node.unary_op == UnaryOp::IntPow) {
                        // x^p: multiply units by p
                        double exp_val = node.p;
                        if (node.unary_op == UnaryOp::IntPow) {
                            exp_val = static_cast<double>(std::clamp(static_cast<int>(std::round(node.p)), 2, 6));
                        }
                        for (int d = 0; d < n_dims; ++d)
                            node_units[i][d] = child_u[d] * exp_val;
                    } else if (node.unary_op == UnaryOp::Abs) {
                        // abs(x) preserves units of x
                        node_units[i] = child_u;
                    } else {
                        // sin, exp, log: argument must be dimensionless
                        // Result is dimensionless too
                        // Penalty for non-zero child units
                        // (units stay as zero - result is dimensionless)
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
                        // §3.105: soft Arithmetic blends +,-,*,soft-div; the old
                        // beta<1.5 test lumped every non-add blend into
                        // multiplication and left divide unrepresentable.
                        // Propagate units of the NEAREST discrete mode under
                        // the same distance metric as arithmetic_soft_weights.
                        // 0=add 1=mul 2=div 3=sub.
                        const double d_add = (node.beta - 1.0) * (node.beta - 1.0) + (node.gamma - 1.0) * (node.gamma - 1.0);
                        const double d_mul = (node.beta - 2.0) * (node.beta - 2.0) + (node.gamma - 1.0) * (node.gamma - 1.0);
                        const double d_div = (node.beta - 2.0) * (node.beta - 2.0) + (node.gamma + 1.0) * (node.gamma + 1.0);
                        const double d_sub = (node.beta - 1.0) * (node.beta - 1.0) + (node.gamma + 1.0) * (node.gamma + 1.0);
                        int mode = 0;
                        double best = d_add;
                        if (d_mul < best) { best = d_mul; mode = 1; }
                        if (d_div < best) { best = d_div; mode = 2; }
                        if (d_sub < best) { best = d_sub; mode = 3; }
                        if (mode == 0 || mode == 3) {
                            // Addition/subtraction: units must match, result = same
                            node_units[i] = left_u;
                        } else if (mode == 1) {
                            // Multiplication: add exponents
                            for (int d = 0; d < n_dims; ++d)
                                node_units[i][d] = left_u[d] + right_u[d];
                        } else {
                            // Soft-division x/sqrt(1+y^2): units of x minus
                            // units of a dimensionless-bounded scale; to first
                            // order this is division-like. Represent as
                            // left - right (exact for the hard-div limit).
                            for (int d = 0; d < n_dims; ++d)
                                node_units[i][d] = left_u[d] - right_u[d];
                        }
                    } else if (node.binary_op == BinaryOp::Division) {
                        // Division: subtract exponents
                        for (int d = 0; d < n_dims; ++d)
                            node_units[i][d] = left_u[d] - right_u[d];
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
            if (node.type == NodeType::Unary && node.unary_op != UnaryOp::Power && node.unary_op != UnaryOp::IntPow && node.unary_op != UnaryOp::Abs) {
                // sin/exp/log argument must be dimensionless.
                // §3.106 note: omega/phi carry no unit metadata in this
                // representation (treated as dimensionless tuning constants),
                // so enforcement is via the child's units: any dimensioned
                // argument is penalized regardless of omega/phi values.
                if (node.left_child >= 0 && node.left_child < n_nodes) {
                    for (int d = 0; d < n_dims; ++d)
                        penalty += node_units[node.left_child][d] * node_units[node.left_child][d];
                }
            }
            if (node.type == NodeType::Binary && node.binary_op == BinaryOp::Arithmetic) {
                // Addition/subtraction: left and right units must match.
                // §3.105: cover the sub-like blend corner too, matching the
                // nearest-mode propagation above (old code checked add only).
                const double d_add = (node.beta - 1.0) * (node.beta - 1.0) + (node.gamma - 1.0) * (node.gamma - 1.0);
                const double d_sub = (node.beta - 1.0) * (node.beta - 1.0) + (node.gamma + 1.0) * (node.gamma + 1.0);
                const double d_mul = (node.beta - 2.0) * (node.beta - 2.0) + (node.gamma - 1.0) * (node.gamma - 1.0);
                const double d_div = (node.beta - 2.0) * (node.beta - 2.0) + (node.gamma + 1.0) * (node.gamma + 1.0);
                const double m = std::min(std::min(d_add, d_sub), std::min(d_mul, d_div));
                const bool is_add_sub = (m == d_add) || (m == d_sub);
                if (is_add_sub) {
                    if (node.left_child >= 0 && node.right_child >= 0 &&
                        node.left_child < n_nodes && node.right_child < n_nodes) {
                        for (int d = 0; d < n_dims; ++d) {
                            double diff = node_units[node.left_child][d] - node_units[node.right_child][d];
                            penalty += diff * diff;
                        }
                    }
                }
            }
        }

        // Output unit check: every term contributing to the weighted sum must
        // carry the target units. §3.107: use the same kOutputWeightActive
        // threshold as evaluation (the old 1e-4 let sub-threshold-but-live
        // terms escape), and check the bias too (dimensionless constant —
        // penalized whenever the target has units).
        if (!config_.output_units.empty()) {
            // Check each active node's units against output units
            for (int i = 0; i < n_nodes && i < static_cast<int>(graph.output_weights.size()); ++i) {
                if (std::abs(graph.output_weights[i]) > kOutputWeightActive) {
                    for (int d = 0; d < n_dims && d < static_cast<int>(config_.output_units.size()); ++d) {
                        double diff = node_units[i][d] - config_.output_units[d];
                        penalty += diff * diff;
                    }
                }
            }
            if (std::abs(graph.output_bias) > kOutputWeightActive) {
                for (int d = 0; d < n_dims && d < static_cast<int>(config_.output_units.size()); ++d) {
                    double diff = config_.output_units[d];
                    penalty += diff * diff;
                }
            }
        }

        return penalty;
    }
};

} // namespace sr
