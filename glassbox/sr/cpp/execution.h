#pragma once

#include <vector>
#include <memory>
#include <omp.h>
#include "ast.h"
#include "eval.h"
#include <chrono>

namespace sr {

// Fwd declarations from evolution if they aren't here
struct EvolutionConfig;

// An engine responsible exclusively for parallel evaluation and caching.
class ParallelExecutionEngine {
private:
    const std::vector<Eigen::ArrayXd>& X_;
    const Eigen::ArrayXd& y_;
    int n_samples_;

    // Shared generation cache storing evaluated subtree basis functions
    SubtreeCache gen_cache_;

public:
    ParallelExecutionEngine(const std::vector<Eigen::ArrayXd>& X, const Eigen::ArrayXd& y)
        : X_(X), y_(y), n_samples_(static_cast<int>(y.size())) {}

    // Evaluate a population safely in parallel, returning nothing 
    // but mutating each individual's fitness.
    // Also builds a shared subtree cache cleanly without thread contention.
    // num_threads: 0 => omp_get_max_threads(). When already inside an outer
    // OpenMP region (island parallel), prefer caller-provided inner budget to
    // avoid process-wide omp_set_num_threads races (E7 / tracker E5).
    template <typename EvaluateFunc>
    void evaluate_population(std::vector<IndividualGraph>& population, EvaluateFunc&& eval_func,
                             int num_threads = 0) {
        gen_cache_.clear();

        int nt = num_threads > 0 ? num_threads : std::max(1, omp_get_max_threads());
        // If already nested under another parallel region and caller did not
        // request a budget, stay serial to avoid oversubscription.
        if (num_threads <= 0 && omp_in_parallel()) {
            nt = 1;
        }
        nt = std::max(1, nt);

        // 1. Thread-local caching to prevent locking during parallel evaluation
        std::vector<SubtreeCache> thread_caches(static_cast<size_t>(nt));

        // 2. Parallel evaluation mapping
        #pragma omp parallel for schedule(dynamic) num_threads(nt)
        for (int i = 0; i < static_cast<int>(population.size()); ++i) {
            int tid = omp_get_thread_num();
            if (tid < 0 || tid >= nt) tid = 0;
            // Call the user-provided evaluation function, passing thread-local cache
            eval_func(population[i], thread_caches[static_cast<size_t>(tid)]);
        }

        // 3. Serial merge of caches back into the global generation cache
        for (auto& tc : thread_caches) {
            for (auto& pair : tc) {
                // try_emplace prevents overwriting if multiple threads found the same subtree
                gen_cache_.try_emplace(pair.first, std::move(pair.second));
            }
        }
    }

    const SubtreeCache& get_gen_cache() const {
        return gen_cache_;
    }

    void clear_cache() {
        gen_cache_.clear();
    }
};

} // namespace sr