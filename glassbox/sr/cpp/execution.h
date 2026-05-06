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
    template <typename EvaluateFunc>
    void evaluate_population(std::vector<IndividualGraph>& population, EvaluateFunc&& eval_func) {
        gen_cache_.clear();
        
        // 1. Thread-local caching to prevent locking during parallel evaluation
        std::vector<SubtreeCache> thread_caches(omp_get_max_threads());

        // 2. Parallel evaluation mapping
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < static_cast<int>(population.size()); ++i) {
            int tid = omp_get_thread_num();
            // Call the user-provided evaluation function, passing thread-local cache
            eval_func(population[i], thread_caches[tid]);
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