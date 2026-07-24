#pragma once

#include <algorithm>
#include <vector>

#include <omp.h>

#include "ast.h"

namespace sr {

// Parallel evaluation and generation-level subtree caching.
// Does not own training data: callers pass X/y into the eval lambda.
// (Engine only merges thread-local SubtreeCache maps.)
class ParallelExecutionEngine {
public:
    ParallelExecutionEngine() = default;

    // Evaluate a population in parallel, mutating each individual's fitness
    // via the caller-provided eval_func. Builds a shared subtree cache without
    // thread contention (thread-local caches merged serially).
    // num_threads: 0 => omp_get_max_threads(). When already inside an outer
    // OpenMP region (island parallel), prefer caller-provided inner budget to
    // avoid process-wide omp_set_num_threads races (E7 / tracker E5).
    template <typename EvaluateFunc>
    void evaluate_population(std::vector<IndividualGraph>& population,
                             EvaluateFunc&& eval_func,
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
                // try_emplace prevents overwriting if multiple threads found
                // the same subtree
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

private:
    // Shared generation cache storing evaluated subtree basis functions
    SubtreeCache gen_cache_;
};

} // namespace sr
