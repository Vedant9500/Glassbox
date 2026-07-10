# Search Optimization Plan

Last updated: 2026-06-03.

This plan focuses on improving Glassbox symbolic-regression search without
rewriting the system around Python ASTs. The current architecture already has a
fast path, universal proposer, specialist composition, and a C++ graph/AST
evolution backend. The best next work is to improve candidate selection,
bounded sparse search, and display-aware acceptance.

## Implementation Status

Implemented in the first optimization pass:

- benchmark run metadata, `--seed`, `--compare-to`, Python ABI, C++ `_core`
  availability, and diagnostic counters in JSON/Markdown reports
- shared display-aware candidate governor in `scripts/benchmark_common.py`
- fast-path candidate ranking with display/holdout/risk-aware scoring
- full-domain and displayed-formula validation before marking exact matches
- bounded sparse beam fallback when exhaustive exact-match combinations exceed
  the configured cap
- Python fast-path semantic deduplication of governed candidate curves
- C++ Pareto-front post-processing through the shared display governor
- direct transform probes for `log(a*x+b)`, `exp(a*x+b)`, and shifted
  exponentials
- lightweight decomposition probes for additive, multiplicative, and rational
  univariate seeds, routed into benchmark guided evolution and sklearn
  specialist candidate pools
- small fidelity-guarded canonical rewrite layer for common exp/log and trig
  identities

Still future work:

- online semantic diversity inside the native C++ population loop
- broader AI-Feynman-style separability/symmetry probes for multivariate
  formulas
- full e-graph/equality-saturation style canonicalization, if the small rewrite
  layer continues proving useful

## Executive Summary

The current benchmark profile is strong but still shows repeated failure modes:

- low raw MSE but ugly or invalid displayed formulas
- surrogate formulas with too many fractional powers or harmonic terms
- brute-force exact-match skips when basis combinations exceed the cap
- C++ refinement unavailable when the active Python ABI does not match `_core`
- borderline run-to-run movement in specialist/evolution paths

The highest-value optimization is not a new tree-search engine from scratch.
Glassbox already has most of the primitives a good tree search would need:
structural hashes, subtree evaluation caches, Pareto fronts, seed graphs,
semantic duplicate cleanup, parsimony pressure, coefficient pruning, and
specialist/residual stages. The missing layer is a disciplined candidate
governor that ranks and accepts formulas using full-domain/display quality,
complexity, domain safety, residual structure, and holdout behavior.

## What Already Exists

### Fast Path

Primary file: `scripts/classifier_fast_path.py`

Current capabilities:

- classifier-guided basis construction
- exact-match search over small basis subsets
- torch/CPU/CUDA routing for exact-match batches
- LASSO/coordinate descent with Python fallback
- edge holdout diagnostics
- residual structure diagnostics
- candidate pools keyed by active basis signatures
- full-domain validation for exact symbolic shortcuts

Current gap:

- exact subset search falls back when combinations exceed
  `exact_match_max_combos`, e.g. `combos=260130 > cap=50000`
- LASSO candidates are not governed by a true Pareto frontier over display MSE,
  complexity, holdout risk, and residual suspicion
- current sparse search can prefer dense harmonic/fractional surrogates over
  simpler structures when they fit the sample grid

### C++ Evolution

Primary files:

- `glassbox/sr/cpp/ast.h`
- `glassbox/sr/cpp/evolution.h`
- `glassbox/sr/cpp/eval.h`
- `glassbox/sr/cpp/simplify_advanced.h`
- `glassbox/sr/cpp/core.cpp`

Current capabilities:

- native graph/AST representation
- structural subtree hashing
- subtree evaluation cache
- island evolution
- NSGA/Pareto-style ranking
- active-complexity parsimony pressure
- coefficient pruning
- inner parameter refinement
- semantic output-correlation cleanup
- seed graph ingestion
- native formula scoring and simplification helpers

Current gap:

- final candidate selection still mostly optimizes native/raw MSE and active
  complexity, then Python display scoring catches drift later
- semantic duplicate cleanup happens post-evolution, not as an online population
  pressure
- Pareto front reporting is not yet used as a full candidate-selection pool in
  Python with display-aware re-scoring

### Specialist And Blackbox Layers

Primary files:

- `glassbox/sr/sklearn_wrapper.py`
- `glassbox/sr/specialist_state.py`
- `glassbox/sr/blackbox_preprocessor.py`

Current capabilities:

- candidate refinement and screening
- specialist pair/composition scoring
- hot-spot/segment diagnostics
- residual symbolic stages
- vault/inception reuse
- feature reduction and interaction discovery
- seeded C++ evolution from candidate formulas

Current gap:

- specialist candidates can still win with formulas that have display/holdout
  drift
- final selection needs a stronger universal risk score shared by fast path,
  specialist, and C++ evolution

## External Research Takeaways

These ideas map well to Glassbox:

- PySR uses a multi-population evolutionary system and emphasizes Pareto-style
  model selection over complexity and loss. Glassbox already has islands and
  Pareto structures; the next step is to re-score the returned frontier with
  displayed-formula and holdout metrics.
- AI Feynman uses divide-and-conquer tests such as separability, symmetry,
  polynomial fitting, and transformations. Glassbox already has blackbox
  interaction probes and specialist states; we should add lightweight additive
  and multiplicative decomposition tests before expensive evolution.
- SINDy frames discovery as sparse selection from a candidate library. This
  matches the fast path exactly. The missing piece is a better sparse subset
  search than exhaustive pairs/triples or plain LASSO.
- Equality saturation/e-graphs are good for algebraic canonicalization and
  rewrite ordering problems, but a full e-graph engine is too heavy for the next
  step. A small post-candidate rewrite/canonicalization pass is more pragmatic.
- Genetic programming bloat work reinforces that parsimony pressure and
  multi-objective selection are necessary but not sufficient; display quality
  and holdout behavior must be part of selection, not only reporting.

References:

- PySR paper: https://arxiv.org/abs/2305.01582
- AI Feynman paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC7159912/
- SINDy paper: https://pubmed.ncbi.nlm.nih.gov/27035946/
- egg/equality saturation paper: https://arxiv.org/abs/2004.03082
- Prioritized grammar enumeration overview:
  https://researchconnect.suny.edu/en/publications/prioritized-grammar-enumeration-symbolic-regression-by-dynamic-pr/

## Recommended Optimization Roadmap

## Phase 1: Display-Aware Candidate Governor

Goal: stop weak surrogates from winning simply because raw MSE is low.

Add a shared scoring helper, likely in `scripts/benchmark_common.py` or a new
module under `glassbox/sr/`, that computes:

- `mse_raw`
- `mse_display`
- `mse_holdout`
- `complexity`
- `n_terms`
- `domain_failure_rate`
- `residual_suspicious`
- `mse_divergence_rel`
- `family_risk_score`

Then define a candidate score:

```text
score =
  display_mse
  + complexity_lambda * max(0, complexity - simple_threshold)
  + holdout_lambda * holdout_gap
  + drift_lambda * raw_display_drift
  + risk_lambda * domain_or_family_risk
```

Integration points:

- fast-path candidate pool in `fast_path_regression`
- specialist final formula selection in `GlassboxRegressor`
- C++ Pareto-front post-processing after `_core.run_evolution`

Acceptance rules:

- never mark exact unless displayed MSE is finite and below exact threshold
- reject `Piecewise` display formulas unless they evaluate cleanly and are
  strictly better than the incumbent
- penalize formulas with many protected fractional powers unless the target
  class strongly supports them
- if raw/display drift is high, keep the formula as a candidate but force
  guided evolution or residual repair

Success metrics:

- fewer `mse_display=None` rows
- fewer high-drift rows
- no reduction in tier 1-3 exact count
- improved weighted score stability across repeated full runs

## Phase 2: Replace Exhaustive Exact-Match Search With Beam/OMP Subset Search

Goal: avoid `Skipping exhaustive exact-match search` on large bases.

Current exact-match search checks all pairs/triples until the combination cap.
When the basis has ~100 terms, triples are too expensive. Replace the skip with
bounded sparse selection:

1. Standardize basis columns.
2. Rank columns by absolute correlation with `y`.
3. Run Orthogonal Matching Pursuit style greedy expansion for `k=1..K`.
4. Keep a beam of top partial supports, not just one greedy path.
5. Refit coefficients with least squares at each expansion.
6. Score candidates with the display-aware governor.

Recommended defaults:

- `beam_width=32`
- `max_terms=6` for normal runs
- `max_terms=10` only for polynomial-only or high-trust cases
- restrict expansion by operator family priors when classifier/proposer is
  confident

Why this fits:

- the fast path already builds the candidate library
- SINDy-style sparse selection is a natural match for this architecture
- it gives a useful fallback when exhaustive triples exceed the cap

Success metrics:

- exact-match skip count decreases
- tier 4-8 approximate/loose results improve without increasing runtime too much
- fewer dense harmonic/fractional LASSO surrogates

## Phase 3: Online Semantic Deduplication In Candidate Pools

Goal: reduce duplicate and near-duplicate candidate work before final cleanup.

C++ already performs semantic cleanup after evolution by comparing node outputs.
Add a lighter version to Python candidate pools and, later, to online C++
population selection.

Python fast-path version:

- evaluate candidate prediction vector on the full grid
- normalize by mean/std
- quantize to a compact signature
- group by high correlation or low relative error
- keep the candidate with lower governor score

C++ online version:

- reuse `graph_signature` and subtree cache
- add optional output-sketch signatures for elites per generation
- prevent multiple equivalent elites from occupying the beam/frontier

Success metrics:

- smaller candidate pools
- lower final formula complexity
- less repeated harmonic clutter

## Phase 4: Decomposition Probes Before Expensive Evolution

Goal: route sums, products, and rational forms into simpler searches.

Implement cheap tests inspired by AI Feynman-style divide-and-conquer:

- additive residual probe:
  - fit a dominant component
  - evaluate whether residual has a clean family signature
- multiplicative probe:
  - if `y` and candidate are sign/domain safe, test `y / f1`
- rational probe:
  - test sparse numerator/denominator libraries separately
- symmetry/even-odd probe:
  - compare `y(x)` with `y(-x)` when domain supports it

Integration points:

- before specialist residual stage in `GlassboxRegressor`
- as additional `candidate_seed_formulas`
- as C++ `seed_graphs_py`

Success metrics:

- tier 5 sums/products stabilize
- tier 6 rational/nested improves
- fewer cases where evolution approximates products with Fourier sums

## Phase 5: Pareto-Front Re-Scoring From C++ Evolution

Goal: use the native Pareto front as a candidate set, not only as diagnostics.

Current C++ can expose Pareto-front entries. Python should:

1. collect top front formulas from `_core.run_evolution`
2. simplify/snap each formula
3. evaluate displayed MSE
4. apply the candidate governor
5. choose the best display-aware candidate, not necessarily the lowest raw MSE

This directly addresses rows where raw MSE is excellent but display formula
scores worse.

Success metrics:

- lower raw/display drift
- fewer display-eval failures
- no major runtime increase because candidates are already produced

## Phase 6: Small Canonical Rewrite Layer

Goal: get e-graph-like benefit without adding a full e-graph system.

Add a deterministic candidate rewrite pass for common cases:

- `E**(a*x)` -> `exp(a*x)`
- `exp(log(z))` -> `z` when domain-safe
- `log(exp(z))` -> `z` when domain-safe
- `sin(x)*cos(x)` <-> `1/2*sin(2*x)` candidate variants
- `sin(x)^2 + cos(x)^2` -> `1`
- `x*x` -> `x^2`
- sort additive/multiplicative operands for canonical display

Do not make this the primary search. Use it as a candidate expansion and
display cleanup tool, with fidelity guard.

Success metrics:

- more human-readable final formulas
- fewer exact-by-numeric but not exact-by-display misses

## Phase 7: Reproducibility And A/B Discipline

Goal: make optimization work measurable.

Add benchmark options:

- `--seed`
- `--runs N`
- `--compare-to results/file.json`
- per-formula bucket transition report
- exact-match skip count
- C++ refinement availability count
- candidate governor rejection reasons

Make full benchmark reports include:

- active Python ABI and `_core` ABI
- whether C++ refinement was loaded
- random seed
- benchmark mode and specialist phase config

Success metrics:

- bucket movement can be explained formula-by-formula
- full-run comparisons are repeatable
- regressions are caught before optimizing the wrong layer

## Prioritized Implementation Order

1. Add benchmark comparison tooling and metadata.
2. Add candidate governor and use it in fast-path candidate ranking.
3. Replace exact-match skip with OMP/beam subset search.
4. Use C++ Pareto front as a display-scored candidate pool.
5. Add decomposition probes and route them into specialist/C++ seeds.
6. Add online semantic dedup to Python candidate pools.
7. Add small canonical rewrite layer with fidelity guard.
8. Consider larger e-graph/equality-saturation work only if the small rewrite
   layer proves valuable and maintainable.

## What Not To Do Yet

- Do not replace the whole system with Python AST tree search.
- Do not add a full e-graph dependency before proving value with a small rewrite
  layer.
- Do not optimize raw MSE without displayed-formula scoring.
- Do not widen fast-path exact thresholds to recover old-looking benchmark
  tables; that hides drift.
- Do not let proposer confidence collapse search diversity.

## Expected Impact

Near-term expected gains:

- fewer loose/fail rows caused by display drift
- better handling of large fast-path bases
- more stable tier 4-8 results
- cleaner formulas with fewer fractional-power and harmonic surrogates

The most realistic target is not a huge jump in aggregate exact percentage. It
is a reduction in unstable bucket movement and a higher fraction of formulas
whose displayed form is simple, finite, and structurally credible.
