# PhySO Noise Handling vs Glassbox

## Scope

This note reviews PhySO from the cloned upstream repo at `.tmp/PhySO` plus its docs at <https://physo.readthedocs.io/en/latest/>, then compares its noise-handling story with the current Glassbox pipeline.

## Short Answer

PhySO is good under noise mostly because it narrows the symbolic search before noise can dominate it. Its strongest protections are not exotic denoisers; they are physical constraints, uncertainty/weight-aware fitting, free-constant optimization inside the reward loop, duplicate/unphysical candidate filtering, and strong benchmark discipline.

Glassbox attacks noise differently. It uses fast analytical recovery, classifier/proposer uncertainty, validation guards, adaptive search routing, residual/specialist passes, Pareto-style candidate selection, displayed-formula scoring, and post-fit noise pruning. That gives Glassbox more machinery for "do I trust this formula?", while PhySO has a cleaner physics-first search space and a simpler weighted NRMSE objective.

## What PhySO Does

### 1. Weighted objective / uncertainty-aware fitting

PhySO exposes point weights through `y_weights` for `physo.SR` and `multi_y_weights` for Class SR. The docs say these weights are used both during free-constant optimization and reward computation, so noisy or low-confidence observations can be downweighted directly.

Evidence:

- `.tmp/PhySO/docs/source/features/doc_features_weights.md:1` introduces "Weighting points & uncertainty".
- `.tmp/PhySO/docs/source/features/doc_features_weights.md:7` says weights affect free-constant optimization and reward.
- `.tmp/PhySO/docs/source/features/doc_features_weights.md:9` documents `y_weights`.
- `.tmp/PhySO/physo/physym/reward.py:29` applies `y_weights * (y_target - y_pred)^2`.
- `.tmp/PhySO/physo/physym/reward.py:170` passes `y_weights` into `batch_optimize_constants`.

Impact: if user knows per-point uncertainty, PhySO can fit closer to maximum-likelihood weighted least squares behavior. This is a direct noise model hook. Without supplied weights, it falls back to uniform weights.

### 2. Squashed NRMSE reward

PhySO's default reward is squashed normalized RMSE:

```text
RMSE = sqrt(mean(y_weights * (y - pred)^2))
NRMSE = RMSE / std(y)
reward = 1 / (1 + NRMSE)
```

Evidence:

- `.tmp/PhySO/physo/physym/reward.py:11` defines `SquashedNRMSE`.
- `.tmp/PhySO/physo/physym/reward.py:29` computes weighted squared error.
- `.tmp/PhySO/physo/physym/reward.py:31` normalizes by target standard deviation.
- `.tmp/PhySO/physo/physym/reward.py:32` squashes into `[0, 1]`.

Impact: reward scale stays stable across target magnitudes, useful when noise level is specified as a fraction of signal RMS. But it is still squared-error based, so outliers matter unless weights handle them.

### 3. Dimensional-analysis search reduction

PhySO's biggest structural noise defense is dimensional analysis. It removes physically invalid expressions before they compete on noisy fit.

Evidence:

- `.tmp/PhySO/README.md:35` says physical units constraints reduce search space with dimensional analysis.
- `.tmp/PhySO/docs/source/features/doc_features_da.md:3` says units reduce symbolic-expression search space by enforcing dimensional constraints.
- `.tmp/PhySO/docs/source/features/doc_features_da.md:9` says `PhysicalUnitsPrior` is enabled in standard configs.
- `.tmp/PhySO/physo/physym/reward.py:111` can zero out unphysical programs.

Impact: noise can make many wrong formulas numerically fit. Units remove many of those wrong formulas before final selection. This is probably central to PhySO's noisy Feynman strength.

### 4. Priors and duplicate suppression

PhySO uses priors to bias token selection and can zero duplicate programs. Duplicate filtering can keep the lowest-complexity duplicate.

Evidence:

- `.tmp/PhySO/docs/source/features/doc_features_priors_intro.md:3` says priors tune symbol-selection probability and reduce search.
- `.tmp/PhySO/physo/physym/reward.py:57` and `:58` expose `zero_out_unphysical` and `zero_out_duplicates`.
- `.tmp/PhySO/physo/physym/reward.py:118` starts duplicate elimination.
- `.tmp/PhySO/physo/physym/reward.py:134` supports keeping lowest-complexity duplicate.

Impact: fewer redundant/noisy variants waste reward updates. Simpler equivalent forms survive more often.

### 5. Free constants optimized per candidate

PhySO optimizes free constants before final reward for candidates with constants.

Evidence:

- `.tmp/PhySO/physo/physym/reward.py:165` starts free-constant optimization.
- `.tmp/PhySO/physo/physym/reward.py:170` calls `programs.batch_optimize_constants`.
- `.tmp/PhySO/physo/physym/free_const.py:1` implements free-constant tables and optimization support.

Impact: structure search is less polluted by bad constant initialization. Under noise, this matters because a correct skeleton with poor constants can otherwise look worse than an overfit wrong skeleton.

### 6. Benchmark evidence

PhySO's README claims strong noisy Feynman performance, including substantial 10% noise.

Evidence:

- `.tmp/PhySO/README.md:45` says PhySO has state-of-the-art performance with noise above 0.1% and robust performance at 10%.
- `.tmp/PhySO/docs/source/others/doc_benchmarks.md:283` defines benchmark noise as `NOISE_LEVEL`.
- `.tmp/PhySO/docs/source/others/doc_benchmarks.md:287-289` add Gaussian noise scaled by target RMS.

Cloned benchmark result files:

| Noise | Recovery | Accurate solution | Total R2 |
|---:|---:|---:|---:|
| 0.0 | 58.45% | 66.90% | 0.9236 |
| 0.01 | 57.93% | 64.83% | 0.9088 |
| 0.10 | 53.45% | 64.14% | 0.8943 |

Evidence:

- `.tmp/PhySO/benchmarking/FeynmanBenchmark/results/noise0.000/results_stats.txt:3`
- `.tmp/PhySO/benchmarking/FeynmanBenchmark/results/noise0.010/results_stats.txt:3`
- `.tmp/PhySO/benchmarking/FeynmanBenchmark/results/noise0.100/results_stats.txt:3`

Interpretation: recovery drops only ~5 points from clean to 10% noise in these shipped files. That supports "robust", though exact comparison depends on same budget, hardware, benchmark methodology, and equivalence criteria.

## Glassbox Comparison

### Glassbox strengths against noise

Glassbox has more explicit trust/routing machinery than PhySO:

- Fast-path/classifier uncertainty changes compute budget. `glassbox/sr/sklearn_wrapper.py:248` defines adaptive budget; `:281-296` uses entropy/margin to shrink or expand effort.
- Blackbox search plan blends feature-selection uncertainty, interaction pressure, candidate strength, fast-path uncertainty, and proposer uncertainty. See `glassbox/sr/sklearn_wrapper.py:2866`, `:2968-2976`, and plan outputs at `:3083`.
- Cross-validation skip guard blocks unstable fast-path formulas from bypassing evolution. See `glassbox/sr/sklearn_wrapper.py:3293`, `:3352-3356`.
- Universal proposer confidence is gated by checkpoint validation metrics, so raw skeleton confidence is diagnostic unless reliable. See `glassbox/universal_proposer/universal_proposer.py:578`, `:624`, `:650`, `:680-688`.
- Formula noise pruning exists after fit through C++ `reduce_formula_noise_cpp`. See `glassbox/sr/sklearn_wrapper.py:5217`.
- Training data generation includes multi-SNR, pink noise, quantization noise, and high-noise regimes. See `glassbox/curve_classifier/generate_curve_data.py:583` and `:609-617`.
- Project docs explicitly target noisy data, false-confidence rate, displayed-formula MSE, residual diagnostics, and robustness across seeds. See `areas_to_improve.md:32`, `:52-72`.

### PhySO strengths over Glassbox

PhySO has cleaner first-principles constraints:

- Units are first-class and deeply integrated into token generation/reward.
- Weighted fitting is documented as public API.
- Free constants are optimized in reward path for every candidate batch.
- Class SR natively fits one functional form across multiple datasets with shared/dataset-specific constants, which can separate law structure from realization noise.
- Its public Feynman noisy results are easy to inspect and cite.

### Glassbox strengths over PhySO

Glassbox is stronger at meta-diagnosis:

- It can decide not to trust a high-scoring formula when CV folds are unstable.
- It can escalate search when uncertainty is high instead of using one mostly fixed search regime.
- It has displayed-formula scoring guards to detect raw engine/display drift.
- It has candidate screening, residual stages, specialist composition, and Pareto selection for "good enough but too complex" formulas.
- Its classifier/proposer can be trained on varied noise profiles, not only Gaussian RMS benchmark noise.

### Current Glassbox gaps relative to PhySO

- No public `sample_weight` / `y_uncertainty` contract comparable to PhySO `y_weights`.
- Units/dimensional analysis are not central in current active pipeline.
- Noise handling is distributed across many heuristics, harder to explain and benchmark.
- Need a clean noisy Feynman/SRBench report table comparable to PhySO's shipped `noise0.000`, `noise0.010`, `noise0.100`.
- Post-fit noise pruning exists, but should be measured as an ablation, not assumed.

## Practical Takeaways

Best ideas to borrow from PhySO:

1. Add public `sample_weight` / `y_uncertainty` support to `GlassboxRegressor.fit`.
2. Thread weights through fast-path scoring, constant fitting, candidate screening, evolution fitness, CV guard, and final displayed-formula scoring.
3. Add optional unit vectors for variables/constants and use them as hard priors where available.
4. Publish a noisy benchmark table with clean, 0.1%, 1%, and 10% RMS Gaussian noise, plus non-Gaussian noise profiles.
5. Add ablations: no weights, no CV guard, no uncertainty routing, no formula-noise pruning, no proposer/classifier.

Bottom line: PhySO is noise-robust because it constrains search with physics and lets uncertainty enter the objective. Glassbox has better routing and validation concepts, but should make weights/units first-class and publish comparable noisy benchmark evidence before claiming stronger noise handling.

## Phase-Wise Plan To Improve Glassbox Noise Handling

### Phase 0 - Baseline And Instrumentation

Goal: make current noise behavior measurable before changing objectives.

Important files:

- `glassbox/sr/cpp/evolution.h`: native fitness, ridge output solve, pruning, cleanup, NSGA-II.
- `glassbox/sr/cpp/core.cpp`: pybind signatures for `run_evolution` and `score_formula_candidates`.
- `glassbox/sr/cpp/simplify_advanced.h`: `reduce_formula_noise_cpp` BIC-style term pruning.
- `glassbox/sr/sklearn_wrapper.py`: active sklearn orchestration, CV guard, search plan, final selection.
- `scripts/classifier_fast_path.py`: guided evolution beam handoff.
- `scripts/benchmark_common.py`, `scripts/run_srbench_local.py`: displayed-formula scoring and benchmark reporting.
- `glassbox/curve_classifier/generate_curve_data.py`: current multi-SNR training noise generation.

Do:

- Add benchmark report columns for `noise_level`, `noise_type`, `sample_weight_mode`, `raw_mse`, `display_mse`, `holdout_mse`, `formula_complexity`, `false_confidence`, and `seed_graphs_used`.
- Create fixed noisy suites: clean, 0.1%, 1%, 10% RMS Gaussian, pink noise, quantization noise, sparse outliers.
- Run at least 5 seeds per tier for noisy Feynman/easy synthetic cases.
- Capture native trace stats using existing `trace_path` support in `core.cpp`.

Don't:

- Do not tune thresholds against one noisy benchmark run.
- Do not use raw native MSE as success metric alone; keep displayed-formula MSE primary.
- Do not compare against PhySO unless noise protocol and budgets are explicit.

Keep In Mind From Research:

- PhySO's published strength comes with clear noise protocol and result files.
- Our current system has many anti-overfit heuristics, but no clean public noisy evidence table.

Expected Outcome:

- A reproducible baseline table showing exact recovery, acceptable recovery, displayed MSE, complexity, and stability under noise.
- Known failure buckets: overfit complexity, wrong structure/high fit, unstable CV, residual memorization, display/raw drift.

### Phase 1 - Public Weight And Uncertainty Contract

Goal: expose PhySO-style uncertainty-aware fitting at the estimator boundary.

Do:

- Add `sample_weight=None` and optional `y_uncertainty=None` to `GlassboxRegressor.fit(X, y, ...)`.
- Normalize weights consistently: finite, nonnegative, mean near 1, zero weight allowed only if documented.
- Store diagnostics: effective sample size, min/max weight, weight source, skipped/invalid weights.
- Thread weights through Python scoring first: `_formula_mse`, `_display_formula_mse`, CV guard, final formula scoring, Pareto selection, candidate screening.

Don't:

- Do not silently reinterpret `y_uncertainty` as weight without documenting rule, e.g. `weight = 1 / sigma^2`.
- Do not let zero/near-zero total weight pass.
- Do not change default behavior when no weights are supplied.

Keep In Mind From Research:

- PhySO weights affect both free-constant optimization and reward. Partial weighting only at final scoring would be weaker and misleading.

Expected Outcome:

- User can downweight noisy regions and see that final selection honors those weights.
- Existing tests pass unchanged with `sample_weight=None`.

### Phase 2 - Weighted Native Candidate Scoring

Goal: make C++ batch candidate scorer match weighted validation semantics.

Do:

- Extend `score_formula_candidates_cpp` in `core.cpp` with optional `fit_weights` and `val_weights`.
- Replace affine fit mean/variance/covariance with weighted versions.
- Return both weighted and unweighted metrics: `weighted_fit_mse`, `weighted_validation_mse`, `validation_mse`, `validation_r2`.
- Update `_refine_candidate_formulas` and specialist screening to pass split weights.
- Add tests in `tests/test_cpp_candidate_scoring.py` for weighted affine scale/bias and weighted validation selection.

Don't:

- Do not overwrite existing `mse` meaning without migration notes.
- Do not report weighted R2 using unweighted variance.
- Do not let NaN weights become zeros silently.

Keep In Mind From Research:

- PhySO uses weights inside reward. Candidate screening is our closest cheap equivalent before full native evolution.

Expected Outcome:

- Candidate pools stop favoring formulas that fit known-noisy points.
- Validation R2 and acceptance thresholds become weight-aware.

### Phase 3 - Weighted C++ Evolution Objective

Goal: make native evolution optimize weighted loss, not only unweighted MSE.

Do:

- Add optional `y_weights` to `run_evolution_cpp` in `core.cpp` and store it in `EvolutionEngine`.
- Update `evaluate_fitness_with_penalty` in `evolution.h` to compute weighted MSE for `fitness` and keep unweighted `raw_mse` for diagnostics.
- Extend `DifferentialGramian`, `solve_output_weights`, snapping MSE, cleanup backward elimination, and inner-parameter refinement to support weighted residuals.
- Return `best_weighted_mse`, `best_mse`, and weight diagnostics separately.
- Add native tests in `glassbox/sr/test_cpp_parity.py` proving weighted evolution changes choice when noisy outliers are downweighted.

Don't:

- Do not remove unweighted `raw_mse`; benchmarks and backward compatibility need it.
- Do not use weights only in selection while fitting output weights unweighted.
- Do not apply complexity penalty to weighted and unweighted metrics inconsistently without naming it.

Keep In Mind From Research:

- Squared error remains outlier-sensitive. Weights help only if they reach every optimizer path.
- Native backend currently has ridge output solve, pruning, and snapping; all MSE comparisons there need weight awareness.

Expected Outcome:

- Evolution can recover correct simple structure when known bad observations would otherwise dominate.
- Pareto front can expose weighted fit vs unweighted robustness tradeoff.

### Phase 4 - Robust Loss Modes For Unknown Noise

Goal: handle noise when user has no weights.

Do:

- Add optional `loss_mode`: `mse`, `huber`, `trimmed_mse`, `student_t`.
- Start in Python candidate screening and final scoring, then port minimal `huber` / `trimmed_mse` to C++ evolution.
- Add residual-scale estimation using MAD on validation residuals.
- Use robust loss for search fitness, but keep displayed unweighted MSE/R2 in reports.

Don't:

- Do not default to robust loss until benchmark evidence supports it.
- Do not let trimmed loss hide systematic residual structure.
- Do not combine robust loss with aggressive residual boosting without stricter holdout guards.

Keep In Mind From Research:

- PhySO benchmark noise is Gaussian RMS. Real user noise can be outliers, quantization, or correlated pink noise.
- Glassbox already trains classifier data with pink/quantization noise; runtime scoring should learn same vocabulary.

Expected Outcome:

- Better recovery under outliers and quantization without requiring user-supplied uncertainties.
- Clear diagnostics showing when robust loss changed final formula.

### Phase 5 - Units And Physics Priors In Active Pipeline

Goal: borrow PhySO's strongest structural defense: reduce search before noise can create wrong fits.

Do:

- Promote existing C++ `input_units`, `output_units`, and `dim_penalty_weight` support into `GlassboxRegressor` public API.
- Add unit validation and examples for dimensionless default.
- Use units to filter candidate formulas and seed graphs before C++ evolution.
- Add hard vs soft unit modes: hard reject for impossible formulas, soft penalty for uncertain/unitless cases.

Don't:

- Do not require units for normal ML/tabular use.
- Do not apply unit penalties to formulas whose unit cannot be inferred safely.
- Do not mix incompatible unit-vector lengths.

Keep In Mind From Research:

- PhySO wins by making physically invalid noise-fitting expressions unavailable or expensive.
- Our C++ backend already has dimensional penalty hooks, but they are not central in the Python estimator.

Expected Outcome:

- On physics-style noisy data, search space shrinks and high-MSE/simple physical candidates beat low-MSE/unphysical overfits.
- Public API can state when Glassbox is doing physics-constrained SR.

### Phase 6 - Noise-Aware Selection, Cleanup, And Residual Guards

Goal: prevent post-processing and residual stages from fitting noise.

Do:

- Extend `_cleanup_formula_with_fidelity_guard` and `reduce_formula_noise_cpp` with weights and holdout checks.
- Replace fixed cleanup slack with noise-aware slack based on residual scale and validation variance.
- Strengthen `_select_blackbox_pareto_formula`: include weighted validation, edge validation, complexity, risk, generalization gap, and robust residual diagnostics.
- Add residual-stage acceptance rule: candidate must improve weighted validation and not worsen unweighted/edge validation beyond slack.
- Track "noise_pruned_terms", "cleanup_rejected_reason", and "residual_rejected_as_noise".

Don't:

- Do not accept residual additions on training MSE alone.
- Do not let BIC pruning remove physically required small terms without holdout/fidelity guard.
- Do not use one slack value for clean and 10% noisy data.

Keep In Mind From Research:

- PhySO duplicate/physical filtering reduces noisy variants early. Glassbox cleanup happens late, so it needs strong guards.
- Current backend already has backward elimination and snapping; these should become noise-calibrated, not more aggressive blindly.

Expected Outcome:

- Final formulas become simpler and less noise-shaped.
- Residual/specialist stages improve true structure instead of memorizing residual noise.

### Phase 7 - Training And Routing Calibration

Goal: make classifier/proposer uncertainty calibrated for noisy runtime decisions.

Do:

- Train/evaluate proposer and classifier on explicit noise profiles from `generate_curve_data.py`: clean, low/medium/high Gaussian, pink, quantization, outliers.
- Add validation metrics by noise bucket: candidate recall, operator F1, skeleton confidence reliability, false-confidence rate.
- Feed runtime noise diagnostics into `_derive_blackbox_search_plan`: residual autocorrelation, outlier fraction, weight effective sample size, validation gap.
- Calibrate thresholds for `prediction_uncertain`, `candidate_acceptance_r2`, and `candidate_shrink_r2` per noise band.

Don't:

- Do not trust raw skeleton logits unless reliability gates pass.
- Do not shrink search diversity just because noisy candidate MSE is low.
- Do not train on noise profiles that benchmark never reports.

Keep In Mind From Research:

- Glassbox's differentiator is routing: know when to trust, refine, or spend compute.
- Noisy data increases false confidence. Calibration must measure that directly.

Expected Outcome:

- Search budget rises on ambiguous/noisy cases and shrinks only on stable verified formulas.
- Diagnostics explain why Glassbox trusted or rejected noisy candidates.

### Phase 8 - Benchmark Release Gate

Goal: make noise handling a release-tested feature, not a claim.

Do:

- Add `scripts/benchmark_noise.py` or extend `run_srbench_local.py` with deterministic noise protocol.
- Report clean vs noisy deltas, not only absolute scores.
- Include ablations: no weights, no robust loss, no units, no CV guard, no uncertainty routing, no noise pruning.
- Add CI smoke tests for small weighted cases and one noisy outlier recovery case.

Don't:

- Do not optimize only for exact recovery; include acceptable simple formula rate.
- Do not hide failed seeds.
- Do not compare methods with different timeouts without reporting budgets.

Keep In Mind From Research:

- PhySO ships result artifacts for multiple noise levels. Glassbox should match that transparency.
- Main paper-worthy claim remains: uncertainty-aware SR that knows when to trust a formula, refine it, or spend compute.

Expected Outcome:

- A stable release gate: no merge should regress noisy recovery, false-confidence rate, or displayed-formula scoring.
- A credible comparison table against PhySO/PySR/gplearn under identical noise protocol.

## Implementation Order

Recommended order:

1. Phase 0 baseline first.
2. Phase 1 Python `sample_weight` contract.
3. Phase 2 weighted C++ candidate scorer.
4. Phase 3 weighted native evolution.
5. Phase 6 cleanup/residual guards.
6. Phase 4 robust loss.
7. Phase 5 units API.
8. Phase 7 calibration.
9. Phase 8 release gate.

Reason: weights are the lowest-risk PhySO lesson and must become end-to-end before robust loss or units add more moving parts.
