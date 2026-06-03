# Specialist Composition Integration Audit

## Current Status (2026-06-03)

This audit documents bugs, fixes, and design tradeoffs from the specialist
composition rollout. Several findings have since been fixed or downgraded in
code and tests. The live implementation map is in `docs/PROJECT_MAP.md`; keep
using this audit for historical rationale and regression context.

Date: 2026-06-01

Scope:

- `specialist_composition_plan.md`
- `glassbox/sr/specialist_state.py`
- `glassbox/sr/sklearn_wrapper.py`
- `glassbox/sr/cpp/seed_graph_builder.py`
- `scripts/benchmark_suite.py`
- `scripts/benchmark_common.py`
- focused specialist/benchmark tests

## Executive Summary

The specialist composition feature is implemented, but it is not integrated as a narrow fast-path adjunct anymore. In the regressor path it now activates several extra stages by default: univariate candidate generation, specialist diagnostics, expanded composition templates, residual symbolic fitting, vault memory, and seed graph construction. Segment scoring itself is not the likely bottleneck. The second pass shows runtime is distributed across several stages; residual symbolic fitting is a real risk, but C++ evolution/adaptive compute and candidate/refinement overhead also need timing before assigning blame.

The second major issue is benchmark integration ambiguity. The ordinary `benchmark_suite.py --with-evolution` path does not run the specialist regressor pipeline. The specialist feature is measured only when `--specialist-regressor` is used. That means tier changes observed in default fast-path/guided-evolution mode may not be caused by the specialist layer, while changes in `--specialist-regressor` include many more moving parts than just composition.

## Second-Pass Triage

Input added for this pass:

- `results/benchmark_20260601_165105.md`
- `results/benchmark_20260601_165105.json`

Important context:

- This report is a `specialist_regressor` run for all 205 formulas.
- It is not the default `run_formula_benchmark` / `--with-evolution` path.
- Total runtime was `2079.5s`.
- Overall exact recovery was `141/205` (`69%`).

### Confirmed Bugs / High-Confidence Issues

#### A. Lowercase `e` Is Parsed As Zero In Benchmark Targets

Report evidence:

- Tier 1 target `e` discovered `0` with MSE `0`.
- Tier 1 target `e*x` discovered `0` with MSE `0`.

Probe evidence:

- `_generate_data("e", ..., 300)` produces all-zero `y`.
- `_generate_data("e*x", ..., 300)` produces all-zero `y`.
- `_generate_data("E", ..., 300)` and `_generate_data("exp(1)", ..., 300)` produce Euler's constant correctly.

Classification:

- Confirmed benchmark correctness bug.

Impact:

- Tier 1 has false positives.
- Any target using lowercase `e` as Euler's constant is scored against the wrong data.

Fix:

- Add `"e": sp.E` to benchmark formula parsing local dictionaries.
- Re-run affected tiers after fixing.

#### B. Raw/Displayed MSE Drift Is Real And Material

Report evidence:

- Tier 8 `sin(exp(-x))`: raw MSE `4.88e-08`, display MSE `1.09e+00`, score `FAIL`.
- Tier 8 `x^2*exp(-x)*cos(3*x)`: raw MSE `2.52e-04`, display MSE `8.17e+01`, score `FAIL`.
- Tier 7 `sin(x)*sin(3*x)*sin(5*x)`: raw MSE `1.27e-02`, display MSE `1.01e+05`, score `FAIL`.
- Tier 7 `log(1+sin(x)^2)`: raw MSE `7.01e-06`, display MSE `3.48e-02`, score `LOOSE`.

Classification:

- Confirmed diagnostic/fidelity bug, not a scoring bug.

Impact:

- Scoring correctly uses displayed MSE.
- But raw MSE can make failures look like nearly solved formulas.
- The code needs clearer separation between engine/raw-prediction fit and displayed-formula fidelity.

Fix:

- Store separate fields:
  - `engine_raw_mse`
  - `formula_before_postprocess_mse`
  - `formula_after_postprocess_mse`
  - `score_mse`
- Add a report column or warning for severe drift.

#### C. `formula_eval_failed` Can Hide A Near-Good Raw Fit

Report evidence:

- Tier 7 `exp(-x)*x^3`: raw MSE `5.77e-05`, display MSE missing, final formula shown as `ERROR: formula_eval_failed`.

Classification:

- Confirmed displayed-formula fidelity failure.

Impact:

- Scoring failure is correct because the displayed formula cannot be evaluated.
- Diagnostics need the unevaluable formula and parse/eval error cause, otherwise this is hard to debug.

Fix:

- Preserve `formula_before_display_error`.
- Preserve `display_eval_error`.
- Include the exact parser/evaluator failure reason in JSON.

#### D. Specialist Regressor Runtime Can Run Far Beyond A "Fast" Pipeline

Report evidence:

- Full specialist run took `2079.5s`.
- Top outlier: Tier 7 `sin(x^2)+cos(x)` took `317.7s` and ended exact.
- Other exact cases took `95.0s`, `43.5s`, `36.8s`, `31.3s`, and `31.0s`.
- Rows with `has_composed_seeds=True` averaged about `11.1s`; rows without averaged about `5.8s`.
- Rows with `boosting_attempted=True` averaged about `16.1s`; rows without averaged about `7.2s`.

Classification:

- Confirmed performance problem.
- The first-pass claim that residual specialist composition is the main source is too narrow.

Refined interpretation:

- `_stage_residual_symbolic_fit(timeout=max(20, self.timeout // 2))` is a real cost risk.
- But the report's worst case (`317.7s`) had `boosting_attempted=False`, so residual boosting is not the only cause.
- Adaptive compute/evolution budget, composition screening, seed generation, and final evolution can dominate.

Fix:

- Add per-phase timing first. Without timing, optimizing one stage risks chasing the wrong bottleneck.
- Track:
  - fast path
  - candidate building/refinement
  - specialist diagnostics
  - composition proposal/refinement
  - residual stage
  - C++ evolution
  - postprocess/display evaluation

#### E. `has_composed_seeds` Is Misleading In Reports

Report evidence:

- `has_composed_seeds=True` for 134/205 rows.
- All 134 still report `specialist_track="incumbent path"`.
- Composition proposals were accepted in exactly those 134 rows, but the final winner was still labeled incumbent.

Classification:

- Confirmed reporting semantics issue, not necessarily solver bug.

Impact:

- Readers may interpret `has_composed_seeds=True` as "composition helped".
- In this report it mostly means "composition proposals were accepted/available".

Fix:

- Rename/report separate booleans:
  - `composition_candidates_accepted`
  - `composition_seeded_evolution`
  - `composition_won_final_selection`
  - `composition_improved_mse`

### Confirmed Design Choices, Not Bugs

#### A. Specialist Is Separate From Default Benchmark Mode

Original finding:

- Default `--with-evolution` does not run specialist composition.

Second-pass classification:

- Design choice.

Reason:

- `benchmark_suite.py` has a distinct `--specialist-regressor` mode.
- The provided report confirms that mode is used through `benchmark_path="specialist_regressor"`.

Optimality:

- Reasonable for isolating API paths.
- Not optimal for A/B attribution because `--specialist-regressor` toggles several specialist mechanisms together.

Recommended change:

- Keep separate modes, but add phase toggles for clean attribution.

#### B. Composition Proposal Uses Training MSE Before Holdout Refinement

Original finding:

- `propose_specialist_compositions()` ranks templates using full-data MSE before `_refine_candidate_formulas()` applies holdout validation.

Second-pass classification:

- Design choice with risk, not a direct bug.

Reason:

- The code intentionally uses MSE only to rank templates within a tiny candidate set.
- It then validates/refines before accepting candidates.
- Operator diversity is preserved before pure MSE ordering.

Optimality:

- Acceptable for add/mul/nested templates.
- Less optimal for high-flexibility templates like `sigmoid_gate`, where full-data ranking can reward local stitching.

Recommended change:

- Do not remove immediately.
- Add phase metrics first.
- Consider feature-flagging expanded templates separately from add/mul/nested.

#### C. Univariate Candidates Are Discarded Unless Composition Succeeds

Original finding:

- Univariate specialist candidates are built/screened, then discarded if no composed seeds are accepted.

Second-pass classification:

- Design choice, not a bug.

Reason:

- It avoids perturbing the evolution seed pool unless specialist composition contributes something new.

Optimality:

- Conservative for accuracy.
- Suboptimal for runtime because screening/refinement cost can be paid with no downstream use.

Recommended change:

- Keep behavior if accuracy stability is priority.
- Add early skip or diagnostics if runtime is priority.

#### D. Residual-Suspicion Routing After Low-MSE Gates

Original finding:

- Suspicious residual checks occur after low-MSE skip gates.

Second-pass classification:

- Design choice.

Reason:

- The benchmark prioritizes displayed MSE and term count over residual shape when the fit is already near exact.

Optimality:

- Good for speed and benchmark score.
- Not optimal for exact structural recovery if a low-MSE surrogate masks missing structure.

Recommended change:

- Do not change globally.
- Add an optional strict-structure mode.

#### E. All-Core C++ Evolution

Original finding:

- C++ beam search uses `multiprocessing.cpu_count()` as thread count.

Second-pass classification:

- Design choice.

Reason:

- Good for single-formula throughput.

Optimality:

- Poor if benchmark formulas are run in parallel externally.
- Add a CLI/config thread cap rather than changing default blindly.

### Low-Priority Or Downgraded Findings

#### A. Duplicate `make_beam_configs()` Call

Classification:

- Confirmed minor inefficiency.

Reason for downgrade:

- It is wasteful and confusing but unlikely to explain tier-level runtime changes.

Fix:

- Remove the first unused call when touching guided evolution.

#### B. Hot-Spot Bonus Uses `candidates[0]`

Classification:

- Confirmed logic smell, not proven benchmark-impacting.

Reason for downgrade:

- Candidate lists are usually already sorted before specialist state construction.
- However, relying on input order is fragile.

Fix:

- Low-risk cleanup: use the same ranked best candidate selected for hot-spot construction.

#### C. Report Output Directory Creation

Classification:

- Confirmed tooling issue.

Reason for downgrade:

- Important for automation, but not related to formula quality or runtime once using a valid `results/` directory.

Fix:

- Validate output directory at startup.

#### D. Report Timestamp Collisions

Classification:

- Confirmed tooling issue.

Reason for downgrade:

- Only affects concurrent or same-second runs.

Fix:

- Include microseconds or process id in report filenames.

### Revised Priority List

1. Fix lowercase `e` target parsing and re-run affected benchmark rows.
2. Add per-phase timing to `GlassboxRegressor.fit()` and benchmark JSON.
3. Split raw/display MSE diagnostics into explicit fields.
4. Preserve display-eval failure details and the unevaluable formula.
5. Clarify composition reporting semantics: accepted vs seeded vs won.
6. Add specialist phase toggles for A/B: diagnostics, add/mul/nested, expanded templates, residual, vault, inception.
7. Add a strict specialist/evolution wall-time cap per formula.
8. Only after timings are available, optimize the dominant stage.

## Fix Pass Implemented

Implemented after the second-pass triage:

- Benchmark target parsing now treats lowercase `e` as Euler's constant in `scripts/benchmark_suite.py`, `scripts/benchmark_common.py`, and `scripts/classifier_fast_path.py`.
- Benchmark results now split scoring and fidelity diagnostics:
  - `engine_raw_mse`
  - `formula_before_postprocess_mse`
  - `formula_after_postprocess_mse`
  - `score_mse`
  - `display_eval_diagnostics`
  - `formula_before_display_error`
- Default fast-path/guided benchmark results now preserve:
  - `fast_path_candidate_formulas`
  - `proposer_candidate_formulas`
  - `evolution_seed_candidates`
  - `winning_stage`
- Specialist metadata now separates composition semantics:
  - `composition_candidates_accepted`
  - `composition_candidate_count`
  - `composition_seeded_evolution`
  - `composition_won_final_selection`
  - `composition_improved_mse`
- Specialist regressor diagnostics now include `phase_timings`.
- Added timing buckets for specialist diagnostics/composition, candidate building, residual symbolic fit, residual boosting, inception reuse, and total fit.
- Benchmark markdown reports now expose severe raw/display MSE drift and the winning benchmark stage.
- Benchmark output directories are validated before formula execution, and timestamped report filenames include microseconds.
- `benchmark_suite.py --specialist-regressor` now has explicit specialist phase switches:
  - `--disable-specialist-diagnostics`
  - `--disable-specialist-composition`
  - `--enable-specialist-residual`
  - `--disable-specialist-vault`
  - `--enable-specialist-inception`
  - `--specialist-full`
- Benchmark specialist mode now defaults to diagnostics + composition + vault only. Residual symbolic fitting and inception reuse are opt-in for benchmark runs.
- The nested residual symbolic fit budget is now capped to a small strict budget instead of using `max(20, self.timeout // 2)`.
- Hot-spot complementarity bonus now uses the same ranked best candidate that generated hot-spot segments instead of relying on input order via `candidates[0]`.
- `_safe_eval_formula_array()` now maintains a per-fit cache for repeated formula/matrix evaluations and exports `formula_eval_count`, `formula_eval_cache_hits`, and `formula_eval_cache_size`.
- The blackbox and univariate candidate-screening branches now share `_run_specialist_candidate_screening()` instead of carrying duplicated specialist composition/residual merge logic.

Bug found during implementation:

- The first timing-helper patch accidentally reset boosting/inception state inside `_add_phase_time()`. This was caught by `test_residual_boosting_records_attempt_and_improvement` and fixed by moving the resets back into `__init__`; `_add_phase_time()` now only accumulates timing.

## Fast-Path / Specialist Pipeline Trace

### Default Benchmark Path

Entry point:

- `scripts/benchmark_suite.py::run_formula_benchmark`
- used by default benchmark mode and `--with-evolution`

Flow:

1. Generate synthetic `x_np, y_np`.
2. Convert to torch tensors.
3. Detect dominant frequencies.
4. Run `scripts.classifier_fast_path.run_fast_path`.
5. If fast-path is exact enough, return formula.
6. If guided evolution should run, build operator hints and optional proposer candidate skeletons.
7. Run `scripts.classifier_fast_path.run_guided_evolution`.
8. Score only the displayed formula MSE.

Expected output:

- formula string
- raw/display MSE
- uncertainty
- optional candidate formulas
- score bucket

Observed integration point:

- Specialist composition is not part of this path.
- Candidate formulas passed to guided evolution come from fast path and the universal proposer, not `compute_specialist_state` or `propose_specialist_compositions`.
- Evidence: `scripts/benchmark_suite.py` initializes `candidate_formulas = None` in the guided path around line 1028 and only fills it from fast-path/proposer data before calling `run_guided_evolution`.

Implication:

- If the user runs normal tier benchmarks, specialist composition is mostly invisible.
- If accuracy changed in normal mode, inspect fast-path/proposer/guided-evolution changes separately from specialist composition.

### Specialist Regressor Benchmark Path

Entry point:

- `scripts/benchmark_suite.py::run_formula_specialist_regressor`
- enabled only by `--specialist-regressor`

Flow:

1. Generate `X, y`.
2. Build `GlassboxRegressor` with:
   - `blackbox_mode=True`
   - `enable_specialist_screening_diagnostics=specialist_enabled`
   - `enable_specialist_composition_screening=specialist_enabled`
   - `enable_residual_stage=specialist_enabled`
   - `enable_specialist_vault_memory=specialist_enabled`
   - `enable_inception_reuse=specialist_enabled`
   - `multi_start_runs=1`
3. `GlassboxRegressor.fit()` runs fast path.
4. It derives a blackbox search plan.
5. It builds candidate formulas from fast path, proposer, blackbox interactions, or univariate basis seeds.
6. It refines candidates through `_refine_candidate_formulas`.
7. It computes specialist diagnostics via `compute_specialist_state`.
8. It proposes compositions via `propose_specialist_compositions`.
9. It refines composed candidates again.
10. For top composed candidates with `validation_r2 >= 0.75`, it may run `_stage_residual_symbolic_fit`.
11. It prunes and passes candidate formulas into `build_seed_graphs_from_candidates`.
12. C++ evolution consumes `seed_graphs_py`.
13. Final formula is selected and scored.

Expected output:

- same benchmark result fields
- plus `specialist_track`
- plus `has_composed_seeds`
- plus `specialist_diagnostics`
- plus `specialist_composition_screening`
- plus vault/inception diagnostics from `benchmark_common.specialist_metadata_from_estimator`

Observed behavior:

- This path measures specialist behavior, but it also changes the solver path substantially compared with the default benchmark.
- It is not a clean A/B for "composition only" because residual stage, vault memory, inception reuse, and blackbox-mode behavior are toggled together.

## Main Findings

### 1. Residual Specialist Composition Can Blow Past the Intended Fast Path Budget

Evidence:

- `glassbox/sr/sklearn_wrapper.py:1846` and `3554`: runs `_stage_residual_symbolic_fit` for up to two specialist candidates.
- `glassbox/sr/sklearn_wrapper.py:2918`: nested residual estimator uses `timeout=max(20, int(self.timeout // 2))`.

Impact:

- A nominal 5-10 second benchmark can spend 20 seconds inside one residual attempt.
- Since up to two composed candidates can trigger this, the worst-case overhead is much larger than the "tiny composition screening" design.
- Second-pass note: the full benchmark report shows this is one contributor, not the only runtime source. Several largest outliers did not attempt boosting/residual fitting.

Suggested fix:

- Replace `max(20, self.timeout // 2)` with a strict small budget such as `min(3, max(1, self.timeout // 5))` for specialist screening, or disable residual symbolic fitting during the fast-path/screening phase.
- Track residual-stage wall time separately in diagnostics.
- Add a hard global per-fit specialist overhead budget.

### 2. Composition Proposal Generation Uses Training MSE Before Validation Gating

Evidence:

- `glassbox/sr/specialist_state.py:954`: `propose_specialist_compositions`.
- Lines around `1012-1122` generate add, multiply, divide, nested, affine, damped product, and sigmoid-gate templates.
- Lines around `1183` sort candidate templates by local MSE and complexity before `_refine_candidate_formulas` later applies holdout scoring.

Impact:

- The proposal phase can prefer templates that overfit the full input data before validation gets a chance to reject them.
- Expanded templates such as `sigmoid_gate` look piecewise-like and can improve local fit while hurting exact formula recovery.

Suggested fix:

- Keep proposal generation structural and cheap; do not rank by full-data MSE except as a coarse invalidity filter.
- Move ranking to the existing holdout validation stage.
- Consider disabling `sigmoid_gate` and `damped_product` by default until the simpler add/mul/nested path is stable.

### 3. Specialist Proposal Cost Is Serial and Repeated

Evidence:

- `compute_specialist_state` evaluates each candidate formula, stores residuals, then proposal generation evaluates many composed templates again.
- `_compose_specialist_candidates` then calls `_refine_candidate_formulas`, which evaluates and constant-refines formulas again.
- `propose_specialist_compositions` tests damped products over 15 beta values and sigmoid gates over 7 cutpoints x 10 slopes per pair.

Impact:

- This is CPU/Python-heavy and not aligned with the rest of the multithreaded/native evolution flow.
- Even if each step is small, the repeated safe-eval/refine cycle compounds.

Suggested fix:

- Cache formula predictions inside a per-fit evaluation cache keyed by normalized formula and X identity/shape.
- Add diagnostics for number of formula evaluations, template evaluations, and time per specialist phase.
- Gate expensive templates behind a per-formula timeout or `specialist_expanded_templates=False` default.

### 4. Univariate Specialist Candidates Are Built Then Often Discarded

Evidence:

- `glassbox/sr/sklearn_wrapper.py:3601`: builds univariate specialist candidates when blackbox state is not enabled and feature count is 1.
- `glassbox/sr/sklearn_wrapper.py:3613`: runs specialist screening.
- `glassbox/sr/sklearn_wrapper.py:3620-3623`: only assigns `candidate_formulas = screened_univariate_candidates` if `has_composed_seeds_` changed; otherwise sets `candidate_formulas = None`.

Impact:

- The pipeline pays candidate-building/refinement/screening overhead even when no composed seeds are accepted.
- Useful validated univariate candidates are not available to seed evolution unless composition succeeds.

Suggested fix:

- If univariate screening runs, keep a capped candidate pool for seeding even when no composition was accepted, or skip the whole univariate screening path unless a cheap precheck says composition is likely.
- Add a diagnostic reason when screened candidates are discarded.

### 5. The Specialist Feature Is Not Part of the Default Benchmark Mode

Evidence:

- `scripts/benchmark_suite.py:1342`: specialist regressor has a separate entry point.
- `scripts/benchmark_suite.py:1841`: selected only when `args.specialist_regressor`.
- Default guided evolution uses `run_formula_benchmark`, not `run_formula_specialist_regressor`.

Impact:

- Tier regressions can be misattributed if the command used was not `--specialist-regressor`.
- There is no single benchmark mode that isolates "old regressor path vs new composition-only path".

Suggested fix:

- Add explicit benchmark modes:
  - baseline regressor
  - specialist diagnostics only
  - specialist add/mul/nested only
  - specialist expanded templates
  - specialist residual stage
  - specialist vault/inception
- Print enabled specialist phases in each per-formula result.

### 6. Blackbox Specialist Screening Logic Is Duplicated

Evidence:

- Helper exists at `glassbox/sr/sklearn_wrapper.py:1792`.
- Similar inline logic appears again around `glassbox/sr/sklearn_wrapper.py:3528-3585`.

Impact:

- Future fixes can be applied to one path and missed in the other.
- Diagnostics and caps can diverge silently.

Suggested fix:

- Route both blackbox and univariate paths through `_run_specialist_candidate_screening`.
- Keep only path-specific candidate construction outside the helper.

### 7. Composed Seed Cap Exists, But Composition Candidates Can Still Dominate Pre-Seeding Work

Evidence:

- `glassbox/sr/cpp/seed_graph_builder.py:637-641`: composed seeds are capped at about 35% of seed budget.
- The cap happens after proposal generation, validation, residual fitting, pruning, and vault augmentation.

Impact:

- Evolution population is protected from composed-seed domination, but runtime before evolution is not protected.

Suggested fix:

- Add pre-seed caps before expensive refinement/residual fitting:
  - max pairs
  - max templates per pair
  - max total proposal evaluations
  - max residual-stage attempts
  - max specialist wall time

### 8. Hot-Spot Bonus Uses `candidates[0]`, Not the Ranked Best Candidate

Evidence:

- `glassbox/sr/specialist_state.py:732`: computes `best_temp_candidate` by validation/ranking for hot-spot segment construction.
- `glassbox/sr/specialist_state.py:888`: hot-spot bonus uses `best_candidate = candidates[0]`.

Impact:

- If input candidate order differs from ranked order, the hot-spot bonus can compare against the wrong candidate.
- This can skew complementarity and template selection.

Suggested fix:

- Preserve the selected best formula key and use that candidate for hot-spot bonus.
- Or sort `candidates` by the same `_candidate_rank` before pair scoring.

### 9. Benchmark Output Directory Creation Can Fail Late

Observed probe:

- Running single-formula specialist benchmarks completed formula evaluation, then failed at `scripts/benchmark_suite.py:1919` with `PermissionError: [WinError 5] Access is denied` while creating the output directory.

Impact:

- Benchmark results can be printed but the process exits non-zero after successful work.
- This makes automation treat successful probes as failures.

Suggested fix:

- Create and validate output directory before running formulas.
- Add `--no-write` or `--dry-run-report` for quick diagnostic runs.
- If output writing fails, preserve a clear summary and exit with a distinct report-write error code.

### 10. Test Runs Emit Pytest Cache Warnings

Observed:

- `python -m pytest tests/test_specialist_state.py -q`: passed, but emitted `PytestCacheWarning` because `.pytest_cache` could not be created under `D:\Glassbox`.
- `python -m pytest tests/test_benchmark_scoring_contract.py -q`: passed with the same warning.

Impact:

- Not a solver bug, but it creates noise in benchmark/test output.

Suggested fix:

- Configure pytest cache to a writable location for this environment, or ignore cache provider warnings in local runs.

## Verification Performed

Commands run:

- `python -m pytest tests\test_specialist_state.py -q`
  - Result: 11 passed
  - Warning: pytest cache path write denied
- `python -m pytest tests\test_benchmark_scoring_contract.py -q`
  - Result: 13 passed
  - Warning: pytest cache path write denied
- `python scripts\benchmark_suite.py --formula "sin(x)+sin(x^2)" --tier 4 --specialist-regressor --timeout 5 --guided-generations 20 --guided-pop-size 20 --n-samples 120 --device cpu --quiet --output-dir D:\tmp\glassbox_bench_probe`
  - Formula work succeeded: exact match in about 0.17s
  - Process failed during output directory creation
- Same benchmark with `--specialist-baseline`
  - Formula work succeeded: exact match in about 0.17s
  - Process failed during output directory creation

Latest verification after fixes:

- `python -m py_compile scripts\benchmark_common.py scripts\benchmark_suite.py scripts\classifier_fast_path.py glassbox\sr\sklearn_wrapper.py`
  - Result: passed
- `python -m pytest tests\test_benchmark_scoring_contract.py tests\test_benchmark_common.py -q`
  - Result: 19 passed
  - Warning: pytest cache path write denied
- `python -m pytest tests\test_specialist_phase_eval.py -q`
  - Result: 14 passed
  - Warning: pytest cache path write denied
- `python -m pytest tests\test_benchmark_scoring_contract.py tests\test_benchmark_common.py tests\test_specialist_state.py -q`
  - Result: 32 passed
  - Warning: pytest cache path write denied
- `python scripts\benchmark_suite.py --formula "e*x" --tier 1 --n-samples 64 --device cpu --quiet --output-dir D:\tmp\glassbox_bench_probe`
  - Result: failed before formula execution because the output directory was not writable, confirming early validation.
- `python scripts\benchmark_suite.py --formula "e*x" --tier 1 --n-samples 64 --device cpu --quiet --output-dir results`
  - Result: passed; recovered `E*x` exactly; wrote timestamped JSON/Markdown reports with microsecond suffixes.
- `python scripts\benchmark_suite.py --formula "sin(x)+sin(x^2)" --tier 4 --specialist-regressor --timeout 5 --guided-generations 20 --guided-pop-size 20 --n-samples 120 --device cpu --quiet --output-dir results`
  - Result: passed; printed phases `diagnostics, composition, vault`, confirming residual and inception are no longer silently enabled in benchmark specialist mode.

## Priority Fix Order

1. Add a global per-formula specialist overhead budget that covers all specialist phases, not just residual fitting.
2. Use the new specialist phase switches for tier A/B runs and keep benchmark residual/inception phases opt-in unless accuracy data justifies them.
3. Add more granular benchmark modes or presets if the CLI switches are still too coarse for attribution.
4. Use `phase_timings` and formula-eval counters from the next full benchmark to identify the dominant remaining stage.
5. Keep helper-based screening paths covered when adding new specialist diagnostics or caps.
6. Cache formula predictions during candidate refinement/composition.
7. Fix hot-spot bonus to use the same ranked best candidate as hot-spot segment construction.
8. Keep monitoring report-write behavior; output-dir validation now runs before formula execution.

## Full Default Pipeline Trace: Fast Path To Evolution To Display

This section traces the ordinary benchmark path, not the specialist regressor path.

Entry point:

- `scripts/benchmark_suite.py::run_formula_benchmark`

Primary called functions:

- `scripts.classifier_fast_path.run_fast_path`
- `scripts.benchmark_suite._guided_evolution_decision`
- `scripts.classifier_fast_path.run_guided_evolution`
- `scripts.classifier_fast_path.beam_search_evolution`
- `scripts.benchmark_common.postprocess_formula_with_fidelity_guard`
- `scripts.benchmark_suite.score_result`

### Stage 1: Data Generation

Input:

- target formula string from `ALL_TIERS`
- `x_range`
- `n_samples`

Process:

1. `_generate_data()` parses the target formula.
2. Generates evenly spaced `x`.
3. Evaluates target `y`.
4. Drops non-finite samples.
5. Converts to torch tensors `x_t`, `y_t`.

Expected output:

- `x_np`: 1D numpy array
- `y_np`: 1D numpy array
- `x_2d`: torch tensor shaped `(n, 1)`
- `y_2d`: torch tensor shaped `(n, 1)`

Notes:

- Target formula parsing uses benchmark-side parsing, while discovered formula display uses classifier/benchmark-common evaluation. Parser differences can matter for edge syntax.

### Stage 2: Frequency Detection

Process:

- `detect_dominant_frequency(x_2d, y_2d, n_frequencies=3)`

Expected output:

- `detected_omegas`: list of candidate periodic frequencies, or `None`

Use downstream:

- fed to `run_fast_path`
- later copied into `operator_hints["frequencies"]` if fast-path hints lack frequencies

### Stage 3: Fast Path

Call site:

- `scripts/benchmark_suite.py:945`

Important parameters:

- `auto_expand=True`
- `exact_match_threads=1`
- `exact_match_enabled=True`
- `exact_match_max_basis=150`
- `simplify_formula_output=False`

Internal fast-path flow:

1. Convert tensors to numpy.
2. Run curve classifier prediction.
3. Fall back to polynomial priors if classifier import/prediction fails.
4. Compute uncertainty metrics.
5. Gate applicability with `should_use_fast_path`.
6. Run `fast_path_with_refinement`.
7. Build formula, raw MSE, details, residual diagnostics, candidate formulas, operator hints, and FPIP v2.

Expected `fp_result` shape:

- `formula`: raw formula string from basis regression because benchmark disables fast-path simplification
- `formula_raw`: same or pre-simplification formula
- `mse`: fast-path internal MSE
- `details`: includes basis names, coefficients, candidate formulas, nonzero counts
- `predictions`: classifier outputs
- `uncertainty`: entropy/margin/top probabilities
- `residual_diagnostics`
- `candidate_formulas`
- `operator_hints`
- `fpip_v2`
- `fpip_v2_valid`

Observed issue: fast-path printed MSE can differ sharply from displayed-formula MSE.

- Probe target: `x^3+sin(x)`
- Fast path printed candidate: `x + 0.8401*x**3 + 0*x**5`
- Fast-path internal MSE printed: about `2.86e-06`
- Benchmark displayed MSE after postprocess/evaluation was about `0.213`
- That displayed MSE is what routed to evolution.

Likely reason:

- `run_fast_path` reports `mse` from the internal regression/refinement path.
- Benchmark then postprocesses/evaluates the string formula independently.
- If raw prediction and displayed formula semantics diverge, `mse_raw` no longer describes the displayed formula.

Recommended instrumentation:

- Add `fast_path_formula_eval_mse` inside `run_fast_path` by evaluating the returned `formula` string immediately before returning.
- Emit `raw_prediction_mse`, `formula_string_mse`, and `postprocess_mse` separately.
- Add a hard warning when `formula_string_mse / raw_prediction_mse > 10`.

### Stage 4: Benchmark Postprocess Guard

Call site:

- `scripts/benchmark_suite.py:963`

Process:

1. Pass `fp_result["formula"]` into `_postprocess_formula_for_benchmark`.
2. This calls `benchmark_common.postprocess_formula_with_fidelity_guard`.
3. It computes raw formula MSE and processed formula MSE.
4. If processed formula fails or gets worse beyond slack, it returns a protected fallback.

Expected output:

- `result["formula_discovered"]`
- `result["postprocess_guard"]`
- `result["mse_raw"] = fp_result["mse"]`
- `result["mse_display"] = evaluate displayed formula`
- `result["mse"] = mse_display`
- divergence stats

Important behavior:

- Scoring uses displayed MSE only.
- `mse_raw` remains fast-path internal MSE, not necessarily raw formula-string MSE.
- The fidelity guard compares raw formula-string MSE against postprocessed formula MSE, not fast-path internal MSE.

Risk:

- If fast-path internal MSE is excellent but formula-string MSE is poor, the guard does not recover the internal predictor. It can only choose among strings.

### Stage 5: Guided Evolution Routing

Call site:

- `scripts/benchmark_suite.py:996`

Routing rules:

1. `--evolution-only`: always run.
2. no `--with-evolution`: do not run.
3. `fp_result is None`: run.
4. invalid displayed MSE: run.
5. displayed MSE `< 1e-12`: skip.
6. relative error `< 1e-5` and terms `<= 10`: skip.
7. displayed MSE `< 1e-7` and terms `<= 6`: skip.
8. displayed MSE `< 1e-6` and terms `<= 10`: skip.
9. displayed MSE `>= 1e-6`: run.
10. terms `> 10`: run.
11. suspicious residual: run.

Observed issue:

- Suspicious residuals are only checked after several low-MSE skip gates.
- A formula with near-exact MSE but structured residual can skip evolution before `residual_suspicious` is considered.

Suggested fix:

- Move residual-suspicion routing before the final low-MSE skip, or make it override only when residual magnitude is meaningful.

### Stage 6: Operator Hints And Candidate Formulas

Call sites:

- `scripts/benchmark_suite.py:1020-1026`
- `scripts/benchmark_suite.py:1066-1084`
- `scripts/benchmark_suite.py:1121-1126`

Process:

1. Start from `fp_result["operator_hints"]`.
2. Normalize fields:
   - `operators`
   - `frequencies`
   - `powers`
   - `has_rational`
   - `has_exp_decay`
   - `active_terms`
   - `uncertainty`
3. If proposer is enabled and available, load proposer and get FPIP v2.
4. Add proposer priors to `operator_hints["operators"]`.
5. Build `candidate_formulas` from fast-path formula and proposer skeletons.
6. If no proposer candidates exist, fallback to a single fast-path candidate.

Expected candidate format:

- `formula`
- `mse`
- optional `score`
- optional `active_terms`
- source flags such as `from_fast_path`, `from_proposer`

Observed issue:

- After guided evolution wins, benchmark sets `result["candidate_formulas"] = fp_result.get("candidate_formulas")`.
- This drops proposer candidate skeletons and any candidate list actually sent into evolution.

Suggested fix:

- Store:
  - `fast_path_candidate_formulas`
  - `evolution_seed_candidates`
  - `proposer_candidate_formulas`
  - `winning_stage`

### Stage 7: Guided Evolution Budget

Call site:

- `scripts/benchmark_suite.py:1131-1140`

Process:

1. Compute `remaining_timeout = timeout - elapsed_since_fast_path_start`.
2. Put that into `guided_plan["timeout_seconds"]` if not already set.
3. Call `run_guided_evolution`.

Expected behavior:

- Evolution should respect remaining per-formula timeout.

Risk:

- Proposer budget can still increase population/generation counts before timeout stops the native run.
- The benchmark does not record requested vs actual evolution budget in result JSON.

Suggested fix:

- Record `guided_reason`, `dynamic_gens`, `dynamic_pop`, `guided_plan`, `n_beams`, `n_rounds`, and `timeout_seconds` in the result.

### Stage 8: Guided Evolution Wrapper

Function:

- `scripts.classifier_fast_path.run_guided_evolution`

Process:

1. Derive `n_beams` and `n_rounds` from generations.
2. Override them from search plan if present.
3. If confidence is high and candidate formulas exist, reduce beam count and rounds.
4. Call `beam_search_evolution`.
5. If C++ beam search fails, fallback to PyTorch evolution.

Observed issue:

- The wrapper returns beam search output if `beam_result["mse"] < inf`.
- There is no benchmark-visible field indicating whether the native C++ path or PyTorch fallback was used.

Suggested fix:

- Add `evolution_backend`, `fallback_used`, and `beam_search_failed_reason`.

### Stage 9: Beam Search / Native C++ Evolution

Function:

- `scripts.classifier_fast_path.beam_search_evolution`

Flow:

1. Convert torch tensors back to numpy.
2. Build operator priors from hints.
3. Detect polynomial mode and adapt power bounds.
4. Generate beam configs.
5. Convert candidate formulas into seed graphs.
6. Add signal-discovered seed graphs.
7. Run `_core.run_evolution` once using an island model.
8. Simplify the winning formula once.
9. Evaluate displayed formula MSE.
10. Return formula, display MSE, raw MSE, model, config, and C++ AST.

Expected output:

- `formula`: simplified display formula
- `mse`: display MSE
- `raw_mse`: native engine best MSE
- `display_mse`
- `time`
- `config`
- `cpp_ast`

Observed issues:

- `configs = make_beam_configs(...)` appears twice before the native run (`scripts/classifier_fast_path.py:3133` and `3141`). The first generated config list is overwritten. This is wasted work and can confuse debugging.
- `total_pop_size = base_pop_size * n_beams` and `total_generations = base_generations * n_rounds`. This is simple and fast, but it means small changes in beams/rounds multiply native compute.
- `num_threads` is set to all CPU cores via `multiprocessing.cpu_count()`. This is good for one formula but can oversubscribe badly if benchmark runs are parallelized externally.
- If post-simplification/evaluation throws, `display_mse` falls back to raw MSE. That can hide displayed-formula evaluation failure.

Suggested fix:

- Remove duplicate config generation.
- Record C++ actual thread count and native wall time in benchmark output.
- If display evaluation fails, return `display_mse=inf` plus an explicit `display_eval_error`, not raw MSE.
- Consider a benchmark CLI option for `--evolution-threads`.

### Stage 10: Guided Result Selection

Call site:

- `scripts/benchmark_suite.py:1150-1176`

Process:

1. Postprocess `guided_result["formula"]` with fidelity guard.
2. Set `guided_mse_raw = guided_result.get("mse")`.
3. Compute `guided_mse_display`.
4. Compare guided displayed MSE against current baseline displayed MSE.
5. Replace result only if:
   - evolution-only, or
   - fast path failed, or
   - guided displayed MSE is better than baseline displayed MSE.

Observed issue:

- `guided_mse_raw` uses `guided_result["mse"]`, but beam search returns display MSE under `"mse"` and raw native MSE under `"raw_mse"`.
- Therefore benchmark `mse_raw` after guided evolution is actually display MSE, not native raw MSE.
- The probe showed `mse_raw` and `mse_display` nearly identical after evolution for this reason.

Suggested fix:

- Set `guided_mse_raw = guided_result.get("raw_mse", guided_result.get("mse", inf))`.
- Preserve `guided_result["display_mse"]` separately.

### Stage 11: Final Display And Scoring

Call sites:

- `scripts/benchmark_suite.py:1184-1204`
- `scripts/benchmark_suite.py:741`

Process:

1. If formula exists and display MSE missing, evaluate it.
2. Use display MSE for `result["mse"]`.
3. Recompute divergence diagnostics.
4. Compute residual diagnostics from displayed formula.
5. Score:
   - `EXACT`: MSE `< 1e-6` and term count `<= 10`
   - `APPROX`: MSE `< 0.01`
   - `LOOSE`: MSE `< 0.1`
   - otherwise `FAIL`

Expected output:

- final formula string
- final displayed MSE
- final score
- residual diagnostics
- report row

Observed behavior from probe:

- Target: `x^3+sin(x)`
- Fast-path only:
  - printed internal MSE around `2.86e-06`
  - benchmark final score was `FAIL` because displayed MSE was much worse
- With evolution:
  - evolved formula: `0.9023 * x ** 3 + 0.8744 * x`
  - display MSE: about `0.00318`
  - score: `APPROX`
  - candidate seed preview was the fast-path formula, so evolution improved fit but did not recover the missing `sin(x)` structure.

### Stage 12: Report Writing

Call sites:

- `scripts/benchmark_suite.py:1919-1934`

Process:

1. Create output directory at end of run.
2. Build timestamp with second precision: `%Y%m%d_%H%M%S`.
3. Write JSON and Markdown reports.
4. Write `benchmark_latest.json` and `benchmark_latest.md`.

Observed issues:

- Output directory errors happen after formula execution, so a successful run can exit non-zero during reporting.
- Two benchmark processes started in the same second can generate the same timestamp filenames and overwrite each other.
- Probe runs launched concurrently both wrote `benchmark_20260601_164631.*`.

Suggested fix:

- Validate output dir before benchmark execution.
- Include microseconds or a short random suffix in report filenames.
- Optionally add process id to report filenames.
- Add `--no-report` for quick probes.

## Full Pipeline Fix Priorities

1. Add explicit intermediate MSE fields: internal model MSE, raw formula-string MSE, postprocessed formula MSE, evolution raw MSE, evolution display MSE.
2. Fix guided evolution result mapping so benchmark `mse_raw` uses `guided_result["raw_mse"]`.
3. Preserve the actual evolution seed candidate list in final results.
4. Move or strengthen residual-suspicion routing before low-MSE skip gates.
5. Remove duplicate `make_beam_configs` call.
6. Add backend/budget/thread diagnostics to guided evolution output.
7. Treat display evaluation failure as failure, not raw-MSE fallback.
8. Fix report directory validation and timestamp collisions.
