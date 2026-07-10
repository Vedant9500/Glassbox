# PhySO Noise Handling — Phase Tracker

Multi-phase plan to close the noise-handling gap with PhySO.
Source of truth for the *what* and *why* lives in `physo_noise_handling.md`.
Measurement-bug audit (2026-07-10): `noise_handling_audit.md`.
This file tracks **status** so we don't lose the thread between phases.

Legend: `[ ]` not started · `[~]` in progress · `[x]` done

---

## Phase 0 — Baseline And Instrumentation

Goal: make current noise behaviour measurable before changing objectives.

Key files: `glassbox/sr/cpp/evolution.h`, `glassbox/sr/cpp/core.cpp`,
`glassbox/sr/cpp/simplify_advanced.h`, `glassbox/sr/sklearn_wrapper.py`,
`scripts/classifier_fast_path.py`, `scripts/benchmark_common.py`,
`scripts/run_srbench_local.py`,
`glassbox/curve_classifier/generate_curve_data.py`.

- [x] Add benchmark report columns: `noise_level`, `noise_type`,
      `sample_weight_mode`, `raw_mse`, `display_mse`, `holdout_mse`,
      `formula_complexity`, `false_confidence`, `seed_graphs_used`.
- [x] Add clean-recovery columns to protocol: `clean_test_mse`,
      `clean_test_r2`, `clean_full_mse`, `acceptable_clean`,
      `false_confidence_vs_clean` (suite: `mse_clean`, `r2_clean`,
      `recovery_exact`, `recovery_acceptable`). See audit.
- [x] Build fixed noisy suites: clean, 0.1%, 1%, 10% RMS Gaussian,
      pink, quantization, sparse outliers.
- [x] Protocol CLI: `python scripts/benchmark_noise.py` (+ `--smoke`).
- [~] Run ≥5 seeds/tier on noisy Feynman + easy synthetic **via**
      `scripts/benchmark_noise.py` and store `noise_protocol_*.json`
      (CLI ready; full baseline still needs a torch-enabled run).
      Suite `--noise high` probes under `results/before_noise/` remain
      noisy-fit EXACT only — prefer CleanMSE/Recov / protocol clean columns.
- [x] Constant-target noise uses amplitude fallback (not free under noise).
- [x] Capture native trace stats via existing `trace_path` in `core.cpp`.
- [x] Do NOT tune thresholds on a single run; keep displayed MSE primary.
- [x] Do NOT compare vs PhySO unless noise protocol + budgets explicit.
- [x] Suite `Score/EXACT` under noise is a noisy-fit band only — do not
      treat as structure recovery (prefer CleanMSE/Recov).

**Expected outcome & when visible:** A reproducible baseline table (exact/acceptable recovery, displayed MSE, complexity, stability) and a known-failure-bucket list. Visible immediately once the suite runs — this phase produces *measurement*, not improvement. Every later phase's claim is validated against this baseline, so nothing downstream is credible until Phase 0 exists. No user-facing formula quality change in this phase.

**Outcome**: reproducible baseline table (exact/acceptable recovery,
displayed MSE, complexity, stability) + known failure buckets
(overfit complexity, wrong-structure-high-fit, unstable CV, residual
memorization, display/raw drift).

---

## Phase 1 — Public Weight And Uncertainty Contract

Goal: expose PhySO-style uncertainty-aware fitting at the estimator
boundary. *(detailed sub-plan: `phase1_sample_weight_plan.md`)*

- [x] Add `sample_weight=None` to `GlassboxRegressor.fit(X, y, ...)`.
- [ ] Optional `y_uncertainty=None` with documented `weight = 1/sigma^2`
      conversion (not implemented yet; do not claim done).
- [x] Normalize weights: finite, non-negative, mean ≈ 1; document
      zero-weight rule.
- [x] Store diagnostics: effective sample size, min/max weight
      (source/skipped counts still minimal).
- [x] Thread weights through `_formula_mse` and CV skip guard.
- [x] Fail-loud on weight length mismatches; slice weights for holdout
      (`sample_weight_indices` / `_slice_sample_weight`).
- [ ] Thread weights through `_display_formula_mse`, display-first final
      scoring, Pareto selection, and candidate screening
      (display path still unweighted by design for benchmark parity;
      Pareto still unweighted; candidate screening weight-aware as of Phase 2;
      evolution remains Phase 3).
- [x] Do NOT change default behaviour when no weights supplied.
- [x] Tests: uniform == none; invalid raises; `sample_weight_` stored;
      CV guard weight-aware. (Selection-shift under weights is limited
      until display/Pareto/C++ paths are weight-aware.)

**Expected outcome & when visible:** `fit(X, y, sample_weight=...)` works end-to-end; weights flow into `_formula_mse`, the CV skip guard, final Python-side selection, and `blackbox_diagnostics_`. *Partial now:* downweighted outlier points already change the CV guard's pass/fail decision and the reported weighted MSE/R², so a user supplying weights will see different *trust* and *diagnostics*. *Not yet visible:* the actual *discovered formula skeleton* is still chosen mostly by the unweighted C++ candidate scorer (`score_formula_candidates`) and unweighted evolution (`run_evolution_cpp`), so under heavy noise the chosen expression may not change much until **Phase 2** (weighted screening) and **Phase 3** (weighted evolution) land. Full PhySO-parity benefit from weights therefore unlocks in Phase 3.

**Outcome (current)**: user can pass `sample_weight`; diagnostics and
CV guard honour it; `_formula_mse` honours it when lengths match.
Final display/Pareto selection and C++ search still largely unweighted.
Existing tests pass with `sample_weight=None`.

---

## Phase 2 — Weighted Native Candidate Scoring

Goal: C++ batch candidate scorer matches weighted validation semantics.

- [x] Extend `score_formula_candidates_cpp` (`core.cpp`) with optional
      `fit_weights` and `val_weights`.
- [x] Replace affine fit mean/var/cov with weighted versions.
- [x] Return both weighted + unweighted: `weighted_fit_mse`,
      `weighted_validation_mse`, plus legacy primary `mse`/`r2`
      (primary becomes weighted when weights provided; unweighted
      diagnostics always present as `unweighted_*`).
- [x] Update `_refine_candidate_formulas` + specialist probes to pass
      split / full-array weights via `_split_sample_weights`.
- [x] Python fallback `_score_formula_candidate` is weight-aware.
- [x] Tests in `tests/test_cpp_candidate_scoring.py` for weighted
      affine scale/bias, outlier downweight, bad weight length, and
      Python fallback.
- [x] Do NOT overwrite unweighted diagnostics — dual metrics returned.
- [x] Weighted R² uses weighted variance; invalid weights raise.

**Expected outcome & when visible:** Candidate pools stop favouring formulas that fit known-noisy points; validation R² and acceptance thresholds become weight-aware. Visible standalone once shipped — the fast-path / specialist screening that picks the incumbent formula before evolution runs will now respect `sample_weight`, so even without Phase 3, weighted data starts shifting which candidates survive screening. Final-formula change under noise becomes substantial once combined with Phase 3's weighted evolution.

**Outcome**: candidate pools stop favouring formulas that fit known-noisy
points; acceptance thresholds become weight-aware.

---

## Phase 3 — Weighted C++ Evolution Objective

Goal: native evolution optimises weighted loss, not only unweighted MSE.

- [ ] Add optional `y_weights` to `run_evolution_cpp` (`core.cpp`);
      store in `EvolutionEngine`.
- [ ] Update `evaluate_fitness_with_penalty` (`evolution.h`) to use
      weighted MSE for `fitness`, keep unweighted `raw_mse` for diag.
- [ ] Extend `DifferentialGramian`, `solve_output_weights`, snapping
      MSE, cleanup backward elimination, inner-param refinement to
      weighted residuals.
- [ ] Return `best_weighted_mse`, `best_mse`, weight diag separately.
- [ ] Native tests in `glassbox/sr/test_cpp_parity.py` proving weighted
      evolution changes choice when outliers downweighted.
- [ ] Do NOT remove unweighted `raw_mse` (bench/back-compat need it).
- [ ] Do NOT apply complexity penalty inconsistently between weighted /
      unweighted metrics without naming it.

**Expected outcome & when visible:** This is where weights deliver their full payoff. Evolution optimises weighted loss, so the *structure* it discovers changes when known-bad observations are downweighted — correct simple structure recovers even when outliers would otherwise dominate. Combined with Phase 1 + 2, this is the first phase that can reproduce PhySO's 'downweight noisy region → recover the right formula' behaviour end-to-end. Pareto front also exposes the weighted-fit vs unweighted-robustness tradeoff. Not visible in isolation: needs Phase 1 (weights exist) and ideally Phase 2 (screening consistent).

**Outcome**: evolution recovers correct simple structure when known-bad
observations would otherwise dominate; Pareto front exposes
weighted-fit vs unweighted-robustness tradeoff.

---

## Phase 4 — Robust Loss Modes For Unknown Noise

Goal: handle noise when user has no weights.

- [ ] Add `loss_mode`: `mse`, `huber`, `trimmed_mse`, `student_t`.
- [ ] Start in Python candidate screening + final scoring, then port
      minimal `huber` / `trimmed_mse` to C++ evolution.
- [ ] Residual-scale estimation via MAD on validation residuals.
- [ ] Use robust loss for search fitness, keep displayed unweighted
      MSE/R² in reports.
- [ ] Do NOT default to robust loss until benchmark evidence supports it.
- [ ] Do NOT combine robust loss with aggressive residual boosting
      without stricter holdout guards.

**Expected outcome & when visible:** Better recovery under outliers and quantization *without* requiring user-supplied weights — this is the no-weights safety net. Standalone visible: a user hitting heavy-tailed or quantization noise gets more robust formulas even with uniform weights. Diagnostics record when robust loss changed the final formula. Most powerful when layered on top of Phase 3 (weighted + robust), but it is *not* a no-op on its own.

**Outcome**: better recovery under outliers/quantization without
requiring user uncertainties; diagnostics show when robust loss
changed the final formula.

---

## Phase 5 — Units And Physics Priors In Active Pipeline

Goal: borrow PhySO's strongest structural defence — shrink search
before noise can create wrong fits.

- [ ] Promote existing C++ `input_units`, `output_units`,
      `dim_penalty_weight` into `GlassboxRegressor` public API.
- [ ] Add unit validation + dimensionless-default examples.
- [ ] Use units to filter candidate formulas and seed graphs before
      C++ evolution.
- [ ] Add hard vs soft unit modes: hard reject impossible, soft penalty
      for unitless/uncertain.
- [ ] Do NOT require units for normal ML/tabular use.
- [ ] Do NOT apply unit penalties when unit can't be inferred safely;
      don't mix incompatible unit-vector lengths.

**Expected outcome & when visible:** On physics-style noisy data, the search space shrinks before noise can drive wrong fits — this is PhySO's single strongest noise defence, so it is high-impact standalone. Users who supply `input_units`/`output_units` get physics-constrained SR where high-MSE-but-physical simple formulas beat low-MSE unphysical overfits. Visible immediately for physics-unit users; no effect for tabular ML users who omit units (back-compat preserved).

**Outcome**: on physics-style noisy data, search space shrinks and
high-MSE/simple-physical candidates beat low-MSE/unphysical overfits;
public API can state when Glassbox is physics-constrained.

---

## Phase 6 — Noise-Aware Selection, Cleanup, And Residual Guards

Goal: prevent post-processing and residual stages from fitting noise.

- [ ] Extend `_cleanup_formula_with_fidelity_guard` and
      `reduce_formula_noise_cpp` with weights + holdout checks.
- [ ] Replace fixed cleanup slack with noise-aware slack based on
      residual scale + validation variance.
- [ ] Strengthen `_select_blackbox_pareto_formula`: weighted validation,
      edge validation, complexity, risk, generalization gap, robust
      residual diagnostics.
- [ ] Residual-stage acceptance rule: candidate must improve weighted
      validation AND not worsen unweighted/edge validation beyond slack.
- [ ] Track `noise_pruned_terms`, `cleanup_rejected_reason`,
      `residual_rejected_as_noise`.
- [ ] Do NOT accept residual additions on training MSE alone.
- [ ] Do NOT let BIC pruning remove physically required small terms
      without holdout/fidelity guard.
- [ ] Do NOT use one slack value for clean and 10% noisy data.

**Expected outcome & when visible:** Final formulas become simpler and less noise-shaped; residual/specialist stages improve true structure instead of memorising residual noise. *Not visible in isolation:* cleanup (`reduce_formula_noise_cpp`) and residual guards only become weight/noise-aware once **Phase 1** (weights exist), **Phase 2** (weighted scoring), and **Phase 3** (weighted evolution) are in place — otherwise there is no weighted signal for the guards to use. The visible win is 'we stop re-introducing noise-shaped terms during cleanup/residual passes', which matters most after Phases 1–3 have already produced a good weighted candidate.

**Outcome**: final formulas simpler and less noise-shaped;
residual/specialist stages improve true structure instead of
memorising residual noise.

---

## Phase 7 — Training And Routing Calibration

Goal: make classifier/proposer uncertainty calibrated for noisy runtime
decisions.

- [ ] Train/evaluate proposer + classifier on explicit noise profiles
      from `generate_curve_data.py`: clean, low/med/high Gaussian, pink,
      quantization, outliers.
- [ ] Add validation metrics by noise bucket: candidate recall,
      operator F1, skeleton confidence reliability, false-confidence rate.
- [ ] Feed runtime noise diagnostics into `_derive_blackbox_search_plan`:
      residual autocorrelation, outlier fraction, weight effective
      sample size, validation gap.
- [ ] Calibrate `prediction_uncertain`, `candidate_acceptance_r2`,
      `candidate_shrink_r2` per noise band.
- [ ] Do NOT trust raw skeleton logits unless reliability gates pass.
- [ ] Do NOT shrink search diversity just because noisy candidate MSE
      is low.
- [ ] Do NOT train on noise profiles the benchmark never reports.

**Expected outcome & when visible:** Search budget rises on ambiguous/noisy cases and shrinks only on stable verified formulas; diagnostics explain *why* Glassbox trusted or rejected a noisy candidate. Not visible in isolation — calibration feeds on the noise diagnostics produced by Phases 1–6 (effective sample size, residual autocorrelation, validation gap). Until those exist, there is little to calibrate against. The benefit is adaptive *compute spend*, so its effect shows up as budget-vs-recovery curves rather than a single formula change.

**Outcome**: search budget rises on ambiguous/noisy cases and shrinks
only on stable verified formulas; diagnostics explain why Glassbox
trusted or rejected noisy candidates.

---

## Phase 8 — Benchmark Release Gate

Goal: make noise handling a release-tested feature, not a claim.

- [ ] Add `scripts/benchmark_noise.py` (or extend `run_srbench_local.py`)
      with deterministic noise protocol.
- [ ] Report clean vs noisy deltas, not only absolute scores.
- [ ] Ablations: no weights, no robust loss, no units, no CV guard, no
      uncertainty routing, no noise pruning.
- [ ] CI smoke tests for small weighted case + one noisy outlier
      recovery case.
- [ ] Do NOT optimise only for exact recovery; include acceptable
      simple-formula rate.
- [ ] Do NOT hide failed seeds.
- [ ] Do NOT compare methods with different timeouts without reporting
      budgets.

**Expected outcome & when visible:** A stable release gate — no merge regresses noisy recovery, false-confidence rate, or displayed-formula scoring — plus a credible comparison table vs PhySO/PySR/gplearn under an identical noise protocol. This phase produces *evidence*, not behaviour. Not useful until Phases 1–7 are shipped; its whole point is to prove their combined effect and prevent regression. Visible as benchmark tables and CI gate status, not as a runtime formula difference.

**Outcome**: stable release gate — no merge regresses noisy recovery,
false-confidence rate, or displayed-formula scoring; credible
comparison table vs PhySO/PySR/gplearn under identical noise protocol.

---

## Implementation Order

1. Phase 0 — baseline.
2. Phase 1 — Python `sample_weight` contract.
3. Phase 2 — weighted C++ candidate scorer.
4. Phase 3 — weighted native evolution.
5. Phase 6 — cleanup/residual guards.
6. Phase 4 — robust loss.
7. Phase 5 — units API.
8. Phase 7 — calibration.
9. Phase 8 — release gate.

Rationale (from `physo_noise_handling.md`): weights are the lowest-risk
PhySO lesson and must become end-to-end before robust loss or units add
more moving parts.
