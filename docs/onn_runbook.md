# Glassbox Runbook

Last updated: 2026-06-03.

This runbook applies to the active hybrid runtime: classifier fast path,
universal proposer, specialist phases, C++ evolution, formula postprocessing,
and benchmark/SRBench runners. ONN-only experiments should still follow the
same scoring and rollback rules when they affect default behavior.

## Release Gates

Before changing a default path, require evidence from focused tests and at least
one benchmark smoke run that covers the touched behavior.

1. Displayed-formula scoring gate
- Benchmark labels must be based on displayed/evaluable formula MSE.
- Raw C++/fast-path MSE can only be diagnostic.
- New code must preserve drift diagnostics (`mse_raw`, `mse_display`, failure
  flags where applicable).

2. Quality gate
- EXACT/APPROX/LOOSE/FAIL counts must not regress on the monitored tier set
  unless the change is explicitly a tradeoff and documented.
- For specialist or blackbox changes, check at least one hard/compositional or
  multivariate case, not only easy tiers.

3. Runtime gate
- Runtime should not materially increase for easy exact cases.
- If a change spends more time, it must be bounded by timeout/search-plan
  controls and justified by hard-case quality.

4. Safety gate
- Formula simplification must be guarded by fidelity checks.
- C++ bridge changes must keep Python fallback behavior when `_core` is missing.
- Device/CUDA selection must fall back to CPU without crashing.

5. Reproducibility gate
- Use fixed seeds for benchmark comparisons when possible.
- Report device, timeout, model paths, and specialist/proposer flags.

## Default Smoke Matrix

Run focused unit tests first:

```bash
pytest tests/test_benchmark_scoring_contract.py -q
pytest tests/test_sklearn_wrapper_cv_guard.py -q
pytest tests/test_exact_match_backend_plumbing.py -q
```

Add targeted tests depending on the change:

```bash
pytest tests/test_universal_proposer.py -q
pytest tests/test_specialist_state.py tests/test_specialist_phase_eval.py -q
pytest tests/test_cpp_candidate_scoring.py tests/test_cpp_simplification.py -q
pytest tests/test_run_srbench_local.py -q
```

Run at least one benchmark smoke:

```bash
python scripts/benchmark_suite.py --tier 2 --tier 3 --device cpu --quiet
python scripts/benchmark_suite.py --tier 6 --with-evolution --device cpu --quiet
python scripts/run_srbench_local.py --track 2 --max-datasets 3 --no-hard-timeout
```

For specialist changes:

```bash
python scripts/benchmark_suite.py --tier 6 --specialist-regressor --device cpu --quiet
python scripts/benchmark_suite.py --tier 6 --specialist-regressor --specialist-full --device cpu --quiet
```

## Rollback Triggers

Rollback or disable the new default if any of these occur:

- Displayed-formula scoring regresses while raw MSE appears improved.
- Simplification produces formulas that fail evaluation or worsen guarded MSE.
- Easy exact cases route into expensive evolution without a clear reason.
- Specialist or residual phases run far beyond the configured budget.
- C++ evolution, candidate scoring, or seed graph ingestion crashes without
  fallback.
- CUDA/torch exact-match routing crashes on CPU-only environments.
- Benchmark reports cannot be written or omit required MSE diagnostics.

## Rollback Levers

- Set `GLASSBOX_USE_LEGACY_FASTPATH=1` to make `GlassboxRegressor` use legacy
  fast-path routing behavior where supported.
- Use `--disable-proposer` in `scripts/benchmark_suite.py`.
- Use `--specialist-baseline`, `--disable-specialist-composition`, or omit
  `--specialist-full` for specialist ablations.
- Use `--exact-match-backend numpy` to bypass torch exact-match routing.
- Use `--no-fast-path` or `--no-guided-evolution` in `scripts/run_srbench_local.py`
  for SRBench ablations.
- Use estimator constructor flags to disable `use_fast_path`,
  `use_guided_evolution`, `enable_residual_stage`,
  `enable_specialist_composition_screening`, or `enable_inception_reuse`.

## Failure Playbook

### Displayed MSE Is Much Worse Than Raw MSE

- Inspect postprocessed formula and formula-eval diagnostics.
- Re-run with simplification disabled or native simplification isolated.
- Add/adjust a fidelity guard test before accepting the displayed form.

### Fast Path Stalls On Exact Match

- Lower `--exact-match-max-combos`.
- Try `--exact-match-backend numpy` or CPU torch backend.
- Check basis size and auto-expansion decisions in `scripts/classifier_fast_path.py`.

### Proposer Seeds Make Evolution Worse

- Inspect FPIP v2 `sequence_uncertainty`, `operator_priors`, and `search_plan`.
- Blend priors closer to uniform when uncertainty is high.
- Keep random explorer/island diversity and cap seed budget.

### Specialist Runtime Blows Up

- Check residual/inception flags first; they are intentionally more expensive.
- Confirm candidate caps and composed seed caps are active.
- Prefer screening-only defaults unless hard-case benchmark evidence supports
  enabling deeper phases.

### Blackbox Results Overfit Train Data

- Inspect validation/holdout diagnostics and final Pareto selection.
- Prefer stable/simple candidates when validation MSE is comparable.
- Confirm reduced-space formulas are remapped back to original variables before
  final scoring.

### C++ Extension Is Missing

- Rebuild:

```bash
cd glassbox/sr/cpp
python setup.py build_ext --inplace
cd ../../..
```

- If rebuild is not possible, tests and runtime paths should either skip native
  checks or use Python fallbacks.

## Ownership

- Fast path and guided evolution: `scripts/classifier_fast_path.py`.
- Benchmark scoring/reporting: `scripts/benchmark_suite.py` and
  `scripts/benchmark_common.py`.
- Sklearn runtime: `glassbox/sr/sklearn_wrapper.py`.
- C++ bridge/backend: `glassbox/sr/cpp/`.
- Proposer: `glassbox/universal_proposer/` and
  `scripts/train_universal_proposer.py`.
- Specialist/blackbox: `glassbox/sr/specialist_state.py` and
  `glassbox/sr/blackbox_preprocessor.py`.

Every default-changing PR should include commands run, short result summaries,
and any formulas/datasets that regressed.
