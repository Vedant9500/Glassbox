# Glassbox Research Roadmap

Last updated: 2026-06-03.

This roadmap supersedes older evolution-only framing. The current project is a
hybrid symbolic regression system with a fast path, universal proposer, C++
guided evolution, blackbox preprocessing, and specialist composition layers. The
research goal remains reliable formula discovery under honest displayed-formula
scoring.

## Current Position

Glassbox should be evaluated as a hybrid system:

- Easy formulas should be solved quickly by classifier/basis/exact-match paths.
- Uncertain or compositional formulas should receive proposer/specialist seeds
  and bounded guided C++ evolution.
- Blackbox/multivariate datasets should use feature ranking, interaction probes,
  reduced-space search, validation guards, and remapping to original variables.
- All benchmarks should score displayed formulas, not raw native fitness.

## Implemented Capabilities

### Benchmark Integrity

- Displayed-formula scoring is the benchmark contract.
- Raw MSE remains as `mse_raw` or diagnostic fields.
- Drift and formula-evaluation failure paths are covered by tests.
- Benchmark reports use JSON/Markdown outputs under `results/`.
- Local SRBench runner supports multi-seed summaries, time-to-discovery style
  fields, hard-timeout control, adaptive budget control, and failure taxonomy.

Primary files:

- `scripts/benchmark_suite.py`
- `scripts/benchmark_common.py`
- `scripts/run_srbench_local.py`
- `tests/test_benchmark_scoring_contract.py`
- `tests/test_run_srbench_local.py`

### Fast Path and Exact Match

- Curve classifier integration supports PyTorch checkpoints.
- Fast path builds basis terms, exact-match candidates, and regression formulas.
- Exact-match backend routing supports NumPy/torch CPU/CUDA controls.
- FPIP v2 payloads expose candidate skeletons, priors, uncertainty, diagnostics,
  and routing signals.

Primary files:

- `scripts/classifier_fast_path.py`
- `glassbox/curve_classifier/curve_classifier_integration.py`
- `glassbox/sr/fpip_v2.py`
- `tests/test_exact_match_backend_plumbing.py`
- `tests/test_fpip_v2_schema.py`

### Universal Proposer

- GLU proposer model implemented.
- Grammar-constrained univariate and multivariate skeleton decoding implemented.
- Proposer output maps to FPIP v2.
- Search-plan generation emits budget/beam/power/seed hints.
- Default benchmark model path is `models/universal_proposer_multi.pt`.

Primary files:

- `glassbox/universal_proposer/universal_proposer.py`
- `scripts/train_universal_proposer.py`
- `tests/test_universal_proposer.py`

### C++ Evolution and Native Helpers

- Native `_core.run_evolution` is the preferred guided evolution backend.
- Seed graph ingestion is wired from candidates and signal heuristics.
- Native candidate scoring, simplification, float snapping, noise reduction, and
  refinement helpers are exposed through pybind11.
- Tests cover candidate scoring, simplification, and seed graph behavior.

Primary files:

- `glassbox/sr/cpp/core.cpp`
- `glassbox/sr/cpp/evolution.h`
- `glassbox/sr/cpp/seed_graph_builder.py`
- `tests/test_cpp_candidate_scoring.py`
- `tests/test_cpp_simplification.py`
- `glassbox/sr/test_seed_graph_builder.py`

### Specialist and Blackbox Layers

- Specialist state, pair scoring, safe compositions, hot-spot segments, vault
  memory, residual symbolic stages, and inception reuse are implemented.
- Benchmark and SRBench CLIs expose specialist modes and ablations.
- Blackbox preprocessing implements feature ranking, interaction discovery,
  reduced-space search, remapping, and seed formulas.

Primary files:

- `glassbox/sr/specialist_state.py`
- `glassbox/sr/blackbox_preprocessor.py`
- `glassbox/sr/sklearn_wrapper.py`
- `scripts/specialist_phase_eval.py`
- `tests/test_specialist_state.py`
- `tests/test_specialist_phase_eval.py`
- `tests/test_sklearn_wrapper_cv_guard.py`

## Near-Term Priorities

### 1. Benchmark Discipline and Comparability

- Keep displayed-formula scoring as a hard invariant.
- Add clearer parity tables for PySR/SRBench-style comparisons: operator set,
  time budget, core/thread budget, seeds, complexity limit, and data splits.
- Avoid treating generated benchmark artifacts as source documentation.
- Expand smoke checks that combine proposer, specialist, blackbox, and C++ paths.

### 2. Blackbox/SRBench Reliability

- Improve validation-gated feature reduction and interaction selection.
- Make reduced-space vs original-space formula selection more transparent in
  reports.
- Continue capping fragile formula families under low trust.
- Improve official SRBench dataset discovery/metadata alignment where local data
  is available.

### 3. Specialist Cost Control

- Keep expensive residual/inception phases opt-in or strongly budget-gated.
- Reduce duplicate screening work.
- Improve metadata that distinguishes screening-only wins, composed-seed wins,
  residual wins, and evolution wins.
- Keep composed seeds capped so they do not dominate C++ initial populations.

### 4. C++/Python Synchronization

- Move more hot formula scoring and simplification work to native C++ where tests
  prove parity.
- Keep Python fallbacks for fresh installs.
- Add bridge tests whenever adding or changing native functions.
- Watch for raw/display drift introduced by native graph simplification.

### 5. Proposer Quality

- Improve replay-data coverage and multivariate skeleton vocabulary.
- Add stronger calibration for uncertainty and prior trust.
- Treat proposer output as guidance, not truth; maintain random/evolution
  diversity.
- Evaluate proposer benefits under displayed-formula and time-to-discovery
  metrics, not only candidate MSE.

## Medium-Term Research Work

- Better decomposition tests for sums, products, separability, symmetry, and
  rational structure.
- More principled multivariate interaction models beyond pairwise heuristics.
- Better bloat/parsimony control inside C++ evolution.
- Stronger coefficient refinement around selected symbolic structures.
- OOD/family-split validation to reduce train/test grammar aliasing.
- Real-world noisy dataset evaluation with stability summaries.

## Known Risks

- Hard nested/composed formulas remain brittle.
- Blackbox feature selection can overfit unless validation guards are enforced.
- Proposer seeds can bias evolution incorrectly when uncertainty is high.
- SymPy fallback can be slow or produce display changes; native simplification
  needs continued parity checks.
- CPU C++ evolution can consume all cores; benchmarks must report budgets.
- Local generated reports may reflect old code and should not be used as
  authoritative architecture docs.

## Success Metrics

Use these as current targets:

| Area | Metric |
|---|---|
| Formula fidelity | Displayed-formula MSE is always available or failure is explicit. |
| Easy-case speed | Fast path preserves low latency on tier 1-3 exact cases. |
| Hard-case recovery | Guided/proposer/specialist modes improve tiers 6-8 without hiding drift. |
| Blackbox robustness | Median and worst-decile R2 improve under multi-seed SRBench-style runs. |
| Stability | Multi-seed variance, failure taxonomy, and time-to-discovery are reported. |
| Native parity | C++ helpers match Python expectations in focused tests before becoming default. |

## Documentation Rule

When CLI defaults, model paths, pipeline stages, or scoring contracts change,
update:

1. `README.md`
2. `docs/PROJECT_MAP.md`
3. `docs/ONN_Architecture.md`
4. `docs/onn_runbook.md`
5. any plan/audit file directly tied to the changed component
