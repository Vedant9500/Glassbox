# Glassbox Project Map

Last updated: 2026-06-03.

This map describes the current worktree. Generated benchmark reports under
`results/` and local checkpoints under `models/` are runtime artifacts, not
source documentation.

## Repository Roles

| Path | Role |
|---|---|
| `glassbox/` | Python package code. |
| `glassbox/sr/sklearn_wrapper.py` | Main sklearn-compatible estimator and orchestration layer. |
| `glassbox/sr/cpp/` | Native C++ backend, pybind11 bridge, seed graph parser/export, simplification helpers. |
| `glassbox/curve_classifier/` | Curve classifier architectures, feature extraction, training-data generation, model loading, operator prediction. |
| `glassbox/universal_proposer/` | Universal proposer model, grammar-constrained skeleton decoding, search-plan generation, FPIP v2 adapter. |
| `scripts/` | CLI entry points for benchmarks, SRBench-style runs, simplification, training, and diagnostics. |
| `tests/` | Regression/unit tests for active behavior. |
| `docs/` | Maintained source documentation. |
| `research_notes/` | Forward-looking audits and research notes; not all suggestions are implemented. |
| `results/` | Generated benchmark outputs. |
| `models/` | Local model checkpoints used by CLIs and estimator defaults. |

## Active Pipeline

The current default runtime is `GlassboxRegressor` in
`glassbox/sr/sklearn_wrapper.py`.

1. `fit(X, y)` validates sklearn-style inputs and initializes diagnostics.
2. If blackbox mode is enabled, `glassbox/sr/blackbox_preprocessor.py` ranks
   features, optionally reduces the search space, discovers interactions, and
   builds blackbox seed formulas.
3. The fast path in `scripts/classifier_fast_path.py` uses classifier
   predictions, basis construction, exact-match search, sparse/linear fitting,
   residual diagnostics, and FPIP v2 handoff.
4. The universal proposer in `glassbox/universal_proposer/universal_proposer.py`
   can emit candidate skeletons, operator priors, uncertainty, multivariate
   hints, and an evolution search plan.
5. Specialist logic in `glassbox/sr/specialist_state.py` computes local
   candidate pools, pair scores, compositions, hot-spot segments, vault entries,
   residual-stage proposals, and inception/subexpression reuse metadata.
6. Guided evolution in `scripts/classifier_fast_path.py` builds seed graphs
   with `glassbox/sr/cpp/seed_graph_builder.py` and calls `_core.run_evolution`.
7. Final formula selection uses direct formula evaluation and displayed-formula
   fidelity guards before storing `formula_` and `best_mse_`.

## Package Modules

### `glassbox/sr`

| File or directory | Current purpose |
|---|---|
| `sklearn_wrapper.py` | Full estimator pipeline: fast path, proposer, blackbox preprocessing, specialist phases, residual/inception passes, guided evolution, final formula cleanup. |
| `blackbox_preprocessor.py` | Feature ranking, interaction discovery, reduced-space remapping, blackbox seed formulas. |
| `specialist_state.py` | Specialist candidate state, pair scores, compositions, vault memory, segment/hot-spot helpers. |
| `fpip_v2.py` | Stable payload contract between fast path/proposer and guided evolution. |
| `core/operation_dag.py` | ONN DAG model. |
| `core/operation_node.py` | ONN operation node and routing. |
| `operations/meta_ops.py` | Meta operations and formula normalization helpers. |
| `optimizers/bfgs_optimizer.py` | BFGS-style constant refinement. |
| `optimizers/hybrid_optimizer.py` | Hybrid optimization support. |
| `hard_concrete.py` | Differentiable hard-concrete selection utilities. |
| `pruning.py` | Formula/model pruning helpers. |
| `phased_regression.py` | Phased symbolic regression utilities. |
| `risk_seeking_policy_gradient.py` | Policy-gradient research mixin for evolution. |
| `visualization.py` | ONN visualizers and network diagrams. |

### `glassbox/sr/cpp`

| File | Current purpose |
|---|---|
| `core.cpp` | pybind11 bridge exposing native functions. |
| `evolution.h` | Native C++ evolution/island search implementation. |
| `eval.h` | Formula/graph evaluation helpers. |
| `ast.h` | C++ expression/graph structures. |
| `formula_parser.h` | Formula-to-graph parsing. |
| `seed_graph_builder.py` | Python formula/signal/candidate seed graph builder for the C++ bridge. |
| `simplify.h`, `simplify_advanced.h` | Native formula simplification and float snapping helpers. |
| `refine.h` | Native constant/frequency/power refinement helpers. |
| `execution.h` | Execution support utilities. |
| `export_pytorch.py` | Wraps C++ graph results for PyTorch-compatible use where possible. |
| `setup.py`, `CMakeLists.txt` | Extension build configuration. |

The active pybind11 exports include `run_evolution`,
`score_formula_candidates`, `simplify_formula_cpp`,
`formula_to_seed_graph_cpp`, `snap_formula_floats_cpp`, and
`reduce_formula_noise_cpp`.

### `glassbox/curve_classifier`

| File | Current purpose |
|---|---|
| `curve_classifier_integration.py` | Loads PyTorch/XGBoost classifiers, resolves devices, predicts operators, detects multi-input interactions, and biases ONN/evolution priors. |
| `generate_curve_data.py` | Synthetic formula generation, PCFG/depth-annealed generation, noise augmentation, feature extraction, invariant/FFT/derivative/curvature features. |
| `train_curve_classifier.py` | PyTorch classifier training path. |

### `glassbox/universal_proposer`

| File | Current purpose |
|---|---|
| `universal_proposer.py` | GLU proposer model, grammar decoders, multivariate skeleton decoding, uncertainty, search-plan builders, checkpoint loading, FPIP v2 mapping. |

### `glassbox/evolution`

| File | Current purpose |
|---|---|
| `evolution.py` | Python ONN/evolution research path: trainer, mutations, refinement, pruning, coefficient finalization, and public training helpers. |

This path is still used as fallback/research infrastructure, but the active
benchmark path prefers C++ evolution when available.

## Script Entry Points

| Script | Purpose |
|---|---|
| `scripts/benchmark_suite.py` | Main 8-tier synthetic benchmark with fast path, guided evolution, pure C++ mode, and specialist-regressor mode. |
| `scripts/run_srbench_local.py` | Local SRBench-style Track 1/Track 2 runner with multi-seed summaries and blackbox controls. |
| `scripts/benchmark_common.py` | Shared formula parsing, postprocessing, scoring, stability, and failure taxonomy helpers. |
| `scripts/classifier_fast_path.py` | Fast-path basis/exact-match implementation plus guided C++ evolution wrapper. |
| `scripts/benchmark_feynman_easy.py` | AI-Feynman easy dataset runner. |
| `scripts/specialist_phase_eval.py` | Specialist phase evaluation harness. |
| `scripts/simplify_formula.py` | Python/SymPy simplification fallback and CLI. |
| `scripts/evolution_pipeline_log.py` | Pipeline tracing and C++ JSONL population snapshots. |
| `scripts/sr_tester.py` | Interactive/single-formula tester. |
| `scripts/train_universal_proposer.py` | Universal proposer training and replay-data loading. |
| `scripts/train_xgboost_classifier.py` | XGBoost classifier training. |
| `scripts/calibrate_classifier.py` | Classifier calibration. |
| `scripts/verify_fast_path.py` | Fast-path smoke verification. |

## Test Coverage Map

| Test file | Behavior covered |
|---|---|
| `tests/test_benchmark_scoring_contract.py` | Displayed-formula scoring, proposer/guided routing, specialist benchmark defaults. |
| `tests/test_benchmark_common.py` | Shared postprocessing/evaluation helpers. |
| `tests/test_run_srbench_local.py` | SRBench harness, budgets, formula evaluation, specialist flags. |
| `tests/test_sklearn_wrapper_cv_guard.py` | Estimator guardrails, blackbox routing, final formula selection, exact-match backend plumbing. |
| `tests/test_specialist_state.py` | Specialist candidate/composition/vault helpers. |
| `tests/test_specialist_phase_eval.py` | Specialist phase harness and candidate scoring. |
| `tests/test_universal_proposer.py` | Proposer decode, FPIP mapping, replay dataset. |
| `tests/test_fpip_v2_schema.py` | FPIP v2 builder/validator. |
| `tests/test_cpp_candidate_scoring.py` | Native candidate scoring. |
| `tests/test_cpp_simplification.py` | Native simplification bridge. |
| `tests/test_exact_match_backend_plumbing.py` | Exact-match backend parameter wiring. |
| `tests/test_feature_extraction.py` | Classifier feature extraction. |
| `tests/test_pcfg_generator.py` | Synthetic formula generation. |
| `tests/test_evolution_reliability.py` | Python evolution reliability guards. |

## Documentation Status

| Document | Status |
|---|---|
| `README.md` | Current runtime overview and quick start. |
| `docs/PROJECT_MAP.md` | Current source-of-truth map. |
| `docs/ONN_Architecture.md` | Current hybrid architecture summary plus ONN legacy/research notes. |
| `docs/onn_runbook.md` | Current release gates and smoke matrix. |
| `docs/Research_Roadmap.md` | Current roadmap/status, with older evolution-first context superseded where noted. |
| `docs/Universal_FastPath_AB_Report.md` | A/B status summary, still qualitative unless paired with generated benchmark artifacts. |
| `docs/CPP_Migration_Roadmap.md` | Migration status and remaining C++ work. |
| `research_notes/*.md` | Research/audit notes. They are intentionally broader than implemented code. |
