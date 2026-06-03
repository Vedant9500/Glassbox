# C++ Migration Roadmap

Last updated: 2026-06-03.

Glassbox already has a substantial native backend. This document tracks what is
implemented and what remains worth migrating.

## Current Native Surface

The pybind11 bridge is `glassbox/sr/cpp/core.cpp`. Current exports include:

- `run_evolution`
- `score_formula_candidates`
- `refine_frequencies`
- `refine_powers`
- `refine_periodic_rational`
- `iterative_elastic_net`
- `lasso_coordinate_descent`
- `simplify_formula_cpp`
- `formula_to_seed_graph_cpp`
- `snap_formula_floats_cpp`
- `reduce_formula_noise_cpp`

Supporting native/source files:

- `evolution.h`: island evolution/search.
- `eval.h`: evaluation helpers.
- `ast.h`: expression graph structures.
- `formula_parser.h`: formula parsing into seed graphs.
- `simplify.h` and `simplify_advanced.h`: native simplification and float
  snapping.
- `refine.h`: numerical refinement helpers.
- `seed_graph_builder.py`: Python helper that prepares seed graphs for the C++
  bridge.

## Implemented Migrations

| Area | Status |
|---|---|
| Guided evolution | Implemented through `_core.run_evolution`. |
| Candidate formula scoring | Implemented through `score_formula_candidates`; covered by tests. |
| Seed graph parsing/bridge | Implemented through formula-to-seed helpers and tests. |
| Native formula simplification/snap | Implemented as C++ bridge helpers with Python/SymPy fallback. |
| Native LASSO/elastic-net helpers | Exposed through pybind11. |
| Frequency/power/periodic-rational refinement | Exposed through pybind11. |

## Remaining Migration Opportunities

### 1. Full Simplification Parity

SymPy remains in `requirements.txt` because Python fallback simplification is
still used in several paths. Continue migrating only when native output is
covered by fidelity tests and does not increase displayed/raw drift.

Primary files:

- `scripts/simplify_formula.py`
- `scripts/benchmark_common.py`
- `glassbox/sr/sklearn_wrapper.py`
- `glassbox/sr/cpp/simplify*.h`

### 2. Classifier/Proposer Inference Runtime

Classifier/proposer loading currently uses PyTorch or XGBoost. ONNX Runtime or a
small Eigen inference path could reduce startup overhead, but only after the
model format and feature preprocessing are stable.

Primary files:

- `glassbox/curve_classifier/curve_classifier_integration.py`
- `glassbox/universal_proposer/universal_proposer.py`
- `scripts/train_universal_proposer.py`

### 3. More Formula Evaluation and Guarding

Shared formula evaluation still has Python/SymPy parsing paths. Native
evaluation could speed scoring, but it must match the project's protected
semantics for signed fractional powers, log handling, constants, and displayed
formula syntax.

Primary files:

- `scripts/benchmark_common.py`
- `scripts/benchmark_suite.py`
- `scripts/run_srbench_local.py`
- `glassbox/sr/cpp/eval.h`

### 4. Specialist Candidate Screening

Specialist screening can generate and score many candidate formulas. Native
batch scoring is partially available; more candidate preparation/refinement
could move native if tests prove parity.

Primary files:

- `glassbox/sr/specialist_state.py`
- `glassbox/sr/sklearn_wrapper.py`
- `glassbox/sr/cpp/core.cpp`

## Guardrails

- Native helpers must preserve displayed-formula scoring semantics.
- Python fallbacks should remain available for fresh installs without `_core`.
- Tests should cover every pybind11 behavior change.
- Do not remove SymPy or PyTorch from dependencies until all active runtime paths
  that require them have native or optional alternatives.

## Verification

Useful focused tests:

```bash
pytest tests/test_cpp_candidate_scoring.py -q
pytest tests/test_cpp_simplification.py -q
pytest tests/test_benchmark_common.py -q
pytest tests/test_benchmark_scoring_contract.py -q
```
