# C++ Candidate Scoring Migration

## Problem

Benchmark runtime increased from roughly 500-600 seconds to 2000+ seconds after the specialist pipeline grew more expensive.

Profiling showed two main costs:

1. The largest bottleneck is the residual specialist path. `_stage_residual_symbolic_fit()` launches a nested `GlassboxRegressor.fit()` on residuals, which can recursively run fast-path, proposer, candidate building, screening, and evolution.
2. The secondary bottleneck is candidate/probe scoring. `_refine_candidate_formulas()` and targeted specialist probes evaluate many formulas in Python. On representative univariate cases this adds about 0.8-1.6 seconds per formula and thousands of formula evaluations.

The first safe migration target was candidate scoring, because it is isolated, deterministic, and embarrassingly parallel across formulas.

## Plan

The migration plan was:

1. Reuse the existing C++ extension instead of creating a new backend.
2. Add a batch formula scoring API exposed through pybind11.
3. Score formulas in parallel with OpenMP.
4. Match Python display-formula semantics so scoring decisions do not change unexpectedly.
5. Wire Python `_refine_candidate_formulas()` to use the C++ scorer when available.
6. Keep Python fallback for unsupported formulas or missing C++ builds.
7. Leave residual fitting for the next step, because that requires replacing recursive Python fitting with a bounded residual mini-search.

## Implementation Logic

The new C++ scorer accepts:

- Formula strings.
- `X_fit`, `y_fit`.
- `X_val`, `y_val`.
- Optional OpenMP thread count.

For each formula it:

1. Parses the formula in C++.
2. Evaluates predictions on fit and validation data.
3. Solves the same affine calibration used by Python: `scale * prediction + bias`.
4. Computes fit MSE, validation MSE, and validation R2.
5. Returns score dictionaries to Python.

The scorer deliberately uses an exact parse-tree evaluator rather than the evolution graph evaluator. The evolution graph uses soft arithmetic nodes for search, which changed numeric semantics for expressions like `sin(3*x0)`. Candidate scoring needs to match Python/display semantics, so an exact evaluator was added inside `core.cpp`.

## Files Changed

### `glassbox/sr/cpp/core.cpp`

Added:

- `score_formula_candidates_cpp(...)` pybind function.
- Exact parse-tree formula evaluator for candidate scoring.
- OpenMP parallel loop over candidate formulas.
- Pybind export as `_core.score_formula_candidates(...)`.

### `glassbox/sr/sklearn_wrapper.py`

Updated `_refine_candidate_formulas()` to:

- Build a de-duplicated formula list.
- Call `_core.score_formula_candidates(...)` when the C++ extension is available.
- Convert C++ scores into the existing Python scoring dictionary format.
- Fall back to `_score_formula_candidate()` per formula when C++ scoring is unavailable or fails.
- Preserve existing constant refinement, risk scoring, generalization gap scoring, sorting, and pruning behavior.

### `tests/test_specialist_phase_eval.py`

Added test coverage for:

- C++ batch candidate scoring.
- Affine recovery for `2*sin(3*x) + 0.5` from the base formula `sin(3*x0)`.

## Verification

Commands run:

```bash
python setup.py build_ext --inplace
python -m pytest tests/test_specialist_phase_eval.py tests/test_specialist_state.py glassbox/sr/test_cpp_parity.py
```

Result:

```text
35 passed
```

Smoke check on `exp(-x^2)*sin(3*x)` showed univariate candidate building still surfaces the intended candidate:

```text
exp(-x0^(2))*sin((3)*x0), validation_r2=1.0
```

## Current Impact

This change moves the main refinement scoring loop into C++ and allows multithreaded formula scoring.

It does not fully solve the 2000+ second benchmark runtime by itself, because the biggest bottleneck remains recursive residual fitting. The next high-impact step is to replace `_stage_residual_symbolic_fit()` with a bounded C++ residual mini-search that does not call nested `GlassboxRegressor.fit()`.

## Next Step

Implement a C++ residual specialist with strict limits:

- No recursive Python estimator.
- Small formula grammar.
- Candidate seeds plus simple residual transforms.
- Hard timeout.
- OpenMP scoring/search.
- Return one residual formula or `None`.

That should address the largest runtime multiplier while keeping the specialist architecture intact.
