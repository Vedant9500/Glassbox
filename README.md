# Glassbox

Glassbox is an experimental symbolic regression project. It tries to recover
readable formulas from data by combining:

- classifier-guided and exact-match fast paths,
- a universal neural proposer that emits candidate formula skeletons,
- C++ island evolution for structural search and constant fitting,
- blackbox feature reduction and interaction discovery for tabular problems,
- specialist composition, residual, vault, and inception passes for hard cases,
- formula post-processing and displayed-formula scoring guards.

The current repository is no longer an ONN-only prototype. The ONN code is still
present and used as a research/legacy path, but the default sklearn-style runtime
is the hybrid pipeline in `glassbox/sr/sklearn_wrapper.py`.

## Current Runtime Pipeline

For a normal `GlassboxRegressor.fit(X, y)` call:

1. Input validation and optional blackbox preprocessing run in
   `glassbox/sr/blackbox_preprocessor.py`.
2. The classifier fast path in `scripts/classifier_fast_path.py` builds basis
   terms, runs exact-match/regression checks, and emits FPIP v2 diagnostics.
3. The universal proposer in `glassbox/universal_proposer/universal_proposer.py`
   can add grammar-constrained skeletons, operator priors, uncertainty, and a
   search plan.
4. Specialist screening/composition in `glassbox/sr/specialist_state.py` can
   build local candidates, composed seeds, residual candidates, vault reuse, and
   inception candidates.
5. Guided evolution calls the native `_core.run_evolution` bridge in
   `glassbox/sr/cpp/core.cpp`, seeded by fast-path/proposer/specialist formulas
   when available.
6. Final cleanup uses native C++ simplification when possible and guarded
   Python/SymPy helpers as fallback. Final benchmark scoring uses the displayed
   formula, not raw engine fitness.

## Installation

```bash
git clone https://github.com/Vedant9500/Glassbox.git
cd Glassbox

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The Python dependencies are listed in `requirements.txt`. The C++ extension uses
pybind11 and Eigen. A built Windows extension may already be present in a local
workspace, but source installs should rebuild it:

```bash
cd glassbox/sr/cpp
python setup.py build_ext --inplace
cd ../../..
```

Optional GPU support is through PyTorch/CUDA. Fast-path exact matching can use a
torch backend, while most native evolution still runs through C++/OpenMP CPU
code.

## Quick Start

Use the sklearn-compatible estimator:

```python
import numpy as np
from glassbox.sr.sklearn_wrapper import GlassboxRegressor

X = np.linspace(-3, 3, 300).reshape(-1, 1)
y = X[:, 0] ** 2 + np.sin(X[:, 0])

est = GlassboxRegressor(timeout=60, random_state=42)
est.fit(X, y)

print(est.get_formula())
print(est.best_mse_)
```

Run the local benchmark suite:

```bash
python scripts/benchmark_suite.py --tier 1 --device cpu --quiet
python scripts/benchmark_suite.py --tier 6 --with-evolution --device cpu
```

Run the SRBench-style local harness:

```bash
python scripts/run_srbench_local.py --track 2 --max-datasets 5 --no-hard-timeout
```

Run the interactive/single-formula tester:

```bash
python scripts/sr_tester.py --mode single --formula "x^2 + sin(x)"
```

## Important CLIs

- `scripts/benchmark_suite.py`: 8-tier synthetic benchmark. Current defaults are
  `models/curve_classifier_multi.pt` and `models/universal_proposer_multi.pt`.
  Supports `--with-evolution`, `--evolution-only`, `--specialist-regressor`,
  `--specialist-full`, `--exact-match-backend`, repeated `--tier`, and report
  output under `results/`.
- `scripts/run_srbench_local.py`: local SRBench-style Track 1/Track 2 harness
  with multi-seed reporting, hard-timeout support, adaptive timeouts,
  specialist flags, blackbox feature limits, and optional official dataset
  discovery.
- `scripts/benchmark_feynman_easy.py`: AI-Feynman easy benchmark runner.
- `scripts/classifier_fast_path.py`: fast-path basis construction, exact match,
  guided evolution handoff, and C++ beam/island evolution wrapper.
- `scripts/simplify_formula.py`: guarded Python/SymPy simplification fallback.
- `scripts/specialist_phase_eval.py`: phase harnesses for specialist composition
  and residual/inception features.
- `scripts/train_universal_proposer.py`: proposer training and replay-data path.
- `glassbox/curve_classifier/train_curve_classifier.py`: classifier training.

## Project Map

See `docs/PROJECT_MAP.md` for a detailed repository map. The short version:

```text
glassbox/
  curve_classifier/        classifier models, feature extraction, integration
  evolution/               Python ONN/evolution research path
  sr/
    core/                  OperationDAG and OperationNode
    operations/            meta operations
    optimizers/            BFGS and hybrid optimizers
    cpp/                   pybind11 bridge, C++ evolution, parser/simplifier
    blackbox_preprocessor.py
    fpip_v2.py
    sklearn_wrapper.py     main sklearn estimator and orchestration layer
    specialist_state.py
  universal_proposer/      neural proposer and FPIP v2 adapter
scripts/                   benchmarks, training, simplification, SRBench
tests/                     unit and regression tests for the active pipeline
docs/                      source documentation and project map
research_notes/            forward-looking research/audit notes
results/                   generated benchmark reports
models/                    local model checkpoints
```

## C++ Backend

The native extension is exposed as `_core` from `glassbox/sr/cpp/core.cpp`.
Current exported functions include:

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

The Python code should continue to treat C++ as an optional acceleration path and
fall back cleanly when `_core` is unavailable.

## Scoring Contract

Benchmarks must score the displayed formula whenever it can be evaluated.
Engine-internal MSE is diagnostic only. Reports should keep both:

- `mse` / `mse_display`: displayed-formula score used for status labels.
- `mse_raw`: native or fast-path internal score used to detect drift.

This contract is covered by tests in `tests/test_benchmark_scoring_contract.py`,
`tests/test_benchmark_common.py`, and related SRBench tests.

## Testing

```bash
pytest tests -q
pytest tests/test_benchmark_scoring_contract.py -q
pytest tests/test_sklearn_wrapper_cv_guard.py -q
```

Use focused benchmark smoke checks before changing defaults:

```bash
python scripts/benchmark_suite.py --tier 2 --tier 3 --device cpu --quiet
python scripts/run_srbench_local.py --track 2 --max-datasets 3 --no-hard-timeout
```

## Documentation

- `docs/PROJECT_MAP.md`: current codebase map and pipeline trace.
- `docs/ONN_Architecture.md`: ONN and active hybrid architecture notes.
- `docs/onn_runbook.md`: operating and release-gate runbook.
- `docs/Research_Roadmap.md`: current roadmap and status.
- `docs/CPP_Migration_Roadmap.md`: native backend migration status.
- `research_notes/`: research/audit notes that may propose features not yet
  implemented.

## Citation

```bibtex
@software{glassbox2026,
  title={Glassbox: Symbolic Regression with Hybrid Fast Path and C++ Evolution},
  year={2026},
  url={https://github.com/Vedant9500/Glassbox}
}
```
