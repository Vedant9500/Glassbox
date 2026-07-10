# Glassbox Architecture

Last updated: 2026-06-03.

Glassbox started as an Operation Neural Network (ONN) symbolic-regression
prototype. The current codebase is a hybrid symbolic regression system. ONN
components still exist under `glassbox/sr/core/` and `glassbox/evolution/`, but
the active default pipeline is the sklearn-compatible estimator in
`glassbox/sr/sklearn_wrapper.py`.

## Current High-Level Architecture

```text
X, y
  |
  v
GlassboxRegressor.fit()
  |
  +--> optional blackbox preprocessing
  |      feature ranking, feature reduction, interaction hints, seed formulas
  |
  +--> classifier fast path
  |      operator prediction, basis construction, exact match, sparse/linear fit
  |
  +--> universal proposer
  |      grammar skeletons, operator priors, uncertainty, FPIP v2 search plan
  |
  +--> specialist layer
  |      candidate pools, compositions, residual candidates, vault/inception reuse
  |
  +--> guided C++ evolution
  |      seed graphs, island search, native evaluation/refinement
  |
  +--> guarded cleanup and displayed-formula scoring
         formula_, best_mse_, diagnostics
```

## Main Components

| Component | File | Role |
|---|---|---|
| Estimator/orchestrator | `glassbox/sr/sklearn_wrapper.py` | Main runtime path and sklearn API. |
| Fast path | `scripts/classifier_fast_path.py` | Basis generation, exact match, regression, FPIP v2 handoff, guided evolution wrapper. |
| Curve classifier | `glassbox/curve_classifier/curve_classifier_integration.py` | PyTorch model loading and operator prediction. |
| Feature extraction | `glassbox/curve_classifier/generate_curve_data.py` | FFT, derivative, curvature, invariant, PCFG, and synthetic-data helpers. |
| Universal proposer | `glassbox/universal_proposer/universal_proposer.py` | GLU proposer, grammar-constrained skeletons, uncertainty, search planning, FPIP v2 adapter. |
| FPIP v2 | `glassbox/sr/fpip_v2.py` | Stable payload schema for fast-path/proposer-to-evolution handoff. |
| Blackbox preprocessing | `glassbox/sr/blackbox_preprocessor.py` | Feature ranking, interaction discovery, reduced-space remapping, seed formulas. |
| Specialist state | `glassbox/sr/specialist_state.py` | Local candidate screening, composition, hot-spot segments, vault memory. |
| Native backend | `glassbox/sr/cpp/core.cpp` | pybind11 bridge to C++ evolution, simplification, scoring, refinement. |
| ONN DAG | `glassbox/sr/core/operation_dag.py` | Legacy/research operation DAG model. |
| ONN node | `glassbox/sr/core/operation_node.py` | Operation-node routing and selection. |
| Meta ops | `glassbox/sr/operations/meta_ops.py` | Power, periodic, arithmetic, exp/log-style operation helpers and formula normalization. |
| Python evolution | `glassbox/evolution/evolution.py` | Legacy/research evolutionary trainer and fallback support. |

## Active Pipeline Details

### 1. Blackbox Preprocessing

`blackbox_mode="auto"` lets the estimator detect higher-dimensional or tabular
problems and prepare a reduced search state. The preprocessor computes multiple
feature rankers, disagreement diagnostics, interaction candidates, remapping
helpers, and seed formulas. The reduced-space formula can be mapped back to
original variables before final selection.

### 2. Classifier Fast Path

The fast path predicts likely operator families, builds candidate bases, performs
exact-match and sparse/linear fits, evaluates residual structure, and emits an
FPIP v2 payload. Exact-match search can use NumPy or torch-backed CPU/CUDA
selection depending on `exact_match_backend`, work thresholds, and device
availability.

### 3. Universal Proposer

The proposer is currently a GLU model over extracted analytical features rather
than the older roadmap's aspirational Set Transformer. It decodes candidates
through constrained univariate/multivariate grammars, produces operator priors
and uncertainty, and builds search-plan knobs such as beam count, generation
multiplier, power bounds, and seed budget.

Default model path in current CLIs: `models/universal_proposer_multi.pt`.

### 4. Specialist Layer

The specialist layer augments hard cases with local candidate pools and composed
formulas. It can run:

- diagnostics and pair scoring,
- safe composition screening,
- composed seed injection,
- residual symbolic stages,
- vault memory,
- inception/subexpression reuse.

The synthetic benchmark exposes this through `--specialist-regressor` and
`--specialist-full`; the sklearn estimator exposes constructor flags.

### 5. Guided C++ Evolution

Guided evolution builds seed graphs from candidate formulas and signal-derived
templates, then calls `_core.run_evolution`. The native bridge supports island
search, migration, seed graphs, OpenMP-style thread use, native candidate
scoring, simplification, and refinement helpers.

The benchmark contract scores the displayed formula after postprocessing.
`mse_raw` remains diagnostic.

## ONN Research/Legacy Model

The ONN DAG remains useful for experimentation and fallback paths:

```text
Input features
  -> OperationDAG layers
  -> OperationNode routing
  -> selected unary/binary/meta operations
  -> output projection
  -> extracted formula
```

### ONN Files

- `glassbox/sr/core/operation_dag.py`: `OperationDAG`.
- `glassbox/sr/core/operation_node.py`: `OperationNode` and routing.
- `glassbox/sr/operations/meta_ops.py`: meta operation implementations.
- `glassbox/sr/hard_concrete.py`: hard-concrete selection helpers.
- `glassbox/evolution/evolution.py`: evolutionary ONN trainer, mutation,
  refinement, pruning, and formula finalization.

### ONN Concept

Each node routes one or more sources, selects an operation, applies it, and
passes the result forward. Operations include power/periodic/exponential/log and
binary arithmetic-style combinations. Evolution explores discrete structure while
gradient-based refinement tunes continuous parameters and coefficients.

The ONN path is not the only runtime anymore. Docs, examples, and experiments
should avoid claiming that all Glassbox runs train an ONN.

## C++ Backend Surface

The pybind11 extension currently exports:

- `score_formula_candidates`
- `run_evolution`
- `refine_frequencies`
- `refine_powers`
- `refine_periodic_rational`
- `iterative_elastic_net`
- `lasso_coordinate_descent`
- `simplify_formula_cpp`
- `formula_to_seed_graph_cpp`
- `snap_formula_floats_cpp`
- `reduce_formula_noise_cpp`

Python callers should keep fallback paths because `_core` may be unavailable on
fresh installs until the extension is built.

## Scoring and Safety Rules

- Score displayed/evaluable formulas for benchmark labels.
- Keep raw native/fast-path MSE as a drift diagnostic.
- Guard postprocessing with fidelity checks before accepting simplified output.
- Do not let a fast-path exact label bypass displayed-formula evaluation.
- Prefer validation/holdout checks for blackbox and specialist candidate
  selection when available.

## Current Limitations

- Hard nested/compositional expressions remain difficult.
- Multivariate blackbox discovery still relies on heuristic feature ranking,
  interaction probes, and bounded templates.
- The proposer is useful as a seed/planner, but it is not yet a full formula
  foundation model.
- Python and C++ paths must stay synchronized; many tests cover bridge behavior
  because formula display, raw fitness, and native graph semantics can drift.
- SymPy remains a fallback dependency while native simplification continues to
  mature.

## Minimal Usage Example

```python
import numpy as np
from glassbox.sr.sklearn_wrapper import GlassboxRegressor

X = np.linspace(-3, 3, 300).reshape(-1, 1)
y = X[:, 0] ** 2 + np.sin(X[:, 0])

est = GlassboxRegressor(timeout=60, random_state=42)
est.fit(X, y)
print(est.get_formula())
```

For ONN-specific experiments:

```python
import torch
from glassbox.sr.core.operation_dag import OperationDAG
from glassbox.evolution import train_onn_evolutionary

x = torch.linspace(-3, 3, 300).reshape(-1, 1)
y = x ** 2

def make_model():
    return OperationDAG(n_inputs=1, n_hidden_layers=1, nodes_per_layer=4)

result = train_onn_evolutionary(make_model, x, y, population_size=15, generations=30)
print(result["formula"])
```
