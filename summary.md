# Glassbox Symbolic Regression — Complete Algorithm & Research Summary

> **Purpose**: Deep technical reference for understanding the full Glassbox SR pipeline,
> its optimizations, observed limitations, and research directions for future improvement.
>
> **Last updated**: 2026-04-26

---

## Table of Contents

1. [System Architecture (High-Level)](#1-system-architecture)
2. [Stage 0 — Data Generation & Classifier Training](#2-stage-0)
3. [Stage 1 — Inference: Curve Classifier Integration](#3-stage-1)
4. [Stage 2 — Fast-Path Direct Regression](#4-stage-2)
5. [Stage 3 — Continuous Refinement Sub-Stages](#5-stage-3)
6. [Stage 4 — Beam Search Evolution (Guided)](#6-stage-4)
7. [Neural-Symbolic Optimization Suite](#7-optimization-suite)
8. [Stage 5 — Raw C++ Evolutionary Search](#8-stage-5)
9. [Stage 6 — Formula Simplification & Noise Reduction](#9-stage-6)
10. [Benchmark Suite & Scoring](#10-benchmark)
11. [SRBench Integration](#11-srbench)
12. [Known Limitations & Failure Patterns](#12-limitations)
13. [Research Directions](#13-research)
14. [Optimization Summary](#14-optimization)

---

## 1. System Architecture (High-Level) <a id="1-system-architecture"></a>

Glassbox is a **hybrid symbolic regression** system that combines a trained neural
classifier with algebraic solvers and evolutionary search. The pipeline flows through
up to 6 stages, with early-exit shortcuts when high-quality solutions are found:

```
Input (x, y) data
       │
       ▼
┌─────────────────────────────────────┐
│  Stage 0: Offline Classifier Train  │  (one-time, offline)
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Stage 1: Classifier Inference      │  predict operator probabilities
│  (curve_classifier_integration.py)  │  → {sin: 0.92, power: 0.85, ...}
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Stage 2: Fast-Path Regression      │  exact-match search → LASSO → OLS refit
│  (classifier_fast_path.py)          │  → instant formula if basis covers target
└──────────┬──────────────────────────┘
           │  if MSE > threshold
           ▼
┌─────────────────────────────────────┐
│  Stage 3: Continuous Refinement     │  gradient-based ω, power, periodic×rational
└──────────┬──────────────────────────┘
           │  if still insufficient
           ▼
┌─────────────────────────────────────┐
│  Stage 4: Beam Search Evolution     │  20 beams × 3 rounds of C++ evolution
│  (guided by classifier hints)       │  tournament selection, config mutation
└──────────┬──────────────────────────┘
           │  if still insufficient
           ▼
┌─────────────────────────────────────┐
│  Stage 5: Raw C++ Evolution         │  full NSGA-II / AFPO search
│  (glassbox/sr/cpp/evolution.h)      │  multi-start runs, macro-mutations
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  Stage 6: Simplification & Snap    │  BIC noise reduction → SymPy simplify
│  (simplify_formula.py)              │  → float snap → final formula
└─────────────────────────────────────┘
```

**Key design philosophy**: The classifier acts as a **search space pruner** — it
narrows the infinite space of possible formulas to a manageable basis set, enabling
the fast-path to solve most well-characterized curves in under 1 second. Evolution
is reserved as a fallback for curves the classifier cannot characterize.

---

## 2. Stage 0 — Data Generation & Classifier Training <a id="2-stage-0"></a>

**Scripts**: `scripts/generate_curve_data.py`, `scripts/train_curve_classifier.py`

### 2.1 Synthetic Data Generation

1. **Template sampling**: Generate formulas from families — polynomial, periodic,
   exponential, logarithmic, rational, and nested compositions.
2. **Coefficient randomization**: Randomize scales, offsets, frequencies, and exponents.
3. **Curve evaluation**: Evaluate each formula over sampled x-domains; reject
   NaN/Inf/divergent outputs.
4. **Noise augmentation**: Optionally add Gaussian noise for robustness.
5. **Feature extraction**: Extract a **366-dimensional feature vector** per curve:
   - Raw curve slice (128 uniform samples)
   - FFT magnitude spectrum (64 bins)
   - First and second derivatives (64 bins)
   - Summary statistics (mean, std, skew, kurtosis, min, max, range)
   - Curvature features
6. **Output**: Save `(features, labels)` pairs to `.npz` datasets.

### 2.2 Label Space (Multi-Hot)

Each curve is tagged with **multi-hot labels** indicating operator families:

| Label          | Meaning                          |
|----------------|----------------------------------|
| `identity`     | Linear/constant component        |
| `sin`          | Sine function present            |
| `cos`          | Cosine function present          |
| `power`        | Polynomial/power terms           |
| `exp`          | Exponential function             |
| `log`          | Logarithmic function             |
| `addition`     | Additive composition             |
| `multiplication` | Multiplicative composition     |
| `rational`     | Rational function (division)     |

Optional constant-indicator classes (`const_pi`, `const_e`) may also be included.

### 2.3 Classifier Architecture & Training

- **Architecture**: MLP or 1D-CNN classifier (configurable).
- **Loss**: `BCEWithLogitsLoss` (multi-label binary cross-entropy).
- **Training details**:
  - Train/val/test split with feature normalization from train split only.
  - Gradient clipping for stability.
  - Optional **isotonic calibration** via `calibrate_classifier.py`:
    temperature scaling + per-class threshold optimization.
  - Checkpoints saved to `models/curve_classifier_v3.1.pt`.

---

## 3. Stage 1 — Inference: Curve Classifier Integration <a id="3-stage-1"></a>

**Script**: `scripts/curve_classifier_integration.py`

### 3.1 Single-Input Prediction Flow

```
Input: x (N,) or (N,1), y (N,)
  │
  ├─ Extract 366-d features from y
  ├─ Apply saved scaler (if present)
  ├─ Run classifier: logits → sigmoid (with temperature)
  ├─ Apply per-class thresholds
  ├─ Build operator probability dict
  └─ Add derived keys:
       periodic   = max(sin, cos)
       exponential = max(exp, log)
       polynomial  = max(power, identity)
```

### 3.2 Multi-Input Prediction Flow (n_vars > 1)

For multivariate problems, the classifier cannot directly process ND data. Instead:

1. Build an ND interpolator for `y(x)` (LinearNDInterpolator → NearestNDInterpolator fallback).
2. For each variable `v`: fix all others at median, sweep `v` across its range,
   generate a 1D slice, extract features, and predict.
3. **Aggregate** across variables by taking the **max probability per class**.
4. Threshold and return final operator dictionary.

### 3.3 Uncertainty Metrics (FPIP)

The **Fast Path Intelligence Packet (FPIP)** computes confidence diagnostics:

- **`prediction_entropy`**: Normalized Shannon entropy of class probabilities
  (0 = certain, 1 = maximally uncertain).
- **`prediction_margin`**: Gap between top-1 and top-2 class probabilities.
- **`prediction_uncertain`**: Boolean flag — `True` if entropy > 0.8 OR margin < 0.1.

These metrics are used downstream for:
- **Uncertainty-coupled budget routing** (Stage 5 compute allocation)
- **Prior trust blending** (Stage 4 beam initialization)

---

## 4. Stage 2 — Fast-Path Direct Regression <a id="4-stage-2"></a>

**Script**: `scripts/classifier_fast_path.py` → `fast_path_regression()`

This is the **primary solver** — it converts the classifier's operator predictions
into a concrete formula using linear algebra alone (no evolution). Most Tier 1–3
problems are solved here in **< 1 second**.

### 4.1 Pre-Checks

- **Constant signal detection**: If `std(y) < 1e-10`, return `y ≈ mean(y)` immediately.
- **Easy multivariate templates**: For `n_vars ≥ 2`, try exact matches against
  physics templates (Euclidean distance, relativistic mass, cosine envelope) before
  building the full basis.

### 4.2 Basis Construction (`build_basis_from_predictions`)

The classifier probabilities control which basis families are included:

| Family              | Basis Functions Generated                                          |
|---------------------|--------------------------------------------------------------------|
| **Polynomial**      | `1, x, x², x³, ..., x⁶` + fractional powers `x^0.5, x^1.5, ...` |
| **Periodic**        | `sin(ωx), cos(ωx)` for ω ∈ {1, 2, 0.5, π, 2π, FFT-detected}     |
| **Exponential**     | `exp(x), exp(-x), exp(-x²), 1/(exp(x)-1), x/(exp(x)-1), ...`     |
| **Logarithmic**     | `log(|x|+1), log(x²+1)`                                          |
| **Compositions**    | `sin(x²), cos(x²), sin(1/x), cos(1/x)`                           |
| **Rational**        | `1/(x²+c), x/(x²+c), 1/(|x|+c)` for c ∈ {0.5, 1, 2}            |
| **Pairwise** (ND)   | `xᵢ·xⱼ, xᵢ/xⱼ, xᵢ·xⱼ/xₖ², √xᵢ/xⱼ, ...`                      |
| **Damped periodic** | `e^(-αx)·sin(ωx), e^(-αx)·cos(ωx)` for α ∈ {0.2, 0.5}           |
| **Cross terms**     | `x·sin(x), x·cos(x), sin(ωx)/(x²+c), cos(ωx)/(x²+c)`           |

**Universal basis mode** (`universal_basis=True`): Always includes polynomial +
periodic + composition terms regardless of classifier confidence. This is the
default and is critical for catching formulas the classifier partially misses.

**Numerical safety**: All basis columns are clamped to `[-1e6, 1e6]` and
NaN/Inf values are replaced with 0.

### 4.3 Out-of-Domain (OOD) Holdout

Before fitting, **10% of the x-domain edges** (5% lowest, 5% highest) are held
out. The fit is performed on the remaining 90%. This enables:

- **Holdout MSE reporting** for generalization assessment
- **OOD penalty** in candidate scoring: if holdout MSE / in-sample MSE > 5.0,
  the candidate's score is penalized, filtering overfitted polynomials.

### 4.4 Exact Symbolic Match Search

**Before** LASSO, a combinatorial search tries to find an exact solution:

1. **Singles**: Try each basis function alone (with/without constant).
2. **Pairs**: Try all C(n,2) pairs of basis functions.
3. **Triples**: Try all C(n,3) triples.
4. **Threading**: Pairs/triples can be parallelized via `ThreadPoolExecutor`.

Acceptance: MSE < 1e-5. If found, **return immediately** — no LASSO needed.

For pure polynomial signals, the search extends to **6-term** combinations.
Gated by `exact_match_max_basis=150` to prevent combinatorial explosion.

### 4.5 LASSO Coordinate Descent

If no exact match, run L1-regularized regression:

```
Objective: min_w ||y - Basis·w||² + α·||w||₁
```

- **Custom implementation** (`lasso_coordinate_descent`): Optimized with
  incremental residual updates — O(n·m) per iteration instead of O(n·m²).
- **Alpha sweep**: Try α ∈ {0, 0.001, 0.01, 0.05, 0.1, 0.2} and keep the
  best by **complexity-penalized score** = MSE + 0.001 × n_terms.
- **Candidate pool**: All alpha results are stored with deduplication by
  active-term signature.

### 4.6 OLS Refit

Each LASSO candidate is **refitted with OLS** on its selected terms only.
This recovers exact coefficients while preserving the sparse structure
selected by LASSO. Critical for converting `coeff=0.9975` → exact `1.0`.

### 4.7 Candidate Selection

Candidates are ranked by: `score = MSE + 0.001 × n_terms + OOD_penalty`.
Top-5 candidates are retained for:
- Reporting the best formula
- Seeding Stage 4 beam search (targeted initialization)

---

## 5. Stage 3 — Continuous Refinement Sub-Stages <a id="5-stage-3"></a>

**Script**: `classifier_fast_path.py` → `fast_path_with_refinement()`

If the fast-path MSE is moderate (not exact), several gradient-based
refinement passes are attempted:

### 5.1 Frequency Refinement (`refine_frequencies`)

- **Trigger**: Periodic signal detected AND 1e-4 ≤ MSE ≤ 0.2.
- **Method**: Adam optimizer over a `FrequencyModel`:
  `y ≈ c₀ + c₁x + c₂x² + Σ[aᵢ·sin(ωᵢx) + bᵢ·cos(ωᵢx)]`
  with learnable ω parameters.
- **Steps**: 150 gradient steps, starting from FFT-detected frequencies.
- **Purpose**: FFT grid resolution limits precision (e.g., detects ω=3.13
  when true ω=3.2). Gradient descent refines to the exact value.

### 5.2 Periodic×Rational Refinement (`refine_periodic_rational`)

- **Trigger**: Periodic + power signals AND MSE > 1e-4.
- **Model**: `y ≈ a·sin(ωx)/(x²+c) + b·cos(ωx)/(x²+c) + dx + e`
  with learnable ω and c (softplus-constrained positive).
- **Grid search**: Tries multiple (ω₀, c₀) initializations.
- **Steps**: 400 Adam steps per initialization.
- **Purpose**: Captures Lorentzian-modulated periodic signals that the
  fixed-basis LASSO cannot represent.

### 5.3 Power Exponent Refinement (`refine_powers`)

- **Trigger**: Power signal detected AND (5e-5 ≤ MSE ≤ 0.2 OR n_terms > 6).
- **Model**: `y ≈ Σ aᵢ·sign(x)·|x|^pᵢ + periodic_terms + c₀ + c₁x`
  with learnable continuous powers pᵢ.
- **Two-stage process**:
  1. Fit powers from initial guesses [0.5, 1.5, 2.5, 3.5].
  2. Check residuals via FFT for hidden frequencies; refit if found.
- **Power snapping**: Final powers are snapped to clean values
  (integers, halves, thirds) if within tolerance 0.08.
- **Purpose**: Handles non-integer exponents like x^2.3 that the
  integer-power basis misses.

### 5.4 Acceptance Logic

New candidates are accepted only if they improve the
**complexity-penalized score** AND don't cause major MSE regressions
on already-good fits (5× guard for MSE < 1e-5).

---

## 6. Stage 4 — Beam Search Evolution (Guided) <a id="6-stage-4"></a>

**Script**: `classifier_fast_path.py` → `beam_search_evolution()`

If Stages 2–3 fail to produce a satisfactory formula, the system launches
a **structured evolutionary search** using **Operation-Based Neural Networks (ONN)**.

### 6.1 Neural-Symbolic Representation (ONN)

The "Glassbox ONN" (implemented in `OperationDAG`) represents formulas as a differentiable compute graph:
- **Hard-Concrete Selectors**: Uses a specialized Gumbel-Softmax variant to make the choice of operations (e.g., sin vs. square) differentiable during exploration.
- **Meta-Operations**: Operators like `MetaPower` and `MetaPeriodic` allow the network to learn continuous parameters (frequency, exponent) *while* evolving the structural type.
- **Lamarckian Inheritance**: Children inherit the *optimized* constants of their parents, significantly accelerating convergence compared to raw structural evolution.

### 6.2 Operator Hint Extraction

From the fast-path candidates, extract:
- **Operators**: Which families were active (sin, cos, power, exp, log)
- **Frequencies**: ω values from active periodic terms
- **Powers**: Exponent values from active power terms
- **Flags**: `has_rational`, `has_exp_decay`
- **Active terms**: Basis function names from top candidates

### 6.2 Targeted Population Initialization (Elite Seeding)

Top-3 fast-path candidates are injected as **elite seeds**:
- Extract operator priors and frequencies from each candidate
- Use these to bias the C++ evolution's initial population
- Warm-starts the search from the fast-path's best structural guesses

### 6.3 Beam Configuration

Each "beam" is a C++ evolution configuration with varied hyperparameters:

| Parameter         | Variation Strategy                                  |
|-------------------|-----------------------------------------------------|
| `op_priors`       | Blended classifier priors × uniform (trust-weighted)|
| `seed_omegas`     | FFT-detected ω values (varied per beam)             |
| `pop_size`        | 30–80 (varied ±20 per mutation)                     |
| `generations`     | 300–800 (varied ±100 per mutation)                  |
| `p_min / p_max`   | Adaptive power bounds (polynomial mode: up to 8.0)  |
| `arithmetic_temp` | 2.0–8.0 (controls arithmetic operator selection)    |

**Prior trust blending**: Classifier priors are blended with uniform priors
using a trust score derived from entropy/margin metrics:
`blended = trust × classifier_prior + (1-trust) × uniform`

### 6.4 Polynomial Mode Detection

A quick polynomial fit (degrees 1–8) determines if the target is
polynomial-like. If so:
- Power bounds are expanded (p_max up to 8.0)
- Beams are biased toward power operations
- Periodic/exp beams are reduced

### 6.5 Tournament Loop

```
Round 1: Generate 20 diverse beam configs
         Run each via C++ evolution (sequential — C++ uses OpenMP internally)
         Sort by MSE

Round 2: Keep top 20% (4 winners)
         Mutate each winner into 4 variants
         Fill remaining slots randomly
         Run all 20 beams

Round 3: Same as Round 2

Return: Overall best formula across all rounds
```

**Early exit**: If MSE < 1e-10 at any point, stop immediately.

---

## 7. Neural-Symbolic Optimization Suite <a id="7-optimization-suite"></a>

Beyond the linear pipeline, Glassbox utilizes a suite of researcher-grade optimization techniques to handle complex landscapes.

### 7.1 Risk-Seeking Policy Gradient (RSPG)
- **Concept**: Instead of minimizing the mean MSE (which can be biased by outliers), RSPG minimizes the **k-th percentile loss** (typically top 10%).
- **Purpose**: Escapes local minima by focusing the search on "lucky" high-quality discovery paths, allowing the model to ignore regions of the space that are currently non-functional.

### 7.2 FFT Frequency Seeding
- **Problem**: Evolutionary search often struggles to find high-frequency periodicities (aliasing).
- **Solution**: Runs a Fast Fourier Transform on input data to detect dominant frequencies ($\omega$), which are then used to seed the initial population's `MetaPeriodic` nodes.

### 7.3 Progressive Rounding & Entropy Annealing
- **Soft Rounding**: Nudges parameters (like exponents) toward integers using a differentiable "potential well" rather than a hard snap, preserving gradient flow during training.
- **Entropy Annealing**: Starts with high entropy (maximum exploration) and gradually strengthens penalties to force discrete "hard" decisions as the search matures.

---

## 8. Stage 5 — Raw C++ Evolutionary Search <a id="8-stage-5"></a>

**Core**: `glassbox/sr/cpp/evolution.h`, `glassbox/sr/cpp/eval.h`

The main symbolic regression workhorse, called both from beam search
(Stage 4) and directly from the sklearn wrapper as a final fallback.

### 8.1 Representation: IndividualGraph (DAG)

Expressions are represented as **Directed Acyclic Graphs** with typed nodes:
- **Input nodes**: x₀, x₁, ...
- **Constant nodes**: Learnable floating-point parameters
- **Unary nodes**: sin, cos, exp, log, abs, sqrt, power
- **Binary nodes**: +, -, ×, ÷ (protected division)

### 8.2 Selection: NSGA-II / Age-Fitness Pareto (AFPO)

Three-objective tournament selection:

| Objective      | Goal     | Purpose                                    |
|----------------|----------|--------------------------------------------|
| **MSE**        | Minimize | Fit quality                                |
| **Complexity** | Minimize | Number of graph nodes (Occam's razor)      |
| **Age**        | Minimize | Generations survived (protects young individuals) |

Young graphs are protected while their continuous parameters mature,
preventing premature elimination of promising but unoptimized structures.

### 8.3 Macro-Mutations (15% probability)

Beyond standard point mutations, structural transforms preserve
winning building blocks:

- **Wrap** (50%): `f(x) → sin(f(x))` or `exp(f(x))`
- **Multiply** (30%): Combine sub-graphs `f(x) × g(x)`
- **Nest** (20%): Submit one sub-graph into another `f(g(x))`

### 8.4 Continuous Parameter Optimization

No PyTorch autograd — all optimization is internal to C++:

1. **Local SGD**: Finite-difference Adam (`refine_inner_params`) tweaks
   frequencies, phases, and decay rates.
2. **Ridge Regression**: Evaluates linear combinations of graph terminal
   nodes to instantly solve output weights.
3. **In-Loop Snapping** (`cleanup_graph`): Actively snaps floating-point
   parameters to exact integers, fractions, or constants (π, e) during
   the run if MSE degradation is < 1.01×.

### 8.5 Multi-Start Runs (sklearn wrapper)

The `GlassboxRegressor.fit()` method runs C++ evolution with
**multi-start** (default 3 runs) with different random seeds.
Budget is split across runs, and the best result is kept.

### 8.6 Uncertainty-Coupled Budget Routing

The compute budget for evolution is **dynamically scaled** based on
FPIP uncertainty metrics:

| Condition                      | Budget Multiplier |
|--------------------------------|-------------------|
| High-confidence classifier hit | ×0.3 (save time)  |
| Moderate confidence            | ×0.9              |
| Low confidence / uncertain     | ×1.3 (more time)  |
| Exact fast-path (R² ≥ 0.995)  | ×0.2 (minimal)    |

Formula: `budget = base_timeout × difficulty_score × uncertainty_scale`

Clamped to `[min_compute_budget, max_compute_budget]` (default [10, 300]s).

---

## 9. Stage 6 — Formula Simplification & Noise Reduction <a id="9-stage-6"></a>

**Scripts**: `scripts/simplify_formula.py`, `sklearn_wrapper.py`

### 9.1 Symbolic Consolidation & Pruning

Before final output, the formula undergoes a **Post-Training Pruning Pipeline** (`pruning.py`):
- **Sensitivity Analysis**: Conducts ablation and gradient-based sensitivity checks to identify nodes that contribute minimally to the total variance.
- **Recursive Graph Pruning**: Depth-first traversal to remove "dead branches" and fold redundant constant operations.
- **Mask & Fine-tune**: Temporarily zeros out terms and performs a "recovery" fine-tuning; terms are only permanently deleted if MSE remains stable.

### 9.2 BIC-Based Noise Reduction (`_reduce_formula_noise`)

Before symbolic simplification, a **greedy backward elimination**
removes spurious terms introduced by L1 regularization:

1. Parse formula into individual terms via SymPy.
2. Evaluate each term as a column in a design matrix.
3. Iteratively drop the term whose removal improves BIC the most:
   `BIC = N·log(MSE) + k·log(N)`
4. Refit remaining terms with OLS.
5. Stop when no further BIC improvement is possible.

### 9.3 Float Snapping (`snap_formula_floats`)

- Snap coefficients near integers: `0.9975 → 1.0` (tol: 0.01)
- Snap near-zero terms to 0: `0.0001924 → 0` (tol: 1e-3)
- Snap to known constants: π, e, √2, etc.

### 9.4 SymPy Simplification (`simplify_onn_formula`)

- Multi-pass (up to 6 passes) simplification pipeline.
- Identity folding (sin²+cos²=1, etc.)
- `nsimplify` for recognizing rational multiples of π/e.
- **Complexity guard**: Formulas > 500 chars or > 24 terms use
  fast snap-only mode (SymPy can hang on huge expressions).

---

## 10. Benchmark Suite & Scoring <a id="10-benchmark"></a>

**Script**: `scripts/benchmark_suite.py`

### 10.1 Tier Structure (~200 formulas across 8 tiers)

| Tier | Name                    | Count | Difficulty Description                                  |
|------|-------------------------|-------|---------------------------------------------------------|
| 1    | **Trivial**             | 30    | Constants, linear, simple polynomials (x², x³, x⁴)     |
| 2    | **Simple Polynomial**   | 26    | Nguyen-1 through Nguyen-4, Chebyshev, factored cubics   |
| 3    | **Basic Transcendental**| 25    | Single sin/cos/exp/log, √x, simple scaled versions      |
| 4    | **Nguyen Suite**        | 28    | Standard SR benchmarks + trig identities, Keijzer        |
| 5    | **Sums & Products**     | 25    | sin(x)+x², exp(-x)+x, mixed sums and products           |
| 6    | **Rational & Nested**   | 26    | 1/(1+x²), sigmoid, sin(x²), exp(sin(x)), damped waves  |
| 7    | **Hard Compositions**   | 26    | Triple products, Gabor wavelets, Fourier 3-term sums    |
| 8    | **Frontier**            | 25    | sin(cos(x)), nested exp, modulated compositions         |

### 10.2 Scoring Criteria

| Score    | Symbol | Condition                         |
|----------|--------|-----------------------------------|
| **EXACT**  | ✅    | MSE < 1e-6 AND ≤ 5 terms         |
| **APPROX** | 🟡    | MSE < 0.01                        |
| **LOOSE**  | 🟠    | MSE < 0.1                         |
| **FAIL**   | ❌    | MSE ≥ 0.1 or error               |

**Weighted scoring**: EXACT = 3pts, APPROX = 2pts, LOOSE = 1pt, FAIL = 0pts.

### 10.3 MSE Pipeline

The benchmark uses **displayed-formula MSE** (not raw C++ MSE) for scoring:
1. Get raw MSE from C++ engine (`mse_raw`)
2. Re-evaluate the simplified/displayed formula against ground truth (`mse_display`)
3. Score based on `mse_display` only
4. Report divergence diagnostics if `|log₁₀(raw/display)| > 1.0`

### 10.4 Execution Modes

- **Fast-path only** (default): Classifier → fast-path regression
- **`--with-evolution`**: Fast-path first, then guided evolution if not exact
- **`--evolution-only`**: Skip fast-path, run guided beam search for everything
- **`--cpp-evolution-only`**: Skip classifier entirely, pure C++ evolution

---

## 11. SRBench Integration <a id="11-srbench"></a>

**Script**: `scripts/run_srbench_local.py`

### 11.1 Two-Track Protocol

**Track 1 — Black-Box Regression (PMLB)**:
- Real-world datasets from the Penn Machine Learning Benchmark suite
- Metric: R², MSE, model size
- Multi-seed protocol (default: 5 seeds: 42, 1337, 2027, 7, 11)
- Stability analysis: IQR, std, worst-decile R²

**Track 2 — Ground-Truth Symbolic Regression**:
- Known-formula problems with exact symbolic targets
- Additional metrics: exact recovery rate, time-to-first-exact,
  time-to-first-acceptable, failure taxonomy
- Exact match criterion: full-data MSE < 1e-6

### 11.2 Adaptive Timeout Scaling

`estimate_timeout_budget()` scales per-problem timeout based on:
- Number of features (more features → more time)
- Training set size
- Base timeout (default 60s for PMLB)

### 11.3 Failure Taxonomy

Non-exact results are classified into failure buckets:
- **structural_miss**: Wrong functional form entirely
- **coefficient_error**: Right structure, wrong constants
- **complexity_blow**: Formula too complex (bloated)
- **timeout**: Hit the time limit
- **numerical_instability**: NaN/Inf during evaluation

### 11.4 Post-Processing Options

- **`--post-simplify`**: Run `simplify_formula_with_guard()` on discovered formulas
- **`--skip-evolution-if-bloated`**: Skip C++ evolution if fast-path formula > 20 terms
- **`--hard-timeout`**: Process-level hard timeout enforcement via subprocess

---

### 12.1 ⚠️ Performance Cliff After Tier 4

The model performs well on Tiers 1–4 (trivial → Nguyen suite) but shows
**significant degradation starting at Tier 5**. The performance cliff
becomes severe at Tier 6+ (rational/nested compositions).

**Root cause analysis**:

| Factor                  | Impact                                                    |
|-------------------------|-----------------------------------------------------------|
| Basis coverage gap      | Nested compositions like `sin(cos(x))` or `exp(sin(x))`  |
|                         | are NOT in the basis set — they cannot be found by LASSO  |
| Classifier limitations  | The classifier identifies *individual* operators but not  |
|                         | their *composition structure* (sin+cos ≠ sin(cos))        |
| Search space explosion  | Evolution must discover nested structures from scratch,   |
|                         | with no warm-start from fast-path for these cases         |
| FFT blindness           | FFT detects periodic signals but cannot distinguish       |
|                         | `sin(x²)` from `sin(ωx)` — both appear periodic          |

### 12.2 ⚠️ The 30-Second Failure Rule

**Observed pattern**: If the system hasn't found the correct answer within
~30 seconds, there is a **near-100% probability it will fail** regardless
of additional compute time.

**Why this happens**:

1. **Fast-path succeeds fast or not at all**: The basis either contains
   the target or it doesn't. LASSO + exact search complete in < 2s.
2. **Beam search plateau**: After 3 rounds (≈15–20s), the beam tournament
   has explored its diversity budget. Further mutations produce diminishing
   returns because the search space has been locally exhausted.
3. **C++ evolution local optima**: The evolutionary search gets trapped
   in local optima (e.g., polynomial approximations of transcendental
   functions). The macro-mutation rate (15%) is insufficient to escape
   these basins for deeply nested targets.
4. **No restart mechanism**: The current pipeline does not implement
   restart-on-stagnation. Once the beam search completes, the system
   continues with raw C++ evolution from the same structural neighborhood.

### 12.3 ⚠️ Black-Box Dataset Struggles (SRBench Track 1)

The model struggles significantly with PMLB black-box datasets:

- **Feature engineering gap**: The classifier was trained on synthetic
  1D curves. Real-world multivariate data has complex feature interactions
  the per-variable-slice approach cannot capture.
- **Curse of dimensionality**: Basis expansion for n_vars > 3 produces
  thousands of terms (product/ratio combinations grow combinatorially).
  LASSO struggles with such wide, correlated design matrices.
- **Distribution mismatch**: PMLB datasets have noise, outliers, and
  non-uniform sampling that synthetic training data doesn't model.

### 12.4 Additional Limitations

**Formula bloat**:
- LASSO can produce formulas with 15+ terms when the true solution is
  compact. The BIC noise reduction helps but is not aggressive enough
  for heavily regularized solutions.

**Constant discovery**:
- The system cannot discover arbitrary constants (e.g., `2.71828*x²`
  without recognizing it as `e·x²`). The float-snapping pipeline only
  recognizes a fixed set of known constants.

**Compositional depth**:
- The basis is fundamentally **depth-1**: `f(g(x))` compositions are
  limited to pre-defined templates (sin(x²), exp(-x²), etc.).
  `f(g(h(x)))` triple-nesting is not representable in the basis at all.

**Multi-input limitations**:
- The per-variable-slice approach for n_vars > 1 loses interaction
  information. `x₁·sin(x₂)` cannot be detected by analyzing x₁ and x₂
  slices independently.

**Simplification fragility**:
- SymPy simplification can sometimes *change* the mathematical meaning
  of a formula (e.g., domain issues with log, sqrt of negative values).
- Very long formulas (> 500 chars) bypass symbolic simplification entirely.

---

## 13. Research Directions <a id="13-research"></a>

### 13.1 Near-Term Improvements (High Impact)

**Restart-on-Stagnation Strategy**:
- If discovery hasn't converged within 15–20s, restart beam search with
  completely different structural priors (e.g., switch from polynomial
  to nested-periodic hypothesis).
- Addresses the 30-second failure rule directly.

**Compositional Basis Expansion**:
- Add depth-2 compositions to the basis: `sin(cos(x))`, `exp(sin(x))`,
  `log(exp(x)+1)`, etc.
- This would directly address the Tier 5+ performance cliff without
  requiring evolution for common nested patterns.

**Residual-Guided Iteration**:
- After fast-path produces a partial solution, analyze residuals for
  structure (currently only done in `refine_powers`).
- Generalize: fit residuals with a second fast-path pass to discover
  `y = fast_path_1(x) + fast_path_2(x)` decompositions.

### 13.2 Medium-Term Improvements

**Classifier Architecture Upgrade**:
- Train a **composition-aware classifier** that predicts not just
  `{sin, cos, power}` but also `{sin∘power, exp∘sin, ...}`.
- Use a hierarchical label space that encodes nesting depth.

**Adaptive Macro-Mutation Rate**:
- Increase macro-mutation probability (currently 15%) when stagnation
  is detected. This would help escape local optima.
- Implement mutation probability annealing: high early (exploration),
  low late (exploitation).

**Multi-Input Feature Engineering**:
- Replace per-variable slicing with a learned feature extractor that
  captures variable interactions.
- Train the classifier on synthetic multivariate formulas.

### 13.3 Long-Term Research

**Neural-Guided Search**:
- Train a transformer model on (data, formula) pairs to directly
  propose formula skeletons, bypassing both the classifier and
  combinatorial basis search.

**Library Learning / DSL Expansion**:
- Maintain a growing library of discovered sub-expressions.
- Compose new formulas from library primitives rather than
  atomic operators.

**Pareto Front Exploitation**:
- The C++ engine already maintains a complexity-vs-accuracy Pareto
  front. Expose this to the user as a menu of solutions at different
  complexity levels, rather than returning only the "best" one.

---


---

## Appendix A: Key File Reference

| File | Purpose |
|------|---------|
| `scripts/generate_curve_data.py` | Synthetic data generation |
| `scripts/train_curve_classifier.py` | Classifier training |
| `scripts/curve_classifier_integration.py` | Inference integration |
| `scripts/classifier_fast_path.py` | Fast-path + beam search + guided evolution |
| `scripts/simplify_formula.py` | Formula simplification |
| `scripts/calibrate_classifier.py` | Isotonic calibration |
| `scripts/benchmark_suite.py` | 8-tier benchmark (~200 formulas) |
| `scripts/run_srbench_local.py` | SRBench PMLB + ground-truth evaluation |
| `glassbox/sr/sklearn_wrapper.py` | Scikit-learn compatible `GlassboxRegressor` |
| `glassbox/sr/evolution.py` | FFT detection, PyTorch evolution utilities |
| `glassbox/sr/cpp/evolution.h` | C++ evolutionary engine core |
| `glassbox/sr/cpp/eval.h` | C++ expression evaluator |
| `glassbox/sr/operation_dag.py` | PyTorch ONN DAG model |
| `glassbox/sr/meta_ops.py` | Meta-operators for ONN |
| `glassbox/sr/phased_regression.py` | Two-phase structure/coefficient solver |
| `glassbox/sr/pruning.py` | Sensitivity-based graph pruning |

## Appendix B: Optimization & Search Tactics Summary

| Technique | Stage | Role / Purpose |
|-----------|-------|----------------|
| Curve Classification | 1 | Zero-shot operator & complexity prior generation |
| FFT Frequency Seeding | 1, 4 | Seed ω values from signal frequencies |
| Harmonic consistency check | 1 | Detect square/triangle waves |
| Iterative L-BFGS Refiner | 2 | Sparsity-aware coefficient discovery with multi-start |
| Universal basis mode | 2 | Ensure coverage even with low classifier confidence |
| Exact symbolic match search | 2 | Find solutions without LASSO approximation |
| LASSO coordinate descent | 2 | O(n·m) incremental residual optimization |
| OLS refit on LASSO support | 2 | Recover exact coefficients from sparse structure |
| OOD holdout scoring | 2 | Penalize overfitting to domain edges |
| Gradient-based ω refinement | 3 | Fix FFT grid quantization errors |
| Continuous power refinement | 3 | Discover non-integer exponents |
| Periodic×rational refinement | 3 | Fit Lorentzian-modulated signals |
| Residual FFT analysis | 3 | Discover hidden frequencies in fit residuals |
| Hard-Concrete Selectors | 4 | Differentiable discrete operation selection |
| Risk-Seeking RL (RSPG) | 4 | Focus search on top-k percentile of solutions |
| Lamarckian Evolution | 4 | Inheritance of optimized weights across generations |
| Entropy Annealing | 4/5 | Transition from soft exploration to hard decisions |
| Classifier trust blending | 4 | Scale beam diversity by classifier confidence |
| Elite seeding | 4 | Warm-start evolution from best candidates |
| Polynomial mode detection | 4 | Expand power bounds for polynomial targets |
| Beam tournament | 4 | Explore diverse C++ evolution configurations |
| NSGA-II / AFPO selection | 5 | Multi-objective diversity preservation |
| Macro-mutations | 5 | Structural exploration (wrap/multiply/nest) |
| Local SGD (C++) | 5 | Continuous parameter refinement in C++ engine |
| Ridge regression (C++) | 5 | Instant linear layer optimization in C++ |
| In-loop constant snapping | 5 | Simplify during evolution, not just post-hoc |
| Uncertainty-budget routing | 5 | Adaptive compute allocation based on confidence |
| Sensitivity-based Pruning | 6 | Ablation-based removal of low-impact nodes |
| Mask & Fine-tune Recovery | 6 | Validating pruning via recovery optimization |
| BIC backward elimination | 6 | Remove spurious LASSO artifacts |
| Multi-pass SymPy simplify | 6 | Algebraic identity folding |
| Float snapping | 6 | π, e, integer recognition |
