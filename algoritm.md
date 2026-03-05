# Curve Classifier End-to-End Algorithm

This document describes the full pipeline from synthetic data generation to final operator prediction and ONN warm-start / fast-path regression.

---

## 1) Data Generation and Classifier Training Stage

Primary scripts:
- `scripts/generate_curve_data.py`
- `scripts/train_curve_classifier.py`

The classifier is trained from synthetic formula templates and label-aware feature extraction (no adversarial self-play stage).

### 1.1 Synthetic Formula and Curve Generation

1. Sample formula templates spanning polynomial, periodic, exponential, logarithmic, rational, and nested compositions.
2. Randomize coefficients/constants and evaluate curves over sampled x-domains.
3. Reject invalid samples (NaN/Inf/divergent outputs), then apply optional noise augmentation.
4. Extract a 366-dimensional feature vector per curve (raw curve slice, FFT, derivatives, summary stats, curvature).
5. Save `(features, labels)` and metadata to `.npz` datasets.

### 1.2 Label Space

The dataset uses multi-hot labels indicating which operator families appear in a generated curve:
- `identity`
- `sin`
- `cos`
- `power`
- `exp`
- `log`
- `addition`
- `multiplication`
- `rational`

Depending on dataset config, additional constant-indicator classes can also be included.

### 1.3 Classifier Training

`train_curve_classifier.py` trains either an MLP or 1D-CNN classifier:
1. Split dataset into train/validation/test sets.
2. Compute feature normalization stats from train split only.
3. Train with `BCEWithLogitsLoss` and gradient clipping for stability.
4. Optionally calibrate logits with temperature scaling and per-class thresholds.
5. Save model checkpoints and training metadata under `models/`.

---

## 2) Inference Stage (Curve Classifier Integration)

Primary script: scripts/curve_classifier_integration.py

### 2.1 Model loading

1. Resolve device (auto/cpu/cuda with fallback warning).
2. Cache model by key "device:model_path".
3. Load .pt (PyTorch) or .pkl/.joblib (XGBoost).
4. Load associated metadata (thresholds, temperature, scaler).

### 2.2 Single-input prediction flow

Input:
- x shape (N,) or (N,1)
- y shape (N,)

Flow:
1. Extract 366-d features from y.
2. Apply saved scaler if present.
3. Run classifier:
   - PyTorch: logits -> sigmoid (with optional temperature)
   - XGBoost: per-class predict_proba
4. Apply thresholds (per-class if saved, otherwise global threshold).
5. Build operator probability dictionary.
6. Add derived compatibility outputs:
   - periodic = max(sin, cos)
   - exponential = max(exp, log)
   - polynomial = max(power, identity)

### 2.3 Multi-input prediction flow (n_vars > 1)

Input:
- x shape (N, n_vars)
- y shape (N,)

Flow:
1. Build ND interpolator for y(x):
   - try LinearNDInterpolator
   - fallback NearestNDInterpolator
2. For each variable v:
   - fix all other variables at median values
   - sweep variable v across its observed min/max range
   - generate 1D slice y_v
   - remove NaN points
   - extract features and predict class probabilities
3. Aggregate across variables by max probability per class.
4. Threshold and return final operator dictionary + derived keys.

---

## 3) Fast-Path Direct Regression & Beam Search Stage

Primary script: `scripts/classifier_fast_path.py`

### 3.1 Trigger
When the classifier outputs probabilities for a curve, we run `run_guided_evolution(x, y, model_path, ...)`:
1. It first attempts the `fast_path_regression` to find an immediate symbolic approximation using LASSO.
2. If the fast path fails to find a high-quality solution, it falls back to `beam_search_evolution`.

### 3.2 Fast-Path Regression Strategy
1. **Basis Construction**: Construct a large, dense matrix of base functions using predicted operators (e.g., polynomials, $\sin(\omega x)$, $e^x$, $\log(|x|)$).
2. **Exact Search**: Do a combinatorial combinatorial exact match over basis pairs/triples (bounded by `exact_match_max_basis` to prevent explosions).
3. **LASSO**: If no exact match, run L1-regularized coordinate descent for feature selection.
4. **OLS Refit**: Take the sparse LASSO terms and refit them with simple Least Squares via `np.linalg.lstsq`.

### 3.3 Beam Search Evolution Fallback
If direct linear algebra fails, we generate diverse starting points for structural evolution:
- Initializes 20 "beams" (15 synthetically handcrafted from classifier priors, 5 purely random).
- Runs 3 rounds of short, parallel C++ evolution sequences.
- Retains the top 20% most diverse and physically accurate graphs across rounds.
- Feeds these winners into the main C++ Evolution loop as elite seeds.

---

## 4) C++ Evolutionary Search Stage

Primary core: `glassbox/sr/cpp/evolution.h` and `glassbox/sr/cpp/eval.h`

### 4.1 The C++ Engine (`EvolutionEngine`)
The main symbolic regression workhorse is now entirely written in C++ for extreme performance. Evaluates expressions fundamentally as `IndividualGraph` Directed Acyclic Graphs (DAGs) rather than PyTorch modules.

### 4.2 Age-Fitness Pareto Optimization (AFPO) & Selection
Evolution maintains diversity and prevents premature convergence by using a 3-objective **NSGA-II** selection tournament:
1. Minimize `MSE`
2. Minimize `Complexity` (number of graph nodes)
3. Minimize `Age` (surviving generations). Young graphs are protected while their continuous parameters mature.

### 4.3 Macro-Mutations
In addition to standard point mutations, the engine applies structural "Macro-Mutations" (15% probability) to preserve winning building blocks:
- **Wrap Mutation** (50%): $f(x) \rightarrow \sin(f(x))$ or $\exp(f(x))$
- **Multiply Mutation** (30%): Combines sub-graphs $f(x) \times g(x)$
- **Nest Mutation** (20%): Submits one sub-graph into another $f(g(x))$

### 4.4 Continuous Optimization & Snapping
Instead of PyTorch Autograd, continuous parameter discovery is handled internally:
1. **Local SGD:** Finite-difference Adam (`refine_inner_params`) tweaks frequencies, phases, and decay rates.
2. **Ridge Regression:** Evaluates linear combinations of graph terminal nodes to instantly solve output weights.
3. **In-Loop Snapping:** `cleanup_graph` actively snaps these floating-point parameters to exact integers, fractions, or constants ($\pi, e$) during the run if it degrades MSE by less than 1.01x.

---

## 5) End-to-End Answer Path (Concise Step List)

Step 1: Generate synthetic formula/curve pairs from templates and extract 366-d features.

Step 2: Train and calibrate the curve classifier (`train_curve_classifier.py`) and save the checkpoint.

Step 3: For a new target curve, extract features (or per-variable slices for multi-input data).

Step 4: Run `curve_classifier_integration.py` to obtain operator probabilities and thresholded class predictions.

Step 5: `classifier_fast_path.py` attempts a direct symbolic fit via exact subset search, LASSO, and OLS refit.

Step 6: If fast-path quality is insufficient, run beam-search-initialized C++ evolution.

Step 7: Evolve/refine expressions, apply simplification/snapping, and return final formula with metrics (MSE/correlation).
