# Context Summary: Fast-Path Symbolic Regression & Scaling

## Current State
We have implemented a **Fast-Path** for symbolic regression that skips evolution for common function types, using a neural classifier to predict operators and LASSO regression for coefficients.

### 1. Classifier-Guided Regression (Fast-Path)
- **Files**: `scripts/classifier_fast_path.py`, `scripts/curve_classifier_integration.py`
- **Logic**: 
  1. Inspects curve shape (x, y) using extracted features (Raw point, FFT, Derivative, Stats, Curvature).
  2. Neural Network predicts operators: `sin`, `cos`, `exp`, `log`, `rational`, `periodic`, etc.
  3. Constructs a **regression basis** based on predictions (Universal basis always includes Poly + Log + Sin/Cos).
  4. Solves `y = Xw` using **LASSO** (for sparsity) followed by **OLS Refit** (for precision).
  5. Includes **Frequency Refinement** (FFT + Gradient Descent) for finding internal frequencies like `sin(3.2x)`.

### 2. Dataset Scaling (v2)
- **Data Generation**: `scripts/generate_curve_data.py` now supports **multiprocessing** and generates 1M samples in minutes.
- **New Features (334 total)**: Added `curvature` (y', y'') features to distinguish `exp(-x)` from `1/x`.
- **New Classes**: Added `rational` class with templates like `1/(x+c)`, `x/(x²+c)`.
- **Model**: Trained on 1M samples with 334 features. Validation Loss: **0.0075** (vs 0.0314 previously).
- **Safety**: `scripts/curve_classifier_integration.py` dynamically detects feature size from checkpoint weights.

### 3. Usage & Verification
- **Test Script**: `scripts/sr_tester.py` (Main entry point).
- **Benchmarks**: `scripts/verify_fast_path.py` (Runs 5 key benchmarks).
- **Verified Success**:
  - Nguyen-7 (`log(x+1)+log(x²+1)`): Passed (MSE 8e-6).
  - Damped Sine (`e^-0.2x sin(5x)`): Passed (MSE 1e-4).
  - Rational (`x/(x²+1)`): Passed (MSE 0.0).

## Next Steps / Known Limitations
- **Rational Functions**: We handle `1/(x+c)` and `1/(x²+c)` well, but complex rationals `(Ax+B)/(Cx+D)` rely on basis approximation or evolution fallback.
- **Nested Functions**: Deep nesting like `sin(cos(exp(x)))` is not in the basis and falls back to evolution.
- **Evolution Fallback**: If Fast-Path MSE > 0.01, it falls back to standard evolution (which is slower).
