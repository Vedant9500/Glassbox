# Curve Classifier End-to-End Algorithm

This document describes the full pipeline from synthetic data generation to final operator prediction and ONN warm-start / fast-path regression.

---

## 1) Data Generation Stage

Primary script: scripts/generate_curve_data.py

### 1.1 Operator label space

The classifier predicts multi-label operator presence over 9 classes:
- identity
- sin
- cos
- power
- exp
- log
- addition
- multiplication
- rational

Each sample can have multiple active labels.

### 1.2 Template families used

Templates are grouped into these families:
- SIMPLE_TEMPLATES
- COMPOUND_TEMPLATES
- RATIONAL_TEMPLATES
- NESTED_TEMPLATES
- PRODUCT_TEMPLATES
- IRRATIONAL_TEMPLATES
- HYPERBOLIC_TEMPLATES
- PHYSICS_TEMPLATES

ALL_TEMPLATES is the concatenation of all groups.

Representative examples by family:
- Simple: x, a*x+b, x**2, sin(a*x+b), exp(a*x), log(|x|+1)
- Compound: sin(x)+x**2, sin(x)*cos(x), x*exp(-x), log(|x|+1)+sin(x)
- Rational: 1/(|x|+c), x/(x**2+c), (x+a)/(x**2+b)
- Nested: sin(x**2), exp(sin(x)), log(x**2+1), sin(exp(-x**2))
- Product/modulated: x*sin(x), x**2*cos(x), exp(-x**2)*sin(a*x)
- Irrational constants: sin(pi*x), e**x, x**sqrt(2)
- Hyperbolic/sigmoid-like: sinh(x), tanh(x), 1/(1+exp(-x))
- Physics-inspired: 1/sqrt(x**2+c), x/(x+c), Gaussian forms, power-law decays

### 1.3 Numeric sampling defaults and ranges

CLI defaults in generation main():
- x-min = -5
- x-max = 5
- n-points = 256
- x-scale-min = 0.8
- x-scale-max = 1.2
- x-shift-std = 0.05 (fraction of x-span)
- noise-std = 0.01 (fraction of y std)
- y-scale-min = 0.8
- y-scale-max = 1.2
- y-offset-std = 0.05
- signed-bd default = on (unless --unsigned-bd)
- safe_eval default = on (unless --unsafe-eval)

Per-sample coefficient ranges inside generate_chunk():
- a ~ Uniform(-5, 5)
- b ~ Uniform(0.3, 6.0)
- c ~ Uniform(-3, 3)
- d ~ Uniform(0.3, 6.0)
- p sampled from:
  [0.25, 0.33, 0.5, 0.67, 1.0, 1.5, 2.0, 2.3, 2.5, 3.0, 4.0, -0.5, -1.0, -2.0]

If signed_bd is enabled, b and d randomly get sign +/-.

### 1.4 Formula evaluation and rejection

For each sampled formula:
1. Evaluate with restricted AST evaluator (safe mode) or eval mode.
2. Reject sample if:
   - NaN/Inf exists in y
   - max(|y|) > 1e6
   - feature extraction fails
3. Continue sampling until target count reached (with max_attempts guard).

### 1.5 Label construction

Labels are derived from AST (not only template tags):
- sin/cos/exp/log calls detected from np.* calls
- power detected from ** and sqrt
- rational detected when denominator depends on x
- addition/multiplication from binary operators
- identity when x is present

Converted into 9-dim multi-hot vector.

### 1.6 Feature extraction (334 dimensions)

Feature vector is concatenation of:
1. Raw shape features (128)
   - resample y to 128 points
   - min-max normalize to [0,1]
2. FFT magnitude features (32)
   - demean + Hann window
   - rFFT magnitudes, normalized
3. Derivative features (128)
   - first derivative (64 resampled points)
   - second derivative (64 resampled points)
4. Statistical features (9)
   - mean, std, min, max, median, skew, kurtosis, zero-crossings, extrema count
5. Curvature features (37)
   - smoothed curvature profile (32)
   - curvature summary stats (5)

Final feature dimension must equal 334.

### 1.7 Output modes

Two generation modes:
- In-memory dataset + npz save
- Streamed mode to memmap files:
  - .features.dat (float32, shape n_samples x 334)
  - .labels.dat (float32, shape n_samples x 9)
  - optional .formulas.txt

---

## 2) Classifier Training Stage

Primary script: scripts/train_curve_classifier.py

### 2.1 Supported model architectures

- MLP (default)
- CNN (optional via --model cnn)

Both are multi-label classifiers with sigmoid(BCE logits) output behavior.

### 2.2 Data loading

Accepts:
- Single .npz file
- Or streamed .features.dat + .labels.dat

Optional load-into-ram flag can force full materialization.

### 2.3 Train/validation split

- Default split ratio: 0.1 validation
- Default behavior: approximate multi-label stratified split enabled
- Optional random split fallback

### 2.4 Feature standardization

Default is ON:
- mean/std computed on training subset
- applied to train and val
- scaler metadata saved in checkpoint

### 2.5 Optimization and metrics

Training uses:
- Adam optimizer
- ReduceLROnPlateau scheduler (on val loss)
- BCEWithLogitsLoss (optional class pos_weight)
- Gradient clipping
- Early stopping by val F1 (default) or val loss

Validation metrics include:
- loss
- binary accuracy
- per-class accuracy
- mean F1
- micro-F1
- per-class F1

### 2.6 Calibration and threshold tuning

Optional post-training extras:
- Temperature scaling calibration (single T)
- Per-class threshold tuning over [0.05..0.95]

Saved checkpoint can include:
- model_state_dict
- operator_classes
- thresholds
- temperature
- feature_scaler
- feature_schema
- feature_dim

---

## 3) Inference Stage (Curve Classifier Integration)

Primary script: scripts/curve_classifier_integration.py

### 3.1 Model loading

1. Resolve device (auto/cpu/cuda with fallback warning).
2. Cache model by key "device:model_path".
3. Load .pt (PyTorch) or .pkl/.joblib (XGBoost).
4. Load associated metadata (thresholds, temperature, scaler).

### 3.2 Single-input prediction flow

Input:
- x shape (N,) or (N,1)
- y shape (N,)

Flow:
1. Extract 334-d features from y.
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

### 3.3 Multi-input prediction flow (n_vars > 1)

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

## 4) ONN Warm-Start Stage

Integration points in glassbox/sr/evolution.py.

### 4.1 Population seeding path

During initialize_population(x,y):
1. Predict operators from curve classifier.
2. Detect dominant FFT frequencies.
3. Randomly initialize individuals.
4. Probabilistically bias operation selector logits based on predicted probabilities.
5. Optionally seed omega values from FFT frequencies.

### 4.2 Train-time bias path

During train(...):
1. Run curve classifier prediction again.
2. Bias logits in population and part of explorer population.
3. Continue constant refinement and evolutionary loop.

Result:
- Search is steered toward likely operator families earlier than purely random init.

---

## 5) Fast-Path Direct Regression Stage

Primary script: scripts/classifier_fast_path.py

### 5.1 Trigger

run_fast_path(x,y, classifier_path, ...):
1. Call predict_operators.
2. If confidence is sufficient, continue; else fallback to full evolution.

### 5.2 Basis construction from predictions

Construct regression basis using predicted operators (and optional constraints):
- Constant term
- Polynomial powers (integer + fractional)
- Periodic sin/cos terms over omega list (defaults + FFT-detected)
- Exponential families
- Log terms
- Composition terms in universal mode (e.g., sin(x^2), sin(1/x))

### 5.3 Solve strategy

1. Optional exact symbolic search over pairs/triples (or 4-term when power-only).
2. If no exact match:
   - normalize basis
   - run LASSO coordinate descent for multiple alpha values
   - pick best complexity-penalized score
   - refit selected terms with OLS
3. Build sparse symbolic formula and compute MSE.

### 5.4 Optional guided evolution fallback

From fast-path formula:
1. Extract operator hints and frequencies.
2. Build operator-constrained ONN factory.
3. Run smaller guided evolution to seek exact symbolic form.

---

## 6) End-to-End Answer Path (Concise Step List)

Step 1: Generate synthetic formulas from template families and random coefficient ranges.

Step 2: Evaluate formulas on sampled x domains with safety checks and reject invalid curves.

Step 3: Convert formulas to multi-label operator targets using AST parsing.

Step 4: Extract 334 curve features (raw + FFT + derivatives + stats + curvature).

Step 5: Train MLP/CNN classifier, then save model with calibration/threshold/scaler metadata.

Step 6: For a new curve, extract features (or multi-input slices), run classifier, and produce operator probabilities.

Step 7: Threshold probabilities to get likely operators; add derived periodic/exponential/polynomial hints.

Step 8: Use hints either to:
- warm-start ONN evolution (bias operation logits), or
- run fast-path basis regression for immediate symbolic approximation.

Step 9: If needed, run guided/full evolution to improve approximation to exact symbolic expression.

Step 10: Return discovered formula and fitness metrics (MSE/correlation), which is the final answer for symbolic regression.
