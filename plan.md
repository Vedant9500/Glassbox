# Data generation + training review plan

## Scope
Focus on data generation in scripts/generate_curve_data.py and training in scripts/train_curve_classifier.py.

## Summary of issues and best‑practice fixes

### 1) Feature dimension mismatch (bugs + silent failures) [x]
- **Issue:** `extract_all_features()` now yields 334 features, but comments/defaults still reference 297.
- **Fix:** Update all defaults and comments to 334; assert feature length at load time.
- **Why it’s best:** Prevents silent shape mismatch and model drift; standard practice to validate feature schemas.

### 2) Inconsistent operator labeling (label noise) [x]
- **Issue:** Templates mix `power`, `rational`, `addition`, `multiplication` inconsistently.
- **Fix:** Normalize operator tagging rules: if formula contains `+` or `-`, add `addition`; if it contains `*` or `/`, add `multiplication`; tag `rational` separately from `power` unless explicitly polynomial.
- **Why it’s best:** Reduces label noise; label quality is the dominant factor for multi‑label learning.

### 3) Redundant/overlapping class ontology [x]
- **Issue:** `periodic` overlaps `sin`/`cos`, `polynomial` overlaps `power`, `exponential` conflates `exp` and `log`.
- **Fix:** Choose a single ontology:
  - **Option A (fine‑grained):** keep only base operators (`sin`, `cos`, `power`, `exp`, `log`, `rational`, `addition`, `multiplication`, `identity`).
  - **Option B (high‑level):** keep only `periodic`, `polynomial`, `exponential`, `rational`, `add`, `mul`.
- **Why it’s best:** Multi‑label models struggle with redundant targets; a consistent ontology improves calibration and evaluation.

### 4) Imbalanced label distribution [x]
- **Issue:** Templates are not balanced; rational and log are rare.
- **Fix:** Enforce class‑aware sampling or per‑template quotas; extend `--rational-ratio` idea to all classes.
- **Why it’s best:** Balanced sampling improves macro‑F1; standard approach for multi‑label imbalance.

### 5) Non‑deterministic multiprocessing seeds [x]
- **Issue:** Per‑worker reseeding uses OS entropy even with `--seed`, breaking reproducibility.
- **Fix:** Derive deterministic worker seeds from the global seed and worker index.
- **Why it’s best:** Reproducibility is critical for research; best practice in parallel data generation.

### 6) Chunking bug when `n_samples < n_workers` [x]
- **Issue:** `chunk_size` becomes 0 for small sample counts, causing wasted workers and uneven sampling.
- **Fix:** Set `n_workers = min(cpu_count-1, n_samples)` and distribute remainder correctly.
- **Why it’s best:** Ensures correct sampling and consistent throughput across dataset sizes.

### 7) `eval()` risk in formula execution [x]
- **Issue:** `eval()` remains risky if formulas ever become external or user‑provided.
- **Fix:** Switch to a safe parser (e.g., `numexpr`, or a small custom AST evaluator) if formulas are external; otherwise keep as internal‑only with strict template control.
- **Why it’s best:** Avoids security risk; standard practice for formula evaluation.

### 8) Feature scaling inconsistencies [x]
- **Issue:** Raw, derivative, stats, FFT, curvature features use different normalization schemes.
- **Fix:** Apply a single dataset‑level standardization (mean/variance) and save the scaler for inference.
- **Why it’s best:** Improves optimization stability and generalization; standard ML practice.

### 9) FFT feature leakage due to resampling artifacts [x]
- **Issue:** FFT of a resampled curve without windowing can introduce high‑frequency artifacts.
- **Fix:** Apply a window (e.g., Hann) or detrend/mean‑remove prior to FFT.
- **Why it’s best:** Reduces spectral leakage; standard DSP practice.

### 10) Curvature features are numerically noisy [x]
- **Issue:** Curvature from discrete derivatives is unstable and clipped.
- **Fix:** Smooth curve before computing derivatives, or compute robust summary stats only.
- **Why it’s best:** Noise reduction yields more reliable features; standard numerical practice.

### 11) Loss + sigmoid stability [x]
- **Issue:** `sigmoid` inside model with `BCELoss` is less stable.
- **Fix:** Use `BCEWithLogitsLoss` and remove final sigmoid in models. Apply sigmoid only at inference.
- **Why it’s best:** More numerically stable; recommended by PyTorch docs.

### 12) Non‑stratified train/val split [x]
- **Issue:** Random split can cause rare labels to disappear in validation.
- **Fix:** Use multi‑label stratification (iterative stratification) or template‑level splitting.
- **Why it’s best:** Ensures reliable validation metrics for rare classes.

### 13) Thresholding at 0.5 for all classes [x]
- **Issue:** Fixed threshold is suboptimal under imbalance.
- **Fix:** Calibrate per‑class thresholds using validation PR curves; store thresholds.
- **Why it’s best:** Improves macro‑F1 and reduces false negatives for rare operators.

### 14) Metrics can be misleading [x]
- **Issue:** Overall accuracy is inflated by negatives.
- **Fix:** Report macro‑F1, micro‑F1, per‑class AP/AUROC; keep accuracy as secondary.
- **Why it’s best:** Multi‑label best practice and robust to imbalance.

### 15) Dataset domain mismatch [x]
- **Issue:** Curves are too clean (no noise, limited scaling/domain shifts).
- **Fix:** Add noise, random x/y scaling, offsets, and domain shifts; include missing‑data patterns.
- **Why it’s best:** Improves generalization to real‑world symbolic regression data.
### 16) Parameter sampling bias [x]
- **Issue:** Parameter ranges are asymmetric (e.g., positive‑only frequency terms) and mixed use of `abs(x)` introduces uncontrolled symmetry.
- **Fix:** Define a parameter sampling policy with controlled symmetry and sign distributions; optionally stratify by symmetry class.
- **Why it’s best:** Reduces dataset bias and improves generalization.

### 17) CNN feature ordering is implicit [x]
- **Issue:** CNN assumes the first 128 features are the raw curve; if feature ordering changes, training silently breaks.
- **Fix:** Encode feature schema with explicit slices or store raw curve separately.
- **Why it’s best:** Prevents brittle coupling and regression bugs.

### 18) Early stopping uses only validation loss [x]
- **Issue:** With imbalanced labels, validation loss may not correlate with macro‑F1.
- **Fix:** Early stop on macro‑F1 or a composite score; log both loss and F1.
- **Why it’s best:** Aligns stopping criterion with target metric.

### 19) Lack of calibration for probabilistic outputs [x]
- **Issue:** Sigmoid outputs are treated as calibrated probabilities but are not calibrated.
- **Fix:** Apply temperature scaling or isotonic calibration on validation data; store calibration parameters.
- **Why it’s best:** Improves decision‑making, threshold tuning, and interpretability.

### 20) Label noise from ambiguous templates [x]
- **Issue:** Some templates encode operations (e.g., multiplication/division) that are not reflected in labels.
- **Fix:** Auto‑derive labels from parsed expression tree to ensure consistency.
- **Why it’s best:** Eliminates human‑error label noise.

### 21) Lack of out‑of‑range/invalid curve handling policy [x]
- **Issue:** Curves with extreme values are discarded but without tracking; this can bias the dataset.
- **Fix:** Log rejection reasons and optionally cap values instead of discarding.
- **Why it’s best:** Preserves distribution fidelity and debuggability.

### 22) No saved feature scaler or metadata with model [x]
- **Issue:** Training does not persist feature normalization parameters or dataset schema with the model.
- **Fix:** Save scaler stats, feature schema, and class ontology alongside the model.
- **Why it’s best:** Ensures reproducible inference and prevents mismatch errors.

---

## Recommended implementation order
1) Fix feature schema/size mismatch and add assertions.  
2) Normalize operator labels and ontology.  
3) Implement deterministic seeding and robust chunking.  
4) Balance sampling across classes/templates.  
5) Standardize features dataset‑wide and store scaler.  
6) Update training loss to `BCEWithLogitsLoss`.  
7) Improve evaluation: stratified split, threshold tuning, macro‑F1/AP.  
8) Add noise/domain augmentation to data.

---

## Notes on “best way” validation
The fixes above reflect standard practices in multi‑label classification, feature pipelines, and DSP. They are widely accepted (e.g., `BCEWithLogitsLoss` for stability, stratified splitting for imbalanced multi‑label data, dataset‑level standardization, per‑class threshold calibration). They should integrate cleanly with the current architecture without requiring a redesign.


# Raw output 

Data generation issues

Feature dimension mismatch in docs and model defaults: extract_all_features() returns 334 features, but comments and defaults elsewhere still mention 297 (old size). This will silently break if the dataset is generated with the new feature size and the model defaults aren’t updated. Fix the comments and ensure all defaults align to 334. See generate_curve_data.py and train_curve_classifier.py.
Operator labeling is inconsistent: Some formulas that are rational are also tagged as power even when they are not pure polynomial/power forms, and some rational templates don’t include addition/multiplication even when present in the formula string. This causes label noise and harms training. See the rational templates in generate_curve_data.py.
Periodic/operator tags are too coarse: periodic is set only when sin/cos appear, but exponential/log tags are mixed under exponential. This conflates two different families and makes labels ambiguous. Consider splitting log and exp into separate high‑level tags or removing the higher‑level labels entirely. See operator definitions in generate_curve_data.py.
Template parameter ranges can create unrealistic curves: b and d are always positive while a and c can be negative; this makes the distribution biased. Some templates use np.abs(x) and others don’t, which changes symmetry in an uncontrolled way. Consider consistent sampling or explicit symmetry control. See template definitions in generate_curve_data.py.
evaluate_formula accepts any valid expression with eval. Even with a restricted globals dict, this is still a security risk if formulas become user‑provided or external. If this ever becomes user‑facing, replace eval with a safe parser. See generate_curve_data.py.
Multiprocessing chunking bug risk: When n_samples < n_workers, chunk_size becomes 0 and most chunks are size 0. This wastes workers and can bias randomness. Use min(n_workers, n_samples) and compute chunk sizes accordingly. See generate_dataset() in generate_curve_data.py.
Seeding per worker: Calling np.random.seed() and random.seed() with no arguments reseeds from OS entropy, which is fine for diversity but breaks reproducibility even when --seed is set. If reproducibility matters, derive per‑worker seeds from the global seed. See generate_chunk() in generate_curve_data.py.
Feature extraction mixes scale choices: You normalize raw features to [0,1], derivatives by max absolute, and stats from z‑scored values. This is inconsistent and may confuse the model. Consider standardizing all features or applying a single normalization pipeline. See extract_* functions in generate_curve_data.py.
FFT features are computed on normalized curve (0–1 range) but without windowing; sharp edges from resampling can inject spurious high‑frequency energy. Consider applying a window or using detrended data. See extract_fft_features() in generate_curve_data.py.
Curvature features can explode: curvature = ddy / (1 + dy**2)**1.5 is clipped to 100, but numeric instability still makes it noisy. Consider smoothing before curvature or using robust statistics only. See extract_curvature_features() in generate_curve_data.py.
Label distribution is likely heavily imbalanced: Some templates are rare (rational) and some operators are implicit (identity). You are not enforcing balanced class distribution, so the model can collapse to majority classes. You do allow a --rational-ratio, but not for other classes. See dataset generation logic in generate_curve_data.py.
Training issues

BCE with sigmoid on raw outputs is fine, but your training uses nn.BCELoss with sigmoid inside the model; this is less numerically stable than using BCEWithLogitsLoss and removing the final sigmoid. This is a standard improvement. See model forward() and criterion in train_curve_classifier.py.
The CNN assumes the “raw curve” is the first 128 features, which is true now, but not enforced. If feature ordering changes later, the CNN silently breaks. Add a schema or separate raw curve storage. See CurveClassifierCNN in train_curve_classifier.py.
There is no train/val stratification. With imbalanced labels, random split can cause some classes to have near‑zero positives in validation. Consider stratified split per class or at least per template. See train/val split in train_curve_classifier.py.
Metrics are misleading: overall_acc is computed as mean of elementwise matches across all labels; with imbalance, a trivial all‑zero predictor can show high accuracy. You do compute F1, but you should emphasize macro‑F1 and maybe AUROC/AP. See evaluate() in train_curve_classifier.py.
Early stopping on validation loss only: With class imbalance and label noise, val loss can be unstable. Consider stopping on macro‑F1. See train_model() in train_curve_classifier.py.
No threshold tuning: You use a fixed 0.5 threshold for all classes. For imbalanced labels, per‑class thresholds or calibration typically improves results. See evaluate() in train_curve_classifier.py.
Lack of dataset normalization or standardization: You feed raw feature values into the network; but features are on very different scales. Consider normalization (z‑score on dataset) and store the scaler. See training data pipeline in train_curve_classifier.py.
Design concerns

Synthetic data domain mismatch: The formulas are narrow and structured; real symbolic regression outputs can include noise, offsets, scaling, and domain restrictions not represented here. The classifier may not generalize. Consider adding noise, random rescaling of x/y, and random domain shifts. See data templates in generate_curve_data.py.
Label noise from ambiguous templates: Example: np.exp(-x ** 2) is tagged power and exp but not multiplication. The model can’t learn consistent operator patterns if templates are not tagged coherently. See templates in generate_curve_data.py.
Class definitions mix low‑level ops with high‑level tags: identity, power, and polynomial are overlapping; periodic overlaps sin/cos. This creates redundancy and can inflate false positives. Consider a single consistent ontology: either fine‑grained or high‑level, not both. See OPERATOR_CLASSES in generate_curve_data.py.