# Curve Classifier Audit Issues

## Critical Issues

1. Inference loader is hard-coded to MLP checkpoint structure
	- File: scripts/curve_classifier_integration.py
	- Problem: _load_pytorch_classifier reads state_dict['net.0.weight'] and reconstructs CurveClassifierMLP only.
	- Impact: Any CNN-trained .pt checkpoint (train_curve_classifier.py supports --model cnn) can fail to load or load incorrectly.

2. Multi-input classifier bug in evolution warm-start
	- File: glassbox/sr/evolution.py
	- Problem: In train(), x_original is flattened before predict_operators is called.
	- Impact: For n_inputs > 1, shape information is destroyed, so multi-input slicing logic in predictor is bypassed or corrupted.

3. Duplicate classifier execution per training run
	- File: glassbox/sr/evolution.py
	- Problem: Classifier prediction path is executed in initialize_population(...) and then again in train(...).
	- Impact: Extra latency, redundant imports, and possible inconsistency if thresholds/model path differ over calls.

## High-Impact Bottlenecks

4. Training pipeline materializes full arrays into RAM even with memmap support
	- File: scripts/train_curve_classifier.py
	- Problem: .dat memmaps are loaded, but then train/val subsets are converted to torch tensors in memory.
	- Impact: Large datasets (hundreds of thousands / millions) can exceed memory and remove streaming benefit.

5. Exact symbolic fast-path search is combinatorial
	- File: scripts/classifier_fast_path.py
	- Problem: Pair/triple search loops over O(B^2) and O(B^3) basis combinations before sparse regression.
	- Impact: Runtime spikes when basis count approaches the exact_match_max_basis threshold.

6. Multi-input inference interpolation cost per call
	- File: scripts/curve_classifier_integration.py
	- Problem: LinearNDInterpolator/NearestNDInterpolator is rebuilt for each prediction call.
	- Impact: Significant overhead for repeated calls inside optimization loops.

## Medium Issues

7. Global operator-class cache is not keyed per model
	- File: scripts/curve_classifier_integration.py
	- Problem: _cached_operator_classes is global and overwritten whenever another model is loaded.
	- Impact: Potential class-order mismatch if multiple models with different class sets are used in one process.

8. Multi-input median is recomputed inside each variable loop
	- File: scripts/curve_classifier_integration.py
	- Problem: x_medians = np.median(x, axis=0) is inside for var_idx in range(n_vars).
	- Impact: Avoidable repeated work.

9. Stratified split top-up uses O(n) list membership checks repeatedly
	- File: scripts/train_curve_classifier.py
	- Problem: remaining = [i for i in indices if i not in val_indices and i not in train_indices]
	- Impact: Can become slow for large n; using sets would be cheaper.

10. Documentation drift: references to CNN as default inference path
	 - File(s): docs/readme messaging vs actual integration code
	 - Problem: Runtime integration is feature-based MLP/XGBoost path; CNN is optional training architecture, not guaranteed inference path.
	 - Impact: Confusion in expectations and debugging.

## Recommended Fix Order

1. Fix multi-input flatten bug in evolution.py.
2. Remove duplicate classifier call path (single source of truth).
3. Make model metadata/operator classes cache key-specific.
4. Add robust checkpoint architecture detection (MLP vs CNN) in integration loader.
5. Optimize multi-input prediction inner loop (precompute medians, reduce interpolation cost).
6. Improve large-data training path (true streamed dataset loader).
7. Bound/short-circuit exact symbolic combinatorial search for large bases.

