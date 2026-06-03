# Codebase Audit & Architectural Review

## 1. Executive Summary
- The audited components are capable, but the integration surface is fragmented. `GlassboxRegressor` has the most complete production path, while standalone classifier, proposer, evolution, and training scripts have diverged defaults and incomplete multivariate behavior.
- The most critical systemic risk is multivariate drift. The universal proposer advertises multivariate support and contains multivariate decoding logic, but the public inference helper rejects real multivariate arrays. The sklearn wrapper works around this by projecting multivariate `X` to a one-dimensional norm proxy, so the true multivariate proposer path is not exercised by the main pipeline.
- The curve classifier has a verified preprocessing inconsistency in its multivariate interaction branch. Most inference paths apply the same selective SymLog transform used by training, but diagonal interaction slices skip it before scaling.
- Evolution has a correctness split between the C++ early-return path and the Python fallback when `normalize_data=True`. The Python path denormalizes before final scoring; the C++ path returns formulas and display MSE in normalized coordinates.
- Several optimization/training scripts are not reliable as reproducible tooling. A removed legacy tree-classifier experiment wrote artifacts that the integration loader could not load, and multiple scripts accepted validation splits that could produce empty validation loaders and late crashes.
- Checkpoint and model-loading boundaries were not explicit. PyTorch checkpoints were loaded with pickle-enabled fallback behavior from user-supplied paths. This is acceptable only for trusted local artifacts, but the code needs to enforce or document that trust boundary.

## 2. Pipeline Data-Flow Simulation & Behavioral Insights
The primary sklearn flow starts in `glassbox/sr/sklearn_wrapper.py`. Input arrays are normalized into `X` and `y`, optional blackbox feature selection reduces the active feature set, the classifier fast path proposes formula candidates, the universal proposer injects FPIP v2 metadata, and C++ evolution consumes candidate formulas, unary/binary priors, seed graphs, timeout policy, and complexity limits. Final formulas are then postprocessed and evaluated for display/scoring consistency.

For univariate data, the data flow is mostly coherent. `predict_operators()` extracts curve features from `y`, applies selective SymLog compression to feature indices `192:398`, applies any checkpoint scaler, and runs the PyTorch classifier. `propose_from_xy()` repeats the feature extraction and scaling discipline for the universal proposer, decodes univariate grammar candidates, builds a search plan, and adapts it into FPIP v2.

For multivariate data, the flow has important gaps. `predict_operators()` switches to `_predict_operators_multi_input()`, builds interpolated one-dimensional slices for each variable, classifies each slice, detects interactions, and optionally classifies diagonal pair slices. Per-variable slices apply the SymLog transform before scaling; diagonal interaction slices do not. The result is that interaction-heavy formulas can be routed through a classifier distribution that differs from training and from the other inference branches.

The universal proposer is internally inconsistent for multivariate execution. `propose_from_xy()` rejects `x` unless it is one-dimensional or shaped `[N, 1]`, but later in the same function it branches to `grammar_decode_multivariate_skeletons()` when `x_for_plan` is two-dimensional. A runtime probe with shape `(32, 2)` raised `ValueError: Expected x to be 1D or [N,1]`. The direct multivariate decoder also mis-binds variables beyond the first two: formulas are generated as `x1*x2`, but evaluation passes a two-column temporary array whose context is named `x0`, `x1`. A probe using `y = x1*x2` with three input columns returned only `x0/x1` formulas in the top candidates, never the actual `x1*x2` interaction.

`GlassboxRegressor` previously avoided the public proposer failure by converting multivariate inputs into a centered norm proxy before calling `propose_fpip_v2_from_xy()`. That avoided exceptions but meant the proposer never saw real feature identities, feature pairs, or high-dimensional interaction structure.

The evolution flow has two different scoring contracts. The native C++ path runs before the PyTorch loop and returns early when `_core.run_evolution()` succeeds. If `normalize_data=True`, the C++ run receives normalized `x_cpu` and `y_cpu`, then display MSE is recomputed against the normalized arrays and the returned formula is left in normalized coordinate space. The Python fallback later denormalizes predictions and targets before final MSE, so the same training option has different semantics depending on backend availability.

The optimization scripts are uneven. `scripts/benchmark_suite.py` and `glassbox/sr/sklearn_wrapper.py` include sophisticated candidate, timeout, and C++ seed routing. By contrast, the removed legacy tree-classifier experiment wrote incompatible artifacts and did not share the classifier integration's SymLog transform. `scripts/train_universal_proposer.py`, `scripts/calibrate_classifier.py`, and related classifier training code accepted validation split values that could create empty validation sets, after which evaluation concatenated empty lists.

## 3. Verified Issues & Vulnerabilities
- **Component:** `glassbox/universal_proposer/universal_proposer.py`
- **Severity:** Major
- **Type:** Bug
- **Description:** `propose_from_xy()` rejects all real multivariate arrays before reaching its own multivariate branch. The function allows `[N, 1]`, then raises if `x.ndim != 1`, but later checks whether `x_for_plan` is two-dimensional to call `grammar_decode_multivariate_skeletons()`.
- **Impact:** Direct users and future benchmark paths cannot use the advertised multivariate proposer API. The main sklearn wrapper must collapse multivariate input to a norm proxy, losing feature identity and pairwise interaction structure.
- **Recommendation:** Preserve two-dimensional `x` when `x.shape[1] > 1`; validate only the sample dimension against `y`. Add a regression test that calls `propose_from_xy(model, X_2d, y)` and asserts `supports_multivariate_formulas is True`.

- **Component:** `glassbox/universal_proposer/universal_proposer.py`
- **Severity:** Major
- **Type:** Logic Error
- **Description:** `grammar_decode_multivariate_skeletons()` generates formulas with original feature names such as `x1*x2`, but evaluates each pair with `np.column_stack([xi, xj])`. `_safe_formula_eval_multivariate()` then names that temporary array `x0`, `x1`, not the original feature names.
- **Impact:** Candidate formulas for pairs beyond `(x0, x1)` are either rejected as undefined or scored against the wrong variables. High-dimensional interaction discovery is therefore biased toward the first two columns and can miss the true generating formula.
- **Recommendation:** Evaluate formulas against the full original `X` or generate pair-local formulas using `x0/x1` and map them back to original variable names after scoring. Add a test for a three-feature target such as `y = x1*x2`.

- **Component:** `glassbox/curve_classifier/curve_classifier_integration.py`
- **Severity:** Major
- **Type:** Logic Error
- **Description:** `_predict_operators_multi_input()` applies SymLog compression to per-variable slice features before scaling, but the diagonal interaction-slice branch extracts `features = extract_all_features(y_slice_valid)` and immediately applies the scaler.
- **Impact:** Interaction slice predictions are made on a feature distribution that does not match training or the other inference branches. This can corrupt operator aggregation for formulas that only reveal their structure when two variables vary together.
- **Recommendation:** Centralize classifier feature preprocessing in one helper and call it from single-input, per-variable, diagonal interaction, and calibration paths.

- **Component:** `glassbox/evolution/evolution.py`
- **Severity:** Major
- **Type:** Bug
- **Description:** When `normalize_data=True` and the C++ backend succeeds, the trainer normalizes `x_cpu` and `y_cpu`, runs native evolution, recomputes display MSE against normalized arrays, and returns early. The later Python path denormalizes predictions and targets before final scoring, but that code is bypassed.
- **Impact:** Users enabling normalization can receive formulas, MSE values, and displayed results in normalized space. Backend availability changes the meaning of the returned result.
- **Recommendation:** Convert C++ formulas/results back to original coordinates before returning, or disable the C++ early return when normalization is enabled until a denormalization contract exists. Add a backend-parity test for `normalize_data=True`.

- **Component:** Legacy tree-classifier experiment and `glassbox/curve_classifier/curve_classifier_integration.py`
- **Severity:** Major
- **Type:** Bug
- **Description:** The removed legacy tree-classifier experiment trained models but saved incomplete serialized payloads. The same script standardized raw features without applying the SymLog transform used by classifier inference.
- **Impact:** Freshly trained legacy artifacts were either unloadable or evaluated on a mismatched feature distribution. This made the script unreliable for reproducing classifier artifacts.
- **Recommendation:** Save `models` in the payload and apply the same shared preprocessing helper used by inference before computing `mean` and `std`.

- **Component:** `scripts/train_universal_proposer.py`, `scripts/calibrate_classifier.py`, `glassbox/curve_classifier/train_curve_classifier.py`
- **Severity:** Minor
- **Type:** Bug
- **Description:** Validation split sizes are computed with `int(n * val_split)` and not bounded. Small datasets or `--val-split 0` can create empty validation datasets. Evaluation code then calls `torch.cat(all_preds)`, `torch.cat(all_logits)`, or equivalent on empty lists.
- **Impact:** Training/calibration runs can fail late with opaque runtime errors instead of rejecting invalid split settings up front.
- **Recommendation:** Validate that both train and validation sets have at least one sample before constructing loaders. For scripts that support zero-validation mode, skip evaluation explicitly rather than building an empty loader.

- **Component:** `glassbox/sr/optimizers/hybrid_optimizer.py`
- **Severity:** Minor
- **Type:** Logic Error
- **Description:** `EvolutionaryOptimizer.evolve_generation()` evaluates the old population, creates a new population through cloning, crossover, mutation, and optional L-BFGS refinement, assigns `self.population = new_population`, then reports stats from `ind.fitness` on the new individuals. Offspring fitness is not recomputed after mutation/refinement.
- **Impact:** Reported `best_fitness`, `mean_fitness`, and `worst_fitness` can be stale or infinite for the generation just produced. This degrades diagnostics and can mislead callers that rely on generation stats.
- **Recommendation:** Evaluate the new population before computing stats, or return stats for the evaluated pre-reproduction generation under explicit field names.

- **Component:** `glassbox/curve_classifier/curve_classifier_integration.py`, `scripts/classifier_fast_path.py`, `glassbox/evolution/evolution.py`
- **Severity:** Minor
- **Type:** Design Flaw
- **Description:** Standalone classifier and evolution defaults previously pointed to `models/curve_classifier_wide.pt`, while `GlassboxRegressor` defaulted to `models/curve_classifier_multi.pt`. The classifier fallback list did not include the current multi checkpoint.
- **Impact:** Standalone tools and direct module usage can silently run older classifier artifacts or fail even when the maintained multi checkpoint exists. This increases configuration drift across scripts.
- **Recommendation:** Replace duplicated constants with one shared model registry/default resolver. Include `curve_classifier_multi.pt` in fallback order and surface the resolved checkpoint in diagnostics.

- **Component:** `glassbox/curve_classifier/curve_classifier_integration.py`, `glassbox/universal_proposer/universal_proposer.py`, `scripts/calibrate_classifier.py`
- **Severity:** Major
- **Type:** Design Flaw
- **Description:** Model loaders called `torch.load(..., weights_only=False)` on caller-provided paths. The curve classifier CLI also evaluated `--formula` with Python `eval()` using globals that did not disable builtins.
- **Impact:** Loading or evaluating untrusted artifacts/formulas can execute arbitrary Python code. These tools are likely intended for trusted local workflows, but the trust boundary is not enforced by code or documented near the APIs.
- **Recommendation:** Treat model paths and CLI formulas as trusted-only by default and document that constraint. Prefer `weights_only=True` where checkpoint format allows, gate pickle fallback behind explicit trusted-path checks, and replace the demo CLI `eval()` with the existing restricted/symbolic formula evaluators.

## 4. Architectural & Design Criticisms
The largest architectural debt is split ownership of pipeline contracts. Feature preprocessing, model default resolution, candidate scoring, formula evaluation, and C++ seed construction are repeated across integration modules and scripts. Repetition has already produced observable drift: SymLog was missing in one classifier branch, a legacy classifier experiment did not match classifier inference, and standalone defaults pointed to different checkpoints than the sklearn wrapper.

The universal proposer needs a single honest multivariate contract. Today it exposes multivariate configuration, multivariate skeleton vocab, and multivariate search planning, but production routing often reduces multivariate data to a one-dimensional proxy. That proxy is useful as a fallback, but it should not be the only working path. A cleaner design is to make `propose_from_xy()` accept `X: [N, D]`, emit feature-aware candidate formulas, and use proxy mode only when a true multivariate checkpoint or decoder is unavailable.

The classifier preprocessing should be an explicit transformer object rather than inline slice mutation. A shared `CurveFeaturePreprocessor` could own SymLog ranges, scaler application, dimensional truncation/padding, and checkpoint metadata validation. Training, calibration, PyTorch inference, and multivariate slicing should all call that object.

The evolution layer should separate backend execution from result-space semantics. C++ and Python paths can differ in performance and search implementation, but they should return formulas, predictions, and MSE in the same coordinate system. A small `EvolutionResult` adapter should own normalization inversion, formula display, raw-vs-display MSE fields, and backend diagnostics.

Optimization scripts should be promoted from ad hoc utilities to tested artifact producers. Any script that writes a model should have a load-and-predict smoke test using the same integration loader that production uses. This would have caught incompatible serialized payloads and preprocessing mismatches immediately.

Finally, the project should make its trusted-artifact policy explicit. Research repositories often load local pickle checkpoints, but production-adjacent APIs should not make unsafe loading look like a general-purpose interface. Model loaders should distinguish trusted local artifacts from user-supplied paths, and CLI demos should avoid unrestricted formula evaluation.
