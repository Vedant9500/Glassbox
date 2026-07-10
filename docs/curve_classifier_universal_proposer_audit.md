# Curve Classifier and Universal Proposer Audit

Date: 2026-06-07

Scope:

- `glassbox/curve_classifier/`
- `glassbox/universal_proposer/`
- `scripts/train_universal_proposer.py`
- Active local artifacts: `models/curve_classifier_multi.pt`, `models/universal_proposer_multi.pt`, `data/curve_dataset_multi.npz`

This document records the current issues, the research context that supports or challenges the diagnosis, and a phase-wise fix plan.

Revalidation note: this file was updated after a deeper pass over the current code, downstream consumers, and active dataset. The goal of this pass was to separate actual defects from intentional heuristics and false-positive signals.

## Executive Verdict

The current curve classifier and universal proposer are usable as weak warm-start and routing hints, but they are not yet a robust learned model for arbitrary symbolic-regression or tabular conditions. The main failure mode is not PyTorch capacity. It is a mismatch between parts of the training data contract and the runtime problem:

- Training historically used ordered synthetic curves while inference could receive row-ordered datasets. Phase 0/1 fixed the univariate public inference path; multivariate still uses heuristic projections.
- Labels are not derived consistently from the final formula in a small but definite set of exact duplicate formulas. The earlier raw AST mismatch rate is an overbroad warning because the taxonomy intentionally treats some syntax as non-semantic.
- Multivariate neural features are intentionally heuristic today: they compress `y` to one-dimensional features while runtime behavior tries interpolation and interaction heuristics.
- Validation uses random row splits over a heavily duplicated formula corpus. This is acceptable for smoke tests, but optimistic for formula-level generalization.
- The proposer mostly acts as a heuristic grammar/MSE planner with learned operator priors. Its skeleton head is weakly supervised and should not be treated as a validated expression generator without extra metrics.

## Local Evidence Summary

Checkpoint metadata:

- `models/curve_classifier_multi.pt`: `model_type=glu`, `n_features=398`, `n_classes=9`, `val_f1=0.7550`, `val_acc=0.8687`, tuned thresholds and scaler present.
- `models/universal_proposer_multi.pt`: `hidden_dim=512`, `n_features=398`, `supports_multivariate_formulas=True`, `val_f1=0.7650`, scaler present, skeleton vocab length 27.

Dataset probe on `data/curve_dataset_multi.npz`:

- Shape: `(500000, 398)` features, `(500000, 9)` labels.
- Metadata: `n_input_features=3`, `multivariate_ratio=0.5`.
- Only `183445` unique formula strings out of `500000` rows.
- Duplicate rows by formula string: `320907`, or `64.18%`.
- Top duplicate: formula `x`, appearing `30342` times.
- In a random 50k sample, raw AST-derived labels disagreed with stored labels in `39.662%` of rows.
- After merging `identity` and `power` into the downstream `polynomial` family, disagreement dropped to `23.294%`.
- Exact duplicate-formula conflict probe found `12` formula strings with multiple stored label vectors, affecting `16,141` rows (`3.23%` of the dataset).
- Full-dataset proposer skeleton-vocab coverage is `82,819 / 500,000 = 16.56%`.
- A simulated random 80/20 split put only `5.28%` of unique validation formulas in train, but `63.64%` of validation rows had a formula string already seen in train.

Historical runtime order probe before Phase 0/1:

- For the same `sin(x)` curve, sorted by `x` vs randomly shuffled rows:
  - Classifier `sin` probability changed by about `0.6729`.
  - Classifier introduced high `power` probability on the shuffled row order.
  - Proposer shifted from sine-first candidates toward power/cosine-polynomial candidates.

Current univariate status after Phase 0/1:

- `predict_operators()` now calls `extract_all_features_xy(x, y)` for univariate input.
- `propose_from_xy()` now calls `extract_all_features_xy(x, y)` for univariate input.
- `extract_all_features_xy()` sorts by `x`, averages duplicate `x` values, and resamples to a canonical grid before feature extraction.
- Regression tests cover shuffled rows, duplicate-row expansion, nonuniform sampling, classifier inference, and proposer inference.

## Validation Pass: Confirmed, Intentional, And False-Positive Checks

| Finding | Revalidated status | Current classification |
| --- | --- | --- |
| Row order as signal | Confirmed historically. Phase 0/1 fixed the univariate public inference path. Multivariate remains projection-based and heuristic. | Resolved for univariate inference; remaining multivariate limitation. |
| Label generation inconsistency | Confirmed by exact duplicate formulas with conflicting labels. Raw AST mismatch overstates the bug because the taxonomy intentionally ignores some wrapper and argument syntax. | Real data bug, but narrower than raw AST mismatch. |
| Multivariate generation | Confirmed as a heuristic design, not an accidental implementation bug. The model advertises multivariate support in configs and payloads, while features are still one-dimensional `y` summaries. | Intentional heuristic that needs clearer contract and separate metrics. |
| Optimistic validation | Confirmed for formula-level generalization. Heavy duplicates make random-row validation optimistic, although duplicates can be intentional augmentation. | Evaluation limitation, not necessarily bad data generation. |
| Universal proposer skeleton supervision | Confirmed. Skeleton loss is skipped for non-vocab formulas and coverage is only `16.56%`; however grammar decoding and MSE ranking are the main candidate path. | Contract/calibration issue, not a total proposer failure. |
| Train/inference architecture drift | Confirmed historically. Phase 0 moved classifier definitions into a shared model module and added metadata validators. | Resolved for current code; keep regression tests. |

False-positive corrections from the deeper pass:

- Do not treat every `x` inside `sin(x)`, `cos(x)`, `tanh(x)`, or similar unary arguments as a required `identity` or `power` label. The local taxonomy has an intentional semantic-vs-syntactic distinction.
- Do not count safety wrappers such as `abs`, epsilon shifts, clipping, or denominator guards as semantic operators unless the template explicitly makes them part of the family being trained.
- Do not interpret `identity` and `power` disagreement as equally severe downstream. Fast path code derives a `polynomial` family from either one, and evolution maps both into polynomial/power-style operators.
- Do not treat duplicate formulas alone as leakage. Repeated formulas with different sampled ranges, noise, constants, or augmentations can be intentional. The bug is using random-row validation as the release metric without also reporting grouped formula/template splits.

## Finding 1: Row Order Is Treated As Signal

### Local Finding

Training generation evaluates formulas on ordered `linspace` grids and still extracts features from `y_aug` only:

- `glassbox/curve_classifier/generate_curve_data.py`: `extract_all_features(y)` takes no `x`.
- `generate_chunk()` samples ordered grids, evaluates `y`, then calls `extract_all_features(y_aug)`.
- Current `predict_operators()` uses `extract_all_features_xy(x, y)` for univariate inference.
- Current `propose_from_xy()` uses `extract_all_features_xy(x, y)` for univariate inference.
- Multivariate proposer inference still falls back to `extract_all_features(y)` for neural features, while grammar candidate generation uses the original multivariate `X`.

Historical risk: row order became an implicit coordinate system. If a user passed tabular rows in arbitrary order, the models saw a random walk through `y`, not the sampled function. The univariate public inference path is now guarded by canonical sorting/resampling; the remaining concern is multivariate, where the current model is intentionally a heuristic projection path rather than a true set model.

### Research Context

- Deep Sets formalizes set-input models where outputs should not depend on input element order and gives a family of permutation-invariant architectures for set data: [Deep Sets, Zaheer et al. 2017](https://arxiv.org/abs/1703.06114).
- Set Transformer extends that idea with attention for interactions among unordered set elements: [Set Transformer, Lee et al. 2019](https://arxiv.org/abs/1810.00825).
- End-to-end symbolic regression Transformer work explicitly treats input points as unordered enough to remove encoder positional embeddings for input-point permutation invariance: [Kamienny et al. 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/42eb37cdbefd7abae0835f4b67548c39-Abstract-Conference.html), [paper PDF](https://papers.nips.cc/paper_files/paper/2022/file/42eb37cdbefd7abae0835f4b67548c39-Paper-Conference.pdf).

### Conclusion

The intuition was confirmed for the old univariate path and is now addressed by Phase 0/1. A symbolic-regression helper model should either:

- sort/resample univariate `(x, y)` into a canonical curve before feature extraction, or
- use a point-set architecture over `(x, y)` pairs.

Current behavior now does the first option for univariate inference. It still does not do the second option for multivariate inference, so multivariate predictions must remain documented as heuristic.

## Finding 2: Label Generation Is Inconsistent

### Local Finding

Historically, PCFG formulas derived operators from the final AST while template and multivariate formulas mostly trusted manually attached operator sets. Before the semantic-labeler fix, `operators_to_labels()` only derived labels from the formula AST when the provided operator set was empty. Current code now uses semantic labels by default whenever a formula is available, while still preserving explicit syntax/operator metadata for auditability. The original issue created inconsistent supervision in some cases, but a raw AST comparison overstated the problem because the intended taxonomy is partly semantic:

- The exact same formula string can have different label vectors. Examples in the active dataset include `np.sin(x)`, `np.cos(x)`, `x ** 2`, `np.tanh(x)`, and `np.sqrt(np.abs(x) + 0.01)`.
- The difference is usually an extra or missing `identity` bit, or a wrapper-related `addition` bit.
- Some apparent mismatches are intentional taxonomy choices: `sin(x)` need not train the `identity` class just because `x` appears as an argument, and `abs`/epsilon guards need not train a semantic operator family.

Measured result: raw AST-derived labels disagreed with stored labels in `39.662%` of a 50k sample, but that is not the real bug rate. The stronger duplicate-formula probe found `12` exact formula strings with conflicting stored labels, affecting `16,141` rows (`3.23%` of the active dataset).

### Research Context

- Multi-label classification expects a consistent binary indicator contract for target labels. Scikit-learn documents the indicator-matrix view where each column corresponds to one independently predicted label: [OneVsRestClassifier multi-label docs](https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html).
- Label-noise literature treats wrong labels as a direct generalization risk for deep networks. A survey summarizes that noisy labels degrade neural-network generalization and motivate robust training or label correction: [Learning from Noisy Labels with Deep Neural Networks, Song et al. 2020](https://arxiv.org/abs/2007.08199).

### Conclusion

The intuition is partially confirmed. The model is trained on a target taxonomy that changes by generator path for exact duplicate formulas, so the label contract needs fixing before retraining. The false-positive correction is that raw AST labels should not be used as the sole acceptance gate.

Important nuance: the AST labeler should not blindly label safety wrappers as semantic operators. For example, `np.abs` added to make `sqrt` safe should not create a new symbolic family. The right fix is a canonical semantic labeler with explicit wrapper filtering, not raw AST labels alone.

## Finding 3: Multivariate Generation Is Not Truly Multivariate

### Local Finding

`_sample_multivariate_x()` builds each input column as a rolled `linspace`. This generates a one-dimensional path through input space rather than independent samples or a grid/surface. Then the model extracts features from only `y`.

At runtime, classifier multi-input prediction builds interpolators, takes per-variable slices, detects pair interactions with a nearest-neighbor proxy, and max-aggregates probabilities. This is useful as a heuristic, but it is not the distribution the neural model was trained on.

### Research Context

- Neural symbolic regression literature frames the input as a set of input-output pairs, not a one-dimensional response trace. Biggio et al. pre-train from sets of sampled inputs and outputs and use the model to guide equation search: [Neural Symbolic Regression that Scales, Biggio et al. 2021](https://arxiv.org/abs/2106.06427), [PMLR PDF](https://proceedings.mlr.press/v139/biggio21a/biggio21a.pdf).
- Partial dependence computes model response over a target feature or feature pair while averaging or otherwise handling complement features. Scikit-learn supports pairs of interacting features in its partial dependence API: [scikit-learn partial_dependence](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.partial_dependence.html).
- Set Transformer is specifically intended to model interactions among set elements, which is closer to multivariate `(x_i, y_i)` input sets than fixed one-dimensional traces: [Set Transformer](https://arxiv.org/abs/1810.00825).

### Conclusion

The intuition is confirmed as a model-contract issue, not as an accidental bug. The current multivariate path should be treated as heuristic routing, not a trained multivariate neural predictor. A real multivariate neural model needs `(X, y)` point-set features or surface features, plus training data sampled from the same distribution expected at runtime.

## Finding 4: Validation Is Likely Optimistic

### Local Finding

The active dataset has heavy duplicate formulas. Training scripts use random row splits:

- Classifier split: random or multilabel-stratified by rows.
- Proposer split: `np.random.shuffle(indices)` by rows.

With duplicate formulas present in both train and validation, validation metrics likely measure interpolation over seen equation families more than generalization to held-out formulas or held-out generator families. In a simulated random 80/20 split, `63.64%` of validation rows had a formula string already seen in train.

### Research Context

- Scikit-learn warns that train/test leakage leads to overly optimistic scores and recommends splitting before preprocessing decisions: [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html).
- GroupKFold is an official split method for keeping groups non-overlapping across folds. That maps directly to formula keys, template ids, or generator-family ids: [scikit-learn GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html).
- SRBench-style work emphasizes diverse benchmark datasets and reproducible evaluation for symbolic regression: [Contemporary Symbolic Regression Methods and their Relative Performance](https://pmc.ncbi.nlm.nih.gov/articles/PMC11074949/) and [SRBench++](https://pmc.ncbi.nlm.nih.gov/articles/PMC12321164/).

### Conclusion

The intuition is confirmed, with a false-positive correction: duplicate formulas are not inherently a data-generation bug because repeated formulas can carry different ranges, noise, constants, or augmentations. The issue is using random-row validation as the release metric. Current validation numbers are useful for smoke testing, but not for claiming formula-level generalization. Add grouped and out-of-distribution validation gates.

## Finding 5: Universal Proposer Is Under-Supervised As A Skeleton Model

### Local Finding

`FormulaReplayDataset._formula_to_skeleton_target()` returns `-1` when a formula is not in the fixed skeleton vocab. The skeleton loss is skipped for invalid targets. In the active dataset, only `82,819 / 500,000 = 16.56%` of rows match the fixed skeleton vocab.

Runtime candidate generation mostly uses grammar candidates scored by operator priors plus affine MSE. Skeleton logits are primarily used as a fallback and for uncertainty.

### Research Context

- Kamienny et al. describe skeleton prediction plus constant fitting as the common two-step SR neural approach, then show benefits from end-to-end expression prediction with later refinement: [End-to-end Symbolic Regression with Transformers](https://proceedings.neurips.cc/paper_files/paper/2022/hash/42eb37cdbefd7abae0835f4b67548c39-Abstract-Conference.html).
- Biggio et al. train a set-to-sequence model from input-output pairs and use beam search plus constant optimization at test time, which is a clearer learned proposer contract than a weak skeleton head used mainly for uncertainty: [Neural Symbolic Regression that Scales](https://proceedings.mlr.press/v139/biggio21a/biggio21a.pdf).
- Neural-guided GP work reports that decoupling neural guidance from many GP generations can work better than overly tight coupling, which supports using weak neural priors cautiously: [Petersen et al. 2021](https://proceedings.neurips.cc/paper_files/paper/2021/hash/d073bb8d0c47f317dd39de9c9f004e9d-Abstract.html).

### Conclusion

The intuition is confirmed as a contract/calibration issue. The current proposer should be described as a heuristic planner with learned operator priors, grammar decoding, and MSE ranking. If the goal is a real learned proposer, it needs much broader skeleton/expression supervision and metrics that measure skeleton top-k accuracy, not only operator F1.

## Finding 6: Train And Inference Architectures Diverge

### Local Finding

Historical issue: the curve classifier architecture was duplicated in training and integration files. The `EQLLayer` exp clamp differed:

- Training: `torch.exp(torch.clamp(chunk, min=-10.0, max=10.0))`
- Inference copy: `torch.exp(torch.clamp(chunk, min=-5.0, max=5.0))`

Before Phase 0, checkpoint weights were loaded into a model that did not exactly match the training forward pass. Current code imports the shared classifier classes from `glassbox/curve_classifier/models.py`, and metadata validators check architecture and feature contracts.

### Research Context

- Scikit-learn's leakage guidance is broader than train/test splits: preprocessing and transforms must be learned and applied consistently between train and test paths: [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html).
- Calibration research shows modern networks are sensitive to architecture and training details, including depth, width, weight decay, and BatchNorm. That supports treating train/inference forward-pass mismatch as a real risk, not a harmless implementation detail: [Guo et al. 2017](https://proceedings.mlr.press/v70/guo17a).

### Conclusion

The intuition was confirmed and Phase 0 addressed it. Keep the shared model module and checkpoint metadata tests as regression protection.

## Phase-Wise Fix Plan

### Phase 0: Freeze Contract And Add Regression Tests

Goal: prevent more silent drift before retraining.

Tasks:

- Define a formal input contract:
  - univariate: accepts `(x, y)`, sorts by `x`, aggregates duplicate `x`, resamples to canonical grid;
  - multivariate: either disabled for neural classifier/proposer or handled by a point-set path.
- Add tests proving row-order invariance for univariate classifier/proposer feature preparation.
- Add tests proving train and inference `EQLLayer` outputs match for the same weights and inputs.
- Add checkpoint metadata validation: `model_type`, `feature_dim`, `feature_schema`, `architecture_version`.

Acceptance gates:

- `sin(x)` predictions differ by less than `0.03` after random row permutation.
- Shared model class is imported by both training and inference.
- Existing public APIs still load current checkpoints, with an explicit compatibility path if needed.

Status as of 2026-06-07: completed.

Implemented:

- Added shared classifier architecture module `glassbox/curve_classifier/models.py`.
- Bound training and inference classifier names to the shared classes.
- Added canonical univariate `(x, y)` preparation:
  - drop non-finite rows,
  - sort by `x`,
  - average duplicate `x` targets,
  - resample to a fixed canonical grid.
- Switched classifier and proposer univariate inference to the canonical `(x, y)` feature path.
- Added classifier and proposer checkpoint metadata validators with legacy-unversioned compatibility.
- Added architecture-version metadata for newly saved classifier and proposer checkpoints.
- Added Phase 0 regression tests in `tests/test_curve_classifier_phase0.py`.
- Added a serial path for `generate_dataset(..., n_workers=1)` so local generation tests do not require multiprocessing pipes.

Verification:

- `python -m pytest tests\test_curve_classifier_phase0.py -q`
- `python -m pytest tests\test_pcfg_generator.py -q -p no:cacheprovider`
- `python -m pytest tests\test_universal_proposer.py tests\test_feature_extraction.py -q -p no:cacheprovider -k "not formula_replay_dataset"`

Note: the full `tests/test_universal_proposer.py tests/test_feature_extraction.py` run still hits a Windows temp-directory permission error in pytest's `tmp_path` fixture setup for the two formula replay dataset tests. The same neighboring test set passes when those two temp-fixture cases are deselected.

### Phase 1: Canonicalize Feature Extraction

Goal: remove row-order leakage for univariate data.

Tasks:

- Add `prepare_univariate_curve_xy(x, y, n_points=256)` in `generate_curve_data.py` or a new shared preprocessing module.
- Make `extract_all_features_xy(x, y)` the default for inference.
- Keep `extract_all_features(y)` as a compatibility wrapper for generated already-ordered curves.
- Update:
  - `curve_classifier_integration.predict_operators()`
  - `universal_proposer.propose_from_xy()`
  - `sklearn_wrapper._run_universal_proposer_dual_path()`
  - benchmark/proposer call sites that pass raw rows.

Acceptance gates:

- Sorted, shuffled, and duplicate-`x` versions of the same clean curve produce close feature vectors.
- Feature extraction remains stable for nonuniform x spacing.
- No regression in current univariate smoke tests.

Status as of 2026-06-07: completed.

Implemented:

- Exported `prepare_univariate_curve_xy` and `extract_all_features_xy` from `glassbox.curve_classifier`.
- Added Phase 1 tests for:
  - sorting and duplicate-`x` averaging,
  - duplicate-row expansion invariance,
  - nonuniform sampling interpolation stability,
  - public `predict_operators()` row-order invariance,
  - public `propose_fpip_v2_from_xy()` row-order invariance.
- Confirmed existing classifier/proposer inference paths now call the canonical univariate `(x, y)` feature path.

Verification:

- `python -m pytest tests\test_curve_classifier_phase1.py -q -p no:cacheprovider`

### Phase 2: Fix Label Semantics And Regenerate Data

Goal: train on one target taxonomy.

Tasks:

- Replace template-supplied labels with a canonical semantic labeler run on the final formula string.
- Add wrapper filtering:
  - safety `abs`, `clip`, epsilon shifts, and denominator guards should not become semantic discoveries unless explicitly part of a template.
- Store both:
  - `semantic_operators`
  - `syntax_operators`
  for auditability, but train on `semantic_operators`.
- Store `template_id`, `generator_family`, `formula_key`, and `labeler_version` in the dataset.
- Regenerate a small validation dataset first, then the full corpus.

Acceptance gates:

- Stored labels match canonical semantic labels for at least `99.5%` of generated rows.
- Mismatches are explainable allowlisted cases.
- Dataset report includes per-class frequencies, duplicate rates, and generator-family counts.

Status as of 2026-06-07: implementation completed; full production corpus regeneration pending.

Implemented:

- Added `SEMANTIC_LABELER_VERSION = "semantic-labeler-v1"`.
- Added `derive_semantic_operators_from_formula()` as the training-label contract.
- Kept `derive_operators_from_formula()` as the syntax/audit labeler.
- Changed `operators_to_labels(..., formula=...)` to use semantic labels by default, with explicit `label_mode="syntax"` and `label_mode="provided"` escape hatches.
- Filtered domain/safety wrappers from semantic labels:
  - `sin(x)` trains `sin`, not `identity + sin`;
  - `sqrt(abs(x) + eps)` trains `power`, not wrapper `addition` or `identity`;
  - `(abs(x) + eps) ** negative_power` trains `power + rational`, not epsilon `addition`.
- Added formula audit metadata:
  - `semantic_operators`,
  - `syntax_operators`,
  - `provided_operators`,
  - `formula_keys`,
  - `generator_families`,
  - `template_ids`,
  - `semantic_labels`,
  - `labels_match_semantic`,
  - `labeler_version`.
- Added optional generation metadata return from `generate_dataset(..., return_metadata=True)` without changing the default public 3-tuple return.
- Updated `save_dataset()` to embed Phase 2 audit metadata in `.npz` datasets.
- Generated a small validation dataset at `scratch/phase2_audit_dataset.npz`; all rows reported `labels_match_semantic=True`.

Verification:

- `python -m pytest tests\test_curve_classifier_phase2.py -q -p no:cacheprovider --basetemp D:\Glassbox\scratch\pytest-basetemp\phase2`
- `python -m pytest tests\test_curve_classifier_phase2.py tests\test_pcfg_generator.py -q -p no:cacheprovider --basetemp D:\Glassbox\scratch\pytest-basetemp\phase2-rerun`
- `python -m pytest tests\test_pcfg_generator.py tests\test_curve_classifier_phase0.py tests\test_curve_classifier_phase1.py -q -p no:cacheprovider --basetemp D:\Glassbox\scratch\pytest-basetemp\phase2-regression`
- `python -m pytest tests\test_universal_proposer.py tests\test_feature_extraction.py -q -p no:cacheprovider -k "not formula_replay_dataset" --basetemp D:\Glassbox\scratch\pytest-basetemp\phase2-universal`

### Phase 3: Replace Random Row Validation With Grouped Evaluation

Goal: measure generalization rather than memorization.

Tasks:

- Add formula-key grouped splits using `GroupKFold` or an equivalent local splitter.
- Add heldout-template-family validation:
  - hold out some templates entirely;
  - hold out PCFG depth bands;
  - hold out x-range/noise combinations.
- Report:
  - random-row validation,
  - formula-group validation,
  - family-heldout validation,
  - row-permutation stress validation.
- For proposer, report:
  - operator micro/macro F1,
  - skeleton top-1/top-5 accuracy on rows with valid skeleton targets,
  - grammar candidate recall after affine fit.

Acceptance gates:

- Validation report is saved next to every checkpoint.
- Release criteria use grouped/family-heldout metrics, not random-row metrics alone.
- Proposer uncertainty is calibrated against actual candidate success, not only skeleton logits.

Status as of 2026-06-07: implementation completed for feature-dataset training; raw-curve affine candidate recall remains explicitly uncomputed until raw `(x, y)` validation corpora are available.

Implemented:

- Added `glassbox/curve_classifier/validation.py` with:
  - formula-key grouped train/validation splitting,
  - generator-family holdout splitting,
  - row split fallback,
  - formula overlap diagnostics,
  - label/family/template distribution reports,
  - JSON-safe validation report writing.
- Updated classifier training:
  - `--split-policy auto|row|stratified|formula_group|family_holdout`;
  - `--heldout-family`;
  - `--validation-report`;
  - `auto` uses formula-group validation when `formula_keys` metadata is present;
  - checkpoints record split policy/details and validation report path;
  - `.validation.json` is saved next to the checkpoint.
- Updated proposer training:
  - `--split-policy auto|row|formula_group|family_holdout`;
  - `--heldout-family`;
  - `--validation-report`;
  - formula-group validation is used when metadata is present;
  - validation report is saved next to the checkpoint.
- Added proposer validation metrics:
  - operator macro F1,
  - operator micro F1,
  - per-operator F1,
  - skeleton target coverage,
  - skeleton top-1 accuracy,
  - skeleton top-5 accuracy,
  - skeleton confidence mean,
  - skeleton ECE over 10 bins.
- Added a report note for `candidate_recall_after_affine_fit`: it is not computable from current precomputed feature-only datasets because raw `(x, y)` curves are not stored.
- Verified tiny smoke checkpoints write Phase 3 reports:
  - `scratch/phase3_classifier_smoke.validation.json`;
  - `scratch/phase3_proposer_smoke.validation.json`;
  both used `formula_group` splits with zero formula overlap.

Verification:

- `python -m pytest tests\test_curve_classifier_phase3.py -q -p no:cacheprovider --basetemp D:\Glassbox\scratch\pytest-basetemp\phase3-rerun`
- `python -m pytest tests\test_curve_classifier_phase0.py tests\test_curve_classifier_phase1.py tests\test_curve_classifier_phase2.py tests\test_curve_classifier_phase3.py tests\test_pcfg_generator.py -q -p no:cacheprovider --basetemp D:\Glassbox\scratch\pytest-basetemp\phase3-combined`
- `python -m pytest tests\test_universal_proposer.py tests\test_feature_extraction.py -q -p no:cacheprovider -k "not formula_replay_dataset" --basetemp D:\Glassbox\scratch\pytest-basetemp\phase3-universal`
- `python -m glassbox.curve_classifier.train_curve_classifier --data scratch\phase2_audit_dataset.npz --epochs 1 --patience 1 --batch-size 16 --device cpu --output scratch\phase3_classifier_smoke.pt --split-policy formula_group --no-tune-thresholds --no-stratified-split`
- `python scripts\train_universal_proposer.py --data scratch\phase2_audit_dataset.npz --epochs 1 --batch-size 16 --patience 1 --device cpu --out scratch\phase3_proposer_smoke.pt --split-policy formula_group`

### Phase 4: Decide The Multivariate Strategy

Goal: stop mixing univariate traces with multivariate claims.

Option A: Conservative short-term path.

- Keep the neural classifier/proposer univariate.
- For multivariate `X`, route through blackbox preprocessing, interaction detection, fast path, and evolution.
- Let neural outputs act only as weak operator priors from safe 1D projections.

Option B: Real multivariate neural path.

- Train on sets of `(x0, ..., xd, y)` points sampled independently from the target domain.
- Use a Deep Sets, Set Transformer, or point-set encoder.
- Preserve variable identity through embeddings.
- Train variable-interaction heads separately from operator heads.
- Evaluate on heldout dimensions, heldout interaction forms, and row permutations.

Acceptance gates:

- If Option A: public status must say neural multivariate mode is heuristic.
- If Option B: model predictions must be invariant to row order and robust to independent resampling of input points.

Status as of 2026-06-07: Option A selected and implemented.

Implemented:

- Added `describe_curve_classifier_inference(x)` so classifier callers can distinguish:
  - `trained_univariate_neural` for canonical univariate `(x, y)` feature extraction;
  - `heuristic_multivariate` for per-variable/interactions slice aggregation.
- Kept `predict_operators()` return values as pure operator probabilities to avoid breaking downstream fast-path consumers.
- Added universal proposer contract metadata:
  - `model_contract`,
  - `neural_feature_mode`,
  - `neural_multivariate_support`,
  - `supports_trained_multivariate_neural_model`.
- Updated multivariate proposer `search_plan` and FPIP v2 payloads to report that multivariate neural priors are heuristic y-projection features while candidates/search use the original multivariate `X`.
- Added `multivariate_neural_mode` to newly saved proposer checkpoint configs.
- Updated sklearn wrapper status from `ok_multivariate` to `ok_multivariate_heuristic`.

Verification:

- `python -m pytest tests\test_curve_classifier_phase4.py -q -p no:cacheprovider`

### Phase 5: Rework The Universal Proposer Contract

Goal: make the proposer either honestly heuristic or genuinely learned.

Short-term:

- Rename docs/status to describe it as learned operator priors plus grammar/MSE planner.
- Stop using skeleton logits as a strong confidence signal unless skeleton metrics support it.
- Calibrate proposer routing against downstream success.

Long-term:

- Expand skeleton/expression vocabulary with canonicalization and grammar production labels.
- Train sequence targets for generated formulas, not only exact fixed-vocab skeleton ids.
- Add beam search over grammar productions.
- Optionally predict constants or constant initialization ranges.

Acceptance gates:

- Skeleton target coverage above `80%` for the chosen training corpus, or sequence targets replacing fixed-vocab skeleton ids.
- Top-k skeleton or grammar recall reported on heldout formulas.
- Downstream benchmark shows proposer improves search budget or success rate on grouped/family-heldout tests.

Status as of 2026-06-07: short-term contract implementation completed; full learned sequence proposer remains future work.

Implemented:

- Defined the proposer runtime role as `learned_operator_priors_plus_grammar_mse_planner`.
- Added proposer contract metadata to runtime output, search plans, FPIP v2 payloads, and new checkpoint configs.
- Kept skeleton logits as diagnostics by default:
  - raw entropy/margin remain available as `raw_entropy` and `raw_margin`;
  - routed `entropy`, `margin`, and `confident` are disabled unless checkpoint validation metrics satisfy coverage/top-k gates.
- Added a skeleton-confidence reliability gate:
  - coverage must be at least `0.80`;
  - top-1 accuracy must be at least `0.60`;
  - top-5 accuracy must be at least `0.80`.
- Changed FPIP routing to use:
  - verified grammar-candidate relative MSE first;
  - validation-gated skeleton confidence second;
  - guided evolution otherwise.
- Added routing calibration status fields so uncalibrated checkpoints report that downstream candidate-success benchmarking is still required.

Verification:

- `python -m pytest tests\test_curve_classifier_phase5.py -q -p no:cacheprovider`

### Phase 6: Recalibrate And Roll Out

Goal: safely ship new checkpoints.

Tasks:

- Retrain classifier and proposer on regenerated datasets.
- Calibrate probabilities and thresholds on grouped validation data.
- Save checkpoint cards with:
  - data generation command,
  - labeler version,
  - feature schema,
  - split policy,
  - random-row and grouped metrics,
  - known unsupported cases.
- Add benchmark comparison:
  - current `curve_classifier_multi.pt` / `universal_proposer_multi.pt`,
  - fixed-label univariate model,
  - optional multivariate point-set model.

Acceptance gates:

- New model beats current checkpoint on grouped/family-heldout metrics.
- Row-order stress tests pass.
- Runtime fallback remains safe when checkpoint is missing or incompatible.

Status as of 2026-06-07: rollout infrastructure completed; full production retraining and benchmark execution remain operational runs.

Implemented:

- Added Phase 6 checkpoint-card helpers with schema `checkpoint_card.phase6.v1`.
- Added rollout comparison helpers with schema `rollout_comparison.phase6.v1`.
- Classifier and proposer training now write checkpoint cards next to saved checkpoints by default.
- Checkpoint cards include:
  - data-generation command,
  - training command,
  - labeler version,
  - feature schema/dimension,
  - validation split policy/details,
  - best validation metrics,
  - calibration/routing-calibration status,
  - row-order stress status,
  - runtime fallback status,
  - known unsupported cases.
- Optional `--baseline-card` writes a rollout comparison report that blocks release unless:
  - grouped/family-heldout validation is present,
  - row-order stress is marked passing,
  - runtime fallback is marked passing,
  - candidate metric beats the baseline metric.
- Added CLI switches:
  - `--checkpoint-card`;
  - `--data-generation-command`;
  - `--baseline-card`;
  - `--rollout-comparison`;
  - `--rollout-metric`;
  - `--min-relative-improvement`.

Verification:

- `python -m pytest tests\test_curve_classifier_phase6.py -q -p no:cacheprovider`

## Recommended Next Implementation Order

With Phases 0-6 implemented, the remaining work is production execution and optional model-scope upgrades:

1. Regenerate the full production corpus with semantic labels and embedded validation metadata.
2. Store a dataset report covering duplicate-label conflicts, per-class frequencies, duplicate rates, generator-family counts, and skeleton coverage.
3. Retrain the classifier and proposer with grouped/family-heldout validation as release metrics, saving checkpoint cards.
4. Compare the new checkpoint cards against `curve_classifier_multi.pt` and `universal_proposer_multi.pt` baseline cards.
5. Run downstream candidate-success benchmarks to replace the current conservative proposer routing calibration placeholder.
6. Revisit a true multivariate point-set model only if grouped/family-heldout benchmarks show the heuristic path is insufficient.

## Source Links

- Deep Sets: https://arxiv.org/abs/1703.06114
- Set Transformer: https://arxiv.org/abs/1810.00825
- End-to-end Symbolic Regression with Transformers: https://proceedings.neurips.cc/paper_files/paper/2022/hash/42eb37cdbefd7abae0835f4b67548c39-Abstract-Conference.html
- End-to-end Symbolic Regression with Transformers PDF: https://papers.nips.cc/paper_files/paper/2022/file/42eb37cdbefd7abae0835f4b67548c39-Paper-Conference.pdf
- Neural Symbolic Regression that Scales: https://arxiv.org/abs/2106.06427
- Neural Symbolic Regression that Scales PDF: https://proceedings.mlr.press/v139/biggio21a/biggio21a.pdf
- Scikit-learn common pitfalls: https://scikit-learn.org/stable/common_pitfalls.html
- Scikit-learn GroupKFold: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html
- Scikit-learn partial dependence: https://scikit-learn.org/stable/modules/generated/sklearn.inspection.partial_dependence.html
- Scikit-learn OneVsRestClassifier multi-label docs: https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html
- Learning from Noisy Labels with Deep Neural Networks: https://arxiv.org/abs/2007.08199
- On Calibration of Modern Neural Networks: https://proceedings.mlr.press/v70/guo17a
- Neural-guided GP seeding for symbolic regression: https://proceedings.neurips.cc/paper_files/paper/2021/hash/d073bb8d0c47f317dd39de9c9f004e9d-Abstract.html
- Contemporary Symbolic Regression Methods and their Relative Performance: https://pmc.ncbi.nlm.nih.gov/articles/PMC11074949/
- SRBench++: https://pmc.ncbi.nlm.nih.gov/articles/PMC12321164/
