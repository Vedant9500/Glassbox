# Handover: Blackbox × Noise Handling

**Branch:** `noise-handling`  
**Date:** 2026-07-11  
**Plan:** `blackbox_noise_plan.md`  
**Prior noise work:** Phases 0–8 (weights, robust loss, units, cleanup, routing, ablations)

---

## Objective

Close Glassbox’s multi-feature **blackbox** gap under noise: make recovery
measurable, stop ranking from dropping true features, and gate robust search
without turning Track 1 into a global robust-loss default.

**Success metric:** clean recovery (`R2clean` / `CleanMSE` / protocol Exact),  
not suite noisy EXACT% or Track-1 noisy-label R² alone.

---

## What shipped this session

### Phase A — Measure multi-var blackbox under noise
- `DEFAULT_BLACKBOX_PROBLEMS`: Pagie-1, Feynman-I.9.18, Vladislavleva-4
- CLI: `--blackbox`, `--smoke --blackbox`
- Protocol row fields: `n_features`, `blackbox_enabled`, `blackbox_reason`,
  `selected_features`, `n_selected_features`, `feature_selection_uncertain`,
  `ranking_sample_weight_mode`, `noise_band`, `noise_pressure`
- Factory under `--blackbox`: `blackbox_min_features_to_select=2`,
  `blackbox_max_features=4` so ranking can actually drop features

### Phase B — Weight-aware ranking
- `sample_weight` through corr / poly / holdout poly / Lasso / ElasticNet /
  ExtraTrees / interactions; MI via weighted resampling
- `fit` passes `sample_weight_` into `prepare_blackbox_search`

### Ranking stability
- Keep all when `n_usable ≤ max_features`
- Near-tie plateau extend only for **strong** scores (no noise-floor cascade)
- Symmetric multi-var (Vlad-like): keep all when scores informative + comparable
- Reasons: `retained_all_features_score_plateau`,
  `selected_top_features_plateau_extended`, `retained_all_features_within_budget`,
  plus existing uncertain/small-problem reasons

### Phase C — Gated robust blackbox search
- Constructor: `blackbox_noise_robust="auto"|True|False`
- **auto:** soft MAD weights from `y` when blackbox multi-feature, no user
  weights, and heavy tails / uncertain selection; re-ranks with weights;
  may switch default `mse → huber`
- Never overrides user-supplied `sample_weight`
- Plan caps relax when medium/high noise **or** selection uncertain **or**
  high noise_pressure (timeout max **1.45** vs hard **1.0**)
- After remap: original-space holdout re-score → diag `original_space_holdout`

### Docs / tests
- Plan: `blackbox_noise_plan.md` (literature do/avoid + corrected phases)
- Tests: preprocessor ranking, protocol columns, soft MAD, plan timeout relax
- Related untracked from earlier phases (include if committing full branch):
  `scripts/calibrate_noise_routing.py`, `tests/test_phase7_phase8_noise_gate.py`

---

## Evidence (smoke results)

### Vlad before ranking fix (`noise_protocol_blackbox_vlad`)
| Tier | selected | reason | R2clean |
|------|----------|--------|---------|
| clean | `[1,2,3,4]` **dropped x0** | `selected_top_features` | ~0.70 |
| outliers | all 5 | uncertain | ~0.76 |

### Vlad after ranking + Phase C (`noise_protocol_blackbox_vlad2`)
| Tier | selected | reason | weights | R2clean |
|------|----------|--------|---------|---------|
| clean | **all 5** | uncertain / plateau path | `provided` (auto MAD) | **~0.79** |
| outliers | **all 5** | uncertain | `provided` | ~0.78 |

Exact still 0; formulas approximate/bloated — **selection fixed, structure recovery still open**.

### Other notes
- Pagie: high R2clean, Exact=0, high complexity (fast-path bloat)
- Feynman-I.9.18: weak ~0.67–0.78; often under-uses structure
- Residual `noise_band` often stays `clean` on white/outlier tiers (geometry
  score soft; auto weights still fire via target MAD / uncertain selection)

---

## Git state (do not assume committed)

```text
Branch: noise-handling
Modified:
  glassbox/sr/blackbox_preprocessor.py
  glassbox/sr/sklearn_wrapper.py
  scripts/benchmark_noise.py
  tests/test_benchmark_noise.py
  tests/test_blackbox_preprocessor.py
  noise_handling_audit.md
  noise_handling_phases.md
Untracked:
  blackbox_noise_plan.md
  blackbox_noise_handover.md   (this file)
  scripts/calibrate_noise_routing.py
  tests/test_phase7_phase8_noise_gate.py
```

Suggested commit message:

```text
feat(noise): blackbox multi-var protocol, weighted ranking, and robust search routing

Measure multi-feature blackbox under noise, stabilize feature selection on
near-ties, and gate soft MAD weights / plan budget when selection is uncertain.
```

Include Phase 7–8 files if this commit is meant to freeze the whole noise branch.

---

## How to verify

```text
# Unit / smoke tests
python -m pytest tests/test_blackbox_preprocessor.py tests/test_phase7_phase8_noise_gate.py tests/test_benchmark_noise.py -q

# Multi-var blackbox protocol (tiny)
python scripts/benchmark_noise.py --smoke --blackbox --output-dir results/noise_protocol_blackbox_smoke

# Vlad selection check
python scripts/benchmark_noise.py --blackbox --problems Vladislavleva-4 --tiers clean,outliers_3pct --seeds 11 --output-dir results/noise_protocol_blackbox_vlad2

# Inspect diags
python -c "import json; from pathlib import Path; rows=json.loads(Path('results/noise_protocol_blackbox_vlad2/noise_protocol_rows.json').read_text()); print([{k:r.get(k) for k in ['tier','selected_features','blackbox_reason','sample_weight_mode','ranking_sample_weight_mode','clean_test_r2']} for r in rows])"
```

---

## Architecture facts (avoid re-learning)

1. `blackbox_mode=True` on **1D** data still disables blackbox (`n_features ≤ 1`).
2. Default `blackbox_min_features_to_select=5` → problems with &lt;5 features keep
   all (no top-k). Protocol `--blackbox` lowers this to 2.
3. Display/Pareto MSE stay **unweighted** by design; search can use weights/huber.
4. Track 1 (PMLB) ≠ GT noise protocol: separate metrics (SRBench lesson).
5. Units help physics multi-var; do **not** fake units on PMLB.
6. Prefer uncertain → keep features over aggressive post-hoc feature bans.

---

## Next work (priority)

### P0 — Structure recovery under multi-var noise
- Pagie/Feynman/Vlad Exact still 0; complexity bloat on fast path
- Ablations: `full` vs `no_weights` / `no_robust_loss` on multi-var protocol
- Consider multi-var seeds / constant quality (GOMEA lesson)

### P1 — Residual noise_band sensitivity
- White/outlier tiers often report `noise_band=clean`; auto weights help via
  target MAD, but plan routing may under-expand budget
- Optional: residual-RMS / SNR term in band score (do not confuse with SNR-as-band
  for pink-only detection)

### P1 — Phase D (optional)
- `run_srbench_local` Track 1: dump `runtime_noise` + `search_plan`; optional
  train-only noise flag labeled as **noisy-label R²**

### P2 — Release
- Multi-feature × outliers CI smoke with `blackbox_enabled=True`
- Update `noise_handling_audit.md` blackbox × noise section with vlad2 evidence
- Full multi-seed protocol for publishable tables

### Out of scope (do not start unless requested)
- Classifier/proposer multi-noise retrain
- DN-CL dual-encoder
- Default Track 1 `loss_mode=huber` globally
- AI Feynman recursive separability as Track 1 dependency

---

## Key files

| Path | Role |
|------|------|
| `blackbox_noise_plan.md` | Full plan + literature do/avoid |
| `glassbox/sr/blackbox_preprocessor.py` | Ranking, weights, plateau / keep-all |
| `glassbox/sr/sklearn_wrapper.py` | fit, soft MAD, plan caps, original holdout |
| `scripts/benchmark_noise.py` | Protocol + `--blackbox` |
| `scripts/calibrate_noise_routing.py` | Phase 7 band calibration (1D) |
| `tests/test_blackbox_preprocessor.py` | Ranking / Vlad plateau tests |
| `tests/test_phase7_phase8_noise_gate.py` | Routing + soft MAD + release smokes |
| `tests/test_benchmark_noise.py` | Protocol contract columns |
| `noise_handling_audit.md` / `noise_handling_phases.md` | Status trackers |
| `physo_noise_handling.md` | PhySO comparison |

---

## Open risks

1. Auto soft weights on **clean** multi-var may fire when selection is uncertain
   (seen on clean Vlad: `sample_weight_mode=provided`). Monitor false activation;
   tighten gate if clean recovery regresses.
2. Plateau “keep all” helps Vlad but can retain decoys under high p / low n —
   fallback ridge path still exists in fit.
3. Complexity bloat (comp 90–100) can look like “good R2clean” without Exact;
   do not declare victory on R2clean alone.

---

## One-line status

**Blackbox × noise measurable; ranking + Phase C on. Honest path: search-space
structure seeds with free-const refine (no Exact auto-win). Clean seed=11:
Vlad/Feynman Exact=1; Pagie near-Exact (CleanMSE~6e-6, Accept=1, Exact=0).
Outliers still hard on Exact; R2clean/Accept improved. Results:
`results/noise_protocol_blackbox_p0_seeds2/`.**
