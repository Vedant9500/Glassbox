# Blackbox × Noise Handling — Implementation Plan

## Goal

Make multi-feature blackbox measurable under noise, then fix the ranking/search
front-end so recovery can actually improve.

Noise handling (Phases 0–8) is already wired into the estimator (weights, robust
loss, residual diagnostics, `noise_band` → search plan, cleanup/Pareto guards).
The **true blackbox path** is still weak because:

| Layer | Today |
|--------|--------|
| Track 1 (`run_srbench_local.py`) | Real PMLB multi-col; no noise tiers; no clean recovery |
| `benchmark_noise.py` | Clean-vs-noisy metrics, but **1D only** → blackbox path off |
| `blackbox_preprocessor` | Feature rank / interactions **unweighted** (outliers flip selection) |
| Plan Phase 7 | Expands budget from residuals; **cannot fix wrong feature subset** |

Success metric remains **clean recovery** (`R2clean` / `CleanMSE` / protocol
exact), not suite noisy EXACT% or Track-1 noisy-label R² alone.

---

## Literature research (findings, not feature shopping)

Sources reviewed (arXiv + internal PhySO note):

| Paper | ID | Why it matters |
|-------|-----|----------------|
| SRBench — Contemporary SR methods | [2107.14351](https://arxiv.org/abs/2107.14351) | Real blackbox vs ground-truth+noise tracks |
| PhySO — units-guided deep SR | [2303.03192](https://arxiv.org/abs/2303.03192) | Noise robustness via search reduction |
| DSR — deep symbolic regression | [1912.04871](https://arxiv.org/abs/1912.04871) | Best-case objectives; constraints; noise curves |
| AI Feynman | [1905.11481](https://arxiv.org/abs/1905.11481) | Multi-var physics structure, not tabular blackbox |
| DN-CL — SR against noise | [2406.14844](https://arxiv.org/abs/2406.14844) | Explicit noise-view training (arch mismatch) |
| GP-GOMEA + coefficients | [2204.12159](https://arxiv.org/abs/2204.12159) | Constants matter for rediscovery |
| Internal | `physo_noise_handling.md` | PhySO vs Glassbox noise story |

### What strong methods actually found

**1. SRBench (La Cava et al.) — separate the two blackbox stories**

- **Real-world tabular (no known formula):** best systems combine **genetic search + parameter estimation** and/or **semantic search drivers**. Metric is error vs complexity, not exact equation.
- **Synthetic ground truth + noise:** exact recovery is a different game; DL and GA perform **similarly** under noise.
- **Implication for Glassbox:** Track 1 (PMLB) and multi-var noise protocol must stay separate. Improving Track-1 R² alone will not prove structure recovery. Do **not** treat noisy test R² as EXACT recovery.

**2. PhySO — noise wins come from search reduction, not denoisers**

- Dimensional analysis / units remove unphysical expressions **before** noise can make them look good.
- Point weights (`y_weights`) enter free-constant opt and reward; default is still **squared error** (squashed NRMSE), so outliers still hurt without weights.
- Free constants optimized **inside** the candidate loop.
- Robust to ~0.1%–10% noise on Feynman when units constrain the space.
- **Implication:** On physics multi-var with units, keep Phase 5 units path. On PMLB tabular blackbox, units usually **do not exist** — do not fake units. Prefer **weights + structural narrowing** over inventing a denoiser front-end.

**3. DSR — optimize best-case recovery; be careful with post-hoc constraints**

- Risk-seeking policy gradient beats expected-reward PG for **exact recovery** (best-case, not average fit).
- **Post-hoc** constraints on GP can **hurt** recovery; **in-situ** constraints during generation work better.
- Recovery falls as noise rises and as dataset shrinks (published noise/size curves).
- **Implication:** Expanding budget under residual noise (Phase 7) is aligned with “best-case recovery.” Hard post-hoc “reject anything using dropped features after full search” is risky; prefer **uncertain → keep more features** (already in preprocessor) over aggressive irreversible drop.

**4. AI Feynman — multi-var physics, not SRBench blackbox**

- Wins from **symmetry / separability / compositionality** and recursive problem splitting.
- **Implication:** Optional later for multi-var Feynman-style problems. **Not** the first lever for PMLB Track 1 (no known symmetries).

**5. DN-CL — train against noise views**

- Contrastive dual-encoder treats noisy vs clean as views of the same expression.
- **Implication:** Architecture mismatch with Glassbox’s C++ evolution + classifier/proposer. **Avoid** as Phase A–C work unless we commit to a separate research path.

**6. GP-GOMEA coefficient mutation**

- Coefficient optimization materially improves rediscovery on SRBench ground-truth sets.
- **Implication:** Keep constant / IRLS / ridge paths strong under noise; ranking-only fixes without good constants will under-deliver.

### Do / avoid (mapped to this plan)

| Do (literature-aligned) | Avoid (unless architecture is built for it) |
|-------------------------|-----------------------------------------------|
| Measure **clean recovery** on multi-var GT + noise (SRBench-style) | Using Track-1 noisy R² as structure-recovery proof |
| Weight-aware ranking + search (PhySO weights) | Assuming default robust `loss_mode` for all Track 1 |
| Keep units on physics multi-var when available | Faking units on unitless PMLB tables |
| Soft expand budget under noise + selection uncertainty | Hard Track-1 caps that always clamp timeout to ≤1.0 under high noise |
| Uncertain selection → retain more features (already present) | Aggressive post-hoc feature bans after search (DSR: post-hoc hurts) |
| Constant-aware recovery (IRLS/ridge, free constants) | Denoiser / contrastive dual-encoder as first PR |
| Separate real-world vs recovery tables | One blended “blackbox score” that mixes both |

---

## Codebase audit (what we have / gaps / plan errors)

### Already in place (estimator)

| Capability | Where | Blackbox-ready? |
|------------|-------|-----------------|
| `sample_weight` end-to-end | `sklearn_wrapper.fit`, MSE/R², CV, Pareto, cleanup | Yes for search **after** prep; **not** passed into ranking |
| Robust `loss_mode` | constructor + C++ evolution | Yes if set; default `mse` |
| Runtime noise diags → `noise_band` | `_compute_runtime_noise_diagnostics` | Yes; plan still uses band |
| Noise-calibrated accept/shrink | `_derive_blackbox_search_plan` + `_NOISE_BAND_THRESHOLDS` | Yes |
| Units remap under selection | fit blackbox branch | Yes when units provided |
| Uncertain feature selection keeps all | `prepare_blackbox_search` uncertain branch | Yes — keep this |

### Hard gaps

| Gap | Evidence |
|-----|----------|
| Noise protocol defaults are **all 1D** | `DEFAULT_BASELINE_PROBLEMS` in `benchmark_noise.py` (~722–729) |
| Factory sets `blackbox_mode=True` but 1D disables blackbox | `prepare_blackbox_search`: `n_features <= 1` → `enabled=False` |
| Ranking / Lasso / trees / MI ignore weights | `_cheap_feature_scores`, `_sparse_linear_scores`, `_tree_importance_scores` — no `sample_weight` |
| `prepare_blackbox_search(...)` has no weight arg | signature ~595–604; call site ~5306–5314 does not pass weights |
| Protocol never calls `fit(..., sample_weight=...)` | ablation can only **block** weights; nothing injects them |
| Track-1 blackbox caps fight noise recovery | when blackbox enabled: `generation_multiplier` ≤ 2.0, `timeout_multiplier` ≤ **1.0** (~4300–4326) |

### Plan / design errors to fix (this document)

1. **“Add Pagie-1 / Feynman multi” alone does not exercise feature ranking.**  
   Default `blackbox_min_features_to_select=5` means:
   - `n_features < 5` → **retain all features**, no top-k selection  
   - Pagie-1 (2), Feynman-I.9.18 (3), Feynman-I.8.14 (4) → blackbox **enabled** but **no drop**  
   - Vladislavleva-4 (5) → ranking **does** run (`n_features < 5` is false)  
   **Fix for Phase A:** either  
   - include ≥1 problem with `n_features ≥ 5` (e.g. Vladislavleva-4), **and/or**  
   - protocol factory sets `blackbox_min_features_to_select` low enough that multi-var problems actually select, **and** log `reason` / `feature_selection_uncertain`.

2. **Factory already sets `blackbox_mode=True`.**  
   Phase A is not “turn blackbox on” — it is “give multi-col data + log whether path really enabled.”

3. **Phase C must not flip global Track-1 `loss_mode` to robust by default.**  
   Literature: real-world winners are GA + constants/semantics; PhySO stays weighted MSE. Keep robust modes **flag/ablation/gated**.

4. **Calibration script is 1D oracle residuals.**  
   `calibrate_noise_routing.py` does not validate multi-var blackbox bands. Optional follow-up, not Phase A blocker.

5. **Suite `blackbox_mode=True` on 1D is misleading** for blackbox×noise claims (`benchmark_suite.py` specialist path). Do not cite suite EXACT under `--noise` as blackbox recovery.

---

## Phase A — Measure (do first) — DONE

**Files:** `scripts/benchmark_noise.py`, `tests/test_benchmark_noise.py`

1. `DEFAULT_BLACKBOX_PROBLEMS`: Pagie-1, Feynman-I.9.18, Vladislavleva-4.
2. CLI `--blackbox` / `--smoke --blackbox` uses multi-var set + lower
   `blackbox_min_features_to_select=2` so ranking can drop features.
3. Row fields: `n_features`, `blackbox_enabled`, `blackbox_reason`,
   `selected_features`, `n_selected_features`, `feature_selection_uncertain`,
   `ranking_sample_weight_mode`, `noise_band`, `noise_pressure`.
4. Clean metrics retained.

**Smoke:**
```text
python scripts/benchmark_noise.py --smoke --blackbox --output-dir results/noise_protocol_blackbox_smoke
```

---

## Phase B — Ranking under noise — DONE

**Files:** `glassbox/sr/blackbox_preprocessor.py`, `glassbox/sr/sklearn_wrapper.py`

1. `sample_weight` on `prepare_blackbox_search`, ranking, interactions.
2. Weighted corr / poly / holdout poly / Lasso / ElasticNet / ExtraTrees;
   MI via weighted resampling.
3. `fit` passes `sample_weight_` into preprocessor.
4. Diagnostics: `ranking_sample_weight_mode` in `state_to_dict`.

**Test:** `test_weighted_ranking_prefers_true_features_under_outliers`.

---

## Phase C — Search defaults when blackbox is noisy — DONE

**Files:** `glassbox/sr/sklearn_wrapper.py`, ranking stability in `blackbox_preprocessor.py`

1. **Ranking stability:** near-tie plateau extension; keep all when score spread is flat
   or `n_usable <= max_features`; reasons:
   `selected_top_features_plateau_extended`, `retained_all_features_score_plateau`,
   `retained_all_features_within_budget`.
2. **`blackbox_noise_robust` (auto|True|False):** auto soft MAD weights from `y`
   when blackbox multi-feature, no user weights, and heavy tails / uncertain
   selection; re-runs ranking with weights; optional switch `mse→huber`.
3. **Plan caps:** when blackbox + (`noise_band` medium/high OR selection
   uncertain OR `noise_pressure` high) relax gen/pop/timeout ceilings (timeout
   max 1.45 vs hard 1.0).
4. **Original-space holdout** re-score after remap (`original_space_holdout` diag).

**Done when:** Vlad clean keeps all 5 features; high-noise plan timeout can exceed 1.0.

---

## Phase D — Track 1 (optional, later)

**File:** `scripts/run_srbench_local.py`

1. Optional `--noise` on `y_train` only; label metrics as **noisy-label** R² (never call it Exact recovery).
2. Always dump `runtime_noise` + `search_plan` into track JSON.
3. Keep SRBench-style reporting: error/complexity Pareto for real data; do not invent clean labels.

**Done when:** one PMLB smoke has noise diag columns filled.

---

## Phase E — CI / release

1. Multi-feature × outliers CI smoke with `blackbox_enabled=True` and, if possible, a case where selection drops features.
2. Ablation table on multi-var noise protocol for release notes.
3. Update `noise_handling_audit.md` with blackbox × noise section + literature do/avoid.
4. Do **not** retrain classifier/proposer on multi-noise in this plan (out of scope).

---

## Order and effort

| Step | Effort | Impact |
|------|--------|--------|
| A measure | S | Unblocks everything; fixes measurement lie |
| B ranking | M | Biggest structural blackbox win under noise |
| C plan/loss (gated) | M | Budget + optional robust search |
| D Track 1 diags | S | Real-data visibility only |
| E CI | S | Regression lock |

---

## Minimal first PR

```text
benchmark_noise:
  - multi-var problems (incl. one n_features>=5 or lower min_features_to_select)
  - row fields: blackbox_enabled, reason, selected_features, noise_band
blackbox_preprocessor:
  - sample_weight into ranking APIs
sklearn_wrapper:
  - pass sample_weight_ into prepare_blackbox_search
test:
  - multi-col outliers → more stable selected features with weights
```

---

## Relevant files

| Path | Role |
|------|------|
| `scripts/benchmark_noise.py` | Noise protocol (extend multi-var + blackbox diags) |
| `scripts/run_srbench_local.py` | Track 1 blackbox; `GROUND_TRUTH_PROBLEMS` multi-var catalogue |
| `scripts/calibrate_noise_routing.py` | Noise-band calibration (1D today) |
| `glassbox/sr/sklearn_wrapper.py` | fit, diagnostics, search plan, weights/loss, blackbox caps |
| `glassbox/sr/blackbox_preprocessor.py` | Ranking, interactions, reduced search (unweighted today) |
| `scripts/classifier_fast_path.py` | Multivariate fast-path (heuristic; not first PR) |
| `tests/test_phase7_phase8_noise_gate.py` | Existing noise routing / release smokes |
| `tests/test_blackbox_preprocessor.py` | Preprocessor unit tests |
| `physo_noise_handling.md` | PhySO comparison (weights, units, not denoisers) |
| `noise_handling_audit.md` | Status tracker to extend |
| `noise_handling_phases.md` | Phase checklist |

---

## Out of scope (for this plan)

- Full classifier/proposer retrain on multi-noise profiles
- Changing default Track 1 `loss_mode` to robust globally
- Physics units on unitless PMLB tabular blackbox
- DN-CL / dual-encoder contrastive noise models
- AI Feynman recursive separability as a Track 1 dependency
- Claiming suite noisy EXACT% as blackbox recovery

---

## Notes

- Display/Pareto MSE stay unweighted by design unless a weighted contract is
  explicitly requested.
- Soft units floor and robust IRLS ridge remain as in the univariate noise work;
  this plan does not re-open those defaults.
- Suite paths that set `blackbox_mode=True` on **1D** data still disable the
  real blackbox preprocessor (`n_features ≤ 1`); multi-var protocol problems are
  required to exercise the path.
- **Selection threshold trap:** `blackbox_min_features_to_select=5` skips top-k
  ranking on problems with fewer than 5 features; protocol design must account
  for that or results will look “blackbox on” without testing ranking under noise.
