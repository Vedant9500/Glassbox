# Post–Phase 3 Protocol Compare

Date: 2026-07-17  
Baseline freeze: `results/noise_protocol_baseline/noise_protocol_20260717_143607`  
Post run: `results/noise_protocol_post_phase3/noise_protocol_20260717_150555`  
Matrix: 6 problems × 7 tiers × 5 seeds = **210 / 210** cells

Related: `noise_handling_baseline_findings.md`, Phase 3 closeout in `noise_handling_phases.md`.

---

## 1. Did Phase 3 wire correctly?

| Check | Result |
|-------|--------|
| Clean tier Exact / Accept | **Unchanged** (83.3% / 100%) — no regression on clean |
| `sample_weight_mode` baseline | all `none` (210) |
| `sample_weight_mode` post | **`auto_soft_mad` 39**, `none` 171 |
| Outliers with auto weights | **29 / 30** cells |
| Pink with auto weights | **0 / 30** (residual soft-MAD rarely fires) |
| Gaussian 10% auto weights | **3 / 30** (mostly still unweighted) |

**Conclusion:** plumbing works for **outlier-like residual structure**. It does **not** yet reweight diffuse Gaussian/pink noise (expected: residual MAD soft weights target sparse heavy tails, not isotropic noise).

---

## 2. Headline deltas (post − base)

| Tier | Exact | Accept | mean R2clean | mean complexity | Auto weights |
|------|------:|-------:|-------------:|----------------:|--------------|
| clean | +0.0% | +0.0% | ~same | 0.0 | 0 |
| gaussian_0.1pct | +0.0% | +0.0% | ~same | 0.0 | 3 |
| gaussian_1pct | +0.0% | +0.0% | ~same | 0.0 | 3 |
| gaussian_10pct | +0.0% | +0.0% | ~same | **−0.3** | 3 |
| pink_5pct | **−3.3%** | +0.0% | ~same | **+1.2** | 0 |
| quantization_64 | +0.0% | +0.0% | ~same | 0.0 | 1 |
| **outliers_3pct** | **+3.3%** (10→13.3) | **−3.3%** (100→96.7) | **↓** (0.987→0.960) | **−0.4** | **29** |

Overall Exact: **44.3% → 44.3%** (flat).  
Hard-tier Exact: **25% → 25%** (flat).  
Clean: fully preserved.

---

## 3. Wins (keep)

| Cell / pattern | What improved |
|----------------|---------------|
| **Keijzer-4 × outliers** | Exact **0% → 20%**; median complexity **16 → 10** |
| **Nguyen-5 × outliers** | Exact **0% → 20%** |
| **Poly-x3-x × gaussian_10pct** | Exact **20% → 40%** |
| **Poly-x2 × outliers** | Complexity **4 → 3**; R2noisy up (0.89 → 0.97); CleanMSE down |
| **Feynman × outliers** | Complexity **10 → 7**; R2clean up on table medians |

Net seed-level Exact flips: **3 gained, 3 lost** (swap, not a broad win).

---

## 4. Losses / risks (fix next)

### P0 — Catastrophic Accept regression

- **Nguyen-1 × outliers × seed 11:** Accept lost; clean_test_r2 **≈ −0.12**; complexity **61** with `auto_soft_mad` on.
- Also Nguyen-1 outliers **median complexity 11 → 28**; Accept cell rate **1.0 → 0.8**.

Soft weights + Huber switch can **over-emphasize a bad structure** when the residual probe mis-labels points or evolution still bloats under reweighted loss.

### P1 — Pink / Gaussian 10% barely touched

- Almost no `auto_soft_mad` on these tiers → Phase 3 path is mostly a **no-op** there.
- Pink Exact **−3.3%** overall (noise / one Feynman seed flip); complexity slightly **up**.

### P2 — Exact still far from targets

From findings targets:

| Focus | Baseline Exact | Post | Target direction |
|-------|---------------:|-----:|------------------|
| outliers_3pct | 10% | **13%** | still ≪ need |
| gaussian_10pct | 23% | **23%** | unchanged |
| pink_5pct | 27% | **23%** | slightly worse |

---

## 5. Interpretation

Phase 3 delivered:

1. **Correct instrumentation** (weights actually reach evolution on outlier cells).
2. **Small, localized structure wins** (Keijzer-4 / Nguyen-5 outliers, some complexity cuts).
3. **Not** a broad Exact recovery story yet.
4. One **serious safety issue** (Nguyen-1 outlier blow-up) → need holdout / complexity / residual guards before relying on auto weights as default.

This matches the plan: Phase 3 is necessary plumbing + partial outlier help; **Phase 4 (robust loss defaults / IRLS)** and **Phase 6 (bloat / residual cleanup)** are the next levers for Exact + complexity, plus a **guard** so auto weights cannot produce Accept failures.

---

## 6. Recommended next work (ordered)

1. **Guardrail (quick):** reject / don't promote candidates that tank **unweighted holdout or clean-proxy R²** when auto weights active; complexity cap under `auto_soft_mad`.
2. **Phase 6:** residual-stage / simplify under noise so Nguyen-1 outliers cannot land complexity 28–61.
3. **Phase 4 tighten:** use robust loss more carefully (Huber without always soft-weighting raw residual probes that fight the true family).
4. **Weight design for pink/10% Gaussian:** residual weights alone won't fire — need uncertainty-from-fit, multi-pass residual reweight after first fit, or mild global robust loss without per-point MAD on y-structure.
5. **Re-measure** after guards on same seeds; compare to both freeze and this post_phase3 stamp.

---

## 7. One-line summary

**Phase 3 is live and safe on clean data; outliers get auto weights and a few Exact/complexity wins (Keijzer-4, Nguyen-5), but overall Exact is flat, pink/10% Gaussian barely change, and Nguyen-1 outliers show a must-fix Accept/complexity regression before claiming victory.**

---

## 8. Guardrail (implemented after this compare)

Auto-weight final guard in `GlassboxRegressor` (2026-07-17):

- Active only when `_blackbox_noise_robust_applied_.active` and source is not user weights.
- Rejects / replaces winners that fail **unweighted** checks:
  - complexity cap (1D: 22)
  - full R² ≥ 0.50
  - holdout R² ≥ 0.40
  - train−holdout gap ≤ 0.45
- Prefers simpler tracked fallbacks / cleaned formula / evolution candidate.
- Residual boosting also blocked when candidate fails the same checks.
- Diagnostics: `blackbox_diagnostics_["auto_weight_final_guard"]`.

Re-run protocol after this change before claiming Nguyen-1 Accept fix.
