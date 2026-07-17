# Noise-Handling Baseline Findings

Date: 2026-07-17  
Branch: `noise-handling`  
Protocol freeze: `results/noise_protocol_baseline/noise_protocol_20260717_143607`  
Related: `physo_noise_handling.md`, `noise_handling_phases.md`, `noise_handling_audit.md`,
`noise_handling_post_phase3_compare.md`

This note freezes what the Phase 0 multi-seed protocol showed and turns it into
a **fix list**. Use it when implementing Phase 3+ so we do not re-litigate the
baseline or optimize the wrong metric.

---

## 1. Freeze artifacts (do not overwrite casually)

| Artifact | Path |
|----------|------|
| Rows | `results/noise_protocol_baseline/noise_protocol_rows.json` |
| Summary | `results/noise_protocol_baseline/noise_protocol_summary.json` |
| Report | `results/noise_protocol_baseline/noise_protocol_report.md` |
| Stamped copy | `results/noise_protocol_baseline/noise_protocol_20260717_143607/` |

**Matrix:** 6 problems × 7 tiers × 5 seeds = **210 cells**, **0 failed seeds**.

**Problems:** Poly-x2, Poly-x3-x, Nguyen-1, Nguyen-5, Keijzer-4, Feynman-I.6.20a  
**Tiers:** clean, gaussian_0.1pct, gaussian_1pct, gaussian_10pct, pink_5pct, quantization_64, outliers_3pct  
**Seeds:** 11, 23, 47, 89, 137  

When re-running after a fix, use the **same** problems / tiers / seeds / budgets
and write to a **new** output dir (e.g. `results/noise_protocol_post_phase3/`).
Compare against this freeze, not against suite `EXACT%` under noise (see audit).

---

## 2. Headline result

Glassbox is **strong on clean-target prediction under noise**, but **weak on
exact symbolic recovery + formula parsimony** once noise is non-trivial.

| Metric | Baseline signal |
|--------|-----------------|
| Accept (clean R² threshold) | ~**100%** almost everywhere |
| R2clean | usually **≥ 0.99** |
| FalseConf | **0** |
| Clean-tier Exact | **~83%** |
| Exact under outliers_3pct | **~10%** |
| Exact under gaussian_10pct | **~23%** |
| Exact under pink_5pct | **~27%** |
| Complexity under noise | rises (e.g. Nguyen-1 7 → 15–22; Keijzer-4 outliers 7 → 16) |
| `sample_weight_mode` on inspected rows | often **`none`** — baseline is mostly unweighted search |

**Interpretation:** the model still approximates the true clean function well
(R2clean / Accept), but under medium–heavy noise it frequently returns the
**wrong or more complex formula** (Exact↓, Complexity↑). That is the core
PhySO-style noise-handling gap for this codebase.

Do **not** treat suite noisy-fit EXACT or raw noisy R² alone as success.
Primary success metrics: **Exact (structure)**, **Complexity**, **CleanMSE /
R2clean**, **Accept**, **FalseConf**.

---

## 3. Exact recovery by tier (all problems, 5 seeds)

Aggregated from freeze rows (approx.):

| Tier | Exact | Accept | R2clean (mean) | Notes |
|------|------:|-------:|---------------:|-------|
| clean | ~83% | 100% | ~1.00 | Solid symbolic baseline |
| gaussian_0.1pct | ~80% | 100% | ~1.00 | Almost free |
| gaussian_1pct | ~47% | 100% | ~1.00 | Exact drops; fit stays good |
| quantization_64 | ~40% | ~97% | high median, rare blow-ups | Seed-sensitive |
| pink_5pct | ~27% | 100% | ~1.00 | Structure less exact |
| gaussian_10pct | ~23% | 100% | ~1.00 | Classic noise-overfit / bloat |
| outliers_3pct | ~10% | 100% | ~0.99 mean, heavier tails | Hardest for Exact |

**Pattern:** Accept stays high while Exact collapses → search is happy with
any expression that tracks clean signal well enough, including bloated
approximations of noise residual.

---

## 4. Per-problem notes

| Problem | Exact (all tiers) | What to fix / watch |
|---------|------------------:|---------------------|
| Poly-x2 | ~74% | Best of set; outliers Exact low but Accept 1 — good canary for Exact↑ without R2clean↓ |
| Nguyen-5 | ~57% | Light noise OK; outliers Exact 0 on freeze table |
| Keijzer-4 | ~51% | Clean Exact OK; **outliers Exact 0**, complexity 7 → ~16 |
| Poly-x3-x | ~40% | Exact fragile from 1% Gaussian up; pink complexity ~10 |
| Nguyen-1 | ~31% | Complexity explodes under noise (7 → 19–22); one quant disaster cell |
| Feynman-I.6.20a | ~11% | **Clean Exact already 0** (Accept 1) — not mainly a noise bug |

### Feynman special case

Do not claim “noise ruined Feynman.” Clean Exact is already 0 while R2clean /
Accept stay excellent. Track **R2clean + complexity** for Feynman until the
clean symbolic match path improves (representation / exact-match / formula
canonicalization), not Exact% alone.

---

## 5. Failure modes to fix (priority order)

### P0 — Outliers destroy exact structure

- **Evidence:** outliers_3pct Exact ~10%; Keijzer-4 / Nguyen-1 / Nguyen-5 Exact 0 on table; complexity jumps; worst clean-R² tails cluster here.
- **Symptom:** a few wild points dominate MSE → evolution / refine chase outliers with extra sin/cos or high-degree junk.
- **Likely levers:**
  - Phase 3: weighted native evolution fitness (down-weight residual outliers once weights exist).
  - Phase 4: robust loss (Huber / trim) on fit + evolution paths.
  - Soft residual / ranking weights already gated in blackbox noise-robust path — ensure protocol and native path actually use them (`sample_weight_mode` should not stay `none` when noise pressure is high).
- **Success (vs freeze):** Exact on outliers_3pct **≫ 10%** without Accept/R2clean regression; median complexity on outlier cells **down** toward clean complexity.

### P1 — Medium noise → wrong formula, same clean R²

- **Evidence:** gaussian_10pct Exact ~23%, pink ~27%; R2clean still ~1.0; Nguyen-1 complexity 15–19.
- **Symptom:** “good enough” approximation of clean signal with extra terms that soak noise.
- **Likely levers:**
  - Phase 3: weights + complexity-aware / weighted fitness in C++ evolution (not only post-hoc candidate scoring from Phase 2).
  - Stronger complexity / parsimony pressure when `noise_pressure` is high.
  - Phase 6: residual / cleanup guards so late stages do not re-bloat.
- **Success:** Exact↑ on gaussian_10pct / pink; complexity median **closer to clean-tier**; CleanMSE not worse.

### P2 — Quantization seed blow-ups

- **Evidence:** Nguyen-1 quantization seed=11 with clean_test_r2 ≈ **-0.12**, complexity **45**; tier median still fine.
- **Symptom:** rare catastrophic overfit / unstable expression, not systematic tier failure.
- **Likely levers:** complexity caps, rejection of non-finite / insane expressions, optional multi-start / holdout guard already present — verify they fire under quantization.
- **Success:** no cells with clean_test_r2 ≪ 0.9 on this matrix; max complexity capped sanely.

### P3 — Weights not driving the search in baseline

- **Evidence:** inspected freeze rows show `sample_weight_mode: "none"`.
- **Symptom:** Phase 2 weighted **candidate scoring** may be live in code, but this baseline still reflects largely **unweighted** fit/evolution behavior.
- **Likely levers:**
  - Wire auto sample weights / noise-robust mode into protocol estimator factory when tier is non-clean (or always compute residual weights after first pass).
  - Phase 3: pass weights into `run_evolution_cpp` / fitness / constant fitting.
- **Success:** non-clean tiers report non-`none` weight mode when robust path is enabled; ablation `full` vs `no_sample_weight` shows delta on outliers / 10% Gaussian.

### P4 — Feynman clean Exact = 0 (orthogonal track)

- Not a noise-handling regression from this freeze.
- Track separately from P0–P3; do not block Phase 3 on Feynman Exact%.

---

## 6. What is already working (do not break)

1. **Accept / R2clean** are excellent — predictive recovery is not the crisis.
2. **FalseConf = 0** on freeze — keep false-confidence definition honest (clean vs noisy).
3. **Clean + 0.1% Gaussian** Exact stays high — regressions here are release blockers.
4. **Phase 0 instrumentation** (clean columns, protocol CLI, dashboard) is good enough to measure fixes.
5. **Phase 2** weighted C++ candidate scoring + Python wiring is in tree; do not re-implement scoring weights — **thread weights deeper** (evolution).

---

## 7. Target deltas after fixes

Compare post-fix protocol to freeze `noise_protocol_20260717_143607`.

| Tier / focus | Baseline Exact (approx.) | Target direction |
|--------------|-------------------------:|------------------|
| clean | 83% | no regression |
| gaussian_0.1pct | 80% | no regression |
| gaussian_1pct | 47% | ↑ |
| gaussian_10pct | 23% | **↑ primary** |
| pink_5pct | 27% | **↑ primary** |
| outliers_3pct | 10% | **↑ primary** |
| quantization_64 | 40% | ↑; kill left-tail disasters |
| Accept all tiers | ~100% | hold |
| FalseConf | 0 | hold |
| Complexity on noisy tiers | elevated | **↓ toward clean** |

Secondary: publish ablation table (`full` vs no-weight / no-robust-loss) once Phase 3–4 land.

---

## 8. Implementation map (where to work)

| Priority | Phase | Primary files / area |
|----------|-------|----------------------|
| P0 / P1 / P3 | **Phase 3** | `glassbox/sr/cpp/core.cpp` / evolution fitness, constant fit, ridge/pruning; `sklearn_wrapper.py` weight plumbing into native evolution |
| P0 | **Phase 4** | robust loss (`loss_mode`, Huber/trim) on fit + evolution + refine |
| P1 | **Phase 6** | residual stage / cleanup guards / bloat rejection under noise |
| P2 | Phase 3 + 6 + existing CV/holdout guards | reject catastrophic candidates |
| P3 | protocol factory + blackbox noise-robust path | `scripts/benchmark_noise.py` factory kwargs; `blackbox_noise_robust` |
| Measure | Phase 0 CLI | `python scripts/benchmark_noise.py` same seeds/tiers; new `--output-dir` |

Suggested implementation order (from phase tracker):  
**3 → 6 → 4 → 5 → 7 → 8** (with re-measure after 3 and after 4).

---

## 9. Reproduction

```bash
# Same freeze command shape (adjust budgets only if freeze log recorded different ones)
python scripts/benchmark_noise.py \
  --output-dir results/noise_protocol_baseline \
  --seeds 11,23,47,89,137

# After a fix — NEW directory
python scripts/benchmark_noise.py \
  --output-dir results/noise_protocol_post_phase3 \
  --seeds 11,23,47,89,137
```

Compact progress dashboard is default; use `--detail` for one line per cell,
`--show-fit-logs` only when debugging fits.

---

## 10. Working checklist

- [x] Phase 0 protocol freeze captured (2026-07-17)
- [x] Issues documented (this file)
- [x] Phase 3: weights in native evolution fitness / constant fitting
      (C++ already; 1D auto residual soft-weights → evolution, 2026-07-17)
- [x] Confirm non-clean runs can report non-`none` sample_weight_mode (`auto_soft_mad`) when robust path on
- [x] Re-measure protocol post Phase 3 → `noise_protocol_post_phase3` (see `noise_handling_post_phase3_compare.md`): Exact flat; outliers +3.3pp; Nguyen-1 outlier Accept regression
- [ ] Re-run protocol → `noise_protocol_post_phase3` and diff Exact/Complexity on outliers + 10% Gaussian + pink
- [ ] Phase 4 robust loss if outliers Exact still weak
- [ ] Phase 6 residual/bloat guards if complexity still climbs under noise
- [ ] Optional: ablation table for release notes
- [ ] Do not declare victory on suite noisy EXACT alone

---

## 11. One-line summary

**Baseline:** prediction under noise is fine; **exact formula recovery and parsimony under outliers / 10% Gaussian / pink are not** — fix weights in evolution (Phase 3), then robust loss (Phase 4) and bloat guards (Phase 6), measuring with clean recovery metrics against freeze `20260717_143607`.

---

## 12. Phase 3 implementation note (2026-07-17)

Native weighted evolution was already implemented (`y_weights` → fitness /
DifferentialGramian / islands). The baseline still showed `sample_weight_mode: none`
because auto soft-MAD only ran for **multi-feature blackbox**.

**Closed in this pass:**

1. `GlassboxRegressor.fit`: `blackbox_noise_robust="auto"` also considers **1D** data.
2. Weights from **residual** soft-MAD (`_auto_residual_soft_weights`), not raw `y`
   (avoids false positives on clean `x²` / Nguyen-1).
3. When active: sets `sample_weight_`, may switch `loss_mode` to `huber`, passes
   weights into `_core.run_evolution` / guided evolution.
4. Protocol `_sample_weight_mode` reports `auto_soft_mad` / `provided` / `none`.
5. Ablation `no_weights` disables auto robust path.

**Still measure:** full protocol re-run vs freeze before claiming Exact gains.
