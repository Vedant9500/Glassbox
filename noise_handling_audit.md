# Noise-Handling Audit (Second Pass)

Date: 2026-07-10  
Branch context: `noise-handling`  
Related plans: `physo_noise_handling.md`, `noise_handling_phases.md`

This note records **confirmed** measurement bugs / gaps that can skew
noise-handling experiments, plus items that looked scary but are **not**
bugs after related-file checks. Prefer this over re-deriving status from
memory.

---

## Confirmed (can skew experiment claims)

### 1. Suite `EXACT` under noise ≠ structure recovery

- `scripts/benchmark_suite.py` fits on **noisy** `y` and scores
  `mse_display` on the same noisy labels.
- `score_result` then uses a noise-aware band:
  `tol = max(1e-6, 4 * noise_level² * var(y_clean))`.
- Real artifact `results/before_noise/benchmark_latest.json`
  (`--noise high`, seed 7): **29/30 EXACT**.
- Several EXACT rows are wrong structure (e.g. `-x → -18*sin(...)`,
  `x²` with additive `sin` terms). They pass the wide noisy band but fail
  a clean-target `1e-6` cut.

**Do not use suite EXACT% as the noise-handling success metric.**

### 2. Protocol vs suite measure different things

| Concern | `scripts/benchmark_noise.py` | `scripts/benchmark_suite.py` |
|---------|------------------------------|------------------------------|
| Exact / EXACT | clean MSE `< 1e-6` | noisy MSE vs noise-aware tol |
| Fit metrics | mostly noisy train/test | noisy display MSE |
| Multi-tier protocol report | tool exists | existing baselines are suite JSONs |

- Protocol clean exact is the **better recovery idea**.
- Suite EXACT is **too loose** for recovery claims.
- **Wrong fix:** “align protocol exact to suite noise-aware tol.”
- **Right fix:** report clean recovery metrics in both; keep suite EXACT
  only as a noisy-fit band (legacy).

### 3. Phase 0 multi-tier baseline artifacts are missing

- Tooling: `scripts/benchmark_noise.py` + tests exist.
- `results/before_noise/` is **suite** format
  (`metadata.noise.preset=high`), not `noise_protocol_*.json`.
- Tracker “Phase 0 done” overstates a locked multi-tier protocol baseline.

### 4. Phase 1 tracker checkboxes overstate code

Implemented:
- `fit(..., sample_weight=...)` validate/store/diagnostics
- `_formula_mse` when `len(w) == len(y)`
- CV skip guard

**Not** implemented (tracker still marked done earlier):
- `y_uncertainty → 1/σ²`
- weighted `_display_formula_mse`
- weighted Pareto / candidate screening
- full “final selection honours weights” (display-first path is unweighted)

Outcome text already said “partial”; checkboxes were ahead of code.

### 5. Protocol dead split + unused clean labels

- `_run_single` called `make_seeded_train_test_split` then discarded it
  and reimplemented selection (equivalent, but fragile).
- `y_train_clean` / `y_test_clean` were unused → no clean holdout metrics.

### 6. Constant targets get ~zero additive Gaussian noise

- Noise scale uses `std(y)`. Constants → near-zero noise.
- Suite constant short-circuit still fires. Free EXACT on `'5'`.

### 7. (Related) Some suite runners ignored `noise_cfg`

- Specialist path applied noise.
- Default `run_formula` / pure C++ path historically called
  `_generate_data` **without** `noise_cfg` even when main passed it.
- Non-specialist `--noise` runs could be silently clean.

---

## Partial (real with weights / future; not breaking no-weight runs)

- Silent weight length fallback in `_formula_mse` (holdout subsets drop
  weights without error).
- Protocol R² / raw / display MSE mostly vs noisy labels; only exact used clean.
- `holdout_mse` stub (= raw) until Phase 6.
- No CLI/`__main__` on `benchmark_noise.py` (hygiene).

---

## False positives / overstated (do not “fix” as bugs)

- Pink noise odd-length broken → false.
- Quantization ignoring seed → by design (deterministic quantize).
- Dead split currently wrong train/test → false (reimpl matched helper).
- “Align protocol exact to suite tol” as the primary fix → false direction.
- Phase 1 “weights completely unused” → overstated (`_formula_mse` + CV work).
- BOM on source files → hygiene only.

---

## Fix order (agreed)

1. **Suite:** report `mse_clean` / `r2_clean` / recovery flags; fix runners that drop `noise_cfg`.
2. **Protocol:** clean metrics columns; remove dead split; use shared split helper cleanly.
3. **Tracker:** honest Phase 0 / Phase 1 checkboxes.
4. ~~weight slicing / fail-loud~~, ~~constant-target noise~~, ~~protocol CLI~~ — done.
5. Next: Phase 2 C++ weighted candidate scoring; full protocol baseline run.

---

## How to read future numbers

- **Recovery:** prefer `recovery_exact` / `mse_clean` / `r2_clean` / protocol `exact_match` on clean targets.
- **Noisy fit quality:** suite `score` / `mse` / protocol `test_r2` on noisy labels.
- Never equate the two.


---

## Fixes applied (2026-07-10)

1. **Suite clean recovery metrics** — `mse_clean`, `r2_clean`, `recovery_exact`,
   `recovery_acceptable` attached in all three runners; markdown report columns
   `CleanMSE` / `R2clean` / `Recov`. Default `run_formula` and pure C++ runners
   now honour `noise_cfg` (previously dropped).
2. **Protocol clean metrics** — `clean_test_mse`, `clean_test_r2`, `clean_full_mse`,
   `acceptable_clean`, `false_confidence_vs_clean`; dead split reimplementation
   removed in favour of `make_seeded_train_test_split`.
3. **Tracker honesty** — `noise_handling_phases.md` Phase 0/1 checkboxes corrected;
   multi-tier protocol baseline marked in-progress until `noise_protocol_*.json`
   is produced.

### Second batch (same day)

4. **Weight fail-loud + slicing** — `_weighted_mse` / `_weighted_r2` raise on
   length mismatch; `_slice_sample_weight` + `_active_sample_weight`; holdout
   scoring slices fit-time weights by `val_idx`; CV guard fails closed on
   weight errors.
5. **Constant-target noise rule** — `noise_amplitude_scale()` falls back to
   mean abs level (then 1.0) so constants are not free under Gaussian/pink/
   outlier noise; suite fallback matches.
6. **Multi-tier protocol CLI** — `python scripts/benchmark_noise.py` with
   `--smoke` / full default problem set writes `noise_protocol_*.json` under
   `results/noise_protocol_baseline/`.

Still open for Phase 2+: C++ candidate scoring / evolution weights, display/
Pareto weight threading, full multi-seed baseline run on real estimator,
`y_uncertainty` API.


---

## Phase 2 completed (weighted candidate scoring)

- C++ `score_formula_candidates(..., fit_weights=None, val_weights=None)`:
  weighted affine fit + dual metrics; primary `mse`/`r2` weighted when weights
  given; `unweighted_*` always returned.
- Python: `_split_sample_weights`, weighted `_score_formula_candidate`, refine +
  specialist probe paths pass weights.
- Extension rebuilt: `glassbox/sr/cpp/_core.cpython-314-x86_64-linux-gnu.so`.
- Next: Phase 3 weighted native evolution.
