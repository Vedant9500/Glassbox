# Glassbox Codebase Audit Tracker

**Scope:** Important directories under `glassbox/` only.  
**Out of scope (for this pass):** `eigen/`, `build/`, `__pycache__/`, top-level `scripts/`, `tests/`, `results/`, `docs/`, `models/`, diffusion, scratch.  
**Goal:** Systematic review for bugs, inefficiencies, correctness gaps, and optimization opportunities — section by section.

| Field | Value |
|------|--------|
| Created | 2026-07-18 |
| Restored | 2026-07-18 (recreated after accidental cleanup) |
| Package root | `glassbox/` |
| Approx first-party LOC | ~40k (excl. Eigen/build) |
| Dominant hotspots | `sr/sklearn_wrapper.py` (~9.3k), `sr/cpp/evolution.h` (~3.7k), `evolution/evolution.py` (~3.5k) |
| Status legend | `pending` · `in_progress` · `done` · `blocked` |

---

## How to use this file

1. Pick a section (order recommended below).
2. Mark section status → `in_progress`.
3. Fill findings tables (bugs / inefficiencies / optimizations).
4. Link related tests or repro notes.
5. Mark section `done` only after a full pass + false-positive check on related call sites.
6. Keep severity honest:
   - **P0** correctness / silent wrong results / crash on common paths  
   - **P1** performance that changes experiment conclusions or multiplies runtime  
   - **P2** maintainability, dead code, micro-opts  

**Recommended order:** S1 → S2 → S4 → S5 → S6 → S3 → S7 → S8 → S9 → S10  
(Rationale: public fit path + noise contract first, then C++ search core, then scoring/refine, then multi-feature / ML side paths.)

---

## Package map (important dirs only)

```
glassbox/
├── model_registry.py              # shared model path resolution
├── curve_classifier/              # feature extract + MLP + integration
├── evolution/                     # Python-side evolution / ONN-style path
├── universal_proposer/            # neural skeleton / operator priors
└── sr/
    ├── sklearn_wrapper.py         # main sklearn API + most orchestration
    ├── blackbox_preprocessor.py   # multi-feature ranking / search prep
    ├── specialist_state.py        # specialist vault / compositions
    ├── fpip_v2.py                 # fast-path → evolution handoff schema
    ├── phased_regression.py       # structure then linear coeff fit
    ├── pruning.py                 # sparsity / prune helpers
    ├── hard_concrete.py           # continuous relaxations
    ├── risk_seeking_policy_gradient.py
    ├── visualization.py
    ├── core/                      # OperationNode / OperationDAG
    ├── operations/                # meta ops
    ├── optimizers/                # BFGS + hybrid
    └── cpp/                       # evolution engine + bindings
        ├── core.cpp               # pybind / run_evolution surface
        ├── evolution.h            # island GA + mutations
        ├── eval.h, ast.h, refine.h
        ├── simplify.h, simplify_advanced.h
        ├── formula_parser.h, execution.h
        └── seed_graph_builder.py
```

**Explicitly excluded from section work:** `glassbox/sr/cpp/eigen/**`, `glassbox/sr/cpp/build/**`.

---

## Section index

| # | Section | Primary paths | ~LOC | Priority | Status |
|---|---------|---------------|------|----------|--------|
| S1 | Fit orchestration & public API | `sr/sklearn_wrapper.py` (GlassboxRegressor + fit/predict routing) | ~4–5k of 9.3k | P0 | done |
| S2 | Noise, weights, robust loss | `sklearn_wrapper` weight/loss helpers; noise auto-path | ~0.8–1.2k | P0 | done |
| S3 | Scoring, refine, cleanup, guards | candidate score, refine, snap, parsimony, final guards | ~2–3k | P0 | pending |
| S4 | C++ evolution engine | `sr/cpp/evolution.h` | 3.7k | P0 | done |
| S5 | C++ eval / AST / refine / simplify | `ast.h`, `eval.h`, `refine.h`, `simplify*.h`, `formula_parser.h`, `execution.h` | ~2.7k | P0 | done |
| S6 | C++ bindings & seed graphs | `core.cpp`, `seed_graph_builder.py`, `setup.py`, `export_pytorch.py` | ~2.2k | P0 | pending |
| S7 | Blackbox multi-feature prep | `blackbox_preprocessor.py` | 1.2k | P1 | pending |
| S8 | Specialist vault & state | `specialist_state.py` + wrapper specialist hooks | 1.2k+ | P1 | pending |
| S9 | Curve classifier stack | `curve_classifier/*` (integration + models + features; de-prioritize train/data gen) | ~2–3k review / 6k+ total | P1 | pending |
| S10 | Proposer, ops, optimizers, Python evolution | `universal_proposer/`, `fpip_v2.py`, `core/`, `operations/`, `optimizers/`, `evolution/`, phased/prune/hc/rspg | ~8k+ | P1–P2 | pending |

---

## S1 — Fit orchestration & public API

**Paths**
- `glassbox/sr/sklearn_wrapper.py` — `GlassboxRegressor`, `__init__`, `fit`, `predict`, budget/holdout routing, phase timers
- Cross-check: `glassbox/sr/__init__.py` exports

**What it owns**
- End-to-end pipeline order: validate → blackbox prep → fast-path → evolution → refine → display formula
- Parameter surface, defaults, compute budget, multi-run orchestration
- sklearn contract (`check_X_y`, `check_is_fitted`, attributes after fit)

**Analysis focus**
- [x] Dead / unreachable branches in fit
- [x] Double work (re-scoring same formula, re-running evolution configs)
- [x] State not reset between fits (`fit` re-entrancy)
- [x] Train/test leakage in internal holdouts
- [x] Inconsistent use of weighted vs unweighted metrics at decision points
- [x] ThreadPool / parallel fit races on shared state
- [x] Default params that hurt noise or multi-feature cases

**Status:** `done` (2026-07-19) — full orchestration pass + call-site / sklearn contract checks

### Phase graph (`GlassboxRegressor.fit`)

```
check_X_y + sample_weight validate
  → restore sticky auto loss_mode (N1 fix)
  → reset composition / eval-cache / vault / phase_timings (partial)
  → prepare_blackbox_search  [may re-run if auto soft-MAD activates]
  → optional structure probe (original space, seed-only)
  → remap X/y to search space; n_features_in_ := selected dim
  → Stage 1: classifier fast-path  [may run twice: auto_expand T then F]
  → universal proposer dual path
  → need_evolution gates (R² skip, CV residual guard, bloat, blackbox rules)
  → EARLY EXIT: fast_path_exact  (_finish_with_formula)
  → candidate screening / basis / engineered basis / structure seeds
  → adaptive compute budget (+ blackbox plan multipliers)
  → Stage 2: guided evolution (1D) → C++ multi_start evolution (seed graphs)
  → Stage 3: blackbox Pareto + original-space polish + residual + inception
  → Phase 6 parsimony + auto-weight final guard
  → formula_ / best_mse_ ; restore user loss_mode
```

### Findings

| ID | Severity | Type | Summary | Evidence | Status |
|----|----------|------|---------|----------|--------|
| S1-1 | **P0** | bug / sklearn contract | **Unfitted `predict` is a silent false success.** `__init__` pre-creates many `*_` attrs (`specialist_vault_`, `phase_timings_`, …), so bare `check_is_fitted(self)` returns OK before `fit`. Missing `formula_` is caught and **returns zeros** instead of `NotFittedError`. | `predict` ~9296–9307; verified runtime: unfitted `predict([[1.0]])` → `[0.]` | fixed |
| S1-2 | **P0** | bug | **Proposer “rapid hit” keeps wrong formula.** Loop finds `best_cand_mse = min(candidate.mse)` but assigns `best_formula = candidate_formulas[0]['formula']`, not the argmin. Can adopt a worse skeleton while recording the best MSE. | `fit` ~8385–8393 | fixed |
| S1-3 | **P0** | bug / re-entrancy | **Sticky cross-fit state poisons later fits.** `evolution_candidate_formula_`, `evolution_candidate_mse_`, `pareto_front_`, `nodes_`, `output_weights_`, etc. are **not cleared at `fit` start**. Auto-weight final guard explicitly re-considers `evolution_candidate_formula_` (~4148). Blackbox Pareto also injects prior `evolution_candidate_formula_` (~8832). Second fit can select a formula from a previous problem. | fit reset block ~7301–7321 vs assigns ~8704–8722, 8832, 4148, 9250 | fixed |
| S1-4 | **P1** | bug / sklearn contract | **`n_features_in_` often ≠ feature count of public formula / predict inputs under blackbox.** After feature selection, `n_features_in_ = X_search.shape[1]` (~7627). Formula is remapped to **original** indices. Full-path stage-3 inception success restores `n_features_in_ = X_original.shape[1]` (~9236), but **`_finish_with_formula` early exact path never restores it**. Exception path in inception restores reduced prior (~9233). `predict` uses `X.shape[1]` for symbol binding (works by accident) but sklearn metadata / external checks lie. | ~7624–7627, 7679–7715, 9184–9236 | fixed |
| S1-5 | **P1** | correctness risk | **Internal holdouts are not true held-out validation for selection.** Candidates are fit/refined on full (or mostly full) data; random/edge/tail slices used for Pareto / residual / linear-fallback decisions still include points already seen during structure search / evolution / constant refine. Tail holdout (`X_original[-holdout_n:]`, ~9022, 9042) is especially biased if data is ordered. Optimistic model selection, not leakage into external test sets. | `_select_blackbox_pareto_formula`, `_final_holdout_scores`, stage-3 tail blocks | fixed |
| S1-6 | **P1** | correctness / metric contract | **Weighted vs unweighted metrics mixed at orchestration decision points.** Search/`_formula_mse` honour weights + robust `loss_mode`; many gates use unweighted `best_mse` / `np.mean((pred-y)**2)` / unweighted R² (`_r2_from_mse` on raw mse, ~7924). CV skip guard uses weighted R² on **fixed** predictions (not refit). Display contract is documented as unweighted; risk is using weighted search winners with unweighted skip/accept thresholds inconsistently under auto soft-MAD/Huber. | `_formula_mse` 3528+, `_display_formula_mse` 3561+, CV guard 6543+, evolution selection 8726+ | fixed |
| S1-7 | **P1** | inefficiency | **Systematic double work on the default path.** (1) `prepare_blackbox_search` can run twice when auto soft-MAD activates (~7343 + ~7417). (2) Fast-path can run twice for multi-var blackbox (auto_expand True then False, ~7728–7780). (3) Structure probed early (`_probe_multivariate_structure_original_space`) and again late (`_fit_original_space_structure_winner` + polish). (4) Same formulas re-scored across screening, Pareto, cleanup, residual, inception, guards. (5) Default `multi_start_runs=3` runs sequential full evolutions. | fit stages 1–3 | fixed |
| S1-8 | **P1** | concurrency / race | **ThreadPoolExecutor mutates shared estimator state without locks.** `_fit_search_space_structure_seeds` and original-space free-const fits parallelize; workers call `_safe_eval_formula_array` which increments `formula_eval_count_` and writes `_formula_eval_cache_` (~6478–6501). Race on cache dict + counters; possible lost updates / rare cache corruption. Global `np.random.seed` / `torch.manual_seed` in `fit` (~7327–7329) also pollutes process-wide RNG (parallel sklearn jobs interfere). | ~2777, ~2345, ~6478, ~7327 | fixed |
| S1-9 | **P1** | defaults / product | **Defaults multiply runtime and can hurt noisy / multi-feature jobs.** `multi_start_runs=3`, `generations=1000`, `population_size=100`, `num_islands=8`, `timeout=120` with adaptive budget up to `max_compute_budget=300`. Blackbox + residual + inception + vault all **on** by default. Noise protocol and multi-feature benches pay full stack even when a simpler path would suffice. | `__init__` 958–1040 | fixed |
| S1-10 | **P1** | bug / weak guard | **`_passes_cross_validation_skip_guard` is not CV of refits.** Evaluates one global formula’s predictions on shuffled folds; does not retrain. Misnamed; only checks residual partition stability. With `n_samples < cv_skip_guard_min_samples` (default 45) it **passes** and can skip evolution on weak evidence. | ~6543–6616 | fixed |
| S1-11 | **P2** | API / export | **`GlassboxRegressor` not exported from `glassbox.sr`.** README/docs use `from glassbox.sr.sklearn_wrapper import GlassboxRegressor`. `sr/__init__.py` loads ONN/meta-ops/FPIP stack but not the public estimator. Also `__all__` lists `'FPIPv2Payload'` while module exports `FPIPv2` (import name mismatch). | `glassbox/sr/__init__.py` ~106–193 | open |
| S1-12 | **P2** | maintainability | **Fragile package→scripts coupling for display MSE.** `_display_formula_mse` imports `scripts.benchmark_common` (~3569). Orchestration scoring depends on top-level scripts path being importable. | 3561–3577 | fixed |
| S1-13 | **P2** | dead / confusing control | **`use_guided_evolution=False` also disables residual boosting** (`_run_residual_boosting_impl` / residual stage checks `use_guided_evolution`). Name suggests evolution-only, but residual stage is gated by the same flag. `use_simplification` is live via cleanup path (OK). | ~6735, ~7038 | fixed |

### Related tests / repro notes

| Finding | How to exercise |
|---------|-----------------|
| S1-1 | `GlassboxRegressor().predict([[0.0]])` → zeros, no `NotFittedError` |
| S1-2 | Unit-level: mock `candidate_formulas=[{mse:1e-3,formula:'A'},{mse:1e-9,formula:'B'}]` into rapid-hit branch; expect B, get A |
| S1-3 | Fit problem A with evolution winner stored; fit problem B with auto soft-MAD and early simple path; inspect whether guard/Pareto can resurrect A’s formula |
| S1-4 | Multi-feature blackbox exact early skip; assert `n_features_in_ == X.shape[1]` and `predict(X)` |
| Existing | `tests/test_sklearn_wrapper_cv_guard.py`, `tests/test_sample_weight_contract.py`, `tests/test_phase6_noise_guards.py` cover parts of CV/weight guards but not S1-1/2/3 |

### Suggested fix direction (not applied this pass)

1. **S1-1:** `check_is_fitted(self, attributes=["formula_"])` (and optionally `n_features_in_`); stop zero-filling predict errors — re-raise or use sklearn missing-value policy.
2. **S1-2:** `best = min(candidates, key=mse); best_formula = best["formula"]`.
3. **S1-3:** At fit entry, clear `evolution_candidate_*`, `pareto_front_`, `nodes_`, `output_*`, boosting/inception diagnostics, or set them only after successful stage-2.
4. **S1-4:** Keep `n_features_in_ = original_n_features_in_` as public contract; track search dim separately (`n_search_features_`). Restore in `_finish_with_formula`.
5. **S1-7/9:** Default `multi_start_runs=1`; cache blackbox prep; skip second fast-path when first already compact; de-dupe structure probe.
6. **S1-8:** Local RNG (`np.random.RandomState`) only; thread-local or lock-free eval (no shared cache writes from workers).
7. **S1-11:** Export `GlassboxRegressor` from `glassbox.sr`; fix `FPIPv2` in `__all__`.

### Notes / open questions

- `sklearn_wrapper.py` alone is ~9.3k LOC — S1 covers orchestration / public API only; noise plumbing details remain S2; refine/guard math details S3; C++ evolution body S4–S6.
- Noise path sticky public `loss_mode` was fixed earlier (tracker **N1**, 2026-07-18): auto Huber no longer permanently mutates public `loss_mode`. Fit still restores via `_restore_user_loss_mode_if_auto_switched` on all exit paths checked (main + early finish).
- Seed graphs **are** wired into C++ evolution when `seed_graph_builder` imports (~8545–8590) — historical “seeds unused” theme is partially addressed on the wrapper path; capacity/fraction still for S4/S6.
- No ThreadPool on multi-start evolution itself (sequential runs); races are structure-seed / free-const ThreadPools + global seed.
- `blackbox_candidate_accepted` is diagnostic/budget flag, not a hard early return (except combined with other gates) — not dead, but easy to misread.

---

## S2 — Noise, sample weights, robust loss

**Paths**
- `glassbox/sr/sklearn_wrapper.py` — `_validate_sample_weight`, `_weighted_*`, `_auto_residual_soft_weights`, `_estimate_diffuse_noise_ratio`, `_robust_loss`, auto Huber / soft-MAD path, auto-weight guards
- Cross-check: C++ `evolution.h` residual objective / `core.cpp` `y_weights` + `loss_mode` bindings
- Tests: `tests/test_sample_weight_contract.py`, `test_robust_loss.py`, `test_phase6_noise_guards.py`, `test_phase7_phase8_noise_gate.py`

**What it owns**
- Sample-weight contract (invalid weights must raise, not silently drop)
- Auto soft-MAD (residual MAD, not raw y)
- Diffuse-noise → auto Huber
- Unweighted diagnostics vs weighted search objective

**Analysis focus**
- [x] Weight shape / NaN / zero-sum edge cases
- [x] Soft-MAD false positives on clean polynomials
- [x] Auto Huber threshold (`~0.02`) sensitivity
- [x] User `loss_mode` / user weights override correctness
- [x] Weights sliced correctly for holdout / residual passes
- [x] C++ vs Python weighted MSE parity
- [x] Display metrics always unweighted (protocol contract)
- [x] Sticky public `loss_mode` after auto Huber (fixed 2026-07-18)

**Status:** `done` (2026-07-19) — full noise/weight/loss pass + runtime probes + existing tests (45 passed: robust_loss + sample_weight_contract + phase6)

### Noise pipeline (fit-time)

```
user sample_weight?
        │
   yes ─┴─ store sample_weight_ (mean≈1); source=user; NO auto soft/Huber override
        │
   no ──┼─ blackbox_noise_robust in {auto, True} and (1D or multi blackbox)?
        │         │
        │        no → (optional forced True only) else leave mse
        │         │
        │        yes → _auto_residual_soft_weights (linear/quad/cubic/median residual)
        │                 │
        │            activate soft? (out_frac / low_weight_mass / selection_uncertain /
        │                            retained_all_features / forced)
        │                 │
        │            yes → sample_weight_=soft; often loss_mode→huber; re-prep blackbox
        │                 │   reason=soft_mad_weights; guards ACTIVE
        │            no  → _estimate_diffuse_noise_ratio (same poly probes)
        │                     ratio≥0.02? → loss_mode→huber (no weights)
        │                     reason=diffuse_noise_huber; guards INACTIVE
        │
  search uses _formula_mse / C++ residual_mse (weights + loss_mode)
  display / protocol uses _display_formula_mse (unweighted plain MSE)
  exit restores public loss_mode if auto-switched (N1)
```

### Findings

| ID | Severity | Type | Summary | Evidence | Status |
|----|----------|------|---------|----------|--------|
| N1 | P1 | bug | Auto Huber/soft-MAD permanently mutated public `loss_mode` across fits | fixed 2026-07-18; `_restore_user_loss_mode_if_auto_switched` + fit-entry undo; `test_auto_huber_does_not_stick_loss_mode_across_fits` | fixed |
| N2 | **P0** | bug / false positive | **Diffuse auto-Huber fires on clean non-polynomials.** `_estimate_diffuse_noise_ratio` only probes median + poly(deg≤3 on x0). Clean `sin`, `exp(-x²)`, `1/(1+x²)` leave large residual → ratio ≫ 0.02 → `loss_mode` auto-switches to huber even with zero noise. Verified: sin ratio≈1.18, exp≈0.17, rational≈0.24; clean poly ratio≈0. | `_estimate_diffuse_noise_ratio` ~388–450; fit ~7446–7485; runtime probe 2026-07-19 | fixed |
| N3 | **P1** | bug / false positive | **Soft-MAD + `retained_all_features` force-activates on small multi-feature clean problems.** Activation OR includes `reason.startswith("retained_all_features")` whenever `soft_w is not None`. Probe: 3-feature clean linear, soft nearly uniform (min≈0.84), `out_frac=0`, blackbox reason `retained_all_features_small_problem` → would activate soft+Huber. Interaction targets also get `out_frac>0.05` from linear-only residual probes. | fit activate block ~7400–7410; `_auto_residual_soft_weights` only uses x0 poly basis for multi-X | fixed |
| N4 | **P1** | correctness gap | **Phase-3/6 unweighted guards inactive under pure `diffuse_noise_huber`.** `_auto_noise_guard_active` only true for `soft_mad_weights` / `auto_soft_mad` source — not for diffuse Huber. So N2 false-positive Huber path also skips complexity/holdout disaster guards and Phase-6 parsimony. Notes already said “intentionally inactive”; combined with N2 this is a real clean-path risk. | `_auto_noise_guard_active` ~3860–3873; residual skip ~7044 only soft path | fixed |
| N5 | **P1** | metric / early-stop | **C++ early-stop / selection objective can be robust loss while `early_stop_mse` threshold is MSE-scaled.** Under Huber, `objective_mse` uses Huber mean (can be ≪ unweighted MSE with outliers) so “exact” early-stop may fire on robust objective while raw MSE is still large. Wrapper reads `best_mse` as **unweighted** `raw_mse` (good), but search already stopped/mutated under robust objective. | `evolution.h` `objective_mse` / early_stop ~412,700; `core.cpp` best_mse=raw_mse ~676–678 | fixed |
| N6 | **P1** | contract risk | **Display MSE fallback breaks unweighted protocol silently.** `_display_formula_mse` imports `scripts.benchmark_common`; on any failure returns `inf`. `_final_formula_score` then falls back to **internal** `_formula_mse` (weighted / Huber / trimmed). Residual/inception acceptance and some cleanups then optimize the search objective under a “display” name. | `_display_formula_mse` 3561–3577; `_final_formula_score` 3579–3594 | fixed |
| N7 | **P2** | parity | **Huber δ (MAD scale): Python can use weighted MAD; C++ `mad_scale` ignores `y_weights`.** Formulas for Huber/student_t match (0.5 r² / linear; log1p). Weighted MSE / weighted mean of loss match. Trimmed keep-k count logic matches. Small δ mismatch only when both weights and auto-δ are on. | Python `_mad_scale` 819–852; C++ `mad_scale` 797–827 | open |
| N8 | **P2** | API / docs | **`_validate_sample_weight` docstring claims empty → None; code raises length mismatch.** NaN / negative / all-zero correctly raise. Partial zeros allowed (re-normalized). | `_validate_sample_weight` 65–90 | open |
| N9 | **P2** | coverage gap | **Auto robust path skips multi-feature when blackbox is off** (`want_robust` requires 1D or multi blackbox). 2D+ tabular with `blackbox_mode=False` never auto soft/Huber even if `blackbox_noise_robust=True` unless forced via other branch (forced True only in `elif` when want_robust false — actually `robust_mode is True` is inside want_robust). Wait: `want_robust` includes `robust_mode is True` regardless of dim — OK for forced True. Only `auto` is dim-gated. | fit ~7372–7380 | open (document; maybe intentional) |
| N10 | **P2** | maintainability | **`sample_weight_provided_` is True for auto soft-MAD**, not only user weights. Downstream “provided” means “weights active”. Diagnostics `source` distinguishes user vs `auto_soft_mad`. Easy to misuse in new code (see S1-3 sticky + guard pool). | fit ~7408–7409; diagnostics ~7562–7574 | open |

### What is solid (checked OK)

| Area | Result |
|------|--------|
| Invalid weights raise | length / NaN / negative / all-zero → `ValueError`; no silent drop (`test_validate_sample_weight_rejects_invalid`) |
| User `sample_weight` not overridden | `want_robust` requires `not sample_weight_provided_` |
| User `loss_mode != mse` not overridden | Phase 4 checks `_user_loss_mode_ == mse` |
| Holdout weight slicing | `_slice_sample_weight` / `_split_sample_weights` / `_formula_mse(..., sample_weight_indices=)` raise on mismatch |
| Soft-MAD on clean **polynomials** | residual probe → ratio≈0, soft=None; `test_auto_soft_weights_skip_clean_1d`, `test_phase4_clean_stays_mse` |
| Sticky public `loss_mode` | N1 fixed; restore on normal fit exits |
| C++ weighted MSE algebra | `sum(w r²)/sum(w)`; rejects bad weights; `best_mse` unweighted for benchmarks |
| Display path intent | documented unweighted; Pareto keeps unweighted diagnostics alongside weighted val |
| Existing unit tests | 45 passed (`test_robust_loss` + `test_sample_weight_contract` + `test_phase6_noise_guards`) |

### Threshold notes (N2/N3)

| Signal | Threshold | Observed |
|--------|-----------|----------|
| Diffuse ratio → Huber | `>= 0.02` | poly+1% noise ≈0.009 (off); poly+2% ≈0.021 (on); **clean sin ≈1.18 (on — false)** |
| Soft out_frac 1D | `>= 0.01` | clean poly 0; sparse outliers high |
| Soft out_frac multi | `>= 0.02` | clean interaction often >0.05 via underfit probe |
| retained_all force | soft_w not None | small multi clean can force nearly-uniform soft + Huber |

### Suggested fix direction (not applied this pass)

1. **N2:** Diffuse ratio only when residual looks *noise-like* (low lag-1 autocorr, or residual after a stronger probe: trig/exp templates, or fast-path residual if available). Raise floor and/or require `residual_autocorr` low + ratio high. Never treat “structure mismatch vs poly” as noise.
2. **N3:** Drop bare `retained_all_features` force; require `out_frac` / ESS / low_weight_mass evidence. Multi-feature residual probes should use multi-linear (all columns) not x0-only.
3. **N4:** Enable a lighter unweighted complexity/holdout guard whenever auto Huber is active (soft or diffuse), or at least when `loss_mode_switched_to_huber`.
4. **N5:** Early-stop on **raw unweighted MSE** (or dual: robust for selection, raw for exact claims).
5. **N6:** Local unweighted MSE fallback in `_display_formula_mse` (no scripts dependency); never fall back to robust internal for “display” decisions.
6. **N7:** Pass weights into C++ MAD or document auto-δ unweighted-only.

### Related tests / gaps

| Covered | Missing |
|---------|---------|
| poly clean vs 10% gauss Huber | clean **sin/exp/rational** must stay mse (N2 regression) |
| user loss_mode / sticky loss_mode | multi-feature retained_all soft force (N3) |
| guard active only with soft_mad | guard should also protect diffuse Huber (N4) |
| weight length / slice / ESS | display fallback without `scripts` (N6) |

### Notes

- Phase 0–6 plumbing mostly sound after N1; **N2 is the largest remaining correctness issue** for default 1D SR (trig/exp families).
- Overlaps S1-6 (weighted vs unweighted decision metrics) and S3 (guards/parsimony). S4 should re-check early-stop under robust loss (N5).
- `blackbox_noise_robust="auto"` is default — N2/N3 fire without user opt-in beyond defaults.

---

## S3 — Scoring, refine, cleanup, guards

**Paths**
- `sklearn_wrapper.py` — `_score_formula_candidate`, `_refine_formula_constants`, snap/cleanup helpers, `_apply_auto_weight_final_guard`, `_phase6_noise_parsimony_pass`, residual/evolution promotion blocks

**What it owns**
- How winners are chosen and rescued
- Complexity / holdout R² / gap caps under auto weights
- Post-evolution polish and parsimony

**Analysis focus**
- [ ] Guard active only on auto path (not user weights)
- [ ] Complexity caps too tight/loose by problem class
- [ ] Residual promotion bloat under noise
- [ ] Clean-path regressions from noise guards
- [ ] Constant snap changing structure identity incorrectly
- [ ] Expensive refine loops with little gain

**Findings**

| ID | Severity | Type | Summary | Status |
|----|----------|------|---------|--------|
| — | — | — | *(none yet)* | — |

---

## S4 — C++ evolution engine

**Paths**
- `glassbox/sr/cpp/evolution.h` (~3666 lines) — primary
- Cross-check: `core.cpp` (`run_evolution` bindings), `execution.h` (OpenMP pop eval), `eval.h` (graph eval + soft arithmetic), `ast.h` (IndividualGraph / active_complexity)
- Python caller: `GlassboxRegressor.fit` Stage 2; tests: `glassbox/sr/test_cpp_parity.py`

**What it owns**
- Island GA, mutations (Lamarckian / macro), population init, early stop
- `y_weights` / loss_mode integration in search fitness
- Complexity penalty, multi-config beams, threading

**Analysis focus**
- [x] Fitness uses weights consistently (or not) with Python scoring
- [x] Seed graph capacity / fraction of population used
- [x] Race conditions under `num_threads`
- [x] Representation bias (additive weighted sum of nodes)
- [x] Early-stop at MSE threshold vs noise-aware stopping
- [x] Redundant full-population evals
- [x] Mutation operators that explode complexity under noise

**Status:** `done` (2026-07-19) — engine + island/threading + fitness/seed audit; runtime probes on `_core`

### Architecture (condensed)

```
run_evolution (core.cpp)
  → EvolutionEngine(config, X, y, seed_omegas, seed_graphs, y_weights)
  → if num_islands>1 && island_size>=4: run_islands()
       parallel init+refine islands (OpenMP outer × inner)
       for gen: parallel evolve_one_generation; ring migration; early-stop check
  → else: run()
       initialize_population (seeds ≤ pop/4)
       loop: evaluate_population → select/reproduce (elite+crossover+macro+Lamarckian+explorers)
             refine_constants / refine_inner_params periodically
  → cleanup_graph(best); return formula + raw_mse + weighted_mse
```

**Representation:** DAG of Input/Constant/Unary/Binary nodes; prediction =  
`bias + Σ_i w_i * node_i(x)` (soft arithmetic blend inside Binary Arithmetic).  
Linear outer layer is ridge / IRLS-refit each refine (`solve_output_weights`).

### Findings

| ID | Severity | Type | Summary | Evidence | Status |
|----|----------|------|---------|----------|--------|
| E1 | **P0** | bug / waste | **Island model with fixed `random_seed` clones islands.** `run_islands` builds each `EvolutionEngine` with the **same** `config.random_seed` and full `seed_graphs_` — no per-island seed offset. With empty `multi_*_priors` (default sklearn path, `num_islands=8`), islands run identical trajectories until migration, and identical migrants keep them clones. Fixed-seed multi-island is deterministic but **does not diversify**; pays ~N× eval cost for ~1× search. Verified: multi-island det with seed=99; no per-island seed mutation in ctor loop ~619–624. | `run_islands` 579–624; wrapper default `num_islands=8` + `random_seed=run_seed` | fixed |
| E2 | **P0** | correctness (S2 N5) | **Early-stop / “exact” metrics use robust `objective_mse`, not raw MSE.** `objective_mse` returns `weighted_mse` (Huber/trimmed/student_t/weights). Early-stop, `is_exact`, `is_acceptable` all gate on that + node count. Under auto-Huber (incl. N2 false positives), search can stop as “exact” while `raw_mse` remains large. Python reads `best_mse=raw_mse` (good for report) but evolution already halted/mutated under robust objective. Complexity penalty also relaxes when `mse < 1e-6` on **search** mse (~1437–1439). | early_stop ~412,700; `update_discovery_metrics` 961–965; `objective_mse` 785–793; core.cpp best_mse=raw 676–678 | fixed |
| E3 | **P1** | seed capacity | **Seeds capped at `pop_size/4` and only first N graphs.** `max_seed = min(seed_graphs.size(), pop_size/4)` (~1296). Islands use **island** pop (`pop/num_islands`), so with 8 islands and pop=100, island_size=12 → **only 3 seeds** per island despite up to 24 built in Python. Remaining seeds discarded; no rotation/shuffle of seed list. Oversized seeds skipped in core.cpp (OK, tested). | `initialize_population` 1290–1306; island_size 582–583 | fixed |
| E4 | **P1** | representation bias | **Kitchen-sink additive basis.** Outer linear combo of **all** node outputs encourages many weak terms; active_complexity only counts \|w\|>1e-4. Soft Arithmetic Binary is continuous mixture (eval.h temperature), so structure is not discrete during search — favors flexible fits over sparse algebraic form. Max nodes hard-capped at 24 (mutate/macro/crossover) — good anti-bloat, but soft blend + multi-term sum still bloats **formula strings** after simplify. | `eval.h` 260–266; `active_complexity` ast.h 140–176; max_nodes 52, macro 1556 | partial |
| E5 | **P1** | concurrency | **`omp_set_num_threads` inside parallel island regions.** Outer `#pragma omp parallel for` then each thread calls `omp_set_num_threads(inner_threads)` (~646–647, 665–666). Global OpenMP max-threads is process-wide → race / oversubscription risk with nested OMP + concurrent Python multi_start. `omp_set_nested(1)` enabled. Pop eval uses thread-local SubtreeCache merge (sound). | `run_islands` 629–669; `execution.h` 32–54 | fixed |
| E6 | **P1** | inefficiency | **Full re-evaluation of population every generation** including elites already scored; children evaluated at birth then again next gen. `refine_constants` re-eval + IRLS; Adam/LM finite-diff costs many graph evals per elite (every 10 gens × top 5). Island clones (E1) multiply all of this × num_islands. | `run` 323, 458–466; `refine_inner_params_adam` 2118–2230 | fixed |
| E7 | **P1** | selection / noise | **`best_overall_` tracked by penalized `fitness`, not raw_mse.** Under weights/Huber, a simpler high-raw-MSE model can win fitness vs a better unweighted structure (or vice versa depending on penalty). Post-run `cleanup_graph` re-optimizes; still no dual “best raw” archive. | track best ~378–380, 703–704 | fixed |
| E8 | **P2** | mutation / bloat under noise | Macro wrap/multiply/divide add nodes (15% offspring + explorers); max_nodes=24 hard stop. Under noisy labels, search_obj can improve by adding weak basis nodes; parsimony is multiplicative 1.2% per active complexity unit — may be weak when robust loss compresses outliers. Nest macro can create f(g(x)) compositions. | macro_mutate 1550–1705; complexity_penalty 1430–1446 | open |
| E9 | **P2** | API / config | **`elite_size` not exposed in `run_evolution` bindings** — always config default 10; islands set `elite_size = max(2, 10/num_islands)`. Parent sampling for macro/crossover is **elite-only** (`parent_dist(0, elite_size-1)`), tournament only for mutation/explorers — strong elitism, less diversity. | core.cpp pop_size only; run 432–436 | open |
| E10 | **P2** | global state | **`arithmetic_temperature` is process-global** (`eval.h` static). Concurrent `run_evolution` calls with different temperatures race. Islands share same temperature (OK). | `set_arithmetic_temperature` eval.h 21–28; core.cpp sync | fixed |

### What is solid (checked OK)

| Area | Result |
|------|--------|
| Weight validation | C++ rejects non-finite / negative / zero-sum `y_weights` |
| Dual metrics | `raw_mse` unweighted; `weighted_mse` = search objective; returned both |
| Huber algebra | Matches Python (0.5 r² / linear); IRLS in `solve_output_weights` for robust modes |
| Seed graph size guard | Oversized seeds skipped (`test_oversized_seed_graphs_are_skipped`) |
| Single-pop determinism | Same `random_seed` → identical formula/mse (`test_random_seed_determinism`) |
| Thread-local eval caches | No lock during parallel fitness; serial merge |
| max_nodes hard cap | Mutation/crossover/macro refuse growth past 24 |
| Anti-trig bloat | Hard +100 fitness if ω≈0 or nested periodic |
| y_weights passed to islands | Same weights on all islands (correct) |
| NSGA-II (optional) | Dominates on **raw_mse**, complexity, age — better metric hygiene than SO path; default off |

### Relation to prior sections

| Theme | Link |
|-------|------|
| N5 / robust early-stop | **E2** |
| Seed graphs unused historically | Partially fixed (wired); **E3** capacity still low under islands |
| Double evaluation | **E6** + S1-7 |
| Weighted vs unweighted | Fitness = robust; best_mse report = raw; **E2/E7** |
| Default multi_start × islands | S1-9 × **E1** → extreme wall-time |

### Suggested fix direction (not applied)

1. **E1:** `current_island_cfg.random_seed = base_seed + i * large_prime` (or `seed_seq`); optionally shard seeds across islands.
2. **E2:** Early-stop / exact / acceptable on **`raw_mse`** (or require both raw and search obj below thresholds). Keep robust obj for ranking only.
3. **E3:** `max_seed = min(seeds, max(4, pop/2))`; shuffle seeds with rng; distribute disjoint seed subsets per island.
4. **E5:** Avoid `omp_set_num_threads` inside parallel regions; use `num_threads` clause only / `omp_set_max_active_levels`.
5. **E6:** Skip re-eval of unchanged elites; cache fitness dirty bit.
6. **E7:** Track `best_raw_overall_` separately for return value vs penalized champion.

### Runtime probes (2026-07-19)

- Single-island seed=7 determinism: **pass**
- Multi-island seed=99 determinism: **pass** (clones still det)
- `num_islands=4` reports `island_outer_threads=4`, inner≥1
- Existing parity tests cover weights, huber mode smoke, oversized seeds

### Notes

- Deep mutation operator correctness / NaN domains deferred to **S5** (`eval.h`).
- Binding defaults / seed_graph_builder schema deferred to **S6**.
- Default production path: Python `num_islands=8` + `random_state` set → **E1 is default-on**.

---

## S5 — C++ eval / AST / refine / simplify

**Paths**
- `glassbox/sr/cpp/ast.h`
- `glassbox/sr/cpp/eval.h`
- `glassbox/sr/cpp/refine.h`
- `glassbox/sr/cpp/simplify.h`
- `glassbox/sr/cpp/simplify_advanced.h`
- `glassbox/sr/cpp/formula_parser.h`
- `glassbox/sr/cpp/execution.h`
- Cross-check: `core.cpp` exact scorer (`evaluate_parse_node_exact`), evolution refine/snap/cleanup call sites

**What it owns**
- Graph/AST representation, vectorized eval, local coeff refine, algebraic simplify, string ↔ graph
- Dual math paths: **soft search graph** (`evaluate_graph*`) vs **exact string scorer** (`parse_formula_exact` / `evaluate_parse_node_exact` in `core.cpp`)

**Analysis focus**
- [x] Numerical stability (log/div/exp domains)
- [x] NaN/Inf propagation into fitness
- [x] Simplify changing semantics (esp. with affine outer layers)
- [x] Parser / printer round-trip fidelity
- [x] Eval allocation patterns (per-candidate malloc)
- [x] Refine not using sample weights when search did

**Findings**

| ID | Severity | Type | Summary | Evidence / locus | Status |
|----|----------|------|---------|------------------|--------|
| **S5-1** | **P0** | correctness | **`abs(x)` graph-compiles to identity.** `ParseNodeType::Abs` → `UnaryOp::Power` with `p=1.0`. Power eval is sign-preserving `sign(x)*\|x\|^p`, so p=1 is **x**, not \|x\|. `simplify_formula_cpp("abs(x)")` → `"x"`; seed graph from abs is wrong for evolution. Exact scorer path still evaluates true abs → dual-path split (S5-2). | `formula_parser.h` Abs case (~460–465); Power eval `eval.h` 183–190; verified: simplify abs → x; score(`abs`) MSE=0 vs \|y\| while simplified string MSE≈0.34 | fixed |
| **S5-2** | **P0** | correctness / contract | **Two incompatible evaluators for the same formula strings.** `score_formula_candidates` uses exact parse-tree math (true `/`, true `abs`, variable exponents, unprotected div). Evolution/search + `formula_to_graph`/`get_formula_string` use soft Arithmetic (soft-div `x/sqrt(1+y²)`), soft Division, Power domains, etc. Candidate ranking can prefer formulas whose **search graph** cannot represent / will not match scored MSE after seed compile. | `core.cpp` 21–90, 235–237 vs `eval.h` 214–226 + `formula_to_graph` | fixed |
| **S5-3** | **P0** | correctness | **Variable-exponent powers silently become `x^1`.** Non-constant RHS of `Pow` compiles to `UnaryOp::Power` with **`p=1.0`** (RHS discarded). `x0^x1` seed graph is identity-like; simplify → `x`. Exact scorer supports `base**exp` per-sample. Seeds/priors with `x_i^{x_j}` are corrupted. | `formula_parser.h` Pow else-branch (~515–520); verified seed `x0^x1` → p=1 | fixed |
| **S5-4** | **P0** | silent wrong display | **Printer ↔ evaluator mismatch for soft ops.** (1) Non-discrete Arithmetic blend: eval uses soft-div `x/sqrt(1+y²)` but blend printer emits true `(l / r)`. (2) Hard `BinaryOp::Division` prints `(l / r)` but eval is `x/(\|y\|+ε)*sign(y)`. (3) `Aggregation` always prints `(l+r)/2` but eval is soft-max / soft-mean via `tau`. Displayed evolution formulas can disagree with search fitness and with Python/`_display_formula_mse` string eval. | `eval.h` 214–234, 536–571; Aggregation display ~570 | fixed |
| **S5-5** | **P1** | correctness | **Subtree cache hash quantizes params to 2 decimals.** `quantize(v, decimals=2)` + SharedCache reuse means e.g. ω=1.004 and 1.006 share a cached ArrayXd → **wrong node values** during pop eval when hashes collide. Also used for simplify CSE / trig identities / mul→square via equal hashes — near-equal different subtrees can be treated identical. | `ast.h` 137–144, 160–176; SharedCache path `eval.h` 141–147, 249–252; simplify_advanced hash merges | fixed |
| **S5-6** | **P1** | silent incomplete formula | **Output-weight thresholds disagree.** Eval includes \|w\|>**1e-6**; `get_formula_string` / `active_complexity` use **1e-4**; compact/simplify often **1e-8**. Terms with 1e-6 < \|w\| ≤ 1e-4 affect predictions/fitness but are **omitted from formula string** returned to Python. | `eval.h` 262; `get_formula_string` ~604; `ast.h` active_complexity 82; `simplify.h` compact 1e-8 | fixed |
| **S5-7** | **P1** | correctness | **`exp(log(·))` simplify → identity, not abs.** Graph Log is `log(\|x\|+ε)`; identity folds Exp∘Log to Power p=1 (identity) or redirects Log∘Exp to child. For signed inputs, true composition is \|x\| (approx), not x. Same root cause as Abs mapping (S5-1). | `simplify_advanced.h` 149–160; verified `exp(log(x))` → `x` | fixed (via Abs) |
| **S5-8** | **P1** | refine incompleteness | **Inner param refine/snap only touch unaries with nonzero *output* weight.** Nested bases (e.g. `sin(ω·f(x))` where f is internal unary, or power inside a binary that is the output basis) never enter Adam/LM/snap loops. Kitchen-sink outer layer assumption breaks deep graphs. | `evolution.h` refine_inner_params_adam active_unary ~2131–2138; snap skip ~2725–2726 | fixed |
| **S5-9** | **P1** | weights gap | **`refine.h` specialist refiners are unweighted only.** `elastic_net_*`, `refine_frequencies_cpp`, `refine_powers_model_cpp`, `refine_periodic_rational_cpp` all use plain SSE / unweighted QR. No `y_weights` / Huber. Under soft-MAD or sample_weight search, Python specialists using these C++ helpers optimize a **different objective** than the main engine (`solve_output_weights` is weight-aware in evolution.h). | entire `refine.h`; contrast evolution ridge/IRLS with weights | partial |
| **S5-10** | **P1** | numerical / scoring | **Exact scorer uses unprotected division.** `left/right` with no ε → Inf/NaN near zeros (`ok=False`, nonfinite). Graph Division is soft. Same formula can be perfect in search-space soft div and invalid in exact score (or vice versa). Exact path clamps exp arg to ±500; graph clamps **output** of exp to ±1e6 (different). | `core.cpp` 41–45, 74; `eval.h` 198–201, 226 | open |
| **S5-11** | **P1** | inefficiency | **`get_child` returns `Eigen::ArrayXd` by value** → full sample-vector copy per child access per node. Simple/CacheOut/Shared paths pay O(nodes × n_samples) extra traffic. Should return const refs / Map / block expressions. Thread-local arena also never shrinks when n_samples drops. | `eval.h` 159–162, 111–116 | fixed |
| **S5-12** | **P1** | simplify temp drift | **Basic `simplify.h` hardcodes arithmetic temperature `t=5.0`** for constant fold of soft Arithmetic, ignoring `get_arithmetic_temperature()`. Advanced simplify correctly uses `arithmetic_soft_weights`. Fold results diverge from live eval if Python changes temperature. Aggregation never constant-folded in either simplifier. | `simplify.h` ~130; vs `simplify_advanced.h` ~189 | fixed |
| **S5-13** | **P2** | complexity metric | **`active_complexity` is a weighted cost, not node count**, but evolution uses `nodes.size() - active_complexity` as “inactive_nodes” for parsimony — unit mismatch (can under/over-penalize). | `ast.h` 78–123; `evolution.h` ~1430–1435 | fixed |
| **S5-14** | **P2** | API / parser | **No implicit multiplication; limited grammar.** `2x`, `2(x+1)`, `sin x` fail. Empty string simplify → empty. `\|…\|` bars normalize to `abs(...)` then hit S5-1. Integer Power display `(x)^n` without abs while non-integer uses `sign/abs` — Python string eval of even integer powers can differ from graph on negative bases for `UnaryOp::Power` (even path uses \|x\|^n). | `formula_parser.h` tokenize/normalize; `eval.h` Power format ~488–492 | open |
| **S5-15** | **P2** | partial eval safety | **`evaluate_graph_partial` indexes `changed[left_child]` without bounds checks**; empty X / num_samples=0. Safe only if topology always valid and only params of existing nodes change (current evolution use). Malformed child indices → UB. | `eval.h` 127–136, 272–277 | fixed |
| **S5-16** | **P2** | dead / dual code | **Legacy `evaluate_fitness` only sets `fitness`**, not `raw_mse`/`weighted_mse`. ParallelExecutionEngine builds gen_cache but evolution also maintains its own cache path — dual caching complexity. `simplify.h` largely superseded by advanced but still used (`snap_formula_floats`). | `eval.h` 291–296; `execution.h`; `core.cpp` snap | open |

**False-positive checks**
- Soft Arithmetic is intentional continuous relaxation during search (also noted E4); S5-4 is about **printer lying** relative to that eval, not about using soft ops per se.
- Exact vs graph dual path is partly intentional for display scoring; S5-2 flags **seed compile / simplify** still going through lossy graph mapping, not the existence of an exact scorer.
- Reduce-noise / BIC path in `simplify_advanced` **does** support optional `y_weights` + unweighted holdout guard — good; not filed as gap. Specialist `refine.h` remains the weight hole (S5-9).
- `sin²+cos²→1` identity works when amp=ω=1 (tests pass); limited by hash quantize + amp/ω gates (covered under S5-5 constraints).

**Tests touched / used**
- `tests/test_cpp_simplification.py` (basic simplify, pythagorean, float snap) — does **not** cover abs identity bug
- `tests/test_cpp_candidate_scoring.py` (exact scorer weights) — orthogonal to graph path
- Manual probes: abs simplify destruction; x0^x1 → p=1; dual score after simplify

**Notes**
- `execution.h` is thin OpenMP pop wrapper with thread-local SubtreeCaches then serial merge — design OK; inherits S5-5 collision risk.
- Display protocol should keep using exact-ish string eval; search should not export soft-blend strings without discretization (ties to S3 cleanup).

## S6 — C++ bindings & seed graphs

**Paths**
- `glassbox/sr/cpp/core.cpp`
- `glassbox/sr/cpp/seed_graph_builder.py`
- `glassbox/sr/cpp/setup.py`
- `glassbox/sr/cpp/export_pytorch.py`
- Parity tests: `glassbox/sr/test_cpp_parity.py`, `glassbox/sr/test_seed_graph_builder.py`

**What it owns**
- Python ↔ C++ surface (`run_evolution`, scoring helpers)
- Building seed graphs from formulas / priors
- Extension build flags

**Analysis focus**
- [ ] Arg defaults mismatch Python vs C++
- [ ] Optional `y_weights` / `loss_mode` plumbing gaps
- [ ] seed_graphs_py not passed from some call sites (known historical gap)
- [ ] dtype / contiguity / row-major assumptions
- [ ] Error handling: silent empty results vs exceptions
- [ ] Build/ABI notes (cpython 3.12/3.14)

**Findings**

| ID | Severity | Type | Summary | Status |
|----|----------|------|---------|--------|
| — | — | — | *(none yet)* | — |

---

## S7 — Blackbox multi-feature prep

**Paths**
- `glassbox/sr/blackbox_preprocessor.py`

**What it owns**
- Feature ranking, standardization, search-space reduction
- Interaction discovery seeds
- Remap formulas reduced ↔ original space

**Analysis focus**
- [ ] Dropping true active variables
- [ ] Scale/mean bookkeeping bugs on inverse map
- [ ] Noise-robust ranking
- [ ] Cost of ExtraTrees / MI vs benefit
- [ ] Determinism across seeds

**Findings**

| ID | Severity | Type | Summary | Status |
|----|----------|------|---------|--------|
| — | — | — | *(none yet)* | — |

---

## S8 — Specialist vault & state

**Paths**
- `glassbox/sr/specialist_state.py`
- Wrapper hooks in `sklearn_wrapper.py` (`_compose_specialist_*`, vault seed/update)

**What it owns**
- Cross-run specialist memory, composition proposals, screening diagnostics

**Analysis focus**
- [ ] Vault poisoning under noisy false positives
- [ ] Composition explosion / complexity bloat
- [ ] State leakage between unrelated fits
- [ ] Compute cost of screening many candidates

**Findings**

| ID | Severity | Type | Summary | Status |
|----|----------|------|---------|--------|
| — | — | — | *(none yet)* | — |

---

## S9 — Curve classifier stack

**Paths** (review priority)
- `glassbox/curve_classifier/curve_classifier_integration.py` (**first**)
- `glassbox/curve_classifier/models.py`
- `glassbox/curve_classifier/validation.py`
- `glassbox/curve_classifier/rollout.py`
- Feature extract used at inference: portions of `generate_curve_data.py` (`extract_all_features*`)
- De-prioritize for audit: `train_curve_classifier.py`, bulk of `generate_curve_data.py` training set synth
- `glassbox/model_registry.py` for path resolution

**What it owns**
- Fast-path operator / skeleton proposals that warm-start search

**Analysis focus**
- [ ] Feature extraction cost on every fit
- [ ] Fail-open vs fail-closed when model missing
- [ ] Calibration / overconfidence → false exact claims
- [ ] Duplicate feature logic paths
- [ ] Device / torch overhead for tiny X

**Findings**

| ID | Severity | Type | Summary | Status |
|----|----------|------|---------|--------|
| — | — | — | *(none yet)* | — |

---

## S10 — Proposer, ops graph, optimizers, Python evolution

**Paths**
- `glassbox/universal_proposer/universal_proposer.py`
- `glassbox/sr/fpip_v2.py`
- `glassbox/sr/core/operation_node.py`, `operation_dag.py`
- `glassbox/sr/operations/meta_ops.py`
- `glassbox/sr/optimizers/bfgs_optimizer.py`, `hybrid_optimizer.py`
- `glassbox/evolution/evolution.py` (large; treat as secondary if C++ path dominates production)
- `glassbox/sr/phased_regression.py`, `pruning.py`, `hard_concrete.py`, `risk_seeking_policy_gradient.py`
- `glassbox/sr/visualization.py` (low priority unless correctness of reported formulas)

**What it owns**
- Neural priors / FPIP handoff, alternate Python search stacks, local optimizers, DAG representation

**Analysis focus**
- [ ] FPIP schema validation vs consumers
- [ ] Dead alternate stacks still paid for at import/fit
- [ ] Optimizer failure modes (non-finite grads)
- [ ] Python evolution still used vs fully superseded by C++
- [ ] Duplication between Python and C++ simplify/score

**Findings**

| ID | Severity | Type | Summary | Status |
|----|----------|------|---------|--------|
| — | — | — | *(none yet)* | — |

---

## Cross-cutting themes (fill as sections complete)

| Theme | Sections | Notes |
|-------|----------|-------|
| Weighted vs unweighted metric split | S2, S3, S4, S5 | Search may weight; display/protocol must not silently mix — **N5/N6** early-stop + display fallback gaps; **S5-9** refine.h specialists unweighted |
| Soft-graph vs exact-string dual eval | S5, S3, S6 | **S5-1..S5-4, S5-10** — score/display exact math ≠ evolution graph; printer mismatches soft ops |
| Subtree hash quantization | S5, S4 | **S5-5** 2-decimal param hash can corrupt SharedCache + simplify CSE |
| Seed graphs unused | S1, S4, S6 | Wired from wrapper; **E3** capacity only pop/4 (worse under islands); oversized skip OK |
| Complexity under noise | S2, S3, S4 | Guard + parsimony + mutation pressure |
| Double evaluation | S1, S3, S4 | Same formula scored in Python and C++ repeatedly; **E6** full pop re-eval every gen + island clones (**E1**) |
| Stochastic local optima | S4, S9, S10 | Same target different seeds / ranges |
| Multi-feature scale remap | S1, S7 | Structure recovery in original space |
| Sticky estimator hyperparameters | S1, S2 | loss_mode auto-switch must restore (fixed N1); **S1-3** sticky evolution/Pareto attrs across fits still open |
| sklearn fit contract | S1 | **S1-1** check_is_fitted false-positive; **S1-4** n_features_in_ under blackbox |
| Multi-start / double pipeline work | S1 | **S1-7** double blackbox/fast-path/structure; default multi_start_runs=3 |

---

## Optimization backlog (global)

| ID | Section | Idea | Expected impact | Status |
|----|---------|------|-----------------|--------|
| O1 | S1 | Default `multi_start_runs=1`; escalate only when R²/uncertainty poor | Large wall-time cut on easy/medium problems | done |
| O2 | S1 | Cache / single-shot `prepare_blackbox_search` when soft weights applied (reweight ranking without full redo where possible) | Avoid 2× feature ranking cost | done |
| O3 | S1 | Skip second fast-path when first result already compact / high confidence | Save exact-match + basis expand cost | done |
| O4 | S1 | Shared formula-eval cache that is process-safe OR disable cache inside ThreadPool workers | Correctness + less lock contention | fixed |
| O5 | S1 | Track search feature count separately; never flip public `n_features_in_` mid-fit | Cleaner predict/metadata; fewer remap bugs | fixed |
| O6 | S2 | Structure-aware diffuse-noise detector (not poly residual only) | Stop auto-Huber on clean trig/exp | done |
| O7 | S2 | Multi-column residual probe for soft-MAD; remove retained_all force | Fewer false soft weights on clean multi-feature | done |
| O8 | S2 | Local unweighted display MSE (no `scripts` import) | Hard display/search separation | done |
| O9 | S4 | Offset island RNG seeds; shard seed_graphs across islands | True island diversity; stop N× clone work | done |
| O10 | S4 | Early-stop / exact on raw_mse (search still uses robust obj) | Correct Exact under noise/Huber | done |
| O11 | S4 | Raise seed cap; skip re-eval of clean elites | Better seed use + wall-time | done |

---

| O12 | S5 | Fix Abs→Power(p=1); map abs to true abs (new op or even power pattern); fix variable Pow compile | Stops abs/seed/simplify destruction | done |
| O13 | S5 | Align printer with eval (soft-div string, aggregation, hard-div); discretize before export | Formula strings match search fitness | done |
| O14 | S5 | Raise structural hash quantize decimals or include full-bit params for SharedCache | Stop silent wrong cached evals | done |
| O15 | S5 | Unify output-weight cutoffs (display ≥ eval threshold or refit-then-print) | No silent dropped terms | done |
| O16 | S5 | `get_child` by const ref / Map; avoid ArrayXd copies in eval hot path | Large wall-time win on pop eval | done |
| O17 | S5 | Weight-aware refine.h or route specialists through evolution residual_mse | Noise/sample_weight consistency | open |

## Progress log

| Date | Section | Action |
|------|---------|--------|
| 2026-07-18 | — | Tracker created; 10 sections defined |
| 2026-07-18 | S2 | Accidental delete during noise-doc cleanup; **restored/recreated** |
| 2026-07-18 | S2 | N1 sticky `loss_mode` found + fixed during noise correctness audit |
| 2026-07-19 | S1 | Full fit orchestration audit; **13 findings** (S1-1..S1-13); section marked done; fixes not applied |
| 2026-07-19 | S2 | Full noise/weight/loss audit; **N1 fixed + N2–N10 open** (N2 diffuse Huber FP on clean sin/exp is P0); section marked done |
| 2026-07-19 | S4 | C++ evolution audit; **E1–E10** (E1 island clone under fixed seed = P0; E2 robust early-stop); section marked done |
| 2026-07-19 | S5 | C++ eval/AST/refine/simplify/parser audit; **S5-1..S5-16** (P0: abs→identity, dual evaluators, variable pow drop, printer/eval mismatch); section marked done; fixes not applied |
| 2026-07-19 | S5 | **S5-1/S5-2 fixed**: `UnaryOp::Abs`; scorer uses `formula_to_graph`+`evaluate_graph` (OOB feature check, sharp soft-arith temp); tests updated |
| 2026-07-19 | S5 | **S5-3/S5-4 fixed**: variable powers via const-fold + exp/log rewrite; printer matches soft-div/protected-div/aggregation; multi-feature print auto-detect |
| 2026-07-20 | Phase0 | **P0 fixes:** S1-1 NotFittedError on formula_; S1-2 rapid-hit argmin; S1-3 sticky fit clear; N2 structure-aware diffuse ratio; E1 island seed offsets; E2/N5 early-stop on raw_mse; tests/test_phase0_correctness.py |
| 2026-07-20 | Phase1 | **N3/N4/N6/S1-6/S1-12/S5-9(partial) fixed**: drop retained_all soft force + multi-linear residual probes; auto guards for diffuse Huber; local unweighted display MSE (no robust fallback); search vs display metric contract docs; weighted iterative elastic net via sqrt(w); `tests/test_phase1_noise_metrics.py` |
| 2026-07-20 | Phase2 | **E3/E5(OMP)/E7(raw champion)/E10/E4(partial)**: seed capacity seed_fraction=0.5 (tiny-pop almost-all); raw_mse tie-break + dual best_raw archive/export; no omp_set_num_threads inside island parallel (eval_num_threads + max_active_levels); ScopedArithmeticTemperature + per-eval config temp; kitchen-sink/soft-arith documented; `tests/test_phase2_evolution_reliability.py` |
| 2026-07-20 | Phase3 | **S5-5/6/8/11/12/13/15 fixed**: hash quantize 8dp; unified kOutputWeightActive=1e-6 for eval/print/active_complexity; nested unary refine/snap; get_child no ArrayXd copy; simplify temp+aggregation fold; active_node_count for parsimony; partial-eval bounds; `tests/test_phase3_graph_eval.py` |
| 2026-07-22 | Phase5 | **S1-9/S1-7/E6 fixed**: multi_start_runs default 1 + auto-escalate; skip soft-weight blackbox re-rank when stable; skip 2nd fast-path on high confidence; reuse exact structure probe; IndividualGraph.fitness_valid elite/child skip re-eval; `tests/test_phase5_performance.py` |
| 2026-07-21 | Phase4 | **S1-4/5/8/10/13 fixed**: public `n_features_in_` vs `n_features_search_`; once-per-fit selection holdout; thread-safe formula cache + local RNG (no global seed); CV skip guard residual-stability + fail-closed small n; residual boosting decoupled via `enable_residual_boosting`; engineered exact skip; `tests/test_phase4_orchestration.py` |

---

## Definition of done (per section)

- [ ] Source read for all primary paths listed  
- [ ] Related call sites checked (no false-positive findings)  
- [ ] Findings table filled (or explicit “clean pass”)  
- [ ] Any quick safe fixes either done or filed with severity  
- [ ] Status set to `done`

## Definition of done (whole audit)

- [ ] All S1–S10 done  
- [ ] Cross-cutting themes updated  
- [ ] Top 10 optimization candidates ranked  
- [ ] Optional: smoke tests for any code changes  

---

## Next action

**S1 + S2 + S4 + S5 done.** Recommended next: **S6** (bindings/seeds), then **S3** guards/scoring, then S7+.

**Priority fix queue (cross-cutting):**
1. **Phase 0–4 done**. Next: Phase 5 defaults/API polish or audit S6
2. **S5-5 / S5-6** hash quantize + weight threshold display gaps
3. **E3** seed capacity under islands (**fixed** Phase 2)
4. **N6/O8** local display MSE
5. **S1-4** n_features_in_ under blackbox (**fixed** Phase 4)
6. Phase 4 complete — continue Phase 5 or audit S6
7. Phase 5 complete — continue Phase 6 audit (S6→S3→S7–S10) or Phase 7 polish
