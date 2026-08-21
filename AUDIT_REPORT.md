# Glassbox Exhaustive Audit Report

**Status:** Canonical / single source of truth  
**Audit date:** 2026-07-25  
**Supersedes:** first-pass audit text and the interim verification pass (formerly split across two files) — merged here as the sole report  
**Scope:** Entire active tree (`glassbox/`, `scripts/`, `tests/`, `docs/`, C++ under `glassbox/sr/cpp/`); Eigen vendored, `.tmp/`, `results/`, local `models/` weights out of scope for product bugs  
**Method:** Multi-pass static analysis + cross-module verification + secondary cascade + full re-verification of every prior finding against current sources  
**Code modified:** H-01…H-22 fixed (C++ core, sklearn wrapper, blackbox, universal proposer, curve classifier, train paths, ONN tau/elite/BN, phased regression, family signatures, specialist vault ranking); M-04/M-05/L-01 fixed; R-01 fixed (swallowed-error diagnostics counter + typed fallbacks); R-02/R-12 fixed (AST-allowlist gate for formula `eval`); R-03 fixed (env-gated pickle checkpoint fallback + weights-only artifacts); report marks them **FIXED**

This document is the **only** audit report to maintain. It combines:

1. Original deep findings (critical/high/perf/robustness catalog with root causes and fixes)
2. Verification verdicts (confirmed / refined / overstated)
3. New issues discovered in the second pass
4. Unified prioritized action plan and patch sketches

---

## 0. How to Read This Report

| Verdict | Meaning |
|---------|---------|
| **CONFIRMED** | Prior claim matches current code; root cause and severity stand |
| **CONFIRMED (refined)** | True positive, but location, severity, or blast radius adjusted |
| **PARTIAL** | Core issue real; description incomplete, overstated, or understated |
| **FALSE POSITIVE** | Claim does not hold on current code |
| **ALREADY ADDRESSED** | Issue fixed or mitigated enough that the original bug report is stale |
| **FIXED** | Remediation landed in tree; original bug report is closed |
| **NEW** | Not in the first pass (or only implied); first-class finding here |

Stable IDs from the first pass (`C-01`…`C-13`, `H-*`, `P-*`, `R-*`, `M-*`, `L-*`) are preserved. New IDs continue the series (`C-14+`, `H-23+`, `P-09+`, `R-10+`, `M-06+`).

---

## 1. Executive Summary

### 1.1 Overall assessment

V1 is a **high-quality, mostly true-positive audit**. Of 13 critical findings, **13 remain true positives** after re-verification; none were false positives. Several need severity or exposure refinements:

- **C-05 (NSGA double-reproduce)** is real in `EvolutionEngine::run()`, but the **default multi-island path** (`num_islands=8` → `evolve_one_generation`) correctly `return`s after NSGA select. Production exposure is **conditional**, not universal.
- **C-03 (budget > timeout)** is real inside `GlassboxRegressor`; **SRBench local runner** partially caps budgets (`test_run_srbench_local.py` asserts caps) — package API still violates the user contract.
- **C-01 cascade** is worse than a single line: `_formula_mse`’s `np.all(np.isfinite(pred))` guard is **dead** because `_safe_eval_formula_array` already zero-fills.


### 1.1a What Glassbox is (architecture)

Glassbox is a hybrid symbolic-regression system. The **default production path** is:

1. `GlassboxRegressor.fit` (`glassbox/sr/sklearn_wrapper.py`, ~10k lines)
2. Optional blackbox preprocessing (`blackbox_preprocessor.py`)
3. Classifier / exact-match fast path (`scripts/classifier_fast_path.py`)
4. Universal proposer skeletons (`universal_proposer.py`)
5. Specialist composition / residual / vault (`specialist_state.py`)
6. Native island evolution via pybind `_core.run_evolution` (`core.cpp` → `evolution.h`)
7. Displayed-formula cleanup and scoring guards

The older **ONN Python evolution path** (`glassbox/evolution/evolution.py` + `sr/core/*`) remains as research/fallback infrastructure. Critical bugs still live in both paths; production defaults favor the C++ island path.

### 1.2 Architecture health scorecard

| Dimension | Pass 1 | Current | Notes |
|-----------|----|----|-------|
| Hybrid pipeline design | B+ | **B+** | Unchanged; stages clear |
| C++ search core | B- | **B-** | C-05 exposure refined; C-07/H-05/R-06′ still serious |
| Python orchestration | C+ | **C** | Zero-fill + 175 bare `except Exception` dominate risk; R-01 now counters/observes them |
| ONN research path | D+ | **D+** | C-08…C-11 all confirmed |
| Classifier / proposer | B- | **B-** | Train/serve + multi-var prefix gaps stand |
| Benchmark honesty | C | **C-** | 1D force exists in both suite **and** `benchmark_common.evaluate_formula_mse` |
| Docs vs code | B | **B-** | Stale 370 vs 398; Phase-2 weight claim; exact-match sympy dead |
| Test coverage of critical bugs | C- | **C-** | Still no regressions for C-01/C-05/C-12/C-07 |

### 1.3 Top residual production risks (ordered)

1. **Non-finite → 0 scoring** (C-01 + C-01′ + C-14 log display) can crown domain-failing formulas.
2. **`predict` silent zeros** (C-02) hide fit/feature errors from sklearn users.
3. **Adaptive budget uncapped by `timeout`** (C-03) — default `timeout=120`, `max_compute_budget=300`.
4. **Multivariate benchmark 1D collapse** (C-04 + C-04′ helper).
5. **Specialist `nest_formulas` multi-var corruption** (C-12).
6. **DAG crossover linear offset** (C-07) + invisible validity rate (M-03).
7. **Macro “divide” seeds multiply gate** (R-06′) under default Arithmetic sampling.

### 1.4 Scale (current tree)

| Artifact | LOC / count |
|----------|-------------|
| `glassbox/sr/sklearn_wrapper.py` | 10,080 |
| `glassbox/sr/cpp/evolution.h` | 3,972 |
| `glassbox/evolution/evolution.py` | 3,517 |
| `scripts/classifier_fast_path.py` | 4,864 |
| `glassbox/sr/cpp/eval.h` | 771 |
| `glassbox/sr/cpp/core.cpp` | 1,290 |
| `except Exception` in `sklearn_wrapper.py` | **170** (175 before R-01) |
| Pytest modules under `tests/` | 44+ |

---

## 2. Verification Log (every prior finding)

### 2.1 Critical findings (C-01 … C-13)

#### C-01 — Non-finite formula predictions coerced to 0  
**Verdict: CONFIRMED**  
**Severity: CRITICAL**  
**Evidence:** `sklearn_wrapper.py:6916`  
```python
out = np.where(np.isfinite(y_pred), y_pred, 0.0)
```
Called from `_safe_eval_formula_array` (`:6832`), which feeds `_formula_mse`, `_plain_unweighted_mse`, `_display_formula_mse`, vault/composition scoring, and `predict`.

**Root cause stands.** Domain failures become perfect near-zero predictors on zero-centered targets.

**Fix (unchanged intent, sharpened):**
- Scoring path: preserve raw non-finites; reject via domain-failure rate / `inf` MSE.
- `predict` path: optional fill policy, separate from search scoring.
- Reuse `_formula_domain_failure_rate` (`:6927+`).

---

#### C-01′ — Zero-fill makes `_formula_mse` finite-check dead  
**Verdict: CONFIRMED (cascade)**  
**Severity: HIGH → treat as part of P0 with C-01**  
**Evidence:**  
- Zero-fill at `:6916`  
- Guard at `:3800–3827`:
```python
pred = self._safe_eval_formula_array(text, X)
...
if not np.all(np.isfinite(pred)):
    return float("inf")
```
After zero-fill, the guard never fires. `_plain_unweighted_mse` finite **mask** is similarly inert (all values “finite”).

Specialist vault (`specialist_state.py:298–302` region) and composition paths that call the same evaluator inherit the poison.

---

#### C-02 — `predict` swallows all errors → silent zeros  
**Verdict: CONFIRMED**  
**Severity: CRITICAL**  
**Evidence:** `sklearn_wrapper.py:10029–10043`  
```python
try:
    return self._safe_eval_formula_array(self.formula_, X)
except Exception as e:
    print(f"Prediction error: {e}")
    return np.zeros(X.shape[0])
```
No feature-count check vs `n_features_in_`. Violates fail-loud sklearn norms; interacts with C-01 zero-fill.

**Fix:** Validate `X.shape[1]`; re-raise or return non-finite with clear error; never silent zero vector for structural failures.

---

#### C-03 — Adaptive compute budget can exceed user `timeout`  
**Verdict: CONFIRMED (refined)**  
**Severity: CRITICAL (API contract)**  
**Evidence:**  
- Defaults: `timeout=120` (`:1032`), `max_compute_budget=300` (`:1039`)  
- `_estimate_compute_budget` (`:1223–1318`) clips only to `[min_compute_budget, max_compute_budget]`, **not** to `self.timeout`  
- Used as `effective_timeout` at `:8818+`  
- Guided path partially caps with `min(effective_timeout, self.timeout)` around `:9053`  
- Multi-start still consumes `effective_timeout` wall (`:9244+`, `:9313`)

**Refinement:** `scripts/run_srbench_local.py` + tests cap `max_compute_budget` to the run budget for SRBench. **GlassboxRegressor itself does not.** Users of the estimator API still see up-to-300s runs on a 120s timeout.

---

#### C-03′ — Inflated budget passed into C++ multi-start  
**Verdict: CONFIRMED**  
**Severity: HIGH**  
Cascades from C-03; `timeout_seconds=run_timeout` derived from inflated `effective_timeout`.

---

#### C-04 — Benchmark suite forces 1D postprocess  
**Verdict: CONFIRMED**  
**Severity: CRITICAL (benchmark honesty)**  
**Evidence:** `scripts/benchmark_suite.py:772–777`  
```python
return bc.postprocess_formula_with_fidelity_guard(
    formula,
    np.asarray(x, dtype=np.float64).reshape(-1, 1),
    y,
)
```
Also rebinds `_evaluate_formula_mse = bc.evaluate_formula_mse` (`:769`), discarding any local multi-var-aware helper above it.

---

#### C-04′ — `benchmark_common.evaluate_formula_mse` is inherently 1D  
**Verdict: NEW (extends C-04)**  
**Severity: CRITICAL (shared helper)**  
**Evidence:** `scripts/benchmark_common.py:776–786`  
```python
X = np.asarray(x, dtype=np.float64).reshape(-1, 1)
```
Multi-var-safe API is `evaluate_formula_mse_on_X`. Any caller of `evaluate_formula_mse` on multi-col data is wrong even outside the suite.

---

#### C-05 — NSGA-II path double-reproduces in `run()`  
**Verdict: CONFIRMED (refined exposure)**  
**Severity: CRITICAL when `run()` is used; **MEDIUM-HIGH in default island config****  
**Evidence:** `evolution.h:359–393` NSGA create-offspring + `nsga2_select`, then **unconditional** “Create next generation” at `:444+` with no `continue`/`return`.

**Refinement (important):**  
- `evolve_one_generation` (`:3740–3797`) **does** `return` after NSGA select — island model is correct.  
- `run_islands()` falls back to `run()` when `num_islands <= 1` or `island_size < 4` (`: ~620–622` region).  
- Default sklearn wrapper: `use_nsga2=False`, `num_islands=8` — bug dormant unless NSGA enabled **and** single-pop/`run()` path taken.

**V1 overstated universal production hit rate; understated residual risk when NSGA+`run()` combine.**

---

#### C-06 — Empty / zero-feature `X` → UB  
**Verdict: CONFIRMED**  
**Severity: CRITICAL**  
**Evidence:**  
- `core.cpp` `run_evolution` (`:474–492`) converts `X_list` with **no** empty / length checks (unlike later helpers e.g. ~`:1167` `empty X` throw on another binding).  
- `create_random_individual` (`evolution.h:1353+`): with `n_inputs==0`, node 0 takes Unary branch and `std::uniform_int_distribution<int>(0, i-1)` → `(0, -1)` **UB**.  
- Feature sampling paths use `feat_dist(0, n_inputs - 1)` when `n_inputs > 0` only in some sites; empty still unsafe at init.

**Also NEW related:** no validation that each `X[j].size() == y.size()` in `run_evolution` entry — mismatched lengths → wrong MSE / partial reads (see R-10).

---

#### C-07 — DAG subtree crossover linear offset  
**Verdict: CONFIRMED**  
**Severity: CRITICAL (search quality)**  
**Evidence:** `evolution.h:1977–2086`  
```cpp
int offset = xo_a - xo_b;
...
donated.left_child = donated.left_child + offset;
```
`collect_subtree` (`:2104+`) returns **reachable** indices (sorted), not a contiguous dense block. Linear offset is invalid for sparse DAG index sets. Failed grafts return parent A; success can still rewire wrong. `last_crossover_valid_` exists but is not exported (M-03).

---

#### C-08 — ONN `router.router.get_primary_sources()`  
**Verdict: CONFIRMED**  
**Severity: CRITICAL (ONN path)**  
**Evidence:**  
- `operation_node.py:525` and `operation_dag.py:492`: `self.router.router.get_primary_sources()`  
- Routers implement `get_primary_sources` **directly** (`operation_node.py:96`, Sparse router ~`:207`)  
- `AttributeError` → silent `[0, 0]`

Formula extraction / compile disagree with true routing.

---

#### C-09 — `'p' in name` matches every `*_ops.*`  
**Verdict: CONFIRMED**  
**Severity: CRITICAL (ONN path)**  
**Evidence:**  
- `evolution.py:642–644` init; `:759–766` mutate  
- `hybrid_optimizer.py:76–82` includes bare `'p'` substring  

`unary_ops.0.omega` contains `p` via **`ops`**. Power init/clamp and “constants-only” LBFGS hit wrong tensors (`op_selector.logits`, etc.).

---

#### C-10 — Sparse top-K cache not invalidated after mutation  
**Verdict: CONFIRMED**  
**Severity: CRITICAL (ONN path)**  
**Evidence:**  
- Cache updates only when `training or not _cache_valid` (`operation_node.py:154+`)  
- `invalidate_cache` exists (`:147`) but **no callers** in `evolution.py` mutation paths  
- Fitness uses `model.eval()` + hard selection → stale top-K after logit/edge mutations

---

#### C-11 — `forward_compiled` ignores `source_window`  
**Verdict: CONFIRMED**  
**Severity: CRITICAL (ONN path)**  
**Evidence:** `operation_dag.py:163–173` windowed `forward` vs `:524–586` compiled always `torch.cat(all_sources)`. Shape mismatch or wrong math when `source_window >= 0`.

---

#### C-12 — `nest_formulas` rewrites every `x\d*`  
**Verdict: CONFIRMED**  
**Severity: CRITICAL**  
**Evidence:** `specialist_state.py:994–997`  
```python
return re.sub(r"\bx\d*\b", f"({g})", f)
```
Matches `x`, `x0`, `x1`, … So `nest_formulas("sin(x0)+x1", "x0**2")` → `sin((x0**2))+(x0**2)` — **destroys multi-var compositions**.

---

#### C-13 — Python seed path drops single-feature `x0`  
**Verdict: CONFIRMED (refined)**  
**Severity: CRITICAL under pure-Python fallback; **lower when C++ parser works****  
**Evidence:** `seed_graph_builder.py`  
- C++ path first (`formula_to_seed_graph` `:590+`); C++ parser **does** accept `x0` (`formula_parser.h` ~`:304`)  
- Python fallback: single free symbol not named `x` → multi path → `len(free)==1: return None` (`:88–89`, `:610–611`)

**Refinement:** Production with working `_core` seed parse is largely fine; pure-Python / C++ parse miss still drops `sin(x0)` seeds.

---

### 2.2 High-severity findings (H-01 … H-22)

| ID | Verdict | Notes |
|----|---------|-------|
| **H-01** | **FIXED** | Non-finite pred → fitness=`1e30`, raw_mse=`+inf`; residual MSE rejects non-finite pred/y |
| **H-02** | **FIXED** | Simplify unary/binary constant folds sanitize Inf/NaN and clamp; Exp matches eval band |
| **H-03** | **FIXED** | `clamp_unary_inner_params`: phi ±4π; Exp tighter omega/phi ±4 — Adam, LM, mutation |
| **H-04** | **FIXED** | LM Jacobian bounds-checks children; FD residual fallback when `|wi|≤1e-6` / nested |
| **H-05** | **FIXED** | Arithmetic snap uses full active-subtree mask (`collect_active_reachable_mask`) |
| **H-06** | **FIXED** | Thread-local arithmetic temperature + process default for fresh OMP workers |
| **H-07** | **FIXED** | Seeds validated with `is_valid_graph_topology` + `feature_idx < n_features`; invalid counted/skipped |
| **H-08** | **FIXED** | Final accept / `best_mse_` use `_final_formula_score` display MSE (not search/engine loss) |
| **H-09** | **FIXED** | Blackbox `_as_sample_weight` raises `ValueError` on invalid weights (None still unweighted) |
| **H-10** | **FIXED** | `evolution_error_` / `evolution_required_`; fail closed when required+failed+no formula; restore loss_mode |
| **H-11** | **FIXED** | Formula eval cache keys content hash (≤4096 elems) or ptr+strides — not bare `id(X)` |
| **H-12** | **FIXED** | `_mad_scale` applies same finite mask to weights; pre-filter length mismatch raises |
| **H-13** | **FIXED** | Multivariate pairs among all cols (D≤8) or top-`max_rank` by \|corr(y,xi)\| — not index prefix |
| **H-14** | **FIXED** | Interpolator cache keys content hash / sample fingerprint — not bare buffer pointers |
| **H-15** | **FIXED** | Train curve gen + synthetic proposer use `extract_all_features_xy` (match inference sort/resample) |
| **H-16** | **FIXED** | Derive `n_features` from `eql` / `other_mlp` weights (or `feature_dim`); no wrong `classifier_in-512` / `fc1` combined width |
| **H-17** | **FIXED** | `set_model_tau` only touches HardConcrete selectors/gates + plain float node tau; skips `MetaAggregation` |
| **H-18** | **FIXED** | Clear `_is_elite` when selection tau changes so elites are re-evaluated under new temperature |
| **H-19** | **FIXED** | BFGS import via `optimizers.bfgs_optimizer`; `_unary_op_kind` uses node op list (`periodic=0`, `power=1`) |
| **H-20** | **FIXED** | `refine_constants` / `quick_refine_internal` run in `eval()` so BN and HardConcrete match fitness |
| **H-21** | **FIXED** | Multi-token family signature (`sin+exp+multiplicative`); nest eligibility accepts token sets |
| **H-22** | **FIXED** | `residual_relevance` set on vault admission; ranks retention/export (MSE, −relevance, complexity); rescore re-sorts |

### 2.3 Performance (P-01 … P-08)

| ID | Verdict | Notes |
|----|---------|-------|
| **P-01** | **FIXED** | `SubtreeCache` is LRU with entry/byte caps (`ast.h`); thread budgets split; mid-depth-only inserts (`eval.h`); diagnostics exported |
| **P-02** | **FIXED** | Default islands & single-pop share `evolve_one_generation` (macro mutations, staged schedule, elite age++, restart, inner refine) |
| **P-03** | **FIXED** | `LBFGSConstantOptimizer(..., max_iter=self.lbfgs_steps)` **and** loop `for _ in range(self.lbfgs_steps)` → up to steps². Fix: single `optimizer.step()` with full `max_iter` budget (`hybrid_optimizer.py`); regression test `tests/test_audit_p03_fix.py` |
| **P-04** | **FIXED** | FD Adam full re-eval per param. Fix: per-step base cache + `evaluate_graph_partial` subtree probes (`eval.h` `evaluate_perturbed_pred`); inert-param skip (p on Periodic/Exp, ω/φ on Power, all on Log/Abs) gated by `fd_skip_inert_params`; LM skips Log/Abs nodes; exports `fd_probes_total`/`fd_probes_skipped_inert`; regression tests `tests/test_audit_p04_fix.py` |
| **P-05** | CONFIRMED | 10k-LOC god module + Python `eval` scoring |
| **P-06** | CONFIRMED | Vault stores full residual/prediction vectors |
| **P-07** | CONFIRMED | Coarse 32-sample moment cache key (`curve_classifier_integration.py:450–470`) |
| **P-08** | CONFIRMED | Nest fallback `return macro_mutate(parent)` (`evolution.h:1954–1956`) without attempt cap |

### 2.4 Robustness (R-01 … R-09)

| ID | Verdict | Notes |
|----|---------|-------|
| **R-01** | **FIXED** | `swallowed_errors_` counter + `_record_swallowed_error`; typed exceptions at conversion/import fallbacks; hot-path instrumentation; `swallowed_errors_summary_` + `blackbox_diagnostics_["swallowed_errors"]`; regression tests `tests/test_audit_r01_swallowed_errors.py` |
| **R-02** | **FIXED** | AST-allowlist gate (`glassbox/sr/formula_safety.py`) before every formula `eval` in sklearn_wrapper, benchmark_common, universal_proposer, specialist_state (+ generate_curve_data raw branch); rejects attribute traversal / subscript / lambda / import; regression tests `tests/test_audit_r02_expression_injection.py` |
| **R-03** | **FIXED** | Env-gated pickle fallback (`GLASSBOX_ALLOW_PICKLE_CHECKPOINT=1` opt-in + trusted `models/`/`artifacts/` path) in both torch checkpoint loaders; shipped artifacts migrated to weights-only; regression tests `tests/test_audit_r03_pickle_checkpoint.py` |
| **R-04** | CONFIRMED | Soft-div `x/sqrt(1+y²)` ≠ algebraic `/` |
| **R-05** | CONFIRMED | Power/log folds on signed domains in `simplify_advanced.h` |
| **R-06** | CONFIRMED → see **R-06′** | Divide macro uses `sample_binary_op()`; can be Aggregation |
| **R-07** | CONFIRMED | `elastic_net_cd_cpp` returns MSE `0.0` for empty (`refine.h:46–47`) |
| **R-08** | CONFIRMED | Estimator shared across ThreadPool formula eval |
| **R-09** | CONFIRMED | No absolute cancel inside long C++ evolution calls |

### 2.5 Medium / low / cascade (M-*, L-*)

| ID | Verdict | Notes |
|----|---------|-------|
| **M-01** | **FIXED** | Vault holdout uses deterministic permuted index split instead of deterministic tail (`specialist_state.py`) |
| **M-02** | **FIXED** | FPIP fast-path builder now assigns candidate probabilities and builds `search_plan` from x/y via `build_search_plan` (`fpip_v2.py`) |
| **M-03** | **FIXED** | `last_crossover_valid`, `crossover_attempts`, `crossover_successes`, and `crossover_valid_rate` exported from `core.cpp` |
| **M-04** | **FIXED** | Meta soft-div = abs-form (`meta_ops.py:573`); C++ **Arithmetic** soft-div = sqrt-form (`eval.h:273`); C++ **Division** matches meta abs-form (`eval.h:280`) — meta ≠ Arithmetic soft |
| **M-05** | **FIXED** | `hard_concrete_log_prob` now the true Hard-Concrete density (`hard_concrete.py:73–109`); regression test `tests/test_audit_m05_fix.py` |
| **L-01** | **FIXED** | `compute_specialist_state` rank-sorts candidates (validation MSE/R²) before the first-N slice (`specialist_state.py:767–786`); regression test `test_compute_specialist_state_ranks_before_slice` |

### 2.6 Docs / dead code (V1 §6) — verification

| Claim | Verdict |
|-------|---------|
| Docstring “C++ weight-aware starting Phase 2” while code already passes `y_weights` | CONFIRMED stale |
| UP docstring 370 vs default/config 398 | CONFIRMED (`universal_proposer.py:111` vs `:174`) |
| `expected_features = 366` fallback in train | CONFIRMED (`train_curve_classifier.py:1377`) |
| Exact-match full SymPy branch dead (`simplification_info` never sets `exact_match`) | CONFIRMED |
| Dead `'R' in name` branch | CONFIRMED (`evolution.py:651`) |
| Pruning looks for node `output_proj` vs `output_scale` | CONFIRMED pattern in `pruning.py` |
| README hybrid pipeline high-level accuracy | CONFIRMED good |

### 2.7 False positives / already addressed

| Claim | Verdict |
|-------|---------|
| None of C-01…C-13 | **No full false positives** |
| C-05 “always double-reproduces in production” | **Overstated** — island NSGA OK; `run()` broken |
| C-13 “always drops x0 seeds” | **Overstated** — C++ parser handles x0; Python fallback drops |
| H-08 “all display/engine mixing is a bug” | **Partial** — some split is deliberate S1-6 contract |
| C-03 on SRBench-only runs | **Partially mitigated** by runner budget cap, not by estimator |

---

## 3. New Issues (second-pass discovery)

### 3.1 Mathematical & symbolic integrity

#### C-14 — Display `log` drops engine ε-protection  
**Severity: CRITICAL (cascade with C-01)**  
**File:** `eval.h:622` vs eval impl `:246–248`  
- Engine: `(abs(x) + 1e-6).log()`  
- Display string: `log(|child|)`  
At zeros / tiny values, Python display eval → `-inf` → C-01 zero-fill → false fitness. Same family as soft-div display honesty, but **directly corrupts selection**.

**Fix:** Emit `log(abs(x)+1e-6)` (or shared protected-log helper) in formatter; keep C-01 fix as backstop.

---

#### R-06′ — Macro “divide” seeds **multiply** Arithmetic gate  
**Severity: HIGH**  
**File:** `evolution.h:1903–1930`  
```cpp
div_node.binary_op = sample_binary_op();
if (div_node.binary_op == BinaryOp::Arithmetic) {
    seed_arithmetic_gate(div_node);
    div_node.beta = 2.0;
    div_node.gamma = 1.0;  // multiply locus, NOT div (gamma=-1)
}
```
V1 R-06 noted non-Division sampling; V2 adds: when Arithmetic is chosen, parameters are seeded to **mul**, not div. “Divide/rational mutation” systematically fails its intent.

**Fix:** Force `BinaryOp::Division` or set `beta=2, gamma=-1` for Arithmetic div intent.

---

#### H-23 — Power parity: eval soft blend vs simplify hard parity  
**Severity: HIGH (numeric/search consistency)**  
**Files:** `eval.h:61–68` `power_sign_blend`; `simplify.h` / `simplify_advanced.h` hard even test  
Near-integer `p` uses discrete even/odd in both, but non-integer paths and constant-fold vs live eval can diverge on negative bases; display power strings often look like true `x**p` while engine uses `abs`-base blend.

**Fix:** Single shared parity policy; display must match engine (abs-base or true complex — pick one).

---

#### H-24 — `log(exp(y)) → y` fold ignores protected log  
**Severity: MEDIUM-HIGH**  
**File:** `simplify_advanced.h:153–154`  
Identity assumes mathematical log/exp; engine log is `log(|·|+ε)`. Fold can change values near singular regions and disagree with display.

---

#### M-06 — Blackbox OOB feature remap → literal `0`  
**Severity: MEDIUM**  
**File:** `blackbox_preprocessor.py:1264–1282`  
Unmapped indices become constant `0` (intentional S7-2). Silent semantic change if seeds/formulas reference dropped features — can look like “fit” on incomplete structure.

**Fix:** Track `remap_dropped_features` diagnostic; optionally reject formulas that required dropped vars.

---

### 3.2 Performance & memory

#### P-09 — TLS eval arena sticky growth under varying batch sizes  
**Severity: MEDIUM**  
**File:** `eval.h:135–141`  
Thread-local `arena` grows with `num_samples` and only shrinks when `rows > num_samples*4`. Long-lived threads across problems can retain large arenas.

---

#### P-10 — Generation subtree cache stores full `ArrayXd` per hash with no byte cap  
**Severity: MEDIUM (extends P-01)**  
**Status: FIXED** (same change as P-01)  
Even with per-gen clear, a single generation on n=1e5 × diverse subtrees can OOM before clear. Cap is now entry+byte LRU on `SubtreeCache`.

---

### 3.3 Edge cases & robustness

#### R-10 — `run_evolution` does not assert `X[i].size() == y.size()`  
**Severity: HIGH**  
**File:** `core.cpp:474–492`  
Features may have unequal lengths vs `y` / each other. Eval uses `num_samples = y.size()`; Eigen maps shorter features → **UB / wrong reads**.

**Fix:** Before engine construct: `X` non-empty; all columns size `y.size()`; `n_features >= 1` (or explicit constant-only mode).

---

#### R-11 — NSGA path in `run()` double-ages elites  
**Severity: LOW-MEDIUM**  
**File:** `evolution.h:359–517`  
NSGA block does `ind.age++` on parents; fall-through elitism does `elite.age++` again. AFPO age semantics corrupted when C-05 triggers.

---

#### R-12 — `predict` / scoring use Python `eval` without AST allowlist  
**Severity: MEDIUM (security/robustness)**  
Extends R-02; production path for all displayed formulas.  
**Status: FIXED** (same change as R-02 — AST-allowlist gate in `formula_safety.py` applied at every production eval site, incl. `predict`/`_safe_eval_formula_array` and `_formula_domain_failure_rate`).

---

#### H-25 — Family signature + nest: multi-op formulas monopolized by first keyword  
**Severity: HIGH (extends H-21)**  
**Status: FIXED** (same change as H-21)  
`sin(x)*exp(x)` → multi-token `"exp+multiplicative+sin"`; nest eligibility uses token sets.

---

#### H-26 — Coarse curve feature cache can cross-talk across distinct curves  
**Severity: MEDIUM-HIGH (extends P-07)**  
Key = `(n, mean/std moments, dot, n_points)` over 32 samples — different formulas with similar low-order stats collide within the 32-entry process cache.

---

### 3.4 Docs vs implementation (additional)

| Issue | Evidence |
|-------|----------|
| **D-01** UP `forward` docstring still says 370 | `universal_proposer.py:174` vs `n_features: int = 398` |
| **D-02** Train CLI default feature dim confusion 366 vs 398 | `train_curve_classifier.py:1377` |
| **D-03** sklearn `fit` docstring still says C++ weights “Phase 2” | `sklearn_wrapper.py:7785–7786` while Phase 3+ already pass `y_weights` |
| **D-04** Exact-match simplification branch documented/implied but dead | `classifier_fast_path.py` simplification_info never sets `exact_match` |

---

## 4. Unified Action Plan

### 4.1 P0 — Ship blockers (correctness of reported results)

| Priority | ID(s) | Action | Files (lines) | Effort | Suggested test |
|----------|-------|--------|---------------|--------|----------------|
| P0.1 | C-01, C-01′, C-14 | Split safe-eval into raw vs fill; scoring uses raw; reject non-finite; fix log formatter ε | `sklearn_wrapper.py:6832–6916,3800+`; `eval.h:622` | S | `log(x)` on x∈(0,1] with zeros present must not beat true model via zero-fill |
| P0.2 | C-02 | Feature check + no silent zeros | `sklearn_wrapper.py:10029–10043` | S | wrong `n_features` raises |
| P0.3 | C-03, C-03′ | `budget = min(budget, self.timeout)` after clip; assert C++ timeouts ≤ remaining | `sklearn_wrapper.py:1318,8818+` | S | `timeout=30, adaptive=True` never schedules >30s |
| P0.4 | C-04, C-04′ | Stop `reshape(-1,1)` in suite; route multi-var through `evaluate_formula_mse_on_X` | `benchmark_suite.py:772–777`; `benchmark_common.py:776` | S | `x0+x1` finite MSE on 2-col data |
| P0.5 | C-12 | Nest only primary feature (or explicit map); never global `x\d*` | `specialist_state.py:994–997` | S | `nest_formulas("sin(x0)+x1","x0**2")` keeps `x1` |
| P0.6 | C-05 | After NSGA select in `run()`, `continue` (mirror island `return`) | `evolution.h:393–444` | S | gen counter / pop genealogy under `use_nsga2` + `num_islands=1` |
| P0.7 | C-06, R-10 | Validate X/y sizes at `run_evolution`; refuse `n_features==0` | `core.cpp:474+`; `evolution.h` init | S | empty X throws; mismatched lengths throw |
| P0.8 | H-01 | **DONE** Non-finite pred → fitness=`1e30`, raw_mse=`+inf` | `evolution.h` | S | NaN pred never sorts above finite |

### 4.2 P1 — Search quality / silent wrong formulas

| ID(s) | Action | Files | Effort |
|-------|--------|-------|--------|
| C-07, M-03 | Index-map crossover; export `crossover_valid_rate` | `evolution.h`, `core.cpp` | L |
| R-06′ | Force Division or div-gate params in macro divide | `evolution.h:1903–1930` | S |
| H-05 | **DONE** Arithmetic snap uses full active mask / binary mask | `evolution.h` | S |
| H-02, H-03 | **DONE** Clamp simplify folds; clamp `phi` (and Exp ω) in Adam/LM | `simplify*.h`, `evolution.h` | S |
| H-04 | **DONE** LM FD fallback + child bounds for nested unaries | `evolution.h` | S |
| H-06 | **DONE** Thread-local arithmetic temperature isolation | `eval.h` | S |
| H-21, H-25 | **DONE** Multi-token family signatures + nest token sets | `sklearn_wrapper.py`, `specialist_state.py` | S |
| H-22 | **DONE** residual_relevance in vault rank/admission/rescore | `specialist_state.py` | S |
| C-13 | Python seed: treat `x0` as single-feature alias of `x` | `seed_graph_builder.py:88,610` | S |
| H-08 | **DONE** Final accept always `_final_formula_score` / display MSE | wrapper + fast path | M |
| H-09 | **DONE** Raise on invalid blackbox weights | `blackbox_preprocessor.py` | S |
| H-07 | **DONE** Seed topology + feature_idx validation | `core.cpp` | S |
| H-10 | **DONE** `evolution_error_`; fail closed when required | `sklearn_wrapper.py` | S |
| H-11 | **DONE** Formula eval cache content fingerprint | `sklearn_wrapper.py` | S |
| H-12 | **DONE** `_mad_scale` finite-mask weights | `sklearn_wrapper.py` | S |
| H-13 | **DONE** All-pairs or relevance-ranked pairs in grammar | `universal_proposer.py` | M |
| H-14 | **DONE** Interpolator cache content fingerprint | `curve_classifier_integration.py` | S |
| H-15 | **DONE** Train with `extract_all_features_xy` | `generate_curve_data.py`, train script | S |
| H-16 | **DONE** Derive CNN/GLU/MLP `n_features` from weights | `curve_classifier_integration.py` | S |
| H-17 | **DONE** `set_model_tau` HardConcrete-only | `evolution.py` | S |
| H-18 | **DONE** Clear elite skip on tau change | `evolution.py` | S |

### 4.3 P2 — ONN research path (if still supported)

| ID | Action | Effort |
|----|--------|--------|
| C-08 | `router.get_primary_sources()` | S |
| C-09 | Token-equality for `p` / include lists | S |
| C-10 | `invalidate_cache` after every mutation | S |
| C-11 | Windowed compiled forward | M |
| H-17–H-18 | **DONE** tau filters + elite re-eval | S |
| H-19–H-20 | **DONE** BFGS import, unary index map, BN mode | S |

### 4.4 P3 — Performance

| ID | Action | Effort |
|----|--------|--------|
| P-01, P-10 | **DONE** LRU/byte-capped `SubtreeCache` + mid-depth inserts + diagnostics | `ast.h`, `execution.h`, `eval.h`, `core.cpp` | M |
| P-02 | **DONE** Share reproduce implementation islands ↔ run | `evolution.h` | M |
| P-03 | **DONE** Single LBFGS step with `max_iter=steps` | S |
| P-04 | **DONE** Incremental FD probes + inert-param skip (+ LM Log/Abs skip) | M |
| P-05 | Split wrapper; C++ display score batch API | L |
| P-06 | Residual sketches for large n | S |
| P-08 | Macro nest attempt counter | S |
| P-09 | Bound TLS arena hard max | S |

### 4.5 P4 — Tests, docs, cleanup

Must-have regression tests (none exist as dedicated cases today):

1. Non-finite scoring rejection (C-01/C-14)  
2. `predict` feature mismatch (C-02)  
3. Budget ≤ timeout (C-03)  
4. Multi-var benchmark MSE (C-04)  
5. NSGA single-island generation accounting (C-05)  
6. Empty/mismatched X throws (C-06/R-10)  
7. `nest_formulas` multi-var (C-12)  
8. Python `x0` seed (C-13)  
9. Crossover map validity rate export (C-07/M-03)  
10. Macro divide creates true division (R-06′)  
11. Family multi-op signature (H-21)  
12. Soft-div meta vs C++ Arithmetic parity (M-04)  

Docs: fix 370/398, Phase-2 weights blurb, exact-match simplify status, soft-div meaning.

---

## 5. Concrete Patch Sketches (tested intent; not applied)

### 5.1 C-01 scoring split (Python)

```python
def _eval_formula_raw(self, formula, X):
    # same prep as _safe_eval_formula_array but:
    y_pred = eval(...)
    return np.asarray(y_pred, dtype=np.float64)  # keep non-finite

def _safe_eval_formula_array(self, formula, X):  # predict-only policy
    raw = self._eval_formula_raw(formula, X)
    return np.where(np.isfinite(raw), raw, 0.0)

def _formula_mse(...):
    pred = self._eval_formula_raw(text, X)
    if pred.shape != target.shape or not np.all(np.isfinite(pred)):
        return float("inf")
    ...
```

### 5.2 C-05 NSGA continue (C++)

```cpp
if (config_.use_nsga2) {
    // ... build combined, nsga2_select ...
    population_ = nsga2_select(combined, config_.pop_size);
    for (auto& ind : population_) consider_champion(ind);
    // periodic refine...
    continue; // DO NOT fall through to second reproduce
}
```

### 5.3 C-12 nest (Python)

```python
def nest_formulas(f: str, g: str, *, primary: str = "x0") -> str:
    # replace only primary feature occurrences, not every x\d*
    return re.sub(rf"\b{re.escape(primary)}\b", f"({g})", f)
```

### 5.4 C-03 budget cap

```python
budget = float(np.clip(budget, float(self.min_compute_budget), float(self.max_compute_budget)))
return float(min(budget, float(max(1, self.timeout))))
```

### 5.5 R-06′ macro divide

```cpp
div_node.type = NodeType::Binary;
div_node.binary_op = BinaryOp::Division; // or Arithmetic with beta=2, gamma=-1
```

### 5.6 C-08 routing

```python
primary_sources = self.router.get_primary_sources()
```

---

## 6. What Still Looks Solid (re-validated)

- Hybrid stage order (fast path → proposer → specialist → C++ islands → cleanup) matches README/PROJECT_MAP at high level.  
- Protected ops + output clamps in `eval.h` are deliberate and mostly consistent **inside** the engine.  
- Soft-div **display** for Arithmetic now matches engine sqrt-form (S5-4) — good; Log display does **not** yet match.  
- Sample-weight validation on the wrapper public API is strict (`_validate_sample_weight`); blackbox path is the weak link (H-09).  
- Island NSGA path correctly returns (contrast `run()`).  
- Dual champion archive (fitness vs raw_mse) is thoughtful.  
- FPIP v2 schema validation exists (`validate_fpip_v2_payload`) even if `search_plan` is empty.  
- Many phase tests exist; they simply do not pin the P0 defects above.

---

## 7. Module Coverage Ledger

| Module | Pass 1 | Re-verify depth | Primary residual risks |
|--------|----|----------|------------------------|
| `sr/sklearn_wrapper.py` | sectional | re-verified hot paths | C-01–03, H-08–12; R-01 **FIXED** |
| `sr/cpp/evolution.h` | yes | re-verified NSGA/XO/macro/fitness | C-05–07, H-01–05, R-06′, P-02/08 |
| `sr/cpp/eval.h` | yes | re-verified | H-06, C-14, R-04, M-04 |
| `sr/cpp/core.cpp` | yes | entry validation gap | C-06, R-10, H-07 |
| `sr/cpp/simplify*.h` | yes | folds | H-02, R-05, H-24 |
| `sr/cpp/seed_graph_builder.py` | partial | C-13 refined | Python x0 drop |
| `sr/specialist_state.py` | partial | nest/vault/holdout | C-12, M-01, H-22, L-01 |
| `sr/core/operation_*` | yes | C-08/10/11 | ONN routing |
| `evolution/evolution.py` | yes | C-09, H-17 | ONN |
| `scripts/benchmark_*` | yes | C-04′ | 1D helpers |
| `universal_proposer` | yes | H-13, D-01 | multi-var prefix |
| `curve_classifier/*` | yes | H-14–16, H-26 | cache/train-serve |
| `fpip_v2` / hard_concrete / meta_ops | yes | M-02/04/05 | parity |
| `tests/*` | map | gap matrix extended | missing P0 tests |

---

## 8. Severity Counts

| Severity | First-pass claimed | Verified + new |
|----------|------------|-------------------|
| CRITICAL | 13 (C-01…C-13) | **14** production-critical themes (C-01…C-13 + C-14); C-05 exposure refined |
| HIGH | 22+ | **24+** open themes originally; **H-01…H-22 FIXED** (remaining open: H-23+ if any, plus C/P/R) |
| MEDIUM / PERF / ROBUST | 30+ | **35+** including P-09/10, M-06, D-* |
| FALSE POSITIVES | — | **0** full FPs among C-*; **2–3** overstatements refined |

---

## 9. Recommended Further Passes (dynamic)

1. **Fuzz harness:** empty X, length-mismatched X/y, 1-row y, all-NaN y, multi-var tiers.  
2. **Differential:** C++ `evaluate_graph` vs display string eval on grids including zeros (log/div/power).  
3. **NSGA accounting test:** instrument generation count under `use_nsga2=True, num_islands=1`.  
4. **RSS profile:** n=10k, pop=200, 60s — validate P-01 peak.  
5. **Mutation testing** on `_final_formula_score` and zero-fill removal.  
6. After P0 pack, re-run this checklist as a delta pass.

---

## 10. Bottom Line

The first-pass roadmap is **actionable and mostly correct**. This consolidated report:

1. **Confirms** all critical IDs with exact current line numbers.  
2. **Refines** C-05 (island OK / `run()` broken), C-13 (C++ OK / Python fallback broken), C-03 (SRBench partial cap).  
3. **Strengthens** C-01 via dead finite-check cascade and **C-14** log display mismatch.  
4. **Adds** R-06′ (divide macro seeds multiply), R-10 (X/y length), H-23–H-26, P-09/10, M-06, D-01–D-04.  
5. **Prioritizes** an ordered P0 pack with file:line anchors and regression test titles.

**No production code was modified in this audit.**  
Report path: `/mnt/windows_d/Glassbox/AUDIT_REPORT.md`

---

*End of audit report.*

---

## 11. Appendix A — Detailed finding narratives (H / P / R)

> The verification log in §2 summarizes verdicts. This appendix preserves the fuller first-pass writeups (root cause, impact, fix) for high/perf/robustness items so implementers do not need a second file.

### A.1 High-severity narratives (H-01 … H-20)

### H-01 — C++ fitness does not sanitize NaN/Inf predictions
**Status: FIXED**

**File:** `evolution.h` fitness path  

After eval clamps, residual NaNs (bad seeds, constant fold Inf) can still produce non-finite MSE. `std::sort` on NaN fitness is ill-defined.

**Fix applied:** If `!pred.isFinite().all()` or non-finite MSE → set `fitness = 1e30`, `raw_mse = +inf`, return early. `residual_mse*` also reject non-finite pred/y.

---

### H-02 — Constant-fold simplify can store Inf/NaN
**Status: FIXED**

**Files:** `simplify.h`, `simplify_advanced.h` fold paths  

`std::exp(omega*v+phi)` folded without clamp (eval path clamps; simplify does not).

**Fix applied:** non-finite → 0; clamp to `[-1e8, 1e8]`; Exp argument clamp + eval-matched band.

---

### H-03 — Adam/LM leave `phi` unconstrained
**Status: FIXED**

**File:** `evolution.h`  

Only `p` and `omega` clamped; `phi` free → Exp argument explosion, flat regions at output clamps.

**Fix applied:** `clamp_unary_inner_params` — phi ±4π; Exp omega/phi ±4; used in Adam, LM `unpack_params`, mutation.

---

### H-04 — LM analytical Jacobian ignores nested active unaries
**Status: FIXED**

**File:** `evolution.h` LM Jacobian  

Uses output weight as sensitivity; nested unaries with near-zero output weight get zero Jacobian. Unchecked `left_child` indexing risk.

**Fix applied:** bounds-check children; residual FD fallback when `|wi| ≤ 1e-6` or child/cache invalid.

---

### H-05 — Arithmetic-gate snapping is dead code
**Status: FIXED**

**File:** `evolution.h` cleanup snap  

`snap_active_unary` only marks Unary nodes; Arithmetic snap requires Binary **and** that mask → never runs.

**Fix applied:** `collect_active_reachable_mask` for full active subtree; Arithmetic snap gates on that mask.

---

### H-06 — Process-global arithmetic temperature races concurrent engines
**Status: FIXED**

**File:** `eval.h`, used from evolution/core  

`static std::atomic<double>` temperature is process-wide. Concurrent `run_evolution` / scoring with different temps stomp each other.

**Fix applied:** thread-local primary storage; process atomic as default for fresh OMP workers; Scoped restores TLS only; fitness re-applies engine config temp.

---

### H-07 — Seeds accepted without topology / feature-index validation
**Status: FIXED**

**File:** `core.cpp` seed_graphs parse  

Bad `left_child` / `feature_idx` evaluate as zero silently.

**Fix applied:** `is_valid_graph_topology` + Input `feature_idx ∈ [0, n_features)`; skip empty/invalid; export `seed_graphs_skipped_invalid`.

---

### H-08 — Display MSE vs engine/search MSE mixed in final accept
**Status: FIXED**

**Files:** `sklearn_wrapper.py` blackbox/evolution accept + final `best_mse_`  

Contract says displayed unweighted MSE for reporting; blackbox/C++ paths often rank with robust/weighted/`best_mse` engine metrics. Post-simplify display can regress without rejection at every boundary.

**Fix applied:** Evolution selection MSEs via `_final_formula_score`; public `best_mse_` / fast-path finish re-scored with display MSE.

---

### H-09 — Blackbox `_as_sample_weight` silently drops invalid weights
**Status: FIXED**

**File:** `blackbox_preprocessor.py`  

Invalid weights → `None` (unweighted) without error; desync if caller believes weights are active.

**Fix applied:** Raise `ValueError` on length/non-finite/negative/all-zero; `None` still means unweighted.

---

### H-10 — C++ / guided evolution failures swallowed
**Status: FIXED**

**File:** `sklearn_wrapper.py` evolution try/except  

`except Exception: print` → fit continues with weak incumbent, no `evolution_error_` flag.

**Fix applied:** set `evolution_error_` / `evolution_required_`; raise `RuntimeError` when required+failed+no formula; restore user `loss_mode` before raise.

---

### H-11 — Formula eval cache keyed by `id(X)`
**Status: FIXED**

**File:** `sklearn_wrapper.py` `_safe_eval_formula_array`  

In-place mutation or recycled ndarray identity → stale predictions.

**Fix applied:** cache key = expr + shape + dtype + content hash (≤4096 elems) or (data ptr, strides).

---

### H-12 — `_mad_scale` weight length mismatch after filtering non-finite residuals
**Status: FIXED**

**File:** `sklearn_wrapper.py` `_mad_scale`  

Finite mask applied to `r` but not weights → silent unweighted MAD.

**Fix applied:** raise on pre-filter length mismatch; apply same finite mask to weights.

---

### H-13 — Multivariate grammar only pairs first `max_rank` columns
**Status: FIXED**

**File:** `universal_proposer/universal_proposer.py`  

`limit = min(n, max_rank)`; pairs only among prefix. `y = x2*x3` with `max_rank=2` fails.

**Fix applied:** D≤8 enumerate all pairs; else top-`max_rank` columns by \|corr(y, xi)\| (variance fallback).

---

### H-14 — Multi-input interpolator cache keyed by buffer pointer
**Status: FIXED**

**File:** `curve_classifier/curve_classifier_integration.py`  

`__array_interface__['data'][0]` only; in-place y mutation reuses stale interpolators.

**Fix applied:** content hash for small arrays; sample+mean+ptr fingerprint for large.

---

### H-15 — Train features ignore x-order; inference sorts by x
**Status: FIXED**

**File:** `generate_curve_data.py` train path; `train_universal_proposer.py` synthetic  

Shuffled curves: large feature shift → train/serve gap.

**Fix applied:** univariate train extraction uses `extract_all_features_xy(x, y)`.

---

### H-16 — CNN/GLU n_features fallback reconstructs wrong model
**Status: FIXED**

**File:** `curve_classifier_integration.py` load path  

Missing `model_config` → wrong dims → load fail → empty predictions (fail-open).

**Fix applied:** `_resolve_n_features_from_state_dict` prefers `model_config['n_features']` / `feature_dim`, else derives from `other_mlp.0` (CNN: `curve_dim + other_dim`) or `eql.linear` (GLU/MLP). Raises rather than guessing when weights are insufficient. GLU arch detection also fixed for missing `model_type`.

---

### H-17 — `set_model_tau` overwrites `MetaAggregation.tau`
**Status: FIXED**

**File:** `evolution.py` `set_model_tau`, `meta_ops.py`  

Selection annealing and aggregation temperature share API → wrong aggregation math.

**Fix applied:** Only set tau on `HardConcreteGate` / `HardConcreteSelector` / `HardConcreteOperationSelector` (and plain float node/DAG `tau`); skip `MetaAggregation` entirely (including its `set_tau`).

---

### H-18 — Elite fitness skip after global tau change
**Status: FIXED**

**File:** `evolution.py` train loop + hybrid ES cycle  

`set_model_tau` then skip re-eval for elites → stale fitness.

**Fix applied:** Track `_last_applied_tau`; when selection tau changes, clear `_is_elite` so `evaluate_fitness` re-scores. Elite skip remains when tau is unchanged.

---

### H-19 — Phased regression: wrong BFGS import + unary index swap
**Status: FIXED**

**File:** `phased_regression.py`  

`from .bfgs_optimizer` always failed (`optimizers/bfgs_optimizer.py`); unary 0/1 mapped opposite of actual `[MetaPeriodic, MetaPower]`.

**Fix applied:** Import from `.optimizers.bfgs_optimizer`; `_unary_op_kind` uses `node._unary_op_names` with fallback `periodic=0, power=1, exp=2, log=3`.

---

### H-20 — Train vs eval BatchNorm mismatch (refine vs fitness)
**Status: FIXED**

**File:** `evolution.py` `refine_constants`, `quick_refine_internal`  

Refine used `model.train()` while fitness used `model.eval()` → BN / HardConcrete discrete path disagree.

**Fix applied:** Both refine helpers force `model.eval()` for the whole optimization (including L-BFGS closures) and leave the model in eval for fitness.

---

### H-21 — Family signature first-token monopoly (extends H-25)
**Status: FIXED**

**File:** `sklearn_wrapper.py` `_formula_family_signature`; nest checks in `specialist_state.py`  

`sin(x)*exp(x)` → family `"sin"` only; diversity prune and nest eligibility mis-bucket multi-op formulas.

**Fix applied:** Multi-token sorted signature joined with `+` (e.g. `exp+multiplicative+sin`); `_formula_family_contains` for operator hints; nest eligibility splits multi-token families.

---

### H-22 — `residual_relevance` computed but unused for vault ranking
**Status: FIXED**

**File:** `specialist_state.py` `SpecialistVault`  

`rescore_against_target` set `residual_relevance = abs(corr(pred, -residual))` but vault order was MSE + complexity only; the metric was exported and ignored for selection.

**Fix applied:** Compute relevance on admission; admission rank includes `-0.08 * relevance`; retention/export order is `(mse, -relevance, complexity)`; `rescore_against_target` refreshes metrics and re-sorts.

---

### A.2 Performance narratives (P-01 … P-08)

### P-01 — Shared subtree eval cache can spike multi-GB
**Status: FIXED**

**File:** `ast.h` SubtreeCache, `execution.h` merge, `eval.h` SharedCache inserts  

Each unique hash stores full `ArrayXd(n_samples)`. Large pop × deep graphs × big N → peak RAM spike (cleared per gen, but peak matters).

**Fix applied:** LRU `SubtreeCache` with default caps (4096 entries / 256 MiB); per-thread budgets split across OpenMP workers; only mid/deep operator subtrees are inserted; `run_evolution` exports cache diagnostics.

---

### P-02 — Island non-NSGA path weaker than `run()` (primary default)
**Status: FIXED**

**File:** `evolution.h` island evolve ~3800–3857  

Default `num_islands=8` uses fitness-only sort, no macro mutations, no staged schedule, no adaptive restart, no elite aging — while single-pop `run()` has them.

**Fix applied:** Unified single-pop `run()` and multi-island `evolve_one_generation()` in `evolution.h` to share full reproduction generation logic (including `is_better_champion` sort, elite `age++`, macro mutations, staged topology boost schedule, mutation rate decay, adaptive restarts, inner-param refinement, and clean NSGA-II return).

---

### P-03 — Hybrid L-BFGS steps squared

**File:** `hybrid_optimizer.py:376–379`  

`LBFGS(max_iter=steps)` called `steps` times → up to `steps²` line searches.

**Fix:** One `optimizer.step()` with `max_iter=lbfgs_steps`.

---

### P-04 — Unvectorized / repeated full-graph FD Adam

**File:** `evolution.h` refine  

Finite-difference Adam re-evaluates full graphs per parameter; expensive and noisy at clamp plateaus.

**Proposed:** Prefer LM for linear-ish params; group FD; clip update norms; skip dead nodes (`|w| < eps`).

---

### P-05 — `sklearn_wrapper` god-module + formula eval via Python `eval`

**File:** `sklearn_wrapper.py`  

~10k LOC orchestration; formula scoring via `eval` with numpy context is slow vs C++ scorer for large candidate pools.

**Proposed:** Route display scoring through `_core.score_formula_candidates` / exact parse when available; split wrapper into modules (fast path, evolution, specialist, scoring).

---

### P-06 — Specialist vault holds residual vectors

**File:** `specialist_state.py`  

Full residual vectors × max_entries × large n → memory growth.

**Fix:** Store residual stats / sketches when `n > 10k`.

---

### P-07 — Curve feature caches coarse / collision-prone

**File:** `curve_classifier_integration.py:452–485`  

Key from coarse moments over 32 samples → wrong feature reuse across problems.

**Fix:** Stronger fingerprint or disable cross-problem cache.

---

### P-08 — Macro-mutate nest can recurse without attempt counter

**File:** `evolution.h:1954–1956`  

`return macro_mutate(parent)` with no depth limit → stack risk.

**Fix:** Max-attempt counter; fall through to wrap/lamarckian.

---

### A.3 Robustness narratives (R-01 … R-09)

### R-01 — Exception swallowing culture
**Status: FIXED**

| Location | Approx. count |
|----------|---------------|
| `sklearn_wrapper.py` | ~175 `except Exception` |
| `classifier_fast_path.py` | ~23 |
| glassbox package total | ~255 |

Many are intentional soft-fail for optional polish, but they hide remap/scoring bugs. Prefer typed exceptions + counters in diagnostics (`swallowed_errors_`).

**Fix applied (`sklearn_wrapper.py`):**
- `swallowed_errors_` counter + thread-safe `_record_swallowed_error(site, exc)` helper (S1-8 ThreadPool-safe).
- Hot-path instrumentation (12 sites): `formula_mse.eval`, `plain_unweighted_mse.eval`, `display_formula_mse.scripts_parity`, `domain_failure_rate.eval`, `safe_eval.protect_fractional_powers`, `safe_eval.data_ptr`, `fast_path_remap.full_mse`, `free_const.remap_eval`, `candidates.structure_seeds`, `final_guard.score`, `predict.eval`.
- Typed exceptions at safe fallback sites: `_clamp_int`/`_clamp_float`/`_finite_float` → `(TypeError, ValueError, OverflowError)`; scipy/sympy/classifier_fast_path/sklearn imports → `ImportError`.
- `fit()` exposes `swallowed_errors_summary_` (total + per-site count/type/last) and mirrors into `blackbox_diagnostics_["swallowed_errors"]`; summary published from every fit exit path via `_finalize_swallowed_errors_summary()` (incl. fast-path early return).
- Final-selection/polish bare-`pass` sites instrumented (9 sites): `final_score.finish_eval`, `structure_seed.promote`, `pareto.prefer_simple`, `polish.original_space_structure`, `original_structure.inlier_eval`, `original_structure.polish`, `original_structure.winner`, `original_space.holdout_rescore`, `final_guard.recompute_mse`.
- Regression tests: `tests/test_audit_r01_swallowed_errors.py` (10 tests).

Remaining ~160 bare `except Exception` are intentional candidate-loop/soft-fail polish paths (a failing candidate is a normal search event, not a bug; typing them would miss heterogeneous failure modes and crash on edge data). The critical scoring/remap/predict and final-selection sites are now observable post-fit.

### R-02 — Formula `eval` with restricted builtins

Used in `sklearn_wrapper`, `benchmark_common`, `universal_proposer`, `specialist_state`. Safer than full `eval`, but still arbitrary expression risk if formula strings are attacker-controlled.

**Mitigation:** AST whitelist, or always evaluate via C++ parser.

### R-03 — Pickle checkpoint fallback

Documented in `SECURITY.md`. `weights_only=False` allowed under trusted `models/` / `artifacts/`. Compromised local checkpoint = RCE.

**Mitigation:** Env-gated fallback; migrate all artifacts to weights-only.

**Status:** FIXED — `_load_torch_checkpoint` in both `universal_proposer.py` and `curve_classifier_integration.py` attempt weights-only first and now refuse pickle fallback unless `GLASSBOX_ALLOW_PICKLE_CHECKPOINT=1` is set in addition to a trusted `models/`/`artifacts/` path. All 16 shipped runtime `.pt` artifacts migrated to weights-only-safe format (embedded numpy arrays → plain Python lists). Regression tests `tests/test_audit_r03_pickle_checkpoint.py`.

### R-04 — Soft arithmetic “division” ≠ algebraic `/`

**File:** `eval.h` soft gate `x/sqrt(1+y²)` vs protected Division  

Intentional for stability; export/snap can mislead users into thinking soft-div is `/`.

**Fix:** Document; snap soft-div gate → `BinaryOp::Division` when weight dominates and fidelity holds.

### R-05 — Simplify identities wrong on signed domains

**File:** `simplify_advanced.h:132–154`  

- `(x^a)^b → x^(a*b)` ignores sign/parity  
- `log(exp(y)) → y` but eval Log is `log(|·|+eps)`  

**Fix:** Guard folds with domain proofs or rewrite to Abs.

### R-06 — Macro “divide” mode often does not create Division

**File:** `evolution.h` macro ~1903–1930  

`sample_binary_op()` may pick Arithmetic/Aggregation.

**Fix:** Force Division (or Arithmetic with discrete div gate) in divide branch.

### R-07 — `elastic_net_cd_cpp` returns MSE 0 for empty problems

**File:** `refine.h:46–47`  

**Fix:** Return `+inf` or throw.

### R-08 — Estimator not thread-safe

ThreadPool formula eval + unlocked diagnostics/vault mutations.

**Fix:** Document; single lock for fit-state; no shared estimator across threads.

### R-09 — Wall-clock timeout not hard-canceled inside long C++ calls

Budget split per start; C++ may overrun slightly; no absolute deadline cancel.

**Fix:** Pass absolute deadline; break multi-start when elapsed ≥ budget.

---


---

## 12. Appendix B — Documentation inconsistencies (detail)

| Claim / location | Reality |
|------------------|---------|
| README / PROJECT_MAP hybrid pipeline | Matches active code — good |
| `sklearn_wrapper` docstring “C++ weight-aware starting Phase 2” | Code already passes `y_weights` |
| `UniversalProposer.forward` features dim 370 | Config/default feature dim often 398; load pads/truncates |
| Integration import docs `scripts.curve_classifier_integration` | Package is `glassbox.curve_classifier...` |
| “Multivariate support” | Heuristic; grammar prefix + benchmark 1D reshape undercut it |
| `PROJECT_MAP` last updated 2026-06-03 | Still largely accurate; specialist/noise phases advanced since |
| ONN docs as legacy/research | Correct; but critical ONN bugs remain if path is used |
| `normalize_operators` in generate_curve_data | No-op |
| Local duplicate model class defs in train/integration | Overridden by `models.py`; edit traps |
| `expected_features = 366` stale vs `FEATURE_DIM=398` | Train path confusion |
| Exact-match simplification branch | `simplification_info` never sets `exact_match` → full SymPy path dead (`classifier_fast_path.py:3540–3559`) |
| Dead `'R' in name` routing init branch | `evolution.py:651–653` |
| Post-training node pruning looks for `output_proj` | Nodes use `output_scale` → ablation no-op (`pruning.py`) |

---

---

## 13. Appendix C — First-pass patch packs (effort bands)

### Patch pack A (Python scoring contract) — ~1–2 hours

1. C-01 non-finite policy  
2. C-02 predict  
3. C-03 timeout cap  
4. C-04 benchmark multi-var  
5. H-09 blackbox weights — **DONE**  
6. H-08 display MSE final accept — **DONE**

### Patch pack B (C++ evolution safety) — ~2–4 hours

1. C-05 NSGA continue  
2. C-06 X/y validation  
3. H-01 finite fitness — **DONE**  
4. H-02 simplify clamp — **DONE**  
5. H-03 phi clamp — **DONE**  
6. H-05 arithmetic snap mask — **DONE**  
7. H-04 LM Jacobian FD fallback — **DONE**  
8. H-06 thread-local arithmetic temperature — **DONE**  
9. H-07 seed topology / feature_idx validation — **DONE**  

### Patch pack C (ONN correctness) — ~2–3 hours

1. C-08 routing  
2. C-09 name tokens  
3. C-10 cache invalidate  
4. H-19 phased import/indices  

### Patch pack D (crossover rewrite) — larger design

Requires index-map graft design + unit tests with non-contiguous DAGs; do not ship partial offset hacks.

---

---

## 14. Appendix D — Test coverage gap matrix

| Bug area | Missing test |
|----------|----------------|
| Zero-fill / domain NaN | Reject non-finite preds in `_formula_mse`, vault, composition |
| Budget > timeout | Assert C++ `timeout_seconds` ≤ `self.timeout` |
| `nest_formulas` multi-var | `nest_formulas("sin(x0)+x1", "x0**2")` preserves `x1` |
| Python seed `x0` | C++ disabled: `sin(x0)` / `x0**2` seed not `None` |
| Family signature | Multi-op formula family for nest eligibility |
| residual_relevance | Correctness + vault ranking |
| Crossover diagnostics | Export / rate of valid XO |
| FPIP search_plan | Non-empty plan / probability when proposer supplies them |
| hard_concrete L0 | log_prob / P(z>0) numerical |
| Meta vs C++ soft-div | Parity on division blend |
| NSGA double reproduce | Single-island `use_nsga2` generation accounting |
| Empty X | Throws, no UB |
| Multi-var benchmark MSE | `x0+x1` scores finite on 2-col data |
| Routing `[0,0]` | Formula primary sources match forward |
| `'p' in name` | `omega` not initialized via power branch |

**Near-miss existing tests:** vault gates (`test_phase6_p1_fixes`), seed cap (`test_specialist_state`), guided remaining-timeout only, FPIP structure (`test_fpip_v2_schema`) — none hit the cascade defects above.

---

---

## 15. Appendix E — What looked solid (first pass, re-validated in §6)

- Protected division / `log(|x|+eps)` / Exp output clamps on **eval** path  
- Child index bounds checks in eval / compact / mark_active  
- OpenMP island outer/inner thread budgeting without nested `omp_set_num_threads` fights  
- `fitness_valid` skip re-eval (E6) design  
- Early-stop uses raw MSE intentionally  
- Weighted residual / IRLS output solve structure  
- pybind weight validation (finite, non-neg, positive sum) for main paths  
- FPIP v2 schema + tests  
- Displayed-formula scoring contract tests (partial)  
- SECURITY.md honest about pickle trust model  
- Dual-path documentation of soft Arithmetic vs discrete ops (E4 note in `eval.h`)  

---

---

*Canonical audit report. Maintain this file only — do not reintroduce a parallel second audit file.*
