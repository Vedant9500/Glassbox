# Glassbox Implementation Plan

**Source:** `glassbox_codebase_audit_tracker.md` (S1, S2, S4, S5 complete; S3/S6–S10 audit pending)  
**Goal:** Turn audit findings into ordered, shippable workstreams.  
**Convention:** phases are *implementation* waves (fix → harden → speed), not audit section numbers.

| Legend | Meaning |
|--------|---------|
| **Done** | Fixed in code (verify tests still green) |
| **P0** | Correctness / silent wrong / crash on common paths |
| **P1** | Experiment-changing correctness or major runtime cost |
| **P2** | Maintainability / micro / polish |
| **O\*** | Optimization backlog item |

---

## Snapshot

### Already shipped (do not re-open unless regression)
| ID | What |
|----|------|
| **S1-1..S1-3, N2, E1, E2/N5** | Phase 0 correctness (2026-07-20) |
| **N1** | Sticky public `loss_mode` after auto Huber/soft-MAD — restore on fit entry/exit |
| **S5-1** | `UnaryOp::Abs` — abs is true abs (not Power p=1) |
| **S5-2** | Candidate scoring uses graph eval (`formula_to_graph` + `evaluate_graph`) |
| **S5-3** | Variable/folded powers compile (const-fold + exp/log rewrite) |
| **S5-4** | Printer matches soft-div / protected-div / aggregation; multi-feature names |
| **S5-7** | `exp(log(·))` → `abs` (via Abs) |
| **O12, O13** | Abs + printer backlog items |

### Still open (this plan)
- **P0 open:** N2, E1, E2/N5, S1-1, S1-2, S1-3  
- **P1 open:** ~30 items across S1/S2/S4/S5  
- **P2 open:** remaining polish  
- **Unaudited:** S3, S6–S10 may add findings — schedule after Phase 0–2 or in parallel with Phase 6

### Recommended implementation order
```
Phase 0  Correctness stop-the-bleed (P0)
Phase 1  Noise & metric contracts (search vs display)
Phase 2  Evolution search reliability
Phase 3  Graph/eval/refine hardens (S5 remainder)
Phase 4  Orchestration quality & sklearn contract (S1 P1)
Phase 5  Performance & defaults
Phase 6  Finish audit (S6→S3→S7–S10) + fold new findings
Phase 7  API polish & cleanup (P2)
```

---

## Phase 0 — Correctness stop-the-bleed (P0)

**Goal:** No silent wrong public results; no wasted identical islands; no false “exact” under robust loss.  
**Exit:** Targeted unit/integration tests; smoke fit on clean sin/poly.

| # | ID | Work | Primary paths | Acceptance |
|---|-----|------|---------------|------------|
| 0.1 | **S1-1** | `check_is_fitted` on real fit attrs (`formula_`, `n_features_in_`); unfitted `predict` → `NotFittedError` | `sklearn_wrapper.py` predict | bare `GlassboxRegressor().predict` raises |
| 0.2 | **S1-2** | Proposer rapid-hit assigns **argmin** formula, not `candidate_formulas[0]` | `fit` rapid-hit branch | unit: min-MSE formula kept |
| 0.3 | **S1-3** | Clear sticky fit state at `fit` start: `evolution_candidate_*`, `pareto_front_`, `nodes_`, `output_weights_`, etc. | `fit` entry reset | second fit cannot resurrect prior problem formula |
| 0.4 | **N2** | Structure-aware diffuse-noise detector (not poly residual only); include sin/exp/rational probes or residual structure | `_estimate_diffuse_noise_ratio` | clean sin/exp ratio low; no auto-Huber; poly noise still detects |
| 0.5 | **E1** | Per-island RNG seed offset (and/or distinct streams); optional seed_graph sharding | `evolution.h` `run_islands` | fixed `random_state` + `num_islands>1` → islands differ |
| 0.6 | **E2 / N5** | Early-stop / “exact” gates use **`raw_mse`** (unweighted MSE); search selection may keep robust `objective_mse` | `evolution.h` early-stop; wrapper reads | Huber path does not early-stop while raw MSE large |

**Also close:** O6 (N2 detector), O9 (island seeds), O10 (early-stop raw_mse).

**Tests:** extend `test_robust_loss`, `test_sample_weight_contract`, `test_cpp_parity` / island determinism probes, sklearn unfitted/predict, multi-fit sticky test.

---

## Phase 1 — Noise & metric contracts

**Goal:** Auto noise path is conservative; search may weight/Huber; **display/protocol always plain unweighted MSE**.

| # | ID | Work | Notes |
|---|-----|------|-------|
| 1.1 | **N3** | Soft-MAD: remove or gate `retained_all_features` force; multi-column residual probes | Avoid soft+Huber on clean multi-feature |
| 1.2 | **N4** | Activate phase-3/6 unweighted guards for `diffuse_noise_huber` as well as soft-MAD | N2 residual risk if auto-Huber still fires |
| 1.3 | **N6** / **S1-12** / **O8** | Local unweighted `_display_formula_mse` — no `scripts.benchmark_common` import fallback to robust | Hard display/search separation |
| 1.4 | **S1-6** | Document and audit decision gates: which use weighted vs unweighted; align skip/accept thresholds | Overlaps S3 |
| 1.5 | **S5-9** | Weight-aware specialist `refine.h` **or** route specialists through evolution residual | Elastic net / freq / power refiners unweighted today |
| 1.6 | **S5-10** | Document dual clamp domains; optionally soft-protect exact scorer if any remaining exact path | Graph path is primary for ranking (S5-2 done) |

**Also close:** O7 (soft-MAD multi-col), O8 (local display MSE).

**Tests:** clean multi-feature no soft force; auto-Huber + guards; display MSE never uses weights/Huber.

---

## Phase 2 — Evolution search reliability

**Goal:** Seeds and selection reflect diversity and raw accuracy under islands.

| # | ID | Work |
|---|-----|------|
| 2.1 | **E3** / **O11** | Raise seed capacity under islands (not `pop/4` global starvation) |
| 2.2 | **E5** | Selection/tie-break prefer better **raw_mse** when fitness equal / for export |
| 2.3 | **E7** | Review OMP nested / `omp_set_num_threads` inside parallel islands (E7) |
| 2.4 | **E4** | Kitchen-sink + soft arithmetic bias — document; optional discrete commit temperature schedule before export (ties S3) |
| 2.5 | **E10** | `arithmetic_temperature` process-global race — thread/local or scoped restore (scorer already sets high temp carefully) |

**Defer heavy:** **E6** full pop re-eval (Phase 5 performance).

---

## Phase 3 — Graph / eval / refine harden (remaining S5)

**Goal:** Eval cache correctness; formula export completeness; refine reaches nested params.

| # | ID | Work |
|---|-----|------|
| 3.1 | **S5-5** / **O14** | Structural hash: more decimals or full bits for SharedCache key; keep coarse quantize only for CSE if desired |
| 3.2 | **S5-6** / **O15** | Unify output-weight cutoffs (eval vs `get_formula_string` vs compact) — no silent dropped terms |
| 3.3 | **S5-8** | Inner refine/snap: include nested unaries that are ancestors of active outputs, not only nonzero output-weight nodes |
| 3.4 | **S5-11** / **O16** | `get_child` by const ref / Map — hot-path copy removal |
| 3.5 | **S5-12** | `simplify.h` use `get_arithmetic_temperature()`; fold Aggregation constants |
| 3.6 | **S5-13–S5-16** | P2: inactive_nodes units; partial-eval bounds; legacy `evaluate_fitness`; dual cache cleanup |

**Done in this area:** S5-1..S5-4, S5-7.

---

## Phase 4 — Orchestration quality & sklearn contract (S1 P1)

**Goal:** Predict/metadata honest; selection less optimistic; safer parallelism.

| # | ID | Work |
|---|-----|------|
| 4.1 | **S1-4** / **O5** | Separate `n_features_search_` vs public `n_features_in_`; always restore original dim on early exact / `_finish_with_formula` |
| 4.2 | **S1-5** | True held-out slices for selection (carve once before structure/evo; never train on tail holdout) |
| 4.3 | **S1-8** / **O4** | Thread-safe formula cache (or disable shared cache in ThreadPool); avoid global `np.random.seed` |
| 4.4 | **S1-10** | Rename or fix CV skip guard: either real refit CV or document residual-stability only; fail closed on small n |
| 4.5 | **S1-13** | Decouple residual boosting from `use_guided_evolution` flag (or rename flag) |

**S3 dependency:** scoring/refine/cleanup/guards audit (S3) should land before over-investing in 4.2/1.4.

---

## Phase 5 — Performance & defaults

**Goal:** Cut wall time without regressing recovery quality.

| # | ID | Work |
|---|-----|------|
| 5.1 | **S1-9** / **O1** | Defaults: `multi_start_runs=1` escalate; revisit islands/timeout for easy problems |
| 5.2 | **S1-7** / **O2,O3** | Dedup blackbox prep / second fast-path / repeated structure probes |
| 5.3 | **E6** / **O11** | Skip re-eval of clean elites; cache generation partial |
| 5.4 | **S5-11** | (if not done in 3.4) eval copy elimination |
| 5.5 | **E1** residual | Island clone already fixed in P0 — measure island diversity ROI |

---

## Phase 6 — Finish remaining audit sections

**Goal:** Complete findings inventory before claiming whole-codebase done.

| Order | Section | Focus |
|-------|---------|--------|
| 6.1 | **S6** | Bindings, seed graphs, setup/export, dtype/contiguity, seed_graphs plumbing |
| 6.2 | **S3** | Scoring, refine, snap, parsimony, final guards (Python) — overlaps N4, S1-5/6 |
| 6.3 | **S7** | Blackbox multi-feature ranking / remap |
| 6.4 | **S8** | Specialist vault poisoning / composition |
| 6.5 | **S9** | Curve classifier integration (not train/data gen) |
| 6.6 | **S10** | Proposer, Python evolution, optimizers, FPIP |

After each: fold new P0/P1 into Phases 0–5 or a **Phase 6.x fix wave**.

---

## Phase 7 — API polish & cleanup (P2)

| ID | Work |
|----|------|
| **S1-11** | Export `GlassboxRegressor` from `glassbox.sr`; fix `__all__` / FPIPv2 name |
| **S1-12** | Already largely O8 — remove scripts coupling |
| **S5-14** | Optional grammar (implicit mul); power display parity |
| **S5-15, S5-16** | Partial-eval safety; dead dual fitness path |
| **E\*** remaining P2 | Global temp docs, kitchen-sink docs |

---

## Cross-cutting themes → phase map

| Theme | Primary phase |
|-------|----------------|
| Weighted vs unweighted metrics | 1, 4, 6.2 (S3) |
| Soft-graph vs display formula | 3 (export), 6.2 (S3 cleanup) |
| Subtree hash quantization | 3.1 |
| Seed graphs / island diversity | 0.5, 2.1, 6.1 |
| Complexity under noise | 1.2, 2, 6.2 |
| Double evaluation | 5 |
| Sticky estimator state | 0.3, N1 done |
| sklearn fit contract | 0.1, 4.1 |
| Multi-start / double pipeline | 5.1–5.2 |

---

## Suggested sprint packaging

### Sprint A (critical) — Phase 0 only
S1-1, S1-2, S1-3, N2, E1, E2/N5  
**Why first:** public API lies + clean-data Huber + island clone waste + false exact.

### Sprint B — Phase 1 + quick O8
N3, N4, display MSE local, guard activation under diffuse Huber.

### Sprint C — Phase 2 + Phase 3.1–3.3
Island seeds residual, seed cap, hash quantize, weight thresholds, nested refine.

### Sprint D — Phase 4 + 5 defaults
n_features_in_, holdouts, thread-safe cache, multi_start=1.

### Sprint E — Phase 6 audit completion + backlog
S6 → S3 → S7–S10; reprioritize.

---

## Dependency graph (simplified)

```
S1-3 (clear state) ──────────────┐
S1-1 / S1-2 (API) ───────────────┼──► trustworthy multi-fit experiments
N2 (noise detect) ──► N4 guards ─┘
E1 (island seeds) ──► E3 seed cap ──► recovery quality benches
E2 raw early-stop ──► N5 / Exact claims
S5-5 hash ──► SharedCache correctness
S5-6 thresholds ──► formula export = what was fit
S1-4 n_features ──► blackbox predict metadata
S3 audit ──► deep scoring/guard redesign (S1-5/6)
```

---

## Tracking checklist (copy into issues)

### Phase 0
- [x] S1-1 unfitted predict
- [x] S1-2 rapid-hit argmin
- [x] S1-3 sticky fit state
- [x] N2 diffuse Huber FP
- [x] E1 island seed offset
- [x] E2/N5 early-stop on raw_mse

### Phase 1
- [ ] N3 soft-MAD retained_all force
- [ ] N4 guards under diffuse Huber
- [ ] N6/O8 local display MSE
- [ ] S1-6 gate metric audit
- [ ] S5-9 refine.h weights

### Phase 2
- [ ] E3 seed capacity
- [ ] E5 raw_mse selection
- [ ] E7 OMP nesting
- [ ] E4/E10 soft arith / global temp

### Phase 3
- [ ] S5-5 hash quantize
- [ ] S5-6 weight cutoffs
- [ ] S5-8 nested refine
- [ ] S5-11 get_child copies
- [ ] S5-12 simplify temp
- [ ] S5-13..16 P2

### Phase 4
- [ ] S1-4 n_features_in_
- [ ] S1-5 true holdouts
- [ ] S1-8 thread-safe cache / RNG
- [ ] S1-10 CV guard honesty
- [ ] S1-13 residual flag naming

### Phase 5
- [ ] S1-9 / O1 defaults
- [ ] S1-7 / O2–O3 double work
- [ ] E6 elite re-eval skip

### Phase 6–7
- [ ] Audit S6, S3, S7–S10
- [ ] S1-11 export API
- [ ] Remaining P2

---

## Out of scope (until reopened)
- `eigen/`, `build/`, diffusion, scratch
- Full rewrite of soft arithmetic (E4 is bias note, not full redesign)
- Training curve-classifier datasets (S9 de-prioritize train)

---

## How to use this plan
1. Open Sprint A issues from Phase 0 table only.  
2. After each fix: mark tracker finding `fixed`, add test, one-line progress log.  
3. Do not mark audit section `done` for S3/S6–S10 until Phase 6 pass.  
4. Re-run: `test_robust_loss`, `test_sample_weight_contract`, `test_phase6_noise_guards`, `test_cpp_simplification`, `test_cpp_candidate_scoring`, plus new unfitted/sticky/island tests.

