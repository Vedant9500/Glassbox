# Exactness experiments

Testing one hypothesis: **the pipeline confuses a numerically-equivalent
approximation with the true closed form, and an explicit
exact-vs-approximate signal improves results.**

This folder is a lab notebook. Every experiment is opt-in and measured against
the unmodified pipeline, so a regression is reported as a regression.

## Principle: general mechanisms only

There are two kinds of "hints", and only one of them is allowed here.

**Allowed — general.** A mechanism applies to a whole class of formulas and
transfers to unseen curves:

* operator closure (products, quotients, reciprocals of existing primitives),
* unary composition closure (`op(g)` for every basis column `g`; measured, but
  it only helps formulas whose inner argument is already a basis term — see
  Finding 9),
* series detection and promotion (a truncated Taylor polynomial → `sin`, `cos`,
  `exp`, `log`, `atan`, `sinh`, `cosh`),
* known-constant snapping (`3.13852 → pi`),
* an exactness verifier trained on labelled pairs.

**Not allowed — benchmark-specific.** A term library that contains the
benchmark's own formulas (`sin(cos(x))`, `log(1+sin(x))`, `1/(1+exp(-x))`,
`sqrt(1+x^2)`, ...). It "aces" the benchmark and teaches nothing about an
unseen curve. An earlier version of `composite_basis.py` did exactly this; it
was removed after it produced +6 on tier 6. That number was not evidence.

Consequence: **the benchmark is a secondary check.** The primary metric is a
fresh PCFG-generated formula set (`heldout_eval.py`) that the pipeline has
never seen, optionally with noise.

## What counts as an approximation

| kind | example | caught by |
|---|---|---|
| truncated series | `x - x^3/6 + x^5/120` for `sin(x)` | series fingerprint, OOD blow-up |
| fitted surrogate | degree-9 polyfit of `sin(x)` | OOD blow-up |
| wrong operator set | Fourier sum for `sin(x)/x` | operator mismatch (`rational` missing) |
| constant approximation | `sin(3.13852*x)` for `sin(pi*x)` | known-constant snapping |
| structural surrogate | `2.43*sin(2x)/(1+exp(x)^2)` for `exp(-x)cos(x)` | operator match, OOD mismatch |

## Bottom line

The hypothesis was that the pipeline accepts approximations as exact answers
and that an exact-vs-approximate signal would improve results. After the
measurements:

* **The benchmark rarely accepts an approximation** (3/205; two of those are
  `e`/`pi` printed at display precision). Exactness *discrimination* is not the
  bottleneck.
* **The verifier learns the distinction** (ROC AUC 0.989 on
  in-domain-indistinguishable negatives) but has nothing to rerank; reranking
  the existing pool makes the benchmark worse.
* **The bottleneck is candidate generation.** Closing the pipeline's own basis
  under `*`, `/` and reciprocal finds exact products it could not express
  before: +16/205 exact on the full suite, +4/84 on unseen formulas, and a
  184x improvement in out-of-domain error.
* **A general mechanism can still be benchmark-specific in effect.** Unary
  composition closure adds +5 on tiers 6-8 but nothing on fresh nested
  formulas, because the benchmark's compositions share the basis vocabulary.

The useful form of "teach exactness" is therefore not a classifier that ranks
candidates; it is a **machine-precision acceptance test on newly generated
candidates**, where the generation step creates the headroom.

## Findings

### 1. The benchmark rarely accepts an approximation

```
python experiments/exactness/analyze_benchmark.py \
  results/exactness_exp/baseline_full/benchmark_latest.json
```

Baseline (205 formulas, fast-path only): **147 EXACT**, weighted 88.8%.

Of those 147 `EXACT` results only **3 (1.5%)** fail extrapolation or
structure:

* `e → 2.71828` and `pi → 3.14159` — the right constant at display precision,
* `sin(x)*log(x+1) → sin(x)*log(Abs(x)+1)` — genuinely wrong outside `x > 0`.

An earlier version of this analysis reported 9 false exacts. Six of those were
**exact trigonometric identities** (`2*sin(x)*cos(x) → sin(2*x)`,
`sin(x)^2+cos(x)^2 → 1`); the operator-set comparison counted the missing
`cos` as an error. `lib.equivalent()` now checks symbolic/numeric equivalence
over an extended domain before calling a rewrite different, and the
false-EXACT count dropped to the three above.

Consequence: **exactness errors in the benchmark are almost nonexistent**.
The official score is not hiding a pile of wrong formulas. The headroom is in
candidate *generation* — 44 of the 58 non-exact winners are genuinely
different functions, not approximations of the target.

Missing critical operators across non-exact winners: `exp` 28, `sin` 10,
`cos` 5, `log` 5, `rational` 1.

### 2. Selection is not the bottleneck

```
python experiments/exactness/diagnose_recall.py \
  results/exactness_exp/baseline_full/benchmark_latest.json
```

Of the 58 non-exact winners: 34 have a candidate whose operator set covers the
target, but **0** have a candidate that would score `EXACT`, and **0** are
rerankable. A perfect verifier over the existing fast-path pool gains nothing.
The missing piece is candidate generation.

### 3. The classifier is inert in the default fast path

`run_fast_path(..., auto_expand=True)` sets `universal_basis=True`; in that mode
`should_use_fast_path()` always returns `True` and the polynomial/trig/exp/log
terms are added regardless of predictions. Swapping `curve_classifier_v7_dual`
for `curve_classifier_multi` on tier 6 gives the identical 11/25 and 23/25
byte-identical discovered formulas.

Fine-tuning the operator head cannot move these numbers unless the fine-tune
also changes what gets *generated* (evolution hints or a non-universal basis).

### 4. Evolution helps, at a cost

Tiers 6-7, `--with-evolution --timeout 30`:

| mode | exact | weighted | runtime |
|---|---|---|---|
| fast path | 21/50 | 78.7% | ~60s |
| guided evolution | 29/50 | 85.3% | 828s |

Evolution also produces very large formulas whose displayed MSE drifts far from
the engine's internal MSE (e.g. raw 3.9e-7 vs display 2.7e-2). That drift is
itself an approximation failure and a natural target for the verifier.

### 5. A general constant check works

`--guard pi` snaps trig frequencies that are rational approximations of pi
(22/7 = 3.142857, 3.13852) to the exact constant. Tier 3: **23/25 → 25/25**.

### 6. Held-out, stratified by structure

The benchmark is a fixed public formula bank, so it cannot be the primary
metric. `heldout_eval.py` generates fresh PCFG formulas and balances them
across structural families, because a guard can only help the families it can
express:

* `product` — a product of primitives (`x^2*cos(x)`),
* `quotient` — a ratio or negative power (`(abs(x)+0.01)^-0.5`),
* `nested` — a transcendental of a non-linear argument (`sin(cos(x))`),
* `transcendental` — a plain sum of primitives (`x + cos(1.2)`).

Two seeds, 84 formulas total, fast path only:

| guard | exact | weighted | OOD median | false exact |
|---|---|---|---|---|
| none | 18/84 | 59.5% | 0.208 | 2 |
| closure | **22/84** | **61.1%** | **0.00113** | 3 |

Per family (exact / n):

| family | none | closure |
|---|---|---|
| product | 4/12 | **6/12** |
| transcendental | 10/24 | **12/24** |
| nested | 3/24 | 3/24 |
| quotient | 1/24 | 1/24 |

Three honest readings:

1. The closure gains **4 exact formulas on formulas it has never seen**, all in
   the families it can express (products and transcendental combinations).
2. The OOD median improves by ~184x because the exact product replaces a
   surrogate that fit inside the domain and diverged outside it. That is the
   exact failure mode this project set out to detect.
3. One added `EXACT` is domain-ambiguous: `x*cos(x)` sampled on `x > 0` is
   matched by `|x|*cos(x)`, which is indistinguishable on the observed domain
   and wrong for `x < 0`. No method can resolve that from the data alone.

The nested and quotient slices are untouched. Composition closure was tried
for the nested slice and did not generalise (Finding 9).

### 7. Operator closure over the pipeline's own primitives

`composite_basis.py` closes the fast path's basis under `*`, `/` and
reciprocal. There is no named formula list: columns are generated from
`build_basis_from_predictions(..., universal_basis=True)` and ranked by
correlation with the target, then OMP selects a sparse subset.

Acceptance is deliberately strict:

* the candidate must beat the incumbent's in-domain MSE,
* it must fit to **machine precision** (`< 1e-11`) — a genuine identity, not a
  dense surrogate (a dense fit can win in-domain and extrapolate worse; that
  measured as OOD median 38 → 94 before the gate),
* it must be no worse than the incumbent at the domain edges.

Tiers 6-8, fast path only:

| mode | tier 6 | tier 7 | tier 8 | exact | weighted |
|---|---|---|---|---|---|
| baseline | 11/25 | 10/25 | 4/25 | 25/75 | 73.8% |
| closure | **16/25** | **13/25** | **10/25** | **39/75** | **80.9%** |

Full suite:

| mode | exact | weighted | FAIL | runtime |
|---|---|---|---|---|
| baseline | 147/205 | 88.8% | 3 | 253s |
| closure | **163/205** | **91.7%** | 2 | 263s |

The newly exact formulas are genuine products found generically, e.g.
`exp(-x)*cos(x)`, `x^2*exp(-x)`, `exp(-x)*sin(2*x)`, `sin(x)*cos(x^2)`,
`exp(-x^2)*sin(x^2)`, `exp(-x)*sin(x)*cos(2*x)`. Only one `EXACT` in the run is
a false exact, and it is the pre-existing `sin(x)*log(x+1)` case from the
baseline.

### 8. The verifier learns the distinction, but has nothing to rerank

Trained on generated pairs, tested on benchmark pairs (`--split source`):

| slice | ROC AUC | n |
|---|---|---|
| all benchmark pairs | 0.9893 | 2273 |
| hard negatives (in-domain indistinguishable) | 0.9892 | 961 |
| in-domain negatives | 0.9935 | 1009 |

The model generalises across sources: it never saw a benchmark target during
training, and it assigns `p ≈ 0.00` to polynomial surrogates of
`sin(x)+exp(-x)` and `sin(cos(x))`. But Finding 2 stands: the fast-path pool
contains no candidate that a perfect verifier could promote, so the verifier
cannot move the benchmark.

Reranking the existing pool with it is worse than doing nothing:

| tiers 6-8 | exact | weighted | FAIL |
|---|---|---|---|
| baseline | 25/75 | 73.8% | 0 |
| verifier rerank | 23/75 | 64.4% | 12 |

The verifier prefers structurally plausible candidates that the display
evaluator then rejects, so it trades exact answers for formula-eval failures.
It is kept as a negative result and is not part of any recommended guard.

Its value is diagnostic, and as a guard on new candidate generators (e.g.
deciding whether a closure rewrite is trustworthy).

Training also exposed a labelling bug: 41% of polynomial/Taylor candidates
were labelled `exact` because OOD error was normalised by the target's own
variance, which explodes for targets such as `sin(x)+exp(-x)`. Labels are now
by construction: a fit is exact only when `lib.equivalent()` says it is the
same function; rescaled/perturbed candidates stay positive because their
structure is correct. The rebuilt dataset has 6271 pairs (3602 exact, 2669
approximation, 107 hard negatives).

### 9. Composition closure: benchmark gain, no general gain

`composition_closure.py` closes the basis under a **uniform** set of unary
operators: for every column `g` it adds `sin(g)`, `cos(g)`, `exp(g)`,
`log(g)`, `sqrt(g)`, `tanh(g)`, `atan(g)`, `abs(g)` when the domain allows.
No named formulas. It is escalated only after the product closure fails, and
it must fit to machine precision with at most **two** terms.

Both constraints were learned the hard way. Running composition first, over the
full 560-column dictionary, produced 3-term surrogates that fitted to 1e-11,
displaced exact product solutions, and failed the display evaluator: tiers 6-8
dropped to 35/75 exact with 3 FAILs. A large dictionary can approximate a
curve without being its formula; escalation plus a 1-2 term identity rule
restores the product wins and admits only genuine compositions.

Tiers 6-8:

| mode | tier 6 | tier 7 | tier 8 | exact | weighted |
|---|---|---|---|---|---|
| baseline | 11 | 10 | 4 | 25/75 | 73.8% |
| closure | 16 | 13 | 10 | 39/75 | 80.9% |
| closure + composition | **18** | 13 | **13** | **44/75** | **83.6%** |

The five added exacts are real one-term compositions: `exp(sin(x))`,
`sin(exp(x))`, `sin(exp(-x))`, `sin(x*cos(x))`, `cos(sin(x))`. Only one
`EXACT` in the run is false, and it is the pre-existing `sin(x)*log(x+1)`
case.

**Held-out: no gain at all.** 30 fresh formulas, seed 1234:

| mode | exact | weighted | nested exact | OOD median |
|---|---|---|---|---|
| closure | 8/30 | 60.0% | 0/8 | 1.9e-4 |
| closure + composition | 8/30 | 60.0% | 0/8 | 1.9e-4 |

Why: the benchmark's nested formulas are built from the *same vocabulary* as
the pipeline's basis (`cos(x)`, `x^2`, `exp(-x)`, `sin(x)`), so `op(g)` can
assemble them. Fresh PCFG formulas use arbitrary constants
(`sin(x/(x^2+0.1))`, `cosh(tanh(sin(1.2211*|x|^0.33)))`), so the required inner
column does not exist and composition cannot build it. The +5 is an artifact
of vocabulary alignment, not evidence that composition generalises. It is kept
as an experiment, not recommended as a general improvement.

This is the same trap as the original named-term library, one level up: a
mechanism can be general in form and still only fire on the benchmark because
the benchmark shares the basis vocabulary.

## Experiment ladder

| id | experiment | status | primary result |
|---|---|---|---|
| E0 | baseline + exactness analysis | done | 147/205, 3 false exact (1.5%) |
| E0.1 | classifier swap ablation | done | 0 on tier 6 |
| E0.5 | known-constant snapping (`pi`) | done | +2 on tier 3 |
| E1 | exactness verifier on hard negatives | done | AUC 0.9997; 0 rerank headroom |
| E2 | verifier rerank of fast-path pool | done | 0 gain, ceiling is 0 |
| E3 | general operator closure | done | full suite 147→163 exact, 88.8→91.7%; held-out +4/84, OOD 184x |
| E3.5 | series promotion | done | inert on benchmark and held-out |
| E4 | unary composition closure (`op(g)`) | done | benchmark +5, held-out 0 — vocabulary artifact |
| E5 | classifier fine-tune with hard negatives | proposed | only meaningful with evolution |

## Dataset

`build_dataset.py` produces the exact/approximation pair database:

```
python experiments/exactness/build_dataset.py \
  --out data/exactness_pairs --n-generated 400 --seed 0
```

Per target formula it samples the curve, then generates Taylor truncations,
least-squares polynomial surrogates, Pade approximants, and structurally
correct formulas with slightly wrong constants (positives).

Labels are by construction, not by threshold: a `taylor`/`polyfit`/`pade`
candidate is exact only when `lib.equivalent()` proves it is the same function;
`identity`/`scaled`/`perturbed` candidates are positive because their structure
is right. The earlier threshold rule mislabelled 41% of fits as exact, because
a polynomial surrogate of an exploding target (`sin(x)+exp(-x)`) keeps a small
*relative* OOD error.

The approximations are stored as **negatives**, never as positives with the
target's operator labels — labelling them correct would teach the model to
accept approximations.

A per-target watchdog bounds SymPy series calls; a pathological formula is
skipped instead of hanging the run. Pairs are written incrementally.

Current build: 605 targets, 568 curves, 1422 pairs, 100 hard negatives.

## Modules

| file | role |
|---|---|
| `lib.py` | series fingerprint, OOD metrics, equivalence, labels, structure |
| `build_dataset.py` | pair database builder |
| `analyze_benchmark.py` | false-EXACT analysis of a benchmark report |
| `diagnose_recall.py` | is the correct candidate already in the pool? |
| `composite_basis.py` | general operator closure (`*`, `/`, reciprocal) |
| `composition_closure.py` | unary composition closure (`op(g)`) |
| `series_rationalize.py` | truncated polynomial → closed form |
| `train_verifier.py` | exact-vs-approximation verifier |
| `run_with_guard.py` | benchmark runner with opt-in guards |
| `heldout_eval.py` | primary metric: unseen generated formulas |
| `stratify_heldout.py` | per-family breakdown of held-out reports |

## Metrics

* `EXACT` — benchmark's own in-domain label (MSE < 1e-6, <= 10 terms).
* `equivalent_rewrite` — the discovered formula is a different spelling of the
  same function (`sin(2*x)` for `2*sin(x)*cos(x)`); not an approximation.
* `false_exact` — `EXACT` but OOD-relative MSE > 1e-2, missing critical
  operators, or a series fingerprint, *and* not equivalent to the target.
* `struct_exact` — `EXACT` *and* OOD-relative MSE <= 1e-3 *and* structurally
  right.
* `rerankable` — a non-exact winner whose pool contains a candidate that would
  score `EXACT`.
* `ood_rel` — OOD MSE divided by the target's OOD variance, so unbounded
  targets such as `exp(x)` are compared fairly.
