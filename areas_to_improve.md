# Areas to Improve

## Current Status (2026-06-03)

This is a research backlog, not a current implementation map. Several items now
have partial implementations: adaptive compute budgets, displayed-formula
scoring, blackbox feature ranking, interaction discovery, C++ candidate scoring,
specialist composition, residual stages, and universal-proposer seeding. For the
live pipeline, see `docs/PROJECT_MAP.md` and `docs/Research_Roadmap.md`.

## Research Direction

The strongest future direction is reliable symbolic regression under messy, real-world conditions, not only clean Nguyen/Feynman-style toy formulas.

Current symbolic regression systems often return a best expression, but they struggle to know whether that expression is structurally correct, whether it will extrapolate, and whether more compute would help. Glassbox has a promising architecture for this because it already combines:

- fast-path analytical/basis recovery,
- neural proposer skeletons and operator priors,
- uncertainty-aware routing,
- residual diagnostics,
- native C++ guided evolution,
- displayed-formula MSE checks instead of only raw engine MSE.

A strong paper-worthy target:

> Uncertainty-aware symbolic regression that knows when to trust a formula, when to refine it, and when to spend more compute.

## Problems Current SR Systems Struggle With

### 1. Noisy Data Without Overfitting

Evolutionary SR often fits noise with large, ugly formulas. Neural SR can propose plausible but wrong structures. Glassbox can work toward using uncertainty, residual diagnostics, formula complexity, and displayed-formula validation to avoid false symbolic discoveries.

### 2. High-Dimensional Inputs

Search explodes as feature count grows. Many SR methods perform well on 1D or low-dimensional symbolic benchmarks but degrade on PMLB-style datasets. Glassbox should improve active-variable detection and interaction discovery before running expensive evolution.

### 3. Extrapolation Reliability

Many methods optimize interpolation MSE but produce formulas that fail outside the training range. Glassbox should evaluate formulas on extrapolation splits and route suspicious formulas back into refinement.

### 4. Compute Allocation

Most SR systems spend similar compute on easy and hard problems. Glassbox can make compute adaptive: solve easy cases with fast-path recovery, then escalate only when uncertainty, residuals, or validation failures justify it.

### 5. Operator/Library Mismatch

SR quality depends heavily on the allowed operator set. Too few operators miss the truth; too many operators explode the search. Glassbox can use classifier/proposer priors to choose operator budgets dynamically.

## Benchmark Plan

Build a benchmark suite focused on robustness, not just clean exact recovery:

- interpolation train/test split,
- extrapolation split outside the training domain,
- multiple noise levels,
- multi-feature problems with irrelevant variables,
- fixed wall-clock budgets,
- repeated seeds.

Important metrics:

- exact recovery rate,
- time-to-first-exact,
- time-to-first-acceptable,
- displayed-formula MSE,
- raw-vs-displayed MSE divergence,
- extrapolation MSE,
- false-confidence rate,
- active-variable recovery,
- formula complexity,
- robustness across seeds.

## Ablations Needed

To support a research claim, compare:

- full system,
- no neural proposer,
- no fast-path,
- no uncertainty routing,
- no candidate skeleton seeding,
- no residual diagnostics,
- raw C++ evolution only,
- fast-path only.

## External Baselines

Compare against strong or commonly used SR systems where feasible:

- PySR,
- gplearn,
- GP-GOMEA or other GP baselines,
- neural/amortized SR baselines if practical,
- SRBench-style reference methods.

## Near-Term Engineering Work

- Centralize the latest optimized path inside `GlassboxRegressor`.
- Make benchmark scripts call the same public estimator path.
- Improve multi-feature guided evolution support.
- Make missing classifier/proposer/C++ backend handling beginner-friendly.
- Make errors user-facing instead of internal messages like `Guided evolution skipped`.
- Keep displayed-formula scoring as the primary metric.
- Add confidence and routing diagnostics to result objects.
