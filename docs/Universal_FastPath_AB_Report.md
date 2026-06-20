# Universal Fast-Path A/B Report

Last updated: 2026-06-03.

This report is a qualitative rollout summary. For current quantitative results,
use generated benchmark artifacts under `results/` from the same commit/worktree
being evaluated.

## Compared Modes

1. Legacy fast path: classifier/basis/exact-match path with no proposer input.
2. Proposer-guided path: neural proposer emits candidates/priors and guided
   evolution consumes them.
3. Hybrid path: fast path handles easy cases; proposer and guided evolution help
   uncertain, residual-suspicious, or out-of-basis cases.

## Current Implementation

- Fast path: `scripts/classifier_fast_path.py`.
- Proposer: `glassbox/universal_proposer/universal_proposer.py`.
- FPIP v2 schema: `glassbox/sr/fpip_v2.py`.
- Estimator routing: `glassbox/sr/sklearn_wrapper.py`.
- Benchmark routing: `scripts/benchmark_suite.py`.

## Qualitative Findings

- Easy polynomial/trigonometric formulas are still best served by the classifier
  fast path and exact-match/regression path.
- The proposer is most useful as structural guidance: candidate skeletons,
  uncertainty, priors, and search-plan hints.
- Hybrid routing is safer than proposer-only routing because poor proposer
  guesses do not need to displace exact fast-path wins.
- Guided C++ evolution benefits from seed graphs, but seed budgets and prior
  trust must remain bounded to avoid biasing the search toward bad candidates.

## Rollback and A/B Controls

- `GLASSBOX_USE_LEGACY_FASTPATH=1`
- `scripts/benchmark_suite.py --disable-proposer`
- `scripts/benchmark_suite.py --trust-proposer-plan`
- `GlassboxRegressor(use_universal_proposer=..., universal_proposer_shadow_mode=...)`

## Reporting Requirements

Future A/B reports should include:

- command line and git/worktree context,
- model checkpoint paths,
- tier/dataset set,
- seed set,
- device and timeout/core budget,
- exact/approx/loose/fail counts,
- displayed MSE and raw MSE drift,
- time-to-first-acceptable where available,
- separate counts for fast-path wins, proposer-seeded wins, specialist wins, and
  evolution wins.

## Recommendation

Keep hybrid routing as the preferred direction, but treat proposer output as
guidance rather than authority. Claims of improvement must be supported by
current displayed-formula benchmark results.
