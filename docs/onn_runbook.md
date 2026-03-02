# ONN Runbook (Detailed)

## Scope
Detailed operating policy for ONN fast-path/evolution changes. Keep `SKILL.md` concise and use this file for process depth.

## Release Gates (Required)
Any fast-path pipeline change must satisfy all gates below before becoming default:

1. No quality regression gate
- EXACT count delta on target tiers (2-5) must be >= 0 versus previous default.
- Weighted score delta must be >= 0.

2. MSE safety gate
- Mean MSE ratio (new/old) must be <= 1.05 on tiers 2-5.

3. Runtime improvement gate
- Mean runtime delta must be <= -3% (faster) or a documented exception is required.

4. Stability gate
- No hangs/crashes in post-processing across full benchmark run.
- Post-processing fallback paths must complete and preserve valid formula output.

If any gate fails, do not change defaults.

## Rollback Criteria and Procedure

### Rollback Triggers
- EXACT count decreases by 1 or more on any monitored tier set.
- Mean MSE worsens by more than 5% on tiers 2-5.
- Runtime improves but introduces score regressions.
- New hangs/timeouts appear in simplification/post-processing.

### Rollback Procedure
1. Revert changed fast-path sequencing/gating logic.
2. Re-run smoke matrix to confirm restoration.
3. Record regression cause and failing formulas in change notes.
4. Re-open optimization under feature branch with explicit guardrail tests.

## Ownership and Review

### Ownership
- Primary owner: SR/ONN architecture maintainer.

### Required reviewers
1. Python fast-path owner (`scripts/classifier_fast_path.py`)
2. Benchmark owner (`scripts/benchmark_suite.py`)
3. C++ evolution owner when fallback behavior changes

### Change control
- Default-strategy changes require benchmark evidence in PR description.
- Safety-rule changes (`eval.h`/domain guards) require explicit risk note.
- New heuristics must include a disable path or rollback plan.

## Failure Playbook

### Symptom: Post-simplification stalls
- Action: confirm snap-only mode for large formulas.
- Verify: benchmark completes without timeout and formula remains valid.

### Symptom: Refinement diverges (MSE spikes)
- Action: tighten refinement trigger windows and acceptance guards.
- Verify: candidate rejected unless score and MSE gates pass.

### Symptom: Fallback over-triggering
- Action: inspect term counting source (structural vs simplified) and scoring thresholds.
- Verify: fallback only for true non-exact cases.

### Symptom: Better runtime, worse exactness
- Action: fail release gate and revert default.
- Verify: compare against previous default on tiers 2-5.

## Test Matrix

### Pre-merge smoke
1. Single known polynomial formula (tier 2/4 representative).
2. Single periodic formula (tier 3 representative).
3. One rational/nested formula (tier 6 representative).
4. One complex mixed formula where simplification guardrails are likely to trigger.

### Pre-release benchmark
- Required set: tiers 2-5.
- Optional confidence set: tier 6.
- Report must include:
  - EXACT / APPROX / LOOSE / FAIL counts
  - weighted score
  - mean MSE
  - mean runtime
  - formulas with largest MSE deltas

## Example Commands
- Standard run:
  - `python scripts/benchmark_suite.py --tier 6 --device cpu --quiet`
- Sensitivity run:
  - `python scripts/benchmark_suite.py --tier 2 --tier 3 --tier 4 --tier 5 --device cpu --quiet`
