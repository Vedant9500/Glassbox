# Specialist Composition Layer Plan

## Goal

Build a shared specialist-state layer that captures promising partial formulas, detects complementary behavior, composes a small number of high-value candidates, and reuses the existing refinement, postprocessing, and seeded evolution pipeline.

This is meant to address the current failure mode where evolution often finds formulas that are close, structurally useful, or locally strong, but fails to assemble them into the final expression.

## What Already Exists

The current codebase already has several pieces we should reuse rather than replace.

### Candidate pool and refinement

- Candidate formulas are built and shared in `glassbox/sr/sklearn_wrapper.py` via `_build_blackbox_candidate_formulas()`.
- Candidate formulas are rescored, affine-adjusted, constant-refined, and validation-ranked in `_refine_candidate_formulas()`.
- Candidate formulas are pruned through `_prune_blackbox_candidate_formulas()`.

This means we already have a strong candidate-screening path. The missing piece is semantic composition, not basic candidate management.

### Seeded evolution

- Candidate formulas can already be converted into C++ seeds through `glassbox/sr/cpp/seed_graph_builder.py` in `build_seed_graphs_from_candidates()`.
- Those seed graphs are already passed into C++ evolution as `seed_graphs_py` in `glassbox/sr/sklearn_wrapper.py`.
- The C++ engine already supports seeding a portion of the population from those graphs.

This is the cleanest integration point for merged specialist formulas that are close but not yet good enough.

### Residual-stage symbolic fitting

- The current pipeline already has a second-stage residual fit in `_stage_residual_symbolic_fit()`.
- That is the closest existing mechanism to "this formula got most of the structure right, now evolve the remaining error."

This should be reused once merged specialist formulas become available.

### Final postprocessing

- Final formula cleanup already exists through `_reduce_formula_noise()` and `_simplify_formula()`.
- These are applied near the end of the fit path and should remain the standard cleanup stage for any composed formula.

This is important because the composition layer will naturally generate some noisy or redundant terms, and the current cleanup logic is the right place to control them.

### Shared metadata pattern

- `glassbox/sr/blackbox_preprocessor.py` already uses a state object (`BlackboxState`) to carry shared search metadata such as selected features, interaction terms, and seed formulas.

This suggests that a separate `SpecialistState` or `CompositionState` object will fit the existing architecture well.

## What Does Not Exist Yet

The following capabilities are not currently implemented:

- No per-candidate residual-profile memory.
- No per-candidate interval or segment scoring.
- No complementarity detection between candidate formulas.
- No composition stage that tries controlled formula merges before evolution.
- No validation-gated reseeding from merged formulas.

That means the new layer should be added as a targeted extension, not as a rewrite of the current pipeline.

## Architectural Direction

Do not put this into `BlackboxState`.

`BlackboxState` is for feature-reduction and multivariate search preparation. The specialist composition layer is a different concern: it is about remembering how formulas behave against the target and using that to compose better starting points.

Recommended structure:

- New module: `glassbox/sr/specialist_state.py`
- New state object: `SpecialistState` or `CompositionState`
- Owned by `GlassboxRegressor.fit()`
- Invoked after candidate refinement and before final evolution decision

The specialist layer should consume the current candidate pool and return:

- enriched specialist diagnostics
- optional composed candidate formulas
- optional composed seed formulas to inject into the existing seed pipeline

## Minimum Specialist Metadata

Each specialist candidate should carry:

- `formula`
- `source`
- `mse`
- `validation_mse`
- `validation_r2`
- `complexity`
- `risk_score`
- `generalization_gap`
- `prediction_vector`
- `residual_vector`
- `segment_scores`
- `family_signature`

This is enough to support complementarity detection and controlled composition without introducing a large new framework.

## Phase Plan

## Phase 0: Instrumentation Only

### Purpose

Verify that complementary specialists actually appear often enough in the current benchmark failures to justify the new layer.

### Build

- Add a lightweight specialist analyzer after candidate refinement.
- Segment the 1D domain into coarse equal bins first: `2`, `4`, maybe `8`.
- For each candidate formula, compute:
  - per-segment MSE
  - per-segment R2
  - residual vector
  - residual hot spots
- For each candidate pair, log a complementarity score.

### Reuse

- Candidate list from `_build_blackbox_candidate_formulas()`
- Existing formula evaluation through `_safe_eval_formula_array()`
- Existing validation split patterns used in candidate scoring

### Rules

- Do not change selection behavior in this phase.
- Diagnostics only.
- Keep the pool small, such as top `6` to `12` candidates.

### Success Criteria

- We repeatedly observe cases where two non-winning formulas are strong in different segments or residual patterns.
- Those cases correlate with current "close but not exact" failures.

## Phase 1: Shared Specialist State

### Purpose

Create a formal data structure that stores specialist formula behavior in one place.

### Build

- Add `glassbox/sr/specialist_state.py`
- Add dataclasses such as:
  - `SpecialistCandidate`
  - `SpecialistPairScore`
  - `SpecialistState`
- Add helpers for:
  - segment scoring
  - complementarity scoring
  - duplicate suppression
  - family-signature assignment

### Reuse

- `_refine_candidate_formulas()` output as the main entry pool
- existing complexity and risk metrics
- existing formula family heuristics where useful

### Rules

- Keep this phase read-only with respect to search behavior at first.
- Prefer compact diagnostics over broad state accumulation.

### Success Criteria

- The pipeline can produce a stable specialist-state object for inspection.
- No change in benchmark behavior yet.

## Phase 2: Safe Composition Screening

### Purpose

Try a very small number of controlled candidate compositions before evolution.

### Initial composition templates

Start with only:

- `f + g`
- `f * g`
- residual correction forms that are algebraically equivalent to additive correction

Do not start with piecewise final formulas.

### Why

- `f + g` naturally matches the current residual-stage logic.
- `f * g` covers envelope-times-oscillation and similar structures.
- Piecewise formulas can become benchmark-cheating local stitches if introduced too early.

### Build

- Pair selector that chooses a very small number of high-complementarity, low-redundancy pairs.
- Composition generator that emits only a few merged candidates.
- Validation-gated scoring of merged formulas.
- Constant refinement and candidate rescoring on those merged formulas.

### Reuse

- `_score_formula_candidate()`
- `_refine_formula_constants()`
- `_refine_candidate_formulas()`
- existing complexity and risk scoring

### Rules

- Hard cap the number of tested pairs.
- Hard cap the number of generated merged formulas.
- Reject pairs whose predictions are nearly duplicates.
- Reject merged formulas that only improve train fit but not validation.

### Success Criteria

- Some current failures improve from screening alone.
- Merged formulas stay reasonably compact after cleanup.

## Phase 3: Feed Winning Compositions Into Seeded Evolution

### Purpose

When a composed formula is clearly better but still imperfect, use it as a guided starting point for evolution rather than hoping evolution discovers the same structure again.

### Build

- Insert accepted merged formulas back into the candidate pool.
- Let them flow into `build_seed_graphs_from_candidates()`.
- Continue using the existing `seed_graphs_py` path into `_core.run_evolution()`.

### Reuse

- `glassbox/sr/cpp/seed_graph_builder.py`
- current candidate-to-seed conversion logic
- existing C++ seeded evolution path

### Rules

- Do not let composed seeds dominate the seed budget.
- Keep other diverse seeds in the population.
- Track whether the winning outcome came from:
  - screening only
  - composed seed + evolution
  - incumbent path

### Success Criteria

- Benchmarks that currently stall near the answer improve more often when composed seeds are present.

## Phase 4: Guided Residual Evolution From Merged Formulas

### Purpose

Use the existing residual stage more deliberately after composition.

### Build

- If a merged formula is close but still leaves structured residuals, run the residual symbolic fit from that merged base.
- Reuse `_stage_residual_symbolic_fit()` where possible rather than creating a second residual pipeline.

### Reuse

- `_stage_residual_symbolic_fit()`
- existing holdout-improvement checks
- existing final cleanup path

### Rules

- Only run this when the merged candidate is already validated as promising.
- Keep compute smaller than the main evolution budget.

### Success Criteria

- Merged near-miss formulas are converted into final winners more often than the current single-formula residual stage.

## Phase 5: Finer Specialist Resolution

### Purpose

Move from coarse segment complementarity to more targeted local correction behavior.

### Build

- richer segmentation strategies:
  - equal-width bins
  - curvature-aware bins
  - residual hot-spot bins
- sharper specialist detection for cases like:
  - one formula fits the global baseline
  - another formula handles a sharp point or oscillatory pocket

### Rules

- This phase should only start after earlier phases prove value.
- Validation gating must become stricter here because local overfitting risk rises quickly.

### Success Criteria

- The finer specialist layer improves difficult local-structure cases without causing benchmark-wide regressions.

## What To Be Careful About

### Search-space explosion

Risk:

- Pairwise formula composition can grow combinatorially.

Mitigation:

- small candidate pool
- small pair budget
- small composition template set

### Local overfitting

Risk:

- A candidate can look good only because it overfits one region or a noisy residual pattern.

Mitigation:

- validation-first acceptance
- domain-edge validation where appropriate
- risk and complexity thresholds

### Formula bloat

Risk:

- Merging formulas can quickly produce unreadable expressions.

Mitigation:

- always pass composed formulas through `_reduce_formula_noise()`
- always pass composed formulas through `_simplify_formula()`
- reject bloated candidates before seeding evolution

### Redundant compositions

Risk:

- Two formulas may look different symbolically but behave almost identically.

Mitigation:

- reject highly correlated prediction vectors
- deduplicate by family signature and behavior, not only by string form

### Piecewise benchmark cheating

Risk:

- Left-half/right-half stitching can create a local fit without discovering the true symbolic law.

Mitigation:

- do not emit piecewise final formulas in the first implementation
- use segment scoring to choose composition candidates, not to justify piecewise outputs

## Best Initial Build Order

Recommended implementation order:

1. Phase 0 instrumentation
2. Phase 1 specialist state
3. Phase 2 controlled `f + g` and `f * g` screening
4. Phase 3 composed-seed injection into evolution
5. Phase 4 merged-formula residual stage

This gives the highest learning value with the lowest risk and reuses the current architecture well.

## Reuse Summary

The main code paths to reuse are:

- candidate build/refine/prune in `glassbox/sr/sklearn_wrapper.py`
- seed generation in `glassbox/sr/cpp/seed_graph_builder.py`
- seeded C++ evolution through `seed_graphs_py`
- residual symbolic fit in `_stage_residual_symbolic_fit()`
- final noise reduction in `_reduce_formula_noise()`
- final simplification in `_simplify_formula()`

The specialist composition layer should sit on top of these pieces, not replace them.
