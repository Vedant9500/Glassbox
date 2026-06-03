# Specialist Composition Layer Plan

## Current Status (2026-06-03)

The specialist layer described here is now substantially implemented. Current
code includes specialist state, pair scoring, safe composition proposals,
composed seed caps, hot-spot segments, residual symbolic stages, vault memory,
and inception/subexpression reuse. Main files are
`glassbox/sr/specialist_state.py`, `glassbox/sr/sklearn_wrapper.py`, and
`scripts/specialist_phase_eval.py`. Use this file for design history and
`docs/PROJECT_MAP.md` for the current implementation map.

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

## Current Failure Analysis (Benchmark Tiers 6–8)

From the latest benchmark results, there are 54 non-exact cases across Tiers 6–8. They cluster into 7 distinct failure categories:

### Category 1: Envelope × Oscillation (damped oscillations)

Examples: `exp(-x²)·sin(3x)`, `exp(-x)·sin(x)²`, `x²·exp(-x)·cos(3x)`

Evolution finds the envelope OR the oscillation, but not both multiplied. The `f+g` composition cannot represent `f·g` when both are non-trivial. The `f*g` template exists but complementarity scoring does not detect "one is an envelope, the other is oscillatory" because segment-level MSE does not distinguish structural roles.

### Category 2: Nested Transcendentals

Examples: `sin(cos(x))`, `cos(sin(x))`, `sin(x·cos(x))`, `sin(x + sin(x))`

The inner function is itself a transcendental. The composition layer only tries `f+g` and `f*g`, never `f(g(x))` (nested application). The seed graph builder can represent `sin(exp(x))` but the composition proposal step never generates this form.

### Category 3: Log-Compound (log of non-trivial arguments)

Examples: `log(2x+1)`, `log(1+exp(x))`, `log(1+sin(x)²)`, `log(1+sin(x))`

Same root cause as Category 2: no nested composition templates.

### Category 4: Rational Functions with Transcendental Numerators

Examples: `sin(πx)/(πx)`, `1/√(1+x²)`, `x/(exp(x)-1)`, `exp(-|x|)`

Division or absolute value combined with transcendentals. The `f/g` template is not in the composition set. Absolute value handling is limited.

### Category 5: Multi-Frequency Products

Examples: `sin(x)·sin(3x)·sin(5x)`, `x·sin(x)·cos(x)`

Products of 3+ oscillatory terms. Pairwise composition (2 formulas) cannot express 3-way products. No iterative composition exists.

### Category 6: Transition/Sigmoid Functions

Examples: `1/(1+exp(-x))`, `x/(1+|x|)`

Bounded, smooth transitions. No sigmoid-gated composition template exists; `f+g` and `f*g` do not naturally produce saturating functions.

### Category 7: Slowly Converging Approximations

Examples: `sqrt(1+x²)`, `exp(-|x|)·cos(2x)`, `(x²-1)·exp(-x²/2)`

Evolution finds APPROX solutions that are structurally close but use wrong envelope or modulation constants. No "refinement from APPROX" exists — the residual stage fires only once and does not iterate.

## Literature Survey: Relevant Techniques

### InceptionSR (Library Learning via Frozen Subexpressions)

Reference: Bartlett et al., "InceptionSR" (2025)

Run SR iteratively. After each round, identify high-performing subexpressions, "freeze" them as new features, and feed them into the next round. Our `_stage_residual_symbolic_fit()` already does one round of this. InceptionSR generalizes to N rounds with frozen building blocks.

Integration point: After Phase 4's residual fit, freeze the base formula as a feature column and re-run a lightweight evolution on the augmented feature space.

### SyRBo (Symbolic Regression Boosting)

Reference: Sipper & Moore, "Symbolic-Regression Boosting" (2021, arXiv:2012.09278)

Apply gradient boosting with SR as the weak learner. Each stage fits the pseudo-residual of the previous stages. Typically 2–5 stages are sufficient. Our single residual stage is literally Stage 1 of SyRBo. Adding 1–2 more stages would complete the boosting loop.

Integration point: Wrap `_stage_residual_symbolic_fit()` in a loop with validation-gated acceptance.

### Semantic Crossover / Geometric Semantic GP (GSGP)

Reference: Moraglio et al. (2012); Uy et al. (2011)

Instead of random subtree swap, compose parents using their semantic (output) vectors. GSGP offspring lie on the semantic segment between parents. Our complementarity scoring already uses prediction vectors and residual correlation, which is a form of semantic distance. We can extend it to generate semantically-interpolated offspring.

Integration point: In `propose_specialist_compositions()`, add a weighted-average composition template: `α·f + (1-α)·g` with α fitted by least squares.

### Cooperative Coevolution / Specialist Islands

Reference: Potter & De Jong (2000); Poli et al. (2008)

Decompose the problem into subpopulations that evolve subcomponents cooperatively. Each island specializes on a data region or functional component. The C++ engine already uses an island model with 8 islands. Currently all islands evolve complete solutions on the same data. We could assign different islands to specialize on different data segments.

Integration point: Pass segment masks to the C++ engine so some islands evolve on subsets of the data, then compose the best per-segment solutions in Python.

### PS-Tree (Piecewise Symbolic Regression Tree)

Reference: Song et al. (2022)

Build a decision tree that partitions the feature space, then fit a symbolic regressor to each leaf node. This is the formal version of the left-half specialist plus right-half specialist idea. The risk is piecewise benchmark cheating, but with smooth gating it becomes legitimate.

Integration point: After the specialist layer identifies strong segment specialists, fit a smooth sigmoid gate between them.

### Quality-Diversity SR (QDSR)

Reference: Bhatia et al. (2025, arXiv)

Maintain a diverse archive of expressions organized by behavioral descriptors such as formula family, complexity, and residual pattern. Our `SpecialistState.candidates` list is a rudimentary archive. QDSR would formalize the behavioral descriptor space and actively promote diversity.

Integration point: Extend `SpecialistState` to maintain a MAP-Elites style archive across evolution generations.

## Codebase Limitations Identified

Only 2 composition templates (`f+g`, `f*g`) exist in `glassbox/sr/specialist_state.py`. This misses nested, ratio, and gated compositions.

Fixed 4-segment equal-width bins in `build_specialist_segment_slices()` cannot detect fine-grained hot spots or curvature-aware regions.

Single residual stage with no iteration in `_stage_residual_symbolic_fit()` gives one shot at residual correction with no boosting loop.

No cross-run memory exists. Multi-start runs do not share specialist discoveries. Good local fits from run 1 are lost in run 2.

Complementarity scoring is MSE-only and does not detect structural roles such as envelope vs oscillation, only raw error magnitude.

Composition candidates are not role-aware. `f+g` is proposed even when `f(g)` or `f/g` would be structurally appropriate.

No frozen subexpression reuse exists. Strong subexpressions discovered in one context cannot be reused as building blocks in another.

## Phase 5: Residual Hot-Spot Segmentation

### Purpose

Replace the fixed 4-segment equal-width bins with adaptive, residual-driven segmentation that identifies where the current best formula is struggling.

### Build

- After computing the best formula's residual vector, identify hot-spot segments as contiguous regions where the cumulative squared residual exceeds a threshold, such as 70% of total squared error concentrated in 30% or less of the domain.
- Implement curvature-aware binning: compute the second derivative via finite differences of the residual and split at inflection points.
- Add a `hot_spot_segments` field to `SpecialistState` alongside the existing equal-width segments.
- When scoring specialist pairs, also score them against hot-spot segments. A formula that excels on a hot-spot region is more valuable than one that excels on an already-solved region.

### Reuse

- `build_specialist_segment_slices()` — extend, not replace
- existing `SpecialistSegment` and `SpecialistSegmentScore` dataclasses
- residual vector computation already in `_stage_residual_symbolic_fit()`

### Rules

- Keep equal-width segments as a fallback; hot-spot segmentation is additive.
- Cap hot-spot segments at 6 to prevent over-fragmentation.
- Require each hot-spot segment to contain at least `min_segment_size` samples.

### Success Criteria

- Hot-spot-driven composition improves at least 2 of the current Tier 7/8 APPROX cases by correctly identifying the region where the base formula fails and composing a local correction.

## Phase 6: Expanded Composition Templates

### Purpose

Move beyond `f+g` and `f*g` to cover the structural patterns observed in benchmark failures.

### New composition templates

Division: `f / g` with `g` bounded away from 0. Targets Category 4 (rational functions).

Nested application: `f(g(x))` where `f` is a unary function (sin, cos, exp, log). Targets Categories 2 and 3 (nested transcendentals).

Affine blend: `α·f + (1-α)·g` with α fitted by OLS. Targets Category 7 (APPROX refinement). This follows the Geometric Semantic GP idea.

Damped product: `f · exp(-β·g²)` with β fitted. Targets Category 1 (envelope × oscillation).

Sigmoid gate: `f·σ(k·(x-c)) + g·(1-σ(k·(x-c)))` where σ is a logistic sigmoid. Targets Category 6 (transitions). Inspired by PS-Tree with smooth gating.

### Safety constraints for each template

- Division: only emit if `min(|g(x)|) > 0.01` across the training domain.
- Nested: only try when one candidate's family signature is unary (pure sin, cos, exp, log) and the other is non-constant with bounded range.
- Affine blend: α must be in (0.05, 0.95) to avoid collapsing to one parent.
- Damped product: β must be positive; reject if the product is near-zero over more than 50% of the domain.
- Sigmoid gate: transition point `c` and sharpness `k` are fitted by scipy minimize or grid search. Only emit if validation R² exceeds both parents.

### Reuse

- `_refine_formula_constants()` for constant optimization of α, β, k, c
- `_score_formula_candidate()` for validation gating
- `_safe_eval_formula_array()` for numerical evaluation
- existing `SpecialistCompositionProposal` dataclass, extended with template type

### Rules

- Hard cap: at most 3 templates per pair, at most 12 total composition candidates.
- Reject any template that increases complexity by more than 2× over the simpler parent.
- Validate every composed formula on the holdout set before accepting.

### Success Criteria

- At least 3 of the following currently-failing cases move from APPROX/FAIL to PASS: `sin(cos(x))`, `1/(1+exp(-x))`, `exp(-x²)·sin(3x)`, `log(1+exp(x))`.

## Phase 7: Iterative Residual Boosting (SyRBo-style)

### Purpose

Generalize the single residual stage to a multi-stage symbolic boosting loop, inspired by SyRBo.

### Build

- Wrap `_stage_residual_symbolic_fit()` in a loop of up to `max_boosting_stages` (default: 3) iterations.
- Each stage:
  1. Compute residual: `r_k = y - ŷ_k` where `ŷ_k` is the cumulative prediction.
  2. Fit a symbolic expression `h_k` to `r_k` using a lightweight sub-regressor.
  3. Update: `ŷ_{k+1} = ŷ_k + η·h_k` with learning rate `η` (default 0.8, tuned on holdout).
  4. Validation gate: accept `h_k` only if holdout R² of `ŷ_{k+1}` improves over `ŷ_k`.
  5. Early stopping: stop if holdout R² improvement is less than 0.005 or if `ŷ_{k+1}` R² exceeds 0.999.
- The final formula is: `base + η₁·h₁ + η₂·h₂ + ...`
- Store each stage's formula in `self.boosting_stages_` for inspection.

### Reuse

- `_stage_residual_symbolic_fit()` — reuse as the inner stage solver
- `_refine_candidate_formulas()` for constant optimization of the combined formula
- `_reduce_formula_noise()` and `_simplify_formula()` for final cleanup

### Rules

- Maximum 3 boosting stages to control compute. Each stage is roughly 50% of the main evolution budget.
- Each stage gets a progressively smaller timeout: `timeout // 2`, `timeout // 4`, `timeout // 8`.
- Total boosting time must not exceed the main evolution timeout.
- If the base formula is already exact-level (R² > 0.9999), skip boosting entirely.

### Success Criteria

- Category 7 (slowly converging approximations) improves: at least 3 APPROX cases move to closer-to-PASS R² values.
- No regression on Tiers 1–5. These are already PASS and should not be affected.

## Phase 8: Cross-Run Specialist Memory

### Purpose

Allow specialist discoveries from one multi-start run to be preserved and reused in subsequent runs, preventing loss of valuable local fits.

### Build

- Add a `SpecialistVault` object to `GlassboxRegressor` that stores vault entries containing: formula, source, validation R², validation MSE, segment scores, residual vector, and the run index where it was discovered.
- After each multi-start run completes, extract the top-3 candidates that are structurally different from the current best and store them in the vault.
- Before each subsequent run:
  1. Compose vault entries with each other and with the current best using Phase 6 templates.
  2. Add the best compositions to the seed pool for the next run.
  3. Re-score vault entries against the latest residual to check if any have become more relevant.
- Deduplication: remove vault entries whose prediction vectors have correlation above 0.98 with existing entries.

### Reuse

- `build_seed_graphs_from_candidates()` for seeding vault entries into evolution
- `SpecialistState` for scoring vault entries
- `_refine_candidate_formulas()` for validating vault-derived compositions
- existing multi-start loop in `sklearn_wrapper.py`

### Rules

- Vault size capped at 8 entries to prevent memory bloat.
- Vault entries older than 3 runs without improvement are evicted.
- Vault compositions get the same 35% seed budget cap as Phase 3 composed seeds.
- Vault is cleared at the start of each `fit()` call. No leakage across datasets.

### Success Criteria

- Multi-start runs with vault memory achieve higher final R² than without, measured on at least 5 of the current multi-start-dependent Tier 7/8 cases.
- The vault does not slow down individual runs by more than 5%.

## Phase 9: Frozen Subexpression Reuse (InceptionSR-style)

### Purpose

Identify strong subexpressions discovered during evolution and freeze them as new feature columns for subsequent search rounds, enabling hierarchical composition.

### Build

- After evolution produces a winning formula, parse its AST and extract subexpressions that:
  1. Have complexity 3 or more (not trivial).
  2. Appear as arguments to outer operators, meaning they are inner building blocks.
  3. When evaluated, have non-trivial variance (not near-constant).
- Add these frozen subexpressions as new columns in X, creating an augmented feature matrix.
- Re-run a lightweight SR on the augmented feature space, targeting the same y.
- The discovery in the augmented space may find simpler representations that compose the frozen blocks.

### Reuse

- `_parse_formula_expr()` from `glassbox/sr/cpp/seed_graph_builder.py` for AST parsing
- `BlackboxState` for feature augmentation (the blackbox preprocessor already handles variable-width X)
- `_build_blackbox_candidate_formulas()` for re-running candidate screening on augmented X
- `_safe_eval_formula_array()` for evaluating frozen subexpressions

### Rules

- Maximum 3 frozen subexpressions per round.
- Maximum 2 inception rounds to prevent exponential blowup.
- Each inception round gets at most 50% of the original timeout.
- Frozen subexpressions must have validation R² above 0.3 as standalone predictors. They must carry useful signal.
- If the first inception round does not improve R² by at least 0.01, skip the second round.

### Success Criteria

- At least 2 of the Category 2/3 failures (nested transcendentals, log-compounds) improve from APPROX to PASS.
- The inception loop discovers compositions like `sin(sin(x))` or `log(1+sin(x)²)` that the flat search space cannot represent.

## What To Be Careful About

### Search-space explosion

Risk:

- Pairwise formula composition can grow combinatorially.
- Phase 6 adds 5 new templates, multiplying the candidate space.

Mitigation:

- small candidate pool
- small pair budget
- small composition template set
- hard cap at 12 total composition candidates across all templates
- at most 3 templates per pair

### Local overfitting

Risk:

- A candidate can look good only because it overfits one region or a noisy residual pattern.
- Hot-spot segmentation (Phase 5) can create tiny segments that invite overfitting.

Mitigation:

- validation-first acceptance
- domain-edge validation where appropriate
- risk and complexity thresholds
- minimum segment size enforced for hot-spot segments

### Formula bloat

Risk:

- Merging formulas can quickly produce unreadable expressions.
- Multi-stage boosting (Phase 7) produces additive chains that grow per stage.

Mitigation:

- always pass composed formulas through `_reduce_formula_noise()`
- always pass composed formulas through `_simplify_formula()`
- reject bloated candidates before seeding evolution
- cap boosting stages at 3

### Redundant compositions

Risk:

- Two formulas may look different symbolically but behave almost identically.

Mitigation:

- reject highly correlated prediction vectors
- deduplicate by family signature and behavior, not only by string form
- vault deduplication at correlation > 0.98

### Piecewise benchmark cheating

Risk:

- Left-half/right-half stitching can create a local fit without discovering the true symbolic law.
- Sigmoid gate template (Phase 6) is inherently piecewise-like.

Mitigation:

- do not emit piecewise final formulas in the first implementation
- use segment scoring to choose composition candidates, not to justify piecewise outputs
- sigmoid gate only accepted if validation R² exceeds both parents

### Compute budget explosion

Risk:

- Iterative boosting (Phase 7) and inception rounds (Phase 9) can multiply total runtime.
- Cross-run memory (Phase 8) adds composition overhead before each multi-start run.

Mitigation:

- progressively smaller timeouts per boosting stage
- total boosting time capped at main evolution timeout
- inception rounds capped at 50% of original timeout each
- vault composition overhead bounded by vault size cap of 8

### Stale cross-run memory

Risk:

- Vault entries from early runs may not be relevant to later search states.

Mitigation:

- re-score vault entries against the latest residual before each run
- evict entries older than 3 runs without improvement
- clear vault at the start of each `fit()` call

## Best Initial Build Order

Recommended implementation order:

1. Phase 0 instrumentation (shipped)
2. Phase 1 specialist state (shipped)
3. Phase 2 controlled `f + g` and `f * g` screening (shipped)
4. Phase 3 composed-seed injection into evolution (shipped)
5. Phase 4 merged-formula residual stage (shipped)
6. Phase 5 residual hot-spot segmentation
7. Phase 6 expanded composition templates
8. Phase 7 iterative residual boosting
9. Phase 8 cross-run specialist memory
10. Phase 9 frozen subexpression reuse

Phases 5 and 6 should be built together. Hot-spot segmentation makes the expanded templates much more targeted. Phase 7 is nearly free since it wraps the existing `_stage_residual_symbolic_fit()` in a loop. Phase 9 is the most complex but has the highest potential payoff for hard nested compositions.

## Reuse Summary

The main code paths to reuse are:

- candidate build/refine/prune in `glassbox/sr/sklearn_wrapper.py`
- seed generation in `glassbox/sr/cpp/seed_graph_builder.py`
- seeded C++ evolution through `seed_graphs_py`
- residual symbolic fit in `_stage_residual_symbolic_fit()`
- final noise reduction in `_reduce_formula_noise()`
- final simplification in `_simplify_formula()`
- formula evaluation in `_safe_eval_formula_array()`
- constant refinement in `_refine_formula_constants()`
- AST parsing in `_parse_formula_expr()` from seed graph builder
- blackbox feature augmentation in `BlackboxState`
- specialist state and segment scoring in `glassbox/sr/specialist_state.py`

The specialist composition layer should sit on top of these pieces, not replace them.
