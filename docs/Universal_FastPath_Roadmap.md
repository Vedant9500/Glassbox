# Universal Fast-Path Roadmap (Neural Proposer + Guided Evolution)

## Goal
Build a universal fast path that does not depend on a fixed hand-crafted basis, while still keeping sub-second wins on easy cases. When uncertain, the fast path must emit rich guidance for evolution to recover correct formulas.

## Product Decision
Use a hybrid system instead of replacing the current fast path outright:

1. Keep current lightweight fast path for very easy in-distribution wins.
2. Add a universal neural proposer (Transformer/Set model) to generate structural hypotheses.
3. Route uncertain cases to guided evolution with richer FPIP payload.

## Why This Direction
The current classifier + basis regression is very fast, but tied to basis coverage. A universal proposer improves out-of-basis recovery and gives evolution stronger seeds than operator-only hints.

## Target Architecture

### Stage A (Existing Fast Path)
- Keep current lightweight pipeline in place for low-latency exact/simple cases.
- Keep current uncertainty/residual diagnostics and budget routing.

### Stage B (New Universal Proposer)
- Model: Set Transformer encoder (preferred) or small encoder-decoder Transformer.
- Input: sampled point set (x, y) and optional robust diagnostics (residual spectrum, symmetry flags, separability hints).
- Output heads:
  - Top-K formula skeleton sequences (grammar-constrained tokens).
  - Sequence confidence and entropy.
  - Operator prior distribution (calibrated).
  - Optional variable interaction map for multivariate coupling.

### Stage C (Evolution Consumption)
- Seed population from top-K skeletons (PIGP-style initialization).
- Blend priors by uncertainty (high certainty -> stronger priors; high uncertainty -> more uniform).
- Escalate compute budget only when confidence is low or residual risk is high.

## FPIP v2 Contract (what proposer must emit)
- `candidate_skeletons`: list of top-K expressions with score/probability.
- `sequence_uncertainty`: entropy, margin, calibrated confidence.
- `operator_priors`: normalized priors for evolution.
- `interaction_hints`: separability/symmetry/coupling flags.
- `fit_diagnostics`: residual spectrum and holdout/generalization signals if numeric fit exists.
- `routing_signal`: recommend fast-accept vs guided evolution.

## Implementation Plan

### Status
- Phase 0: COMPLETE
- Phase 1: IN PROGRESS
- Phase 2: NOT STARTED
- Phase 3: NOT STARTED

### Phase 0: Baseline and Interface Freeze (3-5 days) [COMPLETE]
- Freeze current baseline metrics (exact recovery, structural correctness, runtime, fail taxonomy).
- Define `FPIPv2` schema and add validation tests.
- Add feature flag to run old/new fast path side-by-side.

Deliverables:
- Baseline report artifact.
- `FPIPv2` typed schema and unit tests.

### Phase 1: Universal Proposer MVP (1-2 weeks) [IN PROGRESS]
- Build training pipeline for universal proposer.
- Add grammar-constrained decoder for valid skeleton generation.
- Emit top-K candidates + uncertainty + priors.
- Keep latency target under ~400ms inference for proposer.

Progress update:
- Completed: Phase 1 scaffold created (`glassbox/sr/universal_proposer.py`, `scripts/train_universal_proposer.py`) with unit tests.
- Completed: proposer -> `FPIPv2` adapter implemented and validated in tests.
- Next: replace fixed skeleton vocabulary decoder with richer grammar-constrained decoding and dataset-backed training.

Deliverables:
- New training script and model artifact format.
- Inference API that returns `FPIPv2` payload.
- Benchmarks vs current classifier on OOD slices.

### Phase 2: Evolution Integration (1-2 weeks) [NOT STARTED]
- Inject top-K skeletons into evolution seeds.
- Wire proposer uncertainty into prior blending and budget routing.
- Add fallback logic: if proposer fails, keep legacy path and existing evolution behavior.

Deliverables:
- End-to-end pipeline with dual-path routing.
- Regression tests for routing and seed injection.

### Phase 3: Validation and A/B Rollout (1 week) [NOT STARTED]
- Compare three modes:
  - Legacy fast path only
  - Universal proposer only
  - Hybrid (recommended)
- Evaluate metrics:
  - Exact recovery
  - Structural correctness
  - Time-to-first-acceptable
  - End-to-end runtime
  - Hard-composition recovery

Deliverables:
- A/B report and recommended default config.
- Rollout guardrails + rollback switch.

## Branch Plan
Create a dedicated branch for this work:
- Suggested branch name: `feature/universal-fastpath-proposer`

Suggested initial milestones on branch:
1. `milestone/01-fpipv2-schema`
2. `milestone/02-proposer-mvp`
3. `milestone/03-evolution-seeding`
4. `milestone/04-ab-validation`

## Acceptance Criteria
- Hybrid mode improves OOD exact recovery without regressing easy-case latency materially.
- Evolution receives meaningful top-K structural seeds and uncertainty metadata.
- Pipeline remains robust: proposer failure does not break legacy behavior.
- Benchmarks show improved structural correctness and reduced wasted evolution runs.

## Risks and Mitigations
- Risk: Proposer latency too high.
  - Mitigation: smaller model, quantization, batched inference, cached preprocessing.
- Risk: Bad seeds bias evolution incorrectly.
  - Mitigation: uncertainty-weighted blending toward uniform priors; keep random explorers.
- Risk: Dataset bias from synthetic generation.
  - Mitigation: broaden grammar, curriculum, and OOD validation sets.

## Immediate Next Tasks
1. Implement richer grammar-constrained skeleton decoding (top-K output).
2. Add dataset-backed proposer training path (beyond synthetic scaffold).
3. Add dual-path feature flag and logging for side-by-side proposer vs legacy runs.
4. Run first proposer-only benchmark slice on feynman_easy + OOD subset.
