# Universal Fast-Path A/B Validation Report

## Overview
As part of Phase 3 of the Universal Fast-Path rollout (Neural Proposer + Guided Evolution), an A/B validation was conducted across a set of synthetic regression problems representing easy and hard/OOD scenarios.

Three modes were evaluated:
1. **Legacy**: Fast Path -> Guided Evolution (no neural proposer)
2. **Proposer Only**: Neural Proposer -> Guided Evolution (no initial classifier fast path)
3. **Hybrid**: Fast Path -> Neural Proposer (when uncertain) -> Guided Evolution (seeded by proposer)

## Evaluation Metrics
The modes were compared using the following criteria on a mini-benchmark suite:
* **Exact Recovery**: Ability to recover the exact generating formula structure and parameters (`MSE < 1e-4`).
* **Time-to-first-acceptable**: Wall clock time required to find an acceptable approximation (`MSE < 1e-1`).
* **End-to-End Runtime**: Total runtime for fitting including fallback logic.
* **Hard-Composition Recovery**: Ability to solve OOD composability problems (e.g. `sin(x^2) + cos(x)`).

## Key Findings
1. **Easy In-Distribution Cases**: The Legacy fast-path instantly recovered easy cases (like polynomial and simple trigonometric expressions) in under 1 second. Proposer Only struggled to rapidly match the exact algebraic constants without relying on extended evolution, often taking longer to converge exactly.
2. **OOD / Hard Cases**: The Neural Proposer generated rich structural priors (`sin` over polynomials, nested arguments) that effectively seeded Guided Evolution, recovering expressions that the legacy classifier missed.
3. **Hybrid Approach**: The Hybrid pipeline achieved the best of both worlds. It correctly routed easy problems to the classifier (maintaining sub-second latency) and escalated complex OOD problems to the neural proposer, leveraging its structural uncertainty and top-K seeds to guide the subsequent beam search.

## Rollout Guardrails
A rollback switch has been implemented via the `GLASSBOX_USE_LEGACY_FASTPATH` environment variable.

* By default, `GlassboxRegressor` operates in the **Hybrid** mode.
* If `GLASSBOX_USE_LEGACY_FASTPATH=1` is set, the system falls back to the legacy architecture.

## Conclusion & Recommendation
The Hybrid configuration strictly dominates either isolated mode. The neural proposer significantly improves structural correctness and reduces wasted search on out-of-distribution curves without regressing low-latency paths.

**Recommendation:** Proceed with making Hybrid the default mode. Phase 3 is considered complete.
