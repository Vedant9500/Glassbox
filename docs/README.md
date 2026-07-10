# Glassbox Docs Index

This folder contains maintained source documentation for the current codebase.
Generated benchmark reports live under `results/` and should not be treated as
canonical architecture docs.

## Current Docs

- `PROJECT_MAP.md`: detailed repository map, pipeline trace, source/module
  ownership, script index, and test coverage map.
- `ONN_Architecture.md`: current hybrid symbolic regression architecture, with
  ONN details documented as the legacy/research model path.
- `onn_runbook.md`: release gates, smoke tests, rollback triggers, and failure
  playbooks for fast-path, proposer, specialist, benchmark, and C++ changes.
- `Research_Roadmap.md`: current roadmap and status for benchmark discipline,
  blackbox/SRBench work, specialist composition, C++ migration, proposer usage,
  and remaining research risks.
- `Universal_FastPath_AB_Report.md`: qualitative A/B rollout report for hybrid
  fast path vs proposer-only/legacy modes.
- `CPP_Migration_Roadmap.md`: current C++ backend migration status and remaining
  native-backend opportunities.

## Related Root Docs

- `README.md`: quick start and runtime overview.
- `specialist_composition_plan.md` and `specialist_composition_audit.md`:
  specialist composition design/history and audit notes.
- `blackbox_optimization_plan.md`: blackbox/SRBench feature-reduction and
  interaction-discovery plan.
- `cuda_x_integration_plan.md`: GPU/CUDA-X planning notes.
- `Implementation Plan.md`: native simplification migration notes.

## Research Notes

Long-form theoretical analysis and comparative critiques live in
`research_notes/` at the repository root. Those files intentionally include
ideas that are not implemented in the current runtime. Use `PROJECT_MAP.md` for
the authoritative current implementation map.
