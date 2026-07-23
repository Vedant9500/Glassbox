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
- `CPP_Migration_Roadmap.md`: current C++ backend migration status and remaining
  native-backend opportunities.
- `SEARCH_OPTIMIZATION_PLAN.md`: search-path status plus remaining research
  items (e.g. online semantic diversity, broader separability probes).
- `curve_classifier_universal_proposer_audit.md`: classifier/proposer audit and
  phase status (implementation largely done; production corpus work remaining).

## Related Root Docs

- `README.md`: quick start and runtime overview.
- `glassbox_codebase_audit_tracker.md`: package audit findings (many open P2s).
- `glassbox_implementation_plan.md`: ordered fix plan from the audit (phases
  0–7; residual polish still tracked).

## Research Notes

Long-form theoretical analysis and comparative critiques live in
`research_notes/` at the repository root. Those files intentionally include
ideas that are not implemented in the current runtime. Use `PROJECT_MAP.md` for
the authoritative current implementation map.
