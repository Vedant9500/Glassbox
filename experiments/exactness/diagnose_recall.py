"""Measure candidate-pool recall for exact structures.

For every benchmark result this answers one question: if we had a perfect
exactness verifier, could we improve the result by *reranking* the candidates
the pipeline already produced? If the correct structure is not in the pool,
reranking cannot help and the fix has to be a better search (evolution or
classifier priors).

Usage:
    python experiments/exactness/diagnose_recall.py \
        results/exactness_exp/baseline_full/benchmark_latest.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exactness import lib  # noqa: E402

CRITICAL_OPS = {"sin", "cos", "exp", "log", "rational"}
OOD_GOOD = 1e-3
# A candidate counts as a reranking win only if it would score EXACT on the
# benchmark's own in-domain contract.
EXACT_MSE = 1e-6
# A reranked candidate is allowed to be this much worse in-domain than the
# pipeline winner; beyond that the substitution is not credible.
IN_MSE_SLACK = 10.0


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def score_candidate(candidate: str, target: str, x_range: tuple[float, float]) -> dict[str, Any]:
    metrics = lib.extrapolation_metrics(candidate, target, x_range)
    target_ops = lib.operator_set(target)
    cand_ops = lib.operator_set(candidate)
    missing = sorted((CRITICAL_OPS & target_ops) - cand_ops)
    fingerprint = lib.series_fingerprint(candidate)
    ood_rel = metrics["ood_rel"]
    structural = not missing and fingerprint.get("series") is None
    ood_good = (
        ood_rel is not None
        and math.isfinite(float(ood_rel))
        and float(ood_rel) <= OOD_GOOD
    )
    would_score_exact = bool(
        structural
        and metrics["in_mse"] is not None
        and float(metrics["in_mse"]) < EXACT_MSE
    )
    return {
        "formula": candidate,
        "in_mse": metrics["in_mse"],
        "ood_rel": ood_rel,
        "missing_critical_ops": missing,
        "series_fingerprint": fingerprint.get("series"),
        "structural": structural,
        "ood_good": ood_good,
        "would_score_exact": would_score_exact,
    }


def diagnose(report: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for tier_payload in (report.get("tiers") or {}).values():
        for result in tier_payload.get("results") or []:
            target = str(result.get("formula_target") or "")
            x_range_raw = result.get("x_range") or [0.0, 0.0]
            x_range = (float(x_range_raw[0]), float(x_range_raw[1]))
            winner = str(result.get("formula_discovered") or "")
            pool = [c for c in (result.get("candidate_formulas") or []) if isinstance(c, dict)]
            winner_mse = _finite(result.get("mse"))

            scored = []
            for cand in pool:
                formula = str(cand.get("formula") or "").strip()
                if not formula or formula.replace(" ", "") == winner.replace(" ", ""):
                    continue
                scored.append(score_candidate(formula, target, x_range))

            exact = [c for c in scored if c["would_score_exact"]]
            structural_any = [c for c in scored if c["structural"]]
            best = None
            if exact:
                best = min(
                    exact,
                    key=lambda c: (
                        c["in_mse"] if c["in_mse"] is not None else float("inf")
                    ),
                )
            best_structural = None
            if structural_any:
                best_structural = min(
                    structural_any,
                    key=lambda c: (
                        c["in_mse"] if c["in_mse"] is not None else float("inf")
                    ),
                )
            rerankable = bool(
                best
                and best["in_mse"] is not None
                and (
                    winner_mse is None
                    or best["in_mse"] <= max(winner_mse, 1e-12) * IN_MSE_SLACK
                )
            )
            rows.append(
                {
                    "target": target,
                    "winner": winner,
                    "winner_score": result.get("score"),
                    "winner_mse": winner_mse,
                    "pool_size": len(pool),
                    "n_exact_in_pool": len(exact),
                    "best_exact_candidate": best,
                    "n_structural_in_pool": len(structural_any),
                    "best_structural_candidate": best_structural,
                    "rerankable": rerankable,
                }
            )
    return _summarise(rows)


def _summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    by_score = Counter(r["winner_score"] for r in rows)
    non_exact = [r for r in rows if r["winner_score"] != "EXACT"]
    rerankable = [r for r in non_exact if r["rerankable"]]
    structure_present = [r for r in non_exact if r["n_structural_in_pool"] > 0]
    exact_present = [r for r in non_exact if r["n_exact_in_pool"] > 0]
    return {
        "total": total,
        "by_score": dict(by_score),
        "non_exact": len(non_exact),
        "non_exact_with_exact_candidate": len(exact_present),
        "non_exact_with_correct_structure": len(structure_present),
        "non_exact_rerankable": len(rerankable),
        "rerankable_targets": [r["target"] for r in rerankable],
        "structure_present_targets": [r["target"] for r in structure_present],
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    summary = diagnose(report)

    print("\n" + "=" * 88)
    print("CANDIDATE-POOL RECALL")
    print("=" * 88)
    print(f"  formulas: {summary['total']}   non-exact winners: {summary['non_exact']}")
    print(
        f"  non-exact with correct structure in the pool: "
        f"{summary['non_exact_with_correct_structure']}"
    )
    print(
        f"  non-exact with a would-be EXACT candidate: "
        f"{summary['non_exact_with_exact_candidate']}"
    )
    print(f"  rerankable (exact candidate within {IN_MSE_SLACK:g}x winner MSE): "
          f"{summary['non_exact_rerankable']}")
    print("=" * 88)
    if summary["rerankable_targets"]:
        print("\nRerankable targets:")
        for target in summary["rerankable_targets"]:
            print(f"  - {target}")

    out = args.out or args.report.with_name(f"{args.report.stem}.recall.json")
    out.write_text(json.dumps(lib.json_safe(summary), indent=2), encoding="utf-8")
    print(f"\nRecall JSON: {out}")


if __name__ == "__main__":
    main()
