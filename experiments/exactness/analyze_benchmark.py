"""Analyse a benchmark report for false EXACT results.

A benchmark result is called a *false EXACT* when the displayed formula fits
the sampled domain to the EXACT threshold but fails outside it. Those are the
cases where the pipeline accepted a truncated series or a polynomial surrogate
as the closed form.

Usage:
    python experiments/exactness/analyze_benchmark.py \
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

# OOD relative MSE thresholds (normalised by target OOD variance).
OOD_CLEAN = 1e-3
OOD_BAD = 1e-2


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def analyse_result(result: dict[str, Any], *, factor: float = 2.0) -> dict[str, Any]:
    target = str(result.get("formula_target") or "")
    discovered = str(result.get("formula_discovered") or "")
    x_range_raw = result.get("x_range") or [0.0, 0.0]
    x_range = (float(x_range_raw[0]), float(x_range_raw[1]))

    record: dict[str, Any] = {
        "target": target,
        "discovered": discovered,
        "x_range": list(x_range),
        "bench_score": result.get("score"),
        "bench_mse": _finite(result.get("mse")),
        "human_name": result.get("human_name"),
    }
    if not discovered or discovered in {"N/A", "ERROR", "?"}:
        record.update(
            {
                "in_mse": None,
                "ood_mse": None,
                "ood_rel": None,
                "ood_skipped": True,
                "target_ops": sorted(lib.operator_set(target)),
                "candidate_ops": [],
                "missing_critical_ops": sorted(
                    CRITICAL_OPS & lib.operator_set(target)
                ),
                "extra_ops": [],
                "series_fingerprint": None,
                "structure_ok": False,
                "ood_ok": False,
                "false_exact": False,
            }
        )
        return record

    metrics = lib.extrapolation_metrics(discovered, target, x_range, factor=factor)
    target_ops = lib.operator_set(target)
    cand_ops = lib.operator_set(discovered)
    missing = sorted((CRITICAL_OPS & target_ops) - cand_ops)
    extra = sorted(cand_ops - target_ops)
    fingerprint = lib.series_fingerprint(discovered)

    # An operator-set mismatch is only evidence of an approximation when the
    # formulas are actually different functions. ``2*sin(x)*cos(x)`` and
    # ``sin(2*x)`` are identical, so the missing ``cos`` means nothing.
    equivalent = False
    if missing or fingerprint.get("series") is not None:
        equivalent = lib.equivalent(discovered, target, x_range)
        if equivalent:
            missing = []
            fingerprint = {}

    ood_rel = metrics["ood_rel"]
    ood_ok = (
        ood_rel is not None
        and math.isfinite(float(ood_rel))
        and float(ood_rel) <= OOD_CLEAN
    )
    structure_ok = not missing and fingerprint.get("series") is None
    false_exact = bool(
        record["bench_score"] == "EXACT"
        and (
            (ood_rel is not None and math.isfinite(float(ood_rel)) and float(ood_rel) > OOD_BAD)
            or missing
            or fingerprint.get("series") is not None
        )
    )

    record.update(
        {
            "in_mse": metrics["in_mse"],
            "ood_mse": metrics["ood_mse"],
            "ood_rel": ood_rel,
            "ood_skipped": metrics["ood_skipped"],
            "ood_points": metrics["ood_points"],
            "target_ops": sorted(target_ops),
            "candidate_ops": sorted(cand_ops),
            "missing_critical_ops": missing,
            "extra_ops": extra,
            "equivalent_rewrite": equivalent,
            "series_fingerprint": fingerprint.get("series"),
            "series_score": fingerprint.get("score"),
            "series_scale": fingerprint.get("scale"),
            "structure_ok": structure_ok,
            "ood_ok": ood_ok,
            "false_exact": false_exact,
        }
    )
    return record


def analyse_report(report: dict[str, Any], *, factor: float = 2.0) -> dict[str, Any]:
    tiers = report.get("tiers") or {}
    records: list[dict[str, Any]] = []
    per_tier: dict[str, dict[str, Any]] = {}
    for tier_key, payload in tiers.items():
        tier_records = [
            analyse_result(r, factor=factor)
            for r in (payload.get("results") or [])
        ]
        records.extend(tier_records)
        per_tier[tier_key] = _summarise(tier_records, payload.get("name"))
    return {
        "report": report.get("timestamp"),
        "classifier": report.get("classifier"),
        "overall": _summarise(records, "ALL"),
        "tiers": per_tier,
        "records": records,
    }


def _summarise(records: list[dict[str, Any]], name: str | None) -> dict[str, Any]:
    total = len(records)
    exact = sum(1 for r in records if r["bench_score"] == "EXACT")
    false_exact = sum(1 for r in records if r["false_exact"])
    ood_ok = sum(1 for r in records if r["ood_ok"])
    structure_ok = sum(1 for r in records if r["structure_ok"])
    missing = Counter(
        op for r in records for op in (r.get("missing_critical_ops") or [])
    )
    fingerprints = Counter(
        r["series_fingerprint"] for r in records if r.get("series_fingerprint")
    )
    return {
        "name": name,
        "total": total,
        "bench_exact": exact,
        "false_exact": false_exact,
        "false_exact_rate": round(false_exact / total, 4) if total else 0.0,
        "ood_ok": ood_ok,
        "structure_ok": structure_ok,
        "missing_critical_ops": dict(missing),
        "series_fingerprints": dict(fingerprints),
    }


def print_summary(analysis: dict[str, Any]) -> None:
    overall = analysis["overall"]
    print("\n" + "=" * 88)
    print("EXACTNESS ANALYSIS")
    print("=" * 88)
    print(
        f"  formulas: {overall['total']}   benchmark EXACT: {overall['bench_exact']}   "
        f"false EXACT: {overall['false_exact']} ({overall['false_exact_rate'] * 100:.1f}%)"
    )
    print(
        f"  extrapolation OK: {overall['ood_ok']}   structure OK: {overall['structure_ok']}"
    )
    if overall["series_fingerprints"]:
        print(f"  series fingerprints: {overall['series_fingerprints']}")
    if overall["missing_critical_ops"]:
        print(f"  missing operators:   {overall['missing_critical_ops']}")
    print("-" * 88)
    print(f"  {'tier':<6} {'total':>6} {'exact':>6} {'false':>6} {'ood_ok':>7} {'struct':>7}")
    for tier_key in sorted(analysis["tiers"], key=lambda k: int(k)):
        row = analysis["tiers"][tier_key]
        print(
            f"  {tier_key:<6} {row['total']:>6} {row['bench_exact']:>6} "
            f"{row['false_exact']:>6} {row['ood_ok']:>7} {row['structure_ok']:>7}"
        )
    print("=" * 88)

    bad = [r for r in analysis["records"] if r["false_exact"]]
    if bad:
        print("\nFalse EXACT cases:")
        for r in bad:
            ood = r["ood_rel"]
            ood_s = f"{ood:.2e}" if ood is not None and math.isfinite(ood) else "inf"
            fp = f" series={r['series_fingerprint']}" if r["series_fingerprint"] else ""
            miss = (
                f" missing={','.join(r['missing_critical_ops'])}"
                if r["missing_critical_ops"]
                else ""
            )
            print(
                f"  - {r['target']:<28} -> {r['discovered'][:48]:<48} "
                f"ood_rel={ood_s}{fp}{miss}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Benchmark JSON report")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write the full analysis JSON here (default: next to the report)",
    )
    parser.add_argument(
        "--factor",
        type=float,
        default=2.0,
        help="OOD extension as a multiple of the fitted domain span (default: 2)",
    )
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    analysis = analyse_report(report, factor=args.factor)
    print_summary(analysis)

    out = args.out
    if out is None:
        out = args.report.with_name(f"{args.report.stem}.exactness.json")
    out.write_text(
        json.dumps(lib.json_safe(analysis), indent=2), encoding="utf-8"
    )
    print(f"\nAnalysis JSON: {out}")


if __name__ == "__main__":
    main()
