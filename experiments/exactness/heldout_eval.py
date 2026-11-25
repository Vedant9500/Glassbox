"""Held-out evaluation on generated formulas.

The benchmark is a fixed, public formula bank. Tuning a basis against it is
teaching to the test, so the primary metric for these experiments is a fresh
PCFG-generated formula set that the pipeline has never seen, optionally with
noise.

Reports, for baseline vs a guard:

* benchmark-style score distribution (EXACT / APPROX / LOOSE / FAIL),
* OOD-relative error (does the discovered formula extrapolate?),
* structural match (operator set + no series fingerprint),
* ``struct_exact`` — EXACT *and* extrapolates *and* structurally right.

Usage:
    python experiments/exactness/heldout_eval.py --n 30 --guard closure
    python experiments/exactness/heldout_eval.py --n 30 --guard closure --noise medium
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exactness import lib  # noqa: E402

CRITICAL_OPS = {"sin", "cos", "exp", "log", "rational"}
X_RANGES = ((-3.0, 3.0), (-5.0, 5.0), (-2.0, 2.0), (0.1, 5.0), (0.1, 8.0))


def to_benchmark_dialect(formula: str) -> str:
    """Convert generator syntax to the benchmark's parser dialect."""
    return str(formula).replace("np.", "").strip()


def _benchmark_keys() -> set[str]:
    from scripts.benchmark_suite import ALL_TIERS

    keys = set()
    for _name, formulas in ALL_TIERS.values():
        for formula, _human, _xr in formulas:
            expr = lib.parse_formula(formula)
            if expr is not None:
                keys.add(str(expr))
    return keys


def generate_formulas(
    n: int, seed: int, *, balance: bool = True
) -> list[dict[str, Any]]:
    """Generate fresh PCFG formulas that are usable and non-trivial.

    With ``balance`` the pool is stratified by structural family so every
    mechanism has a measurable slice: operator closure can only help
    ``product``/``quotient`` targets, series promotion only helps
    transcendental ones, and nothing here builds a nested composition.
    """
    from glassbox.curve_classifier.generate_curve_data import PCFGFormulaGenerator

    random.seed(seed)
    np.random.seed(seed)
    generator = PCFGFormulaGenerator(max_depth=4)
    excluded = _benchmark_keys()

    wanted = ("product", "quotient", "nested", "transcendental")
    per_family = max(1, math.ceil(n / len(wanted))) if balance else n
    buckets: dict[str, list[dict[str, Any]]] = {name: [] for name in wanted}
    seen: set[str] = set()
    attempts = 0
    limit = n * 250

    def family_full() -> bool:
        return all(len(buckets[name]) >= per_family for name in wanted)

    while attempts < limit and not family_full():
        attempts += 1
        formula, _ops = generator.generate()
        if "clip" in formula:
            continue
        text = to_benchmark_dialect(formula)
        expr = lib.parse_formula(text)
        if expr is None:
            continue
        key = str(expr)
        if key in excluded or key in seen:
            continue
        family = lib.classify_structure(text)
        if family not in buckets:
            continue
        if len(buckets[family]) >= per_family:
            continue
        ops = lib.operator_set(text)
        if not (CRITICAL_OPS & ops):
            continue
        if family == "product" and not (ops & {"sin", "cos", "exp", "log"}):
            # ``x * (x/2)`` is a polynomial, not a product of primitives.
            continue
        x_range = X_RANGES[attempts % len(X_RANGES)]
        sampled = lib.sample_curve(text, x_range, n_samples=300)
        if sampled is None:
            continue
        x, y = sampled
        if float(np.var(y)) < 1e-9:
            continue
        if not np.all(np.isfinite(y)):
            continue
        seen.add(key)
        buckets[family].append(
            {
                "formula": text,
                "x_range": x_range,
                "ops": sorted(ops),
                "family": family,
                "y_var": float(np.var(y)),
                "attempt": attempts,
            }
        )

    out: list[dict[str, Any]] = []
    for family in wanted:
        out.extend(buckets[family])
    return out[:n]


def score_run(result: dict[str, Any], target: str, x_range: tuple[float, float]) -> dict[str, Any]:
    discovered = str(result.get("formula_discovered") or "")
    record = {
        "target": target,
        "x_range": list(x_range),
        "family": lib.classify_structure(target),
        "score": result.get("score"),
        "mse": result.get("mse"),
        "formula": discovered,
        "error": result.get("error"),
    }
    if not discovered or discovered in {"N/A", "ERROR", "?"}:
        record.update(
            {"ood_rel": None, "missing_critical_ops": sorted(CRITICAL_OPS & lib.operator_set(target)),
             "series_fingerprint": None, "struct_exact": False, "false_exact": False}
        )
        return record

    metrics = lib.extrapolation_metrics(discovered, target, x_range)
    target_ops = lib.operator_set(target)
    cand_ops = lib.operator_set(discovered)
    missing = sorted((CRITICAL_OPS & target_ops) - cand_ops)
    fingerprint = lib.series_fingerprint(discovered)

    # A trig identity is the same function; a missing operator is not an error.
    equivalent = False
    if missing or fingerprint.get("series") is not None:
        equivalent = lib.equivalent(discovered, target, x_range)
        if equivalent:
            missing = []
            fingerprint = {}
    ood_rel = metrics["ood_rel"]
    ood_good = (
        ood_rel is not None
        and math.isfinite(float(ood_rel))
        and float(ood_rel) <= 1e-3
    )
    struct_ok = not missing and fingerprint.get("series") is None
    record.update(
        {
            "ood_rel": ood_rel,
            "missing_critical_ops": missing,
            "equivalent_rewrite": equivalent,
            "series_fingerprint": fingerprint.get("series"),
            "struct_ok": struct_ok,
            "struct_exact": bool(result.get("score") == "EXACT" and ood_good and struct_ok),
            "false_exact": bool(
                result.get("score") == "EXACT"
                and (
                    (ood_rel is not None and math.isfinite(float(ood_rel)) and float(ood_rel) > 1e-2)
                    or missing
                    or fingerprint.get("series") is not None
                )
            ),
        }
    )
    return record


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    counts = Counter(r["score"] for r in records)
    points = {"EXACT": 3, "APPROX": 2, "LOOSE": 1, "FAIL": 0}
    weighted = sum(points.get(r["score"], 0) for r in records)
    ood_values = [
        float(r["ood_rel"])
        for r in records
        if r.get("ood_rel") is not None and math.isfinite(float(r["ood_rel"]))
    ]

    per_family: dict[str, dict[str, Any]] = {}
    for family in sorted({r.get("family") or "unknown" for r in records}):
        subset = [r for r in records if (r.get("family") or "unknown") == family]
        sub_weighted = sum(points.get(r["score"], 0) for r in subset)
        per_family[family] = {
            "n": len(subset),
            "exact": sum(1 for r in subset if r["score"] == "EXACT"),
            "struct_exact": sum(1 for r in subset if r.get("struct_exact")),
            "weighted_pct": round(100.0 * sub_weighted / (3 * len(subset)), 1)
            if subset
            else 0.0,
        }

    return {
        "total": total,
        "counts": dict(counts),
        "exact": counts.get("EXACT", 0),
        "exact_rate": round(counts.get("EXACT", 0) / total, 4) if total else 0.0,
        "weighted": weighted,
        "weighted_pct": round(100.0 * weighted / (3 * total), 1) if total else 0.0,
        "struct_exact": sum(1 for r in records if r.get("struct_exact")),
        "false_exact": sum(1 for r in records if r.get("false_exact")),
        "ood_median": float(np.median(ood_values)) if ood_values else None,
        "ood_good": sum(
            1 for v in ood_values if v <= 1e-3
        ),
        "errors": sum(1 for r in records if r.get("error")),
        "per_family": per_family,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--guard",
        choices=["none", "closure", "compose", "pi", "series", "all"],
        default="none",
    )
    parser.add_argument("--noise", choices=["none", "low", "medium", "high"], default="none")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--out", type=Path, default=Path("results/exactness_exp/heldout")
    )
    args = parser.parse_args()

    formulas = generate_formulas(args.n, args.seed)
    print(f"held-out formulas: {len(formulas)} (seed={args.seed})")

    import scripts.benchmark_suite as bs

    noise_cfg = None
    if args.noise != "none":
        noise_cfg = bs.NOISE_PRESETS[args.noise]

    if args.guard in {"closure", "compose", "all", "series"}:
        from experiments.exactness.run_with_guard import (
            apply_composite_pass,
            apply_composition_pass,
            apply_series_pass,
        )

        original_run_fast_path = bs.run_fast_path
        use_closure = args.guard in {"closure", "compose", "all"}
        use_composition = args.guard in {"compose", "all"}
        use_series = args.guard in {"series", "all"}

        def guarded_run_fast_path(x, y, *fargs, **fkwargs):
            result = original_run_fast_path(x, y, *fargs, **fkwargs)
            if use_closure:
                result = apply_composite_pass(result, x, y, fkwargs)
            if use_composition:
                result = apply_composition_pass(result, x, y, fkwargs)
            if use_series:
                result = apply_series_pass(result, x, y)
            return result

        bs.run_fast_path = guarded_run_fast_path

    if args.guard in {"pi", "all"}:
        from experiments.exactness.run_with_guard import snap_pi

        original_postprocess = bs._postprocess_formula_for_benchmark

        def guarded_postprocess(formula, x, y):
            snapped = snap_pi(formula)
            if snapped != formula:
                return original_postprocess(snapped, x, y)
            return original_postprocess(formula, x, y)

        bs._postprocess_formula_for_benchmark = guarded_postprocess

    records: list[dict[str, Any]] = []
    for i, item in enumerate(formulas, 1):
        result = bs.run_formula(
            item["formula"],
            item["x_range"],
            classifier_path=bs.DEFAULT_CURVE_CLASSIFIER_PATH,
            n_samples=300,
            timeout=args.timeout,
            with_evolution=False,
            proposer_path="models/universal_proposer_multi.pt",
            noise_cfg=noise_cfg,
            noise_seed=args.seed,
        )
        records.append(score_run(result, item["formula"], item["x_range"]))
        if i % 5 == 0 or i == len(formulas):
            s = summarise(records)
            print(
                f"  [{i}/{len(formulas)}] exact={s['exact']} struct_exact={s['struct_exact']} "
                f"false_exact={s['false_exact']} weighted={s['weighted_pct']}%",
                flush=True,
            )

    summary = summarise(records)
    payload = {
        "config": {
            "n": len(formulas),
            "seed": args.seed,
            "guard": args.guard,
            "noise": args.noise,
            "timeout": args.timeout,
        },
        "summary": summary,
        "records": records,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.guard}_{args.noise}_seed{args.seed}"
    path = args.out / f"heldout_{tag}.json"
    path.write_text(json.dumps(lib.json_safe(payload), indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"HELD-OUT RESULT  guard={args.guard} noise={args.noise}")
    print("=" * 72)
    print(f"  formulas:      {summary['total']}")
    print(f"  exact:         {summary['exact']} ({summary['exact_rate'] * 100:.0f}%)")
    print(f"  struct_exact:  {summary['struct_exact']}")
    print(f"  false_exact:   {summary['false_exact']}")
    print(f"  weighted:      {summary['weighted_pct']}%")
    print(f"  ood median:    {summary['ood_median']}")
    print(f"  errors:        {summary['errors']}")
    print("\n  per family:")
    for family, cell in summary["per_family"].items():
        print(
            f"    {family:<16} exact={cell['exact']}/{cell['n']} "
            f"struct={cell['struct_exact']} weighted={cell['weighted_pct']}%"
        )
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
