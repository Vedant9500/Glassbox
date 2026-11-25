"""Build the exact-vs-approximation pair dataset.

For each target formula the builder samples the curve, then generates the
approximations that are actually *possible* for that curve:

* Taylor truncations of the closed form,
* least-squares polynomial surrogates,
* Pade approximants,
* structurally-correct formulas with slightly wrong constants (positives).

Each pair is labelled ``exact`` or ``approximation`` by comparing the
candidate to the target in-domain and outside the fitted domain.

Usage:
    python experiments/exactness/build_dataset.py \
        --out data/exactness_pairs --n-generated 300 --seed 0
"""

from __future__ import annotations

import argparse
import json
import math
import random
import signal
import sys
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exactness import lib  # noqa: E402

CRITICAL_OPS = {"sin", "cos", "exp", "log", "rational"}

X_RANGES = (
    (-3.0, 3.0),
    (-5.0, 5.0),
    (-2.0, 2.0),
    (-6.0, 6.0),
    (0.1, 5.0),
    (0.1, 8.0),
    (-1.0, 1.0),
)


class _TargetTimeout(Exception):
    """Raised when a single target exceeds its wall-clock budget."""


@contextmanager
def _time_limit(seconds: float):
    """Bound one target's work; SymPy series can hang on nested formulas."""
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handler(signum, frame):  # noqa: ARG001
        raise _TargetTimeout()

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old)


def benchmark_targets() -> list[dict[str, Any]]:
    from scripts.benchmark_suite import ALL_TIERS

    out: list[dict[str, Any]] = []
    for tier, (name, formulas) in ALL_TIERS.items():
        for formula, human, x_range in formulas:
            out.append(
                {
                    "formula": formula,
                    "family": f"benchmark_tier{tier}",
                    "tier": tier,
                    "x_range": tuple(float(v) for v in x_range),
                    "source": "benchmark",
                    "name": human,
                }
            )
    return out


def generated_targets(n: int, seed: int) -> list[dict[str, Any]]:
    from glassbox.curve_classifier.generate_curve_data import PCFGFormulaGenerator

    random.seed(seed)
    np.random.seed(seed)
    generator = PCFGFormulaGenerator(max_depth=4)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    attempts = 0
    while len(out) < n and attempts < n * 20:
        attempts += 1
        formula, _ops = generator.generate()
        if "clip" in formula:
            continue
        key = lib.parse_formula(formula)
        if key is None:
            continue
        key_text = str(key)
        if key_text in seen:
            continue
        seen.add(key_text)
        out.append(
            {
                "formula": formula,
                "family": "pcfg",
                "tier": None,
                "x_range": X_RANGES[attempts % len(X_RANGES)],
                "source": "generated",
                "name": formula,
            }
        )
    return out


def candidate_pool(
    target: dict[str, Any],
    x: np.ndarray,
    y: np.ndarray,
    *,
    include_pade: bool,
    include_taylor: bool = True,
) -> list[tuple[str, str]]:
    formula = target["formula"]
    pool: list[tuple[str, str]] = [("identity", formula)]
    pool.extend(lib.scaled_candidates(formula))
    pool.extend(lib.perturb_candidates(formula))
    if include_taylor:
        pool.extend(lib.taylor_candidates(formula, max_order=8, min_order=3))
    pool.extend(lib.polyfit_candidates(x, y, degrees=(3, 4, 5, 6, 7, 8, 9)))
    if include_pade:
        pool.extend(lib.pade_candidates(formula))
    # Deduplicate by normalized text, keeping the first label.
    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for label, text in pool:
        key = str(text).replace(" ", "")
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append((label, text))
    return unique


def build(
    *,
    out_dir: Path,
    n_generated: int,
    seed: int,
    include_pade: bool,
    max_targets: int | None,
    tiers: list[int] | None = None,
    sources: list[str] | None = None,
) -> dict[str, Any]:
    targets = benchmark_targets() + generated_targets(n_generated, seed)
    if tiers:
        wanted = {int(t) for t in tiers}
        targets = [t for t in targets if t.get("tier") in wanted]
    if sources:
        wanted_sources = {str(s) for s in sources}
        targets = [t for t in targets if t["source"] in wanted_sources]
    if max_targets is not None:
        targets = targets[:max_targets]

    pairs: list[dict[str, Any]] = []
    curves: list[np.ndarray] = []
    curve_index: dict[str, int] = {}
    stats: Counter[str] = Counter()
    t0 = time.time()

    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "pairs.jsonl"
    # Write incrementally: a single pathological target must not discard the
    # whole run.
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for i, target in enumerate(targets, 1):
            formula = target["formula"]
            x_range = target["x_range"]
            try:
                with _time_limit(20.0):
                    sampled = lib.sample_curve(formula, x_range, n_samples=300)
                    if sampled is None:
                        stats["target_unusable"] += 1
                        continue
                    x, y = sampled
                    if float(np.var(y)) < 1e-12:
                        stats["target_constant"] += 1
                        continue

                    # ``_generate_data`` occasionally returns a shorter grid
                    # (points dropped during post-processing). Resample so all
                    # stored curves share one shape.
                    if x.size != 300:
                        finite = np.isfinite(y)
                        if int(finite.sum()) < 10:
                            stats["target_unusable"] += 1
                            continue
                        grid = np.linspace(
                            float(x_range[0]), float(x_range[1]), 300
                        )
                        y = np.interp(grid, x[finite], y[finite])
                        x = grid

                    curve_key = f"{formula}|{x_range[0]}|{x_range[1]}"
                    if curve_key not in curve_index:
                        curve_index[curve_key] = len(curves)
                        curves.append(np.stack([x, y]))
                    curve_id = curve_index[curve_key]

                    # Taylor expansions are only cheap for the benchmark's
                    # simple formulas; nested generated formulas can make
                    # SymPy's series() hang.
                    pool = candidate_pool(
                        target,
                        x,
                        y,
                        include_pade=include_pade,
                        include_taylor=target["source"] == "benchmark",
                    )
                    for label_name, candidate in pool:
                        metrics = lib.extrapolation_metrics(candidate, formula, x_range)
                        label = lib.label_pair(
                            in_mse=metrics["in_mse"],
                            ood_rel=metrics["ood_rel"],
                            candidate=candidate,
                            target=formula,
                            x_range=x_range,
                            candidate_kind=label_name,
                        )
                        if label == "other":
                            stats["pair_other"] += 1
                            continue
                        stats[f"pair_{label}"] += 1
                        if label_name.startswith(("taylor", "polyfit", "pade")):
                            stats[f"neg_{label_name.split('_')[0]}"] += 1

                        target_ops = lib.operator_set(formula)
                        cand_ops = lib.operator_set(candidate)
                        missing = sorted((CRITICAL_OPS & target_ops) - cand_ops)
                        fingerprint = lib.series_fingerprint(candidate)

                        row = {
                            "pair_id": len(pairs),
                            "curve_id": curve_id,
                            "target": formula,
                            "candidate": candidate,
                            "label": label,
                            "candidate_kind": label_name,
                            "in_mse": metrics["in_mse"],
                            "ood_mse": metrics["ood_mse"],
                            "ood_rel": metrics["ood_rel"],
                            "ood_skipped": metrics["ood_skipped"],
                            "target_ops": sorted(target_ops),
                            "candidate_ops": sorted(cand_ops),
                            "missing_critical_ops": missing,
                            "series_fingerprint": fingerprint.get("series"),
                            "series_score": fingerprint.get("score"),
                            "family": target["family"],
                            "tier": target["tier"],
                            "source": target["source"],
                            "x_range": list(x_range),
                            "name": target["name"],
                            "hard_negative": bool(
                                label == "approximation"
                                and metrics["in_mse"] is not None
                                and metrics["in_mse"] < 1e-8
                            ),
                        }
                        pairs.append(row)
                        fh.write(json.dumps(lib.json_safe(row)) + "\n")
            except _TargetTimeout:
                stats["target_timeout"] += 1
            except Exception:
                stats["target_error"] += 1

            if i % 25 == 0 or i == len(targets):
                print(
                    f"  [{i}/{len(targets)}] pairs={len(pairs)} "
                    f"exact={stats['pair_exact']} approx={stats['pair_approximation']} "
                    f"timeouts={stats['target_timeout']} ({time.time() - t0:.1f}s)",
                    flush=True,
                )

    np.savez_compressed(
        out_dir / "curves.npz",
        curves=np.asarray(curves, dtype=np.float32),
    )

    summary = {
        "targets": len(targets),
        "curves": len(curves),
        "pairs": len(pairs),
        "counts": dict(stats),
        "label_counts": dict(Counter(p["label"] for p in pairs)),
        "kind_counts": dict(Counter(p["candidate_kind"] for p in pairs)),
        "family_counts": dict(Counter(p["family"] for p in pairs)),
        "source_counts": dict(Counter(p["source"] for p in pairs)),
        "hard_negatives": sum(1 for p in pairs if p["hard_negative"]),
        "seed": seed,
        "n_generated": n_generated,
        "include_pade": include_pade,
        "runtime_seconds": round(time.time() - t0, 2),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(lib.json_safe(summary), indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/exactness_pairs"),
        help="Output directory for pairs.jsonl, curves.npz, summary.json",
    )
    parser.add_argument(
        "--n-generated",
        type=int,
        default=300,
        help="Number of PCFG-generated training formulas (default: 300)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--include-pade",
        action="store_true",
        help="Also generate Pade approximants (slower)",
    )
    parser.add_argument(
        "--max-targets",
        type=int,
        default=None,
        help="Debug: cap the number of target formulas",
    )
    parser.add_argument(
        "--tiers",
        type=int,
        nargs="+",
        default=None,
        help="Only benchmark tiers to include (e.g. --tiers 3 4 5)",
    )
    parser.add_argument(
        "--sources",
        type=str,
        nargs="+",
        choices=["benchmark", "generated"],
        default=None,
        help="Only include these target sources",
    )
    args = parser.parse_args()

    summary = build(
        out_dir=args.out,
        n_generated=args.n_generated,
        seed=args.seed,
        include_pade=args.include_pade,
        max_targets=args.max_targets,
        tiers=args.tiers,
        sources=args.sources,
    )
    print("\nDataset summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
