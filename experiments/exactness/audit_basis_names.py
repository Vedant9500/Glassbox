"""Audit basis column names against the values they claim to describe.

A basis column is fitted as numbers but reported as text. If the text and the
numbers disagree, the pipeline can fit the curve perfectly and still print a
different function. That is not hypothetical: the damped sinusoid columns were
built as ``e^(-a*|x|)*sin(w*x)`` but named ``e^(-a*x)*sin(w*x)``, so every
display evaluation silently described a different curve.

This script builds the universal basis on a grid and evaluates each name with
the same parser the benchmark uses. It classifies mismatches:

* ``domain``   - the name is undefined where the column was regularised
                 (``sqrt(1-x^2)`` past |x|=1, ``log(x+1)`` past x=-1);
* ``floor``    - the column applies a small floor near a singularity
                 (``1/|x|``, ``sqrt(|x|)``);
* ``precision``- the name prints a constant with fewer digits than the column;
* ``structural`` - none of the above: the name is a different function.

Only ``structural`` is a bug. Run from the repository root:

    python experiments/exactness/audit_basis_names.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exactness import lib  # noqa: E402


def _classify(name: str, deviation: float, x: np.ndarray, column: np.ndarray) -> str:
    if "sqrt(1-" in name or "1/(1-" in name:
        return "domain"
    if name in {"1/|x|", "sqrt(|x|)"} or name.startswith("log(x"):
        return "floor"
    if any(token in name for token in ("3.14159", "6.28319", "1.5708")):
        return "precision"
    return "structural"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lo", type=float, default=-4.0)
    parser.add_argument("--hi", type=float, default=4.0)
    parser.add_argument("--n", type=int, default=401)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    from scripts.classifier_fast_path import build_basis_from_predictions

    x = np.linspace(args.lo, args.hi, args.n).reshape(-1, 1)
    predictions = {
        key: 1.0
        for key in (
            "sin",
            "cos",
            "exp",
            "log",
            "power",
            "polynomial",
            "rational",
            "periodic",
            "damped",
        )
    }
    basis, names = build_basis_from_predictions(
        x,
        predictions,
        threshold=0.0,
        max_power=6,
        detected_omegas=[1.0, 2.0],
        universal_basis=True,
    )

    counts: dict[str, int] = {}
    structural: list[tuple[float, str]] = []
    for index, name in enumerate(names):
        column = np.asarray(basis[:, index], dtype=np.float64)
        if not np.all(np.isfinite(column)):
            continue
        values = lib.evaluate(name, x.ravel())
        if values is None or values.shape != column.shape:
            structural.append((float("inf"), name))
            continue
        finite = np.isfinite(values) & np.isfinite(column)
        if int(finite.sum()) < 20:
            continue
        scale = max(1.0, float(np.max(np.abs(column[finite]))))
        deviation = float(np.max(np.abs(values[finite] - column[finite]))) / scale
        if deviation <= args.tolerance:
            continue
        kind = _classify(name, deviation, x.ravel(), column)
        counts[kind] = counts.get(kind, 0) + 1
        if kind == "structural":
            structural.append((deviation, name))

    print(f"columns: {len(names)}")
    for kind in sorted(counts):
        print(f"  {kind:<11} {counts[kind]}")
    print(f"  clean       {len(names) - sum(counts.values())}")
    if structural:
        print("\nstructural mismatches (bugs):")
        for deviation, name in sorted(structural, reverse=True):
            print(f"  {deviation:.2e}  {name}")
    else:
        print("\nno structural mismatches")


if __name__ == "__main__":
    main()
