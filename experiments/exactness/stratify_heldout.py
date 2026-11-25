"""Break a held-out evaluation down by structural family.

A guard can only help the families it can express. Aggregate scores hide that:
operator closure targets products/quotients, series promotion targets
truncated polynomials, and neither can build a nested composition.

Usage:
    python experiments/exactness/stratify_heldout.py \
        results/exactness_exp/heldout/heldout_*.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exactness import lib  # noqa: E402

FUNCS = {"sin", "cos", "tan", "exp", "log", "sqrt", "tanh", "sinh", "cosh", "atan"}


def classify(formula: str) -> str:
    """Structural family, shared with the evaluator."""
    return lib.classify_structure(formula)


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    points = {"EXACT": 3, "APPROX": 2, "LOOSE": 1, "FAIL": 0}
    total = len(records)
    weighted = sum(points.get(r.get("score"), 0) for r in records)
    return {
        "n": total,
        "exact": sum(1 for r in records if r.get("score") == "EXACT"),
        "struct_exact": sum(1 for r in records if r.get("struct_exact")),
        "false_exact": sum(1 for r in records if r.get("false_exact")),
        "weighted_pct": round(100.0 * weighted / (3 * total), 1) if total else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    payloads: dict[str, dict[str, Any]] = {}
    for path in args.reports:
        data = json.loads(path.read_text(encoding="utf-8"))
        key = f"{data['config']['guard']}/{data['config']['noise']}"
        payloads[key] = data

    families: list[str] = sorted(
        {
            classify(r["target"])
            for data in payloads.values()
            for r in data["records"]
        }
    )

    table: dict[str, dict[str, dict[str, Any]]] = {}
    for key, data in payloads.items():
        by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in data["records"]:
            by_family[classify(record["target"])].append(record)
        table[key] = {fam: summarise(by_family[fam]) for fam in families if fam in by_family}

    # Print a compact per-family comparison.
    keys = list(payloads)
    header = f"{'family':<16}" + "".join(f"{k:>22}" for k in keys)
    print(header)
    print("-" * len(header))
    for fam in families:
        row = f"{fam:<16}"
        for key in keys:
            cell = table[key].get(fam)
            if not cell:
                row += f"{'-':>22}"
            else:
                row += f"{cell['exact']}/{cell['n']} ex {cell['weighted_pct']:>5.0f}%".rjust(22)
        print(row)

    print()
    for key in keys:
        overall = payloads[key]["summary"]
        print(
            f"{key:<20} exact={overall['exact']}/{overall['total']} "
            f"struct={overall['struct_exact']} false={overall['false_exact']} "
            f"weighted={overall['weighted_pct']}% errors={overall['errors']}"
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(lib.json_safe({"families": families, "table": table}), indent=2),
            encoding="utf-8",
        )
        print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
