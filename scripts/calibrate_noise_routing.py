"""Evaluate classifier/proposer-style signals by noise bucket (Phase 7).

Does **not** retrain models. Reports reliability-style metrics on synthetic
curves for the same noise profiles the protocol uses (clean, gaussian low/med/high,
pink, quantization, outliers). Training on profiles the benchmark never reports
is intentionally avoided.

Usage:
  python scripts/calibrate_noise_routing.py
  python scripts/calibrate_noise_routing.py --out results/noise_routing_calibration.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from glassbox.sr.sklearn_wrapper import (  # noqa: E402
    GlassboxRegressor,
    _estimate_outlier_fraction,
    _noise_band_from_diagnostics,
    _residual_lag1_autocorr,
)
from scripts.benchmark_noise import NOISE_TIERS, apply_noise_tier  # noqa: E402


def _poly_targets(n: int, seed: int):
    rng = np.random.RandomState(seed)
    x = np.linspace(-2.0, 2.0, n)
    # Two protocol-like skeletons
    y1 = x ** 2
    y2 = x ** 3 - x
    return x, [("x^2", y1), ("x^3-x", y2)]


def _metrics_for_tier(tier: Dict[str, Any], *, n: int = 160, seeds: List[int] = None) -> Dict[str, Any]:
    seeds = seeds or [11, 23, 47]
    rows = []
    for seed in seeds:
        x, targets = _poly_targets(n, seed)
        for name, y_clean in targets:
            y_noisy = apply_noise_tier(y_clean, tier, seed=seed + hash(name) % 1000)
            # Residual vs true structure (oracle residual) — reliability of noise geometry
            resid = y_noisy - y_clean
            ac = abs(_residual_lag1_autocorr(resid))
            out_f = _estimate_outlier_fraction(resid)
            # Incumbent = true formula evaluated; measure validation gap via split
            est = GlassboxRegressor(random_state=seed)
            est.n_features_in_ = 1
            formula = "x0**2" if name == "x^2" else "x0**3 - x0"
            diag = est._compute_runtime_noise_diagnostics(
                x.reshape(-1, 1), y_noisy, formula=formula
            )
            # False confidence proxy: noisy train R2 high but clean R2 low for a constant model
            const = float(np.mean(y_noisy))
            noisy_r2 = 1.0 - float(np.mean((const - y_noisy) ** 2)) / max(float(np.var(y_noisy)), 1e-12)
            clean_r2 = 1.0 - float(np.mean((const - y_clean) ** 2)) / max(float(np.var(y_clean)), 1e-12)
            false_conf = bool(noisy_r2 > 0.5 and clean_r2 < 0.2)
            rows.append({
                "seed": seed,
                "formula": name,
                "residual_autocorr": ac,
                "outlier_fraction": out_f,
                "noise_band": diag.get("noise_band"),
                "validation_gap": diag.get("validation_gap"),
                "false_confidence_const": false_conf,
            })
    return {
        "tier": tier.get("name"),
        "noise_type": tier.get("noise_type"),
        "noise_level": tier.get("noise_level"),
        "n": len(rows),
        "median_outlier_fraction": float(np.median([r["outlier_fraction"] for r in rows])),
        "median_residual_autocorr": float(np.median([r["residual_autocorr"] for r in rows])),
        "noise_band_mode": max(
            set(r["noise_band"] for r in rows),
            key=lambda b: sum(1 for r in rows if r["noise_band"] == b),
        ),
        "false_confidence_rate_const": float(np.mean([1.0 if r["false_confidence_const"] else 0.0 for r in rows])),
        "rows": rows,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Noise-bucket routing calibration report")
    p.add_argument("--out", type=str, default="")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    # Only tiers the protocol reports (do not invent training-only profiles).
    report = {"tiers": [], "note": "Metrics use protocol NOISE_TIERS only; no retrain."}
    for tier in NOISE_TIERS:
        cell = _metrics_for_tier(tier)
        report["tiers"].append({k: v for k, v in cell.items() if k != "rows"})
        if not args.quiet:
            print(
                f"{cell['tier']:18s} band={cell['noise_band_mode']:6s} "
                f"out_frac={cell['median_outlier_fraction']:.3f} "
                f"ac={cell['median_residual_autocorr']:.3f} "
                f"fc_const={cell['false_confidence_rate_const']:.2f}"
            )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if not args.quiet:
            print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
