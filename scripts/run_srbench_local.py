"""
Glassbox SRBench Benchmark
===========================

Full SRBench-style benchmark for Glassbox, running across:
  1. Black-box regression datasets from PMLB (Track 1)
  2. Ground-truth symbolic regression datasets (Track 2)

Usage:
  python scripts/run_srbench_local.py                       # Full suite
  python scripts/run_srbench_local.py --track 1             # Black-box only
  python scripts/run_srbench_local.py --track 2             # Ground-truth only
  python scripts/run_srbench_local.py --max-datasets 10     # Quick smoke test
  python scripts/run_srbench_local.py --pop-size 200 --gens 2000  # Higher budget
"""

import sys
import argparse
import time
import json
import math  # noqa: F401
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np

# Project root
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from glassbox.sr.sklearn_wrapper import GlassboxRegressor

# ---------------------------------------------------------------------------
# Ground-truth formulas (Track 2) — Feynman/Nguyen/Strogatz style
# Each: (name, formula_fn, n_features, x_ranges, formula_str)
# ---------------------------------------------------------------------------

GROUND_TRUTH_PROBLEMS = [
    # --- Nguyen Suite ---
    ("Nguyen-1", lambda X: X[:, 0]**3 + X[:, 0]**2 + X[:, 0],
     1, [(-1, 1)], "x0**3 + x0**2 + x0"),
    ("Nguyen-2", lambda X: X[:, 0]**4 + X[:, 0]**3 + X[:, 0]**2 + X[:, 0],
     1, [(-1, 1)], "x0**4 + x0**3 + x0**2 + x0"),
    ("Nguyen-3", lambda X: X[:, 0]**5 + X[:, 0]**4 + X[:, 0]**3 + X[:, 0]**2 + X[:, 0],
     1, [(-1, 1)], "x0**5 + x0**4 + x0**3 + x0**2 + x0"),
    ("Nguyen-4", lambda X: X[:, 0]**6 + X[:, 0]**5 + X[:, 0]**4 + X[:, 0]**3 + X[:, 0]**2 + X[:, 0],
     1, [(-1, 1)], "x0**6 + x0**5 + x0**4 + x0**3 + x0**2 + x0"),
    ("Nguyen-5", lambda X: np.sin(X[:, 0]**2) * np.cos(X[:, 0]) - 1,
     1, [(-1, 1)], "sin(x0**2)*cos(x0) - 1"),
    ("Nguyen-6", lambda X: np.sin(X[:, 0]) + np.sin(X[:, 0] + X[:, 0]**2),
     1, [(-1, 1)], "sin(x0) + sin(x0 + x0**2)"),
    ("Nguyen-7", lambda X: np.log(X[:, 0] + 1) + np.log(X[:, 0]**2 + 1),
     1, [(0.01, 2)], "log(x0 + 1) + log(x0**2 + 1)"),
    ("Nguyen-8", lambda X: np.sqrt(X[:, 0]),
     1, [(0.01, 4)], "sqrt(x0)"),
    ("Nguyen-9", lambda X: np.sin(X[:, 0]) + np.sin(X[:, 0]**2),
     1, [(-3, 3)], "sin(x0) + sin(x0**2)"),
    ("Nguyen-10", lambda X: 2 * np.sin(X[:, 0]) * np.cos(X[:, 0]),
     1, [(-3, 3)], "2*sin(x0)*cos(x0)"),

    # --- Keijzer Suite ---
    ("Keijzer-4", lambda X: 0.3 * X[:, 0] * np.sin(2 * np.pi * X[:, 0]),
     1, [(-3, 3)], "0.3*x0*sin(2*pi*x0)"),

    # --- Feynman Easy ---
    ("Feynman-I.6.20a", lambda X: np.exp(-X[:, 0]**2 / 2) / np.sqrt(2 * np.pi),
     1, [(-3, 3)], "exp(-x0**2/2) / sqrt(2*pi)"),
    ("Feynman-I.8.14", lambda X: np.sqrt((X[:, 0] - X[:, 1])**2 + (X[:, 2] - X[:, 3])**2),
     4, [(-5, 5)] * 4, "sqrt((x0-x1)**2 + (x2-x3)**2)"),
    ("Feynman-I.9.18", lambda X: X[:, 0] * X[:, 1] / (4 * np.pi * (X[:, 2]**2)),
     3, [(0.1, 5)] * 3, "x0*x1 / (4*pi*x2**2)"),
    ("Feynman-I.10.7", lambda X: X[:, 0] / np.sqrt(1 - (X[:, 1] / X[:, 2])**2),
     3, [(0.1, 5), (0.1, 2), (3, 10)], "x0 / sqrt(1 - (x1/x2)**2)"),

    # --- Simple classics ---
    ("Pagie-1", lambda X: 1.0 / (1.0 + X[:, 0]**(-4)) + 1.0 / (1.0 + X[:, 1]**(-4)),
     2, [(0.1, 5)] * 2, "1/(1+x0**-4) + 1/(1+x1**-4)"),
    ("Korns-11", lambda X: 6.87 + 11 * np.cos(7.23 * X[:, 0]**3),
     1, [(-3, 3)], "6.87 + 11*cos(7.23*x0**3)"),
    ("Vladislavleva-4", lambda X: 10.0 / (5.0 + np.sum((X[:, :5] - 3)**2, axis=1)),
     5, [(0.05, 6.05)] * 5, "10 / (5 + sum((xi-3)**2))"),

    # --- Polynomial ---
    ("Poly-x2", lambda X: X[:, 0]**2,
     1, [(-5, 5)], "x0**2"),
    ("Poly-x3-x", lambda X: X[:, 0]**3 - X[:, 0],
     1, [(-3, 3)], "x0**3 - x0"),
    ("Poly-chebyshev-T4", lambda X: 8 * X[:, 0]**4 - 8 * X[:, 0]**2 + 1,
     1, [(-1, 1)], "8*x0**4 - 8*x0**2 + 1"),

    # --- Trig ---
    ("Trig-sin", lambda X: np.sin(X[:, 0]),
     1, [(-6, 6)], "sin(x0)"),
    ("Trig-sin+cos", lambda X: np.sin(X[:, 0]) + np.cos(X[:, 0]),
     1, [(-6, 6)], "sin(x0) + cos(x0)"),
    ("Trig-damped-sine", lambda X: np.exp(-X[:, 0]) * np.sin(X[:, 0]),
     1, [(0, 10)], "exp(-x0)*sin(x0)"),

    # --- Rational ---
    ("Rational-sigmoid", lambda X: 1.0 / (1.0 + np.exp(-X[:, 0])),
     1, [(-6, 6)], "1/(1+exp(-x0))"),
    ("Rational-lorentz", lambda X: 1.0 / (1.0 + X[:, 0]**2),
     1, [(-5, 5)], "1/(1+x0**2)"),
]

# ---------------------------------------------------------------------------
# PMLB Black-Box Datasets (Track 1) — curated regression subset
# These are the most commonly used PMLB regression datasets in SRBench
# ---------------------------------------------------------------------------

PMLB_DATASETS = [
    "1027_ESL",
    "1028_SWD",
    "1029_LEV",
    "1030_ERA",
    "192_vineyard",
    "195_auto_price",
    "201_pol",
    "210_cloud",
    "215_2dplanes",
    "218_house_8L",
    "225_puma8NH",
    "228_elusage",
    "230_machine_cpu",
    "344_mv",
    "503_wind",
    "505_tecator",
    "519_vinnie",
    "522_pm10",
    "523_analcatdata_neavote",
    "527_pm10",
    "529_pollen",
    "537_houses",
    "542_pollution",
    "547_no2",
    "556_analcatdata_apnea2",
    "557_analcatdata_apnea1",
    "560_bodyfat",
    "561_cpu",
    "564_fried",
    "574_house_16H",
    "579_fri_c0_250_5",
    "581_fri_c3_500_25",
    "582_fri_c1_500_25",
    "584_fri_c4_500_25",
    "586_fri_c3_1000_25",
    "588_fri_c4_1000_100",
    "589_fri_c2_1000_25",
    "590_fri_c0_1000_25",
    "591_fri_c1_1000_25",
    "592_fri_c4_1000_25",
    "593_fri_c1_1000_10",
    "594_fri_c2_1000_5",
    "595_fri_c0_1000_10",
    "596_fri_c2_1000_10",
    "597_fri_c2_1000_50",
    "598_fri_c0_1000_50",
    "599_fri_c2_1000_25",
    "601_fri_c1_1000_5",
    "603_fri_c0_1000_50",
    "604_fri_c4_500_10",
    "607_fri_c4_1000_50",
    "608_fri_c3_1000_10",
    "611_fri_c3_1000_50",
    "612_fri_c1_1000_25",
    "613_fri_c3_250_5",
    "615_fri_c4_250_10",
    "616_fri_c4_500_25",
    "617_fri_c3_500_5",
    "618_fri_c3_1000_5",
    "620_fri_c1_1000_50",
    "621_fri_c0_100_10",
    "623_fri_c4_1000_10",
    "624_fri_c0_100_5",
    "626_fri_c2_500_25",
    "627_fri_c2_500_5",
    "628_fri_c3_1000_25",
    "631_fri_c1_500_5",
    "634_fri_c2_100_10",
    "635_fri_c0_250_10",
    "637_fri_c1_500_50",
    "641_fri_c1_100_10",
    "643_fri_c2_500_50",
    "644_fri_c4_250_25",
    "645_fri_c3_500_50",
    "646_fri_c3_500_10",
    "647_fri_c1_250_50",
    "648_fri_c1_250_10",
    "649_fri_c0_500_5",
    "650_fri_c0_500_25",
    "651_fri_c0_500_10",
    "653_fri_c0_500_50",
    "654_fri_c0_250_50",
    "656_fri_c1_100_5",
    "657_fri_c2_250_10",
    "658_fri_c3_250_10",
    "659_sleuth_ex1714",
    "663_rabe_266",
    "665_sleuth_case2002",
    "666_rmftsa_ladata",
    "678_visualizing_environmental",
    "687_sleuth_ex1605",
    "690_visualizing_galaxy",
    "695_chatfield_4",
    "706_sleuth_case1202",
    "712_chscase_geyser1",
]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def r2_score(y_true, y_pred):
    """Compute R² score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-15:
        return 1.0 if ss_res < 1e-15 else 0.0
    return 1.0 - ss_res / ss_tot


def mse_score(y_true, y_pred):
    """Compute MSE."""
    return float(np.mean((y_true - y_pred) ** 2))


def model_size(formula_str):
    """Rough complexity measure: count operators and terms."""
    if not formula_str:
        return 0
    # Count distinct operators and operands
    ops = sum(1 for c in formula_str if c in '+-*/^')
    funcs = sum(formula_str.count(f) for f in ['sin', 'cos', 'exp', 'log', 'sqrt'])
    return ops + funcs + 1


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def generate_ground_truth_data(problem, n_samples=500, seed=42):
    """Generate data from a ground-truth problem."""
    name, fn, n_features, x_ranges, formula_str = problem
    rng = np.random.RandomState(seed)

    X = np.column_stack([
        rng.uniform(lo, hi, size=n_samples)
        for lo, hi in (x_ranges if len(x_ranges) == n_features else x_ranges * n_features)
    ])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y = fn(X)

    # Filter non-finite
    mask = np.isfinite(y)
    if mask.sum() < 20:
        return None, None, None
    return X[mask], y[mask], formula_str


def run_track1_blackbox(est, datasets, max_datasets=None, n_samples=500, verbose=True):
    """Track 1: Black-box regression on PMLB datasets."""
    try:
        from pmlb import fetch_data
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pmlb"])
        from pmlb import fetch_data

    results = []
    ds_list = datasets[:max_datasets] if max_datasets else datasets

    print(f"\n{'='*70}")
    print(f"  TRACK 1: BLACK-BOX REGRESSION ({len(ds_list)} datasets)")
    print(f"{'='*70}\n")

    for idx, ds_name in enumerate(ds_list):
        try:
            X, y = fetch_data(ds_name, return_X_y=True)
        except Exception as e:
            if verbose:
                print(f"  [{idx+1:3d}/{len(ds_list)}] {ds_name:40s} SKIP (fetch error: {e})")
            results.append({"dataset": ds_name, "r2": None, "mse": None, "error": str(e)})
            continue

        # Subsample if too large
        if len(y) > n_samples:
            rng = np.random.RandomState(42)
            idx_sub = rng.choice(len(y), n_samples, replace=False)
            X, y = X[idx_sub], y[idx_sub]

        # Train/test split (80/20)
        n_train = int(0.8 * len(y))
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]

        t0 = time.time()
        try:
            est_copy = est.__class__(**est.get_params())
            est_copy.fit(X_train, y_train)
            y_pred = est_copy.predict(X_test)
            elapsed = time.time() - t0

            r2 = r2_score(y_test, y_pred)
            mse = mse_score(y_test, y_pred)
            formula = est_copy.get_formula()
            size = model_size(formula)

            results.append({
                "dataset": ds_name,
                "r2": r2,
                "mse": mse,
                "time": elapsed,
                "formula": formula,
                "model_size": size,
                "n_train": n_train,
                "n_test": len(y_test),
                "n_features": X.shape[1],
                "error": None,
            })

            symbol = "✅" if r2 > 0.9 else "🟡" if r2 > 0.5 else "❌"
            if verbose:
                print(f"  [{idx+1:3d}/{len(ds_list)}] {ds_name:40s} R²={r2:7.4f}  MSE={mse:.3e}  {elapsed:5.1f}s  {symbol}")
        except Exception as e:
            elapsed = time.time() - t0
            if verbose:
                print(f"  [{idx+1:3d}/{len(ds_list)}] {ds_name:40s} ERROR: {e}")
            results.append({"dataset": ds_name, "r2": None, "mse": None, "time": elapsed, "error": str(e)})

    return results


def run_track2_ground_truth(est, problems, max_problems=None, n_samples=500, verbose=True):
    """Track 2: Ground-truth symbolic regression."""
    results = []
    prob_list = problems[:max_problems] if max_problems else problems

    print(f"\n{'='*70}")
    print(f"  TRACK 2: GROUND-TRUTH SYMBOLIC REGRESSION ({len(prob_list)} problems)")
    print(f"{'='*70}\n")

    for idx, problem in enumerate(prob_list):
        name = problem[0]
        X, y, true_formula = generate_ground_truth_data(problem, n_samples=n_samples)

        if X is None:
            if verbose:
                print(f"  [{idx+1:3d}/{len(prob_list)}] {name:40s} SKIP (bad data)")
            results.append({"problem": name, "r2": None, "mse": None, "error": "bad_data"})
            continue

        # Train/test split (80/20)
        n_train = int(0.8 * len(y))
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]

        t0 = time.time()
        try:
            est_copy = est.__class__(**est.get_params())
            est_copy.fit(X_train, y_train)
            y_pred_test = est_copy.predict(X_test)
            elapsed = time.time() - t0

            r2 = r2_score(y_test, y_pred_test)
            mse = mse_score(y_test, y_pred_test)
            formula = est_copy.get_formula()
            size = model_size(formula)

            # Check for symbolic match (very low MSE on full data)
            y_pred_all = est_copy.predict(X)
            full_mse = mse_score(y, y_pred_all)
            exact_match = full_mse < 1e-6

            results.append({
                "problem": name,
                "true_formula": true_formula,
                "discovered_formula": formula,
                "r2": r2,
                "mse": mse,
                "full_mse": full_mse,
                "exact_match": exact_match,
                "time": elapsed,
                "model_size": size,
                "error": None,
            })

            symbol = "✅" if exact_match else "🟡" if r2 > 0.9 else "❌"
            match_str = "EXACT" if exact_match else f"R²={r2:.4f}"
            if verbose:
                print(f"  [{idx+1:3d}/{len(prob_list)}] {name:40s} {match_str:12s}  MSE={mse:.3e}  {elapsed:5.1f}s  {symbol}")
                if verbose and formula:
                    print(f"  {'':44s} → {formula[:80]}")
        except Exception as e:
            elapsed = time.time() - t0
            if verbose:
                print(f"  [{idx+1:3d}/{len(prob_list)}] {name:40s} ERROR: {e}")
            results.append({"problem": name, "r2": None, "mse": None, "time": elapsed, "error": str(e)})

    return results


# ---------------------------------------------------------------------------
# Summary & Report
# ---------------------------------------------------------------------------

def print_summary(track1_results, track2_results, output_dir=None):
    """Print aggregate summary and optionally save to JSON."""
    print(f"\n{'='*70}")
    print(f"  SRBENCH RESULTS SUMMARY — Glassbox")
    print(f"{'='*70}")

    # Track 1 summary
    if track1_results:
        valid = [r for r in track1_results if r.get("r2") is not None]
        r2_vals = [r["r2"] for r in valid]
        times = [r.get("time", 0) for r in valid]
        print(f"\n  TRACK 1 — Black-Box Regression")
        print(f"  {'─'*50}")
        print(f"  Datasets tested:    {len(valid)}/{len(track1_results)}")
        if r2_vals:
            print(f"  Mean R²:            {np.mean(r2_vals):.4f}")
            print(f"  Median R²:          {np.median(r2_vals):.4f}")
            print(f"  R² > 0.9:           {sum(1 for v in r2_vals if v > 0.9)}/{len(r2_vals)}")
            print(f"  R² > 0.5:           {sum(1 for v in r2_vals if v > 0.5)}/{len(r2_vals)}")
            print(f"  Mean time:          {np.mean(times):.2f}s")
            print(f"  Total time:         {sum(times):.1f}s")

    # Track 2 summary
    if track2_results:
        valid = [r for r in track2_results if r.get("r2") is not None]
        r2_vals = [r["r2"] for r in valid]
        exact = [r for r in valid if r.get("exact_match")]
        times = [r.get("time", 0) for r in valid]
        print(f"\n  TRACK 2 — Ground-Truth Symbolic Regression")
        print(f"  {'─'*50}")
        print(f"  Problems tested:    {len(valid)}/{len(track2_results)}")
        if r2_vals:
            print(f"  Exact matches:      {len(exact)}/{len(valid)}")
            print(f"  Mean R²:            {np.mean(r2_vals):.4f}")
            print(f"  Median R²:          {np.median(r2_vals):.4f}")
            print(f"  R² > 0.9:           {sum(1 for v in r2_vals if v > 0.9)}/{len(r2_vals)}")
            print(f"  Mean time:          {np.mean(times):.2f}s")
            print(f"  Total time:         {sum(times):.1f}s")

    print(f"\n{'='*70}\n")

    # Save results JSON
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"srbench_results_{timestamp}.json"

        payload = {
            "model": "Glassbox",
            "timestamp": timestamp,
            "track1": track1_results,
            "track2": track2_results,
        }
        with open(results_file, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"  Results saved to: {results_file}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Glassbox SRBench Benchmark")
    parser.add_argument("--track", type=int, choices=[1, 2], default=None,
                        help="Run specific track only (1=black-box, 2=ground-truth)")
    parser.add_argument("--max-datasets", type=int, default=None,
                        help="Limit number of datasets (for quick testing)")
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Number of data points per problem")
    parser.add_argument("--pop-size", type=int, default=100,
                        help="C++ evolution population size")
    parser.add_argument("--gens", type=int, default=1000,
                        help="C++ evolution generations")
    parser.add_argument("--output-dir", type=str, default="results/srbench",
                        help="Output directory for results JSON")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-dataset output")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Max seconds per problem (default: 120)")
    args = parser.parse_args()

    est = GlassboxRegressor(
        population_size=args.pop_size,
        generations=args.gens,
        random_state=42,
        timeout=args.timeout,
    )

    print(f"\n  Glassbox SRBench Benchmark")
    print(f"  Population: {args.pop_size}  |  Generations: {args.gens}")
    print(f"  Samples: {args.n_samples}")

    track1_results = []
    track2_results = []

    if args.track is None or args.track == 2:
        track2_results = run_track2_ground_truth(
            est, GROUND_TRUTH_PROBLEMS,
            max_problems=args.max_datasets,
            n_samples=args.n_samples,
            verbose=not args.quiet,
        )

    if args.track is None or args.track == 1:
        track1_results = run_track1_blackbox(
            est, PMLB_DATASETS,
            max_datasets=args.max_datasets,
            n_samples=args.n_samples,
            verbose=not args.quiet,
        )

    print_summary(track1_results, track2_results, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
