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
import re
from pathlib import Path
from datetime import datetime
from multiprocessing import get_context

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


def evaluate_formula(formula_str, X):
    """Evaluate a discovered formula string strictly on X.

    Strict mode means no silent domain repairs (for example log(abs(x)))
    and no coercion of invalid values to zero.
    """
    if not formula_str:
        return None

    formula = str(formula_str).strip()
    formula = re.sub(r'\|([^|]+)\|', r'abs(\1)', formula)
    formula = formula.replace('^', '**').replace('np.', '')

    context = {
        "np": np,
        "log": np.log,
        "sin": np.sin,
        "cos": np.cos,
        "exp": np.exp,
        "sqrt": np.sqrt,
        "abs": np.abs,
        "Abs": np.abs,
        "pi": np.pi,
        "E": np.e,
    }
    for i in range(X.shape[1]):
        context[f"x{i}"] = X[:, i]
    if X.shape[1] == 1:
        context["x"] = X[:, 0]

    try:
        with np.errstate(all="raise"):
            y_pred = eval(formula, {"__builtins__": None}, context)
        if isinstance(y_pred, (int, float)):
            y_pred = np.full(X.shape[0], y_pred, dtype=np.float64)
        else:
            y_pred = np.asarray(y_pred, dtype=np.float64)
        if y_pred.shape[0] != X.shape[0]:
            return None
        if not np.all(np.isfinite(y_pred)):
            return None
        return y_pred
    except Exception:
        return None


def simplify_formula_with_guard(formula, X_ref, y_ref, mse_slack=0.02):
    """Simplify formula and keep it only if fidelity remains within tolerance."""
    if not formula:
        return formula

    try:
        from simplify_formula import simplify_onn_formula
    except Exception:
        return formula

    try:
        _, simplified_expr = simplify_onn_formula(
            formula,
            int_tol=0.02,
            zero_tol=2e-3,
            use_nsimplify=True,
            max_passes=6,
            use_identities=True,
        )
        simplified = str(simplified_expr)
    except Exception:
        return formula

    if simplified == formula:
        return formula

    y_orig = evaluate_formula(formula, X_ref)
    y_simpl = evaluate_formula(simplified, X_ref)
    if y_orig is None or y_simpl is None:
        return formula
    mse_orig = mse_score(y_ref, y_orig)
    mse_simpl = mse_score(y_ref, y_simpl)

    # Accept simplification if it preserves fit quality and is at least as compact.
    if mse_simpl <= mse_orig * (1.0 + max(0.0, mse_slack)) and model_size(simplified) <= model_size(formula):
        return simplified
    return formula


def estimate_timeout_budget(base_timeout, n_features, n_train, adaptive_timeout):
    """Scale timeout by problem size when adaptive timeout is enabled."""
    if not adaptive_timeout:
        return int(base_timeout)
    complexity = 1.0 + 0.15 * max(0, n_features - 1) + 0.08 * min(1.0, math.log10(max(50, n_train)) / 3.0)
    budget = int(round(base_timeout * complexity))
    return int(min(max(20, budget), base_timeout * 2))


def parse_seed_list(seed_text):
    """Parse a comma-separated seed list and return sorted unique integers."""
    if seed_text is None:
        return [42]
    parts = [p.strip() for p in str(seed_text).split(",") if p.strip()]
    if not parts:
        return [42]
    seeds = []
    for p in parts:
        seeds.append(int(p))
    return sorted(set(seeds))


def compute_stability_stats(values):
    """Compute median/IQR/std and worst-decile summary for a metric list."""
    if not values:
        return {
            "median": None,
            "iqr": None,
            "std": None,
            "worst_decile": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    q25 = float(np.percentile(arr, 25))
    q50 = float(np.percentile(arr, 50))
    q75 = float(np.percentile(arr, 75))
    if arr.size >= 10:
        worst_count = max(1, int(np.ceil(arr.size * 0.1)))
        worst = np.sort(arr)[:worst_count]
        worst_decile = float(np.mean(worst))
    else:
        worst_decile = float(np.min(arr))
    return {
        "median": q50,
        "iqr": float(q75 - q25),
        "std": float(np.std(arr)),
        "worst_decile": worst_decile,
    }


def classify_failure_taxonomy(true_formula, discovered_formula, r2, mse):
    """Heuristic failure taxonomy for closed-loop mutation/operator analysis."""
    true_f = (true_formula or "").lower()
    disc_f = (discovered_formula or "").lower()

    if not disc_f:
        return "no_formula"

    if "exp(-" in true_f and "exp(" in disc_f and "exp(-" not in disc_f:
        return "exp_sign_error"

    if "/" in true_f and "/" not in disc_f:
        return "rational_denominator_missing"

    true_pows = [int(m.group(1)) for m in re.finditer(r"\*\*\s*(\d+)", true_f)]
    disc_pows = [int(m.group(1)) for m in re.finditer(r"\*\*\s*(\d+)", disc_f)]
    if true_pows:
        true_max = max(true_pows)
        disc_max = max(disc_pows) if disc_pows else 1
        if true_max >= 3 and disc_max < true_max:
            return "missing_high_order_terms"

    if ("sin(" in true_f or "cos(" in true_f) and not ("sin(" in disc_f or "cos(" in disc_f):
        return "periodic_structure_missing"

    if r2 is not None and r2 >= 0.8:
        return "near_miss_structural"
    if mse is not None and np.isfinite(mse) and mse > 1.0:
        return "high_error_instability"
    return "other_structural_failure"


def summarize_time_to_discovery(seed_runs, acceptable_r2=0.9, complexity_cap=20, exact_key="exact_match"):
    """Compute time to first exact and first acceptable formula across seeded runs."""
    times_exact = []
    times_acceptable = []
    for run in seed_runs:
        t = run.get("time")
        if t is None or not np.isfinite(t):
            continue
        is_exact = bool(run.get(exact_key, False))
        r2 = run.get("r2")
        size = run.get("model_size")
        is_acceptable = (
            is_exact
            or (
                r2 is not None
                and np.isfinite(r2)
                and r2 >= acceptable_r2
                and size is not None
                and size <= complexity_cap
            )
        )
        if is_exact:
            times_exact.append(float(t))
        if is_acceptable:
            times_acceptable.append(float(t))

    return {
        "time_to_first_exact": (min(times_exact) if times_exact else None),
        "time_to_first_acceptable": (min(times_acceptable) if times_acceptable else None),
    }


def summarize_seed_runs(seed_runs):
    """Aggregate per-seed runs into protocol-compliant stability summaries."""
    valid = [r for r in seed_runs if r.get("r2") is not None]
    if not valid:
        return {
            "seed_count": len(seed_runs),
            "valid_seed_count": 0,
            "r2_stats": compute_stability_stats([]),
            "mse_stats": compute_stability_stats([]),
            "time_stats": compute_stability_stats([]),
            "exact_recovery_rate": None,
            "exact_recovery_stats": compute_stability_stats([]),
        }

    r2_vals = [r["r2"] for r in valid if np.isfinite(r["r2"])]
    mse_vals = [r["mse"] for r in valid if r.get("mse") is not None and np.isfinite(r["mse"])]
    time_vals = [r["time"] for r in valid if r.get("time") is not None and np.isfinite(r["time"])]
    exact_binary = [1.0 if r.get("exact_match") else 0.0 for r in valid]

    return {
        "seed_count": len(seed_runs),
        "valid_seed_count": len(valid),
        "r2_stats": compute_stability_stats(r2_vals),
        "mse_stats": compute_stability_stats(mse_vals),
        "time_stats": compute_stability_stats(time_vals),
        "exact_recovery_rate": float(np.mean(exact_binary)) if exact_binary else None,
        "exact_recovery_stats": compute_stability_stats(exact_binary),
    }


def _fit_worker(payload, queue):
    """Child-process worker for hard timeout enforcement."""
    import signal
    import sys
    
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Process timeout after {payload.get('timeout_seconds', '?')}s")
    
    try:
        timeout_sec = payload.get("timeout_seconds", 120)
        if sys.platform != "win32":
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout_sec))
        
        est = GlassboxRegressor(**payload["est_params"])
        X_train = payload["X_train"]
        y_train = payload["y_train"]
        X_test = payload["X_test"]
        X_full = payload.get("X_full")

        t0 = time.time()
        est.fit(X_train, y_train)
        fit_time = time.time() - t0

        if sys.platform != "win32":
            signal.alarm(0)

        formula = est.get_formula()
        y_pred_test = est.predict(X_test)
        y_pred_full = est.predict(X_full) if X_full is not None else None

        queue.put({
            "status": "ok",
            "fit_time": fit_time,
            "formula": formula,
            "y_pred_test": y_pred_test,
            "y_pred_full": y_pred_full,
        })
    except TimeoutError as te:
        queue.put({"status": "timeout", "error": str(te)})
    except Exception as err:
        queue.put({"status": "error", "error": str(err)})


def run_with_hard_timeout(est_params, X_train, y_train, X_test, timeout_seconds, X_full=None):
    """Run fit/predict in a separate process and enforce a hard wall-clock timeout."""
    ctx = get_context("spawn")
    queue = ctx.Queue(maxsize=1)
    payload = {
        "est_params": est_params,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "X_full": X_full,
        "timeout_seconds": timeout_seconds,
    }

    t0 = time.time()
    process = ctx.Process(target=_fit_worker, args=(payload, queue), daemon=True)
    try:
        process.start()
        wall_timeout = timeout_seconds + 2
        process.join(timeout=wall_timeout)

        if process.is_alive():
            elapsed = time.time() - t0
            try:
                process.terminate()
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=1)
            except Exception:
                pass
            return {"status": "timeout", "error": f"hard timeout after {elapsed:.1f}s"}

        if not queue.empty():
            result = queue.get()
            if result.get("status") == "timeout":
                result["error"] = f"process-level: {result['error']}"
            return result

        if process.exitcode == 0:
            return {"status": "timeout", "error": f"no result after {time.time() - t0:.1f}s"}
        else:
            return {"status": "error", "error": f"worker crashed with code {process.exitcode}"}
    except Exception as outer_err:
        return {"status": "error", "error": f"spawn error: {outer_err}"}


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


def run_track1_blackbox(
    est,
    datasets,
    max_datasets=None,
    n_samples=500,
    seeds=None,
    verbose=True,
    hard_timeout=True,
    adaptive_timeout=True,
    post_simplify=False,
    skip_evolution_if_bloated=True,
):
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

    seeds = list(seeds or [42])

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

        timeout_budget = estimate_timeout_budget(
            base_timeout=est.get_params().get("timeout", 120),
            n_features=X.shape[1],
            n_train=n_train,
            adaptive_timeout=adaptive_timeout,
        )

        seed_runs = []
        for seed in seeds:
            t0 = time.time()
            try:
                est_params = est.get_params().copy()
                est_params["timeout"] = timeout_budget
                est_params["random_state"] = seed

                # If bloat-skip enabled, make evolution skip more aggressively on good fast-path results
                if skip_evolution_if_bloated:
                    est_params["skip_evolution_if_bloated"] = True
                    est_params["bloat_term_threshold"] = 20

                if hard_timeout:
                    run_result = run_with_hard_timeout(
                        est_params,
                        X_train,
                        y_train,
                        X_test,
                        timeout_seconds=timeout_budget + 5,
                    )
                    elapsed = time.time() - t0
                    if run_result["status"] != "ok":
                        seed_runs.append({
                            "seed": seed,
                            "r2": None,
                            "mse": None,
                            "time": elapsed,
                            "error": run_result.get("error", run_result["status"]),
                        })
                        continue

                    formula = run_result.get("formula", "")
                    if post_simplify and formula:
                        formula = simplify_formula_with_guard(formula, X_train, y_train)
                    y_pred = evaluate_formula(formula, X_test)
                else:
                    est_copy = est.__class__(**est_params)
                    est_copy.fit(X_train, y_train)
                    formula = est_copy.get_formula()
                    if post_simplify and formula:
                        formula = simplify_formula_with_guard(formula, X_train, y_train)
                    y_pred = evaluate_formula(formula, X_test)
                    elapsed = time.time() - t0

                if y_pred is None:
                    seed_runs.append({
                        "seed": seed,
                        "r2": None,
                        "mse": None,
                        "time": elapsed,
                        "formula": formula,
                        "model_size": model_size(formula),
                        "error": "formula_eval_failed",
                    })
                    continue

                r2 = r2_score(y_test, y_pred)
                mse = mse_score(y_test, y_pred)
                size = model_size(formula)
                seed_runs.append({
                    "seed": seed,
                    "r2": r2,
                    "mse": mse,
                    "time": elapsed,
                    "formula": formula,
                    "model_size": size,
                    "error": None,
                })
            except Exception as e:
                elapsed = time.time() - t0
                seed_runs.append({"seed": seed, "r2": None, "mse": None, "time": elapsed, "error": str(e)})

        valid_seed_runs = [r for r in seed_runs if r.get("r2") is not None]
        if not valid_seed_runs:
            err_msg = "; ".join(str(r.get("error")) for r in seed_runs if r.get("error")) or "all_seeds_failed"
            if verbose:
                print(f"  [{idx+1:3d}/{len(ds_list)}] {ds_name:40s} ERROR: {err_msg}")
            results.append({
                "dataset": ds_name,
                "r2": None,
                "mse": None,
                "time": None,
                "error": err_msg,
                "seed_runs": seed_runs,
                "stability": summarize_seed_runs(seed_runs),
                "scoring_source": "displayed_formula",
            })
            continue

        stability = summarize_seed_runs(seed_runs)
        aggregate = {
            "dataset": ds_name,
            "r2": stability["r2_stats"]["median"],
            "mse": stability["mse_stats"]["median"],
            "time": stability["time_stats"]["median"],
            "formula": valid_seed_runs[0].get("formula", ""),
            "model_size": valid_seed_runs[0].get("model_size"),
            "n_train": n_train,
            "n_test": len(y_test),
            "n_features": X.shape[1],
            "error": None,
            "seed_runs": seed_runs,
            "stability": stability,
            "time_to_discovery": summarize_time_to_discovery(seed_runs, exact_key="exact_match"),
            "scoring_source": "displayed_formula",
        }
        results.append(aggregate)

        median_r2 = aggregate["r2"]
        median_mse = aggregate["mse"]
        median_time = aggregate["time"]
        symbol = "✅" if median_r2 > 0.9 else "🟡" if median_r2 > 0.5 else "❌"
        if verbose:
            print(
                f"  [{idx+1:3d}/{len(ds_list)}] {ds_name:40s} "
                f"R²(med)={median_r2:7.4f}  MSE(med)={median_mse:.3e}  "
                f"{median_time:5.1f}s  {symbol}  "
                f"(budget={timeout_budget}s, seeds={len(seeds)})"
            )

    return results


def run_track2_ground_truth(
    est,
    problems,
    max_problems=None,
    n_samples=500,
    seeds=None,
    verbose=True,
    hard_timeout=True,
    adaptive_timeout=True,
    post_simplify=False,
    skip_evolution_if_bloated=False,
    acceptable_r2=0.9,
    complexity_cap=20,
):
    """Track 2: Ground-truth symbolic regression."""
    results = []
    prob_list = problems[:max_problems] if max_problems else problems

    print(f"\n{'='*70}")
    print(f"  TRACK 2: GROUND-TRUTH SYMBOLIC REGRESSION ({len(prob_list)} problems)")
    print(f"{'='*70}\n")

    seeds = list(seeds or [42])

    for idx, problem in enumerate(prob_list):
        name = problem[0]
        seed_runs = []
        for seed in seeds:
            X, y, true_formula = generate_ground_truth_data(problem, n_samples=n_samples, seed=seed)

            if X is None:
                seed_runs.append({"seed": seed, "r2": None, "mse": None, "time": None, "error": "bad_data"})
                continue

            # Train/test split (80/20)
            n_train = int(0.8 * len(y))
            X_train, X_test = X[:n_train], X[n_train:]
            y_train, y_test = y[:n_train], y[n_train:]

            timeout_budget = estimate_timeout_budget(
                base_timeout=est.get_params().get("timeout", 120),
                n_features=X.shape[1],
                n_train=n_train,
                adaptive_timeout=adaptive_timeout,
            )
            t0 = time.time()
            try:
                est_params = est.get_params().copy()
                est_params["timeout"] = timeout_budget
                est_params["random_state"] = seed

                # If bloat-skip enabled, make evolution skip more aggressively on good fast-path results
                if skip_evolution_if_bloated:
                    est_params["skip_evolution_if_bloated"] = True
                    est_params["bloat_term_threshold"] = 20

                if hard_timeout:
                    run_result = run_with_hard_timeout(
                        est_params,
                        X_train,
                        y_train,
                        X_test,
                        timeout_seconds=timeout_budget + 5,
                        X_full=X,
                    )
                    elapsed = time.time() - t0
                    if run_result["status"] != "ok":
                        seed_runs.append({
                            "seed": seed,
                            "r2": None,
                            "mse": None,
                            "time": elapsed,
                            "error": run_result.get("error", run_result["status"]),
                        })
                        continue

                    formula = run_result.get("formula", "")
                    if post_simplify and formula:
                        formula = simplify_formula_with_guard(formula, X_train, y_train)
                    y_pred_all = evaluate_formula(formula, X)
                else:
                    est_copy = est.__class__(**est_params)
                    est_copy.fit(X_train, y_train)
                    formula = est_copy.get_formula()
                    if post_simplify and formula:
                        formula = simplify_formula_with_guard(formula, X_train, y_train)
                    y_pred_all = evaluate_formula(formula, X)
                    elapsed = time.time() - t0

                if y_pred_all is None:
                    seed_runs.append({
                        "seed": seed,
                        "r2": None,
                        "mse": None,
                        "time": elapsed,
                        "problem": name,
                        "true_formula": true_formula,
                        "discovered_formula": formula,
                        "model_size": model_size(formula),
                        "error": "formula_eval_failed",
                    })
                    continue

                y_pred_test = y_pred_all[n_train:]

                r2 = r2_score(y_test, y_pred_test)
                mse = mse_score(y_test, y_pred_test)
                size = model_size(formula)

                # Check for symbolic match (very low MSE on full data)
                full_mse = mse_score(y, y_pred_all)
                exact_match = full_mse < 1e-6

                failure_bucket = None
                if not exact_match:
                    failure_bucket = classify_failure_taxonomy(true_formula, formula, r2, full_mse)

                seed_runs.append({
                    "seed": seed,
                    "problem": name,
                    "true_formula": true_formula,
                    "discovered_formula": formula,
                    "r2": r2,
                    "mse": mse,
                    "full_mse": full_mse,
                    "exact_match": exact_match,
                    "time": elapsed,
                    "model_size": size,
                    "failure_bucket": failure_bucket,
                    "error": None,
                })
            except Exception as e:
                elapsed = time.time() - t0
                seed_runs.append({"seed": seed, "r2": None, "mse": None, "time": elapsed, "error": str(e)})

        valid_seed_runs = [r for r in seed_runs if r.get("r2") is not None]
        if not valid_seed_runs:
            err_msg = "; ".join(str(r.get("error")) for r in seed_runs if r.get("error")) or "all_seeds_failed"
            if verbose:
                print(f"  [{idx+1:3d}/{len(prob_list)}] {name:40s} ERROR: {err_msg}")
            results.append({
                "problem": name,
                "r2": None,
                "mse": None,
                "time": None,
                "error": err_msg,
                "seed_runs": seed_runs,
                "stability": summarize_seed_runs(seed_runs),
                "scoring_source": "displayed_formula",
            })
            continue

        stability = summarize_seed_runs(seed_runs)
        time_to_discovery = summarize_time_to_discovery(
            seed_runs,
            acceptable_r2=acceptable_r2,
            complexity_cap=complexity_cap,
            exact_key="exact_match",
        )

        median_run = min(
            valid_seed_runs,
            key=lambda r: abs(float(r["r2"]) - float(stability["r2_stats"]["median"]))
        )

        aggregate = {
            "problem": name,
            "true_formula": median_run.get("true_formula"),
            "discovered_formula": median_run.get("discovered_formula"),
            "r2": stability["r2_stats"]["median"],
            "mse": stability["mse_stats"]["median"],
            "full_mse": float(np.median([r["full_mse"] for r in valid_seed_runs if r.get("full_mse") is not None])),
            "exact_match": bool(stability.get("exact_recovery_rate", 0.0) == 1.0),
            "exact_recovery_rate": stability.get("exact_recovery_rate"),
            "time": stability["time_stats"]["median"],
            "model_size": median_run.get("model_size"),
            "error": None,
            "seed_runs": seed_runs,
            "stability": stability,
            "time_to_discovery": time_to_discovery,
            "scoring_source": "displayed_formula",
        }
        results.append(aggregate)

        median_r2 = aggregate["r2"]
        median_mse = aggregate["mse"]
        median_time = aggregate["time"]
        exact_rate = aggregate.get("exact_recovery_rate", 0.0) or 0.0
        symbol = "✅" if exact_rate >= 0.8 else "🟡" if median_r2 > 0.9 else "❌"
        if verbose:
            print(
                f"  [{idx+1:3d}/{len(prob_list)}] {name:40s} "
                f"EXACT-rate={exact_rate:5.2f}  R²(med)={median_r2:7.4f}  "
                f"MSE(med)={median_mse:.3e}  {median_time:5.1f}s  {symbol}  "
                f"(seeds={len(seeds)})"
            )
            if median_run.get("discovered_formula"):
                print(f"  {'':44s} → {median_run['discovered_formula'][:80]}")

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
        r2_worst_decile = compute_stability_stats(r2_vals).get("worst_decile") if r2_vals else None
        print(f"\n  TRACK 1 — Black-Box Regression")
        print(f"  {'─'*50}")
        print(f"  Datasets tested:    {len(valid)}/{len(track1_results)}")
        if r2_vals:
            print(f"  Mean R²:            {np.mean(r2_vals):.4f}")
            print(f"  Median R²:          {np.median(r2_vals):.4f}")
            print(f"  R² IQR:             {compute_stability_stats(r2_vals)['iqr']:.4f}")
            print(f"  R² Std:             {compute_stability_stats(r2_vals)['std']:.4f}")
            print(f"  R² Worst-decile:    {r2_worst_decile:.4f}")
            print(f"  R² > 0.9:           {sum(1 for v in r2_vals if v > 0.9)}/{len(r2_vals)}")
            print(f"  R² > 0.5:           {sum(1 for v in r2_vals if v > 0.5)}/{len(r2_vals)}")
            print(f"  Mean time:          {np.mean(times):.2f}s")
            print(f"  Total time:         {sum(times):.1f}s")

    # Track 2 summary
    if track2_results:
        valid = [r for r in track2_results if r.get("r2") is not None]
        r2_vals = [r["r2"] for r in valid]
        exact = [r for r in valid if r.get("exact_match")]
        exact_rates = [r.get("exact_recovery_rate") for r in valid if r.get("exact_recovery_rate") is not None]
        tfe = [r.get("time_to_discovery", {}).get("time_to_first_exact") for r in valid if r.get("time_to_discovery", {}).get("time_to_first_exact") is not None]
        tfa = [r.get("time_to_discovery", {}).get("time_to_first_acceptable") for r in valid if r.get("time_to_discovery", {}).get("time_to_first_acceptable") is not None]
        times = [r.get("time", 0) for r in valid]
        failure_counts = {}
        for row in valid:
            for seed_run in row.get("seed_runs", []):
                bucket = seed_run.get("failure_bucket")
                if not bucket:
                    continue
                failure_counts[bucket] = failure_counts.get(bucket, 0) + 1
        print(f"\n  TRACK 2 — Ground-Truth Symbolic Regression")
        print(f"  {'─'*50}")
        print(f"  Problems tested:    {len(valid)}/{len(track2_results)}")
        if r2_vals:
            print(f"  Exact matches:      {len(exact)}/{len(valid)}")
            print(f"  Mean R²:            {np.mean(r2_vals):.4f}")
            print(f"  Median R²:          {np.median(r2_vals):.4f}")
            print(f"  R² IQR:             {compute_stability_stats(r2_vals)['iqr']:.4f}")
            print(f"  R² Std:             {compute_stability_stats(r2_vals)['std']:.4f}")
            print(f"  R² Worst-decile:    {compute_stability_stats(r2_vals)['worst_decile']:.4f}")
            if exact_rates:
                print(f"  Mean exact-rate:    {np.mean(exact_rates):.4f}")
            print(f"  R² > 0.9:           {sum(1 for v in r2_vals if v > 0.9)}/{len(r2_vals)}")
            print(f"  Mean time:          {np.mean(times):.2f}s")
            print(f"  Total time:         {sum(times):.1f}s")
            if tfe:
                print(f"  TTF exact (median): {np.median(tfe):.2f}s")
            if tfa:
                print(f"  TTF accept (median):{np.median(tfa):.2f}s")
            if failure_counts:
                top_failures = sorted(failure_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
                failure_text = ", ".join([f"{k}:{v}" for k, v in top_failures])
                print(f"  Top failures:       {failure_text}")

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
    parser.add_argument("--timeout", type=int, default=60,
                        help="Max seconds per problem (default: 60s for PMLB)")
    parser.add_argument("--no-hard-timeout", action="store_true",
                        help="Disable process-level hard timeout enforcement")
    parser.add_argument("--no-adaptive-timeout", action="store_true",
                        help="Disable adaptive timeout scaling (now enabled by default)")
    parser.add_argument("--post-simplify", action="store_true",
                        help="Post-simplify formulas with a fidelity guard")
    parser.add_argument("--skip-evolution-if-bloated", action="store_true",
                        help="Skip C++ evolution if fast-path formula exceeds 20 terms")
    parser.add_argument("--seeds", type=str, default="42,1337,2027,7,11",
                        help="Comma-separated fixed seeds for reproducible multi-seed protocol")
    parser.add_argument("--acceptable-r2", type=float, default=0.9,
                        help="R2 threshold for acceptable discovery metric")
    parser.add_argument("--complexity-cap", type=int, default=20,
                        help="Complexity cap (model size) for acceptable discovery metric")
    args = parser.parse_args()

    seeds = parse_seed_list(args.seeds)

    est = GlassboxRegressor(
        population_size=args.pop_size,
        generations=args.gens,
        random_state=42,
        timeout=args.timeout,
    )

    print(f"\n  Glassbox SRBench Benchmark")
    print(f"  Population: {args.pop_size}  |  Generations: {args.gens}")
    print(f"  Samples: {args.n_samples}")
    print(f"  Timeout: {args.timeout}s  |  Hard timeout: {not args.no_hard_timeout}")
    adaptive_on = not args.no_adaptive_timeout
    print(f"  Adaptive timeout: {adaptive_on}  |  Post simplify: {args.post_simplify}")
    print(f"  Skip bloat: {args.skip_evolution_if_bloated}")
    print(f"  Multi-seed protocol: {seeds}")
    print(f"  Acceptable criteria: R2>={args.acceptable_r2:.2f}, size<={args.complexity_cap}")

    track1_results = []
    track2_results = []

    adaptive_timeout_enabled = not args.no_adaptive_timeout

    if args.track is None or args.track == 2:
        track2_results = run_track2_ground_truth(
            est, GROUND_TRUTH_PROBLEMS,
            max_problems=args.max_datasets,
            n_samples=args.n_samples,
            seeds=seeds,
            verbose=not args.quiet,
            hard_timeout=not args.no_hard_timeout,
            adaptive_timeout=adaptive_timeout_enabled,
            post_simplify=args.post_simplify,
            skip_evolution_if_bloated=args.skip_evolution_if_bloated,
            acceptable_r2=args.acceptable_r2,
            complexity_cap=args.complexity_cap,
        )

    if args.track is None or args.track == 1:
        track1_results = run_track1_blackbox(
            est, PMLB_DATASETS,
            max_datasets=args.max_datasets,
            n_samples=args.n_samples,
            seeds=seeds,
            verbose=not args.quiet,
            hard_timeout=not args.no_hard_timeout,
            adaptive_timeout=adaptive_timeout_enabled,
            post_simplify=args.post_simplify,
            skip_evolution_if_bloated=args.skip_evolution_if_bloated,
        )

    print_summary(track1_results, track2_results, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
