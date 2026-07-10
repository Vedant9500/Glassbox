"""Phase 0: deterministic noise-handling benchmark protocol.

Goal (see ``noise_handling_phases.md`` Phase 0): make current noise behaviour
*measurable* before changing objectives. This module provides:

- Deterministic, level-pinned noise generators (clean, Gaussian 0.1/1/10% RMS,
  pink, quantization, outliers) extracted from the randomised logic in
  ``glassbox/curve_classifier/generate_curve_data.py`` but made reproducible.
- A fixed ``NOISE_TIERS`` registry.
- ``run_noise_protocol(...)``: problem x tier x seed sweep that emits the
  Phase 0 report columns required by the tracker:
  ``noise_level``, ``noise_type``, ``sample_weight_mode``, ``raw_mse``,
  ``display_mse``, ``holdout_mse``, ``clean_test_mse``, ``clean_test_r2``,
  ``clean_full_mse``, ``formula_complexity``, ``false_confidence``,
  ``false_confidence_vs_clean``, ``seed_graphs_used``, ``exact_match``,
  ``acceptable_clean``.

Clean-target columns measure structure recovery; noisy R2/MSE measure fit
to noisy labels. See ``noise_handling_audit.md``.
- ``summarize_noise_protocol(...)``: clean-vs-noisy delta table + per-tier
  failure-bucket rollup.

This phase produces measurement, not improvement. Downstream phases validate
against the JSON / Markdown tables emitted here.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_REPO_ROOT))
if str(_SCRIPT_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_SCRIPT_DIR))

from scripts import benchmark_common as bc

# Lazy import: run_srbench_local pulls Glassbox/torch. Keep noise generators
# and pure helpers importable without the full stack.
rsl = None  # type: ignore


def _rsl():
    global rsl
    if rsl is None:
        from scripts import run_srbench_local as _mod
        rsl = _mod
    return rsl


# ---------------------------------------------------------------------------
# Deterministic noise generators
# ---------------------------------------------------------------------------
def _require_rng(seed: Any) -> np.random.RandomState:
    return np.random.RandomState(int(seed))


def noise_amplitude_scale(y: np.ndarray) -> float:
    """Signal scale used to size additive noise.

    Prefer ``std(y)`` when the target has spread. Near-constant targets have
    ``std ≈ 0``, which previously wiped out Gaussian/pink/outlier noise and made
    constants look free under noise studies. Fall back to mean absolute level,
    then ``1.0``, so a constant ``5`` at 10% noise still gets ``std ≈ 0.5``.
    """
    arr = np.asarray(y, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 1.0
    y_std = float(np.std(arr))
    y_abs = float(np.mean(np.abs(arr)))
    floor = max(1e-12, 1e-6 * max(y_abs, 1.0))
    if y_std > floor:
        return y_std
    if y_abs > 1e-12:
        return y_abs
    return 1.0


def add_gaussian_noise(
    y: np.ndarray, rms_fraction: float, *, seed: int = 0
) -> np.ndarray:
    """Additive white Gaussian noise; std = ``rms_fraction * noise_amplitude_scale(y)``."""
    rng = _require_rng(seed)
    scale = noise_amplitude_scale(y)
    return np.asarray(y, dtype=np.float64) + rng.normal(
        0.0, float(rms_fraction) * scale, size=np.asarray(y).shape
    )


def add_pink_noise(
    y: np.ndarray, rms_fraction: float, *, seed: int = 0
) -> np.ndarray:
    """Correlated 1/f (pink) noise; std = ``rms_fraction * noise_amplitude_scale(y)``."""
    rng = _require_rng(seed)
    n = int(len(y))
    white = rng.standard_normal(n)
    fft_white = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0)
    freqs[0] = 1.0
    pink = np.fft.irfft(fft_white / np.sqrt(freqs), n=n)
    scale = noise_amplitude_scale(y)
    pink = pink / (float(np.std(pink)) + 1e-12) * (float(rms_fraction) * scale)
    return np.asarray(y, dtype=np.float64) + pink


def add_quantization_noise(
    y: np.ndarray, n_levels: int = 64, *, seed: int = 0
) -> np.ndarray:
    """Round to ``n_levels`` uniform steps (simulates an ADC).

    For near-constant signals (span ≈ 0) a synthetic span of
    ``2 * noise_amplitude_scale(y)`` is used so quantization is not a no-op.
    ``seed`` is accepted for API symmetry; quantization is deterministic given
    ``y`` and ``n_levels``.
    """
    del seed  # deterministic; kept for call-site symmetry
    arr = np.asarray(y, dtype=np.float64).copy()
    y_min, y_max = float(arr.min()), float(arr.max())
    span = y_max - y_min
    if span <= 1e-12:
        mid = float(np.mean(arr))
        half = noise_amplitude_scale(arr)
        y_min, y_max = mid - half, mid + half
        span = y_max - y_min
    quantized = (
        np.round((arr - y_min) / span * int(n_levels)) / int(n_levels) * span + y_min
    )
    return quantized


def add_outlier_noise(
    y: np.ndarray, fraction: float, *, magnitude_stds: float = 3.0, seed: int = 0
) -> np.ndarray:
    """Corrupt ``fraction`` of points with spikes ~``magnitude_stds`` of signal scale."""
    rng = _require_rng(seed)
    arr = np.asarray(y, dtype=np.float64).copy()
    n = int(len(arr))
    n_outliers = max(1, int(round(n * float(fraction))))
    indices = rng.choice(n, size=n_outliers, replace=False)
    scale = noise_amplitude_scale(arr)
    arr[indices] += rng.normal(0.0, float(magnitude_stds) * scale, size=n_outliers)
    return arr


# ---------------------------------------------------------------------------
# Fixed tier registry
# ---------------------------------------------------------------------------
# Each tier: (tier_name, noise_type, level, builder)
# level is RMS-fraction for gaussian/pink, fraction-of-points for outliers,
# n_levels for quantization. clean has level 0.0.
NOISE_TIERS: List[Dict[str, Any]] = [
    {"name": "clean", "noise_type": "clean", "noise_level": 0.0},
    {"name": "gaussian_0.1pct", "noise_type": "gaussian", "noise_level": 0.001},
    {"name": "gaussian_1pct", "noise_type": "gaussian", "noise_level": 0.01},
    {"name": "gaussian_10pct", "noise_type": "gaussian", "noise_level": 0.10},
    {"name": "pink_5pct", "noise_type": "pink", "noise_level": 0.05},
    {"name": "quantization_64", "noise_type": "quantization", "noise_level": 64.0},
    {"name": "outliers_3pct", "noise_type": "outliers", "noise_level": 0.03},
]


def apply_noise_tier(
    y: np.ndarray, tier: Dict[str, Any], *, seed: int
) -> np.ndarray:
    """Apply a single tier deterministically. Clean tier returns y unchanged."""
    y = np.asarray(y, dtype=np.float64)
    ntype = str(tier.get("noise_type", "clean"))
    level = float(tier.get("noise_level", 0.0))
    if ntype == "clean" or level == 0.0:
        return y.copy()
    if ntype == "gaussian":
        return add_gaussian_noise(y, level, seed=seed)
    if ntype == "pink":
        return add_pink_noise(y, level, seed=seed)
    if ntype == "quantization":
        return add_quantization_noise(y, n_levels=int(level), seed=seed)
    if ntype == "outliers":
        return add_outlier_noise(y, fraction=level, seed=seed)
    raise ValueError(f"unknown noise_type: {ntype}")


def make_seeded_train_test_split(
    X, y, *, n_samples: int, seed: int, train_fraction: float = 0.8
):
    """Deterministic subsample + ordered train/test split (numpy-only)."""
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    rng = np.random.RandomState(int(seed))
    n_total = len(y_arr)
    if n_total > int(n_samples):
        indices = rng.choice(n_total, int(n_samples), replace=False)
    else:
        indices = rng.permutation(n_total)
    X_sel = X_arr[indices]
    y_sel = y_arr[indices]
    n_train = int(float(train_fraction) * len(y_sel))
    n_train = max(1, min(len(y_sel) - 1, n_train))
    return X_sel[:n_train], X_sel[n_train:], y_sel[:n_train], y_sel[n_train:]


def generate_ground_truth_data(problem: Tuple, n_samples: int = 500, seed: int = 42):
    """Generate (X, y, formula_str) for a problem tuple without importing Glassbox."""
    name, fn, n_features, x_ranges, formula_str = problem
    rng = np.random.RandomState(int(seed))
    ranges = x_ranges if len(x_ranges) == n_features else list(x_ranges) * int(n_features)
    X = np.column_stack([
        rng.uniform(lo, hi, size=int(n_samples)) for lo, hi in ranges
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y = fn(X)
    mask = np.isfinite(y)
    if int(mask.sum()) < 20:
        return None, None, None
    return X[mask], y[mask], formula_str


# Built-in catalogue for protocol baseline / smoke without torch.
BUILTIN_PROBLEMS: Dict[str, Tuple] = {
    "Poly-x2": (
        "Poly-x2",
        lambda X: X[:, 0] ** 2,
        1,
        [(-5.0, 5.0)],
        "x0**2",
    ),
    "Poly-x3-x": (
        "Poly-x3-x",
        lambda X: X[:, 0] ** 3 - X[:, 0],
        1,
        [(-3.0, 3.0)],
        "x0**3 - x0",
    ),
    "Nguyen-1": (
        "Nguyen-1",
        lambda X: X[:, 0] ** 3 + X[:, 0] ** 2 + X[:, 0],
        1,
        [(-1.0, 1.0)],
        "x0**3 + x0**2 + x0",
    ),
    "Nguyen-5": (
        "Nguyen-5",
        lambda X: np.sin(X[:, 0] ** 2) * np.cos(X[:, 0]) - 1.0,
        1,
        [(-1.0, 1.0)],
        "sin(x0**2)*cos(x0) - 1",
    ),
    "Keijzer-4": (
        "Keijzer-4",
        lambda X: 0.3 * X[:, 0] * np.sin(2.0 * np.pi * X[:, 0]),
        1,
        [(-3.0, 3.0)],
        "0.3*x0*sin(2*pi*x0)",
    ),
    "Feynman-I.6.20a": (
        "Feynman-I.6.20a",
        lambda X: np.exp(-(X[:, 0] ** 2) / 2.0) / np.sqrt(2.0 * np.pi),
        1,
        [(-3.0, 3.0)],
        "exp(-x0**2/2) / sqrt(2*pi)",
    ),
}


# ---------------------------------------------------------------------------
# Phase 0 report-column helpers
# ---------------------------------------------------------------------------
def _complexity(formula: str) -> int:
    try:
        return bc.model_size(formula)
    except Exception:
        return 0


def _sample_weight_mode(estimator: Any) -> str:
    """Record which weight mode the fit used. Phase 1 wiring point."""
    diag = getattr(estimator, "blackbox_diagnostics_", None)
    if isinstance(diag, dict):
        sw = diag.get("sample_weight")
        if isinstance(sw, dict) and sw.get("provided"):
            return "provided"
    return "none"


def _false_confidence(
    *, train_r2: Optional[float], test_r2: Optional[float], threshold: float = 0.95
) -> Optional[bool]:
    """True when train looks great but test collapses (the false-confidence case)."""
    if train_r2 is None or test_r2 is None:
        return None
    if not (math.isfinite(float(train_r2)) and math.isfinite(float(test_r2))):
        return None
    return float(train_r2) >= float(threshold) and float(test_r2) < float(threshold)


def _seed_graphs_used(estimator: Any) -> int:
    md = getattr(estimator, "blackbox_diagnostics_", None)
    if isinstance(md, dict):
        for key in ("seed_graphs_used", "seed_graph_count", "n_seed_graphs"):
            val = md.get(key)
            if isinstance(val, (int, float)):
                return int(val)
    return 0


# ---------------------------------------------------------------------------
# Protocol runner
# ---------------------------------------------------------------------------
def run_noise_protocol(
    estimator_factory,
    problems: Sequence[Tuple],
    *,
    tiers: Optional[Sequence[Dict[str, Any]]] = None,
    seeds: Optional[Iterable[int]] = None,
    n_samples: int = 300,
    train_fraction: float = 0.8,
    verbose: bool = True,
    acceptable_r2: float = 0.9,
) -> List[Dict[str, Any]]:
    """Run problem x tier x seed sweep and collect Phase 0 report rows.

    ``estimator_factory`` is a zero-arg callable returning a fresh unfitted
    estimator (so each run is independent). ``problems`` follow the
    ``GROUND_TRUTH_PROBLEMS`` tuple shape used by ``run_srbench_local``.
    """
    tiers = list(tiers) if tiers is not None else list(NOISE_TIERS)
    seeds = list(seeds) if seeds is not None else [11, 23, 47, 89, 137]
    rows: List[Dict[str, Any]] = []

    for problem in problems:
        name = problem[0]
        for tier in tiers:
            tier_name = str(tier["name"])
            for seed in seeds:
                row = _run_single(
                    estimator_factory,
                    problem,
                    tier,
                    int(seed),
                    n_samples=n_samples,
                    train_fraction=train_fraction,
                    acceptable_r2=acceptable_r2,
                )
                row["problem"] = name
                row["tier"] = tier_name
                row["seed"] = int(seed)
                rows.append(row)
                if verbose:
                    ok = "OK" if row.get("exact_match") else (
                        "MID" if (row.get("test_r2") or 0.0) > acceptable_r2 else "LOW"
                    )
                    print(
                        f"  {name:24s} {tier_name:18s} seed={seed:4d} "
                        f"R2_test={row.get('test_r2')}  exact={row.get('exact_match')}  {ok}"
                    )
    return rows


def _run_single(
    estimator_factory,
    problem: Tuple,
    tier: Dict[str, Any],
    seed: int,
    *,
    n_samples: int,
    train_fraction: float,
    acceptable_r2: float,
) -> Dict[str, Any]:
    name, fn, n_features, x_ranges, formula_str = problem
    X_clean, y_clean, true_formula = generate_ground_truth_data(
        problem, n_samples=n_samples, seed=seed
    )
    if X_clean is None:
        return {
            "noise_type": tier.get("noise_type"),
            "noise_level": tier.get("noise_level"),
            "sample_weight_mode": "none",
            "raw_mse": None,
            "display_mse": None,
            "holdout_mse": None,
            "clean_test_mse": None,
            "clean_test_r2": None,
            "clean_full_mse": None,
            "formula_complexity": None,
            "false_confidence": None,
            "false_confidence_vs_clean": None,
            "seed_graphs_used": 0,
            "train_r2": None,
            "test_r2": None,
            "exact_match": False,
            "acceptable_clean": False,
            "discovered_formula": None,
            "true_formula": true_formula,
            "error": "bad_data",
        }

    # Deterministic subsample + ordered train/test split (shared helper).
    # Noise is applied after the split so train/test share one noise draw on
    # the selected vector while clean labels remain available for recovery.
    X_train, X_test, y_train_clean, y_test_clean = make_seeded_train_test_split(
        X_clean, y_clean, n_samples=n_samples, seed=seed, train_fraction=train_fraction
    )
    X_sel = np.vstack([X_train, X_test])
    y_sel = np.concatenate(
        [
            np.asarray(y_train_clean, dtype=np.float64).reshape(-1),
            np.asarray(y_test_clean, dtype=np.float64).reshape(-1),
        ]
    )
    y_sel_noisy = apply_noise_tier(y_sel, tier, seed=seed)
    n_train = int(len(y_train_clean))
    y_train = y_sel_noisy[:n_train]
    y_test = y_sel_noisy[n_train:]

    est = estimator_factory()
    est_params = dict(est.get_params())
    est_params["random_state"] = int(seed)
    est = est.__class__(**est_params)

    try:
        est.fit(X_train, y_train)
        formula = bc.postprocess_formula(str(est.get_formula()))
    except Exception as exc:  # pragma: no cover - benchmark robustness
        return {
            "noise_type": tier.get("noise_type"),
            "noise_level": tier.get("noise_level"),
            "sample_weight_mode": "none",
            "raw_mse": None,
            "display_mse": None,
            "holdout_mse": None,
            "clean_test_mse": None,
            "clean_test_r2": None,
            "clean_full_mse": None,
            "formula_complexity": None,
            "false_confidence": None,
            "false_confidence_vs_clean": None,
            "seed_graphs_used": 0,
            "train_r2": None,
            "test_r2": None,
            "exact_match": False,
            "acceptable_clean": False,
            "discovered_formula": None,
            "true_formula": true_formula,
            "error": str(exc)[:200],
        }

    # Predictions / metrics.
    try:
        y_pred_train = est.predict(X_train)
        y_pred_test = est.predict(X_test)
    except Exception:
        y_pred_train = np.full_like(y_train, float(np.mean(y_train)))
        y_pred_test = np.full_like(y_test, float(np.mean(y_train)))

    train_r2 = _safe_r2(y_train, y_pred_train)
    test_r2 = _safe_r2(y_test, y_pred_test)
    raw_mse = (
        float(np.mean((y_pred_test - y_test) ** 2))
        if np.all(np.isfinite(y_pred_test))
        else None
    )
    display_mse = bc.evaluate_formula_mse_on_X(formula, X_test, y_test) if formula else None
    if display_mse is None or not math.isfinite(float(display_mse)):
        display_mse = raw_mse

    # Holdout currently mirrors the noisy test split; Phase 6 may add a
    # separate fidelity holdout. Clean columns below are the recovery signal.
    holdout_mse = raw_mse
    clean_test_mse = (
        float(np.mean((np.asarray(y_pred_test, dtype=np.float64) - np.asarray(y_test_clean, dtype=np.float64)) ** 2))
        if np.all(np.isfinite(y_pred_test))
        else None
    )
    clean_test_r2 = _safe_r2(y_test_clean, y_pred_test)

    # Exact-match check uses the *clean* target on the full selection so noise
    # injection does not corrupt the ground-truth equality test.
    try:
        y_pred_full = est.predict(X_sel)
        full_clean_mse = (
            float(np.mean((y_pred_full - y_sel) ** 2))
            if np.all(np.isfinite(y_pred_full))
            else float("inf")
        )
    except Exception:
        full_clean_mse = float("inf")
        y_pred_full = None
    exact_match = bool(full_clean_mse < 1e-6)
    acceptable_clean = bool(
        clean_test_r2 is not None
        and math.isfinite(float(clean_test_r2))
        and float(clean_test_r2) >= float(acceptable_r2)
    )

    return {
        "noise_type": tier.get("noise_type"),
        "noise_level": tier.get("noise_level"),
        "sample_weight_mode": _sample_weight_mode(est),
        "raw_mse": _to_json_float(raw_mse),
        "display_mse": _to_json_float(display_mse),
        "holdout_mse": _to_json_float(holdout_mse),
        "clean_test_mse": _to_json_float(clean_test_mse),
        "clean_test_r2": _to_json_float(clean_test_r2),
        "clean_full_mse": _to_json_float(full_clean_mse if math.isfinite(full_clean_mse) else None),
        "formula_complexity": _complexity(formula),
        "false_confidence": _false_confidence(train_r2=train_r2, test_r2=test_r2),
        "false_confidence_vs_clean": _false_confidence(
            train_r2=train_r2, test_r2=clean_test_r2
        ),
        "seed_graphs_used": _seed_graphs_used(est),
        "train_r2": _to_json_float(train_r2),
        "test_r2": _to_json_float(test_r2),
        "exact_match": exact_match,
        "acceptable_clean": acceptable_clean,
        "discovered_formula": formula,
        "true_formula": true_formula,
        "error": None,
    }



def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape or not np.all(np.isfinite(y_pred)):
        return None
    var = float(np.var(y_true))
    if var < 1e-15:
        return 1.0 if float(np.mean((y_pred - y_true) ** 2)) < 1e-15 else 0.0
    return float(1.0 - np.mean((y_pred - y_true) ** 2) / var)


def _to_json_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except Exception:
        return None
    return f if math.isfinite(f) else None


# ---------------------------------------------------------------------------
# Aggregation / reporting
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = (
    "noise_level",
    "noise_type",
    "sample_weight_mode",
    "raw_mse",
    "display_mse",
    "holdout_mse",
    "clean_test_mse",
    "clean_test_r2",
    "clean_full_mse",
    "formula_complexity",
    "false_confidence",
    "false_confidence_vs_clean",
    "seed_graphs_used",
    "exact_match",
    "acceptable_clean",
)


def assert_row_contract(rows: Sequence[Dict[str, Any]]) -> None:
    """Every row must expose the Phase 0 report columns."""
    for row in rows:
        missing = [c for c in REQUIRED_COLUMNS if c not in row]
        if missing:
            raise AssertionError(f"row missing columns {missing}: {row}")


def summarize_noise_protocol(
    rows: Sequence[Dict[str, Any]], *, acceptable_r2: float = 0.9
) -> Dict[str, Any]:
    """Per (problem, tier) rollup + clean-vs-noisy delta table."""
    assert_row_contract(rows)
    by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        by_key.setdefault((row["problem"], row["tier"]), []).append(row)

    cells: List[Dict[str, Any]] = []
    for (problem, tier), runs in by_key.items():
        valid = [r for r in runs if r.get("test_r2") is not None]
        r2s = [float(r["test_r2"]) for r in valid]
        exact_rate = float(np.mean([1.0 if r.get("exact_match") else 0.0 for r in runs]))
        fc_rate = float(np.mean(
            [1.0 if r.get("false_confidence") else 0.0 for r in runs if r.get("false_confidence") is not None]
        )) if any(r.get("false_confidence") is not None for r in runs) else None
        clean_r2s = [float(r["clean_test_r2"]) for r in runs if r.get("clean_test_r2") is not None]
        acceptable_rate = float(np.mean([1.0 if r.get("acceptable_clean") else 0.0 for r in runs]))
        cells.append({
            "problem": problem,
            "tier": tier,
            "n_runs": len(runs),
            "median_test_r2": float(np.median(r2s)) if r2s else None,
            "median_clean_test_r2": float(np.median(clean_r2s)) if clean_r2s else None,
            "exact_match_rate": exact_rate,
            "acceptable_clean_rate": acceptable_rate,
            "false_confidence_rate": fc_rate,
            "median_raw_mse": _median_key(valid, "raw_mse"),
            "median_display_mse": _median_key(valid, "display_mse"),
            "median_clean_test_mse": _median_key(runs, "clean_test_mse"),
            "median_formula_complexity": _median_key(valid, "formula_complexity"),
        })

    # Delta table: noisy tier vs clean, per problem.
    deltas: List[Dict[str, Any]] = []
    by_problem: Dict[str, List[Dict[str, Any]]] = {}
    for cell in cells:
        by_problem.setdefault(cell["problem"], []).append(cell)
    for problem, pcells in by_problem.items():
        clean = next((c for c in pcells if c["tier"] == "clean"), None)
        if clean is None or clean["median_test_r2"] is None:
            continue
        for cell in pcells:
            if cell["tier"] == "clean" or cell["median_test_r2"] is None:
                continue
            clean_med = clean.get("median_clean_test_r2")
            cell_med = cell.get("median_clean_test_r2")
            clean_r2_delta = (
                float(cell_med) - float(clean_med)
                if clean_med is not None and cell_med is not None
                else None
            )
            deltas.append({
                "problem": problem,
                "tier": cell["tier"],
                "r2_delta_vs_clean": float(cell["median_test_r2"]) - float(clean["median_test_r2"]),
                "clean_r2_delta_vs_clean_tier": clean_r2_delta,
                "exact_rate_delta_vs_clean": cell["exact_match_rate"] - clean["exact_match_rate"],
                "acceptable_clean_rate": cell.get("acceptable_clean_rate"),
                "false_confidence_rate": cell["false_confidence_rate"],
            })

    return {
        "cells": cells,
        "deltas_vs_clean": deltas,
        "n_rows": len(rows),
        "acceptable_r2": acceptable_r2,
    }


def _median_key(runs: Sequence[Dict[str, Any]], key: str) -> Optional[float]:
    vals = [float(r[key]) for r in runs if r.get(key) is not None]
    return float(np.median(vals)) if vals else None


def to_markdown(summary: Dict[str, Any]) -> str:
    """Render the per-tier median table as Markdown for the baseline report."""
    lines = [
        "| Problem | Tier | n | R2noisy | R2clean | Exact | Accept | FalseConf | RawMSE | CleanMSE | Complexity |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in summary["cells"]:
        def fmt(v, prec=4):
            return "-" if v is None else f"{float(v):.{prec}g}"
        lines.append(
            f"| {cell['problem']} | {cell['tier']} | {cell['n_runs']} | "
            f"{fmt(cell['median_test_r2'])} | {fmt(cell.get('median_clean_test_r2'))} | "
            f"{fmt(cell['exact_match_rate'], 2)} | {fmt(cell.get('acceptable_clean_rate'), 2)} | "
            f"{fmt(cell['false_confidence_rate'], 2)} | "
            f"{fmt(cell['median_raw_mse'])} | {fmt(cell.get('median_clean_test_mse'))} | "
            f"{fmt(cell['median_formula_complexity'], 3)} |"
        )
    return "\n".join(lines)


def write_report(rows, summary, output_dir) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "noise_protocol_rows.json"
    summary_path = output_dir / "noise_protocol_summary.json"
    md_path = output_dir / "noise_protocol_report.md"
    with rows_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=_json_default)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Phase 0 — Noise Protocol Baseline\n\n")
        f.write(to_markdown(summary))
        f.write("\n\n## Clean-vs-Noisy Deltas\n\n")
        f.write(
            "| Problem | Tier | R2noisy delta | R2clean delta | Exact delta | Accept | FalseConf |\n"
            "|---|---|---:|---:|---:|---:|---:|\n"
        )
        for d in summary["deltas_vs_clean"]:
            clean_d = d.get("clean_r2_delta_vs_clean_tier")
            clean_s = f"{clean_d:.4f}" if clean_d is not None else "-"
            acc = d.get("acceptable_clean_rate")
            acc_s = f"{acc:.2f}" if acc is not None else "-"
            f.write(
                f"| {d['problem']} | {d['tier']} | {d['r2_delta_vs_clean']:.4f} | "
                f"{clean_s} | {d['exact_rate_delta_vs_clean']:.2f} | {acc_s} | "
                f"{d['false_confidence_rate']} |\n"
            )
    return {"rows": rows_path, "summary": summary_path, "markdown": md_path}


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)



# ---------------------------------------------------------------------------
# CLI — multi-tier protocol baseline
# ---------------------------------------------------------------------------
DEFAULT_BASELINE_PROBLEMS = (
    "Poly-x2",
    "Poly-x3-x",
    "Nguyen-1",
    "Nguyen-5",
    "Keijzer-4",
    "Feynman-I.6.20a",
)


def _select_problems(names: Optional[Sequence[str]] = None):
    """Pick ground-truth problems by name (default: easy multi-tier baseline set).

    Built-in problems are available without torch. Additional names fall back to
    ``run_srbench_local.GROUND_TRUTH_PROBLEMS`` when that module can be imported.
    """
    catalogue = dict(BUILTIN_PROBLEMS)
    try:
        catalogue.update({p[0]: p for p in _rsl().GROUND_TRUTH_PROBLEMS})
    except Exception:
        # torch/Glassbox unavailable — built-ins only
        pass
    chosen = list(names) if names else list(DEFAULT_BASELINE_PROBLEMS)
    problems = []
    missing = []
    for name in chosen:
        if name in catalogue:
            problems.append(catalogue[name])
        else:
            missing.append(name)
    if missing:
        raise ValueError(f"unknown problem names: {missing}")
    return problems


def _default_estimator_factory(
    *,
    generations: int = 40,
    population_size: int = 60,
    timeout: float = 45.0,
    multi_start_runs: int = 1,
    allow_stub: bool = False,
):
    try:
        from glassbox.sr.sklearn_wrapper import GlassboxRegressor
    except Exception:
        if not allow_stub:
            raise
        GlassboxRegressor = None  # type: ignore

    if GlassboxRegressor is not None:
        def factory():
            return GlassboxRegressor(
                random_state=0,
                generations=int(generations),
                population_size=int(population_size),
                timeout=float(timeout),
                multi_start_runs=int(multi_start_runs),
                use_fast_path=True,
                use_guided_evolution=True,
                blackbox_mode=True,
            )
        return factory

    # Minimal mean predictor for tooling smoke when Glassbox/torch is absent.
    class _StubRegressor:
        def __init__(self, **kwargs):
            self.params = kwargs
            self.blackbox_diagnostics_ = {"sample_weight": {"provided": False}}
            self._mean = 0.0

        def get_params(self):
            return dict(self.params)

        def fit(self, X, y):
            self._mean = float(np.mean(y))
            return self

        def predict(self, X):
            return np.full(len(X), self._mean, dtype=np.float64)

        def get_formula(self):
            return f"{self._mean:.6g}"

    def factory():
        return _StubRegressor(
            random_state=0,
            generations=int(generations),
            population_size=int(population_size),
            timeout=float(timeout),
            multi_start_runs=int(multi_start_runs),
        )

    return factory


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the deterministic multi-tier noise protocol and write baseline artifacts."""
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(
        description=(
            "Phase 0 multi-tier noise protocol baseline. "
            "Reports clean recovery metrics separate from noisy fit metrics "
            "(see noise_handling_audit.md)."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(_REPO_ROOT / "results" / "noise_protocol_baseline"),
        help="Directory for noise_protocol_*.json / .md artifacts",
    )
    parser.add_argument(
        "--problems",
        type=str,
        default=",".join(DEFAULT_BASELINE_PROBLEMS),
        help="Comma-separated ground-truth problem names",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="11,23,47,89,137",
        help="Comma-separated integer seeds (Phase 0 default: 5 seeds)",
    )
    parser.add_argument(
        "--tiers",
        type=str,
        default="all",
        help="Comma-separated tier names, or 'all' for NOISE_TIERS",
    )
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--population-size", type=int, default=60)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny run: 2 problems, clean+gaussian_10pct, 1 seed, small budget",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.smoke:
        problem_names = list(DEFAULT_BASELINE_PROBLEMS[:2])
        seeds = [11]
        tier_names = ["clean", "gaussian_10pct"]
        generations, pop, timeout, n_samples = 15, 30, 20.0, 120
    else:
        problem_names = [p.strip() for p in args.problems.split(",") if p.strip()]
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        if args.tiers.strip().lower() == "all":
            tier_names = [t["name"] for t in NOISE_TIERS]
        else:
            tier_names = [t.strip() for t in args.tiers.split(",") if t.strip()]
        generations = int(args.generations)
        pop = int(args.population_size)
        timeout = float(args.timeout)
        n_samples = int(args.n_samples)

    tier_by_name = {t["name"]: t for t in NOISE_TIERS}
    tiers = []
    for name in tier_names:
        if name not in tier_by_name:
            raise SystemExit(f"unknown tier: {name}")
        tiers.append(tier_by_name[name])

    problems = _select_problems(problem_names)
    factory = _default_estimator_factory(
        generations=generations,
        population_size=pop,
        timeout=timeout,
        allow_stub=bool(args.smoke),
    )

    if not args.quiet:
        print(
            f"Noise protocol: {len(problems)} problems × {len(tiers)} tiers × "
            f"{len(seeds)} seeds  (n_samples={n_samples})"
        )
        print(f"Problems: {', '.join(p[0] for p in problems)}")
        print(f"Tiers:    {', '.join(t['name'] for t in tiers)}")
        print(f"Seeds:    {seeds}")

    rows = run_noise_protocol(
        factory,
        problems,
        tiers=tiers,
        seeds=seeds,
        n_samples=n_samples,
        verbose=not args.quiet,
    )
    summary = summarize_noise_protocol(rows)
    out_dir = Path(args.output_dir)
    paths = write_report(rows, summary, out_dir)

    # Also stamp a dated copy for baseline freeze.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stamped = out_dir / f"noise_protocol_{stamp}"
    stamped.mkdir(parents=True, exist_ok=True)
    for key, src in paths.items():
        dest = stamped / src.name
        dest.write_bytes(Path(src).read_bytes())

    if not args.quiet:
        print("\nWrote:")
        for key, p in paths.items():
            print(f"  {key}: {p}")
        print(f"  stamped: {stamped}")
        print("\n" + to_markdown(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
