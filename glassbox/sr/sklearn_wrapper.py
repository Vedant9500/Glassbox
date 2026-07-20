"""
Scikit-learn compatible wrapper for Glassbox Symbolic Regression.

Uses the FULL Glassbox pipeline:
  1. Classifier fast-path (instant for well-characterized curves)
  2. C++ guided evolution (beam search over multiple configs)
  3. Multipass formula simplification (float snapping + SymPy simplification)
"""

import sys
import os
import re
import math
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
try:
    from scipy.optimize import least_squares
except Exception:  # pragma: no cover - scipy is declared but keep import optional
    least_squares = None
from glassbox.sr.blackbox_preprocessor import (
    build_search_space_structure_seeds,
    formula_from_search_to_original_space,
    discover_blackbox_interactions,
    prepare_blackbox_search,
    remap_original_formula_to_reduced,
    remap_reduced_formula_to_original,
    state_to_dict,
)
from glassbox.sr.specialist_state import SpecialistVault
from glassbox.sr.specialist_state import compute_specialist_state
from glassbox.sr.specialist_state import propose_specialist_compositions
from glassbox.model_registry import DEFAULT_CURVE_CLASSIFIER_PATH


def _clamp_int(value, default, lo, hi):
    try:
        value = int(round(float(value)))
    except Exception:
        value = default
    return int(max(lo, min(hi, value)))


def _clamp_float(value, default, lo, hi):
    try:
        value = float(value)
    except Exception:
        value = default
    return float(max(lo, min(hi, value)))


def _finite_float(value, default=0.0):
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def _validate_sample_weight(sample_weight, n_samples):
    """Validate and normalise per-point weights (PhySO `y_weights` analogue).

    Returns a float64 array of length ``n_samples`` with non-negative finite
    entries and mean ~1. ``None`` or empty input resolves to None (uniform).
    Raises ``ValueError`` for length mismatch, non-finite, or all-zero weights.
    """
    if sample_weight is None:
        return None
    try:
        w = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError("sample_weight must be array-like") from exc
    if w.shape[0] != n_samples:
        raise ValueError(
            f"sample_weight has length {w.shape[0]}, expected {n_samples}"
        )
    if not np.all(np.isfinite(w)):
        raise ValueError("sample_weight must contain only finite values")
    if np.any(w < 0):
        raise ValueError("sample_weight must be non-negative")
    total = float(np.sum(w))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("sample_weight must have positive total weight")
    normalised = w / (total / float(n_samples))
    return normalised


def _weighted_mse(pred, target, sample_weight=None):
    """Mean squared error, optionally weighted (weights assumed mean-1).

    When ``sample_weight`` is provided it must match ``target`` length and have
    positive total weight; length/total mismatch raises ``ValueError`` instead
    of silently falling back to unweighted MSE.
    """
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if pred.shape != target.shape:
        return float("inf")
    resid = pred - target
    if not np.all(np.isfinite(resid)):
        return float("inf")
    if sample_weight is None:
        return float(np.mean(resid ** 2))
    w = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    if w.shape != target.shape:
        raise ValueError(
            f"sample_weight length {w.shape[0]} does not match target length {target.shape[0]}"
        )
    total = float(np.sum(w))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("sample_weight must have positive total weight")
    return float(np.sum(w * resid * resid) / total)


def _weighted_r2(pred, target, sample_weight=None):
    """Coefficient of determination, optionally weighted (weighted variance).

    When ``sample_weight`` is provided it must match ``target`` length; mismatch
    raises ``ValueError`` (no silent unweighted fallback).
    """
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if pred.shape != target.shape:
        return -float("inf")
    if sample_weight is None:
        var = float(np.var(target))
        if var < 1e-15:
            return 1.0 if float(np.mean((pred - target) ** 2)) < 1e-15 else 0.0
        return float(1.0 - np.mean((pred - target) ** 2) / var)
    w = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    if w.shape != target.shape:
        raise ValueError(
            f"sample_weight length {w.shape[0]} does not match target length {target.shape[0]}"
        )
    total = float(np.sum(w))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("sample_weight must have positive total weight")
    mean_t = float(np.sum(w * target) / total)
    var = float(np.sum(w * (target - mean_t) ** 2) / total)
    if var < 1e-15:
        return 1.0 if _weighted_mse(pred, target, w) < 1e-15 else 0.0
    return float(1.0 - _weighted_mse(pred, target, w) / var)


def _effective_sample_size(sample_weight):
    """Kish effective sample size; uniform weights -> n."""
    if sample_weight is None:
        return None
    w = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    total = float(np.sum(w))
    if total <= 0:
        return None
    return float((total * total) / float(np.sum(w * w)))


def _residual_lag1_autocorr(resid):
    """Lag-1 residual autocorrelation in [-1, 1]; 0 if undefined."""
    r = np.asarray(resid, dtype=np.float64).reshape(-1)
    r = r[np.isfinite(r)]
    if r.size < 8:
        return 0.0
    r = r - float(np.mean(r))
    den = float(np.dot(r, r))
    if den < 1e-15:
        return 0.0
    num = float(np.dot(r[:-1], r[1:]))
    ac = num / den
    if not np.isfinite(ac):
        return 0.0
    return float(np.clip(ac, -1.0, 1.0))


def _estimate_outlier_fraction(resid, sample_weight=None):
    """Fraction of points beyond 3 * MAD scale (robust outlier rate)."""
    r = np.asarray(resid, dtype=np.float64).reshape(-1)
    if r.size == 0 or not np.any(np.isfinite(r)):
        return 0.0
    scale = float(_mad_scale(r, sample_weight))
    if not np.isfinite(scale) or scale < 1e-12:
        return 0.0
    return float(np.mean(np.abs(r[np.isfinite(r)]) > 3.0 * scale))


def _signal_scale_outlier_fraction(resid, y=None, *, k: float = 2.5):
    """Fraction of residuals larger than ``k`` × robust signal scale.

    Complements MAD-relative outlier fraction: sparse protocol spikes (3%) can
    inflate residual std/MAD so MAD-relative rate under-counts; signal-scale
    rate still fires when spikes are large vs y.
    """
    r = np.asarray(resid, dtype=np.float64).reshape(-1)
    if r.size == 0 or not np.any(np.isfinite(r)):
        return 0.0
    finite = np.isfinite(r)
    r_f = r[finite]
    if y is not None:
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        if y_arr.shape == r.shape:
            y_f = y_arr[finite]
            y_med = float(np.median(y_f))
            y_mad = float(np.median(np.abs(y_f - y_med)))
            scale = 1.4826 * y_mad if y_mad > 1e-12 else float(np.std(y_f))
        else:
            scale = float(np.std(r_f))
    else:
        scale = float(np.std(r_f))
    if not np.isfinite(scale) or scale < 1e-12:
        return 0.0
    return float(np.mean(np.abs(r_f) > float(k) * scale))


def _residual_rms_ratio(resid, y):
    """RMS(residual) / max(std(y), eps), clipped to [0, 1].

    Captures white-noise *amplitude* that pure geometry (autocorr / MAD rate)
    misses. Used for noise_band sensitivity on gaussian_10pct tiers.
    """
    r = np.asarray(resid, dtype=np.float64).reshape(-1)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    if r.size == 0 or y_arr.size == 0 or r.shape != y_arr.shape:
        return 0.0
    finite = np.isfinite(r) & np.isfinite(y_arr)
    if not np.any(finite):
        return 0.0
    r_f = r[finite]
    y_f = y_arr[finite]
    y_std = float(np.std(y_f))
    if not np.isfinite(y_std) or y_std < 1e-12:
        y_std = float(np.mean(np.abs(y_f - np.median(y_f))))
    if not np.isfinite(y_std) or y_std < 1e-12:
        return 0.0
    rms = float(np.sqrt(np.mean(r_f ** 2)))
    if not np.isfinite(rms):
        return 0.0
    return float(np.clip(rms / y_std, 0.0, 1.0))


def _noise_band_from_diagnostics(diag):
    """Map runtime noise diagnostics to a coarse band used for threshold calibration.

    Bands align with protocol tiers conceptually: clean / low / medium / high.
    Not a claim about training-label noise — only residual/weight geometry at fit time.

    Phase E+: residual RMS ratio + signal-scale outlier fraction so white Gaussian
    and sparse protocol spikes no longer collapse to ``clean``.
    """
    if not isinstance(diag, dict):
        return "clean"
    outlier = float(diag.get("outlier_fraction") or 0.0)
    signal_out = float(diag.get("signal_outlier_fraction") or 0.0)
    # Prefer the stronger of MAD-relative vs signal-scale spike rates.
    outlier_eff = max(outlier, signal_out)
    gap = float(diag.get("validation_gap") or 0.0)
    ac = abs(float(diag.get("residual_autocorr") or 0.0))
    ess_ratio = diag.get("ess_ratio")
    ess_ratio = float(ess_ratio) if ess_ratio is not None and np.isfinite(float(ess_ratio)) else 1.0
    rms_ratio = float(diag.get("residual_rms_ratio") or 0.0)
    if not np.isfinite(rms_ratio):
        rms_ratio = 0.0
    rms_ratio = float(np.clip(rms_ratio, 0.0, 1.0))
    score = (
        1.2 * min(max(outlier_eff, 0.0), 1.0)
        + 0.8 * min(max(gap, 0.0), 1.0)
        + 0.4 * min(max(ac, 0.0), 1.0)
        + 0.6 * min(max(1.0 - ess_ratio, 0.0), 1.0)
        # White-noise amplitude channel (gaussian_10pct ≈ 0.1 → +0.15).
        + 1.5 * rms_ratio
    )
    if score < 0.15:
        return "clean"
    if score < 0.40:
        return "low"
    if score < 0.75:
        return "medium"
    return "high"


# Calibrated acceptance/shrink floors by residual noise band (Phase 7).
# High noise: do NOT shrink diversity just because noisy candidate MSE is low —
# require stronger clean-holdout evidence before accepting / shrinking search.
_NOISE_BAND_THRESHOLDS = {
    "clean": {"prediction_uncertain_entropy": 0.85, "candidate_acceptance_r2": 0.985, "candidate_shrink_r2": 0.95},
    "low": {"prediction_uncertain_entropy": 0.80, "candidate_acceptance_r2": 0.975, "candidate_shrink_r2": 0.93},
    "medium": {"prediction_uncertain_entropy": 0.72, "candidate_acceptance_r2": 0.96, "candidate_shrink_r2": 0.90},
    "high": {"prediction_uncertain_entropy": 0.65, "candidate_acceptance_r2": 0.94, "candidate_shrink_r2": 0.86},
}


def _soft_mad_sample_weights(values, *, floor: float = 0.05, cap: float = 1.0):
    """Huber-like soft weights from MAD of ``values`` (target or residual).

    Points beyond ~2.5 MAD get weight decaying as 2.5*scale/|r|, floored at
    ``floor``. Returns None when scale is undefined (no downweighting signal).
    """
    r = np.asarray(values, dtype=np.float64).reshape(-1)
    if r.size < 8 or not np.any(np.isfinite(r)):
        return None
    finite = np.isfinite(r)
    r_f = r[finite]
    med = float(np.median(r_f))
    centered = r_f - med
    mad = float(np.median(np.abs(centered)))
    scale = 1.4826 * mad if mad > 1e-15 else float(np.std(centered))
    if not np.isfinite(scale) or scale < 1e-12:
        return None
    thr = 2.5 * scale
    w = np.ones(r.shape[0], dtype=np.float64)
    abs_c = np.abs(r - med)
    heavy = finite & (abs_c > thr)
    if not np.any(heavy):
        # Still return uniform-ish weights only when some mass is heavy-tailed.
        out_frac = float(np.mean(np.abs(centered) > 3.0 * scale))
        if out_frac < 0.02:
            return None
    w[heavy] = np.clip(thr / np.maximum(abs_c[heavy], 1e-12), float(floor), float(cap))
    mean_w = float(np.mean(w[finite]))
    if mean_w > 1e-12:
        w = w / mean_w
    return w


def _auto_residual_soft_weights(X, y, *, floor: float = 0.05, cap: float = 1.0):
    """Soft MAD weights from residuals of a cheap structure fit (Phase 3 1D path).

    Raw-target soft weights fire on clean nonlinear y (e.g. polynomials) because
    the *level* distribution is heavy-tailed even when residuals are clean.

    Fit residual probes (linear, quadratic/cubic for 1D, median) and keep the
    residual with the smallest MAD scale, then soft-weight *that* residual only.
    Returns ``(weights_or_None, outlier_fraction)``.
    """
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    n = int(y_arr.shape[0])
    if n < 8 or not np.any(np.isfinite(y_arr)):
        return None, 0.0
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    if X_arr.shape[0] != n:
        r = y_arr - float(np.median(y_arr[np.isfinite(y_arr)]))
        soft = _soft_mad_sample_weights(r, floor=floor, cap=cap)
        return soft, float(_estimate_outlier_fraction(r))

    def _resid_scale(r):
        r = np.asarray(r, dtype=np.float64).reshape(-1)
        rf = r[np.isfinite(r)]
        if rf.size < 4:
            return float("inf")
        mad = float(np.median(np.abs(rf - np.median(rf))))
        if mad > 1e-15:
            return 1.4826 * mad
        s = float(np.std(rf))
        return s if np.isfinite(s) and s > 0.0 else 0.0

    residuals = []
    try:
        x0 = X_arr[:, 0]
        A = np.column_stack([x0, np.ones(n, dtype=np.float64)])
        coef, _, _, _ = np.linalg.lstsq(A, y_arr, rcond=None)
        residuals.append(y_arr - A @ coef)
        if X_arr.shape[1] == 1:
            A2 = np.column_stack([x0 ** 2, x0, np.ones(n, dtype=np.float64)])
            coef2, _, _, _ = np.linalg.lstsq(A2, y_arr, rcond=None)
            residuals.append(y_arr - A2 @ coef2)
            A3 = np.column_stack(
                [x0 ** 3, x0 ** 2, x0, np.ones(n, dtype=np.float64)]
            )
            coef3, _, _, _ = np.linalg.lstsq(A3, y_arr, rcond=None)
            residuals.append(y_arr - A3 @ coef3)
    except Exception:
        pass
    y_med = float(np.median(y_arr[np.isfinite(y_arr)]))
    residuals.append(y_arr - y_med)

    if not residuals:
        return None, 0.0
    best_r = min(residuals, key=_resid_scale)
    out_frac = float(_estimate_outlier_fraction(best_r))
    soft = _soft_mad_sample_weights(best_r, floor=floor, cap=cap)
    return soft, out_frac


def _estimate_diffuse_noise_ratio(X, y):
    """Cheap residual noise ratio for diffuse noise without sparse outliers.

    Fits the best residual among simple structure probes (poly, trig, exp,
    rational, multi-linear) and returns ``scale(residual) / scale(y)``.

    Clean structured non-polynomials (sin/exp/rational) must not look like
    heavy noise — poly-only probes false-positive auto-Huber (N2).
    """
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    n = int(y_arr.shape[0])
    if n < 8 or not np.any(np.isfinite(y_arr)):
        return 0.0, 0.0
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    finite = np.isfinite(y_arr)
    if X_arr.shape[0] == n:
        finite = finite & np.all(np.isfinite(X_arr), axis=1)
    y_f = y_arr[finite]
    y_scale = float(np.std(y_f)) if y_f.size else 0.0
    if not np.isfinite(y_scale) or y_scale < 1e-12:
        y_scale = max(float(np.mean(np.abs(y_f))) if y_f.size else 1.0, 1e-12)

    def _resid_scale(r):
        r = np.asarray(r, dtype=np.float64).reshape(-1)
        rf = r[np.isfinite(r)]
        if rf.size < 4:
            return float("inf")
        mad = float(np.median(np.abs(rf - np.median(rf))))
        if mad > 1e-15:
            return 1.4826 * mad
        s = float(np.std(rf))
        return s if np.isfinite(s) else float("inf")

    def _append_lstsq(residuals, cols, y_use):
        try:
            A = np.column_stack([np.asarray(c, dtype=np.float64).reshape(-1) for c in cols])
            if A.shape[0] != y_use.shape[0] or A.shape[0] < A.shape[1] + 2:
                return
            if not np.all(np.isfinite(A)):
                return
            coef, _, _, _ = np.linalg.lstsq(A, y_use, rcond=None)
            pred = A @ coef
            residuals.append(y_use - pred)
        except Exception:
            return

    residuals = []
    y_med = float(np.median(y_f)) if y_f.size else 0.0
    residuals.append(y_arr - y_med)

    if X_arr.shape[0] == n and int(np.sum(finite)) >= 8:
        y_use = y_arr[finite]
        X_use = X_arr[finite]
        x0 = X_use[:, 0]
        ones = np.ones(x0.shape[0], dtype=np.float64)
        # Polynomial family
        _append_lstsq(residuals, [ones, x0], y_use)
        _append_lstsq(residuals, [ones, x0, x0 ** 2], y_use)
        _append_lstsq(residuals, [ones, x0, x0 ** 2, x0 ** 3], y_use)
        # Trig family (clean sin/cos must not trigger diffuse Huber)
        _append_lstsq(residuals, [ones, np.sin(x0), np.cos(x0)], y_use)
        _append_lstsq(residuals, [ones, np.sin(2.0 * x0), np.cos(2.0 * x0)], y_use)
        _append_lstsq(
            residuals,
            [ones, np.sin(2.0 * np.pi * x0), np.cos(2.0 * np.pi * x0)],
            y_use,
        )
        # Exp / Gaussian bump / rational
        x_clip = np.clip(x0, -20.0, 20.0)
        _append_lstsq(residuals, [ones, np.exp(x_clip)], y_use)
        _append_lstsq(residuals, [ones, np.exp(-x0 ** 2)], y_use)
        _append_lstsq(residuals, [ones, 1.0 / (1.0 + x0 ** 2)], y_use)
        # Multi-feature linear probe
        if X_use.shape[1] > 1:
            cols = [ones] + [X_use[:, j] for j in range(X_use.shape[1])]
            _append_lstsq(residuals, cols, y_use)

    if not residuals:
        return 0.0, 0.0
    best_r = min(residuals, key=_resid_scale)
    scale = _resid_scale(best_r)
    best_r = np.asarray(best_r, dtype=np.float64).reshape(-1)
    rf = best_r[np.isfinite(best_r)]
    rms = float(np.sqrt(np.mean(rf ** 2))) if rf.size else 0.0
    # Prefer the larger of MAD-scale and RMS so smooth pink-like residuals still register.
    if np.isfinite(rms):
        scale = max(scale if np.isfinite(scale) else 0.0, rms)
    if not np.isfinite(scale):
        return 0.0, float(_estimate_outlier_fraction(best_r))
    ratio = float(scale / y_scale)
    if not np.isfinite(ratio):
        ratio = 0.0
    return max(0.0, ratio), float(_estimate_outlier_fraction(best_r))




def _slice_sample_weight(sample_weight, indices=None, n_targets=None):
    """Slice or validate per-point weights for a subset of rows.

    ``None`` stays ``None``. If ``indices`` is given, returns ``weight[indices]``.
    If only ``n_targets`` is given, requires ``len(weight) == n_targets``.
    Raises ``ValueError`` on length mismatches so callers cannot silently
    ignore weights on holdout/subset scores.
    """
    if sample_weight is None:
        return None
    w = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    if indices is not None:
        idx = np.asarray(indices, dtype=int).reshape(-1)
        if idx.size and (int(np.min(idx)) < 0 or int(np.max(idx)) >= w.shape[0]):
            raise ValueError(
                f"sample_weight index out of range for length {w.shape[0]}"
            )
        return w[idx]
    if n_targets is not None and w.shape[0] != int(n_targets):
        raise ValueError(
            f"sample_weight length {w.shape[0]} does not match n_targets {n_targets}"
        )
    return w


# Phase 4: robust search losses (display MSE stays plain unweighted MSE).
_VALID_LOSS_MODES = ("mse", "huber", "trimmed_mse", "student_t")

# Phase 5: dimensional analysis (optional; omit units for tabular ML).
_VALID_UNIT_MODES = ("off", "soft", "hard")
_UNIT_UNARY_DIMLESS = frozenset({
    "sin", "cos", "tan", "exp", "log", "ln",
    "asin", "acos", "atan", "sinh", "cosh", "tanh",
})
_UNIT_UNARY_PRESERVE = frozenset({"abs", "sign", "neg", "negation"})
_UNIT_HARD_PENALTY = 1e6
_UNIT_MATCH_TOL = 1e-6


def _validate_loss_mode(loss_mode):
    mode = str(loss_mode or "mse").strip().lower()
    if mode not in _VALID_LOSS_MODES:
        raise ValueError(
            f"loss_mode must be one of {_VALID_LOSS_MODES}, got {loss_mode!r}"
        )
    return mode


def _validate_unit_mode(unit_mode):
    mode = str(unit_mode or "off").strip().lower()
    if mode in ("none", "disabled", "false", "0"):
        mode = "off"
    if mode not in _VALID_UNIT_MODES:
        raise ValueError(
            f"unit_mode must be one of {_VALID_UNIT_MODES}, got {unit_mode!r}"
        )
    return mode


def _as_unit_vector(vec, *, name="units", n_dims=None):
    try:
        arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    except Exception as exc:
        raise ValueError(f"{name} must be array-like of floats") from exc
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    if n_dims is not None and arr.size != int(n_dims):
        raise ValueError(
            f"{name} has length {arr.size}, expected {n_dims} dimensional exponents"
        )
    return arr.tolist()


def _validate_physics_units(input_units, output_units, n_features):
    """Validate PhySO-style unit exponent vectors.

    ``input_units``: list of length ``n_features``, each a vector of base-dimension
    exponents (e.g. SI [M, L, T, ...] or custom). ``output_units``: one vector of
    the same length. Empty/None → inactive (tabular default).
    """
    if input_units is None and output_units is None:
        return None, None
    if input_units is None or output_units is None:
        raise ValueError(
            "input_units and output_units must both be provided, or both omitted"
        )
    n_features = int(n_features)
    if n_features < 1:
        raise ValueError("n_features must be >= 1 when units are provided")

    try:
        rows = list(input_units)
    except TypeError as exc:
        raise ValueError("input_units must be a sequence of unit vectors") from exc
    if len(rows) != n_features:
        raise ValueError(
            f"input_units has {len(rows)} feature rows, expected {n_features}"
        )
    parsed = []
    n_dims = None
    for i, row in enumerate(rows):
        vec = _as_unit_vector(row, name=f"input_units[{i}]", n_dims=n_dims)
        if n_dims is None:
            n_dims = len(vec)
        parsed.append(vec)
    out = _as_unit_vector(output_units, name="output_units", n_dims=n_dims)
    # Reject mixing zero-dim with multi-dim inconsistently (already length-checked).
    return parsed, out


def _units_equal(a, b, tol=_UNIT_MATCH_TOL):
    if a is None or b is None:
        return False
    if len(a) != len(b):
        return False
    return all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b))


def _units_zero(u, tol=_UNIT_MATCH_TOL):
    return u is not None and all(abs(float(x)) <= tol for x in u)


def _units_add(a, b):
    return [float(x) + float(y) for x, y in zip(a, b)]


def _units_sub(a, b):
    return [float(x) - float(y) for x, y in zip(a, b)]


def _units_scale(a, s):
    return [float(x) * float(s) for x in a]


def _infer_formula_units(formula, input_units, output_units=None):
    """Propagate dimensional exponents for a display formula string.

    Returns dict:
      units: list[float] | None
      penalty: float (squared mismatches; higher = more unphysical)
      ok: bool (safe inference succeeded)
      reason: str
    """
    text = str(formula or "").strip()
    if not text or not input_units:
        return {"units": None, "penalty": 0.0, "ok": False, "reason": "no_formula_or_units"}

    n_dims = len(input_units[0])
    zero = [0.0] * n_dims
    penalty = [0.0]

    # Tokenize: numbers, names, operators, punctuation.
    token_re = re.compile(
        r"\s*("
        r"[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
        r"|x\d+"
        r"|[A-Za-z_][A-Za-z0-9_]*"
        r"|[\+\-\*/\^]|\*\*"
        r"|[(),|]"
        r")"
    )
    tokens = []
    pos = 0
    s = text.replace("·", "*").replace("−", "-")
    while pos < len(s):
        m = token_re.match(s, pos)
        if not m:
            return {"units": None, "penalty": 0.0, "ok": False, "reason": "parse_error"}
        tok = m.group(1)
        if tok.strip():
            tokens.append(tok)
        pos = m.end()
    if not tokens:
        return {"units": None, "penalty": 0.0, "ok": False, "reason": "empty_tokens"}

    i = [0]

    def peek():
        return tokens[i[0]] if i[0] < len(tokens) else None

    def get():
        tok = peek()
        i[0] += 1
        return tok

    def parse_expr():
        return parse_add()

    def parse_add():
        left = parse_mul()
        while peek() in ("+", "-"):
            op = get()
            right = parse_mul()
            if left is None or right is None:
                left = None
                continue
            if not _units_equal(left, right):
                for d in range(n_dims):
                    diff = left[d] - right[d]
                    penalty[0] += diff * diff
            # Result units: prefer left if mismatch (C++ style)
            left = list(left)
        return left

    def parse_mul():
        left = parse_pow()
        while peek() in ("*", "/"):
            op = get()
            right = parse_pow()
            if left is None or right is None:
                left = None
                continue
            left = _units_add(left, right) if op == "*" else _units_sub(left, right)
        return left

    def parse_pow():
        base = parse_unary()
        if peek() not in ("^", "**"):
            return base
        get()
        # Constant exponent only (unit-safe scale).
        sign = 1.0
        if peek() == "+":
            get()
        elif peek() == "-":
            get()
            sign = -1.0
        tok = peek()
        if tok is None:
            return None
        try:
            float(tok)
            get()
            exp_val = sign * float(tok)
        except Exception:
            if tok == "(":
                get()
                inner = parse_expr()
                if peek() == ")":
                    get()
                if inner is not None and _units_zero(inner):
                    return list(zero) if base is not None and _units_zero(base) else None
                return None
            parse_unary()
            return None
        if base is None:
            return None
        return _units_scale(base, exp_val)

    def parse_unary():
        if peek() in ("+", "-"):
            get()
            return parse_unary()
        return parse_primary()

    def parse_primary():
        tok = peek()
        if tok is None:
            return None
        if tok == "(":
            get()
            node = parse_expr()
            if peek() == ")":
                get()
            return node
        if tok == "|":
            # abs |expr|
            get()
            node = parse_expr()
            if peek() == "|":
                get()
            return node
        # number
        try:
            float(tok)
            get()
            return list(zero)
        except Exception:
            pass
        # feature xN
        m = re.fullmatch(r"x(\d+)", tok)
        if m:
            get()
            idx = int(m.group(1))
            if 0 <= idx < len(input_units):
                return list(input_units[idx])
            return None
        # function call
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", tok) and i[0] + 1 < len(tokens) and tokens[i[0] + 1] == "(":
            name = get().lower()
            get()  # (
            arg = parse_expr()
            if peek() == ")":
                get()
            if name in _UNIT_UNARY_DIMLESS:
                if arg is not None and not _units_zero(arg):
                    for d in range(n_dims):
                        penalty[0] += arg[d] * arg[d]
                return list(zero)
            if name in _UNIT_UNARY_PRESERVE:
                return list(arg) if arg is not None else None
            if name in ("square",):
                return _units_scale(arg, 2.0) if arg is not None else None
            if name in ("sqrt", "sqr"):
                return _units_scale(arg, 0.5) if arg is not None else None
            # Unknown function: only safe if arg dimensionless → dimensionless result
            if arg is not None and _units_zero(arg):
                return list(zero)
            return None
        # bare identifier (pi, e, etc.) → dimensionless
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", tok):
            get()
            return list(zero)
        return None

    try:
        units = parse_expr()
        if peek() is not None:
            return {"units": None, "penalty": 0.0, "ok": False, "reason": "trailing_tokens"}
    except Exception:
        return {"units": None, "penalty": 0.0, "ok": False, "reason": "exception"}

    if units is None:
        return {"units": None, "penalty": float(penalty[0]), "ok": False, "reason": "unsafe_inference"}

    if output_units is not None:
        for d in range(min(n_dims, len(output_units))):
            diff = units[d] - float(output_units[d])
            penalty[0] += diff * diff

    return {
        "units": units,
        "penalty": float(penalty[0]),
        "ok": True,
        "reason": "ok",
    }


def _formula_unit_compatible(formula, input_units, output_units, unit_mode="soft", *, max_penalty=1e-6):
    """Return (compatible, info) for candidate filtering.

    hard: reject when inference ok and penalty > max_penalty, or when
    inference fails on clearly structured formulas with trig/exp of dimensioned args
    that we *did* score. Soft: never reject; only report penalty.
    off: always compatible.
    """
    mode = _validate_unit_mode(unit_mode)
    if mode == "off" or not input_units:
        return True, {"penalty": 0.0, "ok": False, "reason": "units_inactive"}
    info = _infer_formula_units(formula, input_units, output_units)
    if mode == "soft":
        return True, info
    # hard
    if not info.get("ok"):
        # Do not reject when units cannot be inferred safely.
        return True, info
    pen = float(info.get("penalty") or 0.0)
    if pen > float(max_penalty):
        return False, info
    return True, info


def _mad_scale(resid, sample_weight=None):
    """Robust residual scale via MAD (≈ σ for Gaussian). Floor avoids zero delta."""
    r = np.asarray(resid, dtype=np.float64).reshape(-1)
    if r.size == 0 or not np.any(np.isfinite(r)):
        return 1.0
    r = r[np.isfinite(r)]
    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        if w.shape == r.shape and float(np.sum(w)) > 0:
            # Weighted median via sort + cumulative weights
            order = np.argsort(r)
            rs, ws = r[order], w[order]
            c = np.cumsum(ws)
            mid = 0.5 * float(c[-1])
            med = float(rs[int(np.searchsorted(c, mid))])
            abs_dev = np.abs(rs - med)
            order2 = np.argsort(abs_dev)
            c2 = np.cumsum(ws[order2])
            mad = float(abs_dev[order2][int(np.searchsorted(c2, mid))])
        else:
            med = float(np.median(r))
            mad = float(np.median(np.abs(r - med)))
    else:
        med = float(np.median(r))
        mad = float(np.median(np.abs(r - med)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-12:
        # Fall back to std / absolute level
        s = float(np.std(r))
        if np.isfinite(s) and s > 1e-12:
            return s
        a = float(np.mean(np.abs(r)))
        return a if a > 1e-12 else 1.0
    return scale


def _robust_loss(pred, target, loss_mode="mse", sample_weight=None, *, delta=None, trim_fraction=0.1):
    """Search-objective loss; MSE when mode is mse. Not for display metrics.

    Modes:
      - mse: weighted/unweighted mean squared residual
      - huber: smooth L1 outside ``delta`` (default = MAD scale)
      - trimmed_mse: drop largest ``trim_fraction`` of squared residuals
      - student_t: log(1 + (r/s)^2) with s = MAD scale (heavy-tail)
    """
    mode = _validate_loss_mode(loss_mode)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if pred.shape != target.shape:
        return float("inf")
    resid = pred - target
    if not np.all(np.isfinite(resid)):
        return float("inf")
    w = None
    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        if w.shape != target.shape:
            raise ValueError(
                f"sample_weight length {w.shape[0]} does not match target length {target.shape[0]}"
            )
        total = float(np.sum(w))
        if not np.isfinite(total) or total <= 0:
            raise ValueError("sample_weight must have positive total weight")
    if mode == "mse":
        return _weighted_mse(pred, target, w)

    abs_r = np.abs(resid)
    if mode == "huber":
        scale = float(delta) if delta is not None and float(delta) > 0 else _mad_scale(resid, w)
        # Standard Huber: 0.5 r^2 if |r|<=d else d*(|r|-0.5 d)
        d = max(float(scale), 1e-12)
        quad = 0.5 * resid * resid
        lin = d * (abs_r - 0.5 * d)
        loss = np.where(abs_r <= d, quad, lin)
        if w is None:
            return float(np.mean(loss))
        return float(np.sum(w * loss) / float(np.sum(w)))

    if mode == "trimmed_mse":
        sq = resid * resid
        frac = float(trim_fraction)
        frac = min(max(frac, 0.0), 0.45)
        n = int(sq.size)
        keep = max(1, int(round(n * (1.0 - frac))))
        if w is None:
            order = np.argsort(sq)
            return float(np.mean(sq[order[:keep]]))
        # Soft trim: zero the largest residuals by weight mass
        order = np.argsort(sq)
        kept_idx = order[:keep]
        ww = w[kept_idx]
        total = float(np.sum(ww))
        if total <= 0:
            return float("inf")
        return float(np.sum(ww * sq[kept_idx]) / total)

    # student_t
    s = float(delta) if delta is not None and float(delta) > 0 else _mad_scale(resid, w)
    s = max(s, 1e-12)
    z2 = (resid / s) ** 2
    loss = np.log1p(z2)
    if w is None:
        return float(np.mean(loss))
    return float(np.sum(w * loss) / float(np.sum(w)))


# Path setup
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPTS_DIR = _REPO_ROOT / 'scripts'
_CPP_DIR = Path(__file__).resolve().parent / 'cpp'

for p in [str(_REPO_ROOT), str(_SCRIPTS_DIR), str(_CPP_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import _core  # type: ignore
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False

from glassbox.evolution import detect_dominant_frequency


class GlassboxRegressor(BaseEstimator, RegressorMixin):
    """
    Scikit-learn compatible wrapper for Glassbox Symbolic Regression.

    Uses the full pipeline: classifier fast-path → C++ evolution → formula simplification.

    Optional physics units (Phase 5 / PhySO-style):
      input_units: list of length n_features, each a base-dimension exponent vector
        e.g. SI-like [M, L, T] → length variable [[0,1,0]], time [[0,0,1]].
      output_units: one exponent vector for the target (same length as each input row).
      unit_mode: 'off' (default tabular), 'soft' (penalty/rank), 'hard' (filter).
      dim_penalty_weight: C++ fitness penalty scale when units active.
    Omit units for normal ML/tabular use — no dimensional penalties applied.
    """

    def __init__(
        self,
        population_size=100,
        generations=1000,
        early_stop_mse=1e-6,
        random_state=None,
        p_min=-2.0,
        p_max=3.0,
        use_nsga2=False,
        num_islands=8,
        migration_interval=25,
        migration_size=2,
        arithmetic_temperature=5.0,
        # Pipeline control
        use_fast_path=True,
        use_guided_evolution=True,
        use_simplification=True,
        classifier_path=DEFAULT_CURVE_CLASSIFIER_PATH,
        simplification_int_tol=0.05,
        simplification_zero_tol=1e-3,
        max_power=6,
        timeout=120,
        evolution_skip_r2=0.999,
        multi_start_runs=3,
        adaptive_compute_budget=True,
        min_compute_budget=10,
        max_compute_budget=300,
        cv_skip_guard_enabled=True,
        cv_skip_guard_folds=3,
        cv_skip_guard_min_fold_r2=0.98,
        cv_skip_guard_max_r2_std=0.03,
        cv_skip_guard_min_samples=45,
        use_universal_proposer="auto",
        universal_proposer_path="models/universal_proposer_multi.pt",
        universal_proposer_shadow_mode="auto",
        universal_proposer_log_routing=True,
        universal_proposer_top_k=5,
        blackbox_mode="auto",
        blackbox_max_features=6,
        blackbox_feature_selection=True,
        blackbox_standardize=True,
        blackbox_interaction_search=True,
        blackbox_min_features_to_select=5,
        # Phase C: gated noise-robust blackbox search (auto|True|False).
        # auto = soft residual/y weights + optional huber when blackbox is noisy
        # or selection-uncertain; never overrides user-provided sample_weight.
        blackbox_noise_robust="auto",
        enable_specialist_screening_diagnostics=True,
        enable_specialist_composition_screening=True,
        enable_residual_stage=True,
        max_boosting_stages=3,
        boosting_learning_rates=None,
        residual_mini_search_max_candidates=64,
        residual_mini_search_refine_top_k=6,
        enable_specialist_vault_memory=True,
        specialist_vault_size=8,
        enable_inception_reuse=True,
        max_inception_rounds=2,
        max_frozen_subexpressions=3,
        device=None,
        exact_match_backend="auto",
        exact_match_min_gpu_work=250_000,
        exact_match_max_combos=50_000,
        skip_evolution_if_bloated=False,
        bloat_term_threshold=20,
        loss_mode="mse",
        huber_delta=None,
        trim_fraction=0.1,
        # Phase 5: optional dimensional analysis (PhySO-style). Omit for tabular ML.
        input_units=None,
        output_units=None,
        dim_penalty_weight=0.1,
        unit_mode="off",
    ):
        self.population_size = population_size
        self.generations = generations
        self.early_stop_mse = early_stop_mse
        self.random_state = random_state
        self.loss_mode = _validate_loss_mode(loss_mode)
        self.huber_delta = huber_delta
        self.trim_fraction = float(trim_fraction)
        self.input_units = input_units
        self.output_units = output_units
        self.dim_penalty_weight = float(dim_penalty_weight)
        self.unit_mode = _validate_unit_mode(unit_mode)
        self.p_min = p_min
        self.p_max = p_max
        self.use_nsga2 = use_nsga2
        self.num_islands = num_islands
        self.migration_interval = migration_interval
        self.migration_size = migration_size
        self.arithmetic_temperature = arithmetic_temperature
        self.use_fast_path = use_fast_path
        self.use_guided_evolution = use_guided_evolution
        self.use_simplification = use_simplification
        self.classifier_path = classifier_path
        self.simplification_int_tol = simplification_int_tol
        self.simplification_zero_tol = simplification_zero_tol
        self.max_power = max_power
        self.timeout = timeout
        self.evolution_skip_r2 = evolution_skip_r2
        self.multi_start_runs = multi_start_runs
        self.adaptive_compute_budget = adaptive_compute_budget
        self.min_compute_budget = min_compute_budget
        self.max_compute_budget = max_compute_budget
        self.cv_skip_guard_enabled = cv_skip_guard_enabled
        self.cv_skip_guard_folds = cv_skip_guard_folds
        self.cv_skip_guard_min_fold_r2 = cv_skip_guard_min_fold_r2
        self.cv_skip_guard_max_r2_std = cv_skip_guard_max_r2_std
        import os
        self.cv_skip_guard_min_samples = cv_skip_guard_min_samples
        
        # Rollback switch via environment variable
        legacy_mode = os.environ.get("GLASSBOX_USE_LEGACY_FASTPATH", "0") != "0"
        
        self.use_universal_proposer = not legacy_mode if use_universal_proposer == "auto" else use_universal_proposer
        self.universal_proposer_path = universal_proposer_path
        self.universal_proposer_shadow_mode = legacy_mode if universal_proposer_shadow_mode == "auto" else universal_proposer_shadow_mode
        self.universal_proposer_log_routing = universal_proposer_log_routing
        self.universal_proposer_top_k = universal_proposer_top_k
        self.blackbox_mode = blackbox_mode
        self.blackbox_max_features = blackbox_max_features
        self.blackbox_feature_selection = blackbox_feature_selection
        self.blackbox_standardize = blackbox_standardize
        self.blackbox_interaction_search = blackbox_interaction_search
        self.blackbox_min_features_to_select = blackbox_min_features_to_select
        mode = blackbox_noise_robust
        if isinstance(mode, bool):
            self.blackbox_noise_robust = mode
        else:
            text = str(mode or "auto").strip().lower()
            if text in ("1", "true", "yes", "on"):
                self.blackbox_noise_robust = True
            elif text in ("0", "false", "no", "off"):
                self.blackbox_noise_robust = False
            else:
                self.blackbox_noise_robust = "auto"
        self.enable_specialist_screening_diagnostics = enable_specialist_screening_diagnostics
        self.enable_specialist_composition_screening = enable_specialist_composition_screening
        self.enable_residual_stage = enable_residual_stage
        self.max_boosting_stages = max(0, int(max_boosting_stages))
        self.boosting_learning_rates = list(boosting_learning_rates or [0.5, 0.8, 1.0])
        self.residual_mini_search_max_candidates = max(1, int(residual_mini_search_max_candidates))
        self.residual_mini_search_refine_top_k = max(1, int(residual_mini_search_refine_top_k))
        self.enable_specialist_vault_memory = enable_specialist_vault_memory
        self.specialist_vault_size = max(0, int(specialist_vault_size))
        self.enable_inception_reuse = enable_inception_reuse
        self.max_inception_rounds = max(0, int(max_inception_rounds))
        self.max_frozen_subexpressions = max(0, int(max_frozen_subexpressions))
        self.device = device
        self.exact_match_backend = exact_match_backend
        self.exact_match_min_gpu_work = exact_match_min_gpu_work
        self.exact_match_max_combos = exact_match_max_combos
        self.skip_evolution_if_bloated = skip_evolution_if_bloated
        self.bloat_term_threshold = bloat_term_threshold

        self.input_units_ = None
        self.output_units_ = None
        self.units_active_ = False
        self.physics_constrained_ = False

        self._universal_proposer_model = None
        self.specialist_state_ = None
        self.specialist_vault_ = SpecialistVault(max_entries=self.specialist_vault_size)
        self.specialist_track_ = "incumbent path"
        self.has_composed_seeds_ = False
        self.composition_candidates_accepted_ = False
        self.composition_candidate_count_ = 0
        self.composition_seeded_evolution_ = False
        self.composition_won_final_selection_ = False
        self.composition_improved_mse_ = False
        self.phase_timings_ = {}
        self.formula_eval_count_ = 0
        self.formula_eval_cache_hits_ = 0
        self._formula_eval_cache_ = {}
        self.fast_path_exact_skip_ = False
        self.fast_path_exact_match_diagnostics_ = {}
        self.boosting_stages_ = []
        self.boosting_attempted_ = False
        self.boosting_improved_ = False
        self.boosting_diagnostics_ = {}
        self.inception_rounds_ = []
        self.inception_diagnostics_ = {}

    def _add_phase_time(self, phase: str, elapsed: float) -> None:
        try:
            value = float(elapsed)
        except Exception:
            return
        if value < 0.0 or not np.isfinite(value):
            return
        timings = getattr(self, "phase_timings_", None)
        if not isinstance(timings, dict):
            timings = {}
            self.phase_timings_ = timings
        timings[phase] = float(timings.get(phase, 0.0) or 0.0) + value

    def _estimate_compute_budget(self, X, current_r2, term_count, uncertainty=None):
        """Adaptive compute budget: easy problems get short runs, hard problems get longer runs.

        When *uncertainty* (from the fast-path FPIP) is supplied the budget
        is further scaled:
        - Low entropy + high margin → the classifier is confident, reduce budget.
        - High entropy + low margin → uncertain, give evolution more time.
        - Exact fast-path hit with low uncertainty → minimal budget.
        """
        base_timeout = float(max(1, self.timeout))
        if not self.adaptive_compute_budget:
            return base_timeout

        n_samples = int(X.shape[0])
        n_features = int(X.shape[1])

        score = 1.0
        score += 0.15 * max(0, n_features - 1)
        score += 0.08 * min(1.0, np.log10(max(50, n_samples)) / 3.0)

        # Fast-path confidence gates: reduce budget on easy problems.
        if current_r2 >= 0.995 and term_count <= 5:
            score *= 0.2
        elif current_r2 >= 0.98 and term_count <= 8:
            score *= 0.5
        elif current_r2 >= 0.90:
            score *= 0.9
        else:
            score *= 2.5

        # ── Uncertainty-coupled budget routing ──
        # If classifier uncertainty metrics are available, scale budget:
        # certain classifier + strong R² → avoid expensive guided escalation.
        if isinstance(uncertainty, dict):
            entropy = uncertainty.get('prediction_entropy')
            margin = uncertainty.get('prediction_margin')
            uncertain_flag = bool(uncertainty.get('prediction_uncertain', False))

            if not uncertain_flag and entropy is not None and margin is not None:
                try:
                    ent = float(entropy)
                    mar = float(margin)
                    if np.isfinite(ent) and np.isfinite(mar):
                        # High confidence (low entropy, high margin) → shrink budget
                        confidence = float(np.clip((1.0 - ent) * min(mar / 0.25, 1.0), 0.0, 1.0))
                        
                        # Map confidence ∈ [0,1] to multiplier ∈ [0.1, 1.0] (more aggressive than 0.3)
                        uncertainty_scale = 1.0 - 0.9 * confidence
                        score *= uncertainty_scale
                except (TypeError, ValueError):
                    pass
            elif uncertain_flag:
                # Uncertain → give more time, but cap the escalation
                score *= 1.2

        # Phase 7: residual/weight noise pressure expands budget; never shrink on
        # high residual noise just because current R² looks fine on noisy labels.
        noise_diag = getattr(self, "_runtime_noise_diagnostics_", None)
        if isinstance(noise_diag, dict):
            band = str(noise_diag.get("noise_band") or "clean")
            if band == "high":
                score *= 1.35
            elif band == "medium":
                score *= 1.15
            elif band == "low":
                score *= 1.05

        # ── Proposer-specific budget scaling ──
        # If we have skeletons, we expect faster convergence.
        if getattr(self, 'universal_proposer_fpip_v2_', None):
            payload = self.universal_proposer_fpip_v2_
            if payload.get('valid') and payload.get('candidate_skeletons'):
                # We have seeds! Reduce base budget because we aren't starting from scratch.
                score *= 0.7

        budget = base_timeout * score
        return float(np.clip(budget, float(self.min_compute_budget), float(self.max_compute_budget)))

    def _split_blackbox_holdout(self, X, y, validation_fraction=0.2):
        """Build a deterministic shuffled train/validation split for candidate screening."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        n = int(X.shape[0])
        if n < 24:
            return None
        seed = 0 if self.random_state is None else int(self.random_state)
        rng = np.random.RandomState(seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        holdout_n = int(max(4, round(n * float(validation_fraction))))
        holdout_n = min(holdout_n, n - 12)
        if holdout_n <= 0:
            return None
        fit_idx = idx[:-holdout_n]
        val_idx = idx[-holdout_n:]
        if fit_idx.size < 12 or val_idx.size < 4:
            return None
        return {
            "fit_idx": fit_idx,
            "val_idx": val_idx,
            "X_fit": X[fit_idx],
            "y_fit": y[fit_idx],
            "X_val": X[val_idx],
            "y_val": y[val_idx],
        }

    def _formula_complexity(self, formula):
        text = str(formula or "").strip()
        if not text:
            return 0
        ops = sum(text.count(ch) for ch in "+-*/^")
        funcs = sum(text.count(name) for name in ("sin", "cos", "exp", "log", "sqrt", "abs"))
        return int(max(1, ops + funcs + 1))

    def _formula_risk_score(self, formula, X=None):
        """Penalize structures that often fit train data but fail blackbox holdout."""
        text = str(formula or "").strip()
        if not text:
            return 1.0
        lower = text.lower()
        risk = 0.0
        decimal_powers = [
            float(match)
            for match in re.findall(r"(?:\^|\*\*)\s*(-?\d+\.\d+)", lower)
        ]
        risk += 0.16 * len(decimal_powers)
        if "/" in lower:
            risk += 0.08 * lower.count("/")
        if "exp(" in lower:
            risk += 0.06 * lower.count("exp(")
        if "sqrt(" in lower and "abs(" not in lower:
            risk += 0.12
        if "log(" in lower and "abs(" not in lower:
            risk += 0.10
        risk += 0.012 * max(0, self._formula_complexity(text) - 12)

        if X is not None and "/" in lower:
            # Probe denominator fragility by turning a/b into b where possible.
            for denom in re.findall(r"/\s*(\([^()]+\)|[a-zA-Z0-9_.*+\-]+)", lower):
                denom_text = denom.strip()
                if denom_text.startswith("(") and denom_text.endswith(")"):
                    denom_text = denom_text[1:-1]
                try:
                    values = self._safe_eval_formula_array(denom_text, X)
                    near_zero = float(np.mean(np.abs(values) < 1e-4))
                    risk += min(0.25, 0.5 * near_zero)
                except Exception:
                    risk += 0.05
        return float(np.clip(risk, 0.0, 1.0))

    def _domain_edge_validation_split(self, X, y, validation_fraction=0.2):
        """Hold out boundary and random points to catch fragile blackbox formulas."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        n = int(X.shape[0])
        if n < 24:
            return self._split_blackbox_holdout(X, y, validation_fraction=validation_fraction)

        holdout_n = int(max(4, round(n * float(validation_fraction))))
        holdout_n = min(holdout_n, n - 12)
        if holdout_n <= 0:
            return None

        finite_X = np.where(np.isfinite(X), X, 0.0)
        z = np.zeros_like(finite_X, dtype=np.float64)
        for j in range(finite_X.shape[1]):
            col = finite_X[:, j]
            med = float(np.median(col))
            scale = float(np.percentile(np.abs(col - med), 75))
            if not np.isfinite(scale) or scale < 1e-12:
                scale = float(np.std(col))
            if not np.isfinite(scale) or scale < 1e-12:
                scale = 1.0
            z[:, j] = np.abs((col - med) / scale)
        edge_score = np.max(z, axis=1) if z.size else np.zeros(n)
        edge_n = min(max(2, holdout_n // 2), holdout_n)
        edge_idx = list(np.argsort(edge_score)[-edge_n:])

        seed = 0 if self.random_state is None else int(self.random_state)
        rng = np.random.RandomState(seed + 7919)
        remaining = [idx for idx in range(n) if idx not in set(edge_idx)]
        rng.shuffle(remaining)
        val_idx = np.asarray(edge_idx + remaining[: max(0, holdout_n - len(edge_idx))], dtype=int)
        fit_idx = np.asarray([idx for idx in range(n) if idx not in set(val_idx.tolist())], dtype=int)
        if fit_idx.size < 12 or val_idx.size < 4:
            return self._split_blackbox_holdout(X, y, validation_fraction=validation_fraction)
        return {
            "fit_idx": fit_idx,
            "val_idx": val_idx,
            "X_fit": X[fit_idx],
            "y_fit": y[fit_idx],
            "X_val": X[val_idx],
            "y_val": y[val_idx],
        }

    def _random_blackbox_validation_split(self, X, y, validation_fraction=0.25, *, salt=0):
        """Random interpolation holdout for Track 1 tabular blackbox selection."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        n = int(X.shape[0])
        if n < 24:
            return self._split_blackbox_holdout(X, y, validation_fraction=validation_fraction)

        holdout_n = int(max(4, round(n * float(validation_fraction))))
        holdout_n = min(holdout_n, n - 12)
        if holdout_n <= 0:
            return None

        seed = 0 if self.random_state is None else int(self.random_state)
        rng = np.random.RandomState(seed + 104729 + int(salt))
        order = np.arange(n, dtype=int)
        rng.shuffle(order)
        val_idx = np.asarray(order[:holdout_n], dtype=int)
        fit_idx = np.asarray(order[holdout_n:], dtype=int)
        if fit_idx.size < 12 or val_idx.size < 4:
            return self._split_blackbox_holdout(X, y, validation_fraction=validation_fraction)
        return {
            "fit_idx": fit_idx,
            "val_idx": val_idx,
            "X_fit": X[fit_idx],
            "y_fit": y[fit_idx],
            "X_val": X[val_idx],
            "y_val": y[val_idx],
        }

    def _ridge_tail_validation_r2(self, X, y, columns=None, validation_fraction=0.25):
        """Small ordered-holdout ridge probe used to audit feature reduction."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if columns is not None:
            X = X[:, list(columns)]
        n = int(X.shape[0])
        holdout_n = int(max(8, round(n * float(validation_fraction))))
        holdout_n = min(holdout_n, n - 16)
        if holdout_n <= 0 or X.shape[1] == 0:
            return None
        X_fit = X[:-holdout_n]
        y_fit = y[:-holdout_n]
        X_val = X[-holdout_n:]
        y_val = y[-holdout_n:]
        if X_fit.shape[0] < 16 or X_val.shape[0] < 8:
            return None

        mu = np.mean(X_fit, axis=0)
        sigma = np.std(X_fit, axis=0)
        sigma = np.where(sigma < 1e-10, 1.0, sigma)
        Z_fit = (X_fit - mu) / sigma
        Z_val = (X_val - mu) / sigma
        A_fit = np.column_stack([Z_fit, np.ones(Z_fit.shape[0])])
        A_val = np.column_stack([Z_val, np.ones(Z_val.shape[0])])
        y_var = max(float(np.var(y_val)), 1e-12)
        best = None
        for alpha in np.logspace(-5, 4, 18):
            reg = np.eye(A_fit.shape[1], dtype=np.float64) * float(alpha)
            reg[-1, -1] = 0.0
            try:
                coef = np.linalg.solve(A_fit.T @ A_fit + reg, A_fit.T @ y_fit)
            except Exception:
                continue
            pred = A_val @ coef
            if not np.all(np.isfinite(pred)):
                continue
            mse = float(np.mean((pred - y_val) ** 2))
            if not np.isfinite(mse):
                continue
            r2 = 1.0 - mse / y_var
            if best is None or r2 > best:
                best = float(r2)
        return best

    def _fit_ridge_formula(self, X, y, columns=None, validation_fraction=0.25):
        """Fit a compact linear ridge formula in the original feature space."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        cols = list(range(X.shape[1])) if columns is None else list(columns)
        if not cols:
            return None
        X_sub = X[:, cols]
        n = int(X_sub.shape[0])
        holdout_n = int(max(8, round(n * float(validation_fraction))))
        holdout_n = min(holdout_n, n - 16)
        if holdout_n <= 0:
            return None
        X_fit = X_sub[:-holdout_n]
        y_fit = y[:-holdout_n]
        X_val = X_sub[-holdout_n:]
        y_val = y[-holdout_n:]
        if X_fit.shape[0] < 16:
            return None
        mu = np.mean(X_fit, axis=0)
        sigma = np.std(X_fit, axis=0)
        sigma = np.where(sigma < 1e-10, 1.0, sigma)
        Z_fit = (X_fit - mu) / sigma
        Z_val = (X_val - mu) / sigma
        A_fit = np.column_stack([Z_fit, np.ones(Z_fit.shape[0])])
        A_val = np.column_stack([Z_val, np.ones(Z_val.shape[0])])
        y_var = max(float(np.var(y_val)), 1e-12)
        best = None
        for alpha in np.logspace(-5, 4, 18):
            reg = np.eye(A_fit.shape[1], dtype=np.float64) * float(alpha)
            reg[-1, -1] = 0.0
            try:
                coef = np.linalg.solve(A_fit.T @ A_fit + reg, A_fit.T @ y_fit)
            except Exception:
                continue
            pred = A_val @ coef
            if not np.all(np.isfinite(pred)):
                continue
            val_mse = float(np.mean((pred - y_val) ** 2))
            if not np.isfinite(val_mse):
                continue
            if best is None or val_mse < best["validation_mse"]:
                best = {"coef": coef, "validation_mse": val_mse, "alpha": float(alpha)}
        if best is None:
            return None

        full_mu = np.mean(X_sub, axis=0)
        full_sigma = np.std(X_sub, axis=0)
        full_sigma = np.where(full_sigma < 1e-10, 1.0, full_sigma)
        Z_full = (X_sub - full_mu) / full_sigma
        A_full = np.column_stack([Z_full, np.ones(Z_full.shape[0])])
        reg = np.eye(A_full.shape[1], dtype=np.float64) * float(best["alpha"])
        reg[-1, -1] = 0.0
        try:
            coef_full = np.linalg.solve(A_full.T @ A_full + reg, A_full.T @ y)
        except Exception:
            coef_full = best["coef"]
            full_mu = mu
            full_sigma = sigma

        coef_z = np.asarray(coef_full[:-1], dtype=np.float64)
        intercept_z = float(coef_full[-1])
        weights = coef_z / full_sigma
        bias = intercept_z - float(np.sum(coef_z * full_mu / full_sigma))
        terms = []
        selected_terms = []
        for col, weight in zip(cols, weights):
            if not np.isfinite(weight) or abs(float(weight)) < 1e-10:
                continue
            selected_terms.append(f"x{col}")
            terms.append(f"({float(weight):.12g})*x{col}")
        if abs(bias) > 1e-10 or not terms:
            terms.append(f"({bias:.12g})")
        formula = "+".join(terms) if terms else "0"
        try:
            full_pred = self._safe_eval_formula_array(formula, X)
        except Exception:
            return None
        full_mse = float(np.mean((full_pred - y) ** 2))
        return {
            "formula": formula,
            "mse": full_mse,
            "validation_mse": float(best["validation_mse"]),
            "validation_r2": float(1.0 - best["validation_mse"] / y_var),
            "selected_terms": selected_terms,
            "n_terms": len(selected_terms),
            "complexity": self._formula_complexity(formula),
            "source": "original_linear_ridge",
        }

    def _soft_residual_weights(self, resid, *, floor=0.05, cap=1.0, sample_weight=None):
        """Huber-like soft weights from residual MAD; optional multiply by sample_weight."""
        r = np.asarray(resid, dtype=np.float64).reshape(-1)
        if r.size < 4 or not np.any(np.isfinite(r)):
            return None
        finite = np.isfinite(r)
        r_f = r[finite]
        med = float(np.median(r_f))
        mad = float(np.median(np.abs(r_f - med))) + 1e-12
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale < 1e-12:
            scale = float(np.std(r_f)) + 1e-12
        thr = 2.5 * scale
        w = np.ones(r.shape[0], dtype=np.float64)
        abs_c = np.abs(r - med)
        heavy = finite & (abs_c > thr)
        if np.any(heavy):
            w[heavy] = np.clip(thr / np.maximum(abs_c[heavy], 1e-12), float(floor), float(cap))
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
            if sw.shape == w.shape:
                w = w * np.clip(sw, 0.0, None)
        total = float(np.sum(w[finite]))
        if total <= 0:
            return None
        mean_w = total / max(float(np.sum(finite)), 1.0)
        if mean_w > 1e-12:
            w = w / mean_w
        return w

    def _inlier_mse(self, pred, target, sample_weight=None, *, inlier_frac=0.90):
        """MSE on soft inliers (lowest abs residual mass) for Exact-friendly selection."""
        p = np.asarray(pred, dtype=np.float64).reshape(-1)
        t = np.asarray(target, dtype=np.float64).reshape(-1)
        if p.shape != t.shape or p.size < 4:
            return float("inf")
        resid = p - t
        if not np.all(np.isfinite(resid)):
            return float("inf")
        abs_r = np.abs(resid)
        n = int(abs_r.size)
        keep = max(4, int(round(n * float(inlier_frac))))
        keep = min(keep, n)
        order = np.argsort(abs_r)
        idx = order[:keep]
        if sample_weight is not None:
            w = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
            if w.shape == t.shape:
                ww = w[idx]
                total = float(np.sum(ww))
                if total > 0:
                    return float(np.sum(ww * resid[idx] ** 2) / total)
        return float(np.mean(resid[idx] ** 2))

    def _refine_formula_constants(
        self,
        formula,
        X_fit,
        y_fit,
        X_val,
        y_val,
        *,
        max_constants=8,
        sample_weight=None,
        fit_weights=None,
        robust=True,
        irls_iters=3,
    ):
        """Optimize numeric constants inside a candidate structure with least squares.

        Under outliers, soft MAD / IRLS residual weights keep free-const recovery
        from drifting so near-integer snap can hit Exact on clean labels.
        """
        if least_squares is None:
            return None
        text = str(formula or "").strip()
        if not text:
            return None
        number_pattern = re.compile(r"(?<![A-Za-z_])(?<!\w)([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)")
        matches = []
        for m in number_pattern.finditer(text):
            raw = m.group(0)
            if raw in {"0", "1"}:
                continue
            # Do not free feature indices (x0) or pure integer exponents (x^2, x^4).
            if m.start() > 0 and text[m.start() - 1] == "x":
                continue
            if m.start() > 0 and text[m.start() - 1] == "^":
                try:
                    if float(raw).is_integer() and abs(float(raw)) <= 8:
                        continue
                except Exception:
                    pass
            matches.append(m)
        if not matches or len(matches) > int(max_constants):
            return None

        values = []
        pieces = []
        last = 0
        for idx, match in enumerate(matches):
            try:
                values.append(float(match.group(0)))
            except Exception:
                return None
            pieces.append(text[last:match.start()])
            pieces.append(f"__c{idx}")
            last = match.end()
        pieces.append(text[last:])
        template = "".join(pieces)
        initial = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(initial)):
            return None

        y_fit = np.asarray(y_fit, dtype=np.float64).reshape(-1)
        y_val = np.asarray(y_val, dtype=np.float64).reshape(-1)
        base_w = fit_weights if fit_weights is not None else sample_weight
        if base_w is not None:
            base_w = np.asarray(base_w, dtype=np.float64).reshape(-1)
            if base_w.shape != y_fit.shape:
                base_w = None
        if base_w is None and robust:
            # Target MAD soft weights catch label spikes before residual IRLS.
            base_w = _soft_mad_sample_weights(y_fit)
        if base_w is None:
            base_w = np.ones(y_fit.shape[0], dtype=np.float64)

        def build(params):
            out = template
            for idx, value in enumerate(params):
                out = out.replace(f"__c{idx}", f"({float(value):.12g})")
            return out

        def raw_residuals(params):
            candidate = build(params)
            try:
                pred = self._safe_eval_formula_array(candidate, X_fit)
            except Exception:
                return np.full_like(y_fit, 1e6, dtype=np.float64)
            pred = np.asarray(pred, dtype=np.float64).reshape(-1)
            if pred.shape != y_fit.shape or not np.all(np.isfinite(pred)):
                return np.full_like(y_fit, 1e6, dtype=np.float64)
            return np.clip(pred - y_fit, -1e6, 1e6)

        def weighted_cost(params, w):
            r = raw_residuals(params)
            ww = np.asarray(w, dtype=np.float64).reshape(-1)
            total = float(np.sum(ww))
            if total <= 0:
                return float(np.mean(r ** 2))
            return float(np.sum(ww * r ** 2) / total)

        best_params = None
        best_cost = float("inf")
        w_cur = np.asarray(base_w, dtype=np.float64).reshape(-1)
        n_irls = max(1, int(irls_iters) if robust else 1)
        f_scale = max(1e-6, float(np.std(y_fit)) * (0.05 if robust else 0.1))
        max_nfev = 400 if robust else 200
        x0 = initial.copy()

        for _irls in range(n_irls):
            sw = np.sqrt(np.clip(w_cur, 1e-8, None))

            def residuals(params, _sw=sw):
                return raw_residuals(params) * _sw

            try:
                result = least_squares(
                    residuals,
                    x0,
                    max_nfev=max_nfev,
                    loss="soft_l1",
                    f_scale=f_scale,
                )
            except Exception:
                result = None
            if result is None or not np.all(np.isfinite(result.x)):
                continue
            cost = weighted_cost(result.x, w_cur)
            init_cost = weighted_cost(x0, w_cur)
            accept = np.isfinite(cost) and (
                cost <= init_cost * 1.02 + 1e-15 or getattr(result, "success", False)
            )
            if not accept:
                continue
            if cost < best_cost:
                best_cost = cost
                best_params = np.asarray(result.x, dtype=np.float64).copy()
            x0 = np.asarray(result.x, dtype=np.float64)
            # Residual IRLS reweight for next pass
            if robust and _irls + 1 < n_irls:
                r_next = raw_residuals(x0)
                w_soft = self._soft_residual_weights(r_next, sample_weight=base_w)
                if w_soft is not None:
                    w_cur = w_soft

        if best_params is None:
            # Fallback: single unweighted soft_l1 (legacy path)
            try:
                result = least_squares(
                    raw_residuals,
                    initial,
                    max_nfev=200,
                    loss="soft_l1",
                    f_scale=max(1e-6, float(np.std(y_fit)) * 0.1),
                )
            except Exception:
                return None
            if result is None or not np.all(np.isfinite(result.x)):
                return None
            try:
                init_cost = float(np.mean(raw_residuals(initial) ** 2))
                new_cost = float(np.mean(raw_residuals(result.x) ** 2))
                if not np.isfinite(new_cost) or new_cost > init_cost * 1.01 + 1e-15:
                    if not getattr(result, "success", False):
                        return None
            except Exception:
                if not getattr(result, "success", False):
                    return None
            best_params = np.asarray(result.x, dtype=np.float64)

        refined_formula = build(best_params)
        try:
            pred_fit = self._safe_eval_formula_array(refined_formula, X_fit)
            pred_val = self._safe_eval_formula_array(refined_formula, X_val)
        except Exception:
            return None
        if not (np.all(np.isfinite(pred_fit)) and np.all(np.isfinite(pred_val))):
            return None
        fit_mse = float(np.mean((pred_fit - y_fit) ** 2))
        val_mse = float(np.mean((pred_val - y_val) ** 2))
        if not np.isfinite(fit_mse) or not np.isfinite(val_mse):
            return None
        inlier_fit = self._inlier_mse(pred_fit, y_fit, base_w)
        inlier_val = self._inlier_mse(pred_val, y_val)
        val_var = float(np.var(y_val))
        val_r2 = 1.0 if val_var < 1e-15 and val_mse < 1e-15 else (
            0.0 if val_var < 1e-15 else 1.0 - val_mse / val_var
        )
        return {
            "formula": refined_formula,
            "fit_mse": fit_mse,
            "mse": val_mse,
            "validation_mse": val_mse,
            "validation_r2": float(val_r2),
            "inlier_fit_mse": float(inlier_fit) if np.isfinite(inlier_fit) else None,
            "inlier_val_mse": float(inlier_val) if np.isfinite(inlier_val) else None,
            "complexity": self._formula_complexity(refined_formula),
            "constant_refined": True,
            "robust_refined": bool(robust),
        }

    def _snap_near_integer_constants(self, formula, *, atol=1e-4, max_abs=100):
        """Snap free numeric literals near integers (Exact recovery hygiene)."""
        text = str(formula or "")
        if not text:
            return text
        number_pattern = re.compile(r"(?<![A-Za-z_])(?<!\w)([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)")

        def repl(match):
            raw = match.group(0)
            # Don't rewrite feature indices like x0
            if match.start() > 0 and text[match.start() - 1] == "x":
                return raw
            try:
                val = float(raw)
            except Exception:
                return raw
            if abs(val) > float(max_abs):
                return raw
            nearest = round(val)
            if abs(val - nearest) <= float(atol):
                return str(int(nearest))
            return raw

        return number_pattern.sub(repl, text)

    # Known algebraic / physics constants tried when free-const inliers are excellent.
    # Score on inliers only — never force if structure fit degrades.
    _KNOWN_STRUCTURE_CONSTANTS = (
        1.0 / (4.0 * np.pi),   # ≈ 0.079577… Coulomb / Feynman-I.9.18
        1.0 / (2.0 * np.pi),   # ≈ 0.159155…
        1.0 / np.pi,           # ≈ 0.318310…
        np.pi,
        2.0 * np.pi,
        4.0 * np.pi,
        np.e,
        1.0 / np.e,
        np.sqrt(2.0),
        np.sqrt(3.0),
        0.5,
        2.0,
        5.0,
        10.0,
    )

    def _snap_known_structure_constants(self, formula, *, atol=0.01, max_abs=100):
        """Snap free literals toward a small bank of algebraic/physics constants.

        Used after excellent inlier MSE under spikes so e.g. 0.07855 → 1/(4π)
        without template auto-win. Pure hygiene; caller must re-score inliers.
        """
        text = str(formula or "")
        if not text:
            return text
        number_pattern = re.compile(r"(?<![A-Za-z_])(?<!\w)([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)")
        bank = [float(c) for c in self._KNOWN_STRUCTURE_CONSTANTS if np.isfinite(float(c))]

        def repl(match):
            raw = match.group(0)
            if match.start() > 0 and text[match.start() - 1] == "x":
                return raw
            # Leave pure integer exponents (x^2, x^4).
            if match.start() > 0 and text[match.start() - 1] == "^":
                return raw
            try:
                val = float(raw)
            except Exception:
                return raw
            if abs(val) > float(max_abs) or abs(val) < 1e-15:
                return raw
            # Prefer integer snap when already near int (handled elsewhere);
            # here only non-integer free consts toward known bank.
            nearest_int = round(val)
            if abs(val - nearest_int) <= 1e-9:
                return raw
            best_c = None
            best_d = float("inf")
            for c in bank:
                # Relative + absolute tolerance so 0.07855 hits 0.079577.
                d = abs(val - c)
                rel = d / max(abs(c), abs(val), 1e-12)
                if d <= float(atol) or rel <= 0.03:
                    if d < best_d:
                        best_d = d
                        best_c = c
            if best_c is None:
                return raw
            # Emit compact decimal (keep enough digits for 1/(4π)).
            return f"{best_c:.12g}"

        return number_pattern.sub(repl, text)

    def _aggressive_exact_snap(self, formula, X, y, *, inlier_gate=1e-4):
        """When inliers are excellent, push free constants to integers / known bank.

        Returns (formula, full_mse, inlier_mse). Prefer inlier fidelity over full
        noisy MSE so spike-dominated labels cannot block Exact clean recovery.
        """
        text = str(formula or "").strip()
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        if not text or X_arr.ndim != 2 or y_arr.size < 8:
            return text, float("inf"), float("inf")

        def _score(f):
            try:
                pred = self._safe_eval_formula_array(f, X_arr)
                pred = np.asarray(pred, dtype=np.float64).reshape(-1)
                mse = float(np.mean((pred - y_arr) ** 2))
                inlier = self._inlier_mse(pred, y_arr)
                return mse, inlier
            except Exception:
                return float("inf"), float("inf")

        best_f = text
        best_m, best_in = _score(text)
        if not (np.isfinite(best_in) and best_in <= float(inlier_gate)):
            return best_f, best_m, best_in

        # Wider integer atols when inliers already tiny (centers 3.058→3, 5.1→5).
        int_atols = (5e-3, 1e-2, 2e-2, 5e-2, 8e-2) if best_in < 1e-5 else (5e-3, 1e-2, 2e-2)
        for atol in int_atols:
            snapped = self._snap_near_integer_constants(best_f, atol=atol)
            if not snapped or snapped == best_f:
                continue
            mse_s, in_s = _score(snapped)
            if np.isfinite(in_s) and in_s <= max(best_in * 1.5, 1e-10) + 1e-15:
                best_f, best_m, best_in = snapped, mse_s, in_s

        # Known-constant bank (physics product ratio, π, e, …).
        for atol in (0.005, 0.01, 0.02):
            snapped = self._snap_known_structure_constants(best_f, atol=atol)
            if not snapped or snapped == best_f:
                continue
            mse_s, in_s = _score(snapped)
            if np.isfinite(in_s) and in_s <= max(best_in * 1.5, 1e-10) + 1e-15:
                best_f, best_m, best_in = snapped, mse_s, in_s
                break

        # Second integer pass after known-const (can unlock 10.0 / 5.0).
        for atol in (1e-2, 5e-2):
            snapped = self._snap_near_integer_constants(best_f, atol=atol)
            if not snapped or snapped == best_f:
                continue
            mse_s, in_s = _score(snapped)
            if np.isfinite(in_s) and in_s <= max(best_in * 1.5, 1e-10) + 1e-15:
                best_f, best_m, best_in = snapped, mse_s, in_s

        return best_f, best_m, best_in

    def _parse_outer_affine(self, formula):
        """Parse ((s)*(inner)+(b)) allowing extra paren nesting. Returns (s, inner, b) or None."""
        text = str(formula or "").strip().replace(" ", "")
        if not text:
            return None
        # Peel one layer of wrapping parens if whole expression is parenthesized
        while text.startswith("(") and text.endswith(")"):
            depth = 0
            balanced = True
            for i, ch in enumerate(text):
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0 and i != len(text) - 1:
                        balanced = False
                        break
            if balanced and depth == 0 and len(text) > 2:
                text = text[1:-1]
            else:
                break
        # Match s * (inner) + b  or  (s)*(inner)+(b)
        m = re.match(
            r"^\(?([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\)?\*\((.+)\)\+?\(?([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\)?$",
            text,
        )
        if not m:
            return None
        try:
            return float(m.group(1)), m.group(2), float(m.group(3))
        except Exception:
            return None

    def _strip_near_identity_affine(self, formula, *, scale_tol=0.05, bias_tol=0.05):
        """Unwrap ((s)*(inner)+(b)) when s≈1 and b≈0 for Exact-friendly display."""
        text = str(formula or "").strip()
        if not text:
            return text
        for _ in range(4):
            parsed = self._parse_outer_affine(text)
            if parsed is None:
                break
            scale, inner, bias = parsed
            if abs(scale - 1.0) <= float(scale_tol) and abs(bias) <= float(bias_tol):
                text = inner.strip()
                continue
            break
        return text

    def _fold_outer_affine_into_leading_const(self, formula):
        """((s)*((c)*expr)+(b)) → (s*c)*expr when b≈0 for cleaner Exact snap."""
        text = str(formula or "").strip()
        if not text:
            return formula
        parsed = self._parse_outer_affine(text)
        if parsed is None:
            return formula
        scale, inner, bias = parsed
        # Peel leading constant from inner: (c)*rest or c*rest
        inner_c = inner.strip().replace(" ", "")
        while inner_c.startswith("(") and inner_c.endswith(")"):
            depth = 0
            ok = True
            for i, ch in enumerate(inner_c):
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0 and i != len(inner_c) - 1:
                        ok = False
                        break
            if ok and depth == 0 and len(inner_c) > 2:
                inner_c = inner_c[1:-1]
            else:
                break
        m = re.match(
            r"^\(?([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\)?\*(.+)$",
            inner_c,
        )
        if not m:
            return formula
        try:
            c0 = float(m.group(1))
            rest = m.group(2)
            folded = scale * c0
            if abs(bias) <= 1e-6:
                return f"({folded:.12g})*{rest}"
            return f"(({folded:.12g})*({rest})+({bias:.12g}))"
        except Exception:
            return formula

    def _drop_tiny_outer_bias(self, formula, *, bias_tol=0.05):
        """(c)*expr + b → (c)*expr when |b| is small (Exact under spikes)."""
        text = str(formula or "").strip()
        parsed = self._parse_outer_affine(text)
        if parsed is None:
            return formula
        scale, inner, bias = parsed
        if abs(bias) <= float(bias_tol) and abs(scale) > 1e-15:
            # If scale is already the leading const and inner is product-like
            return f"({scale:.12g})*{inner}"
        return formula

    def _robust_scale_only_refit(self, formula, X, y, *, sample_weight=None, iters=5):
        """Refit leading scale of (c)*expr via L1/median-ratio IRLS (Exact under spikes).

        Soft-Huber on residuals alone under-pulls product ratios when spikes are
        large; median(y/base) + L1 IRLS recovers the clean constant.
        """
        text = str(formula or "").strip().replace(" ", "")
        if not text:
            return formula, None
        # Match (c)*rest  or  c*rest  (no outer bias)
        m = re.match(
            r"^\(?([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\)?\*(.+)$",
            text,
        )
        if not m:
            parsed = self._parse_outer_affine(formula)
            if parsed is None:
                return formula, None
            scale0, inner, bias = parsed
            if abs(bias) > 0.05:
                return formula, None
            rest = inner
            c0 = scale0
        else:
            try:
                c0 = float(m.group(1))
                rest = m.group(2)
            except Exception:
                return formula, None
        try:
            base = self._safe_eval_formula_array(rest, X)
            base = np.asarray(base, dtype=np.float64).reshape(-1)
            y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
            if base.shape != y_arr.shape or not np.all(np.isfinite(base)):
                return formula, None
        except Exception:
            return formula, None
        mask = np.isfinite(base) & np.isfinite(y_arr) & (np.abs(base) > 1e-12)
        if int(np.sum(mask)) < 8:
            return formula, None
        ratios = y_arr[mask] / base[mask]
        c_med = float(np.median(ratios))
        # Trimmed mean of ratios (drop 5% tails)
        order = np.argsort(ratios)
        n_r = int(order.size)
        lo = max(0, int(0.05 * n_r))
        hi = max(lo + 1, int(0.95 * n_r))
        c_trim = float(np.mean(ratios[order[lo:hi]]))
        c = float(c0)
        # Prefer median / trim start when they disagree with OLS-like c0
        for c_start in (c_med, c_trim, c):
            if np.isfinite(c_start):
                c = c_start
                break
        for _ in range(max(2, int(iters))):
            resid = y_arr - c * base
            # L1 IRLS: w ~ 1/|r|; floor to avoid inf
            ww = 1.0 / np.maximum(np.abs(resid), 1e-8)
            ww = np.clip(ww, 0.01, 100.0)
            if sample_weight is not None:
                sw = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
                if sw.shape == ww.shape:
                    ww = ww * np.clip(sw, 0.0, None)
            ww = ww * mask.astype(np.float64)
            den = float(np.sum(ww * base * base))
            if abs(den) < 1e-18:
                break
            c = float(np.sum(ww * y_arr * base) / den)
        # Collapse leading (a)*(b)*rest → (a*b)*rest for cleaner formulas
        rest_c = str(rest).replace(" ", "")
        m2 = re.match(
            r"^\(?([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\)?\*(.+)$",
            rest_c,
        )
        if m2:
            try:
                # base was eval of rest which already embeds m2.group(1); keep c, rewrite rest
                c_inner = float(m2.group(1))
                if abs(c_inner) > 1e-15:
                    # Re-express: c * (c_inner * core) == (c) * rest already correct numerically
                    # Prefer single leading constant times core structure
                    core = m2.group(2)
                    c_combined = c * c_inner
                    new_f = f"({c_combined:.12g})*{core}"
                    pred = c * base  # still valid
                    mse = float(np.mean((pred - y_arr) ** 2))
                    inlier = self._inlier_mse(pred, y_arr)
                    return new_f, {"mse": mse, "inlier_mse": inlier, "scale": c_combined}
            except Exception:
                pass
        new_f = f"({c:.12g})*{rest}"
        try:
            pred = c * base
            mse = float(np.mean((pred - y_arr) ** 2))
            inlier = self._inlier_mse(pred, y_arr)
        except Exception:
            return new_f, None
        return new_f, {"mse": mse, "inlier_mse": inlier, "scale": c}

    def _fit_original_space_structure_winner(self, X_original, y_original, blackbox_state):
        """Fit free-const structure families on original (unstandardized) selected cols.

        Returns best formula already in original feature indices, or None.
        Competing candidate only — no auto-win.
        """
        X_all = np.asarray(X_original, dtype=np.float64)
        y_all = np.asarray(y_original, dtype=np.float64).reshape(-1)
        if X_all.ndim != 2 or y_all.size < 20:
            return None
        selected = list(getattr(blackbox_state, "selected_features", []) or list(range(X_all.shape[1])))
        if len(selected) < 2:
            return None
        if any(int(i) < 0 or int(i) >= X_all.shape[1] for i in selected):
            return None
        X_sel = X_all[:, [int(i) for i in selected]]
        # Skeletons in reduced indices on original-scale data
        skeletons = list(build_search_space_structure_seeds(X_sel.shape[1], max_seeds=16))
        # Prefer free-const affine forms on original scale
        n = X_sel.shape[1]
        radial = []
        # Radial / Vlad-like: free center + numerator/denom (Exact under outliers via IRLS)
        for center in (3.0, 2.0, 1.0, 4.0, 0.0):
            # Non-integer seeds so free-const refine can move them
            c_seed = f"{center + 0.1:.1f}" if abs(center) > 1e-12 else "0.1"
            sq_c = "+".join(f"(x{i}-{c_seed})^2" for i in range(n))
            radial.extend(
                [
                    f"10.1/(5.1+{sq_c})",
                    f"5.1/(5.1+{sq_c})",
                    f"1.1/(1.1+{sq_c})",
                ]
            )
        sq = "+".join(f"(1.1*x{i}+0.1)^2" for i in range(n))
        radial.extend(
            [
                f"10.1/(5.1+{sq})",
                f"5.1/(5.1+{sq})",
                f"1.1/(1.1+{sq})",
            ]
        )
        pagie = [
            "+".join(f"x{i}^4/(1.1+x{i}^4)" for i in range(min(n, 4))),
            "+".join(f"1.1/(1.1+x{i}^4)" for i in range(min(n, 4))),
            "+".join(f"(1.1*x{i}+0.1)^4/(1.1+(1.1*x{i}+0.1)^4)" for i in range(min(n, 4))),
        ]
        product = []
        if n >= 3:
            # 1/(4*pi) ≈ 0.079577; free-const seeds near physics product-ratio
            product = [
                "0.079577*x0*x1/x2^2",
                "0.0796*x0*x1/x2^2",
                "0.08*x0*x1/x2^2",
                "0.1*x0*x1/x2^2",
                "x0*x1/x2^2",
                "(1.1*x0+0.1)*(1.1*x1+0.1)/((1.1*x2+0.1)^2)",
            ]
        # Priority: product / radial / pagie / generic seeds
        skeletons = product + radial + pagie + skeletons
        # Dedup
        seen = set()
        uniq = []
        for s in skeletons:
            if s not in seen:
                seen.add(s)
                uniq.append(s)
        # Cap: product-first for n>=3 so physics ratio wins before heavy radial bank
        skeletons = uniq[:20 if n < 3 else 16]

        n_pts = int(y_all.shape[0])
        n_val = max(8, int(round(0.25 * n_pts)))
        n_val = min(n_val, n_pts - 12)
        idx = np.arange(n_pts)
        rng = np.random.RandomState(43)
        rng.shuffle(idx)
        val_idx, fit_idx = idx[:n_val], idx[n_val:]
        X_fit, y_fit = X_sel[fit_idx], y_all[fit_idx]
        X_val, y_val = X_sel[val_idx], y_all[val_idx]
        # Fit-time soft weights on original y (label spikes)
        fit_w = None
        try:
            stored = self._active_sample_weight(n_targets=n_pts)
            if stored is not None:
                fit_w = np.asarray(stored, dtype=np.float64).reshape(-1)[fit_idx]
        except Exception:
            fit_w = None
        if fit_w is None:
            fit_w = _soft_mad_sample_weights(y_fit)

        def _eval_one_skeleton(skel):
            working = skel
            refined = self._refine_formula_constants(
                skel,
                X_fit,
                y_fit,
                X_val,
                y_val,
                max_constants=14,
                fit_weights=fit_w,
                robust=True,
                irls_iters=2,
            )
            if refined is not None and str(refined.get("formula") or "").strip():
                working = str(refined["formula"])
            # Prefer free-const structure without outer affine (Exact path).
            # Affine outer fit drifts constants under spikes and blocks integer snap.
            candidates = [working]
            scored = self._score_formula_candidate(
                working, X_fit, y_fit, X_val, y_val, fit_weights=fit_w
            )
            if scored is not None and str(scored.get("formula") or "").strip():
                candidates.append(str(scored["formula"]))
            best_item = None
            for formula_red in candidates:
                formula_orig = remap_reduced_formula_to_original(formula_red, selected)
                try:
                    pred = self._safe_eval_formula_array(formula_orig, X_all)
                    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
                    mse = float(np.mean((pred - y_all) ** 2))
                    inlier = self._inlier_mse(pred, y_all)
                except Exception:
                    continue
                if not np.isfinite(mse):
                    continue
                # Scale-only IRLS on product-like free-const forms
                try:
                    scale_f, scale_info = self._robust_scale_only_refit(
                        formula_orig, X_all, y_all, sample_weight=None, iters=5
                    )
                    if scale_f and scale_info is not None:
                        inlier_sc = float(scale_info.get("inlier_mse", float("inf")))
                        if np.isfinite(inlier_sc) and inlier_sc <= max(inlier * 1.05, 1e-12):
                            formula_orig = scale_f
                            mse = float(scale_info.get("mse", mse))
                            inlier = inlier_sc
                except Exception:
                    pass
                snapped = self._snap_near_integer_constants(formula_orig, atol=1e-3)
                try:
                    pred_s = self._safe_eval_formula_array(snapped, X_all)
                    pred_s = np.asarray(pred_s, dtype=np.float64).reshape(-1)
                    mse_s = float(np.mean((pred_s - y_all) ** 2))
                    inlier_s = self._inlier_mse(pred_s, y_all)
                    if np.isfinite(inlier_s) and (
                        inlier_s <= inlier * 1.10 + 1e-15
                        or (np.isfinite(mse_s) and mse_s <= mse * 1.05 + 1e-12)
                    ):
                        formula_orig, mse, inlier = snapped, mse_s, inlier_s
                except Exception:
                    pass
                # Aggressive Exact snap when free-const inliers already excellent.
                if np.isfinite(inlier) and inlier < 1e-3:
                    try:
                        a_f, a_m, a_in = self._aggressive_exact_snap(
                            formula_orig, X_all, y_all, inlier_gate=1e-3
                        )
                        if a_f and np.isfinite(a_in) and a_in <= max(inlier * 1.5, 1e-10):
                            formula_orig, mse, inlier = a_f, a_m, a_in
                    except Exception:
                        pass
                if np.isfinite(inlier) and inlier < 1e-6:
                    stripped = self._strip_near_identity_affine(formula_orig)
                    if stripped and stripped != formula_orig:
                        try:
                            pred_t = self._safe_eval_formula_array(stripped, X_all)
                            pred_t = np.asarray(pred_t, dtype=np.float64).reshape(-1)
                            inlier_t = self._inlier_mse(pred_t, y_all)
                            mse_t = float(np.mean((pred_t - y_all) ** 2))
                            if np.isfinite(inlier_t) and inlier_t <= max(inlier * 1.2, 1e-10):
                                formula_orig, mse, inlier = stripped, mse_t, inlier_t
                        except Exception:
                            pass
                item = {
                    "formula": formula_orig,
                    "mse": mse,
                    "inlier_mse": float(inlier) if np.isfinite(inlier) else mse,
                    "complexity": self._formula_complexity(formula_orig),
                    "skeleton": skel,
                }
                if best_item is None or (
                    float(item["inlier_mse"]),
                    float(item["mse"]),
                    int(item["complexity"]),
                ) < (
                    float(best_item["inlier_mse"]),
                    float(best_item["mse"]),
                    int(best_item["complexity"]),
                ):
                    best_item = item
            return best_item

        # Product skeletons first (small, Exact-friendly); early-exit before radial bank.
        best = None
        priority = [s for s in skeletons if "x0*x1" in s or "x0*x1/" in s or "*x1/" in s]
        rest = [s for s in skeletons if s not in priority]
        ordered = priority + rest

        def _consider(item):
            nonlocal best
            if item is None:
                return
            if best is None:
                best = item
                return
            key = (
                float(item["inlier_mse"]),
                float(item["mse"]),
                int(item["complexity"]),
            )
            best_key = (
                float(best.get("inlier_mse", best["mse"])),
                float(best["mse"]),
                int(best["complexity"]),
            )
            if key < best_key:
                best = item

        # Sequential on priority (usually 1–6) for fast Exact hit
        for skel in priority:
            _consider(_eval_one_skeleton(skel))
            if best is not None and float(best.get("inlier_mse", 1.0)) < 1e-8:
                return best

        # Thread-parallel free-const fits for remaining (scipy releases GIL).
        if rest and not (best is not None and float(best.get("inlier_mse", 1.0)) < 1e-6):
            n_workers = min(8, max(1, len(rest)), max(1, (os.cpu_count() or 4)))
            try:
                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    futures = [pool.submit(_eval_one_skeleton, sk) for sk in rest]
                    for fut in as_completed(futures):
                        try:
                            _consider(fut.result())
                        except Exception:
                            pass
                        if best is not None and float(best.get("inlier_mse", 1.0)) < 1e-8:
                            break
            except Exception:
                for skel in rest:
                    _consider(_eval_one_skeleton(skel))
                    if best is not None and float(best.get("inlier_mse", 1.0)) < 1e-8:
                        break
        return best

    def _polish_original_space_structure_formula(self, formula, X_original, y_original):
        """Re-fit free constants after std→original remap; snap near-integers.

        Soft_l1 free-const refine resists outliers. Returns (formula, mse).
        """
        text = str(formula or "").strip()
        X_arr = np.asarray(X_original, dtype=np.float64)
        y_arr = np.asarray(y_original, dtype=np.float64).reshape(-1)
        if not text or X_arr.ndim != 2 or y_arr.size < 16:
            return formula, self._formula_mse(text, X_arr, y_arr) if text else float("inf")

        def full_mse(f):
            try:
                pred = self._safe_eval_formula_array(f, X_arr)
                return float(np.mean((np.asarray(pred, dtype=np.float64).reshape(-1) - y_arr) ** 2))
            except Exception:
                return float("inf")

        base_mse = full_mse(text)
        best_f, best_m = text, base_mse
        try:
            pred0 = self._safe_eval_formula_array(text, X_arr)
            best_inlier = self._inlier_mse(pred0, y_arr)
        except Exception:
            best_inlier = base_mse

        # Holdout split for free-const refine
        n = int(y_arr.shape[0])
        n_val = max(8, int(round(0.25 * n)))
        n_val = min(n_val, n - 12)
        idx = np.arange(n)
        rng = np.random.RandomState(41)
        rng.shuffle(idx)
        val_idx = idx[:n_val]
        fit_idx = idx[n_val:]
        X_fit, y_fit = X_arr[fit_idx], y_arr[fit_idx]
        X_val, y_val = X_arr[val_idx], y_arr[val_idx]
        fit_w = None
        try:
            stored = self._active_sample_weight(n_targets=n)
            if stored is not None:
                fit_w = np.asarray(stored, dtype=np.float64).reshape(-1)[fit_idx]
        except Exception:
            fit_w = None
        if fit_w is None:
            fit_w = _soft_mad_sample_weights(y_fit)

        # Prefer collapsing outer affine before any re-refine (avoids product drift).
        folded = self._fold_outer_affine_into_leading_const(best_f)
        for cand in (folded, self._strip_near_identity_affine(folded, scale_tol=0.12, bias_tol=0.12)):
            if not cand or cand == best_f:
                continue
            try:
                pred_c = self._safe_eval_formula_array(cand, X_arr)
                inlier_c = self._inlier_mse(pred_c, y_arr)
                mse_c = full_mse(cand)
                if np.isfinite(inlier_c) and inlier_c <= max(best_inlier * 1.30, 1e-9):
                    best_f, best_m, best_inlier = cand, mse_c, inlier_c
            except Exception:
                pass

        # Drop tiny bias: (c)*expr + eps → (c)*expr when inliers hold
        no_bias = self._drop_tiny_outer_bias(best_f)
        if no_bias and no_bias != best_f:
            try:
                pred_nb = self._safe_eval_formula_array(no_bias, X_arr)
                inlier_nb = self._inlier_mse(pred_nb, y_arr)
                mse_nb = full_mse(no_bias)
                if np.isfinite(inlier_nb) and inlier_nb <= max(best_inlier * 1.25, 1e-9):
                    best_f, best_m, best_inlier = no_bias, mse_nb, inlier_nb
            except Exception:
                pass

        # Scale-only IRLS on (c)*structure — recovers Exact under 3% spikes
        try:
            full_w = None
            try:
                full_w = self._active_sample_weight(n_targets=n)
            except Exception:
                full_w = None
            if full_w is None:
                full_w = _soft_mad_sample_weights(y_arr)
            scale_f, scale_info = self._robust_scale_only_refit(
                best_f, X_arr, y_arr, sample_weight=full_w, iters=6
            )
            if scale_f and scale_info is not None:
                inlier_sc = float(scale_info.get("inlier_mse", float("inf")))
                mse_sc = float(scale_info.get("mse", float("inf")))
                if np.isfinite(inlier_sc) and inlier_sc <= max(best_inlier * 1.05, 1e-12):
                    best_f, best_m, best_inlier = scale_f, mse_sc, inlier_sc
        except Exception:
            pass

        # Only full free-const re-refine when inliers are still weak
        if not (np.isfinite(best_inlier) and best_inlier < 1e-4):
            refined = self._refine_formula_constants(
                best_f,
                X_fit,
                y_fit,
                X_val,
                y_val,
                max_constants=16,
                fit_weights=fit_w,
                robust=True,
                irls_iters=2,
            )
            if refined is not None and str(refined.get("formula") or "").strip():
                cand = str(refined["formula"])
                mse_c = full_mse(cand)
                try:
                    pred_c = self._safe_eval_formula_array(cand, X_arr)
                    inlier_c = self._inlier_mse(pred_c, y_arr)
                except Exception:
                    inlier_c = mse_c
                if np.isfinite(inlier_c) and inlier_c < best_inlier * 0.90:
                    best_f, best_m, best_inlier = cand, mse_c, inlier_c

        # Snap near-integers if inlier fidelity holds (Exact under spikes)
        for atol in (5e-4, 1e-3, 2e-3, 5e-3):
            snapped = self._snap_near_integer_constants(best_f, atol=atol)
            if not snapped or snapped == best_f:
                continue
            mse_s = full_mse(snapped)
            try:
                pred_s = self._safe_eval_formula_array(snapped, X_arr)
                inlier_s = self._inlier_mse(pred_s, y_arr)
            except Exception:
                inlier_s = mse_s
            if np.isfinite(inlier_s) and inlier_s <= best_inlier * 1.15 + 1e-15:
                best_f, best_m, best_inlier = snapped, mse_s, inlier_s
                break

        # Affine outer polish only when inliers are not already near-exact
        # (outer affine destroys integer snap needed for Exact under spikes).
        if float(best_inlier) >= 1e-7:
            try:
                pred = self._safe_eval_formula_array(best_f, X_arr)
                pred = np.asarray(pred, dtype=np.float64).reshape(-1)
                if pred.shape == y_arr.shape and np.all(np.isfinite(pred)):
                    A = np.column_stack([pred, np.ones_like(pred)])
                    resid0 = y_arr - pred
                    w = self._soft_residual_weights(resid0)
                    if w is None:
                        mad = float(np.median(np.abs(resid0 - np.median(resid0)))) + 1e-12
                        w = 1.0 / (1.0 + (np.abs(resid0) / (3.0 * mad)) ** 2)
                        w = np.clip(w, 0.05, 1.0)
                    sw = np.sqrt(np.clip(w, 1e-8, None))
                    coef, _, _, _ = np.linalg.lstsq(A * sw[:, None], y_arr * sw, rcond=None)
                    scale, bias = float(coef[0]), float(coef[1])
                    if abs(scale - 1.0) < 0.08 and abs(bias) < 0.08 * (float(np.std(y_arr)) + 1e-12):
                        if abs(scale - 1.0) > 1e-8 or abs(bias) > 1e-8:
                            affine = f"(({scale:.12g})*({best_f})+({bias:.12g}))"
                            mse_a = full_mse(affine)
                            try:
                                pred_a = self._safe_eval_formula_array(affine, X_arr)
                                inlier_a = self._inlier_mse(pred_a, y_arr)
                            except Exception:
                                inlier_a = mse_a
                            if np.isfinite(inlier_a) and inlier_a < best_inlier * 0.99:
                                best_f, best_m, best_inlier = affine, mse_a, inlier_a
            except Exception:
                pass

        # Final aggressive near-integer snap when inliers are excellent (Exact hygiene).
        snap_atols = (5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2) if float(best_inlier) < 1e-5 else (5e-4, 1e-3, 2e-3, 5e-3)
        for atol in snap_atols:
            snapped = self._snap_near_integer_constants(best_f, atol=atol)
            if not snapped or snapped == best_f:
                continue
            mse_s = full_mse(snapped)
            try:
                pred_s = self._safe_eval_formula_array(snapped, X_arr)
                inlier_s = self._inlier_mse(pred_s, y_arr)
            except Exception:
                continue
            # Under spikes, full MSE is dominated by outliers — trust inliers.
            if np.isfinite(inlier_s) and inlier_s <= max(best_inlier * 1.35, 1e-10) + 1e-15:
                best_f, best_m, best_inlier = snapped, mse_s, inlier_s
                # keep widening while improving; don't break early

        # Known-constant + wider integer bank when inliers already excellent.
        # Closes Exact under outliers for product (1/4π) and radial (centers→3).
        if np.isfinite(best_inlier) and best_inlier < 1e-3:
            try:
                snapped_f, snapped_m, snapped_in = self._aggressive_exact_snap(
                    best_f, X_arr, y_arr, inlier_gate=1e-3
                )
                if (
                    snapped_f
                    and np.isfinite(snapped_in)
                    and snapped_in <= max(best_inlier * 1.5, 1e-10) + 1e-15
                ):
                    best_f, best_m, best_inlier = snapped_f, snapped_m, snapped_in
            except Exception:
                pass

        if isinstance(getattr(self, "blackbox_diagnostics_", None), dict):
            self.blackbox_diagnostics_["original_space_polish"] = {
                "base_mse": float(base_mse) if np.isfinite(base_mse) else None,
                "polished_mse": float(best_m) if np.isfinite(best_m) else None,
                "inlier_mse": float(best_inlier) if np.isfinite(best_inlier) else None,
                "improved": bool(np.isfinite(best_m) and np.isfinite(base_mse) and best_m < base_mse),
                "formula": str(best_f)[:160],
            }
        return best_f, best_m

    def _rewrite_structure_seed_init(self, formula, start_map):
        """Replace leading free numeric literals using start_map offsets for multi-start."""
        text = str(formula or "")
        if not text or not start_map:
            return text
        number_pattern = re.compile(r"(?<![A-Za-z_])(?<!\w)([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)")
        matches = [
            m for m in number_pattern.finditer(text)
            if m.group(0) not in {"0", "1"} and not text[max(0, m.start() - 1):m.start()] == "x"
        ]
        if not matches:
            return text
        pieces = []
        last = 0
        for idx, match in enumerate(matches):
            pieces.append(text[last:match.start()])
            try:
                val = float(match.group(0))
            except Exception:
                pieces.append(match.group(0))
                last = match.end()
                continue
            # Cycle start_map values into successive free constants
            keys = list(start_map.keys())
            if keys:
                key = keys[idx % len(keys)]
                val = float(start_map[key]) if idx < len(keys) else val * float(start_map[key])
            pieces.append(f"{val:.6g}")
            last = match.end()
        pieces.append(text[last:])
        return "".join(pieces)

    def _build_std_aware_structure_skeletons(self, n_features, blackbox_state=None):
        """Build free-const skeletons using standardization mean/scale as inits.

        Uses only preprocessing stats (not ground-truth formulas). Under std,
        original (x_j - c) becomes (s_i * x_i' + mu_i - c); we free a_i,b_i,c0.
        """
        n = int(n_features)
        if n < 2:
            return []
        formulas = []
        means = None
        scales = None
        selected = list(range(n))
        if blackbox_state is not None and getattr(blackbox_state, "standardized", False):
            selected = list(getattr(blackbox_state, "selected_features", []) or list(range(n)))
            if len(selected) == n:
                try:
                    x_mean = np.asarray(getattr(blackbox_state, "x_mean", None), dtype=np.float64)
                    x_scale = np.asarray(getattr(blackbox_state, "x_scale", None), dtype=np.float64)
                    if x_mean is not None and x_scale is not None and x_mean.size > max(selected):
                        means = [float(x_mean[int(j)]) for j in selected]
                        scales = [float(x_scale[int(j)]) for j in selected]
                except Exception:
                    means = None
                    scales = None

        # Radial with free affine-per-feature (init from std mean/scale only).
        # Generic original-space center multi-start (not problem-specific).
        if means is not None and scales is not None:
            terms0 = "+".join(f"({scales[i]:.6g}*x{i}+0.01)^2" for i in range(n))
            formulas.append(f"1.1/(1.1+{terms0})")
            formulas.append(f"5.1/(5.1+{terms0})")
            for center in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0):
                terms_c = "+".join(
                    f"({scales[i]:.6g}*x{i}+{(means[i] - center):.6g})^2" for i in range(n)
                )
                formulas.append(f"1.1/(1.1+{terms_c})")
                formulas.append(f"5.1/(5.1+{terms_c})")
            # Pagie-like: original x maps as s*x'+mu (preprocessing inverse init)
            pagie_terms = "+".join(
                f"({scales[i]:.6g}*x{i}+{means[i]:.6g})^4/"
                f"(1.1+({scales[i]:.6g}*x{i}+{means[i]:.6g})^4)"
                for i in range(min(n, 4))
            )
            formulas.append(pagie_terms)
            pagie_inv = "+".join(
                f"1.1/(1.1+({scales[i]:.6g}*x{i}+{means[i]:.6g})^4)"
                for i in range(min(n, 4))
            )
            formulas.append(pagie_inv)
        else:
            terms = "+".join(f"(1.1*x{i}+0.1)^2" for i in range(n))
            formulas.append(f"1.1/(1.1+{terms})")
            formulas.append(f"5.1/(5.1+{terms})")

        if n >= 3:
            formulas.append("x0*x1/x2^2")
            formulas.append("(1.1*x0+0.1)*(1.1*x1+0.1)/((1.1*x2+0.1)^2)")
            if means is not None and scales is not None:
                formulas.append(
                    f"({scales[0]:.6g}*x0+{means[0]:.6g})*"
                    f"({scales[1]:.6g}*x1+{means[1]:.6g})/"
                    f"(({scales[2]:.6g}*x2+{means[2]:.6g})^2)"
                )
        return formulas

    def _fit_search_space_structure_seeds(self, X, y, *, max_seeds=12, blackbox_state=None):
        """Fit free-constant structure skeletons on standardized multi-var search data.

        Seeds compete in the candidate pool (no auto-win). Returns scored candidates
        already in reduced x0..xk space so remap-to-original stays valid.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        if X_arr.ndim != 2 or X_arr.shape[1] < 2 or y_arr.size < 20:
            return []
        n_features = int(X_arr.shape[1])
        skeletons = list(build_search_space_structure_seeds(n_features, max_seeds=max(8, int(max_seeds))))
        # Prepend std-aware parametric families (preprocessing-init, not GT).
        try:
            std_aware = self._build_std_aware_structure_skeletons(n_features, blackbox_state)
            for s in reversed(std_aware):
                if s not in skeletons:
                    skeletons.insert(0, s)
        except Exception:
            pass
        skeletons = skeletons[: max(12, int(max_seeds) + 6)]
        if not skeletons:
            return []
        split = self._random_blackbox_validation_split(X_arr, y_arr, validation_fraction=0.25, salt=31)
        if split is None:
            split = {
                "X_fit": X_arr,
                "y_fit": y_arr,
                "X_val": X_arr,
                "y_val": y_arr,
            }
        X_fit, y_fit = split["X_fit"], split["y_fit"]
        X_val, y_val = split["X_val"], split["y_val"]
        fit_w, val_w = self._split_sample_weights(
            split, n_total=int(y_arr.shape[0])
        ) if hasattr(self, "_split_sample_weights") else (None, None)

        scored = []
        # Multi-start free-const inits (std-space often needs a≈scale, b≈mean offsets).
        multi_starts = [
            None,  # use skeleton as written
            {0: 1.5, 1: 2.0},  # mild affine
            {0: 2.0, 1: 2.5},
            {0: 1.0, 1: 3.0},  # Vlad-like center bias
            {0: 10.0, 1: 5.0},
        ]
        # Soft weights on fit labels if not already provided (outlier spikes).
        if fit_w is None:
            fit_w = _soft_mad_sample_weights(y_fit)

        def _fit_one_skeleton(skeleton):
            candidates_for_skel = []
            base0 = self._score_formula_candidate(
                skeleton, X_fit, y_fit, X_val, y_val, fit_weights=fit_w, val_weights=val_w
            )
            if base0 is not None:
                candidates_for_skel.append((skeleton, base0, False))
            for start_map in multi_starts:
                skel_try = skeleton
                if start_map is not None:
                    skel_try = self._rewrite_structure_seed_init(skeleton, start_map)
                refined_inner = self._refine_formula_constants(
                    skel_try,
                    X_fit,
                    y_fit,
                    X_val,
                    y_val,
                    max_constants=12,
                    fit_weights=fit_w,
                    robust=True,
                    irls_iters=2,
                )
                working = skel_try
                did_refine = False
                if refined_inner is not None and str(refined_inner.get("formula") or "").strip():
                    working = str(refined_inner["formula"])
                    did_refine = True
                base = self._score_formula_candidate(
                    working, X_fit, y_fit, X_val, y_val, fit_weights=fit_w, val_weights=val_w
                )
                if base is not None:
                    if refined_inner is not None and refined_inner.get("inlier_val_mse") is not None:
                        base = dict(base)
                        base["inlier_mse"] = refined_inner.get("inlier_val_mse")
                    candidates_for_skel.append((working, base, did_refine))
            if not candidates_for_skel:
                return None

            def _skel_key(t):
                b = t[1] or {}
                return (
                    float(b.get("inlier_mse", b.get("mse", float("inf")))),
                    float(b.get("mse", float("inf"))),
                    int(b.get("complexity", 999)),
                )

            working, base, did_refine = min(candidates_for_skel, key=_skel_key)
            chosen = dict(base)
            chosen["source"] = "search_space_structure_seed"
            chosen["skeleton"] = skeleton
            chosen["from_structure_seed"] = True
            chosen["inner_constant_refined"] = bool(did_refine)
            mse = float(chosen.get("mse", float("inf")))
            if not np.isfinite(mse):
                return None
            chosen["complexity"] = int(
                chosen.get("complexity") or self._formula_complexity(chosen.get("formula"))
            )
            return chosen

        n_workers = min(8, max(1, len(skeletons)), max(1, (os.cpu_count() or 4)))
        try:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = [pool.submit(_fit_one_skeleton, sk) for sk in skeletons]
                for fut in as_completed(futures):
                    try:
                        chosen = fut.result()
                    except Exception:
                        chosen = None
                    if chosen is not None:
                        scored.append(chosen)
        except Exception:
            for skeleton in skeletons:
                chosen = _fit_one_skeleton(skeleton)
                if chosen is not None:
                    scored.append(chosen)

        scored.sort(
            key=lambda c: (
                float(c.get("inlier_mse", c.get("mse", float("inf")))),
                float(c.get("mse", float("inf"))),
                int(c.get("complexity", 999)),
                str(c.get("formula", "")),
            )
        )
        top = scored[: max(1, int(max_seeds))]
        if isinstance(getattr(self, "blackbox_diagnostics_", None), dict):
            self.blackbox_diagnostics_["search_space_structure_seeds"] = {
                "n_skeletons": len(skeletons),
                "n_scored": len(scored),
                "n_kept": len(top),
                "best_mse": float(top[0]["mse"]) if top else None,
                "best_formula": str(top[0].get("formula") or "")[:160] if top else None,
                "best_skeleton": str(top[0].get("skeleton") or "")[:120] if top else None,
                "n_workers": int(n_workers),
                "role": "competing_candidate",
                "auto_win": False,
            }
        return top

    def _score_formula_candidate(
        self,
        formula,
        X_fit,
        y_fit,
        X_val,
        y_val,
        fit_weights=None,
        val_weights=None,
    ):
        """Fit affine scaling on train and score on validation (Phase 2 weights)."""
        text = str(formula or "").strip()
        if not text:
            return None
        try:
            pred_fit = self._safe_eval_formula_array(text, X_fit)
            pred_val = self._safe_eval_formula_array(text, X_val)
        except Exception:
            return None

        y_fit = np.asarray(y_fit, dtype=np.float64).reshape(-1)
        y_val = np.asarray(y_val, dtype=np.float64).reshape(-1)
        pred_fit = np.asarray(pred_fit, dtype=np.float64).reshape(-1)
        pred_val = np.asarray(pred_val, dtype=np.float64).reshape(-1)
        if pred_fit.shape != y_fit.shape or pred_val.shape != y_val.shape:
            return None

        w_fit = None if fit_weights is None else np.asarray(fit_weights, dtype=np.float64).reshape(-1)
        w_val = None if val_weights is None else np.asarray(val_weights, dtype=np.float64).reshape(-1)
        if w_fit is not None and w_fit.shape != y_fit.shape:
            raise ValueError(
                f"fit_weights length {w_fit.shape[0]} does not match y_fit length {y_fit.shape[0]}"
            )
        if w_val is not None and w_val.shape != y_val.shape:
            raise ValueError(
                f"val_weights length {w_val.shape[0]} does not match y_val length {y_val.shape[0]}"
            )

        fit_mask = np.isfinite(pred_fit) & np.isfinite(y_fit)
        val_mask = np.isfinite(pred_val) & np.isfinite(y_val)
        if w_fit is not None:
            fit_mask = fit_mask & np.isfinite(w_fit) & (w_fit >= 0)
        if w_val is not None:
            val_mask = val_mask & np.isfinite(w_val) & (w_val >= 0)
        if int(fit_mask.sum()) < 8 or int(val_mask.sum()) < 4:
            return None

        x_fit = pred_fit[fit_mask]
        t_fit = y_fit[fit_mask]
        x_val = pred_val[val_mask]
        t_val = y_val[val_mask]
        wf = None if w_fit is None else w_fit[fit_mask]
        wv = None if w_val is None else w_val[val_mask]
        if wf is not None and float(np.sum(wf)) <= 0:
            return None
        if wv is not None and float(np.sum(wv)) <= 0:
            return None

        try:
            if wf is None:
                coef, _, _, _ = np.linalg.lstsq(
                    np.column_stack([x_fit, np.ones_like(x_fit)]),
                    t_fit,
                    rcond=None,
                )
            else:
                sw = np.sqrt(np.maximum(wf, 0.0))
                A = np.column_stack([x_fit, np.ones_like(x_fit)]) * sw[:, None]
                coef, _, _, _ = np.linalg.lstsq(A, t_fit * sw, rcond=None)
            scale = float(coef[0])
            bias = float(coef[1])
            fit_pred = scale * x_fit + bias
            val_pred = scale * x_val + bias
        except Exception:
            return None

        unweighted_fit_mse = float(np.mean((fit_pred - t_fit) ** 2))
        unweighted_val_mse = float(np.mean((val_pred - t_val) ** 2))
        if not np.isfinite(unweighted_fit_mse) or not np.isfinite(unweighted_val_mse):
            return None
        val_var_u = float(np.var(t_val))
        unweighted_r2 = (
            1.0 if val_var_u < 1e-15 and unweighted_val_mse < 1e-15
            else (0.0 if val_var_u < 1e-15 else 1.0 - unweighted_val_mse / val_var_u)
        )

        weighted_fit_mse = None
        weighted_val_mse = None
        weighted_r2 = None
        if wf is not None:
            weighted_fit_mse = float(np.sum(wf * (fit_pred - t_fit) ** 2) / float(np.sum(wf)))
        if wv is not None:
            weighted_val_mse = float(np.sum(wv * (val_pred - t_val) ** 2) / float(np.sum(wv)))
            mean_t = float(np.sum(wv * t_val) / float(np.sum(wv)))
            val_var_w = float(np.sum(wv * (t_val - mean_t) ** 2) / float(np.sum(wv)))
            weighted_r2 = (
                1.0 if val_var_w < 1e-15 and weighted_val_mse < 1e-15
                else (0.0 if val_var_w < 1e-15 else 1.0 - weighted_val_mse / val_var_w)
            )

        # Search objective may use robust loss (Phase 4); keep plain MSE diagnostics.
        loss_kw = self._search_loss_kwargs()
        search_fit = _robust_loss(fit_pred, t_fit, sample_weight=wf, **loss_kw)
        search_val = _robust_loss(val_pred, t_val, sample_weight=wv, **loss_kw)
        if not np.isfinite(search_fit) or not np.isfinite(search_val):
            return None

        fit_mse = float(search_fit)
        val_mse = float(search_val)
        # R² stays unweighted/weighted MSE-based for interpretability
        val_r2 = weighted_r2 if weighted_r2 is not None else unweighted_r2

        complexity = max(
            1,
            text.count("+") + text.count("-") + text.count("*")
            + text.count("/") + text.count("^") + 1,
        )
        refined_formula = text
        if abs(scale - 1.0) > 1e-8 or abs(bias) > 1e-8:
            refined_formula = f"(({scale:.12g})*({text})+({bias:.12g}))"

        risk_score = self._formula_risk_score(refined_formula, X_val)
        if wv is None:
            gap_denom = max(float(np.var(t_val)), 1e-12)
        else:
            mean_t = float(np.sum(wv * t_val) / float(np.sum(wv)))
            gap_denom = max(
                float(np.sum(wv * (t_val - mean_t) ** 2) / float(np.sum(wv))),
                1e-12,
            )
        generalization_gap = float(max(0.0, val_mse - fit_mse) / gap_denom)

        return {
            "formula": refined_formula,
            "base_formula": text,
            "fit_mse": fit_mse,
            "mse": val_mse,
            "r2": float(val_r2),
            "unweighted_fit_mse": unweighted_fit_mse,
            "unweighted_validation_mse": unweighted_val_mse,
            "unweighted_r2": float(unweighted_r2),
            "weighted_fit_mse": weighted_fit_mse,
            "weighted_validation_mse": weighted_val_mse,
            "weighted_r2": None if weighted_r2 is None else float(weighted_r2),
            "weighted": bool(wf is not None or wv is not None),
            "loss_mode": str(loss_kw.get("loss_mode", "mse")),
            "search_fit_loss": fit_mse,
            "search_validation_loss": val_mse,
            "scale": scale,
            "bias": bias,
            "complexity": complexity,
            "risk_score": risk_score,
            "generalization_gap": generalization_gap,
        }

    def _compute_specialist_screening_diagnostics(self, candidate_formulas, X, y, *, max_candidates=6, max_pairs=5):
        """Summarize coarse segment behavior and pair complementarity for top candidates."""
        import time as _time
        _phase_start = _time.time()
        try:
            state = compute_specialist_state(
                candidate_formulas,
                X,
                y,
                evaluate_formula=self._safe_eval_formula_array,
                complexity_fn=self._formula_complexity,
                family_signature_fn=self._formula_family_signature,
                max_candidates=max_candidates,
                max_pairs=max_pairs,
            )
        finally:
            self._add_phase_time("specialist_diagnostics", _time.time() - _phase_start)
        if state is None:
            self.specialist_state_ = None
            return None
        self.specialist_state_ = state
        return state.to_dict()

    def _compose_specialist_candidates(self, candidate_formulas, X, y, *, max_candidates=12):
        """Generate and validate a tiny set of specialist-driven formula compositions."""
        import time as _time
        _phase_start = _time.time()
        state = getattr(self, "specialist_state_", None)
        if state is None:
            return []

        try:
            proposals = propose_specialist_compositions(
                state,
                X,
                y,
                evaluate_formula=self._safe_eval_formula_array,
                max_pairs=3,
                min_complementarity=0.30,
            )
            if not proposals:
                return []

            raw_candidates = [proposal.to_candidate_dict() for proposal in proposals]
            refined = self._refine_candidate_formulas(
                raw_candidates,
                X,
                y,
                max_candidates=max(4, int(max_candidates)),
            )
            if not refined:
                return []

            accepted = []
            seen = set()
            for candidate in refined:
                formula = str((candidate or {}).get("formula", "")).strip()
                if not formula:
                    continue
                val_r2 = _finite_float(candidate.get("validation_r2"), -1.0)
                complexity = int(candidate.get("complexity") or self._formula_complexity(formula))
                risk = _finite_float(candidate.get("risk_score"), 1.0)
                gap = _finite_float(candidate.get("generalization_gap"), 1.0)
                key = re.sub(r"\s+", "", formula.lower())
                if key in seen:
                    continue
                if val_r2 < 0.70 or complexity > 40 or risk > 0.55 or gap > 0.90:
                    continue
                seen.add(key)
                accepted.append(candidate)

            if isinstance(self.blackbox_diagnostics_, dict):
                self.blackbox_diagnostics_["specialist_composition_screening"] = {
                    "proposal_count": len(raw_candidates),
                    "accepted_count": len(accepted),
                    "top_proposals": [
                        {
                            "formula": str(candidate.get("formula", ""))[:160],
                            "validation_r2": candidate.get("validation_r2"),
                            "validation_mse": candidate.get("validation_mse"),
                            "complexity": candidate.get("complexity"),
                            "operator": candidate.get("composition_operator"),
                        }
                        for candidate in accepted[:6]
                    ],
                }
            if accepted:
                self.composition_candidates_accepted_ = True
                self.composition_candidate_count_ = int(getattr(self, "composition_candidate_count_", 0) or 0) + len(accepted)
            return accepted
        finally:
            self._add_phase_time("specialist_composition", _time.time() - _phase_start)

    def _specialist_vault_enabled(self):
        return (
            bool(getattr(self, "enable_specialist_vault_memory", True))
            and int(getattr(self, "specialist_vault_size", 0) or 0) > 0
            and bool(getattr(self, "enable_specialist_composition_screening", True))
        )

    def _vault_seed_candidates_for_run(self, candidate_formulas, X, y, best_formula, best_mse, run_idx, *, max_candidates=8):
        """Return per-run candidate list augmented with Phase 8 vault memory."""
        base_candidates = list(candidate_formulas or [])
        vault = getattr(self, "specialist_vault_", None)
        if run_idx <= 0 or not self._specialist_vault_enabled() or vault is None or not vault.entries:
            return base_candidates

        vault.rescore_against_target(
            X,
            y,
            evaluate_formula=self._safe_eval_formula_array,
        )
        current_best_candidate = None
        if best_formula:
            best_r2 = None
            try:
                best_pred = self._safe_eval_formula_array(best_formula, X)
                best_pred = np.asarray(best_pred, dtype=np.float64).reshape(-1)
                y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
                if best_pred.shape == y_arr.shape:
                    var_y = float(np.var(y_arr))
                    mse_y = float(np.mean((best_pred - y_arr) ** 2))
                    best_r2 = 1.0 if var_y < 1e-15 and mse_y < 1e-15 else (0.0 if var_y < 1e-15 else 1.0 - mse_y / var_y)
            except Exception:
                best_r2 = None
            current_best_candidate = {
                "formula": best_formula,
                "mse": best_mse,
                "validation_mse": best_mse,
                "validation_r2": best_r2,
                "complexity": self._formula_complexity(best_formula),
                "source": "current_best",
            }
        vault_candidates = vault.candidate_dicts()
        vault_compositions = vault.propose_compositions(
            X,
            y,
            evaluate_formula=self._safe_eval_formula_array,
            complexity_fn=self._formula_complexity,
            family_signature_fn=self._formula_family_signature,
            current_best_candidate=current_best_candidate,
            max_candidates=max(4, int(max_candidates)),
        )
        if vault_compositions:
            refined = self._refine_candidate_formulas(
                vault_compositions,
                X,
                y,
                max_candidates=min(6, max(2, int(max_candidates))),
            )
            vault_compositions = []
            for candidate in refined:
                candidate = dict(candidate)
                candidate["source"] = "specialist_vault_composition"
                candidate["from_specialist_vault"] = True
                candidate["from_specialist_composition"] = True
                vault_compositions.append(candidate)

        if vault_candidates or vault_compositions:
            self.has_composed_seeds_ = bool(vault_compositions) or self.has_composed_seeds_
            self.composition_seeded_evolution_ = bool(vault_compositions) or self.composition_seeded_evolution_
            combined = list(vault_compositions) + list(vault_candidates) + base_candidates
            return self._prune_blackbox_candidate_formulas(
                combined,
                max_candidates=max(8, int(max_candidates)),
            )
        return base_candidates

    def _update_specialist_vault_after_run(self, candidate_formulas, X, y, run_idx, current_best_formula, run_formula=None, run_mse=None):
        """Store structurally different useful formulas after a multi-start attempt."""
        if not self._specialist_vault_enabled():
            return 0
        vault = getattr(self, "specialist_vault_", None)
        if vault is None:
            return 0
        candidates = []
        if run_formula:
            candidates.append({
                "formula": run_formula,
                "mse": run_mse,
                "validation_mse": run_mse,
                "validation_r2": None,
                "complexity": self._formula_complexity(run_formula),
                "source": "evolution_run",
            })
        candidates.extend(list(candidate_formulas or [])[:8])
        added = vault.add_candidates(
            candidates,
            X,
            y,
            evaluate_formula=self._safe_eval_formula_array,
            complexity_fn=self._formula_complexity,
            family_signature_fn=self._formula_family_signature,
            run_index=int(run_idx),
            current_best_formula=current_best_formula,
            max_new=3,
        )
        if isinstance(self.blackbox_diagnostics_, dict):
            self.blackbox_diagnostics_["specialist_vault"] = vault.to_dict()
        return int(added)

    def _r2_score_from_prediction(self, y, pred):
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        pred_arr = np.asarray(pred, dtype=np.float64).reshape(-1)
        if pred_arr.shape != y_arr.shape:
            return -float("inf")
        mse = float(np.mean((pred_arr - y_arr) ** 2))
        var = float(np.var(y_arr))
        if var < 1e-15:
            return 1.0 if mse < 1e-15 else 0.0
        return float(1.0 - mse / var)

    def _extract_frozen_subexpressions(self, formula, X, y, *, max_subexpressions=3):
        """Extract useful non-trivial subexpressions from a formula for Phase 9."""
        text = str(formula or "").strip()
        if not text:
            return []
        try:
            from glassbox.sr.cpp.seed_graph_builder import _parse_formula_expr
            import sympy as sp
        except Exception:
            return []
        expr = _parse_formula_expr(text)
        if expr is None:
            return []

        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        seen = set()
        extracted = []
        for node in sp.preorder_traversal(expr):
            if node == expr or getattr(node, "is_Atom", False):
                continue
            node_text = str(node).replace("**", "^")
            key = "".join(node_text.lower().split())
            if key in seen:
                continue
            seen.add(key)
            complexity = self._formula_complexity(node_text)
            if complexity < 2:
                continue
            try:
                values = self._safe_eval_formula_array(node_text, X_arr).reshape(-1)
            except Exception:
                continue
            if values.shape != y_arr.shape or not np.all(np.isfinite(values)):
                continue
            if float(np.std(values)) < 1e-8:
                continue
            # Standalone signal check after affine fit.
            try:
                coef, _, _, _ = np.linalg.lstsq(
                    np.column_stack([values, np.ones_like(values)]),
                    y_arr,
                    rcond=None,
                )
                pred = coef[0] * values + coef[1]
                standalone_r2 = self._r2_score_from_prediction(y_arr, pred)
            except Exception:
                standalone_r2 = -float("inf")
            if standalone_r2 < 0.30:
                continue
            duplicate = False
            for item in extracted:
                try:
                    corr = abs(float(np.corrcoef(values, item["values"])[0, 1]))
                except Exception:
                    corr = 0.0
                if np.isfinite(corr) and corr > 0.995:
                    duplicate = True
                    break
            if duplicate:
                continue
            extracted.append({
                "formula": node_text,
                "values": values,
                "standalone_r2": float(standalone_r2),
                "complexity": complexity,
            })

        extracted.sort(key=lambda item: (-item["standalone_r2"], item["complexity"], item["formula"]))
        return extracted[: max(0, int(max_subexpressions))]

    @staticmethod
    def _substitute_frozen_features(formula, frozen_formulas, base_feature_count):
        text = str(formula or "")
        for offset, frozen in enumerate(frozen_formulas):
            feature_name = f"x{int(base_feature_count) + int(offset)}"
            text = re.sub(rf"\b{re.escape(feature_name)}\b", f"({frozen})", text)
        return text

    def _build_inception_basis_pool(self, base_feature_count, frozen_formulas):
        pool = [f"x{i}" for i in range(int(base_feature_count))]
        for offset, _ in enumerate(frozen_formulas):
            idx = int(base_feature_count) + int(offset)
            pool.extend([
                f"sin(x{idx})",
                f"cos(x{idx})",
                f"exp(-abs(x{idx}))",
                f"x{idx}",
                f"x{idx}^2",
            ])
            for base_idx in range(int(base_feature_count)):
                pool.append(f"x{base_idx}*x{idx}")
        return list(dict.fromkeys(pool))[:48]

    def _run_inception_reuse(self, X, y, base_formula):
        """Phase 9: freeze useful subexpressions as temporary features and refit a compact basis."""
        import time as _time
        _phase_start = _time.time()
        try:
            return self._run_inception_reuse_impl(X, y, base_formula)
        finally:
            self._add_phase_time("inception_reuse", _time.time() - _phase_start)

    def _run_inception_reuse_impl(self, X, y, base_formula):
        """Implementation for _run_inception_reuse with timing wrapper."""
        self.inception_rounds_ = []
        self.inception_diagnostics_ = {
            "enabled": bool(getattr(self, "enable_inception_reuse", True)),
            "accepted_rounds": 0,
            "attempted_rounds": 0,
        }
        if (
            not getattr(self, "enable_inception_reuse", True)
            or int(getattr(self, "max_inception_rounds", 0) or 0) <= 0
            or int(getattr(self, "max_frozen_subexpressions", 0) or 0) <= 0
            or not base_formula
        ):
            return base_formula

        X_base = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        if X_base.ndim != 2 or X_base.shape[0] != y_arr.shape[0]:
            return base_formula

        current_formula = str(base_formula)
        try:
            current_pred = self._safe_eval_formula_array(current_formula, X_base).reshape(-1)
            current_mse = float(np.mean((current_pred - y_arr) ** 2))
        except Exception:
            return base_formula

        base_feature_count = int(X_base.shape[1])
        rounds = int(getattr(self, "max_inception_rounds", 2) or 0)
        for round_idx in range(rounds):
            frozen = self._extract_frozen_subexpressions(
                current_formula,
                X_base,
                y_arr,
                max_subexpressions=int(getattr(self, "max_frozen_subexpressions", 3) or 0),
            )
            self.inception_diagnostics_["attempted_rounds"] = round_idx + 1
            if not frozen:
                break

            X_aug = np.column_stack([X_base] + [item["values"] for item in frozen])
            frozen_formulas = [item["formula"] for item in frozen]
            pool = self._build_inception_basis_pool(base_feature_count, frozen_formulas)
            prior_n_features = getattr(self, "n_features_in_", base_feature_count)
            try:
                self.n_features_in_ = X_aug.shape[1]
                candidate = self._fit_blackbox_basis_model(
                    X_aug,
                    y_arr,
                    pool,
                    max_terms=max(2, min(5, base_feature_count + len(frozen))),
                )
            finally:
                self.n_features_in_ = prior_n_features
            if not candidate or not candidate.get("formula"):
                break

            expanded = self._substitute_frozen_features(
                candidate["formula"],
                frozen_formulas,
                base_feature_count,
            )
            try:
                expanded = self._cleanup_formula_with_fidelity_guard(
                    expanded,
                    X_base,
                    y_arr,
                    stage="inception_reuse",
                )
                pred = self._safe_eval_formula_array(expanded, X_base).reshape(-1)
                mse = float(np.mean((pred - y_arr) ** 2))
            except Exception:
                break
            if not np.isfinite(mse):
                break
            improvement = current_mse - mse
            if improvement <= max(1e-9, 0.01 * max(current_mse, 1e-9)):
                break

            round_info = {
                "round": int(round_idx),
                "frozen_subexpressions": [
                    {
                        "formula": item["formula"],
                        "standalone_r2": item["standalone_r2"],
                        "complexity": item["complexity"],
                    }
                    for item in frozen
                ],
                "candidate_formula": candidate["formula"],
                "expanded_formula": expanded,
                "previous_mse": float(current_mse),
                "mse": float(mse),
                "improvement": float(improvement),
            }
            self.inception_rounds_.append(round_info)
            self.inception_diagnostics_["accepted_rounds"] = len(self.inception_rounds_)
            current_formula = expanded
            current_mse = mse

            if improvement < 0.01 * max(current_mse, 1e-9):
                break

        self.inception_diagnostics_["final_formula"] = current_formula
        self.inception_diagnostics_["final_mse"] = float(current_mse)
        return current_formula

    def _active_sample_weight(self, n_targets=None, indices=None, sample_weight=None):
        """Return weights for a scoring call, sliced/validated or ``None``.

        Prefer explicit ``sample_weight``; otherwise use ``sample_weight_`` when
        provided at fit time. Length mismatches raise ``ValueError``.
        """
        if sample_weight is not None:
            return _slice_sample_weight(sample_weight, indices=indices, n_targets=n_targets)
        if not getattr(self, "sample_weight_provided_", False):
            return None
        stored = getattr(self, "sample_weight_", None)
        if stored is None:
            return None
        return _slice_sample_weight(stored, indices=indices, n_targets=n_targets)

    def _activate_physics_units(self, n_features):
        """Validate constructor units and set fit-time unit state (Phase 5).

        Units are optional. When omitted or ``unit_mode='off'``, behaviour matches
        tabular ML (no dimensional penalties). When provided, enables
        physics-constrained SR via C++ ``dim_penalty_weight`` and candidate filters.
        """
        mode = _validate_unit_mode(getattr(self, "unit_mode", "off"))
        raw_in = getattr(self, "input_units", None)
        raw_out = getattr(self, "output_units", None)
        self.input_units_ = None
        self.output_units_ = None
        self.units_active_ = False
        self.physics_constrained_ = False
        # Auto-enable soft mode when user supplies units but left unit_mode at default off.
        if mode == "off" and raw_in is None and raw_out is None:
            return
        if mode == "off" and (raw_in is not None or raw_out is not None):
            mode = "soft"
            self.unit_mode = mode
        if mode == "off":
            return
        parsed_in, parsed_out = _validate_physics_units(raw_in, raw_out, n_features)
        if not parsed_in:
            return
        self.input_units_ = parsed_in
        self.output_units_ = parsed_out
        self.units_active_ = True
        self.physics_constrained_ = True

    def _evolution_units_kwargs(self):
        """Kwargs for C++ run_evolution dimensional analysis (empty when inactive).

        Soft mode floors a near-default ``dim_penalty_weight`` (0.1) to 2.0 so
        units actually compete with noisy MSE without requiring users to tune.
        Explicit user values above the floor are kept. Hard mode floors at 10.
        """
        if not getattr(self, "units_active_", False):
            return {}
        iu = getattr(self, "input_units_", None)
        ou = getattr(self, "output_units_", None)
        if not iu:
            return {}
        weight = float(getattr(self, "dim_penalty_weight", 0.1) or 0.0)
        mode = _validate_unit_mode(getattr(self, "unit_mode", "soft"))
        if mode == "hard":
            weight = max(weight, 10.0)
        else:
            # Evidence: weight≈1 weak, weight≈10 recovers physical v=L/t; soft floor 2.0.
            if weight <= 0.1000001:
                weight = 2.0
            else:
                weight = max(weight, 0.5)
        return {
            "input_units": [list(row) for row in iu],
            "output_units": list(ou) if ou is not None else [],
            "dim_penalty_weight": weight,
        }

    def _filter_candidates_by_units(self, candidate_formulas, *, max_candidates=None):
        """Filter/annotate candidates with dimensional compatibility (Phase 5).

        hard mode drops incompatible formulas when units infer successfully.
        soft mode keeps all but sorts physical formulas first and records penalty.
        Never applies penalties when inference is unsafe.
        """
        if not candidate_formulas:
            return []
        if not getattr(self, "units_active_", False):
            return list(candidate_formulas)
        iu = getattr(self, "input_units_", None)
        ou = getattr(self, "output_units_", None)
        mode = _validate_unit_mode(getattr(self, "unit_mode", "soft"))
        kept = []
        rejected = []
        for cand in candidate_formulas:
            formula = str((cand or {}).get("formula", "")).strip()
            if not formula:
                continue
            ok, info = _formula_unit_compatible(formula, iu, ou, unit_mode=mode)
            merged = dict(cand)
            merged["unit_penalty"] = float(info.get("penalty") or 0.0)
            merged["unit_ok"] = bool(info.get("ok"))
            merged["unit_reason"] = info.get("reason")
            if ok:
                kept.append(merged)
            else:
                rejected.append({
                    "formula": formula[:160],
                    "unit_penalty": merged["unit_penalty"],
                    "reason": info.get("reason"),
                })
        if mode == "soft":
            kept.sort(
                key=lambda c: (
                    float(c.get("unit_penalty") or 0.0) if c.get("unit_ok") else 1e3,
                    _finite_float(c.get("mse"), float("inf")),
                    _finite_float(c.get("complexity"), float("inf")),
                )
            )
        if max_candidates is not None:
            kept = kept[: max(1, int(max_candidates))]
        if isinstance(getattr(self, "blackbox_diagnostics_", None), dict):
            self.blackbox_diagnostics_["unit_filter"] = {
                "mode": mode,
                "active": True,
                "kept": len(kept),
                "rejected": len(rejected),
                "rejected_examples": rejected[:6],
            }
        return kept

    def _search_loss_kwargs(self):
        """Kwargs for robust search loss (Phase 4). Display path ignores these."""
        return {
            "loss_mode": getattr(self, "loss_mode", "mse") or "mse",
            "delta": getattr(self, "huber_delta", None),
            "trim_fraction": float(getattr(self, "trim_fraction", 0.1) or 0.1),
        }

    def _formula_mse(self, formula, X, y, sample_weight=None, sample_weight_indices=None):
        """Evaluate a formula for *search* scoring (robust loss when configured).

        Display / benchmark metrics use ``_display_formula_mse`` (plain MSE).
        When fit-time weights are active, they are applied if lengths match.
        If ``sample_weight_indices`` is provided, fit-time weights are sliced to
        those rows (for holdout/subset scoring). Length mismatches raise.
        """
        text = str(formula or "").strip()
        if not text:
            return float("inf")
        try:
            pred = self._safe_eval_formula_array(text, X)
        except Exception:
            return float("inf")
        pred = np.asarray(pred, dtype=np.float64).reshape(-1)
        target = np.asarray(y, dtype=np.float64).reshape(-1)
        if pred.shape != target.shape:
            return float("inf")
        if not np.all(np.isfinite(pred)):
            return float("inf")
        w = self._active_sample_weight(
            n_targets=target.shape[0] if sample_weight_indices is None else None,
            indices=sample_weight_indices,
            sample_weight=sample_weight,
        )
        if w is not None and w.shape[0] != target.shape[0]:
            raise ValueError(
                f"sample_weight length {w.shape[0]} does not match target length {target.shape[0]}"
            )
        loss = _robust_loss(pred, target, sample_weight=w, **self._search_loss_kwargs())
        return loss if np.isfinite(loss) else float("inf")

    def _display_formula_mse(self, formula, X, y):
        """Evaluate a formula with the shared benchmark/display evaluator."""
        text = str(formula or "").strip()
        if not text:
            return float("inf")
        try:
            from scripts import benchmark_common as bc
            mse = bc.evaluate_formula_mse_on_X(text, X, y)
        except Exception:
            return float("inf")
        if mse is None:
            return float("inf")
        try:
            mse = float(mse)
        except Exception:
            return float("inf")
        return mse if np.isfinite(mse) else float("inf")

    def _final_formula_score(self, formula, X, y, sample_weight=None, sample_weight_indices=None):
        """Return the display-first score plus internal/display diagnostics.

        Display MSE remains unweighted (benchmark display contract). Internal
        MSE honours fit-time / explicit weights, optionally sliced by indices.
        """
        internal_mse = self._formula_mse(
            formula,
            X,
            y,
            sample_weight=sample_weight,
            sample_weight_indices=sample_weight_indices,
        )
        display_mse = self._display_formula_mse(formula, X, y)
        score = display_mse if np.isfinite(display_mse) else internal_mse
        return score, internal_mse, display_mse

    def _noise_aware_cleanup_slack(self, formula, X, y, *, relative_slack=None, absolute_slack=None):
        """Compute cleanup slack from residual scale + validation variance (Phase 6).

        Clean data keeps tight slack; high residual scale / validation gap
        widens relative slack so BIC pruning can drop noise terms without
        failing the fidelity guard. Returns ``(rel, abs, diag)``.
        """
        base_rel = 0.10 if relative_slack is None else float(relative_slack)
        base_abs = 1e-9 if absolute_slack is None else float(absolute_slack)
        diag = {
            "base_relative_slack": base_rel,
            "base_absolute_slack": base_abs,
            "relative_slack": base_rel,
            "absolute_slack": base_abs,
        }
        try:
            pred = self._safe_eval_formula_array(formula, X)
            pred = np.asarray(pred, dtype=np.float64).reshape(-1)
            target = np.asarray(y, dtype=np.float64).reshape(-1)
            if pred.shape != target.shape or not np.all(np.isfinite(pred)):
                return base_rel, base_abs, diag
            resid = pred - target
            w = self._active_sample_weight(n_targets=target.shape[0])
            # Near-zero residuals → treat as clean (do not use MAD floor=1.0 as noise).
            resid_rms = float(np.sqrt(np.mean(resid ** 2))) if resid.size else 0.0
            if not np.isfinite(resid_rms) or resid_rms < 1e-14:
                scale = 0.0
            else:
                scale = float(_mad_scale(resid, w))
                # If MAD floor inflated a near-perfect fit, prefer rms.
                if scale > 10.0 * max(resid_rms, 1e-15) and resid_rms < 1e-6:
                    scale = resid_rms
            y_scale = float(np.std(target)) if target.size else 1.0
            if not np.isfinite(y_scale) or y_scale < 1e-12:
                y_scale = max(float(np.mean(np.abs(target))), 1e-12)
            noise_ratio = scale / y_scale if y_scale > 0 else 0.0
            # Holdout generalization gap (unweighted display contract on val).
            gap_ratio = 0.0
            try:
                split = self._domain_edge_validation_split(X, y, validation_fraction=0.2)
            except Exception:
                split = None
            if split is not None:
                try:
                    fit_pred = self._safe_eval_formula_array(formula, split["X_fit"])
                    val_pred = self._safe_eval_formula_array(formula, split["X_val"])
                    fit_mse = float(np.mean((fit_pred - split["y_fit"]) ** 2))
                    val_mse = float(np.mean((val_pred - split["y_val"]) ** 2))
                    y_var = max(float(np.var(split["y_val"])), 1e-12)
                    if np.isfinite(fit_mse) and np.isfinite(val_mse):
                        gap_ratio = float(max(0.0, val_mse - fit_mse) / y_var)
                except Exception:
                    gap_ratio = 0.0
            # Map noise/gap into [0.05, 0.35] relative slack (do not use one value for clean vs 10%).
            rel = base_rel + 0.20 * min(max(noise_ratio, 0.0), 1.0) + 0.10 * min(max(gap_ratio, 0.0), 1.0)
            rel = float(min(max(rel, 0.05), 0.35))
            abs_slack = max(base_abs, 1e-12 * (y_scale ** 2), 1e-6 * (scale ** 2))
            diag.update({
                "residual_mad_scale": scale,
                "y_scale": y_scale,
                "noise_ratio": float(noise_ratio),
                "generalization_gap_ratio": float(gap_ratio),
                "relative_slack": rel,
                "absolute_slack": float(abs_slack),
            })
            return rel, float(abs_slack), diag
        except Exception:
            return base_rel, base_abs, diag

    def _cleanup_formula_with_fidelity_guard(
        self,
        formula,
        X,
        y,
        *,
        stage="final_cleanup",
        relative_slack=None,
        absolute_slack=None,
    ):
        """Apply formula cleanup only when evaluation preserves fit (noise-aware slack)."""
        current = str(formula or "").strip()
        if not current:
            return formula

        rel_slack, abs_slack, slack_diag = self._noise_aware_cleanup_slack(
            current, X, y, relative_slack=relative_slack, absolute_slack=absolute_slack
        )
        current_mse = self._formula_mse(current, X, y)
        current_display_mse = self._display_formula_mse(current, X, y)
        diagnostics = []
        rejected_reasons = []
        noise_pruned_terms = 0
        terms_before = max(1, current.count("+") + current.count("-") // 2 + 1)

        def _accepts(candidate_mse, candidate_display_mse):
            if not np.isfinite(candidate_mse) and not np.isfinite(candidate_display_mse):
                return False, "non_finite_candidate"
            if np.isfinite(current_mse) and np.isfinite(candidate_mse):
                internal_allowed = current_mse * (1.0 + max(0.0, float(rel_slack))) + max(0.0, float(abs_slack))
                if candidate_mse > internal_allowed:
                    return False, "internal_mse_regression"
            elif np.isfinite(current_mse):
                return False, "internal_mse_non_finite"

            if np.isfinite(current_display_mse) and np.isfinite(candidate_display_mse):
                display_allowed = current_display_mse * (1.0 + max(0.0, float(rel_slack))) + max(0.0, float(abs_slack))
                if candidate_display_mse > display_allowed:
                    return False, "display_mse_regression"
            elif np.isfinite(current_display_mse):
                return False, "display_mse_non_finite"

            return True, "accepted"

        cleanup_steps = (
            ("reduce_formula_noise", lambda text: self._reduce_formula_noise(text, X, y)),
            ("simplify_formula", self._simplify_formula),
        )
        for step_name, cleanup_fn in cleanup_steps:
            try:
                candidate = str(cleanup_fn(current) or "").strip()
            except Exception:
                candidate = current
            if not candidate or candidate == current:
                continue

            candidate_mse = self._formula_mse(candidate, X, y)
            candidate_display_mse = self._display_formula_mse(candidate, X, y)
            accepted, reason = _accepts(candidate_mse, candidate_display_mse)
            step_diag = {
                "step": step_name,
                "accepted": bool(accepted),
                "reason": reason,
                "before_mse": float(current_mse) if np.isfinite(current_mse) else None,
                "after_mse": float(candidate_mse) if np.isfinite(candidate_mse) else None,
                "before_display_mse": float(current_display_mse) if np.isfinite(current_display_mse) else None,
                "after_display_mse": float(candidate_display_mse) if np.isfinite(candidate_display_mse) else None,
                "relative_slack": float(rel_slack),
                "absolute_slack": float(abs_slack),
            }
            diagnostics.append(step_diag)
            if accepted:
                if step_name == "reduce_formula_noise":
                    terms_after = max(1, candidate.count("+") + candidate.count("-") // 2 + 1)
                    noise_pruned_terms += max(0, terms_before - terms_after)
                    terms_before = terms_after
                current = candidate
                current_mse = candidate_mse
                current_display_mse = candidate_display_mse
            else:
                rejected_reasons.append(f"{step_name}:{reason}")

        if isinstance(getattr(self, "blackbox_diagnostics_", None), dict):
            if diagnostics:
                self.blackbox_diagnostics_.setdefault("formula_cleanup_guard", []).append({
                    "stage": stage,
                    "steps": diagnostics,
                    "noise_aware_slack": slack_diag,
                    "noise_pruned_terms": int(noise_pruned_terms),
                    "cleanup_rejected_reason": rejected_reasons[-1] if rejected_reasons else None,
                })
            if noise_pruned_terms:
                prev = int(self.blackbox_diagnostics_.get("noise_pruned_terms", 0) or 0)
                self.blackbox_diagnostics_["noise_pruned_terms"] = prev + int(noise_pruned_terms)
            if rejected_reasons:
                self.blackbox_diagnostics_["cleanup_rejected_reason"] = rejected_reasons[-1]
        return current

    def _candidate_pool_has_actionable_fit(self, candidate_formulas, incumbent_mse, search_plan=None):
        """Return true when screened candidates are good enough to keep/use."""
        if not candidate_formulas:
            return False
        best_candidate_mse = float("inf")
        best_candidate_r2 = -float("inf")
        for cand in candidate_formulas or []:
            cand_mse = min(
                _finite_float((cand or {}).get("mse"), float("inf")),
                _finite_float((cand or {}).get("validation_mse"), float("inf")),
            )
            cand_r2 = _finite_float((cand or {}).get("validation_r2"), -float("inf"))
            best_candidate_mse = min(best_candidate_mse, cand_mse)
            best_candidate_r2 = max(best_candidate_r2, cand_r2)

        if best_candidate_mse <= max(float(getattr(self, "early_stop_mse", 1e-10)), 1e-10):
            return True
        candidate_acceptance_r2 = _finite_float(
            (search_plan or {}).get("candidate_acceptance_r2"),
            0.985,
        )
        if best_candidate_r2 >= max(candidate_acceptance_r2, min(float(self.evolution_skip_r2), 0.999999)):
            return True
        incumbent = _finite_float(incumbent_mse, float("inf"))
        return np.isfinite(best_candidate_mse) and best_candidate_mse < incumbent

    def _select_final_formula(self, incumbent_formula, incumbent_mse, challenger_formula, challenger_mse, X, y):
        """Choose between incumbent and challenger using direct formula evaluation."""
        incumbent_text = str(incumbent_formula or "").strip()
        challenger_text = str(challenger_formula or "").strip()
        if not challenger_text:
            return incumbent_formula, incumbent_mse, "incumbent"
        if not incumbent_text:
            return challenger_formula, challenger_mse, "challenger"

        incumbent_score, incumbent_eval, incumbent_display = self._final_formula_score(incumbent_text, X, y)
        challenger_score, challenger_eval, challenger_display = self._final_formula_score(challenger_text, X, y)
        if not np.isfinite(incumbent_score):
            incumbent_score = float(incumbent_mse or float("inf"))
        if not np.isfinite(challenger_score):
            challenger_score = float(challenger_mse or float("inf"))

        self.final_formula_selection_diagnostics_ = {
            "incumbent_internal_mse": float(incumbent_eval) if np.isfinite(incumbent_eval) else None,
            "incumbent_display_mse": float(incumbent_display) if np.isfinite(incumbent_display) else None,
            "challenger_internal_mse": float(challenger_eval) if np.isfinite(challenger_eval) else None,
            "challenger_display_mse": float(challenger_display) if np.isfinite(challenger_display) else None,
            "selected_by": "display_first_formula_score",
        }

        if challenger_score + 1e-12 < incumbent_score:
            self.final_formula_selection_diagnostics_["selected"] = "challenger"
            return challenger_formula, challenger_score, "challenger"
        self.final_formula_selection_diagnostics_["selected"] = "incumbent"
        return incumbent_formula, incumbent_score, "incumbent"

    def _final_holdout_scores(self, base_formula, candidate_formula, X, y):
        """Score a final-stage candidate on a deterministic holdout slice."""
        try:
            split = self._domain_edge_validation_split(X, y, validation_fraction=0.25)
        except Exception:
            split = None
        if split is None:
            return None

        val_idx = split.get("val_idx")
        base_score, base_internal, base_display = self._final_formula_score(
            base_formula,
            split["X_val"],
            split["y_val"],
            sample_weight_indices=val_idx,
        )
        candidate_score, candidate_internal, candidate_display = self._final_formula_score(
            candidate_formula,
            split["X_val"],
            split["y_val"],
            sample_weight_indices=val_idx,
        )
        return {
            "base_score": base_score,
            "candidate_score": candidate_score,
            "base_internal_mse": base_internal,
            "candidate_internal_mse": candidate_internal,
            "base_display_mse": base_display,
            "candidate_display_mse": candidate_display,
            "n_val": int(len(split["y_val"])),
        }

    def _restore_user_loss_mode_if_auto_switched(self) -> None:
        """Restore constructor/user loss_mode after Phase 3/4 auto Huber switch."""
        if not getattr(self, "_loss_mode_auto_switched_", False):
            return
        prev = getattr(self, "_user_loss_mode_", None)
        if prev is not None:
            self.loss_mode = prev
        self._loss_mode_auto_switched_ = False

    def _auto_noise_guard_active(self) -> bool:

        """True when Phase-3 auto residual soft-weights were applied (not user weights)."""
        applied = getattr(self, "_blackbox_noise_robust_applied_", None) or {}
        if not isinstance(applied, dict) or not applied.get("active"):
            return False
        diag = getattr(self, "blackbox_diagnostics_", None) or {}
        sw = diag.get("sample_weight") if isinstance(diag, dict) else None
        if isinstance(sw, dict) and str(sw.get("source") or "") == "user":
            return False
        # Explicit auto path, or soft_mad without user source.
        if str(applied.get("reason") or "") == "soft_mad_weights":
            return True
        return str(sw.get("source") if isinstance(sw, dict) else "") == "auto_soft_mad"

    def _unweighted_r2_from_mse(self, mse, y) -> float:
        target = np.asarray(y, dtype=np.float64).reshape(-1)
        if target.size == 0:
            return float("nan")
        y_var = float(np.var(target))
        try:
            mse_f = float(mse)
        except Exception:
            return float("nan")
        if not np.isfinite(mse_f):
            return float("nan")
        if y_var < 1e-15:
            return 1.0 if mse_f < 1e-15 else 0.0
        return float(1.0 - mse_f / y_var)

    def _auto_weight_guard_limits(self, X) -> dict:
        n_feat = int(np.asarray(X).shape[1]) if np.ndim(X) == 2 else 1
        # 1D SR: aggressive cap — Nguyen-1 disaster was complexity 61.
        if n_feat <= 1:
            max_complexity = 22
        elif n_feat <= 4:
            max_complexity = 30
        else:
            max_complexity = 40
        return {
            "max_complexity": int(max_complexity),
            "min_full_r2": 0.50,
            "min_holdout_r2": 0.40,
            "max_gap": 0.45,  # train_r2 - holdout_r2
        }

    def _evaluate_auto_weight_guard(self, formula, X, y, *, limits=None) -> dict:
        """Unweighted safety metrics for a candidate under auto soft-weights."""
        text = str(formula or "").strip()
        limits = dict(limits or self._auto_weight_guard_limits(X))
        out = {
            "formula": text,
            "ok": False,
            "reasons": [],
            "complexity": int(self._formula_complexity(text)) if text else 0,
            "full_mse": float("inf"),
            "full_r2": float("nan"),
            "holdout_mse": float("inf"),
            "holdout_r2": float("nan"),
            "train_r2": float("nan"),
            "gap": float("nan"),
            "limits": limits,
        }
        if not text:
            out["reasons"].append("empty_formula")
            return out

        full_mse = self._display_formula_mse(text, X, y)
        out["full_mse"] = float(full_mse) if np.isfinite(full_mse) else float("inf")
        out["full_r2"] = self._unweighted_r2_from_mse(full_mse, y)

        # Deterministic edge holdout, unweighted (do not use sample_weight).
        try:
            split = self._domain_edge_validation_split(X, y, validation_fraction=0.25)
        except Exception:
            split = None
        if split is not None:
            try:
                train_mse = self._display_formula_mse(text, split["X_fit"], split["y_fit"])
                hold_mse = self._display_formula_mse(text, split["X_val"], split["y_val"])
                out["train_r2"] = self._unweighted_r2_from_mse(train_mse, split["y_fit"])
                out["holdout_mse"] = float(hold_mse) if np.isfinite(hold_mse) else float("inf")
                out["holdout_r2"] = self._unweighted_r2_from_mse(hold_mse, split["y_val"])
                if np.isfinite(out["train_r2"]) and np.isfinite(out["holdout_r2"]):
                    out["gap"] = float(out["train_r2"] - out["holdout_r2"])
            except Exception:
                pass

        reasons = []
        if out["complexity"] > int(limits["max_complexity"]):
            reasons.append("complexity_cap")
        if not np.isfinite(out["full_r2"]) or out["full_r2"] < float(limits["min_full_r2"]):
            reasons.append("full_r2")
        if np.isfinite(out["holdout_r2"]) and out["holdout_r2"] < float(limits["min_holdout_r2"]):
            reasons.append("holdout_r2")
        if np.isfinite(out["gap"]) and out["gap"] > float(limits["max_gap"]):
            reasons.append("generalization_gap")
        out["reasons"] = reasons
        out["ok"] = len(reasons) == 0
        return out

    def _register_auto_weight_fallback_candidate(self, formula, X=None, y=None, *, source="track"):
        """Remember simple formulas seen during fit for auto-weight final rescue."""
        text = str(formula or "").strip()
        if not text:
            return
        if not hasattr(self, "_auto_weight_fallback_candidates_") or self._auto_weight_fallback_candidates_ is None:
            self._auto_weight_fallback_candidates_ = []
        pool = self._auto_weight_fallback_candidates_
        # de-dupe
        for item in pool:
            if str(item.get("formula") or "") == text:
                return
        cx = int(self._formula_complexity(text))
        entry = {"formula": text, "complexity": cx, "source": str(source)}
        if X is not None and y is not None:
            try:
                entry["full_mse"] = float(self._display_formula_mse(text, X, y))
            except Exception:
                pass
        pool.append(entry)
        # keep a small pool of simplest candidates
        pool.sort(key=lambda c: (int(c.get("complexity") or 999), float(c.get("full_mse") or 1e300)))
        del pool[12:]

    def _phase6_noise_parsimony_pass(self, formula, X, y, *, stage="phase6_parsimony"):
        """Phase 6 tighten: under auto soft-MAD, prefer simpler formulas with similar unweighted fit.

        Aggressive reduce/simplify with wider fidelity slack, then keep the simplest
        candidate whose unweighted holdout R² stays within a small gap of the primary.
        """
        text = str(formula or "").strip()
        if not text or not self._auto_noise_guard_active():
            return text or formula

        primary_m = self._evaluate_auto_weight_guard(text, X, y)
        primary_h = primary_m.get("holdout_r2")
        primary_f = primary_m.get("full_r2")
        primary_cx = int(primary_m.get("complexity") or self._formula_complexity(text))
        # Need a finite primary metric; low R² on noisy labels is OK for parsimony.
        if not np.isfinite(primary_f):
            return text

        candidates = [{"formula": text, "source": "primary", "metrics": primary_m}]

        # Wider cleanup slack under noise so BIC/reduce can drop junk terms.
        try:
            cleaned = str(
                self._cleanup_formula_with_fidelity_guard(
                    text,
                    X,
                    y,
                    stage=stage + "_cleanup",
                    relative_slack=0.25,
                    absolute_slack=1e-8,
                )
                or ""
            ).strip()
        except Exception:
            cleaned = ""
        if cleaned and cleaned != text:
            candidates.append({
                "formula": cleaned,
                "source": "aggressive_cleanup",
                "metrics": self._evaluate_auto_weight_guard(cleaned, X, y),
            })
            self._register_auto_weight_fallback_candidate(
                cleaned, X, y, source="phase6_cleanup"
            )

        # Second reduce-only pass with noise-aware (already uses weights).
        try:
            reduced = str(self._reduce_formula_noise(text, X, y) or "").strip()
        except Exception:
            reduced = ""
        if reduced and reduced not in {text, cleaned}:
            candidates.append({
                "formula": reduced,
                "source": "reduce_only",
                "metrics": self._evaluate_auto_weight_guard(reduced, X, y),
            })
            self._register_auto_weight_fallback_candidate(
                reduced, X, y, source="phase6_reduce"
            )

        # Also consider tracked simpler fallbacks.
        for item in list(getattr(self, "_auto_weight_fallback_candidates_", None) or []):
            f = str(item.get("formula") or "").strip()
            if not f:
                continue
            candidates.append({
                "formula": f,
                "source": str(item.get("source") or "fallback"),
                "metrics": self._evaluate_auto_weight_guard(f, X, y),
            })

        # Keep candidates that do not tank unweighted recovery vs primary.
        # Use *relative* thresholds so noisy-label R² can be moderate while still
        # preferring simpler structure (outliers make absolute R² on y_noisy low).
        def _acceptable(m):
            if not m:
                return False
            fr = m.get("full_r2")
            hr = m.get("holdout_r2")
            if not np.isfinite(fr):
                return False
            # Relative to primary full R² (allow tiny regression for simplicity).
            if np.isfinite(primary_f):
                if fr < float(primary_f) - 0.05 and fr < 0.98:
                    return False
            else:
                if fr < 0.5:
                    return False
            if np.isfinite(primary_h) and np.isfinite(hr):
                if hr < float(primary_h) - 0.05 and hr < 0.98:
                    return False
            # Prefer not to promote clearly worse complexity unless R² is much better.
            cx = int(m.get("complexity") or 999)
            if cx > primary_cx + 2:
                return False
            return True

        pool = []
        seen = set()
        for c in candidates:
            f = c["formula"]
            if f in seen:
                continue
            seen.add(f)
            if _acceptable(c["metrics"]):
                pool.append(c)
        if not pool:
            pool = [candidates[0]]

        def _rank(c):
            m = c["metrics"]
            cx = int(m.get("complexity") or 999)
            # Prefer lower complexity, then better holdout, then better full R2.
            hr = m.get("holdout_r2")
            fr = m.get("full_r2")
            hr_key = -float(hr) if hr is not None and np.isfinite(hr) else 0.0
            fr_key = -float(fr) if fr is not None and np.isfinite(fr) else 0.0
            return (cx, hr_key, fr_key)

        pool.sort(key=_rank)
        chosen = pool[0]
        diag = {
            "active": True,
            "stage": stage,
            "primary_complexity": primary_cx,
            "primary_full_r2": primary_f,
            "primary_holdout_r2": primary_h,
            "selected_formula": str(chosen["formula"])[:200],
            "selected_source": chosen.get("source"),
            "selected_complexity": chosen["metrics"].get("complexity"),
            "selected_full_r2": chosen["metrics"].get("full_r2"),
            "selected_holdout_r2": chosen["metrics"].get("holdout_r2"),
            "replaced": str(chosen["formula"]) != text,
            "pool_size": len(pool),
        }
        if isinstance(getattr(self, "blackbox_diagnostics_", None), dict):
            self.blackbox_diagnostics_["phase6_noise_parsimony"] = diag
        return chosen["formula"]

    def _apply_auto_weight_final_guard(self, formula, X, y, *, stage="final_fit"):
        """Block bloated / unweighted-catastrophic formulas when auto soft-weights ran.

        Uses unweighted display MSE/R² and holdout gap (clean y is unavailable at fit).
        Prefer a simpler tracked fallback when the winner fails the guard.
        """
        text = str(formula or "").strip()
        if not text or not self._auto_noise_guard_active():
            return text or formula

        limits = self._auto_weight_guard_limits(X)
        primary = self._evaluate_auto_weight_guard(text, X, y, limits=limits)
        pool = [{"formula": text, "source": "primary", "metrics": primary}]

        for item in list(getattr(self, "_auto_weight_fallback_candidates_", None) or []):
            f = str(item.get("formula") or "").strip()
            if not f or f == text:
                continue
            pool.append({
                "formula": f,
                "source": str(item.get("source") or "fallback"),
                "metrics": self._evaluate_auto_weight_guard(f, X, y, limits=limits),
            })

        evo = str(getattr(self, "evolution_candidate_formula_", "") or "").strip()
        if evo and evo != text:
            pool.append({
                "formula": evo,
                "source": "evolution_candidate",
                "metrics": self._evaluate_auto_weight_guard(evo, X, y, limits=limits),
            })

        # Always consider a cleaned primary (may drop noise terms).
        try:
            cleaned = str(self._cleanup_formula_with_fidelity_guard(
                text, X, y, stage="auto_weight_guard_cleanup"
            ) or "").strip()
        except Exception:
            cleaned = ""
        if cleaned and cleaned != text:
            pool.append({
                "formula": cleaned,
                "source": "cleaned_primary",
                "metrics": self._evaluate_auto_weight_guard(cleaned, X, y, limits=limits),
            })

        def _rank_key(entry):
            m = entry["metrics"]
            ok = 0 if m.get("ok") else 1
            # Prefer finite holdout mse, then full mse, then complexity.
            h = m.get("holdout_mse")
            fmse = m.get("full_mse")
            h = float(h) if h is not None and np.isfinite(h) else 1e300
            fmse = float(fmse) if fmse is not None and np.isfinite(fmse) else 1e300
            cx = int(m.get("complexity") or 999)
            # Soft preference: even among failing, lower complexity + better holdout.
            hard_fail = 0
            reasons = set(m.get("reasons") or [])
            if "full_r2" in reasons or "holdout_r2" in reasons:
                hard_fail = 1
            return (ok, hard_fail, h, fmse, cx)

        pool_unique = []
        seen = set()
        for entry in pool:
            f = entry["formula"]
            if f in seen:
                continue
            seen.add(f)
            pool_unique.append(entry)
        pool_unique.sort(key=_rank_key)
        chosen = pool_unique[0]
        chosen_metrics = chosen["metrics"]
        replaced = chosen["formula"] != text or not primary.get("ok")

        diag = {
            "active": True,
            "stage": stage,
            "primary_ok": bool(primary.get("ok")),
            "primary_reasons": list(primary.get("reasons") or []),
            "primary_complexity": primary.get("complexity"),
            "primary_full_r2": primary.get("full_r2"),
            "primary_holdout_r2": primary.get("holdout_r2"),
            "selected_formula": chosen["formula"][:200],
            "selected_source": chosen.get("source"),
            "selected_ok": bool(chosen_metrics.get("ok")),
            "selected_reasons": list(chosen_metrics.get("reasons") or []),
            "selected_complexity": chosen_metrics.get("complexity"),
            "selected_full_r2": chosen_metrics.get("full_r2"),
            "selected_holdout_r2": chosen_metrics.get("holdout_r2"),
            "replaced": bool(chosen["formula"] != text),
            "pool_size": len(pool_unique),
            "limits": limits,
        }
        if isinstance(getattr(self, "blackbox_diagnostics_", None), dict):
            self.blackbox_diagnostics_["auto_weight_final_guard"] = diag
        self._auto_weight_final_guard_ = diag
        return chosen["formula"]

    def _compare_blackbox_formulas(self, incumbent_formula, challenger_formula, X, y):

        """Compare two formulas on validation, not just in-sample fit."""
        candidates = []
        if incumbent_formula:
            candidates.append({"formula": incumbent_formula, "source": "incumbent"})
        if challenger_formula:
            candidates.append({"formula": challenger_formula, "source": "challenger"})
        if len(candidates) < 2:
            return None
        choice = self._select_blackbox_pareto_formula(candidates, X, y)
        if choice is None:
            return None
        return "challenger" if choice.get("source") == "challenger" else "incumbent"

    def _select_blackbox_pareto_formula(self, candidates, X, y):
        """Select a validation-stable Pareto winner (weighted val + residual diagnostics)."""
        if not candidates:
            return None
        random_split = self._random_blackbox_validation_split(X, y, validation_fraction=0.25, salt=17)
        edge_split = self._domain_edge_validation_split(X, y, validation_fraction=0.25)
        split = random_split or edge_split
        if split is None:
            return None
        y_val = split["y_val"]
        y_var = max(float(np.var(y_val)), 1e-12)
        fit_w, val_w = self._split_sample_weights(split, n_total=int(np.asarray(y).reshape(-1).shape[0]))
        edge_w = None
        if edge_split is not None and edge_split is not split:
            try:
                _, edge_w = self._split_sample_weights(
                    edge_split, n_total=int(np.asarray(y).reshape(-1).shape[0])
                )
            except Exception:
                edge_w = None
        scored = []
        seen = set()
        for candidate in candidates:
            formula = str((candidate or {}).get("formula", "")).strip()
            if not formula:
                continue
            key = re.sub(r"\s+", "", formula.lower())
            if key in seen:
                continue
            seen.add(key)
            try:
                pred_fit = self._safe_eval_formula_array(formula, split["X_fit"])
                pred_val = self._safe_eval_formula_array(formula, split["X_val"])
            except Exception:
                continue
            pred_fit = np.asarray(pred_fit, dtype=np.float64).reshape(-1)
            pred_val = np.asarray(pred_val, dtype=np.float64).reshape(-1)
            if pred_fit.shape != split["y_fit"].shape or pred_val.shape != y_val.shape:
                continue
            if not (np.all(np.isfinite(pred_fit)) and np.all(np.isfinite(pred_val))):
                continue
            # Unweighted diagnostics always retained for display/bench parity.
            fit_mse_u = float(np.mean((pred_fit - split["y_fit"]) ** 2))
            val_mse_u = float(np.mean((pred_val - y_val) ** 2))
            if not np.isfinite(fit_mse_u) or not np.isfinite(val_mse_u):
                continue
            fit_mse = fit_mse_u
            val_mse = val_mse_u
            if fit_w is not None:
                try:
                    fit_mse = _weighted_mse(pred_fit, split["y_fit"], fit_w)
                except Exception:
                    fit_mse = fit_mse_u
            if val_w is not None:
                try:
                    val_mse = _weighted_mse(pred_val, y_val, val_w)
                except Exception:
                    val_mse = val_mse_u
            if not np.isfinite(fit_mse) or not np.isfinite(val_mse):
                continue
            complexity = int((candidate or {}).get("complexity") or self._formula_complexity(formula))
            risk = self._formula_risk_score(formula, split["X_val"])
            gap = float(max(0.0, val_mse - fit_mse) / y_var)
            val_r2 = 1.0 - val_mse_u / y_var
            # Robust residual diagnostics on validation residuals.
            resid_val = pred_val - y_val
            resid_scale = float(_mad_scale(resid_val, val_w))
            resid_std = float(np.std(resid_val)) if resid_val.size else 0.0
            outlier_frac = 0.0
            if resid_scale > 1e-12 and resid_val.size:
                outlier_frac = float(np.mean(np.abs(resid_val) > 3.0 * resid_scale))
            edge_mse = None
            edge_mse_u = None
            edge_r2 = None
            if edge_split is not None and edge_split is not split:
                try:
                    edge_pred = self._safe_eval_formula_array(formula, edge_split["X_val"])
                    edge_pred = np.asarray(edge_pred, dtype=np.float64).reshape(-1)
                    if edge_pred.shape == edge_split["y_val"].shape and np.all(np.isfinite(edge_pred)):
                        edge_mse_u = float(np.mean((edge_pred - edge_split["y_val"]) ** 2))
                        edge_mse = edge_mse_u
                        if edge_w is not None:
                            try:
                                edge_mse = _weighted_mse(edge_pred, edge_split["y_val"], edge_w)
                            except Exception:
                                edge_mse = edge_mse_u
                        edge_var = max(float(np.var(edge_split["y_val"])), 1e-12)
                        edge_r2 = 1.0 - edge_mse_u / edge_var
                except Exception:
                    edge_mse = None
            blend_mse = val_mse
            if edge_mse is not None and np.isfinite(edge_mse):
                blend_mse = 0.72 * val_mse + 0.28 * edge_mse
            # Penalize heavy residual tails / outlier memorization.
            residual_penalty = 0.15 * min(max(outlier_frac, 0.0), 1.0) + 0.05 * min(
                max(resid_scale / max(float(np.std(y_val)), 1e-12), 0.0), 2.0
            )
            n_features_bb = int(np.asarray(X).shape[1]) if X is not None and np.ndim(X) == 2 else 1
            complexity_weight = 0.055 if n_features_bb > 1 else 0.030
            score = blend_mse * (1.0 + complexity_weight * complexity + 0.50 * risk + 0.25 * gap + residual_penalty)
            # Prefer simpler structure when MSE is only modestly worse (Exact recovery).
            if n_features_bb > 1 and complexity > 24:
                score *= 1.0 + 0.015 * (complexity - 24)
            # Mild preference for free-const structure seeds over kitchen-sink when close.
            if n_features_bb > 1 and (
                (candidate or {}).get("from_structure_seed")
                or str((candidate or {}).get("source", "")).startswith("structure_seed")
            ):
                score *= 0.92
            governor = None
            try:
                from scripts import benchmark_common as bc
                governor = bc.score_display_candidate(
                    formula,
                    X,
                    y,
                    raw_mse=(candidate or {}).get("mse", val_mse_u),
                    fit_mse=fit_mse_u,
                    holdout_mse=edge_mse_u,
                    complexity=complexity,
                    postprocess=False,
                )
                governor_score = float(governor.get("score", float("inf")))
                governor_display = governor.get("display_mse")
                if np.isfinite(governor_score):
                    score = 0.82 * score + 0.18 * governor_score
            except Exception:
                governor_score = None
                governor_display = None
            try:
                full_mse_u = float(np.mean((self._safe_eval_formula_array(formula, X) - y) ** 2))
            except Exception:
                full_mse_u = val_mse_u
            scored.append({
                "formula": formula,
                "mse": full_mse_u,
                "validation_mse": val_mse,
                "validation_mse_unweighted": val_mse_u,
                "validation_r2": float(val_r2),
                "edge_validation_mse": edge_mse,
                "edge_validation_mse_unweighted": edge_mse_u,
                "edge_validation_r2": edge_r2,
                "blended_validation_mse": float(blend_mse),
                "fit_mse": fit_mse,
                "fit_mse_unweighted": fit_mse_u,
                "complexity": complexity,
                "risk_score": risk,
                "generalization_gap": gap,
                "residual_mad_scale": resid_scale,
                "residual_std": resid_std,
                "residual_outlier_fraction": outlier_frac,
                "residual_penalty": float(residual_penalty),
                "pareto_score": float(score),
                "display_governor_score": governor_score,
                "display_mse": governor_display,
                "display_governor": governor,
                "source": (candidate or {}).get("source") or (candidate or {}).get("run_label"),
            })
        if not scored:
            return None

        best_raw = min(scored, key=lambda c: c["validation_mse"])
        eligible = [
            c for c in scored
            if c["validation_mse"] <= best_raw["validation_mse"] * 1.08 + 1e-12
            and c["risk_score"] <= max(0.45, best_raw["risk_score"] + 0.15)
        ]
        selected = min(eligible or scored, key=lambda c: (c["pareto_score"], c["complexity"]))
        selected["evaluated_candidates"] = len(scored)
        selected["best_raw_validation_mse"] = best_raw["validation_mse"]
        selected["selected_by"] = "blackbox_validation_pareto"
        selected["weighted_validation"] = bool(val_w is not None)
        return selected

    def _probe_multivariate_structure_original_space(self, X_original, y_original, selected_features):
        """Match Pagie/Vlad/Feynman-like skeletons on original (unstandardized) data.

        Standardization breaks radial/inverse-power templates; probe original space
        before search so Exact recovery is possible under blackbox mode.
        """
        X_all = np.asarray(X_original, dtype=np.float64)
        y_all = np.asarray(y_original, dtype=np.float64).reshape(-1)
        if X_all.ndim != 2 or X_all.shape[1] < 2 or len(selected_features) < 2:
            return None
        try:
            from classifier_fast_path import _maybe_match_easy_multivariate_formula  # type: ignore
        except Exception:
            try:
                from scripts.classifier_fast_path import _maybe_match_easy_multivariate_formula  # type: ignore
            except Exception:
                return None
        cols = [int(i) for i in selected_features]
        if any(i < 0 or i >= X_all.shape[1] for i in cols):
            return None
        X_sel = X_all[:, cols]
        match = _maybe_match_easy_multivariate_formula(X_sel, y_all)
        if match is None:
            return None
        formula_local, mse, details = match
        if not formula_local or not np.isfinite(mse):
            return None
        # Local x0..xk refer to selected columns; map back to original indices.
        formula = remap_reduced_formula_to_original(str(formula_local), cols)
        try:
            pred = self._safe_eval_formula_array(formula, X_all)
            full_mse = float(np.mean((np.asarray(pred, dtype=np.float64).reshape(-1) - y_all) ** 2))
        except Exception:
            full_mse = float(mse)
        if not np.isfinite(full_mse):
            return None
        y_var = max(float(np.var(y_all)), 1e-12)
        r2 = 1.0 - full_mse / y_var
        details = details or {}
        return {
            "formula": formula,
            "mse": full_mse,
            "r2": float(r2),
            "template_match": details.get("template_match"),
            "complexity": self._formula_complexity(formula),
            "exact_match": bool(details.get("exact_match", False) or full_mse <= max(1e-10, 1e-12 * y_var)),
            "robust_match": bool(details.get("robust_match", False)),
            "inlier_fraction": float(details.get("inlier_fraction", 0.0) or 0.0),
            "median_abs_residual": float(details.get("median_abs_residual", full_mse) or full_mse),
        }

    def _validate_blackbox_fast_path_candidate(self, formula, mse, X, y):
        """Decide whether a fast-path formula is safe enough to be incumbent."""
        split = self._domain_edge_validation_split(X, y, validation_fraction=0.25)
        if split is None or not formula:
            return {"accepted": True, "reason": "no_validation_split"}
        scored = self._score_formula_candidate(
            formula,
            split["X_fit"],
            split["y_fit"],
            split["X_val"],
            split["y_val"],
        )
        if scored is None:
            return {"accepted": False, "reason": "validation_failed"}
        val_mse = float(scored.get("validation_mse", scored.get("mse", float("inf"))))
        fit_mse = float(scored.get("fit_mse", float("inf")))
        val_var = max(float(np.var(split["y_val"])), 1e-12)
        risk = float(scored.get("risk_score", self._formula_risk_score(formula, split["X_val"])))
        complexity = self._formula_complexity(formula)
        gap = float(max(0.0, val_mse - fit_mse) / val_var)
        train_ratio = val_mse / max(float(mse), 1e-12) if mse is not None and np.isfinite(mse) else 1.0
        # Multi-var blackbox: reject kitchen-sink fast-path incumbents so
        # structure seeds / evolution can compete on Exact recovery.
        max_complexity = 36
        n_features = int(np.asarray(X).shape[1]) if X is not None and np.ndim(X) == 2 else 1
        if n_features > 1:
            max_complexity = 22
        accepted = (
            np.isfinite(val_mse)
            and val_mse <= 1.25 * val_var
            and train_ratio <= 3.0
            and gap <= 0.75
            and risk <= 0.45
            and complexity <= max_complexity
        )
        reason = "accepted" if accepted else "unstable_validation"
        return {
            "accepted": bool(accepted),
            "reason": reason,
            "validation_mse": val_mse,
            "validation_r2": 1.0 - val_mse / val_var,
            "fit_mse": fit_mse,
            "train_mse": float(mse) if mse is not None and np.isfinite(mse) else None,
            "validation_to_train_mse": float(train_ratio) if np.isfinite(train_ratio) else None,
            "risk_score": risk,
            "generalization_gap": gap,
            "complexity": complexity,
            "candidate_formula": scored.get("formula"),
        }

    def _should_use_universal_fast_path(self, blackbox_state, fast_path_uncertainty):
        """Disable kitchen-sink fast-path expansion when classifier evidence is weak."""
        if blackbox_state is None or not getattr(blackbox_state, "enabled", False):
            return True
        selected = list(getattr(blackbox_state, "selected_features", []) or [])
        if len(selected) <= 1:
            return True
        if not isinstance(fast_path_uncertainty, dict):
            return True
        entropy = _finite_float(fast_path_uncertainty.get("prediction_entropy"), 0.0)
        margin = _finite_float(fast_path_uncertainty.get("prediction_margin"), 1.0)
        uncertain = bool(fast_path_uncertainty.get("prediction_uncertain", False))
        if uncertain and entropy >= 0.80 and margin <= 0.10:
            return False
        return True

    def _constrain_blackbox_operator_hints(self, operator_hints, blackbox_state):
        """Clamp risky operator families when multivariate fast-path evidence is weak."""
        hints = dict(operator_hints or {})
        ops = set(hints.get("operators", set()))
        uncertainty = (getattr(self, "_fp_result", {}) or {}).get("uncertainty", {})
        conservative = not self._should_use_universal_fast_path(blackbox_state, uncertainty)
        if not (blackbox_state is not None and getattr(blackbox_state, "enabled", False) and conservative):
            hints["operators"] = ops
            return hints

        interaction_terms = list(getattr(blackbox_state, "interaction_terms", []) or [])
        interaction_text = " ".join(interaction_terms).lower()
        keep_periodic = "sin(" in interaction_text or "cos(" in interaction_text
        constrained = {"power", "exp", "log", "rational"}
        ops.difference_update(constrained)
        if keep_periodic:
            ops.add("periodic")
        hints["operators"] = ops
        hints["powers"] = [p for p in list(hints.get("powers", [])) if isinstance(p, (int, np.integer)) and int(p) in (2, 3)]
        hints["has_rational"] = False
        hints["has_exp_decay"] = False
        if isinstance(self.blackbox_diagnostics_, dict):
            self.blackbox_diagnostics_["operator_hint_constraint"] = {
                "conservative": True,
                "kept_operators": sorted(ops),
                "dropped_risky_families": sorted(constrained),
            }
        return hints

    def _derive_blackbox_binary_priors(self, blackbox_state, operator_hints=None):
        """Bias C++ binary search away from fragile rational structures in blackbox mode."""
        if blackbox_state is None or not getattr(blackbox_state, "enabled", False):
            return [], []

        uncertainty = (getattr(self, "_fp_result", {}) or {}).get("uncertainty", {})
        conservative = not self._should_use_universal_fast_path(blackbox_state, uncertainty)
        hints = dict(operator_hints or {})
        ops = set(hints.get("operators", set()))
        interaction_terms = list(getattr(blackbox_state, "interaction_terms", []) or [])
        interaction_text = " ".join(interaction_terms).lower()
        has_periodic = "periodic" in ops or "sin(" in interaction_text or "cos(" in interaction_text
        has_rational = bool(hints.get("has_rational", False)) and not conservative

        base = [0.62, 0.12, 0.26]
        if conservative:
            base = [0.72, 0.06, 0.22]
        elif has_rational:
            base = [0.52, 0.24, 0.24]
        elif has_periodic:
            base = [0.58, 0.10, 0.32]

        multi = []
        if getattr(self, "num_islands", 1) > 1:
            multi = [
                [0.78, 0.04, 0.18],
                [0.60, 0.06, 0.34],
                [0.66, 0.12, 0.22],
                [0.54, 0.18, 0.28] if has_rational else [0.70, 0.05, 0.25],
            ]
            while len(multi) < int(self.num_islands):
                multi.append(list(multi[len(multi) % 4]))

        diagnostics = getattr(self, "blackbox_diagnostics_", None)
        if isinstance(diagnostics, dict):
            diagnostics["binary_operator_priors"] = {
                "global": list(base),
                "multi_island": [list(v) for v in multi] if multi else [],
                "conservative": bool(conservative),
                "has_periodic_signal": bool(has_periodic),
                "has_rational_signal": bool(has_rational),
            }
        return list(base), multi

    def _derive_blackbox_unary_policy(self, blackbox_state, operator_hints=None):
        """Build hard unary operator masks for low-trust multivariate blackbox runs."""
        if blackbox_state is None or not getattr(blackbox_state, "enabled", False):
            return [], [], [], []

        uncertainty = (getattr(self, "_fp_result", {}) or {}).get("uncertainty", {})
        conservative = not self._should_use_universal_fast_path(blackbox_state, uncertainty)
        hints = dict(operator_hints or {})
        ops = set(hints.get("operators", set()))
        interaction_terms = list(getattr(blackbox_state, "interaction_terms", []) or [])
        interaction_text = " ".join(interaction_terms).lower()
        has_periodic = "periodic" in ops or "sin(" in interaction_text or "cos(" in interaction_text
        has_rational = bool(hints.get("has_rational", False)) and not conservative
        has_exp = ("exp" in ops or bool(hints.get("has_exp_decay", False))) and not conservative
        has_log = "log" in ops and not conservative
        permissive = [0, 1, 2, 3, 4]
        safe_periodic = [0, 2] if has_periodic else [2]
        exp_mild = sorted(set(([2, 3] if has_exp else [2]) + ([0] if has_periodic else [])))
        log_mild = sorted(set(([2, 4] if (has_log or has_rational) else [2]) + ([0] if has_periodic else [])))

        # Do not globally hard-mask every island. Keep one permissive search lane.
        global_allowed = []
        if conservative:
            multi = [
                [2],
                safe_periodic,
                exp_mild,
                permissive,
            ]
        else:
            multi = [
                [2],
                safe_periodic,
                exp_mild,
                permissive if (has_log or has_rational or has_exp or has_periodic) else [2, 1],
            ]

        multi = [sorted(set(int(v) for v in row)) for row in multi]
        if getattr(self, "num_islands", 1) > 1:
            while len(multi) < int(self.num_islands):
                multi.append(list(multi[len(multi) % 4]))
        else:
            multi = []

        # Likewise, avoid a global binary hard-mask; constrain by island.
        binary_allowed = []
        multi_binary = []
        if getattr(self, "num_islands", 1) > 1:
            multi_binary = [
                [0],
                [0, 2],
                [0, 2],
                [0, 1, 2],
            ]
            while len(multi_binary) < int(self.num_islands):
                multi_binary.append(list(multi_binary[len(multi_binary) % 4]))

        if isinstance(getattr(self, "blackbox_diagnostics_", None), dict):
            self.blackbox_diagnostics_["unary_operator_policy"] = {
                "allowed_unary_ops": list(global_allowed),
                "multi_allowed_unary_ops": [list(v) for v in multi] if multi else [],
                "allowed_binary_ops": list(binary_allowed),
                "multi_allowed_binary_ops": [list(v) for v in multi_binary] if multi_binary else [],
                "conservative": bool(conservative),
            }
        return global_allowed, multi, binary_allowed, multi_binary

    def _split_sample_weights(self, split, n_total=None):
        """Slice fit-time sample_weight_ to fit/val indices of a holdout split.

        Returns ``(fit_weights, val_weights)`` or ``(None, None)`` when no
        fit-time weights were provided. Length mismatches raise.
        """
        if not getattr(self, "sample_weight_provided_", False):
            return None, None
        w = getattr(self, "sample_weight_", None)
        if w is None:
            return None, None
        w = np.asarray(w, dtype=np.float64).reshape(-1)
        if n_total is not None and w.shape[0] != int(n_total):
            raise ValueError(
                f"sample_weight length {w.shape[0]} does not match n_samples {n_total}"
            )
        fit_idx = split.get("fit_idx")
        val_idx = split.get("val_idx")
        if fit_idx is None or val_idx is None:
            return None, None
        return (
            w[np.asarray(fit_idx, dtype=int)],
            w[np.asarray(val_idx, dtype=int)],
        )

    def _refine_candidate_formulas(self, candidate_formulas, X, y, *, max_candidates=12):
        """Refine symbolic candidates with affine scaling and holdout scoring."""
        if not candidate_formulas:
            return []
        split = self._domain_edge_validation_split(X, y, validation_fraction=0.2)
        if split is None:
            return []

        cpp_scores = {}
        ordered_formulas = []
        seen_cpp = set()
        for candidate in candidate_formulas:
            formula = str((candidate or {}).get("formula", "")).strip()
            if not formula:
                continue
            key = re.sub(r"\s+", "", formula.lower())
            if key in seen_cpp:
                continue
            seen_cpp.add(key)
            ordered_formulas.append(formula)
        if CPP_AVAILABLE and ordered_formulas:
            try:
                import _core  # type: ignore
                if hasattr(_core, "score_formula_candidates"):
                    fit_w, val_w = self._split_sample_weights(
                        split, n_total=int(np.asarray(y).reshape(-1).shape[0])
                    )
                    Xf = np.ascontiguousarray(split["X_fit"], dtype=np.float64)
                    yf = np.ascontiguousarray(split["y_fit"], dtype=np.float64)
                    Xv = np.ascontiguousarray(split["X_val"], dtype=np.float64)
                    yv = np.ascontiguousarray(split["y_val"], dtype=np.float64)
                    fw = None if fit_w is None else np.ascontiguousarray(fit_w, dtype=np.float64)
                    vw = None if val_w is None else np.ascontiguousarray(val_w, dtype=np.float64)
                    try:
                        scored_cpp = _core.score_formula_candidates(
                            ordered_formulas, Xf, yf, Xv, yv,
                            fit_weights=fw, val_weights=vw,
                        )
                    except TypeError:
                        # Older extension without weight args.
                        scored_cpp = _core.score_formula_candidates(
                            ordered_formulas, Xf, yf, Xv, yv,
                        )
                    for formula, scored in zip(ordered_formulas, list(scored_cpp)):
                        if isinstance(scored, dict):
                            cpp_scores[re.sub(r"\s+", "", formula.lower())] = dict(scored)
            except Exception:
                cpp_scores = {}

        ranked = []
        seen = set()
        for candidate in candidate_formulas:
            formula = str((candidate or {}).get("formula", "")).strip()
            if not formula:
                continue
            key = re.sub(r"\s+", "", formula.lower())
            if key in seen:
                continue
            seen.add(key)
            scored = None
            cpp_scored = cpp_scores.get(key)
            if cpp_scored is not None:
                if not cpp_scored.get("ok"):
                    error_text = str(cpp_scored.get("error", "")).lower()
                    cpp_invalid_markers = (
                        "nonfinite",
                        "feature_index_out_of_range",
                        "power_domain_error",
                        "domain",
                    )
                    if any(marker in error_text for marker in cpp_invalid_markers):
                        source = str((candidate or {}).get("source", "")).lower()
                        if (
                            not (candidate or {}).get("from_specialist_composition")
                            and not source.startswith("specialist_")
                        ):
                            continue
                else:
                    scale = _finite_float(cpp_scored.get("scale"), 0.0)
                    bias = _finite_float(cpp_scored.get("bias"), 0.0)
                    refined_formula = formula
                    if abs(scale - 1.0) > 1e-8 or abs(bias) > 1e-8:
                        refined_formula = f"(({scale:.12g})*({formula})+({bias:.12g}))"
                    val_mse = _finite_float(cpp_scored.get("mse"), float("inf"))
                    fit_mse = _finite_float(cpp_scored.get("fit_mse"), float("inf"))
                    val_r2 = _finite_float(cpp_scored.get("r2"), -float("inf"))
                    if np.isfinite(val_mse) and np.isfinite(fit_mse) and np.isfinite(val_r2):
                        scored = {
                            "formula": refined_formula,
                            "base_formula": formula,
                            "fit_mse": fit_mse,
                            "mse": val_mse,
                            "r2": val_r2,
                            "unweighted_fit_mse": _finite_float(
                                cpp_scored.get("unweighted_fit_mse"), fit_mse
                            ),
                            "unweighted_validation_mse": _finite_float(
                                cpp_scored.get("unweighted_validation_mse"), val_mse
                            ),
                            "unweighted_r2": _finite_float(
                                cpp_scored.get("unweighted_r2"), val_r2
                            ),
                            "weighted_fit_mse": cpp_scored.get("weighted_fit_mse"),
                            "weighted_validation_mse": cpp_scored.get("weighted_validation_mse"),
                            "weighted_r2": cpp_scored.get("weighted_r2"),
                            "weighted": bool(cpp_scored.get("weighted", False)),
                            "scale": scale,
                            "bias": bias,
                            "complexity": max(1, formula.count("+") + formula.count("-") + formula.count("*") + formula.count("/") + formula.count("^") + 1),
                            "risk_score": self._formula_risk_score(refined_formula, split["X_val"]),
                            "generalization_gap": float(max(0.0, val_mse - fit_mse) / max(float(np.var(split["y_val"])), 1e-12)),
                        }
            if scored is None:
                if "fit_w" not in locals():
                    fit_w, val_w = self._split_sample_weights(
                        split, n_total=int(np.asarray(y).reshape(-1).shape[0])
                    )
                scored = self._score_formula_candidate(
                    formula,
                    split["X_fit"],
                    split["y_fit"],
                    split["X_val"],
                    split["y_val"],
                    fit_weights=fit_w,
                    val_weights=val_w,
                )
            if scored is None:
                continue
            constant_refined = self._refine_formula_constants(
                scored["formula"],
                split["X_fit"],
                split["y_fit"],
                split["X_val"],
                split["y_val"],
            )
            if (
                constant_refined is not None
                and float(constant_refined.get("validation_mse", float("inf")))
                < float(scored.get("mse", float("inf"))) * 0.995
            ):
                constant_refined["risk_score"] = self._formula_risk_score(
                    constant_refined["formula"],
                    split["X_val"],
                )
                constant_refined["generalization_gap"] = float(
                    max(0.0, constant_refined["validation_mse"] - constant_refined["fit_mse"])
                    / max(float(np.var(split["y_val"])), 1e-12)
                )
                scored.update(constant_refined)
            merged = dict(candidate)
            merged.update({
                "formula": scored["formula"],
                "base_formula": scored["base_formula"],
                "mse": scored["mse"],
                "validation_mse": scored["mse"],
                "validation_r2": scored["r2"],
                "fit_mse": scored["fit_mse"],
                "refined_scale": scored["scale"],
                "refined_bias": scored["bias"],
                "complexity": scored["complexity"],
                "risk_score": scored.get("risk_score", 0.0),
                "generalization_gap": scored.get("generalization_gap", 0.0),
                "constant_refined": bool(scored.get("constant_refined", False)),
            })
            ranked.append(merged)

        ranked.sort(
            key=lambda c: (
                _finite_float(c.get("mse"), float("inf")) * (
                    1.0
                    + 0.25 * _finite_float(c.get("risk_score"), 0.0)
                    + 0.20 * _finite_float(c.get("generalization_gap"), 0.0)
                ),
                _finite_float(c.get("complexity"), float("inf")),
                str(c.get("formula", "")),
            )
        )
        return ranked[: max(1, int(max_candidates))]

    def _formula_family_signature(self, formula):
        text = str(formula or "").strip().lower()
        if not text:
            return "empty"
        if "sin(" in text:
            return "sin"
        if "cos(" in text:
            return "cos"
        if "exp(" in text:
            return "exp"
        if "log(" in text:
            return "log"
        if "/" in text:
            return "rational"
        if "*" in text:
            return "multiplicative"
        if "+" in text or "-" in text:
            return "additive"
        if "^" in text:
            return "power"
        return "univariate"

    def _formula_feature_signature(self, formula):
        text = str(formula or "")
        return tuple(sorted({int(match.group(1)) for match in re.finditer(r"\bx(\d+)\b", text)}))

    def _prune_blackbox_candidate_formulas(self, candidate_formulas, *, max_candidates=12):
        """Keep diverse, high-quality blackbox candidates instead of many near-duplicates."""
        if not candidate_formulas:
            return []

        # Phase 5: dimensional filter before diversity prune when units active.
        candidate_formulas = self._filter_candidates_by_units(candidate_formulas)

        ordered = sorted(
            candidate_formulas,
            key=lambda c: (
                float(c.get("unit_penalty") or 0.0) if c.get("unit_ok") else 0.0,
                _finite_float(c.get("mse"), float("inf")),
                -_finite_float(c.get("validation_r2"), -float("inf")),
                _finite_float(c.get("complexity"), float("inf")),
            ),
        )
        kept = []
        seen_formulas = set()
        seen_family_feature = set()
        for cand in ordered:
            formula = str(cand.get("formula", "")).strip()
            if not formula:
                continue
            normalized = re.sub(r"\s+", "", formula.lower())
            if normalized in seen_formulas:
                continue
            family = self._formula_family_signature(formula)
            features = self._formula_feature_signature(formula)
            key = (family, features)

            if key in seen_family_feature:
                if len(kept) >= max(2, int(max_candidates) // 2):
                    continue
            seen_formulas.add(normalized)
            seen_family_feature.add(key)
            kept.append(cand)
            if len(kept) >= max(1, int(max_candidates)):
                break
        return kept

    def _derive_blackbox_operator_hints(self, blackbox_state, candidate_formulas):
        """Convert validated blackbox interactions/candidates into operator-family hints."""
        hints = {
            "operators": set(),
            "powers": [],
            "active_terms": [],
            "has_rational": False,
            "has_exp_decay": False,
        }
        if blackbox_state is None:
            return hints

        interaction_scores = getattr(blackbox_state, "interaction_scores", {}) or {}
        for term in list(getattr(blackbox_state, "interaction_terms", []) or [])[:8]:
            score = float(interaction_scores.get(term, 0.0))
            if score < 0.12:
                continue
            hints["active_terms"].append(term)
            lower = term.lower()
            if "sin(" in lower or "cos(" in lower:
                hints["operators"].add("periodic")
            if "exp(" in lower:
                hints["operators"].add("exp")
                hints["has_exp_decay"] = True
            if "log(" in lower:
                hints["operators"].add("log")
            if "/" in lower:
                hints["operators"].add("rational")
                hints["has_rational"] = True
            if "^2" in lower or "^3" in lower:
                hints["operators"].add("power")
                for power_text in re.findall(r"\^(\d+)", lower):
                    try:
                        hints["powers"].append(int(power_text))
                    except Exception:
                        pass

        for cand in candidate_formulas or []:
            if _finite_float(cand.get("validation_r2"), -1.0) < 0.25:
                continue
            formula = str(cand.get("formula", "")).strip()
            if not formula:
                continue
            hints["active_terms"].append(formula)
            family = self._formula_family_signature(formula)
            if family in ("sin", "cos"):
                hints["operators"].add("periodic")
            elif family == "exp":
                hints["operators"].add("exp")
                hints["has_exp_decay"] = True
            elif family == "log":
                hints["operators"].add("log")
            elif family == "rational":
                hints["operators"].add("rational")
                hints["has_rational"] = True
            elif family == "power":
                hints["operators"].add("power")

        hints["powers"] = sorted(set(int(p) for p in hints["powers"] if isinstance(p, (int, np.integer)) and 1 <= int(p) <= 8))
        hints["active_terms"] = list(dict.fromkeys(hints["active_terms"]))[:12]
        return hints

    def _targeted_specialist_probe_formulas(self, X, y, *, max_formulas=96):
        """Generate deterministic probes for hard univariate composition families.

        These are intentionally raw structural candidates. The existing
        refinement/holdout scorer decides whether any of them are useful.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        if X_arr.ndim != 2 or X_arr.shape[0] != y_arr.shape[0] or X_arr.shape[1] < 1:
            return []

        formulas = []
        seen = set()

        def add(text, source="targeted_specialist_probe", **extra):
            formula = str(text or "").strip()
            if not formula or formula == "0":
                return
            key = re.sub(r"\s+", "", formula.lower())
            if key in seen:
                return
            seen.add(key)
            item = {
                "formula": formula,
                "mse": float("inf"),
                "source": source,
                "from_targeted_specialist_probe": True,
            }
            item.update(extra)
            formulas.append(item)

        def _freq_text(value):
            try:
                v = float(value)
            except Exception:
                return None
            if not np.isfinite(v) or abs(v) < 1e-8 or abs(v) > 20.0:
                return None
            for canonical, text in ((1.0, ""), (2.0, "2*"), (3.0, "3*"), (5.0, "5*"), (np.pi, "pi*")):
                if abs(v - canonical) < 0.08:
                    return text
            return f"{v:.6g}*"

        for feature_idx in range(min(int(X_arr.shape[1]), 2)):
            x = f"x{feature_idx}"
            x_values = X_arr[:, feature_idx]
            if feature_idx > 0:
                finite_steps = np.diff(x_values[np.isfinite(x_values)])
                if finite_steps.size and not (
                    np.mean(finite_steps >= -1e-12) >= 0.95
                    or np.mean(finite_steps <= 1e-12) >= 0.95
                ):
                    continue
            y_var = float(np.var(y_arr)) if y_arr.size else 0.0
            finite_x = x_values[np.isfinite(x_values)]
            crosses_zero = bool(finite_x.size and np.nanmin(finite_x) < 0.0 < np.nanmax(finite_x))

            # Symmetry/cusp probes: target abs/even failures and cusp envelopes.
            if crosses_zero:
                add(f"abs({x})", "symmetry_cusp_probe")
                add(f"{x}/(1+abs({x}))", "symmetry_cusp_probe")
                add(f"sqrt(abs({x}))", "symmetry_cusp_probe")
                add(f"exp(-abs({x}))", "symmetry_cusp_probe")

            # Envelope/factor probes for exp damping and Gaussian damping.
            envelopes = [
                f"exp(-{x})",
                f"exp(-2*{x})",
                f"exp(-{x}^2)",
                f"exp(-{x}^2/2)",
                f"1/(1+exp(-{x}))",
            ]
            if crosses_zero:
                envelopes.append(f"exp(-abs({x}))")
            powers = ["1", x, f"{x}^2", f"{x}^3"]
            for env in envelopes:
                add(env, "envelope_probe")
                for power in powers[1:]:
                    add(f"{power}*{env}", "envelope_factor_probe")

            # Frequency/carrier probes. Include FFT candidates and common hard-suite harmonics.
            freq_values = [1.0, 2.0, 3.0, 5.0, np.pi]
            try:
                detected = self._detect_frequencies(X_arr[:, [feature_idx]], y_arr)
            except Exception:
                detected = []
            for omega in list(detected or [])[:4]:
                try:
                    freq_values.append(float(omega))
                except Exception:
                    pass
            freq_prefixes = []
            for omega in freq_values:
                text = _freq_text(omega)
                if text is not None and text not in freq_prefixes:
                    freq_prefixes.append(text)

            carriers = []
            for prefix in freq_prefixes[:8]:
                sx = f"sin({prefix}{x})"
                cx = f"cos({prefix}{x})"
                carriers.extend([sx, cx])
                add(sx, "carrier_probe")
                add(cx, "carrier_probe")

            for env in envelopes:
                for carrier in carriers[:12]:
                    add(f"{env}*{carrier}", "envelope_carrier_probe")
                    add(f"{x}*{env}*{carrier}", "envelope_carrier_probe")
                    add(f"{x}^2*{env}*{carrier}", "envelope_carrier_probe")

            # Product-of-carriers probes for trig product failures.
            for a in ("", "2*", "3*", "5*"):
                for b in ("2*", "3*", "5*"):
                    if a == b:
                        continue
                    add(f"sin({a}{x})*sin({b}{x})", "carrier_product_probe")
                    add(f"sin({a}{x})*cos({b}{x})", "carrier_product_probe")
            add(f"sin({x})*sin(3*{x})*sin(5*{x})", "carrier_product_probe")
            add(f"{x}^2*exp(-{x})*cos(3*{x})", "envelope_carrier_probe")
            add(f"{x}^2*exp(-{x})*sin({x})", "envelope_carrier_probe")

            # Log/nested composition probes for formulas that fit poorly with
            # additive Fourier/polynomial surrogates.
            log_nested = [
                f"log(1+{x}^2)",
                f"log(1+sin({x})^2)",
                f"log(1+exp({x}))",
                f"sin({x}*cos({x}))",
                f"sin(exp(-{x}))",
                f"cos(sin({x}))",
                f"sin({x}+sin({x}))",
                f"sin({x}+exp(-{x}))",
                f"sqrt(abs(sin({x})))",
                f"sin({x})/sqrt(1+{x}^2)",
            ]
            for formula in log_nested:
                add(formula, "nested_transform_probe")

            if y_var > 1e-12 and crosses_zero:
                # Cheap parity hints: keep both exact parity probes and their
                # damped/carrier variants available to the ranker.
                add(f"({x}^2-1)*exp(-{x}^2/2)", "symmetry_envelope_probe")
                add(f"sin(pi*{x})*exp(-{x}^2)", "envelope_carrier_probe")

        probe_scoring_backend = "none"
        if formulas:
            source_priority = {
                "envelope_carrier_probe": 0,
                "carrier_product_probe": 1,
                "envelope_factor_probe": 2,
                "envelope_probe": 3,
                "carrier_probe": 4,
            }
            if CPP_AVAILABLE:
                try:
                    probe_w = None
                    if (
                        getattr(self, "sample_weight_provided_", False)
                        and getattr(self, "sample_weight_", None) is not None
                    ):
                        probe_w = np.asarray(self.sample_weight_, dtype=np.float64).reshape(-1)
                        if probe_w.shape[0] != y_arr.shape[0]:
                            raise ValueError(
                                f"sample_weight length {probe_w.shape[0]} does not match y length {y_arr.shape[0]}"
                            )
                    Xa = np.ascontiguousarray(X_arr, dtype=np.float64)
                    ya = np.ascontiguousarray(y_arr, dtype=np.float64)
                    formulas_list = [str(item.get("formula", "")) for item in formulas]
                    try:
                        if probe_w is not None:
                            w_c = np.ascontiguousarray(probe_w, dtype=np.float64)
                            scored_cpp = _core.score_formula_candidates(
                                formulas_list, Xa, ya, Xa, ya,
                                fit_weights=w_c, val_weights=w_c,
                            )
                        else:
                            scored_cpp = _core.score_formula_candidates(
                                formulas_list, Xa, ya, Xa, ya,
                            )
                    except TypeError:
                        scored_cpp = _core.score_formula_candidates(
                            formulas_list, Xa, ya, Xa, ya,
                        )
                    ok_count = 0
                    for item, scored in zip(formulas, list(scored_cpp)):
                        if isinstance(scored, dict) and scored.get("ok"):
                            item["probe_mse"] = _finite_float(scored.get("mse"), float("inf"))
                            ok_count += 1
                        else:
                            item["probe_mse"] = float("inf")
                    if ok_count > 0:
                        probe_scoring_backend = "cpp_batch"
                except Exception:
                    probe_scoring_backend = "python_fallback"

            if probe_scoring_backend != "cpp_batch":
                for item in formulas:
                    formula = str(item.get("formula", ""))
                    score_mse = float("inf")
                    try:
                        pred = self._safe_eval_formula_array(formula, X_arr).reshape(-1)
                        if pred.shape == y_arr.shape and np.all(np.isfinite(pred)):
                            p_var = float(np.var(pred))
                            if p_var > 1e-15:
                                scale = float(np.cov(pred, y_arr, bias=True)[0, 1] / p_var)
                                bias = float(np.mean(y_arr) - scale * np.mean(pred))
                                pred = scale * pred + bias
                            score_mse = float(np.mean((pred - y_arr) ** 2))
                    except Exception:
                        score_mse = float("inf")
                    item["probe_mse"] = score_mse
                if probe_scoring_backend == "none":
                    probe_scoring_backend = "python"
            formulas.sort(
                key=lambda item: (
                    float(item.get("probe_mse", float("inf"))),
                    source_priority.get(str(item.get("source", "")), 9),
                    len(str(item.get("formula", ""))),
                    str(item.get("formula", "")),
                )
            )

        if isinstance(getattr(self, "blackbox_diagnostics_", None), dict):
            self.blackbox_diagnostics_["targeted_specialist_probes"] = {
                "count": len(formulas),
                "sources": sorted({str(item.get("source", "")) for item in formulas}),
                "scoring_backend": probe_scoring_backend,
            }
        return formulas[: max(0, int(max_formulas))]

    def _build_blackbox_candidate_formulas(
        self,
        best_formula,
        best_mse,
        proposer_payload,
        blackbox_state,
        X,
        y,
        *,
        max_candidates,
    ):
        """Build, refine, and prune a shared candidate pool for basis fitting and evolution."""
        import time as _time
        _phase_start = _time.time()
        raw_candidates = []
        try:
            if best_formula:
                raw_candidates.append({
                    "formula": best_formula,
                    "mse": best_mse if best_mse is not None else float("inf"),
                    "from_fast_path": True,
                })

            if isinstance(proposer_payload, dict):
                for cand in proposer_payload.get("candidate_skeletons", [])[:10]:
                    formula = str(cand.get("formula", "")).strip()
                    if not formula:
                        continue
                    raw_candidates.append({
                        "formula": formula,
                        "mse": cand.get("mse", float("inf")),
                        "score": cand.get("score", 0.0),
                        "active_terms": cand.get("active_terms", []),
                        "from_proposer": True,
                    })

            if blackbox_state is not None and getattr(blackbox_state, "enabled", False):
                selected_features = list(getattr(blackbox_state, "selected_features", []) or [])
                for term in list(getattr(blackbox_state, "interaction_terms", []) or [])[:8]:
                    seed_formula = remap_original_formula_to_reduced(term, selected_features)
                    raw_candidates.append({
                        "formula": seed_formula,
                        "mse": float("inf"),
                        "score": float((getattr(blackbox_state, "interaction_scores", {}) or {}).get(term, 0.0)),
                        "active_terms": [seed_formula],
                        "from_blackbox_interaction": True,
                    })
                for formula in list(getattr(blackbox_state, "candidate_seed_formulas", []) or [])[:16]:
                    seed_formula = remap_original_formula_to_reduced(formula, selected_features)
                    raw_candidates.append({
                        "formula": seed_formula,
                        "mse": float("inf"),
                        "score": 0.2,
                        "active_terms": [seed_formula],
                        "from_blackbox_seed": True,
                    })
                # Standardized-space structure seeds with free-const refine (compete, no auto-win).
                try:
                    structure_seeds = self._fit_search_space_structure_seeds(
                        X,
                        y,
                        max_seeds=max(6, min(12, int(max_candidates) if max_candidates else 8)),
                        blackbox_state=blackbox_state,
                    )
                except Exception:
                    structure_seeds = []
                for seed in structure_seeds:
                    formula = str((seed or {}).get("formula") or "").strip()
                    if not formula:
                        continue
                    raw_candidates.append({
                        "formula": formula,
                        "mse": float((seed or {}).get("mse", float("inf"))),
                        "score": 1.5,
                        "validation_mse": (seed or {}).get("validation_mse"),
                        "validation_r2": (seed or {}).get("validation_r2"),
                        "complexity": (seed or {}).get("complexity"),
                        "active_terms": [formula],
                        "from_structure_seed": True,
                        "source": "search_space_structure_seed",
                        "skeleton": (seed or {}).get("skeleton"),
                    })
                # Original-space probe remains diagnostic only (not injected).

            if np.asarray(X).ndim == 2 and int(np.asarray(X).shape[1]) == 1:
                raw_candidates.extend(self._targeted_specialist_probe_formulas(X, y, max_formulas=64))

            refined = self._refine_candidate_formulas(
                raw_candidates,
                X,
                y,
                max_candidates=max(
                    int(max_candidates),
                    8,
                ),
            )
            return self._prune_blackbox_candidate_formulas(
                refined,
                max_candidates=max_candidates,
            )
        finally:
            self._add_phase_time("candidate_building", _time.time() - _phase_start)

    def _build_univariate_specialist_candidate_formulas(
        self,
        best_formula,
        best_mse,
        proposer_payload,
        X,
        y,
        *,
        max_candidates,
    ):
        """Build a specialist candidate pool for ordinary 1D runs.

        The original specialist integration only consumed blackbox candidate
        pools, so univariate benchmark/SRBench runs often had no specialist
        diagnostics or composed seeds. This pool keeps the same refinement and
        pruning gates while sourcing candidates from fast-path alternates,
        proposer skeletons, and a small domain-native basis.
        """
        import time as _time
        _phase_start = _time.time()
        raw_candidates = []
        seen = set()

        def add(formula, source, mse=float("inf"), **extra):
            text = str(formula or "").strip()
            if not text or text == "0":
                return
            key = re.sub(r"\s+", "", text.lower())
            if key in seen:
                return
            seen.add(key)
            item = {"formula": text, "mse": mse, "source": source}
            item.update(extra)
            raw_candidates.append(item)

        add(best_formula, "fast_path", best_mse if best_mse is not None else float("inf"), from_fast_path=True)

        fp_result = getattr(self, "_fp_result", None)
        if isinstance(fp_result, dict):
            for cand in list(fp_result.get("candidate_formulas", []) or [])[:10]:
                add(
                    (cand or {}).get("formula"),
                    (cand or {}).get("source") or "fast_path_candidate",
                    (cand or {}).get("mse", float("inf")),
                    from_fast_path_candidate=True,
                )
            details = fp_result.get("details") or {}
            if isinstance(details, dict):
                for cand in list(details.get("candidate_formulas", []) or [])[:10]:
                    add(
                        (cand or {}).get("formula"),
                        (cand or {}).get("source") or "fast_path_candidate",
                        (cand or {}).get("mse", float("inf")),
                        from_fast_path_candidate=True,
                    )

        if isinstance(proposer_payload, dict):
            for cand in list(proposer_payload.get("candidate_skeletons", []) or [])[:10]:
                add(
                    (cand or {}).get("formula"),
                    "proposer",
                    (cand or {}).get("mse", float("inf")),
                    from_proposer=True,
                    score=(cand or {}).get("score", 0.0),
                    active_terms=(cand or {}).get("active_terms", []),
                )

        n_features = int(np.asarray(X).shape[1]) if np.asarray(X).ndim == 2 else 1
        for formula in self._build_blackbox_formula_pool(best_formula, proposer_payload, None, n_features)[:24]:
            add(formula, "specialist_basis_seed", float("inf"), from_specialist_basis_seed=True)

        for cand in self._targeted_specialist_probe_formulas(X, y, max_formulas=64):
            add(
                (cand or {}).get("formula"),
                (cand or {}).get("source") or "targeted_specialist_probe",
                (cand or {}).get("mse", float("inf")),
                from_targeted_specialist_probe=True,
            )

        try:
            refined = self._refine_candidate_formulas(
                raw_candidates,
                X,
                y,
                max_candidates=max(int(max_candidates), 8),
            )
            return self._prune_blackbox_candidate_formulas(refined, max_candidates=max_candidates)
        finally:
            self._add_phase_time("univariate_candidate_building", _time.time() - _phase_start)

    def _run_specialist_candidate_screening(
        self,
        candidate_formulas,
        X,
        y,
        blackbox_search_plan,
        *,
        diagnostics_key="candidate_screening",
    ):
        """Run specialist diagnostics/composition and merge accepted candidates."""
        if not candidate_formulas or not getattr(self, "enable_specialist_screening_diagnostics", True):
            return candidate_formulas or []

        if not isinstance(self.blackbox_diagnostics_, dict):
            self.blackbox_diagnostics_ = {}
        screening_diag = self.blackbox_diagnostics_.setdefault(diagnostics_key, {})
        screening_diag.update({
            "candidate_count": len(candidate_formulas or []),
            "top_candidates": [
                {
                    "formula": str(c.get("formula", ""))[:160],
                    "validation_r2": c.get("validation_r2"),
                    "validation_mse": c.get("validation_mse"),
                    "complexity": c.get("complexity"),
                }
                for c in (candidate_formulas or [])[:6]
            ],
        })

        specialist_screening = self._compute_specialist_screening_diagnostics(
            candidate_formulas,
            X,
            y,
            max_candidates=6,
            max_pairs=5,
        )
        if specialist_screening is None:
            return candidate_formulas or []

        screening_diag["specialist_screening"] = specialist_screening
        best_existing_mse = float("inf")
        best_existing_r2 = -float("inf")
        for cand in candidate_formulas or []:
            cand_mse = _finite_float((cand or {}).get("validation_mse"), float("inf"))
            cand_r2 = _finite_float((cand or {}).get("validation_r2"), -float("inf"))
            best_existing_mse = min(best_existing_mse, cand_mse)
            best_existing_r2 = max(best_existing_r2, cand_r2)

        existing_exact_candidate = (
            best_existing_mse <= max(float(getattr(self, "early_stop_mse", 1e-10)), 1e-10)
            or best_existing_r2 >= 0.999999999
        )
        if existing_exact_candidate:
            screening_diag["residual_skipped_reason"] = "existing_exact_candidate"
            screening_diag["best_existing_validation_mse"] = float(best_existing_mse)
            screening_diag["best_existing_validation_r2"] = float(best_existing_r2)
            return candidate_formulas or []

        specialist_candidates = []
        if getattr(self, "enable_specialist_composition_screening", True):
            specialist_candidates = self._compose_specialist_candidates(
                candidate_formulas,
                X,
                y,
                max_candidates=max(
                    8,
                    int((blackbox_search_plan or {}).get("screening_budget", 8)),
                ),
            )

        residual_merged_candidates = []
        if self.enable_residual_stage and specialist_candidates:
            for cand in list(specialist_candidates)[:2]:
                formula = cand.get("formula")
                if not formula:
                    continue
                val_r2 = _finite_float(cand.get("validation_r2"), -1.0)
                if val_r2 >= 0.75:
                    res_form = self._stage_residual_symbolic_fit(X, y, formula, _allow_recursion=True)
                    if res_form and res_form != "0":
                        combined_formula = f"({formula})+({res_form})"
                        refined_list = self._refine_candidate_formulas(
                            [{
                                "formula": combined_formula,
                                "source": "specialist_residual_composition",
                                "from_specialist_composition": True,
                            }],
                            X,
                            y,
                            max_candidates=1,
                        )
                        if refined_list:
                            residual_merged_candidates.extend(refined_list)

        if specialist_candidates or residual_merged_candidates:
            self.has_composed_seeds_ = True
            self.composition_seeded_evolution_ = True
            candidate_formulas = self._prune_blackbox_candidate_formulas(
                list(residual_merged_candidates) + list(specialist_candidates) + list(candidate_formulas or []),
                max_candidates=max(
                    8,
                    int((blackbox_search_plan or {}).get("seed_budget", 8)),
                ),
            )
            screening_diag["candidate_count"] = len(candidate_formulas or [])
            screening_diag["top_candidates"] = [
                {
                    "formula": str(c.get("formula", ""))[:160],
                    "validation_r2": c.get("validation_r2"),
                    "validation_mse": c.get("validation_mse"),
                    "complexity": c.get("complexity"),
                }
                for c in (candidate_formulas or [])[:6]
            ]
        return candidate_formulas or []

    def _build_blackbox_formula_pool(self, best_formula, proposer_payload, blackbox_state, n_features):
        """Assemble a compact pool of reduced-space formulas for cheap additive fitting."""
        formulas = []
        seen = set()

        def _add(text, family=None):
            formula = str(text or "").strip()
            if not formula or formula == "0":
                return
            key = re.sub(r"\s+", "", formula.lower())
            if key in seen:
                return
            if family is not None:
                existing = [
                    f for f in formulas
                    if self._formula_family_signature(f) == family
                ]
                if len(existing) >= 4:
                    return
            seen.add(key)
            formulas.append(formula)

        if best_formula:
            _add(best_formula)

        if isinstance(proposer_payload, dict):
            for cand in proposer_payload.get("candidate_skeletons", [])[:8]:
                _add(cand.get("formula", ""), family=self._formula_family_signature(cand.get("formula", "")))

        for local_idx in range(int(max(1, n_features))):
            _add(f"x{local_idx}")
            _add(f"x{local_idx}^2")
            _add(f"x{local_idx}^3")
            _add(f"sin(x{local_idx})")
            _add(f"cos(x{local_idx})")
            _add(f"exp(-abs(x{local_idx}))")

        if blackbox_state is not None and getattr(blackbox_state, "enabled", False):
            selected = list(getattr(blackbox_state, "selected_features", []) or [])
            for term in list(getattr(blackbox_state, "interaction_terms", []) or [])[:8]:
                reduced = remap_original_formula_to_reduced(term, selected)
                _add(reduced, family=self._formula_family_signature(reduced))
            for formula in list(getattr(blackbox_state, "candidate_seed_formulas", []) or [])[:16]:
                reduced = remap_original_formula_to_reduced(formula, selected)
                _add(reduced, family=self._formula_family_signature(reduced))

        for i in range(int(max(1, n_features))):
            for j in range(i + 1, int(max(1, n_features))):
                _add(f"x{i}*x{j}")
                _add(f"x{i}+x{j}")
                _add(f"x{i}-x{j}")

        return formulas[:32]

    def _fit_blackbox_basis_model(self, X, y, candidate_formulas, *, max_terms=4):
        """Fit a small additive symbolic model from a screened basis pool."""
        if not candidate_formulas:
            return None
        split = self._split_blackbox_holdout(X, y, validation_fraction=0.2)
        if split is None:
            return None

        X_fit = split["X_fit"]
        y_fit = split["y_fit"]
        X_val = split["X_val"]
        y_val = split["y_val"]
        y_fit = np.asarray(y_fit, dtype=np.float64).reshape(-1)
        y_val = np.asarray(y_val, dtype=np.float64).reshape(-1)
        base_val_mse = float(np.mean((y_val - float(np.mean(y_fit))) ** 2))

        basis = []
        seen_signatures = []
        for formula in candidate_formulas:
            try:
                fit_values = self._safe_eval_formula_array(formula, X_fit).reshape(-1)
                val_values = self._safe_eval_formula_array(formula, X_val).reshape(-1)
                full_values = self._safe_eval_formula_array(formula, X).reshape(-1)
            except Exception:
                continue
            if (
                fit_values.shape[0] != X_fit.shape[0]
                or val_values.shape[0] != X_val.shape[0]
                or full_values.shape[0] != X.shape[0]
            ):
                continue
            if not (np.all(np.isfinite(fit_values)) and np.all(np.isfinite(val_values)) and np.all(np.isfinite(full_values))):
                continue
            if float(np.std(fit_values)) < 1e-10:
                continue

            duplicate = False
            for prev in seen_signatures:
                if np.corrcoef(prev, fit_values)[0, 1] > 0.995:
                    duplicate = True
                    break
            if duplicate:
                continue
            seen_signatures.append(fit_values)
            basis.append({
                "formula": formula,
                "fit": fit_values,
                "val": val_values,
                "full": full_values,
                "complexity": self._formula_complexity(formula),
            })

        if not basis:
            return None

        selected = []
        selected_cols_fit = []
        selected_cols_val = []
        best_val_mse = base_val_mse

        for _ in range(int(max(1, max_terms))):
            best_choice = None
            for cand in basis:
                if cand in selected:
                    continue
                cols_fit = selected_cols_fit + [cand["fit"]]
                cols_val = selected_cols_val + [cand["val"]]
                design_fit = np.column_stack(cols_fit + [np.ones_like(y_fit)])
                design_val = np.column_stack(cols_val + [np.ones_like(y_val)])
                try:
                    coef, _, _, _ = np.linalg.lstsq(design_fit, y_fit, rcond=None)
                    val_pred = design_val @ coef
                    val_mse = float(np.mean((val_pred - y_val) ** 2))
                except Exception:
                    continue
                complexity = sum(item["complexity"] for item in selected) + cand["complexity"]
                penalized = val_mse * (1.0 + 0.003 * complexity)
                if best_choice is None or penalized < best_choice["penalized"]:
                    best_choice = {
                        "cand": cand,
                        "coef": coef,
                        "val_mse": val_mse,
                        "penalized": penalized,
                    }

            if best_choice is None:
                break
            improvement = best_val_mse - float(best_choice["val_mse"])
            if improvement <= max(1e-8, 0.01 * max(best_val_mse, 1e-8)):
                break

            selected.append(best_choice["cand"])
            selected_cols_fit.append(best_choice["cand"]["fit"])
            selected_cols_val.append(best_choice["cand"]["val"])
            best_val_mse = float(best_choice["val_mse"])

        if not selected:
            return None

        design_full = np.column_stack([item["full"] for item in selected] + [np.ones(X.shape[0])])
        y_full = np.asarray(y, dtype=np.float64).reshape(-1)
        try:
            coef_full, _, _, _ = np.linalg.lstsq(design_full, y_full, rcond=None)
        except Exception:
            return None

        terms = []
        for weight, item in zip(coef_full[:-1], selected):
            if not np.isfinite(weight) or abs(float(weight)) < 1e-8:
                continue
            terms.append(f"({float(weight):.12g})*({item['formula']})")
        bias = float(coef_full[-1])
        if abs(bias) > 1e-8 or not terms:
            terms.append(f"({bias:.12g})")
        formula = "+".join(terms) if terms else "0"

        full_pred = self._safe_eval_formula_array(formula, X)
        full_mse = float(np.mean((full_pred - y_full) ** 2))
        y_val_var = float(np.var(y_val))
        val_r2 = 1.0 if y_val_var < 1e-15 and best_val_mse < 1e-15 else (
            0.0 if y_val_var < 1e-15 else 1.0 - best_val_mse / y_val_var
        )

        return {
            "formula": formula,
            "mse": full_mse,
            "validation_mse": best_val_mse,
            "validation_r2": float(val_r2),
            "selected_terms": [item["formula"] for item in selected],
            "n_terms": len(selected),
            "complexity": self._formula_complexity(formula),
        }

    def _fit_blackbox_engineered_basis_model(self, X, y, *, max_terms=10):
        """Fit a compact validation-selected engineered basis for Track 1.

        This is deliberately not a broad kitchen-sink expansion. It gives
        multivariate blackbox datasets a strong symbolic baseline made from
        linear, low-degree polynomial, pairwise interaction, and a few stable
        unary transforms, then exports only the selected terms.
        """
        split = self._random_blackbox_validation_split(X, y, validation_fraction=0.25, salt=31)
        if split is None:
            return None

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        X_fit = split["X_fit"]
        y_fit = np.asarray(split["y_fit"], dtype=np.float64).reshape(-1)
        X_val = split["X_val"]
        y_val = np.asarray(split["y_val"], dtype=np.float64).reshape(-1)
        n_features = int(X.shape[1])

        def add_feature(pool, name, full_values):
            values = np.asarray(full_values, dtype=np.float64).reshape(-1)
            if values.shape[0] != X.shape[0] or not np.all(np.isfinite(values)):
                return
            if float(np.std(values)) < 1e-10:
                return
            pool.append({"name": name, "full": values})

        pool = []
        for j in range(n_features):
            xj = X[:, j]
            add_feature(pool, f"x{j}", xj)
            add_feature(pool, f"x{j}^2", xj ** 2)
            add_feature(pool, f"x{j}^3", xj ** 3)
            add_feature(pool, f"sin(x{j})", np.sin(xj))
            add_feature(pool, f"cos(x{j})", np.cos(xj))
            add_feature(pool, f"exp(-abs(x{j}))", np.exp(-np.abs(np.clip(xj, -20.0, 20.0))))

        for i in range(n_features):
            xi = X[:, i]
            for j in range(i + 1, n_features):
                xj = X[:, j]
                add_feature(pool, f"x{i}*x{j}", xi * xj)
                add_feature(pool, f"(x{i}-x{j})^2", (xi - xj) ** 2)
                add_feature(pool, f"x{i}+x{j}", xi + xj)
                add_feature(pool, f"x{i}-x{j}", xi - xj)
                add_feature(pool, f"x{i}*sin(x{j})", xi * np.sin(xj))
                add_feature(pool, f"x{j}*sin(x{i})", xj * np.sin(xi))
                add_feature(pool, f"x{i}*cos(x{j})", xi * np.cos(xj))
                add_feature(pool, f"x{j}*cos(x{i})", xj * np.cos(xi))
                add_feature(pool, f"x{i}*exp(-abs(x{j}))", xi * np.exp(-np.abs(np.clip(xj, -20.0, 20.0))))
                add_feature(pool, f"x{j}*exp(-abs(x{i}))", xj * np.exp(-np.abs(np.clip(xi, -20.0, 20.0))))
                add_feature(pool, f"x{i}/(abs(x{j})+1e-6)", xi / (np.abs(xj) + 1e-6))
                add_feature(pool, f"x{j}/(abs(x{i})+1e-6)", xj / (np.abs(xi) + 1e-6))

        if not pool:
            return None

        fit_idx = split["fit_idx"]
        val_idx = split["val_idx"]
        A_full = np.column_stack([item["full"] for item in pool])
        A_fit = A_full[fit_idx]
        A_val = A_full[val_idx]

        max_pool_terms = 180 if n_features >= 8 else 260
        if A_fit.shape[1] > max_pool_terms:
            y_probe = y_fit - float(np.mean(y_fit))
            y_scale = float(np.std(y_probe))
            scores = []
            for idx, item in enumerate(pool):
                values = A_fit[:, idx]
                v_scale = float(np.std(values))
                if v_scale < 1e-10 or y_scale < 1e-10:
                    score = 0.0
                else:
                    try:
                        score = abs(float(np.corrcoef(values, y_probe)[0, 1]))
                    except Exception:
                        score = 0.0
                    if not np.isfinite(score):
                        score = 0.0
                linear_bonus = 1.0 if re.fullmatch(r"x\d+", item["name"]) else 0.0
                scores.append((linear_bonus, score, idx))
            scores.sort(key=lambda row: (row[0], row[1]), reverse=True)
            keep = sorted(idx for _, _, idx in scores[:max_pool_terms])
            pool = [pool[idx] for idx in keep]
            A_full = A_full[:, keep]
            A_fit = A_fit[:, keep]
            A_val = A_val[:, keep]

        mu = np.mean(A_fit, axis=0)
        sigma = np.std(A_fit, axis=0)
        sigma = np.where(sigma < 1e-10, 1.0, sigma)
        Z_fit = (A_fit - mu) / sigma
        Z_val = (A_val - mu) / sigma

        y_mean = float(np.mean(y_fit))
        y_center = y_fit - y_mean

        try:
            from sklearn.linear_model import ElasticNetCV, RidgeCV
            from sklearn.exceptions import ConvergenceWarning
        except Exception:
            ElasticNetCV = RidgeCV = None
            ConvergenceWarning = Warning

        candidate_weights = []
        if ElasticNetCV is not None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    enet = ElasticNetCV(
                        l1_ratio=[0.15, 0.35, 0.65, 0.9],
                        alphas=np.logspace(-4, 0, 24),
                        cv=min(5, max(2, len(y_fit) // 40)),
                        max_iter=12000,
                        tol=1e-3,
                        random_state=0,
                    )
                    enet.fit(Z_fit, y_center)
                candidate_weights.append(np.asarray(enet.coef_, dtype=np.float64))
            except Exception:
                pass
        if RidgeCV is not None:
            try:
                ridge = RidgeCV(alphas=np.logspace(-4, 3, 24))
                ridge.fit(Z_fit, y_center)
                candidate_weights.append(np.asarray(ridge.coef_, dtype=np.float64))
            except Exception:
                pass

        if not candidate_weights:
            try:
                coef, _, _, _ = np.linalg.lstsq(
                    np.column_stack([Z_fit, np.ones(Z_fit.shape[0])]),
                    y_fit,
                    rcond=None,
                )
                candidate_weights.append(np.asarray(coef[:-1], dtype=np.float64))
            except Exception:
                return None

        best = None
        y_val_var = max(float(np.var(y_val)), 1e-12)
        term_grid = sorted(set([3, 5, 8, int(max_terms)]))
        for weights in candidate_weights:
            if weights.shape[0] != A_full.shape[1] or not np.all(np.isfinite(weights)):
                continue
            order = np.argsort(np.abs(weights))[::-1]
            for k in term_grid:
                active = [idx for idx in order[: max(1, min(k, len(order)))] if abs(weights[idx]) > 1e-10]
                if not active:
                    continue
                design_fit = np.column_stack([A_fit[:, active], np.ones(A_fit.shape[0])])
                design_val = np.column_stack([A_val[:, active], np.ones(A_val.shape[0])])
                try:
                    coef, _, _, _ = np.linalg.lstsq(design_fit, y_fit, rcond=None)
                    pred_val = design_val @ coef
                    val_mse = float(np.mean((pred_val - y_val) ** 2))
                except Exception:
                    continue
                if not np.isfinite(val_mse):
                    continue
                complexity = sum(self._formula_complexity(pool[idx]["name"]) for idx in active)
                score = val_mse * (1.0 + 0.01 * len(active) + 0.002 * complexity)
                if best is None or score < best["score"]:
                    best = {
                        "active": active,
                        "coef": coef,
                        "validation_mse": val_mse,
                        "score": score,
                    }

        if best is None:
            return None

        active = best["active"]
        design_full = np.column_stack([A_full[:, active], np.ones(A_full.shape[0])])
        try:
            coef_full, _, _, _ = np.linalg.lstsq(design_full, y, rcond=None)
        except Exception:
            return None

        terms = []
        selected_terms = []
        for weight, idx in zip(coef_full[:-1], active):
            if not np.isfinite(weight) or abs(float(weight)) < 1e-8:
                continue
            selected_terms.append(pool[idx]["name"])
            terms.append(f"({float(weight):.12g})*({pool[idx]['name']})")
        bias = float(coef_full[-1])
        if abs(bias) > 1e-8 or not terms:
            terms.append(f"({bias:.12g})")
        formula = "+".join(terms) if terms else "0"

        try:
            pred_full = self._safe_eval_formula_array(formula, X)
        except Exception:
            return None
        full_mse = float(np.mean((pred_full - y) ** 2))
        if not np.isfinite(full_mse):
            return None

        val_r2 = 1.0 - float(best["validation_mse"]) / y_val_var
        return {
            "formula": formula,
            "mse": full_mse,
            "validation_mse": float(best["validation_mse"]),
            "validation_r2": float(val_r2),
            "selected_terms": selected_terms,
            "n_terms": len(selected_terms),
            "complexity": self._formula_complexity(formula),
            "source": "engineered_basis",
        }

    def _compute_runtime_noise_diagnostics(self, X=None, y=None, formula=None):
        """Residual / weight diagnostics for noise-aware routing (Phase 7).

        Uses incumbent formula residuals when available; otherwise target-only
        proxies (weight ESS). Safe when inference fails — returns zeros.

        Phase E+: also residual RMS / signal-scale outlier rate so white and
        sparse-spike tiers do not under-detect as ``clean``.
        """
        diag = {
            "residual_autocorr": 0.0,
            "outlier_fraction": 0.0,
            "signal_outlier_fraction": 0.0,
            "residual_rms_ratio": 0.0,
            "validation_gap": 0.0,
            "ess_ratio": 1.0,
            "effective_sample_size": None,
            "noise_band": "clean",
            "n_samples": 0,
        }
        try:
            n = 0
            if y is not None:
                n = int(np.asarray(y).reshape(-1).shape[0])
            elif X is not None:
                n = int(np.asarray(X).shape[0])
            diag["n_samples"] = n
            w = self._active_sample_weight(n_targets=n if n > 0 else None)
            ess = _effective_sample_size(w) if w is not None else (float(n) if n else None)
            diag["effective_sample_size"] = ess
            if ess is not None and n > 0:
                diag["ess_ratio"] = float(np.clip(ess / float(n), 0.0, 1.0))

            text = str(formula or getattr(self, "formula_", "") or "").strip()
            if X is not None and y is not None and text:
                X_arr = np.asarray(X, dtype=np.float64)
                y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
                try:
                    pred = self._safe_eval_formula_array(text, X_arr)
                    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
                except Exception:
                    pred = None
                if pred is not None and pred.shape == y_arr.shape and np.all(np.isfinite(pred)):
                    resid = pred - y_arr
                    diag["residual_autocorr"] = _residual_lag1_autocorr(resid)
                    diag["outlier_fraction"] = _estimate_outlier_fraction(resid, w)
                    diag["signal_outlier_fraction"] = _signal_scale_outlier_fraction(
                        resid, y_arr, k=2.5
                    )
                    diag["residual_rms_ratio"] = _residual_rms_ratio(resid, y_arr)
                    # Holdout generalization gap (unweighted display contract).
                    try:
                        split = self._domain_edge_validation_split(X_arr, y_arr, validation_fraction=0.2)
                    except Exception:
                        split = None
                    if split is not None:
                        try:
                            p_fit = self._safe_eval_formula_array(text, split["X_fit"])
                            p_val = self._safe_eval_formula_array(text, split["X_val"])
                            fit_mse = float(np.mean((p_fit - split["y_fit"]) ** 2))
                            val_mse = float(np.mean((p_val - split["y_val"]) ** 2))
                            y_var = max(float(np.var(split["y_val"])), 1e-12)
                            if np.isfinite(fit_mse) and np.isfinite(val_mse):
                                diag["validation_gap"] = float(max(0.0, val_mse - fit_mse) / y_var)
                        except Exception:
                            pass
            diag["noise_band"] = _noise_band_from_diagnostics(diag)
        except Exception:
            diag["noise_band"] = "clean"
        self._runtime_noise_diagnostics_ = diag
        if isinstance(getattr(self, "blackbox_diagnostics_", None), dict):
            self.blackbox_diagnostics_["runtime_noise"] = dict(diag)
        return diag

    def _derive_blackbox_search_plan(
        self,
        blackbox_state,
        *,
        fast_path_uncertainty=None,
        proposer_uncertainty=None,
        proposer_plan=None,
        candidate_screening=None,
        noise_diagnostics=None,
    ):
        """Heuristically scale breadth/depth; calibrate thresholds by noise band."""
        noise_diag = noise_diagnostics if isinstance(noise_diagnostics, dict) else (
            getattr(self, "_runtime_noise_diagnostics_", None) or {}
        )
        noise_band = str(noise_diag.get("noise_band") or _noise_band_from_diagnostics(noise_diag) or "clean")
        band_thr = dict(_NOISE_BAND_THRESHOLDS.get(noise_band, _NOISE_BAND_THRESHOLDS["clean"]))
        base_plan = {
            "uncertainty_score": 0.0,
            "selection_uncertainty": 0.0,
            "interaction_pressure": 0.0,
            "candidate_strength": 0.0,
            "candidate_diversity": 0.0,
            "breadth_multiplier": 1.0,
            "depth_multiplier": 1.0,
            "generation_multiplier": 1.0,
            "population_multiplier": 1.0,
            "seed_budget": 8,
            "screening_budget": 8,
            "basis_max_terms": 4,
            "candidate_acceptance_r2": band_thr["candidate_acceptance_r2"],
            "candidate_shrink_r2": band_thr["candidate_shrink_r2"],
            "acceptable_complexity": 15,
            "early_stop_max_nodes": 50,
            "timeout_multiplier": 1.0,
            "focus": "balanced",
            "noise_band": noise_band,
            "prediction_uncertain_entropy": band_thr["prediction_uncertain_entropy"],
            "noise_routing": {
                "residual_autocorr": float(noise_diag.get("residual_autocorr") or 0.0),
                "outlier_fraction": float(noise_diag.get("outlier_fraction") or 0.0),
                "signal_outlier_fraction": float(noise_diag.get("signal_outlier_fraction") or 0.0),
                "residual_rms_ratio": float(noise_diag.get("residual_rms_ratio") or 0.0),
                "validation_gap": float(noise_diag.get("validation_gap") or 0.0),
                "ess_ratio": float(noise_diag.get("ess_ratio") if noise_diag.get("ess_ratio") is not None else 1.0),
            },
            "allowed_unary_ops": [],
            "multi_allowed_unary_ops": [],
            "binary_op_priors": [],
            "multi_binary_op_priors": [],
            "allowed_binary_ops": [],
            "multi_allowed_binary_ops": [],
        }
        if blackbox_state is None or not getattr(blackbox_state, "enabled", False):
            # Still apply noise-band calibration for univariate / non-blackbox plans.
            return base_plan

        selected = list(getattr(blackbox_state, "selected_features", []) or [])
        selected_count = max(1, len(selected))
        original_count = int(getattr(self, "original_n_features_in_", selected_count) or selected_count)

        feature_scores = getattr(blackbox_state, "feature_scores", {}) or {}
        score_values = sorted(
            [float(v) for v in feature_scores.values() if np.isfinite(v)],
            reverse=True,
        )
        top_score = score_values[0] if score_values else 0.0
        next_score = score_values[1] if len(score_values) > 1 else 0.0
        score_gap = max(0.0, top_score - next_score)
        score_gap_ratio = score_gap / max(abs(top_score), 1e-12) if top_score > 0 else 0.0

        selection_uncertainty = 1.0 - float(np.clip(score_gap_ratio, 0.0, 1.0))
        if getattr(blackbox_state, "feature_selection_uncertain", False):
            selection_uncertainty = max(selection_uncertainty, 0.85)
        if getattr(blackbox_state, "reason", "") == "retained_all_features_small_problem":
            selection_uncertainty *= 0.5

        interaction_scores = getattr(blackbox_state, "interaction_scores", {}) or {}
        interaction_terms = list(getattr(blackbox_state, "interaction_terms", []) or [])
        interaction_best = max(
            [float(v) for v in interaction_scores.values() if np.isfinite(v)],
            default=0.0,
        )
        interaction_density = len(interaction_terms) / max(1.0, float(selected_count - 1))
        interaction_pressure = float(np.clip(0.55 * interaction_best + 0.15 * interaction_density, 0.0, 1.0))

        feature_span_pressure = float(np.clip((selected_count - 1) / max(4.0, original_count - 1), 0.0, 1.0))

        screening = candidate_screening if isinstance(candidate_screening, dict) else {}
        candidate_best_r2 = float(np.clip(float(screening.get("best_validation_r2", 0.0) or 0.0), 0.0, 1.0))
        candidate_count = int(max(0, screening.get("candidate_count", 0) or 0))
        candidate_family_count = int(max(0, screening.get("family_count", 0) or 0))
        candidate_strength = candidate_best_r2
        candidate_diversity = float(np.clip(candidate_family_count / max(1.0, min(6.0, float(candidate_count or 1))), 0.0, 1.0))

        def _uncertainty_signal(payload):
            if not isinstance(payload, dict):
                return 0.0
            entropy = payload.get("prediction_entropy")
            margin = payload.get("prediction_margin")
            uncertain_flag = bool(payload.get("prediction_uncertain", False))
            signal = 0.0
            if entropy is not None:
                try:
                    signal = max(signal, float(np.clip(float(entropy), 0.0, 1.0)))
                except Exception:
                    pass
            if margin is not None:
                try:
                    margin = float(margin)
                    signal = max(signal, float(np.clip(1.0 - min(max(margin, 0.0), 1.0), 0.0, 1.0)))
                except Exception:
                    pass
            if uncertain_flag:
                signal = max(signal, 0.75)
            return float(np.clip(signal, 0.0, 1.0))

        fast_uncertainty = _uncertainty_signal(fast_path_uncertainty)
        proposer_unc = 0.0
        if isinstance(proposer_uncertainty, dict):
            proposer_unc = _uncertainty_signal(proposer_uncertainty)
        elif proposer_uncertainty is not None:
            proposer_unc = float(np.clip(float(proposer_uncertainty), 0.0, 1.0))

        # Phase 7 / E+: residual geometry + amplitude — expand budget on noise.
        outlier_frac = float(np.clip(float(noise_diag.get("outlier_fraction") or 0.0), 0.0, 1.0))
        signal_out = float(np.clip(float(noise_diag.get("signal_outlier_fraction") or 0.0), 0.0, 1.0))
        outlier_eff = max(outlier_frac, signal_out)
        val_gap = float(np.clip(float(noise_diag.get("validation_gap") or 0.0), 0.0, 1.0))
        resid_ac = float(np.clip(abs(float(noise_diag.get("residual_autocorr") or 0.0)), 0.0, 1.0))
        ess_ratio = noise_diag.get("ess_ratio")
        ess_ratio = float(ess_ratio) if ess_ratio is not None and np.isfinite(float(ess_ratio)) else 1.0
        ess_ratio = float(np.clip(ess_ratio, 0.0, 1.0))
        rms_ratio = float(np.clip(float(noise_diag.get("residual_rms_ratio") or 0.0), 0.0, 1.0))
        # Weights sum to 1.0; residual RMS catches white noise amplitude.
        noise_pressure = float(np.clip(
            0.30 * outlier_eff
            + 0.20 * val_gap
            + 0.10 * resid_ac
            + 0.10 * (1.0 - ess_ratio)
            + 0.30 * rms_ratio,
            0.0,
            1.0,
        ))

        uncertainty_score = float(np.clip(
            0.28 * selection_uncertainty
            + 0.18 * interaction_pressure
            + 0.14 * fast_uncertainty
            + 0.10 * proposer_unc
            + 0.10 * (1.0 - candidate_strength)
            + 0.20 * noise_pressure,
            0.0,
            1.0,
        ))

        breadth_multiplier = float(np.clip(
            1.0
            + 0.85 * selection_uncertainty
            + 0.22 * fast_uncertainty
            + 0.15 * (1.0 - interaction_pressure)
            + 0.15 * (1.0 - candidate_strength)
            + 0.45 * noise_pressure,
            0.75,
            3.5,
        ))
        depth_multiplier = float(np.clip(
            1.0
            + 0.60 * interaction_pressure
            + 0.22 * proposer_unc
            + 0.18 * feature_span_pressure
            + 0.18 * candidate_diversity
            + 0.35 * noise_pressure
            - 0.25 * candidate_strength,
            0.75,
            4.0,
        ))
        # Do NOT shrink search just because noisy candidate MSE/R2 looks good under noise.
        if noise_band in ("medium", "high") and candidate_strength >= 0.90:
            depth_multiplier = max(depth_multiplier, 1.15)
            breadth_multiplier = max(breadth_multiplier, 1.10)
        # Soft clean-shrink: only when residual amplitude AND pressure are tiny.
        # Prevents under-expansion when band lags but RMS/spikes are real.
        if (
            uncertainty_score < 0.3
            and noise_band == "clean"
            and noise_pressure < 0.05
            and rms_ratio < 0.03
        ):
            breadth_multiplier *= 0.85
            depth_multiplier *= 0.9
        elif uncertainty_score > 0.7 or noise_band == "high" or noise_pressure >= 0.25:
            breadth_multiplier *= 1.08
            depth_multiplier *= 1.08

        # Phase C / E+: when blackbox is noisy and/or selection-uncertain, do not
        # over-clamp Track-1 budget (literature: best-case recovery needs room).
        hard_blackbox_caps = True
        relax_blackbox_caps = (
            getattr(blackbox_state, "enabled", False)
            and (
                noise_band in ("medium", "high", "low")
                or selection_uncertainty >= 0.70
                or noise_pressure >= 0.15
                or rms_ratio >= 0.08
            )
        )
        if getattr(blackbox_state, "enabled", False):
            # For blackbox Track 1, spend uncertainty budget on screening first.
            if relax_blackbox_caps:
                breadth_multiplier = min(breadth_multiplier, 2.75)
                depth_multiplier = min(depth_multiplier, 3.0)
                hard_blackbox_caps = False
            else:
                breadth_multiplier = min(breadth_multiplier, 2.25)
                depth_multiplier = min(depth_multiplier, 2.5)

        generation_multiplier = float(np.clip(depth_multiplier, 0.75, 4.0))
        population_multiplier = float(np.clip(breadth_multiplier, 0.75, 3.5))
        seed_budget = int(np.clip(
            round(8 + 8 * selection_uncertainty + 4 * interaction_pressure + 3 * fast_uncertainty + 2 * proposer_unc + 2 * candidate_diversity),
            8,
            24,
        ))
        screening_budget = int(np.clip(
            round(8 + 10 * selection_uncertainty + 8 * interaction_pressure + 6 * (1.0 - candidate_strength) + 3 * candidate_diversity),
            6,
            28,
        ))
        basis_max_terms = int(np.clip(
            round(3 + 2 * interaction_pressure + 2 * candidate_diversity + 2 * (1.0 - candidate_strength)),
            2,
            8,
        ))
        # Calibrate accept/shrink per noise band; interaction still softens slightly.
        band_accept = float(band_thr["candidate_acceptance_r2"])
        band_shrink = float(band_thr["candidate_shrink_r2"])
        candidate_acceptance_r2 = float(np.clip(
            band_accept - 0.02 * interaction_pressure - 0.01 * candidate_diversity,
            0.90,
            0.985,
        ))
        candidate_shrink_r2 = float(np.clip(
            min(band_shrink, candidate_acceptance_r2 - 0.03),
            0.84,
            0.96,
        ))
        # High residual noise: raise bar to accept / shrink (avoid false confidence).
        if noise_band == "high":
            candidate_acceptance_r2 = max(candidate_acceptance_r2, 0.94)
            candidate_shrink_r2 = min(candidate_shrink_r2, candidate_acceptance_r2 - 0.05)

        if getattr(blackbox_state, "enabled", False):
            if hard_blackbox_caps:
                generation_multiplier = float(np.clip(generation_multiplier, 0.75, 2.0))
                population_multiplier = float(np.clip(population_multiplier, 0.80, 1.85))
                seed_budget = int(np.clip(seed_budget, 6, 14))
                screening_budget = int(np.clip(screening_budget, 8, 24))
                basis_max_terms = int(np.clip(basis_max_terms, 3, 6))
            else:
                generation_multiplier = float(np.clip(generation_multiplier, 0.75, 2.75))
                population_multiplier = float(np.clip(population_multiplier, 0.80, 2.25))
                seed_budget = int(np.clip(seed_budget, 6, 18))
                screening_budget = int(np.clip(screening_budget, 8, 28))
                basis_max_terms = int(np.clip(basis_max_terms, 3, 7))

        acceptable_complexity = int(np.clip(
            round(15 + 5 * uncertainty_score + 3 * interaction_pressure + 2 * feature_span_pressure + 2 * candidate_diversity),
            10,
            80,
        ))
        early_stop_max_nodes = int(np.clip(
            round(50 + 14 * uncertainty_score + 6 * interaction_pressure + 5 * feature_span_pressure - 6 * candidate_strength),
            10,
            120,
        ))
        timeout_multiplier = float(np.clip(
            0.82 + 0.16 * breadth_multiplier + 0.24 * depth_multiplier + 0.14 * (screening_budget / 12.0),
            0.8,
            2.8,
        ))

        if getattr(blackbox_state, "enabled", False):
            if hard_blackbox_caps:
                acceptable_complexity = int(np.clip(acceptable_complexity, 10, 32))
                early_stop_max_nodes = int(np.clip(early_stop_max_nodes, 16, 64))
                timeout_multiplier = float(np.clip(timeout_multiplier, 0.75, 1.0))
            else:
                acceptable_complexity = int(np.clip(acceptable_complexity, 10, 48))
                early_stop_max_nodes = int(np.clip(early_stop_max_nodes, 16, 80))
                timeout_multiplier = float(np.clip(timeout_multiplier, 0.75, 1.45))

        focus = "balanced"
        if candidate_strength >= candidate_acceptance_r2:
            focus = "screen_accept"
        elif screening_budget >= seed_budget + 4:
            focus = "screening"
        elif breadth_multiplier > depth_multiplier + 0.25:
            focus = "breadth"
        elif depth_multiplier > breadth_multiplier + 0.25:
            focus = "depth"

        binary_op_priors, multi_binary_op_priors = self._derive_blackbox_binary_priors(
            blackbox_state,
            {},
        )
        allowed_unary_ops, multi_allowed_unary_ops, allowed_binary_ops, multi_allowed_binary_ops = (
            self._derive_blackbox_unary_policy(blackbox_state, {})
        )
        plan = {
            "uncertainty_score": uncertainty_score,
            "selection_uncertainty": selection_uncertainty,
            "interaction_pressure": interaction_pressure,
            "candidate_strength": candidate_strength,
            "candidate_diversity": candidate_diversity,
            "noise_pressure": noise_pressure,
            "noise_band": noise_band,
            "prediction_uncertain_entropy": band_thr["prediction_uncertain_entropy"],
            "noise_routing": {
                "residual_autocorr": resid_ac,
                "outlier_fraction": outlier_frac,
                "validation_gap": val_gap,
                "ess_ratio": ess_ratio,
            },
            "breadth_multiplier": breadth_multiplier,
            "depth_multiplier": depth_multiplier,
            "generation_multiplier": generation_multiplier,
            "population_multiplier": population_multiplier,
            "seed_budget": seed_budget,
            "screening_budget": screening_budget,
            "basis_max_terms": basis_max_terms,
            "candidate_acceptance_r2": candidate_acceptance_r2,
            "candidate_shrink_r2": candidate_shrink_r2,
            "acceptable_complexity": acceptable_complexity,
            "early_stop_max_nodes": early_stop_max_nodes,
            "timeout_multiplier": timeout_multiplier,
            "focus": focus,
            "allowed_unary_ops": allowed_unary_ops,
            "multi_allowed_unary_ops": multi_allowed_unary_ops,
            "binary_op_priors": binary_op_priors,
            "multi_binary_op_priors": multi_binary_op_priors,
            "allowed_binary_ops": allowed_binary_ops,
            "multi_allowed_binary_ops": multi_allowed_binary_ops,
        }

        if proposer_plan:
            # Let proposer guidance add seeds, but keep blackbox screening-first caps.
            raw_generation_multiplier = plan["generation_multiplier"] * float(
                _clamp_float(proposer_plan.get("generation_multiplier"), 1.0, 0.5, 4.0)
            )
            raw_population_multiplier = plan["population_multiplier"] * float(
                _clamp_float(proposer_plan.get("population_multiplier"), 1.0, 0.5, 3.0)
            )
            raw_seed_budget = max(plan["seed_budget"], int(proposer_plan.get("seed_budget", plan["seed_budget"])))
            raw_complexity = max(
                plan["acceptable_complexity"],
                int(proposer_plan.get("acceptable_complexity", plan["acceptable_complexity"])),
            )
            raw_max_nodes = max(
                plan["early_stop_max_nodes"],
                int(proposer_plan.get("early_stop_max_nodes", plan["early_stop_max_nodes"])),
            )
            if getattr(blackbox_state, "enabled", False):
                plan["generation_multiplier"] = float(np.clip(raw_generation_multiplier, 0.75, 2.0))
                plan["population_multiplier"] = float(np.clip(raw_population_multiplier, 0.80, 1.85))
                plan["seed_budget"] = int(np.clip(raw_seed_budget, 6, 14))
                plan["acceptable_complexity"] = int(np.clip(raw_complexity, 10, 32))
                plan["early_stop_max_nodes"] = int(np.clip(raw_max_nodes, 16, 64))
                plan["screening_budget"] = max(plan["screening_budget"], plan["seed_budget"])
            else:
                plan["generation_multiplier"] = raw_generation_multiplier
                plan["population_multiplier"] = raw_population_multiplier
                plan["seed_budget"] = raw_seed_budget
                plan["acceptable_complexity"] = raw_complexity
                plan["early_stop_max_nodes"] = raw_max_nodes
            plan["timeout_multiplier"] = float(np.clip(
                plan["timeout_multiplier"] * float(_clamp_float(proposer_plan.get("timeout_multiplier"), 1.0, 0.5, 3.0)),
                0.75 if getattr(blackbox_state, "enabled", False) else 0.8,
                1.0 if getattr(blackbox_state, "enabled", False) else 3.0,
            ))
        return plan

    def _resolve_classifier_path(self):
        """Resolve classifier model path relative to repo root."""
        p = Path(self.classifier_path)
        if p.is_absolute() and p.exists():
            return str(p)
        repo_path = _REPO_ROOT / self.classifier_path
        if repo_path.exists():
            return str(repo_path)
        return str(p)

    def _resolve_universal_proposer_path(self):
        """Resolve proposer checkpoint path relative to repo root with fallback."""
        candidates = [
            self.universal_proposer_path,
            "models/universal_proposer_multi.pt",
            "models/universal_proposer_robust.pt",
        ]
        for candidate in candidates:
            p = Path(candidate)
            if p.is_absolute() and p.exists():
                return str(p)
            repo_path = _REPO_ROOT / candidate
            if repo_path.exists():
                return str(repo_path)
        return str(Path(self.universal_proposer_path))

    def _safe_eval_formula_array(self, formula, X):
        """Safely evaluate a symbolic formula over a feature matrix."""
        def _safe_log(x):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return np.where(
                    np.abs(x) > 1e-300,
                    np.log(np.abs(x) + 1e-300),
                    -300.0,
                )

        def _safe_sqrt(x):
            return np.sqrt(np.maximum(x, 0.0))

        def _signed_power(base, power):
            base_arr = np.asarray(base, dtype=np.float64)
            power_val = float(power)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                if power_val < 0:
                    return np.sign(base_arr) / ((np.abs(base_arr) + 1e-12) ** abs(power_val))
                return np.sign(base_arr) * (np.abs(base_arr) ** power_val)

        context = {
            "np": np,
            "log": _safe_log,
            "sin": np.sin,
            "cos": np.cos,
            "exp": lambda x: np.exp(np.clip(x, -500, 500)),
            "sqrt": _safe_sqrt,
            "abs": np.abs,
            "Abs": np.abs,
            "sign": np.sign,
            "_signed_power": _signed_power,
            "pi": np.pi,
            "E": np.e,
            "e": np.e,
        }
        for i in range(X.shape[1]):
            context[f"x{i}"] = X[:, i]
        if X.shape[1] == 1:
            context["x"] = X[:, 0]

        expr = formula.strip()
        expr = re.sub(r'\|([^|]+)\|', r'abs(\1)', expr)
        expr = re.sub(r'\^', r'**', expr)
        expr = expr.replace('np.', '')
        if "**" in expr:
            try:
                from scripts.benchmark_common import protect_fractional_powers
                expr = protect_fractional_powers(expr)
            except Exception:
                pass

        self.formula_eval_count_ = int(getattr(self, "formula_eval_count_", 0) or 0) + 1
        cache = getattr(self, "_formula_eval_cache_", None)
        if not isinstance(cache, dict):
            cache = {}
            self._formula_eval_cache_ = cache
        cache_key = (expr, id(X), tuple(getattr(X, "shape", ())), str(getattr(X, "dtype", "")))
        cached = cache.get(cache_key)
        if cached is not None:
            self.formula_eval_cache_hits_ = int(getattr(self, "formula_eval_cache_hits_", 0) or 0) + 1
            return np.asarray(cached, dtype=np.float64).copy()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_pred = eval(expr, {"__builtins__": None}, context)

        if isinstance(y_pred, (int, float)):
            y_pred = np.full(X.shape[0], y_pred, dtype=np.float64)
        else:
            y_pred = np.asarray(y_pred, dtype=np.float64)
        out = np.where(np.isfinite(y_pred), y_pred, 0.0)
        if len(cache) >= 512:
            cache.clear()
        cache[cache_key] = np.asarray(out, dtype=np.float64).copy()
        return out

    def _formula_domain_failure_rate(self, formula, X):
        """Estimate how often a displayed formula leaves its numeric domain."""
        text = str(formula or "").strip()
        if not text:
            return None
        context = {
            "np": np,
            "log": np.log,
            "sin": np.sin,
            "cos": np.cos,
            "exp": np.exp,
            "sqrt": np.sqrt,
            "abs": np.abs,
            "Abs": np.abs,
            "sign": np.sign,
            "pi": np.pi,
            "E": np.e,
            "e": np.e,
        }
        for i in range(X.shape[1]):
            context[f"x{i}"] = X[:, i]
        if X.shape[1] == 1:
            context["x"] = X[:, 0]

        expr = re.sub(r'\|([^|]+)\|', r'abs(\1)', text)
        expr = re.sub(r'\^', r'**', expr)
        expr = expr.replace('np.', '')
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = eval(expr, {"__builtins__": None}, context)
            if isinstance(raw, (int, float)):
                raw = np.full(X.shape[0], raw, dtype=np.float64)
            raw_arr = np.asarray(raw, dtype=np.float64).reshape(-1)
            if raw_arr.shape[0] != X.shape[0]:
                return 1.0
            return float(np.mean(~np.isfinite(raw_arr)))
        except Exception:
            return 1.0

    def _passes_cross_validation_skip_guard(self, formula, X, y, sample_weight=None):
        """Return True when fast-path formula is stable enough to skip evolution."""
        diagnostics = {
            'enabled': bool(self.cv_skip_guard_enabled),
            'fold_r2': [],
            'min_fold_r2': None,
            'std_fold_r2': None,
            'passed': True,
            'reason': 'disabled',
        }

        if not self.cv_skip_guard_enabled:
            self.fast_path_cv_guard_ = diagnostics
            return True

        n_samples = int(X.shape[0])
        n_folds = int(max(2, self.cv_skip_guard_folds))
        if n_samples < int(max(n_folds * 2, self.cv_skip_guard_min_samples)):
            diagnostics['reason'] = 'insufficient_samples'
            self.fast_path_cv_guard_ = diagnostics
            return True

        try:
            y_pred = self._safe_eval_formula_array(formula, X)
        except Exception:
            diagnostics['passed'] = False
            diagnostics['reason'] = 'formula_eval_failed'
            self.fast_path_cv_guard_ = diagnostics
            return False

        idx = np.arange(n_samples)
        seed = 0 if self.random_state is None else int(self.random_state)
        rng = np.random.RandomState(seed)
        rng.shuffle(idx)
        folds = [f for f in np.array_split(idx, n_folds) if len(f) > 0]

        try:
            w_all = self._active_sample_weight(
                n_targets=n_samples,
                sample_weight=sample_weight,
            )
        except ValueError as exc:
            diagnostics['passed'] = False
            diagnostics['reason'] = f'sample_weight_error:{exc}'
            self.fast_path_cv_guard_ = diagnostics
            return False
        fold_r2 = []
        for fold_idx in folds:
            y_fold = y[fold_idx]
            pred_fold = y_pred[fold_idx]
            w_fold = None if w_all is None else w_all[fold_idx]
            r2_fold = _weighted_r2(pred_fold, y_fold, w_fold)
            if np.isfinite(r2_fold):
                fold_r2.append(float(r2_fold))

        diagnostics['fold_r2'] = fold_r2
        if not fold_r2:
            diagnostics['passed'] = False
            diagnostics['reason'] = 'no_valid_folds'
            self.fast_path_cv_guard_ = diagnostics
            return False

        min_fold_r2 = float(np.min(fold_r2))
        std_fold_r2 = float(np.std(fold_r2))
        diagnostics['min_fold_r2'] = min_fold_r2
        diagnostics['std_fold_r2'] = std_fold_r2

        passed = (
            min_fold_r2 >= float(self.cv_skip_guard_min_fold_r2)
            and std_fold_r2 <= float(self.cv_skip_guard_max_r2_std)
        )
        diagnostics['passed'] = bool(passed)
        diagnostics['reason'] = 'ok' if passed else 'unstable_fold_performance'
        self.fast_path_cv_guard_ = diagnostics
        return bool(passed)

    def _run_universal_proposer_dual_path(self, X, y, fast_path_result, blackbox_state=None):
        """Optional side-by-side proposer run for routing diagnostics.

        Returns:
            Tuple[fpip_payload_or_none, force_evolution_bool]
        """
        if not self.use_universal_proposer:
            return None, False

        try:
            from glassbox.universal_proposer import (
                load_universal_proposer_checkpoint,
                propose_fpip_v2_from_xy,
            )

            if self._universal_proposer_model is None:
                model_path = self._resolve_universal_proposer_path()
                self._universal_proposer_model = load_universal_proposer_checkpoint(model_path, device=self.device)

            X_arr = np.asarray(X, dtype=np.float64)
            if X_arr.ndim == 1 or int(X_arr.shape[1]) == 1:
                x1 = X_arr.reshape(-1)
                proposer_status = "ok"
            else:
                x1 = X_arr
                proposer_status = "ok_multivariate_heuristic"

            y1 = np.asarray(y, dtype=np.float64).reshape(-1)

            fit_diag = {}
            if isinstance(fast_path_result, dict):
                fit_diag["mse"] = fast_path_result.get("mse")
                fit_diag["residual_suspicious"] = bool(
                    (fast_path_result.get("residual_diagnostics") or {}).get("residual_suspicious", False)
                )

            payload = propose_fpip_v2_from_xy(
                self._universal_proposer_model,
                x=x1,
                y=y1,
                top_k=int(max(1, self.universal_proposer_top_k)),
                fit_diagnostics=fit_diag,
                interaction_hints={
                    "multivariate_proxy": False,
                    "selected_feature_count": int(np.asarray(X).shape[1]) if np.asarray(X).ndim > 1 else 1,
                    "selected_features": list(getattr(blackbox_state, "selected_features", [])) if blackbox_state is not None else [],
                    "dropped_features": list(getattr(blackbox_state, "dropped_features", [])) if blackbox_state is not None else [],
                },
                device=self.device,
            )

            if not payload:
                self.universal_proposer_status_ = "error:empty_payload"
                if self.universal_proposer_log_routing:
                    print("  [Proposer skipped: empty payload]")
                return None, False

            self.universal_proposer_status_ = proposer_status
            self.universal_proposer_fpip_v2_ = payload

            if self.universal_proposer_log_routing:
                route = payload.get("routing_signal") or {}
                print(
                    "  [Proposer] "
                    f"guided={route.get('recommend_guided_evolution')} "
                    f"reason={route.get('reason')}"
                )

            force_evolution = (
                (not self.universal_proposer_shadow_mode)
                and bool(payload.get("valid", False))
                and bool((payload.get("routing_signal") or {}).get("recommend_guided_evolution", False))
            )
            return payload, force_evolution
        except Exception as e:
            self.universal_proposer_status_ = f"error:{e}"
            if self.universal_proposer_log_routing:
                print(f"  [Proposer skipped: {e}]")
            return None, False

    def _simplify_formula(self, formula):
        """Apply multipass formula simplification."""
        if not formula or not self.use_simplification:
            return formula
        try:
            from glassbox.sr.cpp import _core
            simplified = _core.simplify_formula(
                formula,
                int_tol=self.simplification_int_tol,
                zero_tol=self.simplification_zero_tol,
                max_passes=6,
                use_nsimplify=True,
                use_identities=True,
                n_features=self.n_features_in_
            )
            return simplified
        except Exception:
            return formula

    def _stage_residual_symbolic_fit(self, X, y, base_formula, *, _allow_recursion=False):
        """Fit a second symbolic stage on the residual when it improves holdout fit."""
        import time as _time
        _phase_start = _time.time()
        try:
            return self._stage_residual_symbolic_fit_impl(X, y, base_formula, _allow_recursion=_allow_recursion)
        finally:
            self._add_phase_time("residual_symbolic_fit", _time.time() - _phase_start)

    def _stage_residual_symbolic_fit_impl(self, X, y, base_formula, *, _allow_recursion=False):
        """Implementation for _stage_residual_symbolic_fit with timing wrapper."""
        self._residual_stage_guard_ = {
            "enabled": bool(self.enable_residual_stage),
            "allowed": bool(_allow_recursion),
            "mode": "bounded_mini_search",
            "accepted": False,
        }
        if not self.enable_residual_stage or not _allow_recursion or not base_formula or not self.use_guided_evolution:
            self._residual_stage_guard_["reason"] = "disabled_or_not_allowed"
            return None
        if X.shape[1] < 1:
            self._residual_stage_guard_["reason"] = "no_features"
            return None

        try:
            y_pred = self._safe_eval_formula_array(base_formula, X)
        except Exception:
            self._residual_stage_guard_["reason"] = "base_eval_failed"
            return None

        residual = np.asarray(y, dtype=np.float64).reshape(-1) - np.asarray(y_pred, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(residual)) or float(np.var(residual)) < 1e-12:
            self._residual_stage_guard_["reason"] = "flat_or_nonfinite_residual"
            return None

        candidate_pool = self._build_residual_mini_search_candidates(
            X,
            residual,
            base_formula,
            max_candidates=getattr(self, "residual_mini_search_max_candidates", 64),
        )
        self._residual_stage_guard_["candidate_count"] = len(candidate_pool)
        if not candidate_pool:
            self._residual_stage_guard_["reason"] = "no_candidate_pool"
            return None

        top_k = max(1, int(getattr(self, "residual_mini_search_refine_top_k", 6)))
        refined = self._refine_candidate_formulas(
            candidate_pool,
            X,
            residual,
            max_candidates=top_k,
        )
        if not refined:
            self._residual_stage_guard_["reason"] = "no_refined_residual_candidates"
            return None

        split = self._domain_edge_validation_split(X, y, validation_fraction=0.2)
        if split is None:
            self._residual_stage_guard_["reason"] = "no_validation_split"
            return None

        # Phase 6: residual acceptance requires weighted val improvement AND
        # unweighted/edge val not worse beyond noise-aware slack.
        n_total = int(np.asarray(y).reshape(-1).shape[0])
        try:
            _, val_w = self._split_sample_weights(split, n_total=n_total)
        except Exception:
            val_w = None
        edge_split = None
        edge_w = None
        try:
            edge_split = self._domain_edge_validation_split(X, y, validation_fraction=0.25)
            if edge_split is not None:
                _, edge_w = self._split_sample_weights(edge_split, n_total=n_total)
        except Exception:
            edge_split = None
            edge_w = None

        try:
            base_pred = self._safe_eval_formula_array(base_formula, split["X_val"])
        except Exception:
            self._residual_stage_guard_["reason"] = "base_val_eval_failed"
            return None
        base_pred = np.asarray(base_pred, dtype=np.float64).reshape(-1)
        y_val = np.asarray(split["y_val"], dtype=np.float64).reshape(-1)

        base_mse_u = float(np.mean((base_pred - y_val) ** 2))
        try:
            base_mse_w = _weighted_mse(base_pred, y_val, val_w) if val_w is not None else base_mse_u
        except Exception:
            base_mse_w = base_mse_u
        if not np.isfinite(base_mse_w):
            self._residual_stage_guard_["reason"] = "base_mse_nonfinite"
            return None

        _, abs_slack, slack_diag = self._noise_aware_cleanup_slack(
            base_formula, split["X_val"], y_val, relative_slack=0.08, absolute_slack=1e-10
        )
        rel_slack = float(slack_diag.get("relative_slack", 0.08))

        base_edge_mse_u = None
        base_edge_pred = None
        if edge_split is not None:
            try:
                base_edge_pred = self._safe_eval_formula_array(base_formula, edge_split["X_val"])
                base_edge_pred = np.asarray(base_edge_pred, dtype=np.float64).reshape(-1)
                base_edge_mse_u = float(np.mean((base_edge_pred - edge_split["y_val"]) ** 2))
            except Exception:
                base_edge_mse_u = None
                base_edge_pred = None

        best = None
        reject_noise = 0
        for cand in refined:
            formula = str((cand or {}).get("formula", "")).strip()
            if not formula or formula == "0":
                continue
            try:
                res_pred = self._safe_eval_formula_array(formula, split["X_val"])
                res_pred = np.asarray(res_pred, dtype=np.float64).reshape(-1)
                combined = base_pred + res_pred
                combined_mse_u = float(np.mean((combined - y_val) ** 2))
                combined_mse_w = (
                    _weighted_mse(combined, y_val, val_w) if val_w is not None else combined_mse_u
                )
            except Exception:
                continue
            if not np.isfinite(combined_mse_w) or not np.isfinite(combined_mse_u):
                continue
            # Must improve weighted (or unweighted if no weights) validation.
            improvement_w = base_mse_w - combined_mse_w
            if improvement_w <= max(1e-10, base_mse_w * 0.002):
                reject_noise += 1
                continue
            # Must not worsen unweighted validation beyond noise-aware slack.
            u_allowed = base_mse_u * (1.0 + rel_slack) + abs_slack
            if combined_mse_u > u_allowed:
                reject_noise += 1
                continue
            # Edge validation guard when available.
            if base_edge_pred is not None and base_edge_mse_u is not None and np.isfinite(base_edge_mse_u):
                try:
                    res_edge = self._safe_eval_formula_array(formula, edge_split["X_val"])
                    res_edge = np.asarray(res_edge, dtype=np.float64).reshape(-1)
                    comb_edge = base_edge_pred + res_edge
                    edge_mse_u = float(np.mean((comb_edge - edge_split["y_val"]) ** 2))
                    edge_allowed = base_edge_mse_u * (1.0 + rel_slack) + abs_slack
                    if np.isfinite(edge_mse_u) and edge_mse_u > edge_allowed:
                        reject_noise += 1
                        continue
                    if edge_w is not None:
                        try:
                            base_e_w = _weighted_mse(base_edge_pred, edge_split["y_val"], edge_w)
                            comb_e_w = _weighted_mse(comb_edge, edge_split["y_val"], edge_w)
                            if comb_e_w > base_e_w * (1.0 + rel_slack) + abs_slack:
                                reject_noise += 1
                                continue
                        except Exception:
                            pass
                except Exception:
                    pass
            score = (
                combined_mse_w,
                _finite_float((cand or {}).get("risk_score"), 0.0),
                _finite_float((cand or {}).get("complexity"), float("inf")),
            )
            if best is None or score < best[0]:
                best = (score, formula, combined_mse_w, combined_mse_u, cand)

        if best is None:
            self._residual_stage_guard_.update({
                "accepted": False,
                "reason": "no_holdout_improvement",
                "residual_rejected_as_noise": True,
                "noise_rejects": int(reject_noise),
                "base_mse": float(base_mse_w),
                "base_mse_unweighted": float(base_mse_u) if np.isfinite(base_mse_u) else None,
                "refined_count": len(refined),
                "relative_slack": rel_slack,
            })
            if isinstance(getattr(self, "blackbox_diagnostics_", None), dict):
                self.blackbox_diagnostics_["residual_rejected_as_noise"] = True
            return None

        _, formula, combined_mse_w, combined_mse_u, cand = best
        self._residual_stage_guard_.update({
            "accepted": True,
            "formula": formula[:240],
            "base_mse": float(base_mse_w),
            "base_mse_unweighted": float(base_mse_u) if np.isfinite(base_mse_u) else None,
            "combined_mse": float(combined_mse_w),
            "combined_mse_unweighted": float(combined_mse_u),
            "validation_r2": cand.get("validation_r2"),
            "refined_count": len(refined),
            "noise_rejects": int(reject_noise),
            "relative_slack": rel_slack,
            "weighted_validation": bool(val_w is not None),
        })
        if isinstance(getattr(self, "blackbox_diagnostics_", None), dict):
            self.blackbox_diagnostics_["residual_rejected_as_noise"] = False
        return formula

    def _build_residual_mini_search_candidates(self, X, residual, base_formula, *, max_candidates=64):
        """Build a bounded residual candidate pool without launching a nested estimator."""
        X_arr = np.asarray(X, dtype=np.float64)
        residual = np.asarray(residual, dtype=np.float64).reshape(-1)
        if X_arr.ndim != 2 or X_arr.shape[0] != residual.size:
            return []

        max_candidates = max(1, int(max_candidates))
        candidates = []
        seen = set()

        def add(formula, source):
            text = str(formula or "").strip()
            if not text or text == "0":
                return
            key = re.sub(r"\s+", "", text.lower())
            if key in seen:
                return
            seen.add(key)
            candidates.append({
                "formula": text,
                "source": source,
                "residual_mini_search": True,
                "complexity": self._formula_complexity(text),
            })

        n_features = int(X_arr.shape[1])
        feature_terms = []
        for j in range(n_features):
            name = f"x{j}"
            feature_terms.append(name)
            templates = [
                name,
                f"{name}^2",
                f"{name}^3",
                f"sin({name})",
                f"cos({name})",
                f"sin(2*{name})",
                f"cos(2*{name})",
                f"sin(3*{name})",
                f"cos(3*{name})",
                f"exp(-{name})",
                f"exp(-{name}^2)",
                f"{name}*sin({name})",
                f"{name}*cos({name})",
                f"{name}*exp(-{name})",
                f"1/(1+{name}^2)",
                f"{name}/(1+{name}^2)",
            ]
            for formula in templates:
                add(formula, "residual_template")

        if n_features >= 2:
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    xi = f"x{i}"
                    xj = f"x{j}"
                    add(f"{xi}*{xj}", "residual_interaction_template")
                    add(f"sin({xi})*cos({xj})", "residual_interaction_template")
                    add(f"{xi}/(1+{xj}^2)", "residual_interaction_template")

        for cand in list(((getattr(self, "_fp_result", {}) or {}).get("candidate_formulas") or []))[:8]:
            add((cand or {}).get("formula"), "residual_fast_path_candidate")

        # Cheap correlation prefilter so refinement sees the most residual-relevant templates first.
        scored = []
        for cand in candidates:
            formula = cand.get("formula")
            try:
                values = self._safe_eval_formula_array(formula, X_arr).reshape(-1)
            except Exception:
                continue
            mask = np.isfinite(values) & np.isfinite(residual)
            if int(mask.sum()) < 8:
                continue
            vals = values[mask]
            res = residual[mask]
            if float(np.std(vals)) < 1e-12:
                corr = 0.0
            else:
                corr = float(abs(np.corrcoef(vals, res)[0, 1]))
                if not np.isfinite(corr):
                    corr = 0.0
            merged = dict(cand)
            merged["residual_corr"] = corr
            scored.append(merged)

        scored.sort(
            key=lambda c: (
                -_finite_float(c.get("residual_corr"), 0.0),
                _finite_float(c.get("complexity"), float("inf")),
                str(c.get("formula", "")),
            )
        )
        return scored[:max_candidates]

    def _run_residual_boosting(self, X, y, base_formula):
        """Run a multi-stage symbolic boosting loop on top of base_formula."""
        import time as _time
        _phase_start = _time.time()
        try:
            return self._run_residual_boosting_impl(X, y, base_formula)
        finally:
            self._add_phase_time("residual_boosting", _time.time() - _phase_start)

    def _run_residual_boosting_impl(self, X, y, base_formula):
        """Implementation for _run_residual_boosting with timing wrapper."""
        self.boosting_stages_ = []
        self.boosting_attempted_ = False
        self.boosting_improved_ = False
        self.boosting_diagnostics_ = {
            "enabled": bool(self.enable_residual_stage and base_formula and self.use_guided_evolution),
            "base_formula": base_formula,
            "initial_holdout_r2": None,
            "final_holdout_r2": None,
            "accepted_stages": 0,
        }
        if not self.enable_residual_stage or not base_formula or not self.use_guided_evolution:
            return base_formula

        # Phase 6 tighten: residual stages often re-bloat under auto soft-MAD
        # (Nguyen-1 outliers). Skip when 1D and base already fits holdout well,
        # or when formula is already at/over the auto-weight complexity cap.
        if self._auto_noise_guard_active():
            n_feat = int(np.asarray(X).shape[1]) if np.ndim(X) == 2 else 1
            base_cx = int(self._formula_complexity(base_formula))
            limits = self._auto_weight_guard_limits(X)
            max_cx = int(limits.get("max_complexity", 22))
            try:
                pred0 = self._safe_eval_formula_array(base_formula, X)
                y0 = np.asarray(y, dtype=np.float64).reshape(-1)
                p0 = np.asarray(pred0, dtype=np.float64).reshape(-1)
                yvar = float(np.var(y0))
                base_r2_full = (
                    1.0
                    if yvar < 1e-15 and float(np.mean((y0 - p0) ** 2)) < 1e-15
                    else (1.0 - float(np.mean((y0 - p0) ** 2)) / max(yvar, 1e-15))
                )
            except Exception:
                base_r2_full = float("nan")
            skip_residual = False
            skip_reason = None
            if n_feat <= 1 and base_cx >= max(12, max_cx - 6):
                skip_residual = True
                skip_reason = "auto_weight_1d_complexity"
            elif n_feat <= 1 and np.isfinite(base_r2_full) and base_r2_full >= 0.995:
                skip_residual = True
                skip_reason = "auto_weight_1d_already_good"
            elif base_cx >= max_cx:
                skip_residual = True
                skip_reason = "auto_weight_at_complexity_cap"
            if skip_residual:
                self.boosting_diagnostics_["skipped"] = True
                self.boosting_diagnostics_["skip_reason"] = skip_reason
                self.boosting_diagnostics_["base_r2"] = (
                    float(base_r2_full) if np.isfinite(base_r2_full) else None
                )
                if isinstance(getattr(self, "blackbox_diagnostics_", None), dict):
                    self.blackbox_diagnostics_["residual_skipped_phase6"] = {
                        "reason": skip_reason,
                        "base_complexity": base_cx,
                        "base_r2": float(base_r2_full) if np.isfinite(base_r2_full) else None,
                    }
                return base_formula

        def local_r2(y_true, y_pred):
            y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
            y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
            var = float(np.var(y_true))
            if var < 1e-15:
                return 1.0 if np.mean((y_true - y_pred)**2) < 1e-15 else 0.0
            return float(1.0 - np.mean((y_true - y_pred)**2) / var)

        max_boosting_stages = getattr(self, "max_boosting_stages", 3)
        learning_rates = list(getattr(self, "boosting_learning_rates", [0.5, 0.8, 1.0]) or [0.5, 0.8, 1.0])
        if int(max_boosting_stages) <= 0:
            return base_formula

        try:
            pred = self._safe_eval_formula_array(base_formula, X)
            base_r2 = local_r2(y, pred)
            if base_r2 > 0.9999:
                self.boosting_diagnostics_["initial_r2"] = float(base_r2)
                self.boosting_diagnostics_["final_r2"] = float(base_r2)
                return base_formula
        except Exception:
            return base_formula

        current_formula = base_formula

        holdout_n = max(1, int(round(X.shape[0] * 0.2)))
        if X.shape[0] < 20 or holdout_n >= X.shape[0]:
            return base_formula

        X_fit, X_holdout = X[:-holdout_n], X[-holdout_n:]
        y_fit, y_holdout = y[:-holdout_n], y[-holdout_n:]
        del X_fit, y_fit
        # Phase 6: optional holdout weights; never accept residual on train MSE alone.
        holdout_w = None
        try:
            holdout_w = self._active_sample_weight(
                indices=np.arange(X.shape[0] - holdout_n, X.shape[0], dtype=int)
            )
        except Exception:
            holdout_w = None
        _, boost_abs_slack, boost_slack_diag = self._noise_aware_cleanup_slack(
            base_formula, X_holdout, y_holdout, relative_slack=0.05, absolute_slack=1e-10
        )
        boost_rel_slack = float(boost_slack_diag.get("relative_slack", 0.05))

        for stage in range(max_boosting_stages):
            try:
                pred_all = self._safe_eval_formula_array(current_formula, X)
                pred_holdout = self._safe_eval_formula_array(current_formula, X_holdout)
            except Exception:
                break

            current_holdout_mse_u = float(np.mean((pred_holdout - y_holdout)**2))
            try:
                current_holdout_mse = (
                    _weighted_mse(pred_holdout, y_holdout, holdout_w)
                    if holdout_w is not None else current_holdout_mse_u
                )
            except Exception:
                current_holdout_mse = current_holdout_mse_u
            current_holdout_r2 = local_r2(y_holdout, pred_holdout)
            if self.boosting_diagnostics_.get("initial_holdout_r2") is None:
                self.boosting_diagnostics_["initial_holdout_r2"] = float(current_holdout_r2)

            if current_holdout_r2 > 0.999:
                break

            orig_timeout = self.timeout
            stage_timeout = max(5, orig_timeout // (2 ** (stage + 1)))
            self.timeout = stage_timeout
            self.boosting_attempted_ = True

            try:
                h_k = self._stage_residual_symbolic_fit(X, y, current_formula, _allow_recursion=True)
            finally:
                self.timeout = orig_timeout

            if not h_k or h_k == "0":
                if isinstance(getattr(self, "blackbox_diagnostics_", None), dict):
                    guard = getattr(self, "_residual_stage_guard_", {}) or {}
                    if guard.get("residual_rejected_as_noise") or guard.get("reason") == "no_holdout_improvement":
                        self.blackbox_diagnostics_["residual_rejected_as_noise"] = True
                break

            best_eta = None
            best_holdout_mse = current_holdout_mse
            best_combined_formula = None

            try:
                h_pred_holdout = self._safe_eval_formula_array(h_k, X_holdout)
            except Exception:
                break

            for eta in learning_rates:
                combined_formula = f"({current_formula}) + (({eta:.6g}) * ({h_k}))"
                try:
                    combined_pred = pred_holdout + eta * h_pred_holdout
                    mse_w = (
                        _weighted_mse(combined_pred, y_holdout, holdout_w)
                        if holdout_w is not None
                        else float(np.mean((combined_pred - y_holdout) ** 2))
                    )
                    mse_u = float(np.mean((combined_pred - y_holdout) ** 2))
                    # Weighted must improve; unweighted must not regress beyond slack.
                    if mse_w >= best_holdout_mse:
                        continue
                    u_allowed = current_holdout_mse_u * (1.0 + boost_rel_slack) + boost_abs_slack
                    if mse_u > u_allowed:
                        continue
                    best_holdout_mse = mse_w
                    best_eta = eta
                    best_combined_formula = combined_formula
                except Exception:
                    continue

            if best_eta is None:
                if isinstance(getattr(self, "blackbox_diagnostics_", None), dict):
                    self.blackbox_diagnostics_["residual_rejected_as_noise"] = True
                self.boosting_diagnostics_["last_reject_reason"] = "holdout_slack"
                break

            try:
                best_pred_holdout = pred_holdout + best_eta * h_pred_holdout
                best_holdout_r2 = local_r2(y_holdout, best_pred_holdout)
            except Exception:
                break

            r2_improvement = best_holdout_r2 - current_holdout_r2
            if r2_improvement < 0.005:
                if isinstance(getattr(self, "blackbox_diagnostics_", None), dict):
                    self.blackbox_diagnostics_["residual_rejected_as_noise"] = True
                self.boosting_diagnostics_["last_reject_reason"] = "insufficient_r2_gain"
                break

            refined_list = self._refine_candidate_formulas(
                [{"formula": best_combined_formula, "source": f"residual_boosting_stage_{stage}"}],
                X,
                y,
                max_candidates=1,
            )
            if refined_list:
                current_formula = refined_list[0]["formula"]
            else:
                current_formula = best_combined_formula

            self.boosting_stages_.append({
                "stage": stage,
                "h_k": h_k,
                "eta": best_eta,
                "holdout_r2_before": float(current_holdout_r2),
                "holdout_r2_after": float(best_holdout_r2),
                "holdout_r2_improvement": float(r2_improvement),
                "combined_formula": current_formula
            })
            self.boosting_improved_ = True
            self.boosting_diagnostics_["accepted_stages"] = len(self.boosting_stages_)
            self.boosting_diagnostics_["final_holdout_r2"] = float(best_holdout_r2)

            if best_holdout_r2 > 0.999:
                break

        if self.boosting_diagnostics_.get("final_holdout_r2") is None:
            try:
                final_pred_holdout = self._safe_eval_formula_array(current_formula, X_holdout)
                self.boosting_diagnostics_["final_holdout_r2"] = float(local_r2(y_holdout, final_pred_holdout))
            except Exception:
                self.boosting_diagnostics_["final_holdout_r2"] = self.boosting_diagnostics_.get("initial_holdout_r2")
        return current_formula

    def _detect_frequencies(self, X, y):
        """Detect dominant frequencies via FFT, with optional phase info."""
        try:
            x_t = torch.tensor(X[:, 0], dtype=torch.float32).reshape(-1, 1)
            y_t = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

            # Get rich phase info for the fast-path pipeline
            phase_info = detect_dominant_frequency(
                x_t, y_t, n_frequencies=3, return_phase_info=True,
            )
            self._fft_phase_info = phase_info  # stash for later use

            omegas = phase_info.get('omegas', [1.0])
            if omegas and omegas[0] == 1.0:
                return []
            return omegas or []
        except Exception:
            self._fft_phase_info = None
            return []

    def fit(self, X, y, sample_weight=None):
        """
        Fit the symbolic regression model using the full Glassbox pipeline:
        1. Fast-path (classifier-guided basis regression)
        2. C++ evolution (if fast-path misses or is approximate)
        3. Formula simplification (float snapping + SymPy)

        Parameters
        ----------
        sample_weight : array-like of shape (n_samples,), optional
            Per-point weights (PhySO ``y_weights`` analogue). Non-negative and
            finite; normalised to mean 1 internally. ``None`` keeps uniform
            weights and default behaviour unchanged. Used by the Python-side
            scoring layers (formula MSE, CV skip guard, final selection) so
            noisy or low-confidence points can be downweighted. Native C++
            scoring becomes weight-aware starting in Phase 2.
        """
        import time as _time

        X, y = check_X_y(X, y, accept_sparse=False)
        self.sample_weight_ = _validate_sample_weight(sample_weight, X.shape[0])
        self.sample_weight_provided_ = self.sample_weight_ is not None
        # Preserve user-requested loss mode so Phase 3/4 auto paths only switch
        # when the caller left the default ``mse``.
        # Undo sticky auto-switch from a previous fit (do not mutate public
        # loss_mode permanently — sklearn reuse / get_params must stay clean).
        if getattr(self, "_loss_mode_auto_switched_", False):
            prev = getattr(self, "_user_loss_mode_", None)
            if prev is not None:
                self.loss_mode = prev
            self._loss_mode_auto_switched_ = False
        self._user_loss_mode_ = str(getattr(self, "loss_mode", "mse") or "mse")
        self._loss_mode_auto_switched_ = False
        self.has_composed_seeds_ = False
        self.composition_candidates_accepted_ = False
        self.composition_candidate_count_ = 0
        self.composition_seeded_evolution_ = False
        self.composition_won_final_selection_ = False
        self.composition_improved_mse_ = False
        self.phase_timings_ = {}
        self.formula_eval_count_ = 0
        self.formula_eval_cache_hits_ = 0
        self._formula_eval_cache_ = {}
        self.fast_path_exact_skip_ = False
        self.fast_path_exact_match_diagnostics_ = {}
        self.specialist_track_ = "incumbent path"
        self.specialist_vault_ = SpecialistVault(max_entries=int(getattr(self, "specialist_vault_size", 8) or 0))
        # Clear cross-fit sticky state so prior problem winners cannot leak (S1-3).
        for _attr in (
            "formula_",
            "best_mse_",
            "evolution_candidate_formula_",
            "evolution_candidate_mse_",
            "pareto_front_",
            "nodes_",
            "output_weights_",
            "output_bias_",
            "blackbox_state_",
            "blackbox_diagnostics_",
        ):
            if hasattr(self, _attr):
                delattr(self, _attr)
        self.n_features_in_ = X.shape[1]
        self.original_n_features_in_ = X.shape[1]
        self._activate_physics_units(self.n_features_in_)
        fit_start = _time.time()

        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)

        blackbox_enabled = (
            bool(self.blackbox_feature_selection)
            and (
                self.blackbox_mode is True
                or (
                    self.blackbox_mode == "auto"
                    and X.shape[1] > 1
                )
            )
        )
        X_original = X
        y_original = y
        X_search, y_search, blackbox_state = prepare_blackbox_search(
            X,
            y,
            enabled=blackbox_enabled,
            max_features=int(self.blackbox_max_features),
            standardize=bool(self.blackbox_standardize),
            min_features_to_select=int(self.blackbox_min_features_to_select),
            interaction_search=bool(self.blackbox_interaction_search),
            sample_weight=self.sample_weight_ if self.sample_weight_provided_ else None,
        )
        # Phase C / Phase 3: gated soft MAD weights when user did not supply
        # sample_weight. Originally multi-feature blackbox only; Phase 3 extends
        # auto mode to 1D SR so native evolution receives y_weights under outliers
        # (noise protocol baseline is almost entirely single-feature).
        # May re-run ranking with weights so selection matches evolution objective.
        self._auto_weight_fallback_candidates_ = []
        self._auto_weight_final_guard_ = None
        self._blackbox_noise_robust_applied_ = {
            "active": False,
            "mode": getattr(self, "blackbox_noise_robust", "auto"),
            "reason": "not_applied",
        }
        robust_mode = getattr(self, "blackbox_noise_robust", "auto")
        multi_feature_blackbox = (
            blackbox_enabled
            and getattr(blackbox_state, "enabled", False)
            and int(X.shape[1]) > 1
        )
        single_feature_sr = int(X.shape[1]) == 1
        want_robust = (
            not self.sample_weight_provided_
            and (
                robust_mode is True
                or (
                    robust_mode == "auto"
                    and (multi_feature_blackbox or single_feature_sr)
                )
            )
        )
        if want_robust:
            # Residual-based soft weights (avoids false positives on clean polynomials).
            soft_w, out_frac = _auto_residual_soft_weights(X, y)
            selection_uncertain = bool(getattr(blackbox_state, "feature_selection_uncertain", False))
            # Activation: need a soft-weight signal plus evidence of heavy tails.
            # 1D SR uses a slightly lower out_frac floor (sparse 3% outliers often
            # land near ~1–2% on residual MAD estimates). Clean polynomials that
            # only trigger weak soft weights (out_frac ~0) stay unweighted.
            out_frac_floor = 0.01 if single_feature_sr else 0.02
            low_weight_mass = 0.0
            if soft_w is not None:
                sw = np.asarray(soft_w, dtype=np.float64).reshape(-1)
                mean_w = float(np.mean(sw)) if sw.size else 1.0
                if mean_w > 1e-12:
                    low_weight_mass = float(np.mean(sw < 0.85 * mean_w))
            activate = soft_w is not None and (
                robust_mode is True
                or selection_uncertain
                or out_frac >= out_frac_floor
                or low_weight_mass >= 0.02
                or (
                    multi_feature_blackbox
                    and str(getattr(blackbox_state, "reason", "")).startswith("retained_all_features")
                )
            )
            if activate and soft_w is not None:
                self.sample_weight_ = _validate_sample_weight(soft_w, X.shape[0])
                self.sample_weight_provided_ = self.sample_weight_ is not None
                # Prefer huber search loss when we auto-soft-weight blackbox noise
                # but only if user left default mse.
                loss_switched = False
                if str(getattr(self, "loss_mode", "mse") or "mse") == "mse":
                    self.loss_mode = "huber"
                    self._loss_mode_auto_switched_ = True
                    loss_switched = True
                X_search, y_search, blackbox_state = prepare_blackbox_search(
                    X,
                    y,
                    enabled=blackbox_enabled,
                    max_features=int(self.blackbox_max_features),
                    standardize=bool(self.blackbox_standardize),
                    min_features_to_select=int(self.blackbox_min_features_to_select),
                    interaction_search=bool(self.blackbox_interaction_search),
                    sample_weight=self.sample_weight_,
                )
                self._blackbox_noise_robust_applied_ = {
                    "active": True,
                    "mode": robust_mode,
                    "reason": "soft_mad_weights",
                    "path": (
                        "1d_sr"
                        if single_feature_sr
                        else ("multi_feature_blackbox" if multi_feature_blackbox else "forced")
                    ),
                    "outlier_fraction_target": float(out_frac),
                    "selection_uncertain": selection_uncertain,
                    "loss_mode_switched_to_huber": loss_switched,
                    "ess_ratio": (
                        float(_effective_sample_size(self.sample_weight_) / max(float(X.shape[0]), 1.0))
                        if self.sample_weight_ is not None
                        else None
                    ),
                    "weights_to_evolution": True,
                }
            else:
                self._blackbox_noise_robust_applied_ = {
                    "active": False,
                    "mode": robust_mode,
                    "reason": "no_heavy_tail_signal",
                    "outlier_fraction_target": float(out_frac),
                }
                # Phase 4: diffuse noise (pink / ~10% Gaussian) often has NO heavy-tail
                # mass for soft-MAD weights. Enable Huber search loss without weights
                # when residual scale vs y is elevated after a cheap structure fit.
                if (
                    str(getattr(self, "_user_loss_mode_", "mse") or "mse") == "mse"
                    and str(getattr(self, "loss_mode", "mse") or "mse") == "mse"
                ):
                    noise_ratio, res_out_frac = _estimate_diffuse_noise_ratio(X, y)
                    # ~10% RMS residual noise → ratio ~0.1; clean structured → ~0.
                    # Floor ~0.025 catches ~5% pink / 10% Gaussian residual ratios; clean/1% stay off.
                    if robust_mode is True or (
                        robust_mode == "auto" and noise_ratio >= 0.02
                    ):
                        self.loss_mode = "huber"
                        self._loss_mode_auto_switched_ = True
                        self._blackbox_noise_robust_applied_ = {
                            "active": True,
                            "mode": robust_mode,
                            "reason": "diffuse_noise_huber",
                            "path": (
                                "1d_sr"
                                if single_feature_sr
                                else (
                                    "multi_feature_blackbox"
                                    if multi_feature_blackbox
                                    else "forced"
                                )
                            ),
                            "outlier_fraction_target": float(res_out_frac),
                            "diffuse_noise_ratio": float(noise_ratio),
                            "loss_mode_switched_to_huber": True,
                            "weights_to_evolution": False,
                            "sample_weight_mode": "none",
                        }
        elif (
            not self.sample_weight_provided_
            and robust_mode is not False
            and str(getattr(self, "_user_loss_mode_", getattr(self, "loss_mode", "mse")) or "mse")
            == "mse"
            and str(getattr(self, "loss_mode", "mse") or "mse") == "mse"
        ):
            # want_robust was false (e.g. unusual shape) but still allow forced True.
            if robust_mode is True:
                noise_ratio, res_out_frac = _estimate_diffuse_noise_ratio(X, y)
                if noise_ratio >= 0.02 or robust_mode is True:
                    self.loss_mode = "huber"
                    self._loss_mode_auto_switched_ = True
                    self._blackbox_noise_robust_applied_ = {
                        "active": True,
                        "mode": robust_mode,
                        "reason": "diffuse_noise_huber",
                        "path": "forced",
                        "outlier_fraction_target": float(res_out_frac),
                        "diffuse_noise_ratio": float(noise_ratio),
                        "loss_mode_switched_to_huber": True,
                        "weights_to_evolution": False,
                        "sample_weight_mode": "none",
                    }
        feature_selection_fallback = None
        if (
            blackbox_enabled
            and getattr(blackbox_state, "enabled", False)
            and X.shape[1] > X_search.shape[1]
            and X.shape[1] <= 14
        ):
            selected_cols = list(getattr(blackbox_state, "selected_features", []) or [])
            selected_tail_r2 = self._ridge_tail_validation_r2(X_original, y_original, selected_cols)
            all_tail_r2 = self._ridge_tail_validation_r2(X_original, y_original, None)
            feature_selection_fallback = {
                "selected_tail_r2": selected_tail_r2,
                "all_tail_r2": all_tail_r2,
                "selected_features": selected_cols,
                "n_original_features": int(X.shape[1]),
            }
            self._blackbox_original_linear_fallback = None
            if (
                selected_tail_r2 is not None
                and all_tail_r2 is not None
                and all_tail_r2 > 0.05
                and (
                    all_tail_r2 > selected_tail_r2 + 0.06
                    or (
                        selected_tail_r2 < 0.65
                        and all_tail_r2 >= selected_tail_r2 - 0.02
                    )
                )
            ):
                self._blackbox_original_linear_fallback = self._fit_ridge_formula(
                    X_original,
                    y_original,
                    None,
                )
                feature_selection_fallback["activated"] = True
                feature_selection_fallback["reason"] = "all_features_tail_ridge_candidate"
                if self._blackbox_original_linear_fallback is not None:
                    feature_selection_fallback["fallback_validation_r2"] = self._blackbox_original_linear_fallback.get("validation_r2")
                    feature_selection_fallback["fallback_n_terms"] = self._blackbox_original_linear_fallback.get("n_terms")
            else:
                feature_selection_fallback["activated"] = False
                self._blackbox_original_linear_fallback = None
        else:
            self._blackbox_original_linear_fallback = None
        self._blackbox_feature_fallback_activated = bool(
            isinstance(feature_selection_fallback, dict)
            and feature_selection_fallback.get("activated")
        )
        self.blackbox_state_ = blackbox_state
        self.blackbox_diagnostics_ = state_to_dict(blackbox_state)
        if isinstance(self.blackbox_diagnostics_, dict):
            if self.sample_weight_provided_ and self.sample_weight_ is not None:
                self.blackbox_diagnostics_["sample_weight"] = {
                    "provided": True,
                    "effective_sample_size": _effective_sample_size(self.sample_weight_),
                    "min_weight": float(np.min(self.sample_weight_)),
                    "max_weight": float(np.max(self.sample_weight_)),
                    "mean_weight": float(np.mean(self.sample_weight_)),
                    "source": (
                        "auto_soft_mad"
                        if (getattr(self, "_blackbox_noise_robust_applied_", {}) or {}).get("active")
                        else "user"
                    ),
                }
            else:
                self.blackbox_diagnostics_["sample_weight"] = {"provided": False}
            self.blackbox_diagnostics_["loss_mode"] = {
                "mode": str(getattr(self, "loss_mode", "mse") or "mse"),
                "huber_delta": getattr(self, "huber_delta", None),
                "trim_fraction": float(getattr(self, "trim_fraction", 0.1) or 0.1),
                "applied_to": "search_scoring_and_evolution",
                "display_mse_unweighted": True,
            }
            self.blackbox_diagnostics_["blackbox_noise_robust"] = dict(
                getattr(self, "_blackbox_noise_robust_applied_", {}) or {}
            )
        if isinstance(self.blackbox_diagnostics_, dict) and feature_selection_fallback is not None:
            self.blackbox_diagnostics_["feature_selection_fallback"] = feature_selection_fallback
        self.blackbox_search_plan_ = {}
        blackbox_candidate_accepted = False
        blackbox_evolution_ran = False
        blackbox_evolution_improved = False

        # Structure templates: seed-only (diag + candidate pool). Never early-exit.
        # Winning Exact on known families must go through search/Pareto competition.
        structure_probe = None
        self._structure_probe_seed_ = None
        if (
            blackbox_state.enabled
            and int(X_original.shape[1]) > 1
            and len(getattr(blackbox_state, "selected_features", []) or []) > 1
        ):
            structure_probe = self._probe_multivariate_structure_original_space(
                X_original,
                y_original,
                list(getattr(blackbox_state, "selected_features", []) or []),
            )
            if structure_probe is not None:
                self._structure_probe_seed_ = dict(structure_probe)
            if isinstance(self.blackbox_diagnostics_, dict) and structure_probe is not None:
                self.blackbox_diagnostics_["structure_probe_original"] = {
                    "template_match": structure_probe.get("template_match"),
                    "mse": structure_probe.get("mse"),
                    "r2": structure_probe.get("r2"),
                    "complexity": structure_probe.get("complexity"),
                    "exact_match": structure_probe.get("exact_match"),
                    "robust_match": structure_probe.get("robust_match"),
                    "inlier_fraction": structure_probe.get("inlier_fraction"),
                    "formula": str(structure_probe.get("formula") or "")[:200],
                    "role": "seed_candidate_only",
                    "auto_win": False,
                }

        if blackbox_state.enabled:
            X = X_search
            y = y_search
            self.n_features_in_ = X.shape[1]
            # Phase 5: remap unit vectors to selected feature subset.
            if getattr(self, "units_active_", False) and getattr(self, "input_units_", None):
                selected = list(getattr(blackbox_state, "selected_features", []) or [])
                full_units = list(self.input_units_)
                if selected and all(0 <= int(i) < len(full_units) for i in selected):
                    self.input_units_ = [list(full_units[int(i)]) for i in selected]
                else:
                    # Cannot safely remap — disable units rather than mix lengths.
                    self.input_units_ = None
                    self.output_units_ = None
                    self.units_active_ = False
                    self.physics_constrained_ = False
            if self.universal_proposer_log_routing:
                print(
                    "  [Blackbox] selected features "
                    f"{blackbox_state.selected_features} / {self.original_n_features_in_}"
                )

        if isinstance(self.blackbox_diagnostics_, dict):
            self.blackbox_diagnostics_["physics_units"] = {
                "active": bool(getattr(self, "units_active_", False)),
                "physics_constrained": bool(getattr(self, "physics_constrained_", False)),
                "unit_mode": _validate_unit_mode(getattr(self, "unit_mode", "off")),
                "dim_penalty_weight": float(getattr(self, "dim_penalty_weight", 0.1) or 0.0),
                "n_features_units": (
                    len(self.input_units_) if getattr(self, "input_units_", None) else 0
                ),
                "n_dims": (
                    len(self.input_units_[0])
                    if getattr(self, "input_units_", None)
                    else 0
                ),
            }

        detected_omegas = self._detect_frequencies(X, y)

        best_formula = None
        best_mse = float('inf')
        operator_hints = {}
        demoted_fast_path_candidate = None
        y_var = float(np.var(y))  # For R² calculation

        def _elapsed():
            return _time.time() - fit_start

        def _r2_from_mse(mse):
            """Compute R² from MSE and target variance."""
            if y_var < 1e-15:
                return 1.0 if mse < 1e-15 else 0.0
            return 1.0 - mse / y_var

        def _finish_with_formula(formula, mse, *, skip_reason=None):
            final_formula = formula
            final_mse = mse
            if final_formula:
                final_formula = self._cleanup_formula_with_fidelity_guard(
                    final_formula,
                    X,
                    y,
                    stage="fast_path_exact",
                )
                if getattr(self, "blackbox_state_", None) is not None and self.blackbox_state_.enabled:
                    final_formula = formula_from_search_to_original_space(
                        final_formula,
                        self.blackbox_state_,
                    )
                try:
                    eval_X = X_original if (
                        getattr(self, "blackbox_state_", None) is not None
                        and self.blackbox_state_.enabled
                    ) else X
                    eval_y = y_original if eval_X is X_original else y
                    self._register_auto_weight_fallback_candidate(
                        final_formula, eval_X, eval_y, source="fast_path"
                    )
                    final_formula = self._apply_auto_weight_final_guard(
                        final_formula, eval_X, eval_y, stage="fast_path_exact"
                    )
                    pred = self._safe_eval_formula_array(final_formula, eval_X)
                    final_mse = float(np.mean((pred - np.asarray(eval_y, dtype=np.float64).reshape(-1)) ** 2))
                except Exception:
                    pass
            self.formula_ = final_formula or "0"
            self.best_mse_ = final_mse
            if skip_reason and isinstance(self.blackbox_diagnostics_, dict):
                self.blackbox_diagnostics_["specialist_skipped_reason"] = skip_reason
            self._restore_user_loss_mode_if_auto_switched()
            self._add_phase_time("total_fit", _time.time() - fit_start)
            return self

        # ── Stage 1: Classifier Fast Path ──
        if self.use_fast_path and _elapsed() < self.timeout:
            try:
                from classifier_fast_path import run_fast_path  # type: ignore

                x_t = torch.tensor(X, dtype=torch.float32)
                y_t = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
                classifier_path = self._resolve_classifier_path()

                fp_result = run_fast_path(
                    x_t, y_t,
                    classifier_path=classifier_path,
                    detected_omegas=detected_omegas,
                    op_constraints=None,
                    auto_expand=True,
                    device=self.device,
                    exact_match_threads=1,
                    exact_match_enabled=True,
                    exact_match_max_basis=200,
                    max_power=self.max_power,
                    exact_match_backend=self.exact_match_backend,
                    exact_match_min_gpu_work=self.exact_match_min_gpu_work,
                    exact_match_max_combos=self.exact_match_max_combos,
                    simplify_formula_output=False,
                )

                if fp_result and fp_result.get('formula'):
                    fp_uncertainty = fp_result.get("uncertainty") or {}
                    fp_details = fp_result.get("details") or {}
                    self.fast_path_exact_match_diagnostics_ = dict(fp_details.get("exact_match_diagnostics") or {})
                    if isinstance(self.blackbox_diagnostics_, dict):
                        self.blackbox_diagnostics_["fast_path_exact_match"] = self.fast_path_exact_match_diagnostics_
                    already_compact = bool(fp_details.get("compact_multivariate_basis", False))
                    if (
                        blackbox_state.enabled
                        and not already_compact
                        and not self._should_use_universal_fast_path(blackbox_state, fp_uncertainty)
                    ):
                        if isinstance(self.blackbox_diagnostics_, dict):
                            self.blackbox_diagnostics_["fast_path_auto_expand"] = False
                        fp_result = run_fast_path(
                            x_t, y_t,
                            classifier_path=classifier_path,
                            detected_omegas=detected_omegas,
                            op_constraints=None,
                            auto_expand=False,
                            device=self.device,
                            exact_match_threads=1,
                            exact_match_enabled=True,
                            exact_match_max_basis=200,
                            max_power=self.max_power,
                            exact_match_backend=self.exact_match_backend,
                            exact_match_min_gpu_work=self.exact_match_min_gpu_work,
                            exact_match_max_combos=self.exact_match_max_combos,
                            simplify_formula_output=False,
                        )
                        if fp_result and fp_result.get('formula'):
                            fp_details = fp_result.get("details") or {}
                            self.fast_path_exact_match_diagnostics_ = dict(fp_details.get("exact_match_diagnostics") or {})
                            if isinstance(self.blackbox_diagnostics_, dict):
                                self.blackbox_diagnostics_["fast_path_exact_match"] = self.fast_path_exact_match_diagnostics_
                    elif isinstance(self.blackbox_diagnostics_, dict) and blackbox_state.enabled:
                        self.blackbox_diagnostics_["fast_path_auto_expand"] = not already_compact
                    best_formula = fp_result['formula']
                    best_mse = fp_result.get('mse', float('inf'))
                    operator_hints = fp_result.get('operator_hints') or {}
                    # Stash for uncertainty-coupled budget routing and candidate seeding
                    self._fp_result = fp_result
                    if blackbox_state.enabled:
                        gate = self._validate_blackbox_fast_path_candidate(best_formula, best_mse, X, y)
                        if isinstance(self.blackbox_diagnostics_, dict):
                            self.blackbox_diagnostics_["fast_path_validation_gate"] = gate
                        if not gate.get("accepted", True):
                            demoted_fast_path_candidate = {
                                "formula": gate.get("candidate_formula") or best_formula,
                                "mse": gate.get("validation_mse", best_mse),
                                "validation_mse": gate.get("validation_mse"),
                                "validation_r2": gate.get("validation_r2"),
                                "complexity": gate.get("complexity", self._formula_complexity(best_formula)),
                                "risk_score": gate.get("risk_score", 0.0),
                                "generalization_gap": gate.get("generalization_gap", 0.0),
                                "from_fast_path": True,
                                "demoted_fast_path": True,
                                "source": "demoted_fast_path",
                            }
                            best_formula = None
                            best_mse = float("inf")
            except Exception as e:
                self._fp_result = None
                print(f"  [Fast-path skipped: {e}]")

        self.universal_proposer_fpip_v2_ = None
        _, proposer_forces_evolution = self._run_universal_proposer_dual_path(
            X,
            y,
            getattr(self, '_fp_result', None),
            getattr(self, "blackbox_state_", None),
        )

        # ── Stage 2: C++ Evolution ──
        # Only run evolution if:
        #   - No formula found yet, OR
        #   - R² is below the skip threshold (default 0.999)
        #   - Cross-validation guard says fast-path fit is unstable
        #   - We haven't exceeded the timeout
        current_r2 = _r2_from_mse(best_mse) if best_formula else -1.0
        term_count = (best_formula.count('+') + best_formula.count('-')) if best_formula else 0
        fast_path_cv_ok = True

        if (
            best_formula is not None
            and best_mse is not None
            and math.isfinite(best_mse)
            and current_r2 >= self.evolution_skip_r2
        ):
            fast_path_cv_ok = self._passes_cross_validation_skip_guard(best_formula, X, y, self.sample_weight_)
        else:
            self.fast_path_cv_guard_ = {
                'enabled': bool(self.cv_skip_guard_enabled),
                'fold_r2': [],
                'min_fold_r2': None,
                'std_fold_r2': None,
                'passed': True,
                'reason': 'not_applicable',
            }

        # Optional benchmark policy: if fast-path is very bloated, keep it as-is
        # and avoid launching evolution search for this sample.
        if (
            self.skip_evolution_if_bloated
            and best_formula is not None
            and term_count > int(self.bloat_term_threshold)
            and not (blackbox_state is not None and blackbox_state.enabled)
        ):
            need_evolution = False
        else:
            need_evolution = (
                best_formula is None or
                best_mse is None or
                not math.isfinite(best_mse) or
                current_r2 < self.evolution_skip_r2 or
                not fast_path_cv_ok or
                term_count > 10 # Higher threshold for Stage 1 bloat
            )
            # Multi-var blackbox: force evolution when fast-path is bloated even if R2 is high.
            if (
                not need_evolution
                and blackbox_state is not None
                and blackbox_state.enabled
                and best_formula is not None
            ):
                fp_comp = self._formula_complexity(best_formula)
                n_sel = len(getattr(blackbox_state, "selected_features", []) or [])
                if n_sel > 1 and (fp_comp > 24 or term_count > 6):
                    need_evolution = True

        if (
            best_formula is not None
            and best_mse is not None
            and math.isfinite(best_mse)
            and best_mse <= max(float(self.early_stop_mse), 1e-10)
            and current_r2 >= min(float(self.evolution_skip_r2), 0.999999)
            and fast_path_cv_ok
        ):
            self.fast_path_exact_skip_ = True
            self.specialist_track_ = "incumbent path"
            return _finish_with_formula(best_formula, best_mse, skip_reason="fast_path_exact")

        # Uncertainty-coupled budget routing: pass FPIP uncertainty metrics
        _fp_uncertainty = None
        _fp = getattr(self, '_fp_result', None)
        if isinstance(_fp, dict):
            _fp_uncertainty = _fp.get('uncertainty')

        # Override/blend with Universal Proposer's uncertainty if available
        if self.universal_proposer_fpip_v2_ and self.universal_proposer_fpip_v2_.get("valid"):
            proposer_unc = self.universal_proposer_fpip_v2_.get("sequence_uncertainty", {})
            if "entropy" in proposer_unc and proposer_unc["entropy"] is not None:
                if _fp_uncertainty is None:
                    _fp_uncertainty = {}
                # Take the max uncertainty between fast-path and proposer
                _fp_uncertainty["prediction_entropy"] = max(
                    _fp_uncertainty.get("prediction_entropy", 0.0), 
                    proposer_unc["entropy"]
                )
                _fp_uncertainty["prediction_margin"] = min(
                    _fp_uncertainty.get("prediction_margin", 1.0), 
                    proposer_unc.get("margin", 1.0)
                )

        proposer_payload = (
            self.universal_proposer_fpip_v2_
            if isinstance(self.universal_proposer_fpip_v2_, dict)
            else {}
        )
        candidate_screening = None
        blackbox_state = getattr(self, "blackbox_state_", None)
        if blackbox_state is not None and blackbox_state.enabled:
            preview_candidates = self._build_blackbox_candidate_formulas(
                best_formula,
                best_mse,
                proposer_payload,
                blackbox_state,
                X,
                y,
                max_candidates=10,
            )
            if demoted_fast_path_candidate is not None:
                preview_candidates = self._prune_blackbox_candidate_formulas(
                    [demoted_fast_path_candidate] + list(preview_candidates or []),
                    max_candidates=10,
                )
            if preview_candidates:
                families = {
                    self._formula_family_signature(c.get("formula", ""))
                    for c in preview_candidates
                    if str(c.get("formula", "")).strip()
                }
                candidate_screening = {
                    "candidate_count": len(preview_candidates),
                    "family_count": len(families),
                    "best_validation_r2": max(
                        _finite_float(c.get("validation_r2"), -1.0)
                        for c in preview_candidates
                    ),
                }
        proposer_plan = proposer_payload.get("search_plan", {})
        if not isinstance(proposer_plan, dict):
            proposer_plan = {}
        # Phase 7: residual/weight diagnostics before routing so plan thresholds adapt.
        noise_diag = self._compute_runtime_noise_diagnostics(
            X, y, formula=best_formula
        )
        # Calibrate prediction_uncertain flag using noise-band entropy floor.
        if isinstance(_fp_uncertainty, dict):
            band = str(noise_diag.get("noise_band") or "clean")
            thr = float(_NOISE_BAND_THRESHOLDS.get(band, _NOISE_BAND_THRESHOLDS["clean"])[
                "prediction_uncertain_entropy"
            ])
            ent = _fp_uncertainty.get("prediction_entropy")
            try:
                ent_f = float(ent) if ent is not None else None
            except Exception:
                ent_f = None
            if ent_f is not None and np.isfinite(ent_f) and ent_f >= thr:
                _fp_uncertainty = dict(_fp_uncertainty)
                _fp_uncertainty["prediction_uncertain"] = True
                _fp_uncertainty["prediction_uncertain_reason"] = f"noise_band_{band}_entropy"
        blackbox_search_plan = self._derive_blackbox_search_plan(
            getattr(self, "blackbox_state_", None),
            fast_path_uncertainty=_fp_uncertainty,
            proposer_uncertainty=proposer_payload.get("sequence_uncertainty", {}),
            proposer_plan=proposer_plan,
            candidate_screening=candidate_screening,
            noise_diagnostics=noise_diag,
        )
        self.blackbox_search_plan_ = blackbox_search_plan
        if isinstance(self.blackbox_diagnostics_, dict):
            self.blackbox_diagnostics_["search_plan"] = blackbox_search_plan
            self.blackbox_diagnostics_["runtime_noise"] = dict(noise_diag)

        candidate_formulas = None
        if blackbox_state is not None and blackbox_state.enabled:
            candidate_formulas = self._build_blackbox_candidate_formulas(
                best_formula,
                best_mse,
                proposer_payload,
                blackbox_state,
                X,
                y,
                max_candidates=max(
                    8,
                    int(blackbox_search_plan.get("seed_budget", 8)),
                ),
            )
            if demoted_fast_path_candidate is not None:
                candidate_formulas = self._prune_blackbox_candidate_formulas(
                    [demoted_fast_path_candidate] + list(candidate_formulas or []),
                    max_candidates=max(
                        8,
                        int(blackbox_search_plan.get("seed_budget", 8)),
                    ),
                )
            interaction_hints = self._derive_blackbox_operator_hints(
                blackbox_state,
                candidate_formulas,
            )
            operator_hints = dict(operator_hints or {})
            operator_hints["operators"] = set(operator_hints.get("operators", set()))
            operator_hints["operators"].update(interaction_hints.get("operators", set()))
            operator_hints["powers"] = sorted(set(
                list(operator_hints.get("powers", [])) + list(interaction_hints.get("powers", []))
            ))
            operator_hints["active_terms"] = list(dict.fromkeys(
                list(operator_hints.get("active_terms", [])) + list(interaction_hints.get("active_terms", []))
            ))[:16]
            operator_hints["has_rational"] = bool(
                operator_hints.get("has_rational", False) or interaction_hints.get("has_rational", False)
            )
            operator_hints["has_exp_decay"] = bool(
                operator_hints.get("has_exp_decay", False) or interaction_hints.get("has_exp_decay", False)
            )
            operator_hints = self._constrain_blackbox_operator_hints(operator_hints, blackbox_state)
            binary_op_priors, multi_binary_op_priors = self._derive_blackbox_binary_priors(
                blackbox_state,
                operator_hints,
            )
            allowed_unary_ops, multi_allowed_unary_ops, allowed_binary_ops, multi_allowed_binary_ops = (
                self._derive_blackbox_unary_policy(blackbox_state, operator_hints)
            )
            if blackbox_search_plan:
                blackbox_search_plan["allowed_unary_ops"] = list(allowed_unary_ops)
                blackbox_search_plan["multi_allowed_unary_ops"] = [list(v) for v in multi_allowed_unary_ops]
                blackbox_search_plan["binary_op_priors"] = list(binary_op_priors)
                blackbox_search_plan["multi_binary_op_priors"] = [list(v) for v in multi_binary_op_priors]
                blackbox_search_plan["allowed_binary_ops"] = list(allowed_binary_ops)
                blackbox_search_plan["multi_allowed_binary_ops"] = [list(v) for v in multi_allowed_binary_ops]
            if isinstance(self.blackbox_diagnostics_, dict):
                self.blackbox_diagnostics_["candidate_screening"] = {
                    "candidate_count": len(candidate_formulas or []),
                    "top_candidates": [
                        {
                            "formula": str(c.get("formula", ""))[:160],
                            "validation_r2": c.get("validation_r2"),
                            "validation_mse": c.get("validation_mse"),
                            "complexity": c.get("complexity"),
                        }
                        for c in (candidate_formulas or [])[:6]
                    ],
                    "interaction_operator_hints": sorted(operator_hints.get("operators", set())),
                }
            candidate_formulas = self._run_specialist_candidate_screening(
                candidate_formulas,
                X,
                y,
                blackbox_search_plan,
                diagnostics_key="candidate_screening",
            )
            # Promote best structure seed over bloated fast-path when it is
            # clearly better on validation (honest competition, not auto-win).
            self._promoted_structure_seed_ = None
            try:
                fp_comp = self._formula_complexity(best_formula) if best_formula else 999
                best_seed = None
                for cand in candidate_formulas or []:
                    if not (cand or {}).get("from_structure_seed"):
                        continue
                    mse_c = float((cand or {}).get("mse", float("inf")))
                    if best_seed is None or mse_c < float(best_seed.get("mse", float("inf"))):
                        best_seed = cand
                if best_seed is not None and np.isfinite(float(best_seed.get("mse", float("inf")))):
                    seed_mse = float(best_seed["mse"])
                    seed_comp = int(best_seed.get("complexity") or self._formula_complexity(best_seed.get("formula")))
                    fp_mse = float(best_mse) if best_mse is not None and np.isfinite(best_mse) else float("inf")
                    better_fit = seed_mse <= fp_mse * 1.02 + 1e-15
                    much_simpler = seed_comp + 12 <= fp_comp
                    strong_fit = seed_mse <= max(1e-4, 0.05 * max(float(np.var(y)), 1e-12))
                    if better_fit and (much_simpler or strong_fit or seed_mse < fp_mse):
                        best_formula = str(best_seed["formula"])
                        best_mse = seed_mse
                        self._promoted_structure_seed_ = {
                            "formula": best_formula,
                            "mse": seed_mse,
                            "complexity": seed_comp,
                            "skeleton": str(best_seed.get("skeleton") or ""),
                            "from_structure_seed": True,
                        }
                        if isinstance(self.blackbox_diagnostics_, dict):
                            self.blackbox_diagnostics_["structure_seed_promoted"] = {
                                "formula": best_formula[:160],
                                "mse": seed_mse,
                                "complexity": seed_comp,
                                "fp_mse": fp_mse if np.isfinite(fp_mse) else None,
                                "fp_complexity": fp_comp,
                                "skeleton": str(best_seed.get("skeleton") or "")[:120],
                            }
            except Exception:
                pass

        if (
            not (blackbox_state is not None and blackbox_state.enabled)
            and getattr(self, "enable_specialist_screening_diagnostics", True)
            and self.n_features_in_ == 1
        ):
            univariate_candidate_formulas = self._build_univariate_specialist_candidate_formulas(
                best_formula,
                best_mse,
                proposer_payload,
                X,
                y,
                max_candidates=max(
                    8,
                    int(blackbox_search_plan.get("seed_budget", 8)),
                ),
            )
            had_composed_seeds = bool(self.has_composed_seeds_)
            screened_univariate_candidates = self._run_specialist_candidate_screening(
                univariate_candidate_formulas,
                X,
                y,
                blackbox_search_plan,
                diagnostics_key="candidate_screening",
            )
            if (
                (self.has_composed_seeds_ and not had_composed_seeds)
                or self._candidate_pool_has_actionable_fit(
                    screened_univariate_candidates,
                    best_mse,
                    blackbox_search_plan,
                )
            ):
                candidate_formulas = screened_univariate_candidates
            else:
                candidate_formulas = None

        if getattr(self, "blackbox_state_", None) is not None and self.blackbox_state_.enabled:
            basis_pool = self._build_blackbox_formula_pool(
                best_formula,
                proposer_payload,
                self.blackbox_state_,
                self.n_features_in_,
            )
            for cand in candidate_formulas or []:
                formula = str(cand.get("formula", "")).strip()
                if formula and formula not in basis_pool:
                    basis_pool.insert(0, formula)
            basis_pool = list(dict.fromkeys(basis_pool))[:32]
            basis_result = self._fit_blackbox_basis_model(
                X,
                y,
                basis_pool,
                max_terms=int(blackbox_search_plan.get("basis_max_terms", 4)),
            )
            if basis_result is not None:
                self.blackbox_basis_model_ = basis_result
                if isinstance(self.blackbox_diagnostics_, dict):
                    self.blackbox_diagnostics_["basis_model"] = {
                        "validation_r2": basis_result.get("validation_r2"),
                        "validation_mse": basis_result.get("validation_mse"),
                        "selected_terms": basis_result.get("selected_terms"),
                        "n_terms": basis_result.get("n_terms"),
                    }
                basis_mse = float(basis_result.get("mse", float("inf")))
                if basis_mse < best_mse or best_formula is None:
                    best_formula = basis_result.get("formula", best_formula)
                    best_mse = basis_mse
                    updated_candidates = [{
                        "formula": best_formula,
                        "mse": best_mse,
                        "validation_mse": basis_result.get("validation_mse", best_mse),
                        "validation_r2": basis_result.get("validation_r2", -1.0),
                        "complexity": basis_result.get("complexity", self._formula_complexity(best_formula)),
                        "from_basis_model": True,
                    }]
                    if candidate_formulas:
                        updated_candidates.extend(candidate_formulas)
                    candidate_formulas = self._prune_blackbox_candidate_formulas(
                        updated_candidates,
                        max_candidates=max(
                            8,
                            int(blackbox_search_plan.get("seed_budget", 8)),
                        ),
                    )
            else:
                self.blackbox_basis_model_ = None
            engineered_result = self._fit_blackbox_engineered_basis_model(
                X,
                y,
                max_terms=(
                    12 if getattr(self, "_blackbox_feature_fallback_activated", False)
                    else max(6, int(blackbox_search_plan.get("basis_max_terms", 4)) + 4)
                ),
            )
            self.blackbox_engineered_basis_model_ = engineered_result
            if engineered_result is not None:
                if isinstance(self.blackbox_diagnostics_, dict):
                    self.blackbox_diagnostics_["engineered_basis_model"] = {
                        "validation_r2": engineered_result.get("validation_r2"),
                        "validation_mse": engineered_result.get("validation_mse"),
                        "selected_terms": engineered_result.get("selected_terms"),
                        "n_terms": engineered_result.get("n_terms"),
                    }
                eng_mse = float(engineered_result.get("mse", float("inf")))
                eng_val_r2 = float(engineered_result.get("validation_r2", -1.0))
                basis_val_r2 = float((basis_result or {}).get("validation_r2", -1.0)) if isinstance(basis_result, dict) else -1.0
                if (
                    best_formula is None
                    or eng_mse < best_mse
                    or eng_val_r2 > max(basis_val_r2, -1.0) + 0.03
                ):
                    best_formula = engineered_result.get("formula", best_formula)
                    best_mse = eng_mse
                    updated_candidates = [dict(engineered_result)]
                    if candidate_formulas:
                        updated_candidates.extend(candidate_formulas)
                    candidate_formulas = self._prune_blackbox_candidate_formulas(
                        updated_candidates,
                        max_candidates=max(
                            8,
                            int(blackbox_search_plan.get("seed_budget", 8)),
                        ),
                    )
        else:
            self.blackbox_basis_model_ = None
            self.blackbox_engineered_basis_model_ = None

        effective_timeout = self._estimate_compute_budget(
            X,
            current_r2,
            term_count,
            uncertainty=_fp_uncertainty,
        ) * float(blackbox_search_plan.get("timeout_multiplier", 1.0))

        basis_result = getattr(self, "blackbox_basis_model_", None)
        if isinstance(basis_result, dict):
            basis_val_r2 = float(basis_result.get("validation_r2", -1.0))
            basis_terms = int(basis_result.get("n_terms", 99))
            if (
                basis_val_r2 >= float(blackbox_search_plan.get("candidate_acceptance_r2", 0.985))
                and basis_terms <= int(blackbox_search_plan.get("basis_max_terms", 4))
            ):
                need_evolution = False
                blackbox_candidate_accepted = True
            elif basis_val_r2 >= float(blackbox_search_plan.get("candidate_shrink_r2", 0.95)):
                effective_timeout = min(effective_timeout, max(20.0, 0.4 * effective_timeout))

        fp_details_for_budget = (
            (getattr(self, "_fp_result", {}) or {}).get("details", {})
            if isinstance(getattr(self, "_fp_result", None), dict)
            else {}
        )
        compact_fast_path = bool(fp_details_for_budget.get("compact_multivariate_basis", False))
        compact_terms = int(fp_details_for_budget.get("n_nonzero", 99) or 99)
        screening_best_r2 = -1.0
        if isinstance(candidate_screening, dict):
            screening_best_r2 = float(candidate_screening.get("best_validation_r2", -1.0) or -1.0)
        fp_complexity_for_budget = self._formula_complexity(best_formula) if best_formula else 0
        bloated_multivar = (
            getattr(self, "blackbox_state_", None) is not None
            and self.blackbox_state_.enabled
            and len(getattr(self.blackbox_state_, "selected_features", []) or []) > 1
            and (fp_complexity_for_budget > 24 or term_count > 6)
        )
        if (
            getattr(self, "blackbox_state_", None) is not None
            and self.blackbox_state_.enabled
            and compact_fast_path
            and compact_terms <= 6
            and current_r2 >= 0.80
            and screening_best_r2 < float(blackbox_search_plan.get("candidate_shrink_r2", 0.95))
            and not bloated_multivar
        ):
            effective_timeout = min(effective_timeout, 18.0)
            blackbox_search_plan["population_multiplier"] = min(
                float(blackbox_search_plan.get("population_multiplier", 1.0)),
                0.90,
            )
            blackbox_search_plan["generation_multiplier"] = min(
                float(blackbox_search_plan.get("generation_multiplier", 1.0)),
                0.75,
            )
            blackbox_search_plan["seed_budget"] = min(
                int(blackbox_search_plan.get("seed_budget", 8)),
                8,
            )
            if isinstance(self.blackbox_diagnostics_, dict):
                self.blackbox_diagnostics_["evolution_budget_policy"] = "short_compact_blackbox_validation_probe"
        elif bloated_multivar and isinstance(self.blackbox_diagnostics_, dict):
            self.blackbox_diagnostics_["evolution_budget_policy"] = "full_budget_bloated_multivar_structure_search"

        basis_val_r2_for_budget = -1.0
        if isinstance(basis_result, dict):
            basis_val_r2_for_budget = float(basis_result.get("validation_r2", -1.0))
        if (
            getattr(self, "blackbox_state_", None) is not None
            and self.blackbox_state_.enabled
            and screening_best_r2 < 0.65
            and basis_val_r2_for_budget < 0.70
        ):
            effective_timeout = min(effective_timeout, 24.0)
            blackbox_search_plan["population_multiplier"] = min(
                float(blackbox_search_plan.get("population_multiplier", 1.0)),
                0.80,
            )
            blackbox_search_plan["generation_multiplier"] = min(
                float(blackbox_search_plan.get("generation_multiplier", 1.0)),
                0.70,
            )
            blackbox_search_plan["seed_budget"] = min(
                int(blackbox_search_plan.get("seed_budget", 8)),
                8,
            )
            blackbox_search_plan["focus"] = "weak_screening_probe"
            if isinstance(self.blackbox_diagnostics_, dict):
                self.blackbox_diagnostics_["evolution_budget_policy"] = "short_weak_screening_probe"

        if (
            getattr(self, "_blackbox_feature_fallback_activated", False)
            and getattr(self, "blackbox_state_", None) is not None
            and self.blackbox_state_.enabled
            and self.n_features_in_ >= 8
            and best_formula is not None
        ):
            need_evolution = False
            blackbox_candidate_accepted = True
            if isinstance(self.blackbox_diagnostics_, dict):
                self.blackbox_diagnostics_["evolution_budget_policy"] = "skip_high_dim_feature_fallback"
                self.blackbox_diagnostics_["evolution_skipped_reason"] = "all_feature_supervised_basis"

        if isinstance(self.blackbox_diagnostics_, dict) and getattr(self, "blackbox_state_", None) is not None and self.blackbox_state_.enabled:
            self.blackbox_diagnostics_["search_inflation"] = {
                "population_multiplier": float(blackbox_search_plan.get("population_multiplier", 1.0)),
                "generation_multiplier": float(blackbox_search_plan.get("generation_multiplier", 1.0)),
                "timeout_multiplier": float(blackbox_search_plan.get("timeout_multiplier", 1.0)),
                "seed_budget": int(blackbox_search_plan.get("seed_budget", 0)),
                "screening_budget": int(blackbox_search_plan.get("screening_budget", 0)),
                "focus": blackbox_search_plan.get("focus", "balanced"),
            }

        if need_evolution and _elapsed() < effective_timeout:
            if not CPP_AVAILABLE:
                if best_formula is None:
                    raise ImportError(
                        "Glassbox C++ core (_core.pyd/.so) not found. "
                        "Please build the backend first."
                    )
            else:
                evo_formula = None
                evo_mse = float('inf')
                # Try guided evolution (beam search) only if R² is low
                if (self.use_guided_evolution and operator_hints
                    and self.n_features_in_ == 1
                    and (current_r2 < self.evolution_skip_r2 or not fast_path_cv_ok)
                    and _elapsed() < effective_timeout):
                    try:
                        from classifier_fast_path import run_guided_evolution  # type: ignore

                        x_t = torch.tensor(X, dtype=torch.float32)
                        y_t = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

                        hints = dict(operator_hints)
                        hints['operators'] = set(hints.get('operators', set()))
                        hints['frequencies'] = list(hints.get('frequencies', detected_omegas or []))
                        hints['powers'] = list(hints.get('powers', []))
                        hints['has_rational'] = bool(hints.get('has_rational', False))
                        hints['has_exp_decay'] = bool(hints.get('has_exp_decay', False))
                        hints['active_terms'] = list(hints.get('active_terms', []))

                        # Blend proposer priors into hints if available
                        if proposer_payload.get("valid"):
                            proposer_priors = proposer_payload.get("operator_priors", {})
                            if proposer_priors:
                                if "operators" not in hints:
                                    hints["operators"] = set()
                                for op, prob in proposer_priors.items():
                                    if prob > 0.15:
                                        hints["operators"].add(op)

                        # Check if any proposer skeleton is ALREADY a very good fit
                        # to avoid launching evolution if we just need minor constant refinement.
                        best_cand = None
                        best_cand_mse = float('inf')
                        for cand in (candidate_formulas or []):
                            mse_c = cand.get('mse', float('inf'))
                            try:
                                mse_c = float(mse_c)
                            except (TypeError, ValueError):
                                mse_c = float('inf')
                            if mse_c < best_cand_mse:
                                best_cand_mse = mse_c
                                best_cand = cand
                        
                        # Short-circuit: if a proposer skeleton is already better than fast-path 
                        # and very good, we can skip full evolution and just use it.
                        # Must keep the argmin formula, not candidate_formulas[0] (S1-2).
                        if (
                            best_cand is not None
                            and best_cand_mse < 1e-6
                            and best_cand_mse < (best_mse or float('inf'))
                        ):
                            print(f"  [Proposer] Rapid hit (MSE={best_cand_mse:.2e}), using skeleton directly.")
                            best_formula = best_cand.get('formula') or best_cand.get('base_formula')
                            best_mse = best_cand_mse
                            blackbox_candidate_accepted = bool(
                                getattr(self, "blackbox_state_", None) is not None
                                and self.blackbox_state_.enabled
                            ) or blackbox_candidate_accepted
                            need_evolution = False 
                        else:
                            # Pass proposer uncertainty to guide beam count when available.
                            p_unc = proposer_payload.get("sequence_uncertainty", {})
                            if not isinstance(p_unc, dict):
                                p_unc = {}
                            p_entropy = p_unc.get("entropy")
                            confidence = 1.0 - (0.5 if p_entropy is None else float(p_entropy))
                            guided_generations = _clamp_int(
                                min(40, self.generations // 10)
                                * float(blackbox_search_plan.get("generation_multiplier", 1.0)),
                                default=min(40, self.generations // 10),
                                lo=10,
                                hi=max(10, int(self.generations)),
                            )
                            guided_population = _clamp_int(
                                min(30, self.population_size)
                                * float(blackbox_search_plan.get("population_multiplier", 1.0)),
                                default=min(30, self.population_size),
                                lo=10,
                                hi=max(10, int(self.population_size)),
                            )
                            guided_search_plan = dict(blackbox_search_plan or {})
                            guided_deadline = min(float(effective_timeout), float(max(1, self.timeout)))
                            guided_remaining = max(1, int(guided_deadline - _elapsed()))
                            guided_search_plan["timeout_seconds"] = min(
                                int(guided_search_plan.get("timeout_seconds", guided_remaining) or guided_remaining),
                                guided_remaining,
                            )
                            guided_kw = dict(
                                generations=guided_generations,
                                population_size=guided_population,
                                device=self.device or "cpu",
                                candidate_formulas=candidate_formulas,
                                confidence=confidence,
                                search_plan=guided_search_plan,
                            )
                            if (
                                getattr(self, "sample_weight_provided_", False)
                                and getattr(self, "sample_weight_", None) is not None
                            ):
                                guided_kw["y_weights"] = np.ascontiguousarray(
                                    self.sample_weight_, dtype=np.float64
                                )
                            mode = str(getattr(self, "loss_mode", "mse") or "mse")
                            if mode != "mse":
                                guided_kw["loss_mode"] = mode
                                if getattr(self, "huber_delta", None) is not None:
                                    guided_kw["huber_delta"] = float(self.huber_delta)
                                guided_kw["trim_fraction"] = float(
                                    getattr(self, "trim_fraction", 0.1) or 0.1
                                )
                            guided_kw.update(self._evolution_units_kwargs())
                            try:
                                guided_result = run_guided_evolution(
                                    x_t, y_t, hints, **guided_kw
                                )
                            except TypeError:
                                guided_kw.pop("y_weights", None)
                                guided_kw.pop("loss_mode", None)
                                guided_kw.pop("huber_delta", None)
                                guided_kw.pop("trim_fraction", None)
                                guided_kw.pop("input_units", None)
                                guided_kw.pop("output_units", None)
                                guided_kw.pop("dim_penalty_weight", None)
                                guided_result = run_guided_evolution(
                                    x_t, y_t, hints, **guided_kw
                                )

                            if guided_result and guided_result.get('formula'):
                                evo_formula = guided_result['formula']
                                evo_mse = guided_result.get('mse', float('inf'))
                    except Exception as e:
                        print(f"  [Guided evolution skipped: {e}]")

                # Fall back to raw C++ evolution
                if (evo_formula is None or evo_mse >= self.early_stop_mse) and _elapsed() < effective_timeout:
                    try:
                        X_list = [X[:, i].astype(np.float64) for i in range(self.n_features_in_)]
                        y_arr = y.astype(np.float64).flatten()
                        if candidate_formulas is None:
                            candidate_formulas = (
                                [{
                                    "formula": best_formula,
                                    "mse": best_mse or float("inf"),
                                    "complexity": self._formula_complexity(best_formula),
                                    "validation_r2": current_r2,
                                    "from_fast_path": True,
                                }]
                                if best_formula else None
                            )

                        best_refined_candidate = None
                        if candidate_formulas:
                            best_refined_candidate = min(
                                candidate_formulas,
                                key=lambda c: (
                                    _finite_float(c.get("mse"), float("inf")),
                                    _finite_float(c.get("complexity"), float("inf")),
                                ),
                            )
                            best_refined_mse = _finite_float(best_refined_candidate.get("mse"), float("inf"))
                            if best_refined_mse < best_mse:
                                best_formula = best_refined_candidate.get("formula", best_formula)
                                best_mse = best_refined_mse
                            if (
                                np.isfinite(best_refined_mse)
                                and (
                                    best_refined_mse <= self.early_stop_mse
                                    or _finite_float(best_refined_candidate.get("validation_r2"), -1.0) >= max(
                                        float(blackbox_search_plan.get("candidate_acceptance_r2", 0.985)),
                                        min(self.evolution_skip_r2, 0.999999),
                                    )
                                )
                            ):
                                blackbox_candidate_accepted = bool(
                                    getattr(self, "blackbox_state_", None) is not None
                                    and self.blackbox_state_.enabled
                                ) or blackbox_candidate_accepted
                                if getattr(self, "blackbox_state_", None) is not None and self.blackbox_state_.enabled:
                                    effective_timeout = min(effective_timeout, max(_elapsed() + 2.0, 3.0))
                                    blackbox_search_plan["population_multiplier"] = min(
                                        float(blackbox_search_plan.get("population_multiplier", 1.0)),
                                        0.50,
                                    )
                                    blackbox_search_plan["generation_multiplier"] = min(
                                        float(blackbox_search_plan.get("generation_multiplier", 1.0)),
                                        0.35,
                                    )
                                    if isinstance(self.blackbox_diagnostics_, dict):
                                        self.blackbox_diagnostics_["evolution_budget_policy"] = "tiny_accepted_candidate_probe"
                                else:
                                    need_evolution = False

                        n_runs = max(1, int(self.multi_start_runs))
                        best_cpp_result = None

                        # Combine operator priors from proposer to pass natively to C++
                        cpp_op_priors = []
                        if proposer_payload.get("valid"):
                            pp = proposer_payload.get("operator_priors", {})
                            if pp:
                                # Order: periodic, power, exp, log
                                cpp_op_priors = [
                                    pp.get("periodic", 0.8),
                                    pp.get("power", 0.08) + pp.get("int_pow", 0.0),
                                    pp.get("exp", 0.02),
                                    pp.get("log", 0.05)
                                ]

                        build_seed_graphs_from_candidates_fn = None
                        try:
                            from glassbox.sr.cpp.seed_graph_builder import (
                                build_seed_graphs_from_candidates,
                            )
                            build_seed_graphs_from_candidates_fn = build_seed_graphs_from_candidates
                        except Exception:
                            build_seed_graphs_from_candidates_fn = None

                        for run_idx in range(n_runs):
                            if not need_evolution:
                                break
                            blackbox_evolution_ran = bool(
                                getattr(self, "blackbox_state_", None) is not None
                                and self.blackbox_state_.enabled
                            ) or blackbox_evolution_ran
                            remaining = max(0.0, effective_timeout - _elapsed())
                            if remaining <= 0.0:
                                break

                            # Split remaining budget across yet-to-run starts.
                            runs_left = max(1, n_runs - run_idx)
                            run_timeout = max(1, int(remaining / runs_left))

                            run_seed = -1
                            if self.random_state is not None:
                                run_seed = int(self.random_state) + run_idx * 9973

                            run_candidate_formulas = self._vault_seed_candidates_for_run(
                                candidate_formulas,
                                X,
                                y,
                                best_formula,
                                best_mse,
                                run_idx,
                                max_candidates=max(
                                    8,
                                    int(blackbox_search_plan.get("seed_budget", 8)),
                                ),
                            )
                            seed_graphs_py = []
                            if build_seed_graphs_from_candidates_fn is not None:
                                try:
                                    seed_graphs_py = build_seed_graphs_from_candidates_fn(
                                        run_candidate_formulas if run_candidate_formulas else (
                                            [{"formula": best_formula, "mse": best_mse}]
                                            if best_formula else None
                                        ),
                                        max_seeds=max(
                                            4,
                                            min(
                                                24,
                                                int(blackbox_search_plan.get("seed_budget", 10)),
                                            ),
                                        ),
                                    )
                                except Exception:
                                    seed_graphs_py = []

                            evo_kwargs = dict(
                                X_list=X_list,
                                y=y_arr,
                                pop_size=_clamp_int(
                                    self.population_size
                                    * float(blackbox_search_plan.get("population_multiplier", 1.0)),
                                    default=self.population_size,
                                    lo=10,
                                    hi=max(10, int(self.population_size * 3)),
                                ),
                                generations=_clamp_int(
                                    self.generations
                                    * float(blackbox_search_plan.get("generation_multiplier", 1.0)),
                                    default=self.generations,
                                    lo=10,
                                    hi=max(10, int(self.generations * 4)),
                                ),
                                early_stop_mse=self.early_stop_mse,
                                seed_omegas=detected_omegas,
                                op_priors=cpp_op_priors,
                                allowed_unary_ops=list(blackbox_search_plan.get("allowed_unary_ops", [])),
                                binary_op_priors=list(blackbox_search_plan.get("binary_op_priors", [])),
                                allowed_binary_ops=list(blackbox_search_plan.get("allowed_binary_ops", [])),
                                timeout_seconds=run_timeout,
                                p_min=_clamp_float(proposer_plan.get("p_min"), self.p_min, -8.0, 3.0),
                                p_max=_clamp_float(proposer_plan.get("p_max"), self.p_max, 1.0, 10.0),
                                use_nsga2=self.use_nsga2,
                                num_islands=self.num_islands,
                                migration_interval=self.migration_interval,
                                migration_size=self.migration_size,
                                arithmetic_temperature=self.arithmetic_temperature,
                                random_seed=run_seed,
                                acceptable_complexity=_clamp_int(
                                    blackbox_search_plan.get("acceptable_complexity"),
                                    default=15,
                                    lo=5,
                                    hi=80,
                                ),
                                early_stop_max_nodes=_clamp_int(
                                    blackbox_search_plan.get("early_stop_max_nodes"),
                                    default=50,
                                    lo=10,
                                    hi=120,
                                ),
                                multi_allowed_unary_ops=blackbox_search_plan.get("multi_allowed_unary_ops", []),
                                multi_binary_op_priors=blackbox_search_plan.get("multi_binary_op_priors", []),
                                multi_allowed_binary_ops=blackbox_search_plan.get("multi_allowed_binary_ops", []),
                                seed_graphs_py=seed_graphs_py,
                            )
                            if (
                                getattr(self, "sample_weight_provided_", False)
                                and getattr(self, "sample_weight_", None) is not None
                            ):
                                evo_w = np.ascontiguousarray(
                                    self.sample_weight_, dtype=np.float64
                                )
                                if evo_w.shape[0] == y_arr.shape[0]:
                                    evo_kwargs["y_weights"] = evo_w
                            mode = str(getattr(self, "loss_mode", "mse") or "mse")
                            if mode != "mse":
                                evo_kwargs["loss_mode"] = mode
                                if getattr(self, "huber_delta", None) is not None:
                                    evo_kwargs["huber_delta"] = float(self.huber_delta)
                                evo_kwargs["trim_fraction"] = float(
                                    getattr(self, "trim_fraction", 0.1) or 0.1
                                )
                            evo_kwargs.update(self._evolution_units_kwargs())
                            try:
                                result = _core.run_evolution(**evo_kwargs)
                            except TypeError:
                                evo_kwargs.pop("y_weights", None)
                                evo_kwargs.pop("loss_mode", None)
                                evo_kwargs.pop("huber_delta", None)
                                evo_kwargs.pop("trim_fraction", None)
                                evo_kwargs.pop("input_units", None)
                                evo_kwargs.pop("output_units", None)
                                evo_kwargs.pop("dim_penalty_weight", None)
                                result = _core.run_evolution(**evo_kwargs)

                            raw_mse = result.get('best_mse', float('inf'))
                            raw_formula = result.get('formula', '')

                            self._update_specialist_vault_after_run(
                                run_candidate_formulas,
                                X,
                                y,
                                run_idx,
                                best_formula,
                                run_formula=raw_formula,
                                run_mse=raw_mse,
                            )

                            if raw_mse < evo_mse:
                                evo_formula = raw_formula
                                evo_mse = raw_mse
                                best_cpp_result = result

                            if raw_mse <= self.early_stop_mse:
                                break

                        if best_cpp_result is not None:
                            # Store best C++ result for inspection
                            self.nodes_ = best_cpp_result.get('nodes', [])
                            self.output_weights_ = best_cpp_result.get('output_weights', [])
                            self.output_bias_ = best_cpp_result.get('output_bias', 0.0)
                            self.evolution_wall_time_sec_ = best_cpp_result.get('evolution_wall_time_sec')
                            self.time_to_first_exact_sec_ = best_cpp_result.get('time_to_first_exact_sec')
                            self.time_to_first_acceptable_sec_ = best_cpp_result.get('time_to_first_acceptable_sec')
                            self.generation_to_first_exact_ = best_cpp_result.get('generation_to_first_exact')
                            self.generation_to_first_acceptable_ = best_cpp_result.get('generation_to_first_acceptable')
                            self.openmp_threads_ = best_cpp_result.get('openmp_threads')
                            self.evolution_random_seed_ = best_cpp_result.get('random_seed')
                            if 'pareto_front' in best_cpp_result:
                                self.pareto_front_ = best_cpp_result['pareto_front']
                    except Exception as e:
                        print(f"  [C++ evolution error: {e}]")

                # Take evolution result if it wins under direct formula evaluation.
                if evo_formula:
                    self.evolution_candidate_formula_ = evo_formula
                    self.evolution_candidate_mse_ = evo_mse
                    if getattr(self, "blackbox_state_", None) is not None and self.blackbox_state_.enabled:
                        selection_source = self._compare_blackbox_formulas(best_formula, evo_formula, X, y)
                        if selection_source == "challenger":
                            selected_formula, selected_mse = evo_formula, self._formula_mse(evo_formula, X, y)
                        else:
                            selected_formula, selected_mse = best_formula, self._formula_mse(best_formula, X, y)
                    else:
                        selected_formula, selected_mse, selection_source = self._select_final_formula(
                            best_formula,
                            best_mse,
                            evo_formula,
                            evo_mse,
                            X,
                            y,
                        )
                    if (
                        getattr(self, "blackbox_state_", None) is not None
                        and self.blackbox_state_.enabled
                        and selection_source == "challenger"
                        and best_formula
                        and np.isfinite(best_mse)
                        and np.isfinite(evo_mse)
                        and evo_mse > 0.88 * float(best_mse)
                    ):
                        # Allow modest MSE regression when evolution is much simpler
                        # (structure recovery over bloated high-R2 kitchen-sink).
                        inc_comp = self._formula_complexity(best_formula)
                        evo_comp = self._formula_complexity(evo_formula)
                        n_feat = int(np.asarray(X).shape[1]) if np.ndim(X) == 2 else 1
                        allow_simpler = (
                            n_feat > 1
                            and evo_comp + 8 <= inc_comp
                            and evo_mse <= 1.15 * float(best_mse)
                        )
                        if not allow_simpler:
                            selection_source = "incumbent"
                            selected_formula = best_formula
                            selected_mse = self._formula_mse(best_formula, X, y)
                    if selection_source == "challenger":
                        # Auto-weight guard: do not promote bloated evolution winners
                        # that fail unweighted full/holdout R² (Nguyen-1 outliers).
                        if self._auto_noise_guard_active():
                            g = self._evaluate_auto_weight_guard(selected_formula, X, y)
                            if not g.get("ok"):
                                selection_source = "incumbent"
                                selected_formula = best_formula
                                selected_mse = self._formula_mse(best_formula, X, y) if best_formula else selected_mse
                                if isinstance(self.blackbox_diagnostics_, dict):
                                    self.blackbox_diagnostics_["evolution_auto_weight_guard"] = g
                            else:
                                self._register_auto_weight_fallback_candidate(
                                    selected_formula, X, y, source="evolution_ok"
                                )
                        if selection_source == "challenger":
                            best_formula = selected_formula
                            best_mse = selected_mse
                            blackbox_evolution_improved = bool(
                                getattr(self, "blackbox_state_", None) is not None
                                and self.blackbox_state_.enabled
                            ) or blackbox_evolution_improved
                    if isinstance(self.blackbox_diagnostics_, dict):
                        self.blackbox_diagnostics_["evolution_selection"] = {
                            "incumbent_formula": best_formula if selection_source != "challenger" else None,
                            "challenger_formula": evo_formula,
                            "challenger_mse": float(evo_mse) if np.isfinite(evo_mse) else None,
                            "selected": selection_source,
                        }
                    print(
                        "  [Evolution] "
                        f"candidate_mse={float(evo_mse):.6g} "
                        f"selected={selection_source} "
                        f"formula={(evo_formula or '0')[:120]}"
                    )
        elif need_evolution and _elapsed() >= effective_timeout:
            print(f"  [Timeout: skipping evolution after {_elapsed():.1f}s (budget={effective_timeout:.1f}s)]")

        # ── Stage 3: Formula Simplification & Noise Reduction ──
        if best_formula:
            if getattr(self, "blackbox_state_", None) is not None and self.blackbox_state_.enabled:
                pareto_candidates = []
                if best_formula:
                    pareto_candidates.append({
                        "formula": best_formula,
                        "mse": best_mse,
                        "complexity": self._formula_complexity(best_formula),
                        "source": "incumbent",
                    })
                for cand in candidate_formulas or []:
                    formula = str(cand.get("formula", "")).strip()
                    if formula:
                        item = dict(cand)
                        item["source"] = item.get("source", "candidate_screening")
                        pareto_candidates.append(item)
                basis_result = getattr(self, "blackbox_basis_model_", None)
                if isinstance(basis_result, dict) and basis_result.get("formula"):
                    pareto_candidates.append({
                        "formula": basis_result.get("formula"),
                        "mse": basis_result.get("mse", float("inf")),
                        "complexity": basis_result.get("complexity", self._formula_complexity(basis_result.get("formula"))),
                        "source": "basis_model",
                    })
                engineered_result = getattr(self, "blackbox_engineered_basis_model_", None)
                if isinstance(engineered_result, dict) and engineered_result.get("formula"):
                    pareto_candidates.append({
                        "formula": engineered_result.get("formula"),
                        "mse": engineered_result.get("mse", float("inf")),
                        "complexity": engineered_result.get("complexity", self._formula_complexity(engineered_result.get("formula"))),
                        "source": "engineered_basis",
                    })
                if getattr(self, "evolution_candidate_formula_", None):
                    pareto_candidates.append({
                        "formula": self.evolution_candidate_formula_,
                        "mse": getattr(self, "evolution_candidate_mse_", float("inf")),
                        "complexity": self._formula_complexity(self.evolution_candidate_formula_),
                        "source": "evolution",
                    })
                # Keep promoted structure seed in final Pareto so kitchen-sink cannot bury it.
                promoted = getattr(self, "_promoted_structure_seed_", None)
                if isinstance(promoted, dict) and promoted.get("formula"):
                    pareto_candidates.append({
                        "formula": promoted["formula"],
                        "mse": promoted.get("mse", float("inf")),
                        "complexity": promoted.get("complexity", self._formula_complexity(promoted["formula"])),
                        "source": "structure_seed",
                        "from_structure_seed": True,
                    })
                for cand in candidate_formulas or []:
                    if (cand or {}).get("from_structure_seed") and (cand or {}).get("formula"):
                        pareto_candidates.append({
                            "formula": cand["formula"],
                            "mse": cand.get("mse", float("inf")),
                            "complexity": cand.get("complexity", self._formula_complexity(cand.get("formula"))),
                            "source": "structure_seed_pool",
                            "from_structure_seed": True,
                        })
                for idx, front_item in enumerate(getattr(self, "pareto_front_", []) or []):
                    if not isinstance(front_item, dict):
                        continue
                    formula = str(front_item.get("formula", "") or "").strip()
                    if not formula:
                        continue
                    pareto_candidates.append({
                        "formula": formula,
                        "mse": front_item.get("mse", float("inf")),
                        "complexity": front_item.get("complexity", self._formula_complexity(formula)),
                        "source": f"evolution_pareto_{idx}",
                        "pareto_rank": front_item.get("pareto_rank"),
                        "raw_nodes": front_item.get("raw_nodes"),
                    })
                pareto_choice = self._select_blackbox_pareto_formula(pareto_candidates, X, y)
                if pareto_choice is not None:
                    # Prefer structure seed when it is nearly as good and much simpler.
                    seed_cands = [
                        c for c in pareto_candidates
                        if (c or {}).get("from_structure_seed") or str((c or {}).get("source", "")).startswith("structure_seed")
                    ]
                    if seed_cands and pareto_choice is not None:
                        best_seed_c = min(
                            seed_cands,
                            key=lambda c: (
                                float(c.get("mse", float("inf"))),
                                int(c.get("complexity") or 999),
                            ),
                        )
                        try:
                            choice_mse = float(pareto_choice.get("mse", float("inf")))
                            seed_mse_f = float(best_seed_c.get("mse", float("inf")))
                            choice_comp = int(pareto_choice.get("complexity") or self._formula_complexity(pareto_choice.get("formula")))
                            seed_comp_f = int(best_seed_c.get("complexity") or self._formula_complexity(best_seed_c.get("formula")))
                            if (
                                np.isfinite(seed_mse_f)
                                and seed_mse_f <= choice_mse * 1.15 + 1e-15
                                and seed_comp_f + 8 <= choice_comp
                            ):
                                pareto_choice = dict(best_seed_c)
                                pareto_choice["source"] = "structure_seed_prefer_simple"
                        except Exception:
                            pass
                    best_formula = pareto_choice["formula"]
                    best_mse = pareto_choice["mse"]
                    if isinstance(self.blackbox_diagnostics_, dict):
                        self.blackbox_diagnostics_["final_pareto_selection"] = {
                            "source": pareto_choice.get("source"),
                            "validation_mse": pareto_choice.get("validation_mse"),
                            "validation_r2": pareto_choice.get("validation_r2"),
                            "complexity": pareto_choice.get("complexity"),
                            "risk_score": pareto_choice.get("risk_score"),
                            "generalization_gap": pareto_choice.get("generalization_gap"),
                            "evaluated_candidates": pareto_choice.get("evaluated_candidates"),
                            "best_raw_validation_mse": pareto_choice.get("best_raw_validation_mse"),
                        }
            _cleanup_kw = {"stage": "final_fit"}
            if self._auto_noise_guard_active():
                # Phase 6: allow more term dropping when auto soft-MAD is active.
                _cleanup_kw["relative_slack"] = 0.22
                _cleanup_kw["absolute_slack"] = 1e-8
            best_formula = self._cleanup_formula_with_fidelity_guard(
                best_formula,
                X,
                y,
                **_cleanup_kw,
            )
            if getattr(self, "blackbox_state_", None) is not None and self.blackbox_state_.enabled:
                remapped = formula_from_search_to_original_space(
                    best_formula,
                    self.blackbox_state_,
                )
                best_formula = remapped
                # Re-fit free constants in original space (std remap inflates constants;
                # soft_l1 helps under outliers). Then snap near-integers for Exact.
                try:
                    best_formula, best_mse = self._polish_original_space_structure_formula(
                        best_formula,
                        X_original,
                        y_original,
                    )
                except Exception:
                    pass
                # Always compete original-space free-const family fit on multi-var
                # blackbox (IRLS under spikes). No auto-win — only if better inliers.
                try:
                    n_sel = len(getattr(self.blackbox_state_, "selected_features", []) or [])
                    if n_sel >= 2:
                        orig_seed = self._fit_original_space_structure_winner(
                            X_original,
                            y_original,
                            self.blackbox_state_,
                        )
                        if orig_seed is not None:
                            o_f = str(orig_seed.get("formula") or "")
                            o_m = float(orig_seed.get("mse", float("inf")))
                            o_in = float(orig_seed.get("inlier_mse", o_m))
                            # Polish the original-space family with IRLS + integer snap
                            try:
                                o_f2, o_m2 = self._polish_original_space_structure_formula(
                                    o_f, X_original, y_original
                                )
                                if o_f2:
                                    o_f = o_f2
                                    o_m = float(o_m2) if np.isfinite(o_m2) else o_m
                                    try:
                                        o_pred = self._safe_eval_formula_array(o_f, X_original)
                                        o_in = self._inlier_mse(o_pred, y_original)
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                            cur_m = self._formula_mse(best_formula, X_original, y_original)
                            try:
                                cur_pred = self._safe_eval_formula_array(best_formula, X_original)
                                cur_in = self._inlier_mse(cur_pred, y_original)
                            except Exception:
                                cur_in = cur_m
                            # Prefer excellent-inlier structure over noisy full-MSE winners.
                            # Protocol Exact cares about clean algebraic form; spike-dominated
                            # full MSE must not block near-Exact free-const families.
                            prefer_structure = False
                            if o_f and np.isfinite(o_in):
                                if o_in < 1e-6 and (not np.isfinite(cur_in) or cur_in > 1e-6):
                                    prefer_structure = True
                                elif o_in < 1e-8 and (
                                    not np.isfinite(cur_in) or cur_in > o_in * 10.0
                                ):
                                    prefer_structure = True
                                elif o_in < cur_in * 0.98:
                                    prefer_structure = True
                                elif np.isfinite(o_m) and o_m < cur_m * 0.98:
                                    prefer_structure = True
                                elif (
                                    o_in <= max(cur_in * 1.05, 1e-8)
                                    and self._formula_complexity(o_f) + 8
                                    <= self._formula_complexity(best_formula)
                                ):
                                    prefer_structure = True
                                elif (
                                    np.isfinite(o_in)
                                    and o_in < 1e-5
                                    and np.isfinite(cur_in)
                                    and cur_in > 1e-4
                                    and self._formula_complexity(o_f)
                                    <= self._formula_complexity(best_formula) + 12
                                ):
                                    prefer_structure = True
                            if prefer_structure:
                                best_formula, best_mse = o_f, o_m
                                if isinstance(self.blackbox_diagnostics_, dict):
                                    self.blackbox_diagnostics_["original_space_structure_winner"] = {
                                        "formula": o_f[:160],
                                        "mse": o_m,
                                        "inlier_mse": o_in if np.isfinite(o_in) else None,
                                        "complexity": self._formula_complexity(o_f),
                                        "replaced_remapped": True,
                                    }
                except Exception:
                    pass
                # Phase C: re-score remapped formula on original-space holdout
                # so scaled-space false confidence cannot lock the winner.
                try:
                    holdout_n = int(max(8, round(len(y_original) * 0.20)))
                    holdout_n = min(holdout_n, max(0, len(y_original) - 16))
                    if holdout_n >= 8:
                        X_val = X_original[-holdout_n:]
                        y_val = np.asarray(y_original[-holdout_n:], dtype=np.float64).reshape(-1)
                        pred_val = self._safe_eval_formula_array(best_formula, X_val)
                        holdout_mse = float(np.mean((np.asarray(pred_val, dtype=np.float64).reshape(-1) - y_val) ** 2))
                        y_var_h = max(float(np.var(y_val)), 1e-12)
                        holdout_r2 = float(1.0 - holdout_mse / y_var_h) if np.isfinite(holdout_mse) else None
                        if isinstance(self.blackbox_diagnostics_, dict):
                            self.blackbox_diagnostics_["original_space_holdout"] = {
                                "n": int(holdout_n),
                                "mse": holdout_mse if np.isfinite(holdout_mse) else None,
                                "r2": holdout_r2,
                            }
                        if np.isfinite(holdout_mse):
                            best_mse = holdout_mse
                except Exception:
                    pass
                original_linear = getattr(self, "_blackbox_original_linear_fallback", None)
                if isinstance(original_linear, dict) and original_linear.get("formula"):
                    holdout_n = int(max(8, round(len(y_original) * 0.25)))
                    holdout_n = min(holdout_n, len(y_original) - 16)
                    tail_split = None
                    if holdout_n > 0:
                        tail_split = {
                            "X_val": X_original[-holdout_n:],
                            "y_val": y_original[-holdout_n:],
                        }
                    if tail_split is not None:
                        try:
                            current_pred = self._safe_eval_formula_array(best_formula, tail_split["X_val"])
                            linear_pred = self._safe_eval_formula_array(original_linear["formula"], tail_split["X_val"])
                            current_val_mse = float(np.mean((current_pred - tail_split["y_val"]) ** 2))
                            linear_val_mse = float(np.mean((linear_pred - tail_split["y_val"]) ** 2))
                        except Exception:
                            current_val_mse = float("inf")
                            linear_val_mse = float("inf")
                        if np.isfinite(linear_val_mse) and (
                            not np.isfinite(current_val_mse)
                            or linear_val_mse <= current_val_mse * 1.03 + 1e-12
                        ):
                            best_formula = original_linear["formula"]
                            best_mse = float(original_linear.get("mse", best_mse))
                            blackbox_candidate_accepted = True
                            if isinstance(self.blackbox_diagnostics_, dict):
                                self.blackbox_diagnostics_["original_linear_fallback_selection"] = {
                                    "selected": True,
                                    "current_tail_mse": current_val_mse,
                                    "fallback_tail_mse": linear_val_mse,
                                    "validation_r2": original_linear.get("validation_r2"),
                                    "n_terms": original_linear.get("n_terms"),
                                }
                        elif isinstance(self.blackbox_diagnostics_, dict):
                            self.blackbox_diagnostics_["original_linear_fallback_selection"] = {
                                "selected": False,
                                "current_tail_mse": current_val_mse,
                                "fallback_tail_mse": linear_val_mse,
                                "validation_r2": original_linear.get("validation_r2"),
                                "n_terms": original_linear.get("n_terms"),
                            }
                if isinstance(self.blackbox_diagnostics_, dict):
                    self.blackbox_diagnostics_["domain_failure_rate"] = self._formula_domain_failure_rate(
                        best_formula,
                        X_original,
                    )
                    selection_outcome = {
                        "candidate_screening_win": bool(blackbox_candidate_accepted and not blackbox_evolution_ran),
                        "evolution_ran": bool(blackbox_evolution_ran),
                        "evolution_win": bool(blackbox_evolution_improved),
                        "source": (
                            "candidate_screening"
                            if blackbox_candidate_accepted and not blackbox_evolution_ran
                            else ("evolution" if blackbox_evolution_improved else "incumbent_or_basis")
                        ),
                    }
                    
                    specialist_track = "incumbent path"
                    final_pareto = self.blackbox_diagnostics_.get("final_pareto_selection")
                    if isinstance(final_pareto, dict):
                        winner_source = final_pareto.get("source")
                        if winner_source == "evolution":
                            if self.has_composed_seeds_:
                                specialist_track = "composed seed + evolution"
                                self.composition_seeded_evolution_ = True
                            else:
                                specialist_track = "incumbent path"
                        elif winner_source in ("specialist_composition", "specialist_residual_composition", "candidate_screening", "proposer", "basis_model", "engineered_basis"):
                            specialist_track = "screening only"
                            if winner_source in ("specialist_composition", "specialist_residual_composition"):
                                self.composition_won_final_selection_ = True
                                self.composition_improved_mse_ = True
                        elif winner_source == "incumbent":
                            specialist_track = "incumbent path"
                    elif selection_outcome["candidate_screening_win"]:
                        specialist_track = "screening only"
                    
                    selection_outcome["specialist_track"] = specialist_track
                    self.blackbox_diagnostics_["selection_outcome"] = selection_outcome
                    self.specialist_track_ = specialist_track

            residual_X = X_original if (
                getattr(self, "blackbox_state_", None) is not None
                and self.blackbox_state_.enabled
            ) else X
            residual_y = y_original if residual_X is X_original else y
            residual_base_formula = best_formula
            residual_prior_n_features = self.n_features_in_
            try:
                self.n_features_in_ = residual_X.shape[1]
                residual_candidate = self._run_residual_boosting(residual_X, residual_y, best_formula)
            finally:
                self.n_features_in_ = residual_prior_n_features
            if residual_candidate and residual_candidate != residual_base_formula:
                residual_prior_n_features = self.n_features_in_
                self.n_features_in_ = residual_X.shape[1]
                residual_base_score, residual_base_mse, residual_base_display_mse = self._final_formula_score(
                    residual_base_formula,
                    residual_X,
                    residual_y,
                )
                residual_candidate_score, residual_candidate_mse, residual_candidate_display_mse = self._final_formula_score(
                    residual_candidate,
                    residual_X,
                    residual_y,
                )
                residual_allowed = residual_base_score * 1.01 + 1e-9 if np.isfinite(residual_base_score) else float("inf")
                residual_accepted = np.isfinite(residual_candidate_score) and residual_candidate_score <= residual_allowed
                residual_holdout = self._final_holdout_scores(residual_base_formula, residual_candidate, residual_X, residual_y)
                self.n_features_in_ = residual_prior_n_features
                if residual_holdout is not None:
                    holdout_base = residual_holdout["base_score"]
                    holdout_candidate = residual_holdout["candidate_score"]
                    holdout_allowed = holdout_base * 1.01 + 1e-9 if np.isfinite(holdout_base) else float("inf")
                    residual_accepted = (
                        residual_accepted
                        and np.isfinite(holdout_candidate)
                        and holdout_candidate <= holdout_allowed
                    )
                if residual_accepted and self._auto_noise_guard_active():
                    # Residual stages often re-bloat under auto weights (Nguyen-1).
                    g = self._evaluate_auto_weight_guard(residual_candidate, residual_X, residual_y)
                    if not g.get("ok"):
                        residual_accepted = False
                        if isinstance(self.blackbox_diagnostics_, dict):
                            self.blackbox_diagnostics_["residual_auto_weight_guard"] = g
                if residual_accepted:
                    best_formula = residual_candidate
                    self._register_auto_weight_fallback_candidate(
                        residual_candidate, residual_X, residual_y, source="residual"
                    )
                if isinstance(self.blackbox_diagnostics_, dict):
                    self.blackbox_diagnostics_["residual_boosting_final_guard"] = {
                        "accepted": bool(residual_accepted),
                        "base_mse": float(residual_base_mse) if np.isfinite(residual_base_mse) else None,
                        "candidate_mse": float(residual_candidate_mse) if np.isfinite(residual_candidate_mse) else None,
                        "base_display_mse": float(residual_base_display_mse) if np.isfinite(residual_base_display_mse) else None,
                        "candidate_display_mse": float(residual_candidate_display_mse) if np.isfinite(residual_candidate_display_mse) else None,
                        "holdout": {
                            key: (float(value) if isinstance(value, (int, float, np.floating)) and np.isfinite(value) else value)
                            for key, value in (residual_holdout or {}).items()
                        } if residual_holdout is not None else None,
                    }
            prior_n_features = self.n_features_in_
            try:
                self.n_features_in_ = X_original.shape[1]
                inception_base_formula = best_formula
                inception_candidate = self._run_inception_reuse(X_original, y_original, best_formula)
                if inception_candidate and inception_candidate != inception_base_formula:
                    inception_base_score, inception_base_mse, inception_base_display_mse = self._final_formula_score(
                        inception_base_formula,
                        X_original,
                        y_original,
                    )
                    inception_candidate_score, inception_candidate_mse, inception_candidate_display_mse = self._final_formula_score(
                        inception_candidate,
                        X_original,
                        y_original,
                    )
                    inception_allowed = inception_base_score * 1.01 + 1e-9 if np.isfinite(inception_base_score) else float("inf")
                    inception_accepted = np.isfinite(inception_candidate_score) and inception_candidate_score <= inception_allowed
                    inception_holdout = self._final_holdout_scores(
                        inception_base_formula,
                        inception_candidate,
                        X_original,
                        y_original,
                    )
                    if inception_holdout is not None:
                        holdout_base = inception_holdout["base_score"]
                        holdout_candidate = inception_holdout["candidate_score"]
                        holdout_allowed = holdout_base * 1.01 + 1e-9 if np.isfinite(holdout_base) else float("inf")
                        inception_accepted = (
                            inception_accepted
                            and np.isfinite(holdout_candidate)
                            and holdout_candidate <= holdout_allowed
                        )
                    if inception_accepted:
                        best_formula = inception_candidate
                    if isinstance(self.blackbox_diagnostics_, dict):
                        self.blackbox_diagnostics_["inception_final_guard"] = {
                            "accepted": bool(inception_accepted),
                            "base_mse": float(inception_base_mse) if np.isfinite(inception_base_mse) else None,
                            "candidate_mse": float(inception_candidate_mse) if np.isfinite(inception_candidate_mse) else None,
                            "base_display_mse": float(inception_base_display_mse) if np.isfinite(inception_base_display_mse) else None,
                            "candidate_display_mse": float(inception_candidate_display_mse) if np.isfinite(inception_candidate_display_mse) else None,
                            "holdout": {
                                key: (float(value) if isinstance(value, (int, float, np.floating)) and np.isfinite(value) else value)
                                for key, value in (inception_holdout or {}).items()
                            } if inception_holdout is not None else None,
                        }
                best_pred = self._safe_eval_formula_array(best_formula, X_original)
                best_mse = float(np.mean((best_pred - np.asarray(y_original, dtype=np.float64).reshape(-1)) ** 2))
            except Exception:
                self.n_features_in_ = prior_n_features
            else:
                self.n_features_in_ = X_original.shape[1]

        # Phase 6 tighten + Phase 3 guardrail: parsimony under auto soft-MAD,
        # then block bloated unweighted-catastrophic winners.
        try:
            eval_X_final = X_original if (
                getattr(self, "blackbox_state_", None) is not None
                and getattr(self.blackbox_state_, "enabled", False)
            ) else X
            eval_y_final = y_original if eval_X_final is X_original else y
            if best_formula:
                self._register_auto_weight_fallback_candidate(
                    best_formula, eval_X_final, eval_y_final, source="pre_guard_best"
                )
            evo_cand = getattr(self, "evolution_candidate_formula_", None)
            if evo_cand:
                self._register_auto_weight_fallback_candidate(
                    evo_cand, eval_X_final, eval_y_final, source="evolution"
                )
            if best_formula and self._auto_noise_guard_active():
                best_formula = self._phase6_noise_parsimony_pass(
                    best_formula, eval_X_final, eval_y_final, stage="final_fit"
                )
            guarded = self._apply_auto_weight_final_guard(
                best_formula,
                X_original if (
                    getattr(self, "blackbox_state_", None) is not None
                    and getattr(self.blackbox_state_, "enabled", False)
                ) else X,
                y_original if (
                    getattr(self, "blackbox_state_", None) is not None
                    and getattr(self.blackbox_state_, "enabled", False)
                ) else y,
                stage="final_fit",
            )
            if guarded and str(guarded) != str(best_formula or ""):
                best_formula = guarded
                try:
                    eval_X = X_original if (
                        getattr(self, "blackbox_state_", None) is not None
                        and getattr(self.blackbox_state_, "enabled", False)
                    ) else X
                    eval_y = y_original if eval_X is X_original else y
                    pred = self._safe_eval_formula_array(best_formula, eval_X)
                    best_mse = float(np.mean((pred - np.asarray(eval_y, dtype=np.float64).reshape(-1)) ** 2))
                except Exception:
                    pass
        except Exception as _guard_exc:
            if isinstance(getattr(self, "blackbox_diagnostics_", None), dict):
                self.blackbox_diagnostics_["auto_weight_final_guard"] = {
                    "active": True,
                    "error": str(_guard_exc)[:200],
                }

        self.formula_ = best_formula or "0"
        self.best_mse_ = best_mse
        self._restore_user_loss_mode_if_auto_switched()
        self._add_phase_time("total_fit", _time.time() - fit_start)
        return self

    def predict(self, X):
        """
        Predict using the discovered symbolic formula.
        Handles edge cases (log of zero, sqrt of negative) gracefully.
        """
        # Require a real fitted formula — many __init__ attrs end with '_' so bare
        # check_is_fitted(self) is a false positive (S1-1).
        check_is_fitted(self, attributes=["formula_"])
        X = check_array(X)

        try:
            return self._safe_eval_formula_array(self.formula_, X)
        except Exception as e:
            print(f"Prediction error: {e}")
            return np.zeros(X.shape[0])

    def get_formula(self):
        """Returns the discovered formula string."""
        check_is_fitted(self, attributes=["formula_"])
        return self.formula_

    def _reduce_formula_noise(self, formula_str, X, y):
        """Greedy backward elimination with optional weights + holdout fidelity (Phase 6)."""
        if not formula_str or formula_str == "0":
            return formula_str

        try:
            from glassbox.sr.cpp import _core
            n_feat = int(getattr(self, "n_features_in_", X.shape[1]))
            X_list = [X[:, j] for j in range(n_feat)]
            w = self._active_sample_weight(n_targets=int(np.asarray(y).reshape(-1).shape[0]))
            # Noise-aware holdout slack for C++ fidelity guard.
            _, _, slack_diag = self._noise_aware_cleanup_slack(formula_str, X, y)
            rel = float(slack_diag.get("relative_slack", 0.10))
            # Phase 6 tighten: under auto soft-MAD allow more aggressive BIC pruning.
            if self._auto_noise_guard_active():
                rel = max(rel, 0.22)
            kwargs = {
                "holdout_fraction": 0.2,
                "relative_slack": rel,
            }
            if w is not None:
                kwargs["y_weights"] = np.asarray(w, dtype=np.float64)
            try:
                return _core.reduce_formula_noise(formula_str, X_list, y, **kwargs)
            except TypeError:
                # Older extension without Phase 6 kwargs.
                if w is not None:
                    try:
                        return _core.reduce_formula_noise(formula_str, X_list, y, w)
                    except TypeError:
                        pass
                return _core.reduce_formula_noise(formula_str, X_list, y)
        except Exception:
            return formula_str
