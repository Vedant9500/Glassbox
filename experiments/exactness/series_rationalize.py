"""Promote series-truncated polynomials to their closed form.

A polynomial can match a transcendental curve numerically while being a
truncated Taylor series. That is the approximation failure mode: in-domain
error is tiny, extrapolation is wrong, and the formula is not the generator.

This pass is general: it detects the series from the polynomial's coefficients
and rewrites it as the corresponding closed form (``sin``, ``cos``, ``exp``,
``log``, ``atan``, ``sinh``, ``cosh``). It does not contain a list of benchmark
formulas, so it applies to any curve whose fit happens to be a truncation.

A rewrite is accepted only when the closed form fits the sampled data at least
as well as the polynomial did. That prevents promoting an unrelated polynomial
that merely shares a few coefficients.

Usage (from run_with_guard.py):
    --guard series
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Callable

import numpy as np

from experiments.exactness import lib

# Minimum coefficient-agreement before a rewrite is attempted.
MIN_SERIES_SCORE = 0.85
MIN_MATCHED_TERMS = 3


def _fmt(value: float) -> str:
    return f"{float(value):.10g}"


def _trig(x: np.ndarray, c: float, a: float, w: float, phi: float) -> np.ndarray:
    return c + a * np.sin(w * x + phi)


def _exp(x: np.ndarray, c: float, a: float, w: float) -> np.ndarray:
    return c + a * np.exp(np.clip(w * x, -50.0, 50.0))


def _log1p(x: np.ndarray, c: float, a: float, w: float) -> np.ndarray:
    arg = 1.0 + w * x
    if np.any(arg <= 0):
        return np.full_like(x, np.nan)
    return c + a * np.log(arg)


def _atan(x: np.ndarray, c: float, a: float, w: float) -> np.ndarray:
    return c + a * np.arctan(w * x)


def _sinh(x: np.ndarray, c: float, a: float, w: float) -> np.ndarray:
    return c + a * np.sinh(np.clip(w * x, -50.0, 50.0))


def _cosh(x: np.ndarray, c: float, a: float, w: float) -> np.ndarray:
    return c + a * np.cosh(np.clip(w * x, -50.0, 50.0))


_MODELS: dict[str, tuple[Callable[..., np.ndarray], int]] = {
    "sin": (_trig, 4),
    "cos": (_trig, 4),
    "exp": (_exp, 3),
    "log1p": (_log1p, 3),
    "atan": (_atan, 3),
    "sinh": (_sinh, 3),
    "cosh": (_cosh, 3),
}


def _dominant_frequency(x: np.ndarray, values: np.ndarray) -> float:
    centered = values - np.mean(values)
    if centered.size < 8 or np.allclose(centered, 0.0):
        return 1.0
    mags = np.abs(np.fft.rfft(centered))
    if mags.size < 2:
        return 1.0
    idx = int(np.argmax(mags[1:]) + 1)
    freqs = np.fft.rfftfreq(centered.size, d=float(x[1] - x[0]) if x.size > 1 else 1.0)
    w = float(freqs[idx] * 2.0 * math.pi)
    return w if math.isfinite(w) and w > 1e-6 else 1.0


def _format_candidate(family: str, params: np.ndarray) -> str:
    if family in {"sin", "cos"}:
        c, a, w, phi = (float(v) for v in params)
        inner = f"{_fmt(w)}*x" if abs(phi) < 1e-9 else f"{_fmt(w)}*x+{_fmt(phi)}"
        body = f"{_fmt(a)}*sin({inner})"
    elif family == "exp":
        c, a, w = (float(v) for v in params)
        body = f"{_fmt(a)}*exp({_fmt(w)}*x)"
    elif family == "log1p":
        c, a, w = (float(v) for v in params)
        body = f"{_fmt(a)}*log(1+{_fmt(w)}*x)"
    elif family == "atan":
        c, a, w = (float(v) for v in params)
        body = f"{_fmt(a)}*atan({_fmt(w)}*x)"
    elif family == "sinh":
        c, a, w = (float(v) for v in params)
        body = f"{_fmt(a)}*sinh({_fmt(w)}*x)"
    elif family == "cosh":
        c, a, w = (float(v) for v in params)
        body = f"{_fmt(a)}*cosh({_fmt(w)}*x)"
    else:
        return ""
    return f"{body} + {_fmt(c)}" if abs(float(params[0])) > 1e-12 else body


def rationalize(
    formula: str,
    x: np.ndarray,
    y: np.ndarray,
    *,
    min_score: float = MIN_SERIES_SCORE,
) -> dict[str, Any] | None:
    """Return a closed-form rewrite when ``formula`` is a series truncation."""
    fingerprint = lib.series_fingerprint(formula)
    family = fingerprint.get("series")
    if not family or family not in _MODELS:
        return None
    if float(fingerprint.get("score") or 0.0) < min_score:
        return None
    if int(fingerprint.get("matched_nonzero") or 0) < MIN_MATCHED_TERMS:
        return None

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    poly_pred = lib.evaluate(formula, x)
    if poly_pred is None or poly_pred.shape != y.shape:
        return None
    poly_mse = float(np.mean((poly_pred - y) ** 2))
    if not math.isfinite(poly_mse):
        return None

    try:
        from scipy.optimize import curve_fit
    except Exception:
        return None

    model, n_params = _MODELS[family]
    c0 = float(np.mean(y))
    a0 = float(np.max(np.abs(y - c0))) or 1.0
    w0 = _dominant_frequency(x, y)

    starts: list[list[float]] = []
    for w_try in (w0, 0.5, 1.0, 2.0, 3.0):
        if not math.isfinite(w_try) or w_try <= 0:
            continue
        if n_params == 4:
            for phi in (0.0, math.pi / 2.0, -math.pi / 2.0):
                starts.append([c0, a0, w_try, phi])
        else:
            starts.append([c0, a0, w_try])

    best: tuple[float, np.ndarray] | None = None
    for p0 in starts:
        try:
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                params, _ = curve_fit(
                    model,
                    x,
                    y,
                    p0=p0,
                    maxfev=4000,
                )
                pred = model(x, *params)
        except Exception:
            continue
        if not np.all(np.isfinite(pred)):
            continue
        mse = float(np.mean((pred - y) ** 2))
        if not math.isfinite(mse):
            continue
        if best is None or mse < best[0]:
            best = (mse, np.asarray(params, dtype=np.float64))

    if best is None:
        return None
    mse, params = best
    # Accept only if the closed form explains the data at least as well as the
    # polynomial did.
    if mse > max(1e-12, 2.0 * poly_mse):
        return None

    rewritten = _format_candidate(family, params)
    if not rewritten:
        return None
    check = lib.evaluate(rewritten, x)
    if check is None or not np.all(np.isfinite(check)):
        return None

    return {
        "formula": rewritten,
        "family": family,
        "series_score": float(fingerprint.get("score") or 0.0),
        "mse": float(np.mean((check - y) ** 2)),
        "polynomial_mse": poly_mse,
        "params": [float(v) for v in params],
    }
