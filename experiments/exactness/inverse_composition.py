"""Inverse-target composition: fit the argument, then re-apply the function.

The forward composition closure can only build ``op(g)`` for a single basis
column ``g``. It cannot represent a function of a *sum* or an affine argument:
``log(1+exp(x))``, ``1/(1+exp(-x))``, ``exp(-abs(x))``, ``sin(x+sin(x))``.

This module inverts the problem. For a target ``y`` it hypothesises
``y = op(F(x))``, computes ``z = op^{-1}(y)`` on the data, fits ``z`` with the
pipeline's own basis, and reconstructs ``op(fit)``. There is no named formula
list: the only inputs are the data, the outer operator, and the basis.

Branch-sensitive operators (``sin``, ``cos``) are accepted only when the
reconstruction fits to machine precision. If the true argument leaves the
principal branch, the inverse is a sawtooth that the basis cannot fit, so the
candidate is rejected rather than guessed.

Usage (from run_with_guard.py):
    --guard inverse
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from experiments.exactness import lib
from experiments.exactness.composite_basis import (
    _correlation_score,
    _finite,
    _format_formula,
    _simplify_formula,
    _usable_primitives,
    closure_columns,
    fit_dictionary,
    primitive_basis,
)

MAX_COLUMNS = 64
MAX_INNER_TERMS = 4
# Exhaustive search is affordable for a handful of terms and avoids OMP's
# greedy first pick, which can prefer a spurious column that wins on the
# training split by a fraction of a percent.
MAX_EXHAUSTIVE_TERMS = 2
MAX_EXHAUSTIVE_CANDIDATES = 16

# op name -> (inverse transform, validity guard on y)
INVERSES: dict[
    str,
    tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], bool]],
] = {
    "exp": (np.log, lambda y: bool(np.all(y > 0.0))),
    "log": (np.exp, lambda y: bool(np.all(y < 30.0))),
    "sqrt": (lambda y: y * y, lambda y: bool(np.all(y >= 0.0))),
    "reciprocal": (lambda y: 1.0 / y, lambda y: bool(np.all(np.abs(y) > 1e-6))),
    "sin": (np.arcsin, lambda y: bool(np.all(np.abs(y) <= 1.0))),
    "cos": (np.arccos, lambda y: bool(np.all(np.abs(y) <= 1.0))),
    "tanh": (np.arctanh, lambda y: bool(np.all(np.abs(y) < 1.0))),
}


def _format_candidate(op: str, inner: str) -> str:
    return f"1/({inner})" if op == "reciprocal" else f"{op}({inner})"


def inverse_dictionary(
    x: np.ndarray,
    y: np.ndarray,
    predictions: dict[str, float],
    omegas: list[float] | None,
) -> list[tuple[str, np.ndarray]]:
    """Primitives + products/quotients + ``abs(primitive)``.

    The argument of a composition is usually a primitive, a sum of two, or an
    absolute value; it is rarely itself a product of two transcendentals, so
    the full composition dictionary is unnecessary here and much slower.
    """
    basis, names = primitive_basis(x, predictions, omegas)
    primitives = _usable_primitives(basis, names)
    closed = closure_columns(
        x, y, predictions, omegas, max_columns=512
    )
    extras: list[tuple[str, np.ndarray]] = []
    for name, values in primitives:
        if np.all(values >= 0.0):
            continue
        extras.append((f"abs({name})", np.abs(values)))
    # Canonical names first: these columns are numerically identical to some
    # generated products, and the dedupe keeps the first occurrence. A bare
    # ``abs(x)`` is both readable and evaluates faithfully.
    return primitives + extras + closed


def _select_columns(
    columns: list[tuple[str, np.ndarray]],
    target: np.ndarray,
    *,
    max_columns: int = MAX_COLUMNS,
) -> list[tuple[str, np.ndarray]]:
    """Rank a fixed dictionary against a new target and keep the best.

    Dedupe happens in dictionary order, before scoring. Several generated
    columns are numerically identical (``abs(x)`` and ``x^2/Abs(x)``); the
    dictionary puts the canonical, faithfully-evaluating name first, so it is
    the one that survives.
    """
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    centered = target - target.mean()
    norm = float(np.linalg.norm(centered))
    if norm < 1e-12:
        return []

    unique: list[tuple[str, np.ndarray]] = []
    seen: set[bytes] = set()
    for name, values in columns:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.size != target.size or not np.all(np.isfinite(values)):
            continue
        if float(np.linalg.norm(values - values.mean())) < 1e-9:
            continue
        key = np.round(values / max(float(np.max(np.abs(values))), 1e-12), 9).tobytes()
        if key in seen:
            continue
        seen.add(key)
        unique.append((name, values))

    scored = [
        (_correlation_score(values, centered, norm), name, values)
        for name, values in unique
    ]
    scored.sort(key=lambda row: row[0], reverse=True)
    return [(name, values) for _score, name, values in scored[:max_columns]]


def _fit_argument(
    x: np.ndarray,
    z: np.ndarray,
    columns: list[tuple[str, np.ndarray]],
    *,
    max_terms: int = MAX_INNER_TERMS,
) -> dict[str, Any] | None:
    """Fit ``z`` with an exhaustive small-k search, then OMP for larger k.

    OMP's greedy first step compares training-split correlations, and a
    spurious column can beat the true one by a fraction of a percent while
    being wrong by orders of magnitude out of sample. At k <= 2 an exhaustive
    search over the best few candidates is cheap and removes that failure.
    """
    from itertools import combinations

    z = np.asarray(z, dtype=np.float64).reshape(-1)
    candidates = columns[:MAX_EXHAUSTIVE_CANDIDATES]
    if not candidates:
        return None
    names = [name for name, _ in candidates]
    B = np.stack([values for _, values in candidates], axis=1)
    z_mean = float(z.mean())
    B_mean = B.mean(axis=0)
    zc = z - z_mean
    Bc = B - B_mean
    norms = np.linalg.norm(Bc, axis=0)
    norms[norms < 1e-12] = 1.0
    Bn = Bc / norms
    tolerance = 1e-12 * max(float(np.mean(zc**2)), 1e-12)

    def _finalize(indices: tuple[int, ...]) -> dict[str, Any] | None:
        design = np.column_stack([B[:, list(indices)], np.ones(z.size)])
        coefficients, *_ = np.linalg.lstsq(design, z, rcond=None)
        prediction = design @ coefficients
        mse = float(np.mean((prediction - z) ** 2))
        formula = _format_formula(
            [names[index] for index in indices],
            coefficients[:-1],
            float(coefficients[-1]),
        )
        scale = max(float(np.var(z)), 1e-30)

        def _matches_fit(text: str) -> bool:
            values = lib.evaluate(text, x)
            if values is None or values.shape != prediction.shape:
                return False
            if not np.all(np.isfinite(values)):
                return False
            return float(np.mean((values - prediction) ** 2)) <= 1e-12 * scale

        chosen = formula
        simplified = _simplify_formula(formula)
        if simplified != formula and _matches_fit(simplified):
            chosen = simplified
        elif not _matches_fit(formula):
            chosen = simplified
        if not _matches_fit(chosen):
            return None
        return {
            "formula": chosen,
            "mse": mse,
            "holdout_mse": None,
            "n_terms": len(indices),
            "active_terms": [names[index] for index in indices],
            "dictionary_size": len(columns),
        }

    for k in range(1, min(MAX_EXHAUSTIVE_TERMS, max_terms) + 1):
        best: tuple[float, tuple[int, ...]] | None = None
        for combination in combinations(range(len(candidates)), k):
            coefficients, *_ = np.linalg.lstsq(Bn[:, combination], zc, rcond=None)
            mse = float(np.mean((Bn[:, combination] @ coefficients - zc) ** 2))
            if best is None or mse < best[0]:
                best = (mse, combination)
        if best is not None and best[0] <= tolerance:
            return _finalize(best[1])

    if max_terms > MAX_EXHAUSTIVE_TERMS:
        fit = fit_dictionary(x, z, candidates, ks=(3, 4))
        if fit is not None:
            return fit
    return None


def fit_inverse_composition(
    x: np.ndarray,
    y: np.ndarray,
    predictions: dict[str, float],
    omegas: list[float] | None,
    *,
    max_inner_terms: int = MAX_INNER_TERMS,
) -> dict[str, Any] | None:
    """Best ``op(F(x))`` whose argument ``F`` is in the pipeline's basis."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size != y.size or y.size < 20:
        return None
    if float(np.var(y)) < 1e-12:
        return None

    columns = inverse_dictionary(x, y, predictions, omegas)
    if not columns:
        return None

    best: dict[str, Any] | None = None
    for op, (inverse, valid) in INVERSES.items():
        if not valid(y):
            continue
        with np.errstate(all="ignore"):
            z = inverse(y)
        z = np.asarray(z, dtype=np.float64).reshape(-1)
        if z.size != y.size or not np.all(np.isfinite(z)):
            continue
        if float(np.var(z)) < 1e-12:
            continue

        selected = _select_columns(columns, z)
        fit = _fit_argument(x, z, selected, max_terms=max_inner_terms)
        if fit is None or int(fit["n_terms"]) > max_inner_terms:
            continue

        candidate = _format_candidate(op, fit["formula"])
        pred = lib.evaluate(candidate, x)
        if pred is None or pred.shape != y.shape or not np.all(np.isfinite(pred)):
            continue
        mse = float(np.mean((pred - y) ** 2))
        if not math.isfinite(mse):
            continue
        if best is None or mse < float(best["mse"]):
            best = {
                "formula": candidate,
                "mse": mse,
                "op": op,
                "inner": fit["formula"],
                "inner_mse": float(fit["mse"]),
                "n_terms": int(fit["n_terms"]),
                "holdout_mse": fit.get("holdout_mse"),
                "active_terms": fit.get("active_terms"),
                "dictionary_size": len(selected),
            }
    return best
