"""Operator closure over the pipeline's own primitive basis.

The fast path fits a linear combination of primitive terms. That basis is not
closed under multiplication or division, so it cannot represent a product such
as ``exp(-x)*sin(x)`` or a ratio such as ``sin(x)/x`` even when both operands
are already in the basis.

This module closes the basis over ``*``, ``/`` and reciprocal using only the
primitives the pipeline already builds. There is **no list of named formula
families**: candidates are generated from the existing basis columns and
ranked by correlation with the target. If a product is representable, the
closure finds it; if it is not, the closure cannot invent it.

Composition (``sin(cos(x))``) is a separate experiment in
``composition_closure.py``. It helps the benchmark's curated nested formulas
but not fresh ones, because the required inner columns are vocabulary terms
the benchmark happens to share. See the README.

Usage (from run_with_guard.py):
    --guard closure
"""

from __future__ import annotations

import math
import re
import warnings
from typing import Any

import numpy as np

from experiments.exactness import lib

# Bound the dictionary so the pursuit stays fast on 300-point curves.
MAX_PRIMITIVES = 120
MAX_COLUMNS = 240


def _finite(values: np.ndarray, clip: float = 1e6) -> np.ndarray:
    out = np.nan_to_num(
        np.asarray(values, dtype=np.float64), nan=0.0, posinf=clip, neginf=-clip
    )
    return np.clip(out, -clip, clip)


def _sanitize_name(name: str) -> str:
    """Make a basis column name safe to embed in a formula string."""
    text = str(name).replace("·", "*").replace("⋅", "*")
    return re.sub(r"\|([^|]+)\|", r"Abs(\1)", text)


def _safe_reciprocal(values: np.ndarray) -> np.ndarray | None:
    """Reciprocal that refuses to create a singularity inside the domain."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return None
    scale = float(np.max(np.abs(values)))
    if scale < 1e-12:
        return None
    eps = 1e-3 * scale
    if np.any(np.abs(values) < eps):
        return None
    return _finite(1.0 / values)


def _correlation_score(values: np.ndarray, y_centered: np.ndarray, y_norm: float) -> float:
    centered = values - values.mean()
    norm = float(np.linalg.norm(centered))
    if norm < 1e-9 or y_norm < 1e-12:
        return 0.0
    score = abs(float(centered @ y_centered)) / (norm * y_norm)
    return score if math.isfinite(score) else 0.0


def primitive_basis(
    x: np.ndarray,
    predictions: dict[str, float],
    omegas: list[float] | None,
    *,
    max_power: int = 6,
) -> tuple[np.ndarray, list[str]]:
    """Return the fast path's own primitive basis for this curve."""
    from scripts.classifier_fast_path import build_basis_from_predictions

    x2 = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    basis, names = build_basis_from_predictions(
        x2,
        predictions,
        threshold=0.3,
        max_power=max_power,
        detected_omegas=omegas,
        universal_basis=True,
    )
    return np.asarray(basis, dtype=np.float64), [str(n) for n in names]


def _usable_primitives(
    basis: np.ndarray,
    names: list[str],
    *,
    max_terms: int = MAX_PRIMITIVES,
) -> list[tuple[str, np.ndarray]]:
    """Keep every finite, non-constant, non-degenerate primitive column."""
    out: list[tuple[str, np.ndarray]] = []
    for idx in range(basis.shape[1]):
        name = names[idx]
        if name.strip() == "1":
            continue
        col = basis[:, idx]
        if not np.all(np.isfinite(col)):
            continue
        centered = col - col.mean()
        if float(np.linalg.norm(centered)) < 1e-9:
            continue
        out.append((name, col))
        if len(out) >= max_terms:
            break
    return out


def closure_columns(
    x: np.ndarray,
    y: np.ndarray,
    predictions: dict[str, float],
    omegas: list[float] | None,
    *,
    max_terms: int = MAX_PRIMITIVES,
    max_columns: int = MAX_COLUMNS,
) -> list[tuple[str, np.ndarray]]:
    """Generate products, quotients and reciprocals of the primitive basis.

    Every candidate is scored against the target before truncation, so the
    column that actually explains the curve is never dropped just because it
    was generated late.
    """
    basis, names = primitive_basis(x, predictions, omegas)
    primitives = _usable_primitives(basis, names, max_terms=max_terms)
    if not primitives:
        return []

    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    y_centered = y_arr - y_arr.mean()
    y_norm = float(np.linalg.norm(y_centered))
    if y_norm < 1e-12:
        return []

    # Pass 1: score every reciprocal/product/quotient without keeping values.
    scored: list[tuple[float, str, int, int]] = []
    for i, (name_i, values_i) in enumerate(primitives):
        recip_i = _safe_reciprocal(values_i)
        if recip_i is not None:
            scored.append(
                (_correlation_score(recip_i, y_centered, y_norm), "recip", i, -1)
            )
        for j, (_name_j, values_j) in enumerate(primitives):
            if i == j:
                continue
            prod = values_i * values_j
            if np.all(np.isfinite(prod)):
                scored.append(
                    (_correlation_score(prod, y_centered, y_norm), "prod", i, j)
                )
            recip_j = _safe_reciprocal(values_j)
            if recip_j is not None:
                quot = values_i * recip_j
                if np.all(np.isfinite(quot)):
                    scored.append(
                        (_correlation_score(quot, y_centered, y_norm), "quot", i, j)
                    )

    scored.sort(key=lambda row: row[0], reverse=True)

    # Pass 2: materialise only the winners.
    unique: list[tuple[str, np.ndarray]] = []
    seen: set[bytes] = set()
    for _score, kind, i, j in scored:
        name_i, values_i = primitives[i]
        if kind == "recip":
            values = _safe_reciprocal(values_i)
            if values is None:
                continue
            name = f"1/({name_i})"
        elif kind == "prod":
            name_j, values_j = primitives[j]
            name = f"({name_i})*({name_j})"
            values = values_i * values_j
        else:
            name_j, values_j = primitives[j]
            recip_j = _safe_reciprocal(values_j)
            if recip_j is None:
                continue
            name = f"({name_i})/({name_j})"
            values = values_i * recip_j
        values = _finite(values)
        if not np.all(np.isfinite(values)):
            continue
        key = np.round(values, 8).tobytes()
        if key in seen:
            continue
        seen.add(key)
        unique.append((_sanitize_name(name), values))
        if len(unique) >= max_columns:
            break
    return unique


def _format_formula(names: list[str], coefs: np.ndarray, intercept: float) -> str:
    terms: list[str] = []
    if abs(intercept) > 1e-12:
        terms.append(f"({intercept:.10g})")
    for name, coef in zip(names, coefs):
        if abs(float(coef)) < 1e-10:
            continue
        terms.append(f"({float(coef):.10g})*({name})")
    return " + ".join(terms) if terms else "0"


def _simplify_formula(formula: str) -> str:
    """Collapse algebraic noise in a short formula (``x/(x/(x^2+1))``).

    Basis column names are generated, not authored, so an exact closure can
    read as ``sqrt((x)/(x/(x^2+1.0)))``. SymPy cancels that to
    ``sqrt(x^2+1)``, which is both readable and robust to the production
    post-processor. Only short formulas are simplified: dense surrogates are
    rejected later and are not worth the CPU.
    """
    try:
        import sympy as sp

        expr = lib._symbolic(formula)
        if expr is None:
            return formula
        text = sp.sstr(sp.simplify(expr)).replace("**", "^")
        if not text:
            return formula
        return text
    except Exception:
        return formula


def fit_dictionary(
    x: np.ndarray,
    y: np.ndarray,
    columns: list[tuple[str, np.ndarray]],
    *,
    ks: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8),
    holdout: float = 0.2,
    seed: int = 0,
) -> dict[str, Any] | None:
    """Sparse OMP fit over a pre-built dictionary with a parsimony rule."""
    try:
        from sklearn.linear_model import OrthogonalMatchingPursuit
    except Exception:
        return None

    if not columns:
        return None
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    names = [name for name, _ in columns]
    B = np.stack([values for _, values in columns], axis=1)
    B_mean = B.mean(axis=0)
    y_mean = float(y.mean())
    Bc = B - B_mean
    yc = y - y_mean
    norms = np.linalg.norm(Bc, axis=0)
    norms[norms < 1e-12] = 1.0
    Bn = Bc / norms

    rng = np.random.RandomState(seed)
    n = len(y)
    idx = rng.permutation(n)
    n_val = max(10, int(round(n * holdout)))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    val_by_k: list[tuple[int, float]] = []
    for k in ks:
        if k > min(len(train_idx), B.shape[1]):
            continue
        model = OrthogonalMatchingPursuit(
            n_nonzero_coefs=int(k), fit_intercept=False
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(Bn[train_idx], yc[train_idx])
        except Exception:
            continue
        pred = model.predict(Bn[val_idx]) + y_mean
        mse = float(np.mean((pred - y[val_idx]) ** 2))
        if math.isfinite(mse):
            val_by_k.append((k, mse))

    if not val_by_k:
        return None
    best_val = min(mse for _, mse in val_by_k)
    # Parsimony: prefer the smallest dictionary that is within 10% of the best
    # validation error. Dense fits extrapolate badly and read badly.
    best_k = min(k for k, mse in val_by_k if mse <= best_val * 1.10 + 1e-12)

    model = OrthogonalMatchingPursuit(n_nonzero_coefs=int(best_k), fit_intercept=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(Bn, yc)
    coefs = model.coef_ / norms
    intercept = y_mean - float(coefs @ B_mean)
    formula = _format_formula(names, coefs, intercept)
    n_active = int(np.sum(np.abs(coefs) > 1e-10))
    if n_active <= 4:
        formula = _simplify_formula(formula)

    # Domain safety: a candidate that goes non-finite on the fitted domain is
    # not a formula, it is a singularity. Reject it here rather than letting
    # the benchmark's display guard fail later.
    check = lib.evaluate(formula, x)
    if check is None or not np.all(np.isfinite(check)):
        return None

    pred = B @ coefs + intercept
    mse = float(np.mean((pred - y) ** 2))
    return {
        "formula": formula,
        "mse": mse,
        "holdout_mse": best_val,
        "n_terms": n_active,
        "active_terms": [n for n, c in zip(names, coefs) if abs(c) > 1e-10],
        "dictionary_size": len(names),
    }


def fit_closure(
    x: np.ndarray,
    y: np.ndarray,
    predictions: dict[str, float],
    omegas: list[float] | None,
    *,
    ks: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8),
    holdout: float = 0.2,
    seed: int = 0,
) -> dict[str, Any] | None:
    """Fit a sparse model over the closed basis and return the best candidate."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    columns = closure_columns(x, y, predictions, omegas)
    return fit_dictionary(x, y, columns, ks=ks, holdout=holdout, seed=seed)
