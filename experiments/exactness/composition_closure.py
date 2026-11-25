"""Unary composition closure over the pipeline's own basis.

The product closure in ``composite_basis.py`` cannot represent
``sin(cos(x))`` or ``exp(sin(x))``: those need a function applied to another
function, not a product of two columns.

This module closes the basis under a **uniform** set of unary operators: for
every column ``g`` the pipeline already builds (primitives, products,
quotients, reciprocals), it adds ``sin(g)``, ``cos(g)``, ``exp(g)``, ``log(g)``,
``sqrt(g)``, ``tanh(g)``, ``atan(g)`` and ``abs(g)`` when the domain allows it.

There is no named formula list and no per-family special case. ``sin(cos(x))``
is generated because ``cos(x)`` is a basis column and ``sin`` is one of the
operators applied to every column. The same rule produces thousands of
candidates that no benchmark contains; they are ranked by correlation and
discarded.

Acceptance uses the same evidence as the product closure: machine-precision
fit, no worse than the incumbent at the domain edges.

Usage (from run_with_guard.py):
    --guard compose
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from experiments.exactness.composite_basis import (
    _correlation_score,
    _finite,
    _sanitize_name,
    _usable_primitives,
    closure_columns,
    fit_dictionary,
    primitive_basis,
)

MAX_COLUMNS = 320


def _clip_exp(values: np.ndarray) -> np.ndarray:
    return np.exp(np.clip(values, -30.0, 30.0))


# (function, domain guard). The guard is checked on the fitted domain only;
# candidates that misbehave outside are caught by the acceptance tests.
UNARY_OPS: dict[str, tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], bool]]] = {
    "sin": (np.sin, lambda v: True),
    "cos": (np.cos, lambda v: True),
    "exp": (_clip_exp, lambda v: True),
    "log": (np.log, lambda v: bool(np.all(v > 0.0))),
    "sqrt": (np.sqrt, lambda v: bool(np.all(v >= 0.0))),
    "tanh": (np.tanh, lambda v: True),
    "atan": (np.arctan, lambda v: True),
    "abs": (np.abs, lambda v: True),
}


def composition_columns(
    x: np.ndarray,
    y: np.ndarray,
    predictions: dict[str, float],
    omegas: list[float] | None,
    *,
    max_columns: int = MAX_COLUMNS,
) -> list[tuple[str, np.ndarray]]:
    """Return the product-closed basis plus every domain-safe ``op(g)``."""
    basis, names = primitive_basis(x, predictions, omegas)
    primitives = _usable_primitives(basis, names)
    closed = closure_columns(x, y, predictions, omegas)
    bases = primitives + closed

    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    y_centered = y_arr - y_arr.mean()
    y_norm = float(np.linalg.norm(y_centered))
    if y_norm < 1e-12:
        return closed

    scored: list[tuple[float, str, np.ndarray]] = []
    for name, values in bases:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.size != y_arr.size:
            continue
        for op_name, (func, guard) in UNARY_OPS.items():
            try:
                if not guard(values):
                    continue
                composed = func(values)
            except Exception:
                continue
            composed = _finite(composed)
            if not np.all(np.isfinite(composed)):
                continue
            if float(np.linalg.norm(composed - composed.mean())) < 1e-9:
                continue
            score = _correlation_score(composed, y_centered, y_norm)
            if score <= 0.0:
                continue
            scored.append((score, f"{op_name}({name})", composed))

    scored.sort(key=lambda row: row[0], reverse=True)

    unique: list[tuple[str, np.ndarray]] = []
    seen: set[bytes] = set()
    for _score, name, values in scored:
        key = np.round(values, 8).tobytes()
        if key in seen:
            continue
        seen.add(key)
        unique.append((_sanitize_name(name), values))
        if len(unique) >= max_columns:
            break

    return closed + unique


def fit_composition(
    x: np.ndarray,
    y: np.ndarray,
    predictions: dict[str, float],
    omegas: list[float] | None,
    *,
    ks: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8),
    holdout: float = 0.2,
    seed: int = 0,
) -> dict[str, Any] | None:
    """Sparse fit over products plus unary compositions."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    columns = composition_columns(x, y, predictions, omegas)
    return fit_dictionary(x, y, columns, ks=ks, holdout=holdout, seed=seed)
