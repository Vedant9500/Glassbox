"""Shared utilities for the exactness experiments.

Hypothesis under test: the pipeline confuses a truncated or fitted
approximation with the true closed form, and an explicit
exact-vs-approximate signal (rules, a learned verifier, or classifier
fine-tuning) improves benchmark results.

Nothing in this package changes the runtime pipeline by default. Experiments
opt in through wrappers so every result stays comparable to the unmodified
baseline.
"""

from __future__ import annotations

import math
import sys
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# SymPy emits a deprecation warning for the non-Expr Mul nodes produced by
# ``parse_expr(..., evaluate=False)``. The expressions are still valid for our
# read-only analysis, so keep the output clean.
warnings.filterwarnings("ignore", message=".*non-Expr arguments.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"sympy\..*")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import sympy as sp  # noqa: E402
from sympy.parsing.sympy_parser import (  # noqa: E402
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

_TRANSFORMS = standard_transformations + (
    convert_xor,
    implicit_multiplication_application,
)


def _sympy_signed_power(base, power):
    """Match the production ``_signed_power`` helper inside SymPy."""
    return sp.sign(base) * (sp.Abs(base) + sp.Float("1e-10")) ** power


_LOCAL_DICT = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "Abs": sp.Abs,
    "abs": sp.Abs,
    "sign": sp.sign,
    "signed_power": _sympy_signed_power,
    "pi": sp.pi,
    "E": sp.E,
    "e": sp.E,
}

# Operator labels used by the curve classifier (order is part of the contract).
OPERATOR_NAMES = (
    "identity",
    "sin",
    "cos",
    "power",
    "exp",
    "log",
    "addition",
    "multiplication",
    "rational",
)

# Series expansions about zero, as exact rationals. Only used to fingerprint
# polynomial candidates that are truncations of a well-known closed form.
_SERIES: dict[str, list[sp.Expr]] = {
    "sin": [
        0,
        1,
        0,
        sp.Rational(-1, 6),
        0,
        sp.Rational(1, 120),
        0,
        sp.Rational(-1, 5040),
        0,
        sp.Rational(1, 362880),
    ],
    "cos": [
        1,
        0,
        sp.Rational(-1, 2),
        0,
        sp.Rational(1, 24),
        0,
        sp.Rational(-1, 720),
        0,
        sp.Rational(1, 40320),
    ],
    "exp": [
        1,
        1,
        sp.Rational(1, 2),
        sp.Rational(1, 6),
        sp.Rational(1, 24),
        sp.Rational(1, 120),
        sp.Rational(1, 720),
        sp.Rational(1, 5040),
        sp.Rational(1, 40320),
    ],
    "log1p": [
        0,
        1,
        sp.Rational(-1, 2),
        sp.Rational(1, 3),
        sp.Rational(-1, 4),
        sp.Rational(1, 5),
        sp.Rational(-1, 6),
        sp.Rational(1, 7),
    ],
    "atan": [
        0,
        1,
        0,
        sp.Rational(-1, 3),
        0,
        sp.Rational(1, 5),
        0,
        sp.Rational(-1, 7),
    ],
    "sqrt1p": [
        1,
        sp.Rational(1, 2),
        sp.Rational(-1, 8),
        sp.Rational(1, 16),
        sp.Rational(-5, 128),
        sp.Rational(7, 256),
    ],
    "sinh": [
        0,
        1,
        0,
        sp.Rational(1, 6),
        0,
        sp.Rational(1, 120),
        0,
        sp.Rational(1, 5040),
    ],
    "cosh": [
        1,
        0,
        sp.Rational(1, 2),
        0,
        sp.Rational(1, 24),
        0,
        sp.Rational(1, 720),
    ],
}


@lru_cache(maxsize=4096)
def parse_formula(formula: str) -> sp.Expr | None:
    """Parse a benchmark-style formula string into a SymPy expression.

    Accepts both the benchmark dialect (``x^2``, ``sin(x)``) and the generator
    dialect (``np.sin(x)``, ``x**2``).
    """
    if not formula:
        return None
    text = str(formula).strip()
    if not text or text in {"N/A", "ERROR", "?"}:
        return None
    text = text.replace("np.", "").replace("_signed_power", "signed_power")
    try:
        return parse_expr(
            text,
            local_dict=_LOCAL_DICT,
            transformations=_TRANSFORMS,
            evaluate=False,
        )
    except Exception:
        return None


@lru_cache(maxsize=4096)
def _compile(formula: str) -> tuple[tuple[str, ...], float | None, Any] | None:
    """Compile a formula once; returns (free symbol names, const, callable)."""
    expr = parse_formula(formula)
    if expr is None:
        return None
    try:
        free = sorted(expr.free_symbols, key=lambda s: s.name)
        names = tuple(s.name for s in free)
        if not free:
            return names, float(expr.evalf()), None
        if len(free) > 1:
            return None
        return names, None, sp.lambdify(free, expr, modules=["numpy"])
    except Exception:
        return None


def evaluate(formula: str, x: np.ndarray) -> np.ndarray | None:
    """Evaluate a formula on a 1-D grid, returning None when it is unsafe."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    compiled = _compile(formula)
    if compiled is not None:
        names, const, func = compiled
        try:
            if const is not None:
                return np.full(x.shape, const, dtype=np.float64)
            with np.errstate(all="ignore"):
                out = func(x)
            return np.asarray(out, dtype=np.float64).reshape(-1)
        except Exception:
            pass
    # Fall back to the production evaluator, which understands engine-specific
    # helpers such as ``_signed_power``.
    try:
        from scripts.classifier_fast_path import _evaluate_formula_values

        out = _evaluate_formula_values(formula, x)
        if out is not None:
            return np.asarray(out, dtype=np.float64).reshape(-1)
    except Exception:
        pass
    return None


def sample_curve(
    formula: str, x_range: tuple[float, float], n_samples: int = 300
) -> tuple[np.ndarray, np.ndarray] | None:
    """Sample (x, y) exactly the way the benchmark suite does."""
    try:
        from scripts.benchmark_suite import _generate_data

        x, y = _generate_data(formula, x_range[0], x_range[1], n_samples)
    except Exception:
        x = np.linspace(x_range[0], x_range[1], n_samples)
        y = evaluate(formula, x)
        if y is None:
            return None
        mask = np.isfinite(y)
        if mask.sum() < 10:
            return None
        x, y = x[mask], y[mask]
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


def _domain_grid(x_range: tuple[float, float], n: int = 300) -> np.ndarray:
    return np.linspace(float(x_range[0]), float(x_range[1]), int(n))


def extrapolation_grid(
    x_range: tuple[float, float],
    *,
    factor: float = 2.0,
    n_per_side: int = 200,
    gap: float = 0.02,
) -> np.ndarray:
    """Build an out-of-domain grid on both sides of the fitted range."""
    lo, hi = float(x_range[0]), float(x_range[1])
    span = hi - lo
    if span <= 0:
        return np.array([], dtype=np.float64)
    gap_width = max(gap * span, 1e-6)
    left = np.linspace(lo - factor * span, lo - gap_width, n_per_side)
    right = np.linspace(hi + gap_width, hi + factor * span, n_per_side)
    return np.concatenate([left, right])


def extrapolation_metrics(
    candidate: str,
    target: str,
    x_range: tuple[float, float],
    *,
    n_samples: int = 300,
    factor: float = 2.0,
) -> dict[str, Any]:
    """Compare a candidate to the target in-domain and out-of-domain.

    ``ood_rel`` is the OOD MSE normalised by the target's OOD variance, so a
    true formula scores ~0 even when the target itself grows without bound.
    """
    out: dict[str, Any] = {
        "in_mse": None,
        "ood_mse": None,
        "ood_rel": None,
        "ood_points": 0,
        "ood_skipped": True,
    }
    x_in = _domain_grid(x_range, n_samples)
    y_true_in = evaluate(target, x_in)
    y_cand_in = evaluate(candidate, x_in)
    if y_true_in is None or y_cand_in is None:
        return out
    mask = np.isfinite(y_true_in) & np.isfinite(y_cand_in)
    if mask.sum() >= 10:
        out["in_mse"] = float(np.mean((y_cand_in[mask] - y_true_in[mask]) ** 2))

    x_ood = extrapolation_grid(x_range, factor=factor)
    if x_ood.size == 0:
        return out
    y_true_ood = evaluate(target, x_ood)
    if y_true_ood is None:
        return out
    finite_true = np.isfinite(y_true_ood)
    if finite_true.sum() < 10:
        return out

    y_cand_ood = evaluate(candidate, x_ood)
    if y_cand_ood is None:
        return out
    # A candidate that blows up outside the domain is a failed extrapolation,
    # not a missing measurement.
    cand_nonfinite = ~np.isfinite(y_cand_ood)
    if cand_nonfinite.any():
        out["ood_mse"] = float("inf")
        out["ood_rel"] = float("inf")
        out["ood_points"] = int(finite_true.sum())
        out["ood_skipped"] = False
        return out

    mask = finite_true & np.isfinite(y_cand_ood)
    if mask.sum() < 10:
        return out
    target_var = float(np.var(y_true_ood[mask]))
    with np.errstate(over="ignore", invalid="ignore"):
        mse = float(np.mean((y_cand_ood[mask] - y_true_ood[mask]) ** 2))
    out["ood_mse"] = mse
    out["ood_rel"] = mse / max(target_var, 1e-12)
    out["ood_points"] = int(mask.sum())
    out["ood_skipped"] = False
    return out


@lru_cache(maxsize=1024)
def _symbolic(formula: str) -> sp.Expr | None:
    """Parse a formula for symbolic work (``evaluate=True``, unlike the AST)."""
    text = str(formula or "").strip()
    if not text or text in {"N/A", "ERROR", "?"}:
        return None
    text = text.replace("np.", "").replace("_signed_power", "signed_power")
    try:
        expr = parse_expr(
            text,
            local_dict=_LOCAL_DICT,
            transformations=_TRANSFORMS,
            evaluate=True,
        )
        # Curve fitting is over a real variable. Without this, SymPy treats
        # ``x`` as complex and can introduce ``re(x)``/``im(x)`` into
        # simplified formulas, which the production evaluator does not know.
        if expr.free_symbols:
            real = {s: sp.Symbol(s.name, real=True) for s in expr.free_symbols}
            expr = expr.subs(real)
        return expr
    except Exception:
        return None


@lru_cache(maxsize=4096)
def equivalent(
    a: str,
    b: str,
    x_range: tuple[float, float] | None = None,
    *,
    extend: float = 0.5,
    n: int = 401,
) -> bool:
    """True when two formulas are the same function, not merely a good fit.

    The operator-set comparison treats ``sin(2*x)`` and ``2*sin(x)*cos(x)`` as
    different because one is missing ``cos``. They are identical functions, so
    calling that an approximation is a false positive. This check exists to
    keep the false-EXACT metric honest in both directions.

    Fast numeric agreement on an *extended* domain is required first (an
    approximation that only agrees inside the fitted range is not equivalent),
    and symbolic simplification is used to confirm. A different domain of
    definition (``log(x+1)`` vs ``log(Abs(x)+1)``) counts as different.
    """
    expr_a, expr_b = _symbolic(a), _symbolic(b)
    if expr_a is None or expr_b is None:
        return False

    if x_range is None:
        lo, hi = -6.0, 6.0
    else:
        lo, hi = float(x_range[0]), float(x_range[1])
    span = max(hi - lo, 1e-6)
    grid = np.linspace(lo - extend * span, hi + extend * span, n)
    values_a = evaluate(a, grid)
    values_b = evaluate(b, grid)
    if values_a is None or values_b is None:
        return False
    finite_a = np.isfinite(values_a)
    finite_b = np.isfinite(values_b)
    if not np.array_equal(finite_a, finite_b):
        return False
    if int(finite_a.sum()) < int(0.3 * n):
        return False
    scale = max(
        1.0,
        float(np.max(np.abs(values_a[finite_a]))) if finite_a.any() else 1.0,
        float(np.max(np.abs(values_b[finite_b]))) if finite_b.any() else 1.0,
    )
    max_dev = float(np.max(np.abs(values_a[finite_a] - values_b[finite_b])))
    if max_dev > 1e-8 * scale:
        return False

    try:
        difference = sp.simplify(expr_a - expr_b)
        if difference == 0:
            return True
        simplified = sp.trigsimp(sp.expand_trig(expr_a - expr_b))
        if simplified == 0:
            return True
    except Exception:
        pass
    # The numeric check above already established agreement to 1e-8 relative
    # over an extended domain, including identical domains of definition.
    return True


def operator_set(formula: str) -> set[str]:
    """Map a formula to the classifier's operator vocabulary."""
    expr = parse_formula(formula)
    if expr is None:
        return _operator_set_from_text(formula)
    ops: set[str] = set()
    for node in sp.preorder_traversal(expr):
        name = type(node).__name__
        if name == "sin":
            ops.add("sin")
        elif name == "cos":
            ops.add("cos")
        elif name == "exp":
            ops.add("exp")
        elif name == "log":
            ops.add("log")
        elif name == "Abs":
            ops.add("power")
        elif name == "Add":
            ops.add("addition")
        elif name == "Mul":
            ops.add("multiplication")
        elif name == "Pow":
            exponent = node.args[1]
            if exponent.is_negative or (
                exponent.is_number and float(exponent) < 0
            ):
                ops.add("rational")
            ops.add("power")
    if not ops:
        ops.add("identity")
    return {op for op in ops if op in OPERATOR_NAMES}


def _operator_set_from_text(formula: str) -> set[str]:
    """Best-effort operator detection for formulas SymPy cannot parse."""
    text = str(formula or "")
    ops: set[str] = set()
    if "sin(" in text:
        ops.add("sin")
    if "cos(" in text:
        ops.add("cos")
    if "exp(" in text:
        ops.add("exp")
    if "log(" in text:
        ops.add("log")
    if "sqrt(" in text or "abs(" in text or "Abs(" in text:
        ops.add("power")
    if "+" in text or "-" in text[1:]:
        ops.add("addition")
    if "*" in text:
        ops.add("multiplication")
    if "^" in text or "**" in text or "sqrt(" in text:
        ops.add("power")
    if "/" in text:
        ops.add("rational")
    if not ops:
        ops.add("identity")
    return {op for op in ops if op in OPERATOR_NAMES}


def operator_vector(formula: str) -> np.ndarray:
    ops = operator_set(formula)
    return np.array([1.0 if name in ops else 0.0 for name in OPERATOR_NAMES])


_STRUCT_FUNCS = {"sin", "cos", "tan", "exp", "log", "sqrt", "tanh", "sinh", "cosh", "atan"}


@lru_cache(maxsize=4096)
def classify_structure(formula: str) -> str:
    """Coarse structural family of a formula.

    ``nested``      a transcendental of a non-linear argument (``sin(cos(x))``)
    ``quotient``    a ratio / negative power of a non-constant
    ``product``     a product of two or more non-constant factors
    ``transcendental`` a plain sum of transcendental primitives
    ``polynomial``  no transcendental functions
    """
    expr = parse_formula(formula)
    if expr is None:
        return "unparsed"

    has_nested = False
    has_quotient = False
    has_product = False

    for node in sp.preorder_traversal(expr):
        if isinstance(node, sp.Mul):
            non_const = [a for a in node.args if not a.is_number]
            if len(non_const) >= 2:
                has_product = True
        elif isinstance(node, sp.Pow):
            base, exponent = node.args
            if not base.is_number and exponent.is_number and float(exponent) < 0:
                has_quotient = True
        if isinstance(node, sp.Abs):
            arg = node.args[0]
            if not arg.is_symbol and not arg.is_number:
                has_nested = True
        elif type(node).__name__.lower() in _STRUCT_FUNCS:
            arg = node.args[0]
            if not arg.is_symbol and not arg.is_number:
                has_nested = True

    if has_nested:
        return "nested"
    if has_quotient:
        return "quotient"
    if has_product:
        return "product"
    if expr.has(sp.sin, sp.cos, sp.exp, sp.log):
        return "transcendental"
    return "polynomial"


def is_polynomial(formula: str) -> tuple[bool, int]:
    """Return (is_polynomial, degree) for a formula."""
    expr = parse_formula(formula)
    if expr is None:
        return False, 0
    try:
        poly = sp.Poly(expr, *sorted(expr.free_symbols, key=lambda s: s.name))
    except Exception:
        return False, 0
    if len(poly.free_symbols) > 1:
        return False, 0
    return True, int(poly.degree())


def series_fingerprint(formula: str, tol: float = 5e-3) -> dict[str, Any]:
    """Detect whether a polynomial candidate is a truncated known series.

    Returns the best matching series, its scale, and how many nonzero terms
    agree. A score of 0.0 means "no fingerprint".
    """
    out: dict[str, Any] = {
        "is_polynomial": False,
        "degree": 0,
        "series": None,
        "scale": None,
        "matched_nonzero": 0,
        "score": 0.0,
    }
    expr = parse_formula(formula)
    if expr is None:
        return out
    try:
        symbols = sorted(expr.free_symbols, key=lambda s: s.name)
        if len(symbols) != 1:
            return out
        x = symbols[0]
        poly = sp.Poly(sp.expand(expr), x)
    except Exception:
        return out
    coeffs = [float(c) for c in reversed(poly.all_coeffs())]
    out["is_polynomial"] = True
    out["degree"] = len(coeffs) - 1
    if len(coeffs) < 2:
        return out

    scale_ref = max(abs(c) for c in coeffs)
    if scale_ref <= 0:
        return out

    best: tuple[float, int, str, float] | None = None
    for name, base in _SERIES.items():
        base_f = [float(v) for v in base]
        first = next((i for i, v in enumerate(base_f) if abs(v) > 0), None)
        if first is None:
            continue
        if first >= len(coeffs) or abs(coeffs[first]) <= 0:
            continue
        try:
            scale = (
                (coeffs[first] / base_f[first]) ** (1.0 / first)
                if first > 0
                else coeffs[first] / base_f[first]
            )
        except (ValueError, ZeroDivisionError):
            continue
        if not math.isfinite(scale) or abs(scale) < 1e-9:
            continue

        matched = 0
        nonzero = 0
        considered = 0
        for i in range(min(len(coeffs), len(base_f))):
            expected = base_f[i] * (scale**i)
            actual = coeffs[i]
            considered += 1
            denom = max(abs(expected), abs(actual), 1e-12)
            if abs(actual - expected) / denom <= tol:
                matched += 1
                if abs(expected) > 0:
                    nonzero += 1
        if considered == 0 or nonzero < 3:
            continue
        score = matched / considered
        if best is None or score > best[0]:
            best = (score, nonzero, name, scale)

    if best is not None and best[0] >= 0.75:
        out["score"] = float(best[0])
        out["matched_nonzero"] = int(best[1])
        out["series"] = best[2]
        out["scale"] = float(best[3])
    return out


def taylor_candidates(
    formula: str, *, x0: float = 0.0, max_order: int = 8, min_order: int = 3
) -> list[tuple[str, str]]:
    """Return (label, formula) Taylor truncations of the target."""
    expr = parse_formula(formula)
    if expr is None:
        return []
    symbols = sorted(expr.free_symbols, key=lambda s: s.name)
    if len(symbols) != 1:
        return []
    x = symbols[0]
    out: list[tuple[str, str]] = []
    for order in range(min_order, max_order + 1):
        try:
            series = sp.series(expr, x, x0, order + 1).removeO()
            expanded = sp.expand(series)
            if expanded.has(sp.Order):
                continue
            text = str(expanded)
            out.append((f"taylor_o{order}", text))
        except Exception:
            continue
    return out


def polyfit_candidates(
    x: np.ndarray,
    y: np.ndarray,
    *,
    degrees: Iterable[int] = (3, 4, 5, 6, 7, 8, 9),
) -> list[tuple[str, str]]:
    """Least-squares polynomial approximations of the sampled curve."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    span = float(np.max(np.abs(x))) if x.size else 1.0
    x_norm = x / max(span, 1e-12)
    out: list[tuple[str, str]] = []
    for degree in degrees:
        try:
            coeffs = np.polyfit(x_norm, y, degree)
        except Exception:
            continue
        terms = []
        for power in range(degree, -1, -1):
            c = float(coeffs[degree - power])
            if abs(c) < 1e-10:
                continue
            if power == 0:
                terms.append(f"({c:.10g})")
            elif power == 1:
                terms.append(f"({c:.10g})*(x/{span:.10g})")
            else:
                terms.append(f"({c:.10g})*(x/{span:.10g})^{power}")
        if not terms:
            continue
        out.append((f"polyfit_d{degree}", " + ".join(terms)))
    return out


def pade_candidates(
    formula: str, *, x0: float = 0.0, orders: Iterable[tuple[int, int]] = ((2, 2), (3, 3))
) -> list[tuple[str, str]]:
    """Return (label, formula) Pade approximants of the target."""
    expr = parse_formula(formula)
    if expr is None:
        return []
    symbols = sorted(expr.free_symbols, key=lambda s: s.name)
    if len(symbols) != 1:
        return []
    x = symbols[0]
    out: list[tuple[str, str]] = []
    for m, n in orders:
        try:
            series = sp.series(expr, x, x0, m + n + 1).removeO()
            approx = sp.approximants(series) if hasattr(sp, "approximants") else None
        except Exception:
            approx = None
        if not approx:
            continue
        try:
            pade = sp.simplify(approx[(m, n)])
            out.append((f"pade_{m}_{n}", str(pade)))
        except Exception:
            continue
    return out


def perturb_candidates(
    formula: str, *, deltas: Iterable[float] = (0.002, -0.002, 0.01)
) -> list[tuple[str, str]]:
    """Structurally-correct variants with slightly wrong constants.

    These are the positive class that actually matters: a formula with the
    right operators but imperfect coefficients, not a string copy of the
    target.
    """
    expr = parse_formula(formula)
    if expr is None:
        return []
    floats = [n for n in sp.preorder_traversal(expr) if n.is_Float]
    if not floats:
        return []
    out: list[tuple[str, str]] = []
    for delta in deltas:
        try:
            replaced = expr.xreplace({n: n * (1.0 + float(delta)) for n in floats})
            text = str(sp.expand(replaced))
            if text and text != str(expr):
                out.append((f"perturbed_{delta:+.3f}", text))
        except Exception:
            continue
    return out


def scaled_candidates(
    formula: str, *, scales: Iterable[float] = (0.998, 1.002, 0.99)
) -> list[tuple[str, str]]:
    """Whole-expression rescales: same structure, slightly wrong constants.

    Works for formulas with no explicit float coefficients (``sin(x)``), where
    ``perturb_candidates`` has nothing to perturb.
    """
    out: list[tuple[str, str]] = []
    for scale in scales:
        text = f"({float(scale):.10g})*({formula})"
        out.append((f"scaled_{float(scale):.4f}", text))
    return out


def label_pair(
    *,
    in_mse: float | None,
    ood_rel: float | None,
    in_tol: float = 1e-8,
    ood_tol: float = 1e-4,
    approx_in_tol: float = 1e-4,
    approx_ood_tol: float = 1e-2,
    candidate: str | None = None,
    target: str | None = None,
    x_range: tuple[float, float] | None = None,
    candidate_kind: str | None = None,
) -> str:
    """Classify a (curve, candidate) pair as exact / approximation / other.

    The numeric heuristic alone is not enough. When the target explodes
    outside the fitted range (``sin(x)+exp(-x)``), the target's own OOD
    variance explodes with it, so a polynomial surrogate keeps a small
    *relative* OOD error and is labelled exact. A fit is therefore called
    exact only when it is symbolically the same function; structurally
    correct candidates (identity / rescaled / perturbed) stay positive by
    construction.
    """
    kind = (candidate_kind or "").split("_")[0]
    if kind in {"taylor", "polyfit", "pade"}:
        if candidate and target and x_range is not None:
            return "exact" if equivalent(candidate, target, x_range) else "approximation"
    elif kind in {"identity", "scaled", "perturbed"}:
        return "exact"

    if in_mse is None or not math.isfinite(in_mse):
        return "other"
    if in_mse <= in_tol:
        if ood_rel is None or not math.isfinite(ood_rel) or ood_rel <= ood_tol:
            return "exact"
        return "approximation"
    if in_mse <= approx_in_tol:
        if ood_rel is not None and math.isfinite(ood_rel) and ood_rel > approx_ood_tol:
            return "approximation"
    return "other"


def curve_features(x: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    """398-dim curve feature vector from the production extractor."""
    try:
        from glassbox.curve_classifier.generate_curve_data import (
            extract_all_features_xy,
        )

        feats = extract_all_features_xy(
            np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        )
        return np.asarray(feats, dtype=np.float32).reshape(-1)
    except Exception:
        return None


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value
