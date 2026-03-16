"""Unit tests for multi-pass symbolic formula simplification."""

import sys
from pathlib import Path

import sympy as sp

# Allow direct execution: python tests/test_simplify_formula.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.simplify_formula import simplify_onn_formula


def _equivalent(expr_a: sp.Expr, expr_b: sp.Expr) -> bool:
    """Check symbolic equivalence through exact simplification."""
    return sp.simplify(expr_a - expr_b) == 0


def test_pythagorean_trig_identity():
    """sin^2(x) + cos^2(x) should collapse to 1."""
    _, simplified = simplify_onn_formula("sin(x)^2 + cos(x)^2")
    assert _equivalent(simplified, sp.Integer(1))


def test_angle_addition_identity():
    """sin(x)cos(y) + cos(x)sin(y) should collapse to sin(x+y)."""
    _, simplified = simplify_onn_formula("sin(x)*cos(y) + cos(x)*sin(y)")
    x, y = sp.symbols("x y")
    assert _equivalent(simplified, sp.sin(x + y))


def test_double_angle_identity():
    """2*sin(x)*cos(x) should collapse to sin(2*x)."""
    _, simplified = simplify_onn_formula("2*sin(x)*cos(x)")
    x = sp.Symbol("x")
    assert _equivalent(simplified, sp.sin(2 * x))


def test_float_snapping_then_simplification():
    """Near-integer coefficients should snap and simplify deterministically."""
    _, simplified = simplify_onn_formula("0.999999999*x + 1.000000001*x + 0.0000000001*y")
    x = sp.Symbol("x")
    assert _equivalent(simplified, 2 * x)


def test_approximate_trig_collapse_is_opt_in():
    """Dominant Fourier cleanup should only run when explicitly enabled."""
    formula = "sin(x) + 0.02*sin(3*x)"
    x = sp.Symbol("x")

    _, exact_only = simplify_onn_formula(formula, approximate_trig=False)
    assert not _equivalent(exact_only, sp.sin(x))

    _, approx = simplify_onn_formula(
        formula,
        approximate_trig=True,
        dominant_trig_ratio=0.95,
        small_term_ratio=0.05,
    )
    assert _equivalent(approx, sp.sin(x))
