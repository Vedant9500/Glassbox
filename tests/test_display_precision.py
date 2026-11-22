"""§3.115/§3.116: displayed formulas keep precision and only snap exact pi."""

import numpy as np
import pytest

from glassbox.sr.cpp import CPP_AVAILABLE, get_cpp_core

pytestmark = pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ _core not built")


def _formula_value(formula, x):
    x = np.asarray(x, dtype=np.float64)
    context = {
        "np": np,
        "sin": np.sin,
        "cos": np.cos,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "abs": np.abs,
        "sign": np.sign,
        "pi": np.pi,
        "x": x,
        "x0": x,
    }
    expr = formula.replace("^", "**")
    return np.asarray(eval(expr, {"__builtins__": None}, context), dtype=np.float64)


def test_display_keeps_constant_precision():
    core = get_cpp_core()
    shown = core.simplify_formula("0.123456789*x")
    x = np.linspace(-2.0, 2.0, 101)
    expected = 0.123456789 * x
    got = _formula_value(shown, x)
    # The old %.4g formatter emitted 0.1235*x (relative error ~3e-4).
    assert np.max(np.abs(got - expected)) < 1e-9
    assert "0.1235" not in shown


def test_fitted_pi_like_constant_is_not_snapped():
    core = get_cpp_core()
    shown = core.simplify_formula("3.14159*x")
    assert "pi" not in shown
    x = np.linspace(-2.0, 2.0, 101)
    assert np.max(np.abs(_formula_value(shown, x) - 3.14159 * x)) < 1e-9


def test_exact_pi_is_still_rendered_symbolically():
    core = get_cpp_core()
    shown = core.simplify_formula("sin(pi*x)").replace(" ", "")
    assert shown == "sin(pi*x)"


def test_exact_eval_uses_canonical_power_contract():
    core = get_cpp_core()
    x = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])

    # Near-integer even exponent uses the abs base (no sign flip).
    even = np.asarray(core.eval_formula_exact("x^2.0000000005", [x]))
    assert np.all(even >= 0.0)
    np.testing.assert_allclose(
        even, (np.abs(x) + 1e-10) ** 2.0000000005, rtol=1e-8
    )

    # Odd integer exponent keeps the sign.
    odd = np.asarray(core.eval_formula_exact("x^3", [x]))
    np.testing.assert_allclose(odd, x**3)

    # Fractional exponent is sign-preserving, matching the graph path.
    frac = np.asarray(core.eval_formula_exact("x^0.5", [x]))
    expected = np.sign(x) * (np.abs(x) + 1e-10) ** 0.5
    np.testing.assert_allclose(frac, expected, rtol=1e-8, atol=1e-12)
