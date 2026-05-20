import sys
import numpy as np
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Ensure C++ build dir is in sys.path
cpp_dir = REPO_ROOT / "glassbox" / "sr" / "cpp"
if str(cpp_dir) not in sys.path:
    sys.path.insert(0, str(cpp_dir))

try:
    import _core
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False

requires_cpp = pytest.mark.skipif(
    not CPP_AVAILABLE,
    reason="C++ _core extension not built.",
)

@requires_cpp
def test_cpp_basic_simplification():
    """Test that basic arithmetic and algebraic simplifications work in C++."""
    # Constant folding
    res = _core.simplify_formula("2 + 3")
    assert float(res) == 5.0

    # Variable and arithmetic snapping/simplification
    res = _core.simplify_formula("x + 0")
    assert res == "x"

    res = _core.simplify_formula("0 + x")
    assert res == "x"

    res = _core.simplify_formula("x - 0")
    assert res == "x"

    res = _core.simplify_formula("x * 1")
    assert res == "x"

    res = _core.simplify_formula("x * 0")
    assert float(res) == 0.0

@requires_cpp
def test_cpp_pythagorean_identity():
    """sin(x)^2 + cos(x)^2 should collapse to 1 in C++."""
    # Since our formula_parser handles sin(x)^2 + cos(x)^2, we test it here:
    res = _core.simplify_formula("sin(x)^2 + cos(x)^2")
    assert float(res) == 1.0

@requires_cpp
def test_cpp_snapping():
    """Near-integer coefficients and values should snap."""
    res = _core.simplify_formula("0.99999999*x + 1.00000001*x")
    # should simplify to 2*x
    # Depending on printing style, it will be 2.0*x or 2*x or 2.000000 * x.
    # Let's clean spaces and check.
    assert "2" in res
    assert "x" in res

@requires_cpp
def test_cpp_unary_minus_precedence():
    """Unary minus must bind looser than power: -x^2 means -(x^2)."""
    x = np.linspace(-3.0, 3.0, 101)
    expr = _core.simplify_formula("exp(-x^2)")
    py_expr = expr.replace("^", "**")
    y_pred = eval(
        py_expr,
        {"__builtins__": None},
        {"x": x, "exp": np.exp, "abs": np.abs, "sign": np.sign},
    )
    y_true = np.exp(-(x ** 2))
    assert np.mean((y_pred - y_true) ** 2) < 1e-12

@requires_cpp
def test_cpp_noise_reduction():
    """BIC noise reduction should prune unnecessary terms."""
    np.random.seed(42)
    # Generate data: y = 2.0 * x0 + 1.5 * x1
    X = np.random.uniform(-3, 3, size=(100, 2))
    y = 2.0 * X[:, 0] + 1.5 * X[:, 1] + np.random.normal(0, 0.01, size=100)

    # Candidate formula with a noise term: 2.0*x0 + 1.5*x1 + 0.1*sin(x0)
    formula = "2.0*x0 + 1.5*x1 + 0.1*sin(x0)"
    X_list = [X[:, 0], X[:, 1]]

    # C++ Noise reduction
    reduced = _core.reduce_formula_noise(formula, X_list, y)
    
    # The noise term 0.1*sin(x0) should be dropped
    assert "sin" not in reduced
    assert "x0" in reduced
    assert "x1" in reduced
