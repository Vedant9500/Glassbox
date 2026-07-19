import sys
import numpy as np
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
def test_cpp_public_api_exports():
    assert hasattr(_core, "simplify_formula_cpp")
    assert hasattr(_core, "reduce_formula_noise_cpp")
    assert hasattr(_core, "formula_to_seed_graph_cpp")
    assert hasattr(_core, "snap_formula_floats_cpp")


@requires_cpp
def test_cpp_basic_simplification():
    assert _core.simplify_formula_cpp("2 + 3") == "5"
    assert _core.simplify_formula_cpp("x + 0") == "x"
    assert _core.simplify_formula_cpp("x * 1") == "x"
    assert _core.simplify_formula_cpp("x * 0") == "0"


@requires_cpp
def test_cpp_pythagorean_identity():
    assert _core.simplify_formula_cpp("sin(x)^2 + cos(x)^2") == "1"


@requires_cpp
def test_cpp_float_snapping():
    out = _core.simplify_formula_cpp("0.99999999*x + 1.00000001*x")
    assert "2" in out
    assert "x" in out


@requires_cpp
def test_cpp_unary_minus_precedence():
    x = np.linspace(-3.0, 3.0, 101)
    expr = _core.simplify_formula_cpp("exp(-x^2)")
    y_pred = eval(expr.replace("^", "**"), {"__builtins__": None}, {"x": x, "exp": np.exp, "abs": np.abs, "sign": np.sign})
    y_true = np.exp(-(x ** 2))
    assert np.mean((y_pred - y_true) ** 2) < 1e-12


@requires_cpp
def test_cpp_noise_reduction():
    np.random.seed(42)
    X = np.random.uniform(-3, 3, size=(100, 2))
    y = 2.0 * X[:, 0] + 1.5 * X[:, 1] + np.random.normal(0, 0.01, size=100)
    reduced = _core.reduce_formula_noise_cpp("2.0*x0 + 1.5*x1 + 0.1*sin(x0)", [X[:, 0], X[:, 1]], y)
    assert "sin" not in reduced
    assert "x0" in reduced
    assert "x1" in reduced


@requires_cpp
def test_cpp_periodic_display_canonicalization():
    assert _core.simplify_formula_cpp("sin(pi*x)").replace(" ", "") == "sin(pi*x)"
    assert _core.simplify_formula_cpp("cos(pi*x)").replace(" ", "") == "cos(pi*x)"


@requires_cpp
def test_cpp_abs_is_not_identity():
    assert _core.simplify_formula_cpp("abs(x)") == "abs(x)"
    assert "abs" in _core.simplify_formula_cpp("1+abs(x)")
    g = _core.formula_to_seed_graph_cpp("abs(x)")
    # UnaryOp::Abs == 5
    assert g["nodes"][-1]["unary_op"] == 5
    assert g["nodes"][-1]["type"] == 2


@requires_cpp
def test_cpp_abs_scores_against_absolute_target():
    x = np.linspace(-2.0, 2.0, 101)
    X = x.reshape(-1, 1)
    y = np.abs(x)
    score = dict(_core.score_formula_candidates(
        ["abs(x0)"],
        np.ascontiguousarray(X, dtype=np.float64),
        np.ascontiguousarray(y, dtype=np.float64),
        np.ascontiguousarray(X, dtype=np.float64),
        np.ascontiguousarray(y, dtype=np.float64),
    )[0])
    assert score["ok"] is True
    assert score["mse"] < 1e-12
    assert abs(score["scale"] - 1.0) < 1e-6


@requires_cpp
def test_cpp_variable_power_and_printer_fidelity():
    # Constant-folded fractional exponent (1/2) becomes Unary Power p=0.5.
    g = _core.formula_to_seed_graph_cpp("x0^(1/2)")
    assert g["nodes"][-1]["unary_op"] == 1  # Power
    assert abs(g["nodes"][-1]["p"] - 0.5) < 1e-12

    # Variable exponent is not identity; multi-feature names preserved.
    g2 = _core.formula_to_seed_graph_cpp("x0^x1")
    assert len(g2["nodes"]) >= 4
    s = _core.simplify_formula_cpp("x0^x1")
    assert "x0" in s and "x1" in s
    assert "exp" in s and "log" in s

    x0 = np.linspace(0.2, 2.0, 60)
    x1 = np.linspace(0.5, 1.5, 60)
    X = np.column_stack([x0, x1])
    y = np.sign(x0) * np.abs(x0) ** x1
    score = dict(_core.score_formula_candidates(
        ["x0^x1"],
        np.ascontiguousarray(X, dtype=np.float64),
        np.ascontiguousarray(y, dtype=np.float64),
        np.ascontiguousarray(X, dtype=np.float64),
        np.ascontiguousarray(y, dtype=np.float64),
    )[0])
    assert score["ok"] is True
    assert score["mse"] < 1e-10


@requires_cpp
def test_cpp_protected_division_print_matches_eval():
    # Printer must not collapse x0/x1 to bare x/x, and must use protected form.
    out = _core.simplify_formula_cpp("x0/x1")
    assert "x0" in out and "x1" in out
    assert "abs" in out and "sign" in out
    x0 = np.linspace(-2.0, 2.0, 80)
    x0[np.abs(x0) < 0.05] = 0.05
    x1 = np.linspace(0.5, 1.5, 80)
    X = np.column_stack([x0, x1])
    # Graph Division semantics ≈ x0/x1 away from zero.
    y = x0 / x1
    score = dict(_core.score_formula_candidates(
        ["x0/x1"],
        np.ascontiguousarray(X, dtype=np.float64),
        np.ascontiguousarray(y, dtype=np.float64),
        np.ascontiguousarray(X, dtype=np.float64),
        np.ascontiguousarray(y, dtype=np.float64),
    )[0])
    assert score["ok"] is True
    assert score["mse"] < 1e-8
