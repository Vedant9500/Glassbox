"""R-02 regression tests: restricted-``eval`` expression-injection surface.

The audit (§2.4 R-02, §3.3 R-12) flagged that formula strings are evaluated
with ``eval(expr, {"__builtins__": None}, context)`` across the estimator,
universal proposer, specialist composition, benchmark and curve-data paths.
Restricting builtins still permits attribute-traversal / subscript escapes
(e.g. ``x.__class__.__mro__[1].__subclasses__()``).  The fix gates every
formula ``eval`` with an AST allowlist (``glassbox/sr/formula_safety.py``).
"""
import numpy as np
import pytest

from glassbox.sr.formula_safety import (
    formula_expression_is_safe,
    validate_formula_expr,
)
from glassbox.sr.sklearn_wrapper import GlassboxRegressor


_ALLOWED = {
    "np", "log", "sin", "cos", "exp", "sqrt", "abs", "Abs", "sign",
    "_signed_power", "pi", "E", "e", "x", "x0", "x1", "x2",
}

# ---------------------------------------------------------------------------
# Shared AST-allowlist gate
# ---------------------------------------------------------------------------
def test_allowlist_accepts_legit_formulas():
    # Expressions mirror the C++ formatter output (eval.h format_node_to_string).
    legit = [
        "x",
        "x0 + x1",
        "sin(x0)",
        "(x0 * exp(x1))",
        "log(abs(x0)+1e-6)",
        "sign(x0)*(abs(x0))**0.5",
        "0.5*(x0 + x1 + abs(x0 - x1))",
        "((x0) * sign(x1) / (abs(x1) + 1e-6))",
        "x**2",
        "1.5*sin(x + pi)",
        "-x",
        "2*pi*x",
        "np.sin(x0)",
        "np.e**x",
        "_signed_power(x0, 0.5)",
        "(x0**2)",
        "exp((-(1.0)) * ((x1)**2))",
    ]
    for expr in legit:
        assert formula_expression_is_safe(expr, _ALLOWED), expr


def test_allowlist_rejects_attribute_traversal():
    escapes = [
        "x.__class__",
        "x.__class__.__mro__[1].__subclasses__()",
        "().__class__.__base__.__subclasses__()",
        "x.__init__.__globals__",
        "x.__getattribute__('__class__')",
        "np.__class__",
        "np.__class__.__mro__",
        "np.sin.__globals__",
        'f"{x.__class__}"',
    ]
    for expr in escapes:
        assert not formula_expression_is_safe(expr, _ALLOWED), expr


def test_allowlist_rejects_subscript_comprehension_lambda_import():
    rejects = [
        "x[0]",
        "[i for i in x]",
        "lambda: 1",
        "x if True else 0",
        "__import__('os')",
        "getattr(x, 'shape')",
        "x < 0",
        "x and y",
        "not x",
        "(x := 1)",
        "x.reshape(1)",
        "{'a': 1}",
        "sin(*x)",
    ]
    for expr in rejects:
        assert not formula_expression_is_safe(expr, _ALLOWED), expr


def test_allowlist_rejects_unknown_names():
    # Any free name the caller does not expose in its context is rejected.
    assert not formula_expression_is_safe("evil(x)", _ALLOWED)
    assert not formula_expression_is_safe("os.system('rm -rf /')", _ALLOWED)
    assert formula_expression_is_safe("sin(x0)", _ALLOWED)


def test_validate_formula_expr_raises_on_unsafe():
    validate_formula_expr("sin(x)", _ALLOWED)  # does not raise
    with pytest.raises(ValueError):
        validate_formula_expr("x.__class__", _ALLOWED)


# ---------------------------------------------------------------------------
# sklearn_wrapper production path
# ---------------------------------------------------------------------------
def test_safe_eval_formula_array_blocks_injection():
    est = GlassboxRegressor(random_state=0)
    X = np.linspace(-2.0, 2.0, 32).reshape(-1, 1)
    with pytest.raises(ValueError):
        est._safe_eval_formula_array("x.__class__.__mro__[1].__subclasses__()", X)


def test_safe_eval_formula_array_still_evaluates_legit():
    est = GlassboxRegressor(random_state=0)
    X = np.linspace(-2.0, 2.0, 32).reshape(-1, 1)
    y = est._safe_eval_formula_array("sin(x)", X)
    assert y.shape == (32,)
    assert np.all(np.isfinite(y))
    y2 = est._safe_eval_formula_array("sign(x)*(abs(x))**0.5", X)
    assert y2.shape == (32,)


def test_domain_failure_rate_fails_closed_on_injection():
    est = GlassboxRegressor(random_state=0)
    X = np.linspace(-2.0, 2.0, 32).reshape(-1, 1)
    assert est._formula_domain_failure_rate("x.__class__", X) == 1.0


def test_predict_returns_zeros_for_injected_formula():
    est = GlassboxRegressor(random_state=0)
    est.formula_ = "x.__class__.__mro__[1].__subclasses__()"
    X = np.linspace(-2.0, 2.0, 8).reshape(-1, 1)
    out = est.predict(X)
    assert out.shape == (8,)
    assert np.all(out == 0.0)
    # Failure was surfaced through the R-01 diagnostics counter.
    assert est.swallowed_errors_["predict.eval"]["count"] >= 1


# ---------------------------------------------------------------------------
# benchmark_common path
# ---------------------------------------------------------------------------
def test_benchmark_evaluate_formula_rejects_injection():
    from scripts.benchmark_common import evaluate_formula

    X = np.linspace(-1.0, 1.0, 21).reshape(-1, 1)
    assert evaluate_formula("x.__class__", X) is None
    assert evaluate_formula("x.__class__.__mro__[1].__subclasses__()", X) is None
    assert evaluate_formula("sin(x)", X) is not None


# ---------------------------------------------------------------------------
# universal proposer path
# ---------------------------------------------------------------------------
def test_universal_proposer_eval_rejects_injection():
    from glassbox.universal_proposer.universal_proposer import _safe_formula_eval

    x = np.linspace(-2.0, 2.0, 31)
    assert _safe_formula_eval("x.__class__", x) is None
    assert _safe_formula_eval("sin(x)", x) is not None
    assert _safe_formula_eval("x**2", x) is not None


def test_universal_proposer_multivariate_eval_rejects_injection():
    from glassbox.universal_proposer.universal_proposer import _safe_formula_eval_multivariate

    X = np.column_stack([np.linspace(-2.0, 2.0, 31), np.linspace(0.0, 3.0, 31)])
    assert _safe_formula_eval_multivariate("x0.__class__", X) is None
    assert _safe_formula_eval_multivariate("x0+x1", X) is not None
    assert _safe_formula_eval_multivariate("np.sin(x0)", X) is not None


# ---------------------------------------------------------------------------
# specialist_state _local_eval path
# ---------------------------------------------------------------------------
def test_specialist_local_eval_path_still_works():
    from glassbox.sr.specialist_state import (
        compute_specialist_state,
        propose_specialist_compositions,
    )

    def _eval_formula(formula, X):
        context = {
            "np": np,
            "sin": np.sin,
            "cos": np.cos,
            "exp": np.exp,
            "log": np.log,
            "sqrt": np.sqrt,
            "abs": np.abs,
        }
        X = np.asarray(X, dtype=np.float64)
        for i in range(X.shape[1]):
            context[f"x{i}"] = X[:, i]
        if X.shape[1] == 1:
            context["x"] = X[:, 0]
        return np.asarray(
            eval(str(formula).replace("^", "**"), {"__builtins__": None}, context),
            dtype=np.float64,
        )

    x = np.linspace(-3.0, 3.0, 120)
    X = np.column_stack([x, np.sin(x)])
    y = np.where(x < 0.0, x * x, np.sin(2.0 * x))
    candidates = [
        {"formula": "x0^2", "validation_r2": 0.2, "validation_mse": 1.0, "source": "poly"},
        {"formula": "sin(2*x0)", "validation_r2": 0.3, "validation_mse": 0.9, "source": "periodic"},
    ]
    state = compute_specialist_state(
        candidates,
        X,
        y,
        evaluate_formula=_eval_formula,
        complexity_fn=lambda formula: len(str(formula)),
        family_signature_fn=lambda formula: "periodic" if "sin" in str(formula) else "poly",
        max_candidates=4,
        max_pairs=2,
    )
    assert state is not None and state.enabled

    # evaluate_formula=None -> the local eval closure (the gated path) is used.
    proposals = propose_specialist_compositions(state, X, y, max_pairs=2)
    assert isinstance(proposals, list)


# ---------------------------------------------------------------------------
# generate_curve_data raw-eval branch
# ---------------------------------------------------------------------------
def test_generate_curve_data_raw_eval_rejects_injection():
    from glassbox.curve_classifier.generate_curve_data import evaluate_formula

    x = np.linspace(-1.0, 1.0, 21)
    y, status = evaluate_formula("x.__class__", x, safe_eval=False)
    assert y is None
    assert status == "eval_fail"
    y, status = evaluate_formula("np.sin(x)", x, safe_eval=False)
    assert status == "ok"
    assert y is not None