"""Regression coverage for §3.277/§3.272: Pareto export strings are simplified."""

import numpy as np

from glassbox.sr.sklearn_wrapper import GlassboxRegressor


def _estimator():
    est = GlassboxRegressor(random_state=0)
    est.blackbox_diagnostics_ = {}
    return est


def test_pareto_entries_are_simplified_and_rescored():
    est = _estimator()
    X = np.linspace(-1.0, 1.0, 20).reshape(-1, 1)
    y = np.sin(X[:, 0])

    est._cleanup_formula_with_fidelity_guard = (
        lambda formula, X_, y_, *, stage=None: (
            "sin(x0)" if "sin" in formula else formula
        )
    )
    est._display_formula_mse = lambda formula, X_, y_: 0.125

    front = [
        {"formula": "(1.0)*(sin(x0))", "mse": 1.0, "complexity": 9},
        {"formula": "x0", "mse": 2.0, "complexity": 1},
        "not-a-dict",
    ]
    out = est._simplify_pareto_front(front, X, y)

    assert out[0]["formula"] == "sin(x0)"
    assert out[0]["pre_simplify_formula"] == "(1.0)*(sin(x0))"
    assert out[0]["mse"] == 0.125
    assert out[0]["complexity"] == est._formula_complexity("sin(x0)")
    # x0 -> x0 is a no-op for the fake cleanup; row is preserved as-is.
    assert out[1]["formula"] == "x0"
    assert out[2] == "not-a-dict"
    assert est.blackbox_diagnostics_["pareto_export_simplification"]["changed"] == 1


def test_pareto_simplification_refuses_variable_rename():
    est = _estimator()
    X = np.linspace(-1.0, 1.0, 20).reshape(-1, 1)
    y = np.sin(X[:, 0])

    # The C++ simplifier canonicalises x0 -> x for single-feature inputs; the
    # public convention forbids that, so the row must stay untouched.
    est._cleanup_formula_with_fidelity_guard = (
        lambda formula, X_, y_, *, stage=None: "sin(x)"
    )
    est._display_formula_mse = lambda formula, X_, y_: 0.0

    front = [{"formula": "sin(x0)", "mse": 1.0, "complexity": 3}]
    out = est._simplify_pareto_front(front, X, y)
    assert out[0]["formula"] == "sin(x0)"
    assert out[0]["mse"] == 1.0
