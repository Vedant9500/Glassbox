import numpy as np

from scripts import benchmark_common as bc


def test_postprocess_formula_rejects_sympy_piecewise_output():
    formula = "-log(2) + 0.141*x**1.5 + sqrt(2)*x**0.67"

    processed = bc.postprocess_formula(formula)

    assert "Piecewise(" not in processed
    assert "Eq(" not in processed

    X = np.linspace(0.1, 5.0, 300).reshape(-1, 1)
    y_pred, diagnostics = bc.evaluate_formula(processed, X, return_diagnostics=True)

    assert diagnostics["ok"] is True
    assert y_pred is not None
    assert y_pred.shape == (300,)


def test_postprocess_formula_fidelity_guard_rejects_worse_cleanup():
    formula = "0.003746*x**2 - 0.07398*x + 0.03161*log(Abs(x)) - 0.07259*sin(0.7038*x + 1.448) + 0.13"
    X = np.linspace(0.1, 5.0, 300).reshape(-1, 1)
    y = bc.evaluate_formula(formula, X)

    processed = bc.postprocess_formula(formula)
    guarded, diagnostics = bc.postprocess_formula_with_fidelity_guard(formula, X, y)

    assert processed != guarded
    assert diagnostics["postprocess_guard_triggered"] is True
    assert diagnostics["postprocess_guard_reason"] == "processed_formula_worse"
    assert bc.evaluate_formula_mse_on_X(guarded, X, y) < bc.evaluate_formula_mse_on_X(processed, X, y)


def test_postprocess_formula_fidelity_guard_accepts_better_fraction_snap():
    formula = "0.4965 * x ** 2 + x"
    X = np.linspace(-2.0, 2.0, 300).reshape(-1, 1)
    y = 0.5 * X[:, 0] ** 2 + X[:, 0]

    processed = bc.postprocess_formula(formula)
    guarded, diagnostics = bc.postprocess_formula_with_fidelity_guard(formula, X, y)

    assert processed == formula
    assert guarded != formula
    assert diagnostics["postprocess_guard_triggered"] is False
    assert diagnostics["postprocess_processed_mse"] == 0.0
    assert bc.evaluate_formula_mse_on_X(guarded, X, y) == 0.0


def test_postprocess_formula_preserves_signed_power_helper_call():
    formula = "-0.2155*x*_signed_power(Abs(x),0.303)+0.4505*x+0.5"

    processed = bc.postprocess_formula(formula)

    assert "_signed_power(" in processed
    assert "_signed_power*" not in processed
    X = np.linspace(-3.0, 3.0, 200).reshape(-1, 1)
    y_pred, diagnostics = bc.evaluate_formula(processed, X, return_diagnostics=True)

    assert diagnostics["ok"] is True
    assert y_pred is not None


def test_postprocess_formula_preserves_abs_function_call():
    formula = "0.158544597542*sin(1.201*log(Abs(x**2+1)))"

    processed = bc.postprocess_formula(formula)

    assert "Abs(" in processed
    assert "Abs*" not in processed
    X = np.linspace(-3.0, 3.0, 200).reshape(-1, 1)
    y_pred, diagnostics = bc.evaluate_formula(processed, X, return_diagnostics=True)

    assert diagnostics["ok"] is True
    assert y_pred is not None


def test_evaluate_formula_protects_log_abs_zero_endpoint():
    formula = "-0.1111111111111111*x + 0.4*sin(1.125*log(Abs(x)) - 1.4166666666666667) + 0.5555555555555556"
    X = np.linspace(0.0, 8.0, 300).reshape(-1, 1)

    y_pred, diagnostics = bc.evaluate_formula(formula, X, return_diagnostics=True)

    assert diagnostics["ok"] is True
    assert y_pred is not None
    assert np.all(np.isfinite(y_pred))


def test_evaluate_formula_allows_harmless_underflow():
    formula = (
        "1.05796688054*sin(0.3112*exp(-0.00284495021337127*Abs(x)^7.725*sign(x))"
        "*sin(1.779*x)*Abs(x)^0.6864*sign(x))"
    )
    X = np.linspace(0.0, 6.0, 300).reshape(-1, 1)

    y_pred, diagnostics = bc.evaluate_formula(formula, X, return_diagnostics=True)

    assert diagnostics["ok"] is True
    assert y_pred is not None
    assert np.all(np.isfinite(y_pred))
