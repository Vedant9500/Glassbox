import numpy as np

from scripts import benchmark_common as bc


def test_normalize_formula_text_handles_unicode_math_symbols():
    assert bc.normalize_formula_text("x² + x³") == "x^2+x^3"
    assert bc.normalize_formula_text("π·x + √(x)") == "pi*x+sqrt(x)"
    assert bc.normalize_formula_text("φ⋅ω") == "phi*omega"


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
    assert bc.evaluate_formula_mse_on_X(guarded, X, y) < bc.evaluate_formula_mse_on_X(
        processed, X, y
    )


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


def test_canonical_rewrites_exp_log_and_trig_identities():
    rewritten = bc.apply_canonical_rewrites("exp(log(x+1)) + log(exp(x))")
    assert "exp(log" not in rewritten
    assert "log(exp" not in rewritten

    trig_product = bc.apply_canonical_rewrites("sin(x)*cos(x)")
    assert "sin(2 * x) / 2" == trig_product

    trig_identity = bc.apply_canonical_rewrites("sin(x)**2 + cos(x)**2")
    assert trig_identity == "1"


def test_round_powers_to_integers_snaps_near_int_exponents():
    assert "x**2" in bc.round_powers_to_integers("x**2.08").replace(" ", "")
    assert "x**3" in bc.round_powers_to_integers("1.0*x**2.95").replace(" ", "")
    # Far from integer should stay fractional (or unchanged structure)
    far = bc.round_powers_to_integers("x**2.5", tol=0.1)
    assert "2.5" in far.replace(" ", "") or "2.5" in far


def test_exactness_pass_prefers_integer_power_when_raw_is_good():
    X = np.linspace(-2.0, 2.0, 200).reshape(-1, 1)
    y = X[:, 0] ** 2
    # Numerical twin with fractional power close to 2; display MSE is weak.
    formula = "x**2.02"
    display_m = bc.evaluate_formula_mse_on_X(formula, X, y)
    out, diag = bc.run_exactness_pass(
        formula,
        X,
        y,
        raw_mse=1e-6,
        display_mse=display_m,
        raw_mse_threshold=1e-3,
    )
    assert diag["attempted"] is True
    assert diag["accepted"] is True
    assert out.replace(" ", "") in {"x**2", "(x)**2"}
    assert bc.evaluate_formula_mse_on_X(out, X, y) <= 1e-12


def test_formula_benchmark_seed_is_stable_per_formula_and_range():
    s1 = bc.formula_benchmark_seed("x**2+sin(x)", (-1, 1), base_seed=0)
    s2 = bc.formula_benchmark_seed("x**2+sin(x)", (-1, 1), base_seed=0)
    s3 = bc.formula_benchmark_seed("x**2+sin(x)", (-2, 2), base_seed=0)
    s4 = bc.formula_benchmark_seed("x**2+sin(x)", (-1, 1), base_seed=7)
    assert s1 == s2
    assert s1 != s3
    assert s1 != s4
    assert 0 <= s1 < 2**31


def test_postprocess_formula_with_fidelity_guard_accepts_safe_trig_rewrite():
    formula = "sin(x)*cos(x)"
    X = np.linspace(-2.0, 2.0, 200).reshape(-1, 1)
    y = np.sin(X[:, 0]) * np.cos(X[:, 0])

    guarded, diagnostics = bc.postprocess_formula_with_fidelity_guard(formula, X, y)

    assert diagnostics["postprocess_guard_triggered"] is False
    assert guarded.replace(" ", "") == "sin(2*x)/2"
    assert bc.evaluate_formula_mse_on_X(guarded, X, y) < 1e-12


def test_postprocess_formula_uses_snap_only_for_mixed_complex_formula():
    formula = (
        "-5.853 + 0.1074*x + 0.04159*sin(2*x) - 3.86*cos(x/2) + 11.59*exp(-x) "
        "+ 8.403*x**2/(exp(x)-1) + 0.8905*x**3/(exp(x)-1) "
        "+ 0.09981*x**3/(x**4+0.5) - 1/4*x**2/(x**4+1.0) "
        "+ 0.1158*x**2/(x**4+2.0) - 0.1806*cos(2.00*x)/(x**2+1.0) "
        "+ (2/pi)*cos(2.00*x)/(x**2+2.0)"
    )
    X = np.linspace(-1.0, 5.0, 300).reshape(-1, 1)

    processed = bc.postprocess_formula(formula)
    y_pred, diagnostics = bc.evaluate_formula(processed, X, return_diagnostics=True)

    assert diagnostics["ok"] is True
    assert y_pred is not None
    assert np.all(np.isfinite(y_pred))


def test_postprocess_formula_guard_rejects_domain_unsafe_raw_eval_failure():
    X = np.linspace(-2.0, -0.1, 80).reshape(-1, 1)
    y = X[:, 0]

    guarded, diagnostics = bc.postprocess_formula_with_fidelity_guard(
        "exp(log(x))", X, y
    )

    assert guarded.replace(" ", "") == "exp(log(x))"
    assert diagnostics["postprocess_guard_triggered"] is True
    assert (
        diagnostics["postprocess_guard_reason"]
        == "raw_formula_eval_failed_after_rewrite"
    )


def test_evaluate_formula_accepts_1d_and_list_inputs():
    y_pred, diagnostics = bc.evaluate_formula(
        "x + 1", [1, 2, 3], return_diagnostics=True
    )

    assert diagnostics["ok"] is True
    assert np.allclose(y_pred, [2, 3, 4])


def test_compute_stability_stats_worst_decile_respects_metric_direction():
    assert bc.compute_stability_stats([0.1, 0.5, 0.9])["worst_decile"] == 0.1
    assert (
        bc.compute_stability_stats([0.001, 0.1, 10.0], higher_is_better=False)[
            "worst_decile"
        ]
        == 10.0
    )


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
