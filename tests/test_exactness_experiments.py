"""Tests for the exactness experiment helpers.

These cover the two signals the experiments rely on: series fingerprinting and
out-of-domain extrapolation. If either silently stops working, every
experiment result becomes meaningless.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exactness import lib  # noqa: E402


def test_series_fingerprint_detects_taylor_sin():
    result = lib.series_fingerprint("x - x^3/6 + x^5/120 - x^7/5040")
    assert result["is_polynomial"] is True
    assert result["series"] == "sin"
    assert result["matched_nonzero"] >= 3
    assert result["score"] >= 0.75


def test_series_fingerprint_detects_taylor_exp():
    result = lib.series_fingerprint("1 + x + x^2/2 + x^3/6 + x^4/24")
    assert result["series"] == "exp"


def test_series_fingerprint_ignores_plain_polynomial():
    result = lib.series_fingerprint("x^2 + 3*x + 7")
    assert result["series"] is None


def test_extrapolation_flags_polynomial_surrogate_for_sin():
    # On [-2, 2] the degree-7 truncation fits well in-domain and fails outside.
    metrics = lib.extrapolation_metrics(
        "x - x^3/6 + x^5/120 - x^7/5040", "sin(x)", (-2.0, 2.0)
    )
    assert metrics["in_mse"] is not None
    assert metrics["in_mse"] < 1e-6
    assert metrics["ood_rel"] is not None
    assert metrics["ood_rel"] > 1e-2


def test_extrapolation_accepts_true_formula():
    metrics = lib.extrapolation_metrics("sin(x)", "sin(x)", (-6.0, 6.0))
    assert metrics["in_mse"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["ood_rel"] == pytest.approx(0.0, abs=1e-12)


def test_extrapolation_handles_unbounded_targets():
    """exp(x) grows without bound; a true copy must still score zero."""
    metrics = lib.extrapolation_metrics("exp(x)", "exp(x)", (-3.0, 3.0))
    assert metrics["ood_rel"] == pytest.approx(0.0, abs=1e-12)
    poly = lib.extrapolation_metrics("1 + x + x^2/2 + x^3/6", "exp(x)", (-3.0, 3.0))
    assert poly["ood_rel"] is not None
    assert poly["ood_rel"] > 1e-2


def test_operator_set_matches_classifier_vocabulary():
    ops = lib.operator_set("sin(x)/(1+x^2)")
    assert "sin" in ops
    assert "rational" in ops
    assert ops <= set(lib.OPERATOR_NAMES)


def test_operator_set_handles_generator_dialect():
    ops = lib.operator_set("np.sin(x) + np.exp(-x**2)")
    assert "sin" in ops
    assert "exp" in ops


def test_label_pair_separates_exact_and_approximation():
    assert lib.label_pair(in_mse=1e-12, ood_rel=1e-6) == "exact"
    assert lib.label_pair(in_mse=1e-12, ood_rel=0.5) == "approximation"
    assert lib.label_pair(in_mse=0.5, ood_rel=0.5) == "other"


def test_scaled_candidates_preserve_operator_structure():
    candidates = lib.scaled_candidates("sin(x)")
    assert candidates
    base = lib.operator_set("sin(x)")
    for _, text in candidates:
        ops = lib.operator_set(text)
        # Scaling adds ``multiplication`` but must not drop a structural op.
        assert base <= ops
        assert "sin" in ops


def test_snap_pi_rewrites_approximations_only():
    from experiments.exactness.run_with_guard import snap_pi

    assert snap_pi("sin(3.142857142857143*x)") == "sin(pi*x)"
    assert snap_pi("cos(3.13852*x)") == "cos(pi*x)"
    assert snap_pi("sin(3.1*x)") == "sin(3.1*x)"
    assert snap_pi("sin(pi*x)") == "sin(pi*x)"


def test_snap_pi_leaves_unrelated_functions_alone():
    from experiments.exactness.run_with_guard import snap_pi

    assert snap_pi("exp(3.14*x)") == "exp(3.14*x)"
    assert snap_pi("x^2 + 1") == "x^2 + 1"


def test_safe_reciprocal_refuses_domain_singularity():
    from experiments.exactness.composite_basis import _safe_reciprocal

    x = np.linspace(-3.0, 3.0, 301)
    assert _safe_reciprocal(x) is None  # x crosses zero
    positive = np.linspace(0.1, 5.0, 301)
    assert _safe_reciprocal(positive) is not None


def test_closure_finds_product_of_primitives():
    from experiments.exactness.composite_basis import fit_closure

    x = np.linspace(0.0, 10.0, 300)
    y = np.exp(-x) * np.sin(x)
    result = fit_closure(x, y, {"sin": 1.0, "exp": 1.0, "power": 1.0}, [1.0, 2.0])
    assert result is not None
    assert result["mse"] < 1e-12
    assert result["n_terms"] == 1


def test_closure_rejects_singular_candidate():
    from experiments.exactness.composite_basis import fit_closure

    x = np.linspace(-3.0, 3.0, 300)
    y = np.sin(x) / (1.0 + x**2)
    result = fit_closure(x, y, {"sin": 1.0, "power": 1.0, "rational": 1.0}, [1.0])
    if result is not None:
        check = lib.evaluate(result["formula"], x)
        assert check is not None
        assert np.all(np.isfinite(check))


def test_series_rationalize_promotes_sin_truncation():
    from experiments.exactness.series_rationalize import rationalize

    x = np.linspace(-2.0, 2.0, 300)
    y = np.sin(x)
    result = rationalize("x - x^3/6 + x^5/120 - x^7/5040", x, y)
    assert result is not None
    assert result["family"] == "sin"
    assert result["mse"] < 1e-12


def test_series_rationalize_promotes_exp_truncation():
    from experiments.exactness.series_rationalize import rationalize

    x = np.linspace(-3.0, 3.0, 300)
    y = np.exp(x)
    result = rationalize("1 + x + x^2/2 + x^3/6 + x^4/24", x, y)
    assert result is not None
    assert result["family"] == "exp"


def test_series_rationalize_rejects_plain_polynomial():
    from experiments.exactness.series_rationalize import rationalize

    x = np.linspace(-2.0, 2.0, 300)
    y = x**2 + 3 * x + 7
    assert rationalize("x^2 + 3*x + 7", x, y) is None


def test_equivalent_accepts_trig_identity_rewrites():
    assert lib.equivalent("2*sin(x)*cos(x)", "sin(2*x)", (-6.0, 6.0))
    assert lib.equivalent("sin(x)^2+cos(x)^2", "1", (-6.0, 6.0))
    assert lib.equivalent("sin(x)^2-cos(x)^2", "-cos(2*x)", (-6.0, 6.0))


def test_equivalent_rejects_approximation_and_semantic_change():
    assert not lib.equivalent("pi", "3.14159", (-5.0, 5.0))
    assert not lib.equivalent("e", "2.71828", (-5.0, 5.0))
    assert not lib.equivalent("sin(x)*log(x+1)", "sin(x)*log(Abs(x)+1)", (0.1, 5.0))
    assert not lib.equivalent("sin(x)", "x - x^3/6", (-2.0, 2.0))


def test_edge_mse_detects_bad_extrapolation():
    from experiments.exactness.run_with_guard import _edge_mse

    x = np.linspace(-3.0, 3.0, 300)
    good = np.sin(x)
    bad = x - x**3 / 6
    good_mse = _edge_mse("sin(x)", x, good)
    bad_mse = _edge_mse("x - x^3/6", x, good)
    assert good_mse is not None and bad_mse is not None
    assert good_mse < 1e-20
    assert bad_mse > 1e-4


def test_composite_pass_accepts_exact_closure_only():
    from experiments.exactness.run_with_guard import apply_composite_pass
    from scripts.benchmark_suite import _generate_data

    # Exact product: the closure should replace the incumbent.
    x, y = _generate_data("exp(-x)*sin(x)", 0.0, 10.0, 300)
    base = {"formula": "0.1*sin(x)", "mse": 1e-3}
    out = apply_composite_pass(base, x, y, {"detected_omegas": [1.0, 2.0]})
    assert out is not base
    assert out["mse"] < 1e-20

    # Nested composition: closure can only fit a dense surrogate (~1e-7), so
    # the precision gate must keep the incumbent.
    x, y = _generate_data("sin(cos(x))", -3.0, 3.0, 300)
    base = {"formula": "0.1*sin(x)", "mse": 1e-3}
    out = apply_composite_pass(base, x, y, {"detected_omegas": [1.0]})
    assert out is base


def test_label_pair_uses_equivalence_not_relative_ood():
    # A polynomial surrogate of an exploding target keeps a small *relative*
    # OOD error, so the threshold rule used to call it exact.
    assert lib.label_pair(
        in_mse=1e-12,
        ood_rel=1e-6,
        candidate="x**8/40320 - x**7/2520 + x**6/720",
        target="sin(x)+exp(-x)",
        x_range=(-5.0, 5.0),
        candidate_kind="polyfit_d8",
    ) == "approximation"

    # A polynomial target fit by a polynomial is the same function.
    assert lib.label_pair(
        in_mse=1e-30,
        ood_rel=1e-30,
        candidate="(25)*(x/5)^2",
        target="x^2",
        x_range=(-5.0, 5.0),
        candidate_kind="polyfit_d2",
    ) == "exact"

    # Structure-preserving rescales stay positive: the constant is a
    # refinement problem, not a structure problem.
    assert lib.label_pair(
        in_mse=4e-6,
        ood_rel=4e-6,
        candidate="0.998*(sin(x))",
        target="sin(x)",
        x_range=(-3.0, 3.0),
        candidate_kind="scaled_0.9980",
    ) == "exact"


def test_composition_closure_finds_nested_forms():
    from experiments.exactness.composition_closure import fit_composition

    x = np.linspace(-3.0, 3.0, 300)
    for formula, y in (
        ("sin(cos(x))", np.sin(np.cos(x))),
        ("exp(sin(x))", np.exp(np.sin(x))),
    ):
        result = fit_composition(x, y, {"sin": 1.0, "cos": 1.0, "exp": 1.0}, [1.0])
        assert result is not None, formula
        assert result["mse"] < 1e-12, formula
        assert result["n_terms"] == 1, formula


def test_composition_closure_keeps_product_support():
    from experiments.exactness.composition_closure import fit_composition

    x = np.linspace(0.0, 10.0, 300)
    y = np.exp(-x) * np.sin(x)
    result = fit_composition(x, y, {"sin": 1.0, "exp": 1.0}, [1.0, 2.0])
    assert result is not None
    assert result["mse"] < 1e-12


def test_simplify_formula_cleans_generated_names():
    from experiments.exactness.composite_basis import _simplify_formula

    cleaned = _simplify_formula("sqrt((x)/(x/(x^2+1.0)))")
    assert cleaned == "sqrt(x^2 + 1.0)"


def test_composition_pass_escalates_and_rejects_dense_surrogates():
    from experiments.exactness.run_with_guard import (
        apply_composite_pass,
        apply_composition_pass,
    )
    from scripts.benchmark_suite import _generate_data

    # Closure misses; a 1-term composition is exact and accepted.
    x, y = _generate_data("sin(cos(x))", -3.0, 3.0, 300)
    base = {"formula": "0.1*sin(x)", "mse": 1e-3}
    first = apply_composite_pass(dict(base), x, y, {"detected_omegas": [1.0]})
    out = apply_composition_pass(first, x, y, {"detected_omegas": [1.0]})
    assert out["mse"] < 1e-20
    assert "guard_composition" in out

    # Not expressible by unary composition: the only fit is a dense
    # surrogate, which the 1-2 term rule rejects.
    x, y = _generate_data("sin(x+sin(x))", -3.0, 3.0, 300)
    base = {"formula": "0.1*sin(x)", "mse": 1e-3}
    out = apply_composition_pass(base, x, y, {"detected_omegas": [1.0]})
    assert out is base


def test_inverse_composition_finds_function_of_sum():
    """The argument of a composition is often a sum, not a basis column."""
    from experiments.exactness.inverse_composition import fit_inverse_composition

    x = np.linspace(-3.0, 3.0, 300)
    cases = (
        ("log(1+exp(x))", np.log1p(np.exp(x)), "log"),
        ("1/(1+exp(-x))", 1.0 / (1.0 + np.exp(-x)), "reciprocal"),
        ("exp(-abs(x))", np.exp(-np.abs(x)), "exp"),
        ("log(x^2+1)", np.log(x * x + 1.0), "log"),
    )
    predictions = {"exp": 1.0, "log": 1.0, "rational": 1.0, "power": 1.0}
    for formula, y, op in cases:
        result = fit_inverse_composition(x, y, predictions, [1.0, 2.0])
        assert result is not None, formula
        assert result["op"] == op, formula
        assert result["mse"] < 1e-12, formula


def test_inverse_composition_rejects_branch_wrapped_argument():
    """arcsin of a wrapped argument is a sawtooth, not the true argument."""
    from experiments.exactness.inverse_composition import fit_inverse_composition

    # x + sin(x) leaves [-pi/2, pi/2] on this domain, so no inverse fit can be
    # exact; the module must not return a machine-precision claim.
    x = np.linspace(-2.5, 2.5, 300)
    y = np.sin(x + np.sin(x))
    result = fit_inverse_composition(x, y, {"sin": 1.0, "power": 1.0}, [1.0])
    assert result is None or result["mse"] > 1e-8


def test_fit_dictionary_rejects_unfaithful_column_name():
    """A formula must describe the matrix column it was fitted on."""
    from experiments.exactness.composite_basis import fit_dictionary

    x = np.linspace(-4.0, 4.0, 300)
    # The name says ``x`` but the values are ``|x|``: formatting this fit
    # would emit ``-x``, a different function. It must be rejected.
    assert fit_dictionary(x, -np.abs(x), [("x", np.abs(x))], ks=(1,)) is None

    honest = fit_dictionary(x, -np.abs(x), [("Abs(x)", np.abs(x))], ks=(1,))
    assert honest is not None
    assert np.allclose(lib.evaluate(honest["formula"], x), -np.abs(x), atol=1e-6)


def test_select_columns_prefers_canonical_name_and_dedupes():
    from experiments.exactness.inverse_composition import _select_columns

    x = np.linspace(-4.0, 4.0, 300)
    canonical = np.abs(x)
    alias = canonical * (1.0 + 1e-10)
    selected = _select_columns(
        [("abs(x)", canonical), ("(x^1.5)/(x^0.5)", alias)], -canonical
    )
    assert len(selected) == 1
    assert selected[0][0] == "abs(x)"


def test_closure_escalates_past_sub_micro_fit():
    """A 5e-7 fast-path fit is still an approximation and must escalate."""
    from experiments.exactness.run_with_guard import apply_composite_pass
    from scripts.benchmark_suite import _generate_data

    x, y = _generate_data("exp(-x)*cos(x)", 0.0, 10.0, 300)
    base = {"formula": "0.1*sin(x)", "mse": 5e-7}
    out = apply_composite_pass(base, x, y, {"detected_omegas": [1.0, 2.0]})
    assert "guard_closure" in out
    assert out["mse"] < 1e-20


def test_inverse_composition_stays_parsimonious_with_many_columns():
    """A dense collinear dictionary must not replace a 1-term identity."""
    from experiments.exactness.inverse_composition import fit_inverse_composition

    x = np.linspace(-4.0, 4.0, 300)
    y = np.exp(-np.abs(x))
    # The benchmark feeds float32 tensors and the classifier's real
    # predictions; both were needed to reproduce the 320-column OMP
    # instability that produced a 4-term surrogate with 1e-7 noise terms.
    xf = x.astype(np.float32).astype(np.float64)
    yf = y.astype(np.float32).astype(np.float64)
    predictions = {
        "exp": 0.80017,
        "multiplication": 0.89603,
        "periodic": 0.55766,
        "exponential": 0.80017,
    }
    result = fit_inverse_composition(
        xf, yf, predictions, [0.7846311730035055, 1.569262346007011, 2.3538935190105166]
    )
    assert result is not None
    assert result["n_terms"] <= 2
    assert "Abs(x)" in result["formula"]
    assert result["mse"] < 1e-12


def test_equivalent_understands_signed_power_helper():
    """Production emits `_signed_power`; the analyzer must parse it."""
    formula = "_signed_power(Abs(x ** 2 + 1), -0.5)"
    assert lib.equivalent(formula, "1/sqrt(1+x^2)", (-4.0, 4.0))
    values = lib.evaluate(formula, np.linspace(-4.0, 4.0, 101))
    assert values is not None
    assert np.all(np.isfinite(values))
