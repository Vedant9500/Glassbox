"""C-01 / C-01′ regression tests: non-finite predictions must fail scoring.

Audit contract (AUDIT_REPORT.md §4.1 P0.1):
- ``_eval_formula_raw`` keeps non-finite predictions as-is; the eval cache
  stores raw values.
- Search scoring (``_formula_mse``) and display/protocol MSE
  (``_plain_unweighted_mse``) must REJECT non-finite predictions with ``inf``
  instead of letting the historical zero-fill crown domain-failing formulas on
  zero-centered targets.
- ``_safe_eval_formula_array`` remains the predict-only fill policy
  (non-finite → 0) so user-facing predictions stay finite.

Also pins the C-14 display-formatter fix: exported Log strings carry the
engine's ε-protection, so external evaluation of ``log(x)`` at x=0 matches the
engine instead of returning -inf.
"""

import numpy as np
import pytest

from glassbox.sr.cpp import CPP_AVAILABLE, get_cpp_core
from glassbox.sr.sklearn_wrapper import GlassboxRegressor


def _est():
    est = GlassboxRegressor(random_state=0)
    est.n_features_in_ = 1
    return est


def _x_with_zero():
    # Grid that includes an exact zero — the classic log/div domain edge.
    return np.array([-1.0, -0.5, 0.0, 0.25, 0.5, 1.0]).reshape(-1, 1)


# ── Raw vs fill split ────────────────────────────────────────────────────


def test_raw_eval_preserves_nonfinite_and_fill_policy_zeros():
    est = _est()
    X = _x_with_zero()
    raw = est._eval_formula_raw("1/(x0 - x0)", X)  # inf everywhere
    assert raw.shape == (X.shape[0],)
    assert not np.any(np.isfinite(raw))
    filled = est._safe_eval_formula_array("1/(x0 - x0)", X)
    assert np.all(filled == 0.0)


def test_overflow_predictions_rejected():
    est = _est()
    X = np.array([1e200, -1e200]).reshape(-1, 1)
    raw = est._eval_formula_raw("x0**2", X)  # overflows to +inf
    assert np.all(np.isinf(raw))
    y = np.array([1.0, 2.0])
    assert est._formula_mse("x0**2", X, y) == float("inf")
    assert est._plain_unweighted_mse("x0**2", X, y) == float("inf")


def test_shared_cache_stores_raw_values():
    est = _est()
    X = _x_with_zero()
    # Predict policy first (populates cache), then raw must still see non-finite.
    filled = est._safe_eval_formula_array("1/(x0 - x0)", X)
    assert np.all(filled == 0.0)
    raw = est._eval_formula_raw("1/(x0 - x0)", X)
    assert not np.any(np.isfinite(raw))
    hits_before = int(getattr(est, "formula_eval_cache_hits_", 0))
    raw_again = est._eval_formula_raw("1/(x0 - x0)", X)
    assert int(est.formula_eval_cache_hits_) == hits_before + 1
    assert not np.any(np.isfinite(raw_again))


# ── Scoring rejection (the C-01/C-01′ core) ─────────────────────────────


@pytest.mark.parametrize(
    "broken",
    [
        "1/(x0 - x0)",       # divide by exact zero → +inf
        "x0/(x0 - x0)",      # 0/0 → nan, k/0 → ±inf
        "(x0 - x0)**(-3)",   # 0.0 ** negative integer → inf
    ],
)
def test_search_scoring_rejects_domain_failure(broken):
    est = _est()
    X = _x_with_zero()
    y = np.zeros(X.shape[0])  # zero-centered target: zero-fill used to win here
    mse = est._formula_mse(broken, X, y)
    assert mse == float("inf"), f"zero-fill regression: {broken} scored {mse}"


def test_display_protocol_rejects_domain_failure():
    est = _est()
    X = _x_with_zero()
    y = np.zeros(X.shape[0])
    # Historical behavior returned exactly 0.0 via zero-fill + dead finite-mask.
    assert est._plain_unweighted_mse("1/(x0 - x0)", X, y) == float("inf")
    assert est._plain_unweighted_mse("", X, y) == float("inf")


def test_true_model_still_beats_broken_formula_on_zero_centered_target():
    """The audit's P0.1 acceptance scenario: domain-failing formulas must not
    beat a true model via zero-fill."""
    est = _est()
    rng = np.random.RandomState(7)
    X = rng.uniform(-1.0, 1.0, size=(64, 1))
    X[3, 0] = 0.0  # force a domain edge into the grid
    y = 0.05 * np.sin(3.0 * X[:, 0])
    true_mse = est._formula_mse("0.05*sin(3*x0)", X, y)
    broken_mse = est._formula_mse("1/(x0*0.5 - x0*0.5)", X, y)
    assert np.isfinite(true_mse)
    assert broken_mse == float("inf")
    assert true_mse < broken_mse


def test_finite_formulas_score_normally():
    est = _est()
    X = _x_with_zero()
    y = 2.0 * X[:, 0]
    assert est._plain_unweighted_mse("2*x0", X, y) == pytest.approx(0.0, abs=1e-12)
    assert np.isfinite(est._formula_mse("2*x0", X, y))


# ── C-14: display Log string mirrors engine ε-protection ────────────────


@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ _core extension not built")
def test_display_log_string_is_epsilon_protected():
    core = get_cpp_core()
    out = core.snap_formula_floats("log(x)", 1)
    assert "abs(x)" in out, f"display log lost abs-protection: {out}"
    assert "1e-06" in out or "1e-6" in out, f"display log lost epsilon: {out}"


@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ _core extension not built")
def test_display_log_matches_engine_at_zero():
    """Engine computes log(|0|+1e-6); the exported string must evaluate to the
    same value under plain numpy semantics (no -inf at zeros)."""
    core = get_cpp_core()
    display = core.snap_formula_floats("log(x)", 1)
    # Turn the display string into plain-numpy code (external-benchmark style).
    expr = display.replace("log(", "np.log(").replace("abs(", "np.abs(")
    expr = expr.replace("^", "**").replace("|", "")
    x0 = np.array([0.0, 1e-9, 1.0])
    with np.errstate(divide="ignore", invalid="ignore"):
        display_vals = eval(expr, {"np": np}, {"x": x0})
    engine_vals = np.log(np.abs(x0) + 1e-6)
    assert np.allclose(display_vals, engine_vals), (
        f"display {display!r} -> {display_vals} != engine {engine_vals}"
    )
