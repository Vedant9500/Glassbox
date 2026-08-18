"""R-01 regression tests: swallowed-exception visibility.

The audit flagged ~175 bare ``except Exception`` sites in
``sklearn_wrapper.py`` that hide remap/scoring bugs. The fix (per report §A.3)
adds a ``swallowed_errors_`` diagnostics counter and typed exceptions at pure
conversion/import fallback sites so soft-fail paths stay observable post-fit.
"""
import numpy as np
import pytest

from glassbox.sr.sklearn_wrapper import GlassboxRegressor


# ---------------------------------------------------------------------------
# Counter infrastructure
# ---------------------------------------------------------------------------
def test_swallowed_errors_counter_initialised():
    est = GlassboxRegressor(random_state=0)
    assert est.swallowed_errors_ == {}
    assert est.swallowed_errors_enabled_ is True


def test_record_swallowed_error_counts_by_site_and_type():
    est = GlassboxRegressor(random_state=0)
    est._record_swallowed_error("scoring.eval", ValueError("boom"))
    est._record_swallowed_error("scoring.eval", ValueError("again"))
    est._record_swallowed_error("scoring.eval", KeyError("nope"))
    est._record_swallowed_error("remap.fail", RuntimeError("x"))

    assert est.swallowed_errors_["scoring.eval"]["count"] == 3
    assert est.swallowed_errors_["scoring.eval"]["types"] == {
        "ValueError": 2,
        "KeyError": 1,
    }
    assert est.swallowed_errors_["remap.fail"]["count"] == 1
    assert est.swallowed_errors_["remap.fail"]["types"] == {"RuntimeError": 1}


def test_record_swallowed_error_respects_disable_flag():
    est = GlassboxRegressor(random_state=0)
    est.swallowed_errors_enabled_ = False
    est._record_swallowed_error("scoring.eval", ValueError("boom"))
    assert est.swallowed_errors_ == {}


def test_record_swallowed_error_initialises_missing_dict():
    est = GlassboxRegressor(random_state=0)
    del est.swallowed_errors_
    est._record_swallowed_error("a", ValueError("boom"))
    assert est.swallowed_errors_["a"]["count"] == 1


# ---------------------------------------------------------------------------
# Type-narrowed fallbacks
# ---------------------------------------------------------------------------
def test_conversion_helpers_use_typed_exceptions():
    from glassbox.sr.sklearn_wrapper import _clamp_int, _clamp_float, _finite_float

    # String garbage still falls back to default (no behaviour change).
    assert _clamp_int("not-a-number", default=5, lo=0, hi=10) == 5
    assert _clamp_float("not-a-number", default=1.5, lo=0.0, hi=10.0) == 1.5
    assert _finite_float("not-a-number", default=3.0) == 3.0
    assert _finite_float(float("nan"), default=3.0) == 3.0
    # Valid values pass through unchanged.
    assert _clamp_int(7, default=5, lo=0, hi=10) == 7
    assert _clamp_float(0.2, default=1.5, lo=0.0, hi=1.0) == 0.2


# ---------------------------------------------------------------------------
# Fit-level integration: diagnostics expose the summary
# ---------------------------------------------------------------------------
def test_fit_exposes_swallowed_errors_summary():
    est = GlassboxRegressor(
        random_state=0,
        use_fast_path=False,
        use_guided_evolution=False,
        use_universal_proposer=False,
    )
    rng = np.random.RandomState(0)
    X = rng.uniform(0.0, 1.0, size=(60, 1))
    y = 2.0 * X[:, 0] + 1.0
    est.fit(X, y)

    assert isinstance(est.swallowed_errors_summary_, dict)
    assert "total" in est.swallowed_errors_summary_
    assert est.swallowed_errors_summary_["total"] >= 0
    assert isinstance(est.swallowed_errors_summary_["sites"], dict)
    # Diagnostics mirror the summary when present.
    if isinstance(getattr(est, "blackbox_diagnostics_", None), dict):
        assert "swallowed_errors" in est.blackbox_diagnostics_


def test_scoring_hot_path_records_eval_failure():
    est = GlassboxRegressor(random_state=0)
    est.swallowed_errors_ = {}
    X = np.asarray([[1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)

    # A formula that references an unknown feature triggers eval failure.
    inf_score = est._formula_mse("x9 + 1", X, y)
    assert not np.isfinite(inf_score)
    assert est.swallowed_errors_.get("formula_mse.eval", {}).get("count", 0) >= 1

    inf_plain = est._plain_unweighted_mse("x9 + 1", X, y)
    assert not np.isfinite(inf_plain)
    assert est.swallowed_errors_.get("plain_unweighted_mse.eval", {}).get("count", 0) >= 1