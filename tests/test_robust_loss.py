"""Phase 4: robust loss modes for search scoring (display stays plain MSE)."""

import numpy as np
import pytest

from glassbox.sr.sklearn_wrapper import (
    GlassboxRegressor,
    _mad_scale,
    _robust_loss,
    _validate_loss_mode,
    _weighted_mse,
)


def test_validate_loss_mode():
    assert _validate_loss_mode("mse") == "mse"
    assert _validate_loss_mode("HUBER") == "huber"
    with pytest.raises(ValueError):
        _validate_loss_mode("nope")


def test_mad_scale_gaussian_approx():
    rng = np.random.RandomState(0)
    r = rng.normal(0, 2.0, size=5000)
    s = _mad_scale(r)
    assert 1.5 < s < 2.5


def test_huber_downweights_outliers_vs_mse():
    pred = np.zeros(20)
    target = np.zeros(20)
    target[-1] = 100.0  # single huge outlier
    mse = _robust_loss(pred, target, "mse")
    huber = _robust_loss(pred, target, "huber", delta=1.0)
    assert huber < mse
    assert huber < 50.0


def test_trimmed_mse_drops_outliers():
    pred = np.zeros(100)
    target = np.zeros(100)
    target[-5:] = 50.0
    mse = _robust_loss(pred, target, "mse")
    trimmed = _robust_loss(pred, target, "trimmed_mse", trim_fraction=0.1)
    assert trimmed < mse * 0.2
    assert trimmed < 1.0


def test_student_t_finite_on_heavy_tails():
    pred = np.zeros(50)
    target = np.zeros(50)
    target[::5] = 1e3
    loss = _robust_loss(pred, target, "student_t")
    assert np.isfinite(loss)
    assert loss < _robust_loss(pred, target, "mse")


def test_robust_loss_respects_sample_weight():
    pred = np.array([0.0, 0.0, 0.0, 10.0])
    target = np.zeros(4)
    w = np.array([1.0, 1.0, 1.0, 0.0])
    # Zero weight on outlier → near-zero mse
    assert _robust_loss(pred, target, "mse", sample_weight=w) == pytest.approx(0.0)
    assert _robust_loss(pred, target, "huber", sample_weight=w, delta=1.0) == pytest.approx(0.0)


def test_estimator_default_loss_is_mse():
    est = GlassboxRegressor(random_state=0, generations=5, population_size=10, timeout=5)
    assert est.loss_mode == "mse"


def test_estimator_rejects_bad_loss_mode():
    with pytest.raises(ValueError):
        GlassboxRegressor(loss_mode="banana")


def test_score_formula_candidate_huber_prefers_clean_structure():
    """Under outliers, huber search loss ranks true linear better than mse alone."""
    x = np.linspace(-2, 2, 120)
    X = x.reshape(-1, 1)
    y = 2.0 * x + 1.0
    y = y.copy()
    y[50:60] += 40.0
    split = 90
    Xf, yf = X[:split], y[:split]
    Xv, yv = X[split:], y[split:]

    est_mse = GlassboxRegressor(random_state=0, loss_mode="mse")
    est_huber = GlassboxRegressor(random_state=0, loss_mode="huber", huber_delta=1.0)

    good_mse = est_mse._score_formula_candidate("x0", Xf, yf, Xv, yv)
    good_h = est_huber._score_formula_candidate("x0", Xf, yf, Xv, yv)
    assert good_mse is not None and good_h is not None
    # Huber search loss on fit residuals should be smaller than plain MSE
    # when outliers dominate the train split.
    assert good_h["search_fit_loss"] < good_mse["unweighted_fit_mse"]
    assert good_h["loss_mode"] == "huber"
    # Display-relevant unweighted diagnostics still present
    assert np.isfinite(good_h["unweighted_validation_mse"])


def test_formula_mse_uses_robust_when_configured():
    est = GlassboxRegressor(random_state=0, loss_mode="huber", huber_delta=1.0)
    x = np.linspace(-1, 1, 40).reshape(-1, 1)
    y = np.zeros(40)
    y[-1] = 100.0
    # Formula "0" predicts zeros
    loss = est._formula_mse("0", x, y)
    plain = _weighted_mse(np.zeros(40), y, None)
    assert loss < plain
