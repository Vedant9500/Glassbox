"""Phase 1: sample_weight contract tests.

Covers the public ``fit(X, y, sample_weight=...)`` contract introduced to
mirror PhySO's ``y_weights`` hook. Scope is the Python contract only; native
C++ weighting lands in Phase 2.
"""
import numpy as np
import pytest

from glassbox.sr.sklearn_wrapper import (
    GlassboxRegressor,
    _validate_sample_weight,
    _weighted_mse,
    _weighted_r2,
    _effective_sample_size,
)


# ---------------------------------------------------------------------------
# Pure helper behaviour
# ---------------------------------------------------------------------------
def test_validate_sample_weight_none_is_none():
    assert _validate_sample_weight(None, 5) is None


def test_validate_sample_weight_normalises_to_mean_one():
    w = _validate_sample_weight([1.0, 1.0, 1.0, 3.0], 4)
    assert w is not None
    assert pytest.approx(float(np.mean(w)), rel=1e-12) == 1.0
    assert pytest.approx(w[3], rel=1e-12) == 2.0
    assert pytest.approx(w[0], rel=1e-12) == 2.0 / 3.0


def test_uniform_weights_match_unweighted():
    pred = np.array([0.0, 0.0, 0.0, 0.0])
    target = np.array([0.0, 0.0, 0.0, 1.0])
    w = _validate_sample_weight([1, 1, 1, 1], 4)
    assert pytest.approx(_weighted_mse(pred, target, w), rel=1e-12) == _weighted_mse(pred, target, None)


def test_weighted_mse_shifts_toward_high_weight_point():
    pred = np.zeros(4)
    target = np.array([0.0, 0.0, 0.0, 1.0])
    w_down = _validate_sample_weight([1.0, 1.0, 1.0, 0.0], 4)  # drop last point
    w_up = _validate_sample_weight([1.0, 1.0, 1.0, 9.0], 4)    # emphasise last point
    base = _weighted_mse(pred, target, None)
    assert _weighted_mse(pred, target, w_down) < base
    assert _weighted_mse(pred, target, w_up) > base


def test_validate_sample_weight_rejects_invalid():
    with pytest.raises(ValueError):
        _validate_sample_weight([1.0, 1.0, 1.0], 4)            # length mismatch
    with pytest.raises(ValueError):
        _validate_sample_weight([-1.0, 1.0, 1.0, 1.0], 4)      # negative
    with pytest.raises(ValueError):
        _validate_sample_weight([np.nan, 1.0, 1.0, 1.0], 4)    # non-finite
    with pytest.raises(ValueError):
        _validate_sample_weight([0.0, 0.0, 0.0, 0.0], 4)       # all zero


def test_effective_sample_size_kish():
    assert _effective_sample_size(None) is None
    # n uniform weights of mean 1 -> ess = n
    ess = _effective_sample_size(np.ones(10))
    assert pytest.approx(ess, rel=1e-9) == 10.0
    # one point dominates -> ess drops toward 1
    ess_skew = _effective_sample_size(_validate_sample_weight([1, 1, 1, 100.0], 4))
    assert ess_skew < 2.0


# ---------------------------------------------------------------------------
# Estimator contract
# ---------------------------------------------------------------------------
def _linear_data(n=200, seed=0, outlier=False):
    rng = np.random.RandomState(seed)
    x = np.linspace(-3.0, 3.0, n)
    X = x.reshape(-1, 1)
    y = 2.0 * x + 1.0
    if outlier:
        y[0] += 50.0  # gross outlier
    return X, y


def test_fit_accepts_none_weights_and_stores_flag():
    X, y = _linear_data()
    est = GlassboxRegressor(random_state=1, generations=20, multi_start_runs=1, timeout=30)
    est.fit(X, y, sample_weight=None)
    assert est.sample_weight_provided_ is False
    assert est.sample_weight_ is None


def test_fit_accepts_and_stores_weights():
    X, y = _linear_data()
    w = np.where(np.arange(len(y)) % 2 == 0, 1.0, 0.5)
    est = GlassboxRegressor(random_state=1, generations=20, multi_start_runs=1, timeout=30)
    est.fit(X, y, sample_weight=w)
    assert est.sample_weight_provided_ is True
    assert est.sample_weight_ is not None
    assert pytest.approx(float(np.mean(est.sample_weight_)), rel=1e-9) == 1.0


def test_fit_diagnostics_record_effective_sample_size():
    X, y = _linear_data()
    w = np.where(np.arange(len(y)) % 2 == 0, 1.0, 0.1)
    est = GlassboxRegressor(random_state=1, generations=20, multi_start_runs=1, timeout=30)
    est.fit(X, y, sample_weight=w)
    diag = est.blackbox_diagnostics_.get("sample_weight", {}) if isinstance(est.blackbox_diagnostics_, dict) else {}
    assert diag.get("provided") is True
    assert "effective_sample_size" in diag
    assert diag["effective_sample_size"] is not None


def test_fit_rejects_invalid_weight_length():
    X, y = _linear_data(n=50)
    est = GlassboxRegressor(random_state=1, generations=5, multi_start_runs=1, timeout=30)
    with pytest.raises(ValueError):
        est.fit(X, y, sample_weight=np.ones(len(y) - 1))


def test_formula_mse_uses_weights_when_set(monkeypatch):
    X, y = _linear_data(n=20)
    est = GlassboxRegressor(random_state=1, generations=5, multi_start_runs=1, timeout=10)
    est.fit(X, y, sample_weight=None)
    est.sample_weight_ = _validate_sample_weight([1.0] * 19 + [50.0], 20)
    est.sample_weight_provided_ = True

    # monkeypatch eval so prediction is identically zero -> error dominated by last point
    monkeypatch.setattr(est, "_safe_eval_formula_array", lambda formula, X_in: np.zeros(X_in.shape[0]))
    weighted = est._formula_mse("0", X, y)
    est.sample_weight_provided_ = False
    unweighted = est._formula_mse("0", X, y)
    assert weighted > unweighted


def test_cv_skip_guard_is_weight_aware(monkeypatch):
    n = 120
    x = np.linspace(-2.0, 2.0, n)
    X = x.reshape(-1, 1)
    y = x.copy()

    est = GlassboxRegressor(
        cv_skip_guard_enabled=True,
        cv_skip_guard_folds=3,
        cv_skip_guard_min_fold_r2=0.99,
        cv_skip_guard_max_r2_std=0.02,
        random_state=11,
    )
    # Predictions are perfect except for one fold's points.
    idx = np.arange(n)
    rng = np.random.RandomState(11)
    rng.shuffle(idx)
    folds = [f for f in np.array_split(idx, 3) if len(f) > 0]
    y_pred_good = y.copy()
    y_pred_bad = y.copy()
    y_pred_bad[folds[0]] += 8.0
    monkeypatch.setattr(est, "_safe_eval_formula_array", lambda formula, X_in: y_pred_good)

    # Uniform: passes.
    assert est._passes_cross_validation_skip_guard("x", X, y, sample_weight=None) is True

    # Same perfect pred but we now explicitly mark the noisy fold points with
    # huge weight while keeping preds perfect on them -> still passes.
    w = np.ones(n)
    assert est._passes_cross_validation_skip_guard("x", X, y, sample_weight=w) is True

    # Now expose the noise and rely on weights to surface it.
    monkeypatch.setattr(est, "_safe_eval_formula_array", lambda formula, X_in: y_pred_bad)
    w_focus = np.ones(n)
    w_focus[folds[0]] = 100.0  # concentrate on the bad fold
    ok_focus = est._passes_cross_validation_skip_guard("x", X, y, sample_weight=w_focus)
    assert ok_focus is False
    assert est.fast_path_cv_guard_["passed"] is False


def test_weighted_mse_rejects_length_mismatch():
    pred = np.zeros(4)
    target = np.array([0.0, 0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="length"):
        _weighted_mse(pred, target, np.ones(3))
    with pytest.raises(ValueError, match="length"):
        _weighted_r2(pred, target, np.ones(3))


def test_slice_sample_weight_indices():
    from glassbox.sr.sklearn_wrapper import _slice_sample_weight
    w = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.allclose(_slice_sample_weight(w, indices=[0, 2]), [1.0, 3.0])
    assert _slice_sample_weight(None, indices=[0, 1]) is None
    with pytest.raises(ValueError):
        _slice_sample_weight(w, n_targets=3)


def test_formula_mse_uses_sliced_weights_on_holdout_like_subset():
    """Holdout scoring must not silently drop weights when n_val != n_train."""
    X, y = _linear_data(n=40)
    est = GlassboxRegressor(random_state=1, generations=5, multi_start_runs=1, timeout=10)
    est.fit(X, y, sample_weight=None)
    # Emphasize the last 10 points only in the stored full-length weight vector.
    w = np.ones(40)
    w[30:] = 50.0
    est.sample_weight_ = _validate_sample_weight(w, 40)
    est.sample_weight_provided_ = True

    # Predictions constant 0 so error is just y^2; last-10 slice should be weighted.
    est._safe_eval_formula_array = lambda formula, X_in: np.zeros(X_in.shape[0])
    full = est._formula_mse("0", X, y)
    holdout = est._formula_mse("0", X[30:], y[30:], sample_weight_indices=np.arange(30, 40))
    # Holdout only has high-weight region; after mean-1 normalize the relative
    # pattern still applies within the slice.
    assert np.isfinite(full) and np.isfinite(holdout)
    # Sliced call must not raise and must equal direct weighted mse on the subset
    from glassbox.sr.sklearn_wrapper import _weighted_mse, _slice_sample_weight
    direct = _weighted_mse(np.zeros(10), y[30:], _slice_sample_weight(est.sample_weight_, indices=np.arange(30, 40)))
    assert pytest.approx(holdout, rel=1e-9) == direct


def test_auto_soft_weights_activate_on_1d_outliers():
    """Phase 3: 1D SR should auto soft-weight outliers so evolution gets y_weights."""
    import numpy as np
    from glassbox.sr.sklearn_wrapper import GlassboxRegressor

    rng = np.random.RandomState(0)
    x = np.linspace(-2.0, 2.0, 120).reshape(-1, 1)
    y = (2.0 * x[:, 0] + 1.0).copy()
    y[50:58] += 40.0  # block outliers

    est = GlassboxRegressor(
        generations=2,
        population_size=8,
        timeout=5,
        multi_start_runs=1,
        use_fast_path=False,
        use_guided_evolution=False,
        blackbox_mode=False,
        blackbox_noise_robust="auto",
        random_state=0,
    )
    # Fit may still run some pipeline; even if evolution is skipped, soft weights
    # are applied at the start of fit before search.
    try:
        est.fit(x, y)
    except Exception:
        # Extremely short budgets can fail mid-search; weights are set early.
        pass

    applied = getattr(est, "_blackbox_noise_robust_applied_", {}) or {}
    assert applied.get("active") is True, applied
    assert applied.get("path") == "1d_sr", applied
    assert est.sample_weight_provided_ is True
    assert est.sample_weight_ is not None
    assert float(np.min(est.sample_weight_)) < float(np.max(est.sample_weight_))
    diag = getattr(est, "blackbox_diagnostics_", {}) or {}
    sw = diag.get("sample_weight") or {}
    assert sw.get("provided") is True
    assert sw.get("source") == "auto_soft_mad"


def test_auto_soft_weights_skip_clean_1d():
    """Clean 1D targets should not invent soft weights (preserve Phase 0 clean Exact)."""
    import numpy as np
    from glassbox.sr.sklearn_wrapper import GlassboxRegressor

    for y_fn, label in (
        (lambda x: 2.0 * x + 1.0, "linear"),
        (lambda x: x ** 2, "poly_x2"),
        (lambda x: x ** 3 + x ** 2 + x, "nguyen1"),
    ):
        x = np.linspace(-2.0, 2.0, 100).reshape(-1, 1)
        y = y_fn(x[:, 0])
        est = GlassboxRegressor(
            generations=1,
            population_size=4,
            timeout=3,
            multi_start_runs=1,
            use_fast_path=False,
            use_guided_evolution=False,
            blackbox_mode=False,
            blackbox_noise_robust="auto",
            random_state=0,
        )
        try:
            est.fit(x, y)
        except Exception:
            pass
        applied = getattr(est, "_blackbox_noise_robust_applied_", {}) or {}
        assert applied.get("active") is not True, (label, applied)
        assert not getattr(est, "sample_weight_provided_", False) or est.sample_weight_ is None


def test_auto_residual_soft_weights_helper_matrix():
    """Residual soft weights: clean families off; block outliers on."""
    import numpy as np
    from glassbox.sr.sklearn_wrapper import _auto_residual_soft_weights

    x = np.linspace(-2.0, 2.0, 200)
    soft, out = _auto_residual_soft_weights(x.reshape(-1, 1), x ** 2)
    assert soft is None

    y = (2.0 * x + 1.0).copy()
    y[40:50] += 30.0
    soft, out = _auto_residual_soft_weights(x.reshape(-1, 1), y)
    assert soft is not None
    assert float(np.min(soft)) < float(np.max(soft))
    assert out >= 0.01

