import numpy as np

from glassbox.sr.sklearn_wrapper import GlassboxRegressor


def test_cv_skip_guard_passes_for_stable_formula(monkeypatch):
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

    monkeypatch.setattr(est, "_safe_eval_formula_array", lambda formula, X_in: y.copy())

    ok = est._passes_cross_validation_skip_guard("x", X, y)

    assert ok is True
    assert est.fast_path_cv_guard_["passed"] is True
    assert est.fast_path_cv_guard_["min_fold_r2"] is not None


def test_cv_skip_guard_fails_for_unstable_formula(monkeypatch):
    n = 120
    x = np.linspace(-2.0, 2.0, n)
    X = x.reshape(-1, 1)
    y = x.copy()

    est = GlassboxRegressor(
        cv_skip_guard_enabled=True,
        cv_skip_guard_folds=3,
        cv_skip_guard_min_fold_r2=0.95,
        cv_skip_guard_max_r2_std=0.02,
        random_state=7,
    )

    idx = np.arange(n)
    rng = np.random.RandomState(7)
    rng.shuffle(idx)
    folds = [f for f in np.array_split(idx, 3) if len(f) > 0]

    y_pred = y.copy()
    y_pred[folds[0]] = y_pred[folds[0]] + 8.0

    monkeypatch.setattr(est, "_safe_eval_formula_array", lambda formula, X_in: y_pred)

    ok = est._passes_cross_validation_skip_guard("x", X, y)

    assert ok is False
    assert est.fast_path_cv_guard_["passed"] is False
    assert est.fast_path_cv_guard_["reason"] == "unstable_fold_performance"


def test_universal_proposer_dual_path_skips_multivariate():
    n = 64
    x1 = np.linspace(-2.0, 2.0, n)
    x2 = np.linspace(1.0, 3.0, n)
    X = np.stack([x1, x2], axis=1)
    y = x1 + x2

    est = GlassboxRegressor(
        use_universal_proposer=True,
        universal_proposer_shadow_mode=True,
        universal_proposer_log_routing=False,
    )

    payload, force = est._run_universal_proposer_dual_path(X, y, fast_path_result=None)

    assert payload is None
    assert force is False
    assert est.universal_proposer_status_ == "skipped_multivariate"


def test_universal_proposer_dual_path_handles_missing_checkpoint():
    n = 64
    x = np.linspace(-2.0, 2.0, n)
    X = x.reshape(-1, 1)
    y = np.sin(x)

    est = GlassboxRegressor(
        use_universal_proposer=True,
        universal_proposer_path="models/does_not_exist.pt",
        universal_proposer_shadow_mode=False,
        universal_proposer_log_routing=False,
    )

    payload, force = est._run_universal_proposer_dual_path(X, y, fast_path_result={"mse": 0.1})

    assert payload is None
    assert force is False
    assert str(est.universal_proposer_status_).startswith("error:")
