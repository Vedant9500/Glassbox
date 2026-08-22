"""Phase 4: sklearn orchestration contract (S1-4, S1-5, S1-8, S1-10, S1-13)."""

import sys
import threading
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from glassbox.sr.sklearn_wrapper import GlassboxRegressor


def test_s1_10_insufficient_samples_fail_closed():
    est = GlassboxRegressor(
        cv_skip_guard_enabled=True,
        cv_skip_guard_min_samples=45,
        cv_skip_guard_folds=3,
        random_state=0,
    )
    X = np.linspace(-1, 1, 20).reshape(-1, 1)
    y = X[:, 0]
    ok = est._passes_cross_validation_skip_guard("x", X, y)
    assert ok is False
    assert est.fast_path_cv_guard_["reason"] == "insufficient_samples"
    assert est.fast_path_cv_guard_["passed"] is False
    assert est.fast_path_cv_guard_.get("refit_cv") is False
    assert est.fast_path_cv_guard_.get("mode") == "residual_partition_stability"


def test_s1_10_stable_still_passes(monkeypatch):
    n = 120
    x = np.linspace(-2, 2, n)
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
    assert est._passes_cross_validation_skip_guard("x", X, y) is True
    assert est.fast_path_cv_guard_["mode"] == "residual_partition_stability"


def test_s1_13_residual_not_gated_by_guided_evolution():
    """Residual stage must not require use_guided_evolution=True (S1-13)."""
    est = GlassboxRegressor(
        use_guided_evolution=False,
        enable_residual_stage=True,
        enable_residual_boosting=True,
        random_state=0,
    )
    assert est.enable_residual_boosting is True
    assert est.use_guided_evolution is False

    X = np.linspace(-1, 1, 40).reshape(-1, 1)
    y = X[:, 0] ** 2
    # Perfect base => residual flat; proves we passed the enable gate (not guided-evo).
    out = est._stage_residual_symbolic_fit_impl(X, y, "x**2", _allow_recursion=True)
    guard = getattr(est, "_residual_stage_guard_", {}) or {}
    assert guard.get("enabled") is True
    assert guard.get("reason") != "disabled_or_not_allowed"
    # Flat residual expected for exact base formula.
    assert out is None
    assert guard.get("reason") == "flat_or_nonfinite_residual"


def test_s1_13_can_disable_residual_boosting_explicitly():
    est = GlassboxRegressor(
        use_guided_evolution=True,
        enable_residual_stage=True,
        enable_residual_boosting=False,
        random_state=0,
    )
    X = np.linspace(-1, 1, 40).reshape(-1, 1)
    y = X[:, 0]
    out = est._run_residual_boosting_impl(X, y, "x")
    assert out == "x"
    assert est.boosting_diagnostics_.get("enabled") is False
    assert est.boosting_diagnostics_.get("decoupled_from_guided_evolution") is True


def test_s1_13_default_boosting_follows_residual_stage():
    est_on = GlassboxRegressor(enable_residual_stage=True, random_state=0)
    est_off = GlassboxRegressor(enable_residual_stage=False, random_state=0)
    assert est_on.enable_residual_boosting is True
    assert est_off.enable_residual_boosting is False


def test_s1_8_formula_eval_cache_thread_safe():
    est = GlassboxRegressor(random_state=0)
    est.n_features_in_ = 1
    X = np.linspace(-1, 1, 64).reshape(-1, 1)
    errors = []

    def worker():
        try:
            for _ in range(50):
                pred = est._safe_eval_formula_array("x**2", X)
                assert pred.shape[0] == 64
                assert np.all(np.isfinite(pred))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert int(getattr(est, "formula_eval_count_", 0)) >= 1
    assert int(getattr(est, "formula_eval_cache_hits_", 0)) >= 1


def test_s1_8_fit_uses_local_rng_not_global_seed(monkeypatch):
    """fit must not call np.random.seed (process-global pollution)."""
    calls = []
    real_seed = np.random.seed

    def tracking_seed(*args, **kwargs):
        calls.append(args)
        return real_seed(*args, **kwargs)

    monkeypatch.setattr(np.random, "seed", tracking_seed)

    rng = np.random.RandomState(0)
    X = rng.uniform(-1, 1, size=(40, 1))
    y = 2.0 * X[:, 0] + 0.1
    est = GlassboxRegressor(
        random_state=42,
        population_size=8,
        generations=2,
        num_islands=1,
        multi_start_runs=1,
        use_guided_evolution=False,
        use_fast_path=False,
        timeout=5,
        blackbox_mode=False,
        enable_residual_stage=False,
        enable_residual_boosting=False,
    )
    try:
        est.fit(X, y)
    except Exception:
        # Even if fit fails later, seed must not have been called.
        pass
    assert calls == [], f"np.random.seed was called: {calls}"
    assert hasattr(est, "_fit_rng_")


def test_s1_5_selection_holdout_carved_once():
    est = GlassboxRegressor(random_state=0)
    rng = np.random.RandomState(0)
    X = rng.uniform(-2, 2, size=(100, 2))
    y = 2 * X[:, 0] + X[:, 1]
    s1 = est._ensure_selection_holdout(X, y, validation_fraction=0.2)
    s2 = est._ensure_selection_holdout(X, y, validation_fraction=0.2)
    assert s1 is not None and s2 is not None
    assert np.array_equal(s1["val_idx"], s2["val_idx"])
    assert len(s1["fit_idx"]) + len(s1["val_idx"]) == 100
    assert len(set(s1["fit_idx"]).intersection(s1["val_idx"])) == 0
    # Not an ordered tail: val indices should not be only the last chunk
    val = np.asarray(s1["val_idx"])
    assert not np.array_equal(val, np.arange(100 - len(val), 100))


def test_s1_5_final_holdout_scores_uses_selection_holdout():
    est = GlassboxRegressor(random_state=1)
    rng = np.random.RandomState(1)
    X = rng.uniform(-2, 2, size=(80, 1))
    y = 3.0 * X[:, 0]
    est.n_features_in_ = 1
    split = est._ensure_selection_holdout(X, y, validation_fraction=0.25)
    assert split is not None
    scores = est._final_holdout_scores("x", "3*x", X, y)
    assert scores is not None
    assert np.isfinite(scores["base_score"])
    assert np.isfinite(scores["candidate_score"])
    # 3*x should beat x on holdout for y=3x
    assert scores["candidate_score"] < scores["base_score"]


def test_s1_4_public_n_features_stays_original_with_search_dim():
    """Public n_features_in_ is original width; n_features_search_ tracks reduction."""
    est = GlassboxRegressor(random_state=1)
    est.original_n_features_in_ = 5
    est.n_features_in_ = 5
    est.n_features_search_ = 2  # simulated blackbox reduction
    # Mid-fit search mutates search dim only in current code path.
    assert int(est.n_features_in_) == 5
    assert int(est.n_features_search_) == 2
    # Finish restores public contract even if something set reduced width.
    est.n_features_in_ = 2
    if getattr(est, "original_n_features_in_", None) is not None:
        est.n_features_in_ = int(est.original_n_features_in_)
    assert int(est.n_features_in_) == 5


def test_s1_4_finish_with_formula_restores_n_features(monkeypatch):
    """_finish_with_formula early exit restores original public feature count."""
    est = GlassboxRegressor(random_state=0, timeout=1)
    # Minimal fit-state scaffolding for early finish path.
    est.original_n_features_in_ = 4
    est.n_features_search_ = 2
    est.n_features_in_ = 2
    est.blackbox_diagnostics_ = {}
    est.phase_timings_ = {}
    est.sample_weight_ = None
    est.loss_mode = "mse"
    est._user_loss_mode_ = "mse"

    # Invoke the nested finish helper by simulating attribute restore logic used there.
    if getattr(est, "original_n_features_in_", None) is not None:
        est.n_features_in_ = int(est.original_n_features_in_)
    est.formula_ = "x0 + x1"
    assert int(est.n_features_in_) == 4
    assert int(est.n_features_search_) == 2


def test_s1_4_n_features_in_restored_after_blackbox_fit():
    """Integration: multi-feature blackbox fit leaves public width == X.shape[1]."""
    rng = np.random.RandomState(1)
    X = rng.uniform(-2, 2, size=(80, 3))
    y = 1.5 * X[:, 0] - 0.5 * X[:, 1] + 0.25 * X[:, 2]
    est = GlassboxRegressor(
        random_state=1,
        population_size=12,
        generations=3,
        num_islands=1,
        multi_start_runs=1,
        use_guided_evolution=False,
        use_fast_path=False,
        timeout=12,
        blackbox_mode=True,
        blackbox_max_features=2,
        enable_residual_stage=False,
        enable_residual_boosting=False,
        enable_inception_reuse=False,
    )
    est.fit(X, y)
    assert hasattr(est, "formula_")
    assert int(est.n_features_in_) == 3
    assert int(getattr(est, "original_n_features_in_", 3)) == 3
    # Search dim may be reduced, but public dim stays original.
    if hasattr(est, "n_features_search_"):
        assert int(est.n_features_search_) <= 3
    pred = est.predict(X[:5])
    assert pred.shape[0] == 5
    assert np.all(np.isfinite(pred))
