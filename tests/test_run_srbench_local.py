import numpy as np

from scripts import run_srbench_local as rsl
from scripts import classifier_fast_path as cfp
from scripts import benchmark_common as bc


def test_run_track1_uses_per_run_params_without_hard_timeout(monkeypatch):
    captured = {"timeouts": [], "max_compute_budgets": []}

    class _FakeEstimator:
        def __init__(self, **kwargs):
            captured["timeouts"].append(kwargs.get("timeout"))
            captured["max_compute_budgets"].append(kwargs.get("max_compute_budget"))
            self.kwargs = kwargs

        def get_params(self):
            return {
                "timeout": 7,
                "random_state": 0,
                "skip_evolution_if_bloated": False,
                "bloat_term_threshold": 20,
                "blackbox_feature_selection": True,
                "blackbox_mode": True,
                "min_compute_budget": 10,
                "max_compute_budget": 300,
            }

        def fit(self, X, y):
            return self

        def get_formula(self):
            return "x0"

    monkeypatch.setattr(rsl, "postprocess_formula", lambda formula: formula)
    monkeypatch.setattr(rsl, "evaluate_formula", lambda formula, X: np.zeros(X.shape[0], dtype=float))
    monkeypatch.setattr(rsl, "r2_score", lambda y_true, y_pred: 0.0)
    monkeypatch.setattr(rsl, "mse_score", lambda y_true, y_pred: 1.0)
    monkeypatch.setattr(rsl, "model_size", lambda formula: 1)
    monkeypatch.setattr(rsl, "estimate_timeout_budget", lambda **kwargs: 19)
    monkeypatch.setattr(rsl, "summarize_seed_runs", lambda runs: {
        "r2_stats": {"median": 0.0},
        "mse_stats": {"median": 1.0},
        "time_stats": {"median": 0.0},
    })
    monkeypatch.setattr(rsl, "summarize_time_to_discovery", lambda *args, **kwargs: {})

    class _FakePMLB:
        @staticmethod
        def fetch_data(name, return_X_y=True):
            X = np.random.RandomState(0).randn(40, 2)
            y = X[:, 0]
            return X, y

    import sys
    sys.modules["pmlb"] = _FakePMLB

    est = _FakeEstimator()
    rsl.run_track1_blackbox(
        est,
        datasets=["dummy_ds"],
        max_datasets=1,
        n_samples=40,
        seeds=[1],
        runs_per_formula=1,
        verbose=False,
        hard_timeout=False,
        adaptive_timeout=False,
        post_simplify=False,
        skip_evolution_if_bloated=False,
        ablation_mode=False,
    )

    # First constructor call is the fixture estimator above; second is the per-run estimator.
    assert captured["timeouts"][-1] == 19
    assert captured["max_compute_budgets"][-1] == 19


def test_srbench_runner_uses_shared_benchmark_helpers():
    assert rsl.postprocess_formula is bc.postprocess_formula
    assert rsl.evaluate_formula is bc.evaluate_formula


def test_make_seeded_train_test_split_changes_with_seed_and_preserves_size():
    X = np.arange(100, dtype=float).reshape(50, 2)
    y = np.arange(50, dtype=float)

    split_a = rsl.make_seeded_train_test_split(X, y, n_samples=20, seed=1)
    split_b = rsl.make_seeded_train_test_split(X, y, n_samples=20, seed=2)
    X_train_a, X_test_a, y_train_a, y_test_a = split_a
    X_train_b, X_test_b, y_train_b, y_test_b = split_b

    assert len(y_train_a) == 16
    assert len(y_test_a) == 4
    assert X_train_a.shape == (16, 2)
    assert X_test_a.shape == (4, 2)
    assert not np.array_equal(y_train_a, y_train_b)
    assert not np.array_equal(y_test_a, y_test_b)


def test_evaluate_formula_supports_log_with_base():
    X = np.random.RandomState(0).randn(10, 3)
    y_pred, diag = rsl.evaluate_formula("log(E,10)", X, return_diagnostics=True)

    assert y_pred is not None
    assert diag["ok"] is True
    assert diag["reason"] == "ok"


def test_evaluate_formula_reports_divide_by_zero():
    X = np.random.RandomState(1).randn(12, 3)
    X[:3, 2] = 0.0

    y_pred, diag = rsl.evaluate_formula("x1/x2", X, return_diagnostics=True)

    assert y_pred is None
    assert diag["ok"] is False
    assert diag["reason"] == "divide_by_zero"


def test_evaluate_formula_protects_fractional_powers_on_negative_inputs():
    X = np.random.RandomState(2).randn(20, 4)
    X[:5, 1] = -np.abs(X[:5, 1]) - 0.1

    y_pred, diag = rsl.evaluate_formula("x1**1.5 + 0.25*x2**0.67", X, return_diagnostics=True)

    assert y_pred is not None
    assert diag["ok"] is True
    assert diag["reason"] == "ok"
    assert np.all(np.isfinite(y_pred))


def test_evaluate_formula_protects_fractional_powers_on_expression_bases():
    X = np.random.RandomState(3).randn(20, 4)

    y_pred, diag = rsl.evaluate_formula(
        "((x1-2.0)/0.5)**1.5 + (x2+x3)**0.67",
        X,
        return_diagnostics=True,
    )

    assert y_pred is not None
    assert diag["ok"] is True
    assert np.all(np.isfinite(y_pred))


def test_evaluate_formula_clips_exp_overflow():
    X = np.random.RandomState(4).randn(10, 2)

    y_pred, diag = rsl.evaluate_formula("exp(1000*x0)", X, return_diagnostics=True)

    assert y_pred is not None
    assert diag["ok"] is True
    assert np.all(np.isfinite(y_pred))


def test_postprocess_formula_protects_fractional_power_terms():
    formula = rsl.postprocess_formula("0.25*x1**1.5 - 0.04272*x2**0.67")

    assert "_signed_power" in formula or "Abs(" in formula or "abs(" in formula or "sign(" in formula


def test_fallback_estimator_predictions_marks_display_formula_failure():
    run_result = {"y_pred_test": np.array([1.0, 2.0, 3.0])}
    eval_diag = {"ok": False, "reason": "invalid_log"}

    y_pred, diag = rsl._fallback_estimator_predictions(run_result, eval_diag, split="test")

    assert np.allclose(y_pred, [1.0, 2.0, 3.0])
    assert diag["ok"] is True
    assert diag["reason"] == "protected_estimator_prediction"
    assert diag["display_formula_failed"] is True
    assert diag["display_formula_reason"] == "invalid_log"


def test_apply_srbench_run_budget_caps_internal_adaptive_budget():
    params = {
        "timeout": 60,
        "min_compute_budget": 10,
        "max_compute_budget": 300,
    }

    capped = rsl._apply_srbench_run_budget(params, 7)

    assert capped["timeout"] == 7
    assert capped["max_compute_budget"] == 7
    assert capped["min_compute_budget"] == 7
    assert params["max_compute_budget"] == 300


def test_specialist_full_enables_residual_phase():
    default = rsl.resolve_specialist_phase_config(
        disable_specialist=False,
        enable_residual_stage=False,
        specialist_full=False,
    )
    full = rsl.resolve_specialist_phase_config(
        disable_specialist=False,
        enable_residual_stage=False,
        specialist_full=True,
    )
    disabled = rsl.resolve_specialist_phase_config(
        disable_specialist=True,
        enable_residual_stage=True,
        specialist_full=True,
    )

    assert default["diagnostics"] is True
    assert default["composition"] is True
    assert default["inception"] is True
    assert default["residual"] is False
    assert full["residual"] is True
    assert full["full"] is True
    assert disabled["enabled"] is False
    assert disabled["residual"] is False


def test_multivariate_universal_fast_path_basis_avoids_fragile_families():
    X = np.random.RandomState(5).randn(40, 3)
    basis, names = cfp.build_basis_from_predictions(
        X,
        predictions={"power": 0.9, "exp": 0.97, "log": 0.97, "periodic": 0.31},
        threshold=0.3,
        universal_basis=True,
    )

    assert basis.shape[0] == X.shape[0]
    assert not any("1/(exp(" in name for name in names)
    assert not any("^1.5" in name or "^0.67" in name or "^1.33" in name for name in names)
    assert not any("sin(1/" in name for name in names)
    assert not any("1/sqrt(1-" in name for name in names)


def test_multivariate_low_trust_fast_path_basis_avoids_fragile_families_even_without_universal():
    X = np.random.RandomState(6).randn(40, 3)
    basis, names = cfp.build_basis_from_predictions(
        X,
        predictions={
            "power": 0.79,
            "exp": 0.97,
            "exponential": 0.97,
            "log": 0.90,
            "periodic": 0.31,
            "addition": 1.0,
            "multiplication": 0.99,
            "rational": 0.98,
        },
        threshold=0.3,
        universal_basis=False,
    )

    assert basis.shape[0] == X.shape[0]
    assert not any("1/(exp(" in name for name in names)
    assert not any("^1.5" in name or "^0.67" in name or "^1.33" in name for name in names)
