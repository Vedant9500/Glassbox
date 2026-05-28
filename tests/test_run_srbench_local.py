import numpy as np

from scripts import run_srbench_local as rsl


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
