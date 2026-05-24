import numpy as np

from scripts import run_srbench_local as rsl


def test_run_track1_uses_per_run_params_without_hard_timeout(monkeypatch):
    captured = {"timeouts": []}

    class _FakeEstimator:
        def __init__(self, **kwargs):
            captured["timeouts"].append(kwargs.get("timeout"))
            self.kwargs = kwargs

        def get_params(self):
            return {
                "timeout": 7,
                "random_state": 0,
                "skip_evolution_if_bloated": False,
                "bloat_term_threshold": 20,
                "blackbox_feature_selection": True,
                "blackbox_mode": True,
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
