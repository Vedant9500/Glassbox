import numpy as np

from glassbox.sr.sklearn_wrapper import GlassboxRegressor
from scripts import benchmark_suite as bs


def test_glassbox_regressor_exposes_exact_match_backend_params():
    est = GlassboxRegressor(
        exact_match_backend="torch_cuda",
        exact_match_min_gpu_work=123,
        exact_match_max_combos=456,
    )

    params = est.get_params()

    assert params["exact_match_backend"] == "torch_cuda"
    assert params["exact_match_min_gpu_work"] == 123
    assert params["exact_match_max_combos"] == 456


def test_benchmark_run_formula_passes_exact_match_backend(monkeypatch):
    captured = {}
    diagnostics = {
        "backend_requested": "torch_cuda",
        "gpu_used": True,
        "validated_on_cpu": True,
    }

    def _fake_run_fast_path(*args, **kwargs):
        captured.update(kwargs)
        return {
            "formula": "x",
            "mse": 0.0,
            "details": {
                "n_nonzero": 1,
                "n_nonzero_simplified": 1,
                "exact_match_diagnostics": diagnostics,
            },
            "candidate_formulas": [],
            "uncertainty": {},
            "residual_diagnostics": {},
            "operator_hints": {},
        }

    monkeypatch.setattr(bs, "run_fast_path", _fake_run_fast_path)
    monkeypatch.setattr(bs, "detect_dominant_frequency", lambda *args, **kwargs: None)
    monkeypatch.setattr(bs, "_display_eval_details", lambda *args, **kwargs: {"mse": 0.0, "diagnostics": {"ok": True}})
    monkeypatch.setattr(bs, "_postprocess_formula_for_benchmark", lambda formula, *args, **kwargs: (formula, {}))

    result = bs.run_formula(
        formula_str="x",
        x_range=(-1.0, 1.0),
        classifier_path="unused.pt",
        n_samples=32,
        device="cuda",
        exact_match_backend="torch_cuda",
        exact_match_min_gpu_work=1,
        exact_match_max_combos=12,
    )

    assert captured["exact_match_backend"] == "torch_cuda"
    assert captured["exact_match_min_gpu_work"] == 1
    assert captured["exact_match_max_combos"] == 12
    assert result["exact_match_diagnostics"] == diagnostics


def test_specialist_benchmark_passes_exact_match_backend(monkeypatch):
    captured = {}
    diagnostics = {
        "backend_requested": "cuda",
        "gpu_used": True,
    }

    class FakeRegressor:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.best_mse_ = 0.0
            self.fast_path_exact_match_diagnostics_ = diagnostics
            self.blackbox_diagnostics_ = {}
            self.phase_timings_ = {}
            self.formula_eval_count_ = 0
            self.formula_eval_cache_hits_ = 0
            self._formula_eval_cache_ = {}
            self.specialist_vault_ = None
            self.inception_rounds_ = []
            self.inception_diagnostics_ = {}

        def fit(self, X, y):
            return self

        def get_formula(self):
            return "x"

    import glassbox.sr.sklearn_wrapper as sw

    monkeypatch.setattr(sw, "GlassboxRegressor", FakeRegressor)
    monkeypatch.setattr(bs, "_display_eval_details", lambda *args, **kwargs: {"mse": 0.0, "diagnostics": {"ok": True}})
    monkeypatch.setattr(bs, "_postprocess_formula_for_benchmark", lambda formula, *args, **kwargs: (formula, {}))
    monkeypatch.setattr(bs.cfp, "_evaluate_formula_values", lambda formula, x: np.asarray(x, dtype=float).reshape(-1))

    result = bs.run_formula_specialist_regressor(
        formula_str="x",
        x_range=(-1.0, 1.0),
        classifier_path="unused.pt",
        proposer_path=None,
        n_samples=32,
        device="cuda",
        exact_match_backend="cuda",
        exact_match_min_gpu_work=1,
        exact_match_max_combos=12,
    )

    assert captured["exact_match_backend"] == "cuda"
    assert captured["exact_match_min_gpu_work"] == 1
    assert captured["exact_match_max_combos"] == 12
    assert result["exact_match_diagnostics"] == diagnostics
