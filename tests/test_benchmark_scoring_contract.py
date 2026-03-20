"""Regression tests for benchmark scoring reliability contracts."""

import numpy as np

from scripts import benchmark_suite as bs


def test_formula_mse_eval_returns_none_on_parse_failure():
    x = np.linspace(-2.0, 2.0, 64)
    y = x ** 2

    assert bs._evaluate_formula_mse("x^2", x, y) is not None
    assert bs._evaluate_formula_mse("sin(", x, y) is None


def test_run_formula_flags_formula_eval_failed(monkeypatch):
    def _fake_fast_path(*args, **kwargs):
        return {
            "formula": "sin(",
            "mse": 1e-12,
            "details": {"n_nonzero": 1, "n_nonzero_simplified": 1},
        }

    monkeypatch.setattr(bs, "run_fast_path", _fake_fast_path)

    result = bs.run_formula(
        formula_str="x^2",
        x_range=(-2.0, 2.0),
        classifier_path="unused.pt",
        n_samples=64,
        device="cpu",
        with_evolution=False,
        evolution_only=False,
    )

    assert result["formula_discovered"]
    assert result["mse_display"] is None
    assert result["error"] == "formula_eval_failed"
    assert result["score"] == "FAIL"
