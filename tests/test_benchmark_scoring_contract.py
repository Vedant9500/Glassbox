"""Regression tests for benchmark scoring reliability contracts."""

import numpy as np

from scripts import benchmark_suite as bs
from scripts import classifier_fast_path as cfp


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


def test_residual_diagnostics_flag_structured_residual():
    x = np.linspace(-2.0, 2.0, 256)
    y_true = np.sin(3.0 * x)
    y_pred = y_true - 0.2 * np.sin(9.0 * x)

    diagnostics = cfp._residual_diagnostics(y_true, y_pred, x)

    assert diagnostics["residual_mse"] is not None
    assert diagnostics["residual_spectral_peak_ratio"] is not None
    assert diagnostics["residual_suspicious"] is True


def test_prediction_uncertainty_metrics():
    metrics = cfp._prediction_uncertainty_metrics({"sin": 0.7, "cos": 0.2, "exp": 0.1})

    assert metrics["prediction_entropy"] is not None
    assert metrics["prediction_margin"] is not None
    assert abs(metrics["prediction_top1"] - 0.7) < 1e-12
    assert abs(metrics["prediction_top2"] - 0.2) < 1e-12
    assert metrics["prediction_uncertain"] is False


def test_residual_diagnostics_handles_nan_mask_with_holdout():
    x = np.linspace(-3.0, 3.0, 128)
    y_true = np.sin(x)
    y_pred = np.sin(x) + 0.05 * np.cos(5.0 * x)
    y_true[5] = np.nan
    y_pred[7] = np.nan

    diagnostics = cfp._residual_diagnostics(y_true, y_pred, x)

    assert diagnostics["residual_mse"] is not None
    assert diagnostics["residual_holdout_ratio"] is not None
    assert np.isfinite(diagnostics["residual_holdout_ratio"])


def test_run_formula_triggers_guided_on_uncertainty(monkeypatch):
    guided_called = {"value": False}

    def _fake_fast_path(*args, **kwargs):
        return {
            "formula": "x",
            "mse": 0.0,
            "details": {"n_nonzero": 1, "n_nonzero_simplified": 1},
            "uncertainty": {
                "prediction_entropy": 0.95,
                "prediction_margin": 0.02,
                "prediction_uncertain": True,
            },
            "residual_diagnostics": {"residual_suspicious": False},
            "operator_hints": {},
        }

    def _fake_guided(*args, **kwargs):
        guided_called["value"] = True
        return {"formula": "x", "mse": 0.0}

    monkeypatch.setattr(bs, "run_fast_path", _fake_fast_path)
    monkeypatch.setattr(bs, "run_guided_evolution", _fake_guided)

    result = bs.run_formula(
        formula_str="x",
        x_range=(-2.0, 2.0),
        classifier_path="unused.pt",
        n_samples=64,
        device="cpu",
        with_evolution=True,
        evolution_only=False,
    )

    assert guided_called["value"] is True
    assert result["formula_discovered"]


def test_run_formula_triggers_guided_on_suspicious_residual(monkeypatch):
    guided_called = {"value": False}

    def _fake_fast_path(*args, **kwargs):
        return {
            "formula": "x",
            "mse": 0.0,
            "details": {"n_nonzero": 1, "n_nonzero_simplified": 1},
            "uncertainty": {
                "prediction_entropy": 0.1,
                "prediction_margin": 0.8,
                "prediction_uncertain": False,
            },
            "residual_diagnostics": {"residual_suspicious": True},
            "operator_hints": {},
        }

    def _fake_guided(*args, **kwargs):
        guided_called["value"] = True
        return {"formula": "x", "mse": 0.0}

    monkeypatch.setattr(bs, "run_fast_path", _fake_fast_path)
    monkeypatch.setattr(bs, "run_guided_evolution", _fake_guided)

    result = bs.run_formula(
        formula_str="x",
        x_range=(-2.0, 2.0),
        classifier_path="unused.pt",
        n_samples=64,
        device="cpu",
        with_evolution=True,
        evolution_only=False,
    )

    assert guided_called["value"] is True
    assert result["formula_discovered"]
