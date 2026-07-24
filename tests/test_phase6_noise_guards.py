"""Phase 6: noise-aware cleanup, weighted Pareto, residual guards."""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glassbox.sr.sklearn_wrapper import GlassboxRegressor, _mad_scale, _weighted_mse

cpp_dir = REPO_ROOT / "glassbox" / "sr" / "cpp"

from glassbox.sr.cpp import CPP_AVAILABLE, get_cpp_core

_core = get_cpp_core()

requires_cpp = pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ _core not built")


def test_noise_aware_cleanup_slack_widens_under_noise():
    rng = np.random.default_rng(0)
    x = np.linspace(-2.0, 2.0, 120)
    X = x.reshape(-1, 1)
    y_clean = np.sin(x)
    y_noisy = y_clean + rng.normal(0.0, 0.4, size=x.shape)

    est = GlassboxRegressor(random_state=1)
    est.n_features_in_ = 1

    rel_c, abs_c, diag_c = est._noise_aware_cleanup_slack("sin(x0)", X, y_clean)
    rel_n, abs_n, diag_n = est._noise_aware_cleanup_slack("sin(x0)", X, y_noisy)

    assert diag_n["noise_ratio"] > diag_c["noise_ratio"]
    assert rel_n >= rel_c
    assert 0.05 <= rel_n <= 0.35
    assert abs_n >= abs_c * 0.5


def test_cleanup_records_reject_reason_and_slack(monkeypatch):
    x = np.linspace(-2.0, 2.0, 80)
    X = x.reshape(-1, 1)
    y = np.sin(x)
    est = GlassboxRegressor(random_state=3)
    est.n_features_in_ = 1
    est.blackbox_diagnostics_ = {}

    monkeypatch.setattr(est, "_reduce_formula_noise", lambda formula, X_in, y_in: formula)
    monkeypatch.setattr(est, "_simplify_formula", lambda formula: "0")

    cleaned = est._cleanup_formula_with_fidelity_guard("sin(x)", X, y, stage="unit")
    assert cleaned == "sin(x)"
    entry = est.blackbox_diagnostics_["formula_cleanup_guard"][0]
    assert entry["cleanup_rejected_reason"] is not None
    assert "noise_aware_slack" in entry
    assert entry["steps"][0]["reason"] in (
        "internal_mse_regression",
        "display_mse_regression",
        "internal_mse_non_finite",
        "display_mse_non_finite",
        "non_finite_candidate",
    )
    assert est.blackbox_diagnostics_.get("cleanup_rejected_reason")


def test_pareto_uses_weights_to_prefer_clean_structure():
    rng = np.random.default_rng(7)
    x = np.linspace(-3.0, 3.0, 100)
    X = x.reshape(-1, 1)
    y_clean = 2.0 * x + 1.0
    y = y_clean.copy()
    y[-8:] += 50.0  # block outliers
    w = np.ones(len(y))
    w[-8:] = 0.01

    est = GlassboxRegressor(random_state=7)
    est.n_features_in_ = 1
    est.sample_weight_ = w
    est.sample_weight_provided_ = True

    candidates = [
        {"formula": "2*x0 + 1", "source": "true", "complexity": 3},
        {"formula": "0", "source": "const", "complexity": 1},
        {"formula": "50*sin(x0) + 2*x0", "source": "overfit", "complexity": 8},
    ]
    selected = est._select_blackbox_pareto_formula(candidates, X, y)
    assert selected is not None
    assert selected.get("weighted_validation") is True
    assert "2" in selected["formula"] and "x0" in selected["formula"]
    assert "residual_mad_scale" in selected
    assert "residual_outlier_fraction" in selected


def test_residual_stage_rejects_when_unweighted_worsens(monkeypatch):
    x = np.linspace(-2.0, 2.0, 80)
    X = x.reshape(-1, 1)
    # Non-flat residual so residual stage does not early-exit.
    y = np.sin(x) + 0.3 * x

    est = GlassboxRegressor(random_state=11, enable_residual_stage=True, use_guided_evolution=True)
    est.n_features_in_ = 1
    est.blackbox_diagnostics_ = {}

    # Residual candidate is pure high-frequency noise — must not pass holdout guards.
    monkeypatch.setattr(
        est,
        "_build_residual_mini_search_candidates",
        lambda *a, **k: [{"formula": "100*sin(20*x0)", "source": "noise", "complexity": 6}],
    )
    monkeypatch.setattr(
        est,
        "_refine_candidate_formulas",
        lambda pool, X_in, y_in, max_candidates=6: [
            {"formula": "100*sin(20*x0)", "source": "noise", "complexity": 6, "risk_score": 0.8, "validation_r2": 0.1}
        ],
    )

    out = est._stage_residual_symbolic_fit(X, y, "sin(x0)", _allow_recursion=True)
    assert out is None
    guard = est._residual_stage_guard_
    assert guard.get("accepted") is False
    assert guard.get("residual_rejected_as_noise") is True or guard.get("reason") == "no_holdout_improvement"
    assert est.blackbox_diagnostics_.get("residual_rejected_as_noise") is True


def test_residual_stage_accepts_real_structure_improvement(monkeypatch):
    x = np.linspace(-2.0, 2.0, 100)
    X = x.reshape(-1, 1)
    # Base misses a linear term
    y = np.sin(x) + 0.5 * x

    est = GlassboxRegressor(random_state=12, enable_residual_stage=True, use_guided_evolution=True)
    est.n_features_in_ = 1
    est.blackbox_diagnostics_ = {}

    monkeypatch.setattr(
        est,
        "_build_residual_mini_search_candidates",
        lambda *a, **k: [{"formula": "0.5*x0", "source": "true_res", "complexity": 2}],
    )
    monkeypatch.setattr(
        est,
        "_refine_candidate_formulas",
        lambda pool, X_in, y_in, max_candidates=6: [
            {
                "formula": "0.5*x0",
                "source": "true_res",
                "complexity": 2,
                "risk_score": 0.05,
                "validation_r2": 0.99,
            }
        ],
    )

    out = est._stage_residual_symbolic_fit(X, y, "sin(x0)", _allow_recursion=True)
    assert out is not None
    assert "x0" in out
    assert est._residual_stage_guard_.get("accepted") is True


@requires_cpp
def test_reduce_formula_noise_weights_and_holdout_kwargs():
    rng = np.random.default_rng(42)
    X = rng.uniform(-3, 3, size=(120, 2))
    y = 2.0 * X[:, 0] + 1.5 * X[:, 1] + rng.normal(0, 0.01, size=120)
    # Spike outliers on last rows
    y[-10:] += 20.0
    w = np.ones(120)
    w[-10:] = 0.01

    formula = "2.0*x0 + 1.5*x1 + 0.1*sin(x0)"
    reduced_u = _core.reduce_formula_noise(formula, [X[:, 0], X[:, 1]], y)
    reduced_w = _core.reduce_formula_noise(
        formula,
        [X[:, 0], X[:, 1]],
        y,
        y_weights=w,
        holdout_fraction=0.2,
        relative_slack=0.15,
    )
    assert "x0" in reduced_w
    assert "x1" in reduced_w
    # Weighted path should still be a string formula
    assert isinstance(reduced_w, str) and reduced_w
    assert isinstance(reduced_u, str) and reduced_u


@requires_cpp
def test_reduce_formula_noise_rejects_bad_weight_length():
    X = np.random.default_rng(0).uniform(-1, 1, size=(40, 1))
    y = 2.0 * X[:, 0]
    with pytest.raises(Exception):
        _core.reduce_formula_noise("2*x0", [X[:, 0]], y, y_weights=np.ones(10))
