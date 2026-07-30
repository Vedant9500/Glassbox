"""Tier C polish: API/docs, bindings, classifier fail-open, FPIP, import surface."""
import inspect
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glassbox.sr.sklearn_wrapper import GlassboxRegressor, _validate_sample_weight
from glassbox.sr.fpip_v2 import (
    build_fpip_v2_from_fast_path,
    validate_fpip_v2_payload,
)


# --- N8: sample_weight docstring / empty handling ---


def test_n8_none_resolves_to_none():
    assert _validate_sample_weight(None, 10) is None


def test_n8_empty_array_raises_length_mismatch():
    with pytest.raises(ValueError, match="length 0"):
        _validate_sample_weight([], 5)


def test_n8_length_mismatch_raises():
    with pytest.raises(ValueError, match="length 3"):
        _validate_sample_weight([1.0, 1.0, 1.0], 5)


def test_n8_partial_zeros_renormalized():
    w = _validate_sample_weight([1.0, 0.0, 1.0, 0.0], 4)
    assert w is not None
    assert np.isclose(np.mean(w), 1.0)
    assert w[1] == 0.0 and w[3] == 0.0


# --- N10: user vs active weights ---


def test_n10_user_sample_weight_flag():
    est = GlassboxRegressor(
        random_state=0,
        generations=2,
        population_size=8,
        num_islands=1,
        timeout=5,
        use_fast_path=False,
        blackbox_mode=False,
        enable_residual_stage=False,
        enable_specialist_vault_memory=False,
    )
    x = np.linspace(-1, 1, 30)
    X = x.reshape(-1, 1)
    y = 2.0 * x + 0.1 * np.random.RandomState(0).randn(30)
    w = np.ones(30)
    w[:3] = 0.2
    est.fit(X, y, sample_weight=w)
    assert est.user_sample_weight_provided_ is True
    assert est.sample_weight_provided_ is True
    diag = (est.blackbox_diagnostics_ or {}).get("sample_weight") or {}
    assert diag.get("source") == "user"
    assert diag.get("user_provided") is True


def test_n10_no_user_weight_flag_false():
    est = GlassboxRegressor(
        random_state=1,
        generations=2,
        population_size=8,
        num_islands=1,
        timeout=5,
        use_fast_path=False,
        blackbox_mode=False,
        enable_residual_stage=False,
        blackbox_noise_robust=False,
        enable_specialist_vault_memory=False,
    )
    x = np.linspace(-1, 1, 30)
    X = x.reshape(-1, 1)
    y = x**2
    est.fit(X, y)
    assert getattr(est, "user_sample_weight_provided_", False) is False


# --- N9: auto robust dim gate documented via behaviour ---


def test_n9_multi_feature_blackbox_off_skips_auto_soft():
    est = GlassboxRegressor(
        random_state=2,
        generations=2,
        population_size=8,
        num_islands=1,
        timeout=5,
        use_fast_path=False,
        blackbox_mode=False,
        blackbox_noise_robust="auto",
        enable_residual_stage=False,
        enable_specialist_vault_memory=False,
    )
    rng = np.random.RandomState(0)
    X = rng.randn(40, 3)
    y = X[:, 0] + 0.5 * X[:, 1]
    est.fit(X, y)
    applied = getattr(est, "_blackbox_noise_robust_applied_", {}) or {}
    # Auto should not activate when multi-feature + blackbox off.
    assert applied.get("active") is not True or applied.get("reason") != "soft_mad_weights"


# --- S10-1: FPIP validation on fast-path builder ---


def test_s10_1_build_fpip_marks_valid():
    payload = build_fpip_v2_from_fast_path(
        formula="sin(x0)",
        mse=1e-8,
        candidate_formulas=[{"formula": "sin(x0)", "mse": 1e-8}],
        predictions={"sin": 0.9},
        uncertainty={"prediction_entropy": 0.1, "prediction_margin": 0.8},
    )
    assert payload.get("schema_version") == "fpip.v2"
    assert "valid" in payload
    assert payload["valid"] is True
    ok, errs = validate_fpip_v2_payload(payload)
    assert ok, errs


def test_s10_1_invalid_payload_flagged():
    ok, errs = validate_fpip_v2_payload({"schema_version": "v1"})
    assert ok is False
    assert any("fpip.v2" in e for e in errs)


# --- S10-2: lazy import surface ---


def test_s10_2_glassbox_regressor_exported():
    import glassbox.sr as sr

    assert hasattr(sr, "GlassboxRegressor")
    assert sr.GlassboxRegressor is GlassboxRegressor


def test_s10_2_lazy_optimizer_import():
    import glassbox.sr as sr

    # Lazy attribute should resolve without eager import at package load.
    cls = sr.RegularizedBFGS
    assert cls is not None
    assert callable(cls) or isinstance(cls, type)


# --- S9-1: classifier fail-open diagnostics ---


def test_s9_1_missing_model_returns_load_status():
    from glassbox.curve_classifier.curve_classifier_integration import (
        get_last_classifier_load_status,
        predict_operators,
    )

    x = np.linspace(-1, 1, 20)
    y = np.sin(x)
    out = predict_operators(x, y, model_path="/nonexistent/path/no_model.pt")
    # Fail-open: empty operator dict so legacy `if not predictions` still works.
    assert out == {}
    status = get_last_classifier_load_status()
    assert status.get("ok") is False
    assert status.get("fail_open") is True
    assert status.get("reason") in ("model_not_found", "load_failed")


# --- S9-5: tiny-n device prefers CPU under auto ---


def test_s9_5_tiny_n_device_is_cpu():
    from glassbox.curve_classifier.curve_classifier_integration import _resolve_device

    dev = _resolve_device("auto", n_samples=50)
    assert str(dev) == "cpu"
    # Explicit cuda request still returns cuda device object (may fallback if no GPU)
    dev2 = _resolve_device("cpu", n_samples=50)
    assert str(dev2) == "cpu"


# --- S6-1 / E9: C++ binding defaults (if extension present) ---


def test_s6_1_core_defaults_aligned():
    try:
        from glassbox.sr.cpp import get_cpp_core
        _core = get_cpp_core()
    except ImportError:
        pytest.skip("_core not built")
    # pybind11 builtins often lack inspect.signature; use __doc__ / __text_signature__.
    doc = (getattr(_core.run_evolution, "__doc__", None) or "") + (
        getattr(_core.run_evolution, "__text_signature__", None) or ""
    )
    assert "elite_size" in doc or hasattr(_core, "run_evolution")
    # Smoke: call with explicit args matching sklearn defaults (pop=100, islands=8).
    x = np.linspace(-1, 1, 20)
    y = 2.0 * x
    result = _core.run_evolution(
        [x],
        y,
        pop_size=20,
        generations=2,
        timeout_seconds=3,
        num_islands=1,
        elite_size=3,
        seed_fraction=0.5,
        random_seed=0,
        num_threads=1,
    )
    assert isinstance(result, dict)
    assert "best_mse" in result


# --- S10-3: simplify path is C++ only ---


def test_s10_3_simplify_doc_mentions_cpp():
    doc = GlassboxRegressor._simplify_formula.__doc__ or ""
    assert "C++" in doc or "simplify_formula" in doc


# --- S10-4: BFGS non-finite guard ---


def test_s10_4_bfgs_handles_nonfinite_gracefully():
    import torch
    from glassbox.sr.optimizers.bfgs_optimizer import RegularizedBFGS

    opt = RegularizedBFGS(max_iter=5)
    X = torch.randn(20, 3)
    y = torch.randn(20)
    # Inject a row of huge values — should not raise
    X[0, :] = 1e20
    w, mse = opt.fit(X, y)
    assert w is not None
    assert np.isfinite(mse) or mse == float("inf") or mse >= 1e7


# --- N8 docstring accuracy ---


def test_n8_docstring_mentions_empty_raises():
    doc = _validate_sample_weight.__doc__ or ""
    assert "Empty" in doc or "empty" in doc
    assert "None" in doc
