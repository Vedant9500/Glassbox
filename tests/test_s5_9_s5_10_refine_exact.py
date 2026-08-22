"""S5-9 weighted specialist refine + S5-10 protected exact division."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
CPP = REPO / "glassbox" / "sr" / "cpp"
for p in (REPO, CPP):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from glassbox.sr.cpp import CPP_AVAILABLE, get_cpp_core

_core = get_cpp_core()

requires_cpp = pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ _core not built")


@requires_cpp
def test_s5_9_refine_frequencies_accepts_sample_weight():
    rng = np.random.default_rng(0)
    x = np.linspace(-2, 2, 120).astype(np.float64)
    true_w = 2.5
    y = (0.5 + 1.2 * x + np.sin(true_w * x)).astype(np.float64)
    # Contaminate a few points heavily
    y_noisy = y.copy()
    y_noisy[-8:] += 50.0
    w = np.ones_like(y)
    w[-8:] = 0.01  # downweight outliers

    omegas_u, mse_u = _core.refine_frequencies(x, y_noisy, [2.0], steps=40, lr=0.05)
    omegas_w, mse_w = _core.refine_frequencies(
        x, y_noisy, [2.0], steps=40, lr=0.05, sample_weight=w
    )
    assert len(omegas_u) == 1 and len(omegas_w) == 1
    # Weighted path should recover omega closer to truth under outliers
    err_u = abs(float(omegas_u[0]) - true_w)
    err_w = abs(float(omegas_w[0]) - true_w)
    assert err_w <= err_u + 0.25  # not worse by much; usually better
    assert np.isfinite(float(mse_w))


@requires_cpp
def test_s5_9_refine_powers_accepts_sample_weight():
    x = np.linspace(0.2, 2.0, 80).astype(np.float64)
    y = (1.0 + 0.5 * x + 0.8 * np.abs(x) ** 1.7).astype(np.float64)
    y_noisy = y.copy()
    y_noisy[0:5] += 30.0
    w = np.ones_like(y)
    w[0:5] = 0.02

    out_u, mse_u = _core.refine_powers(x, y_noisy, [1.5], [], steps=30, lr=0.05)
    out_w, mse_w = _core.refine_powers(
        x, y_noisy, [1.5], [], steps=30, lr=0.05, sample_weight=w
    )
    assert np.isfinite(float(mse_u)) and np.isfinite(float(mse_w))
    # Returned powers list non-empty
    powers_w = list(out_w.get("powers", [])) if isinstance(out_w, dict) else []
    assert len(powers_w) >= 1


@requires_cpp
def test_s5_10_exact_division_near_zero_is_finite():
    assert hasattr(_core, "eval_formula_exact")
    x = np.linspace(-1.0, 1.0, 41).astype(np.float64)  # includes 0
    pred = np.asarray(_core.eval_formula_exact("1/x0", [x]), dtype=np.float64)
    assert pred.shape == x.shape
    assert np.all(np.isfinite(pred))
    # Away from zero should be close to true 1/x
    mask = np.abs(x) > 0.2
    true = 1.0 / x[mask]
    assert np.max(np.abs(pred[mask] - true)) < 1e-9


@requires_cpp
def test_s5_10_exact_noninteger_power_negative_base_finite():
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    pred = np.asarray(_core.eval_formula_exact("x0^0.5", [x]), dtype=np.float64)
    assert np.all(np.isfinite(pred))
    # Positive branch matches sqrt
    assert abs(float(pred[3]) - 1.0) < 1e-9
    assert abs(float(pred[4]) - np.sqrt(2.0)) < 1e-9


@requires_cpp
def test_s5_10_graph_score_div_formula_ok():
    x = np.linspace(-1, 1, 60).astype(np.float64)
    y = (1.0 / (x + 0.3)).astype(np.float64)
    X = x.reshape(-1, 1)
    mid = len(x) // 2
    scores = _core.score_formula_candidates(
        ["1/(x0+0.3)", "x0"],
        X[:mid],
        y[:mid],
        X[mid:],
        y[mid:],
    )
    assert len(scores) >= 1
    # Best structure should score finite
    ok_any = any(
        s.get("ok", False)
        or np.isfinite(float(s.get("validation_mse", s.get("val_mse", np.nan))))
        for s in scores
    )
    assert ok_any
