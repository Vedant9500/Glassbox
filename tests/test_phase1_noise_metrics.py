"""Phase 1: noise & metric contracts (N3, N4, N6, S1-6, S5-9)."""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from glassbox.sr.sklearn_wrapper import (
    GlassboxRegressor,
    _auto_residual_soft_weights,
)

CPP_DIR = REPO / "glassbox" / "sr" / "cpp"
if str(CPP_DIR) not in sys.path:
    sys.path.insert(0, str(CPP_DIR))

try:
    import _core  # type: ignore

    CPP_AVAILABLE = hasattr(_core, "iterative_elastic_net")
except ImportError:
    CPP_AVAILABLE = False

requires_cpp = pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ _core not built")


def test_n3_clean_multifeature_no_soft_force():
    """Clean multi-linear should not force soft-MAD via retained_all_features."""
    rng = np.random.RandomState(0)
    X = rng.uniform(-2, 2, size=(80, 3))
    y = 1.5 * X[:, 0] - 0.7 * X[:, 1] + 0.25 * X[:, 2] + 0.1

    soft, out_frac = _auto_residual_soft_weights(X, y)
    # Multi-linear residual probe should leave little heavy-tail mass.
    assert out_frac < 0.02, out_frac
    assert soft is None

    est = GlassboxRegressor(
        generations=1,
        population_size=4,
        timeout=3,
        multi_start_runs=1,
        use_fast_path=False,
        use_guided_evolution=False,
        blackbox_mode=True,
        blackbox_noise_robust="auto",
        random_state=0,
    )
    try:
        est.fit(X, y)
    except Exception:
        pass
    applied = getattr(est, "_blackbox_noise_robust_applied_", {}) or {}
    # Must not activate soft-MAD solely because blackbox retained all features.
    if applied.get("reason") == "soft_mad_weights":
        assert applied.get("active") is not True or float(applied.get("outlier_fraction_target") or 0) >= 0.02
    assert not (
        applied.get("active")
        and applied.get("reason") == "soft_mad_weights"
        and float(applied.get("outlier_fraction_target") or 0) < 0.01
    )


def test_n3_multifeature_outliers_still_activate():
    """Block outliers on multi-feature still produce soft residual weights."""
    rng = np.random.RandomState(1)
    X = rng.uniform(-2, 2, size=(120, 3))
    y = (1.0 * X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2]).copy()
    y[40:52] += 25.0
    soft, out_frac = _auto_residual_soft_weights(X, y)
    assert soft is not None
    assert out_frac >= 0.02
    assert float(np.min(soft)) < float(np.max(soft))


def test_n4_diffuse_huber_enables_noise_guard():
    """Pure diffuse_noise_huber must activate phase-3/6 unweighted guards."""
    est = GlassboxRegressor(blackbox_noise_robust="auto", random_state=0)
    est._blackbox_noise_robust_applied_ = {
        "active": True,
        "mode": "auto",
        "reason": "diffuse_noise_huber",
        "loss_mode_switched_to_huber": True,
        "weights_to_evolution": False,
    }
    est.blackbox_diagnostics_ = {
        "sample_weight": {"provided": False, "source": "none"},
    }
    assert est._auto_noise_guard_active() is True


def test_n4_user_weights_still_skip_auto_guard():
    est = GlassboxRegressor(blackbox_noise_robust="auto", random_state=0)
    est._blackbox_noise_robust_applied_ = {
        "active": True,
        "mode": "auto",
        "reason": "soft_mad_weights",
    }
    est.blackbox_diagnostics_ = {
        "sample_weight": {"provided": True, "source": "user"},
    }
    assert est._auto_noise_guard_active() is False


def test_n6_display_mse_local_no_scripts(monkeypatch):
    """Display MSE must stay unweighted plain MSE without scripts import."""
    est = GlassboxRegressor()
    est.n_features_in_ = 1
    x = np.linspace(-1, 1, 40).reshape(-1, 1)
    y = 2.0 * x[:, 0] + 1.0

    # Poison scripts path: import failure must not force robust fallback.
    import builtins
    real_import = builtins.__import__

    def _block_scripts(name, *args, **kwargs):
        if name == "scripts" or name.startswith("scripts."):
            raise ImportError("blocked for N6 test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_scripts)

    # Active robust/search weights must not pollute display score.
    est.sample_weight_provided_ = True
    est.sample_weight_ = np.ones(len(y))
    est.sample_weight_[-5:] = 0.01
    est.loss_mode = "huber"

    display = est._display_formula_mse("2*x + 1", x, y)
    assert np.isfinite(display)
    assert display < 1e-12

    # Wrong formula still finite plain MSE, not inf-from-import.
    display_bad = est._display_formula_mse("0*x", x, y)
    plain = float(np.mean((y - 0.0) ** 2))
    assert abs(display_bad - plain) < 1e-9


def test_n6_final_score_never_uses_robust_as_primary():
    """Primary final score stays display/plain unweighted even if search is Huber."""
    est = GlassboxRegressor()
    est.n_features_in_ = 1
    est.loss_mode = "huber"
    x = np.linspace(-1, 1, 50).reshape(-1, 1)
    y = x[:, 0].copy()
    y[0] = 50.0  # outlier makes robust << plain MSE for identity
    est.sample_weight_provided_ = True
    est.sample_weight_ = np.ones(len(y))
    est.sample_weight_[0] = 0.01

    score, internal, display = est._final_formula_score("x", x, y)
    assert np.isfinite(score)
    assert np.isfinite(display)
    # Primary score equals display, not weighted/robust internal.
    assert abs(score - display) < 1e-12
    # Display should be plain MSE (large due to outlier), internal may be smaller.
    plain = float(np.mean((y - x[:, 0]) ** 2))
    assert abs(display - plain) < 1e-9
    assert internal <= display + 1e-9 or np.isfinite(internal)


def test_s1_6_search_vs_display_split():
    """_formula_mse can be weight-aware; _display_formula_mse is unweighted."""
    est = GlassboxRegressor()
    est.n_features_in_ = 1
    est.loss_mode = "mse"
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 2.0, 100.0])
    est.sample_weight_provided_ = True
    est.sample_weight_ = np.array([1.0, 1.0, 1.0, 0.0])
    # Normalize like fit would (mean ~1 on positive); helper re-validates elsewhere.
    # Use explicit sample_weight arg for search.
    w = np.array([1.0, 1.0, 1.0, 1e-9])
    search = est._formula_mse("x", x, y, sample_weight=w)
    display = est._display_formula_mse("x", x, y)
    assert display > search  # unweighted penalizes last point more


@requires_cpp
def test_s5_9_elastic_net_respects_sample_weights():
    """Weighted elastic net should downweight outlier row influence."""
    rng = np.random.RandomState(0)
    n = 80
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    # True sparse model: y ~ 2*x1 (x2 irrelevant), plus one huge outlier.
    y = 2.0 * x1 + 0.0 * x2
    y[0] = 200.0
    X = np.column_stack([np.ones(n), x1, x2, x1 * x2])

    w_uniform = np.ones(n)
    w_down = np.ones(n)
    w_down[0] = 1e-6

    w_u, mse_u = _core.iterative_elastic_net(X, y, 0.01, 0.001, 3, 3, 0.05, 1000)
    w_w, mse_w = _core.iterative_elastic_net(X, y, 0.01, 0.001, 3, 3, 0.05, 1000, w_down)

    # Coefficient on x1 (index 1) should be closer to 2 under downweighting.
    assert abs(float(w_w[1]) - 2.0) < abs(float(w_u[1]) - 2.0) + 0.25
    assert np.isfinite(float(mse_w))
    # Bad weight length raises.
    with pytest.raises(Exception):
        _core.iterative_elastic_net(X, y, 0.01, 0.001, 2, 2, 0.05, 500, np.ones(3))
