"""Tier A open-P2 fixes: E8/N7/S3-3/S3-4/S3-5/S7-2/S8-2 regression coverage."""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glassbox.sr.sklearn_wrapper import GlassboxRegressor, _mad_scale
from glassbox.sr.blackbox_preprocessor import (
    remap_reduced_formula_to_original,
    remap_original_formula_to_reduced,
    formula_from_search_to_original_space,
    prepare_blackbox_search,
)
from glassbox.sr.specialist_state import SpecialistVault

cpp_dir = REPO_ROOT / "glassbox" / "sr" / "cpp"
if str(cpp_dir) not in sys.path:
    sys.path.insert(0, str(cpp_dir))

try:
    import _core

    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False

requires_cpp = pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ _core not built")


# ── S3-3: user noise protocol activates guards ──────────────────────────────

def test_s3_3_user_weights_enable_noise_guard():
    est = GlassboxRegressor(random_state=0)
    est.sample_weight_provided_ = True
    est.blackbox_diagnostics_ = {
        "sample_weight": {"provided": True, "source": "user"},
    }
    est._blackbox_noise_robust_applied_ = {"active": False}
    assert est._auto_noise_guard_active() is True


def test_s3_3_user_huber_enables_noise_guard():
    est = GlassboxRegressor(random_state=0, loss_mode="huber", huber_delta=1.0)
    est.sample_weight_provided_ = False
    est._blackbox_noise_robust_applied_ = {"active": False}
    est.blackbox_diagnostics_ = {}
    assert est._auto_noise_guard_active() is True


def test_s3_3_clean_mse_path_guards_off():
    est = GlassboxRegressor(random_state=0, loss_mode="mse")
    est.sample_weight_provided_ = False
    est._blackbox_noise_robust_applied_ = {"active": False}
    est.blackbox_diagnostics_ = {}
    assert est._auto_noise_guard_active() is False


# ── S3-4: residual acceptance bar under noise ───────────────────────────────

def test_s3_4_residual_rejects_tiny_weighted_gain(monkeypatch):
    """Under weights, a residual term with <1.5% weighted val gain is rejected."""
    rng = np.random.default_rng(0)
    x = np.linspace(-2.0, 2.0, 100)
    X = x.reshape(-1, 1)
    y = np.sin(x) + 0.05 * rng.normal(size=x.shape)
    w = np.ones(len(y))
    w[-10:] = 0.05

    est = GlassboxRegressor(random_state=1)
    est.n_features_in_ = 1
    est.sample_weight_ = w
    est.sample_weight_provided_ = True
    est.loss_mode = "mse"
    est._blackbox_noise_robust_applied_ = {"active": False}
    est.blackbox_diagnostics_ = {
        "sample_weight": {"provided": True, "source": "user"},
    }
    est._residual_stage_guard_ = {}

    base = "sin(x0)"
    # Residual candidate that barely helps (near-noise term).
    tiny = "0.001*x0"

    monkeypatch.setattr(
        est,
        "_build_residual_mini_search_candidates",
        lambda *a, **k: [{"formula": tiny, "source": "unit", "complexity": 2, "risk_score": 0.0}],
    )
    monkeypatch.setattr(
        est,
        "_refine_candidate_formulas",
        lambda cands, *a, **k: [
            {"formula": tiny, "source": "unit", "complexity": 2, "risk_score": 0.0, "validation_r2": 0.01}
        ],
    )

    out = est._stage_residual_symbolic_fit_impl(X, y, base, _allow_recursion=True)
    # Tiny residual should not be accepted under the raised noise bar.
    assert out is None or out == base or (isinstance(out, str) and tiny not in out)
    guard = getattr(est, "_residual_stage_guard_", {}) or {}
    # Either rejected as no improvement or never accepted.
    assert guard.get("accepted") in (False, None) or guard.get("residual_rejected_as_noise")


# ── S3-5: snap fidelity helper ──────────────────────────────────────────────

def test_s3_5_snap_with_fidelity_rejects_bad_snap():
    est = GlassboxRegressor(random_state=0)
    est.n_features_in_ = 1
    x = np.linspace(-1.0, 1.0, 60)
    X = x.reshape(-1, 1)
    # True structure uses 1.7 — snapping to 2 would hurt a lot.
    y = 1.7 * x
    # Force a snap that would change 1.7 → 2 with large atol.
    bad = est._snap_with_fidelity(
        "1.7*x0", X, y, mode="integer", atol=0.5, max_rel_mse_increase=0.01
    )
    assert bad == "1.7*x0"  # fidelity reject


def test_s3_5_snap_with_fidelity_accepts_near_integer():
    est = GlassboxRegressor(random_state=0)
    est.n_features_in_ = 1
    x = np.linspace(-1.0, 1.0, 60)
    X = x.reshape(-1, 1)
    y = 2.0 * x
    good = est._snap_with_fidelity(
        "2.00001*x0", X, y, mode="integer", atol=1e-3, max_rel_mse_increase=0.05
    )
    assert "2" in good and "2.00001" not in good


# ── S7-2: OOB remap → neutral constant ──────────────────────────────────────

def test_s7_2_oob_local_index_maps_to_zero():
    # selected has 3 features (local x0,x1,x2); x5 is OOB local
    mapped = remap_reduced_formula_to_original("x0 + x5", [3, 5, 7])
    assert mapped == "x3 + 0"
    assert "x5" not in mapped  # must not leave OOB as raw index


def test_s7_2_dropped_original_maps_to_zero():
    # selected [3,5,7]; x1 is dropped original
    mapped = remap_original_formula_to_reduced("x3 + x1", [3, 5, 7])
    assert mapped == "x0 + 0"


def test_s7_2_standardized_oob_expands_to_zero():
    rng = np.random.RandomState(0)
    X = rng.randn(40, 4)
    y = X[:, 0] + X[:, 2]
    _, _, state = prepare_blackbox_search(
        X, y, enabled=True, max_features=2, standardize=True, min_features_to_select=2
    )
    if not state.standardized:
        pytest.skip("standardize not applied")
    # Force an OOB local index past selected
    oob_local = len(state.selected_features) + 3
    formula = formula_from_search_to_original_space(f"x0 + x{oob_local}", state)
    assert "+0" in formula.replace(" ", "") or "+ 0" in formula or formula.endswith("+0)")
    assert f"x{oob_local}" not in formula


# ── S8-2: composition cap ───────────────────────────────────────────────────

def _eval_formula(formula, X):
    """Simple eval matching test_specialist_state for vault unit tests."""
    context = {
        "np": np,
        "sin": np.sin,
        "cos": np.cos,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "abs": np.abs,
    }
    X = np.asarray(X, dtype=np.float64)
    for i in range(X.shape[1]):
        context[f"x{i}"] = X[:, i]
    if X.shape[1] == 1:
        context["x"] = X[:, 0]
    expr = str(formula).replace("^", "**")
    return np.asarray(eval(expr, {"__builtins__": None}, context), dtype=np.float64)


def test_s8_2_composition_cap_at_most_four():
    rng = np.random.RandomState(2)
    X = rng.uniform(-1.0, 1.0, size=(100, 2))
    y = X[:, 0] + X[:, 1]
    vault = SpecialistVault(max_entries=8)
    vault.add_candidates(
        [
            {"formula": "x0", "validation_r2": 0.5, "validation_mse": 0.5, "source": "outer"},
            {"formula": "x1", "validation_r2": 0.5, "validation_mse": 0.5, "source": "inner"},
            {"formula": "0.5*x0", "validation_r2": 0.3, "validation_mse": 0.7, "source": "c"},
        ],
        X,
        y,
        evaluate_formula=_eval_formula,
        complexity_fn=lambda formula: 5,
        family_signature_fn=lambda formula: str(formula)[:3],
        run_index=0,
        max_new=3,
    )
    proposals = vault.propose_compositions(
        X,
        y,
        evaluate_formula=_eval_formula,
        complexity_fn=lambda formula: 5,
        family_signature_fn=lambda formula: str(formula)[:3],
        current_best_candidate={
            "formula": "x0+x1",
            "validation_mse": 0.0,
            "validation_r2": 1.0,
            "source": "current",
        },
        max_candidates=6,  # caller asks for 6; vault should clamp to 4
    )
    assert len(proposals) <= 4


def test_s8_2_composition_rejects_over_complexity():
    rng = np.random.RandomState(3)
    X = rng.uniform(-1.0, 1.0, size=(100, 2))
    y = X[:, 0] + X[:, 1]
    vault = SpecialistVault(max_entries=4)
    # Admit specialists with normal complexity, then propose with inflated cx.
    vault.add_candidates(
        [
            {"formula": "x0", "validation_r2": 0.5, "validation_mse": 0.5, "source": "a"},
            {"formula": "x1", "validation_r2": 0.5, "validation_mse": 0.5, "source": "b"},
        ],
        X,
        y,
        evaluate_formula=_eval_formula,
        complexity_fn=lambda formula: 3,
        family_signature_fn=lambda formula: "f",
        run_index=0,
        max_new=2,
    )
    proposals = vault.propose_compositions(
        X,
        y,
        evaluate_formula=_eval_formula,
        complexity_fn=lambda formula: 50,  # all compositions over cx cap
        family_signature_fn=lambda formula: "f",
        max_complexity=28,
    )
    assert proposals == []


# ── N7: C++ weighted MAD scale parity with Python ───────────────────────────

@requires_cpp
def test_n7_cpp_huber_with_weights_runs():
    """Smoke: huber + y_weights still runs; dual metrics returned (N7 weighted MAD)."""
    rng = np.random.RandomState(0)
    x = np.linspace(-2, 2, 80)
    # core expects X as list of per-feature 1d arrays
    X_list = [x.astype(np.float64)]
    y = np.sin(x) + 0.1 * rng.randn(80)
    w = np.ones(80)
    w[-8:] = 0.05
    result = _core.run_evolution(
        X_list,
        y.astype(np.float64),
        generations=30,
        pop_size=40,
        num_islands=1,
        random_seed=7,
        loss_mode="huber",
        huber_delta=-1.0,  # auto MAD (now weight-aware)
        y_weights=w.astype(np.float64),
        timeout_seconds=15,
    )
    assert result is not None
    assert isinstance(result, dict)
    assert "formula" in result or "best_mse" in result


def test_python_mad_scale_weighted_differs_from_unweighted():
    """Sanity: weighted MAD should down-weight outlier residuals."""
    r = np.array([0.0, 0.1, -0.1, 0.05, 10.0, 12.0])
    w = np.array([1.0, 1.0, 1.0, 1.0, 0.01, 0.01])
    s_u = _mad_scale(r)
    s_w = _mad_scale(r, w)
    assert s_w < s_u  # outliers dominate unweighted MAD
