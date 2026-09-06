"""Phase 5: physics units / dimensional analysis public API."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glassbox.sr.sklearn_wrapper import (
    GlassboxRegressor,
    _formula_unit_compatible,
    _infer_formula_units,
    _validate_physics_units,
    _validate_unit_mode,
)

cpp_dir = REPO_ROOT / "glassbox" / "sr" / "cpp"

from glassbox.sr.cpp import CPP_AVAILABLE, get_cpp_core

_core = get_cpp_core()

requires_cpp = pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ _core not built")


def test_unit_mode_validation():
    assert _validate_unit_mode("soft") == "soft"
    assert _validate_unit_mode("HARD") == "hard"
    assert _validate_unit_mode(None) == "off"
    assert _validate_unit_mode("none") == "off"
    with pytest.raises(ValueError):
        _validate_unit_mode("banana")


def test_validate_physics_units_shapes():
    # M, L, T style: mass, length, time
    iu, ou = _validate_physics_units(
        input_units=[[0, 1, 0], [0, 0, 1]],  # x0=length, x1=time
        output_units=[0, 1, -1],  # velocity L/T
        n_features=2,
    )
    assert iu == [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assert ou == [0.0, 1.0, -1.0]

    with pytest.raises(ValueError):
        _validate_physics_units([[1]], [1, 0], n_features=1)  # dim mismatch
    with pytest.raises(ValueError):
        _validate_physics_units([[1], [0]], [1], n_features=1)  # feature count
    with pytest.raises(ValueError):
        _validate_physics_units([[1]], None, n_features=1)


def test_infer_formula_units_basic():
    # x0 length [0,1,0], x1 time [0,0,1]
    iu = [[0, 1, 0], [0, 0, 1]]
    ou = [0, 1, -1]
    ok = _infer_formula_units("x0/x1", iu, ou)
    assert ok["ok"] is True
    assert abs(ok["units"][1] - 1.0) < 1e-9
    assert abs(ok["units"][2] + 1.0) < 1e-9
    assert ok["penalty"] < 1e-9

    bad = _infer_formula_units("x0+x1", iu, ou)
    assert bad["ok"] is True
    assert bad["penalty"] > 0.0  # add mismatch + output mismatch

    trig = _infer_formula_units("sin(x0)", iu, [0, 0, 0])
    assert trig["ok"] is True
    assert trig["penalty"] > 0.0  # sin of length


def test_hard_filter_rejects_unphysical():
    iu = [[0, 1, 0], [0, 0, 1]]
    ou = [0, 1, -1]
    ok, info = _formula_unit_compatible("x0/x1", iu, ou, unit_mode="hard")
    assert ok is True
    assert info["penalty"] < 1e-6

    bad, info2 = _formula_unit_compatible("x0+x1", iu, ou, unit_mode="hard")
    assert bad is False
    assert info2["penalty"] > 0.0

    # soft never rejects
    soft_ok, _ = _formula_unit_compatible("x0+x1", iu, ou, unit_mode="soft")
    assert soft_ok is True


def test_estimator_units_inactive_by_default():
    est = GlassboxRegressor(random_state=0)
    est._activate_physics_units(2)
    assert est.units_active_ is False
    assert est.physics_constrained_ is False
    assert est._evolution_units_kwargs() == {}


def test_estimator_auto_soft_when_units_provided():
    est = GlassboxRegressor(
        random_state=0,
        input_units=[[1.0], [0.0]],
        output_units=[1.0],
        # unit_mode left at default "off" → auto soft
    )
    est._activate_physics_units(2)
    assert est.units_active_ is True
    assert est.physics_constrained_ is True
    # §3.203: public param stays "off" (sklearn clone contract); auto-soft
    # lives on the fitted effective_unit_mode_ attribute.
    assert est.unit_mode == "off"
    assert est.effective_unit_mode_ == "soft"
    kw = est._evolution_units_kwargs()
    assert kw["input_units"] == [[1.0], [0.0]]
    assert kw["output_units"] == [1.0]
    assert "dim_penalty_weight" in kw
    # Soft floor: default 0.1 constructor value becomes >= 2.0 when units active.
    assert kw["dim_penalty_weight"] >= 2.0


def test_filter_candidates_hard_mode():
    est = GlassboxRegressor(
        random_state=0,
        input_units=[[0, 1, 0], [0, 0, 1]],
        output_units=[0, 1, -1],
        unit_mode="hard",
        dim_penalty_weight=1.0,
    )
    est.n_features_in_ = 2
    est.blackbox_diagnostics_ = {}
    est._activate_physics_units(2)
    cands = [
        {"formula": "x0/x1", "mse": 1.0, "complexity": 3},
        {"formula": "x0+x1", "mse": 0.01, "complexity": 3},  # unphysical but low MSE
        {"formula": "sin(x0)", "mse": 0.5, "complexity": 2},
    ]
    kept = est._filter_candidates_by_units(cands)
    formulas = [c["formula"] for c in kept]
    assert "x0/x1" in formulas
    assert "x0+x1" not in formulas
    assert est.blackbox_diagnostics_["unit_filter"]["rejected"] >= 1


@requires_cpp
def test_cpp_dim_penalty_weight_and_units_run():
    rng = np.random.default_rng(0)
    t = rng.uniform(0.5, 2.0, size=80)
    L = 2.0 * t  # not physical; just need run
    X = np.column_stack([L, t])
    y = L / t
    X_list = [X[:, 0], X[:, 1]]
    result = _core.run_evolution(
        X_list,
        y,
        pop_size=12,
        generations=6,
        early_stop_mse=1e-8,
        random_seed=1,
        input_units=[[0, 1, 0], [0, 0, 1]],
        output_units=[0, 1, -1],
        dim_penalty_weight=5.0,
    )
    assert "formula" in result
    assert np.isfinite(result.get("best_mse", float("inf")))


@requires_cpp
def test_cpp_no_units_unchanged_api():
    x = np.linspace(-1, 1, 40)
    y = 2 * x + 1
    result = _core.run_evolution(
        [x], y, pop_size=10, generations=5, early_stop_mse=1e-8, random_seed=2
    )
    assert "formula" in result
