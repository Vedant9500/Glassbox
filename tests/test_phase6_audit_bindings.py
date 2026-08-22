"""Phase 6 audit checks: S6 seed-graph plumbing + basic FPIP validation."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
CPP_DIR = REPO / "glassbox" / "sr" / "cpp"
for p in (REPO, CPP_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from glassbox.sr.cpp import CPP_AVAILABLE, get_cpp_core

_core = get_cpp_core()

requires_cpp = pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ _core not built")


@requires_cpp
def test_s6_seed_graph_short_weights_still_usable():
    """Short output_weights must not create a dead seed (S6 normalize)."""
    x = np.linspace(-1, 1, 48)
    y = 2.0 * x + 0.5
    # Minimal input graph: one Input node, empty/short weights.
    seed = {
        "nodes": [
            {
                "type": 0,  # Input
                "feature_idx": 0,
                "value": 0.0,
                "unary_op": 0,
                "binary_op": 0,
                "p": 1.0,
                "omega": 1.0,
                "phi": 0.0,
                "amplitude": 1.0,
                "beta": 1.5,
                "gamma": 1.0,
                "tau": 1.0,
                "left_child": -1,
                "right_child": -1,
            }
        ],
        "output_weights": [],  # intentionally empty
        "output_bias": 0.0,
    }
    res = _core.run_evolution(
        X_list=[x.astype(np.float64)],
        y=y.astype(np.float64),
        pop_size=16,
        generations=8,
        timeout_seconds=5,
        num_islands=1,
        random_seed=3,
        seed_graphs_py=[seed],
    )
    assert res is not None
    assert int(res.get("seed_graphs_used", 0)) >= 1
    assert np.isfinite(float(res.get("best_mse", np.nan)))


@requires_cpp
def test_s6_run_evolution_defaults_accept_y_weights_and_loss_mode():
    x = np.linspace(-1, 1, 40)
    y = x**2
    w = np.ones_like(y)
    w[:3] = 0.2
    res = _core.run_evolution(
        X_list=[x.astype(np.float64)],
        y=y.astype(np.float64),
        pop_size=12,
        generations=5,
        timeout_seconds=4,
        num_islands=1,
        random_seed=1,
        y_weights=w.astype(np.float64),
        loss_mode="huber",
        huber_delta=0.5,
    )
    assert res.get("weighted") is True
    assert "huber" in str(res.get("loss_mode", "")).lower()
    assert np.isfinite(float(res.get("best_mse", np.nan)))


def test_s10_fpip_v2_validator_rejects_bad_payload():
    from glassbox.sr.fpip_v2 import validate_fpip_v2_payload

    ok, errors = validate_fpip_v2_payload({"schema_version": "nope"})
    assert ok is False
    assert errors


def test_s7_remap_roundtrip_indices():
    from glassbox.sr.blackbox_preprocessor import (
        remap_original_formula_to_reduced,
        remap_reduced_formula_to_original,
    )

    selected = [2, 5, 7]
    reduced = "x0 + 2*x1 - x2"
    original = remap_reduced_formula_to_original(reduced, selected)
    assert "x2" in original and "x5" in original and "x7" in original
    back = remap_original_formula_to_reduced(original, selected)
    assert back.replace(" ", "") == reduced.replace(" ", "")
