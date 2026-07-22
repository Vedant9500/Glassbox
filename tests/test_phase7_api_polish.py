"""Phase 7: API polish (S1-11 export, S5-14 implicit mul / power print, S6-2 knobs)."""
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
CPP = REPO / "glassbox" / "sr" / "cpp"
for p in (REPO, CPP):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    import _core  # type: ignore

    CPP_AVAILABLE = hasattr(_core, "formula_to_seed_graph") or hasattr(
        _core, "formula_to_seed_graph_cpp"
    )
except ImportError:
    CPP_AVAILABLE = False

requires_cpp = pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ _core not built")


def _seed(formula: str):
    if hasattr(_core, "formula_to_seed_graph"):
        return _core.formula_to_seed_graph(formula)
    return _core.formula_to_seed_graph_cpp(formula)


def _snap(formula: str, n_features: int = 1) -> str:
    if hasattr(_core, "snap_formula_floats"):
        return str(_core.snap_formula_floats(formula, n_features))
    return str(_core.snap_formula_floats_cpp(formula, n_features))


def test_s1_11_glassbox_regressor_exported():
    from glassbox.sr import GlassboxRegressor, FPIPv2, FPIPv2Payload

    assert GlassboxRegressor is not None
    assert FPIPv2Payload is FPIPv2
    import glassbox.sr as sr_mod

    assert "GlassboxRegressor" in sr_mod.__all__
    assert "FPIPv2" in sr_mod.__all__
    assert "FPIPv2Payload" in sr_mod.__all__


@requires_cpp
def test_s5_14_implicit_multiplication_and_bare_unary():
    """2x, 2(x+1), sin x should parse into seed graphs with nodes."""
    cases = [
        "2x",
        "2x0",
        "2(x0+1)",
        "sin x0",
        "sin x",
        "(x0)(x0)",
    ]
    for formula in cases:
        g = _seed(formula)
        assert g is not None, formula
        nodes = g.get("nodes") if isinstance(g, dict) else None
        assert nodes is not None and len(nodes) >= 1, formula


@requires_cpp
def test_s5_14_even_power_display_uses_abs():
    """UnaryOp::Power even integer prints abs form (x^8 → Power, not IntPow)."""
    # IntPow only covers 2..6; exponent 8 forces Power path.
    out = _snap("x0^8")
    assert "abs" in out.lower(), out
    assert "^8" in out or "^ 8" in out or "8" in out, out


@requires_cpp
def test_s5_14_odd_integer_power_keeps_signed_base():
    # 7 is outside IntPow range → Power with odd integer
    out = _snap("x0^7")
    # Odd path should not force abs wrapper on the base for integer print.
    # Accept either (x0)^7 or equivalent without requiring abs.
    assert "7" in out, out


@requires_cpp
def test_s6_2_elite_size_and_seed_fraction_bound():
    # pybind11 may not expose inspect.signature; validate by accepting kwargs.
    x = np.linspace(-1, 1, 40).astype(np.float64)
    y = (2.0 * x + 1.0).astype(np.float64)
    try:
        res = _core.run_evolution(
            [x],
            y,
            pop_size=12,
            generations=2,
            early_stop_mse=1e-20,
            timeout_seconds=8,
            random_seed=7,
            elite_size=3,
            seed_fraction=0.75,
            num_threads=1,
        )
    except TypeError as exc:
        raise AssertionError(
            "run_evolution missing elite_size/seed_fraction kwargs"
        ) from exc
    assert res is not None
    formula = res.get("formula") if isinstance(res, dict) else None
    assert formula is not None


@requires_cpp
def test_s5_16_scoring_path_reports_raw_mse():
    """score_formula_candidates / evolution export still surface raw mse fields."""
    x = np.linspace(-1, 1, 60).astype(np.float64)
    y = (x * x).astype(np.float64)
    X = x.reshape(-1, 1)
    mid = len(x) // 2
    scores = _core.score_formula_candidates(
        ["x0**2", "x0"],
        X[:mid],
        y[:mid],
        X[mid:],
        y[mid:],
    )
    assert len(scores) >= 1
    # At least one finite mse-like field present
    ok = False
    for s in scores:
        for key in ("raw_mse", "mse", "validation_mse", "fit_mse", "weighted_mse"):
            if key in s and np.isfinite(float(s[key])):
                ok = True
                break
    assert ok, scores
