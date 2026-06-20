import sys
from pathlib import Path

import numpy as np
import pytest

from glassbox.sr.sklearn_wrapper import GlassboxRegressor


CPP_DIR = Path(__file__).resolve().parents[1] / "glassbox" / "sr" / "cpp"
if str(CPP_DIR) not in sys.path:
    sys.path.insert(0, str(CPP_DIR))

try:
    import _core  # type: ignore

    CPP_AVAILABLE = hasattr(_core, "score_formula_candidates")
except ImportError:
    CPP_AVAILABLE = False


requires_cpp_scorer = pytest.mark.skipif(
    not CPP_AVAILABLE,
    reason="C++ candidate scorer is not built",
)


def _score_one(formula, X, y):
    split = int(round(0.75 * len(y)))
    scores = _core.score_formula_candidates(
        [formula],
        np.ascontiguousarray(X[:split], dtype=np.float64),
        np.ascontiguousarray(y[:split], dtype=np.float64),
        np.ascontiguousarray(X[split:], dtype=np.float64),
        np.ascontiguousarray(y[split:], dtype=np.float64),
        2,
    )
    return dict(scores[0])


@requires_cpp_scorer
def test_cpp_candidate_scorer_recovers_affine_scale_and_bias():
    x = np.linspace(-2.0, 2.0, 80)
    X = x.reshape(-1, 1)
    y = 2.0 * np.sin(3.0 * x) + 0.5

    score = _score_one("sin(3*x0)", X, y)

    assert score["ok"] is True
    assert score["validation_r2"] > 0.999
    assert abs(score["scale"] - 2.0) < 1e-9
    assert abs(score["bias"] - 0.5) < 1e-9


@requires_cpp_scorer
def test_cpp_candidate_scorer_rejects_out_of_range_feature_reference():
    x = np.linspace(-2.0, 2.0, 80)
    X = x.reshape(-1, 1)
    y = np.sin(x)

    score = _score_one("x1", X, y)

    assert score["ok"] is False
    assert "feature" in str(score["error"]).lower() or "symbol" in str(score["error"]).lower()


@requires_cpp_scorer
def test_refine_candidate_formulas_does_not_keep_invalid_cpp_candidates():
    x = np.linspace(-2.0, 2.0, 80)
    X = x.reshape(-1, 1)
    y = np.sin(x)
    reg = GlassboxRegressor(random_state=0)

    refined = reg._refine_candidate_formulas(
        [
            {"formula": "x1"},
            {"formula": "1/(x0-x0)"},
        ],
        X,
        y,
        max_candidates=4,
    )

    formulas = [str(item.get("base_formula") or item.get("formula")) for item in refined]
    assert "x1" not in formulas
    assert "1/(x0-x0)" not in formulas


@requires_cpp_scorer
def test_cpp_candidate_scorer_rejects_unprotected_fractional_power_on_negative_domain():
    x = np.linspace(-2.0, 2.0, 80)
    X = x.reshape(-1, 1)
    y = np.sign(x) * np.sqrt(np.abs(x))

    score = _score_one("x0^(1/2)", X, y)

    assert score["ok"] is False
    assert "power" in str(score["error"]).lower() or "domain" in str(score["error"]).lower()
