import sys
from pathlib import Path

import numpy as np
import pytest

from glassbox.sr.sklearn_wrapper import GlassboxRegressor


CPP_DIR = Path(__file__).resolve().parents[1] / "glassbox" / "sr" / "cpp"

from glassbox.sr.cpp import CPP_AVAILABLE, get_cpp_core

_core = get_cpp_core()


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
    # Graph soft-arithmetic can leave tiny residual blend even at high temp.
    assert abs(score["scale"] - 2.0) < 1e-3
    assert abs(score["bias"] - 0.5) < 1e-3


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
    # OOB feature refs are hard-rejected by the graph scorer.
    assert "x1" not in formulas
    # Soft graph division is finite at 0 (search-safe): 1/(x0-x0) becomes a
    # constant residual and is not a hard domain error. It may still appear as
    # a low-value candidate after affine refine; only OOB is required reject.


@requires_cpp_scorer
def test_cpp_candidate_scorer_graph_signed_sqrt_matches_search_domain():
    """Graph Power uses sign-preserving |x|^p (search domain), not exact complex sqrt.

    Ranking must accept the same domain evolution can represent.
    """
    x = np.linspace(-2.0, 2.0, 80)
    X = x.reshape(-1, 1)
    y = np.sign(x) * np.sqrt(np.abs(x))

    score = _score_one("x0^0.5", X, y)

    assert score["ok"] is True
    assert score["validation_r2"] > 0.999
    assert abs(score["scale"] - 1.0) < 1e-3



@requires_cpp_scorer
def test_cpp_candidate_scorer_weighted_affine_matches_unweighted_when_uniform():
    x = np.linspace(-2.0, 2.0, 100)
    X = x.reshape(-1, 1)
    y = 2.0 * np.sin(3.0 * x) + 0.5
    split = 75
    Xf, yf = X[:split], y[:split]
    Xv, yv = X[split:], y[split:]
    base = dict(_core.score_formula_candidates(
        ["sin(3*x0)"],
        np.ascontiguousarray(Xf, dtype=np.float64),
        np.ascontiguousarray(yf, dtype=np.float64),
        np.ascontiguousarray(Xv, dtype=np.float64),
        np.ascontiguousarray(yv, dtype=np.float64),
    )[0])
    weighted = dict(_core.score_formula_candidates(
        ["sin(3*x0)"],
        np.ascontiguousarray(Xf, dtype=np.float64),
        np.ascontiguousarray(yf, dtype=np.float64),
        np.ascontiguousarray(Xv, dtype=np.float64),
        np.ascontiguousarray(yv, dtype=np.float64),
        fit_weights=np.ones(split),
        val_weights=np.ones(len(y) - split),
    )[0])
    assert base["ok"] and weighted["ok"]
    assert abs(base["scale"] - weighted["scale"]) < 1e-9
    assert abs(base["bias"] - weighted["bias"]) < 1e-9
    assert abs(base["mse"] - weighted["mse"]) < 1e-12
    assert weighted["weighted"] is True
    assert "weighted_validation_mse" in weighted
    assert "unweighted_validation_mse" in weighted


@requires_cpp_scorer
def test_cpp_candidate_scorer_weighted_downweights_outliers():
    """Downweighting noisy fit points should recover true affine scale."""
    rng = np.random.RandomState(0)
    x = np.linspace(-2.0, 2.0, 120)
    X = x.reshape(-1, 1)
    y_true = 2.0 * x + 1.0
    y = y_true.copy()
    # Corrupt last 10 fit points heavily
    y[50:60] += 50.0
    split = 90
    Xf, yf = X[:split], y[:split]
    Xv, yv = X[split:], y[split:]
    # True structure formula is x0; affine should recover ~2x+1 without outliers
    w_fit = np.ones(split)
    w_fit[50:60] = 1e-6  # nearly drop outliers
    w_val = np.ones(len(y) - split)

    unweighted = dict(_core.score_formula_candidates(
        ["x0"],
        np.ascontiguousarray(Xf, dtype=np.float64),
        np.ascontiguousarray(yf, dtype=np.float64),
        np.ascontiguousarray(Xv, dtype=np.float64),
        np.ascontiguousarray(yv, dtype=np.float64),
    )[0])
    weighted = dict(_core.score_formula_candidates(
        ["x0"],
        np.ascontiguousarray(Xf, dtype=np.float64),
        np.ascontiguousarray(yf, dtype=np.float64),
        np.ascontiguousarray(Xv, dtype=np.float64),
        np.ascontiguousarray(yv, dtype=np.float64),
        fit_weights=w_fit,
        val_weights=w_val,
    )[0])
    assert unweighted["ok"] and weighted["ok"]
    # Weighted affine should be closer to true scale=2, bias=1
    assert abs(weighted["scale"] - 2.0) < abs(unweighted["scale"] - 2.0)
    assert abs(weighted["scale"] - 2.0) < 0.15
    assert abs(weighted["bias"] - 1.0) < 0.15
    # Primary mse is weighted; unweighted diagnostic still exposed
    assert np.isfinite(weighted["unweighted_validation_mse"])
    assert np.isfinite(weighted["weighted_validation_mse"])


@requires_cpp_scorer
def test_cpp_candidate_scorer_rejects_bad_weight_length():
    x = np.linspace(-1, 1, 40)
    X = x.reshape(-1, 1)
    y = x
    split = 30
    with pytest.raises(Exception):
        _core.score_formula_candidates(
            ["x0"],
            np.ascontiguousarray(X[:split], dtype=np.float64),
            np.ascontiguousarray(y[:split], dtype=np.float64),
            np.ascontiguousarray(X[split:], dtype=np.float64),
            np.ascontiguousarray(y[split:], dtype=np.float64),
            fit_weights=np.ones(split - 1),
            val_weights=np.ones(len(y) - split),
        )


def test_python_score_formula_candidate_is_weight_aware():
    """Python fallback path also honours fit/val weights."""
    x = np.linspace(-2.0, 2.0, 100)
    X = x.reshape(-1, 1)
    y = 2.0 * x + 1.0
    y = y.copy()
    y[40:50] += 40.0
    split = 80
    Xf, yf = X[:split], y[:split]
    Xv, yv = X[split:], y[split:]
    w_fit = np.ones(split)
    w_fit[40:50] = 1e-6
    reg = GlassboxRegressor(random_state=0)
    unweighted = reg._score_formula_candidate("x0", Xf, yf, Xv, yv)
    weighted = reg._score_formula_candidate(
        "x0", Xf, yf, Xv, yv, fit_weights=w_fit, val_weights=np.ones(len(y) - split)
    )
    assert unweighted is not None and weighted is not None
    assert abs(weighted["scale"] - 2.0) < abs(unweighted["scale"] - 2.0)
    assert weighted["weighted"] is True
    assert weighted["weighted_fit_mse"] is not None


def test_split_sample_weights_slices_fit_val():
    reg = GlassboxRegressor(random_state=0)
    n = 50
    reg.sample_weight_provided_ = True
    reg.sample_weight_ = np.arange(n, dtype=np.float64)
    split = {
        "fit_idx": np.arange(0, 40),
        "val_idx": np.arange(40, 50),
    }
    fw, vw = reg._split_sample_weights(split, n_total=n)
    assert np.allclose(fw, np.arange(0, 40))
    assert np.allclose(vw, np.arange(40, 50))
    reg.sample_weight_provided_ = False
    assert reg._split_sample_weights(split, n_total=n) == (None, None)
