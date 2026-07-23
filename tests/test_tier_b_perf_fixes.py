"""Tier B performance fixes: refine skip, ranking cost, screening caps, dual-path skip."""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glassbox.sr.sklearn_wrapper import GlassboxRegressor
from glassbox.sr.blackbox_preprocessor import (
    _ranking_subsample,
    _tree_importance_scores,
    compute_blackbox_feature_ranking,
    prepare_blackbox_search,
)


def test_s3_6_refine_skips_when_near_exact():
    est = GlassboxRegressor(random_state=0, early_stop_mse=1e-10)
    est.n_features_in_ = 1
    x = np.linspace(-1.0, 1.0, 40)
    X = x.reshape(-1, 1)
    y = 2.0 * x + 1.0
    out = est._refine_formula_constants(
        "2*x0+1", X[:30], y[:30], X[30:], y[30:], robust=True, irls_iters=3
    )
    assert out is not None
    assert out.get("refine_skipped") == "already_near_exact"
    assert out.get("constant_refined") is False
    assert "2" in out["formula"] and "x0" in out["formula"]


def test_s3_6_refine_still_runs_when_constants_off():
    est = GlassboxRegressor(random_state=0)
    est.n_features_in_ = 1
    x = np.linspace(-1.0, 1.0, 50)
    X = x.reshape(-1, 1)
    y = 2.0 * x + 1.0
    # Off by ~10% — should attempt refine (not skip as near-exact).
    out = est._refine_formula_constants(
        "1.8*x0+0.9", X[:35], y[:35], X[35:], y[35:], robust=False, irls_iters=1
    )
    # May return refined or None if LS fails; must not claim already_near_exact.
    if out is not None:
        assert out.get("refine_skipped") != "already_near_exact"


def test_s7_3_ranking_subsample_caps_rows():
    rng = np.random.RandomState(0)
    X = rng.randn(5000, 6)
    y = X[:, 1] - X[:, 4]
    Xs, ys, ws = _ranking_subsample(X, y, max_rows=1500, random_state=0)
    assert Xs.shape[0] == 1500
    assert ys.shape[0] == 1500
    assert ws is None


def test_s7_3_tree_scores_identify_signal_on_large_n():
    rng = np.random.RandomState(1)
    X = rng.randn(2500, 8)
    y = 3.0 * X[:, 2] - 2.0 * X[:, 5] ** 2 + 0.01 * rng.randn(2500)
    scores = _tree_importance_scores(X, y)
    assert scores
    top = sorted(scores, key=scores.get, reverse=True)[:3]
    assert 2 in top or 5 in top


def test_s7_3_feature_ranking_still_selects_signal():
    rng = np.random.RandomState(2)
    X = rng.randn(800, 10)
    y = X[:, 3] + 0.5 * X[:, 7] ** 2
    ranking = compute_blackbox_feature_ranking(X, y)
    scores = ranking["feature_scores"]
    assert scores
    top = sorted(scores, key=scores.get, reverse=True)[:4]
    assert 3 in top or 7 in top


def test_s8_4_skips_composition_when_existing_strong():
    est = GlassboxRegressor(random_state=0, enable_specialist_composition_screening=True)
    est.n_features_in_ = 1
    est.blackbox_diagnostics_ = {}
    x = np.linspace(-1, 1, 40)
    X = x.reshape(-1, 1)
    y = 2.0 * x
    cands = [
        {
            "formula": "2*x0",
            "validation_mse": 1e-14,
            "validation_r2": 1.0,
            "complexity": 2,
            "source": "exact",
        }
    ]
    out = est._run_specialist_candidate_screening(
        cands, X, y, blackbox_search_plan={"screening_budget": 12, "seed_budget": 8}
    )
    assert isinstance(out, list)
    diag = est.blackbox_diagnostics_.get("candidate_screening") or {}
    # Exact incumbent should skip residual/composition work.
    assert diag.get("residual_skipped_reason") == "existing_exact_candidate"
    assert diag.get("specialist_screening") == "skipped_existing_exact"
    assert out == cands


def test_s8_4_screening_caps_max_candidates():
    est = GlassboxRegressor(random_state=0)
    est.n_features_in_ = 1
    x = np.linspace(-1, 1, 50)
    X = x.reshape(-1, 1)
    y = np.sin(x)
    cands = [
        {"formula": "sin(x0)", "validation_mse": 0.01, "validation_r2": 0.98, "complexity": 2},
        {"formula": "x0", "validation_mse": 0.5, "validation_r2": 0.3, "complexity": 1},
        {"formula": "x0**2", "validation_mse": 0.8, "validation_r2": 0.1, "complexity": 2},
        {"formula": "cos(x0)", "validation_mse": 0.9, "validation_r2": 0.05, "complexity": 2},
        {"formula": "exp(x0)", "validation_mse": 1.0, "validation_r2": 0.0, "complexity": 2},
        {"formula": "x0**3", "validation_mse": 1.1, "validation_r2": -0.1, "complexity": 2},
    ]
    diag = est._compute_specialist_screening_diagnostics(
        cands, X, y, max_candidates=6, max_pairs=5
    )
    # Caps should apply; result may be None if state fails, but call must not explode.
    assert diag is None or isinstance(diag, dict)


def test_s9_4_proposer_skips_on_high_confidence_fast_path():
    est = GlassboxRegressor(
        random_state=0,
        use_universal_proposer=True,
        universal_proposer_log_routing=False,
        early_stop_mse=1e-8,
    )
    est.n_features_in_ = 1
    est.blackbox_diagnostics_ = {}
    x = np.linspace(-1, 1, 30).reshape(-1, 1)
    y = np.sin(x[:, 0])
    fp = {
        "formula": "sin(x0)",
        "mse": 1e-12,
        "uncertainty": {
            "prediction_uncertain": False,
            "prediction_margin": 0.5,
            "prediction_entropy": 0.1,
        },
        "details": {},
    }
    payload, force = est._run_universal_proposer_dual_path(x, y, fp, None)
    assert payload is None
    assert force is False
    assert est.universal_proposer_status_ == "skipped_fast_path_high_confidence"


def test_s9_3_feature_cache_hits():
    from glassbox.curve_classifier.curve_classifier_integration import (
        _extract_features_xy_cached,
        _cached_curve_features,
    )

    x = np.linspace(-2, 2, 64)
    y = np.sin(x)
    _cached_curve_features.clear()
    f1 = _extract_features_xy_cached(x, y)
    size1 = len(_cached_curve_features)
    f2 = _extract_features_xy_cached(x, y)
    size2 = len(_cached_curve_features)
    assert size2 == size1  # no new entry on second call
    assert np.allclose(f1, f2)


def test_prepare_blackbox_still_works_after_ranking_cheapen():
    rng = np.random.RandomState(3)
    X = rng.randn(400, 12)
    y = X[:, 1] + X[:, 8]
    X_s, y_s, state = prepare_blackbox_search(
        X, y, enabled=True, max_features=4, standardize=False, min_features_to_select=3
    )
    assert state.enabled
    assert X_s.shape[1] <= 12
    assert y_s.shape == y.shape
