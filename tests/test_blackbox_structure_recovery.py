"""P0 multi-var structure recovery: templates + seed skeletons."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from glassbox.sr.blackbox_preprocessor import (
    build_blackbox_seed_formulas,
    build_search_space_structure_seeds,
)
from scripts.classifier_fast_path import _maybe_match_easy_multivariate_formula


def test_multivariate_templates_match_pagie_vlad_feynman():
    rng = np.random.RandomState(0)

    X = rng.uniform(0.1, 5.0, size=(220, 2))
    y = 1.0 / (1.0 + X[:, 0] ** (-4)) + 1.0 / (1.0 + X[:, 1] ** (-4))
    match = _maybe_match_easy_multivariate_formula(X, y)
    assert match is not None
    formula, mse, details = match
    assert mse < 1e-10
    assert "pagie" in str(details.get("template_match", ""))

    X5 = rng.uniform(0.05, 6.05, size=(400, 5))
    y5 = 10.0 / (5.0 + np.sum((X5 - 3.0) ** 2, axis=1))
    match5 = _maybe_match_easy_multivariate_formula(X5, y5)
    assert match5 is not None
    assert match5[1] < 1e-10
    assert "vlad" in str(match5[2].get("template_match", ""))

    X3 = rng.uniform(0.1, 5.0, size=(220, 3))
    y3 = X3[:, 0] * X3[:, 1] / (4.0 * np.pi * X3[:, 2] ** 2)
    match3 = _maybe_match_easy_multivariate_formula(X3, y3)
    assert match3 is not None
    assert match3[1] < 1e-10
    assert "product_over_square" in str(match3[2].get("template_match", ""))


def test_structure_seeds_present_for_five_feature_problem():
    formulas = build_blackbox_seed_formulas([0, 1, 2, 3, 4], max_seeds=40)
    joined = " ".join(formulas)
    assert any("1/(1+x0" in f for f in formulas)
    assert "10/(5+" in joined or "1/(5+" in joined
    assert any("/x" in f and "^2" in f for f in formulas)


def test_search_space_structure_seeds_exist():
    seeds = build_search_space_structure_seeds(5, max_seeds=40)
    joined = " ".join(seeds)
    assert any("x0^4" in s or "x0**4" in s or "x0^2" in s or "x0+0" in s for s in seeds)
    assert "1.1/(1.1+" in joined or "5.1/(5.1+" in joined or "1/(1+" in joined
    assert any("/x" in s and "^2" in s for s in seeds) or any("x0*x1" in s for s in seeds)


def test_structure_probe_is_seed_only_not_auto_win():
    """Original-space templates must not early-exit / auto-win blackbox fit."""
    from glassbox.sr.sklearn_wrapper import GlassboxRegressor

    rng = np.random.RandomState(0)
    X = rng.uniform(0.05, 6.05, size=(300, 5))
    y = 10.0 / (5.0 + np.sum((X - 3.0) ** 2, axis=1))
    est = GlassboxRegressor(
        blackbox_mode=True,
        blackbox_min_features_to_select=2,
        blackbox_max_features=5,
        timeout=25,
        random_state=0,
        use_fast_path=True,
    )
    est.fit(X, y)
    diag = getattr(est, "blackbox_diagnostics_", {}) or {}
    probe = diag.get("structure_probe_original") or {}
    # Probe may still detect the family for diagnostics.
    if probe:
        assert probe.get("auto_win") is False
        assert probe.get("role") == "seed_candidate_only"
    # Must not take the structure-probe early-exit track.
    track = str(getattr(est, "specialist_track_", "") or "")
    assert "structure_probe" not in track
    assert diag.get("specialist_skipped_reason") not in {
        "structure_probe_exact",
        "structure_probe_robust",
    }
    # Search-space structure seeds should be fitted and recorded.
    ss = diag.get("search_space_structure_seeds") or {}
    assert ss.get("auto_win") is False
    assert int(ss.get("n_scored") or 0) >= 1


def test_original_space_free_const_exact_under_outliers():
    """IRLS free-const + polish recovers Exact clean structure under 3% spikes."""
    from glassbox.sr.sklearn_wrapper import (
        GlassboxRegressor,
        _soft_mad_sample_weights,
        _validate_sample_weight,
    )

    def _with_outliers(y_clean, seed=11, frac=0.03):
        rng = np.random.RandomState(seed)
        y = np.asarray(y_clean, dtype=np.float64).copy()
        n_out = max(1, int(round(frac * len(y))))
        idx = rng.choice(len(y), size=n_out, replace=False)
        y[idx] += rng.normal(0.0, 3.0 * (float(np.std(y_clean)) + 1e-12), size=n_out)
        return y

    def _recover(X, y_clean):
        y = _with_outliers(y_clean)
        est = GlassboxRegressor(blackbox_mode=True, timeout=5, random_state=11)
        est.blackbox_diagnostics_ = {}
        w = _soft_mad_sample_weights(y)
        est.sample_weight_ = _validate_sample_weight(w, len(y))
        est.sample_weight_provided_ = True

        class _S:
            pass

        state = _S()
        state.selected_features = list(range(X.shape[1]))
        state.enabled = True
        state.standardized = False
        winner = est._fit_original_space_structure_winner(X, y, state)
        assert winner is not None and winner.get("formula")
        formula, _ = est._polish_original_space_structure_formula(winner["formula"], X, y)
        pred = est._safe_eval_formula_array(formula, X)
        return float(np.mean((np.asarray(pred) - y_clean) ** 2)), formula

    rng = np.random.RandomState(11)
    Xv = rng.uniform(0.05, 6.05, size=(400, 5))
    yv = 10.0 / (5.0 + np.sum((Xv - 3.0) ** 2, axis=1))
    mse_v, _ = _recover(Xv, yv)
    assert mse_v < 1e-6

    Xp = rng.uniform(0.1, 5.0, size=(220, 2))
    yp = 1.0 / (1.0 + Xp[:, 0] ** (-4)) + 1.0 / (1.0 + Xp[:, 1] ** (-4))
    mse_p, _ = _recover(Xp, yp)
    assert mse_p < 1e-6

    Xf = rng.uniform(0.1, 5.0, size=(220, 3))
    yf = Xf[:, 0] * Xf[:, 1] / (4.0 * np.pi * Xf[:, 2] ** 2)
    mse_f, _ = _recover(Xf, yf)
    assert mse_f < 1e-6
