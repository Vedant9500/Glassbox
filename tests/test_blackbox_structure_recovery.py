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
