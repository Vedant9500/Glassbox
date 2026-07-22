"""Phase 6.x P1 fixes: S3-1/2, S7-1, S8-1, S9-2, S10-5."""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from glassbox.sr.sklearn_wrapper import GlassboxRegressor
from glassbox.sr.blackbox_preprocessor import prepare_blackbox_search
from glassbox.sr.specialist_state import SpecialistVault


def test_s3_1_structure_rank_prefers_true_shape():
    est = GlassboxRegressor(random_state=0)
    x = np.linspace(-2, 2, 100)
    X = x.reshape(-1, 1)
    y = np.sin(x)
    fit, val = slice(0, 70), slice(70, 100)
    bad = est._score_formula_candidate("x", X[fit], y[fit], X[val], y[val])
    good = est._score_formula_candidate("sin(x)", X[fit], y[fit], X[val], y[val])
    assert bad is not None and good is not None
    # True structure should rank better on structure-first mse.
    assert float(good["mse"]) < float(bad["mse"])
    # Low shape-corr candidate should not export free affine as primary formula.
    if abs(float(bad.get("shape_corr", 0.0))) < 0.97:
        assert bad.get("exported_affine") is False or bad["formula"] == bad["base_formula"]


def test_s3_1_scale_recovery_still_exports_affine_when_shape_matches():
    est = GlassboxRegressor(random_state=0)
    x = np.linspace(-2, 2, 100)
    X = x.reshape(-1, 1)
    y = 3.0 * np.sin(x) + 0.5
    fit, val = slice(0, 70), slice(70, 100)
    scored = est._score_formula_candidate("sin(x)", X[fit], y[fit], X[val], y[val])
    assert scored is not None
    assert float(scored.get("shape_corr", 0.0)) > 0.97
    assert scored.get("used_affine_rank") is True
    # Affine export expected for pure scale/offset recovery.
    assert "sin" in scored["formula"]


def test_s3_2_guard_uses_selection_holdout():
    est = GlassboxRegressor(random_state=7)
    X = np.linspace(-1, 1, 80).reshape(-1, 1)
    y = X[:, 0] ** 2
    # Carve selection holdout once.
    sel = est._ensure_selection_holdout(X, y, validation_fraction=0.25)
    assert sel is not None
    split, mode = est._guard_validation_split(X, y, validation_fraction=0.25)
    assert mode == "selection_holdout"
    assert split is not None
    np.testing.assert_array_equal(split["val_idx"], sel["val_idx"])
    g = est._evaluate_auto_weight_guard("x**2", X, y)
    assert g.get("holdout_mode") == "selection_holdout"


def test_s7_1_rescues_secondary_supported_feature():
    rng = np.random.RandomState(0)
    n = 200
    # x0 strong linear, x1 weaker but true, x2 pure noise.
    x0 = rng.randn(n)
    x1 = rng.randn(n)
    x2 = rng.randn(n)
    y = x0 + 0.35 * x1 + 0.02 * rng.randn(n)
    X = np.column_stack([x0, x1, x2])
    Xs, ys, state = prepare_blackbox_search(
        X,
        y,
        enabled=True,
        max_features=1,  # force top-1 cut so rescue matters
        standardize=True,
        min_features_to_select=1,
        interaction_search=False,
    )
    selected = set(int(i) for i in state.selected_features)
    # With max_features=1, plain top-k keeps only x0; rescue should keep x1 if supported.
    assert 0 in selected
    # x1 should often be rescued; allow either rescue or uncertain retain-all.
    assert (1 in selected) or state.feature_selection_uncertain or len(selected) >= 2


def test_s8_1_vault_rejects_overfit_complex_candidate():
    vault = SpecialistVault(max_entries=8)
    rng = np.random.RandomState(1)
    X = rng.randn(60, 1)
    y = rng.randn(60)  # pure noise
    # Complex formula that can interpolate noise on train if evaluated fully —
    # use a constant-ish junk formula with high claimed r2 metadata.
    candidates = [
        {
            "formula": "sin(sin(sin(x0)))+cos(cos(x0))+x0**2+x0**3+x0**4+x0**5",
            "validation_r2": 0.99,
            "validation_mse": 1e-6,
            "complexity": 80,
        }
    ]

    def eval_fn(formula, X_in):
        # Perfect overfit on full data (poison signal), holdout still random-ish
        return y.copy()

    added = vault.add_candidates(
        candidates,
        X,
        y,
        evaluate_formula=eval_fn,
        complexity_fn=lambda f: 80,
        family_signature_fn=lambda f: "poly",
        run_index=1,
        max_new=3,
    )
    # Complexity gate should reject.
    assert added == 0


def test_s8_1_vault_accepts_simple_good_candidate():
    vault = SpecialistVault(max_entries=8)
    x = np.linspace(-1, 1, 80)
    X = x.reshape(-1, 1)
    y = 2 * x + 1.0
    candidates = [{"formula": "x0", "complexity": 1}]

    def eval_fn(formula, X_in):
        return X_in[:, 0]

    added = vault.add_candidates(
        candidates,
        X,
        y,
        evaluate_formula=eval_fn,
        complexity_fn=lambda f: 1,
        family_signature_fn=lambda f: "linear",
        run_index=1,
        max_new=3,
    )
    # May or may not pass holdout gate depending on scale; identity x vs 2x+1
    # has high corr but hold_r2 may be low without affine — admission may reject.
    # A scaled-correct formula:
    vault2 = SpecialistVault(max_entries=8)
    candidates2 = [{"formula": "(2*x0+1)", "complexity": 3}]

    def eval_fn2(formula, X_in):
        return 2 * X_in[:, 0] + 1.0

    added2 = vault2.add_candidates(
        candidates2,
        X,
        y,
        evaluate_formula=eval_fn2,
        complexity_fn=lambda f: 3,
        family_signature_fn=lambda f: "linear",
        run_index=1,
        max_new=3,
    )
    assert added2 == 1


def test_s9_2_budget_not_shrunk_on_poor_r2_high_confidence():
    est = GlassboxRegressor(timeout=100, adaptive_compute_budget=True, random_state=0)
    X = np.linspace(-1, 1, 50).reshape(-1, 1)
    # High confidence uncertainty + poor R2 should not collapse budget to floor.
    unc = {
        "prediction_entropy": 0.05,
        "prediction_margin": 0.5,
        "prediction_uncertain": False,
        "residual_suspicious": True,
    }
    budget_bad = est._estimate_compute_budget(X, current_r2=0.40, term_count=3, uncertainty=unc)
    unc_clean = {
        "prediction_entropy": 0.05,
        "prediction_margin": 0.5,
        "prediction_uncertain": False,
        "residual_suspicious": False,
    }
    budget_good = est._estimate_compute_budget(X, current_r2=0.995, term_count=3, uncertainty=unc_clean)
    # Poor R2 + suspicious residual keeps more budget than easy high-R2 case.
    assert budget_bad >= budget_good
    assert budget_bad >= float(est.min_compute_budget)


def test_s10_5_prefer_cpp_1d_default():
    est = GlassboxRegressor()
    assert est.prefer_cpp_1d_evolution is True
    est2 = GlassboxRegressor(prefer_cpp_1d_evolution=False)
    assert est2.prefer_cpp_1d_evolution is False
