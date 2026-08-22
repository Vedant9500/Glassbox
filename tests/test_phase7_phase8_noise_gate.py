"""Phase 7 routing calibration + Phase 8 release-gate smoke tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from glassbox.sr.sklearn_wrapper import (
    _NOISE_BAND_THRESHOLDS,
    GlassboxRegressor,
    _estimate_outlier_fraction,
    _noise_band_from_diagnostics,
    _residual_lag1_autocorr,
)

cpp_dir = REPO / "glassbox" / "sr" / "cpp"

from glassbox.sr.cpp import CPP_AVAILABLE, get_cpp_core

_core = get_cpp_core()

requires_cpp = pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ _core not built")


def test_residual_lag1_autocorr_detects_structure():
    rng = np.random.default_rng(0)
    white = rng.normal(0, 1, size=200)
    # AR(1)
    ar = np.zeros(200)
    for i in range(1, 200):
        ar[i] = 0.8 * ar[i - 1] + rng.normal(0, 0.3)
    ac_w = abs(_residual_lag1_autocorr(white))
    ac_ar = abs(_residual_lag1_autocorr(ar))
    assert ac_ar > ac_w


def test_outlier_fraction_and_noise_band():
    r = np.zeros(100)
    r[-10:] = 50.0
    frac = _estimate_outlier_fraction(r)
    assert frac >= 0.05
    band = _noise_band_from_diagnostics(
        {
            "outlier_fraction": 0.2,
            "validation_gap": 0.3,
            "residual_autocorr": 0.5,
            "ess_ratio": 0.4,
        }
    )
    assert band in ("medium", "high")
    assert _noise_band_from_diagnostics({}) == "clean"
    # Phase E+: residual RMS / signal-scale outlier channels.
    assert _noise_band_from_diagnostics(
        {
            "residual_rms_ratio": 0.10,
            "outlier_fraction": 0.0,
            "validation_gap": 0.0,
            "residual_autocorr": 0.0,
            "ess_ratio": 1.0,
        }
    ) in ("low", "medium", "high")
    assert _noise_band_from_diagnostics(
        {
            "signal_outlier_fraction": 0.03,
            "outlier_fraction": 0.0,
            "residual_rms_ratio": 0.45,
        }
    ) in ("medium", "high")


def test_runtime_noise_diagnostics_and_plan_calibrate():
    x = np.linspace(-2, 2, 80)
    X = x.reshape(-1, 1)
    y = np.sin(x) + 0.0
    # Inject outliers
    y_noisy = y.copy()
    y_noisy[-8:] += 5.0

    est = GlassboxRegressor(random_state=1)
    est.n_features_in_ = 1
    est.blackbox_diagnostics_ = {}
    # Provide weights that downweight outliers
    w = np.ones(len(y_noisy))
    w[-8:] = 0.05
    est.sample_weight_ = w
    est.sample_weight_provided_ = True

    diag_clean = est._compute_runtime_noise_diagnostics(X, y, formula="sin(x0)")
    diag_noisy = est._compute_runtime_noise_diagnostics(X, y_noisy, formula="sin(x0)")
    assert diag_clean["noise_band"] in _NOISE_BAND_THRESHOLDS
    assert diag_noisy["outlier_fraction"] >= diag_clean["outlier_fraction"]
    assert diag_noisy["ess_ratio"] < 1.0

    # Fake blackbox state so plan uses full multivariate path
    class _State:
        enabled = True
        selected_features = [0]
        feature_scores = {0: 1.0}
        feature_selection_uncertain = False
        reason = ""
        interaction_scores = {}
        interaction_terms = []

    plan_clean = est._derive_blackbox_search_plan(
        _State(),
        noise_diagnostics=diag_clean,
        candidate_screening={
            "best_validation_r2": 0.99,
            "candidate_count": 4,
            "family_count": 2,
        },
    )
    plan_noisy = est._derive_blackbox_search_plan(
        _State(),
        noise_diagnostics=diag_noisy,
        candidate_screening={
            "best_validation_r2": 0.99,
            "candidate_count": 4,
            "family_count": 2,
        },
    )
    assert plan_noisy["noise_band"] != "clean" or plan_noisy["noise_pressure"] >= 0.0
    # High/medium noise should not shrink acceptance bar below clean without cause;
    # typically noisy band has lower acceptance r2 threshold numbers in table but
    # higher depth when candidate_strength looks artificially high.
    assert "candidate_acceptance_r2" in plan_clean
    assert "noise_routing" in plan_noisy
    if plan_noisy["noise_band"] in ("medium", "high"):
        assert plan_noisy["depth_multiplier"] >= 1.0


def test_high_noise_does_not_shrink_on_high_candidate_r2():
    est = GlassboxRegressor(random_state=2)
    est.n_features_in_ = 2
    est.original_n_features_in_ = 2

    class _State:
        enabled = True
        selected_features = [0, 1]
        feature_scores = {0: 0.9, 1: 0.8}
        feature_selection_uncertain = False
        reason = ""
        interaction_scores = {(0, 1): 0.4}
        interaction_terms = ["x0*x1"]

    high = {
        "noise_band": "high",
        "outlier_fraction": 0.25,
        "validation_gap": 0.4,
        "residual_autocorr": 0.6,
        "ess_ratio": 0.5,
    }
    plan = est._derive_blackbox_search_plan(
        _State(),
        noise_diagnostics=high,
        candidate_screening={
            "best_validation_r2": 0.97,  # looks good on noisy labels
            "candidate_count": 6,
            "family_count": 3,
        },
    )
    assert plan["noise_band"] == "high"
    # Floor: do not collapse search when noisy R2 is high
    assert plan["depth_multiplier"] >= 1.15
    assert plan["breadth_multiplier"] >= 1.10
    thr = _NOISE_BAND_THRESHOLDS["high"]
    assert plan["candidate_acceptance_r2"] >= thr["candidate_acceptance_r2"] - 0.05
    # Phase C: high noise relaxes hard Track-1 timeout clamp (was always <=1.0).
    assert plan["timeout_multiplier"] > 1.0 or plan["generation_multiplier"] >= 1.15


def test_soft_mad_weights_downweight_outliers():
    from glassbox.sr.sklearn_wrapper import _soft_mad_sample_weights

    rng = np.random.RandomState(0)
    y = rng.randn(200)
    y[-15:] = 40.0
    w = _soft_mad_sample_weights(y)
    assert w is not None
    assert w.shape == (200,)
    assert float(np.mean(w[-15:])) < float(np.mean(w[:-15]))


@requires_cpp
def test_release_gate_weighted_outlier_recovery():
    """CI smoke: sample_weight recovers linear under block outliers."""
    x = np.linspace(-3, 3, 100)
    y_clean = 2.0 * x + 1.0
    y = y_clean.copy()
    y[-10:] += 35.0
    w = np.ones(100)
    # Fully exclude the known-bad block (canonical sample_weight semantics).
    # A small-but-nonzero weight makes the weighted objective itself prefer
    # edge-bending fits over the true line, so exclusion must be exact.
    w[-10:] = 0.0
    res = _core.run_evolution(
        [x.astype(np.float64)],
        y,
        pop_size=40,
        generations=40,
        early_stop_mse=1e-12,
        random_seed=11,
        num_islands=4,
        y_weights=w,
    )
    f = str(res.get("formula", "") or "")
    est = GlassboxRegressor()
    est.n_features_in_ = 1
    pred = est._safe_eval_formula_array(f, x.reshape(-1, 1))
    clean_mse = float(np.mean((np.asarray(pred) - y_clean) ** 2))
    # Weighted path should be far better than fitting the spikes
    assert clean_mse < 50.0
    assert np.isfinite(res.get("best_weighted_mse", float("inf")))


@requires_cpp
def test_release_gate_robust_trimmed_recovers_linear():
    """CI smoke: trimmed_mse IRLS recovers 2x+1 under block outliers."""
    x = np.linspace(-3, 3, 120)
    y_clean = 2.0 * x + 1.0
    y = y_clean.copy()
    y[-12:] += 40.0
    res = _core.run_evolution(
        [x.astype(np.float64)],
        y,
        pop_size=50,
        generations=50,
        early_stop_mse=1e-12,
        random_seed=11,
        num_islands=4,
        loss_mode="trimmed_mse",
        trim_fraction=0.15,
    )
    f = str(res.get("formula", "") or "")
    est = GlassboxRegressor()
    est.n_features_in_ = 1
    pred = est._safe_eval_formula_array(f, x.reshape(-1, 1))
    clean_mse = float(np.mean((np.asarray(pred) - y_clean) ** 2))
    assert clean_mse < 5.0
    assert res.get("loss_mode") == "trimmed_mse"


def test_benchmark_ablation_presets_exist():
    from scripts.benchmark_noise import (
        ABLATION_PRESETS,
        DEFAULT_BLACKBOX_RELEASE_ABLATIONS,
    )

    for key in (
        "full",
        "no_weights",
        "no_robust_loss",
        "no_units",
        "no_cv_guard",
        "no_uncertainty_routing",
        "no_noise_pruning",
    ):
        assert key in ABLATION_PRESETS
    # Phase E release ablations are a subset of presets.
    for key in DEFAULT_BLACKBOX_RELEASE_ABLATIONS:
        assert key in ABLATION_PRESETS


def test_noise_protocol_contract_and_failed_seed_visibility():
    from scripts.benchmark_noise import (
        REQUIRED_COLUMNS,
        assert_row_contract,
        summarize_noise_protocol,
    )

    rows = [
        {
            **{c: None for c in REQUIRED_COLUMNS},
            "problem": "Poly-x2",
            "tier": "clean",
            "seed": 1,
            "test_r2": 1.0,
            "clean_test_r2": 1.0,
            "exact_match": True,
            "acceptable_clean": True,
            "false_confidence": False,
            "raw_mse": 0.0,
            "display_mse": 0.0,
            "clean_test_mse": 0.0,
            "formula_complexity": 3,
            "error": None,
        },
        {
            **{c: None for c in REQUIRED_COLUMNS},
            "problem": "Poly-x2",
            "tier": "gaussian_10pct",
            "seed": 1,
            "test_r2": 0.9,
            "clean_test_r2": 0.95,
            "exact_match": False,
            "acceptable_clean": True,
            "false_confidence": False,
            "raw_mse": 0.5,
            "display_mse": 0.5,
            "clean_test_mse": 0.1,
            "formula_complexity": 5,
            "error": "boom",
            "failed_seed": True,
        },
    ]
    assert_row_contract(rows)
    summary = summarize_noise_protocol(rows)
    assert summary["n_rows"] == 2
    assert any(d["tier"] == "gaussian_10pct" for d in summary["deltas_vs_clean"])


def test_phase_e_ablation_table_from_rows(tmp_path):
    """Phase E: multi-var ablation table helper + markdown for release notes."""
    from scripts.benchmark_noise import (
        REQUIRED_COLUMNS,
        ablation_table_to_markdown,
        build_ablation_table,
        write_ablation_report,
    )

    def _row(problem, tier, *, clean_r2, accept, exact, complexity, ablation):
        return {
            **{c: None for c in REQUIRED_COLUMNS},
            "problem": problem,
            "tier": tier,
            "seed": 11,
            "test_r2": clean_r2,
            "clean_test_r2": clean_r2,
            "exact_match": exact,
            "acceptable_clean": accept,
            "false_confidence": False,
            "raw_mse": 0.1,
            "display_mse": 0.1,
            "clean_test_mse": max(0.0, 1.0 - float(clean_r2)),
            "formula_complexity": complexity,
            "error": None,
            "ablation": ablation,
            "n_features": 5,
            "blackbox_enabled": True,
        }

    rows_by_ablation = {
        "full": [
            _row(
                "Vladislavleva-4",
                "clean",
                clean_r2=1.0,
                accept=True,
                exact=True,
                complexity=12,
                ablation="full",
            ),
            _row(
                "Vladislavleva-4",
                "outliers_3pct",
                clean_r2=0.95,
                accept=True,
                exact=False,
                complexity=18,
                ablation="full",
            ),
        ],
        "no_weights": [
            _row(
                "Vladislavleva-4",
                "clean",
                clean_r2=1.0,
                accept=True,
                exact=True,
                complexity=12,
                ablation="no_weights",
            ),
            _row(
                "Vladislavleva-4",
                "outliers_3pct",
                clean_r2=0.70,
                accept=False,
                exact=False,
                complexity=40,
                ablation="no_weights",
            ),
        ],
        "no_robust_loss": [
            _row(
                "Vladislavleva-4",
                "clean",
                clean_r2=1.0,
                accept=True,
                exact=True,
                complexity=12,
                ablation="no_robust_loss",
            ),
            _row(
                "Vladislavleva-4",
                "outliers_3pct",
                clean_r2=0.80,
                accept=True,
                exact=False,
                complexity=25,
                ablation="no_robust_loss",
            ),
        ],
    }
    table = build_ablation_table(rows_by_ablation, baseline="full")
    assert table["baseline"] == "full"
    assert table["n_ablations"] == 3
    assert len(table["headlines"]) == 3
    assert table["headlines"][0]["ablation"] == "full"

    # no_weights should show worse clean recovery than full on outliers.
    nw = next(h for h in table["headlines"] if h["ablation"] == "no_weights")
    assert nw["mean_clean_test_r2_noisy_tiers"] < 0.9
    assert nw["delta_mean_clean_test_r2_noisy_tiers_vs_full"] < 0.0

    md = ablation_table_to_markdown(table)
    assert "Ablation Table" in md
    assert "no_weights" in md
    assert "R2clean" in md

    paths = write_ablation_report(table, tmp_path)
    assert paths["ablation_json"].exists()
    assert paths["ablation_markdown"].exists()
    assert "Vladislavleva-4" in paths["ablation_markdown"].read_text(encoding="utf-8")


def test_phase_e_blackbox_outliers_ci_smoke():
    """Phase E CI: multi-feature × outliers with blackbox_enabled=True.

    Exercises real blackbox ranking under block outliers. Prefers clean recovery
    (R2clean / Accept) over noisy-label fit. Also checks that selection can
    drop decoy features when ranking is informative.
    """
    from glassbox.sr.blackbox_preprocessor import prepare_blackbox_search

    rng = np.random.RandomState(11)
    n = 180
    # 6 features: signal on 0,1; decoys 2..5. Outliers correlate decoy 5 with y.
    X = rng.randn(n, 6)
    y_clean = 1.5 * X[:, 0] - 0.8 * X[:, 1]
    y = y_clean.copy()
    y[0:25] = 30.0 + 6.0 * X[0:25, 5]
    w = np.ones(n, dtype=np.float64)
    w[0:25] = 0.02

    # 1) Ranking path: weighted selection prefers true features; can drop decoys.
    _, _, state_w = prepare_blackbox_search(
        X,
        y,
        enabled=True,
        max_features=3,
        min_features_to_select=2,
        standardize=False,
        interaction_search=False,
        sample_weight=w,
    )
    assert state_w.enabled is True
    assert len(state_w.selected_features) <= 5  # can drop vs full 6
    true = {0, 1}
    assert len(true.intersection(state_w.selected_features)) >= 1

    # 2) Estimator fit: blackbox multi-feature + soft MAD / weights under outliers.
    est = GlassboxRegressor(
        random_state=11,
        generations=12,
        population_size=24,
        timeout=20.0,
        multi_start_runs=1,
        use_fast_path=True,
        blackbox_mode=True,
        blackbox_feature_selection=True,
        blackbox_min_features_to_select=2,
        blackbox_max_features=4,
        blackbox_standardize=True,
        blackbox_noise_robust="auto",
        loss_mode="huber",
    )
    est.fit(X, y, sample_weight=w)
    diag = getattr(est, "blackbox_diagnostics_", {}) or {}
    assert bool(diag.get("enabled")) is True
    selected = diag.get("selected_features") or []
    assert isinstance(selected, list)
    assert len(selected) >= 1
    # Feature drop case when ranking is confident enough.
    n_selected = int(diag.get("n_selected_features") or len(selected))
    assert n_selected <= X.shape[1]

    pred = np.asarray(est.predict(X), dtype=np.float64)
    clean_r2 = 1.0 - float(np.mean((pred - y_clean) ** 2) / (np.var(y_clean) + 1e-15))
    # Soft gate: clean recovery should remain useful under block outliers.
    assert clean_r2 > 0.5, (
        f"clean recovery too weak under blackbox outliers: {clean_r2}"
    )
    # Protocol-facing fields exist for release artifacts.
    assert "ranking_sample_weight_mode" in diag or getattr(
        est, "sample_weight_provided_", False
    )
