"""Regression tests for benchmark scoring reliability contracts."""

import json
from pathlib import Path

import numpy as np

from scripts import benchmark_suite as bs
from scripts import benchmark_common as bc
from scripts import classifier_fast_path as cfp


def test_formula_mse_eval_returns_none_on_parse_failure():
    x = np.linspace(-2.0, 2.0, 64)
    y = x ** 2

    assert bs._evaluate_formula_mse("x^2", x, y) is not None
    assert bs._evaluate_formula_mse("sin(", x, y) is None


def test_benchmark_suite_uses_shared_formula_helpers():
    assert bs._postprocess_formula is bc.postprocess_formula
    assert bs._evaluate_formula_mse is bc.evaluate_formula_mse


def test_compare_benchmark_results_counts_same_score_formula_and_mse_changes():
    scratch = Path("scratch")
    scratch.mkdir(exist_ok=True)
    previous_path = scratch / "test_compare_previous.json"
    previous_path.write_text(json.dumps({
        "tiers": {
            "1": {
                "results": [{
                    "formula_target": "x",
                    "score": "EXACT",
                    "mse": 1e-8,
                    "formula_discovered": "x",
                }]
            }
        }
    }), encoding="utf-8")
    current = {
        1: [{
            "formula_target": "x",
            "score": "EXACT",
            "mse": 5e-8,
            "formula_discovered": "1*x",
        }]
    }

    try:
        comparison = bs.compare_benchmark_results(previous_path, current)
    finally:
        previous_path.unlink(missing_ok=True)

    assert comparison["summary"]["same"] == 1
    assert comparison["summary"]["changed"] == 1
    assert comparison["transitions"][0]["formula_changed"] is True
    assert comparison["transitions"][0]["mse_changed"] is True


def test_formula_mse_eval_handles_base_log_constants():
    x = np.linspace(-2.0, 2.0, 64)
    y = np.log(np.e) / np.log(10.0) * x

    assert bs._evaluate_formula_mse("log(E, 10)*x", x, y) is not None
    assert bs.cfp._evaluate_formula_values("log(E, 2)*x", x) is not None


def test_benchmark_target_parser_treats_lowercase_e_as_euler_constant():
    x, y = bs._generate_data("e*x", -2.0, 2.0, 64)

    assert np.allclose(y, np.e * x)


def test_run_formula_flags_formula_eval_failed(monkeypatch):
    def _fake_fast_path(*args, **kwargs):
        return {
            "formula": "sin(",
            "mse": 1e-12,
            "details": {"n_nonzero": 1, "n_nonzero_simplified": 1},
        }

    monkeypatch.setattr(bs, "run_fast_path", _fake_fast_path)

    result = bs.run_formula(
        formula_str="x^2",
        x_range=(-2.0, 2.0),
        classifier_path="unused.pt",
        n_samples=64,
        device="cpu",
        with_evolution=False,
        evolution_only=False,
    )

    assert result["formula_discovered"]
    assert result["mse_display"] is None
    assert result["error"] == "formula_eval_failed"
    assert result["formula_before_display_error"]
    assert result["display_eval_diagnostics"]["ok"] is False
    assert result["score"] == "FAIL"


def test_residual_diagnostics_flag_structured_residual():
    x = np.linspace(-2.0, 2.0, 256)
    y_true = np.sin(3.0 * x)
    y_pred = y_true - 0.2 * np.sin(9.0 * x)

    diagnostics = cfp._residual_diagnostics(y_true, y_pred, x)

    assert diagnostics["residual_mse"] is not None
    assert diagnostics["residual_spectral_peak_ratio"] is not None
    assert diagnostics["residual_suspicious"] is True


def test_prediction_uncertainty_metrics():
    metrics = cfp._prediction_uncertainty_metrics({"sin": 0.7, "cos": 0.2, "exp": 0.1})

    assert metrics["prediction_entropy"] is not None
    assert metrics["prediction_margin"] is not None
    assert abs(metrics["prediction_top1"] - 0.7) < 1e-12
    assert abs(metrics["prediction_top2"] - 0.2) < 1e-12
    assert metrics["prediction_uncertain"] is False


def test_fast_path_direct_transform_template_recovers_log_affine():
    x = np.linspace(0.0, 5.0, 80)
    y = np.log(2.0 * x + 1.0)

    formula, mse, details = cfp.fast_path_regression(
        x,
        y,
        {"log": 0.94, "exponential": 0.94},
        exact_match_enabled=True,
    )

    assert mse < 1e-10
    assert details["exact_match"] is True
    assert details["template_match"] == "log_affine_direct"
    assert "log" in formula


def test_fast_path_direct_transform_template_preempts_exp_polynomial_surrogate():
    x = np.linspace(-1.0, 2.0, 80)
    y = 3.0 + 2.0 * np.exp(0.7 * x)

    formula, mse, details = cfp.fast_path_regression(
        x,
        y,
        {"exp": 0.9, "exponential": 0.9},
        exact_match_enabled=True,
    )

    assert mse < 1e-10
    assert details["exact_match"] is True
    assert details["template_match"] == "shifted_exp_affine"
    assert "exp" in formula


def test_fast_path_candidate_pool_reports_semantic_dedup():
    x = np.linspace(-2.0, 2.0, 80)
    y = np.sin(x) + 0.1 * x

    _, _, details = cfp.fast_path_regression(
        x,
        y,
        {"sin": 0.9, "addition": 0.8, "polynomial": 0.4},
        exact_match_enabled=False,
    )

    dedup = details["candidate_semantic_dedup"]
    assert dedup["enabled"] is True
    assert dedup["before"] >= dedup["after"]
    assert dedup["removed"] == dedup["before"] - dedup["after"]
    assert "numpy" in details["solver_backends"]
    assert any(c.get("solver_backend") for c in details["candidate_formulas"])


def test_decomposition_probe_candidates_capture_product_sum():
    x = np.linspace(-2.0, 2.0, 80)
    y = np.sin(x) * np.cos(x) + 0.25 * x

    candidates = cfp.build_decomposition_probe_candidates(
        x,
        y,
        {"sin": 0.8, "cos": 0.8, "multiplication": 0.8, "addition": 0.8},
        max_candidates=5,
    )

    assert candidates
    assert candidates[0]["mse"] < 1e-10
    assert candidates[0]["source"] == "decomposition_probe"
    assert candidates[0]["decomposition_probe_type"] in {"additive_pair", "multiplicative_pair"}


def test_benchmark_guided_evolution_receives_fast_path_candidate_pool(monkeypatch):
    sent_candidates = [
        {"formula": "x", "mse": 1.0, "source": "fast_path"},
        {
            "formula": "sin(x)*cos(x) + 0.25*x",
            "mse": 0.0,
            "source": "decomposition_probe",
            "decomposition_probe_type": "multiplicative_pair",
        },
    ]
    captured = {}

    def _fake_fast_path(*args, **kwargs):
        return {
            "formula": "x",
            "mse": 1.0,
            "details": {"n_nonzero": 1, "n_nonzero_simplified": 1, "y_variance": 1.0},
            "candidate_formulas": sent_candidates,
            "operator_hints": {},
            "residual_diagnostics": {"residual_suspicious": False},
        }

    def _fake_guided(*args, **kwargs):
        captured["candidate_formulas"] = kwargs.get("candidate_formulas")
        return {"formula": "sin(x)*cos(x) + 0.25*x", "mse": 0.0, "raw_mse": 0.0}

    monkeypatch.setattr(bs, "run_fast_path", _fake_fast_path)
    monkeypatch.setattr(bs, "run_guided_evolution", _fake_guided)
    monkeypatch.setattr(bs, "_evaluate_formula_mse", lambda formula, *a, **k: 1.0 if formula == "x" else 0.0)

    result = bs.run_formula(
        formula_str="sin(x)*cos(x)+0.25*x",
        x_range=(-2.0, 2.0),
        classifier_path="unused.pt",
        n_samples=64,
        device="cpu",
        with_evolution=True,
        disable_proposer=True,
    )

    formulas = [c["formula"] for c in captured["candidate_formulas"]]
    assert "sin(x)*cos(x) + 0.25*x" in formulas
    assert result["evolution_seed_candidates"] == captured["candidate_formulas"]


def test_residual_diagnostics_handles_nan_mask_with_holdout():
    x = np.linspace(-3.0, 3.0, 128)
    y_true = np.sin(x)
    y_pred = np.sin(x) + 0.05 * np.cos(5.0 * x)
    y_true[5] = np.nan
    y_pred[7] = np.nan

    diagnostics = cfp._residual_diagnostics(y_true, y_pred, x)

    assert diagnostics["residual_mse"] is not None
    assert diagnostics["residual_holdout_ratio"] is not None
    assert np.isfinite(diagnostics["residual_holdout_ratio"])


def test_run_formula_triggers_guided_on_uncertainty(monkeypatch):
    guided_called = {"value": False}

    def _fake_fast_path(*args, **kwargs):
        return {
            "formula": "x",
            "mse": 1e-8,
            "details": {"n_nonzero": 1, "n_nonzero_simplified": 1},
            "uncertainty": {
                "prediction_entropy": 0.95,
                "prediction_margin": 0.02,
                "prediction_uncertain": True,
            },
            "residual_diagnostics": {"residual_suspicious": False},
            "operator_hints": {},
        }

    def _fake_guided(*args, **kwargs):
        guided_called["value"] = True
        return {"formula": "x", "mse": 0.0}

    monkeypatch.setattr(bs, "run_fast_path", _fake_fast_path)
    monkeypatch.setattr(bs, "run_guided_evolution", _fake_guided)
    monkeypatch.setattr(bs, "_evaluate_formula_mse", lambda *a, **k: 1e-3)

    result = bs.run_formula(
        formula_str="x",
        x_range=(-2.0, 2.0),
        classifier_path="unused.pt",
        n_samples=64,
        device="cpu",
        with_evolution=True,
        evolution_only=False,
    )

    assert guided_called["value"] is True
    assert result["formula_discovered"]


def test_run_formula_triggers_guided_on_suspicious_residual(monkeypatch):
    guided_called = {"value": False}

    def _fake_fast_path(*args, **kwargs):
        return {
            "formula": "x",
            "mse": 1e-8,
            "details": {"n_nonzero": 1, "n_nonzero_simplified": 1},
            "uncertainty": {
                "prediction_entropy": 0.5,
                "prediction_margin": 0.8,
                "prediction_uncertain": False,
            },
            "residual_diagnostics": {"residual_suspicious": True},
            "operator_hints": {},
        }

    def _fake_guided(*args, **kwargs):
        guided_called["value"] = True
        return {"formula": "x", "mse": 0.0}

    monkeypatch.setattr(bs, "run_fast_path", _fake_fast_path)
    monkeypatch.setattr(bs, "run_guided_evolution", _fake_guided)
    monkeypatch.setattr(bs, "_evaluate_formula_mse", lambda *a, **k: 1e-3)

    result = bs.run_formula(
        formula_str="x",
        x_range=(-2.0, 2.0),
        classifier_path="unused.pt",
        n_samples=64,
        device="cpu",
        with_evolution=True,
        evolution_only=False,
    )

    assert guided_called["value"] is True
    assert result["formula_discovered"]


def test_run_formula_can_trust_proposer_search_plan(monkeypatch):
    captured = {}

    payload = {
        "valid": True,
        "sequence_uncertainty": {"entropy": 0.1, "margin": 0.95, "confident": True},
        "operator_priors": {"power": 0.7},
        "candidate_skeletons": [{"formula": "x^2", "mse": 0.0, "score": 1.0}],
        "search_plan": {
            "strategy": "exploratory",
            "difficulty": 0.9,
            "generation_multiplier": 0.04,
            "population_multiplier": 0.04,
            "n_beams": 11,
            "n_rounds": 2,
            "p_min": -4.0,
            "p_max": 6.0,
            "seed_budget": 9,
            "acceptable_complexity": 17,
            "early_stop_max_nodes": 31,
        },
    }

    def _fake_fast_path(*args, **kwargs):
        return {
            "formula": "x",
            "mse": 1.0,
            "details": {"n_nonzero": 1, "n_nonzero_simplified": 1},
            "uncertainty": {
                "prediction_entropy": 0.95,
                "prediction_margin": 0.02,
                "prediction_uncertain": True,
            },
            "residual_diagnostics": {"residual_suspicious": False},
            "operator_hints": {},
        }

    def _fake_guided(*args, **kwargs):
        captured.update(kwargs)
        return {"formula": "x^2", "mse": 0.0}

    monkeypatch.setattr(bs, "run_fast_path", _fake_fast_path)
    monkeypatch.setattr(bs, "run_guided_evolution", _fake_guided)
    monkeypatch.setattr(bs, "_get_proposer", lambda *args, **kwargs: object())
    monkeypatch.setattr(bs, "_evaluate_formula_mse", lambda formula, *a, **k: 1.0 if formula == "x" else 0.0)

    import glassbox.universal_proposer as up

    monkeypatch.setattr(up, "propose_fpip_v2_from_xy", lambda *args, **kwargs: payload)

    result = bs.run_formula(
        formula_str="x^2",
        x_range=(-2.0, 2.0),
        classifier_path="unused.pt",
        n_samples=64,
        device="cpu",
        with_evolution=True,
        evolution_only=False,
        proposer_path="unused.pt",
        evolution_generations=150,
        evolution_population=50,
        trust_proposer_plan=True,
    )

    assert captured["generations"] == 6
    assert captured["population_size"] == 2
    assert captured["search_plan"]["n_beams"] == 11
    assert captured["search_plan"]["n_rounds"] == 2
    assert captured["search_plan"]["p_min"] == -4.0
    assert captured["search_plan"]["p_max"] == 6.0
    assert captured["search_plan"]["seed_budget"] == 9
    assert captured["search_plan"]["acceptable_complexity"] == 17
    assert captured["search_plan"]["early_stop_max_nodes"] == 31
    assert result["formula_discovered"]


def test_run_formula_passes_candidate_formulas(monkeypatch):
    candidates = [
        {"formula": "x", "mse": 0.0, "score": 0.0, "n_nonzero": 1, "active_terms": ["x"], "alpha": 0.0},
        {"formula": "2*x", "mse": 0.1, "score": 0.101, "n_nonzero": 1, "active_terms": ["x"], "alpha": 0.1},
    ]

    def _fake_fast_path(*args, **kwargs):
        return {
            "formula": "x",
            "mse": 0.0,
            "details": {"n_nonzero": 1, "n_nonzero_simplified": 1},
            "candidate_formulas": candidates,
        }

    monkeypatch.setattr(bs, "run_fast_path", _fake_fast_path)

    result = bs.run_formula(
        formula_str="x",
        x_range=(-2.0, 2.0),
        classifier_path="unused.pt",
        n_samples=64,
        device="cpu",
        with_evolution=False,
        evolution_only=False,
    )

    assert result["candidate_formulas"] == candidates
    assert result["fast_path_candidate_formulas"] == candidates
    assert result["winning_stage"] == "fast_path"


def test_run_formula_preserves_guided_raw_mse_and_seed_candidates(monkeypatch):
    sent_candidates = [
        {"formula": "x", "mse": 1.0, "from_fast_path": True},
    ]

    def _fake_fast_path(*args, **kwargs):
        return {
            "formula": "x",
            "mse": 1.0,
            "details": {"n_nonzero": 1, "n_nonzero_simplified": 1},
            "candidate_formulas": sent_candidates,
            "operator_hints": {},
        }

    def _fake_guided(*args, **kwargs):
        return {"formula": "x^2", "mse": 0.0, "raw_mse": 0.123}

    monkeypatch.setattr(bs, "run_fast_path", _fake_fast_path)
    monkeypatch.setattr(bs, "run_guided_evolution", _fake_guided)
    monkeypatch.setattr(bs, "_evaluate_formula_mse", lambda formula, *a, **k: 1.0 if formula == "x" else 0.0)

    result = bs.run_formula(
        formula_str="x^2",
        x_range=(-2.0, 2.0),
        classifier_path="unused.pt",
        n_samples=64,
        device="cpu",
        with_evolution=True,
        evolution_only=False,
    )

    assert result["winning_stage"] == "guided_evolution"
    assert result["mse_raw"] == 0.123
    assert result["engine_raw_mse"] == 0.123
    assert result["evolution_seed_candidates"]
    assert result["candidate_formulas"] == result["evolution_seed_candidates"]


def test_specialist_regressor_benchmark_defaults_keep_expensive_phases_off(monkeypatch):
    captured = {}

    class FakeRegressor:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.best_mse_ = 0.0
            self.specialist_track_ = "incumbent path"
            self.has_composed_seeds_ = False
            self.phase_timings_ = {"total_fit": 0.0}
            self.boosting_stages_ = []
            self.boosting_attempted_ = False
            self.boosting_improved_ = False
            self.boosting_diagnostics_ = {}
            self.blackbox_diagnostics_ = {}
            self.inception_rounds_ = []
            self.inception_diagnostics_ = {}
            self.specialist_vault_ = None

        def fit(self, X, y):
            return self

        def get_formula(self):
            return "x"

    import glassbox.sr.sklearn_wrapper as sw

    monkeypatch.setattr(sw, "GlassboxRegressor", FakeRegressor)

    result = bs.run_formula_specialist_regressor(
        formula_str="x",
        x_range=(-2.0, 2.0),
        classifier_path="unused.pt",
        proposer_path=None,
        n_samples=64,
        device="cpu",
    )

    assert captured["enable_specialist_screening_diagnostics"] is True
    assert captured["enable_specialist_composition_screening"] is True
    assert captured["enable_specialist_vault_memory"] is True
    assert captured["enable_residual_stage"] is False
    assert captured["enable_inception_reuse"] is False
    assert result["specialist_phase_config"]["residual"] is False
    assert result["specialist_phase_config"]["inception"] is False


def test_specialist_regressor_benchmark_can_enable_full_phases(monkeypatch):
    captured = {}

    class FakeRegressor:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.best_mse_ = 0.0
            self.specialist_track_ = "incumbent path"
            self.has_composed_seeds_ = False
            self.phase_timings_ = {"total_fit": 0.0}
            self.boosting_stages_ = []
            self.boosting_attempted_ = False
            self.boosting_improved_ = False
            self.boosting_diagnostics_ = {}
            self.blackbox_diagnostics_ = {}
            self.inception_rounds_ = []
            self.inception_diagnostics_ = {}
            self.specialist_vault_ = None

        def fit(self, X, y):
            return self

        def get_formula(self):
            return "x"

    import glassbox.sr.sklearn_wrapper as sw

    monkeypatch.setattr(sw, "GlassboxRegressor", FakeRegressor)

    result = bs.run_formula_specialist_regressor(
        formula_str="x",
        x_range=(-2.0, 2.0),
        classifier_path="unused.pt",
        proposer_path=None,
        n_samples=64,
        device="cpu",
        specialist_residual=True,
        specialist_inception=True,
    )

    assert captured["enable_residual_stage"] is True
    assert captured["enable_inception_reuse"] is True
    assert result["specialist_phase_config"]["residual"] is True
    assert result["specialist_phase_config"]["inception"] is True


def test_classifier_prior_trust_from_uncertainty_extremes():
    high_trust = cfp._classifier_prior_trust_from_uncertainty(
        {"prediction_entropy": 0.05, "prediction_margin": 0.45, "prediction_uncertain": False}
    )
    low_trust = cfp._classifier_prior_trust_from_uncertainty(
        {"prediction_entropy": 0.99, "prediction_margin": 0.0, "prediction_uncertain": True}
    )

    assert 0.0 <= low_trust <= high_trust <= 1.0
    assert high_trust > 0.8
    assert low_trust < 0.2


def test_blend_priors_with_uniform_respects_trust():
    base = [0.7, 0.2, 0.08, 0.02]
    almost_uniform = cfp._blend_priors_with_uniform(base, trust=0.0)
    almost_base = cfp._blend_priors_with_uniform(base, trust=1.0)

    assert abs(sum(almost_uniform) - 1.0) < 1e-12
    assert abs(sum(almost_base) - 1.0) < 1e-12
    assert max(abs(v - 0.25) for v in almost_uniform) < 1e-12
    assert max(abs(v - e) for v, e in zip(almost_base, cfp._normalize_priors(base))) < 1e-12
