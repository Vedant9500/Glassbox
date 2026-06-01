from scripts import specialist_phase_eval as spe
import numpy as np
from glassbox.sr.sklearn_wrapper import GlassboxRegressor


def test_phase0_harness_returns_summary_and_cases():
    result = spe.run_phase0(quick=True)

    assert isinstance(result, dict)
    assert "summary" in result
    assert "cases" in result

    summary = result["summary"]
    cases = result["cases"]

    assert summary["phase"] == 0
    assert summary["n_cases"] == len(cases)
    assert len(cases) >= 1

    for case in cases:
        assert "name" in case
        assert "baseline" in case
        assert "phase0" in case
        assert "delta" in case
        assert "error" in case


def test_phase0_harness_emits_specialist_diagnostics_for_phase_run():
    result = spe.run_phase0(quick=True)

    phase_cases = [case["phase0"] for case in result["cases"]]
    assert any(run.get("has_specialist_screening") for run in phase_cases)


def test_phase3_harness_returns_summary_and_cases():
    result = spe.run_phase3(quick=True)

    assert isinstance(result, dict)
    assert "summary" in result
    assert "cases" in result

    summary = result["summary"]
    cases = result["cases"]

    assert summary["phase"] == 3
    assert summary["n_cases"] == len(cases)
    assert len(cases) >= 1
    assert summary["pass"] is True


def test_winning_track_tracking_on_simple_regressor():
    reg = GlassboxRegressor()
    assert reg.specialist_track_ == "incumbent path"

    x = np.linspace(-1.0, 1.0, 50)
    X = x.reshape(-1, 1)
    y = 2.0 * x + 0.5

    reg.fit(X, y)
    assert reg.specialist_track_ in ("incumbent path", "screening only")


def test_phase4_harness_returns_summary_and_cases():
    result = spe.run_phase4(quick=True)

    assert isinstance(result, dict)
    assert "summary" in result
    assert "cases" in result

    summary = result["summary"]
    cases = result["cases"]

    assert summary["phase"] == 4
    assert summary["n_cases"] == len(cases)
    assert len(cases) >= 1
    assert summary["pass"] is True


def test_phase5_harness_returns_summary_and_cases():
    result = spe.run_phase5(quick=True)

    assert isinstance(result, dict)
    assert "summary" in result
    assert "cases" in result

    summary = result["summary"]
    cases = result["cases"]

    assert summary["phase"] == 5
    assert summary["n_cases"] == len(cases)
    assert len(cases) >= 1
    assert summary["pass"] is True


def test_phase6_harness_returns_summary_and_cases():
    result = spe.run_phase6(quick=True)

    assert isinstance(result, dict)
    assert "summary" in result
    assert "cases" in result

    summary = result["summary"]
    cases = result["cases"]

    assert summary["phase"] == 6
    assert summary["n_cases"] == len(cases)
    assert len(cases) >= 1
    assert summary["pass"] is True


def test_phase7_harness_returns_summary_and_cases():
    result = spe.run_phase7(quick=True)

    assert isinstance(result, dict)
    assert "summary" in result
    assert "cases" in result

    summary = result["summary"]
    cases = result["cases"]

    assert summary["phase"] == 7
    assert summary["n_cases"] == len(cases)
    assert len(cases) >= 1
    assert summary["pass"] is True


def test_phase8_harness_returns_summary_and_cases():
    result = spe.run_phase8(quick=True)

    assert isinstance(result, dict)
    assert "summary" in result
    assert "cases" in result

    summary = result["summary"]
    cases = result["cases"]

    assert summary["phase"] == 8
    assert summary["n_cases"] == len(cases)
    assert len(cases) >= 1
    assert summary["pass"] is True


def test_phase9_harness_returns_summary_and_cases():
    result = spe.run_phase9(quick=True)

    assert isinstance(result, dict)
    assert "summary" in result
    assert "cases" in result

    summary = result["summary"]
    cases = result["cases"]

    assert summary["phase"] == 9
    assert summary["n_cases"] == len(cases)
    assert len(cases) >= 1
    assert summary["pass"] is True


def test_residual_boosting_records_attempt_and_improvement():
    class ProbeRegressor(GlassboxRegressor):
        def _stage_residual_symbolic_fit(self, X, y, base_formula, *, _allow_recursion=False):
            return "sin(x0)"

        def _refine_candidate_formulas(self, candidate_formulas, X, y, *, max_candidates=12):
            return list(candidate_formulas)[:max_candidates]

    x = np.linspace(-2.0, 2.0, 80)
    X = x.reshape(-1, 1)
    y = x + np.sin(x)
    reg = ProbeRegressor(
        use_guided_evolution=True,
        enable_residual_stage=True,
        max_boosting_stages=2,
        boosting_learning_rates=[0.5, 1.0],
    )

    formula = reg._run_residual_boosting(X, y, "x0")

    assert "sin(x0)" in formula
    assert reg.boosting_attempted_ is True
    assert reg.boosting_improved_ is True
    assert len(reg.boosting_stages_) == 1
    assert reg.boosting_diagnostics_["accepted_stages"] == 1


def test_univariate_fit_runs_specialist_candidate_screening(monkeypatch):
    class ProbeRegressor(GlassboxRegressor):
        def _build_univariate_specialist_candidate_formulas(self, best_formula, best_mse, proposer_payload, X, y, *, max_candidates):
            return [
                {"formula": "x0", "validation_r2": 0.5, "validation_mse": 0.1, "mse": 0.1, "complexity": 1},
                {"formula": "sin(x0)", "validation_r2": 0.6, "validation_mse": 0.08, "mse": 0.08, "complexity": 2},
            ]

        def _run_residual_boosting(self, X, y, base_formula):
            return base_formula

        def _run_inception_reuse(self, X, y, base_formula):
            return base_formula

    x = np.linspace(-2.0, 2.0, 80)
    X = x.reshape(-1, 1)
    y = np.sin(x)
    reg = ProbeRegressor(
        use_fast_path=False,
        use_guided_evolution=False,
        blackbox_mode=False,
        enable_specialist_screening_diagnostics=True,
        enable_specialist_composition_screening=False,
        enable_residual_stage=False,
    )

    reg.fit(X, y)

    screening = (reg.blackbox_diagnostics_ or {}).get("candidate_screening", {})
    assert screening.get("candidate_count") == 2
    assert isinstance(screening.get("specialist_screening"), dict)
