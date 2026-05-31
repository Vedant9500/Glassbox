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
