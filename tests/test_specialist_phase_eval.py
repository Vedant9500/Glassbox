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


def test_specialist_metadata_splits_composition_semantics():
    from scripts import benchmark_common as bc

    reg = GlassboxRegressor()
    reg.composition_candidates_accepted_ = True
    reg.composition_candidate_count_ = 3
    reg.composition_seeded_evolution_ = True
    reg.composition_won_final_selection_ = False
    reg.composition_improved_mse_ = False
    reg.formula_eval_count_ = 7
    reg.formula_eval_cache_hits_ = 3
    reg._formula_eval_cache_ = {"cached": object()}

    meta = bc.specialist_metadata_from_estimator(reg)

    assert meta["composition_candidates_accepted"] is True
    assert meta["composition_candidate_count"] == 3
    assert meta["composition_seeded_evolution"] is True
    assert meta["composition_won_final_selection"] is False
    assert meta["composition_improved_mse"] is False
    assert meta["formula_eval_count"] == 7
    assert meta["formula_eval_cache_hits"] == 3
    assert meta["formula_eval_cache_size"] == 1


def test_safe_eval_formula_array_caches_repeated_matrix_eval():
    reg = GlassboxRegressor()
    X = np.linspace(-2.0, 2.0, 32).reshape(-1, 1)

    first = reg._safe_eval_formula_array("sin(x)", X)
    second = reg._safe_eval_formula_array("sin(x)", X)

    assert np.allclose(first, second)
    assert reg.formula_eval_count_ == 2
    assert reg.formula_eval_cache_hits_ == 1


def test_targeted_specialist_probes_prioritize_envelope_carrier_products():
    reg = GlassboxRegressor()
    x = np.linspace(-2.0, 2.0, 160)
    X = x.reshape(-1, 1)
    y = np.exp(-(x ** 2)) * np.sin(3.0 * x)

    probes = reg._targeted_specialist_probe_formulas(X, y, max_formulas=12)

    formulas = [probe["formula"] for probe in probes]
    assert "exp(-x0^2)*sin(3*x0)" in formulas


def test_univariate_specialist_candidate_pool_includes_targeted_probes():
    class ProbeRegressor(GlassboxRegressor):
        def _refine_candidate_formulas(self, candidate_formulas, X, y, *, max_candidates=12):
            return list(candidate_formulas)[:max_candidates]

    reg = ProbeRegressor()
    x = np.linspace(-2.0, 2.0, 80)
    X = x.reshape(-1, 1)
    y = np.exp(-x) * np.sin(2.0 * x)

    candidates = reg._build_univariate_specialist_candidate_formulas(
        None,
        None,
        None,
        X,
        y,
        max_candidates=80,
    )

    assert any(candidate.get("from_targeted_specialist_probe") for candidate in candidates)
    assert any(candidate.get("source") == "envelope_carrier_probe" for candidate in candidates)


def test_cpp_batch_candidate_scoring_matches_affine_fit_when_available():
    try:
        import _core  # type: ignore
    except Exception:
        return
    if not hasattr(_core, "score_formula_candidates"):
        return

    x = np.linspace(-2.0, 2.0, 80)
    X = np.ascontiguousarray(x.reshape(-1, 1), dtype=np.float64)
    y = np.ascontiguousarray(2.0 * np.sin(3.0 * x) + 0.5, dtype=np.float64)
    split = 60

    scores = _core.score_formula_candidates(
        ["sin(3*x0)", "x0^2"],
        X[:split],
        y[:split],
        X[split:],
        y[split:],
        2,
    )

    assert scores[0]["ok"] is True
    assert scores[0]["validation_r2"] > 0.999
    assert abs(scores[0]["scale"] - 2.0) < 1e-9
    assert abs(scores[0]["bias"] - 0.5) < 1e-9


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


def test_residual_symbolic_fit_uses_bounded_mini_search_without_recursing():
    class NoRecursiveRegressor(GlassboxRegressor):
        def fit(self, X, y):
            raise AssertionError("residual stage should not launch a nested estimator")

    x = np.linspace(-2.0, 2.0, 96)
    X = x.reshape(-1, 1)
    y = x + x ** 2
    reg = NoRecursiveRegressor(
        use_guided_evolution=True,
        enable_residual_stage=True,
        residual_mini_search_max_candidates=16,
        residual_mini_search_refine_top_k=4,
        random_state=13,
    )

    residual_formula = reg._stage_residual_symbolic_fit(
        X,
        y,
        "x0",
        _allow_recursion=True,
    )

    assert residual_formula
    base_mse = reg._formula_mse("x0", X, y)
    combined_mse = reg._formula_mse(f"(x0)+({residual_formula})", X, y)
    assert combined_mse < base_mse * 0.1
    assert reg._residual_stage_guard_["mode"] == "bounded_mini_search"
    assert reg._residual_stage_guard_["accepted"] is True


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
