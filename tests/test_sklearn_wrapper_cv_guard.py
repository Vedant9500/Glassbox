import sys
from types import SimpleNamespace

import numpy as np

from glassbox.sr.blackbox_preprocessor import prepare_blackbox_search
from glassbox.sr.sklearn_wrapper import GlassboxRegressor


def test_cv_skip_guard_passes_for_stable_formula(monkeypatch):
    n = 120
    x = np.linspace(-2.0, 2.0, n)
    X = x.reshape(-1, 1)
    y = x.copy()

    est = GlassboxRegressor(
        cv_skip_guard_enabled=True,
        cv_skip_guard_folds=3,
        cv_skip_guard_min_fold_r2=0.99,
        cv_skip_guard_max_r2_std=0.02,
        random_state=11,
    )

    monkeypatch.setattr(est, "_safe_eval_formula_array", lambda formula, X_in: y.copy())

    ok = est._passes_cross_validation_skip_guard("x", X, y)

    assert ok is True
    assert est.fast_path_cv_guard_["passed"] is True
    assert est.fast_path_cv_guard_["min_fold_r2"] is not None


def test_cv_skip_guard_fails_for_unstable_formula(monkeypatch):
    n = 120
    x = np.linspace(-2.0, 2.0, n)
    X = x.reshape(-1, 1)
    y = x.copy()

    est = GlassboxRegressor(
        cv_skip_guard_enabled=True,
        cv_skip_guard_folds=3,
        cv_skip_guard_min_fold_r2=0.95,
        cv_skip_guard_max_r2_std=0.02,
        random_state=7,
    )

    idx = np.arange(n)
    rng = np.random.RandomState(7)
    rng.shuffle(idx)
    folds = [f for f in np.array_split(idx, 3) if len(f) > 0]

    y_pred = y.copy()
    y_pred[folds[0]] = y_pred[folds[0]] + 8.0

    monkeypatch.setattr(est, "_safe_eval_formula_array", lambda formula, X_in: y_pred)

    ok = est._passes_cross_validation_skip_guard("x", X, y)

    assert ok is False
    assert est.fast_path_cv_guard_["passed"] is False
    assert est.fast_path_cv_guard_["reason"] == "unstable_fold_performance"


def test_formula_cleanup_guard_rejects_worse_simplification(monkeypatch):
    x = np.linspace(-2.0, 2.0, 80)
    X = x.reshape(-1, 1)
    y = np.sin(x)
    est = GlassboxRegressor(random_state=3)
    est.n_features_in_ = 1
    est.blackbox_diagnostics_ = {}

    monkeypatch.setattr(
        est, "_reduce_formula_noise", lambda formula, X_in, y_in: formula
    )
    monkeypatch.setattr(est, "_simplify_formula", lambda formula: "0")

    cleaned = est._cleanup_formula_with_fidelity_guard("sin(x)", X, y, stage="unit")

    assert cleaned == "sin(x)"
    steps = est.blackbox_diagnostics_["formula_cleanup_guard"][0]["steps"]
    assert steps[0]["step"] == "simplify_formula"
    assert steps[0]["accepted"] is False


def test_formula_cleanup_guard_accepts_equivalent_cleanup(monkeypatch):
    x = np.linspace(-2.0, 2.0, 80)
    X = x.reshape(-1, 1)
    y = np.sin(x)
    est = GlassboxRegressor(random_state=4)
    est.n_features_in_ = 1
    est.blackbox_diagnostics_ = {}

    monkeypatch.setattr(
        est, "_reduce_formula_noise", lambda formula, X_in, y_in: formula
    )
    monkeypatch.setattr(est, "_simplify_formula", lambda formula: "sin(x0)")

    cleaned = est._cleanup_formula_with_fidelity_guard("sin(x)", X, y, stage="unit")

    assert cleaned == "sin(x0)"
    steps = est.blackbox_diagnostics_["formula_cleanup_guard"][0]["steps"]
    assert steps[0]["accepted"] is True


def test_actionable_specialist_candidate_pool_is_retained_without_composition():
    est = GlassboxRegressor(random_state=5)
    est.early_stop_mse = 1e-12
    est.evolution_skip_r2 = 0.999

    candidates = [
        {
            "formula": "exp(-2*x0)",
            "mse": 0.0,
            "validation_mse": 0.0,
            "validation_r2": 1.0,
        }
    ]

    assert (
        est._candidate_pool_has_actionable_fit(
            candidates,
            incumbent_mse=0.1,
            search_plan={"candidate_acceptance_r2": 0.985},
        )
        is True
    )


def test_universal_proposer_dual_path_handles_multivariate_input(monkeypatch):
    import glassbox.universal_proposer as up

    n = 64
    x1 = np.linspace(-2.0, 2.0, n)
    x2 = np.linspace(1.0, 3.0, n)
    X = np.stack([x1, x2], axis=1)
    y = x1 + x2
    captured = {}

    def fake_propose(model, x, y, top_k, fit_diagnostics, interaction_hints, device):
        captured["x_shape"] = np.asarray(x).shape
        return {
            "valid": True,
            "candidate_skeletons": [],
            "interaction_hints": dict(interaction_hints),
            "search_plan": {"supports_multivariate_formulas": np.asarray(x).ndim == 2},
            "routing_signal": {"recommend_guided_evolution": False},
        }

    monkeypatch.setattr(
        up, "load_universal_proposer_checkpoint", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(up, "propose_fpip_v2_from_xy", fake_propose)

    est = GlassboxRegressor(
        use_universal_proposer=True,
        universal_proposer_shadow_mode=True,
        universal_proposer_log_routing=False,
    )

    payload, force = est._run_universal_proposer_dual_path(X, y, fast_path_result=None)

    assert payload is not None
    assert force is False
    assert est.universal_proposer_status_ == "ok_multivariate_heuristic"
    assert captured["x_shape"] == X.shape
    assert payload["interaction_hints"]["multivariate_proxy"] is False
    assert payload["search_plan"]["supports_multivariate_formulas"] is True


def test_universal_proposer_dual_path_handles_missing_checkpoint():
    n = 64
    x = np.linspace(-2.0, 2.0, n)
    X = x.reshape(-1, 1)
    y = np.sin(x)

    est = GlassboxRegressor(
        use_universal_proposer=True,
        universal_proposer_path="models/does_not_exist.pt",
        universal_proposer_shadow_mode=False,
        universal_proposer_log_routing=False,
    )
    est._resolve_universal_proposer_path = lambda: "models/does_not_exist.pt"

    payload, force = est._run_universal_proposer_dual_path(
        X, y, fast_path_result={"mse": 0.1}
    )

    assert payload is None
    assert force is False
    assert str(est.universal_proposer_status_).startswith("error:")


def test_guided_evolution_receives_remaining_timeout(monkeypatch):
    import glassbox.sr.sklearn_wrapper as sw

    captured = {}

    def _fake_fast_path(*args, **kwargs):
        return {
            "formula": "0",
            "mse": 1.0,
            "operator_hints": {"operators": {"sin"}, "frequencies": [1.0]},
            "details": {"n_nonzero": 1, "y_variance": 0.5},
            "uncertainty": {"entropy": 0.8, "margin": 0.1},
        }

    def _fake_guided_evolution(*args, **kwargs):
        captured["search_plan"] = kwargs.get("search_plan")
        return {"formula": "sin(x)", "mse": 0.0}

    monkeypatch.setattr(sw, "CPP_AVAILABLE", True)
    monkeypatch.setitem(
        sys.modules,
        "classifier_fast_path",
        SimpleNamespace(
            run_fast_path=_fake_fast_path,
            run_guided_evolution=_fake_guided_evolution,
        ),
    )

    x = np.linspace(-2.0, 2.0, 80)
    X = x.reshape(-1, 1)
    y = np.sin(x)

    est = GlassboxRegressor(
        use_fast_path=True,
        use_guided_evolution=True,
        use_universal_proposer=False,
        enable_specialist_screening_diagnostics=False,
        population_size=20,
        generations=30,
        timeout=5,
        evolution_skip_r2=0.999999,
    )
    est.fit(X, y)

    search_plan = captured.get("search_plan")
    assert isinstance(search_plan, dict)
    assert 1 <= search_plan.get("timeout_seconds", 0) <= 5


def test_exact_fast_path_skips_specialist_phases(monkeypatch):

    def _fake_fast_path(*args, **kwargs):
        return {
            "formula": "x0",
            "mse": 0.0,
            "operator_hints": {"operators": {"identity"}, "frequencies": []},
            "details": {"n_nonzero": 1, "y_variance": 1.0},
            "uncertainty": {"prediction_entropy": 0.0, "prediction_margin": 1.0},
        }

    monkeypatch.setitem(
        sys.modules,
        "classifier_fast_path",
        SimpleNamespace(run_fast_path=_fake_fast_path),
    )

    x = np.linspace(-2.0, 2.0, 80)
    X = x.reshape(-1, 1)
    y = x.copy()

    est = GlassboxRegressor(
        use_fast_path=True,
        use_guided_evolution=True,
        use_universal_proposer=False,
        enable_specialist_screening_diagnostics=True,
        enable_specialist_composition_screening=True,
        enable_residual_stage=True,
        enable_inception_reuse=True,
        enable_specialist_vault_memory=True,
        random_state=0,
    )

    def _fail_specialist(*args, **kwargs):
        raise AssertionError("specialist phases should not run after exact fast-path")

    monkeypatch.setattr(
        est, "_build_univariate_specialist_candidate_formulas", _fail_specialist
    )
    monkeypatch.setattr(est, "_run_specialist_candidate_screening", _fail_specialist)
    monkeypatch.setattr(est, "_run_residual_boosting", _fail_specialist)
    monkeypatch.setattr(est, "_run_inception_reuse", _fail_specialist)

    est.fit(X, y)

    assert est.fast_path_exact_skip_ is True
    assert (
        est.blackbox_diagnostics_.get("specialist_skipped_reason") == "fast_path_exact"
    )
    assert est.best_mse_ < 1e-12
    assert est.get_formula()


def test_exact_fast_path_skip_overrides_evolution_routing(monkeypatch):

    formula = "+".join(["x0"] * 12)

    def _fake_fast_path(*args, **kwargs):
        return {
            "formula": formula,
            "mse": 0.0,
            "operator_hints": {"operators": {"identity"}, "frequencies": []},
            "details": {"n_nonzero": 12, "y_variance": 1.0},
            "uncertainty": {"prediction_entropy": 0.0, "prediction_margin": 1.0},
        }

    monkeypatch.setitem(
        sys.modules,
        "classifier_fast_path",
        SimpleNamespace(run_fast_path=_fake_fast_path),
    )

    x = np.linspace(-2.0, 2.0, 80)
    X = x.reshape(-1, 1)
    y = 12.0 * x

    est = GlassboxRegressor(
        use_fast_path=True,
        use_guided_evolution=True,
        use_universal_proposer=False,
        enable_specialist_screening_diagnostics=True,
        enable_specialist_composition_screening=True,
        enable_residual_stage=True,
        enable_inception_reuse=True,
        random_state=0,
    )
    monkeypatch.setattr(
        est, "_run_universal_proposer_dual_path", lambda *args, **kwargs: (None, True)
    )

    def _fail_specialist(*args, **kwargs):
        raise AssertionError("specialist phases should not run after exact fast-path")

    monkeypatch.setattr(
        est, "_build_univariate_specialist_candidate_formulas", _fail_specialist
    )
    monkeypatch.setattr(est, "_run_specialist_candidate_screening", _fail_specialist)
    monkeypatch.setattr(est, "_run_residual_boosting", _fail_specialist)
    monkeypatch.setattr(est, "_run_inception_reuse", _fail_specialist)

    est.fit(X, y)

    assert est.fast_path_exact_skip_ is True
    assert est.best_mse_ < 1e-12
    assert set(est.phase_timings_) == {"total_fit"}


def test_specialist_screening_skips_residual_when_candidate_is_exact(monkeypatch):
    x = np.linspace(-2.0, 2.0, 80)
    X = x.reshape(-1, 1)
    y = np.cos(np.pi * x)

    est = GlassboxRegressor(
        enable_specialist_screening_diagnostics=True,
        enable_specialist_composition_screening=True,
        enable_residual_stage=True,
        random_state=0,
    )
    est.blackbox_diagnostics_ = {}

    def _screening(candidates, X_arg, y_arg, **kwargs):
        return {
            "top_candidates": [
                {"formula": "cos(pi*x0)", "validation_mse": 0.0, "validation_r2": 1.0}
            ]
        }

    def _compose(*args, **kwargs):
        raise AssertionError(
            "composition should not run when an exact candidate is already present"
        )

    def _residual(*args, **kwargs):
        raise AssertionError(
            "residual fit should not run when an exact candidate is already present"
        )

    monkeypatch.setattr(est, "_compute_specialist_screening_diagnostics", _screening)
    monkeypatch.setattr(est, "_compose_specialist_candidates", _compose)
    monkeypatch.setattr(est, "_stage_residual_symbolic_fit", _residual)

    candidates = [
        {"formula": "cos(pi*x0)", "validation_mse": 0.0, "validation_r2": 1.0}
    ]
    returned = est._run_specialist_candidate_screening(
        candidates,
        X,
        y,
        {"screening_budget": 8, "seed_budget": 8},
    )

    assert returned == candidates
    diag = est.blackbox_diagnostics_["candidate_screening"]
    assert diag["residual_skipped_reason"] == "existing_exact_candidate"
    assert diag["best_existing_validation_mse"] == 0.0


def test_regressor_formula_eval_uses_signed_fractional_powers():
    from scripts import benchmark_common as bc

    x = np.linspace(-2.0, 2.0, 41)
    X = x.reshape(-1, 1)
    formula = "x**1.5 - 0.25*x"

    est = GlassboxRegressor()
    reg_pred = est._safe_eval_formula_array(formula, X)
    bench_pred = bc.evaluate_formula(bc.postprocess_formula(formula), X)

    assert bench_pred is not None
    np.testing.assert_allclose(reg_pred, bench_pred, rtol=1e-12, atol=1e-12)


def test_blackbox_search_plan_expands_uncertain_breadth_and_interaction_depth():
    rng = np.random.RandomState(13)
    X = rng.randn(160, 5)
    y = X[:, 0] * np.sin(X[:, 1]) + 0.04 * rng.randn(160)

    _, _, state = prepare_blackbox_search(
        X,
        y,
        enabled=True,
        max_features=4,
        standardize=False,
        min_features_to_select=2,
    )
    state.feature_selection_uncertain = True

    est = GlassboxRegressor(population_size=20, generations=30)
    est.original_n_features_in_ = X.shape[1]
    plan = est._derive_blackbox_search_plan(
        state,
        fast_path_uncertainty={
            "prediction_entropy": 0.9,
            "prediction_margin": 0.05,
            "prediction_uncertain": True,
        },
    )

    assert plan["population_multiplier"] > 1.0
    assert plan["generation_multiplier"] > 1.0
    assert plan["seed_budget"] >= 8
    assert plan["acceptable_complexity"] > 15
    assert plan["early_stop_max_nodes"] > 50
    assert plan["timeout_multiplier"] <= 1.45
    # Selection-uncertain blackbox relaxes hard caps (up to 2.25 / 2.75).
    assert plan["population_multiplier"] <= 2.25
    assert plan["generation_multiplier"] <= 2.75


def test_blackbox_search_plan_prefers_screening_when_candidates_are_strong():
    rng = np.random.RandomState(15)
    X = rng.randn(160, 5)
    y = X[:, 0] * np.sin(X[:, 1]) + 0.03 * rng.randn(160)

    _, _, state = prepare_blackbox_search(
        X,
        y,
        enabled=True,
        max_features=4,
        standardize=False,
        min_features_to_select=2,
    )

    est = GlassboxRegressor(population_size=20, generations=30)
    est.original_n_features_in_ = X.shape[1]
    plan = est._derive_blackbox_search_plan(
        state,
        fast_path_uncertainty={
            "prediction_entropy": 0.25,
            "prediction_margin": 0.55,
            "prediction_uncertain": False,
        },
        candidate_screening={
            "candidate_count": 8,
            "family_count": 4,
            "best_validation_r2": 0.97,
        },
    )

    assert plan["screening_budget"] >= plan["seed_budget"]
    assert plan["basis_max_terms"] >= 3
    assert plan["focus"] in {"screening", "screen_accept", "balanced"}
    # With uncertain feature selection, plan may use relaxed blackbox caps.
    assert plan["population_multiplier"] <= 2.25
    assert plan["generation_multiplier"] <= 2.75
    assert plan["candidate_acceptance_r2"] <= 0.985
    assert plan["candidate_shrink_r2"] < plan["candidate_acceptance_r2"]


def test_blackbox_search_plan_caps_proxy_proposer_inflation():
    rng = np.random.RandomState(16)
    X = rng.randn(180, 6)
    y = X[:, 0] * np.sin(X[:, 1]) + 0.05 * rng.randn(180)

    _, _, state = prepare_blackbox_search(
        X,
        y,
        enabled=True,
        max_features=5,
        standardize=False,
        min_features_to_select=2,
    )
    state.feature_selection_uncertain = True

    est = GlassboxRegressor(population_size=20, generations=30)
    est.original_n_features_in_ = X.shape[1]
    plan = est._derive_blackbox_search_plan(
        state,
        fast_path_uncertainty={
            "prediction_entropy": 0.95,
            "prediction_margin": 0.02,
            "prediction_uncertain": True,
        },
        proposer_plan={
            "generation_multiplier": 4.0,
            "population_multiplier": 3.0,
            "seed_budget": 40,
            "acceptable_complexity": 120,
            "early_stop_max_nodes": 200,
            "timeout_multiplier": 3.0,
        },
    )

    assert plan["population_multiplier"] <= 1.85
    assert plan["generation_multiplier"] <= 2.0
    assert plan["seed_budget"] <= 14
    assert plan["screening_budget"] >= plan["seed_budget"]
    assert plan["acceptable_complexity"] <= 32
    assert plan["early_stop_max_nodes"] <= 64
    assert plan["timeout_multiplier"] <= 1.45


def test_multivariate_blackbox_cpp_seeds_use_reduced_indices(monkeypatch):
    import glassbox.sr.sklearn_wrapper as sw

    captured = {}

    class _FakeCore:
        @staticmethod
        def run_evolution(**kwargs):
            captured["seed_graphs"] = kwargs.get("seed_graphs_py", [])
            captured["pop_size"] = kwargs.get("pop_size")
            captured["generations"] = kwargs.get("generations")
            captured["acceptable_complexity"] = kwargs.get("acceptable_complexity")
            captured["early_stop_max_nodes"] = kwargs.get("early_stop_max_nodes")
            return {
                "best_mse": 0.0,
                "formula": "x0*x1",
                "nodes": [],
                "output_weights": [],
            }

        @staticmethod
        def reduce_formula_noise(formula, X_list, y):
            return formula

        @staticmethod
        def simplify_formula(formula, **kwargs):
            return formula

    def _fake_fast_path(*args, **kwargs):
        return None

    monkeypatch.setattr(sw, "CPP_AVAILABLE", True)
    monkeypatch.setattr(sw, "get_cpp_core", lambda: _FakeCore)
    monkeypatch.setitem(
        sys.modules,
        "classifier_fast_path",
        SimpleNamespace(run_fast_path=_fake_fast_path),
    )
    # Keep evolution path live so seed_graphs / search-plan kwargs are exercised.
    monkeypatch.setattr(
        sw.GlassboxRegressor,
        "_fit_blackbox_engineered_basis_model",
        lambda self, *args, **kwargs: None,
    )

    rng = np.random.RandomState(8)
    X = rng.randn(80, 5)
    y = X[:, 1] * X[:, 4]

    est = GlassboxRegressor(
        use_fast_path=True,
        use_guided_evolution=True,
        use_universal_proposer=False,
        blackbox_mode=True,
        blackbox_max_features=2,
        blackbox_standardize=False,
        blackbox_min_features_to_select=2,
        population_size=10,
        generations=10,
        multi_start_runs=1,
        timeout=20,
        random_state=8,
    )
    est.fit(X, y)

    feature_indices = [
        node.get("feature_idx")
        for graph in captured.get("seed_graphs", [])
        for node in graph.get("nodes", [])
        if node.get("type") == 0
    ]
    assert feature_indices
    assert max(feature_indices) < est.n_features_in_
    assert est.blackbox_search_plan_["seed_budget"] >= 8
    assert est.blackbox_diagnostics_["search_plan"] == est.blackbox_search_plan_
    assert captured["pop_size"] >= est.population_size
    assert captured["generations"] >= est.generations
    assert (
        captured["acceptable_complexity"]
        == est.blackbox_search_plan_["acceptable_complexity"]
    )
    assert (
        captured["early_stop_max_nodes"]
        == est.blackbox_search_plan_["early_stop_max_nodes"]
    )


def test_blackbox_refined_candidate_can_skip_cpp(monkeypatch):
    import glassbox.sr.sklearn_wrapper as sw

    called = {"cpp": False}

    class _FakeCore:
        @staticmethod
        def run_evolution(**kwargs):
            called["cpp"] = True
            return {"best_mse": 1.0, "formula": "0", "nodes": [], "output_weights": []}

        @staticmethod
        def reduce_formula_noise(formula, X_list, y):
            return formula

        @staticmethod
        def simplify_formula(formula, **kwargs):
            return formula

    def _fake_fast_path(*args, **kwargs):
        return {
            "formula": "x0 + x1",
            "mse": 10.0,
            "operator_hints": {},
        }

    monkeypatch.setattr(sw, "CPP_AVAILABLE", True)
    monkeypatch.setattr(sw, "get_cpp_core", lambda: _FakeCore)
    monkeypatch.setitem(
        sys.modules,
        "classifier_fast_path",
        SimpleNamespace(run_fast_path=_fake_fast_path),
    )

    rng = np.random.RandomState(17)
    X = rng.randn(120, 2)
    y = X[:, 0] + X[:, 1]

    est = GlassboxRegressor(
        use_fast_path=True,
        use_guided_evolution=False,
        use_universal_proposer=False,
        blackbox_mode=True,
        blackbox_standardize=False,
        blackbox_min_features_to_select=2,
        population_size=10,
        generations=10,
        multi_start_runs=1,
        timeout=20,
        random_state=17,
    )
    est.fit(X, y)

    assert called["cpp"] is False
    assert "x0" in est.get_formula() or "x1" in est.get_formula()


def test_blackbox_basis_model_skips_cpp_on_additive_signal(monkeypatch):
    import glassbox.sr.sklearn_wrapper as sw

    called = {"cpp": False}

    class _FakeCore:
        @staticmethod
        def run_evolution(**kwargs):
            called["cpp"] = True
            return {"best_mse": 10.0, "formula": "0", "nodes": [], "output_weights": []}

        @staticmethod
        def reduce_formula_noise(formula, X_list, y):
            return formula

        @staticmethod
        def simplify_formula(formula, **kwargs):
            return formula

    def _fake_fast_path(*args, **kwargs):
        return None

    monkeypatch.setattr(sw, "CPP_AVAILABLE", True)
    monkeypatch.setattr(sw, "get_cpp_core", lambda: _FakeCore)
    monkeypatch.setitem(
        sys.modules,
        "classifier_fast_path",
        SimpleNamespace(run_fast_path=_fake_fast_path),
    )

    rng = np.random.RandomState(23)
    X = rng.randn(140, 3)
    y = 1.5 * X[:, 0] - 0.75 * X[:, 1] ** 2 + 0.3 * np.sin(X[:, 2])

    est = GlassboxRegressor(
        use_fast_path=True,
        use_guided_evolution=False,
        use_universal_proposer=False,
        blackbox_mode=True,
        blackbox_standardize=False,
        blackbox_min_features_to_select=2,
        population_size=10,
        generations=10,
        multi_start_runs=1,
        timeout=20,
        random_state=23,
    )
    est.fit(X, y)

    assert called["cpp"] is False
    assert getattr(est, "blackbox_basis_model_", None) is not None
    assert est.best_mse_ < 0.5


def test_blackbox_demotes_unstable_fast_path_incumbent(monkeypatch):
    import glassbox.sr.sklearn_wrapper as sw

    called = {"cpp": False}

    class _FakeCore:
        @staticmethod
        def run_evolution(**kwargs):
            called["cpp"] = True
            return {"best_mse": 1.0, "formula": "0", "nodes": [], "output_weights": []}

        @staticmethod
        def reduce_formula_noise(formula, X_list, y):
            return formula

        @staticmethod
        def simplify_formula(formula, **kwargs):
            return formula

    def _fake_fast_path(*args, **kwargs):
        return {
            "formula": "2*x0 + 0.1*x1/(exp(x1)-1) + 0.05*x1**1.5",
            "mse": 1e-4,
            "operator_hints": {},
        }

    monkeypatch.setattr(sw, "CPP_AVAILABLE", True)
    monkeypatch.setattr(sw, "get_cpp_core", lambda: _FakeCore)
    monkeypatch.setitem(
        sys.modules,
        "classifier_fast_path",
        SimpleNamespace(run_fast_path=_fake_fast_path),
    )

    rng = np.random.RandomState(61)
    X = rng.randn(160, 2)
    X[:20, 1] = np.linspace(-1e-3, 1e-3, 20)
    y = 2.0 * X[:, 0]

    est = GlassboxRegressor(
        use_fast_path=True,
        use_guided_evolution=False,
        use_universal_proposer=False,
        blackbox_mode=True,
        blackbox_standardize=False,
        blackbox_min_features_to_select=2,
        population_size=10,
        generations=10,
        multi_start_runs=1,
        timeout=20,
        random_state=61,
    )
    est.fit(X, y)

    gate = est.blackbox_diagnostics_.get("fast_path_validation_gate", {})
    assert gate.get("accepted") is False
    assert "exp(x1)-1" not in est.get_formula()
    assert called["cpp"] is False


def test_blackbox_candidate_screening_handles_none_validation_values():
    est = GlassboxRegressor(random_state=67)
    candidates = [
        {"formula": "x0", "mse": 0.1, "validation_r2": None, "complexity": None},
        {"formula": "x1", "mse": None, "validation_r2": None, "complexity": 1},
    ]

    pruned = est._prune_blackbox_candidate_formulas(candidates, max_candidates=2)
    hints = est._derive_blackbox_operator_hints(None, pruned)

    assert len(pruned) == 2
    assert isinstance(hints, dict)


def test_univariate_specialist_candidate_pool_preserves_decomposition_seeds(
    monkeypatch,
):
    x = np.linspace(-2.0, 2.0, 80)
    X = x.reshape(-1, 1)
    y = np.sin(x) * np.cos(x) + 0.25 * x
    est = GlassboxRegressor(random_state=68)
    est._fp_result = {
        "candidate_formulas": [
            {
                "formula": "sin(x)*cos(x) + 0.25*x",
                "mse": 0.0,
                "source": "decomposition_probe",
                "decomposition_probe_type": "multiplicative_pair",
            }
        ],
        "details": {},
    }

    monkeypatch.setattr(
        est, "_targeted_specialist_probe_formulas", lambda *args, **kwargs: []
    )
    monkeypatch.setattr(est, "_build_blackbox_formula_pool", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        est,
        "_refine_candidate_formulas",
        lambda candidates, *args, **kwargs: candidates,
    )
    monkeypatch.setattr(
        est,
        "_prune_blackbox_candidate_formulas",
        lambda candidates, **kwargs: candidates,
    )

    candidates = est._build_univariate_specialist_candidate_formulas(
        "x",
        1.0,
        None,
        X,
        y,
        max_candidates=8,
    )

    assert any(c.get("source") == "decomposition_probe" for c in candidates)
    assert any(c.get("formula") == "sin(x)*cos(x) + 0.25*x" for c in candidates)


def test_blackbox_evolution_comparison_uses_validation_not_incumbent_train_mse():
    rng = np.random.RandomState(71)
    X = rng.randn(180, 2)
    X[:20, 1] = np.linspace(-1e-3, 1e-3, 20)
    y = 2.0 * X[:, 0]

    est = GlassboxRegressor(random_state=71)
    winner = est._compare_blackbox_formulas(
        "2*x0 + 0.1*x1/(exp(x1)-1) + 0.05*x1**1.5",
        "2*x0",
        X,
        y,
    )

    assert winner == "challenger"


def test_blackbox_high_uncertainty_disables_universal_fast_path(monkeypatch):
    import glassbox.sr.sklearn_wrapper as sw

    calls = []

    class _FakeCore:
        @staticmethod
        def run_evolution(**kwargs):
            return {
                "best_mse": 10.0,
                "formula": "x0+x1",
                "nodes": [],
                "output_weights": [],
            }

        @staticmethod
        def reduce_formula_noise(formula, X_list, y):
            return formula

        @staticmethod
        def simplify_formula(formula, **kwargs):
            return formula

    def _fake_fast_path(*args, **kwargs):
        calls.append(bool(kwargs.get("auto_expand", True)))
        if kwargs.get("auto_expand", True):
            return {
                "formula": "x0 + x1 + x0^1.5",
                "mse": 0.1,
                "operator_hints": {},
                "uncertainty": {
                    "prediction_entropy": 0.95,
                    "prediction_margin": 0.01,
                    "prediction_uncertain": True,
                },
            }
        return {
            "formula": "x0 + x1",
            "mse": 0.05,
            "operator_hints": {},
            "uncertainty": {
                "prediction_entropy": 0.95,
                "prediction_margin": 0.01,
                "prediction_uncertain": True,
            },
        }

    monkeypatch.setattr(sw, "CPP_AVAILABLE", True)
    monkeypatch.setattr(sw, "get_cpp_core", lambda: _FakeCore)
    monkeypatch.setitem(
        sys.modules,
        "classifier_fast_path",
        SimpleNamespace(run_fast_path=_fake_fast_path),
    )

    rng = np.random.RandomState(73)
    X = rng.randn(120, 3)
    y = X[:, 0] + X[:, 1]

    est = GlassboxRegressor(
        use_fast_path=True,
        use_guided_evolution=False,
        use_universal_proposer=False,
        blackbox_mode=True,
        blackbox_standardize=False,
        blackbox_min_features_to_select=2,
        population_size=10,
        generations=10,
        multi_start_runs=1,
        timeout=20,
        random_state=73,
    )
    est.fit(X, y)

    assert calls[:2] == [True, False]
    assert est.blackbox_diagnostics_.get("fast_path_auto_expand") is False


def test_blackbox_low_trust_constraints_drop_risky_operator_hints():
    from glassbox.sr.blackbox_preprocessor import BlackboxState

    est = GlassboxRegressor(random_state=79)
    est._fp_result = {
        "uncertainty": {
            "prediction_entropy": 0.95,
            "prediction_margin": 0.01,
            "prediction_uncertain": True,
        }
    }
    est.blackbox_diagnostics_ = {}
    state = BlackboxState(
        enabled=True,
        selected_features=[0, 1, 2],
        dropped_features=[],
        feature_scores={},
        ranker_votes={},
        x_mean=np.zeros(3),
        x_scale=np.ones(3),
        y_mean=0.0,
        y_scale=1.0,
        standardized=False,
        reason="selected_top_features",
        interaction_terms=["x0*x1"],
    )
    hints = est._constrain_blackbox_operator_hints(
        {
            "operators": {"periodic", "power", "exp", "log", "rational"},
            "powers": [2, 3, 5],
            "has_rational": True,
            "has_exp_decay": True,
        },
        state,
    )

    assert hints["operators"] == {"periodic"}
    assert hints["powers"] == [2, 3]
    assert hints["has_rational"] is False
    assert hints["has_exp_decay"] is False


def test_blackbox_binary_priors_are_conservative_under_low_trust():
    from glassbox.sr.blackbox_preprocessor import BlackboxState

    est = GlassboxRegressor(num_islands=4, random_state=83)
    est._fp_result = {
        "uncertainty": {
            "prediction_entropy": 0.95,
            "prediction_margin": 0.01,
            "prediction_uncertain": True,
        }
    }
    est.blackbox_diagnostics_ = {}
    state = BlackboxState(
        enabled=True,
        selected_features=[0, 1, 2],
        dropped_features=[],
        feature_scores={},
        ranker_votes={},
        x_mean=np.zeros(3),
        x_scale=np.ones(3),
        y_mean=0.0,
        y_scale=1.0,
        standardized=False,
        reason="selected_top_features",
        interaction_terms=["x0*x1"],
    )

    priors, multi = est._derive_blackbox_binary_priors(
        state,
        {"operators": {"periodic"}, "has_rational": False},
    )

    assert len(priors) == 3
    assert priors[1] < priors[0]
    assert priors[1] < priors[2]
    assert len(multi) == 4
    assert all(len(row) == 3 for row in multi)


def test_blackbox_unary_policy_is_conservative_under_low_trust():
    from glassbox.sr.blackbox_preprocessor import BlackboxState

    est = GlassboxRegressor(num_islands=4, random_state=84)
    est._fp_result = {
        "uncertainty": {
            "prediction_entropy": 0.95,
            "prediction_margin": 0.01,
            "prediction_uncertain": True,
        }
    }
    est.blackbox_diagnostics_ = {}
    state = BlackboxState(
        enabled=True,
        selected_features=[0, 1, 2],
        dropped_features=[],
        feature_scores={},
        ranker_votes={},
        x_mean=np.zeros(3),
        x_scale=np.ones(3),
        y_mean=0.0,
        y_scale=1.0,
        standardized=False,
        reason="selected_top_features",
        interaction_terms=["x0*x1"],
    )

    allowed_unary, multi_unary, allowed_binary, multi_binary = (
        est._derive_blackbox_unary_policy(
            state,
            {"operators": {"periodic"}, "has_rational": False},
        )
    )

    assert allowed_unary == []
    assert allowed_binary == []
    assert len(multi_unary) == 4
    assert len(multi_binary) == 4
    assert [0, 1, 2, 3, 4] in multi_unary
    assert [0, 1, 2] in multi_binary


def test_blackbox_cpp_receives_binary_priors(monkeypatch):
    import glassbox.sr.sklearn_wrapper as sw

    captured = {}

    class _FakeCore:
        @staticmethod
        def run_evolution(**kwargs):
            captured["allowed_unary_ops"] = kwargs.get("allowed_unary_ops")
            captured["binary_op_priors"] = kwargs.get("binary_op_priors")
            captured["allowed_binary_ops"] = kwargs.get("allowed_binary_ops")
            captured["multi_allowed_unary_ops"] = kwargs.get("multi_allowed_unary_ops")
            captured["multi_binary_op_priors"] = kwargs.get("multi_binary_op_priors")
            captured["multi_allowed_binary_ops"] = kwargs.get(
                "multi_allowed_binary_ops"
            )
            return {
                "best_mse": 10.0,
                "formula": "x0+x1",
                "nodes": [],
                "output_weights": [],
            }

        @staticmethod
        def reduce_formula_noise(formula, X_list, y):
            return formula

        @staticmethod
        def simplify_formula(formula, **kwargs):
            return formula

    def _fake_fast_path(*args, **kwargs):
        return {
            "formula": "x0 + x1",
            "mse": 10.0,
            "operator_hints": {
                "operators": {"periodic", "rational"},
                "has_rational": True,
                "has_exp_decay": False,
                "powers": [2],
            },
            "uncertainty": {
                "prediction_entropy": 0.95,
                "prediction_margin": 0.01,
                "prediction_uncertain": True,
            },
        }

    monkeypatch.setattr(sw, "CPP_AVAILABLE", True)
    monkeypatch.setattr(sw, "get_cpp_core", lambda: _FakeCore)
    monkeypatch.setitem(
        sys.modules,
        "classifier_fast_path",
        SimpleNamespace(run_fast_path=_fake_fast_path),
    )
    monkeypatch.setattr(
        sw.GlassboxRegressor,
        "_refine_candidate_formulas",
        lambda self, *args, **kwargs: [],
    )
    monkeypatch.setattr(
        sw.GlassboxRegressor,
        "_fit_blackbox_basis_model",
        lambda self, *args, **kwargs: None,
    )
    monkeypatch.setattr(
        sw.GlassboxRegressor,
        "_fit_blackbox_engineered_basis_model",
        lambda self, *args, **kwargs: None,
    )

    rng = np.random.RandomState(89)
    X = rng.randn(140, 3)
    y = X[:, 0] * np.sin(X[:, 1]) + 0.1 * X[:, 2]

    est = GlassboxRegressor(
        use_fast_path=True,
        use_guided_evolution=False,
        use_universal_proposer=False,
        blackbox_mode=True,
        blackbox_standardize=False,
        blackbox_min_features_to_select=2,
        population_size=10,
        generations=10,
        multi_start_runs=1,
        timeout=20,
        num_islands=4,
        random_state=89,
        evolution_skip_r2=0.999999,
        early_stop_mse=1e-12,
    )
    est.fit(X, y)

    assert captured["binary_op_priors"] is not None
    assert captured["allowed_unary_ops"] == []
    assert len(captured["binary_op_priors"]) == 3
    assert captured["binary_op_priors"][1] < captured["binary_op_priors"][0]
    assert captured["allowed_binary_ops"] == []
    assert len(captured["multi_allowed_unary_ops"]) == est.num_islands
    assert len(captured["multi_binary_op_priors"]) == est.num_islands
    assert len(captured["multi_allowed_binary_ops"]) == est.num_islands


def test_blackbox_candidate_screening_exports_interaction_operator_hints(monkeypatch):
    import glassbox.sr.sklearn_wrapper as sw

    called = {"cpp": False}

    class _FakeCore:
        @staticmethod
        def run_evolution(**kwargs):
            called["cpp"] = True
            return {"best_mse": 10.0, "formula": "0", "nodes": [], "output_weights": []}

        @staticmethod
        def reduce_formula_noise(formula, X_list, y):
            return formula

        @staticmethod
        def simplify_formula(formula, **kwargs):
            return formula

    def _fake_fast_path(*args, **kwargs):
        return None

    monkeypatch.setattr(sw, "CPP_AVAILABLE", True)
    monkeypatch.setattr(sw, "get_cpp_core", lambda: _FakeCore)
    monkeypatch.setitem(
        sys.modules,
        "classifier_fast_path",
        SimpleNamespace(run_fast_path=_fake_fast_path),
    )

    rng = np.random.RandomState(41)
    X = rng.randn(150, 3)
    y = X[:, 0] * np.sin(X[:, 1]) + 0.02 * rng.randn(150)

    est = GlassboxRegressor(
        use_fast_path=True,
        use_guided_evolution=False,
        use_universal_proposer=False,
        blackbox_mode=True,
        blackbox_standardize=False,
        blackbox_min_features_to_select=2,
        population_size=10,
        generations=10,
        multi_start_runs=1,
        timeout=20,
        random_state=41,
    )
    est.fit(X, y)

    screening = est.blackbox_diagnostics_.get("candidate_screening", {})
    assert screening.get("candidate_count", 0) > 0
    assert "periodic" in screening.get("interaction_operator_hints", [])
    specialist = screening.get("specialist_screening", {})
    assert specialist.get("enabled") is True
    assert specialist.get("candidate_count", 0) > 0
    assert specialist.get("segment_count", 0) >= 2
    assert specialist.get("segment_axis") in {"x0", "radius", "index"}
    assert specialist.get("top_candidates")


def test_blackbox_candidate_screening_exports_specialist_pair_diagnostics(monkeypatch):
    import glassbox.sr.sklearn_wrapper as sw

    called = {"cpp": False}

    class _FakeCore:
        @staticmethod
        def run_evolution(**kwargs):
            called["cpp"] = True
            return {"best_mse": 10.0, "formula": "0", "nodes": [], "output_weights": []}

        @staticmethod
        def reduce_formula_noise(formula, X_list, y):
            return formula

        @staticmethod
        def simplify_formula(formula, **kwargs):
            return formula

    def _fake_fast_path(*args, **kwargs):
        return None

    monkeypatch.setattr(sw, "CPP_AVAILABLE", True)
    monkeypatch.setattr(sw, "get_cpp_core", lambda: _FakeCore)
    monkeypatch.setitem(
        sys.modules,
        "classifier_fast_path",
        SimpleNamespace(run_fast_path=_fake_fast_path),
    )

    rng = np.random.RandomState(53)
    x = np.linspace(-3.0, 3.0, 160)
    X = np.column_stack([x, np.sin(x), np.cos(x)])
    y = np.where(x < 0.0, x * x, np.sin(2.0 * x))

    est = GlassboxRegressor(
        use_fast_path=True,
        use_guided_evolution=False,
        use_universal_proposer=False,
        blackbox_mode=True,
        blackbox_standardize=False,
        blackbox_min_features_to_select=2,
        population_size=10,
        generations=10,
        multi_start_runs=1,
        timeout=20,
        random_state=53,
    )
    est.fit(X, y)

    screening = est.blackbox_diagnostics_.get("candidate_screening", {})
    specialist = screening.get("specialist_screening", {})
    assert specialist.get("enabled") is True
    assert specialist.get("top_pairs")
    pair = specialist["top_pairs"][0]
    assert 0.0 <= pair.get("complementarity_score", -1.0) <= 1.0
    assert "formula_a" in pair and "formula_b" in pair
    assert "residual_correlation" in pair
    composition = est.blackbox_diagnostics_.get("specialist_composition_screening", {})
    assert composition.get("proposal_count", 0) >= 0


def test_blackbox_candidate_screening_can_disable_specialist_diagnostics(monkeypatch):
    import glassbox.sr.sklearn_wrapper as sw

    class _FakeCore:
        @staticmethod
        def run_evolution(**kwargs):
            return {"best_mse": 10.0, "formula": "0", "nodes": [], "output_weights": []}

        @staticmethod
        def reduce_formula_noise(formula, X_list, y):
            return formula

        @staticmethod
        def simplify_formula(formula, **kwargs):
            return formula

    def _fake_fast_path(*args, **kwargs):
        return None

    monkeypatch.setattr(sw, "CPP_AVAILABLE", True)
    monkeypatch.setattr(sw, "get_cpp_core", lambda: _FakeCore)
    monkeypatch.setitem(
        sys.modules,
        "classifier_fast_path",
        SimpleNamespace(run_fast_path=_fake_fast_path),
    )

    rng = np.random.RandomState(59)
    X = rng.randn(150, 3)
    y = X[:, 0] * np.sin(X[:, 1])

    est = GlassboxRegressor(
        use_fast_path=True,
        use_guided_evolution=False,
        use_universal_proposer=False,
        blackbox_mode=True,
        blackbox_standardize=False,
        blackbox_min_features_to_select=2,
        enable_specialist_screening_diagnostics=False,
        population_size=10,
        generations=10,
        multi_start_runs=1,
        timeout=20,
        random_state=59,
    )
    est.fit(X, y)

    screening = est.blackbox_diagnostics_.get("candidate_screening", {})
    assert screening.get("candidate_count", 0) > 0
    assert "specialist_screening" not in screening


def test_blackbox_candidate_screening_can_accept_specialist_compositions(monkeypatch):
    import glassbox.sr.sklearn_wrapper as sw

    class _FakeCore:
        @staticmethod
        def run_evolution(**kwargs):
            return {"best_mse": 10.0, "formula": "0", "nodes": [], "output_weights": []}

        @staticmethod
        def reduce_formula_noise(formula, X_list, y):
            return formula

        @staticmethod
        def simplify_formula(formula, **kwargs):
            return formula

    def _fake_fast_path(*args, **kwargs):
        return None

    monkeypatch.setattr(sw, "CPP_AVAILABLE", True)
    monkeypatch.setattr(sw, "get_cpp_core", lambda: _FakeCore)
    monkeypatch.setitem(
        sys.modules,
        "classifier_fast_path",
        SimpleNamespace(run_fast_path=_fake_fast_path),
    )

    x = np.linspace(-3.0, 3.0, 180)
    X = np.column_stack([x, np.sin(2.0 * x), np.cos(2.0 * x)])
    y = np.where(x < 0.0, x * x, np.sin(2.0 * x))

    est = GlassboxRegressor(
        use_fast_path=True,
        use_guided_evolution=False,
        use_universal_proposer=False,
        blackbox_mode=True,
        blackbox_standardize=False,
        blackbox_min_features_to_select=2,
        population_size=10,
        generations=10,
        multi_start_runs=1,
        timeout=20,
        random_state=61,
    )
    est.fit(X, y)

    composition = est.blackbox_diagnostics_.get("specialist_composition_screening", {})
    assert composition.get("proposal_count", 0) >= 1
    assert composition.get("accepted_count", 0) >= 1


def test_blackbox_candidate_pool_can_skip_cpp_from_interaction_formula(monkeypatch):
    import glassbox.sr.sklearn_wrapper as sw

    called = {"cpp": False}

    class _FakeCore:
        @staticmethod
        def run_evolution(**kwargs):
            called["cpp"] = True
            return {"best_mse": 10.0, "formula": "0", "nodes": [], "output_weights": []}

        @staticmethod
        def reduce_formula_noise(formula, X_list, y):
            return formula

        @staticmethod
        def simplify_formula(formula, **kwargs):
            return formula

    def _fake_fast_path(*args, **kwargs):
        return None

    monkeypatch.setattr(sw, "CPP_AVAILABLE", True)
    monkeypatch.setattr(sw, "get_cpp_core", lambda: _FakeCore)
    monkeypatch.setitem(
        sys.modules,
        "classifier_fast_path",
        SimpleNamespace(run_fast_path=_fake_fast_path),
    )

    rng = np.random.RandomState(43)
    X = rng.randn(180, 3)
    y = X[:, 0] * np.sin(X[:, 1])

    est = GlassboxRegressor(
        use_fast_path=True,
        use_guided_evolution=False,
        use_universal_proposer=False,
        blackbox_mode=True,
        blackbox_standardize=False,
        blackbox_min_features_to_select=2,
        population_size=10,
        generations=10,
        multi_start_runs=1,
        timeout=20,
        random_state=43,
        early_stop_mse=1e-6,
    )
    est.fit(X, y)

    assert called["cpp"] is False
    assert "sin" in est.get_formula()
    outcome = est.blackbox_diagnostics_.get("selection_outcome", {})
    inflation = est.blackbox_diagnostics_.get("search_inflation", {})
    assert outcome.get("candidate_screening_win") is True
    assert outcome.get("evolution_ran") is False
    assert inflation.get("screening_budget", 0) >= inflation.get("seed_budget", 0)
    assert est.blackbox_diagnostics_.get("domain_failure_rate") == 0.0


def test_evolution_result_is_selected_via_direct_formula_evaluation(monkeypatch):
    import glassbox.sr.sklearn_wrapper as sw

    class _FakeCore:
        @staticmethod
        def run_evolution(**kwargs):
            return {
                "best_mse": 100.0,
                "formula": "x0+x1",
                "nodes": [],
                "output_weights": [],
            }

        @staticmethod
        def reduce_formula_noise(formula, X_list, y):
            return formula

        @staticmethod
        def simplify_formula(formula, **kwargs):
            return formula

    def _fake_fast_path(*args, **kwargs):
        return {
            "formula": "0",
            "mse": 0.01,
            "operator_hints": {},
        }

    monkeypatch.setattr(sw, "CPP_AVAILABLE", True)
    monkeypatch.setattr(sw, "get_cpp_core", lambda: _FakeCore)
    monkeypatch.setitem(
        sys.modules,
        "classifier_fast_path",
        SimpleNamespace(run_fast_path=_fake_fast_path),
    )

    rng = np.random.RandomState(31)
    X = rng.randn(80, 2)
    y = X[:, 0] + X[:, 1]

    est = GlassboxRegressor(
        use_fast_path=True,
        use_guided_evolution=False,
        use_universal_proposer=False,
        blackbox_mode=False,
        population_size=10,
        generations=10,
        multi_start_runs=1,
        timeout=20,
        random_state=31,
        evolution_skip_r2=0.999999,
    )
    est.fit(X, y)

    assert "x0" in est.get_formula() and "x1" in est.get_formula()
    assert getattr(est, "evolution_candidate_formula_", None) == "x0+x1"


def test_blackbox_pareto_selector_prefers_stable_simple_formula():
    rng = np.random.RandomState(53)
    X = rng.randn(160, 2)
    y = 2.0 * X[:, 0] + 0.05 * rng.randn(160)

    est = GlassboxRegressor(random_state=53)
    choice = est._select_blackbox_pareto_formula(
        [
            {"formula": "2*x0", "source": "stable"},
            {"formula": "2*x0 + 0.01*x1/(exp(x1)-1)", "source": "risky"},
        ],
        X,
        y,
    )

    assert choice is not None
    assert choice["source"] == "stable"
    assert choice["risk_score"] < 0.2


def test_constant_refinement_improves_candidate_validation_mse():
    rng = np.random.RandomState(59)
    X = rng.randn(180, 1)
    y = 2.5 * X[:, 0] + 0.75

    est = GlassboxRegressor(random_state=59)
    split = est._domain_edge_validation_split(X, y)
    base_pred = est._safe_eval_formula_array("1.2*x0+0.1", split["X_val"])
    base_mse = float(np.mean((base_pred - split["y_val"]) ** 2))
    refined = est._refine_formula_constants(
        "1.2*x0+0.1",
        split["X_fit"],
        split["y_fit"],
        split["X_val"],
        split["y_val"],
    )

    assert refined is not None
    assert refined["validation_mse"] < base_mse
    assert refined["constant_refined"] is True


def test_cleanup_guard_rejects_display_mse_regression(monkeypatch):
    X = np.linspace(-1.0, 1.0, 40).reshape(-1, 1)
    y = X[:, 0]
    est = GlassboxRegressor(random_state=61)
    est.blackbox_diagnostics_ = {}

    monkeypatch.setattr(
        est, "_reduce_formula_noise", lambda formula, X_in, y_in: "display_bad"
    )
    monkeypatch.setattr(est, "_simplify_formula", lambda formula: formula)
    monkeypatch.setattr(
        est,
        "_formula_mse",
        lambda formula, X_in, y_in, **kwargs: {
            "display_good": 1e-4,
            "display_bad": 1e-5,
        }.get(formula, float("inf")),
    )
    monkeypatch.setattr(
        est,
        "_display_formula_mse",
        lambda formula, X_in, y_in, **kwargs: {
            "display_good": 1e-4,
            "display_bad": 1e-1,
        }.get(formula, float("inf")),
    )

    selected = est._cleanup_formula_with_fidelity_guard("display_good", X, y)

    assert selected == "display_good"
    guard = est.blackbox_diagnostics_["formula_cleanup_guard"][-1]["steps"][0]
    assert guard["accepted"] is False
    assert guard["after_display_mse"] == 1e-1


def test_final_formula_selection_prefers_display_score(monkeypatch):
    X = np.linspace(-1.0, 1.0, 40).reshape(-1, 1)
    y = X[:, 0]
    est = GlassboxRegressor(random_state=62)

    monkeypatch.setattr(
        est,
        "_formula_mse",
        lambda formula, X_in, y_in, **kwargs: {
            "incumbent": 1e-4,
            "challenger": 1e-6,
        }.get(formula, float("inf")),
    )
    monkeypatch.setattr(
        est,
        "_display_formula_mse",
        lambda formula, X_in, y_in, **kwargs: {
            "incumbent": 1e-4,
            "challenger": 1e-1,
        }.get(formula, float("inf")),
    )

    formula, mse, source = est._select_final_formula(
        "incumbent", 1e-4, "challenger", 1e-6, X, y
    )

    assert formula == "incumbent"
    assert mse == 1e-4
    assert source == "incumbent"
    assert est.final_formula_selection_diagnostics_["selected"] == "incumbent"
    assert est.final_formula_selection_diagnostics_["challenger_display_mse"] == 1e-1
