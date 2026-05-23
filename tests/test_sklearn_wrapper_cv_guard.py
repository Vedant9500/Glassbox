import sys
from types import SimpleNamespace

import numpy as np

from glassbox.sr.sklearn_wrapper import GlassboxRegressor
from glassbox.sr.blackbox_preprocessor import prepare_blackbox_search


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


def test_universal_proposer_dual_path_handles_multivariate_proxy(monkeypatch):
    n = 64
    x1 = np.linspace(-2.0, 2.0, n)
    x2 = np.linspace(1.0, 3.0, n)
    X = np.stack([x1, x2], axis=1)
    y = x1 + x2

    est = GlassboxRegressor(
        use_universal_proposer=True,
        universal_proposer_shadow_mode=True,
        universal_proposer_log_routing=False,
    )

    payload, force = est._run_universal_proposer_dual_path(X, y, fast_path_result=None)

    assert payload is not None
    assert force is False
    assert est.universal_proposer_status_ == "ok_multivariate_proxy"
    assert payload["interaction_hints"]["multivariate_proxy"] is True


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

    payload, force = est._run_universal_proposer_dual_path(X, y, fast_path_result={"mse": 0.1})

    assert payload is None
    assert force is False
    assert str(est.universal_proposer_status_).startswith("error:")


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
    assert plan["seed_budget"] > 8
    assert plan["acceptable_complexity"] > 15
    assert plan["early_stop_max_nodes"] > 50


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
            return {"best_mse": 0.0, "formula": "x0*x1", "nodes": [], "output_weights": []}

        @staticmethod
        def reduce_formula_noise(formula, X_list, y):
            return formula

        @staticmethod
        def simplify_formula(formula, **kwargs):
            return formula

    def _fake_fast_path(*args, **kwargs):
        return None

    monkeypatch.setattr(sw, "CPP_AVAILABLE", True)
    monkeypatch.setattr(sw, "_core", _FakeCore)
    monkeypatch.setitem(sys.modules, "classifier_fast_path", SimpleNamespace(run_fast_path=_fake_fast_path))

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
    assert captured["acceptable_complexity"] == est.blackbox_search_plan_["acceptable_complexity"]
    assert captured["early_stop_max_nodes"] == est.blackbox_search_plan_["early_stop_max_nodes"]
