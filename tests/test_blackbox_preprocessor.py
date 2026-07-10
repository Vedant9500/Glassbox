import numpy as np

from glassbox.sr.blackbox_preprocessor import (
    build_blackbox_seed_formulas,
    compute_blackbox_feature_ranking,
    discover_blackbox_interactions,
    formula_from_search_to_original_space,
    prepare_blackbox_search,
    remap_reduced_formula_to_original,
    remap_original_formula_to_reduced,
)


def test_prepare_blackbox_search_selects_informative_features():
    rng = np.random.RandomState(0)
    X = rng.randn(200, 8)
    y = 3.0 * X[:, 2] - 2.0 * X[:, 5] ** 2 + 0.01 * rng.randn(200)

    X_search, y_search, state = prepare_blackbox_search(
        X,
        y,
        enabled=True,
        max_features=3,
        standardize=False,
        min_features_to_select=5,
    )

    assert state.enabled is True
    assert X_search.shape[1] == 3
    assert 2 in state.selected_features
    assert 5 in state.selected_features
    assert isinstance(state.ranker_votes, dict)
    assert state.ranker_votes
    assert y_search.shape == y.shape


def test_prepare_blackbox_search_keeps_small_multivariate_inputs():
    rng = np.random.RandomState(1)
    X = rng.randn(50, 4)
    y = X[:, 0] - 2.0 * X[:, 3]

    X_search, y_search, state = prepare_blackbox_search(
        X,
        y,
        enabled=True,
        max_features=2,
        standardize=False,
        min_features_to_select=5,
    )

    assert state.enabled is True
    assert state.reason == "retained_all_features_small_problem"
    assert X_search.shape[1] == 4
    assert state.selected_features == [0, 1, 2, 3]
    assert y_search.shape == y.shape


def test_prepare_blackbox_search_imputes_non_finite_values():
    rng = np.random.RandomState(21)
    X = rng.randn(80, 4)
    y = X[:, 0] + X[:, 2]
    X[3, 0] = np.nan
    X[7, 2] = np.inf
    y[5] = np.nan

    X_search, y_search, state = prepare_blackbox_search(
        X,
        y,
        enabled=True,
        max_features=3,
        standardize=True,
        min_features_to_select=2,
    )

    assert state.enabled is True
    assert np.all(np.isfinite(X_search))
    assert np.all(np.isfinite(y_search))


def test_prepare_blackbox_search_can_disable_interaction_discovery():
    rng = np.random.RandomState(22)
    X = rng.randn(120, 3)
    y = X[:, 0] * X[:, 1]

    _, _, state = prepare_blackbox_search(
        X,
        y,
        enabled=True,
        max_features=3,
        standardize=False,
        min_features_to_select=2,
        interaction_search=False,
    )

    assert state.interaction_pairs == []
    assert state.interaction_terms == []
    assert state.interaction_scores == {}


def test_reduced_formula_remaps_to_original_features():
    mapped = remap_reduced_formula_to_original("x0 + sin(x2)", [3, 5, 7])
    assert mapped == "x3 + sin(x7)"


def test_original_formula_remaps_to_reduced_features():
    mapped = remap_original_formula_to_reduced("x3 + sin(x7)", [3, 5, 7])
    assert mapped == "x0 + sin(x2)"


def test_standardized_formula_maps_back_with_inverse_target_transform():
    X = np.column_stack([np.arange(10), np.arange(10) * 2.0])
    y = 10.0 + 3.0 * X[:, 1]
    _, _, state = prepare_blackbox_search(
        X,
        y,
        enabled=True,
        max_features=1,
        standardize=True,
        min_features_to_select=2,
    )

    formula = formula_from_search_to_original_space("2*x0", state)
    assert f"x{state.selected_features[0]}" in formula
    assert "*" in formula


def test_discover_blackbox_interactions_returns_pairs():
    rng = np.random.RandomState(4)
    X = rng.randn(120, 4)
    y = X[:, 0] * X[:, 2] + 0.05 * rng.randn(120)

    interactions = discover_blackbox_interactions(X, y, selected_features=[0, 1, 2, 3], max_pairs=3)

    assert interactions["interaction_pairs"]
    assert interactions["interaction_terms"]
    assert any("x0*x2" in term or "x2*x0" in term for term in interactions["interaction_terms"])


def test_blackbox_state_includes_interactions():
    rng = np.random.RandomState(5)
    X = rng.randn(150, 5)
    y = X[:, 1] * X[:, 4] + 0.01 * rng.randn(150)

    _, _, state = prepare_blackbox_search(
        X,
        y,
        enabled=True,
        max_features=4,
        standardize=False,
        min_features_to_select=2,
    )

    assert isinstance(state.interaction_terms, list)
    assert isinstance(state.interaction_scores, dict)


def test_compute_blackbox_feature_ranking_exposes_ranker_votes():
    rng = np.random.RandomState(12)
    X = rng.randn(160, 6)
    y = 2.5 * X[:, 1] - 1.2 * X[:, 4] ** 2 + 0.03 * rng.randn(160)

    ranking = compute_blackbox_feature_ranking(X, y)

    assert "feature_scores" in ranking
    assert "ranker_votes" in ranking
    assert ranking["ranker_votes"]
    assert "holdout_poly" in ranking["ranker_votes"]
    top_features = sorted(
        ranking["feature_scores"],
        key=lambda idx: ranking["feature_scores"][idx],
        reverse=True,
    )[:3]
    assert 1 in top_features
    assert 4 in top_features


def test_uncertain_feature_selection_retains_small_candidate_set():
    rng = np.random.RandomState(6)
    X = rng.randn(120, 5)
    y = rng.randn(120)

    X_search, _, state = prepare_blackbox_search(
        X,
        y,
        enabled=True,
        max_features=4,
        standardize=False,
        min_features_to_select=2,
    )

    assert state.feature_selection_uncertain is True
    assert state.reason == "retained_all_features_uncertain_selection"
    assert X_search.shape[1] == 5


def test_interaction_discovery_includes_nonlinear_terms():
    rng = np.random.RandomState(7)
    X = rng.randn(180, 3)
    y = X[:, 0] * np.sin(X[:, 1]) + 0.01 * rng.randn(180)

    interactions = discover_blackbox_interactions(X, y, selected_features=[0, 1, 2], max_pairs=5)

    assert any("sin" in term for term in interactions["interaction_terms"])


def test_blackbox_seed_formulas_include_features_and_interactions():
    formulas = build_blackbox_seed_formulas([3, 7], interaction_terms=["x3*sin(x7)"], max_seeds=20)

    assert "x3" in formulas
    assert "sin(x7)" in formulas
    assert "x3*sin(x7)" in formulas


def test_interaction_scoring_prefers_holdout_stable_signal():
    rng = np.random.RandomState(9)
    X = rng.randn(220, 3)
    y = X[:, 0] * X[:, 1] + 0.02 * rng.randn(220)

    interactions = discover_blackbox_interactions(
        X,
        y,
        selected_features=[0, 1, 2],
        max_pairs=4,
        validation_fraction=0.25,
    )

    assert interactions["interaction_terms"]
    best_term = interactions["interaction_terms"][0]
    assert "x0*x1" in best_term or "x1*x0" in best_term


def test_interaction_discovery_prunes_redundant_variants():
    rng = np.random.RandomState(19)
    x0 = rng.uniform(0.8, 1.2, 180)
    x1 = 2.0 * x0 + 1e-4 * rng.randn(180)
    x2 = rng.randn(180)
    X = np.column_stack([x0, x1, x2])
    y = x0 * x1 + 0.01 * rng.randn(180)

    interactions = discover_blackbox_interactions(
        X,
        y,
        selected_features=[0, 1, 2],
        max_pairs=6,
    )

    terms = interactions["interaction_terms"]
    assert len(terms) == len(set(terms))
    multiplicative_terms = [term for term in terms if "*" in term and "sin" not in term and "cos" not in term]
    assert len(multiplicative_terms) <= 2
