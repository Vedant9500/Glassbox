import numpy as np

from glassbox.sr.blackbox_preprocessor import (
    build_blackbox_seed_formulas,
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
