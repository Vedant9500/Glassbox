import numpy as np

from glassbox.sr.blackbox_preprocessor import (
    formula_from_search_to_original_space,
    prepare_blackbox_search,
    remap_reduced_formula_to_original,
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


def test_reduced_formula_remaps_to_original_features():
    mapped = remap_reduced_formula_to_original("x0 + sin(x2)", [3, 5, 7])
    assert mapped == "x3 + sin(x7)"


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
