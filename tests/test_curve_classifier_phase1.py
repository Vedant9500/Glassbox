from pathlib import Path

import numpy as np
import pytest

from glassbox.curve_classifier.generate_curve_data import (
    FEATURE_DIM,
    extract_all_features_xy,
    prepare_univariate_curve_xy,
)
from glassbox.curve_classifier.curve_classifier_integration import predict_operators
from glassbox.universal_proposer import (
    UniversalProposer,
    UniversalProposerConfig,
    propose_fpip_v2_from_xy,
)


def test_prepare_univariate_curve_xy_sorts_and_averages_duplicate_x():
    x = np.array([2.0, 0.0, 1.0, 1.0, 3.0])
    y = np.array([4.0, 0.0, 1.0, 3.0, 9.0])

    x_grid, y_grid = prepare_univariate_curve_xy(x, y, n_points=4)

    np.testing.assert_allclose(x_grid, np.array([0.0, 1.0, 2.0, 3.0]))
    np.testing.assert_allclose(y_grid, np.array([0.0, 2.0, 4.0, 9.0]))


def test_extract_all_features_xy_is_invariant_to_duplicate_row_expansion():
    x_base = np.linspace(-3.0, 3.0, 128)
    y_base = np.cos(x_base) + 0.2 * x_base

    x_dup = np.repeat(x_base, 3)
    y_dup = np.repeat(y_base, 3)
    perm = np.random.default_rng(42).permutation(len(x_dup))

    features_base = extract_all_features_xy(x_base, y_base)
    features_dup = extract_all_features_xy(x_dup[perm], y_dup[perm])

    assert features_base.shape == (FEATURE_DIM,)
    np.testing.assert_allclose(features_base, features_dup, atol=1e-7, rtol=1e-7)


def test_prepare_univariate_curve_xy_handles_nonuniform_sampling():
    rng = np.random.default_rng(123)
    x_irregular = np.sort(rng.uniform(-4.0, 4.0, size=600))
    y_irregular = np.sin(x_irregular) + 0.05 * x_irregular ** 2

    x_grid, y_grid = prepare_univariate_curve_xy(x_irregular, y_irregular, n_points=256)
    y_expected = np.sin(x_grid) + 0.05 * x_grid ** 2

    assert np.all(np.diff(x_grid) > 0.0)
    np.testing.assert_allclose(y_grid, y_expected, atol=0.015, rtol=0.015)


def test_public_predict_operators_is_row_order_invariant_if_checkpoint_available():
    model_path = Path("models/curve_classifier_multi.pt")
    if not model_path.exists():
        pytest.skip("Local curve classifier checkpoint is not available")

    x = np.linspace(-5.0, 5.0, 256)
    y = np.sin(x) + 0.1 * x
    perm = np.random.default_rng(456).permutation(len(x))

    sorted_preds = predict_operators(x, y, model_path=str(model_path), threshold=0.1, device="cpu")
    shuffled_preds = predict_operators(x[perm], y[perm], model_path=str(model_path), threshold=0.1, device="cpu")

    assert sorted_preds.keys() == shuffled_preds.keys()
    for key in sorted_preds:
        assert sorted_preds[key] == pytest.approx(shuffled_preds[key], abs=1e-6)


def test_public_proposer_fpip_output_is_row_order_invariant():
    model = UniversalProposer(UniversalProposerConfig(hidden_dim=32))
    x = np.linspace(-2.5, 2.5, 128, dtype=np.float32)
    y = (np.sin(x) + 0.25 * x).astype(np.float32)
    perm = np.random.default_rng(789).permutation(len(x))

    sorted_payload = propose_fpip_v2_from_xy(model, x=x, y=y, top_k=5)
    shuffled_payload = propose_fpip_v2_from_xy(model, x=x[perm], y=y[perm], top_k=5)

    assert sorted_payload["valid"] is True
    assert shuffled_payload["valid"] is True
    assert sorted_payload["operator_priors"] == pytest.approx(shuffled_payload["operator_priors"], abs=1e-6)
    assert [c["formula"] for c in sorted_payload["candidate_skeletons"]] == [
        c["formula"] for c in shuffled_payload["candidate_skeletons"]
    ]
