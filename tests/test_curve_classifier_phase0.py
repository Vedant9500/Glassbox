import importlib
from pathlib import Path

import numpy as np
import pytest
import torch

from glassbox.curve_classifier import curve_classifier_integration as cci
from glassbox.curve_classifier import models as classifier_models
from glassbox.curve_classifier.generate_curve_data import (
    FEATURE_DIM,
    extract_all_features_xy,
)
from glassbox.universal_proposer import (
    UniversalProposer,
    UniversalProposerConfig,
    propose_from_xy,
    validate_universal_proposer_checkpoint_metadata,
)


def test_univariate_xy_features_are_row_order_invariant():
    x = np.linspace(-5.0, 5.0, 256)
    y = np.sin(x) + 0.1 * x
    perm = np.random.default_rng(123).permutation(len(x))

    features_sorted = extract_all_features_xy(x, y)
    features_shuffled = extract_all_features_xy(x[perm], y[perm])

    assert features_sorted.shape == (FEATURE_DIM,)
    np.testing.assert_allclose(features_sorted, features_shuffled, atol=1e-7, rtol=1e-7)


def test_shared_classifier_classes_are_bound_across_training_and_inference():
    train_mod = importlib.import_module(
        "glassbox.curve_classifier.train_curve_classifier"
    )

    assert train_mod.CurveClassifierGLU is classifier_models.CurveClassifierGLU
    assert train_mod.CurveClassifierMLP is classifier_models.CurveClassifierMLP
    assert train_mod.CurveClassifierCNN is classifier_models.CurveClassifierCNN
    assert train_mod.EQLLayer is classifier_models.EQLLayer

    assert cci.CurveClassifierGLU is classifier_models.CurveClassifierGLU
    assert cci.CurveClassifierMLP is classifier_models.CurveClassifierMLP
    assert cci.CurveClassifierCNN is classifier_models.CurveClassifierCNN
    assert cci.EQLLayer is classifier_models.EQLLayer


def test_eql_outputs_match_training_and_inference_names():
    train_mod = importlib.import_module(
        "glassbox.curve_classifier.train_curve_classifier"
    )
    torch.manual_seed(7)
    train_layer = train_mod.EQLLayer(in_features=10, out_features=13)
    infer_layer = cci.EQLLayer(in_features=10, out_features=13)
    infer_layer.load_state_dict(train_layer.state_dict())
    train_layer.eval()
    infer_layer.eval()

    x = torch.randn(4, 10)
    with torch.no_grad():
        train_out = train_layer(x)
        infer_out = infer_layer(x)

    torch.testing.assert_close(train_out, infer_out)


def test_classifier_checkpoint_metadata_validation_accepts_legacy_unversioned():
    checkpoint = {
        "model_state_dict": {},
        "model_type": "glu",
        "model_config": {"n_features": 398},
        "feature_dim": 398,
        "feature_schema": {"raw": (0, 128)},
    }

    report = cci.validate_curve_classifier_checkpoint_metadata(checkpoint)

    assert report["feature_dim"] == 398
    assert report["architecture_version"] == "legacy_unversioned"
    assert any("architecture_version" in msg for msg in report["warnings"])


def test_universal_proposer_checkpoint_metadata_validation_accepts_legacy_unversioned():
    checkpoint = {
        "model_state_dict": {},
        "config": {
            "hidden_dim": 32,
            "n_features": 398,
            "operator_vocab": ["identity"],
            "skeleton_vocab": ["x"],
        },
    }

    report = validate_universal_proposer_checkpoint_metadata(checkpoint)

    assert report["n_features"] == 398
    assert report["architecture_version"] == "legacy_unversioned"


def test_classifier_checkpoint_prediction_is_row_order_invariant_if_available():
    model_path = Path("models/curve_classifier_multi.pt")
    if not model_path.exists():
        pytest.skip("Local curve classifier checkpoint is not available")

    x = np.linspace(-5.0, 5.0, 256)
    y = np.sin(x)
    perm = np.random.default_rng(123).permutation(len(x))

    device = cci._resolve_device("cpu")
    model = cci.load_classifier(str(model_path), device="cpu")
    cache_key = cci._make_cache_key(
        str(cci._resolve_model_path(str(model_path))), device
    )
    metadata = cci._cached_metadata_by_device[cache_key]

    def _probs(x_values, y_values):
        features = cci._prepare_curve_features(
            extract_all_features_xy(x_values, y_values),
            metadata.get("feature_scaler"),
        )
        return cci._predict_pytorch(model, features, metadata, device)

    sorted_probs = _probs(x, y)
    shuffled_probs = _probs(x[perm], y[perm])

    assert float(np.max(np.abs(sorted_probs - shuffled_probs))) < 0.03


def test_universal_proposer_univariate_output_is_row_order_invariant():
    model = UniversalProposer(UniversalProposerConfig(hidden_dim=32))
    x = np.linspace(-3.0, 3.0, 96, dtype=np.float32)
    y = np.sin(x).astype(np.float32)
    perm = np.random.default_rng(321).permutation(len(x))

    sorted_out = propose_from_xy(model, x, y, top_k=4)
    shuffled_out = propose_from_xy(model, x[perm], y[perm], top_k=4)

    assert [c["formula"] for c in sorted_out["candidate_skeletons"]] == [
        c["formula"] for c in shuffled_out["candidate_skeletons"]
    ]
    assert (
        sorted_out["operator_priors"].keys() == shuffled_out["operator_priors"].keys()
    )
    for key in sorted_out["operator_priors"]:
        assert sorted_out["operator_priors"][key] == pytest.approx(
            shuffled_out["operator_priors"][key],
            abs=1e-6,
        )
