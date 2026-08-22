import numpy as np

from glassbox.curve_classifier import (
    CURVE_CLASSIFIER_MULTIVARIATE_NEURAL_MODE,
    CURVE_CLASSIFIER_UNIVARIATE_NEURAL_MODE,
    describe_curve_classifier_inference,
)
from glassbox.universal_proposer import (
    UNIVERSAL_PROPOSER_MULTIVARIATE_NEURAL_MODE,
    UNIVERSAL_PROPOSER_UNIVARIATE_NEURAL_MODE,
    UniversalProposer,
    UniversalProposerConfig,
    propose_from_xy,
    proposer_output_to_fpip_v2,
)


def test_classifier_inference_contract_marks_multivariate_as_heuristic():
    x = np.zeros((32, 3), dtype=np.float32)

    contract = describe_curve_classifier_inference(x)

    assert contract["input_mode"] == "multivariate"
    assert contract["status"] == "heuristic_multivariate"
    assert contract["neural_feature_mode"] == CURVE_CLASSIFIER_MULTIVARIATE_NEURAL_MODE
    assert contract["supports_trained_multivariate_neural_model"] is False
    assert contract["n_input_features"] == 3


def test_classifier_inference_contract_keeps_univariate_trained_status():
    x = np.linspace(-1.0, 1.0, 32, dtype=np.float32)

    contract = describe_curve_classifier_inference(x)

    assert contract["input_mode"] == "univariate"
    assert contract["status"] == "trained_univariate_neural"
    assert contract["neural_feature_mode"] == CURVE_CLASSIFIER_UNIVARIATE_NEURAL_MODE


def test_proposer_multivariate_payload_exposes_heuristic_neural_contract():
    model = UniversalProposer(UniversalProposerConfig(hidden_dim=32, max_input_vars=3))
    x0 = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
    x1 = np.linspace(0.5, 2.0, 64, dtype=np.float32)
    x = np.stack([x0, x1], axis=1)
    y = (x0 * x1).astype(np.float32)

    out = propose_from_xy(model, x, y, top_k=4)
    payload = proposer_output_to_fpip_v2(out)

    assert out["supports_multivariate_formulas"] is True
    assert out["neural_feature_mode"] == UNIVERSAL_PROPOSER_MULTIVARIATE_NEURAL_MODE
    assert out["model_contract"]["neural_multivariate_support"] == "heuristic"
    assert out["model_contract"]["supports_trained_multivariate_neural_model"] is False
    assert out["search_plan"]["neural_multivariate_support"] == "heuristic"
    assert (
        out["search_plan"]["operator_prior_source"]
        == "one_dimensional_y_projection_features"
    )
    assert payload["model_contract"] == out["model_contract"]
    assert payload["search_plan"]["model_contract"] == out["model_contract"]


def test_proposer_univariate_payload_exposes_canonical_neural_contract():
    model = UniversalProposer(UniversalProposerConfig(hidden_dim=32))
    x = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
    y = np.sin(x).astype(np.float32)

    out = propose_from_xy(model, x, y, top_k=3)

    assert out["supports_multivariate_formulas"] is False
    assert out["neural_feature_mode"] == UNIVERSAL_PROPOSER_UNIVARIATE_NEURAL_MODE
    assert out["model_contract"]["input_mode"] == "univariate"
    assert (
        out["search_plan"]["operator_prior_source"]
        == "canonicalized_univariate_xy_features"
    )
