import numpy as np

from glassbox.universal_proposer import (
    UNIVERSAL_PROPOSER_ROLE,
    UniversalProposer,
    UniversalProposerConfig,
    propose_from_xy,
    proposer_output_to_fpip_v2,
)
import glassbox.universal_proposer.universal_proposer as up


def _minimal_proposer_output(sequence_uncertainty, best_relative_mse):
    return {
        "candidate_skeletons": [{"formula": "x", "mse": 1.0, "probability": 0.5}],
        "operator_priors": {"identity": 0.8},
        "sequence_uncertainty": dict(sequence_uncertainty),
        "search_plan": {"signals": {"best_relative_mse": best_relative_mse}},
        "model_contract": {"input_mode": "univariate"},
        "proposer_contract": {"role": UNIVERSAL_PROPOSER_ROLE},
    }


def test_unvalidated_model_keeps_skeleton_logits_diagnostic_only():
    model = UniversalProposer(UniversalProposerConfig(hidden_dim=32))
    x = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
    y = np.sin(x).astype(np.float32)

    out = propose_from_xy(model, x, y, top_k=3)
    uncertainty = out["sequence_uncertainty"]

    assert uncertainty["confidence_source"] == "disabled_unvalidated_skeleton_head"
    assert uncertainty["skeleton_confidence_reliable"] is False
    assert uncertainty["confident"] is False
    assert uncertainty["raw_entropy"] is not None
    assert uncertainty["entropy"] is None
    assert out["proposer_contract"]["role"] == UNIVERSAL_PROPOSER_ROLE
    assert out["proposer_contract"]["skeleton_head_role"] == "diagnostic_only"


def test_unvalidated_raw_skeleton_confidence_does_not_disable_guided_routing():
    proposer_output = _minimal_proposer_output(
        {
            "entropy": None,
            "margin": None,
            "raw_entropy": 0.01,
            "raw_margin": 0.99,
            "raw_confident": True,
            "confident": False,
            "skeleton_confidence_reliable": False,
            "confidence_source": "disabled_unvalidated_skeleton_head",
        },
        best_relative_mse=0.25,
    )

    payload = proposer_output_to_fpip_v2(proposer_output)

    assert payload["valid"] is True
    assert payload["routing_signal"]["recommend_guided_evolution"] is True
    assert payload["routing_signal"]["reason"] == "unvalidated_skeleton_confidence"


def test_verified_candidate_mse_can_disable_guided_routing_without_skeleton_confidence():
    proposer_output = _minimal_proposer_output(
        {
            "entropy": None,
            "margin": None,
            "raw_entropy": 0.8,
            "raw_margin": 0.0,
            "raw_confident": False,
            "confident": False,
            "skeleton_confidence_reliable": False,
            "confidence_source": "disabled_unvalidated_skeleton_head",
        },
        best_relative_mse=1e-10,
    )

    payload = proposer_output_to_fpip_v2(proposer_output)

    assert payload["routing_signal"]["recommend_guided_evolution"] is False
    assert payload["routing_signal"]["reason"] == "candidate_verified_by_mse"
    assert payload["routing_signal"]["confidence_source"] == "grammar_candidate_mse"


def test_validated_skeleton_metrics_can_enable_confidence():
    reliability = up._skeleton_confidence_reliability(
        {
            "skeleton_coverage": 0.9,
            "skeleton_top1_acc": 0.75,
            "skeleton_top5_acc": 0.95,
        }
    )
    uncertainty = up._uncertainty_from_logits([10.0, 0.0, -3.0], reliability)
    proposer_output = _minimal_proposer_output(uncertainty, best_relative_mse=0.25)

    payload = proposer_output_to_fpip_v2(proposer_output)

    assert reliability["reliable"] is True
    assert uncertainty["confident"] is True
    assert payload["routing_signal"]["recommend_guided_evolution"] is False
    assert payload["routing_signal"]["reason"] == "validated_skeleton_confidence"
