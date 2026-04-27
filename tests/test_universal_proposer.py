import numpy as np

from glassbox.sr.universal_proposer import (
    UniversalProposer,
    UniversalProposerConfig,
    propose_from_xy,
    proposer_output_to_fpip_v2,
)


def test_universal_proposer_returns_topk_candidates():
    model = UniversalProposer(UniversalProposerConfig(hidden_dim=32))
    x = np.linspace(-2.0, 2.0, 64, dtype=np.float32)
    y = np.sin(x).astype(np.float32)

    out = propose_from_xy(model, x, y, top_k=4)

    assert "candidate_skeletons" in out
    assert len(out["candidate_skeletons"]) == 4
    assert "operator_priors" in out
    assert len(out["operator_priors"]) > 0
    assert "sequence_uncertainty" in out


def test_proposer_output_maps_to_valid_fpip_v2():
    model = UniversalProposer(UniversalProposerConfig(hidden_dim=32))
    x = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
    y = (x ** 2).astype(np.float32)

    out = propose_from_xy(model, x, y, top_k=3)
    payload = proposer_output_to_fpip_v2(out, fit_diagnostics={"mse": 0.1})

    assert payload["schema_version"] == "fpip.v2"
    assert payload["valid"] is True
    assert len(payload["candidate_skeletons"]) == 3
    assert "routing_signal" in payload
