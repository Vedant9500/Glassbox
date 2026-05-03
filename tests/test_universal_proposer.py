import numpy as np
from pathlib import Path

from glassbox.universal_proposer import (
    UniversalProposer,
    UniversalProposerConfig,
    grammar_decode_topk_skeletons,
    propose_from_xy,
    proposer_output_to_fpip_v2,
)
from scripts.train_universal_proposer import FormulaReplayDataset


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


def test_grammar_decoder_prefers_periodic_candidates():
    x = np.linspace(-2.0, 2.0, 128, dtype=np.float64)
    y = np.sin(x)
    priors = {
        "sin": 0.8,
        "cos": 0.05,
        "periodic": 0.7,
        "identity": 0.2,
        "power": 0.05,
        "exp": 0.01,
        "log": 0.01,
        "rational": 0.01,
    }

    top = grammar_decode_topk_skeletons(priors, x=x, y=y, top_k=5, max_depth=2)
    assert len(top) == 5
    joined = " | ".join([c["formula"] for c in top])
    assert ("sin(" in joined) or ("cos(" in joined)


def test_formula_replay_dataset_loads_npz(tmp_path: Path):
    n = 16
    labels = np.zeros((n, 14), dtype=np.float32)
    labels[:, 1] = 1.0  # sin
    formulas = np.array(["np.sin(x)" for _ in range(n)], dtype=object)
    features = np.zeros((n, 366), dtype=np.float32)
    npz_path = tmp_path / "mini_dataset.npz"
    np.savez_compressed(npz_path, features=features, labels=labels, formulas=formulas)

    ds = FormulaReplayDataset(npz_path, n_points=64)
    points, op_target, skeleton_target = ds[0]

    assert points.shape == (64, 2)
    assert op_target.shape[0] >= 8
    assert int(skeleton_target.item()) == -1
