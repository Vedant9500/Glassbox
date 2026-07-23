import numpy as np
from pathlib import Path

from glassbox.universal_proposer import (
    UniversalProposer,
    UniversalProposerConfig,
    grammar_decode_multivariate_skeletons,
    grammar_decode_topk_skeletons,
    propose_from_xy,
    proposer_output_to_fpip_v2,
)
from scripts.train_universal_proposer import FormulaReplayDataset
from glassbox.curve_classifier.generate_curve_data import FEATURE_DIM


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
    assert "search_plan" in out
    assert out["search_plan"]["strategy"] in {"refine_seed", "focused", "balanced", "exploratory"}
    assert out["search_plan"]["population_multiplier"] > 0


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
    assert "search_plan" in payload
    assert "generation_multiplier" in payload["search_plan"]


def test_universal_proposer_accepts_multivariate_input():
    model = UniversalProposer(UniversalProposerConfig(hidden_dim=32, max_input_vars=3))
    x0 = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
    x1 = np.linspace(0.5, 2.0, 64, dtype=np.float32)
    X = np.stack([x0, x1], axis=1)
    y = (x0 * x1).astype(np.float32)

    out = propose_from_xy(model, X, y, top_k=4)

    assert out["supports_multivariate_formulas"] is True
    assert out["input_variables"] == ["x0", "x1"]
    assert len(out["candidate_skeletons"]) == 4
    joined = " | ".join(c["formula"] for c in out["candidate_skeletons"])
    assert "x0" in joined and "x1" in joined


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


def test_univariate_grammar_includes_product_and_rational_templates():
    from glassbox.universal_proposer.universal_proposer import (
        _build_univariate_grammar_candidates,
    )

    cands = _build_univariate_grammar_candidates(max_depth=2)
    joined = " | ".join(cands)
    assert "x**2*sin(x)" in joined
    assert "x**3/(1+x**4)" in joined
    assert "x/(1+x**2)" in joined


def test_multivariate_grammar_decoder_returns_cross_terms():
    x0 = np.linspace(-2.0, 2.0, 64, dtype=np.float64)
    x1 = np.linspace(1.0, 3.0, 64, dtype=np.float64)
    X = np.stack([x0, x1], axis=1)
    y = x0 * x1
    priors = {
        "sin": 0.05,
        "cos": 0.05,
        "periodic": 0.05,
        "identity": 0.2,
        "power": 0.1,
        "exp": 0.01,
        "log": 0.01,
        "rational": 0.1,
    }

    top = grammar_decode_multivariate_skeletons(priors, X, y, top_k=5, max_rank=2)
    assert len(top) == 5
    joined = " | ".join([c["formula"] for c in top])
    assert "x0*x1" in joined or "x0+x1" in joined


def test_multivariate_grammar_decoder_preserves_feature_names_after_first_pair():
    x0 = np.linspace(-1.0, 1.0, 80, dtype=np.float64)
    x1 = np.linspace(0.2, 1.4, 80, dtype=np.float64)
    x2 = np.linspace(1.0, 2.0, 80, dtype=np.float64)
    X = np.stack([x0, x1, x2], axis=1)
    y = x1 * x2
    priors = {
        "identity": 0.8,
        "power": 0.2,
        "rational": 0.1,
        "sin": 0.05,
        "cos": 0.05,
        "periodic": 0.05,
    }

    top = grammar_decode_multivariate_skeletons(priors, X, y, top_k=8, max_rank=3)
    joined = " | ".join([c["formula"] for c in top])

    assert "x1*x2" in joined


def test_formula_replay_dataset_loads_npz(tmp_path: Path):
    n = 16
    labels = np.zeros((n, 14), dtype=np.float32)
    labels[:, 1] = 1.0  # sin
    formulas = np.array(["np.sin(x)" for _ in range(n)], dtype=object)
    features = np.zeros((n, FEATURE_DIM), dtype=np.float32)
    npz_path = tmp_path / "mini_dataset.npz"
    np.savez_compressed(npz_path, features=features, labels=labels, formulas=formulas)

    ds = FormulaReplayDataset(npz_path, n_points=64)
    points, op_target, skeleton_target = ds[0]

    assert points.shape == (FEATURE_DIM,)
    assert op_target.shape[0] >= 8
    assert int(skeleton_target.item()) >= 0


def test_formula_replay_dataset_matches_multivariate_skeleton(tmp_path: Path):
    n = 12
    labels = np.zeros((n, 14), dtype=np.float32)
    labels[:, 7] = 1.0  # multiplication
    formulas = np.array(["x0*x1" for _ in range(n)], dtype=object)
    features = np.zeros((n, FEATURE_DIM), dtype=np.float32)
    npz_path = tmp_path / "multi_dataset.npz"
    np.savez_compressed(npz_path, features=features, labels=labels, formulas=formulas)

    ds = FormulaReplayDataset(npz_path, n_points=64)
    _, _, skeleton_target = ds[0]

    assert int(skeleton_target.item()) >= 0
