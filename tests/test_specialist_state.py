import numpy as np

from glassbox.sr.specialist_state import (
    build_specialist_segment_slices,
    compute_specialist_state,
    propose_specialist_compositions,
)
from glassbox.sr.cpp.seed_graph_builder import build_seed_graphs_from_candidates
import glassbox.sr.cpp.seed_graph_builder as seed_graph_builder


def _eval_formula(formula, X):
    context = {
        "np": np,
        "sin": np.sin,
        "cos": np.cos,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "abs": np.abs,
    }
    X = np.asarray(X, dtype=np.float64)
    for i in range(X.shape[1]):
        context[f"x{i}"] = X[:, i]
    if X.shape[1] == 1:
        context["x"] = X[:, 0]
    expr = str(formula).replace("^", "**")
    return np.asarray(eval(expr, {"__builtins__": None}, context), dtype=np.float64)


def test_build_specialist_segment_slices_univariate_axis():
    x = np.linspace(-2.0, 2.0, 40)
    X = x.reshape(-1, 1)

    built = build_specialist_segment_slices(X, max_segments=4, min_segment_size=8)
    assert built is not None
    axis, segments = built
    assert axis == "x0"
    assert len(segments) >= 2
    assert sum(seg.n_samples for seg in segments) == X.shape[0]


def test_compute_specialist_state_returns_candidates_and_pairs():
    x = np.linspace(-3.0, 3.0, 160)
    X = np.column_stack([x, np.sin(x), np.cos(x)])
    y = np.where(x < 0.0, x * x, np.sin(2.0 * x))
    candidates = [
        {"formula": "x0^2", "validation_r2": 0.2, "validation_mse": 1.0, "source": "poly"},
        {"formula": "sin(2*x0)", "validation_r2": 0.3, "validation_mse": 0.9, "source": "periodic"},
        {"formula": "x0*sin(x0)", "validation_r2": 0.4, "validation_mse": 0.8, "source": "product"},
    ]

    state = compute_specialist_state(
        candidates,
        X,
        y,
        evaluate_formula=_eval_formula,
        complexity_fn=lambda formula: len(str(formula)),
        family_signature_fn=lambda formula: "periodic" if "sin" in str(formula) else "poly",
        max_candidates=6,
        max_pairs=5,
    )

    assert state is not None
    assert state.enabled is True
    assert state.candidate_count == 3
    assert state.segment_count >= 2
    assert len(state.top_pairs) >= 1
    payload = state.to_dict()
    assert payload["enabled"] is True
    assert payload["candidate_count"] == 3
    assert payload["top_candidates"]
    assert payload["top_pairs"]


def test_propose_specialist_compositions_emits_add_and_mul_forms():
    x = np.linspace(-3.0, 3.0, 160)
    X = np.column_stack([x, np.sin(x), np.cos(x)])
    y = np.where(x < 0.0, x * x, np.sin(2.0 * x))
    candidates = [
        {"formula": "x0^2", "validation_r2": 0.2, "validation_mse": 1.0, "source": "poly"},
        {"formula": "sin(2*x0)", "validation_r2": 0.3, "validation_mse": 0.9, "source": "periodic"},
    ]

    state = compute_specialist_state(
        candidates,
        X,
        y,
        evaluate_formula=_eval_formula,
        complexity_fn=lambda formula: len(str(formula)),
        family_signature_fn=lambda formula: "periodic" if "sin" in str(formula) else "poly",
        max_candidates=4,
        max_pairs=3,
    )
    proposals = propose_specialist_compositions(state, max_pairs=3, min_complementarity=0.0)

    operators = {proposal.operator for proposal in proposals}
    assert "add" in operators
    assert "mul" in operators


def test_build_seed_graphs_does_not_let_composed_seeds_dominate():
    candidate_formulas = [
        {"formula": f"x0 * {i}", "mse": 0.01 * i, "source": "specialist_composition", "from_specialist_composition": True}
        for i in range(1, 6)
    ] + [
        {"formula": f"x0 + {i}", "mse": 0.5 + 0.1 * i, "source": "candidate_screening"}
        for i in range(1, 6)
    ]
    seeds = build_seed_graphs_from_candidates(candidate_formulas, max_seeds=5)
    assert len(seeds) <= 5


def test_build_seed_graphs_enforces_composed_seed_cap(monkeypatch):
    captured = {}

    def fake_build(formulas, max_seeds=10):
        captured["formulas"] = list(formulas)
        return [{"formula": formula} for formula in formulas[:max_seeds]]

    monkeypatch.setattr(seed_graph_builder, "build_seed_graphs_from_formulas", fake_build)
    candidate_formulas = [
        {"formula": f"comp_{i}", "mse": float(i), "from_specialist_composition": True}
        for i in range(10)
    ] + [
        {"formula": f"std_{i}", "mse": 100.0 + i, "source": "candidate_screening"}
        for i in range(2)
    ]

    build_seed_graphs_from_candidates(candidate_formulas, max_seeds=5)

    formulas = captured["formulas"]
    assert sum(formula.startswith("comp_") for formula in formulas) <= 2
    assert sum(formula.startswith("std_") for formula in formulas) == 2


def test_build_hot_spot_segments_and_compute_specialist_state_phase5():
    from glassbox.sr.specialist_state import build_hot_spot_segments
    x = np.linspace(-3.0, 3.0, 100)
    X = x.reshape(-1, 1)

    # Let's create a residual with a sharp concentrated error spike at the center
    best_residual = np.zeros_like(x)
    best_residual[45:55] = 10.0

    hs_segs = build_hot_spot_segments(X, best_residual, max_segments=6, min_segment_size=8)
    assert len(hs_segs) >= 1
    has_concentrated = any(8 <= seg.n_samples <= 10 for seg in hs_segs)
    assert has_concentrated

    # Test compute_specialist_state populates fields
    y = best_residual
    candidates = [
        {"formula": "0*x0", "validation_r2": 0.0, "validation_mse": np.mean(y**2), "source": "candidate_screening"},
        {"formula": "1+0*x0", "validation_r2": 0.0, "validation_mse": np.mean((y-1)**2), "source": "candidate_screening"}
    ]
    state = compute_specialist_state(
        candidates,
        X,
        y,
        evaluate_formula=_eval_formula,
        complexity_fn=lambda formula: 1,
        family_signature_fn=lambda formula: "constant",
        max_candidates=2,
        max_pairs=1
    )
    assert state is not None
    assert len(state.hot_spot_segments) >= 1
    for cand in state.candidates:
        assert len(cand.hot_spot_segment_scores) == len(state.hot_spot_segments)


def test_hot_spot_segments_use_best_metric_candidate_as_base():
    x = np.linspace(-3.0, 3.0, 100)
    X = x.reshape(-1, 1)
    y = x
    candidates = [
        {"formula": "0*x0", "validation_r2": -1.0, "validation_mse": 100.0, "source": "bad"},
        {"formula": "x0", "validation_r2": 1.0, "validation_mse": 0.0, "source": "good"},
    ]

    state = compute_specialist_state(
        candidates,
        X,
        y,
        evaluate_formula=_eval_formula,
        complexity_fn=lambda formula: 1,
        family_signature_fn=lambda formula: "linear",
        max_candidates=2,
        max_pairs=1,
    )

    assert state is not None
    assert state.hot_spot_base_formula == "x0"


def test_propose_specialist_compositions_expanded_templates_phase6():
    x = np.linspace(-1.0, 1.0, 50)
    X = x.reshape(-1, 1)

    y = np.sin(x + 1.0)

    candidates = [
        {"formula": "sin(x0)", "validation_r2": 0.8, "validation_mse": 0.01, "source": "candidate_screening"},
        {"formula": "x0+1.0", "validation_r2": 0.8, "validation_mse": 0.01, "source": "candidate_screening"}
    ]

    state = compute_specialist_state(
        candidates,
        X,
        y,
        evaluate_formula=_eval_formula,
        complexity_fn=lambda formula: 1,
        family_signature_fn=lambda formula: "sin" if "sin" in str(formula) else "poly",
        max_candidates=2,
        max_pairs=1
    )

    assert state is not None

    proposals = propose_specialist_compositions(
        state,
        X,
        y,
        evaluate_formula=_eval_formula,
        max_pairs=1,
        min_complementarity=0.0
    )

    operators = {proposal.operator for proposal in proposals}
    assert "nested" in operators or "affine" in operators or "add" in operators


def test_propose_specialist_compositions_handles_reversed_nested_parent():
    x = np.linspace(-1.0, 1.0, 80)
    X = x.reshape(-1, 1)
    y = np.sin(x + 1.0)
    candidates = [
        {"formula": "x0+1.0", "validation_r2": 0.4, "validation_mse": 0.2, "source": "inner"},
        {"formula": "sin(x0)", "validation_r2": 0.4, "validation_mse": 0.2, "source": "outer"},
    ]

    state = compute_specialist_state(
        candidates,
        X,
        y,
        evaluate_formula=_eval_formula,
        complexity_fn=lambda formula: 1,
        family_signature_fn=lambda formula: "sin" if "sin" in str(formula) else "poly",
        max_candidates=2,
        max_pairs=1,
    )

    proposals = propose_specialist_compositions(
        state,
        X,
        y,
        evaluate_formula=_eval_formula,
        max_pairs=1,
        min_complementarity=0.0,
    )

    assert any(
        proposal.operator == "nested" and "sin((x0+1.0))" in proposal.formula
        for proposal in proposals
    )

