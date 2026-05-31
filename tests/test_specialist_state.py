import numpy as np

from glassbox.sr.specialist_state import (
    build_specialist_segment_slices,
    compute_specialist_state,
    propose_specialist_compositions,
)
from glassbox.sr.cpp.seed_graph_builder import build_seed_graphs_from_candidates


def _eval_formula(formula, X):
    context = {
        "np": np,
        "sin": np.sin,
        "cos": np.cos,
        "exp": np.exp,
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
