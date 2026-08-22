import numpy as np

from glassbox.sr.cpp import seed_graph_builder
from glassbox.sr.cpp.seed_graph_builder import build_seed_graphs_from_candidates
from glassbox.sr.specialist_state import (
    SpecialistVault,
    build_specialist_segment_slices,
    compute_specialist_state,
    propose_specialist_compositions,
)


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
        {
            "formula": "x0^2",
            "validation_r2": 0.2,
            "validation_mse": 1.0,
            "source": "poly",
        },
        {
            "formula": "sin(2*x0)",
            "validation_r2": 0.3,
            "validation_mse": 0.9,
            "source": "periodic",
        },
        {
            "formula": "x0*sin(x0)",
            "validation_r2": 0.4,
            "validation_mse": 0.8,
            "source": "product",
        },
    ]

    state = compute_specialist_state(
        candidates,
        X,
        y,
        evaluate_formula=_eval_formula,
        complexity_fn=lambda formula: len(str(formula)),
        family_signature_fn=lambda formula: (
            "periodic" if "sin" in str(formula) else "poly"
        ),
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
        {
            "formula": "x0^2",
            "validation_r2": 0.2,
            "validation_mse": 1.0,
            "source": "poly",
        },
        {
            "formula": "sin(2*x0)",
            "validation_r2": 0.3,
            "validation_mse": 0.9,
            "source": "periodic",
        },
    ]

    state = compute_specialist_state(
        candidates,
        X,
        y,
        evaluate_formula=_eval_formula,
        complexity_fn=lambda formula: len(str(formula)),
        family_signature_fn=lambda formula: (
            "periodic" if "sin" in str(formula) else "poly"
        ),
        max_candidates=4,
        max_pairs=3,
    )
    proposals = propose_specialist_compositions(
        state, max_pairs=3, min_complementarity=0.0
    )

    operators = {proposal.operator for proposal in proposals}
    assert "add" in operators
    assert "mul" in operators


def test_build_seed_graphs_does_not_let_composed_seeds_dominate():
    candidate_formulas = [
        {
            "formula": f"x0 * {i}",
            "mse": 0.01 * i,
            "source": "specialist_composition",
            "from_specialist_composition": True,
        }
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

    monkeypatch.setattr(
        seed_graph_builder, "build_seed_graphs_from_formulas", fake_build
    )
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

    hs_segs = build_hot_spot_segments(
        X, best_residual, max_segments=6, min_segment_size=8
    )
    assert len(hs_segs) >= 1
    has_concentrated = any(8 <= seg.n_samples <= 10 for seg in hs_segs)
    assert has_concentrated

    # Test compute_specialist_state populates fields
    y = best_residual
    candidates = [
        {
            "formula": "0*x0",
            "validation_r2": 0.0,
            "validation_mse": np.mean(y**2),
            "source": "candidate_screening",
        },
        {
            "formula": "1+0*x0",
            "validation_r2": 0.0,
            "validation_mse": np.mean((y - 1) ** 2),
            "source": "candidate_screening",
        },
    ]
    state = compute_specialist_state(
        candidates,
        X,
        y,
        evaluate_formula=_eval_formula,
        complexity_fn=lambda formula: 1,
        family_signature_fn=lambda formula: "constant",
        max_candidates=2,
        max_pairs=1,
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
        {
            "formula": "0*x0",
            "validation_r2": -1.0,
            "validation_mse": 100.0,
            "source": "bad",
        },
        {
            "formula": "x0",
            "validation_r2": 1.0,
            "validation_mse": 0.0,
            "source": "good",
        },
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
        {
            "formula": "sin(x0)",
            "validation_r2": 0.8,
            "validation_mse": 0.01,
            "source": "candidate_screening",
        },
        {
            "formula": "x0+1.0",
            "validation_r2": 0.8,
            "validation_mse": 0.01,
            "source": "candidate_screening",
        },
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

    assert state is not None

    proposals = propose_specialist_compositions(
        state,
        X,
        y,
        evaluate_formula=_eval_formula,
        max_pairs=1,
        min_complementarity=0.0,
    )

    operators = {proposal.operator for proposal in proposals}
    assert "nested" in operators or "affine" in operators or "add" in operators


def test_propose_specialist_compositions_handles_reversed_nested_parent():
    x = np.linspace(-1.0, 1.0, 80)
    X = x.reshape(-1, 1)
    y = np.sin(x + 1.0)
    candidates = [
        {
            "formula": "x0+1.0",
            "validation_r2": 0.4,
            "validation_mse": 0.2,
            "source": "inner",
        },
        {
            "formula": "sin(x0)",
            "validation_r2": 0.4,
            "validation_mse": 0.2,
            "source": "outer",
        },
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


def test_h22_residual_relevance_computed_and_used_in_vault_rank():
    """H-22: residual_relevance is set on admission and breaks ties in vault order."""
    rng = np.random.RandomState(7)
    X = rng.uniform(-2.0, 2.0, size=(120, 1))
    x0 = X[:, 0]
    # Target has structure shared with sin but residual is linear
    y = np.sin(x0) + 0.4 * x0

    vault = SpecialistVault(max_entries=3, corr_threshold=0.999)
    candidates = [
        # Structural partial: high residual relevance
        {"formula": "sin(x0)", "source": "struct"},
        # Weak constant-like (mean of y is near 0 on symmetric domain → low relevance)
        {"formula": "0.01*x0", "source": "weak"},
        # Complementary linear piece
        {"formula": "0.4*x0", "source": "lin"},
    ]
    added = vault.add_candidates(
        candidates,
        X,
        y,
        evaluate_formula=_eval_formula,
        complexity_fn=lambda formula: len(str(formula)),
        family_signature_fn=lambda formula: str(formula)[:3],
        run_index=0,
        current_best_formula="0",
        max_new=3,
    )
    assert added >= 1
    # Every retained entry must have residual_relevance populated
    for entry in vault.entries:
        assert entry.residual_relevance is not None
        assert 0.0 <= float(entry.residual_relevance) <= 1.0 + 1e-9

    # Rescore re-ranks; order key is (mse, -relevance, complexity)
    vault.rescore_against_target(X, y, evaluate_formula=_eval_formula)
    keys = [vault._rank_entry_key(e) for e in vault.entries]
    assert keys == sorted(keys)

    # Explicit tie-break: equal MSE → higher residual_relevance ranks first
    from glassbox.sr.specialist_state import SpecialistVaultEntry

    low = SpecialistVaultEntry(
        formula="a",
        source="t",
        validation_r2=0.5,
        validation_mse=0.1,
        complexity=5,
        family_signature="a",
        segment_scores=[],
        residual_vector=np.zeros(3),
        prediction_vector=np.zeros(3),
        residual_relevance=0.1,
    )
    high = SpecialistVaultEntry(
        formula="b",
        source="t",
        validation_r2=0.5,
        validation_mse=0.1,
        complexity=5,
        family_signature="b",
        segment_scores=[],
        residual_vector=np.zeros(3),
        prediction_vector=np.zeros(3),
        residual_relevance=0.9,
    )
    assert vault._rank_entry_key(high) < vault._rank_entry_key(low)


def test_specialist_vault_dedupes_by_prediction_correlation_and_caps_entries():
    # 2D additive target: partial specialists x0/x1 both clear S8-1 holdout gate
    # (1D partials often fail tail-holdout R² on sin/poly mixes).
    rng = np.random.RandomState(0)
    X = rng.uniform(-1.0, 1.0, size=(100, 2))
    y = X[:, 0] + X[:, 1]
    vault = SpecialistVault(max_entries=2, corr_threshold=0.98)
    candidates = [
        {"formula": "x0", "validation_r2": 0.5, "validation_mse": 0.5, "source": "a"},
        {
            "formula": "x0+0",
            "validation_r2": 0.5,
            "validation_mse": 0.5,
            "source": "dup",
        },
        {"formula": "x1", "validation_r2": 0.5, "validation_mse": 0.5, "source": "b"},
        {
            "formula": "0.1*x0",
            "validation_r2": 0.1,
            "validation_mse": 0.9,
            "source": "c",
        },
    ]

    added = vault.add_candidates(
        candidates,
        X,
        y,
        evaluate_formula=_eval_formula,
        complexity_fn=lambda formula: len(str(formula)),
        family_signature_fn=lambda formula: str(formula)[:2],
        run_index=0,
        current_best_formula="0",
        max_new=4,
    )

    assert added >= 2
    assert len(vault.entries) == 2
    formulas = {entry.formula for entry in vault.entries}
    assert not {"x0", "x0+0"}.issubset(formulas)
    assert vault.rejected_duplicate_count >= 1


def test_specialist_vault_proposes_capped_composition_candidates():
    # Complementary 2D specialists that pass S8-1 and compose under S8-2 caps.
    rng = np.random.RandomState(1)
    X = rng.uniform(-1.0, 1.0, size=(100, 2))
    y = X[:, 0] + X[:, 1]
    vault = SpecialistVault(max_entries=8)
    vault.add_candidates(
        [
            {
                "formula": "x0",
                "validation_r2": 0.5,
                "validation_mse": 0.5,
                "source": "outer",
            },
            {
                "formula": "x1",
                "validation_r2": 0.5,
                "validation_mse": 0.5,
                "source": "inner",
            },
        ],
        X,
        y,
        evaluate_formula=_eval_formula,
        complexity_fn=lambda formula: 3,
        family_signature_fn=lambda formula: str(formula)[:2],
        run_index=0,
        max_new=2,
    )
    assert len(vault.entries) >= 1

    proposals = vault.propose_compositions(
        X,
        y,
        evaluate_formula=_eval_formula,
        complexity_fn=lambda formula: 3,
        family_signature_fn=lambda formula: str(formula)[:2],
        current_best_candidate={
            "formula": "x0+x1",
            "validation_mse": 0.0,
            "validation_r2": 1.0,
            "source": "current",
        },
    )

    # S8-2: hard cap of 4 (was 6).
    assert len(proposals) <= 4
    assert proposals
    assert all(candidate["from_specialist_vault"] for candidate in proposals)
    assert all(candidate["from_specialist_composition"] for candidate in proposals)


def test_m01_permuted_index_holdout():
    """M-01: SpecialistVault holdout uses deterministic permuted indices rather than deterministic tail.

    When y is ordered (e.g. sorted domain [-2, 2]), the tail slice y[-n_val:] has low variance (~0.09),
    so tail noise causes tail holdout R2 to plummet below 0.25 (rejecting the candidate).
    Permuted holdout samples uniformly across the domain (variance ~1.25), keeping holdout R2 high.
    """
    n = 40
    X = np.linspace(-2, 2, n).reshape(-1, 1)
    y = X[:, 0].copy()

    vault = SpecialistVault(max_entries=5)

    def _eval_tail_noisy(formula, X_in):
        arr = X_in[:, 0].copy()
        if "tail_noisy" in formula:
            arr[-5:] += 0.8
        return arr

    candidates = [{"formula": "tail_noisy_x0"}]
    added = vault.add_candidates(
        candidates,
        X,
        y,
        evaluate_formula=_eval_tail_noisy,
        complexity_fn=lambda f: 5,
        family_signature_fn=lambda f: "identity",
        run_index=0,
    )
    assert added == 1


def test_compute_specialist_state_ranks_before_slice():
    x = np.linspace(-3.0, 3.0, 160)
    X = np.column_stack([x, np.sin(x), np.cos(x)])
    y = np.where(x < 0.0, x * x, np.sin(2.0 * x))
    # Best formula placed LAST in input order; weaker ones first.
    candidates = [
        {
            "formula": "sin(x0)",
            "validation_r2": 0.2,
            "validation_mse": 5.0,
            "source": "weak",
        },
        {
            "formula": "cos(x0)",
            "validation_r2": 0.1,
            "validation_mse": 6.0,
            "source": "weaker",
        },
        {
            "formula": "x0^2",
            "validation_r2": 0.9,
            "validation_mse": 0.1,
            "source": "best",
        },
    ]

    state = compute_specialist_state(
        candidates,
        X,
        y,
        evaluate_formula=_eval_formula,
        complexity_fn=lambda formula: len(str(formula)),
        family_signature_fn=lambda formula: (
            "periodic" if "sin" in str(formula) else "poly"
        ),
        max_candidates=2,
        max_pairs=3,
    )

    assert state is not None
    formula_best = state.hot_spot_base_formula
    assert formula_best == "x0^2"
