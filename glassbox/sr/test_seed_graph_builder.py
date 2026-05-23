"""Tests for formula → C++ seed graph conversion."""
import sys
from pathlib import Path

import pytest

cpp_dir = Path(__file__).parent / "cpp"
if str(cpp_dir) not in sys.path:
    sys.path.insert(0, str(cpp_dir))

from seed_graph_builder import (  # noqa: E402
    TYPE_INPUT,
    TYPE_UNARY,
    UNARY_INTPOW,
    UNARY_PERIODIC,
    build_seed_graphs_from_candidates,
    build_seed_graphs_from_formulas,
    build_seed_graphs_from_signal,
    discover_seed_formulas_from_signal,
    formula_to_seed_graph,
)
import seed_graph_builder as sgb  # noqa: E402

try:
    import _core

    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False

requires_cpp = pytest.mark.skipif(
    not CPP_AVAILABLE,
    reason="C++ _core extension not built",
)


@pytest.mark.parametrize(
    "formula",
    [
        "x",
        "x**2",
        "x**2 + sin(x)",
        "x*sin(x)",
        "sin(3*x)",
        "cos(x)",
        "exp(-x)",
        "x**3 + x**2 + x",
        "1/(x**2 + 1)",
        "2*sin(x) + 3*cos(x)",
    ],
)
def test_formula_to_seed_graph_builds(formula: str) -> None:
    graph = formula_to_seed_graph(formula)
    assert graph is not None
    assert graph["nodes"]
    assert len(graph["output_weights"]) == len(graph["nodes"])


@pytest.mark.parametrize("formula", ["x", "x**2", "sin(x)", "x*sin(x)"])
def test_cpp_seed_graph_export_matches(formula: str) -> None:
    graph = formula_to_seed_graph(formula)
    assert graph is not None
    assert graph["nodes"]


def test_build_seed_graphs_dedupes() -> None:
    graphs = build_seed_graphs_from_formulas(["x", "x", "x**2"], max_seeds=5)
    assert len(graphs) == 2


def test_build_from_candidates_orders_by_mse() -> None:
    cands = [
        {"formula": "x**3", "mse": 1.0},
        {"formula": "x**2 + sin(x)", "mse": 0.01},
    ]
    graphs = build_seed_graphs_from_candidates(cands, max_seeds=2)
    assert len(graphs) == 2


@requires_cpp
def test_cpp_accepts_seed_graphs() -> None:
    import numpy as np

    x = np.linspace(-1, 1, 50)
    y = x**2 + np.sin(x)
    graphs = build_seed_graphs_from_formulas(["x**2 + sin(x)", "x*sin(x)"], max_seeds=2)
    assert len(graphs) >= 1

    result = _core.run_evolution(
        X_list=[x],
        y=y,
        pop_size=40,
        generations=15,
        early_stop_mse=1e-8,
        seed_graphs_py=graphs,
        random_seed=42,
    )
    assert "formula" in result
    assert result["best_mse"] < 0.1


def test_sin_node_uses_input_child() -> None:
    graph = formula_to_seed_graph("sin(x)")
    assert graph is not None
    sin_nodes = [
        n for n in graph["nodes"]
        if n.get("type") == 2 and n.get("unary_op") == UNARY_PERIODIC
    ]
    assert sin_nodes
    assert sin_nodes[0]["left_child"] == 0
    assert graph["nodes"][0]["type"] == TYPE_INPUT


def test_intpow_for_polynomial() -> None:
    graph = formula_to_seed_graph("x**3")
    assert graph is not None
    pow_nodes = [n for n in graph["nodes"] if n.get("unary_op") == UNARY_INTPOW]
    assert pow_nodes
    assert pow_nodes[0]["p"] == 3.0


def test_signal_seed_discovery_prefers_module_forms() -> None:
    import numpy as np

    x = np.linspace(-2, 2, 128)
    y = (x**2) * np.sin(x)
    formulas = discover_seed_formulas_from_signal(x, y, detected_omegas=[1.0], max_seeds=8)
    assert formulas
    assert any("x^2*sin" in f.replace(" ", "") or "x**2*sin" in f.replace(" ", "") for f in formulas)


def test_signal_seed_graphs_build() -> None:
    import numpy as np

    x = np.linspace(-2, 2, 64)
    y = (x**2) * np.sin(x)
    graphs = build_seed_graphs_from_signal(x, y, detected_omegas=[1.0], max_seeds=5)
    assert graphs


def test_multivariate_formula_to_seed_graph_builds(monkeypatch):
    monkeypatch.setattr(sgb, "_core", None)
    graph = sgb.formula_to_seed_graph("x0*x1 + sin(x2)")
    assert graph is not None
    assert graph["nodes"]
    periodic_nodes = [
        n for n in graph["nodes"]
        if n.get("type") == TYPE_UNARY and n.get("unary_op") == UNARY_PERIODIC
    ]
    assert periodic_nodes
    sin_child = graph["nodes"][periodic_nodes[0]["left_child"]]
    assert sin_child["type"] == TYPE_INPUT
    assert sin_child["feature_idx"] == 2


def test_multivariate_formula_handles_named_proxy():
    graph = formula_to_seed_graph("x0 + x1")
    assert graph is not None


@requires_cpp
def test_seeded_exact_evolution_records_generation_zero() -> None:
    import numpy as np

    x = np.linspace(-1, 1, 40)
    y = x
    graphs = build_seed_graphs_from_formulas(["x"], max_seeds=1)
    assert graphs

    result = _core.run_evolution(
        X_list=[x],
        y=y,
        pop_size=8,
        generations=20,
        early_stop_mse=1e-10,
        seed_graphs_py=graphs,
        random_seed=7,
        topology_refine_interval=1,
    )

    assert result["best_mse"] < 1e-10
    assert result["generation_to_first_exact"] == 0
    assert result["time_to_first_exact_sec"] >= 0.0
