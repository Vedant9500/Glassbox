"""§3.297: collinear seed output columns are merged before insertion."""

import numpy as np
import pytest

from glassbox.sr.cpp import CPP_AVAILABLE, get_cpp_core

pytestmark = pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ _core not built")

INPUT, CONSTANT, UNARY, BINARY = 0, 1, 2, 3
ADD, MUL = 1.0, 2.0


def _node(**kw):
    base = {
        "type": CONSTANT,
        "feature_idx": 0,
        "value": 0.0,
        "unary_op": 0,
        "binary_op": 0,
        "p": 1.0,
        "omega": 1.0,
        "phi": 0.0,
        "amplitude": 1.0,
        "beta": ADD,
        "gamma": 1.0,
        "tau": 1.0,
        "left_child": -1,
        "right_child": -1,
    }
    base.update(kw)
    return base


def _predict(graph, x):
    values = np.zeros_like(x, dtype=np.float64)
    for i, w in enumerate(graph["output_weights"]):
        values = values + w * _node_column(graph, i, x)
    return values + graph.get("output_bias", 0.0)


def _soft_power(x, p):
    abs_pow = (np.abs(x) + 1e-10) ** p
    nearest = round(p)
    if abs(p - nearest) < 1e-9 and int(nearest) % 2 == 0:
        return abs_pow
    return np.sign(x) * abs_pow


def _arithmetic_weights(node, temperature=5.0):
    beta, gamma = node["beta"], node["gamma"]
    dists = [
        (beta - 1.0) ** 2 + (gamma - 1.0) ** 2,
        (beta - 2.0) ** 2 + (gamma - 1.0) ** 2,
        (beta - 2.0) ** 2 + (gamma + 1.0) ** 2,
        (beta - 1.0) ** 2 + (gamma + 1.0) ** 2,
    ]
    logits = [-d * temperature for d in dists]
    top = max(logits)
    exps = [np.exp(v - top) for v in logits]
    total = sum(exps)
    return [v / total for v in exps]


def _node_column(graph, idx, x):
    node = graph["nodes"][idx]
    if node["type"] == INPUT:
        return x
    if node["type"] == CONSTANT:
        return np.full_like(x, node["value"], dtype=np.float64)
    left = _node_column(graph, node["left_child"], x)
    if node["type"] == UNARY:
        return _soft_power(left, node["p"])
    right = _node_column(graph, node["right_child"], x)
    w = _arithmetic_weights(node)
    blended = (
        w[0] * (left + right)
        + w[1] * (left * right)
        + w[2] * (left / np.sqrt(1.0 + right**2))
        + w[3] * (left - right)
    )
    return np.clip(blended, -1e6, 1e6)


def test_duplicate_roots_are_merged_without_changing_prediction():
    core = get_cpp_core()
    x = np.linspace(-2.0, 2.0, 64)
    # x and x (two separate output roots with different weights).
    graph = {
        "nodes": [
            _node(type=INPUT),
            _node(type=INPUT),
        ],
        "output_weights": [1.5, 2.5],
        "output_bias": 0.0,
    }
    before = _predict(graph, x)
    out = core.dedupe_collinear_seed_weights(graph, [x])
    after_graph = dict(graph)
    after_graph["output_weights"] = out["output_weights"]
    after_graph["output_bias"] = out["output_bias"]
    assert out["merged"] == 1
    np.testing.assert_allclose(_predict(after_graph, x), before, atol=1e-12)


def test_affine_root_is_merged_into_bias():
    core = get_cpp_core()
    x = np.linspace(-2.0, 2.0, 64)
    # x^2 and (x^2 + 1): centered columns are identical.
    graph = {
        "nodes": [
            _node(type=INPUT),
            _node(type=UNARY, unary_op=1, p=2.0, left_child=0),
            _node(type=CONSTANT, value=1.0),
            _node(type=BINARY, beta=ADD, left_child=1, right_child=2),
        ],
        "output_weights": [0.0, 1.0, 0.0, 1.0],
        "output_bias": 0.0,
    }
    before = _predict(graph, x)
    out = core.dedupe_collinear_seed_weights(graph, [x])
    after_graph = dict(graph)
    after_graph["output_weights"] = out["output_weights"]
    after_graph["output_bias"] = out["output_bias"]
    assert out["merged"] == 1
    np.testing.assert_allclose(_predict(after_graph, x), before, atol=1e-12)


def test_independent_columns_are_kept():
    core = get_cpp_core()
    x = np.linspace(-2.0, 2.0, 64)
    graph = {
        "nodes": [
            _node(type=INPUT),
            _node(type=UNARY, unary_op=1, p=2.0, left_child=0),
        ],
        "output_weights": [1.0, 1.0],
        "output_bias": 0.0,
    }
    out = core.dedupe_collinear_seed_weights(graph, [x])
    assert out["merged"] == 0
    assert out["output_weights"] == [1.0, 1.0]


def test_near_collinear_but_distinct_columns_survive():
    core = get_cpp_core()
    x = np.linspace(-1.0, 1.0, 64)
    # x and sin(x) correlate ~0.97 here; the strict 1e-8 residual gate must
    # not treat them as duplicates.
    graph = {
        "nodes": [
            _node(type=INPUT),
            _node(type=UNARY, unary_op=0, omega=1.0, left_child=0),
        ],
        "output_weights": [1.0, 1.0],
        "output_bias": 0.0,
    }
    out = core.dedupe_collinear_seed_weights(graph, [x])
    assert out["merged"] == 0
