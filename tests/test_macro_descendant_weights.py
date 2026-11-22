"""§3.230: macro composition also clears consumed descendant output weights."""

import pytest

from glassbox.sr.cpp import CPP_AVAILABLE, get_cpp_core

pytestmark = pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ _core not built")


def _graph(types, lefts, rights, weights):
    nodes = [
        {"type": t, "left_child": l, "right_child": r}
        for t, l, r in zip(types, lefts, rights)
    ]
    return {"nodes": nodes, "output_weights": list(weights)}


def test_unshared_descendant_weight_is_zeroed():
    core = get_cpp_core()
    # 0: x0, 1: x0^2 (child 0), 2: x0^3 (child 0) — all three active.
    graph = _graph(
        types=[0, 2, 2],
        lefts=[-1, 0, 0],
        rights=[-1, -1, -1],
        weights=[1.0, 1.0, 1.0],
    )
    out = core.macro_zero_descendant_weights(graph, 1, 2)
    # x0 is now consumed by both operands and feeds nothing else.
    assert out["zeroed"] == 1
    assert out["output_weights"] == [0.0, 1.0, 1.0]


def test_descendant_feeding_an_outside_root_is_preserved():
    core = get_cpp_core()
    # 0: x0, 1: const, 2: f(const), 3: binary(x0, const), 4: g(const) active.
    graph = _graph(
        types=[0, 1, 2, 3, 2],
        lefts=[-1, -1, 1, 0, 1],
        rights=[-1, -1, -1, 1, -1],
        weights=[1.0, 1.0, 1.0, 0.0, 1.0],
    )
    out = core.macro_zero_descendant_weights(graph, 2, 3)
    # x0 (index 0) is exclusive to the product -> zeroed; the constant still
    # feeds root 4 -> preserved.
    assert out["zeroed"] == 1
    assert out["output_weights"] == [0.0, 1.0, 1.0, 0.0, 1.0]


def test_operands_and_new_root_are_never_touched():
    core = get_cpp_core()
    graph = _graph(
        types=[0, 2, 2],
        lefts=[-1, 0, 0],
        rights=[-1, -1, -1],
        weights=[1.0, 1.0, 1.0],
    )
    # left == right == the only active descendants: nothing else to clear.
    out = core.macro_zero_descendant_weights(graph, 0, 0)
    assert out["zeroed"] == 0
    assert out["output_weights"] == [1.0, 1.0, 1.0]
