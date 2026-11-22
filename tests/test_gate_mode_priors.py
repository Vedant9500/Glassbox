"""§3.298-§3.301: gate-mode priors and product/rational island semantics."""

import numpy as np
import pytest

from glassbox.sr.cpp import CPP_AVAILABLE, get_cpp_core
from scripts.classifier_fast_path import (
    PRODUCT_ISLAND_GATE_PRIORS,
    PRODUCT_ISLAND_GATE_PRIORS_MILD,
    RATIONAL_ISLAND_GATE_PRIORS,
)

pytestmark = pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ _core not built")


def test_gate_prior_is_deterministic_for_single_mode():
    core = get_cpp_core()
    assert set(core.sample_gate_modes([0, 1, 0, 0], 200, seed=7)) == {1}
    assert set(core.sample_gate_modes([1, 0, 0, 0], 200, seed=7)) == {0}
    assert set(core.sample_gate_modes([0, 0, 1, 0], 200, seed=7)) == {2}
    assert set(core.sample_gate_modes([0, 0, 0, 1], 200, seed=7)) == {3}


def test_empty_gate_prior_is_uniform():
    core = get_cpp_core()
    modes = set(core.sample_gate_modes([], 400, seed=3))
    assert modes == {0, 1, 2, 3}


def test_engine_accepts_gate_priors_and_island_variants():
    core = get_cpp_core()
    x = np.linspace(-1.0, 1.0, 32)
    y = x * x
    result = core.run_evolution(
        [x],
        y,
        pop_size=12,
        generations=2,
        num_islands=2,
        gate_mode_priors=[0.0, 1.0, 0.0, 0.0],
        multi_gate_mode_priors=[[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        num_threads=1,
        timeout_seconds=30,
        random_seed=11,
    )
    assert "best_mse" in result
    assert np.isfinite(float(result["best_mse"]))


def test_island_gate_priors_are_gate_modes_not_binary_ops():
    for priors in (
        PRODUCT_ISLAND_GATE_PRIORS,
        PRODUCT_ISLAND_GATE_PRIORS_MILD,
        RATIONAL_ISLAND_GATE_PRIORS,
    ):
        assert len(priors) == 4
        assert sum(priors) == pytest.approx(1.0)
        assert min(priors) >= 0.0

    # Product islands ask for multiply (mode 1).
    assert PRODUCT_ISLAND_GATE_PRIORS[1] == max(PRODUCT_ISLAND_GATE_PRIORS)
    # Rational islands ask for the soft-div gate (mode 2), which is distinct
    # from the protected BinaryOp::Division operator.
    assert RATIONAL_ISLAND_GATE_PRIORS[2] == max(RATIONAL_ISLAND_GATE_PRIORS)


def test_macro_divide_mode_emits_binary_division():
    # §3.301 is enforced in macro_mutate: the divide macro constructs
    # BinaryOp::Division directly instead of sampling an unrelated op.
    from pathlib import Path

    source = Path("glassbox/sr/cpp/evolution.h").read_text()
    marker = "// -- Divide / Rational Mutation --"
    assert marker in source
    tail = source.split(marker, 1)[1][:2000]
    assert "div_node.binary_op = BinaryOp::Division;" in tail
