"""§3.286/§3.287: never emit a disallowed op or a forbidden nested wrap."""

import numpy as np
import pytest

from glassbox.sr.cpp import CPP_AVAILABLE, get_cpp_core

pytestmark = pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ _core not built")

INPUT, UNARY = 0, 2
PERIODIC, POWER, INTPOW, EXP, LOG, ABS = 0, 1, 2, 3, 4, 5


def test_wrap_search_fails_when_allowed_and_wrap_sets_do_not_overlap():
    core = get_cpp_core()
    # Power child forbids Power/IntPow wrapping; only IntPow is allowed.
    assert core.find_wrap_safe_unary_op([UNARY], [POWER], 0, [INTPOW], 0) == -1
    # Exp child forbids Exp wrapping; only Exp is allowed.
    assert core.find_wrap_safe_unary_op([UNARY], [EXP], 0, [EXP], 0) == -1
    # Periodic child forbids Periodic; only Periodic is allowed.
    assert core.find_wrap_safe_unary_op([UNARY], [PERIODIC], 0, [PERIODIC], 0) == -1


def test_wrap_search_finds_an_allowed_safe_op():
    core = get_cpp_core()
    assert core.find_wrap_safe_unary_op([INPUT], [PERIODIC], 0, [INTPOW], 0) == INTPOW
    assert core.find_wrap_safe_unary_op([UNARY], [EXP], 0, [PERIODIC], 0) == PERIODIC


def test_wrap_search_never_returns_a_disallowed_op():
    core = get_cpp_core()
    for seed in range(20):
        chosen = core.find_wrap_safe_unary_op(
            [INPUT], [PERIODIC], 0, [PERIODIC], seed
        )
        assert chosen == PERIODIC


def test_engine_with_narrow_unary_mask_emits_only_allowed_ops():
    core = get_cpp_core()
    x = np.linspace(-2.0, 2.0, 48)
    y = x**2
    result = core.run_evolution(
        [x],
        y,
        pop_size=14,
        generations=6,
        allowed_unary_ops=[INTPOW],
        allowed_binary_ops=[0],
        num_islands=1,
        num_threads=1,
        timeout_seconds=60,
        random_seed=5,
    )
    formula = str(result.get("formula", ""))
    for forbidden in ("sin(", "cos(", "exp(", "log(", "abs("):
        assert forbidden not in formula
