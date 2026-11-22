"""§3.330/§3.331: seed power domain clamp and variable-exponent cap parity."""

import numpy as np
import pytest

from glassbox.sr.cpp import CPP_AVAILABLE, get_cpp_core

pytestmark = pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ _core not built")


def test_seed_power_outside_domain_is_clamped():
    core = get_cpp_core()
    x = np.linspace(0.5, 2.0, 40)
    y = x**9
    result = core.run_evolution(
        [x],
        y,
        pop_size=8,
        generations=1,
        num_islands=1,
        p_min=1.0,
        p_max=3.0,
        num_threads=1,
        timeout_seconds=20,
        seed_graphs_py=[core.formula_to_seed_graph("x0^9")],
    )
    assert result["seed_power_clamped"] == 1
    assert np.isfinite(float(result["best_mse"]))


def test_seed_power_inside_domain_is_untouched():
    core = get_cpp_core()
    x = np.linspace(0.5, 2.0, 40)
    y = x**2
    result = core.run_evolution(
        [x],
        y,
        pop_size=8,
        generations=1,
        num_islands=1,
        p_min=1.0,
        p_max=3.0,
        num_threads=1,
        timeout_seconds=20,
        seed_graphs_py=[core.formula_to_seed_graph("x0^2")],
    )
    assert result["seed_power_clamped"] == 0


def test_variable_exponent_overflow_is_capped():
    core = get_cpp_core()
    values = np.array([1e4, -1e4, 1e3])
    out = np.asarray(core.eval_formula_exact("x0^(x0)", [values]))
    assert np.all(np.isfinite(out))
    assert np.all(np.abs(out) <= 1e8)


def test_constant_exponent_overflow_is_capped():
    core = get_cpp_core()
    values = np.array([1e4, -1e4, 1e3])
    for formula in ("x0^6", "x0^7"):
        out = np.asarray(core.eval_formula_exact(formula, [values]))
        assert np.all(np.isfinite(out))
        assert np.all(np.abs(out) <= 1e8)
