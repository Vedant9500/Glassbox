"""Regression tests for audit finding P-04 (FD Adam full re-eval per param).

P-04: ``refine_inner_params_adam`` re-evaluated the whole graph twice per
parameter per Adam step (6 * n_nodes full evals per step), including probes of
parameters that are analytically inert for the node's unary op
(``p`` on Periodic/Exp, ``omega``/``phi`` on Power/IntPow, all three on
Log/Abs) — pure finite-difference noise.

The fix: per-step base cache + incremental subtree re-evaluation
(``evaluate_graph_partial``), inert-parameter skipping, and exported
diagnostics ``fd_probes_total`` / ``fd_probes_skipped_inert``.
"""

import numpy as np
import pytest

from glassbox.sr.cpp import get_cpp_core, CPP_AVAILABLE

pytestmark = pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ core unavailable")


@pytest.fixture(scope="module")
def core():
    return get_cpp_core()


@pytest.fixture(scope="module")
def periodic_power_problem(core):
    """sin + power target whose seed graphs contain inert-param nodes."""
    x = np.linspace(0.2, 3.0, 120)
    y = np.sin(3.0 * x) + x ** 1.5
    seeds = [
        core.formula_to_seed_graph(s)
        for s in ("sin(3*x0)+x0**1.5", "sin(2*x0)", "x0**2", "log(x0+1)")
    ]
    return x, y, seeds


def _run(core, x, y, seeds, use_lm):
    return core.run_evolution(
        [x], y,
        pop_size=30,
        generations=15,
        num_islands=1,
        random_seed=42,
        early_stop_mse=0.0,
        timeout_seconds=120,
        seed_graphs_py=seeds,
        use_lm_inner_optimizer=use_lm,
    )


def test_p04_fd_adam_skips_inert_params(core, periodic_power_problem):
    """FD-Adam path must skip analytically-inert parameter probes."""
    x, y, seeds = periodic_power_problem
    res = _run(core, x, y, seeds, use_lm=False)

    total = res["fd_probes_total"]
    skipped = res["fd_probes_skipped_inert"]

    # Refinement ran and probed live parameters.
    assert total > 0
    # Inert slots (p on Periodic, omega/phi on Power, all on Log) were skipped.
    assert skipped > 0
    # Counter bookkeeping stays consistent.
    assert skipped <= total
    # Search quality preserved on the forced FD-Adam path.
    assert np.isfinite(res["best_mse"])
    assert res["best_mse"] < 0.05


def test_p04_lm_default_path_unchanged(core, periodic_power_problem):
    """Default LM-first path still refines and reports diagnostics."""
    x, y, seeds = periodic_power_problem
    res = _run(core, x, y, seeds, use_lm=True)

    assert np.isfinite(res["best_mse"])
    assert res["best_mse"] < 0.05
    assert res["fd_probes_total"] >= res["fd_probes_skipped_inert"] >= 0


def test_p04_fixed_seed_determinism(core, periodic_power_problem):
    """Same seed produces identical results (refactor must not perturb search)."""
    x, y, seeds = periodic_power_problem
    a = _run(core, x, y, seeds, use_lm=False)
    b = _run(core, x, y, seeds, use_lm=False)
    assert a["best_mse"] == b["best_mse"]
