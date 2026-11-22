"""§3.257/§3.262/§3.353: deadline checks, migration skip, thread budget."""

import time

import numpy as np
import pytest

from glassbox.sr.cpp import CPP_AVAILABLE, get_cpp_core

pytestmark = pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ _core not built")


def _target(seed=0, n=120):
    rng = np.random.RandomState(seed)
    x = np.linspace(0.0, 5.0, n)
    y = np.exp(-0.3 * x) * np.sin(3.1 * x) + 0.15 * rng.normal(size=x.size)
    return x, y


def test_forced_deadline_truncates_generation_work():
    core = get_cpp_core()
    x, y = _target()
    start = time.time()
    result = core.run_evolution(
        [x],
        y,
        pop_size=200,
        generations=200,
        num_islands=4,
        timeout_seconds=5,
        num_threads=4,
        random_seed=2,
        debug_force_deadline=True,
    )
    elapsed = time.time() - start
    assert result["termination_reason"] == "timeout"
    # Offspring reproduction stopped once the (already spent) budget was seen.
    assert result["deadline_truncated_children"] > 0
    # §3.262: no migration is performed when the run is already stopping.
    assert result["migrations_performed"] == 0
    assert elapsed < 10.0


def test_wall_clock_budget_is_respected():
    core = get_cpp_core()
    x, y = _target(seed=1)
    start = time.time()
    result = core.run_evolution(
        [x],
        y,
        pop_size=300,
        generations=5000,
        num_islands=4,
        timeout_seconds=1,
        num_threads=4,
        random_seed=3,
    )
    elapsed = time.time() - start
    assert result["termination_reason"] == "timeout"
    assert elapsed < 5.0


def test_inner_team_is_sized_from_requested_budget():
    core = get_cpp_core()
    x, y = _target(seed=2, n=60)
    for num_threads, islands in ((4, 4), (2, 8), (8, 4)):
        result = core.run_evolution(
            [x],
            y,
            pop_size=8 * islands,
            generations=2,
            num_islands=islands,
            timeout_seconds=30,
            num_threads=num_threads,
            random_seed=1,
        )
        outer = int(result["island_outer_threads"])
        inner = int(result["island_inner_threads"])
        assert outer >= 1 and inner >= 1
        # §3.353: the product must not exceed the requested budget.
        assert outer * inner <= max(1, num_threads)
        assert outer <= islands
