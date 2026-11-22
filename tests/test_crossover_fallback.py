"""Regression coverage for §3.418: failed crossover retries fall back to mutation."""

import numpy as np
import pytest

from glassbox.sr.cpp import CPP_AVAILABLE, get_cpp_core

pytestmark = pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ _core not built")


def test_crossover_fallback_counter_and_result_are_consistent():
    core = get_cpp_core()
    rng = np.random.RandomState(0)
    X = rng.uniform(-2.0, 2.0, size=120)
    y = np.sin(X) + 0.1 * rng.normal(size=120)

    result = core.run_evolution(
        [X],
        y,
        pop_size=10,
        generations=12,
        crossover_rate=1.0,
        elite_size=2,
        random_seed=3,
        early_stop_mse=1e-12,
    )

    attempts = int(result["crossover_attempts"])
    successes = int(result["crossover_successes"])
    fallbacks = int(result["crossover_fallback_mutations"])

    assert attempts > 0
    # Every failed retry batch either produced a valid graft or fell back to a
    # full-rate mutation; a fallback is never also counted as a success.
    assert 0 <= fallbacks
    assert successes + fallbacks <= attempts
    assert np.isfinite(float(result["best_mse"]))
    assert str(result["formula"]).strip()
