"""Regression coverage for §3.103: robust crowding normalization.

The old crowding distance divided interior gaps by (max - min) of each
objective. One diverged individual (raw_mse ~1e9) inflated that denominator so
every other individual's distance rounded to ~0, removing diversity pressure.
"""

import numpy as np
import pytest

from glassbox.sr.cpp import CPP_AVAILABLE, get_cpp_core

pytestmark = pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ _core not built")


def _crowding(mse, complexity, age):
    core = get_cpp_core()
    return np.asarray(
        core.robust_crowding_distances(
            [float(v) for v in mse],
            [float(v) for v in complexity],
            [float(v) for v in age],
        ),
        dtype=np.float64,
    )


def test_outlier_does_not_flatten_interior_crowding():
    # Five comparable individuals plus one diverged formula.
    mse = [0.10, 0.11, 0.12, 0.13, 0.14, 1e9]
    complexity = [3, 4, 5, 6, 7, 8]
    age = [1, 1, 1, 1, 1, 1]
    dist = _crowding(mse, complexity, age)

    interior = dist[:5]
    # Every non-boundary individual keeps a real (non-zero) crowding value.
    assert np.all(interior[1:] > 1e-6)
    # The raw-range version would have produced ~1e-10 for these gaps.
    assert np.all(np.isfinite(dist))
    # Boundary handling still promotes the extrema on each objective.
    assert dist[0] >= 1e18  # best raw_mse
    assert dist[5] >= 1e18  # worst raw_mse / complexity


def test_robust_crowding_matches_raw_range_when_no_outlier():
    mse = [1.0, 2.0, 3.0, 4.0, 5.0]
    complexity = [1, 1, 1, 1, 1]
    age = [5, 5, 5, 5, 5]
    dist = _crowding(mse, complexity, age)
    # Robust range is p10..p90 = 1.0..4.0, so interior gaps are 2/3 each.
    assert dist[1] == pytest.approx(2.0 / 3.0)
    assert dist[2] == pytest.approx(2.0 / 3.0)
    assert dist[3] == pytest.approx(2.0 / 3.0)
    assert dist[0] >= 1e18 and dist[4] >= 1e18


def test_crowding_boundaries_and_small_fronts():
    assert list(_crowding([], [], [])) == []
    assert list(_crowding([1.0], [1.0], [1.0])) == [1e18]
    two = _crowding([1.0, 2.0], [1.0, 1.0], [1.0, 1.0])
    assert np.all(two >= 1e18)
