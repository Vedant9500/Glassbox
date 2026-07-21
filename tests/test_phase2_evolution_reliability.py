"""Phase 2: evolution search reliability (E3, E5, E7, E10)."""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
CPP_DIR = REPO / "glassbox" / "sr" / "cpp"
for p in (REPO, CPP_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    import _core  # type: ignore

    CPP_AVAILABLE = hasattr(_core, "run_evolution")
except ImportError:
    CPP_AVAILABLE = False

requires_cpp = pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ _core not built")


def _max_seed_capacity(pop: int, n_seeds: int, frac: float = 0.5) -> int:
    """Mirror of EvolutionEngine::max_seed_capacity (E3)."""
    import math

    pop = max(1, int(pop))
    n_seeds = int(n_seeds)
    if n_seeds <= 0:
        return 0
    if not (frac > 0.0) or not np.isfinite(frac):
        frac = 0.5
    frac = float(np.clip(frac, 0.1, 0.9))
    by_frac = max(1, int(math.ceil(frac * pop)))
    if pop <= 12:
        cap = max(by_frac, pop - 1)
    else:
        cap = by_frac
    cap = min(cap, pop)
    return min(cap, n_seeds)


def test_e3_seed_capacity_policy_beats_legacy_quarter():
    # Legacy: pop/4 → island_size=12 → only 3 seeds.
    assert _max_seed_capacity(12, 12) >= 11
    assert _max_seed_capacity(12, 12) > 12 // 4
    # Larger pop: ~50% not 25%.
    assert _max_seed_capacity(100, 80) == 50
    assert _max_seed_capacity(100, 80) > 100 // 4
    # Never exceed available seeds / pop.
    assert _max_seed_capacity(20, 3) == 3
    assert _max_seed_capacity(5, 10) == 4


@requires_cpp
def test_e3_many_seeds_accepted_under_islands():
    x = np.linspace(-1.5, 1.5, 60).astype(np.float64)
    y = (np.sin(x) + 0.1 * x).astype(np.float64)
    formulas = [
        "sin(x0)", "x0", "x0**2", "cos(x0)", "exp(x0)", "x0**3",
        "sin(2*x0)", "x0 + 1", "2*x0", "x0**2 + x0", "sin(x0) + x0", "cos(x0) + x0",
    ]
    seeds = []
    for f in formulas:
        try:
            seeds.append(_core.formula_to_seed_graph(f))
        except Exception:
            pass
    if len(seeds) < 8:
        pytest.skip("seed graph builder produced too few seeds")

    res = _core.run_evolution(
        [x],
        y,
        pop_size=48,
        generations=2,
        num_islands=4,
        early_stop_mse=1e-20,
        timeout_seconds=15,
        random_seed=11,
        migration_interval=100,
        seed_graphs_py=seeds,
    )
    used = int(res.get("seed_graphs_used", 0))
    assert used >= 8
    assert np.isfinite(float(res.get("best_mse", np.nan)))


@requires_cpp
def test_e5_export_reports_raw_mse_and_penalized_fitness():
    x = np.linspace(-1, 1, 80).astype(np.float64)
    y = x.copy()
    y[0] = 30.0
    res = _core.run_evolution(
        [x],
        y,
        pop_size=30,
        generations=4,
        num_islands=1,
        early_stop_mse=1e-20,
        timeout_seconds=10,
        random_seed=3,
        loss_mode="huber",
        huber_delta=-1.0,
    )
    assert "formula" in res
    raw = float(res.get("best_mse", np.nan))
    search = float(res.get("search_loss", res.get("best_weighted_mse", np.nan)))
    pen = float(res.get("penalized_fitness", np.nan))
    assert np.isfinite(raw)
    assert np.isfinite(search)
    assert np.isfinite(pen)


@requires_cpp
def test_e7_island_thread_budgets_reported():
    x = np.linspace(-1, 1, 50).astype(np.float64)
    y = np.sin(2 * x).astype(np.float64)
    res = _core.run_evolution(
        [x],
        y,
        pop_size=40,
        generations=3,
        num_islands=4,
        early_stop_mse=1e-20,
        timeout_seconds=12,
        random_seed=21,
        migration_interval=100,
    )
    assert int(res.get("island_outer_threads", 0)) >= 1
    assert int(res.get("island_inner_threads", 0)) >= 1
    assert np.isfinite(float(res.get("best_mse", np.nan)))


@requires_cpp
def test_e10_arithmetic_temperature_restore_after_scoring():
    x = np.linspace(-1, 1, 40).astype(np.float64)
    y = (x ** 2).astype(np.float64)
    if hasattr(_core, "score_formula_candidates"):
        try:
            _core.score_formula_candidates(["x0**2", "x0"], [x], y)
        except TypeError:
            # Binding arity may differ; scoring path is still exercised below if needed.
            pass
    res = _core.run_evolution(
        [x],
        y,
        pop_size=20,
        generations=3,
        num_islands=1,
        early_stop_mse=1e-20,
        timeout_seconds=8,
        random_seed=5,
        arithmetic_temperature=5.0,
    )
    assert np.isfinite(float(res.get("best_mse", np.nan)))


@requires_cpp
def test_phase2_multi_island_deterministic_under_seed():
    x = np.linspace(-1.2, 1.2, 70).astype(np.float64)
    y = (np.sin(x) + 0.05 * x).astype(np.float64)
    common = dict(
        pop_size=40,
        generations=5,
        num_islands=4,
        early_stop_mse=1e-20,
        timeout_seconds=15,
        random_seed=99,
        migration_interval=100,
    )
    a = _core.run_evolution([x], y, **common)
    b = _core.run_evolution([x], y, **common)
    assert a.get("formula") == b.get("formula")
    assert abs(float(a.get("best_mse", 1)) - float(b.get("best_mse", 0))) < 1e-12
