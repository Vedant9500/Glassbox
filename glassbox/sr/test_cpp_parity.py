"""
C++ Backend Parity Tests.

Verifies that the C++ _core extension loads, executes, and returns
structurally valid results. These are smoke tests — they do NOT assert
numerical exactness of the C++ evolution, only that the Python ↔ C++
bridge is functional.

Run with:
    pytest glassbox/sr/test_cpp_parity.py -v
    python glassbox/sr/test_cpp_parity.py      (direct execution still works)
"""

import numpy as np
import sys
from pathlib import Path

import pytest

# Ensure the built C++ extension can be found
cpp_dir = Path(__file__).parent / 'cpp'
if str(cpp_dir) not in sys.path:
    sys.path.insert(0, str(cpp_dir))

# ── Import guard ────────────────────────────────────────────────────────
try:
    import _core
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False

requires_cpp = pytest.mark.skipif(
    not CPP_AVAILABLE,
    reason="C++ _core extension not built. Run `python setup.py build_ext --inplace` in glassbox/sr/cpp/",
)

# ── Shared fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def simple_data():
    """y = 2.0 * x^2 + sin(3.0 * x)"""
    np.random.seed(42)
    X = np.random.uniform(-3, 3, size=(100, 1))
    X_list = [X[:, 0]]
    y = 2.0 * (X[:, 0] ** 2) + np.sin(3.0 * X[:, 0])
    return X_list, y


# ── Tests ────────────────────────────────────────────────────────────────

@requires_cpp
def test_core_import():
    """_core should import and expose run_evolution."""
    assert hasattr(_core, "run_evolution"), "_core is missing run_evolution"


@requires_cpp
def test_basic_execution(simple_data):
    """run_evolution should execute and return a dict with expected keys."""
    X_list, y = simple_data
    result = _core.run_evolution(X_list, y, pop_size=10, generations=5, early_stop_mse=1e-8)

    assert isinstance(result, dict), "Result should be a dict"
    assert "formula" in result, "Result missing 'formula' key"
    assert "best_mse" in result, "Result missing 'best_mse' key"
    assert isinstance(result["formula"], str), "Formula should be a string"
    assert np.isfinite(result["best_mse"]), "best_mse should be finite"


@requires_cpp
def test_seed_omegas(simple_data):
    """run_evolution should accept seed_omegas without crashing."""
    X_list, y = simple_data
    result = _core.run_evolution(
        X_list, y,
        pop_size=10, generations=5,
        early_stop_mse=1e-8,
        seed_omegas=[3.0, 5.0, 7.0],
    )

    assert isinstance(result, dict)
    assert "formula" in result
    assert np.isfinite(result["best_mse"])


@requires_cpp
def test_timeout_parameter(simple_data):
    """run_evolution should respect timeout_seconds without crashing."""
    X_list, y = simple_data
    result = _core.run_evolution(
        X_list, y,
        pop_size=10, generations=500,
        early_stop_mse=1e-12,
        timeout_seconds=2,
    )

    assert isinstance(result, dict)
    assert "best_mse" in result


@requires_cpp
def test_random_seed_determinism(simple_data):
    """Two runs with the same random_seed should produce identical results."""
    X_list, y = simple_data
    kwargs = dict(
        X_list=X_list, y=y,
        pop_size=10, generations=10,
        early_stop_mse=1e-12,
        random_seed=12345,
    )

    result_a = _core.run_evolution(**kwargs)
    result_b = _core.run_evolution(**kwargs)

    assert result_a["formula"] == result_b["formula"], (
        f"Same seed should produce same formula: "
        f"'{result_a['formula']}' vs '{result_b['formula']}'"
    )
    assert abs(result_a["best_mse"] - result_b["best_mse"]) < 1e-12, (
        "Same seed should produce same MSE"
    )


@requires_cpp
def test_result_schema(simple_data):
    """Verify the full result dict schema from the C++ backend."""
    X_list, y = simple_data
    result = _core.run_evolution(
        X_list, y,
        pop_size=10, generations=5,
        early_stop_mse=1e-8,
        random_seed=42,
    )

    expected_keys = [
        "best_mse", "penalized_fitness", "formula",
        "nodes", "output_weights", "output_bias",
        "evolution_wall_time_sec", "random_seed", "openmp_threads",
        "time_to_first_exact_sec", "generation_to_first_exact",
        "time_to_first_acceptable_sec", "generation_to_first_acceptable",
    ]
    for key in expected_keys:
        assert key in result, f"Result missing expected key: '{key}'"

    assert isinstance(result["nodes"], list), "nodes should be a list"
    assert isinstance(result["output_weights"], list), "output_weights should be a list"


# ── Direct execution (backward compatibility) ───────────────────────────

if __name__ == "__main__":
    if not CPP_AVAILABLE:
        print("❌ Failed to import _core")
        print("Please build the C++ extension first using "
              "`python setup.py build_ext --inplace` in the cpp directory.")
        sys.exit(1)

    print("✅ Successfully imported _core")

    np.random.seed(42)
    X = np.random.uniform(-3, 3, size=(100, 1))
    X_list = [X[:, 0]]
    y = 2.0 * (X[:, 0] ** 2) + np.sin(3.0 * X[:, 0])

    print("\n--- Test 1: Basic execution ---")
    res1 = _core.run_evolution(X_list, y, pop_size=10, generations=5, early_stop_mse=1e-8)
    print("Formula:", res1["formula"])
    print("MSE:", res1["best_mse"])

    print("\n--- Test 2: Seed Omegas ---")
    res2 = _core.run_evolution(X_list, y, pop_size=10, generations=5,
                               early_stop_mse=1e-8, seed_omegas=[3.0, 5.0, 7.0])
    print("Formula:", res2["formula"])
    print("MSE:", res2["best_mse"])

    print("\nAll Python API bindings execute without errors.")
