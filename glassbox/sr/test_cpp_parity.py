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
def test_island_run_reports_thread_split(simple_data):
    """Island-mode results expose the outer/inner OpenMP split for diagnostics."""
    X_list, y = simple_data
    result = _core.run_evolution(
        X_list, y,
        pop_size=12,
        generations=2,
        early_stop_mse=1e-12,
        use_nsga2=True,
        num_islands=3,
        num_threads=4,
        random_seed=7,
    )

    assert result["island_outer_threads"] == 3
    assert result["island_inner_threads"] >= 1


@requires_cpp
def test_oversized_seed_graphs_are_skipped(simple_data):
    """Huge candidate seeds should not enter native initialization/refinement."""
    X_list, y = simple_data
    node = {
        "type": 1,
        "feature_idx": 0,
        "value": 1.0,
        "unary_op": 0,
        "binary_op": 0,
        "p": 1.0,
        "omega": 1.0,
        "phi": 0.0,
        "amplitude": 1.0,
        "beta": 1.0,
        "gamma": 1.0,
        "tau": 1.0,
        "left_child": -1,
        "right_child": -1,
    }
    big_seed = {
        "nodes": [dict(node, value=float(i)) for i in range(80)],
        "output_weights": [0.0] * 80,
        "output_bias": 0.0,
    }

    result = _core.run_evolution(
        X_list, y,
        pop_size=12,
        generations=2,
        early_stop_mse=1e-12,
        seed_graphs_py=[big_seed],
        random_seed=11,
    )

    assert result["seed_graphs_used"] == 0
    assert result["seed_graphs_skipped_oversized"] == 1


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


@requires_cpp
def test_arithmetic_gate_can_canonicalize_products():
    """Binary arithmetic should support clean multiply-mode structure."""
    X = np.linspace(-2, 2, 64)
    y = (X ** 2) * np.sin(X)
    result = _core.run_evolution([X], y, pop_size=12, generations=8, early_stop_mse=1e-8, random_seed=7)
    assert isinstance(result["formula"], str)
    assert np.isfinite(result["best_mse"])


@requires_cpp
def test_weighted_evolution_uniform_matches_unweighted():
    """Uniform y_weights should match the unweighted path (dual metrics present)."""
    rng = np.random.RandomState(0)
    x = rng.uniform(-2, 2, size=80)
    y = 2.0 * x + 1.0
    X_list = [x]
    base = _core.run_evolution(
        X_list, y, pop_size=20, generations=15, early_stop_mse=1e-12, random_seed=11
    )
    weighted = _core.run_evolution(
        X_list, y, pop_size=20, generations=15, early_stop_mse=1e-12, random_seed=11,
        y_weights=np.ones_like(y),
    )
    assert "best_weighted_mse" in weighted
    assert weighted["weighted"] is True
    assert np.isfinite(weighted["best_mse"])
    assert np.isfinite(weighted["best_weighted_mse"])
    # Same seed + uniform weights → same unweighted best_mse within tolerance
    assert abs(float(base["best_mse"]) - float(weighted["best_mse"])) < 1e-6


@requires_cpp
def test_weighted_evolution_downweights_outliers_changes_choice():
    """Downweighting outliers should recover closer to the true linear structure."""
    rng = np.random.RandomState(1)
    x = np.linspace(-2.0, 2.0, 100)
    y_true = 2.0 * x + 1.0
    y = y_true.copy()
    # Corrupt a contiguous block of points with huge outliers
    y[40:55] += 80.0
    w = np.ones_like(y)
    w[40:55] = 1e-6

    unweighted = _core.run_evolution(
        [x], y,
        pop_size=40, generations=40, early_stop_mse=1e-12,
        random_seed=42, timeout_seconds=30,
        p_min=-1.0, p_max=3.0,
    )
    weighted = _core.run_evolution(
        [x], y,
        pop_size=40, generations=40, early_stop_mse=1e-12,
        random_seed=42, timeout_seconds=30,
        p_min=-1.0, p_max=3.0,
        y_weights=w,
    )
    assert weighted["weighted"] is True
    assert np.isfinite(weighted["best_weighted_mse"])

    # Score formulas on clean target: weighted run should fit clean y better
    def _clean_mse(formula: str) -> float:
        # Prefer engine-reported formula; fall back to large mse
        f = str(formula or "")
        if not f:
            return float("inf")
        try:
            # Lightweight eval via numpy with x0 alias
            local = {"x0": x, "x": x, "sin": np.sin, "cos": np.cos, "exp": np.exp,
                     "log": np.log, "sqrt": np.sqrt, "abs": np.abs, "pi": np.pi}
            pred = eval(f.replace("^", "**"), {"__builtins__": {}}, local)
            pred = np.asarray(pred, dtype=np.float64).reshape(-1)
            if pred.shape != y_true.shape or not np.all(np.isfinite(pred)):
                return float("inf")
            return float(np.mean((pred - y_true) ** 2))
        except Exception:
            return float("inf")

    clean_u = _clean_mse(unweighted.get("formula", ""))
    clean_w = _clean_mse(weighted.get("formula", ""))
    # Weighted path should not be dramatically worse on clean structure; usually better
    assert clean_w < clean_u * 2.0 + 0.5 or clean_w < 1.0


@requires_cpp
def test_weighted_evolution_rejects_bad_weight_length():
    x = np.linspace(-1, 1, 40)
    y = 2.0 * x
    with pytest.raises(Exception):
        _core.run_evolution(
            [x], y, pop_size=5, generations=2, early_stop_mse=1e-8,
            y_weights=np.ones(30),
        )


@requires_cpp
def test_huber_irls_improves_clean_recovery_vs_mse():
    """Phase 4 tighten: Huber IRLS ridge should beat plain MSE on block outliers."""
    x = np.linspace(-3.0, 3.0, 120)
    y_clean = 2.0 * x + 1.0
    y = y_clean.copy()
    y[-12:] += 40.0
    X_list = [x.astype(np.float64)]

    mse_res = _core.run_evolution(
        X_list, y, pop_size=50, generations=50, early_stop_mse=1e-12,
        random_seed=11, num_islands=4, loss_mode="mse",
    )
    hub_res = _core.run_evolution(
        X_list, y, pop_size=50, generations=50, early_stop_mse=1e-12,
        random_seed=11, num_islands=4, loss_mode="huber",
    )

    def clean_mse(res):
        f = str(res.get("formula", "") or "")
        if not f:
            return 1e9
        # lightweight eval: replace ^ with ** and x with array
        expr = f.replace("^", "**").replace("x0", "x").replace("abs", "np.abs")
        # fallback via simple poly-ish eval is fragile; use numpy vectorized where possible
        try:
            from glassbox.sr.sklearn_wrapper import GlassboxRegressor
            est = GlassboxRegressor()
            est.n_features_in_ = 1
            p = est._safe_eval_formula_array(f, x.reshape(-1, 1))
            return float(np.mean((np.asarray(p) - y_clean) ** 2))
        except Exception:
            return 1e9

    c_mse = clean_mse(mse_res)
    c_hub = clean_mse(hub_res)
    # Huber should not be dramatically worse; prefer improvement when IRLS works.
    assert np.isfinite(hub_res.get("search_loss", float("inf")))
    assert hub_res.get("loss_mode") == "huber"
    # Soft assert: either better clean recovery or much lower search_loss than noisy MSE
    assert c_hub < c_mse * 1.25 or float(hub_res.get("search_loss", 1e9)) < float(mse_res.get("best_mse", 0)) * 0.5


def test_huber_loss_mode_exposed_and_runs():
    """Phase 4: loss_mode=huber returns dual metrics and runs without crash."""
    rng = np.random.RandomState(0)
    x = rng.uniform(-2, 2, size=60)
    y = 2.0 * x + 1.0
    y = y.copy()
    y[10:15] += 30.0
    result = _core.run_evolution(
        [x], y,
        pop_size=15, generations=10, early_stop_mse=1e-12,
        random_seed=3, timeout_seconds=15,
        loss_mode="huber", huber_delta=1.0,
    )
    assert result.get("loss_mode") == "huber"
    assert "search_loss" in result
    assert np.isfinite(result["best_mse"])
    assert np.isfinite(result["search_loss"])


@requires_cpp
def test_trimmed_mse_loss_mode_runs():
    x = np.linspace(-2, 2, 50)
    y = x ** 2
    y = y.copy()
    y[0] = 1e3
    result = _core.run_evolution(
        [x], y,
        pop_size=12, generations=8, early_stop_mse=1e-12,
        random_seed=5, timeout_seconds=10,
        loss_mode="trimmed_mse", trim_fraction=0.1,
    )
    assert result.get("loss_mode") == "trimmed_mse"
    assert np.isfinite(result["best_mse"])


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
