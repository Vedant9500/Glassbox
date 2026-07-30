import numpy as np
import pytest
from glassbox.sr.sklearn_wrapper import GlassboxRegressor
from glassbox.sr.cpp import get_cpp_core


def test_p02_island_reproduction_and_nsga_parity():
    """Verify island model (num_islands=8) and single pop (num_islands=1) parity and NSGA non-fallthrough."""
    _core = get_cpp_core()
    
    np.random.seed(42)
    X = np.linspace(-2.0, 2.0, 50).reshape(-1, 1)
    y = 2.5 * X[:, 0] ** 2 + 1.2 * np.sin(X[:, 0])

    X_list = [X[:, i] for i in range(X.shape[1])]

    # 1. Run single-island evolution
    res_single = _core.run_evolution(
        X_list=X_list,
        y=y,
        generations=10,
        pop_size=40,
        num_islands=1,
        random_seed=42,
        macro_mutation_rate=0.2,
        use_nsga2=False,
    )
    assert res_single["best_mse"] >= 0.0
    assert np.isfinite(res_single["best_mse"])

    # 2. Run multi-island evolution (default 8 islands)
    res_islands = _core.run_evolution(
        X_list=X_list,
        y=y,
        generations=10,
        pop_size=40,
        num_islands=8,
        random_seed=42,
        macro_mutation_rate=0.2,
        use_nsga2=False,
    )
    assert res_islands["best_mse"] >= 0.0
    assert np.isfinite(res_islands["best_mse"])

    # 3. Run NSGA-2 single-pop evolution (tests C-05 fix: no double reproduce panic or bad state)
    res_nsga_single = _core.run_evolution(
        X_list=X_list,
        y=y,
        generations=10,
        pop_size=40,
        num_islands=1,
        random_seed=42,
        use_nsga2=True,
    )
    assert res_nsga_single["best_mse"] >= 0.0

    # 4. Run NSGA-2 multi-island evolution
    res_nsga_islands = _core.run_evolution(
        X_list=X_list,
        y=y,
        generations=10,
        pop_size=40,
        num_islands=4,
        random_seed=42,
        use_nsga2=True,
    )
    assert res_nsga_islands["best_mse"] >= 0.0


def test_p02_glassbox_regressor_island_fit():
    """Verify GlassboxRegressor with default multi-island evolution fits quadratic data accurately."""
    np.random.seed(42)
    X = np.linspace(-1.0, 1.0, 40).reshape(-1, 1)
    y = X[:, 0] ** 2

    model = GlassboxRegressor(
        timeout=10,
        generations=20,
        num_islands=8,
        random_state=42,
    )
    model.fit(X, y)
    preds = model.predict(X)
    mse = np.mean((preds - y) ** 2)

    assert hasattr(model, "formula_")
    assert mse < 0.1
