"""Regression coverage for §3.139: trimmed IRLS optimizes the scored surrogate.

The old IRLS fit kept trimmed rows at a 1e-6 soft weight while the fitness
objective hard-excluded them, so the coefficients and the reported trimmed
objective optimized different problems. The fit now drops the same rows with
weight 0 (by mass when sample weights are set, by count otherwise).
"""

import numpy as np
import pytest

from glassbox.sr.cpp import CPP_AVAILABLE, get_cpp_core
from glassbox.sr.sklearn_wrapper import GlassboxRegressor

pytestmark = pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ _core not built")


def _run_trimmed(n_outliers=15):
    x = np.linspace(-3.0, 3.0, 120)
    y_clean = 2.0 * x + 1.0
    y = y_clean.copy()
    y[-n_outliers:] += 40.0
    result = get_cpp_core().run_evolution(
        [x.astype(np.float64)],
        y,
        pop_size=40,
        generations=40,
        early_stop_mse=1e-12,
        random_seed=11,
        num_islands=4,
        loss_mode="trimmed_mse",
        trim_fraction=0.15,
    )
    return x, y_clean, result


def test_trimmed_irls_hard_drops_match_fitness_surrogate():
    x, y_clean, result = _run_trimmed()
    # The fit must actually use hard zero weights, not the old soft 1e-6.
    assert int(result["trimmed_irls_hard_drops"]) > 0
    assert np.isfinite(float(result["best_mse"]))

    # And the recovered function is the clean linear model, not a compromise
    # that still chases the block outliers.
    est = GlassboxRegressor()
    est.n_features_in_ = 1
    pred = np.asarray(
        est._safe_eval_formula_array(str(result["formula"]), x.reshape(-1, 1))
    ).reshape(-1)
    clean_mse = float(np.mean((pred - y_clean) ** 2))
    assert clean_mse < 5.0


def test_plain_mse_run_reports_no_trimmed_drops():
    x = np.linspace(-2.0, 2.0, 80)
    y = np.sin(x)
    result = get_cpp_core().run_evolution(
        [x.astype(np.float64)],
        y,
        pop_size=12,
        generations=6,
        random_seed=5,
        num_islands=2,
    )
    assert int(result["trimmed_irls_hard_drops"]) == 0
    assert int(result["trimmed_irls_soft_fallbacks"]) == 0
