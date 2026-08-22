"""Direct coverage for residual-boosting machinery (phase7 harness decoupling).

The phase7 specialist harness historically exercised boosting only indirectly:
whether boosting runs depends on which pipeline route wins the case, so the
gate kept going stale as earlier stages got better. This file pins the boosting
machinery itself: given an imperfect base formula with a learnable residual,
``_run_residual_boosting`` must attempt stages and improve the fit.
"""

import numpy as np

from glassbox.sr.sklearn_wrapper import GlassboxRegressor


def _est():
    return GlassboxRegressor(
        random_state=3,
        generations=30,
        population_size=40,
        num_islands=1,
        timeout=60,
        adaptive_compute_budget=False,
        use_fast_path=False,
        blackbox_mode=False,
        enable_residual_stage=True,
        enable_residual_boosting=True,
        enable_specialist_vault_memory=False,
    )


def test_residual_boosting_attempts_and_improves_on_learnable_residual():
    est = _est()
    rng = np.random.RandomState(5)
    x = rng.uniform(-2.0, 2.0, size=(140, 1))
    X = np.column_stack([x[:, 0], np.zeros(140)])  # 2 features: avoids 1D skips
    y = x[:, 0] + 0.8 * np.sin(2.0 * x[:, 0])

    base_formula = "x0"
    base_pred = est._safe_eval_formula_array(base_formula, X)
    base_mse = float(np.mean((base_pred - y) ** 2))
    assert base_mse > 0.05  # base must be imperfect or there is nothing to boost

    out = est._run_residual_boosting(X, y, base_formula)

    assert isinstance(out, str) and out
    assert est.boosting_attempted_ is True
    assert len(getattr(est, "boosting_stages_", []) or []) >= 1
    assert est.boosting_diagnostics_.get("accepted_stages", 0) >= 1

    final_pred = est._safe_eval_formula_array(out, X)
    final_mse = float(np.mean((final_pred - y) ** 2))
    assert est.boosting_improved_ is True
    assert final_mse < 0.5 * base_mse


def test_residual_boosting_skips_when_base_already_exact():
    """A perfect base leaves no residual — boosting must not claim improvement."""
    est = _est()
    x = np.linspace(-1.0, 1.0, 60).reshape(-1, 1)
    X = np.column_stack([x[:, 0], np.zeros(len(x))])
    y = 2.0 * x[:, 0] + 1.0

    out = est._run_residual_boosting(X, y, "2*x0 + 1")

    # Either skipped outright or attempted without claiming improvement.
    if getattr(est, "boosting_attempted_", False):
        assert est.boosting_improved_ is False or est.boosting_diagnostics_.get(
            "accepted_stages", 0
        ) == 0
    assert isinstance(out, str) and out
