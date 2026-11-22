"""§3.190: boosting keeps an untouched final gate and rotates tuning rows."""

import numpy as np

from glassbox.sr.sklearn_wrapper import GlassboxRegressor


class _GateProbe(GlassboxRegressor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fit_rows_seen = None
        self.refine_rows_seen = None

    def _stage_residual_symbolic_fit(
        self, X, y, base_formula, *, _allow_recursion=False
    ):
        self.fit_rows_seen = int(np.asarray(X).shape[0])
        return "sin(x0)"

    def _refine_candidate_formulas(
        self, candidate_formulas, X, y, *, max_candidates=12
    ):
        self.refine_rows_seen = int(np.asarray(X).shape[0])
        return list(candidate_formulas)[:max_candidates]


def test_boosting_excludes_gate_rows_from_fit_and_reports_gate():
    rng = np.random.RandomState(11)
    x = np.sort(rng.uniform(0.0, 10.0, size=120))
    gate_n = 12
    y = x + 0.8 * np.sin(x)
    # Gate rows are exactly the base formula, so any boosted term must hurt
    # there even though it helps the tuning rows.
    y[-gate_n:] = x[-gate_n:]
    X = x.reshape(-1, 1)

    reg = _GateProbe(
        random_state=7,
        generations=20,
        population_size=30,
        num_islands=1,
        timeout=60,
        adaptive_compute_budget=False,
        use_fast_path=False,
        blackbox_mode=False,
        enable_residual_stage=True,
        enable_residual_boosting=True,
        enable_specialist_vault_memory=False,
        max_boosting_stages=2,
        boosting_learning_rates=[0.5, 1.0],
    )

    out = reg._run_residual_boosting(X, y, "x0")
    diag = reg.boosting_diagnostics_

    assert diag["holdout_mode"] == "rotating_tune_block+final_gate"
    assert diag["holdout_is_tuning_set"] is True
    assert diag["gate_is_tuning_set"] is False
    assert diag["gate_size"] == gate_n

    # The gate rows never reach stage fitting or refinement.
    assert reg.fit_rows_seen == X.shape[0] - gate_n
    assert reg.refine_rows_seen in (None, X.shape[0] - gate_n)

    # The untouched gate rejects a formula that only helped the tuning rows.
    assert diag["gate_rejected"] is True
    assert diag["gate_reject_reason"] == "gate_regression"
    assert out == "x0"
    assert reg.boosting_improved_ is False
    assert reg.boosting_stages_ == []


def test_boosting_accepts_when_gate_does_not_regress():
    rng = np.random.RandomState(5)
    x = np.sort(rng.uniform(-2.0, 2.0, size=140))
    y = x + 0.8 * np.sin(x)
    X = x.reshape(-1, 1)

    reg = _GateProbe(
        random_state=3,
        generations=20,
        population_size=30,
        num_islands=1,
        timeout=60,
        adaptive_compute_budget=False,
        use_fast_path=False,
        blackbox_mode=False,
        enable_residual_stage=True,
        enable_residual_boosting=True,
        enable_specialist_vault_memory=False,
        max_boosting_stages=2,
        boosting_learning_rates=[0.5, 1.0],
    )

    out = reg._run_residual_boosting(X, y, "x0")
    diag = reg.boosting_diagnostics_
    assert diag["gate_rejected"] is False
    assert diag["gate_r2"] >= diag["gate_base_r2"] - 1e-3
    assert out != "x0"
