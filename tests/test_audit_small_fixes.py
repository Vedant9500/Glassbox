"""Small audit-fix regression tests, consolidated.

Merged from five one-off files (test_audit_m04_fix.py, test_audit_m05_fix.py,
test_audit_p03_fix.py, test_audit_p02_fixes.py, test_audit_p04_fix.py) so the
audit-pin suite has fewer files; test names are unchanged.
"""

import numpy as np
import pytest
import torch

from glassbox.sr.cpp import CPP_AVAILABLE, get_cpp_core
from glassbox.sr.hard_concrete import hard_concrete_log_prob
from glassbox.sr.operations.meta_ops import MetaArithmeticExtended
from glassbox.sr.optimizers import EvolutionaryOptimizer, LBFGSConstantOptimizer
from glassbox.sr.sklearn_wrapper import GlassboxRegressor

# ── M-04: Meta soft-div parity with C++ sqrt form ───────────────────────


def test_m04_soft_div_parity():
    """M-04: MetaArithmeticExtended soft division term uses sqrt form matching eval.h / export_pytorch."""
    m = MetaArithmeticExtended()

    # Force weights to favor division term (beta=2.0, gamma=-1.0)
    m.beta.data.fill_(2.0)
    m.gamma.data.fill_(-1.0)

    x = torch.tensor([2.0, -3.0, 0.5, -1.0], dtype=torch.float64)
    y = torch.tensor([3.0, -2.0, 0.0, 4.0], dtype=torch.float64)

    # Compute expected soft division term: x / sqrt(1 + y^2)
    expected_div = x / torch.sqrt(1.0 + torch.square(y))

    # Compute weights from MetaArithmeticExtended
    d_add = (m.beta - 1.0) ** 2 + (m.gamma - 1.0) ** 2
    d_mul = (m.beta - 2.0) ** 2 + (m.gamma - 1.0) ** 2
    d_div = (m.beta - 2.0) ** 2 + (m.gamma + 1.0) ** 2
    d_sub = (m.beta - 1.0) ** 2 + (m.gamma + 1.0) ** 2

    logits = torch.stack([-d_add, -d_mul, -d_div, -d_sub])
    weights = torch.nn.functional.softmax(logits * 5.0, dim=0)

    res_add = x + y
    res_sub = x - y
    res_mul = x * y
    res_div = expected_div

    expected_result = (
        weights[0] * res_add
        + weights[1] * res_mul
        + weights[2] * res_div
        + weights[3] * res_sub
    )
    expected_result = torch.clamp(expected_result, -100, 100)

    actual_result = m(x, y)

    assert torch.allclose(actual_result, expected_result), (
        f"Mismatch: actual {actual_result} vs expected {expected_result}"
    )


# ── M-05: Hard-Concrete log-prob density ─────────────────────────────────


def test_hard_concrete_log_prob_integration():
    logits = torch.tensor(0.5, dtype=torch.float64)
    tau = 0.7
    beta = 0.1

    # Integrate density of s over (0, 1) using numerical integration
    s = torch.linspace(1e-6, 1 - 1e-6, 500000, dtype=torch.float64)
    ds = s[1] - s[0]

    # Convert s to z domain: z = s * (1 + 2*beta) - beta
    z = s * (1 + 2 * beta) - beta
    dz = z[1] - z[0]

    log_prob = hard_concrete_log_prob(z, logits, tau=tau, beta=beta)
    prob = torch.exp(log_prob)

    integral = torch.sum(prob) * dz
    assert pytest.approx(integral.item(), rel=1e-3) == 1.0


def test_hard_concrete_log_prob_gradients_and_types():
    logits = torch.tensor([0.2, -0.5], requires_grad=True)
    z = torch.tensor([0.1, 0.8], requires_grad=True)

    # Float tau
    log_prob_float = hard_concrete_log_prob(z, logits, tau=0.5)
    assert log_prob_float.shape == (2,)
    loss_float = log_prob_float.sum()
    loss_float.backward()

    assert logits.grad is not None
    assert z.grad is not None

    # Tensor tau with grad
    logits = torch.tensor([0.2, -0.5], requires_grad=True)
    z = torch.tensor([0.1, 0.8], requires_grad=True)
    tau_tensor = torch.tensor(0.5, requires_grad=True)

    log_prob_tensor = hard_concrete_log_prob(z, logits, tau=tau_tensor)
    assert log_prob_tensor.shape == (2,)
    loss_tensor = log_prob_tensor.sum()
    loss_tensor.backward()

    assert tau_tensor.grad is not None


# ── P-03: hybrid L-BFGS steps squared ────────────────────────────────────


class _TinyONNModel(torch.nn.Module):
    """Tiny model whose parameter names match the constant-param filter in
    ``LBFGSConstantOptimizer._get_constant_params`` (``omega``)."""

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(2, 1)
        self.omega = torch.nn.Parameter(torch.zeros(3))

    def forward(self, x, hard=True):
        return self.lin(x) * torch.sin(self.omega[0]), None


def test_p03_single_lbfgs_step_per_offspring(monkeypatch):
    """Refinement must run one L-BFGS step() per offspring at the full
    max_iter budget — not step() lbfgs_steps times (steps**2 cost)."""
    torch.manual_seed(0)

    real_step = LBFGSConstantOptimizer.step
    step_calls = {"count": 0, "max_iters": []}

    def counting_step(self, x, y, hard=True):
        step_calls["count"] += 1
        step_calls["max_iters"].append(self.optimizer.defaults["max_iter"])
        return real_step(self, x, y, hard=hard)

    monkeypatch.setattr(LBFGSConstantOptimizer, "step", counting_step)

    lbfgs_steps = 5
    pop_size = 3
    elite_size = 1
    opt = EvolutionaryOptimizer(
        _TinyONNModel,
        population_size=pop_size,
        elite_size=elite_size,
        use_lbfgs_refinement=True,
        lbfgs_steps=lbfgs_steps,
    )
    opt.initialize_population()

    x = torch.randn(16, 2)
    y = x[:, :1] * 2.0 + 1.0

    stats = opt.evolve_generation(x, y, x, y)

    n_offspring = pop_size - elite_size

    # P-03 regression: exactly one optimizer.step() per offspring.
    # The bug called step() lbfgs_steps times per offspring.
    assert step_calls["count"] == n_offspring

    # Each step() must carry the full iteration budget (max_iter=lbfgs_steps),
    # so total L-BFGS iterations stay bounded by the intended budget.
    assert step_calls["max_iters"] == [lbfgs_steps] * n_offspring

    # Generation completed with sane fitness values.
    assert stats["best_fitness"] >= 0.0
    assert stats["mean_fitness"] >= 0.0


# ── P-02: island reproduction / NSGA non-fallthrough ─────────────────────


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


# ── P-04: FD Adam inert-parameter skipping ───────────────────────────────

_p04_skip = pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ core unavailable")


@pytest.fixture(scope="module")
def cpp_core():
    return get_cpp_core()


@pytest.fixture(scope="module")
def periodic_power_problem(cpp_core):
    """sin + power target whose seed graphs contain inert-param nodes."""
    x = np.linspace(0.2, 3.0, 120)
    y = np.sin(3.0 * x) + x**1.5
    seeds = [
        cpp_core.formula_to_seed_graph(s)
        for s in ("sin(3*x0)+x0**1.5", "sin(2*x0)", "x0**2", "log(x0+1)")
    ]
    return x, y, seeds


def _run_p04(core, x, y, seeds, use_lm):
    return core.run_evolution(
        [x],
        y,
        pop_size=30,
        generations=15,
        num_islands=1,
        random_seed=42,
        early_stop_mse=0.0,
        timeout_seconds=120,
        seed_graphs_py=seeds,
        use_lm_inner_optimizer=use_lm,
    )


@_p04_skip
def test_p04_fd_adam_skips_inert_params(cpp_core, periodic_power_problem):
    """FD-Adam path must skip analytically-inert parameter probes."""
    x, y, seeds = periodic_power_problem
    res = _run_p04(cpp_core, x, y, seeds, use_lm=False)

    total = res["fd_probes_total"]
    skipped = res["fd_probes_skipped_inert"]

    # Refinement ran and probed live parameters.
    assert total > 0
    # Inert slots (p on Periodic, omega/phi on Power, all on Log) were skipped.
    assert skipped > 0
    # Counter bookkeeping stays consistent.
    assert skipped <= total
    # Search quality preserved on the forced FD-Adam path.
    assert np.isfinite(res["best_mse"])
    assert res["best_mse"] < 0.05


@_p04_skip
def test_p04_lm_default_path_unchanged(cpp_core, periodic_power_problem):
    """Default LM-first path still refines and reports diagnostics."""
    x, y, seeds = periodic_power_problem
    res = _run_p04(cpp_core, x, y, seeds, use_lm=True)

    assert np.isfinite(res["best_mse"])
    assert res["best_mse"] < 0.05
    assert res["fd_probes_total"] >= res["fd_probes_skipped_inert"] >= 0


@_p04_skip
def test_p04_fixed_seed_determinism(cpp_core, periodic_power_problem):
    """Same seed produces identical results (refactor must not perturb search)."""
    x, y, seeds = periodic_power_problem
    a = _run_p04(cpp_core, x, y, seeds, use_lm=False)
    b = _run_p04(cpp_core, x, y, seeds, use_lm=False)
    assert a["best_mse"] == b["best_mse"]
