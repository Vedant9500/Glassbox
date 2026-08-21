"""Regression tests for audit finding P-03 (hybrid L-BFGS steps squared).

P-03: ``EvolutionaryOptimizer.evolve_generation`` constructed
``LBFGSConstantOptimizer(child.model, max_iter=lbfgs_steps)`` and then called
``step()`` inside a ``for _ in range(lbfgs_steps)`` loop. Each
``optimizer.step(closure)`` already runs up to ``max_iter`` internal L-BFGS
iterations, so the loop cost up to ``steps**2`` line searches for the same
total iteration budget.

The fix: a single ``optimizer.step()`` per offspring with
``max_iter=lbfgs_steps``.
"""

import torch
import pytest

from glassbox.sr.optimizers import EvolutionaryOptimizer, LBFGSConstantOptimizer


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
