"""Reliability regression tests for evolution fitness handling."""

import math

import torch
import torch.nn as nn

import glassbox.sr.evolution as evo


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_proj = nn.Linear(1, 1)

    def forward(self, x, hard=True):
        return self.output_proj(x), None

    def entropy_regularization(self):
        return torch.tensor(float("nan"), device=self.output_proj.weight.device)


class _SensitivityDummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.high_logit = nn.Parameter(torch.zeros(4))
        self.low_logit = nn.Parameter(torch.zeros(4))

    def forward(self, x, hard=True):
        return torch.zeros(x.shape[0], 1, device=x.device), None


def _make_trainer(nested_bfgs: bool) -> evo.EvolutionaryONNTrainer:
    trainer = evo.EvolutionaryONNTrainer(
        model_factory=_DummyModel,
        population_size=1,
        elite_size=1,
        use_explorers=False,
        nested_bfgs=nested_bfgs,
        nested_bfgs_steps=2,
        progressive_round_weight=0.1,
        device=torch.device("cpu"),
    )
    trainer.population = [evo.Individual(_DummyModel())]
    trainer.explorers = []
    return trainer


def test_nonfinite_nested_refine_uses_fallback(monkeypatch):
    trainer = _make_trainer(nested_bfgs=True)
    fallback_calls = {"count": 0}

    monkeypatch.setattr(evo, "quick_refine_internal", lambda *args, **kwargs: float("nan"))

    def _fallback_refine(*args, **kwargs):
        fallback_calls["count"] += 1
        return 0.0

    monkeypatch.setattr(evo, "refine_constants", _fallback_refine)
    monkeypatch.setattr(evo, "calculate_complexity", lambda model: 1.0)
    monkeypatch.setattr(evo, "coefficient_sparsity_loss", lambda model: torch.tensor(0.0))
    monkeypatch.setattr(evo, "progressive_round_loss", lambda model: torch.tensor(0.0))

    x = torch.linspace(-1.0, 1.0, 32).unsqueeze(-1)
    y = x.squeeze() ** 2

    trainer.evaluate_fitness(x, y, generation=0, total_generations=1)

    assert fallback_calls["count"] == 1
    assert math.isfinite(trainer.population[0].fitness)


def test_nonfinite_penalty_terms_do_not_poison_fitness(monkeypatch):
    trainer = _make_trainer(nested_bfgs=False)

    monkeypatch.setattr(evo, "calculate_complexity", lambda model: 1.0)
    monkeypatch.setattr(evo, "coefficient_sparsity_loss", lambda model: torch.tensor(float("nan")))
    monkeypatch.setattr(evo, "progressive_round_loss", lambda model: torch.tensor(float("inf")))

    x = torch.linspace(-1.0, 1.0, 32).unsqueeze(-1)
    y = x.squeeze() ** 2

    trainer.evaluate_fitness(x, y, generation=0, total_generations=1)

    assert math.isfinite(trainer.population[0].fitness)


def test_gradient_informed_mutation_protects_sensitive_logits(monkeypatch):
    individual = evo.Individual(_SensitivityDummyModel())

    monkeypatch.setattr(
        evo,
        "compute_param_sensitivity",
        lambda model, x, y: {"high_logit": 10.0, "low_logit": 0.1},
    )
    monkeypatch.setattr(evo.random, "random", lambda: 0.0)
    monkeypatch.setattr(evo.random, "randint", lambda a, b: 0)
    monkeypatch.setattr(evo.torch, "randn_like", lambda tensor: torch.ones_like(tensor))

    x = torch.linspace(-1.0, 1.0, 8).unsqueeze(-1)
    y = torch.zeros(8)

    mutant = evo.mutate_operations_gradient_informed(
        individual,
        x,
        y,
        mutation_rate=1.0,
        sensitivity_bias=1.0,
    )

    high_delta = torch.sum(torch.abs(mutant.model.high_logit - individual.model.high_logit)).item()
    low_delta = torch.sum(torch.abs(mutant.model.low_logit - individual.model.low_logit)).item()

    assert high_delta == 0.0
    assert low_delta > 0.0
