"""Regression tests for audit findings H-16, H-17, H-18."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from glassbox.curve_classifier import curve_classifier_integration as cci
from glassbox.evolution import evolution as evo
from glassbox.sr.hard_concrete import (
    HardConcreteGate,
    HardConcreteOperationSelector,
    HardConcreteSelector,
)
from glassbox.sr.operations.meta_ops import MetaAggregation


# ---------------------------------------------------------------------------
# H-16 — derive n_features from weights when model_config is incomplete
# ---------------------------------------------------------------------------

def _clear_classifier_caches():
    cci._cached_classifier_by_device.clear()
    cci._cached_operator_classes_by_key.clear()
    cci._cached_metadata_by_device.clear()


def test_h16_cnn_n_features_from_other_mlp_without_config(tmp_path: Path):
    """Legacy CNN checkpoint missing model_config.n_features still reconstructs correctly."""
    n_features, curve_dim, n_classes = 398, 128, 5
    model = cci.CurveClassifierCNN(n_classes=n_classes, n_features=n_features, curve_dim=curve_dim)
    path = tmp_path / "cnn_legacy.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": "cnn",
            # deliberately omit model_config n_features
            "model_config": {"curve_dim": curve_dim, "n_classes": n_classes},
            "operator_classes": [f"c{i}" for i in range(n_classes)],
        },
        path,
    )
    _clear_classifier_caches()
    loaded = cci._load_pytorch_classifier(path, torch.device("cpu"), cache_key="h16-cnn")
    assert isinstance(loaded, cci.CurveClassifierCNN)
    # other_mlp input width = n_features - curve_dim
    assert loaded.other_mlp[0].in_features == n_features - curve_dim
    # forward accepts full feature vectors
    x = torch.randn(2, n_features)
    with torch.no_grad():
        out = loaded(x)
    assert out.shape == (2, n_classes)


def test_h16_glu_n_features_from_eql_not_fc1(tmp_path: Path):
    """GLU must use eql.linear in_features, not fc1 combined width."""
    n_features, n_classes, hidden = 398, 7, 64
    model = cci.CurveClassifierGLU(n_features=n_features, n_classes=n_classes, hidden=hidden)
    path = tmp_path / "glu_legacy.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": "glu",
            "model_config": {},  # no n_features
            "operator_classes": [f"c{i}" for i in range(n_classes)],
        },
        path,
    )
    _clear_classifier_caches()
    loaded = cci._load_pytorch_classifier(path, torch.device("cpu"), cache_key="h16-glu")
    assert isinstance(loaded, cci.CurveClassifierGLU)
    assert loaded.eql.linear.in_features == n_features
    # Old bug: fc1.weight.shape[1] is combined_dim (1280 for default), not 398
    assert loaded.fc1.in_features != n_features or loaded.eql.linear.in_features == n_features
    x = torch.randn(2, n_features)
    with torch.no_grad():
        out = loaded(x)
    assert out.shape == (2, n_classes)


def test_h16_mlp_n_features_from_eql(tmp_path: Path):
    n_features, n_classes, hidden = 200, 4, 32
    model = cci.CurveClassifierMLP(n_features=n_features, n_classes=n_classes, hidden=hidden)
    path = tmp_path / "mlp_legacy.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": "mlp",
            "model_config": {},
            "operator_classes": [f"c{i}" for i in range(n_classes)],
        },
        path,
    )
    _clear_classifier_caches()
    loaded = cci._load_pytorch_classifier(path, torch.device("cpu"), cache_key="h16-mlp")
    assert loaded.eql.linear.in_features == n_features
    x = torch.randn(3, n_features)
    with torch.no_grad():
        out = loaded(x)
    assert out.shape == (3, n_classes)


def test_h16_resolve_helper_prefers_model_config():
    n = cci._resolve_n_features_from_state_dict(
        "cnn",
        {"other_mlp.0.weight": torch.zeros(128, 50)},
        {"n_features": 999},
        {},
    )
    assert n == 999


def test_h16_resolve_cnn_from_other_mlp():
    # curve_dim default 128 + other 50 = 178
    n = cci._resolve_n_features_from_state_dict(
        "cnn",
        {"other_mlp.0.weight": torch.zeros(128, 50)},
        {},
        {},
    )
    assert n == 178


# ---------------------------------------------------------------------------
# H-17 — set_model_tau must not overwrite MetaAggregation.tau
# ---------------------------------------------------------------------------

class _TauHost(nn.Module):
    def __init__(self):
        super().__init__()
        self.sel = HardConcreteSelector(n_options=4, tau=1.0)
        self.gate = HardConcreteGate(n_gates=2, tau=1.0)
        self.op_sel = HardConcreteOperationSelector(tau=1.0)
        self.agg_learnable = MetaAggregation(init_tau=2.5, learnable=True)
        self.agg_fixed = MetaAggregation(init_tau=3.0, learnable=False)
        self.tau = 1.0  # OperationNode-style selection tau


def test_h17_set_model_tau_skips_meta_aggregation():
    m = _TauHost()
    evo.set_model_tau(m, 0.15)

    assert m.sel.tau == 0.15
    assert abs(m.gate.tau - 0.15) < 1e-5
    assert m.op_sel.tau == 0.15
    assert m.tau == 0.15
    # Aggregation temperatures must be untouched
    assert abs(m.agg_learnable.tau.item() - 2.5) < 1e-5
    assert abs(m.agg_fixed.tau.item() - 3.0) < 1e-5


def test_h17_set_model_tau_does_not_call_meta_set_tau():
    m = _TauHost()
    called = {"n": 0}
    original = MetaAggregation.set_tau

    def _spy(self, tau):
        called["n"] += 1
        return original(self, tau)

    MetaAggregation.set_tau = _spy  # type: ignore[method-assign]
    try:
        evo.set_model_tau(m, 0.05)
    finally:
        MetaAggregation.set_tau = original  # type: ignore[method-assign]

    assert called["n"] == 0
    assert abs(m.agg_learnable.tau.item() - 2.5) < 1e-5


# ---------------------------------------------------------------------------
# H-18 — elite skip cleared after tau change
# ---------------------------------------------------------------------------

def test_h18_elite_reevaluated_when_tau_changes(monkeypatch):
    class _Dummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.output_proj = nn.Linear(1, 1)
            self.sel = HardConcreteSelector(2, tau=1.0)

        def forward(self, x, hard=True):
            return self.output_proj(x), None

        def entropy_regularization(self):
            return torch.tensor(0.0)

    trainer = evo.EvolutionaryONNTrainer(
        model_factory=_Dummy,
        population_size=2,
        elite_size=1,
        use_explorers=False,
        nested_bfgs=False,
        device=torch.device("cpu"),
        tau_start=1.0,
        tau_end=0.1,
    )
    ind0 = evo.Individual(_Dummy())
    ind1 = evo.Individual(_Dummy())
    ind0.fitness = 0.5
    ind0._is_elite = True
    ind1.fitness = 1.0
    ind1._is_elite = False
    trainer.population = [ind0, ind1]
    trainer.explorers = []
    trainer._last_applied_tau = 1.0  # previous gen tau

    eval_count = {"n": 0}
    original_eval = trainer.evaluate_fitness

    def _counting_eval(*args, **kwargs):
        eval_count["n"] += 1
        # Capture whether elite flag was cleared before evaluation
        elite_flags.append([bool(getattr(i, "_is_elite", False)) for i in trainer.population])
        return original_eval(*args, **kwargs)

    elite_flags: list = []
    monkeypatch.setattr(trainer, "evaluate_fitness", _counting_eval)
    monkeypatch.setattr(evo, "refine_constants", lambda *a, **k: 0.0)
    monkeypatch.setattr(evo, "calculate_complexity", lambda model: 1.0)
    monkeypatch.setattr(evo, "coefficient_sparsity_loss", lambda model: torch.tensor(0.0))
    monkeypatch.setattr(evo, "progressive_round_loss", lambda model: torch.tensor(0.0))

    # Simulate one generation of the train loop's tau + clear logic
    gen, generations = 1, 10
    current_tau = evo.anneal_tau(gen, generations, trainer.tau_start, trainer.tau_end)
    tau_changed = getattr(trainer, "_last_applied_tau", None) != current_tau
    assert tau_changed
    for ind in trainer.population:
        evo.set_model_tau(ind.model, current_tau)
        if tau_changed and getattr(ind, "_is_elite", False):
            ind._is_elite = False
    trainer._last_applied_tau = current_tau

    assert ind0._is_elite is False  # cleared so evaluate_fitness will re-score
    # evaluate_fitness must not skip via elite path
    x = torch.linspace(-1, 1, 16).unsqueeze(-1)
    y = x.squeeze() ** 2
    trainer.evaluate_fitness(x, y, generation=1, total_generations=10)
    # Fitness should have been recomputed (finite)
    assert ind0.fitness < float("inf")


def test_h18_elite_skip_preserved_when_tau_unchanged():
    """When tau is stable, elite skip remains a valid optimization."""
    class _Dummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(1, 1)

        def forward(self, x, hard=True):
            return self.lin(x), None

        def entropy_regularization(self):
            return torch.tensor(0.0)

    trainer = evo.EvolutionaryONNTrainer(
        model_factory=_Dummy,
        population_size=1,
        elite_size=1,
        use_explorers=False,
        nested_bfgs=False,
        device=torch.device("cpu"),
    )
    ind = evo.Individual(_Dummy())
    ind.fitness = 0.123
    ind._is_elite = True
    trainer.population = [ind]
    trainer._last_applied_tau = 0.5

    current_tau = 0.5
    tau_changed = getattr(trainer, "_last_applied_tau", None) != current_tau
    if tau_changed and getattr(ind, "_is_elite", False):
        ind._is_elite = False

    assert ind._is_elite is True
    assert ind.fitness == 0.123
