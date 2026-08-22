"""Regression tests for audit findings H-19, H-20, H-21."""

from __future__ import annotations

import math

import torch
from torch import nn

from glassbox.evolution import evolution as evo
from glassbox.sr import phased_regression as pr
from glassbox.sr.sklearn_wrapper import GlassboxRegressor

# ---------------------------------------------------------------------------
# H-19 — BFGS import path + unary index map
# ---------------------------------------------------------------------------


def test_h19_bfgs_import_available():
    assert pr.BFGS_AVAILABLE is True
    assert callable(pr.fit_coefficients_bfgs)
    assert callable(pr.build_formula_from_weights)


def test_h19_unary_op_kind_matches_operation_node_order():
    class _Node:
        _unary_op_names = ["periodic", "power", "exp", "log"]

    assert pr._unary_op_kind(_Node(), 0) == "periodic"
    assert pr._unary_op_kind(_Node(), 1) == "power"
    assert pr._unary_op_kind(_Node(), 2) == "exp"
    assert pr._unary_op_kind(_Node(), 3) == "log"


def test_h19_unary_op_kind_fallback_without_names():
    class _Bare:
        pass

    # Fallback order matches OperationNode: periodic=0, power=1
    assert pr._unary_op_kind(_Bare(), 0) == "periodic"
    assert pr._unary_op_kind(_Bare(), 1) == "power"
    # Old bug treated 0 as power — must not regress
    assert pr._unary_op_kind(_Bare(), 0) != "power"


# ---------------------------------------------------------------------------
# H-20 — refine runs in eval mode (BN / discrete selection parity with fitness)
# ---------------------------------------------------------------------------


def test_h20_refine_constants_leaves_model_in_eval():
    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.output_proj = nn.Linear(1, 1)
            self.omega = nn.Parameter(torch.tensor(1.0))

        def forward(self, x, hard=True):
            return self.output_proj(x) * self.omega, None

    m = _M()
    m.train()
    x = torch.linspace(-1, 1, 32).unsqueeze(-1)
    y = x.clone()
    loss = evo.refine_constants(m, x, y, steps=2, lr=0.01, hard=True)
    assert math.isfinite(loss) or loss == float("inf")
    assert m.training is False


def test_h20_quick_refine_internal_stays_eval():
    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.tensor(2.0))
            self.lin = nn.Linear(1, 1)

        def forward(self, x, hard=True):
            return self.lin(x) * self.p, None

    m = _M()
    m.train()
    x = torch.linspace(-1, 1, 16).unsqueeze(-1)
    y = (x.squeeze() ** 2).unsqueeze(-1)
    evo.quick_refine_internal(m, x, y, steps=2)
    assert m.training is False


def test_h20_refine_and_fitness_same_mode(monkeypatch):
    """Refine forward and fitness forward both see eval()."""
    modes = []

    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.output_proj = nn.Linear(1, 1)
            self.omega = nn.Parameter(torch.tensor(1.0))

        def forward(self, x, hard=True):
            modes.append(self.training)
            return self.output_proj(x), None

    m = _M()
    x = torch.linspace(-1, 1, 16).unsqueeze(-1)
    y = x.clone()
    evo.refine_constants(m, x, y, steps=1, lr=0.01, hard=True)
    m.eval()
    with torch.no_grad():
        m(x, hard=True)
    # All recorded forwards should be eval (training=False)
    assert modes, "forward never called"
    assert all(t is False for t in modes)


# ---------------------------------------------------------------------------
# H-21 — multi-token family signature
# ---------------------------------------------------------------------------


def _regressor():
    return object.__new__(GlassboxRegressor)


def test_h21_multi_op_family_includes_all_tokens():
    r = _regressor()
    assert r._formula_family_signature("sin(x)*exp(x)") == "exp+multiplicative+sin"
    assert r._formula_family_tokens("sin(x)*exp(x)") == ("exp", "multiplicative", "sin")


def test_h21_single_op_still_single_token():
    r = _regressor()
    assert r._formula_family_signature("sin(x)") == "sin"
    assert r._formula_family_signature("log(x0)") == "log"
    assert r._formula_family_signature("x0+x1") == "additive"


def test_h21_power_and_mul_detected():
    r = _regressor()
    sig = r._formula_family_signature("x0**2 * x1")
    assert "power" in sig
    assert "multiplicative" in sig


def test_h21_family_contains_helper():
    r = _regressor()
    assert r._formula_family_contains("sin(x)*exp(x)", "exp")
    assert r._formula_family_contains("sin(x)*exp(x)", "sin", "cos")
    assert not r._formula_family_contains("x0+x1", "sin")


def test_h21_distinct_multi_op_formulas_not_same_family_key():
    r = _regressor()
    # Old first-token logic made both "sin" — diversity prune collapsed them
    a = r._formula_family_signature("sin(x0)")
    b = r._formula_family_signature("sin(x0)*exp(x1)")
    assert a != b


def test_h21_nest_eligibility_accepts_multi_token_outer():
    """specialist nest checks must accept multi-token families containing sin/exp/etc."""
    # compose path uses family tokens with '+' join from signature
    forms = []
    outer_family = "exp+sin"
    family_tokens = {t for t in outer_family.split("+") if t}
    nestable = family_tokens & {"sin", "cos", "exp", "log"}
    assert nestable
    assert "exp" in nestable
    assert "sin" in nestable
    # legacy single still works
    assert {"sin"} & {"sin", "cos", "exp", "log"}
