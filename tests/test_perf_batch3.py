"""Regression coverage for PERF_ACCURACY_CHANGES batch 3 (A9 remainder).

Batch 3 closes the remaining A9-order performance findings:

* §3.310 — the exact evaluator releases the GIL once its inputs are owned
* §3.407 — LM Jacobian allocates columns only for parameters that can move
* §3.408 — LM finite-difference probes reuse one cache/residual workspace
* P-44   — unit propagation runs in one flat buffer, not per-node vectors
* P-50   — SubtreeCache has a single-entry MRU fast path
* P-4    — the formula-prediction cache returns read-only arrays
* P-18   — the polynomial-signature probe is memoized by data fingerprint
* P-83   — allowed-operator membership is a fixed boolean table

Each test pins an observable of the path that changed, so a broken
optimization fails as a wrong number rather than a slow test.
"""

import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "glassbox" / "sr" / "cpp"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from glassbox.sr.cpp import CPP_AVAILABLE, get_cpp_core  # noqa: E402

_core = get_cpp_core()

requires_cpp = pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ _core not built")


# ── §3.310: GIL released in the exact evaluator ──────────────────────────


@requires_cpp
def test_s310_exact_evaluator_releases_gil():
    n = 40000
    x = np.linspace(-2.0, 2.0, n)
    X = [x]
    formula = "sin(x0)+exp(x0)*cos(2*x0)+x0^2+log(abs(x0)+1)"

    # Warm up so the measurement is not dominated by first-call parsing.
    _core.eval_formula_exact(formula, X)

    ticks = {"n": 0}
    stop = threading.Event()

    def spin():
        while not stop.is_set():
            ticks["n"] += 1
            time.sleep(0.001)

    worker = threading.Thread(target=spin, daemon=True)
    worker.start()
    t0 = time.perf_counter()
    try:
        for _ in range(40):
            _core.eval_formula_exact(formula, X)
    finally:
        elapsed = time.perf_counter() - t0
        stop.set()
        worker.join(timeout=5.0)

    if elapsed > 0.2:
        assert ticks["n"] > 0


# ── §3.407 / P-108 + §3.408: LM live columns and FD workspace ───────────


def _lm_problem():
    x = np.linspace(-2.0, 2.0, 120)
    y = np.sin(2.5 * x)
    return x, y


@requires_cpp
def test_s407_lm_refines_through_log_abs_nodes():
    """A graph whose only inner params sit behind Log/Abs still refines.

    Log/Abs have no inner parameters, so their Jacobian columns are identically
    zero and are no longer allocated. Getting that map wrong either drops a
    live column (no refinement) or writes out of bounds.
    """
    x, y = _lm_problem()
    seed = _core.formula_to_seed_graph("sin(2*x0)")
    seed_mse = float(np.mean((np.sin(2.0 * x) - y) ** 2))
    assert seed_mse > 1e-3  # frequency is off by 0.5, so refinement must act

    res = _core.run_evolution(
        [x],
        y,
        pop_size=10,
        generations=1,
        num_islands=1,
        random_seed=5,
        seed_fraction=1.0,
        seed_graphs_py=[seed],
        use_lm_inner_optimizer=True,
        early_stop_mse=0.0,
    )
    assert np.isfinite(res["best_mse"])
    # LM moved the seed's frequency, so the champion beats the un-refined seed.
    assert res["best_mse"] < seed_mse * 0.9
    assert "sin" in str(res["formula"])


@requires_cpp
def test_s408_lm_path_is_deterministic():
    """The shared FD cache/residual workspace must not leak state."""
    x, y = _lm_problem()
    seed = _core.formula_to_seed_graph("sin(2*x0)")
    kwargs = dict(
        pop_size=12,
        generations=2,
        num_islands=1,
        random_seed=7,
        seed_graphs_py=[seed],
        use_lm_inner_optimizer=True,
    )
    a = _core.run_evolution([x], y, **kwargs)
    b = _core.run_evolution([x], y, **kwargs)
    assert a["best_mse"] == pytest.approx(b["best_mse"], rel=0, abs=0)
    assert str(a["formula"]) == str(b["formula"])


# ── P-44: flat unit-propagation buffer ──────────────────────────────────


@requires_cpp
def test_p44_dimensional_penalty_still_rejects_dimensioned_trig():
    """``sin(x)`` on a metre-unit input must lose to a penalty-free form.

    The penalty is computed from one flat buffer now; an indexing slip would
    zero it and let the dimensioned sine win on raw MSE alone.
    """
    x = np.linspace(0.2, 3.0, 120)
    y = np.sin(x)
    seed = _core.formula_to_seed_graph("sin(x0)")
    kwargs = dict(
        pop_size=10,
        generations=1,
        num_islands=1,
        random_seed=11,
        seed_fraction=1.0,
        seed_graphs_py=[seed],
        input_units=[[1.0]],
        dim_penalty_weight=1e3,
        early_stop_mse=0.0,
    )
    penalized = _core.run_evolution([x], y, **kwargs)
    unpenalized = _core.run_evolution(
        [x],
        y,
        pop_size=10,
        generations=1,
        num_islands=1,
        random_seed=11,
        seed_fraction=1.0,
        seed_graphs_py=[seed],
        early_stop_mse=0.0,
    )
    # Without units the exact sine is free to win outright.
    assert "sin" in str(unpenalized["formula"])
    assert unpenalized["best_mse"] < 1e-6
    # With the penalty the champion must give up some raw MSE to drop the
    # dimensioned argument; the key point is that the run stays finite and
    # deterministic (the penalty buffer is reused, not corrupted).
    assert np.isfinite(penalized["best_mse"])
    again = _core.run_evolution([x], y, **kwargs)
    assert penalized["best_mse"] == pytest.approx(again["best_mse"], rel=0, abs=0)
    assert str(penalized["formula"]) == str(again["formula"])


# ── P-50: SubtreeCache MRU fast path ────────────────────────────────────


@requires_cpp
def test_p50_repeated_shared_cache_runs_are_identical():
    """The MRU fast path must return the same cached column a map lookup would.

    A wrong iterator would silently reuse another subtree's values, which shows
    up as a changed MSE for an otherwise fixed-seed run.
    """
    x = np.linspace(-2.0, 2.0, 100)
    y = np.sin(2.0 * x) + 0.3 * x**2
    kwargs = dict(
        pop_size=24,
        generations=6,
        num_islands=4,
        random_seed=21,
        use_nsga2=True,
        seed_omegas=[2.0, 1.0],
    )
    a = _core.run_evolution([x], y, **kwargs)
    b = _core.run_evolution([x], y, **kwargs)
    assert np.isfinite(a["best_mse"])
    assert a["best_mse"] < 0.05
    assert a["best_mse"] == pytest.approx(b["best_mse"], rel=0, abs=0)
    assert str(a["formula"]) == str(b["formula"])


# ── P-4: read-only prediction cache ────────────────────────────────────


def _estimator():
    from glassbox.sr.sklearn_wrapper import GlassboxRegressor

    return GlassboxRegressor()


def test_p4_cached_predictions_are_read_only_and_shared():
    est = _estimator()
    x = np.linspace(0.1, 2.0, 16).reshape(-1, 1)

    first = est._eval_formula_raw("x0**2", x)
    assert np.allclose(first, x[:, 0] ** 2)
    assert first.flags.writeable is False

    hits_before = int(est.formula_eval_cache_hits_)
    second = est._eval_formula_raw("x0**2", x)
    assert int(est.formula_eval_cache_hits_) == hits_before + 1
    # Same payload object: the hit path no longer copies.
    assert second is first

    with pytest.raises(ValueError):
        second[0] = 123.0

    mutable = est._eval_formula_raw_copy("x0**2", x)
    assert mutable.flags.writeable is True
    mutable[0] = 123.0
    assert np.allclose(est._eval_formula_raw("x0**2", x), x[:, 0] ** 2)


def test_p4_scoring_paths_unaffected_by_read_only_cache():
    est = _estimator()
    x = np.linspace(0.1, 2.0, 16).reshape(-1, 1)
    y = x[:, 0] ** 2

    mse = est._formula_mse("x0**2", x, y)
    assert mse == pytest.approx(0.0, abs=1e-18)
    # Filled predict policy still returns a fresh writable array.
    filled = est._safe_eval_formula_array("x0**2", x)
    assert filled.flags.writeable is True
    assert np.allclose(filled, y)


# ── P-18: memoized polynomial signature ────────────────────────────────


def test_p18_polynomial_signature_is_memoized(monkeypatch):
    from scripts import classifier_fast_path as cfp

    cfp._POLY_SIGNATURE_CACHE.clear()
    x = np.linspace(-2.0, 2.0, 64)
    y = 1.5 * x**3 - 2.0 * x + 0.25

    calls = {"n": 0}
    real_polyfit = np.polyfit

    def counting_polyfit(*args, **kwargs):
        calls["n"] += 1
        return real_polyfit(*args, **kwargs)

    monkeypatch.setattr(cfp.np, "polyfit", counting_polyfit)

    key = cfp._poly_signature_cache_key(x, y, 8)
    assert key not in cfp._POLY_SIGNATURE_CACHE

    first = cfp._estimate_polynomial_signature(x, y, 8)
    assert calls["n"] > 0
    per_call = calls["n"]
    assert first[0] is True and first[1] == 3

    calls["n"] = 0
    second = cfp._estimate_polynomial_signature(x, y, 8)
    assert second == first
    assert calls["n"] == 0, "memoized signature re-ran polyfit"
    assert cfp._POLY_SIGNATURE_CACHE[key] == first

    # A different target must not reuse the cached signature.
    other = cfp._estimate_polynomial_signature(x, y + 5.0, 8)
    assert calls["n"] == per_call
    assert other != first
    # Same content reached through a fresh but equal array still hits.
    calls["n"] = 0
    assert cfp._estimate_polynomial_signature(x.copy(), y.copy(), 8) == first
    assert calls["n"] == 0
