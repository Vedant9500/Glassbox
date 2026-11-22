"""Regression coverage for PERF_ACCURACY_CHANGES batch 2.

Batch 2 closes ten A9 performance findings:

* P-42  — analytic frequency derivatives in ``refine_frequencies``
* P-55  — Torch exact-match search over r x r Gram systems
* P-56  — NumPy exact-match scan over the same Gram layout
* P-76  — rank-one residual scoring for output-weight/bias snapping
* P-89  — owned FFI inputs + GIL release around refinement / elastic net
* P-90  — iterative, arena-backed exact evaluator (same item as §3.406)
* P-94  — settled soft-Arithmetic gate takes only its dominant branch
* P-99  — incremental IRLS Gramian updates
* P-100 — LM setup reuse across unchanged outer iterations

Each test pins observable behaviour of the path that changed, so a broken
optimization shows up as a wrong number rather than a slow test.
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


# ── P-90 / §3.406: exact evaluator arena ─────────────────────────────────


@requires_cpp
def test_p90_exact_evaluator_matches_reference():
    rng = np.random.default_rng(0)
    x0 = np.linspace(-2.0, 2.0, 64)
    x1 = rng.standard_normal(64)
    X = [x0, x1]

    cases = {
        # plain arithmetic / precedence
        "x0+2*x1": x0 + 2.0 * x1,
        "sin(x0)*exp(x1)": np.sin(x0) * np.exp(x1),
        # protected division: l * sign(r) / (|r| + 1e-12)
        "1/(x0+0.5)": 1.0 / (np.abs(x0 + 0.5) + 1e-12) * np.sign(x0 + 0.5),
        # non-integer power on a non-negative base. §3.1/§3.331: the exact
        # path shares the graph power contract, including eps=1e-10.
        "abs(x0)^2.5": (np.abs(x0) + 1e-10) ** 2.5,
        # nested unary chain
        "log(abs(x0)+1)": np.log(np.abs(x0) + 1.0 + 1e-300),
        # repeated node (shared-subexpression path)
        "x0+x0": 2.0 * x0,
    }
    for formula, expected in cases.items():
        got = np.asarray(_core.eval_formula_exact(formula, X), dtype=np.float64)
        assert got.shape == expected.shape
        assert np.allclose(got, expected, rtol=1e-12, atol=1e-12), formula

    # Deep nesting exercises the explicit post-order stack (the old code
    # recursed, so this also pins the non-recursive rewrite).
    depth = 24
    nested = "sin(" * depth + "x0" + ")" * depth
    expected = x0
    for _ in range(depth):
        expected = np.sin(expected)
    got = np.asarray(_core.eval_formula_exact(nested, X), dtype=np.float64)
    assert np.allclose(got, expected, rtol=1e-12, atol=1e-12)


@requires_cpp
def test_p90_exact_evaluator_arena_reuse_is_stateless():
    x = np.linspace(-1.0, 1.0, 41)
    X = [x]
    first = np.asarray(_core.eval_formula_exact("exp(x0)*cos(x0)", X))
    # Interleave other shapes/formulas so any stale arena slot would leak.
    _core.eval_formula_exact("x0^3+sin(x0)", [np.linspace(0.0, 3.0, 7)])
    _core.eval_formula_exact("log(abs(x0)+1)", X)
    again = np.asarray(_core.eval_formula_exact("exp(x0)*cos(x0)", X))
    assert again.shape == first.shape
    assert np.array_equal(again, first)


# ── P-94: settled soft-Arithmetic gate ───────────────────────────────────


@requires_cpp
def test_p94_candidate_scoring_uses_settled_gate_branch():
    """``score_formula_candidates`` evaluates at temperature 100, where a
    clean x0+x1 gate has one exactly-unit softmax weight (P-94's branch).
    The score must still equal the explicit affine fit of x0+x1."""
    rng = np.random.default_rng(3)
    n = 80
    x0 = np.linspace(-2.0, 2.0, n)
    x1 = rng.standard_normal(n)
    y = 3.0 * (x0 + x1) - 0.7 + 0.01 * rng.standard_normal(n)
    split = n // 2
    X = np.column_stack([x0, x1])
    scores = _core.score_formula_candidates(
        ["x0+x1", "sin(x0)*x1"],
        X[:split],
        y[:split],
        X[split:],
        y[split:],
    )
    score = scores[0]
    assert score["ok"] is True

    pred_fit = x0[:split] + x1[:split]
    pred_val = x0[split:] + x1[split:]
    # Mirror the native affine scale/bias fit on the fit split.
    dx = pred_fit - pred_fit.mean()
    dy = y[:split] - y[:split].mean()
    scale = float((dx * dy).sum() / (dx * dx).sum())
    bias = float(y[:split].mean() - scale * pred_fit.mean())
    expected_val_mse = float(np.mean((scale * pred_val + bias - y[split:]) ** 2))
    assert score["mse"] == pytest.approx(expected_val_mse, rel=1e-10, abs=1e-12)


@requires_cpp
def test_p94_high_temperature_engine_run_is_finite():
    """At temperature 100 every resolved Arithmetic gate takes the P-94
    branch; the engine must still evaluate and rank without NaN/aborts."""
    x = np.linspace(-1.5, 1.5, 60)
    y = x + np.sin(2.0 * x)
    res = _core.run_evolution(
        [x],
        y,
        pop_size=24,
        generations=4,
        num_islands=1,
        random_seed=7,
        arithmetic_temperature=100.0,
    )
    assert np.isfinite(res["best_mse"])
    assert res["effective_arithmetic_temperature"] == pytest.approx(100.0)


# ── P-42: analytic frequency refinement ──────────────────────────────────


@requires_cpp
def test_p42_refine_frequencies_recovers_clean_sinusoid():
    x = np.linspace(-2.0, 2.0, 160)
    y = 0.4 + 0.8 * x + 1.3 * np.sin(2.5 * x)
    # lr=0.2 is the step size at which this projected-GD schedule actually
    # reaches the optimum (the routine's convergence rate, not the gradient's
    # definition, sets this); the tolerance then pins an exact fixpoint.
    omegas, mse = _core.refine_frequencies(x, y, [1.0], steps=120, lr=0.2)
    assert np.isfinite(mse)
    assert abs(float(omegas[0]) - 2.5) < 1e-3, omegas
    assert mse < 1e-9


@requires_cpp
def test_p42_analytic_gradient_matches_one_native_step():
    """One native GD step must equal a NumPy replay of the analytic gradient.

    ``refine_frequencies`` returns the best *evaluated* omega, so ``steps=2``
    returns the position one step away from the start (the step improves the
    fit). Replaying design -> WLS -> 2*resid.dcol/sum(w) in NumPy pins the
    derivative the native code actually used.
    """
    n = 96
    x = np.linspace(-2.0, 2.0, n)
    omega0, lr = 2.0, 0.1
    y = 0.4 + 1.3 * np.sin(2.3 * x) + 0.2 * np.cos(1.3 * x)

    def design(w):
        return np.column_stack(
            [np.ones(n), x, x**2, np.sin(w * x), np.cos(w * x)]
        )

    Xd = design(omega0)
    c, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = Xd @ c - y
    dcol = x * (c[3] * np.cos(omega0 * x) - c[4] * np.sin(omega0 * x))
    grad = 2.0 * float(resid @ dcol) / n
    assert abs(grad) > 1e-6  # the step must be observable
    expected = omega0 - lr * grad

    got, mse = _core.refine_frequencies(x, y, [omega0], steps=2, lr=lr)
    assert float(got[0]) == pytest.approx(expected, rel=1e-5, abs=1e-7)
    assert np.isfinite(mse)
    # The step must actually have improved on the start point.
    assert mse < float(np.mean((Xd @ c - y) ** 2))


@requires_cpp
def test_p42_refine_frequencies_weighted_prefers_clean_rows():
    rng = np.random.default_rng(0)
    x = np.linspace(-2.0, 2.0, 120)
    y = 0.5 + 1.2 * x + np.sin(2.5 * x)
    y_noisy = y.copy()
    y_noisy[-8:] += 50.0
    w = np.ones_like(y)
    w[-8:] = 0.01
    _o_u, mse_u = _core.refine_frequencies(x, y_noisy, [2.0], steps=60, lr=0.05)
    _o_w, mse_w = _core.refine_frequencies(
        x, y_noisy, [2.0], steps=60, lr=0.05, sample_weight=w
    )
    assert np.isfinite(mse_u) and np.isfinite(mse_w)
    # Down-weighting the corruption must fit the clean rows better.
    assert mse_w < mse_u


# ── P-89: owned FFI inputs + GIL release ─────────────────────────────────


@requires_cpp
def test_p89_native_calls_do_not_mutate_inputs():
    rng = np.random.default_rng(1)
    x = np.linspace(-1.0, 1.0, 60)
    y = 2.0 * x + 1.0
    X = np.column_stack([x, x**2])
    w = np.linspace(0.5, 1.5, 60)
    snaps = [a.copy() for a in (x, y, X, w)]

    _core.refine_frequencies(x, y, [1.0, 2.0], steps=10, lr=0.05, sample_weight=w)
    _core.refine_powers(x, y, [1.5], [1.0], steps=5, sample_weight=w)
    _core.refine_periodic_rational(x, y, 1.0, 1.0, steps=5, sample_weight=w)
    _core.iterative_elastic_net(X, y, 1e-3, 1e-3, n_starts=2, n_iterations=2,
                                sample_weight=w)
    _core.lasso_coordinate_descent(X, y, 0.01, sample_weight=w)

    for original, snapshot in zip((x, y, X, w), snaps):
        assert np.array_equal(original, snapshot)


@requires_cpp
def test_p89_native_solve_releases_gil():
    rng = np.random.default_rng(2)
    n, p = 1200, 20
    X = rng.standard_normal((n, p))
    y = X[:, 0] - X[:, 5] + 0.01 * rng.standard_normal(n)

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
        _core.iterative_elastic_net(
            X, y, 1e-3, 1e-3, n_starts=8, n_iterations=4, max_iter=500
        )
    finally:
        elapsed = time.perf_counter() - t0
        stop.set()
        worker.join(timeout=5.0)

    # Only meaningful when the native call is long enough to overlap; a GIL
    # held for the whole call would leave the counter at (or near) zero.
    if elapsed > 0.2:
        assert ticks["n"] > 0


# ── P-55 / P-56: Gram-based exact-match search ───────────────────────────


def _fast_path_module():
    from scripts import classifier_fast_path

    return classifier_fast_path


def _pair_basis():
    x = np.linspace(-2.0, 2.0, 96, dtype=np.float64)
    basis = np.column_stack([np.ones_like(x), x, x**2, np.sin(x)])
    names = ["1", "x", "x^2", "sin(x)"]
    y = 2.0 * x - 1.5 * x**2
    return basis, names, y


@pytest.mark.parametrize("backend", ["numpy", "cpu"])
def test_p55_p56_backends_agree_on_pair_match(backend):
    cfp = _fast_path_module()
    basis, names, y = _pair_basis()
    diagnostics = {}
    result = cfp.find_exact_symbolic_match(
        basis,
        names,
        y,
        max_terms=2,
        tolerance=1e-8,
        device="cpu",
        exact_match_backend=backend,
        diagnostics=diagnostics,
    )
    assert result is not None
    _formula, mse, coeffs = result
    assert mse < 1e-8
    assert np.isclose(coeffs[names.index("x")], 2.0)
    assert np.isclose(coeffs[names.index("x^2")], -1.5)


@pytest.mark.parametrize("backend", ["numpy", "cpu"])
def test_p55_p56_gram_scan_finds_triple_support(backend):
    cfp = _fast_path_module()
    x = np.linspace(-1.5, 1.5, 120)
    basis = np.column_stack(
        [np.ones_like(x), x, x**2, np.sin(2.0 * x), np.cos(2.0 * x)]
    )
    names = ["1", "x", "x^2", "sin(2*x)", "cos(2*x)"]
    # Exactly representable only by the 3-term support {1, x^2, cos(2*x)}.
    y = 0.75 + 1.25 * x**2 - 2.0 * np.cos(2.0 * x)
    result = cfp.find_exact_symbolic_match(
        basis,
        names,
        y,
        max_terms=3,
        tolerance=1e-8,
        device="cpu",
        exact_match_backend=backend,
    )
    assert result is not None
    formula, mse, coeffs = result
    assert mse < 1e-8
    assert "cos(2*x)" in formula and "x**2" in formula
    # The Gram path must recover the same support the full-design lstsq would.
    assert coeffs[names.index("x^2")] == pytest.approx(1.25, rel=1e-9)
    assert coeffs[names.index("cos(2*x)")] == pytest.approx(-2.0, rel=1e-9)
    assert coeffs[names.index("x")] == pytest.approx(0.0, abs=1e-8)


def test_p56_singular_gram_supports_do_not_raise():
    """Duplicate basis columns make every Gram submatrix singular; the scan
    must skip them (a dependent triple is still reachable as a pair) without
    raising LinAlgError/divide warnings."""
    cfp = _fast_path_module()
    x = np.linspace(-1.0, 1.0, 48)
    basis = np.column_stack([np.ones_like(x), x, x, 2.0 * x, np.sin(x)])
    names = ["1", "x", "x_dup", "x_dup2", "sin(x)"]
    y = 0.3 + np.sin(x)
    result = cfp.find_exact_symbolic_match(
        basis,
        names,
        y,
        max_terms=3,
        tolerance=1e-8,
        device="cpu",
        exact_match_backend="numpy",
    )
    assert result is not None
    assert result[1] < 1e-8


# ── P-99 / P-100: robust IRLS + LM setup reuse ───────────────────────────


def _robust_outlier_problem():
    x = np.linspace(-1.5, 1.5, 90)
    y = 2.0 * x + 1.0
    y_noisy = y.copy()
    y_noisy[-9:] += 25.0  # heavy outliers inside the fit domain
    return x, y_noisy


@pytest.mark.parametrize("loss_mode", ["huber", "trimmed_mse", "student_t"])
def test_p99_irls_fit_is_deterministic(loss_mode):
    """The IRLS delta-Gramian path must not make repeated runs diverge: same
    seed, same data, same champion and objective."""
    x, y = _robust_outlier_problem()
    kwargs = dict(
        pop_size=24,
        generations=3,
        num_islands=1,
        random_seed=11,
        loss_mode=loss_mode,
        huber_delta=1.0,
        trim_fraction=0.1,
        use_lm_inner_optimizer=True,
    )
    a = _core.run_evolution([x], y, **kwargs)
    b = _core.run_evolution([x], y, **kwargs)
    assert a["best_mse"] == pytest.approx(b["best_mse"], rel=0, abs=0)
    assert a["search_loss"] == pytest.approx(b["search_loss"], rel=0, abs=0)
    assert np.isfinite(a["best_mse"]) and np.isfinite(a["search_loss"])


@requires_cpp
def test_p99_huber_irls_recovers_clean_coefficients_with_outliers():
    """Under Huber the reported search objective must be far below the raw
    MSE the outliers produce — the IRLS chain's whole purpose, and the
    observable the delta-Gramian path has to keep intact."""
    x, y = _robust_outlier_problem()
    seed = _core.formula_to_seed_graph("2*x0+1")
    res = _core.run_evolution(
        [x],
        y,
        pop_size=12,
        generations=2,
        num_islands=1,
        random_seed=3,
        loss_mode="huber",
        huber_delta=1.0,
        seed_graphs_py=[seed],
    )
    assert res["seed_graphs_used"] >= 1
    assert res["effective_loss_mode"] == "huber"
    assert np.isfinite(res["search_loss"]) and np.isfinite(res["best_mse"])
    # The nine 25-unit outliers put raw MSE far above the robust objective.
    assert res["search_loss"] < 0.5 * res["best_mse"]

    # On clean data (no rows down-weighted) the same run must agree with MSE,
    # i.e. the incremental path collapses to the plain weighted solve.
    clean = 2.0 * x + 1.0
    res_clean = _core.run_evolution(
        [x],
        clean,
        pop_size=12,
        generations=2,
        num_islands=1,
        random_seed=3,
        loss_mode="huber",
        huber_delta=1.0,
    )
    assert np.isfinite(res_clean["search_loss"])
    assert res_clean["search_loss"] == pytest.approx(
        res_clean["best_mse"], rel=1e-6, abs=1e-9
    )


@requires_cpp
def test_p100_lm_refinement_deterministic_and_finite():
    """LM setup caching must not leak state between outer iterations."""
    x = np.linspace(-2.0, 2.0, 120)
    y = np.sin(3.0 * x) + 0.5 * x
    seed = _core.formula_to_seed_graph("sin(2*x0)+x0")
    kwargs = dict(
        pop_size=16,
        generations=2,
        num_islands=1,
        random_seed=5,
        use_lm_inner_optimizer=True,
        seed_graphs_py=[seed],
    )
    a = _core.run_evolution([x], y, **kwargs)
    b = _core.run_evolution([x], y, **kwargs)
    assert np.isfinite(a["best_mse"])
    assert a["best_mse"] == pytest.approx(b["best_mse"], rel=0, abs=0)
    assert str(a["formula"]) == str(b["formula"])
