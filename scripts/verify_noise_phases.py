"""Phase 3-6 recovery verification probe (clean recovery, not noisy EXACT%)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from glassbox.sr.sklearn_wrapper import (
    GlassboxRegressor,
    _formula_unit_compatible,
    _infer_formula_units,
    _robust_loss,
)


def mse(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.mean((a - b) ** 2))


def r2(y, p):
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    v = float(np.var(y))
    if v < 1e-15:
        return 1.0 if mse(y, p) < 1e-15 else 0.0
    return 1.0 - mse(y, p) / v


def eval_formula(est, formula, X):
    return np.asarray(est._safe_eval_formula_array(formula, X), dtype=np.float64).reshape(-1)


def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main():
    rng = np.random.default_rng(42)
    results = {}

    # ------------------------------------------------------------------
    # A) Python robust math (Phase 4) — deterministic
    # ------------------------------------------------------------------
    section("A) Phase 4 robust loss math (outliers)")
    y_true = np.zeros(100)
    y_pred = np.zeros(100)
    y_true[-5:] = 100.0  # 5 huge residuals if pred=0
    pred = np.zeros(100)
    target = y_true.copy()
    m = _robust_loss(pred, target, "mse")
    h = _robust_loss(pred, target, "huber")
    t = _robust_loss(pred, target, "trimmed_mse", trim_fraction=0.1)
    s = _robust_loss(pred, target, "student_t")
    print(f"  mse={m:.4f} huber={h:.4f} trimmed={t:.4f} student_t={s:.4f}")
    results["robust_math"] = h < m * 0.1 and t < m * 0.1
    print("  PASS" if results["robust_math"] else "  FAIL")

    # ------------------------------------------------------------------
    # B) C++ weighted evolution (Phase 3) — known-good
    # ------------------------------------------------------------------
    section("B) Phase 3 weighted C++ evolution (block outliers)")
    from glassbox.sr.cpp import require_cpp_core
_core = require_cpp_core()

    x = np.linspace(-3, 3, 120)
    y_clean = 2.0 * x + 1.0
    y_noisy = y_clean.copy()
    y_noisy[-12:] += 40.0
    w = np.ones_like(y_noisy)
    w[-12:] = 0.01
    X_list = [x.astype(np.float64)]

    t0 = time.time()
    res_u = _core.run_evolution(
        X_list, y_noisy, pop_size=40, generations=40, early_stop_mse=1e-12,
        random_seed=7, num_islands=4,
    )
    res_w = _core.run_evolution(
        X_list, y_noisy, pop_size=40, generations=40, early_stop_mse=1e-12,
        random_seed=7, num_islands=4, y_weights=w,
    )
    dt = time.time() - t0

    def clean_of(res):
        f = str(res.get("formula", "") or "")
        if not f:
            return 1e9, f
        # use simple eval via numpy
        try:
            # prefer glassbox eval
            est = GlassboxRegressor()
            est.n_features_in_ = 1
            p = eval_formula(est, f, x.reshape(-1, 1))
            return mse(y_clean, p), f
        except Exception:
            return 1e9, f

    cu, fu = clean_of(res_u)
    cw, fw = clean_of(res_w)
    print(f"  unweighted clean_mse={cu:.3e} formula={fu[:80]}")
    print(f"  weighted   clean_mse={cw:.3e} formula={fw[:80]}")
    print(f"  time={dt:.1f}s")
    results["phase3_weights"] = cw < cu * 0.5 or cw < 1e-3
    print("  PASS" if results["phase3_weights"] else "  FAIL (weights should beat plain on clean recovery)")

    # ------------------------------------------------------------------
    # C) Phase 4 robust C++ modes under same outlier data
    # ------------------------------------------------------------------
    section("C) Phase 4 robust C++ loss modes (same outliers)")
    modes = {
        "mse": {},
        "huber": {"loss_mode": "huber"},
        "trimmed_mse": {"loss_mode": "trimmed_mse", "trim_fraction": 0.15},
    }
    mode_clean = {}
    for name, kw in modes.items():
        r = _core.run_evolution(
            X_list, y_noisy, pop_size=40, generations=40, early_stop_mse=1e-12,
            random_seed=11, num_islands=4, **kw,
        )
        c, f = clean_of(r)
        mode_clean[name] = (c, f, r.get("search_loss"), r.get("best_mse"))
        print(f"  {name:12s} clean_mse={c:.3e} best_mse={r.get('best_mse')} search={r.get('search_loss')} form={f[:60]}")
    best_robust = min(mode_clean["huber"][0], mode_clean["trimmed_mse"][0])
    results["phase4_robust"] = best_robust < mode_clean["mse"][0] * 0.9 or best_robust < 10.0
    print("  PASS" if results["phase4_robust"] else "  WEAK (robust may not fully recover; weights still stronger)")

    # ------------------------------------------------------------------
    # D) Phase 6 noise-aware cleanup slack + residual reject
    # ------------------------------------------------------------------
    section("D) Phase 6 cleanup slack + residual guards")
    est = GlassboxRegressor(random_state=1, enable_residual_stage=True, use_guided_evolution=True)
    est.n_features_in_ = 1
    est.blackbox_diagnostics_ = {}
    x2 = np.linspace(-2, 2, 120)
    X2 = x2.reshape(-1, 1)
    y_c = np.sin(x2)
    y_n = y_c + rng.normal(0, 0.35, size=x2.shape)
    rel_c, _, d_c = est._noise_aware_cleanup_slack("sin(x0)", X2, y_c)
    rel_n, _, d_n = est._noise_aware_cleanup_slack("sin(x0)", X2, y_n)
    print(f"  clean slack={rel_c:.3f} noise_ratio={d_c.get('noise_ratio', 0):.3f}")
    print(f"  noisy slack={rel_n:.3f} noise_ratio={d_n.get('noise_ratio', 0):.3f}")
    slack_ok = rel_n >= rel_c and d_n.get("noise_ratio", 0) > d_c.get("noise_ratio", 0)
    print("  slack PASS" if slack_ok else "  slack FAIL")

    # residual reject noise formula
    y_res = np.sin(x2) + 0.3 * x2
    est2 = GlassboxRegressor(random_state=2, enable_residual_stage=True, use_guided_evolution=True)
    est2.n_features_in_ = 1
    est2.blackbox_diagnostics_ = {}

    def _fake_build(*a, **k):
        return [{"formula": "100*sin(20*x0)", "source": "noise", "complexity": 6}]

    def _fake_refine(pool, X_in, y_in, max_candidates=6):
        return [{
            "formula": "100*sin(20*x0)", "source": "noise",
            "complexity": 6, "risk_score": 0.8, "validation_r2": 0.1,
        }]

    est2._build_residual_mini_search_candidates = _fake_build
    est2._refine_candidate_formulas = _fake_refine
    out = est2._stage_residual_symbolic_fit(X2, y_res, "sin(x0)", _allow_recursion=True)
    residual_ok = out is None and (
        est2._residual_stage_guard_.get("accepted") is False
    )
    print(f"  residual noise rejected: {out is None} guard={est2._residual_stage_guard_.get('reason')}")
    print("  residual PASS" if residual_ok else "  residual FAIL")

    # weighted pareto prefers true structure under outliers
    est3 = GlassboxRegressor(random_state=3)
    est3.n_features_in_ = 1
    yw = y_clean.copy()
    yw[-8:] += 50.0
    ww = np.ones(len(yw))
    ww[-8:] = 0.01
    est3.sample_weight_ = ww
    est3.sample_weight_provided_ = True
    Xw = x.reshape(-1, 1)
    sel = est3._select_blackbox_pareto_formula(
        [
            {"formula": "2*x0 + 1", "source": "true", "complexity": 3},
            {"formula": "0", "source": "const", "complexity": 1},
            {"formula": "50*sin(x0)+2*x0", "source": "overfit", "complexity": 8},
        ],
        Xw,
        yw,
    )
    pareto_ok = sel is not None and "x0" in sel["formula"] and "2" in sel["formula"]
    print(f"  weighted pareto selected: {sel.get('formula') if sel else None}")
    print("  pareto PASS" if pareto_ok else "  pareto FAIL")
    results["phase6"] = slack_ok and residual_ok and pareto_ok

    # ------------------------------------------------------------------
    # E) Phase 5 units filter + C++ dim penalty
    # ------------------------------------------------------------------
    section("E) Phase 5 units (filter + C++ penalty active)")
    # L, T -> velocity L/T
    iu = [[0, 1, 0], [0, 0, 1]]
    ou = [0, 1, -1]
    good = _infer_formula_units("x0/x1", iu, ou)
    bad = _infer_formula_units("x0+x1", iu, ou)
    print(f"  x0/x1 penalty={good['penalty']:.4f} ok={good['ok']}")
    print(f"  x0+x1 penalty={bad['penalty']:.4f} ok={bad['ok']}")
    hard_good, _ = _formula_unit_compatible("x0/x1", iu, ou, "hard")
    hard_bad, _ = _formula_unit_compatible("x0+x1", iu, ou, "hard")
    print(f"  hard accept x0/x1={hard_good} reject x0+x1={not hard_bad}")

    est_u = GlassboxRegressor(
        random_state=5,
        input_units=iu,
        output_units=ou,
        unit_mode="hard",
        dim_penalty_weight=5.0,
    )
    est_u.blackbox_diagnostics_ = {}
    est_u._activate_physics_units(2)
    kept = est_u._filter_candidates_by_units([
        {"formula": "x0/x1", "mse": 1.0, "complexity": 3},
        {"formula": "x0+x1", "mse": 0.001, "complexity": 3},
        {"formula": "sin(x0)", "mse": 0.1, "complexity": 2},
    ])
    kept_f = [c["formula"] for c in kept]
    print(f"  hard filter kept={kept_f}")
    filter_ok = "x0/x1" in kept_f and "x0+x1" not in kept_f

    # Physical recovery: v = L/t with noise; units vs no-units short C++ run
    L = rng.uniform(1.0, 5.0, size=100)
    T = rng.uniform(0.5, 2.0, size=100)
    v_clean = L / T
    v = v_clean + rng.normal(0, 0.05 * np.std(v_clean), size=100)
    X_phys = [L.astype(np.float64), T.astype(np.float64)]
    t0 = time.time()
    r_no = _core.run_evolution(
        X_phys, v, pop_size=30, generations=35, early_stop_mse=1e-12,
        random_seed=3, num_islands=4,
    )
    r_un = _core.run_evolution(
        X_phys, v, pop_size=30, generations=35, early_stop_mse=1e-12,
        random_seed=3, num_islands=4,
        input_units=iu, output_units=ou, dim_penalty_weight=5.0,
    )
    dt = time.time() - t0

    def phys_clean(res):
        f = str(res.get("formula", "") or "")
        est = GlassboxRegressor()
        est.n_features_in_ = 2
        try:
            p = eval_formula(est, f, np.column_stack([L, T]))
            return mse(v_clean, p), f
        except Exception:
            return 1e9, f

    c_no, f_no = phys_clean(r_no)
    c_un, f_un = phys_clean(r_un)
    print(f"  no-units  clean_mse={c_no:.3e} form={f_no[:70]}")
    print(f"  with-units clean_mse={c_un:.3e} form={f_un[:70]}")
    print(f"  time={dt:.1f}s")
    # units should not hurt much; ideally help or match
    units_ok = filter_ok and hard_good and (not hard_bad) and (c_un <= c_no * 1.5 or c_un < 1e-2)
    results["phase5_units"] = units_ok
    print("  PASS" if units_ok else "  WEAK/FAIL")

    # ------------------------------------------------------------------
    # F) End-to-end GlassboxRegressor weights (short budget)
    # ------------------------------------------------------------------
    section("F) End-to-end estimator sample_weight (short budget)")
    X = x.reshape(-1, 1)
    common = dict(
        population_size=40,
        generations=30,
        timeout=25,
        random_state=9,
        use_fast_path=False,
        use_guided_evolution=False,
        use_universal_proposer=False,
        enable_residual_stage=False,
        enable_specialist_composition_screening=False,
        enable_specialist_screening_diagnostics=False,
        num_islands=4,
        multi_start_runs=1,
        adaptive_compute_budget=False,
    )
    t0 = time.time()
    est_plain = GlassboxRegressor(**common)
    est_plain.fit(X, y_noisy)
    est_w = GlassboxRegressor(**common)
    est_w.fit(X, y_noisy, sample_weight=w)
    dt = time.time() - t0

    def clean_est(est):
        f = est.get_formula()
        try:
            p = est.predict(X)
            return mse(y_clean, p), f
        except Exception:
            return 1e9, f

    cp, fp = clean_est(est_plain)
    cw2, fw2 = clean_est(est_w)
    print(f"  plain   clean_mse={cp:.3e} formula={fp}")
    print(f"  weights clean_mse={cw2:.3e} formula={fw2}")
    print(f"  physics_constrained plain={getattr(est_plain,'physics_constrained_',None)} "
          f"weights={getattr(est_w,'physics_constrained_',None)}")
    print(f"  time={dt:.1f}s")
    results["e2e_weights"] = cw2 <= cp * 1.05 or cw2 < 0.5
    print("  PASS" if results["e2e_weights"] else "  WEAK")

    # ------------------------------------------------------------------
    section("SUMMARY")
    for k, v in results.items():
        print(f"  {k:20s} {'PASS' if v else 'FAIL/WEAK'}")
    n_pass = sum(1 for v in results.values() if v)
    print(f"  {n_pass}/{len(results)} probes passed")
    return 0 if n_pass >= 4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
