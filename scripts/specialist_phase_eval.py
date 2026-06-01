"""
Specialist layer phase evaluation harness.

Purpose:
  - compare baseline architecture vs specialist-phase instrumentation/features
  - make each specialist phase testable with explicit success/failure signals
  - start with Phase 0 diagnostics quality and zero-regression checks

Usage:
  python scripts/specialist_phase_eval.py --phase 0
  python scripts/specialist_phase_eval.py --phase 0 --quick
  python scripts/specialist_phase_eval.py --phase 0 --output results/specialist_phase0.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import sys

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from glassbox.sr.sklearn_wrapper import GlassboxRegressor
from scripts import benchmark_common as bc


def _phase0_cases() -> List[Dict[str, Any]]:
    rng = np.random.RandomState(0)

    def grid(a: float, b: float, n: int) -> np.ndarray:
        return np.linspace(float(a), float(b), int(n), dtype=np.float64)

    cases: List[Dict[str, Any]] = []

    x = grid(-3.0, 3.0, 160)
    X = np.column_stack([x, np.sin(x), np.cos(x)])
    y = x * np.sin(x)
    cases.append({
        "name": "product_envelope_sine",
        "X": X,
        "y": y,
        "kind": "compositional",
        "expect_specialist_signal": True,
    })

    x = grid(-3.0, 3.0, 180)
    X = np.column_stack([x, np.sin(2.0 * x), np.cos(2.0 * x)])
    y = np.where(x < 0.0, x * x, np.sin(2.0 * x))
    cases.append({
        "name": "piecewise_proxy_left_poly_right_sine",
        "X": X,
        "y": y,
        "kind": "complementary_regions",
        "expect_specialist_signal": True,
    })

    x = grid(-2.5, 2.5, 170)
    X = np.column_stack([x, x * x, np.exp(-np.abs(x))])
    y = x * x + np.sin(x * x)
    cases.append({
        "name": "poly_plus_nested_periodic",
        "X": X,
        "y": y,
        "kind": "additive_structure",
        "expect_specialist_signal": True,
    })

    X = rng.randn(180, 3)
    y = X[:, 0] + 0.5 * X[:, 1]
    cases.append({
        "name": "simple_linear_control",
        "X": X,
        "y": y,
        "kind": "control",
        "expect_specialist_signal": False,
    })

    X = rng.randn(180, 3)
    y = X[:, 0] * X[:, 1] + 0.02 * rng.randn(180)
    cases.append({
        "name": "bilinear_control",
        "X": X,
        "y": y,
        "kind": "control",
        "expect_specialist_signal": False,
    })

    # Case for Phase 3: composition + evolution constant addition
    x_p3 = grid(-2.5, 2.5, 170)
    X_p3 = np.column_stack([x_p3, np.sin(x_p3), np.cos(x_p3)])
    y_p3 = x_p3 * np.sin(x_p3) + 0.5
    cases.append({
        "name": "composed_product_plus_constant",
        "X": X_p3,
        "y": y_p3,
        "kind": "compositional_seeded_evolution",
        "expect_specialist_signal": True,
    })

    # Case for Phase 4: composition + residual fitting
    x_p4 = grid(-2.5, 2.5, 170)
    X_p4 = np.column_stack([x_p4, np.sin(x_p4), np.cos(x_p4), np.exp(-x_p4)])
    y_p4 = x_p4 * np.sin(x_p4) + np.exp(-x_p4)
    cases.append({
        "name": "composed_product_with_residual",
        "X": X_p4,
        "y": y_p4,
        "kind": "compositional_residual",
        "expect_specialist_signal": True,
    })

    # Phase 6: nested transcendental
    x_p6_1 = grid(-2.0, 2.0, 100)
    X_p6_1 = np.column_stack([x_p6_1, np.sin(x_p6_1), np.cos(x_p6_1)])
    y_p6_1 = np.sin(np.cos(x_p6_1))
    cases.append({
        "name": "phase6_nested_sin_cos",
        "X": X_p6_1,
        "y": y_p6_1,
        "kind": "nested_transcendental",
        "expect_specialist_signal": True,
    })

    # Phase 6: sigmoid gate
    x_p6_2 = grid(-3.0, 3.0, 100)
    X_p6_2 = np.column_stack([x_p6_2, np.exp(-x_p6_2)])
    y_p6_2 = 1.0 / (1.0 + np.exp(-x_p6_2))
    cases.append({
        "name": "phase6_sigmoid_gate",
        "X": X_p6_2,
        "y": y_p6_2,
        "kind": "sigmoid_gate",
        "expect_specialist_signal": True,
    })

    # Phase 6: damped oscillation
    x_p6_3 = grid(-2.0, 2.0, 150)
    X_p6_3 = np.column_stack([x_p6_3, np.sin(3.0 * x_p6_3), np.exp(-x_p6_3**2)])
    y_p6_3 = np.exp(-x_p6_3**2) * np.sin(3.0 * x_p6_3)
    cases.append({
        "name": "phase6_damped_oscillation",
        "X": X_p6_3,
        "y": y_p6_3,
        "kind": "damped_product",
        "expect_specialist_signal": True,
    })

    # Phase 7: slowly converging approximation / multi-stage boosting
    x_p7 = grid(-2.5, 2.5, 120)
    X_p7 = np.column_stack([x_p7, np.sin(x_p7), np.cos(2.0 * x_p7), np.exp(-x_p7**2)])
    y_p7 = np.sin(x_p7) + np.cos(2.0 * x_p7) + np.exp(-x_p7**2)
    cases.append({
        "name": "phase7_boosting_three_terms",
        "X": X_p7,
        "y": y_p7,
        "kind": "multi_stage_boosting",
        "expect_specialist_signal": True,
    })

    # Phase 8: multi-start cross-run memory should retain useful local fits
    x_p8 = grid(-2.5, 2.5, 120)
    X_p8 = np.column_stack([x_p8, np.sin(3.0 * x_p8), np.exp(-x_p8**2), np.cos(x_p8)])
    y_p8 = np.exp(-x_p8**2) * np.sin(3.0 * x_p8) + 0.1 * np.cos(x_p8)
    cases.append({
        "name": "phase8_cross_run_vault",
        "X": X_p8,
        "y": y_p8,
        "kind": "cross_run_specialist_memory",
        "expect_specialist_signal": True,
    })

    return cases


def _make_estimator(
    *,
    enable_specialist_screening_diagnostics: bool,
    enable_specialist_composition_screening: bool = True,
    use_guided_evolution: bool = False,
    multi_start_runs: int = 1,
    enable_specialist_vault_memory: bool = True,
) -> GlassboxRegressor:
    return GlassboxRegressor(
        use_fast_path=False,
        use_guided_evolution=use_guided_evolution,
        use_universal_proposer=False,
        blackbox_mode=True,
        blackbox_standardize=False,
        blackbox_min_features_to_select=2,
        enable_specialist_screening_diagnostics=enable_specialist_screening_diagnostics,
        enable_specialist_composition_screening=enable_specialist_composition_screening,
        enable_residual_stage=True,
        enable_specialist_vault_memory=enable_specialist_vault_memory,
        population_size=12,
        generations=12,
        multi_start_runs=multi_start_runs,
        timeout=20,
        random_state=0,
    )


def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    var = float(np.var(y_true))
    if var < 1e-15:
        return 1.0 if _mse(y_true, y_pred) < 1e-15 else 0.0
    return float(1.0 - _mse(y_true, y_pred) / var)


def _evaluate_run(
    case: Dict[str, Any],
    *,
    enable_specialist_screening_diagnostics: bool,
    enable_specialist_composition_screening: bool = True,
    use_guided_evolution: bool = False,
    multi_start_runs: int = 1,
    enable_specialist_vault_memory: bool = True,
) -> Dict[str, Any]:
    X = np.asarray(case["X"], dtype=np.float64)
    y = np.asarray(case["y"], dtype=np.float64).reshape(-1)

    est = _make_estimator(
        enable_specialist_screening_diagnostics=enable_specialist_screening_diagnostics,
        enable_specialist_composition_screening=enable_specialist_composition_screening,
        use_guided_evolution=use_guided_evolution,
        multi_start_runs=multi_start_runs,
        enable_specialist_vault_memory=enable_specialist_vault_memory,
    )
    t0 = time.time()
    est.fit(X, y)
    elapsed = time.time() - t0

    formula = est.get_formula()
    y_pred, eval_diag = bc.evaluate_formula(formula, X, return_diagnostics=True)
    if y_pred is None:
        y_pred = np.zeros_like(y)

    screening = (getattr(est, "blackbox_diagnostics_", {}) or {}).get("candidate_screening", {})
    specialist = screening.get("specialist_screening")
    composition = (getattr(est, "blackbox_diagnostics_", {}) or {}).get("specialist_composition_screening", {})
    vault = (getattr(est, "blackbox_diagnostics_", {}) or {}).get("specialist_vault", {})
    top_pairs = specialist.get("top_pairs", []) if isinstance(specialist, dict) else []
    max_pair_score = max((float(p.get("complementarity_score", 0.0)) for p in top_pairs), default=0.0)

    return {
        "formula": formula,
        "fit_time_sec": float(elapsed),
        "mse": _mse(y, y_pred),
        "r2": _r2(y, y_pred),
        "eval_ok": bool(eval_diag.get("ok", False)),
        "candidate_count": int(screening.get("candidate_count", 0) or 0),
        "has_specialist_screening": isinstance(specialist, dict),
        "specialist_segment_count": int((specialist or {}).get("segment_count", 0) or 0) if isinstance(specialist, dict) else 0,
        "specialist_pair_count": len(top_pairs),
        "max_complementarity_score": float(max_pair_score),
        "specialist_signal_detected": bool(max_pair_score >= 0.35),
        "composition_proposal_count": int(composition.get("proposal_count", 0) or 0),
        "composition_accepted_count": int(composition.get("accepted_count", 0) or 0),
        "specialist_screening": specialist if enable_specialist_screening_diagnostics else None,
        "specialist_track": getattr(est, "specialist_track_", None),
        "boosting_attempted": bool(getattr(est, "boosting_attempted_", False)),
        "boosting_improved": bool(getattr(est, "boosting_improved_", False)),
        "boosting_stage_count": len(getattr(est, "boosting_stages_", []) or []),
        "boosting_diagnostics": getattr(est, "boosting_diagnostics_", None),
        "specialist_vault": vault if isinstance(vault, dict) else None,
        "specialist_vault_entry_count": int((vault or {}).get("entry_count", 0) or 0) if isinstance(vault, dict) else 0,
        "specialist_vault_composition_count": int((vault or {}).get("composition_count", 0) or 0) if isinstance(vault, dict) else 0,
    }


def run_phase0(*, quick: bool = False) -> Dict[str, Any]:
    cases = _phase0_cases()
    if quick:
        cases = cases[:3]

    results = []
    regression_failures = 0
    specialist_hits = 0
    specialist_expected = 0
    false_positive_controls = 0

    for case in cases:
        case_error = None
        try:
            baseline = _evaluate_run(case, enable_specialist_screening_diagnostics=False)
            phase0 = _evaluate_run(case, enable_specialist_screening_diagnostics=True)
        except Exception as exc:
            baseline = {
                "formula": None,
                "fit_time_sec": None,
                "mse": None,
                "r2": None,
                "eval_ok": False,
                "candidate_count": 0,
                "has_specialist_screening": False,
                "specialist_segment_count": 0,
                "specialist_pair_count": 0,
                "max_complementarity_score": 0.0,
                "specialist_signal_detected": False,
                "specialist_screening": None,
            }
            phase0 = dict(baseline)
            case_error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(limit=10),
            }

        baseline_mse = baseline["mse"]
        phase0_mse = phase0["mse"]
        baseline_r2 = baseline["r2"]
        phase0_r2 = phase0["r2"]
        mse_delta = None if baseline_mse is None or phase0_mse is None else float(phase0_mse - baseline_mse)
        r2_delta = None if baseline_r2 is None or phase0_r2 is None else float(phase0_r2 - baseline_r2)
        if case_error is not None or (mse_delta is not None and r2_delta is not None and (abs(mse_delta) > 1e-9 or abs(r2_delta) > 1e-9)):
            regression_failures += 1

        expect_signal = bool(case.get("expect_specialist_signal", False))
        detected = bool(phase0["specialist_signal_detected"])
        if expect_signal:
            specialist_expected += 1
            if detected:
                specialist_hits += 1
        elif detected:
            false_positive_controls += 1

        results.append({
            "name": case["name"],
            "kind": case["kind"],
            "expect_specialist_signal": expect_signal,
            "baseline": baseline,
            "phase0": phase0,
            "delta": {
                "mse": mse_delta,
                "r2": r2_delta,
            },
            "error": case_error,
        })

    summary = {
        "phase": 0,
        "n_cases": len(results),
        "behavior_regressions": int(regression_failures),
        "specialist_expected_cases": int(specialist_expected),
        "specialist_hits": int(specialist_hits),
        "control_false_positives": int(false_positive_controls),
        "specialist_recall": (
            float(specialist_hits / specialist_expected)
            if specialist_expected > 0 else None
        ),
    }

    summary["pass"] = bool(
        summary["behavior_regressions"] == 0
        and (
            summary["specialist_expected_cases"] == 0
            or summary["specialist_hits"] >= max(1, summary["specialist_expected_cases"] - 1)
        )
        and summary["control_false_positives"] <= 1
    )

    return {
            "summary": summary,
            "cases": results,
        }


def run_phase2(*, quick: bool = False) -> Dict[str, Any]:
    cases = _phase0_cases()
    if quick:
        cases = cases[:3]

    results = []
    composition_hits = 0
    regressions_vs_phase0 = 0

    for case in cases:
        baseline = _evaluate_run(
            case,
            enable_specialist_screening_diagnostics=False,
            enable_specialist_composition_screening=False,
        )
        phase0 = _evaluate_run(
            case,
            enable_specialist_screening_diagnostics=True,
            enable_specialist_composition_screening=False,
        )
        phase2 = _evaluate_run(
            case,
            enable_specialist_screening_diagnostics=True,
            enable_specialist_composition_screening=True,
        )

        if (
            phase2.get("mse") is not None
            and phase0.get("mse") is not None
            and float(phase2["mse"]) > float(phase0["mse"]) + 1e-9
        ):
            regressions_vs_phase0 += 1
        if int(phase2.get("composition_accepted_count", 0) or 0) > 0:
            composition_hits += 1

        results.append({
            "name": case["name"],
            "kind": case["kind"],
            "baseline": baseline,
            "phase0": phase0,
            "phase2": phase2,
        })

    summary = {
        "phase": 2,
        "n_cases": len(results),
        "composition_hits": int(composition_hits),
        "regressions_vs_phase0": int(regressions_vs_phase0),
        "pass": bool(composition_hits >= 1 and regressions_vs_phase0 == 0),
    }
    return {
        "summary": summary,
        "cases": results,
    }


def run_phase3(*, quick: bool = False) -> Dict[str, Any]:
    cases = _phase0_cases()
    if quick:
        p3_cases = [c for c in cases if c["name"] == "composed_product_plus_constant"]
        cases = cases[:3] + p3_cases

    results = []
    phase3_hits = 0

    for case in cases:
        phase3 = _evaluate_run(
            case,
            enable_specialist_screening_diagnostics=True,
            enable_specialist_composition_screening=True,
        )

        results.append({
            "name": case["name"],
            "kind": case["kind"],
            "phase3": phase3,
        })

        if case["name"] == "composed_product_plus_constant":
            track = phase3.get("specialist_track")
            if track in ("composed seed + evolution", "screening only"):
                phase3_hits += 1

    summary = {
        "phase": 3,
        "n_cases": len(results),
        "phase3_hits": int(phase3_hits),
        "pass": bool(phase3_hits >= 1),
    }
    return {
        "summary": summary,
        "cases": results,
    }


def run_phase4(*, quick: bool = False) -> Dict[str, Any]:
    cases = _phase0_cases()
    if quick:
        p4_cases = [c for c in cases if c["name"] == "composed_product_with_residual"]
        cases = cases[:3] + p4_cases

    results = []
    phase4_hits = 0

    for case in cases:
        phase4 = _evaluate_run(
            case,
            enable_specialist_screening_diagnostics=True,
            enable_specialist_composition_screening=True,
            use_guided_evolution=True,
        )

        results.append({
            "name": case["name"],
            "kind": case["kind"],
            "phase4": phase4,
        })

        if case["name"] == "composed_product_with_residual":
            formula = phase4.get("formula")
            if formula and ("sin" in formula or "x1" in formula) and ("exp" in formula or "x3" in formula):
                phase4_hits += 1

    summary = {
        "phase": 4,
        "n_cases": len(results),
        "phase4_hits": int(phase4_hits),
        "pass": bool(phase4_hits >= 1),
    }
    return {
        "summary": summary,
        "cases": results,
    }

def run_phase5(*, quick: bool = False) -> Dict[str, Any]:
    cases = _phase0_cases()
    if quick:
        p5_cases = [c for c in cases if c["name"] == "composed_product_with_residual"]
        cases = cases[:3] + p5_cases

    results = []
    phase5_hits = 0

    for case in cases:
        phase5 = _evaluate_run(
            case,
            enable_specialist_screening_diagnostics=True,
            enable_specialist_composition_screening=True,
            use_guided_evolution=True,
        )

        results.append({
            "name": case["name"],
            "kind": case["kind"],
            "phase5": phase5,
        })

        spec_screening = phase5.get("specialist_screening")
        if isinstance(spec_screening, dict):
            hs_segs = spec_screening.get("hot_spot_segments", [])
            if len(hs_segs) >= 1:
                phase5_hits += 1

    summary = {
        "phase": 5,
        "n_cases": len(results),
        "phase5_hits": int(phase5_hits),
        "pass": bool(phase5_hits >= 1),
    }
    return {
        "summary": summary,
        "cases": results,
    }
def run_phase6(*, quick: bool = False) -> Dict[str, Any]:
    cases = _phase0_cases()
    p6_names = {"phase6_nested_sin_cos", "phase6_sigmoid_gate", "phase6_damped_oscillation"}
    p6_cases = [c for c in cases if c["name"] in p6_names]
    if quick:
        p6_cases = p6_cases[:2]

    results = []
    phase6_hits = 0

    for case in p6_cases:
        phase6 = _evaluate_run(
            case,
            enable_specialist_screening_diagnostics=True,
            enable_specialist_composition_screening=True,
            use_guided_evolution=True,
        )

        results.append({
            "name": case["name"],
            "kind": case["kind"],
            "phase6": phase6,
        })

        if phase6.get("r2", 0.0) >= 0.99:
            phase6_hits += 1

    summary = {
        "phase": 6,
        "n_cases": len(results),
        "phase6_hits": int(phase6_hits),
        "pass": bool(phase6_hits >= 1 if quick else phase6_hits >= 2),
    }
    return {
        "summary": summary,
        "cases": results,
    }
def run_phase7(*, quick: bool = False) -> Dict[str, Any]:
    cases = _phase0_cases()
    p7_cases = [c for c in cases if c["name"] == "phase7_boosting_three_terms"]

    results = []
    phase7_hits = 0

    for case in p7_cases:
        phase7 = _evaluate_run(
            case,
            enable_specialist_screening_diagnostics=True,
            enable_specialist_composition_screening=True,
            use_guided_evolution=True,
        )

        results.append({
            "name": case["name"],
            "kind": case["kind"],
            "phase7": phase7,
        })

        if (
            phase7.get("r2", 0.0) >= 0.99
            and phase7.get("boosting_attempted")
            and phase7.get("boosting_improved")
            and phase7.get("boosting_stage_count", 0) >= 1
        ):
            phase7_hits += 1

    summary = {
        "phase": 7,
        "n_cases": len(results),
        "phase7_hits": int(phase7_hits),
        "pass": bool(phase7_hits >= 1),
    }
    return {
        "summary": summary,
        "cases": results,
    }


def run_phase8(*, quick: bool = False) -> Dict[str, Any]:
    cases = _phase0_cases()
    p8_cases = [c for c in cases if c["name"] == "phase8_cross_run_vault"]

    results = []
    phase8_hits = 0

    for case in p8_cases:
        X = np.asarray(case["X"], dtype=np.float64)
        y = np.asarray(case["y"], dtype=np.float64).reshape(-1)
        baseline = _evaluate_run(
            case,
            enable_specialist_screening_diagnostics=True,
            enable_specialist_composition_screening=True,
            use_guided_evolution=True,
            multi_start_runs=2,
            enable_specialist_vault_memory=False,
        )
        phase8 = _evaluate_run(
            case,
            enable_specialist_screening_diagnostics=True,
            enable_specialist_composition_screening=True,
            use_guided_evolution=True,
            multi_start_runs=2,
            enable_specialist_vault_memory=True,
        )
        vault_probe = _make_estimator(
            enable_specialist_screening_diagnostics=True,
            enable_specialist_composition_screening=True,
            use_guided_evolution=False,
            multi_start_runs=2,
            enable_specialist_vault_memory=True,
        )
        vault_probe.n_features_in_ = X.shape[1]
        vault_probe.blackbox_diagnostics_ = {}
        probe_candidates = [
            {"formula": "x1*x2", "validation_r2": 0.80, "validation_mse": 0.02, "source": "probe_product"},
            {"formula": "0.1*x3", "validation_r2": 0.20, "validation_mse": 0.08, "source": "probe_residual"},
            {"formula": "x1*x2 + 0.1*x3", "validation_r2": 0.99, "validation_mse": 0.001, "source": "probe_combined"},
        ]
        vault_probe._update_specialist_vault_after_run(
            probe_candidates,
            X,
            y,
            0,
            current_best_formula="0*x0",
        )
        seeded_candidates = vault_probe._vault_seed_candidates_for_run(
            probe_candidates,
            X,
            y,
            "0*x0",
            float(np.mean(y ** 2)),
            1,
            max_candidates=8,
        )
        vault_probe_diag = vault_probe.specialist_vault_.to_dict()
        vault_probe_diag["seeded_candidate_count"] = len(seeded_candidates or [])
        vault_probe_diag["seeded_vault_candidate_count"] = sum(
            1 for candidate in (seeded_candidates or [])
            if candidate.get("from_specialist_vault")
        )

        results.append({
            "name": case["name"],
            "kind": case["kind"],
            "baseline": baseline,
            "phase8": phase8,
            "vault_probe": vault_probe_diag,
        })

        if (
            (
                phase8.get("specialist_vault_entry_count", 0) >= 1
                and phase8.get("specialist_vault_composition_count", 0) >= 1
            )
            or (
                vault_probe_diag.get("entry_count", 0) >= 1
                and vault_probe_diag.get("seeded_vault_candidate_count", 0) >= 1
            )
        ):
            phase8_hits += 1

    summary = {
        "phase": 8,
        "n_cases": len(results),
        "phase8_hits": int(phase8_hits),
        "pass": bool(phase8_hits >= 1),
    }
    return {
        "summary": summary,
        "cases": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate specialist-layer phases against baseline behavior.")
    parser.add_argument("--phase", type=int, default=0, help="Specialist phase to evaluate")
    parser.add_argument("--quick", action="store_true", help="Run a smaller smoke subset")
    parser.add_argument("--output", type=str, default="", help="Optional JSON output path")
    args = parser.parse_args()

    if args.phase == 0:
        result = run_phase0(quick=bool(args.quick))
    elif args.phase == 2:
        result = run_phase2(quick=bool(args.quick))
    elif args.phase == 3:
        result = run_phase3(quick=bool(args.quick))
    elif args.phase == 4:
        result = run_phase4(quick=bool(args.quick))
    elif args.phase == 5:
        result = run_phase5(quick=bool(args.quick))
    elif args.phase == 6:
        result = run_phase6(quick=bool(args.quick))
    elif args.phase == 7:
        result = run_phase7(quick=bool(args.quick))
    elif args.phase == 8:
        result = run_phase8(quick=bool(args.quick))
    else:
        raise SystemExit(f"Only phase 0, 2, 3, 4, 5, 6, 7, and 8 are implemented in this harness right now, got phase={args.phase}")
    print(json.dumps(result["summary"], indent=2))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nSaved detailed report to {out_path}")

    return 0 if result["summary"].get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
