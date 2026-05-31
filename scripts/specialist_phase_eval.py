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

    return cases


def _make_estimator(
    *,
    enable_specialist_screening_diagnostics: bool,
    enable_specialist_composition_screening: bool = True,
) -> GlassboxRegressor:
    return GlassboxRegressor(
        use_fast_path=False,
        use_guided_evolution=False,
        use_universal_proposer=False,
        blackbox_mode=True,
        blackbox_standardize=False,
        blackbox_min_features_to_select=2,
        enable_specialist_screening_diagnostics=enable_specialist_screening_diagnostics,
        enable_specialist_composition_screening=enable_specialist_composition_screening,
        population_size=12,
        generations=12,
        multi_start_runs=1,
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
) -> Dict[str, Any]:
    X = np.asarray(case["X"], dtype=np.float64)
    y = np.asarray(case["y"], dtype=np.float64).reshape(-1)

    est = _make_estimator(
        enable_specialist_screening_diagnostics=enable_specialist_screening_diagnostics,
        enable_specialist_composition_screening=enable_specialist_composition_screening,
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
    else:
        raise SystemExit(f"Only phase 0 and phase 2 are implemented in this harness right now, got phase={args.phase}")
    print(json.dumps(result["summary"], indent=2))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nSaved detailed report to {out_path}")

    return 0 if result["summary"].get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
