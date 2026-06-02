"""Shared benchmark helpers for benchmark_suite.py and run_srbench_local.py."""

from __future__ import annotations

import ast
import math
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent


def get_official_pmlb_regression_datasets():
    """Return the current PMLB regression dataset list used by SRBench black-box track."""
    try:
        from pmlb import regression_dataset_names
    except Exception:
        return []
    return list(regression_dataset_names)


def _read_symbolic_formula_from_metadata(dataset_dir):
    """Best-effort extraction of the symbolic target formula from PMLB metadata."""
    for meta_name in ("metadata.yaml", "metadata.yml"):
        meta_path = Path(dataset_dir) / meta_name
        if not meta_path.exists():
            continue
        try:
            text = meta_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        for key in ("formula", "equation", "symbolic_model", "model", "truth"):
            match = re.search(rf"(?im)^\s*{re.escape(key)}\s*:\s*(.+?)\s*$", text)
            if match:
                value = match.group(1).strip().strip("'\"")
                if value:
                    return value
    return ""


def discover_official_ground_truth_problems(data_dir):
    """Discover SRBench ground-truth files from a PMLB datasets checkout."""
    if not data_dir:
        return []

    root = Path(data_dir)
    if not root.exists():
        return []

    candidates = []
    patterns = [
        "strogatz_*/*.tsv",
        "strogatz_*/*.tsv.gz",
        "feynman_*/*.tsv",
        "feynman_*/*.tsv.gz",
        "**/strogatz_*.tsv",
        "**/strogatz_*.tsv.gz",
        "**/feynman_*.tsv",
        "**/feynman_*.tsv.gz",
    ]
    seen = set()
    for pattern in patterns:
        for path in root.glob(pattern):
            if not path.is_file():
                continue
            key = str(path.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            dataset_dir = path.parent
            name = (
                dataset_dir.name
                if dataset_dir.name.startswith(("strogatz_", "feynman_"))
                else path.stem
            )
            if name.endswith(".tsv"):
                name = name[:-4]
            candidates.append(
                {
                    "kind": "file",
                    "name": name,
                    "path": str(path),
                    "true_formula": _read_symbolic_formula_from_metadata(dataset_dir),
                }
            )

    return sorted(candidates, key=lambda problem: problem["name"])


def r2_score(y_true, y_pred):
    """Compute R^2 score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-15:
        return 1.0 if ss_res < 1e-15 else 0.0
    return 1.0 - ss_res / ss_tot


def mse_score(y_true, y_pred):
    """Compute MSE."""
    return float(np.mean((y_true - y_pred) ** 2))


def model_size(formula_str):
    """Rough complexity measure: count operators and terms."""
    if not formula_str:
        return 0
    ops = sum(1 for c in formula_str if c in "+-*/^")
    funcs = sum(formula_str.count(f) for f in ["sin", "cos", "exp", "log", "sqrt"])
    return ops + funcs + 1


def normalize_formula_text(formula):
    """Normalize common unicode/operator variants to a parser-friendly form."""
    if not formula:
        return formula
    formula = (
        str(formula)
        .replace("Â²", "^2")
        .replace("Â³", "^3")
        .replace("Â·", "*")
        .replace("â‹…", "*")
        .replace("Ã—", "*")
        .replace("Ï€", "pi")
        .replace("âˆš", "sqrt")
        .replace("Ï†", "phi")
        .replace("Ï‰", "omega")
        .replace("np.", "")
    )
    return re.sub(r"\s+", "", formula)


def infer_formula_n_features(formula):
    indices = [int(m.group(1)) for m in re.finditer(r"\bx(\d+)\b", str(formula or ""))]
    return max(indices) + 1 if indices else 1


def protect_fractional_powers(formula):
    """Rewrite simple non-integer powers to a real-valued protected form."""
    text = str(formula or "")
    if "**" not in text:
        return text

    class _FractionalPowerProtector(ast.NodeTransformer):
        def visit_BinOp(self, node):  # noqa: N802 - ast API name
            node = self.generic_visit(node)
            if not isinstance(node.op, ast.Pow):
                return node
            exponent = None
            if isinstance(node.right, ast.Constant) and isinstance(node.right.value, (int, float)):
                exponent = float(node.right.value)
            elif (
                isinstance(node.right, ast.UnaryOp)
                and isinstance(node.right.op, ast.USub)
                and isinstance(node.right.operand, ast.Constant)
                and isinstance(node.right.operand.value, (int, float))
            ):
                exponent = -float(node.right.operand.value)
            if exponent is None:
                return node
            nearest_int = round(exponent)
            if abs(exponent - nearest_int) <= 1e-10:
                node.right = ast.Constant(value=int(nearest_int))
                return node
            return ast.Call(
                func=ast.Name(id="_signed_power", ctx=ast.Load()),
                args=[node.left, ast.Constant(value=exponent)],
                keywords=[],
            )

    try:
        tree = ast.parse(text, mode="eval")
        tree = _FractionalPowerProtector().visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    except Exception:
        return text


def simplify_formula_native(formula, int_tol=0.05, zero_tol=1e-3):
    """Simplify with the native C++ simplifier when available."""
    try:
        cpp_dir = _REPO_ROOT / "glassbox" / "sr" / "cpp"
        if str(cpp_dir) not in sys.path:
            sys.path.insert(0, str(cpp_dir))

        import _core  # type: ignore

        if hasattr(_core, "simplify_formula"):
            simplified = _core.simplify_formula(
                formula,
                int_tol=float(int_tol),
                zero_tol=float(zero_tol),
                max_passes=6,
                use_nsimplify=True,
                use_identities=True,
                n_features=infer_formula_n_features(formula),
            )
        elif hasattr(_core, "simplify_formula_cpp"):
            simplified = _core.simplify_formula_cpp(formula)
        else:
            return None

        if simplified and simplified not in {"N/A", "ERROR", "?"}:
            return str(simplified)
    except Exception:
        return None
    return None


def postprocess_formula(
    formula,
    *,
    fraction_tol=0.0,
    max_fraction_denominator=12,
):
    """Apply the shared benchmark formula cleanup pipeline."""
    normalized = normalize_formula_text(formula)
    if not normalized or normalized in {"N/A", "ERROR", "?"}:
        return normalized

    evo_int_tol = 0.05
    evo_zero_tol = 1e-3
    helper_unsafe = "_signed_power" in normalized or "Abs(" in normalized

    if not helper_unsafe:
        native = simplify_formula_native(normalized, evo_int_tol, evo_zero_tol)
        if native is not None:
            normalized = native

    try:
        try:
            from simplify_formula import simplify_onn_formula, SnapConfig, snap_formula_floats
        except ImportError:
            from scripts.simplify_formula import simplify_onn_formula, SnapConfig, snap_formula_floats

        formula_len = len(normalized)
        term_estimate = max(1, len([t for t in re.split(r"\s*[+-]\s*", normalized) if t.strip()]))
        too_complex_for_symbolic = formula_len > 500 or term_estimate > 24
        sympy_unsafe = helper_unsafe or "Piecewise(" in normalized or "Eq(" in normalized

        if too_complex_for_symbolic or sympy_unsafe:
            snapped = snap_formula_floats(
                normalized,
                SnapConfig(
                    int_tol=evo_int_tol,
                    zero_tol=evo_zero_tol,
                    fraction_tol=fraction_tol,
                    max_fraction_denominator=max_fraction_denominator,
                ),
            )
            return protect_fractional_powers(snapped.replace("^", "**"))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"sympy\..*")
            warnings.filterwarnings("ignore", message=r".*Using non-Expr arguments in Mul.*")
            _, simplified_expr = simplify_onn_formula(
                normalized,
                int_tol=evo_int_tol,
                zero_tol=evo_zero_tol,
                fraction_tol=fraction_tol,
                max_fraction_denominator=max_fraction_denominator,
                use_nsimplify=(formula_len <= 300 and term_estimate <= 16),
            )
        simplified_text = str(simplified_expr)
        if "Piecewise(" in simplified_text or "Eq(" in simplified_text:
            snapped = snap_formula_floats(
                normalized,
                SnapConfig(
                    int_tol=evo_int_tol,
                    zero_tol=evo_zero_tol,
                    fraction_tol=fraction_tol,
                    max_fraction_denominator=max_fraction_denominator,
                ),
            )
            return protect_fractional_powers(snapped.replace("^", "**"))
        return protect_fractional_powers(simplified_text.replace("^", "**"))
    except Exception:
        return protect_fractional_powers(normalized.replace("^", "**"))


def evaluate_formula(formula_str, X, *, return_diagnostics=False):
    """Evaluate a discovered formula string strictly on X."""
    diagnostics = {
        "ok": False,
        "reason": None,
        "exception_type": None,
        "exception_message": None,
        "formula": None,
    }

    def _finish(result, reason=None, exc=None):
        diagnostics["ok"] = result is not None
        diagnostics["reason"] = reason
        diagnostics["formula"] = formula if "formula" in locals() else None
        if exc is not None:
            diagnostics["exception_type"] = type(exc).__name__
            diagnostics["exception_message"] = str(exc)
        return (result, diagnostics) if return_diagnostics else result

    if not formula_str:
        return _finish(None, reason="empty_formula")

    formula = normalize_formula_text(formula_str).strip()
    formula = re.sub(r"\|([^|]+)\|", r"abs(\1)", formula)
    formula = formula.replace("^", "**")
    formula = protect_fractional_powers(formula)

    def _safe_numpy_log(x, base=None):
        with np.errstate(divide="ignore", invalid="ignore"):
            x_arr = np.asarray(x, dtype=np.float64)
            # The estimator's formula evaluator treats log(Abs(x)) at x=0 as a
            # protected boundary value. Mirror that here so valid displayed
            # formulas are not rejected only because the benchmark grid includes
            # an endpoint at zero.
            x_arr = np.where(x_arr == 0.0, 1e-300, x_arr)
            out = np.log(x_arr)
            if base is not None:
                out = out / np.log(base)
        return out

    def _signed_power(base, power):
        base_arr = np.asarray(base, dtype=np.float64)
        power_val = float(power)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            if power_val < 0:
                return np.sign(base_arr) / ((np.abs(base_arr) + 1e-12) ** abs(power_val))
            return np.sign(base_arr) * (np.abs(base_arr) ** power_val)

    def _safe_exp(x):
        return np.exp(np.clip(x, -500.0, 500.0))

    context = {
        "np": np,
        "log": _safe_numpy_log,
        "sin": np.sin,
        "cos": np.cos,
        "exp": _safe_exp,
        "sqrt": np.sqrt,
        "abs": np.abs,
        "Abs": np.abs,
        "sign": np.sign,
        "_signed_power": _signed_power,
        "pi": np.pi,
        "E": np.e,
        "e": np.e,
    }
    for i in range(X.shape[1]):
        context[f"x{i}"] = X[:, i]
    if X.shape[1] == 1:
        context["x"] = X[:, 0]

    try:
        with np.errstate(all="raise"):
            y_pred = eval(formula, {"__builtins__": None}, context)
        if isinstance(y_pred, (int, float)):
            y_pred = np.full(X.shape[0], y_pred, dtype=np.float64)
        else:
            y_pred = np.asarray(y_pred, dtype=np.float64)
        if y_pred.ndim == 0:
            y_pred = np.full(X.shape[0], float(y_pred), dtype=np.float64)
        elif y_pred.shape[0] != X.shape[0]:
            return _finish(None, reason="shape_mismatch")
        if not np.all(np.isfinite(y_pred)):
            return _finish(None, reason="non_finite_output")
        return _finish(y_pred, reason="ok")
    except FloatingPointError as exc:
        text = str(exc).lower()
        if "divide by zero" in text:
            reason = "divide_by_zero"
        elif "overflow" in text:
            reason = "overflow"
        elif "invalid value" in text:
            if "sqrt" in formula.lower():
                reason = "invalid_sqrt"
            elif "log" in formula.lower():
                reason = "invalid_log"
            else:
                reason = "invalid_value"
        else:
            reason = "floating_point_error"
        return _finish(None, reason=reason, exc=exc)
    except (SyntaxError, NameError) as exc:
        return _finish(None, reason="parse_error", exc=exc)
    except Exception as exc:
        return _finish(None, reason="eval_error", exc=exc)


def evaluate_formula_mse(formula, x, y):
    """Evaluate displayed formula against ground-truth data and return MSE."""
    if not formula:
        return None
    normalized = normalize_formula_text(formula)
    if not normalized or normalized in {"N/A", "ERROR", "?"}:
        return None

    X = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    y_true = np.asarray(y, dtype=np.float64).reshape(-1)
    y_pred = evaluate_formula(normalized, X)
    if y_pred is None:
        return None

    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_pred.shape != y_true.shape:
        return None

    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    if mask.sum() < 10:
        return None

    mse = float(np.mean((y_pred[mask] - y_true[mask]) ** 2))
    if not math.isfinite(mse):
        return None
    return mse


def evaluate_formula_mse_on_X(formula, X, y):
    """Evaluate displayed formula against ground-truth data for 1D or multivariate X."""
    if not formula:
        return None
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    y_true = np.asarray(y, dtype=np.float64).reshape(-1)
    y_pred = evaluate_formula(formula, X_arr)
    if y_pred is None:
        return None
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_pred.shape != y_true.shape:
        return None
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    if mask.sum() < 10:
        return None
    mse = float(np.mean((y_pred[mask] - y_true[mask]) ** 2))
    return mse if math.isfinite(mse) else None


def postprocess_formula_with_fidelity_guard(
    formula,
    X,
    y,
    *,
    relative_slack=0.10,
    absolute_slack=1e-9,
):
    """Postprocess a formula, but keep the original if cleanup worsens benchmark fit."""
    processed = postprocess_formula(formula, fraction_tol=0.01, max_fraction_denominator=12)
    raw_mse = evaluate_formula_mse_on_X(formula, X, y)
    processed_mse = evaluate_formula_mse_on_X(processed, X, y)

    if processed_mse is None:
        fallback = protect_fractional_powers(normalize_formula_text(formula).replace("^", "**"))
        fallback_mse = evaluate_formula_mse_on_X(fallback, X, y)
        X_diag = np.asarray(X, dtype=np.float64)
        if X_diag.ndim == 1:
            X_diag = X_diag.reshape(-1, 1)
        _, processed_diag = evaluate_formula(processed, X_diag, return_diagnostics=True)
        return (fallback if fallback_mse is not None else processed), {
            "postprocess_guard_triggered": True,
            "postprocess_raw_mse": raw_mse,
            "postprocess_processed_mse": processed_mse,
            "postprocess_fallback_mse": fallback_mse,
            "postprocess_guard_reason": "processed_formula_eval_failed",
            "postprocess_processed_eval_diagnostics": processed_diag,
        }

    if raw_mse is not None:
        allowed = raw_mse * (1.0 + max(0.0, float(relative_slack))) + max(0.0, float(absolute_slack))
        if processed_mse > allowed:
            fallback = protect_fractional_powers(normalize_formula_text(formula).replace("^", "**"))
            fallback_mse = evaluate_formula_mse_on_X(fallback, X, y)
            if fallback_mse is not None and fallback_mse <= processed_mse:
                return fallback, {
                    "postprocess_guard_triggered": True,
                    "postprocess_raw_mse": raw_mse,
                    "postprocess_processed_mse": processed_mse,
                    "postprocess_fallback_mse": fallback_mse,
                    "postprocess_guard_reason": "processed_formula_worse",
                    "postprocess_processed_eval_diagnostics": None,
                }

    return processed, {
        "postprocess_guard_triggered": False,
        "postprocess_raw_mse": raw_mse,
        "postprocess_processed_mse": processed_mse,
        "postprocess_fallback_mse": None,
        "postprocess_guard_reason": None,
        "postprocess_processed_eval_diagnostics": None,
    }


def estimate_timeout_budget(base_timeout, n_features, n_train, adaptive_timeout):
    """Scale timeout by problem size when adaptive timeout is enabled."""
    if not adaptive_timeout:
        return int(base_timeout)
    complexity = (
        1.0
        + 0.15 * max(0, n_features - 1)
        + 0.08 * min(1.0, math.log10(max(50, n_train)) / 3.0)
    )
    budget = int(round(base_timeout * complexity))
    return int(min(max(20, budget), base_timeout * 2))


def parse_seed_list(seed_text):
    """Parse a comma-separated seed list and return sorted unique integers."""
    if seed_text is None:
        return [42]
    parts = [p.strip() for p in str(seed_text).split(",") if p.strip()]
    if not parts:
        return [42]
    seeds = []
    for part in parts:
        seeds.append(int(part))
    return sorted(set(seeds))


def compute_stability_stats(values):
    """Compute median/IQR/std and worst-decile summary for a metric list."""
    if not values:
        return {
            "median": None,
            "iqr": None,
            "std": None,
            "worst_decile": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    q25 = float(np.percentile(arr, 25))
    q50 = float(np.percentile(arr, 50))
    q75 = float(np.percentile(arr, 75))
    if arr.size >= 10:
        worst_count = max(1, int(np.ceil(arr.size * 0.1)))
        worst = np.sort(arr)[:worst_count]
        worst_decile = float(np.mean(worst))
    else:
        worst_decile = float(np.min(arr))
    return {
        "median": q50,
        "iqr": float(q75 - q25),
        "std": float(np.std(arr)),
        "worst_decile": worst_decile,
    }


def classify_failure_taxonomy(true_formula, discovered_formula, r2, mse):
    """Heuristic failure taxonomy for closed-loop mutation/operator analysis."""
    true_f = (true_formula or "").lower()
    disc_f = (discovered_formula or "").lower()

    if not disc_f:
        return "no_formula"

    if "exp(-" in true_f and "exp(" in disc_f and "exp(-" not in disc_f:
        return "exp_sign_error"

    if "/" in true_f and "/" not in disc_f:
        return "rational_denominator_missing"

    true_pows = [int(m.group(1)) for m in re.finditer(r"\*\*\s*(\d+)", true_f)]
    disc_pows = [int(m.group(1)) for m in re.finditer(r"\*\*\s*(\d+)", disc_f)]
    if true_pows:
        true_max = max(true_pows)
        disc_max = max(disc_pows) if disc_pows else 1
        if true_max >= 3 and disc_max < true_max:
            return "missing_high_order_terms"

    if ("sin(" in true_f or "cos(" in true_f) and not ("sin(" in disc_f or "cos(" in disc_f):
        return "periodic_structure_missing"

    if r2 is not None and r2 >= 0.8:
        return "near_miss_structural"
    if mse is not None and np.isfinite(mse) and mse > 1.0:
        return "high_error_instability"
    return "other_structural_failure"


def summarize_time_to_discovery(
    seed_runs,
    acceptable_r2=0.9,
    complexity_cap=20,
    exact_key="exact_match",
):
    """Compute time to first exact and first acceptable formula across seeded runs."""
    times_exact = []
    times_acceptable = []
    for run in seed_runs:
        t = run.get("time")
        if t is None or not np.isfinite(t):
            continue
        is_exact = bool(run.get(exact_key, False))
        r2 = run.get("r2")
        size = run.get("model_size")
        is_acceptable = is_exact or (
            r2 is not None
            and np.isfinite(r2)
            and r2 >= acceptable_r2
            and size is not None
            and size <= complexity_cap
        )
        if is_exact:
            times_exact.append(float(t))
        if is_acceptable:
            times_acceptable.append(float(t))

    return {
        "time_to_first_exact": (min(times_exact) if times_exact else None),
        "time_to_first_acceptable": (min(times_acceptable) if times_acceptable else None),
    }


def summarize_seed_runs(seed_runs):
    """Aggregate per-seed runs into protocol-compliant stability summaries."""
    valid = [run for run in seed_runs if run.get("r2") is not None]
    if not valid:
        return {
            "seed_count": len(seed_runs),
            "valid_seed_count": 0,
            "r2_stats": compute_stability_stats([]),
            "mse_stats": compute_stability_stats([]),
            "time_stats": compute_stability_stats([]),
            "exact_recovery_rate": None,
            "exact_recovery_stats": compute_stability_stats([]),
        }

    r2_vals = [run["r2"] for run in valid if np.isfinite(run["r2"])]
    mse_vals = [run["mse"] for run in valid if run.get("mse") is not None and np.isfinite(run["mse"])]
    time_vals = [run["time"] for run in valid if run.get("time") is not None and np.isfinite(run["time"])]
    exact_binary = [1.0 if run.get("exact_match") else 0.0 for run in valid]

    return {
        "seed_count": len(seed_runs),
        "valid_seed_count": len(valid),
        "r2_stats": compute_stability_stats(r2_vals),
        "mse_stats": compute_stability_stats(mse_vals),
        "time_stats": compute_stability_stats(time_vals),
        "exact_recovery_rate": float(np.mean(exact_binary)) if exact_binary else None,
        "exact_recovery_stats": compute_stability_stats(exact_binary),
    }


def fallback_estimator_predictions(run_result, eval_diag, *, split="test"):
    """Return protected estimator predictions when display-formula scoring fails."""
    if not isinstance(run_result, dict) or not isinstance(eval_diag, dict):
        return None, eval_diag
    key = "y_pred_full" if split == "full" else "y_pred_test"
    pred = run_result.get(key)
    if pred is None:
        return None, eval_diag
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    if pred.size == 0 or not np.all(np.isfinite(pred)):
        return None, eval_diag
    updated = dict(eval_diag)
    updated["display_formula_failed"] = True
    updated["display_formula_reason"] = eval_diag.get("reason")
    updated["reason"] = "protected_estimator_prediction"
    updated["ok"] = True
    return pred, updated


def apply_run_budget(est_params, timeout_budget):
    """Keep estimator-internal adaptive compute inside the run budget."""
    params = dict(est_params)
    budget = int(max(1, round(float(timeout_budget))))
    params["timeout"] = budget
    params["max_compute_budget"] = budget
    params["min_compute_budget"] = min(int(params.get("min_compute_budget", 10) or 10), budget)
    return params


def specialist_metadata_from_estimator(estimator):
    """Extract specialist/composition/boosting diagnostics from a GlassboxRegressor."""
    diagnostics = getattr(estimator, "blackbox_diagnostics_", {}) or {}
    candidate_screening = diagnostics.get("candidate_screening", {}) if isinstance(diagnostics, dict) else {}
    return {
        "specialist_track": getattr(estimator, "specialist_track_", None),
        "has_composed_seeds": bool(getattr(estimator, "has_composed_seeds_", False)),
        "composition_candidates_accepted": bool(getattr(estimator, "composition_candidates_accepted_", False)),
        "composition_candidate_count": int(getattr(estimator, "composition_candidate_count_", 0) or 0),
        "composition_seeded_evolution": bool(getattr(estimator, "composition_seeded_evolution_", False)),
        "composition_won_final_selection": bool(getattr(estimator, "composition_won_final_selection_", False)),
        "composition_improved_mse": bool(getattr(estimator, "composition_improved_mse_", False)),
        "boosting_attempted": bool(getattr(estimator, "boosting_attempted_", False)),
        "boosting_improved": bool(getattr(estimator, "boosting_improved_", False)),
        "boosting_stage_count": len(getattr(estimator, "boosting_stages_", []) or []),
        "boosting_diagnostics": getattr(estimator, "boosting_diagnostics_", None),
        "residual_stage_guard": getattr(estimator, "_residual_stage_guard_", None),
        "phase_timings": dict(getattr(estimator, "phase_timings_", {}) or {}),
        "exact_match_diagnostics": getattr(estimator, "fast_path_exact_match_diagnostics_", None),
        "formula_eval_count": int(getattr(estimator, "formula_eval_count_", 0) or 0),
        "formula_eval_cache_hits": int(getattr(estimator, "formula_eval_cache_hits_", 0) or 0),
        "formula_eval_cache_size": len(getattr(estimator, "_formula_eval_cache_", {}) or {}),
        "specialist_vault": (
            getattr(estimator, "specialist_vault_", None).to_dict()
            if getattr(estimator, "specialist_vault_", None) is not None
            else None
        ),
        "inception_round_count": len(getattr(estimator, "inception_rounds_", []) or []),
        "inception_diagnostics": getattr(estimator, "inception_diagnostics_", None),
        "specialist_diagnostics": (
            candidate_screening.get("specialist_screening")
            if isinstance(candidate_screening, dict)
            else None
        ),
        "specialist_composition_screening": (
            diagnostics.get("specialist_composition_screening")
            if isinstance(diagnostics, dict)
            else None
        ),
    }
