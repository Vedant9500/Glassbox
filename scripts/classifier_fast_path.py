"""
Classifier-Guided Fast Path for Symbolic Regression

When the curve classifier predicts operators with high confidence,
skip Phase 1 evolution entirely and directly run regression.

This can reduce solve time from ~300s to <10s for well-predicted formulas.
"""

import re
import math
import threading
import importlib
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import permutations
import numpy as np
import torch
from typing import Any, Dict, List, Tuple, Optional

from glassbox.model_registry import DEFAULT_CURVE_CLASSIFIER_PATH
from glassbox.sr.operations.meta_ops import get_constant_symbol, normalize_formula_ascii
from glassbox.sr.fpip_v2 import build_fpip_v2_from_fast_path, validate_fpip_v2_payload

# Thread-safe CUDA warning state
_warned_no_cuda = False
_cuda_warning_lock = threading.Lock()

# Pre-compiled regex patterns for performance
_FREQ_SIN_PATTERN = re.compile(r'sin\(([0-9.]+)\*?x', re.IGNORECASE)
_FREQ_COS_PATTERN = re.compile(r'cos\(([0-9.]+)\*?x', re.IGNORECASE)
_POWER_PATTERN = re.compile(r'x\^([0-9.]+)', re.IGNORECASE)

DEFAULT_EXACT_MATCH_MIN_GPU_WORK = 250_000
DEFAULT_EXACT_MATCH_MAX_COMBOS = 50_000


@lru_cache(maxsize=1)
def _load_cpp_core() -> Tuple[Optional[Any], Optional[str]]:
    """Load the optional C++ extension, returning a diagnostic instead of raising."""
    import sys
    from pathlib import Path as _Path

    cpp_dir = _Path(__file__).resolve().parent.parent / 'glassbox' / 'sr' / 'cpp'
    if str(cpp_dir) not in sys.path:
        sys.path.insert(0, str(cpp_dir))

    errors: List[str] = []
    for module_name in ('_core', 'glassbox.sr.cpp._core'):
        try:
            return importlib.import_module(module_name), None
        except ImportError as exc:
            errors.append(f"{module_name}: {exc}")

    built_extensions = sorted(p.name for p in cpp_dir.glob('_core.*'))
    active_abi = getattr(sys.implementation, "cache_tag", "unknown ABI")
    if built_extensions:
        found = ", ".join(built_extensions)
        return None, f"C++ _core extension unavailable for active ABI {active_abi}; found {found}"
    return None, f"C++ _core extension unavailable for active ABI {active_abi} ({'; '.join(errors)})"


def _lasso_coordinate_descent_python(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.1,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> np.ndarray:
    """Small NumPy fallback for the optional C++ coordinate-descent solver."""
    X_np = np.asarray(X, dtype=np.float64)
    y_np = np.asarray(y, dtype=np.float64).reshape(-1)
    if X_np.ndim != 2:
        raise ValueError("X must be a 2D array")
    if X_np.shape[0] != y_np.shape[0]:
        raise ValueError("X and y row counts must match")

    if alpha <= 0.0:
        coeffs, _, _, _ = np.linalg.lstsq(X_np, y_np, rcond=None)
        return np.asarray(coeffs, dtype=np.float64)

    _, n_features = X_np.shape
    coeffs = np.zeros(n_features, dtype=np.float64)
    residual = y_np.copy()
    col_norms = np.sum(X_np * X_np, axis=0)

    for _ in range(max_iter):
        max_delta = 0.0
        for j in range(n_features):
            norm_j = col_norms[j]
            if norm_j <= 1e-18:
                continue

            old = coeffs[j]
            residual += X_np[:, j] * old
            rho = float(np.dot(X_np[:, j], residual))
            coeffs[j] = float(soft_threshold(np.asarray(rho), alpha) / norm_j)
            residual -= X_np[:, j] * coeffs[j]
            max_delta = max(max_delta, abs(coeffs[j] - old))

        if max_delta < tol:
            break

    return coeffs


def _with_derived_predictions(predictions: Dict[str, float]) -> Dict[str, float]:
    """Return predictions augmented with derived periodic/exponential/polynomial keys."""
    derived = dict(predictions)
    periodic_prob = max(derived.get('sin', 0.0), derived.get('cos', 0.0))
    exponential_prob = max(derived.get('exp', 0.0), derived.get('log', 0.0))
    polynomial_prob = max(derived.get('power', 0.0), derived.get('identity', 0.0))

    derived.setdefault('periodic', periodic_prob)
    derived.setdefault('exponential', exponential_prob)
    derived.setdefault('polynomial', polynomial_prob)
    return derived


def _prediction_uncertainty_metrics(predictions: Dict[str, float]) -> Dict[str, float | bool | None]:
    """Summarize classifier confidence with entropy and top-1/top-2 margin."""
    metrics: Dict[str, float | bool | None] = {
        'prediction_entropy': None,
        'prediction_margin': None,
        'prediction_top1': None,
        'prediction_top2': None,
        'prediction_uncertain': False,
    }

    if not predictions:
        return metrics

    probs = np.asarray([float(p) for p in predictions.values() if np.isfinite(p) and p > 0.0], dtype=np.float64)
    if probs.size == 0:
        return metrics

    total = float(np.sum(probs))
    if total <= 0.0:
        return metrics

    probs = probs / total
    sorted_probs = np.sort(probs)[::-1]
    top1 = float(sorted_probs[0])
    top2 = float(sorted_probs[1]) if sorted_probs.size > 1 else 0.0
    if sorted_probs.size > 1:
        entropy = float(-np.sum(sorted_probs * np.log(sorted_probs + 1e-12)) / np.log(sorted_probs.size))
    else:
        entropy = 0.0

    margin = top1 - top2
    metrics['prediction_entropy'] = entropy
    metrics['prediction_margin'] = margin
    metrics['prediction_top1'] = top1
    metrics['prediction_top2'] = top2
    metrics['prediction_uncertain'] = entropy > 0.8 or margin < 0.1
    return metrics


def _display_candidate_score(
    formula: str,
    x: np.ndarray,
    y: np.ndarray,
    *,
    raw_mse: Optional[float] = None,
    fit_mse: Optional[float] = None,
    holdout_mse: Optional[float] = None,
    residual_diagnostics: Optional[Dict[str, Any]] = None,
    complexity: Optional[int] = None,
    n_terms: Optional[int] = None,
    postprocess: bool = False,
) -> Dict[str, Any]:
    """Score a formula with the shared display-aware governor when available."""
    X = np.asarray(x, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    try:
        from scripts import benchmark_common as bc  # type: ignore
    except Exception:
        bc = None

    if bc is not None and hasattr(bc, "score_display_candidate"):
        return bc.score_display_candidate(
            formula,
            X,
            y,
            raw_mse=raw_mse,
            fit_mse=fit_mse,
            holdout_mse=holdout_mse,
            residual_diagnostics=residual_diagnostics,
            complexity=complexity,
            n_terms=n_terms,
            postprocess=postprocess,
        )

    display_mse = raw_mse if raw_mse is not None and np.isfinite(raw_mse) else fit_mse
    comp = int(complexity) if complexity is not None else max(1, formula.count("+") + formula.count("-") + formula.count("*") + formula.count("/") + 1)
    terms = int(n_terms) if n_terms is not None else comp
    risk = min(1.0, 0.05 * formula.count("_signed_power") + 0.012 * max(0, comp - 12))
    base = float(display_mse) if display_mse is not None and np.isfinite(display_mse) else float("inf")
    score = base + 1e-4 * max(0, comp - 8) + 5e-5 * max(0, terms - 6) + 0.01 * risk
    return {
        "formula": formula,
        "formula_original": formula,
        "score": float(score),
        "display_mse": base if np.isfinite(base) else None,
        "raw_mse": raw_mse,
        "holdout_mse": holdout_mse,
        "complexity": comp,
        "n_terms": terms,
        "risk_score": risk,
        "raw_display_drift_rel": None,
        "residual_suspicious": False,
        "postprocess_guard": None,
        "display_eval_ok": np.isfinite(base),
    }


def _empty_residual_diagnostics() -> Dict[str, Any]:
    return {
        'residual_mse': None,
        'residual_skewness': None,
        'residual_excess_kurtosis': None,
        'residual_spectral_peak_ratio': None,
        'residual_holdout_edge_mse': None,
        'residual_holdout_core_mse': None,
        'residual_holdout_ratio': None,
        'residual_suspicious': False,
    }


def _normalize_priors(priors: List[float]) -> List[float]:
    cleaned = [max(0.0, float(p)) for p in priors]
    total = sum(cleaned)
    if total <= 0.0:
        return [0.25, 0.25, 0.25, 0.25]
    return [p / total for p in cleaned]


def _classifier_prior_trust_from_uncertainty(uncertainty: Optional[Dict[str, Any]]) -> float:
    """Map uncertainty diagnostics to trust in classifier-guided priors in [0, 1]."""
    if not isinstance(uncertainty, dict):
        return 1.0

    entropy = uncertainty.get('prediction_entropy')
    margin = uncertainty.get('prediction_margin')
    uncertain_flag = bool(uncertainty.get('prediction_uncertain', False))

    trust_entropy = 1.0
    if entropy is not None:
        try:
            ent = float(entropy)
            if np.isfinite(ent):
                trust_entropy = float(np.clip(1.0 - ent, 0.0, 1.0))
        except Exception:
            trust_entropy = 1.0

    trust_margin = 1.0
    if margin is not None:
        try:
            mar = float(margin)
            if np.isfinite(mar):
                trust_margin = float(np.clip(mar / 0.35, 0.0, 1.0))
        except Exception:
            trust_margin = 1.0

    trust = min(trust_entropy, trust_margin)
    if uncertain_flag:
        trust *= 0.5

    return float(np.clip(trust, 0.0, 1.0))


def _blend_priors_with_uniform(base_priors: List[float], trust: float) -> List[float]:
    base = _normalize_priors(base_priors)
    t = float(np.clip(trust, 0.0, 1.0))
    uniform = [0.25, 0.25, 0.25, 0.25]
    blended = [t * b + (1.0 - t) * u for b, u in zip(base, uniform)]
    return _normalize_priors(blended)


def _resolve_device(device: Optional[str] = None) -> torch.device:
    """Resolve device string to torch.device, with thread-safe CUDA fallback warning."""
    global _warned_no_cuda
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        with _cuda_warning_lock:
            if not _warned_no_cuda:
                print("CUDA requested but not available; falling back to CPU.")
                _warned_no_cuda = True
        return torch.device("cpu")

    return resolved


def _estimate_exact_match_work(n_samples: int, n_basis: int, max_terms: int) -> int:
    """Estimate batched exact-match work for backend selection."""
    max_r = min(int(max_terms), 3)
    if max_r < 2 or n_samples <= 0 or n_basis <= 0:
        return 0

    combos = 0
    for r in range(2, max_r + 1):
        if n_basis >= r:
            combos += math.comb(n_basis, r)
    return int(n_samples * combos)


def _select_exact_match_torch_device(
    backend: str,
    device: Optional[str],
    estimated_work: int,
    min_gpu_work: int,
) -> Tuple[Optional[torch.device], Dict[str, Any]]:
    """Select the torch device for exact-match search and explain the choice."""
    backend_norm = (backend or "auto").strip().lower()
    diagnostics: Dict[str, Any] = {
        "backend_requested": backend_norm,
        "estimated_work": int(estimated_work),
        "min_gpu_work": int(min_gpu_work),
        "torch_used": False,
        "gpu_used": False,
        "resolved_device": None,
        "fallback_reason": None,
    }

    if backend_norm in {"numpy", "np", "off", "none"}:
        diagnostics["fallback_reason"] = "torch_backend_disabled"
        return None, diagnostics

    if backend_norm in {"cuda", "gpu", "torch_cuda"}:
        if not torch.cuda.is_available():
            diagnostics["fallback_reason"] = "cuda_unavailable"
            return None, diagnostics
        selected = torch.device("cuda")
    elif backend_norm in {"cpu", "torch_cpu"}:
        selected = torch.device("cpu")
    elif backend_norm in {"torch"}:
        selected = _resolve_device(device)
    elif backend_norm == "auto":
        requested = _resolve_device(device)
        explicit_cuda = device is not None and requested.type == "cuda"
        if explicit_cuda and estimated_work >= int(min_gpu_work):
            selected = requested
        else:
            selected = torch.device("cpu")
            if explicit_cuda:
                diagnostics["fallback_reason"] = "below_gpu_work_threshold"
    else:
        diagnostics["fallback_reason"] = f"unknown_backend:{backend_norm}"
        selected = torch.device("cpu")

    diagnostics["torch_used"] = True
    diagnostics["gpu_used"] = selected.type == "cuda"
    diagnostics["resolved_device"] = str(selected)
    return selected, diagnostics


def _join_formula_terms(terms: List[str]) -> str:
    """Join symbolic terms and normalize them into ASCII-safe math."""
    filtered_terms = [term for term in terms if term]
    formula = " + ".join(filtered_terms) if filtered_terms else "0"
    return normalize_formula_ascii(formula.replace("+ -", "- "))


def _format_affine_formula(base_expr: str, scale: float, offset: float) -> str:
    """Format y ~= offset + scale * base_expr with readable symbolic coefficients."""
    terms: List[str] = []

    if abs(scale) > 1e-10:
        if abs(scale - 1.0) < 0.01:
            terms.append(base_expr)
        elif abs(scale + 1.0) < 0.01:
            terms.append(f"-({base_expr})")
        else:
            terms.append(f"{get_constant_symbol(scale, 0.05)}*({base_expr})")

    if abs(offset) > 1e-10:
        terms.append(get_constant_symbol(offset, 0.05))

    return _join_formula_terms(terms)


def _candidate_match_tolerance(y: np.ndarray) -> float:
    scale = max(float(np.var(y)), float(np.mean(y ** 2)), 1.0)
    return max(1e-10, 1e-8 * scale)


def _format_regression_term(name: str, coef: float) -> Optional[str]:
    """Format one linear-regression term using the project coefficient snapping rules."""
    if abs(coef) < 1e-6:
        return None
    if name == "1":
        coef_sym = get_constant_symbol(coef, 0.05)
        return None if coef_sym in {"0", "0.0"} else coef_sym
    if abs(coef - 1.0) < 0.01:
        return name
    if abs(coef + 1.0) < 0.01:
        return f"-{name}"
    if abs(coef - round(coef)) < 0.01 and abs(coef) < 100:
        return f"{int(round(coef))}*{name}"
    coef_sym = get_constant_symbol(coef, 0.05)
    if coef_sym in {"0", "0.0"}:
        return None
    return f"{coef_sym}*{name}"


def _find_exact_polynomial_match(
    x: np.ndarray,
    y: np.ndarray,
    basis_names: List[str],
    *,
    max_degree: int,
    tolerance: Optional[float] = None,
) -> Optional[Tuple[str, float, np.ndarray]]:
    """Recover exact univariate polynomials before mixed bases can overfit them."""
    if x.ndim == 2 and x.shape[1] == 1:
        x_flat = x[:, 0]
    elif x.ndim == 1:
        x_flat = x
    else:
        return None

    y_flat = y.reshape(-1)
    if x_flat.size != y_flat.size or x_flat.size < 3:
        return None

    y_var = max(float(np.var(y_flat)), 1e-12)
    exact_tol = tolerance if tolerance is not None else max(1e-10, y_var * 1e-12)
    max_degree = max(1, min(int(max_degree), 10))

    for degree in range(1, max_degree + 1):
        cols = [np.ones_like(x_flat, dtype=np.float64)]
        cols.extend((x_flat.astype(np.float64) ** p) for p in range(1, degree + 1))
        design = np.column_stack(cols)
        try:
            coeffs, _, _, _ = np.linalg.lstsq(design, y_flat, rcond=None)
        except (np.linalg.LinAlgError, ValueError):
            continue

        pred = design @ coeffs
        mse = float(np.mean((pred - y_flat) ** 2))
        if not np.isfinite(mse) or mse > exact_tol:
            continue

        terms: List[str] = []
        full_coeffs = np.zeros(len(basis_names), dtype=np.float64)
        for power, coef in enumerate(coeffs):
            if abs(coef) < 1e-8:
                continue
            if power == 0:
                name = "1"
                basis_key = "1"
            elif power == 1:
                name = "x"
                basis_key = "x"
            else:
                name = f"x**{power}"
                basis_key = f"x^{power}"

            term = _format_regression_term(name, float(coef))
            if term and term != "0":
                terms.append(term)
            if basis_key in basis_names:
                full_coeffs[basis_names.index(basis_key)] = float(coef)

        if not terms:
            continue
        if terms and terms[0] == "0":
            terms = terms[1:]
        formula = _join_formula_terms(terms)
        return formula, mse, full_coeffs

    return None


def _evaluate_formula_values(formula: str, x_np: np.ndarray) -> Optional[np.ndarray]:
    """Evaluate a formula string on numpy inputs using SymPy/lambdify."""
    if not formula:
        return None

    normalized = normalize_formula_ascii(formula)
    if not normalized or normalized in {"N/A", "ERROR", "?"}:
        return None

    try:
        free_symbol_names, const_value, func = _compile_formula_evaluator(normalized)

        if const_value is not None:
            return np.full(x_np.shape[0], const_value, dtype=np.float64)

        if x_np.ndim == 1:
            x_columns = [x_np.reshape(-1)]
        elif x_np.ndim == 2:
            x_columns = [x_np[:, i] for i in range(x_np.shape[1])]
        else:
            return None

        if len(free_symbol_names) > len(x_columns):
            return None

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            y_pred = func(*x_columns[:len(free_symbol_names)])
            
        y_arr = np.asarray(y_pred, dtype=np.float64)
        if y_arr.shape == ():
            y_arr = np.full(x_np.shape[0], float(y_arr), dtype=np.float64)
        return y_arr.reshape(-1)
    except Exception:
        return None


def _safe_numpy_power(x, p):
    """
    Safe power function matching C++ power_sign_blend logic.
    Supports fractional powers of negative numbers via signed power: sign(x) * |x|^p.
    If p is an even integer, returns |x|^p (parity-preserving).
    """
    x = np.asarray(x)
    p = np.asarray(p)
    abs_x = np.abs(x) + 1e-15
    res = np.power(abs_x, p)
    
    # Parity check for even integers
    p_round = np.round(p)
    is_even = (np.abs(p - p_round) < 1e-6) & (p_round.astype(np.int64) % 2 == 0)
    
    if np.isscalar(is_even):
        return res if is_even else np.sign(x) * res
    return np.where(is_even, res, np.sign(x) * res)


def _safe_numpy_log(x, base=None):
    """NumPy log that also supports SymPy's log(x, base) lambdify output."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(x)
        if base is not None:
            out = out / np.log(base)
    return out


@lru_cache(maxsize=256)
def _compile_formula_evaluator(normalized_formula: str) -> Tuple[Tuple[str, ...], Optional[float], Optional[Any]]:
    import sympy as sp
    from sympy.parsing.sympy_parser import (
        convert_xor,
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    transformations = standard_transformations + (convert_xor, implicit_multiplication_application)
    local_dict = {
        "Piecewise": sp.Piecewise,
        "Eq": sp.Eq,
        "Abs": sp.Abs,
        "sign": sp.sign,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "exp": sp.exp,
        "log": sp.log,
        "sqrt": sp.sqrt,
        "pi": sp.pi,
        "E": sp.E,
        "e": sp.E,
    }
    expr = parse_expr(normalized_formula, local_dict=local_dict, transformations=transformations, evaluate=False)
    free_syms = sorted(expr.free_symbols, key=lambda sym: sym.name)

    if not free_syms:
        return tuple(), float(expr), None

    # SymPy lambdifies ``x**1.5`` as NumPy's native power operator, which
    # produces NaN for negative x. The C++ engine and fast-path fractional
    # basis use signed real powers instead: sign(x) * abs(x)**p.
    signed_pow = sp.Function("_gb_signed_pow")
    expr = expr.replace(
        lambda node: isinstance(node, sp.Pow),
        lambda node: signed_pow(node.base, node.exp),
    )

    # Inject safe power/log into lambdify context.
    modules = [
        {
            "_gb_signed_pow": _safe_numpy_power,
            "pow": _safe_numpy_power,
            "Pow": _safe_numpy_power,
            "log": _safe_numpy_log,
        },
        "numpy",
    ]
    func = sp.lambdify(free_syms, expr, modules=modules)
    return tuple(sym.name for sym in free_syms), None, func


def _residual_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    x_np: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Summarize residual structure for fast-path quality checks."""
    diagnostics = _empty_residual_diagnostics()

    try:
        y_true_arr = np.asarray(y_true, dtype=np.float64).reshape(-1)
        y_pred_arr = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    except Exception:
        return diagnostics

    if y_true_arr.shape != y_pred_arr.shape:
        return diagnostics

    mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    if mask.sum() < 10:
        return diagnostics

    y_true_valid = y_true_arr[mask]
    y_pred_valid = y_pred_arr[mask]
    residual = y_true_valid - y_pred_valid
    diagnostics['residual_mse'] = float(np.mean(residual ** 2))
    diagnostics['y_variance'] = float(np.var(y_true_valid))

    centered = residual - residual.mean()
    std = float(np.std(centered))
    if std > 1e-12:
        z = centered / std
        diagnostics['residual_skewness'] = float(np.mean(z ** 3))
        diagnostics['residual_excess_kurtosis'] = float(np.mean(z ** 4) - 3.0)

    if residual.size >= 8:
        fft_vals = np.fft.rfft(centered)
        magnitudes = np.abs(fft_vals[1:])
        if magnitudes.size > 0:
            peak = float(np.max(magnitudes))
            median = float(np.median(magnitudes)) if np.any(magnitudes) else 0.0
            diagnostics['residual_spectral_peak_ratio'] = peak / max(median, 1e-12)

    if x_np is not None:
        try:
            x_arr = np.asarray(x_np)
            if x_arr.ndim == 1 or (x_arr.ndim == 2 and x_arr.shape[1] == 1):
                x_flat = x_arr.reshape(-1)
                if x_flat.shape[0] != y_true_arr.shape[0]:
                    return diagnostics
                x_flat = x_flat[mask]
                order = np.argsort(x_flat)
                n_total = order.size
                holdout_n = max(1, int(round(n_total * 0.1)))
                if n_total >= 20 and 2 * holdout_n < n_total:
                    edge_idx = np.concatenate([order[:holdout_n], order[-holdout_n:]])
                    core_idx = order[holdout_n:-holdout_n]
                    edge_mse = float(np.mean((y_true_valid[edge_idx] - y_pred_valid[edge_idx]) ** 2))
                    core_mse = float(np.mean((y_true_valid[core_idx] - y_pred_valid[core_idx]) ** 2))
                    diagnostics['residual_holdout_edge_mse'] = edge_mse
                    diagnostics['residual_holdout_core_mse'] = core_mse
                    diagnostics['residual_holdout_ratio'] = edge_mse / max(core_mse, 1e-12)
        except Exception:
            pass

    suspicious = False
    if diagnostics['residual_spectral_peak_ratio'] is not None:
        suspicious = suspicious or diagnostics['residual_spectral_peak_ratio'] > 8.0
    if diagnostics['residual_holdout_ratio'] is not None:
        suspicious = suspicious or diagnostics['residual_holdout_ratio'] > 2.0
    if diagnostics['residual_skewness'] is not None:
        suspicious = suspicious or abs(diagnostics['residual_skewness']) > 1.0
    if diagnostics['residual_excess_kurtosis'] is not None:
        suspicious = suspicious or diagnostics['residual_excess_kurtosis'] > 3.0

    diagnostics['residual_suspicious'] = suspicious
    return diagnostics


def _maybe_match_easy_multivariate_formula(
    x: np.ndarray,
    y: np.ndarray,
) -> Optional[Tuple[str, float, Dict[str, Any]]]:
    """Check a few exact low-complexity multivariate templates before basis expansion."""
    if x.ndim != 2 or x.shape[1] < 2:
        return None

    y = y.flatten()
    tol = _candidate_match_tolerance(y)
    best_match: Optional[Tuple[str, float, Dict[str, Any]]] = None

    def try_candidate(base_expr: str, base_values: np.ndarray, template_name: str) -> None:
        nonlocal best_match
        if not np.all(np.isfinite(base_values)):
            return

        X = np.column_stack([np.ones(len(y)), base_values])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            return

        offset, scale = float(coeffs[0]), float(coeffs[1])
        y_pred = X @ coeffs
        mse = float(np.mean((y - y_pred) ** 2))
        if mse >= tol:
            return

        formula = _format_affine_formula(base_expr, scale, offset)
        details = {
            'coefficients': np.array([offset, scale]),
            'basis_names': [base_expr],
            'n_nonzero': int(abs(offset) > 1e-10) + int(abs(scale) > 1e-10),
            'exact_match': True,
            'template_match': template_name,
            'template_tolerance': tol,
        }

        if best_match is None or mse < best_match[1]:
            best_match = (formula, mse, details)

    n_vars = x.shape[1]

    if n_vars >= 4:
        for a, b, c, d in permutations(range(n_vars), 4):
            dist = np.sqrt((x[:, a] - x[:, b]) ** 2 + (x[:, c] - x[:, d]) ** 2)
            try_candidate(
                f"sqrt((x{a}-x{b})^2 + (x{c}-x{d})^2)",
                dist,
                "distance_2d",
            )

    if n_vars >= 3:
        for a, b, c in permutations(range(n_vars), 3):
            denom_sq = x[:, c] ** 2
            if np.any(np.abs(denom_sq) < 1e-12):
                continue
            inside = 1.0 - (x[:, b] ** 2) / denom_sq
            if np.any(inside <= 1e-10):
                continue

            relativistic = x[:, a] / np.sqrt(inside)
            try_candidate(
                f"x{a}/sqrt(1-x{b}^2/x{c}^2)",
                relativistic,
                "relativistic_mass",
            )

    if n_vars >= 4:
        for a, b, c, d in permutations(range(n_vars), 4):
            phase = x[:, b] * x[:, c]
            cosine = np.cos(phase)
            envelope = x[:, a] * (cosine + x[:, d] * cosine ** 2)
            try_candidate(
                f"x{a}*(cos(x{b}*x{c})+x{d}*cos(x{b}*x{c})^2)",
                envelope,
                "cosine_envelope",
            )

    return best_match


def _format_precise_number(value: float) -> str:
    """Format a floating constant tightly enough for template validation."""
    value = float(value)
    if abs(value) < 1e-12:
        return "0"
    nearest = round(value)
    if abs(value - nearest) < 1e-10 and abs(nearest) < 10_000:
        return str(int(nearest))
    return f"{value:.12g}"


def _format_linear_x_formula(slope: float, intercept: float, *, precise: bool = False) -> str:
    """Format slope*x + intercept for direct transform templates."""
    if precise:
        terms: List[str] = []
        if abs(slope) > 1e-12:
            slope_text = _format_precise_number(abs(slope))
            if abs(abs(slope) - 1.0) < 1e-12:
                term = "x"
            else:
                term = f"{slope_text}*x"
            terms.append(f"-{term}" if slope < 0 else term)
        if abs(intercept) > 1e-12:
            const_text = _format_precise_number(abs(intercept))
            terms.append(f"-{const_text}" if intercept < 0 else const_text)
        return _join_formula_terms(terms)

    terms: List[str] = []
    x_term = _format_regression_term("x", float(slope))
    if x_term:
        terms.append(x_term)
    const_term = _format_regression_term("1", float(intercept))
    if const_term:
        terms.append(const_term)
    return _join_formula_terms(terms)


def _maybe_match_univariate_transform_template(
    x: np.ndarray,
    y: np.ndarray,
    predictions: Dict[str, float],
    *,
    tolerance: Optional[float] = None,
) -> Optional[Tuple[str, float, Dict[str, Any]]]:
    """Recover simple log/exp transform identities before broad sparse bases."""
    if x.ndim == 2:
        if x.shape[1] != 1:
            return None
        x_flat = x[:, 0]
        x_eval = x
    elif x.ndim == 1:
        x_flat = x.reshape(-1)
        x_eval = x.reshape(-1, 1)
    else:
        return None

    y_flat = np.asarray(y, dtype=np.float64).reshape(-1)
    x_flat = np.asarray(x_flat, dtype=np.float64).reshape(-1)
    if x_flat.size != y_flat.size or x_flat.size < 4:
        return None
    if not (np.all(np.isfinite(x_flat)) and np.all(np.isfinite(y_flat))):
        return None

    derived = _with_derived_predictions(predictions or {})
    should_try_log = derived.get("log", 0.0) >= 0.25 or derived.get("exponential", 0.0) >= 0.25
    should_try_exp = derived.get("exp", 0.0) >= 0.25 or derived.get("exponential", 0.0) >= 0.25
    if not (should_try_log or should_try_exp):
        return None

    tol = tolerance if tolerance is not None else _candidate_match_tolerance(y_flat)
    accept_tol = min(1e-6, max(1e-10, float(tol)))
    best_match: Optional[Tuple[str, float, Dict[str, Any]]] = None

    def validate_candidate(
        formula: str,
        pred: np.ndarray,
        template_name: str,
        params: Dict[str, float],
    ) -> None:
        nonlocal best_match
        if not np.all(np.isfinite(pred)):
            return
        raw_mse = float(np.mean((y_flat - pred.reshape(-1)) ** 2))
        if not np.isfinite(raw_mse) or raw_mse > accept_tol:
            return

        governed = _display_candidate_score(
            formula,
            x_eval,
            y_flat,
            raw_mse=raw_mse,
            fit_mse=raw_mse,
            complexity=3,
            n_terms=1,
            postprocess=True,
        )
        display_mse = governed.get("display_mse")
        if display_mse is None:
            return
        display_mse_f = float(display_mse)
        if not np.isfinite(display_mse_f) or display_mse_f > accept_tol:
            return

        display_formula = str(governed.get("formula") or formula)
        n_nonzero = max(1, len(params))
        details = {
            "coefficients": np.array(list(params.values()), dtype=np.float64),
            "basis_names": list(params.keys()),
            "n_nonzero": n_nonzero,
            "exact_match": True,
            "template_match": template_name,
            "template_tolerance": accept_tol,
            "candidate_governor": governed,
            "candidate_formulas": [{
                "formula": display_formula,
                "mse": display_mse_f,
                "score": float(governed.get("score", display_mse_f)),
                "n_nonzero": n_nonzero,
                "active_terms": [template_name],
                "alpha": 0.0,
                "raw_mse": raw_mse,
                "display_mse": display_mse_f,
                "governor": governed,
            }],
        }
        if best_match is None or display_mse_f < best_match[1]:
            best_match = (display_formula, display_mse_f, details)

    def fit_affine(target: np.ndarray) -> Optional[Tuple[float, float, np.ndarray]]:
        design = np.column_stack([x_flat, np.ones_like(x_flat)])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
        except (np.linalg.LinAlgError, ValueError):
            return None
        slope, intercept = float(coeffs[0]), float(coeffs[1])
        fitted = design @ coeffs
        return slope, intercept, fitted

    if should_try_log:
        exp_y = np.exp(np.clip(y_flat, -700.0, 700.0))
        if np.all(np.isfinite(exp_y)):
            fit = fit_affine(exp_y)
            if fit is not None:
                slope, intercept, inside = fit
                if abs(slope) > 1e-12 and np.all(inside > 1e-12):
                    inner = _format_linear_x_formula(slope, intercept, precise=True)
                    validate_candidate(
                        f"log({inner})",
                        np.log(inside),
                        "log_affine_direct",
                        {"slope": slope, "intercept": intercept},
                    )

    if should_try_exp:
        for sign, sign_prefix, target_values in (
            (1.0, "", y_flat),
            (-1.0, "-", -y_flat),
        ):
            if np.all(target_values > 1e-12):
                fit = fit_affine(np.log(target_values))
                if fit is not None:
                    slope, intercept, log_fit = fit
                    pred = sign * np.exp(log_fit)
                    inner = _format_linear_x_formula(slope, intercept, precise=True)
                    validate_candidate(
                        f"{sign_prefix}exp({inner})",
                        pred,
                        "exp_affine_direct",
                        {"sign": sign, "slope": slope, "intercept": intercept},
                    )

        order = np.argsort(x_flat)
        x_sorted = x_flat[order]
        y_sorted = y_flat[order]
        dx = np.diff(x_sorted)
        if dx.size >= 3 and np.all(np.abs(dx) > 1e-12):
            uniformity = float(np.std(dx) / max(abs(float(np.mean(dx))), 1e-12))
            if uniformity < 1e-3:
                denom = y_sorted[:-2] + y_sorted[2:] - 2.0 * y_sorted[1:-1]
                valid = np.abs(denom) > 1e-12
                if np.any(valid):
                    offsets = (y_sorted[:-2] * y_sorted[2:] - y_sorted[1:-1] ** 2) / denom
                    offsets = offsets[np.isfinite(offsets) & valid]
                    if offsets.size:
                        offset = float(np.median(offsets))
                        shifted = y_flat - offset
                        for sign, sign_prefix in ((1.0, ""), (-1.0, "-")):
                            signed_shifted = sign * shifted
                            if not np.all(signed_shifted > 1e-12):
                                continue
                            fit = fit_affine(np.log(signed_shifted))
                            if fit is None:
                                continue
                            slope, intercept, log_fit = fit
                            pred = offset + sign * np.exp(log_fit)
                            inner = _format_linear_x_formula(slope, intercept, precise=True)
                            offset_text = _format_precise_number(offset)
                            formula = _join_formula_terms([offset_text, f"{sign_prefix}exp({inner})"])
                            validate_candidate(
                                formula,
                                pred,
                                "shifted_exp_affine",
                                {
                                    "offset": offset,
                                    "sign": sign,
                                    "slope": slope,
                                    "intercept": intercept,
                                },
                            )

    return best_match


def _format_linear_combo_formula(
    names: List[str],
    coeffs: np.ndarray,
    *,
    threshold: float = 1e-8,
) -> str:
    """Format intercept + sum(coeff_i * name_i)."""
    terms: List[str] = []
    coeffs = np.asarray(coeffs, dtype=np.float64).reshape(-1)
    if coeffs.size:
        const_term = _format_regression_term("1", float(coeffs[0]))
        if const_term:
            terms.append(const_term)
    for name, coef in zip(names, coeffs[1:]):
        if abs(float(coef)) < threshold:
            continue
        term = _format_regression_term(name, float(coef))
        if term:
            terms.append(term)
    return _join_formula_terms(terms)


def _univariate_component_library(x_flat: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """Small component library for decomposition probes."""
    x_flat = np.asarray(x_flat, dtype=np.float64).reshape(-1)
    components: List[Tuple[str, np.ndarray]] = [
        ("x", x_flat),
        ("x**2", x_flat ** 2),
        ("x**3", x_flat ** 3),
        ("sin(x)", np.sin(x_flat)),
        ("cos(x)", np.cos(x_flat)),
        ("sin(2*x)", np.sin(2.0 * x_flat)),
        ("cos(2*x)", np.cos(2.0 * x_flat)),
        ("1/(1+x**2)", 1.0 / (1.0 + x_flat ** 2)),
        ("x/(1+x**2)", x_flat / (1.0 + x_flat ** 2)),
        ("log(Abs(x)+1)", np.log(np.abs(x_flat) + 1.0)),
        ("sqrt(Abs(x))", np.sqrt(np.abs(x_flat))),
    ]
    if np.nanmax(np.abs(x_flat)) < 20.0:
        components.extend([
            ("exp(x)", np.exp(x_flat)),
            ("exp(-x)", np.exp(-x_flat)),
        ])
    if np.all(x_flat + 1.0 > 1e-12):
        components.append(("log(x+1)", np.log(x_flat + 1.0)))
    if np.all(x_flat > 1e-12):
        components.append(("log(x)", np.log(x_flat)))
        components.append(("sqrt(x)", np.sqrt(x_flat)))

    filtered: List[Tuple[str, np.ndarray]] = []
    seen: set = set()
    for name, values in components:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if name in seen or arr.size != x_flat.size or not np.all(np.isfinite(arr)):
            continue
        if float(np.std(arr)) < 1e-12:
            continue
        seen.add(name)
        filtered.append((name, arr))
    return filtered


def build_decomposition_probe_candidates(
    x: np.ndarray,
    y: np.ndarray,
    predictions: Optional[Dict[str, float]] = None,
    *,
    max_candidates: int = 8,
) -> List[Dict[str, Any]]:
    """Generate cheap additive/product/rational seed candidates.

    These probes are intentionally small. They are meant to seed later search
    and candidate selection, not replace the main sparse-regression path.
    """
    if x.ndim == 2:
        if x.shape[1] != 1:
            return []
        x_flat = x[:, 0]
        x_eval = x
    elif x.ndim == 1:
        x_flat = x.reshape(-1)
        x_eval = x.reshape(-1, 1)
    else:
        return []

    y_flat = np.asarray(y, dtype=np.float64).reshape(-1)
    x_flat = np.asarray(x_flat, dtype=np.float64).reshape(-1)
    if x_flat.size != y_flat.size or x_flat.size < 8:
        return []
    if not (np.all(np.isfinite(x_flat)) and np.all(np.isfinite(y_flat))):
        return []

    components = _univariate_component_library(x_flat)
    if not components:
        return []

    y_var = max(float(np.var(y_flat)), 1e-12)
    keep_mse = max(1e-8, 0.08 * y_var)
    candidates: List[Dict[str, Any]] = []

    def add_candidate(formula: str, pred: np.ndarray, probe_type: str, active_terms: List[str]) -> None:
        if not formula or not np.all(np.isfinite(pred)):
            return
        raw_mse = float(np.mean((y_flat - np.asarray(pred, dtype=np.float64).reshape(-1)) ** 2))
        if not np.isfinite(raw_mse):
            return
        if raw_mse > keep_mse and len(candidates) >= max_candidates:
            return
        governed = _display_candidate_score(
            formula,
            x_eval,
            y_flat,
            raw_mse=raw_mse,
            fit_mse=raw_mse,
            complexity=max(1, len(active_terms) + formula.count("*") + formula.count("/")),
            n_terms=max(1, len(active_terms)),
            postprocess=False,
        )
        display_mse = governed.get("display_mse")
        if display_mse is None or not np.isfinite(float(display_mse)):
            return
        if float(display_mse) > keep_mse and len(candidates) >= max_candidates:
            return
        candidates.append({
            "formula": str(governed.get("formula") or formula),
            "mse": float(display_mse),
            "score": float(governed.get("score", display_mse)),
            "n_nonzero": max(1, len(active_terms)),
            "active_terms": active_terms,
            "alpha": -2.0,
            "raw_mse": raw_mse,
            "display_mse": float(display_mse),
            "governor": governed,
            "source": "decomposition_probe",
            "decomposition_probe_type": probe_type,
        })

    def fit_design(names: List[str], columns: List[np.ndarray], probe_type: str) -> None:
        design = np.column_stack([np.ones_like(y_flat)] + columns)
        try:
            coeffs, _, _, _ = np.linalg.lstsq(design, y_flat, rcond=None)
        except (np.linalg.LinAlgError, ValueError):
            return
        pred = design @ coeffs
        formula = _format_linear_combo_formula(names, coeffs)
        add_candidate(formula, pred, probe_type, names)

    for name, values in components:
        fit_design([name], [values], "single_component")

    pair_limit = min(len(components), 12)
    for i in range(pair_limit):
        name_a, values_a = components[i]
        for j in range(i + 1, pair_limit):
            name_b, values_b = components[j]
            fit_design([name_a, name_b], [values_a, values_b], "additive_pair")

            product = values_a * values_b
            if np.all(np.isfinite(product)) and float(np.std(product)) > 1e-12:
                fit_design([f"({name_a})*({name_b})"], [product], "multiplicative_pair")

            denom = 1.0 + values_b ** 2
            ratio = values_a / denom
            if np.all(np.isfinite(ratio)) and float(np.std(ratio)) > 1e-12:
                fit_design([f"({name_a})/(1+({name_b})**2)"], [ratio], "rational_pair")

    if not candidates:
        return []

    deduped: Dict[str, Dict[str, Any]] = {}
    for cand in candidates:
        key = normalize_formula_ascii(str(cand.get("formula", ""))).replace(" ", "")
        current = deduped.get(key)
        if current is None or (
            float(cand.get("score", float("inf"))),
            float(cand.get("mse", float("inf"))),
        ) < (
            float(current.get("score", float("inf"))),
            float(current.get("mse", float("inf"))),
        ):
            deduped[key] = cand

    ranked = sorted(
        deduped.values(),
        key=lambda c: (
            float(c.get("score", float("inf"))),
            float(c.get("mse", float("inf"))),
            int(c.get("n_nonzero", 999)),
        ),
    )
    return ranked[:max(1, int(max_candidates))]


def lasso_coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.1,
    max_iter: int = 1000,
    tol: float = 1e-4,
    backend: str = "auto",
) -> np.ndarray:
    """
    LASSO regression using coordinate descent via C++ Eigen backend.
    """
    X_np = np.asarray(X, dtype=np.float64)
    y_np = np.asarray(y, dtype=np.float64).flatten()

    backend_norm = str(backend or "auto").lower()
    if backend_norm in {"numpy", "python", "fallback"}:
        return _lasso_coordinate_descent_python(X_np, y_np, alpha, max_iter, tol)

    _core, _ = _load_cpp_core()
    if _core is None or not hasattr(_core, "lasso_coordinate_descent"):
        return _lasso_coordinate_descent_python(X_np, y_np, alpha, max_iter, tol)
    
    # Run fast C++ coordinate descent
    w_list = _core.lasso_coordinate_descent(X_np, y_np, alpha, max_iter, tol)
    return np.array(w_list, dtype=np.float64)


def soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    """
    Vectorized soft thresholding operator for LASSO.
    
    Works with both scalars and arrays.
    """
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


def build_basis_from_predictions(
    x: np.ndarray,
    predictions: Dict[str, float],
    threshold: float = 0.5,
    max_power: int = 6,
    detected_omegas: Optional[List[float]] = None,
    universal_basis: bool = True,  # NEW: Always include common terms
    op_constraints: Optional[Dict[str, bool]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build regression basis matrix from classifier predictions.
    
    Args:
        x: Input values (N,)
        predictions: Dict mapping operator names to probabilities
        threshold: Minimum probability to include operator
        max_power: Maximum polynomial degree
        detected_omegas: FFT-detected frequencies for sin/cos
        universal_basis: If True, always include polynomial + periodic terms
        
    Returns:
        basis: (N, n_basis) matrix
        names: List of basis function names
    """
    predictions = _with_derived_predictions(predictions)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    elif x.ndim != 2:
        raise ValueError(f"Expected x to be 1D or 2D, got shape {x.shape}")

    n, n_vars = x.shape
    prediction_uncertainty = _prediction_uncertainty_metrics(predictions)
    low_trust_multivariate = (
        n_vars > 1
        and bool(prediction_uncertainty.get("prediction_uncertain", False))
        and float(prediction_uncertainty.get("prediction_entropy") or 0.0) >= 0.80
        and float(prediction_uncertainty.get("prediction_margin") or 1.0) <= 0.10
    )
    multivariate_blackbox = bool(n_vars > 1 and (universal_basis or low_trust_multivariate))
    compact_multivariate = bool(n_vars > 1 and low_trust_multivariate)
    if compact_multivariate:
        max_power = min(int(max_power), 3)
    
    basis_list = []
    names = []
    
    # Always include constant term
    basis_list.append(np.ones(n))
    names.append("1")
    
    # Operator constraints
    constraints = op_constraints or {}
    allow_power = constraints.get('power', True)
    allow_periodic = constraints.get('periodic', True)
    allow_exp = constraints.get('exp', True)
    allow_log = constraints.get('log', True)
    allow_arithmetic = constraints.get('arithmetic', True)

    def var_name(i: int) -> str:
        return "x" if n_vars == 1 else f"x{i}"

    # Polynomial terms (always include in universal mode)
    include_polynomial = allow_power and (
        universal_basis or
        predictions.get('power', 0) >= threshold or
        predictions.get('polynomial', 0) >= threshold
    )

    if include_polynomial:
        for i in range(n_vars):
            xi = x[:, i]
            basis_list.append(xi)
            names.append(var_name(i))

            # Integer powers (2, 3, 4)
            for p in range(2, max_power + 1):
                basis_list.append(xi ** p)
                names.append(f"{var_name(i)}^{p}")

            # Fractional powers are useful in 1D exact recovery, but in
            # multivariate blackbox mode they strongly encourage brittle fits.
            if not multivariate_blackbox:
                xi_safe = np.abs(xi) + 1e-10
                for p in [0.5, 1.5, 2.5, 0.33, 0.67, 1.33, 2.33]:
                    basis_list.append(np.sign(xi) * (xi_safe ** p))
                    names.append(f"{var_name(i)}^{p}")
    
    # Periodic operations - build comprehensive omega list
    # Always include common frequencies: 1.0, 2.0, 0.5
    omegas = [1.0, 2.0, 0.5]  # Standard frequencies
    
    # If the classifier strongly predicts pi, prioritize pi-based frequencies
    if predictions.get('const_pi', 0) >= threshold:
        omegas.extend([math.pi, 2 * math.pi])
        
    if detected_omegas:
        for o in detected_omegas[:3]:
            # Add if not too close to existing
            if all(abs(o - existing) > 0.1 for existing in omegas):
                omegas.append(o)
    
    # Periodic terms (always include in universal mode)
    include_periodic = allow_periodic and (
        (universal_basis and not multivariate_blackbox) or
        predictions.get('sin', 0) >= threshold or
        predictions.get('cos', 0) >= threshold or
        predictions.get('periodic', 0) >= threshold
    )
    if multivariate_blackbox and detected_omegas:
        include_periodic = include_periodic or predictions.get('periodic', 0) >= max(0.55, threshold)

    if include_periodic:
        for i in range(n_vars):
            xi = x[:, i]
            name = var_name(i)

            omega_limit = 2 if compact_multivariate else 6
            for omega in omegas[:omega_limit]:
                basis_list.append(np.sin(omega * xi))
                if omega == 1.0:
                    names.append(f"sin({name})")
                elif omega == 2.0:
                    names.append(f"sin(2*{name})")
                elif omega == 0.5:
                    names.append(f"sin({name}/2)")
                elif abs(omega - math.pi) < 1e-4:
                    names.append(f"sin(pi*{name})")
                elif abs(omega - 2*math.pi) < 1e-4:
                    names.append(f"sin(2*pi*{name})")
                else:
                    names.append(f"sin({omega:.2f}*{name})")

            for omega in omegas[:omega_limit]:
                basis_list.append(np.cos(omega * xi))
                if omega == 1.0:
                    names.append(f"cos({name})")
                elif omega == 2.0:
                    names.append(f"cos(2*{name})")
                elif omega == 0.5:
                    names.append(f"cos({name}/2)")
                elif abs(omega - math.pi) < 1e-4:
                    names.append(f"cos(pi*{name})")
                elif abs(omega - 2*math.pi) < 1e-4:
                    names.append(f"cos(2*pi*{name})")
                else:
                    names.append(f"cos({omega:.2f}*{name})")
    
    # Exponential operations (only if predicted OR universal)
    include_exp = allow_exp and (
        (universal_basis and not multivariate_blackbox) or
        predictions.get('exp', 0) >= threshold or 
        predictions.get('exponential', 0) >= threshold
    )
    if multivariate_blackbox:
        include_exp = include_exp and max(
            predictions.get('exp', 0),
            predictions.get('exponential', 0),
        ) >= max(0.85, threshold + 0.25)
    if include_exp:
        for i in range(n_vars):
            xi = x[:, i]
            name = var_name(i)
            x_clamp = np.clip(xi, -10, 10)
            exp_x = np.exp(x_clamp)
            basis_list.append(exp_x)
            names.append(f"exp({name})")
            basis_list.append(np.exp(-x_clamp))
            names.append(f"exp(-{name})")
            basis_list.append(np.exp(-xi**2))
            names.append(f"exp(-{name}^2)")

            if not multivariate_blackbox:
                denom = exp_x - 1.0
                denom = np.where(np.abs(denom) < 1e-6, np.sign(denom + 1e-12) * 1e-6, denom)
                basis_list.append(1.0 / denom)
                names.append(f"1/(exp({name})-1)")
                basis_list.append(xi / denom)
                names.append(f"{name}/(exp({name})-1)")
                basis_list.append((xi ** 2) / denom)
                names.append(f"{name}^2/(exp({name})-1)")
                basis_list.append((xi ** 3) / denom)
                names.append(f"{name}^3/(exp({name})-1)")
    
    # Logarithmic operations (always include in universal mode for Nguyen-7 etc.)
    include_log = allow_log and ((universal_basis and not multivariate_blackbox) or predictions.get('log', 0) >= threshold)
    if multivariate_blackbox:
        include_log = include_log and predictions.get('log', 0) >= max(0.85, threshold + 0.25)
    if include_log:
        for i in range(n_vars):
            xi = x[:, i]
            name = var_name(i)
            x_safe = np.maximum(np.abs(xi), 1e-10)
            basis_list.append(np.log(x_safe + 1))
            names.append(f"log({name}+1)")
            basis_list.append(np.log(x_safe**2 + 1))
            names.append(f"log({name}^2+1)")
    
    # Composition terms (for sin(x²), etc. - covers Nguyen-10)
    if universal_basis and allow_periodic and not multivariate_blackbox:
        for i in range(n_vars):
            xi = x[:, i]
            name = var_name(i)
            basis_list.append(np.sin(xi**2))
            names.append(f"sin({name}^2)")
            basis_list.append(np.cos(xi**2))
            names.append(f"cos({name}^2)")

            # Topologist's sine curve terms: sin(1/x)
            # Avoid division by zero
            x_safe_div = xi.copy()
            mask_zero = np.abs(x_safe_div) < 1e-3
            x_safe_div[mask_zero] = 1e-3 * np.sign(x_safe_div[mask_zero] + 1e-9) # Keep sign
            
            basis_list.append(np.sin(1.0 / x_safe_div))
            names.append(f"sin(1/{name})")
            basis_list.append(np.cos(1.0 / x_safe_div))
            names.append(f"cos(1/{name})")

    # Power/rational families should be available even if periodic is disabled
    if universal_basis and allow_power and not multivariate_blackbox:
        for i in range(n_vars):
            xi = x[:, i]
            name = var_name(i)
            x_safe = np.maximum(np.abs(xi), 1e-3)
            basis_list.append(np.sqrt(x_safe))
            names.append(f"sqrt(|{name}|)")
            basis_list.append(1.0 / x_safe)
            names.append(f"1/|{name}|")

            x2 = xi**2
            safe_denom = np.maximum(1 - x2, 1e-6)
            basis_list.append(1.0 / np.sqrt(safe_denom))
            names.append(f"1/sqrt(1-{name}^2)")
            basis_list.append(np.sqrt(safe_denom))
            names.append(f"sqrt(1-{name}^2)")
            basis_list.append(xi / np.sqrt(safe_denom))
            names.append(f"{name}/sqrt(1-{name}^2)")
            basis_list.append(1.0 / safe_denom)
            names.append(f"1/(1-{name}^2)")



    # Pairwise interaction terms for multi-input formulas
    if universal_basis and allow_arithmetic and n_vars > 1:
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                basis_list.append(x[:, i] * x[:, j])
                names.append(f"{var_name(i)}*{var_name(j)}")
    
    # Product-ratio terms for physics formulas (e.g., G*m1*m2/r²)
    if universal_basis and allow_arithmetic and n_vars >= 2 and not compact_multivariate:
        epsilon = 1e-8  # Prevent division by zero
        
        # Triple products: a*b*c
        if n_vars >= 3:
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    for k in range(j + 1, n_vars):
                        basis_list.append(x[:, i] * x[:, j] * x[:, k])
                        names.append(f"{var_name(i)}*{var_name(j)}*{var_name(k)}")
        
        # Product-ratio terms: a*b/c, a*b/c²
        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    continue
                xi, xj = x[:, i], x[:, j]
                
                # a*b/c for all pairs
                for k in range(n_vars):
                    if k == i or k == j:
                        continue
                    xk = x[:, k]
                    # a*b/c
                    denom = np.abs(xk) + epsilon
                    basis_list.append((xi * xj) / denom)
                    names.append(f"{var_name(i)}*{var_name(j)}/|{var_name(k)}|")
                    
                    # a*b/c² - critical for gravitational/inverse-square laws
                    denom_sq = xk**2 + epsilon
                    basis_list.append((xi * xj) / denom_sq)
                    names.append(f"{var_name(i)}*{var_name(j)}/{var_name(k)}²")
        
        # Ratio terms: a/b, a/b²
        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    continue
                xi, xj = x[:, i], x[:, j]
                
                # a/b
                denom = np.abs(xj) + epsilon
                basis_list.append(xi / denom)
                names.append(f"{var_name(i)}/|{var_name(j)}|")
                
                # a/b²
                denom_sq = xj**2 + epsilon
                basis_list.append(xi / denom_sq)
                names.append(f"{var_name(i)}/{var_name(j)}²")
                
                # a²/b
                basis_list.append((xi**2) / denom)
                names.append(f"{var_name(i)}²/|{var_name(j)}|")
        
        # Square root ratio terms: sqrt(a)/b, a/sqrt(b) - for relativistic mechanics
        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    continue
                xi, xj = x[:, i], x[:, j]
                
                sqrt_xi = np.sqrt(np.abs(xi) + epsilon)
                sqrt_xj = np.sqrt(np.abs(xj) + epsilon)
                
                basis_list.append(sqrt_xi / (np.abs(xj) + epsilon))
                names.append(f"√|{var_name(i)}|/|{var_name(j)}|")
                
                basis_list.append(xi / sqrt_xj)
                names.append(f"{var_name(i)}/√|{var_name(j)}|")

    # Rational and cross terms (per-variable)
    if universal_basis and allow_power and not compact_multivariate:
        for i in range(n_vars):
            xi = x[:, i]
            name = var_name(i)
            for c in [0.5, 1.0, 2.0]:
                denom_q = xi**2 + c
                basis_list.append(1.0 / denom_q)
                names.append(f"1/({name}^2+{c})")
                basis_list.append(xi / denom_q)
                names.append(f"{name}/({name}^2+{c})")
                basis_list.append(1.0 / (np.abs(xi) + c))
                names.append(f"1/(|{name}|+{c})")

            for c in [0.5, 1.0, 2.0]:
                denom_q4 = xi**4 + c
                basis_list.append(1.0 / denom_q4)
                names.append(f"1/({name}^4+{c})")
                basis_list.append(xi / denom_q4)
                names.append(f"{name}/({name}^4+{c})")
                basis_list.append((xi**2) / denom_q4)
                names.append(f"{name}^2/({name}^4+{c})")
                basis_list.append((xi**3) / denom_q4)
                names.append(f"{name}^3/({name}^4+{c})")

            if allow_periodic and (
                predictions.get('rational', 0) >= threshold or
                predictions.get('power', 0) >= threshold or
                predictions.get('periodic', 0) >= threshold
            ):
                for c in [0.5, 1.0, 2.0]:
                    denom_q = xi**2 + c
                    for omega in omegas[:4]:
                        basis_list.append(np.sin(omega * xi) / denom_q)
                        names.append(f"sin({omega:.2f}*{name})/({name}^2+{c})")
                        basis_list.append(np.cos(omega * xi) / denom_q)
                        names.append(f"cos({omega:.2f}*{name})/({name}^2+{c})")

            basis_list.append(xi * np.sin(xi))
            names.append(f"{name}·sin({name})")
            basis_list.append(xi * np.cos(xi))
            names.append(f"{name}·cos({name})")

            if include_exp and allow_periodic:
                decay_rates = [0.2, 0.5]
                for alpha in decay_rates:
                    decay = np.exp(-alpha * np.abs(xi))
                    for omega in omegas:
                        basis_list.append(decay * np.sin(omega * xi))
                        if abs(omega - 1.0) < 0.1:
                            names.append(f"e^(-{alpha}*{name})·sin({name})")
                        else:
                            names.append(f"e^(-{alpha}*{name})·sin({omega:.2f}*{name})")

                        basis_list.append(decay * np.cos(omega * xi))
                        if abs(omega - 1.0) < 0.1:
                            names.append(f"e^(-{alpha}*{name})·cos({name})")
                        else:
                            names.append(f"e^(-{alpha}*{name})·cos({omega:.2f}*{name})")
    
    basis = np.column_stack(basis_list)
    
    # CRITICAL: Clamp basis to prevent numerical explosion
    basis = np.clip(basis, -1e6, 1e6)
    basis = np.nan_to_num(basis, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return basis, names
        



def find_exact_symbolic_match(
    basis: np.ndarray,
    names: List[str],
    y: np.ndarray,
    max_terms: int = 3,
    tolerance: float = 1e-6,
    num_threads: int = 1,
    device: Optional[str] = None,
    exact_match_backend: str = "auto",
    exact_match_min_gpu_work: int = DEFAULT_EXACT_MATCH_MIN_GPU_WORK,
    exact_match_max_combos: int = DEFAULT_EXACT_MATCH_MAX_COMBOS,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> Optional[Tuple[str, float, np.ndarray]]:
    """
    Search for exact symbolic matches before falling back to LASSO.
    
    Tries single terms, pairs, and triples of basis functions to find
    exact symbolic solutions (MSE < tolerance).
    
    Args:
        basis: (N, n_basis) matrix
        names: List of basis function names
        y: Target values (N,)
        max_terms: Maximum number of terms to try in combination
        tolerance: MSE threshold for "exact" match
        device: Preferred torch device for accelerated batched search
        exact_match_backend: "auto", "cpu", "cuda"/"torch_cuda", "torch", or "numpy"
        exact_match_min_gpu_work: Minimum estimated work before auto uses CUDA
        exact_match_max_combos: Maximum pair/triple combinations to search exhaustively
        diagnostics: Optional dict populated with backend selection/fallback details
        
    Returns:
        (formula, mse, coefficients) if exact match found, else None
    """
    import math
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    n_basis = basis.shape[1]
    y = y.flatten()
    
    def build_formula(indices: List[int], coeffs: np.ndarray) -> Tuple[str, np.ndarray]:
        from glassbox.sr.operations.meta_ops import get_constant_symbol
        terms = []
        full_coeffs = np.zeros(n_basis)
        for idx, c in zip(indices, coeffs):
            if abs(c) < 1e-6:
                continue
            name = names[idx]
            if name == "1":
                terms.append(get_constant_symbol(c, 0.05))
            elif abs(c - 1.0) < 0.01:
                terms.append(name)
            elif abs(c + 1.0) < 0.01:
                terms.append(f"-{name}")
            elif abs(c - round(c)) < 0.01 and abs(c) < 100:
                terms.append(f"{int(round(c))}*{name}")
            else:
                coef_sym = get_constant_symbol(c, 0.05)
                terms.append(f"{coef_sym}*{name}")
            full_coeffs[idx] = c

        formula = _join_formula_terms(terms)
        return formula, full_coeffs

    # Try single basis functions with coefficient fitting
    for i in range(n_basis):
        if names[i] == "1":  # Skip constant-only
            continue
        
        # Try with and without constant
        for include_const in [False, True]:
            if include_const:
                X = np.column_stack([np.ones(len(y)), basis[:, i]])
                try:
                    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                    y_pred = X @ coeffs
                    mse = np.mean((y - y_pred) ** 2)
                    if mse < tolerance:
                        formula = _format_affine_formula(names[i], float(coeffs[1]), float(coeffs[0]))
                        
                        full_coeffs = np.zeros(n_basis)
                        const_idx = names.index("1") if "1" in names else 0
                        full_coeffs[const_idx] = coeffs[0] if include_const else 0
                        full_coeffs[i] = coeffs[1] if include_const else coeffs[0]
                        return formula, mse, full_coeffs
                except (np.linalg.LinAlgError, ValueError):
                    pass
            else:
                X = basis[:, i:i+1]
                try:
                    coeff, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                    y_pred = X @ coeff
                    mse = np.mean((y - y_pred) ** 2)
                    if mse < tolerance:
                        formula = _format_affine_formula(names[i], float(coeff[0]), 0.0)
                        
                        full_coeffs = np.zeros(n_basis)
                        full_coeffs[i] = coeff[0]
                        return formula, mse, full_coeffs
                except (np.linalg.LinAlgError, ValueError):
                    pass
    
    def update_diagnostics(values: Dict[str, Any]) -> None:
        if diagnostics is not None:
            diagnostics.update(values)

    def bounded_sparse_beam_search(
        max_support_size: int,
        *,
        beam_width: int = 32,
        candidate_limit: int = 64,
    ) -> Optional[Tuple[str, float, np.ndarray]]:
        y_centered = y - float(np.mean(y))
        y_norm = float(np.linalg.norm(y_centered))
        if y_norm <= 1e-15:
            return None

        col_scores = []
        for idx in range(n_basis):
            col = np.asarray(basis[:, idx], dtype=np.float64)
            col_centered = col - float(np.mean(col))
            denom = float(np.linalg.norm(col_centered) * y_norm)
            if denom <= 1e-15:
                score = 0.0
            else:
                score = abs(float(np.dot(col_centered, y_centered)) / denom)
            if np.isfinite(score):
                col_scores.append((score, idx))
        col_scores.sort(reverse=True)
        ranked = [idx for _, idx in col_scores[:max(1, min(candidate_limit, len(col_scores)))]]
        if not ranked:
            return None

        const_idx = names.index("1") if "1" in names else None
        beams: List[Tuple[float, Tuple[int, ...], np.ndarray]] = []
        best: Optional[Tuple[float, Tuple[int, ...], np.ndarray]] = None

        def fit_support(support: Tuple[int, ...]) -> Optional[Tuple[float, Tuple[int, ...], np.ndarray]]:
            if not support:
                return None
            X = basis[:, list(support)]
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            except (np.linalg.LinAlgError, ValueError):
                return None
            pred = X @ coeffs
            mse_val = float(np.mean((y - pred) ** 2))
            if not np.isfinite(mse_val):
                return None
            return mse_val, support, np.asarray(coeffs, dtype=np.float64)

        initial_supports = []
        for idx in ranked:
            if const_idx is not None and idx != const_idx:
                initial_supports.append(tuple(sorted((const_idx, idx))))
            initial_supports.append((idx,))

        seen = set()
        for support in initial_supports:
            if support in seen:
                continue
            seen.add(support)
            fit = fit_support(support)
            if fit is None:
                continue
            beams.append(fit)
            if best is None or fit[0] < best[0]:
                best = fit
        beams = sorted(beams, key=lambda item: (item[0], len(item[1])))[:beam_width]

        for size in range(2, max(2, int(max_support_size)) + 1):
            next_beams = []
            for beam_mse, support, beam_coeffs in beams:
                if len(support) >= size:
                    next_beams.append((beam_mse, support, beam_coeffs))
                    continue
                support_set = set(support)
                for idx in ranked:
                    if idx in support_set:
                        continue
                    new_support = tuple(sorted((*support, idx)))
                    if len(new_support) != len(support) + 1 or new_support in seen:
                        continue
                    seen.add(new_support)
                    fit = fit_support(new_support)
                    if fit is None:
                        continue
                    next_beams.append(fit)
                    if best is None or fit[0] < best[0]:
                        best = fit
            if not next_beams:
                break
            beams = sorted(next_beams, key=lambda item: (item[0], len(item[1])))[:beam_width]

        if best is not None and best[0] < tolerance:
            mse_val, support, coeffs = best
            formula, full_coeffs = build_formula(list(support), coeffs)
            update_diagnostics({
                "fallback_reason": "bounded_sparse_beam_match",
                "beam_width": int(beam_width),
                "candidate_limit": int(candidate_limit),
                "support_size": int(len(support)),
            })
            return formula, mse_val, full_coeffs

        if best is not None:
            update_diagnostics({
                "fallback_reason": "bounded_sparse_beam_no_exact_match",
                "beam_best_mse": float(best[0]),
                "beam_width": int(beam_width),
                "candidate_limit": int(candidate_limit),
            })
        return None

    combo_count = 0
    for r in range(2, min(int(max_terms), 3) + 1):
        if n_basis >= r:
            combo_count += math.comb(n_basis, r)
    if exact_match_max_combos is not None and combo_count > int(exact_match_max_combos):
        update_diagnostics({
            "backend_requested": exact_match_backend,
            "fallback_reason": "combo_cap_exceeded",
            "combo_count": int(combo_count),
            "max_combos": int(exact_match_max_combos),
            "torch_used": False,
            "gpu_used": False,
        })
        print(
            "  Skipping exhaustive exact-match search "
            f"(combos={combo_count} > cap={int(exact_match_max_combos)})"
        )
        beam_match = bounded_sparse_beam_search(max_terms)
        if beam_match is not None:
            print(f"  Bounded sparse exact-match search succeeded")
            return beam_match
        return None

    # Optional PyTorch acceleration for pairs and triples
    selected_device, torch_diagnostics = _select_exact_match_torch_device(
        exact_match_backend,
        device,
        _estimate_exact_match_work(len(y), n_basis, max_terms),
        exact_match_min_gpu_work,
    )
    update_diagnostics(torch_diagnostics)

    if selected_device is not None and max_terms >= 2:
        try:
            basis_t = torch.as_tensor(np.ascontiguousarray(basis), dtype=torch.float32, device=selected_device)
            y_t = torch.as_tensor(np.ascontiguousarray(y), dtype=torch.float32, device=selected_device).unsqueeze(1)
            N = basis_t.shape[0]

            def fast_torch_search(r: int):
                idx_cpu = torch.combinations(torch.arange(n_basis, device="cpu"), r=r)
                idx_len = int(idx_cpu.shape[0])
                n_combos = idx_len
                chunk_size = 50000

                for start in range(0, n_combos, chunk_size):
                    end = min(start + chunk_size, n_combos)
                    chunk_idx = idx_cpu[start:end].to(selected_device)
                    
                    X = basis_t[:, chunk_idx] # N x C x r
                    X = X.permute(1, 0, 2) # C x N x r
                    y_batch = y_t.expand(end - start, N, 1)
                    
                    sol = torch.linalg.lstsq(X, y_batch).solution # C x r x 1
                    pred = torch.bmm(X, sol) # C x N x 1
                    mse = torch.mean((y_batch - pred)**2, dim=1).squeeze(-1) # C
                    
                    best_mse, best_idx = torch.min(mse, dim=0)
                    
                    if best_mse < tolerance:
                        idx_in_chunk = best_idx.item()
                        real_idx = start + idx_in_chunk
                        return idx_cpu[real_idx].tolist(), sol[idx_in_chunk].flatten().detach().cpu().numpy(), best_mse.item()
                return None

            if max_terms >= 2:
                res = fast_torch_search(2)
                if res is not None:
                    indices, coeffs, mse = res
                    coeffs_arr = np.asarray(coeffs, dtype=np.float64)
                    y_pred = basis[:, indices] @ coeffs_arr
                    mse_cpu = float(np.mean((y - y_pred) ** 2))
                    if mse_cpu < tolerance:
                        update_diagnostics({"match_backend": str(selected_device), "validated_on_cpu": True})
                        formula, full_coeffs = build_formula(indices, coeffs_arr)
                        return formula, mse_cpu, full_coeffs

            if max_terms >= 3:
                res = fast_torch_search(3)
                if res is not None:
                    indices, coeffs, mse = res
                    coeffs_arr = np.asarray(coeffs, dtype=np.float64)
                    y_pred = basis[:, indices] @ coeffs_arr
                    mse_cpu = float(np.mean((y - y_pred) ** 2))
                    if mse_cpu < tolerance:
                        update_diagnostics({"match_backend": str(selected_device), "validated_on_cpu": True})
                        formula, full_coeffs = build_formula(indices, coeffs_arr)
                        return formula, mse_cpu, full_coeffs
            
            return None
        except Exception as e:
            if selected_device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            update_diagnostics({"fallback_reason": f"torch_search_failed:{type(e).__name__}"})
            print(f"  [PyTorch fast search failed: {e}]")
            pass # Fall back to numpy implementation on any PyTorch error
    
    def chunk_ranges(n: int, chunks: int) -> List[Tuple[int, int]]:
        if chunks <= 1 or n <= 1:
            return [(0, n)]
        size = max(1, math.ceil(n / chunks))
        return [(i, min(i + size, n)) for i in range(0, n, size)]

    def search_pairs_range(start_i: int, end_i: int, stop_event: threading.Event):
        for i in range(start_i, end_i):
            if stop_event.is_set():
                return None
            for j in range(i + 1, n_basis):
                X = basis[:, [i, j]]
                try:
                    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                    y_pred = X @ coeffs
                    mse = np.mean((y - y_pred) ** 2)
                    if mse < tolerance:
                        formula, full_coeffs = build_formula([i, j], coeffs)
                        stop_event.set()
                        return formula, mse, full_coeffs
                except (np.linalg.LinAlgError, ValueError):
                    pass
        return None

    def search_triples_range(start_i: int, end_i: int, stop_event: threading.Event):
        for i in range(start_i, end_i):
            if stop_event.is_set():
                return None
            for j in range(i + 1, n_basis):
                for k in range(j + 1, n_basis):
                    X = basis[:, [i, j, k]]
                    try:
                        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                        y_pred = X @ coeffs
                        mse = np.mean((y - y_pred) ** 2)
                        if mse < tolerance:
                            formula, full_coeffs = build_formula([i, j, k], coeffs)
                            stop_event.set()
                            return formula, mse, full_coeffs
                    except (np.linalg.LinAlgError, ValueError):
                        pass
        return None

    # Try pairs of basis functions (including constant)
    if max_terms >= 2:
        if num_threads and num_threads > 1:
            stop_event = threading.Event()
            ranges = chunk_ranges(n_basis, num_threads)
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(search_pairs_range, start, end, stop_event) for start, end in ranges]
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        return result
        else:
            result = search_pairs_range(0, n_basis, threading.Event())
            if result is not None:
                return result
    
    # Try triples of basis functions (including constant)
    if max_terms >= 3:
        if num_threads and num_threads > 1:
            stop_event = threading.Event()
            ranges = chunk_ranges(n_basis, num_threads)
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(search_triples_range, start, end, stop_event) for start, end in ranges]
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        return result
        else:
            result = search_triples_range(0, n_basis, threading.Event())
            if result is not None:
                return result
    
    return None


def fast_path_regression(
    x: np.ndarray,
    y: np.ndarray,
    predictions: Dict[str, float],
    detected_omegas: Optional[List[float]] = None,
    sparsity_threshold: float = 0.01,
    op_constraints: Optional[Dict[str, bool]] = None,
    universal_basis: bool = True,
    exact_match_threads: int = 1,
    exact_match_enabled: bool = True,
    exact_match_max_basis: int = 150,
    max_power: int = 6,
    holdout_fraction: float = 0.10,
    device: Optional[str] = None,
    exact_match_backend: str = "auto",
    exact_match_min_gpu_work: int = DEFAULT_EXACT_MATCH_MIN_GPU_WORK,
    exact_match_max_combos: int = DEFAULT_EXACT_MATCH_MAX_COMBOS,
) -> Tuple[str, float, Dict]:
    """
    Directly solve for coefficients using least squares regression.
    
    IMPROVED: First searches for exact symbolic matches before LASSO.
    Supports out-of-domain holdout scoring for generalization assessment.
    
    Args:
        x: Input values (N,)
        y: Target values (N,)
        predictions: Classifier predictions
        detected_omegas: FFT-detected frequencies
        sparsity_threshold: Coefficients below this are zeroed
        holdout_fraction: Fraction of domain-edge points held out
            for out-of-domain scoring (default 0.10, i.e. 5% each
            from the lowest and highest x-values). Set to 0 to disable.
        
    Returns:
        formula: String representation
        mse: Mean squared error
        details: Dict with coefficients, basis names, and holdout_mse
    """
    if x.ndim > 2:
        raise ValueError(f"Expected x to be 1D or 2D, got shape {x.shape}")
    y = y.flatten()
    y_variance = float(np.var(y)) if y.size > 0 else 0.0
    exact_match_diagnostics: Dict[str, Any] = {}

    # ── Out-of-domain holdout: hold back domain-edge points ──
    holdout_mask = None
    x_fit, y_fit = x, y  # default: fit on everything
    x_holdout, y_holdout = None, None

    if holdout_fraction > 0 and x.ndim <= 2:
        n = len(y)
        n_edge = max(1, int(n * holdout_fraction / 2))
        if x.ndim == 2 and x.shape[1] > 1:
            # Multi-input: hold out based on L2 norm from center
            x_center = x.mean(axis=0)
            dists = np.linalg.norm(x - x_center, axis=1)
            order = np.argsort(dists)
            holdout_indices = np.concatenate([order[-n_edge:]])
        else:
            x_flat = x.ravel() if x.ndim == 1 else x[:, 0]
            order = np.argsort(x_flat)
            holdout_indices = np.concatenate([order[:n_edge], order[-n_edge:]])

        holdout_mask = np.zeros(n, dtype=bool)
        holdout_mask[holdout_indices] = True
        fit_mask = ~holdout_mask

        if fit_mask.sum() >= 10:  # need enough points to fit
            x_fit = x[fit_mask] if x.ndim == 1 else x[fit_mask]
            y_fit = y[fit_mask]
            x_holdout = x[holdout_indices] if x.ndim == 1 else x[holdout_indices]
            y_holdout = y[holdout_indices]
        else:
            holdout_mask = None  # not enough data, skip

    if x.ndim == 2 and x.shape[1] > 1:
        easy_match = _maybe_match_easy_multivariate_formula(x, y)
        if easy_match is not None:
            formula, mse, details = easy_match
            print(f"  Direct template match: {details['template_match']}")
            return formula, mse, details

    transform_match = _maybe_match_univariate_transform_template(
        x,
        y,
        predictions,
        tolerance=max(1e-10, y_variance * 1e-12),
    )
    if transform_match is not None:
        formula, mse, details = transform_match
        print(f"  Direct transform template match: {details['template_match']}")
        details.setdefault("y_variance", y_variance)
        details.setdefault("holdout_mse", 0.0 if holdout_mask is not None else None)
        return formula, mse, details
    
    # Build basis from predictions (fit subset for holdout; full data for exact-match/evaluation)
    basis_full, names = build_basis_from_predictions(
        x, predictions, 
        threshold=0.3,  # Lower threshold to include more options
        max_power=max_power,
        detected_omegas=detected_omegas,
        op_constraints=op_constraints,
        universal_basis=universal_basis,
    )

    # If holdout is active, build a separate fit-only basis
    if holdout_mask is not None:
        basis = basis_full[~holdout_mask]
        basis_holdout = basis_full[holdout_mask]
    else:
        basis = basis_full
        basis_holdout = None
    
    print(f"  Fast-path basis: {len(names)} terms")
    print(f"  Terms: {names[:10]}{'...' if len(names) > 10 else ''}")
    
    # STEP 1: Try to find exact symbolic matches FIRST (before LASSO).
    # This prevents mixed bases from winning with approximate exp/rational/fractional
    # terms when the target is an exact polynomial.
    if op_constraints:
        allow_periodic = op_constraints.get('periodic', True)
        allow_exp = op_constraints.get('exp', True)
        allow_log = op_constraints.get('log', True)
        allow_power = op_constraints.get('power', True)
    else:
        allow_periodic = allow_exp = allow_log = allow_power = True

    if exact_match_enabled and allow_power:
        exact_poly_match = _find_exact_polynomial_match(
            x,
            y,
            names,
            max_degree=max_power,
            tolerance=max(1e-10, y_variance * 1e-12),
        )
        if exact_poly_match:
            formula, mse, coeffs = exact_poly_match
            print(f"  Found EXACT polynomial match: {formula} (MSE={mse:.2e})")
            active_idx = np.flatnonzero(np.abs(coeffs) >= sparsity_threshold)
            return formula, mse, {
                'coefficients': coeffs,
                'basis_names': names,
                'n_nonzero': sum(1 for c in coeffs if abs(c) >= sparsity_threshold),
                'exact_match': True,
                'exact_match_diagnostics': {
                    'backend_requested': exact_match_backend,
                    'fallback_reason': 'polynomial_shortcut',
                    'torch_used': False,
                    'gpu_used': False,
                },
                'candidate_formulas': [{
                    'formula': formula,
                    'mse': float(mse),
                    'score': float(mse),
                    'n_nonzero': int(np.sum(np.abs(coeffs) >= sparsity_threshold)),
                    'active_terms': [names[i] for i in active_idx],
                    'alpha': 0.0,
                }],
                'y_variance': y_variance,
                'holdout_mse': 0.0 if holdout_mask is not None else None,
            }

    # If only power is allowed, enable 10-term exact match for polynomials
    exact_max_terms = 10 if (allow_power and not allow_periodic and not allow_exp and not allow_log) else 4

    if exact_match_enabled and (exact_match_max_basis is None or basis.shape[1] <= exact_match_max_basis):
        exact_match = find_exact_symbolic_match(
            basis,
            names,
            y_fit,
            max_terms=exact_max_terms,
            tolerance=1e-5,
            num_threads=exact_match_threads,
            device=device,
            exact_match_backend=exact_match_backend,
            exact_match_min_gpu_work=exact_match_min_gpu_work,
            exact_match_max_combos=exact_match_max_combos,
            diagnostics=exact_match_diagnostics,
        )
        if exact_match:
            formula, mse, coeffs = exact_match
            full_pred = basis_full @ coeffs
            full_mse = float(np.mean((y - full_pred) ** 2))
            holdout_mse = None
            if basis_holdout is not None and y_holdout is not None:
                holdout_pred = basis_holdout @ coeffs
                holdout_mse = float(np.mean((y_holdout - holdout_pred) ** 2))
            if not np.isfinite(full_mse) or full_mse >= 1e-6:
                exact_match_diagnostics.update({
                    'rejected_reason': 'full_domain_mse_not_exact',
                    'fit_mse': float(mse),
                    'full_mse': full_mse,
                    'holdout_mse': holdout_mse,
                })
                print(
                    "  Rejected symbolic shortcut: "
                    f"fit MSE={mse:.2e}, full MSE={full_mse:.2e}"
                )
            else:
                n_terms_exact = int(np.sum(np.abs(coeffs) >= sparsity_threshold))
                governed_exact = _display_candidate_score(
                    formula,
                    x,
                    y,
                    raw_mse=full_mse,
                    fit_mse=float(mse),
                    holdout_mse=holdout_mse,
                    complexity=n_terms_exact,
                    n_terms=n_terms_exact,
                    postprocess=True,
                )
                display_mse = governed_exact.get('display_mse')
                if display_mse is None or not np.isfinite(float(display_mse)) or float(display_mse) >= 1e-6:
                    exact_match_diagnostics.update({
                        'rejected_reason': 'display_mse_not_exact',
                        'fit_mse': float(mse),
                        'full_mse': full_mse,
                        'display_mse': display_mse,
                        'holdout_mse': holdout_mse,
                    })
                    print(
                        "  Rejected symbolic shortcut: "
                        f"raw MSE={full_mse:.2e}, display MSE={display_mse}"
                    )
                else:
                    formula = str(governed_exact.get('formula') or formula)
                    print(f"  Found EXACT symbolic match: {formula} (MSE={float(display_mse):.2e})")
                    active_idx = np.flatnonzero(np.abs(coeffs) >= sparsity_threshold)
                    return formula, float(display_mse), {
                        'coefficients': coeffs,
                        'basis_names': names,
                        'n_nonzero': n_terms_exact,
                        'exact_match': True,
                        'exact_match_diagnostics': exact_match_diagnostics,
                        'candidate_formulas': [{
                            'formula': formula,
                            'mse': float(display_mse),
                            'score': float(governed_exact.get('score', display_mse)),
                            'n_nonzero': n_terms_exact,
                            'active_terms': [names[i] for i in active_idx],
                            'alpha': 0.0,
                            'raw_mse': float(full_mse),
                            'display_mse': float(display_mse),
                            'governor': governed_exact,
                        }],
                        'holdout_mse': holdout_mse,
                        'candidate_governor': governed_exact,
                    }
    elif exact_match_enabled:
        print(f"  Skipping exact-match search (basis={basis.shape[1]} > {exact_match_max_basis})")
        exact_match_diagnostics = {
            'backend_requested': exact_match_backend,
            'fallback_reason': 'basis_exceeds_exact_match_max',
            'basis_terms': int(basis.shape[1]),
            'exact_match_max_basis': int(exact_match_max_basis),
        }
    
    # Normalize basis for numerical stability
    basis_std = np.std(basis, axis=0, keepdims=True)
    basis_std[basis_std < 1e-10] = 1.0
    basis_norm = basis / basis_std

    def _coeffs_to_formula(coeffs_arr: np.ndarray) -> str:
        terms_local = []
        for name, coef in zip(names, coeffs_arr):
            if abs(coef) < sparsity_threshold:
                continue
            term = _format_regression_term(name, float(coef))
            if term:
                terms_local.append(term)
        return _join_formula_terms(terms_local)

    def _candidate_signature(coeffs_arr: np.ndarray) -> Tuple[int, ...]:
        return tuple(np.flatnonzero(np.abs(coeffs_arr) >= sparsity_threshold).tolist())

    def _semantic_prediction_signature(coeffs_arr: np.ndarray) -> Optional[Tuple[int, ...]]:
        """Hash a candidate by its sampled prediction curve, not its basis support."""
        try:
            pred = np.asarray(basis_full @ coeffs_arr, dtype=np.float64).reshape(-1)
        except Exception:
            return None
        if pred.size == 0 or not np.all(np.isfinite(pred)):
            return None

        sample_n = min(64, pred.size)
        if sample_n < pred.size:
            sample_idx = np.linspace(0, pred.size - 1, sample_n, dtype=int)
            pred_sample = pred[sample_idx]
        else:
            pred_sample = pred

        scale = max(float(np.std(y)), float(np.std(pred_sample)), 1.0)
        centered = (pred_sample - float(np.mean(pred_sample))) / scale
        return tuple(np.round(centered * 1e7).astype(np.int64).tolist())

    def _update_candidate_pool(
        pool: Dict[Tuple[int, ...], Dict[str, Any]],
        coeffs_arr: np.ndarray,
        mse_val: float,
        alpha_val: float,
        solver_backend: str = "auto",
    ) -> None:
        if not np.isfinite(mse_val):
            return
        n_terms_local = int(np.sum(np.abs(coeffs_arr) >= sparsity_threshold))
        score_local = float(mse_val + COMPLEXITY_PENALTY * n_terms_local)

        # Out-of-domain holdout penalty: penalize solutions that overfit
        if basis_holdout is not None and y_holdout is not None:
            try:
                y_pred_ho = basis_holdout @ coeffs_arr
                ho_mse = float(np.mean((y_holdout - y_pred_ho) ** 2))
                if np.isfinite(ho_mse):
                    ood_ratio = ho_mse / max(mse_val, 1e-12)
                    # Penalize if holdout MSE is much worse than in-sample
                    if ood_ratio > 5.0:
                        score_local += 0.01 * ho_mse
            except Exception:
                pass

        signature = _candidate_signature(coeffs_arr)
        current = pool.get(signature)
        if current is None or score_local < current['score']:
            pool[signature] = {
                'coeffs': coeffs_arr.copy(),
                'mse': float(mse_val),
                'n_terms': n_terms_local,
                'score': score_local,
                'alpha': alpha_val,
                'solver_backend': solver_backend,
            }

    def _holdout_mse_for_best(coeffs_arr: np.ndarray) -> Optional[float]:
        """Compute holdout MSE for the best candidate's coefficients."""
        if basis_holdout is None or y_holdout is None:
            return None
        try:
            y_pred_ho = basis_holdout @ coeffs_arr
            ho_mse = float(np.mean((y_holdout - y_pred_ho) ** 2))
            return ho_mse if np.isfinite(ho_mse) else None
        except Exception:
            return None
    
    # Try LASSO with adaptive alpha (coordinate descent)
    best_coeffs = None
    best_mse = float('inf')
    best_score = float('inf')  # Complexity-penalized score
    candidate_pool: Dict[Tuple[int, ...], Dict[str, Any]] = {}
    
    # Complexity penalty: prefer simpler solutions
    COMPLEXITY_PENALTY = 0.001  # λ in: score = MSE + λ * n_terms
    
    cpp_core, _cpp_reason = _load_cpp_core()
    solver_backends = ["numpy"]
    if cpp_core is not None and hasattr(cpp_core, "lasso_coordinate_descent"):
        solver_backends.insert(0, "cpp")

    # Try multiple alpha values to find best sparsity-accuracy tradeoff
    for alpha in [0.0, 0.001, 0.01, 0.05, 0.1, 0.2]:
        for solver_backend in solver_backends:
            try:
                coeffs = lasso_coordinate_descent(
                    basis_norm,
                    y_fit,
                    alpha=alpha,
                    max_iter=1000,
                    backend=solver_backend,
                )

                # Check for NaN/Inf in coeffs
                if not np.all(np.isfinite(coeffs)):
                    print(f"  Warning: {solver_backend} alpha={alpha} produced non-finite coefficients")
                    continue

                # Unnormalize coefficients
                coeffs = coeffs / basis_std.flatten()

                # Compute MSE on fit data
                y_pred = basis @ coeffs
                mse = np.mean((y_fit - y_pred) ** 2)
                n_terms = np.sum(np.abs(coeffs) > sparsity_threshold)

                # Check if MSE is valid
                if not np.isfinite(mse):
                    print(
                        f"  Warning: {solver_backend} alpha={alpha} produced non-finite MSE "
                        f"(max pred: {np.max(np.abs(y_pred))})"
                    )
                    continue

                # Complexity-penalized score
                score = mse + COMPLEXITY_PENALTY * n_terms

                # Select best based on penalized score (prefers simpler solutions)
                if score < best_score:
                    best_coeffs = coeffs
                    best_mse = mse
                    best_score = score

                _update_candidate_pool(
                    candidate_pool,
                    coeffs,
                    mse,
                    alpha,
                    solver_backend=solver_backend,
                )
            except Exception as e:
                print(f"  Error with {solver_backend} alpha={alpha}: {e}")
                continue
    
    if best_coeffs is None:
        # Fallback to plain least squares
        try:
            best_coeffs, _, _, _ = np.linalg.lstsq(basis, y_fit, rcond=None)
            y_pred = basis @ best_coeffs
            best_mse = np.mean((y_fit - y_pred) ** 2)
            best_score = best_mse + COMPLEXITY_PENALTY * np.sum(np.abs(best_coeffs) >= sparsity_threshold)
            _update_candidate_pool(
                candidate_pool,
                best_coeffs,
                best_mse,
                alpha_val=-1.0,
                solver_backend="lstsq_fallback",
            )
        except np.linalg.LinAlgError:
            return None, float('inf'), {}

    # IMPORTANT: Refit each candidate with OLS on selected terms only.
    # This recovers exact coefficients while preserving sparse structure.
    for signature, candidate in list(candidate_pool.items()):
        coeffs_local = candidate['coeffs']
        selected_mask = np.abs(coeffs_local) >= sparsity_threshold
        if selected_mask.sum() == 0 or selected_mask.sum() == len(coeffs_local):
            continue
        basis_selected = basis[:, selected_mask]
        try:
            refit_coeffs, _, _, _ = np.linalg.lstsq(basis_selected, y_fit, rcond=None)
            y_pred = basis_selected @ refit_coeffs
            refit_mse = float(np.mean((y_fit - y_pred) ** 2))

            if refit_mse <= candidate['mse'] + 0.001:
                updated = np.zeros_like(coeffs_local)
                updated[selected_mask] = refit_coeffs
                n_terms_local = int(np.sum(np.abs(updated) >= sparsity_threshold))
                candidate_pool[signature] = {
                    'coeffs': updated,
                    'mse': refit_mse,
                    'n_terms': n_terms_local,
                    'score': refit_mse + COMPLEXITY_PENALTY * n_terms_local,
                    'alpha': candidate.get('alpha', -1.0),
                    'solver_backend': candidate.get('solver_backend', 'auto'),
                }
        except (np.linalg.LinAlgError, ValueError):
            pass

    if not candidate_pool:
        _update_candidate_pool(
            candidate_pool,
            best_coeffs,
            best_mse,
            alpha_val=-1.0,
            solver_backend="best_fallback",
        )

    governed_candidates: List[Dict[str, Any]] = []
    for candidate in candidate_pool.values():
        cand_coeffs = candidate['coeffs']
        cand_formula = _coeffs_to_formula(cand_coeffs)
        cand_pred_full = basis_full @ cand_coeffs
        cand_full_mse = float(np.mean((y - cand_pred_full) ** 2))
        if not np.isfinite(cand_full_mse):
            cand_full_mse = float(candidate['mse'])
        cand_holdout = _holdout_mse_for_best(cand_coeffs) if holdout_mask is not None else None
        n_terms_local = int(np.sum(np.abs(cand_coeffs) >= sparsity_threshold))
        governed = _display_candidate_score(
            cand_formula,
            x,
            y,
            raw_mse=cand_full_mse,
            fit_mse=float(candidate['mse']),
            holdout_mse=cand_holdout,
            complexity=n_terms_local,
            n_terms=n_terms_local,
        )
        governed_candidates.append({
            **candidate,
            'formula': cand_formula,
            'full_mse': cand_full_mse,
            'holdout_mse': cand_holdout,
            'governor': governed,
            'governor_score': float(governed.get('score', float('inf'))),
            'display_mse': governed.get('display_mse'),
            'display_formula': governed.get('formula') or cand_formula,
        })

    semantic_dedup = {
        'before': len(governed_candidates),
        'after': len(governed_candidates),
        'removed': 0,
        'enabled': True,
    }
    if governed_candidates:
        semantic_pool: Dict[Tuple[int, ...], Dict[str, Any]] = {}
        unique_fallback: List[Dict[str, Any]] = []

        def _candidate_rank_key(c: Dict[str, Any]) -> Tuple[float, float, float, float, int]:
            return (
                float(c.get('governor_score', float('inf'))),
                float(c['display_mse']) if c.get('display_mse') is not None else float('inf'),
                float(c.get('score', float('inf'))),
                float(c.get('mse', float('inf'))),
                int(c.get('n_terms', 9999)),
            )

        for cand in governed_candidates:
            semantic_signature = _semantic_prediction_signature(cand['coeffs'])
            if semantic_signature is None:
                unique_fallback.append(cand)
                continue
            current = semantic_pool.get(semantic_signature)
            if current is None or _candidate_rank_key(cand) < _candidate_rank_key(current):
                semantic_pool[semantic_signature] = cand

        governed_candidates = list(semantic_pool.values()) + unique_fallback
        semantic_dedup['after'] = len(governed_candidates)
        semantic_dedup['removed'] = max(0, semantic_dedup['before'] - semantic_dedup['after'])

    sorted_candidates = sorted(
        governed_candidates,
        key=lambda c: (
            c['governor_score'],
            float(c['display_mse']) if c.get('display_mse') is not None else float('inf'),
            c['score'],
            c['mse'],
        ),
    )
    top_candidates = sorted_candidates[:5]
    best_candidate = top_candidates[0]

    coeffs = best_candidate['coeffs']
    mse = float(best_candidate.get('display_mse') if best_candidate.get('display_mse') is not None else best_candidate.get('full_mse', best_candidate['mse']))
    if not np.isfinite(mse):
        mse = float(best_candidate.get('full_mse', best_candidate['mse']))
    formula = str(best_candidate.get('display_formula') or best_candidate.get('formula') or _coeffs_to_formula(coeffs))

    n_nonzero = int(np.sum(np.abs(coeffs) >= sparsity_threshold))
    exact_match_flag = mse < 1e-6 and n_nonzero <= 10

    candidate_formulas = []
    for cand in top_candidates:
        cand_coeffs = cand['coeffs']
        active_idx = np.flatnonzero(np.abs(cand_coeffs) >= sparsity_threshold)
        candidate_formulas.append({
            'formula': str(cand.get('display_formula') or cand.get('formula') or _coeffs_to_formula(cand_coeffs)),
            'mse': float(cand.get('display_mse') if cand.get('display_mse') is not None else cand.get('full_mse', cand['mse'])),
            'score': float(cand.get('governor_score', cand['score'])),
            'n_nonzero': int(cand['n_terms']),
            'active_terms': [names[i] for i in active_idx],
            'alpha': float(cand.get('alpha', -1.0)),
            'raw_mse': float(cand.get('full_mse', cand['mse'])),
            'display_mse': cand.get('display_mse'),
            'governor': cand.get('governor'),
            'solver_backend': cand.get('solver_backend'),
        })

    decomposition_candidates = build_decomposition_probe_candidates(
        x,
        y,
        predictions,
        max_candidates=8,
    )
    if decomposition_candidates:
        seen_formula_keys = {
            normalize_formula_ascii(str(c.get('formula', ''))).replace(" ", "")
            for c in candidate_formulas
        }
        for cand in decomposition_candidates:
            key = normalize_formula_ascii(str(cand.get('formula', ''))).replace(" ", "")
            if key and key not in seen_formula_keys:
                candidate_formulas.append(cand)
                seen_formula_keys.add(key)

        best_decomp = decomposition_candidates[0]
        best_decomp_mse = float(best_decomp.get('mse', float('inf')))
        if (
            np.isfinite(best_decomp_mse)
            and (
                best_decomp_mse < max(1e-10, mse * 0.90)
                or (best_decomp_mse < 1e-8 and not exact_match_flag)
            )
        ):
            formula = str(best_decomp.get('formula') or formula)
            mse = best_decomp_mse
            exact_match_flag = mse < 1e-6 and int(best_decomp.get('n_nonzero', 99)) <= 10
            n_nonzero = int(best_decomp.get('n_nonzero', n_nonzero))
            best_candidate['governor'] = best_decomp.get('governor')

    return formula, mse, {
        'coefficients': coeffs,
        'basis_names': names,
        'n_nonzero': n_nonzero,
        'exact_match': exact_match_flag,
        'compact_multivariate_basis': bool(x.ndim == 2 and x.shape[1] > 1 and len(names) <= 120),
        'y_variance': y_variance,
        'exact_match_diagnostics': exact_match_diagnostics,
        'candidate_formulas': candidate_formulas,
        'holdout_mse': best_candidate.get('holdout_mse') if holdout_mask is not None else None,
        'candidate_governor': best_candidate.get('governor'),
        'candidate_semantic_dedup': semantic_dedup,
        'decomposition_probe_candidates': decomposition_candidates,
        'solver_backends': solver_backends,
        'winning_solver_backend': best_candidate.get('solver_backend'),
    }


def refine_frequencies(
    x: np.ndarray,
    y: np.ndarray,
    initial_omegas: List[float],
    n_steps: int = 100,
    lr: float = 0.1,
    device: Optional[str] = None,
) -> Tuple[List[float], float]:
    """
    Refine frequency parameters using gradient descent.
    
    This handles cases like ω=3.2 where FFT might detect 3.13.
    """
    _core, reason = _load_cpp_core()
    if _core is None or not hasattr(_core, "refine_frequencies"):
        raise ImportError(reason or "C++ _core extension does not provide refine_frequencies")
    
    x_np = np.asarray(x, dtype=np.float64).flatten()
    y_np = np.asarray(y, dtype=np.float64).flatten()
    return _core.refine_frequencies(x_np, y_np, initial_omegas, n_steps, lr)


def refine_periodic_rational(
    x: np.ndarray,
    y: np.ndarray,
    omega_inits: List[float],
    c_inits: List[float],
    steps: int = 200,
    lr: float = 0.05,
    device: Optional[str] = None,
) -> Optional[Dict]:
    """
    Continuous refinement for terms like sin(ωx)/(x^2+c).
    Fits omega and c (positive) with linear coefficients.
    """
    import torch
    import math
    resolved_device = _resolve_device(device)
    x_t = torch.tensor(x, dtype=torch.float64, device=resolved_device)
    y_t = torch.tensor(y, dtype=torch.float64, device=resolved_device)

    best = None
    best_mse = float('inf')

    for omega0 in omega_inits:
        for c0 in c_inits:
            # Parameters
            omega = torch.nn.Parameter(torch.tensor(float(omega0), dtype=torch.float64, device=resolved_device))
            c_unconstrained = torch.nn.Parameter(torch.tensor(math.log(math.exp(c0) - 1 + 1e-6), dtype=torch.float64, device=resolved_device))
            a = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64, device=resolved_device))
            b = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64, device=resolved_device))
            d = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64, device=resolved_device))
            e = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64, device=resolved_device))

            params = [omega, c_unconstrained, a, b, d, e]
            opt = torch.optim.Adam(params, lr=lr)

            for _ in range(steps):
                opt.zero_grad()
                c = torch.nn.functional.softplus(c_unconstrained) + 1e-6
                denom = x_t**2 + c
                pred = (
                    a * torch.sin(omega * x_t) / denom +
                    b * torch.cos(omega * x_t) / denom +
                    d * x_t + e
                )
                loss = torch.mean((pred - y_t) ** 2)
                loss.backward()
                opt.step()

            # Recompute MSE with final parameters
            pred = (
                a * torch.sin(omega * x_t) / (x_t**2 + torch.nn.functional.softplus(c_unconstrained) + 1e-6) +
                b * torch.cos(omega * x_t) / (x_t**2 + torch.nn.functional.softplus(c_unconstrained) + 1e-6) +
                d * x_t + e
            )
            mse = float(torch.mean((pred - y_t) ** 2).item())
            if mse < best_mse:
                best_mse = mse
                c_val = float(torch.nn.functional.softplus(c_unconstrained).item())
                omega_val = float(omega.item())
                coeffs = (float(a.item()), float(b.item()), float(d.item()), float(e.item()))

                # Build formula
                terms = []
                a_c, b_c, d_c, e_c = coeffs
                denom_str = f"(x^2+{c_val:.4g})"
                if abs(a_c) > 1e-8:
                    terms.append(f"{get_constant_symbol(a_c, 0.05)}*sin({omega_val:.3g}*x)/{denom_str}")
                if abs(b_c) > 1e-8:
                    terms.append(f"{get_constant_symbol(b_c, 0.05)}*cos({omega_val:.3g}*x)/{denom_str}")
                if abs(d_c) > 1e-8:
                    terms.append(f"{get_constant_symbol(d_c, 0.05)}*x")
                if abs(e_c) > 1e-8:
                    terms.append(f"{get_constant_symbol(e_c, 0.05)}")

                formula = _join_formula_terms(terms)

                best = {
                    'formula': formula,
                    'mse': mse,
                    'details': {
                        'coefficients': np.array([a_c, b_c, d_c, e_c]),
                        'basis_names': [
                            f"sin({omega_val:.3g}*x)/(x^2+{c_val:.4g})",
                            f"cos({omega_val:.3g}*x)/(x^2+{c_val:.4g})",
                            "x",
                            "1",
                        ],
                        'n_nonzero': sum(1 for v in [a_c, b_c, d_c, e_c] if abs(v) > 1e-8),
                        'exact_match': mse < 1e-6,
                    }
                }

    return best


def refine_powers(
    x: np.ndarray,
    y: np.ndarray,
    initial_powers: Optional[List[float]] = None,
    detected_omegas: Optional[List[float]] = None,
    n_steps: int = 200,
    lr: float = 0.05,
    device: Optional[str] = None,
) -> Tuple[Optional[Dict], float]:
    """
    Refine power exponent parameters using Eigen VarPro.

    Handles non-integer powers like x^2.3, x^0.7 where the basis
    only has integer powers (x^2, x^3). Can also include periodic terms.

    Model: y ≈ Σ aᵢ·sign(x)·|x|^pᵢ + Σ (bⱼ·sin(ωⱼx) + dⱼ·cos(ωⱼx)) + c₀ + c₁·x

    Args:
        x: Input values (N,)
        y: Target values (N,)
        initial_powers: Starting power guesses (default: [0.5, 1.5, 2.5, 3.5])
        detected_omegas: Optional list of frequencies to include
        n_steps: Gradient descent steps
        lr: Learning rate

    Returns:
        (result_dict, mse) where result_dict has 'formula', 'powers', 'coefficients'
    """
    _core, reason = _load_cpp_core()
    if _core is None or not hasattr(_core, "refine_powers"):
        raise ImportError(reason or "C++ _core extension does not provide refine_powers")

    # Ensure 1D inputs
    if x.ndim > 1:
        x = x.flatten()
    if y.ndim > 1:
        y = y.flatten()

    if initial_powers is None:
        initial_powers = [0.5, 1.5, 2.5, 3.5]

    # Filter out x <= 0 for safe power operations
    valid_mask = np.abs(x) > 1e-8
    if valid_mask.sum() < 10:
        return None, float('inf')
    x_valid = np.asarray(x[valid_mask], dtype=np.float64)
    y_valid = np.asarray(y[valid_mask], dtype=np.float64)

    best_result = None
    best_mse = float('inf')

    # Stage 1: Fit powers (and initial omegas if provided)
    stage1_models = []
    
    def _train_power_model(powers_subset):
        res, mse = _core.refine_powers(x_valid, y_valid, powers_subset, detected_omegas or [], n_steps, lr)
        return mse, res, powers_subset

    import concurrent.futures
    import multiprocessing
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, multiprocessing.cpu_count())) as executor:
        futures = []
        for n_powers in range(1, min(4, len(initial_powers) + 1)):
            for start_idx in range(max(1, len(initial_powers) - n_powers + 1)):
                powers_subset = initial_powers[start_idx:start_idx + n_powers]
                futures.append(executor.submit(_train_power_model, powers_subset))
        for f in concurrent.futures.as_completed(futures):
            stage1_models.append(f.result())
    
    # Sort by MSE and pick best Stage 1 model
    stage1_models.sort(key=lambda x: x[0])
    best_stage1_mse, best_stage1_res, best_powers = stage1_models[0]
    
    final_res = best_stage1_res
    final_mse = best_stage1_mse
    current_omegas = detected_omegas or []
    
    from glassbox.evolution import detect_dominant_frequency
    
    # Check residuals for hidden frequencies
    # We must construct pred_stage1 to get residuals
    # model: c0 + c1*x + sum p_i + sum w_i
    pred_stage1 = np.full_like(y_valid, final_res["constant"])
    pred_stage1 += final_res["linear"] * x_valid
    for i, p in enumerate(final_res["powers"]):
        abs_x = np.abs(x_valid) + 1e-10
        sign_x = np.sign(x_valid)
        is_even = 0.5 * (1.0 + np.cos(p * np.pi))
        abs_pow = np.power(abs_x, p)
        term = (1.0 - is_even) * (sign_x * abs_pow) + is_even * abs_pow
        pred_stage1 += final_res["coeffs"][i] * term
        
    for i, w in enumerate(current_omegas):
        if 2*i+1 < len(final_res["periodic_coeffs"]):
            pred_stage1 += final_res["periodic_coeffs"][2*i] * np.sin(w * x_valid)
            pred_stage1 += final_res["periodic_coeffs"][2*i+1] * np.cos(w * x_valid)
            
    residuals = y_valid - pred_stage1
    res_omegas = detect_dominant_frequency(x_valid, residuals, n_frequencies=2)
    
    new_omegas = []
    for o in res_omegas:
        if o < 0.1:
            continue
        if any(abs(o - eo) < 0.2 for eo in current_omegas):
            continue
        new_omegas.append(o)
    
    if new_omegas:
        combined_omegas = current_omegas + new_omegas
        res2, mse2 = _core.refine_powers(x_valid, y_valid, best_powers, combined_omegas, n_steps, lr)
        if mse2 < final_mse:
            final_res = res2
            final_mse = mse2
            current_omegas = combined_omegas

    # Final extraction from winning model
    best_mse = final_mse
    refined_powers = final_res["powers"]
    refined_coeffs = final_res["coeffs"]
    const_val = final_res["constant"]
    linear_val = final_res["linear"]
    per_coeffs = final_res["periodic_coeffs"]

    # Build formula
    terms = []
    # Power terms
    for p, c in zip(refined_powers, refined_coeffs):
        if abs(c) < 1e-3:
            continue
        
        p_snapped = _snap_power(p)
        
        if p_snapped == '1':
            c += linear_val
            linear_val = 0.0
            
        coef_str = get_constant_symbol(c, 0.05)
        if p_snapped == '1':
            terms.append(f"{coef_str}*x")
        else:
            terms.append(f"{coef_str}*x^{p_snapped}")
    
    # Periodic terms
    if per_coeffs and current_omegas:
        for i, omega in enumerate(current_omegas):
            if 2*i+1 < len(per_coeffs):
                sin_c = per_coeffs[2*i]
                cos_c = per_coeffs[2*i+1]
                
                if abs(sin_c) > 1e-3:
                    sym = get_constant_symbol(sin_c, 0.05)
                    terms.append(f"{sym}*sin({omega:.3g}*x)")
                if abs(cos_c) > 1e-3:
                    sym = get_constant_symbol(cos_c, 0.05)
                    terms.append(f"{sym}*cos({omega:.3g}*x)")
    
    if abs(linear_val) > 1e-3:
        terms.append(f"{get_constant_symbol(linear_val, 0.05)}*x")
    if abs(const_val) > 1e-3:
        terms.append(get_constant_symbol(const_val, 0.05))

    formula = _join_formula_terms(terms)

    best_result = {
        'formula': formula,
        'powers': refined_powers,
        'coefficients': refined_coeffs,
        'mse': best_mse,
    }

    return best_result, best_mse


def _snap_power(p: float, tol: float = 0.08) -> str:
    """Snap a power exponent to a clean value if close."""
    # Integer check
    if abs(p - round(p)) < tol:
        return str(int(round(p)))
    # Common fractions
    fractions = {0.5: '0.5', 1.5: '1.5', 2.5: '2.5', 3.5: '3.5',
                 1/3: '1/3', 2/3: '2/3', 4/3: '4/3', 5/3: '5/3',
                 0.25: '0.25', 0.75: '0.75', 1.25: '1.25', 1.75: '1.75'}
    for frac_val, frac_str in fractions.items():
        if abs(p - frac_val) < tol:
            return frac_str
    return f"{p:.3g}"


def refine_constants(
    x: np.ndarray,
    y: np.ndarray,
    detected_omegas: Optional[List[float]] = None,
    predictions: Optional[Dict[str, float]] = None,
    device: Optional[str] = None,
) -> Dict:
    """
    Unified constant refinement: runs both ω and p gradient refinement.

    Returns the best result from either frequency or power refinement.
    """
    results = {}

    # 1. Frequency refinement (if periodic detected)
    has_periodic = False
    if predictions:
        for key in ['sin', 'cos', 'periodic']:
            if predictions.get(key, 0) > 0.3:
                has_periodic = True
                break

    if has_periodic and detected_omegas:
        try:
            refined_omegas, freq_mse = refine_frequencies(
                x, y, detected_omegas, n_steps=150, device=device
            )
            results['frequency'] = {
                'omegas': refined_omegas,
                'mse': freq_mse,
            }
        except ImportError:
            pass

    # 2. Power refinement (if power predicted)
    has_power = False
    if predictions:
        if predictions.get('power', 0) > 0.3 or predictions.get('polynomial', 0) > 0.3:
            has_power = True

    if has_power:
        try:
            power_result, power_mse = refine_powers(
                x, y, detected_omegas=detected_omegas, device=device
            )
            if power_result is not None:
                results['power'] = power_result
        except ImportError:
            pass

    return results


def fast_path_with_refinement(
    x: np.ndarray,
    y: np.ndarray,
    predictions: Dict[str, float],
    detected_omegas: Optional[List[float]] = None,
    refine_steps: int = 100,
    op_constraints: Optional[Dict[str, bool]] = None,
    auto_expand: bool = True,
    device: Optional[str] = None,
    exact_match_threads: int = 1,
    exact_match_enabled: bool = True,
    exact_match_max_basis: int = 150,
    max_power: int = 6,
    exact_match_backend: str = "auto",
    exact_match_min_gpu_work: int = DEFAULT_EXACT_MATCH_MIN_GPU_WORK,
    exact_match_max_combos: int = DEFAULT_EXACT_MATCH_MAX_COMBOS,
) -> Tuple[str, float, Dict]:
    """
    Fast-path with optional frequency refinement.
    
    1. Run initial fast-path regression
    2. If MSE is moderate (0.01-0.5), try frequency refinement
    3. Re-run regression with refined frequencies
    """
    # Stage 1: Minimal basis (predicted ops only)
    # Stage 2: Optional universal basis expansion when auto_expand is enabled.
    stage_order = [False, True] if auto_expand else [False]
    best_formula = None
    best_mse = float('inf')
    best_details: Dict = {}
    best_universal = True
    complexity_lambda = 1e-4

    def should_accept_candidate(new_mse: float, new_details: Dict) -> bool:
        """Guardrail: avoid accepting large MSE regressions for small complexity gains."""

        new_terms = new_details.get('n_nonzero', 0)
        old_terms = best_details.get('n_nonzero', 0)

        # If current fit is already very good, do not accept major relative regressions
        # unless we cross into a materially simpler symbolic regime.
        if best_mse <= 1e-5 and new_mse > best_mse * 5.0:
            if not (new_mse < 1e-6 and new_terms <= 5 and old_terms > 5):
                return False

        return candidate_score(new_mse, new_details) < candidate_score(best_mse, best_details)

    def candidate_score(mse_val: float, details_val: Dict) -> float:
        n_val = details_val.get('n_nonzero', 0)
        return mse_val + complexity_lambda * max(0, n_val - 4)

    for use_universal in stage_order:
        formula, mse, details = fast_path_regression(
            x, y, predictions, detected_omegas,
            op_constraints=op_constraints,
            universal_basis=use_universal,
            exact_match_threads=exact_match_threads,
            exact_match_enabled=exact_match_enabled,
            exact_match_max_basis=exact_match_max_basis,
            max_power=max_power,
            device=device,
            exact_match_backend=exact_match_backend,
            exact_match_min_gpu_work=exact_match_min_gpu_work,
            exact_match_max_combos=exact_match_max_combos,
        )
        if candidate_score(mse, details) < candidate_score(best_mse, best_details):
            best_mse = mse
            best_formula = formula
            best_details = details
            best_universal = use_universal

        # If exact match or good enough AND simple, return immediately
        n_current = details.get('n_nonzero', 0)
        is_exact = details.get('exact_match', False)
        if (is_exact or mse < 1e-6) and n_current <= 4:
            return formula, mse, details

    # If good enough and exact AND simple, return best
    n_best = best_details.get('n_nonzero', 0)
    if best_mse < 0.01 and best_details.get('exact_match', False) and best_formula is not None and n_best <= 4:
        return best_formula, best_mse, best_details
    
    # If moderate MSE and periodic terms were used, try refinement
    has_periodic_signal = (
        predictions.get('periodic', 0.0) >= 0.5 or
        predictions.get('sin', 0.0) >= 0.5 or
        predictions.get('cos', 0.0) >= 0.5
    )
    is_univariate_input = x.ndim == 1 or (x.ndim == 2 and x.shape[1] == 1)
    should_try_freq = (
        is_univariate_input
        and bool(detected_omegas)
        and has_periodic_signal
        and (1e-4 <= best_mse <= 0.2)
    )

    if should_try_freq:
        print(f"  Attempting frequency refinement (initial MSE={best_mse:.4f})...")
        try:
            refined_omegas, refined_mse = refine_frequencies(
                x, y, detected_omegas, n_steps=refine_steps, device=device
            )
        except Exception as err:
            print(f"  [Frequency refinement skipped: {err}]")
            refined_omegas, refined_mse = [], float('inf')

        if refined_omegas:
            print(f"  Refined frequencies: {[f'{o:.3f}' for o in refined_omegas]}")
            print(f"  Refinement MSE: {refined_mse:.6f}")

            # Re-run regression with refined omegas
            if refined_mse < best_mse:
                formula2, mse2, details2 = fast_path_regression(
                    x, y, predictions, detected_omegas=refined_omegas,
                    op_constraints=op_constraints,
                    universal_basis=best_universal,
                    exact_match_threads=exact_match_threads,
                    exact_match_enabled=exact_match_enabled,
                    exact_match_max_basis=exact_match_max_basis,
                    max_power=max_power,
                    device=device,
                    exact_match_backend=exact_match_backend,
                    exact_match_min_gpu_work=exact_match_min_gpu_work,
                    exact_match_max_combos=exact_match_max_combos,
                )
                if should_accept_candidate(mse2, details2):
                    return formula2, mse2, details2
    
    # Additional continuous refinement for periodic×rational terms
    if op_constraints:
        allow_periodic = op_constraints.get('periodic', True)
        allow_power = op_constraints.get('power', True)
    else:
        allow_periodic = allow_power = True

    if (x.ndim == 1 or (x.ndim == 2 and x.shape[1] == 1)) and allow_periodic and allow_power and best_mse > 5e-3:
        # Only trigger refinement for genuinely poor fits (MSE > 5e-3).
        # Previously 1e-4, which caused ~12s of wasted refinement on moderate fits.
        omega_pool = (detected_omegas or []) + [1.0, 2.0, 3.0]
        # Deduplicate and keep reasonable range
        omega_inits = []
        for o in omega_pool:
            if 0.1 < o < 20 and all(abs(o - e) > 0.05 for e in omega_inits):
                omega_inits.append(o)

        refined = refine_periodic_rational(
            x, y,
            omega_inits=omega_inits,
            c_inits=[0.5, 1.0],
            steps=100,
            lr=0.1,
            device=device,
        )
        if refined:
            print(f"  Periodic×Rational refinement MSE: {refined['mse']:.6f}")
            refined_details = refined.get('details', {})
            if should_accept_candidate(refined['mse'], refined_details):
                return refined['formula'], refined['mse'], refined_details

    # Complexity check
    n_terms = best_details.get('n_nonzero', 0)
    # Power exponent refinement for non-integer powers
    # Trigger if MSE is moderate OR if formula is complex (likely overfitted)
    has_power_signal = (
        predictions.get('power', 0.0) >= 0.4 or
        predictions.get('polynomial', 0.0) >= 0.4
    )
    should_try_power = (x.ndim == 1 or (x.ndim == 2 and x.shape[1] == 1)) and allow_power and has_power_signal and (1e-3 <= best_mse <= 0.2 or n_terms > 8)

    if should_try_power:
        print(f"  Attempting power refinement (MSE={best_mse:.4f}, Terms={n_terms})...")
        try:
            power_result, power_mse = refine_powers(
                x, y, detected_omegas=detected_omegas, device=device
            )
        except Exception as err:
            print(f"  [Power refinement skipped: {err}]")
            power_result, power_mse = None, float('inf')
        
        if power_result is not None:
            # Acceptance logic:
            # 1. Much better MSE (classic case)
            # 2. Good MSE (< 1e-3) and Much Simpler (prevent overfitting)
            
            is_better_mse = power_mse < best_mse - 1e-4
            is_good_and_simpler = (power_mse < 1e-3) and (len(power_result['powers']) + 2 < n_terms)
            
            if is_better_mse or is_good_and_simpler:
                print(f"  Power refinement SUCCESS (MSE: {power_mse:.6f}, Terms: {len(power_result['powers'])})")
                print(f"  Power formula: {power_result['formula']}")
                power_details = {
                    'coefficients': np.array(power_result['coefficients']),
                    'basis_names': [f"x^{_snap_power(p)}" for p in power_result['powers']],
                    'n_nonzero': sum(1 for c in power_result['coefficients'] if abs(c) > 1e-8),
                    'exact_match': power_mse < 1e-6,
                }
                if should_accept_candidate(power_mse, power_details):
                    return power_result['formula'], power_mse, power_details

    return best_formula, best_mse, best_details


def should_use_fast_path(
    predictions: Dict[str, float],
    confidence_threshold: float = 0.6,  # Lowered from 0.8 to allow more fast-path usage
    min_operators: int = 1,
    universal_basis: bool = True,
) -> bool:
    """
    Decide whether to use fast path based on classifier confidence.
    
    Fast path is used when:
    1. At least one operator predicted with high confidence
    2. The predicted operators are well-covered by our basis
    """
    # In universal-basis mode the classifier is a guide, not an applicability
    # gate. Low confidence should not make simple valid formulas like -x^2
    # skip fast-path entirely; the regression stage can still test the common
    # polynomial/trig/exp/rational families and reject them by fit quality.
    if universal_basis:
        return True

    # Get high-confidence predictions
    high_conf = [name for name, prob in predictions.items() if prob >= confidence_threshold]
    
    # Check if we have enough operators predicted
    if len(high_conf) < min_operators:
        return False
    
    # Check for unsupported operations that would need evolution
    # (Currently none, but could add later)
    
    # Fast path supported
    return True


def run_fast_path(
    x: torch.Tensor,
    y: torch.Tensor,
    classifier_path: str = DEFAULT_CURVE_CLASSIFIER_PATH,
    detected_omegas: Optional[List[float]] = None,
    op_constraints: Optional[Dict[str, bool]] = None,
    auto_expand: bool = True,
    device: Optional[str] = None,
    exact_match_threads: int = 1,
    exact_match_enabled: bool = True,
    exact_match_max_basis: int = 150,
    simplify_formula_output: bool = True,
    simplification_int_tol: float = 1e-5,
    simplification_zero_tol: float = 1e-8,
    simplification_log: bool = True,
    max_power: int = 6,
    exact_match_backend: str = "auto",
    exact_match_min_gpu_work: int = DEFAULT_EXACT_MATCH_MIN_GPU_WORK,
    exact_match_max_combos: int = DEFAULT_EXACT_MATCH_MAX_COMBOS,
) -> Optional[Dict]:
    """
    Run the complete fast-path pipeline.
    
    Returns:
        Dict with formula, mse, and timing if successful
        None if fast path not applicable
    """
    import time
    try:
        from glassbox.curve_classifier.curve_classifier_integration import predict_operators
    except ImportError:
        try:
            from curve_classifier_integration import predict_operators
        except ImportError:
            try:
                from scripts.curve_classifier_integration import predict_operators
            except ImportError:
                # Last resort: try to find it in the new package relative to root
                try:
                    import glassbox.curve_classifier.curve_classifier_integration as cci
                    predict_operators = cci.predict_operators
                except ImportError:
                    print("Warning: predict_operators not found. Fast path will be limited.")
                    def predict_operators(*args, **kwargs): return {}
    
    start_time = time.time()
    
    # Convert to numpy
    if hasattr(x, 'cpu'):
        x_np = x.cpu().numpy()
    else:
        x_np = x
    if hasattr(y, 'cpu'):
        y_np = y.cpu().numpy().flatten()
    else:
        y_np = y.flatten()
    
    # Get classifier predictions
    print("\n" + "="*60)
    print("FAST PATH: Classifier-Guided Regression")
    print("="*60)
    
    # Early return for constant signals (e.g. y=5, sin²+cos²=1)
    y_std = np.std(y_np)
    if y_std < 1e-10:
        elapsed = time.time() - start_time
        const_val = float(np.mean(y_np))
        # Format nicely: use integer if close to one
        if abs(const_val - round(const_val)) < 1e-6:
            formula = str(int(round(const_val)))
        else:
            formula = f"{const_val:.6g}"
        print(f"  Constant signal detected: y ≈ {const_val}")
        print(f"  Formula: {formula}")
        print("  MSE: 0.000000")
        print("="*60)
        y_pred = np.full_like(y_np, const_val, dtype=np.float64)
        residual_diagnostics = _residual_diagnostics(y_np, y_pred, x_np)
        result = {
            'formula': formula,
            'mse': 0.0,
            'time': elapsed,
            'details': {'n_nonzero': 1, 'exact_match': True,
                        'basis_names': ['1'], 'coefficients': np.array([const_val]),
                        'candidate_formulas': [{
                            'formula': formula,
                            'mse': 0.0,
                            'score': 0.0,
                            'n_nonzero': 1,
                            'active_terms': ['1'],
                            'alpha': 0.0,
                        }]},
            'predictions': {'identity': 1.0},
            'uncertainty': _prediction_uncertainty_metrics({'identity': 1.0}),
            'residual_diagnostics': residual_diagnostics,
            'candidate_formulas': [{
                'formula': formula,
                'mse': 0.0,
                'score': 0.0,
                'n_nonzero': 1,
                'active_terms': ['1'],
                'alpha': 0.0,
            }],
            'operator_hints': {'operators': set(), 'frequencies': [],
                               'powers': [], 'has_rational': False,
                               'has_exp_decay': False, 'active_terms': ['1']},
        }
        fpip_v2 = build_fpip_v2_from_fast_path(
            formula=result['formula'],
            mse=result['mse'],
            candidate_formulas=result.get('candidate_formulas', []),
            predictions=result.get('predictions', {}),
            uncertainty=result.get('uncertainty', {}),
            residual_diagnostics=result.get('residual_diagnostics', {}),
            operator_hints=result.get('operator_hints', {}),
        )
        fpip_ok, fpip_errors = validate_fpip_v2_payload(fpip_v2)
        result['fpip_v2'] = fpip_v2
        result['fpip_v2_valid'] = fpip_ok
        if not fpip_ok:
            result['fpip_v2_errors'] = fpip_errors
        return result
    
    predictions = predict_operators(
        x_np,
        y_np,
        classifier_path,
        threshold=0.3,
        device=device,
    )

    if not predictions:
        # Do not make classifier failure an applicability failure. The fast path
        # uses a universal basis by default, so conservative polynomial priors
        # are enough to attempt regression and let MSE decide.
        predictions = {'identity': 1.0, 'power': 1.0, 'polynomial': 1.0}

    uncertainty_metrics = _prediction_uncertainty_metrics(predictions or {})
    
    print(f"  Predictions: {[(k, f'{v:.2f}') for k, v in sorted(predictions.items(), key=lambda x: -x[1])]}")
    print(
        "  Uncertainty: "
        f"entropy={uncertainty_metrics['prediction_entropy']}, "
        f"margin={uncertainty_metrics['prediction_margin']}"
    )

    n_vars = int(x_np.shape[1]) if getattr(x_np, "ndim", 1) == 2 else 1
    low_trust_multivariate = (
        n_vars > 1
        and bool(uncertainty_metrics.get("prediction_uncertain", False))
        and float(uncertainty_metrics.get("prediction_entropy") or 0.0) >= 0.80
        and float(uncertainty_metrics.get("prediction_margin") or 1.0) <= 0.10
    )
    if low_trust_multivariate and auto_expand:
        auto_expand = False
        exact_match_max_basis = min(int(exact_match_max_basis or 150), 120)
        max_power = min(int(max_power), 3)
        print("  Low-trust multivariate classifier signal; using compact fast-path basis")
    
    # Check if fast path is applicable (lowered threshold to 0.6 for broader coverage)
    if not should_use_fast_path(predictions, confidence_threshold=0.6, universal_basis=auto_expand):
        print("  Classifier confidence too low - falling back to evolution")
        return None
    
    # Run fast path regression with optional frequency refinement
    formula, mse, details = fast_path_with_refinement(
        x_np, y_np, predictions, 
        detected_omegas=detected_omegas,
        refine_steps=150,  # More steps for better refinement
        op_constraints=op_constraints,
        auto_expand=auto_expand,
        device=device,
        exact_match_threads=exact_match_threads,
        exact_match_enabled=exact_match_enabled,
        exact_match_max_basis=exact_match_max_basis,
        max_power=max_power,
        exact_match_backend=exact_match_backend,
        exact_match_min_gpu_work=exact_match_min_gpu_work,
        exact_match_max_combos=exact_match_max_combos,
    )

    raw_formula = formula
    simplification_info = {
        'applied': False,
        'snapped_formula': None,
        'error': None,
    }

    if simplify_formula_output and formula:
        if simplification_log:
            print("  [Post] Running simplify_formula pipeline...")
        try:
            try:
                from simplify_formula import simplify_onn_formula, SnapConfig, snap_formula_floats
            except ImportError:
                from scripts.simplify_formula import simplify_onn_formula, SnapConfig, snap_formula_floats

            formula_len = len(formula)
            term_estimate = max(1, len([t for t in re.split(r'\s*[+-]\s*', formula) if t.strip()]))
            too_complex_for_symbolic = formula_len > 500 or term_estimate > 24

            if too_complex_for_symbolic or not simplification_info.get('exact_match', False):
                snapped_formula = snap_formula_floats(
                    formula,
                    SnapConfig(int_tol=simplification_int_tol, zero_tol=simplification_zero_tol),
                )
                simplified_formula = snapped_formula
                if simplification_log:
                    print(f"  [Post] Formula simplified using fast snap-only mode (no symbolic algebra) to bypass SymPy bottleneck.")
            else:
                snapped_formula, simplified_expr = simplify_onn_formula(
                    formula,
                    int_tol=simplification_int_tol,
                    zero_tol=simplification_zero_tol,
                    use_nsimplify=False,  # disabled nsimplify for speed
                )
                simplified_formula = str(simplified_expr)

            if simplification_log:
                print(f"  [Post] Snapped: {snapped_formula}")
                if too_complex_for_symbolic:
                    print(f"  [Post] Simplified (snap-only): {simplified_formula}")
                else:
                    print(f"  [Post] Simplified: {simplified_formula}")

            formula = simplified_formula
            simplification_info = {
                'applied': True,
                'snapped_formula': snapped_formula,
                'error': None,
            }
        except Exception as simpl_err:
            simplification_info = {
                'applied': False,
                'snapped_formula': None,
                'error': str(simpl_err),
            }
            if simplification_log:
                err_text = str(simpl_err).encode("ascii", errors="backslashreplace").decode("ascii")
                if len(err_text) > 300:
                    err_text = err_text[:297] + "..."
                print(f"  [Post] Simplification skipped (error): {err_text}")
    
    elapsed = time.time() - start_time

    # Keep a post-processed term estimate while preserving structural sparsity count.
    final_term_count = max(1, len([t for t in re.split(r'\s*[+-]\s*', formula) if t.strip()])) if formula else 0
    details['n_nonzero_simplified'] = final_term_count
    y_pred = _evaluate_formula_values(formula, x_np)
    residual_diagnostics = _residual_diagnostics(y_np, y_pred, x_np) if y_pred is not None else _empty_residual_diagnostics()
    
    print(f"\n  Formula: {formula}")
    print(f"  MSE: {mse:.6f}")
    holdout_mse = details.get('holdout_mse')
    if holdout_mse is not None:
        print(f"  Holdout MSE (domain edges): {holdout_mse:.6f}")
    print(f"  Non-zero terms: {details.get('n_nonzero', 0)}")
    if residual_diagnostics['residual_suspicious']:
        print(
            "  Residual diagnostics: suspicious structure "
            f"(fft_peak_ratio={residual_diagnostics['residual_spectral_peak_ratio']}, "
            f"holdout_ratio={residual_diagnostics['residual_holdout_ratio']})"
        )
    print(f"  Time: {elapsed:.2f}s")
    print("="*60)
    
    # Extract operator hints from the formula for guided evolution
    operator_hints = extract_operator_hints(formula, details.get('basis_names', []),
                                            details.get('coefficients', []),
                                            predictions=predictions)    
    result = {
        'formula': formula,
        'formula_raw': raw_formula,
        'mse': mse,
        'time': elapsed,
        'details': details,
        'predictions': predictions,
        'uncertainty': uncertainty_metrics,
        'residual_diagnostics': residual_diagnostics,
        'candidate_formulas': details.get('candidate_formulas', []),
        'simplification': simplification_info,
        'operator_hints': operator_hints,
    }

    fpip_v2 = build_fpip_v2_from_fast_path(
        formula=result['formula'],
        mse=result['mse'],
        candidate_formulas=result.get('candidate_formulas', []),
        predictions=result.get('predictions', {}),
        uncertainty=result.get('uncertainty', {}),
        residual_diagnostics=result.get('residual_diagnostics', {}),
        operator_hints=result.get('operator_hints', {}),
    )
    fpip_ok, fpip_errors = validate_fpip_v2_payload(fpip_v2)
    result['fpip_v2'] = fpip_v2
    result['fpip_v2_valid'] = fpip_ok
    if not fpip_ok:
        result['fpip_v2_errors'] = fpip_errors

    return result


def extract_operator_hints(
    formula: str,
    basis_names: List[str],
    coefficients: np.ndarray,
    threshold: float = 0.01,
    predictions: Dict[str, float] = None,
) -> Dict[str, Any]:
    """
    Extract operator hints from a fast-path formula for guided evolution.
    
    Analyzes the formula to determine which operators/structures are needed,
    which can be used to initialize a constrained ONN.
    
    Args:
        formula: The discovered formula string
        basis_names: List of basis function names
        coefficients: Array of coefficients (same length as basis_names)
        threshold: Minimum coefficient magnitude to consider active
        
    Returns:
        Dict with operator hints:
        - 'operators': Set of operator types used (sin, cos, exp, log, power, rational)
        - 'frequencies': List of detected frequencies for periodic terms
        - 'powers': List of detected power values
        - 'has_rational': Whether rational terms are present
        - 'active_terms': List of active basis term names
    """
    hints = {
        'operators': set(),
        'frequencies': [],
        'powers': [],
        'has_rational': False,
        'has_exp_decay': False,
        'active_terms': [],
    }
    
    if coefficients is None or len(coefficients) == 0:
        return hints
    
    # Find active terms
    for name, coef in zip(basis_names, coefficients):
        if abs(coef) < threshold:
            continue
        
        hints['active_terms'].append(name)
        
        # Detect operator types from basis names
        name_lower = name.lower()
        
        # Periodic operators
        if 'sin(' in name_lower:
            hints['operators'].add('sin')
            hints['operators'].add('periodic')
            # Extract frequency if present (using precompiled pattern)
            freq_match = _FREQ_SIN_PATTERN.search(name_lower)
            if freq_match:
                hints['frequencies'].append(float(freq_match.group(1)))
            elif 'sin(x)' in name_lower or 'sin(x/' in name_lower:
                hints['frequencies'].append(1.0)
                
        if 'cos(' in name_lower:
            hints['operators'].add('cos')
            hints['operators'].add('periodic')
            freq_match = _FREQ_COS_PATTERN.search(name_lower)
            if freq_match:
                hints['frequencies'].append(float(freq_match.group(1)))
        
        # Power operators
        if 'x^' in name_lower:
            hints['operators'].add('power')
            power_match = _POWER_PATTERN.search(name_lower)
            if power_match:
                hints['powers'].append(float(power_match.group(1)))
        
        # Exponential
        if 'exp(' in name_lower or 'e^(' in name_lower or 'e^-' in name_lower:
            hints['operators'].add('exp')
            if '-' in name_lower:  # e^(-ax)
                hints['has_exp_decay'] = True
        
        # Logarithm
        if 'log(' in name_lower:
            hints['operators'].add('log')
        
        # Rational terms
        if '/(x' in name_lower or '/(' in name_lower:
            hints['has_rational'] = True
            hints['operators'].add('rational')
            hints['operators'].add('power')  # Rational uses power (inv)
        
        # Square root
        if 'sqrt' in name_lower:
            hints['operators'].add('sqrt')
            hints['operators'].add('power')
            hints['powers'].append(0.5)
    
    # Inject raw classifier predictions to ensure evolution priors aren't blinded by poor basis regression
    if predictions:
        if predictions.get('sin', 0) > 0.3 or predictions.get('cos', 0) > 0.3 or predictions.get('periodic', 0) > 0.3:
            hints['operators'].add('periodic')
        if predictions.get('exp', 0) > 0.3 or predictions.get('exponential', 0) > 0.3:
            hints['operators'].add('exp')
        if predictions.get('log', 0) > 0.3:
            hints['operators'].add('log')
        if predictions.get('rational', 0) > 0.3:
            hints['operators'].add('rational')
            hints['operators'].add('power')
        if predictions.get('power', 0) > 0.3 or predictions.get('polynomial', 0) > 0.3:
            hints['operators'].add('power')

    # Deduplicate frequencies
    hints['frequencies'] = list(set(hints['frequencies']))
    hints['powers'] = list(set(hints['powers']))
    
    return hints


def create_guided_onn_factory(
    operator_hints: Dict,
    n_inputs: int = 1,
    n_hidden_layers: int = 2,
    nodes_per_layer: int = 4,
):
    """
    Create an ONN model factory that is biased toward specific operators.
    
    This allows evolution to start with a population already inclined
    toward the operators identified by the fast-path.
    
    Args:
        operator_hints: Dict from extract_operator_hints()
        n_inputs: Number of input features
        n_hidden_layers: Number of hidden layers
        nodes_per_layer: Nodes per layer
        
    Returns:
        A factory function that creates operator-biased ONNs
    """
    from glassbox.sr.core.operation_dag import OperationDAG

    if operator_hints is None:
        operator_hints = {}
    
    # Determine if we need full ops or simplified
    needs_exp = 'exp' in operator_hints.get('operators', set())
    needs_log = 'log' in operator_hints.get('operators', set())
    use_simplified = not (needs_exp or needs_log)
    
    def factory():
        model = OperationDAG(
            n_inputs=n_inputs,
            n_hidden_layers=n_hidden_layers,
            nodes_per_layer=nodes_per_layer,
            n_outputs=1,
            simplified_ops=use_simplified,
            fair_mode=True,
        )
        
        # Bias operation selectors toward detected operators
        bias_onn_toward_operators(model, operator_hints)
        
        return model
    
    return factory


def bias_onn_toward_operators(model, operator_hints: Dict, bias_strength: float = 2.0):
    """
    Bias an ONN's operation selectors toward specific operators.
    
    Modifies the model's selector logits to favor the operators
    identified in the fast-path formula.
    
    Args:
        model: OperationDAG model
        operator_hints: Dict with 'operators' set
        bias_strength: How strongly to bias (higher = more deterministic)
    """
    if operator_hints is None:
        operator_hints = {}

    operators = operator_hints.get('operators', set())
    frequencies = operator_hints.get('frequencies', [])
    
    for layer in model.layers:
        for node in layer.nodes:
            # Access the operation selector
            selector = node.op_selector
            
            # Bias unary operation selection
            # Indices in simplified mode: 0=periodic, 1=power
            # Indices in full mode: 0=periodic, 1=power, 2=exp, 3=log
            with torch.no_grad():
                unary_start = 2
                unary_end = 2 + selector.n_unary
                # Bias toward periodic if sin/cos detected
                if 'periodic' in operators or 'sin' in operators or 'cos' in operators:
                    if unary_start < unary_end:
                        selector.logits.data[unary_start + 0] += bias_strength
                
                # Bias toward power if power/rational detected
                if 'power' in operators or 'rational' in operators:
                    if unary_start + 1 < unary_end:
                        selector.logits.data[unary_start + 1] += bias_strength
                
                # Full mode: exp and log
                if selector.n_unary >= 3 and 'exp' in operators:
                    selector.logits.data[unary_start + 2] += bias_strength
                if selector.n_unary >= 4 and 'log' in operators:
                    selector.logits.data[unary_start + 3] += bias_strength
                
                # If we have specific frequencies, bias the omega parameter
                if frequencies and hasattr(node, 'unary_ops'):
                    for op in node.unary_ops:
                        if hasattr(op, 'omega') and len(frequencies) > 0:
                            # Set omega to average detected frequency
                            avg_freq = sum(frequencies) / len(frequencies)
                            op.omega.data.fill_(avg_freq)


def _build_cpp_seed_graphs(
    candidate_formulas: Optional[List[Dict]],
    max_seeds: int = 10,
) -> List[Dict]:
    """Convert fast-path / proposer formula strings into C++ seed graph dicts."""
    if not candidate_formulas:
        return []
    try:
        from glassbox.sr.cpp.seed_graph_builder import build_seed_graphs_from_candidates
    except ImportError:
        import sys
        from pathlib import Path as _Path

        _cpp_dir = _Path(__file__).resolve().parent.parent / "glassbox" / "sr" / "cpp"
        if str(_cpp_dir) not in sys.path:
            sys.path.insert(0, str(_cpp_dir))
        from seed_graph_builder import build_seed_graphs_from_candidates  # type: ignore

    return build_seed_graphs_from_candidates(candidate_formulas, max_seeds=max_seeds)


def _build_signal_seed_graphs(
    x: torch.Tensor,
    y: torch.Tensor,
    operator_hints: Dict,
    max_seeds: int = 10,
) -> List[Dict]:
    """Build universal separability-aware seed graphs from the observed signal."""
    try:
        from glassbox.sr.cpp.seed_graph_builder import build_seed_graphs_from_signal
    except ImportError:
        import sys
        from pathlib import Path as _Path

        _cpp_dir = _Path(__file__).resolve().parent.parent / "glassbox" / "sr" / "cpp"
        if str(_cpp_dir) not in sys.path:
            sys.path.insert(0, str(_cpp_dir))
        from seed_graph_builder import build_seed_graphs_from_signal  # type: ignore

    if operator_hints is None:
        operator_hints = {}

    x_np = x.cpu().numpy().ravel()
    y_np = y.cpu().numpy().ravel()
    frequencies = operator_hints.get('frequencies', [])
    return build_seed_graphs_from_signal(
        x_np,
        y_np,
        detected_omegas=frequencies,
        max_seeds=max_seeds,
    )


def beam_search_evolution(
    x: torch.Tensor,
    y: torch.Tensor,
    operator_hints: Dict,
    n_beams: int = 20,
    n_rounds: int = 3,
    keep_fraction: float = 0.2,
    base_pop_size: int = 50,
    base_generations: int = 500,
    device: str = 'cpu',
    candidate_formulas: Optional[List[Dict]] = None,
    confidence: float = 0.5, # New parameter
    search_plan: Optional[Dict[str, Any]] = None,
) -> Dict:
    """
    Beam search over diverse C++ evolution configurations.
    
    Generates K diverse beams (different op_priors, seed_omegas, graph sizes),
    runs them in parallel via C++ backend, prunes bottom 80%, mutates top 20%,
    and repeats for R rounds.
    
    Args:
        x: Input tensor (N,) or (N,1)
        y: Target tensor (N,) or (N,1)
        operator_hints: From extract_operator_hints()
        n_beams: Number of beams per round (default 20)
        n_rounds: Number of tournament rounds (default 3)
        keep_fraction: Fraction of beams to keep each round (default 0.2)
        base_pop_size: Base population size for C++ runs
        base_generations: Base generation count for C++ runs
        device: Device string
        candidate_formulas: Optional list of top-K fast-path candidates,
            each a dict with at least 'formula', 'mse', 'active_terms'.
            Used for targeted population initialization (elite seeding).
        
    Returns:
        Dict with 'formula', 'mse', 'model', 'time', or None if C++ unavailable
    """
    import time
    _core, _ = _load_cpp_core()
    if _core is None or not hasattr(_core, "run_evolution"):
        return None

    if operator_hints is None:
        operator_hints = {}
    if search_plan is None:
        search_plan = {}
    
    start_time = time.time()
    
    # Prepare data for C++
    x_np = x.cpu().numpy().ravel()
    y_np = y.cpu().numpy().ravel()
    X_list = [x_np]
    
    # Extract hints
    operators = operator_hints.get('operators', set())
    frequencies = operator_hints.get('frequencies', [])
    powers = operator_hints.get('powers', [])
    
    has_sin = 'sin' in operators or 'periodic' in operators or 'cos' in operators
    has_power = 'power' in operators
    has_exp = 'exp' in operators
    has_log = 'log' in operators

    def _estimate_polynomial_signature(
        x_values: np.ndarray,
        y_values: np.ndarray,
        max_degree: int = 8,
    ) -> Tuple[bool, int, float]:
        """Return (is_poly_like, best_degree, relative_mse) from a quick poly fit."""
        if x_values.size < max_degree + 2:
            return (False, 0, float('inf'))
        try:
            x_span = float(np.max(x_values) - np.min(x_values))
            x_scaled = (x_values - float(np.mean(x_values))) / max(0.5 * x_span, 1e-8)
            y_var = max(float(np.var(y_values)), 1e-12)

            best_degree = 1
            best_mse = float('inf')
            mse_by_degree: List[Tuple[int, float]] = []
            for degree in range(1, max_degree + 1):
                coeffs = np.polyfit(x_scaled, y_values, degree)
                pred = np.polyval(coeffs, x_scaled)
                mse = float(np.mean((pred - y_values) ** 2))
                mse_by_degree.append((degree, mse))
                if np.isfinite(mse) and mse < best_mse:
                    best_degree = degree
                    best_mse = mse

            # Prefer the smallest degree that is effectively exact. This avoids
            # always returning max_degree when multiple degrees fit near machine precision.
            effective_degree = best_degree
            for degree, mse in mse_by_degree:
                if (mse / y_var) < 1e-11:
                    effective_degree = degree
                    break

            relative_mse = best_mse / y_var
            is_poly_like = effective_degree >= 2 and relative_mse < 1e-11
            return (is_poly_like, effective_degree, relative_mse)
        except Exception:
            return (False, 0, float('inf'))

    hinted_max_power = 0.0
    for power in powers:
        try:
            hinted_max_power = max(hinted_max_power, abs(float(power)))
        except Exception:
            continue

    poly_like_data, poly_degree, poly_rel_mse = _estimate_polynomial_signature(x_np, y_np)
    hint_poly_only = has_power and not (has_sin or has_exp or has_log)
    polynomial_mode = bool(poly_like_data or hint_poly_only or hinted_max_power > 3.0)

    adaptive_p_min = -2.0
    adaptive_p_max = 3.0
    if polynomial_mode:
        target_power = max(float(poly_degree), hinted_max_power)
        if target_power > 3.0:
            # Keep one exponent of headroom so x^k can refine without clamping.
            adaptive_p_max = min(8.0, max(4.0, float(math.ceil(target_power + 1.0))))
        elif hint_poly_only and hinted_max_power > 0.0:
            adaptive_p_max = min(8.0, max(4.0, float(math.ceil(hinted_max_power + 1.0))))

    if "p_min" in search_plan:
        try:
            adaptive_p_min = float(search_plan["p_min"])
        except (TypeError, ValueError):
            pass
    if "p_max" in search_plan:
        try:
            adaptive_p_max = float(search_plan["p_max"])
        except (TypeError, ValueError):
            pass
    if adaptive_p_min >= adaptive_p_max:
        adaptive_p_min, adaptive_p_max = min(adaptive_p_min, adaptive_p_max - 0.5), max(adaptive_p_max, adaptive_p_min + 0.5)
    
    # Build classifier-guided op_priors base
    # Op order: [Periodic, Power, Exp, Log]
    classifier_priors_raw = [
        0.6 if has_sin else 0.1,
        0.6 if has_power else 0.1,
        0.4 if has_exp else 0.05,
        0.2 if has_log else 0.05,
    ]
    prior_trust = _classifier_prior_trust_from_uncertainty(operator_hints.get('uncertainty'))
    classifier_priors = _blend_priors_with_uniform(classifier_priors_raw, prior_trust)
    
    # Build diverse frequency sets
    fft_freqs = frequencies[:3] if frequencies else []
    integer_harmonics = [[1, 2, 3], [2, 3, 5], [1, 3, 5], [3, 6, 9]]
    pi_freqs = [3.14159, 6.28318, 1.5708]
    
    # ── Generate initial beam configurations ──
    def make_beam_configs(n: int, round_idx: int = 0):
        configs = []
        
        # If we have high confidence skeletons, we can be much more selective
        # with the beams we run, focusing on the seeds rather than random exploration.
        is_confident_proposer = bool(confidence > 0.8 and candidate_formulas)

        def add_config(
            op_priors_cfg: List[float],
            seed_omegas_cfg: List[float],
            pop_size_cfg: int,
            generations_cfg: int,
            label_cfg: str,
        ) -> None:
            configs.append({
                'op_priors': op_priors_cfg,
                'seed_omegas': seed_omegas_cfg,
                'pop_size': max(10, int(pop_size_cfg)),
                'generations': max(10, int(generations_cfg)),
                'p_min': adaptive_p_min,
                'p_max': adaptive_p_max,
                'label': label_cfg,
            })
        
        # ── Targeted initialization: inject top-K fast-path candidates as elite seeds ──
        if candidate_formulas:
            for ci, cand in enumerate(candidate_formulas[:3]):
                active_terms = cand.get('active_terms', [])
                cand_priors = list(classifier_priors)
                has_trig = any('sin' in t or 'cos' in t for t in active_terms)
                has_poly = any('x^' in t or 'x**' in t for t in active_terms)
                has_exp_ = any('exp' in t for t in active_terms)
                has_log_ = any('log' in t for t in active_terms)
                if has_trig and len(cand_priors) > 0:
                    cand_priors[0] = min(cand_priors[0] * 1.5, 0.6)
                if has_poly and len(cand_priors) > 1:
                    cand_priors[1] = min(cand_priors[1] * 1.5, 0.6)
                if has_exp_ and len(cand_priors) > 2:
                    cand_priors[2] = min(cand_priors[2] * 1.5, 0.6)
                if has_log_ and len(cand_priors) > 3:
                    cand_priors[3] = min(cand_priors[3] * 1.5, 0.6)
                total_cp = sum(cand_priors) or 1.0
                cand_priors = [p / total_cp for p in cand_priors]

                cand_omegas = list(fft_freqs)
                for t in active_terms:
                    m = re.search(r'(?:sin|cos)\((\d+\.?\d*)\*', t)
                    if m:
                        try:
                            cand_omegas.append(float(m.group(1)))
                        except ValueError:
                            pass

                add_config(cand_priors, cand_omegas[:4], base_pop_size, base_generations,
                           f'candidate-seed-{ci}')
        
        if is_confident_proposer:
            # For confident proposer, we only add a few fallback exploratory beams
            # instead of the full diverse suite.
            add_config(classifier_priors, fft_freqs, base_pop_size, base_generations, 'classifier-guided')
            add_config([0.25, 0.25, 0.25, 0.25], fft_freqs, base_pop_size, base_generations, 'uniform')
            return configs[:n]

        # 1. Classifier-guided (primary hypothesis)
        add_config(classifier_priors, fft_freqs, base_pop_size, base_generations, 'classifier-guided')
        
        # 2. Sin-heavy
        add_config([0.8, 0.1, 0.05, 0.05], fft_freqs or [1.0, 2.0, 3.0], base_pop_size, base_generations, 'sin-heavy')
        
        # 3. Power-heavy
        add_config([0.1, 0.8, 0.05, 0.05], [], base_pop_size, base_generations, 'power-heavy')

        if polynomial_mode and adaptive_p_max > 3.0:
            add_config(
                [0.04, 0.92, 0.02, 0.02],
                [],
                base_pop_size,
                int(base_generations * 1.4),
                'poly-high-power',
            )
            add_config(
                [0.02, 0.95, 0.02, 0.01],
                [],
                max(18, base_pop_size // 2),
                int(base_generations * 2.0),
                'poly-depth',
            )
        
        # 4. Exp-heavy
        add_config([0.05, 0.1, 0.8, 0.05], [], base_pop_size, base_generations, 'exp-heavy')
        
        # 5. Uniform exploration
        add_config([0.25, 0.25, 0.25, 0.25], fft_freqs, base_pop_size, base_generations, 'uniform')
        
        # 6. Sin+Power combo (common case like x^2 + sin(x))
        add_config([0.45, 0.45, 0.05, 0.05], fft_freqs or [1.0, 2.0], base_pop_size, base_generations, 'sin+power')
        
        # 7. Exp+Sin combo (damped oscillation)
        add_config([0.4, 0.1, 0.4, 0.1], fft_freqs or [1.0, 3.0], base_pop_size, base_generations, 'exp+sin')
        
        # 8-11. Integer harmonic frequency variants
        for i, harmonics in enumerate(integer_harmonics):
            add_config(
                classifier_priors,
                [float(h) for h in harmonics],
                base_pop_size,
                base_generations // 2,
                f'harmonics-{harmonics}',
            )
        
        # 12. Pi-based frequencies
        add_config(classifier_priors, pi_freqs, base_pop_size, base_generations // 2, 'pi-freqs')
        
        # 13. No priors (pure uniform random)
        add_config([], [], base_pop_size, base_generations, 'no-priors')
        
        # 14. Large population, fewer gens (breadth-first)
        add_config(classifier_priors, fft_freqs, base_pop_size * 2, base_generations // 3, 'breadth-first')
        
        # 15. Small population, more gens (depth-first)
        add_config(classifier_priors, fft_freqs, max(15, base_pop_size // 3), base_generations * 2, 'depth-first')
        
        # 16-20. Random perturbations of classifier priors
        rng = np.random.RandomState(42 + round_idx)
        while len(configs) < n:
            noise = rng.dirichlet([2, 2, 1, 1])
            blended = [0.5 * c + 0.5 * n_ for c, n_ in zip(classifier_priors, noise)]
            total_b = sum(blended)
            blended = [b / total_b for b in blended]
            
            # Random omega selection
            all_omegas = fft_freqs + [1.0, 2.0, 3.0, 4.0, 5.0, 3.14]
            n_pick = rng.randint(1, min(4, len(all_omegas) + 1))
            picked = list(rng.choice(all_omegas, size=n_pick, replace=False))
            
            add_config(blended, picked, base_pop_size, base_generations, f'random-{len(configs)}')
        
        return configs[:n]
    
    def run_single_beam(config):
        """Run one C++ evolution beam. Returns (mse, result_dict, config)."""
        try:
            result = _core.run_evolution(
                X_list=X_list,
                y=y_np,
                pop_size=config['pop_size'],
                generations=config['generations'],
                early_stop_mse=1e-10,
                seed_omegas=config['seed_omegas'],
                op_priors=config['op_priors'],
                p_min=float(config.get('p_min', -2.0)),
                p_max=float(config.get('p_max', 3.0)),
                num_threads=config.get('num_threads', -1),
            )
            raw_mse = result['best_mse']
            formula_str = result.get('formula', '0')
            
            # Defer expensive SymPy parsing until after the parallel phase
            result['raw_mse'] = raw_mse
            result['best_mse'] = raw_mse
            return (raw_mse, result, config)
        except Exception:
            return (float('inf'), None, config)
    
    def mutate_config(config, rng):
        """Create a mutated variant of a winning beam config."""
        new = dict(config)
        new['label'] = f"mut-{config['label']}"
        
        # Perturb op_priors by ±20%
        if config['op_priors']:
            priors = list(config['op_priors'])
            noise = rng.uniform(0.8, 1.2, size=len(priors))
            priors = [max(0.01, p * n) for p, n in zip(priors, noise)]
            total_p = sum(priors)
            new['op_priors'] = [p / total_p for p in priors]
        
        # Randomly add/remove a frequency
        omegas = list(config.get('seed_omegas', []))
        if rng.random() < 0.5 and len(omegas) > 1:
            omegas.pop(rng.randint(0, len(omegas)))
        elif rng.random() < 0.5:
            # Add a nearby frequency
            new_omega = rng.choice([1, 2, 3, 4, 5, 6, 3.14, 6.28])
            omegas.append(float(new_omega))
        new['seed_omegas'] = omegas
        
        # Slight generation/pop variation
        if rng.random() < 0.3:
            new['generations'] = int(config['generations'] * rng.uniform(0.7, 1.5))
        if rng.random() < 0.3:
            new['pop_size'] = max(15, int(config['pop_size'] * rng.uniform(0.7, 1.3)))

        if polynomial_mode and rng.random() < 0.4:
            p_max = float(new.get('p_max', adaptive_p_max))
            p_max += float(rng.choice([-1.0, 0.0, 1.0]))
            new['p_max'] = float(np.clip(p_max, 3.0, 8.0))
            new['p_min'] = min(float(new.get('p_min', adaptive_p_min)), new['p_max'] - 0.5)
        
        return new
    
    # ── Main beam search loop ──
    print("\n" + "="*60)
    print("BEAM SEARCH EVOLUTION")
    print("="*60)
    print(f"  Beams per round: {n_beams}")
    print(f"  Rounds: {n_rounds}")
    print(f"  Confidence: {confidence:.2f}")
    print(f"  Keep fraction: {keep_fraction}")
    print(f"  Base config: pop={base_pop_size}, gens={base_generations}")
    print(f"  Classifier prior trust: {prior_trust:.2f}")
    print(f"  Power bounds: [{adaptive_p_min:.1f}, {adaptive_p_max:.1f}]")
    if polynomial_mode:
        print(
            f"  Polynomial mode: on (deg={poly_degree}, rel={poly_rel_mse:.2e}, "
            f"target_p={max(float(poly_degree), hinted_max_power)})"
        )
    
    # 1. Get initial configs
    configs = make_beam_configs(n=n_beams, round_idx=0)
    
    best_overall_mse = float('inf')
    best_overall_result = None
    best_overall_config = None
    rng = np.random.RandomState(123)
    
    # 1. Get initial configs
    configs = make_beam_configs(n=n_beams, round_idx=0)

    # 2. Extract multi-priors and multi-seed-omegas
    multi_op_priors = [cfg.get('op_priors', []) for cfg in configs]
    multi_seed_omegas = [cfg.get('seed_omegas', []) for cfg in configs]

    # 3. Launch the native C++ Island Model (one call to rule them all)
    total_pop_size = base_pop_size * n_beams
    total_generations = base_generations * max(1, n_rounds)

    print(f"  Launching Native Diverse Island Model:")
    print(f"  - Islands: {n_beams}")
    print(f"  - Total Population: {total_pop_size}")
    print(f"  - Generations: {total_generations}")

    import multiprocessing
    max_physical_threads = multiprocessing.cpu_count()

    # Seed first fraction of each island population from proposer / fast-path skeletons.
    try:
        planned_seed_budget = int(search_plan.get("seed_budget", 0) or 0)
    except (TypeError, ValueError):
        planned_seed_budget = 0
    total_seed_budget = planned_seed_budget if planned_seed_budget > 0 else 12
    candidate_seed_limit = (
        planned_seed_budget
        if planned_seed_budget > 0
        else min(total_seed_budget, max(3, len(candidate_formulas or []) + 1))
    )
    seed_graphs_py = _build_cpp_seed_graphs(
        candidate_formulas,
        max_seeds=candidate_seed_limit,
    )
    signal_seed_graphs = _build_signal_seed_graphs(
        x,
        y,
        operator_hints,
        max_seeds=max(0, total_seed_budget - len(seed_graphs_py)),
    )
    if signal_seed_graphs:
        seed_graphs_py = (seed_graphs_py or []) + signal_seed_graphs
    if seed_graphs_py:
        preview = []
        for cand in (candidate_formulas or [])[:3]:
            f = str(cand.get("formula", "") or "")
            if f:
                preview.append(f[:60] + ("..." if len(f) > 60 else ""))
        print(f"  Evolution seeds: {len(seed_graphs_py)} graph(s) from candidates")
        if preview:
            print(f"    e.g. {preview[0]}")

    try:
        acceptable_complexity = int(search_plan.get("acceptable_complexity", 15) or 15)
        early_stop_max_nodes = int(search_plan.get("early_stop_max_nodes", 50) or 50)
        acceptable_mse = float(search_plan.get("acceptable_mse", 1e-8) or 1e-8)
        timeout_seconds = search_plan.get("timeout_seconds")
        if timeout_seconds is not None:
            timeout_seconds = max(1, int(float(timeout_seconds)))
        evolution_kwargs = {
            "X_list": X_list,
            "y": y_np,
            "pop_size": total_pop_size,
            "generations": total_generations,
            "early_stop_mse": 1e-10,
            "use_nsga2": True,
            "num_islands": n_beams,
            "migration_interval": 25,
            "migration_size": 2,
            "p_min": adaptive_p_min,
            "p_max": adaptive_p_max,
            "acceptable_mse": acceptable_mse,
            "acceptable_complexity": max(1, acceptable_complexity),
            "early_stop_max_nodes": max(1, early_stop_max_nodes),
            "multi_op_priors": multi_op_priors,
            "multi_seed_omegas": multi_seed_omegas,
            "num_threads": max_physical_threads,  # Use all available cores via OpenMP
            "seed_graphs_py": seed_graphs_py,
        }
        if timeout_seconds is not None:
            evolution_kwargs["timeout_seconds"] = timeout_seconds
        result = _core.run_evolution(**evolution_kwargs)
    except Exception as e:
        print(f"  \u274c Native Island Search failed: {e}")
        return None

    best_overall_mse = result.get('best_mse', float('inf'))
    best_overall_result = result
    best_overall_config = configs[0] # Default, as they were merged

    elapsed = time.time() - start_time    
    if best_overall_result is None:
        print("  \u274c Beam search failed (no valid results)")
        return None
        
    # [Optimized] Run SymPy ONLY ONCE on the absolute global winner
    formula_str = best_overall_result.get('formula', '0')
    display_mse = best_overall_mse
    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import parse_expr, standard_transformations, convert_xor, implicit_multiplication_application
        try:
            from simplify_formula import simplify_onn_formula, SnapConfig, snap_formula_floats
        except ImportError:
            from scripts.simplify_formula import simplify_onn_formula, SnapConfig, snap_formula_floats

        sympy_unsafe = "Piecewise(" in formula_str or "Eq(" in formula_str
        if sympy_unsafe:
            formula_str = snap_formula_floats(
                formula_str,
                SnapConfig(int_tol=1e-5, zero_tol=1e-8),
            )
        else:
            try:
                import warnings
                with warnings.catch_warnings():
                    try:
                        from sympy.utilities.exceptions import SymPyDeprecationWarning
                    except Exception:
                        SymPyDeprecationWarning = DeprecationWarning
                    warnings.filterwarnings("ignore", category=SymPyDeprecationWarning)
                    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"sympy\..*")
                    warnings.filterwarnings("ignore", message=r"\s*Using non-Expr arguments in Mul.*")
                    sp_expr = simplify_onn_formula(formula_str)[1]
            except Exception:
                transformations = standard_transformations + (convert_xor, implicit_multiplication_application)
                snapped = snap_formula_floats(formula_str, SnapConfig(int_tol=1e-5, zero_tol=1e-8))
                sp_expr = parse_expr(snapped, transformations=transformations, evaluate=False)
            formula_str = str(sp_expr)

        y_pred = _evaluate_formula_values(formula_str, x_np)
        if y_pred is None:
            display_mse = float('inf')
        else:
            display_mse = float(np.mean((y_pred - y_np)**2))
        if not np.isfinite(display_mse):
            display_mse = float('inf')
    except Exception:
        display_mse = best_overall_mse

    drift_penalty = abs(best_overall_mse - display_mse) / max(best_overall_mse, 1e-12)
    best_overall_result['display_mse'] = display_mse
    best_overall_result['drift_penalty'] = drift_penalty
    best_overall_result['formula'] = formula_str

    # Build CppGraphModule from best result
    formula = formula_str
    
    try:
        from glassbox.sr.cpp.export_pytorch import CppGraphModule
        model = CppGraphModule(best_overall_result).to(device)
    except Exception:
        model = None
    
    print(f"\n  Best formula: {formula}")
    print(f"  Display MSE: {display_mse:.2e}")
    print(f"  Raw MSE: {best_overall_mse:.2e} (engine internal)")
    drift = best_overall_result.get('drift_penalty', 0.0)
    print(f"  Drift Penalty: {drift:.2f}")
    print(f"  Best config: {best_overall_config['label']}")
    print(f"  Total beam search time: {elapsed:.2f}s")
    print("="*60)
    
    return {
        'formula': formula,
        'mse': display_mse,
        'raw_mse': best_overall_mse,
        'display_mse': display_mse,
        'model': model,
        'time': elapsed,
        'config': best_overall_config,
        'cpp_ast': best_overall_result,
    }


def run_guided_evolution(
    x: torch.Tensor,
    y: torch.Tensor,
    operator_hints: Dict,
    generations: int = 20,
    population_size: int = 20,
    device: str = 'cpu',
    visualizer = None,
    candidate_formulas: Optional[List[Dict]] = None,
    confidence: float = 0.5,
    search_plan: Optional[Dict[str, Any]] = None,
) -> Dict:
    """
    Run evolution guided by fast-path operator hints.
    
    Primary strategy: beam search over diverse C++ evolution configs.
    Fallback: single PyTorch EvolutionaryONNTrainer run.
    
    Args:
        x: Input tensor
        y: Target tensor
        operator_hints: From extract_operator_hints()
        generations: Number of evolution generations
        population_size: Population size
        device: Device to run on
        visualizer: Optional visualizer
        
    Returns:
        Dict with evolved formula, mse, and timing
    """
    import time

    if operator_hints is None:
        operator_hints = {}
    if search_plan is None:
        search_plan = {}
    
    # ── Primary: Beam Search (fast C++ path) ──
    # Adjust beams and rounds based on requested generations and confidence
    n_beams = 10 if generations >= 100 else max(3, generations // 10)
    n_rounds = 2 if generations >= 100 else 1
    if "n_beams" in search_plan:
        try:
            n_beams = max(1, int(search_plan["n_beams"]))
        except (TypeError, ValueError):
            pass
    if "n_rounds" in search_plan:
        try:
            n_rounds = max(1, int(search_plan["n_rounds"]))
        except (TypeError, ValueError):
            pass
    
    base_pop = population_size
    base_gens = generations

    if confidence > 0.8 and candidate_formulas:
        # High confidence in skeletons → focus beams on refinement
        if "n_beams" not in search_plan:
            n_beams = min(n_beams, len(candidate_formulas) + 2)
        if "n_rounds" not in search_plan:
            n_rounds = 1 # One round is enough to check the seeds
        # REMOVED: base_gens = min(base_gens, 150)
        # REMOVED: base_pop = min(base_pop, 50)
        print(f"  [Adaptive] Confident proposer: focusing search on {n_beams} beams, {n_rounds} round(s), {base_pop} pop, {base_gens} gens.")
    elif candidate_formulas:
        # Proposer gave skeletons but isn't super confident
        if "n_beams" not in search_plan:
            n_beams = min(n_beams, 7)
        if "n_rounds" not in search_plan:
            n_rounds = 1
        # REMOVED: base_gens = min(base_gens, 250)
        # REMOVED: base_pop = min(base_pop, 60)
        print(f"  [Adaptive] Proposer candidates available: focusing search on {n_beams} beams, {n_rounds} round(s), {base_pop} pop, {base_gens} gens.")

    beam_result = beam_search_evolution(
        x, y,
        operator_hints,
        n_beams=n_beams,
        n_rounds=n_rounds,
        keep_fraction=0.3,
        base_pop_size=base_pop,
        base_generations=base_gens,
        device=device,
        candidate_formulas=candidate_formulas,
        confidence=confidence,
        search_plan=search_plan,
    )
    
    if beam_result is not None and beam_result['mse'] < float('inf'):
        return beam_result
    
    # ── Fallback: Single PyTorch ONN evolution ──
    print("\n⚠️ Beam search unavailable, falling back to single PyTorch evolution...")
    
    from glassbox.evolution import EvolutionaryONNTrainer, finalize_model_coefficients
    
    print("\n" + "="*60)
    print("GUIDED EVOLUTION: Operator-Constrained Search (PyTorch)")
    print("="*60)
    print(f"  Active operators: {operator_hints.get('operators', set())}")
    print(f"  Detected frequencies: {operator_hints.get('frequencies', [])}")
    print(f"  Active terms: {operator_hints.get('active_terms', [])[:5]}...")
    
    start_time = time.time()
    
    # Create operator-biased model factory
    factory = create_guided_onn_factory(
        operator_hints,
        n_inputs=1,
        n_hidden_layers=2,
        nodes_per_layer=4,
    )
    
    # Create trainer with smaller population (faster, since we have hints)
    trainer = EvolutionaryONNTrainer(
        model_factory=factory,
        population_size=population_size,
        elite_size=4,
        mutation_rate=0.4,
        constant_refine_steps=50,
        complexity_penalty=0.01,
        device=device,
        lamarckian=True,
        use_explorers=True,
        explorer_fraction=0.3,
        explorer_mutation_rate=0.7,
        nested_bfgs=True,
        nested_bfgs_every=5,
        nested_bfgs_steps=20,
    )
    
    # Initialize and evolve
    trainer.initialize_population()
    results = trainer.train(x, y, generations=generations, print_every=5)
    
    # Finalize best model
    if trainer.best_ever:
        final_model = trainer.best_ever.model
        final_mse, final_formula = finalize_model_coefficients(
            final_model, x, y,
            refine_internal_constants=True,
        )
    elif results and 'model' in results:
        final_model = results['model']
        final_mse = results.get('final_mse', float('inf'))
        final_formula = results.get('formula', 'Unknown')
    else:
        return None
    
    elapsed = time.time() - start_time
    
    print(f"\n  Evolved Formula: {final_formula}")
    print(f"  Final MSE: {final_mse:.6f}")
    print(f"  Evolution Time: {elapsed:.2f}s")
    print("="*60)
    
    return {
        'formula': final_formula,
        'mse': final_mse,
        'time': elapsed,
        'model': final_model,
    }
