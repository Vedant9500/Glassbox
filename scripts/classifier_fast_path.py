"""
Classifier-Guided Fast Path for Symbolic Regression

When the curve classifier predicts operators with high confidence,
skip Phase 1 evolution entirely and directly run regression.

This can reduce solve time from ~300s to <10s for well-predicted formulas.
"""

import re
import math
import threading
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import permutations
import numpy as np
import torch
from typing import Any, Dict, List, Tuple, Optional

from glassbox.sr.operations.meta_ops import get_constant_symbol, normalize_formula_ascii
from glassbox.sr.fpip_v2 import build_fpip_v2_from_fast_path, validate_fpip_v2_payload

# Thread-safe CUDA warning state
_warned_no_cuda = False
_cuda_warning_lock = threading.Lock()

# Pre-compiled regex patterns for performance
_FREQ_SIN_PATTERN = re.compile(r'sin\(([0-9.]+)\*?x', re.IGNORECASE)
_FREQ_COS_PATTERN = re.compile(r'cos\(([0-9.]+)\*?x', re.IGNORECASE)
_POWER_PATTERN = re.compile(r'x\^([0-9.]+)', re.IGNORECASE)

DEFAULT_CURVE_CLASSIFIER_PATH = "models/curve_classifier_v3.1.pt"


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
    expr = parse_expr(normalized_formula, transformations=transformations, evaluate=False)
    free_syms = sorted(expr.free_symbols, key=lambda sym: sym.name)

    if not free_syms:
        return tuple(), float(expr), None

    # Inject safe power into lambdify context
    modules = [{"pow": _safe_numpy_power, "Pow": _safe_numpy_power}, "numpy"]
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


def lasso_coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.1,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> np.ndarray:
    """
    LASSO regression using coordinate descent.
    
    Solves: min_w ||y - X @ w||^2 + alpha * ||w||_1
    
    Optimized: Computes residual incrementally per-feature instead of
    full matrix multiply, reducing complexity from O(n×m²) to O(n×m) per iteration.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        alpha: L1 regularization strength
        max_iter: Maximum iterations
        tol: Convergence tolerance
        
    Returns:
        w: Coefficient vector (n_features,)
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    
    # Precompute for efficiency
    X_sq = (X ** 2).sum(axis=0)
    X_sq[X_sq < 1e-10] = 1e-10  # Avoid division by zero
    
    # Precompute threshold
    threshold = alpha * n_samples
    
    for iteration in range(max_iter):
        w_old = w.copy()
        
        # Compute residual once per iteration (O(n×m))
        residual = y - X @ w
        
        for j in range(n_features):
            # Add back current feature contribution
            residual_j = residual + X[:, j] * w[j]
            
            # Correlation
            rho = X[:, j] @ residual_j
            
            # Soft thresholding
            if alpha == 0:
                w[j] = rho / X_sq[j]
            else:
                w[j] = soft_threshold(rho, threshold) / X_sq[j]
            
            # Update residual incrementally (O(n) instead of O(n×m))
            residual = residual_j - X[:, j] * w[j]
        
        # Check convergence
        if np.max(np.abs(w - w_old)) < tol:
            break
    
    return w


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
            
            # Fractional powers - critical for formulas like x^2.3, x^1.5
            # Use absolute value to handle negative x values
            xi_safe = np.abs(xi) + 1e-10  # Avoid 0^fractional
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
        universal_basis or
        predictions.get('sin', 0) >= threshold or
        predictions.get('cos', 0) >= threshold or
        predictions.get('periodic', 0) >= threshold
    )

    if include_periodic:
        for i in range(n_vars):
            xi = x[:, i]
            name = var_name(i)

            for omega in omegas[:6]:  # Increased to 6 to fit pi frequencies
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

            for omega in omegas[:6]:
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
        universal_basis or
        predictions.get('exp', 0) >= threshold or 
        predictions.get('exponential', 0) >= threshold
    )
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
            
            # If the classifier strongly predicts 'e' or 'pi' as a base
            if predictions.get('const_e', 0) >= threshold:
                exps = np.exp(x_clamp * math.e)
                basis_list.append(exps)
                names.append(f"exp(e*{name})")
                basis_list.append(np.exp(-x_clamp * math.e))
                names.append(f"exp(-e*{name})")
                
            if predictions.get('const_pi', 0) >= threshold:
                exps = np.exp(x_clamp * math.pi)
                basis_list.append(exps)
                names.append(f"exp(pi*{name})")
                basis_list.append(np.exp(-x_clamp * math.pi))
                names.append(f"exp(-pi*{name})")
    
    # Logarithmic operations (always include in universal mode for Nguyen-7 etc.)
    if allow_log and (universal_basis or predictions.get('log', 0) >= threshold):
        for i in range(n_vars):
            xi = x[:, i]
            name = var_name(i)
            x_safe = np.maximum(np.abs(xi), 1e-10)
            basis_list.append(np.log(x_safe + 1))
            names.append(f"log({name}+1)")
            basis_list.append(np.log(x_safe**2 + 1))
            names.append(f"log({name}^2+1)")
    
    # Composition terms (for sin(x²), etc. - covers Nguyen-10)
    if universal_basis and allow_periodic:
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
    if universal_basis and allow_power:
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
    if universal_basis and allow_arithmetic and n_vars >= 2:
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
    if universal_basis and allow_power:
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
        
    Returns:
        (formula, mse, coefficients) if exact match found, else None
    """
    n_basis = basis.shape[1]
    y = y.flatten()
    
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
    
    def chunk_ranges(n: int, chunks: int) -> List[Tuple[int, int]]:
        if chunks <= 1 or n <= 1:
            return [(0, n)]
        size = max(1, math.ceil(n / chunks))
        return [(i, min(i + size, n)) for i in range(0, n, size)]

    def build_formula(indices: List[int], coeffs: np.ndarray) -> Tuple[str, np.ndarray]:
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
    
    # STEP 1: Try to find exact symbolic match FIRST (before LASSO)
    # This prevents LASSO from finding approximate solutions when exact ones exist
    if op_constraints:
        allow_periodic = op_constraints.get('periodic', True)
        allow_exp = op_constraints.get('exp', True)
        allow_log = op_constraints.get('log', True)
        allow_power = op_constraints.get('power', True)
    else:
        allow_periodic = allow_exp = allow_log = allow_power = True

    # If only power is allowed, enable 10-term exact match for polynomials
    exact_max_terms = 10 if (allow_power and not allow_periodic and not allow_exp and not allow_log) else 4

    if exact_match_enabled and (exact_match_max_basis is None or basis.shape[1] <= 128):
        exact_match = find_exact_symbolic_match(
            basis,
            names,
            y_fit,
            max_terms=exact_max_terms,
            tolerance=1e-5,
            num_threads=exact_match_threads,
        )
        if exact_match:
            formula, mse, coeffs = exact_match
            print(f"  Found EXACT symbolic match: {formula} (MSE={mse:.2e})")
            active_idx = np.flatnonzero(np.abs(coeffs) >= sparsity_threshold)
            return formula, mse, {
                'coefficients': coeffs,
                'basis_names': names,
                'n_nonzero': sum(1 for c in coeffs if abs(c) >= sparsity_threshold),
                'exact_match': True,
                'candidate_formulas': [{
                    'formula': formula,
                    'mse': float(mse),
                    'score': float(mse),
                    'n_nonzero': int(np.sum(np.abs(coeffs) >= sparsity_threshold)),
                    'active_terms': [names[i] for i in active_idx],
                    'alpha': 0.0,
                }],
            }
    elif exact_match_enabled:
        print(f"  Skipping exact-match search (basis={basis.shape[1]} > {exact_match_max_basis})")
    
    # Normalize basis for numerical stability
    basis_std = np.std(basis, axis=0, keepdims=True)
    basis_std[basis_std < 1e-10] = 1.0
    basis_norm = basis / basis_std

    def _coeffs_to_formula(coeffs_arr: np.ndarray) -> str:
        terms_local = []
        for name, coef in zip(names, coeffs_arr):
            if abs(coef) < sparsity_threshold:
                continue

            if name == "1":
                terms_local.append(get_constant_symbol(coef, threshold=0.05))
            elif abs(coef - 1.0) < 0.01:
                terms_local.append(name)
            elif abs(coef + 1.0) < 0.01:
                terms_local.append(f"-{name}")
            elif abs(coef - round(coef)) < 0.01:
                terms_local.append(f"{int(round(coef))}*{name}")
            else:
                coef_sym = get_constant_symbol(coef, threshold=0.05)
                terms_local.append(f"{coef_sym}*{name}")
        return _join_formula_terms(terms_local)

    def _candidate_signature(coeffs_arr: np.ndarray) -> Tuple[int, ...]:
        return tuple(np.flatnonzero(np.abs(coeffs_arr) >= sparsity_threshold).tolist())

    def _update_candidate_pool(
        pool: Dict[Tuple[int, ...], Dict[str, Any]],
        coeffs_arr: np.ndarray,
        mse_val: float,
        alpha_val: float,
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
    
    # Try multiple alpha values to find best sparsity-accuracy tradeoff
    for alpha in [0.0, 0.001, 0.01, 0.05, 0.1, 0.2]:
        try:
            coeffs = lasso_coordinate_descent(basis_norm, y_fit, alpha=alpha, max_iter=1000)
            
            # Check for NaN/Inf in coeffs
            if not np.all(np.isfinite(coeffs)):
                print(f"  Warning: Alpha={alpha} produced non-finite coefficients")
                continue
            
            # Unnormalize coefficients
            coeffs = coeffs / basis_std.flatten()
            
            # Compute MSE on fit data
            y_pred = basis @ coeffs
            mse = np.mean((y_fit - y_pred) ** 2)
            n_terms = np.sum(np.abs(coeffs) > sparsity_threshold)
            
            # Check if MSE is valid
            if not np.isfinite(mse):
                print(f"  Warning: Alpha={alpha} produced non-finite MSE (max pred: {np.max(np.abs(y_pred))})")
                continue
            
            # Complexity-penalized score
            score = mse + COMPLEXITY_PENALTY * n_terms
            
            # Select best based on penalized score (prefers simpler solutions)
            if score < best_score:
                best_coeffs = coeffs
                best_mse = mse
                best_score = score

            _update_candidate_pool(candidate_pool, coeffs, mse, alpha)
        except Exception as e:
            print(f"  Error with alpha={alpha}: {e}")
            continue
    
    if best_coeffs is None:
        # Fallback to plain least squares
        try:
            best_coeffs, _, _, _ = np.linalg.lstsq(basis, y_fit, rcond=None)
            y_pred = basis @ best_coeffs
            best_mse = np.mean((y_fit - y_pred) ** 2)
            best_score = best_mse + COMPLEXITY_PENALTY * np.sum(np.abs(best_coeffs) >= sparsity_threshold)
            _update_candidate_pool(candidate_pool, best_coeffs, best_mse, alpha_val=-1.0)
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
                }
        except (np.linalg.LinAlgError, ValueError):
            pass

    if not candidate_pool:
        _update_candidate_pool(candidate_pool, best_coeffs, best_mse, alpha_val=-1.0)

    sorted_candidates = sorted(candidate_pool.values(), key=lambda c: (c['score'], c['mse']))
    top_candidates = sorted_candidates[:5]
    best_candidate = top_candidates[0]

    coeffs = best_candidate['coeffs']
    # Report MSE on *full* dataset (including holdout), not just fit subset
    y_pred_full = basis_full @ coeffs
    mse = float(np.mean((y - y_pred_full) ** 2))
    if not np.isfinite(mse):
        mse = float(best_candidate['mse'])
    formula = _coeffs_to_formula(coeffs)

    n_nonzero = int(np.sum(np.abs(coeffs) >= sparsity_threshold))
    exact_match_flag = mse < 1e-6 and n_nonzero <= 10

    candidate_formulas = []
    for cand in top_candidates:
        cand_coeffs = cand['coeffs']
        active_idx = np.flatnonzero(np.abs(cand_coeffs) >= sparsity_threshold)
        candidate_formulas.append({
            'formula': _coeffs_to_formula(cand_coeffs),
            'mse': float(cand['mse']),
            'score': float(cand['score']),
            'n_nonzero': int(cand['n_terms']),
            'active_terms': [names[i] for i in active_idx],
            'alpha': float(cand.get('alpha', -1.0)),
        })

    return formula, mse, {
        'coefficients': coeffs,
        'basis_names': names,
        'n_nonzero': n_nonzero,
        'exact_match': exact_match_flag,
        'candidate_formulas': candidate_formulas,
        'holdout_mse': _holdout_mse_for_best(coeffs) if holdout_mask is not None else None,
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
    
    Args:
        x: Input values (N,)
        y: Target values (N,)
        initial_omegas: Starting frequencies from FFT
        n_steps: Gradient descent steps
        lr: Learning rate
        
    Returns:
        refined_omegas: List of refined frequencies
        final_mse: MSE after refinement
    """
    import torch
    import torch.nn as nn
    
    resolved_device = _resolve_device(device)
    x_t = torch.tensor(x, dtype=torch.float32, device=resolved_device)
    y_t = torch.tensor(y, dtype=torch.float32, device=resolved_device)
    
    # Learnable parameters: constant, linear coef, and sin/cos for each omega
    # Model: c0 + c1*x + sum_i [a_i*sin(omega_i*x) + b_i*cos(omega_i*x)]
    
    class FrequencyModel(nn.Module):
        def __init__(self, omegas):
            super().__init__()
            n_freq = len(omegas)
            self.omegas = nn.Parameter(torch.tensor(omegas, dtype=torch.float32))
            self.sin_coeffs = nn.Parameter(torch.zeros(n_freq))
            self.cos_coeffs = nn.Parameter(torch.zeros(n_freq))
            self.constant = nn.Parameter(torch.tensor(0.0))
            self.linear = nn.Parameter(torch.tensor(0.0))
            self.quadratic = nn.Parameter(torch.tensor(0.0))
        
        def forward(self, x):
            # Polynomial part
            result = self.constant + self.linear * x + self.quadratic * x**2
            
            # Periodic parts
            for i, omega in enumerate(self.omegas):
                result = result + self.sin_coeffs[i] * torch.sin(omega * x)
                result = result + self.cos_coeffs[i] * torch.cos(omega * x)
            
            return result
    
    model = FrequencyModel(initial_omegas).to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_mse = float('inf')
    best_omegas = initial_omegas.copy()
    
    for step in range(n_steps):
        optimizer.zero_grad()
        pred = model(x_t)
        loss = ((pred - y_t) ** 2).mean()
        loss.backward()
        optimizer.step()
        
        mse = loss.item()
        if mse < best_mse:
            best_mse = mse
            best_omegas = model.omegas.detach().cpu().numpy().tolist()
    
    return best_omegas, best_mse


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
    Refine power exponent parameters using gradient descent.

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
    import torch
    import torch.nn as nn

    # Ensure 1D inputs
    if x.ndim > 1:
        x = x.flatten()
    if y.ndim > 1:
        y = y.flatten()

    if initial_powers is None:
        initial_powers = [0.5, 1.5, 2.5, 3.5]

    resolved_device = _resolve_device(device)
    x_t = torch.tensor(x, dtype=torch.float64, device=resolved_device)
    y_t = torch.tensor(y, dtype=torch.float64, device=resolved_device)

    # Filter out x <= 0 for safe power operations
    valid_mask = x_t.abs() > 1e-8
    if valid_mask.sum() < 10:
        return None, float('inf')
    x_valid = x_t[valid_mask]
    y_valid = y_t[valid_mask]

    class PowerModel(nn.Module):
        def __init__(self, powers, omegas=None):
            super().__init__()
            n_pow = len(powers)
            self.powers = nn.Parameter(torch.tensor(powers, dtype=torch.float64))
            self.coeffs = nn.Parameter(torch.zeros(n_pow, dtype=torch.float64))
            self.constant = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
            self.linear = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
            
            self.omegas = omegas
            if omegas:
                self.periodic_coeffs = nn.Parameter(torch.zeros(2 * len(omegas), dtype=torch.float64))
            else:
                self.periodic_coeffs = None

        def forward(self, x):
            abs_x = torch.abs(x) + 1e-10
            sign_x = torch.sign(x)
            result = self.constant + self.linear * x
            
            # Power terms
            for i, p in enumerate(self.powers):
                # Even/odd symmetry based on p
                is_even = 0.5 * (1.0 + torch.cos(p * math.pi))
                abs_pow = torch.pow(abs_x, p)
                term = (1.0 - is_even) * (sign_x * abs_pow) + is_even * abs_pow
                result = result + self.coeffs[i] * term
                
            # Periodic terms
            if self.periodic_coeffs is not None:
                for i, omega in enumerate(self.omegas):
                    result = result + self.periodic_coeffs[2*i] * torch.sin(omega * x)
                    result = result + self.periodic_coeffs[2*i+1] * torch.cos(omega * x)
                    
            return result

    best_result = None
    best_mse = float('inf')

    # Stage 1: Fit powers (and initial omegas if provided)
    # We try different power combinations
    stage1_models = []
    
    for n_powers in range(1, min(4, len(initial_powers) + 1)):
        for start_idx in range(max(1, len(initial_powers) - n_powers + 1)):
            powers_subset = initial_powers[start_idx:start_idx + n_powers]

            model = PowerModel(powers_subset, detected_omegas).to(resolved_device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            for step in range(n_steps):
                optimizer.zero_grad()
                pred = model(x_valid)
                loss = ((pred - y_valid) ** 2).mean()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    model.powers.data.clamp_(-2.0, 5.0)
            
            mse = ((model(x_valid) - y_valid) ** 2).mean().item()
            stage1_models.append((mse, model, powers_subset))
    
    # Sort by MSE and pick best Stage 1 model
    stage1_models.sort(key=lambda x: x[0])
    best_stage1_mse, best_stage1_model, best_powers = stage1_models[0]
    
    # Stage 2: Check residuals for hidden frequencies (if not already found)
    # Only if we have a trend (MSE is low but maybe not perfect, or just to be safe)
    # We assume the main trend handles the non-periodic part.
    final_model = best_stage1_model
    final_mse = best_stage1_mse
    
    # detected_omegas is local variable, update it if we find more
    current_omegas = detected_omegas
    
    from glassbox.evolution import detect_dominant_frequency
    
    with torch.no_grad():
        pred_stage1 = best_stage1_model(x_valid)
        residuals = y_valid - pred_stage1
        
        # Run FFT on residuals
        # We need to reshape for detect_dominant_frequency
        res_omegas = detect_dominant_frequency(x_valid, residuals, n_frequencies=2)
        
        # Filter new omegas - ignore if close to existing ones or 0
        new_omegas = []
        existing_omegas = current_omegas or []
        for o in res_omegas:
            if o < 0.1:
                continue
            if any(abs(o - eo) < 0.2 for eo in existing_omegas):
                continue
            new_omegas.append(o)
    
    if new_omegas:
        # Refit with new omegas
        # print(f"  [Refine] Found hidden frequencies in residuals: {new_omegas}")
        combined_omegas = (current_omegas or []) + new_omegas
        
        # Create new model with best powers + combined omegas
        # Initialize with learned powers/coeffs to speed up
        new_model = PowerModel(best_powers, combined_omegas).to(resolved_device)
        
        # Copy learned params
        with torch.no_grad():
            new_model.powers.data.copy_(best_stage1_model.powers.data)
            new_model.coeffs.data.copy_(best_stage1_model.coeffs.data)
            new_model.constant.data.copy_(best_stage1_model.constant.data)
            new_model.linear.data.copy_(best_stage1_model.linear.data)
            if best_stage1_model.periodic_coeffs is not None:
                # Copy existing periodic coeffs to the beginning
                n_old = len(best_stage1_model.periodic_coeffs)
                new_model.periodic_coeffs.data[:n_old].copy_(best_stage1_model.periodic_coeffs.data)
        
        optimizer = torch.optim.Adam(new_model.parameters(), lr=lr)
        
        for step in range(n_steps):
            optimizer.zero_grad()
            pred = new_model(x_valid)
            loss = ((pred - y_valid) ** 2).mean()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                new_model.powers.data.clamp_(-2.0, 5.0)
            
        mse2 = ((new_model(x_valid) - y_valid) ** 2).mean().item()
        
        if mse2 < final_mse:
            final_model = new_model
            final_mse = mse2
            # Update detected_omegas for reporting
            current_omegas = combined_omegas

    # Final extraction from winning model
    best_mse = final_mse
    with torch.no_grad():
        refined_powers = final_model.powers.detach().cpu().numpy().tolist()
        refined_coeffs = final_model.coeffs.detach().cpu().numpy().tolist()
        const_val = final_model.constant.item()
        linear_val = final_model.linear.item()
        per_coeffs = []
        if final_model.periodic_coeffs is not None:
            per_coeffs = final_model.periodic_coeffs.detach().cpu().numpy().tolist()

    # Build formula
    terms = []
    # Power terms
    # Power terms
    for p, c in zip(refined_powers, refined_coeffs):
        if abs(c) < 1e-3:
            continue
        
        p_snapped = _snap_power(p)
        
        # Merge linear term if power is effectively 1
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
        refined_omegas, freq_mse = refine_frequencies(
            x, y, detected_omegas, n_steps=150, device=device
        )
        results['frequency'] = {
            'omegas': refined_omegas,
            'mse': freq_mse,
        }

    # 2. Power refinement (if power predicted)
    has_power = False
    if predictions:
        if predictions.get('power', 0) > 0.3 or predictions.get('polynomial', 0) > 0.3:
            has_power = True

    if has_power:
        power_result, power_mse = refine_powers(
            x, y, detected_omegas=detected_omegas, device=device
        )
        if power_result is not None:
            results['power'] = power_result

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
    should_try_freq = bool(detected_omegas) and has_periodic_signal and (1e-4 <= best_mse <= 0.2)

    if should_try_freq:
        print(f"  Attempting frequency refinement (initial MSE={best_mse:.4f})...")
        
        refined_omegas, refined_mse = refine_frequencies(
            x, y, detected_omegas, n_steps=refine_steps, device=device
        )
        
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
            )
            if should_accept_candidate(mse2, details2):
                return formula2, mse2, details2
    
    # Additional continuous refinement for periodic×rational terms
    if op_constraints:
        allow_periodic = op_constraints.get('periodic', True)
        allow_power = op_constraints.get('power', True)
    else:
        allow_periodic = allow_power = True

    if (x.ndim == 1 or (x.ndim == 2 and x.shape[1] == 1)) and allow_periodic and allow_power and best_mse > 1e-4:
        omega_pool = (detected_omegas or []) + [1.0, 2.0, 3.0, 5.0, 10.0]
        # Deduplicate and keep reasonable range
        omega_inits = []
        for o in omega_pool:
            if 0.1 < o < 20 and all(abs(o - e) > 0.05 for e in omega_inits):
                omega_inits.append(o)

        refined = refine_periodic_rational(
            x, y,
            omega_inits=omega_inits,
            c_inits=[0.5, 1.0, 2.0, 3.0],
            steps=400,
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
    should_try_power = (x.ndim == 1 or (x.ndim == 2 and x.shape[1] == 1)) and allow_power and has_power_signal and (5e-5 <= best_mse <= 0.2 or n_terms > 6)

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
) -> bool:
    """
    Decide whether to use fast path based on classifier confidence.
    
    Fast path is used when:
    1. At least one operator predicted with high confidence
    2. The predicted operators are well-covered by our basis
    """
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
) -> Optional[Dict]:
    """
    Run the complete fast-path pipeline.
    
    Returns:
        Dict with formula, mse, and timing if successful
        None if fast path not applicable
    """
    import time
    try:
        from curve_classifier_integration import predict_operators
    except ImportError:
        from scripts.curve_classifier_integration import predict_operators
    
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
    uncertainty_metrics = _prediction_uncertainty_metrics(predictions or {})
    
    if not predictions:
        print("  No operators predicted - falling back to evolution")
        return None
    
    print(f"  Predictions: {[(k, f'{v:.2f}') for k, v in sorted(predictions.items(), key=lambda x: -x[1])]}")
    print(
        "  Uncertainty: "
        f"entropy={uncertainty_metrics['prediction_entropy']}, "
        f"margin={uncertainty_metrics['prediction_margin']}"
    )
    
    # Check if fast path is applicable (lowered threshold to 0.6 for broader coverage)
    if not should_use_fast_path(predictions, confidence_threshold=0.6):
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

            if too_complex_for_symbolic:
                snapped_formula = snap_formula_floats(
                    formula,
                    SnapConfig(int_tol=simplification_int_tol, zero_tol=simplification_zero_tol),
                )
                simplified_formula = snapped_formula
                if simplification_log:
                    print(f"  [Post] Large formula detected (len={formula_len}, terms≈{term_estimate}); using fast snap-only mode.")
            else:
                snapped_formula, simplified_expr = simplify_onn_formula(
                    formula,
                    int_tol=simplification_int_tol,
                    zero_tol=simplification_zero_tol,
                    use_nsimplify=(formula_len <= 300 and term_estimate <= 16),
                )
                simplified_formula = str(simplified_expr)

            if simplification_log:
                print(f"  [Post] Snapped: {snapped_formula}")
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
                                            details.get('coefficients', []))
    
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
    import sys
    from pathlib import Path as _Path
    
    # Try importing C++ backend
    try:
        cpp_dir = _Path(__file__).parent.parent / 'glassbox' / 'sr' / 'cpp'
        if str(cpp_dir) not in sys.path:
            sys.path.insert(0, str(cpp_dir))
        import _core
    except ImportError:
        # Also try relative to the glassbox package
        try:
            cpp_dir = _Path(__file__).resolve().parent.parent / 'glassbox' / 'sr' / 'cpp'
            if str(cpp_dir) not in sys.path:
                sys.path.insert(0, str(cpp_dir))
            import _core
        except ImportError:
            return None
    
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
            )
            raw_mse = result['best_mse']
            formula_str = result.get('formula', '0')
            
            try:
                import sympy as sp
                from sympy.parsing.sympy_parser import parse_expr, standard_transformations, convert_xor, implicit_multiplication_application
                try:
                    from simplify_formula import simplify_onn_formula, SnapConfig, snap_formula_floats
                except ImportError:
                    from scripts.simplify_formula import simplify_onn_formula, SnapConfig, snap_formula_floats
                
                try:
                    sp_expr = simplify_onn_formula(formula_str)[1]
                except Exception:
                    transformations = standard_transformations + (convert_xor, implicit_multiplication_application)
                    snapped = snap_formula_floats(formula_str, SnapConfig(int_tol=1e-5, zero_tol=1e-8))
                    sp_expr = parse_expr(snapped, transformations=transformations, evaluate=False)
                
                free_syms = list(sp_expr.free_symbols)
                if not free_syms:
                    y_pred = np.full_like(y_np, float(sp_expr))
                else:
                    func = sp.lambdify(free_syms[0], sp_expr, modules=['numpy'])
                    y_pred = func(x_np)
                
                if np.isscalar(y_pred):
                    y_pred = np.full_like(y_np, y_pred)
                    
                display_mse = float(np.mean((y_pred - y_np)**2))
                if not np.isfinite(display_mse):
                    display_mse = float('inf')
                    
            except Exception:
                display_mse = raw_mse
                
            drift_penalty = abs(raw_mse - display_mse) / max(raw_mse, 1e-12)
            result['raw_mse'] = raw_mse
            result['display_mse'] = display_mse
            result['drift_penalty'] = drift_penalty
            
            # Use display_mse as the primary ranking metric instead of raw_mse
            result['best_mse'] = display_mse
            return (display_mse, result, config)
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
    
    for round_idx in range(n_rounds):
        if round_idx == 0:
            configs = make_beam_configs(n_beams, round_idx)
        else:
            # Mutate winners into new beams
            configs = []
            n_keep = max(1, int(n_beams * keep_fraction))
            n_mutants_per_winner = max(1, (n_beams - n_keep) // n_keep)
            
            for winner_cfg in winner_configs:  # noqa: F821 - set at end of loop
                configs.append(winner_cfg)  # Keep original
                for _ in range(n_mutants_per_winner):
                    configs.append(mutate_config(winner_cfg, rng))
            
            # Fill remaining with random configs
            while len(configs) < n_beams:
                configs.append(mutate_config(configs[rng.randint(0, len(configs))], rng))
            configs = configs[:n_beams]
        
        # Run all beams sequentially (C++ uses OpenMP internally, so 
        # threading would cause oversubscription and hurt performance)
        round_start = time.time()
        results = []
        
        for cfg in configs:
            results.append(run_single_beam(cfg))
        
        # Sort by MSE
        results.sort(key=lambda r: r[0])
        round_time = time.time() - round_start
        
        # Report
        best_mse = results[0][0]
        worst_kept = results[max(0, int(n_beams * keep_fraction) - 1)][0]
        
        print(f"\n  Round {round_idx + 1}/{n_rounds}: "
              f"best={best_mse:.2e}, "
              f"worst_kept={worst_kept:.2e}, "
              f"time={round_time:.2f}s, "
              f"winner='{results[0][2]['label']}'")
        
        # Track overall best
        if results[0][0] < best_overall_mse:
            best_overall_mse = results[0][0]
            best_overall_result = results[0][1]
            best_overall_config = results[0][2]
        
        # Early exit if we found essentially perfect MSE
        if best_overall_mse < 1e-10:
            print("  [PASS] Perfect MSE achieved, stopping early!")
            break
        
        # Keep top fraction as winners for next round
        n_keep = max(1, int(n_beams * keep_fraction))
        winner_configs = [r[2] for r in results[:n_keep]]
    
    elapsed = time.time() - start_time
    
    if best_overall_result is None:
        print("  \u274c Beam search failed (no valid results)")
        return None
    
    # Build CppGraphModule from best result
    formula = best_overall_result.get('formula', '0')
    
    try:
        from glassbox.sr.cpp.export_pytorch import CppGraphModule
        model = CppGraphModule(best_overall_result).to(device)
    except Exception:
        model = None
    
    print(f"\n  Best formula: {formula}")
    print(f"  Best MSE: {best_overall_mse:.2e} (displayed formula)")
    if 'raw_mse' in best_overall_result:
        raw = best_overall_result['raw_mse']
        drift = best_overall_result.get('drift_penalty', 0.0)
        print(f"  Raw MSE: {raw:.2e} (engine internal)")
        print(f"  Drift Penalty: {drift:.2f}")
    print(f"  Best config: {best_overall_config['label']}")
    print(f"  Total beam search time: {elapsed:.2f}s")
    print("="*60)
    
    return {
        'formula': formula,
        'mse': best_overall_mse,
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
    
    # ── Primary: Beam Search (fast C++ path) ──
    # Adjust beams and rounds based on requested generations and confidence
    n_beams = 10 if generations >= 100 else max(3, generations // 10)
    n_rounds = 2 if generations >= 100 else 1
    
    base_pop = population_size
    base_gens = generations

    if confidence > 0.8 and candidate_formulas:
        # High confidence in skeletons → focus beams on refinement
        n_beams = min(n_beams, len(candidate_formulas) + 2)
        n_rounds = 1 # One round is enough to check the seeds
        base_gens = min(base_gens, 150)
        base_pop = min(base_pop, 50)
        print(f"  [Adaptive] Confident proposer: reducing search to {n_beams} beams, 1 round, {base_pop} pop, {base_gens} gens.")
    elif candidate_formulas:
        # Proposer gave skeletons but isn't super confident
        # We can still reduce search from full random exploration
        n_beams = min(n_beams, 7)
        n_rounds = 1
        base_gens = min(base_gens, 250)
        base_pop = min(base_pop, 60)
        print(f"  [Adaptive] Proposer candidates available: reducing search to {n_beams} beams, 1 round, {base_pop} pop, {base_gens} gens.")

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

