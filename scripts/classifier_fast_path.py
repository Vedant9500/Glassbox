"""
Classifier-Guided Fast Path for Symbolic Regression

When the curve classifier predicts operators with high confidence,
skip Phase 1 evolution entirely and directly run regression.

This can reduce solve time from ~300s to <10s for well-predicted formulas.
"""

import re
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path

# Thread-safe CUDA warning state
_warned_no_cuda = False
_cuda_warning_lock = threading.Lock()

# Pre-compiled regex patterns for performance
_FREQ_SIN_PATTERN = re.compile(r'sin\(([0-9.]+)\*?x', re.IGNORECASE)
_FREQ_COS_PATTERN = re.compile(r'cos\(([0-9.]+)\*?x', re.IGNORECASE)
_POWER_PATTERN = re.compile(r'x\^([0-9.]+)', re.IGNORECASE)


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
            old_w_j = w[j]
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
    max_power: int = 4,
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
        return "x" if n_vars == 1 else f"x{i+1}"

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

            for p in range(2, max_power + 1):
                basis_list.append(xi ** p)
                names.append(f"{var_name(i)}^{p}")
    
    # Periodic operations - build comprehensive omega list
    # Always include common frequencies: 1.0, 2.0, 0.5
    omegas = [1.0, 2.0, 0.5]  # Standard frequencies
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

            for omega in omegas[:4]:  # Top 4 frequencies
                basis_list.append(np.sin(omega * xi))
                if omega == 1.0:
                    names.append(f"sin({name})")
                elif omega == 2.0:
                    names.append(f"sin(2*{name})")
                elif omega == 0.5:
                    names.append(f"sin({name}/2)")
                else:
                    names.append(f"sin({omega:.2f}*{name})")

            for omega in omegas[:4]:
                basis_list.append(np.cos(omega * xi))
                if omega == 1.0:
                    names.append(f"cos({name})")
                elif omega == 2.0:
                    names.append(f"cos(2*{name})")
                elif omega == 0.5:
                    names.append(f"cos({name}/2)")
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

            denom = np.maximum(exp_x - 1.0, 1e-6)
            basis_list.append(1.0 / denom)
            names.append(f"1/(exp({name})-1)")
            basis_list.append(xi / denom)
            names.append(f"{name}/(exp({name})-1)")
            basis_list.append((xi ** 2) / denom)
            names.append(f"{name}^2/(exp({name})-1)")
            basis_list.append((xi ** 3) / denom)
            names.append(f"{name}^3/(exp({name})-1)")
    
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

            if include_exp:
                x_clamp = np.clip(xi, -10, 10)
                expm1 = np.expm1(x_clamp)
                expm1 = np.where(np.abs(expm1) < 1e-6, np.sign(expm1) * 1e-6, expm1)
                basis_list.append(1.0 / expm1)
                names.append(f"1/(exp({name})-1)")
                basis_list.append(xi / expm1)
                names.append(f"{name}/(exp({name})-1)")
                basis_list.append(xi**2 / expm1)
                names.append(f"{name}^2/(exp({name})-1)")
                basis_list.append(xi**3 / expm1)
                names.append(f"{name}^3/(exp({name})-1)")

    # Pairwise interaction terms for multi-input formulas
    if universal_basis and allow_arithmetic and n_vars > 1:
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                basis_list.append(x[:, i] * x[:, j])
                names.append(f"{var_name(i)}*{var_name(j)}")

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
                        # Build formula
                        const_part = f"{coeffs[0]:.4g}" if abs(coeffs[0]) > 1e-6 else ""
                        if abs(coeffs[1] - 1.0) < 0.01:
                            term_part = names[i]
                        elif abs(coeffs[1] + 1.0) < 0.01:
                            term_part = f"-{names[i]}"
                        else:
                            term_part = f"{coeffs[1]:.4g}*{names[i]}"
                        
                        if const_part and term_part:
                            formula = f"{const_part} + {term_part}" if coeffs[1] > 0 else f"{const_part} - {term_part[1:]}" if term_part.startswith('-') else f"{const_part} + {term_part}"
                        else:
                            formula = term_part or const_part
                        formula = formula.replace("+ -", "- ")
                        
                        full_coeffs = np.zeros(n_basis)
                        full_coeffs[0] = coeffs[0] if include_const else 0  # Assuming index 0 is constant
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
                        if abs(coeff[0] - 1.0) < 0.01:
                            formula = names[i]
                        elif abs(coeff[0] + 1.0) < 0.01:
                            formula = f"-{names[i]}"
                        else:
                            formula = f"{coeff[0]:.4g}*{names[i]}"
                        
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
                terms.append(f"{c:.4g}")
            elif abs(c - 1.0) < 0.01:
                terms.append(name)
            elif abs(c + 1.0) < 0.01:
                terms.append(f"-{name}")
            elif abs(c - round(c)) < 0.01 and abs(c) < 100:
                terms.append(f"{int(round(c))}*{name}")
            else:
                terms.append(f"{c:.4g}*{name}")
            full_coeffs[idx] = c

        formula = " + ".join(terms).replace("+ -", "- ")
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
) -> Tuple[str, float, Dict]:
    """
    Directly solve for coefficients using least squares regression.
    
    IMPROVED: First searches for exact symbolic matches before LASSO.
    
    Args:
        x: Input values (N,)
        y: Target values (N,)
        predictions: Classifier predictions
        detected_omegas: FFT-detected frequencies
        sparsity_threshold: Coefficients below this are zeroed
        
    Returns:
        formula: String representation
        mse: Mean squared error
        details: Dict with coefficients and basis names
    """
    if x.ndim == 1:
        x = x.flatten()
    elif x.ndim == 2:
        pass
    else:
        raise ValueError(f"Expected x to be 1D or 2D, got shape {x.shape}")
    y = y.flatten()
    
    # Build basis
    basis, names = build_basis_from_predictions(
        x, predictions, 
        threshold=0.3,  # Lower threshold to include more options
        detected_omegas=detected_omegas,
        op_constraints=op_constraints,
        universal_basis=universal_basis,
    )
    
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

    # If only power is allowed, enable 4-term exact match for polynomials
    exact_max_terms = 4 if (allow_power and not allow_periodic and not allow_exp and not allow_log) else 3

    if exact_match_enabled and (exact_match_max_basis is None or basis.shape[1] <= exact_match_max_basis):
        exact_match = find_exact_symbolic_match(
            basis,
            names,
            y,
            max_terms=exact_max_terms,
            tolerance=1e-5,
            num_threads=exact_match_threads,
        )
        if exact_match:
            formula, mse, coeffs = exact_match
            print(f"  Found EXACT symbolic match: {formula} (MSE={mse:.2e})")
            return formula, mse, {
                'coefficients': coeffs,
                'basis_names': names,
                'n_nonzero': sum(1 for c in coeffs if abs(c) >= sparsity_threshold),
                'exact_match': True,
            }
    elif exact_match_enabled:
        print(f"  Skipping exact-match search (basis={basis.shape[1]} > {exact_match_max_basis})")
    
    # Normalize basis for numerical stability
    basis_std = np.std(basis, axis=0, keepdims=True)
    basis_std[basis_std < 1e-10] = 1.0
    basis_norm = basis / basis_std
    
    # Try LASSO with adaptive alpha (coordinate descent)
    best_coeffs = None
    best_mse = float('inf')
    best_n_terms = float('inf')
    best_score = float('inf')  # Complexity-penalized score
    
    # Complexity penalty: prefer simpler solutions
    COMPLEXITY_PENALTY = 0.001  # λ in: score = MSE + λ * n_terms
    
    # Try multiple alpha values to find best sparsity-accuracy tradeoff
    for alpha in [0.0, 0.001, 0.01, 0.05, 0.1, 0.2]:
        try:
            coeffs = lasso_coordinate_descent(basis_norm, y, alpha=alpha, max_iter=1000)
            
            # Check for NaN/Inf in coeffs
            if not np.all(np.isfinite(coeffs)):
                print(f"  Warning: Alpha={alpha} produced non-finite coefficients")
                continue
            
            # Unnormalize coefficients
            coeffs = coeffs / basis_std.flatten()
            
            # Compute MSE
            y_pred = basis @ coeffs
            mse = np.mean((y - y_pred) ** 2)
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
                best_n_terms = n_terms
                best_score = score
        except Exception as e:
            print(f"  Error with alpha={alpha}: {e}")
            continue
    
    if best_coeffs is None:
        # Fallback to plain least squares
        try:
            best_coeffs, _, _, _ = np.linalg.lstsq(basis, y, rcond=None)
            y_pred = basis @ best_coeffs
            best_mse = np.mean((y - y_pred) ** 2)
        except np.linalg.LinAlgError:
            return None, float('inf'), {}
    
    # IMPORTANT: Refit with OLS on selected terms only (LASSO shrinks coefficients)
    # This recovers exact coefficients while keeping the sparse structure
    selected_mask = np.abs(best_coeffs) > sparsity_threshold
    if selected_mask.sum() > 0 and selected_mask.sum() < len(best_coeffs):
        basis_selected = basis[:, selected_mask]
        try:
            refit_coeffs, _, _, _ = np.linalg.lstsq(basis_selected, y, rcond=None)
            y_pred = basis_selected @ refit_coeffs
            refit_mse = np.mean((y - y_pred) ** 2)
            
            # Use refit if it's better
            if refit_mse <= best_mse + 0.001:  # Allow small tolerance
                best_coeffs = np.zeros_like(best_coeffs)
                best_coeffs[selected_mask] = refit_coeffs
                best_mse = refit_mse
        except (np.linalg.LinAlgError, ValueError):
            pass  # Keep LASSO solution
    
    coeffs = best_coeffs
    mse = best_mse
    
    # Build formula string (sparse)
    from glassbox.sr.meta_ops import get_constant_symbol
    terms = []
    for i, (name, coef) in enumerate(zip(names, coeffs)):
        if abs(coef) < sparsity_threshold:
            continue
        
        # Format coefficient nicely
        if name == "1":
            terms.append(get_constant_symbol(coef, threshold=0.05))
        elif abs(coef - 1.0) < 0.01:
            terms.append(name)
        elif abs(coef + 1.0) < 0.01:
            terms.append(f"-{name}")
        elif abs(coef - round(coef)) < 0.01:
            terms.append(f"{int(round(coef))}*{name}")
        else:
            coef_sym = get_constant_symbol(coef, threshold=0.05)
            terms.append(f"{coef_sym}*{name}")
    
    formula = " + ".join(terms) if terms else "0"
    formula = formula.replace("+ -", "- ")
    
    n_nonzero = sum(1 for c in coeffs if abs(c) >= sparsity_threshold)
    exact_match_flag = mse < 1e-6 and n_nonzero <= 4

    return formula, mse, {
        'coefficients': coeffs,
        'basis_names': names,
        'n_nonzero': n_nonzero,
        'exact_match': exact_match_flag,
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
    from glassbox.sr.meta_ops import get_constant_symbol

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

            mse = float(loss.item())
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

                formula = " + ".join(terms) if terms else "0"
                formula = formula.replace("+ -", "- ")

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
) -> Tuple[str, float, Dict]:
    """
    Fast-path with optional frequency refinement.
    
    1. Run initial fast-path regression
    2. If MSE is moderate (0.01-0.5), try frequency refinement
    3. Re-run regression with refined frequencies
    """
    # Stage 1: Minimal basis (predicted ops only)
    stage_order = [False, True] if auto_expand else [True]
    best_formula = None
    best_mse = float('inf')
    best_details: Dict = {}
    best_universal = True

    for use_universal in stage_order:
        formula, mse, details = fast_path_regression(
            x, y, predictions, detected_omegas,
            op_constraints=op_constraints,
            universal_basis=use_universal,
            exact_match_threads=exact_match_threads,
            exact_match_enabled=exact_match_enabled,
            exact_match_max_basis=exact_match_max_basis,
        )
        if mse < best_mse:
            best_mse = mse
            best_formula = formula
            best_details = details
            best_universal = use_universal

        # If exact match or good enough, return immediately
        if details.get('exact_match', False) or mse < 1e-6:
            return formula, mse, details

    # If good enough and exact, return best
    if best_mse < 0.01 and best_details.get('exact_match', False) and best_formula is not None:
        return best_formula, best_mse, best_details
    
    # If moderate MSE and periodic terms were used, try refinement
    if best_mse < 0.5 and detected_omegas:
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
            )
            if mse2 < best_mse:
                return formula2, mse2, details2
    
    # Additional continuous refinement for periodic×rational terms
    if op_constraints:
        allow_periodic = op_constraints.get('periodic', True)
        allow_power = op_constraints.get('power', True)
    else:
        allow_periodic = allow_power = True

    if x.ndim == 1 and allow_periodic and allow_power and best_mse > 1e-4:
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
            if refined['mse'] < best_mse - 1e-4:
                return refined['formula'], refined['mse'], refined['details']

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
    classifier_path: str = "models/curve_classifier.pt",
    detected_omegas: Optional[List[float]] = None,
    op_constraints: Optional[Dict[str, bool]] = None,
    auto_expand: bool = True,
    device: Optional[str] = None,
    exact_match_threads: int = 1,
    exact_match_enabled: bool = True,
    exact_match_max_basis: int = 150,
) -> Optional[Dict]:
    """
    Run the complete fast-path pipeline.
    
    Returns:
        Dict with formula, mse, and timing if successful
        None if fast path not applicable
    """
    import time
    from curve_classifier_integration import predict_operators
    
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
    
    predictions = predict_operators(
        x_np,
        y_np,
        classifier_path,
        threshold=0.3,
        device=device,
    )
    
    if not predictions:
        print("  No operators predicted - falling back to evolution")
        return None
    
    print(f"  Predictions: {[(k, f'{v:.2f}') for k, v in sorted(predictions.items(), key=lambda x: -x[1])]}")
    
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
    )
    
    elapsed = time.time() - start_time
    
    print(f"\n  Formula: {formula}")
    print(f"  MSE: {mse:.6f}")
    print(f"  Non-zero terms: {details.get('n_nonzero', 0)}")
    print(f"  Time: {elapsed:.2f}s")
    print("="*60)
    
    # Extract operator hints from the formula for guided evolution
    operator_hints = extract_operator_hints(formula, details.get('basis_names', []), 
                                            details.get('coefficients', []))
    
    return {
        'formula': formula,
        'mse': mse,
        'time': elapsed,
        'details': details,
        'predictions': predictions,
        'operator_hints': operator_hints,
    }


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
    from glassbox.sr.operation_dag import OperationDAG
    
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


def run_guided_evolution(
    x: torch.Tensor,
    y: torch.Tensor,
    operator_hints: Dict,
    generations: int = 20,
    population_size: int = 20,
    device: str = 'cpu',
    visualizer = None,
) -> Dict:
    """
    Run evolution guided by fast-path operator hints.
    
    Creates a constrained search space based on the operators found
    in the fast-path approximation, then evolves to find exact form.
    
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
    from glassbox.sr.evolution import EvolutionaryONNTrainer, finalize_model_coefficients
    
    print("\n" + "="*60)
    print("GUIDED EVOLUTION: Operator-Constrained Search")
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
        mutation_rate=0.4,  # Lower mutation since we're already biased
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
    
    # Initialize and evolve using trainer.train
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
