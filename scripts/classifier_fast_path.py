"""
Classifier-Guided Fast Path for Symbolic Regression

When the curve classifier predicts operators with high confidence,
skip Phase 1 evolution entirely and directly run regression.

This can reduce solve time from ~300s to <10s for well-predicted formulas.
"""

import re
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from pathlib import Path


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
    
    for iteration in range(max_iter):
        w_old = w.copy()
        
        for j in range(n_features):
            # Compute residual without feature j
            residual = y - X @ w + X[:, j] * w[j]
            
            # Correlation
            rho = X[:, j] @ residual
            
            # Soft thresholding
            if alpha == 0:
                w[j] = rho / X_sq[j]
            else:
                w[j] = soft_threshold(rho, alpha * n_samples) / X_sq[j]
        
        # Check convergence
        if np.max(np.abs(w - w_old)) < tol:
            break
    
    return w


def soft_threshold(x: float, threshold: float) -> float:
    """Soft thresholding operator for LASSO."""
    if x > threshold:
        return x - threshold
    elif x < -threshold:
        return x + threshold
    else:
        return 0.0


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
    x = x.flatten()
    n = len(x)
    
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

    # Polynomial terms (always include in universal mode)
    include_polynomial = allow_power and (
        universal_basis or 
        predictions.get('power', 0) >= threshold or 
        predictions.get('polynomial', 0) >= threshold
    )
    
    if include_polynomial:
        # Linear term
        basis_list.append(x)
        names.append("x")
        
        # Higher powers
        for p in range(2, max_power + 1):
            basis_list.append(x ** p)
            names.append(f"x^{p}")
    
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
        # Sin terms
        for omega in omegas[:4]:  # Top 4 frequencies
            basis_list.append(np.sin(omega * x))
            if omega == 1.0:
                names.append("sin(x)")
            elif omega == 2.0:
                names.append("sin(2*x)")
            elif omega == 0.5:
                names.append("sin(x/2)")
            else:
                names.append(f"sin({omega:.2f}*x)")
        
        # Cos terms
        for omega in omegas[:4]:
            basis_list.append(np.cos(omega * x))
            if omega == 1.0:
                names.append("cos(x)")
            elif omega == 2.0:
                names.append("cos(2*x)")
            elif omega == 0.5:
                names.append("cos(x/2)")
            else:
                names.append(f"cos({omega:.2f}*x)")
    
    # Exponential operations (only if predicted OR universal)
    include_exp = allow_exp and (
        universal_basis or
        predictions.get('exp', 0) >= threshold or 
        predictions.get('exponential', 0) >= threshold
    )
    if include_exp:
        # Clamp to prevent overflow
        x_clamp = np.clip(x, -10, 10)
        exp_x = np.exp(x_clamp)
        basis_list.append(exp_x)
        names.append("exp(x)")
        basis_list.append(np.exp(-x_clamp))
        names.append("exp(-x)")
        # Gaussian
        basis_list.append(np.exp(-x**2))
        names.append("exp(-x^2)")
        
        # Planck-style terms: x^n / (exp(x) - 1)
        denom = np.maximum(exp_x - 1.0, 1e-6)
        basis_list.append(1.0 / denom)
        names.append("1/(exp(x)-1)")
        basis_list.append(x / denom)
        names.append("x/(exp(x)-1)")
        basis_list.append((x ** 2) / denom)
        names.append("x^2/(exp(x)-1)")
        basis_list.append((x ** 3) / denom)
        names.append("x^3/(exp(x)-1)")
    
    # Logarithmic operations (always include in universal mode for Nguyen-7 etc.)
    if allow_log and (universal_basis or predictions.get('log', 0) >= threshold):
        # Avoid log(0) issues
        x_safe = np.maximum(np.abs(x), 1e-10)
        basis_list.append(np.log(x_safe + 1))
        names.append("log(x+1)")
        basis_list.append(np.log(x_safe**2 + 1))
        names.append("log(x^2+1)")
    
    # Composition terms (for sin(x²), etc. - covers Nguyen-10)
    if universal_basis and allow_periodic:
        # sin/cos of x²
        basis_list.append(np.sin(x**2))
        names.append("sin(x^2)")
        basis_list.append(np.cos(x**2))
        names.append("cos(x^2)")

    # Power/rational families should be available even if periodic is disabled
    if universal_basis and allow_power:
        # sqrt and reciprocal (positive x only parts)
        x_safe = np.maximum(np.abs(x), 1e-3)  # Increased safety margin
        basis_list.append(np.sqrt(x_safe))
        names.append("sqrt(|x|)")
        basis_list.append(1.0 / x_safe)
        names.append("1/|x|")
        
        # Relativistic / elliptic forms: 1/sqrt(1-x^2), sqrt(1-x^2)
        # These are critical for physics formulas
        x2 = x**2
        safe_denom = np.maximum(1 - x2, 1e-6)  # Avoid division by zero near |x|=1
        basis_list.append(1.0 / np.sqrt(safe_denom))
        names.append("1/sqrt(1-x^2)")
        basis_list.append(np.sqrt(safe_denom))
        names.append("sqrt(1-x^2)")
        basis_list.append(x / np.sqrt(safe_denom))
        names.append("x/sqrt(1-x^2)")
        # Also 1/(1-x^2) without sqrt
        basis_list.append(1.0 / safe_denom)
        names.append("1/(1-x^2)")
        
        # Planck/exp-denominator terms (critical for x^3 / (exp(x)-1))
        if include_exp:
            x_clamp = np.clip(x, -10, 10)
            expm1 = np.expm1(x_clamp)  # exp(x)-1, more stable near 0
            expm1 = np.where(np.abs(expm1) < 1e-6, np.sign(expm1) * 1e-6, expm1)
            basis_list.append(1.0 / expm1)
            names.append("1/(exp(x)-1)")
            basis_list.append(x / expm1)
            names.append("x/(exp(x)-1)")
            basis_list.append(x**2 / expm1)
            names.append("x^2/(exp(x)-1)")
            basis_list.append(x**3 / expm1)
            names.append("x^3/(exp(x)-1)")

        # Rational function terms (Lorentzian, Cauchy type)
        for c in [0.5, 1.0, 2.0]:
            # Quadratic denominator
            denom_q = x**2 + c
            basis_list.append(1.0 / denom_q)
            names.append(f"1/(x^2+{c})")
            basis_list.append(x / denom_q)
            names.append(f"x/(x^2+{c})")
            
            # Linear denominator (1/(x+c)) - handle singularity
            # Use softplus-like smooth denominator or just offset
            basis_list.append(1.0 / (np.abs(x) + c))
            names.append(f"1/(|x|+{c})")

        # Automatic cross-terms: periodic / rational
        # Enable when periodic + power/rational are plausible
        if allow_periodic and (
            predictions.get('rational', 0) >= threshold or
            predictions.get('power', 0) >= threshold or
            predictions.get('periodic', 0) >= threshold
        ):
            for c in [0.5, 1.0, 2.0]:
                denom_q = x**2 + c
                for omega in omegas[:4]:
                    basis_list.append(np.sin(omega * x) / denom_q)
                    names.append(f"sin({omega:.2f}*x)/(x^2+{c})")
                    basis_list.append(np.cos(omega * x) / denom_q)
                    names.append(f"cos({omega:.2f}*x)/(x^2+{c})")
        
        # x*sin(x), x*cos(x) - product terms
        basis_list.append(x * np.sin(x))
        names.append("x·sin(x)")
        basis_list.append(x * np.cos(x))
        names.append("x·cos(x)")
        
        # Damped oscillator terms: exp(-αx)*sin(ωx), exp(-αx)*cos(ωx)
        # Only include when exp and periodic are both allowed
        if include_exp and allow_periodic:
            # Use fewer decay rates but include ALL frequencies (including FFT-detected)
            decay_rates = [0.2, 0.5]  # Most common decay rates
        
            for alpha in decay_rates:
                decay = np.exp(-alpha * np.abs(x))
            
                # Use ALL omega frequencies (including FFT-detected ones like 5.01)
                for omega in omegas:
                    basis_list.append(decay * np.sin(omega * x))
                    if abs(omega - 1.0) < 0.1:
                        names.append(f"e^(-{alpha}x)·sin(x)")
                    else:
                        names.append(f"e^(-{alpha}x)·sin({omega:.2f}x)")
                
                    basis_list.append(decay * np.cos(omega * x))
                    if abs(omega - 1.0) < 0.1:
                        names.append(f"e^(-{alpha}x)·cos(x)")
                    else:
                        names.append(f"e^(-{alpha}x)·cos({omega:.2f}x)")
    
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
                except:
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
                except:
                    pass
    
    # Try pairs of basis functions (including constant)
    if max_terms >= 2:
        indices_all = list(range(n_basis))
        from itertools import combinations
        
        for (i, j) in combinations(indices_all, 2):
            X = basis[:, [i, j]]
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                y_pred = X @ coeffs
                mse = np.mean((y - y_pred) ** 2)
                if mse < tolerance:
                    terms = []
                    for idx, (k, c) in enumerate([(i, coeffs[0]), (j, coeffs[1])]):
                        if abs(c) < 1e-6:
                            continue
                        if names[k] == "1":
                            terms.append(f"{c:.4g}")
                        elif abs(c - 1.0) < 0.01:
                            terms.append(names[k])
                        elif abs(c + 1.0) < 0.01:
                            terms.append(f"-{names[k]}")
                        elif abs(c - round(c)) < 0.01 and abs(c) < 100:
                            terms.append(f"{int(round(c))}*{names[k]}")
                        else:
                            terms.append(f"{c:.4g}*{names[k]}")
                    
                    formula = " + ".join(terms).replace("+ -", "- ")
                    full_coeffs = np.zeros(n_basis)
                    full_coeffs[i] = coeffs[0]
                    full_coeffs[j] = coeffs[1]
                    return formula, mse, full_coeffs
            except:
                pass
    
    # Try triples of basis functions (including constant)
    if max_terms >= 3:
        indices_all = list(range(n_basis))
        
        for (i, j, k) in combinations(indices_all, 3):
            X = basis[:, [i, j, k]]
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                y_pred = X @ coeffs
                mse = np.mean((y - y_pred) ** 2)
                if mse < tolerance:
                    terms = []
                    indices = [i, j, k]
                    for idx, c in enumerate(coeffs):
                        if abs(c) < 1e-6:
                            continue
                        name = names[indices[idx]]
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
                    
                    formula = " + ".join(terms).replace("+ -", "- ")
                    full_coeffs = np.zeros(n_basis)
                    full_coeffs[i] = coeffs[0]
                    full_coeffs[j] = coeffs[1]
                    full_coeffs[k] = coeffs[2]
                    return formula, mse, full_coeffs
            except:
                pass
    
    return None


def fast_path_regression(
    x: np.ndarray,
    y: np.ndarray,
    predictions: Dict[str, float],
    detected_omegas: Optional[List[float]] = None,
    sparsity_threshold: float = 0.01,
    op_constraints: Optional[Dict[str, bool]] = None,
    universal_basis: bool = True,
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
    x = x.flatten()
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

    exact_match = find_exact_symbolic_match(
        basis, names, y, max_terms=exact_max_terms, tolerance=1e-5
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
        except:
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
    
    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    
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
    
    model = FrequencyModel(initial_omegas)
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
            best_omegas = model.omegas.detach().numpy().tolist()
    
    return best_omegas, best_mse


def refine_periodic_rational(
    x: np.ndarray,
    y: np.ndarray,
    omega_inits: List[float],
    c_inits: List[float],
    steps: int = 200,
    lr: float = 0.05,
) -> Optional[Dict]:
    """
    Continuous refinement for terms like sin(ωx)/(x^2+c).
    Fits omega and c (positive) with linear coefficients.
    """
    import torch
    import math
    from glassbox.sr.meta_ops import get_constant_symbol

    x_t = torch.tensor(x, dtype=torch.float64)
    y_t = torch.tensor(y, dtype=torch.float64)

    best = None
    best_mse = float('inf')

    for omega0 in omega_inits:
        for c0 in c_inits:
            # Parameters
            omega = torch.nn.Parameter(torch.tensor(float(omega0), dtype=torch.float64))
            c_unconstrained = torch.nn.Parameter(torch.tensor(math.log(math.exp(c0) - 1 + 1e-6), dtype=torch.float64))
            a = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
            b = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
            d = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
            e = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

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
            x, y, detected_omegas, n_steps=refine_steps
        )
        
        print(f"  Refined frequencies: {[f'{o:.3f}' for o in refined_omegas]}")
        print(f"  Refinement MSE: {refined_mse:.6f}")
        
        # Re-run regression with refined omegas
        if refined_mse < best_mse:
            formula2, mse2, details2 = fast_path_regression(
                x, y, predictions, detected_omegas=refined_omegas,
                op_constraints=op_constraints,
                universal_basis=best_universal,
            )
            if mse2 < best_mse:
                return formula2, mse2, details2
    
    # Additional continuous refinement for periodic×rational terms
    if op_constraints:
        allow_periodic = op_constraints.get('periodic', True)
        allow_power = op_constraints.get('power', True)
    else:
        allow_periodic = allow_power = True

    if allow_periodic and allow_power and best_mse > 1e-4:
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
) -> Optional[Dict]:
    """
    Run the complete fast-path pipeline.
    
    Returns:
        Dict with formula, mse, and timing if successful
        None if fast path not applicable
    """
    import time
    from scripts.curve_classifier_integration import predict_operators
    
    start_time = time.time()
    
    # Convert to numpy
    x_np = x.cpu().numpy().flatten() if hasattr(x, 'cpu') else x.flatten()
    y_np = y.cpu().numpy().flatten() if hasattr(y, 'cpu') else y.flatten()
    
    # Get classifier predictions
    print("\n" + "="*60)
    print("FAST PATH: Classifier-Guided Regression")
    print("="*60)
    
    predictions = predict_operators(x_np, y_np, classifier_path, threshold=0.3)
    
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
) -> Dict[str, any]:
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
            # Extract frequency if present
            freq_match = re.search(r'sin\(([0-9.]+)\*?x', name_lower)
            if freq_match:
                hints['frequencies'].append(float(freq_match.group(1)))
            elif 'sin(x)' in name_lower or 'sin(x/' in name_lower:
                hints['frequencies'].append(1.0)
                
        if 'cos(' in name_lower:
            hints['operators'].add('cos')
            hints['operators'].add('periodic')
            freq_match = re.search(r'cos\(([0-9.]+)\*?x', name_lower)
            if freq_match:
                hints['frequencies'].append(float(freq_match.group(1)))
        
        # Power operators
        if 'x^' in name_lower:
            hints['operators'].add('power')
            power_match = re.search(r'x\^([0-9.]+)', name_lower)
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
