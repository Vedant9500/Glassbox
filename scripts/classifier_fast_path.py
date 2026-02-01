"""
Classifier-Guided Fast Path for Symbolic Regression

When the curve classifier predicts operators with high confidence,
skip Phase 1 evolution entirely and directly run regression.

This can reduce solve time from ~300s to <10s for well-predicted formulas.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from pathlib import Path


def build_basis_from_predictions(
    x: np.ndarray,
    predictions: Dict[str, float],
    threshold: float = 0.5,
    max_power: int = 3,
    detected_omegas: Optional[List[float]] = None,
    universal_basis: bool = True,  # NEW: Always include common terms
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
    
    # Polynomial terms (always include in universal mode)
    include_polynomial = (
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
    include_periodic = (
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
    
    # Exponential operations (only if predicted)
    if predictions.get('exp', 0) >= threshold or predictions.get('exponential', 0) >= threshold:
        # Clamp to prevent overflow
        x_clamp = np.clip(x, -10, 10)
        basis_list.append(np.exp(x_clamp))
        names.append("exp(x)")
        basis_list.append(np.exp(-x_clamp))
        names.append("exp(-x)")
    
    # Logarithmic operations (only if predicted)
    if predictions.get('log', 0) >= threshold:
        basis_list.append(np.log(np.abs(x) + 1))
        names.append("log(|x|+1)")
    
    basis = np.column_stack(basis_list)
    return basis, names


def fast_path_regression(
    x: np.ndarray,
    y: np.ndarray,
    predictions: Dict[str, float],
    detected_omegas: Optional[List[float]] = None,
    sparsity_threshold: float = 0.01,
) -> Tuple[str, float, Dict]:
    """
    Directly solve for coefficients using least squares regression.
    
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
        detected_omegas=detected_omegas
    )
    
    print(f"  Fast-path basis: {len(names)} terms")
    print(f"  Terms: {names[:10]}{'...' if len(names) > 10 else ''}")
    
    # Solve least squares: basis @ coeffs = y
    try:
        coeffs, residuals, rank, s = np.linalg.lstsq(basis, y, rcond=None)
    except np.linalg.LinAlgError:
        return None, float('inf'), {}
    
    # Compute MSE
    y_pred = basis @ coeffs
    mse = np.mean((y - y_pred) ** 2)
    
    # Build formula string (sparse)
    terms = []
    for i, (name, coef) in enumerate(zip(names, coeffs)):
        if abs(coef) < sparsity_threshold:
            continue
        
        # Format coefficient nicely
        if name == "1":
            terms.append(f"{coef:.4g}")
        elif abs(coef - 1.0) < 0.01:
            terms.append(name)
        elif abs(coef + 1.0) < 0.01:
            terms.append(f"-{name}")
        elif abs(coef - round(coef)) < 0.01:
            terms.append(f"{int(round(coef))}*{name}")
        else:
            terms.append(f"{coef:.4g}*{name}")
    
    formula = " + ".join(terms) if terms else "0"
    formula = formula.replace("+ -", "- ")
    
    return formula, mse, {
        'coefficients': coeffs,
        'basis_names': names,
        'n_nonzero': sum(1 for c in coeffs if abs(c) >= sparsity_threshold),
    }


def should_use_fast_path(
    predictions: Dict[str, float],
    confidence_threshold: float = 0.8,
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
    
    # Check if fast path is applicable
    if not should_use_fast_path(predictions, confidence_threshold=0.8):
        print("  Classifier confidence too low - falling back to evolution")
        return None
    
    # Run fast path regression
    formula, mse, details = fast_path_regression(
        x_np, y_np, predictions, 
        detected_omegas=detected_omegas
    )
    
    elapsed = time.time() - start_time
    
    print(f"\n  Formula: {formula}")
    print(f"  MSE: {mse:.6f}")
    print(f"  Non-zero terms: {details.get('n_nonzero', 0)}")
    print(f"  Time: {elapsed:.2f}s")
    print("="*60)
    
    return {
        'formula': formula,
        'mse': mse,
        'time': elapsed,
        'details': details,
        'predictions': predictions,
    }
