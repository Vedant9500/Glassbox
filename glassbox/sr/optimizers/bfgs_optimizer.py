"""
Multi-Start BFGS Optimizer for Symbolic Regression Constants.

Problem: Least squares regression only works well when Phase 1 finds the EXACT
correct structure. If Phase 1 finds "almost right" structure (e.g., sin(x) + x
instead of sin(x) + x²), least squares can't fix it.

Solution: Use BFGS with:
1. Multiple random starts (escape local minima)
2. Regularization (L1 sparsity + L2 smoothness)
3. Iterative pruning (remove low-weight terms, refit)

This is more robust than plain least squares and can help "discover" the right
coefficients even when structure is imperfect.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import LBFGS
from typing import Optional, Dict, List, Tuple, Callable
import numpy as np
import copy


class RegularizedBFGS:
    """
    BFGS optimizer with L1/L2 regularization for coefficient fitting.
    
    Loss function:
        L = MSE(y, X @ w) + λ1 * ||w||_1 + λ2 * ||w||_2^2
    
    where:
        - MSE ensures good fit
        - L1 encourages sparsity (zero out unused terms)
        - L2 prevents coefficient explosion
    """
    
    def __init__(
        self,
        l1_weight: float = 0.01,
        l2_weight: float = 0.001,
        max_iter: int = 100,
        lr: float = 1.0,
        tolerance_grad: float = 1e-7,
        tolerance_change: float = 1e-9,
        line_search_fn: str = 'strong_wolfe',
    ):
        """
        Args:
            l1_weight: L1 regularization weight (sparsity)
            l2_weight: L2 regularization weight (smoothness)
            max_iter: Maximum BFGS iterations
            lr: Learning rate
            tolerance_grad: Gradient tolerance for convergence
            tolerance_change: Parameter change tolerance
            line_search_fn: Line search method
        """
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.max_iter = max_iter
        self.lr = lr
        self.tolerance_grad = tolerance_grad
        self.tolerance_change = tolerance_change
        self.line_search_fn = line_search_fn
    
    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        initial_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float]:
        """
        Fit coefficients using L-BFGS with regularization.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,) or (n_samples, 1)
            initial_weights: Optional initial weights (n_features,)
        
        Returns:
            (weights, final_loss)
        """
        n_features = X.shape[1]
        y = y.squeeze()
        
        # Initialize weights
        if initial_weights is not None:
            weights = initial_weights.clone().detach().requires_grad_(True)
        else:
            weights = torch.zeros(n_features, dtype=X.dtype, device=X.device, requires_grad=True)
        
        # Create optimizer
        optimizer = LBFGS(
            [weights],
            lr=self.lr,
            max_iter=self.max_iter,
            tolerance_grad=self.tolerance_grad,
            tolerance_change=self.tolerance_change,
            line_search_fn=self.line_search_fn,
        )
        
        # Optimize
        def closure():
            optimizer.zero_grad()
            
            # Predictions
            pred = X @ weights
            
            # MSE loss
            mse_loss = F.mse_loss(pred, y)
            
            # L1 regularization (sparsity)
            l1_loss = weights.abs().sum()
            
            # L2 regularization (smoothness)
            l2_loss = (weights ** 2).sum()
            
            # Total loss
            loss = mse_loss + self.l1_weight * l1_loss + self.l2_weight * l2_loss
            
            # Guard against both NaN and Inf to prevent poisoning the
            # LBFGS line search with non-finite objective values.
            if torch.isfinite(loss):
                loss.backward()
            else:
                # Return a large but finite fallback so LBFGS can
                # reject this step gracefully instead of diverging.
                loss = torch.tensor(1e8, dtype=loss.dtype, device=loss.device)
            
            return loss
        
        try:
            optimizer.step(closure)
        except Exception:
            # BFGS can fail on ill-conditioned problems; keep current weights.
            pass
        
        # Final loss
        with torch.no_grad():
            pred = X @ weights
            final_mse = F.mse_loss(pred, y).item()
        
        return weights.detach(), final_mse


class MultiStartBFGS:
    """
    Run BFGS from multiple random initializations and keep the best.
    
    This helps escape local minima, especially important when the structure
    from Phase 1 is imperfect.
    """
    
    def __init__(
        self,
        n_starts: int = 5,
        l1_weight: float = 0.01,
        l2_weight: float = 0.001,
        max_iter: int = 100,
        initialization_scale: float = 0.1,
    ):
        """
        Args:
            n_starts: Number of random initializations to try
            l1_weight: L1 regularization weight
            l2_weight: L2 regularization weight
            max_iter: Max BFGS iterations per start
            initialization_scale: Scale of random initialization noise
        """
        self.n_starts = n_starts
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.max_iter = max_iter
        self.initialization_scale = initialization_scale
        
        self.bfgs = RegularizedBFGS(
            l1_weight=l1_weight,
            l2_weight=l2_weight,
            max_iter=max_iter,
        )
    
    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, float, List[Tuple[torch.Tensor, float]]]:
        """
        Fit using multiple starts and return the best.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            verbose: Print progress
        
        Returns:
            (best_weights, best_loss, all_results)
        """
        n_features = X.shape[1]
        all_results = []
        
        # Try multiple starts
        for start_idx in range(self.n_starts):
            # Random initialization
            if start_idx == 0:
                # First start: zero initialization (least squares equivalent)
                init_weights = torch.zeros(n_features, dtype=X.dtype, device=X.device)
            else:
                # Random starts
                init_weights = torch.randn(n_features, dtype=X.dtype, device=X.device) * self.initialization_scale
            
            # Fit
            weights, loss = self.bfgs.fit(X, y, initial_weights=init_weights)
            all_results.append((weights.clone(), loss))
            
            if verbose:
                print(f"  Start {start_idx + 1}/{self.n_starts}: MSE = {loss:.6f}")
        
        # Find best
        best_weights, best_loss = min(all_results, key=lambda x: x[1])
        
        if verbose:
            print(f"  Best: MSE = {best_loss:.6f}")
        
        return best_weights, best_loss, all_results


class IterativeBFGSRefiner:
    """
    Iteratively refine coefficients by alternating BFGS and pruning.
    
    Algorithm:
    1. Fit all coefficients with BFGS
    2. Prune coefficients below threshold
    3. Refit remaining coefficients
    4. Repeat until convergence or max iterations
    
    This can effectively "discover" that certain terms should be zero,
    helping when Phase 1 structure is imperfect.
    """
    
    def __init__(
        self,
        n_iterations: int = 3,
        prune_threshold: float = 0.05,
        n_starts_per_iteration: int = 3,
        l1_weight: float = 0.01,
        l2_weight: float = 0.001,
    ):
        """
        Args:
            n_iterations: Number of prune-refit iterations
            prune_threshold: Prune weights with |w| < threshold
            n_starts_per_iteration: BFGS starts per iteration
            l1_weight: L1 regularization
            l2_weight: L2 regularization
        """
        self.n_iterations = n_iterations
        self.prune_threshold = prune_threshold
        self.n_starts_per_iteration = n_starts_per_iteration
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        
        self.multistart = MultiStartBFGS(
            n_starts=n_starts_per_iteration,
            l1_weight=l1_weight,
            l2_weight=l2_weight,
        )
    
    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        feature_names: Optional[List[str]] = None,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, float, List[int]]:
        """
        Iteratively refine coefficients.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            feature_names: Optional names for logging
            verbose: Print progress
        
        Returns:
            (final_weights, final_mse, active_indices)
        """
        n_features = X.shape[1]
        active_mask = torch.ones(n_features, dtype=torch.bool, device=X.device)
        
        best_weights = None
        best_loss = float('inf')
        
        for iteration in range(self.n_iterations):
            if verbose:
                print(f"\n--- Iteration {iteration + 1}/{self.n_iterations} ---")
                active_count = active_mask.sum().item()
                print(f"Active features: {active_count}/{n_features}")
            
            # Snapshot the mask BEFORE this iteration's pruning so we can
            # detect true convergence (no change between iterations).
            prev_active_mask = active_mask.clone()
            
            # Get active features
            X_active = X[:, active_mask]
            
            if X_active.shape[1] == 0:
                if verbose:
                    print("  No active features left, stopping.")
                break
            
            # Fit active features
            weights_active, loss, _ = self.multistart.fit(X_active, y, verbose=verbose)
            
            # Expand weights to full size
            weights_full = torch.zeros(n_features, dtype=X.dtype, device=X.device)
            weights_full[active_mask] = weights_active
            
            # Track best
            if loss < best_loss:
                best_loss = loss
                best_weights = weights_full.clone()
            
            # Prune small coefficients
            abs_weights = weights_active.abs()
            max_weight = abs_weights.max().item()
            
            if max_weight > 0:
                # Prune based on absolute value
                prune_mask_active = abs_weights < self.prune_threshold
                n_pruned = prune_mask_active.sum().item()
                
                if verbose and n_pruned > 0:
                    print(f"  Pruning {n_pruned} features with |weight| < {self.prune_threshold}")
                    if feature_names is not None:
                        active_names = [name for name, m in zip(feature_names, active_mask) if m]
                        for i, (name, w, prune) in enumerate(zip(active_names, weights_active, prune_mask_active)):
                            if prune:
                                print(f"    Pruned: {name} (weight={w.item():.4f})")
                
                # Update active mask
                # Map prune_mask_active back to full feature space
                active_indices = torch.where(active_mask)[0]
                for i, prune in enumerate(prune_mask_active):
                    if prune:
                        active_mask[active_indices[i]] = False
            
            # Stop if active set is unchanged from prior iteration
            if iteration > 0 and torch.all(active_mask == prev_active_mask):
                if verbose:
                    print("  Converged (no change in active set)")
                break
        
        # Return best result
        active_indices = torch.where(active_mask)[0].tolist()
        
        if verbose:
            print(f"\n--- Final Result ---")
            print(f"Active features: {len(active_indices)}/{n_features}")
            print(f"Final MSE: {best_loss:.6f}")
        
        return best_weights, best_loss, active_indices


# ============================================================================
# High-level API
# ============================================================================

def fit_coefficients_bfgs(
    X: torch.Tensor,
    y: torch.Tensor,
    feature_names: Optional[List[str]] = None,
    method: str = 'iterative',
    n_starts: int = 5,
    n_iterations: int = 3,
    l1_weight: float = 0.01,
    l2_weight: float = 0.001,
    prune_threshold: float = 0.05,
    verbose: bool = False,
) -> Tuple[torch.Tensor, float, str]:
    """
    Fit coefficients using BFGS (high-level API).

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        feature_names: Optional feature names for formula building
        method: 'simple', 'multistart', or 'iterative'
        n_starts: Number of random starts (for multistart/iterative)
        n_iterations: Number of pruning iterations (for iterative)
        l1_weight: L1 regularization weight
        l2_weight: L2 regularization weight
        prune_threshold: Pruning threshold for iterative method
        verbose: Print progress

    Returns:
        (weights, mse, formula_string)
    """
    if method == 'iterative':
        # Try C++ native backend first
        try:
            import sys
            from pathlib import Path as _Path
            import numpy as np
            try:
                import _core
            except ImportError:
                try:
                    cpp_dir = _Path(__file__).parent.parent / 'cpp'
                    if str(cpp_dir) not in sys.path:
                        sys.path.insert(0, str(cpp_dir))
                    import _core
                except ImportError:
                    raise ImportError("C++ core not found")

            weights_out, mse = _core.iterative_elastic_net(
                X.detach().cpu().numpy().astype(np.float64),
                y.detach().cpu().numpy().astype(np.float64),
                l1_weight, l2_weight,
                n_starts, n_iterations,
                prune_threshold, 1000
            )
            weights = torch.tensor(weights_out, dtype=X.dtype, device=X.device)
            active_indices = [i for i, w in enumerate(weights_out) if abs(w) > 0]
            if verbose:
                print(f"[C++] Iterative Elastic Net MSE: {mse:.6f}")
        except Exception as e:
            if verbose:
                print(f"  [Fallback] C++ backend failed: {e}")
            iterative = IterativeBFGSRefiner(
                n_iterations=n_iterations,
                prune_threshold=prune_threshold,
                n_starts_per_iteration=n_starts,
                l1_weight=l1_weight,
                l2_weight=l2_weight,
            )
            weights, mse, active_indices = iterative.fit(X, y, feature_names=feature_names, verbose=verbose)

    elif method == 'simple':
        # Simple BFGS from zero init
        bfgs = RegularizedBFGS(l1_weight=l1_weight, l2_weight=l2_weight)
        weights, mse = bfgs.fit(X, y)
        active_indices = list(range(X.shape[1]))

    elif method == 'multistart':
        # Multi-start BFGS
        multistart = MultiStartBFGS(
            n_starts=n_starts,
            l1_weight=l1_weight,
            l2_weight=l2_weight,
        )
        weights, mse, _ = multistart.fit(X, y, verbose=verbose)
        active_indices = list(range(X.shape[1]))

    else:
        raise ValueError(f"Unknown method: {method}")

    # Build formula string
    formula = build_formula_from_weights(weights, feature_names, threshold=prune_threshold)

    return weights, mse, formula

def build_formula_from_weights(
    weights: torch.Tensor,
    feature_names: Optional[List[str]] = None,
    threshold: float = 0.01,
    snap_constants: bool = True,
    snap_threshold: float = 0.05,
) -> str:
    """
    Build a formula string from weights.
    
    Args:
        weights: Coefficient weights (n_features,)
        feature_names: Feature names (e.g., ['x', 'x²', 'sin(x)'])
        threshold: Only include terms with |weight| > threshold
        snap_constants: If True, snap weights to known constants (π, e, etc.)
        snap_threshold: Threshold for constant snapping (default 5%)
    
    Returns:
        Formula string (e.g., "π*sin(x) + (1/π)*x²")
    """
    # Import constant snapping utilities
    try:
        from glassbox.sr.operations.meta_ops import snap_to_constant, get_constant_symbol
    except ImportError:
        snap_constants = False
    
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(len(weights))]
    
    # Build formula
    terms = []
    for i, (w, name) in enumerate(zip(weights, feature_names)):
        w_val = w.item()
        if abs(w_val) > threshold:
            # Try to snap to known constant
            if snap_constants:
                coef_str = get_constant_symbol(w_val, snap_threshold)
            else:
                coef_str = None
            
            # Format the term
            if abs(w_val - 1.0) < 0.01:
                terms.append(name)
            elif abs(w_val + 1.0) < 0.01:
                terms.append(f"-{name}")
            elif coef_str is not None and coef_str not in [str(int(round(w_val))), f"{w_val:.4g}"]:
                # Use symbolic constant (π, e, etc.)
                if w_val > 0:
                    terms.append(f"{coef_str}*{name}")
                else:
                    terms.append(f"({coef_str})*{name}")
            elif w_val > 0:
                terms.append(f"{w_val:.4f}*{name}")
            else:
                terms.append(f"({w_val:.4f}*{name})")
    
    formula = " + ".join(terms) if terms else "0"
    
    # Clean up
    formula = formula.replace("+ -", "- ")
    formula = formula.replace("+ (-", "- (")
    
    return formula


# ============================================================================
# Testing
# ============================================================================

def test_bfgs_basic():
    """Test basic BFGS fitting."""
    print("Testing RegularizedBFGS...")
    
    # Generate synthetic data: y = 2*x1 + 3*x2 + noise
    torch.manual_seed(42)
    n_samples = 100
    X = torch.randn(n_samples, 2)
    true_weights = torch.tensor([2.0, 3.0])
    y = X @ true_weights + torch.randn(n_samples) * 0.1
    
    # Fit
    bfgs = RegularizedBFGS(l1_weight=0.001, l2_weight=0.0001)
    weights, mse = bfgs.fit(X, y)
    
    print(f"True weights: {true_weights}")
    print(f"Fitted weights: {weights}")
    print(f"MSE: {mse:.6f}")
    
    # Check accuracy
    weight_error = (weights - true_weights).abs().max().item()
    assert weight_error < 0.5, f"Weight error too large: {weight_error}"
    assert mse < 0.1, f"MSE too large: {mse}"
    
    print("✓ Basic BFGS test passed\n")


def test_multistart():
    """Test multi-start BFGS."""
    print("Testing MultiStartBFGS...")
    
    # Create non-convex problem by adding noisy features
    torch.manual_seed(42)
    n_samples = 100
    X = torch.randn(n_samples, 5)  # 5 features
    true_weights = torch.tensor([2.0, 0.0, 3.0, 0.0, 0.0])  # Only 2 active
    y = X @ true_weights + torch.randn(n_samples) * 0.1
    
    # Fit with multi-start
    multistart = MultiStartBFGS(n_starts=5, l1_weight=0.05)
    weights, mse, all_results = multistart.fit(X, y, verbose=True)
    
    print(f"\nTrue weights: {true_weights}")
    print(f"Best weights: {weights}")
    print(f"Best MSE: {mse:.6f}")
    
    # Check that we found the correct sparse solution
    active_true = (true_weights != 0).float()
    active_fitted = (weights.abs() > 0.1).float()
    sparsity_match = (active_true == active_fitted).float().mean().item()
    
    print(f"Sparsity pattern match: {sparsity_match:.1%}")
    assert sparsity_match >= 0.8, "Should find correct sparsity pattern"
    
    print("✓ Multi-start BFGS test passed\n")


def test_iterative():
    """Test iterative refinement."""
    print("Testing IterativeBFGSRefiner...")
    
    # Create problem: y = π*sin(x) + x²/π
    torch.manual_seed(42)
    n_samples = 100
    x = torch.linspace(-3, 3, n_samples).unsqueeze(1)
    
    # Build features: [x, x², x³, sin(x), cos(x)]
    import math
    PI = math.pi
    X = torch.cat([
        x,
        x ** 2,
        x ** 3,
        torch.sin(x),
        torch.cos(x),
    ], dim=1)
    
    feature_names = ['x', 'x²', 'x³', 'sin(x)', 'cos(x)']
    
    # True formula: π*sin(x) + x²/π
    true_weights = torch.tensor([0.0, 1/PI, 0.0, PI, 0.0])
    y = X @ true_weights
    
    # Fit with iterative refinement
    refiner = IterativeBFGSRefiner(
        n_iterations=3,
        prune_threshold=0.1,
        n_starts_per_iteration=3,
        l1_weight=0.02,
    )
    
    weights, mse, active_indices = refiner.fit(X, y, feature_names=feature_names, verbose=True)
    
    print(f"\nTrue formula: π*sin(x) + x²/π")
    print(f"True weights: {true_weights}")
    print(f"Fitted weights: {weights}")
    print(f"Active indices: {active_indices}")
    
    formula = build_formula_from_weights(weights, feature_names, threshold=0.05)
    print(f"\nDiscovered formula: {formula}")
    
    # Check that we found the right terms
    assert weights[1].abs() > 0.1, "Should find x² term"
    assert weights[3].abs() > 1.0, "Should find sin(x) term"
    assert weights[2].abs() < 0.1, "Should prune x³ term"
    
    print("\n✓ Iterative refinement test passed\n")


if __name__ == '__main__':
    test_bfgs_basic()
    test_multistart()
    test_iterative()
    print("\n" + "="*50)
    print("ALL TESTS PASSED!")
    print("="*50)
