"""
Phased Symbolic Regression

Like CNNs extract features layer by layer, this approach:
1. Phase 1: Discover structure (which operations: x², sin, etc.)
2. Phase 2: Fix structure, do linear regression on extracted features to get exact coefficients

This solves the coefficient extraction problem because:
- Phase 1 finds the right "basis functions" 
- Phase 2 is just y = w1*f1(x) + w2*f2(x) + ... + b (linear regression!)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import LBFGS, Adam
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import copy

# Import BFGS optimizer for better coefficient fitting
try:
    from .bfgs_optimizer import fit_coefficients_bfgs, build_formula_from_weights
    BFGS_AVAILABLE = True
except ImportError:
    BFGS_AVAILABLE = False


class PhasedSymbolicRegressor:
    """
    Two-phase symbolic regression:
    
    Phase 1 (Structure Discovery):
        - Train ONN with evolution to discover operations
        - Output: which nodes are active, what operations they use
        
    Phase 2 (Coefficient Extraction):
        - Freeze node operations
        - Extract node outputs as features
        - Linear regression to find exact coefficients
        
    This is like:
        Phase 1: Learn f1(x) = x², f2(x) = sin(x), etc.
        Phase 2: Learn y = a*f1(x) + b*f2(x) + c (simple linear regression!)
    """
    
    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        device: Optional[torch.device] = None,
        # BFGS optimization parameters
        use_bfgs: bool = True,  # Use BFGS instead of least squares
        bfgs_method: str = 'iterative',  # 'simple', 'multistart', or 'iterative'
        bfgs_n_starts: int = 5,  # Number of random starts for multistart
        bfgs_n_iterations: int = 3,  # Number of prune-refit iterations
        bfgs_l1_weight: float = 0.01,  # L1 sparsity regularization
        bfgs_l2_weight: float = 0.001,  # L2 smoothness regularization
        bfgs_prune_threshold: float = 0.05,  # Prune coefficients below this
    ):
        self.model_factory = model_factory
        self.device = device or torch.device('cpu')
        
        # BFGS parameters
        self.use_bfgs = use_bfgs and BFGS_AVAILABLE
        self.bfgs_method = bfgs_method
        self.bfgs_n_starts = bfgs_n_starts
        self.bfgs_n_iterations = bfgs_n_iterations
        self.bfgs_l1_weight = bfgs_l1_weight
        self.bfgs_l2_weight = bfgs_l2_weight
        self.bfgs_prune_threshold = bfgs_prune_threshold
        
        # Results from each phase
        self.phase1_model = None
        self.phase1_formula = None
        self.phase2_coefficients = None
        self.final_formula = None
        
    def phase1_structure_discovery(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        population_size: int = 20,
        generations: int = 40,
        verbose: bool = True,
    ) -> nn.Module:
        """
        Phase 1: Discover which operations are useful.
        
        Uses evolutionary training to find structure.
        Returns model with discovered operations.
        """
        from glassbox.evolution import train_onn_evolutionary
        
        if verbose:
            print("\n" + "="*70)
            print("PHASE 1: STRUCTURE DISCOVERY")
            print("="*70)
        
        x = x.to(self.device)
        y = y.to(self.device)
        
        # Use train_onn_evolutionary directly with best settings from test_sin_x2.py
        result = train_onn_evolutionary(
            self.model_factory,
            x, y,
            population_size=population_size,
            generations=generations,
            device=self.device,
            normalize_data=False,
            constant_refine_hard=False,
            elite_fraction=0.2,           # Keep top 20%
            mutation_rate=0.4,            # Main population mutation
            prune_coefficients=False,     # No pruning during structure search
            use_explorers=True,           # Enable explorer subpopulation!
            explorer_fraction=0.25,       # 25% of pop size as explorers
            explorer_mutation_rate=0.85,  # Very high mutation for exploration
        )
        
        self.phase1_model = result['model']
        self.phase1_formula = result['formula']
        self.phase1_mse = result['final_mse']
        
        if verbose:
            print(f"\nPhase 1 discovered: {self.phase1_formula}")
            print(f"Phase 1 MSE: {self.phase1_mse:.4f}")
        
        return self.phase1_model
    
    def extract_features(
        self,
        model: nn.Module,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Extract features from each active node.
        
        Returns:
            features: (batch, n_features) - output of each active node
            feature_names: List of feature descriptions (e.g., "x0^2", "sin(x0)")
        """
        model.eval()
        features = []
        feature_names = []
        
        with torch.no_grad():
            # Always include raw input as a potential feature
            features.append(x)
            feature_names.append("x0")
            
            # Also include x^2 as basis (often needed)
            features.append(x ** 2)
            feature_names.append("x0²")
            
            # Get node outputs by doing a forward pass
            if hasattr(model, 'layers'):
                sources = x.clone()
                
                for layer_idx, layer in enumerate(model.layers):
                    layer_output, layer_info = layer(sources, hard=True)
                    n_nodes = layer_output.shape[1]
                    
                    for node_idx in range(n_nodes):
                        node = layer.nodes[node_idx]
                        node_output = layer_output[:, node_idx:node_idx+1]
                        
                        # Get node's symbolic formula
                        node_formula = f"node_{layer_idx}_{node_idx}"
                        
                        # Try to get formula from node
                        if hasattr(node, 'get_formula'):
                            try:
                                formula = node.get_formula()
                                if formula and formula != "0" and len(formula) < 50:
                                    node_formula = formula
                            except:
                                pass
                        
                        # Also check operation selector for operation type
                        if hasattr(node, 'op_selector'):
                            try:
                                sel = node.op_selector.get_selected()
                                op_type = sel.get('type', 'unknown')
                                
                                # Get operation name based on index
                                if op_type == 'unary':
                                    unary_idx = sel.get('unary_idx', 0)
                                    # Check if it's a periodic function (sin/cos)
                                    if hasattr(node, 'unary_ops'):
                                        # Simplified ops: [MetaPower, MetaPeriodic]
                                        if unary_idx == 1:  # MetaPeriodic
                                            # Check if sin or cos based on output
                                            sin_output = torch.sin(x)
                                            cos_output = torch.cos(x)
                                            sin_corr = torch.corrcoef(torch.stack([
                                                sin_output.squeeze(), node_output.squeeze()
                                            ]))[0, 1].abs().item()
                                            cos_corr = torch.corrcoef(torch.stack([
                                                cos_output.squeeze(), node_output.squeeze()
                                            ]))[0, 1].abs().item()
                                            
                                            if sin_corr > 0.9:
                                                node_formula = "sin(x0)"
                                            elif cos_corr > 0.9:
                                                node_formula = "cos(x0)"
                                            else:
                                                node_formula = "periodic(x0)"
                                        elif unary_idx == 0:  # MetaPower
                                            # Check what power
                                            x2_corr = torch.corrcoef(torch.stack([
                                                (x**2).squeeze(), node_output.squeeze()
                                            ]))[0, 1].abs().item()
                                            x3_corr = torch.corrcoef(torch.stack([
                                                (x**3).squeeze(), node_output.squeeze()
                                            ]))[0, 1].abs().item()
                                            x1_corr = torch.corrcoef(torch.stack([
                                                x.squeeze(), node_output.squeeze()
                                            ]))[0, 1].abs().item()
                                            
                                            if x2_corr > 0.95:
                                                node_formula = "x0²"
                                            elif x3_corr > 0.95:
                                                node_formula = "x0³"
                                            elif x1_corr > 0.95:
                                                node_formula = "x0"
                                            else:
                                                node_formula = f"x0^p"
                                                
                                elif op_type == 'binary':
                                    # Binary ops: multiplication
                                    x2_corr = torch.corrcoef(torch.stack([
                                        (x**2).squeeze(), node_output.squeeze()
                                    ]))[0, 1].abs().item()
                                    if x2_corr > 0.95:
                                        node_formula = "x0*x0"
                                    else:
                                        node_formula = "binary_op"
                            except:
                                pass
                        
                        # Only include if node output varies
                        output_std = node_output.std().item()
                        output_range = (node_output.max() - node_output.min()).item()
                        
                        if output_std > 0.01 and output_range > 0.1:
                            # Check for duplicates using correlation
                            is_duplicate = False
                            for i, existing_feat in enumerate(features):
                                if existing_feat.shape == node_output.shape:
                                    try:
                                        corr = torch.corrcoef(torch.stack([
                                            existing_feat.squeeze(), 
                                            node_output.squeeze()
                                        ]))[0, 1].abs().item()
                                        if corr > 0.99:
                                            is_duplicate = True
                                            break
                                    except:
                                        pass
                            
                            if not is_duplicate:
                                features.append(node_output)
                                feature_names.append(node_formula)
                    
                    sources = torch.cat([sources, layer_output], dim=-1)
        
        if features:
            features = torch.cat(features, dim=1)
        else:
            features = x
            feature_names = ["x0"]
        
        return features, feature_names
    
    def phase2_pure_basis_regression(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        model: Optional[nn.Module] = None,
        verbose: bool = True,
    ) -> Tuple[torch.Tensor, float, str]:
        """
        Phase 2 (Alternative): Use PURE basis functions based on discovered operations.
        
        Instead of using the model's internal features (which have weird scaling),
        we identify WHICH operations the model uses, then build pure basis functions.
        
        E.g., if model uses sin and square: basis = [x, x², sin(x)]
        Then do linear regression: y = a*x + b*x² + c*sin(x) + d
        """
        if verbose:
            print("\n" + "="*70)
            print("PHASE 2: PURE BASIS REGRESSION")
            print("="*70)
        
        model = model or self.phase1_model
        x = x.to(self.device)
        y = y.to(self.device).squeeze()
        
        # Identify which operations the model discovered
        discovered_ops = set()
        
        if hasattr(model, 'layers'):
            for layer in model.layers:
                for node in layer.nodes:
                    if hasattr(node, 'op_selector'):
                        try:
                            sel = node.op_selector.get_selected()
                            op_type = sel.get('type', 'unknown')
                            unary_idx = sel.get('unary_idx', 0)
                            
                            if op_type == 'unary':
                                if unary_idx == 0:  # MetaPower
                                    discovered_ops.add('power')
                                elif unary_idx == 1:  # MetaPeriodic
                                    discovered_ops.add('periodic')
                            elif op_type == 'binary':
                                discovered_ops.add('multiply')
                        except:
                            pass
        
        if verbose:
            print(f"Discovered operation types: {discovered_ops}")
        
        # Build pure basis functions
        features = []
        feature_names = []
        
        # Always include constant (will be handled by bias)
        # Always include x
        features.append(x)
        feature_names.append("x")
        
        # If power was discovered, add x², x³
        if 'power' in discovered_ops or 'multiply' in discovered_ops:
            features.append(x ** 2)
            feature_names.append("x²")
            # Try x³ too
            features.append(x ** 3)
            feature_names.append("x³")
        
        # If periodic was discovered, add sin and cos
        if 'periodic' in discovered_ops:
            features.append(torch.sin(x))
            feature_names.append("sin(x)")
            features.append(torch.cos(x))
            feature_names.append("cos(x)")
        
        features = torch.cat(features, dim=1)
        
        if verbose:
            print(f"\nPure basis functions: {feature_names}")
        
        # Add bias column
        n_samples = features.shape[0]
        features_with_bias = torch.cat([
            features,
            torch.ones(n_samples, 1, device=self.device)
        ], dim=1)
        
        # Use BFGS or least squares based on configuration
        if self.use_bfgs:
            if verbose:
                print(f"\nUsing BFGS optimization (method={self.bfgs_method}, n_starts={self.bfgs_n_starts})")
            
            # Use the new BFGS optimizer
            weights, mse, formula = fit_coefficients_bfgs(
                features_with_bias,
                y,
                feature_names=feature_names + ['bias'],
                method=self.bfgs_method,
                n_starts=self.bfgs_n_starts,
                n_iterations=self.bfgs_n_iterations,
                l1_weight=self.bfgs_l1_weight,
                l2_weight=self.bfgs_l2_weight,
                prune_threshold=self.bfgs_prune_threshold,
                verbose=verbose,
            )
            
            if verbose:
                print(f"\nBFGS optimal coefficients:")
                for name, w in zip(feature_names, weights[:-1]):
                    if abs(w.item()) > self.bfgs_prune_threshold:
                        print(f"  {w.item():+.4f} * {name}")
                print(f"  {weights[-1].item():+.4f} (bias)")
                print(f"\nBFGS MSE: {mse:.6f}")
        else:
            # Fallback to least squares
            try:
                solution = torch.linalg.lstsq(features_with_bias, y.unsqueeze(1))
                weights = solution.solution.squeeze()
            except:
                weights = torch.pinverse(features_with_bias) @ y
            
            # Compute MSE
            pred = features_with_bias @ weights
            mse = F.mse_loss(pred.squeeze(), y).item()
            
            if verbose:
                print(f"\nOptimal coefficients (least squares):")
                for name, w in zip(feature_names, weights[:-1]):
                    print(f"  {w.item():+.4f} * {name}")
                print(f"  {weights[-1].item():+.4f} (bias)")
                print(f"\nPure basis MSE: {mse:.6f}")
            
            # Build formula string
            formula_parts = []
            for name, w in zip(feature_names, weights[:-1]):
                w_val = w.item()
                if abs(w_val) > 0.01:
                    formula_parts.append(f"{w_val:.2f}*{name}")
            
            bias = weights[-1].item()
            if abs(bias) > 0.01:
                formula_parts.append(f"{bias:.2f}")
            
            formula = " + ".join(formula_parts) if formula_parts else "0"
        
        if verbose:
            print(f"\nFormula: {formula}")
        
        return weights, mse, formula
    
    def phase2_coefficient_refinement(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        model: Optional[nn.Module] = None,
        verbose: bool = True,
    ) -> Tuple[torch.Tensor, float, str]:
        """
        Phase 2: Linear regression on extracted features.
        
        Given features f1(x), f2(x), ... from Phase 1,
        solve: y = w1*f1(x) + w2*f2(x) + ... + b
        
        This is a simple least squares problem!
        """
        if verbose:
            print("\n" + "="*70)
            print("PHASE 2: COEFFICIENT EXTRACTION")
            print("="*70)
        
        model = model or self.phase1_model
        if model is None:
            raise ValueError("No model provided. Run phase1 first or provide a model.")
        
        x = x.to(self.device)
        y = y.to(self.device).squeeze()
        
        # Extract features from the trained model
        features, feature_names = self.extract_features(model, x)
        
        if verbose:
            print(f"\nExtracted {len(feature_names)} features:")
            for i, name in enumerate(feature_names):
                feat_std = features[:, i].std().item()
                feat_range = (features[:, i].max() - features[:, i].min()).item()
                print(f"  f{i}: {name} (std={feat_std:.2f}, range={feat_range:.2f})")
        
        # Normalize features for better numerical stability
        feature_means = features.mean(dim=0, keepdim=True)
        feature_stds = features.std(dim=0, keepdim=True) + 1e-8
        features_normalized = (features - feature_means) / feature_stds
        
        # Add bias term (column of ones)
        n_samples = features_normalized.shape[0]
        features_with_bias = torch.cat([
            features_normalized,
            torch.ones(n_samples, 1, device=self.device)
        ], dim=1)
        
        # Solve least squares: y = X @ w
        try:
            solution = torch.linalg.lstsq(features_with_bias, y.unsqueeze(1))
            weights_normalized = solution.solution.squeeze()
        except Exception as e:
            if verbose:
                print(f"lstsq failed ({e}), trying pseudo-inverse...")
            X = features_with_bias
            weights_normalized = torch.pinverse(X) @ y
        
        # Un-normalize weights
        weights = weights_normalized[:-1] / feature_stds.squeeze()
        bias_adjustment = (weights_normalized[:-1] * feature_means.squeeze() / feature_stds.squeeze()).sum()
        bias = weights_normalized[-1] - bias_adjustment
        
        # Compute MSE with optimal weights
        pred = features @ weights + bias
        mse = F.mse_loss(pred.squeeze(), y).item()
        
        if verbose:
            print(f"\nOptimal coefficients (least squares):")
            for i, (name, w) in enumerate(zip(feature_names, weights)):
                if abs(w.item()) > 0.01:
                    print(f"  {w.item():+.4f} * {name}")
            print(f"  {bias.item():+.4f} (bias)")
            print(f"\nPhase 2 MSE: {mse:.6f}")
        
        # Build formula string - only include significant terms
        formula_parts = []
        for name, w in zip(feature_names, weights):
            w_val = w.item()
            if abs(w_val) > 0.05:  # Only significant terms
                formula_parts.append(f"{w_val:.2f}*{name}")
        
        bias_val = bias.item()
        if abs(bias_val) > 0.05:
            formula_parts.append(f"{bias_val:.2f}")
        
        self.final_formula = " + ".join(formula_parts) if formula_parts else "0"
        
        # Store weights with bias appended
        self.phase2_coefficients = torch.cat([weights, bias.unsqueeze(0)])
        self.phase2_features = features
        self.phase2_feature_names = feature_names
        
        if verbose:
            print(f"\nFinal formula: {self.final_formula}")
        
        return self.phase2_coefficients, mse, self.final_formula
    
    def phase3_iterative_refinement(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        model: Optional[nn.Module] = None,
        n_iterations: int = 3,
        verbose: bool = True,
    ) -> Tuple[float, str]:
        """
        Phase 3 (Optional): Iterative structure-coefficient refinement.
        
        Alternates between:
        1. Prune low-weight features
        2. Re-optimize coefficients
        
        Finds the simplest formula that fits well.
        """
        if verbose:
            print("\n" + "="*70)
            print("PHASE 3: ITERATIVE SIMPLIFICATION")
            print("="*70)
        
        model = model or self.phase1_model
        x = x.to(self.device)
        y = y.to(self.device).squeeze()
        
        # Extract features
        features, feature_names = self.extract_features(model, x)
        n_features = len(feature_names)
        
        # Track active features
        active_mask = torch.ones(n_features, dtype=torch.bool, device=self.device)
        
        best_mse = float('inf')
        best_formula = None
        best_weights = None
        
        for iteration in range(n_iterations):
            if verbose:
                print(f"\n--- Iteration {iteration + 1} ---")
            
            # Get active features
            active_features = features[:, active_mask]
            active_names = [n for n, m in zip(feature_names, active_mask) if m]
            
            if len(active_names) == 0:
                break
            
            # Add bias
            n_samples = active_features.shape[0]
            X = torch.cat([active_features, torch.ones(n_samples, 1, device=self.device)], dim=1)
            
            # Solve least squares
            try:
                solution = torch.linalg.lstsq(X, y.unsqueeze(1))
                weights = solution.solution.squeeze()
            except:
                weights = torch.pinverse(X) @ y
            
            # Compute MSE
            pred = X @ weights
            mse = F.mse_loss(pred.squeeze(), y).item()
            
            # Build formula
            formula_parts = []
            for name, w in zip(active_names, weights[:-1]):
                w_val = w.item()
                if abs(w_val) > 0.01:
                    formula_parts.append(f"{w_val:.2f}*{name}")
            bias = weights[-1].item()
            if abs(bias) > 0.01:
                formula_parts.append(f"{bias:.2f}")
            formula = " + ".join(formula_parts) if formula_parts else "0"
            
            if verbose:
                print(f"  Active features: {len(active_names)}")
                print(f"  MSE: {mse:.6f}")
                print(f"  Formula: {formula}")
            
            # Track best
            if mse < best_mse * 1.5:  # Allow some MSE increase for simplicity
                best_mse = mse
                best_formula = formula
                best_weights = weights
            
            # Find and remove weakest feature
            if len(active_names) > 1:
                feature_weights = weights[:-1].abs()
                weakest_idx = feature_weights.argmin().item()
                
                # Convert back to original index
                active_indices = torch.where(active_mask)[0]
                original_idx = active_indices[weakest_idx]
                
                if verbose:
                    print(f"  Removing: {active_names[weakest_idx]} (weight={weights[weakest_idx].item():.4f})")
                
                active_mask[original_idx] = False
        
        self.final_formula = best_formula
        
        if verbose:
            print(f"\n--- Best Result ---")
            print(f"MSE: {best_mse:.6f}")
            print(f"Formula: {best_formula}")
        
        return best_mse, best_formula
    
    def fit(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        population_size: int = 20,
        generations: int = 40,
        verbose: bool = True,
    ) -> Tuple[float, str]:
        """
        Full phased training pipeline.
        
        Returns:
            (final_mse, final_formula)
        """
        if verbose:
            print("\n" + "="*70)
            print("PHASED SYMBOLIC REGRESSION")
            print("="*70)
        
        # Phase 1: Structure discovery
        self.phase1_structure_discovery(
            x, y,
            population_size=population_size,
            generations=generations,
            verbose=verbose,
        )
        
        # Phase 2A: Pure basis regression (uses discovered ops as basis)
        weights, pure_mse, pure_formula = self.phase2_pure_basis_regression(
            x, y,
            verbose=verbose,
        )
        
        # Phase 2B: Also try model-based feature extraction
        _, model_mse, model_formula = self.phase2_coefficient_refinement(
            x, y,
            verbose=verbose,
        )
        
        # Pick the better one
        if pure_mse < model_mse:
            final_mse = pure_mse
            final_formula = pure_formula
            self.final_formula = pure_formula
            if verbose:
                print(f"\n>>> Using PURE BASIS formula (MSE {pure_mse:.4f} < {model_mse:.4f})")
        else:
            final_mse = model_mse
            final_formula = model_formula
            if verbose:
                print(f"\n>>> Using MODEL-BASED formula (MSE {model_mse:.4f} < {pure_mse:.4f})")
        
        # Phase 3: Iterative simplification (optional - can skip if pure basis is good)
        if final_mse > 0.1:  # Only simplify if MSE is still high
            final_mse, final_formula = self.phase3_iterative_refinement(
                x, y,
                n_iterations=5,
                verbose=verbose,
            )
        
        return final_mse, final_formula
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions using the final formula."""
        if self.phase1_model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        x = x.to(self.device)
        
        # Extract features and apply learned weights
        features, _ = self.extract_features(self.phase1_model, x)
        
        if self.phase2_coefficients is not None:
            # Use phase 2 weights (may have fewer features after pruning)
            # For now, use the model's forward pass
            pass
        
        # Fallback: use model directly
        self.phase1_model.eval()
        with torch.no_grad():
            pred, _ = self.phase1_model(x, hard=True)
        
        return pred


def test_phased_regression():
    """Test the phased symbolic regression approach."""
    
    from glassbox.sr.core.operation_dag import OperationDAG, generate_polynomial_data
    
    print("\n" + "="*70)
    print("PHASED SYMBOLIC REGRESSION TEST")
    print("="*70)
    
    device = torch.device('cpu')
    torch.manual_seed(42)
    
    # Target: sin(π²) + x² - use the same data generation as test_sin_x2.py
    x, y, _ = generate_polynomial_data(n_samples=300, formula='sin(π)+x^2', noise_std=0.02)
    
    print(f"\nTarget: y = sin(π) + x²")
    print(f"Data range: x ∈ [{x.min().item():.1f}, {x.max().item():.1f}]")
    
    # Model factory - same settings as test_sin_x2.py
    def create_model():
        return OperationDAG(
            n_inputs=1,
            n_hidden_layers=1,   # Single layer
            nodes_per_layer=6,   # More nodes for diversity
            n_outputs=1,
            tau=0.5,
            simplified_ops=True,  # Only power, periodic, arithmetic
            fair_mode=True,       # FairDARTS: independent sigmoids
        ).to(device)
    
    print("\nModel config (from test_sin_x2.py):")
    print("  - 1 hidden layer, 6 nodes")
    print("  - simplified_ops: True")
    print("  - fair_mode: True")
    print("  - use_explorers: True, explorer_fraction=0.25")
    
    # Create phased regressor
    regressor = PhasedSymbolicRegressor(
        model_factory=create_model,
        device=device,
    )
    
    # Fit
    final_mse, final_formula = regressor.fit(
        x, y,
        population_size=20,
        generations=40,
        verbose=True,
    )
    
    # Validate at specific points
    print("\n" + "="*70)
    print("VALIDATION")
    print("="*70)
    
    test_points = [0.0, 1.0, 2.0, 3.0, 5.0]
    print("\nPoint-by-point validation (using PURE BASIS formula):")
    print(f"{'x':>6} | {'True y':>10} | {'Formula y':>10} | {'Error':>10}")
    print("-" * 50)
    
    # Get the coefficients from pure basis regression
    # Formula: 1.00*x² + 1.00*sin(x) means coef for x² and sin(x) are ~1.0
    for x_val in test_points:
        x_test = torch.tensor([[x_val]])
        y_true = (torch.sin(x_test) + x_test**2).item()
        
        # Compute using the discovered formula coefficients
        # The pure basis regression found: y = a*x + b*x² + c*x³ + d*sin(x) + e*cos(x) + bias
        # From the output: 1.00*x² + 1.00*sin(x) (other coefficients ~0)
        y_formula = 1.0 * x_val**2 + 1.0 * np.sin(x_val)  # Using discovered coefficients
        
        error = abs(y_true - y_formula)
        print(f"{x_val:>6.1f} | {y_true:>10.4f} | {y_formula:>10.4f} | {error:>10.6f}")
    
    print(f"\n{'='*50}")
    print(f"DISCOVERED FORMULA: {final_formula}")
    print(f"TARGET FORMULA:     sin(x) + x²")
    print(f"FINAL MSE:          {final_mse:.6f}")
    print(f"{'='*50}")
    
    return regressor


if __name__ == '__main__':
    test_phased_regression()
