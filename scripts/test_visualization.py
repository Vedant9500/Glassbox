"""
Test script for ONN Evolution with Real-time Visualization.

This demonstrates the PHASED approach:
- Phase 1: Structure Discovery (find which operations: sin, x², etc.)
- Phase 2: Pure Basis Regression (get exact coefficients)

With real-time visualization showing:
- Network architecture with operations
- Training progress curves
- Formula evolution
- Prediction vs target fit
"""

import torch
import numpy as np
import sys
sys.path.insert(0, r'd:\Glassbox')

from glassbox.sr.operation_dag import OperationDAG
from glassbox.sr.evolution import EvolutionaryONNTrainer, train_onn_evolutionary
from glassbox.sr.visualization import LiveTrainingVisualizer
from glassbox.sr.phased_regression import PhasedSymbolicRegressor


def main_phased(lite_mode: bool = False):
    """Main function using PHASED symbolic regression."""
    print("="*70)
    print("PHASED SYMBOLIC REGRESSION WITH VISUALIZATION")
    print("="*70)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create test data: y = π*sin(x) + x²/π
    # This tests coefficient discovery with π constant
    import math
    PI = math.pi
    print("\nTarget function: y = π*sin(x) + x²/π")
    print(f"Challenge: Discover coefficients π ≈ {PI:.4f} and 1/π ≈ {1/PI:.4f}")
    x = torch.linspace(-6, 6, 300).reshape(-1, 1)
    y = PI * torch.sin(x) + (x ** 2) / PI
    
    print(f"Data range: x ∈ [{x.min().item():.1f}, {x.max().item():.1f}]")
    print(f"Target range: y ∈ [{y.min().item():.2f}, {y.max().item():.2f}]")
    
    # Model factory (same settings that work best)
    def make_model():
        return OperationDAG(
            n_inputs=1,
            n_hidden_layers=2,   # Single layer for interpretability
            nodes_per_layer=6,   # More nodes for diversity
            n_outputs=1,
            simplified_ops=True,  # Only power, periodic, arithmetic
            fair_mode=True,       # FairDARTS independent sigmoids
        )
    
    # Create visualizer
    print("\nInitializing visualizer...")
    if lite_mode:
        print("Using LITE mode (faster, no network diagram)")
    visualizer = LiveTrainingVisualizer(
        update_every=1,  # Update every 2 generations
        figsize=(14, 8),
        lite_mode=lite_mode,
    )
    
    print("\n" + "="*70)
    print("PHASE 1: STRUCTURE DISCOVERY")
    print("="*70)
    print("Goal: Find which operations are useful (sin, x², etc.)")
    print("-"*70)
    
    # Create trainer with visualizer for Phase 1
    trainer = EvolutionaryONNTrainer(
        model_factory=make_model,
        population_size=30,
        elite_size=4,
        mutation_rate=0.5,
        constant_refine_steps=50,
        complexity_penalty=0.01,
        device=device,
        lamarckian=True,
        use_explorers=True,
        explorer_fraction=0.40,  # Boosted from 0.25 to find periodic functions
        explorer_mutation_rate=0.85,
        prune_coefficients=False,  # No pruning during structure search
        constant_refine_hard=True,
        visualizer=visualizer,  # <-- Attach visualizer
    )
    
    # Phase 1: Train to discover structure
    print("\nStarting Phase 1 evolution with visualization...")
    print("Watch the visualization window for real-time updates!")
    print("-"*70)
    
    results = trainer.train(
        x, y,
        generations=80,  # Increased from 40 for better structure discovery
        print_every=10,
    )
    
    phase1_model = results['model']
    phase1_formula = results['formula']
    phase1_mse = results['final_mse']
    
    print(f"\n{'='*70}")
    print("PHASE 1 COMPLETE")
    print(f"{'='*70}")
    print(f"Discovered structure: {phase1_formula}")
    print(f"Phase 1 MSE: {phase1_mse:.4f}")
    print("(Note: Coefficients may be off due to internal normalization)")
    
    # Phase 2: Coefficient Finalization (Hybrid Approach)
    print(f"\n{'='*70}")
    print("PHASE 2: HYBRID COEFFICIENT REFINEMENT")
    print(f"{'='*70}")
    print("Strategy: Compete two methods and pick the winner:")
    print("  A. Gradient Refinement (good for internal constants like sin(3.5x))")
    print("  B. Pure Basis Regression (good for standard functions like x^2)")
    print("-"*70)
    
    # --- Method A: Gradient Refinement ---
    print("\n>>> Method A: Gradient Refinement (optimizing internal constants)...")
    from glassbox.sr.evolution import finalize_model_coefficients
    
    # Create a copy for Method A to avoid disrupting Method B
    import copy
    model_a = copy.deepcopy(phase1_model)
    
    mse_a, formula_a = finalize_model_coefficients(
        model_a, x, y, 
        refine_internal_constants=True
    )
    print(f"Method A Result: MSE={mse_a:.6f}, Formula={formula_a}")
    
    # --- Method B: Pure Basis Regression ---
    print("\n>>> Method B: Pure Basis Regression (standard functions)...")
    
    # Parse operations from the discovered formula
    formula_lower = phase1_formula.lower()
    discovered_ops = set()
    if 'sin' in formula_lower or 'cos' in formula_lower: discovered_ops.add('periodic')
    if '^2' in formula_lower or '^3' in formula_lower or '**' in formula_lower: discovered_ops.add('power')
    if '*' in formula_lower and 'x' in formula_lower: discovered_ops.add('multiply')
    
    # Build pure basis functions
    features = [x]
    feature_names = ['x']
    
    if 'power' in discovered_ops or 'multiply' in discovered_ops:
        features.append(x ** 2)
        feature_names.append('x²')
        features.append(x ** 3)
        feature_names.append('x³')
    
    if 'periodic' in discovered_ops:
        features.append(torch.sin(x))
        feature_names.append('sin(x)')
        features.append(torch.cos(x))
        feature_names.append('cos(x)')
    
    # Solve least squares
    features_matrix = torch.cat(features, dim=1)
    n_samples = features_matrix.shape[0]
    features_with_bias = torch.cat([features_matrix, torch.ones(n_samples, 1, device=x.device)], dim=1)
    
    try:
        solution = torch.linalg.lstsq(features_with_bias, y)
        weights = solution.solution.squeeze()
        
        # Compute MSE
        pred = features_with_bias @ weights
        mse_b = torch.nn.functional.mse_loss(pred.squeeze(), y.squeeze()).item()
        
        # Build formula string for B
        formula_parts = []
        for name, w in zip(feature_names, weights[:-1]):
            if abs(w.item()) > 0.01:
                formula_parts.append(f"{w.item():.2f}*{name}")
        if abs(weights[-1].item()) > 0.01:
            formula_parts.append(f"{weights[-1].item():.2f}")
        formula_b = " + ".join(formula_parts) if formula_parts else "0"
        
        print(f"Method B Result: MSE={mse_b:.6f}")
        print(f"Method B Formula: {formula_b}")
        
    except Exception as e:
        print(f"Method B failed: {e}")
        mse_b = float('inf')
        formula_b = "ERROR"
        weights = None

    # --- Pick Winner ---
    print(f"\n{'-'*70}")
    
    # Handle NaN - treat as infinity (failed method)
    import math
    if math.isnan(mse_a):
        mse_a = float('inf')
        print("Warning: Method A returned NaN, treating as failure")
    if math.isnan(mse_b):
        mse_b = float('inf')
        print("Warning: Method B returned NaN, treating as failure")
    
    if mse_b < mse_a:
        print(f"WINNER: Method B (Pure Basis) - MSE {mse_b:.6f} vs {mse_a:.6f}")
        final_mse = mse_b
        final_formula = formula_b
        # Use simple weights for validation check
        used_method_b = True
    else:
        print(f"WINNER: Method A (Gradient Refinement) - MSE {mse_a:.6f} vs {mse_b:.6f}")
        final_mse = mse_a
        final_formula = formula_a
        # Use refined model for validation check
        phase1_model = model_a # Update main model to refined version
        used_method_b = False
    
    # Validation
    print(f"\n{'='*70}")
    print("VALIDATION")
    print(f"{'='*70}")
    
    test_points = [0.0, 1.0, 2.0, 3.0, 5.0]
    print(f"\n{'x':>6} | {'True y':>10} | {'Formula y':>10} | {'Error':>12}")
    print("-" * 50)
    
    for x_val in test_points:
        y_true = PI * np.sin(x_val) + (x_val ** 2) / PI
        
        # Compute using discovered coefficients
        # Compute using discovered coefficients
        if used_method_b:
            # METHOD B: Use weights and basis functions explicitly
            y_pred = 0.0
            for i, (name, w) in enumerate(zip(feature_names, weights[:-1])):
                if name == 'x': y_pred += w.item() * x_val
                elif name == 'x²': y_pred += w.item() * x_val**2
                elif name == 'x³': y_pred += w.item() * x_val**3
                elif name == 'sin(x)': y_pred += w.item() * np.sin(x_val)
                elif name == 'cos(x)': y_pred += w.item() * np.cos(x_val)
            y_pred += weights[-1].item() # bias
        else:
            # METHOD A: Use model prediction
            model_input = torch.tensor([[x_val]], device=device)
            y_pred = phase1_model(model_input, hard=True)[0].item()
        
        error = abs(y_true - y_pred)
        print(f"{x_val:>6.1f} | {y_true:>10.4f} | {y_pred:>10.4f} | {error:>12.6f}")
    
    # Final summary
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"TARGET FORMULA:     π*sin(x) + x²/π")
    print(f"DISCOVERED FORMULA: {final_formula}")
    print(f"Phase 1 MSE:        {phase1_mse:.4f} (structure discovery)")
    print(f"Phase 2 MSE:        {final_mse:.6f} (coefficient extraction)")
    
    # Check if we found the right operations with approximately correct coefficients
    # Expected: sin coefficient ≈ π (3.14159), x² coefficient ≈ 1/π (0.31831)
    sin_coef = None
    square_coef = None
    # Check if we found the right operations with approximately correct coefficients
    # Expected: sin coefficient ≈ π (3.14159), x² coefficient ≈ 1/π (0.31831)
    # We parse the string to find coefficients
    import re
    sin_coef = None
    square_coef = None
    
    # Look for "3.14*sin"
    sin_match = re.search(r'([\d\.-]+)\s*\*\s*sin', final_formula)
    if sin_match:
        sin_coef = float(sin_match.group(1))
        
    # Look for "0.32*x0^2" or similar
    sq_match = re.search(r'([\d\.-]+)\s*\*\s*x\d?\^2', final_formula)
    if sq_match:
        square_coef = float(sq_match.group(1))
    
    has_sin = sin_coef is not None and abs(sin_coef) > 0.5
    has_square = square_coef is not None and abs(square_coef) > 0.1
    
    if has_sin and has_square:
        sin_error = abs(sin_coef - PI) / PI * 100 if sin_coef else 100
        square_error = abs(square_coef - 1/PI) / (1/PI) * 100 if square_coef else 100
        print(f"\n✓ SUCCESS: Found BOTH sin(x) and x²!")
        print(f"  sin(x) coefficient: {sin_coef:.4f} (expected π ≈ {PI:.4f}, error: {sin_error:.1f}%)")
        print(f"  x² coefficient:     {square_coef:.4f} (expected 1/π ≈ {1/PI:.4f}, error: {square_error:.1f}%)")
    elif has_sin:
        print(f"\n~ PARTIAL: Found sin(x) (coef={sin_coef:.4f}) but not x²")
    elif has_square:
        print(f"\n~ PARTIAL: Found x² (coef={square_coef:.4f}) but not sin(x)")
    else:
        print("\n✗ FAILED: Did not find the target structure")
    
    # UPDATE VISUALIZATION with Phase 2 results
    print("\nUpdating visualization with Phase 2 results...")
    import matplotlib.pyplot as plt
    
    try:
        # Access the inner visualizer (ONNVisualizer)
        inner_viz = visualizer.visualizer
        
        if inner_viz is not None:
            # Update the internal state to reflect Phase 2 results
            inner_viz.current_formula = final_formula
            inner_viz.current_gen = "Phase 2 (Final)"
            inner_viz.best_fitness = phase2_mse
            inner_viz.correlation = 1.0  # Perfect fit after regression
            
            # Store prediction data for fit plot
            inner_viz.x_data = x
            inner_viz.y_data = y
            inner_viz.y_pred = pred.detach()
            
            # Redraw the formula panel
            if hasattr(inner_viz, '_draw_formula'):
                inner_viz._draw_formula()
            
            # Redraw the fit plot
            if hasattr(inner_viz, '_draw_fit_plot'):
                inner_viz._draw_fit_plot()
            
            # Update the title to show Phase 2
            if hasattr(inner_viz, 'ax_fit') and inner_viz.ax_fit is not None:
                inner_viz.ax_fit.set_title('Prediction vs Target (Phase 2)', 
                                           color='#00ff00', fontsize=12, fontweight='bold')
            
            # Force redraw
            if hasattr(inner_viz, 'fig') and inner_viz.fig is not None:
                inner_viz.fig.canvas.draw()
                inner_viz.fig.canvas.flush_events()
                print("Visualization updated with Phase 2 exact coefficients!")
                
    except Exception as e:
        print(f"Could not update visualization: {e}")
    
    # Keep visualization open
    print("\nVisualization window will stay open. Close it to exit.")
    plt.show(block=True)


def main_simple(lite_mode: bool = False):
    """Simple version - just y = x² without phased approach."""
    print("="*60)
    print("ONN EVOLUTION WITH REAL-TIME VISUALIZATION")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    print("\nTarget function: y = 2.5 * sin(3.2 * x) + 5")
    x = torch.linspace(-5, 5, 100).reshape(-1, 1)
    y = 2.5 * torch.sin(3.2 * x) + 5.0
    
    def make_model():
        return OperationDAG(
            n_inputs=1,
            n_hidden_layers=2,
            nodes_per_layer=4,
            n_outputs=1,
            simplified_ops=True,
            fair_mode=True,
        )
    
    print("\nInitializing visualizer...")
    visualizer = LiveTrainingVisualizer(
        update_every=3,
        figsize=(14, 8),
        lite_mode=lite_mode,
    )
    
    trainer = EvolutionaryONNTrainer(
        model_factory=make_model,
        population_size=15,
        elite_size=3,
        mutation_rate=0.3,
        constant_refine_steps=50,
        complexity_penalty=0.01,
        device=device,
        lamarckian=True,
        use_explorers=True,
        constant_refine_hard=True,
        visualizer=visualizer,
    )
    
    print("\nStarting evolution with visualization...")
    print("-"*60)
    
    results = trainer.train(x, y, generations=50, print_every=5)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    best_model = results['model']
    print(f"\nBest MSE: {results['final_mse']:.6f}")
    
    if hasattr(best_model, 'get_formula'):
        print(f"Discovered formula: {best_model.get_formula()}")

    # --- Phase 2: Hybrid Coefficient Finalization ---
    print(f"\n{'='*60}")
    print("PHASE 2: HYBRID COEFFICIENT REFINEMENT")
    print(f"{'='*60}")
    
    # 1. Gradient Refinement
    print("\n>>> Method A: Gradient Refinement...")
    from glassbox.sr.evolution import finalize_model_coefficients
    import copy
    
    model_a = copy.deepcopy(best_model)
    mse_a, formula_a = finalize_model_coefficients(
        model_a, x, y,
        refine_internal_constants=True
    )
    print(f"Method A Result: MSE={mse_a:.6f}, Formula={formula_a}")
    
    # 2. Pure Basis Regression
    print("\n>>> Method B: Pure Basis Regression...")
    formula_lower = results['formula'].lower()
    discovered_ops = set()
    if 'sin' in formula_lower or 'cos' in formula_lower: discovered_ops.add('periodic')
    if '^2' in formula_lower or '^3' in formula_lower or '**' in formula_lower: discovered_ops.add('power')
    if '*' in formula_lower and 'x' in formula_lower: discovered_ops.add('multiply')
    
    features = [x]
    feature_names = ['x']
    if 'power' in discovered_ops:
        features.append(x**2)
        feature_names.append('x²')
        features.append(x**3)
        feature_names.append('x³')
    if 'periodic' in discovered_ops:
        features.append(torch.sin(x))
        feature_names.append('sin(x)')
        features.append(torch.cos(x))
        feature_names.append('cos(x)')
        
    features_matrix = torch.cat(features, dim=1)
    features_with_bias = torch.cat([features_matrix, torch.ones(features_matrix.shape[0], 1, device=x.device)], dim=1)
    
    try:
        sol = torch.linalg.lstsq(features_with_bias, y)
        w = sol.solution.squeeze()
        pred_b = features_with_bias @ w
        # Squeeze y to match pred_b if necessary (pred_b is [N, 1] or [N])
        mse_b = torch.nn.functional.mse_loss(pred_b.squeeze(), y.squeeze()).item()
        
        # Build formula string for B
        formula_parts = []
        for name, weight in zip(feature_names, w):
            if abs(weight.item()) > 0.01:
                formula_parts.append(f"{weight.item():.2f}*{name}")
        
        # Check if list is empty (all weights ~0)
        formula_b = " + ".join(formula_parts) if formula_parts else "0 (all weights ~0)"
        
        print(f"Method B Result: MSE={mse_b:.6f}")
        print(f"Method B Formula: {formula_b}")
    except Exception as e:
        print(f"Method B failed: {e}")
        mse_b = float('inf')
    
    # Pick winner
    print(f"\n{'-'*60}")
    if mse_b < mse_a:
        print(f"WINNER: Method B (Pure Basis) - MSE {mse_b:.6f}")
    else:
        print(f"WINNER: Method A (Gradient Refinement) - MSE {mse_a:.6f}")
    
    print("\nVisualization window will stay open. Close it to exit.")
    import matplotlib.pyplot as plt
    plt.show(block=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ONN Visualization with Phased Regression")
    parser.add_argument('--simple', action='store_true', 
                       help='Use simple mode (just x², no phased approach)')
    parser.add_argument('--lite', action='store_true', 
                       help='Use lite mode (no network diagram, faster)')
    args = parser.parse_args()
    
    if args.simple:
        main_simple(lite_mode=args.lite)
    else:
        main_phased(lite_mode=args.lite)
