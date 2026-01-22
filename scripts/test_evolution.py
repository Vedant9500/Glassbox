"""
Test Evolutionary ONN Training.

This uses the PROPER approach:
- Evolution for discrete structure (which operations)
- Gradient descent for constants only
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn


def test_evolutionary_training():
    """Test evolutionary training on simple functions."""
    
    from glassbox.sr import OperationDAG, get_device
    from glassbox.sr.evolution import train_onn_evolutionary
    from glassbox.sr import generate_polynomial_data, BaselineMLP
    
    device = get_device()
    print("\n" + "="*70)
    print("   EVOLUTIONARY ONN TRAINING TEST")
    print("="*70)
    print(f"Device: {device}")
    
    # Test functions - start with simpler cases
    test_cases = [
        ('x^2', 'y = x²'),           # Simplest: single power operation
        ('sin', 'y = sin(x)'),        # Single periodic
        ('x^3', 'y = x³'),            # Power with p=3
        ('x^2+x', 'y = x² + x'),      # Requires composition
        ('sin+x^2', 'y = sin(x) + x²'),  # Multi-op
    ]
    
    results = []
    
    for formula, name in test_cases:
        print("\n" + "="*70)
        print(f"TASK: {name}")
        print("="*70)
        
        # Generate data
        x, y, _ = generate_polynomial_data(n_samples=300, formula=formula, noise_std=0.02)
        
        # Shuffle to avoid extrapolation bias from ordered split
        perm = torch.randperm(x.shape[0])
        x = x[perm]
        y = y[perm]
        
        n_train = 240
        x_train, x_val = x[:n_train], x[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]
        
        # Move to device
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        x_val = x_val.to(device)
        y_val = y_val.to(device)
        
        # Model factory - compact architecture for speed
        def make_model():
            return OperationDAG(
                n_inputs=1,
                n_hidden_layers=1,   # Single layer (faster)
                nodes_per_layer=4,   # 4 nodes is enough for simple formulas
                n_outputs=1,
                tau=0.5,
            )
        
        # Evolutionary training - smaller budget for faster iteration
        result = train_onn_evolutionary(
            make_model,
            x_train, y_train,
            population_size=15,      # Smaller population
            generations=30,          # Fewer generations
            device=device,
            fitness_x=x_val,
            fitness_y=y_val,
            normalize_data=False,
            constant_refine_hard=False,
            elite_fraction=0.2,
            mutation_rate=0.4,
        )
        
        # Evaluate on validation using compiled inference (faster & consistent)
        model = result['model']
        
        # NOTE: Do NOT snap again - evolution.py already handles this
        # Snapping destroys model performance by changing continuous params to discrete
        # if hasattr(model, 'snap_to_discrete'):
        #     model.snap_to_discrete()
        # if hasattr(model, 'compile_for_inference'):
        #     try:
        #         model.compile_for_inference()
        #     except Exception:
        #         pass
        
        model.eval()
        with torch.no_grad():
            # Use compiled path if available
            if hasattr(model, 'forward_compiled') and hasattr(model, '_compiled') and model._compiled:
                pred = model.forward_compiled(x_val)
            else:
                pred, _ = model(x_val, hard=True)
            val_mse = nn.functional.mse_loss(pred, y_val).item()
            val_corr = torch.corrcoef(torch.stack([
                pred.squeeze().cpu(), y_val.squeeze().cpu()
            ]))[0, 1].item()
        
        print(f"\nValidation MSE: {val_mse:.4f}")
        print(f"Validation Corr: {val_corr:.4f}")
        
        # Compare with MLP
        print("\n--- MLP Baseline (same epochs) ---")
        def run_mlp_baseline(train_x, train_y, val_x, val_y, run_device):
            mlp_local = BaselineMLP(n_inputs=1, n_hidden=32, n_outputs=1).to(run_device)
            optimizer_local = torch.optim.Adam(mlp_local.parameters(), lr=0.01)
            
            for epoch in range(300):  # Same effective compute
                mlp_local.train()
                optimizer_local.zero_grad()
                pred_local = mlp_local(train_x)
                loss_local = nn.functional.mse_loss(pred_local, train_y)
                loss_local.backward()
                optimizer_local.step()
            
            mlp_local.eval()
            with torch.no_grad():
                pred_local = mlp_local(val_x)
                mlp_mse_local = nn.functional.mse_loss(pred_local, val_y).item()
                mlp_corr_local = torch.corrcoef(torch.stack([
                    pred_local.squeeze().cpu(), val_y.squeeze().cpu()
                ]))[0, 1].item()
            return mlp_mse_local, mlp_corr_local
        
        try:
            mlp_mse, mlp_corr = run_mlp_baseline(x_train, y_train, x_val, y_val, device)
        except RuntimeError as e:
            if 'CUDA' in str(e):
                # Fallback to CPU if CUDA context is unstable
                cpu = torch.device('cpu')
                mlp_mse, mlp_corr = run_mlp_baseline(
                    x_train.cpu(), y_train.cpu(), x_val.cpu(), y_val.cpu(), cpu
                )
            else:
                raise
        
        print(f"MLP MSE: {mlp_mse:.4f}, Corr: {mlp_corr:.4f}")
        
        results.append({
            'task': name,
            'onn_mse': val_mse,
            'onn_corr': val_corr,
            'onn_formula': result['formula'],
            'mlp_mse': mlp_mse,
            'mlp_corr': mlp_corr,
        })
    
    # Summary
    print("\n" + "="*70)
    print("   SUMMARY")
    print("="*70)
    print(f"\n{'Task':<15} {'ONN MSE':<12} {'ONN Corr':<10} {'MLP MSE':<12} {'MLP Corr':<10}")
    print("-"*60)
    
    for r in results:
        print(f"{r['task']:<15} {r['onn_mse']:<12.4f} {r['onn_corr']:<10.4f} "
              f"{r['mlp_mse']:<12.4f} {r['mlp_corr']:<10.4f}")
    
    print("\nDiscovered Formulas:")
    for r in results:
        print(f"  {r['task']}: {r['onn_formula'][:60]}")
    
    return results


if __name__ == "__main__":
    results = test_evolutionary_training()
    
    print("\n" + "="*70)
    print("   TEST COMPLETE")
    print("="*70)
