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
    
    # Test functions
    test_cases = [
        ('x^2', 'y = x²'),
        ('sin', 'y = sin(x)'),
        ('x^2+x', 'y = x² + x'),
    ]
    
    results = []
    
    for formula, name in test_cases:
        print("\n" + "="*70)
        print(f"TASK: {name}")
        print("="*70)
        
        # Generate data
        x, y, _ = generate_polynomial_data(n_samples=300, formula=formula, noise_std=0.02)
        
        n_train = 240
        x_train, x_val = x[:n_train], x[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]
        
        # Move to device
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        x_val = x_val.to(device)
        y_val = y_val.to(device)
        
        # Model factory
        def make_model():
            return OperationDAG(
                n_inputs=1,
                n_hidden_layers=1,  # Simpler for single-variable
                nodes_per_layer=4,
                n_outputs=1,
            )
        
        # Evolutionary training
        result = train_onn_evolutionary(
            make_model,
            x_train, y_train,
            population_size=15,
            generations=30,
            device=device,
        )
        
        # Evaluate on validation
        model = result['model']
        model.eval()
        with torch.no_grad():
            pred, _ = model(x_val, hard=True)
            val_mse = nn.functional.mse_loss(pred, y_val).item()
            val_corr = torch.corrcoef(torch.stack([
                pred.squeeze().cpu(), y_val.squeeze().cpu()
            ]))[0, 1].item()
        
        print(f"\nValidation MSE: {val_mse:.4f}")
        print(f"Validation Corr: {val_corr:.4f}")
        
        # Compare with MLP
        print("\n--- MLP Baseline (same epochs) ---")
        mlp = BaselineMLP(n_inputs=1, n_hidden=32, n_outputs=1).to(device)
        optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01)
        
        for epoch in range(300):  # Same effective compute
            mlp.train()
            optimizer.zero_grad()
            pred = mlp(x_train)
            loss = nn.functional.mse_loss(pred, y_train)
            loss.backward()
            optimizer.step()
        
        mlp.eval()
        with torch.no_grad():
            pred = mlp(x_val)
            mlp_mse = nn.functional.mse_loss(pred, y_val).item()
            mlp_corr = torch.corrcoef(torch.stack([
                pred.squeeze().cpu(), y_val.squeeze().cpu()
            ]))[0, 1].item()
        
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
