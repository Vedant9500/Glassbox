"""
Long Benchmark Script for ONN v2.

Runs comprehensive benchmarks with improved training (500 epochs).
Compares ONN against MLP, LSTM, CNN on multiple tasks.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import time


def run_long_benchmark():
    """Run comprehensive ONN benchmarks with improved training."""
    
    from glassbox.sr import (
        OperationDAG,
        BaselineMLP,
        BaselineLSTM,
        BaselineCNN,
        generate_polynomial_data,
        generate_multivariate_data,
        get_device,
    )
    from glassbox.sr.training import train_onn_improved
    
    device = get_device()
    print("="*70)
    print("   ONN v2 COMPREHENSIVE BENCHMARK")
    print("="*70)
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Benchmark configurations
    benchmarks = [
        {'formula': 'x^2', 'name': 'y = x²', 'n_inputs': 1},
        {'formula': 'sin', 'name': 'y = sin(x)', 'n_inputs': 1},
        {'formula': 'x^2+x', 'name': 'y = x² + x', 'n_inputs': 1},
        {'formula': 'exp', 'name': 'y = exp(-x²)', 'n_inputs': 1},
    ]
    
    epochs = 500
    results = []
    
    for bench in benchmarks:
        print("\n" + "="*70)
        print(f"BENCHMARK: {bench['name']}")
        print("="*70)
        
        # Generate data
        x, y, _ = generate_polynomial_data(
            n_samples=500,
            noise_std=0.05,
            formula=bench['formula']
        )
        
        # Split
        n_train = 400
        x_train, x_val = x[:n_train], x[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]
        
        bench_result = {'task': bench['name'], 'models': {}}
        
        # ========== ONN ==========
        print("\n--- ONN (Improved Training) ---")
        onn = OperationDAG(
            n_inputs=bench['n_inputs'],
            n_hidden_layers=2,
            nodes_per_layer=4,
            n_outputs=1,
        )
        
        start = time.time()
        onn_result = train_onn_improved(
            onn, x_train, y_train,
            epochs=epochs,
            lr=0.02,
            print_every=100,
            device=device,
        )
        onn_time = time.time() - start
        
        # Evaluate on validation
        onn.eval()
        x_val_dev = x_val.to(device)
        y_val_dev = y_val.to(device)
        
        # Normalize for evaluation
        x_mean, x_std = x_train.mean(), x_train.std()
        y_mean, y_std = y_train.mean(), y_train.std()
        
        with torch.no_grad():
            x_val_norm = (x_val_dev - x_mean.to(device)) / x_std.to(device)
            pred_norm, _ = onn(x_val_norm, hard=True)
            pred = pred_norm * y_std.to(device) + y_mean.to(device)
            onn_mse = nn.functional.mse_loss(pred, y_val_dev).item()
            onn_corr = torch.corrcoef(torch.stack([
                pred.squeeze().cpu(), y_val.squeeze()
            ]))[0, 1].item()
        
        bench_result['models']['ONN'] = {
            'mse': onn_mse,
            'corr': onn_corr,
            'time': onn_time,
            'formula': onn_result['formula'],
            'params': sum(p.numel() for p in onn.parameters()),
        }
        
        print(f"\nONN Result: MSE={onn_mse:.4f}, Corr={onn_corr:.4f}")
        
        # ========== MLP Baseline ==========
        print("\n--- MLP Baseline ---")
        mlp = BaselineMLP(n_inputs=bench['n_inputs'], n_hidden=64, n_layers=2, n_outputs=1).to(device)
        optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01)
        
        start = time.time()
        x_train_dev = x_train.to(device)
        y_train_dev = y_train.to(device)
        
        for epoch in range(epochs):
            mlp.train()
            optimizer.zero_grad()
            pred = mlp(x_train_dev)
            loss = nn.functional.mse_loss(pred, y_train_dev)
            loss.backward()
            optimizer.step()
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
        
        mlp_time = time.time() - start
        
        mlp.eval()
        with torch.no_grad():
            pred = mlp(x_val_dev)
            mlp_mse = nn.functional.mse_loss(pred, y_val_dev).item()
            mlp_corr = torch.corrcoef(torch.stack([
                pred.squeeze().cpu(), y_val.squeeze()
            ]))[0, 1].item()
        
        bench_result['models']['MLP'] = {
            'mse': mlp_mse,
            'corr': mlp_corr,
            'time': mlp_time,
            'formula': 'Black box',
            'params': sum(p.numel() for p in mlp.parameters()),
        }
        
        print(f"MLP Result: MSE={mlp_mse:.4f}, Corr={mlp_corr:.4f}")
        
        results.append(bench_result)
    
    # ========== Summary ==========
    print("\n" + "="*70)
    print("   FINAL COMPARISON SUMMARY")
    print("="*70)
    
    print(f"\n{'Task':<20} {'Model':<8} {'MSE':<12} {'Corr':<10} {'Params':<10} {'Time':<8}")
    print("-"*70)
    
    for result in results:
        for model_name, stats in result['models'].items():
            print(f"{result['task']:<20} {model_name:<8} {stats['mse']:<12.4f} "
                  f"{stats['corr']:<10.4f} {stats['params']:<10} {stats['time']:<8.1f}")
    
    # ONN vs MLP summary
    print("\n" + "="*70)
    print("   ONN ADVANTAGES")
    print("="*70)
    
    print("\nDiscovered Formulas (ONN only):")
    for result in results:
        onn_stats = result['models'].get('ONN', {})
        print(f"  {result['task']}: {onn_stats.get('formula', 'N/A')[:60]}")
    
    print("\nParameter Efficiency:")
    for result in results:
        onn_params = result['models'].get('ONN', {}).get('params', 0)
        mlp_params = result['models'].get('MLP', {}).get('params', 0)
        ratio = mlp_params / max(onn_params, 1)
        print(f"  {result['task']}: ONN uses {ratio:.1f}x fewer parameters")
    
    return results


def run_multivariate_benchmark():
    """Benchmark on multivariate functions."""
    from glassbox.sr import (
        OperationDAG,
        BaselineMLP,
        generate_multivariate_data,
        get_device,
    )
    from glassbox.sr.training import train_onn_improved
    
    device = get_device()
    
    print("\n" + "="*70)
    print("   MULTIVARIATE BENCHMARK: y = x₁·x₂ + sin(x₃)")
    print("="*70)
    
    x, y, name = generate_multivariate_data(n_samples=500, n_features=3)
    n_train = 400
    x_train, x_val = x[:n_train], x[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    
    # ONN
    print("\n--- ONN ---")
    onn = OperationDAG(n_inputs=3, n_hidden_layers=2, nodes_per_layer=6, n_outputs=1)
    result = train_onn_improved(onn, x_train, y_train, epochs=500, print_every=100, device=device)
    
    # MLP
    print("\n--- MLP ---")
    mlp = BaselineMLP(n_inputs=3, n_hidden=64, n_outputs=1).to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01)
    
    x_train_dev = x_train.to(device)
    y_train_dev = y_train.to(device)
    
    for epoch in range(500):
        mlp.train()
        optimizer.zero_grad()
        pred = mlp(x_train_dev)
        loss = nn.functional.mse_loss(pred, y_train_dev)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    
    mlp.eval()
    x_val_dev = x_val.to(device)
    y_val_dev = y_val.to(device)
    with torch.no_grad():
        pred = mlp(x_val_dev)
        mlp_mse = nn.functional.mse_loss(pred, y_val_dev).item()
        mlp_corr = torch.corrcoef(torch.stack([pred.squeeze().cpu(), y_val.squeeze()]))[0, 1].item()
    
    print(f"\nSummary:")
    print(f"  ONN MSE: {result['final_mse']:.4f}, Corr: {result['correlation']:.4f}")
    print(f"  MLP MSE: {mlp_mse:.4f}, Corr: {mlp_corr:.4f}")
    print(f"  ONN Formula: {result['formula'][:80]}")


if __name__ == "__main__":
    results = run_long_benchmark()
    run_multivariate_benchmark()
    
    print("\n" + "="*70)
    print("   BENCHMARK COMPLETE!")
    print("="*70)
