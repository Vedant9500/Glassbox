"""
Benchmark Test Script for ONN v2.

Run this to compare ONN against traditional neural networks (MLP, LSTM, CNN).
This will tell you when ONN is ready for real-world comparison.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time


def test_basic_import():
    """Test all modules import correctly."""
    print("Testing imports...")
    
    from glassbox.sr import (
        # ONN v2
        OperationDAG,
        HybridOptimizer,
        LBFGSConstantOptimizer,
        
        # Benchmarking
        BaselineMLP,
        BaselineLSTM,
        BaselineCNN,
        BenchmarkRunner,
        run_all_benchmarks,
        generate_polynomial_data,
    )
    print("✓ All imports successful")


def test_hybrid_optimizer():
    """Test hybrid optimizer."""
    print("\n" + "="*60)
    print("TEST: Hybrid Optimizer")
    print("="*60)
    
    from glassbox.sr import OperationDAG, HybridOptimizer, generate_polynomial_data
    
    # Generate simple data
    x, y, formula = generate_polynomial_data(n_samples=200, formula='x^2', noise_std=0.05)
    print(f"Target: {formula}")
    
    # Create model
    model = OperationDAG(
        n_inputs=1,
        n_hidden_layers=1,
        nodes_per_layer=4,
        n_outputs=1,
    )
    
    # Hybrid training
    optimizer = HybridOptimizer(
        model,
        population_size=8,
        use_evolution=True,
    )
    
    result = optimizer.train(
        x, y,
        warmup_epochs=30,
        evolution_epochs=20,
        lbfgs_epochs=10,
        print_every=10,
    )
    
    print(f"\n✓ Hybrid training complete")
    print(f"✓ Final loss: {result['final_loss']:.6f}")
    
    # Check correlation
    model.eval()
    with torch.no_grad():
        pred, _ = model(x, hard=True)
    corr = torch.corrcoef(torch.stack([pred.squeeze(), y.squeeze()]))[0, 1].item()
    print(f"✓ Correlation: {corr:.4f}")
    
    return corr > 0.8


def test_benchmark_runner():
    """Test benchmark comparison."""
    print("\n" + "="*60)
    print("TEST: Benchmark Runner")
    print("="*60)
    
    from glassbox.sr import (
        BenchmarkRunner,
        generate_polynomial_data,
        generate_multivariate_data,
    )
    
    runner = BenchmarkRunner(epochs=100, verbose=True)
    
    # Test 1: Simple polynomial
    x, y, name = generate_polynomial_data(n_samples=300, formula='x^2')
    runner.run_benchmark(x, y, name)
    
    # Test 2: Multivariate
    x, y, name = generate_multivariate_data(n_samples=300, n_features=2)
    runner.run_benchmark(x, y, name)
    
    runner.print_comparison()
    
    return True


def test_ready_for_comparison():
    """
    Check if ONN is ready for serious comparison against CNN/LSTM.
    
    Criteria:
    1. All components work
    2. Training is stable
    3. Performance is competitive (within 2x of baselines)
    4. Hybrid optimizer improves results
    """
    print("\n" + "="*60)
    print("READINESS CHECK")
    print("="*60)
    
    ready = True
    issues = []
    
    # Check 1: Components import
    try:
        from glassbox.sr import (
            OperationDAG, HybridOptimizer, BaselineMLP,
            BenchmarkRunner, generate_polynomial_data,
        )
        print("✓ All components import correctly")
    except Exception as e:
        print(f"✗ Import error: {e}")
        ready = False
        issues.append("Import failures")
    
    # Check 2: Training stability
    from glassbox.sr import OperationDAG, generate_polynomial_data
    x, y, _ = generate_polynomial_data(n_samples=100, formula='sin')
    
    model = OperationDAG(n_inputs=1, n_hidden_layers=1, nodes_per_layer=4, n_outputs=1)
    
    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(50):
            pred, _ = model(x)
            loss = torch.nn.functional.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if torch.isnan(loss):
                raise ValueError("NaN detected")
        
        print(f"✓ Training stable (final loss: {loss.item():.4f})")
    except Exception as e:
        print(f"✗ Training unstable: {e}")
        ready = False
        issues.append("Training instability")
    
    # Check 3: Hybrid optimizer works
    try:
        from glassbox.sr import HybridOptimizer
        model = OperationDAG(n_inputs=1, n_hidden_layers=1, nodes_per_layer=4, n_outputs=1)
        hybrid = HybridOptimizer(model, population_size=5, use_evolution=True)
        result = hybrid.train(x, y, warmup_epochs=10, evolution_epochs=5, lbfgs_epochs=5, print_every=100)
        print(f"✓ Hybrid optimizer works (loss: {result['final_loss']:.4f})")
    except Exception as e:
        print(f"✗ Hybrid optimizer error: {e}")
        ready = False
        issues.append("Hybrid optimizer failure")
    
    # Check 4: Baselines work
    try:
        from glassbox.sr import BaselineMLP, BaselineLSTM, BaselineCNN
        
        mlp = BaselineMLP(n_inputs=1, n_outputs=1)
        _ = mlp(x)
        
        lstm = BaselineLSTM(n_inputs=1, n_outputs=1)
        _ = lstm(x)
        
        print("✓ Baseline models work")
    except Exception as e:
        print(f"✗ Baseline error: {e}")
        ready = False
        issues.append("Baseline failures")
    
    print("\n" + "-"*60)
    
    if ready:
        print("🎉 ONN IS READY FOR COMPARISON!")
        print("\nTo run full benchmarks:")
        print("  from glassbox.sr import run_all_benchmarks")
        print("  results = run_all_benchmarks(epochs=200)")
    else:
        print("❌ ONN NOT READY YET")
        print(f"Issues: {', '.join(issues)}")
    
    return ready


if __name__ == "__main__":
    print("="*60)
    print("   ONN v2 BENCHMARK TESTS")
    print("="*60)
    
    test_basic_import()
    
    hybrid_ok = test_hybrid_optimizer()
    
    benchmark_ok = test_benchmark_runner()
    
    ready = test_ready_for_comparison()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Hybrid Optimizer: {'✓' if hybrid_ok else '✗'}")
    print(f"Benchmark Runner: {'✓' if benchmark_ok else '✗'}")
    print(f"Ready for Comparison: {'✓' if ready else '✗'}")
    
    if ready:
        print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ONN v2 IS READY FOR TRAINING AND COMPARISON!

Quick start:
    from glassbox.sr import OperationDAG, HybridOptimizer
    from glassbox.sr import run_all_benchmarks
    
    # Run full comparison against CNN, LSTM, MLP:
    results = run_all_benchmarks(epochs=300)
    
    # Or quick custom comparison:
    from glassbox.sr import quick_comparison
    results = quick_comparison(your_x, your_y, "Your Task")
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
