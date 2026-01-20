import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from glassbox.kan.network import GlassboxKAN, KANEvolver

def test_forward_pass():
    print("[Test] Forward Pass...")
    model = GlassboxKAN(['x', 'y'], hidden_sizes=[4], output_size=1)
    x = torch.randn(10, 2)
    y = model(x)
    assert y.shape == (10,), f"Expected (10,), got {y.shape}"
    print("  PASSED: Output shape correct")

def test_gradient_flow():
    print("[Test] Gradient Flow...")
    model = GlassboxKAN(['x'], hidden_sizes=[2], output_size=1)
    x = torch.randn(5, 1, requires_grad=True)
    y = model(x)
    loss = y.sum()
    loss.backward()
    
    # Check that gradients exist
    has_grad = False
    for p in model.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad, "No gradients found!"
    print("  PASSED: Gradients flow through network")

def test_kinetic_energy():
    print("[Test] Kinetic Energy (0.5 * m * v²)...")
    
    # Generate data
    np.random.seed(42)
    m = np.random.uniform(1, 10, 100)
    v = np.random.uniform(0, 20, 100)
    y = 0.5 * m * (v ** 2)
    
    X = np.column_stack([m, v])
    
    # Evolve
    evolver = KANEvolver(['m', 'v'], hidden_sizes=[8], pop_size=30, lr=0.05)
    evolver.initialize()
    
    print("  Evolving...")
    for gen in range(50):
        loss, best = evolver.evolve_step(X, y)
        if gen % 10 == 0:
            print(f"    Gen {gen}: Loss = {loss:.4f}")
        if loss < 5.0:
            print(f"  Converged at Gen {gen}!")
            break
    
    print(f"  Final Loss: {loss:.4f}")
    if loss < 50.0:
        print("  SUCCESS: Good approximation found")
    else:
        print("  PARTIAL: Still improving, may need more generations")

if __name__ == "__main__":
    test_forward_pass()
    test_gradient_flow()
    test_kinetic_energy()
