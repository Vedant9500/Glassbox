"""
Test: Differentiable Tree-based Symbolic Regression
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from glassbox.sr.searcher import SymbolicSearcher

# Force CPU to avoid CUDA memory issues
DEVICE = torch.device('cpu')
print(f"Using device: {DEVICE}")


def test_kinetic_energy():
    """Test: Discover y = 0.5 * m * v^2"""
    print("\n" + "="*60)
    print("TEST: Kinetic Energy (0.5 * m * v²)")
    print("="*60)
    
    # Generate data
    np.random.seed(42)
    m = np.random.uniform(1, 10, 1000)
    v = np.random.uniform(0, 10, 1000)
    y = 0.5 * m * (v ** 2)
    
    # Normalize
    X_raw = np.column_stack([m, v])
    X_mean, X_std = X_raw.mean(axis=0), X_raw.std(axis=0)
    y_mean, y_std = y.mean(), y.std()
    X_norm = (X_raw - X_mean) / X_std
    y_norm = (y - y_mean) / y_std
    
    X = torch.tensor(X_norm, dtype=torch.float32)
    y_t = torch.tensor(y_norm, dtype=torch.float32)
    
    searcher = SymbolicSearcher(
        var_names=['m', 'v'],
        max_depth=2,
        lr=0.05,
        complexity_weight=0.0001
    )
    
    formula = searcher.train(X, y_t, epochs=2000, print_every=200)
    searcher.print_pareto()
    
    return formula


def test_simple_quadratic():
    """Test: Discover y = x^2 + 2x + 1"""
    print("\n" + "="*60)
    print("TEST: Simple Quadratic (x² + 2x + 1)")
    print("="*60)
    
    np.random.seed(42)
    x = np.random.uniform(-5, 5, 500)
    y = x**2 + 2*x + 1
    
    # Normalize data to prevent scale issues
    x_mean, x_std = x.mean(), x.std()
    y_mean, y_std = y.mean(), y.std()
    x_norm = (x - x_mean) / x_std
    y_norm = (y - y_mean) / y_std
    
    X = torch.tensor(x_norm.reshape(-1, 1), dtype=torch.float32)
    y_t = torch.tensor(y_norm, dtype=torch.float32)
    
    searcher = SymbolicSearcher(
        var_names=['x'],
        max_depth=2,  # Simpler tree
        lr=0.05,       # Higher learning rate
        complexity_weight=0.0001
    )
    
    formula = searcher.train(X, y_t, epochs=1000, print_every=100)
    searcher.print_pareto()
    
    return formula


def test_trig():
    """Test: Discover y = sin(x) + cos(x)"""
    print("\n" + "="*60)
    print("TEST: Trigonometry (sin(x) + cos(x))")
    print("="*60)
    
    np.random.seed(42)
    x = np.random.uniform(0, 2 * np.pi, 500)
    y = np.sin(x) + np.cos(x)
    
    # Normalize
    x_norm = (x - x.mean()) / x.std()
    y_norm = (y - y.mean()) / y.std()
    
    X = torch.tensor(x_norm.reshape(-1, 1), dtype=torch.float32)
    y_t = torch.tensor(y_norm, dtype=torch.float32)
    
    searcher = SymbolicSearcher(
        var_names=['x'],
        max_depth=2,
        lr=0.05,
        complexity_weight=0.0001
    )
    
    formula = searcher.train(X, y_t, epochs=1000, print_every=100)
    searcher.print_pareto()
    
    return formula


if __name__ == "__main__":
    # Run all tests
    test_simple_quadratic()
    test_trig()
    test_kinetic_energy()
