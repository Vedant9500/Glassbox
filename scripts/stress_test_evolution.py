
import numpy as np
import sys
import time
from pathlib import Path

# Setup path
cpp_dir = Path(__file__).resolve().parent.parent / "glassbox" / "sr" / "cpp"
sys.path.insert(0, str(cpp_dir))

import _core

def stress_test():
    print("=" * 72)
    print("EVOLUTION STRESS TEST (Differential Gramian Optimization)")
    print("=" * 72)
    
    # Complex formula with many parameters to trigger heavy LM refinement
    # Formula: a*sin(b*x + c) + d*exp(e*x) + f*x^2 + g*x + h
    # This involves many parameters and derivatives.
    def target_fn(x):
        return 2.5 * np.sin(1.2 * x + 0.5) + 0.8 * np.exp(-0.3 * x) + 0.1 * x**2 - 0.5 * x + 1.2

    # Vary N (number of points) to see the O(N) scaling
    for n_samples in [200, 1000, 5000]:
        print(f"\nTesting with N = {n_samples} points...")
        
        X = np.linspace(-5, 5, n_samples).astype(np.float64)
        y = target_fn(X).astype(np.float64)
        
        t0 = time.perf_counter()
        res = _core.run_evolution(
            [X],
            y,
            pop_size=50,
            generations=200,  # Small number of generations for quick test
            early_stop_mse=1e-10
        )
        elapsed = time.perf_counter() - t0
        
        print(f"  Time taken: {elapsed:.4f}s")
        print(f"  Best MSE:   {res['best_mse']:.6e}")
        print(f"  Formula:    {res['formula'][:80]}...")
        
        # Calculate approximate time per generation per sample
        time_per_gen = elapsed / 200
        time_per_eval = time_per_gen / 50 # roughly
        print(f"  Avg time per generation: {time_per_gen*1000:.2f}ms")

if __name__ == "__main__":
    stress_test()
