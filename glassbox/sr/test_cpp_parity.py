import numpy as np
import sys
import os
from pathlib import Path

cpp_dir = Path(__file__).parent / 'cpp'
sys.path.insert(0, str(cpp_dir))

def main():
    try:
        import _core
        print("✅ Successfully imported _core")
    except ImportError as e:
        print(f"❌ Failed to import _core: {e}")
        print("Please build the C++ extension first using `python setup.py build_ext --inplace` in the cpp directory.")
        return

    # Generate some simple data: y = 2.0 * x^2.0 + sin(3.0 * x)
    np.random.seed(42)
    X = np.random.uniform(-3, 3, size=(100, 1))
    X_list = [X[:, 0]]
    
    y = 2.0 * (X[:, 0]**2) + np.sin(3.0 * X[:, 0])

    print("\n--- Test 1: Basic execution ---")
    res1 = _core.run_evolution(X_list, y, pop_size=10, generations=5, early_stop_mse=1e-8)
    print("Formula:", res1["formula"])
    print("MSE:", res1["best_mse"])

    print("\n--- Test 2: Seed Omegas ---")
    res2 = _core.run_evolution(X_list, y, pop_size=10, generations=5, early_stop_mse=1e-8, seed_omegas=[3.0, 5.0, 7.0])
    print("Formula:", res2["formula"])
    print("MSE:", res2["best_mse"])

    print("\nAll Python API bindings execute without errors.")

if __name__ == "__main__":
    main()
