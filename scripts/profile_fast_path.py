"""Quick profiler to identify where fast-path time is spent.
Runs 5 representative problems and prints per-phase timing."""
import sys, os, time
import numpy as np

# Setup path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from classifier_fast_path import run_fast_path

# Representative test problems with increasing difficulty
TEST_PROBLEMS = [
    ("sin(x)",             (-6, 6)),     # trivial
    ("x^2 + sin(x) + 1",  (-5, 5)),     # medium
    ("sin(3*x)",           (-6, 6)),     # frequency detection
    ("exp(-x)*sin(x)",     (0, 10)),     # damped sine  
    ("x^2*exp(-x)*cos(3*x)", (0, 8)),   # hard composite
]

CLASSIFIER_PATH = "models/curve_classifier_wide.pt"

import torch

for formula, (xmin, xmax) in TEST_PROBLEMS:
    print(f"\n{'='*60}")
    print(f"FORMULA: {formula}  [{xmin}, {xmax}]")
    print(f"{'='*60}")
    
    x = np.linspace(xmin, xmax, 300)
    safe_env = {
        'x': x, 'np': np,
        'sin': np.sin, 'cos': np.cos,
        'exp': lambda z: np.exp(np.clip(z, -30, 30)),
        'log': lambda z: np.log(np.abs(z) + 1e-6),
        'abs': np.abs, 'sqrt': np.sqrt,
        'pi': np.pi,
    }
    y = eval(formula.replace('^', '**'), {'__builtins__': None}, safe_env)
    y = np.asarray(y, dtype=np.float64).flatten()
    
    x_t = torch.tensor(x, dtype=torch.float32).reshape(-1, 1)
    y_t = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    
    t0 = time.perf_counter()
    result = run_fast_path(x_t, y_t, classifier_path=CLASSIFIER_PATH, device='cpu')
    elapsed = time.perf_counter() - t0
    
    if result:
        print(f"\n  >>> RESULT: {result.get('formula', 'N/A')[:60]}")
        print(f"  >>> MSE: {result.get('mse', 'N/A')}")
    else:
        print(f"\n  >>> FAST PATH NOT APPLICABLE")
    print(f"  >>> TOTAL TIME: {elapsed:.3f}s")

print(f"\n{'='*60}")
print("DONE")
