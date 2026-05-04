
import numpy as np
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# Setup path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

# Try to find _core
cpp_dir = root_dir / "glassbox" / "sr" / "cpp"
sys.path.insert(0, str(cpp_dir))

import _core

def target_fn(x):
    return 2.5 * np.sin(1.2 * x + 0.5) + 0.8 * np.exp(-0.3 * x) + 0.1 * x**2 - 0.5 * x + 1.2

def run_single_evolution(idx, n_samples, num_threads=-1):
    X = np.linspace(-5, 5, n_samples).astype(np.float64)
    y = target_fn(X).astype(np.float64)
    
    t0 = time.perf_counter()
    res = _core.run_evolution(
        [X],
        y,
        pop_size=100,
        generations=50,
        early_stop_mse=1e-12,
        num_threads=num_threads
    )
    elapsed = time.perf_counter() - t0
    print(f"Beam {idx} finished in {elapsed:.4f}s with MSE {res['best_mse']:.4e}")
    return elapsed

def test_concurrency():
    num_beams = 8
    n_samples = 2000 # Smaller for faster verification
    total_cores = os.cpu_count()
    threads_per_beam = max(1, total_cores // num_beams)
    
    print(f"System: {total_cores} cores")
    print(f"Config: {num_beams} beams, each using {threads_per_beam} threads.")
    
    print(f"\n--- Running {num_beams} beams sequentially (Full CPU for each) ---")
    seq_times = []
    for i in range(num_beams):
        seq_times.append(run_single_evolution(i, n_samples, num_threads=total_cores))
    avg_seq = sum(seq_times) / len(seq_times)
    t_seq = sum(seq_times)
    print(f"Total Sequential Time: {t_seq:.4f}s")
    
    print(f"\n--- Running {num_beams} beams in parallel (Isolated threads) ---")
    t_start = time.perf_counter()
    par_times = []
    with ThreadPoolExecutor(max_workers=num_beams) as executor:
        futures = [executor.submit(run_single_evolution, i, n_samples, num_threads=threads_per_beam) for i in range(num_beams)]
        for f in as_completed(futures):
            par_times.append(f.result())
    t_par = time.perf_counter() - t_start
    print(f"Total Parallel Time: {t_par:.4f}s")
    
    speedup = t_seq / t_par
    print(f"\nSpeedup: {speedup:.2f}x")
    
    if speedup < 1.2:
        print("\nWARNING: No significant speedup detected! Likely GIL contention or oversubscription.")
    else:
        print("\nSUCCESS: Multi-threading is working!")

if __name__ == "__main__":
    test_concurrency()
