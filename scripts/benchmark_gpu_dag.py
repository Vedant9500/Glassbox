"""
Benchmark script for GPU-Optimized DAG.

Compares performance of:
1. Vanilla OperationDAG (Python loop, divergent)
2. GPUOptimizedDAG (Triton or Vectorized Torch)

Usage:
    python scripts/benchmark_gpu_dag.py
"""

import sys
import os
import time
import torch
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from glassbox.sr import OperationDAG, GPUOptimizedDAG, wrap_dag_for_gpu

def benchmark(n_layers=10, nodes_per_layer=50, batch_size=1024, iterations=100):
    print(f"Benchmarking with: Layers={n_layers}, Nodes/Layer={nodes_per_layer}, Batch={batch_size}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create large DAG
    dag = OperationDAG(
        n_inputs=10,
        n_hidden_layers=n_layers,
        nodes_per_layer=nodes_per_layer,
        n_outputs=1,
    ).to(device)
    dag.eval()
    
    # Wrap
    gpu_dag = wrap_dag_for_gpu(dag)
    
    # Show optimization info
    compile_status = "Enabled" if getattr(gpu_dag, '_compile_available', False) else "Disabled"
    backend = "Triton" if gpu_dag.use_triton else "Vectorized Torch"
    print(f"torch.compile: {compile_status}")
    print(f"Batched Ops Backend: {backend}")
    
    # Data
    x = torch.randn(batch_size, 10, device=device)
    
    # Warmup
    print("Warming up...")
    for _ in range(10):
        dag(x)
        gpu_dag(x)
    torch.cuda.synchronize()
    
    # Time Vanilla
    print("Running Vanilla DAG...")
    start = time.time()
    for _ in range(iterations):
        dag(x)
    torch.cuda.synchronize()
    vanilla_time = (time.time() - start) / iterations
    print(f"Vanilla Mean Time: {vanilla_time*1000:.2f} ms")
    
    # Time Optim
    print(f"Running GPU DAG ({backend})...")
    start = time.time()
    for _ in range(iterations):
        gpu_dag(x)
    torch.cuda.synchronize()
    optim_time = (time.time() - start) / iterations
    print(f"Optim Mean Time:   {optim_time*1000:.2f} ms")
    
    # Results
    speedup = vanilla_time / optim_time
    print("-" * 40)
    print(f"SPEEDUP: {speedup:.2f}x")
    print("-" * 40)
    
if __name__ == "__main__":
    benchmark()
