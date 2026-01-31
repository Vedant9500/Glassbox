"""
Benchmark script for ONN sparse routing optimization.

Measures forward pass time and memory with varying network sizes
to demonstrate the O(n) vs O(k) scaling improvement.

Usage:
    python scripts/benchmark_scaling.py
    python scripts/benchmark_scaling.py --sparse  # Test sparse routing
"""

import torch
import time
import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glassbox.sr.operation_dag import OperationDAG


def benchmark_forward_pass(
    model: torch.nn.Module,
    x: torch.Tensor,
    n_warmup: int = 5,
    n_runs: int = 20,
) -> float:
    """Benchmark forward pass time in milliseconds."""
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x, hard=True)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Timed runs
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(x, hard=True)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    elapsed = (time.perf_counter() - start) / n_runs * 1000  # ms
    return elapsed


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_benchmark(sparse_routing: bool = False, sparse_topk: int = 5):
    """Run scaling benchmark."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"ONN SCALING BENCHMARK")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Sparse Routing: {sparse_routing}")
    if sparse_routing:
        print(f"Sparse Top-K: {sparse_topk}")
    print(f"{'='*60}\n")
    
    # Test configurations: (n_layers, nodes_per_layer)
    configs = [
        (2, 4),    # Baseline: 8 total nodes
        (2, 8),    # 2x nodes
        (2, 16),   # 4x nodes
        (2, 32),   # 8x nodes
        (2, 64),   # 16x nodes - test for O(n²)
        (3, 16),   # More layers
        (4, 16),
        (4, 32),   # Larger network
        (6, 16),   # Deep network
    ]
    
    batch_size = 256
    n_inputs = 1
    
    print(f"{'Config':<15} {'Params':<12} {'n_sources(L)':<14} {'Time (ms)':<12} {'vs Baseline':<12}")
    print(f"{'-'*65}")
    
    baseline_time = None
    
    for n_layers, nodes_per_layer in configs:
        try:
            model = OperationDAG(
                n_inputs=n_inputs,
                n_hidden_layers=n_layers,
                nodes_per_layer=nodes_per_layer,
                sparse_routing=sparse_routing,
                sparse_topk=sparse_topk,
            ).to(device)
            
            x = torch.randn(batch_size, n_inputs, device=device)
            
            # Verify model works
            with torch.no_grad():
                output, _ = model(x, hard=True)
                assert output.shape == (batch_size, 1), f"Bad output shape: {output.shape}"
            
            # Benchmark
            elapsed_ms = benchmark_forward_pass(model, x)
            n_params = count_parameters(model)
            
            # Calculate n_sources for last layer (indicator of complexity)
            n_sources_last = n_inputs + (n_layers - 1) * nodes_per_layer
            
            if baseline_time is None:
                baseline_time = elapsed_ms
                ratio_str = "1.00x"
            else:
                ratio = elapsed_ms / baseline_time
                ratio_str = f"{ratio:.2f}x"
            
            config_str = f"L{n_layers} x N{nodes_per_layer}"
            print(f"{config_str:<15} {n_params:<12,} {n_sources_last:<14} {elapsed_ms:<12.3f} {ratio_str:<12}")
            
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"L{n_layers} x N{nodes_per_layer}: ERROR - {e}")
    
    print(f"\n{'='*60}")
    print("INTERPRETATION:")
    print("- 'vs Baseline' shows timing ratio compared to smallest config")
    print("- Ideal linear scaling: 2x nodes = 2x time")
    print("- Dense routing: 2x nodes ≈ 4x time (O(n²))")
    print("- Sparse routing: 2x nodes ≈ 2x time (O(n))")
    print(f"{'='*60}\n")


def compare_dense_vs_sparse():
    """Compare dense and sparse routing directly."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*70}")
    print("DENSE vs WINDOWED SOURCE COMPARISON")
    print(f"{'='*70}\n")
    
    configs = [
        (2, 16),
        (2, 32),
        (2, 64),
        (4, 16),
        (4, 32),
        (6, 16),
    ]
    
    batch_size = 256
    n_inputs = 1
    
    print(f"{'Config':<15} {'Dense (ms)':<12} {'Window=1':<12} {'Window=2':<12} {'Speedup(W1)':<12}")
    print(f"{'-'*65}")
    
    for n_layers, nodes_per_layer in configs:
        try:
            # Dense model (default)
            dense_model = OperationDAG(
                n_inputs=n_inputs,
                n_hidden_layers=n_layers,
                nodes_per_layer=nodes_per_layer,
                source_window=-1,  # Dense
            ).to(device)
            
            # Windowed model (window=1: only previous layer)
            window1_model = OperationDAG(
                n_inputs=n_inputs,
                n_hidden_layers=n_layers,
                nodes_per_layer=nodes_per_layer,
                source_window=1,
            ).to(device)
            
            # Windowed model (window=2)
            window2_model = OperationDAG(
                n_inputs=n_inputs,
                n_hidden_layers=n_layers,
                nodes_per_layer=nodes_per_layer,
                source_window=2,
            ).to(device)
            
            x = torch.randn(batch_size, n_inputs, device=device)
            
            dense_time = benchmark_forward_pass(dense_model, x)
            window1_time = benchmark_forward_pass(window1_model, x)
            window2_time = benchmark_forward_pass(window2_model, x)
            speedup = dense_time / window1_time
            
            config_str = f"L{n_layers} x N{nodes_per_layer}"
            print(f"{config_str:<15} {dense_time:<12.3f} {window1_time:<12.3f} {window2_time:<12.3f} {speedup:<12.2f}x")
            
            del dense_model, window1_model, window2_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"L{n_layers} x N{nodes_per_layer}: ERROR - {e}")
    
    print(f"\n{'='*70}")
    print("INTERPRETATION:")  
    print("- Window=1: Each layer only sees inputs + previous layer output")
    print("- Window=2: Each layer sees inputs + last 2 layers' outputs")
    print("- Speedup shows improvement from Window=1 vs Dense")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark ONN routing scaling")
    parser.add_argument("--sparse", action="store_true", help="Enable sparse routing")
    parser.add_argument("--topk", type=int, default=5, help="Top-K for sparse routing")
    parser.add_argument("--compare", action="store_true", help="Compare dense vs sparse")
    args = parser.parse_args()
    
    if args.compare:
        compare_dense_vs_sparse()
    else:
        run_benchmark(sparse_routing=args.sparse, sparse_topk=args.topk)


if __name__ == "__main__":
    main()
