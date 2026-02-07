"""
Test script for GPU-optimized DAG implementation.

Verifies:
1. Correctness: GPUOptimizedDAG produces same output as vanilla OperationDAG
2. Registry: BatchedOperationRegistry correctly groups nodes by operation
3. Gradients: Backward pass produces valid gradients
"""

import sys
import os
import torch
import torch.nn as nn

# Add project root to path regardless of OS
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from glassbox.sr import (
    OperationDAG,
    GPUOptimizedDAG,
    BatchedOperationRegistry,
    wrap_dag_for_gpu,
)


def test_registry_grouping():
    """Test that registry correctly groups nodes by operation type."""
    print("=" * 60)
    print("Test 1: Registry Operation Grouping")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create a small DAG
    dag = OperationDAG(
        n_inputs=1,
        n_hidden_layers=2,
        nodes_per_layer=4,
        n_outputs=1,
    ).to(device)
    
    # Create registry and update from model
    registry = BatchedOperationRegistry(device=device)
    registry.update_from_model(dag)
    
    print(registry.summary())
    print()
    
    # Check that all nodes are accounted for
    total_nodes_in_batches = sum(batch.n_nodes for batch in registry.batches.values())
    expected_nodes = 2 * 4  # n_hidden_layers * nodes_per_layer
    
    print(f"Total nodes in batches: {total_nodes_in_batches}")
    print(f"Expected nodes: {expected_nodes}")
    
    assert total_nodes_in_batches == expected_nodes, \
        f"Mismatch: {total_nodes_in_batches} != {expected_nodes}"
    
    print("✓ Registry grouping test PASSED")
    return True


def test_forward_correctness():
    """Test that GPUOptimizedDAG produces same output as vanilla DAG."""
    print("\n" + "=" * 60)
    print("Test 2: Forward Pass Correctness")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create DAG
    dag = OperationDAG(
        n_inputs=1,
        n_hidden_layers=2,
        nodes_per_layer=4,
        n_outputs=1,
    ).to(device)
    dag.eval()
    
    # Wrap for GPU
    gpu_dag = GPUOptimizedDAG(dag, use_triton=False)  # Use fallback for correctness test
    
    # Test data
    x = torch.randn(32, 1, device=device)
    
    # Forward passes
    with torch.no_grad():
        vanilla_output, _ = dag(x)
        gpu_output, _ = gpu_dag(x)
    
    # Compare
    max_diff = (vanilla_output - gpu_output).abs().max().item()
    print(f"Max difference: {max_diff:.2e}")
    
    assert max_diff < 1e-5, f"Outputs differ by {max_diff}"
    
    print("✓ Forward correctness test PASSED")
    return True


def test_gradient_flow():
    """Test that gradients flow properly through the wrapper."""
    print("\n" + "=" * 60)
    print("Test 3: Gradient Flow")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create DAG
    dag = OperationDAG(
        n_inputs=1,
        n_hidden_layers=2,
        nodes_per_layer=4,
        n_outputs=1,
    ).to(device)
    
    gpu_dag = wrap_dag_for_gpu(dag)
    
    # Test data
    x = torch.randn(32, 1, device=device)
    y = torch.randn(32, 1, device=device)
    
    # Forward + backward
    gpu_dag.train()
    output, _ = gpu_dag(x)
    loss = nn.MSELoss()(output, y)
    loss.backward()
    
    # Check that gradients exist
    grad_count = 0
    for name, param in gpu_dag.named_parameters():
        if param.grad is not None:
            grad_count += 1
            if param.grad.isnan().any():
                print(f"  WARNING: NaN gradient in {name}")
    
    print(f"Parameters with gradients: {grad_count}")
    
    assert grad_count > 0, "No gradients computed"
    
    print("✓ Gradient flow test PASSED")
    return True


def test_cache_invalidation():
    """Test that cache invalidation works correctly."""
    print("\n" + "=" * 60)
    print("Test 4: Cache Invalidation")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dag = OperationDAG(
        n_inputs=1,
        n_hidden_layers=1,
        nodes_per_layer=2,
        n_outputs=1,
    ).to(device)
    
    gpu_dag = GPUOptimizedDAG(dag)
    
    # First forward builds cache
    x = torch.randn(8, 1, device=device)
    gpu_dag(x)
    
    assert gpu_dag._cache_valid, "Cache should be valid after forward"
    
    # Invalidate cache
    gpu_dag.invalidate_cache()
    
    assert not gpu_dag._cache_valid, "Cache should be invalid after invalidate_cache()"
    
    # Forward should rebuild cache
    gpu_dag(x)
    
    assert gpu_dag._cache_valid, "Cache should be valid after second forward"
    
    print("✓ Cache invalidation test PASSED")
    return True


def run_all_tests():
    """Run all verification tests."""
    print("\n" + "=" * 60)
    print("GPU-Optimized DAG Test Suite")
    print("=" * 60 + "\n")
    
    tests = [
        test_registry_grouping,
        test_forward_correctness,
        test_gradient_flow,
        test_cache_invalidation,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED with error: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
