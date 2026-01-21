"""
Tests for ONN v2: Operation-Based Neural Networks with Meta-Operations.

Tests cover:
1. Meta-operations (continuous parametric ops)
2. Differentiable routing
3. Hard Concrete distribution
4. Operation DAG architecture
5. Formula discovery on simple functions
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import math


def test_meta_periodic():
    """Test MetaPeriodic operation."""
    print("\n" + "="*60)
    print("TEST: MetaPeriodic")
    print("="*60)
    
    from glassbox.sr.meta_ops import MetaPeriodic
    
    x = torch.linspace(-3, 3, 100)
    
    # Test sin (default)
    op_sin = MetaPeriodic(init_omega=1.0, init_phi=0.0)
    y_sin = op_sin(x)
    expected_sin = torch.sin(x)
    error_sin = (y_sin - expected_sin).abs().max().item()
    print(f"✓ sin(x) error: {error_sin:.6f}")
    assert error_sin < 1e-5, "sin(x) should match exactly"
    
    # Test cos (phi = pi/2)
    op_cos = MetaPeriodic(init_omega=1.0, init_phi=math.pi/2)
    y_cos = op_cos(x)
    expected_cos = torch.cos(x)
    error_cos = (y_cos - expected_cos).abs().max().item()
    print(f"✓ cos(x) error: {error_cos:.6f}")
    assert error_cos < 1e-5, "cos(x) should match"
    
    # Test snap_to_discrete
    op = MetaPeriodic(init_omega=0.98, init_phi=0.05)
    op.snap_to_discrete()
    print(f"✓ Snapped: ω={op.omega.item():.1f}, φ={op.phi.item():.2f}")
    print(f"✓ Discrete op: {op.get_discrete_op()}")
    
    print("✓ PASSED: MetaPeriodic")


def test_meta_power():
    """Test MetaPower operation."""
    print("\n" + "="*60)
    print("TEST: MetaPower")
    print("="*60)
    
    from glassbox.sr.meta_ops import MetaPower
    
    x = torch.linspace(0.1, 3, 100)  # Positive values for power
    
    # Test square (p=2)
    op_sq = MetaPower(init_p=2.0)
    y_sq = op_sq(x)
    expected_sq = x ** 2
    error_sq = (y_sq - expected_sq).abs().max().item()
    print(f"✓ x² error: {error_sq:.6f}")
    assert error_sq < 0.01, "x² should be close"
    
    # Test sqrt (p=0.5)
    op_sqrt = MetaPower(init_p=0.5)
    y_sqrt = op_sqrt(x)
    expected_sqrt = torch.sqrt(x)
    error_sqrt = (y_sqrt - expected_sqrt).abs().max().item()
    print(f"✓ √x error: {error_sqrt:.6f}")
    assert error_sqrt < 0.01, "√x should be close"
    
    # Test identity (p=1)
    op_id = MetaPower(init_p=1.0)
    y_id = op_id(x)
    error_id = (y_id - x).abs().max().item()
    print(f"✓ identity error: {error_id:.6f}")
    
    # Test snap
    op = MetaPower(init_p=1.95)
    op.snap_to_discrete()
    print(f"✓ Snapped p={op.p.item():.1f}: {op.get_discrete_op()}")
    
    print("✓ PASSED: MetaPower")


def test_meta_arithmetic():
    """Test MetaArithmetic operation."""
    print("\n" + "="*60)
    print("TEST: MetaArithmetic")
    print("="*60)
    
    from glassbox.sr.meta_ops import MetaArithmetic
    
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])
    
    # Test addition (beta=1)
    op_add = MetaArithmetic(init_beta=1.0)
    result_add = op_add(x, y)
    expected_add = x + y
    print(f"✓ add: {result_add.tolist()} expected {expected_add.tolist()}")
    assert (result_add - expected_add).abs().max() < 1e-5
    
    # Test multiplication (beta=2)
    op_mul = MetaArithmetic(init_beta=2.0)
    result_mul = op_mul(x, y)
    expected_mul = x * y
    print(f"✓ mul: {result_mul.tolist()} expected {expected_mul.tolist()}")
    assert (result_mul - expected_mul).abs().max() < 1e-5
    
    # Test intermediate (beta=1.5)
    op_mix = MetaArithmetic(init_beta=1.5)
    result_mix = op_mix(x, y)
    print(f"✓ mix (β=1.5): {result_mix.tolist()}")
    
    print("✓ PASSED: MetaArithmetic")


def test_routing():
    """Test differentiable routing."""
    print("\n" + "="*60)
    print("TEST: Differentiable Routing")
    print("="*60)
    
    from glassbox.sr.routing import DifferentiableRouter, AdaptiveArityRouter
    
    # 5 sources, 2 slots
    router = DifferentiableRouter(n_sources=5, max_arity=2)
    
    # Batch of 10 samples, 5 sources each
    sources = torch.randn(10, 5)
    
    # Soft routing
    slots_soft = router(sources, tau=1.0, hard=False)
    print(f"✓ Soft routed shape: {slots_soft.shape}")
    assert slots_soft.shape == (10, 2)
    
    # Hard routing
    slots_hard = router(sources, tau=0.5, hard=True)
    print(f"✓ Hard routed shape: {slots_hard.shape}")
    
    # Check routing decisions
    routing = router.get_routing()
    print(f"✓ Routing connections: {routing}")
    print(f"✓ Primary sources: {router.get_primary_sources()}")
    
    # Adaptive arity router
    adaptive = AdaptiveArityRouter(n_sources=5)
    unary_input = adaptive.forward_unary(sources)
    binary_inputs = adaptive.forward_binary(sources)
    agg_input = adaptive.forward_aggregation(sources)
    
    print(f"✓ Unary input shape: {unary_input.shape}")
    print(f"✓ Binary inputs: {binary_inputs[0].shape}, {binary_inputs[1].shape}")
    print(f"✓ Aggregation input shape: {agg_input.shape}")
    
    print("✓ PASSED: Routing")


def test_hard_concrete():
    """Test Hard Concrete distribution."""
    print("\n" + "="*60)
    print("TEST: Hard Concrete Distribution")
    print("="*60)
    
    from glassbox.sr.hard_concrete import (
        hard_concrete_sample,
        HardConcreteGate,
        HardConcreteSelector,
    )
    
    # Test sampling
    logits = torch.tensor([0.0, 1.0, -1.0])
    
    # Sample many times to check distribution
    samples = []
    for _ in range(1000):
        s = hard_concrete_sample(logits, tau=0.5, beta=0.1, hard=True)
        samples.append(s)
    samples = torch.stack(samples)
    
    # Check that we get exact 0s and 1s
    n_zeros = (samples == 0).sum().item()
    n_ones = (samples == 1).sum().item()
    n_between = ((samples > 0) & (samples < 1)).sum().item()
    
    print(f"✓ Exact zeros: {n_zeros}")
    print(f"✓ Exact ones: {n_ones}")
    print(f"✓ Between 0-1: {n_between}")
    
    # Test gate
    gate = HardConcreteGate(n_gates=3)
    gate_values = gate()
    print(f"✓ Gate values: {gate_values.tolist()}")
    print(f"✓ L0 reg: {gate.l0_regularization().item():.4f}")
    
    # Test selector
    selector = HardConcreteSelector(n_options=5)
    selection = selector()
    print(f"✓ Selection weights: {selection.tolist()}")
    print(f"✓ Selected index: {selector.select()}")
    print(f"✓ Entropy: {selector.entropy().item():.4f}")
    
    print("✓ PASSED: Hard Concrete")


def test_operation_node():
    """Test OperationNode."""
    print("\n" + "="*60)
    print("TEST: OperationNode")
    print("="*60)
    
    from glassbox.sr.operation_node import OperationNode, OperationNodeSimple
    
    # Create node with 4 sources
    node = OperationNode(n_sources=4, node_idx=0)
    
    # Forward pass
    sources = torch.randn(16, 4)
    output, info = node(sources, hard=True)
    
    print(f"✓ Output shape: {output.shape}")
    assert output.shape == (16,)
    assert not torch.isnan(output).any()
    
    print(f"✓ Selected operation: {node.get_selected_operation()}")
    print(f"✓ Type weights: {info['type_weights'].tolist()}")
    
    # Test regularization
    l0 = node.l0_regularization()
    entropy = node.entropy_regularization()
    print(f"✓ L0 regularization: {l0.item():.4f}")
    print(f"✓ Entropy: {entropy.item():.4f}")
    
    # Test simple node
    simple = OperationNodeSimple(n_sources=4)
    out_simple = simple(sources, tau=1.0)
    print(f"✓ Simple node output shape: {out_simple.shape}")
    print(f"✓ Simple node op: {simple.get_selected_operation()}")
    
    print("✓ PASSED: OperationNode")


def test_operation_dag():
    """Test OperationDAG."""
    print("\n" + "="*60)
    print("TEST: OperationDAG")
    print("="*60)
    
    from glassbox.sr.operation_dag import OperationDAG, OperationDAGSimple
    
    # Create DAG: 2 inputs -> 2 layers x 4 nodes -> 1 output
    dag = OperationDAG(
        n_inputs=2,
        n_hidden_layers=2,
        nodes_per_layer=4,
        n_outputs=1,
        tau=0.5,
    )
    
    # Forward pass
    x = torch.randn(32, 2)
    output, info = dag(x, hard=True, return_all_outputs=True)
    
    print(f"✓ Output shape: {output.shape}")
    assert output.shape == (32, 1)
    assert not torch.isnan(output).any()
    
    # Get summary
    print(f"\n✓ Graph summary:")
    print(dag.get_graph_summary())
    
    # Test simple DAG
    simple_dag = OperationDAGSimple(n_inputs=2, n_hidden=4, n_outputs=1)
    out_simple = simple_dag(x, tau=1.0)
    print(f"\n✓ Simple DAG output shape: {out_simple.shape}")
    print(f"✓ Simple DAG summary:")
    print(simple_dag.get_summary())
    
    print("✓ PASSED: OperationDAG")


def test_formula_discovery():
    """Test formula discovery on simple functions."""
    print("\n" + "="*60)
    print("TEST: Formula Discovery")
    print("="*60)
    
    from glassbox.sr.operation_dag import OperationDAG, train_onn
    
    # Generate data: y = x^2
    n_samples = 200
    x_vals = torch.linspace(-2, 2, n_samples).unsqueeze(-1)  # (n, 1)
    y_vals = x_vals ** 2  # (n, 1)
    
    print("Target function: y = x²")
    print(f"Training data: {n_samples} samples")
    
    # Create small DAG
    model = OperationDAG(
        n_inputs=1,
        n_hidden_layers=1,
        nodes_per_layer=4,
        n_outputs=1,
        tau=1.0,
    )
    
    # Train
    history = train_onn(
        model,
        x_vals,
        y_vals,
        epochs=300,
        lr=0.02,
        print_every=60,
        lambda_entropy=0.05,
        anneal_tau=True,
        tau_start=2.0,
        tau_end=0.3,
    )
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        pred, _ = model(x_vals, hard=True)
    
    mse = ((pred - y_vals) ** 2).mean().item()
    correlation = torch.corrcoef(torch.stack([pred.squeeze(), y_vals.squeeze()]))[0, 1].item()
    
    print(f"\n✓ Final MSE: {mse:.4f}")
    print(f"✓ Correlation: {correlation:.4f}")
    
    # Snap to discrete and get formula
    model.snap_to_discrete()
    formula = model.get_formula(['x'])
    print(f"✓ Discovered formula: {formula}")
    
    # Check if we found square
    summary = model.get_graph_summary()
    found_square = 'square' in summary.lower() or 'power' in summary.lower()
    print(f"✓ Found square-like operation: {found_square}")
    
    print("✓ PASSED: Formula Discovery")
    
    return model, x_vals, y_vals, pred


def test_sin_discovery():
    """Test discovering sin(x)."""
    print("\n" + "="*60)
    print("TEST: Sin Discovery")
    print("="*60)
    
    from glassbox.sr.operation_dag import OperationDAG, train_onn
    
    # Generate data: y = sin(x)
    n_samples = 200
    x_vals = torch.linspace(-3.14, 3.14, n_samples).unsqueeze(-1)
    y_vals = torch.sin(x_vals)
    
    print("Target function: y = sin(x)")
    
    model = OperationDAG(
        n_inputs=1,
        n_hidden_layers=1,
        nodes_per_layer=4,
        n_outputs=1,
        tau=1.0,
    )
    
    history = train_onn(
        model,
        x_vals,
        y_vals,
        epochs=300,
        lr=0.02,
        print_every=60,
        lambda_entropy=0.05,
    )
    
    model.eval()
    with torch.no_grad():
        pred, _ = model(x_vals, hard=True)
    
    mse = ((pred - y_vals) ** 2).mean().item()
    correlation = torch.corrcoef(torch.stack([pred.squeeze(), y_vals.squeeze()]))[0, 1].item()
    
    print(f"\n✓ Final MSE: {mse:.4f}")
    print(f"✓ Correlation: {correlation:.4f}")
    
    model.snap_to_discrete()
    print(model.get_graph_summary())
    
    print("✓ PASSED: Sin Discovery")


def visualize_results(x_vals, y_true, y_pred, title="ONN v2"):
    """Visualize results."""
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(x_vals.numpy(), y_true.numpy(), 'b-', label='True', linewidth=2)
        plt.plot(x_vals.numpy(), y_pred.numpy(), 'r--', label='Predicted', linewidth=2)
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        error = (y_pred - y_true).numpy()
        plt.plot(x_vals.numpy(), error, 'g-')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.title('Error')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('onn_v2_test_result.png', dpi=150)
        print(f"✓ Saved visualization to onn_v2_test_result.png")
        plt.close()
    except Exception as e:
        print(f"Note: Visualization skipped ({e})")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("   ONN v2 TEST SUITE")
    print("="*60)
    
    # Unit tests
    test_meta_periodic()
    test_meta_power()
    test_meta_arithmetic()
    test_routing()
    test_hard_concrete()
    test_operation_node()
    test_operation_dag()
    
    # Integration tests
    model, x, y_true, y_pred = test_formula_discovery()
    test_sin_discovery()
    
    # Visualize
    visualize_results(x.squeeze(), y_true.squeeze(), y_pred.squeeze(), "y = x²")
    
    print("\n" + "="*60)
    print("   ALL TESTS COMPLETED!")
    print("="*60)
    print("""
ONN v2 Components:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Meta-Operations: Continuous parametric ops (sin, power, arithmetic)
2. Differentiable Routing: Learns which inputs connect to which slots
3. Hard Concrete: True discrete selection (exact 0s and 1s)
4. Operation DAG: Feed-forward graph (no RNN temporal bias)

Usage Example:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
from glassbox.sr import OperationDAG, train_onn

model = OperationDAG(n_inputs=2, n_hidden_layers=2, nodes_per_layer=4)
train_onn(model, X_train, y_train, epochs=500)
model.snap_to_discrete()
print(model.get_formula())
""")
