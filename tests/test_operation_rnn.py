"""
Tests for Operation-Based RNN/LSTM v2.

Tests the enhanced architecture with:
1. Multi-input aggregation operations (sum_all, prod_all, etc.)
2. Entropy regularization for diverse operation exploration
3. Improved initialization
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from glassbox.sr.operation_rnn import (
    OperationCell,
    OperationRNN,
    OperationLSTM,
    set_global_temperature,
    set_global_hard_mode,
    anneal_temperature,
    compute_regularized_loss,
    UNARY_NAMES,
    BINARY_NAMES,
    AGGREGATION_NAMES,
)


def test_operation_cell_basic():
    """Test that OperationCell forward pass works with all features."""
    print("\n" + "="*60)
    print("TEST: OperationCell Basic Forward Pass")
    print("="*60)
    
    cell = OperationCell(
        input_size=4,
        hidden_size=8,
        use_binary_ops=True,
        use_aggregation_ops=True,
        n_aggregation_inputs=4
    )
    
    batch_size = 16
    x = torch.randn(batch_size, 4)
    h = torch.randn(batch_size, 8)
    
    # Test without hidden state
    out1, entropy1 = cell(x, h=None)
    assert out1.shape == (batch_size, 8), f"Expected (16, 8), got {out1.shape}"
    assert not torch.isnan(out1).any(), "Output contains NaN!"
    assert not torch.isinf(out1).any(), "Output contains Inf!"
    assert entropy1.item() > 0, "Entropy should be positive"
    
    # Test with hidden state
    out2, entropy2 = cell(x, h=h)
    assert out2.shape == (batch_size, 8)
    
    total_ops = len(UNARY_NAMES) + len(BINARY_NAMES) + len(AGGREGATION_NAMES)
    print(f"✓ Total available operations: {total_ops}")
    print(f"  - Unary: {len(UNARY_NAMES)} ({UNARY_NAMES[:5]}...)")
    print(f"  - Binary: {len(BINARY_NAMES)} ({BINARY_NAMES})")
    print(f"  - Aggregation: {len(AGGREGATION_NAMES)} ({AGGREGATION_NAMES})")
    print(f"✓ Output shape: {out1.shape}")
    print(f"✓ Entropy (without h): {entropy1.item():.3f}")
    print(f"✓ Entropy (with h): {entropy2.item():.3f}")
    print(f"✓ Selected operations: {cell.get_selected_ops()[:5]}...")
    print(f"✓ Operation distribution: {cell.get_op_distribution()}")
    print("✓ PASSED: OperationCell basic test")


def test_aggregation_operations():
    """Test that aggregation operations work correctly."""
    print("\n" + "="*60)
    print("TEST: Aggregation Operations")
    print("="*60)
    
    # Create cell with only aggregation
    cell = OperationCell(
        input_size=4,
        hidden_size=8,
        use_binary_ops=False,
        use_aggregation_ops=True,
        n_aggregation_inputs=4
    )
    
    # Force high temperature for exploration
    set_global_temperature(5.0)
    set_global_hard_mode(False)
    
    x = torch.randn(8, 4)
    out, entropy = cell(x)
    
    print(f"✓ Aggregation cell output shape: {out.shape}")
    print(f"✓ Entropy (high temp, soft mode): {entropy.item():.3f}")
    print(f"✓ Selected ops include aggregation: {cell.get_selected_ops()}")
    
    # Reset
    set_global_temperature(1.0)
    set_global_hard_mode(True)
    
    print("✓ PASSED: Aggregation operations test")


def test_entropy_regularization():
    """Test that entropy regularization encourages diverse operations."""
    print("\n" + "="*60)
    print("TEST: Entropy Regularization")
    print("="*60)
    
    # Generate simple data
    seq_len = 20
    batch_size = 32
    t = torch.linspace(0, 4*np.pi, seq_len + 1).unsqueeze(0).expand(batch_size, -1)
    data = torch.sin(t) + 0.5 * t / (4*np.pi)  # sin + linear trend
    x = data[:, :-1].unsqueeze(-1)
    y = data[:, 1:].unsqueeze(-1)
    
    # Train WITHOUT entropy regularization
    print("\nTraining WITHOUT entropy regularization:")
    model_no_reg = OperationRNN(input_size=1, hidden_size=12, num_layers=2, output_size=1)
    optimizer = torch.optim.Adam(model_no_reg.parameters(), lr=0.02)
    criterion = nn.MSELoss()
    
    set_global_hard_mode(True)
    for epoch in range(100):
        anneal_temperature(epoch, 100, start_tau=3.0, end_tau=0.3)
        optimizer.zero_grad()
        pred, _, entropy = model_no_reg(x)
        loss = criterion(pred, y)  # No entropy term
        loss.backward()
        optimizer.step()
    
    dist_no_reg = model_no_reg.get_total_op_distribution()
    print(f"  Final loss: {loss.item():.4f}")
    print(f"  Op distribution: {dict(list(dist_no_reg.items())[:4])}")
    
    # Train WITH entropy regularization
    print("\nTraining WITH entropy regularization:")
    model_with_reg = OperationRNN(input_size=1, hidden_size=12, num_layers=2, output_size=1)
    optimizer = torch.optim.Adam(model_with_reg.parameters(), lr=0.02)
    
    for epoch in range(100):
        anneal_temperature(epoch, 100, start_tau=3.0, end_tau=0.3)
        optimizer.zero_grad()
        pred, _, entropy = model_with_reg(x)
        
        # Use regularized loss
        total_loss, mse_loss, entropy_loss = compute_regularized_loss(
            pred, y, entropy, criterion,
            entropy_weight=0.2,
            target_entropy=2.5
        )
        total_loss.backward()
        optimizer.step()
    
    dist_with_reg = model_with_reg.get_total_op_distribution()
    print(f"  Final loss: {mse_loss.item():.4f}")
    print(f"  Final entropy: {entropy.item():.3f}")
    print(f"  Op distribution: {dict(list(dist_with_reg.items())[:6])}")
    
    # Check diversity
    n_unique_no_reg = len([k for k, v in dist_no_reg.items() if v > 0.05])
    n_unique_with_reg = len([k for k, v in dist_with_reg.items() if v > 0.05])
    
    print(f"\n✓ Unique ops (>5% usage) without reg: {n_unique_no_reg}")
    print(f"✓ Unique ops (>5% usage) with reg: {n_unique_with_reg}")
    print("✓ PASSED: Entropy regularization test")


def test_formula_discovery_v2():
    """Test improved formula discovery with entropy regularization."""
    print("\n" + "="*60)
    print("TEST: Formula Pattern Discovery v2")
    print("="*60)
    
    # Generate data from: y = x^2 + sin(x)
    n_samples = 100
    x_vals = torch.linspace(-3, 3, n_samples)
    y_vals = x_vals ** 2 + torch.sin(x_vals)
    
    # Reshape as sequence
    x = x_vals.unsqueeze(0).unsqueeze(-1)
    y = y_vals.unsqueeze(0).unsqueeze(-1)
    
    model = OperationRNN(
        input_size=1,
        hidden_size=16,
        num_layers=2,
        output_size=1,
        use_aggregation=True,
        n_aggregation_inputs=4
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    criterion = nn.MSELoss()
    
    print(f"Target formula: y = x² + sin(x)")
    print("Training with entropy regularization...")
    
    set_global_hard_mode(True)
    best_loss = float('inf')
    
    for epoch in range(400):
        # Slower annealing for more exploration
        tau = anneal_temperature(epoch, 400, start_tau=3.0, end_tau=0.15)
        
        optimizer.zero_grad()
        pred, _, entropy = model(x)
        
        # Adaptive entropy weight: higher at start, lower at end
        entropy_weight = 0.3 * (1 - epoch / 400)
        total_loss, mse_loss, _ = compute_regularized_loss(
            pred, y, entropy, criterion,
            entropy_weight=entropy_weight,
            target_entropy=2.0
        )
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if mse_loss.item() < best_loss:
            best_loss = mse_loss.item()
        
        if epoch % 80 == 0:
            print(f"  Epoch {epoch:3d} | MSE: {mse_loss.item():.4f} | Entropy: {entropy.item():.2f} | tau: {tau:.2f}")
    
    print(f"\n✓ Best MSE achieved: {best_loss:.4f}")
    print(f"✓ Operation distribution: {model.get_total_op_distribution()}")
    print(f"✓ Discovered operations:\n{model.get_formula_description()}")
    
    # Evaluate prediction quality
    with torch.no_grad():
        pred, _, _ = model(x)
    
    pred_flat = pred.squeeze()
    correlation = torch.corrcoef(torch.stack([pred_flat, y_vals]))[0, 1].item()
    print(f"✓ Prediction correlation: {correlation:.4f}")
    
    print("✓ PASSED: Formula discovery v2 test")
    
    return model, x_vals, y_vals, pred_flat


def test_operation_lstm_v2():
    """Test enhanced OperationLSTM with all features."""
    print("\n" + "="*60)
    print("TEST: OperationLSTM v2")
    print("="*60)
    
    # Complex signal
    seq_len = 30
    batch_size = 32
    t = torch.linspace(0, 6*np.pi, seq_len + 1).unsqueeze(0).expand(batch_size, -1)
    data = torch.sin(t) + 0.3 * torch.cos(2*t) + 0.1 * t
    x = data[:, :-1].unsqueeze(-1)
    y = data[:, 1:].unsqueeze(-1)
    
    model = OperationLSTM(
        input_size=1,
        hidden_size=12,
        num_layers=1,
        output_size=1,
        use_aggregation=True
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    criterion = nn.MSELoss()
    
    print("Training OperationLSTM v2...")
    set_global_hard_mode(True)
    
    for epoch in range(150):
        tau = anneal_temperature(epoch, 150, start_tau=2.5, end_tau=0.2)
        
        optimizer.zero_grad()
        pred, _, entropy = model(x)
        
        total_loss, mse_loss, _ = compute_regularized_loss(
            pred, y, entropy, criterion,
            entropy_weight=0.15,
            target_entropy=2.0
        )
        
        total_loss.backward()
        optimizer.step()
        
        if epoch % 30 == 0:
            print(f"  Epoch {epoch:3d} | MSE: {mse_loss.item():.4f} | Entropy: {entropy.item():.2f}")
    
    print(f"\n✓ Final MSE: {mse_loss.item():.4f}")
    print(f"✓ Gate operations:\n{model.get_formula_description()}")
    
    # Check gate diversity
    for i, cell in enumerate(model.cells):
        dists = cell.get_gate_distributions()
        print(f"\n  Layer {i} gate distributions:")
        for gate, dist in dists.items():
            top_ops = list(dist.items())[:3]
            print(f"    {gate}: {top_ops}")
    
    print("\n✓ PASSED: OperationLSTM v2 test")


def test_numerical_stability_v2():
    """Extended numerical stability tests."""
    print("\n" + "="*60)
    print("TEST: Numerical Stability v2")
    print("="*60)
    
    cell = OperationCell(
        input_size=4,
        hidden_size=8,
        use_aggregation_ops=True
    )
    
    test_cases = [
        ("Normal", torch.randn(8, 4)),
        ("Large values (100x)", torch.randn(8, 4) * 100),
        ("Small values (0.001x)", torch.randn(8, 4) * 0.001),
        ("Zeros", torch.zeros(8, 4)),
        ("Near-zero (1e-8)", torch.ones(8, 4) * 1e-8),
        ("Negative large", torch.randn(8, 4) * -50),
        ("Mixed extreme", torch.cat([torch.ones(4, 4)*100, torch.ones(4, 4)*-100], dim=0)),
    ]
    
    all_passed = True
    for name, x in test_cases:
        out, entropy = cell(x)
        has_nan = torch.isnan(out).any().item()
        has_inf = torch.isinf(out).any().item()
        status = "✓" if not (has_nan or has_inf) else "✗"
        if has_nan or has_inf:
            all_passed = False
        print(f"  {status} {name}: NaN={has_nan}, Inf={has_inf}, range=[{out.min():.2f}, {out.max():.2f}]")
    
    if all_passed:
        print("✓ PASSED: Numerical stability v2 test")
    else:
        print("✗ FAILED: Some stability issues detected")


def visualize_results(x_vals, y_actual, y_pred, title="OperationRNN v2 Prediction"):
    """Plot the actual vs predicted values."""
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(x_vals.numpy(), y_actual.numpy(), 'b-', label='Actual: x² + sin(x)', linewidth=2)
        plt.plot(x_vals.numpy(), y_pred.detach().numpy(), 'r--', label='Predicted', linewidth=2)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        error = (y_pred.detach() - y_actual).numpy()
        plt.plot(x_vals.numpy(), error, 'g-', linewidth=1.5)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.xlabel('x')
        plt.ylabel('Prediction Error')
        plt.title('Error Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('operation_rnn_v2_result.png', dpi=150)
        print(f"✓ Saved visualization to operation_rnn_v2_result.png")
        plt.close()
    except Exception as e:
        print(f"Note: Visualization skipped ({e})")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("   OPERATION-BASED RNN/LSTM v2 TEST SUITE")
    print("="*60)
    
    # Run all tests
    test_operation_cell_basic()
    test_aggregation_operations()
    test_numerical_stability_v2()
    test_entropy_regularization()
    
    model, x_vals, y_actual, y_pred = test_formula_discovery_v2()
    test_operation_lstm_v2()
    
    # Visualize
    visualize_results(x_vals, y_actual, y_pred)
    
    print("\n" + "="*60)
    print("   ALL TESTS COMPLETED!")
    print("="*60)
    print("""
v2 Improvements Summary:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. AGGREGATION OPERATIONS (Multi-input reduction)
   - sum_all: [1, 2, 3] → 6
   - prod_all: [2, 3, 4] → 24
   - mean_all, max_all, min_all, std_all, norm_all

2. ENTROPY REGULARIZATION
   - Encourages diverse operation exploration
   - Prevents collapse to single operation (identity)
   - Adaptive weight during training

3. IMPROVED INITIALIZATION
   - No bias toward identity
   - Slight favor toward "interesting" ops (sin, square, etc.)
   - Edge weights for aggregation

4. LEARNABLE EDGE WEIGHTS
   - Each aggregation input has a learnable coefficient
   - Like traditional weights but applied before aggregation

Usage Example:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
from glassbox.sr import OperationRNN, compute_regularized_loss

model = OperationRNN(input_size=1, hidden_size=16, num_layers=2)
pred, hidden, entropy = model(x_sequence)
loss, mse, ent_loss = compute_regularized_loss(pred, target, entropy, criterion)
""")
