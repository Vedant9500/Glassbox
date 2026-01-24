"""
Test sin(x) + x² formula discovery specifically.
Diagnoses whether the issue is operation selection or output weighting.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn


def diagnose_model(model, x, y):
    """Diagnose what each node is computing and how output weights combine them."""
    model.eval()
    
    print("\n--- MODEL DIAGNOSIS ---")
    
    # Get node info
    if hasattr(model, 'layers'):
        for layer_idx, layer in enumerate(model.layers):
            print(f"\nLayer {layer_idx}:")
            for node_idx, node in enumerate(layer.nodes):
                # Get operation selection
                if hasattr(node, 'op_selector'):
                    sel = node.op_selector.get_selected()
                    print(f"  Node {node_idx}: {sel['type']}, unary_idx={sel['unary_idx']}, binary_idx={sel['binary_idx']}")
    
    # Get output projection weights
    if hasattr(model, 'output_proj'):
        weights = model.output_proj.weight.data[0].cpu()
        bias = model.output_proj.bias.data[0].item() if model.output_proj.bias is not None else 0
        
        print(f"\nOutput projection weights: {weights.tolist()[:8]}...")  # First 8
        print(f"Output projection bias: {bias:.4f}")
        
        # Show which weights are significant
        significant = [(i, w.item()) for i, w in enumerate(weights) if abs(w.item()) > 0.1]
        significant.sort(key=lambda x: abs(x[1]), reverse=True)
        print(f"Significant weights (>0.1): {significant[:5]}")
    
    # Forward pass to see intermediate values
    with torch.no_grad():
        if hasattr(model, 'layers'):
            sources = x.clone()
            for layer_idx, layer in enumerate(model.layers):
                layer_output, layer_info = layer(sources, hard=True)
                print(f"\nLayer {layer_idx} node outputs (sample of first row):")
                for i in range(min(4, layer_output.shape[1])):
                    out_val = layer_output[0, i].item()
                    print(f"  Node {i}: {out_val:.4f}")
                sources = torch.cat([sources, layer_output], dim=-1)
    
    print("--- END DIAGNOSIS ---\n")


def test_sin_x2():
    """Test sin(x) + x² formula specifically."""
    
    from glassbox.sr import OperationDAG, get_device
    from glassbox.sr.evolution import train_onn_evolutionary
    from glassbox.sr import generate_polynomial_data, BaselineMLP
    
    device = get_device()
    print("\n" + "="*70)
    print("   sin(x) + x² FORMULA DISCOVERY TEST")
    print("="*70)
    print(f"Device: {device}")
    
    # Generate data for sin(x) + x²
    x, y, _ = generate_polynomial_data(n_samples=300, formula='sin+x^2', noise_std=0.02)
    
    # Shuffle 
    perm = torch.randperm(x.shape[0])
    x = x[perm]
    y = y[perm]
    
    n_train = 240
    x_train, x_val = x[:n_train], x[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    
    # Move to device
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_val = x_val.to(device)
    y_val = y_val.to(device)
    
    print(f"\nData range: x ∈ [{x.min():.2f}, {x.max():.2f}]")
    print(f"Target range: y ∈ [{y.min():.2f}, {y.max():.2f}]")
    print(f"Ground truth: y = sin(x) + x²")
    
    # Model factory - more nodes for better diversity
    def make_model():
        return OperationDAG(
            n_inputs=1,
            n_hidden_layers=1,   # Single layer
            nodes_per_layer=6,   # More nodes for diversity
            n_outputs=1,
            tau=0.5,
            simplified_ops=True,  # Only power, periodic, arithmetic
            fair_mode=True,       # FairDARTS: independent sigmoids
        )
    
    print("\nTraining configuration:")
    print("  - 1 hidden layer, 6 nodes")
    print("  - simplified_ops: True (power, periodic, arithmetic only)")
    print("  - fair_mode: True (FairDARTS independent sigmoids)")
    print("  - use_explorers: True (high-mutation scouts)")
    print("  - Higher mutation rate (0.5) for more exploration")
    
    # Evolutionary training with explorers
    result = train_onn_evolutionary(
        make_model,
        x_train, y_train,
        population_size=20,      # Main population
        generations=40,          # Enough generations
        device=device,
        fitness_x=x_val,
        fitness_y=y_val,
        normalize_data=False,
        constant_refine_hard=False,
        elite_fraction=0.2,      # Keep top 20%
        mutation_rate=0.4,       # Main population mutation
        prune_coefficients=False,  # No pruning
        use_explorers=True,        # Enable explorer subpopulation!
        explorer_fraction=0.25,    # 25% of pop size as explorers
        explorer_mutation_rate=0.85,  # Very high mutation for exploration
    )
    
    model = result['model']
    model.eval()
    
    # Diagnose the model
    diagnose_model(model, x_val, y_val)
    
    with torch.no_grad():
        pred, _ = model(x_val, hard=True)
        val_mse = nn.functional.mse_loss(pred, y_val).item()
        val_corr = torch.corrcoef(torch.stack([
            pred.squeeze().cpu(), y_val.squeeze().cpu()
        ]))[0, 1].item()
    
    print(f"\n" + "="*70)
    print("RESULTS:")
    print(f"  Formula discovered: {result['formula']}")
    print(f"  Validation MSE: {val_mse:.4f}")
    print(f"  Validation Corr: {val_corr:.4f}")
    
    # Compare with MLP
    print("\n--- MLP Baseline ---")
    mlp = BaselineMLP(n_inputs=1, n_hidden=32, n_outputs=1).to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01)
    
    for epoch in range(300):
        mlp.train()
        optimizer.zero_grad()
        pred = mlp(x_train)
        loss = nn.functional.mse_loss(pred, y_train)
        loss.backward()
        optimizer.step()
    
    mlp.eval()
    with torch.no_grad():
        pred = mlp(x_val)
        mlp_mse = nn.functional.mse_loss(pred, y_val).item()
        mlp_corr = torch.corrcoef(torch.stack([
            pred.squeeze().cpu(), y_val.squeeze().cpu()
        ]))[0, 1].item()
    
    print(f"  MLP MSE: {mlp_mse:.4f}")
    print(f"  MLP Corr: {mlp_corr:.4f}")
    
    print("\n" + "="*70)
    print("ANALYSIS:")
    formula = result['formula']
    has_sin = 'sin' in formula.lower()
    has_square = 'x0^2' in formula or '^2' in formula
    
    if has_sin and has_square:
        print("  ✓ Found BOTH sin and x² - SUCCESS!")
    elif has_sin:
        print("  ~ Found sin but not x²")
    elif has_square:
        print("  ~ Found x² but not sin")
    else:
        print("  ✗ Found neither sin nor x²")
    
    return result


if __name__ == '__main__':
    test_sin_x2()
