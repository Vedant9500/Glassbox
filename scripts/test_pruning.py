"""
Test Post-Training Pruning

Demonstrates the full pruning pipeline:
1. Train ONN on target function
2. Run sensitivity analysis
3. Apply recursive graph pruning
4. Mask and fine-tune
5. Compare before/after formulas
"""

import torch
import numpy as np
import sys
sys.path.insert(0, 'd:/Glassbox')

from glassbox.sr import (
    OperationDAG,
    train_onn_evolutionary,
    EvolutionaryONNTrainer,
)
from glassbox.sr.pruning import PostTrainingPruner, prune_model


def test_pruning_pipeline():
    """Test full pruning pipeline on sin(x) + x^2"""
    
    print("="*70)
    print("POST-TRAINING PRUNING TEST")
    print("="*70)
    
    # Setup
    device = torch.device('cpu')
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate data: y = sin(x) + x^2
    # Use WIDE range to force learning the true structure, not just local approximation
    x = torch.linspace(-6, 6, 300).reshape(-1, 1)
    y = torch.sin(x) + x**2
    
    print(f"\nTarget function: y = sin(x) + x²")
    print(f"Data points: {len(x)}")
    
    # Create and train model
    print("\n" + "-"*70)
    print("PHASE 1: TRAINING")
    print("-"*70)
    
    # Create model factory
    def create_model():
        return OperationDAG(
            n_inputs=1,
            n_hidden_layers=2,
            nodes_per_layer=4,
            simplified_ops=True,
        ).to(device)
    
    # Quick training
    trainer = EvolutionaryONNTrainer(
        model_factory=create_model,
        population_size=15,
        mutation_rate=0.3,
        device=device,
    )
    
    result = trainer.train(x, y, generations=30, print_every=2)
    best_model = result['model']
    history = result['history']
    
    pre_prune_mse = result['final_mse']
    pre_prune_formula = best_model.get_formula()
    
    print(f"\nPre-pruning results:")
    print(f"  MSE: {pre_prune_mse:.6f}")
    print(f"  Formula: {pre_prune_formula}")
    
    # Run pruning pipeline
    print("\n" + "-"*70)
    print("PHASE 2: POST-TRAINING PRUNING")
    print("-"*70)
    
    pruner = PostTrainingPruner(best_model, x, y)
    
    # 1. Sensitivity Analysis
    print("\n>>> Running Sensitivity Analysis...")
    sensitivity = pruner.sensitivity_analysis(verbose=True)
    
    # 2. Recursive Graph Pruning
    print("\n>>> Running Recursive Graph Pruning...")
    nodes_pruned = pruner.recursive_graph_prune(importance_threshold=0.01, verbose=True)
    
    # 3. Symbolic Consolidation
    print("\n>>> Running Symbolic Consolidation...")
    merges = pruner.symbolic_consolidation(verbose=True)
    
    # 4. Mask and Fine-tune
    print("\n>>> Running Mask and Fine-tune...")
    mse_after_mask = pruner.mask_and_finetune(weight_threshold=0.05, verbose=True)
    
    # Final results
    post_prune_mse = pruner.get_mse()
    post_prune_formula = pruner.get_formula()
    
    print("\n" + "="*70)
    print("PRUNING RESULTS SUMMARY")
    print("="*70)
    print(f"\nBefore pruning:")
    print(f"  MSE: {pre_prune_mse:.6f}")
    print(f"  Formula: {pre_prune_formula}")
    print(f"\nAfter pruning:")
    print(f"  MSE: {post_prune_mse:.6f}")
    print(f"  Formula: {post_prune_formula}")
    
    improvement = (pre_prune_mse - post_prune_mse) / pre_prune_mse * 100 if pre_prune_mse > 0 else 0
    print(f"\nMSE change: {improvement:+.1f}%")
    
    # Count active terms
    import re
    pre_terms = len(re.findall(r'[+-]?\s*\d*\.?\d*\*?[a-z]', pre_prune_formula))
    post_terms = len(re.findall(r'[+-]?\s*\d*\.?\d*\*?[a-z]', post_prune_formula))
    print(f"Formula complexity: {pre_terms} terms -> {post_terms} terms")


def test_full_pipeline():
    """Test the convenience function"""
    
    print("\n" + "="*70)
    print("FULL PIPELINE TEST (Convenience Function)")
    print("="*70)
    
    device = torch.device('cpu')
    torch.manual_seed(123)
    
    # Target: y = 0.5*x^2 + 2*x + 1
    x = torch.linspace(-3, 3, 150).reshape(-1, 1)
    y = 0.5 * x**2 + 2*x + 1
    
    print(f"\nTarget: y = 0.5*x² + 2*x + 1")
    
    # Train model
    def create_model():
        return OperationDAG(
            n_inputs=1,
            n_hidden_layers=2,
            nodes_per_layer=4,
            simplified_ops=True,
        ).to(device)
    
    trainer = EvolutionaryONNTrainer(
        model_factory=create_model,
        population_size=15,
        device=device,
    )
    
    result = trainer.train(x, y, generations=15, print_every=5)
    best_model = result['model']
    
    pre_formula = best_model.get_formula()
    print(f"\nBefore pruning: {pre_formula}")
    
    # Use convenience function
    final_mse, final_formula = prune_model(
        best_model, x, y, 
        mse_tolerance=1.5, 
        verbose=True
    )
    
    print(f"\nFinal: {final_formula}")
    print(f"Final MSE: {final_mse:.6f}")


def test_iterative_backward():
    """Test iterative backward pruning"""
    
    print("\n" + "="*70)
    print("ITERATIVE BACKWARD PRUNING TEST")
    print("="*70)
    
    device = torch.device('cpu')
    torch.manual_seed(456)
    
    # Target: simple quadratic
    x = torch.linspace(-2, 2, 100).reshape(-1, 1)
    y = x**2
    
    print(f"\nTarget: y = x²")
    
    def create_model():
        return OperationDAG(
            n_inputs=1,
            n_hidden_layers=2,
            nodes_per_layer=6,  # More nodes to prune
            simplified_ops=True,
        ).to(device)
    
    trainer = EvolutionaryONNTrainer(
        model_factory=create_model,
        population_size=10,
        device=device,
    )
    
    result = trainer.train(x, y, generations=15, print_every=5)
    best_model = result['model']
    
    print(f"\nBefore: {best_model.get_formula()}")
    
    pruner = PostTrainingPruner(best_model, x, y)
    
    # Run iterative backward pruning
    nodes_pruned = pruner.iterative_backward_prune(
        mse_tolerance=1.5,
        min_importance=0.1,
        verbose=True
    )
    
    print(f"\nAfter: {pruner.get_formula()}")
    print(f"Total nodes pruned: {nodes_pruned}")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("POST-TRAINING PRUNING TESTS")
    print("="*70)
    
    # Run tests
    test_pruning_pipeline()
    
    print("\n" + "="*70 + "\n")
    
    test_full_pipeline()
    
    print("\n" + "="*70 + "\n")
    
    test_iterative_backward()
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETE")
    print("="*70)
