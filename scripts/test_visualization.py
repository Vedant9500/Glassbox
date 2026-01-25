"""
Test script for ONN Evolution with Real-time Visualization.

This demonstrates the visualization showing:
- Network architecture with operations
- Training progress curves
- Formula evolution
- Prediction vs target fit
"""

import torch
import sys
sys.path.insert(0, r'd:\Glassbox')

from glassbox.sr.operation_dag import OperationDAG
from glassbox.sr.evolution import EvolutionaryONNTrainer
from glassbox.sr.visualization import LiveTrainingVisualizer

def main(lite_mode: bool = False):
    print("="*60)
    print("ONN EVOLUTION WITH REAL-TIME VISUALIZATION")
    print("="*60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create test data: y = x^2
    print("\nTarget function: y = x²")
    x = torch.linspace(-3, 3, 100).reshape(-1, 1)
    y = x.pow(2)
    
    # Model factory
    def make_model():
        return OperationDAG(
            n_inputs=1,
            n_hidden_layers=2,
            nodes_per_layer=4,
            n_outputs=1,
            simplified_ops=True,  # Use smaller op menu
            fair_mode=True,        # FairDARTS
        )
    
    # Create visualizer
    print("\nInitializing visualizer...")
    if lite_mode:
        print("Using LITE mode (faster, no network diagram)")
    visualizer = LiveTrainingVisualizer(
        update_every=3,  # Update every 3 generations (less laggy)
        figsize=(14, 8),  # Slightly smaller for better performance
        lite_mode=lite_mode,
    )
    
    # Create trainer with visualizer
    trainer = EvolutionaryONNTrainer(
        model_factory=make_model,
        population_size=15,
        elite_size=3,
        mutation_rate=0.3,
        constant_refine_steps=50,
        complexity_penalty=0.01,
        device=device,
        lamarckian=True,
        use_explorers=True,
        constant_refine_hard=True,
        visualizer=visualizer,  # <-- Attach visualizer
    )
    
    # Train with visualization
    print("\nStarting evolution with visualization...")
    print("Watch the visualization window for real-time updates!")
    print("-"*60)
    
    results = trainer.train(
        x, y,
        generations=30,
        print_every=5,
    )
    
    # Final results
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    best_model = results['model']  # Key is 'model' not 'best_model'
    best_fitness = results['final_mse']
    
    print(f"\nBest MSE: {best_fitness:.6f}")
    
    if hasattr(best_model, 'get_formula'):
        formula = best_model.get_formula()
        print(f"Discovered formula: {formula}")
    
    # Evaluate final fit
    best_model.eval()
    with torch.no_grad():
        pred, _ = best_model(x.to(device), hard=True)
        pred = pred.cpu()
        mse = torch.nn.functional.mse_loss(pred.squeeze(), y.squeeze()).item()
        
        # Correlation
        pred_np = pred.squeeze().numpy()
        y_np = y.squeeze().numpy()
        import numpy as np
        corr = np.corrcoef(pred_np, y_np)[0, 1]
        
        print(f"Final MSE: {mse:.6f}")
        print(f"Final Correlation: {corr:.6f}")
    
    # Keep visualization open
    print("\nVisualization window will stay open. Close it to exit.")
    import matplotlib.pyplot as plt
    plt.show(block=True)


def test_multiple_functions():
    """Test visualization with multiple target functions."""
    import numpy as np
    import matplotlib.pyplot as plt
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_cases = [
        ("y = x²", lambda x: x.pow(2)),
        ("y = sin(x)", lambda x: torch.sin(x)),
        ("y = x³", lambda x: x.pow(3)),
    ]
    
    for name, func in test_cases:
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print("="*60)
        
        x = torch.linspace(-3, 3, 100).reshape(-1, 1)
        y = func(x)
        
        def make_model():
            return OperationDAG(
                n_inputs=1,
                n_hidden_layers=2,
                nodes_per_layer=4,
                n_outputs=1,
                simplified_ops=True,
                fair_mode=True,
            )
        
        visualizer = LiveTrainingVisualizer(update_every=1)
        
        trainer = EvolutionaryONNTrainer(
            model_factory=make_model,
            population_size=15,
            device=device,
            constant_refine_hard=True,
            visualizer=visualizer,
        )
        
        results = trainer.train(x, y, generations=20, print_every=5)
        
        best_model = results['best_model']
        if hasattr(best_model, 'get_formula'):
            print(f"Discovered: {best_model.get_formula()}")
        
        plt.show(block=False)
        plt.pause(2)  # Show for 2 seconds
        
        # Close visualizer
        visualizer.close()
    
    print("\nAll tests complete!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--multi', action='store_true', help='Test multiple functions')
    parser.add_argument('--lite', action='store_true', help='Use lite mode (no network diagram, faster)')
    args = parser.parse_args()
    
    if args.multi:
        test_multiple_functions()
    else:
        # Pass lite mode to main
        main(lite_mode=args.lite)
