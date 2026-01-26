"""
N-Queens Challenge: Can Glassbox discover a formula for Q(n)?

THE CATCH: There is NO known closed-form formula for the N-Queens sequence!
This is a famous unsolved problem in combinatorics.

The sequence grows roughly like n! but with no simple expression.
Let's see what approximation Glassbox comes up with!
"""

import torch
import numpy as np
import sys
sys.path.insert(0, r'd:\Glassbox')

from glassbox.sr.operation_dag import OperationDAG
from glassbox.sr.evolution import EvolutionaryONNTrainer
from glassbox.sr.visualization import LiveTrainingVisualizer


def main():
    print("="*70)
    print("N-QUEENS CHALLENGE: THE UNSOLVABLE FORMULA")
    print("="*70)
    print("\nThis is a FUN challenge - there is NO known closed-form formula!")
    print("The N-Queens sequence counts ways to place N queens on an NxN board")
    print("such that no two queens attack each other.\n")
    
    # N-Queens data - starting from N=7 for more consistent growth pattern
    # Including larger values up to N=27
    nqueens_data = [
        (7, 40),
        (8, 92),
        (9, 352),
        (10, 724),
        (11, 2680),
        (12, 14200),
        (13, 73712),
        (14, 365596),
        (15, 2279184),
        (16, 14772512),
        (17, 95815104),
        (18, 666090624),
        (19, 4968057848),
        (20, 39029188884),
        (21, 314666233184),
        (22, 2691697673360),
        (23, 24196498129568),
        (24, 229190929136000),
        (25, 2277351529880000),
        (26, 23592948014352000),
        (27, 257597713970050000),
    ]
    
    # Use log scale because values span many orders of magnitude
    print("Using LOG scale for y-values (spans 7 orders of magnitude!)")
    
    # Filter out zeros (can't take log of 0)
    nqueens_data = [(n, q) for n, q in nqueens_data if q > 0]
    
    x_data = torch.tensor([[n] for n, q in nqueens_data], dtype=torch.float32)
    y_data = torch.tensor([[np.log(q)] for n, q in nqueens_data], dtype=torch.float32)
    
    print(f"\nData points: N = {[n for n, q in nqueens_data]}")
    print(f"Solutions:   Q = {[q for n, q in nqueens_data]}")
    print(f"Log(Q):      {[f'{np.log(q):.2f}' for n, q in nqueens_data]}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Model factory
    def make_model():
        return OperationDAG(
            n_inputs=1,
            n_hidden_layers=2,  # More layers for complex pattern
            nodes_per_layer=8,  # More nodes
            n_outputs=1,
            simplified_ops=False,  # Allow exp, log for factorial-like growth
            fair_mode=True,
        )
    
    # Create visualizer
    print("\nInitializing visualizer...")
    visualizer = LiveTrainingVisualizer(
        update_every=5,
        figsize=(14, 8),
        lite_mode=False,
    )
    
    # Create trainer with high exploration
    trainer = EvolutionaryONNTrainer(
        model_factory=make_model,
        population_size=30,
        elite_size=5,
        mutation_rate=0.5,
        constant_refine_steps=50,
        complexity_penalty=0.005,  # Less penalty, allow complex formulas
        device=device,
        lamarckian=True,
        use_explorers=True,
        explorer_fraction=0.40,
        explorer_mutation_rate=0.9,
        prune_coefficients=False,
        constant_refine_hard=True,
        visualizer=visualizer,
    )
    
    print("\n" + "="*70)
    print("SEARCHING FOR A FORMULA (that doesn't exist!)")
    print("="*70)
    print("Goal: Find the best approximation to log(Q(n))")
    print("-"*70)
    
    # Train
    results = trainer.train(
        x_data, y_data,
        generations=100,  # More generations for this hard problem
        print_every=10,
    )
    
    phase1_model = results['model']
    phase1_formula = results['formula']
    phase1_mse = results['final_mse']
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Best discovered formula: {phase1_formula}")
    print(f"MSE on log(Q): {phase1_mse:.4f}")
    
    # Validate
    print(f"\n{'='*70}")
    print("VALIDATION")
    print(f"{'='*70}")
    print(f"\n{'N':>4} | {'True Q(N)':>12} | {'Predicted':>12} | {'Ratio':>8}")
    print("-" * 50)
    
    with torch.no_grad():
        for n, q_true in nqueens_data:
            x_test = torch.tensor([[float(n)]], dtype=torch.float32).to(device)
            output = phase1_model(x_test)
            # Model may return tuple (output, hidden_state)
            if isinstance(output, tuple):
                output = output[0]
            log_q_pred = output.item()
            q_pred = np.exp(log_q_pred)
            ratio = q_pred / q_true if q_true > 0 else float('inf')
            print(f"{n:>4} | {q_true:>12,} | {q_pred:>12,.0f} | {ratio:>8.2f}x")
    
    # The punchline
    print(f"\n{'='*70}")
    print("THE TRUTH")
    print(f"{'='*70}")
    print("There is NO known closed-form formula for the N-Queens sequence!")
    print("It's related to the permanent of a matrix - a #P-complete problem.")
    print("Even the best approximation will fail for larger N.")
    print("\nBut hey, we tried! 🎲👸")
    
    # Keep visualization open
    import matplotlib.pyplot as plt
    print("\nVisualization window will stay open. Close it to exit.")
    plt.show(block=True)


if __name__ == "__main__":
    main()
