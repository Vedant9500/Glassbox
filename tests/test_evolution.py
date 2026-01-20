import numpy as np
import sys
import os
import random

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from glassbox.core.evolution import EvolutionEngine

def test_evolution_simple():
    print("Testing Evolution for target: y = x * x ...")
    
    # 1. Generate Dataset
    X = np.linspace(-5, 5, 20)
    # Target: y = x^2
    y_target = X**2
    
    input_data = {'x': X}
    
    # 2. Initialize Engine
    engine = EvolutionEngine(input_names=['x'], population_size=200, mutation_rate=0.3)
    engine.initialize_population()
    
    # 3. Evolve
    print(f"Generations: 0, Best Loss: ?")
    
    for gen in range(50):
        loss, best_genome = engine.evolve_step(input_data, y_target)
        
        if gen % 5 == 0:
            print(f"Gen {gen}: Loss = {loss:.4f}")
            # print(f"Best Genome:\n{best_genome}")
            
        if loss < 0.01:
            print(f"Converged at Gen {gen}!")
            print(f"Final Loss: {loss}")
            print("Best Genome Structure:")
            print(best_genome)
            break
            
    # Check if we got close
    if loss < 0.1:
        print("SUCCESS: Evolution found a good approximation.")
    else:
        print("FAILURE: Evolution did not converge.")
        print("Best Genome Structure:")
        print(best_genome)

if __name__ == "__main__":
    # Seed for reproducibility (optional but good for testing)
    # random.seed(42)
    # np.random.seed(42)
    test_evolution_simple()
