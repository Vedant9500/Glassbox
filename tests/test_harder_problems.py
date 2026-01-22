import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from glassbox.sr.evolution import EvolutionaryONNTrainer as EvolutionEngine

def run_test(name, input_names, X_dict, y_target, generations=100):
    print(f"\n[Test] {name}")
    print(f"Target Formula hidden in data...")
    
    engine = EvolutionEngine(input_names=input_names, population_size=500, mutation_rate=0.4)
    engine.initialize_population()
    
    converged = False
    for gen in range(generations):
        loss, best_genome = engine.evolve_step(X_dict, y_target)
        
        if gen % 10 == 0:
            print(f"  Gen {gen}: Loss = {loss:.5f}")
            
        if loss < 0.05: # Stricter threshold check
            print(f"  Converged at Gen {gen}!")
            print(f"  Final Loss: {loss:.5f}")
            print(f"  Structure:\n{best_genome}")
            converged = True
            break
    
    if not converged:
        print(f"  Failed to converge. Best Loss: {loss:.5f}")
        print(f"  Best Genome:\n{best_genome}")

def test_kinetic_energy():
    # K = 0.5 * m * v^2
    m = np.random.uniform(1, 10, 50)
    v = np.random.uniform(0, 20, 50)
    y = 0.5 * m * (v**2)
    run_test("Kinetic Energy (0.5 * m * v^2)", ['m', 'v'], {'m': m, 'v': v}, y)

def test_trig_combo():
    # y = sin(x) + cos(x)
    x = np.linspace(-np.pi, np.pi, 50)
    y = np.sin(x) + np.cos(x)
    run_test("Sin(x) + Cos(x)", ['x'], {'x': x}, y)

def test_cubic():
    # y = x^3 - x
    x = np.linspace(-2, 2, 50)
    y = x**3 - x
    run_test("Cubic (x^3 - x)", ['x'], {'x': x}, y)

if __name__ == "__main__":
    print("Running Harder Problems Stress Test...")
    test_kinetic_energy()
    test_trig_combo()
    test_cubic()
