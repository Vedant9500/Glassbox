import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from glassbox.core.operations import Add, Mul, Variable
from glassbox.core.genome import Genome, Node

def test_genome_manual():
    print("Testing Manual Genome Construction for y = 2 * x + 10...")
    
    # Formula: y = (x * 2) + 10
    # Inputs: x (index 0)
    # Node 1: Constant 2 (index 1) <- actually let's use Constant Op
    # Node 2: Mul(0, 1) -> 2x (index 2)
    # Node 3: Constant 10 (index 3)
    # Node 4: Add(2, 3) -> 2x+10 (index 4)
    
    from glassbox.core.operations import Constant
    
    g = Genome(['x'])
    # g.nodes[0] is Variable(x)
    
    g.add_node(Constant(2.0), []) # Index 1
    g.add_node(Mul(), [0, 1])      # Index 2: x * 2
    g.add_node(Constant(10.0), [])# Index 3
    g.add_node(Add(), [2, 3])      # Index 4: (x*2) + 10
    
    # Evaluate
    x_val = np.array([1, 2, 3, 4, 5])
    y_target = x_val * 2 + 10
    
    y_pred = g.evaluate({'x': x_val})
    
    print(f"Input: {x_val}")
    print(f"Target: {y_target}")
    print(f"Pred:   {y_pred}")
    
    assert np.allclose(y_pred, y_target), "Forward pass failed!"
    print("Success!")

if __name__ == "__main__":
    test_genome_manual()
