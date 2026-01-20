import numpy as np
from .operations import PRIMITIVES, Constant, Variable

class Node:
    """A single node in the computational graph."""
    def __init__(self, op, inputs=None):
        self.op = op  # Instance of an Operation subclass
        self.inputs = inputs or []  # List of indices of input nodes
    
    def __repr__(self):
        if self.op.arity == 0:
            return f"{self.op}"
        return f"{self.op}({', '.join(map(str, self.inputs))})"

class Genome:
    """
    Represents an individual solution (a mathematical formula).
    Structure:
    - Inputs are virtual nodes at indices 0 to n_inputs-1.
    - Hidden/Output nodes are at indices n_inputs to end.
    - Feed-forward constraint: Node i can only take inputs from nodes j < i.
    """
    def __init__(self, input_names):
        self.input_names = input_names
        self.nodes = [] # List of Node objects (excluding pure inputs, or including? Let's include everything for simplicity)
        
        # Initialize input nodes
        for name in input_names:
            self.nodes.append(Node(Variable(name)))
            
    def add_node(self, op, input_indices):
        """Add a new operation node to the genome."""
        # Validate constraint: inputs must ideally be < current index (len(nodes))
        # But we will enforce this during construction/mutation
        self.nodes.append(Node(op, input_indices))

    def evaluate(self, input_values):
        """
        Evaluate the graph.
        input_values: dict {var_name: np.array/float}
        Returns: output of the last node (or specific output nodes)
        """
        # Cache results: index -> value
        # Since we enforce Feed-Forward (j < i), we can just iterate top-down.
        
        results = {}
        
        # 1. Set Input Values
        for i, name in enumerate(self.input_names):
            results[i] = input_values[name]
            
        # 2. Compute Hidden/Output Nodes
        # Nodes start after inputs
        start_idx = len(self.input_names)
        
        for i in range(start_idx, len(self.nodes)):
            node = self.nodes[i]
            
            # Gather input values from previous results
            args = []
            for input_idx in node.inputs:
                args.append(results[input_idx])
            
            # Execute Operation
            if node.op.arity == 0:
                # Constant
                results[i] = node.op() 
            else:
                # Binary/Unary Op
                results[i] = node.op(*args)
                
        # Return the value of the last node as the 'output' of the network
        if not self.nodes:
            return 0.0
        return results[len(self.nodes) - 1]

    def __repr__(self):
        lines = []
        for i, node in enumerate(self.nodes):
            lines.append(f"{i}: {node}")
        return "\n".join(lines)
