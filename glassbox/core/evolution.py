import numpy as np
import random
import copy
from .genome import Genome, Node
from .operations import PRIMITIVES, Constant, Add, Sub, Mul, Div, Sin, Cos

class EvolutionEngine:
    def __init__(self, input_names, population_size=100, mutation_rate=0.2):
        self.input_names = input_names
        self.pop_size = population_size
        self.mutation_rate = mutation_rate
        self.population = []
        self.generation = 0

    def initialize_population(self):
        """Create initial random population."""
        self.population = []
        for _ in range(self.pop_size):
            genome = Genome(self.input_names)
            # Start with a small random graph (e.g., 1-3 ops)
            n_ops = random.randint(1, 3)
            for _ in range(n_ops):
                self._add_random_node(genome)
            self.population.append(genome)
    
    def _add_random_node(self, genome):
        """Adds a random valid node to the genome."""
        op_class = random.choice(PRIMITIVES + [Constant])
        
        if op_class == Constant:
            # Create a constant node
            val = np.random.uniform(-10, 10)
            # Constants have 0 inputs
            genome.add_node(Constant(val), [])
        else:
            # Create an operation node
            op = op_class()
            # Pick inputs from existing nodes (0 to len(genome.nodes)-1)
            # We need 'arity' inputs.
            # Available indices: 0 ... len(genome.nodes)-1
            valid_indices = list(range(len(genome.nodes)))
            
            if not valid_indices: 
                # Should not happen as we init with inputs
                return

            inputs = []
            for _ in range(op.arity):
                inputs.append(random.choice(valid_indices))
            
            genome.add_node(op, inputs)

    def mutate(self, genome):
        """
        Apply random mutations to a genome.
        Types of mutation:
        1. Add Node
        2. Change Op (of existing node)
        3. Change Input (of existing node)
        """
        if random.random() < 0.5:
            # Add a new node (grows the graph)
            self._add_random_node(genome)
        else:
            # Modify an existing node (if any exist beyond inputs)
            bias = len(self.input_names)
            if len(genome.nodes) > bias:
                idx = random.randint(bias, len(genome.nodes) - 1)
                node = genome.nodes[idx]
                
                # Flip: Change Op or Change Inputs or Mutate Constant
                mutation_type = random.random()
                
                if mutation_type < 0.33:
                    # Type A: Change Op
                    # Try to pick op with same arity to keep inputs valid,
                    # otherwise we must resample inputs
                    new_op_class = random.choice(PRIMITIVES)
                    if new_op_class.arity == node.op.arity:
                        node.op = new_op_class()
                    else:
                        # Arity mismatch, need to fix inputs
                        node.op = new_op_class()
                        # Resample inputs
                        valid_indices = list(range(idx)) # Inputs must be before this node
                        inputs = []
                        for _ in range(node.op.arity):
                            inputs.append(random.choice(valid_indices))
                        node.inputs = inputs

                elif mutation_type < 0.66:
                    # Type B: Change Inputs
                    # Resample one input
                    if node.op.arity > 0:
                        input_slot = random.randint(0, node.op.arity - 1)
                        valid_indices = list(range(idx))
                        node.inputs[input_slot] = random.choice(valid_indices)
                        
                else:
                    # Type C: Mutate Constant Value (if it is a Constant)
                    if isinstance(node.op, Constant):
                        # Hill climbing: add small random noise
                        # perturb by gaussian noise or uniform
                        delta = np.random.normal(0, 0.5) 
                        node.op.value += delta
                    else:
                        # If not a constant, fallback to changing inputs or op
                        # Just do nothing effectively (small no-op probability) or recurse?
                        # Let's just default to changing inputs to be safe
                        if node.op.arity > 0:
                            input_slot = random.randint(0, node.op.arity - 1)
                            valid_indices = list(range(idx))
                            node.inputs[input_slot] = random.choice(valid_indices)

    def evolve_step(self, X, y_target):
        """
        Run one generation of evolution.
        X: dict of input arrays
        y_target: target output array
        """
        # 1. Evaluate Fitness
        scored_pop = []
        for genome in self.population:
            try:
                y_pred = genome.evaluate(X)
                # MSE Loss
                loss = np.mean((y_pred - y_target)**2)
                # Handle NaNs or Infs
                if np.isnan(loss) or np.isinf(loss):
                    loss = float('inf')
            except Exception:
                loss = float('inf')
            
            scored_pop.append((loss, genome))
        
        # Sort by loss (asc)
        scored_pop.sort(key=lambda x: x[0])
        self.best_loss = scored_pop[0][0]
        self.best_genome = scored_pop[0][1]
        
        # 2. Selection (Elitism + Tournament/Random)
        # Keep top 10%
        n_elites = max(1, int(self.pop_size * 0.1))
        next_gen = [copy.deepcopy(g) for _, g in scored_pop[:n_elites]]
        
        # Fill the rest
        while len(next_gen) < self.pop_size:
            # Tournament selection
            parent = self._tournament_select(scored_pop)
            child = copy.deepcopy(parent)
            self.mutate(child)
            next_gen.append(child)
            
        self.population = next_gen
        self.generation += 1
        
        return self.best_loss, self.best_genome

    def _tournament_select(self, scored_pop, k=3):
        cutoff = int(len(scored_pop) * 0.5) # restrict to top 50% to spur convergence
        candidates = random.sample(scored_pop[:cutoff], k)
        # Return genome with lowest loss
        return min(candidates, key=lambda x: x[0])[1]
