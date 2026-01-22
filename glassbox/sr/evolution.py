"""
Evolutionary Training for Operation-Based Neural Networks.

The PROPER approach: Evolution for discrete structure, gradient for constants.

Key insight from research:
- Gradient descent CANNOT effectively select discrete operations
- Evolution is better for structural search (which operation)
- Gradient descent is better for continuous optimization (what parameters)

This implements the "EmbedGrad-Evolution" strategy:
1. Population of individuals with different operation structures
2. Fitness = actual prediction error (not training loss)
3. Discrete mutations on operation selection
4. Gradient refinement on constants only
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, LBFGS
from typing import Optional, Dict, List, Tuple, Callable
import copy
import random
import math
import time


class Individual:
    """
    An individual in the evolutionary population.
    
    Contains a model with specific operation configuration.
    """
    
    def __init__(
        self,
        model: nn.Module,
        fitness: float = float('inf'),
        generation: int = 0,
    ):
        self.model = model
        self.fitness = fitness
        self.generation = generation
        self.structure_hash = self._compute_structure_hash()
    
    def _compute_structure_hash(self) -> str:
        """Hash the operation selection for diversity tracking."""
        ops = []
        for name, param in self.model.named_parameters():
            if 'logit' in name or 'selector' in name:
                ops.append(param.argmax().item())
        return str(ops)
    
    def clone(self) -> 'Individual':
        """Deep copy this individual."""
        new_model = copy.deepcopy(self.model)
        return Individual(
            model=new_model,
            fitness=self.fitness,
            generation=self.generation + 1,
        )


def random_operation_init(model: nn.Module, bias_strength: float = 2.0):
    """
    Initialize with RANDOM operation selections.
    
    Unlike identity init, this explores the full operation space.
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'logit' in name or 'selector' in name:
                # Strong random bias toward one operation
                param.normal_(0, bias_strength)
            elif 'p' in name and 'output' not in name:
                # Random power: 0.5 to 3.0
                param.fill_(random.uniform(0.5, 3.0))
            elif 'omega' in name:
                # Random frequency: 0.5 to 2.0
                param.fill_(random.uniform(0.5, 2.0))
            elif 'beta' in name and 'selector' not in name:
                # Random arithmetic blend
                param.fill_(random.uniform(0.5, 2.5))
            elif 'R' in name:
                # Random routing
                param.normal_(0, 1.0)


def mutate_operations(individual: Individual, mutation_rate: float = 0.3) -> Individual:
    """
    Mutate operation selections DISCRETELY.
    
    This is the key difference from gradient-based training:
    - We randomly change which operation is selected
    - Not small continuous nudges, but discrete swaps
    """
    mutant = individual.clone()
    
    with torch.no_grad():
        for name, param in mutant.model.named_parameters():
            if random.random() < mutation_rate:
                if 'logit' in name or 'selector' in name:
                    # DISCRETE mutation: completely re-roll the selection
                    if random.random() < 0.5:
                        # Option 1: Shift to a different option
                        current_best = param.argmax()
                        param.zero_()
                        new_choice = random.randint(0, param.numel() - 1)
                        param.view(-1)[new_choice] = 3.0  # Strong bias to new choice
                    else:
                        # Option 2: Add noise to current selection
                        param.add_(torch.randn_like(param) * 1.0)
                
                elif 'p' in name and 'output' not in name:
                    # Mutate power parameter
                    if random.random() < 0.3:
                        # Discrete jump to common powers
                        param.fill_(random.choice([0.5, 1.0, 2.0, 3.0, -1.0]))
                    else:
                        param.add_(random.uniform(-0.5, 0.5))
                        param.clamp_(0.1, 4.0)
                
                elif 'omega' in name:
                    # Mutate frequency
                    param.mul_(random.uniform(0.8, 1.2))
                    param.clamp_(0.1, 5.0)
                
                elif 'R' in name:
                    # Mutate routing
                    if random.random() < 0.3:
                        # Swap routing connections
                        param.add_(torch.randn_like(param) * 0.5)
    
    return mutant


def refine_constants(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    steps: int = 50,
    lr: float = 0.01,
) -> float:
    """
    Use gradient descent to optimize ONLY the constants.
    
    Lock operation selection, only tune parameters.
    """
    import gc
    
    model.train()
    
    # Identify constant parameters (not selection logits)
    constant_params = []
    for name, param in model.named_parameters():
        if 'logit' not in name and 'selector' not in name:
            if param.requires_grad:
                constant_params.append(param)
    
    if not constant_params:
        return float('inf')
    
    try:
        optimizer = Adam(constant_params, lr=lr)
    except (MemoryError, RuntimeError):
        # Clear memory and try again
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            optimizer = Adam(constant_params, lr=lr)
        except Exception:
            return float('inf')
    
    best_loss = float('inf')
    
    for step in range(steps):
        optimizer.zero_grad()
        
        try:
            pred, _ = model(x, hard=True)
            loss = F.mse_loss(pred, y)
            
            if torch.isnan(loss):
                del optimizer
                gc.collect()
                return float('inf')
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(constant_params, max_norm=1.0)
            optimizer.step()
            
            best_loss = min(best_loss, loss.item())
        except Exception:
            del optimizer
            gc.collect()
            return float('inf')
    
    # Cleanup
    del optimizer
    gc.collect()
    
    return best_loss


def calculate_complexity(model: nn.Module) -> float:
    """
    Calculate complexity score for a model.
    Higher score = more complex operations.
    """
    complexity = 0.0
    
    if hasattr(model, 'layers'):
        for layer in model.layers:
            for node in layer.nodes:
                op_str = node.get_selected_operation().lower()
                
                # High cost: Transcendental functions
                if 'exp' in op_str or 'log' in op_str or 'sin' in op_str or 'cos' in op_str:
                    complexity += 2.0
                # Medium cost: Powers and roots
                elif 'sqrt' in op_str or 'pow' in op_str or '^' in op_str:
                    complexity += 1.5
                # Low cost: Arithmetic and identity
                elif 'identity' in op_str:
                    complexity += 0.1  # Very low cost for pass-through
                else:
                    complexity += 1.0
                    
    return complexity


class EvolutionaryONNTrainer:
    """
    Proper evolutionary training for ONN.
    
    Algorithm:
    1. Initialize population with RANDOM operations
    2. Evaluate each individual's fitness
    3. Select top performers
    4. Create offspring via discrete mutation
    5. Refine constants with gradient descent
    6. Repeat
    """
    
    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        population_size: int = 20,
        elite_size: int = 5,
        mutation_rate: float = 0.3,
        constant_refine_steps: int = 30,
        complexity_penalty: float = 0.02,
        device: Optional[torch.device] = None,
    ):
        self.model_factory = model_factory
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.constant_refine_steps = constant_refine_steps
        self.complexity_penalty = complexity_penalty
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.population: List[Individual] = []
        self.generation = 0
        self.best_ever: Optional[Individual] = None
        self.history = []
    
    def initialize_population(self):
        """Create diverse initial population."""
        self.population = []
        
        for i in range(self.population_size):
            model = self.model_factory().to(self.device)
            random_operation_init(model, bias_strength=2.0 + i * 0.1)
            individual = Individual(model, generation=0)
            self.population.append(individual)
        
        print(f"Initialized population of {self.population_size} individuals")
    
    def evaluate_fitness(self, x: torch.Tensor, y: torch.Tensor):
        """Evaluate fitness of all individuals."""
        x = x.to(self.device)
        y = y.to(self.device)
        
        for ind in self.population:
            ind.model.eval()
            try:
                with torch.no_grad():
                    pred, _ = ind.model(x, hard=True)
                    mse = F.mse_loss(pred, y).item()
                    
                    if math.isnan(mse) or math.isinf(mse):
                        ind.fitness = float('inf')
                    else:
                        # Add complexity penalty (Parsimony)
                        complexity = calculate_complexity(ind.model)
                        ind.fitness = mse + self.complexity_penalty * complexity
            except Exception:
                ind.fitness = float('inf')
        
        # Track best ever
        best_current = min(self.population, key=lambda ind: ind.fitness)
        if self.best_ever is None or best_current.fitness < self.best_ever.fitness:
            self.best_ever = best_current.clone()
    
    def select_and_reproduce(self, x: torch.Tensor, y: torch.Tensor, diversity: int = 10):
        """Selection and reproduction."""
        x = x.to(self.device)
        y = y.to(self.device)
        
        # Sort by fitness
        sorted_pop = sorted(self.population, key=lambda ind: ind.fitness)
        
        new_population = []
        
        # Mass Mutation condition: Low diversity
        if diversity < 3:
            # print("(!) Low diversity detected. Triggering MASS MUTATION.")
            # Keep only the very best (Elite = 1 or 2)
            for elite in sorted_pop[:2]:
                new_population.append(elite)
            
            # Fill the rest with fresh random individuals + heavy mutation
            while len(new_population) < self.population_size:
                if random.random() < 0.5:
                    # Brand new random individual (Injection)
                    model = self.model_factory().to(self.device)
                    random_operation_init(model, bias_strength=3.0)
                    child = Individual(model, generation=self.generation)
                else:
                    # Heavily mutated elite (Exploration)
                    parent = sorted_pop[0] # Best one
                    child = mutate_operations(parent, mutation_rate=0.8) # High mutation rate
                
                # Fast refine
                refine_constants(
                    child.model, x, y,
                    steps=10, # Fewer steps for speed
                    lr=0.02,
                )
                new_population.append(child)
                
        else:
            # Normal Evolution
            
            # Keep elite unchanged
            for elite in sorted_pop[:self.elite_size]:
                new_population.append(elite)
            
            # Fill rest with mutations of top performers
            while len(new_population) < self.population_size:
                # Tournament selection
                candidates = random.sample(sorted_pop[:self.population_size//2], 
                                           min(3, len(sorted_pop)))
                parent = min(candidates, key=lambda ind: ind.fitness)
                
                # Mutate
                child = mutate_operations(parent, self.mutation_rate)
                
                # Refine constants
                refine_constants(
                    child.model, x, y,
                    steps=self.constant_refine_steps,
                    lr=0.02,
                )
                
                new_population.append(child)
        
        self.population = new_population
        self.generation += 1
    
    def train(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        generations: int = 50,
        print_every: int = 5,
    ) -> Dict:
        """
        Full evolutionary training.
        
        Args:
            x: Input data
            y: Target data
            generations: Number of generations
            print_every: Print frequency
            
        Returns:
            Training results
        """
        print("="*60)
        print("EVOLUTIONARY ONN TRAINING")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Population: {self.population_size}")
        print(f"Generations: {generations}")
        print("-"*60)
        
        x = x.to(self.device)
        y = y.to(self.device)
        
        # Initialize
        self.initialize_population()
        
        # Initial constant refinement for all
        print("Refining initial constants...")
        for ind in self.population:
            refine_constants(ind.model, x, y, steps=self.constant_refine_steps)
        
        start_time = time.time()
        
        for gen in range(generations):
            # Evaluate
            self.evaluate_fitness(x, y)
            
            # Stats
            fitnesses = [ind.fitness for ind in self.population if ind.fitness < float('inf')]
            if fitnesses:
                best_fit = min(fitnesses)
                mean_fit = sum(fitnesses) / len(fitnesses)
                diversity = len(set(ind.structure_hash for ind in self.population))
            else:
                best_fit = mean_fit = float('inf')
                diversity = 0
            
            self.history.append({
                'generation': gen,
                'best_fitness': best_fit,
                'mean_fitness': mean_fit,
                'diversity': diversity,
                'best_ever': self.best_ever.fitness if self.best_ever else float('inf'),
            })
            
            if gen % print_every == 0 or gen == generations - 1:
                print(f"Gen {gen:3d} | Best: {best_fit:.4f} | "
                      f"Mean: {mean_fit:.4f} | Diversity: {diversity} | "
                      f"Best Ever: {self.best_ever.fitness:.4f}")
            
            # Reproduce (skip on last generation)
            if gen < generations - 1:
                self.select_and_reproduce(x, y, diversity=diversity)
        
        elapsed = time.time() - start_time
        
        # Final evaluation
        print("-"*60)
        print(f"Training complete in {elapsed:.1f}s")
        
        # Get best model
        best_model = self.best_ever.model if self.best_ever else self.population[0].model
        best_model.eval()
        
        with torch.no_grad():
            pred, _ = best_model(x, hard=True)
            final_mse = F.mse_loss(pred, y).item()
            corr = torch.corrcoef(torch.stack([
                pred.squeeze().cpu(), y.squeeze().cpu()
            ]))[0, 1].item()
        
        # Get formula
        if hasattr(best_model, 'snap_to_discrete'):
            best_model.snap_to_discrete()
        
        formula = "N/A"
        if hasattr(best_model, 'get_formula'):
            formula = best_model.get_formula()
        elif hasattr(best_model, 'get_graph_summary'):
            formula = best_model.get_graph_summary()
        
        print(f"Final MSE: {final_mse:.4f}")
        print(f"Correlation: {corr:.4f}")
        print(f"Formula: {formula}")
        
        return {
            'model': best_model,
            'history': self.history,
            'final_mse': final_mse,
            'correlation': corr,
            'formula': formula,
            'training_time': elapsed,
        }


def train_onn_evolutionary(
    model_factory: Callable[[], nn.Module],
    x: torch.Tensor,
    y: torch.Tensor,
    population_size: int = 20,
    generations: int = 50,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Convenience function for evolutionary ONN training.
    
    Usage:
        from glassbox.sr import OperationDAG
        from glassbox.sr.evolution import train_onn_evolutionary
        
        def make_model():
            return OperationDAG(n_inputs=1, n_hidden_layers=2, nodes_per_layer=4)
        
        results = train_onn_evolutionary(make_model, x, y, generations=50)
        print(results['formula'])
    """
    trainer = EvolutionaryONNTrainer(
        model_factory=model_factory,
        population_size=population_size,
        device=device,
    )
    return trainer.train(x, y, generations=generations)
