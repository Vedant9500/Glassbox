"""
Hybrid Optimizer for Operation-Based Neural Networks.

Implements the "EmbedGrad-Evolution" strategy from research:
1. L-BFGS for continuous parameters (edge constants, meta-op params)
2. Evolution for discrete topology (which operations, connections)
3. Gradient-guided mutation to bias evolution toward promising directions

Key insight: Pure gradient descent gets stuck in local minima.
             Pure evolution is too slow for constant fitting.
             Hybrid combines the best of both.

Research reference: docs/research.md Section 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, LBFGS
from typing import Optional, Dict, List, Tuple, Callable
import copy
import random
import math


class LBFGSConstantOptimizer:
    """
    Use L-BFGS to optimize continuous constants while keeping topology fixed.
    
    L-BFGS is significantly better than Adam for finding exact constants
    in symbolic regression (see research.md Section 5.2).
    
    Usage:
        optimizer = LBFGSConstantOptimizer(model)
        for epoch in range(n_epochs):
            loss = optimizer.step(x, y)
    """
    
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1.0,
        max_iter: int = 20,
        history_size: int = 10,
        line_search: str = 'strong_wolfe',
    ):
        """
        Args:
            model: ONN model
            lr: Learning rate (1.0 works well for L-BFGS)
            max_iter: Max iterations per step
            history_size: L-BFGS history size
            line_search: Line search method
        """
        self.model = model
        
        # Collect continuous parameters (constants, meta-op params)
        self.constant_params = self._get_constant_params()
        
        self.optimizer = LBFGS(
            self.constant_params,
            lr=lr,
            max_iter=max_iter,
            history_size=history_size,
            line_search_fn=line_search,
        )
        
        self.loss_fn = nn.MSELoss()
    
    def _get_constant_params(self) -> List[nn.Parameter]:
        """Get only the continuous constant parameters."""
        constant_params = []
        
        for name, param in self.model.named_parameters():
            # Include: edge weights, meta-op parameters (omega, phi, p, beta, etc.)
            # Exclude: selection logits (these are for topology)
            if any(key in name for key in [
                'edge_weights', 'omega', 'phi', 'amplitude', 'p', 'beta',
                'scale', 'constant', 'weights', 'log_base', 'rate',
                'output_scale', 'output_proj'
            ]):
                constant_params.append(param)
        
        return constant_params
    
    def step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        hard: bool = True,
    ) -> float:
        """
        One L-BFGS optimization step.
        
        Args:
            x: Input data (batch, n_features)
            y: Target data (batch, n_outputs)
            hard: Use hard selection in model
            
        Returns:
            Final loss value
        """
        self.model.train()
        
        def closure():
            self.optimizer.zero_grad()
            pred, _ = self.model(x, hard=hard)
            
            # Check for NaN in predictions
            if torch.isnan(pred).any():
                return torch.tensor(float('inf'))
            
            loss = self.loss_fn(pred, y)
            
            # Check for NaN in loss
            if torch.isnan(loss):
                return torch.tensor(float('inf'))
            
            loss.backward()
            
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            return loss
        
        try:
            loss = self.optimizer.step(closure)
            loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss
            
            # Return infinity if NaN
            if not math.isfinite(loss_val):
                return float('inf')
            
            return loss_val
        except RuntimeError:
            # L-BFGS can fail on some line searches
            return float('inf')



class Individual:
    """
    An individual in the evolutionary population.
    
    Contains:
    - A copy of the model (topology + parameters)
    - Fitness score
    - Lineage information
    """
    
    def __init__(
        self,
        model: nn.Module,
        fitness: float = float('inf'),
        generation: int = 0,
        parent_id: Optional[int] = None,
    ):
        self.model = model
        self.fitness = fitness
        self.generation = generation
        self.parent_id = parent_id
        self.id = id(self)  # Unique ID
    
    def clone(self) -> 'Individual':
        """Create a deep copy."""
        new_model = copy.deepcopy(self.model)
        return Individual(
            model=new_model,
            fitness=self.fitness,
            generation=self.generation + 1,
            parent_id=self.id,
        )


class EvolutionaryOptimizer:
    """
    Evolutionary search for operation selection (topology).
    
    Algorithm:
    1. Initialize population of random models
    2. For each generation:
       a. Evaluate fitness (loss on validation data)
       b. Select top performers
       c. Create offspring via mutation + crossover
       d. (Optional) Apply L-BFGS refinement to new individuals
    3. Return best individual
    """
    
    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        population_size: int = 20,
        elite_size: int = 5,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.3,
        use_lbfgs_refinement: bool = True,
        lbfgs_steps: int = 5,
    ):
        """
        Args:
            model_factory: Function that creates a new model instance
            population_size: Number of individuals in population
            elite_size: Number of top performers to keep
            mutation_rate: Probability of mutating each parameter
            crossover_rate: Probability of crossover between parents
            use_lbfgs_refinement: Apply L-BFGS after creating offspring
            lbfgs_steps: Number of L-BFGS steps for refinement
        """
        self.model_factory = model_factory
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.use_lbfgs_refinement = use_lbfgs_refinement
        self.lbfgs_steps = lbfgs_steps
        
        self.population: List[Individual] = []
        self.generation = 0
        self.best_ever: Optional[Individual] = None
        
        self.loss_fn = nn.MSELoss()
    
    def initialize_population(self):
        """Create initial random population."""
        self.population = []
        for _ in range(self.population_size):
            model = self.model_factory()
            self._randomize_topology(model)
            individual = Individual(model, generation=0)
            self.population.append(individual)
    
    def _randomize_topology(self, model: nn.Module):
        """Randomize the operation selection parameters."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'logit' in name or 'selector' in name:
                    # Random initialization for selection weights
                    param.normal_(mean=0, std=1.0)
    
    def evaluate_population(self, x: torch.Tensor, y: torch.Tensor):
        """Evaluate fitness of all individuals."""
        for individual in self.population:
            individual.model.eval()
            with torch.no_grad():
                pred, _ = individual.model(x, hard=True)
                loss = self.loss_fn(pred, y)
                individual.fitness = loss.item()
        
        # Update best ever
        best_current = min(self.population, key=lambda ind: ind.fitness)
        if self.best_ever is None or best_current.fitness < self.best_ever.fitness:
            self.best_ever = best_current.clone()
    
    def select_parents(self) -> List[Individual]:
        """Select parents for next generation using tournament selection."""
        parents = []
        
        # Keep elite
        sorted_pop = sorted(self.population, key=lambda ind: ind.fitness)
        elite = sorted_pop[:self.elite_size]
        parents.extend(elite)
        
        # Tournament selection for rest
        while len(parents) < self.population_size:
            # Tournament of 3
            candidates = random.sample(self.population, min(3, len(self.population)))
            winner = min(candidates, key=lambda ind: ind.fitness)
            parents.append(winner)
        
        return parents
    
    def mutate(self, individual: Individual) -> Individual:
        """
        Mutate an individual's topology.
        
        Mutations:
        - Perturb operation selection logits
        - Perturb routing logits
        - Add noise to constants
        """
        mutant = individual.clone()
        
        with torch.no_grad():
            for name, param in mutant.model.named_parameters():
                if random.random() < self.mutation_rate:
                    if 'logit' in name or 'selector' in name or 'op_' in name:
                        # Stronger mutation for topology
                        noise = torch.randn_like(param) * 0.5
                        param.add_(noise)
                    elif 'route' in name or 'R' in name:
                        # Routing mutation
                        noise = torch.randn_like(param) * 0.3
                        param.add_(noise)
                    else:
                        # Small noise for constants
                        noise = torch.randn_like(param) * 0.1
                        param.add_(noise)
        
        return mutant
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """
        Crossover between two parents.
        
        Strategy: Uniform crossover on layer level.
        Each layer randomly inherits from parent1 or parent2.
        """
        if random.random() > self.crossover_rate:
            return parent1.clone()
        
        child = parent1.clone()
        
        with torch.no_grad():
            # Collect layer parameters from both parents
            p1_params = dict(parent1.model.named_parameters())
            p2_params = dict(parent2.model.named_parameters())
            
            for name, param in child.model.named_parameters():
                if name in p2_params:
                    # 50% chance to inherit from parent2
                    if random.random() < 0.5:
                        param.copy_(p2_params[name])
        
        return child
    
    def evolve_generation(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Evolve one generation.
        
        Returns:
            Stats about the generation
        """
        # Use validation set for fitness if provided
        x_eval = x_val if x_val is not None else x_train
        y_eval = y_val if y_val is not None else y_train
        
        # Evaluate current population
        self.evaluate_population(x_eval, y_eval)
        
        # Select parents
        parents = self.select_parents()
        
        # Create new population
        new_population = []
        
        # Keep elite unchanged
        sorted_pop = sorted(self.population, key=lambda ind: ind.fitness)
        for elite in sorted_pop[:self.elite_size]:
            new_population.append(elite.clone())
        
        # Create offspring
        while len(new_population) < self.population_size:
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            
            # Crossover
            child = self.crossover(parent1, parent2)
            
            # Mutation
            child = self.mutate(child)
            
            # L-BFGS refinement
            if self.use_lbfgs_refinement:
                lbfgs = LBFGSConstantOptimizer(child.model, max_iter=self.lbfgs_steps)
                for _ in range(self.lbfgs_steps):
                    lbfgs.step(x_train, y_train)
            
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1
        
        # Stats
        fitnesses = [ind.fitness for ind in self.population]
        return {
            'generation': self.generation,
            'best_fitness': min(fitnesses),
            'mean_fitness': sum(fitnesses) / len(fitnesses),
            'worst_fitness': max(fitnesses),
            'best_ever': self.best_ever.fitness if self.best_ever else None,
        }


class HybridOptimizer:
    """
    Complete hybrid optimization combining:
    1. Gradient descent (Adam) for initial exploration
    2. L-BFGS for precise constant fitting
    3. Evolution for topology search
    
    Training schedule:
    - Phase 1: Warm-up with Adam
    - Phase 2: Alternate between evolution and L-BFGS
    - Phase 3: Final L-BFGS refinement
    """
    
    def __init__(
        self,
        model: nn.Module,
        model_factory: Optional[Callable[[], nn.Module]] = None,
        population_size: int = 10,
        use_evolution: bool = True,
    ):
        """
        Args:
            model: Initial model (used as template)
            model_factory: Optional factory for creating new models
            population_size: Size of evolutionary population
            use_evolution: If False, only use gradient-based optimization
        """
        self.model = model
        self.model_factory = model_factory or (lambda: copy.deepcopy(model))
        self.use_evolution = use_evolution
        
        # Optimizers
        self.adam = Adam(model.parameters(), lr=0.01)
        self.lbfgs = LBFGSConstantOptimizer(model)
        
        if use_evolution:
            self.evolution = EvolutionaryOptimizer(
                self.model_factory,
                population_size=population_size,
            )
        
        self.loss_fn = nn.MSELoss()
        self.history = []
    
    def train(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        epochs: int = 100,
        warmup_epochs: int = 20,
        evolution_epochs: int = 30,
        lbfgs_epochs: int = 10,
        print_every: int = 10,
    ) -> Dict:
        """
        Full hybrid training.
        
        Args:
            x_train: Training inputs
            y_train: Training targets
            epochs: Total epochs
            warmup_epochs: Initial Adam warmup
            evolution_epochs: Epochs for evolutionary search
            lbfgs_epochs: Final L-BFGS refinement
            print_every: Print frequency
            
        Returns:
            Training history
        """
        # Phase 1: Warm-up with Adam
        print("Phase 1: Adam Warm-up")
        for epoch in range(warmup_epochs):
            self.model.train()
            self.adam.zero_grad()
            pred, _ = self.model(x_train, hard=False)  # Soft during warmup
            loss = self.loss_fn(pred, y_train)
            loss.backward()
            self.adam.step()
            
            self.history.append({'epoch': epoch, 'phase': 'warmup', 'loss': loss.item()})
            
            if epoch % print_every == 0:
                print(f"  Epoch {epoch}: Loss = {loss.item():.4f}")
        
        # Phase 2: Evolution + L-BFGS (if enabled)
        if self.use_evolution:
            print("\nPhase 2: Evolutionary Search")
            
            # Initialize population with current model
            self.evolution.initialize_population()
            # Replace one individual with our trained model
            self.evolution.population[0] = Individual(copy.deepcopy(self.model))
            
            for gen in range(evolution_epochs):
                stats = self.evolution.evolve_generation(x_train, y_train)
                
                self.history.append({
                    'epoch': warmup_epochs + gen,
                    'phase': 'evolution',
                    'loss': stats['best_fitness'],
                })
                
                if gen % print_every == 0:
                    print(f"  Gen {gen}: Best = {stats['best_fitness']:.4f}, "
                          f"Mean = {stats['mean_fitness']:.4f}")
            
            # Use best evolved model
            if self.evolution.best_ever:
                self.model = copy.deepcopy(self.evolution.best_ever.model)
        
        # Phase 3: Final L-BFGS refinement
        print("\nPhase 3: L-BFGS Refinement")
        self.lbfgs = LBFGSConstantOptimizer(self.model, max_iter=50)
        
        for epoch in range(lbfgs_epochs):
            loss = self.lbfgs.step(x_train, y_train, hard=True)
            
            self.history.append({
                'epoch': warmup_epochs + evolution_epochs + epoch,
                'phase': 'lbfgs',
                'loss': loss,
            })
            
            if epoch % max(1, print_every // 2) == 0:
                print(f"  L-BFGS {epoch}: Loss = {loss:.6f}")
        
        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            pred, _ = self.model(x_train, hard=True)
            final_loss = self.loss_fn(pred, y_train).item()
        
        print(f"\nFinal Loss: {final_loss:.6f}")
        
        return {
            'history': self.history,
            'final_loss': final_loss,
        }


# ============================================================================
# Gradient-Guided Mutation (Optional Enhancement)
# ============================================================================

class GradientGuidedEvolution(EvolutionaryOptimizer):
    """
    Enhanced evolution that uses gradients to guide mutations.
    
    If the gradient suggests increasing a particular operation's logit,
    we're more likely to mutate in that direction.
    """
    
    def compute_gradient_guidance(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute gradients of loss w.r.t. operation selection logits.
        
        Returns dict mapping parameter name to gradient direction.
        """
        model.train()
        model.zero_grad()
        
        pred, _ = model(x, hard=False)  # Soft for gradient computation
        loss = self.loss_fn(pred, y)
        loss.backward()
        
        guidance = {}
        for name, param in model.named_parameters():
            if 'logit' in name and param.grad is not None:
                # Negative gradient = direction of improvement
                guidance[name] = -param.grad.clone()
        
        return guidance
    
    def mutate_guided(
        self,
        individual: Individual,
        guidance: Dict[str, torch.Tensor],
        guidance_strength: float = 0.5,
    ) -> Individual:
        """
        Mutate with gradient guidance.
        
        Mutations are biased toward the gradient direction.
        """
        mutant = individual.clone()
        
        with torch.no_grad():
            for name, param in mutant.model.named_parameters():
                if random.random() < self.mutation_rate:
                    if name in guidance:
                        # Guided mutation: noise + gradient direction
                        noise = torch.randn_like(param) * 0.3
                        grad_direction = guidance[name]
                        grad_direction = grad_direction / (grad_direction.norm() + 1e-8)
                        
                        mutation = (1 - guidance_strength) * noise + guidance_strength * grad_direction * 0.5
                        param.add_(mutation)
                    else:
                        # Regular mutation
                        noise = torch.randn_like(param) * 0.1
                        param.add_(noise)
        
        return mutant
