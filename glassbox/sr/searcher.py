"""
Symbolic Regression Searcher.

Manages training, complexity regularization, and Pareto frontier tracking.
Uses Gumbel-Softmax temperature annealing for stable training at deeper depths.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from .tree import SymbolicTree

# Force CPU to avoid CUDA memory issues
DEVICE = torch.device('cpu')


class SymbolicSearcher:
    """
    Trains a SymbolicTree to discover formulas from data.
    
    Uses Gumbel-Softmax with hard selection to prevent NaN at depth > 2.
    Temperature is annealed from high (exploration) to low (exploitation).
    """
    
    def __init__(
        self,
        var_names: List[str],
        max_depth: int = 3,
        lr: float = 0.01,
        complexity_weight: float = 0.001,
        start_tau: float = 2.0,
        end_tau: float = 0.1,
        hard_selection: bool = True
    ):
        self.var_names = var_names
        self.max_depth = max_depth
        self.lr = lr
        self.complexity_weight = complexity_weight
        self.start_tau = start_tau
        self.end_tau = end_tau
        
        # Configure Gumbel-Softmax settings
        SymbolicTree.set_hard_mode(hard_selection)
        SymbolicTree.set_temperature(start_tau)
        
        self.tree = SymbolicTree(var_names, max_depth).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.tree.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        # Track Pareto frontier: (complexity, loss, formula)
        self.pareto_frontier: List[Tuple[int, float, str]] = []
        self.best_loss = float('inf')
        self.best_formula = ""
    
    def train(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        epochs: int = 1000,
        print_every: int = 100,
        anneal_temperature: bool = True
    ) -> str:
        """
        Train the tree to fit (X, y).
        Returns the discovered formula.
        
        Args:
            X: Input tensor (batch, n_vars)
            y: Target tensor (batch,)
            epochs: Number of training epochs
            print_every: Print progress every N epochs
            anneal_temperature: Whether to anneal Gumbel-Softmax temperature
        """
        X = X.to(DEVICE)
        y = y.to(DEVICE)
        
        print(f"[SymbolicSearcher] Training on {X.shape[0]} samples...")
        print(f"[SymbolicSearcher] Tree depth: {self.max_depth}, Complexity: {self.tree.complexity()}")
        print(f"[SymbolicSearcher] Using Gumbel-Softmax hard selection for stability")
        
        for epoch in range(epochs):
            # Anneal temperature (high -> low) for exploration -> exploitation
            if anneal_temperature:
                tau = SymbolicTree.anneal_temperature(epoch, epochs, self.start_tau, self.end_tau)
            
            self.optimizer.zero_grad()
            
            pred = self.tree(X)
            mse = self.criterion(pred, y)
            
            # Complexity penalty
            complexity = self.tree.complexity()
            loss = mse + self.complexity_weight * complexity
            
            # Check for NaN (should be rare now with Gumbel-Softmax hard selection)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[Epoch {epoch}] NaN/Inf detected! tau={tau:.3f}")
                print(f"[Epoch {epoch}] This is unexpected with hard selection. Check your data.")
                break
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.tree.parameters(), 1.0)
            self.optimizer.step()
            
            if epoch % print_every == 0:
                formula = self.tree.get_formula()
                tau_str = f"tau={tau:.2f}" if anneal_temperature else ""
                print(f"[Epoch {epoch:4d}] MSE: {mse.item():.4f} | Complexity: {complexity} | {tau_str} | Formula: {formula[:70]}...")
                
                # Update Pareto frontier
                self._update_pareto(complexity, mse.item(), formula)
            
            if mse.item() < self.best_loss:
                self.best_loss = mse.item()
                self.best_formula = self.tree.get_formula()
        
        print(f"\n[SymbolicSearcher] Training complete.")
        print(f"[SymbolicSearcher] Best Loss: {self.best_loss:.4f}")
        print(f"[SymbolicSearcher] Best Formula: {self.best_formula}")
        
        return self.best_formula
    
    def _update_pareto(self, complexity: int, loss: float, formula: str):
        """Update Pareto frontier (non-dominated solutions)."""
        # A solution is dominated if another has BOTH lower loss AND lower complexity
        dominated = False
        new_frontier = []
        
        for c, l, f in self.pareto_frontier:
            if c <= complexity and l <= loss and (c < complexity or l < loss):
                dominated = True
            if not (complexity <= c and loss <= l and (complexity < c or loss < l)):
                new_frontier.append((c, l, f))
        
        if not dominated:
            new_frontier.append((complexity, loss, formula))
        
        self.pareto_frontier = new_frontier
    
    def print_pareto(self):
        """Print all Pareto-optimal formulas."""
        print("\n--- Pareto Frontier (Complexity vs Loss) ---")
        for c, l, f in sorted(self.pareto_frontier, key=lambda x: x[0]):
            print(f"  [C={c:2d}] Loss={l:.4f} | {f[:60]}...")
        print("---------------------------------------------\n")
