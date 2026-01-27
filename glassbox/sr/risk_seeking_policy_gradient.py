"""
Risk-Seeking Policy Gradient for Symbolic Regression.

In symbolic regression, we only care about finding THE BEST formula, not
the average performance across all candidates. This module implements
risk-seeking optimization that focuses on the top-k percentile of solutions.

Key insight from research (Tier 2 priority):
- Traditional evolution optimizes E[fitness] = mean(population_fitness)
- Risk-seeking optimizes percentile(population_fitness, k) where k=10-20%
- This concentrates search on the most promising solutions

Activation Strategy:
- Only activate RSPG when gradients are stuck or exploding
- Use normal evolution for exploration, RSPG for exploitation
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Tuple
import numpy as np
from collections import deque


class GradientMonitor:
    """
    Monitor gradient statistics to detect stuck or exploding gradients.
    
    Tracks:
    - Loss improvement over sliding window
    - Gradient magnitude statistics
    - Triggers for switching to risk-seeking mode
    """
    
    def __init__(
        self,
        window_size: int = 10,
        stuck_threshold: float = 1e-6,
        explosion_multiplier: float = 10.0,
    ):
        """
        Args:
            window_size: Number of recent iterations to track
            stuck_threshold: Min improvement to not be considered stuck
            explosion_multiplier: Gradient norm must be > median * this to explode
        """
        self.window_size = window_size
        self.stuck_threshold = stuck_threshold
        self.explosion_multiplier = explosion_multiplier
        
        # Sliding windows
        self.loss_history = deque(maxlen=window_size)
        self.grad_norm_history = deque(maxlen=window_size * 2)  # Longer for median
        
        # State
        self.iteration = 0
        self.is_rspg_active = False
    
    def update(self, loss: float, grad_norm: Optional[float] = None):
        """Update monitoring statistics."""
        self.loss_history.append(loss)
        if grad_norm is not None:
            self.grad_norm_history.append(grad_norm)
        self.iteration += 1
    
    def is_stuck(self) -> bool:
        """Check if optimization is stuck (no improvement)."""
        if len(self.loss_history) < self.window_size:
            return False
        
        # Compare first half to second half of window
        mid = self.window_size // 2
        old_losses = list(self.loss_history)[:mid]
        new_losses = list(self.loss_history)[mid:]
        
        old_mean = np.mean(old_losses)
        new_mean = np.mean(new_losses)
        
        # Stuck if improvement is too small (either direction counts as stuck)
        improvement = abs(old_mean - new_mean)
        
        return improvement < self.stuck_threshold
    
    def is_exploding(self) -> bool:
        """Check if gradients are exploding."""
        if len(self.grad_norm_history) < 5:
            return False
        
        recent_norm = self.grad_norm_history[-1]
        median_norm = np.median(list(self.grad_norm_history))
        
        # Exploding if current norm is much larger than median
        if median_norm < 1e-8:
            return False  # Can't judge if median too small
        
        return recent_norm > median_norm * self.explosion_multiplier
    
    def should_activate_rspg(self) -> bool:
        """Determine if risk-seeking mode should be activated."""
        # Need enough data first
        if len(self.loss_history) < self.window_size:
            return False
        
        stuck = self.is_stuck()
        exploding = self.is_exploding()
        
        # Activate if stuck OR exploding
        should_activate = stuck or exploding
        
        # Update state
        if should_activate and not self.is_rspg_active:
            self.is_rspg_active = True
        elif not should_activate and self.is_rspg_active:
            # Deactivate after some cooldown
            # For now, stay in RSPG once activated
            pass
        
        return should_activate
    
    def get_stats(self) -> Dict[str, float]:
        """Return current statistics."""
        stats = {
            'iteration': self.iteration,
            'is_rspg_active': float(self.is_rspg_active),
        }
        
        if len(self.loss_history) > 0:
            stats['current_loss'] = self.loss_history[-1]
            stats['mean_loss'] = np.mean(list(self.loss_history))
        
        if len(self.grad_norm_history) > 0:
            stats['current_grad_norm'] = self.grad_norm_history[-1]
            stats['median_grad_norm'] = np.median(list(self.grad_norm_history))
        
        stats['is_stuck'] = float(self.is_stuck())
        stats['is_exploding'] = float(self.is_exploding())
        
        return stats


def compute_risk_seeking_fitness(
    fitnesses: List[float],
    percentile: float = 10.0,
) -> float:
    """
    Compute risk-seeking fitness: focus on top-k percentile.
    
    Instead of minimizing mean(fitness), minimize the k-th percentile.
    This concentrates search on the best solutions.
    
    Args:
        fitnesses: List of fitness values (lower is better)
        percentile: Percentile to optimize (10 = top 10%)
    
    Returns:
        The k-th percentile fitness value
    """
    if not fitnesses:
        return float('inf')
    
    # Percentile function expects 0-100 range
    return float(np.percentile(fitnesses, percentile))


def compute_selection_probabilities_rspg(
    fitnesses: List[float],
    percentile: float = 10.0,
    temperature: float = 1.0,
) -> List[float]:
    """
    Compute selection probabilities that favor top-k percentile.
    
    Instead of uniform tournament selection, bias selection heavily
    toward individuals in the top percentile.
    
    Args:
        fitnesses: List of fitness values (lower is better)
        percentile: Threshold percentile (individuals below this get boosted)
        temperature: Softmax temperature (lower = more extreme)
    
    Returns:
        List of selection probabilities (sum to 1)
    """
    if not fitnesses:
        return []
    
    fitnesses = np.array(fitnesses)
    threshold = np.percentile(fitnesses, percentile)
    
    # Compute "advantage" = how much better than threshold
    # Lower fitness is better, so advantage = threshold - fitness
    advantages = threshold - fitnesses
    
    # Apply softmax to get probabilities
    # Lower temperature = more concentrated on best
    exp_advantages = np.exp(advantages / (temperature + 1e-8))
    probabilities = exp_advantages / (exp_advantages.sum() + 1e-8)
    
    return probabilities.tolist()


class RiskSeekingEvolutionMixin:
    """
    Mixin class that adds risk-seeking capabilities to evolutionary trainers.
    
    Usage:
        class MyTrainer(RiskSeekingEvolutionMixin, EvolutionaryONNTrainer):
            ...
    
    Note: This mixin is designed to be safe to use even if init_risk_seeking()
    is never called - methods will gracefully handle missing state.
    """
    
    # Default attribute values - ensures safe access even if init never called
    enable_rspg: bool = False
    rspg_percentile: float = 10.0
    rspg_temperature: float = 0.5
    gradient_monitor: Optional['GradientMonitor'] = None
    
    def init_risk_seeking(
        self,
        enable_rspg: bool = True,
        rspg_percentile: float = 10.0,
        rspg_temperature: float = 0.5,
        monitor_window_size: int = 10,
    ):
        """
        Initialize risk-seeking components.
        
        Args:
            enable_rspg: Whether to enable risk-seeking mode
            rspg_percentile: Target percentile for risk-seeking (10 = top 10%)
            rspg_temperature: Softmax temperature for selection
            monitor_window_size: Window for gradient monitoring
        """
        self.enable_rspg = enable_rspg
        self.rspg_percentile = rspg_percentile
        self.rspg_temperature = rspg_temperature
        
        if enable_rspg:
            self.gradient_monitor = GradientMonitor(window_size=monitor_window_size)
        else:
            self.gradient_monitor = None
    
    def update_gradient_monitor(
        self,
        loss: float,
        model: Optional[nn.Module] = None,
    ):
        """Update gradient monitor with latest stats."""
        gradient_monitor = getattr(self, 'gradient_monitor', None)
        if gradient_monitor is None:
            return
        
        # Compute gradient norm if model provided
        grad_norm = None
        if model is not None:
            grad_norm = self._compute_grad_norm(model)
        
        gradient_monitor.update(loss, grad_norm)
    
    def _compute_grad_norm(self, model: nn.Module) -> float:
        """Compute L2 norm of all gradients."""
        total_norm = 0.0
        param_count = 0
        
        for param in model.parameters():
            if param.grad is not None:
                total_norm += param.grad.norm(2).item() ** 2
                param_count += 1
        
        if param_count == 0:
            return 0.0
        
        return (total_norm ** 0.5)
    
    def should_use_rspg(self) -> bool:
        """Check if risk-seeking mode should be active."""
        # Use getattr with defaults for safe access even if init was never called
        if not getattr(self, 'enable_rspg', False):
            return False
        
        gradient_monitor = getattr(self, 'gradient_monitor', None)
        if gradient_monitor is None:
            return False
        
        return gradient_monitor.should_activate_rspg()
    
    def select_parents_rspg(
        self,
        population: List,
        fitnesses: List[float],
        n_parents: int,
    ) -> List:
        """
        Select parents using risk-seeking probabilities.
        
        Args:
            population: List of individuals
            fitnesses: Corresponding fitness values
            n_parents: Number of parents to select
        
        Returns:
            List of selected parent individuals
        """
        if not fitnesses:
            return []
        
        # Compute selection probabilities
        probabilities = compute_selection_probabilities_rspg(
            fitnesses,
            percentile=self.rspg_percentile,
            temperature=self.rspg_temperature,
        )
        
        # Sample parents according to probabilities
        indices = np.random.choice(
            len(population),
            size=n_parents,
            replace=True,
            p=probabilities,
        )
        
        return [population[i] for i in indices]
    
    def get_rspg_stats(self) -> Dict[str, float]:
        """Get risk-seeking statistics for logging."""
        gradient_monitor = getattr(self, 'gradient_monitor', None)
        if gradient_monitor is None:
            return {'rspg_enabled': 0.0}
        
        stats = gradient_monitor.get_stats()
        stats['rspg_enabled'] = 1.0
        stats['rspg_percentile'] = getattr(self, 'rspg_percentile', 10.0)
        stats['rspg_temperature'] = getattr(self, 'rspg_temperature', 0.5)
        
        return stats


# ============================================================================
# Testing / Demo
# ============================================================================

def test_gradient_monitor():
    """Test the gradient monitor."""
    print("Testing GradientMonitor...")
    
    # Simulate stuck scenario with higher threshold to match test noise
    monitor = GradientMonitor(window_size=10, stuck_threshold=1e-3)
    
    # Simulate stuck scenario
    print("\n1. Stuck scenario:")
    for i in range(15):
        loss = 1.0 + np.random.normal(0, 0.0001)  # Barely changing (std=0.0001)
        grad_norm = 0.1
        monitor.update(loss, grad_norm)
        
        if i >= 10:
            print(f"  Iter {i}: loss={loss:.6f}, stuck={monitor.is_stuck()}")
    
    assert monitor.is_stuck(), "Should detect stuck after 10 iterations"
    print("  ✓ Correctly detected stuck gradient")
    
    # Simulate exploding scenario
    print("\n2. Exploding scenario:")
    monitor = GradientMonitor(window_size=10)
    
    for i in range(15):
        grad_norm = 0.1 if i < 12 else 5.0  # Sudden spike
        loss = 1.0 / (i + 1)
        monitor.update(loss, grad_norm)
        
        if i >= 12:
            print(f"  Iter {i}: grad_norm={grad_norm:.2f}, exploding={monitor.is_exploding()}")
    
    assert monitor.is_exploding(), "Should detect explosion"
    print("  ✓ Correctly detected exploding gradient")
    
    print("\n✓ All tests passed!")


def test_risk_seeking_selection():
    """Test risk-seeking selection probabilities."""
    print("\nTesting risk-seeking selection...")
    
    # Create population with varied fitness
    fitnesses = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    
    # Normal vs risk-seeking
    print(f"\nFitnesses: {fitnesses}")
    print(f"Top 20% threshold: {np.percentile(fitnesses, 20):.2f}")
    
    probs = compute_selection_probabilities_rspg(
        fitnesses,
        percentile=20.0,
        temperature=0.5,
    )
    
    print("\nSelection probabilities (temperature=0.5):")
    for i, (f, p) in enumerate(zip(fitnesses, probs)):
        print(f"  Individual {i}: fitness={f:.2f}, prob={p:.4f} {'***' if p > 0.2 else ''}")
    
    # Check that best individuals have higher probability
    top_2_prob = sum(probs[-2:])  # Best 2 individuals
    print(f"\nTop 2 individuals account for {top_2_prob:.1%} of selections")
    assert top_2_prob > 0.4, "Top 2 should have >40% of probability"
    
    print("\n✓ Risk-seeking selection working correctly")


if __name__ == '__main__':
    test_gradient_monitor()
    test_risk_seeking_selection()
