"""
Hard Concrete Distribution for Operation-Based Neural Networks.

The Hard Concrete distribution is a continuous relaxation of discrete random
variables that can produce EXACT zeros and ones (unlike Gumbel-Softmax).

This is critical for:
1. True pruning: edges/operations can be completely removed during training
2. Closing the train-test gap: behavior during training matches inference
3. Sparse architectures: many zeros = efficient computation

Reference: "The Concrete Distribution" (Maddison et al., 2016)
           "Learning Sparse Neural Networks through L0 Regularization" (Louizos et al., 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


def hard_concrete_sample(
    logits: torch.Tensor,
    tau: float = 0.5,
    beta: float = 0.1,
    hard: bool = True,
    training: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Sample from Hard Concrete distribution.
    
    The Hard Concrete stretches the sigmoid to [-β, 1+β] and then clips to [0, 1].
    This allows exact 0s and 1s to be sampled.
    
    Args:
        logits: Log-odds of the underlying Bernoulli (can be any shape)
        tau: Temperature (lower = more discrete)
        beta: Stretch parameter (higher = more zeros/ones)
        hard: If True, use straight-through estimator
        training: If False, return deterministic sigmoid(logits)
        eps: Small constant for numerical stability
        
    Returns:
        Samples in [0, 1] with exact 0s and 1s possible
    """
    if not training:
        # Deterministic during inference - return soft sigmoid values
        # The actual discrete selection is handled at a higher level
        return torch.sigmoid(logits).clamp(0, 1)
    
    # Sample uniform noise
    u = torch.rand_like(logits).clamp(eps, 1 - eps)
    
    # Inverse CDF of logistic distribution (Gumbel-like sampling)
    s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + logits) / tau)
    
    # Stretch to [-beta, 1+beta]
    s_stretched = s * (1 + 2 * beta) - beta
    
    # Hard clip to [0, 1] (this is what creates exact 0s and 1s)
    z = s_stretched.clamp(0, 1)
    
    if hard:
        # Straight-through estimator: forward uses hard values, backward uses soft
        z_hard = (z > 0.5).float()
        z = z + (z_hard - z).detach()
    
    return z


def hard_concrete_log_prob(
    z: torch.Tensor,
    logits: torch.Tensor,
    tau: float = 0.5,
    beta: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute log probability under Hard Concrete distribution.
    
    Used for regularization (penalize high probability of being "on").
    
    Args:
        z: Sampled values in [0, 1]
        logits: Log-odds parameter
        tau: Temperature
        beta: Stretch parameter
        
    Returns:
        Log probabilities (same shape as z)
    """
    # This is approximate; the true density is complex at the boundaries
    # For regularization purposes, we use the stretched BinaryConcrete density
    
    # Transform z back to pre-clipped space
    s = (z + beta) / (1 + 2 * beta)
    s = s.clamp(eps, 1 - eps)
    
    # Log-density of logistic
    log_s = torch.log(s)
    log_1_minus_s = torch.log(1 - s)
    
    log_prob = (logits / tau) - (1 / tau + 1) * log_s - (1 / tau + 1) * log_1_minus_s
    log_prob = log_prob - math.log(1 + 2 * beta)
    
    return log_prob


class HardConcreteGate(nn.Module):
    """
    Learnable gate using Hard Concrete distribution.
    
    Can be used to:
    - Prune operations (gate = 0 means operation disabled)
    - Prune edges (gate = 0 means edge removed)
    - Select discrete options (multiple gates, take argmax)
    """
    
    def __init__(
        self,
        n_gates: int = 1,
        init_logit: float = 0.0,
        tau: float = 0.5,
        beta: float = 0.1,
        learn_tau: bool = False,
    ):
        """
        Args:
            n_gates: Number of independent gates
            init_logit: Initial log-odds (0 = 50% chance of on)
            tau: Temperature
            beta: Stretch parameter
            learn_tau: If True, tau is learnable
        """
        super().__init__()
        
        self.logits = nn.Parameter(torch.full((n_gates,), init_logit))
        self.beta = beta
        
        if learn_tau:
            self.log_tau = nn.Parameter(torch.tensor(math.log(tau)))
            self._learn_tau = True
        else:
            self.register_buffer('log_tau', torch.tensor(math.log(tau)))
            self._learn_tau = False
    
    @property
    def tau(self) -> float:
        """Get tau value. Returns float for non-learnable, keeps tensor for learnable."""
        return torch.exp(self.log_tau).item()
    
    @property
    def tau_tensor(self) -> torch.Tensor:
        """Get tau as tensor (preserves gradients when learnable)."""
        return torch.exp(self.log_tau)
    
    def forward(self, hard: bool = True) -> torch.Tensor:
        """
        Sample gate values.
        
        Args:
            hard: Use straight-through estimator
            
        Returns:
            Gate values in [0, 1], shape (n_gates,)
        """
        # Use tensor tau when learnable to preserve gradients
        tau = self.tau_tensor if self._learn_tau else self.tau
        return hard_concrete_sample(
            self.logits,
            tau=tau,
            beta=self.beta,
            hard=hard,
            training=self.training,
        )

    def set_tau(self, tau: float):
        """Set temperature for annealing (supports learnable and fixed tau)."""
        with torch.no_grad():
            if self._learn_tau:
                self.log_tau.copy_(torch.log(torch.tensor(tau)))
            else:
                self.log_tau.copy_(torch.tensor(math.log(tau)))
    
    def get_mask(self, threshold: float = 0.5) -> torch.Tensor:
        """Get deterministic binary mask."""
        return (torch.sigmoid(self.logits) > threshold).float()
    
    def l0_regularization(self) -> torch.Tensor:
        """
        Compute L0 regularization term (expected number of active gates).
        
        This encourages sparsity by penalizing the probability of gates being on.
        """
        # Probability that gate is non-zero (not clipped to 0)
        # P(z > 0) = sigmoid(logits - beta * tau)
        prob_nonzero = torch.sigmoid(self.logits - self.beta * self.tau)
        return prob_nonzero.sum()
    
    def expected_gates(self) -> torch.Tensor:
        """Get expected value of gates (for soft computation without sampling)."""
        return torch.sigmoid(self.logits)


class HardConcreteSelector(nn.Module):
    """
    Select one option from K choices using Hard Concrete.
    
    Unlike softmax which always sums to 1, this allows:
    - All options to be "off" (sparse)
    - True discrete selection (one-hot at inference)
    
    Great for operation selection!
    """
    
    def __init__(
        self,
        n_options: int,
        init_mode: str = 'uniform',
        tau: float = 0.5,
        beta: float = 0.1,
    ):
        """
        Args:
            n_options: Number of options to select from
            init_mode: 'uniform' (all equal), 'first' (favor first option)
            tau: Temperature
            beta: Stretch parameter
        """
        super().__init__()
        self.n_options = n_options
        self.tau = tau
        self.beta = beta
        
        # Logits for each option
        self.logits = nn.Parameter(torch.zeros(n_options))
        
        if init_mode == 'uniform':
            pass  # Already zeros
        elif init_mode == 'first':
            self.logits.data[0] = 1.0
        else:
            nn.init.normal_(self.logits, mean=0, std=0.1)
    
    def forward(self, hard: bool = True) -> torch.Tensor:
        """
        Sample selection weights.
        
        Args:
            hard: Use straight-through for discrete selection
            
        Returns:
            Weights in [0, 1]^K (NOT guaranteed to sum to 1)
        """
        # Discrete inference: return one-hot based on argmax
        if hard and not self.training:
            one_hot = torch.zeros_like(self.logits)
            one_hot[self.logits.argmax()] = 1.0
            return one_hot
        
        # Training: use hard concrete sampling
        gates = hard_concrete_sample(
            self.logits,
            tau=self.tau,
            beta=self.beta,
            hard=hard,
            training=self.training,
        )
        
        if hard and self.training:
            # Normalize to make it more like one-hot during training
            # But allow zeros (if all gates are 0, output all 0)
            total = gates.sum() + 1e-8
            gates = gates / total
        
        return gates
    
    def select(self) -> int:
        """Get deterministic selection (argmax of expected values)."""
        return self.logits.argmax().item()
    
    def set_tau(self, tau: float):
        """Update temperature for annealing during training."""
        self.tau = tau
        
    def entropy(self) -> torch.Tensor:
        """Compute entropy of selection distribution."""
        probs = F.softmax(self.logits, dim=-1)
        return -(probs * torch.log(probs + 1e-10)).sum()
    
    def l0_regularization(self) -> torch.Tensor:
        """
        Compute L0 regularization (expected number of active options).
        
        Encourages sparsity by penalizing probability of being on.
        """
        # Probability that each option is non-zero
        prob_nonzero = torch.sigmoid(self.logits - self.beta * self.tau)
        return prob_nonzero.sum()



class HardConcreteOperationSelector(nn.Module):
    """
    Specialized selector for choosing between operation types.
    
    Designed for ONN where we need to select:
    - Unary vs Binary vs Aggregation
    - Which specific meta-operation within each type
    
    Two-level selection:
    1. Select operation TYPE (unary/binary/aggregation)
    2. Select specific operation within type
    
    OPTIMIZED: Batches all logits into a single hard_concrete_sample call.
    """
    
    def __init__(
        self,
        n_unary: int = 4,      # periodic, power, exp, log
        n_binary: int = 2,     # arithmetic, aggregation
        tau: float = 0.5,
        beta: float = 0.1,
    ):
        super().__init__()
        self.n_unary = n_unary
        self.n_binary = n_binary
        self.tau = tau
        self.beta = beta
        
        # Batch all logits together: [type(2), unary(n_unary), binary(n_binary)]
        total = 2 + n_unary + n_binary
        self.logits = nn.Parameter(torch.zeros(total))
        
        # Slice indices
        self._type_end = 2
        self._unary_end = 2 + n_unary
    
    def forward(self, hard: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample operation selection.
        
        Returns:
            type_weights: (2,) weights for [unary, binary]
            unary_weights: (n_unary,) weights for unary ops
            binary_weights: (n_binary,) weights for binary ops
        """
        if hard and not self.training:
            # Deterministic discrete selection at inference
            type_weights = torch.zeros(self._type_end, device=self.logits.device)
            unary_weights = torch.zeros(self.n_unary, device=self.logits.device)
            binary_weights = torch.zeros(self.n_binary, device=self.logits.device)
            type_weights[self.logits[:self._type_end].argmax()] = 1.0
            unary_weights[self.logits[self._type_end:self._unary_end].argmax()] = 1.0
            binary_weights[self.logits[self._unary_end:].argmax()] = 1.0
            return type_weights, unary_weights, binary_weights

        # Training (and soft mode): use Hard Concrete sampling
        all_gates = hard_concrete_sample(
            self.logits,
            tau=self.tau,
            beta=self.beta,
            hard=hard,
            training=self.training,
        )
        
        # Split into components
        type_weights = all_gates[:self._type_end]
        unary_weights = all_gates[self._type_end:self._unary_end]
        binary_weights = all_gates[self._unary_end:]
        
        if hard and self.training:
            # Normalize to approximate one-hot while allowing zeros
            type_weights = type_weights / (type_weights.sum() + 1e-8)
            unary_weights = unary_weights / (unary_weights.sum() + 1e-8)
            binary_weights = binary_weights / (binary_weights.sum() + 1e-8)

        return type_weights, unary_weights, binary_weights
    
    def get_selected(self) -> dict:
        """Get deterministic selection."""
        type_idx = self.logits[:self._type_end].argmax().item()
        unary_idx = self.logits[self._type_end:self._unary_end].argmax().item()
        binary_idx = self.logits[self._unary_end:].argmax().item()
        
        return {
            'type': 'unary' if type_idx == 0 else 'binary',
            'unary_idx': unary_idx,
            'binary_idx': binary_idx,
        }
    
    def set_tau(self, tau: float):
        """Update temperature for annealing during training."""
        self.tau = tau
    
    def l0_regularization(self) -> torch.Tensor:
        """Total L0 regularization for sparsity."""
        prob_nonzero = torch.sigmoid(self.logits - self.beta * self.tau)
        return prob_nonzero.sum()
    
    def entropy_regularization(self) -> torch.Tensor:
        """Total entropy (negative = encourages discrete selection)."""
        # Compute entropy for each group
        type_probs = F.softmax(self.logits[:self._type_end], dim=-1)
        unary_probs = F.softmax(self.logits[self._type_end:self._unary_end], dim=-1)
        binary_probs = F.softmax(self.logits[self._unary_end:], dim=-1)
        
        type_ent = -(type_probs * torch.log(type_probs + 1e-10)).sum()
        unary_ent = -(unary_probs * torch.log(unary_probs + 1e-10)).sum()
        binary_ent = -(binary_probs * torch.log(binary_probs + 1e-10)).sum()
        
        return type_ent + unary_ent + binary_ent


# ============================================================================
# Annealing Utilities
# ============================================================================

def anneal_tau(
    step: int,
    total_steps: int,
    tau_start: float = 1.0,
    tau_end: float = 0.1,
    schedule: str = 'cosine',
) -> float:
    """
    Anneal temperature from high (exploration) to low (exploitation).
    
    Args:
        step: Current training step
        total_steps: Total training steps
        tau_start: Initial temperature
        tau_end: Final temperature
        schedule: 'linear', 'cosine', or 'exponential'
        
    Returns:
        Current temperature
    """
    progress = min(step / max(total_steps, 1), 1.0)
    
    if schedule == 'linear':
        tau = tau_start + (tau_end - tau_start) * progress
    elif schedule == 'cosine':
        tau = tau_end + 0.5 * (tau_start - tau_end) * (1 + math.cos(math.pi * progress))
    elif schedule == 'exponential':
        tau = tau_start * (tau_end / tau_start) ** progress
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
    
    return tau


def anneal_beta(
    step: int,
    total_steps: int,
    beta_start: float = 0.05,
    beta_end: float = 0.2,
) -> float:
    """
    Anneal stretch parameter (higher beta = more zeros/ones).
    
    Start with small beta (softer) and increase to force discreteness.
    """
    progress = min(step / max(total_steps, 1), 1.0)
    return beta_start + (beta_end - beta_start) * progress
