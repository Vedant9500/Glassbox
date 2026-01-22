"""
Differentiable Routing for Operation-Based Neural Networks.

Solves the variable arity problem: how does a binary operation know
which 2 inputs to use when connected to N previous nodes?

Solution: Routing Matrix with soft input slot selection.
Each node has input "slots" (e.g., 2 for binary ops).
The routing matrix learns which source nodes connect to which slots.

Key features:
- Fully differentiable via softmax
- Supports variable arity (1, 2, or N inputs)
- Can be combined with Hard Concrete for discrete pruning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class DifferentiableRouter(nn.Module):
    """
    Learns which input sources connect to which input slots.
    
    For a node with max_arity=2 (binary operation) receiving from N sources:
    - Slot 0 learns to select one of N sources
    - Slot 1 learns to select one of N sources (can be same or different)
    
    Uses softmax for differentiable soft selection during training.
    During inference, can use hard argmax.
    """
    
    def __init__(
        self,
        n_sources: int,
        max_arity: int = 2,
        init_mode: str = 'uniform',
    ):
        """
        Args:
            n_sources: Number of potential input sources (previous nodes)
            max_arity: Maximum number of input slots (2 for binary ops)
            init_mode: How to initialize routing weights
                - 'uniform': All sources equally likely
                - 'identity': Each slot prefers corresponding source
                - 'random': Random initialization
        """
        super().__init__()
        self.n_sources = n_sources
        self.max_arity = max_arity
        
        # Routing logits: R[slot, source] = preference for connecting source to slot
        self.R = nn.Parameter(torch.zeros(max_arity, n_sources))
        
        self._init_routing(init_mode)
    
    def _init_routing(self, mode: str):
        """Initialize routing weights."""
        with torch.no_grad():
            if mode == 'uniform':
                # All sources equally likely
                self.R.zero_()
            elif mode == 'identity':
                # Slot i prefers source i (if available)
                for i in range(min(self.max_arity, self.n_sources)):
                    self.R[i, i] = 1.0
            elif mode == 'random':
                nn.init.normal_(self.R, mean=0.0, std=0.1)
            else:
                raise ValueError(f"Unknown init_mode: {mode}")
    
    def forward(
        self,
        sources: torch.Tensor,
        tau: float = 1.0,
        hard: bool = False,
    ) -> torch.Tensor:
        """
        Route inputs from sources to slots.
        
        Args:
            sources: (batch, n_sources) or list of (batch,) tensors
            tau: Temperature for softmax (lower = more discrete)
            hard: If True, use straight-through estimator for discrete selection
            
        Returns:
            slots: (batch, max_arity) - the routed inputs
        """
        if isinstance(sources, list):
            sources = torch.stack(sources, dim=-1)  # (batch, n_sources)
        
        # OPTIMIZED: Use fast matmul path for soft selection
        # Only use expensive Gumbel-softmax when hard=True AND training
        if hard and self.training:
            # Gumbel-softmax with straight-through (slower but needed for discrete gradients)
            # Optimization: sample Gumbel noise directly instead of expand
            gumbel_noise = -torch.empty_like(self.R).exponential_().log()
            gumbel_noise = gumbel_noise - torch.empty_like(self.R).exponential_().log()
            y = (self.R + gumbel_noise) / tau
            probs_soft = F.softmax(y, dim=-1)
            # Straight-through: hard forward, soft backward
            index = probs_soft.argmax(dim=-1, keepdim=True)
            probs_hard = torch.zeros_like(probs_soft).scatter_(-1, index, 1.0)
            probs = probs_hard - probs_soft.detach() + probs_soft  # (max_arity, n_sources)
            # Fast matmul: slots = sources @ probs.T
            slots = sources @ probs.T  # (batch, max_arity)
        else:
            # FAST PATH: Simple softmax + matmul (50x faster)
            probs = F.softmax(self.R / tau, dim=-1)  # (max_arity, n_sources)
            if hard:
                # Inference: use argmax for true discrete selection
                index = probs.argmax(dim=-1, keepdim=True)
                probs = torch.zeros_like(probs).scatter_(-1, index, 1.0)
            # Fast matmul: slots = sources @ probs.T
            slots = sources @ probs.T  # (batch, max_arity)
        
        return slots
    
    def get_routing(self, threshold: float = 0.3) -> List[Tuple[int, int]]:
        """
        Get discrete routing as list of (slot, source) connections.
        
        Args:
            threshold: Only include connections with prob > threshold
            
        Returns:
            List of (slot_idx, source_idx) tuples
        """
        probs = F.softmax(self.R, dim=-1)
        connections = []
        
        for slot in range(self.max_arity):
            source = probs[slot].argmax().item()
            prob = probs[slot, source].item()
            if prob > threshold:
                connections.append((slot, source, prob))
        
        return connections
    
    def get_primary_sources(self) -> List[int]:
        """Get the most likely source for each slot."""
        return [self.R[slot].argmax().item() for slot in range(self.max_arity)]


class EdgeWeights(nn.Module):
    """
    Learnable edge constants that scale inputs.
    
    Each edge from source j to target i has a constant c_ji.
    The signal becomes: c_ji * output_j
    
    Supports:
    - Per-edge constants
    - Edge pruning via Hard Concrete
    - Bias terms
    """
    
    def __init__(
        self,
        n_sources: int,
        init_scale: float = 1.0,
        with_bias: bool = False,
        learnable: bool = True,
    ):
        super().__init__()
        self.n_sources = n_sources
        
        if learnable:
            self.weights = nn.Parameter(torch.ones(n_sources) * init_scale)
            if with_bias:
                self.bias = nn.Parameter(torch.zeros(1))
            else:
                self.register_buffer('bias', torch.zeros(1))
        else:
            self.register_buffer('weights', torch.ones(n_sources) * init_scale)
            self.register_buffer('bias', torch.zeros(1))
    
    def forward(self, sources: torch.Tensor) -> torch.Tensor:
        """
        Apply edge weights to sources.
        
        Args:
            sources: (batch, n_sources) tensor
            
        Returns:
            weighted: (batch, n_sources) tensor with weights applied
        """
        return sources * self.weights + self.bias
    
    def get_active_edges(self, threshold: float = 0.1) -> List[int]:
        """Get indices of edges with weight above threshold."""
        return [i for i in range(self.n_sources) 
                if abs(self.weights[i].item()) > threshold]
    
    def prune_weak_edges(self, threshold: float = 0.1):
        """Set weak edge weights to exactly zero."""
        with torch.no_grad():
            mask = self.weights.abs() > threshold
            self.weights.mul_(mask.float())


class RoutedOperationInput(nn.Module):
    """
    Complete input processing for an operation node.
    
    Combines:
    1. Edge weights (scaling)
    2. Routing (source selection)
    
    For a binary operation receiving from N sources, this:
    1. Applies edge weights to all N sources
    2. Routes to 2 input slots
    """
    
    def __init__(
        self,
        n_sources: int,
        max_arity: int = 2,
        with_edge_weights: bool = True,
        init_routing: str = 'uniform',
    ):
        super().__init__()
        self.n_sources = n_sources
        self.max_arity = max_arity
        
        if with_edge_weights:
            self.edge_weights = EdgeWeights(n_sources)
        else:
            self.edge_weights = None
        
        self.router = DifferentiableRouter(n_sources, max_arity, init_routing)
    
    def forward(
        self,
        sources: torch.Tensor,
        tau: float = 1.0,
        hard: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Process inputs for operation.
        
        Args:
            sources: (batch, n_sources) tensor
            tau: Temperature for routing
            hard: Use hard routing
            
        Returns:
            Tuple of (slot_0, slot_1, ...) tensors, each (batch,)
        """
        # Apply edge weights
        if self.edge_weights is not None:
            sources = self.edge_weights(sources)
        
        # Route to slots
        slots = self.router(sources, tau=tau, hard=hard)  # (batch, max_arity)
        
        # Return as tuple for easy unpacking
        return tuple(slots[:, i] for i in range(self.max_arity))
    
    def get_routing_summary(self) -> dict:
        """Get summary of routing and edge weights."""
        return {
            'primary_sources': self.router.get_primary_sources(),
            'routing': self.router.get_routing(),
            'active_edges': self.edge_weights.get_active_edges() if self.edge_weights else list(range(self.n_sources)),
        }


class AdaptiveArityRouter(nn.Module):
    """
    Router that can dynamically handle different operation arities.
    
    For unary ops (arity=1): Only slot 0 is used
    For binary ops (arity=2): Both slots are used
    For aggregation (arity=N): All sources are used directly
    
    The arity is determined by the selected operation type.
    """
    
    def __init__(
        self,
        n_sources: int,
        max_arity: int = 2,
    ):
        super().__init__()
        self.n_sources = n_sources
        self.max_arity = max_arity
        
        # Edge weights (applied to all sources)
        self.edge_weights = nn.Parameter(torch.ones(n_sources))
        
        # Routing for slot-based operations
        self.router = DifferentiableRouter(n_sources, max_arity)
    
    def forward_unary(
        self,
        sources: torch.Tensor,
        tau: float = 1.0,
        hard: bool = False,
    ) -> torch.Tensor:
        """Get single routed input for unary operation."""
        weighted = sources * self.edge_weights
        slots = self.router(weighted, tau=tau, hard=hard)
        return slots[:, 0]  # Only first slot
    
    def forward_binary(
        self,
        sources: torch.Tensor,
        tau: float = 1.0,
        hard: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get two routed inputs for binary operation."""
        weighted = sources * self.edge_weights
        slots = self.router(weighted, tau=tau, hard=hard)
        return slots[:, 0], slots[:, 1]
    
    def forward_aggregation(
        self,
        sources: torch.Tensor,
    ) -> torch.Tensor:
        """Get all weighted inputs for aggregation operation."""
        return sources * self.edge_weights  # (batch, n_sources)
    
    def forward(
        self,
        sources: torch.Tensor,
        arity: int = 2,
        tau: float = 1.0,
        hard: bool = False,
    ) -> torch.Tensor:
        """
        Forward with specified arity.
        
        Returns:
            For arity=1: (batch,) single input
            For arity=2: (batch, 2) two inputs
            For arity>2: (batch, n_sources) all weighted inputs
        """
        if arity == 1:
            return self.forward_unary(sources, tau, hard).unsqueeze(-1)
        elif arity == 2:
            x, y = self.forward_binary(sources, tau, hard)
            return torch.stack([x, y], dim=-1)
        else:
            return self.forward_aggregation(sources)
