"""
Operation Node for Operation-Based Neural Networks v2.

An OperationNode is a single computational unit that:
1. Receives inputs from previous nodes via differentiable routing
2. Applies edge weights (scaling constants)
3. Selects and applies a meta-operation using Hard Concrete
4. Outputs the result

This is the fundamental building block of the ONN DAG.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List

from .meta_ops import (
    MetaPeriodic,
    MetaPower,
    MetaArithmetic,
    MetaAggregation,
    MetaExp,
    MetaLog,
)
from .routing import AdaptiveArityRouter
from .hard_concrete import HardConcreteOperationSelector, hard_concrete_sample


class OperationNode(nn.Module):
    """
    A single node in the Operation-Based Neural Network.
    
    Architecture:
    ```
    Sources (n_sources)
         │
         ▼
    ┌────────────────┐
    │  Edge Weights  │  (learnable scaling)
    └────────────────┘
         │
         ▼
    ┌────────────────┐
    │    Routing     │  (which inputs to which slots)
    └────────────────┘
         │
         ▼
    ┌────────────────┐
    │  Op Selection  │  (Hard Concrete: unary vs binary)
    └────────────────┘
         │
         ▼
    ┌────────────────┐
    │ Meta-Operation │  (parametric: sin, power, arithmetic, etc.)
    └────────────────┘
         │
         ▼
      Output
    ```
    """
    
    def __init__(
        self,
        n_sources: int,
        node_idx: int = 0,
        tau: float = 0.5,
        beta: float = 0.1,
        simplified_ops: bool = False,  # Use smaller op menu
        fair_mode: bool = False,       # FairDARTS-style independent sigmoids
        branch_norm: bool = False,     # Apply LayerNorm inside branches (can hurt SR)
    ):
        """
        Args:
            n_sources: Number of potential input sources (previous nodes + inputs)
            node_idx: Index of this node (for debugging/visualization)
            tau: Temperature for Hard Concrete
            beta: Stretch parameter for Hard Concrete
            simplified_ops: If True, use smaller op menu (no exp/log/agg)
                           Recommended for simple formulas - faster convergence
            fair_mode: If True, use FairDARTS independent sigmoids
                      Prevents operation "unfair advantage" during search
            branch_norm: If True, apply LayerNorm inside branches before combining.
                        WARNING: This normalizes to mean=0,std=1 which destroys
                        scale information. Can hurt symbolic regression where
                        actual output magnitudes matter. Disabled by default.
        """
        super().__init__()
        self.n_sources = n_sources
        self.node_idx = node_idx
        self.tau = tau
        self.beta = beta
        self.simplified_ops = simplified_ops
        self.fair_mode = fair_mode
        self.branch_norm_enabled = branch_norm
        
        # Routing: handles edge weights and input slot selection
        self.router = AdaptiveArityRouter(n_sources, max_arity=2)
        
        if simplified_ops:
            # SIMPLIFIED: Only power, periodic, arithmetic
            # Research shows exp/log/agg are often incorrectly selected
            self.op_selector = HardConcreteOperationSelector(
                n_unary=2,    # periodic, power only
                n_binary=1,   # arithmetic only (no aggregation)
                tau=tau,
                beta=beta,
                fair_mode=fair_mode,
            )
            
            self.unary_ops = nn.ModuleList([
                MetaPeriodic(),
                MetaPower(),
            ])
            
            self.binary_ops = nn.ModuleList([
                MetaArithmetic(),
            ])
        else:
            # FULL: All operations (original behavior)
            self.op_selector = HardConcreteOperationSelector(
                n_unary=4,    # periodic, power, exp, log
                n_binary=2,   # arithmetic, aggregation
                tau=tau,
                beta=beta,
                fair_mode=fair_mode,
            )
            
            self.unary_ops = nn.ModuleList([
                MetaPeriodic(),
                MetaPower(),
                MetaExp(),
                MetaLog(),
            ])
            
            self.binary_ops = nn.ModuleList([
                MetaArithmetic(),
                MetaAggregation(),
            ])
        
        # Output BatchNorm for gradient normalization
        self.output_norm = nn.BatchNorm1d(1, affine=False)
        
        # Branch-level LayerNorm to prevent gradient starvation (OPTIONAL)
        # Multiplicative gradients scale with 1/x (explode near 0, vanish for large x)
        # Additive gradients are constant. This imbalance causes additive to dominate.
        # LayerNorm inside each branch equalizes gradient magnitudes before combining.
        # WARNING: Disabled by default because normalizing to mean=0,std=1
        # destroys scale information, which hurts symbolic regression tasks
        # where the actual output magnitude matters (e.g., y = x^2).
        # Reference: research3.md Section 4.2
        if branch_norm:
            self.unary_norm = nn.LayerNorm(1, elementwise_affine=False)
            self.binary_norm = nn.LayerNorm(1, elementwise_affine=False)
        else:
            self.unary_norm = None
            self.binary_norm = None
        
        # Learnable output scale
        self.output_scale = nn.Parameter(torch.ones(1))
    
    def forward(
        self,
        sources: torch.Tensor,
        hard: bool = True,
        normalize: bool = True,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass through the operation node.
        
        Args:
            sources: (batch, n_sources) tensor of inputs from previous nodes
            hard: Use Hard Concrete for discrete selection
            normalize: Apply output normalization
            
        Returns:
            output: (batch,) tensor
            info: Dict with selection info for interpretability
        """
        batch_size = sources.shape[0]
        device = sources.device
        dtype = sources.dtype
        
        # Get operation selection weights
        type_weights, unary_weights, binary_weights = self.op_selector(hard=hard)
        
        # torch.compile COMPATIBILITY: Use torch.where() masking instead of if/else branching
        # This avoids data-dependent control flow that breaks compilation.
        # We compute both paths but mask with weights (which are ~0 for unused paths).
        # Reference: todo.md Tier 2 - torch.compile compatibility
        
        # --- UNARY BRANCH ---
        # Always compute (multiplication by near-zero weight effectively skips)
        unary_input = self.router.forward_unary(sources, tau=self.tau, hard=hard)
        unary_result = torch.zeros(batch_size, device=device, dtype=dtype)
        for i, op in enumerate(self.unary_ops):
            # Weight-masked accumulation (near-zero weights = effectively skipped)
            unary_result = unary_result + op(unary_input) * unary_weights[i]
        
        # Optional branch-level LayerNorm to prevent gradient starvation
        # Only apply if explicitly enabled (disabled by default for SR)
        if self.branch_norm_enabled and self.unary_norm is not None and batch_size > 1:
            unary_result = self.unary_norm(unary_result.unsqueeze(-1)).squeeze(-1)
        
        # --- BINARY BRANCH ---
        binary_result = torch.zeros(batch_size, device=device, dtype=dtype)
        
        # Arithmetic operation (always index 0)
        x, y = self.router.forward_binary(sources, tau=self.tau, hard=hard)
        binary_result = binary_result + self.binary_ops[0](x, y) * binary_weights[0]
        
        # Aggregation operation (only if we have more than 1 binary op)
        if len(self.binary_ops) > 1:
            agg_input = self.router.forward_aggregation(sources)
            binary_result = binary_result + self.binary_ops[1](agg_input, dim=-1) * binary_weights[1]
        
        # Optional branch-level LayerNorm to prevent gradient starvation
        if self.branch_norm_enabled and self.binary_norm is not None and batch_size > 1:
            binary_result = self.binary_norm(binary_result.unsqueeze(-1)).squeeze(-1)
        
        # --- COMBINE WITH TYPE WEIGHTS ---
        output = type_weights[0] * unary_result + type_weights[1] * binary_result
        
        # Normalize output (skip if coefficients have been finalized)
        skip_norm = getattr(self, '_skip_output_norm', False)
        if normalize and batch_size > 1 and not skip_norm:
            output = self.output_norm(output.unsqueeze(-1)).squeeze(-1)
        
        # Apply output scale
        output = output * self.output_scale
        
        # Clamp to prevent explosion
        output = torch.clamp(output, -100, 100)
        
        # Build info dict for interpretability
        info = {
            'node_idx': self.node_idx,
            'type_weights': type_weights.detach(),
            'unary_weights': unary_weights.detach(),
            'binary_weights': binary_weights.detach(),
            'primary_sources': self.router.get_primary_sources() if hasattr(self.router, 'get_primary_sources') else [],
        }
        
        return output, info
    
    def get_selected_operation(self) -> str:
        """Get human-readable description of selected operation."""
        selection = self.op_selector.get_selected()
        
        if selection['type'] == 'unary':
            op_names = ['periodic', 'power', 'exp', 'log']
            op_idx = selection['unary_idx']
            op = self.unary_ops[op_idx]
            return f"{op_names[op_idx]}: {op.get_discrete_op()}"
        else:
            op_names = ['arithmetic', 'aggregation']
            op_idx = selection['binary_idx']
            op = self.binary_ops[op_idx]
            return f"{op_names[op_idx]}: {op.get_discrete_op()}"
    
    def get_routing_info(self) -> Dict:
        """Get routing information for visualization."""
        # Get primary sources from the nested DifferentiableRouter
        try:
            primary_sources = self.router.router.get_primary_sources()
        except (AttributeError, RuntimeError):
            primary_sources = [0, 0]  # Default fallback
        
        # Get edge weights
        try:
            edge_weights = self.router.edge_weights.detach().tolist()
        except (AttributeError, RuntimeError):
            edge_weights = []
        
        return {
            'edge_weights': edge_weights,
            'primary_sources': primary_sources,
        }
    
    def snap_to_discrete(self):
        """Snap all meta-operations to their discrete equivalents."""
        for op in self.unary_ops:
            if hasattr(op, 'snap_to_discrete'):
                op.snap_to_discrete()
        for op in self.binary_ops:
            if hasattr(op, 'snap_to_discrete'):
                op.snap_to_discrete()
    
    def l0_regularization(self) -> torch.Tensor:
        """L0 regularization for sparsity."""
        return self.op_selector.l0_regularization()
    
    def entropy_regularization(self) -> torch.Tensor:
        """Entropy of operation selection (lower = more discrete)."""
        return self.op_selector.entropy_regularization()
    
    def zero_one_loss(self) -> torch.Tensor:
        """Zero-One Loss (FairDARTS) to close discretization gap."""
        return self.op_selector.zero_one_loss()
    
    def gate_regularization(self) -> torch.Tensor:
        """Gate regularization to force binary gates."""
        return self.op_selector.gate_regularization()
    
    def beta_decay_loss(self) -> torch.Tensor:
        """Beta-Decay regularization to prevent premature convergence."""
        return self.op_selector.beta_decay_loss()


class OperationNodeSimple(nn.Module):
    """
    Simplified operation node for faster experimentation.
    
    Uses only 3 meta-operations:
    - MetaPower (covers identity, square, sqrt, etc.)
    - MetaPeriodic (covers sin, cos)
    - MetaArithmetic (covers add, mul)
    
    Selection is done via simple softmax (not Hard Concrete).
    """
    
    def __init__(
        self,
        n_sources: int,
        node_idx: int = 0,
    ):
        super().__init__()
        self.n_sources = n_sources
        self.node_idx = node_idx
        
        # Edge weights
        self.edge_weights = nn.Parameter(torch.ones(n_sources))
        
        # Routing for binary ops (2 slots)
        self.route_logits = nn.Parameter(torch.zeros(2, n_sources))
        
        # 3 meta-ops
        self.power = MetaPower()
        self.periodic = MetaPeriodic()
        self.arithmetic = MetaArithmetic()
        
        # Selection weights: [power, periodic, arithmetic]
        self.op_logits = nn.Parameter(torch.zeros(3))
        
        # Learnable constant for constant operations
        self.constant = nn.Parameter(torch.ones(1))
    
    def forward(
        self,
        sources: torch.Tensor,
        tau: float = 1.0,
        hard: bool = True,  # Ignored for simple node, but accepted for API compatibility
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass.
        
        Args:
            sources: (batch, n_sources)
            tau: Temperature for softmax
            hard: Ignored (for API compatibility with OperationNode)
            
        Returns:
            output: (batch,)
            info: Empty dict (for API compatibility)
        """
        # Apply edge weights
        weighted = sources * self.edge_weights
        
        # Soft routing
        route_probs = F.softmax(self.route_logits / tau, dim=-1)  # (2, n_sources)
        slot1 = (weighted * route_probs[0]).sum(dim=-1)  # (batch,)
        slot2 = (weighted * route_probs[1]).sum(dim=-1)  # (batch,)
        
        # Compute all outputs
        outputs = torch.stack([
            self.power(slot1),
            self.periodic(slot1),
            self.arithmetic(slot1, slot2),
        ], dim=-1)  # (batch, 3)
        
        # Soft selection
        op_probs = F.softmax(self.op_logits / tau, dim=-1)  # (3,)
        output = (outputs * op_probs).sum(dim=-1)  # (batch,)
        
        return torch.clamp(output, -100, 100), {'node_idx': self.node_idx}
    
    def get_selected_operation(self) -> str:
        """Get selected operation."""
        idx = self.op_logits.argmax().item()
        ops = [self.power, self.periodic, self.arithmetic]
        names = ['power', 'periodic', 'arithmetic']
        return f"{names[idx]}: {ops[idx].get_discrete_op()}"
    
    def get_routing(self) -> List[int]:
        """Get routing as [slot1_source, slot2_source]."""
        return self.route_logits.argmax(dim=-1).tolist()
    
    def l0_regularization(self) -> torch.Tensor:
        """Return 0 for simple nodes (no Hard Concrete selection)."""
        return torch.tensor(0.0)
    
    def entropy_regularization(self) -> torch.Tensor:
        """Compute entropy of operation selection."""
        probs = F.softmax(self.op_logits, dim=-1)
        return -(probs * torch.log(probs + 1e-10)).sum()


class OperationLayer(nn.Module):
    """
    A layer of multiple operation nodes computed in parallel.
    
    All nodes in a layer can connect to:
    - Original inputs
    - Outputs from all previous layers
    
    This creates a dense DAG structure.
    """
    
    def __init__(
        self,
        n_sources: int,
        n_nodes: int,
        layer_idx: int = 0,
        tau: float = 0.5,
        use_simple_nodes: bool = False,
        simplified_ops: bool = False,  # Use smaller op menu
        fair_mode: bool = False,       # FairDARTS-style independent sigmoids
    ):
        """
        Args:
            n_sources: Number of sources (inputs + outputs from previous layers)
            n_nodes: Number of nodes in this layer
            layer_idx: Index of this layer
            tau: Temperature for selection
            use_simple_nodes: If True, use simplified nodes (faster)
            simplified_ops: If True, use smaller op menu (no exp/log/agg)
            fair_mode: If True, use FairDARTS independent sigmoids
        """
        super().__init__()
        self.n_sources = n_sources
        self.n_nodes = n_nodes
        self.layer_idx = layer_idx
        self.use_simple_nodes = use_simple_nodes
        self.simplified_ops = simplified_ops
        self.fair_mode = fair_mode
        
        if use_simple_nodes:
            self.nodes = nn.ModuleList([
                OperationNodeSimple(n_sources, node_idx=i)
                for i in range(n_nodes)
            ])
        else:
            self.nodes = nn.ModuleList([
                OperationNode(n_sources, node_idx=i, tau=tau, simplified_ops=simplified_ops, fair_mode=fair_mode)
                for i in range(n_nodes)
            ])
    
    def forward(
        self,
        sources: torch.Tensor,
        hard: bool = True,
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Forward through all nodes in layer.
        
        Args:
            sources: (batch, n_sources)
            hard: Use hard selection
            
        Returns:
            outputs: (batch, n_nodes)
            infos: List of info dicts per node
        """
        outputs = []
        infos = []
        
        for node in self.nodes:
            out, info = node(sources, hard=hard)
            outputs.append(out)
            infos.append(info)
        
        return torch.stack(outputs, dim=-1), infos
    
    def get_layer_summary(self) -> List[str]:
        """Get summary of operations in this layer."""
        return [node.get_selected_operation() for node in self.nodes]
    
    def l0_regularization(self) -> torch.Tensor:
        """Sum of L0 regularization across nodes."""
        return sum(node.l0_regularization() for node in self.nodes)
    
    def entropy_regularization(self) -> torch.Tensor:
        """Sum of entropy across nodes."""
        return sum(node.entropy_regularization() for node in self.nodes)
    
    def zero_one_loss(self) -> torch.Tensor:
        """Sum of Zero-One Loss across nodes (FairDARTS discretization)."""
        return sum(node.zero_one_loss() for node in self.nodes)
    
    def gate_regularization(self) -> torch.Tensor:
        """Sum of gate regularization across nodes."""
        return sum(node.gate_regularization() for node in self.nodes)
    
    def beta_decay_loss(self) -> torch.Tensor:
        """Sum of Beta-Decay regularization across nodes."""
        return sum(node.beta_decay_loss() for node in self.nodes)
