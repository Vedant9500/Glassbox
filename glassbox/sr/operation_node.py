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
    ):
        """
        Args:
            n_sources: Number of potential input sources (previous nodes + inputs)
            node_idx: Index of this node (for debugging/visualization)
            tau: Temperature for Hard Concrete
            beta: Stretch parameter for Hard Concrete
        """
        super().__init__()
        self.n_sources = n_sources
        self.node_idx = node_idx
        self.tau = tau
        self.beta = beta
        
        # Routing: handles edge weights and input slot selection
        self.router = AdaptiveArityRouter(n_sources, max_arity=2)
        
        # Operation selection via Hard Concrete
        self.op_selector = HardConcreteOperationSelector(
            n_unary=4,    # periodic, power, exp, log
            n_binary=2,   # arithmetic, aggregation
            tau=tau,
            beta=beta,
        )
        
        # Meta-operations (unary)
        self.unary_ops = nn.ModuleList([
            MetaPeriodic(),
            MetaPower(),
            MetaExp(),
            MetaLog(),
        ])
        
        # Meta-operations (binary)
        self.binary_ops = nn.ModuleList([
            MetaArithmetic(),
            MetaAggregation(),
        ])
        
        # Output BatchNorm for gradient normalization
        self.output_norm = nn.BatchNorm1d(1, affine=False)
        
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
        
        # Get operation selection weights
        type_weights, unary_weights, binary_weights = self.op_selector(hard=hard)
        
        # Compute unary outputs
        unary_input = self.router.forward_unary(sources, tau=self.tau, hard=hard)
        unary_outputs = []
        for i, op in enumerate(self.unary_ops):
            out = op(unary_input)
            unary_outputs.append(out * unary_weights[i])
        unary_result = sum(unary_outputs)
        
        # Compute binary outputs
        x, y = self.router.forward_binary(sources, tau=self.tau, hard=hard)
        binary_outputs = []
        
        # Arithmetic operation
        binary_outputs.append(self.binary_ops[0](x, y) * binary_weights[0])
        
        # Aggregation operation (needs all weighted sources)
        agg_input = self.router.forward_aggregation(sources)
        binary_outputs.append(self.binary_ops[1](agg_input, dim=-1) * binary_weights[1])
        
        binary_result = sum(binary_outputs)
        
        # Combine unary and binary based on type selection
        output = type_weights[0] * unary_result + type_weights[1] * binary_result
        
        # Normalize output
        if normalize and batch_size > 1:
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
        return {
            'edge_weights': self.router.edge_weights.tolist() if hasattr(self.router, 'edge_weights') else [],
            'primary_sources': self.router.router.get_primary_sources(),
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
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            sources: (batch, n_sources)
            tau: Temperature for softmax
            
        Returns:
            output: (batch,)
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
        
        return torch.clamp(output, -100, 100)
    
    def get_selected_operation(self) -> str:
        """Get selected operation."""
        idx = self.op_logits.argmax().item()
        ops = [self.power, self.periodic, self.arithmetic]
        names = ['power', 'periodic', 'arithmetic']
        return f"{names[idx]}: {ops[idx].get_discrete_op()}"
    
    def get_routing(self) -> List[int]:
        """Get routing as [slot1_source, slot2_source]."""
        return self.route_logits.argmax(dim=-1).tolist()


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
    ):
        """
        Args:
            n_sources: Number of sources (inputs + outputs from previous layers)
            n_nodes: Number of nodes in this layer
            layer_idx: Index of this layer
            tau: Temperature for selection
        """
        super().__init__()
        self.n_sources = n_sources
        self.n_nodes = n_nodes
        self.layer_idx = layer_idx
        
        self.nodes = nn.ModuleList([
            OperationNode(n_sources, node_idx=i, tau=tau)
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
