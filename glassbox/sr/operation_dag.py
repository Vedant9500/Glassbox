"""
Operation DAG for Operation-Based Neural Networks v2.

A Directed Acyclic Graph of OperationNodes that can learn:
1. Which operations each node performs
2. How nodes are connected (routing)
3. Edge constants for scaling

Key features:
- Feed-forward DAG (no RNN temporal dependencies)
- Dense connectivity (any node can connect to any previous node)
- Progressive growth (can add layers dynamically)
- Full interpretability (can extract formula from learned graph)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List

from .operation_node import OperationNode, OperationNodeSimple, OperationLayer


class OperationDAG(nn.Module):
    """
    Directed Acyclic Graph of operation nodes.
    
    Architecture:
    ```
    Input Layer (n_inputs)
          │
          ▼
    ┌─────────────────────────────────────┐
    │  Hidden Layer 1 (n_nodes each)      │
    │  [Node_0] [Node_1] ... [Node_k]     │
    └─────────────────────────────────────┘
          │ (all outputs available)
          ▼
    ┌─────────────────────────────────────┐
    │  Hidden Layer 2                      │
    │  (can connect to inputs + layer 1)  │
    └─────────────────────────────────────┘
          │
          ▼
        ...
          │
          ▼
    Output Projection
    ```
    
    Each hidden node can connect to ANY previous output (dense connectivity).
    """
    
    def __init__(
        self,
        n_inputs: int,
        n_hidden_layers: int = 2,
        nodes_per_layer: int = 4,
        n_outputs: int = 1,
        tau: float = 0.5,
        use_simple_nodes: bool = False,
    ):
        """
        Args:
            n_inputs: Number of input features
            n_hidden_layers: Number of hidden layers
            nodes_per_layer: Number of nodes per hidden layer
            n_outputs: Number of output features
            tau: Temperature for Hard Concrete selection
            use_simple_nodes: If True, use simplified nodes (faster)
        """
        super().__init__()
        self.n_inputs = n_inputs
        self.n_hidden_layers = n_hidden_layers
        self.nodes_per_layer = nodes_per_layer
        self.n_outputs = n_outputs
        self.tau = tau
        
        # Build layers
        self.layers = nn.ModuleList()
        
        current_n_sources = n_inputs
        for layer_idx in range(n_hidden_layers):
            layer = OperationLayer(
                n_sources=current_n_sources,
                n_nodes=nodes_per_layer,
                layer_idx=layer_idx,
                tau=tau,
            )
            self.layers.append(layer)
            current_n_sources += nodes_per_layer  # Add this layer's outputs
        
        # Output projection from all hidden nodes
        total_hidden = n_hidden_layers * nodes_per_layer
        self.output_proj = nn.Linear(total_hidden + n_inputs, n_outputs)
    
    def forward(
        self,
        x: torch.Tensor,
        hard: bool = True,
        return_all_outputs: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass through the DAG.
        
        Args:
            x: (batch, n_inputs) input tensor
            hard: Use hard selection for operations
            return_all_outputs: If True, return intermediate outputs
            
        Returns:
            output: (batch, n_outputs)
            info: Dict with layer info (if return_all_outputs=True)
        """
        batch_size = x.shape[0]
        
        # Collect all available sources (starts with inputs)
        all_sources = [x]
        layer_infos = []
        
        for layer in self.layers:
            # Concatenate all sources so far
            sources = torch.cat(all_sources, dim=-1)  # (batch, current_n_sources)
            
            # Forward through layer
            layer_output, layer_info = layer(sources, hard=hard)
            
            # Add to available sources
            all_sources.append(layer_output)
            layer_infos.append(layer_info)
        
        # Final output: project from all sources
        final_sources = torch.cat(all_sources, dim=-1)
        output = self.output_proj(final_sources)
        
        if return_all_outputs:
            info = {
                'layer_infos': layer_infos,
                'all_sources': [s.detach() for s in all_sources],
            }
            return output, info
        
        return output, None
    
    def get_graph_summary(self) -> str:
        """Get human-readable summary of the learned graph."""
        lines = [f"OperationDAG: {self.n_inputs} inputs → {self.n_outputs} outputs"]
        lines.append(f"Hidden: {self.n_hidden_layers} layers × {self.nodes_per_layer} nodes")
        lines.append("")
        
        for layer_idx, layer in enumerate(self.layers):
            lines.append(f"Layer {layer_idx}:")
            for node_idx, op_str in enumerate(layer.get_layer_summary()):
                lines.append(f"  Node {node_idx}: {op_str}")
        
        return "\n".join(lines)
    
    def get_formula(self, var_names: Optional[List[str]] = None) -> str:
        """
        Attempt to extract a symbolic formula from the learned graph.
        
        Note: This is approximate and works best after snap_to_discrete().
        """
        if var_names is None:
            var_names = [f"x{i}" for i in range(self.n_inputs)]
        
        # Track expressions for each node
        expressions = list(var_names)  # Start with input variables
        
        for layer_idx, layer in enumerate(self.layers):
            for node_idx, node in enumerate(layer.nodes):
                op = node.get_selected_operation()
                routing = node.get_routing_info()
                sources = routing.get('primary_sources', [0, 1])
                
                if 'binary' in op or 'arithmetic' in op:
                    # Binary operation
                    src1 = sources[0] if len(sources) > 0 else 0
                    src2 = sources[1] if len(sources) > 1 else 0
                    expr1 = expressions[src1] if src1 < len(expressions) else "?"
                    expr2 = expressions[src2] if src2 < len(expressions) else "?"
                    
                    if 'add' in op.lower():
                        new_expr = f"({expr1} + {expr2})"
                    elif 'mul' in op.lower():
                        new_expr = f"({expr1} * {expr2})"
                    else:
                        new_expr = f"{op}({expr1}, {expr2})"
                else:
                    # Unary operation
                    src = sources[0] if len(sources) > 0 else 0
                    expr = expressions[src] if src < len(expressions) else "?"
                    
                    if 'sin' in op.lower():
                        new_expr = f"sin({expr})"
                    elif 'cos' in op.lower():
                        new_expr = f"cos({expr})"
                    elif 'square' in op.lower():
                        new_expr = f"({expr})²"
                    elif 'identity' in op.lower():
                        new_expr = expr
                    elif 'sqrt' in op.lower():
                        new_expr = f"√({expr})"
                    else:
                        new_expr = f"{op.split(':')[0]}({expr})"
                
                expressions.append(new_expr)
        
        # Output is a linear combination of all expressions
        # For single output, try to identify the dominant term
        output_weights = self.output_proj.weight.data.abs()
        top_idx = output_weights[0].argmax().item()
        
        if top_idx < len(expressions):
            return expressions[top_idx]
        return "complex_expression"
    
    def snap_to_discrete(self):
        """Snap all operations to discrete equivalents."""
        for layer in self.layers:
            for node in layer.nodes:
                node.snap_to_discrete()
    
    def l0_regularization(self) -> torch.Tensor:
        """Total L0 regularization across all layers."""
        return sum(layer.l0_regularization() for layer in self.layers)
    
    def entropy_regularization(self) -> torch.Tensor:
        """Total entropy regularization."""
        return sum(layer.entropy_regularization() for layer in self.layers)


class OperationDAGSimple(nn.Module):
    """
    Simplified DAG for quick experimentation.
    
    Uses OperationNodeSimple and standard softmax selection.
    Faster training but less sophisticated selection.
    """
    
    def __init__(
        self,
        n_inputs: int,
        n_hidden: int = 8,
        n_outputs: int = 1,
    ):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        
        # Single layer of simple nodes
        self.nodes = nn.ModuleList([
            OperationNodeSimple(n_sources=n_inputs, node_idx=i)
            for i in range(n_hidden)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(n_hidden, n_outputs)
    
    def forward(
        self,
        x: torch.Tensor,
        tau: float = 1.0,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch, n_inputs)
            tau: Temperature for selection
            
        Returns:
            (batch, n_outputs)
        """
        outputs = []
        for node in self.nodes:
            outputs.append(node(x, tau=tau))
        
        hidden = torch.stack(outputs, dim=-1)  # (batch, n_hidden)
        return self.output_proj(hidden)
    
    def get_summary(self) -> str:
        """Get summary of learned operations."""
        lines = []
        for i, node in enumerate(self.nodes):
            lines.append(f"Node {i}: {node.get_selected_operation()}")
        return "\n".join(lines)


# ============================================================================
# Loss Functions
# ============================================================================

class ONNLoss(nn.Module):
    """
    Complete loss function for Operation-Based Neural Networks.
    
    L = L_mse + λ1·L1(edges) + λ2·Entropy + λ3·Complexity
    """
    
    def __init__(
        self,
        lambda_l1: float = 0.01,
        lambda_entropy: float = 0.1,
        lambda_l0: float = 0.01,
    ):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_entropy = lambda_entropy
        self.lambda_l0 = lambda_l0
        self.mse = nn.MSELoss()
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        model: OperationDAG,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute total loss.
        
        Returns:
            total_loss: Scalar tensor
            loss_components: Dict with individual loss terms
        """
        # MSE loss
        mse_loss = self.mse(pred, target)
        
        # L1 on edge weights (sparsity)
        l1_loss = torch.tensor(0.0, device=pred.device)
        for layer in model.layers:
            for node in layer.nodes:
                if hasattr(node.router, 'edge_weights'):
                    l1_loss = l1_loss + node.router.edge_weights.abs().sum()
        
        # Entropy regularization (encourage discrete selection)
        entropy_loss = -model.entropy_regularization()  # Negative because we minimize
        
        # L0 regularization (sparsity in selection)
        l0_loss = model.l0_regularization()
        
        # Total loss
        total_loss = (
            mse_loss +
            self.lambda_l1 * l1_loss +
            self.lambda_entropy * entropy_loss +
            self.lambda_l0 * l0_loss
        )
        
        loss_components = {
            'mse': mse_loss.item(),
            'l1': l1_loss.item(),
            'entropy': entropy_loss.item(),
            'l0': l0_loss.item(),
            'total': total_loss.item(),
        }
        
        return total_loss, loss_components


# ============================================================================
# Training Utilities
# ============================================================================

def train_step(
    model: OperationDAG,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: ONNLoss,
    hard: bool = True,
) -> Dict:
    """Single training step."""
    model.train()
    optimizer.zero_grad()
    
    pred, _ = model(x, hard=hard)
    loss, components = loss_fn(pred, y, model)
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return components


def train_onn(
    model: OperationDAG,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int = 500,
    lr: float = 0.01,
    print_every: int = 50,
    lambda_l1: float = 0.01,
    lambda_entropy: float = 0.1,
    anneal_tau: bool = True,
    tau_start: float = 2.0,
    tau_end: float = 0.2,
) -> Dict:
    """
    Train an Operation DAG.
    
    Args:
        model: OperationDAG instance
        x_train: Training inputs (n_samples, n_inputs)
        y_train: Training targets (n_samples, n_outputs)
        epochs: Number of training epochs
        lr: Learning rate
        print_every: Print frequency
        lambda_l1: L1 regularization weight
        lambda_entropy: Entropy regularization weight
        anneal_tau: Whether to anneal temperature
        tau_start: Starting temperature
        tau_end: Ending temperature
        
    Returns:
        Training history dict
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = ONNLoss(lambda_l1=lambda_l1, lambda_entropy=lambda_entropy)
    
    history = {'mse': [], 'total': []}
    best_mse = float('inf')
    
    for epoch in range(epochs):
        # Anneal temperature
        if anneal_tau:
            progress = epoch / max(epochs - 1, 1)
            tau = tau_end + 0.5 * (tau_start - tau_end) * (1 + torch.cos(torch.tensor(torch.pi * progress)))
            model.tau = tau.item()
            for layer in model.layers:
                for node in layer.nodes:
                    node.tau = model.tau
        
        # Training step
        components = train_step(model, optimizer, x_train, y_train, loss_fn)
        
        history['mse'].append(components['mse'])
        history['total'].append(components['total'])
        
        if components['mse'] < best_mse:
            best_mse = components['mse']
        
        if epoch % print_every == 0:
            tau_str = f"τ={model.tau:.2f}" if anneal_tau else ""
            print(f"Epoch {epoch:4d} | MSE: {components['mse']:.4f} | Total: {components['total']:.4f} | {tau_str}")
    
    print(f"\nBest MSE: {best_mse:.4f}")
    print("\nLearned Operations:")
    print(model.get_graph_summary())
    
    return history
