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
        self.use_simple_nodes = use_simple_nodes
        
        # Build layers
        self.layers = nn.ModuleList()
        
        current_n_sources = n_inputs
        for layer_idx in range(n_hidden_layers):
            layer = OperationLayer(
                n_sources=current_n_sources,
                n_nodes=nodes_per_layer,
                layer_idx=layer_idx,
                tau=tau,
                use_simple_nodes=use_simple_nodes,
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
        
        # Track expressions for each node position in the graph
        # Index 0 to n_inputs-1 are input variables
        # Then each layer adds nodes_per_layer expressions
        expressions = list(var_names)  # Start with input variables
        
        for layer_idx, layer in enumerate(self.layers):
            for node_idx, node in enumerate(layer.nodes):
                # Get the selected operation info
                op_str = node.get_selected_operation()
                routing_info = node.get_routing_info()
                primary_sources = routing_info.get('primary_sources', [0, 0])
                
                # Get the source expressions (clamp to valid indices)
                src1_idx = min(primary_sources[0], len(expressions) - 1)
                src2_idx = min(primary_sources[1] if len(primary_sources) > 1 else 0, len(expressions) - 1)
                src1_expr = expressions[src1_idx] if src1_idx < len(expressions) else var_names[0]
                src2_expr = expressions[src2_idx] if src2_idx < len(expressions) else var_names[0]
                
                # Parse the operation string to build expression
                op_lower = op_str.lower()
                
                # Unary operations
                if 'power:' in op_lower:
                    # Extract the specific power operation
                    if 'square' in op_lower:
                        new_expr = f"({src1_expr})²"
                    elif 'cube' in op_lower:
                        new_expr = f"({src1_expr})³"
                    elif 'sqrt' in op_lower:
                        new_expr = f"√({src1_expr})"
                    elif 'identity' in op_lower:
                        new_expr = src1_expr
                    elif 'reciprocal' in op_lower:
                        new_expr = f"1/({src1_expr})"
                    else:
                        # Try to extract power value
                        import re
                        p_match = re.search(r'p=([0-9.-]+)', op_lower)
                        if p_match:
                            p_val = float(p_match.group(1))
                            if abs(p_val - 2.0) < 0.15:
                                new_expr = f"({src1_expr})²"
                            elif abs(p_val - 1.0) < 0.15:
                                new_expr = src1_expr
                            elif abs(p_val - 0.5) < 0.15:
                                new_expr = f"√({src1_expr})"
                            else:
                                new_expr = f"({src1_expr})^{p_val:.1f}"
                        else:
                            new_expr = f"pow({src1_expr})"
                            
                elif 'periodic:' in op_lower:
                    if 'sin' in op_lower and '-sin' not in op_lower:
                        new_expr = f"sin({src1_expr})"
                    elif 'cos' in op_lower and '-cos' not in op_lower:
                        new_expr = f"cos({src1_expr})"
                    elif '-sin' in op_lower:
                        new_expr = f"-sin({src1_expr})"
                    elif '-cos' in op_lower:
                        new_expr = f"-cos({src1_expr})"
                    else:
                        new_expr = f"periodic({src1_expr})"
                        
                elif 'exp:' in op_lower:
                    if 'constant' in op_lower:
                        new_expr = "1"
                    elif '2^' in op_lower:
                        new_expr = f"2^({src1_expr})"
                    else:
                        new_expr = f"exp({src1_expr})"
                        
                elif 'log:' in op_lower:
                    if 'ln' in op_lower:
                        new_expr = f"ln({src1_expr})"
                    elif 'log2' in op_lower:
                        new_expr = f"log2({src1_expr})"
                    elif 'log10' in op_lower:
                        new_expr = f"log10({src1_expr})"
                    else:
                        new_expr = f"log({src1_expr})"
                
                # Binary operations
                elif 'arithmetic:' in op_lower:
                    if 'add' in op_lower:
                        new_expr = f"({src1_expr} + {src2_expr})"
                    elif 'multiply' in op_lower or 'mul' in op_lower:
                        new_expr = f"({src1_expr} * {src2_expr})"
                    else:
                        new_expr = f"arith({src1_expr}, {src2_expr})"
                        
                elif 'aggregation:' in op_lower:
                    if 'max' in op_lower:
                        new_expr = f"max({src1_expr}, {src2_expr})"
                    elif 'mean' in op_lower:
                        new_expr = f"mean({src1_expr}, {src2_expr})"
                    elif 'sum' in op_lower:
                        new_expr = f"sum({src1_expr}, {src2_expr})"
                    else:
                        new_expr = f"agg({src1_expr})"
                else:
                    # Fallback: use the operation name
                    # Safely handle cases without colon
                    if ':' in op_str:
                        op_name = op_str.split(':')[0]
                    else:
                        op_name = op_str
                    new_expr = f"{op_name}({src1_expr})"
                
                expressions.append(new_expr)
        
        # Now analyze output projection to find most influential expressions
        output_weights = self.output_proj.weight.data[0].cpu()  # First output
        output_bias = self.output_proj.bias.data[0].item() if self.output_proj.bias is not None else 0
        
        # Build formula from weighted contributions
        formula_parts = []
        weight_threshold = 0.05  # Lower threshold to catch more contributions
        
        # Collect all significant contributions
        contributions = []
        for idx in range(len(expressions)):
            if idx < len(output_weights):
                weight = output_weights[idx].item()
                if abs(weight) > weight_threshold:
                    contributions.append((abs(weight), weight, idx, expressions[idx]))
        
        # Sort by absolute weight (descending)
        contributions.sort(reverse=True, key=lambda x: x[0])
        
        # Take top contributors (up to 5)
        for _, weight, idx, expr in contributions[:5]:
            # Skip if it's just a raw input variable AND there are non-trivial expressions
            if expr in var_names and len(contributions) > 1:
                # Check if there's a better expression available
                has_better = any(c[3] not in var_names for c in contributions[:5])
                if has_better:
                    continue
            
            if abs(weight - 1.0) < 0.1:
                formula_parts.append(expr)
            elif abs(weight + 1.0) < 0.1:
                formula_parts.append(f"-{expr}")
            elif weight > 0:
                formula_parts.append(f"{weight:.2f}*{expr}")
            else:
                formula_parts.append(f"({weight:.2f}*{expr})")
        
        # Add bias if significant
        if abs(output_bias) > weight_threshold:
            if output_bias > 0:
                formula_parts.append(f"{output_bias:.2f}")
            else:
                formula_parts.append(f"({output_bias:.2f})")
        
        if not formula_parts:
            # Fallback: return the last computed expression
            non_trivial = [e for e in expressions if e not in var_names]
            if non_trivial:
                return non_trivial[-1]
            return expressions[-1] if expressions else "?"
        
        raw_formula = " + ".join(formula_parts)
        return self._simplify_formula(raw_formula)
    
    def _simplify_formula(self, formula_str: str) -> str:
        """Simplify formula using sympy."""
        try:
            from sympy import symbols, sympify, simplify
            
            # Create symbols for input variables
            local_dict = {f"x{i}": symbols(f"x{i}") for i in range(self.n_inputs)}
            
            # Add other common functions to local dict to be safe
            import sympy
            local_dict.update({
                'sin': sympy.sin,
                'cos': sympy.cos,
                'exp': sympy.exp,
                'log': sympy.log,
                'ln': sympy.log,
                'sqrt': sympy.sqrt,
                'pow': sympy.Pow,
            })
            
            # Parse and simplify
            # Handle some custom names manually
            formula_str = formula_str.replace("max(", "Max(")
            
            # Handle float precision to avoid huge decimals
            import re
            # Replace 0.00*x with 0
            formula_str = re.sub(r'0\.00\*\w+', '0', formula_str)
            
            expr = sympify(formula_str, locals=local_dict)
            simplified = simplify(expr)
            
            # Round floats in the simplified expression for readability
            # This is a bit tricky with sympy, so we'll just return the simplified string
            return str(simplified).replace('**', '^')
        except Exception as e:
            # If simplification fails (e.g. unknown function), return raw
            return formula_str

    
    def snap_to_discrete(self):
        """Snap all operations to discrete equivalents."""
        for layer in self.layers:
            for node in layer.nodes:
                node.snap_to_discrete()
    
    def compile_for_inference(self):
        """
        Compile the DAG for fast inference by caching selections.
        
        After training, call this method to:
        1. Cache the selected operation for each node
        2. Cache the routing (which sources to use)
        3. Enable fast-path inference that skips selection logic
        
        This can provide 5-10x speedup for inference.
        
        Returns self for chaining: dag.snap_to_discrete().compile_for_inference().eval()
        """
        self._compiled = True
        self._compiled_ops = []
        
        for layer_idx, layer in enumerate(self.layers):
            layer_ops = []
            for node in layer.nodes:
                # Get deterministic selection
                selection = node.op_selector.get_selected()
                
                # Get routing info 
                try:
                    routing = node.router.router.get_primary_sources()
                except (AttributeError, RuntimeError):
                    routing = [0, 0]
                
                # Get edge weights
                try:
                    edge_weights = node.router.edge_weights.detach().clone()
                except (AttributeError, RuntimeError):
                    edge_weights = None
                
                # Get the actual operation callable
                if selection['type'] == 'unary':
                    op_idx = selection['unary_idx']
                    op = node.unary_ops[op_idx]
                    op_type = 'unary'
                else:
                    op_idx = selection['binary_idx']
                    op = node.binary_ops[op_idx]
                    op_type = 'binary' if op_idx == 0 else 'aggregation'
                
                layer_ops.append({
                    'op': op,
                    'op_type': op_type,
                    'routing': routing,
                    'edge_weights': edge_weights,
                    'output_scale': node.output_scale.detach().clone(),
                })
            self._compiled_ops.append(layer_ops)
        
        return self
    
    def forward_compiled(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fast forward pass using compiled operations.
        
        Must call compile_for_inference() first.
        
        Args:
            x: (batch, n_inputs) input tensor
            
        Returns:
            output: (batch, n_outputs)
        """
        if not hasattr(self, '_compiled') or not self._compiled:
            raise RuntimeError("Call compile_for_inference() before forward_compiled()")
        
        batch_size = x.shape[0]
        all_sources = [x]
        
        for layer_idx, layer_ops in enumerate(self._compiled_ops):
            # Concatenate all sources so far
            sources = torch.cat(all_sources, dim=-1)  # (batch, current_n_sources)
            
            layer_outputs = []
            for node_info in layer_ops:
                # Apply edge weights
                if node_info['edge_weights'] is not None:
                    weighted = sources * node_info['edge_weights']
                else:
                    weighted = sources
                
                routing = node_info['routing']
                op = node_info['op']
                op_type = node_info['op_type']
                
                if op_type == 'unary':
                    # Get single input
                    inp = weighted[:, routing[0]]
                    out = op(inp)
                elif op_type == 'binary':
                    # Get two inputs
                    inp1 = weighted[:, routing[0]]
                    inp2 = weighted[:, routing[1]] if len(routing) > 1 else inp1
                    out = op(inp1, inp2)
                else:  # aggregation
                    out = op(weighted, dim=-1)
                
                # Apply output scale and clamp
                out = out * node_info['output_scale']
                out = torch.clamp(out, -100, 100)
                
                layer_outputs.append(out)
            
            # Stack layer outputs
            layer_output = torch.stack(layer_outputs, dim=-1)
            all_sources.append(layer_output)
        
        # Final output
        final_sources = torch.cat(all_sources, dim=-1)
        return self.output_proj(final_sources)
    
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
                    # Also update op_selector tau for proper annealing
                    if hasattr(node, 'op_selector') and hasattr(node.op_selector, 'set_tau'):
                        node.op_selector.set_tau(model.tau)
        
        # Use soft selection early (exploration), hard selection later (exploitation)
        progress = epoch / max(epochs - 1, 1)
        use_hard = progress > 0.7
        
        # Training step
        components = train_step(model, optimizer, x_train, y_train, loss_fn, hard=use_hard)
        
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
