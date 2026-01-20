import torch
import torch.nn as nn
import torch.nn.functional as F
from .ops_torch import OP_REGISTRY

# Module-level Gumbel-Softmax settings (can be controlled globally)
_GUMBEL_TAU = 1.0
_GUMBEL_HARD = True

def set_gumbel_temperature(tau: float):
    """Set Gumbel-Softmax temperature globally for all DifferentiableEdges."""
    global _GUMBEL_TAU
    _GUMBEL_TAU = tau

def set_gumbel_hard(hard: bool):
    """Enable/disable hard selection for all DifferentiableEdges."""
    global _GUMBEL_HARD
    _GUMBEL_HARD = hard

def get_gumbel_settings():
    """Get current Gumbel-Softmax settings."""
    return _GUMBEL_TAU, _GUMBEL_HARD


class DifferentiableEdge(nn.Module):
    """
    Computes weighted sum of candidate operations using Gumbel-Softmax.
    
    With hard=True (default):
        - Forward: One-hot selection (only ONE operation is computed)
        - Backward: Gradients flow through all operations (soft)
        
    This prevents NaN from ghost branches (e.g., log(negative) when
    log has near-zero weight but is still computed).
    """
    
    def __init__(self, candidates=None):
        super().__init__()
        self.ops = nn.ModuleList([op() for op in (candidates or OP_REGISTRY)])
        # Architecture weights (alpha)
        self.weights = nn.Parameter(torch.zeros(len(self.ops)))
        
    def forward(self, x):
        # Gumbel-Softmax with hard selection
        # hard=True: one-hot in forward, soft in backward
        probs = F.gumbel_softmax(self.weights, tau=_GUMBEL_TAU, hard=_GUMBEL_HARD)
        
        # Weighted sum of all op outputs
        # With hard=True, only the selected op contributes (others multiplied by 0)
        result = 0.0
        for i, op in enumerate(self.ops):
            result = result + probs[i] * op(x)
            
        return result
    
    def prune(self):
        """Returns the single best operation (and its index)."""
        avg_weights = self.weights.detach()
        best_idx = torch.argmax(avg_weights).item()
        return self.ops[best_idx], best_idx, avg_weights

class DifferentiableLayer(nn.Module):
    """KAN Layer where every edge is a DifferentiableEdge."""
    
    def __init__(self, in_features, out_features, candidates=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.edges = nn.ModuleList([
            nn.ModuleList([DifferentiableEdge(candidates) for _ in range(in_features)])
            for _ in range(out_features)
        ])
        
    def forward(self, x):
        batch_size = x.shape[0]
        outputs = []
        for out_idx in range(self.out_features):
            out_sum = torch.zeros(batch_size, device=x.device)
            for in_idx in range(self.in_features):
                edge_out = self.edges[out_idx][in_idx](x[:, in_idx])
                out_sum = out_sum + edge_out
            outputs.append(out_sum)
        
        return torch.stack(outputs, dim=1)

class DifferentiableProductEdge(nn.Module):
    """
    Computes weighted product of two inputs.
    Similar to DifferentiableEdge, but fixed op is Mul.
    We just learn IF this edge should exist (via weight).
    """
    def __init__(self):
        super().__init__()
        self.scale1 = nn.Parameter(torch.tensor(0.1))
        self.scale2 = nn.Parameter(torch.tensor(0.1))
        # Architecture weight: how much to use this product?
        self.weight = nn.Parameter(torch.tensor(0.0)) # Start low
        
    def forward(self, x1, x2):
        # Sigmoid gate [0, 1]
        gate = torch.sigmoid(self.weight)
        return gate * (self.scale1 * x1 * self.scale2 * x2)

