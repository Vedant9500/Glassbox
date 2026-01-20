import torch
import torch.nn as nn
from .ops_torch import OP_REGISTRY, Identity

class KANEdge(nn.Module):
    """A single edge in the KAN with a selectable operation."""
    
    def __init__(self, op_class=Identity):
        super().__init__()
        self.op = op_class()
        self.op_idx = OP_REGISTRY.index(op_class) if op_class in OP_REGISTRY else 0
    
    def forward(self, x):
        return self.op(x)
    
    def set_op(self, op_class):
        """Change the operation (for evolution)."""
        # Preserve device
        device = next(self.op.parameters()).device
        self.op = op_class().to(device)
        self.op_idx = OP_REGISTRY.index(op_class) if op_class in OP_REGISTRY else 0
    
    def __repr__(self):
        return f"Edge({self.op})"

class KANLayer(nn.Module):
    """
    A single KAN layer.
    Each input connects to each output via a KANEdge (with its own op).
    Output = sum of all edge outputs.
    """
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Create edges: [out_features][in_features]
        self.edges = nn.ModuleList([
            nn.ModuleList([KANEdge() for _ in range(in_features)])
            for _ in range(out_features)
        ])
    
    def forward(self, x):
        """
        x: (batch, in_features)
        output: (batch, out_features)
        """
        batch_size = x.shape[0]
        outputs = []
        
        for out_idx in range(self.out_features):
            # Sum contributions from all input edges
            out_sum = torch.zeros(batch_size, device=x.device)
            for in_idx in range(self.in_features):
                edge_out = self.edges[out_idx][in_idx](x[:, in_idx])
                out_sum = out_sum + edge_out
            outputs.append(out_sum)
        
        return torch.stack(outputs, dim=1)
    
    def get_formula(self, input_names):
        """Return a readable formula for this layer."""
        formulas = []
        for out_idx in range(self.out_features):
            terms = []
            for in_idx in range(self.in_features):
                edge = self.edges[out_idx][in_idx]
                terms.append(f"{edge.op.name}({input_names[in_idx]})")
            formulas.append(" + ".join(terms))
        return formulas
