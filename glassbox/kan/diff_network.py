import torch
import torch.nn as nn
import torch.nn.functional as F
from .diff_layer import DifferentiableLayer, DifferentiableProductEdge, set_gumbel_temperature, set_gumbel_hard
from .ops_torch import SAFE_OPS

class GlassboxDiffKAN(nn.Module):
    """
    Differentiable KAN with Mixing Layers.
    Structure:
    [Layer 1 Ops] -> [Hidden Features] -> [Layer 2 Products] -> Output
    """
    
    def __init__(self, input_names, hidden_sizes=[4], output_size=1):
        super().__init__()
        self.input_names = input_names
        
        # Layer 1: Transformation (Generates terms like v^2, log(m), etc.)
        self.layer1 = DifferentiableLayer(len(input_names), hidden_sizes[0], candidates=SAFE_OPS)
        
        # Layer 2: Mixing (Weighted Sum + Weighted Products of Hidden Features)
        # We want to allow the network to pick: h1 + h2 OR h1 * h2
        self.layer2_linear = nn.Linear(hidden_sizes[0], output_size, bias=False)
        
        # Quadratic Mixing: All pairs of hidden features
        self.n_hidden = hidden_sizes[0]
        self.product_edges = nn.ModuleList()
        for i in range(self.n_hidden):
            for j in range(i, self.n_hidden):
                self.product_edges.append(DifferentiableProductEdge())
                
    def forward(self, x):
        # 1. Transform inputs -> Hidden Features
        # h dimensions: (batch, n_hidden)
        h = self.layer1(x).squeeze(-1) # DifferentiableLayer returns (batch, out, 1) -> squeeze to (batch, out)
        
        # 2. Linear Combination
        linear_out = self.layer2_linear(h)
        
        # 3. Product Combination
        prod_sum = 0
        prod_idx = 0
        for i in range(self.n_hidden):
            for j in range(i, self.n_hidden):
                prod_sum = prod_sum + self.product_edges[prod_idx](h[:, i], h[:, j])
                prod_idx += 1
                
        # Result is (Linear Sum) + (Product Sum)
        # This allows: c1*h1 + c2*h2 + c3*(h1*h2)
        # If h1=m, h2=v^2, then h1*h2 = m*v^2. BOOM.
        return linear_out.squeeze(-1) + prod_sum

    def entropy_loss(self):
        loss = 0
        # Layer 1 Ops
        for row in self.layer1.edges:
            for edge in row:
                probs = F.softmax(edge.weights, dim=0)
                entropy = -torch.sum(probs * torch.log(probs + 1e-7))
                loss += entropy
                    
        # L1 Regularization for Product Gates
        for edge in self.product_edges:
            gate = torch.sigmoid(edge.weight)
            loss += gate # Minimize the number of active products
            
        return loss

    def prune_and_print(self):
        print("\n--- Discovered Formula Structure ---")
        
        # Layer 1: What are the hidden features?
        h_formulas = []
        for out_idx in range(self.n_hidden):
            terms = []
            for in_idx in range(self.layer1.in_features):
                edge = self.layer1.edges[out_idx][in_idx]
                best_op, idx, weights = edge.prune()
                prob = F.softmax(weights, dim=0)[idx].item()
                if prob > 0.1:
                    input_name = self.input_names[in_idx]
                    terms.append(f"{best_op}({input_name})") # Use input name instead of generic 'x'
            f = " + ".join(terms) if terms else "0"
            h_formulas.append(f"({f})")
            print(f"h_{out_idx} = {f}")
            
        print("\n--- Output Combination ---")
        # Products
        prod_idx = 0
        for i in range(self.n_hidden):
            for j in range(i, self.n_hidden):
                edge = self.product_edges[prod_idx]
                gate = torch.sigmoid(edge.weight).item()
                if gate > 0.1:
                    scale = edge.scale1.item() * edge.scale2.item()
                    print(f" + [{gate:.2f}] * {scale:.2f} * h_{i} * h_{j}")
                prod_idx += 1
