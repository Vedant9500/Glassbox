import torch
import torch.nn as nn
from .layer import KANLayer
from .ops_torch import OP_REGISTRY, ProductEdge
import random
import copy

# Auto-detect device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[Glassbox] Using device: {DEVICE}")

class GlassboxKAN(nn.Module):
    """
    A KAN-style network with explicit math operations.
    Supports hybrid training: evolution for op selection, gradient for params.
    """
    
    def __init__(self, input_names, hidden_sizes=[4], output_size=1):
        super().__init__()
        self.input_names = input_names
        self.input_size = len(input_names)
        
        # Build layers
        layer_sizes = [self.input_size] + hidden_sizes + [output_size]
        self.layers = nn.ModuleList()
        
        for i in range(len(layer_sizes) - 1):
            self.layers.append(KANLayer(layer_sizes[i], layer_sizes[i+1]))
        
        # Add product edges for pairwise input interactions
        self.product_edges = nn.ModuleList()
        for i in range(self.input_size):
            for j in range(i, self.input_size):
                self.product_edges.append(ProductEdge())
        self.n_products = len(self.product_edges)
    
    def forward(self, x):
        """x: (batch, input_size)"""
        # Compute product features
        prod_idx = 0
        prod_sum = torch.zeros(x.shape[0], device=x.device)
        for i in range(self.input_size):
            for j in range(i, self.input_size):
                prod_sum = prod_sum + self.product_edges[prod_idx](x[:, i], x[:, j])
                prod_idx += 1
        
        # Normal KAN forward
        out = x
        for layer in self.layers:
            out = layer(out)
        
        # Add product contributions to output
        return out.squeeze(-1) + prod_sum
    
    def randomize_ops(self):
        """Randomly assign operations to all edges."""
        for layer in self.layers:
            for out_edges in layer.edges:
                for edge in out_edges:
                    op_class = random.choice(OP_REGISTRY)
                    edge.set_op(op_class)
    
    def mutate(self):
        """Mutate one random edge's operation."""
        layer = random.choice(self.layers)
        out_idx = random.randint(0, layer.out_features - 1)
        in_idx = random.randint(0, layer.in_features - 1)
        edge = layer.edges[out_idx][in_idx]
        new_op = random.choice(OP_REGISTRY)
        edge.set_op(new_op)
    
    def get_formula(self):
        """Return a human-readable formula representation."""
        names = self.input_names.copy()
        result = []
        for i, layer in enumerate(self.layers):
            formulas = layer.get_formula(names)
            new_names = [f"h{i}_{j}" for j in range(layer.out_features)]
            for j, f in enumerate(formulas):
                result.append(f"{new_names[j]} = {f}")
            names = new_names
        return "\n".join(result)

def _optimize_model(model, X_t, y_t, lr, steps, device):
    """Gradient descent on parameters for a fixed structure."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for _ in range(steps):
        optimizer.zero_grad()
        pred = model(X_t)
        loss = criterion(pred, y_t)
        if torch.isnan(loss) or torch.isinf(loss):
            return float('inf'), model.cpu()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    return loss.item(), model.cpu()

class KANEvolver:
    """Evolutionary optimizer for GlassboxKAN operation selection."""
    
    def __init__(self, input_names, hidden_sizes=[4], pop_size=20, lr=0.01):
        self.input_names = input_names
        self.hidden_sizes = hidden_sizes
        self.pop_size = pop_size
        self.lr = lr
        self.population = []
        self.best_model = None
        self.best_loss = float('inf')
        self.generation = 0
    
    def initialize(self):
        """Create initial population with random ops."""
        self.population = []
        for _ in range(self.pop_size):
            model = GlassboxKAN(self.input_names, self.hidden_sizes)
            model.randomize_ops()
            self.population.append(model)
    
    def evolve_step(self, X, y, opt_steps=30):
        """One generation of evolution (sequential for GPU compatibility)."""
        if isinstance(X, dict):
            X_arr = list(X.values())
            X_np = torch.tensor(list(zip(*X_arr)), dtype=torch.float32)
        else:
            X_np = torch.tensor(X, dtype=torch.float32)
        
        y_t = torch.tensor(y, dtype=torch.float32)
        
        # Move data to GPU once
        X_gpu = X_np.to(DEVICE)
        y_gpu = y_t.to(DEVICE)
        
        # Sequential evaluation (GPU handles batch parallelism)
        scored = []
        for model in self.population:
            loss, model = _optimize_model(model, X_gpu, y_gpu, self.lr, opt_steps, DEVICE)
            scored.append((loss, model))
        
        scored.sort(key=lambda x: x[0])
        self.best_loss = scored[0][0]
        self.best_model = scored[0][1]
        
        # Selection: keep top 30%
        n_keep = max(2, self.pop_size // 3)
        survivors = [copy.deepcopy(m) for _, m in scored[:n_keep]]
        
        # Fill rest with mutated copies
        new_pop = survivors.copy()
        while len(new_pop) < self.pop_size:
            parent = random.choice(survivors)
            child = copy.deepcopy(parent)
            child.mutate()
            new_pop.append(child)
        
        self.population = new_pop
        self.generation += 1
        return self.best_loss, self.best_model
