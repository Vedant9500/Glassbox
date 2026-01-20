import torch
import torch.nn as nn
import math

class SymbolicOp(nn.Module):
    """Base class for symbolic operations with learnable scale/bias."""
    name = "Op"
    
    def __init__(self):
        super().__init__()
        # Initialize scale close to 1, bias close to 0 (small perturbation)
        self.scale = nn.Parameter(torch.tensor(0.1))
        self.bias = nn.Parameter(torch.tensor(0.0))
    
    def transform(self, x):
        """Apply scale and bias before operation."""
        return self.scale * x + self.bias
    
    def op(self, x):
        raise NotImplementedError
    
    def forward(self, x):
        return self.op(self.transform(x))
    
    def __repr__(self):
        return f"{self.name}({self.scale.item():.3f}*x + {self.bias.item():.3f})"

# --- Identity (simple linear) ---
class Identity(SymbolicOp):
    name = "id"
    def op(self, x):
        return x

# --- Power Operations (safe) ---
class Square(SymbolicOp):
    name = "x²"
    def op(self, x):
        return x ** 2

class Cube(SymbolicOp):
    name = "x³"
    def op(self, x):
        return torch.clamp(x, -10, 10) ** 3  # Prevent explosion

# --- Trigonometric (bounded output) ---
class Sin(SymbolicOp):
    name = "sin"
    def op(self, x):
        return torch.sin(x)

class Cos(SymbolicOp):
    name = "cos"
    def op(self, x):
        return torch.cos(x)

# --- Exponential/Log (VERY careful) ---
class Exp(SymbolicOp):
    name = "exp"
    def op(self, x):
        # Very aggressive clipping to prevent overflow
        return torch.exp(torch.clamp(x, -5, 5))

class Log(SymbolicOp):
    name = "log"
    def op(self, x):
        return torch.log(torch.abs(x) + 1e-4)

# --- Multiplication (for combining inputs) ---
class Mul(SymbolicOp):
    """Special op that takes product."""
    name = "mul"
    def op(self, x):
        # This is just identity but marked for special handling
        return x

# Registry - REMOVE Exp for now (too unstable), keep safe ops
OP_REGISTRY = [Identity, Square, Sin, Cos]  # Start simple
SAFE_OPS = [Identity, Square, Sin, Cos, Cube, Log]

# --- Product Edge (multiplies two inputs) ---
class ProductEdge(nn.Module):
    """Multiplies two inputs with learnable scales."""
    name = "prod"
    
    def __init__(self):
        super().__init__()
        self.scale1 = nn.Parameter(torch.tensor(0.1))
        self.scale2 = nn.Parameter(torch.tensor(0.1))
        self.bias = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, x1, x2):
        return self.scale1 * x1 * self.scale2 * x2 + self.bias
    
    def __repr__(self):
        return f"prod({self.scale1.item():.3f}*a * {self.scale2.item():.3f}*b + {self.bias.item():.3f})"
