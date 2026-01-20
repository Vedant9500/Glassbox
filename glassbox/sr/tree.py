"""
Differentiable Expression Tree for Symbolic Regression.

Each node represents a weighted combination of operations.
Structure and constants are learned via gradient descent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

# ============================================================
# Unary Operations (applied to a single subtree)
# ============================================================
UNARY_OPS = {
    'id':     lambda x: x,
    'neg':    lambda x: -x,
    'square': lambda x: x ** 2,
    'cube':   lambda x: torch.clamp(x ** 3, -1e6, 1e6),
    'sqrt':   lambda x: torch.sqrt(torch.abs(x) + 1e-6),
    'sin':    lambda x: torch.sin(x),
    'cos':    lambda x: torch.cos(x),
    'exp':    lambda x: torch.exp(torch.clamp(x, -10, 10)),
    'log':    lambda x: torch.log(torch.abs(x) + 1e-6),
}
UNARY_NAMES = list(UNARY_OPS.keys())

# ============================================================
# Binary Operations (combine left and right subtrees)
# ============================================================
BINARY_OPS = {
    'add': lambda l, r: l + r,
    'sub': lambda l, r: l - r,
    'mul': lambda l, r: l * r,
    'div': lambda l, r: l / (r + 1e-6 * torch.sign(r + 1e-9)),
}
BINARY_NAMES = list(BINARY_OPS.keys())


class DiffTreeNode(nn.Module):
    """
    A differentiable expression tree node.
    
    If is_leaf=True:
        Output = softmax(var_weights) @ inputs
        (Selects which input variable to use)
    
    If is_leaf=False:
        left_out = left_child(inputs)
        right_out = right_child(inputs)
        combined = softmax(binary_weights) @ [add, sub, mul, div](left, right)
        Output = softmax(unary_weights) @ [id, neg, square, ...](combined)
    
    Uses Gumbel-Softmax with hard=True to prevent NaN from ghost branches.
    This makes discrete operation selection during forward pass while keeping
    gradients flowing during backward pass.
    """
    
    # Reduced op set for stability (9 unary ops: indices 0-8)
    # We'll bias toward id (0), square (2) which are safe
    
    # Class-level temperature (can be annealed during training)
    tau = 1.0
    hard = True  # Use hard selection by default for stability
    
    def __init__(self, n_vars: int, depth: int, max_depth: int):
        super().__init__()
        self.n_vars = n_vars
        self.depth = depth
        self.max_depth = max_depth
        self.is_leaf = (depth >= max_depth)
        
        if self.is_leaf:
            # Leaf: Select which input variable (+ optional constant)
            # Bias toward variables, not constant
            init_weights = torch.zeros(n_vars + 1)
            init_weights[:n_vars] = 1.0  # Prefer variables
            self.var_weights = nn.Parameter(init_weights)
            self.constant = nn.Parameter(torch.tensor(1.0))
        else:
            # Internal node: Has two children
            self.left = DiffTreeNode(n_vars, depth + 1, max_depth)
            self.right = DiffTreeNode(n_vars, depth + 1, max_depth)
            
            # Leaf vs operator router (for hybrid mode)
            # [0] = leaf probability, [1] = operator probability  
            self.router_weights = nn.Parameter(torch.tensor([0.0, 2.0]))  # Bias toward operator for internal nodes
            
            # Weights for binary operation selection
            # Bias toward add (0) and mul (2)
            binary_init = torch.tensor([1.0, 0.0, 1.0, 0.0])  # add, sub, mul, div
            self.binary_weights = nn.Parameter(binary_init)
            
            # Weights for unary operation selection
            # Bias HEAVILY toward id (0) and square (2)
            # Order: id, neg, square, cube, sqrt, sin, cos, exp, log
            unary_init = torch.tensor([3.0, 0.0, 2.0, 0.0, 0.0, 1.0, 1.0, -5.0, -5.0])
            # exp and log are heavily penalized (-5) to avoid NaN
            self.unary_weights = nn.Parameter(unary_init)
            
            # Scale and bias
            self.scale = nn.Parameter(torch.tensor(1.0))
            self.bias = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, n_vars)
        Returns: (batch,)
        
        Uses Gumbel-Softmax with hard=True for discrete operation selection.
        Clamps intermediate values to prevent gradient explosion.
        """
        if self.is_leaf:
            # Gumbel-Softmax selection of input variable
            probs = F.gumbel_softmax(self.var_weights, tau=DiffTreeNode.tau, hard=DiffTreeNode.hard)
            # Append constant to inputs
            batch_size = x.shape[0]
            x_with_const = torch.cat([x, self.constant.expand(batch_size, 1)], dim=1)
            return (probs * x_with_const).sum(dim=1)
        else:
            # Recurse and CLAMP intermediate results to prevent explosion
            left_out = torch.clamp(self.left(x), -1e4, 1e4)
            right_out = torch.clamp(self.right(x), -1e4, 1e4)
            
            # Gumbel-Softmax hard selection of binary op
            binary_probs = F.gumbel_softmax(self.binary_weights, tau=DiffTreeNode.tau, hard=DiffTreeNode.hard)
            
            # Binary ops with safety measures
            add_result = left_out + right_out
            sub_result = left_out - right_out
            mul_result = torch.clamp(left_out * right_out, -1e4, 1e4)  # Clamp product
            # Safe division
            div_result = left_out / (right_out.abs() + 1e-4) * torch.sign(right_out + 1e-9)
            
            combined = (
                binary_probs[0] * add_result +
                binary_probs[1] * sub_result +
                binary_probs[2] * mul_result +
                binary_probs[3] * div_result
            )
            combined = torch.clamp(combined, -1e4, 1e4)  # Safety clamp
            
            # Gumbel-Softmax hard selection of unary op
            unary_probs = F.gumbel_softmax(self.unary_weights, tau=DiffTreeNode.tau, hard=DiffTreeNode.hard)
            
            # Unary ops - use REDUCED safe set for stability
            # Order: id, neg, square, cube, sqrt, sin, cos, exp, log
            output = (
                unary_probs[0] * combined +                                              # id
                unary_probs[1] * (-combined) +                                           # neg
                unary_probs[2] * torch.clamp(combined ** 2, 0, 1e4) +                   # square (clamped)
                unary_probs[3] * torch.clamp(combined ** 3, -1e4, 1e4) +               # cube (clamped)
                unary_probs[4] * torch.sqrt(torch.abs(combined) + 1e-6) +              # sqrt
                unary_probs[5] * torch.sin(combined) +                                  # sin
                unary_probs[6] * torch.cos(combined) +                                  # cos
                unary_probs[7] * torch.exp(torch.clamp(combined, -5, 5)) +             # exp (tight clamp!)
                unary_probs[8] * torch.log(torch.abs(combined) + 1e-4)                 # log
            )
            
            # Final output clamp
            output = torch.clamp(output, -1e4, 1e4)
            
            return self.scale * output + self.bias
    
    def get_formula(self, var_names: List[str], threshold: float = 0.5) -> str:
        """
        Extract a human-readable formula by taking the argmax of weights.
        """
        if self.is_leaf:
            probs = F.softmax(self.var_weights, dim=0).detach()
            best_idx = torch.argmax(probs).item()
            if best_idx < len(var_names):
                return var_names[best_idx]
            else:
                return f"{self.constant.item():.3f}"
        else:
            left_formula = self.left.get_formula(var_names, threshold)
            right_formula = self.right.get_formula(var_names, threshold)
            
            binary_probs = F.softmax(self.binary_weights, dim=0).detach()
            best_binary = BINARY_NAMES[torch.argmax(binary_probs).item()]
            
            unary_probs = F.softmax(self.unary_weights, dim=0).detach()
            best_unary = UNARY_NAMES[torch.argmax(unary_probs).item()]
            
            # Format binary
            if best_binary == 'add':
                combined = f"({left_formula} + {right_formula})"
            elif best_binary == 'sub':
                combined = f"({left_formula} - {right_formula})"
            elif best_binary == 'mul':
                combined = f"({left_formula} * {right_formula})"
            elif best_binary == 'div':
                combined = f"({left_formula} / {right_formula})"
            else:
                combined = f"{best_binary}({left_formula}, {right_formula})"
            
            # Format unary
            if best_unary == 'id':
                result = combined
            elif best_unary == 'neg':
                result = f"-{combined}"
            elif best_unary == 'square':
                result = f"({combined})²"
            elif best_unary == 'cube':
                result = f"({combined})³"
            else:
                result = f"{best_unary}({combined})"
            
            # Add scale/bias if significant
            scale = self.scale.item()
            bias = self.bias.item()
            if abs(scale - 1.0) > 0.01 or abs(bias) > 0.01:
                result = f"({scale:.3f} * {result} + {bias:.3f})"
            
            return result
    
    def complexity(self) -> int:
        """Count number of nodes in tree (for regularization)."""
        if self.is_leaf:
            return 1
        else:
            return 1 + self.left.complexity() + self.right.complexity()


class SymbolicTree(nn.Module):
    """
    Wrapper for the root DiffTreeNode.
    
    Supports Gumbel-Softmax temperature annealing for training stability.
    """
    
    def __init__(self, var_names: List[str], max_depth: int = 3):
        super().__init__()
        self.var_names = var_names
        self.n_vars = len(var_names)
        self.max_depth = max_depth
        self.root = DiffTreeNode(self.n_vars, depth=0, max_depth=max_depth)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.root(x)
    
    def get_formula(self) -> str:
        return self.root.get_formula(self.var_names)
    
    def complexity(self) -> int:
        return self.root.complexity()
    
    @staticmethod
    def set_temperature(tau: float):
        """
        Set Gumbel-Softmax temperature for all DiffTreeNodes.
        
        Higher tau (e.g., 2.0) = softer selection (more exploration)
        Lower tau (e.g., 0.1) = harder selection (more exploitation)
        
        Typical annealing: start at 2.0, decrease to 0.1 during training.
        """
        DiffTreeNode.tau = tau
    
    @staticmethod
    def set_hard_mode(hard: bool):
        """
        Enable/disable hard selection mode.
        
        hard=True (default): One-hot selection, prevents NaN from ghost branches.
        hard=False: Soft selection, may cause NaN at depth > 2.
        """
        DiffTreeNode.hard = hard
    
    @staticmethod
    def anneal_temperature(epoch: int, total_epochs: int, start_tau: float = 2.0, end_tau: float = 0.1):
        """
        Anneal temperature from start_tau to end_tau over training.
        Call this at the start of each epoch.
        """
        progress = epoch / max(total_epochs - 1, 1)
        tau = start_tau * (end_tau / start_tau) ** progress
        DiffTreeNode.tau = tau
        return tau
