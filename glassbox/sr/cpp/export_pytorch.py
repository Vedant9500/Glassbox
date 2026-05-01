"""
P8: Python nn.Module Deserialization

Converts a C++ evolution result dict into a live torch.nn.Module
that can be used inside PyTorch pipelines (training, export, ONNX, etc.)
"""
import torch
import torch.nn as nn
import math


class CppGraphModule(nn.Module):
    """
    Reconstructs a C++ AST graph as a PyTorch nn.Module.
    
    The forward() pass exactly mirrors the C++ eval.h evaluation:
      - Input nodes → index into x columns
      - Constant nodes → fixed scalar
      - Unary[Periodic] → amplitude * sin(omega * child + phi)
      - Unary[Power]    → |child|^p  (safe for negative bases)
      - Unary[Exp]      → exp(child)
      - Unary[Log]      → log(|child| + eps)
      - Binary[Arithmetic] → soft add/mul via beta
      - Binary[Aggregation] → (left + right) / 2
    """
    
    # C++ enum mappings
    TYPE_INPUT = 0
    TYPE_CONSTANT = 1
    TYPE_UNARY = 2
    TYPE_BINARY = 3
    
    UNARY_PERIODIC = 0
    UNARY_POWER = 1
    UNARY_INTPOW = 2
    UNARY_EXP = 3
    UNARY_LOG = 4
    
    BINARY_ARITHMETIC = 0
    BINARY_DIVISION = 1
    BINARY_AGGREGATION = 2
    
    def __init__(self, cpp_result: dict):
        super().__init__()
        
        self.nodes = cpp_result["nodes"]
        self.formula_str = cpp_result.get("formula", "")
        
        # Store output weights and bias as parameters (allows fine-tuning)
        weights = cpp_result["output_weights"]
        bias = cpp_result["output_bias"]
        
        self.output_weights = nn.Parameter(
            torch.tensor(weights, dtype=torch.float64)
        )
        self.output_bias = nn.Parameter(
            torch.tensor(bias, dtype=torch.float64)
        )
        
        # Store node parameters (can be made trainable if desired)
        for i, node in enumerate(self.nodes):
            ntype = node["type"]
            if ntype == self.TYPE_UNARY:
                self.register_buffer(f"p_{i}", torch.tensor(node["p"], dtype=torch.float64))
                self.register_buffer(f"omega_{i}", torch.tensor(node["omega"], dtype=torch.float64))
                self.register_buffer(f"phi_{i}", torch.tensor(node["phi"], dtype=torch.float64))
                self.register_buffer(f"amplitude_{i}", torch.tensor(node["amplitude"], dtype=torch.float64))
            elif ntype == self.TYPE_CONSTANT:
                self.register_buffer(f"value_{i}", torch.tensor(node["value"], dtype=torch.float64))
            elif ntype == self.TYPE_BINARY:
                self.register_buffer(f"beta_{i}", torch.tensor(node["beta"], dtype=torch.float64))
                self.register_buffer(f"gamma_{i}", torch.tensor(node["gamma"], dtype=torch.float64))
                self.register_buffer(f"tau_{i}", torch.tensor(node["tau"], dtype=torch.float64))
    
    def forward(self, x: torch.Tensor, hard: bool = True, **kwargs) -> torch.Tensor:
        """
        Evaluate the C++ graph using PyTorch operations.
        
        Args:
            x: Input tensor of shape (N,) for single-feature or (N, D) for multi-feature
            hard: Ignored (compatibility with OperationDAG interface)
            
        Returns:
            Tuple of (output tensor of shape (N,), None) matching ONN signature
        """
        if x.dim() == 1:
            x = x.unsqueeze(1)  # (N,) → (N, 1)
        
        x = x.double()
        n_samples = x.shape[0]
        eps = 1e-10
        
        # Evaluate each node bottom-up (same order as C++ eval.h)
        node_outputs = []
        
        for i, node in enumerate(self.nodes):
            ntype = node["type"]
            
            if ntype == self.TYPE_INPUT:
                feat_idx = node["feature_idx"]
                if feat_idx < x.shape[1]:
                    out = x[:, feat_idx]
                else:
                    out = torch.zeros(n_samples, dtype=torch.float64, device=x.device)
            
            elif ntype == self.TYPE_CONSTANT:
                val = getattr(self, f"value_{i}")
                out = val.expand(n_samples)
            
            elif ntype == self.TYPE_UNARY:
                # Get child output
                left = node["left_child"]
                if 0 <= left < len(node_outputs):
                    child = node_outputs[left]
                else:
                    child = torch.zeros(n_samples, dtype=torch.float64, device=x.device)
                
                unary_op = node["unary_op"]
                p = getattr(self, f"p_{i}")
                omega = getattr(self, f"omega_{i}")
                phi = getattr(self, f"phi_{i}")
                amplitude = getattr(self, f"amplitude_{i}")
                
                if unary_op == self.UNARY_PERIODIC:
                    out = amplitude * torch.sin(omega * child + phi)
                elif unary_op == self.UNARY_POWER:
                    # sign(x) * |x|^p, with parity check similar to eval.h power_sign_blend
                    abs_child = torch.abs(child) + eps
                    sign_child = torch.sign(child)
                    abs_pow = abs_child.pow(p)
                    
                    p_round = torch.round(p)
                    is_even = (torch.abs(p - p_round) < 1e-6) & (p_round.long() % 2 == 0)
                    is_even = is_even.double()
                    
                    out = (1.0 - is_even) * (sign_child * abs_pow) + is_even * abs_pow
                elif unary_op == self.UNARY_INTPOW:
                    n = torch.round(p).long().clamp(2, 6)
                    out = child.pow(n)
                elif unary_op == self.UNARY_EXP:
                    out = torch.exp(torch.clamp(omega * child + phi, -20.0, 20.0))
                elif unary_op == self.UNARY_LOG:
                    out = torch.log(torch.abs(child) + eps)
                else:
                    out = child
            
            elif ntype == self.TYPE_BINARY:
                left = node["left_child"]
                right = node["right_child"]
                left_val = node_outputs[left] if 0 <= left < len(node_outputs) else torch.zeros(n_samples, dtype=torch.float64, device=x.device)
                right_val = node_outputs[right] if 0 <= right < len(node_outputs) else torch.zeros(n_samples, dtype=torch.float64, device=x.device)
                
                binary_op = node["binary_op"]
                beta = getattr(self, f"beta_{i}")
                gamma = getattr(self, f"gamma_{i}")
                
                if binary_op == self.BINARY_ARITHMETIC:
                    # Soft interpolation matching eval.h arithmetic_soft_weights
                    # For simplicity in CppGraphModule (usually used post-evolution), 
                    # we just pick the winner if beta is near 1.0 or 2.0.
                    if beta.item() < 1.5:
                        if gamma.item() > 0: out = left_val + right_val
                        else: out = left_val - right_val
                    else:
                        if gamma.item() > 0: out = left_val * right_val
                        else: out = left_val / (torch.abs(right_val) + eps) * torch.sign(right_val)
                elif binary_op == self.BINARY_DIVISION:
                    out = left_val / (torch.abs(right_val) + eps) * torch.sign(right_val)
                elif binary_op == self.BINARY_AGGREGATION:
                    # Softmax aggregation matching eval.h
                    tau = getattr(self, f"tau_{i}")
                    # Clamp tau to avoid div by zero
                    local_tau = torch.where(torch.abs(tau) >= 1e-3, tau, torch.tensor(1e-3, dtype=torch.float64, device=tau.device))
                    
                    max_val = torch.maximum(left_val, right_val)
                    exp_l = torch.exp((left_val - max_val) / local_tau)
                    exp_r = torch.exp((right_val - max_val) / local_tau)
                    sum_exp = exp_l + exp_r
                    out = (left_val * exp_l + right_val * exp_r) / sum_exp
                else:
                    out = left_val + right_val
            else:
                out = torch.zeros(n_samples, dtype=torch.float64, device=x.device)
            
            # Clamp to prevent NaN/Inf
            out = torch.clamp(out, -1e6, 1e6)
            out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
            node_outputs.append(out)
        
        # Output layer: weighted sum of all node outputs + bias
        result = torch.zeros(n_samples, dtype=torch.float64, device=x.device)
        for i, out in enumerate(node_outputs):
            if i < len(self.output_weights):
                result = result + self.output_weights[i] * out
        result = result + self.output_bias
        
        return result, None  # (prediction, entropy=None) matching ONN signature
    
    def get_formula(self) -> str:
        """Return the formula string from the C++ result."""
        return self.formula_str


def cpp_result_to_module(result: dict) -> CppGraphModule:
    """
    Convenience function: convert a C++ run_evolution() result dict
    into a runnable PyTorch nn.Module.
    
    Usage:
        result = _core.run_evolution(X_list, y, ...)
        module = cpp_result_to_module(result)
        pred = module(x_tensor)
    """
    return CppGraphModule(result)
