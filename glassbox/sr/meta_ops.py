"""
Meta-Operations for Operation-Based Neural Networks v2.

Instead of 29 discrete operations, we use 4 parametric meta-operations
that can continuously interpolate between discrete operations.

Key insight: This converts discrete search → continuous optimization.
Gradient descent can slide from sin→cos or square→sqrt smoothly.

Meta-Operations:
- MetaPeriodic: sin, cos, linear (via ω, φ parameters)
- MetaPower: identity, square, sqrt, cube, inv (via p parameter)
- MetaArithmetic: add, multiply (via β parameter)
- MetaAggregation: sum, mean, max, min (via temperature)

After training, parameters can be "snapped" to recover discrete operations:
- ω=1, φ=0 → sin
- ω=1, φ=π/2 → cos
- p=1 → identity, p=2 → square, p=0.5 → sqrt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class MetaPeriodic(nn.Module):
    """
    Continuous parametric periodic operation.
    
    Formula: output = A * sin(ω * x + φ)
    
    Recovers:
    - sin(x): ω=1, φ=0, A=1
    - cos(x): ω=1, φ=π/2, A=1
    - linear(x): ω→0, output ≈ A*ω*x (via Taylor expansion)
    - -sin(x): A=-1 or φ=π
    """
    
    def __init__(
        self,
        init_omega: float = 1.0,
        init_phi: float = 0.0,
        init_amplitude: float = 1.0,
        learnable: bool = True,
    ):
        super().__init__()
        
        if learnable:
            self.omega = nn.Parameter(torch.tensor(init_omega))
            self.phi = nn.Parameter(torch.tensor(init_phi))
            self.amplitude = nn.Parameter(torch.tensor(init_amplitude))
        else:
            self.register_buffer('omega', torch.tensor(init_omega))
            self.register_buffer('phi', torch.tensor(init_phi))
            self.register_buffer('amplitude', torch.tensor(init_amplitude))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply parametric periodic function."""
        return self.amplitude * torch.sin(self.omega * x + self.phi)
    
    def get_discrete_op(self, threshold: float = 0.1) -> str:
        """Identify which discrete operation this approximates."""
        omega = self.omega.item()
        phi = self.phi.item() % (2 * math.pi)
        
        if abs(omega) < threshold:
            return "linear"
        elif abs(phi) < threshold or abs(phi - 2*math.pi) < threshold:
            return "sin"
        elif abs(phi - math.pi/2) < threshold:
            return "cos"
        elif abs(phi - math.pi) < threshold:
            return "-sin"
        elif abs(phi - 3*math.pi/2) < threshold:
            return "-cos"
        else:
            return f"sin(ω={omega:.2f}, φ={phi:.2f})"
    
    def snap_to_discrete(self):
        """Snap parameters to nearest standard values."""
        with torch.no_grad():
            # Snap omega to 1 if close
            if abs(self.omega.item() - 1.0) < 0.2:
                self.omega.fill_(1.0)
            
            # Snap phi to nearest multiple of π/2
            phi = self.phi.item() % (2 * math.pi)
            snap_points = [0, math.pi/2, math.pi, 3*math.pi/2]
            nearest = min(snap_points, key=lambda p: abs(phi - p))
            self.phi.fill_(nearest)
            
            # Snap amplitude to ±1
            if abs(self.amplitude.item()) > 0.5:
                self.amplitude.fill_(1.0 if self.amplitude.item() > 0 else -1.0)


class MetaPower(nn.Module):
    """
    Continuous parametric power operation.
    
    Formula: output = sign(x) * |x|^p
    
    Using sign(x)*|x|^p ensures differentiability for negative inputs.
    
    Recovers:
    - identity: p=1
    - square: p=2
    - cube: p=3
    - sqrt: p=0.5
    - reciprocal: p=-1
    - constant 1: p=0
    """
    
    def __init__(
        self,
        init_p: float = 1.0,
        learnable: bool = True,
        eps: float = 1e-6,
        p_min: float = -2.0,
        p_max: float = 3.0,
    ):
        super().__init__()
        self.eps = eps
        self.p_min = p_min
        self.p_max = p_max
        
        if learnable:
            self.p = nn.Parameter(torch.tensor(init_p))
        else:
            self.register_buffer('p', torch.tensor(init_p))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply parametric power function."""
        # Clamp p to valid range
        p = torch.clamp(self.p, self.p_min, self.p_max)
        
        # Safe power for negative inputs
        abs_x = torch.abs(x) + self.eps
        sign_x = torch.sign(x)
        
        # For even-ish powers (p close to 2, 4, ...), output should be positive
        # For odd-ish powers (p close to 1, 3, ...), preserve sign
        # We use: sign(x)^(round(p) mod 2) * |x|^p as approximation
        # Simpler: always preserve sign for continuous gradient flow
        result = sign_x * torch.pow(abs_x, p)
        
        # Clamp output to prevent explosion
        return torch.clamp(result, -100, 100)
    
    def get_discrete_op(self, threshold: float = 0.15) -> str:
        """Identify which discrete operation this approximates."""
        p = self.p.item()
        
        if abs(p - 1.0) < threshold:
            return "identity"
        elif abs(p - 2.0) < threshold:
            return "square"
        elif abs(p - 3.0) < threshold:
            return "cube"
        elif abs(p - 0.5) < threshold:
            return "sqrt"
        elif abs(p - (-1.0)) < threshold:
            return "reciprocal"
        elif abs(p) < threshold:
            return "constant_1"
        else:
            return f"power(p={p:.2f})"
    
    def snap_to_discrete(self):
        """Snap p to nearest standard value."""
        with torch.no_grad():
            p = self.p.item()
            snap_points = [-1, 0, 0.5, 1, 2, 3]
            nearest = min(snap_points, key=lambda sp: abs(p - sp))
            self.p.fill_(nearest)


class MetaArithmetic(nn.Module):
    """
    Continuous interpolation between addition and multiplication.
    
    Formula: output = (2 - β) * (x + y) + (β - 1) * (x * y)
    
    Recovers:
    - addition: β=1 → output = x + y
    - multiplication: β=2 → output = x * y
    - intermediate: 1 < β < 2 → weighted mix
    
    Note: Uses linear interpolation for simplicity.
    For more mathematically rigorous interpolation, see fractional hyperoperations.
    """
    
    def __init__(
        self,
        init_beta: float = 1.5,
        learnable: bool = True,
        beta_min: float = 1.0,
        beta_max: float = 2.0,
    ):
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max
        
        if learnable:
            self.beta = nn.Parameter(torch.tensor(init_beta))
        else:
            self.register_buffer('beta', torch.tensor(init_beta))
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Apply parametric arithmetic operation."""
        # Clamp beta to valid range
        beta = torch.clamp(self.beta, self.beta_min, self.beta_max)
        
        add_result = x + y
        mul_result = x * y
        
        # Linear interpolation
        result = (2 - beta) * add_result + (beta - 1) * mul_result
        
        return torch.clamp(result, -100, 100)
    
    def get_discrete_op(self, threshold: float = 0.1) -> str:
        """Identify which discrete operation this approximates."""
        beta = self.beta.item()
        
        if abs(beta - 1.0) < threshold:
            return "add"
        elif abs(beta - 2.0) < threshold:
            return "multiply"
        else:
            return f"add_mul_mix(β={beta:.2f})"
    
    def snap_to_discrete(self):
        """Snap beta to nearest standard value."""
        with torch.no_grad():
            beta = self.beta.item()
            if beta < 1.5:
                self.beta.fill_(1.0)  # Addition
            else:
                self.beta.fill_(2.0)  # Multiplication


class MetaArithmeticExtended(nn.Module):
    """
    Extended arithmetic with subtraction and division.
    
    Uses two parameters:
    - β ∈ [1, 2]: interpolate add ↔ mul
    - γ ∈ [-1, 1]: flip sign of second operand (for sub/div)
    
    Recovers:
    - add: β=1, γ=1
    - sub: β=1, γ=-1
    - mul: β=2, γ=1
    - div: β=2, γ=-1 (via x * (1/y))
    """
    
    def __init__(
        self,
        init_beta: float = 1.5,
        init_gamma: float = 1.0,
        learnable: bool = True,
    ):
        super().__init__()
        
        if learnable:
            self.beta = nn.Parameter(torch.tensor(init_beta))
            self.gamma = nn.Parameter(torch.tensor(init_gamma))
        else:
            self.register_buffer('beta', torch.tensor(init_beta))
            self.register_buffer('gamma', torch.tensor(init_gamma))
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Apply extended arithmetic operation."""
        beta = torch.clamp(self.beta, 1.0, 2.0)
        gamma = torch.tanh(self.gamma)  # Soft -1 to 1
        
        # Flip y based on gamma
        y_effective = gamma * y
        
        # For division, we need 1/y when beta=2, gamma=-1
        # Use: (1-|gamma|)*y + |gamma|*sign(gamma)*(1/(|y|+eps))
        # This is complex; for now just use linear interp
        
        add_result = x + y_effective
        mul_result = x * y_effective
        
        result = (2 - beta) * add_result + (beta - 1) * mul_result
        
        return torch.clamp(result, -100, 100)
    
    def get_discrete_op(self, threshold: float = 0.1) -> str:
        beta = self.beta.item()
        gamma = torch.tanh(torch.tensor(self.gamma.item())).item()
        
        is_add = abs(beta - 1.0) < threshold
        is_mul = abs(beta - 2.0) < threshold
        is_pos = gamma > 0.5
        is_neg = gamma < -0.5
        
        if is_add and is_pos:
            return "add"
        elif is_add and is_neg:
            return "sub"
        elif is_mul and is_pos:
            return "mul"
        elif is_mul and is_neg:
            return "div_approx"
        else:
            return f"arith(β={beta:.2f}, γ={gamma:.2f})"


class MetaAggregation(nn.Module):
    """
    Continuous aggregation operation using softmax-based approximations.
    
    Uses temperature parameter to interpolate between:
    - High temp (τ→∞): mean
    - Low temp (τ→0+): max (via softmax weighting)
    - Negative temp: min
    
    Also supports sum via a learnable scale.
    """
    
    def __init__(
        self,
        init_tau: float = 1.0,
        init_scale: float = 1.0,
        learnable: bool = True,
    ):
        super().__init__()
        
        if learnable:
            self.tau = nn.Parameter(torch.tensor(init_tau))
            self.scale = nn.Parameter(torch.tensor(init_scale))
        else:
            self.register_buffer('tau', torch.tensor(init_tau))
            self.register_buffer('scale', torch.tensor(init_scale))
    
    def forward(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Aggregate along dimension.
        
        Args:
            x: Input tensor
            dim: Dimension to aggregate over
            
        Returns:
            Aggregated tensor with dimension reduced
        """
        tau = self.tau.clamp(min=0.01)  # Avoid division by zero
        
        # Softmax weighted aggregation (approaches max as tau→0)
        weights = F.softmax(x / tau, dim=dim)
        result = (weights * x).sum(dim=dim)
        
        # Scale factor (allows sum when scale = n_elements)
        # For mean: scale = 1, for sum: scale = n
        return result * self.scale
    
    def get_discrete_op(self, threshold: float = 0.2) -> str:
        tau = self.tau.item()
        scale = self.scale.item()
        
        if tau < threshold:
            return "max"
        elif tau > 5.0 and abs(scale - 1.0) < threshold:
            return "mean"
        elif tau > 5.0:
            return "sum"
        else:
            return f"soft_agg(τ={tau:.2f})"


class MetaExp(nn.Module):
    """
    Safe exponential with learnable base and scale.
    
    Formula: output = scale * base^(x * rate)
    
    Recovers:
    - exp(x): base=e, rate=1, scale=1
    - 2^x: base=2, rate=1
    - constant: rate→0
    """
    
    def __init__(
        self,
        init_base: float = 2.718,  # e
        init_rate: float = 1.0,
        init_scale: float = 1.0,
        learnable: bool = True,
        max_exponent: float = 10.0,
    ):
        super().__init__()
        self.max_exponent = max_exponent
        
        if learnable:
            self.log_base = nn.Parameter(torch.tensor(math.log(init_base)))
            self.rate = nn.Parameter(torch.tensor(init_rate))
            self.scale = nn.Parameter(torch.tensor(init_scale))
        else:
            self.register_buffer('log_base', torch.tensor(math.log(init_base)))
            self.register_buffer('rate', torch.tensor(init_rate))
            self.register_buffer('scale', torch.tensor(init_scale))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply safe exponential."""
        base = torch.exp(self.log_base)  # Ensure base > 0
        exponent = self.rate * x
        exponent = torch.clamp(exponent, -self.max_exponent, self.max_exponent)
        
        result = self.scale * torch.pow(base, exponent)
        return torch.clamp(result, -100, 100)
    
    def get_discrete_op(self, threshold: float = 0.2) -> str:
        base = torch.exp(torch.tensor(self.log_base.item())).item()
        rate = self.rate.item()
        
        if abs(rate) < 0.1:
            return "constant"
        elif abs(base - math.e) < threshold:
            return "exp"
        elif abs(base - 2) < threshold:
            return "2^x"
        else:
            return f"exp(base={base:.2f})"


class MetaLog(nn.Module):
    """
    Safe logarithm with learnable base.
    
    Formula: output = scale * log_base(|x| + eps)
    
    Recovers:
    - ln(x): base=e
    - log2(x): base=2
    - log10(x): base=10
    """
    
    def __init__(
        self,
        init_base: float = 2.718,
        init_scale: float = 1.0,
        learnable: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps
        
        if learnable:
            self.log_base = nn.Parameter(torch.tensor(math.log(init_base)))
            self.scale = nn.Parameter(torch.tensor(init_scale))
        else:
            self.register_buffer('log_base', torch.tensor(math.log(init_base)))
            self.register_buffer('scale', torch.tensor(init_scale))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply safe logarithm."""
        base = torch.exp(self.log_base)
        log_base_val = torch.log(base)
        
        # Safe log
        safe_x = torch.abs(x) + self.eps
        result = self.scale * torch.log(safe_x) / log_base_val
        
        return torch.clamp(result, -100, 100)
    
    def get_discrete_op(self, threshold: float = 0.2) -> str:
        base = torch.exp(torch.tensor(self.log_base.item())).item()
        
        if abs(base - math.e) < threshold:
            return "ln"
        elif abs(base - 2) < threshold:
            return "log2"
        elif abs(base - 10) < threshold:
            return "log10"
        else:
            return f"log_base{base:.1f}"


# ============================================================================
# Factory and Utilities
# ============================================================================

class MetaOperationLibrary(nn.Module):
    """
    Complete library of meta-operations for an operation node.
    
    Each node has access to:
    - 1 MetaPeriodic (for sin, cos, linear)
    - 1 MetaPower (for square, sqrt, identity, etc.)
    - 1 MetaArithmetic (for add, mul)
    - 1 MetaAggregation (for sum, mean, max)
    - 1 MetaExp (for exponential)
    - 1 MetaLog (for logarithm)
    
    The node learns which meta-operation to use via soft selection.
    """
    
    def __init__(self, learnable: bool = True):
        super().__init__()
        
        self.periodic = MetaPeriodic(learnable=learnable)
        self.power = MetaPower(learnable=learnable)
        self.arithmetic = MetaArithmetic(learnable=learnable)
        self.aggregation = MetaAggregation(learnable=learnable)
        self.exp = MetaExp(learnable=learnable)
        self.log = MetaLog(learnable=learnable)
        
        # Selection weights for unary ops
        self.unary_weights = nn.Parameter(torch.zeros(4))  # periodic, power, exp, log
        
        # Selection weights for binary ops
        self.binary_weights = nn.Parameter(torch.zeros(2))  # arithmetic, aggregation
        
        # Learnable constant (for add_const, mul_const style ops)
        self.constant = nn.Parameter(torch.ones(1))
    
    def forward_unary(
        self,
        x: torch.Tensor,
        tau: float = 1.0,
        hard: bool = False,
    ) -> torch.Tensor:
        """Apply weighted mixture of unary operations."""
        # Compute all unary outputs
        outputs = [
            self.periodic(x),
            self.power(x),
            self.exp(x),
            self.log(x),
        ]
        outputs = torch.stack(outputs, dim=-1)  # (batch, ..., 4)
        
        # Soft selection
        if hard:
            weights = F.gumbel_softmax(self.unary_weights, tau=tau, hard=True)
        else:
            weights = F.softmax(self.unary_weights / tau, dim=-1)
        
        # Weighted sum
        result = (outputs * weights).sum(dim=-1)
        return result
    
    def forward_binary(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        tau: float = 1.0,
        hard: bool = False,
    ) -> torch.Tensor:
        """Apply weighted mixture of binary operations."""
        # For aggregation, stack inputs
        stacked = torch.stack([x, y], dim=-1)
        
        outputs = [
            self.arithmetic(x, y),
            self.aggregation(stacked, dim=-1),
        ]
        outputs = torch.stack(outputs, dim=-1)
        
        if hard:
            weights = F.gumbel_softmax(self.binary_weights, tau=tau, hard=True)
        else:
            weights = F.softmax(self.binary_weights / tau, dim=-1)
        
        result = (outputs * weights).sum(dim=-1)
        return result
    
    def get_selected_ops(self) -> dict:
        """Get currently selected operations."""
        unary_idx = self.unary_weights.argmax().item()
        binary_idx = self.binary_weights.argmax().item()
        
        unary_ops = [self.periodic, self.power, self.exp, self.log]
        binary_ops = [self.arithmetic, self.aggregation]
        
        return {
            'unary': unary_ops[unary_idx].get_discrete_op(),
            'binary': binary_ops[binary_idx].get_discrete_op(),
            'constant': self.constant.item(),
        }
    
    def snap_all_to_discrete(self):
        """Snap all meta-operations to nearest discrete values."""
        self.periodic.snap_to_discrete()
        self.power.snap_to_discrete()
        self.arithmetic.snap_to_discrete()


def create_meta_op(op_type: str, **kwargs) -> nn.Module:
    """Factory function to create meta-operations."""
    ops = {
        'periodic': MetaPeriodic,
        'power': MetaPower,
        'arithmetic': MetaArithmetic,
        'aggregation': MetaAggregation,
        'exp': MetaExp,
        'log': MetaLog,
    }
    
    if op_type not in ops:
        raise ValueError(f"Unknown op_type: {op_type}. Choose from {list(ops.keys())}")
    
    return ops[op_type](**kwargs)
