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
from typing import Optional, Tuple, List, Dict, Union
import math
import numpy as np


def safe_numpy_power(x, p):
    """
    Safe power function matching C++ power_sign_blend logic.
    Supports fractional powers of negative numbers via signed power: sign(x) * |x|^p.
    If p is an even integer, returns |x|^p (parity-preserving).
    """
    x = np.asarray(x)
    p = np.asarray(p)
    abs_x = np.abs(x) + 1e-15
    res = np.power(abs_x, p)
    
    # Parity check for even integers
    p_round = np.round(p)
    is_even = (np.abs(p - p_round) < 1e-6) & (p_round.astype(np.int64) % 2 == 0)
    
    if np.isscalar(is_even):
        return res if is_even else np.sign(x) * res
    return np.where(is_even, res, np.sign(x) * res)


# ============================================================================
# Known Mathematical Constants for Edge Weight Snapping
# ============================================================================

KNOWN_CONSTANTS: Dict[str, float] = {
    # Fundamental constants
    "π": math.pi,                          # 3.141592653589793
    "e": math.e,                           # 2.718281828459045
    "φ": (1 + math.sqrt(5)) / 2,          # Golden ratio: 1.618033988749895
    "√2": math.sqrt(2),                    # 1.4142135623730951
    "√3": math.sqrt(3),                    # 1.7320508075688772
    "√5": math.sqrt(5),                    # 2.23606797749979
    
    # Common fractions of pi
    "π/2": math.pi / 2,                    # 1.5707963267948966
    "π/3": math.pi / 3,                    # 1.0471975511965976
    "π/4": math.pi / 4,                    # 0.7853981633974483
    "π/6": math.pi / 6,                    # 0.5235987755982988
    "2π": 2 * math.pi,                     # 6.283185307179586
    
    # Logarithmic constants
    "ln(2)": math.log(2),                  # 0.6931471805599453
    "ln(10)": math.log(10),                # 2.302585092994046
    "log₂(e)": math.log2(math.e),          # 1.4426950408889634
    "log₁₀(e)": math.log10(math.e),        # 0.4342944819032518
    
    # Other useful constants
    "1/π": 1 / math.pi,                    # 0.3183098861837907
    "1/e": 1 / math.e,                     # 0.36787944117144233
    "2/π": 2 / math.pi,                    # 0.6366197723675814
    "π²": math.pi ** 2,                    # 9.869604401089358
    "e²": math.e ** 2,                     # 7.3890560989306495
    
    # Simple integers and fractions (commonly appearing)
    "1/2": 0.5,
    "1/3": 1/3,
    "2/3": 2/3,
    "1/4": 0.25,
    "3/4": 0.75,
}

# Also include negative versions
_negative_constants = {f"-{k}": -v for k, v in KNOWN_CONSTANTS.items()}
KNOWN_CONSTANTS.update(_negative_constants)


def snap_to_constant(
    value: float,
    threshold: float = 0.05,
    constants: Optional[Dict[str, float]] = None,
) -> Tuple[float, Optional[str]]:
    """
    Check if a value is close to a known mathematical constant and snap to it.
    
    Args:
        value: The value to check
        threshold: Maximum relative or absolute difference to consider "close"
                   Uses relative difference for values > 1, absolute for smaller values
        constants: Dictionary of {name: value} constants to check against.
                   Defaults to KNOWN_CONSTANTS.
    
    Returns:
        Tuple of (snapped_value, constant_name) if a match is found,
        or (original_value, None) if no match.
    
    Example:
        >>> snap_to_constant(3.14)
        (3.141592653589793, 'π')
        >>> snap_to_constant(2.72)
        (2.718281828459045, 'e')
        >>> snap_to_constant(1.23)
        (1.23, None)
    """
    if constants is None:
        constants = KNOWN_CONSTANTS
    
    best_match = None
    best_distance = float('inf')
    
    for name, const_value in constants.items():
        # Calculate distance (relative for larger values, absolute for small)
        if abs(const_value) > 1:
            distance = abs(value - const_value) / abs(const_value)
        else:
            distance = abs(value - const_value)
        
        if distance < threshold and distance < best_distance:
            best_distance = distance
            best_match = (const_value, name)
    
    if best_match is not None:
        return best_match
    return (value, None)


def snap_tensor_to_constants(
    tensor: torch.Tensor,
    threshold: float = 0.05,
    constants: Optional[Dict[str, float]] = None,
) -> Tuple[torch.Tensor, Dict[int, str]]:
    """
    Snap all values in a tensor to known constants where applicable.
    
    Args:
        tensor: Input tensor
        threshold: Snapping threshold
        constants: Constants dictionary
    
    Returns:
        Tuple of (snapped_tensor, mapping) where mapping is {flat_index: constant_name}
    """
    result = tensor.clone()
    mapping = {}
    
    flat = result.flatten()
    for i in range(flat.numel()):
        val = flat[i].item()
        snapped, name = snap_to_constant(val, threshold, constants)
        if name is not None:
            flat[i] = snapped
            mapping[i] = name
    
    return result.view_as(tensor), mapping


def get_constant_symbol(value: float, threshold: float = 0.05) -> str:
    """
    Get the symbolic representation of a value if it matches a known constant.
    
    Args:
        value: The value to check
        threshold: Matching threshold
    
    Returns:
        Symbolic string (e.g., "π", "e", "√2") or formatted number
    """
    _, name = snap_to_constant(value, threshold)
    if name is not None:
        return name
    
    # Format as a clean number
    if abs(value - round(value)) < 0.001:
        return str(int(round(value)))
    return f"{value:.4g}"


def normalize_formula_ascii(formula: str) -> str:
    """Convert pretty symbolic output into parser-safe ASCII math."""
    normalized = formula

    replacements = [
        ("log₁₀(e)", "log(E,10)"),
        ("log₂(e)", "log(E,2)"),
        ("ln(10)", "log(10)"),
        ("ln(2)", "log(2)"),
        ("π²", "(pi^2)"),
        ("e²", "(E^2)"),
        ("1/π", "(1/pi)"),
        ("2/π", "(2/pi)"),
        ("2π", "(2*pi)"),
        ("π/2", "(pi/2)"),
        ("π/3", "(pi/3)"),
        ("π/4", "(pi/4)"),
        ("π/6", "(pi/6)"),
        ("√2", "sqrt(2)"),
        ("√3", "sqrt(3)"),
        ("√5", "sqrt(5)"),
        ("φ", "((1+sqrt(5))/2)"),
        ("π", "pi"),
        ("·", "*"),
        ("⋅", "*"),
        ("×", "*"),
        ("−", "-"),
        ("–", "-"),
        ("²", "^2"),
        ("³", "^3"),
    ]
    for old, new in replacements:
        normalized = normalized.replace(old, new)

    normalized = re.sub(r"√\|([^|]+)\|/\|([^|]+)\|", r"sqrt(abs(\1))/abs(\2)", normalized)
    normalized = re.sub(r"([A-Za-z0-9_()]+)\s*/√\|([^|]+)\|", r"\1/sqrt(abs(\2))", normalized)
    normalized = re.sub(r"√\|([^|]+)\|", r"sqrt(abs(\1))", normalized)
    normalized = re.sub(r"\|([^|]+)\|", r"abs(\1)", normalized)

    # Replace the standalone Euler constant symbol while leaving scientific notation intact.
    normalized = re.sub(r"(?<![A-Za-z0-9_])e(?![A-Za-z0-9_])", "E", normalized)
    normalized = normalized.replace("^", "**")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


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
    
    def snap_to_discrete(self, omega_threshold: float = 0.2, phi_threshold: float = 0.15):
        """Snap parameters to nearest standard values.
        
        Args:
            omega_threshold: Absolute distance to snap omega toward 1.0.
            phi_threshold: Absolute distance to snap phi to nearest π/2 multiple.
        """
        with torch.no_grad():
            # Snap omega to nearest integer if close
            omega_val = self.omega.item()
            omega_rounded = round(omega_val)
            if omega_rounded != 0 and abs(omega_val - omega_rounded) < omega_threshold:
                self.omega.fill_(float(omega_rounded))
            
            # Snap phi to nearest multiple of π/2
            phi = self.phi.item() % (2 * math.pi)
            snap_points = [0, math.pi/2, math.pi, 3*math.pi/2]
            nearest = min(snap_points, key=lambda p: abs(phi - p))
            if abs(phi - nearest) < phi_threshold:
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
        abs_pow = torch.pow(abs_x, p)
        
        # Determine even/odd symmetry based on p
        # p=0, 2, 4... -> Even (cos(p*pi) = 1)
        # p=1, 3...    -> Odd  (cos(p*pi) = -1)
        # Smooth blend: is_even = 0.5 * (1 + cos(p*pi))
        # Note: This aligns perfectly with integers.
        is_even = 0.5 * (1.0 + torch.cos(p * math.pi))
        
        # Result blends between odd (preserves sign) and even (absolute) behavior
        # odd_result = sign(x) * |x|^p
        # even_result = |x|^p
        result = (1.0 - is_even) * (sign_x * abs_pow) + is_even * abs_pow
        
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
    
    def snap_to_discrete(self, snap_points=None, threshold: float = 0.3):
        """Snap p to nearest standard value.
        
        Args:
            snap_points: List of candidate values to snap to.
                         Defaults to [-1, 0, 0.5, 1, 2, 3].
            threshold: Maximum absolute distance for snapping.
        """
        if snap_points is None:
            snap_points = [-1, 0, 0.5, 1, 2, 3]
        with torch.no_grad():
            p = self.p.item()
            nearest = min(snap_points, key=lambda sp: abs(p - sp))
            if abs(p - nearest) < threshold:
                self.p.fill_(nearest)


class MetaArithmetic(nn.Module):
    """
    Continuous interpolation between addition and multiplication.
    
    Uses SIGMOID parameterization to avoid gradient plateau.
    
    OLD (linear): output = (2 - β) * (x + y) + (β - 1) * (x * y)
        Problem: ∂f/∂β = xy - (x+y) = 0 when xy = x+y, causing plateaus
    
    NEW (sigmoid): output = (1 - σ(α)) * (x + y) + σ(α) * (x * y)
        where σ(α) = sigmoid(α), learnable α ∈ (-∞, ∞)
        Gradient: ∂f/∂α = σ(α)(1-σ(α)) * (xy - (x+y))
        Still has same zero point, but σ'(α) provides additional gradient signal
        near the boundaries (α → ±∞ gives clear add/mul selection)
    
    Recovers:
    - addition: α << 0 → σ(α) ≈ 0 → output ≈ x + y
    - multiplication: α >> 0 → σ(α) ≈ 1 → output ≈ x * y
    """
    
    def __init__(
        self,
        init_alpha: float = 0.0,  # Start at midpoint (σ(0) = 0.5)
        learnable: bool = True,
    ):
        super().__init__()
        
        if learnable:
            self.alpha = nn.Parameter(torch.tensor(init_alpha))
        else:
            self.register_buffer('alpha', torch.tensor(init_alpha))
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Apply parametric arithmetic operation."""
        # Sigmoid blend factor - avoids gradient plateau of linear interpolation
        blend = torch.sigmoid(self.alpha)
        
        add_result = x + y
        mul_result = x * y
        
        # Sigmoid interpolation
        result = (1 - blend) * add_result + blend * mul_result
        
        return torch.clamp(result, -100, 100)
    
    @property
    def beta(self) -> float:
        """Compatibility: return equivalent beta value (1=add, 2=mul)."""
        return 1.0 + torch.sigmoid(self.alpha).item()
    
    def get_discrete_op(self, threshold: float = 0.3) -> str:
        """Identify which discrete operation this approximates."""
        blend = torch.sigmoid(self.alpha).item()
        
        if blend < threshold:
            return "add"
        elif blend > (1 - threshold):
            return "multiply"
        else:
            return f"({blend:.2f}*mul + {1-blend:.2f}*add)"
    
    def snap_to_discrete(self):
        """Snap alpha to push toward pure add or mul."""
        with torch.no_grad():
            blend = torch.sigmoid(self.alpha).item()
            if blend < 0.5:
                self.alpha.fill_(-5.0)  # σ(-5) ≈ 0.007 → addition
            else:
                self.alpha.fill_(5.0)   # σ(5) ≈ 0.993 → multiplication


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
        # Use simple parameters to control mix of Add, Mul, Div
        # We interpret beta/gamma to maintain backward compatibility with 'learnable' init
        # But we implement a cleaner logic:
        # beta: 1->Add, 2->Mul/Div
        # gamma: >0 -> Mul, <0 -> Div
        
        beta_val = self.beta.item() if isinstance(self.beta, nn.Parameter) else self.beta
        gamma_val = self.gamma.item() if isinstance(self.gamma, nn.Parameter) else self.gamma
        
        # Calculate mixing weights based on beta/gamma distance
        # Target points: Add(1, 1), Mul(2, 1), Div(2, -1), Sub(1, -1)
        
        # Differentiable soft selection
        # We transform (beta, gamma) into a 4-way classification
        # Distances to prototypes:
        d_add = (self.beta - 1.0)**2 + (self.gamma - 1.0)**2
        d_mul = (self.beta - 2.0)**2 + (self.gamma - 1.0)**2
        d_div = (self.beta - 2.0)**2 + (self.gamma + 1.0)**2
        d_sub = (self.beta - 1.0)**2 + (self.gamma + 1.0)**2
        
        # Logits = -distance (closer is higher logit)
        logits = torch.stack([-d_add, -d_mul, -d_div, -d_sub])
        weights = F.softmax(logits * 5.0, dim=0) # Temperature 5 for sharpness
        
        # Compute all 4 results
        res_add = x + y
        res_sub = x - y
        res_mul = x * y
        res_div = x / (torch.abs(y) + 1e-6) * torch.sign(y) # Safe division
        
        # Weighted sum
        result = (
            weights[0] * res_add + 
            weights[1] * res_mul + 
            weights[2] * res_div + 
            weights[3] * res_sub
        )
        
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
        # Avoid division by zero while preserving sign (for min/max)
        tau = torch.where(self.tau.abs() < 0.01, torch.sign(self.tau + 1e-8) * 0.01, self.tau)
        
        # Softmax weighted aggregation (approaches max as tau->0+, min as tau->0-)
        weights = F.softmax(x / tau, dim=dim)
        result = (weights * x).sum(dim=dim)
        
        # Scale factor (allows sum when scale = n_elements)
        # For mean: scale = 1, for sum: scale = n
        return result * self.scale

    def set_tau(self, tau: float):
        """Update temperature for annealing."""
        with torch.no_grad():
            if isinstance(self.tau, torch.nn.Parameter):
                self.tau.copy_(torch.tensor(tau))
            else:
                self.tau = torch.tensor(tau, device=self.tau.device)
    
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
        log_base_val = torch.log(base) + 1e-8  # Prevent division by zero
        
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


# ============================================================================
# Edge Weight Snapping Utilities
# ============================================================================

def snap_edge_weights(
    module: nn.Module,
    threshold: float = 0.05,
    constants: Optional[Dict[str, float]] = None,
    verbose: bool = False,
) -> Dict[str, str]:
    """
    Snap all learnable parameters in a module to known mathematical constants.
    
    This is useful after training to make discovered formulas more interpretable.
    For example, if training discovers a coefficient of 3.14159, this will snap
    it to π for cleaner formula representation.
    
    Args:
        module: PyTorch module containing parameters to snap
        threshold: Maximum distance to snap (default 0.05 = 5%)
        constants: Custom constants dict, or None to use KNOWN_CONSTANTS
        verbose: If True, print each snap operation
    
    Returns:
        Dictionary mapping parameter names to their snapped constant names
    
    Example:
        >>> model = SomeModel()
        >>> # After training...
        >>> snapped = snap_edge_weights(model, threshold=0.05)
        >>> print(snapped)
        {'layer.weight[0]': 'π', 'layer.bias[1]': 'e'}
    """
    snapped_params = {}
    
    with torch.no_grad():
        for name, param in module.named_parameters():
            if param.numel() == 0:
                continue
            
            flat = param.data.flatten()
            for i in range(flat.numel()):
                val = flat[i].item()
                snapped_val, const_name = snap_to_constant(val, threshold, constants)
                
                if const_name is not None:
                    flat[i] = snapped_val
                    key = f"{name}[{i}]" if flat.numel() > 1 else name
                    snapped_params[key] = const_name
                    
                    if verbose:
                        print(f"Snapped {key}: {val:.6f} -> {const_name} ({snapped_val:.6f})")
            
            # Write back reshaped data
            param.data = flat.view_as(param.data)
    
    return snapped_params


def snap_value_to_constant(
    value: Union[float, torch.Tensor],
    threshold: float = 0.05,
) -> Tuple[Union[float, torch.Tensor], Optional[str]]:
    """
    Convenience wrapper that handles both float and single-element tensors.
    
    Args:
        value: Float or single-element tensor
        threshold: Snapping threshold
    
    Returns:
        Tuple of (snapped_value, constant_name)
    """
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("snap_value_to_constant only works with scalar tensors")
        val = value.item()
        snapped, name = snap_to_constant(val, threshold)
        if name is not None:
            return torch.tensor(snapped, dtype=value.dtype, device=value.device), name
        return value, None
    else:
        return snap_to_constant(value, threshold)


class ConstantAwareLinear(nn.Linear):
    """
    A Linear layer that can snap its weights/bias to known constants.
    
    Extends nn.Linear with constant-snapping capability for more
    interpretable coefficient discovery.
    """
    
    def snap_to_constants(self, threshold: float = 0.05, verbose: bool = False) -> Dict[str, str]:
        """Snap weights and bias to known constants."""
        snapped = {}
        
        with torch.no_grad():
            # Snap weights
            for i in range(self.weight.numel()):
                val = self.weight.data.flatten()[i].item()
                snapped_val, name = snap_to_constant(val, threshold)
                if name is not None:
                    self.weight.data.flatten()[i] = snapped_val
                    snapped[f"weight[{i}]"] = name
                    if verbose:
                        print(f"weight[{i}]: {val:.6f} -> {name}")
            
            # Snap bias if present
            if self.bias is not None:
                for i in range(self.bias.numel()):
                    val = self.bias.data[i].item()
                    snapped_val, name = snap_to_constant(val, threshold)
                    if name is not None:
                        self.bias.data[i] = snapped_val
                        snapped[f"bias[{i}]"] = name
                        if verbose:
                            print(f"bias[{i}]: {val:.6f} -> {name}")
        
        return snapped
    
    def get_symbolic_weights(self, threshold: float = 0.05) -> List[str]:
        """Get symbolic representation of weights."""
        return [get_constant_symbol(w.item(), threshold) for w in self.weight.flatten()]
