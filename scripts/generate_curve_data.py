"""
Curve Classifier Training Data Generator

Generates synthetic (curve, operator_labels) pairs for training a 
curve classifier that predicts which mathematical operators are present.

Usage:
    python scripts/generate_curve_data.py --n-samples 100000 --output data/curve_dataset.npz
"""

import numpy as np
import argparse
import random
from typing import List, Tuple, Dict, Set
from pathlib import Path
import ast

# Make scipy optional - provide fallbacks
try:
    from scipy.stats import skew, kurtosis
    from scipy.signal import savgol_filter as _savgol_filter
    _HAS_SAVGOL = True
except ImportError:
    _HAS_SAVGOL = False
    # Fallback implementations
    def skew(x):
        """Simple skewness calculation."""
        m = np.mean(x)
        s = np.std(x)
        if s < 1e-10:
            return 0.0
        return np.mean(((x - m) / s) ** 3)
    
    def kurtosis(x):
        """Simple kurtosis calculation."""
        m = np.mean(x)
        s = np.std(x)
        if s < 1e-10:
            return 0.0
        return np.mean(((x - m) / s) ** 4) - 3.0

# Make tqdm optional
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x  # Just pass through


# =============================================================================
# OPERATOR DEFINITIONS (matching ONN's meta-ops)
# =============================================================================

OPERATOR_CLASSES = {
    # Unary operators
    'identity': 0,       # x (implicit in many formulas)
    'sin': 1,            # sin(ωx + φ)
    'cos': 2,            # cos(ωx + φ)
    'power': 3,          # x^p (includes x, x², x³, √x, 1/x)
    'exp': 4,            # e^(βx)
    'log': 5,            # log(|x| + ε)

    # Binary indicators
    'addition': 6,       # Terms added
    'multiplication': 7, # Terms multiplied

    # Rational functions
    'rational': 8,       # 1/(x+c), x/(x²+c) type
}

N_CLASSES = len(OPERATOR_CLASSES)

# Feature dimensionality (see extract_all_features)
FEATURE_DIM = 366

# Feature schema (slices into feature vector)
FEATURE_SCHEMA = {
    "raw": (0, 128),
    "fft": (128, 160),
    "fft_phase": (160, 192),
    "deriv": (192, 320),
    "stats": (320, 329),
    "curv": (329, 366),
}


# =============================================================================
# FORMULA TEMPLATES
# =============================================================================

# Each template: (formula_string, set_of_operators)
# Placeholders: {a}, {b}, {c} = random coefficients, {p} = random power

SIMPLE_TEMPLATES = [
    # Identity / Linear
    ("x", {'identity'}),
    ("{a} * x", {'identity', 'multiplication'}),
    ("{a} * x + {b}", {'identity', 'addition', 'multiplication'}),
    
    # Powers
    ("x ** 2", {'power'}),
    ("x ** 3", {'power'}),
    ("x ** {p}", {'power'}),
    ("{a} * x ** 2", {'power', 'multiplication'}),
    ("{a} * x ** 2 + {b} * x + {c}", {'power', 'addition', 'multiplication'}),
    ("np.sqrt(np.abs(x) + 0.01)", {'power'}),
    ("1 / (np.abs(x) + 0.1)", {'rational'}),
    
    # Periodic
    ("np.sin(x)", {'sin'}),
    ("np.cos(x)", {'cos'}),
    ("np.sin({a} * x)", {'sin', 'multiplication'}),
    ("np.cos({a} * x)", {'cos', 'multiplication'}),
    ("np.sin({a} * x + {b})", {'sin', 'addition', 'multiplication'}),
    ("{a} * np.sin({b} * x)", {'sin', 'multiplication'}),
    ("{a} * np.cos({b} * x)", {'cos', 'multiplication'}),
    
    # Exponential
    ("np.exp({a} * x)", {'exp', 'multiplication'}),
    ("np.exp(-x ** 2)", {'exp', 'power'}),
    ("{a} * np.exp({b} * x)", {'exp', 'multiplication'}),
    
    # Logarithmic
    ("np.log(np.abs(x) + 1)", {'log'}),
    ("{a} * np.log(np.abs(x) + 1)", {'log', 'multiplication'}),
]

COMPOUND_TEMPLATES = [
    # Polynomial combinations
    ("x ** 2 + x", {'power', 'addition'}),
    ("x ** 3 - x", {'power', 'addition'}),
    ("{a} * x ** 3 + {b} * x ** 2 + {c} * x", {'power', 'addition', 'multiplication'}),
    
    # Periodic + Polynomial
    ("np.sin(x) + x", {'sin', 'power', 'addition'}),
    ("np.sin(x) + x ** 2", {'sin', 'power', 'addition'}),
    ("np.cos(x) + x ** 2", {'cos', 'power', 'addition'}),
    ("{a} * np.sin({b} * x) + {c} * x", {'sin', 'power', 'addition', 'multiplication'}),
    ("x * np.sin(x)", {'sin', 'power', 'multiplication'}),
    ("x ** 2 * np.cos(x)", {'cos', 'power', 'multiplication'}),
    
    # Periodic combinations
    ("np.sin(x) + np.cos(x)", {'sin', 'cos', 'addition'}),
    ("np.sin(x) * np.cos(x)", {'sin', 'cos', 'multiplication'}),
    ("{a} * np.sin({b} * x) + {c} * np.cos({d} * x)", {'sin', 'cos', 'addition', 'multiplication'}),
    
    # Exponential combinations
    ("np.exp(x) + np.exp(-x)", {'exp', 'addition'}),  # cosh-like
    ("np.exp(x) - np.exp(-x)", {'exp', 'addition'}),  # sinh-like
    ("np.exp(-x ** 2 / 2)", {'exp', 'power'}),  # Gaussian
    ("x * np.exp(-x)", {'exp', 'power', 'multiplication'}),
    
    # Log combinations
    ("x * np.log(np.abs(x) + 1)", {'log', 'power', 'multiplication'}),
    ("np.log(np.abs(x) + 1) ** 2", {'log', 'power'}),
    
    # Mixed
    ("np.sin(x) * np.exp(-x ** 2)", {'sin', 'exp', 'multiplication'}),
    ("np.log(np.abs(x) + 1) + np.sin(x)", {'log', 'sin', 'addition'}),
]

# NEW: Rational function templates
RATIONAL_TEMPLATES = [
    # Simple rational (1/x type)
    ("1 / (np.abs(x) + {c})", {'rational'}),
    ("{a} / (np.abs(x) + {c})", {'rational', 'multiplication'}),
    
    # Quadratic denominator (Lorentzian type)
    ("1 / (x**2 + {c})", {'rational'}),
    ("{a} / (x**2 + {c})", {'rational', 'multiplication'}),
    ("x / (x**2 + {c})", {'rational'}),
    
    # Mixed rational
    ("(x + {a}) / (x**2 + {b})", {'rational', 'addition'}),
    ("x**2 / (x**2 + {c})", {'rational'}),
    ("(x**2 + {a}) / (x**2 + {b})", {'rational', 'addition'}),
    
    # Rational + other
    ("1 / (x**2 + 1) + np.sin(x)", {'rational', 'sin', 'addition'}),
    ("x / (x**2 + 1) + {a}", {'rational', 'addition'}),
]

# NEW: Nested composition templates (sin(x²), exp(sin(x)), etc.)
NESTED_TEMPLATES = [
    # sin/cos of polynomial (CRITICAL - currently missing!)
    ("np.sin(x ** 2)", {'sin', 'power'}),
    ("np.cos(x ** 2)", {'cos', 'power'}),
    ("np.sin({a} * x ** 2)", {'sin', 'power', 'multiplication'}),
    ("np.cos({a} * x ** 2)", {'cos', 'power', 'multiplication'}),
    ("np.sin(x ** 2 + {a} * x)", {'sin', 'power', 'addition'}),
    
    # exp of trig
    ("np.exp(np.sin(x))", {'exp', 'sin'}),
    ("np.exp(np.cos(x))", {'exp', 'cos'}),
    ("np.exp(-np.cos(x))", {'exp', 'cos'}),
    ("{a} * np.exp(np.sin({b} * x))", {'exp', 'sin', 'multiplication'}),
    
    # log of polynomial
    ("np.log(x ** 2 + 1)", {'log', 'power'}),
    ("np.log(np.abs(x ** 2 + {a}) + 0.01)", {'log', 'power'}),
    ("np.log(x ** 2 + {a} * x + 1)", {'log', 'power', 'addition'}),
    
    # trig of exp (complex oscillations)
    ("np.sin(np.exp(-x ** 2))", {'sin', 'exp', 'power'}),
    ("np.cos(np.exp(-np.abs(x)))", {'cos', 'exp'}),
    
    # Double trig
    ("np.sin(np.sin(x))", {'sin'}),
    ("np.sin(np.cos(x))", {'sin', 'cos'}),
    
    # Power of trig
    ("np.sin(x) ** 2", {'sin', 'power'}),
    ("np.cos(x) ** 2", {'cos', 'power'}),
    ("{a} * np.sin(x) ** 2 + {b} * np.cos(x) ** 2", {'sin', 'cos', 'power', 'addition', 'multiplication'}),
]

# NEW: Product/modulated terms (amplitude modulation, damped oscillations)
PRODUCT_TEMPLATES = [
    # Linear modulation
    ("x * np.sin(x)", {'power', 'sin', 'multiplication'}),
    ("x * np.cos(x)", {'power', 'cos', 'multiplication'}),
    ("x * np.sin({a} * x)", {'power', 'sin', 'multiplication'}),
    
    # Quadratic modulation
    ("x ** 2 * np.sin(x)", {'power', 'sin', 'multiplication'}),
    ("x ** 2 * np.cos(x)", {'power', 'cos', 'multiplication'}),
    
    # Exponential damping (damped oscillations - common in physics!)
    ("np.exp(-x) * np.sin({a} * x)", {'exp', 'sin', 'multiplication'}),
    ("np.exp(-np.abs(x)) * np.cos({a} * x)", {'exp', 'cos', 'multiplication'}),
    ("np.exp(-x ** 2) * np.sin({a} * x)", {'exp', 'sin', 'power', 'multiplication'}),
    ("{a} * np.exp(-{b} * x ** 2) * np.sin({c} * x)", {'exp', 'sin', 'power', 'multiplication'}),
    
    # Gaussian-modulated
    ("x * np.exp(-x ** 2)", {'power', 'exp', 'multiplication'}),
    ("x ** 2 * np.exp(-x ** 2)", {'power', 'exp', 'multiplication'}),
    
    # Product of trig
    ("np.sin(x) * np.cos(x)", {'sin', 'cos', 'multiplication'}),
    ("np.sin({a} * x) * np.cos({b} * x)", {'sin', 'cos', 'multiplication'}),
]

# NEW: Irrational constant templates (π, e, √2)
IRRATIONAL_TEMPLATES = [
    # Pi-related
    ("np.pi * x", {'identity', 'multiplication'}),
    ("np.sin(np.pi * x)", {'sin', 'multiplication'}),
    ("np.cos(np.pi * x)", {'cos', 'multiplication'}),
    ("np.sin(2 * np.pi * x)", {'sin', 'multiplication'}),
    ("{a} * np.sin(np.pi * x) + {b}", {'sin', 'addition', 'multiplication'}),
    
    # e-related (np.e = 2.718...)
    ("np.e * x", {'identity', 'multiplication'}),
    ("np.e ** x", {'exp'}),  # Same as exp(x)
    ("{a} * np.e ** (-x ** 2)", {'exp', 'power', 'multiplication'}),
    
    # sqrt(2) related
    ("np.sqrt(2) * x", {'identity', 'multiplication'}),
    ("x ** np.sqrt(2)", {'power'}),
    ("np.sin(np.sqrt(2) * x)", {'sin', 'multiplication'}),
]

# NEW: Hyperbolic functions (common in ML activations and physics)
HYPERBOLIC_TEMPLATES = [
    ("np.sinh(x)", {'exp', 'addition'}),
    ("np.cosh(x)", {'exp', 'addition'}),
    ("np.tanh(x)", {'exp', 'rational'}),
    ("{a} * np.tanh({b} * x)", {'exp', 'rational', 'multiplication'}),
    ("np.sinh({a} * x)", {'exp', 'addition', 'multiplication'}),
    ("np.cosh({a} * x)", {'exp', 'addition', 'multiplication'}),
    
    # Hyperbolic + polynomial
    ("np.tanh(x) + {a} * x", {'exp', 'rational', 'addition'}),
    ("x * np.tanh(x)", {'exp', 'rational', 'power', 'multiplication'}),
    
    # Sigmoid-like (logistic function)
    ("1 / (1 + np.exp(-x))", {'exp', 'rational'}),
    ("1 / (1 + np.exp(-{a} * x))", {'exp', 'rational', 'multiplication'}),
]

# NEW: Physics-inspired templates (common in scientific formulas)
PHYSICS_TEMPLATES = [
    # Coulomb/gravitational-like (1/r, 1/r²)
    ("1 / np.sqrt(x**2 + {c})", {'power', 'rational'}),
    ("{a} / np.sqrt(x**2 + {c})", {'power', 'rational', 'multiplication'}),
    
    # Radioactive decay / charging curves
    ("np.exp(-x) * (1 + x)", {'exp', 'power', 'addition', 'multiplication'}),
    ("1 - np.exp(-{a} * x)", {'exp', 'addition', 'multiplication'}),
    ("{a} * (1 - np.exp(-{b} * x))", {'exp', 'addition', 'multiplication'}),
    
    # Michaelis-Menten / saturation kinetics
    ("x / (x + {c})", {'rational'}),
    ("{a} * x / (x + {c})", {'rational', 'multiplication'}),
    ("x / ({a} + x)", {'rational'}),
    
    # Gaussian / bell curve
    ("np.exp(-x**2 / {c})", {'exp', 'power'}),
    ("{a} * np.exp(-(x - {b})**2 / {c})", {'exp', 'power', 'multiplication'}),
    
    # Power-law decay
    ("1 / (1 + x**2)", {'rational'}),
    ("1 / (1 + np.abs(x))", {'rational'}),
]

ALL_TEMPLATES = (
    SIMPLE_TEMPLATES + 
    COMPOUND_TEMPLATES + 
    RATIONAL_TEMPLATES + 
    NESTED_TEMPLATES + 
    PRODUCT_TEMPLATES + 
    IRRATIONAL_TEMPLATES +
    HYPERBOLIC_TEMPLATES +
    PHYSICS_TEMPLATES
)


# =============================================================================
# PCFG-BASED FORMULA GENERATION
# =============================================================================

class PCFGFormulaGenerator:
    """Probabilistic Context-Free Grammar formula generator.
    
    Generates formulas of arbitrary depth using recursive grammar rules,
    covering compositions (e.g. sin(cos(x²))) that fixed templates cannot.
    
    Grammar:
        EXPR → UNARY(EXPR) | BINARY(EXPR, EXPR) | TERM
        UNARY → sin | cos | exp | log | sqrt | abs | sinh | cosh | tanh
        BINARY → + | - | * | /
        TERM → x | CONST | CONST*x | x**POWER | CONST*x**POWER
    
    Uses Lample & Charton-style depth budget splitting for balanced
    depth distribution in binary nodes.
    """
    
    # Production probabilities (at non-terminal depth)
    DEFAULT_WEIGHTS = {
        'unary': 0.30,
        'binary': 0.25,
        'term': 0.45,
    }
    
    UNARY_OPS = [
        ('np.sin',  'sin'),
        ('np.cos',  'cos'),
        ('np.exp',  'exp'),
        ('np.log',  'log'),
        ('np.sqrt', 'power'),
        ('np.abs',  'identity'),
        ('np.sinh', 'exp'),       # sinh decomposes to exp
        ('np.cosh', 'exp'),       # cosh decomposes to exp
        ('np.tanh', 'exp'),       # tanh decomposes to exp/rational
    ]
    
    BINARY_OPS = [
        ('+', 'addition'),
        ('-', 'addition'),
        ('*', 'multiplication'),
        ('/', 'rational'),
    ]
    
    POWER_CHOICES = [0.5, 2, 3, 4, -1, -0.5, 1.5, 2.5, 0.33]
    
    def __init__(self, max_depth: int = 4, weights: Dict | None = None):
        self.max_depth = max_depth
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}
    
    def generate(self) -> Tuple[str, Set[str]]:
        """Generate a random formula and its operator set.
        
        Returns:
            (formula_string, operator_set) matching template interface.
        """
        ops: Set[str] = set()
        formula = self._generate_expr(self.max_depth, ops)
        return formula, ops
    
    def _generate_expr(self, depth_budget: int, ops: Set[str]) -> str:
        """Recursively generate an expression with given depth budget."""
        if depth_budget <= 1:
            return self._generate_term(ops)
        
        # Choose production rule
        r = random.random()
        cumulative = 0.0
        choice = 'term'
        for rule, prob in self.weights.items():
            cumulative += prob
            if r < cumulative:
                choice = rule
                break
        
        if choice == 'unary':
            return self._generate_unary(depth_budget - 1, ops)
        elif choice == 'binary':
            return self._generate_binary(depth_budget - 1, ops)
        else:
            return self._generate_term(ops)
    
    def _generate_unary(self, child_budget: int, ops: Set[str]) -> str:
        """Generate UNARY(EXPR)."""
        func_str, op_class = random.choice(self.UNARY_OPS)
        ops.add(op_class)
        
        child = self._generate_expr(child_budget, ops)
        
        # Safety wrapping for dangerous functions
        if func_str == 'np.log':
            return f"{func_str}(np.abs({child}) + 0.01)"
        elif func_str == 'np.exp':
            # Clip argument to prevent overflow
            return f"{func_str}(np.clip({child}, -10, 10))"
        elif func_str == 'np.sqrt':
            return f"{func_str}(np.abs({child}) + 0.01)"
        elif func_str == 'np.tanh':
            ops.add('rational')  # tanh = (e^x - e^-x)/(e^x + e^-x)
            return f"{func_str}({child})"
        else:
            return f"{func_str}({child})"
    
    def _generate_binary(self, depth_budget: int, ops: Set[str]) -> str:
        """Generate BINARY(EXPR, EXPR) with Lample & Charton depth split."""
        op_str, op_class = random.choice(self.BINARY_OPS)
        ops.add(op_class)
        
        # Split depth budget randomly between children (Lample & Charton style)
        if depth_budget <= 1:
            left_budget = 1
            right_budget = 1
        else:
            split = random.randint(1, depth_budget - 1)
            left_budget = split
            right_budget = depth_budget - split
        
        left = self._generate_expr(left_budget, ops)
        right = self._generate_expr(right_budget, ops)
        
        if op_str == '/' :
            # Division: protect denominator from zero
            ops.add('rational')
            return f"(({left}) / (({right}) ** 2 + 0.1))"
        else:
            return f"(({left}) {op_str} ({right}))"
    
    def _generate_term(self, ops: Set[str]) -> str:
        """Generate a terminal expression."""
        choice = random.random()
        
        if choice < 0.30:
            # Just x
            ops.add('identity')
            return "x"
        elif choice < 0.45:
            # Constant
            c = self._random_const()
            return str(c)
        elif choice < 0.65:
            # c * x
            c = self._random_const()
            ops.add('identity')
            ops.add('multiplication')
            return f"({c} * x)"
        elif choice < 0.85:
            # x ** p
            p = random.choice(self.POWER_CHOICES)
            ops.add('power')
            if p < 0:
                # Negative power = rational-like
                return f"(np.abs(x) + 0.01) ** {p}"
            elif p == int(p):
                return f"x ** {int(p)}"
            else:
                return f"np.abs(x) ** {p}"
        else:
            # c * x ** p
            c = self._random_const()
            p = random.choice(self.POWER_CHOICES)
            ops.add('power')
            ops.add('multiplication')
            if p < 0:
                return f"({c} * (np.abs(x) + 0.01) ** {p})"
            elif p == int(p):
                return f"({c} * x ** {int(p)})"
            else:
                return f"({c} * np.abs(x) ** {p})"
    
    def _random_const(self) -> float:
        """Generate a random constant, biased toward 'nice' values."""
        if random.random() < 0.3:
            # Nice constants
            return random.choice([
                1.0, -1.0, 2.0, -2.0, 0.5, -0.5,
                3.14159, 2.71828, 1.41421,  # π, e, √2
            ])
        else:
            return round(np.random.uniform(-5, 5), 4)


# =============================================================================
# MULTI-SNR NOISE INJECTION
# =============================================================================

def apply_noise_augmentation(y: np.ndarray, noise_profile: str = 'multi') -> np.ndarray:
    """Apply noise augmentation to a curve.
    
    Args:
        y: Clean curve values.
        noise_profile: 'multi' for randomized multi-SNR noise,
                       'legacy' for backward-compatible fixed Gaussian.
    
    Returns:
        Noisy curve (copy of input, original is not modified).
    """
    y_noisy = y.copy()
    y_std = np.std(y_noisy) + 1e-10
    
    if noise_profile == 'legacy':
        # Legacy: single additive Gaussian (backward compat)
        y_noisy += np.random.normal(0.0, 0.01 * y_std, size=y_noisy.shape)
        return y_noisy
    
    # Multi-SNR: randomly pick a noise type
    r = random.random()
    
    if r < 0.20:
        # Clean — no noise
        pass
    elif r < 0.45:
        # White Gaussian, low noise (SNR ~30-40 dB → std ≈ 0.3-1% of signal std)
        snr_std = np.random.uniform(0.003, 0.01) * y_std
        y_noisy += np.random.normal(0.0, snr_std, size=y_noisy.shape)
    elif r < 0.65:
        # White Gaussian, medium noise (SNR ~15-25 dB → std ≈ 1-5%)
        snr_std = np.random.uniform(0.01, 0.05) * y_std
        y_noisy += np.random.normal(0.0, snr_std, size=y_noisy.shape)
    elif r < 0.75:
        # White Gaussian, high noise (SNR ~5-10 dB → std ≈ 10-30%)
        snr_std = np.random.uniform(0.1, 0.3) * y_std
        y_noisy += np.random.normal(0.0, snr_std, size=y_noisy.shape)
    elif r < 0.85:
        # Pink (1/f) noise — correlated, common in physical sensors
        n = len(y_noisy)
        white = np.random.randn(n)
        fft_white = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n, d=1.0)
        freqs[0] = 1.0  # Avoid division by zero
        fft_pink = fft_white / np.sqrt(freqs)
        pink = np.fft.irfft(fft_pink, n=n)
        pink_std = np.random.uniform(0.01, 0.05) * y_std
        pink = pink / (np.std(pink) + 1e-10) * pink_std
        y_noisy += pink
    elif r < 0.95:
        # Quantization noise — round to N levels (simulates ADC)
        n_levels = random.choice([16, 32, 64, 128, 256])
        y_min, y_max = y_noisy.min(), y_noisy.max()
        span = y_max - y_min + 1e-10
        y_noisy = np.round((y_noisy - y_min) / span * n_levels) / n_levels * span + y_min
    else:
        # Outlier spikes — random point corruption (1-5% of points)
        n = len(y_noisy)
        frac = np.random.uniform(0.01, 0.05)
        n_outliers = max(1, int(n * frac))
        indices = np.random.choice(n, size=n_outliers, replace=False)
        y_noisy[indices] += np.random.normal(0.0, 3.0 * y_std, size=n_outliers)
    
    return y_noisy


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_raw_features(y: np.ndarray, n_points: int = 128, curvature_alpha: float = 5.0) -> np.ndarray:
    """Normalize and resample curve to fixed size with curvature-aware resampling.
    
    Instead of uniform resampling, concentrates sample points near
    high-curvature regions of the curve for better shape discrimination.
    
    Args:
        y: Input curve values
        n_points: Number of output sample points
        curvature_alpha: Curvature concentration strength (0 = uniform)
    """
    if len(y) < 3:
        # Too short to compute curvature — fall back to uniform
        if len(y) != n_points:
            x_old = np.linspace(0, 1, len(y))
            x_new = np.linspace(0, 1, n_points)
            y = np.interp(x_new, x_old, y)
        y_min, y_max = y.min(), y.max()
        if y_max - y_min > 1e-10:
            return (y - y_min) / (y_max - y_min)
        return np.zeros(n_points, dtype=y.dtype)
    
    # Compute local curvature κ = |y''| / (1 + y'^2)^1.5
    dy = np.gradient(y)
    ddy = np.gradient(dy)
    kappa = np.abs(ddy) / (1.0 + dy**2)**1.5
    kappa = np.nan_to_num(kappa, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Build concentration density w(i) = 1 + α·κ(i)
    w = 1.0 + curvature_alpha * kappa
    
    # Accumulate CDF from w, then invert to get non-uniform sample positions
    cdf = np.cumsum(w)
    cdf = cdf / cdf[-1]  # normalize to [0, 1]
    
    # Target uniform positions in CDF space → non-uniform in original space
    target_cdf = np.linspace(0, 1, n_points)
    # Map to original indices (fractional)
    x_old = np.linspace(0, 1, len(y))
    sample_positions = np.interp(target_cdf, cdf, x_old)
    
    # Interpolate y at the non-uniform sample positions
    y_resampled = np.interp(sample_positions, x_old, y)
    
    # Normalize to [0, 1] range
    y_min, y_max = y_resampled.min(), y_resampled.max()
    if y_max - y_min > 1e-10:
        y_norm = (y_resampled - y_min) / (y_max - y_min)
    else:
        y_norm = np.zeros_like(y_resampled)
    
    return y_norm


def extract_fft_features(y: np.ndarray, n_freqs: int = 32) -> np.ndarray:
    """Extract dominant frequencies - detects periodicity."""
    # Detrend and apply a Hann window to reduce spectral leakage
    if len(y) > 1:
        y = y - np.mean(y)
        window = np.hanning(len(y))
        y = y * window

    fft = np.fft.rfft(y)
    magnitudes = np.abs(fft)[:n_freqs]
    
    # Pad if signal too short to fill n_freqs bins
    if len(magnitudes) < n_freqs:
        magnitudes = np.pad(magnitudes, (0, n_freqs - len(magnitudes)))
    
    # Normalize
    mag_max = magnitudes.max()
    if mag_max > 1e-10:
        magnitudes = magnitudes / mag_max
    
    return magnitudes


def extract_fft_phase_features(y: np.ndarray, n_bins: int = 32) -> np.ndarray:
    """Extract FFT phase features to discriminate signals with same magnitude spectrum.
    
    Phase captures the relative alignment of frequency components,
    allowing the classifier to distinguish e.g. sin(x)+sin(3x) from sin(x)*sin(3x)
    which have similar magnitude spectra but very different phase profiles.
    
    Args:
        y: Input curve values
        n_bins: Number of phase bins to extract (matching magnitude bins)
        
    Returns:
        Phase features normalized to [-1, 1] range (32 values)
    """
    # Same preprocessing as magnitude features
    if len(y) > 1:
        y = y - np.mean(y)
        window = np.hanning(len(y))
        y = y * window
    
    fft = np.fft.rfft(y)
    phases = np.angle(fft)[:n_bins]  # range [-π, π]
    
    # Normalize to [-1, 1] by dividing by π
    phases = phases / np.pi
    
    # Zero out phase where magnitude is negligible (phase is meaningless there)
    magnitudes = np.abs(fft)[:n_bins]
    mag_max = magnitudes.max() if len(magnitudes) > 0 else 0.0
    if mag_max > 1e-10:
        insignificant = magnitudes / mag_max < 0.01
        phases[insignificant] = 0.0
    
    # Pad if signal too short to fill n_bins
    if len(phases) < n_bins:
        phases = np.pad(phases, (0, n_bins - len(phases)))
    
    return phases


def _smooth_signal(y: np.ndarray) -> np.ndarray:
    """Smooth a signal before differentiation to reduce noise amplification.
    
    Uses Savitzky-Golay filter (window=11, polyorder=3) when scipy is available,
    otherwise falls back to a simple moving average (window=7).
    """
    if len(y) < 11:
        return y  # Too short to smooth
    
    if _HAS_SAVGOL:
        # Savitzky-Golay preserves peaks and shape better than moving average
        win = min(11, len(y) if len(y) % 2 == 1 else len(y) - 1)
        return _savgol_filter(y, window_length=win, polyorder=min(3, win - 1))
    else:
        # Fallback: simple moving average (window=7)
        kernel_size = min(7, len(y))
        kernel = np.ones(kernel_size) / kernel_size
        return np.convolve(y, kernel, mode='same')


def extract_derivative_features(y: np.ndarray, n_points: int = 64) -> np.ndarray:
    """First and second derivatives - detect polynomial degree, inflection points.
    
    Applies smoothing before differentiation to reduce noise amplification,
    producing more reliable derivative features.
    """
    # Smooth before differentiation to reduce noise amplification
    y_smooth = _smooth_signal(y)
    
    # Handle very short signals (need at least 3 points for ddy)
    if len(y_smooth) < 3:
        return np.zeros(n_points * 2, dtype=np.float64)
    
    dy = np.gradient(y_smooth)   # First derivative (central difference)
    ddy = np.gradient(dy)        # Second derivative (central difference)
    
    # Resample to fixed size
    dy_resampled = np.interp(
        np.linspace(0, 1, n_points), 
        np.linspace(0, 1, len(dy)), 
        dy
    )
    ddy_resampled = np.interp(
        np.linspace(0, 1, n_points), 
        np.linspace(0, 1, len(ddy)), 
        ddy
    )
    
    # Normalize
    dy_max = np.abs(dy_resampled).max()
    ddy_max = np.abs(ddy_resampled).max()
    
    if dy_max > 1e-10:
        dy_resampled = dy_resampled / dy_max
    if ddy_max > 1e-10:
        ddy_resampled = ddy_resampled / ddy_max
    
    return np.concatenate([dy_resampled, ddy_resampled])


def extract_stat_features(y: np.ndarray) -> np.ndarray:
    """Global statistics about the curve."""
    # Normalize for consistent stats
    y_norm = (y - y.mean()) / (y.std() + 1e-10)
    
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        features = [
            np.mean(y_norm),
            np.std(y_norm),
            np.min(y_norm),
            np.max(y_norm),
            np.median(y_norm),
            float(skew(y_norm)),              # Asymmetry
            float(kurtosis(y_norm)),          # Peakedness
            np.sum(np.diff(np.sign(y_norm)) != 0),    # Zero crossings
            np.sum(np.diff(np.sign(np.diff(y_norm))) != 0),  # Extrema count
        ]
    
    result = np.array(features)
    # scipy skew/kurtosis return NaN for constant signals — replace with 0
    return np.nan_to_num(result, nan=0.0)


def extract_curvature_features(y: np.ndarray, n_points: int = 32) -> np.ndarray:
    """
    Extract curvature features to distinguish rational from exponential decay.
    Rational functions have distinct acceleration profiles.
    """
    # Smooth curve to reduce derivative noise
    if len(y) < 2:
        # np.gradient needs at least 2 points; return zeros
        return np.zeros(n_points + 5, dtype=np.float64)
    if len(y) >= 7:
        window = np.ones(7, dtype=np.float64)
        window = window / window.sum()
        y_smooth = np.convolve(y, window, mode='same')
    else:
        y_smooth = y

    # First derivative (velocity)
    dy = np.gradient(y_smooth)
    
    # Second derivative (acceleration)
    ddy = np.gradient(dy)
    
    # Curvature: κ = y'' / (1 + y'^2)^1.5
    curvature = ddy / (1 + dy**2)**1.5
    
    # Handle infinities
    curvature = np.clip(curvature, -100, 100)
    curvature = np.nan_to_num(curvature, nan=0.0, posinf=100, neginf=-100)
    
    # Resample to fixed size
    curvature_resampled = np.interp(
        np.linspace(0, 1, n_points),
        np.linspace(0, 1, len(curvature)),
        curvature
    )
    
    # Normalize
    curv_max = np.abs(curvature_resampled).max()
    if curv_max > 1e-10:
        curvature_resampled = curvature_resampled / curv_max
    
    # Additional curvature statistics
    curvature_stats = np.array([
        np.mean(curvature_resampled),
        np.std(curvature_resampled),
        np.min(curvature_resampled),
        np.max(curvature_resampled),
        np.sum(np.diff(np.sign(curvature_resampled)) != 0),  # Sign changes
    ])
    
    return np.concatenate([curvature_resampled, curvature_stats])


def extract_all_features(y: np.ndarray) -> np.ndarray:
    """Combine all features into single vector."""
    raw = extract_raw_features(y, n_points=128)           # 128
    fft = extract_fft_features(y, n_freqs=32)             # 32
    fft_phase = extract_fft_phase_features(y, n_bins=32)  # 32  (NEW)
    deriv = extract_derivative_features(y, n_points=64)   # 128
    stats = extract_stat_features(y)                       # 9
    curv = extract_curvature_features(y, n_points=32)      # 37 (32 + 5)
    
    features = np.concatenate([raw, fft, fft_phase, deriv, stats, curv])
    if features.shape[0] != FEATURE_DIM:
        raise ValueError(f"Feature vector size mismatch: {features.shape[0]} != {FEATURE_DIM}")
    return features  # Total: 366


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_random_formula() -> Tuple[str, Set[str]]:
    """Pick a random template and fill in coefficients."""
    template, operators = random.choice(ALL_TEMPLATES)
    
    # Random coefficients
    formula = template.format(
        a=np.random.uniform(-3, 3),
        b=np.random.uniform(0.5, 3),
        c=np.random.uniform(-2, 2),
        d=np.random.uniform(0.5, 3),
        p=np.random.choice([0.5, 2, 3, 4, -1, -0.5]),  # common powers
    )
    
    return formula, operators


def _safe_eval_ast(node: ast.AST, x: np.ndarray) -> np.ndarray:
    """Evaluate a restricted AST for numeric expressions involving x and np.*."""
    allowed_funcs = {
        'sin': np.sin,
        'cos': np.cos,
        'exp': np.exp,
        'log': np.log,
        'sqrt': np.sqrt,
        'abs': np.abs,
        'sinh': np.sinh,
        'cosh': np.cosh,
        'tanh': np.tanh,
        'clip': np.clip,
    }

    if isinstance(node, ast.Expression):
        return _safe_eval_ast(node.body, x)
    if isinstance(node, ast.Constant):
        return np.array(node.value)
    if isinstance(node, ast.Name):
        if node.id == 'x':
            return x
        if node.id == 'np':
            return np
        raise ValueError("Unsafe name")
    if isinstance(node, ast.Attribute):
        value = _safe_eval_ast(node.value, x)
        if value is np and node.attr in allowed_funcs:
            return allowed_funcs[node.attr]
        # Allow np.pi, np.e constants
        if value is np and node.attr == 'pi':
            return np.pi
        if value is np and node.attr == 'e':
            return np.e
        raise ValueError("Unsafe attribute")
    if isinstance(node, ast.Call):
        func = _safe_eval_ast(node.func, x)
        args = [_safe_eval_ast(a, x) for a in node.args]
        return func(*args)
    if isinstance(node, ast.UnaryOp):
        operand = _safe_eval_ast(node.operand, x)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ValueError("Unsafe unary op")
    if isinstance(node, ast.BinOp):
        left = _safe_eval_ast(node.left, x)
        right = _safe_eval_ast(node.right, x)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left ** right
        raise ValueError("Unsafe binary op")
    raise ValueError("Unsafe expression")


def evaluate_formula(formula: str, x: np.ndarray, safe_eval: bool = True) -> Tuple[np.ndarray | None, str]:
    """Safely evaluate a formula string.

    Returns:
        (y, status) where status is one of: ok, nan_or_inf, extreme, eval_fail
    """
    try:
        # Suppress warnings during evaluation (domain errors are expected)
        with np.errstate(all='ignore'):
            if safe_eval:
                tree = ast.parse(formula, mode='eval')
                y = _safe_eval_ast(tree, x)
            else:
                y = eval(formula, {"x": x, "np": np})
            
            # Broadcast scalar results (e.g. from constant PCFG formulas)
            y = np.asarray(y, dtype=np.float64)
            if y.ndim == 0:
                y = np.broadcast_to(y, x.shape).copy()
        
        # Check for invalid values
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            return None, "nan_or_inf"
        
        # Check for extreme values
        if np.abs(y).max() > 1e6:
            return None, "extreme"
        
        return y, "ok"
    except Exception:
        return None, "eval_fail"


def _ast_contains_x(node: ast.AST) -> bool:
    """Return True if AST node contains variable x."""
    if isinstance(node, ast.Name) and node.id == 'x':
        return True
    for child in ast.iter_child_nodes(node):
        if _ast_contains_x(child):
            return True
    return False


def derive_operators_from_formula(formula: str) -> Set[str]:
    """Derive operator labels directly from the expression AST."""
    ops: Set[str] = set()
    try:
        tree = ast.parse(formula, mode='eval')
    except Exception:
        return ops

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == 'x':
            ops.add('identity')

        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == 'np':
                if func.attr in ('sin', 'cos', 'exp', 'log'):
                    ops.add(func.attr)
                elif func.attr == 'sqrt':
                    ops.add('power')  # sqrt is x^0.5
                elif func.attr in ('sinh', 'cosh'):
                    ops.add('exp')    # hyperbolic = exp composition
                    ops.add('addition')
                elif func.attr == 'tanh':
                    ops.add('exp')
                    ops.add('rational')

        if isinstance(node, ast.BinOp):
            if isinstance(node.op, (ast.Add, ast.Sub)):
                ops.add('addition')
            if isinstance(node.op, ast.Mult):
                ops.add('multiplication')
            if isinstance(node.op, ast.Pow):
                ops.add('power')
            if isinstance(node.op, ast.Div):
                # Mark rational if denominator depends on x
                if _ast_contains_x(node.right):
                    ops.add('rational')
                else:
                    ops.add('multiplication')  # Division by constant is just scaling

    return ops


def normalize_operators(operators: Set[str], formula: str) -> Set[str]:
    """Normalize operator labels to reduce template noise."""
    ops = set(operators)
    # Keep both power and rational if both are present - they're independent features
    return ops


def operators_to_labels(operators: Set[str], formula: str | None = None) -> np.ndarray:
    """Convert operator set to multi-hot label vector (optionally normalized)."""
    if formula is not None:
        derived = derive_operators_from_formula(formula)
        if derived:
            operators = derived
        operators = normalize_operators(operators, formula)

    labels = np.zeros(N_CLASSES, dtype=np.float32)
    for op in operators:
        if op in OPERATOR_CLASSES:
            labels[OPERATOR_CLASSES[op]] = 1.0
    return labels


def generate_dataset(
    n_samples: int,
    x_range: Tuple[float, float] = (-5, 5),
    n_points: int = 256,
    show_progress: bool = True,
    templates: List[Tuple[str, Set[str]]] | None = None,
    seed: int | None = None,
    balance_templates: bool = False,
    balance_classes: bool = False,
    n_workers: int | None = None,
    x_ranges: List[Tuple[float, float]] | None = None,
    x_scale_min: float = 1.0,
    x_scale_max: float = 1.0,
    x_shift_std: float = 0.0,
    noise_std: float = 0.0,
    y_scale_min: float = 1.0,
    y_scale_max: float = 1.0,
    y_offset_std: float = 0.0,
    safe_eval: bool = True,
    signed_bd: bool = False,
    pcfg_ratio: float = 0.0,
    pcfg_max_depth: int = 4,
    noise_profile: str = 'legacy',
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Generate labeled curve dataset.
    
    Returns:
        features: (n_samples, 334) feature vectors
        labels: (n_samples, n_classes) multi-hot labels
        formulas: list of formula strings (for debugging)
    """
    if templates is None:
        templates = ALL_TEMPLATES
    
    x = np.linspace(x_range[0], x_range[1], n_points)
    
    features_list = []
    labels_list = []
    formulas_list = []
    
    # Multiprocessing for speed
    import multiprocessing
    from functools import partial
    
    # Determine number of processes
    if n_workers is None:
        n_workers = max(1, min(multiprocessing.cpu_count() - 1, n_samples))
    
    # Split workload
    chunk_size = n_samples // n_workers
    remainder = n_samples % n_workers
    chunks = [chunk_size + (1 if i < remainder else 0) for i in range(n_workers)]

    # Build class -> template index mapping for class-balanced sampling
    class_to_templates = None
    class_sampling_weights = None
    if balance_classes:
        class_to_templates = {name: [] for name in OPERATOR_CLASSES.keys()}
        for i, (_, ops) in enumerate(templates):
            for op in ops:
                if op in class_to_templates:
                    class_to_templates[op].append(i)
        # Uniform weights across classes that have templates
        classes = [c for c, idxs in class_to_templates.items() if idxs]
        n_classes_with_templates = len(classes)
        weights = [1.0 / n_classes_with_templates for _ in classes] if n_classes_with_templates > 0 else []
        class_sampling_weights = (classes, weights)

    # Deterministic worker seeds (if provided)
    if seed is not None:
        seed_seq = np.random.SeedSequence(seed)
        worker_seeds = [int(s.generate_state(1)[0]) for s in seed_seq.spawn(n_workers)]
    else:
        worker_seeds = [None] * n_workers

    work_items = list(zip(chunks, worker_seeds))
    
    print(f"Generating data using {n_workers} workers...")
    
    with multiprocessing.Pool(n_workers) as pool:
        # Create partial function with fixed arguments
        worker_func = partial(
            generate_chunk, 
            x_range=x_range, 
            n_points=n_points,
            templates=templates,
            balance_templates=balance_templates,
            balance_classes=balance_classes,
            class_to_templates=class_to_templates,
            class_sampling_weights=class_sampling_weights,
            x_ranges=x_ranges,
            x_scale_min=x_scale_min,
            x_scale_max=x_scale_max,
            x_shift_std=x_shift_std,
            noise_std=noise_std,
            y_scale_min=y_scale_min,
            y_scale_max=y_scale_max,
            y_offset_std=y_offset_std,
            safe_eval=safe_eval,
            signed_bd=signed_bd,
            pcfg_ratio=pcfg_ratio,
            pcfg_max_depth=pcfg_max_depth,
            noise_profile=noise_profile,
        )
        
        # Run workers
        if show_progress:
            results = list(tqdm(
                pool.imap(worker_func, work_items),
                total=n_workers,
                desc="Generating chunks"
            ))
        else:
            results = list(pool.imap(worker_func, work_items))
    
    # Combine results
    features_list = []
    labels_list = []
    formulas_list = []
    
    reject_stats = {
        'nan_or_inf': 0,
        'extreme': 0,
        'feature_invalid': 0,
        'eval_fail': 0,
        'max_attempts': 0,
    }

    for feats, lbls, forms, stats in results:
        features_list.extend(feats)
        labels_list.extend(lbls)
        formulas_list.extend(forms)
        for k in reject_stats:
            reject_stats[k] += stats.get(k, 0)
    
    # Shuffle
    indices = np.random.permutation(len(features_list))
    features = np.array(features_list, dtype=np.float32)[indices]
    labels = np.array(labels_list, dtype=np.float32)[indices]
    formulas = [formulas_list[i] for i in indices]
    
    # Trim to exact size
    features = features[:n_samples]
    labels = labels[:n_samples]
    formulas = formulas[:n_samples]
    
    if sum(reject_stats.values()) > 0:
        print("\nRejection summary:")
        for k, v in reject_stats.items():
            print(f"  {k:15s}: {v}")

    return features, labels, formulas


def generate_dataset_streamed(
    n_samples: int,
    output_path: Path,
    x_range: Tuple[float, float] = (-5, 5),
    n_points: int = 256,
    show_progress: bool = True,
    templates: List[Tuple[str, Set[str]]] | None = None,
    seed: int | None = None,
    balance_templates: bool = False,
    balance_classes: bool = False,
    n_workers: int | None = None,
    x_ranges: List[Tuple[float, float]] | None = None,
    x_scale_min: float = 1.0,
    x_scale_max: float = 1.0,
    x_shift_std: float = 0.0,
    noise_std: float = 0.0,
    y_scale_min: float = 1.0,
    y_scale_max: float = 1.0,
    y_offset_std: float = 0.0,
    safe_eval: bool = True,
    signed_bd: bool = False,
    save_formulas: bool = False,
    pcfg_ratio: float = 0.0,
    pcfg_max_depth: int = 4,
    noise_profile: str = 'legacy',
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Generate dataset while streaming features/labels to disk."""
    if templates is None:
        templates = ALL_TEMPLATES
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_path = output_path.with_suffix(".features.dat")
    labels_path = output_path.with_suffix(".labels.dat")
    formulas_path = output_path.with_suffix(".formulas.txt")

    features_mm = np.memmap(features_path, dtype=np.float32, mode='w+', shape=(n_samples, FEATURE_DIM))
    labels_mm = np.memmap(labels_path, dtype=np.float32, mode='w+', shape=(n_samples, N_CLASSES))
    formulas_list: List[str] = []
    
    # Use ExitStack for proper resource management
    from contextlib import ExitStack
    import multiprocessing
    from functools import partial
    
    exit_stack = ExitStack()
    formulas_file = None
    if save_formulas:
        formulas_file = exit_stack.enter_context(open(formulas_path, "w", encoding="utf-8"))

    if n_workers is None:
        n_workers = max(1, min(multiprocessing.cpu_count() - 1, n_samples))

    chunk_size = n_samples // n_workers
    remainder = n_samples % n_workers
    chunks = [chunk_size + (1 if i < remainder else 0) for i in range(n_workers)]

    class_to_templates = None
    class_sampling_weights = None
    if balance_classes:
        class_to_templates = {name: [] for name in OPERATOR_CLASSES.keys()}
        for i, (_, ops) in enumerate(templates):
            for op in ops:
                if op in class_to_templates:
                    class_to_templates[op].append(i)
        # Uniform weights across classes that have templates
        classes = [c for c, idxs in class_to_templates.items() if idxs]
        n_classes_with_templates = len(classes)
        weights = [1.0 / n_classes_with_templates for _ in classes] if n_classes_with_templates > 0 else []
        class_sampling_weights = (classes, weights)

    if seed is not None:
        seed_seq = np.random.SeedSequence(seed)
        worker_seeds = [int(s.generate_state(1)[0]) for s in seed_seq.spawn(n_workers)]
    else:
        worker_seeds = [None] * n_workers

    work_items = list(zip(chunks, worker_seeds))

    worker_func = partial(
        generate_chunk,
        x_range=x_range,
        n_points=n_points,
        templates=templates,
        balance_templates=balance_templates,
        balance_classes=balance_classes,
        class_to_templates=class_to_templates,
        class_sampling_weights=class_sampling_weights,
        x_ranges=x_ranges,
        x_scale_min=x_scale_min,
        x_scale_max=x_scale_max,
        x_shift_std=x_shift_std,
        noise_std=noise_std,
        y_scale_min=y_scale_min,
        y_scale_max=y_scale_max,
        y_offset_std=y_offset_std,
        safe_eval=safe_eval,
        signed_bd=signed_bd,
        pcfg_ratio=pcfg_ratio,
        pcfg_max_depth=pcfg_max_depth,
        noise_profile=noise_profile,
    )

    reject_stats = {
        'nan_or_inf': 0,
        'extreme': 0,
        'feature_invalid': 0,
        'eval_fail': 0,
        'max_attempts': 0,
    }

    print(f"Generating data using {n_workers} workers...")
    cursor = 0

    with multiprocessing.Pool(n_workers) as pool:
        if show_progress:
            results = tqdm(pool.imap(worker_func, work_items), total=n_workers, desc="Generating chunks")
        else:
            results = pool.imap(worker_func, work_items)

        for feats, lbls, forms, stats in results:
            count = len(feats)
            if count == 0:
                continue
            end = min(cursor + count, n_samples)
            actual = end - cursor
            features_mm[cursor:end] = np.array(feats[:actual], dtype=np.float32)
            labels_mm[cursor:end] = np.array(lbls[:actual], dtype=np.float32)
            if save_formulas:
                for f in forms[:actual]:
                    formulas_file.write(f + "\n")
            cursor = end

            for k in reject_stats:
                reject_stats[k] += stats.get(k, 0)

    # Close file handles via context manager
    exit_stack.close()

    if cursor < n_samples:
        raise RuntimeError(f"Generated only {cursor} samples out of {n_samples} requested")

    # Avoid loading full arrays into memory; return memmap views
    features = features_mm
    labels = labels_mm

    formulas: List[str] = []
    if save_formulas:
        with open(formulas_path, "r", encoding="utf-8") as f:
            formulas = [line.rstrip("\n") for line in f]

    if sum(reject_stats.values()) > 0:
        print("\nRejection summary:")
        for k, v in reject_stats.items():
            print(f"  {k:15s}: {v}")

    return np.asarray(features), np.asarray(labels), formulas


def generate_chunk(
    task: Tuple[int, int | None],
    x_range: Tuple[float, float],
    n_points: int,
    templates: List[Tuple[str, Set[str]]],
    balance_templates: bool,
    balance_classes: bool,
    class_to_templates: Dict[str, List[int]] | None,
    class_sampling_weights: Tuple[List[str], List[float]] | None,
    x_ranges: List[Tuple[float, float]] | None,
    x_scale_min: float,
    x_scale_max: float,
    x_shift_std: float,
    noise_std: float,
    y_scale_min: float,
    y_scale_max: float,
    y_offset_std: float,
    safe_eval: bool,
    signed_bd: bool,
    pcfg_ratio: float = 0.0,
    pcfg_max_depth: int = 4,
    noise_profile: str = 'legacy',
) -> Tuple[List, List, List, Dict[str, int]]:
    """Generate a chunk of samples (worker function)."""
    n, seed = task

    # Re-seed for each process to ensure diversity (deterministic if seed provided)
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    else:
        np.random.seed()
        random.seed()
    
    x = np.linspace(x_range[0], x_range[1], n_points)
    
    features_local = []
    labels_local = []
    formulas_local = []
    
    # Generate batch with some buffer for failures
    target = n
    attempts = 0
    max_attempts = n * 5
    
    template_indices = list(range(len(templates)))
    template_idx = 0
    if balance_templates:
        random.shuffle(template_indices)

    stats = {
        'nan_or_inf': 0,
        'extreme': 0,
        'feature_invalid': 0,
        'eval_fail': 0,
        'max_attempts': 0,
    }

    # Create PCFG generator if needed
    pcfg_gen = PCFGFormulaGenerator(max_depth=pcfg_max_depth) if pcfg_ratio > 0 else None

    while len(features_local) < target and attempts < max_attempts:
        attempts += 1
        
        # Decide: PCFG or template-based generation
        use_pcfg = pcfg_gen is not None and random.random() < pcfg_ratio
        
        if use_pcfg:
            # PCFG-based generation
            formula, _ = pcfg_gen.generate()
            # Deriving operators from AST ensures labels match the final string exactly
            # (fixes issue where generator tracking missed implicit ops or drifted)
            operators = derive_operators_from_formula(formula)
        elif balance_classes and class_to_templates:
            if class_sampling_weights is not None:
                classes, weights = class_sampling_weights
                target_class = np.random.choice(classes, p=weights)
            else:
                available = [k for k, v in class_to_templates.items() if v]
                target_class = random.choice(available)
            template_idx = random.choice(class_to_templates[target_class])
            template, operators = templates[template_idx]
        elif balance_templates:
            if template_idx % len(templates) == 0:
                random.shuffle(template_indices)
            template, operators = templates[template_indices[template_idx % len(templates)]]
            template_idx += 1
        else:
            template, operators = random.choice(templates)
        
        # Fill in template coefficients (only for template-based formulas)
        if not use_pcfg:
            b = np.random.uniform(0.3, 6.0)  # Wider frequency range
            d = np.random.uniform(0.3, 6.0)
            if signed_bd:
                b *= np.random.choice([-1.0, 1.0])
                d *= np.random.choice([-1.0, 1.0])

            # Expanded power choices: includes fractional powers (1.5, 2.3, etc.)
            POWER_CHOICES = [
                0.25, 0.33, 0.5, 0.67,  # Fractional roots
                1.0, 1.5, 2.0, 2.3, 2.5, 3.0, 4.0,  # Positive powers (including fractional)
                -0.5, -1.0, -2.0,  # Negative powers
            ]
            
            formula = template.format(
                a=np.random.uniform(-5, 5),  # Wider amplitude
                b=b,
                c=np.random.uniform(-3, 3),  # Slightly wider
                d=d,
                p=np.random.choice(POWER_CHOICES),
            )
        
        # Sample x-range per curve if provided
        if x_ranges:
            xmin, xmax = random.choice(x_ranges)
            x = np.linspace(xmin, xmax, n_points)
        else:
            x = np.linspace(x_range[0], x_range[1], n_points)

        # Optional x scaling and shifting
        if x_scale_min != 1.0 or x_scale_max != 1.0:
            x = x * np.random.uniform(x_scale_min, x_scale_max)
        if x_shift_std > 0.0:
            span = (x.max() - x.min())
            x = x + np.random.normal(0.0, x_shift_std * span)

        # Evaluate
        y, status = evaluate_formula(formula, x, safe_eval=safe_eval)
        if y is None:
            stats[status] += 1
            continue
        
        # Apply augmentation AFTER making a copy to preserve original for validation
        y_aug = y.copy()
        if y_scale_min != 1.0 or y_scale_max != 1.0:
            y_aug = y_aug * np.random.uniform(y_scale_min, y_scale_max)
        if y_offset_std > 0.0:
            y_aug = y_aug + np.random.normal(0.0, y_offset_std)
        
        # Noise injection: multi-SNR or legacy
        if noise_profile == 'multi':
            y_aug = apply_noise_augmentation(y_aug, noise_profile='multi')
        elif noise_std > 0.0:
            y_aug = y_aug + np.random.normal(0.0, noise_std * (np.std(y_aug) + 1e-10), size=y_aug.shape)

        # Extract features from augmented curve
        try:
            features = extract_all_features(y_aug)
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                stats['feature_invalid'] += 1
                continue
        except Exception:
            stats['feature_invalid'] += 1
            continue
        
        # Create labels
        labels = operators_to_labels(operators, formula=formula)
        
        features_local.append(features)
        labels_local.append(labels)
        formulas_local.append(formula)
            
    if len(features_local) < target:
        stats['max_attempts'] += 1

    return features_local, labels_local, formulas_local, stats


def save_dataset(
    filepath: Path,
    features: np.ndarray,
    labels: np.ndarray,
    formulas: List[str],
):
    """Save dataset to npz file."""
    np.savez_compressed(
        filepath,
        features=features,
        labels=labels,
        formulas=np.array(formulas, dtype=object),
        operator_classes=list(OPERATOR_CLASSES.keys()),
        feature_dim=FEATURE_DIM,
        feature_schema=FEATURE_SCHEMA,
    )
    print(f"Saved {len(features)} samples to {filepath}")


def load_dataset(filepath: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load dataset from npz file."""
    data = np.load(filepath, allow_pickle=True)
    return data['features'], data['labels'], data['formulas'].tolist()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate curve classifier training data")
    parser.add_argument("--n-samples", type=int, default=100000,
                        help="Number of samples to generate (default: 100000)")
    parser.add_argument("--output", type=str, default="data/curve_dataset.npz",
                        help="Output file path (default: data/curve_dataset.npz)")
    parser.add_argument("--x-min", type=float, default=-5,
                        help="Minimum x value (default: -5)")
    parser.add_argument("--x-max", type=float, default=5,
                        help="Maximum x value (default: 5)")
    parser.add_argument("--x-ranges", type=str, default="",
                        help="Comma-separated x ranges like '-1:1,-5:5,-10:10' (overrides --x-min/--x-max)")
    parser.add_argument("--x-scale-min", type=float, default=0.8,
                        help="Minimum multiplicative scale for x (default: 0.8)")
    parser.add_argument("--x-scale-max", type=float, default=1.2,
                        help="Maximum multiplicative scale for x (default: 1.2)")
    parser.add_argument("--x-shift-std", type=float, default=0.05,
                        help="Additive shift std for x as fraction of x span (default: 0.05)")
    parser.add_argument("--n-points", type=int, default=256,
                        help="Number of points per curve (default: 256)")
    parser.add_argument("--rational-ratio", type=float, default=0.0,
                        help="Fraction of samples forced to be rational P/Q (0-1). Example: 0.5 = half rational")
    parser.add_argument("--balance-templates", action="store_true",
                        help="Balance sampling across templates")
    parser.add_argument("--balance-classes", action="store_true",
                        help="Balance sampling across operator classes (overrides --balance-templates)")
    parser.add_argument("--no-balance-classes", action="store_true",
                        help="Disable class-balanced sampling")
    parser.add_argument("--n-workers", type=int, default=None,
                        help="Number of worker processes (default: cpu_count-1)")
    parser.add_argument("--stream", action="store_true",
                        help="Stream generation to disk to reduce memory usage")
    parser.add_argument("--no-formulas", action="store_true",
                        help="Do not store formulas in the output dataset")
    parser.add_argument("--noise-std", type=float, default=0.01,
                        help="Additive noise std as fraction of curve std (default: 0.01)")
    parser.add_argument("--y-scale-min", type=float, default=0.8,
                        help="Minimum multiplicative scale for y (default: 0.8)")
    parser.add_argument("--y-scale-max", type=float, default=1.2,
                        help="Maximum multiplicative scale for y (default: 1.2)")
    parser.add_argument("--y-offset-std", type=float, default=0.05,
                        help="Additive offset std for y (default: 0.05)")
    parser.add_argument("--safe-eval", action="store_true",
                        help="Use restricted AST-based evaluation instead of eval")
    parser.add_argument("--unsafe-eval", action="store_true",
                        help="Use eval instead of the restricted AST evaluator")
    parser.add_argument("--signed-bd", action="store_true",
                        help="Allow b and d coefficients to be negative (default: on)")
    parser.add_argument("--unsigned-bd", action="store_true",
                        help="Keep b and d coefficients positive")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--pcfg-ratio", type=float, default=0.0,
                        help="Fraction of samples generated via PCFG grammar (0-1, default: 0)")
    parser.add_argument("--pcfg-max-depth", type=int, default=4,
                        help="Maximum tree depth for PCFG formulas (default: 4)")
    parser.add_argument("--noise-profile", type=str, default='legacy',
                        choices=['legacy', 'multi'],
                        help="Noise injection mode: 'legacy' (fixed Gaussian) or 'multi' (randomized multi-SNR)")
    
    args = parser.parse_args()

    safe_eval = not args.unsafe_eval
    balance_classes = not args.no_balance_classes
    if args.balance_classes:
        balance_classes = True
    signed_bd = not args.unsigned_bd
    save_formulas = not args.no_formulas
    
    # Set seeds
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {args.n_samples} curves...")
    print(f"  x range: [{args.x_min}, {args.x_max}]")
    print(f"  n_points: {args.n_points}")
    if args.rational_ratio > 0:
        print(f"  rational_ratio: {args.rational_ratio:.2f}")
    if args.pcfg_ratio > 0:
        print(f"  pcfg_ratio: {args.pcfg_ratio:.2f} (max_depth={args.pcfg_max_depth})")
    print(f"  noise_profile: {args.noise_profile}")
    print(f"  n_classes: {N_CLASSES}")
    print(f"  Operators: {list(OPERATOR_CLASSES.keys())}")
    print()
    
    # Parse x-ranges if provided
    x_ranges = None
    if args.x_ranges.strip():
        x_ranges = []
        for part in args.x_ranges.split(','):
            bounds = part.split(':')
            if len(bounds) == 2:
                xmin = float(bounds[0])
                xmax = float(bounds[1])
                if xmin < xmax:
                    x_ranges.append((xmin, xmax))
        if not x_ranges:
            x_ranges = None

    # Generate (optionally balanced with rational-heavy subset)
    if args.rational_ratio > 0:
        if args.stream:
            raise ValueError("Streaming mode does not support --rational-ratio; generate separate datasets and merge offline.")
        n_rational = int(args.n_samples * args.rational_ratio)
        n_general = args.n_samples - n_rational
        print(f"  Generating {n_rational} rational + {n_general} general samples")
        
        feats_r, labels_r, forms_r = generate_dataset(
            n_samples=n_rational,
            x_range=(args.x_min, args.x_max),
            n_points=args.n_points,
            templates=RATIONAL_TEMPLATES,
            seed=args.seed,
            balance_templates=args.balance_templates,
            balance_classes=balance_classes,
            n_workers=args.n_workers,
            x_ranges=x_ranges,
            x_scale_min=args.x_scale_min,
            x_scale_max=args.x_scale_max,
            x_shift_std=args.x_shift_std,
            noise_std=args.noise_std,
            y_scale_min=args.y_scale_min,
            y_scale_max=args.y_scale_max,
            y_offset_std=args.y_offset_std,
            safe_eval=safe_eval,
            signed_bd=signed_bd,
            pcfg_ratio=args.pcfg_ratio,
            pcfg_max_depth=args.pcfg_max_depth,
            noise_profile=args.noise_profile,
        )
        feats_g, labels_g, forms_g = generate_dataset(
            n_samples=n_general,
            x_range=(args.x_min, args.x_max),
            n_points=args.n_points,
            templates=ALL_TEMPLATES,
            seed=args.seed + 1,
            balance_templates=args.balance_templates,
            balance_classes=balance_classes,
            n_workers=args.n_workers,
            x_ranges=x_ranges,
            x_scale_min=args.x_scale_min,
            x_scale_max=args.x_scale_max,
            x_shift_std=args.x_shift_std,
            noise_std=args.noise_std,
            y_scale_min=args.y_scale_min,
            y_scale_max=args.y_scale_max,
            y_offset_std=args.y_offset_std,
            safe_eval=safe_eval,
            signed_bd=signed_bd,
            pcfg_ratio=args.pcfg_ratio,
            pcfg_max_depth=args.pcfg_max_depth,
            noise_profile=args.noise_profile,
        )
        
        features = np.concatenate([feats_r, feats_g], axis=0)
        labels = np.concatenate([labels_r, labels_g], axis=0)
        formulas = forms_r + forms_g
        
        indices = np.random.permutation(len(features))
        features = features[indices]
        labels = labels[indices]
        formulas = [formulas[i] for i in indices]
    else:
        if args.stream:
            features, labels, formulas = generate_dataset_streamed(
                n_samples=args.n_samples,
                output_path=output_path,
                x_range=(args.x_min, args.x_max),
                n_points=args.n_points,
                seed=args.seed,
                balance_templates=args.balance_templates,
                balance_classes=balance_classes,
                n_workers=args.n_workers,
                x_ranges=x_ranges,
                x_scale_min=args.x_scale_min,
                x_scale_max=args.x_scale_max,
                x_shift_std=args.x_shift_std,
                noise_std=args.noise_std,
                y_scale_min=args.y_scale_min,
                y_scale_max=args.y_scale_max,
                y_offset_std=args.y_offset_std,
                safe_eval=safe_eval,
                signed_bd=signed_bd,
                save_formulas=save_formulas,
                pcfg_ratio=args.pcfg_ratio,
                pcfg_max_depth=args.pcfg_max_depth,
                noise_profile=args.noise_profile,
            )
        else:
            features, labels, formulas = generate_dataset(
                n_samples=args.n_samples,
                x_range=(args.x_min, args.x_max),
                n_points=args.n_points,
                seed=args.seed,
                balance_templates=args.balance_templates,
                balance_classes=balance_classes,
                n_workers=args.n_workers,
                x_ranges=x_ranges,
                x_scale_min=args.x_scale_min,
                x_scale_max=args.x_scale_max,
                x_shift_std=args.x_shift_std,
                noise_std=args.noise_std,
                y_scale_min=args.y_scale_min,
                y_scale_max=args.y_scale_max,
                y_offset_std=args.y_offset_std,
                safe_eval=safe_eval,
                signed_bd=signed_bd,
                pcfg_ratio=args.pcfg_ratio,
                pcfg_max_depth=args.pcfg_max_depth,
                noise_profile=args.noise_profile,
            )
    
    # Stats
    print(f"\nGenerated {len(features)} valid samples")
    print(f"Feature shape: {features.shape}")
    print(f"Label shape: {labels.shape}")
    
    # Class distribution
    print("\nClass distribution:")
    for name, idx in OPERATOR_CLASSES.items():
        count = labels[:, idx].sum()
        pct = 100 * count / len(labels)
        print(f"  {name:15s}: {int(count):6d} ({pct:5.1f}%)")
    
    # Save - skip for streaming mode (already saved to .dat files)
    if args.stream:
        print(f"\nStreamed data already saved to:")
        print(f"  {output_path.with_suffix('.features.dat')}")
        print(f"  {output_path.with_suffix('.labels.dat')}")
        if save_formulas:
            print(f"  {output_path.with_suffix('.formulas.txt')}")
    else:
        save_dataset(output_path, features, labels, formulas)
    
    # Show some examples (only if formulas were saved and not too many)
    if len(formulas) > 0 and len(formulas) <= 100000:
        print("\nExample formulas:")
        for i in range(min(5, len(formulas))):
            ops = [name for name, idx in OPERATOR_CLASSES.items() if labels[i, idx] > 0]
            print(f"  {formulas[i][:50]:50s} -> {ops}")


if __name__ == "__main__":
    main()
