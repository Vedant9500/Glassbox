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

# Make scipy optional - provide fallbacks
try:
    from scipy.stats import skew, kurtosis
except ImportError:
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
    # Unary operators (matching MetaPeriodic, MetaPower, MetaExp, MetaLog)
    'identity': 0,       # x (implicit in many formulas)
    'sin': 1,            # sin(ωx + φ)
    'cos': 2,            # cos(ωx + φ)
    'power': 3,          # x^p (includes x, x², x³, √x, 1/x)
    'exp': 4,            # e^(βx)
    'log': 5,            # log(|x| + ε)
    
    # Structure indicators (higher-level patterns)
    'periodic': 6,       # Contains sin or cos
    'polynomial': 7,     # Pure polynomial (only power ops)
    'exponential': 8,    # Contains exp or log
    
    # Binary indicators
    'addition': 9,       # Terms added
    'multiplication': 10, # Terms multiplied
    
    # NEW: Rational functions
    'rational': 11,      # 1/(x+c), x/(x²+c) type
}

N_CLASSES = len(OPERATOR_CLASSES)


# =============================================================================
# FORMULA TEMPLATES
# =============================================================================

# Each template: (formula_string, set_of_operators)
# Placeholders: {a}, {b}, {c} = random coefficients, {p} = random power

SIMPLE_TEMPLATES = [
    # Identity / Linear
    ("x", {'identity', 'power', 'polynomial'}),
    ("{a} * x", {'identity', 'power', 'polynomial', 'multiplication'}),
    ("{a} * x + {b}", {'identity', 'power', 'polynomial', 'addition', 'multiplication'}),
    
    # Powers
    ("x ** 2", {'power', 'polynomial'}),
    ("x ** 3", {'power', 'polynomial'}),
    ("x ** {p}", {'power', 'polynomial'}),
    ("{a} * x ** 2", {'power', 'polynomial', 'multiplication'}),
    ("{a} * x ** 2 + {b} * x + {c}", {'power', 'polynomial', 'addition', 'multiplication'}),
    ("np.sqrt(np.abs(x) + 0.01)", {'power', 'polynomial'}),
    ("1 / (np.abs(x) + 0.1)", {'power', 'polynomial'}),
    
    # Periodic
    ("np.sin(x)", {'sin', 'periodic'}),
    ("np.cos(x)", {'cos', 'periodic'}),
    ("np.sin({a} * x)", {'sin', 'periodic', 'multiplication'}),
    ("np.cos({a} * x)", {'cos', 'periodic', 'multiplication'}),
    ("np.sin({a} * x + {b})", {'sin', 'periodic', 'addition', 'multiplication'}),
    ("{a} * np.sin({b} * x)", {'sin', 'periodic', 'multiplication'}),
    ("{a} * np.cos({b} * x)", {'cos', 'periodic', 'multiplication'}),
    
    # Exponential
    ("np.exp({a} * x)", {'exp', 'exponential', 'multiplication'}),
    ("np.exp(-x ** 2)", {'exp', 'power', 'exponential'}),
    ("{a} * np.exp({b} * x)", {'exp', 'exponential', 'multiplication'}),
    
    # Logarithmic
    ("np.log(np.abs(x) + 1)", {'log', 'exponential'}),
    ("{a} * np.log(np.abs(x) + 1)", {'log', 'exponential', 'multiplication'}),
]

COMPOUND_TEMPLATES = [
    # Polynomial combinations
    ("x ** 2 + x", {'power', 'polynomial', 'addition'}),
    ("x ** 3 - x", {'power', 'polynomial', 'addition'}),
    ("{a} * x ** 3 + {b} * x ** 2 + {c} * x", {'power', 'polynomial', 'addition', 'multiplication'}),
    
    # Periodic + Polynomial
    ("np.sin(x) + x", {'sin', 'periodic', 'power', 'addition'}),
    ("np.sin(x) + x ** 2", {'sin', 'periodic', 'power', 'addition'}),
    ("np.cos(x) + x ** 2", {'cos', 'periodic', 'power', 'addition'}),
    ("{a} * np.sin({b} * x) + {c} * x", {'sin', 'periodic', 'power', 'addition', 'multiplication'}),
    ("x * np.sin(x)", {'sin', 'periodic', 'power', 'multiplication'}),
    ("x ** 2 * np.cos(x)", {'cos', 'periodic', 'power', 'multiplication'}),
    
    # Periodic combinations
    ("np.sin(x) + np.cos(x)", {'sin', 'cos', 'periodic', 'addition'}),
    ("np.sin(x) * np.cos(x)", {'sin', 'cos', 'periodic', 'multiplication'}),
    ("{a} * np.sin({b} * x) + {c} * np.cos({d} * x)", {'sin', 'cos', 'periodic', 'addition', 'multiplication'}),
    
    # Exponential combinations
    ("np.exp(x) + np.exp(-x)", {'exp', 'exponential', 'addition'}),  # cosh-like
    ("np.exp(x) - np.exp(-x)", {'exp', 'exponential', 'addition'}),  # sinh-like
    ("np.exp(-x ** 2 / 2)", {'exp', 'power', 'exponential'}),  # Gaussian
    ("x * np.exp(-x)", {'exp', 'power', 'exponential', 'multiplication'}),
    
    # Log combinations
    ("x * np.log(np.abs(x) + 1)", {'log', 'power', 'exponential', 'multiplication'}),
    ("np.log(np.abs(x) + 1) ** 2", {'log', 'power', 'exponential'}),
    
    # Mixed
    ("np.sin(x) * np.exp(-x ** 2)", {'sin', 'exp', 'periodic', 'exponential', 'multiplication'}),
    ("np.log(np.abs(x) + 1) + np.sin(x)", {'log', 'sin', 'periodic', 'exponential', 'addition'}),
]

# NEW: Rational function templates
RATIONAL_TEMPLATES = [
    # Simple rational (1/x type)
    ("1 / (np.abs(x) + {c})", {'rational', 'power'}),
    ("{a} / (np.abs(x) + {c})", {'rational', 'power', 'multiplication'}),
    
    # Quadratic denominator (Lorentzian type)
    ("1 / (x**2 + {c})", {'rational', 'power'}),
    ("{a} / (x**2 + {c})", {'rational', 'power', 'multiplication'}),
    ("x / (x**2 + {c})", {'rational', 'power'}),
    
    # Mixed rational
    ("(x + {a}) / (x**2 + {b})", {'rational', 'power', 'addition'}),
    ("x**2 / (x**2 + {c})", {'rational', 'power'}),
    ("(x**2 + {a}) / (x**2 + {b})", {'rational', 'power', 'addition'}),
    
    # Rational + other
    ("1 / (x**2 + 1) + np.sin(x)", {'rational', 'sin', 'periodic', 'addition'}),
    ("x / (x**2 + 1) + {a}", {'rational', 'power', 'addition'}),
]

ALL_TEMPLATES = SIMPLE_TEMPLATES + COMPOUND_TEMPLATES + RATIONAL_TEMPLATES


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_raw_features(y: np.ndarray, n_points: int = 128) -> np.ndarray:
    """Normalize and resample curve to fixed size."""
    if len(y) != n_points:
        x_old = np.linspace(0, 1, len(y))
        x_new = np.linspace(0, 1, n_points)
        y = np.interp(x_new, x_old, y)
    
    # Normalize to [0, 1] range
    y_min, y_max = y.min(), y.max()
    if y_max - y_min > 1e-10:
        y_norm = (y - y_min) / (y_max - y_min)
    else:
        y_norm = np.zeros_like(y)
    
    return y_norm


def extract_fft_features(y: np.ndarray, n_freqs: int = 32) -> np.ndarray:
    """Extract dominant frequencies - detects periodicity."""
    fft = np.fft.rfft(y)
    magnitudes = np.abs(fft)[:n_freqs]
    
    # Normalize
    mag_max = magnitudes.max()
    if mag_max > 1e-10:
        magnitudes = magnitudes / mag_max
    
    return magnitudes


def extract_derivative_features(y: np.ndarray, n_points: int = 64) -> np.ndarray:
    """First and second derivatives - detect polynomial degree, inflection points."""
    dy = np.diff(y)   # First derivative
    ddy = np.diff(dy) # Second derivative
    
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
    
    return np.array(features)


def extract_curvature_features(y: np.ndarray, n_points: int = 32) -> np.ndarray:
    """
    Extract curvature features to distinguish rational from exponential decay.
    Rational functions have distinct acceleration profiles.
    """
    # First derivative (velocity)
    dy = np.gradient(y)
    
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
    raw = extract_raw_features(y, n_points=128)      # 128
    fft = extract_fft_features(y, n_freqs=32)        # 32
    deriv = extract_derivative_features(y, n_points=64)  # 128
    stats = extract_stat_features(y)                  # 9
    curv = extract_curvature_features(y, n_points=32) # 37 (32 + 5)
    
    return np.concatenate([raw, fft, deriv, stats, curv])  # Total: 334


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


def evaluate_formula(formula: str, x: np.ndarray) -> np.ndarray:
    """Safely evaluate a formula string."""
    try:
        # Suppress warnings during evaluation (domain errors are expected)
        with np.errstate(all='ignore'):
            y = eval(formula, {"x": x, "np": np})
        
        # Check for invalid values
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            return None
        
        # Check for extreme values
        if np.abs(y).max() > 1e6:
            return None
        
        return y
    except Exception:
        return None


def operators_to_labels(operators: Set[str]) -> np.ndarray:
    """Convert operator set to multi-hot label vector."""
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
    templates: List[Tuple[str, Set[str]]] = ALL_TEMPLATES,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Generate labeled curve dataset.
    
    Returns:
        features: (n_samples, 297) feature vectors
        labels: (n_samples, n_classes) multi-hot labels
        formulas: list of formula strings (for debugging)
    """
    x = np.linspace(x_range[0], x_range[1], n_points)
    
    features_list = []
    labels_list = []
    formulas_list = []
    
    iterator = range(n_samples * 2)  # Generate extra to account for failures
    
    # Multiprocessing for speed
    import multiprocessing
    from functools import partial
    
    # Determine number of processes
    n_workers = max(1, multiprocessing.cpu_count() - 1)
    
    # Split workload
    chunk_size = n_samples // n_workers
    chunks = [chunk_size] * n_workers
    # Add remainder to last chunk
    chunks[-1] += n_samples - sum(chunks)
    
    print(f"Generating data using {n_workers} workers...")
    
    with multiprocessing.Pool(n_workers) as pool:
        # Create partial function with fixed arguments
        worker_func = partial(
            generate_chunk, 
            x_range=x_range, 
            n_points=n_points,
            templates=templates,
        )
        
        # Run workers
        results = list(tqdm(
            pool.imap(worker_func, chunks), 
            total=n_workers, 
            desc="Generating chunks"
        ))
    
    # Combine results
    features_list = []
    labels_list = []
    formulas_list = []
    
    for feats, lbls, forms in results:
        features_list.extend(feats)
        labels_list.extend(lbls)
        formulas_list.extend(forms)
    
    # Shuffle
    indices = np.random.permutation(len(features_list))
    features = np.array(features_list, dtype=np.float32)[indices]
    labels = np.array(labels_list, dtype=np.float32)[indices]
    formulas = [formulas_list[i] for i in indices]
    
    # Trim to exact size
    features = features[:n_samples]
    labels = labels[:n_samples]
    formulas = formulas[:n_samples]
    
    return features, labels, formulas


def generate_chunk(
    n: int,
    x_range: Tuple[float, float],
    n_points: int,
    templates: List[Tuple[str, Set[str]]],
) -> Tuple[List, List, List]:
    """Generate a chunk of samples (worker function)."""
    # Re-seed for each process to ensure diversity
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
    
    while len(features_local) < target and attempts < max_attempts:
        attempts += 1
        
        # Generate random formula from selected template pool
        template, operators = random.choice(templates)
        formula = template.format(
            a=np.random.uniform(-3, 3),
            b=np.random.uniform(0.5, 3),
            c=np.random.uniform(-2, 2),
            d=np.random.uniform(0.5, 3),
            p=np.random.choice([0.5, 2, 3, 4, -1, -0.5]),
        )
        
        # Evaluate
        y = evaluate_formula(formula, x)
        if y is None:
            continue
        
        # Extract features
        try:
            features = extract_all_features(y)
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                continue
        except Exception:
            continue
        
        # Create labels
        labels = operators_to_labels(operators)
        
        features_local.append(features)
        labels_local.append(labels)
        formulas_local.append(formula)
            
    return features_local, labels_local, formulas_local


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
    parser.add_argument("--n-points", type=int, default=256,
                        help="Number of points per curve (default: 256)")
    parser.add_argument("--rational-ratio", type=float, default=0.0,
                        help="Fraction of samples forced to be rational P/Q (0-1). Example: 0.5 = half rational")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
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
    print(f"  n_classes: {N_CLASSES}")
    print(f"  Operators: {list(OPERATOR_CLASSES.keys())}")
    print()
    
    # Generate (optionally balanced with rational-heavy subset)
    if args.rational_ratio > 0:
        n_rational = int(args.n_samples * args.rational_ratio)
        n_general = args.n_samples - n_rational
        print(f"  Generating {n_rational} rational + {n_general} general samples")
        
        feats_r, labels_r, forms_r = generate_dataset(
            n_samples=n_rational,
            x_range=(args.x_min, args.x_max),
            n_points=args.n_points,
            templates=RATIONAL_TEMPLATES,
        )
        feats_g, labels_g, forms_g = generate_dataset(
            n_samples=n_general,
            x_range=(args.x_min, args.x_max),
            n_points=args.n_points,
            templates=ALL_TEMPLATES,
        )
        
        features = np.concatenate([feats_r, feats_g], axis=0)
        labels = np.concatenate([labels_r, labels_g], axis=0)
        formulas = forms_r + forms_g
        
        # Shuffle combined
        indices = np.random.permutation(len(features))
        features = features[indices]
        labels = labels[indices]
        formulas = [formulas[i] for i in indices]
    else:
        features, labels, formulas = generate_dataset(
            n_samples=args.n_samples,
            x_range=(args.x_min, args.x_max),
            n_points=args.n_points,
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
    
    # Save
    save_dataset(output_path, features, labels, formulas)
    
    # Show some examples
    print("\nExample formulas:")
    for i in range(min(5, len(formulas))):
        ops = [name for name, idx in OPERATOR_CLASSES.items() if labels[i, idx] > 0]
        print(f"  {formulas[i][:50]:50s} -> {ops}")


if __name__ == "__main__":
    main()
