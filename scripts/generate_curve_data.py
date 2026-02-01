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
from scipy.stats import skew, kurtosis
from tqdm import tqdm


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

ALL_TEMPLATES = SIMPLE_TEMPLATES + COMPOUND_TEMPLATES


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


def extract_all_features(y: np.ndarray) -> np.ndarray:
    """Combine all features into single vector."""
    raw = extract_raw_features(y, n_points=128)      # 128
    fft = extract_fft_features(y, n_freqs=32)        # 32
    deriv = extract_derivative_features(y, n_points=64)  # 128
    stats = extract_stat_features(y)                  # 9
    
    return np.concatenate([raw, fft, deriv, stats])   # Total: 297


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
    if show_progress:
        iterator = tqdm(iterator, desc="Generating curves")
    
    for _ in iterator:
        if len(features_list) >= n_samples:
            break
        
        # Generate random formula
        formula, operators = generate_random_formula()
        
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
        
        features_list.append(features)
        labels_list.append(labels)
        formulas_list.append(formula)
    
    return (
        np.array(features_list, dtype=np.float32),
        np.array(labels_list, dtype=np.float32),
        formulas_list,
    )


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
    print(f"  n_classes: {N_CLASSES}")
    print(f"  Operators: {list(OPERATOR_CLASSES.keys())}")
    print()
    
    # Generate
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
