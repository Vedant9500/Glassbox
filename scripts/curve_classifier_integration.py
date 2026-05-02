"""
Curve Classifier Integration with ONN

Uses the trained curve classifier to predict operators and warm-start ONN evolution.

Usage:
    # Test on synthetic data
    python scripts/curve_classifier_integration.py --model models/curve_classifier_v3.1.pt --formula "sin(x) + x**2"
    
    # Integrate with ONN (in your training script)
    from scripts.curve_classifier_integration import predict_operators, bias_onn_from_predictions
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from scripts.generate_curve_data import extract_all_features, OPERATOR_CLASSES
except ImportError:
    from generate_curve_data import extract_all_features, OPERATOR_CLASSES


DEFAULT_CURVE_CLASSIFIER_PATH = "models/curve_classifier_v3.1.pt"


# =============================================================================
# MODEL DEFINITION (must match training)
# =============================================================================

class CurveClassifierMLP(nn.Module):
    """Simple MLP classifier for curve features."""
    
    def __init__(self, n_features: int = 370, n_classes: int = 9, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden // 2, n_classes),
        )
    
    def forward(self, x):
        return self.net(x)


class CurveClassifierCNN(nn.Module):
    """1D CNN classifier matching training architecture."""

    def __init__(self, n_classes: int = 9, n_features: int = 370, curve_dim: int = 128):
        super().__init__()
        self.curve_dim = min(curve_dim, n_features)

        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
        )

        other_dim = max(1, n_features - self.curve_dim)
        self.other_mlp = nn.Sequential(
            nn.Linear(other_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        raw_curve = x[:, :self.curve_dim]
        other_features = x[:, self.curve_dim:]

        raw_curve = raw_curve.unsqueeze(1)
        conv_out = self.conv(raw_curve)
        conv_out = conv_out.flatten(1)
        other_out = self.other_mlp(other_features)

        combined = torch.cat([conv_out, other_out], dim=1)
        return self.classifier(combined)


class CurveClassifierGLU(nn.Module):
    """
    First-Principles Mathematical Classifier using Gated Linear Units (GLU).
    Mathematically models multiplicative function composition (e.g. x * sin(x)) natively.
    Replaces both the redundant CNN and deep ReLU MLPs with a cache-contiguous 2-layer network.
    """
    def __init__(self, n_features: int = 370, n_classes: int = 9, hidden: int = 256):
        super().__init__()
        
        # A GLU layer splits its output in half along dim=1. 
        # To maintain a 'hidden' dimension size, we project to hidden * 2.
        self.fc1 = nn.Linear(n_features, hidden * 2)
        self.bn1 = nn.BatchNorm1d(hidden * 2)
        
        self.fc2 = nn.Linear(hidden, hidden * 2)
        self.bn2 = nn.BatchNorm1d(hidden * 2)
        
        self.classifier = nn.Linear(hidden, n_classes)
        self.dropout = nn.Dropout(0.2)

        self._init_weights()

    def _init_weights(self):
        """Hardware-sympathetic initialization for multiplicative gating."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier initialization optimizes variance for the linear path
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    # Initialize biases slightly positive so the sigmoid gate starts "open"
                    nn.init.constant_(m.bias, 0.1)
                    
    def forward(self, x):
        # Layer 1: Invariants and FFT multi-hot gate the raw derivatives
        # Mathematically computes: (xW_1 + b_1) ⊗ σ(xW_2 + b_2)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.glu(x, dim=1)
        x = self.dropout(x)
        
        # Layer 2: Higher-order multiplicative compositions (e.g. power * exp * trig)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.glu(x, dim=1)
        x = self.dropout(x)
        
        # Linear projection to class logits
        return self.classifier(x)


# =============================================================================
# CLASSIFIER LOADING
# =============================================================================

_cached_classifier_by_device = {}
_cached_operator_classes_by_key = {}
_cached_metadata_by_device = {}
_cached_interpolators_by_signature: Dict[Tuple, Tuple[Optional[object], object]] = {}
_warned_no_cuda = False


def _resolve_device(device: Optional[str] = None) -> torch.device:
    global _warned_no_cuda
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        if not _warned_no_cuda:
            print("CUDA requested but not available; falling back to CPU.")
            _warned_no_cuda = True
        return torch.device("cpu")

    return resolved


def _make_cache_key(model_path: str, resolved_device: torch.device) -> str:
    return f"{str(resolved_device)}:{str(Path(model_path).resolve())}"


def load_classifier(
    model_path: str = DEFAULT_CURVE_CLASSIFIER_PATH,
    device: Optional[str] = None,
):
    """Load the trained curve classifier (supports PyTorch .pt and XGBoost .pkl)."""
    global _cached_classifier_by_device
    
    resolved_device = _resolve_device(device)
    # Create cache key using both device and absolute model path
    cache_key = _make_cache_key(model_path, resolved_device)
    if cache_key in _cached_classifier_by_device:
        return _cached_classifier_by_device[cache_key]
    
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        # Clean failure if model not found
        raise FileNotFoundError(f"Classifier model not found at {model_path}")
    
    # Check file extension to determine model type
    if model_path_obj.suffix in ('.pkl', '.joblib'):
        # XGBoost model
        return _load_xgboost_classifier(model_path_obj, cache_key)
    else:
        # PyTorch model
        return _load_pytorch_classifier(model_path_obj, resolved_device, cache_key)


def _load_xgboost_classifier(model_path: Path, cache_key: str):
    """Load XGBoost classifier from .pkl file."""
    global _cached_classifier_by_device, _cached_operator_classes_by_key, _cached_metadata_by_device
    
    import joblib
    
    payload = joblib.load(model_path)
    
    operator_classes = payload.get('operator_classes', list(OPERATOR_CLASSES.keys()))
    _cached_operator_classes_by_key[cache_key] = operator_classes
    
    # Store the XGBoost models and metadata
    _cached_classifier_by_device[cache_key] = {
        'type': 'xgboost',
        'models': payload['models'],
        'thresholds': payload.get('thresholds'),
    }
    _cached_metadata_by_device[cache_key] = {
        'thresholds': payload.get('thresholds'),
        'feature_scaler': payload.get('feature_scaler'),
        'type': 'xgboost',
        'operator_classes': operator_classes,
        'isotonic_calibration': payload.get('isotonic_calibration'),
    }
    
    print(f"Loaded XGBoost curve classifier from {model_path}")
    print(f"  {len([m for m in payload['models'] if m is not None])} active classifiers")
    
    return _cached_classifier_by_device[cache_key]


def _load_pytorch_classifier(model_path: Path, resolved_device: torch.device, cache_key: str) -> nn.Module:
    """Load PyTorch classifier from .pt file."""
    global _cached_classifier_by_device, _cached_operator_classes_by_key, _cached_metadata_by_device
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Error loading checkpoint from {model_path}: {e}")
        raise
    
    # Get operator classes
    operator_classes = checkpoint.get('operator_classes', list(OPERATOR_CLASSES.keys()))
    _cached_operator_classes_by_key[cache_key] = operator_classes
    n_classes = len(operator_classes)
    
    state_dict = checkpoint['model_state_dict']
    model_type = checkpoint.get('model_type')
    model_config = checkpoint.get('model_config') or {}

    # Backward-compatible architecture detection for older checkpoints
    if model_type is None:
        if any(k.startswith('conv.') for k in state_dict.keys()):
            model_type = 'cnn'
        elif any(k.startswith('net.') for k in state_dict.keys()):
            model_type = 'mlp'
        else:
            raise ValueError(
                "Unable to infer classifier architecture from checkpoint; "
                "expected MLP keys ('net.*') or CNN keys ('conv.*')."
            )

    if model_type == 'cnn':
        if 'n_features' in model_config:
            n_features = int(model_config['n_features'])
        else:
            # Derive from first conv classifier layer input: 128*4 + 128(other)
            classifier_in = state_dict['classifier.0.weight'].shape[1]
            n_features = int(max(1, classifier_in - (128 * 4)))

        curve_dim = int(model_config.get('curve_dim', min(128, n_features)))
        model = CurveClassifierCNN(
            n_classes=int(model_config.get('n_classes', n_classes)),
            n_features=n_features,
            curve_dim=curve_dim,
        )
    elif model_type == 'glu':
        input_weights = state_dict['fc1.weight']
        n_features = int(model_config.get('n_features', input_weights.shape[1]))
        hidden_size = int(model_config.get('hidden', input_weights.shape[0] // 2))
        model = CurveClassifierGLU(n_features=n_features, n_classes=n_classes, hidden=hidden_size)
    else:
        input_weights = state_dict['net.0.weight']
        n_features = int(model_config.get('n_features', input_weights.shape[1]))
        hidden_size = int(model_config.get('hidden', input_weights.shape[0]))
        model = CurveClassifierMLP(n_features=n_features, n_classes=n_classes, hidden=hidden_size)

    model.load_state_dict(state_dict)
    model.to(resolved_device)
    model.eval()

    _cached_classifier_by_device[cache_key] = model
    _cached_metadata_by_device[cache_key] = {
        'thresholds': checkpoint.get('thresholds'),
        'temperature': checkpoint.get('temperature'),
        'feature_scaler': checkpoint.get('feature_scaler'),
        'type': 'pytorch',
        'model_type': model_type,
        'operator_classes': operator_classes,
        'isotonic_calibration': checkpoint.get('isotonic_calibration'),
    }
    print(f"Loaded PyTorch curve classifier from {model_path}")
    if 'val_acc' in checkpoint:
        print(f"  Val accuracy: {checkpoint.get('val_acc'):.4f}")
    print(f"  Device: {resolved_device}")
    
    return model


# =============================================================================
# PREDICTION
# =============================================================================

def _predict_xgboost(model_dict: dict, features: np.ndarray, metadata: dict | None = None) -> np.ndarray:
    """Predict using XGBoost models, with optional isotonic calibration."""
    models = model_dict['models']
    n_classes = len(models)
    probs = np.zeros(n_classes, dtype=np.float32)
    
    features_2d = features.reshape(1, -1)
    
    for i, m in enumerate(models):
        if m is None:
            probs[i] = 0.0
        else:
            probs[i] = m.predict_proba(features_2d)[0, 1]
    
    # Apply per-class isotonic calibration if available
    if metadata and metadata.get('isotonic_calibration'):
        probs = _apply_isotonic_calibration(probs, metadata['isotonic_calibration'])
    
    return probs


def _predict_pytorch(model: nn.Module, features: np.ndarray, metadata: dict, device: torch.device) -> np.ndarray:
    """Predict using PyTorch model, with optional isotonic calibration."""
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(features_tensor).squeeze()
        temperature = metadata.get('temperature')
        if temperature is not None:
            logits = logits / float(temperature)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
    
    # Apply per-class isotonic calibration if available
    isotonic_maps = metadata.get('isotonic_calibration')
    if isotonic_maps:
        probs = _apply_isotonic_calibration(probs, isotonic_maps)
    
    return probs


def _apply_isotonic_calibration(
    raw_probs: np.ndarray,
    calibration_maps: list,
) -> np.ndarray:
    """Apply per-class isotonic regression calibration to raw probabilities.
    
    Each calibration map is a dict with 'boundaries' (bin edges) and 'values'
    (calibrated probability for each bin). Uses np.digitize for fast lookup.
    """
    if not calibration_maps:
        return raw_probs
    
    single = raw_probs.ndim == 1
    if single:
        raw_probs = raw_probs.reshape(1, -1)
    
    calibrated = raw_probs.copy()
    n_classes = raw_probs.shape[1]
    
    for c in range(min(n_classes, len(calibration_maps))):
        cmap = calibration_maps[c]
        boundaries = np.array(cmap['boundaries'])
        values = np.array(cmap['values'])
        indices = np.digitize(raw_probs[:, c], boundaries, right=False) - 1
        indices = np.clip(indices, 0, len(values) - 1)
        calibrated[:, c] = values[indices]
    
    if single:
        return calibrated[0]
    return calibrated


def predict_operators(
    x: np.ndarray,
    y: np.ndarray,
    model_path: str = DEFAULT_CURVE_CLASSIFIER_PATH,
    threshold: float = 0.5,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """
    Predict which operators are likely present in the data.
    
    For multi-input data (n_vars > 1), uses per-variable 1D slicing:
    - Takes 1D cross-sections through the data (fixing other vars at midpoint)
    - Runs classifier on each slice
    - Aggregates predictions across all variables
    
    Args:
        x: Input values - 1D array (N,) or 2D array (N, n_vars)
        y: Output values (1D array)
        model_path: Path to trained classifier
        threshold: Probability threshold for reporting
        
    Returns:
        Dictionary mapping operator names to probabilities
    """
    # Check if model exists before trying to load
    if not Path(model_path).exists():
        print(f"Warning: Curve classifier model not found at {model_path}. Skipping prediction.")
        return {}

    # Load classifier
    try:
        model = load_classifier(model_path, device=device)
    except Exception as e:
        print(f"Warning: Failed to load curve classifier: {e}")
        return {}
    
    # Get cache key for metadata lookup
    resolved_device = _resolve_device(device)
    cache_key = _make_cache_key(model_path, resolved_device)
    metadata = _cached_metadata_by_device.get(cache_key, {})
    
    # Detect multi-input
    x = np.asarray(x)
    y = np.asarray(y).flatten()
    
    if x.ndim == 1:
        n_vars = 1
        x = x.reshape(-1, 1)
    else:
        n_vars = x.shape[1]
    
    # For multi-input: use per-variable slicing
    if n_vars > 1:
        return _predict_operators_multi_input(
            x, y, model, metadata, resolved_device, threshold, n_vars, cache_key
        )
    
    # Single-input: standard prediction
    features = extract_all_features(y)
    scaler = metadata.get('feature_scaler')
    if scaler is not None:
        features = (features - scaler['mean']) / (scaler['std'] + 1e-8)
    
    # Check if this is XGBoost or PyTorch model
    if metadata.get('type') == 'xgboost':
        probs = _predict_xgboost(model, features, metadata)
    else:
        probs = _predict_pytorch(model, features, metadata, resolved_device)
    
    return _build_result_dict(probs, threshold, metadata, cache_key)


def _predict_operators_multi_input(
    x: np.ndarray,
    y: np.ndarray,
    model,
    metadata: dict,
    device: torch.device,
    threshold: float,
    n_vars: int,
    cache_key: str,
) -> Dict[str, float]:
    """
    Predict operators for multi-input data using per-variable 1D slicing.
    
    For each variable:
    1. Fix all other variables at their median value
    2. Vary the target variable across its range
    3. Compute y values for this slice
    4. Run classifier on the 1D slice
    5. Aggregate predictions across all variables (max probability)
    """
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    
    # Build/reuse interpolators for y values (hot path for repeated calls)
    interp_signature = (
        int(x.__array_interface__['data'][0]),
        tuple(x.shape),
        str(x.dtype),
        int(y.__array_interface__['data'][0]),
        tuple(y.shape),
        str(y.dtype),
    )
    interpolators = _cached_interpolators_by_signature.get(interp_signature)
    if interpolators is None:
        linear_interp = None
        nearest_interp = None
        try:
            linear_interp = LinearNDInterpolator(x, y, fill_value=np.nan)
        except Exception:
            linear_interp = None
        try:
            nearest_interp = NearestNDInterpolator(x, y)
        except Exception:
            nearest_interp = None
        if linear_interp is None and nearest_interp is None:
            raise RuntimeError("Failed to build interpolation models for multi-input prediction.")
        interpolators = (linear_interp, nearest_interp)
        _cached_interpolators_by_signature[interp_signature] = interpolators
        if len(_cached_interpolators_by_signature) > 8:
            _cached_interpolators_by_signature.pop(next(iter(_cached_interpolators_by_signature)))

    linear_interp, nearest_interp = interpolators

    operator_classes = (
        metadata.get('operator_classes')
        or _cached_operator_classes_by_key.get(cache_key)
        or list(OPERATOR_CLASSES.keys())
    )
    all_probs = np.zeros((n_vars, len(operator_classes)), dtype=np.float32)
    
    scaler = metadata.get('feature_scaler')
    x_medians = np.median(x, axis=0)
    
    for var_idx in range(n_vars):
        # Create 1D slice: fix other variables at median, vary this one
        x_min_var = x[:, var_idx].min()
        x_max_var = x[:, var_idx].max()
        
        # Sample points along this variable
        n_slice_points = min(256, len(y))
        x_slice_1d = np.linspace(x_min_var, x_max_var, n_slice_points)
        
        # Build full query points (other vars at median)
        x_query = np.tile(x_medians, (n_slice_points, 1))
        x_query[:, var_idx] = x_slice_1d
        
        # Get y values for this slice
        if linear_interp is not None:
            y_slice = linear_interp(x_query)
            if nearest_interp is not None:
                nan_mask = ~np.isfinite(y_slice)
                if np.any(nan_mask):
                    y_slice[nan_mask] = nearest_interp(x_query[nan_mask])
        else:
            y_slice = nearest_interp(x_query)
        
        # Handle NaN values from interpolation
        valid_mask = np.isfinite(y_slice)
        if valid_mask.sum() < 10:
            # Not enough valid points, skip this variable
            continue
        
        y_slice_valid = y_slice[valid_mask]
        
        # Extract features and predict
        try:
            features = extract_all_features(y_slice_valid)
            if scaler is not None:
                features = (features - scaler['mean']) / (scaler['std'] + 1e-8)
            
            if metadata.get('type') == 'xgboost':
                probs = _predict_xgboost(model, features, metadata)
            else:
                probs = _predict_pytorch(model, features, metadata, device)
            
            all_probs[var_idx] = probs
        except Exception as e:
            # Skip this variable if feature extraction fails
            print(f"  Warning: Slice {var_idx} failed: {e}")
            continue
    
    # Aggregate: use max probability across all variables
    # This captures operators that appear in ANY variable
    aggregated_probs = np.max(all_probs, axis=0)
    
    return _build_result_dict(aggregated_probs, threshold, metadata, cache_key)


def _build_result_dict(probs: np.ndarray, threshold: float, metadata: dict, cache_key: str) -> Dict[str, float]:
    """Build result dictionary from probability array."""
    operator_classes = (
        metadata.get('operator_classes')
        or _cached_operator_classes_by_key.get(cache_key)
        or list(OPERATOR_CLASSES.keys())
    )
    thresholds = metadata.get('thresholds')
    if thresholds is None:
        thresholds = np.full((len(operator_classes),), threshold, dtype=np.float32)
    
    result = {}
    for i, name in enumerate(operator_classes):
        if i < len(probs) and probs[i] >= thresholds[i]:
            result[name] = float(probs[i])
    
    # Derived compatibility outputs
    name_to_idx = {name: i for i, name in enumerate(operator_classes)}
    periodic_prob = max(
        probs[name_to_idx.get('sin', 0)] if 'sin' in name_to_idx else 0.0,
        probs[name_to_idx.get('cos', 0)] if 'cos' in name_to_idx else 0.0,
    )
    exponential_prob = max(
        probs[name_to_idx.get('exp', 0)] if 'exp' in name_to_idx else 0.0,
        probs[name_to_idx.get('log', 0)] if 'log' in name_to_idx else 0.0,
    )
    polynomial_prob = max(
        probs[name_to_idx.get('power', 0)] if 'power' in name_to_idx else 0.0,
        probs[name_to_idx.get('identity', 0)] if 'identity' in name_to_idx else 0.0,
    )

    if periodic_prob >= threshold:
        result['periodic'] = float(periodic_prob)
    if exponential_prob >= threshold:
        result['exponential'] = float(exponential_prob)
    if polynomial_prob >= threshold:
        result['polynomial'] = float(polynomial_prob)

    return result


def print_predictions(predictions: Dict[str, float]):
    """Pretty print predictions."""
    print("\nPredicted operators:")
    sorted_preds = sorted(predictions.items(), key=lambda x: -x[1])
    for name, prob in sorted_preds:
        bar = "█" * int(prob * 20)
        print(f"  {name:15s}: {prob:.3f} {bar}")


# =============================================================================
# ONN INTEGRATION
# =============================================================================

def bias_onn_from_predictions(
    model,
    predictions: Dict[str, float],
    threshold: float = 0.3,
    boost_factor: float = 2.0,
    verbose: bool = True,
):
    """
    Use classifier predictions to bias ONN operation selection.
    
    Args:
        model: OperationDAG model
        predictions: Dict from predict_operators()
        threshold: Minimum probability to apply bias
        boost_factor: How much to boost logits for predicted operators
    """
    # Mapping from classifier classes to ONN meta-op indices
    # HardConcreteOperationSelector.logits layout:
    # [0:2] = type weights (unary vs binary)
    # [2:2+n_unary] = unary op weights
    # [2+n_unary:] = binary op weights
    
    # For simplified_ops=True: n_unary=2, n_binary=1
    # unary_ops: [MetaPeriodic (0), MetaPower (1)]
    
    # For simplified_ops=False: n_unary=4, n_binary=2
    # unary_ops: [MetaPeriodic (0), MetaPower (1), MetaExp (2), MetaLog (3)]
    
    simplified = getattr(model, 'simplified_ops', True)
    
    periodic_prob = max(predictions.get('sin', 0), predictions.get('cos', 0))

    if simplified:
        unary_map = {
            'sin': 0, 'cos': 0,  # MetaPeriodic
            'power': 1, 'identity': 1,  # MetaPower
            'rational': 1,  # Bias toward reciprocal via MetaPower
        }
        n_unary = 2
    else:
        unary_map = {
            'sin': 0, 'cos': 0,  # MetaPeriodic
            'power': 1, 'identity': 1,  # MetaPower
            'rational': 1,  # Bias toward reciprocal via MetaPower
            'exp': 2,  # MetaExp
            'log': 3,  # MetaLog
        }
        n_unary = 4
    
    n_biased = 0
    
    for layer in model.layers:
        for node in layer.nodes:
            # Get operation selector
            if not hasattr(node, 'op_selector'):
                continue
                
            selector = node.op_selector
            
            # HardConcreteOperationSelector has single logits tensor
            # Layout: [type(2), unary(n_unary), binary(n_binary)]
            if hasattr(selector, 'logits') and hasattr(selector, '_type_end'):
                with torch.no_grad():
                    # Bias type toward unary if sin/cos/periodic predicted
                    periodic_prob = max(predictions.get('sin', 0), 
                                       predictions.get('cos', 0))
                    if periodic_prob >= threshold:
                        selector.logits.data[0] += periodic_prob * boost_factor  # unary type
                        n_biased += 1
                    
                    # Bias specific unary ops
                    for op_name, prob in predictions.items():
                        if prob >= threshold and op_name in unary_map:
                            idx = unary_map[op_name]
                            logit_idx = 2 + idx  # Skip 2 type logits
                            if logit_idx < len(selector.logits):
                                selector.logits.data[logit_idx] += prob * boost_factor
                                n_biased += 1
    
    if n_biased > 0 and verbose:
        print(f"Biased {n_biased} operation logits based on classifier predictions")
    return model


# =============================================================================
# MAIN TEST
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test curve classifier")
    parser.add_argument("--model", type=str, default=DEFAULT_CURVE_CLASSIFIER_PATH,
                        help=f"Path to trained model (default: {DEFAULT_CURVE_CLASSIFIER_PATH})")
    parser.add_argument("--formula", type=str, default="np.sin(x) + x**2",
                        help="Formula to test (uses numpy)")
    parser.add_argument("--x-min", type=float, default=-5)
    parser.add_argument("--x-max", type=float, default=5)
    parser.add_argument("--n-points", type=int, default=256)
    
    args = parser.parse_args()
    
    # Generate test data
    x = np.linspace(args.x_min, args.x_max, args.n_points)
    
    try:
        y = eval(args.formula, {"x": x, "np": np})
    except Exception as e:
        print(f"Error evaluating formula: {e}")
        return
    
    print(f"Testing formula: {args.formula}")
    print(f"  x range: [{args.x_min}, {args.x_max}]")
    print(f"  y range: [{y.min():.4f}, {y.max():.4f}]")
    
    # Predict
    predictions = predict_operators(x, y, args.model, threshold=0.3)
    print_predictions(predictions)
    
    # Show expected vs predicted
    print("\nNote: periodic/exponential are derived from sin/cos and exp/log")


if __name__ == "__main__":
    main()
