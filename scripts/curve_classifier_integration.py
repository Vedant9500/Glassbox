"""
Curve Classifier Integration with ONN

Uses the trained curve classifier to predict operators and warm-start ONN evolution.

Usage:
    # Test on synthetic data
    python scripts/test_curve_classifier.py --model models/curve_classifier.pt --formula "sin(x) + x**2"
    
    # Integrate with ONN (in your training script)
    from scripts.curve_classifier_integration import predict_operators, bias_onn_from_predictions
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from scripts.generate_curve_data import extract_all_features, OPERATOR_CLASSES
except ImportError:
    from generate_curve_data import extract_all_features, OPERATOR_CLASSES


# =============================================================================
# MODEL DEFINITION (must match training)
# =============================================================================

class CurveClassifierMLP(nn.Module):
    """Simple MLP classifier for curve features."""
    
    def __init__(self, n_features: int = 297, n_classes: int = 11, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(hidden),
            
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(hidden),
            
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden // 2, n_classes),
        )
    
    def forward(self, x):
        return torch.sigmoid(self.net(x))


# =============================================================================
# CLASSIFIER LOADING
# =============================================================================

_cached_classifier_by_device = {}
_cached_operator_classes = None
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


def load_classifier(
    model_path: str = "models/curve_classifier.pt",
    device: Optional[str] = None,
) -> nn.Module:
    """Load the trained curve classifier."""
    global _cached_classifier_by_device, _cached_operator_classes
    
    resolved_device = _resolve_device(device)
    # Create cache key using both device and model path
    cache_key = f"{str(resolved_device)}:{model_path}"
    if cache_key in _cached_classifier_by_device:
        return _cached_classifier_by_device[cache_key]
    
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        # Clean failure if model not found
        raise FileNotFoundError(f"Classifier model not found at {model_path}")
    
    try:
        checkpoint = torch.load(model_path_obj, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Error loading checkpoint from {model_path}: {e}")
        raise
    
    # Get operator classes
    _cached_operator_classes = checkpoint.get('operator_classes', list(OPERATOR_CLASSES.keys()))
    n_classes = len(_cached_operator_classes)
    
    # Determine input features from checkpoint weights
    state_dict = checkpoint['model_state_dict']
    input_weights = state_dict['net.0.weight']
    n_features = input_weights.shape[1]
    
    # Create model
    model = CurveClassifierMLP(n_features=n_features, n_classes=n_classes, hidden=256)
    model.load_state_dict(state_dict)
    model.to(resolved_device)
    model.eval()

    _cached_classifier_by_device[cache_key] = model
    print(f"Loaded curve classifier from {model_path}")
    print(f"  Val accuracy: {checkpoint.get('val_acc', 'N/A'):.4f}")
    print(f"  Device: {resolved_device}")
    
    return model


# =============================================================================
# PREDICTION
# =============================================================================

def predict_operators(
    x: np.ndarray,
    y: np.ndarray,
    model_path: str = "models/curve_classifier.pt",
    threshold: float = 0.5,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """
    Predict which operators are likely present in the data.
    
    Args:
        x: Input values (1D array)
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
        
    resolved_device = _resolve_device(device)
    
    # Extract features
    features = extract_all_features(y)
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(resolved_device)
    
    # Predict
    with torch.no_grad():
        probs = model(features_tensor).squeeze().detach().cpu().numpy()
    
    # Build result dict
    operator_classes = _cached_operator_classes or list(OPERATOR_CLASSES.keys())
    result = {}
    for i, name in enumerate(operator_classes):
        if probs[i] >= threshold:
            result[name] = float(probs[i])
    
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
    
    if simplified:
        unary_map = {
            'sin': 0, 'cos': 0, 'periodic': 0,  # MetaPeriodic
            'power': 1, 'polynomial': 1, 'identity': 1,  # MetaPower
            'rational': 1,  # Bias toward reciprocal via MetaPower
        }
        n_unary = 2
    else:
        unary_map = {
            'sin': 0, 'cos': 0, 'periodic': 0,  # MetaPeriodic
            'power': 1, 'polynomial': 1, 'identity': 1,  # MetaPower
            'rational': 1,  # Bias toward reciprocal via MetaPower
            'exp': 2, 'exponential': 2,  # MetaExp
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
                                       predictions.get('cos', 0),
                                       predictions.get('periodic', 0))
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
    parser.add_argument("--model", type=str, default="models/curve_classifier.pt",
                        help="Path to trained model")
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
    print("\nNote: 'periodic' = sin/cos, 'exponential' = exp/log")


if __name__ == "__main__":
    main()
