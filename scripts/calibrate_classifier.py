"""
Calibrate an existing curve classifier model using isotonic regression.

This script loads a trained model checkpoint, generates calibration data
(if not provided), fits per-class isotonic regression maps, and saves
the updated checkpoint with calibration metadata.

Usage:
    # Calibrate using existing dataset
    python scripts/calibrate_classifier.py --model models/curve_classifier_v3.1.pt --data data/curve_dataset_500k_v2.npz

    # Calibrate with custom split ratio
    python scripts/calibrate_classifier.py --model models/curve_classifier_v3.1.pt --data data/curve_dataset_500k_v2.npz --val-ratio 0.2

    # Calibrate and save to new path
    python scripts/calibrate_classifier.py --model models/curve_classifier_v3.1.pt --data data/curve_dataset_500k_v2.npz --output models/curve_classifier_v3.2_calibrated.pt
"""

import numpy as np
import torch
import argparse
from pathlib import Path
import sys

# Add parent to path for imports
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / 'scripts'))

from train_curve_classifier import (
    calibrate_isotonic_per_class,
    tune_thresholds,
    CurveClassifierMLP,
    CurveClassifierCNN,
    IndexedFeatureDataset,
    evaluate,
)
from generate_curve_data import OPERATOR_CLASSES

try:
    from curve_classifier_integration import (
        load_classifier,
        _cached_metadata_by_device,
        _make_cache_key,
        _resolve_device,
        predict_operators,
    )
    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False


def load_calibration_data(
    data_path: str,
    n_samples: int | None,
    val_ratio: float,
    seed: int,
):
    """Load validation data for calibration from .npz file."""
    data = np.load(data_path, allow_pickle=True)
    features = data["features"]
    labels = data["labels"]
    
    if n_samples is not None:
        features = features[:n_samples]
        labels = labels[:n_samples]
    
    # Use a held-out portion for calibration
    rng = np.random.RandomState(seed)
    indices = np.arange(len(features))
    rng.shuffle(indices)
    
    n_val = int(len(indices) * val_ratio)
    val_indices = indices[:n_val]
    
    val_features = features[val_indices]
    val_labels = labels[val_indices]
    
    return val_features, val_labels


def calibrate_existing_model(
    model_path: str,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    device: torch.device,
    output_path: str,
    n_bins: int = 20,
    retune_thresholds: bool = True,
):
    """
    Fit isotonic calibration maps on validation data and update checkpoint.
    
    Args:
        model_path: Path to existing .pt checkpoint
        val_features: Validation features (N, feature_dim)
        val_labels: Validation labels (N, n_classes)
        device: Torch device
        output_path: Where to save the calibrated checkpoint
        n_bins: Number of bins for discretized isotonic maps
        retune_thresholds: Whether to re-tune thresholds on calibrated probs
    """
    # Load existing checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    operator_classes = checkpoint.get('operator_classes', list(OPERATOR_CLASSES.keys()))
    n_classes = len(operator_classes)
    state_dict = checkpoint['model_state_dict']
    model_type = checkpoint.get('model_type')
    model_config = checkpoint.get('model_config', {})

    # Backward-compatible architecture detection for older checkpoints.
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
    
    # Reconstruct model
    if model_type == 'cnn':
        if 'n_features' in model_config:
            n_features = int(model_config['n_features'])
        else:
            classifier_in = state_dict['classifier.0.weight'].shape[1]
            n_features = int(max(1, classifier_in - (128 * 4)))
        curve_dim = int(model_config.get('curve_dim', min(128, n_features)))
        model = CurveClassifierCNN(
            n_classes=int(model_config.get('n_classes', n_classes)),
            n_features=n_features,
            curve_dim=curve_dim,
        )
    else:
        input_weights = state_dict['net.0.weight']
        n_features = int(model_config.get('n_features', input_weights.shape[1]))
        hidden_size = int(model_config.get('hidden', input_weights.shape[0]))
        model = CurveClassifierMLP(n_features=n_features, n_classes=n_classes, hidden=hidden_size)
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Standardize features if scaler exists
    scaler = checkpoint.get('feature_scaler')
    if scaler is not None:
        val_features = (val_features - scaler['mean']) / (scaler['std'] + 1e-8)
    
    # Create DataLoader for evaluation
    val_dataset = IndexedFeatureDataset(
        val_features, val_labels,
        indices=np.arange(len(val_features)),
        scaler=None,  # Already standardized
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1024, shuffle=False, num_workers=0
    )
    
    # Get logits on validation set
    print("\nComputing logits on validation set...")
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            all_logits.append(logits.cpu())
            all_labels.append(y_batch)
    
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    
    print(f"  Validation samples: {all_labels.shape[0]}")
    print(f"  Classes: {n_classes}")
    
    # Evaluate before calibration
    temperature = checkpoint.get('temperature')
    print(f"\n--- Before calibration ---")
    metrics_before = evaluate(
        model, val_loader, torch.nn.BCEWithLogitsLoss(), device,
        temperature=temperature, return_preds=True
    )
    print(f"  Val accuracy: {metrics_before['accuracy']:.4f}")
    print(f"  Val F1: {metrics_before['f1_mean']:.4f}")
    print(f"  Val Micro-F1: {metrics_before['micro_f1']:.4f}")
    
    # Fit isotonic calibration
    print(f"\nFitting per-class isotonic regression ({n_bins} bins)...")
    isotonic_maps = calibrate_isotonic_per_class(
        all_logits, all_labels,
        temperature=temperature,
        n_bins=n_bins,
    )
    
    if not isotonic_maps:
        print("  ERROR: Isotonic calibration failed. Check sklearn installation.")
        return
    
    # Report calibration maps
    print("\nCalibration maps fitted:")
    for i, name in enumerate(operator_classes):
        if i < len(isotonic_maps):
            cmap = isotonic_maps[i]
            values = cmap['values']
            # Show range of calibration
            print(f"  {name:15s}: calibrated range [{min(values):.3f}, {max(values):.3f}]")
    
    # Evaluate after calibration (apply to predictions)
    print(f"\n--- After calibration ---")
    raw_probs = torch.sigmoid(all_logits / temperature if temperature else all_logits).numpy()
    
    # Apply calibration manually for evaluation
    from train_curve_classifier import apply_isotonic_calibration
    calibrated_probs = apply_isotonic_calibration(raw_probs, isotonic_maps)
    
    # Compute calibrated metrics
    calibrated_preds = torch.from_numpy(calibrated_probs)
    current_thresholds = checkpoint.get('thresholds')
    if current_thresholds is not None:
        current_thresholds = torch.from_numpy(current_thresholds)
    
    binary_preds = (calibrated_preds > (current_thresholds if current_thresholds is not None else 0.5)).float()
    cal_acc = (binary_preds == all_labels).float().mean().item()
    
    tp = ((binary_preds == 1) & (all_labels == 1)).float().sum(dim=0)
    fp = ((binary_preds == 1) & (all_labels == 0)).float().sum(dim=0)
    fn = ((binary_preds == 0) & (all_labels == 1)).float().sum(dim=0)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    cal_f1 = f1.mean().item()
    
    print(f"  Val accuracy: {cal_acc:.4f}")
    print(f"  Val F1: {cal_f1:.4f}")
    
    # Optionally retune thresholds on calibrated probabilities
    if retune_thresholds:
        print("\nRe-tuning thresholds on calibrated probabilities...")
        new_thresholds = tune_thresholds(calibrated_preds, all_labels)
        checkpoint['thresholds'] = new_thresholds.numpy()
        
        # Re-evaluate with new thresholds
        binary_preds_new = (calibrated_preds > new_thresholds).float()
        tp = ((binary_preds_new == 1) & (all_labels == 1)).float().sum(dim=0)
        fp = ((binary_preds_new == 1) & (all_labels == 0)).float().sum(dim=0)
        fn = ((binary_preds_new == 0) & (all_labels == 1)).float().sum(dim=0)
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        print(f"  Val F1 (retuned): {f1.mean().item():.4f}")
        
        print("\nNew thresholds:")
        for i, name in enumerate(operator_classes):
            if i < len(new_thresholds):
                old_t = float(current_thresholds[i]) if current_thresholds is not None else 0.5
                new_t = float(new_thresholds[i])
                print(f"  {name:15s}: {old_t:.3f} → {new_t:.3f}")
    
    # Save updated checkpoint
    checkpoint['isotonic_calibration'] = isotonic_maps
    checkpoint['calibration_info'] = {
        'method': 'isotonic_regression',
        'n_bins': n_bins,
        'n_calibration_samples': len(all_labels),
        'val_acc_before': metrics_before['accuracy'],
        'val_f1_before': metrics_before['f1_mean'],
        'val_acc_after': cal_acc,
        'val_f1_after': cal_f1,
    }
    
    torch.save(checkpoint, output_path)
    print(f"\nCalibrated model saved to {output_path}")


def quick_calibration_test(
    model_path: str,
    device: torch.device,
    n_test: int = 10,
):
    """Quick sanity check: run predict_operators on a few synthetic examples."""
    if not INTEGRATION_AVAILABLE:
        print("\nSkipping quick test (integration module not available)")
        return
    
    print(f"\n--- Quick calibration test ({n_test} synthetic examples) ---")
    
    # Load the calibrated model
    try:
        model = load_classifier(model_path, device=str(device))
    except Exception as e:
        print(f"  Failed to load model: {e}")
        return
    
    cache_key = _make_cache_key(model_path, _resolve_device(str(device)))
    metadata = _cached_metadata_by_device.get(cache_key, {})
    
    has_calibration = bool(metadata.get('isotonic_calibration'))
    print(f"  Has isotonic calibration: {has_calibration}")
    
    if not has_calibration:
        print("  No calibration found — skipping test")
        return
    
    # Test on simple synthetic curves
    test_cases = [
        ("sin(x)", lambda x: np.sin(x), np.linspace(0, 2*np.pi, 128)),
        ("x^2", lambda x: x**2, np.linspace(0, 3, 128)),
        ("exp(x)", lambda x: np.exp(x), np.linspace(0, 2, 128)),
    ]
    
    for name, func, x_range in test_cases[:n_test]:
        y = func(x_range)
        try:
            result = predict_operators(x_range, y, model_path=model_path, threshold=0.1, device=str(device))
            top_ops = sorted(result.items(), key=lambda x: -x[1])[:3]
            print(f"  {name:15s}: {top_ops}")
        except Exception as e:
            print(f"  {name:15s}: ERROR - {e}")


def main():
    parser = argparse.ArgumentParser(description="Calibrate an existing curve classifier")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to existing .pt checkpoint")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to .npz dataset for calibration data")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: overwrite input model)")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Number of samples to use from dataset")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                        help="Fraction of data to use for calibration (default: 0.15)")
    parser.add_argument("--n-bins", type=int, default=20,
                        help="Number of bins for isotonic maps (default: 20)")
    parser.add_argument("--no-retune-thresholds", action="store_true",
                        help="Skip threshold retuning")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (auto, cpu, cuda)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for data split")
    parser.add_argument("--quick-test", action="store_true",
                        help="Run quick sanity test after calibration")
    
    args = parser.parse_args()
    
    # Resolve device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Output path
    output_path = args.output or args.model
    if output_path == args.model:
        print(f"WARNING: Will overwrite {args.model}")
        backup = args.model + ".backup"
        import shutil
        shutil.copy(args.model, backup)
        print(f"  Backup saved to {backup}")
    
    # Load calibration data
    print(f"\nLoading calibration data from {args.data}...")
    val_features, val_labels = load_calibration_data(
        args.data, args.n_samples,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"  Calibration samples: {len(val_labels)}")
    print(f"  Positive rate per class: {val_labels.mean(axis=0)}")
    
    # Run calibration
    calibrate_existing_model(
        args.model, val_features, val_labels,
        device, output_path,
        n_bins=args.n_bins,
        retune_thresholds=not args.no_retune_thresholds,
    )
    
    # Quick test
    if args.quick_test:
        quick_calibration_test(output_path, device)


if __name__ == "__main__":
    main()
