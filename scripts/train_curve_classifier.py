"""
Curve Classifier Training Script

Trains a neural network to predict which mathematical operators are present
in a curve based on its features.

Usage:
    python scripts/train_curve_classifier.py --data data/curve_dataset_10k.npz --epochs 50
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Optional, List


# =============================================================================
# MODEL ARCHITECTURES
# =============================================================================

class CurveClassifierMLP(nn.Module):
    """Simple MLP classifier for curve features."""
    
    def __init__(self, n_features: int = 366, n_classes: int = 9, hidden: int = 256):
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
        
        # Apply weight initialization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.net(x)


class CurveClassifierCNN(nn.Module):
    """1D CNN that operates on the raw curve portion of features."""
    
    def __init__(self, n_classes: int = 9, n_features: int = 366, curve_dim: int = 128):
        super().__init__()
        
        # Dynamically determine curve dimension (use min of curve_dim and n_features)
        self.curve_dim = min(curve_dim, n_features)
        
        # CNN for raw curve (first curve_dim features)
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
        
        # MLP for other features (FFT, derivatives, stats)
        other_dim = max(1, n_features - self.curve_dim)
        self.other_mlp = nn.Sequential(
            nn.Linear(other_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )
        
        # Apply weight initialization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Split features using dynamic curve dimension
        raw_curve = x[:, :self.curve_dim]
        other_features = x[:, self.curve_dim:]
        
        # CNN path
        raw_curve = raw_curve.unsqueeze(1)  # (batch, 1, curve_dim)
        conv_out = self.conv(raw_curve)     # (batch, 128, 4)
        conv_out = conv_out.flatten(1)      # (batch, 512)
        
        # MLP path
        other_out = self.other_mlp(other_features)  # (batch, 128)
        
        # Combine
        combined = torch.cat([conv_out, other_out], dim=1)
        return self.classifier(combined)


class IndexedFeatureDataset(Dataset):
    """Dataset view over feature/label arrays using explicit indices."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        scaler: Optional[dict] = None,
    ):
        self.features = features
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)
        self.scaler = scaler

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        sample_idx = int(self.indices[idx])
        x = np.asarray(self.features[sample_idx], dtype=np.float32)
        if self.scaler is not None:
            x = (x - self.scaler['mean']) / (self.scaler['std'] + 1e-8)
        y = np.asarray(self.labels[sample_idx], dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


def compute_feature_stats(
    features: np.ndarray,
    indices: np.ndarray,
    chunk_size: int = 65536,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute feature mean/std on selected rows without full subset materialization."""
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) == 0:
        raise ValueError("Cannot compute feature stats for empty indices")

    n_features = int(features.shape[1])
    total_count = 0
    sum_x = np.zeros(n_features, dtype=np.float64)
    sum_x2 = np.zeros(n_features, dtype=np.float64)

    for start in range(0, len(indices), chunk_size):
        batch_idx = indices[start:start + chunk_size]
        batch = np.asarray(features[batch_idx], dtype=np.float64)
        sum_x += batch.sum(axis=0)
        sum_x2 += np.square(batch).sum(axis=0)
        total_count += batch.shape[0]

    mean = sum_x / max(total_count, 1)
    var = (sum_x2 / max(total_count, 1)) - np.square(mean)
    var = np.maximum(var, 0.0)
    std = np.sqrt(var) + 1e-8
    return mean.astype(np.float32), std.astype(np.float32)


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device, max_grad_norm: float = 1.0):
    """Train for one epoch with gradient clipping."""
    model.train()
    total_loss = 0
    n_batches = 0
    
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def evaluate(
    model,
    dataloader,
    criterion,
    device,
    thresholds: torch.Tensor | None = None,
    return_preds: bool = False,
    return_logits: bool = False,
    temperature: float | None = None,
):
    """Evaluate model on dataset."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_logits = []
    
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            
            total_loss += loss.item()
            if temperature is None:
                preds = torch.sigmoid(logits)
            else:
                preds = torch.sigmoid(logits / temperature)

            all_preds.append(preds.cpu())
            all_labels.append(y_batch.cpu())
            if return_logits:
                all_logits.append(logits.cpu())
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    # Metrics
    avg_loss = total_loss / len(dataloader)
    
    # Per-class accuracy (threshold = 0.5 or tuned thresholds)
    if thresholds is None:
        binary_preds = (all_preds > 0.5).float()
    else:
        binary_preds = (all_preds > thresholds).float()
    per_class_acc = ((binary_preds == all_labels).float().mean(dim=0))
    overall_acc = (binary_preds == all_labels).float().mean()
    
    # F1 score per class
    tp = ((binary_preds == 1) & (all_labels == 1)).float().sum(dim=0)
    fp = ((binary_preds == 1) & (all_labels == 0)).float().sum(dim=0)
    fn = ((binary_preds == 0) & (all_labels == 1)).float().sum(dim=0)
    
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    # Micro-F1
    tp_sum = tp.sum()
    fp_sum = fp.sum()
    fn_sum = fn.sum()
    micro_f1 = (2 * tp_sum) / (2 * tp_sum + fp_sum + fn_sum + 1e-10)
    
    metrics = {
        'loss': avg_loss,
        'accuracy': overall_acc.item(),
        'per_class_acc': per_class_acc.numpy(),
        'f1_mean': f1.mean().item(),
        'micro_f1': micro_f1.item(),
        'f1_per_class': f1.numpy(),
    }

    if return_preds:
        metrics['preds'] = all_preds
        metrics['labels'] = all_labels
    if return_logits:
        metrics['logits'] = torch.cat(all_logits)


    return metrics


def calibrate_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50) -> float:
    """Single-temperature scaling for multi-label logits."""
    device = logits.device
    log_t = torch.zeros(1, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)
    criterion = nn.BCEWithLogitsLoss()

    def _closure():
        optimizer.zero_grad()
        t = torch.exp(log_t)
        loss = criterion(logits / t, labels)
        loss.backward()
        return loss

    optimizer.step(_closure)
    return float(torch.exp(log_t).detach().cpu().item())


def calibrate_isotonic_per_class(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float | None = None,
    n_bins: int = 20,
) -> list[dict]:
    """
    Fit per-class isotonic regression calibration maps on validation data.

    For each class c:
      1. Compute raw probabilities p = sigmoid(logits_c / T) if T is given,
         else p = sigmoid(logits_c).
      2. Fit an isotonic regression (monotonically increasing step function)
         mapping raw p → calibrated P(y=1|p).
      3. Return a list of dicts, one per class, each containing:
         - 'boundaries': sorted array of bin edges (length n_bins+1)
         - 'values': calibrated probability for each bin (length n_bins)

    This is more expressive than global temperature scaling because each
    operator class gets its own calibration curve, handling the common
    multi-label problem where rare classes are systematically under-confident
    and common classes are over-confident.

    Returns:
        List of calibration dicts, or empty list if sklearn is unavailable.
    """
    try:
        from sklearn.isotonic import IsotonicRegression
    except ImportError:
        print("  Warning: sklearn not available, skipping isotonic calibration.")
        return []

    n_classes = labels.shape[1]
    calibration_maps = []

    for c in range(n_classes):
        if temperature is not None:
            raw_probs = torch.sigmoid(logits[:, c] / temperature).numpy()
        else:
            raw_probs = torch.sigmoid(logits[:, c]).numpy()
        true_labels = labels[:, c].numpy()

        # Skip calibration if class has too few positive samples
        n_pos = int(true_labels.sum())
        n_neg = len(true_labels) - n_pos
        if n_pos < 10 or n_neg < 10:
            # Not enough data: use near-identity mapping at bin centers.
            # Using centers avoids edge artifacts from step-bin lookup.
            bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            calibration_maps.append({
                'boundaries': bin_edges.tolist(),
                'values': bin_centers.tolist(),
            })
            continue

        # Fit isotonic regression
        ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
        ir.fit(raw_probs, true_labels)

        # Discretize into bins for compact storage in checkpoint
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        calibrated_values = ir.predict(bin_centers)

        calibration_maps.append({
            'boundaries': bin_edges.tolist(),
            'values': calibrated_values.tolist(),
        })

    return calibration_maps


def apply_isotonic_calibration(
    raw_probs: np.ndarray,
    calibration_maps: list[dict],
) -> np.ndarray:
    """
    Apply per-class isotonic calibration to raw probability outputs.

    Args:
        raw_probs: Array of shape (n_classes,) or (batch, n_classes) with raw sigmoid outputs.
        calibration_maps: List of dicts from calibrate_isotonic_per_class().

    Returns:
        Calibrated probabilities, same shape as input.
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
        # np.digitize returns index i such that boundaries[i-1] <= x < boundaries[i]
        indices = np.digitize(raw_probs[:, c], boundaries, right=False) - 1
        indices = np.clip(indices, 0, len(values) - 1)
        calibrated[:, c] = values[indices]

    if single:
        return calibrated[0]
    return calibrated


def tune_thresholds(all_preds: torch.Tensor, all_labels: torch.Tensor, steps: int = 19) -> torch.Tensor:
    """Tune per-class thresholds to maximize F1 on validation data."""
    thresholds = torch.full((all_labels.shape[1],), 0.5, dtype=torch.float32)
    candidates = torch.linspace(0.05, 0.95, steps)

    for c in range(all_labels.shape[1]):
        best_f1 = -1.0
        best_t = 0.5
        for t in candidates:
            preds = (all_preds[:, c] > t).float()
            labels = all_labels[:, c]
            tp = ((preds == 1) & (labels == 1)).float().sum()
            fp = ((preds == 1) & (labels == 0)).float().sum()
            fn = ((preds == 0) & (labels == 1)).float().sum()
            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        thresholds[c] = best_t

    return thresholds


def multilabel_stratified_split(labels: np.ndarray, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Approximate multi-label stratified split without external dependencies."""
    rng = np.random.RandomState(seed)
    n_samples = labels.shape[0]
    n_val = int(n_samples * val_ratio)

    # Fast path for large datasets: random split approximates stratification well
    if n_samples >= 200_000:
        indices = np.arange(n_samples)
        rng.shuffle(indices)
        return indices[n_val:], indices[:n_val]

    # Desired positives per class in validation set
    class_pos = labels.sum(axis=0)
    desired_val = np.round(class_pos * val_ratio).astype(int)

    indices = np.arange(n_samples)
    rng.shuffle(indices)

    val_indices = []
    train_indices = []
    remaining_val = n_val
    current_val_counts = np.zeros_like(desired_val, dtype=np.float32)

    for idx in indices:
        if remaining_val <= 0:
            train_indices.append(idx)
            continue

        sample_labels = labels[idx].astype(np.float32)
        needs = np.maximum(desired_val - current_val_counts, 0)
        score = (sample_labels * needs).sum()

        if score > 0:
            val_indices.append(idx)
            current_val_counts += sample_labels
            remaining_val -= 1
        else:
            train_indices.append(idx)

    # If we didn't fill validation set, top up randomly
    if remaining_val > 0:
        assigned = set(val_indices)
        assigned.update(train_indices)
        remaining = [i for i in indices if i not in assigned]
        rng.shuffle(remaining)
        val_indices.extend(remaining[:remaining_val])
        train_indices.extend(remaining[remaining_val:])

    return np.array(train_indices), np.array(val_indices)


def train_model(
    model,
    train_loader,
    val_loader,
    epochs: int,
    lr: float,
    device,
    save_path: Path,
    operator_classes: list,
    model_type: str,
    model_config: dict,
    patience: int = 10,
    early_stop_metric: str = "f1",
    tune_thresholds_flag: bool = True,
    calibrate_flag: bool = False,
    class_weights: Optional[torch.Tensor] = None,
):
    """Full training loop with early stopping."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Use class weights for imbalanced labels if provided
    if class_weights is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))
        print(f"Using class weights: {class_weights.numpy()}")
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    best_val_loss = float('inf')
    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Scheduler step
        scheduler.step(val_metrics['loss'])
        
        # Logging
        print(f"Epoch {epoch+1:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['f1_mean']:.4f} | "
            f"Val Micro-F1: {val_metrics['micro_f1']:.4f}")
        
        # Save best model
        if early_stop_metric == "f1":
            is_best = val_metrics['f1_mean'] > best_val_f1
        else:
            is_best = val_metrics['loss'] < best_val_loss

        if is_best:
            best_val_loss = val_metrics['loss']
            best_val_f1 = val_metrics['f1_mean']
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['accuracy'],
                'val_f1': val_metrics['f1_mean'],
                'operator_classes': operator_classes,
                'model_type': model_type,
                'model_config': model_config,
            }, save_path)
            print(f"  -> Saved best model (val_loss: {val_metrics['loss']:.4f}, val_f1: {val_metrics['f1_mean']:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break
    
    print(f"\nBest model at epoch {best_epoch} with val_loss: {best_val_loss:.4f}, val_f1: {best_val_f1:.4f}")
    
    # Reload best model for final evaluation
    checkpoint = torch.load(save_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    val_metrics = evaluate(model, val_loader, criterion, device, return_preds=True, return_logits=True)
    
    # Final per-class report using best model
    print("\nPer-class F1 scores (best model):")
    for i, name in enumerate(operator_classes):
        print(f"  {name:15s}: {val_metrics['f1_per_class'][i]:.4f}")

    # Tune thresholds and store alongside model checkpoint
    checkpoint = torch.load(save_path, weights_only=False)

    # Optional calibration
    temperature = None
    if calibrate_flag:
        temperature = calibrate_temperature(val_metrics['logits'], val_metrics['labels'])
        checkpoint['temperature'] = temperature
        print(f"\nCalibrated temperature saved to checkpoint: {temperature:.4f}")

    # Isotonic per-class calibration (more expressive than temperature scaling)
    isotonic_maps = calibrate_isotonic_per_class(
        val_metrics['logits'],
        val_metrics['labels'],
        temperature=temperature,
    )
    if isotonic_maps:
        checkpoint['isotonic_calibration'] = isotonic_maps
        print(f"Per-class isotonic calibration maps saved ({len(isotonic_maps)} classes)")

    # Threshold tuning (optionally on calibrated probabilities)
    if tune_thresholds_flag:
        preds_for_tuning = val_metrics['preds']
        if temperature is not None:
            preds_for_tuning = torch.sigmoid(val_metrics['logits'] / temperature)
        if isotonic_maps:
            calibrated_np = apply_isotonic_calibration(
                preds_for_tuning.detach().cpu().numpy(),
                isotonic_maps,
            )
            preds_for_tuning = torch.from_numpy(calibrated_np).to(preds_for_tuning.dtype)
        thresholds = tune_thresholds(preds_for_tuning, val_metrics['labels'])
        checkpoint['thresholds'] = thresholds.numpy()
        print("\nTuned per-class thresholds saved to checkpoint")

    torch.save(checkpoint, save_path)
    
    return model


def load_training_data(
    data_args: List[str],
    n_samples: Optional[int],
    feature_dim: int,
    n_classes: int,
    load_into_ram: bool,
):
    """Load training data from .npz or streamed .dat files."""
    # Case 1: single .npz file
    if len(data_args) == 1 and data_args[0].endswith(".npz"):
        data = np.load(data_args[0], allow_pickle=True)
        features = data["features"]
        labels = data["labels"]
        operator_classes = data["operator_classes"].tolist()
        detected_feature_dim = int(data["feature_dim"]) if "feature_dim" in data else features.shape[1]
        feature_schema = data["feature_schema"].item() if "feature_schema" in data else None
        return features, labels, operator_classes, detected_feature_dim, feature_schema

    # Case 2: base path or explicit feature/label files
    features_path = None
    labels_path = None

    if len(data_args) == 1:
        base = Path(data_args[0])
        features_path = base.with_suffix(".features.dat")
        labels_path = base.with_suffix(".labels.dat")
    else:
        for arg in data_args:
            if arg.endswith(".features.dat"):
                features_path = Path(arg)
            elif arg.endswith(".labels.dat"):
                labels_path = Path(arg)

    if features_path is None or labels_path is None or not features_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            "Expected either a .npz file or .features.dat and .labels.dat files."
        )

    if n_samples is None:
        file_size = features_path.stat().st_size
        n_samples = file_size // (feature_dim * 4)
        print(f"Inferred n_samples={n_samples} from {features_path.name}")

    features = np.memmap(features_path, dtype=np.float32, mode="r", shape=(n_samples, feature_dim))
    labels = np.memmap(labels_path, dtype=np.float32, mode="r", shape=(n_samples, n_classes))

    if load_into_ram:
        print("Loading features into RAM...")
        features = np.array(features)
        print(f"  Features loaded: {features.nbytes / 1e9:.2f} GB")
        print("Loading labels into RAM...")
        labels = np.array(labels)

    operator_classes = [
        "identity", "sin", "cos", "power", "exp",
        "log", "addition", "multiplication", "rational",
    ][:n_classes]
    return features, labels, operator_classes, feature_dim, None


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train curve classifier")
    parser.add_argument("--data", type=str, nargs="+", required=True,
                        help="Path to training data (.npz file) or base path / .features.dat + .labels.dat")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Number of samples (required for .dat if file size cannot be inferred)")
    parser.add_argument("--feature-dim", type=int, default=366,
                        help="Feature dimension for .dat files (default: 366)")
    parser.add_argument("--n-classes", type=int, default=9,
                        help="Number of classes for .dat files (default: 9)")
    parser.add_argument("--load-into-ram", action="store_true",
                        help="Load memmap data into RAM for faster training")
    parser.add_argument("--model", type=str, default="mlp",
                        choices=["mlp", "cnn"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--hidden", type=int, default=512,
                        help="Hidden layer size (MLP only, default: 512)")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Validation split ratio")
    parser.add_argument("--output", type=str, default="models/curve_classifier.pt",
                        help="Output model path")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (auto, cpu, cuda)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--early-stop", type=str, default="f1",
                        choices=["loss", "f1"], help="Early stopping metric")
    parser.add_argument("--standardize", action="store_true",
                        help="Standardize features using training set statistics (default: on)")
    parser.add_argument("--no-standardize", action="store_true",
                        help="Disable feature standardization")
    parser.add_argument("--tune-thresholds", action="store_true",
                        help="Tune per-class thresholds on validation set (default: on)")
    parser.add_argument("--no-tune-thresholds", action="store_true",
                        help="Disable per-class threshold tuning")
    parser.add_argument("--stratified-split", action="store_true",
                        help="Use approximate multi-label stratified train/val split (default: on)")
    parser.add_argument("--no-stratified-split", action="store_true",
                        help="Disable stratified split")
    parser.add_argument("--calibrate", action="store_true",
                        help="Calibrate probabilities with temperature scaling")
    parser.add_argument("--class-weights", action="store_true",
                        help="Use inverse frequency class weights for imbalanced labels")
    
    args = parser.parse_args()

    standardize = not args.no_standardize
    tune_thresholds_flag = not args.no_tune_thresholds
    stratified_split = not args.no_stratified_split
    
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading data from {args.data}...")
    features, labels, operator_classes, feature_dim, feature_schema = load_training_data(
        args.data,
        args.n_samples,
        args.feature_dim,
        args.n_classes,
        args.load_into_ram,
    )
    
    print(f"  Features: {features.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Classes: {operator_classes}")

    expected_features = feature_dim or 366
    if features.shape[1] != expected_features:
        print(f"Warning: expected {expected_features} features, got {features.shape[1]}. ")
    if feature_schema is not None:
        print(f"  Feature schema: {feature_schema}")
    
    # Train/val split
    if stratified_split:
        train_idx, val_idx = multilabel_stratified_split(labels, args.val_split, args.seed)
    else:
        n_val = int(len(features) * args.val_split)
        indices = np.random.permutation(len(features))
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

    scaler = None
    if standardize:
        mean, std = compute_feature_stats(features, train_idx)
        scaler = {'mean': mean, 'std': std}

    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")
    
    # Data loaders with optimizations and lazy memmap-backed access
    train_dataset = IndexedFeatureDataset(features, labels, train_idx, scaler=scaler)
    val_dataset = IndexedFeatureDataset(features, labels, val_idx, scaler=scaler)
    
    # Use pin_memory for GPU and num_workers for parallel data loading
    # optimized to use more available CPU cores and persistent workers
    import os
    use_cuda = device.type == 'cuda'
    num_cpus = os.cpu_count() or 4
    n_workers = min(12, max(2, num_cpus - 2)) if use_cuda else 0
    
    loader_kwargs = {
        'num_workers': n_workers,
        'pin_memory': use_cuda,
    }
    if n_workers > 0:
        loader_kwargs['prefetch_factor'] = 4
        loader_kwargs['persistent_workers'] = True
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        **loader_kwargs
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        **loader_kwargs
    )
    
    # Model
    n_features = features.shape[1]
    n_classes = labels.shape[1]
    
    if args.model == "mlp":
        model = CurveClassifierMLP(n_features, n_classes, args.hidden)
        model_config = {
            'n_features': int(n_features),
            'n_classes': int(n_classes),
            'hidden': int(args.hidden),
        }
    else:
        curve_dim = 128
        if feature_schema is not None and "raw" in feature_schema:
            raw_slice = feature_schema["raw"]
            if isinstance(raw_slice, (list, tuple)) and len(raw_slice) == 2:
                curve_dim = int(raw_slice[1] - raw_slice[0])
        model = CurveClassifierCNN(n_classes=n_classes, n_features=n_features, curve_dim=curve_dim)
        model_config = {
            'n_classes': int(n_classes),
            'n_features': int(n_features),
            'curve_dim': int(curve_dim),
        }
    
    model = model.to(device)
    print(f"\nModel: {args.model.upper()}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Compute class weights if requested
    class_weights = None
    if args.class_weights:
        # Inverse frequency weighting: weight = total / (n_classes * class_count)
        train_labels_np = np.asarray(labels[train_idx], dtype=np.float32)
        pos_counts = train_labels_np.sum(axis=0)
        neg_counts = len(train_idx) - pos_counts
        # pos_weight is applied to positive samples: weight = neg/pos
        pos_weights = neg_counts / (pos_counts + 1e-6)
        # Normalize to reasonable range
        pos_weights = np.clip(pos_weights, 0.5, 5.0)
        class_weights = torch.tensor(pos_weights, dtype=torch.float32)
        print(f"Class label counts: {pos_counts.astype(int)}")
        print(f"Computed pos_weights: {pos_weights.round(2)}")
    
    # Train
    print(f"\nTraining for {args.epochs} epochs...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_path=output_path,
        operator_classes=operator_classes,
        model_type=args.model,
        model_config=model_config,
        patience=args.patience,
        early_stop_metric=args.early_stop,
        tune_thresholds_flag=tune_thresholds_flag,
        calibrate_flag=args.calibrate,
        class_weights=class_weights,
    )

    # Persist scaler and schema metadata
    checkpoint = torch.load(output_path, weights_only=False)
    if scaler is not None:
        checkpoint['feature_scaler'] = scaler
    checkpoint['feature_schema'] = feature_schema
    checkpoint['feature_dim'] = feature_dim or features.shape[1]
    torch.save(checkpoint, output_path)
    
    print(f"\nModel saved to {output_path}")


if __name__ == "__main__":
    main()
