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
from torch.utils.data import DataLoader, TensorDataset
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Optional, List


# =============================================================================
# MODEL ARCHITECTURES
# =============================================================================

class CurveClassifierMLP(nn.Module):
    """Simple MLP classifier for curve features."""
    
    def __init__(self, n_features: int = 334, n_classes: int = 9, hidden: int = 256):
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
    
    def __init__(self, n_classes: int = 9, n_features: int = 334, curve_dim: int = 128):
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


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device, max_grad_norm: float = 1.0):
    """Train for one epoch with gradient clipping."""
    model.train()
    total_loss = 0
    n_batches = 0
    
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
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
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
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
        remaining = [i for i in indices if i not in val_indices and i not in train_indices]
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
    patience: int = 10,
    early_stop_metric: str = "f1",
    tune_thresholds_flag: bool = True,
    calibrate_flag: bool = False,
):
    """Full training loop with early stopping."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
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

    # Threshold tuning (optionally on calibrated probabilities)
    if tune_thresholds_flag:
        preds_for_tuning = val_metrics['preds']
        if temperature is not None:
            preds_for_tuning = torch.sigmoid(val_metrics['logits'] / temperature)
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
    parser.add_argument("--feature-dim", type=int, default=334,
                        help="Feature dimension for .dat files (default: 334)")
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
    parser.add_argument("--hidden", type=int, default=256,
                        help="Hidden layer size (MLP only)")
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

    expected_features = feature_dim or 334
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

    train_np = features[train_idx]
    val_np = features[val_idx]

    scaler = None
    if standardize:
        mean = train_np.mean(axis=0)
        std = train_np.std(axis=0) + 1e-8
        train_np = (train_np - mean) / std
        val_np = (val_np - mean) / std
        scaler = {'mean': mean, 'std': std}

    train_features = torch.tensor(train_np, dtype=torch.float32)
    train_labels = torch.tensor(labels[train_idx], dtype=torch.float32)
    val_features = torch.tensor(val_np, dtype=torch.float32)
    val_labels = torch.tensor(labels[val_idx], dtype=torch.float32)
    
    print(f"  Train: {len(train_features)}, Val: {len(val_features)}")
    
    # Data loaders with optimizations
    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)
    
    # Use pin_memory for GPU and num_workers for parallel data loading
    use_cuda = device.type == 'cuda'
    loader_kwargs = {
        'num_workers': 4 if use_cuda else 0,
        'pin_memory': use_cuda,
    }
    
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
    else:
        curve_dim = 128
        if feature_schema is not None and "raw" in feature_schema:
            raw_slice = feature_schema["raw"]
            if isinstance(raw_slice, (list, tuple)) and len(raw_slice) == 2:
                curve_dim = int(raw_slice[1] - raw_slice[0])
        model = CurveClassifierCNN(n_classes=n_classes, n_features=n_features, curve_dim=curve_dim)
    
    model = model.to(device)
    print(f"\nModel: {args.model.upper()}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
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
        patience=args.patience,
        early_stop_metric=args.early_stop,
        tune_thresholds_flag=tune_thresholds_flag,
        calibrate_flag=args.calibrate,
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
