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


# =============================================================================
# MODEL ARCHITECTURES
# =============================================================================

class CurveClassifierMLP(nn.Module):
    """Simple MLP classifier for curve features."""
    
    def __init__(self, n_features: int = 297, n_classes: int = 11, hidden: int = 256):
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
        return torch.sigmoid(self.net(x))


class CurveClassifierCNN(nn.Module):
    """1D CNN that operates on the raw curve portion of features."""
    
    def __init__(self, n_classes: int = 11, n_features: int = 334, curve_dim: int = 128):
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
        return torch.sigmoid(self.classifier(combined))


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
        pred = model(x_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on dataset."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            
            total_loss += loss.item()
            all_preds.append(pred.cpu())
            all_labels.append(y_batch.cpu())
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    # Metrics
    avg_loss = total_loss / len(dataloader)
    
    # Per-class accuracy (threshold = 0.5)
    binary_preds = (all_preds > 0.5).float()
    per_class_acc = ((binary_preds == all_labels).float().mean(dim=0))
    overall_acc = (binary_preds == all_labels).float().mean()
    
    # F1 score per class
    tp = ((binary_preds == 1) & (all_labels == 1)).float().sum(dim=0)
    fp = ((binary_preds == 1) & (all_labels == 0)).float().sum(dim=0)
    fn = ((binary_preds == 0) & (all_labels == 1)).float().sum(dim=0)
    
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    
    return {
        'loss': avg_loss,
        'accuracy': overall_acc.item(),
        'per_class_acc': per_class_acc.numpy(),
        'f1_mean': f1.mean().item(),
        'f1_per_class': f1.numpy(),
    }


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
):
    """Full training loop with early stopping."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = nn.BCELoss()
    
    best_val_loss = float('inf')
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
              f"Val F1: {val_metrics['f1_mean']:.4f}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['accuracy'],
                'operator_classes': operator_classes,
            }, save_path)
            print(f"  -> Saved best model (val_loss: {val_metrics['loss']:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break
    
    print(f"\nBest model at epoch {best_epoch} with val_loss: {best_val_loss:.4f}")
    
    # Reload best model for final evaluation
    checkpoint = torch.load(save_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    val_metrics = evaluate(model, val_loader, criterion, device)
    
    # Final per-class report using best model
    print("\nPer-class F1 scores (best model):")
    for i, name in enumerate(operator_classes):
        print(f"  {name:15s}: {val_metrics['f1_per_class'][i]:.4f}")
    
    return model


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train curve classifier")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to training data (.npz file)")
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
    
    args = parser.parse_args()
    
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
    data = np.load(args.data, allow_pickle=True)
    features = data['features']
    labels = data['labels']
    operator_classes = data['operator_classes'].tolist()
    
    print(f"  Features: {features.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Classes: {operator_classes}")
    
    # Train/val split
    n_val = int(len(features) * args.val_split)
    indices = np.random.permutation(len(features))
    
    train_features = torch.tensor(features[indices[n_val:]], dtype=torch.float32)
    train_labels = torch.tensor(labels[indices[n_val:]], dtype=torch.float32)
    val_features = torch.tensor(features[indices[:n_val]], dtype=torch.float32)
    val_labels = torch.tensor(labels[indices[:n_val]], dtype=torch.float32)
    
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
        model = CurveClassifierCNN(n_classes=n_classes, n_features=n_features)
    
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
    )
    
    print(f"\nModel saved to {output_path}")


if __name__ == "__main__":
    main()
