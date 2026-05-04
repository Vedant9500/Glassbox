"""Train script scaffold for Phase 1 universal proposer MVP.

This script intentionally starts with synthetic data so iteration is fast and
independent of a finalized dataset schema.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List, Tuple, Optional

# Add the repository root to sys.path so we can import glassbox
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from glassbox.universal_proposer import (
    UniversalProposer,
    UniversalProposerConfig,
    DEFAULT_OPERATOR_VOCAB,
    DEFAULT_SKELETON_VOCAB,
)

try:
    from glassbox.curve_classifier.generate_curve_data import extract_all_features, evaluate_formula
except Exception:
    from glassbox.curve_classifier.generate_curve_data import extract_all_features, evaluate_formula


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
        
        # Apply SymLog compression to make distributions Gaussian-friendly
        batch = np.sign(batch) * np.log1p(np.abs(batch))
        
        sum_x += batch.sum(axis=0)
        sum_x2 += np.square(batch).sum(axis=0)
        total_count += batch.shape[0]

    mean = sum_x / max(total_count, 1)
    var = (sum_x2 / max(total_count, 1)) - np.square(mean)
    var = np.maximum(var, 0.0)
    std = np.sqrt(var) + 1e-8
    return mean.astype(np.float32), std.astype(np.float32)


class SyntheticCurveDataset(Dataset):
    def __init__(self, n_samples: int = 2000, n_points: int = 128, seed: int = 0):
        self.n_samples = int(n_samples)
        self.n_points = int(n_points)
        self.rng = np.random.RandomState(seed)
        self.operator_vocab = list(DEFAULT_OPERATOR_VOCAB)
        self.skeleton_vocab = list(DEFAULT_SKELETON_VOCAB)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        x = np.linspace(-2.0, 2.0, self.n_points, dtype=np.float32)
        kind = self.rng.randint(0, len(self.skeleton_vocab))

        if kind == 0:
            y = x
            ops = ["identity"]
        elif kind == 1:
            y = x ** 2
            ops = ["power"]
        elif kind == 2:
            y = np.sin(x)
            ops = ["sin", "periodic"]
        elif kind == 3:
            y = np.cos(x)
            ops = ["cos", "periodic"]
        elif kind == 4:
            y = np.exp(np.clip(x, -3.0, 3.0))
            ops = ["exp"]
        elif kind == 5:
            y = np.log(np.abs(x) + 1e-6)
            ops = ["log"]
        elif kind == 6:
            y = 1.0 / (x + 1e-3)
            ops = ["rational"]
        elif kind == 7:
            y = x * np.sin(x)
            ops = ["identity", "sin", "periodic"]
        else:
            y = x + np.sin(x)
            ops = ["identity", "sin", "periodic"]

        # Extract real analytical features (Invariants, FFT, Derivatives)
        y = y + 0.01 * self.rng.randn(*y.shape).astype(np.float32)
        features = extract_all_features(y)

        # Apply SymLog (No scaler for synthetic MVP)
        features = np.sign(features) * np.log1p(np.abs(features))

        op_target = np.zeros(len(self.operator_vocab), dtype=np.float32)
        for op in ops:
            if op in self.operator_vocab:
                op_target[self.operator_vocab.index(op)] = 1.0

        return (
            torch.from_numpy(features.astype(np.float32)),
            torch.from_numpy(op_target),
            torch.tensor(kind, dtype=torch.long),
        )


class FormulaReplayDataset(Dataset):
    """Dataset-backed proposer training from generated formula corpora (.npz)."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        scaler: Optional[dict] = None,
    ):
        self.features = features
        self.labels = labels
        self.indices = indices
        self.scaler = scaler
        self.operator_vocab = list(DEFAULT_OPERATOR_VOCAB)

    def __len__(self) -> int:
        return len(self.indices)

    def _labels_to_operator_target(self, row: np.ndarray) -> np.ndarray:
        op = np.zeros(len(self.operator_vocab), dtype=np.float32)
        mapping = {
            "identity": 0, "sin": 1, "cos": 2, "power": 3, "exp": 4, 
            "log": 5, "rational": 8
        }
        for name, idx in mapping.items():
            if idx < row.shape[0]:
                op[self.operator_vocab.index(name)] = row[idx]
        if "periodic" in self.operator_vocab:
            op[self.operator_vocab.index("periodic")] = max(row[1], row[2])
        return op

    def __getitem__(self, idx: int):
        sample_idx = self.indices[idx]
        feat = self.features[sample_idx]
        
        # Apply SymLog + Scaling
        feat = np.sign(feat) * np.log1p(np.abs(feat))
        if self.scaler is not None:
            feat = (feat - self.scaler['mean']) / (self.scaler['std'] + 1e-8)
            
        op_target = self._labels_to_operator_target(self.labels[sample_idx])
        return torch.from_numpy(feat.astype(np.float32)), torch.from_numpy(op_target), torch.tensor(-1, dtype=torch.long)


def _train_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    total = 0

    for features, op_target, skeleton_target in loader:
        features = features.to(device, non_blocking=True)
        op_target = op_target.to(device, non_blocking=True)

        out = model(features)
        loss = F.binary_cross_entropy_with_logits(out["operator_logits"], op_target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * features.shape[0]
        total += features.shape[0]

    return total_loss / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description="Train universal proposer (Phase 1 scaffold)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--n-points", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--out", type=str, default="models/universal_proposer_robust.pt")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data", type=str, default="", help="Optional dataset .npz path from generate_curve_data")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap when --data is used")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    config = UniversalProposerConfig(hidden_dim=args.hidden)
    model = UniversalProposer(config).to(device)

    if args.data:
        blob = np.load(args.data, allow_pickle=True)
        features = np.asarray(blob["features"], dtype=np.float32)
        labels = np.asarray(blob["labels"], dtype=np.float32)
        
        if args.max_samples > 0:
            features = features[:args.max_samples]
            labels = labels[:args.max_samples]

        indices = np.arange(len(features))
        np.random.shuffle(indices)
        
        print("Computing feature statistics (SymLog + Standardize)...")
        mean, std = compute_feature_stats(features, indices)
        scaler = {'mean': mean, 'std': std}

        ds = FormulaReplayDataset(features, labels, indices, scaler=scaler)
        print(f"dataset=FormulaReplayDataset samples={len(ds)} path={args.data}")
    else:
        scaler = None
        ds = SyntheticCurveDataset(n_samples=args.n_samples, n_points=args.n_points)
        print(f"dataset=SyntheticCurveDataset samples={len(ds)}")
        
    import os
    use_cuda = device.type == 'cuda'
    num_cpus = os.cpu_count() or 4
    n_workers = min(8, max(2, num_cpus - 2)) if use_cuda else 0
    
    loader_kwargs = {'num_workers': n_workers, 'pin_memory': use_cuda}
    if n_workers > 0:
        loader_kwargs['prefetch_factor'] = 2
        loader_kwargs['persistent_workers'] = True

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Training GLU Proposer on {device}...")
    for epoch in range(1, args.epochs + 1):
        loss = _train_epoch(model, loader, opt, device)
        print(f"epoch={epoch:03d} loss={loss:.5f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "hidden_dim": config.hidden_dim,
                "n_features": config.n_features,
                "operator_vocab": model.operator_vocab,
                "skeleton_vocab": model.skeleton_vocab,
            },
            "feature_scaler": scaler,
        },
        out_path,
    )
    print(f"saved model -> {out_path}")


if __name__ == "__main__":
    main()
