"""Train script scaffold for Phase 1 universal proposer MVP.

This script intentionally starts with synthetic data so iteration is fast and
independent of a finalized dataset schema.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from glassbox.sr.universal_proposer import (
    UniversalProposer,
    UniversalProposerConfig,
    DEFAULT_OPERATOR_VOCAB,
    DEFAULT_SKELETON_VOCAB,
)


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

        # Small observation noise to avoid trivial overfitting.
        y = y + 0.01 * self.rng.randn(*y.shape).astype(np.float32)
        points = np.stack([x, y.astype(np.float32)], axis=1)

        op_target = np.zeros(len(self.operator_vocab), dtype=np.float32)
        for op in ops:
            op_target[self.operator_vocab.index(op)] = 1.0

        return (
            torch.from_numpy(points),
            torch.from_numpy(op_target),
            torch.tensor(kind, dtype=torch.long),
        )


def _train_epoch(model, loader, optimizer, device) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0

    for points, op_target, skeleton_target in loader:
        points = points.to(device)
        op_target = op_target.to(device)
        skeleton_target = skeleton_target.to(device)

        out = model(points)
        op_loss = F.binary_cross_entropy_with_logits(out["operator_logits"], op_target)
        sk_loss = F.cross_entropy(out["skeleton_logits"], skeleton_target)
        loss = op_loss + sk_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * points.shape[0]
        pred = torch.argmax(out["skeleton_logits"], dim=1)
        correct += int((pred == skeleton_target).sum().item())
        total += points.shape[0]

    return total_loss / max(total, 1), correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description="Train universal proposer (Phase 1 scaffold)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--n-points", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--out", type=str, default="models/universal_proposer_mvp.pt")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    config = UniversalProposerConfig(hidden_dim=args.hidden)
    model = UniversalProposer(config).to(device)

    ds = SyntheticCurveDataset(n_samples=args.n_samples, n_points=args.n_points)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        loss, acc = _train_epoch(model, loader, opt, device)
        print(f"epoch={epoch:03d} loss={loss:.5f} skeleton_acc={acc:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "hidden_dim": config.hidden_dim,
                "point_mlp_layers": config.point_mlp_layers,
                "operator_vocab": model.operator_vocab,
                "skeleton_vocab": model.skeleton_vocab,
            },
        },
        out_path,
    )
    print(f"saved model -> {out_path}")


if __name__ == "__main__":
    main()
