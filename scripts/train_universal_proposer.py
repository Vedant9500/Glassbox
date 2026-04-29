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

from glassbox.sr.universal_proposer import (
    UniversalProposer,
    UniversalProposerConfig,
    DEFAULT_OPERATOR_VOCAB,
    DEFAULT_SKELETON_VOCAB,
)

try:
    from scripts.generate_curve_data import evaluate_formula
except Exception:
    from generate_curve_data import evaluate_formula


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


class FormulaReplayDataset(Dataset):
    """Dataset-backed proposer training from generated formula corpora (.npz).

    Expects `.npz` with keys:
    - labels: [N, C] multi-hot operator labels
    - formulas: [N] formula strings
    """

    def __init__(
        self,
        data_path: Path,
        n_points: int = 128,
        x_min: float = -2.0,
        x_max: float = 2.0,
        max_samples: Optional[int] = None,
    ):
        blob = np.load(data_path, allow_pickle=True)
        labels = np.asarray(blob["labels"], dtype=np.float32)
        formulas = blob["formulas"].tolist()

        if max_samples is not None:
            limit = int(max_samples)
            labels = labels[:limit]
            formulas = formulas[:limit]

        self.labels = labels
        self.formulas = formulas
        self.n_points = int(n_points)
        self.x = np.linspace(float(x_min), float(x_max), self.n_points, dtype=np.float32)

        self.operator_vocab = list(DEFAULT_OPERATOR_VOCAB)
        self.skeleton_vocab = list(DEFAULT_SKELETON_VOCAB)

    def __len__(self) -> int:
        return len(self.formulas)

    def _labels_to_operator_target(self, row: np.ndarray) -> np.ndarray:
        # Map existing operator labels (from classifier dataset) into proposer vocab.
        # Known index mapping from generate_curve_data.OPERATOR_CLASSES:
        # identity=0,sin=1,cos=2,power=3,exp=4,log=5,addition=6,multiplication=7,rational=8,...
        op = np.zeros(len(self.operator_vocab), dtype=np.float32)
        if row.shape[0] >= 9:
            op[self.operator_vocab.index("identity")] = row[0]
            op[self.operator_vocab.index("sin")] = row[1]
            op[self.operator_vocab.index("cos")] = row[2]
            op[self.operator_vocab.index("power")] = row[3]
            op[self.operator_vocab.index("exp")] = row[4]
            op[self.operator_vocab.index("log")] = row[5]
            op[self.operator_vocab.index("rational")] = row[8]
            # Derived periodic tag from sin/cos.
            op[self.operator_vocab.index("periodic")] = max(row[1], row[2])
        return op

    def __getitem__(self, idx: int):
        formula = str(self.formulas[idx])
        y, status = evaluate_formula(formula, self.x, safe_eval=True)
        if y is None or status != "ok":
            # Fallback to stable placeholder curve to keep batch shapes valid.
            y = np.zeros_like(self.x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if y.shape != self.x.shape or np.any(~np.isfinite(y)):
            y = np.zeros_like(self.x, dtype=np.float32)

        points = np.stack([self.x, y], axis=1)
        op_target = self._labels_to_operator_target(self.labels[idx])

        # For dataset-backed mode we currently train skeleton head weakly;
        # use -1 target so CE loss can be masked out.
        skeleton_target = -1

        return (
            torch.from_numpy(points),
            torch.from_numpy(op_target),
            torch.tensor(skeleton_target, dtype=torch.long),
        )


def _train_epoch(model, loader, optimizer, device) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0

    for points, op_target, skeleton_target in loader:
        points = points.to(device, non_blocking=True)
        op_target = op_target.to(device, non_blocking=True)
        skeleton_target = skeleton_target.to(device, non_blocking=True)

        out = model(points)
        op_loss = F.binary_cross_entropy_with_logits(out["operator_logits"], op_target)
        valid_skeleton = skeleton_target >= 0
        if bool(valid_skeleton.any()):
            sk_loss = F.cross_entropy(out["skeleton_logits"][valid_skeleton], skeleton_target[valid_skeleton])
        else:
            sk_loss = torch.tensor(0.0, device=device)
        loss = op_loss + sk_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * points.shape[0]
        pred = torch.argmax(out["skeleton_logits"], dim=1)
        if bool(valid_skeleton.any()):
            correct += int((pred[valid_skeleton] == skeleton_target[valid_skeleton]).sum().item())
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
    parser.add_argument("--data", type=str, default="", help="Optional dataset .npz path from generate_curve_data")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap when --data is used")
    args = parser.parse_args()

    device = torch.device(args.device)

    config = UniversalProposerConfig(hidden_dim=args.hidden)
    model = UniversalProposer(config).to(device)

    if args.data:
        ds = FormulaReplayDataset(
            data_path=Path(args.data),
            n_points=args.n_points,
            max_samples=(args.max_samples if args.max_samples > 0 else None),
        )
        print(f"dataset=FormulaReplayDataset samples={len(ds)} path={args.data}")
    else:
        ds = SyntheticCurveDataset(n_samples=args.n_samples, n_points=args.n_points)
        print(f"dataset=SyntheticCurveDataset samples={len(ds)}")
        
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

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)

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
