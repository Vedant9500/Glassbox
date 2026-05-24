"""Train script scaffold for Phase 1 universal proposer MVP.

This script intentionally starts with synthetic data so iteration is fast and
independent of a finalized dataset schema.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
import sys
from typing import List, Tuple, Optional, Sequence

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
    DEFAULT_UNIVARIATE_SKELETON_VOCAB,
    DEFAULT_MULTIVARIATE_SKELETON_VOCAB,
    normalize_formula_key,
)

try:
    from glassbox.curve_classifier.generate_curve_data import (
        extract_all_features,
        evaluate_formula,
        OPERATOR_CLASSES,
        FEATURE_DIM,
        FEATURE_SCHEMA,
        N_CLASSES,
    )
except Exception:
    from glassbox.curve_classifier.generate_curve_data import (
        extract_all_features,
        evaluate_formula,
        OPERATOR_CLASSES,
        FEATURE_DIM,
        FEATURE_SCHEMA,
        N_CLASSES,
    )


def apply_feature_transform(features: np.ndarray) -> np.ndarray:
    """Apply the same selective SymLog transform used by classifier training."""
    x = np.array(features, dtype=np.float32, copy=True)
    if x.ndim == 1:
        end = min(x.shape[0], FEATURE_DIM)
        if end > 192:
            x[192:end] = np.sign(x[192:end]) * np.log1p(np.abs(x[192:end]))
    else:
        end = min(x.shape[1], FEATURE_DIM)
        if end > 192:
            x[:, 192:end] = np.sign(x[:, 192:end]) * np.log1p(np.abs(x[:, 192:end]))
    return x


def _coerce_operator_classes(raw, n_classes: int) -> List[str]:
    if raw is None:
        return list(OPERATOR_CLASSES.keys())[:n_classes]
    if isinstance(raw, np.ndarray):
        raw = raw.tolist()
    return [str(x) for x in raw]


def _normalize_formula_key(formula: str) -> str:
    return normalize_formula_key(formula)


def _canonicalize_formula_for_vocab(formula: str, n_input_vars: int = 1) -> str:
    key = _normalize_formula_key(formula)
    if n_input_vars <= 1:
        return key
    vars_in_key = sorted(set(re.findall(r"\bx\d+\b", key)))
    if len(vars_in_key) == 1:
        key = re.sub(r"\bx\d+\b", "x", key)
    return key


def load_training_data(
    data_path: str,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray, Optional[List[str]], Optional[List[str]], Optional[int], Optional[dict]]:
    """Load proposer training data from the current curve dataset format."""
    blob = np.load(data_path, allow_pickle=True)
    features = np.asarray(blob["features"], dtype=np.float32)
    labels = np.asarray(blob["labels"], dtype=np.float32)
    formulas = blob["formulas"].tolist() if "formulas" in blob else None
    operator_classes = _coerce_operator_classes(
        blob["operator_classes"] if "operator_classes" in blob else None,
        n_classes,
    )
    feature_dim = int(blob["feature_dim"]) if "feature_dim" in blob else int(features.shape[1])
    feature_schema = blob["feature_schema"].item() if "feature_schema" in blob else None
    return features, labels, formulas, operator_classes, feature_dim, feature_schema


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

        batch = apply_feature_transform(batch).astype(np.float64, copy=False)

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
        self.skeleton_vocab = list(DEFAULT_UNIVARIATE_SKELETON_VOCAB)

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

        features = apply_feature_transform(features)

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
    """Dataset-backed proposer training from generated formula corpora (.npz).
    Supports loading directly to VRAM for maximum throughput.
    """

    def __init__(
        self,
        features: np.ndarray | str | Path,
        labels: Optional[np.ndarray] = None,
        indices: Optional[np.ndarray] = None,
        operator_classes: Optional[Sequence[str]] = None,
        formulas: Optional[Sequence[str]] = None,
        scaler: Optional[dict] = None,
        device: Optional[torch.device] = None,
        n_points: Optional[int] = None,
    ):
        if labels is None and isinstance(features, (str, Path)):
            (
                features,
                labels,
                formulas,
                operator_classes,
                _feature_dim,
                _feature_schema,
            ) = load_training_data(str(features), n_classes=N_CLASSES)

        if labels is None:
            raise ValueError("labels must be provided when features is not a dataset path")

        self.indices = (
            np.asarray(indices, dtype=np.int64)
            if indices is not None
            else np.arange(len(features), dtype=np.int64)
        )
        self.scaler = scaler
        self.operator_vocab = list(DEFAULT_OPERATOR_VOCAB)
        self.skeleton_vocab = list(DEFAULT_SKELETON_VOCAB)
        self.operator_classes = _coerce_operator_classes(operator_classes, int(labels.shape[1]))
        self.formulas = list(formulas) if formulas is not None else None
        self.n_points = n_points
        self.n_input_vars = 1
        if self.formulas:
            self.n_input_vars = max(1, self._infer_formula_input_vars(self.formulas))
        self.skeleton_vocab_keys = [self._canonical_vocab_key(item) for item in self.skeleton_vocab]

        self.is_on_device = False
        if device is not None and device.type == 'cuda':
            print(f"Transferring dataset to {device}...")
            # Slice first to save memory
            x_sliced = apply_feature_transform(features[self.indices])
            y_sliced = np.asarray(labels[self.indices], dtype=np.float32)

            if self.scaler is not None:
                x_sliced = (x_sliced - self.scaler['mean']) / (self.scaler['std'] + 1e-8)
            
            # Convert labels to targets eagerly
            op_targets = np.zeros((len(y_sliced), len(self.operator_vocab)), dtype=np.float32)
            skeleton_targets = np.full(len(y_sliced), -1, dtype=np.int64)
            for i, row in enumerate(y_sliced):
                op_targets[i] = self._labels_to_operator_target(row)
                if self.formulas is not None:
                    skeleton_targets[i] = self._formula_to_skeleton_target(self.formulas[int(self.indices[i])])
                
            self.features = torch.from_numpy(x_sliced).to(device)
            self.labels = torch.from_numpy(op_targets).to(device)
            self.skeleton_targets = torch.from_numpy(skeleton_targets).to(device)
            self.indices = np.arange(len(self.indices), dtype=np.int64)
            self.is_on_device = True
        else:
            self.features = features
            self.labels = labels
            self.skeleton_targets = None

    def __len__(self) -> int:
        return len(self.indices)

    def _infer_formula_input_vars(self, formulas: Sequence[str]) -> int:
        max_vars = 1
        for formula in formulas:
            vars_found = set(re.findall(r"\bx\d+\b", str(formula)))
            if re.search(r"\bx\b", str(formula)):
                vars_found.add("x")
            max_vars = max(max_vars, len(vars_found))
        return max_vars

    def _canonical_vocab_key(self, formula: str) -> str:
        key = _normalize_formula_key(formula)
        if self.n_input_vars > 1:
            vars_in_key = sorted(set(re.findall(r"\bx\d+\b", key)))
            if len(vars_in_key) == 1:
                key = re.sub(r"\bx\d+\b", "x", key)
        return key

    def _labels_to_operator_target(self, row: np.ndarray) -> np.ndarray:
        op = np.zeros(len(self.operator_vocab), dtype=np.float32)
        row = np.asarray(row, dtype=np.float32)
        source_idx = {name: i for i, name in enumerate(self.operator_classes)}

        for name in self.operator_vocab:
            if name == "periodic":
                continue
            idx = source_idx.get(name)
            if idx is not None and idx < row.shape[0]:
                op[self.operator_vocab.index(name)] = row[idx]

        if "periodic" in self.operator_vocab:
            sin_idx = source_idx.get("sin")
            cos_idx = source_idx.get("cos")
            sin_val = row[sin_idx] if sin_idx is not None and sin_idx < row.shape[0] else 0.0
            cos_val = row[cos_idx] if cos_idx is not None and cos_idx < row.shape[0] else 0.0
            op[self.operator_vocab.index("periodic")] = max(float(sin_val), float(cos_val))
        return op

    def _formula_to_skeleton_target(self, formula: str) -> int:
        key = self._canonical_vocab_key(formula)
        if key not in self.skeleton_vocab_keys and self.n_input_vars > 1:
            key = _normalize_formula_key(formula)
        try:
            return self.skeleton_vocab_keys.index(key)
        except ValueError:
            return -1

    def __getitem__(self, idx: int):
        sample_idx = int(self.indices[idx])
        
        if self.is_on_device:
            return self.features[sample_idx], self.labels[sample_idx], self.skeleton_targets[sample_idx]
            
        feat = apply_feature_transform(self.features[sample_idx])
        
        if self.scaler is not None:
            feat = (feat - self.scaler['mean']) / (self.scaler['std'] + 1e-8)
            
        op_target = self._labels_to_operator_target(self.labels[sample_idx])
        skeleton_target = -1
        if self.formulas is not None:
            skeleton_target = self._formula_to_skeleton_target(self.formulas[sample_idx])
        return (
            torch.from_numpy(feat.astype(np.float32)),
            torch.from_numpy(op_target),
            torch.tensor(skeleton_target, dtype=torch.long),
        )


def _train_epoch(model, loader, optimizer, device, scaler=None) -> float:
    model.train()
    
    # Fast-path for VRAM-resident datasets (Bypasses Python DataLoader overhead)
    ds = loader.dataset
    if hasattr(ds, "is_on_device") and ds.is_on_device and device.type == 'cuda':
        total_loss = torch.zeros(1, device=device)
        n_samples = len(ds)
        batch_size = loader.batch_size
        
        # GPU-side shuffle (Zero CPU overhead)
        indices = torch.randperm(n_samples, device=device)
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_idx = indices[start_idx:end_idx]
            
            features = ds.features[batch_idx]
            op_target = ds.labels[batch_idx]
            
            optimizer.zero_grad(set_to_none=True)
            # Automatic Mixed Precision for max Tensor Core utilization
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                out = model(features)
                loss = F.binary_cross_entropy_with_logits(out["operator_logits"], op_target)
                skeleton_target = ds.skeleton_targets[batch_idx]
                valid_skeleton = skeleton_target >= 0
                if valid_skeleton.any():
                    loss = loss + 0.2 * F.cross_entropy(
                        out["skeleton_logits"][valid_skeleton],
                        skeleton_target[valid_skeleton],
                    )
                
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            total_loss += loss.detach() * (end_idx - start_idx)
            
        return float((total_loss / max(n_samples, 1)).item())

    # Standard path for RAM/Disk loaded datasets
    total_loss = torch.zeros(1, device=device)
    total_samples = 0

    # Ensure device is CUDA for CUDA graphs (optional, but we'll just optimize the loop)
    for features, op_target, skeleton_target in loader:
        features = features.to(device, non_blocking=True)
        op_target = op_target.to(device, non_blocking=True)
        skeleton_target = skeleton_target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True) 
        
        with torch.autocast(device_type=device.type, enabled=device.type=='cuda', dtype=torch.float16):
            out = model(features)
            loss = F.binary_cross_entropy_with_logits(out["operator_logits"], op_target)
            valid_skeleton = skeleton_target >= 0
            if valid_skeleton.any():
                loss = loss + 0.2 * F.cross_entropy(
                    out["skeleton_logits"][valid_skeleton],
                    skeleton_target[valid_skeleton],
                )

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.detach() * features.shape[0]
        total_samples += features.shape[0]

    return float((total_loss / max(total_samples, 1)).item())


def _evaluate(model, loader, device) -> dict:
    model.eval()
    
    ds = loader.dataset
    all_preds = []
    all_labels = []
    
    if hasattr(ds, "is_on_device") and ds.is_on_device and device.type == 'cuda':
        total_loss = 0.0
        n_samples = len(ds)
        batch_size = loader.batch_size
        
        with torch.no_grad():
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                
                features = ds.features[start_idx:end_idx]
                op_target = ds.labels[start_idx:end_idx]
                
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    out = model(features)
                    loss = F.binary_cross_entropy_with_logits(out["operator_logits"], op_target)
                    skeleton_target = ds.skeleton_targets[start_idx:end_idx]
                    valid_skeleton = skeleton_target >= 0
                    if valid_skeleton.any():
                        loss = loss + 0.2 * F.cross_entropy(
                            out["skeleton_logits"][valid_skeleton],
                            skeleton_target[valid_skeleton],
                        )
                    
                total_loss += loss.item() * (end_idx - start_idx)
                all_preds.append(torch.sigmoid(out["operator_logits"]).cpu())
                all_labels.append(op_target.cpu())
                
        avg_loss = total_loss / max(n_samples, 1)
    else:
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for features, op_target, skeleton_target in loader:
                features = features.to(device, non_blocking=True)
                op_target = op_target.to(device, non_blocking=True)
                skeleton_target = skeleton_target.to(device, non_blocking=True)
                
                with torch.autocast(device_type=device.type, enabled=device.type=='cuda', dtype=torch.float16):
                    out = model(features)
                    loss = F.binary_cross_entropy_with_logits(out["operator_logits"], op_target)
                    valid_skeleton = skeleton_target >= 0
                    if valid_skeleton.any():
                        loss = loss + 0.2 * F.cross_entropy(
                            out["skeleton_logits"][valid_skeleton],
                            skeleton_target[valid_skeleton],
                        )
                    
                total_loss += loss.item() * features.shape[0]
                total_samples += features.shape[0]
                all_preds.append(torch.sigmoid(out["operator_logits"]).cpu())
                all_labels.append(op_target.cpu())
                
        avg_loss = total_loss / max(total_samples, 1)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    binary_preds = (all_preds > 0.5).float()
    tp = ((binary_preds == 1) & (all_labels == 1)).float().sum()
    fp = ((binary_preds == 1) & (all_labels == 0)).float().sum()
    fn = ((binary_preds == 0) & (all_labels == 1)).float().sum()
    
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    
    return {
        "loss": avg_loss,
        "f1": f1.mean().item()
    }

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
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--data", type=str, default="", help="Optional dataset .npz path from generate_curve_data")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap when --data is used")
    parser.add_argument("--load-into-ram", "--load-into-vram", dest="load_into_ram", action="store_true",
                        help="Load dataset fully into RAM/VRAM for maximum throughput")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile (PyTorch 2.0+)")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Enable TF32 for better performance on Ampere+ GPUs (as suggested by the warning)
    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')

    config = UniversalProposerConfig(hidden_dim=args.hidden)
    model = UniversalProposer(config).to(device)
    
    if args.compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile...")
        try:
            # Wrap in try-except because torch.compile often fails on Windows 
            # due to Triton/Inductor environment issues.
            model = torch.compile(model)
        except Exception as e:
            print(f"Warning: torch.compile failed ({e}). Falling back to eager mode.")

    if args.data:
        if args.data.endswith(".npz"):
            (
                features,
                labels,
                formulas,
                operator_classes,
                feature_dim,
                feature_schema,
            ) = load_training_data(args.data, n_classes=N_CLASSES)
        else:
            # Try loading streamed .dat files
            base = Path(args.data)
            features_path = base.with_suffix(".features.dat")
            labels_path = base.with_suffix(".labels.dat")
            if not features_path.exists() or not labels_path.exists():
                raise FileNotFoundError(f"Could not find .npz or .dat files for {args.data}")
            
            # Infer sizes
            # We assume features are n_samples x 398 (the new feature dim)
            feature_dim = FEATURE_DIM
            n_classes = N_CLASSES
            file_size = features_path.stat().st_size
            n_samples = file_size // (feature_dim * 4)
            print(f"Inferred n_samples={n_samples} from {features_path.name}")
            
            features = np.memmap(features_path, dtype=np.float32, mode="r", shape=(n_samples, feature_dim))
            labels = np.memmap(labels_path, dtype=np.float32, mode="r", shape=(n_samples, n_classes))
            formulas = None
            operator_classes = list(OPERATOR_CLASSES.keys())[:n_classes]
            feature_schema = None
            
        if args.max_samples > 0:
            features = features[:args.max_samples]
            labels = labels[:args.max_samples]

        indices = np.arange(len(features))
        np.random.shuffle(indices)
        
        n_val = int(len(features) * args.val_split)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]
        
        print("Computing feature statistics (SymLog + Standardize)...")
        mean, std = compute_feature_stats(features, train_idx)
        feature_scaler = {'mean': mean, 'std': std}

        # VRAM loading option
        load_to_vram = args.load_into_ram
        
        train_ds = FormulaReplayDataset(
            features, labels, train_idx, operator_classes=operator_classes, formulas=formulas, scaler=feature_scaler,
            device=device if load_to_vram else None
        )
        val_ds = FormulaReplayDataset(
            features, labels, val_idx, operator_classes=operator_classes, formulas=formulas, scaler=feature_scaler,
            device=device if load_to_vram else None
        )
        print(f"train_samples={len(train_ds)} val_samples={len(val_ds)} path={args.data}")
        if feature_schema is not None:
            print(f"  Feature schema: {feature_schema}")
    else:
        feature_scaler = None
        # Minimal synthetic dataset fallback
        train_ds = SyntheticCurveDataset(n_samples=args.n_samples, n_points=args.n_points)
        val_ds = SyntheticCurveDataset(n_samples=int(args.n_samples * args.val_split), n_points=args.n_points)
        print(f"train_samples={len(train_ds)} val_samples={len(val_ds)}")
        
    import os
    import platform
    use_cuda = device.type == 'cuda'
    
    # On Windows, num_workers > 0 with large datasets often causes pickling errors or deadlocks
    # due to the 'spawn' method. If data is already in VRAM, workers are unnecessary.
    n_workers = 0 
    if platform.system() != "Windows" and use_cuda and not getattr(train_ds, "is_on_device", False):
        num_cpus = os.cpu_count() or 4
        n_workers = min(8, max(2, num_cpus - 2))
    
    loader_kwargs = {'num_workers': n_workers, 'pin_memory': use_cuda and not getattr(train_ds, "is_on_device", False)}
    if n_workers > 0:
        loader_kwargs['prefetch_factor'] = 2
        loader_kwargs['persistent_workers'] = True

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5)
    
    amp_scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    patience_counter = 0

    print(f"Training GLU Proposer on {device}...")
    for epoch in range(1, args.epochs + 1):
        try:
            train_loss = _train_epoch(model, train_loader, opt, device, amp_scaler)
        except Exception as e:
            if args.compile and "inductor" in str(e).lower():
                print(f"\n[!] torch.compile failed during first forward pass: {e}")
                print("[!] Falling back to eager mode for the rest of training.")
                if hasattr(model, "_orig_mod"):
                    model = model._orig_mod
                args.compile = False 
                train_loss = _train_epoch(model, train_loader, opt, device, amp_scaler)
            else:
                raise e
        
        val_metrics = _evaluate(model, val_loader, device)
        val_loss = val_metrics["loss"]
        val_f1 = val_metrics["f1"]
        
        scheduler.step(val_f1)

        print(f"Epoch {epoch:03d}/{args.epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val F1: {val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "hidden_dim": config.hidden_dim,
                        "n_features": config.n_features,
                        "supports_multivariate_formulas": config.supports_multivariate_formulas,
                        "max_input_vars": config.max_input_vars,
                        "operator_vocab": model.operator_vocab,
                        "skeleton_vocab": model.skeleton_vocab,
                    },
                    "feature_scaler": feature_scaler,
                    "epoch": epoch,
                    "val_f1": best_f1,
                },
                out_path,
            )
            print(f"  -> Saved best model (val_f1: {val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    print(f"\nTraining complete. Best Val F1: {best_f1:.4f}. Model saved to {out_path}")


if __name__ == "__main__":
    main()
