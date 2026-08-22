"""Train script scaffold for Phase 1 universal proposer MVP.

This script intentionally starts with synthetic data so iteration is fast and
independent of a finalized dataset schema.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from pathlib import Path

# Add the repository root to sys.path so we can import glassbox
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from glassbox.universal_proposer import (
    DEFAULT_OPERATOR_VOCAB,
    DEFAULT_SKELETON_VOCAB,
    DEFAULT_UNIVARIATE_SKELETON_VOCAB,
    UNIVERSAL_PROPOSER_ARCHITECTURE_VERSION,
    UNIVERSAL_PROPOSER_CONTRACT_VERSION,
    UNIVERSAL_PROPOSER_ROLE,
    UniversalProposer,
    UniversalProposerConfig,
    normalize_formula_key,
)

try:
    from glassbox.curve_classifier.generate_curve_data import (
        FEATURE_DIM,
        N_CLASSES,
        OPERATOR_CLASSES,
        extract_all_features,
        extract_all_features_xy,
    )
    from glassbox.curve_classifier.rollout import (
        build_checkpoint_card,
        build_rollout_comparison,
        default_checkpoint_card_path,
        default_rollout_comparison_path,
        load_json_report,
        write_checkpoint_card,
        write_rollout_comparison,
    )
    from glassbox.curve_classifier.validation import (
        build_validation_report,
        default_validation_report_path,
        family_holdout_split,
        formula_keys_from_metadata_or_formulas,
        grouped_train_val_split,
        metrics_to_json_dict,
        multilabel_metrics_by_group,
        row_train_val_split,
        write_validation_report,
    )
except Exception:
    from glassbox.curve_classifier.generate_curve_data import (
        FEATURE_DIM,
        N_CLASSES,
        OPERATOR_CLASSES,
        extract_all_features,
        extract_all_features_xy,
    )
    from glassbox.curve_classifier.rollout import (
        build_checkpoint_card,
        build_rollout_comparison,
        default_checkpoint_card_path,
        default_rollout_comparison_path,
        load_json_report,
        write_checkpoint_card,
        write_rollout_comparison,
    )
    from glassbox.curve_classifier.validation import (
        build_validation_report,
        default_validation_report_path,
        family_holdout_split,
        formula_keys_from_metadata_or_formulas,
        grouped_train_val_split,
        metrics_to_json_dict,
        multilabel_metrics_by_group,
        row_train_val_split,
        write_validation_report,
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


def _coerce_operator_classes(raw, n_classes: int) -> list[str]:
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
    return_metadata: bool = False,
) -> tuple:
    """Load proposer training data from the current curve dataset format."""
    blob = np.load(data_path, allow_pickle=True)
    features = np.asarray(blob["features"], dtype=np.float32)
    labels = np.asarray(blob["labels"], dtype=np.float32)
    formulas = blob["formulas"].tolist() if "formulas" in blob else None
    operator_classes = _coerce_operator_classes(
        blob["operator_classes"] if "operator_classes" in blob else None,
        n_classes,
    )
    feature_dim = (
        int(blob["feature_dim"]) if "feature_dim" in blob else int(features.shape[1])
    )
    feature_schema = blob["feature_schema"].item() if "feature_schema" in blob else None
    n_loaded = int(features.shape[0])
    metadata = {
        "dataset_path": str(data_path),
        "formula_keys": formula_keys_from_metadata_or_formulas(
            blob["formula_keys"] if "formula_keys" in blob else None,
            formulas,
            limit=n_loaded,
        ),
        "generator_families": (
            np.asarray(blob["generator_families"][:n_loaded], dtype=object)
            if "generator_families" in blob
            else None
        ),
        "template_ids": (
            np.asarray(blob["template_ids"][:n_loaded], dtype=object)
            if "template_ids" in blob
            else None
        ),
        "labeler_version": str(blob["labeler_version"])
        if "labeler_version" in blob
        else None,
        "labels_match_semantic": (
            np.asarray(blob["labels_match_semantic"][:n_loaded], dtype=bool)
            if "labels_match_semantic" in blob
            else None
        ),
    }
    if return_metadata:
        return (
            features,
            labels,
            formulas,
            operator_classes,
            feature_dim,
            feature_schema,
            metadata,
        )
    return features, labels, formulas, operator_classes, feature_dim, feature_schema


def compute_feature_stats(
    features: np.ndarray,
    indices: np.ndarray,
    chunk_size: int = 65536,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute feature mean/std on selected rows without full subset materialization."""
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) == 0:
        raise ValueError("Cannot compute feature stats for empty indices")

    n_features = int(features.shape[1])
    total_count = 0
    sum_x = np.zeros(n_features, dtype=np.float64)
    sum_x2 = np.zeros(n_features, dtype=np.float64)

    for start in range(0, len(indices), chunk_size):
        batch_idx = indices[start : start + chunk_size]
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


def weights_only_safe_scaler(scaler):
    """Convert NumPy scaler values to plain Python data for weights-only loading."""
    if not isinstance(scaler, dict):
        return scaler
    safe = {}
    for key, value in scaler.items():
        if isinstance(value, np.ndarray):
            safe[key] = value.astype(np.float32).tolist()
        elif isinstance(value, np.generic):
            safe[key] = value.item()
        else:
            safe[key] = value
    return safe


def compute_operator_pos_weight(
    labels: np.ndarray,
    indices: np.ndarray,
    operator_classes: Sequence[str],
    cap: float = 3.0,
) -> torch.Tensor:
    """Compute BCE positive weights for the proposer operator vocabulary."""
    subset = np.asarray(labels[np.asarray(indices, dtype=np.int64)], dtype=np.float32)
    source_idx = {
        name: i
        for i, name in enumerate(
            _coerce_operator_classes(operator_classes, subset.shape[1])
        )
    }
    weights = []
    for name in DEFAULT_OPERATOR_VOCAB:
        if name == "periodic":
            sin_idx = source_idx.get("sin")
            cos_idx = source_idx.get("cos")
            sin_val = (
                subset[:, sin_idx]
                if sin_idx is not None and sin_idx < subset.shape[1]
                else 0.0
            )
            cos_val = (
                subset[:, cos_idx]
                if cos_idx is not None and cos_idx < subset.shape[1]
                else 0.0
            )
            positive = np.maximum(sin_val, cos_val)
        else:
            idx = source_idx.get(name)
            positive = (
                subset[:, idx]
                if idx is not None and idx < subset.shape[1]
                else np.zeros(subset.shape[0])
            )
        pos = float(np.sum(positive > 0.5))
        neg = float(max(0, subset.shape[0] - pos))
        weights.append(np.clip(neg / max(pos, 1.0), 0.5, float(cap)))
    return torch.tensor(weights, dtype=torch.float32)


def skeleton_loss_enabled_from_coverage(
    dataset: Dataset, min_coverage: float = 0.80
) -> tuple[bool, float]:
    """Enable skeleton loss only when fixed vocab covers enough training rows."""
    targets = getattr(dataset, "skeleton_targets", None)
    if targets is not None:
        valid = (targets >= 0).float().mean().item() if len(dataset) else 0.0
        return bool(valid >= min_coverage), float(valid)

    valid = 0
    total = len(dataset)
    for i in range(total):
        _features, _op_target, skeleton_target = dataset[i]
        if int(skeleton_target.item()) >= 0:
            valid += 1
    coverage = valid / max(1, total)
    return bool(coverage >= min_coverage), float(coverage)


def select_checkpoint_metric(metrics: dict) -> float:
    """Use runtime-relevant metric when present; otherwise micro-F1 operator prior quality."""
    recall = metrics.get("candidate_recall_after_affine_fit")
    if recall is not None:
        try:
            recall_val = float(recall)
            if np.isfinite(recall_val):
                return recall_val
        except (TypeError, ValueError):
            pass
    micro_f1 = metrics.get("micro_f1")
    if micro_f1 is not None:
        return float(micro_f1)
    return float(metrics.get("f1", 0.0))


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
            y = x**2
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

        # H-15: match inference — extract_all_features_xy sorts/resamples by x.
        y = y + 0.01 * self.rng.randn(*y.shape).astype(np.float32)
        try:
            features = extract_all_features_xy(x, y)
        except Exception:
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
        labels: np.ndarray | None = None,
        indices: np.ndarray | None = None,
        operator_classes: Sequence[str] | None = None,
        formulas: Sequence[str] | None = None,
        scaler: dict | None = None,
        device: torch.device | None = None,
        n_points: int | None = None,
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
            raise ValueError(
                "labels must be provided when features is not a dataset path"
            )

        self.indices = (
            np.asarray(indices, dtype=np.int64)
            if indices is not None
            else np.arange(len(features), dtype=np.int64)
        )
        self.scaler = scaler
        self.operator_vocab = list(DEFAULT_OPERATOR_VOCAB)
        self.skeleton_vocab = list(DEFAULT_SKELETON_VOCAB)
        self.operator_classes = _coerce_operator_classes(
            operator_classes, int(labels.shape[1])
        )
        self.formulas = list(formulas) if formulas is not None else None
        self.n_points = n_points
        self.n_input_vars = 1
        if self.formulas:
            self.n_input_vars = max(1, self._infer_formula_input_vars(self.formulas))
        self.skeleton_vocab_keys = [
            self._canonical_vocab_key(item) for item in self.skeleton_vocab
        ]

        self.is_on_device = False
        if device is not None and device.type == "cuda":
            print(f"Transferring dataset to {device}...")
            # Slice first to save memory
            x_sliced = apply_feature_transform(features[self.indices])
            y_sliced = np.asarray(labels[self.indices], dtype=np.float32)

            if self.scaler is not None:
                x_sliced = (x_sliced - self.scaler["mean"]) / (
                    self.scaler["std"] + 1e-8
                )

            # Convert labels to targets eagerly
            op_targets = np.zeros(
                (len(y_sliced), len(self.operator_vocab)), dtype=np.float32
            )
            skeleton_targets = np.full(len(y_sliced), -1, dtype=np.int64)
            for i, row in enumerate(y_sliced):
                op_targets[i] = self._labels_to_operator_target(row)
                if self.formulas is not None:
                    skeleton_targets[i] = self._formula_to_skeleton_target(
                        self.formulas[int(self.indices[i])]
                    )

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
            sin_val = (
                row[sin_idx] if sin_idx is not None and sin_idx < row.shape[0] else 0.0
            )
            cos_val = (
                row[cos_idx] if cos_idx is not None and cos_idx < row.shape[0] else 0.0
            )
            op[self.operator_vocab.index("periodic")] = max(
                float(sin_val), float(cos_val)
            )
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
            return (
                self.features[sample_idx],
                self.labels[sample_idx],
                self.skeleton_targets[sample_idx],
            )

        feat = apply_feature_transform(self.features[sample_idx])

        if self.scaler is not None:
            feat = (feat - self.scaler["mean"]) / (self.scaler["std"] + 1e-8)

        op_target = self._labels_to_operator_target(self.labels[sample_idx])
        skeleton_target = -1
        if self.formulas is not None:
            skeleton_target = self._formula_to_skeleton_target(
                self.formulas[sample_idx]
            )
        return (
            torch.from_numpy(feat.astype(np.float32)),
            torch.from_numpy(op_target),
            torch.tensor(skeleton_target, dtype=torch.long),
        )


def _train_epoch(
    model,
    loader,
    optimizer,
    device,
    scaler=None,
    operator_pos_weight: torch.Tensor | None = None,
    skeleton_loss_weight: float = 0.2,
) -> float:
    model.train()

    # Fast-path for VRAM-resident datasets (Bypasses Python DataLoader overhead)
    ds = loader.dataset
    if hasattr(ds, "is_on_device") and ds.is_on_device and device.type == "cuda":
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
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(features)
                loss = F.binary_cross_entropy_with_logits(
                    out["operator_logits"],
                    op_target,
                    pos_weight=operator_pos_weight,
                )
                skeleton_target = ds.skeleton_targets[batch_idx]
                valid_skeleton = skeleton_target >= 0
                if skeleton_loss_weight > 0.0 and valid_skeleton.any():
                    loss = loss + skeleton_loss_weight * F.cross_entropy(
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

        with torch.autocast(
            device_type=device.type, enabled=device.type == "cuda", dtype=torch.float16
        ):
            out = model(features)
            loss = F.binary_cross_entropy_with_logits(
                out["operator_logits"],
                op_target,
                pos_weight=operator_pos_weight,
            )
            valid_skeleton = skeleton_target >= 0
            if skeleton_loss_weight > 0.0 and valid_skeleton.any():
                loss = loss + skeleton_loss_weight * F.cross_entropy(
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


def _skeleton_metric_summary(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    if logits.numel() == 0 or targets.numel() == 0:
        return {
            "skeleton_valid_count": 0,
            "skeleton_top1_acc": None,
            "skeleton_top5_acc": None,
            "skeleton_ece_10": None,
            "skeleton_confidence_mean": None,
        }

    valid = targets >= 0
    valid_count = int(valid.sum().item())
    if valid_count == 0:
        return {
            "skeleton_valid_count": 0,
            "skeleton_top1_acc": None,
            "skeleton_top5_acc": None,
            "skeleton_ece_10": None,
            "skeleton_confidence_mean": None,
        }

    logits_valid = logits[valid]
    targets_valid = targets[valid]
    probs = torch.softmax(logits_valid, dim=1)
    conf, pred = probs.max(dim=1)
    correct = (pred == targets_valid).float()
    top_k = min(5, probs.shape[1])
    topk = torch.topk(probs, k=top_k, dim=1).indices
    top5_correct = (topk == targets_valid.unsqueeze(1)).any(dim=1).float()

    ece = torch.tensor(0.0)
    bin_edges = torch.linspace(0.0, 1.0, 11)
    for start, end in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (conf >= start) & (conf < end)
        if end >= 1.0:
            mask = (conf >= start) & (conf <= end)
        if not mask.any():
            continue
        bin_acc = correct[mask].mean()
        bin_conf = conf[mask].mean()
        ece = ece + (mask.float().mean() * torch.abs(bin_acc - bin_conf))

    return {
        "skeleton_valid_count": valid_count,
        "skeleton_top1_acc": float(correct.mean().item()),
        "skeleton_top5_acc": float(top5_correct.mean().item()),
        "skeleton_ece_10": float(ece.item()),
        "skeleton_confidence_mean": float(conf.mean().item()),
    }


def _evaluate(
    model,
    loader,
    device,
    operator_pos_weight: torch.Tensor | None = None,
    skeleton_loss_weight: float = 0.2,
) -> dict:
    model.eval()

    ds = loader.dataset
    all_preds = []
    all_labels = []
    all_skeleton_logits = []
    all_skeleton_targets = []

    if hasattr(ds, "is_on_device") and ds.is_on_device and device.type == "cuda":
        total_loss = 0.0
        n_samples = len(ds)
        batch_size = loader.batch_size

        with torch.no_grad():
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)

                features = ds.features[start_idx:end_idx]
                op_target = ds.labels[start_idx:end_idx]

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = model(features)
                    loss = F.binary_cross_entropy_with_logits(
                        out["operator_logits"],
                        op_target,
                        pos_weight=operator_pos_weight,
                    )
                    skeleton_target = ds.skeleton_targets[start_idx:end_idx]
                    valid_skeleton = skeleton_target >= 0
                    if skeleton_loss_weight > 0.0 and valid_skeleton.any():
                        loss = loss + skeleton_loss_weight * F.cross_entropy(
                            out["skeleton_logits"][valid_skeleton],
                            skeleton_target[valid_skeleton],
                        )

                total_loss += loss.item() * (end_idx - start_idx)
                all_preds.append(torch.sigmoid(out["operator_logits"]).cpu())
                all_labels.append(op_target.cpu())
                all_skeleton_logits.append(out["skeleton_logits"].detach().cpu())
                all_skeleton_targets.append(skeleton_target.detach().cpu())

        avg_loss = total_loss / max(n_samples, 1)
    else:
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for features, op_target, skeleton_target in loader:
                features = features.to(device, non_blocking=True)
                op_target = op_target.to(device, non_blocking=True)
                skeleton_target = skeleton_target.to(device, non_blocking=True)

                with torch.autocast(
                    device_type=device.type,
                    enabled=device.type == "cuda",
                    dtype=torch.float16,
                ):
                    out = model(features)
                    loss = F.binary_cross_entropy_with_logits(
                        out["operator_logits"],
                        op_target,
                        pos_weight=operator_pos_weight,
                    )
                    valid_skeleton = skeleton_target >= 0
                    if skeleton_loss_weight > 0.0 and valid_skeleton.any():
                        loss = loss + skeleton_loss_weight * F.cross_entropy(
                            out["skeleton_logits"][valid_skeleton],
                            skeleton_target[valid_skeleton],
                        )

                total_loss += loss.item() * features.shape[0]
                total_samples += features.shape[0]
                all_preds.append(torch.sigmoid(out["operator_logits"]).cpu())
                all_labels.append(op_target.cpu())
                all_skeleton_logits.append(out["skeleton_logits"].detach().cpu())
                all_skeleton_targets.append(skeleton_target.detach().cpu())

        avg_loss = total_loss / max(total_samples, 1)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    binary_preds = (all_preds > 0.5).float()
    tp_class = ((binary_preds == 1) & (all_labels == 1)).float().sum(dim=0)
    fp_class = ((binary_preds == 1) & (all_labels == 0)).float().sum(dim=0)
    fn_class = ((binary_preds == 0) & (all_labels == 1)).float().sum(dim=0)

    precision = tp_class / (tp_class + fp_class + 1e-10)
    recall = tp_class / (tp_class + fn_class + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    tp = tp_class.sum()
    fp = fp_class.sum()
    fn = fn_class.sum()
    micro_f1 = 2 * tp / (2 * tp + fp + fn + 1e-10)

    skeleton_logits = (
        torch.cat(all_skeleton_logits) if all_skeleton_logits else torch.empty(0, 0)
    )
    skeleton_targets = (
        torch.cat(all_skeleton_targets)
        if all_skeleton_targets
        else torch.empty(0, dtype=torch.long)
    )
    skeleton_metrics = _skeleton_metric_summary(skeleton_logits, skeleton_targets)

    return {
        "loss": avg_loss,
        "f1": f1.mean().item(),
        "micro_f1": micro_f1.item(),
        "precision_per_operator": precision.numpy(),
        "recall_per_operator": recall.numpy(),
        "f1_per_operator": f1.numpy(),
        "skeleton_coverage": skeleton_metrics["skeleton_valid_count"]
        / max(1, int(all_labels.shape[0])),
        "candidate_recall_after_affine_fit": None,
        "candidate_recall_after_affine_fit_note": (
            "Not computed from precomputed feature datasets; requires raw (x, y) curves."
        ),
        "preds": all_preds,
        "labels": all_labels,
        **skeleton_metrics,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train universal proposer (Phase 1 scaffold)"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--n-points", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument(
        "--out", type=str, default="models/universal_proposer_robust.pt"
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--val-split", type=float, default=0.1, help="Validation split ratio"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="",
        help="Optional dataset .npz path from generate_curve_data",
    )
    parser.add_argument(
        "--max-samples", type=int, default=0, help="Optional cap when --data is used"
    )
    parser.add_argument(
        "--load-into-ram",
        "--load-into-vram",
        dest="load_into_ram",
        action="store_true",
        help="Load dataset fully into RAM/VRAM for maximum throughput",
    )
    parser.add_argument(
        "--compile", action="store_true", help="Use torch.compile (PyTorch 2.0+)"
    )
    parser.add_argument(
        "--split-policy",
        type=str,
        default="auto",
        choices=["auto", "row", "formula_group", "family_holdout"],
        help="Validation split policy. auto uses formula groups when dataset metadata is present.",
    )
    parser.add_argument(
        "--heldout-family",
        type=str,
        default="",
        help="Generator family to hold out when --split-policy=family_holdout",
    )
    parser.add_argument(
        "--validation-report",
        type=str,
        default="",
        help="Optional output path for Phase 3 validation report JSON",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--checkpoint-card",
        type=str,
        default="",
        help="Optional output path for Phase 6 checkpoint card JSON",
    )
    parser.add_argument(
        "--data-generation-command",
        type=str,
        default="",
        help="Command used to generate this training dataset, saved in the checkpoint card",
    )
    parser.add_argument(
        "--baseline-card",
        type=str,
        default="",
        help="Optional baseline checkpoint card for Phase 6 rollout comparison",
    )
    parser.add_argument(
        "--rollout-comparison",
        type=str,
        default="",
        help="Optional output path for Phase 6 rollout comparison JSON",
    )
    parser.add_argument(
        "--rollout-metric",
        type=str,
        default="val_f1",
        help="Metric name used for optional baseline comparison",
    )
    parser.add_argument(
        "--min-relative-improvement",
        type=float,
        default=0.0,
        help="Minimum relative improvement over the baseline metric for rollout readiness",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable inverse-frequency positive weights for operator BCE",
    )
    parser.add_argument(
        "--class-weight-cap",
        type=float,
        default=3.0,
        help="Maximum positive class weight for operator BCE",
    )
    parser.add_argument(
        "--skeleton-min-coverage",
        type=float,
        default=0.80,
        help="Minimum train-set fixed-vocab skeleton coverage required to train skeleton loss",
    )
    parser.add_argument(
        "--skeleton-loss-weight",
        type=float,
        default=0.2,
        help="Skeleton cross-entropy loss weight when coverage gate passes",
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Enable TF32 for better performance on Ampere+ GPUs (as suggested by the warning)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    config = UniversalProposerConfig(hidden_dim=args.hidden)
    model = UniversalProposer(config).to(device)
    dataset_metadata = {}

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
                dataset_metadata,
            ) = load_training_data(args.data, n_classes=N_CLASSES, return_metadata=True)
        else:
            # Try loading streamed .dat files
            base = Path(args.data)
            features_path = base.with_suffix(".features.dat")
            labels_path = base.with_suffix(".labels.dat")
            if not features_path.exists() or not labels_path.exists():
                raise FileNotFoundError(
                    f"Could not find .npz or .dat files for {args.data}"
                )

            # Infer sizes
            # We assume features are n_samples x 398 (the new feature dim)
            feature_dim = FEATURE_DIM
            n_classes = N_CLASSES
            file_size = features_path.stat().st_size
            n_samples = file_size // (feature_dim * 4)
            print(f"Inferred n_samples={n_samples} from {features_path.name}")

            features = np.memmap(
                features_path,
                dtype=np.float32,
                mode="r",
                shape=(n_samples, feature_dim),
            )
            labels = np.memmap(
                labels_path, dtype=np.float32, mode="r", shape=(n_samples, n_classes)
            )
            formulas = None
            operator_classes = list(OPERATOR_CLASSES.keys())[:n_classes]
            feature_schema = None
            dataset_metadata = {"dataset_path": str(args.data)}

        if args.max_samples > 0:
            features = features[: args.max_samples]
            labels = labels[: args.max_samples]
            if formulas is not None:
                formulas = formulas[: args.max_samples]
            for key in (
                "formula_keys",
                "generator_families",
                "template_ids",
                "labels_match_semantic",
            ):
                value = dataset_metadata.get(key)
                if value is not None:
                    dataset_metadata[key] = np.asarray(value)[: args.max_samples]

        formula_keys = dataset_metadata.get("formula_keys")
        generator_families = dataset_metadata.get("generator_families")
        template_ids = dataset_metadata.get("template_ids")
        split_policy = args.split_policy
        split_details = {}
        if split_policy == "family_holdout" or args.heldout_family:
            if generator_families is None:
                raise ValueError(
                    "--split-policy=family_holdout requires generator_families metadata"
                )
            heldout_family = args.heldout_family
            if not heldout_family:
                family_counts = {}
                for family in np.asarray(generator_families, dtype=object).astype(str):
                    family_counts[family] = family_counts.get(family, 0) + 1
                heldout_family = min(family_counts, key=family_counts.get)
            train_idx, val_idx, split_details = family_holdout_split(
                generator_families, heldout_family
            )
            split_policy = "family_holdout"
        elif split_policy in {"auto", "formula_group"} and formula_keys is not None:
            train_idx, val_idx, split_details = grouped_train_val_split(
                formula_keys, args.val_split, args.seed
            )
            split_policy = str(split_details.get("policy", "formula_group"))
        else:
            train_idx, val_idx = row_train_val_split(
                len(features), args.val_split, args.seed
            )
            split_policy = "row"
            split_details = {"policy": "row", "exclusive_groups": False}

        print("Computing feature statistics (SymLog + Standardize)...")
        mean, std = compute_feature_stats(features, train_idx)
        feature_scaler = {"mean": mean, "std": std}

        # VRAM loading option
        load_to_vram = args.load_into_ram

        train_ds = FormulaReplayDataset(
            features,
            labels,
            train_idx,
            operator_classes=operator_classes,
            formulas=formulas,
            scaler=feature_scaler,
            device=device if load_to_vram else None,
        )
        val_ds = FormulaReplayDataset(
            features,
            labels,
            val_idx,
            operator_classes=operator_classes,
            formulas=formulas,
            scaler=feature_scaler,
            device=device if load_to_vram else None,
        )
        operator_pos_weight = None
        if not args.no_class_weights:
            operator_pos_weight = compute_operator_pos_weight(
                labels,
                train_idx,
                operator_classes,
                cap=args.class_weight_cap,
            ).to(device)
            print(
                f"  Operator pos_weight: {operator_pos_weight.detach().cpu().numpy().round(2).tolist()}"
            )
        skeleton_enabled, skeleton_coverage = skeleton_loss_enabled_from_coverage(
            train_ds,
            min_coverage=args.skeleton_min_coverage,
        )
        skeleton_loss_weight = float(
            args.skeleton_loss_weight if skeleton_enabled else 0.0
        )
        print(
            "  Skeleton loss: "
            f"{'enabled' if skeleton_enabled else 'disabled'} "
            f"(coverage={skeleton_coverage:.3f}, min={args.skeleton_min_coverage:.3f}, "
            f"weight={skeleton_loss_weight:.3f})"
        )
        print(
            f"train_samples={len(train_ds)} val_samples={len(val_ds)} path={args.data}"
        )
        validation_report = build_validation_report(
            dataset_path=str(args.data),
            split_policy=split_policy,
            train_idx=train_idx,
            val_idx=val_idx,
            labels=np.asarray(labels, dtype=np.float32),
            operator_classes=operator_classes,
            formula_keys=formula_keys,
            generator_families=generator_families,
            template_ids=template_ids,
            split_details=split_details,
            notes=[
                "Proposer Phase 3 report includes skeleton top-k metrics on formulas covered by the fixed skeleton vocabulary.",
                "candidate_recall_after_affine_fit is not computed for feature-only datasets because raw (x, y) curves are unavailable.",
            ],
        )
        print(f"  Split policy: {split_policy}")
        if validation_report["formula_overlap"].get("available"):
            overlap = validation_report["formula_overlap"]
            print(
                "  Formula overlap: "
                f"{overlap['overlap_unique_formulas']} unique, "
                f"{overlap['val_rows_with_train_formula_fraction']:.3f} val-row fraction"
            )
        if feature_schema is not None:
            print(f"  Feature schema: {feature_schema}")
        val_groups_for_metrics = (
            np.asarray(generator_families, dtype=object)[val_idx]
            if generator_families is not None
            else None
        )
    else:
        feature_scaler = None
        operator_pos_weight = None
        skeleton_loss_weight = float(args.skeleton_loss_weight)
        skeleton_coverage = 1.0
        val_groups_for_metrics = None
        validation_report = None
        split_policy = "synthetic_row"
        split_details = {"policy": "synthetic_row", "exclusive_groups": False}
        n_val = int(args.n_samples * args.val_split)
        if n_val < 1 or args.n_samples < 1:
            raise ValueError(
                f"--val-split={args.val_split} with --n-samples={args.n_samples} "
                "must create at least one validation sample."
            )
        # Minimal synthetic dataset fallback
        train_ds = SyntheticCurveDataset(
            n_samples=args.n_samples, n_points=args.n_points
        )
        val_ds = SyntheticCurveDataset(n_samples=n_val, n_points=args.n_points)
        print(f"train_samples={len(train_ds)} val_samples={len(val_ds)}")

    import os
    import platform

    use_cuda = device.type == "cuda"

    # On Windows, num_workers > 0 with large datasets often causes pickling errors or deadlocks
    # due to the 'spawn' method. If data is already in VRAM, workers are unnecessary.
    n_workers = 0
    if (
        platform.system() != "Windows"
        and use_cuda
        and not getattr(train_ds, "is_on_device", False)
    ):
        num_cpus = os.cpu_count() or 4
        n_workers = min(8, max(2, num_cpus - 2))

    loader_kwargs = {
        "num_workers": n_workers,
        "pin_memory": use_cuda and not getattr(train_ds, "is_on_device", False),
    }
    if n_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=5
    )

    amp_scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    best_selection_metric = -1.0
    best_metrics = {}
    patience_counter = 0

    print(f"Training GLU Proposer on {device}...")
    for epoch in range(1, args.epochs + 1):
        try:
            train_loss = _train_epoch(
                model,
                train_loader,
                opt,
                device,
                amp_scaler,
                operator_pos_weight=operator_pos_weight,
                skeleton_loss_weight=skeleton_loss_weight,
            )
        except Exception as e:
            if args.compile and "inductor" in str(e).lower():
                print(f"\n[!] torch.compile failed during first forward pass: {e}")
                print("[!] Falling back to eager mode for the rest of training.")
                if hasattr(model, "_orig_mod"):
                    model = model._orig_mod
                args.compile = False
                train_loss = _train_epoch(
                    model,
                    train_loader,
                    opt,
                    device,
                    amp_scaler,
                    operator_pos_weight=operator_pos_weight,
                    skeleton_loss_weight=skeleton_loss_weight,
                )
            else:
                raise e

        val_metrics = _evaluate(
            model,
            val_loader,
            device,
            operator_pos_weight=operator_pos_weight,
            skeleton_loss_weight=skeleton_loss_weight,
        )
        val_loss = val_metrics["loss"]
        val_f1 = val_metrics["f1"]
        selection_metric = select_checkpoint_metric(val_metrics)

        scheduler.step(selection_metric)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | Train Loss: {train_loss:.5f} | "
            f"Val Loss: {val_loss:.5f} | Val F1: {val_f1:.4f} | "
            f"Val Micro F1: {val_metrics['micro_f1']:.4f} | Select: {selection_metric:.4f}"
        )

        if selection_metric > best_selection_metric:
            best_f1 = val_f1
            best_selection_metric = selection_metric
            best_metrics = metrics_to_json_dict(val_metrics)
            best_metrics["selection_metric"] = float(selection_metric)
            best_metrics["selection_metric_name"] = (
                "candidate_recall_after_affine_fit"
                if val_metrics.get("candidate_recall_after_affine_fit") is not None
                else "micro_f1"
            )
            if val_groups_for_metrics is not None:
                best_metrics["by_family"] = multilabel_metrics_by_group(
                    val_metrics["preds"].detach().cpu().numpy(),
                    val_metrics["labels"].detach().cpu().numpy(),
                    val_groups_for_metrics,
                    model.operator_vocab,
                )
            best_metrics["skeleton_loss_weight"] = float(skeleton_loss_weight)
            best_metrics["train_skeleton_coverage"] = float(skeleton_coverage)
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "hidden_dim": config.hidden_dim,
                        "n_features": config.n_features,
                        "supports_multivariate_formulas": config.supports_multivariate_formulas,
                        "multivariate_neural_mode": config.multivariate_neural_mode,
                        "proposer_contract_version": config.proposer_contract_version,
                        "max_input_vars": config.max_input_vars,
                        "operator_vocab": model.operator_vocab,
                        "skeleton_vocab": model.skeleton_vocab,
                        "architecture_version": UNIVERSAL_PROPOSER_ARCHITECTURE_VERSION,
                    },
                    "architecture_version": UNIVERSAL_PROPOSER_ARCHITECTURE_VERSION,
                    "proposer_contract_version": UNIVERSAL_PROPOSER_CONTRACT_VERSION,
                    "proposer_role": UNIVERSAL_PROPOSER_ROLE,
                    "routing_calibration": {
                        "status": "uncalibrated",
                        "method": "candidate_mse_gate_plus_validation_gated_skeleton_confidence",
                        "requires": "downstream_candidate_success_benchmark",
                    },
                    "feature_scaler": weights_only_safe_scaler(feature_scaler),
                    "epoch": epoch,
                    "val_f1": best_f1,
                    "selection_metric": best_selection_metric,
                    "selection_metric_name": best_metrics["selection_metric_name"],
                    "val_micro_f1": val_metrics.get("micro_f1"),
                    "validation_split_policy": split_policy,
                    "validation_split_details": dict(split_details),
                    "validation_metrics": best_metrics,
                    "operator_pos_weight": (
                        operator_pos_weight.detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                        .tolist()
                        if operator_pos_weight is not None
                        else None
                    ),
                    "skeleton_loss_weight": float(skeleton_loss_weight),
                    "train_skeleton_coverage": float(skeleton_coverage),
                },
                out_path,
            )
            print(
                f"  -> Saved best model (select={selection_metric:.4f}, val_f1={val_f1:.4f})"
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(
                    f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)"
                )
                break

    checkpoint = torch.load(out_path, weights_only=False)
    best_checkpoint_metrics = dict(checkpoint.get("validation_metrics") or best_metrics)
    best_checkpoint_metrics.setdefault("val_f1", checkpoint.get("val_f1"))
    best_checkpoint_metrics.setdefault("val_micro_f1", checkpoint.get("val_micro_f1"))
    if validation_report is not None:
        validation_report["metrics"] = {
            "best_checkpoint": dict(best_checkpoint_metrics)
        }
        report_path = (
            Path(args.validation_report)
            if args.validation_report
            else default_validation_report_path(out_path)
        )
        write_validation_report(report_path, validation_report)
        checkpoint["validation_report_path"] = str(report_path)
        torch.save(checkpoint, out_path)
        print(f"Validation report saved to {report_path}")

    validation_report_for_card = validation_report or {
        "split_policy": split_policy,
        "split_details": dict(split_details),
        "metrics": {"best_checkpoint": dict(best_checkpoint_metrics)},
        "notes": [
            "Synthetic proposer training run; production rollout requires dataset-backed grouped validation."
        ],
    }
    checkpoint["labeler_version"] = dataset_metadata.get("labeler_version")
    checkpoint["data_generation_command"] = (
        args.data_generation_command or "not_provided"
    )
    checkpoint_card = build_checkpoint_card(
        model_kind="universal_proposer",
        checkpoint_path=out_path,
        validation_report=validation_report_for_card,
        checkpoint_metadata=checkpoint,
        data_generation_command=args.data_generation_command,
        training_command=" ".join(sys.argv),
        runtime_contract={
            "role": UNIVERSAL_PROPOSER_ROLE,
            "contract_version": UNIVERSAL_PROPOSER_CONTRACT_VERSION,
            "candidate_generation": "grammar_decode_with_mse_ranking",
        },
    )
    checkpoint_card_path = (
        Path(args.checkpoint_card)
        if args.checkpoint_card
        else default_checkpoint_card_path(out_path)
    )
    write_checkpoint_card(checkpoint_card_path, checkpoint_card)
    checkpoint["checkpoint_card_path"] = str(checkpoint_card_path)

    if args.baseline_card:
        baseline_card_path = Path(args.baseline_card)
        if baseline_card_path.exists():
            baseline_card = load_json_report(baseline_card_path)
            rollout_comparison = build_rollout_comparison(
                candidate_card=checkpoint_card,
                baseline_card=baseline_card,
                metric_name=args.rollout_metric,
                min_relative_improvement=args.min_relative_improvement,
            )
            rollout_comparison_path = (
                Path(args.rollout_comparison)
                if args.rollout_comparison
                else default_rollout_comparison_path(out_path)
            )
            write_rollout_comparison(rollout_comparison_path, rollout_comparison)
            checkpoint["rollout_comparison_path"] = str(rollout_comparison_path)
        else:
            print(
                f"Warning: baseline card not found at {baseline_card_path}; "
                "skipping rollout comparison."
            )

    torch.save(checkpoint, out_path)

    print(f"\nTraining complete. Best Val F1: {best_f1:.4f}. Model saved to {out_path}")
    print(f"Checkpoint card saved to {checkpoint_card_path}")
    if checkpoint.get("rollout_comparison_path"):
        print(f"Rollout comparison saved to {checkpoint['rollout_comparison_path']}")


if __name__ == "__main__":
    main()
