"""
Curve Classifier Training Script

Trains a neural network to predict which mathematical operators are present
in a curve based on its features.

Usage:
    python scripts/train_curve_classifier.py --data data/curve_dataset_10k.npz --epochs 50
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

# §3.11: SymLog column range derived from the shared feature schema instead
# of repeated 192:398 literals (schema drift would silently desync data
# loading from training statistics). Falls back to literals if the schema
# module is unavailable.
try:
    from glassbox.curve_classifier.generate_curve_data import FEATURE_SCHEMA as _FEATURE_SCHEMA

    SYMLOG_START = int(_FEATURE_SCHEMA["deriv"][0])
    SYMLOG_END = int(_FEATURE_SCHEMA["invariants"][1])
except Exception:
    SYMLOG_START = 192
    SYMLOG_END = 398

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# =============================================================================
# MODEL ARCHITECTURES
# =============================================================================


class CurveClassifierMLP(nn.Module):
    """Deep MLP classifier for curve features."""

    def __init__(self, n_features: int = 398, n_classes: int = 9, hidden: int = 512):
        super().__init__()

        eql_out_dim = 256
        self.eql = EQLLayer(in_features=n_features, out_features=eql_out_dim)

        layers = []
        combined_dim = n_features + eql_out_dim

        layers.extend(
            [
                nn.Linear(combined_dim, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Dropout(0.2),
            ]
        )

        for _ in range(6):
            layers.extend(
                [
                    nn.Linear(hidden, hidden),
                    nn.BatchNorm1d(hidden),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                ]
            )

        layers.extend(
            [
                nn.Linear(hidden, hidden // 2),
                nn.BatchNorm1d(hidden // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden // 2, n_classes),
            ]
        )

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        eql_feats = self.eql(x)
        combined = torch.cat([x, eql_feats], dim=1)
        return self.net(combined)


class CurveClassifierCNN(nn.Module):
    """1D CNN that operates on the raw curve portion of features."""

    def __init__(self, n_classes: int = 9, n_features: int = 398, curve_dim: int = 128):
        super().__init__()

        self.curve_dim = min(curve_dim, n_features)

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

        other_dim = max(1, n_features - self.curve_dim)
        self.other_mlp = nn.Sequential(
            nn.Linear(other_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        raw_curve = x[:, : self.curve_dim]
        other_features = x[:, self.curve_dim :]

        raw_curve = raw_curve.unsqueeze(1)
        conv_out = self.conv(raw_curve)
        conv_out = conv_out.flatten(1)

        other_out = self.other_mlp(other_features)

        combined = torch.cat([conv_out, other_out], dim=1)
        return self.classifier(combined)


class SemanticFeatureAttention(nn.Module):
    """
    Semantic Attention that treats each feature group as a distinct token.
    Allows the model to attend across different modalities (FFT, derivatives, stats, etc.).
    """

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim

        # Project each semantic group independently
        self.proj_raw = nn.Linear(128, embed_dim)
        self.proj_fft = nn.Linear(32, embed_dim)
        self.proj_fft_phase = nn.Linear(32, embed_dim)
        self.proj_deriv = nn.Linear(128, embed_dim)
        self.proj_stats = nn.Linear(9, embed_dim)
        self.proj_curv = nn.Linear(37, embed_dim)
        self.proj_invars = nn.Linear(32, embed_dim)

        # 7 feature tokens + 1 CLS token
        self.n_tokens = 8
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Token type embeddings (tells attention which modality is which)
        self.token_type_embed = nn.Parameter(
            torch.randn(1, self.n_tokens, embed_dim) * 0.02
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=4, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.dropout = nn.Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x):
        b = x.size(0)

        # Slice based on FEATURE_SCHEMA from generate_curve_data.py
        # Fallbacks added in case feature dimension doesn't perfectly match
        raw = x[:, 0:128]
        fft = x[:, 128:160]
        fft_phase = x[:, 160:192]
        deriv = x[:, 192:320]
        stats = x[:, 320:329]
        curv = x[:, 329:366]

        # Handle cases where feature dimension might be slightly different
        if x.shape[1] > 366:
            invars = x[:, 366:398]
        else:
            invars = torch.zeros(b, 32, device=x.device, dtype=x.dtype)

        # Create tokens
        t_raw = self.proj_raw(raw).unsqueeze(1)
        t_fft = self.proj_fft(fft).unsqueeze(1)
        t_fft_phase = self.proj_fft_phase(fft_phase).unsqueeze(1)
        t_deriv = self.proj_deriv(deriv).unsqueeze(1)
        t_stats = self.proj_stats(stats).unsqueeze(1)
        t_curv = self.proj_curv(curv).unsqueeze(1)
        t_invars = self.proj_invars(invars).unsqueeze(1)

        cls_tokens = self.cls_token.expand(b, -1, -1)

        # Sequence of 8 tokens
        tokens = torch.cat(
            [cls_tokens, t_raw, t_fft, t_fft_phase, t_deriv, t_stats, t_curv, t_invars],
            dim=1,
        )

        tokens = tokens + self.token_type_embed
        tokens = self.dropout(tokens)

        # Attention
        attn_out, _ = self.attention(tokens, tokens, tokens)
        tokens = self.norm1(tokens + attn_out)

        # FFN
        ffn_out = self.ffn(tokens)
        tokens = self.norm2(tokens + ffn_out)

        # Flatten all tokens to maintain the 1024-dim output expected
        return tokens.flatten(1)


class EQLLayer(nn.Module):
    """
    Equation Learner (EQL) Layer.
    Applies explicit mathematical transformations to the input to act as a
    'cheat sheet' for the network to detect mathematical operators.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        # 6 explicit mathematical functions
        self.n_funcs = 6
        self.features_per_func = out_features // self.n_funcs
        self.rem_features = out_features % self.n_funcs

        # Project input features to the space where functions will be applied
        self.linear = nn.Linear(in_features, out_features)

        # Initialize weights to be small to prevent extreme values going into exp/log early on
        nn.init.xavier_normal_(self.linear.weight, gain=0.1)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        z = self.linear(x)

        out = []
        start_idx = 0

        for i in range(self.n_funcs):
            end_idx = (
                start_idx
                + self.features_per_func
                + (self.rem_features if i == 0 else 0)
            )
            chunk = z[:, start_idx:end_idx]

            if i == 0:
                # Identity
                out.append(chunk)
            elif i == 1:
                # Sin
                out.append(torch.sin(chunk))
            elif i == 2:
                # Cos
                out.append(torch.cos(chunk))
            elif i == 3:
                # Exp (clamped to prevent inf/nan)
                out.append(torch.exp(torch.clamp(chunk, min=-10.0, max=10.0)))
            elif i == 4:
                # Log (safe log)
                out.append(torch.log(torch.abs(chunk) + 1e-6))
            elif i == 5:
                # Square
                out.append(torch.square(chunk))

            start_idx = end_idx

        return torch.cat(out, dim=1)


class CurveClassifierGLU(nn.Module):
    """
    First-Principles Mathematical Classifier using Gated Linear Units (GLU).
    Mathematically models multiplicative function composition (e.g. x * sin(x)) natively.
    """

    def __init__(self, n_features: int = 398, n_classes: int = 9, hidden: int = 512):
        super().__init__()

        # 1. Semantic Feature Attention
        n_tokens = 8
        embed_dim = 128
        self.attn = SemanticFeatureAttention(embed_dim=embed_dim)
        attn_out_dim = n_tokens * embed_dim

        # 2. EQL Layer
        eql_out_dim = 256
        self.eql = EQLLayer(in_features=n_features, out_features=eql_out_dim)

        # 3. Combine outputs
        combined_dim = attn_out_dim + eql_out_dim

        self.fc1 = nn.Linear(combined_dim, hidden * 2)
        self.bn1 = nn.BatchNorm1d(hidden * 2)

        self.fc2 = nn.Linear(hidden, hidden * 2)
        self.bn2 = nn.BatchNorm1d(hidden * 2)

        self.fc3 = nn.Linear(hidden, hidden * 2)
        self.bn3 = nn.BatchNorm1d(hidden * 2)

        self.fc4 = nn.Linear(hidden, hidden * 2)
        self.bn4 = nn.BatchNorm1d(hidden * 2)

        self.classifier = nn.Linear(hidden, n_classes)
        self.dropout = nn.Dropout(0.2)

        self._init_weights()

    def _init_weights(self):
        """Hardware-sympathetic initialization for multiplicative gating."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        # Extract abstract semantic tokens
        attn_features = self.attn(x)

        # Extract explicit mathematical transformations
        eql_features = self.eql(x)

        # Combine
        x = torch.cat([attn_features, eql_features], dim=1)

        x = self.fc1(x)
        x = self.bn1(x)
        x = F.glu(x, dim=1)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.glu(x, dim=1)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = F.glu(x, dim=1)
        x = self.dropout(x)

        x = self.fc4(x)
        x = self.bn4(x)
        x = F.glu(x, dim=1)
        x = self.dropout(x)

        return self.classifier(x)


try:
    from .models import (
        CURVE_CLASSIFIER_ARCHITECTURE_VERSION,
    )
    from .models import (
        CurveClassifierCNN as CurveClassifierCNN,
    )
    from .models import (
        CurveClassifierGLU as CurveClassifierGLU,
    )
    from .models import (
        CurveClassifierMLP as CurveClassifierMLP,
    )
    from .models import (
        EQLLayer as EQLLayer,
    )
    from .models import (
        SemanticFeatureAttention as SemanticFeatureAttention,
    )
    from .rollout import (
        build_checkpoint_card,
        build_rollout_comparison,
        default_checkpoint_card_path,
        default_rollout_comparison_path,
        load_json_report,
        write_checkpoint_card,
        write_rollout_comparison,
    )
    from .validation import (
        build_validation_report,
        default_validation_report_path,
        family_holdout_split,
        formula_keys_from_metadata_or_formulas,
        grouped_train_val_split,
        multilabel_metrics_by_group,
        row_train_val_split,
        write_validation_report,
    )
except (ImportError, ValueError):
    from glassbox.curve_classifier.models import (
        CURVE_CLASSIFIER_ARCHITECTURE_VERSION,
    )
    from glassbox.curve_classifier.models import (
        CurveClassifierCNN as CurveClassifierCNN,
    )
    from glassbox.curve_classifier.models import (
        CurveClassifierGLU as CurveClassifierGLU,
    )
    from glassbox.curve_classifier.models import (
        CurveClassifierMLP as CurveClassifierMLP,
    )
    from glassbox.curve_classifier.models import (
        EQLLayer as EQLLayer,
    )
    from glassbox.curve_classifier.models import (
        SemanticFeatureAttention as SemanticFeatureAttention,
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
        multilabel_metrics_by_group,
        row_train_val_split,
        write_validation_report,
    )


class IndexedFeatureDataset(Dataset):
    """Dataset view over feature/label arrays using explicit indices.
    Supports pre-loading entire dataset into RAM or VRAM for maximum throughput.
    """

    def __init__(
        self,
        features: np.ndarray | torch.Tensor,
        labels: np.ndarray | torch.Tensor,
        indices: np.ndarray,
        scaler: dict | None = None,
        device: torch.device | None = None,
    ):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.scaler = scaler

        # Determine if data is already on target device
        self.is_on_device = False

        if device is not None and device.type == "cuda":
            print(f"Transferring dataset split to {device}...")
            # We slice the required features/labels first to save memory before moving to GPU
            x_sliced = np.asarray(features[self.indices], dtype=np.float32)

            # Apply SymLog compression selectively to non-raw/fft features
            if x_sliced.shape[1] < SYMLOG_END:
                raise ValueError(
                    f"feature width {x_sliced.shape[1]} < SYMLOG_END {SYMLOG_END}")
            x_sliced[:, SYMLOG_START:SYMLOG_END] = np.sign(x_sliced[:, SYMLOG_START:SYMLOG_END]) * np.log1p(
                np.abs(x_sliced[:, SYMLOG_START:SYMLOG_END])
            )
            if self.scaler is not None:
                x_sliced = (x_sliced - self.scaler["mean"]) / (
                    self.scaler["std"] + 1e-8
                )
            y_sliced = np.asarray(labels[self.indices], dtype=np.float32)

            # Store directly as tensors on device. Reset indices mapping since we sliced.
            self.features = torch.from_numpy(x_sliced).to(device)
            self.labels = torch.from_numpy(y_sliced).to(device)
            # Re-map indices to 0...N since we stored the sliced array
            self.indices = np.arange(len(self.indices), dtype=np.int64)
            self.is_on_device = True
        else:
            self.features = features
            self.labels = labels

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        sample_idx = int(self.indices[idx])

        if self.is_on_device:
            # Zero-copy, already scaled and on GPU
            return self.features[sample_idx], self.labels[sample_idx]

        # CPU/Memmap path
        # np.asarray() can return a writable view into the backing ndarray/memmap.
        # The feature transform below must be per-read only; mutating here would
        # repeatedly log-compress columns 192:398 across epochs.
        x = np.array(self.features[sample_idx], dtype=np.float32, copy=True)

        # Apply SymLog compression selectively to non-raw/fft features
        if x.shape[0] < SYMLOG_END:
            raise ValueError(
                f"feature width {x.shape[0]} < SYMLOG_END {SYMLOG_END}")
        x[SYMLOG_START:SYMLOG_END] = np.sign(x[SYMLOG_START:SYMLOG_END]) * np.log1p(np.abs(x[SYMLOG_START:SYMLOG_END]))

        if self.scaler is not None:
            x = (x - self.scaler["mean"]) / (self.scaler["std"] + 1e-8)
        y = np.asarray(self.labels[sample_idx], dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


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

        # Apply SymLog compression selectively to non-raw/fft features
        if batch.shape[1] < SYMLOG_END:
            raise ValueError(
                f"feature width {batch.shape[1]} < SYMLOG_END {SYMLOG_END}")
        batch[:, SYMLOG_START:SYMLOG_END] = np.sign(batch[:, SYMLOG_START:SYMLOG_END]) * np.log1p(
            np.abs(batch[:, SYMLOG_START:SYMLOG_END])
        )

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


def train_epoch(
    model, dataloader, optimizer, criterion, device, scaler, max_grad_norm: float = 1.0
):
    """Train for one epoch with gradient clipping and AMP scaling."""
    model.train()

    # Fast-path for VRAM-resident datasets (Bypasses Python DataLoader overhead)
    ds = dataloader.dataset
    if hasattr(ds, "is_on_device") and ds.is_on_device and device.type == "cuda":
        total_loss = torch.zeros(1, device=device)
        n_samples = len(ds)
        batch_size = dataloader.batch_size

        # Fast GPU-side shuffle
        indices = torch.randperm(n_samples, device=device)

        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_idx = indices[start_idx:end_idx]

            x_batch = ds.features[batch_idx]
            y_batch = ds.labels[batch_idx]

            optimizer.zero_grad(set_to_none=True)

            # Use AMP if scaler is present
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits = model(x_batch)
                    loss = criterion(logits, y_batch)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x_batch)
                loss = criterion(logits, y_batch)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=max_grad_norm
                )
                optimizer.step()

            total_loss += loss.detach() * (end_idx - start_idx)

        return float((total_loss / max(n_samples, 1)).item())

    # Standard path for RAM/Disk loaded datasets
    total_loss = torch.zeros(1, device=device)
    n_batches = 0

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # No AMP - Use FP32
        if True:
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

        # Accumulate without .item() to keep the CPU/GPU working in parallel
        total_loss += loss.detach()
        n_batches += 1

    return float((total_loss / max(n_batches, 1)).item())


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
    all_preds = []
    all_labels = []
    all_logits = []

    ds = dataloader.dataset
    if hasattr(ds, "is_on_device") and ds.is_on_device and device.type == "cuda":
        total_loss = 0.0
        n_samples = len(ds)
        batch_size = dataloader.batch_size

        with torch.no_grad():
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)

                x_batch = ds.features[start_idx:end_idx]
                y_batch = ds.labels[start_idx:end_idx]

                # No AMP - Use FP32
                logits = model(x_batch)
                loss = criterion(logits, y_batch)

                total_loss += loss.item() * (end_idx - start_idx)

                if temperature is None:
                    preds = torch.sigmoid(logits)
                else:
                    preds = torch.sigmoid(logits / temperature)

                all_preds.append(preds.cpu())
                all_labels.append(y_batch.cpu())
                if return_logits:
                    all_logits.append(logits.cpu())

        avg_loss = total_loss / max(n_samples, 1)
    else:
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)

                # No AMP - Use FP32
                if True:
                    logits = model(x_batch)
                    loss = criterion(logits, y_batch)
                total_loss += loss.item() * x_batch.shape[0]
                total_samples += x_batch.shape[0]

                if temperature is None:
                    preds = torch.sigmoid(logits)
                else:
                    preds = torch.sigmoid(logits / temperature)

                all_preds.append(preds.cpu())
                all_labels.append(y_batch.cpu())
                if return_logits:
                    all_logits.append(logits.cpu())

        avg_loss = total_loss / max(total_samples, 1)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Per-class accuracy (threshold = 0.5 or tuned thresholds)
    if thresholds is None:
        binary_preds = (all_preds > 0.5).float()
    else:
        thresholds_cpu = (
            thresholds.detach().cpu()
            if isinstance(thresholds, torch.Tensor)
            else torch.as_tensor(thresholds)
        )
        binary_preds = (all_preds > thresholds_cpu).float()
    per_class_acc = (binary_preds == all_labels).float().mean(dim=0)
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
        "loss": avg_loss,
        "accuracy": overall_acc.item(),
        "per_class_acc": per_class_acc.numpy(),
        "f1_mean": f1.mean().item(),
        "micro_f1": micro_f1.item(),
        "precision_per_class": precision.numpy(),
        "recall_per_class": recall.numpy(),
        "f1_per_class": f1.numpy(),
    }

    if return_preds:
        metrics["preds"] = all_preds
        metrics["labels"] = all_labels
    if return_logits:
        metrics["logits"] = torch.cat(all_logits)

    return metrics


def calibrate_temperature(
    logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50
) -> float:
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
            calibration_maps.append(
                {
                    "boundaries": bin_edges.tolist(),
                    "values": bin_centers.tolist(),
                }
            )
            continue

        # Fit isotonic regression
        ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        ir.fit(raw_probs, true_labels)

        # Discretize into bins for compact storage in checkpoint
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        calibrated_values = ir.predict(bin_centers)

        calibration_maps.append(
            {
                "boundaries": bin_edges.tolist(),
                "values": calibrated_values.tolist(),
            }
        )

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
        boundaries = np.array(cmap["boundaries"])
        values = np.array(cmap["values"])
        # np.digitize returns index i such that boundaries[i-1] <= x < boundaries[i]
        indices = np.digitize(raw_probs[:, c], boundaries, right=False) - 1
        indices = np.clip(indices, 0, len(values) - 1)
        calibrated[:, c] = values[indices]

    if single:
        return calibrated[0]
    return calibrated


def tune_thresholds(
    all_preds: torch.Tensor,
    all_labels: torch.Tensor,
    steps: int = 19,
    beta: float = 1.0,
) -> torch.Tensor:
    """Tune per-class thresholds to maximize F-beta on calibration data."""
    thresholds = torch.full((all_labels.shape[1],), 0.5, dtype=torch.float32)
    candidates = torch.linspace(0.05, 0.95, steps)
    beta_sq = float(beta) ** 2

    for c in range(all_labels.shape[1]):
        best_score = -1.0
        best_t = 0.5
        for t in candidates:
            preds = (all_preds[:, c] > t).float()
            labels = all_labels[:, c]
            tp = ((preds == 1) & (labels == 1)).float().sum()
            fp = ((preds == 1) & (labels == 0)).float().sum()
            fn = ((preds == 0) & (labels == 1)).float().sum()
            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            score = (
                (1.0 + beta_sq)
                * precision
                * recall
                / (beta_sq * precision + recall + 1e-10)
            )
            if score > best_score:
                best_score = score
                best_t = float(t)
        thresholds[c] = best_t

    return thresholds


def multilabel_stratified_split(
    labels: np.ndarray, val_ratio: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate multi-label stratified split without external dependencies."""
    rng = np.random.RandomState(seed)
    n_samples = labels.shape[0]
    n_val = int(n_samples * val_ratio)
    if n_val < 1 or n_samples - n_val < 1:
        raise ValueError(
            f"val_ratio={val_ratio} creates train={n_samples - n_val} val={n_val}; "
            "both splits must contain at least one sample."
        )

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


def split_validation_calibration(
    val_idx: np.ndarray,
    labels: np.ndarray,
    calibration_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, object]]:
    """Split validation rows into report and calibration/threshold rows."""
    val_idx = np.asarray(val_idx, dtype=np.int64)
    calibration_ratio = float(calibration_ratio)
    if calibration_ratio <= 0.0 or len(val_idx) < 4:
        return (
            val_idx,
            None,
            {
                "calibration_split": False,
                "reason": "disabled_or_too_small",
            },
        )

    local_train, local_cal = multilabel_stratified_split(
        np.asarray(labels[val_idx], dtype=np.float32),
        calibration_ratio,
        seed,
    )
    eval_idx = val_idx[local_train]
    cal_idx = val_idx[local_cal]
    return (
        eval_idx,
        cal_idx,
        {
            "calibration_split": True,
            "calibration_ratio": calibration_ratio,
            "eval_rows": len(eval_idx),
            "calibration_rows": len(cal_idx),
        },
    )


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
    threshold_beta: float = 0.5,
    calibrate_flag: bool = False,
    class_weights: torch.Tensor | None = None,
    calibration_loader=None,
    val_groups: np.ndarray | None = None,
):
    """Full training loop with early stopping."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler_mode = "max" if early_stop_metric == "f1" else "min"
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=scheduler_mode, factor=0.5, patience=5
    )

    # Use class weights for imbalanced labels if provided
    if class_weights is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))
        print(f"Using class weights: {class_weights.numpy()}")
    else:
        criterion = nn.BCEWithLogitsLoss()

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_val_loss = float("inf")
    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        try:
            train_loss = train_epoch(
                model, train_loader, optimizer, criterion, device, scaler
            )
        except Exception as e:
            # Handle lazy compilation failures (common on Windows)
            if "inductor" in str(e).lower() and hasattr(model, "_orig_mod"):
                print(f"\n[!] torch.compile failed during first forward pass: {e}")
                print("[!] Falling back to eager mode for the rest of training.")
                model = model._orig_mod
                train_loss = train_epoch(
                    model, train_loader, optimizer, criterion, device, scaler
                )
            else:
                raise e

        # Evaluate
        val_metrics = evaluate(model, val_loader, criterion, device)

        # Scheduler step
        scheduler_value = (
            val_metrics["f1_mean"] if early_stop_metric == "f1" else val_metrics["loss"]
        )
        scheduler.step(scheduler_value)

        # Logging
        print(
            f"Epoch {epoch + 1:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['f1_mean']:.4f} | "
            f"Val Micro-F1: {val_metrics['micro_f1']:.4f}"
        )

        # Save best model
        if early_stop_metric == "f1":
            is_best = val_metrics["f1_mean"] > best_val_f1
        else:
            is_best = val_metrics["loss"] < best_val_loss

        if is_best:
            best_val_loss = val_metrics["loss"]
            best_val_f1 = val_metrics["f1_mean"]
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_metrics["loss"],
                    "val_acc": val_metrics["accuracy"],
                    "val_f1": val_metrics["f1_mean"],
                    "val_micro_f1": val_metrics["micro_f1"],
                    "val_precision_per_class": val_metrics["precision_per_class"],
                    "val_recall_per_class": val_metrics["recall_per_class"],
                    "val_f1_per_class": val_metrics["f1_per_class"],
                    "val_per_class_acc": val_metrics["per_class_acc"],
                    "operator_classes": operator_classes,
                    "model_type": model_type,
                    "model_config": {
                        **model_config,
                        "architecture_version": CURVE_CLASSIFIER_ARCHITECTURE_VERSION,
                    },
                    "architecture_version": CURVE_CLASSIFIER_ARCHITECTURE_VERSION,
                },
                save_path,
            )
            print(
                f"  -> Saved best model (val_loss: {val_metrics['loss']:.4f}, val_f1: {val_metrics['f1_mean']:.4f})"
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"\nEarly stopping at epoch {epoch + 1} (no improvement for {patience} epochs)"
                )
                break

    print(
        f"\nBest model at epoch {best_epoch} with val_loss: {best_val_loss:.4f}, val_f1: {best_val_f1:.4f}"
    )

    # Reload best model for final evaluation
    checkpoint = torch.load(save_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_metrics = evaluate(
        model, val_loader, criterion, device, return_preds=True, return_logits=True
    )
    if val_groups is not None:
        checkpoint["val_metrics_by_family"] = multilabel_metrics_by_group(
            val_metrics["preds"].detach().cpu().numpy(),
            val_metrics["labels"].detach().cpu().numpy(),
            val_groups,
            operator_classes,
        )
    calibration_metrics = (
        evaluate(
            model,
            calibration_loader,
            criterion,
            device,
            return_preds=True,
            return_logits=True,
        )
        if calibration_loader is not None
        else val_metrics
    )

    # Final per-class report using best model
    print("\nPer-class F1 scores (best model):")
    for i, name in enumerate(operator_classes):
        print(f"  {name:15s}: {val_metrics['f1_per_class'][i]:.4f}")

    # Tune thresholds and store alongside model checkpoint
    checkpoint = torch.load(save_path, weights_only=False)

    # Optional calibration
    temperature = None
    isotonic_maps = []
    if calibrate_flag:
        temperature = calibrate_temperature(
            calibration_metrics["logits"], calibration_metrics["labels"]
        )
        checkpoint["temperature"] = temperature
        print(f"\nCalibrated temperature saved to checkpoint: {temperature:.4f}")

        # Isotonic per-class calibration (more expressive than temperature scaling)
        isotonic_maps = calibrate_isotonic_per_class(
            calibration_metrics["logits"],
            calibration_metrics["labels"],
            temperature=temperature,
        )
        if isotonic_maps:
            checkpoint["isotonic_calibration"] = isotonic_maps
            print(
                f"Per-class isotonic calibration maps saved ({len(isotonic_maps)} classes)"
            )

    # Threshold tuning (optionally on calibrated probabilities)
    if tune_thresholds_flag:
        preds_for_tuning = calibration_metrics["preds"]
        if temperature is not None:
            preds_for_tuning = torch.sigmoid(
                calibration_metrics["logits"] / temperature
            )
        if isotonic_maps:
            calibrated_np = apply_isotonic_calibration(
                preds_for_tuning.detach().cpu().numpy(),
                isotonic_maps,
            )
            preds_for_tuning = torch.from_numpy(calibrated_np).to(
                preds_for_tuning.dtype
            )
        thresholds = tune_thresholds(
            preds_for_tuning,
            calibration_metrics["labels"],
            beta=threshold_beta,
        )
        checkpoint["thresholds"] = thresholds.numpy()
        checkpoint["threshold_beta"] = float(threshold_beta)
        thresholded_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            thresholds=thresholds.to(device),
            temperature=temperature,
        )
        checkpoint["thresholded_val_f1"] = thresholded_metrics["f1_mean"]
        checkpoint["thresholded_val_micro_f1"] = thresholded_metrics["micro_f1"]
        checkpoint["thresholded_precision_per_class"] = thresholded_metrics[
            "precision_per_class"
        ]
        checkpoint["thresholded_recall_per_class"] = thresholded_metrics[
            "recall_per_class"
        ]
        checkpoint["thresholded_f1_per_class"] = thresholded_metrics["f1_per_class"]
        print(
            f"\nTuned per-class thresholds saved to checkpoint (F-beta beta={threshold_beta:.2f})"
        )

    torch.save(checkpoint, save_path)

    return model


def load_training_data(
    data_args: list[str],
    n_samples: int | None,
    feature_dim: int,
    n_classes: int,
    load_into_ram: bool,
    return_metadata: bool = False,
):
    """Load training data from .npz or streamed .dat files."""
    dataset_metadata: dict[str, object] = {}
    # Case 1: single .npz file
    if len(data_args) == 1 and data_args[0].endswith(".npz"):
        data = np.load(data_args[0], allow_pickle=True)
        features = data["features"]
        labels = data["labels"]
        operator_classes = data["operator_classes"].tolist()
        detected_feature_dim = (
            int(data["feature_dim"]) if "feature_dim" in data else features.shape[1]
        )
        feature_schema = (
            data["feature_schema"].item() if "feature_schema" in data else None
        )
        formulas = data["formulas"].tolist() if "formulas" in data else None
        n_loaded = int(features.shape[0])
        dataset_metadata = {
            "dataset_path": str(data_args[0]),
            "formulas": formulas,
            "formula_keys": formula_keys_from_metadata_or_formulas(
                data["formula_keys"] if "formula_keys" in data else None,
                formulas,
                limit=n_loaded,
            ),
            "generator_families": (
                np.asarray(data["generator_families"][:n_loaded], dtype=object)
                if "generator_families" in data
                else None
            ),
            "template_ids": (
                np.asarray(data["template_ids"][:n_loaded], dtype=object)
                if "template_ids" in data
                else None
            ),
            "labeler_version": str(data["labeler_version"])
            if "labeler_version" in data
            else None,
            "labels_match_semantic": (
                np.asarray(data["labels_match_semantic"][:n_loaded], dtype=bool)
                if "labels_match_semantic" in data
                else None
            ),
        }
        if return_metadata:
            return (
                features,
                labels,
                operator_classes,
                detected_feature_dim,
                feature_schema,
                dataset_metadata,
            )
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

    if (
        features_path is None
        or labels_path is None
        or not features_path.exists()
        or not labels_path.exists()
    ):
        raise FileNotFoundError(
            "Expected either a .npz file or .features.dat and .labels.dat files."
        )

    if n_samples is None:
        file_size = features_path.stat().st_size
        n_samples = file_size // (feature_dim * 4)
        print(f"Inferred n_samples={n_samples} from {features_path.name}")

    features = np.memmap(
        features_path, dtype=np.float32, mode="r", shape=(n_samples, feature_dim)
    )
    labels = np.memmap(
        labels_path, dtype=np.float32, mode="r", shape=(n_samples, n_classes)
    )

    if load_into_ram:
        print("Loading features into RAM...")
        features = np.array(features)
        print(f"  Features loaded: {features.nbytes / 1e9:.2f} GB")
        print("Loading labels into RAM...")
        labels = np.array(labels)

    operator_classes = [
        "identity",
        "sin",
        "cos",
        "power",
        "exp",
        "log",
        "addition",
        "multiplication",
        "rational",
    ][:n_classes]
    dataset_metadata = {"dataset_path": str(data_args[0]) if data_args else None}
    if return_metadata:
        return features, labels, operator_classes, feature_dim, None, dataset_metadata
    return features, labels, operator_classes, feature_dim, None


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Train curve classifier")
    parser.add_argument(
        "--data",
        type=str,
        nargs="+",
        required=True,
        help="Path to training data (.npz file) or base path / .features.dat + .labels.dat",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Number of samples (required for .dat if file size cannot be inferred)",
    )
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=398,
        help="Feature dimension for .dat files (default: 398)",
    )
    parser.add_argument(
        "--n-classes",
        type=int,
        default=9,
        help="Number of classes for .dat files (default: 9)",
    )
    parser.add_argument(
        "--load-into-ram",
        "--load-into-vram",
        dest="load_into_ram",
        action="store_true",
        help="Load dataset fully into RAM/VRAM for maximum throughput",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="glu",
        choices=["glu", "mlp", "cnn"],
        help="Model architecture",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=8192, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--hidden", type=int, default=512, help="Hidden layer size (default: 512)"
    )
    parser.add_argument(
        "--val-split", type=float, default=0.1, help="Validation split ratio"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/curve_classifier.pt",
        help="Output model path",
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device (auto, cpu, cuda)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )
    parser.add_argument(
        "--early-stop",
        type=str,
        default="f1",
        choices=["loss", "f1"],
        help="Early stopping metric",
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Standardize features using training set statistics (default: on)",
    )
    parser.add_argument(
        "--no-standardize", action="store_true", help="Disable feature standardization"
    )
    parser.add_argument(
        "--tune-thresholds",
        action="store_true",
        help="Tune per-class thresholds on validation set (default: on)",
    )
    parser.add_argument(
        "--no-tune-thresholds",
        action="store_true",
        help="Disable per-class threshold tuning",
    )
    parser.add_argument(
        "--threshold-beta",
        type=float,
        default=0.5,
        help="F-beta used for threshold tuning; beta<1 favors precision",
    )
    parser.add_argument(
        "--stratified-split",
        action="store_true",
        help="Use approximate multi-label stratified train/val split (default: on)",
    )
    parser.add_argument(
        "--no-stratified-split", action="store_true", help="Disable stratified split"
    )
    parser.add_argument(
        "--split-policy",
        type=str,
        default="auto",
        choices=["auto", "row", "stratified", "formula_group", "family_holdout"],
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
        "--calibrate",
        action="store_true",
        help="Calibrate probabilities with temperature scaling",
    )
    parser.add_argument(
        "--calibration-split",
        type=float,
        default=0.25,
        help="Fraction of validation rows reserved for calibration/threshold tuning",
    )
    parser.add_argument(
        "--class-weights",
        action="store_true",
        help="Use inverse frequency class weights for imbalanced labels",
    )
    parser.add_argument(
        "--class-weight-cap",
        type=float,
        default=3.0,
        help="Maximum positive class weight when --class-weights is enabled",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile (PyTorch 2.0+) for kernel fusion",
    )

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

    # Enable TF32 for better performance on Ampere+ GPUs
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    # Load data
    print(f"Loading data from {args.data}...")
    (
        features,
        labels,
        operator_classes,
        feature_dim,
        feature_schema,
        dataset_metadata,
    ) = load_training_data(
        args.data,
        args.n_samples,
        args.feature_dim,
        args.n_classes,
        args.load_into_ram,
        return_metadata=True,
    )

    print(f"  Features: {features.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Classes: {operator_classes}")

    expected_features = feature_dim or 366
    if features.shape[1] != expected_features:
        print(
            f"Warning: expected {expected_features} features, got {features.shape[1]}. "
        )
    if feature_schema is not None:
        print(f"  Feature schema: {feature_schema}")

    formula_keys = dataset_metadata.get("formula_keys")
    generator_families = dataset_metadata.get("generator_families")
    template_ids = dataset_metadata.get("template_ids")
    split_details = {}
    split_policy = args.split_policy

    # Train/val split. Phase 3 defaults dataset-backed training to formula
    # groups when metadata is present so checkpoint metrics are not row-leaky.
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
            if not family_counts:
                raise ValueError(
                    "No generator families available for family_holdout split"
                )
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
    elif split_policy in {"auto", "stratified"} and stratified_split:
        train_idx, val_idx = multilabel_stratified_split(
            labels, args.val_split, args.seed
        )
        split_policy = "stratified"
        split_details = {"policy": "stratified", "exclusive_groups": False}
    else:
        train_idx, val_idx = row_train_val_split(
            len(features), args.val_split, args.seed
        )
        split_policy = "row"
        split_details = {"policy": "row", "exclusive_groups": False}

    eval_idx, calibration_idx, calibration_split_details = split_validation_calibration(
        val_idx,
        np.asarray(labels, dtype=np.float32),
        args.calibration_split,
        args.seed + 1,
    )
    split_details["calibration"] = calibration_split_details

    scaler = None
    if standardize:
        mean, std = compute_feature_stats(features, train_idx)
        scaler = {"mean": mean, "std": std}

    validation_report = build_validation_report(
        dataset_path=str(args.data[0]) if args.data else None,
        split_policy=split_policy,
        train_idx=train_idx,
        val_idx=eval_idx,
        labels=np.asarray(labels, dtype=np.float32),
        operator_classes=operator_classes,
        formula_keys=formula_keys,
        generator_families=generator_families,
        template_ids=template_ids,
        split_details=split_details,
        notes=[
            "Phase 3 report uses formula-key group validation when metadata is available.",
            "Row-permutation stress is covered by Phase 1 univariate feature/inference regression tests.",
        ],
    )

    print(f"  Split policy: {split_policy}")
    print(f"  Train: {len(train_idx)}, Val(eval): {len(eval_idx)}")
    if calibration_idx is not None:
        print(f"  Calibration/threshold rows: {len(calibration_idx)}")
    if validation_report["formula_overlap"].get("available"):
        overlap = validation_report["formula_overlap"]
        print(
            "  Formula overlap: "
            f"{overlap['overlap_unique_formulas']} unique, "
            f"{overlap['val_rows_with_train_formula_fraction']:.3f} val-row fraction"
        )

    # Data loaders with optimizations and lazy memmap-backed access
    train_dataset = IndexedFeatureDataset(
        features,
        labels,
        train_idx,
        scaler=scaler,
        device=device if args.load_into_ram else None,
    )
    val_dataset = IndexedFeatureDataset(
        features,
        labels,
        eval_idx,
        scaler=scaler,
        device=device if args.load_into_ram else None,
    )
    calibration_dataset = (
        IndexedFeatureDataset(
            features,
            labels,
            calibration_idx,
            scaler=scaler,
            device=device if args.load_into_ram else None,
        )
        if calibration_idx is not None
        else None
    )

    # Use pin_memory for GPU and num_workers for parallel data loading
    # On Windows, num_workers > 0 often causes deadlocks or hangs.
    import os
    import platform

    use_cuda = device.type == "cuda"

    # Default to 0 workers on Windows for stability, or allow override
    n_workers = 0
    if platform.system() != "Windows" and use_cuda:
        num_cpus = os.cpu_count() or 4
        n_workers = min(12, max(2, num_cpus - 2))

    loader_kwargs = {
        "num_workers": n_workers,
        "pin_memory": use_cuda,
    }

    # Only use advanced loader options on Linux/CUDA
    if n_workers > 0 and platform.system() != "Windows":
        loader_kwargs["prefetch_factor"] = 4
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **loader_kwargs
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, **loader_kwargs)
    calibration_loader = (
        DataLoader(
            calibration_dataset,
            batch_size=args.batch_size,
            **loader_kwargs,
        )
        if calibration_dataset is not None
        else None
    )

    # Model
    n_features = features.shape[1]
    n_classes = labels.shape[1]

    if args.model == "glu":
        model = CurveClassifierGLU(n_features, n_classes, args.hidden)
        model_config = {
            "n_features": int(n_features),
            "n_classes": int(n_classes),
            "hidden": int(args.hidden),
        }
    elif args.model == "mlp":
        model = CurveClassifierMLP(n_features, n_classes, args.hidden)
        model_config = {
            "n_features": int(n_features),
            "n_classes": int(n_classes),
            "hidden": int(args.hidden),
        }
    else:
        curve_dim = 128
        if feature_schema is not None and "raw" in feature_schema:
            raw_slice = feature_schema["raw"]
            if isinstance(raw_slice, (list, tuple)) and len(raw_slice) == 2:
                curve_dim = int(raw_slice[1] - raw_slice[0])
        model = CurveClassifierCNN(
            n_classes=n_classes, n_features=n_features, curve_dim=curve_dim
        )
        model_config = {
            "n_classes": int(n_classes),
            "n_features": int(n_features),
            "curve_dim": int(curve_dim),
        }

    model = model.to(device)
    print(f"\nModel: {args.model.upper()}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile...")
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"Warning: torch.compile failed ({e}). Falling back to eager mode.")

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
        pos_weights = np.clip(pos_weights, 0.5, float(args.class_weight_cap))
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
        threshold_beta=args.threshold_beta,
        calibrate_flag=args.calibrate,
        class_weights=class_weights,
        calibration_loader=calibration_loader,
        val_groups=(
            np.asarray(generator_families, dtype=object)[eval_idx]
            if generator_families is not None
            else None
        ),
    )

    # Persist scaler and schema metadata
    checkpoint = torch.load(output_path, weights_only=False)
    if scaler is not None:
        checkpoint["feature_scaler"] = scaler
    checkpoint["feature_schema"] = feature_schema
    checkpoint["feature_dim"] = feature_dim or features.shape[1]
    checkpoint["validation_split_policy"] = split_policy
    checkpoint["validation_split_details"] = dict(split_details)
    validation_report_path = (
        Path(args.validation_report)
        if args.validation_report
        else default_validation_report_path(output_path)
    )
    validation_report["metrics"] = {
        "best_checkpoint": {
            "epoch": checkpoint.get("epoch"),
            "val_loss": checkpoint.get("val_loss"),
            "val_acc": checkpoint.get("val_acc"),
            "val_f1": checkpoint.get("val_f1"),
            "val_micro_f1": checkpoint.get("val_micro_f1"),
            "f1_per_class": checkpoint.get("val_f1_per_class"),
            "precision_per_class": checkpoint.get("val_precision_per_class"),
            "recall_per_class": checkpoint.get("val_recall_per_class"),
            "per_class_acc": checkpoint.get("val_per_class_acc"),
            "threshold_beta": checkpoint.get("threshold_beta"),
            "thresholds": checkpoint.get("thresholds"),
            "thresholded_val_f1": checkpoint.get("thresholded_val_f1"),
            "thresholded_val_micro_f1": checkpoint.get("thresholded_val_micro_f1"),
            "thresholded_f1_per_class": checkpoint.get("thresholded_f1_per_class"),
            "thresholded_precision_per_class": checkpoint.get(
                "thresholded_precision_per_class"
            ),
            "thresholded_recall_per_class": checkpoint.get(
                "thresholded_recall_per_class"
            ),
            "operator_classes": checkpoint.get("operator_classes"),
            "by_family": checkpoint.get("val_metrics_by_family"),
        }
    }
    write_validation_report(validation_report_path, validation_report)
    checkpoint["validation_report_path"] = str(validation_report_path)
    checkpoint["labeler_version"] = dataset_metadata.get("labeler_version")
    checkpoint["data_generation_command"] = (
        args.data_generation_command or "not_provided"
    )
    checkpoint_card = build_checkpoint_card(
        model_kind="curve_classifier",
        checkpoint_path=output_path,
        validation_report=validation_report,
        checkpoint_metadata=checkpoint,
        data_generation_command=args.data_generation_command,
        training_command=" ".join(sys.argv),
        runtime_contract={
            "univariate": "trained_univariate_neural",
            "multivariate": "heuristic_slice_aggregation",
        },
    )
    checkpoint_card_path = (
        Path(args.checkpoint_card)
        if args.checkpoint_card
        else default_checkpoint_card_path(output_path)
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
                else default_rollout_comparison_path(output_path)
            )
            write_rollout_comparison(rollout_comparison_path, rollout_comparison)
            checkpoint["rollout_comparison_path"] = str(rollout_comparison_path)
        else:
            print(
                f"Warning: baseline card not found at {baseline_card_path}; "
                "skipping rollout comparison."
            )

    torch.save(checkpoint, output_path)

    print(f"\nModel saved to {output_path}")
    print(f"Validation report saved to {validation_report_path}")
    print(f"Checkpoint card saved to {checkpoint_card_path}")
    if checkpoint.get("rollout_comparison_path"):
        print(f"Rollout comparison saved to {checkpoint['rollout_comparison_path']}")


if __name__ == "__main__":
    main()
