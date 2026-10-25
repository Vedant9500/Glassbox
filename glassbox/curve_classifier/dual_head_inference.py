"""Inference for dual-head curve-classifier checkpoints.

The operator head is also exposed through the legacy ``predict_operators``
path (the checkpoint stores operator-only weights under ``model_state_dict``).
This module adds the shape-property head and returns both heads in one call.

Usage:
    from glassbox.curve_classifier.dual_head_inference import predict_dual_head

    result = predict_dual_head(x, y, "models/curve_classifier_v7_dual.pt")
    result["operators"]  # {"sin": 0.94, ...}
    result["shape"]      # {"periodic": 0.91, ...}
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from glassbox.curve_classifier.generate_curve_data import extract_all_features_xy
from glassbox.curve_classifier.models import DualHeadCurveClassifier

SYMLOG_START = 192
SYMLOG_END = 398


def load_dual_head_checkpoint(
    model_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[DualHeadCurveClassifier, dict]:
    """Load a dual-head checkpoint with weights-only deserialization."""
    checkpoint = torch.load(
        Path(model_path), map_location="cpu", weights_only=True
    )
    if not checkpoint.get("dual_head"):
        raise ValueError(
            f"{model_path} is not a dual-head checkpoint "
            "(missing dual_head=True)"
        )
    if "dual_head_state_dict" not in checkpoint:
        raise ValueError(f"{model_path} has no dual_head_state_dict")

    config = checkpoint.get("model_config") or {}
    operator_classes = [str(x) for x in checkpoint["operator_classes"]]
    shape_label_names = [str(x) for x in checkpoint["shape_label_names"]]
    model = DualHeadCurveClassifier(
        n_features=int(config["n_features"]),
        n_operator_classes=len(operator_classes),
        n_shape_labels=len(shape_label_names),
        hidden=int(config["hidden"]),
    )
    model.load_state_dict(checkpoint["dual_head_state_dict"])
    model.to(torch.device(device))
    model.eval()
    return model, checkpoint


def prepare_features(
    features: np.ndarray,
    scaler: dict | None = None,
) -> np.ndarray:
    """Apply the training transform: SymLog on [192:398], then standardize."""
    prepared = np.asarray(features, dtype=np.float32).copy()
    end = min(prepared.shape[-1], SYMLOG_END)
    if end > SYMLOG_START:
        prepared[..., SYMLOG_START:end] = np.sign(
            prepared[..., SYMLOG_START:end]
        ) * np.log1p(np.abs(prepared[..., SYMLOG_START:end]))
    if scaler is not None:
        mean = np.asarray(scaler["mean"], dtype=np.float32)
        std = np.asarray(scaler["std"], dtype=np.float32)
        prepared = prepared[..., : len(mean)]
        prepared = (prepared - mean) / (std + 1e-8)
    return prepared


def _probabilities(
    model: DualHeadCurveClassifier,
    features: np.ndarray,
    checkpoint: dict,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    prepared = prepare_features(features, checkpoint.get("feature_scaler"))
    tensor = torch.from_numpy(np.asarray(prepared, dtype=np.float32)).to(device)
    with torch.no_grad():
        operator_logits, shape_logits = model(tensor)
    return (
        torch.sigmoid(operator_logits).cpu().numpy(),
        torch.sigmoid(shape_logits).cpu().numpy(),
    )


def _resolve_thresholds(
    checkpoint: dict,
    key: str,
    class_names: list[str],
    fallback: float,
) -> np.ndarray:
    """Per-class thresholds from a checkpoint, or a constant fallback.

    The checkpoint stores operator thresholds as a flat list (legacy
    ``predict_operators`` indexes it positionally) and shape thresholds under
    ``shape_thresholds``.
    """
    raw = checkpoint.get(key)
    if raw is None:
        return np.full(len(class_names), float(fallback), dtype=np.float32)
    values = np.asarray([float(v) for v in raw], dtype=np.float32)
    if values.shape != (len(class_names),):
        raise ValueError(
            f"{key} has {values.size} entries but the checkpoint declares "
            f"{len(class_names)} classes"
        )
    return values


def predict_dual_head(
    x: np.ndarray,
    y: np.ndarray,
    model_path: str | Path,
    threshold: float = 0.5,
    device: str | torch.device = "cpu",
    use_checkpoint_thresholds: bool = True,
) -> dict:
    """Predict operator and shape-property probabilities from ``(x, y)``.

    When ``use_checkpoint_thresholds`` is true (the default) and the checkpoint
    carries tuned per-class thresholds (``thresholds`` / ``shape_thresholds``),
    those decide ``*_above_threshold``. Otherwise ``threshold`` is used for
    every class. The raw probabilities are always returned unchanged.
    """
    model, checkpoint = load_dual_head_checkpoint(model_path, device)
    features = extract_all_features_xy(np.asarray(x), np.asarray(y))
    operator_probs, shape_probs = _probabilities(
        model, features[None, :], checkpoint, torch.device(device)
    )

    operator_classes = [str(v) for v in checkpoint["operator_classes"]]
    shape_label_names = [str(v) for v in checkpoint["shape_label_names"]]
    if use_checkpoint_thresholds:
        operator_thresholds = _resolve_thresholds(
            checkpoint, "thresholds", operator_classes, threshold
        )
        shape_thresholds = _resolve_thresholds(
            checkpoint, "shape_thresholds", shape_label_names, threshold
        )
    else:
        operator_thresholds = np.full(
            len(operator_classes), float(threshold), dtype=np.float32
        )
        shape_thresholds = np.full(
            len(shape_label_names), float(threshold), dtype=np.float32
        )
    operators = {
        name: float(prob)
        for name, prob in zip(operator_classes, operator_probs[0], strict=False)
    }
    shape = {
        name: float(prob)
        for name, prob in zip(shape_label_names, shape_probs[0], strict=False)
    }
    return {
        "operators": operators,
        "shape": shape,
        "operators_above_threshold": [
            name
            for name, prob, cut in zip(
                operator_classes, operator_probs[0], operator_thresholds, strict=False
            )
            if prob >= cut
        ],
        "shape_above_threshold": [
            name
            for name, prob, cut in zip(
                shape_label_names, shape_probs[0], shape_thresholds, strict=False
            )
            if prob >= cut
        ],
        "thresholds": {
            "operator": {
                name: float(cut)
                for name, cut in zip(
                    operator_classes, operator_thresholds, strict=False
                )
            },
            "shape": {
                name: float(cut)
                for name, cut in zip(
                    shape_label_names, shape_thresholds, strict=False
                )
            },
        },
        "shape_label_names": shape_label_names,
    }


__all__ = [
    "load_dual_head_checkpoint",
    "predict_dual_head",
    "prepare_features",
]
