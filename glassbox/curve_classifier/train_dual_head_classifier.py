"""Train a dual-head curve classifier on operator + shape-property labels.

The operator head keeps the production taxonomy (9 classes). The shape head
predicts affine-invariant geometric properties that transfer across generator
families. Both heads share one ``CurveClassifierGLU`` trunk.

Checkpoint compatibility: ``model_state_dict`` holds the operator-only GLU
weights, so existing ``predict_operators`` inference keeps working unchanged.
The full two-head weights are stored separately in ``dual_head_state_dict``.

Decision thresholds: the validation rows are split into disjoint eval and
calibration sets. Per-class thresholds are tuned on calibration and scored on
eval, then stored in ``thresholds`` (flat operator list) and
``shape_thresholds``. Family holdouts report raw 0.5 metrics plus
``*_tuned`` metrics whose thresholds come from a disjoint slice of that
family's own rows.

Usage:
    python -m glassbox.curve_classifier.train_dual_head_classifier \
        --data data/curve_dataset_v7_dual.npz \
        --output models/curve_classifier_v7_dual.pt \
        --epochs 150 --batch-size 512 --hidden 256 --weight-decay 1.0 \
        --holdout-family compound --holdout-family nested \
        --class-weights --class-weight-cap 16 \
        --pos-weight-override addition=4,exp=4,log=8,identity=3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from glassbox.curve_classifier.models import (
    CURVE_CLASSIFIER_ARCHITECTURE_VERSION,
    DualHeadCurveClassifier,
)
from glassbox.curve_classifier.rollout import (
    build_checkpoint_card,
    default_checkpoint_card_path,
    write_checkpoint_card,
)
from glassbox.curve_classifier.stable_losses import (
    StableBCEWithLogitsLoss,
)
from glassbox.curve_classifier.validation import (
    build_validation_report,
    default_validation_report_path,
    family_holdout_split,
    formula_keys_from_metadata_or_formulas,
    grouped_train_val_split,
    multilabel_metric_summary,
    write_validation_report,
)

# Production preprocessing: SymLog on the derivative/invariant block, then
# standardization. Applied in that order; the reverse order changes logits.
SYMLOG_START = 192
SYMLOG_END = 398


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_dual_dataset(path: str | Path) -> dict:
    """Load the arrays the dual-head trainer needs, without the raw curves."""
    with np.load(path, allow_pickle=True) as data:
        files = set(data.files)
        if "shape_labels" not in files:
            raise ValueError(
                f"{path} has no shape_labels; build it with "
                "grokking_experiments/make_dual_label_dataset.py"
            )
        features = np.asarray(data["features"], dtype=np.float32)
        labels = np.asarray(data["labels"], dtype=np.float32)
        shape_labels = np.asarray(data["shape_labels"], dtype=np.float32)
        operator_classes = [str(x) for x in data["operator_classes"].tolist()]
        shape_label_names = [str(x) for x in data["shape_label_names"].tolist()]
        formulas = data["formulas"].tolist() if "formulas" in files else None
        formula_keys = formula_keys_from_metadata_or_formulas(
            data["formula_keys"] if "formula_keys" in files else None,
            formulas,
        )
        families = (
            np.asarray([str(x) for x in data["generator_families"]], dtype=object)
            if "generator_families" in files
            else None
        )
        template_ids = (
            np.asarray(data["template_ids"], dtype=np.int64)
            if "template_ids" in files
            else None
        )
        feature_schema = (
            data["feature_schema"].item() if "feature_schema" in files else None
        )
        feature_dim = (
            int(data["feature_dim"]) if "feature_dim" in files else features.shape[1]
        )
        labeler_version = (
            str(data["labeler_version"]) if "labeler_version" in files else None
        )

    if features.shape[1] < SYMLOG_END:
        raise ValueError(
            f"feature width {features.shape[1]} < SYMLOG_END {SYMLOG_END}"
        )
    if labels.shape[1] != len(operator_classes):
        raise ValueError("operator label width does not match operator_classes")
    if shape_labels.shape[1] != len(shape_label_names):
        raise ValueError("shape label width does not match shape_label_names")

    return {
        "features": features,
        "labels": labels,
        "shape_labels": shape_labels,
        "operator_classes": operator_classes,
        "shape_label_names": shape_label_names,
        "formula_keys": formula_keys,
        "generator_families": families,
        "template_ids": template_ids,
        "feature_schema": feature_schema,
        "feature_dim": feature_dim,
        "labeler_version": labeler_version,
    }


def compute_feature_stats(
    features: np.ndarray,
    indices: np.ndarray,
    chunk_size: int = 65536,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean/std after SymLog on the invariant block, train rows only."""
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) == 0:
        raise ValueError("Cannot compute feature stats for empty indices")

    n_features = int(features.shape[1])
    total = 0
    sum_x = np.zeros(n_features, dtype=np.float64)
    sum_x2 = np.zeros(n_features, dtype=np.float64)

    for start in range(0, len(indices), chunk_size):
        batch = np.asarray(features[indices[start : start + chunk_size]], dtype=np.float64)
        batch[:, SYMLOG_START:SYMLOG_END] = np.sign(
            batch[:, SYMLOG_START:SYMLOG_END]
        ) * np.log1p(np.abs(batch[:, SYMLOG_START:SYMLOG_END]))
        sum_x += batch.sum(axis=0)
        sum_x2 += np.square(batch).sum(axis=0)
        total += batch.shape[0]

    mean = sum_x / max(total, 1)
    var = np.maximum(sum_x2 / max(total, 1) - np.square(mean), 0.0)
    std = np.sqrt(var) + 1e-8
    return mean.astype(np.float32), std.astype(np.float32)


def preprocess_features(
    features: np.ndarray,
    scaler: dict | None = None,
) -> np.ndarray:
    """Apply SymLog then standardization to a feature matrix (copy)."""
    x = np.asarray(features, dtype=np.float32).copy()
    x[:, SYMLOG_START:SYMLOG_END] = np.sign(x[:, SYMLOG_START:SYMLOG_END]) * np.log1p(
        np.abs(x[:, SYMLOG_START:SYMLOG_END])
    )
    if scaler is not None:
        x = (x - scaler["mean"]) / (scaler["std"] + 1e-8)
    return x.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def micro_f1(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> float:
    pred = np.asarray(probs) > threshold
    truth = np.asarray(labels) > 0.5
    tp = np.logical_and(pred, truth).sum(dtype=np.float64)
    fp = np.logical_and(pred, ~truth).sum(dtype=np.float64)
    fn = np.logical_and(~pred, truth).sum(dtype=np.float64)
    return float(2.0 * tp / max(2.0 * tp + fp + fn, 1e-10))


def summarize_head(
    probs: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    thresholds: np.ndarray | None = None,
) -> dict:
    summary = multilabel_metric_summary(
        probs, labels, class_names, thresholds=thresholds
    )
    support = [int(n) for n in summary["support_per_class"]]
    f1_values = [float(v) for v in summary["f1_per_class"]]
    supported = [v for v, n in zip(f1_values, support, strict=False) if n > 0]
    n_rows = int(np.asarray(labels).shape[0])
    return {
        "micro_f1": micro_f1(probs, labels, threshold=0.5)
        if thresholds is None
        else _micro_f1_with_thresholds(probs, labels, thresholds),
        "macro_f1": summary["macro_f1"],
        "macro_f1_supported": float(np.mean(supported)) if supported else None,
        "per_class_f1": {
            name: float(f1)
            for name, f1 in zip(class_names, summary["f1_per_class"], strict=False)
        },
        "support": {
            name: int(n)
            for name, n in zip(class_names, summary["support_per_class"], strict=False)
        },
        "zero_support_classes": [
            name for name, n in zip(class_names, support, strict=False) if n == 0
        ],
        # Precision is undefined (and F1 is vacuous) when every row is positive.
        "all_positive_classes": [
            name
            for name, n in zip(class_names, support, strict=False)
            if n_rows > 0 and n == n_rows
        ],
        "thresholds": None
        if thresholds is None
        else {
            name: float(t)
            for name, t in zip(class_names, thresholds, strict=False)
        },
    }


def _micro_f1_with_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray,
) -> float:
    pred = np.asarray(probs) > np.asarray(thresholds).reshape(1, -1)
    truth = np.asarray(labels) > 0.5
    tp = np.logical_and(pred, truth).sum(dtype=np.float64)
    fp = np.logical_and(pred, ~truth).sum(dtype=np.float64)
    fn = np.logical_and(~pred, truth).sum(dtype=np.float64)
    return float(2.0 * tp / max(2.0 * tp + fp + fn, 1e-10))


def tune_multilabel_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    steps: int = 25,
    beta: float = 1.0,
    min_support: int = 5,
    min_threshold: float = 0.01,
    max_threshold: float = 0.9,
) -> np.ndarray:
    """Per-class thresholds maximizing F-beta on held-in calibration rows.

    The geometric grid puts most resolution where the weak classes need it
    (0.01-0.1) while still covering conservative thresholds.
    """
    probs_arr = np.asarray(probs, dtype=np.float64)
    truth = np.asarray(labels) > 0.5
    n_classes = truth.shape[1]
    thresholds = np.full(n_classes, 0.5, dtype=np.float32)
    candidates = np.unique(
        np.concatenate(
            [
                np.geomspace(min_threshold, max_threshold, steps),
                np.asarray([0.5]),
            ]
        )
    )
    beta_sq = float(beta) ** 2
    # Visit candidates nearest 0.5 first so plateaus (equal F-beta over a wide
    # range) resolve to the least aggressive threshold instead of the lowest.
    ordered = sorted(candidates, key=lambda t: abs(float(t) - 0.5))

    for c in range(n_classes):
        y = truth[:, c]
        if int(y.sum()) < min_support:
            continue
        best_score = -1.0
        best_t = 0.5
        for t in ordered:
            pred = probs_arr[:, c] >= t
            tp = float(np.logical_and(pred, y).sum())
            fp = float(np.logical_and(pred, ~y).sum())
            fn = float(np.logical_and(~pred, y).sum())
            precision = tp / max(tp + fp, 1e-10)
            recall = tp / max(tp + fn, 1e-10)
            score = (
                (1.0 + beta_sq)
                * precision
                * recall
                / max(beta_sq * precision + recall, 1e-10)
            )
            if score > best_score + 1e-12:
                best_score = score
                best_t = float(t)
        thresholds[c] = best_t
    return thresholds


def compute_pos_weights(
    labels: np.ndarray,
    class_names: list[str],
    *,
    cap: float = 16.0,
    overrides: dict[str, float] | None = None,
) -> np.ndarray:
    """Inverse-frequency positive weights, clipped and optionally overridden."""
    labels_arr = np.asarray(labels, dtype=np.float64)
    positives = labels_arr.sum(axis=0)
    negatives = labels_arr.shape[0] - positives
    weights = negatives / np.maximum(positives, 1.0)
    weights = np.clip(weights, 0.5, float(cap))
    for name, value in (overrides or {}).items():
        if name not in class_names:
            raise ValueError(
                f"unknown class {name!r} in pos-weight override; "
                f"known: {class_names}"
            )
        weights[class_names.index(name)] = float(value)
    return weights.astype(np.float32)


def parse_weight_overrides(text: str) -> dict[str, float]:
    overrides: dict[str, float] = {}
    for part in str(text or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(
                f"pos-weight override {part!r} must be name=value"
            )
        name, value = part.split("=", 1)
        overrides[name.strip()] = float(value)
    return overrides


def split_calibration(
    indices: np.ndarray,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Split rows into fit/calibration; calibration is used only for thresholds."""
    indices = np.asarray(indices, dtype=np.int64)
    fraction = float(fraction)
    if fraction <= 0.0 or len(indices) < 20:
        return indices, None
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(indices))
    n_cal = max(1, round(len(indices) * fraction))
    calib = np.sort(indices[perm[:n_cal]])
    fit = np.sort(indices[perm[n_cal:]])
    return fit, calib



# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------


def evaluate(
    model: DualHeadCurveClassifier,
    features: torch.Tensor,
    operator_labels: torch.Tensor,
    shape_labels: torch.Tensor,
    device: torch.device,
    batch_size: int,
    operator_loss_fn,
    shape_loss_fn,
    shape_loss_weight: float,
) -> dict:
    model.eval()
    op_probs: list[np.ndarray] = []
    shape_probs: list[np.ndarray] = []
    total_loss = 0.0
    n_rows = 0

    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            xb = features[start : start + batch_size]
            yb_op = operator_labels[start : start + batch_size]
            yb_shape = shape_labels[start : start + batch_size]
            op_logits, shape_logits = model(xb)
            loss = operator_loss_fn(op_logits, yb_op) + shape_loss_weight * shape_loss_fn(
                shape_logits, yb_shape
            )
            total_loss += float(loss.detach()) * xb.shape[0]
            n_rows += xb.shape[0]
            op_probs.append(torch.sigmoid(op_logits).cpu().numpy())
            shape_probs.append(torch.sigmoid(shape_logits).cpu().numpy())

    return {
        "loss": total_loss / max(n_rows, 1),
        "operator_probs": np.concatenate(op_probs, axis=0),
        "shape_probs": np.concatenate(shape_probs, axis=0),
    }


def train_model(
    model: DualHeadCurveClassifier,
    features: torch.Tensor,
    operator_labels: torch.Tensor,
    shape_labels: torch.Tensor,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    shape_loss_weight: float,
    patience: int,
    eval_every: int,
    eval_batch_size: int,
    operator_classes: list[str],
    shape_label_names: list[str],
    device: torch.device,
    seed: int,
    label: str,
    select_best: bool = True,
    operator_loss_fn=None,
    shape_loss_fn=None,
) -> tuple[DualHeadCurveClassifier, dict, dict]:
    """Train one model; return (model, best_state, history)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=weight_decay
    )
    if operator_loss_fn is None:
        operator_loss_fn = StableBCEWithLogitsLoss()
    if shape_loss_fn is None:
        shape_loss_fn = StableBCEWithLogitsLoss()

    train_idx_t = torch.from_numpy(np.asarray(train_idx, dtype=np.int64))
    val_idx_t = torch.from_numpy(np.asarray(val_idx, dtype=np.int64))
    y_op = operator_labels
    y_shape = shape_labels

    best_state: dict | None = None
    best_score = -1.0
    best_epoch = 0
    best_metrics: dict = {}
    last_metrics: dict = {}
    history: list[dict] = []
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        perm = train_idx_t[torch.randperm(len(train_idx_t))]
        running_loss = 0.0
        n_seen = 0

        for start in range(0, len(perm), batch_size):
            idx = perm[start : start + batch_size]
            xb = features[idx]
            yb_op = y_op[idx]
            yb_shape = y_shape[idx]
            optimizer.zero_grad(set_to_none=True)
            op_logits, shape_logits = model(xb)
            loss = operator_loss_fn(op_logits, yb_op) + shape_loss_weight * shape_loss_fn(
                shape_logits, yb_shape
            )
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach()) * xb.shape[0]
            n_seen += xb.shape[0]

        should_eval = epoch == 1 or epoch % eval_every == 0 or epoch == epochs
        if not should_eval:
            continue

        val = evaluate(
            model,
            features[val_idx_t],
            y_op[val_idx_t],
            y_shape[val_idx_t],
            device,
            eval_batch_size,
            operator_loss_fn,
            shape_loss_fn,
            shape_loss_weight,
        )
        op_summary = summarize_head(
            val["operator_probs"],
            y_op[val_idx_t].cpu().numpy(),
            operator_classes,
        )
        shape_summary = summarize_head(
            val["shape_probs"],
            y_shape[val_idx_t].cpu().numpy(),
            shape_label_names,
        )
        score = op_summary["micro_f1"] + shape_summary["micro_f1"]
        record = {
            "epoch": epoch,
            "train_loss": running_loss / max(n_seen, 1),
            "val_loss": val["loss"],
            "operator_micro_f1": op_summary["micro_f1"],
            "shape_micro_f1": shape_summary["micro_f1"],
            "elapsed_s": time.time() - start_time,
        }
        history.append(record)
        print(
            f"[{label}] epoch {epoch:4d}  train {record['train_loss']:.4f}  "
            f"val {record['val_loss']:.4f}  "
            f"op_f1 {record['operator_micro_f1']:.4f}  "
            f"shape_f1 {record['shape_micro_f1']:.4f}"
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
            best_metrics = {
                "epoch": epoch,
                "val_loss": val["loss"],
                "operator": op_summary,
                "shape": shape_summary,
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience > 0 and patience_counter >= patience:
                print(f"[{label}] early stopping at epoch {epoch} (best {best_epoch})")
                break

        last_metrics = {
            "epoch": epoch,
            "val_loss": val["loss"],
            "operator": op_summary,
            "shape": shape_summary,
        }

    if select_best and best_state is not None:
        model.load_state_dict(best_state)
        return model, best_metrics, {"history": history, "best_epoch": best_epoch}
    # Family-holdout runs must not select on the held-out rows; report the
    # final epoch instead.
    return model, last_metrics, {"history": history, "best_epoch": epochs}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def operator_state_dict(model: DualHeadCurveClassifier) -> dict:
    """Strip the dual-head wrapper so plain GLU inference can load it."""
    state = {}
    for key, value in model.state_dict().items():
        if key.startswith("shape_head."):
            continue
        key = key.removeprefix("backbone.")
        state[key] = value
    return state


def build_checkpoint(
    model: DualHeadCurveClassifier,
    *,
    dataset: dict,
    model_config: dict,
    scaler: dict | None,
    best_metrics: dict,
    tuned_metrics: dict | None,
    operator_thresholds: np.ndarray | None,
    shape_thresholds: np.ndarray | None,
    threshold_meta: dict,
    split_policy: str,
    split_details: dict,
    validation_report_path: Path,
    checkpoint_card_path: Path,
) -> dict:
    op = best_metrics.get("operator", {})
    shape = best_metrics.get("shape", {})
    tuned_op = (tuned_metrics or {}).get("operator", {})
    tuned_shape = (tuned_metrics or {}).get("shape", {})
    # Inference reads ``feature_scaler`` and loads with weights_only=True, so
    # store plain lists rather than numpy arrays.
    feature_scaler = (
        {
            "mean": [float(x) for x in scaler["mean"]],
            "std": [float(x) for x in scaler["std"]],
        }
        if scaler is not None
        else None
    )
    return {
        "model_state_dict": operator_state_dict(model),
        "dual_head_state_dict": model.state_dict(),
        "dual_head": True,
        "epoch": int(best_metrics.get("epoch", 0)),
        "val_loss": float(best_metrics.get("val_loss", 0.0)),
        "val_f1": float(op.get("micro_f1", 0.0)),
        "val_micro_f1": float(op.get("micro_f1", 0.0)),
        "val_macro_f1": op.get("macro_f1"),
        "val_f1_per_class": [
            op.get("per_class_f1", {}).get(name, 0.0)
            for name in dataset["operator_classes"]
        ],
        "thresholded_val_f1": float(tuned_op.get("micro_f1", op.get("micro_f1", 0.0))),
        "thresholded_val_macro_f1": tuned_op.get("macro_f1", op.get("macro_f1")),
        "thresholds": None
        if operator_thresholds is None
        else [float(x) for x in operator_thresholds],
        "shape_thresholds": None
        if shape_thresholds is None
        else [float(x) for x in shape_thresholds],
        "thresholds_meta": dict(threshold_meta),
        "shape_val_micro_f1": float(shape.get("micro_f1", 0.0)),
        "shape_val_macro_f1": shape.get("macro_f1"),
        "shape_val_f1_per_class": [
            shape.get("per_class_f1", {}).get(name, 0.0)
            for name in dataset["shape_label_names"]
        ],
        "shape_thresholded_val_micro_f1": float(
            tuned_shape.get("micro_f1", shape.get("micro_f1", 0.0))
        ),
        "operator_classes": list(dataset["operator_classes"]),
        "shape_label_names": list(dataset["shape_label_names"]),
        "model_type": "glu",
        "model_config": {
            **model_config,
            # Legacy consumers read ``n_classes`` for the operator head.
            "n_classes": int(model_config["n_operator_classes"]),
            "architecture_version": CURVE_CLASSIFIER_ARCHITECTURE_VERSION,
            "dual_head": True,
        },
        "architecture_version": CURVE_CLASSIFIER_ARCHITECTURE_VERSION,
        "feature_schema": dataset["feature_schema"],
        "feature_dim": int(dataset["feature_dim"]),
        "labeler_version": dataset["labeler_version"],
        "feature_scaler": feature_scaler,
        "symlog_start": SYMLOG_START,
        "symlog_end": SYMLOG_END,
        "validation_split_policy": split_policy,
        "validation_split_details": split_details,
        "validation_report_path": str(validation_report_path),
        "checkpoint_card_path": str(checkpoint_card_path),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Dual-label .npz dataset")
    parser.add_argument(
        "--output",
        default="models/curve_classifier_dual_head.pt",
        help="Checkpoint path",
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0)
    parser.add_argument("--shape-loss-weight", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=0, help="0 disables early stopping")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--split-policy",
        choices=["formula_group", "row"],
        default="formula_group",
    )
    parser.add_argument(
        "--holdout-family",
        action="append",
        default=[],
        help="Train an extra model without this family and report holdout metrics",
    )
    parser.add_argument(
        "--holdout-epochs",
        type=int,
        default=None,
        help="Epochs for family-holdout runs (defaults to --epochs)",
    )
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument(
        "--calibration-split",
        type=float,
        default=0.25,
        help=(
            "Fraction of validation rows reserved for threshold tuning. The "
            "remaining rows drive model selection, so thresholds never see the "
            "rows they are scored on. 0 disables the split and tuning."
        ),
    )
    parser.add_argument(
        "--tune-thresholds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Tune per-class decision thresholds on the calibration split",
    )
    parser.add_argument(
        "--threshold-beta",
        type=float,
        default=1.0,
        help="F-beta beta used when tuning thresholds (1.0 = F1)",
    )
    parser.add_argument("--threshold-steps", type=int, default=25)
    parser.add_argument("--threshold-min", type=float, default=0.01)
    parser.add_argument("--threshold-max", type=float, default=0.9)
    parser.add_argument(
        "--class-weights",
        action="store_true",
        help="Use inverse-frequency positive weights for the operator head",
    )
    parser.add_argument(
        "--shape-class-weights",
        action="store_true",
        help="Use inverse-frequency positive weights for the shape head",
    )
    parser.add_argument(
        "--class-weight-cap",
        type=float,
        default=16.0,
        help="Maximum positive class weight (applies to both heads)",
    )
    parser.add_argument(
        "--pos-weight-override",
        default="",
        help="Comma-separated operator overrides, e.g. addition=4,log=8",
    )
    parser.add_argument(
        "--shape-pos-weight-override",
        default="",
        help="Comma-separated shape-label overrides, e.g. linear=4",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--notes", default="")
    return parser.parse_args(argv)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = resolve_device(args.device)
    print(f"Device: {device}")

    dataset = load_dual_dataset(args.data)
    features_np = dataset.pop("features")
    operator_labels = np.asarray(dataset["labels"], dtype=np.float32)
    shape_labels = np.asarray(dataset["shape_labels"], dtype=np.float32)
    n_rows, n_features = features_np.shape
    n_operator_classes = operator_labels.shape[1]
    n_shape_labels = shape_labels.shape[1]
    print(
        f"Dataset: {n_rows} rows, {n_features} features, "
        f"{n_operator_classes} operator labels, {n_shape_labels} shape labels"
    )

    model_meta_operator = list(dataset["operator_classes"])
    model_meta_shape = list(dataset["shape_label_names"])

    groups = dataset["formula_keys"]
    if args.split_policy == "formula_group":
        if groups is None:
            raise ValueError("formula_group split requires formula keys")
        train_idx, val_idx, split_details = grouped_train_val_split(
            groups, args.val_split, args.seed
        )
        split_policy = "formula_group"
    else:
        rng = np.random.RandomState(args.seed)
        perm = rng.permutation(n_rows)
        n_val = max(1, round(n_rows * args.val_split))
        val_idx = np.sort(perm[:n_val])
        train_idx = np.sort(perm[n_val:])
        split_policy = "row"
        split_details = {"policy": "row", "exclusive_groups": False}

    # Threshold tuning uses rows that never influence model selection.
    if args.tune_thresholds and args.calibration_split > 0.0:
        eval_idx, calib_idx = split_calibration(
            val_idx, args.calibration_split, args.seed
        )
        if calib_idx is None:
            print(
                "Calibration split requested but validation set is too small; "
                "threshold tuning disabled."
            )
    else:
        eval_idx, calib_idx = val_idx, None
    print(
        f"Validation rows: {len(eval_idx)} eval"
        + (
            f", {len(calib_idx)} calibration"
            if calib_idx is not None
            else " (threshold tuning disabled)"
        )
    )

    operator_pos_weights = None
    shape_pos_weights = None
    if args.class_weights or args.pos_weight_override:
        operator_pos_weights = compute_pos_weights(
            operator_labels[train_idx],
            model_meta_operator,
            cap=args.class_weight_cap,
            overrides=parse_weight_overrides(args.pos_weight_override),
        )
    if args.shape_class_weights or args.shape_pos_weight_override:
        shape_pos_weights = compute_pos_weights(
            shape_labels[train_idx],
            model_meta_shape,
            cap=args.class_weight_cap,
            overrides=parse_weight_overrides(args.shape_pos_weight_override),
        )
    if operator_pos_weights is not None:
        print(
            "Operator pos_weight: "
            + ", ".join(
                f"{name}={weight:.2f}"
                for name, weight in zip(
                    model_meta_operator, operator_pos_weights, strict=False
                )
            )
        )
    if shape_pos_weights is not None:
        print(
            "Shape pos_weight: "
            + ", ".join(
                f"{name}={weight:.2f}"
                for name, weight in zip(
                    model_meta_shape, shape_pos_weights, strict=False
                )
            )
        )

    operator_loss_fn = StableBCEWithLogitsLoss(
        pos_weight=None
        if operator_pos_weights is None
        else torch.from_numpy(operator_pos_weights).to(device)
    )
    shape_loss_fn = StableBCEWithLogitsLoss(
        pos_weight=None
        if shape_pos_weights is None
        else torch.from_numpy(shape_pos_weights).to(device)
    )

    scaler = None
    if not args.no_standardize:
        mean, std = compute_feature_stats(features_np, train_idx)
        scaler = {"mean": mean, "std": std}

    print("Preprocessing features (SymLog -> standardization)...")
    x_all = preprocess_features(features_np, scaler)
    del features_np
    features = torch.from_numpy(x_all).to(device)
    y_op = torch.from_numpy(operator_labels).to(device)
    y_shape = torch.from_numpy(shape_labels).to(device)

    model_config = {
        "n_features": int(n_features),
        "n_operator_classes": int(n_operator_classes),
        "n_shape_labels": int(n_shape_labels),
        "hidden": int(args.hidden),
    }
    model = DualHeadCurveClassifier(**model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    model, best_metrics, train_info = train_model(
        model,
        features,
        y_op,
        y_shape,
        train_idx,
        eval_idx,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        shape_loss_weight=args.shape_loss_weight,
        patience=args.patience,
        eval_every=args.eval_every,
        eval_batch_size=args.eval_batch_size,
        operator_classes=model_meta_operator,
        shape_label_names=model_meta_shape,
        device=device,
        seed=args.seed,
        label="main",
        operator_loss_fn=operator_loss_fn,
        shape_loss_fn=shape_loss_fn,
    )
    print(
        f"Best epoch {train_info['best_epoch']}: "
        f"operator micro-F1 {best_metrics['operator']['micro_f1']:.4f}, "
        f"shape micro-F1 {best_metrics['shape']['micro_f1']:.4f}"
    )

    # Tune thresholds on calibration rows, then score them on the disjoint
    # eval rows. The tuned numbers are therefore not fitted to what they report.
    operator_thresholds = None
    shape_thresholds = None
    tuned_metrics: dict = {}
    threshold_meta = {
        "enabled": calib_idx is not None,
        "source": "validation_calibration_split",
        "calibration_rows": 0 if calib_idx is None else len(calib_idx),
        "eval_rows": len(eval_idx),
        "beta": float(args.threshold_beta),
        "steps": int(args.threshold_steps),
        "range": [float(args.threshold_min), float(args.threshold_max)],
    }
    if calib_idx is not None:
        calib_eval = evaluate(
            model,
            features[torch.from_numpy(calib_idx)],
            y_op[torch.from_numpy(calib_idx)],
            y_shape[torch.from_numpy(calib_idx)],
            device,
            args.eval_batch_size,
            operator_loss_fn,
            shape_loss_fn,
            args.shape_loss_weight,
        )
        operator_thresholds = tune_multilabel_thresholds(
            calib_eval["operator_probs"],
            operator_labels[calib_idx],
            steps=args.threshold_steps,
            beta=args.threshold_beta,
            min_threshold=args.threshold_min,
            max_threshold=args.threshold_max,
        )
        shape_thresholds = tune_multilabel_thresholds(
            calib_eval["shape_probs"],
            shape_labels[calib_idx],
            steps=args.threshold_steps,
            beta=args.threshold_beta,
            min_threshold=args.threshold_min,
            max_threshold=args.threshold_max,
        )
        eval_scored = evaluate(
            model,
            features[torch.from_numpy(eval_idx)],
            y_op[torch.from_numpy(eval_idx)],
            y_shape[torch.from_numpy(eval_idx)],
            device,
            args.eval_batch_size,
            operator_loss_fn,
            shape_loss_fn,
            args.shape_loss_weight,
        )
        tuned_metrics = {
            "operator": summarize_head(
                eval_scored["operator_probs"],
                operator_labels[eval_idx],
                model_meta_operator,
                thresholds=operator_thresholds,
            ),
            "shape": summarize_head(
                eval_scored["shape_probs"],
                shape_labels[eval_idx],
                model_meta_shape,
                thresholds=shape_thresholds,
            ),
        }
        print(
            "Tuned operator micro-F1 "
            f"{tuned_metrics['operator']['micro_f1']:.4f} "
            f"(raw {best_metrics['operator']['micro_f1']:.4f}); "
            "tuned shape micro-F1 "
            f"{tuned_metrics['shape']['micro_f1']:.4f} "
            f"(raw {best_metrics['shape']['micro_f1']:.4f})"
        )

    # Family holdout: train a separate model without each held-out family.
    family_holdout: dict = {}
    holdout_epochs = args.holdout_epochs or args.epochs
    families = dataset["generator_families"]
    if args.holdout_family and families is None:
        raise ValueError("--holdout-family requires generator_families metadata")
    for family in args.holdout_family:
        print(f"\n=== family holdout: {family} ===")
        h_train_idx, h_val_idx, h_details = family_holdout_split(families, family)
        # Thresholds for a held-out family are fitted on a disjoint slice of
        # that family's own rows. The headline ``operator``/``shape`` entries
        # stay at 0.5 so they remain comparable to earlier transfer runs.
        if args.tune_thresholds and args.calibration_split > 0.0:
            h_eval_idx, h_calib_idx = split_calibration(
                h_val_idx, args.calibration_split, args.seed
            )
        else:
            h_eval_idx, h_calib_idx = h_val_idx, None
        h_model = DualHeadCurveClassifier(**model_config).to(device)
        h_model, h_metrics, h_info = train_model(
            h_model,
            features,
            y_op,
            y_shape,
            h_train_idx,
            h_eval_idx,
            epochs=holdout_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            shape_loss_weight=args.shape_loss_weight,
            patience=0,
            eval_every=args.eval_every,
            eval_batch_size=args.eval_batch_size,
            operator_classes=model_meta_operator,
            shape_label_names=model_meta_shape,
            device=device,
            seed=args.seed,
            label=f"holdout:{family}",
            select_best=False,
            operator_loss_fn=operator_loss_fn,
            shape_loss_fn=shape_loss_fn,
        )
        holdout_entry = {
            "split_details": h_details,
            "train_rows": len(h_train_idx),
            "val_rows": len(h_val_idx),
            "eval_rows": len(h_eval_idx),
            "calibration_rows": 0 if h_calib_idx is None else len(h_calib_idx),
            "best_epoch": int(h_info["best_epoch"]),
            "operator": h_metrics["operator"],
            "shape": h_metrics["shape"],
        }
        if h_calib_idx is not None:
            h_eval_scored = evaluate(
                h_model,
                features[torch.from_numpy(h_eval_idx)],
                y_op[torch.from_numpy(h_eval_idx)],
                y_shape[torch.from_numpy(h_eval_idx)],
                device,
                args.eval_batch_size,
                operator_loss_fn,
                shape_loss_fn,
                args.shape_loss_weight,
            )
            h_calib_scored = evaluate(
                h_model,
                features[torch.from_numpy(h_calib_idx)],
                y_op[torch.from_numpy(h_calib_idx)],
                y_shape[torch.from_numpy(h_calib_idx)],
                device,
                args.eval_batch_size,
                operator_loss_fn,
                shape_loss_fn,
                args.shape_loss_weight,
            )
            h_op_thresholds = tune_multilabel_thresholds(
                h_calib_scored["operator_probs"],
                operator_labels[h_calib_idx],
                steps=args.threshold_steps,
                beta=args.threshold_beta,
                min_threshold=args.threshold_min,
                max_threshold=args.threshold_max,
            )
            h_shape_thresholds = tune_multilabel_thresholds(
                h_calib_scored["shape_probs"],
                shape_labels[h_calib_idx],
                steps=args.threshold_steps,
                beta=args.threshold_beta,
                min_threshold=args.threshold_min,
                max_threshold=args.threshold_max,
            )
            holdout_entry["operator_tuned"] = summarize_head(
                h_eval_scored["operator_probs"],
                operator_labels[h_eval_idx],
                model_meta_operator,
                thresholds=h_op_thresholds,
            )
            holdout_entry["shape_tuned"] = summarize_head(
                h_eval_scored["shape_probs"],
                shape_labels[h_eval_idx],
                model_meta_shape,
                thresholds=h_shape_thresholds,
            )
            holdout_entry["thresholds"] = {
                "operator": {
                    name: float(t)
                    for name, t in zip(
                        model_meta_operator, h_op_thresholds, strict=False
                    )
                },
                "shape": {
                    name: float(t)
                    for name, t in zip(
                        model_meta_shape, h_shape_thresholds, strict=False
                    )
                },
                "source": "holdout_calibration_split",
            }
        family_holdout[str(family)] = holdout_entry
        print(
            f"[holdout:{family}] operator micro-F1 "
            f"{h_metrics['operator']['micro_f1']:.4f}, shape micro-F1 "
            f"{h_metrics['shape']['micro_f1']:.4f}"
        )
        if "operator_tuned" in holdout_entry:
            print(
                f"[holdout:{family}] tuned operator micro-F1 "
                f"{holdout_entry['operator_tuned']['micro_f1']:.4f}, "
                "tuned shape micro-F1 "
                f"{holdout_entry['shape_tuned']['micro_f1']:.4f}"
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = default_validation_report_path(output_path)
    card_path = default_checkpoint_card_path(output_path)

    validation_report = build_validation_report(
        dataset_path=str(args.data),
        split_policy=split_policy,
        train_idx=train_idx,
        val_idx=val_idx,
        labels=operator_labels,
        operator_classes=dataset["operator_classes"],
        formula_keys=dataset["formula_keys"],
        generator_families=dataset["generator_families"],
        template_ids=dataset["template_ids"],
        split_details={
            **split_details,
            "eval_rows": len(eval_idx),
            "calibration_rows": 0 if calib_idx is None else len(calib_idx),
            "threshold_source": threshold_meta["source"]
            if calib_idx is not None
            else None,
        },
        metrics={
            "best_checkpoint": {
                "epoch": int(best_metrics["epoch"]),
                "val_loss": float(best_metrics["val_loss"]),
                "val_f1": float(best_metrics["operator"]["micro_f1"]),
                "operator_micro_f1": float(best_metrics["operator"]["micro_f1"]),
                "shape_micro_f1": float(best_metrics["shape"]["micro_f1"]),
                "eval_rows": len(eval_idx),
            },
            "operator": best_metrics["operator"],
            "shape": best_metrics["shape"],
            "operator_tuned": tuned_metrics.get("operator"),
            "shape_tuned": tuned_metrics.get("shape"),
            "thresholds_meta": threshold_meta,
        },
        thresholds=None
        if operator_thresholds is None
        else {
            "operator": {
                name: float(t)
                for name, t in zip(
                    model_meta_operator, operator_thresholds, strict=False
                )
            },
            "shape": {
                name: float(t)
                for name, t in zip(model_meta_shape, shape_thresholds, strict=False)
            },
            **threshold_meta,
        },
        shape_label_names=dataset["shape_label_names"],
        shape_metrics=best_metrics["shape"],
        family_holdout=family_holdout or None,
        notes=[
            "Dual-head model: shared GLU trunk, operator head + shape-property head.",
            "Preprocessing is SymLog on features[192:398] followed by standardization.",
            "Family-holdout rows are excluded from training for their own entry.",
            (
                "Thresholds are tuned on a validation calibration split and "
                "scored on the disjoint eval rows; each family holdout tunes "
                "its own thresholds on a disjoint slice of its held-out rows."
            ),
            (
                "Family-holdout 'operator'/'shape' entries stay at threshold "
                "0.5; '*_tuned' entries use target-calibrated thresholds."
            ),
            "macro_f1_supported excludes classes with zero eval support.",
            *( [args.notes] if args.notes else [] ),
        ],
    )
    write_validation_report(report_path, validation_report)
    print(f"\nWrote validation report: {report_path}")

    checkpoint = build_checkpoint(
        model,
        dataset=dataset,
        model_config=model_config,
        scaler=scaler,
        best_metrics=best_metrics,
        tuned_metrics=tuned_metrics,
        operator_thresholds=operator_thresholds,
        shape_thresholds=shape_thresholds,
        threshold_meta=threshold_meta,
        split_policy=split_policy,
        split_details=split_details,
        validation_report_path=report_path,
        checkpoint_card_path=card_path,
    )
    checkpoint["family_holdout"] = family_holdout
    checkpoint["training_history"] = train_info["history"]
    torch.save(checkpoint, output_path)
    print(f"Wrote checkpoint: {output_path}")

    card = build_checkpoint_card(
        model_kind="curve_classifier",
        checkpoint_path=output_path,
        validation_report=validation_report,
        checkpoint_metadata=checkpoint,
        data_generation_command=f"python grokking_experiments/make_dual_label_dataset.py --output {args.data}",
        training_command="python -m glassbox.curve_classifier.train_dual_head_classifier "
        + " ".join(sys.argv[1:]),
        runtime_contract={
            "dual_head": True,
            "shape_label_names": list(dataset["shape_label_names"]),
            "preprocessing": "symlog[192:398]+standardize",
        },
    )
    write_checkpoint_card(card_path, card)
    print(f"Wrote checkpoint card: {card_path}")

    summary = {
        "checkpoint": str(output_path),
        "validation_report": str(report_path),
        "checkpoint_card": str(card_path),
        "rows": int(n_rows),
        "train_rows": len(train_idx),
        "val_rows": len(val_idx),
        "eval_rows": len(eval_idx),
        "calibration_rows": 0 if calib_idx is None else len(calib_idx),
        "best_epoch": int(best_metrics["epoch"]),
        "operator_micro_f1": float(best_metrics["operator"]["micro_f1"]),
        "shape_micro_f1": float(best_metrics["shape"]["micro_f1"]),
        "operator_micro_f1_tuned": None
        if not tuned_metrics
        else float(tuned_metrics["operator"]["micro_f1"]),
        "shape_micro_f1_tuned": None
        if not tuned_metrics
        else float(tuned_metrics["shape"]["micro_f1"]),
        "family_holdout": {
            name: {
                "operator_micro_f1": float(entry["operator"]["micro_f1"]),
                "shape_micro_f1": float(entry["shape"]["micro_f1"]),
                "operator_micro_f1_tuned": None
                if "operator_tuned" not in entry
                else float(entry["operator_tuned"]["micro_f1"]),
                "shape_micro_f1_tuned": None
                if "shape_tuned" not in entry
                else float(entry["shape_tuned"]["micro_f1"]),
            }
            for name, entry in family_holdout.items()
        },
    }
    print("\n" + json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
