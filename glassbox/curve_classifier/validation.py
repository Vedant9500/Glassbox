"""Validation split and report helpers for curve classifier datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .generate_curve_data import formula_to_key


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_validation_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(dict(report)), indent=2, sort_keys=True), encoding="utf-8")


def default_validation_report_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name(f"{checkpoint_path.stem}.validation.json")


def object_array_to_list(raw: Any, limit: int | None = None) -> List[Any] | None:
    if raw is None:
        return None
    if isinstance(raw, np.ndarray):
        out = raw.tolist()
    else:
        out = list(raw)
    if limit is not None:
        out = out[:limit]
    return out


def formula_keys_from_metadata_or_formulas(
    metadata_keys: Sequence[Any] | None,
    formulas: Sequence[str] | None,
    limit: int | None = None,
) -> np.ndarray | None:
    if metadata_keys is not None:
        keys = [str(x) for x in object_array_to_list(metadata_keys, limit=limit)]
        return np.asarray(keys, dtype=object)
    if formulas is None:
        return None
    formula_list = list(formulas)
    if limit is not None:
        formula_list = formula_list[:limit]
    return np.asarray([formula_to_key(str(f)) for f in formula_list], dtype=object)


def row_train_val_split(n_samples: int, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    n_val = int(n_samples * val_ratio)
    if n_val < 1 or n_samples - n_val < 1:
        raise ValueError(
            f"val_ratio={val_ratio} creates train={n_samples - n_val} val={n_val}; "
            "both splits must contain at least one sample."
        )
    rng = np.random.RandomState(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    return indices[n_val:], indices[:n_val]


def grouped_train_val_split(
    groups: Sequence[Any],
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Split rows by whole-group membership so train/val groups do not overlap."""
    groups_arr = np.asarray(groups, dtype=object)
    n_samples = int(groups_arr.shape[0])
    if n_samples < 2:
        raise ValueError("Need at least two samples for a train/validation split")

    by_group: Dict[str, List[int]] = {}
    for idx, group in enumerate(groups_arr.tolist()):
        by_group.setdefault(str(group), []).append(idx)

    if len(by_group) < 2:
        train_idx, val_idx = row_train_val_split(n_samples, val_ratio, seed)
        return train_idx, val_idx, {
            "policy": "row_fallback_single_group",
            "group_count": len(by_group),
            "exclusive_groups": False,
        }

    target_val = max(1, int(round(n_samples * val_ratio)))
    target_groups = max(1, int(round(len(by_group) * val_ratio)))
    if len(by_group) > 2:
        target_groups = max(2, target_groups)
    target_groups = min(target_groups, len(by_group) - 1)
    rng = np.random.RandomState(seed)
    group_items = list(by_group.items())
    rng.shuffle(group_items)
    # Prefer formula diversity over row-count-perfect validation. Large duplicate
    # formula groups can otherwise dominate validation and hide whole operators.
    group_items.sort(key=lambda item: len(item[1]))

    val_groups: set[str] = set()
    val_count = 0
    for group, members in group_items:
        if val_count >= target_val and len(val_groups) >= target_groups:
            continue
        if len(val_groups) < len(group_items) - 1:
            val_groups.add(group)
            val_count += len(members)

    if not val_groups or len(val_groups) == len(group_items):
        train_idx, val_idx = row_train_val_split(n_samples, val_ratio, seed)
        return train_idx, val_idx, {
            "policy": "row_fallback_group_balance",
            "group_count": len(by_group),
            "exclusive_groups": False,
        }

    val_idx = np.asarray(
        [idx for group in val_groups for idx in by_group[group]],
        dtype=np.int64,
    )
    train_idx = np.asarray(
        [idx for group, members in by_group.items() if group not in val_groups for idx in members],
        dtype=np.int64,
    )
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    return train_idx, val_idx, {
        "policy": "formula_group",
        "group_count": len(by_group),
        "val_group_count": len(val_groups),
        "exclusive_groups": True,
        "target_val_rows": target_val,
        "target_val_groups": target_groups,
    }


def family_holdout_split(
    families: Sequence[Any],
    heldout_family: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    families_arr = np.asarray(families, dtype=object).astype(str)
    val_mask = families_arr == str(heldout_family)
    val_idx = np.flatnonzero(val_mask).astype(np.int64)
    train_idx = np.flatnonzero(~val_mask).astype(np.int64)
    if len(train_idx) < 1 or len(val_idx) < 1:
        raise ValueError(
            f"Cannot hold out family {heldout_family!r}: "
            f"train={len(train_idx)} val={len(val_idx)}"
        )
    return train_idx, val_idx, {
        "policy": "generator_family_holdout",
        "heldout_family": str(heldout_family),
        "exclusive_groups": True,
    }


def formula_overlap_report(
    formula_keys: Sequence[Any] | None,
    train_idx: Sequence[int],
    val_idx: Sequence[int],
) -> Dict[str, Any]:
    if formula_keys is None:
        return {"available": False}
    keys = np.asarray(formula_keys, dtype=object).astype(str)
    train_keys = set(keys[np.asarray(train_idx, dtype=np.int64)].tolist())
    val_keys_list = keys[np.asarray(val_idx, dtype=np.int64)].tolist()
    val_keys = set(val_keys_list)
    overlap = train_keys & val_keys
    val_rows_seen = sum(1 for key in val_keys_list if key in train_keys)
    return {
        "available": True,
        "train_unique_formulas": len(train_keys),
        "val_unique_formulas": len(val_keys),
        "overlap_unique_formulas": len(overlap),
        "val_unique_overlap_fraction": len(overlap) / max(1, len(val_keys)),
        "val_rows_with_train_formula": int(val_rows_seen),
        "val_rows_with_train_formula_fraction": val_rows_seen / max(1, len(val_keys_list)),
    }


def value_distribution(values: Sequence[Any] | None, indices: Sequence[int] | None = None) -> Dict[str, int]:
    if values is None:
        return {}
    arr = np.asarray(values, dtype=object).astype(str)
    if indices is not None:
        arr = arr[np.asarray(indices, dtype=np.int64)]
    unique, counts = np.unique(arr, return_counts=True)
    return {str(k): int(v) for k, v in zip(unique.tolist(), counts.tolist())}


def label_distribution(labels: np.ndarray, operator_classes: Sequence[str], indices: Sequence[int]) -> Dict[str, int]:
    subset = np.asarray(labels[np.asarray(indices, dtype=np.int64)], dtype=np.float32)
    counts = subset.sum(axis=0)
    return {
        str(name): int(counts[i])
        for i, name in enumerate(operator_classes)
        if i < len(counts)
    }


def multilabel_metric_summary(
    probs: np.ndarray,
    labels: np.ndarray,
    operator_classes: Sequence[str],
    thresholds: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Compute per-class precision/recall/F1/support from probabilities."""
    probs_arr = np.asarray(probs, dtype=np.float32)
    labels_arr = np.asarray(labels, dtype=np.float32)
    if thresholds is None:
        binary = probs_arr > 0.5
    else:
        binary = probs_arr > np.asarray(thresholds, dtype=np.float32).reshape(1, -1)
    truth = labels_arr > 0.5

    tp = np.logical_and(binary, truth).sum(axis=0).astype(np.float64)
    fp = np.logical_and(binary, ~truth).sum(axis=0).astype(np.float64)
    fn = np.logical_and(~binary, truth).sum(axis=0).astype(np.float64)
    support = truth.sum(axis=0).astype(np.float64)
    precision = tp / np.maximum(tp + fp, 1e-10)
    recall = tp / np.maximum(tp + fn, 1e-10)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-10)

    return {
        "operator_classes": [str(x) for x in operator_classes],
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1.tolist(),
        "support_per_class": support.astype(int).tolist(),
        "macro_f1": float(np.mean(f1)) if len(f1) else None,
    }


def multilabel_metrics_by_group(
    probs: np.ndarray,
    labels: np.ndarray,
    groups: Sequence[Any] | None,
    operator_classes: Sequence[str],
    min_rows: int = 25,
) -> Dict[str, Any]:
    """Compute class metrics per generator family/template group."""
    if groups is None:
        return {}
    groups_arr = np.asarray(groups, dtype=object).astype(str)
    out: Dict[str, Any] = {}
    for group in sorted(set(groups_arr.tolist())):
        mask = groups_arr == group
        if int(mask.sum()) < min_rows:
            continue
        summary = multilabel_metric_summary(
            np.asarray(probs)[mask],
            np.asarray(labels)[mask],
            operator_classes,
        )
        summary["rows"] = int(mask.sum())
        out[str(group)] = summary
    return out


def build_validation_report(
    *,
    dataset_path: str | None,
    split_policy: str,
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    labels: np.ndarray,
    operator_classes: Sequence[str],
    formula_keys: Sequence[Any] | None = None,
    generator_families: Sequence[Any] | None = None,
    template_ids: Sequence[Any] | None = None,
    split_details: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    notes: Iterable[str] | None = None,
) -> Dict[str, Any]:
    train_idx_arr = np.asarray(train_idx, dtype=np.int64)
    val_idx_arr = np.asarray(val_idx, dtype=np.int64)
    report: Dict[str, Any] = {
        "schema_version": "validation.phase3.v1",
        "dataset_path": dataset_path,
        "split_policy": split_policy,
        "split_details": dict(split_details or {}),
        "n_samples": int(labels.shape[0]),
        "train_rows": int(len(train_idx_arr)),
        "val_rows": int(len(val_idx_arr)),
        "formula_overlap": formula_overlap_report(formula_keys, train_idx_arr, val_idx_arr),
        "label_distribution": {
            "train": label_distribution(labels, operator_classes, train_idx_arr),
            "val": label_distribution(labels, operator_classes, val_idx_arr),
        },
        "generator_family_distribution": {
            "train": value_distribution(generator_families, train_idx_arr),
            "val": value_distribution(generator_families, val_idx_arr),
        },
        "template_distribution": {
            "train": value_distribution(template_ids, train_idx_arr),
            "val": value_distribution(template_ids, val_idx_arr),
        },
        "metrics": dict(metrics or {}),
        "notes": list(notes or []),
    }
    return report


def metrics_to_json_dict(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    excluded = {"preds", "labels", "logits"}
    return {str(k): _json_safe(v) for k, v in metrics.items() if k not in excluded}
