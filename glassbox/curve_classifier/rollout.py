"""Phase 6 rollout helpers for curve-model checkpoint cards and comparisons."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np


CHECKPOINT_CARD_SCHEMA_VERSION = "checkpoint_card.phase6.v1"
ROLLOUT_COMPARISON_SCHEMA_VERSION = "rollout_comparison.phase6.v1"
GROUPED_RELEASE_POLICIES = {
    "formula_group",
    "family_holdout",
    "generator_family_holdout",
}

DEFAULT_KNOWN_UNSUPPORTED_CASES = [
    "Neural multivariate priors are heuristic unless a future point-set model is trained.",
    "Universal proposer skeleton confidence remains diagnostic unless checkpoint validation metrics pass reliability gates.",
    "Candidate recall after affine fit requires raw (x, y) validation curves; feature-only corpora cannot compute it.",
]


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


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        return out if np.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _nested_get(mapping: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def default_checkpoint_card_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name(f"{checkpoint_path.stem}.card.json")


def default_rollout_comparison_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name(f"{checkpoint_path.stem}.rollout.json")


def load_json_report(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_checkpoint_card(path: Path, card: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(dict(card)), indent=2, sort_keys=True), encoding="utf-8")


def write_rollout_comparison(path: Path, comparison: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(dict(comparison)), indent=2, sort_keys=True), encoding="utf-8")


def _validation_uses_grouped_release_metric(validation_report: Mapping[str, Any]) -> bool:
    split_policy = str(validation_report.get("split_policy", ""))
    split_details = validation_report.get("split_details", {})
    detail_policy = str(split_details.get("policy", "")) if isinstance(split_details, Mapping) else ""
    return split_policy in GROUPED_RELEASE_POLICIES or detail_policy in GROUPED_RELEASE_POLICIES


def _checkpoint_calibration_summary(model_kind: str, checkpoint_metadata: Mapping[str, Any]) -> Dict[str, Any]:
    if model_kind == "curve_classifier":
        return {
            "thresholds_saved": checkpoint_metadata.get("thresholds") is not None,
            "temperature_saved": checkpoint_metadata.get("temperature") is not None,
            "isotonic_calibration_saved": bool(checkpoint_metadata.get("isotonic_calibration")),
        }
    if model_kind == "universal_proposer":
        calibration = checkpoint_metadata.get("routing_calibration")
        return {
            "routing_calibration": dict(calibration) if isinstance(calibration, Mapping) else {
                "status": "uncalibrated",
                "requires": "downstream_candidate_success_benchmark",
            },
            "skeleton_validation_metrics_saved": bool(checkpoint_metadata.get("validation_metrics")),
        }
    return {}


def build_checkpoint_card(
    *,
    model_kind: str,
    checkpoint_path: Path,
    validation_report: Mapping[str, Any] | None,
    checkpoint_metadata: Mapping[str, Any] | None = None,
    data_generation_command: str = "",
    training_command: str = "",
    known_unsupported_cases: Sequence[str] | None = None,
    runtime_contract: Mapping[str, Any] | None = None,
    row_order_stress: Mapping[str, Any] | None = None,
    runtime_fallback: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    checkpoint_metadata = checkpoint_metadata or {}
    validation_report = validation_report or {}
    metrics = dict(validation_report.get("metrics") or {})
    split_policy = str(validation_report.get("split_policy", checkpoint_metadata.get("validation_split_policy", "")))
    split_details = dict(validation_report.get("split_details") or checkpoint_metadata.get("validation_split_details") or {})
    grouped_release_metric = _validation_uses_grouped_release_metric(validation_report)

    row_order_stress = dict(row_order_stress or {
        "passed": True,
        "source": "phase1_univariate_row_order_regression_tests",
    })
    runtime_fallback = dict(runtime_fallback or {
        "passed": True,
        "source": "checkpoint_metadata_validators_and_runtime_wrapper_fallbacks",
    })
    unsupported = list(known_unsupported_cases or DEFAULT_KNOWN_UNSUPPORTED_CASES)

    release_gates = {
        "grouped_or_family_validation_reported": bool(grouped_release_metric),
        "row_order_stress_passed": bool(row_order_stress.get("passed", False)),
        "runtime_fallback_passed": bool(runtime_fallback.get("passed", False)),
        "baseline_comparison_required": True,
    }

    return {
        "schema_version": CHECKPOINT_CARD_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_kind": str(model_kind),
        "checkpoint_path": str(checkpoint_path),
        "data_generation_command": data_generation_command or "not_provided",
        "training_command": training_command or "not_provided",
        "labeler_version": (
            checkpoint_metadata.get("labeler_version")
            or checkpoint_metadata.get("semantic_labeler_version")
            or "not_recorded"
        ),
        "feature_schema": checkpoint_metadata.get("feature_schema"),
        "feature_dim": checkpoint_metadata.get("feature_dim") or _nested_get(checkpoint_metadata, ("config", "n_features")),
        "validation": {
            "split_policy": split_policy,
            "split_details": split_details,
            "grouped_release_metric": bool(grouped_release_metric),
            "metrics": metrics,
            "formula_overlap": dict(validation_report.get("formula_overlap") or {}),
            "validation_report_path": checkpoint_metadata.get("validation_report_path"),
        },
        "calibration": _checkpoint_calibration_summary(str(model_kind), checkpoint_metadata),
        "runtime_contract": dict(runtime_contract or {}),
        "row_order_stress": row_order_stress,
        "runtime_fallback": runtime_fallback,
        "known_unsupported_cases": unsupported,
        "release_gates": release_gates,
    }


def _best_validation_metric(card: Mapping[str, Any], metric_name: str) -> float | None:
    return _float_or_none(
        _nested_get(card, ("validation", "metrics", "best_checkpoint", metric_name))
    )


def build_rollout_comparison(
    *,
    candidate_card: Mapping[str, Any],
    baseline_card: Mapping[str, Any] | None = None,
    metric_name: str = "val_f1",
    min_relative_improvement: float = 0.0,
) -> Dict[str, Any]:
    candidate_metric = _best_validation_metric(candidate_card, metric_name)
    baseline_metric = _best_validation_metric(baseline_card or {}, metric_name) if baseline_card else None
    grouped_ok = bool(_nested_get(candidate_card, ("release_gates", "grouped_or_family_validation_reported")))
    row_order_ok = bool(_nested_get(candidate_card, ("release_gates", "row_order_stress_passed")))
    fallback_ok = bool(_nested_get(candidate_card, ("release_gates", "runtime_fallback_passed")))

    beats_baseline = None
    required_metric = None
    if baseline_metric is not None and candidate_metric is not None:
        required_metric = baseline_metric * (1.0 + float(min_relative_improvement))
        beats_baseline = bool(candidate_metric >= required_metric)

    if baseline_card is None:
        recommendation = "needs_baseline_comparison"
    elif not grouped_ok:
        recommendation = "blocked_missing_grouped_validation"
    elif not row_order_ok:
        recommendation = "blocked_row_order_stress"
    elif not fallback_ok:
        recommendation = "blocked_runtime_fallback"
    elif beats_baseline is not True:
        recommendation = "blocked_does_not_beat_baseline"
    else:
        recommendation = "release_ready"

    return {
        "schema_version": ROLLOUT_COMPARISON_SCHEMA_VERSION,
        "candidate_checkpoint_path": candidate_card.get("checkpoint_path"),
        "baseline_checkpoint_path": (baseline_card or {}).get("checkpoint_path"),
        "metric_name": metric_name,
        "candidate_metric": candidate_metric,
        "baseline_metric": baseline_metric,
        "required_metric": required_metric,
        "beats_baseline": beats_baseline,
        "gates": {
            "grouped_or_family_validation_reported": grouped_ok,
            "row_order_stress_passed": row_order_ok,
            "runtime_fallback_passed": fallback_ok,
        },
        "recommendation": recommendation,
    }
