"""FPIP v2 schema and helpers for fast-path to evolution handoff.

This module defines a stable, minimal contract that can be expanded as the
universal proposer prototype matures.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CandidateSkeleton:
    formula: str
    score: Optional[float] = None
    probability: Optional[float] = None
    mse: Optional[float] = None


@dataclass
class SequenceUncertainty:
    entropy: Optional[float] = None
    margin: Optional[float] = None
    confident: Optional[bool] = None


@dataclass
class RoutingSignal:
    recommend_guided_evolution: bool
    reason: str


@dataclass
class FPIPv2:
    schema_version: str = "fpip.v2"
    candidate_skeletons: List[CandidateSkeleton] = field(default_factory=list)
    sequence_uncertainty: SequenceUncertainty = field(default_factory=SequenceUncertainty)
    operator_priors: Dict[str, float] = field(default_factory=dict)
    interaction_hints: Dict[str, Any] = field(default_factory=dict)
    fit_diagnostics: Dict[str, Any] = field(default_factory=dict)
    routing_signal: RoutingSignal = field(
        default_factory=lambda: RoutingSignal(False, "default_fast_accept")
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _normalize_priors(predictions: Dict[str, float]) -> Dict[str, float]:
    cleaned: Dict[str, float] = {}
    total = 0.0
    for k, v in predictions.items():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if fv <= 0.0:
            continue
        cleaned[k] = fv
        total += fv

    if total <= 0.0:
        return {}
    return {k: v / total for k, v in cleaned.items()}


def _routing_from_signals(uncertainty: Dict[str, Any], residual_diag: Dict[str, Any]) -> RoutingSignal:
    uncertain = bool(uncertainty.get("prediction_uncertain", False))
    suspicious_residual = bool(residual_diag.get("residual_suspicious", False))

    if uncertain and suspicious_residual:
        return RoutingSignal(True, "uncertain_and_residual_suspicious")
    if uncertain:
        return RoutingSignal(True, "uncertain_predictions")
    if suspicious_residual:
        return RoutingSignal(True, "residual_suspicious")
    return RoutingSignal(False, "high_confidence_fast_path")


def build_fpip_v2_from_fast_path(
    *,
    formula: str,
    mse: float,
    candidate_formulas: Optional[List[Dict[str, Any]]] = None,
    predictions: Optional[Dict[str, float]] = None,
    uncertainty: Optional[Dict[str, Any]] = None,
    residual_diagnostics: Optional[Dict[str, Any]] = None,
    operator_hints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build FPIP v2 payload from current fast-path outputs."""
    predictions = predictions or {}
    uncertainty = uncertainty or {}
    residual_diagnostics = residual_diagnostics or {}
    operator_hints = operator_hints or {}
    candidate_formulas = candidate_formulas or []

    top_candidates: List[CandidateSkeleton] = []
    for cand in candidate_formulas[:5]:
        top_candidates.append(
            CandidateSkeleton(
                formula=str(cand.get("formula", "")),
                score=_to_float_or_none(cand.get("score")),
                probability=None,
                mse=_to_float_or_none(cand.get("mse")),
            )
        )

    # Ensure there is always at least one candidate to seed downstream systems.
    if not top_candidates:
        top_candidates.append(CandidateSkeleton(formula=formula, score=_to_float_or_none(mse), mse=_to_float_or_none(mse)))

    interaction_hints = {
        "operators": sorted(list(operator_hints.get("operators", []))) if isinstance(operator_hints.get("operators"), (set, list, tuple)) else [],
        "frequencies": list(operator_hints.get("frequencies", [])) if isinstance(operator_hints.get("frequencies"), (list, tuple)) else [],
        "powers": list(operator_hints.get("powers", [])) if isinstance(operator_hints.get("powers"), (list, tuple)) else [],
        "has_rational": bool(operator_hints.get("has_rational", False)),
        "has_exp_decay": bool(operator_hints.get("has_exp_decay", False)),
    }

    fpip = FPIPv2(
        candidate_skeletons=top_candidates,
        sequence_uncertainty=SequenceUncertainty(
            entropy=_to_float_or_none(uncertainty.get("prediction_entropy")),
            margin=_to_float_or_none(uncertainty.get("prediction_margin")),
            confident=not bool(uncertainty.get("prediction_uncertain", False)) if uncertainty else None,
        ),
        operator_priors=_normalize_priors(predictions),
        interaction_hints=interaction_hints,
        fit_diagnostics={
            "mse": _to_float_or_none(mse),
            "residual_suspicious": bool(residual_diagnostics.get("residual_suspicious", False)),
            "residual_spectral_peak_ratio": _to_float_or_none(residual_diagnostics.get("residual_spectral_peak_ratio")),
            "residual_holdout_ratio": _to_float_or_none(residual_diagnostics.get("residual_holdout_ratio")),
        },
        routing_signal=_routing_from_signals(uncertainty, residual_diagnostics),
    )

    return fpip.to_dict()


def validate_fpip_v2_payload(payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate minimal FPIP v2 structure for safe downstream consumption."""
    errors: List[str] = []
    if not isinstance(payload, dict):
        return False, ["payload must be a dict"]

    if payload.get("schema_version") != "fpip.v2":
        errors.append("schema_version must be 'fpip.v2'")

    candidates = payload.get("candidate_skeletons")
    if not isinstance(candidates, list) or len(candidates) == 0:
        errors.append("candidate_skeletons must be a non-empty list")
    else:
        for idx, c in enumerate(candidates):
            if not isinstance(c, dict) or not str(c.get("formula", "")).strip():
                errors.append(f"candidate_skeletons[{idx}] must include formula")

    uncertainty = payload.get("sequence_uncertainty")
    if not isinstance(uncertainty, dict):
        errors.append("sequence_uncertainty must be a dict")

    priors = payload.get("operator_priors")
    if not isinstance(priors, dict):
        errors.append("operator_priors must be a dict")

    routing = payload.get("routing_signal")
    if not isinstance(routing, dict):
        errors.append("routing_signal must be a dict")
    else:
        if "recommend_guided_evolution" not in routing:
            errors.append("routing_signal.recommend_guided_evolution is required")
        if "reason" not in routing:
            errors.append("routing_signal.reason is required")

    return len(errors) == 0, errors


def _to_float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
