"""Shared specialist-screening state and diagnostics helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np


def _clean_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None


@dataclass
class SpecialistSegment:
    segment_index: int
    n_samples: int
    axis_min: float
    axis_max: float
    indices: np.ndarray = field(repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_index": int(self.segment_index),
            "n_samples": int(self.n_samples),
            "axis_min": float(self.axis_min),
            "axis_max": float(self.axis_max),
        }


@dataclass
class SpecialistSegmentScore:
    segment_index: int
    n_samples: int
    mse: float
    r2: float
    axis_min: float
    axis_max: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_index": int(self.segment_index),
            "n_samples": int(self.n_samples),
            "mse": float(self.mse),
            "r2": float(self.r2),
            "axis_min": float(self.axis_min),
            "axis_max": float(self.axis_max),
        }


@dataclass
class SpecialistCandidate:
    formula: str
    source: str
    validation_r2: Optional[float]
    validation_mse: Optional[float]
    complexity: int
    family_signature: str
    segment_scores: List[SpecialistSegmentScore]
    best_segment: Dict[str, Any]
    worst_segment: Dict[str, Any]
    residual_vector: np.ndarray = field(repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "formula": str(self.formula)[:160],
            "source": self.source,
            "validation_r2": self.validation_r2,
            "validation_mse": self.validation_mse,
            "complexity": int(self.complexity),
            "family_signature": self.family_signature,
            "best_segment": dict(self.best_segment),
            "worst_segment": dict(self.worst_segment),
            "segment_scores": [segment.to_dict() for segment in self.segment_scores],
        }


@dataclass
class SpecialistPairScore:
    formula_a: str
    formula_b: str
    source_a: str
    source_b: str
    family_a: str
    family_b: str
    left_segment_wins: int
    right_segment_wins: int
    segment_switches: int
    residual_correlation: float
    complementarity_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "formula_a": str(self.formula_a)[:160],
            "formula_b": str(self.formula_b)[:160],
            "source_a": self.source_a,
            "source_b": self.source_b,
            "family_a": self.family_a,
            "family_b": self.family_b,
            "left_segment_wins": int(self.left_segment_wins),
            "right_segment_wins": int(self.right_segment_wins),
            "segment_switches": int(self.segment_switches),
            "residual_correlation": float(self.residual_correlation),
            "complementarity_score": float(self.complementarity_score),
        }


@dataclass
class SpecialistState:
    enabled: bool
    segment_axis: str
    segments: List[SpecialistSegment]
    candidates: List[SpecialistCandidate]
    top_pairs: List[SpecialistPairScore]

    @property
    def candidate_count(self) -> int:
        return int(len(self.candidates))

    @property
    def segment_count(self) -> int:
        return int(len(self.segments))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "candidate_count": self.candidate_count,
            "segment_axis": self.segment_axis,
            "segment_count": self.segment_count,
            "segments": [segment.to_dict() for segment in self.segments],
            "top_candidates": [candidate.to_dict() for candidate in self.candidates],
            "top_pairs": [pair.to_dict() for pair in self.top_pairs],
        }


@dataclass
class SpecialistCompositionProposal:
    formula: str
    operator: str
    formula_a: str
    formula_b: str
    source_a: str
    source_b: str
    family_a: str
    family_b: str
    complementarity_score: float

    def to_candidate_dict(self) -> Dict[str, Any]:
        return {
            "formula": self.formula,
            "source": "specialist_composition",
            "composition_operator": self.operator,
            "composition_parent_a": self.formula_a,
            "composition_parent_b": self.formula_b,
            "composition_source_a": self.source_a,
            "composition_source_b": self.source_b,
            "composition_family_a": self.family_a,
            "composition_family_b": self.family_b,
            "composition_score": float(self.complementarity_score),
            "from_specialist_composition": True,
        }


def build_specialist_segment_slices(
    X: Any,
    *,
    max_segments: int = 4,
    min_segment_size: int = 8,
) -> Optional[tuple[str, List[SpecialistSegment]]]:
    """Build coarse contiguous diagnostic segments over the current search space."""
    X_arr = np.asarray(X, dtype=np.float64)
    n = int(X_arr.shape[0])
    if n < max(8, int(min_segment_size)):
        return None

    if X_arr.ndim != 2 or X_arr.shape[1] == 0:
        axis_values = np.arange(n, dtype=np.float64)
        axis_name = "index"
    elif X_arr.shape[1] == 1:
        axis_values = np.asarray(X_arr[:, 0], dtype=np.float64)
        axis_name = "x0"
    else:
        centered = X_arr - np.nanmedian(X_arr, axis=0, keepdims=True)
        axis_values = np.linalg.norm(np.where(np.isfinite(centered), centered, 0.0), axis=1)
        axis_name = "radius"

    axis_values = np.where(np.isfinite(axis_values), axis_values, 0.0)
    order = np.argsort(axis_values, kind="mergesort")
    n_segments = int(min(max_segments, max(2, n // max(int(min_segment_size), 1))))
    if n_segments < 2:
        return None

    raw_slices = [chunk for chunk in np.array_split(order, n_segments) if int(len(chunk)) >= int(min_segment_size)]
    if len(raw_slices) < 2:
        return None

    segments: List[SpecialistSegment] = []
    for idx, chunk in enumerate(raw_slices):
        idx_arr = np.asarray(chunk, dtype=int)
        if idx_arr.size == 0:
            continue
        chunk_axis = axis_values[idx_arr]
        segments.append(
            SpecialistSegment(
                segment_index=int(idx),
                indices=idx_arr,
                n_samples=int(idx_arr.size),
                axis_min=float(np.min(chunk_axis)),
                axis_max=float(np.max(chunk_axis)),
            )
        )
    if len(segments) < 2:
        return None
    return axis_name, segments


def infer_specialist_source(candidate: Dict[str, Any]) -> str:
    """Normalize the source tag for a candidate formula."""
    if not isinstance(candidate, dict):
        return "candidate"
    explicit = str(candidate.get("source") or "").strip()
    if explicit:
        return explicit
    if candidate.get("from_basis_model"):
        return "basis_model"
    if candidate.get("from_fast_path"):
        return "fast_path"
    if candidate.get("from_proposer"):
        return "proposer"
    if candidate.get("from_blackbox_interaction"):
        return "blackbox_interaction"
    if candidate.get("from_blackbox_seed"):
        return "blackbox_seed"
    return "candidate"


def compute_specialist_state(
    candidate_formulas: Any,
    X: Any,
    y: Any,
    *,
    evaluate_formula: Callable[[str, Any], Any],
    complexity_fn: Callable[[str], int],
    family_signature_fn: Callable[[str], str],
    max_candidates: int = 6,
    max_pairs: int = 5,
) -> Optional[SpecialistState]:
    """Summarize coarse segment behavior and pair complementarity for top candidates."""
    if not candidate_formulas:
        return None

    built = build_specialist_segment_slices(X, max_segments=4, min_segment_size=8)
    if not built:
        return None
    axis_name, segments = built

    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    if y_arr.shape[0] != int(X_arr.shape[0]):
        return None

    candidates: List[SpecialistCandidate] = []
    for candidate in list(candidate_formulas)[: max(1, int(max_candidates))]:
        formula = str((candidate or {}).get("formula", "")).strip()
        if not formula:
            continue
        try:
            pred = evaluate_formula(formula, X_arr)
        except Exception:
            continue
        pred = np.asarray(pred, dtype=np.float64).reshape(-1)
        if pred.shape != y_arr.shape or not np.all(np.isfinite(pred)):
            continue

        residual = pred - y_arr
        segment_scores: List[SpecialistSegmentScore] = []
        for segment in segments:
            idx = segment.indices
            y_seg = y_arr[idx]
            pred_seg = pred[idx]
            mse = float(np.mean((pred_seg - y_seg) ** 2))
            y_var = float(np.var(y_seg))
            r2 = 1.0 if y_var < 1e-15 and mse < 1e-15 else (0.0 if y_var < 1e-15 else 1.0 - mse / y_var)
            segment_scores.append(
                SpecialistSegmentScore(
                    segment_index=int(segment.segment_index),
                    n_samples=int(segment.n_samples),
                    mse=mse,
                    r2=float(r2),
                    axis_min=float(segment.axis_min),
                    axis_max=float(segment.axis_max),
                )
            )

        best_segment = max(segment_scores, key=lambda item: (item.r2, -item.mse))
        worst_segment = min(segment_scores, key=lambda item: (item.r2, item.mse))
        candidates.append(
            SpecialistCandidate(
                formula=formula,
                source=infer_specialist_source(candidate),
                validation_r2=_clean_float((candidate or {}).get("validation_r2")),
                validation_mse=_clean_float((candidate or {}).get("validation_mse")),
                complexity=int((candidate or {}).get("complexity") or complexity_fn(formula)),
                family_signature=str(family_signature_fn(formula)),
                segment_scores=segment_scores,
                best_segment={
                    "segment_index": int(best_segment.segment_index),
                    "r2": float(best_segment.r2),
                },
                worst_segment={
                    "segment_index": int(worst_segment.segment_index),
                    "r2": float(worst_segment.r2),
                },
                residual_vector=residual,
            )
        )

    if not candidates:
        return None

    pair_scores: List[SpecialistPairScore] = []
    for left_idx in range(len(candidates)):
        for right_idx in range(left_idx + 1, len(candidates)):
            left = candidates[left_idx]
            right = candidates[right_idx]
            if len(left.segment_scores) != len(right.segment_scores):
                continue

            left_wins = 0
            right_wins = 0
            segment_switches = 0
            prev_winner = None
            segment_margin_sum = 0.0
            for seg_l, seg_r in zip(left.segment_scores, right.segment_scores):
                if seg_l.mse + 1e-12 < seg_r.mse:
                    winner = 0
                    left_wins += 1
                elif seg_r.mse + 1e-12 < seg_l.mse:
                    winner = 1
                    right_wins += 1
                else:
                    winner = -1
                if prev_winner is not None and winner != -1 and prev_winner != -1 and winner != prev_winner:
                    segment_switches += 1
                if winner != -1:
                    prev_winner = winner
                denom = max(seg_l.mse, seg_r.mse, 1e-12)
                segment_margin_sum += abs(seg_l.mse - seg_r.mse) / denom

            residual_corr = 0.0
            left_res = np.asarray(left.residual_vector, dtype=np.float64)
            right_res = np.asarray(right.residual_vector, dtype=np.float64)
            if left_res.size >= 8 and right_res.size == left_res.size:
                try:
                    if np.std(left_res) > 1e-12 and np.std(right_res) > 1e-12:
                        residual_corr = float(np.corrcoef(left_res, right_res)[0, 1])
                        if not np.isfinite(residual_corr):
                            residual_corr = 0.0
                except Exception:
                    residual_corr = 0.0

            split_score = min(left_wins, right_wins) / max(1.0, float(len(left.segment_scores)))
            switch_score = segment_switches / max(1.0, float(len(left.segment_scores) - 1))
            margin_score = segment_margin_sum / max(1.0, float(len(left.segment_scores)))
            residual_disagreement = 1.0 - abs(float(np.clip(residual_corr, -1.0, 1.0)))
            complementarity = float(np.clip(
                0.45 * split_score
                + 0.20 * switch_score
                + 0.20 * min(1.0, margin_score)
                + 0.15 * residual_disagreement,
                0.0,
                1.0,
            ))
            pair_scores.append(
                SpecialistPairScore(
                    formula_a=left.formula,
                    formula_b=right.formula,
                    source_a=left.source,
                    source_b=right.source,
                    family_a=left.family_signature,
                    family_b=right.family_signature,
                    left_segment_wins=int(left_wins),
                    right_segment_wins=int(right_wins),
                    segment_switches=int(segment_switches),
                    residual_correlation=float(residual_corr),
                    complementarity_score=complementarity,
                )
            )

    pair_scores.sort(
        key=lambda item: (
            -float(item.complementarity_score),
            -int(item.segment_switches),
            float(abs(item.residual_correlation)),
            item.formula_a,
            item.formula_b,
        )
    )

    return SpecialistState(
        enabled=True,
        segment_axis=axis_name,
        segments=segments,
        candidates=candidates,
        top_pairs=pair_scores[: max(0, int(max_pairs))],
    )


def propose_specialist_compositions(
    state: Optional[SpecialistState],
    *,
    max_pairs: int = 3,
    min_complementarity: float = 0.30,
) -> List[SpecialistCompositionProposal]:
    """Generate a tiny set of composition proposals from the best specialist pairs."""
    if state is None or not state.enabled:
        return []

    proposals: List[SpecialistCompositionProposal] = []
    seen = set()
    for pair in list(state.top_pairs)[: max(0, int(max_pairs))]:
        if float(pair.complementarity_score) < float(min_complementarity):
            continue

        forms = [
            ("add", f"(({pair.formula_a})+({pair.formula_b}))"),
            ("mul", f"(({pair.formula_a})*({pair.formula_b}))"),
        ]
        for operator, formula in forms:
            key = "".join(str(formula).lower().split())
            if key in seen:
                continue
            seen.add(key)
            proposals.append(
                SpecialistCompositionProposal(
                    formula=formula,
                    operator=operator,
                    formula_a=pair.formula_a,
                    formula_b=pair.formula_b,
                    source_a=pair.source_a,
                    source_b=pair.source_b,
                    family_a=pair.family_a,
                    family_b=pair.family_b,
                    complementarity_score=float(pair.complementarity_score),
                )
            )
    return proposals
