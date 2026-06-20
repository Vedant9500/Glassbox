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
    hot_spot_segment_scores: List[SpecialistSegmentScore] = field(default_factory=list)

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
            "hot_spot_segment_scores": [segment.to_dict() for segment in self.hot_spot_segment_scores],
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
    hot_spot_segments: List[SpecialistSegment] = field(default_factory=list)
    hot_spot_base_formula: Optional[str] = None

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
            "hot_spot_segments": [segment.to_dict() for segment in self.hot_spot_segments],
            "hot_spot_base_formula": self.hot_spot_base_formula,
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


@dataclass
class SpecialistVaultEntry:
    formula: str
    source: str
    validation_r2: Optional[float]
    validation_mse: Optional[float]
    complexity: int
    family_signature: str
    segment_scores: List[Dict[str, Any]]
    residual_vector: np.ndarray = field(repr=False)
    prediction_vector: np.ndarray = field(repr=False)
    run_index: int = 0
    last_improved_run: int = 0
    residual_relevance: Optional[float] = None

    def to_candidate_dict(self, *, source: str = "specialist_vault") -> Dict[str, Any]:
        return {
            "formula": self.formula,
            "source": source,
            "validation_r2": self.validation_r2,
            "validation_mse": self.validation_mse,
            "mse": self.validation_mse,
            "complexity": self.complexity,
            "family_signature": self.family_signature,
            "from_specialist_vault": True,
            "specialist_vault_run_index": int(self.run_index),
            "specialist_vault_residual_relevance": self.residual_relevance,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "formula": str(self.formula)[:160],
            "source": self.source,
            "validation_r2": self.validation_r2,
            "validation_mse": self.validation_mse,
            "complexity": int(self.complexity),
            "family_signature": self.family_signature,
            "run_index": int(self.run_index),
            "last_improved_run": int(self.last_improved_run),
            "residual_relevance": self.residual_relevance,
            "segment_scores": list(self.segment_scores),
        }


@dataclass
class SpecialistVault:
    max_entries: int = 8
    max_stale_runs: int = 3
    corr_threshold: float = 0.98
    entries: List[SpecialistVaultEntry] = field(default_factory=list)
    added_count: int = 0
    rejected_duplicate_count: int = 0
    evicted_count: int = 0
    composition_count: int = 0

    def clear(self) -> None:
        self.entries.clear()
        self.added_count = 0
        self.rejected_duplicate_count = 0
        self.evicted_count = 0
        self.composition_count = 0

    @staticmethod
    def _formula_key(formula: str) -> str:
        return "".join(str(formula or "").lower().split())

    @staticmethod
    def _prediction_corr(left: np.ndarray, right: np.ndarray) -> float:
        left = np.asarray(left, dtype=np.float64).reshape(-1)
        right = np.asarray(right, dtype=np.float64).reshape(-1)
        if left.size != right.size or left.size < 3:
            return 0.0
        if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
            return 1.0 if np.allclose(left, right, atol=1e-9, rtol=1e-6) else 0.0
        try:
            corr = float(np.corrcoef(left, right)[0, 1])
        except Exception:
            return 0.0
        return corr if np.isfinite(corr) else 0.0

    def _evict_stale(self, run_index: int) -> None:
        kept = []
        for entry in self.entries:
            if int(run_index) - int(entry.last_improved_run) > int(self.max_stale_runs):
                self.evicted_count += 1
            else:
                kept.append(entry)
        self.entries = kept

    def add_candidates(
        self,
        candidate_formulas: Any,
        X: Any,
        y: Any,
        *,
        evaluate_formula: Callable[[str, Any], Any],
        complexity_fn: Callable[[str], int],
        family_signature_fn: Callable[[str], str],
        run_index: int,
        current_best_formula: Optional[str] = None,
        max_new: int = 3,
    ) -> int:
        if not candidate_formulas:
            self._evict_stale(run_index)
            return 0

        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        current_best_key = self._formula_key(current_best_formula or "")
        existing_keys = {self._formula_key(entry.formula) for entry in self.entries}
        y_var = max(float(np.var(y_arr)), 1e-15)

        scored: List[tuple[float, Dict[str, Any], np.ndarray, np.ndarray]] = []
        for candidate in list(candidate_formulas):
            formula = str((candidate or {}).get("formula", "")).strip()
            if not formula:
                continue
            key = self._formula_key(formula)
            if key == current_best_key or key in existing_keys:
                continue
            try:
                pred = np.asarray(evaluate_formula(formula, X_arr), dtype=np.float64).reshape(-1)
            except Exception:
                continue
            if pred.shape != y_arr.shape or not np.all(np.isfinite(pred)):
                continue
            if any(abs(self._prediction_corr(pred, entry.prediction_vector)) >= self.corr_threshold for entry in self.entries):
                self.rejected_duplicate_count += 1
                continue
            residual = pred - y_arr
            mse = _clean_float((candidate or {}).get("validation_mse"))
            if mse is None:
                mse = float(np.mean(residual ** 2))
            r2 = _clean_float((candidate or {}).get("validation_r2"))
            if r2 is None:
                r2 = float(1.0 - mse / y_var)
            complexity = int((candidate or {}).get("complexity") or complexity_fn(formula))
            rank = -float(r2) + 0.002 * float(complexity)
            scored.append((rank, candidate, pred, residual))

        scored.sort(key=lambda item: item[0])
        added = 0
        max_new = max(0, int(max_new))
        for _, candidate, pred, residual in scored:
            if added >= max_new:
                break
            formula = str((candidate or {}).get("formula", "")).strip()
            key = self._formula_key(formula)
            if key in existing_keys:
                continue
            if any(abs(self._prediction_corr(pred, entry.prediction_vector)) >= self.corr_threshold for entry in self.entries):
                self.rejected_duplicate_count += 1
                continue
            segment_scores = []
            for segment in (candidate or {}).get("segment_scores", []) or []:
                if isinstance(segment, dict):
                    segment_scores.append(dict(segment))
            mse = _clean_float((candidate or {}).get("validation_mse"))
            if mse is None:
                mse = float(np.mean(residual ** 2))
            r2 = _clean_float((candidate or {}).get("validation_r2"))
            if r2 is None:
                r2 = float(1.0 - mse / y_var)
            entry = SpecialistVaultEntry(
                formula=formula,
                source=str((candidate or {}).get("source") or "candidate"),
                validation_r2=r2,
                validation_mse=mse,
                complexity=int((candidate or {}).get("complexity") or complexity_fn(formula)),
                family_signature=str((candidate or {}).get("family_signature") or family_signature_fn(formula)),
                segment_scores=segment_scores,
                residual_vector=np.asarray(residual, dtype=np.float64),
                prediction_vector=np.asarray(pred, dtype=np.float64),
                run_index=int(run_index),
                last_improved_run=int(run_index),
            )
            self.entries.append(entry)
            existing_keys.add(key)
            self.added_count += 1
            added += 1

        self.entries.sort(key=lambda entry: (
            float("inf") if entry.validation_mse is None else float(entry.validation_mse),
            int(entry.complexity),
            entry.formula,
        ))
        if len(self.entries) > int(self.max_entries):
            self.evicted_count += len(self.entries) - int(self.max_entries)
            self.entries = self.entries[: int(self.max_entries)]
        self._evict_stale(run_index)
        return int(added)

    def rescore_against_target(
        self,
        X: Any,
        y: Any,
        *,
        evaluate_formula: Callable[[str, Any], Any],
    ) -> None:
        if not self.entries:
            return
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        for entry in self.entries:
            try:
                pred = np.asarray(evaluate_formula(entry.formula, X_arr), dtype=np.float64).reshape(-1)
            except Exception:
                entry.residual_relevance = None
                continue
            if pred.shape != y_arr.shape or not np.all(np.isfinite(pred)):
                entry.residual_relevance = None
                continue
            residual = pred - y_arr
            entry.prediction_vector = pred
            entry.residual_vector = residual
            target_residual = -residual
            entry.residual_relevance = abs(self._prediction_corr(pred, target_residual))

    def candidate_dicts(self) -> List[Dict[str, Any]]:
        return [entry.to_candidate_dict() for entry in self.entries]

    def propose_compositions(
        self,
        X: Any,
        y: Any,
        *,
        evaluate_formula: Callable[[str, Any], Any],
        complexity_fn: Callable[[str], int],
        family_signature_fn: Callable[[str], str],
        current_best_candidate: Optional[Dict[str, Any]] = None,
        max_candidates: int = 6,
    ) -> List[Dict[str, Any]]:
        if not self.entries:
            return []
        raw_candidates = self.candidate_dicts()
        if current_best_candidate and current_best_candidate.get("formula"):
            raw_candidates.insert(0, dict(current_best_candidate))
        state = compute_specialist_state(
            raw_candidates,
            X,
            y,
            evaluate_formula=evaluate_formula,
            complexity_fn=complexity_fn,
            family_signature_fn=family_signature_fn,
            max_candidates=max(2, min(len(raw_candidates), int(max_candidates))),
            max_pairs=4,
        )
        proposals = propose_specialist_compositions(
            state,
            X,
            y,
            evaluate_formula=evaluate_formula,
            max_pairs=4,
            min_complementarity=0.20,
        )
        out = []
        for proposal in proposals[:6]:
            candidate = proposal.to_candidate_dict()
            candidate["source"] = "specialist_vault_composition"
            candidate["from_specialist_vault"] = True
            candidate["from_specialist_composition"] = True
            out.append(candidate)
        self.composition_count += len(out)
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_count": int(len(self.entries)),
            "max_entries": int(self.max_entries),
            "added_count": int(self.added_count),
            "rejected_duplicate_count": int(self.rejected_duplicate_count),
            "evicted_count": int(self.evicted_count),
            "composition_count": int(self.composition_count),
            "entries": [entry.to_dict() for entry in self.entries],
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


def build_hot_spot_segments(
    X: Any,
    best_residual: np.ndarray,
    *,
    max_segments: int = 6,
    min_segment_size: int = 8,
) -> List[SpecialistSegment]:
    """Build hot-spot driven and curvature-aware segments for diagnostic screening."""
    X_arr = np.asarray(X, dtype=np.float64)
    n = int(X_arr.shape[0])
    if n < max(8, int(min_segment_size)):
        return []

    if X_arr.ndim != 2 or X_arr.shape[1] == 0:
        axis_values = np.arange(n, dtype=np.float64)
    elif X_arr.shape[1] == 1:
        axis_values = np.asarray(X_arr[:, 0], dtype=np.float64)
    else:
        centered = X_arr - np.nanmedian(X_arr, axis=0, keepdims=True)
        axis_values = np.linalg.norm(np.where(np.isfinite(centered), centered, 0.0), axis=1)

    axis_values = np.where(np.isfinite(axis_values), axis_values, 0.0)
    order = np.argsort(axis_values, kind="mergesort")
    sorted_x = axis_values[order]

    sorted_res = best_residual[order]
    sq_res = sorted_res**2
    total_sq_res = np.sum(sq_res)

    concentrated_seg = None
    if total_sq_res >= 1e-12:
        # Find shortest slice [i:j] with sum >= 0.7 * total_sq_res and length <= 0.3 * n
        max_len = int(0.3 * n)
        best_slice = None
        if max_len >= min_segment_size:
            cumsum = np.concatenate([[0.0], np.cumsum(sq_res)])
            for L in range(min_segment_size, max_len + 1):
                window_sums = cumsum[L:] - cumsum[:-L]
                max_idx = np.argmax(window_sums)
                if window_sums[max_idx] >= 0.7 * total_sq_res:
                    best_slice = (max_idx, max_idx + L)
                    break
            if best_slice is not None:
                i, j = best_slice
                idx_arr = order[i:j]
                concentrated_seg = SpecialistSegment(
                    segment_index=-1,
                    indices=idx_arr,
                    n_samples=int(idx_arr.size),
                    axis_min=float(np.min(sorted_x[i:j])),
                    axis_max=float(np.max(sorted_x[i:j])),
                )

    # Curvature-aware binning (inflection points of smoothed residuals)
    smoothed_res = sorted_res.copy()
    window_size = 5
    if n >= window_size:
        window = np.ones(window_size) / window_size
        smoothed_res = np.convolve(sorted_res, window, mode="same")
        half = window_size // 2
        smoothed_res[:half] = sorted_res[:half]
        smoothed_res[-half:] = sorted_res[-half:]

    dx = np.diff(sorted_x)
    dx = np.where(dx == 0.0, 1e-5, dx)
    d1 = np.diff(smoothed_res) / dx

    mid_x = (sorted_x[1:] + sorted_x[:-1]) / 2.0
    dmid_x = np.diff(mid_x)
    dmid_x = np.where(dmid_x == 0.0, 1e-5, dmid_x)
    d2 = np.diff(d1) / dmid_x

    inflection_candidates = []
    for k in range(1, len(d2)):
        if d2[k] * d2[k-1] < 0:
            idx_in_sorted = k + 1
            score = abs(d2[k] - d2[k-1])
            inflection_candidates.append((idx_in_sorted, score))

    inflection_candidates.sort(key=lambda x: -x[1])
    top_candidates = [item[0] for item in inflection_candidates[:15]]
    top_candidates.sort()

    splits = [0]
    for p in top_candidates:
        if p - splits[-1] >= min_segment_size and n - p >= min_segment_size:
            splits.append(p)
            if len(splits) - 1 >= max_segments - 1:
                break
    splits.append(n)

    curvature_segments = []
    for idx in range(len(splits) - 1):
        i, j = splits[idx], splits[idx+1]
        idx_arr = order[i:j]
        curvature_segments.append(
            SpecialistSegment(
                segment_index=idx,
                indices=idx_arr,
                n_samples=int(idx_arr.size),
                axis_min=float(np.min(sorted_x[i:j])),
                axis_max=float(np.max(sorted_x[i:j])),
            )
        )

    all_hs_segments = []
    seg_idx = 0
    if concentrated_seg is not None:
        concentrated_seg.segment_index = seg_idx
        all_hs_segments.append(concentrated_seg)
        seg_idx += 1

    for seg in curvature_segments:
        seg.segment_index = seg_idx
        all_hs_segments.append(seg)
        seg_idx += 1

    return all_hs_segments[:max_segments]


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

    # First, evaluate and create temporary candidate info
    temp_candidates = []
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

        temp_candidates.append({
            "formula": formula,
            "candidate": candidate,
            "pred": pred,
            "residual": residual,
            "segment_scores": segment_scores,
            "best_segment": best_segment,
            "worst_segment": worst_segment,
        })

    if not temp_candidates:
        return None

    def _candidate_rank(item: Dict[str, Any]) -> tuple:
        candidate = item.get("candidate") or {}
        val_mse = _clean_float(candidate.get("validation_mse"))
        val_r2 = _clean_float(candidate.get("validation_r2"))
        if val_mse is None:
            val_mse = float(np.mean(np.asarray(item["residual"], dtype=np.float64) ** 2))
        if val_r2 is None:
            val_r2 = -float("inf")
        return (float(val_mse), -float(val_r2))

    best_temp_candidate = min(temp_candidates, key=_candidate_rank)
    best_residual = best_temp_candidate["residual"]
    best_formula_for_hot_spots = str(best_temp_candidate["formula"])

    # Build hot-spot segments using the best candidate's residual
    hot_spot_segments = build_hot_spot_segments(
        X_arr,
        best_residual,
        max_segments=6,
        min_segment_size=8,
    )

    # Now build SpecialistCandidate objects
    candidates: List[SpecialistCandidate] = []
    for tc in temp_candidates:
        pred = tc["pred"]
        formula = tc["formula"]
        candidate = tc["candidate"]

        # Compute scores on hot_spot_segments
        hs_segment_scores: List[SpecialistSegmentScore] = []
        for hs_seg in hot_spot_segments:
            idx = hs_seg.indices
            y_seg = y_arr[idx]
            pred_seg = pred[idx]
            mse = float(np.mean((pred_seg - y_seg) ** 2))
            y_var = float(np.var(y_seg))
            r2 = 1.0 if y_var < 1e-15 and mse < 1e-15 else (0.0 if y_var < 1e-15 else 1.0 - mse / y_var)
            hs_segment_scores.append(
                SpecialistSegmentScore(
                    segment_index=int(hs_seg.segment_index),
                    n_samples=int(hs_seg.n_samples),
                    mse=mse,
                    r2=float(r2),
                    axis_min=float(hs_seg.axis_min),
                    axis_max=float(hs_seg.axis_max),
                )
            )

        candidates.append(
            SpecialistCandidate(
                formula=formula,
                source=infer_specialist_source(candidate),
                validation_r2=_clean_float((candidate or {}).get("validation_r2")),
                validation_mse=_clean_float((candidate or {}).get("validation_mse")),
                complexity=int((candidate or {}).get("complexity") or complexity_fn(formula)),
                family_signature=str(family_signature_fn(formula)),
                segment_scores=tc["segment_scores"],
                best_segment={
                    "segment_index": int(tc["best_segment"].segment_index),
                    "r2": float(tc["best_segment"].r2),
                },
                worst_segment={
                    "segment_index": int(tc["worst_segment"].segment_index),
                    "r2": float(tc["worst_segment"].r2),
                },
                residual_vector=tc["residual"],
                hot_spot_segment_scores=hs_segment_scores,
            )
        )
    best_candidate_for_hot_spots = next(
        (candidate for candidate in candidates if candidate.formula == best_formula_for_hot_spots),
        candidates[0],
    )

    pair_scores: List[SpecialistPairScore] = []
    for left_idx in range(len(candidates)):
        for right_idx in range(left_idx + 1, len(candidates)):
            left = candidates[left_idx]
            right = candidates[right_idx]

            # Standard segment complementarity
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

            split_score = min(left_wins, right_wins) / max(1.0, float(len(left.segment_scores)))
            switch_score = segment_switches / max(1.0, float(len(left.segment_scores) - 1))
            margin_score = segment_margin_sum / max(1.0, float(len(left.segment_scores)))

            # Residual correlation
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
            residual_disagreement = 1.0 - abs(float(np.clip(residual_corr, -1.0, 1.0)))

            comp_std = float(np.clip(
                0.45 * split_score
                + 0.20 * switch_score
                + 0.20 * min(1.0, margin_score)
                + 0.15 * residual_disagreement,
                0.0,
                1.0,
            ))

            # Hot-spot segment complementarity
            comp_hs = 0.0
            hs_left_wins = 0
            hs_right_wins = 0
            hs_segment_switches = 0
            hs_prev_winner = None
            hs_segment_margin_sum = 0.0

            if len(hot_spot_segments) >= 2:
                for seg_l, seg_r in zip(left.hot_spot_segment_scores, right.hot_spot_segment_scores):
                    if seg_l.mse + 1e-12 < seg_r.mse:
                        winner = 0
                        hs_left_wins += 1
                    elif seg_r.mse + 1e-12 < seg_l.mse:
                        winner = 1
                        hs_right_wins += 1
                    else:
                        winner = -1
                    if hs_prev_winner is not None and winner != -1 and hs_prev_winner != -1 and winner != hs_prev_winner:
                        hs_segment_switches += 1
                    if winner != -1:
                        hs_prev_winner = winner
                    denom = max(seg_l.mse, seg_r.mse, 1e-12)
                    hs_segment_margin_sum += abs(seg_l.mse - seg_r.mse) / denom

                hs_split_score = min(hs_left_wins, hs_right_wins) / max(1.0, float(len(hot_spot_segments)))
                hs_switch_score = hs_segment_switches / max(1.0, float(len(hot_spot_segments) - 1))
                hs_margin_score = hs_segment_margin_sum / max(1.0, float(len(hot_spot_segments)))

                comp_hs = float(np.clip(
                    0.45 * hs_split_score
                    + 0.20 * hs_switch_score
                    + 0.20 * min(1.0, hs_margin_score)
                    + 0.15 * residual_disagreement,
                    0.0,
                    1.0,
                ))

            # Excel-on-hot-spot bonus
            hs_excel_bonus = 0.0
            total_best_mse = float(np.mean(best_candidate_for_hot_spots.residual_vector ** 2))
            if total_best_mse >= 1e-12 and len(hot_spot_segments) > 0:
                for seg_idx, (seg_l, seg_r) in enumerate(zip(left.hot_spot_segment_scores, right.hot_spot_segment_scores)):
                    seg_best = best_candidate_for_hot_spots.hot_spot_segment_scores[seg_idx]
                    if seg_best.mse > 1.2 * total_best_mse:
                        if min(seg_l.mse, seg_r.mse) < 0.7 * seg_best.mse:
                            hs_excel_bonus += 0.10
                hs_excel_bonus = min(0.20, hs_excel_bonus)

            # Combined score
            if len(hot_spot_segments) >= 2:
                complementarity = 0.5 * comp_std + 0.5 * comp_hs + hs_excel_bonus
            else:
                complementarity = comp_std + hs_excel_bonus
            complementarity = float(np.clip(complementarity, 0.0, 1.0))

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
        hot_spot_segments=hot_spot_segments,
        hot_spot_base_formula=best_formula_for_hot_spots,
    )


def nest_formulas(f: str, g: str) -> str:
    """Replace variable (e.g. x0, x1, x) in f with (g)."""
    import re
    return re.sub(r"\bx\d*\b", f"({g})", f)


def _dedupe_append(forms: List[tuple[str, str]], operator: str, formula: str) -> None:
    key = (operator, "".join(str(formula).lower().split()))
    if key not in {(_op, "".join(str(_formula).lower().split())) for _op, _formula in forms}:
        forms.append((operator, formula))


def propose_specialist_compositions(
    state: Optional[SpecialistState],
    X: Any = None,
    y: Any = None,
    *,
    evaluate_formula: Optional[Callable[[str, Any], Any]] = None,
    max_pairs: int = 3,
    min_complementarity: float = 0.30,
) -> List[SpecialistCompositionProposal]:
    """Generate a tiny set of composition proposals from the best specialist pairs."""
    if state is None or not state.enabled:
        return []

    import re
    proposals: List[SpecialistCompositionProposal] = []
    seen = set()

    # Simple local eval helper if evaluate_formula is not provided
    def _local_eval(expr_str: str, X_val: np.ndarray) -> np.ndarray:
        context = {
            "np": np,
            "sin": np.sin,
            "cos": np.cos,
            "exp": np.exp,
            "log": np.log,
            "sqrt": np.sqrt,
            "abs": np.abs,
        }
        X_val = np.asarray(X_val, dtype=np.float64)
        for idx in range(X_val.shape[1]):
            context[f"x{idx}"] = X_val[:, idx]
        if X_val.shape[1] == 1:
            context["x"] = X_val[:, 0]
        cleaned_expr = str(expr_str).replace("^", "**")
        return np.asarray(eval(cleaned_expr, {"__builtins__": None}, context), dtype=np.float64)

    eval_fn = evaluate_formula if evaluate_formula is not None else _local_eval

    # We map formula strings to candidates for quick retrieval of residual_vector
    cand_map = {c.formula: c for c in state.candidates}

    for pair in list(state.top_pairs)[: max(0, int(max_pairs))]:
        if float(pair.complementarity_score) < float(min_complementarity):
            continue

        left = cand_map.get(pair.formula_a)
        right = cand_map.get(pair.formula_b)
        if left is None or right is None:
            continue

        # Get evaluated predictions if y is provided
        pred_f, pred_g = None, None
        if y is not None:
            y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
            pred_f = left.residual_vector + y_arr
            pred_g = right.residual_vector + y_arr

        forms: List[tuple[str, str]] = []
        _dedupe_append(forms, "add", f"(({pair.formula_a})+({pair.formula_b}))")
        _dedupe_append(forms, "mul", f"(({pair.formula_a})*({pair.formula_b}))")

        # 1. Division: f / g and g / f, guarded by the denominator prediction.
        if y is not None and pred_f is not None and pred_g is not None:
            if np.min(np.abs(pred_g)) > 0.01:
                _dedupe_append(forms, "div", f"(({pair.formula_a})/({pair.formula_b}))")
            if np.min(np.abs(pred_f)) > 0.01:
                _dedupe_append(forms, "div", f"(({pair.formula_b})/({pair.formula_a}))")
        else:
            _dedupe_append(forms, "div", f"(({pair.formula_a})/({pair.formula_b}))")
            _dedupe_append(forms, "div", f"(({pair.formula_b})/({pair.formula_a}))")

        # 2. Nested: f(g)
        def _maybe_add_nested(outer_formula: str, inner_formula: str, outer_family: str, inner_pred: Optional[np.ndarray]) -> None:
            if outer_family not in {"sin", "cos", "exp", "log"}:
                return
            if y is not None and inner_pred is not None:
                std_inner = np.std(inner_pred)
                range_inner = np.max(inner_pred) - np.min(inner_pred)
                if std_inner <= 1e-5 or range_inner >= 20.0:
                    return
                if outer_family == "exp" and np.max(inner_pred) >= 5.0:
                    return
                if outer_family == "log" and np.min(inner_pred) <= 0.01:
                    return
            _dedupe_append(forms, "nested", nest_formulas(outer_formula, inner_formula))

        _maybe_add_nested(pair.formula_a, pair.formula_b, pair.family_a, pred_g)
        _maybe_add_nested(pair.formula_b, pair.formula_a, pair.family_b, pred_f)

        # 3. Affine blend: a*f + (1-a)*g
        if y is not None and pred_f is not None and pred_g is not None:
            diff = pred_f - pred_g
            denom = np.sum(diff**2)
            alpha = np.sum(diff * (y_arr - pred_g)) / denom if denom > 1e-9 else 0.5
            if 0.05 < float(alpha) < 0.95:
                forms.append(("affine", f"(({alpha:.6g})*({pair.formula_a})) + (((1.0 - {alpha:.6g}))*({pair.formula_b}))"))
        else:
            forms.append(("affine", f"((0.5)*({pair.formula_a})) + (((1.0 - 0.5))*({pair.formula_b}))"))

        # 4. Damped product: f * exp(-beta * g^2)
        def _maybe_add_damped(base_formula: str, damp_formula: str, base_pred: np.ndarray, damp_pred: np.ndarray) -> None:
            betas = np.logspace(-3, 2, 15)
            best_beta = 0.1
            best_mse = float('inf')
            for b in betas:
                pred_hs = base_pred * np.exp(-b * (damp_pred**2))
                mse = np.mean((pred_hs - y_arr)**2)
                if mse < best_mse:
                    best_mse = mse
                    best_beta = b
            final_damp = np.exp(-best_beta * (damp_pred**2))
            if np.mean(final_damp < 1e-3) <= 0.5:
                _dedupe_append(forms, "damped_product", f"(({base_formula}) * exp((-{best_beta:.6g}) * (({damp_formula})**2)))")

        if y is not None and pred_f is not None and pred_g is not None:
            _maybe_add_damped(pair.formula_a, pair.formula_b, pred_f, pred_g)
            _maybe_add_damped(pair.formula_b, pair.formula_a, pred_g, pred_f)
        else:
            forms.append(("damped_product", f"(({pair.formula_a}) * exp((-1.0) * (({pair.formula_b})**2)))"))
            forms.append(("damped_product", f"(({pair.formula_b}) * exp((-1.0) * (({pair.formula_a})**2)))"))

        # 5. Sigmoid gate: f * sig + g * (1 - sig)
        if y is not None and X is not None and pred_f is not None and pred_g is not None:
            X_arr = np.asarray(X, dtype=np.float64)
            if X_arr.ndim != 2 or X_arr.shape[1] == 0:
                t = np.arange(len(y_arr), dtype=np.float64)
            elif X_arr.shape[1] == 1:
                t = np.asarray(X_arr[:, 0], dtype=np.float64)
            else:
                centered = X_arr - np.nanmedian(X_arr, axis=0, keepdims=True)
                t = np.linalg.norm(np.where(np.isfinite(centered), centered, 0.0), axis=1)

            if X_arr.shape[1] == 1:
                gate_var = "x0"
            else:
                best_corr = -1.0
                best_feat = 0
                for i in range(X_arr.shape[1]):
                    corr = abs(float(np.corrcoef(X_arr[:, i], t)[0, 1]))
                    if np.isfinite(corr) and corr > best_corr:
                        best_corr = corr
                        best_feat = i
                gate_var = f"x{best_feat}"

            c_candidates = np.percentile(t, [20, 30, 40, 50, 60, 70, 80])
            k_candidates = [-10.0, -5.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 5.0, 10.0]

            best_k, best_c = 1.0, np.median(t)
            best_mse = float('inf')
            for c_val in c_candidates:
                for k_val in k_candidates:
                    arg = -k_val * (t - c_val)
                    arg = np.clip(arg, -50.0, 50.0)
                    sig = 1.0 / (1.0 + np.exp(arg))
                    pred_hs = pred_f * sig + pred_g * (1.0 - sig)
                    mse = np.mean((pred_hs - y_arr)**2)
                    if mse < best_mse:
                        best_mse = mse
                        best_k = k_val
                        best_c = c_val

            parent_best_mse = min(
                float(np.mean((pred_f - y_arr) ** 2)),
                float(np.mean((pred_g - y_arr) ** 2)),
            )
            if best_mse + 1e-12 < parent_best_mse:
                forms.append(("sigmoid_gate", f"(({pair.formula_a}) * (1.0 / (1.0 + exp((-{best_k:.6g}) * ({gate_var} - ({best_c:.6g})))))) + (({pair.formula_b}) * (1.0 - (1.0 / (1.0 + exp((-{best_k:.6g}) * ({gate_var} - ({best_c:.6g})))))))"))
        else:
            forms.append(("sigmoid_gate", f"(({pair.formula_a}) * (1.0 / (1.0 + exp((-1.0) * (x0 - (0.0)))))) + (({pair.formula_b}) * (1.0 - (1.0 / (1.0 + exp((-1.0) * (x0 - (0.0)))))))"))

        # Grade templates by training MSE and complexity
        candidate_templates = []
        for operator, formula in forms:
            if y is not None and X is not None:
                try:
                    pred_comp = eval_fn(formula, X_arr)
                    pred_comp = np.asarray(pred_comp, dtype=np.float64).reshape(-1)
                    if pred_comp.shape == y_arr.shape and np.all(np.isfinite(pred_comp)):
                        mse_val = np.mean((pred_comp - y_arr)**2)
                    else:
                        mse_val = float('inf')
                except Exception:
                    mse_val = float('inf')
            else:
                mse_val = 0.0

            # Reject too complex templates
            comp_f = len(pair.formula_a)
            comp_g = len(pair.formula_b)
            if left is not None and right is not None:
                comp_f = left.complexity
                comp_g = right.complexity

            op_extra = 1
            if operator == "div":
                op_extra = 3
            elif operator == "nested":
                op_extra = 2
            elif operator == "affine":
                op_extra = 5
            elif operator == "damped_product":
                op_extra = 6
            elif operator == "sigmoid_gate":
                op_extra = 12

            total_comp = comp_f + comp_g + op_extra
            simpler_comp = min(comp_f, comp_g)
            complexity_limit = max(15, 2.0 * simpler_comp)
            if operator == "sigmoid_gate":
                complexity_limit = max(30, 4.0 * simpler_comp)
            elif operator in {"damped_product", "affine"}:
                complexity_limit = max(20, 3.0 * simpler_comp)
            if total_comp > complexity_limit:
                continue

            candidate_templates.append((operator, formula, mse_val, total_comp))

        # Filter out invalid
        candidate_templates = [ct for ct in candidate_templates if ct[2] != float('inf')]
        # Preserve operator diversity before validation. MSE still breaks ties within each template family.
        priority = {
            "nested": 0,
            "div": 1,
            "sigmoid_gate": 2,
            "damped_product": 3,
            "affine": 4,
            "mul": 5,
            "add": 6,
        }
        candidate_templates.sort(key=lambda ct: (priority.get(ct[0], 99), ct[2], ct[3]))

        selected_templates = []
        selected_ops = set()
        for legacy_op in ("add", "mul"):
            legacy_candidates = [ct for ct in candidate_templates if ct[0] == legacy_op]
            if legacy_candidates:
                selected_templates.append(min(legacy_candidates, key=lambda ct: (ct[2], ct[3])))
                selected_ops.add(legacy_op)
        for ct in candidate_templates:
            if ct[0] in selected_ops:
                continue
            selected_templates.append(ct)
            selected_ops.add(ct[0])
            if len(selected_templates) >= 3:
                break

        for operator, formula, _, _ in selected_templates:
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

    return proposals[:12]
