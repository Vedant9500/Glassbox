"""Blackbox preprocessing and feature ranking utilities.

This module is intentionally lightweight: it avoids adding a hard dependency on
tree models or mutual information for the first milestone, and uses robust
correlation plus univariate least-squares probes to reduce multivariate search
space before symbolic evolution.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class BlackboxState:
    enabled: bool
    selected_features: List[int]
    dropped_features: List[int]
    feature_scores: Dict[int, float]
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: float
    y_scale: float
    standardized: bool
    reason: str
    interaction_pairs: List[Tuple[int, int]] = field(default_factory=list)
    interaction_terms: List[str] = field(default_factory=list)
    interaction_scores: Dict[str, float] = field(default_factory=dict)


def _safe_std(values: np.ndarray) -> np.ndarray:
    scale = np.asarray(np.nanstd(values, axis=0), dtype=np.float64)
    scale[~np.isfinite(scale) | (scale < 1e-12)] = 1.0
    return scale


def _corr_score(x: np.ndarray, y: np.ndarray) -> float:
    try:
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return 0.0
        corr = np.corrcoef(x, y)[0, 1]
        return float(abs(corr)) if np.isfinite(corr) else 0.0
    except Exception:
        return 0.0


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def _univariate_poly_score(x: np.ndarray, y: np.ndarray) -> float:
    """Return validation-free relative R2 from a small univariate polynomial probe."""
    try:
        y_var = max(float(np.var(y)), 1e-12)
        cols = [x, x * x, x * x * x, np.ones_like(x)]
        design = np.column_stack(cols)
        coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        pred = design @ coef
        mse = float(np.mean((pred - y) ** 2))
        if not np.isfinite(mse):
            return 0.0
        return float(np.clip(1.0 - mse / y_var, 0.0, 1.0))
    except Exception:
        return 0.0


def rank_blackbox_features(X: np.ndarray, y: np.ndarray) -> Dict[int, float]:
    """Rank features using cheap, deterministic univariate signals."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    scores: Dict[int, float] = {}

    y_rank = _rankdata(y)
    for j in range(X.shape[1]):
        col = X[:, j]
        finite = np.isfinite(col) & np.isfinite(y)
        if int(finite.sum()) < 8:
            scores[j] = 0.0
            continue
        xj = col[finite]
        yj = y[finite]
        pearson = _corr_score(xj, yj)
        spearman = _corr_score(_rankdata(xj), y_rank[finite])
        poly = _univariate_poly_score(xj, yj)
        scores[j] = float(0.35 * pearson + 0.25 * spearman + 0.40 * poly)
    return scores


def discover_blackbox_interactions(
    X: np.ndarray,
    y: np.ndarray,
    selected_features: Optional[List[int]] = None,
    max_pairs: int = 6,
) -> Dict[str, Any]:
    """Score a small set of pairwise interaction candidates."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if X.ndim != 2 or X.shape[1] < 2:
        return {"interaction_pairs": [], "interaction_terms": [], "interaction_scores": {}}

    cols = list(range(X.shape[1]))
    labels = list(selected_features) if selected_features is not None else list(cols)
    if len(cols) < 2:
        return {"interaction_pairs": [], "interaction_terms": [], "interaction_scores": {}}

    y_var = max(float(np.var(y)), 1e-12)
    base_scores = rank_blackbox_features(X, y)
    candidate_rows: List[Tuple[float, Tuple[int, int], str]] = []

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a = cols[i]
            b = cols[j]
            la = labels[i]
            lb = labels[j]
            xi = X[:, a]
            xj = X[:, b]
            mask = np.isfinite(xi) & np.isfinite(xj) & np.isfinite(y)
            if int(mask.sum()) < 12:
                continue

            xi = xi[mask]
            xj = xj[mask]
            yj = y[mask]

            candidates = {
                f"x{la}*x{lb}": xi * xj,
                f"x{la}+x{lb}": xi + xj,
                f"x{la}-x{lb}": xi - xj,
                f"x{la}/(x{lb}+1e-6)": xi / (xj + 1e-6),
                f"x{la}^2+x{lb}^2": xi * xi + xj * xj,
            }

            best_term = None
            best_score = -np.inf
            for name, values in candidates.items():
                if not np.all(np.isfinite(values)):
                    continue
                try:
                    coef, _, _, _ = np.linalg.lstsq(
                        np.column_stack([values, np.ones_like(values)]),
                        yj,
                        rcond=None,
                    )
                    pred = coef[0] * values + coef[1]
                    mse = float(np.mean((pred - yj) ** 2))
                    rel = float(np.clip(1.0 - mse / y_var, 0.0, 1.0))
                    score = rel + 0.15 * max(base_scores.get(a, 0.0), base_scores.get(b, 0.0))
                    if score > best_score:
                        best_score = score
                        best_term = name
                except Exception:
                    continue

            if best_term is not None:
                candidate_rows.append((best_score, (la, lb), best_term))

    candidate_rows.sort(key=lambda item: item[0], reverse=True)
    top = candidate_rows[: max(0, int(max_pairs))]
    return {
        "interaction_pairs": [pair for _, pair, _ in top],
        "interaction_terms": [term for _, _, term in top],
        "interaction_scores": {term: float(score) for score, _, term in top},
    }


def prepare_blackbox_search(
    X: np.ndarray,
    y: np.ndarray,
    *,
    enabled: bool,
    max_features: int = 6,
    standardize: bool = True,
    min_features_to_select: int = 5,
) -> tuple[np.ndarray, np.ndarray, BlackboxState]:
    """Return reduced/standardized search data and remapping metadata."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n_features = int(X.shape[1])

    x_mean = np.nanmean(X, axis=0)
    x_mean = np.where(np.isfinite(x_mean), x_mean, 0.0)
    x_scale = _safe_std(X)
    y_mean = float(np.nanmean(y)) if np.isfinite(np.nanmean(y)) else 0.0
    y_scale = float(np.nanstd(y))
    if not np.isfinite(y_scale) or y_scale < 1e-12:
        y_scale = 1.0

    if not enabled or n_features <= 1:
        state = BlackboxState(
            enabled=False,
            selected_features=list(range(n_features)),
            dropped_features=[],
            feature_scores={},
            x_mean=x_mean,
            x_scale=x_scale,
            y_mean=y_mean,
            y_scale=y_scale,
            standardized=False,
            reason="disabled_or_low_dimensional",
        )
        return X, y, state

    variances = np.nanvar(X, axis=0)
    usable = [j for j in range(n_features) if np.isfinite(variances[j]) and variances[j] > 1e-12]
    if not usable:
        state = BlackboxState(
            enabled=False,
            selected_features=list(range(n_features)),
            dropped_features=[],
            feature_scores={},
            x_mean=x_mean,
            x_scale=x_scale,
            y_mean=y_mean,
            y_scale=y_scale,
            standardized=False,
            reason="no_variable_features",
        )
        return X, y, state

    X_scaled_all = (X - x_mean) / x_scale if standardize else X.copy()
    y_scaled = (y - y_mean) / y_scale if standardize else y.copy()
    if n_features < int(min_features_to_select):
        selected = sorted(usable)
        feature_scores = {idx: 0.0 for idx in usable}
        dropped = [j for j in range(n_features) if j not in selected]
        reason = "retained_all_features_small_problem"
    else:
        scores = rank_blackbox_features(X_scaled_all[:, usable], y_scaled)
        feature_scores = {usable[j]: score for j, score in scores.items()}

        k = int(max(1, min(max_features, len(usable))))
        selected = sorted(usable, key=lambda idx: feature_scores.get(idx, 0.0), reverse=True)[:k]
        selected = sorted(selected)
        dropped = [j for j in range(n_features) if j not in selected]
        reason = "selected_top_features"

    interaction_state = discover_blackbox_interactions(
        X_scaled_all[:, selected],
        y_scaled,
        selected_features=selected,
    )

    state = BlackboxState(
        enabled=True,
        selected_features=selected,
        dropped_features=dropped,
        feature_scores=feature_scores,
        x_mean=x_mean,
        x_scale=x_scale,
        y_mean=y_mean,
        y_scale=y_scale,
        standardized=bool(standardize),
        reason=reason,
        interaction_pairs=list(interaction_state["interaction_pairs"]),
        interaction_terms=list(interaction_state["interaction_terms"]),
        interaction_scores=dict(interaction_state["interaction_scores"]),
    )
    return X_scaled_all[:, selected], y_scaled, state


def remap_reduced_formula_to_original(formula: str, selected_features: List[int]) -> str:
    """Map reduced feature names x0..xk back to original xJ names."""
    if not formula or not selected_features:
        return formula

    def repl(match: re.Match[str]) -> str:
        local_idx = int(match.group(1))
        if 0 <= local_idx < len(selected_features):
            return f"x{int(selected_features[local_idx])}"
        return match.group(0)

    mapped = re.sub(r"\bx(\d+)\b", repl, formula)
    if len(selected_features) == 1:
        mapped = re.sub(r"\bx\b", f"x{int(selected_features[0])}", mapped)
    return mapped


def formula_from_search_to_original_space(formula: str, state: BlackboxState) -> str:
    """Convert a reduced/search-space formula back to original feature names.

    If standardization was used, local variables are expanded as
    ``(x_j - mean_j) / scale_j`` and the target inverse transform is applied.
    """
    if not formula or not state.enabled:
        return formula
    if not state.standardized:
        return remap_reduced_formula_to_original(formula, state.selected_features)

    def feature_expr(local_idx: int) -> str:
        original_idx = int(state.selected_features[local_idx])
        mean = float(state.x_mean[original_idx])
        scale = float(state.x_scale[original_idx])
        return f"((x{original_idx}-{mean:.12g})/{scale:.12g})"

    def repl(match: re.Match[str]) -> str:
        local_idx = int(match.group(1))
        if 0 <= local_idx < len(state.selected_features):
            return feature_expr(local_idx)
        return match.group(0)

    mapped = re.sub(r"\bx(\d+)\b", repl, formula)
    if len(state.selected_features) == 1:
        mapped = re.sub(r"\bx\b", feature_expr(0), mapped)

    return f"({state.y_mean:.12g}+{state.y_scale:.12g}*({mapped}))"


def state_to_dict(state: Optional[BlackboxState]) -> Dict[str, Any]:
    if state is None:
        return {"enabled": False}
    return {
        "enabled": state.enabled,
        "selected_features": list(state.selected_features),
        "dropped_features": list(state.dropped_features),
        "feature_scores": dict(state.feature_scores),
        "standardized": state.standardized,
        "reason": state.reason,
        "interaction_pairs": list(state.interaction_pairs),
        "interaction_terms": list(state.interaction_terms),
        "interaction_scores": dict(state.interaction_scores),
    }
