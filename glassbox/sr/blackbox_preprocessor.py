"""Blackbox preprocessing and feature ranking utilities.

This module is intentionally lightweight: it avoids adding a hard dependency on
tree models or mutual information for the first milestone, and uses robust
correlation plus univariate least-squares probes to reduce multivariate search
space before symbolic evolution.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.linear_model import ElasticNetCV, LassoCV
except Exception:  # pragma: no cover - optional sklearn helpers
    ExtraTreesRegressor = None
    mutual_info_regression = None
    ElasticNetCV = None
    LassoCV = None


@dataclass
class BlackboxState:
    enabled: bool
    selected_features: List[int]
    dropped_features: List[int]
    feature_scores: Dict[int, float]
    ranker_votes: Dict[str, Dict[int, float]]
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: float
    y_scale: float
    standardized: bool
    reason: str
    interaction_pairs: List[Tuple[int, int]] = field(default_factory=list)
    interaction_terms: List[str] = field(default_factory=list)
    interaction_scores: Dict[str, float] = field(default_factory=dict)
    feature_selection_uncertain: bool = False
    candidate_seed_formulas: List[str] = field(default_factory=list)


def _safe_std(values: np.ndarray) -> np.ndarray:
    finite_values = np.where(np.isfinite(values), values, np.nan)
    scale = np.asarray(np.nanstd(finite_values, axis=0), dtype=np.float64)
    scale[~np.isfinite(scale) | (scale < 1e-12)] = 1.0
    return scale


def _as_sample_weight(sample_weight, n: int) -> Optional[np.ndarray]:
    """Validate optional per-point weights; return length-n array or None."""
    if sample_weight is None:
        return None
    try:
        w = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if w.shape[0] != int(n) or not np.all(np.isfinite(w)) or np.any(w < 0.0):
        return None
    if float(np.sum(w)) <= 1e-15:
        return None
    return w


def _weighted_mean(values: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return 0.0
    if weights is None:
        return float(np.mean(values))
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    total = float(np.sum(w))
    if total <= 1e-15:
        return float(np.mean(values))
    return float(np.dot(w, values) / total)


def _weighted_var(values: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return 0.0
    mu = _weighted_mean(values, weights)
    centered = values - mu
    if weights is None:
        return float(np.mean(centered * centered))
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    total = float(np.sum(w))
    if total <= 1e-15:
        return float(np.mean(centered * centered))
    return float(np.dot(w, centered * centered) / total)


def _corr_score(x: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> float:
    try:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        w = _as_sample_weight(sample_weight, x.shape[0])
        if w is None:
            if np.std(x) < 1e-12 or np.std(y) < 1e-12:
                return 0.0
            corr = np.corrcoef(x, y)[0, 1]
            return float(abs(corr)) if np.isfinite(corr) else 0.0
        mx = _weighted_mean(x, w)
        my = _weighted_mean(y, w)
        xc = x - mx
        yc = y - my
        denom = math.sqrt(max(_weighted_var(x, w), 0.0) * max(_weighted_var(y, w), 0.0))
        if denom < 1e-15:
            return 0.0
        num = float(np.dot(w, xc * yc) / float(np.sum(w)))
        corr = num / denom
        return float(abs(corr)) if np.isfinite(corr) else 0.0
    except Exception:
        return 0.0


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def _weighted_lstsq(design: np.ndarray, target: np.ndarray, sample_weight: Optional[np.ndarray] = None):
    design = np.asarray(design, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    w = _as_sample_weight(sample_weight, target.shape[0])
    if w is None:
        coef, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
        return coef
    sw = np.sqrt(np.maximum(w, 0.0))
    coef, _, _, _ = np.linalg.lstsq(design * sw[:, None], target * sw, rcond=None)
    return coef


def _univariate_poly_score(
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> float:
    """Return validation-free relative R2 from a small univariate polynomial probe."""
    try:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        w = _as_sample_weight(sample_weight, x.shape[0])
        y_var = max(_weighted_var(y, w), 1e-12)
        cols = [x, x * x, x * x * x, np.ones_like(x)]
        design = np.column_stack(cols)
        coef = _weighted_lstsq(design, y, w)
        pred = design @ coef
        resid = pred - y
        if w is None:
            mse = float(np.mean(resid * resid))
        else:
            mse = float(np.dot(w, resid * resid) / float(np.sum(w)))
        if not np.isfinite(mse):
            return 0.0
        return float(np.clip(1.0 - mse / y_var, 0.0, 1.0))
    except Exception:
        return 0.0


def _univariate_holdout_poly_score(
    x: np.ndarray,
    y: np.ndarray,
    *,
    validation_fraction: float = 0.25,
    random_state: int = 0,
    sample_weight: Optional[np.ndarray] = None,
) -> float:
    """Return holdout R2 from a small univariate polynomial probe."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    w_all = _as_sample_weight(sample_weight, x.shape[0])
    mask = np.isfinite(x) & np.isfinite(y)
    if w_all is not None:
        mask = mask & np.isfinite(w_all)
    if int(mask.sum()) < 12:
        return 0.0

    x = x[mask]
    y = y[mask]
    w = w_all[mask] if w_all is not None else None
    n = int(x.shape[0])
    if n < 24:
        return _univariate_poly_score(x, y, w)

    rng = np.random.RandomState(int(random_state))
    indices = np.arange(n)
    rng.shuffle(indices)
    holdout_n = int(max(4, round(n * float(validation_fraction))))
    holdout_n = min(holdout_n, n - 8)
    if holdout_n <= 0:
        holdout_n = max(1, n // 5)
    train_idx = indices[:-holdout_n]
    val_idx = indices[-holdout_n:]
    if train_idx.size < 8 or val_idx.size < 4:
        return 0.0

    x_train = x[train_idx]
    y_train = y[train_idx]
    x_val = x[val_idx]
    y_val = y[val_idx]
    w_train = w[train_idx] if w is not None else None
    w_val = w[val_idx] if w is not None else None
    try:
        design_train = np.column_stack([x_train, x_train * x_train, x_train * x_train * x_train, np.ones_like(x_train)])
        design_val = np.column_stack([x_val, x_val * x_val, x_val * x_val * x_val, np.ones_like(x_val)])
        coef = _weighted_lstsq(design_train, y_train, w_train)
        pred_val = design_val @ coef
        resid = pred_val - y_val
        val_var = max(_weighted_var(y_val, w_val), 1e-12)
        if w_val is None:
            mse = float(np.mean(resid * resid))
        else:
            mse = float(np.dot(w_val, resid * resid) / float(np.sum(w_val)))
        if not np.isfinite(mse):
            return 0.0
        return float(np.clip(1.0 - mse / val_var, 0.0, 1.0))
    except Exception:
        return 0.0


def _affine_relative_score(
    values: np.ndarray,
    y: np.ndarray,
    *,
    validation_fraction: float = 0.25,
    random_state: int = 0,
    sample_weight: Optional[np.ndarray] = None,
) -> tuple[float, float]:
    """Return holdout-aware and train relative R2 for a 1D candidate signal."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    w_all = _as_sample_weight(sample_weight, values.shape[0])
    mask = np.isfinite(values) & np.isfinite(y)
    if w_all is not None:
        mask = mask & np.isfinite(w_all)
    if int(mask.sum()) < 12:
        return 0.0, 0.0

    x = values[mask]
    target = y[mask]
    w = w_all[mask] if w_all is not None else None
    n = int(x.shape[0])
    if n < 24:
        try:
            design = np.column_stack([x, np.ones_like(x)])
            coef = _weighted_lstsq(design, target, w)
            pred = design @ coef
            resid = pred - target
            y_var = max(_weighted_var(target, w), 1e-12)
            if w is None:
                mse = float(np.mean(resid * resid))
            else:
                mse = float(np.dot(w, resid * resid) / float(np.sum(w)))
            rel = float(np.clip(1.0 - mse / y_var, 0.0, 1.0))
            return rel, rel
        except Exception:
            return 0.0, 0.0

    rng = np.random.RandomState(int(random_state))
    indices = np.arange(n)
    rng.shuffle(indices)
    holdout_n = int(max(4, round(n * float(validation_fraction))))
    holdout_n = min(holdout_n, n - 8)
    if holdout_n <= 0:
        holdout_n = max(1, n // 5)
    train_idx = indices[:-holdout_n]
    val_idx = indices[-holdout_n:]
    if train_idx.size < 8 or val_idx.size < 4:
        return 0.0, 0.0

    x_train = x[train_idx]
    y_train = target[train_idx]
    x_val = x[val_idx]
    y_val = target[val_idx]
    w_train = w[train_idx] if w is not None else None
    w_val = w[val_idx] if w is not None else None
    try:
        design_train = np.column_stack([x_train, np.ones_like(x_train)])
        coef = _weighted_lstsq(design_train, y_train, w_train)
        train_pred = coef[0] * x_train + coef[1]
        val_pred = coef[0] * x_val + coef[1]
        train_var = max(_weighted_var(y_train, w_train), 1e-12)
        val_var = max(_weighted_var(y_val, w_val), 1e-12)
        if w_train is None:
            train_mse = float(np.mean((train_pred - y_train) ** 2))
        else:
            train_mse = float(np.dot(w_train, (train_pred - y_train) ** 2) / float(np.sum(w_train)))
        if w_val is None:
            val_mse = float(np.mean((val_pred - y_val) ** 2))
        else:
            val_mse = float(np.dot(w_val, (val_pred - y_val) ** 2) / float(np.sum(w_val)))
        train_rel = float(np.clip(1.0 - train_mse / train_var, 0.0, 1.0))
        val_rel = float(np.clip(1.0 - val_mse / val_var, 0.0, 1.0))
        return val_rel, train_rel
    except Exception:
        return 0.0, 0.0


def _normalize_score_dict(scores: Dict[int, float]) -> Dict[int, float]:
    if not scores:
        return {}
    finite = {
        int(idx): max(0.0, float(value))
        for idx, value in scores.items()
        if np.isfinite(value)
    }
    if not finite:
        return {int(idx): 0.0 for idx in scores}
    max_value = max(finite.values(), default=0.0)
    if max_value <= 1e-12:
        return {int(idx): 0.0 for idx in scores}
    return {
        int(idx): float(np.clip(finite.get(int(idx), 0.0) / max_value, 0.0, 1.0))
        for idx in scores
    }


def _cheap_feature_scores(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> Dict[int, float]:
    """Rank features using deterministic univariate signals only."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    w_all = _as_sample_weight(sample_weight, y.shape[0])
    scores: Dict[int, float] = {}

    y_rank = _rankdata(y)
    for j in range(X.shape[1]):
        col = X[:, j]
        finite = np.isfinite(col) & np.isfinite(y)
        if w_all is not None:
            finite = finite & np.isfinite(w_all) & (w_all > 0.0)
        if int(finite.sum()) < 8:
            scores[j] = 0.0
            continue
        xj = col[finite]
        yj = y[finite]
        wj = w_all[finite] if w_all is not None else None
        pearson = _corr_score(xj, yj, wj)
        spearman = _corr_score(_rankdata(xj), y_rank[finite], wj)
        poly = _univariate_poly_score(xj, yj, wj)
        holdout_poly = _univariate_holdout_poly_score(
            xj,
            yj,
            random_state=97 * (j + 1),
            sample_weight=wj,
        )
        scores[j] = float(0.20 * pearson + 0.15 * spearman + 0.25 * poly + 0.40 * holdout_poly)
    return scores


def _ranking_subsample(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    *,
    max_rows: int = 2000,
    random_state: int = 0,
) -> tuple:
    """Cap ranking rows on large n so MI/trees/Lasso stay sub-second.

    Deterministic subsample preserves relative feature signal for ranking.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    w = _as_sample_weight(sample_weight, y.shape[0])
    n = int(y.shape[0])
    if n <= int(max_rows):
        return X, y, w
    rng = np.random.RandomState(int(random_state))
    if w is not None and float(np.sum(w)) > 0:
        p = np.asarray(w, dtype=np.float64).reshape(-1)
        p = np.clip(p, 0.0, None)
        total = float(np.sum(p))
        if total > 0:
            p = p / total
            idx = rng.choice(n, size=int(max_rows), replace=False, p=p)
        else:
            idx = rng.choice(n, size=int(max_rows), replace=False)
    else:
        idx = rng.choice(n, size=int(max_rows), replace=False)
    idx = np.sort(idx)
    Xs = X[idx] if X.ndim == 2 else X
    ys = y[idx]
    ws = w[idx] if w is not None else None
    return Xs, ys, ws


def _mutual_information_scores(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> Dict[int, float]:
    if mutual_info_regression is None:
        return {}
    try:
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        # S7-3: subsample large n before MI (O(n log n) per feature).
        X_arr, y_arr, w = _ranking_subsample(X_arr, y_arr, sample_weight, max_rows=1500, random_state=0)
        # sklearn MI has no sample_weight; approximate by weighted resampling.
        if w is not None and X_arr.ndim == 2 and X_arr.shape[0] == y_arr.shape[0]:
            rng = np.random.RandomState(0)
            p = w / float(np.sum(w))
            idx = rng.choice(y_arr.shape[0], size=int(y_arr.shape[0]), replace=True, p=p)
            X_arr = X_arr[idx]
            y_arr = y_arr[idx]
        n_neighbors = 3 if X_arr.shape[0] >= 200 else 5
        scores = mutual_info_regression(X_arr, y_arr, random_state=0, n_neighbors=n_neighbors)
        return _normalize_score_dict({j: float(scores[j]) for j in range(len(scores))})
    except Exception:
        return {}


def _sparse_linear_scores(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> Dict[int, float]:
    if LassoCV is None or ElasticNetCV is None:
        return {}
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    # S7-3: subsample large n; cheaper CV grids.
    X, y, w = _ranking_subsample(X, y, sample_weight, max_rows=1200, random_state=1)
    if X.ndim != 2 or X.shape[1] == 0 or X.shape[0] < max(24, X.shape[1] + 4):
        return {}

    # Prefer Lasso only when p is large (ElasticNet CV is expensive).
    n, p = int(X.shape[0]), int(X.shape[1])
    run_enet = p <= 32 and n <= 800

    score_pool: List[np.ndarray] = []
    try:
        lasso = LassoCV(
            cv=3,
            random_state=0,
            max_iter=2000,
            alphas=16 if n >= 400 else 24,
            n_jobs=1,
        )
        if w is None:
            lasso.fit(X, y)
        else:
            lasso.fit(X, y, sample_weight=w)
        score_pool.append(np.abs(np.asarray(lasso.coef_, dtype=np.float64)))
    except Exception:
        pass
    if run_enet:
        try:
            enet = ElasticNetCV(
                cv=3,
                random_state=0,
                max_iter=2000,
                alphas=12 if n >= 400 else 16,
                l1_ratio=(0.5, 0.9, 1.0),
                n_jobs=1,
            )
            if w is None:
                enet.fit(X, y)
            else:
                enet.fit(X, y, sample_weight=w)
            score_pool.append(np.abs(np.asarray(enet.coef_, dtype=np.float64)))
        except Exception:
            pass
    if not score_pool:
        return {}

    mean_scores = np.mean(np.vstack(score_pool), axis=0)
    return _normalize_score_dict({j: float(mean_scores[j]) for j in range(mean_scores.shape[0])})


def _tree_importance_scores(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> Dict[int, float]:
    if ExtraTreesRegressor is None:
        return {}
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    # S7-3: subsample + fewer trees (was 64 on full n).
    X, y, w = _ranking_subsample(X, y, sample_weight, max_rows=1500, random_state=2)
    if X.ndim != 2 or X.shape[1] == 0 or X.shape[0] < 24:
        return {}
    try:
        n = int(X.shape[0])
        p = int(X.shape[1])
        n_estimators = 24 if n >= 500 or p >= 20 else 32
        model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_depth=min(6, max(3, int(np.sqrt(max(1, n))))),
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=0,
            n_jobs=1,
        )
        if w is None:
            model.fit(X, y)
        else:
            model.fit(X, y, sample_weight=w)
        importances = np.asarray(model.feature_importances_, dtype=np.float64)
        return _normalize_score_dict({j: float(importances[j]) for j in range(importances.shape[0])})
    except Exception:
        return {}


def _ranker_disagreement(
    ranker_votes: Dict[str, Dict[int, float]],
    feature_indices: List[int],
) -> float:
    """Estimate disagreement across rankers from per-feature rank spread."""
    if not ranker_votes or not feature_indices:
        return 0.0

    rank_positions: Dict[int, List[int]] = {int(idx): [] for idx in feature_indices}
    active_rankers = 0
    for scores in ranker_votes.values():
        ordered = sorted(
            feature_indices,
            key=lambda idx: float(scores.get(int(idx), 0.0)),
            reverse=True,
        )
        if not ordered:
            continue
        active_rankers += 1
        for pos, idx in enumerate(ordered):
            rank_positions[int(idx)].append(pos)

    if active_rankers < 2:
        return 0.0

    spreads = []
    denom = max(1.0, float(len(feature_indices) - 1))
    for idx in feature_indices:
        positions = rank_positions.get(int(idx), [])
        if len(positions) < 2:
            continue
        spreads.append(float(np.std(np.asarray(positions, dtype=np.float64)) / denom))
    if not spreads:
        return 0.0
    return float(np.clip(np.mean(spreads) * 2.0, 0.0, 1.0))


def compute_blackbox_feature_ranking(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Rank features with deterministic and optional model-based votes."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    w_all = _as_sample_weight(sample_weight, y.shape[0])
    n_features = int(X.shape[1]) if X.ndim == 2 else 0
    if n_features <= 0:
        return {"feature_scores": {}, "ranker_votes": {}, "sample_weight_mode": "none"}

    pearson_scores: Dict[int, float] = {}
    spearman_scores: Dict[int, float] = {}
    poly_scores: Dict[int, float] = {}
    holdout_poly_scores: Dict[int, float] = {}

    y_rank = _rankdata(y)
    for j in range(n_features):
        col = X[:, j]
        finite = np.isfinite(col) & np.isfinite(y)
        if w_all is not None:
            finite = finite & np.isfinite(w_all) & (w_all > 0.0)
        if int(finite.sum()) < 8:
            pearson_scores[j] = 0.0
            spearman_scores[j] = 0.0
            poly_scores[j] = 0.0
            holdout_poly_scores[j] = 0.0
            continue
        xj = col[finite]
        yj = y[finite]
        wj = w_all[finite] if w_all is not None else None
        pearson_scores[j] = _corr_score(xj, yj, wj)
        spearman_scores[j] = _corr_score(_rankdata(xj), y_rank[finite], wj)
        poly_scores[j] = _univariate_poly_score(xj, yj, wj)
        holdout_poly_scores[j] = _univariate_holdout_poly_score(
            xj,
            yj,
            random_state=97 * (j + 1),
            sample_weight=wj,
        )

    ranker_votes: Dict[str, Dict[int, float]] = {
        "pearson": _normalize_score_dict(pearson_scores),
        "spearman": _normalize_score_dict(spearman_scores),
        "poly": _normalize_score_dict(poly_scores),
        "holdout_poly": _normalize_score_dict(holdout_poly_scores),
    }

    mi_scores = _mutual_information_scores(X, y, w_all)
    if mi_scores:
        ranker_votes["mutual_information"] = mi_scores

    sparse_scores = _sparse_linear_scores(X, y, w_all)
    if sparse_scores:
        ranker_votes["sparse_linear"] = sparse_scores

    tree_scores = _tree_importance_scores(X, y, w_all)
    if tree_scores:
        ranker_votes["tree"] = tree_scores

    weights = {
        "pearson": 0.14,
        "spearman": 0.12,
        "poly": 0.18,
        "holdout_poly": 0.24,
        "mutual_information": 0.12,
        "sparse_linear": 0.10,
        "tree": 0.10,
    }
    active_weights = {
        name: weight
        for name, weight in weights.items()
        if name in ranker_votes
    }
    weight_total = sum(active_weights.values()) or 1.0
    feature_scores: Dict[int, float] = {}
    for j in range(n_features):
        score = 0.0
        for ranker_name, ranker_weight in active_weights.items():
            score += ranker_weight * float(ranker_votes.get(ranker_name, {}).get(j, 0.0))
        feature_scores[j] = float(score / weight_total)

    return {
        "feature_scores": feature_scores,
        "ranker_votes": ranker_votes,
        "sample_weight_mode": "provided" if w_all is not None else "none",
    }


def rank_blackbox_features(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> Dict[int, float]:
    """Backwards-compatible feature score helper."""
    return compute_blackbox_feature_ranking(X, y, sample_weight=sample_weight)["feature_scores"]


def discover_blackbox_interactions(
    X: np.ndarray,
    y: np.ndarray,
    selected_features: Optional[List[int]] = None,
    max_pairs: int = 6,
    validation_fraction: float = 0.25,
    sample_weight: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Score a small set of pairwise interaction candidates."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    w_all = _as_sample_weight(sample_weight, y.shape[0])
    if X.ndim != 2 or X.shape[1] < 2:
        return {"interaction_pairs": [], "interaction_terms": [], "interaction_scores": {}}

    cols = list(range(X.shape[1]))
    labels = list(selected_features) if selected_features is not None else list(cols)
    if len(cols) < 2:
        return {"interaction_pairs": [], "interaction_terms": [], "interaction_scores": {}}

    base_scores = _cheap_feature_scores(X, y, w_all)
    candidate_rows: List[Tuple[float, Tuple[int, int], str, np.ndarray]] = []

    def _interaction_family(term: str) -> str:
        lower = term.lower()
        if "sin(" in lower or "cos(" in lower:
            return "periodic"
        if "exp(" in lower:
            return "exp"
        if "log(" in lower:
            return "log"
        if "/" in lower:
            return "rational"
        if "^" in lower:
            return "power_sum"
        if "*" in lower:
            return "product"
        if "+" in lower or "-" in lower:
            return "linear_combo"
        return "other"

    def _normalized_signal(values: np.ndarray) -> Optional[np.ndarray]:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        mask = np.isfinite(arr)
        if int(mask.sum()) < 8:
            return None
        centered = arr[mask] - float(np.mean(arr[mask]))
        scale = float(np.std(centered))
        if not np.isfinite(scale) or scale < 1e-10:
            return None
        out = np.zeros_like(arr, dtype=np.float64)
        out[mask] = centered / scale
        return out

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a = cols[i]
            b = cols[j]
            la = labels[i]
            lb = labels[j]
            xi = X[:, a]
            xj = X[:, b]
            mask = np.isfinite(xi) & np.isfinite(xj) & np.isfinite(y)
            if w_all is not None:
                mask = mask & np.isfinite(w_all) & (w_all > 0.0)
            if int(mask.sum()) < 12:
                continue

            xi = xi[mask]
            xj = xj[mask]
            yj = y[mask]
            wj = w_all[mask] if w_all is not None else None

            candidates = {
                f"x{la}*x{lb}": xi * xj,
                f"x{la}+x{lb}": xi + xj,
                f"x{la}-x{lb}": xi - xj,
                f"x{la}/(x{lb}+1e-6)": xi / (xj + 1e-6),
                f"x{la}^2+x{lb}^2": xi * xi + xj * xj,
                f"x{la}*sin(x{lb})": xi * np.sin(xj),
                f"x{lb}*sin(x{la})": xj * np.sin(xi),
                f"x{la}*cos(x{lb})": xi * np.cos(xj),
                f"x{lb}*cos(x{la})": xj * np.cos(xi),
                f"x{la}*exp(-abs(x{lb}))": xi * np.exp(-np.clip(np.abs(xj), 0.0, 60.0)),
                f"x{lb}*exp(-abs(x{la}))": xj * np.exp(-np.clip(np.abs(xi), 0.0, 60.0)),
                f"x{la}*log(abs(x{lb})+1e-6)": xi * np.log(np.abs(xj) + 1e-6),
                f"x{lb}*log(abs(x{la})+1e-6)": xj * np.log(np.abs(xi) + 1e-6),
            }

            best_term = None
            best_score = -np.inf
            best_signal = None
            for name, values in candidates.items():
                if not np.all(np.isfinite(values)):
                    continue
                try:
                    val_rel, train_rel = _affine_relative_score(
                        values,
                        yj,
                        validation_fraction=validation_fraction,
                        random_state=31 * (a + 1) + 17 * (b + 1),
                        sample_weight=wj,
                    )
                    score = (
                        0.75 * val_rel
                        + 0.15 * train_rel
                        + 0.10 * max(base_scores.get(a, 0.0), base_scores.get(b, 0.0))
                    )
                    if score > best_score:
                        best_score = score
                        best_term = name
                        best_signal = _normalized_signal(values)
                except Exception:
                    continue

            if best_term is not None and best_signal is not None:
                candidate_rows.append((best_score, (la, lb), best_term, best_signal))

    candidate_rows.sort(key=lambda item: item[0], reverse=True)
    top: List[Tuple[float, Tuple[int, int], str, np.ndarray]] = []
    family_counts: Dict[str, int] = {}
    for score, pair, term, signal in candidate_rows:
        if len(top) >= max(0, int(max_pairs)):
            break
        family = _interaction_family(term)
        # Keep the pool diverse: one dominant template is useful, many
        # near-collinear variants waste seeds and inflate operator hints.
        if family_counts.get(family, 0) >= 2 and len(top) >= max(2, int(max_pairs) // 2):
            continue
        redundant = False
        for _, existing_pair, existing_term, existing_signal in top:
            same_pair = tuple(pair) == tuple(existing_pair)
            same_family = _interaction_family(existing_term) == family
            if not (same_pair or same_family):
                continue
            try:
                corr = float(np.corrcoef(signal, existing_signal)[0, 1])
            except Exception:
                corr = 0.0
            if np.isfinite(corr) and abs(corr) >= 0.985:
                redundant = True
                break
        if redundant:
            continue
        family_counts[family] = family_counts.get(family, 0) + 1
        top.append((score, pair, term, signal))

    return {
        "interaction_pairs": [pair for _, pair, _, _ in top],
        "interaction_terms": [term for _, _, term, _ in top],
        "interaction_scores": {term: float(score) for score, _, term, _ in top},
    }


def prepare_blackbox_search(
    X: np.ndarray,
    y: np.ndarray,
    *,
    enabled: bool,
    max_features: int = 6,
    standardize: bool = True,
    min_features_to_select: int = 5,
    interaction_search: bool = True,
    sample_weight: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, BlackboxState]:
    """Return reduced/standardized search data and remapping metadata."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n_features = int(X.shape[1])

    X_finite = np.where(np.isfinite(X), X, np.nan)
    y_finite = np.where(np.isfinite(y), y, np.nan)
    x_mean = np.nanmean(X_finite, axis=0)
    x_mean = np.where(np.isfinite(x_mean), x_mean, 0.0)
    X_clean = np.where(np.isfinite(X), X, x_mean.reshape(1, -1))
    x_scale = _safe_std(X)
    y_mean_raw = float(np.nanmean(y_finite)) if np.any(np.isfinite(y_finite)) else 0.0
    y_mean = y_mean_raw if np.isfinite(y_mean_raw) else 0.0
    y_clean = np.where(np.isfinite(y), y, y_mean)
    y_scale = float(np.nanstd(y_finite)) if np.any(np.isfinite(y_finite)) else 1.0
    if not np.isfinite(y_scale) or y_scale < 1e-12:
        y_scale = 1.0

    if not enabled or n_features <= 1:
        state = BlackboxState(
            enabled=False,
            selected_features=list(range(n_features)),
            dropped_features=[],
            feature_scores={},
            ranker_votes={},
            x_mean=x_mean,
            x_scale=x_scale,
            y_mean=y_mean,
            y_scale=y_scale,
            standardized=False,
            reason="disabled_or_low_dimensional",
        )
        return X_clean, y_clean, state

    variances = np.nanvar(X_clean, axis=0)
    usable = [j for j in range(n_features) if np.isfinite(variances[j]) and variances[j] > 1e-12]
    if not usable:
        state = BlackboxState(
            enabled=False,
            selected_features=list(range(n_features)),
            dropped_features=[],
            feature_scores={},
            ranker_votes={},
            x_mean=x_mean,
            x_scale=x_scale,
            y_mean=y_mean,
            y_scale=y_scale,
            standardized=False,
            reason="no_variable_features",
        )
        return X_clean, y_clean, state

    X_scaled_all = (X_clean - x_mean) / x_scale if standardize else X_clean.copy()
    y_scaled = (y_clean - y_mean) / y_scale if standardize else y_clean.copy()
    w_all = _as_sample_weight(sample_weight, y_clean.shape[0])
    ranking_weight_mode = "provided" if w_all is not None else "none"
    if n_features < int(min_features_to_select):
        selected = sorted(usable)
        feature_scores = {idx: 0.0 for idx in usable}
        ranker_votes = {}
        dropped = [j for j in range(n_features) if j not in selected]
        reason = "retained_all_features_small_problem"
    else:
        cheap_scores = _cheap_feature_scores(X_scaled_all[:, usable], y_scaled, w_all)
        ranking = compute_blackbox_feature_ranking(
            X_scaled_all[:, usable], y_scaled, sample_weight=w_all
        )
        ranking_weight_mode = str(ranking.get("sample_weight_mode") or ranking_weight_mode)
        feature_scores = {
            usable[j]: score
            for j, score in (ranking.get("feature_scores") or {}).items()
        }
        ranker_votes = {
            ranker_name: {
                usable[j]: float(score)
                for j, score in ranker_scores.items()
            }
            for ranker_name, ranker_scores in (ranking.get("ranker_votes") or {}).items()
        }

        k = int(max(1, min(max_features, len(usable))))
        ranked_usable = sorted(usable, key=lambda idx: feature_scores.get(idx, 0.0), reverse=True)
        # No need to drop when everything fits in the budget.
        if len(usable) <= k:
            selected = list(usable)
            reason = "retained_all_features_within_budget"
            top_score = float(feature_scores.get(ranked_usable[0], 0.0)) if ranked_usable else 0.0
            kth_score = float(feature_scores.get(ranked_usable[-1], 0.0)) if ranked_usable else 0.0
            next_score = 0.0
            top_cheap_score = max((float(v) for v in cheap_scores.values()), default=0.0)
            score_gap = top_score
            disagreement = _ranker_disagreement(ranker_votes, ranked_usable)
        else:
            selected = list(ranked_usable[:k])
            top_score = float(feature_scores.get(selected[0], 0.0)) if selected else 0.0
            orig_kth = float(feature_scores.get(selected[-1], 0.0)) if selected else 0.0
            # Near-tie plateau: keep *strong* features scored almost as high as
            # the k-th cut. Fixed threshold (no cascade). Require min score so
            # a weak k-th cut does not pull in the entire noise floor.
            plateau_tol = max(0.02, 0.08 * max(abs(top_score), 1e-12))
            min_meaningful = max(0.12, 0.35 * max(abs(top_score), 1e-12))
            extend_i = k
            max_extra = max(1, min(3, len(usable) - k))  # S7-1: allow slightly wider plateau
            while extend_i < len(ranked_usable) and (extend_i - k) < max_extra:
                s_next = float(feature_scores.get(ranked_usable[extend_i], 0.0))
                if s_next < min_meaningful:
                    break
                abs_gap = orig_kth - s_next
                rel_gap = abs_gap / max(abs(orig_kth), 1e-12) if abs(orig_kth) > 1e-12 else abs_gap
                if s_next >= orig_kth - plateau_tol or rel_gap <= 0.10:
                    selected.append(ranked_usable[extend_i])
                    extend_i += 1
                    continue
                break
            # Drop weak fillers already inside the top-k cut when a clear
            # score cliff exists after the strong head (informative ranking).
            strong = [
                idx for idx in selected
                if float(feature_scores.get(idx, 0.0)) >= min_meaningful
            ]
            if len(strong) >= 2 and len(strong) < len(selected):
                head_min = min(float(feature_scores.get(idx, 0.0)) for idx in strong)
                weak = [
                    idx for idx in selected
                    if float(feature_scores.get(idx, 0.0)) < min_meaningful
                ]
                if weak and head_min >= 2.5 * max(
                    float(feature_scores.get(weak[0], 0.0)), 1e-12
                ):
                    selected = strong
                    extend_i = k  # treat as non-extended for reason labeling

            # S7-1: rescue weak-but-supported features past top-k so true
            # secondary variables are less often dropped under confident ranking.
            top_cheap = max((float(v) for v in cheap_scores.values()), default=0.0)
            rescue_floor = max(0.06, 0.15 * max(abs(top_score), 1e-12))
            max_selected = min(len(usable), max(k + 3, int(max_features) + 2))
            for idx in ranked_usable:
                if idx in selected:
                    continue
                if len(selected) >= max_selected:
                    break
                s_idx = float(feature_scores.get(idx, 0.0))
                if s_idx < rescue_floor:
                    continue
                supported = False
                support_votes = 0
                for votes in (ranker_votes or {}).values():
                    if not isinstance(votes, dict) or not votes:
                        continue
                    ordered = sorted(
                        votes.keys(),
                        key=lambda j: float(votes.get(j, 0.0)),
                        reverse=True,
                    )
                    top_v = float(max(votes.values())) if votes else 0.0
                    # Top-2 among usable features with non-trivial relative vote.
                    if (
                        idx in ordered[: min(2, len(ordered))]
                        and float(votes.get(idx, 0.0)) >= 0.20 * max(top_v, 1e-12)
                    ):
                        support_votes += 1
                if support_votes >= 2:
                    supported = True
                cheap_i = float(cheap_scores.get(idx, 0.0))
                if (
                    not supported
                    and top_cheap > 0
                    and cheap_i >= 0.30 * top_cheap
                    and s_idx >= 0.12 * max(abs(top_score), 1e-12)
                ):
                    supported = True
                # Ensemble score itself is a support signal when not near noise floor.
                if (
                    not supported
                    and s_idx >= 0.20 * max(abs(top_score), 1e-12)
                    and s_idx >= 3.0 * max(
                        float(feature_scores.get(ranked_usable[-1], 0.0)), 1e-12
                    )
                ):
                    supported = True
                if supported:
                    selected.append(int(idx))

            kth_score = float(feature_scores.get(selected[-1], 0.0)) if selected else 0.0
            # Recompute next after possible head trim.
            ranked_sel = sorted(
                selected, key=lambda idx: feature_scores.get(idx, 0.0), reverse=True
            )
            tail_rank = 0
            for pos, idx in enumerate(ranked_usable):
                if idx in selected:
                    tail_rank = pos
            next_score = (
                float(feature_scores.get(ranked_usable[tail_rank + 1], 0.0))
                if tail_rank + 1 < len(ranked_usable)
                else 0.0
            )
            top_cheap_score = max((float(v) for v in cheap_scores.values()), default=0.0)
            score_gap = kth_score - next_score
            disagreement = _ranker_disagreement(ranker_votes, ranked_usable)
            score_values = np.asarray(
                [float(feature_scores.get(idx, 0.0)) for idx in usable],
                dtype=np.float64,
            )
            score_max = float(np.max(score_values)) if score_values.size else 0.0
            score_min = float(np.min(score_values)) if score_values.size else 0.0
            # Symmetric multi-var (e.g. Vladislavleva-4): every coord has
            # comparable *and informative* score — keep all rather than drop
            # one weak sample draw. Pure-noise rankings have low cheap scores.
            near_plateau = (
                len(usable) <= 6
                and score_max >= 0.25
                and top_cheap_score >= 0.18
                and score_min >= 0.35 * score_max
            )
            near_boundary_tie = (
                next_score >= min_meaningful
                and score_gap < max(0.03, 0.10 * max(abs(top_score), 1e-12))
            )
            weak_signal = top_cheap_score <= 0.12 or score_max <= 0.15
            uncertain = near_plateau or (
                len(usable) <= max(5, k + 2)
                and (
                    weak_signal
                    or (near_boundary_tie and disagreement >= 0.35)
                    or (
                        top_cheap_score <= 0.20
                        and score_gap < max(0.08, 0.20 * max(abs(top_score), 1e-12))
                        and disagreement >= 0.35
                    )
                )
            )
            if uncertain:
                selected = list(usable)
                reason = (
                    "retained_all_features_score_plateau"
                    if near_plateau
                    else "retained_all_features_uncertain_selection"
                )
            elif len(selected) > k:
                reason = "selected_top_features_plateau_extended"
            else:
                reason = "selected_top_features"
        selected = sorted(set(int(i) for i in selected))
        dropped = [j for j in range(n_features) if j not in selected]

    if interaction_search:
        interaction_state = discover_blackbox_interactions(
            X_scaled_all[:, selected],
            y_scaled,
            selected_features=selected,
            sample_weight=w_all,
        )
    else:
        interaction_state = {
            "interaction_pairs": [],
            "interaction_terms": [],
            "interaction_scores": {},
        }

    state = BlackboxState(
        enabled=True,
        selected_features=selected,
        dropped_features=dropped,
        feature_scores=feature_scores,
        ranker_votes=ranker_votes,
        x_mean=x_mean,
        x_scale=x_scale,
        y_mean=y_mean,
        y_scale=y_scale,
        standardized=bool(standardize),
        reason=reason,
        interaction_pairs=list(interaction_state["interaction_pairs"]),
        interaction_terms=list(interaction_state["interaction_terms"]),
        interaction_scores=dict(interaction_state["interaction_scores"]),
        feature_selection_uncertain=reason in (
            "retained_all_features_uncertain_selection",
            "retained_all_features_score_plateau",
        ),
        candidate_seed_formulas=build_blackbox_seed_formulas(selected, interaction_state["interaction_terms"]),
    )
    # Attach ranking weight mode for diagnostics (not a dataclass field).
    state.ranking_sample_weight_mode = ranking_weight_mode  # type: ignore[attr-defined]
    return X_scaled_all[:, selected], y_scaled, state


def build_search_space_structure_seeds(
    n_features: int,
    *,
    max_seeds: int = 12,
) -> List[str]:
    """Structure skeletons in *reduced/search* indices x0..xk (standardized space).

    Free numeric constants are intentional so constant refine / affine fit can
    recover Pagie-like, radial-rational, and product/square families without
    original-space template auto-win.
    """
    n = int(n_features)
    if n < 2:
        return []
    formulas: List[str] = []

    def add(formula: str) -> None:
        text = str(formula or "").strip()
        if text and text not in formulas:
            formulas.append(text)

    idxs = list(range(n))
    # Priority order: one family head each, then fill — avoid starving product seeds.
    # Free constants must NOT be bare 0/1: constant refine skips those literals.
    # Pagie under std: free affine inside power — (a*x+b)^4 / (c+(a*x+b)^4)
    k = min(4, n)
    add("+".join(f"(1.1*x{i}+0.1)^4/(1.1+(1.1*x{i}+0.1)^4)" for i in idxs[:k]))
    add("+".join(f"(0.5*x{i}+0.2)^4/(0.8+(0.5*x{i}+0.2)^4)" for i in idxs[:k]))
    add("+".join(f"x{i}^4/(1.1+x{i}^4)" for i in idxs[:k]))
    add("+".join(f"1.1/(1.1+x{i}^4)" for i in idxs[:k]))
    # Radial / anisotropic (Vlad-like under std): free a_i,b_i,c
    sq0 = "+".join(f"x{i}^2" for i in idxs)
    add(f"1.1/(1.1+{sq0})")
    add(f"5.1/(5.1+{sq0})")
    sq_ab = "+".join(f"(1.1*x{i}+0.1)^2" for i in idxs)
    add(f"1.1/(1.1+{sq_ab})")
    add(f"5.1/(5.1+{sq_ab})")
    # Product / square (Feynman-like) — free scales inside
    if n >= 3:
        for a, b, c in ((0, 1, 2), (0, 2, 1), (1, 2, 0)):
            if a < n and b < n and c < n:
                add(f"x{a}*x{b}/x{c}^2")
                add(f"x{a}*x{b}/(1.1+x{c}^2)")
                add(f"(1.1*x{a})*(1.1*x{b})/(1.1+(1.1*x{c})^2)")
                add(f"(1.1*x{a}+0.1)*(1.1*x{b}+0.1)/((1.1*x{c}+0.1)^2)")
    # Fill remaining slots
    for i in idxs[: min(2, n)]:
        add(f"(1.1*x{i}+0.1)^4/(1.1+(1.1*x{i}+0.1)^4)")
        add(f"1.1/(1.1+x{i}^4)")
    for center in (0.5, -0.5):
        if len(formulas) >= max_seeds:
            break
        c_txt = f"{center:g}"
        sq = "+".join(f"(x{i}-{c_txt})^2" for i in idxs)
        add(f"1.1/(1.1+{sq})")
    for a_i, a in enumerate(idxs[:3]):
        for b in idxs[a_i + 1 : 3]:
            add(f"x{a}*x{b}")
            add(f"1.1/(1.1+x{a}^2+x{b}^2)")
            if len(formulas) >= max_seeds:
                return formulas[:max_seeds]

    return formulas[:max_seeds]


def build_blackbox_seed_formulas(
    selected_features: List[int],
    interaction_terms: Optional[List[str]] = None,
    max_seeds: int = 24,
) -> List[str]:
    """Build original-indexed multivariate seed formulas for blackbox search."""
    formulas: List[str] = []

    def add(formula: str) -> None:
        text = str(formula or "").strip()
        if text and text not in formulas:
            formulas.append(text)

    # Multivariate Track 1 cases suffer when the seed budget is consumed by
    # univariate basis terms before cross-feature structures are offered.
    # Reserve the first slice for validated interactions and simple pairwise
    # compositions, then fill the remainder with unary feature transforms.
    for term in interaction_terms or []:
        add(term)
        if len(formulas) >= max_seeds:
            return formulas[:max_seeds]

    # Structure-recovery skeletons (Pagie / Vlad / Feynman-like).
    # Cap early so interaction/unary seeds still get budget.
    structure_budget = min(max_seeds, max(6, int(round(max_seeds * 0.35))))
    if len(selected_features) >= 2:
        pagie_inv = "+".join(f"1/(1+x{i}^(-4))" for i in selected_features[:4])
        pagie_pow = "+".join(f"1/(1+x{i}^4)" for i in selected_features[:4])
        add(pagie_inv)
        add(pagie_pow)
        for idx in selected_features[:2]:
            add(f"1/(1+x{idx}^(-4))")
            add(f"1/(1+x{idx}^4)")
        for center in (0, 3):
            if len(formulas) >= structure_budget:
                break
            sq = "+".join(f"(x{i}-{center})^2" for i in selected_features)
            add(f"1/(5+{sq})")
            add(f"10/(5+{sq})")
            add(f"1/(1+{sq})")
    if len(selected_features) >= 3 and len(formulas) < structure_budget:
        feats = selected_features[:4]
        for a_i, a in enumerate(feats):
            for b in feats[a_i + 1:]:
                for c in feats:
                    if c == a or c == b:
                        continue
                    add(f"x{a}*x{b}/x{c}^2")
                    if len(formulas) >= structure_budget:
                        break
                if len(formulas) >= structure_budget:
                    break
            if len(formulas) >= structure_budget:
                break

    pair_budget = max(2, int(round(max_seeds * 0.40))) if len(selected_features) > 1 else 0
    for a_i, a in enumerate(selected_features):
        for b in selected_features[a_i + 1:]:
            add(f"x{a}*x{b}")
            add(f"x{a}+x{b}")
            add(f"x{a}-x{b}")
            add(f"(x{a}-x{b})^2")
            if len(formulas) >= pair_budget:
                break
        if len(formulas) >= pair_budget:
            break

    # First pass: one essential unary per feature so none are starved by structure seeds.
    for idx in selected_features:
        add(f"x{idx}")
        add(f"sin(x{idx})")
        if len(formulas) >= max_seeds:
            return formulas[:max_seeds]
    for idx in selected_features:
        add(f"x{idx}^2")
        add(f"x{idx}^3")
        add(f"cos(x{idx})")
        add(f"exp(-abs(x{idx}))")
        if len(formulas) >= max_seeds:
            return formulas[:max_seeds]

    for a_i, a in enumerate(selected_features):
        for b in selected_features[a_i + 1:]:
            add(f"x{a}*x{b}")
            add(f"x{a}+x{b}")
            if len(formulas) >= max_seeds:
                return formulas[:max_seeds]

    return formulas[:max_seeds]


def remap_reduced_formula_to_original(formula: str, selected_features: List[int]) -> str:
    """Map reduced feature names x0..xk back to original xJ names.

    S7-2: unmapped local indices (OOB vs selected_features) become ``0`` so
    the formula never references a feature that does not exist in original X.
    """
    if not formula or not selected_features:
        return formula

    def repl(match: re.Match[str]) -> str:
        local_idx = int(match.group(1))
        if 0 <= local_idx < len(selected_features):
            return f"x{int(selected_features[local_idx])}"
        # OOB local index — safe neutral constant (not a phantom feature).
        return "0"

    mapped = re.sub(r"\bx(\d+)\b", repl, formula)
    if len(selected_features) == 1:
        mapped = re.sub(r"\bx\b", f"x{int(selected_features[0])}", mapped)
    return mapped


def remap_original_formula_to_reduced(formula: str, selected_features: List[int]) -> str:
    """Map original feature names xJ into reduced search-space names x0..xk.

    S7-2: features not in the selected set (dropped vars) become ``0`` so the
    reduced-space formula never references an OOB local index after reverse map.
    """
    if not formula or not selected_features:
        return formula

    inverse = {int(original): local for local, original in enumerate(selected_features)}

    def repl(match: re.Match[str]) -> str:
        original_idx = int(match.group(1))
        if original_idx in inverse:
            return f"x{inverse[original_idx]}"
        # Dropped / unselected original feature → neutral constant.
        return "0"

    return re.sub(r"\bx(\d+)\b", repl, formula)


def formula_from_search_to_original_space(formula: str, state: BlackboxState) -> str:
    """Convert a reduced/search-space formula back to original feature names.

    If standardization was used, local variables are expanded as
    ``(x_j - mean_j) / scale_j`` and the target inverse transform is applied.
    S7-2: OOB local indices expand to ``0`` (not left as raw ``xK``).
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
        return "0"

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
        "ranker_votes": {
            str(name): dict(scores)
            for name, scores in state.ranker_votes.items()
        },
        "standardized": state.standardized,
        "reason": state.reason,
        "interaction_pairs": list(state.interaction_pairs),
        "interaction_terms": list(state.interaction_terms),
        "interaction_scores": dict(state.interaction_scores),
        "feature_selection_uncertain": bool(state.feature_selection_uncertain),
        "candidate_seed_formulas": list(state.candidate_seed_formulas),
        "ranking_sample_weight_mode": str(
            getattr(state, "ranking_sample_weight_mode", "none") or "none"
        ),
        "n_selected_features": int(len(state.selected_features)),
        "n_dropped_features": int(len(state.dropped_features)),
    }
