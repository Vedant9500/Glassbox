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


def _univariate_holdout_poly_score(
    x: np.ndarray,
    y: np.ndarray,
    *,
    validation_fraction: float = 0.25,
    random_state: int = 0,
) -> float:
    """Return holdout R2 from a small univariate polynomial probe."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 12:
        return 0.0

    x = x[mask]
    y = y[mask]
    n = int(x.shape[0])
    if n < 24:
        return _univariate_poly_score(x, y)

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
    try:
        design_train = np.column_stack([x_train, x_train * x_train, x_train * x_train * x_train, np.ones_like(x_train)])
        design_val = np.column_stack([x_val, x_val * x_val, x_val * x_val * x_val, np.ones_like(x_val)])
        coef, _, _, _ = np.linalg.lstsq(design_train, y_train, rcond=None)
        pred_val = design_val @ coef
        val_var = max(float(np.var(y_val)), 1e-12)
        mse = float(np.mean((pred_val - y_val) ** 2))
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
) -> tuple[float, float]:
    """Return holdout-aware and train relative R2 for a 1D candidate signal."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(values) & np.isfinite(y)
    if int(mask.sum()) < 12:
        return 0.0, 0.0

    x = values[mask]
    target = y[mask]
    n = int(x.shape[0])
    if n < 24:
        try:
            coef, _, _, _ = np.linalg.lstsq(
                np.column_stack([x, np.ones_like(x)]),
                target,
                rcond=None,
            )
            pred = coef[0] * x + coef[1]
            mse = float(np.mean((pred - target) ** 2))
            y_var = max(float(np.var(target)), 1e-12)
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
    try:
        coef, _, _, _ = np.linalg.lstsq(
            np.column_stack([x_train, np.ones_like(x_train)]),
            y_train,
            rcond=None,
        )
        train_pred = coef[0] * x_train + coef[1]
        val_pred = coef[0] * x_val + coef[1]
        train_var = max(float(np.var(y_train)), 1e-12)
        val_var = max(float(np.var(y_val)), 1e-12)
        train_rel = float(np.clip(1.0 - float(np.mean((train_pred - y_train) ** 2)) / train_var, 0.0, 1.0))
        val_rel = float(np.clip(1.0 - float(np.mean((val_pred - y_val) ** 2)) / val_var, 0.0, 1.0))
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


def _cheap_feature_scores(X: np.ndarray, y: np.ndarray) -> Dict[int, float]:
    """Rank features using deterministic univariate signals only."""
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
        holdout_poly = _univariate_holdout_poly_score(
            xj,
            yj,
            random_state=97 * (j + 1),
        )
        scores[j] = float(0.20 * pearson + 0.15 * spearman + 0.25 * poly + 0.40 * holdout_poly)
    return scores


def _mutual_information_scores(X: np.ndarray, y: np.ndarray) -> Dict[int, float]:
    if mutual_info_regression is None:
        return {}
    try:
        scores = mutual_info_regression(
            np.asarray(X, dtype=np.float64),
            np.asarray(y, dtype=np.float64).reshape(-1),
            random_state=0,
        )
        return _normalize_score_dict({j: float(scores[j]) for j in range(len(scores))})
    except Exception:
        return {}


def _sparse_linear_scores(X: np.ndarray, y: np.ndarray) -> Dict[int, float]:
    if LassoCV is None or ElasticNetCV is None:
        return {}
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if X.ndim != 2 or X.shape[1] == 0 or X.shape[0] < max(24, X.shape[1] + 4):
        return {}

    score_pool: List[np.ndarray] = []
    try:
        lasso = LassoCV(
            cv=3,
            random_state=0,
            max_iter=5000,
            alphas=32,
        )
        lasso.fit(X, y)
        score_pool.append(np.abs(np.asarray(lasso.coef_, dtype=np.float64)))
    except Exception:
        pass
    try:
        enet = ElasticNetCV(
            cv=3,
            random_state=0,
            max_iter=5000,
            alphas=24,
            l1_ratio=(0.2, 0.5, 0.8, 0.95, 1.0),
        )
        enet.fit(X, y)
        score_pool.append(np.abs(np.asarray(enet.coef_, dtype=np.float64)))
    except Exception:
        pass
    if not score_pool:
        return {}

    mean_scores = np.mean(np.vstack(score_pool), axis=0)
    return _normalize_score_dict({j: float(mean_scores[j]) for j in range(mean_scores.shape[0])})


def _tree_importance_scores(X: np.ndarray, y: np.ndarray) -> Dict[int, float]:
    if ExtraTreesRegressor is None:
        return {}
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if X.ndim != 2 or X.shape[1] == 0 or X.shape[0] < 24:
        return {}
    try:
        model = ExtraTreesRegressor(
            n_estimators=64,
            max_depth=min(8, max(3, int(np.sqrt(max(1, X.shape[0]))))),
            min_samples_leaf=2,
            random_state=0,
            n_jobs=1,
        )
        model.fit(X, y)
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


def compute_blackbox_feature_ranking(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Rank features with deterministic and optional model-based votes."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n_features = int(X.shape[1]) if X.ndim == 2 else 0
    if n_features <= 0:
        return {"feature_scores": {}, "ranker_votes": {}}

    pearson_scores: Dict[int, float] = {}
    spearman_scores: Dict[int, float] = {}
    poly_scores: Dict[int, float] = {}
    holdout_poly_scores: Dict[int, float] = {}

    y_rank = _rankdata(y)
    for j in range(n_features):
        col = X[:, j]
        finite = np.isfinite(col) & np.isfinite(y)
        if int(finite.sum()) < 8:
            pearson_scores[j] = 0.0
            spearman_scores[j] = 0.0
            poly_scores[j] = 0.0
            holdout_poly_scores[j] = 0.0
            continue
        xj = col[finite]
        yj = y[finite]
        pearson_scores[j] = _corr_score(xj, yj)
        spearman_scores[j] = _corr_score(_rankdata(xj), y_rank[finite])
        poly_scores[j] = _univariate_poly_score(xj, yj)
        holdout_poly_scores[j] = _univariate_holdout_poly_score(
            xj,
            yj,
            random_state=97 * (j + 1),
        )

    ranker_votes: Dict[str, Dict[int, float]] = {
        "pearson": _normalize_score_dict(pearson_scores),
        "spearman": _normalize_score_dict(spearman_scores),
        "poly": _normalize_score_dict(poly_scores),
        "holdout_poly": _normalize_score_dict(holdout_poly_scores),
    }

    mi_scores = _mutual_information_scores(X, y)
    if mi_scores:
        ranker_votes["mutual_information"] = mi_scores

    sparse_scores = _sparse_linear_scores(X, y)
    if sparse_scores:
        ranker_votes["sparse_linear"] = sparse_scores

    tree_scores = _tree_importance_scores(X, y)
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
    }


def rank_blackbox_features(X: np.ndarray, y: np.ndarray) -> Dict[int, float]:
    """Backwards-compatible feature score helper."""
    return compute_blackbox_feature_ranking(X, y)["feature_scores"]


def discover_blackbox_interactions(
    X: np.ndarray,
    y: np.ndarray,
    selected_features: Optional[List[int]] = None,
    max_pairs: int = 6,
    validation_fraction: float = 0.25,
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

    base_scores = _cheap_feature_scores(X, y)
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
            for name, values in candidates.items():
                if not np.all(np.isfinite(values)):
                    continue
                try:
                    val_rel, train_rel = _affine_relative_score(
                        values,
                        yj,
                        validation_fraction=validation_fraction,
                        random_state=31 * (a + 1) + 17 * (b + 1),
                    )
                    score = (
                        0.75 * val_rel
                        + 0.15 * train_rel
                        + 0.10 * max(base_scores.get(a, 0.0), base_scores.get(b, 0.0))
                    )
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
            ranker_votes={},
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
            ranker_votes={},
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
        ranker_votes = {}
        dropped = [j for j in range(n_features) if j not in selected]
        reason = "retained_all_features_small_problem"
    else:
        cheap_scores = _cheap_feature_scores(X_scaled_all[:, usable], y_scaled)
        ranking = compute_blackbox_feature_ranking(X_scaled_all[:, usable], y_scaled)
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
        selected = ranked_usable[:k]
        top_score = float(feature_scores.get(selected[0], 0.0)) if selected else 0.0
        kth_score = float(feature_scores.get(selected[-1], 0.0)) if selected else 0.0
        next_score = float(feature_scores.get(ranked_usable[k], 0.0)) if k < len(ranked_usable) else 0.0
        top_cheap_score = max((float(v) for v in cheap_scores.values()), default=0.0)
        score_gap = kth_score - next_score
        disagreement = _ranker_disagreement(ranker_votes, ranked_usable)
        uncertain = (
            len(usable) <= max(5, k + 2)
            and (
                top_cheap_score <= 0.12
                or (
                    next_score > 0.0
                    and score_gap < max(0.04, 0.12 * max(top_score, 1e-12))
                )
                or (
                    top_cheap_score <= 0.20
                    and score_gap < max(0.08, 0.20 * max(top_score, 1e-12))
                    and disagreement >= 0.35
                )
            )
        )
        if uncertain:
            selected = list(usable)
            reason = "retained_all_features_uncertain_selection"
        else:
            reason = "selected_top_features"
        selected = sorted(selected)
        dropped = [j for j in range(n_features) if j not in selected]

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
        feature_selection_uncertain=(reason == "retained_all_features_uncertain_selection"),
        candidate_seed_formulas=build_blackbox_seed_formulas(selected, interaction_state["interaction_terms"]),
    )
    return X_scaled_all[:, selected], y_scaled, state


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

    for idx in selected_features:
        add(f"x{idx}")
        add(f"x{idx}^2")
        add(f"x{idx}^3")
        add(f"sin(x{idx})")
        add(f"cos(x{idx})")
        add(f"exp(-abs(x{idx}))")
        if len(formulas) >= max_seeds:
            return formulas[:max_seeds]

    for term in interaction_terms or []:
        add(term)
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


def remap_original_formula_to_reduced(formula: str, selected_features: List[int]) -> str:
    """Map original feature names xJ into reduced search-space names x0..xk."""
    if not formula or not selected_features:
        return formula

    inverse = {int(original): local for local, original in enumerate(selected_features)}

    def repl(match: re.Match[str]) -> str:
        original_idx = int(match.group(1))
        if original_idx in inverse:
            return f"x{inverse[original_idx]}"
        return match.group(0)

    return re.sub(r"\bx(\d+)\b", repl, formula)


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
    }
