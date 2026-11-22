"""§3.171/§3.172: collinear-aware ranking and weighted kNN MI."""

import numpy as np

from glassbox.sr.blackbox_preprocessor import (
    _collinear_feature_groups,
    _mutual_information_scores,
    _weighted_knn_mi,
    compute_blackbox_feature_ranking,
)


def test_weighted_mi_detects_dependence_and_ignores_noise():
    rng = np.random.RandomState(0)
    n = 300
    x = rng.uniform(-2.0, 2.0, n)
    y = np.sin(2.0 * x)
    noise = rng.uniform(-2.0, 2.0, n)

    mi_signal = _weighted_knn_mi(x, y, None, k=3)
    mi_noise = _weighted_knn_mi(noise, y, None, k=3)
    assert mi_signal > 0.2
    assert mi_signal > mi_noise


def test_weighted_mi_does_not_inflate_with_duplicated_rows():
    rng = np.random.RandomState(1)
    n = 240
    x = rng.uniform(-2.0, 2.0, n)
    y = np.sin(1.5 * x) + 0.05 * rng.normal(size=n)

    base = _weighted_knn_mi(x, y, None, k=3)

    # Duplicate every row with half weight: the weighted estimator sees the
    # same effective sample and must not report a much larger MI (the old
    # resampling approach made duplicates their own nearest neighbours).
    x_dup = np.concatenate([x, x])
    y_dup = np.concatenate([y, y])
    w_dup = np.concatenate([np.full(n, 0.5), np.full(n, 0.5)])
    dup = _weighted_knn_mi(x_dup, y_dup, w_dup, k=3)
    assert abs(dup - base) < 0.15 * max(base, 1e-6)


def test_mi_scores_report_split_half_variance():
    rng = np.random.RandomState(2)
    n = 260
    x0 = rng.uniform(-2.0, 2.0, n)
    x1 = rng.normal(size=n)
    y = np.cos(2.0 * x0) + 0.1 * rng.normal(size=n)
    X = np.column_stack([x0, x1])

    variance: dict[int, float] = {}
    scores = _mutual_information_scores(X, y, variance_out=variance)
    assert scores
    assert set(variance) == set(scores)
    assert all(v >= 0.0 for v in variance.values())
    assert scores[0] > scores[1]


def test_collinear_columns_are_detected_and_share_importance():
    rng = np.random.RandomState(3)
    n = 220
    x = rng.uniform(-2.0, 2.0, n)
    X = np.column_stack([x, 2.0 * x, rng.normal(size=n)])
    y = x + 0.05 * rng.normal(size=n)

    groups = _collinear_feature_groups(X)
    assert any(set(g) == {0, 1} for g in groups)

    ranking = compute_blackbox_feature_ranking(X, y)
    scores = ranking["feature_scores"]
    assert scores[0] == scores[1]
    assert ranking["collinear_groups"]
    assert set(ranking["ranker_variance"]) <= {"mutual_information"}


def test_affine_collinearity_is_detected():
    rng = np.random.RandomState(4)
    n = 200
    x = rng.uniform(-2.0, 2.0, n)
    X = np.column_stack([x, x + 3.0])
    groups = _collinear_feature_groups(X)
    assert any(set(g) == {0, 1} for g in groups)
