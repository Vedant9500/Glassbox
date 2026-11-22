"""Regression coverage for §3.36: two-tailed multivariate fast-path holdout."""

import numpy as np

from scripts import classifier_fast_path as cfp


def test_multivariate_holdout_takes_both_distance_tails():
    rng = np.random.RandomState(7)
    x = rng.normal(size=(120, 3))

    idx = cfp._domain_edge_holdout_indices(x, 0.20)
    dists = np.linalg.norm(x - x.mean(axis=0), axis=1)
    order = np.argsort(dists)
    n_edge = max(1, int(len(x) * 0.20 / 2))

    # Both the nearest and the farthest rows are held out.
    assert set(order[:n_edge]).issubset(set(idx.tolist()))
    assert set(order[-n_edge:]).issubset(set(idx.tolist()))
    # Total holdout covers the requested fraction (no one-tailed half).
    assert len(idx) == 2 * n_edge
    assert int(np.argmin(dists)) in idx
    assert int(np.argmax(dists)) in idx


def test_univariate_holdout_still_takes_both_value_tails():
    x = np.linspace(-5.0, 5.0, 100).reshape(-1, 1)
    idx = cfp._domain_edge_holdout_indices(x, 0.20)
    n_edge = max(1, int(len(x) * 0.20 / 2))

    values = x[idx, 0]
    assert len(idx) == 2 * n_edge
    assert values.min() == x[:, 0].min()
    assert values.max() == x[:, 0].max()
    expected = set(range(n_edge)) | set(range(len(x) - n_edge, len(x)))
    assert set(idx.tolist()) == expected
