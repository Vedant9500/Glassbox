"""§3.155: fast-path beam ranking keeps family coverage and residual signal."""

import numpy as np

from scripts.classifier_fast_path import coverage_preserving_basis_ranking


def _basis(names, x):
    cols = {
        "1": np.ones_like(x),
        "x": x,
        "x^2": x**2,
        "x^3": x**3,
        "x^5": x**5,
        "sin(3*x)": np.sin(3.0 * x),
        "exp(x)": np.exp(x),
    }
    return np.column_stack([cols[name] for name in names])


def test_family_coverage_keeps_uncorrelated_periodic_column():
    names = ["1", "x", "x^2", "x^3", "sin(3*x)", "exp(x)"]
    x = np.linspace(-2.0, 2.0, 200)
    basis = _basis(names, x)
    # x dominates; sin(3x) is a small but real component.
    y = x + 0.01 * np.sin(3.0 * x)

    ranked, diagnostics = coverage_preserving_basis_ranking(
        basis, y, names, candidate_limit=3
    )
    kept_families = {
        "periodic" if "sin" in names[i] else "poly" if names[i] != "1" else "const"
        for i in ranked
    }
    assert "periodic" in kept_families
    assert len(ranked) <= 3
    assert len(set(ranked)) == len(ranked)
    assert diagnostics["truncated"] is True


def test_residual_guided_pass_recovers_hidden_component():
    # The poly family (x, x^3, x^5) plus exp fill the candidate budget, so
    # plain |corr| ranking drops the small periodic component entirely.
    names = ["1", "x", "x^2", "x^3", "x^5", "sin(3*x)", "exp(x)"]
    x = np.linspace(-2.0, 2.0, 200)
    basis = _basis(names, x)
    y = 10.0 * x + 0.05 * np.sin(3.0 * x)

    ranked, diagnostics = coverage_preserving_basis_ranking(
        basis, y, names, candidate_limit=4, residual_guided=4
    )
    sin_idx = names.index("sin(3*x)")
    assert sin_idx in ranked
    assert diagnostics["family_added"] or diagnostics["residual_added"]


def test_roundoff_residual_does_not_evict_a_useful_column():
    # x, x^3 and exp already span y to round-off. The residual-guided pass
    # must not treat that numerical dust as signal and swap x^3 for x^2.
    names = ["1", "x", "x^2", "x^3", "exp(x)"]
    x = np.linspace(-2.0, 2.0, 200)
    basis = _basis(names, x)
    y = 10.0 * x + 0.05 * x**3 + 0.2 * np.exp(x)

    ranked, diagnostics = coverage_preserving_basis_ranking(
        basis, y, names, candidate_limit=3, residual_guided=4
    )
    assert names.index("x^3") in ranked
    assert diagnostics["residual_added"] == []


def test_no_truncation_keeps_every_column():
    names = ["1", "x", "x^2"]
    x = np.linspace(-1.0, 1.0, 50)
    basis = _basis(names, x)
    y = x + x**2
    ranked, diagnostics = coverage_preserving_basis_ranking(
        basis, y, names, candidate_limit=64
    )
    assert sorted(ranked) == [0, 1, 2]
    assert diagnostics["truncated"] is False
