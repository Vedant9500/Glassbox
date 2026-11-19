"""PH6-SNAP: widening zero tolerance during formula cleanup.

The simplifier's ``zero_tol`` decides which additive offsets collapse to
nothing. At the default ``1e-3`` an offset such as ``x - 0.00245`` survives,
which keeps an otherwise-exact formula from reaching its simple exact form and
leaves residual error behind (observed on the tier-6 ``exp(sin(x))`` benchmark
case: ``exp(sin(x - 0.00245))`` at 4.7e-06 instead of ``exp(sin(x))`` at 0).

``_tighten_formula_constants`` re-simplifies with a wider tolerance, but must
never buy tidiness with accuracy or rename the public ``x0``/``x1`` convention.
"""

import re

import numpy as np
import pytest

from glassbox.sr.sklearn_wrapper import GlassboxRegressor


def _estimator() -> GlassboxRegressor:
    est = GlassboxRegressor()
    est.n_features_in_ = 1
    return est


def test_tighten_constants_removes_insignificant_offset():
    """A surviving offset must collapse into the simple exact form."""
    x = np.linspace(-3.0, 3.0, 300)
    X = x.reshape(-1, 1)
    y = np.exp(np.sin(x))
    est = _estimator()

    before = "exp(sin(x - 0.00245))"
    after = est._tighten_formula_constants(before, X, y)

    mse_before = est._display_formula_mse(before, X, y)
    mse_after = est._display_formula_mse(after, X, y)
    assert mse_after <= mse_before
    assert mse_after < 1e-9
    assert "0.00245" not in after


def test_tighten_constants_never_renames_variables():
    """x0 is the public convention; tightening must not rename it."""
    x = np.linspace(-2.0, 2.0, 80)
    X = x.reshape(-1, 1)
    y = np.sin(x)
    est = _estimator()

    out = est._tighten_formula_constants("sin(x0)", X, y)

    assert not re.search(r"\bx(?!\d)", out)


def test_tighten_constants_never_worsens_fit():
    """Already-exact input may only be kept or simplified at equal fit."""
    x = np.linspace(-3.0, 3.0, 300)
    X = x.reshape(-1, 1)
    y = np.exp(np.sin(x))
    est = _estimator()

    before = "exp(sin(x))"
    after = est._tighten_formula_constants(before, X, y)

    assert est._display_formula_mse(
        after, X, y
    ) <= est._display_formula_mse(before, X, y) + 1e-15


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
