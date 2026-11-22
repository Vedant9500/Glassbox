"""Regression coverage for §3.196: inception selection uses a fit split."""

import numpy as np

from glassbox.sr.sklearn_wrapper import GlassboxRegressor


def test_inception_extracts_and_fits_on_selection_fit_rows():
    rng = np.random.RandomState(0)
    X = rng.uniform(-2.0, 2.0, size=(120, 1))
    y = np.sin(X[:, 0])

    est = GlassboxRegressor(
        random_state=0,
        enable_inception_reuse=True,
        max_inception_rounds=1,
        max_frozen_subexpressions=1,
    )

    seen = {}

    def fake_extract(formula, X_sel, y_sel, *, max_subexpressions=3):
        seen["extract_rows"] = int(np.asarray(X_sel).shape[0])
        seen["extract_y_rows"] = int(np.asarray(y_sel).shape[0])
        return [
            {
                "formula": "sin(x0)",
                "values": np.sin(np.asarray(X_sel)[:, 0]),
                "standalone_r2": 0.9,
                "complexity": 2,
            }
        ]

    def fake_fit(X_aug, y_fit, pool, *, max_terms=4):
        seen["fit_rows"] = int(np.asarray(X_aug).shape[0])
        seen["fit_y_rows"] = int(np.asarray(y_fit).shape[0])
        return None  # stop after the selection/fit call

    est._extract_frozen_subexpressions = fake_extract
    est._fit_blackbox_basis_model = fake_fit

    est._run_inception_reuse_impl(X, y, "sin(x0)")

    # Both feature extraction and the augmented basis fit see only the fit
    # split, not all rows.
    assert seen["extract_rows"] == seen["fit_rows"] == seen["fit_y_rows"]
    assert 0 < seen["extract_rows"] < X.shape[0]
    assert est.inception_diagnostics_["selection_split"] == "shuffled_holdout"
    assert est.inception_diagnostics_["selection_fit_rows"] == seen["extract_rows"]
    assert est.inception_diagnostics_["selection_val_rows"] > 0
    assert (
        est.inception_diagnostics_["selection_fit_rows"]
        + est.inception_diagnostics_["selection_val_rows"]
        == X.shape[0]
    )
