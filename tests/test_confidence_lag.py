"""Regression coverage for §3.90: confidence uses the current stability count."""

import pytest

from glassbox.evolution.evolution import StructureConfidenceTracker


def test_confidence_uses_freshly_updated_stability_count():
    tracker = StructureConfidenceTracker(
        mse_threshold=0.01, stability_generations=6
    )

    confidence = tracker.update(
        generation=0, correlation=0.999, mse=0.001
    )

    assert tracker.generations_stable == 1
    # corr=1.0, mse=1.0, stability=1/6 -> 0.5 + 0.25 + 0.25/6.
    assert confidence == pytest.approx(0.5 + 0.25 + 0.25 / 6.0)


def test_validation_bonus_still_requires_prior_stable_generation():
    tracker = StructureConfidenceTracker(
        mse_threshold=0.01, stability_generations=6
    )

    first = tracker.update(
        generation=0, correlation=0.999, mse=0.001, validation_corr=0.99
    )
    # M-131 stays intact: the first good generation gets no validation bonus.
    assert first == pytest.approx(0.5 + 0.25 + 0.25 / 6.0)

    second = tracker.update(
        generation=1, correlation=0.999, mse=0.001, validation_corr=0.99
    )
    # Prior generation was stable, so the bonus now applies.
    assert second == pytest.approx(0.5 + 0.25 + 0.25 * (2.0 / 6.0) + 0.1)
