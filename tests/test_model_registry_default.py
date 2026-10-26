"""Guards for the shared curve-classifier artifact defaults."""

from __future__ import annotations

from pathlib import Path

from glassbox.model_registry import (
    DEFAULT_CURVE_CLASSIFIER_PATH,
    iter_curve_classifier_candidates,
    resolve_curve_classifier_path,
)


def test_default_classifier_is_dual_head_phase1_checkpoint() -> None:
    assert Path(DEFAULT_CURVE_CLASSIFIER_PATH).name == "curve_classifier_v7_dual.pt"


def test_previous_single_head_model_is_a_fallback() -> None:
    candidates = [Path(p).name for p in iter_curve_classifier_candidates()]
    assert "curve_classifier_multi.pt" in candidates
    assert candidates[0] == "curve_classifier_v7_dual.pt"


def test_explicit_path_has_no_fallbacks() -> None:
    candidates = list(iter_curve_classifier_candidates("models/custom.pt"))
    assert candidates == [Path("models/custom.pt")]


def test_resolve_prefers_explicit_existing_path(tmp_path) -> None:
    explicit = tmp_path / "explicit.pt"
    explicit.write_bytes(b"stub")
    assert resolve_curve_classifier_path(str(explicit)) == explicit
