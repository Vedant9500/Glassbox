"""Shared model artifact defaults and resolution helpers."""

from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CURVE_CLASSIFIER_PATH = "models/curve_classifier_multi.pt"
PYTORCH_CLASSIFIER_FALLBACKS = (
    "models/curve_classifier_wide.pt",
    "models/curve_classifier_mlp_eql.pt",
    "models/curve_classfier_v4.pt",
    "models/curve_classifier_wider.pt",
    "models/curve_classifier.pt",
)


def iter_curve_classifier_candidates(model_path: str = DEFAULT_CURVE_CLASSIFIER_PATH) -> Iterable[Path]:
    """Yield classifier artifact candidates in canonical fallback order."""
    requested = Path(model_path)
    yield requested

    if model_path == DEFAULT_CURVE_CLASSIFIER_PATH:
        for candidate in PYTORCH_CLASSIFIER_FALLBACKS:
            yield Path(candidate)


def _existing_path(candidate: Path) -> Path | None:
    if candidate.is_absolute():
        return candidate if candidate.exists() else None
    if candidate.exists():
        return candidate
    repo_path = REPO_ROOT / candidate
    if repo_path.exists():
        return repo_path
    return None


def resolve_curve_classifier_path(model_path: str = DEFAULT_CURVE_CLASSIFIER_PATH) -> Path:
    """Resolve a classifier checkpoint path with the shared fallback order."""
    for candidate in iter_curve_classifier_candidates(model_path):
        resolved = _existing_path(candidate)
        if resolved is not None:
            return resolved
    raise FileNotFoundError(f"Classifier model not found at {model_path}")
