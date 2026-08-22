"""Regression tests for audit finding R-03: pickle/weights_only=False fallback policy.

The production checkpoint loaders must attempt weights-only loading first and may
only fall back to pickle loading when BOTH the checkpoint lives under a trusted
local directory (models/ or artifacts/) AND the operator explicitly opts in via
the GLASSBOX_ALLOW_PICKLE_CHECKPOINT=1 environment variable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from glassbox.curve_classifier import curve_classifier_integration as cci
from glassbox.universal_proposer import universal_proposer as up


def _write_pickle_backed_checkpoint(path: Path) -> Path:
    """Write a checkpoint that fails weights-only loading (embedded numpy array)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": {"layer.weight": torch.ones(2, 2)},
            "thresholds": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        },
        path,
    )
    assert_weights_only_load_fails(path)
    return path


def _write_weights_only_checkpoint(path: Path) -> Path:
    """Write a checkpoint that loads fine with weights_only=True."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": {"layer.weight": torch.ones(2, 2)},
            "thresholds": [0.1, 0.2, 0.3],
        },
        path,
    )
    return path


def assert_weights_only_load_fails(path: Path) -> None:
    with pytest.raises(Exception):
        torch.load(path, map_location="cpu", weights_only=True)


def _make_trusted_layout(tmp_path: Path) -> Path:
    """Create models/ + artifacts/ under tmp_path and repoint module roots at it."""
    models_dir = tmp_path / "models"
    artifacts_dir = tmp_path / "artifacts"
    models_dir.mkdir(exist_ok=True)
    artifacts_dir.mkdir(exist_ok=True)
    up._REPO_ROOT = tmp_path
    cci._ROOT = tmp_path
    return models_dir


@pytest.fixture(autouse=True)
def _reset_module_roots():
    yield
    up._REPO_ROOT = Path(__file__).resolve().parents[1]
    cci._ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("GLASSBOX_ALLOW_PICKLE_CHECKPOINT", raising=False)
    monkeypatch.delenv("GLASSBOX_VERBOSE_CHECKPOINT_LOAD", raising=False)


# ---------------------------------------------------------------------------
# Universal proposer loader
# ---------------------------------------------------------------------------


def test_up_weights_only_checkpoint_loads_without_env(tmp_path: Path):
    models_dir = _make_trusted_layout(tmp_path)
    ckpt = _write_weights_only_checkpoint(models_dir / "safe.pt")
    loaded = up._load_torch_checkpoint(ckpt)
    assert "model_state_dict" in loaded


def test_up_pickle_checkpoint_refused_without_env(tmp_path: Path):
    models_dir = _make_trusted_layout(tmp_path)
    ckpt = _write_pickle_backed_checkpoint(models_dir / "legacy.pt")
    with pytest.raises(RuntimeError, match="GLASSBOX_ALLOW_PICKLE_CHECKPOINT"):
        up._load_torch_checkpoint(ckpt)


def test_up_pickle_checkpoint_allowed_with_env(tmp_path: Path, monkeypatch):
    models_dir = _make_trusted_layout(tmp_path)
    ckpt = _write_pickle_backed_checkpoint(models_dir / "legacy.pt")
    monkeypatch.setenv("GLASSBOX_ALLOW_PICKLE_CHECKPOINT", "1")
    loaded = up._load_torch_checkpoint(ckpt)
    assert "model_state_dict" in loaded


def test_up_pickle_checkpoint_outside_trusted_dir_refused_even_with_env(
    tmp_path: Path, monkeypatch
):
    _make_trusted_layout(tmp_path)
    ckpt = _write_pickle_backed_checkpoint(tmp_path / "outside" / "legacy.pt")
    monkeypatch.setenv("GLASSBOX_ALLOW_PICKLE_CHECKPOINT", "1")
    with pytest.raises(RuntimeError, match="outside trusted local model directories"):
        up._load_torch_checkpoint(ckpt)


# ---------------------------------------------------------------------------
# Curve classifier loader
# ---------------------------------------------------------------------------


def test_cci_weights_only_checkpoint_loads_without_env(tmp_path: Path):
    models_dir = _make_trusted_layout(tmp_path)
    ckpt = _write_weights_only_checkpoint(models_dir / "safe.pt")
    loaded = cci._load_torch_checkpoint(ckpt)
    assert "model_state_dict" in loaded


def test_cci_pickle_checkpoint_refused_without_env(tmp_path: Path):
    models_dir = _make_trusted_layout(tmp_path)
    ckpt = _write_pickle_backed_checkpoint(models_dir / "legacy.pt")
    with pytest.raises(RuntimeError, match="GLASSBOX_ALLOW_PICKLE_CHECKPOINT"):
        cci._load_torch_checkpoint(ckpt)


def test_cci_pickle_checkpoint_allowed_with_env(tmp_path: Path, monkeypatch):
    models_dir = _make_trusted_layout(tmp_path)
    ckpt = _write_pickle_backed_checkpoint(models_dir / "legacy.pt")
    monkeypatch.setenv("GLASSBOX_ALLOW_PICKLE_CHECKPOINT", "1")
    loaded = cci._load_torch_checkpoint(ckpt)
    assert "model_state_dict" in loaded


def test_cci_pickle_checkpoint_outside_trusted_dir_refused_even_with_env(
    tmp_path: Path, monkeypatch
):
    _make_trusted_layout(tmp_path)
    ckpt = _write_pickle_backed_checkpoint(tmp_path / "outside" / "legacy.pt")
    monkeypatch.setenv("GLASSBOX_ALLOW_PICKLE_CHECKPOINT", "1")
    with pytest.raises(RuntimeError, match="outside trusted local model directories"):
        cci._load_torch_checkpoint(ckpt)
