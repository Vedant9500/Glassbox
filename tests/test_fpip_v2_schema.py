import numpy as np
import torch

from glassbox.sr.fpip_v2 import build_fpip_v2_from_fast_path, validate_fpip_v2_payload
from scripts.classifier_fast_path import run_fast_path


def test_fpip_v2_builder_and_validator():
    payload = build_fpip_v2_from_fast_path(
        formula="sin(x)",
        mse=1e-4,
        candidate_formulas=[
            {"formula": "sin(x)", "mse": 1e-4, "score": 0.01},
            {"formula": "cos(x)", "mse": 0.1, "score": 0.2},
        ],
        predictions={"sin": 0.8, "cos": 0.2},
        uncertainty={"prediction_entropy": 0.2, "prediction_margin": 0.6, "prediction_uncertain": False},
        residual_diagnostics={"residual_suspicious": False, "residual_spectral_peak_ratio": 0.1},
        operator_hints={"operators": {"sin", "periodic"}, "frequencies": [1.0]},
    )

    ok, errors = validate_fpip_v2_payload(payload)
    assert ok is True
    assert errors == []
    assert payload["schema_version"] == "fpip.v2"
    assert len(payload["candidate_skeletons"]) == 2


def test_run_fast_path_constant_emits_fpip_v2():
    x = torch.linspace(-1.0, 1.0, 64).reshape(-1, 1)
    y = torch.ones((64, 1), dtype=torch.float32) * 3.0

    result = run_fast_path(x, y, classifier_path="unused.pt")

    assert result is not None
    assert "fpip_v2" in result
    assert result.get("fpip_v2_valid") is True
    ok, errors = validate_fpip_v2_payload(result["fpip_v2"])
    assert ok is True
    assert errors == []
    assert np.isfinite(result["mse"])
