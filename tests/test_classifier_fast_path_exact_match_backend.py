import numpy as np
import pytest
import torch

from scripts import classifier_fast_path as cfp


def _basis_data():
    x = np.linspace(-2.0, 2.0, 96, dtype=np.float64)
    basis = np.column_stack([np.ones_like(x), x, x ** 2, np.sin(x)])
    names = ["1", "x", "x^2", "sin(x)"]
    y = 2.0 * x - 1.5 * x ** 2
    return basis, names, y


def test_exact_match_backend_cpu_records_diagnostics():
    basis, names, y = _basis_data()
    diagnostics = {}

    result = cfp.find_exact_symbolic_match(
        basis,
        names,
        y,
        max_terms=2,
        tolerance=1e-8,
        device="cpu",
        exact_match_backend="cpu",
        diagnostics=diagnostics,
    )

    assert result is not None
    _, mse, coeffs = result
    assert mse < 1e-8
    assert diagnostics["torch_used"] is True
    assert diagnostics["gpu_used"] is False
    assert diagnostics["resolved_device"] == "cpu"
    assert np.isclose(coeffs[names.index("x")], 2.0)
    assert np.isclose(coeffs[names.index("x^2")], -1.5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_exact_match_backend_cuda_matches_cpu_solution():
    basis, names, y = _basis_data()
    cpu_diagnostics = {}
    cuda_diagnostics = {}

    cpu_result = cfp.find_exact_symbolic_match(
        basis,
        names,
        y,
        max_terms=2,
        tolerance=1e-8,
        device="cpu",
        exact_match_backend="cpu",
        diagnostics=cpu_diagnostics,
    )
    cuda_result = cfp.find_exact_symbolic_match(
        basis,
        names,
        y,
        max_terms=2,
        tolerance=1e-8,
        device="cuda",
        exact_match_backend="torch_cuda",
        exact_match_min_gpu_work=1,
        diagnostics=cuda_diagnostics,
    )

    assert cpu_result is not None
    assert cuda_result is not None
    assert cuda_diagnostics["gpu_used"] is True
    assert cuda_diagnostics["validated_on_cpu"] is True
    assert cuda_result[1] < 1e-8
    assert np.allclose(cuda_result[2], cpu_result[2], atol=1e-4)
