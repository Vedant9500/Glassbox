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


def test_exact_match_uses_bounded_fallback_after_large_combination_skip():
    x = np.linspace(-1.0, 1.0, 64, dtype=np.float64)
    basis = np.column_stack([x ** (i % 7 + 1) + 0.001 * i for i in range(110)])
    names = [f"b{i}" for i in range(basis.shape[1])]
    y = x + x ** 2
    diagnostics = {}

    result = cfp.find_exact_symbolic_match(
        basis,
        names,
        y,
        max_terms=3,
        tolerance=1e-8,
        exact_match_backend="torch_cuda",
        exact_match_max_combos=50_000,
        diagnostics=diagnostics,
    )

    assert result is not None
    assert diagnostics["fallback_reason"] == "bounded_sparse_beam_match"
    assert diagnostics["combo_count"] > diagnostics["max_combos"]
    assert diagnostics["gpu_used"] is False
    assert result[1] < 1e-8


def test_multivariate_fast_path_skips_univariate_frequency_refinement(monkeypatch):
    X = np.random.RandomState(0).randn(48, 3)
    y = X[:, 0] + 0.5 * X[:, 1]

    def _fake_fast_path_regression(*args, **kwargs):
        return "x0 + 0.5*x1", 0.05, {"n_nonzero": 2, "exact_match": False}

    def _fail_refine(*args, **kwargs):
        raise AssertionError("multivariate path should not call refine_frequencies")

    monkeypatch.setattr(cfp, "fast_path_regression", _fake_fast_path_regression)
    monkeypatch.setattr(cfp, "refine_frequencies", _fail_refine)

    formula, mse, details = cfp.fast_path_with_refinement(
        X,
        y,
        predictions={"periodic": 0.95, "sin": 0.95},
        detected_omegas=[1.0, 2.0],
        auto_expand=False,
    )

    assert formula == "x0 + 0.5*x1"
    assert mse == 0.05
    assert details["n_nonzero"] == 2
