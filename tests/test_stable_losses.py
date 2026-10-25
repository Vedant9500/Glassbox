"""Gradient tests for the stable softmax / CE / BCE fixes (arXiv:2610.31642)."""

import pytest
import torch
import torch.nn.functional as F

from glassbox.curve_classifier.stable_losses import (
    StableBCEWithLogitsLoss,
    StableCrossEntropyLoss,
    StableSoftTargetCrossEntropyLoss,
    binary_cross_entropy_with_logits,
    cross_entropy,
    log_softmax,
    soft_target_cross_entropy,
    softmax,
)


def _grad(loss_fn, tensor):
    tensor = tensor.detach().clone().requires_grad_(True)
    loss = loss_fn(tensor)
    loss.backward()
    return loss.detach(), tensor.grad.detach()


def test_log_softmax_and_softmax_match_torch_for_normal_logits():
    torch.manual_seed(0)
    z = torch.randn(8, 5)

    assert torch.allclose(log_softmax(z), F.log_softmax(z, dim=-1), atol=1e-6)
    assert torch.allclose(softmax(z), F.softmax(z, dim=-1), atol=1e-6)


def test_standard_ce_collapse_reproduced_and_fixed():
    """Paper example: z=[9,-9,-9], t=0.

    Standard fp32 CE returns a zero loss and a target gradient of zero while
    the non-target gradients stay positive (sum > 0).  The fixed loss keeps
    the true fp64 values and its gradient sums to zero.
    """

    z = torch.tensor([[9.0, -9.0, -9.0]])
    target = torch.tensor([0])

    std_loss, std_grad = _grad(lambda t: F.cross_entropy(t, target), z)
    assert std_loss.item() == 0.0
    assert std_grad[0, 0].item() == 0.0
    assert std_grad.sum().item() > 0.0

    fixed_loss, fixed_grad = _grad(lambda t: cross_entropy(t, target), z)

    z64 = z.double().requires_grad_(True)
    ref_loss = F.cross_entropy(z64, target)
    ref_loss.backward()
    ref_grad = z64.grad.float()

    assert fixed_loss.item() == pytest.approx(ref_loss.item(), rel=1e-5)
    assert torch.allclose(fixed_grad, ref_grad, rtol=1e-5, atol=1e-12)
    assert abs(fixed_grad.sum().item()) < 1e-14


def test_fixed_ce_matches_standard_in_normal_regime():
    torch.manual_seed(1)
    z = torch.randn(16, 6)
    target = torch.randint(0, 6, (16,))

    std_loss, std_grad = _grad(lambda t: F.cross_entropy(t, target), z)
    fixed_loss, fixed_grad = _grad(lambda t: cross_entropy(t, target), z)

    assert fixed_loss.item() == pytest.approx(std_loss.item(), rel=1e-5)
    assert torch.allclose(fixed_grad, std_grad, rtol=1e-4, atol=1e-7)


def test_fixed_ce_keeps_large_margin_gradient_against_fp64():
    z = torch.tensor([[30.0, -10.0, -12.0]])
    target = torch.tensor([0])

    z64 = z.double().requires_grad_(True)
    F.cross_entropy(z64, target).backward()

    _, fixed_grad = _grad(lambda t: cross_entropy(t, target), z)
    assert torch.allclose(fixed_grad, z64.grad.float(), rtol=1e-4, atol=1e-14)
    assert abs(fixed_grad.sum().item()) < 1e-15


def test_fixed_bce_keeps_confident_positive_gradient():
    """BCEWithLogitsLoss has the same collapse for a confident positive."""

    z = torch.tensor([20.0])
    target = torch.tensor([1.0])

    std_loss, std_grad = _grad(
        lambda t: F.binary_cross_entropy_with_logits(t, target), z
    )
    fixed_loss, fixed_grad = _grad(
        lambda t: binary_cross_entropy_with_logits(t, target), z
    )

    z64 = z.double().requires_grad_(True)
    ref_loss = F.binary_cross_entropy_with_logits(z64, target.double())
    ref_loss.backward()
    ref_grad = z64.grad.float()

    assert std_loss.item() == pytest.approx(ref_loss.item(), rel=1e-6)
    assert std_grad.item() == 0.0
    assert fixed_loss.item() == pytest.approx(ref_loss.item(), rel=1e-6)
    assert fixed_grad.item() == pytest.approx(ref_grad.item(), rel=1e-4)
    assert fixed_grad.item() < 0.0


def test_fixed_bce_matches_standard_for_normal_logits():
    torch.manual_seed(2)
    logits = torch.randn(32, 9)
    targets = torch.randint(0, 2, (32, 9)).float()
    pos_weight = torch.linspace(0.5, 3.0, 9)

    std_loss, std_grad = _grad(
        lambda t: F.binary_cross_entropy_with_logits(
            t, targets, pos_weight=pos_weight
        ),
        logits,
    )
    fixed_loss, fixed_grad = _grad(
        lambda t: binary_cross_entropy_with_logits(
            t, targets, pos_weight=pos_weight
        ),
        logits,
    )

    assert fixed_loss.item() == pytest.approx(std_loss.item(), rel=1e-5)
    assert torch.allclose(fixed_grad, std_grad, rtol=1e-4, atol=1e-6)


def test_fixed_bce_matches_standard_for_negative_class():
    z = torch.tensor([-20.0])
    target = torch.tensor([0.0])
    std_loss, std_grad = _grad(
        lambda t: F.binary_cross_entropy_with_logits(t, target), z
    )
    fixed_loss, fixed_grad = _grad(
        lambda t: binary_cross_entropy_with_logits(t, target), z
    )

    z64 = z.double().requires_grad_(True)
    ref_loss = F.binary_cross_entropy_with_logits(z64, target.double())
    ref_loss.backward()
    ref_grad = z64.grad.float()

    # PyTorch's forward also loses this tiny loss (it returns 0.0), while the
    # stable form keeps the true value; both gradients agree.
    assert std_loss.item() <= ref_loss.item()
    assert fixed_loss.item() == pytest.approx(ref_loss.item(), rel=1e-5)
    assert fixed_grad.item() == pytest.approx(ref_grad.item(), rel=1e-4)
    assert std_grad.item() == pytest.approx(ref_grad.item(), rel=1e-4)


def test_soft_target_matches_fp64_reference():
    """Paper example: z_T=[2,1,0], z_S = z_T + [-1e-7, 1e-7, 0]."""

    z_t = torch.tensor([[2.0, 1.0, 0.0]])
    z_s = z_t + torch.tensor([[-1e-7, 1e-7, 0.0]])

    def std_loss(student):
        return -(F.softmax(z_t, dim=-1) * F.log_softmax(student, dim=-1)).sum()

    _std_loss_val, std_grad = _grad(std_loss, z_s)
    fixed_loss, fixed_grad = _grad(
        lambda t: soft_target_cross_entropy(t, z_t), z_s
    )

    z_s64 = z_s.double().requires_grad_(True)
    z_t64 = z_t.double()
    ref_loss = -(F.softmax(z_t64, dim=-1) * F.log_softmax(z_s64, dim=-1)).sum()
    ref_loss.backward()

    assert fixed_loss.item() == pytest.approx(ref_loss.item(), rel=1e-5)
    assert torch.allclose(fixed_grad, z_s64.grad.float(), rtol=1e-4, atol=1e-14)
    assert not torch.allclose(std_grad, z_s64.grad.float(), rtol=1e-3, atol=1e-12)


def test_soft_target_gradient_is_zero_when_student_equals_teacher():
    z = torch.tensor([[2.0, 1.0, 0.0]])
    _, fixed_grad = _grad(lambda t: soft_target_cross_entropy(t, z), z)
    assert torch.count_nonzero(fixed_grad) == 0


def test_soft_target_forward_matches_standard_for_normal_logits():
    torch.manual_seed(3)
    z_s = torch.randn(12, 7)
    z_t = torch.randn(12, 7)

    std_loss = -(F.softmax(z_t, dim=-1) * F.log_softmax(z_s, dim=-1)).sum(-1).mean()
    fixed_loss = soft_target_cross_entropy(z_s, z_t)
    assert fixed_loss.item() == pytest.approx(std_loss.item(), rel=1e-5)


def test_reduction_modes_and_module_wrappers():
    torch.manual_seed(4)
    z = torch.randn(6, 4)
    target = torch.randint(0, 4, (6,))
    logits = torch.randn(6, 4)
    labels = torch.randint(0, 2, (6, 4)).float()

    none = cross_entropy(z, target, reduction="none")
    assert none.shape == (6,)
    assert cross_entropy(z, target, reduction="sum").item() == pytest.approx(
        none.sum().item()
    )
    assert cross_entropy(z, target, reduction="mean").item() == pytest.approx(
        none.mean().item()
    )

    ce = StableCrossEntropyLoss()
    assert ce(z, target).item() == pytest.approx(cross_entropy(z, target).item())

    bce = StableBCEWithLogitsLoss()
    assert bce(logits, labels).item() == pytest.approx(
        binary_cross_entropy_with_logits(logits, labels).item()
    )

    soft = StableSoftTargetCrossEntropyLoss()
    assert soft(logits, z).item() == pytest.approx(
        soft_target_cross_entropy(logits, z).item()
    )


def test_invalid_reduction_and_shapes_raise():
    z = torch.randn(2, 3)
    with pytest.raises(ValueError):
        cross_entropy(z, torch.tensor([0, 1]), reduction="banana")
    with pytest.raises(ValueError):
        cross_entropy(z, torch.tensor([0]), reduction="mean")
    with pytest.raises(ValueError):
        binary_cross_entropy_with_logits(z, torch.zeros(2, 2))
    with pytest.raises(ValueError):
        soft_target_cross_entropy(z, torch.randn(2, 2))
