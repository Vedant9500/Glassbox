import math
import torch
import torch.nn.functional as F
import pytest

from glassbox.sr.hard_concrete import hard_concrete_log_prob


def test_hard_concrete_log_prob_integration():
    logits = torch.tensor(0.5, dtype=torch.float64)
    tau = 0.7
    beta = 0.1

    # Integrate density of s over (0, 1) using numerical integration
    s = torch.linspace(1e-6, 1 - 1e-6, 500000, dtype=torch.float64)
    ds = s[1] - s[0]

    # Convert s to z domain: z = s * (1 + 2*beta) - beta
    z = s * (1 + 2 * beta) - beta
    dz = z[1] - z[0]

    log_prob = hard_concrete_log_prob(z, logits, tau=tau, beta=beta)
    prob = torch.exp(log_prob)

    integral = torch.sum(prob) * dz
    assert pytest.approx(integral.item(), rel=1e-3) == 1.0


def test_hard_concrete_log_prob_gradients_and_types():
    logits = torch.tensor([0.2, -0.5], requires_grad=True)
    z = torch.tensor([0.1, 0.8], requires_grad=True)

    # Float tau
    log_prob_float = hard_concrete_log_prob(z, logits, tau=0.5)
    assert log_prob_float.shape == (2,)
    loss_float = log_prob_float.sum()
    loss_float.backward()

    assert logits.grad is not None
    assert z.grad is not None

    # Tensor tau with grad
    logits = torch.tensor([0.2, -0.5], requires_grad=True)
    z = torch.tensor([0.1, 0.8], requires_grad=True)
    tau_tensor = torch.tensor(0.5, requires_grad=True)

    log_prob_tensor = hard_concrete_log_prob(z, logits, tau=tau_tensor)
    assert log_prob_tensor.shape == (2,)
    loss_tensor = log_prob_tensor.sum()
    loss_tensor.backward()

    assert tau_tensor.grad is not None
