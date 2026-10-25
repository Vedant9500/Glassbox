"""Numerically stable softmax, cross-entropy, and binary cross-entropy.

PyTorch ports of the fixes in Barron, Gnanapragasam & Tschannen,
"A Fix for Softmax and Cross-Entropy" (arXiv:2610.31642).

The standard implementation shifts logits by their max and then forms
``1 + sum(exp(z_j - z_max))``.  Once the logit margin exceeds the fp32
precision cliff (~16.6), the ``1`` absorbs the rest of the sum: the target
probability rounds to one and its gradient is computed as ``q_t - 1 = 0``.
Non-target gradients are evaluated without adding ``1`` and stay positive,
so the gradients no longer sum to zero.  Under Adam/AdamW this produces the
"slingshot" spikes that stall grokking.

The fixed losses keep the target (or argmax) logit out of the log-sum-exp
and combine the margin with ``softplus``.  The target logit then only appears
through differences ``z_j - z_t``, so autograd derives the target gradient as
the negated sum of the non-target gradients instead of ``q_t - 1``.

``binary_cross_entropy_with_logits`` applies the same idea to the multi-label
case: the stable piecewise form ``t * softplus(-z) + (1 - t) * softplus(z)``
keeps the confident-positive gradient (``-sigmoid(-z)``) instead of letting
``sigmoid(z) - 1`` round to zero.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "StableBCEWithLogitsLoss",
    "StableCrossEntropyLoss",
    "StableSoftTargetCrossEntropyLoss",
    "binary_cross_entropy_with_logits",
    "cross_entropy",
    "log_softmax",
    "soft_target_cross_entropy",
    "softmax",
]

_REDUCTIONS = ("none", "mean", "sum")


def _check_reduction(reduction: str) -> None:
    if reduction not in _REDUCTIONS:
        raise ValueError(
            f"reduction must be one of {_REDUCTIONS}, got {reduction!r}"
        )


def _reduce(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.mean()


def _last_dim(dim: int, ndim: int) -> int:
    resolved = dim if dim >= 0 else ndim + dim
    if resolved != ndim - 1:
        raise ValueError("stable losses expect the class dimension to be last")
    return resolved


def log_softmax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Log-softmax that does not collapse the argmax gradient.

    Equivalent to ``torch.log_softmax`` in exact arithmetic, but the argmax
    entry is kept as an exact zero and excluded from the log-sum-exp.  The
    returned tensor is differentiable and its gradient matches the exact
    (fp64) softmax Jacobian down to exponent underflow.
    """

    _last_dim(dim, z.ndim)
    if z.shape[-1] == 0:
        raise ValueError("log_softmax requires at least one class")

    z_max = z.max(dim=-1, keepdim=True).values
    is_max = torch.zeros_like(z, dtype=torch.bool)
    is_max.scatter_(-1, z.argmax(dim=-1, keepdim=True), True)

    shifted = torch.where(is_max, torch.zeros_like(z), z - z_max)
    others = torch.where(
        is_max, torch.full_like(z, float("-inf")), shifted
    )
    margin = -torch.logsumexp(others, dim=-1, keepdim=True)
    return shifted - F.softplus(-margin)


def softmax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax built from :func:`log_softmax` (paper Eq. 5)."""

    return torch.exp(log_softmax(z, dim=dim))


def cross_entropy(
    z: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Hard-target cross-entropy with a non-collapsing target gradient.

    ``z`` has shape ``(..., K)`` and ``target`` has shape ``(...)`` with
    integer class indices in ``[0, K)``.
    """

    _check_reduction(reduction)
    if z.ndim < 2:
        raise ValueError("cross_entropy expects z with a trailing class dim")
    if target.shape != z.shape[:-1]:
        raise ValueError(
            f"target shape {tuple(target.shape)} does not match logits "
            f"{tuple(z.shape[:-1])}"
        )
    if z.shape[-1] == 0:
        raise ValueError("cross_entropy requires at least one class")

    z_t = z.gather(-1, target.unsqueeze(-1))
    is_t = F.one_hot(target, z.shape[-1]).to(torch.bool)
    others = torch.where(is_t, torch.full_like(z, float("-inf")), z - z_t)
    margin = -torch.logsumexp(others, dim=-1)
    return _reduce(F.softplus(-margin), reduction)


def binary_cross_entropy_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor | float | None = None,
    weight: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Stable binary cross-entropy for multi-label logits.

    Uses ``t * softplus(-z) + (1 - t) * softplus(z)`` rather than
    ``sigmoid(z) - t`` in the backward pass.  ``pos_weight`` follows
    ``torch.nn.BCEWithLogitsLoss`` semantics (it scales the positive term).
    """

    _check_reduction(reduction)
    if logits.shape != targets.shape:
        raise ValueError(
            f"logits shape {tuple(logits.shape)} does not match targets "
            f"{tuple(targets.shape)}"
        )

    targets = targets.to(logits.dtype)
    positive = F.softplus(-logits)
    negative = F.softplus(logits)
    if pos_weight is not None:
        pw = torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device)
        positive = positive * pw
    loss = targets * positive + (1.0 - targets) * negative
    if weight is not None:
        loss = loss * weight.to(loss.dtype)
    return _reduce(loss, reduction)


def soft_target_cross_entropy(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Soft-target cross-entropy without Softmax Cancellation (paper Eq. 9).

    The forward value is the usual ``-sum(p * log q)``.  The backward pass
    substitutes the exact gradient ``q - p`` computed in log space with
    ``expm1``/``log1p``, so the gradient is exactly zero when student and
    teacher logits are equal.
    """

    _check_reduction(reduction)
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"student logits {tuple(student_logits.shape)} do not match "
            f"teacher logits {tuple(teacher_logits.shape)}"
        )
    if student_logits.ndim < 2:
        raise ValueError("soft_target_cross_entropy expects (..., K) logits")

    log_q = log_softmax(student_logits)
    q = torch.exp(log_q)
    p = softmax(teacher_logits)

    delta = student_logits - teacher_logits
    centered = delta - (p * delta).sum(dim=-1, keepdim=True)
    log_ratio = centered - torch.log1p(
        (p * torch.expm1(centered)).sum(dim=-1, keepdim=True)
    )
    diff = p * torch.expm1(log_ratio)
    grad = diff - q * diff.sum(dim=-1, keepdim=True)

    loss = -(p * log_q.detach()).sum(dim=-1)
    # Inject ``grad`` as d(loss)/d(student_logits) while leaving the forward
    # value untouched (the PyTorch equivalent of jax.lax.stop_gradient).
    loss = loss + (grad.detach() * (student_logits - student_logits.detach())).sum(
        dim=-1
    )
    return _reduce(loss, reduction)


class StableCrossEntropyLoss(nn.Module):
    """Module wrapper around :func:`cross_entropy`."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        _check_reduction(reduction)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return cross_entropy(logits, target, reduction=self.reduction)


class StableBCEWithLogitsLoss(nn.Module):
    """Drop-in replacement for ``nn.BCEWithLogitsLoss``.

    Matches the standard module for ordinary logits, but keeps the gradient
    of confident positives instead of rounding it to zero.
    """

    def __init__(
        self,
        pos_weight: torch.Tensor | float | None = None,
        weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        _check_reduction(reduction)
        self.register_buffer(
            "pos_weight",
            None if pos_weight is None else torch.as_tensor(pos_weight),
        )
        self.register_buffer(
            "weight", None if weight is None else torch.as_tensor(weight)
        )
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            weight=self.weight,
            reduction=self.reduction,
        )


class StableSoftTargetCrossEntropyLoss(nn.Module):
    """Module wrapper around :func:`soft_target_cross_entropy`."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        _check_reduction(reduction)
        self.reduction = reduction

    def forward(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        return soft_target_cross_entropy(
            student_logits, teacher_logits, reduction=self.reduction
        )
