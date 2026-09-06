"""C++ graph -> PyTorch nn.Module.

Converts a C++ evolution result dict into a live torch.nn.Module for use
inside PyTorch pipelines (training, export, ONNX, etc.).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from glassbox.sr.cpp import graph_enums as _ge


class CppGraphModule(nn.Module):
    """Reconstructs a C++ AST graph as a PyTorch nn.Module.

    The forward() pass mirrors glassbox/sr/cpp/eval.h:
      - Input / Constant nodes
      - Unary Periodic / Power / IntPow / Exp / Log / Abs
      - Binary Arithmetic: soft blend of +/*/soft-div/- (same distances as
        arithmetic_soft_weights; soft-div is x / sqrt(1 + y^2))
      - Binary Division: protected x * sign(y) / (|y| + eps)
      - Binary Aggregation: softmax-weighted mean at tau
    """

    TYPE_INPUT = _ge.TYPE_INPUT
    TYPE_CONSTANT = _ge.TYPE_CONSTANT
    TYPE_UNARY = _ge.TYPE_UNARY
    TYPE_BINARY = _ge.TYPE_BINARY

    UNARY_PERIODIC = _ge.UNARY_PERIODIC
    UNARY_POWER = _ge.UNARY_POWER
    UNARY_INTPOW = _ge.UNARY_INTPOW
    UNARY_EXP = _ge.UNARY_EXP
    UNARY_LOG = _ge.UNARY_LOG
    UNARY_ABS = _ge.UNARY_ABS

    BINARY_ARITHMETIC = _ge.BINARY_ARITHMETIC
    BINARY_DIVISION = _ge.BINARY_DIVISION
    BINARY_AGGREGATION = _ge.BINARY_AGGREGATION

    # Match eval.h get_arithmetic_temperature default when no process hook.
    DEFAULT_ARITHMETIC_TEMPERATURE = 5.0

    def __init__(self, cpp_result: dict, arithmetic_temperature: float | None = None,
                 strict_nonfinite: bool = True):
        super().__init__()

        self.nodes = cpp_result["nodes"]
        self.formula_str = cpp_result.get("formula", "")
        # §3.5: strict parity with C++ eval (reject non-finite) by default.
        # strict_nonfinite=False restores legacy zero-fill for training probes.
        self.strict_nonfinite = bool(strict_nonfinite)
        self.arithmetic_temperature = (
            float(arithmetic_temperature)
            if arithmetic_temperature is not None
            else self.DEFAULT_ARITHMETIC_TEMPERATURE
        )

        weights = cpp_result["output_weights"]
        bias = cpp_result["output_bias"]

        self.output_weights = nn.Parameter(torch.tensor(weights, dtype=torch.float64))
        self.output_bias = nn.Parameter(torch.tensor(bias, dtype=torch.float64))

        for i, node in enumerate(self.nodes):
            ntype = node["type"]
            if ntype == self.TYPE_UNARY:
                p_val = float(node["p"])
                # §3.347: IntPow evaluates rounded (C++ and forward() both
                # round to 2..6), so a fractional stored p is invisible yet
                # real (see §3.329). Canonicalize the buffer at export so
                # serialization matches evaluation (no dD/dp either way —
                # integer cast is non-differentiable by design). NOTE: must
                # match C++ std::round (half away from zero), NOT Python
                # banker's round(): round(2.5) is 2 in Python, 3 in C++.
                if node["unary_op"] == self.UNARY_INTPOW:
                    p_val = float(min(6, max(2, math.floor(p_val + 0.5))))
                self.register_buffer(
                    f"p_{i}", torch.tensor(p_val, dtype=torch.float64)
                )
                self.register_buffer(
                    f"omega_{i}", torch.tensor(node["omega"], dtype=torch.float64)
                )
                self.register_buffer(
                    f"phi_{i}", torch.tensor(node["phi"], dtype=torch.float64)
                )
                self.register_buffer(
                    f"amplitude_{i}",
                    torch.tensor(node["amplitude"], dtype=torch.float64),
                )
            elif ntype == self.TYPE_CONSTANT:
                self.register_buffer(
                    f"value_{i}", torch.tensor(node["value"], dtype=torch.float64)
                )
            elif ntype == self.TYPE_BINARY:
                self.register_buffer(
                    f"beta_{i}", torch.tensor(node["beta"], dtype=torch.float64)
                )
                self.register_buffer(
                    f"gamma_{i}", torch.tensor(node["gamma"], dtype=torch.float64)
                )
                self.register_buffer(
                    f"tau_{i}", torch.tensor(node["tau"], dtype=torch.float64)
                )

    def _arithmetic_soft_weights(
        self, beta: torch.Tensor, gamma: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Port of eval.h arithmetic_soft_weights (max-logit stable softmax)."""
        t = self.arithmetic_temperature
        d_add = (beta - 1.0) ** 2 + (gamma - 1.0) ** 2
        d_mul = (beta - 2.0) ** 2 + (gamma - 1.0) ** 2
        d_div = (beta - 2.0) ** 2 + (gamma + 1.0) ** 2
        d_sub = (beta - 1.0) ** 2 + (gamma + 1.0) ** 2
        logits = torch.stack([-d_add * t, -d_mul * t, -d_div * t, -d_sub * t])
        max_logit = torch.max(logits)
        w = torch.exp(logits - max_logit)
        s = w.sum()
        if (not torch.isfinite(s)) or float(s) <= 0.0:
            quarter = torch.tensor(0.25, dtype=torch.float64, device=beta.device)
            return quarter, quarter, quarter, quarter
        w = w / s
        return w[0], w[1], w[2], w[3]

    def forward(self, x: torch.Tensor, hard: bool = True, **kwargs) -> torch.Tensor:
        """
        Evaluate the C++ graph using PyTorch operations.

        Args:
            x: Input tensor of shape (N,) for single-feature or (N, D) for multi-feature
            hard: Ignored (compatibility with OperationDAG interface)

        Returns:
            Tuple of (output tensor of shape (N,), None) matching ONN signature
        """
        if x.dim() == 1:
            x = x.unsqueeze(1)  # (N,) -> (N, 1)

        x = x.double()
        n_samples = x.shape[0]
        eps = 1e-10
        device = x.device

        node_outputs = []

        for i, node in enumerate(self.nodes):
            ntype = node["type"]

            if ntype == self.TYPE_INPUT:
                feat_idx = node["feature_idx"]
                if feat_idx < x.shape[1]:
                    out = x[:, feat_idx]
                else:
                    out = torch.zeros(n_samples, dtype=torch.float64, device=device)

            elif ntype == self.TYPE_CONSTANT:
                val = getattr(self, f"value_{i}")
                out = val.expand(n_samples)

            elif ntype == self.TYPE_UNARY:
                left = node["left_child"]
                if 0 <= left < len(node_outputs):
                    child = node_outputs[left]
                else:
                    child = torch.zeros(n_samples, dtype=torch.float64, device=device)

                unary_op = node["unary_op"]
                p = getattr(self, f"p_{i}")
                omega = getattr(self, f"omega_{i}")
                phi = getattr(self, f"phi_{i}")
                amplitude = getattr(self, f"amplitude_{i}")

                if unary_op == self.UNARY_PERIODIC:
                    out = amplitude * torch.sin(omega * child + phi)
                elif unary_op == self.UNARY_POWER:
                    # §3.1 canonical parity tol 1e-9 + eps 1e-10 (matches eval.h).
                    abs_child = torch.abs(child) + eps
                    sign_child = torch.sign(child)
                    abs_pow = abs_child.pow(p)
                    p_round = torch.round(p)
                    is_even = (torch.abs(p - p_round) < 1e-9) & (
                        p_round.long() % 2 == 0
                    )
                    is_even = is_even.double()
                    out = (1.0 - is_even) * (sign_child * abs_pow) + is_even * abs_pow
                    out = torch.clamp(out, -1e8, 1e8)
                elif unary_op == self.UNARY_INTPOW:
                    n = torch.round(p).long().clamp(2, 6)
                    out = torch.clamp(child.pow(n), -1e8, 1e8)
                elif unary_op == self.UNARY_EXP:
                    # eval.h clamps *output* of exp to +/- 1e6 (not arg to +/-20).
                    out = torch.clamp(torch.exp(omega * child + phi), -1e6, 1e6)
                elif unary_op == self.UNARY_LOG:
                    out = torch.clamp(torch.log(torch.abs(child) + 1e-6), -1e6, 1e6)
                elif unary_op == self.UNARY_ABS:
                    out = torch.abs(child)
                else:
                    out = child

            elif ntype == self.TYPE_BINARY:
                left = node["left_child"]
                right = node["right_child"]
                left_val = (
                    node_outputs[left]
                    if 0 <= left < len(node_outputs)
                    else torch.zeros(n_samples, dtype=torch.float64, device=device)
                )
                right_val = (
                    node_outputs[right]
                    if 0 <= right < len(node_outputs)
                    else torch.zeros(n_samples, dtype=torch.float64, device=device)
                )

                binary_op = node["binary_op"]
                beta = getattr(self, f"beta_{i}")
                gamma = getattr(self, f"gamma_{i}")

                if binary_op == self.BINARY_ARITHMETIC:
                    # Soft blend matching eval.h arithmetic_soft_weights.
                    w_add, w_mul, w_div, w_sub = self._arithmetic_soft_weights(
                        beta, gamma
                    )
                    res_add = left_val + right_val
                    res_sub = left_val - right_val
                    res_mul = left_val * right_val
                    # Soft division: x / sqrt(1 + y^2) (S5-4 / P6-002).
                    res_div = left_val / torch.sqrt(1.0 + right_val * right_val)
                    out = (
                        w_add * res_add
                        + w_mul * res_mul
                        + w_div * res_div
                        + w_sub * res_sub
                    )
                    out = torch.clamp(out, -1e6, 1e6)
                elif binary_op == self.BINARY_DIVISION:
                    # §3.348: protected division x/(|y|+1e-6)*sign(y). Note
                    # for training consumers: autograd's d/dy flows only
                    # through the denominator (sign' = 0), so the gradient
                    # wrt the right child vanishes at y=0 even though the
                    # forward value swings sign there. C++ never assumes this
                    # gradient is useful; keep trainable structure away from
                    # exact-zero denominators or use output-only training.
                    out = (
                        left_val / (torch.abs(right_val) + 1e-6) * torch.sign(right_val)
                    )
                    out = torch.clamp(out, -1e6, 1e6)
                elif binary_op == self.BINARY_AGGREGATION:
                    # §3.6: mirror eval.h signed behavior (min via -max(-x)).
                    tau = getattr(self, f"tau_{i}")
                    local_tau = torch.where(
                        torch.abs(tau) >= 1e-3,
                        tau,
                        torch.tensor(1e-3, dtype=torch.float64, device=tau.device)
                        * torch.sign(tau + 1e-30),
                    )
                    is_min = bool(float(local_tau.detach().cpu()) < 0.0)
                    ax = -left_val if is_min else left_val
                    ay = -right_val if is_min else right_val
                    mag = torch.abs(local_tau)
                    max_val = torch.maximum(ax, ay)
                    exp_l = torch.exp((ax - max_val) / mag)
                    exp_r = torch.exp((ay - max_val) / mag)
                    sum_exp = exp_l + exp_r
                    blended = (ax * exp_l + ay * exp_r) / sum_exp
                    out = -blended if is_min else blended
                else:
                    out = left_val + right_val
            else:
                out = torch.zeros(n_samples, dtype=torch.float64, device=device)

            # §3.345/§3.349: no second global clamp. Per-operator clamps above
            # already mirror eval.h exactly (Power/IntPow ±1e8, Exp/Log and
            # Arithmetic/Division ±1e6, Abs unbounded, Aggregation unclamped).
            # The old unconditional ±1e6 pass diverged from C++ for Power in
            # (1e6, 1e8] and for large Aggregation/Abs values. Non-finite
            # policy stays in the strict_nonfinite block below (§3.5).
            # §3.5: preserve C++ failure semantics by default; legacy
            # zero-fill only when strict_nonfinite=False.
            if not self.strict_nonfinite:
                out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
            node_outputs.append(out)

        result = torch.zeros(n_samples, dtype=torch.float64, device=device)
        for i, out in enumerate(node_outputs):
            if i < len(self.output_weights):
                # Match eval kOutputWeightActive: skip near-zero weights.
                if torch.abs(self.output_weights[i]) > 1e-6:
                    result = result + self.output_weights[i] * out
        result = result + self.output_bias

        return result, None

    def get_formula(self) -> str:
        """Return the formula string from the C++ result."""
        return self.formula_str


def cpp_result_to_module(
    result: dict, arithmetic_temperature: float | None = None,
    strict_nonfinite: bool = True,
) -> CppGraphModule:
    """
    Convenience: convert a C++ run_evolution() result dict into a nn.Module.

    Usage:
        from glassbox.sr.cpp import get_cpp_core, cpp_result_to_module
        result = get_cpp_core().run_evolution(X_list, y, ...)
        module = cpp_result_to_module(result)
        pred, _ = module(x_tensor)
    """
    return CppGraphModule(result, arithmetic_temperature=arithmetic_temperature,
                          strict_nonfinite=strict_nonfinite)
