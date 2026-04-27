"""Universal fast-path proposer scaffold.

Phase 1 MVP module:
- point-set encoder model
- grammar-constrained top-k skeleton decoding
- FPIP v2 adapter for downstream evolution
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from .fpip_v2 import validate_fpip_v2_payload


DEFAULT_OPERATOR_VOCAB: List[str] = [
    "identity",
    "sin",
    "cos",
    "power",
    "exp",
    "log",
    "rational",
    "periodic",
]

DEFAULT_SKELETON_VOCAB: List[str] = [
    "x",
    "x^2",
    "sin(x)",
    "cos(x)",
    "exp(x)",
    "log(abs(x)+1e-6)",
    "1/(x+1e-3)",
    "x*sin(x)",
    "x+sin(x)",
]


@dataclass
class UniversalProposerConfig:
    hidden_dim: int = 128
    point_mlp_layers: int = 2
    operator_vocab: Optional[List[str]] = None
    skeleton_vocab: Optional[List[str]] = None

    def resolved_operator_vocab(self) -> List[str]:
        return list(self.operator_vocab) if self.operator_vocab else list(DEFAULT_OPERATOR_VOCAB)

    def resolved_skeleton_vocab(self) -> List[str]:
        return list(self.skeleton_vocab) if self.skeleton_vocab else list(DEFAULT_SKELETON_VOCAB)


class UniversalProposer(nn.Module):
    """Simple Set-style proposer over (x, y) point clouds."""

    def __init__(self, config: Optional[UniversalProposerConfig] = None):
        super().__init__()
        self.config = config or UniversalProposerConfig()
        operator_vocab = self.config.resolved_operator_vocab()
        skeleton_vocab = self.config.resolved_skeleton_vocab()

        point_layers: List[nn.Module] = []
        in_dim = 2
        for _ in range(max(1, int(self.config.point_mlp_layers))):
            point_layers.append(nn.Linear(in_dim, self.config.hidden_dim))
            point_layers.append(nn.ReLU())
            in_dim = self.config.hidden_dim
        self.point_encoder = nn.Sequential(*point_layers)

        # Aggregate with mean/max pooling and project once more.
        self.trunk = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
        )

        self.operator_head = nn.Linear(self.config.hidden_dim, len(operator_vocab))
        self.skeleton_head = nn.Linear(self.config.hidden_dim, len(skeleton_vocab))
        self.uncertainty_head = nn.Linear(self.config.hidden_dim, 2)  # entropy proxy, margin proxy

        self.operator_vocab = operator_vocab
        self.skeleton_vocab = skeleton_vocab

    def forward(self, points_xy: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            points_xy: Tensor[B, N, 2] with columns (x, y).
        """
        if points_xy.ndim != 3 or points_xy.shape[-1] != 2:
            raise ValueError(f"Expected points_xy shape [B,N,2], got {tuple(points_xy.shape)}")

        feats = self.point_encoder(points_xy)
        pooled_mean = feats.mean(dim=1)
        pooled_max = feats.max(dim=1).values
        pooled = torch.cat([pooled_mean, pooled_max], dim=1)
        trunk = self.trunk(pooled)

        return {
            "operator_logits": self.operator_head(trunk),
            "skeleton_logits": self.skeleton_head(trunk),
            "uncertainty_raw": self.uncertainty_head(trunk),
        }


def _safe_softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    denom = np.sum(exp)
    if denom <= 0.0 or not np.isfinite(denom):
        return np.ones_like(logits) / max(1, logits.size)
    return exp / denom


def _topk_indices(values: np.ndarray, k: int) -> np.ndarray:
    if values.size == 0:
        return np.asarray([], dtype=np.int64)
    k = int(max(1, min(k, values.size)))
    part = np.argpartition(values, -k)[-k:]
    return part[np.argsort(values[part])[::-1]]


def decode_topk_skeletons(
    skeleton_logits: Sequence[float],
    skeleton_vocab: Sequence[str],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Grammar-constrained decode via fixed valid skeleton vocabulary."""
    probs = _safe_softmax(np.asarray(skeleton_logits, dtype=np.float64))
    idx = _topk_indices(probs, top_k)
    out: List[Dict[str, Any]] = []
    for i in idx.tolist():
        out.append(
            {
                "formula": str(skeleton_vocab[i]),
                "probability": float(probs[i]),
                "score": float(1.0 - probs[i]),
            }
        )
    return out


def _operator_priors(operator_logits: Sequence[float], operator_vocab: Sequence[str]) -> Dict[str, float]:
    probs = _safe_softmax(np.asarray(operator_logits, dtype=np.float64))
    return {str(op): float(p) for op, p in zip(operator_vocab, probs)}


def _uncertainty_from_logits(logits: Sequence[float]) -> Dict[str, Any]:
    probs = _safe_softmax(np.asarray(logits, dtype=np.float64))
    if probs.size == 0:
        return {"entropy": None, "margin": None, "confident": None}

    sorted_probs = np.sort(probs)[::-1]
    top1 = float(sorted_probs[0])
    top2 = float(sorted_probs[1]) if sorted_probs.size > 1 else 0.0
    entropy = 0.0
    if sorted_probs.size > 1:
        entropy = float(-np.sum(sorted_probs * np.log(sorted_probs + 1e-12)) / np.log(sorted_probs.size))
    margin = top1 - top2
    return {
        "entropy": entropy,
        "margin": margin,
        "confident": bool(entropy < 0.65 and margin > 0.12),
    }


def propose_from_xy(
    model: UniversalProposer,
    x: np.ndarray,
    y: np.ndarray,
    top_k: int = 5,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Run proposer on a single curve and return decoded candidates + priors."""
    if x.ndim == 2 and x.shape[1] == 1:
        x = x[:, 0]
    if x.ndim != 1:
        raise ValueError(f"Expected x to be 1D or [N,1], got shape {x.shape}")

    y = y.reshape(-1)
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have same length")

    points = np.stack([x.astype(np.float32), y.astype(np.float32)], axis=1)
    points_t = torch.from_numpy(points).unsqueeze(0)

    if device is not None:
        model = model.to(torch.device(device))
        points_t = points_t.to(torch.device(device))

    model.eval()
    with torch.no_grad():
        pred = model(points_t)

    operator_logits = pred["operator_logits"][0].detach().cpu().numpy()
    skeleton_logits = pred["skeleton_logits"][0].detach().cpu().numpy()

    candidates = decode_topk_skeletons(skeleton_logits, model.skeleton_vocab, top_k=top_k)
    priors = _operator_priors(operator_logits, model.operator_vocab)
    uncertainty = _uncertainty_from_logits(skeleton_logits)

    return {
        "candidate_skeletons": candidates,
        "operator_priors": priors,
        "sequence_uncertainty": uncertainty,
    }


def proposer_output_to_fpip_v2(
    proposer_output: Dict[str, Any],
    fit_diagnostics: Optional[Dict[str, Any]] = None,
    interaction_hints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Map proposer output to FPIP v2 payload shape."""
    fit_diagnostics = fit_diagnostics or {}
    interaction_hints = interaction_hints or {}

    uncertainty = proposer_output.get("sequence_uncertainty", {})
    confident = uncertainty.get("confident")
    recommend_guided = False if confident is True else True

    payload = {
        "schema_version": "fpip.v2",
        "candidate_skeletons": list(proposer_output.get("candidate_skeletons", [])),
        "sequence_uncertainty": {
            "entropy": uncertainty.get("entropy"),
            "margin": uncertainty.get("margin"),
            "confident": confident,
        },
        "operator_priors": dict(proposer_output.get("operator_priors", {})),
        "interaction_hints": dict(interaction_hints),
        "fit_diagnostics": dict(fit_diagnostics),
        "routing_signal": {
            "recommend_guided_evolution": bool(recommend_guided),
            "reason": "proposer_low_confidence" if recommend_guided else "proposer_confident",
        },
    }

    valid, errors = validate_fpip_v2_payload(payload)
    payload["valid"] = valid
    if not valid:
        payload["validation_errors"] = errors
    return payload
