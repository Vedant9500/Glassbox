"""Universal fast-path proposer scaffold.

Phase 1 MVP module:
- point-set encoder model
- grammar-constrained top-k skeleton decoding
- FPIP v2 adapter for downstream evolution
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from glassbox.sr.fpip_v2 import validate_fpip_v2_payload


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
    hidden_dim: int = 256
    n_features: int = 370
    operator_vocab: Optional[List[str]] = None
    skeleton_vocab: Optional[List[str]] = None

    def resolved_operator_vocab(self) -> List[str]:
        return list(self.operator_vocab) if self.operator_vocab else list(DEFAULT_OPERATOR_VOCAB)

    def resolved_skeleton_vocab(self) -> List[str]:
        return list(self.skeleton_vocab) if self.skeleton_vocab else list(DEFAULT_SKELETON_VOCAB)


class UniversalProposer(nn.Module):
    """
    Multiplicative Gated Proposer using Gated Linear Units (GLU).
    Mathematically synchronized with CurveClassifierGLU to leverage high-level analytical features.
    """

    def __init__(self, config: Optional[UniversalProposerConfig] = None):
        super().__init__()
        self.config = config or UniversalProposerConfig()
        operator_vocab = self.config.resolved_operator_vocab()
        skeleton_vocab = self.config.resolved_skeleton_vocab()
        hidden = self.config.hidden_dim
        n_features = self.config.n_features

        # GLU Trunk (Synchronized with Classifier architecture)
        self.fc1 = nn.Linear(n_features, hidden * 2)
        self.bn1 = nn.BatchNorm1d(hidden * 2)
        
        self.fc2 = nn.Linear(hidden, hidden * 2)
        self.bn2 = nn.BatchNorm1d(hidden * 2)

        # Multi-head decoding
        self.operator_head = nn.Linear(hidden, len(operator_vocab))
        self.skeleton_head = nn.Linear(hidden, len(skeleton_vocab))
        self.uncertainty_head = nn.Linear(hidden, 2)

        self.operator_vocab = operator_vocab
        self.skeleton_vocab = skeleton_vocab
        self.dropout = nn.Dropout(0.2)

        self._init_weights()

    def _init_weights(self):
        """Hardware-sympathetic initialization for multiplicative gating."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass using pre-computed high-level features.

        Args:
            features: Tensor[B, 370] containing derivatives, invariants, FFT, and stats.
        """
        if features.ndim == 1:
            features = features.unsqueeze(0)
            
        # Layer 1 GLU projection
        x = self.fc1(features)
        x = self.bn1(x)
        x = F.glu(x, dim=1)
        x = self.dropout(x)
        
        # Layer 2 GLU composition
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.glu(x, dim=1)
        x = self.dropout(x)

        return {
            "operator_logits": self.operator_head(x),
            "skeleton_logits": self.skeleton_head(x),
            "uncertainty_raw": self.uncertainty_head(x),
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


def _formula_operator_tags(formula: str) -> Set[str]:
    tags: Set[str] = set()
    f = formula.lower()
    if "sin(" in f:
        tags.add("sin")
        tags.add("periodic")
    if "cos(" in f:
        tags.add("cos")
        tags.add("periodic")
    if "exp(" in f:
        tags.add("exp")
    if "log(" in f:
        tags.add("log")
    if "/" in f:
        tags.add("rational")
    if "x**" in f or "^" in f:
        tags.add("power")
    if "x" in f:
        tags.add("identity")
    return tags


def _build_univariate_grammar_candidates(max_depth: int = 2) -> List[str]:
    # Grammar-controlled expression set for Phase 1.
    base = [
        "x",
        "x**2",
        "x**3",
        "sin(x)",
        "cos(x)",
        "exp(x)",
        "exp(-x**2)",
        "log(abs(x)+1e-6)",
        "1/(x+1e-3)",
    ]
    if max_depth <= 1:
        return base

    composed = [
        "sin(x**2)",
        "cos(x**2)",
        "x*sin(x)",
        "x*cos(x)",
        "x+sin(x)",
        "x+cos(x)",
        "x**2+sin(x)",
        "x**2+cos(x)",
        "sin(x)*cos(x)",
        "exp(-x)*sin(x)",
        "exp(-x**2)*sin(x)",
        "log(abs(x)+1e-6)+sin(x)",
    ]
    return base + composed


def _safe_formula_eval(formula: str, x: np.ndarray) -> Optional[np.ndarray]:
    context = {
        "np": np,
        "x": x,
        "sin": np.sin,
        "cos": np.cos,
        "exp": lambda z: np.exp(np.clip(z, -30.0, 30.0)),
        "log": lambda z: np.log(np.abs(z) + 1e-6),
        "abs": np.abs,
    }
    expr = formula.replace("^", "**")
    try:
        y = eval(expr, {"__builtins__": None}, context)
    except Exception:
        return None

    if isinstance(y, (int, float)):
        y = np.full_like(x, float(y), dtype=np.float64)
    else:
        y = np.asarray(y, dtype=np.float64)
    if y.shape != x.shape:
        return None
    if not np.all(np.isfinite(y)):
        return None
    return y


def _fit_affine_mse(y_true: np.ndarray, y_basis: np.ndarray) -> float:
    # Solve y_true ~= a*y_basis + b by least squares.
    A = np.stack([y_basis, np.ones_like(y_basis)], axis=1)
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, y_true, rcond=None)
    except Exception:
        return float("inf")
    y_pred = A @ coeffs
    mse = float(np.mean((y_true - y_pred) ** 2))
    return mse if np.isfinite(mse) else float("inf")


def grammar_decode_topk_skeletons(
    operator_priors: Dict[str, float],
    x: np.ndarray,
    y: np.ndarray,
    top_k: int = 5,
    max_depth: int = 2,
) -> List[Dict[str, Any]]:
    """Decode top-k skeletons from a constrained grammar.

    Candidate ranking combines:
    - prior compatibility (operator tags vs predicted operator priors)
    - optional data fit quality via affine fit MSE
    """
    candidates = _build_univariate_grammar_candidates(max_depth=max_depth)
    y = y.reshape(-1)
    x = x.reshape(-1)
    y_var = float(np.var(y)) + 1e-12

    scored: List[Dict[str, Any]] = []
    for formula in candidates:
        tags = _formula_operator_tags(formula)
        if tags:
            prior_score = float(np.mean([operator_priors.get(t, 1e-6) for t in tags]))
        else:
            prior_score = 1e-6

        basis = _safe_formula_eval(formula, x)
        mse = float("inf")
        fit_score = 0.0
        if basis is not None:
            mse = _fit_affine_mse(y, basis)
            fit_score = float(np.exp(-mse / y_var)) if np.isfinite(mse) else 0.0

        # Weighted blend; keep score in [0,1] neighborhood.
        score = 0.65 * prior_score + 0.35 * fit_score
        scored.append(
            {
                "formula": formula,
                "probability": float(max(1e-9, score)),
                "score": float(1.0 - min(score, 1.0)),
                "mse": None if not np.isfinite(mse) else float(mse),
            }
        )

    scored.sort(key=lambda d: (-d["probability"], d["score"]))
    return scored[: max(1, int(top_k))]


def _operator_priors(operator_logits: Sequence[float], operator_vocab: Sequence[str]) -> Dict[str, float]:
    logits = np.asarray(operator_logits, dtype=np.float64)
    # Use sigmoid for multi-label independent operator probabilities
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -100, 100)))
    
    predictions = {str(op): float(p) for op, p in zip(operator_vocab, probs)}
    
    # Mathematical Entailment & Sparsification
    implications = [
        ("sin", "periodic"),
        ("cos", "periodic"),
        ("exp", "exponential"),
        ("log", "exponential"),
        ("rational", "power"),
        ("identity", "polynomial")
    ]
    
    for child, parent in implications:
        if child in predictions and parent in predictions:
            predictions[parent] = max(predictions[parent], predictions[child])
            
    # Entropy-based Sparsification: Silence weak guesses
    for op in list(predictions.keys()):
        if predictions[op] < 0.4:
            del predictions[op]
            
    return predictions


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
    features: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Run proposer on a single curve and return decoded candidates + priors."""
    if x.ndim == 2 and x.shape[1] == 1:
        x = x[:, 0]
    if x.ndim != 1:
        raise ValueError(f"Expected x to be 1D or [N,1], got shape {x.shape}")

    y = y.reshape(-1)
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have same length")

    # If features are not provided, we must extract them (legacy behavior)
    if features is None:
        from scripts.generate_curve_data import extract_all_features
        features = extract_all_features(y)
    
    # Apply SymLog + Scaling (Synchronization with GLU training)
    features = np.sign(features) * np.log1p(np.abs(features))
    if hasattr(model, 'feature_scaler') and model.feature_scaler is not None:
        mean = model.feature_scaler['mean']
        std = model.feature_scaler['std']
        features = (features - mean) / (std + 1e-8)

    features_t = torch.from_numpy(features.astype(np.float32)).unsqueeze(0)

    if device is not None:
        model = model.to(torch.device(device))
        features_t = features_t.to(torch.device(device))

    model.eval()
    with torch.no_grad():
        pred = model(features_t)

    operator_logits = pred["operator_logits"][0].detach().cpu().numpy()
    skeleton_logits = pred["skeleton_logits"][0].detach().cpu().numpy()

    priors = _operator_priors(operator_logits, model.operator_vocab)
    # Grammar-constrained decoding uses priors and quick data-fit checks.
    candidates = grammar_decode_topk_skeletons(
        priors,
        x=x.astype(np.float64),
        y=y.astype(np.float64),
        top_k=top_k,
        max_depth=2,
    )

    # Fallback to direct head decode if grammar decoding unexpectedly returns empty.
    if not candidates:
        candidates = decode_topk_skeletons(skeleton_logits, model.skeleton_vocab, top_k=top_k)

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


def load_universal_proposer_checkpoint(
    checkpoint_path: str,
    device: Optional[str] = None,
) -> UniversalProposer:
    """Load UniversalProposer from checkpoint saved by train_universal_proposer.py."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg_raw = ckpt.get("config", {})
    
    # Map new GLU config (n_features) and handle legacy point_mlp_layers
    config = UniversalProposerConfig(
        hidden_dim=int(cfg_raw.get("hidden_dim", 256)),
        n_features=int(cfg_raw.get("n_features", 370)),
        operator_vocab=cfg_raw.get("operator_vocab"),
        skeleton_vocab=cfg_raw.get("skeleton_vocab"),
    )
    model = UniversalProposer(config)
    model.load_state_dict(ckpt["model_state_dict"])
    
    # Attach scaler for automatic normalization during inference
    model.feature_scaler = ckpt.get("feature_scaler")
    
    if device is not None:
        model = model.to(torch.device(device))
    model.eval()
    return model


def propose_fpip_v2_from_xy(
    model: UniversalProposer,
    x: np.ndarray,
    y: np.ndarray,
    top_k: int = 5,
    fit_diagnostics: Optional[Dict[str, Any]] = None,
    interaction_hints: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience wrapper: proposer inference + FPIP v2 adaptation."""
    out = propose_from_xy(model, x=x, y=y, top_k=top_k, device=device)
    return proposer_output_to_fpip_v2(
        proposer_output=out,
        fit_diagnostics=fit_diagnostics,
        interaction_hints=interaction_hints,
    )
