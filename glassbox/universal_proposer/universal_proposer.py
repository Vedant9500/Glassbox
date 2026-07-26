"""Universal fast-path proposer scaffold.

The current neural model consumes precomputed one-dimensional curve features.
For multivariate inputs it supplies heuristic operator priors while grammar
decoding and downstream search use the original multivariate `X`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from glassbox.sr.fpip_v2 import validate_fpip_v2_payload

_REPO_ROOT = Path(__file__).resolve().parents[2]


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

DEFAULT_UNIVARIATE_SKELETON_VOCAB: List[str] = [
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

DEFAULT_MULTIVARIATE_SKELETON_VOCAB: List[str] = [
    "x0",
    "x1",
    "x0^2",
    "x1^2",
    "x0*x1",
    "x0+x1",
    "x0-x1",
    "x0/(x1+1e-3)",
    "x1/(x0+1e-3)",
    "x0*sin(x1)",
    "x1*sin(x0)",
    "x0*cos(x1)",
    "x0+sin(x1)",
    "x1+sin(x0)",
    "sqrt((x0-x1)^2+1e-6)",
    "sqrt(x0^2+x1^2)",
    "x0^2+x1^2",
    "x0*x1+x0+x1",
]

DEFAULT_SKELETON_VOCAB: List[str] = list(dict.fromkeys(
    DEFAULT_UNIVARIATE_SKELETON_VOCAB + DEFAULT_MULTIVARIATE_SKELETON_VOCAB
))

UNIVERSAL_PROPOSER_ARCHITECTURE_VERSION = "universal-proposer-glu-v1"
UNIVERSAL_PROPOSER_MULTIVARIATE_CONTRACT_VERSION = "multivariate-contract-v1"
UNIVERSAL_PROPOSER_CONTRACT_VERSION = "operator-prior-grammar-planner-v1"
UNIVERSAL_PROPOSER_ROLE = "learned_operator_priors_plus_grammar_mse_planner"
UNIVERSAL_PROPOSER_MULTIVARIATE_NEURAL_MODE = "heuristic_y_projection"
UNIVERSAL_PROPOSER_UNIVARIATE_NEURAL_MODE = "canonical_univariate_xy"
SKELETON_CONFIDENCE_MIN_COVERAGE = 0.80
SKELETON_CONFIDENCE_MIN_TOP1_ACC = 0.60
SKELETON_CONFIDENCE_MIN_TOP5_ACC = 0.80
CANDIDATE_SUCCESS_REL_MSE_THRESHOLD = 1e-8


def normalize_formula_key(formula: str) -> str:
    text = str(formula)
    text = text.replace("np.", "")
    text = text.replace(" ", "")
    text = text.replace("**", "^")
    return text


def _canonicalize_formula_key_for_vocab(formula: str) -> str:
    """Normalize a formula for exact-vocab matching.

    Single-variable formulas sometimes arrive as `x0` instead of `x`; when a
    formula references only one symbolic input, collapse that lone variable to
    `x` so univariate vocab entries still match.
    """
    key = normalize_formula_key(formula)
    vars_in_key = sorted(set(re.findall(r"\bx\d+\b", key)))
    if len(vars_in_key) == 1:
        key = re.sub(r"\bx\d+\b", "x", key)
    return key


@dataclass
class UniversalProposerConfig:
    hidden_dim: int = 256
    n_features: int = 398
    supports_multivariate_formulas: bool = True
    multivariate_neural_mode: str = UNIVERSAL_PROPOSER_MULTIVARIATE_NEURAL_MODE
    proposer_contract_version: str = UNIVERSAL_PROPOSER_CONTRACT_VERSION
    max_input_vars: int = 4
    operator_vocab: Optional[List[str]] = None
    skeleton_vocab: Optional[List[str]] = None

    def resolved_operator_vocab(self) -> List[str]:
        return list(self.operator_vocab) if self.operator_vocab else list(DEFAULT_OPERATOR_VOCAB)

    def resolved_skeleton_vocab(self) -> List[str]:
        if self.skeleton_vocab:
            return list(self.skeleton_vocab)
        if self.supports_multivariate_formulas:
            return list(DEFAULT_SKELETON_VOCAB)
        return list(DEFAULT_UNIVARIATE_SKELETON_VOCAB)


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


def _safe_formula_eval_multivariate(formula: str, x: np.ndarray) -> Optional[np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2 or x.shape[1] == 0:
        return None

    context: Dict[str, Any] = {
        "np": np,
        "sin": np.sin,
        "cos": np.cos,
        "exp": lambda z: np.exp(np.clip(z, -30.0, 30.0)),
        "log": lambda z: np.log(np.abs(z) + 1e-6),
        "sqrt": lambda z: np.sqrt(np.abs(z) + 1e-6),
        "abs": np.abs,
    }
    for i in range(x.shape[1]):
        context[f"x{i}"] = x[:, i]
    if x.shape[1] == 1:
        context["x"] = x[:, 0]

    expr = formula.replace("^", "**")
    try:
        y = eval(expr, {"__builtins__": None}, context)
    except Exception:
        return None

    if isinstance(y, (int, float)):
        y = np.full(x.shape[0], float(y), dtype=np.float64)
    else:
        y = np.asarray(y, dtype=np.float64)

    if y.ndim != 1 or y.shape[0] != x.shape[0]:
        return None
    if not np.all(np.isfinite(y)):
        return None
    return y


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
        # Product templates (benchmark-driven; seeds for x**2*sin(x) class)
        "x**2*sin(x)",
        "x**2*cos(x)",
        "x**3*sin(x)",
        "x**3*cos(x)",
        "x*sin(x)**2",
        "sin(x)*cos(x)",
        # One-layer rationals (fixes rational-collapse seeds)
        "x/(1+x**2)",
        "x**2/(1+x**2)",
        "x**3/(1+x**4)",
        "x/(1+abs(x))",
        "1/(1+x**2)",
        "x**2/(1+x**4)",
        # Mixed sum/product common on mid tiers
        "x**3+sin(x)",
        "x**2+sin(x)+cos(x)",
        "exp(-x)*x",
        "exp(-x)*x**2",
    ]
    return base + composed


def _safe_formula_eval(formula: str, x: np.ndarray) -> Optional[np.ndarray]:
    if np.asarray(x).ndim > 1:
        return _safe_formula_eval_multivariate(formula, x)
    context = {
        "np": np,
        "x": x,
        "sin": np.sin,
        "cos": np.cos,
        "exp": lambda z: np.exp(np.clip(z, -30.0, 30.0)),
        "log": lambda z: np.log(np.abs(z) + 1e-6),
        "sqrt": lambda z: np.sqrt(np.abs(z) + 1e-6),
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


def _rank_columns_by_target_relevance(x: np.ndarray, y: np.ndarray) -> List[int]:
    """Order feature columns by |corr| with y (variance fallback). H-13."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = int(x.shape[1]) if x.ndim == 2 else 0
    scores: List[Tuple[float, int]] = []
    for i in range(n):
        xi = x[:, i]
        mask = np.isfinite(xi) & np.isfinite(y)
        if int(mask.sum()) < 4:
            scores.append((0.0, i))
            continue
        a = xi[mask]
        b = y[mask]
        sa = float(np.std(a))
        sb = float(np.std(b))
        if sa < 1e-12 or sb < 1e-12:
            rel = float(np.var(a))
        else:
            corr = float(np.corrcoef(a, b)[0, 1])
            rel = abs(corr) if np.isfinite(corr) else float(np.var(a))
        scores.append((rel, i))
    scores.sort(key=lambda t: (-t[0], t[1]))
    return [i for _, i in scores]


def grammar_decode_multivariate_skeletons(
    operator_priors: Dict[str, float],
    x: np.ndarray,
    y: np.ndarray,
    top_k: int = 5,
    max_rank: int = 2,
) -> List[Dict[str, Any]]:
    """Decode multivariate skeletons from a constrained algebraic grammar."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or x.shape[1] < 2:
        return grammar_decode_topk_skeletons(operator_priors, x.reshape(-1), y, top_k=top_k, max_depth=2)

    y_var = float(np.var(y)) + 1e-12
    scored: List[Dict[str, Any]] = []
    feature_names = [f"x{i}" for i in range(x.shape[1])]
    n_features = int(x.shape[1])
    limit = min(n_features, max(2, int(max_rank)))
    # H-13: pair among target-relevant columns, not the column-index prefix.
    # Small-D: enumerate all pairs (cheap). Larger-D: top-`limit` by |corr(y, xi)|.
    if n_features <= 8:
        selected_cols = list(range(n_features))
    else:
        ranked = _rank_columns_by_target_relevance(x, y)
        selected_cols = sorted(ranked[:limit])

    for a_idx in range(len(selected_cols)):
        for b_idx in range(a_idx + 1, len(selected_cols)):
            i = selected_cols[a_idx]
            j = selected_cols[b_idx]
            xi = x[:, i]
            xj = x[:, j]
            mask = np.isfinite(xi) & np.isfinite(xj) & np.isfinite(y)
            if int(mask.sum()) < 8:
                continue
            x_pair_full = x[mask, :]
            xi = xi[mask]
            xj = xj[mask]
            yj = y[mask]
            vi = feature_names[i]
            vj = feature_names[j]
            candidates = [
                f"{vi}+{vj}",
                f"{vi}-{vj}",
                f"{vi}*{vj}",
                f"{vi}/({vj}+1e-3)",
                f"{vj}/({vi}+1e-3)",
                f"{vi}^2+{vj}^2",
                f"{vi}*sin({vj})",
                f"{vj}*sin({vi})",
                f"{vi}*cos({vj})",
                f"{vj}*cos({vi})",
                f"sqrt(({vi}-{vj})^2+1e-6)",
                f"sqrt({vi}^2+{vj}^2)",
                f"{vi}*{vj}+{vi}+{vj}",
            ]
            for formula in candidates:
                basis = _safe_formula_eval_multivariate(formula, x_pair_full)
                mse = float("inf")
                fit_score = 0.0
                if basis is not None:
                    mse = _fit_affine_mse(yj, basis)
                    fit_score = float(np.exp(-mse / y_var)) if np.isfinite(mse) else 0.0
                tags = _formula_operator_tags(formula)
                if tags:
                    prior_score = float(np.mean([operator_priors.get(t, 1e-6) for t in tags]))
                else:
                    prior_score = 1e-6
                score = 0.6 * prior_score + 0.4 * fit_score
                scored.append(
                    {
                        "formula": formula,
                        "probability": float(max(1e-9, score)),
                        "score": float(1.0 - min(score, 1.0)),
                        "mse": None if not np.isfinite(mse) else float(mse),
                    }
                )

    if not scored:
        return grammar_decode_topk_skeletons(operator_priors, x[:, 0], y, top_k=top_k, max_depth=2)
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


def _proposer_model_contract(
    *,
    is_multivariate: bool,
    n_input_vars: int,
    neural_feature_mode: str,
) -> Dict[str, Any]:
    if is_multivariate:
        operator_prior_source = (
            "caller_supplied_features"
            if neural_feature_mode == "caller_supplied_features"
            else "one_dimensional_y_projection_features"
        )
        return {
            "contract_version": UNIVERSAL_PROPOSER_MULTIVARIATE_CONTRACT_VERSION,
            "input_mode": "multivariate",
            "n_input_features": int(n_input_vars),
            "supports_multivariate_formulas": True,
            "multivariate_candidate_mode": "grammar_search_original_X",
            "neural_multivariate_support": "heuristic",
            "neural_feature_mode": neural_feature_mode,
            "supports_trained_multivariate_neural_model": False,
            "operator_prior_source": operator_prior_source,
            "notes": [
                "Neural operator priors are not produced by a trained multivariate point-set model.",
                "Multivariate candidates and search planning use the original X through grammar/MSE heuristics.",
            ],
        }
    operator_prior_source = (
        "caller_supplied_features"
        if neural_feature_mode == "caller_supplied_features"
        else "canonicalized_univariate_xy_features"
    )
    return {
        "contract_version": UNIVERSAL_PROPOSER_MULTIVARIATE_CONTRACT_VERSION,
        "input_mode": "univariate",
        "n_input_features": 1,
        "supports_multivariate_formulas": False,
        "multivariate_candidate_mode": "not_applicable",
        "neural_multivariate_support": "not_applicable",
        "neural_feature_mode": neural_feature_mode,
        "supports_trained_multivariate_neural_model": False,
        "operator_prior_source": operator_prior_source,
    }


def _float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        out = float(value)
        return out if np.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _skeleton_confidence_reliability(metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = metrics if isinstance(metrics, dict) else {}
    coverage = _float_or_none(metrics.get("skeleton_coverage"))
    top1 = _float_or_none(metrics.get("skeleton_top1_acc"))
    top5 = _float_or_none(metrics.get("skeleton_top5_acc"))

    missing = []
    if coverage is None:
        missing.append("skeleton_coverage")
    if top1 is None:
        missing.append("skeleton_top1_acc")
    if top5 is None:
        missing.append("skeleton_top5_acc")

    reliable = (
        coverage is not None and coverage >= SKELETON_CONFIDENCE_MIN_COVERAGE
        and top1 is not None and top1 >= SKELETON_CONFIDENCE_MIN_TOP1_ACC
        and top5 is not None and top5 >= SKELETON_CONFIDENCE_MIN_TOP5_ACC
    )
    reasons = []
    if coverage is None or coverage < SKELETON_CONFIDENCE_MIN_COVERAGE:
        reasons.append("insufficient_skeleton_coverage")
    if top1 is None or top1 < SKELETON_CONFIDENCE_MIN_TOP1_ACC:
        reasons.append("insufficient_skeleton_top1_acc")
    if top5 is None or top5 < SKELETON_CONFIDENCE_MIN_TOP5_ACC:
        reasons.append("insufficient_skeleton_top5_acc")

    return {
        "reliable": bool(reliable),
        "coverage": coverage,
        "top1_acc": top1,
        "top5_acc": top5,
        "required_coverage": SKELETON_CONFIDENCE_MIN_COVERAGE,
        "required_top1_acc": SKELETON_CONFIDENCE_MIN_TOP1_ACC,
        "required_top5_acc": SKELETON_CONFIDENCE_MIN_TOP5_ACC,
        "missing_metrics": missing,
        "reasons": [] if reliable else reasons,
    }


def _routing_calibration_status(model: Optional[UniversalProposer]) -> Dict[str, Any]:
    calibration = getattr(model, "routing_calibration", None)
    if isinstance(calibration, dict):
        return dict(calibration)
    return {
        "status": "uncalibrated",
        "method": "candidate_mse_gate_plus_validation_gated_skeleton_confidence",
        "requires": "downstream_candidate_success_benchmark",
    }


def _proposer_contract(
    *,
    skeleton_reliability: Dict[str, Any],
    routing_calibration: Dict[str, Any],
) -> Dict[str, Any]:
    skeleton_role = (
        "validated_confidence_signal"
        if skeleton_reliability.get("reliable")
        else "diagnostic_only"
    )
    return {
        "contract_version": UNIVERSAL_PROPOSER_CONTRACT_VERSION,
        "role": UNIVERSAL_PROPOSER_ROLE,
        "operator_prior_role": "learned_neural_hint",
        "candidate_generation": "grammar_decode_with_mse_ranking",
        "skeleton_head_role": skeleton_role,
        "skeleton_confidence_reliability": dict(skeleton_reliability),
        "routing_calibration": dict(routing_calibration),
    }


def _uncertainty_from_logits(
    logits: Sequence[float],
    skeleton_reliability: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    probs = _safe_softmax(np.asarray(logits, dtype=np.float64))
    skeleton_reliability = skeleton_reliability or _skeleton_confidence_reliability(None)
    if probs.size == 0:
        return {
            "entropy": None,
            "margin": None,
            "raw_entropy": None,
            "raw_margin": None,
            "confident": False,
            "raw_confident": False,
            "skeleton_confidence_reliable": bool(skeleton_reliability.get("reliable", False)),
            "confidence_source": "no_skeleton_logits",
            "skeleton_reliability": dict(skeleton_reliability),
        }

    sorted_probs = np.sort(probs)[::-1]
    top1 = float(sorted_probs[0])
    top2 = float(sorted_probs[1]) if sorted_probs.size > 1 else 0.0
    entropy = 0.0
    if sorted_probs.size > 1:
        entropy = float(-np.sum(sorted_probs * np.log(sorted_probs + 1e-12)) / np.log(sorted_probs.size))
    margin = top1 - top2
    raw_confident = bool(entropy < 0.65 and margin > 0.12)
    reliable = bool(skeleton_reliability.get("reliable", False))
    confident = bool(raw_confident and reliable)
    if reliable:
        confidence_source = "validated_skeleton_logits"
        routed_entropy = entropy
        routed_margin = margin
    else:
        confidence_source = "disabled_unvalidated_skeleton_head"
        routed_entropy = None
        routed_margin = None
    return {
        "entropy": routed_entropy,
        "margin": routed_margin,
        "raw_entropy": entropy,
        "raw_margin": margin,
        "confident": confident,
        "raw_confident": raw_confident,
        "skeleton_confidence_reliable": reliable,
        "confidence_source": confidence_source,
        "skeleton_reliability": dict(skeleton_reliability),
    }


def _signal_complexity(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Cheap curve complexity diagnostics for search planning."""
    x_arr = np.asarray(x, dtype=np.float64)
    x = x_arr[:, 0].reshape(-1) if x_arr.ndim == 2 else x_arr.reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    out = {
        "roughness": 0.0,
        "turning_rate": 0.0,
        "y_variance": float(np.var(y)) if y.size else 0.0,
    }
    if x.size < 5 or y.size < 5:
        return out

    try:
        finite = np.isfinite(x) & np.isfinite(y)
        x_valid = x[finite]
        y_valid = y[finite]
        if x_valid.size < 5:
            return out

        order = np.argsort(x_valid)
        x_sorted = x_valid[order]
        y_sorted = y_valid[order]
        x_unique, unique_idx = np.unique(x_sorted, return_index=True)
        if x_unique.size < 5:
            return out
        y_unique = y_sorted[unique_idx]

        dy = np.gradient(y_unique, x_unique)
        ddy = np.gradient(dy, x_unique)
        if not (np.all(np.isfinite(dy)) and np.all(np.isfinite(ddy))):
            return out
        y_scale = float(np.std(y)) + 1e-12
        x_span = float(np.max(x_unique) - np.min(x_unique)) + 1e-12
        out["roughness"] = float(np.clip(np.mean(np.abs(ddy)) * x_span / y_scale, 0.0, 10.0))

        signs = np.sign(dy)
        signs[np.abs(dy) < 1e-10] = 0.0
        nz = signs[signs != 0.0]
        if nz.size > 1:
            out["turning_rate"] = float(np.mean(nz[1:] != nz[:-1]))
    except Exception:
        pass
    return out


def build_search_plan(
    *,
    operator_priors: Dict[str, float],
    candidates: Sequence[Dict[str, Any]],
    uncertainty: Dict[str, Any],
    x: np.ndarray,
    y: np.ndarray,
) -> Dict[str, Any]:
    """Build an evolution search plan from proposer evidence.

    This is intentionally heuristic for now. It gives the proposer a planner
    role without requiring a new checkpoint format. Future training can replace
    these rules with explicit budget/depth heads.
    """
    complexity = _signal_complexity(x, y)
    y_var = max(float(complexity.get("y_variance", 0.0)), 1e-12)

    finite_mses = [
        float(c.get("mse"))
        for c in candidates
        if c.get("mse") is not None and np.isfinite(float(c.get("mse")))
    ]
    best_rel_mse = (min(finite_mses) / y_var) if finite_mses else float("inf")

    entropy = uncertainty.get("entropy")
    margin = uncertainty.get("margin")
    entropy_f = 0.75 if entropy is None else float(np.clip(entropy, 0.0, 1.0))
    margin_f = 0.0 if margin is None else float(np.clip(margin, 0.0, 1.0))
    uncertain = 0.65 * entropy_f + 0.35 * (1.0 - margin_f)

    roughness = float(complexity.get("roughness", 0.0))
    turning_rate = float(complexity.get("turning_rate", 0.0))
    has_periodic = max(operator_priors.get("periodic", 0.0), operator_priors.get("sin", 0.0), operator_priors.get("cos", 0.0))
    has_power = operator_priors.get("power", 0.0)
    has_exp = operator_priors.get("exp", 0.0)
    has_log = operator_priors.get("log", 0.0)
    has_rational = operator_priors.get("rational", 0.0)

    difficulty = 0.0
    difficulty += 0.35 * uncertain
    difficulty += 0.20 * float(np.clip(np.log10(best_rel_mse + 1e-12) + 6.0, 0.0, 6.0) / 6.0)
    difficulty += 0.15 * float(np.clip(roughness / 4.0, 0.0, 1.0))
    difficulty += 0.10 * float(np.clip(turning_rate * 3.0, 0.0, 1.0))
    difficulty += 0.10 * float(max(has_rational, has_exp, has_log))
    difficulty += 0.10 * float(has_periodic > 0.4 and has_power > 0.25)
    difficulty = float(np.clip(difficulty, 0.0, 1.0))

    if best_rel_mse < 1e-8 and uncertainty.get("confident") is True:
        strategy = "refine_seed"
    elif difficulty < 0.35:
        strategy = "focused"
    elif difficulty < 0.70:
        strategy = "balanced"
    else:
        strategy = "exploratory"

    population_multiplier = float(0.7 + 1.4 * difficulty)
    generation_multiplier = float(0.8 + 2.2 * difficulty)
    n_beams = int(np.clip(round(3 + 8 * difficulty), 3, 12))
    n_rounds = 1 if difficulty < 0.65 else 2

    p_min = -2.0
    p_max = 3.0
    if has_power > 0.35 or has_rational > 0.25:
        p_max = 5.0
    if roughness > 3.0 and has_power > 0.25:
        p_max = 6.0
    if has_rational > 0.35:
        p_min = -4.0

    max_complexity = int(np.clip(round(10 + 28 * difficulty + 8 * max(has_rational, has_periodic)), 10, 50))
    seed_budget = int(np.clip(len(candidates) + round(4 + 8 * difficulty), 4, 16))

    return {
        "strategy": strategy,
        "difficulty": difficulty,
        "population_multiplier": population_multiplier,
        "generation_multiplier": generation_multiplier,
        "n_beams": n_beams,
        "n_rounds": n_rounds,
        "p_min": p_min,
        "p_max": p_max,
        "early_stop_max_nodes": max_complexity,
        "acceptable_complexity": min(max_complexity, 20),
        "seed_budget": seed_budget,
        "signals": {
            "best_relative_mse": None if not np.isfinite(best_rel_mse) else float(best_rel_mse),
            "uncertainty": float(uncertain),
            "roughness": roughness,
            "turning_rate": turning_rate,
        },
    }


def build_multivariate_search_plan(
    *,
    operator_priors: Dict[str, float],
    candidates: Sequence[Dict[str, Any]],
    uncertainty: Dict[str, Any],
    x: np.ndarray,
    y: np.ndarray,
    input_variables: Optional[List[str]] = None,
) -> Dict[str, Any]:
    plan = build_search_plan(
        operator_priors=operator_priors,
        candidates=candidates,
        uncertainty=uncertainty,
        x=x[:, 0] if np.asarray(x).ndim == 2 and np.asarray(x).shape[1] > 0 else x,
        y=y,
    )
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim != 2 or x_arr.shape[1] < 2:
        plan["supports_multivariate_formulas"] = False
        return plan

    n_features = int(x_arr.shape[1])
    input_variables = list(input_variables) if input_variables else [f"x{i}" for i in range(n_features)]
    interaction_strength = 0.0
    for term in candidates:
        formula = str(term.get("formula", ""))
        if "*" in formula or "/" in formula or "sqrt((" in formula:
            interaction_strength = max(interaction_strength, float(term.get("probability", 0.0)))

    plan["supports_multivariate_formulas"] = True
    plan["contract_version"] = UNIVERSAL_PROPOSER_MULTIVARIATE_CONTRACT_VERSION
    plan["neural_multivariate_support"] = "heuristic"
    plan["neural_feature_mode"] = UNIVERSAL_PROPOSER_MULTIVARIATE_NEURAL_MODE
    plan["supports_trained_multivariate_neural_model"] = False
    plan["operator_prior_source"] = "one_dimensional_y_projection_features"
    plan["candidate_source"] = "multivariate_grammar_with_mse_ranking"
    plan["input_variables"] = input_variables[: min(len(input_variables), n_features)]
    plan["feature_count"] = n_features
    plan["interaction_strength"] = float(interaction_strength)
    plan["seed_budget"] = max(int(plan.get("seed_budget", 0)), min(24, 6 + 2 * n_features))
    plan["generation_multiplier"] = float(plan.get("generation_multiplier", 1.0)) * (1.0 + 0.1 * max(0, n_features - 1))
    plan["population_multiplier"] = float(plan.get("population_multiplier", 1.0)) * (1.0 + 0.08 * max(0, n_features - 1))
    if interaction_strength > 0.2:
        plan["early_stop_max_nodes"] = max(int(plan.get("early_stop_max_nodes", 20)), 24 + 4 * n_features)
        plan["acceptable_complexity"] = max(int(plan.get("acceptable_complexity", 15)), 12 + 2 * n_features)
    return plan


def propose_from_xy(
    model: UniversalProposer,
    x: np.ndarray,
    y: np.ndarray,
    top_k: int = 5,
    device: Optional[str] = None,
    features: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Run proposer on a single curve and return decoded candidates + priors."""
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim == 2 and x.shape[1] == 1:
        x = x[:, 0]
    elif x.ndim not in (1, 2):
        raise ValueError(f"Expected x to be 1D, [N,1], or [N,D], got shape {x.shape}")

    y = y.reshape(-1)
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have same length")

    is_multivariate = bool(x.ndim == 2 and x.shape[1] > 1)
    neural_feature_mode = "caller_supplied_features"

    # If features are not provided, we must extract them (legacy behavior)
    if features is None:
        from glassbox.curve_classifier.generate_curve_data import (
            extract_all_features,
            extract_all_features_xy,
        )
        if not is_multivariate:
            x_univariate = x.reshape(-1)
            features = extract_all_features_xy(x_univariate, y)
            neural_feature_mode = UNIVERSAL_PROPOSER_UNIVARIATE_NEURAL_MODE
        else:
            # Multivariate neural features remain heuristic until the point-set
            # model phase; runtime candidates still use the multivariate grammar.
            features = extract_all_features(y)
            neural_feature_mode = UNIVERSAL_PROPOSER_MULTIVARIATE_NEURAL_MODE
    
    # Handle dimension mismatch (e.g. model trained with 370 features, codebase extracts 398)
    expected_dim = model.config.n_features
    if len(features) > expected_dim:
        features = features[:expected_dim]
    elif len(features) < expected_dim:
        features = np.pad(features, (0, expected_dim - len(features)))

    # Apply the same selective SymLog + scaling used during proposer training.
    features = np.array(features, dtype=np.float32, copy=True)
    if features.ndim == 1 and features.shape[0] > 192:
        end = min(features.shape[0], expected_dim)
        features[192:end] = np.sign(features[192:end]) * np.log1p(np.abs(features[192:end]))
    elif features.ndim > 1 and features.shape[1] > 192:
        end = min(features.shape[1], expected_dim)
        features[:, 192:end] = np.sign(features[:, 192:end]) * np.log1p(np.abs(features[:, 192:end]))

    scaler = getattr(model, "feature_scaler", None)
    if isinstance(scaler, dict) and "mean" in scaler and "std" in scaler:
        mean = np.asarray(scaler["mean"], dtype=np.float32)
        std = np.asarray(scaler["std"], dtype=np.float32)
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
    x_for_plan = x.astype(np.float64)
    y_for_plan = y.astype(np.float64)
    n_input_vars = int(x_for_plan.shape[1]) if is_multivariate else 1

    if x_for_plan.ndim == 1 or (x_for_plan.ndim == 2 and x_for_plan.shape[1] <= 1):
        candidates = grammar_decode_topk_skeletons(
            priors,
            x=x_for_plan.reshape(-1),
            y=y_for_plan,
            top_k=top_k,
            max_depth=2,
        )
    else:
        candidates = grammar_decode_multivariate_skeletons(
            priors,
            x=x_for_plan,
            y=y_for_plan,
            top_k=top_k,
            max_rank=min(model.config.max_input_vars, x_for_plan.shape[1]),
        )

    # Fallback to direct head decode if grammar decoding unexpectedly returns empty.
    if not candidates:
        candidates = decode_topk_skeletons(skeleton_logits, model.skeleton_vocab, top_k=top_k)

    skeleton_reliability = _skeleton_confidence_reliability(
        getattr(model, "validation_metrics", None)
    )
    routing_calibration = _routing_calibration_status(model)
    proposer_contract = _proposer_contract(
        skeleton_reliability=skeleton_reliability,
        routing_calibration=routing_calibration,
    )
    uncertainty = _uncertainty_from_logits(skeleton_logits, skeleton_reliability)
    if x_for_plan.ndim == 1 or (x_for_plan.ndim == 2 and x_for_plan.shape[1] <= 1):
        search_plan = build_search_plan(
            operator_priors=priors,
            candidates=candidates,
            uncertainty=uncertainty,
            x=x_for_plan.reshape(-1),
            y=y_for_plan,
        )
    else:
        search_plan = build_multivariate_search_plan(
            operator_priors=priors,
            candidates=candidates,
            uncertainty=uncertainty,
            x=x_for_plan,
            y=y_for_plan,
            input_variables=[f"x{i}" for i in range(x_for_plan.shape[1])],
        )

    model_contract = _proposer_model_contract(
        is_multivariate=is_multivariate,
        n_input_vars=n_input_vars,
        neural_feature_mode=neural_feature_mode,
    )
    search_plan["model_contract"] = dict(model_contract)
    search_plan["proposer_contract"] = dict(proposer_contract)
    search_plan["contract_version"] = model_contract["contract_version"]
    search_plan["neural_feature_mode"] = model_contract["neural_feature_mode"]
    search_plan["operator_prior_source"] = model_contract["operator_prior_source"]
    search_plan["supports_trained_multivariate_neural_model"] = model_contract[
        "supports_trained_multivariate_neural_model"
    ]
    if is_multivariate:
        search_plan["neural_multivariate_support"] = model_contract["neural_multivariate_support"]

    return {
        "candidate_skeletons": candidates,
        "operator_priors": priors,
        "sequence_uncertainty": uncertainty,
        "search_plan": search_plan,
        "supports_multivariate_formulas": is_multivariate,
        "model_contract": model_contract,
        "proposer_contract": proposer_contract,
        "neural_feature_mode": neural_feature_mode,
        "input_variables": [f"x{i}" for i in range(x_for_plan.shape[1])] if x_for_plan.ndim == 2 else ["x"],
    }


def _routing_signal_from_proposer_output(
    proposer_output: Dict[str, Any],
    search_plan: Dict[str, Any],
    confident: bool,
) -> Dict[str, Any]:
    uncertainty = proposer_output.get("sequence_uncertainty", {})
    if not isinstance(uncertainty, dict):
        uncertainty = {}
    signals = search_plan.get("signals", {}) if isinstance(search_plan, dict) else {}
    best_rel_mse = _float_or_none(signals.get("best_relative_mse"))
    skeleton_reliable = bool(uncertainty.get("skeleton_confidence_reliable", False))
    confidence_source = str(uncertainty.get("confidence_source") or "unknown")

    if best_rel_mse is not None and best_rel_mse <= CANDIDATE_SUCCESS_REL_MSE_THRESHOLD:
        return {
            "recommend_guided_evolution": False,
            "reason": "candidate_verified_by_mse",
            "confidence_source": "grammar_candidate_mse",
            "best_relative_mse": best_rel_mse,
            "candidate_success_threshold": CANDIDATE_SUCCESS_REL_MSE_THRESHOLD,
            "skeleton_confidence_reliable": skeleton_reliable,
            "calibration_status": "per_curve_candidate_success",
        }

    if confident and skeleton_reliable:
        return {
            "recommend_guided_evolution": False,
            "reason": "validated_skeleton_confidence",
            "confidence_source": confidence_source,
            "best_relative_mse": best_rel_mse,
            "candidate_success_threshold": CANDIDATE_SUCCESS_REL_MSE_THRESHOLD,
            "skeleton_confidence_reliable": True,
            "calibration_status": "validation_metric_gated",
        }

    reason = "proposer_low_confidence"
    if not skeleton_reliable:
        reason = "unvalidated_skeleton_confidence"
    return {
        "recommend_guided_evolution": True,
        "reason": reason,
        "confidence_source": confidence_source,
        "best_relative_mse": best_rel_mse,
        "candidate_success_threshold": CANDIDATE_SUCCESS_REL_MSE_THRESHOLD,
        "skeleton_confidence_reliable": skeleton_reliable,
        "calibration_status": "requires_downstream_success_benchmark",
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
    if not isinstance(uncertainty, dict):
        uncertainty = {}
    confident = bool(uncertainty.get("confident") is True)
    raw_search_plan = proposer_output.get("search_plan", {})
    search_plan = dict(raw_search_plan) if isinstance(raw_search_plan, dict) else {}
    routing_signal = _routing_signal_from_proposer_output(proposer_output, search_plan, confident)
    raw_model_contract = proposer_output.get("model_contract", {})
    model_contract = dict(raw_model_contract) if isinstance(raw_model_contract, dict) else {}
    raw_proposer_contract = proposer_output.get("proposer_contract", {})
    proposer_contract = dict(raw_proposer_contract) if isinstance(raw_proposer_contract, dict) else {}

    payload = {
        "schema_version": "fpip.v2",
        "candidate_skeletons": list(proposer_output.get("candidate_skeletons", [])),
        "sequence_uncertainty": {
            "entropy": uncertainty.get("entropy"),
            "margin": uncertainty.get("margin"),
            "confident": confident,
            "raw_entropy": uncertainty.get("raw_entropy"),
            "raw_margin": uncertainty.get("raw_margin"),
            "raw_confident": uncertainty.get("raw_confident"),
            "confidence_source": uncertainty.get("confidence_source"),
            "skeleton_confidence_reliable": uncertainty.get("skeleton_confidence_reliable"),
        },
        "operator_priors": dict(proposer_output.get("operator_priors", {})),
        "interaction_hints": dict(interaction_hints),
        "fit_diagnostics": dict(fit_diagnostics),
        "search_plan": search_plan,
        "model_contract": model_contract,
        "proposer_contract": proposer_contract,
        "routing_signal": routing_signal,
    }

    valid, errors = validate_fpip_v2_payload(payload)
    payload["valid"] = valid
    if not valid:
        payload["validation_errors"] = errors
    return payload


def _is_trusted_checkpoint_path(checkpoint_path: Path) -> bool:
    resolved = checkpoint_path.resolve()
    trusted_roots = [
        (_REPO_ROOT / "models").resolve(),
        (_REPO_ROOT / "artifacts").resolve(),
    ]
    return any(resolved == root or root in resolved.parents for root in trusted_roots)


def _load_torch_checkpoint(checkpoint_path: Path):
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as safe_error:
        if not _is_trusted_checkpoint_path(checkpoint_path):
            raise RuntimeError(
                "Refusing unsafe pickle checkpoint load outside trusted local model directories. "
                f"Move {checkpoint_path} under models/ or artifacts/, or convert it to a weights-only checkpoint."
            ) from safe_error
        if os.environ.get("GLASSBOX_VERBOSE_CHECKPOINT_LOAD"):
            print(
                "weights-only checkpoint load failed; falling back to trusted local "
                f"pickle checkpoint at {checkpoint_path}."
            )
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def validate_universal_proposer_checkpoint_metadata(
    checkpoint: Dict[str, Any],
    *,
    strict: bool = False,
) -> Dict[str, Any]:
    """Validate proposer checkpoint metadata with legacy compatibility."""
    warnings: List[str] = []
    errors: List[str] = []

    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dictionary")
    if "model_state_dict" not in checkpoint:
        errors.append("missing model_state_dict")

    cfg = checkpoint.get("config")
    if not isinstance(cfg, dict):
        errors.append("missing config")
        cfg = {}

    n_features = cfg.get("n_features")
    try:
        n_features = int(n_features)
        if n_features <= 0:
            errors.append("config.n_features must be positive")
    except Exception:
        warnings.append("missing or invalid config.n_features")
        n_features = None

    skeleton_vocab = cfg.get("skeleton_vocab")
    if skeleton_vocab is not None and not isinstance(skeleton_vocab, list):
        errors.append("config.skeleton_vocab must be a list when present")

    operator_vocab = cfg.get("operator_vocab")
    if operator_vocab is not None and not isinstance(operator_vocab, list):
        errors.append("config.operator_vocab must be a list when present")

    architecture_version = (
        checkpoint.get("architecture_version")
        or cfg.get("architecture_version")
    )
    if architecture_version is None:
        architecture_version = "legacy_unversioned"
        warnings.append("missing architecture_version; treating checkpoint as legacy")

    if errors or (strict and warnings):
        details = "; ".join(errors + warnings)
        raise ValueError(f"Invalid universal proposer checkpoint metadata: {details}")

    return {
        "n_features": n_features,
        "architecture_version": architecture_version,
        "multivariate_neural_mode": str(
            cfg.get("multivariate_neural_mode", UNIVERSAL_PROPOSER_MULTIVARIATE_NEURAL_MODE)
        ),
        "proposer_contract_version": str(
            checkpoint.get("proposer_contract_version")
            or cfg.get("proposer_contract_version")
            or UNIVERSAL_PROPOSER_CONTRACT_VERSION
        ),
        "warnings": warnings,
    }


def load_universal_proposer_checkpoint(
    checkpoint_path: str,
    device: Optional[str] = None,
) -> UniversalProposer:
    """Load UniversalProposer from checkpoint saved by train_universal_proposer.py."""
    ckpt = _load_torch_checkpoint(Path(checkpoint_path))
    metadata_report = validate_universal_proposer_checkpoint_metadata(ckpt)
    if os.environ.get("GLASSBOX_VERBOSE_CHECKPOINT_LOAD"):
        for warning in metadata_report.get("warnings", []):
            print(f"  Proposer checkpoint metadata warning: {warning}")
    cfg_raw = ckpt.get("config", {})
    state_dict = ckpt["model_state_dict"]
    skeleton_vocab = cfg_raw.get("skeleton_vocab")
    if skeleton_vocab is None:
        head_weight = state_dict.get("skeleton_head.weight")
        if head_weight is not None and int(head_weight.shape[0]) == len(DEFAULT_UNIVARIATE_SKELETON_VOCAB):
            skeleton_vocab = list(DEFAULT_UNIVARIATE_SKELETON_VOCAB)
        else:
            skeleton_vocab = list(DEFAULT_SKELETON_VOCAB)
    
    # Map new GLU config (n_features) and handle legacy point_mlp_layers
    config = UniversalProposerConfig(
        hidden_dim=int(cfg_raw.get("hidden_dim", 256)),
        n_features=int(cfg_raw.get("n_features", 370)),
        supports_multivariate_formulas=bool(cfg_raw.get("supports_multivariate_formulas", True)),
        multivariate_neural_mode=str(
            cfg_raw.get("multivariate_neural_mode", UNIVERSAL_PROPOSER_MULTIVARIATE_NEURAL_MODE)
        ),
        proposer_contract_version=str(
            cfg_raw.get("proposer_contract_version", UNIVERSAL_PROPOSER_CONTRACT_VERSION)
        ),
        max_input_vars=int(cfg_raw.get("max_input_vars", 4)),
        operator_vocab=cfg_raw.get("operator_vocab"),
        skeleton_vocab=skeleton_vocab,
    )
    model = UniversalProposer(config)
    model.load_state_dict(state_dict)
    model.architecture_version = metadata_report.get("architecture_version")
    model.proposer_contract_version = metadata_report.get("proposer_contract_version")
    validation_metrics = ckpt.get("validation_metrics")
    model.validation_metrics = dict(validation_metrics) if isinstance(validation_metrics, dict) else {}
    routing_calibration = ckpt.get("routing_calibration")
    model.routing_calibration = dict(routing_calibration) if isinstance(routing_calibration, dict) else None
    
    # Attach scaler for automatic normalization during inference. Some older
    # proposer checkpoints accidentally stored an AMP GradScaler here.
    feature_scaler = ckpt.get("feature_scaler")
    if not (
        isinstance(feature_scaler, dict)
        and "mean" in feature_scaler
        and "std" in feature_scaler
    ):
        feature_scaler = None
    else:
        feature_scaler = {
            "mean": np.asarray(feature_scaler["mean"], dtype=np.float32),
            "std": np.asarray(feature_scaler["std"], dtype=np.float32),
        }
    model.feature_scaler = feature_scaler
    
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
