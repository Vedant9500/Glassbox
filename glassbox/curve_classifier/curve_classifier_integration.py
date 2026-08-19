"""
Curve Classifier Integration with ONN

Uses the trained curve classifier to predict operators and warm-start ONN evolution.

Usage:
    # Test on synthetic data
    python -m glassbox.curve_classifier.curve_classifier_integration --model models/curve_classifier_multi.pt --formula "sin(x) + x**2"
    
    # Integrate with ONN (in your training script)
    from scripts.curve_classifier_integration import predict_operators, bias_onn_from_predictions
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import os
from glassbox.model_registry import (
    DEFAULT_CURVE_CLASSIFIER_PATH,
    resolve_curve_classifier_path,
)

# Add repo root to path for imports
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from .generate_curve_data import extract_all_features, extract_all_features_xy, OPERATOR_CLASSES
except (ImportError, ValueError):
    try:
        from glassbox.curve_classifier.generate_curve_data import extract_all_features, extract_all_features_xy, OPERATOR_CLASSES
    except ImportError:
        try:
            import glassbox.curve_classifier.generate_curve_data as gcd
            extract_all_features = gcd.extract_all_features
            extract_all_features_xy = gcd.extract_all_features_xy
            OPERATOR_CLASSES = gcd.OPERATOR_CLASSES
        except ImportError:
            from glassbox.curve_classifier.generate_curve_data import extract_all_features, extract_all_features_xy, OPERATOR_CLASSES

CURVE_CLASSIFIER_UNIVARIATE_NEURAL_MODE = "canonical_univariate_xy"
CURVE_CLASSIFIER_MULTIVARIATE_NEURAL_MODE = "heuristic_slice_aggregation"

# S9-1: last load/predict status (fail-open diagnostics). Not mixed into
# operator probability dicts so callers treating predictions as {op: float}
# remain correct (empty dict still means "no operators / fail-open").
LAST_CLASSIFIER_LOAD_STATUS: Dict[str, object] = {
    "ok": False,
    "reason": "never_called",
    "fail_open": None,
}


def get_last_classifier_load_status() -> Dict[str, object]:
    """Return a copy of the most recent classifier load/predict status (S9-1)."""
    return dict(LAST_CLASSIFIER_LOAD_STATUS)


def describe_curve_classifier_inference(x: np.ndarray) -> Dict[str, object]:
    """Describe the public classifier inference contract for the given input shape."""
    x_arr = np.asarray(x)
    if x_arr.ndim == 1:
        n_vars = 1
    elif x_arr.ndim == 2:
        n_vars = int(x_arr.shape[1])
    else:
        raise ValueError(f"Expected x to be 1D, [N,1], or [N,D], got shape {x_arr.shape}")

    if n_vars <= 1:
        return {
            "input_mode": "univariate",
            "n_input_features": 1,
            "status": "trained_univariate_neural",
            "neural_feature_mode": CURVE_CLASSIFIER_UNIVARIATE_NEURAL_MODE,
            "supports_trained_multivariate_neural_model": False,
            "operator_prior_source": "canonicalized_univariate_xy_features",
        }

    return {
        "input_mode": "multivariate",
        "n_input_features": n_vars,
        "status": "heuristic_multivariate",
        "neural_feature_mode": CURVE_CLASSIFIER_MULTIVARIATE_NEURAL_MODE,
        "supports_trained_multivariate_neural_model": False,
        "operator_prior_source": "per_variable_xy_slices_plus_interaction_diagonal_slices",
        "notes": [
            "The neural classifier is trained on one-dimensional curve features.",
            "Multivariate inference uses interpolation slices and max-aggregated univariate priors.",
        ],
    }

# =============================================================================
# MODEL DEFINITION (must match training)
# =============================================================================

class CurveClassifierMLP(nn.Module):
    """Deep MLP classifier for curve features."""
    
    def __init__(self, n_features: int = 398, n_classes: int = 9, hidden: int = 512):
        super().__init__()
        
        eql_out_dim = 256
        self.eql = EQLLayer(in_features=n_features, out_features=eql_out_dim)
        
        layers = []
        combined_dim = n_features + eql_out_dim
        
        layers.extend([
            nn.Linear(combined_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.2)
        ])
        
        for _ in range(6):
            layers.extend([
                nn.Linear(hidden, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            
        layers.extend([
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, n_classes)
        ])
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        eql_feats = self.eql(x)
        combined = torch.cat([x, eql_feats], dim=1)
        return self.net(combined)


class CurveClassifierCNN(nn.Module):
    """1D CNN classifier matching training architecture."""

    def __init__(self, n_classes: int = 9, n_features: int = 398, curve_dim: int = 128):
        super().__init__()
        self.curve_dim = min(curve_dim, n_features)

        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
        )

        other_dim = max(1, n_features - self.curve_dim)
        self.other_mlp = nn.Sequential(
            nn.Linear(other_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        raw_curve = x[:, :self.curve_dim]
        other_features = x[:, self.curve_dim:]

        raw_curve = raw_curve.unsqueeze(1)
        conv_out = self.conv(raw_curve)
        conv_out = conv_out.flatten(1)
        other_out = self.other_mlp(other_features)

        combined = torch.cat([conv_out, other_out], dim=1)
        return self.classifier(combined)


class SemanticFeatureAttention(nn.Module):
    """
    Semantic Attention that treats each feature group as a distinct token.
    Allows the model to attend across different modalities (FFT, derivatives, stats, etc.).
    """
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Project each semantic group independently
        self.proj_raw = nn.Linear(128, embed_dim)
        self.proj_fft = nn.Linear(32, embed_dim)
        self.proj_fft_phase = nn.Linear(32, embed_dim)
        self.proj_deriv = nn.Linear(128, embed_dim)
        self.proj_stats = nn.Linear(9, embed_dim)
        self.proj_curv = nn.Linear(37, embed_dim)
        self.proj_invars = nn.Linear(32, embed_dim)
        
        # 7 feature tokens + 1 CLS token
        self.n_tokens = 8 
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Token type embeddings (tells attention which modality is which)
        self.token_type_embed = nn.Parameter(torch.randn(1, self.n_tokens, embed_dim) * 0.02)
        
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.dropout = nn.Dropout(0.1)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x):
        b = x.size(0)
        
        # Slice based on FEATURE_SCHEMA from generate_curve_data.py
        # Fallbacks added in case feature dimension doesn't perfectly match
        raw = x[:, 0:128]
        fft = x[:, 128:160]
        fft_phase = x[:, 160:192]
        deriv = x[:, 192:320]
        stats = x[:, 320:329]
        curv = x[:, 329:366]
        
        # Handle cases where feature dimension might be slightly different
        if x.shape[1] > 366:
            invars = x[:, 366:398]
        else:
            invars = torch.zeros(b, 32, device=x.device, dtype=x.dtype)
            
        # Create tokens
        t_raw = self.proj_raw(raw).unsqueeze(1)
        t_fft = self.proj_fft(fft).unsqueeze(1)
        t_fft_phase = self.proj_fft_phase(fft_phase).unsqueeze(1)
        t_deriv = self.proj_deriv(deriv).unsqueeze(1)
        t_stats = self.proj_stats(stats).unsqueeze(1)
        t_curv = self.proj_curv(curv).unsqueeze(1)
        t_invars = self.proj_invars(invars).unsqueeze(1)
        
        cls_tokens = self.cls_token.expand(b, -1, -1)
        
        # Sequence of 8 tokens
        tokens = torch.cat([
            cls_tokens, t_raw, t_fft, t_fft_phase, t_deriv, t_stats, t_curv, t_invars
        ], dim=1)
        
        tokens = tokens + self.token_type_embed
        tokens = self.dropout(tokens)
        
        # Attention
        attn_out, _ = self.attention(tokens, tokens, tokens)
        tokens = self.norm1(tokens + attn_out)
        
        # FFN
        ffn_out = self.ffn(tokens)
        tokens = self.norm2(tokens + ffn_out)
        
        # Flatten all tokens to maintain the 1024-dim output expected
        return tokens.flatten(1)


class EQLLayer(nn.Module):
    """
    Equation Learner (EQL) Layer.
    Applies explicit mathematical transformations to the input to act as a 
    'cheat sheet' for the network to detect mathematical operators.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        
        # 6 explicit mathematical functions
        self.n_funcs = 6
        self.features_per_func = out_features // self.n_funcs
        self.rem_features = out_features % self.n_funcs
        
        # Project input features to the space where functions will be applied
        self.linear = nn.Linear(in_features, out_features)
        
        # Initialize weights to be small to prevent extreme values going into exp/log early on
        nn.init.xavier_normal_(self.linear.weight, gain=0.1)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        z = self.linear(x)
        
        out = []
        start_idx = 0
        
        for i in range(self.n_funcs):
            end_idx = start_idx + self.features_per_func + (self.rem_features if i == 0 else 0)
            chunk = z[:, start_idx:end_idx]
            
            if i == 0:
                # Identity
                out.append(chunk)
            elif i == 1:
                # Sin
                out.append(torch.sin(chunk))
            elif i == 2:
                # Cos
                out.append(torch.cos(chunk))
            elif i == 3:
                # Exp (clamped to prevent inf/nan and massive BN shifts)
                out.append(torch.exp(torch.clamp(chunk, min=-5.0, max=5.0)))
            elif i == 4:
                # Log (safe log)
                out.append(torch.log(torch.abs(chunk) + 1e-6))
            elif i == 5:
                # Square
                out.append(torch.square(chunk))
                
            start_idx = end_idx
            
        return torch.cat(out, dim=1)


class CurveClassifierGLU(nn.Module):
    """
    First-Principles Mathematical Classifier using Gated Linear Units (GLU).
    Mathematically models multiplicative function composition (e.g. x * sin(x)) natively.
    """
    def __init__(self, n_features: int = 398, n_classes: int = 9, hidden: int = 512):
        super().__init__()
        
        # 1. Semantic Feature Attention
        n_tokens = 8
        embed_dim = 128
        self.attn = SemanticFeatureAttention(embed_dim=embed_dim)
        attn_out_dim = n_tokens * embed_dim
        
        # 2. EQL Layer
        eql_out_dim = 256
        self.eql = EQLLayer(in_features=n_features, out_features=eql_out_dim)
        
        # 3. Combine outputs
        combined_dim = attn_out_dim + eql_out_dim
        
        self.fc1 = nn.Linear(combined_dim, hidden * 2)
        self.bn1 = nn.BatchNorm1d(hidden * 2)
        
        self.fc2 = nn.Linear(hidden, hidden * 2)
        self.bn2 = nn.BatchNorm1d(hidden * 2)
        
        self.fc3 = nn.Linear(hidden, hidden * 2)
        self.bn3 = nn.BatchNorm1d(hidden * 2)
        
        self.fc4 = nn.Linear(hidden, hidden * 2)
        self.bn4 = nn.BatchNorm1d(hidden * 2)
        
        self.classifier = nn.Linear(hidden, n_classes)
        self.dropout = nn.Dropout(0.2)

        self._init_weights()

    def _init_weights(self):
        """Hardware-sympathetic initialization for multiplicative gating."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)
                    
    def forward(self, x):
        # Extract abstract semantic tokens
        attn_features = self.attn(x)
        
        # Extract explicit mathematical transformations
        eql_features = self.eql(x)
        
        # Combine
        x = torch.cat([attn_features, eql_features], dim=1)
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.glu(x, dim=1)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.glu(x, dim=1)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.glu(x, dim=1)
        x = self.dropout(x)
        
        x = self.fc4(x)
        x = self.bn4(x)
        x = F.glu(x, dim=1)
        x = self.dropout(x)
        
        return self.classifier(x)


try:
    from .models import (
        CURVE_CLASSIFIER_ARCHITECTURE_VERSION,
        CurveClassifierCNN as CurveClassifierCNN,
        CurveClassifierGLU as CurveClassifierGLU,
        CurveClassifierMLP as CurveClassifierMLP,
        EQLLayer as EQLLayer,
        SemanticFeatureAttention as SemanticFeatureAttention,
    )
except (ImportError, ValueError):
    from glassbox.curve_classifier.models import (
        CURVE_CLASSIFIER_ARCHITECTURE_VERSION,
        CurveClassifierCNN as CurveClassifierCNN,
        CurveClassifierGLU as CurveClassifierGLU,
        CurveClassifierMLP as CurveClassifierMLP,
        EQLLayer as EQLLayer,
        SemanticFeatureAttention as SemanticFeatureAttention,
    )


# =============================================================================
# CLASSIFIER LOADING
# =============================================================================

_cached_classifier_by_device = {}
_cached_operator_classes_by_key = {}
_cached_metadata_by_device = {}
_cached_interpolators_by_signature: Dict[Tuple, Tuple[Optional[object], object]] = {}
# S9-3: short-lived curve feature cache (same fit / dual-path re-entry).
_cached_curve_features: Dict[Tuple, np.ndarray] = {}
_warned_no_cuda = False


def _curve_feature_cache_key(x: np.ndarray, y: np.ndarray, n_points: int = 256) -> Tuple:
    """Coarse deterministic key for feature-vector reuse within a process."""
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    n = int(min(x_arr.size, y_arr.size))
    if n <= 0:
        return (0, 0.0, 0.0, 0.0, 0.0, int(n_points))
    # Sample up to 32 points for a cheap content fingerprint.
    step = max(1, n // 32)
    xs = x_arr[::step][:32]
    ys = y_arr[::step][:32]
    return (
        int(n),
        float(np.nanmean(xs)),
        float(np.nanstd(xs)),
        float(np.nanmean(ys)),
        float(np.nanstd(ys)),
        float(np.nansum(ys * xs)),
        int(n_points),
    )


def _extract_features_xy_cached(x: np.ndarray, y: np.ndarray, n_points: int = 256) -> np.ndarray:
    """extract_all_features_xy with a small process-local cache (S9-3)."""
    key = _curve_feature_cache_key(x, y, n_points=n_points)
    hit = _cached_curve_features.get(key)
    if hit is not None:
        return hit
    feats = extract_all_features_xy(x, y, n_points=n_points)
    _cached_curve_features[key] = feats
    # Bound memory: keep last ~32 feature vectors.
    if len(_cached_curve_features) > 32:
        _cached_curve_features.pop(next(iter(_cached_curve_features)))
    return feats


def _resolve_device(
    device: Optional[str] = None,
    *,
    n_samples: Optional[int] = None,
) -> torch.device:
    """Resolve torch device for classifier inference.

    S9-5: when ``device`` is None/\"auto\" and ``n_samples`` is small
    (default threshold 256), force CPU so CUDA transfer overhead does not
    dominate tiny fits. Explicit ``device='cuda'`` still requests GPU.
    """
    global _warned_no_cuda
    tiny_n_cpu_threshold = 256
    if device is None or device == "auto":
        if n_samples is not None and int(n_samples) < tiny_n_cpu_threshold:
            return torch.device("cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        if not _warned_no_cuda:
            print("CUDA requested but not available; falling back to CPU.")
            _warned_no_cuda = True
        return torch.device("cpu")

    return resolved


def _make_cache_key(model_path: str, resolved_device: torch.device) -> str:
    return f"{str(resolved_device)}:{str(Path(model_path).resolve())}"


def _is_trusted_checkpoint_path(model_path: Path) -> bool:
    resolved = model_path.resolve()
    trusted_roots = [
        (_ROOT / "models").resolve(),
        (_ROOT / "artifacts").resolve(),
    ]
    return any(resolved == root or root in resolved.parents for root in trusted_roots)


def _load_torch_checkpoint(model_path: Path):
    try:
        return torch.load(model_path, map_location='cpu', weights_only=True)
    except Exception as safe_error:
        if not _is_trusted_checkpoint_path(model_path):
            raise RuntimeError(
                "Refusing unsafe pickle checkpoint load outside trusted local model directories. "
                f"Move {model_path} under models/ or artifacts/, or convert it to a weights-only checkpoint."
            ) from safe_error
        if not os.environ.get("GLASSBOX_ALLOW_PICKLE_CHECKPOINT"):
            raise RuntimeError(
                "weights-only checkpoint load failed for a trusted local checkpoint; refusing "
                "unsafe pickle fallback without explicit opt-in. Set "
                "GLASSBOX_ALLOW_PICKLE_CHECKPOINT=1 to allow pickle loading for trusted local "
                f"checkpoints, or convert {model_path} to a weights-only checkpoint."
            ) from safe_error
        if os.environ.get("GLASSBOX_VERBOSE_CHECKPOINT_LOAD"):
            print(
                "weights-only checkpoint load failed; falling back to trusted local "
                f"pickle checkpoint at {model_path}."
            )
        return torch.load(model_path, map_location='cpu', weights_only=False)


def validate_curve_classifier_checkpoint_metadata(
    checkpoint: dict,
    *,
    strict: bool = False,
) -> Dict[str, object]:
    """Validate classifier checkpoint metadata with legacy compatibility.

    Current checkpoints should include model type, feature dimension, feature
    schema, and architecture version. Older local checkpoints may be missing
    some metadata; non-strict validation returns warnings instead of failing so
    existing artifacts remain loadable.
    """
    warnings: List[str] = []
    errors: List[str] = []

    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dictionary")
    if "model_state_dict" not in checkpoint:
        errors.append("missing model_state_dict")

    model_config = checkpoint.get("model_config") or {}
    model_type = checkpoint.get("model_type")
    if model_type is None:
        warnings.append("missing model_type; falling back to state_dict architecture detection")
    elif model_type not in {"glu", "mlp", "cnn"}:
        errors.append(f"unsupported model_type={model_type!r}")

    feature_dim = checkpoint.get("feature_dim", model_config.get("n_features"))
    try:
        feature_dim = int(feature_dim)
        if feature_dim <= 0:
            errors.append("feature_dim must be positive")
    except Exception:
        feature_dim = None
        warnings.append("missing or invalid feature_dim")

    feature_schema = checkpoint.get("feature_schema")
    if feature_schema is None:
        warnings.append("missing feature_schema")

    architecture_version = (
        checkpoint.get("architecture_version")
        or model_config.get("architecture_version")
    )
    if architecture_version is None:
        architecture_version = "legacy_unversioned"
        warnings.append("missing architecture_version; treating checkpoint as legacy")

    if errors or (strict and warnings):
        details = "; ".join(errors + warnings)
        raise ValueError(f"Invalid curve classifier checkpoint metadata: {details}")

    return {
        "model_type": model_type,
        "feature_dim": feature_dim,
        "feature_schema": feature_schema,
        "architecture_version": architecture_version,
        "warnings": warnings,
    }


def _resolve_model_path(model_path: str) -> Path:
    resolved = resolve_curve_classifier_path(model_path)
    if str(resolved) != str(Path(model_path)):
        print(
            f"Curve classifier model not found at {model_path}; "
            f"using {resolved} instead."
        )
    return resolved


def load_classifier(
    model_path: str = DEFAULT_CURVE_CLASSIFIER_PATH,
    device: Optional[str] = None,
    *,
    n_samples: Optional[int] = None,
):
    """Load the trained PyTorch curve classifier.

    Parameters
    ----------
    n_samples :
        Optional row count so device=auto can prefer CPU on tiny problems (S9-5).
    """
    global _cached_classifier_by_device
    
    resolved_device = _resolve_device(device, n_samples=n_samples)
    model_path_obj = _resolve_model_path(model_path)
    if model_path_obj.suffix.lower() not in ('.pt', '.pth'):
        raise ValueError(
            f"Unsupported classifier artifact {model_path_obj}. "
            "Legacy non-PyTorch classifier payloads have been removed; use a PyTorch .pt checkpoint."
        )

    # Create cache key using both device and absolute model path
    cache_key = _make_cache_key(str(model_path_obj), resolved_device)
    if cache_key in _cached_classifier_by_device:
        return _cached_classifier_by_device[cache_key]

    return _load_pytorch_classifier(model_path_obj, resolved_device, cache_key)


def _resolve_n_features_from_state_dict(
    model_type: str,
    state_dict: dict,
    model_config: dict,
    checkpoint: dict,
) -> int:
    """Resolve input feature dimension for classifier reconstruction (H-16).

    Preference order:
    1. ``model_config['n_features']`` (current checkpoints)
    2. top-level ``feature_dim`` (legacy metadata)
    3. architecture-specific weight shapes that encode true input width

    Avoids the old CNN heuristic ``classifier_in - 512`` (yields curve_dim, not
    full feature dim) and the GLU/MLP mistake of reading *combined* layer widths
    (``fc1`` / ``net.0``) instead of the EQL input projection.
    """
    if 'n_features' in model_config and model_config['n_features'] is not None:
        return int(model_config['n_features'])

    feature_dim = checkpoint.get('feature_dim')
    if feature_dim is not None:
        return int(feature_dim)

    if model_type == 'cnn':
        # other_mlp.0: Linear(other_dim, 128) where other_dim = n_features - curve_dim
        if 'other_mlp.0.weight' in state_dict:
            other_dim = int(state_dict['other_mlp.0.weight'].shape[1])
            curve_dim = int(model_config.get('curve_dim', 128))
            return int(max(1, curve_dim + other_dim))
        # Last resort: classifier is Linear(128*4 + 128, …) — cannot recover
        # full n_features; raise rather than silently build a wrong model.
        raise ValueError(
            "Cannot derive CNN n_features from checkpoint: missing model_config"
            "['n_features'], feature_dim, and other_mlp.0.weight. "
            "Re-export the checkpoint with n_features in model_config."
        )

    if model_type == 'glu':
        # eql.linear: Linear(n_features, eql_out_dim) — true input width
        if 'eql.linear.weight' in state_dict:
            return int(state_dict['eql.linear.weight'].shape[1])
        raise ValueError(
            "Cannot derive GLU n_features from checkpoint: missing model_config"
            "['n_features'], feature_dim, and eql.linear.weight. "
            "Re-export the checkpoint with n_features in model_config."
        )

    # MLP (default)
    if 'eql.linear.weight' in state_dict:
        return int(state_dict['eql.linear.weight'].shape[1])
    if 'net.0.weight' in state_dict:
        # net.0 is Linear(n_features + eql_out, hidden); eql_out is 256 in arch
        combined = int(state_dict['net.0.weight'].shape[1])
        eql_out_dim = 256
        n_features = combined - eql_out_dim
        if n_features < 1:
            raise ValueError(
                f"Derived MLP n_features={n_features} from net.0 combined "
                f"width {combined}; checkpoint is corrupt or architecture mismatch."
            )
        return int(n_features)
    raise ValueError(
        "Cannot derive MLP n_features from checkpoint: missing model_config"
        "['n_features'], feature_dim, eql.linear.weight, and net.0.weight. "
        "Re-export the checkpoint with n_features in model_config."
    )


def _load_pytorch_classifier(model_path: Path, resolved_device: torch.device, cache_key: str) -> nn.Module:
    """Load PyTorch classifier from .pt file."""
    global _cached_classifier_by_device, _cached_operator_classes_by_key, _cached_metadata_by_device
    
    try:
        checkpoint = _load_torch_checkpoint(model_path)
    except Exception as e:
        print(f"Error loading checkpoint from {model_path}: {e}")
        raise

    metadata_report = validate_curve_classifier_checkpoint_metadata(checkpoint)
    if os.environ.get("GLASSBOX_VERBOSE_CHECKPOINT_LOAD"):
        for warning in metadata_report.get("warnings", []):
            print(f"  Checkpoint metadata warning: {warning}")
    
    # Get operator classes
    operator_classes = checkpoint.get('operator_classes', list(OPERATOR_CLASSES.keys()))
    _cached_operator_classes_by_key[cache_key] = operator_classes
    n_classes = len(operator_classes)
    
    state_dict = checkpoint['model_state_dict']
    model_type = checkpoint.get('model_type')
    model_config = checkpoint.get('model_config') or {}

    # Backward-compatible architecture detection for older checkpoints
    if model_type is None:
        if any(k.startswith('conv.') for k in state_dict.keys()):
            model_type = 'cnn'
        elif any(k.startswith('fc1.') for k in state_dict.keys()) and any(
            k.startswith('eql.') for k in state_dict.keys()
        ):
            model_type = 'glu'
        elif any(k.startswith('net.') for k in state_dict.keys()):
            model_type = 'mlp'
        else:
            raise ValueError(
                "Unable to infer classifier architecture from checkpoint; "
                "expected MLP keys ('net.*'), CNN keys ('conv.*'), or GLU keys ('fc1.*'+'eql.*')."
            )

    n_features = _resolve_n_features_from_state_dict(
        model_type, state_dict, model_config, checkpoint
    )

    if model_type == 'cnn':
        curve_dim = int(model_config.get('curve_dim', min(128, n_features)))
        model = CurveClassifierCNN(
            n_classes=int(model_config.get('n_classes', n_classes)),
            n_features=n_features,
            curve_dim=curve_dim,
        )
    elif model_type == 'glu':
        if 'hidden' in model_config and model_config['hidden'] is not None:
            hidden_size = int(model_config['hidden'])
        elif 'fc1.weight' in state_dict:
            # fc1 is Linear(combined, hidden*2) for GLU
            hidden_size = int(state_dict['fc1.weight'].shape[0] // 2)
        else:
            hidden_size = 512
        model = CurveClassifierGLU(n_features=n_features, n_classes=n_classes, hidden=hidden_size)
    else:
        if 'hidden' in model_config and model_config['hidden'] is not None:
            hidden_size = int(model_config['hidden'])
        elif 'net.0.weight' in state_dict:
            hidden_size = int(state_dict['net.0.weight'].shape[0])
        else:
            hidden_size = 512
        model = CurveClassifierMLP(n_features=n_features, n_classes=n_classes, hidden=hidden_size)

    model.load_state_dict(state_dict)
    model.to(resolved_device)
    model.eval()

    _cached_classifier_by_device[cache_key] = model
    _cached_metadata_by_device[cache_key] = {
        'thresholds': checkpoint.get('thresholds'),
        'temperature': checkpoint.get('temperature'),
        'feature_scaler': checkpoint.get('feature_scaler'),
        'type': 'pytorch',
        'model_type': model_type,
        'n_features': n_features,
        'operator_classes': operator_classes,
        'isotonic_calibration': checkpoint.get('isotonic_calibration'),
        'architecture_version': metadata_report.get('architecture_version'),
    }
    print(f"Loaded PyTorch curve classifier from {model_path}")
    if 'val_acc' in checkpoint:
        print(f"  Val accuracy: {checkpoint.get('val_acc'):.4f}")
    print(f"  Device: {resolved_device}")
    
    return model


# =============================================================================
# PREDICTION
# =============================================================================

def _predict_pytorch(model: nn.Module, features: np.ndarray, metadata: dict, device: torch.device) -> np.ndarray:
    """Predict using PyTorch model, with optional isotonic calibration."""
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(features_tensor).squeeze()
        temperature = metadata.get('temperature')
        if temperature is not None:
            logits = logits / float(temperature)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
    
    # Apply per-class isotonic calibration if available
    isotonic_maps = metadata.get('isotonic_calibration')
    if isotonic_maps:
        probs = _apply_isotonic_calibration(probs, isotonic_maps)
    
    return probs


def _apply_isotonic_calibration(
    raw_probs: np.ndarray,
    calibration_maps: list,
) -> np.ndarray:
    """Apply per-class isotonic regression calibration to raw probabilities.
    
    Each calibration map is a dict with 'boundaries' (bin edges) and 'values'
    (calibrated probability for each bin). Uses np.digitize for fast lookup.
    """
    if not calibration_maps:
        return raw_probs
    
    single = raw_probs.ndim == 1
    if single:
        raw_probs = raw_probs.reshape(1, -1)
    
    calibrated = raw_probs.copy()
    n_classes = raw_probs.shape[1]
    
    for c in range(min(n_classes, len(calibration_maps))):
        cmap = calibration_maps[c]
        boundaries = np.array(cmap['boundaries'])
        values = np.array(cmap['values'])
        indices = np.digitize(raw_probs[:, c], boundaries, right=False) - 1
        indices = np.clip(indices, 0, len(values) - 1)
        calibrated[:, c] = values[indices]
    
    if single:
        return calibrated[0]
    return calibrated


def _evaluate_demo_formula(formula: str, x: np.ndarray) -> np.ndarray:
    """Evaluate CLI demo formulas through the shared restricted evaluator."""
    from glassbox.universal_proposer.universal_proposer import _safe_formula_eval

    y = _safe_formula_eval(formula, x)
    if y is None:
        raise ValueError("formula could not be evaluated safely over the requested x range")
    return np.asarray(y, dtype=np.float64)


def _prepare_curve_features(features: np.ndarray, scaler: Optional[dict] = None) -> np.ndarray:
    """Apply the classifier feature transform used by training and inference."""
    prepared = np.asarray(features, dtype=np.float32).copy()
    end = min(prepared.shape[0], 398)
    if end > 192:
        prepared[192:end] = np.sign(prepared[192:end]) * np.log1p(np.abs(prepared[192:end]))

    if scaler is not None:
        mean = np.asarray(scaler['mean'], dtype=np.float32)
        std = np.asarray(scaler['std'], dtype=np.float32)
        prepared = prepared[:len(mean)]
        prepared = (prepared - mean) / (std + 1e-8)

    return prepared


def predict_operators(
    x: np.ndarray,
    y: np.ndarray,
    model_path: str = DEFAULT_CURVE_CLASSIFIER_PATH,
    threshold: float = 0.5,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """
    Predict which operators are likely present in the data.
    
    For multi-input data (n_vars > 1), uses heuristic per-variable 1D slicing:
    - Takes 1D cross-sections through the data (fixing other vars at midpoint)
    - Runs classifier on each slice
    - Aggregates predictions across all variables
    This is not a trained multivariate neural model; call
    describe_curve_classifier_inference(x) for the explicit inference contract.
    
    Args:
        x: Input values - 1D array (N,) or 2D array (N, n_vars)
        y: Output values (1D array)
        model_path: Path to trained classifier
        threshold: Probability threshold for reporting
        
    Returns:
        Dictionary mapping operator names to probabilities
    """
    global LAST_CLASSIFIER_LOAD_STATUS

    # Detect multi-input early (needed for S9-5 tiny-n device choice).
    x = np.asarray(x)
    y = np.asarray(y).flatten()
    if x.ndim == 1:
        n_vars = 1
        x = x.reshape(-1, 1)
    else:
        n_vars = x.shape[1]
    n_samples = int(x.shape[0])

    # S9-1: fail-open with explicit diagnostics (module-level, not in op dict).
    # Returning {} keeps legacy callers (if not predictions: ...) correct.
    load_status: Dict[str, object] = {
        "ok": False,
        "reason": None,
        "model_path": str(model_path),
        "fail_open": None,
    }

    # Check if model exists before trying to load
    if not Path(model_path).exists():
        try:
            model_path = str(_resolve_model_path(model_path))
            load_status["model_path"] = model_path
        except FileNotFoundError:
            print(
                f"Warning: Curve classifier model not found at {model_path}. "
                "Skipping prediction (fail-open → evolution)."
            )
            load_status.update({
                "ok": False,
                "reason": "model_not_found",
                "model_path": str(model_path),
                "fail_open": True,
            })
            LAST_CLASSIFIER_LOAD_STATUS = dict(load_status)
            return {}

    # Load classifier (S9-5: prefer CPU for tiny n under device=auto)
    resolved_device = _resolve_device(device, n_samples=n_samples)
    try:
        model = load_classifier(model_path, device=str(resolved_device), n_samples=n_samples)
        load_status["ok"] = True
    except Exception as e:
        print(
            f"Warning: Failed to load curve classifier: {e}. "
            "Skipping prediction (fail-open → evolution)."
        )
        load_status.update({
            "ok": False,
            "reason": "load_failed",
            "error": str(e),
            "model_path": str(model_path),
            "fail_open": True,
        })
        LAST_CLASSIFIER_LOAD_STATUS = dict(load_status)
        return {}

    # Get cache key for metadata lookup (same device as load)
    cache_key = _make_cache_key(str(_resolve_model_path(model_path)), resolved_device)
    metadata = _cached_metadata_by_device.get(cache_key, {})

    load_status.update({
        "ok": True,
        "reason": "loaded",
        "fail_open": False,
        "device": str(resolved_device),
        "model_path": str(model_path),
    })
    LAST_CLASSIFIER_LOAD_STATUS = dict(load_status)

    # For multi-input: use per-variable slicing
    if n_vars > 1:
        return _predict_operators_multi_input(
            x, y, model, metadata, resolved_device, threshold, n_vars, cache_key
        )

    # Single-input: standard prediction (S9-3: cached feature extract).
    features = _prepare_curve_features(
        _extract_features_xy_cached(x[:, 0], y),
        metadata.get('feature_scaler'),
    )
    probs = _predict_pytorch(model, features, metadata, resolved_device)

    return _build_result_dict(probs, threshold, metadata, cache_key)


def detect_variable_interactions(
    x: np.ndarray, 
    y: np.ndarray,
    interp,
    n_grid: int = 15,
    interaction_threshold: float = 0.1,
) -> List[Tuple[int, int, float]]:
    """Detect pairwise variable interactions using partial dependence.
    
    Returns list of (var_i, var_j, H_statistic) for interacting pairs.
    """
    n_vars = x.shape[1]
    interactions = []
    
    var_y = np.var(y)
    if var_y < 1e-12:
        return interactions
        
    x_template = np.median(x, axis=0)
    y_mean = float(np.mean(y))
    
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            xi_grid = np.linspace(x[:, i].min(), x[:, i].max(), n_grid)
            xj_grid = np.linspace(x[:, j].min(), x[:, j].max(), n_grid)
            
            # Vectorized PD computation
            XI, XJ = np.meshgrid(xi_grid, xj_grid, indexing='ij')
            x_query_ij = np.tile(x_template, (n_grid * n_grid, 1))
            x_query_ij[:, i] = XI.flatten()
            x_query_ij[:, j] = XJ.flatten()
            
            pd_ij_flat = interp(x_query_ij)
            pd_ij_flat = np.nan_to_num(pd_ij_flat, nan=y_mean)
            pd_ij = pd_ij_flat.reshape(n_grid, n_grid)
            
            x_query_i = np.tile(x_template, (n_grid, 1))
            x_query_i[:, i] = xi_grid
            pd_i = interp(x_query_i)
            pd_i = np.nan_to_num(pd_i, nan=y_mean)
            
            x_query_j = np.tile(x_template, (n_grid, 1))
            x_query_j[:, j] = xj_grid
            pd_j = interp(x_query_j)
            pd_j = np.nan_to_num(pd_j, nan=y_mean)
            
            interaction_surface = pd_ij - pd_i[:, None] - pd_j[None, :] + y_mean
            var_pd_ij = max(np.var(pd_ij), 1e-12)
            H = np.var(interaction_surface) / var_pd_ij
            
            if H > interaction_threshold:
                interactions.append((i, j, float(H)))
                
    return sorted(interactions, key=lambda t: -t[2])


def _predict_operators_multi_input(
    x: np.ndarray,
    y: np.ndarray,
    model,
    metadata: dict,
    device: torch.device,
    threshold: float,
    n_vars: int,
    cache_key: str,
) -> Dict[str, float]:
    """
    Predict operators for multi-input data using per-variable 1D slicing.
    
    For each variable:
    1. Fix all other variables at their median value
    2. Vary the target variable across its range
    3. Compute y values for this slice
    4. Run classifier on the 1D slice
    5. Aggregate predictions across all variables (max probability)
    """
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    
    # Build/reuse interpolators for y values (hot path for repeated calls).
    # H-14: never key only on buffer pointers — in-place y/x mutation would
    # reuse a stale interpolator. Content fingerprint (or sample+ptr) invalidates.
    def _interp_array_fp(arr: np.ndarray) -> Tuple:
        a = np.asarray(arr)
        shape = tuple(a.shape)
        dtype_s = str(a.dtype)
        n_elem = int(np.prod(shape)) if shape else 0
        if n_elem > 0 and n_elem <= 8192 and a.dtype.kind in "fcibu":
            return (shape, dtype_s, hash(np.ascontiguousarray(a).tobytes()))
        flat = np.ascontiguousarray(a).reshape(-1)
        step = max(1, int(flat.size) // 64) if flat.size else 1
        sample = flat[::step][:64]
        try:
            ptr = int(a.__array_interface__["data"][0])
        except Exception:
            ptr = id(a)
        sample_fp = hash(np.ascontiguousarray(sample).tobytes()) if sample.size else 0
        return (shape, dtype_s, ptr, sample_fp, float(np.nanmean(flat)) if flat.size else 0.0)

    interp_signature = (_interp_array_fp(x), _interp_array_fp(y))
    interpolators = _cached_interpolators_by_signature.get(interp_signature)
    if interpolators is None:
        linear_interp = None
        nearest_interp = None
        try:
            linear_interp = LinearNDInterpolator(x, y, fill_value=np.nan)
        except Exception:
            linear_interp = None
        try:
            nearest_interp = NearestNDInterpolator(x, y)
        except Exception:
            nearest_interp = None
        if linear_interp is None and nearest_interp is None:
            raise RuntimeError("Failed to build interpolation models for multi-input prediction.")
        interpolators = (linear_interp, nearest_interp)
        _cached_interpolators_by_signature[interp_signature] = interpolators
        if len(_cached_interpolators_by_signature) > 8:
            _cached_interpolators_by_signature.pop(next(iter(_cached_interpolators_by_signature)))

    linear_interp, nearest_interp = interpolators

    operator_classes = (
        metadata.get('operator_classes')
        or _cached_operator_classes_by_key.get(cache_key)
        or list(OPERATOR_CLASSES.keys())
    )
    all_probs = np.zeros((n_vars, len(operator_classes)), dtype=np.float32)
    
    scaler = metadata.get('feature_scaler')
    x_medians = np.median(x, axis=0)
    
    interactions = []
    # S9-3: skip expensive pairwise interaction PD when p is large or n is huge.
    if nearest_interp is not None and n_vars <= 6 and len(y) <= 4000:
        try:
            interactions = detect_variable_interactions(
                x, y, nearest_interp, n_grid=11 if n_vars >= 4 else 15
            )
        except Exception as e:
            print(f"  Warning: Interaction detection failed: {e}")

    # S9-3: cap per-variable slices and grid density on large problems.
    max_vars = min(n_vars, 8)
    n_slice_points = min(128 if len(y) >= 500 else 256, len(y))

    for var_idx in range(max_vars):
        # Create 1D slice: fix other variables at median, vary this one
        x_min_var = x[:, var_idx].min()
        x_max_var = x[:, var_idx].max()
        
        x_slice_1d = np.linspace(x_min_var, x_max_var, n_slice_points)
        
        # Build full query points (other vars at median)
        x_query = np.tile(x_medians, (n_slice_points, 1))
        x_query[:, var_idx] = x_slice_1d
        
        # Get y values for this slice
        if linear_interp is not None:
            y_slice = linear_interp(x_query)
            if nearest_interp is not None:
                nan_mask = ~np.isfinite(y_slice)
                if np.any(nan_mask):
                    y_slice[nan_mask] = nearest_interp(x_query[nan_mask])
        else:
            y_slice = nearest_interp(x_query)
        
        # Handle NaN values from interpolation
        valid_mask = np.isfinite(y_slice)
        if valid_mask.sum() < 10:
            # Not enough valid points, skip this variable
            continue
        
        y_slice_valid = y_slice[valid_mask]
        
        # Extract features and predict (S9-3: cached extract)
        try:
            features = _prepare_curve_features(
                _extract_features_xy_cached(x_slice_1d[valid_mask], y_slice_valid),
                scaler,
            )
            probs = _predict_pytorch(model, features, metadata, device)
            
            all_probs[var_idx] = probs
        except Exception as e:
            # Skip this variable if feature extraction fails
            print(f"  Warning: Slice {var_idx} failed: {e}")
            continue

    # Handle interactions by processing diagonal slices and boosting multiplication
    if interactions:
        mult_idx = -1
        if 'multiplication' in operator_classes:
            mult_idx = operator_classes.index('multiplication')
            
        for i, j, H in interactions:
            # 1. Boost multiplication probability directly based on interaction strength
            if mult_idx >= 0:
                boost = min(0.99, H * 2.0)
                interaction_probs = np.zeros(len(operator_classes), dtype=np.float32)
                interaction_probs[mult_idx] = boost
                all_probs = np.vstack([all_probs, interaction_probs])
                
            # 2. Diagonal slice to catch features that only appear when both vary
            x_min_i, x_max_i = x[:, i].min(), x[:, i].max()
            x_min_j, x_max_j = x[:, j].min(), x[:, j].max()
            
            n_slice_points = min(128 if len(y) >= 500 else 256, len(y))
            xi_slice = np.linspace(x_min_i, x_max_i, n_slice_points)
            xj_slice = np.linspace(x_min_j, x_max_j, n_slice_points)
            
            x_query = np.tile(x_medians, (n_slice_points, 1))
            x_query[:, i] = xi_slice
            x_query[:, j] = xj_slice
            
            if linear_interp is not None:
                y_slice = linear_interp(x_query)
                if nearest_interp is not None:
                    nan_mask = ~np.isfinite(y_slice)
                    if np.any(nan_mask):
                        y_slice[nan_mask] = nearest_interp(x_query[nan_mask])
            else:
                y_slice = nearest_interp(x_query)
            
            valid_mask = np.isfinite(y_slice)
            if valid_mask.sum() < 10:
                continue
                
            y_slice_valid = y_slice[valid_mask]
            
            try:
                features = _prepare_curve_features(
                    _extract_features_xy_cached(xi_slice[valid_mask], y_slice_valid),
                    scaler,
                )
                probs = _predict_pytorch(model, features, metadata, device)
                
                all_probs = np.vstack([all_probs, probs])
            except Exception as e:
                print(f"  Warning: Interaction diagonal slice {i}-{j} failed: {e}")

    # Aggregate: use max probability across all variables and diagonal slices
    aggregated_probs = np.max(all_probs, axis=0)
    
    return _build_result_dict(aggregated_probs, threshold, metadata, cache_key)


def _build_result_dict(probs: np.ndarray, threshold: float, metadata: dict, cache_key: str) -> Dict[str, float]:
    """Build result dictionary from probability array."""
    operator_classes = (
        metadata.get('operator_classes')
        or _cached_operator_classes_by_key.get(cache_key)
        or list(OPERATOR_CLASSES.keys())
    )
    thresholds = metadata.get('thresholds')
    if thresholds is None:
        thresholds = np.full((len(operator_classes),), threshold, dtype=np.float32)
    
    result = {}
    for i, name in enumerate(operator_classes):
        if i < len(probs) and probs[i] >= thresholds[i]:
            result[name] = float(probs[i])
    
    # Derived compatibility outputs
    name_to_idx = {name: i for i, name in enumerate(operator_classes)}
    periodic_prob = max(
        probs[name_to_idx.get('sin', 0)] if 'sin' in name_to_idx else 0.0,
        probs[name_to_idx.get('cos', 0)] if 'cos' in name_to_idx else 0.0,
    )
    exponential_prob = max(
        probs[name_to_idx.get('exp', 0)] if 'exp' in name_to_idx else 0.0,
        probs[name_to_idx.get('log', 0)] if 'log' in name_to_idx else 0.0,
    )
    polynomial_prob = max(
        probs[name_to_idx.get('power', 0)] if 'power' in name_to_idx else 0.0,
        probs[name_to_idx.get('identity', 0)] if 'identity' in name_to_idx else 0.0,
    )

    if periodic_prob >= threshold:
        result['periodic'] = float(periodic_prob)
    if exponential_prob >= threshold:
        result['exponential'] = float(exponential_prob)
    if polynomial_prob >= threshold:
        result['polynomial'] = float(polynomial_prob)

    return result


def print_predictions(predictions: Dict[str, float]):
    """Pretty print predictions."""
    print("\nPredicted operators:")
    sorted_preds = sorted(predictions.items(), key=lambda x: -x[1])
    for name, prob in sorted_preds:
        bar = "█" * int(prob * 20)
        print(f"  {name:15s}: {prob:.3f} {bar}")


# =============================================================================
# ONN INTEGRATION
# =============================================================================

def bias_onn_from_predictions(
    model,
    predictions: Dict[str, float],
    threshold: float = 0.3,
    boost_factor: float = 2.0,
    verbose: bool = True,
):
    """
    Use classifier predictions to bias ONN operation selection.
    
    Args:
        model: OperationDAG model
        predictions: Dict from predict_operators()
        threshold: Minimum probability to apply bias
        boost_factor: How much to boost logits for predicted operators
    """
    # Mapping from classifier classes to ONN meta-op indices
    # HardConcreteOperationSelector.logits layout:
    # [0:2] = type weights (unary vs binary)
    # [2:2+n_unary] = unary op weights
    # [2+n_unary:] = binary op weights
    
    # For simplified_ops=True: n_unary=2, n_binary=1
    # unary_ops: [MetaPeriodic (0), MetaPower (1)]
    
    # For simplified_ops=False: n_unary=4, n_binary=2
    # unary_ops: [MetaPeriodic (0), MetaPower (1), MetaExp (2), MetaLog (3)]
    
    simplified = getattr(model, 'simplified_ops', True)
    
    periodic_prob = max(predictions.get('sin', 0), predictions.get('cos', 0))

    if simplified:
        unary_map = {
            'sin': 0, 'cos': 0,  # MetaPeriodic
            'power': 1, 'identity': 1,  # MetaPower
            'rational': 1,  # Bias toward reciprocal via MetaPower
        }
        n_unary = 2
    else:
        unary_map = {
            'sin': 0, 'cos': 0,  # MetaPeriodic
            'power': 1, 'identity': 1,  # MetaPower
            'rational': 1,  # Bias toward reciprocal via MetaPower
            'exp': 2,  # MetaExp
            'log': 3,  # MetaLog
        }
        n_unary = 4
    
    n_biased = 0
    
    for layer in model.layers:
        for node in layer.nodes:
            # Get operation selector
            if not hasattr(node, 'op_selector'):
                continue
                
            selector = node.op_selector
            
            # HardConcreteOperationSelector has single logits tensor
            # Layout: [type(2), unary(n_unary), binary(n_binary)]
            if hasattr(selector, 'logits') and hasattr(selector, '_type_end'):
                with torch.no_grad():
                    # Bias type toward unary if sin/cos/periodic predicted
                    periodic_prob = max(predictions.get('sin', 0), 
                                       predictions.get('cos', 0))
                    if periodic_prob >= threshold:
                        selector.logits.data[0] += periodic_prob * boost_factor  # unary type
                        n_biased += 1
                    
                    # Bias specific unary ops
                    for op_name, prob in predictions.items():
                        if prob >= threshold and op_name in unary_map:
                            idx = unary_map[op_name]
                            logit_idx = 2 + idx  # Skip 2 type logits
                            if logit_idx < len(selector.logits):
                                selector.logits.data[logit_idx] += prob * boost_factor
                                n_biased += 1
    
    if n_biased > 0 and verbose:
        print(f"Biased {n_biased} operation logits based on classifier predictions")
    return model


# =============================================================================
# MAIN TEST
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test curve classifier")
    parser.add_argument("--model", type=str, default=DEFAULT_CURVE_CLASSIFIER_PATH,
                        help=f"Path to trained model (default: {DEFAULT_CURVE_CLASSIFIER_PATH})")
    parser.add_argument("--formula", type=str, default="np.sin(x) + x**2",
                        help="Formula to test (uses numpy)")
    parser.add_argument("--x-min", type=float, default=-5)
    parser.add_argument("--x-max", type=float, default=5)
    parser.add_argument("--n-points", type=int, default=256)
    
    args = parser.parse_args()
    
    # Generate test data
    x = np.linspace(args.x_min, args.x_max, args.n_points)
    
    try:
        y = _evaluate_demo_formula(args.formula, x)
    except Exception as e:
        print(f"Error evaluating formula: {e}")
        return
    
    print(f"Testing formula: {args.formula}")
    print(f"  x range: [{args.x_min}, {args.x_max}]")
    print(f"  y range: [{y.min():.4f}, {y.max():.4f}]")
    
    # Predict
    predictions = predict_operators(x, y, args.model, threshold=0.3)
    print_predictions(predictions)
    
    # Show expected vs predicted
    print("\nNote: periodic/exponential are derived from sin/cos and exp/log")


if __name__ == "__main__":
    main()
