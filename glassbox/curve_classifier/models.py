"""Shared curve-classifier model definitions.

Training and inference import these classes so checkpoint weights run through
the same forward pass in both paths.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


CURVE_CLASSIFIER_ARCHITECTURE_VERSION = "curve-classifier-shared-v1"


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
            nn.Dropout(0.2),
        ])

        for _ in range(6):
            layers.extend([
                nn.Linear(hidden, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])

        layers.extend([
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, n_classes),
        ])

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        eql_feats = self.eql(x)
        combined = torch.cat([x, eql_feats], dim=1)
        return self.net(combined)


class CurveClassifierCNN(nn.Module):
    """1D CNN classifier over raw-curve features plus summary features."""

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

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        raw_curve = x[:, :self.curve_dim]
        other_features = x[:, self.curve_dim:]

        raw_curve = raw_curve.unsqueeze(1)
        conv_out = self.conv(raw_curve).flatten(1)
        other_out = self.other_mlp(other_features)

        combined = torch.cat([conv_out, other_out], dim=1)
        return self.classifier(combined)


class SemanticFeatureAttention(nn.Module):
    """Treat each feature group as a token and attend across modalities."""

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim

        self.proj_raw = nn.Linear(128, embed_dim)
        self.proj_fft = nn.Linear(32, embed_dim)
        self.proj_fft_phase = nn.Linear(32, embed_dim)
        self.proj_deriv = nn.Linear(128, embed_dim)
        self.proj_stats = nn.Linear(9, embed_dim)
        self.proj_curv = nn.Linear(37, embed_dim)
        self.proj_invars = nn.Linear(32, embed_dim)

        self.n_tokens = 8
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.token_type_embed = nn.Parameter(torch.randn(1, self.n_tokens, embed_dim) * 0.02)

        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.dropout = nn.Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x):
        b = x.size(0)

        raw = x[:, 0:128]
        fft = x[:, 128:160]
        fft_phase = x[:, 160:192]
        deriv = x[:, 192:320]
        stats = x[:, 320:329]
        curv = x[:, 329:366]

        if x.shape[1] > 366:
            invars = x[:, 366:398]
        else:
            invars = torch.zeros(b, 32, device=x.device, dtype=x.dtype)

        tokens = torch.cat([
            self.cls_token.expand(b, -1, -1),
            self.proj_raw(raw).unsqueeze(1),
            self.proj_fft(fft).unsqueeze(1),
            self.proj_fft_phase(fft_phase).unsqueeze(1),
            self.proj_deriv(deriv).unsqueeze(1),
            self.proj_stats(stats).unsqueeze(1),
            self.proj_curv(curv).unsqueeze(1),
            self.proj_invars(invars).unsqueeze(1),
        ], dim=1)

        tokens = self.dropout(tokens + self.token_type_embed)
        attn_out, _ = self.attention(tokens, tokens, tokens)
        tokens = self.norm1(tokens + attn_out)
        ffn_out = self.ffn(tokens)
        tokens = self.norm2(tokens + ffn_out)
        return tokens.flatten(1)


class EQLLayer(nn.Module):
    """Equation Learner layer used by both training and inference."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.n_funcs = 6
        self.features_per_func = out_features // self.n_funcs
        self.rem_features = out_features % self.n_funcs
        self.linear = nn.Linear(in_features, out_features)

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
                out.append(chunk)
            elif i == 1:
                out.append(torch.sin(chunk))
            elif i == 2:
                out.append(torch.cos(chunk))
            elif i == 3:
                out.append(torch.exp(torch.clamp(chunk, min=-10.0, max=10.0)))
            elif i == 4:
                out.append(torch.log(torch.abs(chunk) + 1e-6))
            elif i == 5:
                out.append(torch.square(chunk))

            start_idx = end_idx

        return torch.cat(out, dim=1)


class CurveClassifierGLU(nn.Module):
    """GLU classifier over semantic attention and EQL features."""

    def __init__(self, n_features: int = 398, n_classes: int = 9, hidden: int = 512):
        super().__init__()

        n_tokens = 8
        embed_dim = 128
        self.attn = SemanticFeatureAttention(embed_dim=embed_dim)
        attn_out_dim = n_tokens * embed_dim

        eql_out_dim = 256
        self.eql = EQLLayer(in_features=n_features, out_features=eql_out_dim)

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
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        attn_features = self.attn(x)
        eql_features = self.eql(x)

        x = torch.cat([attn_features, eql_features], dim=1)

        x = self.dropout(F.glu(self.bn1(self.fc1(x)), dim=1))
        x = self.dropout(F.glu(self.bn2(self.fc2(x)), dim=1))
        x = self.dropout(F.glu(self.bn3(self.fc3(x)), dim=1))
        x = self.dropout(F.glu(self.bn4(self.fc4(x)), dim=1))

        return self.classifier(x)


__all__ = [
    "CURVE_CLASSIFIER_ARCHITECTURE_VERSION",
    "CurveClassifierMLP",
    "CurveClassifierCNN",
    "SemanticFeatureAttention",
    "EQLLayer",
    "CurveClassifierGLU",
]
