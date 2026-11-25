"""Train an exact-vs-approximation verifier on the pair dataset.

Input at inference is exactly what the pipeline has: the sampled curve and a
candidate formula. The verifier never sees the target formula, so it must
learn the structural cues (operator agreement, series coefficients, residual
shape) rather than memorise answers.

Usage:
    python experiments/exactness/train_verifier.py \
        --data data/exactness_pairs --out data/exactness_pairs/verifier.pt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exactness import lib  # noqa: E402

CURVE_FEATURES = 398


def _residual_features(
    candidate: str, x: np.ndarray, y: np.ndarray
) -> tuple[list[float], float | None]:
    pred = lib.evaluate(candidate, x)
    if pred is None or pred.shape != y.shape:
        return [0.0] * 6, None
    mask = np.isfinite(pred) & np.isfinite(y)
    if mask.sum() < 10:
        return [0.0] * 6, None
    residual = y[mask] - pred[mask]
    var = float(np.var(y[mask]))
    mse = float(np.mean(residual**2))
    centered = residual - residual.mean()
    std = float(np.std(centered))
    skew = float(np.mean((centered / std) ** 3)) if std > 1e-12 else 0.0
    kurt = float(np.mean((centered / std) ** 4) - 3.0) if std > 1e-12 else 0.0
    peak_ratio = 0.0
    if centered.size >= 8:
        mags = np.abs(np.fft.rfft(centered)[1:])
        if mags.size:
            peak_ratio = float(np.max(mags)) / max(float(np.median(mags)), 1e-12)
    holdout_ratio = 0.0
    n = residual.size
    if n >= 20:
        h = max(1, int(round(n * 0.1)))
        edge = np.concatenate([residual[:h], residual[-h:]])
        core = residual[h:-h]
        if core.size:
            holdout_ratio = float(np.mean(edge**2)) / max(
                float(np.mean(core**2)), 1e-12
            )
    feats = [
        math.log10(max(mse / max(var, 1e-12), 1e-16)),
        min(abs(skew), 10.0),
        min(abs(kurt), 50.0),
        math.log1p(min(peak_ratio, 1e6)),
        math.log1p(min(holdout_ratio, 1e6)),
        1.0 if mse < 1e-12 * max(var, 1e-12) else 0.0,
    ]
    return feats, mse


def candidate_features(
    candidate: str, x: np.ndarray, y: np.ndarray, in_mse: float | None
) -> tuple[np.ndarray, list[str]]:
    ops = lib.operator_vector(candidate)
    is_poly, degree = lib.is_polynomial(candidate)
    fingerprint = lib.series_fingerprint(candidate)
    residual, residual_mse = _residual_features(candidate, x, y)
    try:
        from scripts.benchmark_common import formula_complexity_score

        complexity = float(formula_complexity_score(candidate))
    except Exception:
        complexity = float(len(candidate))

    scalar = [
        float(complexity) / 50.0,
        1.0 if is_poly else 0.0,
        float(degree) / 10.0,
        float(fingerprint.get("score") or 0.0),
        float(fingerprint.get("matched_nonzero") or 0) / 8.0,
        float(ops.sum()) / 9.0,
        math.log10(max(float(in_mse), 1e-16)) if in_mse is not None else -16.0,
        math.log10(max(float(residual_mse), 1e-16))
        if residual_mse is not None
        else -16.0,
    ]
    names = [
        "complexity",
        "is_polynomial",
        "degree",
        "series_score",
        "series_terms",
        "op_count",
        "log_in_mse",
        "log_residual_mse",
    ] + [f"residual_{i}" for i in range(len(residual))]
    return np.asarray(ops.tolist() + scalar + residual, dtype=np.float32), names


def build_matrix(
    pairs: list[dict[str, Any]],
    curves: np.ndarray,
    *,
    curve_cache: dict[int, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    cache = curve_cache if curve_cache is not None else {}
    rows: list[np.ndarray] = []
    labels: list[float] = []
    names: list[str] | None = None
    for pair in pairs:
        curve_id = int(pair["curve_id"])
        if curve_id not in cache:
            x = curves[curve_id, 0]
            y = curves[curve_id, 1]
            feats = lib.curve_features(x, y)
            if feats is None:
                feats = np.zeros(CURVE_FEATURES, dtype=np.float32)
            cache[curve_id] = feats
        curve_feats = cache[curve_id]
        x = curves[curve_id, 0].astype(np.float64)
        y = curves[curve_id, 1].astype(np.float64)
        cand, cand_names = candidate_features(
            pair["candidate"], x, y, pair.get("in_mse")
        )
        if names is None:
            names = [f"curve_{i}" for i in range(curve_feats.shape[0])] + cand_names
        rows.append(np.concatenate([curve_feats, cand]))
        labels.append(1.0 if pair["label"] == "exact" else 0.0)
    return np.stack(rows), np.asarray(labels, dtype=np.float32), names or []


class Verifier(nn.Module):
    def __init__(self, n_features: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        roc_auc_score,
    )

    pred = (y_prob >= 0.5).astype(np.float32)
    out = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "pos_rate": float(np.mean(y_true)),
    }
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    return out


def _threshold_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> dict[str, float]:
    pred = (y_prob >= threshold).astype(np.float32)
    tp = float(np.sum((pred == 1) & (y_true == 1)))
    fp = float(np.sum((pred == 1) & (y_true == 0)))
    fn = float(np.sum((pred == 0) & (y_true == 1)))
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fp": int(fp),
        "fn": int(fn),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/exactness_pairs"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--split",
        choices=["source", "group"],
        default="source",
        help="source: train on generated, test on benchmark. group: random formula split.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    pairs = [
        json.loads(line)
        for line in (args.data / "pairs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    curves = np.load(args.data / "curves.npz")["curves"]
    print(f"pairs={len(pairs)} curves={curves.shape[0]}")
    print("labels:", dict(Counter(p["label"] for p in pairs)))

    if args.split == "source":
        train_pairs = [p for p in pairs if p["source"] == "generated"]
        test_pairs = [p for p in pairs if p["source"] == "benchmark"]
    else:
        targets = sorted({p["target"] for p in pairs})
        rng = np.random.RandomState(args.seed)
        rng.shuffle(targets)
        n_train = int(0.75 * len(targets))
        train_targets = set(targets[:n_train])
        train_pairs = [p for p in pairs if p["target"] in train_targets]
        test_pairs = [p for p in pairs if p["target"] not in train_targets]

    print(f"train pairs={len(train_pairs)} test pairs={len(test_pairs)}")
    curve_cache: dict[int, np.ndarray] = {}
    X_train, y_train, names = build_matrix(
        train_pairs, curves, curve_cache=curve_cache
    )
    X_test, y_test, _ = build_matrix(test_pairs, curves, curve_cache=curve_cache)
    print(f"features={X_train.shape[1]} train_pos={y_train.mean():.3f}")

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    model = Verifier(X_train.shape[1], hidden=args.hidden)
    pos_weight = torch.tensor(
        [(y_train == 0).sum() / max((y_train == 1).sum(), 1)], dtype=torch.float32
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_train), torch.tensor(y_train)
    )
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for xb, yb in loader:
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(xb)
            count += len(xb)
        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                train_prob = torch.sigmoid(model(torch.tensor(X_train))).numpy()
            m = _metrics(y_train, train_prob)
            print(
                f"  epoch {epoch:>3} loss={total / max(count, 1):.4f} "
                f"acc={m['accuracy']:.3f} auc={m.get('roc_auc', 0):.3f}"
            )

    model.eval()
    with torch.no_grad():
        test_prob = torch.sigmoid(model(torch.tensor(X_test))).numpy()
        train_prob = torch.sigmoid(model(torch.tensor(X_train))).numpy()

    results = {
        "n_features": int(X_train.shape[1]),
        "feature_names": names,
        "train": _metrics(y_train, train_prob),
        "test": _metrics(y_test, test_prob),
        "thresholds": {
            str(t): _threshold_metrics(y_test, test_prob, t)
            for t in (0.3, 0.5, 0.7, 0.9)
        },
        "split": args.split,
        "epochs": args.epochs,
        "seed": args.seed,
        "train_pairs": len(train_pairs),
        "test_pairs": len(test_pairs),
    }
    print("\nTrain:", results["train"])
    print("Test: ", results["test"])
    for threshold, m in results["thresholds"].items():
        print(
            f"  t={threshold}: precision={m['precision']:.3f} "
            f"recall={m['recall']:.3f} f1={m['f1']:.3f}"
        )

    out = args.out or (args.data / "verifier.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            "hidden": args.hidden,
            "n_features": int(X_train.shape[1]),
            "feature_names": names,
            "metrics": results,
        },
        out,
    )
    (args.data / "verifier_metrics.json").write_text(
        json.dumps(lib.json_safe(results), indent=2), encoding="utf-8"
    )
    print(f"\nVerifier: {out}")


if __name__ == "__main__":
    main()
