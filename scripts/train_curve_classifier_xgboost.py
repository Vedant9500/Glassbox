"""
Train an XGBoost-based multi-label classifier on curve features.

Usage:
    python scripts/train_curve_classifier_xgboost.py --data training_data/curve_dataset_100k.npz --output models/curve_classifier_xgb.pkl
"""

import argparse
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score


def multilabel_stratified_split(labels: np.ndarray, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Approximate multi-label stratified split without external dependencies."""
    rng = np.random.RandomState(seed)
    n_samples = labels.shape[0]
    n_val = int(n_samples * val_ratio)

    class_pos = labels.sum(axis=0)
    desired_val = np.round(class_pos * val_ratio).astype(int)

    indices = np.arange(n_samples)
    rng.shuffle(indices)

    val_indices = []
    train_indices = []
    remaining_val = n_val
    current_val_counts = np.zeros_like(desired_val, dtype=np.float32)

    for idx in indices:
        if remaining_val <= 0:
            train_indices.append(idx)
            continue

        sample_labels = labels[idx].astype(np.float32)
        needs = np.maximum(desired_val - current_val_counts, 0)
        score = (sample_labels * needs).sum()

        if score > 0:
            val_indices.append(idx)
            current_val_counts += sample_labels
            remaining_val -= 1
        else:
            train_indices.append(idx)

    if remaining_val > 0:
        remaining = [i for i in indices if i not in val_indices and i not in train_indices]
        rng.shuffle(remaining)
        val_indices.extend(remaining[:remaining_val])
        train_indices.extend(remaining[remaining_val:])

    return np.array(train_indices), np.array(val_indices)


def tune_thresholds(all_preds: np.ndarray, all_labels: np.ndarray, steps: int = 19) -> np.ndarray:
    """Tune per-class thresholds to maximize F1 on validation data."""
    thresholds = np.full((all_labels.shape[1],), 0.5, dtype=np.float32)
    candidates = np.linspace(0.05, 0.95, steps)

    for c in range(all_labels.shape[1]):
        best_f1 = -1.0
        best_t = 0.5
        for t in candidates:
            preds = (all_preds[:, c] > t).astype(np.float32)
            labels = all_labels[:, c]
            f1 = f1_score(labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        thresholds[c] = best_t

    return thresholds


def main() -> None:
    parser = argparse.ArgumentParser(description="Train curve classifier with XGBoost (one-vs-rest)")
    parser.add_argument("--data", type=str, required=True, help="Path to training data (.npz file)")
    parser.add_argument("--output", type=str, default="models/curve_classifier_xgb.pkl", help="Output model path")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--standardize", action="store_true", help="Standardize features using training stats (default: on)")
    parser.add_argument("--no-standardize", action="store_true", help="Disable feature standardization")
    parser.add_argument("--stratified-split", action="store_true", help="Use multi-label stratified split (default: on)")
    parser.add_argument("--no-stratified-split", action="store_true", help="Disable stratified split")
    parser.add_argument("--tune-thresholds", action="store_true", help="Tune per-class thresholds (default: on)")
    parser.add_argument("--no-tune-thresholds", action="store_true", help="Disable threshold tuning")
    parser.add_argument("--n-estimators", type=int, default=400, help="Number of boosting rounds")
    parser.add_argument("--max-depth", type=int, default=6, help="Max tree depth")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--subsample", type=float, default=0.8, help="Subsample ratio")
    parser.add_argument("--colsample-bytree", type=float, default=0.8, help="Column subsample ratio")
    parser.add_argument("--early-stopping-rounds", type=int, default=30, help="Early stopping rounds")
    parser.add_argument("--tree-method", type=str, default="hist", help="Tree method (hist, approx, gpu_hist)")
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration if available")
    parser.add_argument("--n-jobs", type=int, default=0, help="Parallel threads (0 = all cores)")

    args = parser.parse_args()

    standardize = not args.no_standardize
    stratified_split = not args.no_stratified_split
    tune_thresholds_flag = not args.no_tune_thresholds

    np.random.seed(args.seed)

    data = np.load(args.data, allow_pickle=True)
    features = data['features']
    labels = data['labels']
    operator_classes = data['operator_classes'].tolist()
    feature_dim = int(data['feature_dim']) if 'feature_dim' in data else features.shape[1]
    feature_schema = data['feature_schema'].item() if 'feature_schema' in data else None

    if stratified_split:
        train_idx, val_idx = multilabel_stratified_split(labels, args.val_split, args.seed)
    else:
        n_val = int(len(features) * args.val_split)
        indices = np.random.permutation(len(features))
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

    train_np = features[train_idx]
    val_np = features[val_idx]

    scaler = None
    if standardize:
        mean = train_np.mean(axis=0)
        std = train_np.std(axis=0) + 1e-8
        train_np = (train_np - mean) / std
        val_np = (val_np - mean) / std
        scaler = {'mean': mean, 'std': std}

    n_classes = labels.shape[1]
    models = []
    val_probs = np.zeros((len(val_idx), n_classes), dtype=np.float32)

    tree_method = "gpu_hist" if args.gpu else args.tree_method

    for c in range(n_classes):
        y_train_c = labels[train_idx][:, c]
        unique_vals = np.unique(y_train_c)
        if unique_vals.size == 1:
            models.append(None)
            val_probs[:, c] = float(unique_vals[0])
            continue

        model = xgb.XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            objective='binary:logistic',
            eval_metric='logloss',
            tree_method=tree_method,
            n_jobs=args.n_jobs,
            random_state=args.seed,
        )
        model.fit(
            train_np,
            y_train_c,
            eval_set=[(val_np, labels[val_idx][:, c])],
            verbose=False,
        )
        models.append(model)
        val_probs[:, c] = model.predict_proba(val_np)[:, 1]

    if tune_thresholds_flag:
        thresholds = tune_thresholds(val_probs, labels[val_idx])
    else:
        thresholds = np.full((n_classes,), 0.5, dtype=np.float32)

    val_preds = (val_probs > thresholds).astype(np.float32)
    f1_macro = f1_score(labels[val_idx], val_preds, average='macro', zero_division=0)
    f1_micro = f1_score(labels[val_idx], val_preds, average='micro', zero_division=0)

    print(f"Val macro-F1: {f1_macro:.4f}")
    print(f"Val micro-F1: {f1_micro:.4f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        'models': models,
        'thresholds': thresholds,
        'operator_classes': operator_classes,
        'feature_scaler': scaler,
        'feature_schema': feature_schema,
        'feature_dim': feature_dim,
    }
    joblib.dump(payload, output_path)
    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    main()
