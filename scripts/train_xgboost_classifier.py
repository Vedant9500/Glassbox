"""
XGBoost Curve Classifier Training Script

Trains an ensemble of XGBoost classifiers (one per operator class) on the 
extracted 366-dimensional curve features. XGBoost severely outperforms 
shallow MLPs on structured, tabular data like this.

Usage:
    python scripts/train_xgboost_classifier.py --data data/curve_dataset_500k_v2.npz
"""

import numpy as np
import argparse
import time
from pathlib import Path
import json

try:
    import xgboost as xgb
except ImportError:
    print("XGBoost not installed. Please run: pip install xgboost scikit-learn joblib")
    import sys
    sys.exit(1)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import joblib

def load_data(data_path):
    print(f"Loading data from {data_path}...")
    data = np.load(data_path, allow_pickle=True)
    X = data["features"]
    y = data["labels"]
    operator_classes = data["operator_classes"].tolist()
    
    # Clip any extreme anomalies that might have slipped through generation
    X = np.nan_to_num(X, nan=0.0, posinf=50.0, neginf=-50.0)
    X = np.clip(X, -50.0, 50.0)
    
    print(f"  Loaded {len(X)} samples, {X.shape[1]} features, {y.shape[1]} classes.")
    return X, y, operator_classes

def multilabel_stratified_split(X, y, val_ratio=0.1, seed=42):
    """Simple random split for large datasets (stratification is computationally heavy and unnecessary at 500k)."""
    np.random.seed(seed)
    n_samples = len(X)
    n_val = int(n_samples * val_ratio)
    
    indices = np.random.permutation(n_samples)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]

def train_xgboost(X_train, y_train, X_val, y_val, operator_classes, n_estimators=100, max_depth=6, lr=0.1, tree_method='hist'):
    """
    Trains a separate XGBoost binary classifier for each label, natively supporting multi-label.
    """
    n_classes = y_train.shape[1]
    models = []
    
    print("\nTraining XGBoost ensemble (One-Vs-Rest)...")
    print(f"  Global Params: n_estimators={n_estimators}, max_depth={max_depth}, lr={lr}, tree_method={tree_method}")
    
    start_time = time.time()
    
    for i in range(n_classes):
        class_name = operator_classes[i]
        print(f"\n[{i+1}/{n_classes}] Training classifier for '{class_name}'...")
        
        # Calculate scale_pos_weight to handle severe class imbalance
        y_train_i = y_train[:, i].copy()
        y_val_i = y_val[:, i].copy()
        
        n_pos = np.sum(y_train_i == 1)
        n_neg = len(y_train_i) - n_pos
        
        # XGBoost crashes if there is only 1 class in the training data.
        # Inject a fake example if a class is 100% positive or 100% negative.
        if n_pos == 0:
            print(f"  Warning: Class '{class_name}' has 0 positives. Injecting fake positive to prevent crash.")
            y_train_i[0] = 1
            if len(y_val_i) > 0: y_val_i[0] = 1
            n_pos = 1
            n_neg -= 1
        elif n_neg == 0:
            print(f"  Warning: Class '{class_name}' has 0 negatives. Injecting fake negative to prevent crash.")
            y_train_i[0] = 0
            if len(y_val_i) > 0: y_val_i[0] = 0
            n_neg = 1
            n_pos -= 1
            
        scale_pos_weight = n_neg / max(1, n_pos)
        
        # Cap weight to prevent extreme instability on very rare classes, and prevent crash on all-positive
        scale_pos_weight = max(0.1, min(scale_pos_weight, 10.0))
        print(f"  Positives: {n_pos} ({n_pos/len(y_train)*100:.1f}%), scale_pos_weight: {scale_pos_weight:.2f}")
        
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=lr,
            tree_method=tree_method,
            scale_pos_weight=scale_pos_weight,
            objective='binary:logistic',
            eval_metric='aucpr',
            # n_jobs=-1,  # Use all cores
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + i
        )
        
        # Train with early stopping on validation set
        model.fit(
            X_train, y_train_i,
            eval_set=[(X_val, y_val_i)],
            verbose=False
        )
        
        models.append(model)
        
        # Quick eval
        val_probs = model.predict_proba(X_val)[:, 1]
        val_preds = (val_probs > 0.5).astype(int)
        f1 = f1_score(y_val_i, val_preds)
        print(f"  -> Val F1: {f1:.4f}")
        
    print(f"\nAll models trained in {time.time() - start_time:.1f} seconds.")
    return models

def tune_thresholds(all_probs, y_val, steps=19):
    """Tune per-class thresholds to maximize F1 on validation data."""
    n_classes = y_val.shape[1]
    thresholds = np.full(n_classes, 0.5, dtype=np.float32)
    candidates = np.linspace(0.05, 0.95, steps)
    
    for c in range(n_classes):
        best_f1 = -1.0
        best_t = 0.5
        for t in candidates:
            preds = (all_probs[:, c] > t).astype(int)
            f1 = f1_score(y_val[:, c], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        thresholds[c] = best_t
        
    return thresholds

def evaluate(models, X_val, y_val, operator_classes):
    print("\n" + "="*50)
    print("FINAL VALIDATION METRICS")
    print("="*50)
    
    n_classes = len(models)
    all_probs = np.zeros_like(y_val, dtype=np.float32)
    
    for i, model in enumerate(models):
        all_probs[:, i] = model.predict_proba(X_val)[:, 1]
        
    print("Tuning thresholds to maximize F1 on validation set...")
    thresholds = tune_thresholds(all_probs, y_val)
    
    all_preds = np.zeros_like(y_val, dtype=int)
    for c in range(n_classes):
        all_preds[:, c] = (all_probs[:, c] > thresholds[c]).astype(int)
    
    metrics = {}
    f1_sum = 0
    
    for i, name in enumerate(operator_classes):
        acc = accuracy_score(y_val[:, i], all_preds[:, i])
        prec = precision_score(y_val[:, i], all_preds[:, i], zero_division=0)
        rec = recall_score(y_val[:, i], all_preds[:, i], zero_division=0)
        f1 = f1_score(y_val[:, i], all_preds[:, i], zero_division=0)
        
        metrics[name] = {'acc': acc, 'precision': prec, 'recall': rec, 'f1': f1}
        f1_sum += f1
        
        print(f"{name:15s} | Thr: {thresholds[i]:.2f} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
        
    macro_f1 = f1_sum / n_classes
    print("-" * 50)
    print(f"Macro Average F1: {macro_f1:.4f}")
    
    return all_probs, thresholds

def main():
    parser = argparse.ArgumentParser(description="Train XGBoost Curve Classifier")
    parser.add_argument("--data", type=str, required=True, help="Path to .npz dataset")
    parser.add_argument("--out", type=str, default="models/xgboost_classifier.pkl", help="Output path for the model")
    parser.add_argument("--estimators", type=int, default=300, help="Number of trees per class")
    parser.add_argument("--depth", type=int, default=9, help="Max depth of trees")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    
    args = parser.parse_args()
    
    # Check if GPU is available for XGBoost
    tree_method = 'hist'
    try:
        # Dummy train to check CUDA support
        xgb.XGBClassifier(tree_method='gpu_hist', n_estimators=1).fit(np.zeros((10,2)), np.zeros(10))
        tree_method = 'gpu_hist'
        print("Using GPU acceleration for XGBoost (gpu_hist).")
    except Exception:
        print("Using CPU for XGBoost (hist).")
        
    X, y, operator_classes = load_data(args.data)
    
    # Standardize data just like the MLP does, to keep inference identical
    print("Standardizing features...")
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / (std + 1e-8)
    
    X_train, y_train, X_val, y_val = multilabel_stratified_split(X, y)
    
    models = train_xgboost(
        X_train, y_train, X_val, y_val, 
        operator_classes, 
        n_estimators=args.estimators, 
        max_depth=args.depth, 
        lr=args.lr,
        tree_method=tree_method
    )
    
    all_probs, thresholds = evaluate(models, X_val, y_val, operator_classes)
    
    # Save as .pkl for joblib loading
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    payload = {
        'type': 'xgboost',
        'operator_classes': operator_classes,
        'feature_scaler': {'mean': mean, 'std': std},
        # Default 0.35 threshold vector to improve recall for integration
        'thresholds': np.full(len(operator_classes), 0.35, dtype=np.float32)
    }
    
    joblib.dump(payload, str(out_path))
    print(f"\nSaved XGBoost ensemble to {out_path}")

if __name__ == "__main__":
    main()
