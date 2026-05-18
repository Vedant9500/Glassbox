"""Comprehensive Curve Classifier Evaluation.

Tests three things:
1. Y-INVARIANCE: Do predictions stay identical when y -> 5*y + 100? (should be 100%)
2. IN-DISTRIBUTION ACCURACY: Per-class F1 on standard domain [-5, 5]
3. OOD ACCURACY: Per-class F1 on moderate shifted domain [5, 15]
"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

import torch
import numpy as np
from sklearn.metrics import f1_score
from glassbox.curve_classifier.generate_curve_data import (
    extract_all_features, generate_dataset, ALL_TEMPLATES, OPERATOR_CLASSES
)
from glassbox.curve_classifier.curve_classifier_integration import CurveClassifierGLU, CurveClassifierMLP

DEFAULT_MODEL_CANDIDATES = [
    "models/curve_classifier_wide.pt",
    "models/curve_classifier_mlp_eql.pt",
    "models/curve_classfier_v4.pt",
    "models/curve_classifier_wider.pt",
    "models/curve_classifier.pt",
]


def load_model(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    config = ckpt['model_config']
    model_type = ckpt.get('model_type', 'glu')
    model_cls = CurveClassifierMLP if model_type == 'mlp' else CurveClassifierGLU
    model = model_cls(
        n_features=config['n_features'],
        n_classes=config['n_classes'],
        hidden=config['hidden']
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt.get('feature_scaler'), ckpt.get('thresholds')


def apply_curve_feature_transform(features):
    features = np.array(features, dtype=np.float32, copy=True)
    end = min(features.shape[1], 398)
    if end > 192:
        features[:, 192:end] = np.sign(features[:, 192:end]) * np.log1p(np.abs(features[:, 192:end]))
    return features


def predict_batch(model, scaler, thresholds, features):
    features = apply_curve_feature_transform(features)
    ft = torch.tensor(features, dtype=torch.float32)
    if scaler:
        ft = (ft - torch.tensor(scaler['mean'])) / (torch.tensor(scaler['std']) + 1e-8)
    with torch.no_grad():
        probs = torch.sigmoid(model(ft)).numpy()
    if thresholds is None:
        thresholds = np.full(probs.shape[1], 0.5, dtype=np.float32)
    return (probs > np.asarray(thresholds, dtype=np.float32)).astype(int)


def resolve_model_path():
    for path in DEFAULT_MODEL_CANDIDATES:
        if os.path.exists(path):
            return path
    return DEFAULT_MODEL_CANDIDATES[0]


def test_y_invariance(model, scaler, thresholds):
    """Gold-standard basis-independence test: predictions under y -> a*y + b."""
    x = np.linspace(-5, 5, 256)
    curves = [
        ("sin(x)",       np.sin(x)),
        ("cos(2x)",      np.cos(2*x)),
        ("exp(0.5x)",    np.exp(0.5*x)),
        ("x^2",          x**2),
        ("x^3",          x**3),
        ("log(|x|+1)",   np.log(np.abs(x) + 1)),
        ("1/(1+x^2)",    1/(1+x**2)),
        ("sqrt(|x|)",    np.sqrt(np.abs(x))),
        ("sin(x)*x",     np.sin(x)*x),
        ("x^2+sin(x)",   x**2 + np.sin(x)),
    ]
    op_names = list(OPERATOR_CLASSES.keys())
    
    print("\n" + "="*80)
    print("TEST 1: Y-INVARIANCE  (y -> 5*y + 100, predictions must be identical)")
    print("="*80)
    print(f"{'Formula':<18} {'Original':<28} {'Transformed':<28} {'Match'}")
    print("-"*80)
    
    matches = 0
    for name, y in curves:
        f1 = extract_all_features(y).reshape(1, -1)
        f2 = extract_all_features(5 * y + 100).reshape(1, -1)
        p1 = predict_batch(model, scaler, thresholds, f1)[0]
        p2 = predict_batch(model, scaler, thresholds, f2)[0]
        
        ops1 = [op_names[i] for i in range(len(p1)) if p1[i]]
        ops2 = [op_names[i] for i in range(len(p2)) if p2[i]]
        match = np.array_equal(p1, p2)
        matches += int(match)
        
        mark = "YES" if match else "NO !!!"
        print(f"{name:<18} {str(ops1):<28} {str(ops2):<28} {mark}")
    
    pct = 100 * matches / len(curves)
    print(f"\nY-Invariance: {matches}/{len(curves)} = {pct:.0f}%")
    return pct


def test_accuracy(model, scaler, thresholds, x_range, n_samples=2000):
    """Per-class accuracy on a specific domain."""
    features, labels, _ = generate_dataset(
        n_samples=n_samples, x_range=x_range, n_points=256,
        templates=ALL_TEMPLATES, noise_std=0.01,
        balance_classes=True, seed=999
    )
    preds = predict_batch(model, scaler, thresholds, features)
    op_names = list(OPERATOR_CLASSES.keys())
    
    print(f"\n{'='*60}")
    print(f"ACCURACY on domain {x_range}  ({n_samples} samples)")
    print(f"{'='*60}")
    print(f"  {'Operator':<15} {'True':>6} {'Pred':>6} {'Prec':>7} {'Recall':>7} {'F1':>7}")
    print(f"  {'-'*52}")
    
    for i, op in enumerate(op_names):
        tp = int(((preds[:,i]==1) & (labels[:,i]==1)).sum())
        tc = int(labels[:,i].sum())
        pc = int(preds[:,i].sum())
        p = tp / pc if pc > 0 else 0
        r = tp / tc if tc > 0 else 0
        f = 2*p*r / (p+r+1e-12)
        print(f"  {op:<15} {tc:>6} {pc:>6} {p:>7.3f} {r:>7.3f} {f:>7.3f}")
    
    macro = f1_score(labels, preds, average='macro', zero_division=0)
    print(f"\n  Macro-F1: {macro:.4f}")
    return macro


def main():
    path = resolve_model_path()
    if not os.path.exists(path):
        print(f"Model not found: {path}"); return
    
    model, scaler, thresholds = load_model(path)
    
    print("="*60)
    print("COMPREHENSIVE CURVE CLASSIFIER EVALUATION")
    print(f"Model: {path}")
    print("="*60)
    
    y_inv = test_y_invariance(model, scaler, thresholds)
    f1_std = test_accuracy(model, scaler, thresholds, (-5, 5))
    f1_ood = test_accuracy(model, scaler, thresholds, (5, 15))
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  Y-Invariance:      {y_inv:.0f}%")
    print(f"  F1 on (-5, 5):     {f1_std:.4f}")
    print(f"  F1 on (5, 15):     {f1_ood:.4f}")
    print(f"  OOD Gap:           {abs(f1_std - f1_ood):.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
