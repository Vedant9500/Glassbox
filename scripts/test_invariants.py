import numpy as np
from glassbox.curve_classifier.generate_curve_data import extract_raw_features, ALL_TEMPLATES, generate_dataset
import time

def compute_differential_invariants(y, dx=1.0):
    n = len(y)
    if n < 7:
        return [1.0, 1.0, 1.0, 1.0]
        
    k1 = np.array([-3, -2, -1, 0, 1, 2, 3]) / (28.0 * dx)
    k2 = np.array([5, 0, -3, -4, -3, 0, 5]) / (42.0 * dx**2)
    k3 = np.array([-1, 1, 1, 0, -1, -1, 1]) / (6.0 * dx**3)
    
    inv_exps, inv_sins, inv_pows, inv_rats = [], [], [], []
    
    for i in range(3, n - 3):
        window = y[i-3:i+4]
        dy = np.sum(window * k1)
        ddy = np.sum(window * k2)
        dddy = np.sum(window * k3)
        
        if np.abs(dy) < 1e-6:
            continue
            
        inv_exp = ddy / dy
        inv_sin = dddy / dy
        inv_pow = (dy * dddy) / (ddy**2) if np.abs(ddy) > 1e-6 else 0
        inv_rat = inv_sin - 1.5 * (inv_exp**2)
        
        inv_exps.append(inv_exp)
        inv_sins.append(inv_sin)
        inv_pows.append(inv_pow)
        inv_rats.append(np.abs(inv_rat))
        
    if len(inv_exps) < 2:
        return [1.0, 1.0, 1.0, 1.0]
        
    return [
        np.min([np.var(inv_exps), 1.0]),
        np.min([np.var(inv_sins), 1.0]),
        np.min([np.var(inv_pows), 1.0]),
        np.min([np.mean(inv_rats), 1.0])
    ]

if __name__ == '__main__':
    # Generate some clean and noisy data
    print("Generating test data...")
    features, labels, formulas = generate_dataset(
        n_samples=1000, 
        n_points=256, 
        show_progress=False, 
        noise_profile='multi'
    )

    print(f"Generated {len(features)} curves")

    t0 = time.time()
    invariants = []
    for i in range(len(features)):
        raw = features[i, :128] # Raw curve part
        inv = compute_differential_invariants(raw, dx=1.0/128.0)
        invariants.append(inv)
    t1 = time.time()
    invariants = np.array(invariants)

    print(f"Computed invariants in {t1-t0:.4f}s for {len(features)} curves (avg {(t1-t0)/len(features)*1000:.3f}ms per curve)")

    # See how well they correlate with actual labels
    # Labels: 'identity':0, 'sin':1, 'cos':2, 'power':3, 'exp':4, 'log':5, 'addition':6, 'multiplication':7, 'rational':8
    for cls_idx, cls_name in enumerate(['identity', 'sin', 'cos', 'power', 'exp', 'log', 'addition', 'multiplication', 'rational']):
        mask = labels[:, cls_idx] == 1
        if np.sum(mask) == 0: continue
        
        mean_inv = np.mean(invariants[mask], axis=0)
        print(f"Class {cls_name:15s} ({np.sum(mask)} samples) -> Mean Invariants: Var_Exp={mean_inv[0]:.4f}, Var_Sin={mean_inv[1]:.4f}, Var_Pow={mean_inv[2]:.4f}, Mean_Rat={mean_inv[3]:.4f}")
