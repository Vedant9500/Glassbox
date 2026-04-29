import numpy as np
import time
from glassbox.sr.sklearn_wrapper import GlassboxRegressor

# A small set of test functions
funcs = [
    ("easy_poly", lambda x: 2.5 * x**2 - 1.2 * x + 3.0),
    ("easy_sin", lambda x: 1.5 * np.sin(2.0 * x)),
    ("ood_composition", lambda x: np.sin(x**2) + np.cos(x)),
    ("hard_rational", lambda x: (x**2 + 1) / (x + 2)),
]

modes = [
    ("Legacy", dict(use_fast_path=True, use_universal_proposer=False, use_guided_evolution=True)),
    ("ProposerOnly", dict(use_fast_path=False, use_universal_proposer=True, universal_proposer_shadow_mode=False, use_guided_evolution=True)),
    ("Hybrid", dict(use_fast_path=True, use_universal_proposer=True, universal_proposer_shadow_mode=False, use_guided_evolution=True)),
]

results = {m[0]: {"exact": 0, "approx": 0, "time": 0.0, "time_first_acc": 0.0} for m in modes}

for name, fn in funcs:
    np.random.seed(42)
    x = np.linspace(-3.0, 3.0, 100, dtype=np.float64).reshape(-1, 1)
    y = fn(x.ravel())
    
    print(f"\nEvaluating: {name}")
    for mode_name, kwargs in modes:
        gb = GlassboxRegressor(
            generations=5, # fast for test
            population_size=10,
            timeout=10,
            random_state=42,
            **kwargs
        )
        
        start = time.time()
        gb.fit(x, y)
        elapsed = time.time() - start
        
        preds = gb.predict(x)
        mse = np.mean((y - preds)**2)
        
        # Determine exactness
        exact = mse < 1e-4
        approx = mse < 1e-1
        
        results[mode_name]["time"] += elapsed
        if hasattr(gb, "time_to_first_acceptable_sec_") and gb.time_to_first_acceptable_sec_:
            results[mode_name]["time_first_acc"] += gb.time_to_first_acceptable_sec_
        else:
            results[mode_name]["time_first_acc"] += elapsed
            
        if exact:
            results[mode_name]["exact"] += 1
        if approx:
            results[mode_name]["approx"] += 1
            
        print(f"  {mode_name}: MSE={mse:.5f}, Time={elapsed:.2f}s, Formula={gb.formula_}")

print("\n--- Summary ---")
for m, r in results.items():
    print(f"{m}: Exact={r['exact']}/{len(funcs)}, Approx={r['approx']}/{len(funcs)}, AvgTime={r['time']/len(funcs):.2f}s")
