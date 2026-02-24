"""
Test the C++ evolution engine (P1-P8 optimizations) against formulas
from the benchmark suite.
"""
import numpy as np
import sys
import time
from pathlib import Path

# Setup path
cpp_dir = Path(__file__).resolve().parent.parent / "glassbox" / "sr" / "cpp"
sys.path.insert(0, str(cpp_dir))

try:
    import _core
    print("✅ _core imported successfully\n")
except ImportError as e:
    print(f"❌ Failed to import _core: {e}")
    sys.exit(1)

# ── Test Formulas ──────────────────────────────────────────────────────
# (formula_string, lambda, x_range, pop_size, generations, description)
TEST_CASES = [
    # Tier 1 — Trivial
    ("2*x + 3",       lambda x: 2*x + 3,        (-5, 5),   30,  500,  "Linear"),
    ("x^2",           lambda x: x**2,            (-5, 5),   30,  500,  "Simple quadratic"),

    # Tier 2 — Polynomial
    ("x^3 + x^2 + x", lambda x: x**3 + x**2 + x, (-3, 3), 50,  1000, "Nguyen-1"),

    # Tier 3 — Transcendental (tests P3 Adam on omega)
    ("sin(3x)",       lambda x: np.sin(3*x),     (-6, 6),   50,  1500, "sin(3x) — P3 omega test"),
    ("exp(-x)",       lambda x: np.exp(-x),      (-2, 4),   50,  1000, "Exponential decay"),

    # Tier 5 — Mixed (tests P1 crossover combining subtrees)
    ("x^2 + sin(x)",  lambda x: x**2 + np.sin(x), (-5, 5),  50, 2000, "Poly+trig — P1 crossover test"),
]

print("=" * 72)
print(f"{'Formula':<25} {'MSE':>12} {'Time':>8}  Result")
print("=" * 72)

results = []

for formula_str, func, (xmin, xmax), pop, gens, desc in TEST_CASES:
    np.random.seed(42)
    X = np.linspace(xmin, xmax, 200)
    y = func(X)
    X_list = [X.astype(np.float64)]
    y = y.astype(np.float64)

    t0 = time.perf_counter()
    res = _core.run_evolution(
        X_list, y,
        pop_size=pop,
        generations=gens,
        early_stop_mse=1e-8,
        seed_omegas=[1.0, 2.0, 3.0, 3.14159],
    )
    elapsed = time.perf_counter() - t0

    mse = res["best_mse"]
    formula = res["formula"]

    # Grade
    if mse < 1e-6:
        grade = "🟢 EXACT"
    elif mse < 1e-2:
        grade = "🟡 APPROX"
    elif mse < 1.0:
        grade = "🟠 LOOSE"
    else:
        grade = "🔴 FAIL"

    print(f"{desc:<25} {mse:>12.6e} {elapsed:>7.2f}s  {grade}")
    print(f"  → {formula[:70]}")
    results.append((desc, mse, elapsed, grade))

print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)
exact = sum(1 for _, m, _, _ in results if m < 1e-6)
approx = sum(1 for _, m, _, _ in results if 1e-6 <= m < 1e-2)
loose = sum(1 for _, m, _, _ in results if 1e-2 <= m < 1.0)
fail = sum(1 for _, m, _, _ in results if m >= 1.0)
total_time = sum(t for _, _, t, _ in results)
print(f"  Exact: {exact}/{len(results)}  Approx: {approx}/{len(results)}  "
      f"Loose: {loose}/{len(results)}  Fail: {fail}/{len(results)}")
print(f"  Total time: {total_time:.2f}s  Avg: {total_time/len(results):.2f}s")

# ── P4 Classifier Priors Test ──────────────────────────────────────────
print("\n" + "=" * 72)
print("P4 TEST: Classifier Priors (90% Periodic bias on sin(3x))")
print("=" * 72)

np.random.seed(42)
X = np.linspace(-6, 6, 200)
y = np.sin(3*X)

t0 = time.perf_counter()
res_prior = _core.run_evolution(
    [X.astype(np.float64)], y.astype(np.float64),
    pop_size=50,
    generations=1000,
    early_stop_mse=1e-8,
    seed_omegas=[3.0],
    op_priors=[0.9, 0.03, 0.03, 0.04],  # Heavy Periodic bias
)
elapsed_prior = time.perf_counter() - t0
print(f"  MSE:     {res_prior['best_mse']:.6e}")
print(f"  Time:    {elapsed_prior:.2f}s")
print(f"  Formula: {res_prior['formula']}")

# ── P5 TEST: NSGA-II Multi-Objective ──────────────────────────────────
print("\n" + "=" * 72)
print("P5 TEST: NSGA-II Multi-Objective (Pareto Front)")
print("=" * 72)

np.random.seed(42)
X = np.linspace(-5, 5, 200)
y = (X**2 + np.sin(X)).astype(np.float64)

t0 = time.perf_counter()
res_nsga2 = _core.run_evolution(
    [X.astype(np.float64)], y,
    pop_size=50,
    generations=500,
    early_stop_mse=1e-8,
    use_nsga2=True,
)
elapsed_nsga2 = time.perf_counter() - t0
print(f"  Best MSE:   {res_nsga2['best_mse']:.6e}")
print(f"  Time:       {elapsed_nsga2:.2f}s")
print(f"  Formula:    {res_nsga2['formula'][:70]}")

if "pareto_front" in res_nsga2:
    pf = res_nsga2["pareto_front"]
    print(f"  Pareto front size: {len(pf)}")
    for i, sol in enumerate(pf[:5]):  # Show top 5
        print(f"    [{i}] MSE={sol['mse']:.6e}  Complexity={sol['complexity']}  → {sol['formula'][:50]}")
    
    # Verify non-domination: no solution should dominate another
    dominated = False
    for i, a in enumerate(pf):
        for j, b in enumerate(pf):
            if i != j:
                if a["mse"] <= b["mse"] and a["complexity"] <= b["complexity"]:
                    if a["mse"] < b["mse"] or a["complexity"] < b["complexity"]:
                        dominated = True
    print(f"  Non-domination check: {'✅ PASS' if not dominated else '❌ FAIL'}")
else:
    print("  ❌ No pareto_front key in result!")

# ── P6 TEST: Island Model ────────────────────────────────────────────
print("\n" + "=" * 72)
print("P6 TEST: Island Model (4 islands)")
print("=" * 72)

np.random.seed(42)
X = np.linspace(-5, 5, 200)
y = (X**2 + np.sin(X)).astype(np.float64)

t0 = time.perf_counter()
res_island = _core.run_evolution(
    [X.astype(np.float64)], y,
    pop_size=40,  # 10 per island
    generations=500,
    early_stop_mse=1e-8,
    num_islands=4,
    migration_interval=20,
    migration_size=2,
)
elapsed_island = time.perf_counter() - t0
print(f"  MSE:     {res_island['best_mse']:.6e}")
print(f"  Time:    {elapsed_island:.2f}s")
print(f"  Formula: {res_island['formula'][:70]}")

# Grade
mse_island = res_island["best_mse"]
if mse_island < 1e-6:
    print(f"  Result:  🟢 EXACT")
elif mse_island < 1.0:
    print(f"  Result:  🟡 APPROX/LOOSE")
else:
    print(f"  Result:  🔴 FAIL")

# ── P8 TEST: nn.Module Deserialization ────────────────────────────────
print("\n" + "=" * 72)
print("P8 TEST: CppGraphModule (nn.Module Deserialization)")
print("=" * 72)

try:
    import torch
    # Use the basic result from the first test
    np.random.seed(42)
    X = np.linspace(-5, 5, 200)
    y = (2*X + 3).astype(np.float64)

    res_p8 = _core.run_evolution(
        [X.astype(np.float64)], y,
        pop_size=30, generations=500, early_stop_mse=1e-8,
    )

    # Add glassbox to path for import
    glassbox_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(glassbox_root))
    from glassbox.sr.cpp.export_pytorch import CppGraphModule, cpp_result_to_module

    module = cpp_result_to_module(res_p8)
    print(f"  Module type: {type(module).__name__}")
    print(f"  Formula:     {module.get_formula()[:60]}")

    # Test forward pass
    x_tensor = torch.tensor(X, dtype=torch.float64).unsqueeze(1)
    with torch.no_grad():
        pred = module(x_tensor)

    assert pred.shape == (200,), f"Output shape mismatch: {pred.shape}"
    pred_np = pred.numpy()
    mse_torch = np.mean((pred_np - y) ** 2)
    print(f"  C++ MSE:     {res_p8['best_mse']:.6e}")
    print(f"  PyTorch MSE: {mse_torch:.6e}")

    # Check that PyTorch prediction roughly matches C++ MSE
    if mse_torch < res_p8['best_mse'] * 10 + 1e-4:
        print(f"  Parity:      ✅ PASS (PyTorch matches C++)")
    else:
        print(f"  Parity:      ⚠️ LOOSE (MSE diverged)")

    # Test parameter count
    n_params = sum(p.numel() for p in module.parameters())
    n_buffers = sum(b.numel() for b in module.buffers())
    print(f"  Parameters:  {n_params}  Buffers: {n_buffers}")

except ImportError as e:
    print(f"  ⚠️ Skipped (torch not available: {e})")
except Exception as e:
    print(f"  ❌ Failed: {e}")

print("\n" + "=" * 72)
print("ALL P1-P8 TESTS COMPLETE")
print("=" * 72)
