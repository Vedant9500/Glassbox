"""
Test the C++ evolution engine (P1-P4 optimizations) against formulas
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
