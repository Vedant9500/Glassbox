# Replace SymPy with Native C++ Formula Simplification

## Current Status (2026-06-03)

This plan is partially implemented. Native simplification, float snapping,
formula-to-seed-graph parsing, noise reduction, and candidate scoring are
available through `_core`, but SymPy remains in `requirements.txt` as a guarded
fallback in active Python paths. See `docs/CPP_Migration_Roadmap.md` for the
current native-backend status.

## Problem Statement

The Glassbox SR pipeline currently depends on SymPy (pure Python) for formula simplification, parsing, and display. This creates two major problems:

1. **Performance**: SymPy adds 1–5 seconds per formula simplification call. In the `sklearn_wrapper.fit()` pipeline, this is called on every single fit. The C++ ↔ Python boundary crossing for `seed_graph_builder.py` (formula→AST) and `simplify_formula.py` (AST→simplified string) causes measurable overhead.
2. **Dependency weight**: SymPy pulls in a heavy dependency tree and adds >1s just for import startup.

Your existing C++ layer already has:
- **AST representation**: [ast.h](file:///d:/Glassbox/glassbox/sr/cpp/ast.h) — `OpNode`, `IndividualGraph`, structural hashing
- **Evaluation engine**: [eval.h](file:///d:/Glassbox/glassbox/sr/cpp/eval.h) — full graph evaluation + `get_formula_string()`
- **Basic simplification**: [simplify.h](file:///d:/Glassbox/glassbox/sr/cpp/simplify.h) — constant folding + dead node elimination
- **Numerical refinement**: [refine.h](file:///d:/Glassbox/glassbox/sr/cpp/refine.h) — elastic net, frequency/power refinement

But `simplify.h` currently only does:
- Constant folding (evaluate unary/binary ops on constant children)
- Dead node removal (compact unused nodes)
- Output bias folding (fold constant nodes into `output_bias`)

It does **not** do any algebraic simplification (like `x + x → 2x`, `sin²+cos²→1`, `exp(log(x))→x`, term collection, etc.).

---

## SymPy Usage Audit — All Sites

### 🔴 HOT PATH / HIGH PRIORITY — Called during `fit()` pipeline

| File | SymPy Usage | Context |
|------|------------|---------|
| [simplify_formula.py](file:///d:/Glassbox/scripts/simplify_formula.py) | `parse_expr`, `simplify`, `factor_terms`, `powsimp`, `together`, `cancel`, `ratsimp`, `trigsimp`, `expand_trig`, `nsimplify`, `count_ops`, `sympify`, `srepr`, `N`, `diff`, `expand`, `Abs`, `sign`, `Float`, `Integer`, `Add.make_args`, `as_coeff_Mul` | **Core simplification pipeline.** Multi-pass algebraic simplification of formula strings. Called by `sklearn_wrapper._simplify_formula()` on every fit. |
| [sklearn_wrapper.py](file:///d:/Glassbox/glassbox/sr/sklearn_wrapper.py#L811-L905) | `parse_expr`, `Add.make_args`, `lambdify`, `Symbol` | **Noise reduction** (`_reduce_formula_noise`): parses formula into SymPy, decomposes into additive terms, uses BIC-guided backward elimination. Called on every fit. |
| [seed_graph_builder.py](file:///d:/Glassbox/glassbox/sr/cpp/seed_graph_builder.py) | `parse_expr`, `expand`, `Symbol`, `Mul`, `Add`, `Pow`, `sin`, `cos`, `exp`, `log`, `Abs`, `sqrt`, `Piecewise`, `Eq`, `Float`, `Integer`, `Rational`, `is_Number`, `as_coeff_Mul`, `free_symbols`, `evalf` | **Seed graph construction**: Parses formula strings into C++ `IndividualGraph` dicts for seeding evolution. Called during fit when proposer/fast-path produces candidate formulas. |

### 🟡 WARM PATH — Called during training but not innermost loop

| File | SymPy Usage | Context |
|------|------------|---------|
| [operation_dag.py](file:///d:/Glassbox/glassbox/sr/core/operation_dag.py#L422-L459) | `symbols`, `sympify`, `simplify`, `sin`, `cos`, `exp`, `log`, `sqrt`, `Pow` | Legacy PyTorch DAG `_simplify_formula()` method. Used when extracting formulas from the ONN (torch-based path). |
| [evolution.py](file:///d:/Glassbox/glassbox/evolution/evolution.py#L2647-L2680) | `Symbol`, `sympify`, `lambdify` | Post-C++ evolution: converts formula string back to a callable for MSE verification. Cold path (once per fit). |

### 🟢 COLD PATH — Scripts, benchmarks, tests

| File | SymPy Usage | Context |
|------|------------|---------|
| [benchmark_suite.py](file:///d:/Glassbox/scripts/benchmark_suite.py) | `parse_expr` | Formula parsing in benchmarks |
| [classifier_fast_path.py](file:///d:/Glassbox/scripts/classifier_fast_path.py) | `parse_expr` | Formula parsing in fast-path script |
| [evolution_pipeline_log.py](file:///d:/Glassbox/scripts/evolution_pipeline_log.py) | `Symbol`, `sympify`, `lambdify` | Logging/evaluation script |
| [srbench_interface/regressor.py](file:///d:/Glassbox/scripts/srbench_interface/regressor.py) | `sympy` | SRBench compatibility layer |
| [test_simplify_formula.py](file:///d:/Glassbox/tests/test_simplify_formula.py) | `simplify`, `Integer`, `symbols`, `sin` | Unit tests |

---

## Open Questions

> [!IMPORTANT]
> **Q1: Scope of replacement — full or partial?**
> Option A: Replace ALL SymPy usage (full independence from sympy). This means building a formula **parser** (string → AST) in C++ and full algebraic simplification.
> Option B: Replace only the hot-path simplification (`simplify_formula.py` logic) in C++ but keep SymPy as an optional fallback for seed graph building and the cold-path scripts.
> **Recommendation**: Phase 1 = Option B (high-value, lower risk). Phase 2 = Option A (full independence).

> [!IMPORTANT]
> **Q2: Trig identity depth — how much do we need?**
> The SymPy pipeline currently applies `trigsimp(method="fu")`, `expand_trig`, and full trig identity resolution (sin²+cos²=1, angle addition, double angle). These are complex to implement from scratch.
> Option A: Implement only the identities that commonly arise in SR output (sin²+cos²=1, exp(log(x))=x, power simplification).
> Option B: Implement a full trigonometric identity engine.
> **Recommendation**: Option A — the evolution engine already produces clean parameterized forms (amplitude×sin(ω×x+φ)), so deep trig rewriting is rarely needed.

> [!IMPORTANT]
> **Q3: Formula parser — custom or lightweight library?**
> `seed_graph_builder.py` currently uses SymPy's `parse_expr` to parse arbitrary formula strings into AST. Replacing this requires a math expression parser in C++.
> Option A: Write a simple recursive-descent parser (covers `+`, `-`, `*`, `/`, `^`, `sin`, `cos`, `exp`, `log`, `abs`, `sqrt`, parentheses, variables).
> Option B: Use a lightweight C++ expression parser library (e.g., ExprTk, muParser).
> **Recommendation**: Option A — the grammar is small and controlled. You already know the exact function set from your AST.

---

## Proposed Changes

### Phase 1: Native Algebraic Simplification Engine (High Priority)

This phase replaces `simplify_formula.py` and the SymPy calls in `sklearn_wrapper._simplify_formula()`.

---

#### [MODIFY] [simplify.h](file:///d:/Glassbox/glassbox/sr/cpp/simplify.h)

Expand from 157 lines to ~600 lines. Add the following simplification passes:

**1. Float Snapping** (replaces `snap_formula_floats`):
- Walk all nodes, snap `value`/`p`/`omega`/`phi`/`amplitude` to nearest integer within tolerance
- Zero out tiny values (< 1e-8)
- Snap output weights and bias

**2. Algebraic Identity Rules** (replaces `sympy.simplify` + `trigsimp`):
- **Identity operations**: `x^1 → x`, `x^0 → 1`, `1*x → x`, `0+x → x`, `x-x → 0`
- **Power collapse**: `(x^a)^b → x^(a*b)`, `x^2 * x^3 → x^5`
- **Inverse cancellation**: `exp(log(x)) → x`, `log(exp(x)) → x`
- **Trig Pythagorean**: When two sibling nodes are `sin²(ωx+φ)` and `cos²(ωx+φ)` with same params, collapse to constant 1
- **Double negation**: `--x → x`
- **Multiplication by zero**: Any branch multiplied by 0 → prune

**3. Like-Term Collection** (replaces `factor_terms` + `cancel`):
- In the output layer: merge nodes that compute the same subtree (using structural hashing from `ast.h`)
- Sum their output weights: if `w1 * f(x) + w2 * f(x)`, consolidate to `(w1+w2) * f(x)`

**4. Output Layer Cleanup**:
- Remove nodes with output weight ≈ 0 (already done, but tighten threshold)
- Sort nodes by absolute weight descending for canonical ordering
- Re-fold absorbed constants into bias

**5. Integer Rounding for Display** (replaces `nsimplify`):
- If `p` is within 1e-4 of an integer, snap to integer
- If `omega` is within 1e-4 of a "nice" value (π, 2π, π/2, etc.), snap for display

---

#### [NEW] `simplify_advanced.h`

New file (~400 lines) for the more complex transforms:

**1. Noise Reduction / BIC Pruning** (replaces `_reduce_formula_noise` in sklearn_wrapper):
- Given graph + data (X, y), evaluate each output node independently
- Build a term-by-term design matrix
- Run greedy backward BIC elimination using Eigen (reuse `solve_linear` from refine.h)
- Zero out output weights for eliminated terms
- This is currently done in Python with sklearn's `LinearRegression` — moving it to C++ with Eigen eliminates the SymPy `parse_expr` + `lambdify` round-trip entirely

**2. Dominant Trig Mode Collapse** (replaces `_collapse_dominant_trig_mode`):
- Scan output nodes for periodic terms with the same frequency
- If one frequency dominates (>90% of total amplitude), drop weak harmonics
- This is much simpler operating on the C++ AST directly than on SymPy expressions

---

#### [NEW] `formula_parser.h`

New file (~300 lines) — a recursive-descent parser for mathematical formula strings:

```
Grammar:
  expr     → term (('+' | '-') term)*
  term     → unary (('*' | '/') unary)*
  unary    → ('-')? power
  power    → primary ('^' | '**') unary | primary
  primary  → NUMBER | VARIABLE | FUNCTION '(' expr ')' | '(' expr ')' | '|' expr '|'
  FUNCTION → 'sin' | 'cos' | 'exp' | 'log' | 'sqrt' | 'abs' | 'sign'
  VARIABLE → 'x' | 'x0' | 'x1' | ...
```

This replaces `seed_graph_builder.py`'s SymPy-based formula parsing, allowing `formula_to_seed_graph` to be called entirely from C++.

Returns an `IndividualGraph` directly, avoiding the Python→dict→C++ serialization overhead.

---

#### [MODIFY] [core.cpp](file:///d:/Glassbox/glassbox/sr/cpp/core.cpp)

Add pybind11 bindings for:
- `simplify_formula_cpp(formula_str) → simplified_str` — full pipeline: parse → AST → simplify → format
- `formula_to_seed_graph_cpp(formula_str) → graph_dict` — replaces Python `formula_to_seed_graph`
- `reduce_formula_noise_cpp(graph_dict, X, y) → graph_dict` — BIC-based term elimination
- `snap_formula_floats_cpp(formula_str) → snapped_str` — float snapping

---

#### [MODIFY] [sklearn_wrapper.py](file:///d:/Glassbox/glassbox/sr/sklearn_wrapper.py)

- `_simplify_formula()`: Replace SymPy call with `_core.simplify_formula_cpp()`
- `_reduce_formula_noise()`: Replace SymPy-based decomposition with `_core.reduce_formula_noise_cpp()`
- Remove `import sympy` from this file entirely

---

#### [MODIFY] [seed_graph_builder.py](file:///d:/Glassbox/glassbox/sr/cpp/seed_graph_builder.py)

- Add C++ fallback: try `_core.formula_to_seed_graph_cpp()` first, fall back to SymPy-based parsing
- Eventually delete the SymPy-based implementation once C++ parser is proven

---

### Phase 2: Full SymPy Independence (Future)

#### [MODIFY] [operation_dag.py](file:///d:/Glassbox/glassbox/sr/core/operation_dag.py)
- Replace `_simplify_formula()` to use C++ simplification or a lightweight Python AST-based approach (no SymPy)

#### [MODIFY] [evolution.py](file:///d:/Glassbox/glassbox/evolution/evolution.py)
- Replace `sympify` + `lambdify` MSE verification with direct C++ graph evaluation (graph already exists)

#### [DELETE] SymPy from `requirements.txt` (once all paths migrated)

---

## Verification Plan

### Automated Tests

1. **Parity tests**: Extend [test_cpp_parity.py](file:///d:/Glassbox/glassbox/sr/test_cpp_parity.py) with simplification parity checks:
   - Feed the same formula strings to both SymPy pipeline and C++ pipeline
   - Verify output strings are algebraically equivalent (evaluate at random points)

2. **Round-trip tests**: `formula_string → parse → AST → simplify → format → string` should produce stable results:
   ```
   input: "0.99999*sin(3.00001*x) + 0.0000001*cos(x)"
   expected: "sin(3*x)"
   ```

3. **Regression suite**: Port all 5 test cases from [test_simplify_formula.py](file:///d:/Glassbox/tests/test_simplify_formula.py) to C++:
   - Pythagorean identity: `sin(x)^2 + cos(x)^2 → 1`
   - Float snapping: `0.999999999*x + 1.000000001*x → 2*x`
   - Dominant trig collapse

4. **Benchmark**: Compare wall-clock time of `simplify_onn_formula()` (Python/SymPy) vs `simplify_formula_cpp()` for the formulas from the SRBench suite

### Manual Verification

- Run the full `sklearn_wrapper.fit()` pipeline on 5 representative benchmark problems
- Verify that formula output quality is ≥ SymPy output quality (check R², formula complexity, readability)
- Profile to confirm the SymPy overhead is eliminated
