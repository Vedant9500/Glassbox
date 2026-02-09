# Glassbox Research Roadmap

## The Razor vs Axe Problem

**Current state**: Fast path (razor) excels at simple formulas but can't handle nested/complex expressions that SOTA (axe) can.

| Difficulty | Example | Razor | Axe (PySR) |
|------------|---------|-------|------------|
| Easy | `sin(x) + x²` | ✅ 0.2s | ✅ 30s |
| Medium | `sin(3.7x) + x^2.3` | ⚠️ 0.001 MSE | ✅ exact |
| Hard | `sin(x²) + log(1+x)` | ❌ | ✅ |
| Very Hard | `exp(-x²)·sin(1/x)` | ❌ | ⚠️ |

---

## Thesis Positioning

> **"Glassbox achieves 100x speedup on 80% of real-world formulas by using learned operator priors to construct targeted basis functions."**

**Key claim**: Most scientific formulas ARE simple. Your razor wins on the common case.

---

## Research Phases

### Phase 1: Strengthen the Razor (Months 1-3)

**Goal**: Handle 90%+ of "easy/medium" formulas with <1s solve time.

- [ ] **Expand classifier coverage**
  - Add: rational, sqrt, inverse, abs, sigmoid
  - Train on 500K synthetic formulas
  - Target: 98%+ accuracy

- [ ] **Smarter basis construction**
  - Nested terms: `sin(x²)`, `exp(-x²)`
  - Product terms: `x·sin(x)`, `x²·exp(-x)`
  - Ratio detection for rational functions

- [ ] **Better constant detection**
  - Symbolic constants: π, e, √2, φ
  - FFT harmonics for multi-frequency
  - Gradient refinement for ω, p

### Phase 2: Add a Chainsaw Mode (Months 4-6)

**Goal**: For hard formulas, use a smarter fallback than pure evolution.

- [ ] **Recursive decomposition**
  - Detect if formula is `f(g(x))` via residual analysis
  - Solve outer function, then inner
  - Example: `sin(x²)` → detect periodic on `y` → find `g(x)=x²`

- [ ] **Grammar-guided search**
  - Define formula grammar: `EXPR → UNARY(EXPR) | BINARY(EXPR, EXPR) | TERM`
  - Use classifier to weight grammar rules
  - Beam search over derivations

- [ ] **Hybrid neural-symbolic**
  - Train a small transformer on (features → formula tokens)
  - Use as proposal generator, verify with regression

### Phase 3: Benchmarking & Paper (Months 7-9)

- [ ] **Standard benchmarks**: Nguyen, Keijzer, Feynman datasets
- [ ] **Speed vs accuracy Pareto plots**
- [ ] **Ablation studies**: classifier, basis expansion, constant detection
- [ ] **Paper draft**: NeurIPS/ICML/AAAI submission

---

## Honest Limitations to Address in Thesis

1. **No nested compositions by default** - Mitigate with recursive decomposition
2. **Requires well-sampled data** - Noise sensitivity analysis
3. **Single-variable focus** - Multi-input is weaker

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Nguyen benchmark accuracy | ~60% | 85%+ |
| Solve time (easy formulas) | 0.2s | <0.5s |
| Solve time (medium formulas) | 23s | <5s |
| Classifier accuracy | 96.7% | 98%+ |
