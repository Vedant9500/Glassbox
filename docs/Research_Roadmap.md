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

### Phase 1: Strengthen the Razor (Months 1-3) — COMPLETED

**Goal**: Handle 90%+ of "easy/medium" formulas with <1s solve time.

- [x] **Expand classifier coverage**
  - Add: rational, sqrt, inverse, abs, sigmoid
  - Train on 500K synthetic formulas
  - Target: 96.0% F1 (Achieved — v3 model)

- [x] **Smarter basis construction**
  - Nested terms: `sin(x²)`, `exp(-x²)`
  - Product terms: `x·sin(x)`, `x²·exp(-x)`
  - Ratio detection for rational functions (product-ratio terms added)

- [x] **Better constant detection**
  - Symbolic constants: π, e, √2, φ (Added to templates)
  - FFT harmonics for multi-frequency (Added, range widened to 0.1–50.0)
  - Gradient refinement for ω (`refine_frequencies`), p (`refine_powers`)

---

### Phase 2: Harden the Pipeline (Months 4-6)

**Goal**: Fix known fragilities in feature extraction and search, without a full rewrite. These are high-value, low-disruption changes.

#### 2A — Feature Extraction Fixes ✅

- [x] **Add FFT phase features** *(from §1.1.1)*
  - Added `extract_fft_phase_features()` — 32 phase bins normalized to [-1, 1]
  - Zeroes out phase for low-magnitude bins (<1% of max)
  - Feature vector grew from 334 to 366 dims
  - Rationale: Phase carries structural info (distinguishes `sin(x)+sin(3x)` from `sin(x)·sin(3x)`)

- [x] **Smooth derivative features** *(from §1.1.2)*
  - Added `_smooth_signal()` — Savitzky-Golay filter (window=11, polyorder=3) with moving average fallback
  - Applied before differentiation in `extract_derivative_features()`
  - Classifier retrained on 500K dataset → F1=0.9603 (v3)

- [x] **Curvature-aware resampling** *(from §1.1.3)*
  - Modified `extract_raw_features()` to concentrate samples near high-curvature regions
  - Uses κ = |y''| / (1 + y'²)^1.5 to build non-uniform CDF
  - Configurable via `curvature_alpha` (default=5.0, 0=uniform)

#### 2B — Data Generation Improvements

- [x] **PCFG-based formula generation** *(from §2.2)*
  - Replace fixed templates with recursive grammar rules
  - Grammar: `EXPR → UNARY(EXPR) | BINARY(EXPR, EXPR) | TERM`
  - Generates compositions like `sin(cos(sin(x)))` that no template covers
  - Use uniform tree sampling (Lample & Charton style) for balanced depth distribution

- [x] **Noise-robust training data** *(from §2.1)*
  - Add controlled noise injection at multiple SNR levels during data generation
  - Train classifier to be robust to noisy real-world data
  - Already partially done (noise_std param exists), but increase coverage

#### 2C — Search Improvements

- [ ] **Enhance gradient-informed mutation** *(from §4.3)*
  - Already have `mutate_operations_gradient_informed` — improve it
  - Add subtree-level sensitivity (StruSR-style): protect high-sensitivity subtrees from mutation
  - Target low-sensitivity nodes for aggressive mutation
  - Reference: StruSR masked attribution approach

- [ ] **Recursive decomposition** *(existing roadmap item)*
  - Detect if formula is `f(g(x))` via residual analysis
  - Solve outer function, then inner
  - Example: `sin(x²)` → detect periodic on `y` → find `g(x)=x²`

- [ ] **Grammar-guided search** *(existing + enhanced by §4.4)*
  - Define formula grammar: `EXPR → UNARY(EXPR) | BINARY(EXPR, EXPR) | TERM`
  - Use classifier probabilities to weight grammar rules (LogicSR-style policy)
  - Beam search over derivations
  - Optional: MCTS with classifier as policy network for look-ahead planning

- [ ] **Segmented/Orchestrated Regression (Optional)**
  - Split curve into sections (e.g., by curvature or stationary points)
  - Fit local populations to each section
  - "Orchestrator" merges local expressions into a global formula
  - *Challenge*: Overfitting on small segments
  - *Solution*: Overlapping windows or global structure consistency

---

### Phase 3: Multi-Variate & Structural Priors (Months 7-9)

**Goal**: Fix the multi-variate bottleneck and move from logit biasing to structural seeding. These are larger architectural changes.

#### 3A — Fix Multi-Variate Pipeline

- [ ] **AI Feynman separability test** *(from §3.2)*
  - Before slicing: construct matrix M_ij = F(x_i, y_j), check rank
  - Low rank → function is separable → safe to decompose
  - High rank → non-separable → need interaction-aware approach
  - Lightweight add-on, doesn't require replacing the full slicing system

- [ ] **Symmetry detection** *(from §3.2)*
  - Test translational symmetry: F(x, y) ≈ F(x+Δ, y-Δ) → depends on (x+y)
  - Test scale symmetry: F(λx, λ⁻¹y) ≈ F(x, y) → depends on (xy)
  - Test rotational symmetry: depends on x²+y²
  - Use detected symmetries to reduce dimensionality before regression

- [ ] **Interaction-aware attention** *(from §3.3, Rec. 2)*
  - Treat multivariate tuple (x₁, x₂, ..., xₙ, y) as tokens
  - Self-attention transformer learns coupling strength between variables
  - Extract interaction graph from attention weights → constrain search
  - Replaces 1D slicing for non-separable functions

#### 3B — Structural Seeding (PIGP-Style)

- [ ] **Autoregressive formula generator** *(from §4.2, Rec. 3)*
  - Upgrade classifier from class probabilities → token sequence output
  - Seq2Seq model: features → formula tokens (e.g., ["mul", "x", "sin", "y"])
  - Use beam search (width=50) to generate top-50 candidate equations

- [ ] **Population seeding from generator** *(from §4.2)*
  - Initialize ONN population with the 50 generated trees
  - Keep remaining population random for diversity
  - Replaces current logit biasing with "hot-start" initialization
  - Reference: PIGP (Partially Initialized Genetic Programming)

- [ ] **Hybrid neural-symbolic proposal** *(existing roadmap item, enhanced)*
  - Train a small transformer on (features → formula tokens)
  - Use as proposal generator, verify with regression
  - Now feeds into PIGP seeding instead of standalone

---

### Phase 4: Benchmarking & Paper (Months 10-12)

**Goal**: Rigorous evaluation and publication.

- [ ] **Standard benchmarks**: Nguyen, Keijzer, Feynman datasets
- [ ] **Speed vs accuracy Pareto plots** — key differentiator vs PySR
- [ ] **Ablation studies**: classifier, basis expansion, constant detection, phase features, PCFG data, structural seeding
- [ ] **Multi-variate evaluation**: Feynman multi-input problems specifically
- [ ] **OOD generalization test** *(from §2.4)* — test on Strogatz/Black-Box benchmarks the model was NOT trained on
- [ ] **Paper draft**: NeurIPS/ICML/AAAI submission

---

## Future / Long-Term (Post-Thesis)

These are high-impact but highly disruptive changes from the architecture critique. They represent near-complete rewrites of subsystems and are better suited for a follow-up project.

- [ ] **Dual-Encoder contrastive architecture (MMSR-style)** *(Rec. 1)*
  - Replace 366-feature MLP with Set Transformer/PointNet++ data encoder
  - Add skeleton encoder for expression trees
  - Train with InfoNCE contrastive loss to align data ↔ equation embeddings
  - Enables zero-shot generalization to unseen function types

- [ ] **LLM-distilled data generation (EQUATE)** *(from §2.3)*
  - Query LLM to generate domain-specific equations (physics, biology, etc.)
  - Numerically verify, then use as training data
  - Captures scientific prior that PCFGs miss

- [ ] **GNN variable interaction (EvoNUDGE)** *(from §3.3)*
  - Represent variables as graph nodes, learn edge weights for interaction strength
  - GNN outputs skeleton graph linking interacting variables
  - Fully replaces slicing heuristic

---

## Honest Limitations to Address in Thesis

1. **No nested compositions by default** — Mitigate with recursive decomposition (Phase 2)
2. **Requires well-sampled data** — Noise sensitivity analysis + smoothed features (Phase 2)
3. **Single-variable focus** — Multi-input fixed with separability tests + attention (Phase 3)
4. **Feature bottleneck** — Phase features and smoothing help (Phase 2), full fix is post-thesis
5. **Template-induced OOD ceiling** — PCFG data generation expands coverage (Phase 2)

---

## Success Metrics

| Metric | Current | Phase 2 Target | Phase 3 Target | Phase 4 Target |
|--------|---------|----------------|----------------|----------------|
| Nguyen benchmark accuracy | ~60% | 75%+ | 85%+ | 90%+ |
| Solve time (easy formulas) | 0.2s | <0.5s | <0.5s | <0.5s |
| Solve time (medium formulas) | <1s | <3s | <5s | <5s |
| Classifier accuracy (F1) | 95.6% | 97%+ | 98%+ | 98%+ |
| Multi-variate accuracy | Poor | Poor | 70%+ | 80%+ |
| OOD generalization (Strogatz) | Untested | Untested | 50%+ | 65%+ |

---

## Architecture Critique Coverage Map

Every change from [SR_architecture.md](SR_architecture.md) is accounted for:

| Critique Section | Recommendation | Roadmap Location |
|---|---|---|
| §1.1.1 FFT phase erasure | Add phase features | Phase 2A |
| §1.1.2 Derivative instability | Savitzky-Golay smoothing | Phase 2A |
| §1.1.3 Fixed resampling | Adaptive curvature resampling | Phase 2A |
| §1.2 E2E representation learning | Set Transformer / PointNet | Future |
| §1.3 PointNet encoder | PointNet++ data encoder | Future |
| §2.1 Template bias | PCFG formula generation | Phase 2B |
| §2.2 PCFG data generation | Recursive grammar rules | Phase 2B |
| §2.3 LLM distillation (EQUATE) | LLM-generated equations | Future |
| §2.4 OOD generalization | Strogatz/Black-Box benchmarks | Phase 4 |
| §3.1 1D slicing failure modes | Separability test pre-check | Phase 3A |
| §3.2 AI Feynman symmetry detection | Symmetry tests | Phase 3A |
| §3.3 GNN interaction (EvoNUDGE) | GNN variable interaction | Future |
| §3.3 Attention interaction map | Interaction-aware attention | Phase 3A |
| §3.4 Interpolation artifacts | (Implicit: replaced by attention) | Phase 3A |
| §4.1 Logit biasing weakness | PIGP structural seeding | Phase 3B |
| §4.2 PIGP structural seeding | Autoregressive generator + beam search | Phase 3B |
| §4.3 StruSR guided mutation | Enhance gradient-informed mutation | Phase 2C |
| §4.4 LogicSR MCTS | MCTS option in grammar search | Phase 2C |
