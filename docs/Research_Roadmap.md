# Glassbox Research Roadmap (Evolution-First)

## Strategic Shift (March 2026)

- Evolution path is now the primary research focus.
- Fast path (v3.1 MLP trained on about 500K curve-formula pairs) is retained as a baseline and optional hint source only.
- Core objective: beat PySR on the evolution path under fair compute and complexity budgets.

---

## Baseline vs Main Engine

**Current state**: Fast path is a useful baseline for easy formulas, but thesis value now comes from making evolution competitive on medium/hard symbolic discovery.

| Difficulty | Example | Fast-Path Baseline | Evolution Path (Main) | Axe (PySR) |
|------------|---------|--------------------|------------------------|------------|
| Easy | `sin(x) + x^2` | ✅ very fast | ✅ should match quickly | ✅ |
| Medium | `sin(3.7x) + x^2.3` | ⚠️ often approximate | 🎯 exact or near-exact target | ✅ exact |
| Hard | `sin(x^2) + log(1+x)` | ❌ | 🎯 exact target | ✅ |
| Very Hard | `exp(-x^2)*sin(1/x)` | ❌ | 🎯 robust approximation target | ⚠️ |

---

## Thesis Positioning

> **"Glassbox delivers an evolution-first symbolic regression engine that approaches or exceeds PySR on exact recovery and time-to-discovery, with strict formula-fidelity scoring and reproducible multi-seed performance."**

**Key claim**: The main contribution is not fast-path speed alone; it is an evolution system that is accurate, stable, and scientifically trustworthy.

---

## Research Phases

### Phase 0: Benchmark Integrity and Evaluation Discipline (Immediate, 2-4 weeks) — NEW

**Goal**: Eliminate false progress and enforce fair, reproducible comparison before algorithmic changes.

- [ ] **Standardized benchmark protocol (Glassbox vs PySR parity)**
  - Same operator sets, complexity limits, data ranges, noise levels, and timeout/wall-clock budgets
  - Same core/thread budget and hardware reporting
  - Fixed seed set (for example 10-20 seeds) for all comparisons

- [ ] **Objective-formula alignment (hard requirement)**
  - Always score exported/displayed formula on evaluation data
  - Keep raw internal fitness for diagnostics only
  - Add drift metric between raw and displayed MSE; do not allow exact labels from raw-only fit

- [ ] **Multi-seed stability reporting by default**
  - Report median, IQR/std, and worst-decile performance
  - Track variance of exact recovery, not only best run

- [ ] **Time-to-discovery metrics**
  - Time to first exact expression
  - Time to first acceptable expression under complexity cap
  - Track budget-to-quality curve, not just end-of-run quality

- [ ] **Failure taxonomy pipeline**
  - Auto-bucket failures: exp sign errors, product-to-sum collapse, missing high-order terms, rational denominator instability, etc.
  - Use taxonomy as a closed-loop input to mutation/operator updates

- [ ] **Data leakage and identifiability audit**
  - Formula-family split checks between train/test generation grammars
  - Multi-range evaluation to reduce local-shape aliasing

---

### Phase 1: Fast-Path Baseline Track (Months 1-3) — COMPLETED, FROZEN

**Goal**: Keep as baseline; avoid spending core research bandwidth here.

- [x] **Expand classifier coverage**
  - Added rational, sqrt, inverse, abs, sigmoid
  - Trained on 500K synthetic formulas
  - Achieved target F1 on v3-class models

- [x] **Smarter basis construction**
  - Added nested terms and product terms
  - Added ratio-aware terms for rational forms

- [x] **Better constant detection**
  - Added symbolic constants and wider FFT harmonic range
  - Added frequency/power refinement helpers

**Policy for Phase 1 artifacts**:
- Fast-path improvements are maintenance only.
- No major roadmap decisions should depend on fast-path gains.

---

### Phase 2: Evolution Core Reliability (Months 4-6)

**Goal**: Improve pure evolution quality, fidelity, and robustness without full architecture replacement.

#### 2A — Search and Mutation Quality

- [ ] **Enhance gradient-informed mutation**
  - Upgrade current gradient-informed mutation with subtree-level sensitivity masks
  - Protect high-sensitivity subtrees; mutate low-sensitivity regions more aggressively

- [ ] **Structure-preserving crossover/mutation**
  - Preserve useful blocks (products, compositions, rational skeletons) during crossover
  - Reduce product-to-sum collapse and sign-flip degeneration

- [ ] **Adaptive parsimony and bloat control**
  - Dynamic complexity pressure based on generation statistics
  - Monitor complexity distribution drift and prune bloat systematically

- [ ] **Operator and nesting constraints**
  - Add explicit nesting constraints to block pathological expressions
  - Constrain high-risk operator compositions for better search efficiency

- [ ] **Exp-decay robustness**
  - Add targeted handling/tests for exp-decay sign and structure
  - Introduce targeted mutation tests for `exp(-x)` families

#### 2B — Objective-Formula Alignment in Search Loop

- [ ] **Displayed-formula-first ranking**
  - Re-rank hall-of-fame and beam candidates by displayed-formula MSE where available

- [ ] **Drift-aware selection**
  - Penalize large raw/display drift during candidate ranking
  - Add threshold-based rejection for structurally misleading candidates

- [ ] **Exactness policy hardening**
  - Exact status requires displayed-formula criteria, complexity criteria, and domain check

#### 2C — Constant Optimization and Numerical Robustness

- [ ] **Increase constant optimization cadence/restarts**
  - More frequent local constant tuning for top candidates
  - Multiple restarts to avoid local minima in hard tiers

- [ ] **Protected operator policy audit**
  - Verify closure behavior and numerical guards across all operators
  - Reduce hidden domain hacks that hurt symbolic fidelity

- [ ] **Mutation schedule for constants and structure**
  - Tune relative rates for structure vs parameter changes based on failure taxonomy

#### 2D — Decomposition and Grammar-Guided Evolution

- [ ] **Recursive decomposition**
  - Detect `f(g(x))` patterns and decompose search when warranted

- [ ] **Grammar-guided search**
  - Use grammar priors and optional look-ahead search (including MCTS option)

- [ ] **Segmented/orchestrated regression (optional)**
  - Explore local-to-global assembly with overlap and consistency constraints

#### 2E — Baseline Support (Low Priority, Completed)

- [x] Added FFT phase features
- [x] Added derivative smoothing
- [x] Added curvature-aware resampling
- [x] Added PCFG-style generation and better noise robustness

---

### Phase 3: Evolution Intelligence for Complex and Multi-Variate Problems (Months 7-9)

**Goal**: Solve multivariate bottlenecks and hard compositions with evolution-first mechanisms.

#### 3A — Multi-Variate Structure Discovery

- [ ] **AI Feynman separability tests before decomposition**
- [ ] **Symmetry detection (translation/scale/rotation patterns)**
- [ ] **Interaction-aware attention/graph constraints for non-separable functions**

#### 3B — Structural Seeding for Evolution (Not Fast-Path Dependent)

- [ ] **Autoregressive formula proposal model**
  - Generate candidate symbolic structures from data features

- [ ] **Population seeding from proposals (PIGP-style)**
  - Seed top candidate structures into initial population, keep random remainder for diversity

- [ ] **Failure-taxonomy-driven seeding**
  - Convert recurring failure classes into targeted seed templates/guesses

- [ ] **Hybrid proposal verification loop**
  - Proposal model suggests structure, evolution verifies and refines constants/structure

#### 3C — Leakage and Identifiability Hardening

- [ ] **Family-level split validation**
  - Ensure proposal/search gains are not from train/test symbolic overlap

- [ ] **Multi-range and multi-resolution scoring**
  - Require candidates to hold across domain shifts to reduce alias fits

- [ ] **OOD stress test integration in development loop**
  - Add OOD checks before accepting major search changes

---

### Phase 4: Competitive Benchmarking and Paper (Months 10-12)

**Goal**: Rigorous, evolution-centered evaluation and publication.

- [ ] **Standard benchmarks**
  - Nguyen, Keijzer, Feynman suites with strict protocol from Phase 0

- [ ] **Real-world noisy dataset track**
  - Add at least one non-synthetic scientific track to avoid synthetic-only claims

- [ ] **Glassbox vs PySR parity study**
  - Equal budgets, same operators/constraints where possible, multi-seed statistics

- [ ] **Speed vs accuracy Pareto and time-to-discovery plots**
  - Include both final quality and budget efficiency

- [ ] **Ablation studies**
  - Mutation strategy, parsimony schedule, structural seeding, decomposition, drift-aware ranking

- [ ] **OOD generalization test**
  - Strogatz/Black-Box (or equivalent unseen-family) evaluation

- [ ] **Paper draft and submission**
  - NeurIPS/ICML/AAAI style benchmark + methods paper

---

## Risk Gates and Pivot Criteria — NEW

- [ ] **Gate A (end of Phase 0)**
  - No algorithm work accepted without reproducible multi-seed protocol and formula-fidelity scoring

- [ ] **Gate B (end of Phase 2)**
  - Require clear gains in median exact recovery and drift reduction on hard tiers
  - If not achieved: prioritize objective alignment and mutation redesign before new model complexity

- [ ] **Gate C (end of Phase 3)**
  - Require measurable multivariate gains under parity budgets
  - If not achieved: pivot to stronger structural constraints/template-guided evolution for multivariate tasks

- [ ] **Gate D (Phase 4 readiness)**
  - Require stable gains across seeds and at least one real-data track before paper claims

---

## Future / Long-Term (Post-Thesis)

These are higher-impact but more disruptive changes.

- [ ] **Dual-Encoder contrastive architecture (MMSR-style)**
  - Set Transformer/PointNet++ style data encoder plus equation skeleton encoder

- [ ] **LLM-distilled data generation (EQUATE-style)**
  - Domain-specific equation mining with numerical verification

- [ ] **GNN variable interaction priors (EvoNUDGE-style)**
  - Learn variable interaction graph to constrain multivariate search

- [ ] **Fast-path distillation into evolution seeding (optional)**
  - Only if it improves evolution metrics under parity protocol

---

## Honest Limitations to Address in Thesis

1. **Evolution is compute-heavy** compared with baseline heuristics
2. **Hard composition search remains brittle** without strong structural bias
3. **Multi-variate coupling is still the largest unresolved bottleneck**
4. **Train-test symbolic aliasing can inflate apparent performance** without strict audits
5. **Raw-fit vs displayed-formula drift can mislead results** unless enforced in scoring
6. **Synthetic-only evaluations can overstate practical utility**
7. **Fast-path can bias priorities** unless kept explicitly as baseline-only

---

## Success Metrics (Evolution-First)

| Metric | Current | Phase 2 Target | Phase 3 Target | Phase 4 Target |
|--------|---------|----------------|----------------|----------------|
| Nguyen exact recovery (median over seeds) | ~41% | 60%+ | 75%+ | 85%+ |
| Hard tiers (6-8) exact recovery | Low | 20%+ | 35%+ | 50%+ |
| Time to first exact (median, benchmark-defined) | Untracked | Track + improve 25% | Improve 40% | Improve 50% |
| Formula-fidelity drift rate (high-drift rows) | High | Reduce by 50% | Reduce by 70% | Reduce by 85% |
| Stability (IQR/std of exact recovery) | Untracked | Track + reduce variance | 30% variance reduction | 50% variance reduction |
| PySR parity score under equal budget | Below parity | Close gap significantly | Reach parity on selected suites | Beat parity on at least one major suite |
| Multi-variate accuracy | Poor | Poor-to-moderate | 70%+ | 80%+ |
| OOD generalization (Strogatz/Black-Box) | Untested | Baseline run complete | 50%+ | 65%+ |
| Fast-path baseline speed (reference only) | Strong | Maintain | Maintain | Maintain |

---

## Architecture Critique Coverage Map

Every change from [SR_architecture.md](SR_architecture.md) is accounted for:

| Critique Section | Recommendation | Roadmap Location |
|---|---|---|
| §1.1.1 FFT phase erasure | Add phase features | Phase 2E |
| §1.1.2 Derivative instability | Savitzky-Golay smoothing | Phase 2E |
| §1.1.3 Fixed resampling | Adaptive curvature resampling | Phase 2E |
| §1.2 E2E representation learning | Set Transformer / PointNet | Future |
| §1.3 PointNet encoder | PointNet++ data encoder | Future |
| §2.1 Template bias | PCFG formula generation | Phase 2E |
| §2.2 PCFG data generation | Recursive grammar rules | Phase 2E |
| §2.3 LLM distillation (EQUATE) | LLM-generated equations | Future |
| §2.4 OOD generalization | Strogatz/Black-Box benchmarks | Phase 3C + Phase 4 |
| §3.1 1D slicing failure modes | Separability pre-check | Phase 3A |
| §3.2 AI Feynman symmetry detection | Symmetry tests | Phase 3A |
| §3.3 GNN interaction (EvoNUDGE) | GNN variable interaction | Future |
| §3.3 Attention interaction map | Interaction-aware attention | Phase 3A |
| §3.4 Interpolation artifacts | Interaction-aware alternatives | Phase 3A |
| §4.1 Logit biasing weakness | Structural seeding for evolution | Phase 3B |
| §4.2 PIGP structural seeding | Autoregressive generator + population seeding | Phase 3B |
| §4.3 StruSR guided mutation | Enhanced gradient-informed mutation | Phase 2A |
| §4.4 LogicSR MCTS | MCTS option in grammar-guided search | Phase 2D |

---

## Added 2026 Evolution-Focus Coverage Map

| Added Requirement | Roadmap Location |
|---|---|
| Benchmark integrity and fairness protocol | Phase 0 |
| Objective-formula alignment | Phase 0 + Phase 2B |
| Stability and variance reporting | Phase 0 + Phase 4 |
| Time-to-discovery metrics | Phase 0 + Phase 4 |
| Failure taxonomy loop | Phase 0 + Phase 2D + Phase 3B |
| Data leakage/identifiability checks | Phase 0 + Phase 3C |
| Real-data evaluation track | Phase 4 |
| Risk gates and pivot criteria | Risk Gates section |
