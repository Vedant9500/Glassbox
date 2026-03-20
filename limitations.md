 Main takeaway: evolution is struggling mostly with structural correctness, not raw numeric fit.

| Metric | v3.1_w_simplification | v1_pure_evolution |
|---|---:|---:|
| Exact | 121 | 85 |
| Approx | 84 | 68 |
| Loose | 0 | 6 |
| Fail | 0 | 46 |
| Exact rate | 59% | 41% |
| Runtime | 1967s | 2764s |

### Where evolution is struggling
1. Hard compositional classes: biggest collapse is Tier 6-8 and Tier 2.
2. Exponential decay forms: many exp(-x) targets become -exp(x) or similar wrong-sign surrogates.
3. Product structure: targets like x*sin(x) collapse into additive forms.
4. High-order polynomial structure: missing highest-degree terms in several Tier 2 failures.
5. Structural drift: I found 42 rows in v1 with very large mismatch between score-MSE and raw-MSE (ratio > 1000), concentrated in Tier 2 and Tier 7.

### Why this is happening
1. These two reports are not apples-to-apples.
- v1 is evolution-only mode (from filename and your terminal command), while v3.1 appears from an older path.
- v1 uses score-MSE that prefers displayed-formula MSE when available (benchmark_suite.py, benchmark_suite.py, benchmark_suite.py).
- v3.1 report format lacks raw/display split, so many old “exact” entries were likely optimistic.

2. Soft internal operators vs hard printed formula mismatch.
- C++ evaluation uses soft blends for arithmetic ops (eval.h).
- Printed formula snaps that soft op to one nearest discrete op (eval.h).
- Aggregation is displayed as simple average (eval.h).
- This is a direct source of “low raw MSE, wrong displayed expression”.

3. Operator/domain choices encourage surrogate expressions.
- Log is exported as log(abs(x)) (eval.h), which can distort symbolic meaning.
- Snapping accepts up to 1.5x degradation during parameter snap in places (evolution.h), which can keep prettier but less faithful forms.

4. Beam search currently optimizes on raw best_mse and does not pass some advanced engine options.
- Beam candidate score uses result best_mse from C++ (classifier_fast_path.py).
- C++ engine supports NSGA-II/islands/other knobs (core.cpp), but beam call currently passes a narrower set.

### Research-backed good practices you are missing or underusing
- SRBench paper: best methods pair genetic search with strong parameter estimation and semantic guidance.  
  https://arxiv.org/abs/2107.14351
- Deep Symbolic Regression: risk-seeking objective helps top-percentile discovery.  
  https://arxiv.org/abs/1912.04871
- AI Feynman 2.0: exploit modularity/symmetry and Pareto tradeoff explicitly.  
  https://arxiv.org/abs/2006.10782
- SymbolicRegression.jl/PySR docs: strong practice set includes nested constraints, adaptive parsimony, frequent constant optimization restarts, dimensional constraints, seeding guesses, Pareto-front selection.  
  https://ai.damtp.cam.ac.uk/symbolicregression/dev/api  
  https://arxiv.org/abs/2305.01582
- gplearn docs: bloat control, protected operators, tournament pressure tuning, ramped half-and-half init are core GP hygiene.  
  https://gplearn.readthedocs.io/en/stable/intro.html

### Practical fix plan (priority order)
1. Make final selection formula-honest.
- Re-rank final candidates by displayed-formula MSE, not only raw engine MSE.
- Add a drift penalty term for raw-vs-display mismatch when choosing winner.

2. Reduce soft-to-hard operator drift.
- Add an entropy penalty on soft arithmetic weights to force near one-hot ops before export.
- Tighten snapping acceptance thresholds (for example from 1.5x toward 1.02-1.05x).

3. Fix exp(-x) failure mode.
- Add explicit neg unary operator (or explicit exp_decay primitive).
- Add hints/priors for decay detection and inner-sign structure (not just exp presence).

4. Increase structural constraints.
- Add nested operator constraints (for example banning pathological nesting).
- Restrict log behavior by domain instead of always defaulting to abs-based logs.

5. Enable stronger multi-objective search.
- Use NSGA-II mode and/or islands in beam runs where complexity-vs-loss tradeoff is hard.

6. Improve benchmark rigor for exact recovery.
- Keep multiple domains and an out-of-domain holdout for scoring.
- Add a structure-sensitive metric (or symbolic-equivalence checks where feasible).

7. Use seeding and template constraints on known hard families.
- For damped oscillations, rationals, and nested trig-exp forms, seed templates/guesses.

If you want, I can implement the top 3 changes directly:
1. formula-honest beam reranking,
2. tighter snapping thresholds,
3. explicit neg/decay operator path.