# Blackbox Optimization Plan

## Why Blackbox Problems Are Hard Here

Glassbox currently performs best on clean, low-dimensional symbolic problems. Blackbox SRBench/PMLB-style problems are harder because they often have:

- multiple input features,
- irrelevant or redundant columns,
- noisy empirical targets,
- unknown or non-symbolic ground truth,
- train/test generalization requirements,
- feature scale differences,
- weak interactions that are hard to discover by random expression search.

The current project architecture also has a specific bottleneck:

- `GlassboxRegressor._run_universal_proposer_dual_path` skips multivariate inputs.
- Guided beam-search evolution is currently guarded to single-feature inputs.
- The C++ fallback can accept multiple features, but it receives the full feature set and only limited blackbox-specific guidance.

So blackbox failures are not only because data is multivariate, but multivariate search-space explosion is the first major problem to solve.

## Research Notes

The broader SR literature points in the same direction:

- SRBench separates ground-truth symbolic problems from blackbox regression problems and uses PMLB datasets for blackbox evaluation. This means blackbox should be treated as predictive symbolic approximation, not exact rediscovery.  
  Source: https://cavalab.org/srbench/

- High-dimensional symbolic regression suffers because many features increase search complexity and make GP-style methods overfit or fail to identify important variables. Recent work explicitly uses feature selection to improve high-dimensional SR generalization.  
  Source: https://link.springer.com/article/10.1007/s41019-024-00270-x

- Other high-dimensional SR approaches also combine symbolic search with feature selection or multiple sub-searches to reduce the effective problem size.  
  Source: https://www.sciencedirect.com/science/article/pii/S1568494619301322

- AI Feynman-style methods show the value of decomposing problems using separability/modularity before running lower-dimensional symbolic regression.  
  Source: https://arxiv.org/abs/2006.10782

Takeaway: blackbox optimization should begin with feature-space reduction, interaction discovery, and staged decomposition before expensive symbolic evolution.

## Target Architecture

Add a `blackbox_mode` path inside `GlassboxRegressor`.

The pipeline should be:

1. Preprocess data.
2. Rank and select active features.
3. Detect simple univariate structure per active feature.
4. Detect pairwise/triple interactions.
5. Build candidate symbolic seeds and operator hints.
6. Run C++ evolution on a reduced feature matrix.
7. Validate on holdout/CV.
8. Optionally add residual symbolic stages.
9. Export the formula mapped back to original feature names/indices.

## Phase 1: Blackbox Preprocessor

Add a local module, likely:

```text
glassbox/sr/blackbox_preprocessor.py
```

Responsibilities:

- remove constant and near-constant features,
- impute or reject non-finite rows cleanly,
- standardize `X` and `y` for search,
- keep inverse-transform metadata,
- map selected reduced feature indices back to original feature indices,
- store diagnostics in `est.blackbox_diagnostics_`.

Estimator parameters to add:

- `blackbox_mode="auto"` or `True/False`,
- `blackbox_max_features=6`,
- `blackbox_feature_selection=True`,
- `blackbox_standardize=True`,
- `blackbox_interaction_search=True`.

Auto-enable blackbox mode when:

- `n_features_in_ > 1`, or
- target noise/holdout residual appears high, or
- dataset is passed through SRBench Track 1.

## Phase 2: Active Feature Ranking

Implement a lightweight ensemble ranker:

- absolute Pearson/Spearman correlation,
- mutual information regression if sklearn is available,
- Lasso/ElasticNet coefficients on standardized data,
- ExtraTrees or RandomForest permutation/impurity importance,
- single-feature fast-path score where cheap enough.

Output:

```python
{
    "selected_features": [0, 3, 7],
    "feature_scores": {...},
    "ranker_votes": {...},
    "dropped_features": [...],
}
```

Use top `k` features for expensive evolution, default `k <= 6`.

Important: keep a fallback where all features are retained if selection is uncertain and `n_features <= 4`.

## Phase 3: Interaction Discovery

After selecting top features, test cheap pairwise candidates:

- `xi + xj`,
- `xi * xj`,
- `xi / (xj + eps)`,
- `xi - xj`,
- `xi^2 + xj^2`,
- simple products with `sin`, `cos`, `exp`, `log` where operator priors suggest them.

Score each candidate using cross-validated or holdout MSE improvement over univariate fits.

Output:

```python
{
    "interaction_pairs": [(0, 3), (1, 4)],
    "interaction_terms": ["x0*x3", "x1/x4"],
    "interaction_scores": {...},
}
```

These should feed:

- `operator_hints`,
- `candidate_formulas`,
- seed graphs,
- C++ evolution `X_list` feature subset.

## Phase 4: Multivariate Seed Graphs

Current `seed_graph_builder.py` is mostly univariate:

- input nodes default to `feature_idx=0`,
- signal-discovered formulas use `x`,
- formula-to-graph needs reliable handling of `x0`, `x1`, etc.

Needed improvements:

- build seeds from multivariate formulas like `x0*x1`, `x0 + sin(x2)`,
- add feature-index-aware signal seeds,
- build pairwise interaction seeds from Phase 3,
- cap seed count based on the proposer/search plan.

This is one of the most important blackbox upgrades because good seeds reduce blind search.

## Phase 5: Blackbox Search Plan

Extend the universal proposer/search planner for multivariate data.

For now, use heuristics:

- increase population/generations with selected feature count,
- increase breadth when feature selection uncertainty is high,
- increase depth/complexity only when pairwise interactions help validation,
- restrict operator families to those supported by feature-wise diagnostics.

Future trained planner heads:

- active feature count,
- feature subset probabilities,
- interaction probabilities,
- operator family per feature,
- recommended population multiplier,
- generation multiplier,
- max nodes/tree complexity,
- exploration vs refinement mode.

## Phase 6: Residual Staged Symbolic Additive Model

Instead of one monolithic expression, fit:

```text
f(x) = f1(selected_features) + f2(residual_features) + ...
```

Workflow:

1. Fit simplest symbolic expression.
2. Evaluate validation residual.
3. If residual has structure, fit another small symbolic term.
4. Keep the term only if validation improves.
5. Stop when residual improvement is small or complexity cap is reached.

This is especially useful for blackbox approximation, where exact compact formulas may not exist.

## Phase 7: Validation and Metrics

Blackbox should be optimized with different metrics from exact symbolic recovery:

- validation/test R2,
- median R2 across seeds,
- worst-decile R2,
- formula size,
- selected feature count,
- prediction stability,
- domain failure rate,
- time-to-acceptable model,
- displayed-formula MSE/R2, not raw engine-only score.

For Track 1 SRBench, exact recovery should not be the headline metric.

## Integration Points

### `glassbox/sr/sklearn_wrapper.py`

Add blackbox preprocessing before fast-path/evolution:

```python
X_search, y_search, blackbox_state = prepare_blackbox_search(X, y, ...)
```

Then use `X_search` for evolution and map final formulas back to original features.

### `scripts/run_srbench_local.py`

Add flags:

- `--blackbox-mode`,
- `--blackbox-max-features`,
- `--no-blackbox-feature-selection`,
- `--blackbox-interactions`.

Track 1 should enable blackbox mode by default.

### `glassbox/universal_proposer`

Current proposer is univariate. Keep it for 1D, but add a separate multivariate planner path:

```text
MultivariateSearchPlanner
```

This can be heuristic first, trained later.

### `glassbox/sr/cpp/seed_graph_builder.py`

Add multivariate candidate parsing and seed graph generation.

## Implementation Order

1. Add blackbox preprocessor and feature ranker.
2. Wire selected feature subset into `GlassboxRegressor`.
3. Add formula remapping from reduced `x0..xk` back to original `xj`.
4. Add interaction diagnostics and candidate formula seeds.
5. Improve multivariate seed graph builder.
6. Add blackbox mode flags to SRBench runner.
7. Add residual staged additive fitting.
8. Train multivariate proposer/planner after heuristic path proves useful.

## First Milestone

Goal: make Track 1 blackbox less fragile without touching core C++.

Deliverables:

- `blackbox_preprocessor.py`,
- top-k feature selection,
- reduced `X` evolution,
- formula index remapping,
- SRBench Track 1 diagnostics,
- ablation: all features vs selected features.

Success criteria:

- improved median Track 1 R2,
- fewer timeouts,
- smaller formulas,
- stable or improved worst-decile R2,
- no regression on Track 2 single-feature symbolic problems.

