lets # TODO

## Fixed

- [x] Fix entropy sign: ONNLoss currently subtracts entropy, rewarding softer gates; flip sign or remove negation so entropy is penalized [glassbox/sr/operation_dag.py#L607-L626](glassbox/sr/operation_dag.py#L607-L626).
- [x] Fix OperationDAGSimple stacking: layer stacks `(output, info)` tuples; take tensor element before stacking [glassbox/sr/operation_dag.py#L553-L563](glassbox/sr/operation_dag.py#L553-L563).
- [x] Preserve dtype in OperationNode zero tensors: initialize zeros with `dtype=sources.dtype` to avoid silent casts [glassbox/sr/operation_node.py#L166-L185](glassbox/sr/operation_node.py#L166-L185).
- [x] Align compile path with BatchNorm: forward_compiled skips output_norm, causing inference drift; apply same normalization or disable compile path [glassbox/sr/operation_dag.py#L430-L489](glassbox/sr/operation_dag.py#L430-L489).
- [x] HardConcreteSelector normalization: hard-training path normalizes gates to sum to 1, blocking all-off sparsity; added `normalize_gates` flag to control behavior [glassbox/sr/hard_concrete.py#L361-L392](glassbox/sr/hard_concrete.py#L361-L392).
- [x] refine_constants scales_only filter: name check uses "edge_weight" (singular) so edge weights are skipped; match actual parameter name "edge_weights" [glassbox/sr/evolution.py#L423-L432](glassbox/sr/evolution.py#L423-L432).
- [x] HardConcreteGate L0 gradient: L0 uses float tau even when learnable, so log_tau gets no gradient; use tau_tensor in the L0 term [glassbox/sr/hard_concrete.py#L187-L204](glassbox/sr/hard_concrete.py#L187-L204).
- [x] MetaArithmetic gradient plateau: Already fixed - uses sigmoid parameterization instead of linear β interpolation [glassbox/sr/meta_ops.py#L180-L250](glassbox/sr/meta_ops.py#L180-L250).

## Remaining (Feature Requests from Research)

### Tier 1 - High Priority

- [x] **Zero-One Loss (FairDARTS)**: Add L_{0-1} = -Σ(σ(α) - 0.5)² to force architecture weights toward binary 0/1. Recommended weight ~10. This closes the discretization gap by soft-discretizing during training [research3.md Section 2.3.2].
  - **Implemented**: Added `zero_one_loss()` method to `HardConcreteOperationSelector`, propagated through `OperationNode` → `OperationLayer` → `OperationDAG`. Added `lambda_zero_one` parameter to `ONNLoss`.
  - **Usage**: Set `lambda_zero_one=10.0` in `ONNLoss` to enable.

- [x] **Gate Regularization**: Add L_gate = Σ g_i(1-g_i) or Σ min(g_i, 1-g_i) to force gates to be binary. Without this, gates converge to 0.5 (ambiguous half-add/half-multiply) [research3.md Section 5.4].
  - **Implemented**: Added `gate_regularization()` method to `HardConcreteOperationSelector`, propagated through `OperationNode` → `OperationLayer` → `OperationDAG`. Added `lambda_gate` parameter to `ONNLoss`.
  - **Usage**: Set `lambda_gate=1.0` in `ONNLoss` to enable. Alternative to zero_one_loss.

- [x] **Gradient Starvation Prevention**: Add LayerNorm or spectral normalization *inside* operation branches before combining. Multiplicative gradients scale with 1/x (explode near zero, vanish for large x) while additive gradients are constant. This imbalance causes the additive path to dominate [research3.md Section 4.2].
  - **Implemented**: Added `unary_norm` and `binary_norm` (`nn.LayerNorm`) inside `OperationNode`. Applied to unary and binary branch outputs before weighted combination.
  - **Active by default**: No configuration needed.

### Tier 2 - Medium Priority

- [ ] **Beta-Decay Regularization**: Penalize magnitude/variance of architecture params α to prevent any single operation from dominating too quickly. Acts as "speed limit" on entropy collapse [research3.md Section 2.3.3].

- [ ] **torch.compile compatibility**: Replace `if compute_unary:` / `if compute_binary:` branching with `torch.where()` masking to avoid graph breaks. Current data-dependent control flow prevents kernel fusion [research3.md Section 6.2.1, operation_node.py#L175-L184].

- [ ] **Risk-Seeking Policy Gradient**: For evolutionary search, optimize top-k percentile rather than expected reward. In symbolic regression we only care about the single best formula, not average performance [research3.md Section 5.1].

### Tier 3 - Already Implemented (verify wiring)

- [~] Hoyer regularization: Implemented in `evolution.py` but verify it's used in training loops.
- [~] Entropy annealing: `anneal_entropy_weight()` exists but verify consistent application.
- [~] Progressive rounding: `soft_round()` and `progressive_round_loss()` exist but verify usage.
- [~] Lamarckian inheritance: `mutate_operations_lamarckian()` exists.
- [~] Gradient-informed mutations: `mutate_operations_gradient_informed()` exists.
- [~] Complexity penalty (BIC-style): `calculate_complexity()` exists.

## Notes from Research Papers

### Key Warnings

1. **MetaExp/MetaLog instability**: These ops frequently cause overflow/underflow. Consider removing from default op library or adding stronger clamping. Paper recommends monitoring which ops consistently fail and pruning them [research2.md Section 3].

2. **Discretization shock**: The snap_to_discrete() approach causes performance collapse. Prefer progressive soft-rounding over hard snapping [research2.md Tier 1].

3. **Recursive structures**: If using tree-based parsers, rewrite recursive traversals as iterative loops with explicit stacks for torch.compile compatibility [research3.md Section 6.2.2].

4. **Gradient variance**: Gumbel-Softmax provides low-variance pathwise gradients; if routing variance is problematic, ensure Straight-Through estimator is properly implemented [research2.md Section 2].
