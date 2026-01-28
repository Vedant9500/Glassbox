# Code Review: `glassbox/sr/`

**Date**: 2026-01-28

---

## File: `evolution.py` (2556 lines)

---

## Issues Summary

### 🔴 Bugs / Correctness Issues

| # | Issue | Severity | Line(s) | Description |
|---|-------|----------|---------|-------------|
| 1 | Redundant `import math` | Low | 384 | `math` already imported at module level (line 25) |
| 2 | Redundant `import copy` | Low | 2133 | `copy` already imported at module level (line 23) |
| 3 | Redundant `import numpy` | Low | 2047 | Import inside training loop; should be at module level |
| 4 | Inconsistent NaN handling | Medium | 717 vs 1183 | `refine_constants` returns `inf` tensor for NaN, `finalize_model_coefficients` skips backward but returns NaN |
| 5 | Potential RSPG method missing | Medium | 1825 | `should_use_rspg()` may not exist if RSPG unavailable (dummy base class) |
| 6 | Inconsistent closure grad zeroing | Low | 1331-1339 | Manual grad zeroing differs from other closures using `optimizer.zero_grad()` |

---

### 🟡 Performance Issues

| # | Issue | Severity | Line(s) | Description |
|---|-------|----------|---------|-------------|
| 1 | Expensive `deepcopy` for cloning | Medium | 240 | `copy.deepcopy(model)` is slow; consider `state_dict()` pattern |
| 2 | CPU-GPU transfers in loops | Medium | 975, 2047, 2282-2284 | `.cpu()` calls inside loops cause sync overhead |
| 3 | Repeated `named_parameters()` iterations | Low | 1017, 1026, 1046, 1062 | Multiple iterations over same parameters; cache the list |
| 4 | Redundant `model.eval()` calls | Low | 1961, 2042 | Model already in eval mode; unnecessary mode switches |

---

### 🟢 Code Quality Issues

| # | Issue | Severity | Line(s) | Description |
|---|-------|----------|---------|-------------|
| 1 | Magic numbers not constants | Low | 1736, 1798, 2462, 2502 | Values like `4.0`, `3.0`, `1.0` should be named constants |
| 2 | Very long `train()` method | Medium | 1859-2303 | ~450 lines; should be refactored into smaller methods |
| 3 | Missing return type annotation | Low | 1859 | `-> Dict` should be `-> Dict[str, Any]` |
| 4 | Unused variable in branch | Low | 2078-2082 | `n_pruned` from non-adaptive branch checked but may be unset |

---

## Recommended Fixes (Priority Order)

1. **High Priority**: Fix RSPG safety check (add `hasattr` guard)
2. **High Priority**: Standardize NaN handling in closures
3. **Medium Priority**: Replace `deepcopy` with `state_dict()` pattern for performance
4. **Medium Priority**: Move numpy import to module level
5. **Low Priority**: Remove redundant imports (`math`, `copy`)
6. **Low Priority**: Extract `train()` into smaller helper methods

---

## File: `meta_ops.py` (625 lines)

---

### 🔴 Bugs / Correctness Issues

| # | Issue | Severity | Line(s) | Description |
|---|-------|----------|---------|-------------|
| 1 | Division by zero in MetaLog | High | 478-483 | When `log_base=0` (base=1), `log_base_val` is 0 causing division error |
| 2 | Device mismatch in set_tau | Medium | 376 | Buffer reassignment creates CPU tensor, loses device |
| 3 | Unnecessary tensor creation | Low | 434, 488 | `torch.exp(torch.tensor(...))` should use `math.exp()` |
| 4 | Division not implemented | Low | 289-299 | MetaArithmeticExtended claims div support but only approximates |

---

### 🟡 Performance Issues

| # | Issue | Severity | Line(s) | Description |
|---|-------|----------|---------|-------------|
| 1 | All ops computed even when one selected | Medium | 546-551 | 4x wasted compute when `hard=True` - should short-circuit |
| 2 | Stack + weighted sum allocations | Low | 552, 579 | Intermediate tensor allocations; could use accumulator |

---

### 🟢 Code Quality / Suggestions

| # | Issue | Severity | Line(s) | Description |
|---|-------|----------|---------|-------------|
| 1 | Missing snap_to_discrete | Low | MetaExp/MetaLog | These classes lack snap_to_discrete() unlike others |
| 2 | Add explicit Identity op | Medium | N/A | Currently relies on MetaPower(p=1); explicit is faster/cleaner |
| 3 | Lazy evaluation pattern | Medium | 538-562 | When `hard=True`, compute only selected op for 4x speedup |

---

### Recommended Fixes for `meta_ops.py`

1. **High Priority**: Guard MetaLog against base=1 (add eps to log_base_val)
2. **Medium Priority**: Fix set_tau to preserve device: `self.tau.fill_(tau)`
3. **Medium Priority**: Implement lazy evaluation for hard selection mode
4. **Low Priority**: Add snap_to_discrete to MetaExp/MetaLog

---

## File: `risk_seeking_policy_gradient.py` (414 lines)

---

### 🔴 Bugs / Correctness Issues

| # | Issue | Severity | Line(s) | Description |
|---|-------|----------|---------|-------------|
| 1 | RSPG never deactivates | Medium | 113-116 | Once RSPG activates, the `pass` statement means it never deactivates - comment says "stay in RSPG once activated" but no actual cooldown logic |
| 2 | Unused `param_count` variable affects nothing | Low | 265-270 | Variable counted but not used in result calculation |

---

### 🟡 Performance Issues

| # | Issue | Severity | Line(s) | Description |
|---|-------|----------|---------|-------------|
| 1 | Repeated list conversions from deque | Low | 73-74, 90, 129-133 | `list(self.loss_history)` called multiple times; could cache |
| 2 | np.percentile computed each call | Low | 188 | Could cache threshold if fitnesses don't change often |

---

### 🟢 Code Quality / Suggestions

| # | Issue | Severity | Line(s) | Description |
|---|-------|----------|---------|-------------|
| 1 | Class defaults shadow instance attributes | Low | 215-218 | Class-level defaults work but can be confusing; move to `__init__` stub |
| 2 | Test functions included in production code | Low | 344-413 | Consider moving tests to separate test file |
| 3 | Missing type hint for population parameter | Low | 291 | `population: List` should be `population: List[Any]` |
| 4 | Add cooldown logic for RSPG deactivation | Medium | 113-116 | Implement the TODO for deactivation cooldown |

---

### ✅ Well-Designed Aspects

| Aspect | Description |
|--------|-------------|
| Defensive coding | Uses `getattr` with defaults for safe access even if init never called (lines 251, 280, 283) |
| Clear documentation | Excellent docstrings explaining the research insight and usage |
| Modular design | Mixin pattern allows easy integration with existing trainers |
| Gradient monitoring | Smart activation strategy based on stuck/exploding detection |

---

### Recommended Fixes for `risk_seeking_policy_gradient.py`

1. **Medium Priority**: Implement RSPG deactivation cooldown (line 113-116)
2. **Low Priority**: Move test functions to separate test file
3. **Low Priority**: Cache list conversions from deque in hot paths

---

## File: `pruning.py` (765 lines)

---

### 🔴 Bugs / Correctness Issues

| # | Issue | Severity | Line(s) | Description |
|---|-------|----------|---------|-------------|
| 1 | Bare `except` swallows all errors | Medium | 462 | `except:` catches KeyboardInterrupt too; use `except Exception:` |
| 2 | Merge doesn't combine weights | Medium | 476-496 | Comment says "sum up weights" but code just zeroes duplicates without adding to primary |
| 3 | min_importance filter excludes 0 | Low | 553-558 | `0 < score` excludes already-pruned (score=0) but may cause confusion |
| 4 | Redundant state save + restore | Low | 98, 127-130, 177 | Manual restore per node, then full load_state_dict |

---

### 🟡 Performance Issues

| # | Issue | Severity | Line(s) | Description |
|---|-------|----------|---------|-------------|
| 1 | O(N²) complexity in iterative pruning | High | 550, 124 | Fresh sensitivity_analysis (O(N) forward passes) each iteration |
| 2 | Excessive deepcopy of model state | Medium | 98, 657, 670, 678, 690, 698 | Multiple deep copies; expensive for large models |
| 3 | Redundant get_mse() calls | Low | 648-703 | Same MSE recomputed multiple times |

---

### 🟢 Code Quality / Suggestions

| # | Issue | Severity | Line(s) | Description |
|---|-------|----------|---------|-------------|
| 1 | Magic numbers for thresholds | Low | 141, 173, 181-184 | 1.0/0.1/0.01 for status classification should be constants |
| 2 | Inconsistent case ("DEAD" vs "dead") | Low | 141 vs 173 | Same threshold different case |
| 3 | Dead code: weight always 1.0 | Low | 460-461 | Variable set but never computed |
| 4 | Consider gradient-based importance | Medium | 75-187 | Faster than ablation: 1 backward vs N forward passes |
| 5 | Add progress indicator | Low | 108-142 | Long loops have no progress bar |

---

### Recommended Fixes for `pruning.py`

1. **High Priority**: Fix merge logic to actually sum weights from duplicates (line 483-494)
2. **High Priority**: Replace O(N²) sensitivity analysis with cached/incremental approach
3. **Medium Priority**: Change bare `except:` to `except Exception:`
4. **Medium Priority**: Consider gradient-based importance as faster alternative to ablation
5. **Low Priority**: Define constants for status thresholds
