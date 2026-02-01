# Implementation Plan: Reducing Approximation in Symbolic Regression

## Goal
Reduce “close but wrong” approximations by adding operator constraints, nesting rules, domain-aware sampling, precision controls, explicit inverse operators, and early-stop checks. This plan targets difficult formulas such as Planck’s Law, Relativistic Mass, and Nesting Doll.

---

## 1) Operator Constraints by Formula
**What**: Enable/disable operator families per formula (e.g., block periodic ops for Planck’s Law).

**Where**:
- [scripts/sr_tester.py](../scripts/sr_tester.py) — add CLI flags and config fields
- [glassbox/sr/operation_node.py](../glassbox/sr/operation_node.py) — apply op gating to selectors
- [scripts/verify_fast_path.py](../scripts/verify_fast_path.py) — pass per‑benchmark flags

**How**:
- Add CLI flags: `--ops-periodic`, `--ops-exp`, `--ops-log`, `--ops-power`, `--ops-agg` (default all enabled).
- In `OperationNode`, mask selector logits for disallowed ops (set logits to large negative values before sampling).
- For formula‑specific benchmarks, pass appropriate flags (e.g., Planck: allow exp, disallow periodic).

**Expected Outcome**:
- Prevents the model from “cheating” with unrelated operators.

**Drawbacks**:
- Too strict constraints can block valid solutions.

---

## 2) Nested-Operator Constraints
**What**: Forbid or penalize specific operator nestings (e.g., `exp(sin(x))` unless needed).

**Where**:
- [glassbox/sr/meta_ops.py](../glassbox/sr/meta_ops.py) — nesting checks
- [glassbox/sr/operation_node.py](../glassbox/sr/operation_node.py) — apply penalties in forward

**How**:
- Track last op types in node outputs (metadata in forward pass).
- Apply penalties or gating when disallowed nests are detected.
- Add config in `sr_tester.py` for `--allow-nesting-exp-sin`, etc.

**Expected Outcome**:
- Reduces spurious nested approximations for non‑nested targets.

**Drawbacks**:
- Requires careful tuning for formulas that truly need nesting.

---

## 3) Domain‑Aware Sampling / Weights
**What**: Avoid or down‑weight singular regions (e.g., near `x=±1` for Relativistic Mass).

**Where**:
- [scripts/sr_tester.py](../scripts/sr_tester.py) — data generation
- [glassbox/sr/evolution.py](../glassbox/sr/evolution.py) — weighted loss

**How**:
- Add `--sample-avoid` / `--sample-epsilon` flags.
- Apply a weight function in fitness (e.g., $
  w(x) = \min(1, \frac{|1-x^2|}{\epsilon})
  $) for singular denominators.

**Expected Outcome**:
- Stabilizes training and improves structure discovery on singular formulas.

**Drawbacks**:
- Might underfit near singularities if weights are too low.

---

## 4) Precision + Normalization Controls
**What**: Allow high‑precision evaluation to reduce numeric approximation drift.

**Where**:
- [glassbox/sr/evolution.py](../glassbox/sr/evolution.py)
- [scripts/sr_tester.py](../scripts/sr_tester.py)

**How**:
- Add `--precision 64` and `--normalize-data` flags.
- Use float64 for `x/y` and model eval when precision is 64.

**Expected Outcome**:
- More stable numeric fits, fewer “almost exact” models.

**Drawbacks**:
- Slower and higher memory usage.

---

## 5) Explicit Inverse/Rational Operators
**What**: Add a dedicated `inv(x)` operator for rational recovery.

**Where**:
- [glassbox/sr/meta_ops.py](../glassbox/sr/meta_ops.py)
- [glassbox/sr/operation_node.py](../glassbox/sr/operation_node.py)

**How**:
- Extend `MetaPower` with `p=-1` or add explicit `MetaInv`.
- Add operator to selector and update op listings.

**Expected Outcome**:
- Improves exact recovery of rational formulas (Planck, Relativistic Mass).

**Drawbacks**:
- Increases search space, may need stronger regularization.

---

## 6) Early‑Stop Exact‑Match Checks
**What**: Stop evolution when an exact‑match structure is detected.

**Where**:
- [scripts/classifier_fast_path.py](../scripts/classifier_fast_path.py)
- [glassbox/sr/evolution.py](../glassbox/sr/evolution.py)

**How**:
- Add per‑generation exact‑match checks (MSE threshold + operator compliance).
- Early stop if found.

**Expected Outcome**:
- Faster convergence on exact formulas.

**Drawbacks**:
- Risk of stopping on near‑exact approximations if threshold too loose.

---

## Recommended Order of Implementation
1) Operator constraints by formula
2) Domain‑aware sampling/weights
3) Explicit inv/rational operators
4) Nested-operator constraints
5) Precision/normalization controls
6) Early-stop exact-match checks

---

## Benchmark-Specific Suggestions
- **Planck’s Law**: enable `exp`, `inv`, disable periodic; use domain‑aware sampling near $x=0$ and higher precision.
- **Relativistic Mass**: enable power+inv, disable periodic; avoid $|x|\approx 1$; use weights.
- **Nesting Doll**: allow nested periodic/exp; expect more generations/population.
