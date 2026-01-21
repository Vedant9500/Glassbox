# Operation-Based Neural Networks: Key Insights & Actionable Recommendations

> **Source:** `docs/research.md` - Comprehensive research synthesis  
> **Last Updated:** 2026-01-21

---

## 1. Core Problem Definition

### What We're Building
- **Edges** = carry VALUES (constants like 0.5, 1, 2)
- **Neurons** = ARE OPERATIONS (sin, +, *, ^2)
- **Goal**: Learn WHICH operation each neuron should be

### Why Current Approaches Fail
| Problem | Why It's Hard |
|---------|---------------|
| Backprop assumes continuous params | Operation selection is DISCRETE |
| Loss landscape is discontinuous | sin→cos = jump, not gradient |
| Gumbel-Softmax is a hack | Soft≠Hard: training behavior ≠ inference |
| RNN imposes sequential bias | Formulas are HIERARCHICAL (trees/DAGs) |

---

## 2. The "EmbedGrad-Evolution" Framework (Key Solution)

### Architecture Overview
```
┌─────────────────────────────────────────────────────────────────┐
│                    EMBEDGRAD FRAMEWORK                          │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1: Meta-Operations (Continuous Parametric Ops)           │
│  ├── MetaPeriodic(x; ω, φ, A) = A·sin(ωx + φ)                  │
│  ├── MetaPower(x; p) = sign(x)·|x|^p                           │
│  └── MetaArithmetic(x,y; β) = interpolate(+, ×)                │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: Differentiable Routing                                │
│  └── Routing Matrix R ∈ ℝ^(Amax × |V_prev|) for input slots    │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3: Hybrid Optimization                                   │
│  ├── Evolution: Global topology search (which ops)              │
│  └── L-BFGS: Local constant fitting (edge values)               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Actionable Technical Solutions

### 3.1 Replace Discrete Ops with Meta-Operations

**Problem:** 29 discrete operations = combinatorial explosion

**Solution:** 3-4 parametric meta-operations that INCLUDE all discrete ops

| Meta-Operation | Formula | Recovers |
|----------------|---------|----------|
| `MetaPeriodic` | `A·sin(ωx + φ)` | sin (φ=0), cos (φ=π/2), linear (ω→0) |
| `MetaPower` | `sign(x)·|x|^p` | identity (p=1), square (p=2), sqrt (p=0.5), inv (p=-1) |
| `MetaArithmetic` | `(2-β)(x+y) + (β-1)(xy)` | add (β=1), multiply (β=2) |
| `MetaMax/Min` | Softmax approximation | max, min |

**Implementation:**
```python
class MetaPeriodic(nn.Module):
    def __init__(self):
        self.omega = nn.Parameter(torch.tensor(1.0))  # frequency
        self.phi = nn.Parameter(torch.tensor(0.0))    # phase
        self.amplitude = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x):
        return self.amplitude * torch.sin(self.omega * x + self.phi)

class MetaPower(nn.Module):
    def __init__(self):
        self.p = nn.Parameter(torch.tensor(1.0))  # exponent
    
    def forward(self, x):
        return torch.sign(x) * torch.pow(torch.abs(x) + 1e-6, self.p)
```

**Post-training:** Snap parameters to nearest integer/known constant

---

### 3.2 Implement Differentiable Routing

**Problem:** Variable arity - how does a binary op know which 2 inputs to use?

**Solution:** Routing Matrix with soft input slot selection

```python
class RoutingLayer(nn.Module):
    def __init__(self, n_inputs, max_arity=2):
        # R[slot, input] = logit for connecting input to slot
        self.R = nn.Parameter(torch.zeros(max_arity, n_inputs))
    
    def forward(self, inputs):  # inputs: list of tensors
        # Soft selection of which input goes to which slot
        probs = F.softmax(self.R, dim=-1)  # (max_arity, n_inputs)
        inputs_stacked = torch.stack(inputs, dim=-1)  # (batch, n_inputs)
        slots = torch.einsum('si,bi->bs', probs, inputs_stacked)  # (batch, max_arity)
        return slots[:, 0], slots[:, 1]  # slot1, slot2
```

---

### 3.3 Use Hard Concrete Distribution (Not Gumbel-Softmax)

**Problem:** Gumbel-Softmax never produces exact 0 or 1

**Solution:** Hard Concrete stretches to [-β, 1+β] then clips

```python
def hard_concrete_sample(logits, tau=0.5, beta=0.1):
    # Sample from stretched Gumbel
    u = torch.rand_like(logits)
    gumbel = torch.log(u + 1e-10) - torch.log(1 - u + 1e-10)
    s = torch.sigmoid((logits + gumbel) / tau)
    
    # Stretch to [-beta, 1+beta]
    s_stretched = s * (1 + 2*beta) - beta
    
    # Hard clip to [0, 1]
    return torch.clamp(s_stretched, 0, 1)
```

**Why it helps:** Network experiences ACTUAL pruning (z=0) during training

---

### 3.4 Apply BatchNorm After Each Operation

**Problem:** Operations have vastly different gradient magnitudes
- identity: gradient = 1
- exp(x): gradient = e^x (can explode!)
- sin(x): gradient ∈ [-1, 1]

**Solution:** Normalize each operation's output before mixing

```python
def supernode_forward(self, x):
    outputs = []
    for op in self.operations:
        out = op(x)
        out = self.batch_norm[op.name](out)  # Normalize!
        outputs.append(out)
    
    # Now fair mixing
    weights = F.softmax(self.alpha, dim=-1)
    return sum(w * o for w, o in zip(weights, outputs))
```

---

### 3.5 Hybrid Optimization: Evolution + L-BFGS

**Problem:** Gradient descent is local; gets stuck in basins

**Solution:** Two-level optimization

```
OUTER LOOP (Evolution):
  For each individual in population:
    1. Freeze topology (which operations)
    2. INNER LOOP: Optimize constants with L-BFGS
    3. Evaluate fitness = -loss + complexity_penalty
  Selection, Mutation, Crossover
  Repeat
```

**Key Insight from Research:** L-BFGS >> Adam for constant fitting in symbolic regression

```python
# Use scipy's L-BFGS or PyTorch LBFGS
optimizer = torch.optim.LBFGS(
    [edge_constants],
    lr=1.0,
    max_iter=20,
    line_search_fn='strong_wolfe'
)
```

---

### 3.6 Use DAG Architecture (Not RNN)

**Problem:** RNN assumes sequential time dependencies; formulas are hierarchical

**Solution:** Cell-based DAG or Cartesian Genetic Programming grid

```
Cartesian GP Grid:
┌─────┬─────┬─────┬─────┐
│ x1  │ [+] │ [*] │ out │
├─────┼─────┼─────┼─────┤
│ x2  │ [sin]│ [^2]│     │
└─────┴─────┴─────┴─────┘
```

**Library to Explore:** dCGP (Differentiable Cartesian Genetic Programming)
- GitHub: https://github.com/darioizzo/dcgp

---

## 4. Loss Function Design

**Complete Loss:**
```
L = L_mse + λ1·L1(c) + λ2·ΣH(p_i) + λ3·Complexity(G)
```

| Term | Purpose |
|------|---------|
| `L_mse` | Fit the data |
| `L1(c)` | Sparsity on edge constants (prune weak edges) |
| `ΣH(p_i)` | Entropy → force discrete operation selection |
| `Complexity(G)` | Parsimony → prefer simpler graphs |

**Annealing Strategy:**
- Start: High τ (exploration), λ2=0 (allow soft mixing)
- End: Low τ (exploitation), λ2 high (force discrete choices)

---

## 5. Research Questions Answered

| Question | Answer from Research |
|----------|---------------------|
| Q1: Gradient analog in op space? | Use gradients in **probability space** (∇α) or **parameter space** (∇ω,φ) |
| Q2: Right loss landscape? | **Smoothed** via meta-ops + entropy regularization |
| Q3: Right architecture? | **DAG/Grid** (not RNN) - use Cartesian GP or NASNet-like cells |
| Q4: Compositionality? | **Depth** + **Progressive Growth** (add layers when loss plateaus) |
| Q5: Edge values interaction? | Edge c acts as **scaling factor** (c→0 = pruned edge) |

---

## 6. Comparison with Alternatives

| Aspect | Our ONN | KAN | Traditional MLP |
|--------|---------|-----|-----------------|
| Basic unit | Node = Operation | Edge = Spline | Neuron = σ(Wx+b) |
| What's learned | Topology + Ops + Constants | Spline shapes | Only weights |
| Binary ops (x*y) | **Direct** support | Must use exp(log x + log y) | Implicit in weights |
| Interpretability | **High** (readable formula) | Medium (spline shapes) | Low (black box) |
| Optimization | Hybrid (Evo + Grad) | SGD | SGD |

**Key advantage over KAN:** We can directly represent multiplication, which KANs struggle with.

---

## 7. Implementation Priority

### Phase 1: Foundation (Current → Next)
- [x] OperationCell with Gumbel-Softmax
- [x] Basic aggregation operations  
- [ ] **Replace with Meta-Operations** (MetaPeriodic, MetaPower, MetaArithmetic)
- [ ] **Add Differentiable Routing Matrix**

### Phase 2: Optimization
- [ ] Implement Hard Concrete distribution
- [ ] Add BatchNorm after each meta-operation
- [ ] Switch from RNN to DAG structure
- [ ] Implement L-BFGS for constant optimization

### Phase 3: Hybrid Search
- [ ] Wrap in evolutionary outer loop
- [ ] Gradient-guided mutation
- [ ] Progressive depth growth

### Phase 4: Validation
- [ ] Benchmark on standard symbolic regression problems
- [ ] Compare with PySR, dCGP, KAN

---

## 8. Key Papers to Cite

| Topic | Paper/Resource |
|-------|----------------|
| DARTS | "Differentiable Architecture Search" (ICLR 2019) |
| Hard Concrete | "Concrete Distribution" (BDL 2016) |
| NALU | "Neural Arithmetic Logic Unit" (NeurIPS 2018) |
| dCGP | github.com/darioizzo/dcgp |
| KAN | "Kolmogorov-Arnold Networks" (arXiv 2024) |
| L-BFGS for SR | "Benchmarking constant optimization" (arXiv 2024) |

---

## 9. Quick Reference: Code Snippets to Implement

### Meta-Operations
```python
# MetaPeriodic: sin, cos, linear - all in one
class MetaPeriodic(nn.Module):
    def __init__(self):
        self.omega = nn.Parameter(torch.ones(1))
        self.phi = nn.Parameter(torch.zeros(1))
        self.A = nn.Parameter(torch.ones(1))
    def forward(self, x):
        return self.A * torch.sin(self.omega * x + self.phi)

# MetaPower: identity, square, sqrt, cube, inv - all in one
class MetaPower(nn.Module):
    def __init__(self):
        self.p = nn.Parameter(torch.ones(1))
    def forward(self, x):
        return torch.sign(x) * torch.pow(torch.abs(x) + 1e-6, self.p)

# MetaArithmetic: interpolate between + and *
class MetaArithmetic(nn.Module):
    def __init__(self):
        self.beta = nn.Parameter(torch.ones(1))  # 1=add, 2=mul
    def forward(self, x, y):
        return (2 - self.beta) * (x + y) + (self.beta - 1) * (x * y)
```

### Routing Matrix
```python
class DifferentiableRouter(nn.Module):
    def __init__(self, n_sources, n_slots=2):
        self.R = nn.Parameter(torch.zeros(n_slots, n_sources))
    
    def forward(self, sources):
        # sources: (batch, n_sources)
        probs = F.softmax(self.R, dim=-1)  # (n_slots, n_sources)
        return sources @ probs.T  # (batch, n_slots)
```

### Hard Concrete
```python
def hard_concrete(logits, tau=0.5, beta=0.1, hard=True):
    if self.training:
        u = torch.rand_like(logits).clamp(1e-6, 1-1e-6)
        s = torch.sigmoid((torch.log(u/(1-u)) + logits) / tau)
        s = s * (1 + 2*beta) - beta
        z = s.clamp(0, 1)
    else:
        z = torch.sigmoid(logits)
    return z
```

---

*This document is a living reference. Update as implementation progresses.*
