# Operation Neural Network (ONN) - Technical Documentation

## Overview

The Operation Neural Network (ONN) is an experimental approach to **symbolic regression** - the task of discovering mathematical formulas from data. Unlike traditional neural networks that learn numerical weights, ONN learns to select and compose mathematical operations to form interpretable equations.

**Goal:** Given input-output pairs (x, y), discover the symbolic formula f such that y ≈ f(x).

**Example:**
- Input: x = [-3, -2, ..., 2, 3], y = [9, 4, ..., 4, 9]
- Desired output: `y = x²`

---

## Architecture

### High-Level Structure

```
Input (x) → [Layer 1] → [Layer 2] → ... → [Output Projection] → Prediction (ŷ)
              ↓            ↓
           [Node 1]    [Node 1]
           [Node 2]    [Node 2]
           [Node 3]    [Node 3]
              ...         ...
```

The ONN is a **Directed Acyclic Graph (DAG)** where:
- Each **layer** contains multiple **operation nodes**
- Each **node** selects ONE operation from a menu of options
- Nodes can read from inputs OR outputs of previous layers
- Final output is a **learned linear combination** of all node outputs

### Key Components

#### 1. OperationDAG (Main Model)
**File:** `glassbox/sr/operation_dag.py`

```python
OperationDAG(
    n_inputs=1,           # Number of input variables
    n_hidden_layers=1,    # Number of operation layers
    nodes_per_layer=4,    # Nodes per layer
    n_outputs=1,          # Output dimension
    tau=0.5,              # Temperature for selection
)
```

The DAG orchestrates the flow:
1. Collects all available sources (inputs + previous layer outputs)
2. Passes sources through each layer
3. Projects final concatenated features to output

#### 2. OperationLayer
**File:** `glassbox/sr/operation_dag.py`

A container for multiple OperationNodes. Each layer:
- Receives all sources available so far
- Runs each node independently
- Outputs concatenated node results

#### 3. OperationNode (Core Unit)
**File:** `glassbox/sr/operation_node.py`

Each node performs:
1. **Routing:** Select which input sources to use
2. **Operation Selection:** Choose unary OR binary operation
3. **Apply Operation:** Compute result
4. **Scale Output:** Learnable output scale

```
Sources [x, prev_outputs] 
    ↓
[Router] → Select 2 sources (s1, s2)
    ↓
[Operation Selector] → Choose op type (unary/binary)
    ↓
[Apply] → unary(s1) OR binary(s1, s2)
    ↓
[Scale] → output * learned_scale
```

---

## Meta-Operations

Instead of fixed operations like `sin`, `square`, the ONN uses **parametric meta-operations** that can smoothly interpolate between different functions.

**File:** `glassbox/sr/meta_ops.py`

### MetaPower (Unary)
Learnable power function: `f(x) = x^p`

| Parameter p | Function |
|-------------|----------|
| p = 0 | constant 1 |
| p = 0.5 | √x |
| p = 1 | x (identity) |
| p = 2 | x² |
| p = 3 | x³ |
| p = -1 | 1/x |

```python
class MetaPower(nn.Module):
    def __init__(self):
        self.p = nn.Parameter(torch.tensor(1.0))  # Learnable
    
    def forward(self, x):
        return safe_pow(torch.abs(x) + 1e-8, self.p)
```

### MetaPeriodic (Unary)
Learnable periodic function: `f(x) = A * sin(ωx + φ)`

| Parameter φ | Function |
|-------------|----------|
| φ = 0 | sin(x) |
| φ = π/2 | cos(x) |
| φ = π | -sin(x) |

### MetaArithmetic (Binary)
Interpolates between addition and multiplication:

`f(x, y) = (2-β)(x+y) + (β-1)(x*y)`

| Parameter β | Function |
|-------------|----------|
| β = 1 | x + y |
| β = 2 | x * y |
| 1 < β < 2 | weighted mix |

### MetaExp (Unary)
Exponential with learnable rate: `f(x) = exp(k*x)` where k ∈ [-2, 2]

### MetaLog (Unary)  
Logarithm: `f(x) = ln(|x| + ε)`

### MetaAggregation (Aggregation)
Interpolates between sum, mean, max, min using learnable temperature.

---

## Operation Selection (Hard Concrete)

**File:** `glassbox/sr/hard_concrete.py`

The key challenge: How to make discrete selection (choosing ONE operation) differentiable?

### The Problem
Standard argmax is not differentiable:
```python
selected = operations[logits.argmax()]  # No gradients!
```

### The Solution: Hard Concrete Distribution

Hard Concrete is a continuous relaxation that:
1. **Training:** Samples soft weights that are often exactly 0 or 1
2. **Inference:** Uses deterministic argmax

```python
def hard_concrete_sample(logits, tau=0.5, beta=0.1):
    if not training:
        # Deterministic: one-hot argmax
        return one_hot(logits.argmax())
    
    # Sample from stretched concrete distribution
    u = torch.rand_like(logits)
    s = sigmoid((log(u) - log(1-u) + logits) / tau)
    
    # Stretch to allow exact 0s and 1s
    s_stretched = s * (1 + 2*beta) - beta
    z = clamp(s_stretched, 0, 1)
    
    # Straight-through estimator
    z_hard = (z > 0.5).float()
    return z + (z_hard - z).detach()
```

### HardConcreteOperationSelector

Manages selection between:
- Unary vs Binary operation type
- Which specific unary operation
- Which specific binary operation

```python
class HardConcreteOperationSelector:
    def __init__(self):
        self.type_selector = HardConcreteSelector(2)      # [unary, binary]
        self.unary_selector = HardConcreteSelector(4)     # [power, periodic, exp, log]
        self.binary_selector = HardConcreteSelector(2)    # [arithmetic, aggregation]
```

---

## Routing (Input Selection)

**File:** `glassbox/sr/operation_node.py`

Each node must choose which sources to use as inputs. For binary operations, it needs two inputs.

### AdaptiveArityRouter

Uses Gumbel-Softmax for differentiable source selection:

```python
class AdaptiveArityRouter:
    def __init__(self, n_sources, n_selections=2):
        self.logits = nn.Parameter(torch.randn(n_selections, n_sources))
    
    def forward(self, sources, hard=False):
        if hard and not self.training:
            # Deterministic: select top sources
            indices = self.logits.argmax(dim=-1)
            return sources[:, indices]
        else:
            # Soft: weighted combination
            weights = F.softmax(self.logits / tau, dim=-1)
            return weights @ sources
```

### EdgeWeights

Learnable scalar weights for each input source:

```python
class EdgeWeights:
    def __init__(self, n_sources):
        self.weights = nn.Parameter(torch.ones(n_sources))
    
    def forward(self, sources):
        return sources * self.weights  # Element-wise scaling
```

---

## Training: Evolutionary Approach

**File:** `glassbox/sr/evolution.py`

### Why Not Pure Gradient Descent?

The operation selection is fundamentally **discrete** - you either use sin(x) or you don't. Gradient descent works poorly for:
1. Finding the right discrete structure
2. Escaping local minima in the operation space

### Evolutionary Strategy

The training uses a hybrid approach:
1. **Evolution:** Explore different operation structures (which ops to use)
2. **Gradient Descent:** Optimize continuous parameters (scales, power values)

```
┌─────────────────────────────────────────────────────────┐
│                    EVOLUTION LOOP                        │
├─────────────────────────────────────────────────────────┤
│  1. Initialize population of N random models            │
│  2. For each generation:                                │
│     a. Evaluate fitness of all individuals              │
│     b. Select best individuals (elites)                 │
│     c. Create children via mutation                     │
│     d. Refine constants with gradient descent           │
│  3. Return best individual                              │
└─────────────────────────────────────────────────────────┘
```

### Training Steps in Detail

#### Step 1: Population Initialization

```python
for i in range(population_size):
    model = OperationDAG(...)
    random_operation_init(model)  # Random logits for operation selection
    population.append(Individual(model))
```

`random_operation_init()` sets random biases for operation selection logits, encouraging diversity.

#### Step 2: Fitness Evaluation

For each individual:
```python
model.eval()
with torch.no_grad():
    pred, _ = model(x, hard=True)  # Deterministic forward
    mse = F.mse_loss(pred, y)
    
    complexity = calculate_complexity(model)  # Prefer simpler formulas
    entropy = model.entropy_regularization()  # Prefer discrete selections
    
    fitness = mse + 0.001 * complexity + 0.01 * entropy
```

#### Step 3: Selection and Reproduction

```python
# Sort by fitness (lower is better)
sorted_pop = sorted(population, key=lambda ind: ind.fitness)

# Keep elites (top 20%)
elites = sorted_pop[:elite_count]

# Fill rest with mutated children
children = []
for _ in range(population_size - elite_count):
    parent = tournament_select(sorted_pop)
    child = mutate_operations(parent, mutation_rate=0.3)
    children.append(child)

population = elites + children
```

#### Step 4: Mutation

```python
def mutate_operations(individual, mutation_rate=0.3):
    mutant = individual.clone()
    
    for name, param in mutant.model.named_parameters():
        if random.random() < mutation_rate:
            if 'logit' in name:
                # Discrete mutation: shift to different operation
                param.zero_()
                new_choice = random.randint(0, param.numel() - 1)
                param.view(-1)[new_choice] = 3.0
            
            elif 'p' in name:  # Power parameter
                # Small continuous mutation OR jump to common value
                if random.random() < 0.3:
                    param.fill_(random.choice([0.5, 1.0, 2.0, 3.0]))
                else:
                    param.add_(random.uniform(-0.5, 0.5))
    
    return mutant
```

#### Step 5: Constant Refinement

After mutation, fine-tune continuous parameters with gradient descent:

```python
def refine_constants(model, x, y, steps=50, lr=0.01):
    # Only optimize non-selection parameters
    constant_params = [p for name, p in model.named_parameters() 
                       if 'logit' not in name and 'selector' not in name]
    
    optimizer = Adam(constant_params, lr=lr)
    
    for step in range(steps):
        pred, _ = model(x, hard=True)
        loss = F.mse_loss(pred, y)
        loss.backward()
        optimizer.step()
```

#### Step 6: Temperature Annealing

The selection temperature τ decreases over generations:

```python
tau = tau_start * (tau_end / tau_start) ** (gen / generations)
# tau: 1.0 → 0.1 (soft → hard selection)
```

Early: Soft selection allows exploring combinations
Late: Hard selection commits to specific operations

---

## Forward Pass (Inference)

### Standard Forward

```python
def forward(self, x, hard=True):
    all_sources = [x]  # Start with inputs
    
    for layer in self.layers:
        sources = torch.cat(all_sources, dim=-1)
        layer_output, _ = layer(sources, hard=hard)
        all_sources.append(layer_output)
    
    # Project all features to output
    final = torch.cat(all_sources, dim=-1)
    return self.output_proj(final)
```

### Per-Node Forward

```python
def node_forward(self, sources, hard=True):
    # 1. Route: select input sources
    weighted_sources = self.router(sources, hard=hard)
    s1 = weighted_sources[:, 0]  # Primary input
    s2 = weighted_sources[:, 1]  # Secondary input (for binary ops)
    
    # 2. Select operation type and compute
    type_weights = self.op_selector.get_type_weights(hard=hard)
    unary_weights = self.op_selector.get_unary_weights(hard=hard)
    binary_weights = self.op_selector.get_binary_weights(hard=hard)
    
    # 3. Compute all unary operations
    unary_outputs = [op(s1) for op in self.unary_ops]
    unary_result = sum(w * out for w, out in zip(unary_weights, unary_outputs))
    
    # 4. Compute all binary operations
    binary_outputs = [op(s1, s2) for op in self.binary_ops]
    binary_result = sum(w * out for w, out in zip(binary_weights, binary_outputs))
    
    # 5. Combine based on type selection
    output = type_weights[0] * unary_result + type_weights[1] * binary_result
    
    # 6. Scale output
    return output * self.output_scale
```

---

## Formula Extraction

After training, extract human-readable formula:

**File:** `glassbox/sr/operation_dag.py` - `get_formula()`

```python
def get_formula(self):
    # Track expression for each node
    expressions = ['x0', 'x1', ...]  # Input variables
    
    for layer in self.layers:
        for node in layer.nodes:
            op_str = node.get_selected_operation()  # e.g., "square", "sin"
            sources = node.get_routing_info()['primary_sources']
            
            src1_expr = expressions[sources[0]]
            src2_expr = expressions[sources[1]]
            
            if op_str == 'square':
                expr = f"({src1_expr})²"
            elif op_str == 'sin':
                expr = f"sin({src1_expr})"
            elif op_str == 'add':
                expr = f"({src1_expr} + {src2_expr})"
            # ... etc
            
            expressions.append(expr)
    
    # Combine with output projection weights
    formula_parts = []
    for i, weight in enumerate(self.output_proj.weight[0]):
        if abs(weight) > 0.05:
            formula_parts.append(f"{weight:.2f}*{expressions[i]}")
    
    return " + ".join(formula_parts)
```

---

## Optimizations Implemented

### 1. Efficient Routing
**Before:** Gumbel-softmax with expensive einsum operations
**After:** Simple softmax + matmul
**Speedup:** ~2x

### 2. Batched Selection Sampling
**Before:** 3 separate `hard_concrete_sample()` calls per node
**After:** Single batched call, then split
**Speedup:** ~2x

### 3. Operation Short-Circuit
**Before:** Compute ALL operations, then weight
**After:** Skip operations with weight < 1e-6
**Speedup:** ~1.5x in hard mode

### 4. Compiled Inference (Disabled)
**Concept:** Cache selected operations after training for fast inference
**Status:** Disabled due to bugs causing output discrepancy
**Potential speedup:** ~10x

---

## Configuration Parameters

### Model Architecture

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `n_inputs` | Number of input variables | 1-5 |
| `n_hidden_layers` | Depth of operation graph | 1-2 |
| `nodes_per_layer` | Width of each layer | 4-6 |
| `tau` | Initial selection temperature | 0.5 |

### Training (Evolutionary)

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `population_size` | Number of individuals | 10-20 |
| `generations` | Number of evolution cycles | 15-30 |
| `elite_fraction` | Proportion kept unchanged | 0.2 |
| `mutation_rate` | Probability of parameter mutation | 0.3-0.5 |
| `constant_refine_steps` | GD steps per generation | 30-50 |
| `complexity_penalty` | Regularization for simplicity | 0.001 |
| `entropy_weight` | Regularization for discreteness | 0.01 |

---

## Limitations

### 1. Speed
- **ONN:** ~60-180 seconds for simple formula
- **MLP:** ~0.1 seconds for same task
- **Ratio:** 100-1000x slower

### 2. Reliability
- Correct formula found ~40-60% of the time
- Depends heavily on random initialization
- May find wrong operations that happen to fit training data

### 3. Search Space Explosion
- 4 unary ops × 2 binary ops × 4 nodes = huge space
- Evolution struggles to explore efficiently
- Many local minima

### 4. Snap-to-Discrete Problem
- Continuous parameters (p=1.97) work well during training
- Snapping to discrete (p=2.0) destroys performance
- Current solution: Don't snap, keep continuous values

### 5. Overfitting Risk
- Small formulas can memorize training data
- Extra terms (e.g., `0.08*sin(x)`) absorb noise
- Need coefficient pruning for clean formulas

---

## Example Results

| Task | ONN MSE | MLP MSE | ONN Formula | Correct? |
|------|---------|---------|-------------|----------|
| y = x² | 0.114 | 0.002 | `exp(x) + sin(x)` | ❌ |
| y = sin(x) | 0.001 | 0.001 | `x - x^2.4` | ≈ |
| y = x³ | 0.019 | 0.014 | `2.78*x³ + 0.15*x²` | ✓ |
| y = x² + x | 0.002 | 0.004 | `3.44*x + agg(x)` | ❌ |

---

## Future Improvements

1. **Simpler Search Space**
   - Fewer meta-ops (just power, sin, arithmetic)
   - Stronger bias toward common operations

2. **Coefficient Pruning**
   - Remove terms with |weight| < 0.1
   - Re-fit remaining terms

3. **Multi-Run Ensemble**
   - Run 3-5 times with different seeds
   - Select best by validation MSE

4. **Better Initialization**
   - Start with identity-biased operations
   - Gradually unlock complexity

5. **Hybrid Search**
   - Use genetic programming for structure
   - Gradient descent only for constants

---

## Code References

| Component | File | Main Class/Function |
|-----------|------|---------------------|
| DAG Model | `operation_dag.py` | `OperationDAG` |
| Operation Node | `operation_node.py` | `OperationNode` |
| Meta Operations | `meta_ops.py` | `MetaPower`, `MetaPeriodic`, etc. |
| Hard Concrete | `hard_concrete.py` | `HardConcreteOperationSelector` |
| Routing | `operation_node.py` | `AdaptiveArityRouter` |
| Evolution | `evolution.py` | `EvolutionaryONNTrainer` |
| Training | `operation_dag.py` | `train_onn()` |
| Benchmarks | `scripts/benchmark_suite.py` | CLI benchmark runner |

---

## Usage Example

```python
import torch
from glassbox.sr import OperationDAG
from glassbox.sr.evolution import train_onn_evolutionary

# Generate data
x = torch.linspace(-3, 3, 300).reshape(-1, 1)
y = x**2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model factory
def make_model():
    return OperationDAG(
        n_inputs=1,
        n_hidden_layers=1,
        nodes_per_layer=4,
        n_outputs=1,
        tau=0.5,
    )

# Train
result = train_onn_evolutionary(
    make_model,
    x.to(device),
    y.to(device),
    population_size=15,
    generations=30,
    device=device,
)

# Extract formula
print(f"MSE: {result['final_mse']:.4f}")
print(f"Formula: {result['formula']}")
```

---

*Last updated: January 2026*
