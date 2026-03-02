# Glassbox: Symbolic Regression with Operation Neural Networks

**Glassbox** is an experimental symbolic regression framework that discovers interpretable mathematical formulas from data. Unlike black-box neural networks, Glassbox produces human-readable equations like `y = x² + sin(x)` or `y = m₀/√(1-v²/c²)`.

## Key Features

- **Operation Neural Networks (ONN)**: Neural architecture where neurons represent mathematical operations rather than weights
- **Meta-Operations**: Parametric operations (MetaPower, MetaPeriodic, MetaExp) that smoothly interpolate between functions
- **Hybrid Optimization**: Combines evolutionary search (topology) with gradient-based fitting (constants)
- **Curve Classifier Fast-Path**: Pre-trained classifier predicts likely operators, enabling direct regression without evolution
- **Optimized Fast-Path (Default)**: Single strategy with refinement gating and acceptance guardrails for stable runtime
- **Formula Post-Processing**: Tolerance-based float snapping + SymPy simplification with snap-only fallback for very large expressions
- **Fast C++ Backend**: Native C++ engine for topology evolution and constant fitting
- **Hybrid Optimization**: Combines evolutionary search (topology) with analytical SVD-based linear fitting
- **OpenMP Multithreading**: Utilizes all CPU cores for massively parallel population evaluation
- **Comprehensive Benchmark Suite**: 8-tier evaluation framework covering ~200 formulas
- **GPU Acceleration**: CUDA support for classifier inference and refinement
- **Flexible Classifier Inference**: Supports PyTorch checkpoints (MLP/CNN) and XGBoost payloads

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/glassbox.git
cd glassbox

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# (Optional) For GPU support, install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Requirements

- Python 3.8+
- PyTorch 1.12+
- NumPy, SciPy, Matplotlib
- Rich (for TUI)
- Graphviz (optional, for visualization)

## Quick Start

### Interactive Mode (TUI)

```bash
python scripts/sr_tester.py
```

### Single Formula Test

```bash
python scripts/sr_tester.py --mode single --formula "x^2 + sin(x)"
```

### Comprehensive Benchmark
```bash
python scripts/benchmark_suite.py --with-evolution
```

### Fast-Path Benchmark (AI-Feynman)

```bash
python scripts/benchmark_feynman_easy.py
```

## Architecture Overview

### Operation Neural Network (ONN)

```
Input (x) → [Layer 1] → [Layer 2] → [Output Projection] → Prediction (ŷ)
               ↓            ↓
            [Node 1]    [Node 1]    ← Each node selects ONE operation
            [Node 2]    [Node 2]
            [Node 3]    [Node 3]
```

Each **OperationNode** performs:
1. **Routing**: Select input sources from previous layers
2. **Operation Selection**: Choose unary (sin, power, exp) or binary (+, ×)
3. **Apply**: Compute the selected operation
4. **Scale**: Learnable output scaling

### Meta-Operations

Instead of discrete operations, ONN uses **parametric meta-operations**:

| Meta-Operation | Formula | Recovers |
|----------------|---------|----------|
| `MetaPeriodic` | `A·sin(ωx + φ)` | sin, cos, linear |
| `MetaPower` | `sign(x)·\|x\|^p` | identity, square, sqrt, inverse |
| `MetaArithmetic` | `(2-β)(x+y) + (β-1)(xy)` | add, multiply |
| `MetaExp` | `exp(αx)` | exp, decay |

### Fast-Path Pipeline

For well-characterized curves, the fast-path bypasses evolution:

```
Data (x, y)
    ↓
[Curve Classifier] → Predict operators (sin, power, exp, ...)
    ↓
[Basis Builder] → Generate candidate terms (x, x², sin(x), ...)
    ↓
[LASSO/OLS Regression] → Fit coefficients
    ↓
[Post-Processing] → Float snapping + symbolic simplify → Output: "2*x^2 + sin(x)"
```

## Usage Examples

### Basic Symbolic Regression

```python
import torch
from glassbox.sr.evolution import train_onn_evolutionary

# Generate data
x = torch.linspace(-3, 3, 100).reshape(-1, 1)
y = x**2 + torch.sin(x)

# Train ONN
result = train_onn_evolutionary(x, y, generations=50)
print(f"Discovered formula: {result['formula']}")
print(f"MSE: {result['final_mse']:.6f}")
```

### Full Suite Benchmark
```bash
python scripts/benchmark_suite.py --tier 4 --with-evolution
```

### Fast-Path with Classifier
```bash
from scripts.classifier_fast_path import run_fast_path

result = run_fast_path(
    x, y,
    classifier_path="models/curve_classifier_v3.1.pt",
    device="cuda",  # or "cpu", "auto"
)
print(f"Formula: {result['formula']}")
```

### Custom Benchmark

```python
from scripts.benchmark_feynman_easy import run_dataset

result = run_dataset(
    dataset={"name": "my_data.txt", "url": None},
    data_dir="data/",
  classifier_path="models/curve_classifier_v3.1.pt",
    precision=64,
    max_rows=5000,
    sample=2000,
    seed=42,
    auto_expand=True,
    device="cuda",
    exact_match_threads=8,
    exact_match_enabled=True,
    exact_match_max_basis=150,
)
```

## CLI Options

### sr_tester.py

```bash
python scripts/sr_tester.py [OPTIONS]

Options:
  --mode {interactive,single,evolution,viz,pruning}
  --formula FORMULA       Target formula (for single mode)
  --generations N         Number of evolution generations
  --population N          Population size
  --hidden-layers N       ONN hidden layers
  --nodes-per-layer N     Nodes per layer
  --precision {32,64}     Float precision
  --curve-classifier      Enable curve classifier warm-start
  --fast-path-only        Skip evolution, use fast-path only
  --ops-periodic          Enable/disable periodic operators
  --ops-exp               Enable/disable exponential operators
  --ops-log               Enable/disable logarithmic operators
  --ops-power             Enable/disable power operators
```

### benchmark_suite.py
```bash
python scripts/benchmark_suite.py [OPTIONS]

Options:
  --classifier-model PATH      Path to curve classifier (default: models/curve_classifier_v3.1.pt)
  --with-evolution             Enable C++ evolution fallback for approximate fast-path results
  --tier N                     Target specific tier(s), repeatable: --tier 2 --tier 3
  --output-dir PATH            Results directory (default: results/)
  --n-samples N                Number of points per formula (default: 300)
  --device {auto,cpu,cuda}     Device for classifier inference (default: auto)
  --timeout SEC                Timeout per formula (default: 60)
  --quiet                      Suppress per-formula logs
  --formula TEXT               Run only formulas matching text
  --cpp-evolution-only         Skip fast-path and run pure C++ evolution
  --pop-size N                 C++ evolution population size
  --generations N              C++ evolution generations
```

### benchmark_feynman_easy.py
```bash
python scripts/benchmark_feynman_easy.py [OPTIONS]

Options:
  --data-dir PATH         Directory for datasets
  --classifier-path PATH  Path to curve classifier model
  --precision {32,64}     Float precision
  --sample N              Random sample size
  --seed N                Random seed
  --device {auto,cpu,cuda}  Device for inference
  --no-auto-expand        Disable auto basis expansion
  --skip-exact-match      Skip exact-match combinatorial search
  --exact-match-threads N Threads for exact-match search
  --exact-match-max-basis N  Skip exact-match when basis exceeds this size
```

## Project Structure

```
glassbox/
├── glassbox/
│   └── sr/
│       ├── operation_dag.py      # Main ONN model
│       ├── operation_node.py     # Individual operation nodes
│       ├── meta_ops.py           # Parametric meta-operations
│       ├── evolution.py          # Python evolutionary trainer gateway
│       ├── cpp/                  # High-performance C++ backend
│       │   ├── evolution.h       # C++ Evolution Engine (OpenMP + SVD)
│       │   ├── ast.h             # C++ Expression DAG structures
│       │   └── core.cpp          # Pybind11 bridge
│       ├── hard_concrete.py      # Differentiable selection
│       ├── pruning.py            # Post-training pruning
│       └── visualization.py      # Training visualization
├── scripts/
│   ├── sr_tester.py              # Main testing tool (TUI)
│   ├── benchmark_suite.py        # Comprehensive 8-tier benchmark
│   ├── classifier_fast_path.py   # Fast-path regression
│   ├── benchmark_feynman_easy.py # AI-Feynman benchmark
│   ├── curve_classifier_integration.py  # Classifier loading
│   ├── generate_curve_data.py    # Training data generation
│   └── train_curve_classifier.py # Classifier training
├── models/
│   └── curve_classifier_v3.1.pt  # Pre-trained classifier
├── data/
│   └── feynman_easy/             # Benchmark datasets
├── docs/
│   ├── ONN_Architecture.md       # Technical documentation
│   ├── Research_Roadmap.md       # Development roadmap
│   └── key_insights.md           # Research notes
└── requirements.txt
```

## How It Works

### 1. Evolutionary Search (Phase 1)

The evolutionary trainer maintains a population of ONN models:

1. **Initialize**: Random operation selections
2. **Evaluate**: Compute MSE fitness on training data
3. **Select**: Keep elite performers
4. **Mutate**: Perturb operation selections and parameters
5. **Crossover**: Combine successful topologies
6. **Repeat**: Until convergence or generation limit

### 2. Constant Fitting (Phase 2)

After topology discovery, L-BFGS refines constants:

- Fix operation selections
- Optimize meta-operation parameters (ω, φ, p, etc.)
- Fine-tune output projection weights

### 3. Pruning (Phase 3)

Remove inactive or redundant nodes:

- Analyze coefficient magnitudes
- Prune near-zero contributions
- Simplify formula representation
### 4. High-Performance C++ Backend (Phase 4)
For complex formulas where the fast-path fails, Glassbox triggers a native C++ evolution engine:
- **Topology Mutation**: Efficient DAG structural mutations beyond rigid layer formats.
- **Analytical Constant Fitting**: Uses Eigen SVD to solve for linear parameters in O(1) time per graph.
- **OpenMP Multithreading**: Parallelizes population evaluation across all CPU cores.
- **Performance**: Capable of running 2,000 generations in <0.1s on modern 8-core CPUs.

## Performance Tips

1. **Use Fast-Path for Simple Formulas**: `--fast-path-only` can solve polynomial/trigonometric formulas in <1 second

2. **Skip Exact-Match for Large Basis**: When basis >150 terms, exact-match search is slow; use `--skip-exact-match` or let the auto-skip handle it

3. **GPU for Classifier**: Use `--device cuda` for faster classifier inference

4. **Reduce Sample Size**: For quick iteration, use `--sample 500` instead of full dataset

5. **Operator Constraints**: If you know the formula family, disable irrelevant operators (e.g., `--no-ops-periodic` for purely polynomial targets)

## Troubleshooting

### "Stalling at Fast-path basis: N terms"

The exact-match search is slow for large bases. Solutions:
- Use `--skip-exact-match`
- Reduce `--exact-match-max-basis` (default: 150)
- Use `--no-auto-expand` for smaller basis

### "Stalling at [Post] Running simplify_formula pipeline..."

Large symbolic expressions can be expensive to simplify. Current fast-path includes a large-expression guard that switches to snap-only mode automatically. If you still hit stalls, reduce basis growth (`--no-auto-expand` in Feynman benchmark) or test with fewer samples first.

### "CUDA not available"

Your PyTorch installation is CPU-only. Install CUDA version:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### "Classifier model not found"

Train the classifier or download pre-trained weights:
```bash
python scripts/train_curve_classifier.py
```

## Citation

If you use Glassbox in your research, please cite:

```bibtex
@software{glassbox2026,
  title={Glassbox: Symbolic Regression with Operation Neural Networks},
  year={2026},
  url={https://github.com/your-org/glassbox}
}
```

## Contributing

Contributions welcome! Open an issue or pull request with a clear description and reproducible steps.
