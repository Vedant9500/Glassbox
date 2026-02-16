# Glassbox: Symbolic Regression with Operation Neural Networks

**Glassbox** is an experimental symbolic regression framework that discovers interpretable mathematical formulas from data. Unlike black-box neural networks, Glassbox produces human-readable equations like `y = x² + sin(x)` or `y = m₀/√(1-v²/c²)`.

## Key Features

- **Operation Neural Networks (ONN)**: Neural architecture where neurons represent mathematical operations rather than weights
- **Meta-Operations**: Parametric operations (MetaPower, MetaPeriodic, MetaExp) that smoothly interpolate between functions
- **Hybrid Optimization**: Combines evolutionary search (topology) with gradient-based fitting (constants)
- **Curve Classifier Fast-Path**: Pre-trained classifier predicts likely operators, enabling direct regression without evolution
- **Flexible Classifier Inference**: Runtime supports PyTorch checkpoints (MLP/CNN auto-detected) and XGBoost payloads
- **GPU Acceleration**: CUDA support for classifier inference and refinement
- **Multi-threaded Search**: Parallel exact-match combinatorial search

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
[Formula Simplification] → Output: "2*x^2 + sin(x)"
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
print(f"MSE: {result['mse']:.6f}")
```

### Fast-Path with Classifier

```python
from scripts.classifier_fast_path import run_fast_path

result = run_fast_path(
    x, y,
    classifier_path="models/curve_classifier.pt",
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
    classifier_path="models/curve_classifier.pt",
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
  --mode {interactive,single,evolution,visualization,pruning}
  --formula FORMULA       Target formula (for single mode)
  --generations N         Number of evolution generations
  --population N          Population size
  --hidden-layers N       ONN hidden layers
  --nodes-per-layer N     Nodes per layer
  --precision {32,64}     Float precision
  --use-curve-classifier  Enable curve classifier warm-start
  --fast-path-only        Skip evolution, use fast-path only
  --ops-periodic          Enable/disable periodic operators
  --ops-exp               Enable/disable exponential operators
  --ops-log               Enable/disable logarithmic operators
  --ops-power             Enable/disable power operators
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
│       ├── evolution.py          # Evolutionary trainer
│       ├── hard_concrete.py      # Differentiable selection
│       ├── pruning.py            # Post-training pruning
│       └── visualization.py      # Training visualization
├── scripts/
│   ├── sr_tester.py              # Main testing tool (TUI)
│   ├── classifier_fast_path.py   # Fast-path regression
│   ├── benchmark_feynman_easy.py # AI-Feynman benchmark
│   ├── curve_classifier_integration.py  # Classifier loading
│   ├── generate_curve_data.py    # Training data generation
│   └── train_curve_classifier.py # Classifier training
├── models/
│   └── curve_classifier.pt       # Pre-trained classifier
├── data/
│   └── feynman_easy/             # Benchmark datasets
├── docs/
│   ├── ONN_Architecture.md       # Technical documentation
│   ├── implementation_plan.md    # Development roadmap
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

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please read CONTRIBUTING.md for guidelines.
