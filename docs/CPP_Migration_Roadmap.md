# C++ Migration Roadmap

This document outlines computationally expensive Python steps identified in the Glassbox SR pipeline that are prime candidates for migration to the C++ backend for massive performance gains and dependency reduction.

## 1. Multi-Start LBFGS Regularized Pruning (`bfgs_optimizer.py`)
Currently, `RegularizedBFGS` uses PyTorch's `LBFGS` optimizer with L1/L2 penalties to prune formulas (e.g., zeroing out terms so `sin(x) + 0.001*x^2` becomes `sin(x)`).
* **The Bottleneck:** Runs a complex PyTorch autograd graph iteratively over scalar equations.
* **The C++ Fix:** Implement a Regularized Least Squares solver natively in Eigen. Eigen supports Ridge (L2) and Lasso (L1) regression via proximal gradient methods or coordinate descent. Migrating this makes the final pruning step instantaneous.

## 2. The SymPy Algebraic Simplification Pipeline (`simplify_formula.py`)
A heavily customized pipeline parses strings, applies trigonometric transformations, and simplifies formulas.
* **The Bottleneck:** SymPy is written in pure Python, is notoriously slow, and adds 1-5 seconds just to print the final formula.
* **The C++ Fix:** Implement a lightweight native algebraic simplifier using the existing C++ AST in `ast.h`. It can recursively walk the tree and merge terms (e.g., `x + x -> 2*x`, `sin(x)*sin(x) -> sin(x)^2`). This removes the `sympy` dependency and drastically reduces CLI startup time.

## 3. Neural Network Inference (Curve Classifier & Universal Proposer)
The fast-path uses PyTorch to load `.pt` files to predict probabilities for mathematical operators.
* **The Bottleneck:** Loading PyTorch runtime takes >1s, and batch-size 1 inference on a small MLP is overkill.
* **The C++ Fix:** Export these models to ONNX and use `ONNXRuntime` in C++, or implement matrix multiplications directly in Eigen. This drops PyTorch from runtime dependencies entirely.

## 4. Legacy PyTorch Evolution Code (`glassbox/evolution/` & `hybrid_optimizer.py`)
Hundreds of lines of legacy PyTorch-based neural evolution loops exist.
* **The Bottleneck:** Technical debt. The Python interpreter parses these files, cluttering the architecture.
* **The Fix:** Deep clean the repository to delete the legacy Python evolution engine once the C++ `core.run_evolution` is fully feature-complete.
