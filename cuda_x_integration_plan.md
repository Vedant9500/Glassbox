# CUDA-X Integration Plan for Glassbox

Date: 2026-06-02

## Executive Summary

CUDA-X can help this project, but the highest-return path is not a blanket GPU rewrite. Glassbox already has a fast native C++ backend, OpenMP parallelism, PyTorch CUDA support for neural training/inference, and GPU-capable XGBoost training. The best plan is to add GPU acceleration where the workload is dense, batched, and numerically stable, while keeping CPU fallbacks for small jobs and exact validation.

Recommended integration order:

1. Add benchmark instrumentation and CPU/GPU parity guardrails.
2. Use the existing PyTorch CUDA install for Python fast-path exact matching and large basis solves.
3. Rewrite the residual symbolic stage into a bounded mini-search before pushing more work to CUDA.
4. Add optional native CUDA C++ linear algebra via cuBLAS/cuSOLVER for dense solves inside evolution/refinement.
5. Add optional RAPIDS/cuML paths for blackbox preprocessing on Linux/WSL2, not native Windows.
6. Consider TensorRT/ONNX and nvmath-python only after profiling shows inference or Python math dispatch is a bottleneck.

On this machine, CUDA is real and usable: PyTorch 2.5.1+cu121 reports CUDA available on an NVIDIA GeForce RTX 4060 Laptop GPU with 8 GB VRAM. The CUDA toolkit is installed as nvcc 13.0. CuPy and RAPIDS are not currently installed.

## Research Sources

- NVIDIA CUDA-X libraries: https://developer.nvidia.com/cuda/cuda-x-libraries
- cuBLAS documentation: https://docs.nvidia.com/cuda/cublas/index.html
- cuSOLVER documentation: https://docs.nvidia.com/cuda/cusolver/index.html
- cuRAND documentation: https://docs.nvidia.com/cuda/curand/index.html
- CUDA C++ Core Libraries / CCCL: https://nvidia.github.io/cccl/
- RAPIDS cuML overview: https://developer.nvidia.com/topics/ai/data-science/cuda-x-data-science-libraries/cuml
- RAPIDS install guide: https://docs.rapids.ai/install
- CuPy overview: https://docs.cupy.dev/en/stable/overview.html
- TensorRT documentation: https://docs.nvidia.com/deeplearning/tensorrt/latest/
- nvmath-python overview: https://docs.nvidia.com/cuda/nvmath-python/latest/overview.html

## Current Codebase Working Model

The main user-facing path is `GlassboxRegressor.fit` in `glassbox/sr/sklearn_wrapper.py`. The fit flow is roughly:

1. Preprocess and rank features using Python/scikit-learn logic.
2. Run classifier/universal-proposer assisted fast paths.
3. Build candidate symbolic formulas and priors.
4. Score/refine candidates with Python and C++ paths.
5. Run guided evolution and raw C++ evolution.
6. Simplify, clean, refit, and optionally run residual boosting/inception reuse.

Important files and execution roles:

- `glassbox/sr/sklearn_wrapper.py`
  - Orchestrates fitting, candidate refinement, residual boosting, specialist probes, guided evolution, and final model assembly.
  - Calls `run_fast_path(...)` with `device=self.device`, so neural/model-side code can use CUDA.
  - Uses `_core.score_formula_candidates` for C++ formula scoring when available.
  - Uses recursive residual fitting in `_stage_residual_symbolic_fit`, which can dominate runtime.

- `scripts/classifier_fast_path.py`
  - Builds dense symbolic bases in `build_basis_from_predictions`.
  - Performs exact symbolic matching in `find_exact_symbolic_match`.
  - Uses `np.linalg.lstsq` heavily.
  - Contains a PyTorch least-squares branch, but tensors are currently created on CPU; this is the cleanest first CUDA win.
  - Calls native C++ `_core.lasso_coordinate_descent`, `_core.refine_powers`, and `_core.run_evolution`.

- `glassbox/sr/cpp/core.cpp`
  - Pybind11 extension entrypoint.
  - Exposes `score_formula_candidates`, `run_evolution`, `refine_frequencies`, `refine_powers`, `iterative_elastic_net`, `lasso_coordinate_descent`, simplification helpers, and seed graph conversion.

- `glassbox/sr/cpp/evolution.h`
  - Main native evolution engine.
  - Uses Eigen and OpenMP.
  - Dense math hot spots include `DifferentialGramian`, `solve_output_weights`, `refine_constants`, `refine_inner_params_adam`, and `refine_inner_params_lm`.

- `glassbox/sr/cpp/refine.h`
  - Coordinate descent, elastic net, frequency/power refinement, and linear solves.

- `glassbox/sr/blackbox_preprocessor.py`
  - Uses sklearn feature ranking, mutual information, ExtraTrees, LassoCV, ElasticNetCV, and interaction discovery.
  - Good optional RAPIDS/cuML candidate, but only after core symbolic path improvements.

- `glassbox/universal_proposer/universal_proposer.py`
  - Uses PyTorch inference on the selected device, then CPU grammar decoding and affine least-squares ranking.
  - GPU inference is already natural here; TensorRT is lower priority unless profiling shows inference dominates.

- `glassbox/curve_classifier/train_curve_classifier.py`
  - Already has CUDA-aware training, AMP, GPU-resident dataset handling, and TF32 setup.

- `scripts/train_xgboost_classifier.py`
  - Already attempts GPU XGBoost training.
  - This is useful but separate from the CUDA-X symbolic regression runtime.

## Python/C++ Overlap To Preserve

Several optimizations exist in both Python and C++ or are orchestrated in Python while executed in C++. Any CUDA integration must avoid divergent behavior between these implementations.

- Formula scoring:
  - Python fallback: candidate scoring/refinement inside `sklearn_wrapper.py`.
  - C++ path: `_core.score_formula_candidates`.
  - CUDA plan: do not replace arbitrary expression scoring first. Start with GPU scoring only for dense basis matrices or fixed operator templates.

- Least squares, ridge, and affine refits:
  - Python: `np.linalg.lstsq`, `np.linalg.solve`, SciPy refinements.
  - C++: Eigen QR/LDLT in `evolution.h` and `refine.h`.
  - CUDA plan: PyTorch CUDA first in Python; cuBLAS/cuSOLVER later in C++.

- Sparse selection:
  - Python orchestration around LASSO/OLS candidate pool.
  - C++ `_core.lasso_coordinate_descent` and `_core.iterative_elastic_net`.
  - CUDA plan: do not port coordinate descent until dense solve/basis work is measured.

- Refinement:
  - Python calls into `_core.refine_powers`, `_core.refine_frequencies`, and formula constant refinement.
  - C++ does inner-parameter optimization and dense solve steps.
  - CUDA plan: accelerate repeated dense solves and Gramian construction first.

- Evolution:
  - Current production path is native C++ evolution.
  - Legacy PyTorch evolution exists in older modules.
  - CUDA plan: keep native C++ as the source of truth and add optional CUDA math kernels beneath it.

- Simplification:
  - Python/SymPy cleanup and C++ simplification helpers both exist.
  - CUDA plan: no CUDA work here; keep deterministic CPU simplification.

## Library Fit Matrix

| Library | Fit | Where It Applies | Priority |
| --- | --- | --- | --- |
| PyTorch CUDA | Already installed and working | Python fast-path exact search, batched least squares, neural inference/training | Very high |
| CuPy | NumPy-like GPU arrays | Basis generation, large candidate matrices, optional GPU array backend | Medium |
| cuBLAS | NVIDIA dense BLAS | C++ Gramian, matrix multiply, dot products, batched dense math | High after Python pilot |
| cuSOLVER | NVIDIA dense/sparse solvers | C++ QR/SVD/Cholesky/ridge/LM solves | High after Python pilot |
| cuRAND | GPU RNG | Full GPU evolution/mutation if population becomes GPU resident | Low initially |
| CCCL/Thrust/CUB | GPU primitives | Reductions, top-k, sort/select, population scoring kernels | Medium-late |
| RAPIDS cuML | GPU sklearn-like ML | Blackbox preprocessing, feature selection, RF/linear models on Linux/WSL2 | Optional |
| TensorRT/ONNX Runtime CUDA | Optimized inference | Curve classifier/universal proposer inference | Low until inference bottleneck |
| nvmath-python | Python access to NVIDIA math libs | Possible future alternative to CuPy/PyTorch math dispatch | Experimental/low |

## Phase 0: Benchmarking And Guardrails

Goal: make speedups measurable and regressions catchable before changing math kernels.

Implementation targets:

- Add or standardize phase timings in `GlassboxRegressor.fit`.
- Track time spent in:
  - blackbox preprocessing
  - fast-path basis construction
  - exact symbolic match
  - candidate scoring
  - specialist probes
  - guided evolution
  - raw C++ evolution
  - residual stage
  - simplification/final refit
- Add GPU diagnostics when CUDA is enabled:
  - selected backend
  - tensor/device placement
  - fallback count
  - peak allocated GPU memory where available
  - chunk sizes used
- Add parity tests comparing CPU and GPU outputs on fixed small and medium formula sets.

Acceptance criteria:

- CPU-only behavior remains unchanged.
- Benchmarks produce comparable JSON/CSV timing output.
- GPU paths can be disabled globally.
- Final formula validation MSE is always computed through the existing trusted CPU/display path before accepting a GPU-derived candidate.

## Phase 1: Python Fast-Path CUDA Pilot

Goal: accelerate the dense exact-match and basis search path with the PyTorch CUDA stack already present.

Primary targets:

- `scripts/classifier_fast_path.py`
  - `find_exact_symbolic_match`
  - `build_basis_from_predictions`
  - exact polynomial/symbolic matching sections in `fast_path_regression`

Current opportunity:

- The PyTorch least-squares branch builds tensors using `torch.from_numpy(...).float()` but does not explicitly move them to CUDA.
- This means a path that looks GPU-friendly is currently CPU-bound unless a global default tensor setting has been changed.

Plan:

1. Add backend selection: `cpu`, `torch_cuda`, `auto`.
2. Move eligible dense basis and target tensors to CUDA when `auto` decides the work is large enough.
3. Use VRAM-aware chunking for combinations and candidate matrix blocks.
4. Keep CPU for small matrices where GPU launch and transfer overhead will dominate.
5. Compare GPU-derived candidates with CPU final scoring before acceptance.
6. Log whether CUDA was used or skipped and why.

Expected improvement:

- Large basis exact-match workloads can plausibly see 2x to 20x speedup in the exact search/least-squares portion.
- End-to-end speedup depends on whether residual/evolution dominates the run.
- Small datasets may be slower on GPU, so thresholding is required.

Regression risks and solutions:

- Risk: tensor transfer overhead erases the win.
  - Solution: use CUDA only above a work threshold such as `n_samples * n_basis` and candidate combination count.
- Risk: GPU OOM on large bases.
  - Solution: chunk by estimated bytes, catch CUDA OOM, clear cache, and fall back to CPU.
- Risk: numeric drift from float32.
  - Solution: use float64 for parity-sensitive solves where feasible; otherwise use GPU for screening and CPU for final validation.
- Risk: result ordering changes.
  - Solution: deterministic tie breakers and CPU validation of the selected candidate.

## Phase 2: Residual Stage Rewrite

Goal: fix the largest runtime stability risk before investing heavily in native CUDA.

Primary target:

- `glassbox/sr/sklearn_wrapper.py`
  - `_stage_residual_symbolic_fit`
  - `_run_residual_boosting`

Current issue:

- Residual boosting currently constructs another `GlassboxRegressor` and runs a nested symbolic fit with reduced budgets.
- Existing audit notes show specialist/residual paths can balloon runtime. Even if C++ math gets faster, recursive orchestration can still dominate wall time.

Plan:

1. Replace recursive residual estimator fitting with a bounded residual mini-search.
2. Generate residual candidates from:
   - existing candidate formulas
   - specialist probes
   - low-complexity unary/binary templates
   - blackbox interaction hints
   - current residual shape statistics
3. Score candidates with `_core.score_formula_candidates`.
4. Refine only the top candidates.
5. Apply strict candidate caps and timeout caps.
6. Keep the old recursive residual path behind a disabled compatibility flag during transition.

Expected improvement:

- This may be the biggest end-to-end wall-time improvement for hard benchmarks.
- It is not purely a CUDA optimization, but it makes later CUDA work more visible by removing recursive overhead.

Regression risks and solutions:

- Risk: residual quality drops because the recursive estimator explores more.
  - Solution: keep a bounded fallback mode for benchmark comparison; expand templates only when residual error remains high.
- Risk: fewer surprising formulas are discovered.
  - Solution: include seed formulas from universal proposer, fast path, and specialist candidates.
- Risk: changed search behavior makes benchmark comparisons noisy.
  - Solution: add per-phase timing and final MSE/formula complexity reporting before and after the rewrite.

## Phase 3: Native CUDA C++ Linear Algebra Backend

Goal: accelerate dense numeric kernels inside the native C++ backend without changing the symbolic search semantics.

Primary targets:

- `glassbox/sr/cpp/CMakeLists.txt`
- `glassbox/sr/cpp/core.cpp`
- `glassbox/sr/cpp/evolution.h`
- `glassbox/sr/cpp/refine.h`
- new files:
  - `glassbox/sr/cpp/cuda_backend.h`
  - `glassbox/sr/cpp/cuda_backend.cu`

Hot spots to route through a backend abstraction:

- `DifferentialGramian`
  - dense design matrix construction
  - Gramian `A.T @ A`
  - RHS `A.T @ y`
  - ridge solve
- `solve_output_weights`
  - dense basis construction
  - normal equations
  - ridge diagonal
  - LDLT/QR solve
- `refine_inner_params_lm`
  - design and Jacobian matrices
  - projected residual solve
  - small-to-medium dense Hessian solves
- `refine_frequencies_cpp` / `refine_powers_model_cpp`
  - repeated refits and dense solves

Plan:

1. Add `GLASSBOX_ENABLE_CUDA` CMake option, default off.
2. Keep the existing Eigen/OpenMP CPU path as the default backend.
3. Add a small C++ backend interface for dense operations:
   - matrix multiply
   - Gramian construction
   - matrix-vector multiply
   - ridge/least-squares solve
4. Use cuBLAS for GEMM/GEMV/dot-style operations.
5. Use cuSOLVER for QR/SVD/Cholesky or least-squares solves.
6. Add runtime capability detection and return to CPU on unsupported dimensions, allocation failure, or solver failure.
7. Do not port arbitrary graph evaluation in this phase.

Expected improvement:

- Best for large `n_samples`, larger active basis sizes, repeated refits, and longer evolution runs.
- Plausible numeric-kernel speedups: 1.5x to 8x for dense solve-heavy sections.
- End-to-end speedup will be lower if expression evaluation, Python orchestration, or residual recursion dominates.

Regression risks and solutions:

- Risk: native build complexity increases.
  - Solution: optional compile flag, CPU-only build remains first-class, CI keeps CPU path as default.
- Risk: CUDA toolkit / PyTorch CUDA version mismatch.
  - Solution: do not link against PyTorch CUDA; build the native extension against the system CUDA toolkit only when explicitly enabled.
- Risk: solver results differ from Eigen.
  - Solution: parity tests on fixed matrices and formula graphs with relative/absolute tolerances.
- Risk: GPU is slower for small matrices.
  - Solution: route small systems to Eigen using measured thresholds.
- Risk: device allocation overhead.
  - Solution: reuse buffers per evolution run where possible; avoid allocate/free inside inner loops.

## Phase 4: GPU Candidate And Formula Scoring

Goal: accelerate scoring after dense-basis wins are proven.

Primary target:

- `_core.score_formula_candidates`
- specialist candidate scoring in `GlassboxRegressor`
- dense formula matrices created in Python fast path

Plan:

1. Keep the current C++ OpenMP parser/evaluator for arbitrary formulas.
2. Add GPU scoring only when formulas can be represented as dense candidate output matrices or fixed templates.
3. For generated basis matrices, compute MSE/R2/reductions on GPU.
4. Consider CCCL/Thrust/CUB for reductions and top-k selection.
5. Consider generated CUDA kernels only for common expression templates after profiling.

Expected improvement:

- Good speedup for large candidate batches.
- Limited or negative benefit for small formula lists or highly branchy expression trees.

Regression risks and solutions:

- Risk: arbitrary AST evaluation on GPU becomes complex and slow.
  - Solution: do not start there. Start with dense matrices and repeated templates.
- Risk: semantic divergence in safe operations such as log, division, power clipping.
  - Solution: centralize operation semantics and add CPU/GPU property tests.

## Phase 5: RAPIDS/cuML For Blackbox Preprocessing

Goal: optionally accelerate sklearn-like preprocessing and model training for large tabular datasets.

Primary targets:

- `glassbox/sr/blackbox_preprocessor.py`
- `scripts/train_xgboost_classifier.py`

Plan:

1. Add optional cuML imports guarded by backend detection.
2. Use cuML only on supported platforms, mainly Linux/WSL2.
3. Candidate replacements:
   - random forest / extra trees style feature importance where API-compatible
   - linear models where API-compatible
   - mutual information alternatives if available and stable
4. Keep sklearn as the canonical fallback.
5. Update XGBoost GPU configuration to the current supported style after checking installed XGBoost version.

Expected improvement:

- Useful for large preprocessing datasets.
- Lower impact for small symbolic regression runs where search dominates.

Regression risks and solutions:

- Risk: RAPIDS install friction on Windows.
  - Solution: document RAPIDS as WSL2/Linux optional, not a required dependency.
- Risk: cuML APIs do not exactly match sklearn behavior.
  - Solution: isolate each replacement behind a wrapper and compare feature rankings on representative data.
- Risk: dependency conflicts with Python 3.12 or local CUDA versions.
  - Solution: use a separate optional requirements file and environment marker notes.

## Phase 6: Inference Runtime Optimization

Goal: reduce neural inference overhead only if profiling proves it matters.

Primary targets:

- `glassbox/universal_proposer/universal_proposer.py`
- `glassbox/curve_classifier/train_curve_classifier.py`
- classifier model loading and inference helpers

Plan:

1. Keep PyTorch CUDA as the default inference backend.
2. Profile model load time and inference time separately.
3. Consider ONNX Runtime CUDA first if model portability is useful.
4. Consider TensorRT if inference is a repeated hot path and model shapes are stable.

Expected improvement:

- Potentially faster inference and lower latency.
- Probably not the main bottleneck today compared with symbolic search, residual recursion, and dense solves.

Regression risks and solutions:

- Risk: deployment artifacts become harder to manage.
  - Solution: keep exported engines optional and rebuildable.
- Risk: numerical or classification differences change candidate priors.
  - Solution: compare top-k candidate skeletons and final formula quality against PyTorch.

## Proposed Configuration Surface

Environment variables:

- `GLASSBOX_GPU_BACKEND=off|auto|torch_cuda|cupy|cuda_cpp`
- `GLASSBOX_GPU_MIN_WORK=<integer>`
- `GLASSBOX_CUDA_CHUNK_BYTES=<bytes>`
- `GLASSBOX_FORCE_CPU_PARITY=0|1`
- `GLASSBOX_ENABLE_CUDA_CPP=0|1`

Python estimator parameters:

- `gpu_backend="auto"`
- `gpu_min_work=None`
- `gpu_max_memory_fraction=0.5`
- `gpu_validate_final=True`
- `gpu_verbose=False`

Defaults should preserve current CPU behavior unless CUDA is explicitly requested or `auto` is safely enabled.

## Validation Plan

Unit and parity tests:

- CPU vs GPU exact-match result parity for `find_exact_symbolic_match`.
- CPU vs GPU dense least-squares parity on fixed generated matrices.
- C++ Eigen vs cuBLAS/cuSOLVER parity for ridge and least-squares solves.
- Formula candidate scoring parity for any GPU scoring templates.
- Residual mini-search quality comparison against the old recursive residual path on selected formulas.

Benchmark runs:

- Run existing benchmark suite before and after each phase.
- Include small, medium, and hard formulas.
- Include runs with residual stage enabled and disabled.
- Track:
  - wall time
  - phase timings
  - final displayed formula MSE
  - formula complexity
  - GPU memory peak
  - number of CPU fallbacks
  - candidate counts

Acceptance rule:

- A GPU path should be accepted only when it is faster on the intended workload class and does not worsen final validated formula quality beyond an explicit tolerance.

## Expected Speedup Summary

| Area | Expected Speedup | Confidence | Notes |
| --- | --- | --- | --- |
| Python exact basis search with PyTorch CUDA | 2x-20x for large workloads | High | Existing CUDA stack works; current tensors stay on CPU |
| Residual bounded mini-search | Potentially very large wall-time reduction | High | Removes recursive estimator overhead; not purely CUDA |
| C++ dense linear solves with cuBLAS/cuSOLVER | 1.5x-8x for solve-heavy large runs | Medium | Requires native CUDA backend and thresholding |
| GPU candidate reductions/top-k | 2x-10x for large dense batches | Medium | Useful after dense candidate representation exists |
| RAPIDS/cuML preprocessing | 2x-10x for large tabular preprocessing | Medium-low | Platform/install constraints |
| TensorRT/ONNX inference | 1.2x-5x for repeated inference | Low initially | Probably not the current bottleneck |

Small workloads can regress because GPU launch, transfer, and allocation overhead can exceed compute savings. The project should use dynamic thresholds and CPU fallbacks instead of forcing GPU everywhere.

## Potential Regressions And Solutions

### Numerical Drift

Regression:

- GPU float32, TF32, solver differences, or different reduction order can change coefficients and candidate rankings.

Solutions:

- Use float64 for parity-sensitive regression solves where feasible.
- Disable TF32 in validation/parity mode.
- Always compute final displayed formula metrics through the trusted CPU validation path.
- Use relative/absolute tolerances in tests instead of exact coefficient equality.

### Non-Determinism

Regression:

- CUDA kernels, cuSOLVER algorithms, random seeds, and parallel reductions can produce run-to-run variation.

Solutions:

- Seed Python, NumPy, PyTorch, C++ RNG, and cuRAND if introduced.
- Add a deterministic debug/parity mode.
- Compare distributions across multiple seeds for performance claims.
- Avoid accepting formula string equality as the only correctness metric.

### Transfer Overhead

Regression:

- Moving data from NumPy/Eigen CPU arrays to GPU can make small and medium runs slower.

Solutions:

- Use GPU only above measured work thresholds.
- Keep basis matrices and targets resident on GPU across screening/refit steps.
- Batch multiple candidate solves before transferring results back.
- Add logging when the backend chooses CPU because work is too small.

### VRAM Exhaustion

Regression:

- Large bases, candidate matrices, or population outputs can exceed 8 GB VRAM.

Solutions:

- Estimate matrix memory before allocation.
- Chunk by byte budget, not only candidate count.
- Catch CUDA OOM and fall back to CPU.
- Expose `gpu_max_memory_fraction` and chunk-size controls.

### Slower Small Cases

Regression:

- GPU launch overhead dominates simple formulas and small datasets.

Solutions:

- Benchmark threshold cutoffs.
- Keep CPU as default for small `n_samples`, small `n_basis`, and small active graph sizes.
- Add phase timing so the auto selector can be tuned.

### Packaging And CI Breakage

Regression:

- CUDA dependencies can break CPU-only users and CI.

Solutions:

- Make all CUDA libraries optional.
- Keep `requirements.txt` CPU-compatible.
- Add separate GPU docs or `requirements-gpu.txt`.
- Skip GPU tests when CUDA is unavailable.
- Keep C++ CUDA build behind `GLASSBOX_ENABLE_CUDA`.

### Platform Fragmentation

Regression:

- RAPIDS/cuML is not a clean native Windows dependency.

Solutions:

- Treat RAPIDS as WSL2/Linux optional.
- Native Windows GPU plan should prioritize PyTorch CUDA and optional CUDA C++.
- Document platform-specific install paths.

### Search Behavior Changes

Regression:

- Faster GPU scoring may change candidate ordering, ties, pruning, and final formulas.

Solutions:

- Add deterministic tie breakers.
- Preserve candidate order where scores are equal within tolerance.
- Log top-k CPU/GPU candidate overlap during rollout.
- Validate final selected formulas using the same CPU metric path.

### Python/C++ Divergence

Regression:

- Python and C++ implementations of scoring, safe math, snapping, or simplification may drift further apart.

Solutions:

- Declare source-of-truth semantics for safe division, log, exp clipping, power handling, coefficient snapping, and affine scoring.
- Add property tests over random inputs for Python/C++/GPU equivalence.
- Avoid adding separate CUDA semantics for arbitrary formula parsing in the first phases.

### Native Backend Complexity

Regression:

- cuBLAS/cuSOLVER integration can make the C++ extension harder to debug and maintain.

Solutions:

- Add a narrow dense-linalg backend interface.
- Keep Eigen as the reference implementation.
- Use backend-specific tests with tiny, medium, and large matrices.
- Add explicit fallback on CUDA allocation or solver status errors.

## Concrete File Targets

First implementation wave:

- `scripts/classifier_fast_path.py`
  - Add PyTorch CUDA placement and auto thresholding for exact symbolic matching.
  - Add chunked GPU least-squares/search helpers.
  - Keep CPU validation.

- `glassbox/sr/sklearn_wrapper.py`
  - Add clearer phase timing/fallback reporting.
  - Add GPU backend configuration propagation.
  - Replace recursive residual symbolic fit with bounded residual candidate search.

- `tests/`
  - Add CPU/GPU parity tests guarded by CUDA availability.
  - Add residual mini-search regression tests.

Second implementation wave:

- `glassbox/sr/cpp/CMakeLists.txt`
  - Add optional CUDA build path.

- `glassbox/sr/cpp/cuda_backend.h`
- `glassbox/sr/cpp/cuda_backend.cu`
  - Add cuBLAS/cuSOLVER dense-linalg wrappers.

- `glassbox/sr/cpp/evolution.h`
- `glassbox/sr/cpp/refine.h`
- `glassbox/sr/cpp/core.cpp`
  - Route dense solves through CPU/CUDA backend abstraction.

Optional later wave:

- `glassbox/sr/blackbox_preprocessor.py`
  - Add RAPIDS/cuML wrappers for supported platforms.

- `scripts/train_xgboost_classifier.py`
  - Modernize GPU parameter selection based on installed XGBoost version.

- `glassbox/universal_proposer/universal_proposer.py`
  - Consider ONNX/TensorRT only if inference is measured as hot.

## Rollout Plan

1. Baseline and instrumentation
   - No behavior change.
   - Produce timing and quality reports.

2. PyTorch CUDA exact-match pilot
   - Behind `gpu_backend`.
   - Local RTX 4060 should be enough for useful validation.

3. Residual bounded mini-search
   - CPU/C++ first.
   - Compare against old recursive residual path.

4. Native CUDA C++ dense-linalg backend
   - Optional compile flag.
   - Start with Gramian/ridge/least-squares paths.

5. GPU candidate reductions/templates
   - Only after dense matrix representation and thresholds are proven.

6. RAPIDS/cuML and TensorRT optional paths
   - Add only when benchmark data shows these stages matter.

## First Change To Make

The first real implementation should be the PyTorch CUDA exact-match pilot in `scripts/classifier_fast_path.py`, plus instrumentation in `glassbox/sr/sklearn_wrapper.py`.

Reason:

- PyTorch CUDA is already installed and working locally.
- It avoids immediate native build complexity.
- The current PyTorch least-squares branch appears to remain on CPU.
- The exact-match/basis path is dense and batched, which is the right shape for GPU acceleration.
- CPU validation can preserve correctness while we learn real speedup numbers.

The native cuBLAS/cuSOLVER backend is promising, but it should come after profiling and after the residual stage is bounded. Otherwise the project may speed up inner math while still losing wall time to recursive orchestration.
