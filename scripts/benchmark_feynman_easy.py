import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import urllib.request

import numpy as np
import torch

# Add repo root and scripts dir to path for imports
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
scripts_dir = os.path.join(repo_root, "scripts")
sys.path.insert(0, repo_root)
sys.path.insert(0, scripts_dir)

from classifier_fast_path import run_fast_path
from glassbox.sr.evolution import detect_dominant_frequency, EvolutionaryONNTrainer
from glassbox.sr.operation_dag import OperationDAG


# Feynman equations reference (for verification):
# example1: I.6.2a  -> f = exp(-theta^2/2)/sqrt(2*pi)
# example2: I.9.18  -> F = G*m1*m2/((x2-x1)^2+(y2-y1)^2+(z2-z1)^2)
# example3: I.10.7  -> m = m0/sqrt(1-v^2/c^2)

DATASETS: List[Dict[str, str]] = [
    {
        "name": "example1.txt",
        "formula": "exp(-theta^2/2)/sqrt(2*pi)",  # I.6.2a: Gaussian
        "url": "https://raw.githubusercontent.com/SJ001/AI-Feynman/master/example_data/example1.txt",
    },
    {
        "name": "example2.txt",
        "formula": "G*m1*m2/r^2",  # I.9.18: Gravitational force (simplified)
        "url": "https://raw.githubusercontent.com/SJ001/AI-Feynman/master/example_data/example2.txt",
    },
    {
        "name": "example3.txt",
        "formula": "m0/sqrt(1-v^2/c^2)",  # I.10.7: Relativistic mass
        "url": "https://raw.githubusercontent.com/SJ001/AI-Feynman/master/example_data/example3.txt",
    },
]


def download_if_missing(url: str, dest_path: Path) -> bool:
    if dest_path.exists():
        return False
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest_path)  # nosec - trusted source
    return True


def load_dataset(
    path: Path,
    max_rows: Optional[int] = None,
    sample: Optional[int] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    data = np.loadtxt(path, max_rows=max_rows)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if sample is not None and sample > 0 and sample < data.shape[0]:
        rng = np.random.default_rng(seed)
        idx = rng.choice(data.shape[0], size=sample, replace=False)
        data = data[idx]
    return data


def run_dataset(
    dataset: Dict[str, str],
    data_dir: Path,
    classifier_path: str,
    precision: int,
    max_rows: Optional[int],
    sample: Optional[int],
    seed: Optional[int],
    auto_expand: bool,
    device: Optional[str],
    exact_match_threads: int,
    exact_match_enabled: bool,
    exact_match_max_basis: int,
    mse_threshold: float,
    use_evolution_fallback: bool,
    evolution_generations: int,
    evolution_population: int,
) -> Dict[str, object]:
    name = dataset["name"]
    url = dataset["url"]
    dest = data_dir / name

    downloaded = download_if_missing(url, dest)
    if downloaded:
        print(f"Downloaded {name} -> {dest}")

    data = load_dataset(dest, max_rows=max_rows, sample=sample, seed=seed)
    if data.shape[1] < 2:
        print(f"SKIP {name}: expected at least 2 columns, got {data.shape[1]}")
        return {"name": name, "status": "skip", "reason": "not_enough_columns"}

    x = data[:, :-1]
    y = data[:, -1]
    if x.shape[1] < 1:
        print(f"SKIP {name}: expected at least 1 input, got {x.shape[1]}")
        return {"name": name, "status": "skip", "reason": "no_inputs"}

    dtype = torch.float64 if precision == 64 else torch.float32
    x_t = torch.tensor(x, dtype=dtype)
    y_t = torch.tensor(y, dtype=dtype)  # 1D tensor, run_fast_path handles reshaping

    detected_omegas = None
    if x_t.shape[1] == 1:
        detected_omegas = detect_dominant_frequency(x_t, y_t, n_frequencies=3)

    import time
    
    print("\n" + "=" * 72)
    print(f"DATASET: {name} | rows={x_t.shape[0]} | n_inputs={x_t.shape[1]} | precision={precision}")
    if dataset.get('formula'):
        print(f"TARGET:  {dataset['formula']}")
    print("=" * 72)
    
    start_time = time.time()

    result = run_fast_path(
        x_t,
        y_t,
        classifier_path=classifier_path,
        detected_omegas=detected_omegas,
        op_constraints=None,
        auto_expand=auto_expand,
        device=device,
        exact_match_threads=exact_match_threads,
        exact_match_enabled=exact_match_enabled,
        exact_match_max_basis=exact_match_max_basis,
    )

    elapsed = time.time() - start_time
    
    if result is None:
        print(f"FAST PATH: not applicable (took {elapsed:.2f}s)")
        mse = float('inf')
    else:
        mse = result.get("mse", float('inf'))
        print(f"FAST PATH: MSE={mse:.6f} | formula={result.get('formula', '')[:60]}...")
    
    # Fallback to evolution if MSE too high
    if use_evolution_fallback and mse > mse_threshold:
        print(f"\nFAST PATH MSE ({mse:.6f}) > threshold ({mse_threshold}), falling back to evolution...")
        
        # Reshape for evolution
        y_evo = y_t.reshape(-1, 1) if y_t.ndim == 1 else y_t
        
        # Create model factory (trainer needs callable, not instance)
        n_inputs = x_t.shape[1]
        resolved_device = device if device and device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        def model_factory():
            return OperationDAG(
                n_inputs=n_inputs,
                n_hidden_layers=2,
                nodes_per_layer=4,
                simplified_ops=False,  # Full ops for complex formulas
            )
        
        # Run evolution
        trainer = EvolutionaryONNTrainer(
            model_factory,
            population_size=evolution_population,
            mutation_rate=0.5,
            elite_size=4,
            explorer_fraction=0.3,
            device=resolved_device,
        )
        
        evo_start = time.time()
        evo_result = trainer.train(
            x_t.to(resolved_device),
            y_evo.to(resolved_device),
            generations=evolution_generations,
            print_every=10,
        )
        evo_elapsed = time.time() - evo_start
        
        return {
            "name": name,
            "status": "evolution",
            "mse": evo_result.get("final_mse", float('inf')),
            "time": elapsed + evo_elapsed,
            "formula": evo_result.get("formula", ""),
            "fast_path_mse": mse,
        }

    return {
        "name": name,
        "status": "ok" if mse < mse_threshold else "approx",
        "mse": mse,
        "time": result.get("time") if result else elapsed,
        "formula": result.get("formula", "") if result else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark on AI-Feynman easy examples (example1-3)."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/feynman_easy",
        help="Local directory to store downloaded datasets",
    )
    parser.add_argument(
        "--classifier-path",
        type=str,
        default="models/curve_classifier.pt",
        help="Path to the curve classifier model",
    )
    parser.add_argument(
        "--precision",
        type=int,
        choices=[32, 64],
        default=32,
        help="Floating-point precision to use",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional max rows to load (for speed)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=2000,
        help="Optional random sample size (0 to disable)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--no-auto-expand",
        action="store_true",
        help="Disable auto basis expansion",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for classifier/refinement (auto uses CUDA if available)",
    )
    parser.add_argument(
        "--exact-match-threads",
        type=int,
        default=1,
        help="Threads for exact-match search (pairs/triples); 1 disables threading",
    )
    parser.add_argument(
        "--skip-exact-match",
        action="store_true",
        help="Skip exact-match combinatorial search",
    )
    parser.add_argument(
        "--exact-match-max-basis",
        type=int,
        default=150,
        help="Skip exact-match search when basis exceeds this size",
    )
    parser.add_argument(
        "--mse-threshold",
        type=float,
        default=0.001,
        help="MSE threshold below which to consider success (default: 0.001)",
    )
    parser.add_argument(
        "--evolution-fallback",
        action="store_true",
        help="Fall back to evolution when fast path MSE exceeds threshold",
    )
    parser.add_argument(
        "--evolution-generations",
        type=int,
        default=40,
        help="Generations for evolution fallback (default: 40)",
    )
    parser.add_argument(
        "--evolution-population",
        type=int,
        default=30,
        help="Population size for evolution fallback (default: 30)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Run only specific dataset by name (e.g., 'example2')",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    classifier_path = args.classifier_path
    precision = args.precision
    max_rows = args.max_rows
    sample = args.sample if args.sample and args.sample > 0 else None
    seed = args.seed
    auto_expand = not args.no_auto_expand
    device = args.device
    exact_match_threads = max(1, args.exact_match_threads)
    exact_match_enabled = not args.skip_exact_match
    exact_match_max_basis = args.exact_match_max_basis
    mse_threshold = args.mse_threshold
    use_evolution_fallback = args.evolution_fallback
    evolution_generations = args.evolution_generations
    evolution_population = args.evolution_population

    # Filter datasets if --dataset specified
    datasets_to_run = DATASETS
    if args.dataset:
        datasets_to_run = [d for d in DATASETS if args.dataset.lower() in d['name'].lower()]
        if not datasets_to_run:
            print(f"No dataset matching '{args.dataset}'. Available: {[d['name'] for d in DATASETS]}")
            return

    results = []
    for dataset in datasets_to_run:
        results.append(
            run_dataset(
                dataset,
                data_dir,
                classifier_path,
                precision,
                max_rows,
                sample,
                seed,
                auto_expand,
                device,
                exact_match_threads,
                exact_match_enabled,
                exact_match_max_basis,
                mse_threshold,
                use_evolution_fallback,
                evolution_generations,
                evolution_population,
            )
        )

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for r in results:
        status = r.get("status")
        if status not in ("ok", "evolution", "approx"):
            print(f"{r['name']}: {status}")
            continue
        mse = r.get("mse", float("nan"))
        t = r.get("time", float("nan"))
        f = r.get("formula", "")
        f_disp = (f[:60] + "...") if len(f) > 60 else f
        print(f"{r['name']}: [{status}] mse={mse:.6f} time={t:.2f}s formula={f_disp}")


if __name__ == "__main__":
    main()
