import argparse
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# Add repo root and scripts dir to path for imports.
REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from classifier_fast_path import run_fast_path
from glassbox.evolution import detect_dominant_frequency, EvolutionaryONNTrainer
from glassbox.sr.core.operation_dag import OperationDAG


# AI-Feynman demo mappings from the upstream README/Table 4:
# example1: I.8.14  -> d = sqrt((x2-x1)^2 + (y2-y1)^2)
# example2: I.10.7  -> m = m0/sqrt(1-v^2/c^2)
# example3: I.50.26 -> x = x1*(cos(omega*t)+alpha*cos(omega*t)^2)

DATASETS: List[Dict[str, str]] = [
    {
        "name": "example1.txt",
        "formula": "sqrt((x1-x0)^2 + (x3-x2)^2)",  # I.8.14: 2D distance
        "url": "https://raw.githubusercontent.com/SJ001/AI-Feynman/master/example_data/example1.txt",
    },
    {
        "name": "example2.txt",
        "formula": "x0/sqrt(1-x1^2/x2^2)",  # I.10.7: Relativistic mass
        "url": "https://raw.githubusercontent.com/SJ001/AI-Feynman/master/example_data/example2.txt",
    },
    {
        "name": "example3.txt",
        "formula": "x0*(cos(x1*x2)+x3*cos(x1*x2)^2)",  # I.50.26: cosine envelope
        "url": "https://raw.githubusercontent.com/SJ001/AI-Feynman/master/example_data/example3.txt",
    },
]

DEFAULT_CLASSIFIER_PATH = "models/curve_classifier_v3.1.pt"


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def validate_classifier_path(classifier_path: str) -> Path:
    model_path = resolve_repo_path(classifier_path)
    if model_path.exists():
        return model_path

    models_dir = REPO_ROOT / "models"
    available: List[str] = []
    if models_dir.exists():
        for pattern in ("curve_classifier*.pt", "*.pkl", "*.joblib"):
            available.extend(sorted(path.name for path in models_dir.glob(pattern)))

    available_text = ", ".join(dict.fromkeys(available)) if available else "none found in models/"
    raise FileNotFoundError(
        f"Classifier model not found at {model_path}. Available models: {available_text}"
    )


def download_if_missing(url: str, dest_path: Path) -> bool:
    if dest_path.exists():
        return False
    if not url:
        raise FileNotFoundError(f"Dataset missing at {dest_path} and no download URL was provided")
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
    url = dataset.get("url", "")
    dest = data_dir / name

    try:
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
        if x_t.ndim == 1:
            x_t = x_t.reshape(-1, 1)
        y_t = torch.tensor(y, dtype=dtype).reshape(-1, 1)

        detected_omegas = None
        if x_t.shape[1] == 1:
            try:
                detected_omegas = detect_dominant_frequency(x_t, y_t, n_frequencies=3)
            except Exception as fft_err:
                print(f"FFT warm-start skipped: {fft_err}")

        print("\n" + "=" * 72)
        print(f"DATASET: {name} | rows={x_t.shape[0]} | n_inputs={x_t.shape[1]} | precision={precision}")
        if dataset.get("formula"):
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
            print(f"FAST PATH: no usable candidate (took {elapsed:.2f}s)")
            if not use_evolution_fallback:
                return {
                    "name": name,
                    "status": "fast_path_unavailable",
                    "time": elapsed,
                    "reason": "not_applicable",
                }
            mse = float("inf")
            fast_path_formula = ""
        else:
            mse = result.get("mse", float("inf"))
            fast_path_formula = result.get("formula", "")
            print(f"FAST PATH: MSE={mse:.6f} | formula={fast_path_formula[:60]}...")

        if use_evolution_fallback and mse > mse_threshold:
            print(f"\nFAST PATH MSE ({mse:.6f}) > threshold ({mse_threshold}), falling back to evolution...")

            n_inputs = x_t.shape[1]
            resolved_device = device if device and device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

            def model_factory():
                return OperationDAG(
                    n_inputs=n_inputs,
                    n_hidden_layers=2,
                    nodes_per_layer=4,
                    simplified_ops=False,
                )

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
                y_t.to(resolved_device),
                generations=evolution_generations,
                print_every=10,
            )
            evo_elapsed = time.time() - evo_start
            evo_mse = evo_result.get("final_mse", float("inf"))

            if evo_mse < mse:
                return {
                    "name": name,
                    "status": "evolution",
                    "mse": evo_mse,
                    "time": elapsed + evo_elapsed,
                    "formula": evo_result.get("formula", ""),
                    "fast_path_mse": mse,
                }

            print(f"EVOLUTION: no improvement (MSE={evo_mse:.6f}); keeping fast-path result")
            if result is None:
                return {
                    "name": name,
                    "status": "error",
                    "reason": "evolution_failed_to_improve",
                    "mse": evo_mse,
                    "time": elapsed + evo_elapsed,
                    "formula": evo_result.get("formula", ""),
                }

            return {
                "name": name,
                "status": "ok" if mse < mse_threshold else "approx",
                "mse": mse,
                "time": elapsed + evo_elapsed,
                "formula": fast_path_formula,
                "evolution_mse": evo_mse,
            }

        return {
            "name": name,
            "status": "ok" if mse < mse_threshold else "approx",
            "mse": mse,
            "time": result.get("time") if result else elapsed,
            "formula": fast_path_formula,
        }
    except Exception as err:
        print(f"ERROR {name}: {err}")
        return {
            "name": name,
            "status": "error",
            "reason": str(err),
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
        default=DEFAULT_CLASSIFIER_PATH,
        help=f"Path to the curve classifier model (default: {DEFAULT_CLASSIFIER_PATH})",
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

    data_dir = resolve_repo_path(args.data_dir)
    try:
        classifier_path = str(validate_classifier_path(args.classifier_path))
    except FileNotFoundError as err:
        parser.error(str(err))
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

    if max_rows is not None and sample is not None and sample >= max_rows:
        print(
            f"NOTE: --sample {sample} has no effect when --max-rows {max_rows} loads fewer rows first."
        )

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
        if status == "skip":
            print(f"{r['name']}: [skip] {r.get('reason', 'skipped')}")
            continue
        if status == "fast_path_unavailable":
            print(f"{r['name']}: [fast_path_unavailable] {r.get('reason', 'not_applicable')}")
            continue
        if status == "error":
            print(f"{r['name']}: [error] {r.get('reason', 'unknown_error')}")
            continue
        mse = r.get("mse", float("nan"))
        t = r.get("time", float("nan"))
        f = r.get("formula", "")
        f_disp = (f[:60] + "...") if len(f) > 60 else f
        extra = ""
        if status == "evolution" and "fast_path_mse" in r:
            extra = f" fast_path_mse={r['fast_path_mse']:.6f}"
        elif "evolution_mse" in r:
            extra = f" evolution_mse={r['evolution_mse']:.6f}"
        print(f"{r['name']}: [{status}] mse={mse:.6f} time={t:.2f}s formula={f_disp}{extra}")


if __name__ == "__main__":
    main()
