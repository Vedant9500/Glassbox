import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional
import urllib.request

import numpy as np
import torch

from scripts.classifier_fast_path import run_fast_path
from glassbox.sr.evolution import detect_dominant_frequency


DATASETS: List[Dict[str, str]] = [
    {
        "name": "example1.txt",
        "url": "https://raw.githubusercontent.com/SJ001/AI-Feynman/master/example_data/example1.txt",
    },
    {
        "name": "example2.txt",
        "url": "https://raw.githubusercontent.com/SJ001/AI-Feynman/master/example_data/example2.txt",
    },
    {
        "name": "example3.txt",
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
    if x.shape[1] != 1:
        print(f"SKIP {name}: expects 1D inputs, got {x.shape[1]}D")
        return {"name": name, "status": "skip", "reason": "multi_dim_inputs"}

    dtype = torch.float64 if precision == 64 else torch.float32
    x_t = torch.tensor(x.squeeze(), dtype=dtype).reshape(-1, 1)
    y_t = torch.tensor(y, dtype=dtype).reshape(-1, 1)

    detected_omegas = detect_dominant_frequency(x_t, y_t, n_frequencies=3)

    print("\n" + "=" * 72)
    print(f"DATASET: {name} | rows={x_t.shape[0]} | precision={precision}")
    print("=" * 72)

    result = run_fast_path(
        x_t,
        y_t,
        classifier_path=classifier_path,
        detected_omegas=detected_omegas,
        op_constraints=None,
        auto_expand=auto_expand,
    )

    if result is None:
        print("FAST PATH: not applicable")
        return {"name": name, "status": "no_fast_path"}

    return {
        "name": name,
        "status": "ok",
        "mse": result.get("mse"),
        "time": result.get("time"),
        "formula": result.get("formula"),
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

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    classifier_path = args.classifier_path
    precision = args.precision
    max_rows = args.max_rows
    sample = args.sample if args.sample and args.sample > 0 else None
    seed = args.seed
    auto_expand = not args.no_auto_expand

    results = []
    for dataset in DATASETS:
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
            )
        )

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for r in results:
        status = r.get("status")
        if status != "ok":
            print(f"{r['name']}: {status}")
            continue
        mse = r.get("mse", float("nan"))
        t = r.get("time", float("nan"))
        f = r.get("formula", "")
        f_disp = (f[:60] + "...") if len(f) > 60 else f
        print(f"{r['name']}: mse={mse:.6f} time={t:.2f}s formula={f_disp}")


if __name__ == "__main__":
    main()
