#!/usr/bin/env python3
"""
Detailed pipeline logger for Python or C++ evolution training.

This script records a JSONL event stream with per-generation and per-individual
snapshots, plus optimization/refinement call metadata.

Examples:
  python scripts/evolution_pipeline_log.py --formula "x^2+sin(x)" --x-min -3 --x-max 3
  python scripts/evolution_pipeline_log.py --data-npz data/custom.npz --x-key X --y-key y
    python scripts/evolution_pipeline_log.py --formula "x^2+sin(x)" --cpp-trace
"""

from __future__ import annotations

import argparse

import gzip
import json
import math
import time
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType
from typing import Any, Dict, Optional

import numpy as np
import torch
from sympy import Symbol, sympify
from sympy.utilities.lambdify import lambdify

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CPP_DIR = ROOT / "glassbox" / "sr" / "cpp"
if str(CPP_DIR) not in sys.path:
    sys.path.insert(0, str(CPP_DIR))

from glassbox.sr.operation_dag import OperationDAG
import glassbox.sr.evolution as evo_mod
from glassbox.sr.evolution import EvolutionaryONNTrainer


class JsonlLogger:
    def __init__(self, out_path: Path, compress: bool = False) -> None:
        self.out_path = out_path
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        if compress or self.out_path.suffix == ".gz":
            if not self.out_path.suffix == ".gz":
                self.out_path = self.out_path.with_suffix(self.out_path.suffix + ".gz")
            self._fp = gzip.open(self.out_path, "wt", encoding="utf-8")
        else:
            self._fp = self.out_path.open("w", encoding="utf-8")

    def close(self) -> None:
        self._fp.close()

    def log(self, event: str, **payload: Any) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **_to_jsonable(payload),
        }
        self._fp.write(json.dumps(record, ensure_ascii=True) + "\n")
        self._fp.flush()





def _to_jsonable(x: Any) -> Any:
    if isinstance(x, (str, int, float, bool)) or x is None:
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return x
    if isinstance(x, (np.floating, np.integer)):
        v = x.item()
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            v = x.detach().cpu().item()
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
            return v
        return x.detach().cpu().tolist()
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return str(x)


def _formula_data(formula: str, x_min: float, x_max: float, n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    expr = sympify(formula.replace("^", "**"))
    x_sym = Symbol("x")
    fn = lambdify(x_sym, expr, modules=["numpy"])

    x = np.linspace(x_min, x_max, n_samples, dtype=np.float64)
    y = np.asarray(fn(x), dtype=np.float64)

    if y.shape != x.shape:
        y = y.reshape(x.shape)

    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < max(20, n_samples // 5):
        raise ValueError("Formula produced too few finite samples for logging run")

    x = x[mask].reshape(-1, 1)
    y = y[mask].reshape(-1, 1)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def _npz_data(path: Path, x_key: str, y_key: str) -> tuple[torch.Tensor, torch.Tensor]:
    with np.load(path) as data:
        if x_key not in data or y_key not in data:
            raise KeyError(f"Keys not found in npz: x_key={x_key}, y_key={y_key}")
        x = np.asarray(data[x_key], dtype=np.float64)
        y = np.asarray(data[y_key], dtype=np.float64)

    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    mask = np.all(np.isfinite(x), axis=1) & np.all(np.isfinite(y), axis=1)
    x = x[mask]
    y = y[mask]

    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def _individual_snapshot(ind: evo_mod.Individual, idx: int, include_formula: bool) -> Dict[str, Any]:
    snap: Dict[str, Any] = {
        "idx": idx,
        "fitness": float(ind.fitness) if math.isfinite(ind.fitness) else None,
        "raw_mse": float(ind.raw_mse) if math.isfinite(ind.raw_mse) else None,
        "complexity": float(ind.complexity) if math.isfinite(ind.complexity) else None,
        "generation": int(ind.generation),
        "structure_hash": ind.structure_hash,
        "is_elite": bool(getattr(ind, "_is_elite", False)),
        "is_explorer": bool(getattr(ind, "is_explorer", False)),
    }

    if include_formula and hasattr(ind.model, "get_formula"):
        try:
            snap["formula"] = ind.model.get_formula()
        except Exception as exc:
            snap["formula_error"] = str(exc)
    return snap


def instrument_pipeline(
    trainer: EvolutionaryONNTrainer,
    logger: JsonlLogger,
    include_formulas: bool,
    log_pop_every: int = 1,
    log_pop_elite_only: bool = False,
    log_skip_formulas_non_elite: bool = False,
) -> Dict[str, Any]:
    originals: Dict[str, Any] = {}

    originals["refine_constants"] = evo_mod.refine_constants
    originals["quick_refine_internal"] = evo_mod.quick_refine_internal

    def refine_constants_wrapper(model, x, y, steps=50, lr=0.01, use_lbfgs=False, scales_only=False,
                                hard=True, refine_internal=False, use_amp=True):
        t0 = time.time()
        loss = originals["refine_constants"](
            model, x, y,
            steps=steps,
            lr=lr,
            use_lbfgs=use_lbfgs,
            scales_only=scales_only,
            hard=hard,
            refine_internal=refine_internal,
            use_amp=use_amp,
        )
        logger.log(
            "opt.refine_constants",
            model_id=id(model),
            steps=steps,
            lr=lr,
            use_lbfgs=use_lbfgs,
            scales_only=scales_only,
            hard=hard,
            refine_internal=refine_internal,
            use_amp=use_amp,
            loss=loss,
            elapsed_s=time.time() - t0,
        )
        return loss

    def quick_refine_wrapper(model, x, y, steps=5):
        t0 = time.time()
        loss = originals["quick_refine_internal"](model, x, y, steps=steps)
        logger.log(
            "opt.quick_refine_internal",
            model_id=id(model),
            steps=steps,
            loss=loss,
            elapsed_s=time.time() - t0,
        )
        return loss

    evo_mod.refine_constants = refine_constants_wrapper
    evo_mod.quick_refine_internal = quick_refine_wrapper

    originals["trainer_evaluate_fitness"] = trainer.evaluate_fitness
    originals["trainer_select_and_reproduce"] = trainer.select_and_reproduce
    originals["trainer_evolve_explorers"] = trainer.evolve_explorers

    def eval_wrapper(self, x, y, generation=0, total_generations=50):
        t0 = time.time()
        out = originals["trainer_evaluate_fitness"](x, y, generation=generation, total_generations=total_generations)
        
        # Determine if we should log the full population
        should_log_full = (generation % log_pop_every == 0) or (generation == total_generations - 1)
        
        pop_to_log = []
        exp_to_log = []
        
        if should_log_full:
            if log_pop_elite_only:
                # Only log elites
                pop_to_log = [_individual_snapshot(ind, i, include_formulas) for i, ind in enumerate(self.population) if getattr(ind, "_is_elite", False)]
                exp_to_log = [_individual_snapshot(ind, i, include_formulas) for i, ind in enumerate(self.explorers) if getattr(ind, "_is_elite", False)]
            else:
                # Log everyone, but maybe skip formulas for non-elites
                def get_snap(ind, i):
                    is_elite = getattr(ind, "_is_elite", False)
                    inc_f = include_formulas and (not log_skip_formulas_non_elite or is_elite)
                    return _individual_snapshot(ind, i, inc_f)
                
                pop_to_log = [get_snap(ind, i) for i, ind in enumerate(self.population)]
                exp_to_log = [get_snap(ind, i) for i, ind in enumerate(self.explorers)]

        logger.log(
            "generation.evaluate",
            generation=generation,
            total_generations=total_generations,
            elapsed_s=time.time() - t0,
            best_ever_fitness=(float(self.best_ever.fitness) if self.best_ever and math.isfinite(self.best_ever.fitness) else None),
            population=pop_to_log,
            explorers=exp_to_log,
            is_full_snapshot=should_log_full,
        )
        return out

    def select_wrapper(self, x, y, diversity=10, mutation_rate=None):
        logger.log(
            "generation.select_start",
            generation=self.generation,
            diversity=diversity,
            mutation_rate=mutation_rate,
            confidence=float(self.confidence_tracker.confidence),
        )
        t0 = time.time()
        out = originals["trainer_select_and_reproduce"](x, y, diversity=diversity, mutation_rate=mutation_rate)
        logger.log(
            "generation.select_end",
            generation=self.generation,
            elapsed_s=time.time() - t0,
            new_diversity=len(set(ind.structure_hash for ind in self.population)),
        )
        return out

    def explorer_wrapper(self, x, y):
        logger.log("generation.explorers_start", generation=self.generation, explorers=len(self.explorers))
        t0 = time.time()
        out = originals["trainer_evolve_explorers"](x, y)
        logger.log("generation.explorers_end", generation=self.generation, explorers=len(self.explorers), elapsed_s=time.time() - t0)
        return out

    trainer.evaluate_fitness = MethodType(eval_wrapper, trainer)
    trainer.select_and_reproduce = MethodType(select_wrapper, trainer)
    trainer.evolve_explorers = MethodType(explorer_wrapper, trainer)

    return originals


def restore_pipeline(trainer: EvolutionaryONNTrainer, originals: Dict[str, Any]) -> None:
    evo_mod.refine_constants = originals["refine_constants"]
    evo_mod.quick_refine_internal = originals["quick_refine_internal"]
    trainer.evaluate_fitness = originals["trainer_evaluate_fitness"]
    trainer.select_and_reproduce = originals["trainer_select_and_reproduce"]
    trainer.evolve_explorers = originals["trainer_evolve_explorers"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Detailed evolution pipeline logger (JSONL)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--formula", type=str, help="Target formula, e.g. 'x^2+sin(x)'")
    source.add_argument("--data-npz", type=str, help="Path to NPZ containing X/y arrays")

    parser.add_argument("--x-min", type=float, default=-3.0)
    parser.add_argument("--x-max", type=float, default=3.0)
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--x-key", type=str, default="X")
    parser.add_argument("--y-key", type=str, default="y")

    parser.add_argument("--generations", type=int, default=25)
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument("--nodes-per-layer", type=int, default=4)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--mutation-rate", type=float, default=0.3)
    parser.add_argument("--constant-refine-steps", type=int, default=30)
    parser.add_argument("--print-every", type=int, default=1)

    parser.add_argument("--use-explorers", action="store_true", default=True)
    parser.add_argument("--no-explorers", action="store_true")
    parser.add_argument("--include-formulas", action="store_true", default=True)
    parser.add_argument("--skip-formulas", action="store_true")
    parser.add_argument("--no-cpp-backend", action="store_true", help="Disallow trainer from using C++ backend")
    parser.add_argument("--cpp-trace", action="store_true", help="Run native C++ evolution and emit C++ JSONL trace events")
    parser.add_argument("--early-stop-mse", type=float, default=1e-6)
    parser.add_argument("--arithmetic-temperature", type=float, default=5.0)

    parser.add_argument("--output-dir", type=str, default="results/pipeline_logs")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--compress", action="store_true", help="Compress log with gzip")
    parser.add_argument("--log-pop-every", type=int, default=1, help="Log full population every N generations")
    parser.add_argument("--log-pop-elite-only", action="store_true", help="In non-sampled generations, only log elites")
    parser.add_argument("--log-skip-formulas-non-elite", action="store_true", help="Skip formulas for non-elite individuals")
    args = parser.parse_args()

    if args.no_explorers:
        args.use_explorers = False
    if args.skip_formulas:
        args.include_formulas = False
    if args.no_cpp_backend:
        args.allow_cpp_backend = False
    else:
        args.allow_cpp_backend = True

    if args.formula:
        x_t, y_t = _formula_data(args.formula, args.x_min, args.x_max, args.n_samples)
        source_desc = {"type": "formula", "formula": args.formula, "x_min": args.x_min, "x_max": args.x_max, "n_samples": args.n_samples}
    else:
        x_t, y_t = _npz_data(Path(args.data_npz), args.x_key, args.y_key)
        source_desc = {"type": "npz", "path": args.data_npz, "x_key": args.x_key, "y_key": args.y_key}

    n_inputs = int(x_t.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def model_factory() -> torch.nn.Module:
        return OperationDAG(
            n_inputs=n_inputs,
            n_hidden_layers=args.hidden_layers,
            nodes_per_layer=args.nodes_per_layer,
            n_outputs=1,
            tau=0.5,
        )

    elite_size = max(2, args.population_size // 4)
    trainer = EvolutionaryONNTrainer(
        model_factory=model_factory,
        population_size=args.population_size,
        elite_size=elite_size,
        mutation_rate=args.mutation_rate,
        constant_refine_steps=args.constant_refine_steps,
        device=device,
        use_explorers=args.use_explorers,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"evo_trace_{stamp}"
    out_path = Path(args.output_dir) / f"{run_name}.jsonl"

    if args.cpp_trace:
        import _core

        x_np = x_t.detach().cpu().numpy().astype(np.float64)
        y_np = y_t.detach().cpu().numpy().astype(np.float64).reshape(-1)
        x_list = [np.ascontiguousarray(x_np[:, i]) for i in range(x_np.shape[1])]

        result = _core.run_evolution(
            X_list=x_list,
            y=y_np,
            pop_size=args.population_size,
            generations=args.generations,
            early_stop_mse=args.early_stop_mse,
            arithmetic_temperature=args.arithmetic_temperature,
            trace_path=str(out_path),
            trace_include_formulas=args.include_formulas,
        )

        print(f"C++ native pipeline log written to: {out_path}")
        print(f"Best formula: {result.get('formula', '<none>')}")
        print(f"Best MSE: {result.get('best_mse', None)}")
        return

    logger = JsonlLogger(out_path, compress=args.compress)
    out_path = logger.out_path # Update in case .gz was added

    originals = instrument_pipeline(
        trainer, 
        logger, 
        include_formulas=args.include_formulas,
        log_pop_every=args.log_pop_every,
        log_pop_elite_only=args.log_pop_elite_only,
        log_skip_formulas_non_elite=args.log_skip_formulas_non_elite
    )
    logger.log(
        "run.start",
        run_name=run_name,
        device=str(device),
        source=source_desc,
        config={
            "generations": args.generations,
            "population_size": args.population_size,
            "elite_size": elite_size,
            "mutation_rate": args.mutation_rate,
            "constant_refine_steps": args.constant_refine_steps,
            "hidden_layers": args.hidden_layers,
            "nodes_per_layer": args.nodes_per_layer,
            "use_explorers": args.use_explorers,
            "include_formulas": args.include_formulas,
            "force_python_backend": not args.allow_cpp_backend,
        },
    )

    t0 = time.time()
    try:
        result = trainer.train(x_t, y_t, generations=args.generations, print_every=args.print_every)

        logger.log("run.end", elapsed_s=time.time() - t0, result=result)

        print(f"Detailed pipeline log written to: {out_path}")
        print("Tip: each line is a JSON event; load in pandas for filtering by event type.")
    finally:
        restore_pipeline(trainer, originals)
        logger.close()


if __name__ == "__main__":
    main()
