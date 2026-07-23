"""
Glassbox SR Benchmark Suite
============================

Comprehensive evaluation of the symbolic regression pipeline across ~200
formulas of increasing complexity, organized into 8 difficulty tiers.

Usage:
  python scripts/benchmark_suite.py                                            # Full suite (fast-path only)
  python scripts/benchmark_suite.py --tier 1                                   # Only tier 1
  python scripts/benchmark_suite.py --specialist-regressor --specialist-full   # Use specialist regressor for all tiers with fast-path 
  python scripts/benchmark_suite.py --evolution-only                           # Guided evolution only (skip fast-path)
  python scripts/benchmark_suite.py --classifier-model models/v3.pt            # Custom model
  python scripts/benchmark_suite.py --output-dir results/                      # Custom output dir

Scoring:
    Uses displayed-formula MSE only for scoring; raw MSE is diagnostic.
  EXACT   — MSE < 1e-6 (or noise-aware band under --noise) AND ≤ 10 terms
  APPROX  — MSE < 0.01
  LOOSE   — MSE < 0.1
  FAIL    — MSE ≥ 0.1 or error

  Under --noise, EXACT is a *noisy-fit* band, not structure recovery.
  Prefer CleanMSE / R2clean / Recov columns (see noise_handling_audit.md).
"""

import argparse
import json
import math
import os
import pickle
import platform
import random
import re
import subprocess
import sys
import time
import warnings
import traceback

# Fix UnicodeEncodeError on Windows
if sys.platform == 'win32':
    import io
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    else:
        # Fallback for older python versions
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
try:
    from sympy.utilities.exceptions import SymPyDeprecationWarning
except Exception:  # pragma: no cover - SymPy warning class may be absent
    SymPyDeprecationWarning = DeprecationWarning

warnings.filterwarnings("ignore", category=SymPyDeprecationWarning)

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))

import classifier_fast_path as cfp  # noqa: E402
from scripts import benchmark_common as bc  # noqa: E402
from classifier_fast_path import run_fast_path, run_guided_evolution  # noqa: E402
from glassbox.evolution import detect_dominant_frequency  # noqa: E402
from glassbox.sr.cpp.seed_graph_builder import build_seed_graphs_from_signal  # noqa: E402


def _finite_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _finite_int(value: Any, default: int) -> int:
    try:
        out = int(round(float(value)))
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _planned_guided_budget(
    search_plan: Dict[str, Any],
    base_generations: int,
    base_population: int,
    *,
    trust_plan: bool,
) -> Tuple[int, int, Dict[str, Any]]:
    """Translate proposer search_plan metadata into guided-evolution kwargs."""
    gen_mult = _finite_float(search_plan.get("generation_multiplier"), 1.0)
    pop_mult = _finite_float(search_plan.get("population_multiplier"), 1.0)
    if gen_mult <= 0.0:
        gen_mult = 1.0
    if pop_mult <= 0.0:
        pop_mult = 1.0
    if not trust_plan:
        # Bounded mode should let the proposer shape search, not silently turn
        # every approximate fast-path case into a much larger benchmark run.
        gen_mult = float(np.clip(gen_mult, 0.25, 1.25))
        pop_mult = float(np.clip(pop_mult, 0.25, 1.15))

    if trust_plan:
        generations = max(1, int(round(base_generations * gen_mult)))
        population = max(1, int(round(base_population * pop_mult)))
    else:
        generations = int(np.clip(
            round(base_generations * gen_mult),
            20,
            max(20, base_generations * 4),
        ))
        population = int(np.clip(
            round(base_population * pop_mult),
            20,
            max(20, base_population * 3),
        ))

    guided_plan: Dict[str, Any] = {}
    # Always forward a per-formula deterministic seed when present.
    if "random_seed" in search_plan and search_plan.get("random_seed") is not None:
        guided_plan["random_seed"] = int(search_plan["random_seed"])
    if trust_plan:
        if "n_beams" in search_plan:
            guided_plan["n_beams"] = max(1, _finite_int(search_plan.get("n_beams"), 1))
        if "n_rounds" in search_plan:
            guided_plan["n_rounds"] = max(1, _finite_int(search_plan.get("n_rounds"), 1))
        if "p_min" in search_plan:
            guided_plan["p_min"] = _finite_float(search_plan.get("p_min"), -2.0)
        if "p_max" in search_plan:
            guided_plan["p_max"] = _finite_float(search_plan.get("p_max"), 3.0)
        if "seed_budget" in search_plan:
            guided_plan["seed_budget"] = max(0, _finite_int(search_plan.get("seed_budget"), 0))
        if "acceptable_complexity" in search_plan:
            guided_plan["acceptable_complexity"] = max(
                1, _finite_int(search_plan.get("acceptable_complexity"), 15)
            )
        if "early_stop_max_nodes" in search_plan:
            guided_plan["early_stop_max_nodes"] = max(
                1, _finite_int(search_plan.get("early_stop_max_nodes"), 50)
            )
        if "acceptable_mse" in search_plan:
            guided_plan["acceptable_mse"] = max(
                0.0, _finite_float(search_plan.get("acceptable_mse"), 1e-8)
            )

    return generations, population, guided_plan


# ---------------------------------------------------------------------------
# Benchmark Formula Bank  (~200 formulas across 8 tiers)
# ---------------------------------------------------------------------------
# Each entry: (formula_string, human_name, x_range, n_inputs)
# x_range is (x_min, x_max); negative log/sqrt domains handled via auto-shrink

TIER_1_TRIVIAL = [
    ("5",                        "Constant 5",               (-5, 5)),
    ("x",                        "Identity",                 (-5, 5)),
    ("-x",                       "Negation",                 (-5, 5)),
    ("2*x",                      "Linear 2x",                (-5, 5)),
    ("0.5*x",                    "Linear 0.5x",              (-5, 5)),
    ("x+1",                      "Linear x+1",               (-5, 5)),
    ("3*x-2",                    "Linear 3x-2",              (-5, 5)),
    ("x^2",                      "x²",                       (-5, 5)),
    ("x^3",                      "x³",                       (-5, 5)),
    ("x^4",                      "x⁴",                       (-5, 5)),
    ("-x^2",                     "−x²",                      (-5, 5)),
    ("2*x^2",                    "2x²",                      (-3, 3)),
    ("x^2+1",                    "x²+1",                     (-5, 5)),
    ("x^3-x",                    "x³−x",                     (-3, 3)),
    ("x^2-1",                    "x²−1",                     (-5, 5)),
    ("0.5*x^2+x",               "½x²+x",                    (-5, 5)),
    ("-3*x+7",                   "−3x+7",                    (-5, 5)),
    ("x^2+x+1",                 "x²+x+1",                   (-5, 5)),
    ("2*x^3",                    "2x³",                      (-3, 3)),
    ("x^4-x^2",                 "x⁴−x²",                    (-3, 3)),
    ("x^2/2",                    "x²/2",                     (-5, 5)),
    ("x^5",                      "x⁵",                       (-2, 2)),
    ("10",                       "Constant 10",              (-5, 5)),
    ("pi*x",                     "πx",                       (-3, 3)),
    ("x/3",                      "x/3",                      (-5, 5)),
    ("e",                        "Constant e",               (-5, 5)),
    ("pi",                       "Constant π",               (-5, 5)),
    ("2.0",                      "Constant 2.0",             (-5, 5)),
    ("0.5",                      "Constant 0.5",             (-5, 5)),
    ("e*x",                      "Linear e·x",               (-3, 3)),
]

TIER_2_SIMPLE_POLY = [
    ("x^3+x^2+x",               "Nguyen-1: x³+x²+x",       (-3, 3)),
    ("x^4+x^3+x^2+x",           "Nguyen-2: x⁴+x³+x²+x",   (-3, 3)),
    ("x^5+x^4+x^3+x^2+x",      "Nguyen-3: x⁵+x⁴+x³+x²+x", (-2, 2)),
    ("x^6+x^5+x^4+x^3+x^2+x",  "Nguyen-4: deg-6 poly",     (-2, 2)),
    ("3*x^3+2*x^2+x",           "3x³+2x²+x",               (-3, 3)),
    ("x^4-2*x^2+1",             "(x²−1)²",                  (-3, 3)),
    ("x^3+3*x^2+3*x+1",         "(x+1)³",                   (-3, 3)),
    ("x^4+4*x^3+6*x^2+4*x+1",   "(x+1)⁴",                   (-2, 2)),
    ("x^2-2*x+1",               "(x−1)²",                   (-5, 5)),
    ("4*x^3-3*x",               "Chebyshev T₃",             (-1, 1)),
    ("8*x^4-8*x^2+1",           "Chebyshev T₄",             (-1, 1)),
    ("16*x^5-20*x^3+5*x",       "Chebyshev T₅",             (-1, 1)),
    ("x^2+2*x-3",               "Quadratic roots ±1,−3",    (-5, 5)),
    ("x^3-6*x^2+11*x-6",        "Cubic roots 1,2,3",        (-1, 5)),
    ("-x^4+x^2",                "−x⁴+x²",                  (-3, 3)),
    ("x^2*x",                   "x³ (product form)",        (-3, 3)),
    ("(x+2)*(x-1)",             "x²+x−2",                   (-3, 3)),
    ("x*(x-1)*(x+1)",           "x³−x",                     (-3, 3)),
    ("0.1*x^5-0.5*x^3+x",       "Odd polynomial",           (-2, 2)),
    ("x^6-1",                    "x⁶−1",                     (-2, 2)),
    ("x^3/3-x",                 "x³/3−x",                   (-3, 3)),
    ("x^4/4-x^2/2",             "x⁴/4−x²/2",               (-3, 3)),
    ("2*x^2-5*x+2",             "Quadratic 2x²−5x+2",      (-5, 5)),
    ("x*(x+1)*(x+2)",           "Rising factorial",          (-3, 3)),
    ("(x^2+1)*(x-1)",           "Cubic factored",            (-3, 3)),
]

TIER_3_BASIC_TRANSCENDENTAL = [
    ("sin(x)",                   "sin(x)",                   (-6, 6)),
    ("cos(x)",                   "cos(x)",                   (-6, 6)),
    ("sin(2*x)",                 "sin(2x)",                  (-6, 6)),
    ("cos(2*x)",                 "cos(2x)",                  (-6, 6)),
    ("sin(x/2)",                 "sin(x/2)",                 (-6, 6)),
    ("2*sin(x)",                 "2sin(x)",                  (-6, 6)),
    ("sin(x)+1",                 "sin(x)+1",                 (-6, 6)),
    ("-cos(x)",                  "−cos(x)",                  (-6, 6)),
    ("3*cos(x)-1",               "3cos(x)−1",                (-6, 6)),
    ("exp(x)",                   "eˣ",                       (-3, 3)),
    ("exp(-x)",                  "e⁻ˣ",                      (-3, 3)),
    ("exp(-x^2)",                "Gaussian e⁻ˣ²",            (-3, 3)),
    ("log(x+1)",                 "log(x+1)",                 (0.01, 5)),
    ("log(x^2+1)",               "log(x²+1)",                (-5, 5)),
    ("exp(x)-1",                 "eˣ−1",                     (-3, 3)),
    ("exp(-x)-1",                "e⁻ˣ−1",                    (-3, 3)),
    ("2*exp(-x)",                "2e⁻ˣ",                     (-3, 3)),
    ("sin(pi*x)",                "sin(πx)",                  (-2, 2)),
    ("cos(pi*x)",                "cos(πx)",                  (-2, 2)),
    ("exp(x/2)",                 "e^(x/2)",                  (-4, 4)),
    ("log(2*x+1)",               "log(2x+1)",                (0.01, 5)),
    ("sin(3*x)",                 "sin(3x)",                  (-6, 6)),
    ("cos(3*x)",                 "cos(3x)",                  (-6, 6)),
    ("exp(-2*x)",                "e⁻²ˣ",                     (-2, 4)),
    ("sqrt(x)",                  "√x",                       (0.01, 10)),
]

TIER_4_NGUYEN = [
    # Nguyen benchmark suite (standard SR benchmark)
    ("x^3+x^2+x",               "Nguyen-1",                 (-1, 1)),
    ("x^4+x^3+x^2+x",           "Nguyen-2",                 (-1, 1)),
    ("x^5+x^4+x^3+x^2+x",      "Nguyen-3",                 (-1, 1)),
    ("x^6+x^5+x^4+x^3+x^2+x",  "Nguyen-4",                 (-1, 1)),
    ("sin(x^2)*cos(x)-1",        "Nguyen-5",                 (-1, 1)),
    ("sin(x)+sin(x+x^2)",        "Nguyen-6",                 (-1, 1)),
    ("log(x+1)+log(x^2+1)",      "Nguyen-7",                 (0.01, 2)),
    ("sqrt(x)",                  "Nguyen-8",                  (0.01, 4)),
    # Nguyen-9: sin(x) + sin(x^2)
    ("sin(x)+sin(x^2)",          "Nguyen-9",                 (-3, 3)),
    # Nguyen-10: 2*sin(x)*cos(x) = sin(2x)
    ("2*sin(x)*cos(x)",          "Nguyen-10",                (-3, 3)),
    # Additional Nguyen-like
    ("x^3+x",                    "Nguyen-like: x³+x",        (-3, 3)),
    ("x^4-x",                    "Nguyen-like: x⁴−x",        (-2, 2)),
    ("sin(x)*cos(x)",            "sin·cos identity",         (-6, 6)),
    ("sin(x)^2",                 "sin²(x)",                  (-6, 6)),
    ("cos(x)^2",                 "cos²(x)",                  (-6, 6)),
    ("sin(x)^2+cos(x)^2",        "Pythagorean identity",     (-6, 6)),
    ("sin(x)^2-cos(x)^2",        "−cos(2x)",                 (-6, 6)),
    ("x*sin(x)",                 "x·sin(x)",                 (-6, 6)),
    ("x*cos(x)",                 "x·cos(x)",                 (-6, 6)),
    ("x^2*sin(x)",               "x²·sin(x)",               (-4, 4)),
    # Keijzer benchmarks
    ("0.3*x*sin(2*pi*x)",        "Keijzer-4",                (-3, 3)),
    ("x^3*exp(-x)*cos(x)*sin(x)*(sin(x)^2*cos(x)-1)", "Keijzer-complex", (-2, 2)),
    # R (Rational/Polynomial mix)
    ("x^2+x+1",                  "R1: x²+x+1",               (-3, 3)),
    ("x^4+x^3+x^2+x+1",         "R2: deg-4+constant",        (-2, 2)),
    ("2*x^3-3*x^2+x",            "R3: factorable cubic",      (-2, 3)),
]

TIER_5_SUMS_AND_PRODUCTS = [
    ("sin(x)+x^2",               "sin(x)+x²",                (-5, 5)),
    ("cos(x)+x^2",               "cos(x)+x²",                (-5, 5)),
    ("sin(x)+cos(x)",            "sin(x)+cos(x)",             (-6, 6)),
    ("sin(x)+sin(2*x)",          "sin(x)+sin(2x)",           (-6, 6)),
    ("sin(x)+sin(3*x)",          "sin(x)+sin(3x)",           (-6, 6)),
    ("cos(x)+cos(2*x)",          "cos(x)+cos(2x)",           (-6, 6)),
    ("cos(x)+cos(3*x)",          "cos(x)+cos(3x)",           (-6, 6)),
    ("sin(x)-cos(x)",            "sin(x)−cos(x)",            (-6, 6)),
    ("sin(x)+x",                 "sin(x)+x",                 (-5, 5)),
    ("cos(x)+x",                 "cos(x)+x",                 (-5, 5)),
    ("x^2+exp(-x)",              "x²+e⁻ˣ",                  (-3, 3)),
    ("x+exp(-x)",                "x+e⁻ˣ",                   (-3, 3)),
    ("sin(x)+exp(-x)",           "sin(x)+e⁻ˣ",              (-3, 3)),
    ("x^2+log(x+1)",             "x²+log(x+1)",             (0.01, 5)),
    ("x^3+sin(x)",               "x³+sin(x)",                (-3, 3)),
    ("x*sin(x)",                 "x·sin(x)",                 (-6, 6)),
    ("x*exp(-x)",                "x·e⁻ˣ",                   (-2, 5)),
    ("x^2+sin(x)+1",             "x²+sin(x)+1",             (-5, 5)),
    ("exp(-x)+exp(-2*x)",        "e⁻ˣ+e⁻²ˣ",               (-1, 5)),
    ("sin(x)*sin(2*x)",          "sin(x)·sin(2x)",           (-6, 6)),
    ("sin(x)+cos(2*x)+x",        "Mixed trig+linear",        (-5, 5)),
    ("x^2-sin(x)",               "x²−sin(x)",               (-5, 5)),
    ("2*sin(x)+3*cos(x)",        "2sin(x)+3cos(x)",          (-6, 6)),
    ("sin(x)^3",                 "sin³(x)",                  (-6, 6)),
    ("cos(x)+sin(2*x)+x^2",      "cos+sin2+x²",             (-4, 4)),
]

TIER_6_RATIONAL_AND_NESTED = [
    ("1/(1+x^2)",                "Witch of Agnesi",          (-5, 5)),
    ("x/(1+x^2)",               "x/(1+x²)",                (-5, 5)),
    ("1/(1+exp(-x))",            "Sigmoid σ(x)",              (-6, 6)),
    ("x/(1+abs(x))",             "SoftSign",                 (-5, 5)),
    ("sin(x^2)",                 "sin(x²)",                  (-3, 3)),
    ("cos(x^2)",                 "cos(x²)",                  (-3, 3)),
    ("exp(-x^2)",                "Gaussian",                 (-3, 3)),
    ("x*exp(-x^2)",              "x·Gaussian",               (-3, 3)),
    ("sin(exp(x))",              "sin(eˣ)",                  (-2, 2)),
    ("exp(sin(x))",              "exp(sin(x))",              (-3, 3)),
    ("log(1+x^2)",               "log(1+x²)",                (-5, 5)),
    ("log(1+exp(x))",            "Softplus",                 (-3, 3)),
    ("sqrt(1+x^2)",              "√(1+x²)",                  (-5, 5)),
    ("1/(x^2+0.5)",              "Lorentzian",               (-5, 5)),
    ("x^2/(1+x^2)",              "x²/(1+x²)",               (-5, 5)),
    ("sin(x)/x",                 "Sinc (unnormalized)",       (0.1, 10)),
    ("(1-x^2)/(1+x^2)",          "Rational symmetric",       (-3, 3)),
    ("x^3/(1+x^4)",              "Rational odd",             (-3, 3)),
    ("exp(-abs(x))",             "Laplacian",                (-5, 5)),
    ("x/(exp(x)-1)",             "Planck-like",              (0.1, 5)),
    ("sin(pi*x)/(pi*x)",         "Sinc (normalized)",         (0.1, 5)),
    ("1/sqrt(1+x^2)",            "Inv-√(1+x²)",              (-5, 5)),
    ("exp(-x)*sin(x)",           "Damped sine",              (0, 10)),
    ("exp(-x)*cos(x)",           "Damped cosine",            (0, 10)),
    ("x^2*exp(-x)",              "x²·e⁻ˣ",                  (0, 8)),
]

TIER_7_HARD_COMPOSITIONS = [
    ("x^2*exp(-x)*sin(x)",       "x²·e⁻ˣ·sin(x)",           (0, 8)),
    ("sin(x)*cos(2*x)+x",        "sin·cos2+x",               (-5, 5)),
    ("exp(-x^2)*sin(3*x)",       "Gauss·sin(3x)",            (-3, 3)),
    ("sin(x+sin(x))",            "sin(x+sin(x))",            (-3, 3)),
    ("x*log(x+1)",               "x·log(x+1)",               (0.01, 5)),
    ("exp(-x)*sin(2*x)",         "Damped sin(2x)",           (0, 10)),
    ("sin(x)/(1+x^2)",           "sin/(1+x²)",               (-5, 5)),
    ("cos(x)/(1+x^2)",           "cos/(1+x²)",               (-5, 5)),
    ("x^2*sin(1/x)",             "x²·sin(1/x)",              (0.1, 5)),
    ("exp(sin(x))*cos(x)",       "exp(sin)·cos",             (-3, 3)),
    ("sin(x)*exp(-x^2/2)",       "sin·Gaussian",             (-4, 4)),
    ("log(1+sin(x)^2)",          "log(1+sin²)",              (-3, 3)),
    ("x*exp(-abs(x))*sin(x)",    "x·Lap·sin",                (-5, 5)),
    ("sin(x^2)+cos(x)",          "sin(x²)+cos(x)",           (-3, 3)),
    ("(sin(x)+cos(x))^2",        "1+sin(2x)",                (-6, 6)),
    ("exp(-x)*x^3",              "x³·e⁻ˣ",                  (0, 8)),
    ("sin(x)*sin(3*x)*sin(5*x)", "Triple sine product",      (-3, 3)),
    ("sqrt(abs(sin(x)))",        "√|sin(x)|",                (-6, 6)),
    ("x^2/(exp(x)-1)",           "Bose-like",                (0.1, 5)),
    ("exp(-x^2/2)*cos(5*x)",     "Gabor wavelet",            (-3, 3)),
    ("sin(x)+sin(2*x)+sin(3*x)", "Fourier 3-term",           (-6, 6)),
    ("cos(x)+cos(2*x)+cos(3*x)", "Cosine 3-term",            (-6, 6)),
    ("x*sin(x)*cos(x)",          "x·sin·cos",                (-5, 5)),
    ("sin(x)*log(x+1)",          "sin·log",                   (0.01, 5)),
    ("exp(-x)*(x^2-2*x+1)",      "e⁻ˣ·(x−1)²",             (0, 8)),
]

TIER_8_FRONTIER = [
    ("sin(cos(x))",              "sin(cos(x))",              (-3, 3)),
    ("cos(sin(x))",              "cos(sin(x))",              (-3, 3)),
    ("sin(x*cos(x))",            "sin(x·cos(x))",            (-3, 3)),
    ("log(1+sin(x))",            "log(1+sin(x))",            (-1, 1)),
    ("exp(-x)*sin(x)^2",         "e⁻ˣ·sin²(x)",             (0, 10)),
    ("sin(exp(-x))",             "sin(e⁻ˣ)",                (-1, 4)),
    ("x^2*exp(-x)*cos(3*x)",     "x²·e⁻ˣ·cos(3x)",         (0, 8)),
    ("1/(1+exp(-x))-0.5",        "Centered sigmoid",         (-6, 6)),
    ("sin(x^2)*exp(-x)",         "sin(x²)·e⁻ˣ",            (0, 6)),
    ("log(x)*sin(x)",            "log(x)·sin(x)",            (0.1, 10)),
    ("sqrt(abs(x))*sin(x)",      "√|x|·sin(x)",             (-5, 5)),
    ("exp(-x^2)*sin(x^2)",       "Gauss·sin(x²)",            (-3, 3)),
    ("sin(x)/sqrt(1+x^2)",       "sin/√(1+x²)",              (-5, 5)),
    ("x/(1+x^4)",               "x/(1+x⁴)",                (-3, 3)),
    ("exp(-abs(x))*cos(2*x)",    "Laplace·cos(2x)",          (-5, 5)),
    ("sin(x+exp(-x))",           "sin(x+e⁻ˣ)",              (-2, 4)),
    ("cos(x^2)*sin(x)",          "cos(x²)·sin(x)",           (-3, 3)),
    ("(sin(x)+x)/(1+x^2)",       "(sin+x)/(1+x²)",           (-5, 5)),
    ("exp(-x)*(sin(x)+cos(x))",  "e⁻ˣ·(sin+cos)",           (0, 10)),
    ("x^2*sin(x)/(1+x^2)",       "x²sin/(1+x²)",            (-5, 5)),
    ("log(1+x^2)*sin(x)",        "log(1+x²)·sin",           (-5, 5)),
    ("sin(x)*cos(x)*exp(-x^2)",  "sin·cos·Gauss",            (-3, 3)),
    ("exp(-x)*sin(x)*cos(2*x)",  "Damped modulated",         (0, 10)),
    ("(x^2-1)*exp(-x^2/2)",      "Hermite-like",             (-4, 4)),
    ("sin(pi*x)*exp(-x^2)",      "sin(πx)·Gauss",            (-3, 3)),
]

ALL_TIERS = {
    1: ("Trivial",                TIER_1_TRIVIAL),
    2: ("Simple Polynomial",      TIER_2_SIMPLE_POLY),
    3: ("Basic Transcendental",   TIER_3_BASIC_TRANSCENDENTAL),
    4: ("Nguyen Suite",           TIER_4_NGUYEN),
    5: ("Sums & Products",        TIER_5_SUMS_AND_PRODUCTS),
    6: ("Rational & Nested",      TIER_6_RATIONAL_AND_NESTED),
    7: ("Hard Compositions",      TIER_7_HARD_COMPOSITIONS),
    8: ("Frontier",               TIER_8_FRONTIER),
}

# ---------------------------------------------------------------------------
# Formula evaluator (reused from sr_tester.py logic)
# ---------------------------------------------------------------------------

def _safe_numpy_power(x, p):
    """Safe power matching C++ signed power logic."""
    x = np.asarray(x)
    p = np.asarray(p)
    abs_x = np.abs(x) + 1e-15
    res = np.power(abs_x, p)
    p_round = np.round(p)
    is_even = (np.abs(p - p_round) < 1e-6) & (p_round.astype(np.int64) % 2 == 0)
    if np.isscalar(is_even):
        return res if is_even else np.sign(x) * res
    return np.where(is_even, res, np.sign(x) * res)


def _safe_numpy_log(x, base=None):
    """NumPy log that also supports SymPy's log(x, base) lambdify output."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(x)
        if base is not None:
            out = out / np.log(base)
    return out


def _parse_formula(formula_str: str) -> Callable[[np.ndarray], np.ndarray]:
    """Parse formula string into a numpy function with safe power handling."""
    import sympy as sp
    from sympy.parsing.sympy_parser import (
        convert_xor,
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )
    
    # Normalize unicode and common variants
    formula = _normalize_formula_text(formula_str).strip()
    # Handle C++ |x| notation
    formula = re.sub(r'\|([^|]+)\|', r'abs(\1)', formula)
    
    try:
        transformations = standard_transformations + (convert_xor, implicit_multiplication_application)        
        local_dict = {
            "Piecewise": sp.Piecewise,
            "Eq": sp.Eq,
            "Abs": sp.Abs,
            "sign": sp.sign,
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "exp": sp.exp,
            "log": sp.log,
            "sqrt": sp.sqrt,
            "pi": sp.pi,
            "E": sp.E,
            "e": sp.E,
        }
        expr = parse_expr(formula, local_dict=local_dict, transformations=transformations, evaluate=False)     
        free_syms = sorted(expr.free_symbols, key=lambda sym: sym.name)
        # Inject safe power into lambdify
        modules = [
            {"pow": _safe_numpy_power, "Pow": _safe_numpy_power, "log": _safe_numpy_log},
            "numpy",
        ]
        func = sp.lambdify(free_syms, expr, modules=modules)
        
        def fn(x_in: np.ndarray) -> np.ndarray:
            if x_in.ndim == 1:
                cols = [x_in]
            else:
                cols = [x_in[:, i] for i in range(x_in.shape[1])]
            
            # Match columns to symbols
            args = []
            for sym in free_syms:
                # Expect symbols like x, x0, x1...
                if sym.name == 'x':
                    args.append(cols[0])
                elif sym.name.startswith('x') and sym.name[1:].isdigit():
                    idx = int(sym.name[1:])
                    args.append(cols[idx] if idx < len(cols) else np.zeros_like(cols[0]))
                else:
                    args.append(np.zeros_like(cols[0]))
            
            if not args and not free_syms:
                # Constant expression
                val = float(expr.evalf())
                return np.full_like(x_in if x_in.ndim == 1 else x_in[:, 0], val)
                
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                res = func(*args)
            return np.asarray(res, dtype=np.float64).reshape(-1)
            
        return fn
    except Exception as e:
        err_msg = str(e)
        # Fallback to a very basic eval if sympy fails (unlikely)
        def fallback_fn(x: np.ndarray) -> np.ndarray:
            raise ValueError(f"SymPy parse failed for '{formula_str}': {err_msg}")
        return fallback_fn


# Preset noise levels for --noise flag (RMS as fraction of signal std).
# Mirrors the Phase 0 tiers in scripts/benchmark_noise.py but kept as simple
# Gaussian presets so the existing clean benchmark can run under noise without
# pulling the full protocol.
def _noise_type_for_preset(preset: str) -> str:
    cfg = NOISE_PRESETS.get(preset)
    return (cfg or {}).get("noise_type", "clean") if cfg else "clean"


def _noise_level_for_preset(preset: str) -> float:
    cfg = NOISE_PRESETS.get(preset)
    return (cfg or {}).get("noise_level", 0.0) if cfg else 0.0


NOISE_PRESETS: Dict[str, Optional[Dict[str, Any]]] = {
    "none": None,
    "low": {"noise_type": "gaussian", "noise_level": 0.001},
    "medium": {"noise_type": "gaussian", "noise_level": 0.01},
    "high": {"noise_type": "gaussian", "noise_level": 0.10},
    "outliers": {"noise_type": "outliers", "noise_level": 0.03},
}


def _apply_noise_to_y(y: np.ndarray, noise_cfg: Optional[Dict[str, Any]], *, seed: int) -> np.ndarray:
    """Inject noise into a clean target vector using benchmark_noise generators."""
    if not noise_cfg:
        return y
    try:
        from scripts import benchmark_noise as bn
    except Exception:
        # Fallback: simple additive Gaussian so --noise still works if the
        # Phase 0 module is unavailable. Match constant-target scale rule.
        rng = np.random.RandomState(int(seed))
        arr = np.asarray(y, dtype=np.float64)
        y_std = float(np.std(arr))
        y_abs = float(np.mean(np.abs(arr))) if arr.size else 0.0
        floor = max(1e-12, 1e-6 * max(y_abs, 1.0))
        if y_std <= floor:
            scale = y_abs if y_abs > 1e-12 else 1.0
        else:
            scale = y_std
        return arr + rng.normal(0.0, float(noise_cfg.get("noise_level", 0.0)) * scale, size=arr.shape)
    return bn.apply_noise_tier(y, noise_cfg, seed=int(seed))


def _generate_data(
    formula_str: str,
    x_min: float = -5.0,
    x_max: float = 5.0,
    n_samples: int = 300,
    noise_cfg: Optional[Dict[str, Any]] = None,
    noise_seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate (x, y) data from a formula string, optionally adding noise."""
    fn = _parse_formula(formula_str)
    x = np.linspace(x_min, x_max, n_samples)
    y = fn(x)

    # Filter out non-finite values
    mask = np.isfinite(y)
    if mask.sum() < 10:
        raise ValueError(f"Too few valid points ({mask.sum()}) for '{formula_str}'")
    x = x[mask]
    y = y[mask]

    if noise_cfg:
        y = _apply_noise_to_y(np.asarray(y, dtype=np.float64), noise_cfg, seed=int(noise_seed))
    return x, y


def _build_universal_evolution_seed_graphs(
    x_values: np.ndarray,
    y_values: np.ndarray,
    detected_omegas: Optional[List[float]],
    max_seeds: int = 12,
) -> List[Dict[str, Any]]:
    """Build data-driven generic seed graphs for the pure C++ search path."""
    return build_seed_graphs_from_signal(
        x_values,
        y_values,
        detected_omegas=detected_omegas,
        max_seeds=max_seeds,
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _count_terms(formula: str) -> int:
    """Rough count of additive terms in a formula string."""
    if not formula:
        return 0
    # Split on top-level + or - (not inside parentheses)
    depth = 0
    terms = 1
    for ch in formula:
        if ch in '(':
            depth += 1
        elif ch in ')':
            depth -= 1
        elif ch in '+-' and depth == 0:
            terms += 1
    return terms


def _normalize_formula_text(formula: str) -> str:
    """Normalize common unicode/operator variants to a parser-friendly form."""
    if not formula:
        return formula
    formula = (
        formula
        .replace('²', '^2')
        .replace('³', '^3')
        .replace('·', '*')
        .replace('⋅', '*')
        .replace('×', '*')
        .replace('π', 'pi')
        .replace('√', 'sqrt')
        .replace('φ', 'phi')
        .replace('ω', 'omega')
    )
    return re.sub(r"\s+", "", formula)


def _simplify_formula_native(formula: str, int_tol: float, zero_tol: float) -> Optional[str]:
    """Simplify with the native C++ Phase-1 simplifier, if available."""
    try:
        cpp_dir = _REPO_ROOT / "glassbox" / "sr" / "cpp"
        if str(cpp_dir) not in sys.path:
            sys.path.insert(0, str(cpp_dir))

        import _core  # type: ignore

        simplified = _core.simplify_formula(
            formula,
            int_tol=float(int_tol),
            zero_tol=float(zero_tol),
            max_passes=6,
            use_nsimplify=True,
            use_identities=True,
            n_features=1,
        )
        if simplified and simplified not in {"N/A", "ERROR", "?"}:
            return str(simplified)
    except Exception:
        return None
    return None


def _postprocess_formula(formula: str) -> str:
    """Apply the same simplify_formula pipeline used by fast-path outputs.
    
    Uses wider snap tolerances than the fast-path because evolution-produced
    formulas have Ridge regression noise on coefficients (e.g. 0.9975 → 1.0).
    """
    normalized = _normalize_formula_text(formula)
    if not normalized or normalized in {"N/A", "ERROR", "?"}:
        return normalized

    evo_int_tol = 0.05
    evo_zero_tol = 1e-3

    native = _simplify_formula_native(normalized, evo_int_tol, evo_zero_tol)
    if native is not None:
        normalized = native

    try:
        try:
            from simplify_formula import simplify_onn_formula, SnapConfig, snap_formula_floats
        except ImportError:
            from scripts.simplify_formula import simplify_onn_formula, SnapConfig, snap_formula_floats

        formula_len = len(normalized)
        term_estimate = max(1, len([t for t in re.split(r'\s*[+-]\s*', normalized) if t.strip()]))
        too_complex_for_symbolic = formula_len > 500 or term_estimate > 24
        sympy_unsafe = "Piecewise(" in normalized or "Eq(" in normalized

        # Evolution formulas need wider tolerances than fast-path:
        # Ridge regression produces coefficients like 0.9975 instead of 1.0
        # and small spurious bias terms like 0.0001924 instead of 0.0
        evo_int_tol = 0.05    # snap 7.955 → 8, 2.04 → 2, etc.
        evo_zero_tol = 1e-3   # snap 0.0001924 → 0

        if too_complex_for_symbolic or sympy_unsafe:
            return snap_formula_floats(
                normalized,
                SnapConfig(int_tol=evo_int_tol, zero_tol=evo_zero_tol),
            )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=SymPyDeprecationWarning)
            warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"sympy\..*")
            warnings.filterwarnings("ignore", message=r"\s*Using non-Expr arguments in Mul.*")
            _, simplified_expr = simplify_onn_formula(
                normalized,
                int_tol=evo_int_tol,
                zero_tol=evo_zero_tol,
                use_nsimplify=(formula_len <= 300 and term_estimate <= 16),
            )
        return str(simplified_expr)
    except Exception:
        return normalized


def _evaluate_formula_mse(formula: str, x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Evaluate displayed formula against ground truth data and return MSE."""
    if not formula:
        return None
    normalized = _normalize_formula_text(formula)
    if not normalized or normalized in {"N/A", "ERROR", "?"}:
        return None

    y_pred = None
    try:
        y_pred = cfp._evaluate_formula_values(normalized, x)
    except Exception:
        y_pred = None

    if y_pred is None:
        try:
            fn = _parse_formula(normalized)
            y_pred = fn(x)
        except Exception:
            y_pred = None

    if y_pred is None:
        return None

    if y_pred.shape != y.shape:
        try:
            y_pred = np.asarray(y_pred, dtype=np.float64).reshape(y.shape)
        except Exception:
            return None

    mask = np.isfinite(y_pred) & np.isfinite(y)
    if mask.sum() < 10:
        return None

    mse = float(np.mean((y_pred[mask] - y[mask]) ** 2))
    if not math.isfinite(mse):
        return None
    return mse


_normalize_formula_text = bc.normalize_formula_text
_simplify_formula_native = bc.simplify_formula_native
_postprocess_formula = bc.postprocess_formula
_evaluate_formula_mse = bc.evaluate_formula_mse


def _postprocess_formula_for_benchmark(formula: str, x: np.ndarray, y: np.ndarray) -> Tuple[str, Dict[str, Any]]:
    return bc.postprocess_formula_with_fidelity_guard(
        formula,
        np.asarray(x, dtype=np.float64).reshape(-1, 1),
        y,
    )


def _select_score_mse(mse_display: Optional[float]) -> Optional[float]:
    """Choose MSE for scoring: use displayed-formula MSE only."""
    if mse_display is not None and math.isfinite(mse_display):
        return float(mse_display)
    return None


def _mse_divergence_stats(mse_display: Optional[float], mse_raw: Optional[float]) -> Dict[str, Any]:
    """Return absolute/relative raw-vs-display MSE divergence diagnostics."""
    out = {
        "mse_divergence_abs": None,
        "mse_divergence_rel": None,
        "mse_divergence_flag": False,
    }
    if (
        mse_display is None
        or mse_raw is None
        or not math.isfinite(mse_display)
        or not math.isfinite(mse_raw)
    ):
        return out

    abs_gap = abs(float(mse_raw) - float(mse_display))
    denom = max(abs(float(mse_display)), 1e-12)
    rel_gap = abs_gap / denom
    out["mse_divergence_abs"] = abs_gap
    out["mse_divergence_rel"] = rel_gap
    # Conservative threshold: flag when discrepancy is large enough to affect ranking.
    out["mse_divergence_flag"] = rel_gap > 0.10
    return out


def _display_eval_details(formula: str, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Evaluate a displayed formula and return MSE plus parser diagnostics."""
    X = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    y_pred, diagnostics = bc.evaluate_formula(formula, X, return_diagnostics=True)
    return {
        "mse": _evaluate_formula_mse(formula, x, y),
        "diagnostics": diagnostics,
    }


def _record_display_eval_failure(
    result: Dict[str, Any],
    formula: str,
    x: np.ndarray,
    y: np.ndarray,
) -> None:
    """Attach display-evaluation failure details without changing scoring policy."""
    details = _display_eval_details(formula, x, y)
    result["display_eval_diagnostics"] = details.get("diagnostics")
    if details.get("mse") is None and formula:
        result["formula_before_display_error"] = formula
        if result.get("error") is None:
            result["error"] = "formula_eval_failed"


# Noise-aware scoring context: set by main() when --noise is used so that
# structurally-correct formulas recovered under noise still count as EXACT.
# Defaults to clean semantics (1e-6 wall) when unset.
_NOISE_AWARE_EXACT_TOL: float = 1e-6


_NOISE_LEVEL: float = 0.0


def set_noise_aware_exact_tol(noise_level: float, y_var: float) -> None:
    """Configure noise-aware EXACT scoring from the injected noise level.

    Stores the noise level; the per-formula loop calls update_noise_aware_y_var
    with each target's variance so the EXACT threshold scales correctly with
    signal magnitude. Stays at the clean 1e-6 wall when noise_level == 0.
    """
    global _NOISE_AWARE_EXACT_TOL, _NOISE_LEVEL
    _NOISE_LEVEL = max(0.0, float(noise_level or 0.0))
    _NOISE_AWARE_EXACT_TOL = max(1e-6, 4.0 * (_NOISE_LEVEL ** 2) * float(y_var)) if _NOISE_LEVEL > 0 else 1e-6


def update_noise_aware_y_var(y_var: float) -> None:
    """Recompute the EXACT threshold for the current target's variance."""
    global _NOISE_AWARE_EXACT_TOL
    if _NOISE_LEVEL > 0:
        _NOISE_AWARE_EXACT_TOL = max(1e-6, 4.0 * (_NOISE_LEVEL ** 2) * float(y_var))
    else:
        _NOISE_AWARE_EXACT_TOL = 1e-6


def score_result(mse: float, formula: str) -> str:
    """Classify a result as EXACT / APPROX / LOOSE / FAIL."""
    if mse is None or not math.isfinite(mse):
        return "FAIL"
    n_terms = _count_terms(formula)
    if mse < _NOISE_AWARE_EXACT_TOL and n_terms <= 10:
        return "EXACT"
    if mse < 0.01:
        return "APPROX"
    if mse < 0.1:
        return "LOOSE"
    return "FAIL"


def _guided_evolution_decision(
    *,
    evolution_only: bool,
    with_evolution: bool,
    fp_result: Optional[Dict[str, Any]],
    mse: Optional[float],
    n_terms: int,
) -> Tuple[bool, str]:
    """Decide whether guided evolution should run and return a short reason.
    
    ADAPTIVE ROUTING: Instead of hardcoded entropy/margin thresholds that
    break when the classifier is retrained, we use relative fit quality
    (MSE normalized by signal variance) as the primary decision signal.
    
    This makes the decision robust to classifier architecture changes,
    retraining, or feature modifications.
    """
    if evolution_only:
        return True, "evolution_only"

    if not with_evolution:
        return False, "disabled"

    if fp_result is None:
        return True, "fast_path_not_applicable"

    if mse is None or not math.isfinite(mse):
        return True, "invalid_mse"

    # ── Tier 1: Perfect fit — skip evolution unconditionally ──
    if mse < 1e-12:
        return False, "fast_path_exact_zero"

    # ── Tier 2: Relative fit quality (ADAPTIVE) ──
    # Use MSE relative to signal variance. This is classifier-independent:
    # if fast-path explains >99.9999% of variance, evolution can't help.
    y_var = None
    details = fp_result.get("details", {})
    if isinstance(details, dict):
        y_var = details.get("y_variance")
    
    # Fallback: estimate from residual diagnostics
    if y_var is None:
        residual = fp_result.get("residual_diagnostics", {})
        if isinstance(residual, dict):
            y_var = residual.get("y_variance")
    
    if y_var is not None and y_var > 1e-10:
        relative_error = mse / y_var
        # Fast-path explains >99.999% of variance → skip evolution
        if relative_error < 1e-5 and n_terms <= 10:
            return False, "fast_path_high_r2"
    
    # ── Tier 3: Absolute MSE quality gate ──
    # Near-exact fit with reasonable complexity → skip
    if mse < 1e-7 and n_terms <= 6:
        return False, "fast_path_confident_exact"
    
    # Good fit (MSE < 1e-6) with reasonable complexity → skip
    if mse < 1e-6 and n_terms <= 10:
        return False, "fast_path_good_fit"

    # ── Tier 4: MSE too high — evolution needed ──
    # If the fit isn't very good, we need evolution
    if mse >= 1e-6:
        return True, "mse_above_exact"

    if n_terms > 10:
        return True, "formula_too_complex"

    # ── Tier 5: Residual analysis (classifier-independent) ──
    residual = fp_result.get("residual_diagnostics")
    if isinstance(residual, dict) and residual.get("residual_suspicious") is True:
        # If residuals show a clear periodic structure, evolution might find it
        return True, "suspicious_residual"

    return False, "fast_path_confident"



def _safe_r2_np(y_true, y_pred) -> Optional[float]:
    """Unweighted R2; None when prediction is invalid."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.shape != y_pred.shape or not np.all(np.isfinite(y_pred)):
        return None
    var = float(np.var(y_true))
    if var < 1e-15:
        return 1.0 if float(np.mean((y_pred - y_true) ** 2)) < 1e-15 else 0.0
    return float(1.0 - np.mean((y_pred - y_true) ** 2) / var)


def _attach_clean_target_metrics(
    result: Dict[str, Any],
    x: np.ndarray,
    y_clean: np.ndarray,
    *,
    exact_tol: float = 1e-6,
    acceptable_r2: float = 0.99,
) -> Dict[str, Any]:
    """Attach clean-target recovery metrics (independent of noisy EXACT band).

    Suite ``score`` / ``mse`` may use noisy labels under ``--noise``. Clean
    metrics answer: "does the discovered formula recover the true signal?"
    See ``noise_handling_audit.md``.
    """
    formula = str(result.get("formula_discovered") or "").strip()
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y_clean, dtype=np.float64).reshape(-1)

    mse_clean = None
    r2_clean = None
    if formula:
        mse_clean = _evaluate_formula_mse(formula, x_arr, y_arr)
        y_pred = cfp._evaluate_formula_values(formula, x_arr)
        if y_pred is not None:
            r2_clean = _safe_r2_np(y_arr, y_pred)

    n_terms = int(result.get("n_terms") or _count_terms(formula))
    recovery_exact = bool(
        mse_clean is not None
        and math.isfinite(float(mse_clean))
        and float(mse_clean) < float(exact_tol)
        and n_terms <= 10
    )
    recovery_acceptable = bool(
        r2_clean is not None
        and math.isfinite(float(r2_clean))
        and float(r2_clean) >= float(acceptable_r2)
    )

    result["mse_clean"] = (
        float(mse_clean) if mse_clean is not None and math.isfinite(float(mse_clean)) else None
    )
    result["r2_clean"] = (
        float(r2_clean) if r2_clean is not None and math.isfinite(float(r2_clean)) else None
    )
    result["recovery_exact"] = recovery_exact
    result["recovery_acceptable"] = recovery_acceptable
    return result


def _generate_xy_with_optional_noise(
    formula_str: str,
    x_min: float,
    x_max: float,
    n_samples: int,
    noise_cfg: Optional[Dict[str, Any]] = None,
    noise_seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(x, y_fit, y_clean)``. ``y_fit`` equals ``y_clean`` when no noise."""
    x_np, y_clean = _generate_data(formula_str, x_min, x_max, n_samples)
    y_clean = np.asarray(y_clean, dtype=np.float64).reshape(-1)
    if noise_cfg:
        y_fit = _apply_noise_to_y(y_clean, noise_cfg, seed=int(noise_seed))
    else:
        y_fit = y_clean
    return np.asarray(x_np, dtype=np.float64), np.asarray(y_fit, dtype=np.float64), y_clean


SCORE_SYMBOLS = {
    "EXACT":  "[PASS]",
    "APPROX": "[APPROX]",
    "LOOSE":  "[LOOSE]",
    "FAIL":   "[FAIL]",
}

_SCORE_POINTS = {"FAIL": 0, "LOOSE": 1, "APPROX": 2, "EXACT": 3}


_PROPOSER_CACHE = {}
def _get_proposer(path: str, device: str):
    if path not in _PROPOSER_CACHE:
        from glassbox.universal_proposer import load_universal_proposer_checkpoint
        _PROPOSER_CACHE[path] = load_universal_proposer_checkpoint(path, device=device)
    return _PROPOSER_CACHE[path]

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_formula(
    formula_str: str,
    x_range: Tuple[float, float],
    classifier_path: str,
    n_samples: int = 300,
    device: Optional[str] = None,
    timeout: float = 60.0,
    with_evolution: bool = False,
    evolution_only: bool = False,
    proposer_path: Optional[str] = None,
    disable_proposer: bool = False,
    evolution_generations: int = 150,
    evolution_population: int = 50,
    trust_proposer_plan: bool = False,
    exact_match_backend: str = "auto",
    exact_match_min_gpu_work: int = 250_000,
    exact_match_max_combos: int = 50_000,
    noise_cfg: Optional[Dict[str, Any]] = None,
    noise_seed: int = 0,
) -> Dict[str, Any]:
    """Run fast-path and/or guided evolution on a single formula."""
    x_min, x_max = x_range
    result = {
        "formula_target": formula_str,
        "x_range": list(x_range),
        "formula_discovered": "",
        "mse": None,
        "mse_raw": None,
        "mse_display": None,
        "mse_clean": None,
        "r2_clean": None,
        "recovery_exact": False,
        "recovery_acceptable": False,
        "time": None,
        "score": "FAIL",
        "error": None,
        "n_terms": 0,
        "uncertainty": None,
        "residual_diagnostics": None,
        "candidate_formulas": None,
        "fast_path_candidate_formulas": None,
        "evolution_seed_candidates": None,
        "proposer_candidate_formulas": None,
        "winning_stage": None,
        "engine_raw_mse": None,
        "formula_before_postprocess_mse": None,
        "formula_after_postprocess_mse": None,
        "score_mse": None,
        "display_eval_diagnostics": None,
        "formula_before_display_error": None,
        "mse_divergence_abs": None,
        "mse_divergence_rel": None,
        "mse_divergence_flag": False,
        "exact_match_diagnostics": None,
    }

    try:
        # Generate data (keep clean labels for recovery; fit on optional noise)
        x_np, y_np, y_clean = _generate_xy_with_optional_noise(
            formula_str, x_min, x_max, n_samples,
            noise_cfg=noise_cfg, noise_seed=noise_seed,
        )

        # Early return for constant targets. This avoids evolution-only
        # degenerating to formula "0" on flat signals.
        y_std = np.std(y_np)
        if y_std < 1e-10:
            const_val = float(np.mean(y_np))
            if abs(const_val - round(const_val)) < 1e-6:
                formula = str(int(round(const_val)))
            else:
                formula = f"{const_val:.6g}"
            formula, guard = _postprocess_formula_for_benchmark(formula, x_np, y_np)
            result["formula_discovered"] = formula
            result["postprocess_guard"] = guard
            result["mse_raw"] = 0.0
            result["engine_raw_mse"] = 0.0
            result["formula_before_postprocess_mse"] = guard.get("postprocess_raw_mse")
            result["formula_after_postprocess_mse"] = guard.get("postprocess_processed_mse")
            result["mse_display"] = 0.0
            result["mse"] = 0.0
            result["score_mse"] = 0.0
            result["winning_stage"] = "constant_shortcut"
            result["time"] = 0.0
            result["n_terms"] = _count_terms(formula)
            result["uncertainty"] = cfp._prediction_uncertainty_metrics({'identity': 1.0})
            y_pred = cfp._evaluate_formula_values(formula, x_np)
            result["residual_diagnostics"] = (
                cfp._residual_diagnostics(y_np, y_pred, x_np)
                if y_pred is not None
                else None
            )
            result["exact_match_diagnostics"] = {
                    "backend_requested": exact_match_backend,
                    "fallback_reason": "constant_shortcut",
                    "max_combos": int(exact_match_max_combos),
                    "torch_used": False,
                    "gpu_used": False,
                }
            result["candidate_formulas"] = [{
                "formula": formula,
                "mse": 0.0,
                "score": 0.0,
                "n_nonzero": result["n_terms"],
                "active_terms": ["1"],
                "alpha": 0.0,
            }]
            result["score"] = score_result(0.0, formula)
            _attach_clean_target_metrics(result, x_np, y_clean)
            return result

        # Convert to torch
        x_t = torch.tensor(x_np, dtype=torch.float32)
        y_t = torch.tensor(y_np, dtype=torch.float32)

        # Detect frequencies
        x_2d = x_t.reshape(-1, 1)
        y_2d = y_t.reshape(-1, 1)
        try:
            detected_omegas = detect_dominant_frequency(x_2d, y_2d, n_frequencies=3)
        except Exception:
            detected_omegas = None

        # Run fast path unless evolution-only mode is requested.
        fp_result = None
        elapsed = 0.0
        if not evolution_only:
            t0 = time.time()
            fp_result = run_fast_path(
                x_2d, y_2d,
                classifier_path=classifier_path,
                detected_omegas=detected_omegas,
                op_constraints=None,
                auto_expand=True,
                device=device,
                exact_match_threads=1,
                exact_match_enabled=True,
                exact_match_max_basis=150,
                exact_match_backend=exact_match_backend,
                exact_match_min_gpu_work=exact_match_min_gpu_work,
                exact_match_max_combos=exact_match_max_combos,
                simplify_formula_output=False,
                noise_level=(noise_cfg or {}).get("noise_level", 0.0) if noise_cfg else 0.0,
            )
            elapsed = time.time() - t0

            if fp_result is None:
                result["error"] = "fast_path_not_applicable"
                result["time"] = elapsed
            else:
                fp_formula = fp_result.get("formula", "")
                result["formula_discovered"], result["postprocess_guard"] = _postprocess_formula_for_benchmark(
                    fp_formula,
                    x_np,
                    y_np,
                )
                result["mse_raw"] = fp_result.get("mse", float("inf"))
                result["engine_raw_mse"] = fp_result.get("mse", float("inf"))
                result["formula_before_postprocess_mse"] = result["postprocess_guard"].get("postprocess_raw_mse")
                result["formula_after_postprocess_mse"] = result["postprocess_guard"].get("postprocess_processed_mse")
                display_details = _display_eval_details(result["formula_discovered"], x_np, y_np)
                result["mse_display"] = display_details["mse"]
                result["display_eval_diagnostics"] = display_details["diagnostics"]
                result["mse"] = _select_score_mse(result["mse_display"])
                result["score_mse"] = result["mse"]
                result.update(_mse_divergence_stats(result["mse_display"], result["mse_raw"]))
                result["uncertainty"] = fp_result.get("uncertainty")
                result["candidate_formulas"] = fp_result.get("candidate_formulas")
                result["fast_path_candidate_formulas"] = fp_result.get("candidate_formulas")
                result["winning_stage"] = "fast_path"
                if result["formula_discovered"] and result["mse_display"] is None:
                    result["formula_before_display_error"] = result["formula_discovered"]
                    if result["error"] is None:
                        result["error"] = "formula_eval_failed"
                result["time"] = elapsed
                str_term_count = _count_terms(result["formula_discovered"])
                details = fp_result.get("details", {}) if isinstance(fp_result, dict) else {}
                structural_terms = details.get("n_nonzero", 0)
                simplified_terms = details.get("n_nonzero_simplified", 0)
                result["exact_match_diagnostics"] = details.get("exact_match_diagnostics")
                result["n_terms"] = max(
                    int(str_term_count),
                    int(structural_terms) if structural_terms is not None else 0,
                    int(simplified_terms) if simplified_terms is not None else 0,
                )
        else:
            result["time"] = 0.0

        # -----------------------------------------------------------------
        # Guided Evolution (latest path with beam search)
        # -----------------------------------------------------------------
        # Ensure we have y_variance for the adaptive decision
        if fp_result and "details" in fp_result and "y_variance" not in fp_result["details"]:
             fp_result["details"]["y_variance"] = float(np.var(y_np))

        should_run_guided, guided_reason = _guided_evolution_decision(
            evolution_only=evolution_only,
            with_evolution=with_evolution,
            fp_result=fp_result,
            mse=result["mse"],
            n_terms=result["n_terms"],
        )

        if should_run_guided:
            if evolution_only:
                print("\n  [Evolution Only] Skipping fast-path. Running guided evolution (beam search)...")
            elif fp_result is None:
                print("\n  [Latest Path] Fast-path not applicable. Running guided evolution (beam search)...")
            else:
                mse_str = f"{result['mse']:.2e}" if result['mse'] is not None else "N/A"
                print(
                    f"\n  [Latest Path] Fast-path candidate (MSE={mse_str}, reason={guided_reason}). "
                    "Running guided evolution (beam search)..."
                )

            # Build hints from fast-path output when available; otherwise,
            # use conservative defaults and FFT frequencies.
            operator_hints = fp_result.get("operator_hints", {}) if fp_result else {}
            operator_hints = dict(operator_hints) if operator_hints else {}
            operator_hints["operators"] = set(operator_hints.get("operators", set()))
            operator_hints["frequencies"] = list(operator_hints.get("frequencies", detected_omegas or []))
            operator_hints["powers"] = list(operator_hints.get("powers", []))
            operator_hints["has_rational"] = bool(operator_hints.get("has_rational", False))
            operator_hints["has_exp_decay"] = bool(operator_hints.get("has_exp_decay", False))
            operator_hints["active_terms"] = list(operator_hints.get("active_terms", []))
            operator_hints["uncertainty"] = fp_result.get("uncertainty") if fp_result else None

            candidate_formulas = None
            proposer_confidence = 0.5
            dynamic_gens = int(max(20, evolution_generations))
            dynamic_pop = int(max(20, evolution_population))
            guided_plan: Dict[str, Any] = {}
            # Per-formula deterministic C++ seed (same target + range → same seed).
            try:
                guided_plan["random_seed"] = bc.formula_benchmark_seed(
                    formula_str,
                    x_range,
                    base_seed=int(noise_seed or 0),
                    n_samples=n_samples,
                )
            except Exception:
                pass
            
            if not disable_proposer and proposer_path:
                try:
                    from glassbox.universal_proposer import propose_fpip_v2_from_xy
                    model = _get_proposer(proposer_path, device)
                    payload = propose_fpip_v2_from_xy(
                        model, x=x_np, y=y_np, top_k=5, device=device
                    )
                    if payload and payload.get("valid"):
                        seq_unc = payload.get("sequence_uncertainty", {})
                        search_plan = payload.get("search_plan", {})
                        if not isinstance(search_plan, dict):
                            search_plan = {}
                        
                        # Calculate mathematical difficulty [0.0, 1.0]
                        entropy = float(seq_unc.get("entropy") or 0.0)
                        margin = float(seq_unc.get("margin") or 0.0)
                        difficulty = np.clip((entropy / 1.5) + (1.0 - margin), 0.0, 1.0)
                        proposer_confidence = float(np.clip(1.0 - difficulty, 0.0, 1.0))
                        
                        formula_seed = guided_plan.get("random_seed")
                        dynamic_gens, dynamic_pop, guided_plan = _planned_guided_budget(
                            search_plan,
                            evolution_generations,
                            evolution_population,
                            trust_plan=trust_proposer_plan,
                        )
                        if formula_seed is not None and "random_seed" not in guided_plan:
                            guided_plan["random_seed"] = formula_seed
                        
                        proposer_priors = payload.get("operator_priors", {})
                        if proposer_priors:
                            for op, prob in proposer_priors.items():
                                if prob > 0.15:
                                    operator_hints["operators"].add(op)
                        proposer_skeletons = payload.get("candidate_skeletons", [])
                        result["proposer_candidate_formulas"] = list(proposer_skeletons or [])
                        candidate_formulas = []
                        if fp_result and fp_result.get("formula"):
                            candidate_formulas.append({
                                "formula": fp_result["formula"],
                                "mse": fp_result.get("mse", float("inf")),
                                "from_fast_path": True,
                            })
                        for cand in proposer_skeletons:
                            f_str = cand.get("formula", "")
                            if f_str:
                                active = [t.strip() for t in f_str.replace("-", "+").split("+") if t.strip()]
                                candidate_formulas.append({
                                    "formula": f_str,
                                    "mse": cand.get("mse", float("inf")),
                                    "score": cand.get("score", 0.0),
                                    "active_terms": active,
                                    "from_proposer": True
                                })
                        if candidate_formulas:
                            finite_candidate_mses = [
                                float(c.get("mse"))
                                for c in candidate_formulas
                                if c.get("mse") is not None and math.isfinite(float(c.get("mse")))
                            ]
                            best_candidate_mse = min(finite_candidate_mses) if finite_candidate_mses else float("inf")
                            baseline_for_conf = result["mse"] if result["mse"] is not None else float("inf")
                            plan_signals = search_plan.get("signals", {}) if isinstance(search_plan, dict) else {}
                            best_rel_mse = plan_signals.get("best_relative_mse")
                            rel_confident = (
                                best_rel_mse is not None
                                and math.isfinite(float(best_rel_mse))
                                and float(best_rel_mse) < 1e-4
                            )
                            if (
                                seq_unc.get("confident") is True
                                and (
                                    best_candidate_mse <= baseline_for_conf * 1.05
                                    or rel_confident
                                )
                            ):
                                proposer_confidence = max(proposer_confidence, 0.9)
                            print(f"\n  [Universal Proposer] Active FPIPv2 metadata injected!")
                            strategy = search_plan.get("strategy", "legacy")
                            planned_difficulty = search_plan.get("difficulty", difficulty)
                            planner_mode = "trusted" if trust_proposer_plan else "bounded"
                            print(
                                f"  [Universal Proposer] Strategy: {strategy} ({planner_mode}) "
                                f"difficulty={float(planned_difficulty):.2f} -> "
                                f"Budget: {dynamic_gens} gens, {dynamic_pop} pop"
                            )
                            if guided_plan:
                                print(f"  [Universal Proposer] Search plan: {guided_plan}")
                except Exception as e:
                    print(f"\n  [Universal Proposer] Warning: execution failed: {e}")

            if candidate_formulas is None and fp_result and fp_result.get("formula"):
                candidate_formulas = [{
                    "formula": fp_result["formula"],
                    "mse": fp_result.get("mse", float("inf")),
                    "from_fast_path": True,
                }]

            if fp_result:
                fp_seed_candidates = list(fp_result.get("candidate_formulas") or [])
                if fp_seed_candidates:
                    if candidate_formulas is None:
                        candidate_formulas = []
                    seen_seed_formulas = {
                        str(c.get("formula", "")).replace(" ", "")
                        for c in candidate_formulas
                        if isinstance(c, dict)
                    }
                    for cand in fp_seed_candidates:
                        if not isinstance(cand, dict):
                            continue
                        formula_seed = str(cand.get("formula", "") or "").strip()
                        if not formula_seed:
                            continue
                        seed_key = formula_seed.replace(" ", "")
                        if seed_key in seen_seed_formulas:
                            continue
                        merged = dict(cand)
                        merged.setdefault("from_fast_path_candidate_pool", True)
                        candidate_formulas.append(merged)
                        seen_seed_formulas.add(seed_key)
            result["evolution_seed_candidates"] = list(candidate_formulas or [])

            t1 = time.time()
            try:
                guided_plan = dict(guided_plan or {})
                remaining_timeout = max(1, int(float(timeout) - float(t1 - (t0 if "t0" in locals() else t1))))
                guided_plan.setdefault("timeout_seconds", remaining_timeout)
                guided_result = run_guided_evolution(
                    x_2d,
                    y_2d,
                    operator_hints,
                    generations=dynamic_gens,
                    population_size=dynamic_pop,
                    device=device,
                    candidate_formulas=candidate_formulas,
                    confidence=proposer_confidence,
                    search_plan=guided_plan,
                )

                guided_elapsed = time.time() - t1
                base_elapsed = result["time"] if result["time"] is not None else elapsed
                result["time"] = base_elapsed + guided_elapsed

                if guided_result and guided_result.get("formula"):
                    guided_raw_formula = guided_result.get("formula", "")
                    guided_formula, guided_guard = _postprocess_formula_for_benchmark(
                        guided_raw_formula,
                        x_np,
                        y_np,
                    )
                    guided_mse_raw = guided_result.get("raw_mse", guided_result.get("mse", float("inf")))
                    guided_display_details = _display_eval_details(guided_formula, x_np, y_np)
                    guided_mse_display = guided_display_details["mse"]
                    guided_mse_score = _select_score_mse(guided_mse_display)
                    guided_mse_for_compare = (
                        guided_mse_score if guided_mse_score is not None else float("inf")
                    )
                    baseline_mse = result["mse"] if result["mse"] is not None else float("inf")

                    if evolution_only or fp_result is None or guided_mse_for_compare < baseline_mse:
                        result["formula_discovered"] = guided_formula
                        result["mse_raw"] = guided_mse_raw
                        result["engine_raw_mse"] = guided_mse_raw
                        result["formula_before_postprocess_mse"] = guided_guard.get("postprocess_raw_mse")
                        result["formula_after_postprocess_mse"] = guided_guard.get("postprocess_processed_mse")
                        result["mse_display"] = guided_mse_display
                        result["mse"] = guided_mse_for_compare
                        result["score_mse"] = result["mse"]
                        result["display_eval_diagnostics"] = guided_display_details["diagnostics"]
                        result.update(_mse_divergence_stats(result["mse_display"], result["mse_raw"]))
                        result["uncertainty"] = fp_result.get("uncertainty") if fp_result else None
                        result["candidate_formulas"] = result["evolution_seed_candidates"]
                        result["winning_stage"] = "guided_evolution"
                        if result["formula_discovered"] and result["mse_display"] is None:
                            result["formula_before_display_error"] = result["formula_discovered"]
                            result["error"] = "formula_eval_failed"
                        else:
                            result["error"] = None
                        result["n_terms"] = _count_terms(guided_formula)
                        result["postprocess_guard"] = guided_guard
                elif evolution_only or fp_result is None:
                    result["error"] = "guided_evolution_failed"
            except Exception as guided_err:
                if evolution_only or fp_result is None:
                    result["error"] = f"guided_evolution_error: {guided_err}"
                print(f"  [Guided Evolution Error: {guided_err}]", end="")

        if result["formula_discovered"]:
            if result["mse_display"] is None:
                display_details = _display_eval_details(result["formula_discovered"], x_np, y_np)
                result["mse_display"] = display_details["mse"]
                result["display_eval_diagnostics"] = display_details["diagnostics"]
            result["mse"] = _select_score_mse(result["mse_display"])
            result["score_mse"] = result["mse"]
            result.update(_mse_divergence_stats(result["mse_display"], result["mse_raw"]))
            y_pred = cfp._evaluate_formula_values(result["formula_discovered"], x_np)
            result["residual_diagnostics"] = (
                cfp._residual_diagnostics(y_np, y_pred, x_np)
                if y_pred is not None
                else None
            )
            if result["formula_discovered"] and result["mse_display"] is None:
                result["formula_before_display_error"] = result["formula_discovered"]
                if result["error"] is None:
                    result["error"] = "formula_eval_failed"

    except Exception as e:
        result["error"] = str(e)
        import traceback; traceback.print_exc()
        result["time"] = 0.0

    # Score (noisy-fit band) + clean-target recovery metrics
    result["score_mse"] = result["mse"]
    if result.get("error") and "timeout" in str(result.get("error")).lower():
        result["score"] = "FAIL"
    else:
        result["score"] = score_result(result["mse"], result["formula_discovered"])
    if "y_clean" in locals():
        _attach_clean_target_metrics(result, x_np, y_clean)
    return result


def run_formula_cpp_evolution(
    formula_str: str,
    x_range: Tuple[float, float],
    n_samples: int = 300,
    pop_size: int = 100,
    generations: int = 1000,
    device: Optional[str] = None,
    timeout: Optional[float] = None,
    noise_cfg: Optional[Dict[str, Any]] = None,
    noise_seed: int = 0,
) -> Dict[str, Any]:
    """Run pure C++ evolution on a single formula (no classifier fast-path)."""
    import sys
    from pathlib import Path
    
    x_min, x_max = x_range
    result = {
        "formula_target": formula_str,
        "x_range": list(x_range),
        "formula_discovered": "",
        "mse": None,
        "mse_raw": None,
        "mse_display": None,
        "mse_clean": None,
        "r2_clean": None,
        "recovery_exact": False,
        "recovery_acceptable": False,
        "time": None,
        "score": "FAIL",
        "error": None,
        "n_terms": 0,
        "residual_diagnostics": None,
        "engine_raw_mse": None,
        "formula_before_postprocess_mse": None,
        "formula_after_postprocess_mse": None,
        "score_mse": None,
        "display_eval_diagnostics": None,
        "formula_before_display_error": None,
        "mse_divergence_abs": None,
        "mse_divergence_rel": None,
        "mse_divergence_flag": False,
    }
    
    try:
        # Import C++ backend
        cpp_dir = Path(__file__).resolve().parent.parent / 'glassbox' / 'sr' / 'cpp'
        if str(cpp_dir) not in sys.path:
            sys.path.insert(0, str(cpp_dir))
        import _core
        
        # Generate data (keep clean labels for recovery; fit on optional noise)
        x_np, y_np, y_clean = _generate_xy_with_optional_noise(
            formula_str, x_min, x_max, n_samples,
            noise_cfg=noise_cfg, noise_seed=noise_seed,
        )
        
        # Early return for constant signals (evolution can't find pure constants)
        y_std = np.std(y_np)
        if y_std < 1e-10:
            elapsed = 0.0
            const_val = float(np.mean(y_np))
            if abs(const_val - round(const_val)) < 1e-6:
                formula = str(int(round(const_val)))
            else:
                formula = f"{const_val:.6g}"
            formula, guard = _postprocess_formula_for_benchmark(formula, x_np, y_np)
            result["formula_discovered"] = formula
            result["postprocess_guard"] = guard
            result["mse_raw"] = 0.0
            result["engine_raw_mse"] = 0.0
            result["formula_before_postprocess_mse"] = guard.get("postprocess_raw_mse")
            result["formula_after_postprocess_mse"] = guard.get("postprocess_processed_mse")
            result["mse_display"] = 0.0
            result["mse"] = 0.0
            result["score_mse"] = 0.0
            result["time"] = elapsed
            result["n_terms"] = 1
            result["residual_diagnostics"] = {
                "residual_mse": 0.0,
                "residual_skewness": 0.0,
                "residual_excess_kurtosis": 0.0,
                "residual_spectral_peak_ratio": 0.0,
                "residual_holdout_edge_mse": 0.0,
                "residual_holdout_core_mse": 0.0,
                "residual_holdout_ratio": 0.0,
                "residual_suspicious": False,
            }
            result["score"] = score_result(0.0, formula)
            _attach_clean_target_metrics(result, x_np, y_clean)
            return result
        
        X_list = [x_np]
        
        # FFT frequency detection
        x_t = torch.tensor(x_np, dtype=torch.float32).reshape(-1, 1)
        y_t = torch.tensor(y_np, dtype=torch.float32).reshape(-1, 1)
        try:
            detected_omegas = detect_dominant_frequency(x_t, y_t, n_frequencies=3)
            if detected_omegas and detected_omegas[0] == 1.0:
                detected_omegas = []
        except Exception:
            detected_omegas = []

        seed_graphs_py = _build_universal_evolution_seed_graphs(
            x_np,
            y_np,
            detected_omegas,
            max_seeds=12,
        )
        
        # Run pure C++ evolution (deterministic per formula+range when possible)
        try:
            cpp_seed = bc.formula_benchmark_seed(
                formula_str, x_range, base_seed=int(noise_seed or 0), n_samples=n_samples
            )
        except Exception:
            cpp_seed = -1
        t0 = time.time()
        evo_kwargs = dict(
            X_list=X_list,
            y=y_np,
            pop_size=pop_size,
            generations=generations,
            early_stop_mse=1e-10,
            seed_omegas=detected_omegas or [],
            seed_graphs_py=seed_graphs_py,
        )
        if cpp_seed >= 0:
            evo_kwargs["random_seed"] = int(cpp_seed)
        cpp_result = _core.run_evolution(**evo_kwargs)
        elapsed = time.time() - t0
        
        result["formula_discovered"], result["postprocess_guard"] = _postprocess_formula_for_benchmark(
            cpp_result.get("formula", ""),
            x_np,
            y_np,
        )
        result["mse_raw"] = cpp_result.get("best_mse", float("inf"))
        result["engine_raw_mse"] = cpp_result.get("best_mse", float("inf"))
        result["formula_before_postprocess_mse"] = result["postprocess_guard"].get("postprocess_raw_mse")
        result["formula_after_postprocess_mse"] = result["postprocess_guard"].get("postprocess_processed_mse")
        display_details = _display_eval_details(result["formula_discovered"], x_np, y_np)
        result["mse_display"] = display_details["mse"]
        result["display_eval_diagnostics"] = display_details["diagnostics"]
        result["mse"] = _select_score_mse(result["mse_display"])
        result["score_mse"] = result["mse"]
        result.update(_mse_divergence_stats(result["mse_display"], result["mse_raw"]))
        y_pred = cfp._evaluate_formula_values(result["formula_discovered"], x_np)
        result["residual_diagnostics"] = (
            cfp._residual_diagnostics(y_np, y_pred, x_np)
            if y_pred is not None
            else None
        )
        if result["formula_discovered"] and result["mse_display"] is None:
            result["formula_before_display_error"] = result["formula_discovered"]
            if result["error"] is None:
                result["error"] = "formula_eval_failed"
        result["time"] = elapsed
        result["n_terms"] = _count_terms(result["formula_discovered"])
        if timeout is not None and elapsed > float(timeout):
            result["error"] = f"timeout_exceeded after {elapsed:.1f}s"
            result["score"] = "FAIL"
        
    except ImportError as e:
        result["error"] = f"C++ backend not available: {e}"
        result["time"] = 0.0
    except Exception as e:
        result["error"] = str(e)
        import traceback; traceback.print_exc()
        result["time"] = 0.0
    
    result["score_mse"] = result["mse"]
    if result.get("error") and "timeout" in str(result.get("error")).lower():
        result["score"] = "FAIL"
    else:
        result["score"] = score_result(result["mse"], result["formula_discovered"])
    if "y_clean" in locals():
        _attach_clean_target_metrics(result, x_np, y_clean)
    return result


def run_formula_specialist_regressor(
    formula_str: str,
    x_range: Tuple[float, float],
    classifier_path: str,
    proposer_path: Optional[str],
    n_samples: int = 300,
    device: Optional[str] = None,
    timeout: float = 60.0,
    population_size: int = 50,
    generations: int = 150,
    specialist_enabled: bool = True,
    specialist_diagnostics: bool = True,
    specialist_composition: bool = True,
    specialist_residual: bool = False,
    specialist_vault: bool = True,
    specialist_inception: bool = False,
    exact_match_backend: str = "auto",
    exact_match_min_gpu_work: int = 250_000,
    exact_match_max_combos: int = 50_000,
    noise_cfg: Optional[Dict[str, Any]] = None,
    noise_seed: int = 0,
) -> Dict[str, Any]:
    """Run the sklearn regressor path so specialist composition/boosting is measurable."""
    x_min, x_max = x_range
    diagnostics_enabled = bool(specialist_enabled and specialist_diagnostics)
    composition_enabled = bool(specialist_enabled and specialist_composition and diagnostics_enabled)
    residual_enabled = bool(specialist_enabled and specialist_residual and composition_enabled)
    vault_enabled = bool(specialist_enabled and specialist_vault and composition_enabled)
    inception_enabled = bool(specialist_enabled and specialist_inception)
    result = {
        "formula_target": formula_str,
        "x_range": list(x_range),
        "formula_discovered": "",
        "mse": None,
        "mse_raw": None,
        "mse_display": None,
        "mse_clean": None,
        "r2_clean": None,
        "recovery_exact": False,
        "recovery_acceptable": False,
        "time": None,
        "score": "FAIL",
        "error": None,
        "n_terms": 0,
        "uncertainty": None,
        "residual_diagnostics": None,
        "candidate_formulas": None,
        "engine_raw_mse": None,
        "formula_before_postprocess_mse": None,
        "formula_after_postprocess_mse": None,
        "score_mse": None,
        "display_eval_diagnostics": None,
        "formula_before_display_error": None,
        "mse_divergence_abs": None,
        "mse_divergence_rel": None,
        "mse_divergence_flag": False,
        "exact_match_diagnostics": None,
        "benchmark_path": "specialist_regressor" if specialist_enabled else "regressor_baseline",
        "specialist_enabled": bool(specialist_enabled),
        "specialist_phase_config": {
            "diagnostics": diagnostics_enabled,
            "composition": composition_enabled,
            "residual": residual_enabled,
            "vault": vault_enabled,
            "inception": inception_enabled,
        },
        "specialist_track": None,
        "has_composed_seeds": False,
        "composition_candidates_accepted": False,
        "composition_candidate_count": 0,
        "composition_seeded_evolution": False,
        "composition_won_final_selection": False,
        "composition_improved_mse": False,
        "boosting_attempted": False,
        "boosting_improved": False,
        "boosting_stage_count": 0,
        "boosting_diagnostics": None,
        "phase_timings": None,
        "formula_eval_count": 0,
        "formula_eval_cache_hits": 0,
        "formula_eval_cache_size": 0,
        "specialist_diagnostics": None,
        "specialist_composition_screening": None,
    }

    try:
        from glassbox.sr.sklearn_wrapper import GlassboxRegressor

        x_np, y_np, y_clean = _generate_xy_with_optional_noise(
            formula_str, x_min, x_max, n_samples,
            noise_cfg=noise_cfg, noise_seed=noise_seed,
        )
        X = np.asarray(x_np, dtype=np.float64).reshape(-1, 1)
        y = np.asarray(y_np, dtype=np.float64).reshape(-1)

        y_std = np.std(y)
        if y_std < 1e-10:
            const_val = float(np.mean(y))
            formula = str(int(round(const_val))) if abs(const_val - round(const_val)) < 1e-6 else f"{const_val:.6g}"
            formula, guard = _postprocess_formula_for_benchmark(formula, x_np, y_np)
            result["formula_discovered"] = formula
            result["postprocess_guard"] = guard
            result["mse_raw"] = 0.0
            result["engine_raw_mse"] = 0.0
            result["formula_before_postprocess_mse"] = guard.get("postprocess_raw_mse")
            result["formula_after_postprocess_mse"] = guard.get("postprocess_processed_mse")
            result["mse_display"] = 0.0
            result["mse"] = 0.0
            result["score_mse"] = 0.0
            result["time"] = 0.0
            result["n_terms"] = _count_terms(formula)
            result["exact_match_diagnostics"] = {
                "backend_requested": exact_match_backend,
                "fallback_reason": "constant_shortcut",
                "max_combos": int(exact_match_max_combos),
                "torch_used": False,
                "gpu_used": False,
            }
            result["score"] = score_result(0.0, formula)
            _attach_clean_target_metrics(result, x_np, y_clean)
            return result

        reg = GlassboxRegressor(
            population_size=max(20, int(population_size)),
            generations=max(20, int(generations)),
            timeout=max(1, int(timeout)),
            classifier_path=classifier_path,
            universal_proposer_path=proposer_path or "models/universal_proposer_multi.pt",
            use_universal_proposer=bool(proposer_path),
            universal_proposer_shadow_mode=False,
            universal_proposer_log_routing=False,
            blackbox_mode=True,
            blackbox_feature_selection=True,
            blackbox_standardize=True,
            enable_specialist_screening_diagnostics=diagnostics_enabled,
            enable_specialist_composition_screening=composition_enabled,
            enable_residual_stage=residual_enabled,
            enable_specialist_vault_memory=vault_enabled,
            enable_inception_reuse=inception_enabled,
            use_guided_evolution=True,
            use_fast_path=True,
            multi_start_runs=1,
            device=device,
            exact_match_backend=exact_match_backend,
            exact_match_min_gpu_work=exact_match_min_gpu_work,
            exact_match_max_combos=exact_match_max_combos,
            random_state=0,
        )

        t0 = time.time()
        reg.fit(X, y)
        elapsed = time.time() - t0

        formula, guard = _postprocess_formula_for_benchmark(reg.get_formula(), x_np, y_np)
        result["formula_discovered"] = formula
        result["postprocess_guard"] = guard
        result["mse_raw"] = _finite_float(getattr(reg, "best_mse_", None), float("inf"))
        result["engine_raw_mse"] = result["mse_raw"]
        result["formula_before_postprocess_mse"] = guard.get("postprocess_raw_mse")
        result["formula_after_postprocess_mse"] = guard.get("postprocess_processed_mse")
        display_details = _display_eval_details(formula, x_np, y_np)
        result["mse_display"] = display_details["mse"]
        result["display_eval_diagnostics"] = display_details["diagnostics"]
        result["mse"] = _select_score_mse(result["mse_display"])
        result["score_mse"] = result["mse"]
        result.update(_mse_divergence_stats(result["mse_display"], result["mse_raw"]))
        result["time"] = elapsed
        result["n_terms"] = _count_terms(formula)

        y_pred = cfp._evaluate_formula_values(formula, x_np)
        result["residual_diagnostics"] = (
            cfp._residual_diagnostics(y_np, y_pred, x_np)
            if y_pred is not None
            else None
        )
        if formula and result["mse_display"] is None:
            result["formula_before_display_error"] = formula
            result["error"] = "formula_eval_failed"

        result.update(bc.specialist_metadata_from_estimator(reg))
        result["exact_match_diagnostics"] = getattr(reg, "fast_path_exact_match_diagnostics_", None)

    except Exception as e:
        result["error"] = str(e)
        traceback.print_exc()
        result["time"] = 0.0

    result["score_mse"] = result["mse"]
    result["score"] = score_result(result["mse"], result["formula_discovered"])
    if "y_clean" in locals():
        _attach_clean_target_metrics(result, x_np, y_clean)
    return result


def _timeout_result(formula_str: str, x_range: Tuple[float, float], elapsed: float, error: str) -> Dict[str, Any]:
    return {
        "formula_target": formula_str,
        "x_range": list(x_range),
        "formula_discovered": "",
        "mse": None,
        "mse_raw": None,
        "mse_display": None,
        "time": float(elapsed),
        "score": "FAIL",
        "error": error,
        "n_terms": 0,
        "score_mse": None,
        "display_eval_diagnostics": None,
    }


def _benchmark_worker_cli(input_path: str, output_path: str) -> int:
    try:
        with open(input_path, "rb") as fh:
            call_spec = pickle.load(fh)
        fn = globals()[call_spec["function"]]
        payload = {"status": "ok", "result": fn(**call_spec["kwargs"])}
    except Exception as exc:
        payload = {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
    with open(output_path, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return 0


def run_benchmark_call_with_timeout(
    function_name: str,
    kwargs: Dict[str, Any],
    timeout_seconds: Optional[float],
) -> Dict[str, Any]:
    """Run a benchmark formula call in a child process with hard wall-clock timeout."""
    timeout_seconds = float(timeout_seconds or 0.0)
    if timeout_seconds <= 0:
        return globals()[function_name](**kwargs)

    worker_dir = Path("scratch") / "benchmark_workers"
    worker_dir.mkdir(parents=True, exist_ok=True)
    unique = f"{os.getpid()}_{time.time_ns()}"
    input_path = worker_dir / f"{unique}.in.pkl"
    output_path = worker_dir / f"{unique}.out.pkl"
    t0 = time.time()
    with open(input_path, "wb") as fh:
        pickle.dump({"function": function_name, "kwargs": kwargs}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    try:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--_benchmark-worker",
            str(input_path),
            str(output_path),
        ]
        completed = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent.parent), timeout=timeout_seconds)
        elapsed = time.time() - t0
        if completed.returncode != 0:
            return _timeout_result(
                kwargs.get("formula_str", ""),
                kwargs.get("x_range", (0.0, 0.0)),
                elapsed,
                f"worker exited with code {completed.returncode}",
            )
        if not output_path.exists():
            return _timeout_result(
                kwargs.get("formula_str", ""),
                kwargs.get("x_range", (0.0, 0.0)),
                elapsed,
                "worker produced no result",
            )
        with open(output_path, "rb") as fh:
            payload = pickle.load(fh)
        if payload.get("status") == "ok":
            result = payload["result"]
            elapsed = float(result.get("time") or (time.time() - t0))
            if elapsed > timeout_seconds:
                result["error"] = f"timeout_exceeded after {elapsed:.1f}s"
                result["score"] = "FAIL"
            return result

        error = payload.get("error", "worker error")
        return _timeout_result(kwargs.get("formula_str", ""), kwargs.get("x_range", (0.0, 0.0)), elapsed, error)
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        return _timeout_result(
            kwargs.get("formula_str", ""),
            kwargs.get("x_range", (0.0, 0.0)),
            elapsed,
            f"hard timeout after {elapsed:.1f}s",
        )
    finally:
        for path in (input_path, output_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _tier_summary(tier_results: List[Dict]) -> Dict[str, int]:
    """Count scores in a tier."""
    counts = {"EXACT": 0, "APPROX": 0, "LOOSE": 0, "FAIL": 0, "total": len(tier_results)}
    for r in tier_results:
        s = r.get("score", "FAIL")
        counts[s] = counts.get(s, 0) + 1
    return counts


def print_summary(all_results: Dict[int, List[Dict]]) -> None:
    """Print a console summary table."""
    print("\n" + "=" * 90)
    print("GLASSBOX SR BENCHMARK RESULTS")
    print("=" * 90)

    header = f"{'Tier':<4} {'Name':<24} {'Total':>5} {'✅ Exact':>9} {'🟡 Approx':>10} {'🟠 Loose':>9} {'❌ Fail':>8} {'Score':>7}"
    print(header)
    print("-" * 90)

    grand = {"EXACT": 0, "APPROX": 0, "LOOSE": 0, "FAIL": 0, "total": 0}
    for tier_num in sorted(all_results.keys()):
        tier_name = ALL_TIERS[tier_num][0]
        results = all_results[tier_num]
        s = _tier_summary(results)

        pct = (s["EXACT"] / s["total"] * 100) if s["total"] > 0 else 0
        print(
            f"  {tier_num:<3} {tier_name:<24} {s['total']:>4}  "
            f"{s['EXACT']:>6}    {s['APPROX']:>6}    {s['LOOSE']:>6}   {s['FAIL']:>5}   {pct:>5.1f}%"
        )

        for k in ("EXACT", "APPROX", "LOOSE", "FAIL", "total"):
            grand[k] += s[k]

    print("-" * 90)
    pct = (grand["EXACT"] / grand["total"] * 100) if grand["total"] > 0 else 0
    print(
        f"  {'ALL':<27} {grand['total']:>4}  "
        f"{grand['EXACT']:>6}    {grand['APPROX']:>6}    {grand['LOOSE']:>6}   {grand['FAIL']:>5}   {pct:>5.1f}%"
    )
    print("=" * 90)

    # Also print overall weighted score (exact=3, approx=2, loose=1, fail=0)
    total_points = grand["EXACT"] * 3 + grand["APPROX"] * 2 + grand["LOOSE"] * 1
    max_points = grand["total"] * 3
    weighted_pct = (total_points / max_points * 100) if max_points > 0 else 0
    print(f"\nWeighted Score: {total_points}/{max_points} ({weighted_pct:.1f}%)")
    print(f"  Scoring: EXACT=3pts, APPROX=2pts, LOOSE=1pt, FAIL=0pts\n")


def generate_markdown_report(
    all_results: Dict[int, List[Dict]],
    output_path: Path,
    classifier_path: str,
    total_time: float,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a detailed Markdown report."""
    lines = []
    lines.append("# Glassbox SR Benchmark Report\n")
    lines.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"**Classifier**: `{classifier_path}`\n")
    lines.append(f"**Total runtime**: {total_time:.1f}s\n")
    if metadata:
        lines.append(f"**Mode**: `{metadata.get('mode', '')}`\n")
        noise_meta = metadata.get("noise") or {}
        if noise_meta and noise_meta.get("preset") not in (None, "none"):
            lines.append(
                f"**Noise**: `{noise_meta.get('preset')}` ({noise_meta.get('noise_type')}, "
                f"level={noise_meta.get('noise_level')}, seed={noise_meta.get('seed')})\n"
            )
            lines.append(
                "**Note**: `Score/EXACT` uses noisy-label display MSE with a noise-aware "
                "band. Prefer `CleanMSE` / `R2clean` / `Recov` for structure recovery "
                "(see `noise_handling_audit.md`).\n"
            )
        lines.append(f"**Python ABI**: `{metadata.get('python_abi', '')}`\n")
        lines.append(f"**C++ core**: `{metadata.get('cpp_core_status', {}).get('status', 'unknown')}`\n")
        if metadata.get("seed") is not None:
            lines.append(f"**Seed**: `{metadata.get('seed')}`\n")

    # Overall summary table
    lines.append("\n## Summary\n")
    lines.append("| Tier | Name | Total | ✅ Exact | 🟡 Approx | 🟠 Loose | ❌ Fail | Exact % |")
    lines.append("|------|------|-------|---------|----------|---------|--------|---------|")

    grand = {"EXACT": 0, "APPROX": 0, "LOOSE": 0, "FAIL": 0, "total": 0}
    for tier_num in sorted(all_results.keys()):
        tier_name = ALL_TIERS[tier_num][0]
        s = _tier_summary(all_results[tier_num])
        pct = (s["EXACT"] / s["total"] * 100) if s["total"] > 0 else 0
        lines.append(
            f"| {tier_num} | {tier_name} | {s['total']} | {s['EXACT']} | {s['APPROX']} | "
            f"{s['LOOSE']} | {s['FAIL']} | {pct:.0f}% |"
        )
        for k in ("EXACT", "APPROX", "LOOSE", "FAIL", "total"):
            grand[k] += s[k]

    pct = (grand["EXACT"] / grand["total"] * 100) if grand["total"] > 0 else 0
    lines.append(
        f"| **ALL** | — | **{grand['total']}** | **{grand['EXACT']}** | **{grand['APPROX']}** | "
        f"**{grand['LOOSE']}** | **{grand['FAIL']}** | **{pct:.0f}%** |"
    )

    # Per-tier details
    for tier_num in sorted(all_results.keys()):
        tier_name = ALL_TIERS[tier_num][0]
        results = all_results[tier_num]
        lines.append(f"\n## Tier {tier_num}: {tier_name}\n")
        lines.append("| # | Score | Target | Discovered | MSE(score) | MSE(raw) | MSE(display) | CleanMSE | R2clean | Recov | Drift | Stage | Time | Terms |")
        lines.append("|---|-------|--------|------------|------------|----------|--------------|----------|---------|-------|-------|-------|------|-------|")

        for i, r in enumerate(results, 1):
            sym = SCORE_SYMBOLS.get(r["score"], "?")
            target = r["formula_target"]
            disc = r.get("formula_discovered", "")
            if len(disc) > 50:
                disc = disc[:47] + "..."
            mse_s = f"{r['mse']:.2e}" if r["mse"] is not None and math.isfinite(r["mse"]) else "—"
            mse_raw = r.get("mse_raw")
            mse_raw_s = f"{mse_raw:.2e}" if mse_raw is not None and math.isfinite(mse_raw) else "—"
            mse_display = r.get("mse_display")
            mse_display_s = (
                f"{mse_display:.2e}" if mse_display is not None and math.isfinite(mse_display) else "—"
            )
            mse_clean = r.get("mse_clean")
            mse_clean_s = (
                f"{mse_clean:.2e}" if mse_clean is not None and math.isfinite(mse_clean) else "—"
            )
            r2_clean = r.get("r2_clean")
            r2_clean_s = (
                f"{r2_clean:.3f}" if r2_clean is not None and math.isfinite(r2_clean) else "—"
            )
            if r.get("recovery_exact"):
                recov_s = "exact"
            elif r.get("recovery_acceptable"):
                recov_s = "ok"
            else:
                recov_s = "—"
            time_s = f"{r['time']:.2f}s" if r["time"] is not None else "—"
            drift_rel = r.get("mse_divergence_rel")
            if drift_rel is not None and math.isfinite(drift_rel):
                drift_s = f"{drift_rel:.1e}" if r.get("mse_divergence_flag") else ""
            else:
                drift_s = ""
            stage = r.get("winning_stage") or r.get("benchmark_path") or ""
            if r.get("composition_won_final_selection"):
                stage = "composition"
            elif r.get("composition_seeded_evolution") and stage:
                stage = f"{stage}+comp"
            elif r.get("composition_seeded_evolution"):
                stage = "composition_seeded"
            if len(stage) > 28:
                stage = stage[:25] + "..."
            n_terms = r.get("n_terms", 0)
            err = r.get("error", "")
            if err:
                disc = f"ERROR: {err[:40]}"
            lines.append(
                f"| {i} | {sym} | `{target}` | `{disc}` | {mse_s} | {mse_raw_s} | {mse_display_s} | {mse_clean_s} | {r2_clean_s} | {recov_s} | {drift_s} | {stage} | {time_s} | {n_terms} |"
            )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_json_results(
    all_results: Dict[int, List[Dict]],
    output_path: Path,
    classifier_path: str,
    total_time: float,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save full results to JSON."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "classifier": classifier_path,
        "total_time_seconds": round(total_time, 2),
        "metadata": metadata or {},
        "tiers": {},
    }

    grand = {"EXACT": 0, "APPROX": 0, "LOOSE": 0, "FAIL": 0, "total": 0}
    for tier_num in sorted(all_results.keys()):
        tier_name = ALL_TIERS[tier_num][0]
        results = all_results[tier_num]
        s = _tier_summary(results)
        data["tiers"][str(tier_num)] = {
            "name": tier_name,
            "summary": s,
            "results": results,
        }
        for k in ("EXACT", "APPROX", "LOOSE", "FAIL", "total"):
            grand[k] += s[k]

    data["overall"] = grand
    total_points = grand["EXACT"] * 3 + grand["APPROX"] * 2 + grand["LOOSE"] * 1
    max_points = grand["total"] * 3
    data["weighted_score"] = {
        "points": total_points,
        "max_points": max_points,
        "percentage": round(total_points / max_points * 100, 1) if max_points > 0 else 0,
    }

    output_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _cpp_core_status() -> Dict[str, Any]:
    core, reason = cfp._load_cpp_core()
    built_extensions: List[str] = []
    try:
        cpp_dir = _REPO_ROOT / "glassbox" / "sr" / "cpp"
        built_extensions = sorted(p.name for p in cpp_dir.glob("_core.*"))
    except Exception:
        built_extensions = []
    return {
        "status": "available" if core is not None else "unavailable",
        "reason": reason,
        "built_extensions": built_extensions,
    }


def _collect_report_diagnostics(all_results: Dict[int, List[Dict]]) -> Dict[str, int]:
    diagnostics = {
        "exact_match_combo_cap_count": 0,
        "bounded_sparse_beam_count": 0,
        "display_eval_failure_count": 0,
        "mse_divergence_flag_count": 0,
        "candidate_governor_count": 0,
    }
    for results in all_results.values():
        for result in results:
            exact_diag = result.get("exact_match_diagnostics")
            if isinstance(exact_diag, dict):
                if exact_diag.get("combo_count", 0) and exact_diag.get("max_combos", 0):
                    if exact_diag.get("combo_count", 0) > exact_diag.get("max_combos", 0):
                        diagnostics["exact_match_combo_cap_count"] += 1
                if str(exact_diag.get("fallback_reason", "")).startswith("bounded_sparse_beam"):
                    diagnostics["bounded_sparse_beam_count"] += 1
            if result.get("display_eval_diagnostics") and result.get("mse_display") is None:
                diagnostics["display_eval_failure_count"] += 1
            if result.get("mse_divergence_flag"):
                diagnostics["mse_divergence_flag_count"] += 1
            details = result.get("details")
            if isinstance(details, dict) and details.get("candidate_governor"):
                diagnostics["candidate_governor_count"] += 1
            for cand in result.get("candidate_formulas") or []:
                if isinstance(cand, dict) and cand.get("governor"):
                    diagnostics["candidate_governor_count"] += 1
                    break
    return diagnostics


def _build_run_metadata(
    args: argparse.Namespace,
    *,
    device: str,
    mode: str,
    tiers_to_run: List[int],
    total_formulas: int,
    all_results: Optional[Dict[int, List[Dict]]] = None,
) -> Dict[str, Any]:
    metadata = {
        "mode": mode,
        "device": device,
        "seed": args.seed,
        "runs": int(args.runs),
        "tiers": list(tiers_to_run),
        "total_formulas": int(total_formulas),
        "n_samples": int(args.n_samples),
        "python_version": platform.python_version(),
        "python_abi": getattr(sys.implementation, "cache_tag", "unknown"),
        "platform": platform.platform(),
        "classifier_model": args.classifier_model,
        "proposer_model": None if args.disable_proposer else args.proposer_model,
        "exact_match_backend": args.exact_match_backend,
        "exact_match_min_gpu_work": int(args.exact_match_min_gpu_work),
        "exact_match_max_combos": int(args.exact_match_max_combos),
        "cpp_core_status": _cpp_core_status(),
        "noise": {
            "preset": getattr(args, "noise", "none"),
            "noise_type": _noise_type_for_preset(getattr(args, "noise", "none")),
            "noise_level": _noise_level_for_preset(getattr(args, "noise", "none")),
            "seed": int(getattr(args, "noise_seed", 0)),
        },
        "specialist_phase_config": {
            "specialist_regressor": bool(args.specialist_regressor),
            "specialist_baseline": bool(args.specialist_baseline),
            "diagnostics": not bool(args.disable_specialist_diagnostics),
            "composition": not bool(args.disable_specialist_composition),
            "residual": bool(args.enable_specialist_residual or args.specialist_full),
            "vault": not bool(args.disable_specialist_vault),
            "inception": bool(args.enable_specialist_inception or args.specialist_full),
        },
    }
    if all_results is not None:
        metadata["report_diagnostics"] = _collect_report_diagnostics(all_results)
    return metadata


def _flatten_json_report_results(report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    flattened: Dict[str, Dict[str, Any]] = {}
    tiers = report.get("tiers", {})
    if not isinstance(tiers, dict):
        return flattened
    for tier_payload in tiers.values():
        if not isinstance(tier_payload, dict):
            continue
        for result in tier_payload.get("results", []) or []:
            if not isinstance(result, dict):
                continue
            key = str(result.get("formula_target") or result.get("human_name") or "").strip()
            if key:
                flattened[key] = result
    return flattened


def _flatten_current_results(all_results: Dict[int, List[Dict]]) -> Dict[str, Dict[str, Any]]:
    flattened: Dict[str, Dict[str, Any]] = {}
    for results in all_results.values():
        for result in results:
            key = str(result.get("formula_target") or result.get("human_name") or "").strip()
            if key:
                flattened[key] = result
    return flattened


def compare_benchmark_results(
    previous_report_path: Path,
    current_results: Dict[int, List[Dict]],
) -> Dict[str, Any]:
    previous_report = json.loads(previous_report_path.read_text(encoding="utf-8"))
    previous = _flatten_json_report_results(previous_report)
    current = _flatten_current_results(current_results)
    transitions: List[Dict[str, Any]] = []
    summary = {
        "previous_only": 0,
        "current_only": 0,
        "same": 0,
        "improved": 0,
        "regressed": 0,
        "changed": 0,
    }

    for key, cur in sorted(current.items()):
        prev = previous.get(key)
        if prev is None:
            summary["current_only"] += 1
            transitions.append({
                "formula": key,
                "previous_score": None,
                "current_score": cur.get("score"),
                "direction": "new",
            })
            continue

        prev_score = str(prev.get("score", "FAIL"))
        cur_score = str(cur.get("score", "FAIL"))
        prev_points = _SCORE_POINTS.get(prev_score, 0)
        cur_points = _SCORE_POINTS.get(cur_score, 0)
        if cur_points > prev_points:
            direction = "improved"
            summary["improved"] += 1
        elif cur_points < prev_points:
            direction = "regressed"
            summary["regressed"] += 1
        else:
            direction = "same"
            summary["same"] += 1

        prev_mse = prev.get("mse")
        cur_mse = cur.get("mse")
        prev_formula = prev.get("formula_discovered")
        cur_formula = cur.get("formula_discovered")
        mse_changed = False
        try:
            if prev_mse is None or cur_mse is None:
                mse_changed = prev_mse != cur_mse
            else:
                mse_changed = not math.isclose(float(prev_mse), float(cur_mse), rel_tol=1e-9, abs_tol=1e-12)
        except Exception:
            mse_changed = prev_mse != cur_mse
        formula_changed = prev_formula != cur_formula
        if direction != "same" or prev_score != cur_score or mse_changed or formula_changed:
            summary["changed"] += 1
        transitions.append({
            "formula": key,
            "previous_score": prev_score,
            "current_score": cur_score,
            "direction": direction,
            "previous_mse": prev_mse,
            "current_mse": cur_mse,
            "previous_formula": prev_formula,
            "current_formula": cur_formula,
            "mse_changed": mse_changed,
            "formula_changed": formula_changed,
        })

    for key in sorted(set(previous) - set(current)):
        summary["previous_only"] += 1
        transitions.append({
            "formula": key,
            "previous_score": previous[key].get("score"),
            "current_score": None,
            "direction": "missing",
        })

    return {
        "previous_report": str(previous_report_path),
        "summary": summary,
        "transitions": transitions,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Glassbox SR Benchmark Suite — evaluate symbolic regression across 200 formulas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/benchmark_suite.py                                    # Full suite
  python scripts/benchmark_suite.py --tier 1                           # Tier 1 only
  python scripts/benchmark_suite.py --tier 1 --tier 2                  # Tiers 1 & 2
  python scripts/benchmark_suite.py --classifier-model models/v3.pt   # Custom model
  python scripts/benchmark_suite.py --output-dir results/              # Custom output
  python scripts/benchmark_suite.py --compare-to results/old.json      # Compare reports
        """,
    )
    parser.add_argument(
        "--classifier-model", type=str, default="models/curve_classifier_multi.pt",
        help="Path to the curve classifier model (default: models/curve_classifier_multi.pt)",
    )
    parser.add_argument(
        "--proposer-model", type=str, default="models/universal_proposer_multi.pt",
        help="Path to the universal neural proposer model (default: models/universal_proposer_multi.pt)",
    )
    parser.add_argument(
        "--disable-proposer", action="store_true",
        help="Disable the neural proposer and rely purely on legacy classifier hints",
    )
    parser.add_argument(
        "--trust-proposer-plan", action="store_true",
        help=(
            "Let the universal proposer control guided-evolution budget and search-plan "
            "knobs instead of applying benchmark clamps"
        ),
    )
    parser.add_argument(
        "--with-evolution", action="store_true",
        help="Run latest guided evolution (beam-search path) when fast-path is not exact",
    )
    parser.add_argument(
        "--evolution-only", action="store_true",
        help="Skip fast-path and run latest guided evolution (beam-search path) for every formula",
    )
    parser.add_argument(
        "--specialist-regressor", action="store_true",
        help="Run formulas through GlassboxRegressor so specialist composition/boosting is measured",
    )
    parser.add_argument(
        "--specialist-baseline", action="store_true",
        help="With --specialist-regressor, disable specialist screening/composition/residual stages for A/B comparison",
    )
    parser.add_argument(
        "--disable-specialist-diagnostics", action="store_true",
        help="With --specialist-regressor, skip specialist pair/segment diagnostics",
    )
    parser.add_argument(
        "--disable-specialist-composition", action="store_true",
        help="With --specialist-regressor, keep diagnostics but skip specialist composition proposals",
    )
    parser.add_argument(
        "--enable-specialist-residual", action="store_true",
        help="With --specialist-regressor, enable the residual symbolic stage for accepted compositions",
    )
    parser.add_argument(
        "--disable-specialist-vault", action="store_true",
        help="With --specialist-regressor, disable cross-run specialist vault memory",
    )
    parser.add_argument(
        "--enable-specialist-inception", action="store_true",
        help="With --specialist-regressor, enable inception/subexpression reuse",
    )
    parser.add_argument(
        "--specialist-full", action="store_true",
        help="With --specialist-regressor, enable all specialist phases including residual and inception",
    )
    parser.add_argument(
        "--tier", type=int, action="append", default=None, dest="tiers",
        help="Run only specific tier(s). Can be repeated: --tier 1 --tier 2",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory for JSON and Markdown reports (default: results/)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=300,
        help="Number of data points per formula (default: 300)",
    )
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
        help="Device for classifier inference (default: auto)",
    )
    parser.add_argument(
        "--exact-match-backend",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "torch_cuda", "torch", "numpy"],
        help=(
            "Backend for fast-path exact symbolic matching. In auto mode, CUDA is used "
            "only when device resolves to cuda and the work threshold is met."
        ),
    )
    parser.add_argument(
        "--exact-match-min-gpu-work",
        type=int,
        default=250_000,
        help="Minimum estimated exact-match work before auto mode uses CUDA (default: 250000)",
    )
    parser.add_argument(
        "--exact-match-max-combos",
        type=int,
        default=50_000,
        help="Maximum pair/triple exact-match combinations before falling back to sparse search (default: 50000)",
    )
    parser.add_argument(
        "--timeout", type=float, default=60.0,
        help="Engine timeout budget per formula in seconds (default: 60)",
    )
    parser.add_argument(
        "--hard-timeout", action="store_true",
        help="Run each formula in a subprocess and enforce --timeout as a hard wall-clock limit",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-formula output, only show summary",
    )
    parser.add_argument(
        "--formula", type=str, default=None,
        help="Run a single formula by searching all tiers (e.g., --formula 'sin(x)')",
    )
    parser.add_argument(
        "--cpp-evolution-only", action="store_true",
        help="Skip classifier fast-path; use pure C++ evolution for every formula",
    )
    parser.add_argument(
        "--pop-size", type=int, default=100,
        help="Population size for C++ evolution (default: 100, used with --cpp-evolution-only)",
    )
    parser.add_argument(
        "--generations", type=int, default=1000,
        help="Generations for C++ evolution (default: 1000, used with --cpp-evolution-only)",
    )
    parser.add_argument(
        "--guided-generations", type=int, default=150,
        help="Generations for guided evolution in --with-evolution/--evolution-only mode (default: 150)",
    )
    parser.add_argument(
        "--guided-pop-size", type=int, default=50,
        help="Population per island for guided evolution in --with-evolution/--evolution-only mode (default: 50)",
    )
    parser.add_argument(
        "--runs", type=int, default=1,
        help="Number of times to run each formula. Returns best result. (default: 1)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Set Python, NumPy, and Torch random seeds for repeatable benchmark runs",
    )
    parser.add_argument(
        "--noise", type=str, default="none",
        choices=list(NOISE_PRESETS.keys()),
        help=(
            "Inject noise into the (otherwise clean) benchmark targets so "
            "noise-handling can be measured. Presets: none, low (0.1%% RMS), "
            "medium (1%% RMS), high (10%% RMS), outliers (3%% spikes). "
            "Default: none (clean benchmark)."
        ),
    )
    parser.add_argument(
        "--noise-seed", type=int, default=0,
        help="Seed for --noise injection so runs are reproducible (default: 0).",
    )
    parser.add_argument(
        "--compare-to", type=str, default=None,
        help="Compare this run against a prior benchmark JSON report",
    )
    args = parser.parse_args()

    exclusive_modes = sum(bool(v) for v in (args.cpp_evolution_only, args.evolution_only, args.specialist_regressor))
    if exclusive_modes > 1:
        print("Error: --cpp-evolution-only, --evolution-only, and --specialist-regressor are mutually exclusive.")
        sys.exit(1)
    if args.specialist_baseline and not args.specialist_regressor:
        print("Error: --specialist-baseline requires --specialist-regressor.")
        sys.exit(1)
    specialist_phase_flags = (
        args.disable_specialist_diagnostics,
        args.disable_specialist_composition,
        args.enable_specialist_residual,
        args.disable_specialist_vault,
        args.enable_specialist_inception,
        args.specialist_full,
    )
    if any(specialist_phase_flags) and not args.specialist_regressor:
        print("Error: specialist phase flags require --specialist-regressor.")
        sys.exit(1)

    specialist_diagnostics = not args.disable_specialist_diagnostics
    specialist_composition = specialist_diagnostics and not args.disable_specialist_composition
    specialist_residual = bool(args.enable_specialist_residual or args.specialist_full)
    specialist_vault = not args.disable_specialist_vault
    specialist_inception = bool(args.enable_specialist_inception or args.specialist_full)
    if args.specialist_full:
        specialist_diagnostics = True
        specialist_composition = True
        specialist_vault = True

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    compare_to_path = Path(args.compare_to) if args.compare_to else None
    if compare_to_path is not None and not compare_to_path.exists():
        print(f"Error: --compare-to report does not exist: {compare_to_path}")
        sys.exit(2)

    # Resolve --noise preset into a concrete noise config (None == clean).
    noise_cfg = NOISE_PRESETS.get(args.noise)
    noise_seed = int(args.noise_seed)

    # Configure noise-aware EXACT scoring so structurally-correct formulas
    # recovered under noise are still classed EXACT (not penalised down to
    # APPROX by residual noise MSE). Uses a representative y variance; the
    # fast-path also recomputes per-formula, but the global default keeps
    # score_result consistent for non-fast-path formulas too.
    if noise_cfg is not None:
        set_noise_aware_exact_tol(float(noise_cfg.get("noise_level", 0.0)), 1.0)
    else:
        set_noise_aware_exact_tol(0.0, 1.0)

    # Validate report output before doing expensive formula work.
    output_dir = Path(args.output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        probe_path = output_dir / ".benchmark_write_probe"
        probe_path.write_text("", encoding="utf-8")
        probe_path.unlink(missing_ok=True)
    except Exception as exc:
        print(f"Error: cannot write benchmark reports to '{output_dir}': {exc}")
        sys.exit(2)

    # Resolve device
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    args.exact_match_min_gpu_work = max(0, int(args.exact_match_min_gpu_work))
    args.exact_match_max_combos = max(0, int(args.exact_match_max_combos))

    # Determine which tiers to run
    tiers_to_run = args.tiers if args.tiers else list(ALL_TIERS.keys())

    # Validate tiers
    for t in tiers_to_run:
        if t not in ALL_TIERS:
            print(f"Error: Tier {t} does not exist. Valid: {list(ALL_TIERS.keys())}")
            sys.exit(1)

    # Filter by --formula if specified
    if args.formula:
        found = False
        for tier_num in list(tiers_to_run):
            tier_name, formulas = ALL_TIERS[tier_num]
            matching = [f for f in formulas if args.formula.lower() in f[0].lower()]
            if matching:
                ALL_TIERS[tier_num] = (tier_name, matching)
                found = True
            else:
                if tier_num in tiers_to_run:
                    tiers_to_run.remove(tier_num)
        if not found:
            print(f"No formula matching '{args.formula}' found in any tier.")
            sys.exit(1)

    # Count total formulas
    total_formulas = sum(len(ALL_TIERS[t][1]) for t in tiers_to_run)

    print("=" * 90)
    print("GLASSBOX SR BENCHMARK SUITE")
    print("=" * 90)
    if args.cpp_evolution_only:
        mode_str = "Pure C++ Evolution (No Classifier/Proposer)"
    elif args.specialist_regressor:
        mode_str = "GlassboxRegressor Specialist" if not args.specialist_baseline else "GlassboxRegressor Baseline"
    elif args.evolution_only:
        mode_str = "Guided Evolution Only (latest path)"
    else:
        mode_str = "Hybrid Fast-Path"
    print(f"  Mode:        {mode_str}")
    if args.cpp_evolution_only:
        print(f"  Pop size:    {args.pop_size}")
        print(f"  Generations: {args.generations}")
    elif args.evolution_only:
        print("  Strategy:    guided beam-search evolution")
    elif args.specialist_regressor:
        print("  Strategy:    sklearn regressor path")
        print(f"  Specialist:  {'disabled baseline' if args.specialist_baseline else 'enabled'}")
        if not args.specialist_baseline:
            phase_txt = ", ".join(
                name
                for name, enabled in (
                    ("diagnostics", specialist_diagnostics),
                    ("composition", specialist_composition),
                    ("residual", specialist_residual),
                    ("vault", specialist_vault),
                    ("inception", specialist_inception),
                )
                if enabled
            ) or "none"
            print(f"  Phases:      {phase_txt}")
    else:
        print(f"  Classifier:  {args.classifier_model}")
        proposer_txt = args.proposer_model if not args.disable_proposer else "DISABLED"
        print(f"  Proposer:    {proposer_txt}")
        if not args.disable_proposer and args.trust_proposer_plan:
            print("  Planner:     universal proposer search plan")
        print("  Strategy:    optimized")
    print(f"  Device:      {device}")
    if not args.cpp_evolution_only:
        print(
            f"  Exact match: {args.exact_match_backend} "
            f"(min GPU work={args.exact_match_min_gpu_work}, max combos={args.exact_match_max_combos})"
        )
    print(f"  Tiers:       {tiers_to_run}")
    print(f"  Formulas:    {total_formulas}")
    print(f"  Samples/ea:  {args.n_samples}")
    if noise_cfg is not None:
        nt = noise_cfg.get("noise_type", "?")
        nl = noise_cfg.get("noise_level", 0.0)
        print(f"  Noise:       {args.noise} ({nt}, level={nl}, seed={noise_seed})")
    else:
        print(f"  Noise:       none (clean)")
    if args.seed is not None:
        print(f"  Seed:        {args.seed}")
    if compare_to_path is not None:
        print(f"  Compare to:  {compare_to_path}")
    print("=" * 90)

    # Run benchmark
    all_results: Dict[int, List[Dict]] = {}
    formula_idx = 0
    t_start = time.time()

    for tier_num in sorted(tiers_to_run):
        tier_name, formulas = ALL_TIERS[tier_num]
        print(f"\n{'-' * 90}")
        print(f"  TIER {tier_num}: {tier_name}  ({len(formulas)} formulas)")
        print(f"{'-' * 90}")

        tier_results = []
        for formula_str, human_name, x_range in formulas:
            formula_idx += 1
            # Refresh noise-aware EXACT threshold for this target's variance so
            # structurally-correct formulas under noise are scored EXACT.
            try:
                _probe_x, _probe_y = _generate_data(
                    formula_str, x_range[0], x_range[1], args.n_samples,
                )
                update_noise_aware_y_var(float(np.var(_probe_y)) if _probe_y.size else 1.0)
            except Exception:
                pass

            if not args.quiet:
                try:
                    print(f"  [{formula_idx}/{total_formulas}] {human_name:<30} ", end="", flush=True)
                except UnicodeEncodeError:
                    human_name = human_name.encode('ascii', 'ignore').decode('ascii')
                    print(f"  [{formula_idx}/{total_formulas}] {human_name:<30} ", end="", flush=True)

            best_result = None
            
            for _ in range(args.runs):
                if args.cpp_evolution_only:
                    call_name = "run_formula_cpp_evolution"
                    call_kwargs = {
                        "formula_str": formula_str,
                        "x_range": x_range,
                        "n_samples": args.n_samples,
                        "pop_size": args.pop_size,
                        "generations": args.generations,
                        "device": device,
                        "timeout": args.timeout,
                        "noise_cfg": noise_cfg,
                        "noise_seed": noise_seed,
                    }
                elif args.specialist_regressor:
                    call_name = "run_formula_specialist_regressor"
                    call_kwargs = {
                        "formula_str": formula_str,
                        "x_range": x_range,
                        "classifier_path": args.classifier_model,
                        "proposer_path": None if args.disable_proposer else args.proposer_model,
                        "n_samples": args.n_samples,
                        "device": device,
                        "timeout": args.timeout,
                        "population_size": args.guided_pop_size,
                        "generations": args.guided_generations,
                        "specialist_enabled": not args.specialist_baseline,
                        "specialist_diagnostics": specialist_diagnostics,
                        "specialist_composition": specialist_composition,
                        "specialist_residual": specialist_residual,
                        "specialist_vault": specialist_vault,
                        "specialist_inception": specialist_inception,
                        "exact_match_backend": args.exact_match_backend,
                        "exact_match_min_gpu_work": args.exact_match_min_gpu_work,
                        "exact_match_max_combos": args.exact_match_max_combos,
                        "noise_cfg": noise_cfg,
                        "noise_seed": noise_seed,
                    }
                else:
                    call_name = "run_formula"
                    call_kwargs = {
                        "formula_str": formula_str,
                        "x_range": x_range,
                        "classifier_path": args.classifier_model,
                        "n_samples": args.n_samples,
                        "device": device,
                        "timeout": args.timeout,
                        "with_evolution": args.with_evolution,
                        "evolution_only": args.evolution_only,
                        "proposer_path": args.proposer_model,
                        "disable_proposer": args.disable_proposer,
                        "evolution_generations": args.guided_generations,
                        "evolution_population": args.guided_pop_size,
                        "trust_proposer_plan": args.trust_proposer_plan,
                        "exact_match_backend": args.exact_match_backend,
                        "exact_match_min_gpu_work": args.exact_match_min_gpu_work,
                        "exact_match_max_combos": args.exact_match_max_combos,
                        "noise_cfg": noise_cfg,
                        "noise_seed": noise_seed,
                    }
                if args.hard_timeout:
                    result = run_benchmark_call_with_timeout(call_name, call_kwargs, args.timeout)
                else:
                    result = globals()[call_name](**call_kwargs)
                
                # Keep the best result based on displayed MSE (or just any valid MSE if best_result is None)
                if best_result is None:
                    best_result = result
                else:
                    # Compare MSE
                    best_mse = best_result.get("mse")
                    curr_mse = result.get("mse")
                    if best_mse is None or math.isnan(best_mse):
                        best_result = result
                    elif curr_mse is not None and not math.isnan(curr_mse) and curr_mse < best_mse:
                        best_result = result
                        
            result = best_result
            result["human_name"] = human_name
            tier_results.append(result)

            if not args.quiet:
                sym = SCORE_SYMBOLS.get(result["score"], "?")
                mse_s = f"MSE={result['mse']:.2e}" if result["mse"] is not None and math.isfinite(result["mse"]) else "N/A     "
                mse_raw = result.get("mse_raw")
                if (
                    mse_raw is not None and
                    math.isfinite(mse_raw) and
                    result["mse"] is not None and
                    math.isfinite(result["mse"])
                ):
                    drift = abs(math.log10((mse_raw + 1e-30) / (result["mse"] + 1e-30)))
                    if drift > 1.0:
                        mse_s = f"{mse_s} (raw={mse_raw:.2e})"
                time_s = f"{result['time']:.2f}s" if result["time"] is not None else "—    "
                disc = result.get("formula_discovered", "")
                if len(disc) > 40:
                    disc = disc[:37] + "..."
                print(f"{sym}  {mse_s}  {time_s}  {disc}")

        all_results[tier_num] = tier_results

        # Print tier subtotal
        s = _tier_summary(tier_results)
        pct = (s["EXACT"] / s["total"] * 100) if s["total"] > 0 else 0
        print(f"  -- Tier {tier_num} subtotal: {s['EXACT']}/{s['total']} exact ({pct:.0f}%)")

    total_time = time.time() - t_start

    # Print summary
    print_summary(all_results)

    # Save reports
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    metadata = _build_run_metadata(
        args,
        device=device,
        mode=mode_str,
        tiers_to_run=tiers_to_run,
        total_formulas=total_formulas,
        all_results=all_results,
    )

    json_path = output_dir / f"benchmark_{ts}.json"
    save_json_results(all_results, json_path, args.classifier_model, total_time, metadata=metadata)
    print(f"JSON report: {json_path}")

    md_path = output_dir / f"benchmark_{ts}.md"
    generate_markdown_report(all_results, md_path, args.classifier_model, total_time, metadata=metadata)
    print(f"Markdown report: {md_path}")

    if compare_to_path is not None:
        comparison = compare_benchmark_results(compare_to_path, all_results)
        compare_path = output_dir / f"benchmark_compare_{ts}.json"
        compare_path.write_text(json.dumps(comparison, indent=2, default=str), encoding="utf-8")
        summary = comparison["summary"]
        print(
            "Comparison: "
            f"+{summary['improved']} / -{summary['regressed']} / "
            f"{summary['same']} same / {summary['current_only']} new"
        )
        print(f"Comparison report: {compare_path}")

    # Also save a "latest" copy for easy access
    json_latest = output_dir / "benchmark_latest.json"
    save_json_results(all_results, json_latest, args.classifier_model, total_time, metadata=metadata)

    md_latest = output_dir / "benchmark_latest.md"
    generate_markdown_report(all_results, md_latest, args.classifier_model, total_time, metadata=metadata)
    print(f"Latest links: {json_latest}, {md_latest}")

    print(f"\nTotal time: {total_time:.1f}s")


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--_benchmark-worker":
        raise SystemExit(_benchmark_worker_cli(sys.argv[2], sys.argv[3]))
    main()
