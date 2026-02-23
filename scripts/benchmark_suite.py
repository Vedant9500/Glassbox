"""
Glassbox SR Benchmark Suite
============================

Comprehensive evaluation of the symbolic regression pipeline across ~200
formulas of increasing complexity, organized into 8 difficulty tiers.

Usage:
  python scripts/benchmark_suite.py                                    # Full suite (fast-path only)
  python scripts/benchmark_suite.py --tier 1                           # Only tier 1
  python scripts/benchmark_suite.py --with-evolution                   # Include evolution fallback
  python scripts/benchmark_suite.py --classifier-model models/v3.pt   # Custom model
  python scripts/benchmark_suite.py --output-dir results/              # Custom output dir

Scoring:
  EXACT   — MSE < 1e-6  AND  ≤ 5 terms
  APPROX  — MSE < 0.01
  LOOSE   — MSE < 0.1
  FAIL    — MSE ≥ 0.1 or error
"""

import argparse
import json
import math
import os
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))

from classifier_fast_path import run_fast_path  # noqa: E402
from glassbox.sr.evolution import detect_dominant_frequency  # noqa: E402

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

def _parse_formula(formula_str: str) -> Callable[[np.ndarray], np.ndarray]:
    """Parse formula string into a numpy function."""
    formula = formula_str.strip().lower()
    # x^N -> x**N
    formula = re.sub(r'\^(\d+)', r'**\1', formula)
    formula = re.sub(r'\^(\()', r'**\1', formula)
    formula = re.sub(r'\^(\d+\.\d+)', r'**\1', formula)  # fractional powers
    formula = formula.replace('np.', '')

    # Replace math functions with numpy equivalents
    formula = formula.replace('sin(', 'np.sin(')
    formula = formula.replace('cos(', 'np.cos(')
    formula = formula.replace('tan(', 'np.tan(')
    formula = formula.replace('exp(', 'np.exp(')
    formula = formula.replace('log(', 'np.log(')
    # sqrt of numeric constants first
    formula = re.sub(r'sqrt\(\s*([0-9\.]+)\s*\)', lambda m: str(math.sqrt(float(m.group(1)))), formula)
    formula = formula.replace('sqrt(', 'np.sqrt(')
    formula = formula.replace('abs(', 'np.abs(')
    formula = formula.replace('pi', str(math.pi))
    formula = re.sub(r'\be\b', str(math.e), formula)

    def fn(x: np.ndarray) -> np.ndarray:
        try:
            result = eval(formula)
            if isinstance(result, (int, float)):
                return np.full_like(x, result, dtype=np.float64)
            return np.asarray(result, dtype=np.float64)
        except Exception as e:
            raise ValueError(f"Error evaluating '{formula_str}': {e}")

    return fn


def _generate_data(
    formula_str: str,
    x_min: float = -5.0,
    x_max: float = 5.0,
    n_samples: int = 300,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate clean (x, y) data from a formula string."""
    fn = _parse_formula(formula_str)
    x = np.linspace(x_min, x_max, n_samples)
    y = fn(x)

    # Filter out non-finite values
    mask = np.isfinite(y)
    if mask.sum() < 10:
        raise ValueError(f"Too few valid points ({mask.sum()}) for '{formula_str}'")
    x = x[mask]
    y = y[mask]
    return x, y


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


def score_result(mse: float, formula: str) -> str:
    """Classify a result as EXACT / APPROX / LOOSE / FAIL."""
    if mse is None or not math.isfinite(mse):
        return "FAIL"
    n_terms = _count_terms(formula)
    if mse < 1e-6 and n_terms <= 5:
        return "EXACT"
    if mse < 0.01:
        return "APPROX"
    if mse < 0.1:
        return "LOOSE"
    return "FAIL"


SCORE_SYMBOLS = {
    "EXACT":  "✅",
    "APPROX": "🟡",
    "LOOSE":  "🟠",
    "FAIL":   "❌",
}


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
) -> Dict[str, Any]:
    """Run the fast-path pipeline on a single formula and return result dict."""
    x_min, x_max = x_range
    result = {
        "formula_target": formula_str,
        "x_range": list(x_range),
        "formula_discovered": "",
        "mse": None,
        "time": None,
        "score": "FAIL",
        "error": None,
        "n_terms": 0,
    }

    try:
        # Generate data
        x_np, y_np = _generate_data(formula_str, x_min, x_max, n_samples)

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

        # Run fast path
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
        )
        elapsed = time.time() - t0

        if fp_result is None:
            result["error"] = "fast_path_not_applicable"
            result["time"] = elapsed
        else:
            result["formula_discovered"] = fp_result.get("formula", "")
            result["mse"] = fp_result.get("mse", float("inf"))
            result["time"] = elapsed
            result["n_terms"] = _count_terms(result["formula_discovered"])
            
            # -----------------------------------------------------------------
            # Evolution Fallback (Phase 1 + 2)
            # -----------------------------------------------------------------
            # If we didn't get an EXACT match, and evolution is enabled
            if with_evolution and (result["mse"] >= 1e-6 or result["n_terms"] > 5):
                print(f"\n  [Fallback] Fast-path approximate (MSE={result['mse']:.2e}). Running C++ Evolution...")
                try:
                    from glassbox.sr.evolution import EvolutionaryONNTrainer
                    from glassbox.sr.operation_dag import OperationDAG
                    
                    def make_model() -> OperationDAG:
                        return OperationDAG(
                            n_inputs=1,
                            n_hidden_layers=2,
                            nodes_per_layer=4,
                            n_outputs=1,
                            simplified_ops=True,
                            fair_mode=True,
                        )
                    
                    trainer = EvolutionaryONNTrainer(
                        model_factory=make_model,
                        population_size=100,
                        device=device or "cuda",
                        use_curve_classifier=False, # We already ran fast-path
                        lamarckian=True,
                        prune_coefficients=False,
                    )
                    
                    # Ensure device matches
                    actual_device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
                    
                    # Re-run evolution
                    t1 = time.time()
                    evo_results = trainer.train(x_t.reshape(-1, 1).to(actual_device), y_t.reshape(-1, 1).to(actual_device), generations=2000)
                    
                    # For benchmark, we just take the raw C++ output
                    
                    evo_mse = evo_results.get("final_mse", float("inf"))
                    if evo_mse < result["mse"]:
                        result["formula_discovered"] = evo_results.get("formula", "")
                        result["mse"] = evo_mse
                        result["time"] = elapsed + (time.time() - t1)
                        result["n_terms"] = _count_terms(result["formula_discovered"])
                except Exception as eval_err:
                    print(f"  [Evolution Fallback Error: {eval_err}]", end="")
                    pass

    except Exception as e:
        result["error"] = str(e)
        result["time"] = 0.0

    # Score
    result["score"] = score_result(result["mse"], result["formula_discovered"])
    return result


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
) -> None:
    """Write a detailed Markdown report."""
    lines = []
    lines.append("# Glassbox SR Benchmark Report\n")
    lines.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"**Classifier**: `{classifier_path}`\n")
    lines.append(f"**Total runtime**: {total_time:.1f}s\n")

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
        lines.append("| # | Score | Target | Discovered | MSE | Time | Terms |")
        lines.append("|---|-------|--------|------------|-----|------|-------|")

        for i, r in enumerate(results, 1):
            sym = SCORE_SYMBOLS.get(r["score"], "?")
            target = r["formula_target"]
            disc = r.get("formula_discovered", "")
            if len(disc) > 50:
                disc = disc[:47] + "..."
            mse_s = f"{r['mse']:.2e}" if r["mse"] is not None and math.isfinite(r["mse"]) else "—"
            time_s = f"{r['time']:.2f}s" if r["time"] is not None else "—"
            n_terms = r.get("n_terms", 0)
            err = r.get("error", "")
            if err:
                disc = f"ERROR: {err[:40]}"
            lines.append(f"| {i} | {sym} | `{target}` | `{disc}` | {mse_s} | {time_s} | {n_terms} |")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_json_results(
    all_results: Dict[int, List[Dict]],
    output_path: Path,
    classifier_path: str,
    total_time: float,
) -> None:
    """Save full results to JSON."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "classifier": classifier_path,
        "total_time_seconds": round(total_time, 2),
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
        """,
    )
    parser.add_argument(
        "--classifier-model", type=str, default="models/curve_classifier_v3.1.pt",
        help="Path to the curve classifier model (default: models/curve_classifier_v3.1.pt)",
    )
    parser.add_argument(
        "--with-evolution", action="store_true",
        help="Run evolution fallback if fast-path fails to find an exact match",
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
        "--timeout", type=float, default=60.0,
        help="Timeout per formula in seconds (default: 60)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-formula output, only show summary",
    )
    parser.add_argument(
        "--formula", type=str, default=None,
        help="Run a single formula by searching all tiers (e.g., --formula 'sin(x)')",
    )

    args = parser.parse_args()

    # Resolve device
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

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
    print(f"  Classifier:  {args.classifier_model}")
    print(f"  Device:      {device}")
    print(f"  Tiers:       {tiers_to_run}")
    print(f"  Formulas:    {total_formulas}")
    print(f"  Samples/ea:  {args.n_samples}")
    print("=" * 90)

    # Run benchmark
    all_results: Dict[int, List[Dict]] = {}
    formula_idx = 0
    t_start = time.time()

    for tier_num in sorted(tiers_to_run):
        tier_name, formulas = ALL_TIERS[tier_num]
        print(f"\n{'─' * 90}")
        print(f"  TIER {tier_num}: {tier_name}  ({len(formulas)} formulas)")
        print(f"{'─' * 90}")

        tier_results = []
        for formula_str, human_name, x_range in formulas:
            formula_idx += 1
            if not args.quiet:
                print(f"  [{formula_idx}/{total_formulas}] {human_name:<30} ", end="", flush=True)

            result = run_formula(
                formula_str,
                x_range,
                classifier_path=args.classifier_model,
                n_samples=args.n_samples,
                device=device,
                timeout=args.timeout,
                with_evolution=args.with_evolution,
            )
            result["human_name"] = human_name
            tier_results.append(result)

            if not args.quiet:
                sym = SCORE_SYMBOLS.get(result["score"], "?")
                mse_s = f"MSE={result['mse']:.2e}" if result["mse"] is not None and math.isfinite(result["mse"]) else "N/A     "
                time_s = f"{result['time']:.2f}s" if result["time"] is not None else "—    "
                disc = result.get("formula_discovered", "")
                if len(disc) > 40:
                    disc = disc[:37] + "..."
                print(f"{sym}  {mse_s}  {time_s}  {disc}")

        all_results[tier_num] = tier_results

        # Print tier subtotal
        s = _tier_summary(tier_results)
        pct = (s["EXACT"] / s["total"] * 100) if s["total"] > 0 else 0
        print(f"  ── Tier {tier_num} subtotal: {s['EXACT']}/{s['total']} exact ({pct:.0f}%)")

    total_time = time.time() - t_start

    # Print summary
    print_summary(all_results)

    # Save reports
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = output_dir / f"benchmark_{ts}.json"
    save_json_results(all_results, json_path, args.classifier_model, total_time)
    print(f"JSON report: {json_path}")

    md_path = output_dir / f"benchmark_{ts}.md"
    generate_markdown_report(all_results, md_path, args.classifier_model, total_time)
    print(f"Markdown report: {md_path}")

    # Also save a "latest" copy for easy access
    json_latest = output_dir / "benchmark_latest.json"
    save_json_results(all_results, json_latest, args.classifier_model, total_time)

    md_latest = output_dir / "benchmark_latest.md"
    generate_markdown_report(all_results, md_latest, args.classifier_model, total_time)
    print(f"Latest links: {json_latest}, {md_latest}")

    print(f"\nTotal time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
