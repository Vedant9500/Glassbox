"""
Glassbox SR Benchmark Suite
============================

Comprehensive evaluation of the symbolic regression pipeline across ~200
formulas of increasing complexity, organized into 8 difficulty tiers.

Usage:
  python scripts/benchmark_suite.py                                    # Full suite (fast-path only)
  python scripts/benchmark_suite.py --tier 1                           # Only tier 1
    python scripts/benchmark_suite.py --with-evolution                   # Include guided evolution (latest path)
    python scripts/benchmark_suite.py --evolution-only                   # Guided evolution only (skip fast-path)
  python scripts/benchmark_suite.py --classifier-model models/v3.pt   # Custom model
  python scripts/benchmark_suite.py --output-dir results/              # Custom output dir

Scoring:
    Uses displayed-formula MSE only for scoring; raw MSE is diagnostic.
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

import classifier_fast_path as cfp  # noqa: E402
from classifier_fast_path import run_fast_path, run_guided_evolution  # noqa: E402
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
        expr = parse_expr(formula, transformations=transformations, evaluate=False)
        free_syms = sorted(expr.free_symbols, key=lambda sym: sym.name)
        
        # Inject safe power into lambdify
        modules = [{"pow": _safe_numpy_power, "Pow": _safe_numpy_power}, "numpy"]
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
                
            res = func(*args)
            return np.asarray(res, dtype=np.float64).reshape(-1)
            
        return fn
    except Exception as e:
        err_msg = str(e)
        # Fallback to a very basic eval if sympy fails (unlikely)
        def fallback_fn(x: np.ndarray) -> np.ndarray:
            raise ValueError(f"SymPy parse failed for '{formula_str}': {err_msg}")
        return fallback_fn


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


def _normalize_formula_text(formula: str) -> str:
    """Normalize common unicode/operator variants to a parser-friendly form."""
    if not formula:
        return formula
    return (
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


def _postprocess_formula(formula: str) -> str:
    """Apply the same simplify_formula pipeline used by fast-path outputs.
    
    Uses wider snap tolerances than the fast-path because evolution-produced
    formulas have Ridge regression noise on coefficients (e.g. 0.9975 → 1.0).
    """
    normalized = _normalize_formula_text(formula)
    if not normalized or normalized in {"N/A", "ERROR", "?"}:
        return normalized

    try:
        try:
            from simplify_formula import simplify_onn_formula, SnapConfig, snap_formula_floats
        except ImportError:
            from scripts.simplify_formula import simplify_onn_formula, SnapConfig, snap_formula_floats

        formula_len = len(normalized)
        term_estimate = max(1, len([t for t in re.split(r'\s*[+-]\s*', normalized) if t.strip()]))
        too_complex_for_symbolic = formula_len > 500 or term_estimate > 24

        # Evolution formulas need wider tolerances than fast-path:
        # Ridge regression produces coefficients like 0.9975 instead of 1.0
        # and small spurious bias terms like 0.0001924 instead of 0.0
        evo_int_tol = 0.05    # snap 7.955 → 8, 2.04 → 2, etc.
        evo_zero_tol = 1e-3   # snap 0.0001924 → 0

        if too_complex_for_symbolic:
            return snap_formula_floats(
                normalized,
                SnapConfig(int_tol=evo_int_tol, zero_tol=evo_zero_tol),
            )

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

    try:
        fn = _parse_formula(normalized)
        y_pred = fn(x)
    except Exception:
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


def score_result(mse: float, formula: str) -> str:
    """Classify a result as EXACT / APPROX / LOOSE / FAIL."""
    if mse is None or not math.isfinite(mse):
        return "FAIL"
    n_terms = _count_terms(formula)
    if mse < 1e-6 and n_terms <= 10:
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
    """Decide whether guided evolution should run and return a short reason."""
    if evolution_only:
        return True, "evolution_only"

    if not with_evolution:
        return False, "disabled"

    if fp_result is None:
        return True, "fast_path_not_applicable"

    if mse is None or not math.isfinite(mse):
        return True, "invalid_mse"

    # Implementation: Early exit if fast-path found a perfect match.
    # We define "perfect" as MSE < 1e-12 and complex enough to not be a constant,
    # OR high-confidence classifier result with MSE < 1e-7.
    is_effectively_zero = (mse < 1e-12)
    is_high_confidence = False
    
    uncertainty = fp_result.get("uncertainty")
    if isinstance(uncertainty, dict):
        entropy = uncertainty.get("prediction_entropy")
        # Lower entropy means more certain. Threshold 0.3 is very strict.
        if entropy is not None and math.isfinite(float(entropy)) and float(entropy) < 0.3:
            is_high_confidence = True

    if is_effectively_zero:
        return False, "fast_path_exact_zero"
    
    if is_high_confidence and mse < 1e-7 and n_terms <= 6:
        return False, "fast_path_confident_exact"

    if mse >= 1e-6:
        return True, "mse_above_exact"

    if n_terms > 10:
        return True, "formula_too_complex"

    uncertainty = fp_result.get("uncertainty")
    if isinstance(uncertainty, dict):
        if uncertainty.get("prediction_uncertain") is True:
            return True, "prediction_uncertain"
        entropy = uncertainty.get("prediction_entropy")
        margin = uncertainty.get("prediction_margin")
        if entropy is not None and math.isfinite(float(entropy)) and float(entropy) > 0.85:
            return True, "high_entropy"
        if margin is not None and math.isfinite(float(margin)) and float(margin) < 0.1:
            return True, "low_margin"

    residual = fp_result.get("residual_diagnostics")
    if isinstance(residual, dict) and residual.get("residual_suspicious") is True:
        return True, "suspicious_residual"

    return False, "fast_path_confident"


SCORE_SYMBOLS = {
    "EXACT":  "[PASS]",
    "APPROX": "[APPROX]",
    "LOOSE":  "[LOOSE]",
    "FAIL":   "[FAIL]",
}


_PROPOSER_CACHE = {}
def _get_proposer(path: str, device: str):
    if path not in _PROPOSER_CACHE:
        from glassbox.sr.universal_proposer import load_universal_proposer_checkpoint
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
        "time": None,
        "score": "FAIL",
        "error": None,
        "n_terms": 0,
        "uncertainty": None,
        "residual_diagnostics": None,
        "candidate_formulas": None,
        "mse_divergence_abs": None,
        "mse_divergence_rel": None,
        "mse_divergence_flag": False,
    }

    try:
        # Generate data
        x_np, y_np = _generate_data(formula_str, x_min, x_max, n_samples)

        # Early return for constant targets. This avoids evolution-only
        # degenerating to formula "0" on flat signals.
        y_std = np.std(y_np)
        if y_std < 1e-10:
            const_val = float(np.mean(y_np))
            if abs(const_val - round(const_val)) < 1e-6:
                formula = str(int(round(const_val)))
            else:
                formula = f"{const_val:.6g}"
            formula = _postprocess_formula(formula)
            result["formula_discovered"] = formula
            result["mse_raw"] = 0.0
            result["mse_display"] = 0.0
            result["mse"] = 0.0
            result["time"] = 0.0
            result["n_terms"] = _count_terms(formula)
            result["uncertainty"] = cfp._prediction_uncertainty_metrics({'identity': 1.0})
            y_pred = cfp._evaluate_formula_values(formula, x_np)
            result["residual_diagnostics"] = (
                cfp._residual_diagnostics(y_np, y_pred, x_np)
                if y_pred is not None
                else None
            )
            result["candidate_formulas"] = [{
                "formula": formula,
                "mse": 0.0,
                "score": 0.0,
                "n_nonzero": result["n_terms"],
                "active_terms": ["1"],
                "alpha": 0.0,
            }]
            result["score"] = score_result(0.0, formula)
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
            )
            elapsed = time.time() - t0

            if fp_result is None:
                result["error"] = "fast_path_not_applicable"
                result["time"] = elapsed
            else:
                result["formula_discovered"] = fp_result.get("formula", "")
                result["mse_raw"] = fp_result.get("mse", float("inf"))
                result["mse_display"] = _evaluate_formula_mse(result["formula_discovered"], x_np, y_np)
                result["mse"] = _select_score_mse(result["mse_display"])
                result.update(_mse_divergence_stats(result["mse_display"], result["mse_raw"]))
                result["uncertainty"] = fp_result.get("uncertainty")
                result["candidate_formulas"] = fp_result.get("candidate_formulas")
                if result["formula_discovered"] and result["mse_display"] is None and result["error"] is None:
                    result["error"] = "formula_eval_failed"
                result["time"] = elapsed
                str_term_count = _count_terms(result["formula_discovered"])
                details = fp_result.get("details", {}) if isinstance(fp_result, dict) else {}
                structural_terms = details.get("n_nonzero", 0)
                simplified_terms = details.get("n_nonzero_simplified", 0)
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
                print(
                    f"\n  [Latest Path] Fast-path candidate (MSE={result['mse']:.2e}, reason={guided_reason}). "
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
            if not disable_proposer and proposer_path:
                try:
                    from glassbox.sr.universal_proposer import propose_fpip_v2_from_xy
                    model = _get_proposer(proposer_path, device)
                    payload = propose_fpip_v2_from_xy(
                        model, x=x_np, y=y_np, top_k=5, device=device
                    )
                    if payload and payload.get("valid"):
                        seq_unc = payload.get("sequence_uncertainty", {})
                        if seq_unc.get("confident") is True:
                            proposer_confidence = 1.0
                        
                        proposer_priors = payload.get("operator_priors", {})
                        if proposer_priors:
                            for op, prob in proposer_priors.items():
                                if prob > 0.15:
                                    operator_hints["operators"].add(op)
                        proposer_skeletons = payload.get("candidate_skeletons", [])
                        candidate_formulas = []
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
                            print(f"\n  [Universal Proposer] Active FPIPv2 metadata injected! Skeletons: {[c['formula'] for c in candidate_formulas]}")
                except Exception as e:
                    print(f"\n  [Universal Proposer] Warning: execution failed: {e}")

            t1 = time.time()
            try:
                guided_result = run_guided_evolution(
                    x_2d,
                    y_2d,
                    operator_hints,
                    generations=500,
                    population_size=100,
                    device=device,
                    candidate_formulas=candidate_formulas,
                    confidence=proposer_confidence,
                )

                guided_elapsed = time.time() - t1
                base_elapsed = result["time"] if result["time"] is not None else elapsed
                result["time"] = base_elapsed + guided_elapsed

                if guided_result and guided_result.get("formula"):
                    guided_formula = _postprocess_formula(guided_result.get("formula", ""))
                    guided_mse_raw = guided_result.get("mse", float("inf"))
                    guided_mse_display = _evaluate_formula_mse(guided_formula, x_np, y_np)
                    guided_mse_score = _select_score_mse(guided_mse_display)
                    guided_mse_for_compare = (
                        guided_mse_score if guided_mse_score is not None else float("inf")
                    )
                    baseline_mse = result["mse"] if result["mse"] is not None else float("inf")

                    if evolution_only or fp_result is None or guided_mse_for_compare < baseline_mse:
                        result["formula_discovered"] = guided_formula
                        result["mse_raw"] = guided_mse_raw
                        result["mse_display"] = guided_mse_display
                        result["mse"] = guided_mse_for_compare
                        result.update(_mse_divergence_stats(result["mse_display"], result["mse_raw"]))
                        result["uncertainty"] = fp_result.get("uncertainty") if fp_result else None
                        result["candidate_formulas"] = fp_result.get("candidate_formulas") if fp_result else None
                        if result["formula_discovered"] and result["mse_display"] is None:
                            result["error"] = "formula_eval_failed"
                        else:
                            result["error"] = None
                        result["n_terms"] = _count_terms(guided_formula)
                elif evolution_only or fp_result is None:
                    result["error"] = "guided_evolution_failed"
            except Exception as guided_err:
                if evolution_only or fp_result is None:
                    result["error"] = f"guided_evolution_error: {guided_err}"
                print(f"  [Guided Evolution Error: {guided_err}]", end="")

        if result["formula_discovered"]:
            if result["mse_display"] is None:
                result["mse_display"] = _evaluate_formula_mse(result["formula_discovered"], x_np, y_np)
            result["mse"] = _select_score_mse(result["mse_display"])
            result.update(_mse_divergence_stats(result["mse_display"], result["mse_raw"]))
            y_pred = cfp._evaluate_formula_values(result["formula_discovered"], x_np)
            result["residual_diagnostics"] = (
                cfp._residual_diagnostics(y_np, y_pred, x_np)
                if y_pred is not None
                else None
            )
            if result["formula_discovered"] and result["mse_display"] is None and result["error"] is None:
                result["error"] = "formula_eval_failed"

    except Exception as e:
        result["error"] = str(e)
        result["time"] = 0.0

    # Score
    result["score"] = score_result(result["mse"], result["formula_discovered"])
    return result


def run_formula_cpp_evolution(
    formula_str: str,
    x_range: Tuple[float, float],
    n_samples: int = 300,
    pop_size: int = 100,
    generations: int = 1000,
    device: Optional[str] = None,
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
        "time": None,
        "score": "FAIL",
        "error": None,
        "n_terms": 0,
        "residual_diagnostics": None,
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
        
        # Generate data
        x_np, y_np = _generate_data(formula_str, x_min, x_max, n_samples)
        
        # Early return for constant signals (evolution can't find pure constants)
        y_std = np.std(y_np)
        if y_std < 1e-10:
            elapsed = 0.0
            const_val = float(np.mean(y_np))
            if abs(const_val - round(const_val)) < 1e-6:
                formula = str(int(round(const_val)))
            else:
                formula = f"{const_val:.6g}"
            formula = _postprocess_formula(formula)
            result["formula_discovered"] = formula
            result["mse_raw"] = 0.0
            result["mse_display"] = 0.0
            result["mse"] = 0.0
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
        
        # Run pure C++ evolution
        t0 = time.time()
        cpp_result = _core.run_evolution(
            X_list=X_list,
            y=y_np,
            pop_size=pop_size,
            generations=generations,
            early_stop_mse=1e-10,
            seed_omegas=detected_omegas or [],
        )
        elapsed = time.time() - t0
        
        result["formula_discovered"] = _postprocess_formula(cpp_result.get("formula", ""))
        result["mse_raw"] = cpp_result.get("best_mse", float("inf"))
        result["mse_display"] = _evaluate_formula_mse(result["formula_discovered"], x_np, y_np)
        result["mse"] = _select_score_mse(result["mse_display"])
        result.update(_mse_divergence_stats(result["mse_display"], result["mse_raw"]))
        y_pred = cfp._evaluate_formula_values(result["formula_discovered"], x_np)
        result["residual_diagnostics"] = (
            cfp._residual_diagnostics(y_np, y_pred, x_np)
            if y_pred is not None
            else None
        )
        if result["formula_discovered"] and result["mse_display"] is None and result["error"] is None:
            result["error"] = "formula_eval_failed"
        result["time"] = elapsed
        result["n_terms"] = _count_terms(result["formula_discovered"])
        
    except ImportError as e:
        result["error"] = f"C++ backend not available: {e}"
        result["time"] = 0.0
    except Exception as e:
        result["error"] = str(e)
        result["time"] = 0.0
    
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
        lines.append("| # | Score | Target | Discovered | MSE(score) | MSE(raw) | MSE(display) | Time | Terms |")
        lines.append("|---|-------|--------|------------|------------|----------|--------------|------|-------|")

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
            time_s = f"{r['time']:.2f}s" if r["time"] is not None else "—"
            n_terms = r.get("n_terms", 0)
            err = r.get("error", "")
            if err:
                disc = f"ERROR: {err[:40]}"
            lines.append(
                f"| {i} | {sym} | `{target}` | `{disc}` | {mse_s} | {mse_raw_s} | {mse_display_s} | {time_s} | {n_terms} |"
            )

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
        "--proposer-model", type=str, default="models/universal_proposer_v1.pt",
        help="Path to the universal neural proposer model (default: models/universal_proposer_v1.pt)",
    )
    parser.add_argument(
        "--disable-proposer", action="store_true",
        help="Disable the neural proposer and rely purely on legacy classifier hints",
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
        "--runs", type=int, default=1,
        help="Number of times to run each formula. Returns best result. (default: 1)",
    )
    args = parser.parse_args()

    if args.cpp_evolution_only and args.evolution_only:
        print("Error: --cpp-evolution-only and --evolution-only cannot be used together.")
        sys.exit(1)

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
    if args.cpp_evolution_only:
        mode_str = "Pure C++ Evolution (No Classifier/Proposer)"
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
    else:
        print(f"  Classifier:  {args.classifier_model}")
        proposer_txt = args.proposer_model if not args.disable_proposer else "DISABLED"
        print(f"  Proposer:    {proposer_txt}")
        print("  Strategy:    optimized")
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
        print(f"\n{'-' * 90}")
        print(f"  TIER {tier_num}: {tier_name}  ({len(formulas)} formulas)")
        print(f"{'-' * 90}")

        tier_results = []
        for formula_str, human_name, x_range in formulas:
            formula_idx += 1
            if not args.quiet:
                try:
                    print(f"  [{formula_idx}/{total_formulas}] {human_name:<30} ", end="", flush=True)
                except UnicodeEncodeError:
                    human_name = human_name.encode('ascii', 'ignore').decode('ascii')
                    print(f"  [{formula_idx}/{total_formulas}] {human_name:<30} ", end="", flush=True)

            best_result = None
            
            for _ in range(args.runs):
                if args.cpp_evolution_only:
                    result = run_formula_cpp_evolution(
                        formula_str,
                        x_range,
                        n_samples=args.n_samples,
                        pop_size=args.pop_size,
                        generations=args.generations,
                        device=device,
                    )
                else:
                    result = run_formula(
                        formula_str,
                        x_range,
                        classifier_path=args.classifier_model,
                        n_samples=args.n_samples,
                        device=device,
                        timeout=args.timeout,
                        with_evolution=args.with_evolution,
                        evolution_only=args.evolution_only,
                        proposer_path=args.proposer_model,
                        disable_proposer=args.disable_proposer,
                    )
                
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
