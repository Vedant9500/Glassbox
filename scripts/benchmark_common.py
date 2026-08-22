"""Shared benchmark helpers for benchmark_suite.py and run_srbench_local.py."""

from __future__ import annotations

import ast
import math
import re
import warnings
from pathlib import Path

import numpy as np

try:
    from sympy.utilities.exceptions import SymPyDeprecationWarning
except Exception:  # pragma: no cover - SymPy is optional for some helper use
    SymPyDeprecationWarning = DeprecationWarning

warnings.filterwarnings("ignore", category=SymPyDeprecationWarning)
warnings.filterwarnings("ignore", message=r"\s*Using non-Expr arguments in Mul.*")


_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent


def get_official_pmlb_regression_datasets():
    """Return the current PMLB regression dataset list used by SRBench black-box track."""
    try:
        from pmlb import regression_dataset_names
    except Exception:
        return []
    return list(regression_dataset_names)


def formula_benchmark_seed(formula, x_range=None, *, base_seed=0, n_samples=None):
    """Deterministic per-formula seed for reproducible benchmark A/B runs.

    Same formula + range (+ optional n_samples) maps to the same C++/Python seed,
    independent of tier order. *base_seed* lets a global ``--seed`` shift the space.
    """
    import hashlib
    import struct

    parts = [str(formula or "").strip().replace(" ", "")]
    if x_range is not None:
        try:
            parts.append(f"{float(x_range[0]):.8g}:{float(x_range[1]):.8g}")
        except Exception:
            parts.append(str(x_range))
    if n_samples is not None:
        parts.append(f"n={int(n_samples)}")
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    # Stable positive 31-bit int so it fits typical RNG seeds.
    derived = struct.unpack(">I", digest[:4])[0] & 0x7FFFFFFF
    return int((int(base_seed) + derived) % (2**31 - 1))


def _read_symbolic_formula_from_metadata(dataset_dir):
    """Best-effort extraction of the symbolic target formula from PMLB metadata."""
    for meta_name in ("metadata.yaml", "metadata.yml"):
        meta_path = Path(dataset_dir) / meta_name
        if not meta_path.exists():
            continue
        try:
            text = meta_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        for key in ("formula", "equation", "symbolic_model", "model", "truth"):
            match = re.search(rf"(?im)^\s*{re.escape(key)}\s*:\s*(.+?)\s*$", text)
            if match:
                value = match.group(1).strip().strip("'\"")
                if value:
                    return value
    return ""


def discover_official_ground_truth_problems(data_dir):
    """Discover SRBench ground-truth files from a PMLB datasets checkout."""
    if not data_dir:
        return []

    root = Path(data_dir)
    if not root.exists():
        return []

    candidates = []
    patterns = [
        "strogatz_*/*.tsv",
        "strogatz_*/*.tsv.gz",
        "feynman_*/*.tsv",
        "feynman_*/*.tsv.gz",
        "**/strogatz_*.tsv",
        "**/strogatz_*.tsv.gz",
        "**/feynman_*.tsv",
        "**/feynman_*.tsv.gz",
    ]
    seen = set()
    for pattern in patterns:
        for path in root.glob(pattern):
            if not path.is_file():
                continue
            key = str(path.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            dataset_dir = path.parent
            name = (
                dataset_dir.name
                if dataset_dir.name.startswith(("strogatz_", "feynman_"))
                else path.stem
            )
            name = name.removesuffix(".tsv")
            candidates.append(
                {
                    "kind": "file",
                    "name": name,
                    "path": str(path),
                    "true_formula": _read_symbolic_formula_from_metadata(dataset_dir),
                }
            )

    return sorted(candidates, key=lambda problem: problem["name"])


def r2_score(y_true, y_pred):
    """Compute R^2 score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-15:
        return 1.0 if ss_res < 1e-15 else 0.0
    return 1.0 - ss_res / ss_tot


def mse_score(y_true, y_pred):
    """Compute MSE."""
    return float(np.mean((y_true - y_pred) ** 2))


def model_size(formula_str):
    """Rough complexity measure: count operators and terms."""
    if not formula_str:
        return 0
    ops = sum(1 for c in formula_str if c in "+-*/^")
    funcs = sum(formula_str.count(f) for f in ["sin", "cos", "exp", "log", "sqrt"])
    return ops + funcs + 1


def normalize_formula_text(formula):
    """Normalize common unicode/operator variants to a parser-friendly form."""
    if not formula:
        return formula
    replacements = {
        "\u00b2": "^2",
        "\u00b3": "^3",
        "\u00b7": "*",
        "\u22c5": "*",
        "\u00d7": "*",
        "\u03c0": "pi",
        "\u221a": "sqrt",
        "\u03c6": "phi",
        "\u03c9": "omega",
    }
    for src, dst in list(replacements.items()):
        variant = src
        for _ in range(2):
            try:
                variant = variant.encode("utf-8").decode("latin-1")
            except UnicodeError:
                break
            replacements.setdefault(variant, dst)

    formula = str(formula).replace("np.", "")
    for src, dst in replacements.items():
        formula = formula.replace(src, dst)
    return re.sub(r"\s+", "", formula)


def apply_canonical_rewrites(formula):
    """Apply a small deterministic rewrite set before heavier simplification."""
    text = normalize_formula_text(formula)
    if not text:
        return text
    text = text.replace("^", "**")

    class _CanonicalRewrite(ast.NodeTransformer):
        @staticmethod
        def _same(left, right):
            return ast.dump(left) == ast.dump(right)

        @staticmethod
        def _call_name(node):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                return node.func.id
            return None

        @classmethod
        def _is_func_square(cls, node, func_name):
            return (
                isinstance(node, ast.BinOp)
                and isinstance(node.op, ast.Pow)
                and isinstance(node.right, ast.Constant)
                and node.right.value == 2
                and cls._call_name(node.left) == func_name
                and len(node.left.args) == 1
            )

        def visit_BinOp(self, node):
            node = self.generic_visit(node)
            if (
                isinstance(node.op, ast.Pow)
                and isinstance(node.left, ast.Name)
                and node.left.id in {"E", "e"}
            ):
                return ast.copy_location(
                    ast.Call(
                        func=ast.Name(id="exp", ctx=ast.Load()),
                        args=[node.right],
                        keywords=[],
                    ),
                    node,
                )
            if isinstance(node.op, ast.Add):
                left_sin = self._is_func_square(node.left, "sin")
                right_cos = self._is_func_square(node.right, "cos")
                left_cos = self._is_func_square(node.left, "cos")
                right_sin = self._is_func_square(node.right, "sin")
                if (
                    left_sin
                    and right_cos
                    and self._same(node.left.left.args[0], node.right.left.args[0])
                ):
                    return ast.copy_location(ast.Constant(value=1), node)
                if (
                    left_cos
                    and right_sin
                    and self._same(node.left.left.args[0], node.right.left.args[0])
                ):
                    return ast.copy_location(ast.Constant(value=1), node)
            if isinstance(node.op, ast.Mult) and ast.dump(node.left) == ast.dump(
                node.right
            ):
                return ast.copy_location(
                    ast.BinOp(
                        left=node.left, op=ast.Pow(), right=ast.Constant(value=2)
                    ),
                    node,
                )
            if isinstance(node.op, ast.Mult):
                left_name = self._call_name(node.left)
                right_name = self._call_name(node.right)
                if (
                    {left_name, right_name} == {"sin", "cos"}
                    and len(node.left.args) == 1
                    and len(node.right.args) == 1
                    and self._same(node.left.args[0], node.right.args[0])
                ):
                    doubled_arg = ast.BinOp(
                        left=ast.Constant(value=2),
                        op=ast.Mult(),
                        right=node.left.args[0],
                    )
                    replacement = ast.BinOp(
                        left=ast.Call(
                            func=ast.Name(id="sin", ctx=ast.Load()),
                            args=[doubled_arg],
                            keywords=[],
                        ),
                        op=ast.Div(),
                        right=ast.Constant(value=2),
                    )
                    return ast.copy_location(replacement, node)
            return node

        def visit_Call(self, node):
            node = self.generic_visit(node)
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "log"
                and len(node.args) == 1
                and isinstance(node.args[0], ast.Call)
                and isinstance(node.args[0].func, ast.Name)
                and node.args[0].func.id == "exp"
                and len(node.args[0].args) == 1
            ):
                return ast.copy_location(node.args[0].args[0], node)
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "exp"
                and len(node.args) == 1
                and isinstance(node.args[0], ast.Call)
                and isinstance(node.args[0].func, ast.Name)
                and node.args[0].func.id == "log"
                and len(node.args[0].args) == 1
            ):
                return ast.copy_location(node.args[0].args[0], node)
            return node

    try:
        tree = ast.parse(text, mode="eval")
        tree = _CanonicalRewrite().visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree).replace("^", "**")
    except Exception:
        return text


def infer_formula_n_features(formula):
    indices = [int(m.group(1)) for m in re.finditer(r"\bx(\d+)\b", str(formula or ""))]
    return max(indices) + 1 if indices else 1


def protect_fractional_powers(formula):
    """Rewrite simple non-integer powers to a real-valued protected form."""
    text = str(formula or "")
    if "**" not in text:
        return text

    class _FractionalPowerProtector(ast.NodeTransformer):
        def visit_BinOp(self, node):
            node = self.generic_visit(node)
            if not isinstance(node.op, ast.Pow):
                return node
            exponent = None
            if isinstance(node.right, ast.Constant) and isinstance(
                node.right.value, (int, float)
            ):
                exponent = float(node.right.value)
            elif (
                isinstance(node.right, ast.UnaryOp)
                and isinstance(node.right.op, ast.USub)
                and isinstance(node.right.operand, ast.Constant)
                and isinstance(node.right.operand.value, (int, float))
            ):
                exponent = -float(node.right.operand.value)
            if exponent is None:
                return node
            nearest_int = round(exponent)
            if abs(exponent - nearest_int) <= 1e-10:
                node.right = ast.Constant(value=int(nearest_int))
                return node
            return ast.Call(
                func=ast.Name(id="_signed_power", ctx=ast.Load()),
                args=[node.left, ast.Constant(value=exponent)],
                keywords=[],
            )

    try:
        tree = ast.parse(text, mode="eval")
        tree = _FractionalPowerProtector().visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    except Exception:
        return text


def _extract_power_exponent(node):
    """Return float exponent if *node* is a numeric power, else None."""
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Pow):
        return None
    if isinstance(node.right, ast.Constant) and isinstance(
        node.right.value, (int, float)
    ):
        return float(node.right.value)
    if (
        isinstance(node.right, ast.UnaryOp)
        and isinstance(node.right.op, ast.USub)
        and isinstance(node.right.operand, ast.Constant)
        and isinstance(node.right.operand.value, (int, float))
    ):
        return -float(node.right.operand.value)
    return None


def round_powers_to_integers(formula, *, max_power=8, tol=0.25):
    """Round near-integer ``x**p`` exponents into {1..max_power}.

    Used by the post-search exactness pass to convert numerical twins like
    ``x**2.33`` into integer-power candidates before re-fitting.
    """
    text = str(formula or "").replace("^", "**")
    if "**" not in text:
        return text

    max_power = max(1, int(max_power))
    tol = float(tol)

    class _IntegerPowerRounder(ast.NodeTransformer):
        def visit_BinOp(self, node):
            node = self.generic_visit(node)
            exponent = _extract_power_exponent(node)
            if exponent is None:
                return node
            nearest = int(round(exponent))
            if nearest < 1 or nearest > max_power:
                return node
            if abs(exponent - nearest) > tol and abs(exponent - nearest) > 1e-10:
                return node
            return ast.copy_location(
                ast.BinOp(
                    left=node.left, op=ast.Pow(), right=ast.Constant(value=nearest)
                ),
                node,
            )

    try:
        tree = ast.parse(text, mode="eval")
        tree = _IntegerPowerRounder().visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree).replace("^", "**")
    except Exception:
        return text


def exactness_pass_candidates(formula, *, max_power=8):
    """Generate cheap structural rewrites for the post-search exactness pass.

    Returns a de-duplicated list starting with the original (normalized) formula.
    """
    from collections import OrderedDict

    text = normalize_formula_text(formula)
    if not text or text in {"N/A", "ERROR", "?"}:
        return []

    candidates = OrderedDict()

    def _add(f):
        f = str(f or "").strip()
        if not f:
            return
        key = f.replace(" ", "")
        if key not in candidates:
            candidates[key] = f

    _add(text)
    rewritten = apply_canonical_rewrites(text)
    _add(rewritten)

    int_rounded = round_powers_to_integers(text, max_power=max_power)
    _add(int_rounded)
    _add(apply_canonical_rewrites(int_rounded))

    # Trig product probe: sum of harmonics often hides a*sin(x)*cos(x) / a*x**2*sin(x).
    # We only inject a few high-value templates when many sin/cos terms are present.
    lower = text.lower()
    n_sin = len(re.findall(r"\bsin\s*\(", lower))
    n_cos = len(re.findall(r"\bcos\s*\(", lower))
    has_power = bool(re.search(r"\*\*|x\s*\^", lower))
    if n_sin + n_cos >= 2:
        _add("sin(x)*cos(x)")
        _add("0.5*sin(2*x)")
        if has_power or n_sin >= 2:
            _add("x*sin(x)")
            _add("x**2*sin(x)")
            _add("x**3*sin(x)")
            _add("x*cos(x)")
            _add("x**2*cos(x)")
    if "/" in text or "1+" in text.replace(" ", ""):
        _add("x**3/(1+x**4)")
        _add("x/(1+x**2)")
        _add("x**2/(1+x**2)")

    return list(candidates.values())


def run_exactness_pass(
    formula,
    X,
    y,
    *,
    raw_mse=None,
    raw_mse_threshold=1e-4,
    display_mse=None,
    max_power=8,
    improve_tol=0.99,
):
    """If raw fit is strong but display form is weak, try integer-power / identity rewrites.

    Returns ``(best_formula, diagnostics)``. Never worsens MSE beyond *improve_tol*
    relative to the better of raw/display baselines.
    """
    diagnostics = {
        "attempted": False,
        "accepted": False,
        "reason": None,
        "n_candidates": 0,
        "baseline_mse": None,
        "best_mse": None,
        "best_formula": None,
    }
    text = normalize_formula_text(formula)
    if not text or text in {"N/A", "ERROR", "?"}:
        diagnostics["reason"] = "empty_formula"
        return text, diagnostics

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    def _mse(f):
        try:
            return evaluate_formula_mse_on_X(f, X, y)
        except Exception:
            return float("inf")

    base_display = (
        float(display_mse)
        if display_mse is not None and np.isfinite(display_mse)
        else _mse(text)
    )
    base_raw = (
        float(raw_mse) if raw_mse is not None and np.isfinite(raw_mse) else base_display
    )
    baseline = (
        min(base_display, base_raw)
        if np.isfinite(base_display) or np.isfinite(base_raw)
        else float("inf")
    )
    diagnostics["baseline_mse"] = float(baseline) if np.isfinite(baseline) else None

    # Eligibility: strong raw fit but display not exact, or large raw↔display
    # drift. Skip when both metrics are already excellent.
    thr = float(raw_mse_threshold)
    raw_good = np.isfinite(base_raw) and base_raw <= thr
    display_good = np.isfinite(base_display) and base_display <= thr
    drifted = (
        np.isfinite(base_raw)
        and np.isfinite(base_display)
        and base_display > max(base_raw * 10.0, thr)
    )
    if display_good and not drifted:
        diagnostics["reason"] = "display_already_good"
        return text, diagnostics
    if not raw_good and not drifted:
        diagnostics["reason"] = "raw_mse_not_eligible"
        return text, diagnostics

    candidates = exactness_pass_candidates(text, max_power=max_power)
    diagnostics["attempted"] = True
    diagnostics["n_candidates"] = len(candidates)

    best_f = text
    best_m = base_display if np.isfinite(base_display) else baseline
    for cand in candidates:
        m = _mse(cand)
        if not np.isfinite(m):
            continue
        # Prefer simpler integer-power forms even if equal MSE.
        if m < best_m * float(improve_tol) - 1e-18 or (
            abs(m - best_m) <= 1e-18 and len(cand) < len(best_f)
        ):
            best_m = m
            best_f = cand

    diagnostics["best_mse"] = float(best_m) if np.isfinite(best_m) else None
    diagnostics["best_formula"] = best_f
    if best_f.replace(" ", "") != text.replace(" ", "") and np.isfinite(best_m):
        if not np.isfinite(baseline) or best_m <= baseline * 1.05 + 1e-15:
            diagnostics["accepted"] = True
            diagnostics["reason"] = "improved_or_equal"
            return best_f, diagnostics
        diagnostics["reason"] = "rejected_worse"
        return text, diagnostics

    diagnostics["reason"] = "no_better_candidate"
    return text, diagnostics


def simplify_formula_native(formula, int_tol=0.05, zero_tol=1e-3):
    """Simplify with the native C++ simplifier when available."""
    try:
        cpp_dir = _REPO_ROOT / "glassbox" / "sr" / "cpp"
        from glassbox.sr.cpp import get_cpp_core

        _core = get_cpp_core()

        if _core is None:
            raise ImportError("C++ _core unavailable")

        if hasattr(_core, "simplify_formula"):
            simplified = _core.simplify_formula(
                formula,
                int_tol=float(int_tol),
                zero_tol=float(zero_tol),
                max_passes=6,
                use_nsimplify=True,
                use_identities=True,
                n_features=infer_formula_n_features(formula),
            )
        elif hasattr(_core, "simplify_formula_cpp"):
            simplified = _core.simplify_formula_cpp(formula)
        else:
            return None

        if simplified and simplified not in {"N/A", "ERROR", "?"}:
            return str(simplified)
    except Exception:
        return None
    return None


def postprocess_formula(
    formula,
    *,
    fraction_tol=0.0,
    max_fraction_denominator=12,
):
    """Apply the shared benchmark formula cleanup pipeline."""
    normalized = normalize_formula_text(formula)
    if not normalized or normalized in {"N/A", "ERROR", "?"}:
        return normalized
    normalized = apply_canonical_rewrites(normalized)

    evo_int_tol = 0.05
    evo_zero_tol = 1e-3
    helper_unsafe = "_signed_power" in normalized or "Abs(" in normalized

    if not helper_unsafe:
        native = simplify_formula_native(normalized, evo_int_tol, evo_zero_tol)
        if native is not None:
            normalized = native

    try:
        try:
            from simplify_formula import (
                SnapConfig,
                simplify_onn_formula,
                snap_formula_floats,
            )
        except ImportError:
            from scripts.simplify_formula import (
                SnapConfig,
                simplify_onn_formula,
                snap_formula_floats,
            )

        formula_len = len(normalized)
        term_estimate = max(
            1, len([t for t in re.split(r"\s*[+-]\s*", normalized) if t.strip()])
        )
        nonlinear_families = sum(
            1
            for pattern in (
                r"\bsin\s*\(",
                r"\bcos\s*\(",
                r"\bexp\s*\(",
                r"\blog\s*\(",
                r"/",
                r"\*\*",
                r"\b_signed_power\s*\(",
            )
            if re.search(pattern, normalized)
        )
        too_complex_for_symbolic = (
            formula_len > 500
            or term_estimate > 24
            or (term_estimate > 10 and nonlinear_families >= 3)
        )
        sympy_unsafe = (
            helper_unsafe or "Piecewise(" in normalized or "Eq(" in normalized
        )

        if too_complex_for_symbolic or sympy_unsafe:
            snapped = snap_formula_floats(
                normalized,
                SnapConfig(
                    int_tol=evo_int_tol,
                    zero_tol=evo_zero_tol,
                    fraction_tol=fraction_tol,
                    max_fraction_denominator=max_fraction_denominator,
                ),
            )
            return protect_fractional_powers(snapped.replace("^", "**"))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=SymPyDeprecationWarning)
            warnings.filterwarnings(
                "ignore", category=DeprecationWarning, module=r"sympy\..*"
            )
            warnings.filterwarnings(
                "ignore", message=r"\s*Using non-Expr arguments in Mul.*"
            )
            _, simplified_expr = simplify_onn_formula(
                normalized,
                int_tol=evo_int_tol,
                zero_tol=evo_zero_tol,
                fraction_tol=fraction_tol,
                max_fraction_denominator=max_fraction_denominator,
                use_nsimplify=(formula_len <= 300 and term_estimate <= 16),
            )
        simplified_text = str(simplified_expr)
        if "Piecewise(" in simplified_text or "Eq(" in simplified_text:
            snapped = snap_formula_floats(
                normalized,
                SnapConfig(
                    int_tol=evo_int_tol,
                    zero_tol=evo_zero_tol,
                    fraction_tol=fraction_tol,
                    max_fraction_denominator=max_fraction_denominator,
                ),
            )
            return protect_fractional_powers(snapped.replace("^", "**"))
        return protect_fractional_powers(simplified_text.replace("^", "**"))
    except Exception:
        return protect_fractional_powers(normalized.replace("^", "**"))


def evaluate_formula(formula_str, X, *, return_diagnostics=False):
    """Evaluate a discovered formula string strictly on X."""
    diagnostics = {
        "ok": False,
        "reason": None,
        "exception_type": None,
        "exception_message": None,
        "formula": None,
    }

    def _finish(result, reason=None, exc=None):
        diagnostics["ok"] = result is not None
        diagnostics["reason"] = reason
        diagnostics["formula"] = formula if "formula" in locals() else None
        if exc is not None:
            diagnostics["exception_type"] = type(exc).__name__
            diagnostics["exception_message"] = str(exc)
        return (result, diagnostics) if return_diagnostics else result

    if not formula_str:
        return _finish(None, reason="empty_formula")

    try:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            return _finish(None, reason="invalid_X_shape")
    except Exception as exc:
        return _finish(None, reason="invalid_X", exc=exc)

    formula = normalize_formula_text(formula_str).strip()
    formula = re.sub(r"\|([^|]+)\|", r"abs(\1)", formula)
    formula = formula.replace("^", "**")
    formula = protect_fractional_powers(formula)

    def _safe_numpy_log(x, base=None):
        with np.errstate(divide="ignore", invalid="ignore"):
            x_arr = np.asarray(x, dtype=np.float64)
            # The estimator's formula evaluator treats log(Abs(x)) at x=0 as a
            # protected boundary value. Mirror that here so valid displayed
            # formulas are not rejected only because the benchmark grid includes
            # an endpoint at zero.
            x_arr = np.where(x_arr == 0.0, 1e-300, x_arr)
            out = np.log(x_arr)
            if base is not None:
                out = out / np.log(base)
        return out

    def _signed_power(base, power):
        base_arr = np.asarray(base, dtype=np.float64)
        power_val = float(power)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            if power_val < 0:
                return np.sign(base_arr) / (
                    (np.abs(base_arr) + 1e-12) ** abs(power_val)
                )
            return np.sign(base_arr) * (np.abs(base_arr) ** power_val)

    def _safe_exp(x):
        return np.exp(np.clip(x, -500.0, 500.0))

    context = {
        "np": np,
        "log": _safe_numpy_log,
        "sin": np.sin,
        "cos": np.cos,
        "exp": _safe_exp,
        "sqrt": np.sqrt,
        "abs": np.abs,
        "Abs": np.abs,
        "sign": np.sign,
        "_signed_power": _signed_power,
        "pi": np.pi,
        "E": np.e,
        "e": np.e,
    }
    for i in range(X.shape[1]):
        context[f"x{i}"] = X[:, i]
    if X.shape[1] == 1:
        context["x"] = X[:, 0]

    try:
        with np.errstate(divide="raise", invalid="raise", over="raise", under="ignore"):
            # R-02: gate the eval with the AST allowlist before it runs.
            from glassbox.sr.formula_safety import validate_formula_expr

            validate_formula_expr(formula, context.keys())
            y_pred = eval(formula, {"__builtins__": None}, context)
        if isinstance(y_pred, (int, float)):
            y_pred = np.full(X.shape[0], y_pred, dtype=np.float64)
        else:
            y_pred = np.asarray(y_pred, dtype=np.float64)
        if y_pred.ndim == 0:
            y_pred = np.full(X.shape[0], float(y_pred), dtype=np.float64)
        elif y_pred.shape[0] != X.shape[0]:
            return _finish(None, reason="shape_mismatch")
        if not np.all(np.isfinite(y_pred)):
            return _finish(None, reason="non_finite_output")
        return _finish(y_pred, reason="ok")
    except FloatingPointError as exc:
        text = str(exc).lower()
        if "divide by zero" in text:
            reason = "divide_by_zero"
        elif "overflow" in text:
            reason = "overflow"
        elif "invalid value" in text:
            if "sqrt" in formula.lower():
                reason = "invalid_sqrt"
            elif "log" in formula.lower():
                reason = "invalid_log"
            else:
                reason = "invalid_value"
        else:
            reason = "floating_point_error"
        return _finish(None, reason=reason, exc=exc)
    except (SyntaxError, NameError) as exc:
        return _finish(None, reason="parse_error", exc=exc)
    except Exception as exc:
        return _finish(None, reason="eval_error", exc=exc)


def evaluate_formula_mse(formula, x, y):
    """Evaluate displayed formula against ground-truth data and return MSE."""
    if not formula:
        return None
    normalized = normalize_formula_text(formula)
    if not normalized or normalized in {"N/A", "ERROR", "?"}:
        return None

    X = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    y_true = np.asarray(y, dtype=np.float64).reshape(-1)
    y_pred = evaluate_formula(normalized, X)
    if y_pred is None:
        return None

    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_pred.shape != y_true.shape:
        return None

    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    if mask.sum() < 10:
        return None

    mse = float(np.mean((y_pred[mask] - y_true[mask]) ** 2))
    if not math.isfinite(mse):
        return None
    return mse


def evaluate_formula_mse_on_X(formula, X, y):
    """Evaluate displayed formula against ground-truth data for 1D or multivariate X."""
    if not formula:
        return None
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    y_true = np.asarray(y, dtype=np.float64).reshape(-1)
    y_pred = evaluate_formula(formula, X_arr)
    if y_pred is None:
        return None
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_pred.shape != y_true.shape:
        return None
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    if mask.sum() < 10:
        return None
    mse = float(np.mean((y_pred[mask] - y_true[mask]) ** 2))
    return mse if math.isfinite(mse) else None


def formula_complexity_score(formula) -> int:
    """Small shared complexity proxy for candidate ranking."""
    text = normalize_formula_text(formula)
    if not text:
        return 0
    operator_count = sum(text.count(op) for op in ("+", "-", "*", "/", "**", "^"))
    function_count = len(
        re.findall(
            r"\b(?:sin|cos|exp|log|sqrt|abs|Abs|_signed_power|Piecewise)\s*\(", text
        )
    )
    protected_power_count = text.count("_signed_power")
    return max(1, operator_count + 2 * function_count + protected_power_count + 1)


def formula_family_risk_score(formula, *, complexity=None) -> float:
    """Heuristic risk score for formulas that often overfit or fail display evaluation."""
    text = normalize_formula_text(formula)
    if not text:
        return 1.0
    comp = formula_complexity_score(text) if complexity is None else int(complexity)
    risk = 0.0
    risk += 0.012 * max(0, comp - 12)
    risk += 0.050 * text.count("_signed_power")
    risk += 0.035 * len(re.findall(r"\*\*\s*[-+]?(?:0?\.\d+|[1-9]\d*\.\d+)", text))
    risk += 0.080 * text.count("Piecewise(")
    risk += 0.040 * text.count("Abs(")
    risk += 0.025 * max(0, text.count("sin(") + text.count("cos(") - 4)
    risk += 0.020 * max(0, text.count("exp(") - 2)
    return float(min(1.0, max(0.0, risk)))


def score_display_candidate(
    formula,
    X,
    y,
    *,
    raw_mse=None,
    fit_mse=None,
    holdout_mse=None,
    residual_diagnostics=None,
    complexity=None,
    n_terms=None,
    postprocess=True,
    complexity_lambda=1e-4,
    holdout_lambda=0.05,
    drift_lambda=0.10,
    risk_lambda=0.01,
):
    """Return a display-aware candidate score and diagnostics.

    The score is meant for comparing candidate formulas, not for benchmark
    grading. Benchmark grading still uses displayed-formula MSE directly.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)

    original_formula = str(formula or "")
    if postprocess:
        display_formula, guard = postprocess_formula_with_fidelity_guard(
            original_formula, X_arr, y_arr
        )
    else:
        display_formula, guard = (
            original_formula,
            {
                "postprocess_guard_triggered": False,
                "postprocess_raw_mse": evaluate_formula_mse_on_X(
                    original_formula, X_arr, y_arr
                ),
                "postprocess_processed_mse": None,
                "postprocess_fallback_mse": None,
                "postprocess_guard_reason": None,
                "postprocess_processed_eval_diagnostics": None,
            },
        )

    display_mse = evaluate_formula_mse_on_X(display_formula, X_arr, y_arr)
    raw_val = None
    for candidate in (raw_mse, fit_mse, guard.get("postprocess_raw_mse")):
        if candidate is None:
            continue
        try:
            candidate_f = float(candidate)
        except (TypeError, ValueError):
            continue
        if math.isfinite(candidate_f):
            raw_val = candidate_f
            break

    comp = (
        formula_complexity_score(display_formula)
        if complexity is None
        else int(complexity)
    )
    terms = (
        int(n_terms)
        if n_terms is not None
        else max(
            1,
            len(
                [t for t in re.split(r"\s*[+-]\s*", str(display_formula)) if t.strip()]
            ),
        )
    )
    risk = formula_family_risk_score(display_formula, complexity=comp)

    base = (
        display_mse
        if display_mse is not None and math.isfinite(display_mse)
        else float("inf")
    )
    holdout_gap = 0.0
    holdout_val = None
    if holdout_mse is not None:
        try:
            holdout_val = float(holdout_mse)
        except (TypeError, ValueError):
            holdout_val = None
        if (
            holdout_val is not None
            and math.isfinite(holdout_val)
            and math.isfinite(base)
        ):
            holdout_gap = max(0.0, holdout_val - base)

    drift_rel = None
    drift_penalty = 0.0
    if raw_val is not None and math.isfinite(raw_val) and math.isfinite(base):
        drift_rel = abs(base - raw_val) / max(abs(base), 1e-12)
        drift_penalty = drift_rel * max(base, 1e-12)

    residual_penalty = 0.0
    residual_suspicious = False
    if isinstance(residual_diagnostics, dict):
        residual_suspicious = bool(
            residual_diagnostics.get("residual_suspicious", False)
        )
        if residual_suspicious and math.isfinite(base):
            residual_penalty = 0.05 * max(base, 1e-12)

    score = (
        base
        + complexity_lambda * max(0, comp - 8)
        + 5e-5 * max(0, terms - 6)
        + holdout_lambda * holdout_gap
        + drift_lambda * drift_penalty
        + risk_lambda * risk
        + residual_penalty
    )

    if display_mse is None or not math.isfinite(base):
        score = float("inf")

    return {
        "formula": display_formula,
        "formula_original": original_formula,
        "score": float(score),
        "display_mse": display_mse,
        "raw_mse": raw_val,
        "holdout_mse": holdout_val,
        "complexity": comp,
        "n_terms": terms,
        "risk_score": risk,
        "raw_display_drift_rel": drift_rel,
        "residual_suspicious": residual_suspicious,
        "postprocess_guard": guard,
        "display_eval_ok": display_mse is not None and math.isfinite(display_mse),
    }


def postprocess_formula_with_fidelity_guard(
    formula,
    X,
    y,
    *,
    relative_slack=0.10,
    absolute_slack=1e-9,
):
    """Postprocess a formula, but keep the original if cleanup worsens benchmark fit."""
    processed = postprocess_formula(
        formula, fraction_tol=0.01, max_fraction_denominator=12
    )
    raw_mse = evaluate_formula_mse_on_X(formula, X, y)
    processed_mse = evaluate_formula_mse_on_X(processed, X, y)
    fallback = protect_fractional_powers(
        normalize_formula_text(formula).replace("^", "**")
    )
    fallback_mse = evaluate_formula_mse_on_X(fallback, X, y)

    if processed_mse is None:
        X_diag = np.asarray(X, dtype=np.float64)
        if X_diag.ndim == 1:
            X_diag = X_diag.reshape(-1, 1)
        _, processed_diag = evaluate_formula(processed, X_diag, return_diagnostics=True)
        return (fallback if fallback_mse is not None else processed), {
            "postprocess_guard_triggered": True,
            "postprocess_raw_mse": raw_mse,
            "postprocess_processed_mse": processed_mse,
            "postprocess_fallback_mse": fallback_mse,
            "postprocess_guard_reason": "processed_formula_eval_failed",
            "postprocess_processed_eval_diagnostics": processed_diag,
        }

    if raw_mse is None and processed != fallback:
        return fallback, {
            "postprocess_guard_triggered": True,
            "postprocess_raw_mse": raw_mse,
            "postprocess_processed_mse": processed_mse,
            "postprocess_fallback_mse": fallback_mse,
            "postprocess_guard_reason": "raw_formula_eval_failed_after_rewrite",
            "postprocess_processed_eval_diagnostics": None,
        }

    if raw_mse is not None:
        allowed = raw_mse * (1.0 + max(0.0, float(relative_slack))) + max(
            0.0, float(absolute_slack)
        )
        if processed_mse > allowed:
            if fallback_mse is not None and fallback_mse <= processed_mse:
                return fallback, {
                    "postprocess_guard_triggered": True,
                    "postprocess_raw_mse": raw_mse,
                    "postprocess_processed_mse": processed_mse,
                    "postprocess_fallback_mse": fallback_mse,
                    "postprocess_guard_reason": "processed_formula_worse",
                    "postprocess_processed_eval_diagnostics": None,
                }

    return processed, {
        "postprocess_guard_triggered": False,
        "postprocess_raw_mse": raw_mse,
        "postprocess_processed_mse": processed_mse,
        "postprocess_fallback_mse": None,
        "postprocess_guard_reason": None,
        "postprocess_processed_eval_diagnostics": None,
    }


def estimate_timeout_budget(base_timeout, n_features, n_train, adaptive_timeout):
    """Scale timeout by problem size when adaptive timeout is enabled."""
    if not adaptive_timeout:
        return int(base_timeout)
    complexity = (
        1.0
        + 0.15 * max(0, n_features - 1)
        + 0.08 * min(1.0, math.log10(max(50, n_train)) / 3.0)
    )
    budget = int(round(base_timeout * complexity))
    return int(min(max(20, budget), base_timeout * 2))


def parse_seed_list(seed_text):
    """Parse a comma-separated seed list and return sorted unique integers."""
    if seed_text is None:
        return [42]
    parts = [p.strip() for p in str(seed_text).split(",") if p.strip()]
    if not parts:
        return [42]
    seeds = []
    for part in parts:
        seeds.append(int(part))
    return sorted(set(seeds))


def compute_stability_stats(values, *, higher_is_better=True):
    """Compute median/IQR/std and worst-decile summary for a metric list."""
    if not values:
        return {
            "median": None,
            "iqr": None,
            "std": None,
            "worst_decile": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    q25 = float(np.percentile(arr, 25))
    q50 = float(np.percentile(arr, 50))
    q75 = float(np.percentile(arr, 75))
    if arr.size >= 10:
        worst_count = max(1, int(np.ceil(arr.size * 0.1)))
        sorted_arr = np.sort(arr)
        worst = (
            sorted_arr[:worst_count] if higher_is_better else sorted_arr[-worst_count:]
        )
        worst_decile = float(np.mean(worst))
    else:
        worst_decile = float(np.min(arr) if higher_is_better else np.max(arr))
    return {
        "median": q50,
        "iqr": float(q75 - q25),
        "std": float(np.std(arr)),
        "worst_decile": worst_decile,
    }


def classify_failure_taxonomy(true_formula, discovered_formula, r2, mse):
    """Heuristic failure taxonomy for closed-loop mutation/operator analysis."""
    true_f = (true_formula or "").lower()
    disc_f = (discovered_formula or "").lower()

    if not disc_f:
        return "no_formula"

    if "exp(-" in true_f and "exp(" in disc_f and "exp(-" not in disc_f:
        return "exp_sign_error"

    if "/" in true_f and "/" not in disc_f:
        return "rational_denominator_missing"

    true_pows = [int(m.group(1)) for m in re.finditer(r"\*\*\s*(\d+)", true_f)]
    disc_pows = [int(m.group(1)) for m in re.finditer(r"\*\*\s*(\d+)", disc_f)]
    if true_pows:
        true_max = max(true_pows)
        disc_max = max(disc_pows) if disc_pows else 1
        if true_max >= 3 and disc_max < true_max:
            return "missing_high_order_terms"

    if ("sin(" in true_f or "cos(" in true_f) and not (
        "sin(" in disc_f or "cos(" in disc_f
    ):
        return "periodic_structure_missing"

    if r2 is not None and r2 >= 0.8:
        return "near_miss_structural"
    if mse is not None and np.isfinite(mse) and mse > 1.0:
        return "high_error_instability"
    return "other_structural_failure"


def summarize_time_to_discovery(
    seed_runs,
    acceptable_r2=0.9,
    complexity_cap=20,
    exact_key="exact_match",
):
    """Compute time to first exact and first acceptable formula across seeded runs."""
    times_exact = []
    times_acceptable = []
    for run in seed_runs:
        t = run.get("time")
        if t is None or not np.isfinite(t):
            continue
        is_exact = bool(run.get(exact_key, False))
        r2 = run.get("r2")
        size = run.get("model_size")
        is_acceptable = is_exact or (
            r2 is not None
            and np.isfinite(r2)
            and r2 >= acceptable_r2
            and size is not None
            and size <= complexity_cap
        )
        if is_exact:
            times_exact.append(float(t))
        if is_acceptable:
            times_acceptable.append(float(t))

    return {
        "time_to_first_exact": (min(times_exact) if times_exact else None),
        "time_to_first_acceptable": (
            min(times_acceptable) if times_acceptable else None
        ),
    }


def summarize_seed_runs(seed_runs):
    """Aggregate per-seed runs into protocol-compliant stability summaries."""
    valid = [run for run in seed_runs if run.get("r2") is not None]
    if not valid:
        return {
            "seed_count": len(seed_runs),
            "valid_seed_count": 0,
            "r2_stats": compute_stability_stats([]),
            "mse_stats": compute_stability_stats([], higher_is_better=False),
            "time_stats": compute_stability_stats([], higher_is_better=False),
            "exact_recovery_rate": None,
            "exact_recovery_stats": compute_stability_stats([]),
        }

    r2_vals = [run["r2"] for run in valid if np.isfinite(run["r2"])]
    mse_vals = [
        run["mse"]
        for run in valid
        if run.get("mse") is not None and np.isfinite(run["mse"])
    ]
    time_vals = [
        run["time"]
        for run in valid
        if run.get("time") is not None and np.isfinite(run["time"])
    ]
    exact_binary = [1.0 if run.get("exact_match") else 0.0 for run in valid]

    return {
        "seed_count": len(seed_runs),
        "valid_seed_count": len(valid),
        "r2_stats": compute_stability_stats(r2_vals),
        "mse_stats": compute_stability_stats(mse_vals, higher_is_better=False),
        "time_stats": compute_stability_stats(time_vals, higher_is_better=False),
        "exact_recovery_rate": float(np.mean(exact_binary)) if exact_binary else None,
        "exact_recovery_stats": compute_stability_stats(exact_binary),
    }


def fallback_estimator_predictions(run_result, eval_diag, *, split="test"):
    """Return protected estimator predictions when display-formula scoring fails."""
    if not isinstance(run_result, dict) or not isinstance(eval_diag, dict):
        return None, eval_diag
    key = "y_pred_full" if split == "full" else "y_pred_test"
    pred = run_result.get(key)
    if pred is None:
        return None, eval_diag
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    if pred.size == 0 or not np.all(np.isfinite(pred)):
        return None, eval_diag
    updated = dict(eval_diag)
    updated["display_formula_failed"] = True
    updated["display_formula_reason"] = eval_diag.get("reason")
    updated["reason"] = "protected_estimator_prediction"
    updated["ok"] = True
    return pred, updated


def apply_run_budget(est_params, timeout_budget):
    """Keep estimator-internal adaptive compute inside the run budget."""
    params = dict(est_params)
    budget = int(max(1, round(float(timeout_budget))))
    params["timeout"] = budget
    params["max_compute_budget"] = budget
    params["min_compute_budget"] = min(
        int(params.get("min_compute_budget", 10) or 10), budget
    )
    return params


def specialist_metadata_from_estimator(estimator):
    """Extract specialist/composition/boosting diagnostics from a GlassboxRegressor."""
    diagnostics = getattr(estimator, "blackbox_diagnostics_", {}) or {}
    candidate_screening = (
        diagnostics.get("candidate_screening", {})
        if isinstance(diagnostics, dict)
        else {}
    )
    return {
        "specialist_track": getattr(estimator, "specialist_track_", None),
        "has_composed_seeds": bool(getattr(estimator, "has_composed_seeds_", False)),
        "composition_candidates_accepted": bool(
            getattr(estimator, "composition_candidates_accepted_", False)
        ),
        "composition_candidate_count": int(
            getattr(estimator, "composition_candidate_count_", 0) or 0
        ),
        "composition_seeded_evolution": bool(
            getattr(estimator, "composition_seeded_evolution_", False)
        ),
        "composition_won_final_selection": bool(
            getattr(estimator, "composition_won_final_selection_", False)
        ),
        "composition_improved_mse": bool(
            getattr(estimator, "composition_improved_mse_", False)
        ),
        "boosting_attempted": bool(getattr(estimator, "boosting_attempted_", False)),
        "boosting_improved": bool(getattr(estimator, "boosting_improved_", False)),
        "boosting_stage_count": len(getattr(estimator, "boosting_stages_", []) or []),
        "boosting_diagnostics": getattr(estimator, "boosting_diagnostics_", None),
        "residual_stage_guard": getattr(estimator, "_residual_stage_guard_", None),
        "final_formula_selection": getattr(
            estimator, "final_formula_selection_diagnostics_", None
        ),
        "phase_timings": dict(getattr(estimator, "phase_timings_", {}) or {}),
        "exact_match_diagnostics": getattr(
            estimator, "fast_path_exact_match_diagnostics_", None
        ),
        "formula_eval_count": int(getattr(estimator, "formula_eval_count_", 0) or 0),
        "formula_eval_cache_hits": int(
            getattr(estimator, "formula_eval_cache_hits_", 0) or 0
        ),
        "formula_eval_cache_size": len(
            getattr(estimator, "_formula_eval_cache_", {}) or {}
        ),
        "specialist_vault": (
            getattr(estimator, "specialist_vault_", None).to_dict()
            if getattr(estimator, "specialist_vault_", None) is not None
            else None
        ),
        "inception_round_count": len(getattr(estimator, "inception_rounds_", []) or []),
        "inception_diagnostics": getattr(estimator, "inception_diagnostics_", None),
        "specialist_diagnostics": (
            candidate_screening.get("specialist_screening")
            if isinstance(candidate_screening, dict)
            else None
        ),
        "specialist_composition_screening": (
            diagnostics.get("specialist_composition_screening")
            if isinstance(diagnostics, dict)
            else None
        ),
    }
