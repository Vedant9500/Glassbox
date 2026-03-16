"""Post-process ONN symbolic formulas with deterministic cleanup.

Pipeline:
1) Tolerance-based float snapping using a custom Python AST traversal.
2) Multi-pass SymPy simplification with identity-aware transforms.
3) Optional approximate dominant-mode collapse for Fourier-like leftovers.

Example:
    python scripts/simplify_formula.py "0.99999*x + 1.00001*x + 0.0000001*y"
"""

from __future__ import annotations

import ast
import argparse
import math
import re
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Set

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)


@dataclass(frozen=True)
class SnapConfig:
    """Tolerance configuration for float snapping."""

    int_tol: float = 1e-5
    zero_tol: float = 1e-8


@dataclass(frozen=True)
class SimplifyPassConfig:
    """Configuration for iterative symbolic simplification passes."""

    max_passes: int = 6
    use_nsimplify: bool = True
    use_identities: bool = True
    approximate_trig: bool = False
    dominant_trig_ratio: float = 0.9
    small_term_ratio: float = 0.08


_NSIMPLIFY_CONSTANTS = (
    sp.pi,
    sp.E,
    sp.sqrt(2),
    sp.sqrt(3),
    sp.sqrt(5),
    (1 + sp.sqrt(5)) / 2,
)


def _normalize_formula_syntax(formula: str) -> str:
    """Normalize minor syntax differences before parsing.

    - Converts '^' to '**' for exponentiation compatibility with Python AST.
    - Converts common alias 'ln' -> 'log' so SymPy can parse consistently.
    """
    normalized = formula

    # Normalize common unicode/operator glyphs emitted by pretty-printing.
    normalized = (
        normalized
        .replace("·", "*")
        .replace("⋅", "*")
        .replace("•", "*")
        .replace("×", "*")
        .replace("÷", "/")
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
    )

    # Some consoles/codepages inject replacement chars for unsupported glyphs.
    normalized = normalized.replace("�", "*")

    # Convert known symbolic constants used by ONN pretty formatter.
    constant_replacements = {
        "π²": "(pi**2)",
        "e²": "(e**2)",
        "2π": "(2*pi)",
        "π/2": "(pi/2)",
        "π/3": "(pi/3)",
        "π/4": "(pi/4)",
        "π/6": "(pi/6)",
        "1/π": "(1/pi)",
        "2/π": "(2/pi)",
        "√2": "sqrt(2)",
        "√3": "sqrt(3)",
        "√5": "sqrt(5)",
        "φ": "((1+sqrt(5))/2)",
        "log₂(e)": "(log(E, 2))",
        "log₁₀(e)": "(log(E, 10))",
    }
    for src, dst in sorted(constant_replacements.items(), key=lambda kv: len(kv[0]), reverse=True):
        normalized = normalized.replace(src, dst)

    # Replace remaining pi glyphs and ln alias.
    normalized = normalized.replace("π", "pi")
    normalized = re.sub(r"\bln\(", "log(", normalized)

    # Convert absolute bars |expr| to abs(expr). Repeat for nested bars.
    abs_pattern = re.compile(r"\|([^|]+)\|")
    prev = None
    while prev != normalized:
        prev = normalized
        normalized = abs_pattern.sub(r"abs(\1)", normalized)

    # Convert e^(...) style to exp(...), and plain caret to Python power.
    normalized = re.sub(r"\be\^\(([^)]+)\)", r"exp(\1)", normalized)
    normalized = normalized.replace("^", "**")

    # Map standalone e to Euler's E constant for Python AST/SymPy parsing.
    normalized = re.sub(r"\be\b", "E", normalized)

    # Normalize occasional multiplication artifacts without touching exponentiation (**).
    normalized = re.sub(r"\*\s+\*", "*", normalized)
    normalized = normalized.replace("*+", "+").replace("*-", "-")
    return normalized


def _snap_float(value: float, cfg: SnapConfig) -> float | int:
    """Snap a float to 0 or nearest integer when inside tolerance windows."""
    if abs(value) <= cfg.zero_tol:
        return 0

    nearest_int = round(value)
    if math.isclose(value, nearest_int, rel_tol=0.0, abs_tol=cfg.int_tol + 1e-12):
        return int(nearest_int)

    return value


class FloatSnapTransformer(ast.NodeTransformer):
    """AST transformer that snaps noisy floating-point literals.

    Step-1 parser/traversal details:
    - Python's `ast.parse(..., mode="eval")` turns the formula into a syntax tree.
    - We walk the tree bottom-up via `NodeTransformer` and only modify numeric leaves.
    - Every float constant is checked against tolerance rules:
        * near-zero -> 0
        * near-integer -> integer
    - We also fold unary signs (`-1.0000001`) into a single literal before snapping,
      so negative coefficients are handled consistently.
    """

    def __init__(self, cfg: SnapConfig):
        super().__init__()
        self.cfg = cfg

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, float):
            snapped = _snap_float(node.value, self.cfg)
            return ast.copy_location(ast.Constant(value=snapped), node)
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        # Visit children first so traversal is bottom-up.
        node = self.generic_visit(node)

        if not isinstance(node.operand, ast.Constant):
            return node

        operand_val = node.operand.value
        if not isinstance(operand_val, (int, float)):
            return node

        if isinstance(node.op, ast.UAdd):
            signed_val = float(operand_val)
        elif isinstance(node.op, ast.USub):
            signed_val = -float(operand_val)
        else:
            return node

        snapped = _snap_float(signed_val, self.cfg)
        return ast.copy_location(ast.Constant(value=snapped), node)


def snap_formula_floats(raw_formula: str, cfg: SnapConfig) -> str:
    """Apply tolerance-based float snapping to all numeric literals in a formula."""
    normalized = _normalize_formula_syntax(raw_formula)

    try:
        parsed = ast.parse(normalized, mode="eval")
    except SyntaxError:
        # Fallback path: parse with SymPy and snap Float atoms directly.
        # This is more tolerant of non-Pythonic pretty-printed formulas.
        local_dict = _build_local_dict(normalized)
        transformations = standard_transformations + (
            convert_xor,
            implicit_multiplication_application,
        )
        try:
            expr = parse_expr(
                normalized,
                local_dict=local_dict,
                transformations=transformations,
                evaluate=False,
            )
            replacements = {}
            for atom in expr.atoms(sp.Float):
                snapped = _snap_float(float(atom), cfg)
                replacements[atom] = sp.Integer(snapped) if isinstance(snapped, int) else sp.Float(snapped)
            if replacements:
                expr = expr.xreplace(replacements)
            return str(expr)
        except Exception as exc:
            raise ValueError(f"Invalid formula syntax for AST parsing: {raw_formula}") from exc

    transformed = FloatSnapTransformer(cfg).visit(parsed)
    ast.fix_missing_locations(transformed)

    # `ast.unparse` reconstructs a valid expression string from the transformed tree.
    return ast.unparse(transformed)


def _discover_symbols(expr_text: str) -> Set[str]:
    """Infer variable names in the expression (excluding known function names)."""
    tokens = set(re.findall(r"\b[A-Za-z_]\w*\b", expr_text))
    reserved = {
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "sinh",
        "cosh",
        "tanh",
        "exp",
        "log",
        "sqrt",
        "abs",
        "sign",
        "Max",
        "Min",
        "pi",
        "E",
    }
    return {name for name in tokens if name not in reserved}


def _build_local_dict(expr_text: str) -> Dict[str, object]:
    """Build a parse dictionary with symbols and allowed math functions."""
    locals_dict: Dict[str, object] = {
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "asin": sp.asin,
        "acos": sp.acos,
        "atan": sp.atan,
        "sinh": sp.sinh,
        "cosh": sp.cosh,
        "tanh": sp.tanh,
        "exp": sp.exp,
        "log": sp.log,
        "sqrt": sp.sqrt,
        "abs": sp.Abs,
        "sign": sp.sign,
        "Max": sp.Max,
        "Min": sp.Min,
        "pi": sp.pi,
        "E": sp.E,
    }

    for sym_name in _discover_symbols(expr_text):
        locals_dict[sym_name] = sp.Symbol(sym_name)

    return locals_dict


def _expr_key(expr: sp.Expr) -> str:
    """Create a stable key used to detect fixed points across passes."""
    try:
        return sp.srepr(expr)
    except Exception:
        return str(expr)


def _expr_score(expr: sp.Expr) -> tuple[int, int]:
    """Rank candidates by symbolic operation count, then by string length."""
    try:
        op_count = int(sp.count_ops(expr, visual=False))
    except Exception:
        op_count = 10**9
    return op_count, len(str(expr))


def _safe_expr_transform(
    transform: Callable[[sp.Expr], sp.Expr],
    expr: sp.Expr,
) -> Optional[sp.Expr]:
    """Run a symbolic transform and safely return None on failure."""
    try:
        candidate = transform(expr)
    except Exception:
        return None

    if isinstance(candidate, sp.Expr):
        return candidate

    try:
        return sp.sympify(candidate)
    except Exception:
        return None


def _dedupe_exprs(candidates: Iterable[sp.Expr]) -> list[sp.Expr]:
    """Deduplicate symbolic candidates while preserving order."""
    unique: list[sp.Expr] = []
    seen: Set[str] = set()
    for candidate in candidates:
        key = _expr_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _nsimplify_with_constants(expr: sp.Expr) -> sp.Expr:
    """Snap decimal constants to compact symbolic forms when possible."""
    return sp.nsimplify(expr, constants=_NSIMPLIFY_CONSTANTS, rational=True)


def _iter_simplification_candidates(expr: sp.Expr, cfg: SimplifyPassConfig) -> list[sp.Expr]:
    """Generate candidate expressions for a single simplification pass."""
    transforms: list[Callable[[sp.Expr], sp.Expr]] = [
        sp.simplify,
        sp.factor_terms,
        lambda e: sp.powsimp(e, force=True),
        sp.together,
        sp.cancel,
        sp.ratsimp,
    ]

    if cfg.use_identities:
        transforms.extend(
            [
                lambda e: sp.trigsimp(e, method="fu"),
                lambda e: sp.trigsimp(sp.expand_trig(e), method="fu"),
                lambda e: sp.simplify(sp.trigsimp(e, method="fu")),
            ]
        )

    candidates: list[sp.Expr] = [expr]
    for transform in transforms:
        transformed = _safe_expr_transform(transform, expr)
        if transformed is not None:
            candidates.append(transformed)

    if cfg.use_nsimplify:
        for candidate in list(candidates):
            snapped = _safe_expr_transform(_nsimplify_with_constants, candidate)
            if snapped is not None:
                candidates.append(snapped)

    return _dedupe_exprs(candidates)


def _run_exact_multi_pass(expr: sp.Expr, cfg: SimplifyPassConfig) -> sp.Expr:
    """Run iterative exact simplification passes until a fixed point is reached."""
    current = expr
    for _ in range(max(1, int(cfg.max_passes))):
        candidates = _iter_simplification_candidates(current, cfg)
        best = min(candidates, key=_expr_score)
        if _expr_key(best) == _expr_key(current):
            break
        current = best
    return current


def _extract_numeric_linear_frequency(arg: sp.Expr, symbol: sp.Symbol) -> Optional[float]:
    """Extract |d(arg)/d(symbol)| when arg is linear in symbol, else None."""
    try:
        if not sp.diff(arg, symbol, 2).equals(0):
            return None

        slope = sp.diff(arg, symbol)
        if slope.free_symbols:
            return None

        slope_value = float(sp.N(slope))
        if not math.isfinite(slope_value):
            return None
        return abs(slope_value)
    except Exception:
        return None


def _extract_trig_term_signature(term: sp.Expr, symbol: sp.Symbol) -> Optional[tuple[float, float]]:
    """Extract (amplitude, frequency) from coeff*sin(a*x+b) or coeff*cos(a*x+b)."""
    coeff, remainder = term.as_coeff_Mul()
    if remainder.func not in (sp.sin, sp.cos):
        return None

    try:
        amplitude = abs(float(sp.N(coeff)))
    except Exception:
        return None

    if not math.isfinite(amplitude) or amplitude <= 0.0:
        return None

    frequency = _extract_numeric_linear_frequency(remainder.args[0], symbol)
    if frequency is None or frequency < 1e-12:
        return None

    return amplitude, frequency


def _collapse_dominant_trig_mode(expr: sp.Expr, cfg: SimplifyPassConfig) -> sp.Expr:
    """Optionally drop weak harmonic leftovers when one trig mode clearly dominates.

    This pass is intentionally conservative and disabled by default.
    """
    if not cfg.approximate_trig:
        return expr

    symbols = sorted(expr.free_symbols, key=lambda s: s.name)
    if len(symbols) != 1:
        return expr

    symbol = symbols[0]
    terms = list(sp.Add.make_args(sp.expand(expr)))

    grouped_terms: Dict[float, list[sp.Expr]] = {}
    grouped_weight: Dict[float, float] = {}
    non_trig_terms: list[sp.Expr] = []

    for term in terms:
        signature = _extract_trig_term_signature(term, symbol)
        if signature is None:
            non_trig_terms.append(term)
            continue

        amplitude, frequency = signature
        frequency_key = round(frequency, 4)
        grouped_terms.setdefault(frequency_key, []).append(term)
        grouped_weight[frequency_key] = grouped_weight.get(frequency_key, 0.0) + amplitude

    if not grouped_weight:
        return expr

    total_weight = sum(grouped_weight.values())
    dominant_frequency = max(grouped_weight, key=grouped_weight.get)
    dominant_weight = grouped_weight[dominant_frequency]
    residual_weight = total_weight - dominant_weight

    if total_weight <= 0:
        return expr
    if dominant_weight / total_weight < cfg.dominant_trig_ratio:
        return expr
    if residual_weight > max(cfg.small_term_ratio * dominant_weight, 1e-12):
        return expr

    collapsed = sp.Add(*(non_trig_terms + grouped_terms[dominant_frequency]))
    return sp.trigsimp(collapsed, method="fu")


def sympy_simplify_formula(
    clean_formula: str,
    *,
    use_nsimplify: bool = True,
    max_passes: int = 6,
    use_identities: bool = True,
    approximate_trig: bool = False,
    dominant_trig_ratio: float = 0.9,
    small_term_ratio: float = 0.08,
) -> sp.Expr:
    """Run deterministic multi-pass algebraic simplification with SymPy.

    The core loop applies several exact simplification transforms and keeps the
    smallest candidate each pass. An optional approximate pass can collapse weak
    Fourier-like harmonics when one mode clearly dominates.
    """
    transformations = standard_transformations + (
        convert_xor,
        implicit_multiplication_application,
    )

    local_dict = _build_local_dict(clean_formula)

    expr = parse_expr(
        clean_formula,
        local_dict=local_dict,
        transformations=transformations,
        evaluate=False,
    )

    cfg = SimplifyPassConfig(
        max_passes=max_passes,
        use_nsimplify=use_nsimplify,
        use_identities=use_identities,
        approximate_trig=approximate_trig,
        dominant_trig_ratio=dominant_trig_ratio,
        small_term_ratio=small_term_ratio,
    )

    simplified = _run_exact_multi_pass(expr, cfg)

    collapsed = _collapse_dominant_trig_mode(simplified, cfg)
    if _expr_key(collapsed) != _expr_key(simplified):
        # Re-run exact passes after dropping weak harmonics.
        exact_cfg = SimplifyPassConfig(
            max_passes=cfg.max_passes,
            use_nsimplify=cfg.use_nsimplify,
            use_identities=cfg.use_identities,
        )
        simplified = _run_exact_multi_pass(collapsed, exact_cfg)

    return simplified


def simplify_onn_formula(
    raw_formula: str,
    *,
    int_tol: float = 1e-5,
    zero_tol: float = 1e-8,
    use_nsimplify: bool = True,
    max_passes: int = 6,
    use_identities: bool = True,
    approximate_trig: bool = False,
    dominant_trig_ratio: float = 0.9,
    small_term_ratio: float = 0.08,
) -> tuple[str, sp.Expr]:
    """Run the full ONN post-processing pipeline.

    Returns:
        (snapped_formula_string, simplified_sympy_expression)
    """
    cfg = SnapConfig(int_tol=int_tol, zero_tol=zero_tol)
    snapped = snap_formula_floats(raw_formula, cfg)
    simplified = sympy_simplify_formula(
        snapped,
        use_nsimplify=use_nsimplify,
        max_passes=max_passes,
        use_identities=use_identities,
        approximate_trig=approximate_trig,
        dominant_trig_ratio=dominant_trig_ratio,
        small_term_ratio=small_term_ratio,
    )
    return snapped, simplified


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simplify bloated ONN formulas with float snapping + multi-pass SymPy simplification.",
    )
    parser.add_argument(
        "formula",
        nargs="?",
        help="Raw formula string. If omitted, you will be prompted.",
    )
    parser.add_argument(
        "--int-tol",
        type=float,
        default=1e-5,
        help="Tolerance for snapping a float to nearest integer (default: 1e-5).",
    )
    parser.add_argument(
        "--zero-tol",
        type=float,
        default=1e-8,
        help="Tolerance for snapping tiny values to 0 (default: 1e-8).",
    )
    parser.add_argument(
        "--no-nsimplify",
        action="store_true",
        help="Disable nsimplify after simplify().",
    )
    parser.add_argument(
        "--max-passes",
        type=int,
        default=6,
        help="Maximum identity/simplification passes (default: 6).",
    )
    parser.add_argument(
        "--no-identities",
        action="store_true",
        help="Disable identity-focused passes (trig/power rewrites).",
    )
    parser.add_argument(
        "--approx-trig",
        action="store_true",
        help="Allow approximate dominant-mode trig collapse for Fourier-like leftovers.",
    )
    parser.add_argument(
        "--dominant-trig-ratio",
        type=float,
        default=0.9,
        help="Required dominant trig energy ratio when --approx-trig is enabled (default: 0.9).",
    )
    parser.add_argument(
        "--small-term-ratio",
        type=float,
        default=0.08,
        help="Maximum residual harmonic ratio when --approx-trig is enabled (default: 0.08).",
    )

    args = parser.parse_args()
    raw_formula = args.formula if args.formula is not None else input("Raw formula: ").strip()

    if not raw_formula:
        raise ValueError("Formula cannot be empty.")
    if args.max_passes < 1:
        raise ValueError("--max-passes must be >= 1")
    if not (0.0 < args.dominant_trig_ratio <= 1.0):
        raise ValueError("--dominant-trig-ratio must be in (0, 1]")
    if args.small_term_ratio < 0.0:
        raise ValueError("--small-term-ratio must be >= 0")

    snapped, simplified = simplify_onn_formula(
        raw_formula,
        int_tol=args.int_tol,
        zero_tol=args.zero_tol,
        use_nsimplify=not args.no_nsimplify,
        max_passes=args.max_passes,
        use_identities=not args.no_identities,
        approximate_trig=args.approx_trig,
        dominant_trig_ratio=args.dominant_trig_ratio,
        small_term_ratio=args.small_term_ratio,
    )

    print("Raw       :", raw_formula)
    print("Snapped   :", snapped)
    print("Simplified:", simplified)


if __name__ == "__main__":
    main()
