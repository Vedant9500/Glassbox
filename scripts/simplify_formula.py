"""Post-process ONN symbolic formulas with deterministic cleanup.

Pipeline:
1) Tolerance-based float snapping using a custom Python AST traversal.
2) Bottom-up algebraic simplification with SymPy.

Example:
    python scripts/simplify_formula.py "0.99999*x + 1.00001*x + 0.0000001*y"
"""

from __future__ import annotations

import ast
import argparse
import math
import re
from dataclasses import dataclass
from typing import Dict, Set

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


def sympy_simplify_formula(clean_formula: str, *, use_nsimplify: bool = True) -> sp.Expr:
    """Run deterministic bottom-up algebraic simplification with SymPy."""
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

    simplified = sp.simplify(expr)
    if use_nsimplify:
        simplified = sp.nsimplify(simplified)

    return simplified


def simplify_onn_formula(
    raw_formula: str,
    *,
    int_tol: float = 1e-5,
    zero_tol: float = 1e-8,
    use_nsimplify: bool = True,
) -> tuple[str, sp.Expr]:
    """Run the full ONN post-processing pipeline.

    Returns:
        (snapped_formula_string, simplified_sympy_expression)
    """
    cfg = SnapConfig(int_tol=int_tol, zero_tol=zero_tol)
    snapped = snap_formula_floats(raw_formula, cfg)
    simplified = sympy_simplify_formula(snapped, use_nsimplify=use_nsimplify)
    return snapped, simplified


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simplify bloated ONN formulas with float snapping + SymPy simplification.",
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

    args = parser.parse_args()
    raw_formula = args.formula if args.formula is not None else input("Raw formula: ").strip()

    if not raw_formula:
        raise ValueError("Formula cannot be empty.")

    snapped, simplified = simplify_onn_formula(
        raw_formula,
        int_tol=args.int_tol,
        zero_tol=args.zero_tol,
        use_nsimplify=not args.no_nsimplify,
    )

    print("Raw       :", raw_formula)
    print("Snapped   :", snapped)
    print("Simplified:", simplified)


if __name__ == "__main__":
    main()
