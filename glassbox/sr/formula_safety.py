"""R-02: AST-allowlist gate for Python ``eval`` of displayed formula strings.

The estimator, universal proposer, specialist composition, benchmark and
curve-data paths evaluate formula strings with
``eval(expr, {"__builtins__": None}, context)``.  Restricting builtins is
safer than bare ``eval`` but still permits expression-injection escapes via
attribute traversal / subscripting, e.g. ``x.__class__.__mro__[1]`` or
``().__class__.__base__.__subclasses__()``.  Every formula ``eval`` must be
gated by :func:`validate_formula_expr` so only the restricted arithmetic
expression language the system actually produces can be evaluated.
"""

from __future__ import annotations

import ast
from typing import Iterable, Set

# Node types a formula expression may contain.  Deliberately excludes
# attribute chains, subscripts, comprehensions, lambdas, imports, statements,
# boolean logic and comparison expressions - none of which the C++ formatter
# or grammar generators emit.
_ALLOWED_NODE_TYPES: tuple = (
    ast.Expression,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.FloorDiv,
    ast.MatMult,
    ast.UnaryOp,
    ast.UAdd,
    ast.USub,
    ast.Call,
    ast.keyword,
    ast.Attribute,
)

# Function/constant attributes allowed directly on the trusted ``np`` module.
_NP_ALLOWED_ATTRS: Set[str] = {
    "sin",
    "cos",
    "exp",
    "log",
    "sqrt",
    "abs",
    "sign",
    "sinh",
    "cosh",
    "tanh",
    "clip",
    "where",
    "maximum",
    "minimum",
    "power",
    "square",
    "floor",
    "ceil",
    "round",
    "pi",
    "e",
}


def formula_expression_is_safe(expr: str, allowed_names: Iterable[str]) -> bool:
    """Return ``True`` only if *expr* is a restricted arithmetic expression.

    *allowed_names* must contain every free variable the caller exposes in its
    ``eval`` context (e.g. ``sin``, ``exp``, ``x``, ``x0``, ``_signed_power``).
    The ``np`` name is always permitted, and attribute access is allowed only
    one level deep and only directly on ``np`` (trusted numpy module) with an
    allow-listed attribute.  Anything else - nested attribute chains,
    subscripts, comprehensions, lambdas, imports, statements - is rejected.
    """
    allowed = set(allowed_names)
    allowed.add("np")
    try:
        tree = ast.parse(expr, mode="eval")
    except (SyntaxError, ValueError, TypeError, RecursionError, MemoryError):
        return False

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODE_TYPES):
            return False
        if isinstance(node, ast.Name):
            if node.id not in allowed:
                return False
        elif isinstance(node, ast.Attribute):
            value = node.value
            if not (isinstance(value, ast.Name) and value.id == "np"):
                return False
            if node.attr not in _NP_ALLOWED_ATTRS:
                return False
    return True


def validate_formula_expr(expr: str, allowed_names: Iterable[str]) -> None:
    """Raise ``ValueError`` unless *expr* passes the R-02 allowlist."""
    if not formula_expression_is_safe(expr, allowed_names):
        raise ValueError(
            "formula expression rejected by R-02 allowlist "
            "(non-arithmetic or attribute-traversal construct)"
        )