import sys
from pathlib import Path

# Add project root to path to import glassbox
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import sympy

from glassbox.sr.sklearn_wrapper import GlassboxRegressor

# Define the estimator instance for SRBench
est = GlassboxRegressor(
    population_size=100,
    generations=1000,
    random_state=42,  # SRBench usually controls seed via the environment or params
)


def model(est, X=None):
    """
    Returns the discovered model as a sympy-compatible string.
    SRBench uses this to evaluate the mathematical correctness/complexity.
    """
    if not hasattr(est, "formula_"):
        return "0"

    formula = est.formula_

    # SRBench expects sympy-compatible string.
    # Handle |x| -> abs(x)
    import re

    formula = re.sub(r"\|([^|]+)\|", r"abs(\1)", formula)

    # Glassbox uses ^ for power, which is sympy-compatible.
    # It also uses sin, cos, exp, log, sqrt which are standard.

    # Simple normalization if needed (glassbox already produces fairly clean strings)
    # Ensure constants are represented correctly
    try:
        # §3.69: guard the previously unrestricted parse_expr. Fitted formula
        # strings are untrusted input here: parse with an explicit local
        # namespace (indexed features x0..xn plus the small constant/function
        # set glassbox emits) and reject anything else instead of letting
        # sympy construct arbitrary expressions.
        cleaned = formula.replace("^", "**")
        if "__" in cleaned:
            raise ValueError("dunder names are not allowed in SRBench formulas")
        indexed = sorted({int(m) for m in re.findall(r"x(\d+)", cleaned)})
        allowed_symbols = {f"x{i}" for i in indexed} | {"x"}
        allowed_names = (
            allowed_symbols
            | {"pi", "E", "e"}
            | {"sin", "cos", "tan", "exp", "log", "sqrt", "abs"}
        )
        for name in sorted(set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", cleaned))):
            if name not in allowed_names:
                raise ValueError(f"symbol {name!r} is not in the SRBench allowlist")
        local_dict = {name: sympy.Symbol(name) for name in allowed_symbols}
        local_dict.update(
            {
                "pi": sympy.pi,
                "E": sympy.E,
                "e": sympy.E,
                "sin": sympy.sin,
                "cos": sympy.cos,
                "tan": sympy.tan,
                "exp": sympy.exp,
                "log": sympy.log,
                "sqrt": sympy.sqrt,
                "abs": sympy.Abs,
            }
        )
        # Test parse via sympy
        expr = sympy.parse_expr(cleaned, local_dict=local_dict)
        free = {str(s) for s in expr.free_symbols}
        if not free <= allowed_symbols:
            raise ValueError(f"unexpected free symbols: {sorted(free)}")
        return str(expr)
    except Exception as e:
        print(f"Sympy parse error in SRBench interface: {e}")
        return formula
