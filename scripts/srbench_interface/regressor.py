import sys
from pathlib import Path

# Add project root to path to import glassbox
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from glassbox.sr.sklearn_wrapper import GlassboxRegressor
import sympy

# Define the estimator instance for SRBench
est = GlassboxRegressor(
    population_size=100,
    generations=1000,
    random_state=42, # SRBench usually controls seed via the environment or params
)

def model(est, X=None):
    """
    Returns the discovered model as a sympy-compatible string.
    SRBench uses this to evaluate the mathematical correctness/complexity.
    """
    if not hasattr(est, 'formula_'):
        return "0"
    
    formula = est.formula_
    
    # SRBench expects sympy-compatible string. 
    # Handle |x| -> abs(x)
    import re
    formula = re.sub(r'\|([^|]+)\|', r'abs(\1)', formula)
    
    # Glassbox uses ^ for power, which is sympy-compatible.
    # It also uses sin, cos, exp, log, sqrt which are standard.
    
    # Simple normalization if needed (glassbox already produces fairly clean strings)
    # Ensure constants are represented correctly
    try:
        # Test parse via sympy
        expr = sympy.parse_expr(formula.replace('^', '**'))
        return str(expr)
    except Exception as e:
        print(f"Sympy parse error in SRBench interface: {e}")
        return formula
