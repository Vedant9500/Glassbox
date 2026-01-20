import numpy as np

class Operation:
    """Base class for all mathematical operations (neurons)."""
    name = "Op"
    arity = 0  # Number of inputs required

    def __call__(self, *args):
        raise NotImplementedError

    def __repr__(self):
        return self.name

# --- Binary Operations ---

class Add(Operation):
    name = "+"
    arity = 2
    def __call__(self, x, y):
        # Element-wise addition, safe for numpy arrays
        return np.add(x, y)

class Sub(Operation):
    name = "-"
    arity = 2
    def __call__(self, x, y):
        return np.subtract(x, y)

class Mul(Operation):
    name = "*"
    arity = 2
    def __call__(self, x, y):
        return np.multiply(x, y)

class Div(Operation):
    name = "/"
    arity = 2
    def __call__(self, x, y):
        # Protected division to avoid NaNs during random evolution
        return np.divide(x, np.where(np.abs(y) < 1e-7, 1e-7, y))

# --- Unary Operations ---

class Sin(Operation):
    name = "sin"
    arity = 1
    def __call__(self, x):
        return np.sin(x)

class Cos(Operation):
    name = "cos"
    arity = 1
    def __call__(self, x):
        return np.cos(x)

class Exp(Operation):
    name = "exp"
    arity = 1
    def __call__(self, x):
        # Clip to avoid overflow
        return np.exp(np.clip(x, -20, 20))

class Log(Operation):
    name = "log"
    arity = 1
    def __call__(self, x):
        # Protected log
        return np.log(np.abs(x) + 1e-7)

# --- Terminals ---

class Constant(Operation):
    name = "Const"
    arity = 0
    def __init__(self, value=1.0):
        self.value = value
    
    def __call__(self):
        return self.value
    
    def __repr__(self):
        return f"{self.value:.2f}"

class Variable(Operation):
    name = "Var"
    arity = 0
    def __init__(self, name="x"):
        self.var_name = name
    
    def __call__(self, context):
        # Context is a dict of {var_name: value}
        return context.get(self.var_name, 0.0)
    
    def __repr__(self):
        return self.var_name

# Registry of non-terminal operations for mutation
PRIMITIVES = [Add, Sub, Mul, Div, Sin, Cos, Exp, Log]
