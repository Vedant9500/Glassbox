"""
Build C++ IndividualGraph seed dicts from formula strings.

Used to inject fast-path / proposer skeletons into evolution initialization
(seed_graphs_py on _core.run_evolution).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import math

try:
    import _core  # type: ignore
except Exception:  # pragma: no cover - fallback for direct execution
    try:
        import _core  # type: ignore
    except Exception:  # pragma: no cover
        _core = None

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

# Match glassbox/sr/cpp/export_pytorch.py and core.cpp enums
TYPE_INPUT = 0
TYPE_CONSTANT = 1
TYPE_UNARY = 2
TYPE_BINARY = 3

UNARY_PERIODIC = 0
UNARY_POWER = 1
UNARY_INTPOW = 2
UNARY_EXP = 3
UNARY_LOG = 4

BINARY_ARITHMETIC = 0
BINARY_DIVISION = 1
BINARY_AGGREGATION = 2

_TRANSFORMATIONS = standard_transformations + (
    convert_xor,
    implicit_multiplication_application,
)

_LOCAL_DICT = {
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
}


def _cpp_seed_graph_from_formula(formula: str, x_name: str = "x") -> Optional[Dict[str, Any]]:
    if _core is None or not hasattr(_core, "formula_to_seed_graph"):
        return None
    try:
        graph = _core.formula_to_seed_graph(formula)
        if graph is None:
            return None
        if x_name != "x":
            return None
        return graph
    except Exception:
        return None


def _normalize_formula_text(formula: str) -> str:
    text = str(formula).strip()
    text = text.replace("^", "**")
    text = re.sub(r"\|([^|]+)\|", r"abs(\1)", text)
    return text


def _parse_formula_expr(formula: str) -> Optional[sp.Expr]:
    text = _normalize_formula_text(formula)
    if not text:
        return None
    try:
        expr = parse_expr(
            text,
            local_dict=dict(_LOCAL_DICT),
            transformations=_TRANSFORMATIONS,
            evaluate=False,
        )
    except Exception:
        return None
    return sp.expand(expr)


def _default_node(**overrides: Any) -> Dict[str, Any]:
    node = {
        "type": TYPE_CONSTANT,
        "feature_idx": 0,
        "value": 0.0,
        "unary_op": UNARY_PERIODIC,
        "binary_op": BINARY_ARITHMETIC,
        "p": 1.0,
        "omega": 1.0,
        "phi": 0.0,
        "amplitude": 1.0,
        "beta": 1.0,
        "gamma": 1.0,
        "tau": 1.0,
        "left_child": -1,
        "right_child": -1,
    }
    node.update(overrides)
    return node


class _GraphBuilder:
    """Incremental AST → C++ graph (additive output layer)."""

    def __init__(self, x_sym: sp.Symbol) -> None:
        self.x_sym = x_sym
        self.nodes: List[Dict[str, Any]] = []
        self.output_weights: List[float] = []
        self.output_bias = 0.0

    def _append(self, node: Dict[str, Any]) -> int:
        idx = len(self.nodes)
        self.nodes.append(node)
        self.output_weights.append(0.0)
        return idx

    def _input_node(self) -> int:
        if not self.nodes:
            return self._append(_default_node(type=TYPE_INPUT, feature_idx=0))
        return 0

    def _set_root_weight(self, node_idx: int, weight: float) -> None:
        if 0 <= node_idx < len(self.output_weights):
            self.output_weights[node_idx] = float(weight)

    def _mul_node(self, left: int, right: int) -> int:
        idx = self._append(
            _default_node(
                type=TYPE_BINARY,
                binary_op=BINARY_ARITHMETIC,
                beta=2.0,
                gamma=1.0,
                left_child=left,
                right_child=right,
            )
        )
        self.output_weights[left] = 0.0
        self.output_weights[right] = 0.0
        return idx

    def _div_node(self, left: int, right: int) -> int:
        idx = self._append(
            _default_node(
                type=TYPE_BINARY,
                binary_op=BINARY_DIVISION,
                beta=2.0,
                gamma=-1.0,
                left_child=left,
                right_child=right,
            )
        )
        self.output_weights[left] = 0.0
        self.output_weights[right] = 0.0
        return idx

    def _linear_in_x(self, expr: sp.Expr) -> Optional[Tuple[float, float]]:
        """Return (omega, phi) for expr ≈ omega*x + phi, else None."""
        if expr == self.x_sym:
            return 1.0, 0.0
        if expr.is_Number:
            return 0.0, float(expr)
        if isinstance(expr, sp.Mul):
            omega = 1.0
            has_x = False
            for factor in expr.args:
                if factor == self.x_sym:
                    has_x = True
                elif factor.is_Number:
                    omega *= float(factor)
                else:
                    return None
            if has_x:
                return omega, 0.0
            return None
        if isinstance(expr, sp.Add):
            omega = 0.0
            phi = 0.0
            for term in expr.args:
                parsed = self._linear_in_x(term)
                if parsed is None:
                    return None
                o, p = parsed
                omega += o
                phi += p
            return omega, phi
        return None

    def build(self, expr: sp.Expr) -> Optional[int]:
        expr = sp.expand(expr)

        if expr.is_Number or isinstance(expr, (sp.Integer, sp.Float, sp.Rational)):
            idx = self._append(
                _default_node(type=TYPE_CONSTANT, value=float(expr))
            )
            return idx

        if expr == self.x_sym:
            return self._input_node()

        if isinstance(expr, sp.Symbol):
            if str(expr) == str(self.x_sym):
                return self._input_node()
            return None

        if isinstance(expr, sp.Mul):
            coeff = 1.0
            numer: List[sp.Expr] = []
            denom: List[sp.Expr] = []
            for factor in expr.args:
                if factor.is_Number:
                    coeff *= float(factor)
                elif isinstance(factor, sp.Pow) and factor.exp.is_Number and float(factor.exp) < 0:
                    denom.append(factor.base)
                else:
                    numer.append(factor)

            if denom:
                num_expr = sp.Mul(*numer) if numer else sp.S.One
                den_expr = sp.Mul(*denom) if len(denom) > 1 else denom[0]
                left = self.build(num_expr)
                right = self.build(den_expr)
                if left is None or right is None:
                    return None
                div_idx = self._div_node(left, right)
                self._set_root_weight(div_idx, coeff)
                return div_idx

            if not numer:
                return self._append(_default_node(type=TYPE_CONSTANT, value=coeff))

            built: List[int] = []
            for part in numer:
                child = self.build(part)
                if child is None:
                    return None
                built.append(child)

            prod = built[0]
            for child in built[1:]:
                prod = self._mul_node(prod, child)
            self._set_root_weight(prod, coeff)
            return prod

        if isinstance(expr, sp.Add):
            term_nodes: List[Tuple[int, float]] = []
            for term in expr.args:
                if term.is_Number:
                    self.output_bias += float(term)
                    continue
                coeff = 1.0
                core = term
                if isinstance(term, sp.Mul):
                    c, rest = term.as_coeff_Mul()
                    coeff = float(c)
                    if rest is sp.S.One:
                        core = sp.S.One
                    else:
                        core = rest
                node = self.build(core)
                if node is None:
                    return None
                term_nodes.append((node, coeff))

            if not term_nodes:
                idx = self._append(_default_node(type=TYPE_CONSTANT, value=0.0))
                return idx
            if len(term_nodes) == 1:
                node, coeff = term_nodes[0]
                self._set_root_weight(node, coeff)
                return node
            for node, coeff in term_nodes:
                self._set_root_weight(node, coeff)
            return term_nodes[-1][0]

        if isinstance(expr, sp.Pow) or getattr(expr, "func", None) == sp.Pow:
            base, exp = expr.args
            if exp.is_Number:
                p_val = float(exp)
                if abs(p_val) < 1e-12:
                    return self._append(_default_node(type=TYPE_CONSTANT, value=1.0))
                if abs(p_val - 1.0) < 1e-12:
                    base_idx = self.build(base)
                    return base_idx
            base_idx = self.build(base)
            if base_idx is None:
                return None
            if exp.is_Number:
                p_val = float(exp)
                if abs(p_val - round(p_val)) < 1e-9 and 2 <= int(round(p_val)) <= 6:
                    return self._append(
                        _default_node(
                            type=TYPE_UNARY,
                            unary_op=UNARY_INTPOW,
                            p=float(int(round(p_val))),
                            left_child=base_idx,
                        )
                    )
                return self._append(
                    _default_node(
                        type=TYPE_UNARY,
                        unary_op=UNARY_POWER,
                        p=p_val,
                        left_child=base_idx,
                    )
                )
            return None

        if getattr(expr, "func", None) == sp.sin:
            (arg,) = expr.args
            lin = self._linear_in_x(sp.expand(arg))
            if lin is None:
                return None
            omega, phi = lin
            return self._append(
                _default_node(
                    type=TYPE_UNARY,
                    unary_op=UNARY_PERIODIC,
                    omega=float(omega) if abs(omega) > 1e-15 else 1.0,
                    phi=float(phi),
                    amplitude=1.0,
                    left_child=self._input_node(),
                )
            )

        if getattr(expr, "func", None) == sp.cos:
            (arg,) = expr.args
            lin = self._linear_in_x(sp.expand(arg))
            if lin is None:
                return None
            omega, phi = lin
            return self._append(
                _default_node(
                    type=TYPE_UNARY,
                    unary_op=UNARY_PERIODIC,
                    omega=float(omega) if abs(omega) > 1e-15 else 1.0,
                    phi=float(phi) + math.pi / 2.0,
                    amplitude=1.0,
                    left_child=self._input_node(),
                )
            )

        if getattr(expr, "func", None) == sp.exp:
            (arg,) = expr.args
            lin = self._linear_in_x(sp.expand(arg))
            if lin is not None:
                omega, phi = lin
                return self._append(
                    _default_node(
                        type=TYPE_UNARY,
                        unary_op=UNARY_EXP,
                        omega=float(omega) if abs(omega) > 1e-15 else 1.0,
                        phi=float(phi),
                        left_child=self._input_node(),
                    )
                )
            inner = self.build(arg)
            if inner is None:
                return None
            return self._append(
                _default_node(
                    type=TYPE_UNARY,
                    unary_op=UNARY_EXP,
                    omega=1.0,
                    phi=0.0,
                    left_child=inner,
                )
            )

        if getattr(expr, "func", None) == sp.log:
            (arg,) = expr.args
            inner = self.build(arg)
            if inner is None:
                return None
            return self._append(
                _default_node(
                    type=TYPE_UNARY,
                    unary_op=UNARY_LOG,
                    left_child=inner,
                )
            )

        if getattr(expr, "func", None) == sp.Abs:
            (arg,) = expr.args
            inner = self.build(arg)
            if inner is None:
                return None
            return self._append(
                _default_node(
                    type=TYPE_UNARY,
                    unary_op=UNARY_POWER,
                    p=1.0,
                    left_child=inner,
                )
            )

        if getattr(expr, "func", None) == sp.sqrt:
            (arg,) = expr.args
            inner = self.build(arg)
            if inner is None:
                return None
            return self._append(
                _default_node(
                    type=TYPE_UNARY,
                    unary_op=UNARY_POWER,
                    p=0.5,
                    left_child=inner,
                )
            )

        return None

    def to_graph_dict(self, root_idx: int) -> Dict[str, Any]:
        # build() sets weights for sums/products; only default single-node graphs.
        if abs(self.output_weights[root_idx]) < 1e-12:
            self._set_root_weight(root_idx, 1.0)
        return {
            "nodes": self.nodes,
            "output_weights": self.output_weights,
            "output_bias": float(self.output_bias),
        }


def formula_to_seed_graph(formula: str, x_name: str = "x") -> Optional[Dict[str, Any]]:
    """Parse a formula string into a C++-compatible seed graph dict."""
    cpp_graph = _cpp_seed_graph_from_formula(formula, x_name=x_name)
    if cpp_graph is not None:
        return cpp_graph

    expr = _parse_formula_expr(formula)
    if expr is None:
        return None

    free = expr.free_symbols
    if not free:
        builder = _GraphBuilder(sp.Symbol(x_name))
        val = float(expr.evalf())
        idx = builder._append(_default_node(type=TYPE_CONSTANT, value=val))
        return builder.to_graph_dict(idx)

    if len(free) != 1:
        return None
    x_sym = next(iter(free))
    if x_sym.name != x_name:
        return None

    builder = _GraphBuilder(x_sym)
    root = builder.build(expr)
    if root is None:
        return None

    return builder.to_graph_dict(root)


def build_seed_graphs_from_formulas(
    formulas: List[str],
    max_seeds: int = 8,
) -> List[Dict[str, Any]]:
    """Build up to max_seeds unique seed graphs from formula strings."""
    graphs: List[Dict[str, Any]] = []
    seen: set = set()

    for formula in formulas:
        if len(graphs) >= max_seeds:
            break
        text = str(formula or "").strip()
        if not text or text == "0":
            continue
        key = re.sub(r"\s+", "", text.lower())
        if key in seen:
            continue
        seen.add(key)

        graph = formula_to_seed_graph(text)
        if graph is None or not graph.get("nodes"):
            continue
        graphs.append(graph)

    return graphs


def build_seed_graphs_from_candidates(
    candidate_formulas: Optional[List[Dict[str, Any]]],
    max_seeds: int = 10,
) -> List[Dict[str, Any]]:
    """Extract formulas from fast-path / proposer candidate dicts and build seeds."""
    if not candidate_formulas:
        return []

    ordered: List[str] = []
    seen: set = set()

    def _add(formula: str) -> None:
        text = str(formula or "").strip()
        if not text or text == "0":
            return
        key = re.sub(r"\s+", "", text.lower())
        if key in seen:
            return
        seen.add(key)
        ordered.append(text)

    # Prefer lower-MSE candidates first
    ranked = sorted(
        candidate_formulas,
        key=lambda c: float(c.get("mse", float("inf")) or float("inf")),
    )
    for cand in ranked:
        _add(cand.get("formula", ""))

    return build_seed_graphs_from_formulas(ordered, max_seeds=max_seeds)
