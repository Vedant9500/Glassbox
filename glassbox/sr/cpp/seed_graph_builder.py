"""
Build C++ IndividualGraph seed dicts from formula strings.

Used to inject fast-path / proposer skeletons into evolution initialization
(seed_graphs_py on _core.run_evolution).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import math
import numpy as np

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


def _multi_feature_formula_to_seed_graph(formula: str) -> Optional[Dict[str, Any]]:
    expr = _parse_formula_expr(formula)
    if expr is None:
        return None

    free = sorted(expr.free_symbols, key=lambda s: str(s))
    if not free:
        builder = _GraphBuilder(sp.Symbol("x"))
        val = float(expr.evalf())
        idx = builder._append(_default_node(type=TYPE_CONSTANT, value=val))
        return builder.to_graph_dict(idx)

    if len(free) == 1:
        return None

    try:
        for sym in free:
            name = str(sym)
            if not re.fullmatch(r"x\d+", name) and name != "x":
                return None
    except Exception:
        return None

    mapped_expr = sp.expand(expr)
    lookup = {
        str(sym): (0 if str(sym) == "x" else int(str(sym)[1:]))
        for sym in free
    }
    builder = _GraphBuilder(sp.Symbol("x"), feature_lookup=lookup)
    root = builder.build(mapped_expr)
    if root is None:
        return None
    return builder.to_graph_dict(root)


def _normalize_formula_text(formula: str) -> str:
    text = str(formula).strip()
    text = text.replace("^", "**")
    text = re.sub(r"\|([^|]+)\|", r"abs(\1)", text)
    return text


def _parse_formula_expr(formula: str) -> Optional[sp.Expr]:
    text = _normalize_formula_text(formula)
    if not text:
        return None
    local_dict = dict(_LOCAL_DICT)
    for name in set(re.findall(r"\bx\d*\b", text)):
        if name == "x" or re.fullmatch(r"x\d+", name):
            local_dict[name] = sp.Symbol(name)
    try:
        expr = parse_expr(
            text,
            local_dict=local_dict,
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

    def __init__(self, x_sym: sp.Symbol, feature_lookup: Optional[Dict[str, int]] = None) -> None:
        self.x_sym = x_sym
        self.feature_lookup = dict(feature_lookup or {})
        self.nodes: List[Dict[str, Any]] = []
        self.output_weights: List[float] = []
        self.output_bias = 0.0
        self.input_nodes: Dict[int, int] = {}

    def _append(self, node: Dict[str, Any]) -> int:
        idx = len(self.nodes)
        self.nodes.append(node)
        self.output_weights.append(0.0)
        return idx

    def _input_node(self, feature_idx: int = 0) -> int:
        feature_idx = int(feature_idx)
        if feature_idx in self.input_nodes:
            return self.input_nodes[feature_idx]
        idx = self._append(_default_node(type=TYPE_INPUT, feature_idx=feature_idx))
        self.input_nodes[feature_idx] = idx
        return idx

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

    def _feature_index_for_symbol(self, expr: sp.Expr) -> Optional[int]:
        if not isinstance(expr, sp.Symbol):
            return None
        name = str(expr)
        if name in self.feature_lookup:
            return int(self.feature_lookup[name])
        if name == str(self.x_sym) or name == "x":
            return 0
        if re.fullmatch(r"x\d+", name):
            return int(name[1:])
        return None

    def _linear_in_any_feature(self, expr: sp.Expr) -> Optional[Tuple[int, float, float]]:
        """Return (feature_idx, omega, phi) for expr ~= omega*x_i + phi."""
        feature_idx = self._feature_index_for_symbol(expr)
        if feature_idx is not None:
            return feature_idx, 1.0, 0.0
        if expr.is_Number:
            return -1, 0.0, float(expr)
        if isinstance(expr, sp.Mul):
            omega = 1.0
            feature_idx = None
            for factor in expr.args:
                factor_feature_idx = self._feature_index_for_symbol(factor)
                if factor_feature_idx is not None:
                    if feature_idx is not None and feature_idx != factor_feature_idx:
                        return None
                    feature_idx = factor_feature_idx
                elif factor.is_Number:
                    omega *= float(factor)
                else:
                    return None
            if feature_idx is not None:
                return feature_idx, omega, 0.0
            return None
        if isinstance(expr, sp.Add):
            feature_idx = None
            omega = 0.0
            phi = 0.0
            for term in expr.args:
                parsed = self._linear_in_any_feature(term)
                if parsed is None:
                    return None
                term_feature_idx, o, p = parsed
                if term_feature_idx >= 0:
                    if feature_idx is not None and feature_idx != term_feature_idx:
                        return None
                    feature_idx = term_feature_idx
                omega += o
                phi += p
            if feature_idx is None:
                return -1, omega, phi
            return feature_idx, omega, phi
        return None

    def build(self, expr: sp.Expr) -> Optional[int]:
        expr = sp.expand(expr)

        if expr.is_Number or isinstance(expr, (sp.Integer, sp.Float, sp.Rational)):
            idx = self._append(
                _default_node(type=TYPE_CONSTANT, value=float(expr))
            )
            return idx

        if expr == self.x_sym:
            return self._input_node(0)

        if isinstance(expr, sp.Symbol):
            name = str(expr)
            if name in self.feature_lookup:
                return self._input_node(self.feature_lookup[name])
            if str(expr) == str(self.x_sym):
                return self._input_node(0)
            if re.fullmatch(r"x\d+", name):
                return self._input_node(int(name[1:]))
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
            lin = self._linear_in_any_feature(sp.expand(arg))
            if lin is None:
                return None
            feature_idx, omega, phi = lin
            if feature_idx < 0:
                return self._append(_default_node(type=TYPE_CONSTANT, value=math.sin(float(phi))))
            return self._append(
                _default_node(
                    type=TYPE_UNARY,
                    unary_op=UNARY_PERIODIC,
                    omega=float(omega) if abs(omega) > 1e-15 else 1.0,
                    phi=float(phi),
                    amplitude=1.0,
                    left_child=self._input_node(feature_idx),
                )
            )

        if getattr(expr, "func", None) == sp.cos:
            (arg,) = expr.args
            lin = self._linear_in_any_feature(sp.expand(arg))
            if lin is None:
                return None
            feature_idx, omega, phi = lin
            if feature_idx < 0:
                return self._append(_default_node(type=TYPE_CONSTANT, value=math.cos(float(phi))))
            return self._append(
                _default_node(
                    type=TYPE_UNARY,
                    unary_op=UNARY_PERIODIC,
                    omega=float(omega) if abs(omega) > 1e-15 else 1.0,
                    phi=float(phi) + math.pi / 2.0,
                    amplitude=1.0,
                    left_child=self._input_node(feature_idx),
                )
            )

        if getattr(expr, "func", None) == sp.exp:
            (arg,) = expr.args
            lin = self._linear_in_any_feature(sp.expand(arg))
            if lin is not None:
                feature_idx, omega, phi = lin
                if feature_idx < 0:
                    return self._append(_default_node(type=TYPE_CONSTANT, value=math.exp(float(phi))))
                return self._append(
                    _default_node(
                        type=TYPE_UNARY,
                        unary_op=UNARY_EXP,
                        omega=float(omega) if abs(omega) > 1e-15 else 1.0,
                        phi=float(phi),
                        left_child=self._input_node(feature_idx),
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
        return _multi_feature_formula_to_seed_graph(formula)
    x_sym = next(iter(free))
    if x_sym.name != x_name:
        return _multi_feature_formula_to_seed_graph(formula)

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

    # Separate composed candidates and standard candidates
    composed_cands = []
    standard_cands = []
    for c in candidate_formulas:
        if c.get("from_specialist_composition") or c.get("source") == "specialist_composition":
            composed_cands.append(c)
        else:
            standard_cands.append(c)

    # Sort both groups by MSE (lower is better)
    composed_cands.sort(key=lambda c: float(c.get("mse", float("inf")) or float("inf")))
    standard_cands.sort(key=lambda c: float(c.get("mse", float("inf")) or float("inf")))

    # Limit composed seeds so they don't dominate (max 35% of max_seeds, min 1 if any exists)
    max_composed = 0
    if composed_cands:
        max_composed = max(1, int(round(max_seeds * 0.35)))

    selected_composed = composed_cands[:max_composed]
    remaining_budget = max(0, max_seeds - len(selected_composed))
    selected_standard = standard_cands[:remaining_budget]

    # Re-combine and sort by MSE
    combined = selected_composed + selected_standard
    combined.sort(key=lambda c: float(c.get("mse", float("inf")) or float("inf")))

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

    for cand in combined:
        _add(cand.get("formula", ""))

    return build_seed_graphs_from_formulas(ordered, max_seeds=max_seeds)


def _safe_array(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        return arr
    return arr[np.isfinite(arr)]


def _estimate_polynomial_signature(
    x_values: np.ndarray,
    y_values: np.ndarray,
    max_degree: int = 6,
) -> Tuple[bool, int, float]:
    """Detect whether the signal is well-approximated by a low-degree polynomial."""
    if x_values.size < max_degree + 2 or y_values.size < max_degree + 2:
        return False, 0, float("inf")

    try:
        x_span = float(np.max(x_values) - np.min(x_values))
        x_scaled = (x_values - float(np.mean(x_values))) / max(0.5 * x_span, 1e-8)
        y_var = max(float(np.var(y_values)), 1e-12)

        best_degree = 1
        best_mse = float("inf")
        for degree in range(1, max_degree + 1):
            coeffs = np.polyfit(x_scaled, y_values, degree)
            pred = np.polyval(coeffs, x_scaled)
            mse = float(np.mean((pred - y_values) ** 2))
            if np.isfinite(mse) and mse < best_mse:
                best_mse = mse
                best_degree = degree

        rel_mse = best_mse / y_var
        return best_degree >= 2 and rel_mse < 1e-11, best_degree, rel_mse
    except Exception:
        return False, 0, float("inf")


def _evaluate_formula_signal(formula: str, x_values: np.ndarray) -> Optional[np.ndarray]:
    expr = _parse_formula_expr(formula)
    if expr is None:
        return None

    free = list(expr.free_symbols)
    if len(free) > 1:
        return None

    try:
        if not free:
            value = float(expr.evalf())
            return np.full_like(x_values, value, dtype=float)

        fn = sp.lambdify(free[0], expr, modules=["numpy"])
        values = np.asarray(fn(x_values), dtype=float).ravel()
        if values.shape != x_values.shape:
            values = np.broadcast_to(values, x_values.shape).astype(float, copy=False)
        return values
    except Exception:
        return None


def _affine_fit_mse(pred: np.ndarray, target: np.ndarray) -> float:
    mask = np.isfinite(pred) & np.isfinite(target)
    if int(mask.sum()) < 8:
        return float("inf")

    x = pred[mask].astype(float, copy=False)
    y = target[mask].astype(float, copy=False)
    if np.allclose(x, x[0]):
        fitted = np.full_like(y, float(np.mean(y)))
        return float(np.mean((fitted - y) ** 2))

    design = np.column_stack([x, np.ones_like(x)])
    try:
        coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        fitted = design @ coef
        return float(np.mean((fitted - y) ** 2))
    except Exception:
        return float("inf")


def discover_seed_formulas_from_signal(
    x_values: Any,
    y_values: Any,
    detected_omegas: Optional[List[float]] = None,
    max_seeds: int = 12,
) -> List[str]:
    """Discover universal module/product seed formulas from the observed signal."""
    x = _safe_array(x_values)
    y = _safe_array(y_values)
    if x.size == 0 or y.size == 0:
        return []

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 8 or y.size < 8:
        return []

    poly_like, degree, _ = _estimate_polynomial_signature(x, y)

    def _dedupe(items: List[str]) -> List[str]:
        ordered: List[str] = []
        seen: set = set()
        for item in items:
            text = str(item or "").strip()
            if not text:
                continue
            key = re.sub(r"\s+", "", text.lower())
            if key in seen:
                continue
            seen.add(key)
            ordered.append(text)
        return ordered

    power_terms = ["x", "x^2", "x^3"]
    rational_terms = [
        "1/(x^2+1)",
        "x/(x^2+1)",
        "x^2/(x^2+1)",
        "x/(x^4+1)",
        "x^2/(x^4+1)",
        "x^3/(x^4+1)",
    ]
    decay_terms = ["exp(-x)", "exp(-x^2)"]
    periodic_terms = ["sin(x)", "cos(x)"]

    if poly_like:
        for d in range(2, min(7, degree + 2)):
            power_terms.append(f"x^{d}")
            periodic_terms.append(f"sin(x^{d})")
            periodic_terms.append(f"cos(x^{d})")
            decay_terms.append(f"exp(-x^{d})")

    if detected_omegas:
        for omega in detected_omegas[:3]:
            omega_text = f"{float(omega):.6g}"
            periodic_terms.append(f"sin({omega_text}*x)")
            periodic_terms.append(f"cos({omega_text}*x)")

    power_terms = _dedupe(power_terms)
    rational_terms = _dedupe(rational_terms)
    decay_terms = _dedupe(decay_terms)
    periodic_terms = _dedupe(periodic_terms)

    candidate_formulas: List[str] = []

    def _add(formula: str) -> None:
        text = str(formula or "").strip()
        if not text:
            return
        candidate_formulas.append(text)

    for formula in power_terms + rational_terms + decay_terms + periodic_terms:
        _add(formula)

    for power in power_terms:
        for periodic in periodic_terms:
            _add(f"{power}*{periodic}")
        for decay in decay_terms:
            _add(f"{power}*{decay}")
        for periodic in periodic_terms:
            for decay in decay_terms:
                _add(f"{power}*{decay}*{periodic}")

    for periodic in periodic_terms:
        for decay in decay_terms:
            _add(f"{decay}*{periodic}")

    scored: List[Tuple[float, int, str]] = []
    y_var = max(float(np.var(y)), 1e-12)
    for formula in _dedupe(candidate_formulas):
        pred = _evaluate_formula_signal(formula, x)
        if pred is None:
            continue
        mse = _affine_fit_mse(pred, y)
        if not np.isfinite(mse):
            continue
        rel_mse = mse / y_var
        complexity = len(formula)
        # Prefer formulas that match the observed structure, but keep the pool broad.
        bonus = 0
        if "*" in formula:
            bonus -= 2
        if "^" in formula or "**" in formula:
            bonus -= 1
        scored.append((rel_mse, complexity + bonus, formula))

    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    ordered: List[str] = []
    seen: set = set()
    for _, _, formula in scored:
        key = re.sub(r"\s+", "", formula.lower())
        if key in seen:
            continue
        seen.add(key)
        ordered.append(formula)
        if len(ordered) >= max_seeds:
            break
    return ordered


def build_seed_graphs_from_signal(
    x_values: Any,
    y_values: Any,
    detected_omegas: Optional[List[float]] = None,
    max_seeds: int = 12,
) -> List[Dict[str, Any]]:
    """Build universal separability-aware seed graphs from the observed signal."""
    formulas = discover_seed_formulas_from_signal(
        x_values,
        y_values,
        detected_omegas=detected_omegas,
        max_seeds=max_seeds,
    )
    return build_seed_graphs_from_formulas(formulas, max_seeds=max_seeds)
