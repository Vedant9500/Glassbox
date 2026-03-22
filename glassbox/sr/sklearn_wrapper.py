"""
Scikit-learn compatible wrapper for Glassbox Symbolic Regression.

Uses the FULL Glassbox pipeline:
  1. Classifier fast-path (instant for well-characterized curves)
  2. C++ guided evolution (beam search over multiple configs)
  3. Multipass formula simplification (float snapping + SymPy simplification)
"""

import sys
import re
import math
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

# Path setup
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPTS_DIR = _REPO_ROOT / 'scripts'
_CPP_DIR = Path(__file__).resolve().parent / 'cpp'

for p in [str(_REPO_ROOT), str(_SCRIPTS_DIR), str(_CPP_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import _core
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False

from .evolution import detect_dominant_frequency


class GlassboxRegressor(BaseEstimator, RegressorMixin):
    """
    Scikit-learn compatible wrapper for Glassbox Symbolic Regression.

    Uses the full pipeline: classifier fast-path → C++ evolution → formula simplification.
    """

    def __init__(
        self,
        population_size=100,
        generations=1000,
        early_stop_mse=1e-6,
        random_state=None,
        p_min=-2.0,
        p_max=3.0,
        use_nsga2=False,
        num_islands=1,
        migration_interval=25,
        migration_size=2,
        arithmetic_temperature=5.0,
        # Pipeline control
        use_fast_path=True,
        use_guided_evolution=True,
        use_simplification=True,
        classifier_path="models/curve_classifier_v3.1.pt",
        simplification_int_tol=0.01,
        simplification_zero_tol=1e-3,
        max_power=6,
        timeout=120,
        evolution_skip_r2=0.999,
        device=None,
    ):
        self.population_size = population_size
        self.generations = generations
        self.early_stop_mse = early_stop_mse
        self.random_state = random_state
        self.p_min = p_min
        self.p_max = p_max
        self.use_nsga2 = use_nsga2
        self.num_islands = num_islands
        self.migration_interval = migration_interval
        self.migration_size = migration_size
        self.arithmetic_temperature = arithmetic_temperature
        self.use_fast_path = use_fast_path
        self.use_guided_evolution = use_guided_evolution
        self.use_simplification = use_simplification
        self.classifier_path = classifier_path
        self.simplification_int_tol = simplification_int_tol
        self.simplification_zero_tol = simplification_zero_tol
        self.max_power = max_power
        self.timeout = timeout
        self.evolution_skip_r2 = evolution_skip_r2
        self.device = device

    def _resolve_classifier_path(self):
        """Resolve classifier model path relative to repo root."""
        p = Path(self.classifier_path)
        if p.is_absolute() and p.exists():
            return str(p)
        repo_path = _REPO_ROOT / self.classifier_path
        if repo_path.exists():
            return str(repo_path)
        return str(p)

    def _simplify_formula(self, formula):
        """Apply multipass formula simplification."""
        if not formula or not self.use_simplification:
            return formula
        try:
            from simplify_formula import simplify_onn_formula
            _, simplified_expr = simplify_onn_formula(
                formula,
                int_tol=self.simplification_int_tol,
                zero_tol=self.simplification_zero_tol,
                use_nsimplify=True,
                max_passes=6,
                use_identities=True,
            )
            return str(simplified_expr)
        except Exception:
            return formula

    def _detect_frequencies(self, X, y):
        """Detect dominant frequencies via FFT."""
        try:
            x_t = torch.tensor(X[:, 0], dtype=torch.float32).reshape(-1, 1)
            y_t = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
            omegas = detect_dominant_frequency(x_t, y_t, n_frequencies=3)
            if omegas and omegas[0] == 1.0:
                return []
            return omegas or []
        except Exception:
            return []

    def fit(self, X, y):
        """
        Fit the symbolic regression model using the full Glassbox pipeline:
        1. Fast-path (classifier-guided basis regression)
        2. C++ evolution (if fast-path misses or is approximate)
        3. Formula simplification (float snapping + SymPy)
        """
        import time as _time

        X, y = check_X_y(X, y, accept_sparse=False)
        self.n_features_in_ = X.shape[1]
        fit_start = _time.time()

        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)

        detected_omegas = self._detect_frequencies(X, y)

        best_formula = None
        best_mse = float('inf')
        operator_hints = {}
        y_var = float(np.var(y))  # For R² calculation

        def _elapsed():
            return _time.time() - fit_start

        def _r2_from_mse(mse):
            """Compute R² from MSE and target variance."""
            if y_var < 1e-15:
                return 1.0 if mse < 1e-15 else 0.0
            return 1.0 - mse / y_var

        # ── Stage 1: Classifier Fast Path ──
        if self.use_fast_path and _elapsed() < self.timeout:
            try:
                from classifier_fast_path import run_fast_path

                x_t = torch.tensor(X, dtype=torch.float32)
                y_t = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
                classifier_path = self._resolve_classifier_path()

                fp_result = run_fast_path(
                    x_t, y_t,
                    classifier_path=classifier_path,
                    detected_omegas=detected_omegas,
                    op_constraints=None,
                    auto_expand=True,
                    device=self.device,
                    exact_match_threads=1,
                    exact_match_enabled=True,
                    exact_match_max_basis=200,
                    max_power=self.max_power,
                    simplify_formula_output=False,
                )

                if fp_result and fp_result.get('formula'):
                    best_formula = fp_result['formula']
                    best_mse = fp_result.get('mse', float('inf'))
                    operator_hints = fp_result.get('operator_hints', {})
            except Exception as e:
                print(f"  [Fast-path skipped: {e}]")

        # ── Stage 2: C++ Evolution ──
        # Only run evolution if:
        #   - No formula found yet, OR
        #   - R² is below the skip threshold (default 0.999)
        #   - We haven't exceeded the timeout
        current_r2 = _r2_from_mse(best_mse) if best_formula else -1.0
        term_count = (best_formula.count('+') + best_formula.count('-')) if best_formula else 0
        need_evolution = (
            best_formula is None or
            best_mse is None or
            not math.isfinite(best_mse) or
            current_r2 < self.evolution_skip_r2 or
            term_count > 8
        )

        if need_evolution and _elapsed() < self.timeout:
            if not CPP_AVAILABLE:
                if best_formula is None:
                    raise ImportError(
                        "Glassbox C++ core (_core.pyd/.so) not found. "
                        "Please build the backend first."
                    )
            else:
                evo_formula = None
                evo_mse = float('inf')

                # Try guided evolution (beam search) only if R² is low
                if (self.use_guided_evolution and operator_hints
                        and current_r2 < self.evolution_skip_r2
                        and _elapsed() < self.timeout):
                    try:
                        from classifier_fast_path import run_guided_evolution

                        x_t = torch.tensor(X, dtype=torch.float32)
                        y_t = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

                        hints = dict(operator_hints)
                        hints['operators'] = set(hints.get('operators', set()))
                        hints['frequencies'] = list(hints.get('frequencies', detected_omegas or []))
                        hints['powers'] = list(hints.get('powers', []))
                        hints['has_rational'] = bool(hints.get('has_rational', False))
                        hints['has_exp_decay'] = bool(hints.get('has_exp_decay', False))
                        hints['active_terms'] = list(hints.get('active_terms', []))

                        guided_result = run_guided_evolution(
                            x_t, y_t, hints,
                            generations=min(40, self.generations // 10),
                            population_size=min(30, self.population_size),
                            device=self.device or "cpu",
                        )

                        if guided_result and guided_result.get('formula'):
                            evo_formula = guided_result['formula']
                            evo_mse = guided_result.get('mse', float('inf'))
                    except Exception as e:
                        print(f"  [Guided evolution skipped: {e}]")

                # Fall back to raw C++ evolution
                if (evo_formula is None or evo_mse >= self.early_stop_mse) and _elapsed() < self.timeout:
                    try:
                        X_list = [X[:, i].astype(np.float64) for i in range(self.n_features_in_)]
                        y_arr = y.astype(np.float64).flatten()

                        result = _core.run_evolution(
                            X_list=X_list,
                            y=y_arr,
                            pop_size=self.population_size,
                            generations=self.generations,
                            early_stop_mse=self.early_stop_mse,
                            seed_omegas=detected_omegas,
                            timeout_seconds=max(1, int(self.timeout - _elapsed())),
                            p_min=self.p_min,
                            p_max=self.p_max,
                            use_nsga2=self.use_nsga2,
                            num_islands=self.num_islands,
                            migration_interval=self.migration_interval,
                            migration_size=self.migration_size,
                            arithmetic_temperature=self.arithmetic_temperature,
                        )

                        raw_mse = result.get('best_mse', float('inf'))
                        raw_formula = result.get('formula', '')

                        if raw_mse < evo_mse:
                            evo_formula = raw_formula
                            evo_mse = raw_mse

                        # Store raw C++ results for inspection
                        self.nodes_ = result.get('nodes', [])
                        self.output_weights_ = result.get('output_weights', [])
                        self.output_bias_ = result.get('output_bias', 0.0)
                        if 'pareto_front' in result:
                            self.pareto_front_ = result['pareto_front']
                    except Exception as e:
                        print(f"  [C++ evolution error: {e}]")

                # Take evolution result if better than fast-path
                if evo_formula and (evo_mse < best_mse or best_formula is None):
                    best_formula = evo_formula
                    best_mse = evo_mse
        elif need_evolution and _elapsed() >= self.timeout:
            print(f"  [Timeout: skipping evolution after {_elapsed():.1f}s]")

        # ── Stage 3: Formula Simplification & Noise Reduction ──
        if best_formula:
            best_formula = self._reduce_formula_noise(best_formula, X, y)
            best_formula = self._simplify_formula(best_formula)

        self.formula_ = best_formula or "0"
        self.best_mse_ = best_mse
        return self

    def predict(self, X):
        """
        Predict using the discovered symbolic formula.
        Handles edge cases (log of zero, sqrt of negative) gracefully.
        """
        check_is_fitted(self)
        X = check_array(X)

        def _safe_log(x):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return np.where(
                    np.abs(x) > 1e-300,
                    np.log(np.abs(x) + 1e-300),
                    -300.0,
                )

        def _safe_sqrt(x):
            return np.sqrt(np.maximum(x, 0.0))

        context = {
            "np": np,
            "log": _safe_log,
            "sin": np.sin,
            "cos": np.cos,
            "exp": lambda x: np.exp(np.clip(x, -500, 500)),
            "sqrt": _safe_sqrt,
            "abs": np.abs,
            "Abs": np.abs,  # sympy uses Abs
            "pi": np.pi,
            "E": np.e,
        }
        for i in range(self.n_features_in_):
            context[f"x{i}"] = X[:, i]
        # C++ engine uses bare 'x' for single-feature problems
        if self.n_features_in_ == 1:
            context["x"] = X[:, 0]

        # Parse formula string into python-evaluable form
        formula = self.formula_.strip()
        # |x| -> abs(x)
        formula = re.sub(r'\|([^|]+)\|', r'abs(\1)', formula)
        # x^N -> x**N
        formula = re.sub(r'\^', r'**', formula)
        # Route math functions through our safe wrappers
        formula = formula.replace('np.', '')

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y_pred = eval(formula, {"__builtins__": None}, context)
            if isinstance(y_pred, (int, float)):
                y_pred = np.full(X.shape[0], y_pred)
            else:
                y_pred = np.asarray(y_pred, dtype=np.float64)
            y_pred = np.where(np.isfinite(y_pred), y_pred, 0.0)
            return y_pred
        except Exception as e:
            print(f"Prediction error: {e}")
            return np.zeros(X.shape[0])

    def get_formula(self):
        """Returns the discovered formula string."""
        check_is_fitted(self)
        return self.formula_

    def _reduce_formula_noise(self, formula_str, X, y):
        """Greedy backward elimination of terms to reduce noise from L1 regularization."""
        if not formula_str or formula_str == "0":
            return formula_str
            
        try:
            import sympy as sp
            from sympy.parsing.sympy_parser import parse_expr
            from sklearn.linear_model import LinearRegression
            
            expr = parse_expr(formula_str.replace('^', '**'))
            terms = list(sp.Add.make_args(expr))
            
            if len(terms) <= 1 or len(terms) > 20:
                return formula_str

            term_funcs = []
            x_syms = [sp.Symbol(f"x{i}") for i in range(self.n_features_in_)]
            syms = x_syms + [sp.Symbol('x')]
            
            for t in terms:
                fn = sp.lambdify(syms, t, modules=['numpy'])
                term_funcs.append(fn)
                
            N = len(y)
            Z = np.zeros((N, len(terms)))
            
            for i, fn in enumerate(term_funcs):
                args = [X[:, j] for j in range(self.n_features_in_)]
                args.append(X[:, 0]) # for 'x' fallback
                val = fn(*args)
                if isinstance(val, (int, float)):
                    Z[:, i] = np.full(N, val)
                else:
                    Z[:, i] = val
                    
            def get_bic(mask):
                if not np.any(mask):
                    return float('inf'), None
                Z_sub = Z[:, mask]
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    model = LinearRegression(fit_intercept=False).fit(Z_sub, y)
                    preds = model.predict(Z_sub)
                mse = np.mean((y - preds)**2)
                if mse < 1e-15:
                    mse = 1e-15
                k = np.sum(mask)
                return N * np.log(mse) + k * np.log(N), model.coef_

            current_mask = np.ones(len(terms), dtype=bool)
            best_bic, best_coef = get_bic(current_mask)
            
            while np.sum(current_mask) > 1:
                best_drop_idx = -1
                best_drop_bic = best_bic
                best_drop_coef = best_coef
                
                for i in range(len(terms)):
                    if current_mask[i]:
                        test_mask = current_mask.copy()
                        test_mask[i] = False
                        bic, coef = get_bic(test_mask)
                        
                        if bic < best_drop_bic:
                            best_drop_bic = bic
                            best_drop_idx = i
                            best_drop_coef = coef
                            
                if best_drop_idx != -1:
                    current_mask[best_drop_idx] = False
                    best_bic = best_drop_bic
                    best_coef = best_drop_coef
                else:
                    break
                    
            final_terms = []
            coef_idx = 0
            for i, t in enumerate(terms):
                if current_mask[i]:
                    c = best_coef[coef_idx]
                    if abs(c) > 1e-8:
                        final_terms.append(c * t)
                    coef_idx += 1
                    
            if not final_terms:
                return "0"
            return str(sum(final_terms))
            
        except Exception as e:
            print(f"  [Noise Reduction Skipped: {e}]")
            return formula_str
