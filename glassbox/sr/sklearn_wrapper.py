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
try:
    from scipy.optimize import least_squares
except Exception:  # pragma: no cover - scipy is declared but keep import optional
    least_squares = None
from glassbox.sr.blackbox_preprocessor import (
    formula_from_search_to_original_space,
    discover_blackbox_interactions,
    prepare_blackbox_search,
    remap_original_formula_to_reduced,
    state_to_dict,
)
from glassbox.sr.specialist_state import compute_specialist_state
from glassbox.sr.specialist_state import propose_specialist_compositions


def _clamp_int(value, default, lo, hi):
    try:
        value = int(round(float(value)))
    except Exception:
        value = default
    return int(max(lo, min(hi, value)))


def _clamp_float(value, default, lo, hi):
    try:
        value = float(value)
    except Exception:
        value = default
    return float(max(lo, min(hi, value)))


def _finite_float(value, default=0.0):
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


# Path setup
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPTS_DIR = _REPO_ROOT / 'scripts'
_CPP_DIR = Path(__file__).resolve().parent / 'cpp'

for p in [str(_REPO_ROOT), str(_SCRIPTS_DIR), str(_CPP_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import _core  # type: ignore
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False

from glassbox.evolution import detect_dominant_frequency


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
        num_islands=8,
        migration_interval=25,
        migration_size=2,
        arithmetic_temperature=5.0,
        # Pipeline control
        use_fast_path=True,
        use_guided_evolution=True,
        use_simplification=True,
        classifier_path="models/curve_classifier_multi.pt",
        simplification_int_tol=0.05,
        simplification_zero_tol=1e-3,
        max_power=6,
        timeout=120,
        evolution_skip_r2=0.999,
        multi_start_runs=3,
        adaptive_compute_budget=True,
        min_compute_budget=10,
        max_compute_budget=300,
        cv_skip_guard_enabled=True,
        cv_skip_guard_folds=3,
        cv_skip_guard_min_fold_r2=0.98,
        cv_skip_guard_max_r2_std=0.03,
        cv_skip_guard_min_samples=45,
        use_universal_proposer="auto",
        universal_proposer_path="models/universal_proposer_multi.pt",
        universal_proposer_shadow_mode="auto",
        universal_proposer_log_routing=True,
        universal_proposer_top_k=5,
        blackbox_mode="auto",
        blackbox_max_features=6,
        blackbox_feature_selection=True,
        blackbox_standardize=True,
        blackbox_min_features_to_select=5,
        enable_specialist_screening_diagnostics=True,
        enable_specialist_composition_screening=True,
        enable_residual_stage=True,
        device=None,
        skip_evolution_if_bloated=False,
        bloat_term_threshold=20,
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
        self.multi_start_runs = multi_start_runs
        self.adaptive_compute_budget = adaptive_compute_budget
        self.min_compute_budget = min_compute_budget
        self.max_compute_budget = max_compute_budget
        self.cv_skip_guard_enabled = cv_skip_guard_enabled
        self.cv_skip_guard_folds = cv_skip_guard_folds
        self.cv_skip_guard_min_fold_r2 = cv_skip_guard_min_fold_r2
        self.cv_skip_guard_max_r2_std = cv_skip_guard_max_r2_std
        import os
        self.cv_skip_guard_min_samples = cv_skip_guard_min_samples
        
        # Rollback switch via environment variable
        legacy_mode = os.environ.get("GLASSBOX_USE_LEGACY_FASTPATH", "0") != "0"
        
        self.use_universal_proposer = not legacy_mode if use_universal_proposer == "auto" else use_universal_proposer
        self.universal_proposer_path = universal_proposer_path
        self.universal_proposer_shadow_mode = legacy_mode if universal_proposer_shadow_mode == "auto" else universal_proposer_shadow_mode
        self.universal_proposer_log_routing = universal_proposer_log_routing
        self.universal_proposer_top_k = universal_proposer_top_k
        self.blackbox_mode = blackbox_mode
        self.blackbox_max_features = blackbox_max_features
        self.blackbox_feature_selection = blackbox_feature_selection
        self.blackbox_standardize = blackbox_standardize
        self.blackbox_min_features_to_select = blackbox_min_features_to_select
        self.enable_specialist_screening_diagnostics = enable_specialist_screening_diagnostics
        self.enable_specialist_composition_screening = enable_specialist_composition_screening
        self.enable_residual_stage = enable_residual_stage
        self.device = device
        self.skip_evolution_if_bloated = skip_evolution_if_bloated
        self.bloat_term_threshold = bloat_term_threshold

        self._universal_proposer_model = None
        self.specialist_state_ = None
        self.specialist_track_ = "incumbent path"
        self.has_composed_seeds_ = False

    def _estimate_compute_budget(self, X, current_r2, term_count, uncertainty=None):
        """Adaptive compute budget: easy problems get short runs, hard problems get longer runs.

        When *uncertainty* (from the fast-path FPIP) is supplied the budget
        is further scaled:
        - Low entropy + high margin → the classifier is confident, reduce budget.
        - High entropy + low margin → uncertain, give evolution more time.
        - Exact fast-path hit with low uncertainty → minimal budget.
        """
        base_timeout = float(max(1, self.timeout))
        if not self.adaptive_compute_budget:
            return base_timeout

        n_samples = int(X.shape[0])
        n_features = int(X.shape[1])

        score = 1.0
        score += 0.15 * max(0, n_features - 1)
        score += 0.08 * min(1.0, np.log10(max(50, n_samples)) / 3.0)

        # Fast-path confidence gates: reduce budget on easy problems.
        if current_r2 >= 0.995 and term_count <= 5:
            score *= 0.2
        elif current_r2 >= 0.98 and term_count <= 8:
            score *= 0.5
        elif current_r2 >= 0.90:
            score *= 0.9
        else:
            score *= 2.5

        # ── Uncertainty-coupled budget routing ──
        # If classifier uncertainty metrics are available, scale budget:
        # certain classifier + strong R² → avoid expensive guided escalation.
        if isinstance(uncertainty, dict):
            entropy = uncertainty.get('prediction_entropy')
            margin = uncertainty.get('prediction_margin')
            uncertain_flag = bool(uncertainty.get('prediction_uncertain', False))

            if not uncertain_flag and entropy is not None and margin is not None:
                try:
                    ent = float(entropy)
                    mar = float(margin)
                    if np.isfinite(ent) and np.isfinite(mar):
                        # High confidence (low entropy, high margin) → shrink budget
                        confidence = float(np.clip((1.0 - ent) * min(mar / 0.25, 1.0), 0.0, 1.0))
                        
                        # Map confidence ∈ [0,1] to multiplier ∈ [0.1, 1.0] (more aggressive than 0.3)
                        uncertainty_scale = 1.0 - 0.9 * confidence
                        score *= uncertainty_scale
                except (TypeError, ValueError):
                    pass
            elif uncertain_flag:
                # Uncertain → give more time, but cap the escalation
                score *= 1.2

        # ── Proposer-specific budget scaling ──
        # If we have skeletons, we expect faster convergence.
        if getattr(self, 'universal_proposer_fpip_v2_', None):
            payload = self.universal_proposer_fpip_v2_
            if payload.get('valid') and payload.get('candidate_skeletons'):
                # We have seeds! Reduce base budget because we aren't starting from scratch.
                score *= 0.7

        budget = base_timeout * score
        return float(np.clip(budget, float(self.min_compute_budget), float(self.max_compute_budget)))

    def _split_blackbox_holdout(self, X, y, validation_fraction=0.2):
        """Build a deterministic shuffled train/validation split for candidate screening."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        n = int(X.shape[0])
        if n < 24:
            return None
        seed = 0 if self.random_state is None else int(self.random_state)
        rng = np.random.RandomState(seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        holdout_n = int(max(4, round(n * float(validation_fraction))))
        holdout_n = min(holdout_n, n - 12)
        if holdout_n <= 0:
            return None
        fit_idx = idx[:-holdout_n]
        val_idx = idx[-holdout_n:]
        if fit_idx.size < 12 or val_idx.size < 4:
            return None
        return {
            "fit_idx": fit_idx,
            "val_idx": val_idx,
            "X_fit": X[fit_idx],
            "y_fit": y[fit_idx],
            "X_val": X[val_idx],
            "y_val": y[val_idx],
        }

    def _formula_complexity(self, formula):
        text = str(formula or "").strip()
        if not text:
            return 0
        ops = sum(text.count(ch) for ch in "+-*/^")
        funcs = sum(text.count(name) for name in ("sin", "cos", "exp", "log", "sqrt", "abs"))
        return int(max(1, ops + funcs + 1))

    def _formula_risk_score(self, formula, X=None):
        """Penalize structures that often fit train data but fail blackbox holdout."""
        text = str(formula or "").strip()
        if not text:
            return 1.0
        lower = text.lower()
        risk = 0.0
        decimal_powers = [
            float(match)
            for match in re.findall(r"(?:\^|\*\*)\s*(-?\d+\.\d+)", lower)
        ]
        risk += 0.16 * len(decimal_powers)
        if "/" in lower:
            risk += 0.08 * lower.count("/")
        if "exp(" in lower:
            risk += 0.06 * lower.count("exp(")
        if "sqrt(" in lower and "abs(" not in lower:
            risk += 0.12
        if "log(" in lower and "abs(" not in lower:
            risk += 0.10
        risk += 0.012 * max(0, self._formula_complexity(text) - 12)

        if X is not None and "/" in lower:
            # Probe denominator fragility by turning a/b into b where possible.
            for denom in re.findall(r"/\s*(\([^()]+\)|[a-zA-Z0-9_.*+\-]+)", lower):
                denom_text = denom.strip()
                if denom_text.startswith("(") and denom_text.endswith(")"):
                    denom_text = denom_text[1:-1]
                try:
                    values = self._safe_eval_formula_array(denom_text, X)
                    near_zero = float(np.mean(np.abs(values) < 1e-4))
                    risk += min(0.25, 0.5 * near_zero)
                except Exception:
                    risk += 0.05
        return float(np.clip(risk, 0.0, 1.0))

    def _domain_edge_validation_split(self, X, y, validation_fraction=0.2):
        """Hold out boundary and random points to catch fragile blackbox formulas."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        n = int(X.shape[0])
        if n < 24:
            return self._split_blackbox_holdout(X, y, validation_fraction=validation_fraction)

        holdout_n = int(max(4, round(n * float(validation_fraction))))
        holdout_n = min(holdout_n, n - 12)
        if holdout_n <= 0:
            return None

        finite_X = np.where(np.isfinite(X), X, 0.0)
        z = np.zeros_like(finite_X, dtype=np.float64)
        for j in range(finite_X.shape[1]):
            col = finite_X[:, j]
            med = float(np.median(col))
            scale = float(np.percentile(np.abs(col - med), 75))
            if not np.isfinite(scale) or scale < 1e-12:
                scale = float(np.std(col))
            if not np.isfinite(scale) or scale < 1e-12:
                scale = 1.0
            z[:, j] = np.abs((col - med) / scale)
        edge_score = np.max(z, axis=1) if z.size else np.zeros(n)
        edge_n = min(max(2, holdout_n // 2), holdout_n)
        edge_idx = list(np.argsort(edge_score)[-edge_n:])

        seed = 0 if self.random_state is None else int(self.random_state)
        rng = np.random.RandomState(seed + 7919)
        remaining = [idx for idx in range(n) if idx not in set(edge_idx)]
        rng.shuffle(remaining)
        val_idx = np.asarray(edge_idx + remaining[: max(0, holdout_n - len(edge_idx))], dtype=int)
        fit_idx = np.asarray([idx for idx in range(n) if idx not in set(val_idx.tolist())], dtype=int)
        if fit_idx.size < 12 or val_idx.size < 4:
            return self._split_blackbox_holdout(X, y, validation_fraction=validation_fraction)
        return {
            "fit_idx": fit_idx,
            "val_idx": val_idx,
            "X_fit": X[fit_idx],
            "y_fit": y[fit_idx],
            "X_val": X[val_idx],
            "y_val": y[val_idx],
        }

    def _random_blackbox_validation_split(self, X, y, validation_fraction=0.25, *, salt=0):
        """Random interpolation holdout for Track 1 tabular blackbox selection."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        n = int(X.shape[0])
        if n < 24:
            return self._split_blackbox_holdout(X, y, validation_fraction=validation_fraction)

        holdout_n = int(max(4, round(n * float(validation_fraction))))
        holdout_n = min(holdout_n, n - 12)
        if holdout_n <= 0:
            return None

        seed = 0 if self.random_state is None else int(self.random_state)
        rng = np.random.RandomState(seed + 104729 + int(salt))
        order = np.arange(n, dtype=int)
        rng.shuffle(order)
        val_idx = np.asarray(order[:holdout_n], dtype=int)
        fit_idx = np.asarray(order[holdout_n:], dtype=int)
        if fit_idx.size < 12 or val_idx.size < 4:
            return self._split_blackbox_holdout(X, y, validation_fraction=validation_fraction)
        return {
            "fit_idx": fit_idx,
            "val_idx": val_idx,
            "X_fit": X[fit_idx],
            "y_fit": y[fit_idx],
            "X_val": X[val_idx],
            "y_val": y[val_idx],
        }

    def _ridge_tail_validation_r2(self, X, y, columns=None, validation_fraction=0.25):
        """Small ordered-holdout ridge probe used to audit feature reduction."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if columns is not None:
            X = X[:, list(columns)]
        n = int(X.shape[0])
        holdout_n = int(max(8, round(n * float(validation_fraction))))
        holdout_n = min(holdout_n, n - 16)
        if holdout_n <= 0 or X.shape[1] == 0:
            return None
        X_fit = X[:-holdout_n]
        y_fit = y[:-holdout_n]
        X_val = X[-holdout_n:]
        y_val = y[-holdout_n:]
        if X_fit.shape[0] < 16 or X_val.shape[0] < 8:
            return None

        mu = np.mean(X_fit, axis=0)
        sigma = np.std(X_fit, axis=0)
        sigma = np.where(sigma < 1e-10, 1.0, sigma)
        Z_fit = (X_fit - mu) / sigma
        Z_val = (X_val - mu) / sigma
        A_fit = np.column_stack([Z_fit, np.ones(Z_fit.shape[0])])
        A_val = np.column_stack([Z_val, np.ones(Z_val.shape[0])])
        y_var = max(float(np.var(y_val)), 1e-12)
        best = None
        for alpha in np.logspace(-5, 4, 18):
            reg = np.eye(A_fit.shape[1], dtype=np.float64) * float(alpha)
            reg[-1, -1] = 0.0
            try:
                coef = np.linalg.solve(A_fit.T @ A_fit + reg, A_fit.T @ y_fit)
            except Exception:
                continue
            pred = A_val @ coef
            if not np.all(np.isfinite(pred)):
                continue
            mse = float(np.mean((pred - y_val) ** 2))
            if not np.isfinite(mse):
                continue
            r2 = 1.0 - mse / y_var
            if best is None or r2 > best:
                best = float(r2)
        return best

    def _fit_ridge_formula(self, X, y, columns=None, validation_fraction=0.25):
        """Fit a compact linear ridge formula in the original feature space."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        cols = list(range(X.shape[1])) if columns is None else list(columns)
        if not cols:
            return None
        X_sub = X[:, cols]
        n = int(X_sub.shape[0])
        holdout_n = int(max(8, round(n * float(validation_fraction))))
        holdout_n = min(holdout_n, n - 16)
        if holdout_n <= 0:
            return None
        X_fit = X_sub[:-holdout_n]
        y_fit = y[:-holdout_n]
        X_val = X_sub[-holdout_n:]
        y_val = y[-holdout_n:]
        if X_fit.shape[0] < 16:
            return None
        mu = np.mean(X_fit, axis=0)
        sigma = np.std(X_fit, axis=0)
        sigma = np.where(sigma < 1e-10, 1.0, sigma)
        Z_fit = (X_fit - mu) / sigma
        Z_val = (X_val - mu) / sigma
        A_fit = np.column_stack([Z_fit, np.ones(Z_fit.shape[0])])
        A_val = np.column_stack([Z_val, np.ones(Z_val.shape[0])])
        y_var = max(float(np.var(y_val)), 1e-12)
        best = None
        for alpha in np.logspace(-5, 4, 18):
            reg = np.eye(A_fit.shape[1], dtype=np.float64) * float(alpha)
            reg[-1, -1] = 0.0
            try:
                coef = np.linalg.solve(A_fit.T @ A_fit + reg, A_fit.T @ y_fit)
            except Exception:
                continue
            pred = A_val @ coef
            if not np.all(np.isfinite(pred)):
                continue
            val_mse = float(np.mean((pred - y_val) ** 2))
            if not np.isfinite(val_mse):
                continue
            if best is None or val_mse < best["validation_mse"]:
                best = {"coef": coef, "validation_mse": val_mse, "alpha": float(alpha)}
        if best is None:
            return None

        full_mu = np.mean(X_sub, axis=0)
        full_sigma = np.std(X_sub, axis=0)
        full_sigma = np.where(full_sigma < 1e-10, 1.0, full_sigma)
        Z_full = (X_sub - full_mu) / full_sigma
        A_full = np.column_stack([Z_full, np.ones(Z_full.shape[0])])
        reg = np.eye(A_full.shape[1], dtype=np.float64) * float(best["alpha"])
        reg[-1, -1] = 0.0
        try:
            coef_full = np.linalg.solve(A_full.T @ A_full + reg, A_full.T @ y)
        except Exception:
            coef_full = best["coef"]
            full_mu = mu
            full_sigma = sigma

        coef_z = np.asarray(coef_full[:-1], dtype=np.float64)
        intercept_z = float(coef_full[-1])
        weights = coef_z / full_sigma
        bias = intercept_z - float(np.sum(coef_z * full_mu / full_sigma))
        terms = []
        selected_terms = []
        for col, weight in zip(cols, weights):
            if not np.isfinite(weight) or abs(float(weight)) < 1e-10:
                continue
            selected_terms.append(f"x{col}")
            terms.append(f"({float(weight):.12g})*x{col}")
        if abs(bias) > 1e-10 or not terms:
            terms.append(f"({bias:.12g})")
        formula = "+".join(terms) if terms else "0"
        try:
            full_pred = self._safe_eval_formula_array(formula, X)
        except Exception:
            return None
        full_mse = float(np.mean((full_pred - y) ** 2))
        return {
            "formula": formula,
            "mse": full_mse,
            "validation_mse": float(best["validation_mse"]),
            "validation_r2": float(1.0 - best["validation_mse"] / y_var),
            "selected_terms": selected_terms,
            "n_terms": len(selected_terms),
            "complexity": self._formula_complexity(formula),
            "source": "original_linear_ridge",
        }

    def _refine_formula_constants(self, formula, X_fit, y_fit, X_val, y_val, *, max_constants=8):
        """Optimize numeric constants inside a candidate structure with least squares."""
        if least_squares is None:
            return None
        text = str(formula or "").strip()
        if not text:
            return None
        number_pattern = re.compile(r"(?<![A-Za-z_])(?<!\w)([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)")
        matches = [
            m for m in number_pattern.finditer(text)
            if m.group(0) not in {"0", "1"} and not text[max(0, m.start() - 1):m.start()] == "x"
        ]
        if not matches or len(matches) > int(max_constants):
            return None

        values = []
        pieces = []
        last = 0
        for idx, match in enumerate(matches):
            try:
                values.append(float(match.group(0)))
            except Exception:
                return None
            pieces.append(text[last:match.start()])
            pieces.append(f"__c{idx}")
            last = match.end()
        pieces.append(text[last:])
        template = "".join(pieces)
        initial = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(initial)):
            return None

        def build(params):
            out = template
            for idx, value in enumerate(params):
                out = out.replace(f"__c{idx}", f"({float(value):.12g})")
            return out

        def residuals(params):
            candidate = build(params)
            try:
                pred = self._safe_eval_formula_array(candidate, X_fit)
            except Exception:
                return np.full_like(y_fit, 1e6, dtype=np.float64)
            pred = np.asarray(pred, dtype=np.float64).reshape(-1)
            if pred.shape != y_fit.shape or not np.all(np.isfinite(pred)):
                return np.full_like(y_fit, 1e6, dtype=np.float64)
            return np.clip(pred - y_fit, -1e6, 1e6)

        try:
            result = least_squares(
                residuals,
                initial,
                max_nfev=120,
                loss="soft_l1",
                f_scale=max(1e-6, float(np.std(y_fit)) * 0.1),
            )
        except Exception:
            return None
        if not getattr(result, "success", False):
            return None

        refined_formula = build(result.x)
        try:
            pred_fit = self._safe_eval_formula_array(refined_formula, X_fit)
            pred_val = self._safe_eval_formula_array(refined_formula, X_val)
        except Exception:
            return None
        if not (np.all(np.isfinite(pred_fit)) and np.all(np.isfinite(pred_val))):
            return None
        fit_mse = float(np.mean((pred_fit - y_fit) ** 2))
        val_mse = float(np.mean((pred_val - y_val) ** 2))
        if not np.isfinite(fit_mse) or not np.isfinite(val_mse):
            return None
        val_var = float(np.var(y_val))
        val_r2 = 1.0 if val_var < 1e-15 and val_mse < 1e-15 else (
            0.0 if val_var < 1e-15 else 1.0 - val_mse / val_var
        )
        return {
            "formula": refined_formula,
            "fit_mse": fit_mse,
            "mse": val_mse,
            "validation_mse": val_mse,
            "validation_r2": float(val_r2),
            "complexity": self._formula_complexity(refined_formula),
            "constant_refined": True,
        }

    def _score_formula_candidate(self, formula, X_fit, y_fit, X_val, y_val):
        """Fit affine scaling on training data and score on validation data."""
        text = str(formula or "").strip()
        if not text:
            return None
        try:
            pred_fit = self._safe_eval_formula_array(text, X_fit)
            pred_val = self._safe_eval_formula_array(text, X_val)
        except Exception:
            return None

        fit_mask = np.isfinite(pred_fit) & np.isfinite(y_fit)
        val_mask = np.isfinite(pred_val) & np.isfinite(y_val)
        if int(fit_mask.sum()) < 8 or int(val_mask.sum()) < 4:
            return None

        x_fit = pred_fit[fit_mask]
        t_fit = y_fit[fit_mask]
        x_val = pred_val[val_mask]
        t_val = y_val[val_mask]
        try:
            coef, _, _, _ = np.linalg.lstsq(
                np.column_stack([x_fit, np.ones_like(x_fit)]),
                t_fit,
                rcond=None,
            )
            scale = float(coef[0])
            bias = float(coef[1])
            fit_pred = scale * x_fit + bias
            val_pred = scale * x_val + bias
        except Exception:
            return None

        fit_mse = float(np.mean((fit_pred - t_fit) ** 2))
        val_mse = float(np.mean((val_pred - t_val) ** 2))
        if not np.isfinite(fit_mse) or not np.isfinite(val_mse):
            return None

        val_var = float(np.var(t_val))
        val_r2 = 1.0 if val_var < 1e-15 and val_mse < 1e-15 else (
            0.0 if val_var < 1e-15 else 1.0 - val_mse / val_var
        )
        complexity = max(1, text.count("+") + text.count("-") + text.count("*") + text.count("/") + text.count("^") + 1)

        refined_formula = text
        if abs(scale - 1.0) > 1e-8 or abs(bias) > 1e-8:
            refined_formula = f"(({scale:.12g})*({text})+({bias:.12g}))"

        risk_score = self._formula_risk_score(refined_formula, X_val)
        generalization_gap = float(max(0.0, val_mse - fit_mse) / max(float(np.var(t_val)), 1e-12))

        return {
            "formula": refined_formula,
            "base_formula": text,
            "fit_mse": fit_mse,
            "mse": val_mse,
            "r2": float(val_r2),
            "scale": scale,
            "bias": bias,
            "complexity": complexity,
            "risk_score": risk_score,
            "generalization_gap": generalization_gap,
        }

    def _compute_specialist_screening_diagnostics(self, candidate_formulas, X, y, *, max_candidates=6, max_pairs=5):
        """Summarize coarse segment behavior and pair complementarity for top candidates."""
        state = compute_specialist_state(
            candidate_formulas,
            X,
            y,
            evaluate_formula=self._safe_eval_formula_array,
            complexity_fn=self._formula_complexity,
            family_signature_fn=self._formula_family_signature,
            max_candidates=max_candidates,
            max_pairs=max_pairs,
        )
        if state is None:
            self.specialist_state_ = None
            return None
        self.specialist_state_ = state
        return state.to_dict()

    def _compose_specialist_candidates(self, candidate_formulas, X, y, *, max_candidates=12):
        """Generate and validate a tiny set of specialist-driven formula compositions."""
        state = getattr(self, "specialist_state_", None)
        if state is None:
            return []

        proposals = propose_specialist_compositions(
            state,
            X,
            y,
            evaluate_formula=self._safe_eval_formula_array,
            max_pairs=3,
            min_complementarity=0.30,
        )
        if not proposals:
            return []

        raw_candidates = [proposal.to_candidate_dict() for proposal in proposals]
        refined = self._refine_candidate_formulas(
            raw_candidates,
            X,
            y,
            max_candidates=max(4, int(max_candidates)),
        )
        if not refined:
            return []

        accepted = []
        seen = set()
        for candidate in refined:
            formula = str((candidate or {}).get("formula", "")).strip()
            if not formula:
                continue
            val_r2 = _finite_float(candidate.get("validation_r2"), -1.0)
            complexity = int(candidate.get("complexity") or self._formula_complexity(formula))
            risk = _finite_float(candidate.get("risk_score"), 1.0)
            gap = _finite_float(candidate.get("generalization_gap"), 1.0)
            key = re.sub(r"\s+", "", formula.lower())
            if key in seen:
                continue
            if val_r2 < 0.70 or complexity > 40 or risk > 0.55 or gap > 0.90:
                continue
            seen.add(key)
            accepted.append(candidate)

        if isinstance(self.blackbox_diagnostics_, dict):
            self.blackbox_diagnostics_["specialist_composition_screening"] = {
                "proposal_count": len(raw_candidates),
                "accepted_count": len(accepted),
                "top_proposals": [
                    {
                        "formula": str(candidate.get("formula", ""))[:160],
                        "validation_r2": candidate.get("validation_r2"),
                        "validation_mse": candidate.get("validation_mse"),
                        "complexity": candidate.get("complexity"),
                        "operator": candidate.get("composition_operator"),
                    }
                    for candidate in accepted[:6]
                ],
            }
        return accepted

    def _formula_mse(self, formula, X, y):
        """Evaluate a formula directly on data and return MSE, or inf on failure."""
        text = str(formula or "").strip()
        if not text:
            return float("inf")
        try:
            pred = self._safe_eval_formula_array(text, X)
        except Exception:
            return float("inf")
        pred = np.asarray(pred, dtype=np.float64).reshape(-1)
        target = np.asarray(y, dtype=np.float64).reshape(-1)
        if pred.shape != target.shape:
            return float("inf")
        if not np.all(np.isfinite(pred)):
            return float("inf")
        mse = float(np.mean((pred - target) ** 2))
        return mse if np.isfinite(mse) else float("inf")

    def _select_final_formula(self, incumbent_formula, incumbent_mse, challenger_formula, challenger_mse, X, y):
        """Choose between incumbent and challenger using direct formula evaluation."""
        incumbent_text = str(incumbent_formula or "").strip()
        challenger_text = str(challenger_formula or "").strip()
        if not challenger_text:
            return incumbent_formula, incumbent_mse, "incumbent"
        if not incumbent_text:
            return challenger_formula, challenger_mse, "challenger"

        incumbent_eval = self._formula_mse(incumbent_text, X, y)
        challenger_eval = self._formula_mse(challenger_text, X, y)

        incumbent_score = incumbent_eval if np.isfinite(incumbent_eval) else float(incumbent_mse or float("inf"))
        challenger_score = challenger_eval if np.isfinite(challenger_eval) else float(challenger_mse or float("inf"))

        if challenger_score + 1e-12 < incumbent_score:
            return challenger_formula, challenger_score, "challenger"
        return incumbent_formula, incumbent_score, "incumbent"

    def _compare_blackbox_formulas(self, incumbent_formula, challenger_formula, X, y):
        """Compare two formulas on validation, not just in-sample fit."""
        candidates = []
        if incumbent_formula:
            candidates.append({"formula": incumbent_formula, "source": "incumbent"})
        if challenger_formula:
            candidates.append({"formula": challenger_formula, "source": "challenger"})
        if len(candidates) < 2:
            return None
        choice = self._select_blackbox_pareto_formula(candidates, X, y)
        if choice is None:
            return None
        return "challenger" if choice.get("source") == "challenger" else "incumbent"

    def _select_blackbox_pareto_formula(self, candidates, X, y):
        """Select a validation-stable Pareto winner for blackbox approximation."""
        if not candidates:
            return None
        random_split = self._random_blackbox_validation_split(X, y, validation_fraction=0.25, salt=17)
        edge_split = self._domain_edge_validation_split(X, y, validation_fraction=0.25)
        split = random_split or edge_split
        if split is None:
            return None
        y_val = split["y_val"]
        y_var = max(float(np.var(y_val)), 1e-12)
        scored = []
        seen = set()
        for candidate in candidates:
            formula = str((candidate or {}).get("formula", "")).strip()
            if not formula:
                continue
            key = re.sub(r"\s+", "", formula.lower())
            if key in seen:
                continue
            seen.add(key)
            try:
                pred_fit = self._safe_eval_formula_array(formula, split["X_fit"])
                pred_val = self._safe_eval_formula_array(formula, split["X_val"])
            except Exception:
                continue
            pred_fit = np.asarray(pred_fit, dtype=np.float64).reshape(-1)
            pred_val = np.asarray(pred_val, dtype=np.float64).reshape(-1)
            if pred_fit.shape != split["y_fit"].shape or pred_val.shape != y_val.shape:
                continue
            if not (np.all(np.isfinite(pred_fit)) and np.all(np.isfinite(pred_val))):
                continue
            fit_mse = float(np.mean((pred_fit - split["y_fit"]) ** 2))
            val_mse = float(np.mean((pred_val - y_val) ** 2))
            if not np.isfinite(fit_mse) or not np.isfinite(val_mse):
                continue
            complexity = int((candidate or {}).get("complexity") or self._formula_complexity(formula))
            risk = self._formula_risk_score(formula, split["X_val"])
            gap = float(max(0.0, val_mse - fit_mse) / y_var)
            val_r2 = 1.0 - val_mse / y_var
            edge_mse = None
            edge_r2 = None
            if edge_split is not None and edge_split is not split:
                try:
                    edge_pred = self._safe_eval_formula_array(formula, edge_split["X_val"])
                    edge_pred = np.asarray(edge_pred, dtype=np.float64).reshape(-1)
                    if edge_pred.shape == edge_split["y_val"].shape and np.all(np.isfinite(edge_pred)):
                        edge_mse = float(np.mean((edge_pred - edge_split["y_val"]) ** 2))
                        edge_var = max(float(np.var(edge_split["y_val"])), 1e-12)
                        edge_r2 = 1.0 - edge_mse / edge_var
                except Exception:
                    edge_mse = None
            blend_mse = val_mse
            if edge_mse is not None and np.isfinite(edge_mse):
                blend_mse = 0.72 * val_mse + 0.28 * edge_mse
            score = blend_mse * (1.0 + 0.030 * complexity + 0.50 * risk + 0.25 * gap)
            scored.append({
                "formula": formula,
                "mse": float(np.mean((self._safe_eval_formula_array(formula, X) - y) ** 2)),
                "validation_mse": val_mse,
                "validation_r2": float(val_r2),
                "edge_validation_mse": edge_mse,
                "edge_validation_r2": edge_r2,
                "blended_validation_mse": float(blend_mse),
                "fit_mse": fit_mse,
                "complexity": complexity,
                "risk_score": risk,
                "generalization_gap": gap,
                "pareto_score": float(score),
                "source": (candidate or {}).get("source") or (candidate or {}).get("run_label"),
            })
        if not scored:
            return None

        best_raw = min(scored, key=lambda c: c["validation_mse"])
        eligible = [
            c for c in scored
            if c["validation_mse"] <= best_raw["validation_mse"] * 1.08 + 1e-12
            and c["risk_score"] <= max(0.45, best_raw["risk_score"] + 0.15)
        ]
        selected = min(eligible or scored, key=lambda c: (c["pareto_score"], c["complexity"]))
        selected["evaluated_candidates"] = len(scored)
        selected["best_raw_validation_mse"] = best_raw["validation_mse"]
        selected["selected_by"] = "blackbox_validation_pareto"
        return selected

    def _validate_blackbox_fast_path_candidate(self, formula, mse, X, y):
        """Decide whether a fast-path formula is safe enough to be incumbent."""
        split = self._domain_edge_validation_split(X, y, validation_fraction=0.25)
        if split is None or not formula:
            return {"accepted": True, "reason": "no_validation_split"}
        scored = self._score_formula_candidate(
            formula,
            split["X_fit"],
            split["y_fit"],
            split["X_val"],
            split["y_val"],
        )
        if scored is None:
            return {"accepted": False, "reason": "validation_failed"}
        val_mse = float(scored.get("validation_mse", scored.get("mse", float("inf"))))
        fit_mse = float(scored.get("fit_mse", float("inf")))
        val_var = max(float(np.var(split["y_val"])), 1e-12)
        risk = float(scored.get("risk_score", self._formula_risk_score(formula, split["X_val"])))
        complexity = self._formula_complexity(formula)
        gap = float(max(0.0, val_mse - fit_mse) / val_var)
        train_ratio = val_mse / max(float(mse), 1e-12) if mse is not None and np.isfinite(mse) else 1.0
        accepted = (
            np.isfinite(val_mse)
            and val_mse <= 1.25 * val_var
            and train_ratio <= 3.0
            and gap <= 0.75
            and risk <= 0.45
            and complexity <= 36
        )
        reason = "accepted" if accepted else "unstable_validation"
        return {
            "accepted": bool(accepted),
            "reason": reason,
            "validation_mse": val_mse,
            "validation_r2": 1.0 - val_mse / val_var,
            "fit_mse": fit_mse,
            "train_mse": float(mse) if mse is not None and np.isfinite(mse) else None,
            "validation_to_train_mse": float(train_ratio) if np.isfinite(train_ratio) else None,
            "risk_score": risk,
            "generalization_gap": gap,
            "complexity": complexity,
            "candidate_formula": scored.get("formula"),
        }

    def _should_use_universal_fast_path(self, blackbox_state, fast_path_uncertainty):
        """Disable kitchen-sink fast-path expansion when classifier evidence is weak."""
        if blackbox_state is None or not getattr(blackbox_state, "enabled", False):
            return True
        selected = list(getattr(blackbox_state, "selected_features", []) or [])
        if len(selected) <= 1:
            return True
        if not isinstance(fast_path_uncertainty, dict):
            return True
        entropy = _finite_float(fast_path_uncertainty.get("prediction_entropy"), 0.0)
        margin = _finite_float(fast_path_uncertainty.get("prediction_margin"), 1.0)
        uncertain = bool(fast_path_uncertainty.get("prediction_uncertain", False))
        if uncertain and entropy >= 0.80 and margin <= 0.10:
            return False
        return True

    def _constrain_blackbox_operator_hints(self, operator_hints, blackbox_state):
        """Clamp risky operator families when multivariate fast-path evidence is weak."""
        hints = dict(operator_hints or {})
        ops = set(hints.get("operators", set()))
        uncertainty = (getattr(self, "_fp_result", {}) or {}).get("uncertainty", {})
        conservative = not self._should_use_universal_fast_path(blackbox_state, uncertainty)
        if not (blackbox_state is not None and getattr(blackbox_state, "enabled", False) and conservative):
            hints["operators"] = ops
            return hints

        interaction_terms = list(getattr(blackbox_state, "interaction_terms", []) or [])
        interaction_text = " ".join(interaction_terms).lower()
        keep_periodic = "sin(" in interaction_text or "cos(" in interaction_text
        constrained = {"power", "exp", "log", "rational"}
        ops.difference_update(constrained)
        if keep_periodic:
            ops.add("periodic")
        hints["operators"] = ops
        hints["powers"] = [p for p in list(hints.get("powers", [])) if isinstance(p, (int, np.integer)) and int(p) in (2, 3)]
        hints["has_rational"] = False
        hints["has_exp_decay"] = False
        if isinstance(self.blackbox_diagnostics_, dict):
            self.blackbox_diagnostics_["operator_hint_constraint"] = {
                "conservative": True,
                "kept_operators": sorted(ops),
                "dropped_risky_families": sorted(constrained),
            }
        return hints

    def _derive_blackbox_binary_priors(self, blackbox_state, operator_hints=None):
        """Bias C++ binary search away from fragile rational structures in blackbox mode."""
        if blackbox_state is None or not getattr(blackbox_state, "enabled", False):
            return [], []

        uncertainty = (getattr(self, "_fp_result", {}) or {}).get("uncertainty", {})
        conservative = not self._should_use_universal_fast_path(blackbox_state, uncertainty)
        hints = dict(operator_hints or {})
        ops = set(hints.get("operators", set()))
        interaction_terms = list(getattr(blackbox_state, "interaction_terms", []) or [])
        interaction_text = " ".join(interaction_terms).lower()
        has_periodic = "periodic" in ops or "sin(" in interaction_text or "cos(" in interaction_text
        has_rational = bool(hints.get("has_rational", False)) and not conservative

        base = [0.62, 0.12, 0.26]
        if conservative:
            base = [0.72, 0.06, 0.22]
        elif has_rational:
            base = [0.52, 0.24, 0.24]
        elif has_periodic:
            base = [0.58, 0.10, 0.32]

        multi = []
        if getattr(self, "num_islands", 1) > 1:
            multi = [
                [0.78, 0.04, 0.18],
                [0.60, 0.06, 0.34],
                [0.66, 0.12, 0.22],
                [0.54, 0.18, 0.28] if has_rational else [0.70, 0.05, 0.25],
            ]
            while len(multi) < int(self.num_islands):
                multi.append(list(multi[len(multi) % 4]))

        diagnostics = getattr(self, "blackbox_diagnostics_", None)
        if isinstance(diagnostics, dict):
            diagnostics["binary_operator_priors"] = {
                "global": list(base),
                "multi_island": [list(v) for v in multi] if multi else [],
                "conservative": bool(conservative),
                "has_periodic_signal": bool(has_periodic),
                "has_rational_signal": bool(has_rational),
            }
        return list(base), multi

    def _derive_blackbox_unary_policy(self, blackbox_state, operator_hints=None):
        """Build hard unary operator masks for low-trust multivariate blackbox runs."""
        if blackbox_state is None or not getattr(blackbox_state, "enabled", False):
            return [], [], [], []

        uncertainty = (getattr(self, "_fp_result", {}) or {}).get("uncertainty", {})
        conservative = not self._should_use_universal_fast_path(blackbox_state, uncertainty)
        hints = dict(operator_hints or {})
        ops = set(hints.get("operators", set()))
        interaction_terms = list(getattr(blackbox_state, "interaction_terms", []) or [])
        interaction_text = " ".join(interaction_terms).lower()
        has_periodic = "periodic" in ops or "sin(" in interaction_text or "cos(" in interaction_text
        has_rational = bool(hints.get("has_rational", False)) and not conservative
        has_exp = ("exp" in ops or bool(hints.get("has_exp_decay", False))) and not conservative
        has_log = "log" in ops and not conservative
        permissive = [0, 1, 2, 3, 4]
        safe_periodic = [0, 2] if has_periodic else [2]
        exp_mild = sorted(set(([2, 3] if has_exp else [2]) + ([0] if has_periodic else [])))
        log_mild = sorted(set(([2, 4] if (has_log or has_rational) else [2]) + ([0] if has_periodic else [])))

        # Do not globally hard-mask every island. Keep one permissive search lane.
        global_allowed = []
        if conservative:
            multi = [
                [2],
                safe_periodic,
                exp_mild,
                permissive,
            ]
        else:
            multi = [
                [2],
                safe_periodic,
                exp_mild,
                permissive if (has_log or has_rational or has_exp or has_periodic) else [2, 1],
            ]

        multi = [sorted(set(int(v) for v in row)) for row in multi]
        if getattr(self, "num_islands", 1) > 1:
            while len(multi) < int(self.num_islands):
                multi.append(list(multi[len(multi) % 4]))
        else:
            multi = []

        # Likewise, avoid a global binary hard-mask; constrain by island.
        binary_allowed = []
        multi_binary = []
        if getattr(self, "num_islands", 1) > 1:
            multi_binary = [
                [0],
                [0, 2],
                [0, 2],
                [0, 1, 2],
            ]
            while len(multi_binary) < int(self.num_islands):
                multi_binary.append(list(multi_binary[len(multi_binary) % 4]))

        if isinstance(getattr(self, "blackbox_diagnostics_", None), dict):
            self.blackbox_diagnostics_["unary_operator_policy"] = {
                "allowed_unary_ops": list(global_allowed),
                "multi_allowed_unary_ops": [list(v) for v in multi] if multi else [],
                "allowed_binary_ops": list(binary_allowed),
                "multi_allowed_binary_ops": [list(v) for v in multi_binary] if multi_binary else [],
                "conservative": bool(conservative),
            }
        return global_allowed, multi, binary_allowed, multi_binary

    def _refine_candidate_formulas(self, candidate_formulas, X, y, *, max_candidates=12):
        """Refine symbolic candidates with affine scaling and holdout scoring."""
        if not candidate_formulas:
            return []
        split = self._domain_edge_validation_split(X, y, validation_fraction=0.2)
        if split is None:
            return []

        ranked = []
        seen = set()
        for candidate in candidate_formulas:
            formula = str((candidate or {}).get("formula", "")).strip()
            if not formula:
                continue
            key = re.sub(r"\s+", "", formula.lower())
            if key in seen:
                continue
            seen.add(key)
            scored = self._score_formula_candidate(
                formula,
                split["X_fit"],
                split["y_fit"],
                split["X_val"],
                split["y_val"],
            )
            if scored is None:
                continue
            constant_refined = self._refine_formula_constants(
                scored["formula"],
                split["X_fit"],
                split["y_fit"],
                split["X_val"],
                split["y_val"],
            )
            if (
                constant_refined is not None
                and float(constant_refined.get("validation_mse", float("inf")))
                < float(scored.get("mse", float("inf"))) * 0.995
            ):
                constant_refined["risk_score"] = self._formula_risk_score(
                    constant_refined["formula"],
                    split["X_val"],
                )
                constant_refined["generalization_gap"] = float(
                    max(0.0, constant_refined["validation_mse"] - constant_refined["fit_mse"])
                    / max(float(np.var(split["y_val"])), 1e-12)
                )
                scored.update(constant_refined)
            merged = dict(candidate)
            merged.update({
                "formula": scored["formula"],
                "base_formula": scored["base_formula"],
                "mse": scored["mse"],
                "validation_mse": scored["mse"],
                "validation_r2": scored["r2"],
                "fit_mse": scored["fit_mse"],
                "refined_scale": scored["scale"],
                "refined_bias": scored["bias"],
                "complexity": scored["complexity"],
                "risk_score": scored.get("risk_score", 0.0),
                "generalization_gap": scored.get("generalization_gap", 0.0),
                "constant_refined": bool(scored.get("constant_refined", False)),
            })
            ranked.append(merged)

        ranked.sort(
            key=lambda c: (
                _finite_float(c.get("mse"), float("inf")) * (
                    1.0
                    + 0.25 * _finite_float(c.get("risk_score"), 0.0)
                    + 0.20 * _finite_float(c.get("generalization_gap"), 0.0)
                ),
                _finite_float(c.get("complexity"), float("inf")),
                str(c.get("formula", "")),
            )
        )
        return ranked[: max(1, int(max_candidates))]

    def _formula_family_signature(self, formula):
        text = str(formula or "").strip().lower()
        if not text:
            return "empty"
        if "sin(" in text:
            return "sin"
        if "cos(" in text:
            return "cos"
        if "exp(" in text:
            return "exp"
        if "log(" in text:
            return "log"
        if "/" in text:
            return "rational"
        if "*" in text:
            return "multiplicative"
        if "+" in text or "-" in text:
            return "additive"
        if "^" in text:
            return "power"
        return "univariate"

    def _formula_feature_signature(self, formula):
        text = str(formula or "")
        return tuple(sorted({int(match.group(1)) for match in re.finditer(r"\bx(\d+)\b", text)}))

    def _prune_blackbox_candidate_formulas(self, candidate_formulas, *, max_candidates=12):
        """Keep diverse, high-quality blackbox candidates instead of many near-duplicates."""
        if not candidate_formulas:
            return []

        ordered = sorted(
            candidate_formulas,
            key=lambda c: (
                _finite_float(c.get("mse"), float("inf")),
                -_finite_float(c.get("validation_r2"), -float("inf")),
                _finite_float(c.get("complexity"), float("inf")),
            ),
        )
        kept = []
        seen_formulas = set()
        seen_family_feature = set()
        for cand in ordered:
            formula = str(cand.get("formula", "")).strip()
            if not formula:
                continue
            normalized = re.sub(r"\s+", "", formula.lower())
            if normalized in seen_formulas:
                continue
            family = self._formula_family_signature(formula)
            features = self._formula_feature_signature(formula)
            key = (family, features)

            if key in seen_family_feature:
                if len(kept) >= max(2, int(max_candidates) // 2):
                    continue
            seen_formulas.add(normalized)
            seen_family_feature.add(key)
            kept.append(cand)
            if len(kept) >= max(1, int(max_candidates)):
                break
        return kept

    def _derive_blackbox_operator_hints(self, blackbox_state, candidate_formulas):
        """Convert validated blackbox interactions/candidates into operator-family hints."""
        hints = {
            "operators": set(),
            "powers": [],
            "active_terms": [],
            "has_rational": False,
            "has_exp_decay": False,
        }
        if blackbox_state is None:
            return hints

        interaction_scores = getattr(blackbox_state, "interaction_scores", {}) or {}
        for term in list(getattr(blackbox_state, "interaction_terms", []) or [])[:8]:
            score = float(interaction_scores.get(term, 0.0))
            if score < 0.12:
                continue
            hints["active_terms"].append(term)
            lower = term.lower()
            if "sin(" in lower or "cos(" in lower:
                hints["operators"].add("periodic")
            if "exp(" in lower:
                hints["operators"].add("exp")
                hints["has_exp_decay"] = True
            if "log(" in lower:
                hints["operators"].add("log")
            if "/" in lower:
                hints["operators"].add("rational")
                hints["has_rational"] = True
            if "^2" in lower or "^3" in lower:
                hints["operators"].add("power")
                for power_text in re.findall(r"\^(\d+)", lower):
                    try:
                        hints["powers"].append(int(power_text))
                    except Exception:
                        pass

        for cand in candidate_formulas or []:
            if _finite_float(cand.get("validation_r2"), -1.0) < 0.25:
                continue
            formula = str(cand.get("formula", "")).strip()
            if not formula:
                continue
            hints["active_terms"].append(formula)
            family = self._formula_family_signature(formula)
            if family in ("sin", "cos"):
                hints["operators"].add("periodic")
            elif family == "exp":
                hints["operators"].add("exp")
                hints["has_exp_decay"] = True
            elif family == "log":
                hints["operators"].add("log")
            elif family == "rational":
                hints["operators"].add("rational")
                hints["has_rational"] = True
            elif family == "power":
                hints["operators"].add("power")

        hints["powers"] = sorted(set(int(p) for p in hints["powers"] if isinstance(p, (int, np.integer)) and 1 <= int(p) <= 8))
        hints["active_terms"] = list(dict.fromkeys(hints["active_terms"]))[:12]
        return hints

    def _build_blackbox_candidate_formulas(
        self,
        best_formula,
        best_mse,
        proposer_payload,
        blackbox_state,
        X,
        y,
        *,
        max_candidates,
    ):
        """Build, refine, and prune a shared candidate pool for basis fitting and evolution."""
        raw_candidates = []
        if best_formula:
            raw_candidates.append({
                "formula": best_formula,
                "mse": best_mse if best_mse is not None else float("inf"),
                "from_fast_path": True,
            })

        if isinstance(proposer_payload, dict):
            for cand in proposer_payload.get("candidate_skeletons", [])[:10]:
                formula = str(cand.get("formula", "")).strip()
                if not formula:
                    continue
                raw_candidates.append({
                    "formula": formula,
                    "mse": cand.get("mse", float("inf")),
                    "score": cand.get("score", 0.0),
                    "active_terms": cand.get("active_terms", []),
                    "from_proposer": True,
                })

        if blackbox_state is not None and getattr(blackbox_state, "enabled", False):
            selected_features = list(getattr(blackbox_state, "selected_features", []) or [])
            for term in list(getattr(blackbox_state, "interaction_terms", []) or [])[:8]:
                seed_formula = remap_original_formula_to_reduced(term, selected_features)
                raw_candidates.append({
                    "formula": seed_formula,
                    "mse": float("inf"),
                    "score": float((getattr(blackbox_state, "interaction_scores", {}) or {}).get(term, 0.0)),
                    "active_terms": [seed_formula],
                    "from_blackbox_interaction": True,
                })
            for formula in list(getattr(blackbox_state, "candidate_seed_formulas", []) or [])[:16]:
                seed_formula = remap_original_formula_to_reduced(formula, selected_features)
                raw_candidates.append({
                    "formula": seed_formula,
                    "mse": float("inf"),
                    "score": 0.2,
                    "active_terms": [seed_formula],
                    "from_blackbox_seed": True,
                })

        refined = self._refine_candidate_formulas(
            raw_candidates,
            X,
            y,
            max_candidates=max(
                int(max_candidates),
                8,
            ),
        )
        return self._prune_blackbox_candidate_formulas(
            refined,
            max_candidates=max_candidates,
        )

    def _build_blackbox_formula_pool(self, best_formula, proposer_payload, blackbox_state, n_features):
        """Assemble a compact pool of reduced-space formulas for cheap additive fitting."""
        formulas = []
        seen = set()

        def _add(text, family=None):
            formula = str(text or "").strip()
            if not formula or formula == "0":
                return
            key = re.sub(r"\s+", "", formula.lower())
            if key in seen:
                return
            if family is not None:
                existing = [
                    f for f in formulas
                    if self._formula_family_signature(f) == family
                ]
                if len(existing) >= 4:
                    return
            seen.add(key)
            formulas.append(formula)

        if best_formula:
            _add(best_formula)

        if isinstance(proposer_payload, dict):
            for cand in proposer_payload.get("candidate_skeletons", [])[:8]:
                _add(cand.get("formula", ""), family=self._formula_family_signature(cand.get("formula", "")))

        for local_idx in range(int(max(1, n_features))):
            _add(f"x{local_idx}")
            _add(f"x{local_idx}^2")
            _add(f"x{local_idx}^3")
            _add(f"sin(x{local_idx})")
            _add(f"cos(x{local_idx})")
            _add(f"exp(-abs(x{local_idx}))")

        if blackbox_state is not None and getattr(blackbox_state, "enabled", False):
            selected = list(getattr(blackbox_state, "selected_features", []) or [])
            for term in list(getattr(blackbox_state, "interaction_terms", []) or [])[:8]:
                reduced = remap_original_formula_to_reduced(term, selected)
                _add(reduced, family=self._formula_family_signature(reduced))
            for formula in list(getattr(blackbox_state, "candidate_seed_formulas", []) or [])[:16]:
                reduced = remap_original_formula_to_reduced(formula, selected)
                _add(reduced, family=self._formula_family_signature(reduced))

        for i in range(int(max(1, n_features))):
            for j in range(i + 1, int(max(1, n_features))):
                _add(f"x{i}*x{j}")
                _add(f"x{i}+x{j}")
                _add(f"x{i}-x{j}")

        return formulas[:32]

    def _fit_blackbox_basis_model(self, X, y, candidate_formulas, *, max_terms=4):
        """Fit a small additive symbolic model from a screened basis pool."""
        if not candidate_formulas:
            return None
        split = self._split_blackbox_holdout(X, y, validation_fraction=0.2)
        if split is None:
            return None

        X_fit = split["X_fit"]
        y_fit = split["y_fit"]
        X_val = split["X_val"]
        y_val = split["y_val"]
        y_fit = np.asarray(y_fit, dtype=np.float64).reshape(-1)
        y_val = np.asarray(y_val, dtype=np.float64).reshape(-1)
        base_val_mse = float(np.mean((y_val - float(np.mean(y_fit))) ** 2))

        basis = []
        seen_signatures = []
        for formula in candidate_formulas:
            try:
                fit_values = self._safe_eval_formula_array(formula, X_fit).reshape(-1)
                val_values = self._safe_eval_formula_array(formula, X_val).reshape(-1)
                full_values = self._safe_eval_formula_array(formula, X).reshape(-1)
            except Exception:
                continue
            if (
                fit_values.shape[0] != X_fit.shape[0]
                or val_values.shape[0] != X_val.shape[0]
                or full_values.shape[0] != X.shape[0]
            ):
                continue
            if not (np.all(np.isfinite(fit_values)) and np.all(np.isfinite(val_values)) and np.all(np.isfinite(full_values))):
                continue
            if float(np.std(fit_values)) < 1e-10:
                continue

            duplicate = False
            for prev in seen_signatures:
                if np.corrcoef(prev, fit_values)[0, 1] > 0.995:
                    duplicate = True
                    break
            if duplicate:
                continue
            seen_signatures.append(fit_values)
            basis.append({
                "formula": formula,
                "fit": fit_values,
                "val": val_values,
                "full": full_values,
                "complexity": self._formula_complexity(formula),
            })

        if not basis:
            return None

        selected = []
        selected_cols_fit = []
        selected_cols_val = []
        best_val_mse = base_val_mse

        for _ in range(int(max(1, max_terms))):
            best_choice = None
            for cand in basis:
                if cand in selected:
                    continue
                cols_fit = selected_cols_fit + [cand["fit"]]
                cols_val = selected_cols_val + [cand["val"]]
                design_fit = np.column_stack(cols_fit + [np.ones_like(y_fit)])
                design_val = np.column_stack(cols_val + [np.ones_like(y_val)])
                try:
                    coef, _, _, _ = np.linalg.lstsq(design_fit, y_fit, rcond=None)
                    val_pred = design_val @ coef
                    val_mse = float(np.mean((val_pred - y_val) ** 2))
                except Exception:
                    continue
                complexity = sum(item["complexity"] for item in selected) + cand["complexity"]
                penalized = val_mse * (1.0 + 0.003 * complexity)
                if best_choice is None or penalized < best_choice["penalized"]:
                    best_choice = {
                        "cand": cand,
                        "coef": coef,
                        "val_mse": val_mse,
                        "penalized": penalized,
                    }

            if best_choice is None:
                break
            improvement = best_val_mse - float(best_choice["val_mse"])
            if improvement <= max(1e-8, 0.01 * max(best_val_mse, 1e-8)):
                break

            selected.append(best_choice["cand"])
            selected_cols_fit.append(best_choice["cand"]["fit"])
            selected_cols_val.append(best_choice["cand"]["val"])
            best_val_mse = float(best_choice["val_mse"])

        if not selected:
            return None

        design_full = np.column_stack([item["full"] for item in selected] + [np.ones(X.shape[0])])
        y_full = np.asarray(y, dtype=np.float64).reshape(-1)
        try:
            coef_full, _, _, _ = np.linalg.lstsq(design_full, y_full, rcond=None)
        except Exception:
            return None

        terms = []
        for weight, item in zip(coef_full[:-1], selected):
            if not np.isfinite(weight) or abs(float(weight)) < 1e-8:
                continue
            terms.append(f"({float(weight):.12g})*({item['formula']})")
        bias = float(coef_full[-1])
        if abs(bias) > 1e-8 or not terms:
            terms.append(f"({bias:.12g})")
        formula = "+".join(terms) if terms else "0"

        full_pred = self._safe_eval_formula_array(formula, X)
        full_mse = float(np.mean((full_pred - y_full) ** 2))
        y_val_var = float(np.var(y_val))
        val_r2 = 1.0 if y_val_var < 1e-15 and best_val_mse < 1e-15 else (
            0.0 if y_val_var < 1e-15 else 1.0 - best_val_mse / y_val_var
        )

        return {
            "formula": formula,
            "mse": full_mse,
            "validation_mse": best_val_mse,
            "validation_r2": float(val_r2),
            "selected_terms": [item["formula"] for item in selected],
            "n_terms": len(selected),
            "complexity": self._formula_complexity(formula),
        }

    def _fit_blackbox_engineered_basis_model(self, X, y, *, max_terms=10):
        """Fit a compact validation-selected engineered basis for Track 1.

        This is deliberately not a broad kitchen-sink expansion. It gives
        multivariate blackbox datasets a strong symbolic baseline made from
        linear, low-degree polynomial, pairwise interaction, and a few stable
        unary transforms, then exports only the selected terms.
        """
        split = self._random_blackbox_validation_split(X, y, validation_fraction=0.25, salt=31)
        if split is None:
            return None

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        X_fit = split["X_fit"]
        y_fit = np.asarray(split["y_fit"], dtype=np.float64).reshape(-1)
        X_val = split["X_val"]
        y_val = np.asarray(split["y_val"], dtype=np.float64).reshape(-1)
        n_features = int(X.shape[1])

        def add_feature(pool, name, full_values):
            values = np.asarray(full_values, dtype=np.float64).reshape(-1)
            if values.shape[0] != X.shape[0] or not np.all(np.isfinite(values)):
                return
            if float(np.std(values)) < 1e-10:
                return
            pool.append({"name": name, "full": values})

        pool = []
        for j in range(n_features):
            xj = X[:, j]
            add_feature(pool, f"x{j}", xj)
            add_feature(pool, f"x{j}^2", xj ** 2)
            add_feature(pool, f"x{j}^3", xj ** 3)
            add_feature(pool, f"sin(x{j})", np.sin(xj))
            add_feature(pool, f"cos(x{j})", np.cos(xj))
            add_feature(pool, f"exp(-abs(x{j}))", np.exp(-np.abs(np.clip(xj, -20.0, 20.0))))

        for i in range(n_features):
            xi = X[:, i]
            for j in range(i + 1, n_features):
                xj = X[:, j]
                add_feature(pool, f"x{i}*x{j}", xi * xj)
                add_feature(pool, f"(x{i}-x{j})^2", (xi - xj) ** 2)
                add_feature(pool, f"x{i}+x{j}", xi + xj)
                add_feature(pool, f"x{i}-x{j}", xi - xj)
                add_feature(pool, f"x{i}*sin(x{j})", xi * np.sin(xj))
                add_feature(pool, f"x{j}*sin(x{i})", xj * np.sin(xi))
                add_feature(pool, f"x{i}*cos(x{j})", xi * np.cos(xj))
                add_feature(pool, f"x{j}*cos(x{i})", xj * np.cos(xi))
                add_feature(pool, f"x{i}*exp(-abs(x{j}))", xi * np.exp(-np.abs(np.clip(xj, -20.0, 20.0))))
                add_feature(pool, f"x{j}*exp(-abs(x{i}))", xj * np.exp(-np.abs(np.clip(xi, -20.0, 20.0))))
                add_feature(pool, f"x{i}/(abs(x{j})+1e-6)", xi / (np.abs(xj) + 1e-6))
                add_feature(pool, f"x{j}/(abs(x{i})+1e-6)", xj / (np.abs(xi) + 1e-6))

        if not pool:
            return None

        fit_idx = split["fit_idx"]
        val_idx = split["val_idx"]
        A_full = np.column_stack([item["full"] for item in pool])
        A_fit = A_full[fit_idx]
        A_val = A_full[val_idx]

        max_pool_terms = 180 if n_features >= 8 else 260
        if A_fit.shape[1] > max_pool_terms:
            y_probe = y_fit - float(np.mean(y_fit))
            y_scale = float(np.std(y_probe))
            scores = []
            for idx, item in enumerate(pool):
                values = A_fit[:, idx]
                v_scale = float(np.std(values))
                if v_scale < 1e-10 or y_scale < 1e-10:
                    score = 0.0
                else:
                    try:
                        score = abs(float(np.corrcoef(values, y_probe)[0, 1]))
                    except Exception:
                        score = 0.0
                    if not np.isfinite(score):
                        score = 0.0
                linear_bonus = 1.0 if re.fullmatch(r"x\d+", item["name"]) else 0.0
                scores.append((linear_bonus, score, idx))
            scores.sort(key=lambda row: (row[0], row[1]), reverse=True)
            keep = sorted(idx for _, _, idx in scores[:max_pool_terms])
            pool = [pool[idx] for idx in keep]
            A_full = A_full[:, keep]
            A_fit = A_fit[:, keep]
            A_val = A_val[:, keep]

        mu = np.mean(A_fit, axis=0)
        sigma = np.std(A_fit, axis=0)
        sigma = np.where(sigma < 1e-10, 1.0, sigma)
        Z_fit = (A_fit - mu) / sigma
        Z_val = (A_val - mu) / sigma

        y_mean = float(np.mean(y_fit))
        y_center = y_fit - y_mean

        try:
            from sklearn.linear_model import ElasticNetCV, RidgeCV
            from sklearn.exceptions import ConvergenceWarning
        except Exception:
            ElasticNetCV = RidgeCV = None
            ConvergenceWarning = Warning

        candidate_weights = []
        if ElasticNetCV is not None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    enet = ElasticNetCV(
                        l1_ratio=[0.15, 0.35, 0.65, 0.9],
                        alphas=np.logspace(-4, 0, 24),
                        cv=min(5, max(2, len(y_fit) // 40)),
                        max_iter=12000,
                        tol=1e-3,
                        random_state=0,
                    )
                    enet.fit(Z_fit, y_center)
                candidate_weights.append(np.asarray(enet.coef_, dtype=np.float64))
            except Exception:
                pass
        if RidgeCV is not None:
            try:
                ridge = RidgeCV(alphas=np.logspace(-4, 3, 24))
                ridge.fit(Z_fit, y_center)
                candidate_weights.append(np.asarray(ridge.coef_, dtype=np.float64))
            except Exception:
                pass

        if not candidate_weights:
            try:
                coef, _, _, _ = np.linalg.lstsq(
                    np.column_stack([Z_fit, np.ones(Z_fit.shape[0])]),
                    y_fit,
                    rcond=None,
                )
                candidate_weights.append(np.asarray(coef[:-1], dtype=np.float64))
            except Exception:
                return None

        best = None
        y_val_var = max(float(np.var(y_val)), 1e-12)
        term_grid = sorted(set([3, 5, 8, int(max_terms)]))
        for weights in candidate_weights:
            if weights.shape[0] != A_full.shape[1] or not np.all(np.isfinite(weights)):
                continue
            order = np.argsort(np.abs(weights))[::-1]
            for k in term_grid:
                active = [idx for idx in order[: max(1, min(k, len(order)))] if abs(weights[idx]) > 1e-10]
                if not active:
                    continue
                design_fit = np.column_stack([A_fit[:, active], np.ones(A_fit.shape[0])])
                design_val = np.column_stack([A_val[:, active], np.ones(A_val.shape[0])])
                try:
                    coef, _, _, _ = np.linalg.lstsq(design_fit, y_fit, rcond=None)
                    pred_val = design_val @ coef
                    val_mse = float(np.mean((pred_val - y_val) ** 2))
                except Exception:
                    continue
                if not np.isfinite(val_mse):
                    continue
                complexity = sum(self._formula_complexity(pool[idx]["name"]) for idx in active)
                score = val_mse * (1.0 + 0.01 * len(active) + 0.002 * complexity)
                if best is None or score < best["score"]:
                    best = {
                        "active": active,
                        "coef": coef,
                        "validation_mse": val_mse,
                        "score": score,
                    }

        if best is None:
            return None

        active = best["active"]
        design_full = np.column_stack([A_full[:, active], np.ones(A_full.shape[0])])
        try:
            coef_full, _, _, _ = np.linalg.lstsq(design_full, y, rcond=None)
        except Exception:
            return None

        terms = []
        selected_terms = []
        for weight, idx in zip(coef_full[:-1], active):
            if not np.isfinite(weight) or abs(float(weight)) < 1e-8:
                continue
            selected_terms.append(pool[idx]["name"])
            terms.append(f"({float(weight):.12g})*({pool[idx]['name']})")
        bias = float(coef_full[-1])
        if abs(bias) > 1e-8 or not terms:
            terms.append(f"({bias:.12g})")
        formula = "+".join(terms) if terms else "0"

        try:
            pred_full = self._safe_eval_formula_array(formula, X)
        except Exception:
            return None
        full_mse = float(np.mean((pred_full - y) ** 2))
        if not np.isfinite(full_mse):
            return None

        val_r2 = 1.0 - float(best["validation_mse"]) / y_val_var
        return {
            "formula": formula,
            "mse": full_mse,
            "validation_mse": float(best["validation_mse"]),
            "validation_r2": float(val_r2),
            "selected_terms": selected_terms,
            "n_terms": len(selected_terms),
            "complexity": self._formula_complexity(formula),
            "source": "engineered_basis",
        }

    def _derive_blackbox_search_plan(
        self,
        blackbox_state,
        *,
        fast_path_uncertainty=None,
        proposer_uncertainty=None,
        proposer_plan=None,
        candidate_screening=None,
    ):
        """Heuristically scale breadth/depth for multivariate blackbox search."""
        base_plan = {
            "uncertainty_score": 0.0,
            "selection_uncertainty": 0.0,
            "interaction_pressure": 0.0,
            "candidate_strength": 0.0,
            "candidate_diversity": 0.0,
            "breadth_multiplier": 1.0,
            "depth_multiplier": 1.0,
            "generation_multiplier": 1.0,
            "population_multiplier": 1.0,
            "seed_budget": 8,
            "screening_budget": 8,
            "basis_max_terms": 4,
            "candidate_acceptance_r2": 0.985,
            "candidate_shrink_r2": 0.95,
            "acceptable_complexity": 15,
            "early_stop_max_nodes": 50,
            "timeout_multiplier": 1.0,
            "focus": "balanced",
            "allowed_unary_ops": [],
            "multi_allowed_unary_ops": [],
            "binary_op_priors": [],
            "multi_binary_op_priors": [],
            "allowed_binary_ops": [],
            "multi_allowed_binary_ops": [],
        }
        if blackbox_state is None or not getattr(blackbox_state, "enabled", False):
            return base_plan

        selected = list(getattr(blackbox_state, "selected_features", []) or [])
        selected_count = max(1, len(selected))
        original_count = int(getattr(self, "original_n_features_in_", selected_count) or selected_count)

        feature_scores = getattr(blackbox_state, "feature_scores", {}) or {}
        score_values = sorted(
            [float(v) for v in feature_scores.values() if np.isfinite(v)],
            reverse=True,
        )
        top_score = score_values[0] if score_values else 0.0
        next_score = score_values[1] if len(score_values) > 1 else 0.0
        score_gap = max(0.0, top_score - next_score)
        score_gap_ratio = score_gap / max(abs(top_score), 1e-12) if top_score > 0 else 0.0

        selection_uncertainty = 1.0 - float(np.clip(score_gap_ratio, 0.0, 1.0))
        if getattr(blackbox_state, "feature_selection_uncertain", False):
            selection_uncertainty = max(selection_uncertainty, 0.85)
        if getattr(blackbox_state, "reason", "") == "retained_all_features_small_problem":
            selection_uncertainty *= 0.5

        interaction_scores = getattr(blackbox_state, "interaction_scores", {}) or {}
        interaction_terms = list(getattr(blackbox_state, "interaction_terms", []) or [])
        interaction_best = max(
            [float(v) for v in interaction_scores.values() if np.isfinite(v)],
            default=0.0,
        )
        interaction_density = len(interaction_terms) / max(1.0, float(selected_count - 1))
        interaction_pressure = float(np.clip(0.55 * interaction_best + 0.15 * interaction_density, 0.0, 1.0))

        feature_span_pressure = float(np.clip((selected_count - 1) / max(4.0, original_count - 1), 0.0, 1.0))

        screening = candidate_screening if isinstance(candidate_screening, dict) else {}
        candidate_best_r2 = float(np.clip(float(screening.get("best_validation_r2", 0.0) or 0.0), 0.0, 1.0))
        candidate_count = int(max(0, screening.get("candidate_count", 0) or 0))
        candidate_family_count = int(max(0, screening.get("family_count", 0) or 0))
        candidate_strength = candidate_best_r2
        candidate_diversity = float(np.clip(candidate_family_count / max(1.0, min(6.0, float(candidate_count or 1))), 0.0, 1.0))

        def _uncertainty_signal(payload):
            if not isinstance(payload, dict):
                return 0.0
            entropy = payload.get("prediction_entropy")
            margin = payload.get("prediction_margin")
            uncertain_flag = bool(payload.get("prediction_uncertain", False))
            signal = 0.0
            if entropy is not None:
                try:
                    signal = max(signal, float(np.clip(float(entropy), 0.0, 1.0)))
                except Exception:
                    pass
            if margin is not None:
                try:
                    margin = float(margin)
                    signal = max(signal, float(np.clip(1.0 - min(max(margin, 0.0), 1.0), 0.0, 1.0)))
                except Exception:
                    pass
            if uncertain_flag:
                signal = max(signal, 0.75)
            return float(np.clip(signal, 0.0, 1.0))

        fast_uncertainty = _uncertainty_signal(fast_path_uncertainty)
        proposer_unc = 0.0
        if isinstance(proposer_uncertainty, dict):
            proposer_unc = _uncertainty_signal(proposer_uncertainty)
        elif proposer_uncertainty is not None:
            proposer_unc = float(np.clip(float(proposer_uncertainty), 0.0, 1.0))

        uncertainty_score = float(np.clip(
            0.36 * selection_uncertainty
            + 0.22 * interaction_pressure
            + 0.18 * fast_uncertainty
            + 0.12 * proposer_unc
            + 0.12 * (1.0 - candidate_strength),
            0.0,
            1.0,
        ))

        breadth_multiplier = float(np.clip(
            1.0
            + 0.95 * selection_uncertainty
            + 0.25 * fast_uncertainty
            + 0.18 * (1.0 - interaction_pressure)
            + 0.20 * (1.0 - candidate_strength),
            0.75,
            3.5,
        ))
        depth_multiplier = float(np.clip(
            1.0
            + 0.70 * interaction_pressure
            + 0.25 * proposer_unc
            + 0.20 * feature_span_pressure
            + 0.20 * candidate_diversity
            - 0.35 * candidate_strength,
            0.75,
            4.0,
        ))
        if uncertainty_score < 0.3:
            breadth_multiplier *= 0.85
            depth_multiplier *= 0.9
        elif uncertainty_score > 0.7:
            breadth_multiplier *= 1.05
            depth_multiplier *= 1.05

        if getattr(blackbox_state, "enabled", False):
            # For blackbox Track 1, spend uncertainty budget on screening first.
            breadth_multiplier = min(breadth_multiplier, 2.25)
            depth_multiplier = min(depth_multiplier, 2.5)

        generation_multiplier = float(np.clip(depth_multiplier, 0.75, 4.0))
        population_multiplier = float(np.clip(breadth_multiplier, 0.75, 3.5))
        seed_budget = int(np.clip(
            round(8 + 8 * selection_uncertainty + 4 * interaction_pressure + 3 * fast_uncertainty + 2 * proposer_unc + 2 * candidate_diversity),
            8,
            24,
        ))
        screening_budget = int(np.clip(
            round(8 + 10 * selection_uncertainty + 8 * interaction_pressure + 6 * (1.0 - candidate_strength) + 3 * candidate_diversity),
            6,
            28,
        ))
        basis_max_terms = int(np.clip(
            round(3 + 2 * interaction_pressure + 2 * candidate_diversity + 2 * (1.0 - candidate_strength)),
            2,
            8,
        ))
        candidate_acceptance_r2 = float(np.clip(
            0.985 - 0.03 * interaction_pressure - 0.01 * candidate_diversity,
            0.93,
            0.985,
        ))
        candidate_shrink_r2 = float(np.clip(
            candidate_acceptance_r2 - 0.04,
            0.88,
            0.96,
        ))

        if getattr(blackbox_state, "enabled", False):
            generation_multiplier = float(np.clip(generation_multiplier, 0.75, 2.0))
            population_multiplier = float(np.clip(population_multiplier, 0.80, 1.85))
            seed_budget = int(np.clip(seed_budget, 6, 14))
            screening_budget = int(np.clip(screening_budget, 8, 24))
            basis_max_terms = int(np.clip(basis_max_terms, 3, 6))

        acceptable_complexity = int(np.clip(
            round(15 + 5 * uncertainty_score + 3 * interaction_pressure + 2 * feature_span_pressure + 2 * candidate_diversity),
            10,
            80,
        ))
        early_stop_max_nodes = int(np.clip(
            round(50 + 14 * uncertainty_score + 6 * interaction_pressure + 5 * feature_span_pressure - 6 * candidate_strength),
            10,
            120,
        ))
        timeout_multiplier = float(np.clip(
            0.82 + 0.16 * breadth_multiplier + 0.24 * depth_multiplier + 0.14 * (screening_budget / 12.0),
            0.8,
            2.8,
        ))

        if getattr(blackbox_state, "enabled", False):
            acceptable_complexity = int(np.clip(acceptable_complexity, 10, 32))
            early_stop_max_nodes = int(np.clip(early_stop_max_nodes, 16, 64))
            timeout_multiplier = float(np.clip(timeout_multiplier, 0.75, 1.0))

        focus = "balanced"
        if candidate_strength >= candidate_acceptance_r2:
            focus = "screen_accept"
        elif screening_budget >= seed_budget + 4:
            focus = "screening"
        elif breadth_multiplier > depth_multiplier + 0.25:
            focus = "breadth"
        elif depth_multiplier > breadth_multiplier + 0.25:
            focus = "depth"

        binary_op_priors, multi_binary_op_priors = self._derive_blackbox_binary_priors(
            blackbox_state,
            {},
        )
        allowed_unary_ops, multi_allowed_unary_ops, allowed_binary_ops, multi_allowed_binary_ops = (
            self._derive_blackbox_unary_policy(blackbox_state, {})
        )
        plan = {
            "uncertainty_score": uncertainty_score,
            "selection_uncertainty": selection_uncertainty,
            "interaction_pressure": interaction_pressure,
            "candidate_strength": candidate_strength,
            "candidate_diversity": candidate_diversity,
            "breadth_multiplier": breadth_multiplier,
            "depth_multiplier": depth_multiplier,
            "generation_multiplier": generation_multiplier,
            "population_multiplier": population_multiplier,
            "seed_budget": seed_budget,
            "screening_budget": screening_budget,
            "basis_max_terms": basis_max_terms,
            "candidate_acceptance_r2": candidate_acceptance_r2,
            "candidate_shrink_r2": candidate_shrink_r2,
            "acceptable_complexity": acceptable_complexity,
            "early_stop_max_nodes": early_stop_max_nodes,
            "timeout_multiplier": timeout_multiplier,
            "focus": focus,
            "allowed_unary_ops": allowed_unary_ops,
            "multi_allowed_unary_ops": multi_allowed_unary_ops,
            "binary_op_priors": binary_op_priors,
            "multi_binary_op_priors": multi_binary_op_priors,
            "allowed_binary_ops": allowed_binary_ops,
            "multi_allowed_binary_ops": multi_allowed_binary_ops,
        }

        if proposer_plan:
            # The current multivariate proposer is a proxy. Let it add seeds,
            # but do not let it undo blackbox screening-first caps.
            raw_generation_multiplier = plan["generation_multiplier"] * float(
                _clamp_float(proposer_plan.get("generation_multiplier"), 1.0, 0.5, 4.0)
            )
            raw_population_multiplier = plan["population_multiplier"] * float(
                _clamp_float(proposer_plan.get("population_multiplier"), 1.0, 0.5, 3.0)
            )
            raw_seed_budget = max(plan["seed_budget"], int(proposer_plan.get("seed_budget", plan["seed_budget"])))
            raw_complexity = max(
                plan["acceptable_complexity"],
                int(proposer_plan.get("acceptable_complexity", plan["acceptable_complexity"])),
            )
            raw_max_nodes = max(
                plan["early_stop_max_nodes"],
                int(proposer_plan.get("early_stop_max_nodes", plan["early_stop_max_nodes"])),
            )
            if getattr(blackbox_state, "enabled", False):
                plan["generation_multiplier"] = float(np.clip(raw_generation_multiplier, 0.75, 2.0))
                plan["population_multiplier"] = float(np.clip(raw_population_multiplier, 0.80, 1.85))
                plan["seed_budget"] = int(np.clip(raw_seed_budget, 6, 14))
                plan["acceptable_complexity"] = int(np.clip(raw_complexity, 10, 32))
                plan["early_stop_max_nodes"] = int(np.clip(raw_max_nodes, 16, 64))
                plan["screening_budget"] = max(plan["screening_budget"], plan["seed_budget"])
            else:
                plan["generation_multiplier"] = raw_generation_multiplier
                plan["population_multiplier"] = raw_population_multiplier
                plan["seed_budget"] = raw_seed_budget
                plan["acceptable_complexity"] = raw_complexity
                plan["early_stop_max_nodes"] = raw_max_nodes
            plan["timeout_multiplier"] = float(np.clip(
                plan["timeout_multiplier"] * float(_clamp_float(proposer_plan.get("timeout_multiplier"), 1.0, 0.5, 3.0)),
                0.75 if getattr(blackbox_state, "enabled", False) else 0.8,
                1.0 if getattr(blackbox_state, "enabled", False) else 3.0,
            ))
        return plan

    def _resolve_classifier_path(self):
        """Resolve classifier model path relative to repo root."""
        p = Path(self.classifier_path)
        if p.is_absolute() and p.exists():
            return str(p)
        repo_path = _REPO_ROOT / self.classifier_path
        if repo_path.exists():
            return str(repo_path)
        return str(p)

    def _resolve_universal_proposer_path(self):
        """Resolve proposer checkpoint path relative to repo root with fallback."""
        candidates = [
            self.universal_proposer_path,
            "models/universal_proposer_multi.pt",
            "models/universal_proposer_robust.pt",
        ]
        for candidate in candidates:
            p = Path(candidate)
            if p.is_absolute() and p.exists():
                return str(p)
            repo_path = _REPO_ROOT / candidate
            if repo_path.exists():
                return str(repo_path)
        return str(Path(self.universal_proposer_path))

    def _safe_eval_formula_array(self, formula, X):
        """Safely evaluate a symbolic formula over a feature matrix."""
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
            "Abs": np.abs,
            "sign": np.sign,
            "pi": np.pi,
            "E": np.e,
        }
        for i in range(X.shape[1]):
            context[f"x{i}"] = X[:, i]
        if X.shape[1] == 1:
            context["x"] = X[:, 0]

        expr = formula.strip()
        expr = re.sub(r'\|([^|]+)\|', r'abs(\1)', expr)
        expr = re.sub(r'\^', r'**', expr)
        expr = expr.replace('np.', '')

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_pred = eval(expr, {"__builtins__": None}, context)

        if isinstance(y_pred, (int, float)):
            y_pred = np.full(X.shape[0], y_pred, dtype=np.float64)
        else:
            y_pred = np.asarray(y_pred, dtype=np.float64)
        return np.where(np.isfinite(y_pred), y_pred, 0.0)

    def _formula_domain_failure_rate(self, formula, X):
        """Estimate how often a displayed formula leaves its numeric domain."""
        text = str(formula or "").strip()
        if not text:
            return None
        context = {
            "np": np,
            "log": np.log,
            "sin": np.sin,
            "cos": np.cos,
            "exp": np.exp,
            "sqrt": np.sqrt,
            "abs": np.abs,
            "Abs": np.abs,
            "sign": np.sign,
            "pi": np.pi,
            "E": np.e,
        }
        for i in range(X.shape[1]):
            context[f"x{i}"] = X[:, i]
        if X.shape[1] == 1:
            context["x"] = X[:, 0]

        expr = re.sub(r'\|([^|]+)\|', r'abs(\1)', text)
        expr = re.sub(r'\^', r'**', expr)
        expr = expr.replace('np.', '')
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = eval(expr, {"__builtins__": None}, context)
            if isinstance(raw, (int, float)):
                raw = np.full(X.shape[0], raw, dtype=np.float64)
            raw_arr = np.asarray(raw, dtype=np.float64).reshape(-1)
            if raw_arr.shape[0] != X.shape[0]:
                return 1.0
            return float(np.mean(~np.isfinite(raw_arr)))
        except Exception:
            return 1.0

    def _passes_cross_validation_skip_guard(self, formula, X, y):
        """Return True when fast-path formula is stable enough to skip evolution."""
        diagnostics = {
            'enabled': bool(self.cv_skip_guard_enabled),
            'fold_r2': [],
            'min_fold_r2': None,
            'std_fold_r2': None,
            'passed': True,
            'reason': 'disabled',
        }

        if not self.cv_skip_guard_enabled:
            self.fast_path_cv_guard_ = diagnostics
            return True

        n_samples = int(X.shape[0])
        n_folds = int(max(2, self.cv_skip_guard_folds))
        if n_samples < int(max(n_folds * 2, self.cv_skip_guard_min_samples)):
            diagnostics['reason'] = 'insufficient_samples'
            self.fast_path_cv_guard_ = diagnostics
            return True

        try:
            y_pred = self._safe_eval_formula_array(formula, X)
        except Exception:
            diagnostics['passed'] = False
            diagnostics['reason'] = 'formula_eval_failed'
            self.fast_path_cv_guard_ = diagnostics
            return False

        idx = np.arange(n_samples)
        seed = 0 if self.random_state is None else int(self.random_state)
        rng = np.random.RandomState(seed)
        rng.shuffle(idx)
        folds = [f for f in np.array_split(idx, n_folds) if len(f) > 0]

        fold_r2 = []
        for fold_idx in folds:
            y_fold = y[fold_idx]
            pred_fold = y_pred[fold_idx]
            var_fold = float(np.var(y_fold))
            if var_fold < 1e-15:
                r2_fold = 1.0 if float(np.mean((pred_fold - y_fold) ** 2)) < 1e-15 else 0.0
            else:
                mse_fold = float(np.mean((pred_fold - y_fold) ** 2))
                r2_fold = 1.0 - mse_fold / var_fold
            if np.isfinite(r2_fold):
                fold_r2.append(float(r2_fold))

        diagnostics['fold_r2'] = fold_r2
        if not fold_r2:
            diagnostics['passed'] = False
            diagnostics['reason'] = 'no_valid_folds'
            self.fast_path_cv_guard_ = diagnostics
            return False

        min_fold_r2 = float(np.min(fold_r2))
        std_fold_r2 = float(np.std(fold_r2))
        diagnostics['min_fold_r2'] = min_fold_r2
        diagnostics['std_fold_r2'] = std_fold_r2

        passed = (
            min_fold_r2 >= float(self.cv_skip_guard_min_fold_r2)
            and std_fold_r2 <= float(self.cv_skip_guard_max_r2_std)
        )
        diagnostics['passed'] = bool(passed)
        diagnostics['reason'] = 'ok' if passed else 'unstable_fold_performance'
        self.fast_path_cv_guard_ = diagnostics
        return bool(passed)

    def _run_universal_proposer_dual_path(self, X, y, fast_path_result, blackbox_state=None):
        """Optional side-by-side proposer run for routing diagnostics.

        Returns:
            Tuple[fpip_payload_or_none, force_evolution_bool]
        """
        if not self.use_universal_proposer:
            return None, False

        try:
            from glassbox.universal_proposer import (
                load_universal_proposer_checkpoint,
                propose_fpip_v2_from_xy,
            )

            if self._universal_proposer_model is None:
                model_path = self._resolve_universal_proposer_path()
                self._universal_proposer_model = load_universal_proposer_checkpoint(model_path, device=self.device)

            X_arr = np.asarray(X, dtype=np.float64)
            if X_arr.ndim == 1 or int(X_arr.shape[1]) == 1:
                x1 = X_arr.reshape(-1)
                proposer_status = "ok"
            else:
                x_centered = X_arr - np.mean(X_arr, axis=0, keepdims=True)
                x1 = np.linalg.norm(x_centered, axis=1)
                proposer_status = "ok_multivariate_proxy"

            y1 = np.asarray(y, dtype=np.float64).reshape(-1)

            fit_diag = {}
            if isinstance(fast_path_result, dict):
                fit_diag["mse"] = fast_path_result.get("mse")
                fit_diag["residual_suspicious"] = bool(
                    (fast_path_result.get("residual_diagnostics") or {}).get("residual_suspicious", False)
                )

            payload = propose_fpip_v2_from_xy(
                self._universal_proposer_model,
                x=x1,
                y=y1,
                top_k=int(max(1, self.universal_proposer_top_k)),
                fit_diagnostics=fit_diag,
                interaction_hints={
                    "multivariate_proxy": bool(int(np.asarray(X).ndim) > 1 and np.asarray(X).shape[1] > 1),
                    "selected_feature_count": int(np.asarray(X).shape[1]) if np.asarray(X).ndim > 1 else 1,
                    "selected_features": list(getattr(blackbox_state, "selected_features", [])) if blackbox_state is not None else [],
                    "dropped_features": list(getattr(blackbox_state, "dropped_features", [])) if blackbox_state is not None else [],
                },
                device=self.device,
            )

            if not payload:
                self.universal_proposer_status_ = "error:empty_payload"
                if self.universal_proposer_log_routing:
                    print("  [Proposer skipped: empty payload]")
                return None, False

            self.universal_proposer_status_ = proposer_status
            self.universal_proposer_fpip_v2_ = payload

            if self.universal_proposer_log_routing:
                route = payload.get("routing_signal") or {}
                print(
                    "  [Proposer] "
                    f"guided={route.get('recommend_guided_evolution')} "
                    f"reason={route.get('reason')}"
                )

            force_evolution = (
                (not self.universal_proposer_shadow_mode)
                and bool(payload.get("valid", False))
                and bool((payload.get("routing_signal") or {}).get("recommend_guided_evolution", False))
            )
            return payload, force_evolution
        except Exception as e:
            self.universal_proposer_status_ = f"error:{e}"
            if self.universal_proposer_log_routing:
                print(f"  [Proposer skipped: {e}]")
            return None, False

    def _simplify_formula(self, formula):
        """Apply multipass formula simplification."""
        if not formula or not self.use_simplification:
            return formula
        try:
            from glassbox.sr.cpp import _core
            simplified = _core.simplify_formula(
                formula,
                int_tol=self.simplification_int_tol,
                zero_tol=self.simplification_zero_tol,
                max_passes=6,
                use_nsimplify=True,
                use_identities=True,
                n_features=self.n_features_in_
            )
            return simplified
        except Exception:
            return formula

    def _stage_residual_symbolic_fit(self, X, y, base_formula, *, _allow_recursion=False):
        """Fit a second symbolic stage on the residual when it improves holdout fit."""
        if not self.enable_residual_stage or not _allow_recursion or not base_formula or not self.use_guided_evolution:
            return None
        if X.shape[1] < 1:
            return None

        try:
            y_pred = self._safe_eval_formula_array(base_formula, X)
        except Exception:
            return None

        residual = np.asarray(y, dtype=np.float64).reshape(-1) - np.asarray(y_pred, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(residual)) or float(np.var(residual)) < 1e-12:
            return None

        residual_state = discover_blackbox_interactions(X, residual)
        self.residual_blackbox_diagnostics_ = residual_state
        self._residual_stage_guard_ = {"enabled": True, "allowed": bool(_allow_recursion)}

        holdout_n = max(1, int(round(X.shape[0] * 0.2)))
        if X.shape[0] < 20 or holdout_n >= X.shape[0]:
            return None
        X_fit, X_holdout = X[:-holdout_n], X[-holdout_n:]
        y_fit, y_holdout = residual[:-holdout_n], residual[-holdout_n:]

        residual_est = self.__class__(
            population_size=max(20, int(self.population_size // 2)),
            generations=max(20, int(self.generations // 3)),
            early_stop_mse=self.early_stop_mse,
            random_state=self.random_state,
            p_min=self.p_min,
            p_max=self.p_max,
            use_nsga2=self.use_nsga2,
            num_islands=self.num_islands,
            migration_interval=self.migration_interval,
            migration_size=self.migration_size,
            arithmetic_temperature=self.arithmetic_temperature,
            use_fast_path=self.use_fast_path,
            use_guided_evolution=self.use_guided_evolution,
            use_simplification=self.use_simplification,
            classifier_path=self.classifier_path,
            simplification_int_tol=self.simplification_int_tol,
            simplification_zero_tol=self.simplification_zero_tol,
            max_power=self.max_power,
            timeout=max(20, int(self.timeout // 2)),
            evolution_skip_r2=self.evolution_skip_r2,
            multi_start_runs=1,
            adaptive_compute_budget=self.adaptive_compute_budget,
            min_compute_budget=self.min_compute_budget,
            max_compute_budget=self.max_compute_budget,
            cv_skip_guard_enabled=self.cv_skip_guard_enabled,
            cv_skip_guard_folds=self.cv_skip_guard_folds,
            cv_skip_guard_min_fold_r2=self.cv_skip_guard_min_fold_r2,
            cv_skip_guard_max_r2_std=self.cv_skip_guard_max_r2_std,
            cv_skip_guard_min_samples=self.cv_skip_guard_min_samples,
            use_universal_proposer=self.use_universal_proposer,
            universal_proposer_path=self.universal_proposer_path,
            universal_proposer_shadow_mode=self.universal_proposer_shadow_mode,
            universal_proposer_log_routing=False,
            universal_proposer_top_k=self.universal_proposer_top_k,
            blackbox_feature_selection=False,
            blackbox_mode=False,
            blackbox_max_features=self.blackbox_max_features,
            blackbox_standardize=self.blackbox_standardize,
            blackbox_min_features_to_select=self.blackbox_min_features_to_select,
            enable_residual_stage=False,
            device=self.device,
            skip_evolution_if_bloated=self.skip_evolution_if_bloated,
            bloat_term_threshold=self.bloat_term_threshold,
        )

        try:
            residual_est.fit(X_fit, y_fit)
            residual_formula = residual_est.get_formula()
            if residual_formula and residual_formula != "0":
                try:
                    base_pred = self._safe_eval_formula_array(base_formula, X_holdout)
                    res_pred = residual_est._safe_eval_formula_array(residual_formula, X_holdout)
                    combined = base_pred + res_pred
                    base_mse = float(np.mean((base_pred - (y[-holdout_n:])) ** 2))
                    combined_mse = float(np.mean((combined - (y[-holdout_n:])) ** 2))
                    if np.isfinite(combined_mse) and combined_mse < base_mse:
                        return residual_formula
                except Exception:
                    return None
        except Exception:
            return None
        return None

    def _run_residual_boosting(self, X, y, base_formula):
        """Run a multi-stage symbolic boosting loop on top of base_formula."""
        if not self.enable_residual_stage or not base_formula or not self.use_guided_evolution:
            return base_formula

        def local_r2(y_true, y_pred):
            y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
            y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
            var = float(np.var(y_true))
            if var < 1e-15:
                return 1.0 if np.mean((y_true - y_pred)**2) < 1e-15 else 0.0
            return float(1.0 - np.mean((y_true - y_pred)**2) / var)

        max_boosting_stages = getattr(self, "max_boosting_stages", 3)

        try:
            pred = self._safe_eval_formula_array(base_formula, X)
            base_r2 = local_r2(y, pred)
            if base_r2 > 0.9999:
                return base_formula
        except Exception:
            return base_formula

        current_formula = base_formula
        self.boosting_stages_ = []

        holdout_n = max(1, int(round(X.shape[0] * 0.2)))
        if X.shape[0] < 20 or holdout_n >= X.shape[0]:
            return base_formula

        X_fit, X_holdout = X[:-holdout_n], X[-holdout_n:]
        y_fit, y_holdout = y[:-holdout_n], y[-holdout_n:]

        learning_rates = [0.5, 0.8, 1.0]

        for stage in range(max_boosting_stages):
            try:
                pred_all = self._safe_eval_formula_array(current_formula, X)
                pred_holdout = self._safe_eval_formula_array(current_formula, X_holdout)
            except Exception:
                break

            current_holdout_mse = float(np.mean((pred_holdout - y_holdout)**2))
            current_holdout_r2 = local_r2(y_holdout, pred_holdout)

            if current_holdout_r2 > 0.999:
                break

            orig_timeout = self.timeout
            stage_timeout = max(5, orig_timeout // (2 ** (stage + 1)))
            self.timeout = stage_timeout

            try:
                h_k = self._stage_residual_symbolic_fit(X, y, current_formula, _allow_recursion=True)
            finally:
                self.timeout = orig_timeout

            if not h_k or h_k == "0":
                break

            best_eta = None
            best_holdout_mse = current_holdout_mse
            best_combined_formula = None

            try:
                h_pred_holdout = self._safe_eval_formula_array(h_k, X_holdout)
            except Exception:
                break

            for eta in learning_rates:
                combined_formula = f"({current_formula}) + (({eta:.6g}) * ({h_k}))"
                try:
                    combined_pred = pred_holdout + eta * h_pred_holdout
                    mse = float(np.mean((combined_pred - y_holdout)**2))
                    if mse < best_holdout_mse:
                        best_holdout_mse = mse
                        best_eta = eta
                        best_combined_formula = combined_formula
                except Exception:
                    continue

            if best_eta is None:
                break

            try:
                best_pred_holdout = pred_holdout + best_eta * h_pred_holdout
                best_holdout_r2 = local_r2(y_holdout, best_pred_holdout)
            except Exception:
                break

            r2_improvement = best_holdout_r2 - current_holdout_r2
            if r2_improvement < 0.005:
                break

            refined_list = self._refine_candidate_formulas(
                [{"formula": best_combined_formula, "source": f"residual_boosting_stage_{stage}"}],
                X,
                y,
                max_candidates=1,
            )
            if refined_list:
                current_formula = refined_list[0]["formula"]
            else:
                current_formula = best_combined_formula

            self.boosting_stages_.append({
                "stage": stage,
                "h_k": h_k,
                "eta": best_eta,
                "combined_formula": current_formula
            })

            if best_holdout_r2 > 0.999:
                break

        return current_formula

    def _detect_frequencies(self, X, y):
        """Detect dominant frequencies via FFT, with optional phase info."""
        try:
            x_t = torch.tensor(X[:, 0], dtype=torch.float32).reshape(-1, 1)
            y_t = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

            # Get rich phase info for the fast-path pipeline
            phase_info = detect_dominant_frequency(
                x_t, y_t, n_frequencies=3, return_phase_info=True,
            )
            self._fft_phase_info = phase_info  # stash for later use

            omegas = phase_info.get('omegas', [1.0])
            if omegas and omegas[0] == 1.0:
                return []
            return omegas or []
        except Exception:
            self._fft_phase_info = None
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
        self.has_composed_seeds_ = False
        self.specialist_track_ = "incumbent path"
        self.n_features_in_ = X.shape[1]
        self.original_n_features_in_ = X.shape[1]
        fit_start = _time.time()

        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)

        blackbox_enabled = (
            bool(self.blackbox_feature_selection)
            and (
                self.blackbox_mode is True
                or (
                    self.blackbox_mode == "auto"
                    and X.shape[1] > 1
                )
            )
        )
        X_original = X
        y_original = y
        X_search, y_search, blackbox_state = prepare_blackbox_search(
            X,
            y,
            enabled=blackbox_enabled,
            max_features=int(self.blackbox_max_features),
            standardize=bool(self.blackbox_standardize),
            min_features_to_select=int(self.blackbox_min_features_to_select),
        )
        feature_selection_fallback = None
        if (
            blackbox_enabled
            and getattr(blackbox_state, "enabled", False)
            and X.shape[1] > X_search.shape[1]
            and X.shape[1] <= 14
        ):
            selected_cols = list(getattr(blackbox_state, "selected_features", []) or [])
            selected_tail_r2 = self._ridge_tail_validation_r2(X_original, y_original, selected_cols)
            all_tail_r2 = self._ridge_tail_validation_r2(X_original, y_original, None)
            feature_selection_fallback = {
                "selected_tail_r2": selected_tail_r2,
                "all_tail_r2": all_tail_r2,
                "selected_features": selected_cols,
                "n_original_features": int(X.shape[1]),
            }
            self._blackbox_original_linear_fallback = None
            if (
                selected_tail_r2 is not None
                and all_tail_r2 is not None
                and all_tail_r2 > 0.05
                and (
                    all_tail_r2 > selected_tail_r2 + 0.06
                    or (
                        selected_tail_r2 < 0.65
                        and all_tail_r2 >= selected_tail_r2 - 0.02
                    )
                )
            ):
                self._blackbox_original_linear_fallback = self._fit_ridge_formula(
                    X_original,
                    y_original,
                    None,
                )
                feature_selection_fallback["activated"] = True
                feature_selection_fallback["reason"] = "all_features_tail_ridge_candidate"
                if self._blackbox_original_linear_fallback is not None:
                    feature_selection_fallback["fallback_validation_r2"] = self._blackbox_original_linear_fallback.get("validation_r2")
                    feature_selection_fallback["fallback_n_terms"] = self._blackbox_original_linear_fallback.get("n_terms")
            else:
                feature_selection_fallback["activated"] = False
                self._blackbox_original_linear_fallback = None
        else:
            self._blackbox_original_linear_fallback = None
        self._blackbox_feature_fallback_activated = bool(
            isinstance(feature_selection_fallback, dict)
            and feature_selection_fallback.get("activated")
        )
        self.blackbox_state_ = blackbox_state
        self.blackbox_diagnostics_ = state_to_dict(blackbox_state)
        if isinstance(self.blackbox_diagnostics_, dict) and feature_selection_fallback is not None:
            self.blackbox_diagnostics_["feature_selection_fallback"] = feature_selection_fallback
        self.blackbox_search_plan_ = {}
        blackbox_candidate_accepted = False
        blackbox_evolution_ran = False
        blackbox_evolution_improved = False

        if blackbox_state.enabled:
            X = X_search
            y = y_search
            self.n_features_in_ = X.shape[1]
            if self.universal_proposer_log_routing:
                print(
                    "  [Blackbox] selected features "
                    f"{blackbox_state.selected_features} / {self.original_n_features_in_}"
                )

        detected_omegas = self._detect_frequencies(X, y)

        best_formula = None
        best_mse = float('inf')
        operator_hints = {}
        demoted_fast_path_candidate = None
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
                from classifier_fast_path import run_fast_path  # type: ignore

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
                    fp_uncertainty = fp_result.get("uncertainty") or {}
                    fp_details = fp_result.get("details") or {}
                    already_compact = bool(fp_details.get("compact_multivariate_basis", False))
                    if (
                        blackbox_state.enabled
                        and not already_compact
                        and not self._should_use_universal_fast_path(blackbox_state, fp_uncertainty)
                    ):
                        if isinstance(self.blackbox_diagnostics_, dict):
                            self.blackbox_diagnostics_["fast_path_auto_expand"] = False
                        fp_result = run_fast_path(
                            x_t, y_t,
                            classifier_path=classifier_path,
                            detected_omegas=detected_omegas,
                            op_constraints=None,
                            auto_expand=False,
                            device=self.device,
                            exact_match_threads=1,
                            exact_match_enabled=True,
                            exact_match_max_basis=200,
                            max_power=self.max_power,
                            simplify_formula_output=False,
                        )
                    elif isinstance(self.blackbox_diagnostics_, dict) and blackbox_state.enabled:
                        self.blackbox_diagnostics_["fast_path_auto_expand"] = not already_compact
                    best_formula = fp_result['formula']
                    best_mse = fp_result.get('mse', float('inf'))
                    operator_hints = fp_result.get('operator_hints') or {}
                    # Stash for uncertainty-coupled budget routing and candidate seeding
                    self._fp_result = fp_result
                    if blackbox_state.enabled:
                        gate = self._validate_blackbox_fast_path_candidate(best_formula, best_mse, X, y)
                        if isinstance(self.blackbox_diagnostics_, dict):
                            self.blackbox_diagnostics_["fast_path_validation_gate"] = gate
                        if not gate.get("accepted", True):
                            demoted_fast_path_candidate = {
                                "formula": gate.get("candidate_formula") or best_formula,
                                "mse": gate.get("validation_mse", best_mse),
                                "validation_mse": gate.get("validation_mse"),
                                "validation_r2": gate.get("validation_r2"),
                                "complexity": gate.get("complexity", self._formula_complexity(best_formula)),
                                "risk_score": gate.get("risk_score", 0.0),
                                "generalization_gap": gate.get("generalization_gap", 0.0),
                                "from_fast_path": True,
                                "demoted_fast_path": True,
                                "source": "demoted_fast_path",
                            }
                            best_formula = None
                            best_mse = float("inf")
            except Exception as e:
                self._fp_result = None
                print(f"  [Fast-path skipped: {e}]")

        self.universal_proposer_fpip_v2_ = None
        _, proposer_forces_evolution = self._run_universal_proposer_dual_path(
            X,
            y,
            getattr(self, '_fp_result', None),
            getattr(self, "blackbox_state_", None),
        )

        # ── Stage 2: C++ Evolution ──
        # Only run evolution if:
        #   - No formula found yet, OR
        #   - R² is below the skip threshold (default 0.999)
        #   - Cross-validation guard says fast-path fit is unstable
        #   - We haven't exceeded the timeout
        current_r2 = _r2_from_mse(best_mse) if best_formula else -1.0
        term_count = (best_formula.count('+') + best_formula.count('-')) if best_formula else 0
        fast_path_cv_ok = True

        if (
            best_formula is not None
            and best_mse is not None
            and math.isfinite(best_mse)
            and current_r2 >= self.evolution_skip_r2
        ):
            fast_path_cv_ok = self._passes_cross_validation_skip_guard(best_formula, X, y)
        else:
            self.fast_path_cv_guard_ = {
                'enabled': bool(self.cv_skip_guard_enabled),
                'fold_r2': [],
                'min_fold_r2': None,
                'std_fold_r2': None,
                'passed': True,
                'reason': 'not_applicable',
            }

        # Optional benchmark policy: if fast-path is very bloated, keep it as-is
        # and avoid launching evolution search for this sample.
        if (
            self.skip_evolution_if_bloated
            and best_formula is not None
            and term_count > int(self.bloat_term_threshold)
            and not (blackbox_state is not None and blackbox_state.enabled)
        ):
            need_evolution = False
        else:
            need_evolution = (
                best_formula is None or
                best_mse is None or
                not math.isfinite(best_mse) or
                current_r2 < self.evolution_skip_r2 or
                not fast_path_cv_ok or
                term_count > 10 # Higher threshold for Stage 1 bloat
            )

        # Uncertainty-coupled budget routing: pass FPIP uncertainty metrics
        _fp_uncertainty = None
        _fp = getattr(self, '_fp_result', None)
        if isinstance(_fp, dict):
            _fp_uncertainty = _fp.get('uncertainty')

        # Override/blend with Universal Proposer's uncertainty if available
        if self.universal_proposer_fpip_v2_ and self.universal_proposer_fpip_v2_.get("valid"):
            proposer_unc = self.universal_proposer_fpip_v2_.get("sequence_uncertainty", {})
            if "entropy" in proposer_unc and proposer_unc["entropy"] is not None:
                if _fp_uncertainty is None:
                    _fp_uncertainty = {}
                # Take the max uncertainty between fast-path and proposer
                _fp_uncertainty["prediction_entropy"] = max(
                    _fp_uncertainty.get("prediction_entropy", 0.0), 
                    proposer_unc["entropy"]
                )
                _fp_uncertainty["prediction_margin"] = min(
                    _fp_uncertainty.get("prediction_margin", 1.0), 
                    proposer_unc.get("margin", 1.0)
                )

        proposer_payload = (
            self.universal_proposer_fpip_v2_
            if isinstance(self.universal_proposer_fpip_v2_, dict)
            else {}
        )
        candidate_screening = None
        blackbox_state = getattr(self, "blackbox_state_", None)
        if blackbox_state is not None and blackbox_state.enabled:
            preview_candidates = self._build_blackbox_candidate_formulas(
                best_formula,
                best_mse,
                proposer_payload,
                blackbox_state,
                X,
                y,
                max_candidates=10,
            )
            if demoted_fast_path_candidate is not None:
                preview_candidates = self._prune_blackbox_candidate_formulas(
                    [demoted_fast_path_candidate] + list(preview_candidates or []),
                    max_candidates=10,
                )
            if preview_candidates:
                families = {
                    self._formula_family_signature(c.get("formula", ""))
                    for c in preview_candidates
                    if str(c.get("formula", "")).strip()
                }
                candidate_screening = {
                    "candidate_count": len(preview_candidates),
                    "family_count": len(families),
                    "best_validation_r2": max(
                        _finite_float(c.get("validation_r2"), -1.0)
                        for c in preview_candidates
                    ),
                }
        proposer_plan = proposer_payload.get("search_plan", {})
        if not isinstance(proposer_plan, dict):
            proposer_plan = {}
        blackbox_search_plan = self._derive_blackbox_search_plan(
            getattr(self, "blackbox_state_", None),
            fast_path_uncertainty=_fp_uncertainty,
            proposer_uncertainty=proposer_payload.get("sequence_uncertainty", {}),
            proposer_plan=proposer_plan,
            candidate_screening=candidate_screening,
        )
        self.blackbox_search_plan_ = blackbox_search_plan
        if isinstance(self.blackbox_diagnostics_, dict):
            self.blackbox_diagnostics_["search_plan"] = blackbox_search_plan

        candidate_formulas = None
        if blackbox_state is not None and blackbox_state.enabled:
            candidate_formulas = self._build_blackbox_candidate_formulas(
                best_formula,
                best_mse,
                proposer_payload,
                blackbox_state,
                X,
                y,
                max_candidates=max(
                    8,
                    int(blackbox_search_plan.get("seed_budget", 8)),
                ),
            )
            if demoted_fast_path_candidate is not None:
                candidate_formulas = self._prune_blackbox_candidate_formulas(
                    [demoted_fast_path_candidate] + list(candidate_formulas or []),
                    max_candidates=max(
                        8,
                        int(blackbox_search_plan.get("seed_budget", 8)),
                    ),
                )
            interaction_hints = self._derive_blackbox_operator_hints(
                blackbox_state,
                candidate_formulas,
            )
            operator_hints = dict(operator_hints or {})
            operator_hints["operators"] = set(operator_hints.get("operators", set()))
            operator_hints["operators"].update(interaction_hints.get("operators", set()))
            operator_hints["powers"] = sorted(set(
                list(operator_hints.get("powers", [])) + list(interaction_hints.get("powers", []))
            ))
            operator_hints["active_terms"] = list(dict.fromkeys(
                list(operator_hints.get("active_terms", [])) + list(interaction_hints.get("active_terms", []))
            ))[:16]
            operator_hints["has_rational"] = bool(
                operator_hints.get("has_rational", False) or interaction_hints.get("has_rational", False)
            )
            operator_hints["has_exp_decay"] = bool(
                operator_hints.get("has_exp_decay", False) or interaction_hints.get("has_exp_decay", False)
            )
            operator_hints = self._constrain_blackbox_operator_hints(operator_hints, blackbox_state)
            binary_op_priors, multi_binary_op_priors = self._derive_blackbox_binary_priors(
                blackbox_state,
                operator_hints,
            )
            allowed_unary_ops, multi_allowed_unary_ops, allowed_binary_ops, multi_allowed_binary_ops = (
                self._derive_blackbox_unary_policy(blackbox_state, operator_hints)
            )
            if blackbox_search_plan:
                blackbox_search_plan["allowed_unary_ops"] = list(allowed_unary_ops)
                blackbox_search_plan["multi_allowed_unary_ops"] = [list(v) for v in multi_allowed_unary_ops]
                blackbox_search_plan["binary_op_priors"] = list(binary_op_priors)
                blackbox_search_plan["multi_binary_op_priors"] = [list(v) for v in multi_binary_op_priors]
                blackbox_search_plan["allowed_binary_ops"] = list(allowed_binary_ops)
                blackbox_search_plan["multi_allowed_binary_ops"] = [list(v) for v in multi_allowed_binary_ops]
            if isinstance(self.blackbox_diagnostics_, dict):
                self.blackbox_diagnostics_["candidate_screening"] = {
                    "candidate_count": len(candidate_formulas or []),
                    "top_candidates": [
                        {
                            "formula": str(c.get("formula", ""))[:160],
                            "validation_r2": c.get("validation_r2"),
                            "validation_mse": c.get("validation_mse"),
                            "complexity": c.get("complexity"),
                        }
                        for c in (candidate_formulas or [])[:6]
                    ],
                    "interaction_operator_hints": sorted(operator_hints.get("operators", set())),
                }
                if self.enable_specialist_screening_diagnostics:
                    specialist_screening = self._compute_specialist_screening_diagnostics(
                        candidate_formulas,
                        X,
                        y,
                        max_candidates=6,
                        max_pairs=5,
                    )
                    if specialist_screening is not None:
                        self.blackbox_diagnostics_["candidate_screening"]["specialist_screening"] = specialist_screening
                        
                        specialist_candidates = []
                        if getattr(self, "enable_specialist_composition_screening", True):
                            specialist_candidates = self._compose_specialist_candidates(
                                candidate_formulas,
                                X,
                                y,
                                max_candidates=max(
                                    8,
                                    int(blackbox_search_plan.get("screening_budget", 8)),
                                ),
                            )
                        
                        # Phase 4: Guided Residual Evolution From Merged Formulas
                        residual_merged_candidates = []
                        if self.enable_residual_stage and specialist_candidates:
                            for cand in list(specialist_candidates)[:2]:  # Cap at top 2 compositions to keep compute budget small
                                formula = cand.get("formula")
                                if not formula:
                                    continue
                                val_r2 = _finite_float(cand.get("validation_r2"), -1.0)
                                if val_r2 >= 0.75:
                                    res_form = self._stage_residual_symbolic_fit(X, y, formula, _allow_recursion=True)
                                    if res_form and res_form != "0":
                                        combined_formula = f"({formula})+({res_form})"
                                        refined_list = self._refine_candidate_formulas(
                                            [{
                                                "formula": combined_formula,
                                                "source": "specialist_residual_composition",
                                                "from_specialist_composition": True,
                                            }],
                                            X,
                                            y,
                                            max_candidates=1,
                                        )
                                        if refined_list:
                                            residual_merged_candidates.extend(refined_list)

                        if specialist_candidates or residual_merged_candidates:
                            self.has_composed_seeds_ = True
                            candidate_formulas = self._prune_blackbox_candidate_formulas(
                                list(residual_merged_candidates) + list(specialist_candidates) + list(candidate_formulas or []),
                                max_candidates=max(
                                    8,
                                    int(blackbox_search_plan.get("seed_budget", 8)),
                                ),
                            )
                            self.blackbox_diagnostics_["candidate_screening"]["candidate_count"] = len(candidate_formulas or [])
                            self.blackbox_diagnostics_["candidate_screening"]["top_candidates"] = [
                                {
                                    "formula": str(c.get("formula", ""))[:160],
                                    "validation_r2": c.get("validation_r2"),
                                    "validation_mse": c.get("validation_mse"),
                                    "complexity": c.get("complexity"),
                                }
                                for c in (candidate_formulas or [])[:6]
                            ]

        if getattr(self, "blackbox_state_", None) is not None and self.blackbox_state_.enabled:
            basis_pool = self._build_blackbox_formula_pool(
                best_formula,
                proposer_payload,
                self.blackbox_state_,
                self.n_features_in_,
            )
            for cand in candidate_formulas or []:
                formula = str(cand.get("formula", "")).strip()
                if formula and formula not in basis_pool:
                    basis_pool.insert(0, formula)
            basis_pool = list(dict.fromkeys(basis_pool))[:32]
            basis_result = self._fit_blackbox_basis_model(
                X,
                y,
                basis_pool,
                max_terms=int(blackbox_search_plan.get("basis_max_terms", 4)),
            )
            if basis_result is not None:
                self.blackbox_basis_model_ = basis_result
                if isinstance(self.blackbox_diagnostics_, dict):
                    self.blackbox_diagnostics_["basis_model"] = {
                        "validation_r2": basis_result.get("validation_r2"),
                        "validation_mse": basis_result.get("validation_mse"),
                        "selected_terms": basis_result.get("selected_terms"),
                        "n_terms": basis_result.get("n_terms"),
                    }
                basis_mse = float(basis_result.get("mse", float("inf")))
                if basis_mse < best_mse or best_formula is None:
                    best_formula = basis_result.get("formula", best_formula)
                    best_mse = basis_mse
                    updated_candidates = [{
                        "formula": best_formula,
                        "mse": best_mse,
                        "validation_mse": basis_result.get("validation_mse", best_mse),
                        "validation_r2": basis_result.get("validation_r2", -1.0),
                        "complexity": basis_result.get("complexity", self._formula_complexity(best_formula)),
                        "from_basis_model": True,
                    }]
                    if candidate_formulas:
                        updated_candidates.extend(candidate_formulas)
                    candidate_formulas = self._prune_blackbox_candidate_formulas(
                        updated_candidates,
                        max_candidates=max(
                            8,
                            int(blackbox_search_plan.get("seed_budget", 8)),
                        ),
                    )
            else:
                self.blackbox_basis_model_ = None
            engineered_result = self._fit_blackbox_engineered_basis_model(
                X,
                y,
                max_terms=(
                    12 if getattr(self, "_blackbox_feature_fallback_activated", False)
                    else max(6, int(blackbox_search_plan.get("basis_max_terms", 4)) + 4)
                ),
            )
            self.blackbox_engineered_basis_model_ = engineered_result
            if engineered_result is not None:
                if isinstance(self.blackbox_diagnostics_, dict):
                    self.blackbox_diagnostics_["engineered_basis_model"] = {
                        "validation_r2": engineered_result.get("validation_r2"),
                        "validation_mse": engineered_result.get("validation_mse"),
                        "selected_terms": engineered_result.get("selected_terms"),
                        "n_terms": engineered_result.get("n_terms"),
                    }
                eng_mse = float(engineered_result.get("mse", float("inf")))
                eng_val_r2 = float(engineered_result.get("validation_r2", -1.0))
                basis_val_r2 = float((basis_result or {}).get("validation_r2", -1.0)) if isinstance(basis_result, dict) else -1.0
                if (
                    best_formula is None
                    or eng_mse < best_mse
                    or eng_val_r2 > max(basis_val_r2, -1.0) + 0.03
                ):
                    best_formula = engineered_result.get("formula", best_formula)
                    best_mse = eng_mse
                    updated_candidates = [dict(engineered_result)]
                    if candidate_formulas:
                        updated_candidates.extend(candidate_formulas)
                    candidate_formulas = self._prune_blackbox_candidate_formulas(
                        updated_candidates,
                        max_candidates=max(
                            8,
                            int(blackbox_search_plan.get("seed_budget", 8)),
                        ),
                    )
        else:
            self.blackbox_basis_model_ = None
            self.blackbox_engineered_basis_model_ = None

        effective_timeout = self._estimate_compute_budget(
            X,
            current_r2,
            term_count,
            uncertainty=_fp_uncertainty,
        ) * float(blackbox_search_plan.get("timeout_multiplier", 1.0))

        basis_result = getattr(self, "blackbox_basis_model_", None)
        if isinstance(basis_result, dict):
            basis_val_r2 = float(basis_result.get("validation_r2", -1.0))
            basis_terms = int(basis_result.get("n_terms", 99))
            if (
                basis_val_r2 >= float(blackbox_search_plan.get("candidate_acceptance_r2", 0.985))
                and basis_terms <= int(blackbox_search_plan.get("basis_max_terms", 4))
            ):
                need_evolution = False
                blackbox_candidate_accepted = True
            elif basis_val_r2 >= float(blackbox_search_plan.get("candidate_shrink_r2", 0.95)):
                effective_timeout = min(effective_timeout, max(20.0, 0.4 * effective_timeout))

        fp_details_for_budget = (
            (getattr(self, "_fp_result", {}) or {}).get("details", {})
            if isinstance(getattr(self, "_fp_result", None), dict)
            else {}
        )
        compact_fast_path = bool(fp_details_for_budget.get("compact_multivariate_basis", False))
        compact_terms = int(fp_details_for_budget.get("n_nonzero", 99) or 99)
        screening_best_r2 = -1.0
        if isinstance(candidate_screening, dict):
            screening_best_r2 = float(candidate_screening.get("best_validation_r2", -1.0) or -1.0)
        if (
            getattr(self, "blackbox_state_", None) is not None
            and self.blackbox_state_.enabled
            and compact_fast_path
            and compact_terms <= 6
            and current_r2 >= 0.80
            and screening_best_r2 < float(blackbox_search_plan.get("candidate_shrink_r2", 0.95))
        ):
            effective_timeout = min(effective_timeout, 18.0)
            blackbox_search_plan["population_multiplier"] = min(
                float(blackbox_search_plan.get("population_multiplier", 1.0)),
                0.90,
            )
            blackbox_search_plan["generation_multiplier"] = min(
                float(blackbox_search_plan.get("generation_multiplier", 1.0)),
                0.75,
            )
            blackbox_search_plan["seed_budget"] = min(
                int(blackbox_search_plan.get("seed_budget", 8)),
                8,
            )
            if isinstance(self.blackbox_diagnostics_, dict):
                self.blackbox_diagnostics_["evolution_budget_policy"] = "short_compact_blackbox_validation_probe"

        basis_val_r2_for_budget = -1.0
        if isinstance(basis_result, dict):
            basis_val_r2_for_budget = float(basis_result.get("validation_r2", -1.0))
        if (
            getattr(self, "blackbox_state_", None) is not None
            and self.blackbox_state_.enabled
            and screening_best_r2 < 0.65
            and basis_val_r2_for_budget < 0.70
        ):
            effective_timeout = min(effective_timeout, 24.0)
            blackbox_search_plan["population_multiplier"] = min(
                float(blackbox_search_plan.get("population_multiplier", 1.0)),
                0.80,
            )
            blackbox_search_plan["generation_multiplier"] = min(
                float(blackbox_search_plan.get("generation_multiplier", 1.0)),
                0.70,
            )
            blackbox_search_plan["seed_budget"] = min(
                int(blackbox_search_plan.get("seed_budget", 8)),
                8,
            )
            blackbox_search_plan["focus"] = "weak_screening_probe"
            if isinstance(self.blackbox_diagnostics_, dict):
                self.blackbox_diagnostics_["evolution_budget_policy"] = "short_weak_screening_probe"

        if (
            getattr(self, "_blackbox_feature_fallback_activated", False)
            and getattr(self, "blackbox_state_", None) is not None
            and self.blackbox_state_.enabled
            and self.n_features_in_ >= 8
            and best_formula is not None
        ):
            need_evolution = False
            blackbox_candidate_accepted = True
            if isinstance(self.blackbox_diagnostics_, dict):
                self.blackbox_diagnostics_["evolution_budget_policy"] = "skip_high_dim_feature_fallback"
                self.blackbox_diagnostics_["evolution_skipped_reason"] = "all_feature_supervised_basis"

        if isinstance(self.blackbox_diagnostics_, dict) and getattr(self, "blackbox_state_", None) is not None and self.blackbox_state_.enabled:
            self.blackbox_diagnostics_["search_inflation"] = {
                "population_multiplier": float(blackbox_search_plan.get("population_multiplier", 1.0)),
                "generation_multiplier": float(blackbox_search_plan.get("generation_multiplier", 1.0)),
                "timeout_multiplier": float(blackbox_search_plan.get("timeout_multiplier", 1.0)),
                "seed_budget": int(blackbox_search_plan.get("seed_budget", 0)),
                "screening_budget": int(blackbox_search_plan.get("screening_budget", 0)),
                "focus": blackbox_search_plan.get("focus", "balanced"),
            }

        if need_evolution and _elapsed() < effective_timeout:
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
                    and self.n_features_in_ == 1
                    and (current_r2 < self.evolution_skip_r2 or not fast_path_cv_ok)
                    and _elapsed() < effective_timeout):
                    try:
                        from classifier_fast_path import run_guided_evolution  # type: ignore

                        x_t = torch.tensor(X, dtype=torch.float32)
                        y_t = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

                        hints = dict(operator_hints)
                        hints['operators'] = set(hints.get('operators', set()))
                        hints['frequencies'] = list(hints.get('frequencies', detected_omegas or []))
                        hints['powers'] = list(hints.get('powers', []))
                        hints['has_rational'] = bool(hints.get('has_rational', False))
                        hints['has_exp_decay'] = bool(hints.get('has_exp_decay', False))
                        hints['active_terms'] = list(hints.get('active_terms', []))

                        # Blend proposer priors into hints if available
                        if proposer_payload.get("valid"):
                            proposer_priors = proposer_payload.get("operator_priors", {})
                            if proposer_priors:
                                if "operators" not in hints:
                                    hints["operators"] = set()
                                for op, prob in proposer_priors.items():
                                    if prob > 0.15:
                                        hints["operators"].add(op)

                        # Check if any proposer skeleton is ALREADY a very good fit
                        # to avoid launching evolution if we just need minor constant refinement.
                        best_cand_mse = float('inf')
                        for cand in (candidate_formulas or []):
                            if cand.get('mse', float('inf')) < best_cand_mse:
                                best_cand_mse = cand['mse']
                        
                        # Short-circuit: if a proposer skeleton is already better than fast-path 
                        # and very good, we can skip full evolution and just use it.
                        if best_cand_mse < 1e-6 and best_cand_mse < (best_mse or float('inf')):
                            print(f"  [Proposer] Rapid hit (MSE={best_cand_mse:.2e}), using skeleton directly.")
                            best_formula = candidate_formulas[0]['formula']
                            best_mse = best_cand_mse
                            blackbox_candidate_accepted = bool(
                                getattr(self, "blackbox_state_", None) is not None
                                and self.blackbox_state_.enabled
                            ) or blackbox_candidate_accepted
                            need_evolution = False 
                        else:
                            # Pass proposer uncertainty to guide beam count when available.
                            p_unc = proposer_payload.get("sequence_uncertainty", {})
                            if not isinstance(p_unc, dict):
                                p_unc = {}
                            confidence = 1.0 - p_unc.get("entropy", 0.5)
                            guided_generations = _clamp_int(
                                min(40, self.generations // 10)
                                * float(blackbox_search_plan.get("generation_multiplier", 1.0)),
                                default=min(40, self.generations // 10),
                                lo=10,
                                hi=max(10, int(self.generations)),
                            )
                            guided_population = _clamp_int(
                                min(30, self.population_size)
                                * float(blackbox_search_plan.get("population_multiplier", 1.0)),
                                default=min(30, self.population_size),
                                lo=10,
                                hi=max(10, int(self.population_size)),
                            )
                            guided_result = run_guided_evolution(
                                x_t, y_t, hints,
                                generations=guided_generations,
                                population_size=guided_population,
                                device=self.device or "cpu",
                                candidate_formulas=candidate_formulas,
                                confidence=confidence,  # New parameter
                            )

                            if guided_result and guided_result.get('formula'):
                                evo_formula = guided_result['formula']
                                evo_mse = guided_result.get('mse', float('inf'))
                    except Exception as e:
                        print(f"  [Guided evolution skipped: {e}]")

                # Fall back to raw C++ evolution
                if (evo_formula is None or evo_mse >= self.early_stop_mse) and _elapsed() < effective_timeout:
                    try:
                        X_list = [X[:, i].astype(np.float64) for i in range(self.n_features_in_)]
                        y_arr = y.astype(np.float64).flatten()
                        if candidate_formulas is None:
                            candidate_formulas = (
                                [{
                                    "formula": best_formula,
                                    "mse": best_mse or float("inf"),
                                    "complexity": self._formula_complexity(best_formula),
                                    "validation_r2": current_r2,
                                    "from_fast_path": True,
                                }]
                                if best_formula else None
                            )

                        best_refined_candidate = None
                        if candidate_formulas:
                            best_refined_candidate = min(
                                candidate_formulas,
                                key=lambda c: (
                                    _finite_float(c.get("mse"), float("inf")),
                                    _finite_float(c.get("complexity"), float("inf")),
                                ),
                            )
                            best_refined_mse = _finite_float(best_refined_candidate.get("mse"), float("inf"))
                            if best_refined_mse < best_mse:
                                best_formula = best_refined_candidate.get("formula", best_formula)
                                best_mse = best_refined_mse
                            if (
                                np.isfinite(best_refined_mse)
                                and (
                                    best_refined_mse <= self.early_stop_mse
                                    or _finite_float(best_refined_candidate.get("validation_r2"), -1.0) >= max(
                                        float(blackbox_search_plan.get("candidate_acceptance_r2", 0.985)),
                                        min(self.evolution_skip_r2, 0.999999),
                                    )
                                )
                            ):
                                blackbox_candidate_accepted = bool(
                                    getattr(self, "blackbox_state_", None) is not None
                                    and self.blackbox_state_.enabled
                                ) or blackbox_candidate_accepted
                                if getattr(self, "blackbox_state_", None) is not None and self.blackbox_state_.enabled:
                                    effective_timeout = min(effective_timeout, max(_elapsed() + 2.0, 3.0))
                                    blackbox_search_plan["population_multiplier"] = min(
                                        float(blackbox_search_plan.get("population_multiplier", 1.0)),
                                        0.50,
                                    )
                                    blackbox_search_plan["generation_multiplier"] = min(
                                        float(blackbox_search_plan.get("generation_multiplier", 1.0)),
                                        0.35,
                                    )
                                    if isinstance(self.blackbox_diagnostics_, dict):
                                        self.blackbox_diagnostics_["evolution_budget_policy"] = "tiny_accepted_candidate_probe"
                                else:
                                    need_evolution = False

                        n_runs = max(1, int(self.multi_start_runs))
                        best_cpp_result = None

                        # Combine operator priors from proposer to pass natively to C++
                        cpp_op_priors = []
                        if proposer_payload.get("valid"):
                            pp = proposer_payload.get("operator_priors", {})
                            if pp:
                                # Order: periodic, power, exp, log
                                cpp_op_priors = [
                                    pp.get("periodic", 0.8),
                                    pp.get("power", 0.08) + pp.get("int_pow", 0.0),
                                    pp.get("exp", 0.02),
                                    pp.get("log", 0.05)
                                ]

                        seed_graphs_py = []
                        try:
                            from glassbox.sr.cpp.seed_graph_builder import (
                                build_seed_graphs_from_candidates,
                            )

                            seed_graphs_py = build_seed_graphs_from_candidates(
                                candidate_formulas if candidate_formulas else (
                                    [{"formula": best_formula, "mse": best_mse}]
                                    if best_formula else None
                                ),
                                max_seeds=max(
                                    4,
                                    min(
                                        24,
                                        int(blackbox_search_plan.get("seed_budget", 10)),
                                    ),
                                ),
                            )
                        except Exception:
                            seed_graphs_py = []

                        for run_idx in range(n_runs):
                            if not need_evolution:
                                break
                            blackbox_evolution_ran = bool(
                                getattr(self, "blackbox_state_", None) is not None
                                and self.blackbox_state_.enabled
                            ) or blackbox_evolution_ran
                            remaining = max(0.0, effective_timeout - _elapsed())
                            if remaining <= 0.0:
                                break

                            # Split remaining budget across yet-to-run starts.
                            runs_left = max(1, n_runs - run_idx)
                            run_timeout = max(1, int(remaining / runs_left))

                            run_seed = -1
                            if self.random_state is not None:
                                run_seed = int(self.random_state) + run_idx * 9973

                            result = _core.run_evolution(
                                X_list=X_list,
                                y=y_arr,
                                pop_size=_clamp_int(
                                    self.population_size
                                    * float(blackbox_search_plan.get("population_multiplier", 1.0)),
                                    default=self.population_size,
                                    lo=10,
                                    hi=max(10, int(self.population_size * 3)),
                                ),
                                generations=_clamp_int(
                                    self.generations
                                    * float(blackbox_search_plan.get("generation_multiplier", 1.0)),
                                    default=self.generations,
                                    lo=10,
                                    hi=max(10, int(self.generations * 4)),
                                ),
                                early_stop_mse=self.early_stop_mse,
                                seed_omegas=detected_omegas,
                                op_priors=cpp_op_priors,
                                allowed_unary_ops=list(blackbox_search_plan.get("allowed_unary_ops", [])),
                                binary_op_priors=list(blackbox_search_plan.get("binary_op_priors", [])),
                                allowed_binary_ops=list(blackbox_search_plan.get("allowed_binary_ops", [])),
                                timeout_seconds=run_timeout,
                                p_min=_clamp_float(proposer_plan.get("p_min"), self.p_min, -8.0, 3.0),
                                p_max=_clamp_float(proposer_plan.get("p_max"), self.p_max, 1.0, 10.0),
                                use_nsga2=self.use_nsga2,
                                num_islands=self.num_islands,
                                migration_interval=self.migration_interval,
                                migration_size=self.migration_size,
                                arithmetic_temperature=self.arithmetic_temperature,
                                random_seed=run_seed,
                                acceptable_complexity=_clamp_int(
                                    blackbox_search_plan.get("acceptable_complexity"),
                                    default=15,
                                    lo=5,
                                    hi=80,
                                ),
                                early_stop_max_nodes=_clamp_int(
                                    blackbox_search_plan.get("early_stop_max_nodes"),
                                    default=50,
                                    lo=10,
                                    hi=120,
                                ),
                                multi_allowed_unary_ops=blackbox_search_plan.get("multi_allowed_unary_ops", []),
                                multi_binary_op_priors=blackbox_search_plan.get("multi_binary_op_priors", []),
                                multi_allowed_binary_ops=blackbox_search_plan.get("multi_allowed_binary_ops", []),
                                seed_graphs_py=seed_graphs_py,
                            )

                            raw_mse = result.get('best_mse', float('inf'))
                            raw_formula = result.get('formula', '')

                            if raw_mse < evo_mse:
                                evo_formula = raw_formula
                                evo_mse = raw_mse
                                best_cpp_result = result

                            if raw_mse <= self.early_stop_mse:
                                break

                        if best_cpp_result is not None:
                            # Store best C++ result for inspection
                            self.nodes_ = best_cpp_result.get('nodes', [])
                            self.output_weights_ = best_cpp_result.get('output_weights', [])
                            self.output_bias_ = best_cpp_result.get('output_bias', 0.0)
                            self.evolution_wall_time_sec_ = best_cpp_result.get('evolution_wall_time_sec')
                            self.time_to_first_exact_sec_ = best_cpp_result.get('time_to_first_exact_sec')
                            self.time_to_first_acceptable_sec_ = best_cpp_result.get('time_to_first_acceptable_sec')
                            self.generation_to_first_exact_ = best_cpp_result.get('generation_to_first_exact')
                            self.generation_to_first_acceptable_ = best_cpp_result.get('generation_to_first_acceptable')
                            self.openmp_threads_ = best_cpp_result.get('openmp_threads')
                            self.evolution_random_seed_ = best_cpp_result.get('random_seed')
                            if 'pareto_front' in best_cpp_result:
                                self.pareto_front_ = best_cpp_result['pareto_front']
                    except Exception as e:
                        print(f"  [C++ evolution error: {e}]")

                # Take evolution result if it wins under direct formula evaluation.
                if evo_formula:
                    self.evolution_candidate_formula_ = evo_formula
                    self.evolution_candidate_mse_ = evo_mse
                    if getattr(self, "blackbox_state_", None) is not None and self.blackbox_state_.enabled:
                        selection_source = self._compare_blackbox_formulas(best_formula, evo_formula, X, y)
                        if selection_source == "challenger":
                            selected_formula, selected_mse = evo_formula, self._formula_mse(evo_formula, X, y)
                        else:
                            selected_formula, selected_mse = best_formula, self._formula_mse(best_formula, X, y)
                    else:
                        selected_formula, selected_mse, selection_source = self._select_final_formula(
                            best_formula,
                            best_mse,
                            evo_formula,
                            evo_mse,
                            X,
                            y,
                        )
                    if (
                        getattr(self, "blackbox_state_", None) is not None
                        and self.blackbox_state_.enabled
                        and selection_source == "challenger"
                        and best_formula
                        and np.isfinite(best_mse)
                        and np.isfinite(evo_mse)
                        and evo_mse > 0.88 * float(best_mse)
                    ):
                        selection_source = "incumbent"
                        selected_formula = best_formula
                        selected_mse = self._formula_mse(best_formula, X, y)
                    if selection_source == "challenger":
                        best_formula = selected_formula
                        best_mse = selected_mse
                        blackbox_evolution_improved = bool(
                            getattr(self, "blackbox_state_", None) is not None
                            and self.blackbox_state_.enabled
                        ) or blackbox_evolution_improved
                    if isinstance(self.blackbox_diagnostics_, dict):
                        self.blackbox_diagnostics_["evolution_selection"] = {
                            "incumbent_formula": best_formula if selection_source != "challenger" else None,
                            "challenger_formula": evo_formula,
                            "challenger_mse": float(evo_mse) if np.isfinite(evo_mse) else None,
                            "selected": selection_source,
                        }
                    print(
                        "  [Evolution] "
                        f"candidate_mse={float(evo_mse):.6g} "
                        f"selected={selection_source} "
                        f"formula={(evo_formula or '0')[:120]}"
                    )
        elif need_evolution and _elapsed() >= effective_timeout:
            print(f"  [Timeout: skipping evolution after {_elapsed():.1f}s (budget={effective_timeout:.1f}s)]")

        # ── Stage 3: Formula Simplification & Noise Reduction ──
        if best_formula:
            if getattr(self, "blackbox_state_", None) is not None and self.blackbox_state_.enabled:
                pareto_candidates = []
                if best_formula:
                    pareto_candidates.append({
                        "formula": best_formula,
                        "mse": best_mse,
                        "complexity": self._formula_complexity(best_formula),
                        "source": "incumbent",
                    })
                for cand in candidate_formulas or []:
                    formula = str(cand.get("formula", "")).strip()
                    if formula:
                        item = dict(cand)
                        item["source"] = item.get("source", "candidate_screening")
                        pareto_candidates.append(item)
                basis_result = getattr(self, "blackbox_basis_model_", None)
                if isinstance(basis_result, dict) and basis_result.get("formula"):
                    pareto_candidates.append({
                        "formula": basis_result.get("formula"),
                        "mse": basis_result.get("mse", float("inf")),
                        "complexity": basis_result.get("complexity", self._formula_complexity(basis_result.get("formula"))),
                        "source": "basis_model",
                    })
                engineered_result = getattr(self, "blackbox_engineered_basis_model_", None)
                if isinstance(engineered_result, dict) and engineered_result.get("formula"):
                    pareto_candidates.append({
                        "formula": engineered_result.get("formula"),
                        "mse": engineered_result.get("mse", float("inf")),
                        "complexity": engineered_result.get("complexity", self._formula_complexity(engineered_result.get("formula"))),
                        "source": "engineered_basis",
                    })
                if getattr(self, "evolution_candidate_formula_", None):
                    pareto_candidates.append({
                        "formula": self.evolution_candidate_formula_,
                        "mse": getattr(self, "evolution_candidate_mse_", float("inf")),
                        "complexity": self._formula_complexity(self.evolution_candidate_formula_),
                        "source": "evolution",
                    })
                pareto_choice = self._select_blackbox_pareto_formula(pareto_candidates, X, y)
                if pareto_choice is not None:
                    best_formula = pareto_choice["formula"]
                    best_mse = pareto_choice["mse"]
                    if isinstance(self.blackbox_diagnostics_, dict):
                        self.blackbox_diagnostics_["final_pareto_selection"] = {
                            "source": pareto_choice.get("source"),
                            "validation_mse": pareto_choice.get("validation_mse"),
                            "validation_r2": pareto_choice.get("validation_r2"),
                            "complexity": pareto_choice.get("complexity"),
                            "risk_score": pareto_choice.get("risk_score"),
                            "generalization_gap": pareto_choice.get("generalization_gap"),
                            "evaluated_candidates": pareto_choice.get("evaluated_candidates"),
                            "best_raw_validation_mse": pareto_choice.get("best_raw_validation_mse"),
                        }
            best_formula = self._reduce_formula_noise(best_formula, X, y)
            best_formula = self._simplify_formula(best_formula)
            if getattr(self, "blackbox_state_", None) is not None and self.blackbox_state_.enabled:
                best_formula = formula_from_search_to_original_space(
                    best_formula,
                    self.blackbox_state_,
                )
                original_linear = getattr(self, "_blackbox_original_linear_fallback", None)
                if isinstance(original_linear, dict) and original_linear.get("formula"):
                    holdout_n = int(max(8, round(len(y_original) * 0.25)))
                    holdout_n = min(holdout_n, len(y_original) - 16)
                    tail_split = None
                    if holdout_n > 0:
                        tail_split = {
                            "X_val": X_original[-holdout_n:],
                            "y_val": y_original[-holdout_n:],
                        }
                    if tail_split is not None:
                        try:
                            current_pred = self._safe_eval_formula_array(best_formula, tail_split["X_val"])
                            linear_pred = self._safe_eval_formula_array(original_linear["formula"], tail_split["X_val"])
                            current_val_mse = float(np.mean((current_pred - tail_split["y_val"]) ** 2))
                            linear_val_mse = float(np.mean((linear_pred - tail_split["y_val"]) ** 2))
                        except Exception:
                            current_val_mse = float("inf")
                            linear_val_mse = float("inf")
                        if np.isfinite(linear_val_mse) and (
                            not np.isfinite(current_val_mse)
                            or linear_val_mse <= current_val_mse * 1.03 + 1e-12
                        ):
                            best_formula = original_linear["formula"]
                            best_mse = float(original_linear.get("mse", best_mse))
                            blackbox_candidate_accepted = True
                            if isinstance(self.blackbox_diagnostics_, dict):
                                self.blackbox_diagnostics_["original_linear_fallback_selection"] = {
                                    "selected": True,
                                    "current_tail_mse": current_val_mse,
                                    "fallback_tail_mse": linear_val_mse,
                                    "validation_r2": original_linear.get("validation_r2"),
                                    "n_terms": original_linear.get("n_terms"),
                                }
                        elif isinstance(self.blackbox_diagnostics_, dict):
                            self.blackbox_diagnostics_["original_linear_fallback_selection"] = {
                                "selected": False,
                                "current_tail_mse": current_val_mse,
                                "fallback_tail_mse": linear_val_mse,
                                "validation_r2": original_linear.get("validation_r2"),
                                "n_terms": original_linear.get("n_terms"),
                            }
                if isinstance(self.blackbox_diagnostics_, dict):
                    self.blackbox_diagnostics_["domain_failure_rate"] = self._formula_domain_failure_rate(
                        best_formula,
                        X_original,
                    )
                    selection_outcome = {
                        "candidate_screening_win": bool(blackbox_candidate_accepted and not blackbox_evolution_ran),
                        "evolution_ran": bool(blackbox_evolution_ran),
                        "evolution_win": bool(blackbox_evolution_improved),
                        "source": (
                            "candidate_screening"
                            if blackbox_candidate_accepted and not blackbox_evolution_ran
                            else ("evolution" if blackbox_evolution_improved else "incumbent_or_basis")
                        ),
                    }
                    
                    specialist_track = "incumbent path"
                    final_pareto = self.blackbox_diagnostics_.get("final_pareto_selection")
                    if isinstance(final_pareto, dict):
                        winner_source = final_pareto.get("source")
                        if winner_source == "evolution":
                            if self.has_composed_seeds_:
                                specialist_track = "composed seed + evolution"
                            else:
                                specialist_track = "incumbent path"
                        elif winner_source in ("specialist_composition", "specialist_residual_composition", "candidate_screening", "proposer", "basis_model", "engineered_basis"):
                            specialist_track = "screening only"
                        elif winner_source == "incumbent":
                            specialist_track = "incumbent path"
                    elif selection_outcome["candidate_screening_win"]:
                        specialist_track = "screening only"
                    
                    selection_outcome["specialist_track"] = specialist_track
                    self.blackbox_diagnostics_["selection_outcome"] = selection_outcome
                    self.specialist_track_ = specialist_track

            best_formula = self._run_residual_boosting(X, y, best_formula)

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

        try:
            return self._safe_eval_formula_array(self.formula_, X)
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
            from glassbox.sr.cpp import _core
            X_list = [X[:, j] for j in range(self.n_features_in_)]
            return _core.reduce_formula_noise(formula_str, X_list, y)
        except Exception:
            return formula_str
