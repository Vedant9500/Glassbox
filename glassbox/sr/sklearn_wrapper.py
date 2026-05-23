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
from glassbox.sr.blackbox_preprocessor import (
    formula_from_search_to_original_space,
    discover_blackbox_interactions,
    prepare_blackbox_search,
    remap_original_formula_to_reduced,
    state_to_dict,
)


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
        num_islands=1,
        migration_interval=25,
        migration_size=2,
        arithmetic_temperature=5.0,
        # Pipeline control
        use_fast_path=True,
        use_guided_evolution=True,
        use_simplification=True,
        classifier_path="models/curve_classifier_wide.pt",
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
        universal_proposer_path="models/universal_proposer_robust.pt",
        universal_proposer_shadow_mode="auto",
        universal_proposer_log_routing=True,
        universal_proposer_top_k=5,
        blackbox_mode="auto",
        blackbox_max_features=6,
        blackbox_feature_selection=True,
        blackbox_standardize=True,
        blackbox_min_features_to_select=5,
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
        self.enable_residual_stage = enable_residual_stage
        self.device = device
        self.skip_evolution_if_bloated = skip_evolution_if_bloated
        self.bloat_term_threshold = bloat_term_threshold

        self._universal_proposer_model = None

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
        candidates = [self.universal_proposer_path, "models/universal_proposer_robust.pt"]
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
        self.blackbox_state_ = blackbox_state
        self.blackbox_diagnostics_ = state_to_dict(blackbox_state)

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
                    operator_hints = fp_result.get('operator_hints') or {}
                    # Stash for uncertainty-coupled budget routing and candidate seeding
                    self._fp_result = fp_result
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

        effective_timeout = self._estimate_compute_budget(X, current_r2, term_count, uncertainty=_fp_uncertainty)

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
                candidate_formulas = None
                proposer_payload = (
                    self.universal_proposer_fpip_v2_
                    if isinstance(self.universal_proposer_fpip_v2_, dict)
                    else {}
                )
                proposer_plan = proposer_payload.get("search_plan", {})
                if not isinstance(proposer_plan, dict):
                    proposer_plan = {}

                # Try guided evolution (beam search) only if R² is low
                if (self.use_guided_evolution and operator_hints
                    and self.n_features_in_ == 1
                    and (current_r2 < self.evolution_skip_r2 or not fast_path_cv_ok)
                    and _elapsed() < effective_timeout):
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

                        candidate_formulas = []
                        if best_formula:
                            candidate_formulas.append({
                                "formula": best_formula,
                                "mse": best_mse or float("inf"),
                                "from_fast_path": True,
                            })

                        # Blend proposer priors into hints if available
                        if proposer_payload.get("valid"):
                            proposer_priors = proposer_payload.get("operator_priors", {})
                            if proposer_priors:
                                if "operators" not in hints:
                                    hints["operators"] = set()
                                for op, prob in proposer_priors.items():
                                    if prob > 0.15:
                                        hints["operators"].add(op)

                            for cand in proposer_payload.get("candidate_skeletons", []):
                                formula_str = cand.get("formula", "")
                                if formula_str:
                                    active_terms = [
                                        t.strip()
                                        for t in formula_str.replace("-", "+").split("+")
                                        if t.strip()
                                    ]
                                    candidate_formulas.append({
                                        "formula": formula_str,
                                        "mse": cand.get("mse", float("inf")),
                                        "score": cand.get("score", 0.0),
                                        "active_terms": active_terms,
                                        "from_proposer": True,
                                    })

                        blackbox_state = getattr(self, "blackbox_state_", None)
                        if blackbox_state is not None and getattr(blackbox_state, "interaction_terms", None):
                            for term in blackbox_state.interaction_terms[:5]:
                                seed_formula = term
                                if getattr(blackbox_state, "enabled", False):
                                    seed_formula = remap_original_formula_to_reduced(
                                        term,
                                        blackbox_state.selected_features,
                                    )
                                candidate_formulas.append({
                                    "formula": seed_formula,
                                    "mse": float("inf"),
                                    "score": 0.5,
                                    "active_terms": [seed_formula],
                                    "from_blackbox_interaction": True,
                                })

                        if not candidate_formulas:
                            candidate_formulas = None

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
                            need_evolution = False 
                        else:
                            # Pass proposer uncertainty to guide beam count when available.
                            p_unc = proposer_payload.get("sequence_uncertainty", {})
                            if not isinstance(p_unc, dict):
                                p_unc = {}
                            confidence = 1.0 - p_unc.get("entropy", 0.5)
                            guided_generations = _clamp_int(
                                min(40, self.generations // 10)
                                * _clamp_float(proposer_plan.get("generation_multiplier"), 1.0, 0.5, 4.0),
                                default=min(40, self.generations // 10),
                                lo=10,
                                hi=max(10, int(self.generations)),
                            )
                            guided_population = _clamp_int(
                                min(30, self.population_size)
                                * _clamp_float(proposer_plan.get("population_multiplier"), 1.0, 0.5, 3.0),
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
                            candidate_formulas = []
                            if best_formula:
                                candidate_formulas.append({
                                    "formula": best_formula,
                                    "mse": best_mse or float("inf"),
                                    "from_fast_path": True,
                                })
                            blackbox_state = getattr(self, "blackbox_state_", None)
                            if blackbox_state is not None and getattr(blackbox_state, "candidate_seed_formulas", None):
                                for formula in blackbox_state.candidate_seed_formulas[:16]:
                                    seed_formula = formula
                                    if getattr(blackbox_state, "enabled", False):
                                        seed_formula = remap_original_formula_to_reduced(
                                            formula,
                                            blackbox_state.selected_features,
                                        )
                                    candidate_formulas.append({
                                        "formula": seed_formula,
                                        "mse": float("inf"),
                                        "score": 0.25,
                                        "active_terms": [seed_formula],
                                        "from_blackbox_seed": True,
                                    })
                            if not candidate_formulas:
                                candidate_formulas = None

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
                                max_seeds=10,
                            )
                        except Exception:
                            seed_graphs_py = []

                        for run_idx in range(n_runs):
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
                                    * _clamp_float(proposer_plan.get("population_multiplier"), 1.0, 0.5, 3.0),
                                    default=self.population_size,
                                    lo=10,
                                    hi=max(10, int(self.population_size * 3)),
                                ),
                                generations=_clamp_int(
                                    self.generations
                                    * _clamp_float(proposer_plan.get("generation_multiplier"), 1.0, 0.5, 4.0),
                                    default=self.generations,
                                    lo=10,
                                    hi=max(10, int(self.generations * 4)),
                                ),
                                early_stop_mse=self.early_stop_mse,
                                seed_omegas=detected_omegas,
                                op_priors=cpp_op_priors,
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
                                    proposer_plan.get("acceptable_complexity"),
                                    default=15,
                                    lo=5,
                                    hi=80,
                                ),
                                early_stop_max_nodes=_clamp_int(
                                    proposer_plan.get("early_stop_max_nodes"),
                                    default=50,
                                    lo=10,
                                    hi=120,
                                ),
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

                # Take evolution result if better than fast-path
                if evo_formula and (evo_mse < best_mse or best_formula is None):
                    best_formula = evo_formula
                    best_mse = evo_mse
        elif need_evolution and _elapsed() >= effective_timeout:
            print(f"  [Timeout: skipping evolution after {_elapsed():.1f}s (budget={effective_timeout:.1f}s)]")

        # ── Stage 3: Formula Simplification & Noise Reduction ──
        if best_formula:
            best_formula = self._reduce_formula_noise(best_formula, X, y)
            best_formula = self._simplify_formula(best_formula)
            if getattr(self, "blackbox_state_", None) is not None and self.blackbox_state_.enabled:
                best_formula = formula_from_search_to_original_space(
                    best_formula,
                    self.blackbox_state_,
                )

            residual_formula = self._stage_residual_symbolic_fit(X, y, best_formula, _allow_recursion=True)
            if residual_formula:
                combined = f"({best_formula})+({residual_formula})"
                try:
                    combined_pred = self._safe_eval_formula_array(combined, X)
                    base_pred = self._safe_eval_formula_array(best_formula, X)
                    if np.mean((combined_pred - y) ** 2) < np.mean((base_pred - y) ** 2):
                        best_formula = combined
                except Exception:
                    pass

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
