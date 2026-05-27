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

        return {
            "formula": refined_formula,
            "base_formula": text,
            "fit_mse": fit_mse,
            "mse": val_mse,
            "r2": float(val_r2),
            "scale": scale,
            "bias": bias,
            "complexity": complexity,
        }

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

    def _refine_candidate_formulas(self, candidate_formulas, X, y, *, max_candidates=12):
        """Refine symbolic candidates with affine scaling and holdout scoring."""
        if not candidate_formulas:
            return []
        split = self._split_blackbox_holdout(X, y, validation_fraction=0.2)
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
            })
            ranked.append(merged)

        ranked.sort(
            key=lambda c: (
                float(c.get("mse", float("inf"))),
                float(c.get("complexity", float("inf"))),
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
                float(c.get("mse", float("inf"))),
                -float(c.get("validation_r2", -float("inf"))),
                float(c.get("complexity", float("inf"))),
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
            if float(cand.get("validation_r2", -1.0)) < 0.25:
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
            timeout_multiplier = float(np.clip(timeout_multiplier, 0.85, 1.45))

        focus = "balanced"
        if candidate_strength >= candidate_acceptance_r2:
            focus = "screen_accept"
        elif screening_budget >= seed_budget + 4:
            focus = "screening"
        elif breadth_multiplier > depth_multiplier + 0.25:
            focus = "breadth"
        elif depth_multiplier > breadth_multiplier + 0.25:
            focus = "depth"

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
        }

        if proposer_plan:
            plan["generation_multiplier"] *= float(_clamp_float(proposer_plan.get("generation_multiplier"), 1.0, 0.5, 4.0))
            plan["population_multiplier"] *= float(_clamp_float(proposer_plan.get("population_multiplier"), 1.0, 0.5, 3.0))
            plan["seed_budget"] = max(plan["seed_budget"], int(proposer_plan.get("seed_budget", plan["seed_budget"])))
            plan["acceptable_complexity"] = max(
                plan["acceptable_complexity"],
                int(proposer_plan.get("acceptable_complexity", plan["acceptable_complexity"])),
            )
            plan["early_stop_max_nodes"] = max(
                plan["early_stop_max_nodes"],
                int(proposer_plan.get("early_stop_max_nodes", plan["early_stop_max_nodes"])),
            )
            plan["timeout_multiplier"] = float(np.clip(
                plan["timeout_multiplier"] * float(_clamp_float(proposer_plan.get("timeout_multiplier"), 1.0, 0.5, 3.0)),
                0.8,
                3.0,
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
        self.blackbox_search_plan_ = {}

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
                        float(c.get("validation_r2", -1.0))
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
        else:
            self.blackbox_basis_model_ = None

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
            elif basis_val_r2 >= float(blackbox_search_plan.get("candidate_shrink_r2", 0.95)):
                effective_timeout = min(effective_timeout, max(20.0, 0.4 * effective_timeout))

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
                                    float(c.get("mse", float("inf"))),
                                    float(c.get("complexity", float("inf"))),
                                ),
                            )
                            best_refined_mse = float(best_refined_candidate.get("mse", float("inf")))
                            if best_refined_mse < best_mse:
                                best_formula = best_refined_candidate.get("formula", best_formula)
                                best_mse = best_refined_mse
                            if (
                                np.isfinite(best_refined_mse)
                                and (
                                    best_refined_mse <= self.early_stop_mse
                                    or best_refined_candidate.get("validation_r2", -1.0) >= max(
                                        float(blackbox_search_plan.get("candidate_acceptance_r2", 0.985)),
                                        min(self.evolution_skip_r2, 0.999999),
                                    )
                                )
                            ):
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
                    selected_formula, selected_mse, selection_source = self._select_final_formula(
                        best_formula,
                        best_mse,
                        evo_formula,
                        evo_mse,
                        X,
                        y,
                    )
                    if selection_source == "challenger":
                        best_formula = selected_formula
                        best_mse = selected_mse
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
