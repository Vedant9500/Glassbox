"""Run the benchmark with an exactness guard applied to the pipeline output.

Guards are opt-in wrappers so the baseline stays untouched:

``pi``
    Snap frequencies that are a rational approximation of pi (22/7,
    3.13852, ...) to the exact constant inside ``sin``/``cos``.

``verifier``
    Rerank the fast-path candidate pool with the learned
    exact-vs-approximation verifier. Measured: it *hurts* (25 -> 23 exact on
    tiers 6-8, 12 formula-eval failures), so it is kept as a negative result.

``closure``
    Second-pass product/quotient closure of the pipeline's own basis when the
    primary fast path missed the exact threshold.

``compose``
    ``closure`` plus unary composition closure (``sin(g)``, ``exp(g)``,
    ``log(g)``, ... for every basis column ``g``).

``inverse``
    ``closure`` plus inverse-target composition: for an invertible outer
    function, invert the target, fit the argument with the basis, and
    re-apply. Reaches ``log(1+exp(x))``, ``1/(1+exp(-x))``, ``exp(-abs(x))``.

``both``
    ``pi`` + ``verifier``.

``all``
    Every guard above, including the verifier.

Usage:
    python experiments/exactness/run_with_guard.py --guard closure \
        --output-dir results/exactness_exp/e3_composite
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Multiples of pi that appear inside trig arguments, with the canonical text.
_PI_FAMILIES = (
    (0.25, "pi/4"),
    (1.0 / 3.0, "pi/3"),
    (0.5, "pi/2"),
    (2.0 / 3.0, "2*pi/3"),
    (1.0, "pi"),
    (1.5, "3*pi/2"),
    (2.0, "2*pi"),
)

_TRIG_ARG = re.compile(r"(sin|cos)\(\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*\*\s*x\s*\)")
_PI_TOL = 2e-3


def snap_pi(formula: str) -> str:
    """Replace near-pi trig frequencies with the exact constant."""
    if not formula or "x" not in formula:
        return formula

    def _replace(match: re.Match[str]) -> str:
        func, number = match.group(1), match.group(2)
        try:
            value = float(number)
        except ValueError:
            return match.group(0)
        if value == 0:
            return match.group(0)
        ratio = value / math.pi
        for multiple, text in _PI_FAMILIES:
            if abs(ratio - multiple) / multiple <= _PI_TOL:
                return f"{func}({text}*x)"
        return match.group(0)

    return _TRIG_ARG.sub(_replace, formula)


class VerifierReranker:
    """Score fast-path candidates with the trained exactness verifier."""

    def __init__(self, checkpoint: Path):
        import torch

        from experiments.exactness.train_verifier import candidate_features

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self._torch = torch
        self._candidate_features = candidate_features
        self.mean = np.asarray(payload["mean"], dtype=np.float32)
        self.std = np.asarray(payload["std"], dtype=np.float32)
        hidden = int(payload["hidden"])
        n_features = int(payload["n_features"])

        from experiments.exactness.train_verifier import Verifier

        self.model = Verifier(n_features, hidden=hidden)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()

    def probability(self, x: np.ndarray, y: np.ndarray, candidate: str, in_mse: float | None) -> float:
        from experiments.exactness import lib

        curve = lib.curve_features(x, y)
        if curve is None:
            return 0.5
        cand, _ = self._candidate_features(candidate, x, y, in_mse)
        feats = np.concatenate([curve, cand]).astype(np.float32)
        feats = (feats - self.mean) / self.std
        with self._torch.no_grad():
            logit = self.model(self._torch.tensor(feats).unsqueeze(0))
            return float(self._torch.sigmoid(logit).item())


def rerank_fast_path_result(
    fp_result: dict[str, Any] | None,
    x: np.ndarray,
    y: np.ndarray,
    reranker: VerifierReranker,
    *,
    penalty: float = 100.0,
) -> dict[str, Any] | None:
    """Swap in a candidate the verifier prefers, keeping in-domain fit sane."""
    if not fp_result:
        return fp_result
    pool = [c for c in (fp_result.get("candidate_formulas") or []) if isinstance(c, dict)]
    if len(pool) < 2:
        return fp_result
    x_flat = np.asarray(x, dtype=np.float64).reshape(-1)
    y_flat = np.asarray(y, dtype=np.float64).reshape(-1)

    scored = []
    for cand in pool:
        formula = str(cand.get("formula") or "").strip()
        if not formula:
            continue
        mse = cand.get("mse")
        try:
            mse_f = float(mse)
        except (TypeError, ValueError):
            mse_f = float("inf")
        if not math.isfinite(mse_f):
            continue
        prob = reranker.probability(x_flat, y_flat, formula, mse_f)
        adjusted = mse_f * (1.0 if prob >= 0.5 else penalty)
        scored.append((adjusted, mse_f, prob, formula))

    if not scored:
        return fp_result
    scored.sort(key=lambda row: row[0])
    adjusted, mse_f, prob, formula = scored[0]
    current = str(fp_result.get("formula") or "")
    if formula.replace(" ", "") == current.replace(" ", ""):
        return fp_result

    out = dict(fp_result)
    out["formula"] = formula
    out["mse"] = mse_f
    out["guard_rerank"] = {
        "previous_formula": current,
        "verifier_probability": prob,
        "candidates_scored": len(scored),
    }
    return out


def _formula_mse(formula: str | None, x: np.ndarray, y: np.ndarray) -> float | None:
    """Display-equivalent MSE of a formula on the sampled curve."""
    if not formula:
        return None
    from experiments.exactness import lib

    pred = lib.evaluate(formula, x)
    if pred is None or pred.shape != y.shape:
        return None
    mask = np.isfinite(pred) & np.isfinite(y)
    if mask.sum() < 10:
        return None
    mse = float(np.mean((pred[mask] - y[mask]) ** 2))
    return mse if math.isfinite(mse) else None


def _edge_mse(
    formula: str | None,
    x: np.ndarray,
    y: np.ndarray,
    *,
    fraction: float = 0.1,
) -> float | None:
    """MSE on the outer slices of the domain (the extrapolation test)."""
    if not formula:
        return None
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < 20:
        return None
    order = np.argsort(x)
    x_sorted, y_sorted = x[order], y[order]
    k = max(5, int(round(x_sorted.size * fraction)))
    edge = np.concatenate([np.arange(k), np.arange(x_sorted.size - k, x_sorted.size)])
    return _formula_mse(formula, x_sorted[edge], y_sorted[edge])


def _predict_operators(
    x_flat: np.ndarray, y_flat: np.ndarray, fkwargs: dict[str, Any]
) -> dict[str, float]:
    """Classifier predictions used to build the primitive basis."""
    try:
        from glassbox.curve_classifier.curve_classifier_integration import (
            predict_operators,
        )
        from glassbox.model_registry import DEFAULT_CURVE_CLASSIFIER_PATH

        return predict_operators(
            x_flat.reshape(-1, 1),
            y_flat,
            model_path=str(fkwargs.get("classifier_path") or DEFAULT_CURVE_CLASSIFIER_PATH),
            device=fkwargs.get("device"),
        )
    except Exception:
        return {}


def _accept_candidate(
    fp_result: dict[str, Any],
    x_flat: np.ndarray,
    y_flat: np.ndarray,
    candidate: dict[str, Any] | None,
    *,
    key: str,
    mse_f: float,
    max_terms: int | None = None,
) -> dict[str, Any]:
    """Accept a closure candidate only on exact-fit + edge evidence."""
    if not candidate:
        return fp_result
    try:
        cand_mse = float(candidate["mse"])
    except (TypeError, ValueError, KeyError):
        return fp_result
    if not math.isfinite(cand_mse) or cand_mse >= mse_f:
        return fp_result

    # A genuine structural identity fits to machine precision. A dense
    # surrogate can beat the incumbent inside the domain while extrapolating
    # worse (measured: OOD median 38 -> 94). Only accept near-exact closures.
    if cand_mse > 1e-11:
        return fp_result

    # A large dictionary can approximate a curve to 1e-11 with a few columns
    # without being its formula. A real closure identity needs one or two.
    if max_terms is not None and int(candidate.get("n_terms") or 0) > max_terms:
        return fp_result

    # In-domain fit alone is not evidence: a denser fit can win inside the
    # sampled range and extrapolate worse. Only accept the rewrite when it is
    # also no worse at the domain edges.
    inc_edge = _edge_mse(str(fp_result.get("formula") or ""), x_flat, y_flat)
    cand_edge = _edge_mse(candidate["formula"], x_flat, y_flat)
    if cand_edge is None:
        return fp_result
    if inc_edge is not None and math.isfinite(inc_edge):
        edge_limit = max(inc_edge * 1.25, 1e-9)
        if cand_edge > edge_limit and cand_edge > 1e-7:
            return fp_result
    elif cand_edge > 1e-3:
        # No usable incumbent at the edges, and the candidate blows up there.
        return fp_result

    out = dict(fp_result)
    out["formula"] = candidate["formula"]
    out["mse"] = cand_mse
    out[key] = {
        "previous_formula": fp_result.get("formula"),
        "previous_mse": mse_f,
        "holdout_mse": candidate.get("holdout_mse"),
        "active_terms": candidate.get("active_terms"),
        "dictionary_size": candidate.get("dictionary_size"),
        "incumbent_edge_mse": inc_edge,
        "candidate_edge_mse": cand_edge,
    }
    return out


def apply_composite_pass(
    fp_result: dict[str, Any] | None,
    x: Any,
    y: Any,
    fkwargs: dict[str, Any],
) -> dict[str, Any] | None:
    """Second-pass operator closure, only when the primary fast path missed exact."""
    if not fp_result:
        return fp_result
    try:
        mse_f = float(fp_result.get("mse"))
    except (TypeError, ValueError):
        mse_f = float("inf")
    # Escalate on anything that is not machine-precision exact. The
    # benchmark's EXACT wall (1e-6) is far looser than an identity; a
    # fast-path fit at 5e-7 is still an approximation and must not block the
    # closure pass that can replace it with the true formula.
    if math.isfinite(mse_f) and mse_f < 1e-11:
        return fp_result

    from experiments.exactness.composite_basis import fit_closure

    x_np = x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)
    y_np = y.cpu().numpy() if hasattr(y, "cpu") else np.asarray(y)
    x_flat = np.asarray(x_np, dtype=np.float64).reshape(-1)
    y_flat = np.asarray(y_np, dtype=np.float64).reshape(-1)
    if x_flat.size != y_flat.size or y_flat.size < 20:
        return fp_result

    predictions = _predict_operators(x_flat, y_flat, fkwargs)
    omegas = fkwargs.get("detected_omegas")
    candidate = fit_closure(x_flat, y_flat, predictions, omegas)
    return _accept_candidate(
        fp_result, x_flat, y_flat, candidate, key="guard_closure", mse_f=mse_f
    )


def apply_composition_pass(
    fp_result: dict[str, Any] | None,
    x: Any,
    y: Any,
    fkwargs: dict[str, Any],
) -> dict[str, Any] | None:
    """Unary composition closure: products plus ``op(g)`` for every column."""
    if not fp_result:
        return fp_result
    try:
        mse_f = float(fp_result.get("mse"))
    except (TypeError, ValueError):
        mse_f = float("inf")
    # Escalate on anything that is not machine-precision exact. The
    # benchmark's EXACT wall (1e-6) is far looser than an identity; a
    # fast-path fit at 5e-7 is still an approximation and must not block the
    # closure pass that can replace it with the true formula.
    if math.isfinite(mse_f) and mse_f < 1e-11:
        return fp_result

    from experiments.exactness.composition_closure import fit_composition

    x_np = x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)
    y_np = y.cpu().numpy() if hasattr(y, "cpu") else np.asarray(y)
    x_flat = np.asarray(x_np, dtype=np.float64).reshape(-1)
    y_flat = np.asarray(y_np, dtype=np.float64).reshape(-1)
    if x_flat.size != y_flat.size or y_flat.size < 20:
        return fp_result

    predictions = _predict_operators(x_flat, y_flat, fkwargs)
    omegas = fkwargs.get("detected_omegas")
    candidate = fit_composition(x_flat, y_flat, predictions, omegas)
    return _accept_candidate(
        fp_result,
        x_flat,
        y_flat,
        candidate,
        key="guard_composition",
        mse_f=mse_f,
        max_terms=2,
    )


def apply_inverse_pass(
    fp_result: dict[str, Any] | None,
    x: Any,
    y: Any,
    fkwargs: dict[str, Any],
) -> dict[str, Any] | None:
    """Inverse-target composition: fit the argument, re-apply the function."""
    if not fp_result:
        return fp_result
    try:
        mse_f = float(fp_result.get("mse"))
    except (TypeError, ValueError):
        mse_f = float("inf")
    # Escalate on anything that is not machine-precision exact. The
    # benchmark's EXACT wall (1e-6) is far looser than an identity; a
    # fast-path fit at 5e-7 is still an approximation and must not block the
    # closure pass that can replace it with the true formula.
    if math.isfinite(mse_f) and mse_f < 1e-11:
        return fp_result

    from experiments.exactness.inverse_composition import fit_inverse_composition

    x_np = x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)
    y_np = y.cpu().numpy() if hasattr(y, "cpu") else np.asarray(y)
    x_flat = np.asarray(x_np, dtype=np.float64).reshape(-1)
    y_flat = np.asarray(y_np, dtype=np.float64).reshape(-1)
    if x_flat.size != y_flat.size or y_flat.size < 20:
        return fp_result

    predictions = _predict_operators(x_flat, y_flat, fkwargs)
    omegas = fkwargs.get("detected_omegas")
    candidate = fit_inverse_composition(x_flat, y_flat, predictions, omegas)
    return _accept_candidate(
        fp_result,
        x_flat,
        y_flat,
        candidate,
        key="guard_inverse",
        mse_f=mse_f,
        max_terms=4,
    )


def apply_series_pass(
    fp_result: dict[str, Any] | None,
    x: Any,
    y: Any,
) -> dict[str, Any] | None:
    """Promote any series-truncated polynomial candidate to its closed form."""
    if not fp_result:
        return fp_result
    from experiments.exactness.series_rationalize import rationalize

    x_np = x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)
    y_np = y.cpu().numpy() if hasattr(y, "cpu") else np.asarray(y)
    x_flat = np.asarray(x_np, dtype=np.float64).reshape(-1)
    y_flat = np.asarray(y_np, dtype=np.float64).reshape(-1)
    if x_flat.size != y_flat.size or y_flat.size < 20:
        return fp_result

    try:
        current_mse = float(fp_result.get("mse"))
    except (TypeError, ValueError):
        current_mse = float("inf")

    formulas = [str(fp_result.get("formula") or "")]
    for cand in fp_result.get("candidate_formulas") or []:
        if isinstance(cand, dict):
            formulas.append(str(cand.get("formula") or ""))

    best: dict[str, Any] | None = None
    for formula in formulas:
        if not formula:
            continue
        rewritten = rationalize(formula, x_flat, y_flat)
        if rewritten is None:
            continue
        if rewritten["mse"] >= current_mse:
            continue
        if best is None or rewritten["mse"] < best["mse"]:
            best = rewritten

    if best is None:
        return fp_result
    out = dict(fp_result)
    out["formula"] = best["formula"]
    out["mse"] = float(best["mse"])
    out["guard_series"] = best
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--guard",
        choices=[
            "pi",
            "verifier",
            "closure",
            "compose",
            "inverse",
            "series",
            "composite",
            "both",
            "all",
        ],
        default="pi",
        help="composite is an alias for closure (kept for older commands)",
    )
    parser.add_argument(
        "--verifier",
        type=Path,
        default=Path("data/exactness_pairs/verifier.pt"),
    )
    parser.add_argument("--output-dir", type=str, default=None)
    args, remaining = parser.parse_known_args()

    # Import after sys.path setup so the script sees the repo packages.
    import scripts.benchmark_suite as bs

    use_verifier = args.guard in {"verifier", "both", "all"}
    use_composite = args.guard in {"closure", "composite", "compose", "inverse", "all"}
    use_composition = args.guard in {"compose", "all"}
    use_inverse = args.guard in {"inverse", "all"}
    use_series = args.guard in {"series", "all"}

    if use_verifier or use_composite or use_composition or use_inverse or use_series:
        reranker = VerifierReranker(args.verifier) if use_verifier else None
        original_run_fast_path = bs.run_fast_path

        def guarded_run_fast_path(x, y, *fargs, **fkwargs):
            result = original_run_fast_path(x, y, *fargs, **fkwargs)
            if use_verifier:
                result = rerank_fast_path_result(result, x, y, reranker)
            if use_composite:
                result = apply_composite_pass(result, x, y, fkwargs)
            if use_composition:
                result = apply_composition_pass(result, x, y, fkwargs)
            if use_inverse:
                result = apply_inverse_pass(result, x, y, fkwargs)
            if use_series:
                result = apply_series_pass(result, x, y)
            return result

        bs.run_fast_path = guarded_run_fast_path

    if args.guard in {"pi", "both", "all"}:
        original_postprocess = bs._postprocess_formula_for_benchmark

        def guarded_postprocess(formula, x, y):
            snapped = snap_pi(formula)
            if snapped != formula:
                return original_postprocess(snapped, x, y)
            return original_postprocess(formula, x, y)

        bs._postprocess_formula_for_benchmark = guarded_postprocess

    # Rebuild argv for the benchmark parser.
    sys.argv = [str(Path(bs.__file__).resolve()), *remaining]
    if args.output_dir:
        sys.argv.extend(["--output-dir", args.output_dir])
    bs.main()


if __name__ == "__main__":
    main()
