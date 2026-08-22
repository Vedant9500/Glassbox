"""Phase 0: deterministic noise-handling benchmark protocol.

Goal (see ``noise_handling_phases.md`` Phase 0): make current noise behaviour
*measurable* before changing objectives. This module provides:

- Deterministic, level-pinned noise generators (clean, Gaussian 0.1/1/10% RMS,
  pink, quantization, outliers) extracted from the randomised logic in
  ``glassbox/curve_classifier/generate_curve_data.py`` but made reproducible.
- A fixed ``NOISE_TIERS`` registry.
- ``run_noise_protocol(...)``: problem x tier x seed sweep that emits the
  Phase 0 report columns required by the tracker:
  ``noise_level``, ``noise_type``, ``sample_weight_mode``, ``raw_mse``,
  ``display_mse``, ``holdout_mse``, ``clean_test_mse``, ``clean_test_r2``,
  ``clean_full_mse``, ``formula_complexity``, ``false_confidence``,
  ``false_confidence_vs_clean``, ``seed_graphs_used``, ``exact_match``,
  ``acceptable_clean``.

Clean-target columns measure structure recovery; noisy R2/MSE measure fit
to noisy labels. See ``noise_handling_audit.md``.
- ``summarize_noise_protocol(...)``: clean-vs-noisy delta table + per-tier
  failure-bucket rollup.

This phase produces measurement, not improvement. Downstream phases validate
against the JSON / Markdown tables emitted here.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import sys
import time
import warnings
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_REPO_ROOT))
if str(_SCRIPT_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_SCRIPT_DIR))

from scripts import benchmark_common as bc

# Lazy import: run_srbench_local pulls Glassbox/torch. Keep noise generators
# and pure helpers importable without the full stack.
rsl = None  # type: ignore


def _rsl():
    global rsl
    if rsl is None:
        from scripts import run_srbench_local as _mod

        rsl = _mod
    return rsl


# ---------------------------------------------------------------------------
# Deterministic noise generators
# ---------------------------------------------------------------------------
def _require_rng(seed: Any) -> np.random.RandomState:
    return np.random.RandomState(int(seed))


def noise_amplitude_scale(y: np.ndarray) -> float:
    """Signal scale used to size additive noise.

    Prefer ``std(y)`` when the target has spread. Near-constant targets have
    ``std ≈ 0``, which previously wiped out Gaussian/pink/outlier noise and made
    constants look free under noise studies. Fall back to mean absolute level,
    then ``1.0``, so a constant ``5`` at 10% noise still gets ``std ≈ 0.5``.
    """
    arr = np.asarray(y, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 1.0
    y_std = float(np.std(arr))
    y_abs = float(np.mean(np.abs(arr)))
    floor = max(1e-12, 1e-6 * max(y_abs, 1.0))
    if y_std > floor:
        return y_std
    if y_abs > 1e-12:
        return y_abs
    return 1.0


def add_gaussian_noise(
    y: np.ndarray, rms_fraction: float, *, seed: int = 0
) -> np.ndarray:
    """Additive white Gaussian noise; std = ``rms_fraction * noise_amplitude_scale(y)``."""
    rng = _require_rng(seed)
    scale = noise_amplitude_scale(y)
    return np.asarray(y, dtype=np.float64) + rng.normal(
        0.0, float(rms_fraction) * scale, size=np.asarray(y).shape
    )


def add_pink_noise(y: np.ndarray, rms_fraction: float, *, seed: int = 0) -> np.ndarray:
    """Correlated 1/f (pink) noise; std = ``rms_fraction * noise_amplitude_scale(y)``."""
    rng = _require_rng(seed)
    n = len(y)
    white = rng.standard_normal(n)
    fft_white = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0)
    freqs[0] = 1.0
    pink = np.fft.irfft(fft_white / np.sqrt(freqs), n=n)
    scale = noise_amplitude_scale(y)
    pink = pink / (float(np.std(pink)) + 1e-12) * (float(rms_fraction) * scale)
    return np.asarray(y, dtype=np.float64) + pink


def add_quantization_noise(
    y: np.ndarray, n_levels: int = 64, *, seed: int = 0
) -> np.ndarray:
    """Round to ``n_levels`` uniform steps (simulates an ADC).

    For near-constant signals (span ≈ 0) a synthetic span of
    ``2 * noise_amplitude_scale(y)`` is used so quantization is not a no-op.
    ``seed`` is accepted for API symmetry; quantization is deterministic given
    ``y`` and ``n_levels``.
    """
    del seed  # deterministic; kept for call-site symmetry
    arr = np.asarray(y, dtype=np.float64).copy()
    y_min, y_max = float(arr.min()), float(arr.max())
    span = y_max - y_min
    if span <= 1e-12:
        mid = float(np.mean(arr))
        half = noise_amplitude_scale(arr)
        y_min, y_max = mid - half, mid + half
        span = y_max - y_min
    quantized = (
        np.round((arr - y_min) / span * int(n_levels)) / int(n_levels) * span + y_min
    )
    return quantized


def add_outlier_noise(
    y: np.ndarray, fraction: float, *, magnitude_stds: float = 3.0, seed: int = 0
) -> np.ndarray:
    """Corrupt ``fraction`` of points with spikes ~``magnitude_stds`` of signal scale."""
    rng = _require_rng(seed)
    arr = np.asarray(y, dtype=np.float64).copy()
    n = len(arr)
    n_outliers = max(1, int(round(n * float(fraction))))
    indices = rng.choice(n, size=n_outliers, replace=False)
    scale = noise_amplitude_scale(arr)
    arr[indices] += rng.normal(0.0, float(magnitude_stds) * scale, size=n_outliers)
    return arr


# ---------------------------------------------------------------------------
# Fixed tier registry
# ---------------------------------------------------------------------------
# Each tier: (tier_name, noise_type, level, builder)
# level is RMS-fraction for gaussian/pink, fraction-of-points for outliers,
# n_levels for quantization. clean has level 0.0.
NOISE_TIERS: list[dict[str, Any]] = [
    {"name": "clean", "noise_type": "clean", "noise_level": 0.0},
    {"name": "gaussian_0.1pct", "noise_type": "gaussian", "noise_level": 0.001},
    {"name": "gaussian_1pct", "noise_type": "gaussian", "noise_level": 0.01},
    {"name": "gaussian_10pct", "noise_type": "gaussian", "noise_level": 0.10},
    {"name": "pink_5pct", "noise_type": "pink", "noise_level": 0.05},
    {"name": "quantization_64", "noise_type": "quantization", "noise_level": 64.0},
    {"name": "outliers_3pct", "noise_type": "outliers", "noise_level": 0.03},
]


def apply_noise_tier(y: np.ndarray, tier: dict[str, Any], *, seed: int) -> np.ndarray:
    """Apply a single tier deterministically. Clean tier returns y unchanged."""
    y = np.asarray(y, dtype=np.float64)
    ntype = str(tier.get("noise_type", "clean"))
    level = float(tier.get("noise_level", 0.0))
    if ntype == "clean" or level == 0.0:
        return y.copy()
    if ntype == "gaussian":
        return add_gaussian_noise(y, level, seed=seed)
    if ntype == "pink":
        return add_pink_noise(y, level, seed=seed)
    if ntype == "quantization":
        return add_quantization_noise(y, n_levels=int(level), seed=seed)
    if ntype == "outliers":
        return add_outlier_noise(y, fraction=level, seed=seed)
    raise ValueError(f"unknown noise_type: {ntype}")


def make_seeded_train_test_split(
    X, y, *, n_samples: int, seed: int, train_fraction: float = 0.8
):
    """Deterministic subsample + ordered train/test split (numpy-only)."""
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    rng = np.random.RandomState(int(seed))
    n_total = len(y_arr)
    if n_total > int(n_samples):
        indices = rng.choice(n_total, int(n_samples), replace=False)
    else:
        indices = rng.permutation(n_total)
    X_sel = X_arr[indices]
    y_sel = y_arr[indices]
    n_train = int(float(train_fraction) * len(y_sel))
    n_train = max(1, min(len(y_sel) - 1, n_train))
    return X_sel[:n_train], X_sel[n_train:], y_sel[:n_train], y_sel[n_train:]


def generate_ground_truth_data(problem: tuple, n_samples: int = 500, seed: int = 42):
    """Generate (X, y, formula_str) for a problem tuple without importing Glassbox."""
    name, fn, n_features, x_ranges, formula_str = problem
    rng = np.random.RandomState(int(seed))
    ranges = (
        x_ranges if len(x_ranges) == n_features else list(x_ranges) * int(n_features)
    )
    X = np.column_stack([rng.uniform(lo, hi, size=int(n_samples)) for lo, hi in ranges])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y = fn(X)
    mask = np.isfinite(y)
    if int(mask.sum()) < 20:
        return None, None, None
    return X[mask], y[mask], formula_str


# Built-in catalogue for protocol baseline / smoke without torch.
BUILTIN_PROBLEMS: dict[str, tuple] = {
    "Poly-x2": (
        "Poly-x2",
        lambda X: X[:, 0] ** 2,
        1,
        [(-5.0, 5.0)],
        "x0**2",
    ),
    "Poly-x3-x": (
        "Poly-x3-x",
        lambda X: X[:, 0] ** 3 - X[:, 0],
        1,
        [(-3.0, 3.0)],
        "x0**3 - x0",
    ),
    "Nguyen-1": (
        "Nguyen-1",
        lambda X: X[:, 0] ** 3 + X[:, 0] ** 2 + X[:, 0],
        1,
        [(-1.0, 1.0)],
        "x0**3 + x0**2 + x0",
    ),
    "Nguyen-5": (
        "Nguyen-5",
        lambda X: np.sin(X[:, 0] ** 2) * np.cos(X[:, 0]) - 1.0,
        1,
        [(-1.0, 1.0)],
        "sin(x0**2)*cos(x0) - 1",
    ),
    "Keijzer-4": (
        "Keijzer-4",
        lambda X: 0.3 * X[:, 0] * np.sin(2.0 * np.pi * X[:, 0]),
        1,
        [(-3.0, 3.0)],
        "0.3*x0*sin(2*pi*x0)",
    ),
    "Feynman-I.6.20a": (
        "Feynman-I.6.20a",
        lambda X: np.exp(-(X[:, 0] ** 2) / 2.0) / np.sqrt(2.0 * np.pi),
        1,
        [(-3.0, 3.0)],
        "exp(-x0**2/2) / sqrt(2*pi)",
    ),
}


# ---------------------------------------------------------------------------
# Phase 0 report-column helpers
# ---------------------------------------------------------------------------
def _complexity(formula: str) -> int:
    try:
        return bc.model_size(formula)
    except Exception:
        return 0


def _sample_weight_mode(estimator: Any) -> str:
    """Record weight/robust mode used at fit (user / auto soft-MAD / huber-only / none)."""
    diag = getattr(estimator, "blackbox_diagnostics_", None)
    if isinstance(diag, dict):
        sw = diag.get("sample_weight")
        if isinstance(sw, dict) and sw.get("provided"):
            src = str(sw.get("source") or "user")
            if src == "auto_soft_mad":
                return "auto_soft_mad"
            return "provided"
        robust = diag.get("blackbox_noise_robust")
        if isinstance(robust, dict) and robust.get("active"):
            reason = str(robust.get("reason") or "")
            if reason == "diffuse_noise_huber":
                return "auto_huber"
            if reason == "soft_mad_weights":
                return "auto_soft_mad"
            return "auto_soft_mad"
        loss = diag.get("loss_mode")
        if isinstance(loss, dict) and str(loss.get("mode") or "mse") != "mse":
            return f"loss_{loss.get('mode')}"
    applied = getattr(estimator, "_blackbox_noise_robust_applied_", None) or {}
    if isinstance(applied, dict) and applied.get("active"):
        if str(applied.get("reason") or "") == "diffuse_noise_huber":
            return "auto_huber"
        if getattr(estimator, "sample_weight_provided_", False):
            return "auto_soft_mad"
        return "auto_huber"
    if (
        getattr(estimator, "sample_weight_provided_", False)
        and getattr(estimator, "sample_weight_", None) is not None
    ):
        return "provided"
    return "none"


def _blackbox_diag_fields(estimator: Any, n_features: int) -> dict[str, Any]:
    """Extract blackbox / noise-routing fields for protocol rows (Phase A)."""
    diag = getattr(estimator, "blackbox_diagnostics_", None)
    if not isinstance(diag, dict):
        diag = {}
    selected = diag.get("selected_features")
    if not isinstance(selected, list):
        selected = []
    runtime = (
        diag.get("runtime_noise") if isinstance(diag.get("runtime_noise"), dict) else {}
    )
    plan = getattr(estimator, "blackbox_search_plan_", None)
    if not isinstance(plan, dict):
        plan = (
            diag.get("search_plan") if isinstance(diag.get("search_plan"), dict) else {}
        )
    noise_band = runtime.get("noise_band") or plan.get("noise_band")
    noise_pressure = plan.get("noise_pressure")
    if noise_pressure is None and isinstance(plan.get("noise_routing"), dict):
        noise_pressure = plan.get("noise_routing", {}).get("noise_pressure")
    enabled = (
        bool(diag.get("enabled"))
        if "enabled" in diag
        else bool(selected) and int(n_features) > 1
    )
    return {
        "n_features": int(n_features),
        "blackbox_enabled": enabled,
        "blackbox_reason": str(
            diag.get("reason")
            or ("unknown" if enabled else "disabled_or_low_dimensional")
        ),
        "selected_features": [int(i) for i in selected],
        "n_selected_features": int(diag.get("n_selected_features") or len(selected)),
        "feature_selection_uncertain": bool(
            diag.get("feature_selection_uncertain", False)
        ),
        "ranking_sample_weight_mode": str(
            diag.get("ranking_sample_weight_mode") or "none"
        ),
        "noise_band": str(noise_band) if noise_band is not None else None,
        "noise_pressure": _to_json_float(noise_pressure),
    }


def _empty_blackbox_fields(n_features: int = 0) -> dict[str, Any]:
    return {
        "n_features": int(n_features),
        "blackbox_enabled": False,
        "blackbox_reason": "unavailable",
        "selected_features": [],
        "n_selected_features": 0,
        "feature_selection_uncertain": False,
        "ranking_sample_weight_mode": "none",
        "noise_band": None,
        "noise_pressure": None,
    }


def _false_confidence(
    *, train_r2: float | None, test_r2: float | None, threshold: float = 0.95
) -> bool | None:
    """True when train looks great but test collapses (the false-confidence case)."""
    if train_r2 is None or test_r2 is None:
        return None
    if not (math.isfinite(float(train_r2)) and math.isfinite(float(test_r2))):
        return None
    return float(train_r2) >= float(threshold) and float(test_r2) < float(threshold)


def _seed_graphs_used(estimator: Any) -> int:
    md = getattr(estimator, "blackbox_diagnostics_", None)
    if isinstance(md, dict):
        for key in ("seed_graphs_used", "seed_graph_count", "n_seed_graphs"):
            val = md.get(key)
            if isinstance(val, (int, float)):
                return int(val)
    return 0


# ---------------------------------------------------------------------------
# Protocol runner
# ---------------------------------------------------------------------------
def default_parallel_config(
    n_jobs: int | None = None,
    omp_num_threads: int | None = None,
    *,
    cpu_count: int | None = None,
) -> tuple[int, int]:
    """Choose outer workers × per-worker OpenMP threads for protocol cells.

    Goal: ``workers * OMP_NUM_THREADS ≈ logical CPUs`` without massive
    oversubscription. On an 8c/16t laptop (e.g. Ryzen 7 7840HS) the default is
    ``4 workers × 4 OMP``.

    ``n_jobs <= 0`` or ``None`` means auto. ``omp_num_threads <= 0`` or ``None``
    means derive from remaining CPU budget.
    """
    cpus = int(cpu_count if cpu_count is not None else (os.cpu_count() or 4))
    cpus = max(1, cpus)
    if n_jobs is None or int(n_jobs) <= 0:
        # Prefer a few fat outer jobs over many tiny ones.
        if cpus >= 16 or cpus >= 8:
            jobs = 4
        elif cpus >= 4:
            jobs = 2
        else:
            jobs = 1
    else:
        jobs = max(1, int(n_jobs))
    jobs = min(jobs, cpus)

    if omp_num_threads is None or int(omp_num_threads) <= 0:
        omp = max(1, cpus // jobs)
        # SR OpenMP rarely needs more than 8 threads per fit.
        omp = min(8, omp)
    else:
        omp = max(1, int(omp_num_threads))
    return int(jobs), int(omp)


def _set_worker_thread_env(omp_num_threads: int) -> None:
    """Pin BLAS/OpenMP threads for a protocol worker process."""
    n = str(max(1, int(omp_num_threads)))
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[key] = n


@contextlib.contextmanager
def _silence_stdio(enabled: bool = True):
    """Suppress fit-time print spam (proposer/evolution banners) from workers."""
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                yield


def _cell_status(row: dict[str, Any], *, acceptable_r2: float) -> str:
    if row.get("error"):
        return "ERR"
    if row.get("exact_match"):
        return "OK"
    clean_r2 = row.get("clean_test_r2")
    if clean_r2 is None:
        clean_r2 = row.get("test_r2")
    try:
        clean_r2_f = float(clean_r2) if clean_r2 is not None else float("nan")
    except (TypeError, ValueError):
        clean_r2_f = float("nan")
    if math.isfinite(clean_r2_f) and clean_r2_f > float(acceptable_r2):
        return "MID"
    return "LOW"


def _format_eta(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "--:--"
    seconds = int(round(seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:d}h{m:02d}m"
    return f"{m:02d}m{s:02d}s"


class _ProtocolDashboard:
    """Single-pane progress view: current cell + overall completion."""

    def __init__(self, total: int, *, enabled: bool = True, detail: bool = False):
        self.total = max(0, int(total))
        self.enabled = bool(enabled)
        self.detail = bool(detail)
        self.done = 0
        self.ok = 0
        self.mid = 0
        self.low = 0
        self.err = 0
        self.started = time.time()
        self.current: str | None = None
        self.last_line: str | None = None
        self._use_tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
        if self.enabled and self.total > 0:
            self._render(prefix="START")

    def _bar(self, width: int = 24) -> str:
        if self.total <= 0:
            return "[" + ("-" * width) + "]"
        frac = min(1.0, self.done / float(self.total))
        filled = int(round(frac * width))
        return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"

    def _render(self, *, prefix: str = "RUN", last: str | None = None) -> None:
        if not self.enabled:
            return
        elapsed = time.time() - self.started
        rate = (self.done / elapsed) if elapsed > 0 and self.done else 0.0
        remaining = (self.total - self.done) / rate if rate > 0 else float("nan")
        pct = (100.0 * self.done / self.total) if self.total else 100.0
        current = self.current or "-"
        line = (
            f"{prefix} {self._bar()} {self.done}/{self.total} ({pct:5.1f}%)  "
            f"ETA {_format_eta(remaining)}  elapsed {_format_eta(elapsed)}  "
            f"OK={self.ok} MID={self.mid} LOW={self.low} ERR={self.err}"
        )
        task = f"  current: {current}"
        if last:
            task = f"  last: {last} | current: {current}"

        if self.detail:
            print(line)
            print(task)
            return

        # Compact: overwrite a 2-line dashboard on TTYs; else one line per update.
        if self._use_tty and self.last_line is not None:
            # Move up 2 lines and clear them.
            sys.stdout.write("\x1b[2A\x1b[2K\x1b[1B\x1b[2K\x1b[1A")
        print(line)
        print(task)
        sys.stdout.flush()
        self.last_line = line

    def start_cell(self, problem: Any, tier: Any, seed: Any) -> None:
        self.current = f"{problem} / {tier} / seed={seed}"
        # On a TTY, refresh so the user sees the cell that just started.
        # When redirected to a log file, only emit on finish_cell to avoid spam.
        if self._use_tty or self.detail:
            self._render(prefix="RUN ")

    def finish_cell(
        self,
        row: dict[str, Any],
        *,
        acceptable_r2: float,
        next_cell: str | None = None,
    ) -> None:
        status = _cell_status(row, acceptable_r2=acceptable_r2)
        if status == "OK":
            self.ok += 1
        elif status == "MID":
            self.mid += 1
        elif status == "ERR":
            self.err += 1
        else:
            self.low += 1
        self.done += 1
        last = f"{row.get('problem')}/{row.get('tier')}/s{row.get('seed')} → {status}"
        if next_cell is not None:
            self.current = next_cell
        elif self.done >= self.total:
            self.current = "done"
        self._render(prefix="DONE" if self.done >= self.total else "RUN ", last=last)
        if self.detail:
            print(_format_protocol_progress(row, acceptable_r2=acceptable_r2))

    def note_parallel_start(self, n_jobs: int, omp: int) -> None:
        if not self.enabled:
            return
        print(
            f"  parallel: {self.total} cells on {n_jobs} workers "
            f"(OMP_NUM_THREADS={omp} per worker)"
        )
        self._render(prefix="RUN ")


def _format_protocol_progress(row: dict[str, Any], *, acceptable_r2: float) -> str:
    ok = (
        "OK"
        if row.get("exact_match")
        else ("MID" if (row.get("test_r2") or 0.0) > acceptable_r2 else "LOW")
    )
    return (
        f"  {row.get('problem')!s:24s} {row.get('tier')!s:18s} "
        f"seed={int(row.get('seed') or 0):4d} "
        f"R2_test={row.get('test_r2')}  exact={row.get('exact_match')}  {ok}"
    )


def _run_protocol_job(payload: dict[str, Any]) -> dict[str, Any]:
    """Spawn-safe worker: rebuild estimator + problem, run one protocol cell.

    Problem tuples contain lambdas (not picklable), so jobs pass problem names
    and re-resolve via ``_select_problems`` inside the child process.
    """
    omp = payload.get("omp_num_threads")
    if omp is not None:
        _set_worker_thread_env(int(omp))

    factory_kwargs = dict(payload.get("factory_kwargs") or {})
    factory = _default_estimator_factory(**factory_kwargs)
    problem_name = str(payload["problem_name"])
    problem = _select_problems([problem_name])[0]
    tier = dict(payload["tier"])
    seed = int(payload["seed"])
    n_samples = int(payload.get("n_samples", 300))
    train_fraction = float(payload.get("train_fraction", 0.8))
    acceptable_r2 = float(payload.get("acceptable_r2", 0.9))
    silence_fit = bool(payload.get("silence_fit", True))

    with _silence_stdio(enabled=silence_fit):
        row = _run_single(
            factory,
            problem,
            tier,
            seed,
            n_samples=n_samples,
            train_fraction=train_fraction,
            acceptable_r2=acceptable_r2,
        )
    row["problem"] = problem_name
    row["tier"] = str(tier.get("name"))
    row["seed"] = seed
    return row


def run_noise_protocol(
    estimator_factory,
    problems: Sequence[tuple],
    *,
    tiers: Sequence[dict[str, Any]] | None = None,
    seeds: Iterable[int] | None = None,
    n_samples: int = 300,
    train_fraction: float = 0.8,
    verbose: bool = True,
    detail: bool = False,
    silence_fit: bool = True,
    acceptable_r2: float = 0.9,
    n_jobs: int = 1,
    omp_num_threads: int | None = None,
    factory_kwargs: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run problem x tier x seed sweep and collect Phase 0 report rows.

    ``estimator_factory`` is a zero-arg callable returning a fresh unfitted
    estimator (so each run is independent). ``problems`` follow the
    ``GROUND_TRUTH_PROBLEMS`` tuple shape used by ``run_srbench_local``.

    Progress
    --------
    By default (``verbose=True``) a compact dashboard shows:
    overall completion bar, ETA, OK/MID/LOW/ERR counts, and the current cell.
    Pass ``detail=True`` for the older one-line-per-cell log.
    Fit-time Glassbox prints are silenced when ``silence_fit=True``.

    Parallelism
    -----------
    ``n_jobs > 1`` runs independent cells in a process pool (spawn). This needs
    ``factory_kwargs`` so workers can rebuild the estimator without pickling a
    nested factory / problem lambdas. When ``factory_kwargs`` is omitted,
    falls back to sequential execution with a warning.
    """
    tiers = list(tiers) if tiers is not None else list(NOISE_TIERS)
    seeds = [
        int(s) for s in (list(seeds) if seeds is not None else [11, 23, 47, 89, 137])
    ]
    jobs, omp = default_parallel_config(n_jobs=n_jobs, omp_num_threads=omp_num_threads)

    use_pool = jobs > 1 and factory_kwargs is not None
    if jobs > 1 and factory_kwargs is None:
        warnings.warn(
            "run_noise_protocol(n_jobs>1) requires factory_kwargs for process-pool "
            "workers; falling back to sequential execution.",
            RuntimeWarning,
            stacklevel=2,
        )
        jobs = 1

    total_cells = len(problems) * len(tiers) * len(seeds)
    dash = _ProtocolDashboard(total_cells, enabled=bool(verbose), detail=bool(detail))

    if not use_pool:
        if omp_num_threads is not None and int(omp_num_threads) > 0:
            _set_worker_thread_env(int(omp_num_threads))
        rows: list[dict[str, Any]] = []
        for problem in problems:
            name = problem[0]
            for tier in tiers:
                tier_name = str(tier["name"])
                for seed in seeds:
                    dash.start_cell(name, tier_name, int(seed))
                    with _silence_stdio(enabled=bool(silence_fit)):
                        row = _run_single(
                            estimator_factory,
                            problem,
                            tier,
                            int(seed),
                            n_samples=n_samples,
                            train_fraction=train_fraction,
                            acceptable_r2=acceptable_r2,
                        )
                    row["problem"] = name
                    row["tier"] = tier_name
                    row["seed"] = int(seed)
                    rows.append(row)
                    dash.finish_cell(row, acceptable_r2=acceptable_r2)
        return rows

    # Process-pool path: one independent cell per job.
    payloads: list[dict[str, Any]] = []
    for problem in problems:
        name = str(problem[0])
        for tier in tiers:
            for seed in seeds:
                payloads.append(
                    {
                        "problem_name": name,
                        "tier": dict(tier),
                        "seed": int(seed),
                        "n_samples": int(n_samples),
                        "train_fraction": float(train_fraction),
                        "acceptable_r2": float(acceptable_r2),
                        "factory_kwargs": dict(factory_kwargs or {}),
                        "omp_num_threads": int(omp),
                        "silence_fit": bool(silence_fit),
                    }
                )

    dash.note_parallel_start(int(jobs), int(omp))

    rows_out: list[dict[str, Any] | None] = [None] * len(payloads)
    # Track in-flight labels for the dashboard (best-effort).
    pending_labels = {
        i: f"{p['problem_name']} / {p['tier'].get('name')} / seed={p['seed']}"
        for i, p in enumerate(payloads)
    }
    if pending_labels:
        dash.current = f"queued {len(pending_labels)} cells"
        dash._render(prefix="RUN ")

    ctx = get_context("spawn")
    with ProcessPoolExecutor(max_workers=int(jobs), mp_context=ctx) as pool:
        future_map = {
            pool.submit(_run_protocol_job, payload): idx
            for idx, payload in enumerate(payloads)
        }
        inflight = {future_map[f]: f for f in future_map}
        for fut in as_completed(future_map):
            idx = future_map[fut]
            row = fut.result()
            rows_out[idx] = row
            pending_labels.pop(idx, None)
            next_label = None
            if pending_labels:
                # show one remaining / still-pending cell as "current"
                next_label = next(iter(pending_labels.values()))
                if len(pending_labels) > 1:
                    next_label = f"{next_label} (+{len(pending_labels) - 1} pending)"
            dash.finish_cell(row, acceptable_r2=acceptable_r2, next_cell=next_label)

    rows = [r for r in rows_out if r is not None]
    # Stable report order: problem → tier → seed (payload order).
    return rows


def _run_single(
    estimator_factory,
    problem: tuple,
    tier: dict[str, Any],
    seed: int,
    *,
    n_samples: int,
    train_fraction: float,
    acceptable_r2: float,
) -> dict[str, Any]:
    name, fn, n_features, x_ranges, formula_str = problem
    X_clean, y_clean, true_formula = generate_ground_truth_data(
        problem, n_samples=n_samples, seed=seed
    )
    if X_clean is None:
        out = {
            "noise_type": tier.get("noise_type"),
            "noise_level": tier.get("noise_level"),
            "sample_weight_mode": "none",
            "raw_mse": None,
            "display_mse": None,
            "holdout_mse": None,
            "clean_test_mse": None,
            "clean_test_r2": None,
            "clean_full_mse": None,
            "formula_complexity": None,
            "false_confidence": None,
            "false_confidence_vs_clean": None,
            "seed_graphs_used": 0,
            "train_r2": None,
            "test_r2": None,
            "exact_match": False,
            "acceptable_clean": False,
            "discovered_formula": None,
            "true_formula": true_formula,
            "error": "bad_data",
        }
        out.update(_empty_blackbox_fields(int(n_features)))
        return out

    # Deterministic subsample + ordered train/test split (shared helper).
    # Noise is applied after the split so train/test share one noise draw on
    # the selected vector while clean labels remain available for recovery.
    X_train, X_test, y_train_clean, y_test_clean = make_seeded_train_test_split(
        X_clean, y_clean, n_samples=n_samples, seed=seed, train_fraction=train_fraction
    )
    X_sel = np.vstack([X_train, X_test])
    y_sel = np.concatenate(
        [
            np.asarray(y_train_clean, dtype=np.float64).reshape(-1),
            np.asarray(y_test_clean, dtype=np.float64).reshape(-1),
        ]
    )
    y_sel_noisy = apply_noise_tier(y_sel, tier, seed=seed)
    n_train = len(y_train_clean)
    y_train = y_sel_noisy[:n_train]
    y_test = y_sel_noisy[n_train:]

    est = estimator_factory()
    est_params = dict(est.get_params())
    est_params["random_state"] = int(seed)
    est = est.__class__(**est_params)

    try:
        est.fit(X_train, y_train)
        formula = bc.postprocess_formula(str(est.get_formula()))
    except Exception as exc:  # pragma: no cover - benchmark robustness
        out = {
            "noise_type": tier.get("noise_type"),
            "noise_level": tier.get("noise_level"),
            "sample_weight_mode": "none",
            "raw_mse": None,
            "display_mse": None,
            "holdout_mse": None,
            "clean_test_mse": None,
            "clean_test_r2": None,
            "clean_full_mse": None,
            "formula_complexity": None,
            "false_confidence": None,
            "false_confidence_vs_clean": None,
            "seed_graphs_used": 0,
            "train_r2": None,
            "test_r2": None,
            "exact_match": False,
            "acceptable_clean": False,
            "discovered_formula": None,
            "true_formula": true_formula,
            "error": str(exc)[:200],
        }
        out.update(_empty_blackbox_fields(int(n_features)))
        return out

    # Predictions / metrics.
    try:
        y_pred_train = est.predict(X_train)
        y_pred_test = est.predict(X_test)
    except Exception:
        y_pred_train = np.full_like(y_train, float(np.mean(y_train)))
        y_pred_test = np.full_like(y_test, float(np.mean(y_train)))

    train_r2 = _safe_r2(y_train, y_pred_train)
    test_r2 = _safe_r2(y_test, y_pred_test)
    raw_mse = (
        float(np.mean((y_pred_test - y_test) ** 2))
        if np.all(np.isfinite(y_pred_test))
        else None
    )
    display_mse = (
        bc.evaluate_formula_mse_on_X(formula, X_test, y_test) if formula else None
    )
    if display_mse is None or not math.isfinite(float(display_mse)):
        display_mse = raw_mse

    # Holdout currently mirrors the noisy test split; Phase 6 may add a
    # separate fidelity holdout. Clean columns below are the recovery signal.
    holdout_mse = raw_mse
    clean_test_mse = (
        float(
            np.mean(
                (
                    np.asarray(y_pred_test, dtype=np.float64)
                    - np.asarray(y_test_clean, dtype=np.float64)
                )
                ** 2
            )
        )
        if np.all(np.isfinite(y_pred_test))
        else None
    )
    clean_test_r2 = _safe_r2(y_test_clean, y_pred_test)

    # Exact-match check uses the *clean* target on the full selection so noise
    # injection does not corrupt the ground-truth equality test.
    try:
        y_pred_full = est.predict(X_sel)
        full_clean_mse = (
            float(np.mean((y_pred_full - y_sel) ** 2))
            if np.all(np.isfinite(y_pred_full))
            else float("inf")
        )
    except Exception:
        full_clean_mse = float("inf")
        y_pred_full = None
    exact_match = bool(full_clean_mse < 1e-6)
    acceptable_clean = bool(
        clean_test_r2 is not None
        and math.isfinite(float(clean_test_r2))
        and float(clean_test_r2) >= float(acceptable_r2)
    )

    row = {
        "noise_type": tier.get("noise_type"),
        "noise_level": tier.get("noise_level"),
        "sample_weight_mode": _sample_weight_mode(est),
        "raw_mse": _to_json_float(raw_mse),
        "display_mse": _to_json_float(display_mse),
        "holdout_mse": _to_json_float(holdout_mse),
        "clean_test_mse": _to_json_float(clean_test_mse),
        "clean_test_r2": _to_json_float(clean_test_r2),
        "clean_full_mse": _to_json_float(
            full_clean_mse if math.isfinite(full_clean_mse) else None
        ),
        "formula_complexity": _complexity(formula),
        "false_confidence": _false_confidence(train_r2=train_r2, test_r2=test_r2),
        "false_confidence_vs_clean": _false_confidence(
            train_r2=train_r2, test_r2=clean_test_r2
        ),
        "seed_graphs_used": _seed_graphs_used(est),
        "train_r2": _to_json_float(train_r2),
        "test_r2": _to_json_float(test_r2),
        "exact_match": exact_match,
        "acceptable_clean": acceptable_clean,
        "discovered_formula": formula,
        "true_formula": true_formula,
        "error": None,
    }
    row.update(_blackbox_diag_fields(est, int(n_features)))
    return row


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape or not np.all(np.isfinite(y_pred)):
        return None
    var = float(np.var(y_true))
    if var < 1e-15:
        return 1.0 if float(np.mean((y_pred - y_true) ** 2)) < 1e-15 else 0.0
    return float(1.0 - np.mean((y_pred - y_true) ** 2) / var)


def _to_json_float(value) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except Exception:
        return None
    return f if math.isfinite(f) else None


# ---------------------------------------------------------------------------
# Aggregation / reporting
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = (
    "noise_level",
    "noise_type",
    "sample_weight_mode",
    "raw_mse",
    "display_mse",
    "holdout_mse",
    "clean_test_mse",
    "clean_test_r2",
    "clean_full_mse",
    "formula_complexity",
    "false_confidence",
    "false_confidence_vs_clean",
    "seed_graphs_used",
    "exact_match",
    "acceptable_clean",
    # Phase A blackbox × noise measurement columns
    "n_features",
    "blackbox_enabled",
    "blackbox_reason",
    "selected_features",
    "n_selected_features",
    "feature_selection_uncertain",
    "ranking_sample_weight_mode",
    "noise_band",
    "noise_pressure",
)


def assert_row_contract(rows: Sequence[dict[str, Any]]) -> None:
    """Every row must expose the Phase 0 report columns."""
    for row in rows:
        missing = [c for c in REQUIRED_COLUMNS if c not in row]
        if missing:
            raise AssertionError(f"row missing columns {missing}: {row}")


def summarize_noise_protocol(
    rows: Sequence[dict[str, Any]], *, acceptable_r2: float = 0.9
) -> dict[str, Any]:
    """Per (problem, tier) rollup + clean-vs-noisy delta table."""
    assert_row_contract(rows)
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_key.setdefault((row["problem"], row["tier"]), []).append(row)

    cells: list[dict[str, Any]] = []
    for (problem, tier), runs in by_key.items():
        valid = [r for r in runs if r.get("test_r2") is not None]
        r2s = [float(r["test_r2"]) for r in valid]
        exact_rate = float(
            np.mean([1.0 if r.get("exact_match") else 0.0 for r in runs])
        )
        fc_rate = (
            float(
                np.mean(
                    [
                        1.0 if r.get("false_confidence") else 0.0
                        for r in runs
                        if r.get("false_confidence") is not None
                    ]
                )
            )
            if any(r.get("false_confidence") is not None for r in runs)
            else None
        )
        clean_r2s = [
            float(r["clean_test_r2"])
            for r in runs
            if r.get("clean_test_r2") is not None
        ]
        acceptable_rate = float(
            np.mean([1.0 if r.get("acceptable_clean") else 0.0 for r in runs])
        )
        cells.append(
            {
                "problem": problem,
                "tier": tier,
                "n_runs": len(runs),
                "median_test_r2": float(np.median(r2s)) if r2s else None,
                "median_clean_test_r2": float(np.median(clean_r2s))
                if clean_r2s
                else None,
                "exact_match_rate": exact_rate,
                "acceptable_clean_rate": acceptable_rate,
                "false_confidence_rate": fc_rate,
                "median_raw_mse": _median_key(valid, "raw_mse"),
                "median_display_mse": _median_key(valid, "display_mse"),
                "median_clean_test_mse": _median_key(runs, "clean_test_mse"),
                "median_formula_complexity": _median_key(valid, "formula_complexity"),
            }
        )

    # Delta table: noisy tier vs clean, per problem.
    deltas: list[dict[str, Any]] = []
    by_problem: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        by_problem.setdefault(cell["problem"], []).append(cell)
    for problem, pcells in by_problem.items():
        clean = next((c for c in pcells if c["tier"] == "clean"), None)
        if clean is None or clean["median_test_r2"] is None:
            continue
        for cell in pcells:
            if cell["tier"] == "clean" or cell["median_test_r2"] is None:
                continue
            clean_med = clean.get("median_clean_test_r2")
            cell_med = cell.get("median_clean_test_r2")
            clean_r2_delta = (
                float(cell_med) - float(clean_med)
                if clean_med is not None and cell_med is not None
                else None
            )
            deltas.append(
                {
                    "problem": problem,
                    "tier": cell["tier"],
                    "r2_delta_vs_clean": float(cell["median_test_r2"])
                    - float(clean["median_test_r2"]),
                    "clean_r2_delta_vs_clean_tier": clean_r2_delta,
                    "exact_rate_delta_vs_clean": cell["exact_match_rate"]
                    - clean["exact_match_rate"],
                    "acceptable_clean_rate": cell.get("acceptable_clean_rate"),
                    "false_confidence_rate": cell["false_confidence_rate"],
                }
            )

    return {
        "cells": cells,
        "deltas_vs_clean": deltas,
        "n_rows": len(rows),
        "acceptable_r2": acceptable_r2,
    }


def _median_key(runs: Sequence[dict[str, Any]], key: str) -> float | None:
    vals = [float(r[key]) for r in runs if r.get(key) is not None]
    return float(np.median(vals)) if vals else None


def to_markdown(summary: dict[str, Any]) -> str:
    """Render the per-tier median table as Markdown for the baseline report."""
    lines = [
        "| Problem | Tier | n | R2noisy | R2clean | Exact | Accept | FalseConf | RawMSE | CleanMSE | Complexity |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in summary["cells"]:

        def fmt(v, prec=4):
            return "-" if v is None else f"{float(v):.{prec}g}"

        lines.append(
            f"| {cell['problem']} | {cell['tier']} | {cell['n_runs']} | "
            f"{fmt(cell['median_test_r2'])} | {fmt(cell.get('median_clean_test_r2'))} | "
            f"{fmt(cell['exact_match_rate'], 2)} | {fmt(cell.get('acceptable_clean_rate'), 2)} | "
            f"{fmt(cell['false_confidence_rate'], 2)} | "
            f"{fmt(cell['median_raw_mse'])} | {fmt(cell.get('median_clean_test_mse'))} | "
            f"{fmt(cell['median_formula_complexity'], 3)} |"
        )
    return "\n".join(lines)


def write_report(rows, summary, output_dir) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "noise_protocol_rows.json"
    summary_path = output_dir / "noise_protocol_summary.json"
    md_path = output_dir / "noise_protocol_report.md"
    with rows_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=_json_default)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Phase 0 — Noise Protocol Baseline\n\n")
        f.write(to_markdown(summary))
        f.write("\n\n## Clean-vs-Noisy Deltas\n\n")
        f.write(
            "| Problem | Tier | R2noisy delta | R2clean delta | Exact delta | Accept | FalseConf |\n"
            "|---|---|---:|---:|---:|---:|---:|\n"
        )
        for d in summary["deltas_vs_clean"]:
            clean_d = d.get("clean_r2_delta_vs_clean_tier")
            clean_s = f"{clean_d:.4f}" if clean_d is not None else "-"
            acc = d.get("acceptable_clean_rate")
            acc_s = f"{acc:.2f}" if acc is not None else "-"
            f.write(
                f"| {d['problem']} | {d['tier']} | {d['r2_delta_vs_clean']:.4f} | "
                f"{clean_s} | {d['exact_rate_delta_vs_clean']:.2f} | {acc_s} | "
                f"{d['false_confidence_rate']} |\n"
            )
    return {"rows": rows_path, "summary": summary_path, "markdown": md_path}


# ---------------------------------------------------------------------------
# Phase E — multi-var ablation table (release notes)
# ---------------------------------------------------------------------------
# Default ablations for blackbox × noise release comparisons. Keep budgets
# identical across these; only estimator knobs change.
DEFAULT_BLACKBOX_RELEASE_ABLATIONS: tuple[str, ...] = (
    "full",
    "no_weights",
    "no_robust_loss",
)


def build_ablation_table(
    rows_by_ablation: dict[str, Sequence[dict[str, Any]]],
    *,
    baseline: str = "full",
) -> dict[str, Any]:
    """Compare multi-var noise protocol ablations for release notes.

    ``rows_by_ablation`` maps ablation name -> protocol rows (same problems /
    tiers / seeds / budgets). Metrics are clean-recovery first (R2clean /
    Accept / Exact), never noisy-label R² alone.
    """
    if not rows_by_ablation:
        raise ValueError("rows_by_ablation is empty")
    if baseline not in rows_by_ablation:
        raise ValueError(
            f"baseline ablation {baseline!r} missing; have {sorted(rows_by_ablation)}"
        )

    summaries: dict[str, dict[str, Any]] = {}
    for name, rows in rows_by_ablation.items():
        summary = summarize_noise_protocol(rows)
        summary["ablation"] = str(name)
        summaries[str(name)] = summary

    # Per (problem, tier, ablation) cell table with delta vs baseline.
    baseline_cells = {
        (c["problem"], c["tier"]): c for c in summaries[baseline]["cells"]
    }
    comparison_rows: list[dict[str, Any]] = []
    for name, summary in summaries.items():
        for cell in summary["cells"]:
            key = (cell["problem"], cell["tier"])
            base = baseline_cells.get(key)
            row = {
                "ablation": name,
                "problem": cell["problem"],
                "tier": cell["tier"],
                "n_runs": cell["n_runs"],
                "median_clean_test_r2": cell.get("median_clean_test_r2"),
                "acceptable_clean_rate": cell.get("acceptable_clean_rate"),
                "exact_match_rate": cell.get("exact_match_rate"),
                "median_test_r2": cell.get("median_test_r2"),
                "median_formula_complexity": cell.get("median_formula_complexity"),
                "false_confidence_rate": cell.get("false_confidence_rate"),
                "is_baseline": name == baseline,
            }
            if base is not None and name != baseline:
                for metric in (
                    "median_clean_test_r2",
                    "acceptable_clean_rate",
                    "exact_match_rate",
                    "median_test_r2",
                    "median_formula_complexity",
                ):
                    cur = cell.get(metric)
                    ref = base.get(metric)
                    if cur is not None and ref is not None:
                        row[f"delta_{metric}_vs_{baseline}"] = float(cur) - float(ref)
                    else:
                        row[f"delta_{metric}_vs_{baseline}"] = None
            comparison_rows.append(row)

    # Aggregate headline: mean Accept/R2clean on non-clean tiers per ablation.
    headlines: list[dict[str, Any]] = []
    for name, summary in summaries.items():
        noisy_cells = [c for c in summary["cells"] if c.get("tier") != "clean"]
        if not noisy_cells:
            noisy_cells = list(summary["cells"])

        def _mean_key(cells, key):
            vals = [float(c[key]) for c in cells if c.get(key) is not None]
            return float(np.mean(vals)) if vals else None

        headlines.append(
            {
                "ablation": name,
                "n_cells": len(summary["cells"]),
                "mean_clean_test_r2_noisy_tiers": _mean_key(
                    noisy_cells, "median_clean_test_r2"
                ),
                "mean_acceptable_clean_rate_noisy_tiers": _mean_key(
                    noisy_cells, "acceptable_clean_rate"
                ),
                "mean_exact_match_rate_noisy_tiers": _mean_key(
                    noisy_cells, "exact_match_rate"
                ),
                "mean_formula_complexity_noisy_tiers": _mean_key(
                    noisy_cells, "median_formula_complexity"
                ),
                "is_baseline": name == baseline,
            }
        )

    # Ensure baseline is first, then attach deltas.
    headlines.sort(key=lambda h: (0 if h["ablation"] == baseline else 1, h["ablation"]))
    base_head = next(h for h in headlines if h["ablation"] == baseline)
    for head in headlines:
        if head["ablation"] == baseline:
            continue
        for metric in (
            "mean_clean_test_r2_noisy_tiers",
            "mean_acceptable_clean_rate_noisy_tiers",
            "mean_exact_match_rate_noisy_tiers",
            "mean_formula_complexity_noisy_tiers",
        ):
            cur = head.get(metric)
            ref = base_head.get(metric)
            if cur is not None and ref is not None:
                head[f"delta_{metric}_vs_{baseline}"] = float(cur) - float(ref)
            else:
                head[f"delta_{metric}_vs_{baseline}"] = None

    return {
        "baseline": baseline,
        "ablations": list(summaries.keys()),
        "summaries": summaries,
        "comparison_rows": comparison_rows,
        "headlines": headlines,
        "n_ablations": len(summaries),
    }


def ablation_table_to_markdown(table: dict[str, Any]) -> str:
    """Render Phase E multi-var ablation headlines + per-cell deltas as Markdown."""
    baseline = table.get("baseline", "full")
    lines = [
        "# Blackbox × Noise — Ablation Table (Phase E)",
        "",
        f"Baseline ablation: `{baseline}`",
        "",
        "## Headline (noisy tiers, clean recovery)",
        "",
        "| Ablation | R2clean | Accept | Exact | Complexity | ΔR2clean | ΔAccept |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    def fmt(v, prec=4):
        return "-" if v is None else f"{float(v):.{prec}g}"

    for h in table.get("headlines", []):
        d_r2 = h.get(f"delta_mean_clean_test_r2_noisy_tiers_vs_{baseline}")
        d_acc = h.get(f"delta_mean_acceptable_clean_rate_noisy_tiers_vs_{baseline}")
        lines.append(
            f"| {h['ablation']} | {fmt(h.get('mean_clean_test_r2_noisy_tiers'))} | "
            f"{fmt(h.get('mean_acceptable_clean_rate_noisy_tiers'), 2)} | "
            f"{fmt(h.get('mean_exact_match_rate_noisy_tiers'), 2)} | "
            f"{fmt(h.get('mean_formula_complexity_noisy_tiers'), 3)} | "
            f"{fmt(d_r2)} | {fmt(d_acc, 2)} |"
        )

    lines.extend(
        [
            "",
            "## Per problem × tier",
            "",
            "| Ablation | Problem | Tier | R2clean | Accept | Exact | Complexity |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in table.get("comparison_rows", []):
        lines.append(
            f"| {row['ablation']} | {row['problem']} | {row['tier']} | "
            f"{fmt(row.get('median_clean_test_r2'))} | "
            f"{fmt(row.get('acceptable_clean_rate'), 2)} | "
            f"{fmt(row.get('exact_match_rate'), 2)} | "
            f"{fmt(row.get('median_formula_complexity'), 3)} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_ablation_report(table: dict[str, Any], output_dir) -> dict[str, Path]:
    """Write multi-var ablation JSON + Markdown for release notes."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "noise_protocol_ablation_table.json"
    md_path = output_dir / "noise_protocol_ablation_table.md"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(table, f, indent=2, default=_json_default)
    with md_path.open("w", encoding="utf-8") as f:
        f.write(ablation_table_to_markdown(table))
    return {"ablation_json": json_path, "ablation_markdown": md_path}


# ---------------------------------------------------------------------------
# Phase E+ — multi-seed publishable tables (release freeze)
# ---------------------------------------------------------------------------
# Default seed set for publishable multi-var blackbox × noise claims.
# Seed 11 is the locked single-seed lock; 7/23/42 expand variance coverage.
DEFAULT_PUBLISH_SEEDS: tuple[int, ...] = (11, 7, 23, 42)
DEFAULT_PUBLISH_TIERS: tuple[str, ...] = ("clean", "outliers_3pct")


def build_publish_table(
    rows: Sequence[dict[str, Any]],
    *,
    seeds: Sequence[int] | None = None,
    min_seeds: int = 2,
) -> dict[str, Any]:
    """Build a multi-seed publishable recovery table from protocol rows.

    Clean-recovery first: Exact / Accept / R2clean rates with per-seed
    visibility. Suitable for release notes — never use noisy-label R² alone.
    """
    assert_row_contract(rows)
    seed_list = [
        int(s)
        for s in (
            seeds
            if seeds is not None
            else sorted({int(r["seed"]) for r in rows if r.get("seed") is not None})
        )
    ]
    if not seed_list:
        seed_list = list(DEFAULT_PUBLISH_SEEDS)

    summary = summarize_noise_protocol(rows)
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_key.setdefault((row["problem"], row["tier"]), []).append(row)

    cells: list[dict[str, Any]] = []
    for (problem, tier), runs in sorted(by_key.items()):
        seed_exact = {}
        seed_accept = {}
        seed_r2clean = {}
        seed_formula = {}
        for r in runs:
            s = int(r["seed"])
            seed_exact[s] = bool(r.get("exact_match"))
            seed_accept[s] = bool(r.get("acceptable_clean"))
            if r.get("clean_test_r2") is not None:
                seed_r2clean[s] = float(r["clean_test_r2"])
            if r.get("formula") is not None:
                seed_formula[s] = str(r.get("formula") or "")[:120]
        n_seeds = len({int(r["seed"]) for r in runs})
        clean_r2s = [
            float(r["clean_test_r2"])
            for r in runs
            if r.get("clean_test_r2") is not None
        ]
        cells.append(
            {
                "problem": problem,
                "tier": tier,
                "n_runs": len(runs),
                "n_seeds": n_seeds,
                "seeds": sorted({int(r["seed"]) for r in runs}),
                "exact_match_rate": float(
                    np.mean([1.0 if r.get("exact_match") else 0.0 for r in runs])
                ),
                "acceptable_clean_rate": float(
                    np.mean([1.0 if r.get("acceptable_clean") else 0.0 for r in runs])
                ),
                "median_clean_test_r2": float(np.median(clean_r2s))
                if clean_r2s
                else None,
                "mean_clean_test_r2": float(np.mean(clean_r2s)) if clean_r2s else None,
                "seed_exact": seed_exact,
                "seed_accept": seed_accept,
                "seed_r2clean": seed_r2clean,
                "seed_formula": seed_formula,
                "median_formula_complexity": _median_key(runs, "formula_complexity"),
                "median_clean_full_mse": _median_key(runs, "clean_full_mse"),
                "publishable_seed_coverage": bool(n_seeds >= int(min_seeds)),
            }
        )

    # Headline: clean tiers vs outlier tiers across all multi-var problems.
    clean_cells = [c for c in cells if c["tier"] == "clean"]
    outlier_cells = [c for c in cells if "outlier" in str(c["tier"]).lower()]
    if not outlier_cells:
        outlier_cells = [c for c in cells if c["tier"] != "clean"]

    def _mean_cells(cs, key):
        vals = [float(c[key]) for c in cs if c.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    headlines = {
        "clean": {
            "n_cells": len(clean_cells),
            "mean_exact_match_rate": _mean_cells(clean_cells, "exact_match_rate"),
            "mean_acceptable_clean_rate": _mean_cells(
                clean_cells, "acceptable_clean_rate"
            ),
            "mean_median_clean_test_r2": _mean_cells(
                clean_cells, "median_clean_test_r2"
            ),
        },
        "outliers": {
            "n_cells": len(outlier_cells),
            "mean_exact_match_rate": _mean_cells(outlier_cells, "exact_match_rate"),
            "mean_acceptable_clean_rate": _mean_cells(
                outlier_cells, "acceptable_clean_rate"
            ),
            "mean_median_clean_test_r2": _mean_cells(
                outlier_cells, "median_clean_test_r2"
            ),
        },
    }

    n_seed_obs = max((c["n_seeds"] for c in cells), default=0)
    return {
        "seeds": seed_list,
        "min_seeds": int(min_seeds),
        "n_rows": len(rows),
        "n_cells": len(cells),
        "seed_coverage_ok": bool(n_seed_obs >= int(min_seeds)),
        "cells": cells,
        "headlines": headlines,
        "summary": summary,
        "contract": {
            "exact_definition": "clean_full_mse < 1e-6",
            "acceptable_definition": "clean_test_r2 >= 0.9",
            "metrics_are_clean_recovery": True,
            "do_not_use_suite_noisy_exact": True,
        },
    }


def publish_table_to_markdown(table: dict[str, Any]) -> str:
    """Render multi-seed publishable recovery table as Markdown for release notes."""
    seeds = table.get("seeds") or []
    lines = [
        "# Blackbox × Noise — Multi-Seed Publish Table (Phase E+)",
        "",
        f"Seeds: `{', '.join(str(s) for s in seeds)}`",
        f"Seed coverage OK (≥{table.get('min_seeds', 2)}): "
        f"**{bool(table.get('seed_coverage_ok'))}**",
        "",
        "## Headline",
        "",
        "| Bucket | Exact | Accept | R2clean |",
        "|---|---:|---:|---:|",
    ]

    def fmt(v, prec=4):
        return "-" if v is None else f"{float(v):.{prec}g}"

    for bucket in ("clean", "outliers"):
        h = (table.get("headlines") or {}).get(bucket) or {}
        lines.append(
            f"| {bucket} | {fmt(h.get('mean_exact_match_rate'), 2)} | "
            f"{fmt(h.get('mean_acceptable_clean_rate'), 2)} | "
            f"{fmt(h.get('mean_median_clean_test_r2'))} |"
        )

    lines.extend(
        [
            "",
            "## Per problem × tier (multi-seed rates)",
            "",
            "| Problem | Tier | n_seeds | Exact | Accept | R2clean | CleanFullMSE | Complexity |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for cell in table.get("cells", []):
        lines.append(
            f"| {cell['problem']} | {cell['tier']} | {cell['n_seeds']} | "
            f"{fmt(cell.get('exact_match_rate'), 2)} | "
            f"{fmt(cell.get('acceptable_clean_rate'), 2)} | "
            f"{fmt(cell.get('median_clean_test_r2'))} | "
            f"{fmt(cell.get('median_clean_full_mse'))} | "
            f"{fmt(cell.get('median_formula_complexity'), 3)} |"
        )

    lines.extend(
        [
            "",
            "## Per-seed Exact matrix",
            "",
        ]
    )
    # Build seed columns from observed seeds in cells
    obs_seeds = sorted(
        {int(s) for cell in table.get("cells", []) for s in (cell.get("seeds") or [])}
    )
    if obs_seeds:
        header = "| Problem | Tier | " + " | ".join(f"s{s}" for s in obs_seeds) + " |"
        sep = "|---|---|" + "|".join(["---:" for _ in obs_seeds]) + "|"
        lines.append(header)
        lines.append(sep)
        for cell in table.get("cells", []):
            seed_exact = cell.get("seed_exact") or {}
            bits = []
            for s in obs_seeds:
                v = seed_exact.get(s)
                if v is None:
                    bits.append("-")
                else:
                    bits.append("1" if v else "0")
            lines.append(
                f"| {cell['problem']} | {cell['tier']} | " + " | ".join(bits) + " |"
            )
    lines.append("")
    lines.append(
        "_Contract: Exact = clean_full_mse < 1e-6; Accept = clean_test_r2 ≥ 0.9. "
        "Do not cite suite noisy EXACT% as structure recovery._"
    )
    lines.append("")
    return "\n".join(lines)


def write_publish_report(table: dict[str, Any], output_dir) -> dict[str, Path]:
    """Write multi-seed publish JSON + Markdown for release freeze."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "noise_protocol_publish_table.json"
    md_path = output_dir / "noise_protocol_publish_table.md"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(table, f, indent=2, default=_json_default)
    with md_path.open("w", encoding="utf-8") as f:
        f.write(publish_table_to_markdown(table))
    return {"publish_json": json_path, "publish_markdown": md_path}


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


# ---------------------------------------------------------------------------
# CLI — multi-tier protocol baseline
# ---------------------------------------------------------------------------
DEFAULT_BASELINE_PROBLEMS = (
    "Poly-x2",
    "Poly-x3-x",
    "Nguyen-1",
    "Nguyen-5",
    "Keijzer-4",
    "Feynman-I.6.20a",
)

# Multi-feature problems that exercise the real blackbox path (n_features > 1).
# Vladislavleva-4 (5-D) is required for top-k ranking under default
# blackbox_min_features_to_select=5; Pagie/Feynman exercise blackbox-on keep-all.
DEFAULT_BLACKBOX_PROBLEMS = (
    "Pagie-1",
    "Feynman-I.9.18",
    "Vladislavleva-4",
)


def _select_problems(names: Sequence[str] | None = None):
    """Pick ground-truth problems by name (default: easy multi-tier baseline set).

    Built-in problems are available without torch. Additional names fall back to
    ``run_srbench_local.GROUND_TRUTH_PROBLEMS`` when that module can be imported.
    """
    catalogue = dict(BUILTIN_PROBLEMS)
    try:
        catalogue.update({p[0]: p for p in _rsl().GROUND_TRUTH_PROBLEMS})
    except Exception:
        # torch/Glassbox unavailable — built-ins only
        pass
    chosen = list(names) if names else list(DEFAULT_BASELINE_PROBLEMS)
    problems = []
    missing = []
    for name in chosen:
        if name in catalogue:
            problems.append(catalogue[name])
        else:
            missing.append(name)
    if missing:
        raise ValueError(f"unknown problem names: {missing}")
    return problems


# Named ablations for release-gate comparisons (Phase 8).
# Each mutates GlassboxRegressor kwargs; budgets must stay comparable.
ABLATION_PRESETS: dict[str, dict[str, Any]] = {
    "full": {},
    # Disable user weights AND auto soft-MAD / robust path so evolution stays unweighted.
    "no_weights": {
        "_force_no_sample_weight": True,
        "blackbox_noise_robust": False,
        "loss_mode": "mse",
    },
    "no_robust_loss": {"loss_mode": "mse"},
    "no_units": {"input_units": None, "output_units": None, "unit_mode": "off"},
    "no_cv_guard": {"cv_skip_guard_enabled": False},
    "no_uncertainty_routing": {"adaptive_compute_budget": False},
    "no_noise_pruning": {
        # Disable residual stage + fidelity-sensitive residual boosting.
        "enable_residual_stage": False,
    },
}


def _default_estimator_factory(
    *,
    generations: int = 40,
    population_size: int = 60,
    timeout: float = 45.0,
    multi_start_runs: int = 1,
    allow_stub: bool = False,
    ablation: str = "full",
    extra_params: dict[str, Any] | None = None,
    blackbox_protocol: bool = False,
):
    try:
        from glassbox.sr.sklearn_wrapper import GlassboxRegressor
    except Exception:
        if not allow_stub:
            raise
        GlassboxRegressor = None  # type: ignore

    ablation_key = str(ablation or "full").strip().lower()
    if ablation_key not in ABLATION_PRESETS:
        raise ValueError(
            f"unknown ablation {ablation!r}; choose from {sorted(ABLATION_PRESETS)}"
        )
    ablation_params = dict(ABLATION_PRESETS[ablation_key])
    force_no_sw = bool(ablation_params.pop("_force_no_sample_weight", False))
    if extra_params:
        ablation_params.update(extra_params)

    if GlassboxRegressor is not None:

        def factory():
            kwargs = dict(
                random_state=0,
                generations=int(generations),
                population_size=int(population_size),
                timeout=float(timeout),
                multi_start_runs=int(multi_start_runs),
                use_fast_path=True,
                use_guided_evolution=True,
                blackbox_mode=True,
                blackbox_feature_selection=True,
                # Protocol dashboard owns the console; keep fit internals quiet.
                universal_proposer_log_routing=False,
            )
            if blackbox_protocol:
                # Ensure multi-var problems can actually drop features in ranking.
                kwargs.setdefault("blackbox_min_features_to_select", 2)
                kwargs.setdefault("blackbox_max_features", 4)
                kwargs.setdefault("blackbox_standardize", True)
            kwargs.update(ablation_params)
            est = GlassboxRegressor(**kwargs)
            if force_no_sw:
                # Wrap fit so protocol cannot inject sample_weight later.
                _orig_fit = est.fit

                def _fit_no_weight(X, y, sample_weight=None):
                    return _orig_fit(X, y, sample_weight=None)

                est.fit = _fit_no_weight  # type: ignore[method-assign]
            return est

        return factory

    # Minimal mean predictor for tooling smoke when Glassbox/torch is absent.
    class _StubRegressor:
        def __init__(self, **kwargs):
            self.params = kwargs
            self.blackbox_diagnostics_ = {"sample_weight": {"provided": False}}
            self._mean = 0.0

        def get_params(self):
            return dict(self.params)

        def fit(self, X, y):
            self._mean = float(np.mean(y))
            return self

        def predict(self, X):
            return np.full(len(X), self._mean, dtype=np.float64)

        def get_formula(self):
            return f"{self._mean:.6g}"

    def factory():
        return _StubRegressor(
            random_state=0,
            generations=int(generations),
            population_size=int(population_size),
            timeout=float(timeout),
            multi_start_runs=int(multi_start_runs),
        )

    return factory


def main(argv: Sequence[str] | None = None) -> int:
    """Run the deterministic multi-tier noise protocol and write baseline artifacts."""
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(
        description=(
            "Phase 0 multi-tier noise protocol baseline. "
            "Reports clean recovery metrics separate from noisy fit metrics "
            "(see noise_handling_audit.md)."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(_REPO_ROOT / "results" / "noise_protocol_baseline"),
        help="Directory for noise_protocol_*.json / .md artifacts",
    )
    parser.add_argument(
        "--problems",
        type=str,
        default=",".join(DEFAULT_BASELINE_PROBLEMS),
        help="Comma-separated ground-truth problem names",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="11,23,47,89,137",
        help="Comma-separated integer seeds (Phase 0 default: 5 seeds)",
    )
    parser.add_argument(
        "--tiers",
        type=str,
        default="all",
        help="Comma-separated tier names, or 'all' for NOISE_TIERS",
    )
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--population-size", type=int, default=60)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny run: 2 problems, clean+gaussian_10pct, 1 seed, small budget",
    )
    parser.add_argument(
        "--blackbox",
        action="store_true",
        help=(
            "Multi-feature blackbox × noise protocol: use DEFAULT_BLACKBOX_PROBLEMS "
            "and lower min_features_to_select so ranking can drop features"
        ),
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default="full",
        help=(
            "Estimator ablation preset for release-gate comparisons: "
            + ", ".join(sorted(ABLATION_PRESETS))
            + ". Use --ablation-table to run several and emit a comparison."
        ),
    )
    parser.add_argument(
        "--ablation-table",
        action="store_true",
        help=(
            "Phase E: run DEFAULT_BLACKBOX_RELEASE_ABLATIONS (or --ablations) "
            "with identical budgets and write multi-var ablation table for "
            "release notes. Implies multi-feature blackbox protocol defaults "
            "when --blackbox is also set (recommended)."
        ),
    )
    parser.add_argument(
        "--ablations",
        type=str,
        default=",".join(DEFAULT_BLACKBOX_RELEASE_ABLATIONS),
        help=(
            "Comma-separated ablation presets for --ablation-table "
            f"(default: {','.join(DEFAULT_BLACKBOX_RELEASE_ABLATIONS)})"
        ),
    )
    parser.add_argument(
        "--publish-table",
        action="store_true",
        help=(
            "Phase E+: emit multi-seed publishable recovery table "
            "(Exact/Accept/R2clean + per-seed Exact matrix) for release freeze. "
            "Recommended with --blackbox --seeds 11,7,23,42 --tiers clean,outliers_3pct."
        ),
    )
    parser.add_argument(
        "--publish-seeds",
        action="store_true",
        help=(
            "Use DEFAULT_PUBLISH_SEEDS "
            f"({','.join(str(s) for s in DEFAULT_PUBLISH_SEEDS)}) "
            "unless --seeds is explicitly set for a non-default list."
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help=(
            "Parallel process-pool workers for independent protocol cells "
            "(problem×tier×seed). 0/negative = auto from CPU count "
            "(default on 16-thread CPUs: 4). Use 1 for sequential."
        ),
    )
    parser.add_argument(
        "--omp-num-threads",
        type=int,
        default=0,
        help=(
            "OpenMP/BLAS threads per worker. 0/negative = auto "
            "(≈ cpu_count / jobs, capped at 8). Recommended with --jobs 4: 4."
        ),
    )
    parser.add_argument("--quiet", action="store_true", help="No progress output")
    parser.add_argument(
        "--detail",
        action="store_true",
        help="Print one result line per cell (old verbose log) instead of compact dashboard",
    )
    parser.add_argument(
        "--show-fit-logs",
        action="store_true",
        help="Do not silence Glassbox fit-time prints (debug only; very noisy)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    n_jobs, omp_threads = default_parallel_config(
        n_jobs=int(args.jobs),
        omp_num_threads=int(args.omp_num_threads),
    )
    # Sequential path still benefits from an explicit OMP pin.
    if n_jobs <= 1:
        _set_worker_thread_env(int(omp_threads))

    if args.smoke and args.blackbox:
        problem_names = list(DEFAULT_BLACKBOX_PROBLEMS[:2])
        seeds = [11]
        tier_names = ["clean", "gaussian_10pct", "outliers_3pct"]
        generations, pop, timeout, n_samples = 15, 30, 25.0, 120
    elif args.smoke:
        problem_names = list(DEFAULT_BASELINE_PROBLEMS[:2])
        seeds = [11]
        tier_names = ["clean", "gaussian_10pct"]
        generations, pop, timeout, n_samples = 15, 30, 20.0, 120
    else:
        if args.blackbox and args.problems == ",".join(DEFAULT_BASELINE_PROBLEMS):
            problem_names = list(DEFAULT_BLACKBOX_PROBLEMS)
        else:
            problem_names = [p.strip() for p in args.problems.split(",") if p.strip()]
        if args.publish_seeds and args.seeds == "11,23,47,89,137":
            seeds = list(DEFAULT_PUBLISH_SEEDS)
        else:
            seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        if args.tiers.strip().lower() == "all":
            # Publish-table freeze focuses on clean + outliers by default.
            if args.publish_table or args.publish_seeds:
                tier_names = list(DEFAULT_PUBLISH_TIERS)
            else:
                tier_names = [t["name"] for t in NOISE_TIERS]
        else:
            tier_names = [t.strip() for t in args.tiers.split(",") if t.strip()]
        generations = int(args.generations)
        pop = int(args.population_size)
        timeout = float(args.timeout)
        n_samples = int(args.n_samples)

    # Phase E ablation-table defaults: multi-var + outliers, modest budget.
    if args.ablation_table and not args.smoke:
        if args.blackbox and args.problems == ",".join(DEFAULT_BASELINE_PROBLEMS):
            problem_names = list(DEFAULT_BLACKBOX_PROBLEMS)
        if args.tiers.strip().lower() == "all" and not args.smoke:
            # Keep release ablation tables focused (not full tier matrix).
            tier_names = ["clean", "outliers_3pct", "gaussian_10pct"]

    tier_by_name = {t["name"]: t for t in NOISE_TIERS}
    tiers = []
    for name in tier_names:
        if name not in tier_by_name:
            raise SystemExit(f"unknown tier: {name}")
        tiers.append(tier_by_name[name])

    problems = _select_problems(problem_names)

    def _annotate_rows(rows, ablation_name: str) -> list[dict[str, Any]]:
        annotated = []
        for row in rows:
            r = dict(row)
            r["ablation"] = str(ablation_name)
            r["budget_generations"] = int(generations)
            r["budget_population"] = int(pop)
            r["budget_timeout"] = float(timeout)
            if r.get("error"):
                r["failed_seed"] = True
            annotated.append(r)
        return annotated

    def _budget_dict() -> dict[str, Any]:
        return {
            "generations": int(generations),
            "population_size": int(pop),
            "timeout": float(timeout),
            "n_samples": int(n_samples),
            "seeds": list(seeds),
            "n_jobs": int(n_jobs),
            "omp_num_threads": int(omp_threads),
        }

    def _factory_kwargs(ablation_name: str) -> dict[str, Any]:
        return {
            "generations": int(generations),
            "population_size": int(pop),
            "timeout": float(timeout),
            "allow_stub": bool(args.smoke),
            "ablation": str(ablation_name),
            "blackbox_protocol": bool(args.blackbox),
        }

    out_dir = Path(args.output_dir)

    # ------------------------------------------------------------------
    # Phase E: multi-ablation comparison table
    # ------------------------------------------------------------------
    if args.ablation_table:
        ablation_names = [
            a.strip().lower() for a in str(args.ablations).split(",") if a.strip()
        ]
        if not ablation_names:
            raise SystemExit("--ablations resolved to empty list")
        unknown = [a for a in ablation_names if a not in ABLATION_PRESETS]
        if unknown:
            raise SystemExit(
                f"unknown ablation(s) {unknown}; choose from {sorted(ABLATION_PRESETS)}"
            )
        if "full" not in ablation_names:
            # Always include baseline for deltas.
            ablation_names = ["full"] + ablation_names

        if not args.quiet:
            print(
                f"Ablation table: {len(ablation_names)} ablations × "
                f"{len(problems)} problems × {len(tiers)} tiers × "
                f"{len(seeds)} seeds  (n_samples={n_samples})"
            )
            print(f"Ablations: {', '.join(ablation_names)}")
            print(f"Problems:  {', '.join(p[0] for p in problems)}")
            print(f"Tiers:     {', '.join(t['name'] for t in tiers)}")
            print(f"Seeds:     {seeds}")
            print(
                f"Budget:    generations={generations} population={pop} "
                f"timeout={timeout}s"
            )
            print(
                f"Parallel:  jobs={n_jobs}  omp_num_threads={omp_threads} "
                f"(workers × OMP ≈ {n_jobs * omp_threads})"
            )
            print(f"Blackbox protocol: {bool(args.blackbox)}")

        rows_by_ablation: dict[str, list[dict[str, Any]]] = {}
        all_rows: list[dict[str, Any]] = []
        for abl in ablation_names:
            try:
                factory = _default_estimator_factory(**_factory_kwargs(abl))
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
            if not args.quiet:
                print(f"\n--- ablation={abl} ---")
            rows = run_noise_protocol(
                factory,
                problems,
                tiers=tiers,
                seeds=seeds,
                n_samples=n_samples,
                verbose=not args.quiet,
                detail=bool(args.detail),
                silence_fit=not bool(args.show_fit_logs),
                n_jobs=n_jobs,
                omp_num_threads=omp_threads,
                factory_kwargs=_factory_kwargs(abl),
            )
            rows = _annotate_rows(rows, abl)
            rows_by_ablation[abl] = rows
            all_rows.extend(rows)

        table = build_ablation_table(rows_by_ablation, baseline="full")
        table["budget"] = _budget_dict()
        table["blackbox_protocol"] = bool(args.blackbox)
        table["problems"] = [p[0] for p in problems]
        table["tiers"] = [t["name"] for t in tiers]

        # Primary protocol report uses full ablation rows.
        full_rows = rows_by_ablation.get("full", all_rows)
        summary = summarize_noise_protocol(full_rows)
        summary["ablation"] = "full"
        summary["budget"] = _budget_dict()
        summary["ablation_table"] = True
        failed = [r for r in all_rows if r.get("error")]
        summary["failed_seeds"] = [
            {
                "problem": r.get("problem"),
                "tier": r.get("tier"),
                "seed": r.get("seed"),
                "ablation": r.get("ablation"),
                "error": r.get("error"),
            }
            for r in failed
        ]
        summary["n_failed_seeds"] = len(failed)

        paths = write_report(all_rows, summary, out_dir)
        abl_paths = write_ablation_report(table, out_dir)
        paths.update(abl_paths)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stamped = out_dir / f"noise_protocol_ablation_{stamp}"
        stamped.mkdir(parents=True, exist_ok=True)
        for key, src in paths.items():
            dest = stamped / Path(src).name
            dest.write_bytes(Path(src).read_bytes())

        if not args.quiet:
            print("\nWrote:")
            for key, p in paths.items():
                print(f"  {key}: {p}")
            print(f"  stamped: {stamped}")
            print("\n" + ablation_table_to_markdown(table))
        return 0

    # ------------------------------------------------------------------
    # Single-ablation protocol (default)
    # ------------------------------------------------------------------
    try:
        factory = _default_estimator_factory(**_factory_kwargs(str(args.ablation)))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if not args.quiet:
        print(
            f"Noise protocol: {len(problems)} problems × {len(tiers)} tiers × "
            f"{len(seeds)} seeds  (n_samples={n_samples})"
        )
        print(f"Problems: {', '.join(p[0] for p in problems)}")
        print(f"Tiers:    {', '.join(t['name'] for t in tiers)}")
        print(f"Seeds:    {seeds}")
        print(f"Ablation: {args.ablation}")
        print(
            f"Budget:   generations={generations} population={pop} "
            f"timeout={timeout}s (report budgets when comparing methods)"
        )
        print(
            f"Parallel: jobs={n_jobs}  omp_num_threads={omp_threads} "
            f"(workers × OMP ≈ {n_jobs * omp_threads})"
        )

    rows = run_noise_protocol(
        factory,
        problems,
        tiers=tiers,
        seeds=seeds,
        n_samples=n_samples,
        verbose=not args.quiet,
        detail=bool(args.detail),
        silence_fit=not bool(args.show_fit_logs),
        n_jobs=n_jobs,
        omp_num_threads=omp_threads,
        factory_kwargs=_factory_kwargs(str(args.ablation)),
    )
    rows = _annotate_rows(rows, str(args.ablation))
    summary = summarize_noise_protocol(rows)
    summary["ablation"] = str(args.ablation)
    summary["budget"] = _budget_dict()
    failed = [r for r in rows if r.get("error")]
    summary["failed_seeds"] = [
        {
            "problem": r.get("problem"),
            "tier": r.get("tier"),
            "seed": r.get("seed"),
            "error": r.get("error"),
        }
        for r in failed
    ]
    summary["n_failed_seeds"] = len(failed)
    paths = write_report(rows, summary, out_dir)

    # Phase E+: multi-seed publish table (Exact/Accept matrix for release freeze).
    if args.publish_table or (args.blackbox and len(seeds) >= 2):
        pub = build_publish_table(rows, seeds=seeds)
        pub["budget"] = _budget_dict()
        pub["blackbox_protocol"] = bool(args.blackbox)
        pub["problems"] = [p[0] for p in problems]
        pub["tiers"] = [t["name"] for t in tiers]
        pub["ablation"] = str(args.ablation)
        pub_paths = write_publish_report(pub, out_dir)
        paths.update(pub_paths)
        summary["publish_table"] = True
        summary["seed_coverage_ok"] = bool(pub.get("seed_coverage_ok"))
        if not args.quiet:
            print("\n" + publish_table_to_markdown(pub))

    # Also stamp a dated copy for baseline freeze.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stamped = out_dir / f"noise_protocol_{stamp}"
    stamped.mkdir(parents=True, exist_ok=True)
    for key, src in paths.items():
        dest = stamped / Path(src).name
        dest.write_bytes(Path(src).read_bytes())

    if not args.quiet:
        print("\nWrote:")
        for key, p in paths.items():
            print(f"  {key}: {p}")
        print(f"  stamped: {stamped}")
        print("\n" + to_markdown(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
