"""Phase 0: deterministic noise-protocol benchmark tests.

Covers the public surface in ``scripts/benchmark_noise.py``:
noise-generator determinism + level sanity, the report-column contract, and a
tiny end-to-end protocol run on one problem x one tier x one seed.
"""

import numpy as np
import pytest

from scripts import benchmark_noise as bn


# ---------------------------------------------------------------------------
# Noise generators
# ---------------------------------------------------------------------------
def test_noise_generators_are_deterministic_per_seed():
    y = np.sin(np.linspace(0.0, 10.0, 200))
    for tier in bn.NOISE_TIERS:
        a = bn.apply_noise_tier(y, tier, seed=7)
        b = bn.apply_noise_tier(y, tier, seed=7)
        assert np.allclose(a, b), f"{tier['name']} not reproducible"


def test_seed_changes_noise_output():
    y = np.sin(np.linspace(0.0, 10.0, 200))
    for tier in bn.NOISE_TIERS:
        # Quantization of a fixed signal is seed-independent by construction
        # (deterministic rounding), so exclude it from the seed-sensitivity check.
        if tier["noise_type"] in ("clean", "quantization"):
            continue
        a = bn.apply_noise_tier(y, tier, seed=7)
        b = bn.apply_noise_tier(y, tier, seed=8)
        assert not np.allclose(a, b), f"{tier['name']} ignores seed"


def test_clean_tier_is_noop():
    y = np.sin(np.linspace(0.0, 10.0, 64))
    out = bn.apply_noise_tier(y, bn.NOISE_TIERS[0], seed=1)
    assert np.allclose(out, y)


def test_gaussian_levels_match_target_rms():
    rng = np.random.RandomState(0)
    y = rng.uniform(-1, 1, size=4000)
    for tier_name, target in (
        ("gaussian_0.1pct", 0.001),
        ("gaussian_1pct", 0.01),
        ("gaussian_10pct", 0.10),
    ):
        tier = next(t for t in bn.NOISE_TIERS if t["name"] == tier_name)
        yn = bn.apply_noise_tier(y, tier, seed=3)
        rms = float(np.std(yn - y)) / (float(np.std(y)) + 1e-12)
        # Generous tolerance (small-sample jitter) but order-of-magnitude must hold.
        assert abs(rms - target) < target * 0.5, (
            f"{tier_name}: rms={rms} target={target}"
        )


def test_outlier_tier_only_touches_a_few_points():
    rng = np.random.RandomState(0)
    y = rng.uniform(-1, 1, size=500)
    tier = next(t for t in bn.NOISE_TIERS if t["name"] == "outliers_3pct")
    yn = bn.apply_noise_tier(y, tier, seed=5)
    changed = int(np.sum(~np.isclose(yn, y)))
    # 3% of 500 = 15 expected; allow small slack.
    assert 5 <= changed <= 40


def test_quantization_reduces_unique_values():
    y = np.linspace(-1, 1, 200)
    tier = next(t for t in bn.NOISE_TIERS if t["name"] == "quantization_64")
    yn = bn.apply_noise_tier(y, tier, seed=0)
    assert len(np.unique(yn)) <= 65


# ---------------------------------------------------------------------------
# Report-column contract
# ---------------------------------------------------------------------------
def test_required_columns_present():
    assert "noise_level" in bn.REQUIRED_COLUMNS
    assert "false_confidence" in bn.REQUIRED_COLUMNS
    assert "seed_graphs_used" in bn.REQUIRED_COLUMNS
    assert "clean_test_mse" in bn.REQUIRED_COLUMNS
    assert "clean_test_r2" in bn.REQUIRED_COLUMNS
    assert "acceptable_clean" in bn.REQUIRED_COLUMNS
    assert "blackbox_enabled" in bn.REQUIRED_COLUMNS
    assert "n_features" in bn.REQUIRED_COLUMNS
    assert "noise_band" in bn.REQUIRED_COLUMNS
    assert "selected_features" in bn.REQUIRED_COLUMNS


def test_assert_row_contract_passes_for_well_formed_row():
    row = {col: None for col in bn.REQUIRED_COLUMNS}
    row["problem"] = "p"
    row["tier"] = "clean"
    row["seed"] = 1
    bn.assert_row_contract([row])


def test_assert_row_contract_rejects_missing_column():
    row = {col: None for col in bn.REQUIRED_COLUMNS}
    del row["false_confidence"]
    with pytest.raises(AssertionError):
        bn.assert_row_contract([row])


def test_false_confidence_helper():
    assert bn._false_confidence(train_r2=0.99, test_r2=0.3) is True
    assert bn._false_confidence(train_r2=0.99, test_r2=0.99) is False
    assert bn._false_confidence(train_r2=0.5, test_r2=0.3) is False
    assert bn._false_confidence(train_r2=None, test_r2=0.3) is None


# ---------------------------------------------------------------------------
# Tiny end-to-end protocol run
# ---------------------------------------------------------------------------
def _toy_problem():
    return (
        "Toy-Linear",
        lambda X: 2.0 * X[:, 0] + 1.0,
        1,
        [(-3.0, 3.0)],
        "2*x0 + 1",
    )


def test_protocol_run_emits_contract_rows():

    def factory():
        from glassbox.sr.sklearn_wrapper import GlassboxRegressor

        return GlassboxRegressor(
            random_state=1,
            generations=15,
            multi_start_runs=1,
            population_size=40,
            timeout=20,
        )

    clean_tier = [bn.NOISE_TIERS[0]]
    rows = bn.run_noise_protocol(
        factory,
        [_toy_problem()],
        tiers=clean_tier,
        seeds=[42],
        n_samples=120,
        verbose=False,
    )
    assert len(rows) == 1
    bn.assert_row_contract(rows)
    r = rows[0]
    assert r["problem"] == "Toy-Linear"
    assert r["tier"] == "clean"
    assert r["seed"] == 42
    assert r["sample_weight_mode"] == "none"
    assert r["error"] is None
    assert "clean_test_mse" in r
    assert "clean_test_r2" in r
    assert "acceptable_clean" in r
    assert "false_confidence_vs_clean" in r


def test_summary_delta_table_keys():

    def factory():
        from glassbox.sr.sklearn_wrapper import GlassboxRegressor

        return GlassboxRegressor(
            random_state=1,
            generations=10,
            multi_start_runs=1,
            population_size=30,
            timeout=15,
        )

    tiers = [bn.NOISE_TIERS[0], bn.NOISE_TIERS[3]]  # clean + gaussian 10%
    rows = bn.run_noise_protocol(
        factory, [_toy_problem()], tiers=tiers, seeds=[42], n_samples=120, verbose=False
    )
    summary = bn.summarize_noise_protocol(rows)
    assert "cells" in summary and "deltas_vs_clean" in summary
    assert len(summary["cells"]) == 2
    deltas = summary["deltas_vs_clean"]
    assert len(deltas) == 1  # noisy tier vs clean
    assert deltas[0]["tier"] == "gaussian_10pct"
    md = bn.to_markdown(summary)
    assert "R2noisy" in md
    assert "R2clean" in md
    assert "CleanMSE" in md


def test_write_report_creates_files(tmp_path):
    rows = [{col: None for col in bn.REQUIRED_COLUMNS}]
    rows[0].update(
        {
            "problem": "p",
            "tier": "clean",
            "seed": 1,
            "test_r2": 0.9,
            "exact_match": False,
            "true_formula": "x",
            "discovered_formula": "x",
            "error": None,
        }
    )
    summary = bn.summarize_noise_protocol(rows)
    paths = bn.write_report(rows, summary, tmp_path)
    assert all(p.exists() for p in paths.values())


def test_clean_vs_noisy_metrics_diverge_under_noise():
    """Under noise, noisy R2 can look good while clean recovery fails."""
    rng = np.random.RandomState(0)
    x = np.linspace(-3.0, 3.0, 200)
    y_clean = 2.0 * x + 1.0
    y_noisy = bn.add_gaussian_noise(y_clean, 0.10, seed=0)
    # Wrong structure that still roughly tracks trend on noisy labels
    y_pred = 2.0 * x + 1.0 + 0.05 * np.sin(3.0 * x)
    noisy_r2 = 1.0 - float(np.mean((y_pred - y_noisy) ** 2) / np.var(y_noisy))
    clean_mse = float(np.mean((y_pred - y_clean) ** 2))
    assert clean_mse > 1e-6
    assert noisy_r2 > 0.9  # noisy fit still looks strong


def test_markdown_includes_clean_columns():
    rows = [{col: None for col in bn.REQUIRED_COLUMNS}]
    rows[0].update(
        {
            "problem": "p",
            "tier": "clean",
            "seed": 1,
            "test_r2": 0.9,
            "clean_test_r2": 0.95,
            "exact_match": True,
            "acceptable_clean": True,
            "true_formula": "x",
            "discovered_formula": "x",
            "error": None,
            "raw_mse": 0.01,
            "display_mse": 0.01,
            "clean_test_mse": 0.0,
            "formula_complexity": 1,
        }
    )
    summary = bn.summarize_noise_protocol(rows)
    md = bn.to_markdown(summary)
    assert "R2clean" in md
    assert "Accept" in md


def test_constant_targets_receive_nonzero_gaussian_noise():
    y = np.full(200, 5.0)
    tier = next(t for t in bn.NOISE_TIERS if t["name"] == "gaussian_10pct")
    yn = bn.apply_noise_tier(y, tier, seed=3)
    rms = float(np.std(yn - y))
    # scale falls back to |mean|=5, 10% -> ~0.5
    assert rms > 0.1
    assert abs(rms - 0.5) < 0.25


def test_noise_amplitude_scale_constant_vs_spread():
    assert bn.noise_amplitude_scale(np.full(50, 5.0)) == pytest.approx(5.0)
    y = np.linspace(-2, 2, 100)
    assert bn.noise_amplitude_scale(y) == pytest.approx(float(np.std(y)))


def test_select_problems_default_set():
    problems = bn._select_problems()
    assert len(problems) == len(bn.DEFAULT_BASELINE_PROBLEMS)
    assert problems[0][0] == bn.DEFAULT_BASELINE_PROBLEMS[0]


def test_default_parallel_config_ryzen_class():
    # 8c/16t class laptop: prefer 4 workers × 4 OMP.
    jobs, omp = bn.default_parallel_config(n_jobs=0, omp_num_threads=0, cpu_count=16)
    assert jobs == 4
    assert omp == 4
    jobs2, omp2 = bn.default_parallel_config(n_jobs=6, omp_num_threads=0, cpu_count=16)
    assert jobs2 == 6
    assert omp2 == 2
    jobs3, omp3 = bn.default_parallel_config(n_jobs=1, omp_num_threads=8, cpu_count=16)
    assert jobs3 == 1
    assert omp3 == 8


def test_protocol_parallel_jobs_match_sequential_order():
    """Process-pool path keeps payload order and emits contract rows."""
    factory_kwargs = {
        "generations": 5,
        "population_size": 10,
        "timeout": 5.0,
        "allow_stub": True,
        "ablation": "full",
        "blackbox_protocol": False,
    }
    factory = bn._default_estimator_factory(**factory_kwargs)
    problems = bn._select_problems(["Poly-x2"])
    tiers = [bn.NOISE_TIERS[0], bn.NOISE_TIERS[3]]  # clean + gaussian_10pct
    seeds = [11, 23]

    seq = bn.run_noise_protocol(
        factory,
        problems,
        tiers=tiers,
        seeds=seeds,
        n_samples=40,
        verbose=False,
        n_jobs=1,
        factory_kwargs=factory_kwargs,
    )
    par = bn.run_noise_protocol(
        factory,
        problems,
        tiers=tiers,
        seeds=seeds,
        n_samples=40,
        verbose=False,
        n_jobs=2,
        omp_num_threads=1,
        factory_kwargs=factory_kwargs,
    )
    assert len(par) == len(seq) == 4
    bn.assert_row_contract(par)
    assert [(r["problem"], r["tier"], r["seed"]) for r in par] == [
        (r["problem"], r["tier"], r["seed"]) for r in seq
    ]
    # Stub mean-fit should be deterministic across process boundary.
    assert [r.get("discovered_formula") for r in par] == [
        r.get("discovered_formula") for r in seq
    ]


def test_select_blackbox_problems_are_multivariate():
    problems = bn._select_problems(bn.DEFAULT_BLACKBOX_PROBLEMS)
    assert len(problems) == len(bn.DEFAULT_BLACKBOX_PROBLEMS)
    for name, _fn, n_features, _ranges, _formula in problems:
        assert int(n_features) > 1, name
    # At least one problem can exercise top-k ranking under default min_features=5.
    assert any(int(p[2]) >= 5 for p in problems)


def test_build_ablation_table_release_keys(tmp_path):
    """Phase E: ablation table compares full vs no_weights on multi-var rows."""

    def _mk(problem, tier, ablation, clean_r2, accept):
        row = {c: None for c in bn.REQUIRED_COLUMNS}
        row.update(
            {
                "problem": problem,
                "tier": tier,
                "seed": 11,
                "test_r2": clean_r2,
                "clean_test_r2": clean_r2,
                "exact_match": clean_r2 > 0.99,
                "acceptable_clean": accept,
                "false_confidence": False,
                "raw_mse": 0.2,
                "display_mse": 0.2,
                "clean_test_mse": max(0.0, 1.0 - clean_r2),
                "formula_complexity": 10 if ablation == "full" else 30,
                "error": None,
                "ablation": ablation,
                "blackbox_enabled": True,
                "n_features": 5,
            }
        )
        return row

    rows_by = {
        "full": [
            _mk("Pagie-1", "clean", "full", 1.0, True),
            _mk("Pagie-1", "outliers_3pct", "full", 0.92, True),
        ],
        "no_weights": [
            _mk("Pagie-1", "clean", "no_weights", 1.0, True),
            _mk("Pagie-1", "outliers_3pct", "no_weights", 0.55, False),
        ],
    }
    table = bn.build_ablation_table(rows_by, baseline="full")
    assert "full" in table["ablations"]
    assert "no_weights" in table["ablations"]
    md = bn.ablation_table_to_markdown(table)
    assert "Phase E" in md or "Ablation Table" in md
    paths = bn.write_ablation_report(table, tmp_path)
    assert paths["ablation_json"].exists()
    assert paths["ablation_markdown"].exists()
    assert set(bn.DEFAULT_BLACKBOX_RELEASE_ABLATIONS) <= set(bn.ABLATION_PRESETS)


def test_build_publish_table_multi_seed(tmp_path):
    """Phase E+: multi-seed publish table exposes Exact matrix + coverage."""

    def _mk(problem, tier, seed, exact, accept, r2=1.0):
        row = {c: None for c in bn.REQUIRED_COLUMNS}
        row.update(
            {
                "problem": problem,
                "tier": tier,
                "seed": seed,
                "test_r2": r2,
                "clean_test_r2": r2,
                "clean_full_mse": 1e-9 if exact else 0.05,
                "exact_match": exact,
                "acceptable_clean": accept,
                "false_confidence": False,
                "raw_mse": 0.1,
                "display_mse": 0.1,
                "clean_test_mse": max(0.0, 1.0 - r2),
                "formula_complexity": 12,
                "formula": "10/(5+(x0-3)^2+(x1-3)^2)",
                "error": None,
                "blackbox_enabled": True,
                "n_features": 5,
            }
        )
        return row

    rows = []
    for seed in (11, 7, 23):
        rows.append(_mk("Vladislavleva-4", "clean", seed, True, True, 1.0))
        rows.append(
            _mk("Vladislavleva-4", "outliers_3pct", seed, seed != 7, True, 0.99)
        )
        rows.append(_mk("Pagie-1", "clean", seed, True, True, 1.0))
        rows.append(_mk("Pagie-1", "outliers_3pct", seed, False, True, 0.95))

    table = bn.build_publish_table(rows, seeds=(11, 7, 23), min_seeds=2)
    assert table["seed_coverage_ok"] is True
    assert table["n_cells"] == 4
    md = bn.publish_table_to_markdown(table)
    assert "Multi-Seed Publish Table" in md
    assert "Per-seed Exact matrix" in md
    paths = bn.write_publish_report(table, tmp_path)
    assert paths["publish_json"].exists()
    assert paths["publish_markdown"].exists()
    assert bn.DEFAULT_PUBLISH_SEEDS[0] == 11


def test_blackbox_diag_fields_from_estimator_attrs():
    class _E:
        blackbox_diagnostics_ = {
            "enabled": True,
            "reason": "selected_top_features",
            "selected_features": [0, 2],
            "n_selected_features": 2,
            "feature_selection_uncertain": False,
            "ranking_sample_weight_mode": "provided",
            "runtime_noise": {"noise_band": "low"},
        }
        blackbox_search_plan_ = {"noise_band": "low", "noise_pressure": 0.4}

    fields = bn._blackbox_diag_fields(_E(), n_features=5)
    assert fields["blackbox_enabled"] is True
    assert fields["n_features"] == 5
    assert fields["selected_features"] == [0, 2]
    assert fields["noise_band"] == "low"
    assert fields["noise_pressure"] == pytest.approx(0.4)
    assert fields["ranking_sample_weight_mode"] == "provided"


def test_main_smoke_writes_protocol_artifacts(tmp_path, monkeypatch):
    """CLI smoke path writes protocol artifacts without a full multi-seed suite.

    Uses a tiny fake estimator so this does not require the C++ backend.
    """

    class _FakeEst:
        def __init__(self, **kwargs):
            self.params = kwargs
            self.blackbox_diagnostics_ = {"sample_weight": {"provided": False}}

        def get_params(self):
            return dict(self.params)

        def fit(self, X, y):
            self._ymean = float(np.mean(y))
            return self

        def predict(self, X):
            return np.full(len(X), getattr(self, "_ymean", 0.0))

        def get_formula(self):
            return "0"

    def fake_factory(**kwargs):
        def factory():
            return _FakeEst(**kwargs)

        return factory

    monkeypatch.setattr(
        bn,
        "_default_estimator_factory",
        lambda **kw: fake_factory(**kw),
    )
    rc = bn.main(
        [
            "--smoke",
            "--output-dir",
            str(tmp_path / "out"),
            "--quiet",
        ]
    )
    assert rc == 0
    out = tmp_path / "out"
    assert (out / "noise_protocol_rows.json").exists()
    assert (out / "noise_protocol_summary.json").exists()
    assert (out / "noise_protocol_report.md").exists()
    stamped = list(out.glob("noise_protocol_*"))
    assert any(p.is_dir() for p in stamped)
