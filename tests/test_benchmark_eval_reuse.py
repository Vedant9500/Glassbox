"""P-27 / benchmark-suite reuse: compiled display formulas and reused predictions."""

import builtins

import numpy as np

import scripts.benchmark_common as bc
import scripts.benchmark_suite as bs


def _compile_calls(monkeypatch):
    calls = {"n": 0}
    original = builtins.compile

    def counting(source, filename, mode, *args, **kwargs):
        if filename == "<benchmark_display_formula>":
            calls["n"] += 1
        return original(source, filename, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "compile", counting)
    return calls


def test_display_formula_compiled_once_per_string(monkeypatch):
    bc._FORMULA_COMPILE_CACHE.clear()
    calls = _compile_calls(monkeypatch)
    X = np.linspace(-1.0, 1.0, 40).reshape(-1, 1)

    first = bc.evaluate_formula("sin(x0) + 1.0*x0", X)
    second = bc.evaluate_formula("sin(x0) + 1.0*x0", X)

    assert first is not None and second is not None
    assert np.allclose(first, second)
    assert calls["n"] == 1


def test_display_eval_details_skips_parser_on_success(monkeypatch):
    monkeypatch.setattr(
        bs, "_evaluate_formula_mse", lambda formula, x, y: 0.25
    )
    called = {"n": 0}

    def _fail_if_called(*args, **kwargs):
        called["n"] += 1
        raise AssertionError("strict evaluator should not run on success")

    monkeypatch.setattr(bc, "evaluate_formula", _fail_if_called)
    x = np.linspace(-1.0, 1.0, 20)
    y = np.sin(x)
    details = bs._display_eval_details("sin(x0)", x, y)
    assert details["mse"] == 0.25
    assert details["diagnostics"] is None
    assert called["n"] == 0


def test_display_eval_details_reports_parser_failure(monkeypatch):
    monkeypatch.setattr(bs, "_evaluate_formula_mse", lambda formula, x, y: None)
    x = np.linspace(-1.0, 1.0, 20)
    y = np.sin(x)
    details = bs._display_eval_details("sin(", x, y)
    assert details["mse"] is None
    assert details["diagnostics"]["ok"] is False


def test_clean_metrics_reuse_prediction(monkeypatch):
    result = {"formula_discovered": "sin(x0)", "n_terms": 1}
    x = np.linspace(-1.0, 1.0, 40)
    y = np.sin(x)
    pred = np.sin(x)

    def _no_eval(*args, **kwargs):
        raise AssertionError("prediction should be reused")

    monkeypatch.setattr(bs.cfp, "_evaluate_formula_values", _no_eval)
    bs._attach_clean_target_metrics(result, x, y, y_pred=pred)
    assert result["r2_clean"] == 1.0
    assert result["recovery_acceptable"] is True


def test_best_result_key_breaks_mse_ties_by_complexity_and_text():
    def key(result):
        # Mirrors the nested helper in main(): finite flag, mse, complexity,
        # formula text.
        import math

        mse = result.get("score_mse", result.get("mse"))
        finite = mse is not None and math.isfinite(float(mse))
        formula = str(result.get("formula_discovered") or "").strip()
        complexity = result.get("formula_complexity")
        if complexity is None and formula:
            complexity = bc.formula_complexity_score(formula)
        return (
            0 if finite else 1,
            float(mse) if finite else float("inf"),
            int(complexity),
            formula,
        )

    simple = {"formula_discovered": "x0", "mse": 0.0}
    complex_ = {"formula_discovered": "x0^3 + x0^2 + x0 + 1", "mse": 0.0}
    assert key(simple) < key(complex_)
    # Same complexity: deterministic formula-text ordering.
    a = {"formula_discovered": "x0 + 1", "mse": 0.0}
    b = {"formula_discovered": "x0 - 1", "mse": 0.0}
    assert key(a) != key(b)
