import numpy as np
import pytest
import torch

from glassbox.curve_classifier.models import (
    CurveClassifierGLU,
    DualHeadCurveClassifier,
)
from glassbox.curve_classifier.train_dual_head_classifier import (
    build_checkpoint,
    compute_pos_weights,
    operator_state_dict,
    parse_weight_overrides,
    preprocess_features,
    split_calibration,
    summarize_head,
    tune_multilabel_thresholds,
)
from glassbox.curve_classifier.validation import build_validation_report


def _batch(n: int = 4, n_features: int = 398) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randn(n, n_features, generator=generator)


def test_forward_features_matches_single_head_forward():
    model = CurveClassifierGLU(n_features=398, n_classes=9, hidden=32)
    model.eval()
    x = _batch()

    with torch.no_grad():
        direct = model(x)
        recomposed = model.classifier(model.forward_features(x))

    assert torch.allclose(direct, recomposed, atol=1e-6)


def test_dual_head_returns_both_logit_sets_and_shares_trunk():
    model = DualHeadCurveClassifier(
        n_features=398, n_operator_classes=9, n_shape_labels=10, hidden=32
    )
    model.eval()
    x = _batch()

    with torch.no_grad():
        operator_logits, shape_logits = model(x)

    assert operator_logits.shape == (4, 9)
    assert shape_logits.shape == (4, 10)
    assert model.operator_head is model.backbone.classifier
    assert model.shape_head.in_features == 32


def test_operator_state_dict_loads_into_plain_glu():
    dual = DualHeadCurveClassifier(
        n_features=398, n_operator_classes=9, n_shape_labels=10, hidden=32
    )
    dual.eval()
    plain = CurveClassifierGLU(n_features=398, n_classes=9, hidden=32)
    missing, unexpected = plain.load_state_dict(
        operator_state_dict(dual), strict=True
    )
    assert missing == []
    assert unexpected == []
    plain.eval()

    x = _batch()
    with torch.no_grad():
        dual_operator, _ = dual(x)
        plain_logits = plain(x)

    assert torch.allclose(dual_operator, plain_logits, atol=1e-6)


def test_preprocess_applies_symlog_before_standardization():
    features = np.zeros((3, 398), dtype=np.float32)
    features[:, 192:398] = 5.0
    features[:, 0] = 1.0
    scaler = {"mean": np.zeros(398, dtype=np.float32), "std": np.ones(398, dtype=np.float32)}

    out = preprocess_features(features, scaler)

    np.testing.assert_allclose(out[:, 192:398], np.log1p(5.0), rtol=1e-6)
    np.testing.assert_allclose(out[:, 0], 1.0, rtol=1e-6)


def test_summarize_head_reports_micro_and_per_class():
    labels = np.array([[1, 0], [1, 1], [0, 1], [0, 0]], dtype=np.float32)
    probs = np.array([[0.9, 0.1], [0.8, 0.7], [0.2, 0.6], [0.1, 0.2]], dtype=np.float32)

    summary = summarize_head(probs, labels, ["a", "b"])

    assert summary["micro_f1"] == 1.0
    assert summary["per_class_f1"] == {"a": 1.0, "b": 1.0}
    assert summary["support"] == {"a": 2, "b": 2}
    assert summary["zero_support_classes"] == []
    assert summary["macro_f1_supported"] == 1.0
    assert summary["thresholds"] is None


def test_summarize_head_excludes_zero_support_classes_from_supported_macro():
    labels = np.zeros((10, 3), dtype=np.float32)
    labels[:5, 0] = 1.0
    probs = np.zeros((10, 3), dtype=np.float32)
    probs[:5, 0] = 0.9

    summary = summarize_head(probs, labels, ["a", "b", "c"])

    assert summary["zero_support_classes"] == ["b", "c"]
    assert summary["support"] == {"a": 5, "b": 0, "c": 0}
    # Raw macro-F1 averages in the vacuous zeros; the supported variant does not.
    assert summary["macro_f1"] == pytest.approx(1.0 / 3.0)
    assert summary["macro_f1_supported"] == 1.0


def test_summarize_head_flags_classes_with_no_negatives():
    labels = np.ones((6, 2), dtype=np.float32)
    probs = np.ones((6, 2), dtype=np.float32)

    summary = summarize_head(probs, labels, ["a", "b"])

    # Precision (and therefore F1) is vacuous when every row is positive.
    assert summary["all_positive_classes"] == ["a", "b"]
    assert summary["zero_support_classes"] == []


def test_tune_multilabel_thresholds_recovers_a_shifted_operating_point():
    n = 2000
    labels = np.zeros((n, 2), dtype=np.float32)
    labels[:200, 0] = 1.0
    labels[:800, 1] = 1.0
    probs = np.full((n, 2), 0.02, dtype=np.float32)
    probs[:200, 0] = 0.15  # rare class is ranked correctly but scores far below 0.5
    probs[:800, 1] = 0.8

    thresholds = tune_multilabel_thresholds(probs, labels, steps=25)

    assert thresholds[0] < 0.15
    assert thresholds[1] == pytest.approx(0.5, abs=0.2)

    raw = summarize_head(probs, labels, ["a", "b"])
    tuned = summarize_head(probs, labels, ["a", "b"], thresholds=thresholds)
    assert raw["per_class_f1"]["a"] == 0.0
    assert tuned["per_class_f1"]["a"] > 0.95
    assert tuned["thresholds"]["a"] == pytest.approx(float(thresholds[0]))


def test_tune_multilabel_thresholds_leaves_zero_support_classes_at_default():
    labels = np.zeros((50, 2), dtype=np.float32)
    labels[:10, 0] = 1.0
    probs = np.zeros((50, 2), dtype=np.float32)
    probs[:10, 0] = 0.9

    thresholds = tune_multilabel_thresholds(probs, labels)

    assert thresholds[1] == pytest.approx(0.5)


def test_compute_pos_weights_clips_and_applies_overrides():
    labels = np.zeros((100, 3), dtype=np.float32)
    labels[:10, 0] = 1.0  # neg/pos = 9
    labels[:50, 1] = 1.0  # neg/pos = 1
    labels[:2, 2] = 1.0  # neg/pos = 49, capped

    weights = compute_pos_weights(labels, ["a", "b", "c"], cap=16.0)
    np.testing.assert_allclose(weights, [9.0, 1.0, 16.0], rtol=1e-6)

    overridden = compute_pos_weights(
        labels, ["a", "b", "c"], cap=16.0, overrides={"c": 4.0}
    )
    np.testing.assert_allclose(overridden, [9.0, 1.0, 4.0], rtol=1e-6)

    with pytest.raises(ValueError, match="unknown class"):
        compute_pos_weights(labels, ["a", "b", "c"], overrides={"zzz": 2.0})


def test_parse_weight_overrides_parses_and_rejects_malformed_pairs():
    assert parse_weight_overrides("addition=4, log=8") == {
        "addition": 4.0,
        "log": 8.0,
    }
    assert parse_weight_overrides("") == {}
    with pytest.raises(ValueError, match="name=value"):
        parse_weight_overrides("addition")


def test_split_calibration_is_disjoint_deterministic_and_skips_tiny_sets():
    indices = np.arange(100)

    fit, calib = split_calibration(indices, 0.25, seed=7)

    assert len(calib) == 25
    assert len(fit) == 75
    assert set(fit).isdisjoint(set(calib.tolist()))
    assert sorted(fit.tolist() + calib.tolist()) == indices.tolist()

    _, calib_again = split_calibration(indices, 0.25, seed=7)
    np.testing.assert_array_equal(calib, calib_again)

    _, too_small = split_calibration(np.arange(10), 0.25, seed=0)
    assert too_small is None


def test_checkpoint_thresholds_are_a_flat_operator_list_for_legacy_inference(tmp_path):
    from glassbox.curve_classifier.curve_classifier_integration import (
        _build_result_dict,
    )

    operator_classes = [f"op{i}" for i in range(9)]
    shape_names = [f"shape{i}" for i in range(10)]
    model = DualHeadCurveClassifier(
        n_features=398, n_operator_classes=9, n_shape_labels=10, hidden=8
    )
    operator_thresholds = np.linspace(0.05, 0.85, 9, dtype=np.float32)
    shape_thresholds = np.linspace(0.1, 0.9, 10, dtype=np.float32)
    dataset = {
        "operator_classes": operator_classes,
        "shape_label_names": shape_names,
        "feature_schema": "v7",
        "feature_dim": 398,
        "labeler_version": "v7",
    }

    checkpoint = build_checkpoint(
        model,
        dataset=dataset,
        model_config={
            "n_features": 398,
            "n_operator_classes": 9,
            "n_shape_labels": 10,
            "hidden": 8,
        },
        scaler=None,
        best_metrics={
            "epoch": 3,
            "val_loss": 0.5,
            "operator": {"micro_f1": 0.9, "macro_f1": 0.8, "per_class_f1": {}},
            "shape": {"micro_f1": 0.95, "macro_f1": 0.9, "per_class_f1": {}},
        },
        tuned_metrics={
            "operator": {"micro_f1": 0.93, "macro_f1": 0.85},
            "shape": {"micro_f1": 0.96, "macro_f1": 0.91},
        },
        operator_thresholds=operator_thresholds,
        shape_thresholds=shape_thresholds,
        threshold_meta={"enabled": True, "beta": 1.0},
        split_policy="formula_group",
        split_details={"policy": "formula_group"},
        validation_report_path=tmp_path / "report.json",
        checkpoint_card_path=tmp_path / "card.json",
    )

    assert isinstance(checkpoint["thresholds"], list)
    assert len(checkpoint["thresholds"]) == 9
    assert checkpoint["thresholds"] == [float(x) for x in operator_thresholds]
    assert len(checkpoint["shape_thresholds"]) == 10
    assert checkpoint["thresholded_val_f1"] == pytest.approx(0.93)
    assert checkpoint["thresholds_meta"]["enabled"] is True

    path = tmp_path / "dual.pt"
    torch.save(checkpoint, path)
    reloaded = torch.load(path, weights_only=True)
    assert reloaded["thresholds"] == checkpoint["thresholds"]

    # Legacy inference reads a flat list indexed by class position.
    probs = operator_thresholds + np.array(
        [0.02, -0.02] * 4 + [0.02], dtype=np.float32
    )
    result = _build_result_dict(
        probs,
        0.5,
        {"operator_classes": operator_classes, "thresholds": reloaded["thresholds"]},
        "test-cache-key",
    )
    for name, threshold, prob in zip(
        operator_classes, operator_thresholds, probs, strict=False
    ):
        assert (name in result) is bool(prob >= threshold)


def test_holdout_training_reports_final_epoch_not_best_on_holdout():
    from glassbox.curve_classifier.train_dual_head_classifier import train_model

    torch.manual_seed(0)
    model = DualHeadCurveClassifier(
        n_features=398, n_operator_classes=2, n_shape_labels=2, hidden=8
    )
    features = torch.randn(8, 398)
    operator_labels = (torch.rand(8, 2) > 0.5).float()
    shape_labels = (torch.rand(8, 2) > 0.5).float()
    idx = np.arange(8)

    _, metrics, info = train_model(
        model,
        features,
        operator_labels,
        shape_labels,
        idx,
        idx,
        epochs=2,
        batch_size=4,
        lr=1e-3,
        weight_decay=0.0,
        shape_loss_weight=1.0,
        patience=0,
        eval_every=1,
        eval_batch_size=8,
        operator_classes=["a", "b"],
        shape_label_names=["x", "y"],
        device=torch.device("cpu"),
        seed=0,
        label="test",
        select_best=False,
    )

    assert metrics["epoch"] == 2
    assert info["best_epoch"] == 2


def test_validation_report_dual_head_fields_are_optional():
    labels = np.zeros((4, 2), dtype=np.float32)
    labels[:2, 0] = 1.0
    labels[2:, 1] = 1.0
    train_idx = np.array([0, 1])
    val_idx = np.array([2, 3])

    base = build_validation_report(
        dataset_path="scratch/example.npz",
        split_policy="formula_group",
        train_idx=train_idx,
        val_idx=val_idx,
        labels=labels,
        operator_classes=["a", "b"],
    )
    assert "shape_head" not in base
    assert "family_holdout" not in base
    assert "thresholds" not in base

    extended = build_validation_report(
        dataset_path="scratch/example.npz",
        split_policy="formula_group",
        train_idx=train_idx,
        val_idx=val_idx,
        labels=labels,
        operator_classes=["a", "b"],
        shape_label_names=["periodic", "even"],
        shape_metrics={"micro_f1": 0.9},
        thresholds={
            "operator": {"a": 0.2, "b": 0.6},
            "shape": {"periodic": 0.5, "even": 0.5},
            "enabled": True,
        },
        family_holdout={"compound": {"operator": {"micro_f1": 0.7}}},
    )
    assert extended["schema_version"] == "validation.phase3.v1"
    assert extended["shape_head"]["label_names"] == ["periodic", "even"]
    assert extended["shape_head"]["metrics"]["micro_f1"] == 0.9
    assert extended["family_holdout"]["compound"]["operator"]["micro_f1"] == 0.7
    assert extended["thresholds"]["operator"]["a"] == 0.2
    assert extended["thresholds"]["enabled"] is True


def test_main_wires_calibration_thresholds_and_class_weights(tmp_path):
    import json

    from glassbox.curve_classifier import train_dual_head_classifier as trainer

    rng = np.random.RandomState(0)
    n_rows, n_features = 120, 398
    features = rng.randn(n_rows, n_features).astype(np.float32)
    operator_labels = (rng.rand(n_rows, 9) < 0.3).astype(np.float32)
    shape_labels = (rng.rand(n_rows, 10) < 0.4).astype(np.float32)
    data_path = tmp_path / "dual.npz"
    np.savez(
        data_path,
        features=features,
        labels=operator_labels,
        shape_labels=shape_labels,
        operator_classes=np.array([f"op{i}" for i in range(9)], dtype=object),
        shape_label_names=np.array([f"s{i}" for i in range(10)], dtype=object),
        formula_keys=np.array(
            [f"f{i // 6}" for i in range(n_rows)], dtype=object
        ),
        generator_families=np.array(
            ["a" if i % 2 else "b" for i in range(n_rows)], dtype=object
        ),
        feature_schema=np.array("test"),
        feature_dim=np.array(n_features),
        labeler_version=np.array("test"),
    )
    output = tmp_path / "model.pt"

    exit_code = trainer.main(
        [
            "--data",
            str(data_path),
            "--output",
            str(output),
            "--epochs",
            "2",
            "--batch-size",
            "16",
            "--hidden",
            "8",
            "--val-split",
            "0.25",
            "--calibration-split",
            "0.25",
            "--class-weights",
            "--pos-weight-override",
            "op0=4",
        ]
    )

    assert exit_code == 0
    checkpoint = torch.load(output, weights_only=True)
    assert len(checkpoint["thresholds"]) == 9
    assert len(checkpoint["shape_thresholds"]) == 10
    assert checkpoint["thresholds_meta"]["enabled"] is True
    assert checkpoint["thresholds_meta"]["calibration_rows"] > 0
    assert checkpoint["model_config"]["dual_head"] is True

    report = json.loads(
        output.with_name(f"{output.stem}.validation.json").read_text()
    )
    assert "operator_tuned" in report["metrics"]
    assert "thresholds" in report
    assert report["split_details"]["calibration_rows"] > 0
    assert report["split_details"]["eval_rows"] > 0


def test_dual_head_inference_round_trips_a_checkpoint(tmp_path):
    from glassbox.curve_classifier.dual_head_inference import (
        load_dual_head_checkpoint,
        predict_dual_head,
    )

    model = DualHeadCurveClassifier(
        n_features=398, n_operator_classes=9, n_shape_labels=10, hidden=16
    )
    checkpoint = {
        "dual_head": True,
        "dual_head_state_dict": model.state_dict(),
        "model_config": {
            "n_features": 398,
            "n_operator_classes": 9,
            "n_shape_labels": 10,
            "hidden": 16,
        },
        "operator_classes": [
            "identity",
            "sin",
            "cos",
            "power",
            "exp",
            "log",
            "addition",
            "multiplication",
            "rational",
        ],
        "shape_label_names": [
            "periodic",
            "monotonic_increasing",
            "monotonic_decreasing",
            "even",
            "odd",
            "linear",
            "has_local_extrema",
            "diverges",
            "convex",
            "concave",
        ],
        "feature_scaler": {
            "mean": [0.0] * 398,
            "std": [1.0] * 398,
        },
    }
    path = tmp_path / "dual.pt"
    torch.save(checkpoint, path)

    loaded, metadata = load_dual_head_checkpoint(path)
    assert metadata["dual_head"] is True
    assert loaded.shape_head.out_features == 10

    x = np.linspace(-5.0, 5.0, 256)
    y = np.sin(x) + 0.1 * x
    result = predict_dual_head(x, y, path)

    assert set(result["operators"]) == set(checkpoint["operator_classes"])
    assert set(result["shape"]) == set(checkpoint["shape_label_names"])
    assert all(0.0 <= p <= 1.0 for p in result["operators"].values())
    assert all(0.0 <= p <= 1.0 for p in result["shape"].values())
    assert result["shape_label_names"] == checkpoint["shape_label_names"]


def test_dual_head_inference_uses_checkpoint_thresholds(tmp_path):
    from glassbox.curve_classifier.dual_head_inference import predict_dual_head

    operator_classes = [
        "identity",
        "sin",
        "cos",
        "power",
        "exp",
        "log",
        "addition",
        "multiplication",
        "rational",
    ]
    shape_names = [
        "periodic",
        "monotonic_increasing",
        "monotonic_decreasing",
        "even",
        "odd",
        "linear",
        "has_local_extrema",
        "diverges",
        "convex",
        "concave",
    ]
    model = DualHeadCurveClassifier(
        n_features=398, n_operator_classes=9, n_shape_labels=10, hidden=16
    )
    checkpoint = {
        "dual_head": True,
        "dual_head_state_dict": model.state_dict(),
        "model_config": {
            "n_features": 398,
            "n_operator_classes": 9,
            "n_shape_labels": 10,
            "hidden": 16,
        },
        "operator_classes": operator_classes,
        "shape_label_names": shape_names,
        "feature_scaler": {"mean": [0.0] * 398, "std": [1.0] * 398},
        # Every class fires at 0.0 so the decision is driven purely by the cut.
        "thresholds": [0.0] * 8 + [1.0],
        "shape_thresholds": [0.0] * 10,
    }
    path = tmp_path / "dual_thresholds.pt"
    torch.save(checkpoint, path)

    x = np.linspace(-5.0, 5.0, 256)
    y = np.sin(x)
    result = predict_dual_head(x, y, path, threshold=0.99)

    assert result["thresholds"]["operator"]["rational"] == 1.0
    assert "rational" not in result["operators_above_threshold"]
    assert set(result["operators_above_threshold"]) == set(operator_classes[:-1])
    assert set(result["shape_above_threshold"]) == set(shape_names)


def test_dual_head_inference_rejects_threshold_length_mismatch(tmp_path):
    from glassbox.curve_classifier.dual_head_inference import predict_dual_head

    model = DualHeadCurveClassifier(
        n_features=398, n_operator_classes=9, n_shape_labels=10, hidden=8
    )
    checkpoint = {
        "dual_head": True,
        "dual_head_state_dict": model.state_dict(),
        "model_config": {
            "n_features": 398,
            "n_operator_classes": 9,
            "n_shape_labels": 10,
            "hidden": 8,
        },
        "operator_classes": [f"op{i}" for i in range(9)],
        "shape_label_names": [f"shape{i}" for i in range(10)],
        "feature_scaler": {"mean": [0.0] * 398, "std": [1.0] * 398},
        "thresholds": [0.5] * 8,
    }
    path = tmp_path / "bad_thresholds.pt"
    torch.save(checkpoint, path)

    x = np.linspace(-5.0, 5.0, 256)
    with pytest.raises(ValueError, match="declares 9 classes"):
        predict_dual_head(x, np.sin(x), path)
