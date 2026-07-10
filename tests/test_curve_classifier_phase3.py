from pathlib import Path

import numpy as np
import torch

from glassbox.curve_classifier.generate_curve_data import (
    FEATURE_DIM,
    N_CLASSES,
    formula_to_key,
    operators_to_labels,
    save_dataset,
)
from glassbox.curve_classifier.validation import (
    build_validation_report,
    family_holdout_split,
    formula_overlap_report,
    grouped_train_val_split,
    multilabel_metrics_by_group,
    multilabel_metric_summary,
    row_train_val_split,
)
from glassbox.curve_classifier import train_curve_classifier as tcc
from scripts import train_universal_proposer as tup


def test_grouped_train_val_split_has_no_formula_overlap():
    groups = np.array(["a", "a", "b", "b", "c", "c", "d", "d"], dtype=object)

    train_idx, val_idx, details = grouped_train_val_split(groups, val_ratio=0.25, seed=7)

    assert len(train_idx) > 0
    assert len(val_idx) > 0
    assert details["exclusive_groups"] is True
    overlap = formula_overlap_report(groups, train_idx, val_idx)
    assert overlap["overlap_unique_formulas"] == 0
    assert overlap["val_rows_with_train_formula_fraction"] == 0.0


def test_grouped_train_val_split_prefers_formula_diversity_over_largest_groups():
    groups = np.array(["huge_a"] * 40 + ["huge_b"] * 35 + [f"small_{i}" for i in range(20)], dtype=object)

    train_idx, val_idx, details = grouped_train_val_split(groups, val_ratio=0.2, seed=3)

    overlap = formula_overlap_report(groups, train_idx, val_idx)
    assert overlap["overlap_unique_formulas"] == 0
    assert details["val_group_count"] >= details["target_val_groups"]
    assert overlap["val_unique_formulas"] > 4


def test_family_holdout_split_uses_only_heldout_family_for_validation():
    families = np.array(["simple", "simple", "pcfg", "pcfg", "nested"], dtype=object)

    train_idx, val_idx, details = family_holdout_split(families, "pcfg")

    assert details["policy"] == "generator_family_holdout"
    assert set(families[val_idx].tolist()) == {"pcfg"}
    assert "pcfg" not in set(families[train_idx].tolist())


def test_validation_report_includes_overlap_and_distributions():
    labels = np.vstack([
        operators_to_labels(set(), formula="np.sin(x)"),
        operators_to_labels(set(), formula="np.sin(x)"),
        operators_to_labels(set(), formula="x ** 2"),
        operators_to_labels(set(), formula="x ** 2"),
    ])
    keys = np.array([formula_to_key("np.sin(x)")] * 2 + [formula_to_key("x ** 2")] * 2, dtype=object)
    families = np.array(["simple", "simple", "simple", "simple"], dtype=object)
    templates = np.array([1, 1, 2, 2], dtype=object)
    train_idx = np.array([0, 1])
    val_idx = np.array([2, 3])

    report = build_validation_report(
        dataset_path="scratch/example.npz",
        split_policy="formula_group",
        train_idx=train_idx,
        val_idx=val_idx,
        labels=labels,
        operator_classes=["identity", "sin", "cos", "power", "exp", "log", "addition", "multiplication", "rational"],
        formula_keys=keys,
        generator_families=families,
        template_ids=templates,
        split_details={"exclusive_groups": True},
        metrics={"best_checkpoint": {"val_f1": 0.5}},
    )

    assert report["schema_version"] == "validation.phase3.v1"
    assert report["formula_overlap"]["overlap_unique_formulas"] == 0
    assert report["label_distribution"]["train"]["sin"] == 2
    assert report["label_distribution"]["val"]["power"] == 2
    assert report["template_distribution"]["val"]["2"] == 2
    assert report["metrics"]["best_checkpoint"]["val_f1"] == 0.5


def test_multilabel_metric_summary_reports_precision_recall_and_group_breakdown():
    probs = np.array([
        [0.9, 0.1],
        [0.8, 0.7],
        [0.2, 0.6],
        [0.1, 0.4],
    ], dtype=np.float32)
    labels = np.array([
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1],
    ], dtype=np.float32)
    groups = np.array(["simple", "simple", "pcfg", "pcfg"], dtype=object)

    summary = multilabel_metric_summary(probs, labels, ["a", "b"])
    by_group = multilabel_metrics_by_group(probs, labels, groups, ["a", "b"], min_rows=1)

    assert summary["precision_per_class"][0] == 1.0
    assert summary["recall_per_class"][0] == 1.0
    assert summary["recall_per_class"][1] == 0.5
    assert by_group["simple"]["rows"] == 2
    assert by_group["pcfg"]["rows"] == 2


def test_training_data_loader_reads_phase2_validation_metadata():
    formulas = ["np.sin(x)", "x ** 2", "np.cos(x)", "np.tanh(x)"]
    labels = np.vstack([operators_to_labels(set(), formula=f) for f in formulas])
    features = np.zeros((len(formulas), FEATURE_DIM), dtype=np.float32)
    generation_metadata = [
        {
            "formula_key": formula_to_key(formula),
            "generator_family": "simple" if i < 3 else "hyperbolic",
            "template_id": i,
            "provided_operators": tuple(),
            "syntax_operators": tuple(),
            "semantic_operators": tuple(),
            "labeler_version": "semantic-labeler-v1",
        }
        for i, formula in enumerate(formulas)
    ]
    out = Path("scratch") / "pytest_phase3_dataset.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_dataset(out, features, labels, formulas, generation_metadata=generation_metadata)

    old_shape_loaded = tcc.load_training_data([str(out)], None, FEATURE_DIM, N_CLASSES, False)
    assert len(old_shape_loaded) == 5

    loaded = tcc.load_training_data([str(out)], None, FEATURE_DIM, N_CLASSES, False, return_metadata=True)
    _, _, operator_classes, feature_dim, _schema, metadata = loaded

    assert feature_dim == FEATURE_DIM
    assert operator_classes[:4] == ["identity", "sin", "cos", "power"]
    assert metadata["formula_keys"].tolist() == [formula_to_key(f) for f in formulas]
    assert metadata["generator_families"].tolist() == ["simple", "simple", "simple", "hyperbolic"]
    assert metadata["template_ids"].tolist() == [0, 1, 2, 3]
    assert metadata["labels_match_semantic"].tolist() == [True, True, True, True]


def test_proposer_replay_dataset_preserves_loader_compatibility():
    formulas = ["np.sin(x)", "x ** 2"]
    labels = np.vstack([operators_to_labels(set(), formula=f) for f in formulas])
    features = np.zeros((len(formulas), FEATURE_DIM), dtype=np.float32)
    out = Path("scratch") / "pytest_phase3_proposer_dataset.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_dataset(out, features, labels, formulas)

    old_shape_loaded = tup.load_training_data(str(out), n_classes=N_CLASSES)
    assert len(old_shape_loaded) == 6

    metadata_shape_loaded = tup.load_training_data(str(out), n_classes=N_CLASSES, return_metadata=True)
    assert len(metadata_shape_loaded) == 7

    ds = tup.FormulaReplayDataset(out)
    sample_features, op_target, skeleton_target = ds[0]

    assert sample_features.shape == (FEATURE_DIM,)
    assert op_target.shape[0] == len(tup.DEFAULT_OPERATOR_VOCAB)
    assert int(skeleton_target.item()) >= 0


def test_proposer_skeleton_metric_summary_reports_topk_and_calibration():
    logits = torch.tensor([
        [5.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
        [0.0, 0.0, 3.0],
        [3.0, 0.0, 0.0],
    ])
    targets = torch.tensor([0, 1, 1, -1])

    metrics = tup._skeleton_metric_summary(logits, targets)

    assert metrics["skeleton_valid_count"] == 3
    assert abs(metrics["skeleton_top1_acc"] - (2 / 3)) < 1e-6
    assert metrics["skeleton_top5_acc"] == 1.0
    assert metrics["skeleton_confidence_mean"] is not None
    assert metrics["skeleton_ece_10"] is not None


def test_proposer_operator_pos_weight_maps_periodic_from_sin_cos():
    labels = np.zeros((10, N_CLASSES), dtype=np.float32)
    labels[:2, 1] = 1.0  # sin
    labels[:1, 2] = 1.0  # cos

    weights = tup.compute_operator_pos_weight(
        labels,
        np.arange(10),
        ["identity", "sin", "cos", "power", "exp", "log", "addition", "multiplication", "rational"],
        cap=8.0,
    )

    sin_weight = weights[tup.DEFAULT_OPERATOR_VOCAB.index("sin")]
    periodic_weight = weights[tup.DEFAULT_OPERATOR_VOCAB.index("periodic")]
    assert float(sin_weight) == 4.0
    assert float(periodic_weight) == 4.0


def test_proposer_skeleton_loss_gate_disables_low_coverage_dataset():
    formulas = ["not_in_vocab(x)" for _ in range(10)] + ["np.sin(x)"]
    labels = np.zeros((len(formulas), N_CLASSES), dtype=np.float32)
    features = np.zeros((len(formulas), FEATURE_DIM), dtype=np.float32)
    ds = tup.FormulaReplayDataset(features, labels, formulas=formulas)

    enabled, coverage = tup.skeleton_loss_enabled_from_coverage(ds, min_coverage=0.8)

    assert enabled is False
    assert coverage < 0.8


def test_proposer_checkpoint_metric_prefers_candidate_recall_then_micro_f1():
    assert tup.select_checkpoint_metric({"candidate_recall_after_affine_fit": 0.7, "micro_f1": 0.2}) == 0.7
    assert tup.select_checkpoint_metric({"candidate_recall_after_affine_fit": None, "micro_f1": 0.8}) == 0.8


def test_row_train_val_split_preserves_sizes():
    train_idx, val_idx = row_train_val_split(10, val_ratio=0.3, seed=1)

    assert len(train_idx) == 7
    assert len(val_idx) == 3
    assert set(train_idx).isdisjoint(set(val_idx))


def test_classifier_validation_calibration_split_keeps_eval_disjoint():
    labels = np.zeros((20, N_CLASSES), dtype=np.float32)
    labels[:10, 1] = 1.0
    labels[10:, 2] = 1.0
    val_idx = np.arange(20)

    eval_idx, calibration_idx, details = tcc.split_validation_calibration(
        val_idx,
        labels,
        calibration_ratio=0.25,
        seed=5,
    )

    assert details["calibration_split"] is True
    assert calibration_idx is not None
    assert set(eval_idx).isdisjoint(set(calibration_idx))
    assert sorted(np.concatenate([eval_idx, calibration_idx]).tolist()) == val_idx.tolist()


def test_classifier_threshold_beta_can_prefer_precision():
    preds = torch.tensor([0.95, 0.80, 0.70, 0.60, 0.40, 0.30]).reshape(-1, 1)
    labels = torch.tensor([1, 0, 0, 0, 1, 1], dtype=torch.float32).reshape(-1, 1)

    f1_threshold = tcc.tune_thresholds(preds, labels, beta=1.0, steps=19)
    precision_threshold = tcc.tune_thresholds(preds, labels, beta=0.5, steps=19)

    assert float(precision_threshold[0]) >= float(f1_threshold[0])
