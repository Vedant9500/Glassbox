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


def test_row_train_val_split_preserves_sizes():
    train_idx, val_idx = row_train_val_split(10, val_ratio=0.3, seed=1)

    assert len(train_idx) == 7
    assert len(val_idx) == 3
    assert set(train_idx).isdisjoint(set(val_idx))
