from pathlib import Path

import numpy as np

from glassbox.curve_classifier.generate_curve_data import (
    FEATURE_DIM,
    N_CLASSES,
    OPERATOR_CLASSES,
    SEMANTIC_LABELER_VERSION,
    build_formula_audit_metadata,
    derive_operators_from_formula,
    derive_semantic_operators_from_formula,
    formula_to_key,
    generate_dataset,
    operators_to_labels,
    save_dataset,
)


def _active_ops(labels):
    return {name for name, idx in OPERATOR_CLASSES.items() if float(labels[idx]) > 0.5}


def test_semantic_labeler_suppresses_unary_argument_identity():
    syntax_ops = derive_operators_from_formula("np.sin(x)")
    semantic_ops = derive_semantic_operators_from_formula("np.sin(x)")

    assert syntax_ops == {"identity", "sin"}
    assert semantic_ops == {"sin"}


def test_semantic_labeler_filters_domain_guard_wrappers():
    assert derive_semantic_operators_from_formula("np.sqrt(np.abs(x) + 0.01)") == {
        "power"
    }
    assert derive_semantic_operators_from_formula("np.log(np.abs(x) + 1)") == {"log"}
    assert derive_semantic_operators_from_formula("1 / (x**2 + 0.1)") == {
        "power",
        "rational",
    }
    assert derive_semantic_operators_from_formula("(np.abs(x) + 0.01) ** -0.5") == {
        "power",
        "rational",
    }


def test_semantic_labeler_keeps_affine_structure_inside_function_arguments():
    ops = derive_semantic_operators_from_formula("np.sin(2 * x + 0.5)")

    assert ops == {"sin", "multiplication"}


def test_semantic_labeler_treats_addition_as_two_dependent_terms():
    assert derive_semantic_operators_from_formula("x + 1") == {"identity"}
    assert derive_semantic_operators_from_formula("np.sin(x) + x") == {
        "identity",
        "sin",
        "addition",
    }
    assert derive_semantic_operators_from_formula("np.sinh(x)") == {"exp"}


def test_operators_to_labels_defaults_to_semantic_when_formula_is_available():
    semantic = operators_to_labels({"identity", "sin"}, formula="np.sin(x)")
    syntax = operators_to_labels(
        {"identity", "sin"}, formula="np.sin(x)", label_mode="syntax"
    )
    provided = operators_to_labels(
        {"identity", "sin"}, formula="np.sin(x)", label_mode="provided"
    )

    assert _active_ops(semantic) == {"sin"}
    assert _active_ops(syntax) == {"identity", "sin"}
    assert _active_ops(provided) == {"identity", "sin"}


def test_duplicate_formula_label_conflict_is_canonicalized():
    formula = "np.cos(x)"
    labels_from_template_a = operators_to_labels({"cos"}, formula=formula)
    labels_from_template_b = operators_to_labels({"identity", "cos"}, formula=formula)

    np.testing.assert_array_equal(labels_from_template_a, labels_from_template_b)
    assert _active_ops(labels_from_template_a) == {"cos"}


def test_generate_dataset_can_return_phase2_generation_metadata():
    templates = [
        ("np.sin(x)", {"identity", "sin"}),
        ("np.sqrt(np.abs(x) + 0.01)", {"identity", "power", "addition"}),
    ]

    features, labels, formulas, metadata = generate_dataset(
        n_samples=4,
        n_points=64,
        templates=templates,
        seed=123,
        n_workers=1,
        balance_templates=True,
        show_progress=False,
        noise_std=0.0,
        y_scale_min=1.0,
        y_scale_max=1.0,
        y_offset_std=0.0,
        pcfg_ratio=0.0,
        return_metadata=True,
    )

    assert features.shape == (4, FEATURE_DIM)
    assert labels.shape == (4, N_CLASSES)
    assert len(formulas) == 4
    assert len(metadata) == 4

    for formula, row_labels, row_meta in zip(formulas, labels, metadata):
        expected = operators_to_labels(set(), formula=formula)
        np.testing.assert_array_equal(row_labels, expected)
        assert row_meta["labeler_version"] == SEMANTIC_LABELER_VERSION
        assert row_meta["formula_key"] == formula_to_key(formula)
        assert row_meta["generator_family"] in {"simple", "template"}
        assert row_meta["template_id"] >= 0
        assert tuple(row_meta["semantic_operators"]) == tuple(
            sorted(_active_ops(row_labels))
        )


def test_save_dataset_embeds_phase2_audit_metadata():
    formulas = ["np.sin(x)", "np.sqrt(np.abs(x) + 0.01)"]
    labels = np.vstack(
        [
            operators_to_labels({"identity", "sin"}, formula=formulas[0]),
            operators_to_labels({"identity", "power", "addition"}, formula=formulas[1]),
        ]
    )
    features = np.zeros((2, FEATURE_DIM), dtype=np.float32)
    generation_metadata = [
        {
            "formula_key": formula_to_key(formulas[0]),
            "generator_family": "simple",
            "template_id": 10,
            "provided_operators": ("identity", "sin"),
            "syntax_operators": tuple(
                sorted(derive_operators_from_formula(formulas[0]))
            ),
            "semantic_operators": tuple(
                sorted(derive_semantic_operators_from_formula(formulas[0]))
            ),
            "labeler_version": SEMANTIC_LABELER_VERSION,
        },
        {
            "formula_key": formula_to_key(formulas[1]),
            "generator_family": "simple",
            "template_id": 11,
            "provided_operators": ("addition", "identity", "power"),
            "syntax_operators": tuple(
                sorted(derive_operators_from_formula(formulas[1]))
            ),
            "semantic_operators": tuple(
                sorted(derive_semantic_operators_from_formula(formulas[1]))
            ),
            "labeler_version": SEMANTIC_LABELER_VERSION,
        },
    ]

    out = Path("scratch") / "pytest_phase2_dataset.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_dataset(
        out, features, labels, formulas, generation_metadata=generation_metadata
    )

    data = np.load(out, allow_pickle=True)
    assert str(data["labeler_version"]) == SEMANTIC_LABELER_VERSION
    assert data["formula_keys"].tolist() == [formula_to_key(f) for f in formulas]
    assert data["generator_families"].tolist() == ["simple", "simple"]
    assert data["template_ids"].tolist() == [10, 11]
    assert data["semantic_labels"].shape == (2, N_CLASSES)
    assert data["labels_match_semantic"].tolist() == [True, True]
    assert tuple(data["semantic_operators"][0]) == ("sin",)
    assert tuple(data["provided_operators"][0]) == ("identity", "sin")


def test_build_formula_audit_metadata_falls_back_without_generation_records():
    formulas = ["np.tanh(x)"]
    labels = np.vstack([operators_to_labels(set(), formula=formulas[0])])

    metadata = build_formula_audit_metadata(formulas, labels=labels)

    assert metadata["labeler_version"] == SEMANTIC_LABELER_VERSION
    assert metadata["formula_keys"].tolist() == [formula_to_key(formulas[0])]
    assert metadata["generator_families"].tolist() == ["unknown"]
    assert metadata["template_ids"].tolist() == [-1]
    assert tuple(metadata["semantic_operators"][0]) == ("exp", "rational")
    assert bool(metadata["labels_match_semantic"][0]) is True
