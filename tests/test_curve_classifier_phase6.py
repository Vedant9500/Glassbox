from pathlib import Path

from glassbox.curve_classifier.rollout import (
    CHECKPOINT_CARD_SCHEMA_VERSION,
    ROLLOUT_COMPARISON_SCHEMA_VERSION,
    build_checkpoint_card,
    build_rollout_comparison,
    default_checkpoint_card_path,
    default_rollout_comparison_path,
)


def _validation_report(split_policy="formula_group", val_f1=0.8):
    return {
        "split_policy": split_policy,
        "split_details": {"policy": split_policy, "exclusive_groups": split_policy != "row"},
        "metrics": {"best_checkpoint": {"val_f1": val_f1, "val_micro_f1": val_f1 + 0.01}},
        "formula_overlap": {"available": True, "overlap_unique_formulas": 0},
    }


def test_checkpoint_card_records_phase6_rollout_contract():
    checkpoint = {
        "feature_dim": 398,
        "feature_schema": {"raw": [0, 128]},
        "labeler_version": "semantic-labeler-v1",
        "thresholds": [0.3, 0.4],
        "isotonic_calibration": [{"boundaries": [0.0, 1.0], "values": [0.0, 1.0]}],
        "validation_report_path": "models/candidate.validation.json",
    }

    card = build_checkpoint_card(
        model_kind="curve_classifier",
        checkpoint_path=Path("models/candidate.pt"),
        validation_report=_validation_report(),
        checkpoint_metadata=checkpoint,
        data_generation_command="python -m glassbox.curve_classifier.generate_curve_data --out data/fixed.npz",
        training_command="python -m glassbox.curve_classifier.train_curve_classifier --data data/fixed.npz",
        runtime_contract={"univariate": "trained_univariate_neural"},
    )

    assert card["schema_version"] == CHECKPOINT_CARD_SCHEMA_VERSION
    assert card["data_generation_command"].startswith("python -m glassbox")
    assert card["labeler_version"] == "semantic-labeler-v1"
    assert card["feature_dim"] == 398
    assert card["validation"]["grouped_release_metric"] is True
    assert card["calibration"]["thresholds_saved"] is True
    assert card["calibration"]["isotonic_calibration_saved"] is True
    assert card["release_gates"]["grouped_or_family_validation_reported"] is True
    assert card["release_gates"]["row_order_stress_passed"] is True
    assert card["release_gates"]["runtime_fallback_passed"] is True
    assert card["known_unsupported_cases"]


def test_rollout_comparison_blocks_without_baseline_and_passes_when_candidate_wins():
    candidate = build_checkpoint_card(
        model_kind="curve_classifier",
        checkpoint_path=Path("models/new.pt"),
        validation_report=_validation_report(val_f1=0.82),
        checkpoint_metadata={"feature_dim": 398},
    )
    baseline = build_checkpoint_card(
        model_kind="curve_classifier",
        checkpoint_path=Path("models/old.pt"),
        validation_report=_validation_report(val_f1=0.80),
        checkpoint_metadata={"feature_dim": 398},
    )

    missing_baseline = build_rollout_comparison(candidate_card=candidate)
    comparison = build_rollout_comparison(candidate_card=candidate, baseline_card=baseline)

    assert missing_baseline["schema_version"] == ROLLOUT_COMPARISON_SCHEMA_VERSION
    assert missing_baseline["recommendation"] == "needs_baseline_comparison"
    assert comparison["beats_baseline"] is True
    assert comparison["recommendation"] == "release_ready"


def test_rollout_comparison_blocks_row_split_even_if_metric_improves():
    candidate = build_checkpoint_card(
        model_kind="curve_classifier",
        checkpoint_path=Path("models/new.pt"),
        validation_report=_validation_report(split_policy="row", val_f1=0.9),
        checkpoint_metadata={"feature_dim": 398},
    )
    baseline = build_checkpoint_card(
        model_kind="curve_classifier",
        checkpoint_path=Path("models/old.pt"),
        validation_report=_validation_report(val_f1=0.8),
        checkpoint_metadata={"feature_dim": 398},
    )

    comparison = build_rollout_comparison(candidate_card=candidate, baseline_card=baseline)

    assert comparison["beats_baseline"] is True
    assert comparison["gates"]["grouped_or_family_validation_reported"] is False
    assert comparison["recommendation"] == "blocked_missing_grouped_validation"


def test_default_phase6_paths_are_next_to_checkpoint():
    checkpoint = Path("models/example.pt")

    assert default_checkpoint_card_path(checkpoint) == Path("models/example.card.json")
    assert default_rollout_comparison_path(checkpoint) == Path("models/example.rollout.json")
