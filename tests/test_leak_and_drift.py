"""Regression tests for two bugs that were caught by hand and would not fail loudly."""

from __future__ import annotations

import pytest

from loraforge.compare import compare_runs, require_controlled
from loraforge.data import Example, Split, SplitLeakError, assert_disjoint, assert_no_leaks
from loraforge.data import DatasetBundle
from loraforge.provenance import EvidenceError


def split(name: str, ids: range) -> Split:
    return Split(
        name,
        tuple(Example(row_id=f"row-{i}", text=f"t{i}", label=i % 4, source_index=i) for i in ids),
    )


# --- leak: validation rows sitting inside the training set ------------------


def test_disjoint_splits_pass() -> None:
    assert_disjoint(split("train", range(0, 100)), split("validation", range(100, 120)))


def test_a_validation_row_inside_training_is_refused() -> None:
    with pytest.raises(SplitLeakError, match="also appear in train"):
        assert_disjoint(split("train", range(0, 100)), split("validation", range(95, 115)))


def test_growing_training_to_swallow_validation_is_refused() -> None:
    """The exact shape of the bug: training expands over the validation rows."""
    validation = split("validation", range(2_000, 2_500))
    honest = DatasetBundle(train=split("train", range(0, 2_000)), validation=validation)
    assert_no_leaks(honest)  # 118k-style split: training stops short of validation

    greedy = DatasetBundle(train=split("train", range(0, 3_000)), validation=validation)
    with pytest.raises(SplitLeakError):
        assert_no_leaks(greedy)  # 120k-style split: training swallowed validation


def test_test_contamination_is_refused_in_both_directions() -> None:
    test = split("test", range(500, 600))
    with pytest.raises(SplitLeakError):
        assert_no_leaks(
            DatasetBundle(split("train", range(550, 700)), split("validation", range(0, 10)), test)
        )
    with pytest.raises(SplitLeakError):
        assert_no_leaks(
            DatasetBundle(split("train", range(0, 10)), split("validation", range(590, 620)), test)
        )


def test_cross_publisher_split_duplicate_is_found_despite_distinct_row_ids() -> None:
    trained = Split(
        "train",
        (Example(row_id="train-namespaced", text="same article", label=0, source_index=1),),
    )
    judged = Split(
        "test",
        (Example(row_id="test-namespaced", text="same article", label=0, source_index=99),),
    )
    with pytest.raises(SplitLeakError):
        assert_disjoint(trained, judged)


def test_whitespace_variants_that_render_identically_are_a_leak() -> None:
    trained = Split(
        "train",
        (Example(row_id="a", text="same\n article", label=0, source_index=1),),
    )
    judged = Split(
        "test",
        (Example(row_id="b", text=" same   article ", label=2, source_index=2),),
    )
    with pytest.raises(SplitLeakError):
        assert_disjoint(trained, judged)


def test_the_guard_is_not_a_bare_assertion() -> None:
    """`python -O` strips assert statements; a real exception type survives it."""
    assert issubclass(SplitLeakError, ValueError)


# --- drift: two runs compared across different library stacks ---------------


def report(packages: dict, rank: int, macro_f1: float, gpu: str = "Tesla T4") -> dict:
    return {
        "environment": {"packages": packages, "gpu_name": gpu},
        "config": {"lora": {"rank": rank, "alpha": rank * 2}, "training": {"epochs": 2}},
        "epochs": [{"epoch": 1, "validation": {"macro_f1": macro_f1}}],
        "selection": {"selected_epoch": 1},
        "validation_label_sha256": "same-validation-labels",
        "parameters": {"trainable_parameters": rank * 2_621_440},
        "wall_time_seconds": 13_000,
    }


STACK = {"transformers": "5.13.1", "peft": "0.19.1", "torch": "2.11.0+cu128"}


def test_a_clean_rank_ablation_is_controlled() -> None:
    comparison = compare_runs(
        report(STACK, 16, 0.9310),
        report(STACK, 4, 0.9295),
        expected_config_changes={"lora.rank", "lora.alpha"},
    )
    assert comparison["controlled"] is True
    assert comparison["validation_macro_f1"]["delta"] == pytest.approx(-0.0015)
    require_controlled(comparison)


def test_missing_field_cannot_collide_with_its_display_marker() -> None:
    from loraforge.compare import MISSING, differences

    assert differences({"field": MISSING}, {}) == {"field": (MISSING, MISSING)}


def test_a_changed_library_version_blocks_the_comparison() -> None:
    drifted = dict(STACK, transformers="5.14.0")
    comparison = compare_runs(
        report(STACK, 16, 0.9310),
        report(drifted, 4, 0.8500),
        expected_config_changes={"lora.rank", "lora.alpha"},
    )
    assert comparison["controlled"] is False
    with pytest.raises(EvidenceError, match="library versions changed"):
        require_controlled(comparison)


def test_a_different_host_torch_is_reported_but_not_blocking() -> None:
    """torch comes from the host image, so it is disclosed rather than policed."""
    other = dict(STACK, torch="2.10.0+cu121")
    comparison = compare_runs(
        report(STACK, 16, 0.9310),
        report(other, 4, 0.9295),
        expected_config_changes={"lora.rank", "lora.alpha"},
    )
    assert comparison["controlled"] is True
    assert "torch" in comparison["package_differences"]


def test_a_different_gpu_blocks_the_controlled_comparison() -> None:
    comparison = compare_runs(
        report(STACK, 16, 0.9310, gpu="Tesla T4"),
        report(STACK, 4, 0.9295, gpu="A100"),
        expected_config_changes={"lora.rank", "lora.alpha"},
    )
    assert comparison["controlled"] is False
    with pytest.raises(EvidenceError, match="GPU model changed"):
        require_controlled(comparison)


def test_different_validation_rows_block_the_controlled_comparison() -> None:
    variant = report(STACK, 4, 0.9295)
    variant["validation_label_sha256"] = "different-validation-labels"
    comparison = compare_runs(
        report(STACK, 16, 0.9310),
        variant,
        expected_config_changes={"lora.rank", "lora.alpha"},
    )
    assert comparison["controlled"] is False
    with pytest.raises(EvidenceError, match="validation label digest"):
        require_controlled(comparison)


def test_reported_metric_comes_from_the_rule_selected_epoch() -> None:
    baseline = report(STACK, 16, 0.90)
    baseline["epochs"] = [
        {"epoch": 1, "validation": {"macro_f1": 0.90}},
        {"epoch": 2, "validation": {"macro_f1": 0.95}},
    ]
    baseline["selection"]["selected_epoch"] = 2
    variant = report(STACK, 4, 0.91)
    comparison = compare_runs(
        baseline, variant, expected_config_changes={"lora.rank", "lora.alpha"}
    )
    assert comparison["validation_macro_f1"]["baseline"] == 0.95


def test_selection_that_bypasses_the_tie_rule_is_refused() -> None:
    baseline = report(STACK, 16, 0.90)
    baseline["epochs"] = [
        {"epoch": 1, "validation": {"macro_f1": 0.90}},
        {"epoch": 2, "validation": {"macro_f1": 0.90}},
    ]
    baseline["selection"]["selected_epoch"] = 2
    with pytest.raises(EvidenceError, match="selection rule"):
        compare_runs(
            baseline,
            report(STACK, 4, 0.91),
            expected_config_changes={"lora.rank", "lora.alpha"},
        )


def test_an_unexpected_config_change_blocks_the_comparison() -> None:
    variant = report(STACK, 4, 0.9295)
    variant["config"]["training"]["epochs"] = 3
    comparison = compare_runs(
        report(STACK, 16, 0.9310), variant, expected_config_changes={"lora.rank", "lora.alpha"}
    )
    with pytest.raises(EvidenceError, match="unexpected config changes"):
        require_controlled(comparison)


def test_an_ablation_that_forgot_to_change_anything_is_refused() -> None:
    comparison = compare_runs(
        report(STACK, 16, 0.9310),
        report(STACK, 16, 0.9308),
        expected_config_changes={"lora.rank", "lora.alpha"},
    )
    with pytest.raises(EvidenceError, match="did not actually change"):
        require_controlled(comparison)
