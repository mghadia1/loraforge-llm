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


def test_the_guard_is_not_a_bare_assertion() -> None:
    """`python -O` strips assert statements; a real exception type survives it."""
    assert issubclass(SplitLeakError, ValueError)


# --- drift: two runs compared across different library stacks ---------------


def report(packages: dict, rank: int, macro_f1: float, gpu: str = "Tesla T4") -> dict:
    return {
        "environment": {"packages": packages, "gpu_name": gpu},
        "config": {"lora": {"rank": rank, "alpha": rank * 2}, "training": {"epochs": 2}},
        "epochs": [{"epoch": 2, "validation": {"macro_f1": macro_f1}}],
        "selection": {"selected_epoch": 2},
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


def test_the_comparison_reads_the_selected_epoch_not_the_best_one() -> None:
    """Under the tie-break the selected epoch can differ from argmax."""
    from loraforge.compare import compare_runs

    tie_broken = report(STACK, 4, 0.9295)
    tie_broken["epochs"] = [
        {"epoch": 1, "validation": {"macro_f1": 0.9295}},
        {"epoch": 2, "validation": {"macro_f1": 0.9400}},  # higher, but not selected
    ]
    tie_broken["selection"] = {"selected_epoch": 1}

    comparison = compare_runs(
        report(STACK, 16, 0.9310),
        tie_broken,
        expected_config_changes={"lora.rank", "lora.alpha"},
    )
    assert comparison["validation_macro_f1"]["variant"] == pytest.approx(0.9295)
    assert comparison["selected_epoch"]["variant"] == 1


def test_a_report_selecting_an_epoch_it_does_not_contain_is_refused() -> None:
    from loraforge.compare import compare_runs

    broken = report(STACK, 4, 0.9295)
    broken["selection"] = {"selected_epoch": 7}
    with pytest.raises(EvidenceError, match="not among its epochs"):
        compare_runs(report(STACK, 16, 0.9310), broken)


def test_the_same_article_in_two_splits_is_caught_despite_namespaced_row_ids() -> None:
    """row_id mixes in the split name, so cross-split duplicates need content identity."""
    train = Split(
        "train",
        (Example(row_id="train-0", text="Sprint  buys Nextel", label=3, source_index=0),),
    )
    test = Split(
        "test",
        (Example(row_id="test-0", text="Sprint buys  Nextel", label=3, source_index=0),),
    )
    assert train.examples[0].row_id != test.examples[0].row_id
    with pytest.raises(SplitLeakError, match="identical article text and label"):
        assert_disjoint(train, test)


def test_different_articles_across_splits_are_not_flagged() -> None:
    train = Split("train", (Example(row_id="a", text="one story", label=0, source_index=0),))
    test = Split("test", (Example(row_id="b", text="another story", label=0, source_index=0),))
    assert_disjoint(train, test)
