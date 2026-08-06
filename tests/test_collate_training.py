from __future__ import annotations

import numpy as np
import pytest

from loraforge.collate import (
    LABEL_IGNORE_INDEX,
    build_supervised_features,
    pad_batch,
    supervised_loss_reference,
)
from loraforge.data import Example, Split
from loraforge.training import select_checkpoint


class FakeTokenizer:
    """Mirrors the frozen chat contract: prompt tokens [1, 20, 21] then one code token."""

    eos_token_id = 2
    pad_token_id = 2

    def encode(self, value, add_special_tokens=False):
        codes = {"A": 11, "B": 12, "C": 13, "D": 14}
        return [1, 20, 21] if value == "<chat>" else [1, 20, 21, codes[value[-1]]]

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return [1, 20, 21] if tokenize else "<chat>"


def split_of(labels):
    return Split(
        "train",
        tuple(
            Example(row_id=f"row-{index}", text=f"article {index}", label=label, source_index=index)
            for index, label in enumerate(labels)
        ),
    )


def test_collator_pads_inputs_with_pad_id_and_labels_with_ignore_index() -> None:
    batch = pad_batch(
        [
            {"input_ids": [1, 20, 21, 13, 2], "attention_mask": [1] * 5, "labels": [-100, -100, -100, 13, 2]},
            {"input_ids": [1, 20, 2], "attention_mask": [1] * 3, "labels": [-100, -100, 2]},
        ],
        pad_token_id=2,
    )
    assert batch["input_ids"][1] == [1, 20, 2, 2, 2]
    assert batch["attention_mask"][1] == [1, 1, 1, 0, 0]
    assert batch["labels"][1] == [-100, -100, 2, -100, -100]
    assert all(len(row) == 5 for row in batch["input_ids"])


def test_collator_rejects_empty_and_ragged_batches() -> None:
    with pytest.raises(ValueError, match="empty batch"):
        pad_batch([], pad_token_id=2)
    with pytest.raises(ValueError, match="same length"):
        pad_batch(
            [{"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100]}], pad_token_id=2
        )


def test_padding_tokens_never_contribute_to_the_loss() -> None:
    rng = np.random.default_rng(11)
    logits = rng.normal(size=(5, 16))
    labels = [-100, -100, -100, 13, 2]
    padded_logits = np.vstack([logits, rng.normal(size=(3, 16))])
    padded_labels = [*labels, LABEL_IGNORE_INDEX, LABEL_IGNORE_INDEX, LABEL_IGNORE_INDEX]
    assert supervised_loss_reference(logits, labels) == pytest.approx(
        supervised_loss_reference(padded_logits, padded_labels)
    )


def test_loss_ignores_prompt_positions_but_follows_the_answer_token() -> None:
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(5, 16))
    labels = [-100, -100, -100, 13, 2]
    original = supervised_loss_reference(logits, labels)

    prompt_changed = logits.copy()
    prompt_changed[0] += 50.0  # position 0 predicts a masked prompt token
    assert supervised_loss_reference(prompt_changed, labels) == pytest.approx(original)

    answer_changed = logits.copy()
    answer_changed[2, 13] += 50.0  # position 2 predicts the supervised answer code
    assert supervised_loss_reference(answer_changed, labels) < original


def test_supervised_features_supervise_exactly_the_code_and_eos() -> None:
    features = build_supervised_features(FakeTokenizer(), split_of([0, 2, 3]), max_length=16)
    assert [item["labels"] for item in features] == [
        [-100, -100, -100, 11, 2],
        [-100, -100, -100, 13, 2],
        [-100, -100, -100, 14, 2],
    ]
    assert all(item["input_ids"][:3] == [1, 20, 21] for item in features)


def test_selection_prefers_higher_macro_f1() -> None:
    records = [
        {"epoch": 1, "validation": {"macro_f1": 0.81}},
        {"epoch": 2, "validation": {"macro_f1": 0.87}},
    ]
    assert select_checkpoint(records)["epoch"] == 2


def test_selection_tie_breaks_to_the_earlier_epoch() -> None:
    records = [
        {"epoch": 1, "validation": {"macro_f1": 0.87}},
        {"epoch": 2, "validation": {"macro_f1": 0.87}},
    ]
    assert select_checkpoint(records)["epoch"] == 1


def test_selection_rejects_empty_and_duplicate_records() -> None:
    with pytest.raises(ValueError, match="no checkpoint records"):
        select_checkpoint([])
    with pytest.raises(ValueError, match="duplicate epoch"):
        select_checkpoint(
            [
                {"epoch": 1, "validation": {"macro_f1": 0.5}},
                {"epoch": 1, "validation": {"macro_f1": 0.6}},
            ]
        )
