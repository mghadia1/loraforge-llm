from __future__ import annotations

import numpy as np
import pytest

from loraforge.metrics import (
    evaluation_block,
    expected_calibration_error,
    fit_temperature,
    softmax,
)
from loraforge.prompts import (
    class_code_token_ids,
    encode_supervised_example,
    inference_messages,
    parse_code,
    training_messages,
)


class FakeTokenizer:
    eos_token_id = 2

    def encode(self, value, add_special_tokens=False):
        mapping = {
            "<chat>": [1, 20, 21],
            "<chat>A": [1, 20, 21, 11],
            "<chat>B": [1, 20, 21, 12],
            "<chat>C": [1, 20, 21, 13],
            "<chat>D": [1, 20, 21, 14],
        }
        return mapping[value]

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert add_generation_prompt is True
        return [1, 20, 21] if tokenize else "<chat>"


def test_prompt_has_no_gold_label_in_user_message() -> None:
    messages = training_messages("Stocks rose after earnings.", 2)
    assert messages[-1] == {"role": "assistant", "content": "C"}
    assert "Business" not in messages[1]["content"]
    assert inference_messages("a story") == messages[:2] or len(messages[:2]) == 2


def test_class_codes_are_unique_single_tokens() -> None:
    assert class_code_token_ids(FakeTokenizer()) == (11, 12, 13, 14)


def test_multi_token_code_is_rejected() -> None:
    class BrokenTokenizer(FakeTokenizer):
        def encode(self, value, add_special_tokens=False):
            result = super().encode(value, add_special_tokens=add_special_tokens)
            return result + [15] if value == "<chat>D" else result

    with pytest.raises(ValueError, match="not one contextual token"):
        class_code_token_ids(BrokenTokenizer())


def test_supervised_encoding_masks_prompt_and_trains_only_answer() -> None:
    encoded = encode_supervised_example(FakeTokenizer(), "story", 2, max_length=8)
    assert encoded["input_ids"] == [1, 20, 21, 13, 2]
    assert encoded["labels"] == [-100, -100, -100, 13, 2]
    assert encoded["attention_mask"] == [1] * 5


def test_code_parser_is_exact_not_fuzzy() -> None:
    assert parse_code(" A ") == 0
    assert parse_code("`d`") == 3
    assert parse_code("The answer is A") is None


def test_metric_math_has_known_answer() -> None:
    logits = np.array(
        [[10, 0, 0, 0], [0, 10, 0, 0], [0, 0, 10, 0], [0, 0, 0, 10]],
        dtype=float,
    )
    report = evaluation_block(logits, [0, 1, 2, 3])
    assert report["macro_f1"] == 1.0
    assert report["accuracy"] == 1.0
    assert report["calibration"]["ece"] < 0.001


def test_ece_zero_when_confidence_matches_accuracy() -> None:
    probabilities = np.tile([0.7, 0.1, 0.1, 0.1], (100, 1))
    labels = [0] * 70 + [1] * 30
    assert expected_calibration_error(probabilities, labels, n_bins=10)["ece"] == pytest.approx(0)


def test_temperature_scaling_does_not_change_argmax() -> None:
    rng = np.random.default_rng(4)
    logits = rng.normal(size=(100, 4))
    temperature = fit_temperature(logits, rng.integers(0, 4, 100).tolist())
    assert np.array_equal(softmax(logits).argmax(1), softmax(logits / temperature).argmax(1))
