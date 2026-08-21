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
    prompt_token_ids,
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


class BatchEncodingTokenizer(FakeTokenizer):
    """Newer transformers returns a dict-like BatchEncoding, not a list of IDs."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        if not tokenize:
            return "<chat>"
        return {"input_ids": [1, 20, 21], "attention_mask": [1, 1, 1]}


class BatchOfOneTokenizer(FakeTokenizer):
    """Some versions additionally nest the single prompt in a batch dimension."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        if not tokenize:
            return "<chat>"
        return {"input_ids": [[1, 20, 21]], "attention_mask": [[1, 1, 1]]}


@pytest.mark.parametrize(
    "tokenizer", [FakeTokenizer(), BatchEncodingTokenizer(), BatchOfOneTokenizer()]
)
def test_prompt_ids_are_flat_whatever_the_tokenizer_returns(tokenizer) -> None:
    assert prompt_token_ids(tokenizer, "story") == [1, 20, 21]


@pytest.mark.parametrize(
    "tokenizer", [FakeTokenizer(), BatchEncodingTokenizer(), BatchOfOneTokenizer()]
)
def test_supervised_encoding_survives_every_chat_template_return_shape(tokenizer) -> None:
    encoded = encode_supervised_example(tokenizer, "story", 2, max_length=8)
    assert encoded["input_ids"] == [1, 20, 21, 13, 2]
    assert encoded["labels"] == [-100, -100, -100, 13, 2]


def test_oversized_prompt_is_rejected_rather_than_silently_truncated() -> None:
    class LongTokenizer(FakeTokenizer):
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            if not tokenize:
                return "<chat>"
            return {"input_ids": list(range(600))}

    with pytest.raises(ValueError, match="exceeding frozen max"):
        encode_supervised_example(LongTokenizer(), "story", 0, max_length=512)


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
    labels = rng.integers(0, 4, 200).tolist()
    # Logits must carry signal about the labels, or the NLL-minimizing temperature
    # is unbounded: flattening a random predictor forever keeps improving its NLL.
    logits = rng.normal(scale=1.5, size=(200, 4))
    logits[np.arange(200), np.asarray(labels)] += 3.0
    temperature = fit_temperature(logits, labels)
    assert 0.05 < temperature < 10.0
    assert np.array_equal(softmax(logits).argmax(1), softmax(logits / temperature).argmax(1))


def test_a_temperature_search_that_hits_its_own_wall_is_refused() -> None:
    """Returning the boundary would report a failed search as a fitted value."""
    rng = np.random.default_rng(4)
    labels = rng.integers(0, 4, 200).tolist()
    uninformative = rng.normal(scale=200.0, size=(200, 4))  # optimum far above the ceiling
    with pytest.raises(ValueError, match="converged to its boundary"):
        fit_temperature(uninformative, labels)
