from __future__ import annotations

import numpy as np
import pytest

from loraforge.metrics import (
    evaluate_predictions,
    evaluation_block,
    expected_calibration_error,
    fit_temperature,
    negative_log_likelihood,
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


@pytest.mark.parametrize("label", [True, False, 1.0, "1"])
def test_prompt_construction_rejects_non_integer_labels(label) -> None:
    with pytest.raises(ValueError, match="label must be an integer"):
        training_messages("synthetic article", label)
    with pytest.raises(ValueError, match="label must be an integer"):
        encode_supervised_example(FakeTokenizer(), "synthetic article", label)


def test_class_codes_are_unique_single_tokens() -> None:
    assert class_code_token_ids(FakeTokenizer()) == (11, 12, 13, 14)


def test_multi_token_code_is_rejected() -> None:
    class BrokenTokenizer(FakeTokenizer):
        def encode(self, value, add_special_tokens=False):
            result = super().encode(value, add_special_tokens=add_special_tokens)
            return result + [15] if value == "<chat>D" else result

    with pytest.raises(ValueError, match="not one contextual token"):
        class_code_token_ids(BrokenTokenizer())


def test_class_code_context_matches_the_tokenized_scoring_prompt() -> None:
    class DriftedTokenizer(FakeTokenizer):
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            if not tokenize:
                return "<chat>"
            return [99, 20, 21]

    with pytest.raises(ValueError, match="different token context"):
        class_code_token_ids(DriftedTokenizer())


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


@pytest.mark.parametrize("max_length", [True, 0, -1, 8.0])
def test_supervised_encoding_requires_a_positive_integer_max_length(max_length) -> None:
    with pytest.raises(ValueError, match="max_length must be a positive integer"):
        encode_supervised_example(FakeTokenizer(), "story", 0, max_length=max_length)


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


def test_nll_does_not_cap_extreme_finite_logit_gaps() -> None:
    logits = np.array([[1000.0, -1000.0, -1000.0, -1000.0]])
    assert negative_log_likelihood(logits, [1]) == pytest.approx(2000.0)

    offset_logits = np.full((1, 4), 1e300)
    assert negative_log_likelihood(offset_logits, [0]) == pytest.approx(np.log(4.0))


def test_ece_zero_when_confidence_matches_accuracy() -> None:
    probabilities = np.tile([0.7, 0.1, 0.1, 0.1], (100, 1))
    labels = [0] * 70 + [1] * 30
    assert expected_calibration_error(probabilities, labels, n_bins=10)["ece"] == pytest.approx(0)


@pytest.mark.parametrize("n_bins", [0, -1, True, 1.5])
def test_ece_requires_a_positive_integer_bin_count(n_bins) -> None:
    probabilities = np.array([[0.7, 0.1, 0.1, 0.1]])
    with pytest.raises(ValueError, match="n_bins must be a positive integer"):
        expected_calibration_error(probabilities, [0], n_bins=n_bins)


@pytest.mark.parametrize(
    "probabilities",
    [
        np.array([[1.1, -0.1, 0.0, 0.0]]),
        np.array([[np.nan, 0.0, 0.0, 1.0]]),
    ],
)
def test_ece_rejects_invalid_probabilities(probabilities) -> None:
    with pytest.raises(ValueError, match="probabilities must"):
        expected_calibration_error(probabilities, [0])


def test_metrics_reject_invalid_labels_and_logit_shapes() -> None:
    with pytest.raises(ValueError, match="class IDs 0 through 3"):
        evaluation_block(np.zeros((1, 4)), [-1])
    with pytest.raises(ValueError, match=r"shape \[rows, 4\]"):
        evaluation_block(np.zeros((1, 5)), [0])
    with pytest.raises(ValueError, match="finite values"):
        evaluation_block(np.array([[np.inf, 0.0, 0.0, 0.0]]), [0])


def test_invalid_predictions_remain_measured_not_silently_dropped() -> None:
    report = evaluate_predictions([0, 1], [0, 9])
    assert report["accuracy"] == pytest.approx(0.5)
    assert report["invalid_prediction_rate"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"low": 0.0},
        {"low": 2.0, "high": 1.0},
        {"tolerance": 0.0},
        {"tolerance": -1.0},
        {"low": 1.0, "high": 2.0, "tolerance": 1.0},
    ],
)
def test_temperature_search_rejects_invalid_bounds(kwargs) -> None:
    logits = np.eye(4, dtype=float)
    with pytest.raises(ValueError, match="temperature search"):
        fit_temperature(logits, [0, 1, 2, 3], **kwargs)


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
