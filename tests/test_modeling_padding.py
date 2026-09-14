from __future__ import annotations

import pytest

from loraforge.modeling import (
    _left_pad_token_rows,
    resolve_last_logit_kwargs,
    score_class_codes,
    validate_model_tokenizer_vocabulary,
)


def test_left_padding_builds_plain_rectangular_model_inputs() -> None:
    padded = _left_pad_token_rows([[1, 20, 21], [1, 30]], pad_token_id=2)

    assert padded == {
        "input_ids": [[1, 20, 21], [2, 1, 30]],
        "attention_mask": [[1, 1, 1], [0, 1, 1]],
    }
    assert all(type(row) is list for row in padded["input_ids"])


def test_left_padding_rejects_an_empty_batch() -> None:
    with pytest.raises(ValueError, match="empty prompt batch"):
        _left_pad_token_rows([], pad_token_id=2)


@pytest.mark.parametrize("batch_size", [True, 0, -1, 1.5])
def test_scoring_requires_a_positive_integer_batch_size(batch_size) -> None:
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        score_class_codes(None, None, ["synthetic article"], batch_size=batch_size)


@pytest.mark.parametrize("max_length", [True, 0, -1, 512.0])
def test_scoring_requires_a_positive_integer_max_length(max_length) -> None:
    with pytest.raises(ValueError, match="max_length must be a positive integer"):
        score_class_codes(None, None, ["synthetic article"], max_length=max_length)


def test_scoring_rejects_an_empty_collection_before_importing_torch() -> None:
    with pytest.raises(ValueError, match="empty text collection"):
        score_class_codes(None, None, [])


class ModernCausalLM:
    def forward(self, input_ids=None, attention_mask=None, logits_to_keep=0):
        raise NotImplementedError


class LegacyCausalLM:
    def forward(self, input_ids=None, attention_mask=None, num_logits_to_keep=0):
        raise NotImplementedError


class OldCausalLM:
    def forward(self, input_ids=None, attention_mask=None):
        raise NotImplementedError


class PeftWrapper:
    """PEFT hides the real signature behind pass-through kwargs, as here."""

    def __init__(self, inner):
        self.base_model = inner

    def forward(self, *args, **kwargs):
        raise NotImplementedError


@pytest.mark.parametrize(
    ("inner", "expected"),
    [
        (ModernCausalLM(), {"logits_to_keep": 1}),
        (LegacyCausalLM(), {"num_logits_to_keep": 1}),
        (OldCausalLM(), {}),
    ],
)
def test_last_logit_kwarg_is_found_through_the_peft_wrapper(inner, expected) -> None:
    assert resolve_last_logit_kwargs(inner) == expected
    assert resolve_last_logit_kwargs(PeftWrapper(inner)) == expected


def test_unknown_model_shape_falls_back_to_computing_every_logit() -> None:
    class Cyclic:
        def forward(self, *args, **kwargs):
            raise NotImplementedError

    cyclic = Cyclic()
    cyclic.base_model = cyclic  # must terminate rather than loop
    assert resolve_last_logit_kwargs(cyclic) == {}
    assert resolve_last_logit_kwargs(object()) == {}


class VocabularyTokenizer:
    eos_token_id = 2
    pad_token_id = 2

    def __init__(self, size=15, code_ids=(11, 12, 13, 14)):
        self.size = size
        self.code_ids = dict(zip("ABCD", code_ids))

    def __len__(self):
        return self.size

    def encode(self, value, add_special_tokens=False):
        prompt = [1, 7, 8]
        if value[-1:] in self.code_ids:
            return [*prompt, self.code_ids[value[-1]]]
        return prompt

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return [1, 7, 8] if tokenize else "<chat>"


class VocabularyModel:
    def __init__(self, *, config_size=15, input_size=15, output_size=15):
        self.config = type("Config", (), {"vocab_size": config_size})()
        self.input_embeddings = type("Input", (), {"num_embeddings": input_size})()
        self.output_embeddings = type("Output", (), {"out_features": output_size})()

    def get_input_embeddings(self):
        return self.input_embeddings

    def get_output_embeddings(self):
        return self.output_embeddings


def test_model_and_tokenizer_vocabulary_contract_accepts_aligned_ids() -> None:
    assert validate_model_tokenizer_vocabulary(
        VocabularyModel(), VocabularyTokenizer()
    ) == (11, 12, 13, 14)


@pytest.mark.parametrize(
    ("model", "tokenizer", "message"),
    [
        (VocabularyModel(input_size=14), VocabularyTokenizer(), "vocabulary mismatch"),
        (VocabularyModel(output_size=14), VocabularyTokenizer(), "vocabulary mismatch"),
        (VocabularyModel(config_size=14), VocabularyTokenizer(), "vocabulary mismatch"),
        (VocabularyModel(), VocabularyTokenizer(size=16), "vocabulary mismatch"),
        (
            VocabularyModel(),
            VocabularyTokenizer(code_ids=(11, 12, 13, 15)),
            "class-code token IDs",
        ),
    ],
)
def test_model_tokenizer_vocabulary_drift_is_rejected(model, tokenizer, message) -> None:
    with pytest.raises(ValueError, match=message):
        validate_model_tokenizer_vocabulary(model, tokenizer)
