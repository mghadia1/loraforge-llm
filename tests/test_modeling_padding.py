from __future__ import annotations

import pytest

from loraforge.modeling import _left_pad_token_rows, resolve_last_logit_kwargs


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
