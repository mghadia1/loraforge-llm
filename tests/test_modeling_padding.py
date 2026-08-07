from __future__ import annotations

import pytest

from loraforge.modeling import _left_pad_token_rows


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
