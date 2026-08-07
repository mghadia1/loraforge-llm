"""One frozen prompt contract shared by baseline and tuned evaluation."""

from __future__ import annotations

from typing import Any

from .data import CLASS_CODES, CLASS_NAMES, CODE_TO_LABEL, LABEL_TO_CODE


LABEL_MENU = "; ".join(
    f"{code}={name}" for code, name in zip(CLASS_CODES, CLASS_NAMES)
)
SYSTEM_PROMPT = (
    "Classify the news article into exactly one topic. "
    f"Allowed codes: {LABEL_MENU}. "
    "Return one code only: A, B, C, or D. Do not explain."
)


def inference_messages(text: str) -> list[dict[str, str]]:
    cleaned = " ".join(text.split())
    if not cleaned:
        raise ValueError("article text cannot be empty")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Article:\n{cleaned}\n\nTopic code:"},
    ]


def training_messages(text: str, label: int) -> list[dict[str, str]]:
    if label not in LABEL_TO_CODE:
        raise ValueError(f"label must be 0-{len(CLASS_NAMES) - 1}")
    return inference_messages(text) + [
        {"role": "assistant", "content": LABEL_TO_CODE[label]}
    ]


def render_inference_prompt(tokenizer: Any, text: str) -> str:
    return tokenizer.apply_chat_template(
        inference_messages(text), tokenize=False, add_generation_prompt=True
    )


def render_training_text(tokenizer: Any, text: str, label: int) -> str:
    return tokenizer.apply_chat_template(
        training_messages(text, label), tokenize=False, add_generation_prompt=False
    )


def prompt_token_ids(tokenizer: Any, text: str) -> list[int]:
    """Return the rendered prompt as flat token IDs.

    `apply_chat_template(tokenize=True)` returns a plain list on older
    transformers and a `BatchEncoding` (and sometimes a batch-of-one nesting) on
    newer ones. Normalizing here keeps one prompt contract for training,
    baseline scoring, and the final evaluation.
    """
    encoded = tokenizer.apply_chat_template(
        inference_messages(text), tokenize=True, add_generation_prompt=True
    )
    ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    if len(ids) and isinstance(ids[0], (list, tuple)):
        if len(ids) != 1:
            raise ValueError(f"expected one prompt, tokenizer returned {len(ids)}")
        ids = ids[0]
    flat = [int(value) for value in ids]
    if not flat:
        raise ValueError("tokenizer produced an empty prompt")
    return flat


def class_code_token_ids(tokenizer: Any) -> tuple[int, ...]:
    probe_prompt = render_inference_prompt(tokenizer, "Tokenizer contract probe.")
    prompt_ids = tokenizer.encode(probe_prompt, add_special_tokens=False)
    ids = []
    for code in CLASS_CODES:
        encoded = tokenizer.encode(probe_prompt + code, add_special_tokens=False)
        if len(encoded) != len(prompt_ids) + 1 or encoded[:-1] != prompt_ids:
            raise ValueError(f"class code {code!r} is not one contextual token")
        ids.append(int(encoded[-1]))
    if len(ids) != len(set(ids)):
        raise ValueError("class codes do not map to unique tokens")
    return tuple(ids)


def encode_supervised_example(
    tokenizer: Any, text: str, label: int, *, max_length: int = 512
) -> dict[str, list[int]]:
    """Tokenize one row while masking every non-answer token from SFT loss."""
    if label not in LABEL_TO_CODE:
        raise ValueError(f"label must be 0-{len(CLASS_NAMES) - 1}")
    prompt_ids = prompt_token_ids(tokenizer, text)
    answer_id = class_code_token_ids(tokenizer)[label]
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("tokenizer has no EOS token")
    input_ids = [*prompt_ids, answer_id, int(eos_id)]
    if len(input_ids) > max_length:
        raise ValueError(
            f"formatted example has {len(input_ids)} tokens, exceeding frozen max {max_length}"
        )
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + [answer_id, int(eos_id)],
    }


def parse_code(value: str) -> int | None:
    normalized = value.strip().strip('"\'`').upper()
    return CODE_TO_LABEL.get(normalized)
