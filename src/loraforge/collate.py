"""Tokenized supervised dataset and the padding collator used for QLoRA training."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .data import Split
from .prompts import encode_supervised_example


LABEL_IGNORE_INDEX = -100


def build_supervised_features(
    tokenizer: Any, split: Split, *, max_length: int = 512
) -> list[dict[str, list[int]]]:
    """Encode a whole split; every prompt token stays masked out of the loss."""
    features = []
    for item in split.examples:
        encoded = encode_supervised_example(
            tokenizer, item.text, item.label, max_length=max_length
        )
        if sum(label != LABEL_IGNORE_INDEX for label in encoded["labels"]) != 2:
            raise ValueError(
                f"row {item.row_id} supervises {sum(label != LABEL_IGNORE_INDEX for label in encoded['labels'])} "
                "tokens; exactly the answer code and EOS must be supervised"
            )
        features.append(encoded)
    return features


def pad_batch(
    features: Sequence[dict[str, list[int]]], *, pad_token_id: int
) -> dict[str, list[list[int]]]:
    """Right-pad a batch: inputs with the pad ID, labels with -100, mask with 0."""
    if not features:
        raise ValueError("cannot collate an empty batch")
    width = max(len(item["input_ids"]) for item in features)
    batch: dict[str, list[list[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
    for item in features:
        length = len(item["input_ids"])
        if len(item["labels"]) != length or len(item["attention_mask"]) != length:
            raise ValueError("input_ids, attention_mask, and labels must be the same length")
        padding = width - length
        batch["input_ids"].append([*item["input_ids"], *[pad_token_id] * padding])
        batch["attention_mask"].append([*item["attention_mask"], *[0] * padding])
        batch["labels"].append([*item["labels"], *[LABEL_IGNORE_INDEX] * padding])
    return batch


class SupervisedCollator:
    """Torch collator wrapping :func:`pad_batch`; padded labels never reach the loss."""

    def __init__(self, pad_token_id: int):
        if pad_token_id is None:
            raise ValueError("tokenizer must expose a pad token id")
        self.pad_token_id = int(pad_token_id)

    def __call__(self, features: Sequence[dict[str, list[int]]]):
        import torch

        batch = pad_batch(features, pad_token_id=self.pad_token_id)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def supervised_loss_reference(logits: np.ndarray, labels: Sequence[int]) -> float:
    """NumPy copy of the causal-LM shifted cross-entropy, for masking tests only.

    Mirrors what Hugging Face computes: position ``t`` predicts token ``t + 1``, and
    ``-100`` labels are dropped. Used to prove prompt logits cannot move the loss.
    """
    values = np.asarray(logits, dtype=float)
    if values.ndim != 2 or len(values) != len(labels):
        raise ValueError("logits must be [sequence, vocabulary] aligned with labels")
    targets = np.asarray(labels, dtype=int)
    shifted_logits = values[:-1]
    shifted_targets = targets[1:]
    keep = shifted_targets != LABEL_IGNORE_INDEX
    if not keep.any():
        raise ValueError("no supervised tokens in this sequence")
    selected = shifted_logits[keep]
    stabilized = selected - selected.max(axis=-1, keepdims=True)
    log_probabilities = stabilized - np.log(np.exp(stabilized).sum(axis=-1, keepdims=True))
    chosen = log_probabilities[np.arange(len(selected)), shifted_targets[keep]]
    return float(-chosen.mean())
