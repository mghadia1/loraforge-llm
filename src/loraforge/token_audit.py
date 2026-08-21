"""Deterministic tokenizer-length evidence for a frozen development split."""

from __future__ import annotations

import hashlib
import json
from statistics import median
from typing import Any

from .config import ExperimentConfig
from .data import CLASS_CODES, DatasetBundle, Split
from .prompts import class_code_token_ids, prompt_token_ids
from .provenance import EvidenceError


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _nearest_rank(values: list[int], percentile: int) -> int:
    """Return an integer percentile without library-version interpolation drift."""
    if not values:
        raise ValueError("cannot summarize an empty token-length collection")
    ordered = sorted(values)
    rank = max(1, (percentile * len(ordered) + 99) // 100)
    return ordered[rank - 1]


def _split_block(tokenizer: Any, split: Split, max_length: int) -> dict[str, Any]:
    lengths = [len(prompt_token_ids(tokenizer, item.text)) + 2 for item in split.examples]
    if not lengths:
        raise ValueError(f"cannot audit empty {split.name} split")
    length_rows = [
        {"row_id": item.row_id, "prompt_plus_answer_tokens": length}
        for item, length in zip(split.examples, lengths, strict=True)
    ]
    return {
        "rows": len(lengths),
        "row_ids_sha256": split.id_sha256(),
        "token_lengths_sha256": _sha256_json(length_rows),
        "prompt_plus_answer_tokens": {
            "minimum": min(lengths),
            "median": median(lengths),
            "p95_nearest_rank": _nearest_rank(lengths, 95),
            "maximum": max(lengths),
            "rows_over_max_sequence_length": sum(length > max_length for length in lengths),
        },
    }


def build_token_length_audit(
    tokenizer: Any,
    bundle: DatasetBundle,
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Audit every train/validation row without loading publisher-test content."""
    if bundle.test is not None:
        raise ValueError("token-length audits must not load the publisher test split")

    max_length = config.training.max_sequence_length
    split_blocks = {
        split.name: _split_block(tokenizer, split, max_length)
        for split in (bundle.train, bundle.validation)
    }
    rows_over_limit = sum(
        block["prompt_plus_answer_tokens"]["rows_over_max_sequence_length"]
        for block in split_blocks.values()
    )
    code_ids = class_code_token_ids(tokenizer)
    return {
        "schema_version": 1,
        "config_sha256": _sha256_json(config.to_dict()),
        "dataset": config.data.dataset_name,
        "dataset_revision": config.data.dataset_revision,
        "model_tokenizer": config.model_name,
        "model_revision": config.model_revision,
        "max_sequence_length": max_length,
        "test_loaded": False,
        "class_code_token_ids": dict(zip(CLASS_CODES, code_ids, strict=True)),
        "splits": split_blocks,
        "development_rows": sum(block["rows"] for block in split_blocks.values()),
        "rows_over_max_sequence_length": rows_over_limit,
        "safe_to_train_without_truncation": rows_over_limit == 0,
    }


def verify_token_length_audit(
    stored: dict[str, Any],
    tokenizer: Any,
    bundle: DatasetBundle,
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Recompute a token audit and reject any edited or stale field."""
    recomputed = build_token_length_audit(tokenizer, bundle, config)
    if stored != recomputed:
        raise EvidenceError(
            "stored token-length audit does not match the pinned config, rows, and tokenizer"
        )
    if not recomputed["safe_to_train_without_truncation"]:
        raise EvidenceError(
            f"{recomputed['rows_over_max_sequence_length']} development rows exceed "
            f"max_sequence_length={config.training.max_sequence_length}"
        )
    return recomputed
