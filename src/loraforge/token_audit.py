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


LEGACY_DEFAULT_AUDIT_SHA256 = (
    "908a384ead2f0a6cbaea8f5d0a5e06725ae78a83898bb01e7ecdc626b19ea47f"
)
LEGACY_DEFAULT_CONFIG_SHA256 = (
    "ecdd884bd3cc1fce6216d1363296e8d9a421ce8412a40d96ce50f7c90a8eb65c"
)


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


def _split_block(split: Split, lengths: list[int], max_length: int) -> dict[str, Any]:
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


def _token_lengths(tokenizer: Any, bundle: DatasetBundle) -> dict[str, list[int]]:
    if bundle.test is not None:
        raise ValueError("token-length audits must not load the publisher test split")
    return {
        split.name: [
            len(prompt_token_ids(tokenizer, item.text)) + 2
            for item in split.examples
        ]
        for split in (bundle.train, bundle.validation)
    }


def _build_current_audit(
    tokenizer: Any,
    bundle: DatasetBundle,
    config: ExperimentConfig,
    lengths_by_split: dict[str, list[int]],
) -> dict[str, Any]:
    max_length = config.training.max_sequence_length
    split_blocks = {
        split.name: _split_block(split, lengths_by_split[split.name], max_length)
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


def _legacy_default_projection(
    stored: dict[str, Any],
    current: dict[str, Any],
    lengths_by_split: dict[str, list[int]],
) -> dict[str, Any]:
    lengths = [
        *lengths_by_split["train"],
        *lengths_by_split["validation"],
    ]
    max_length = current["max_sequence_length"]
    return {
        "schema_version": 1,
        "created_at_utc": stored["created_at_utc"],
        "model_tokenizer": current["model_tokenizer"],
        "model_revision": current["model_revision"],
        "development_rows": len(lengths),
        "test_loaded": False,
        "prompt_plus_answer_tokens": {
            "median": median(lengths),
            "p95": _nearest_rank(lengths, 95),
            "maximum": max(lengths),
            "rows_over_384": sum(length > 384 for length in lengths),
            f"rows_over_{max_length}": sum(length > max_length for length in lengths),
        },
        "decision": (
            f"Use max_sequence_length={max_length} so no development article is "
            "silently truncated."
        ),
        "class_code_token_ids": current["class_code_token_ids"],
    }


def build_token_length_audit(
    tokenizer: Any,
    bundle: DatasetBundle,
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Audit every train/validation row without loading publisher-test content."""
    lengths_by_split = _token_lengths(tokenizer, bundle)
    return _build_current_audit(tokenizer, bundle, config, lengths_by_split)


def verify_token_length_audit(
    stored: dict[str, Any],
    tokenizer: Any,
    bundle: DatasetBundle,
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Recompute a token audit and reject any edited or stale field."""
    lengths_by_split = _token_lengths(tokenizer, bundle)
    recomputed = _build_current_audit(tokenizer, bundle, config, lengths_by_split)
    if "config_sha256" in stored:
        expected = recomputed
    else:
        if _sha256_json(stored) != LEGACY_DEFAULT_AUDIT_SHA256:
            raise EvidenceError(
                "legacy default token-length audit does not match its canonical digest"
            )
        if recomputed["config_sha256"] != LEGACY_DEFAULT_CONFIG_SHA256:
            raise EvidenceError(
                "legacy default token-length audit requires the frozen default config"
            )
        expected = _legacy_default_projection(stored, recomputed, lengths_by_split)
    if stored != expected:
        raise EvidenceError(
            "stored token-length audit does not match the pinned config, rows, and tokenizer"
        )
    if not recomputed["safe_to_train_without_truncation"]:
        raise EvidenceError(
            f"{recomputed['rows_over_max_sequence_length']} development rows exceed "
            f"max_sequence_length={config.training.max_sequence_length}"
        )
    return recomputed
