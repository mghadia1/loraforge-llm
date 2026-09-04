from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

import loraforge.token_audit as token_audit_module
from loraforge.config import default_config
from loraforge.data import DatasetBundle, Example, Split
from loraforge.provenance import EvidenceError
from loraforge.token_audit import build_token_length_audit, verify_token_length_audit


class VariableLengthTokenizer:
    eos_token_id = 2

    def encode(self, value, add_special_tokens=False):
        codes = {"A": 11, "B": 12, "C": 13, "D": 14}
        code_id = codes.get(value[-1])
        prompt = value[:-1] if code_id is not None else value
        words = int(prompt.removeprefix("<chat:").removesuffix(">"))
        ids = [1, 20, 21, *range(30, 30 + words)]
        return [*ids, code_id] if code_id is not None else ids

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        article = messages[-1]["content"].removeprefix("Article:\n").removesuffix(
            "\n\nTopic code:"
        )
        words = len(article.split())
        return [1, 20, 21, *range(30, 30 + words)] if tokenize else f"<chat:{words}>"


def make_split(name: str, word_counts: list[int], offset: int = 0) -> Split:
    return Split(
        name,
        tuple(
            Example(
                row_id=f"{name}-{offset + index}",
                text=" ".join(["word"] * count),
                label=index % 4,
                source_index=offset + index,
            )
            for index, count in enumerate(word_counts)
        ),
    )


def test_token_audit_binds_lengths_to_rows_and_frozen_inputs() -> None:
    config = replace(
        default_config(),
        training=replace(default_config().training, max_sequence_length=10),
    )
    bundle = DatasetBundle(
        train=make_split("train", [1, 2, 3]),
        validation=make_split("validation", [4, 5], offset=3),
    )

    first = build_token_length_audit(VariableLengthTokenizer(), bundle, config)
    second = build_token_length_audit(VariableLengthTokenizer(), bundle, config)

    assert first == second
    assert first["test_loaded"] is False
    assert first["development_rows"] == 5
    assert first["class_code_token_ids"] == {"A": 11, "B": 12, "C": 13, "D": 14}
    assert first["splits"]["train"]["row_ids_sha256"] == bundle.train.id_sha256()
    assert first["splits"]["train"]["prompt_plus_answer_tokens"] == {
        "minimum": 6,
        "median": 7,
        "p95_nearest_rank": 8,
        "maximum": 8,
        "rows_over_max_sequence_length": 0,
    }
    assert first["safe_to_train_without_truncation"] is True


def test_token_audit_reports_rows_that_would_exceed_the_frozen_limit() -> None:
    config = replace(
        default_config(),
        training=replace(default_config().training, max_sequence_length=7),
    )
    bundle = DatasetBundle(
        train=make_split("train", [1, 4]),
        validation=make_split("validation", [1], offset=2),
    )

    audit = build_token_length_audit(VariableLengthTokenizer(), bundle, config)

    assert audit["rows_over_max_sequence_length"] == 1
    assert audit["safe_to_train_without_truncation"] is False


def test_token_audit_refuses_a_bundle_that_contains_publisher_test() -> None:
    bundle = DatasetBundle(
        train=make_split("train", [1]),
        validation=make_split("validation", [1], offset=1),
        test=make_split("test", [1], offset=2),
    )

    with pytest.raises(ValueError, match="must not load the publisher test"):
        build_token_length_audit(VariableLengthTokenizer(), bundle, default_config())


def test_token_audit_verifier_rejects_an_edited_summary() -> None:
    config = default_config()
    bundle = DatasetBundle(
        train=make_split("train", [1, 2]),
        validation=make_split("validation", [1], offset=2),
    )
    audit = build_token_length_audit(VariableLengthTokenizer(), bundle, config)
    audit["splits"]["train"]["prompt_plus_answer_tokens"]["maximum"] = 1

    with pytest.raises(EvidenceError, match="does not match the pinned"):
        verify_token_length_audit(audit, VariableLengthTokenizer(), bundle, config)


def test_tracked_legacy_default_audit_is_bound_to_its_canonical_digest() -> None:
    stored = json.loads(
        Path("docs/evidence/token-length-audit.json").read_text(encoding="utf-8")
    )

    assert (
        token_audit_module._sha256_json(stored)
        == token_audit_module.LEGACY_DEFAULT_AUDIT_SHA256
    )
    assert (
        token_audit_module._sha256_json(default_config().to_dict())
        == token_audit_module.LEGACY_DEFAULT_CONFIG_SHA256
    )


def test_legacy_default_audit_is_recomputed_without_rewriting_it(monkeypatch) -> None:
    config = default_config()
    bundle = DatasetBundle(
        train=make_split("train", [1, 2]),
        validation=make_split("validation", [3, 4], offset=2),
    )
    stored = {
        "schema_version": 1,
        "created_at_utc": "2026-01-01T00:00:00Z",
        "model_tokenizer": config.model_name,
        "model_revision": config.model_revision,
        "development_rows": 4,
        "test_loaded": False,
        "prompt_plus_answer_tokens": {
            "median": 7.5,
            "p95": 9,
            "maximum": 9,
            "rows_over_384": 0,
            "rows_over_512": 0,
        },
        "decision": (
            "Use max_sequence_length=512 so no development article is silently truncated."
        ),
        "class_code_token_ids": {"A": 11, "B": 12, "C": 13, "D": 14},
    }
    original = json.loads(json.dumps(stored))
    monkeypatch.setattr(
        token_audit_module,
        "LEGACY_DEFAULT_AUDIT_SHA256",
        token_audit_module._sha256_json(stored),
    )

    verified = verify_token_length_audit(
        stored, VariableLengthTokenizer(), bundle, config
    )

    assert stored == original
    assert verified["development_rows"] == 4
    assert verified["rows_over_max_sequence_length"] == 0
    assert verified["safe_to_train_without_truncation"] is True


def test_edited_legacy_default_audit_is_rejected_by_its_digest(monkeypatch) -> None:
    stored = json.loads(
        Path("docs/evidence/token-length-audit.json").read_text(encoding="utf-8")
    )
    trusted_digest = token_audit_module._sha256_json(stored)
    stored["prompt_plus_answer_tokens"]["maximum"] = 1
    monkeypatch.setattr(
        token_audit_module, "LEGACY_DEFAULT_AUDIT_SHA256", trusted_digest
    )
    bundle = DatasetBundle(
        train=make_split("train", [1]),
        validation=make_split("validation", [1], offset=1),
    )

    with pytest.raises(EvidenceError, match="canonical digest"):
        verify_token_length_audit(
            stored, VariableLengthTokenizer(), bundle, default_config()
        )
