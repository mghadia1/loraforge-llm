from __future__ import annotations

from pathlib import Path

import pytest

from loraforge.config import load_config
from loraforge.provenance import EvidenceError, read_json, write_json


@pytest.mark.parametrize(
    "payload",
    [
        '{"schema_version": 999, "schema_version": 1}',
        '{"nested": {"metric": 999, "metric": 0.5}}',
    ],
)
def test_evidence_reader_rejects_duplicate_object_keys(tmp_path: Path, payload: str) -> None:
    path = tmp_path / "artifact.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(EvidenceError, match="duplicate JSON object key"):
        read_json(path)


def test_experiment_config_rejects_duplicate_protocol_fields(tmp_path: Path) -> None:
    path = tmp_path / "experiment.json"
    path.write_text(
        Path("configs/experiment.json")
        .read_text(encoding="utf-8")
        .replace('"schema_version": 1', '"schema_version": 999, "schema_version": 1'),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_config(path)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_evidence_writer_rejects_non_standard_json_numbers(
    tmp_path: Path, value: float
) -> None:
    path = tmp_path / "artifact.json"

    with pytest.raises(EvidenceError, match="non-standard JSON evidence"):
        write_json({"diagnostic": value}, path)

    assert not path.exists()
