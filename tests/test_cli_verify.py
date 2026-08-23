from __future__ import annotations

import json
import sys

import pytest

from loraforge import cli
from loraforge.data import DatasetBundle, Example, Split
from loraforge.intervals import INTERVALS_REPORT


def one_row_split(name: str, label: int) -> Split:
    return Split(
        name,
        (Example(row_id=f"{name}-row", text=f"{name} article", label=label, source_index=0),),
    )


def write_report_markers(root, *, with_test: bool) -> None:
    outputs = root / "outputs"
    outputs.mkdir()
    (outputs / "training-report.json").write_text(
        json.dumps({"config": {"data": {}}}), encoding="utf-8"
    )
    if with_test:
        (outputs / "final-test-report.json").write_text("{}", encoding="utf-8")
        (root / INTERVALS_REPORT).write_text("{}", encoding="utf-8")


def test_verify_loads_the_pinned_dataset_once_for_every_report(
    tmp_path, monkeypatch, capsys
) -> None:
    write_report_markers(tmp_path, with_test=True)
    bundle = DatasetBundle(
        train=one_row_split("train", 0),
        validation=one_row_split("validation", 1),
        test=one_row_split("test", 2),
    )
    loads = []
    received = {}

    def fake_load_dataset(*, allow_test, config):
        loads.append((allow_test, config))
        return bundle

    def fake_training(report, *, root, labels, verify_adapters):
        received["training"] = (root, labels, verify_adapters)
        return {"epoch": 2}

    def fake_final(*, root, validation_labels, test_split):
        received["final"] = (root, validation_labels, test_split)
        return {"verified": True}

    def fake_intervals(*, root, labels):
        received["intervals"] = (root, labels)
        return {"verified": True}

    monkeypatch.setattr(cli, "load_dataset", fake_load_dataset)
    monkeypatch.setattr("loraforge.selection.verify_training_report", fake_training)
    monkeypatch.setattr("loraforge.final_test.verify_final_report", fake_final)
    monkeypatch.setattr("loraforge.intervals.verify_intervals", fake_intervals)
    monkeypatch.setattr(
        sys, "argv", ["loraforge", "verify", "--root", str(tmp_path), "--reports-only"]
    )

    assert cli.main() == 0
    output = json.loads(capsys.readouterr().out)

    assert len(loads) == 1
    assert loads[0][0] is True
    assert received["training"] == (tmp_path, [1], False)
    assert received["final"] == (tmp_path, [1], bundle.test)
    assert received["intervals"] == (tmp_path, [2])
    assert set(output) == {"training_report", "final_test_report", "test_intervals"}


def test_training_only_verification_keeps_the_test_split_locked(
    tmp_path, monkeypatch, capsys
) -> None:
    write_report_markers(tmp_path, with_test=True)
    bundle = DatasetBundle(
        train=one_row_split("train", 0),
        validation=one_row_split("validation", 1),
    )
    allowed = []

    def fake_load_dataset(*, allow_test, config):
        allowed.append(allow_test)
        return bundle

    monkeypatch.setattr(cli, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(
        "loraforge.selection.verify_training_report",
        lambda report, *, root, labels, verify_adapters: {"epoch": 1},
    )
    monkeypatch.setattr(
        "loraforge.final_test.verify_final_report",
        lambda **kwargs: pytest.fail("training-only verification reached final evidence"),
    )
    monkeypatch.setattr(
        "loraforge.intervals.verify_intervals",
        lambda **kwargs: pytest.fail("training-only verification reached test intervals"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["loraforge", "verify", "--root", str(tmp_path), "--training-only"],
    )

    assert cli.main() == 0
    output = json.loads(capsys.readouterr().out)
    assert allowed == [False]
    assert output == {
        "training_report": {
            "verified": True,
            "selected_epoch": 1,
            "adapter_files_verified": True,
        }
    }
