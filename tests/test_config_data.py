from __future__ import annotations

import json

import pytest

import loraforge.data as data_module
from loraforge.config import DataConfig, ExperimentConfig, default_config, load_config
from loraforge.data import (
    CLASS_NAMES,
    DatasetBundle,
    LockedTestSplitError,
    Split,
    deterministic_development_split,
    load_dataset,
)


def synthetic_rows(per_class: int = 8):
    return [
        {"text": f"article {label}-{index}", "label": label}
        for label in range(len(CLASS_NAMES))
        for index in range(per_class)
    ]


def test_frozen_configuration_is_qlora_and_validation_selected() -> None:
    config = default_config()
    assert config.quantization.load_in_4bit is True
    assert config.quantization.quant_type == "nf4"
    assert config.lora.rank == 16
    assert config.training.select_metric == "macro_f1"
    assert config.test_evaluations_allowed == 1
    assert config.resume_eligible is False


def test_invalid_protocol_is_rejected() -> None:
    with pytest.raises(ValueError, match="zero or one"):
        ExperimentConfig(test_evaluations_allowed=2).validate()


def test_development_split_is_balanced_disjoint_and_deterministic() -> None:
    config = DataConfig(train_per_class=4, validation_per_class=2)
    first_train, first_validation = deterministic_development_split(
        synthetic_rows(), config
    )
    second_train, second_validation = deterministic_development_split(
        synthetic_rows(), config
    )
    assert first_train.class_counts() == {name: 4 for name in CLASS_NAMES}
    assert first_validation.class_counts() == {name: 2 for name in CLASS_NAMES}
    assert first_train.id_sha256() == second_train.id_sha256()
    assert first_validation.id_sha256() == second_validation.id_sha256()
    assert {x.row_id for x in first_train.examples}.isdisjoint(
        x.row_id for x in first_validation.examples
    )


def test_test_split_is_locked_by_default() -> None:
    train, validation = deterministic_development_split(
        synthetic_rows(), DataConfig(train_per_class=4, validation_per_class=2)
    )
    bundle = DatasetBundle(train, validation)
    with pytest.raises(LockedTestSplitError, match="one final evaluation"):
        bundle.require_test()


def test_default_development_budget_is_exactly_ten_thousand() -> None:
    config = DataConfig()
    assert len(CLASS_NAMES) * config.train_per_class == 8_000
    assert len(CLASS_NAMES) * config.validation_per_class == 2_000


class FakePublisherSplit:
    def __init__(self, rows: int) -> None:
        self.rows = rows
        self.features = {"label": type("LabelFeature", (), {"names": CLASS_NAMES})()}

    def __len__(self) -> int:
        return self.rows

    def __iter__(self):
        return (
            {"text": f"publisher row {index}", "label": index % len(CLASS_NAMES)}
            for index in range(self.rows)
        )


def test_default_loader_never_requests_the_publisher_test_split(monkeypatch) -> None:
    requested = []

    def fake_load_dataset(name, *, revision, split):
        requested.append((name, revision, split))
        assert split == "train"
        return FakePublisherSplit(120_000)

    empty = Split("empty", ())
    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    monkeypatch.setattr(
        data_module,
        "deterministic_development_split",
        lambda rows, config: (empty, empty),
    )

    bundle = load_dataset(allow_test=False)

    assert bundle.test is None
    assert [split for _, _, split in requested] == ["train"]


def test_authorized_loader_requests_test_only_after_train(monkeypatch) -> None:
    requested = []

    def fake_load_dataset(name, *, revision, split):
        requested.append(split)
        rows = 120_000 if split == "train" else 7_600
        return FakePublisherSplit(rows)

    empty = Split("empty", ())
    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    monkeypatch.setattr(
        data_module,
        "deterministic_development_split",
        lambda rows, config: (empty, empty),
    )

    bundle = load_dataset(allow_test=True)

    assert requested == ["train", "test"]
    assert bundle.require_test().name == "test"
    assert len(bundle.require_test()) == 7_600

def test_expanded_training_keeps_validation_fixed_and_adds_rows() -> None:
    rows = synthetic_rows(per_class=8)
    original_train, original_validation = deterministic_development_split(
        rows, DataConfig(train_per_class=2, validation_per_class=2)
    )
    expanded_train, expanded_validation = deterministic_development_split(
        rows,
        DataConfig(
            train_per_class=6,
            validation_per_class=2,
            validation_start_per_class=2,
        ),
    )

    assert expanded_validation.id_sha256() == original_validation.id_sha256()
    assert {item.row_id for item in original_train.examples} < {
        item.row_id for item in expanded_train.examples
    }
    assert len(expanded_train) == 24
    assert {item.row_id for item in expanded_train.examples}.isdisjoint(
        item.row_id for item in expanded_validation.examples
    )


def test_json_config_loader_preserves_expanded_split_contract(tmp_path) -> None:
    path = tmp_path / "experiment.json"
    payload = default_config().to_dict()
    payload["data"].update(
        {"train_per_class": 4_000, "validation_start_per_class": 2_000}
    )
    payload["training"]["epochs"] = 1
    payload["test_evaluations_allowed"] = 0
    path.write_text(json.dumps(payload), encoding="utf-8")

    config = load_config(path)
    assert config.data.train_per_class == 4_000
    assert config.data.validation_start_per_class == 2_000
    assert config.training.epochs == 1
    assert config.test_evaluations_allowed == 0
    assert isinstance(config.lora.target_modules, tuple)
