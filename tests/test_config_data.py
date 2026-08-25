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


@pytest.mark.parametrize("value", [True, False])
def test_boolean_test_budget_is_rejected(tmp_path, value) -> None:
    """JSON booleans must not inherit Python's integer test-budget semantics."""
    path = tmp_path / "experiment.json"
    payload = default_config().to_dict()
    payload["test_evaluations_allowed"] = value
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON booleans are not a test budget"):
        load_config(path)


def test_resume_eligibility_cannot_be_enabled_by_config(tmp_path) -> None:
    path = tmp_path / "experiment.json"
    payload = default_config().to_dict()
    payload["resume_eligible"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="separate explanation gate"):
        load_config(path)


def test_boolean_schema_version_is_not_version_one(tmp_path) -> None:
    path = tmp_path / "experiment.json"
    payload = default_config().to_dict()
    payload["schema_version"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported experiment schema"):
        load_config(path)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("data", "seed"), True),
        (("data", "publisher_test_rows"), 7_600.0),
        (("quantization", "use_double_quant"), 1),
        (("lora", "rank"), True),
        (("lora", "dropout"), "0.05"),
        (("lora", "dropout"), float("nan")),
        (("lora", "target_modules"), "q_proj"),
        (("training", "epochs"), True),
        (("training", "learning_rate"), "0.0002"),
        (("training", "learning_rate"), 10**400),
        (("training", "per_device_train_batch_size"), 2.0),
        (("training", "gradient_checkpointing"), 1),
    ],
)
def test_nested_config_fields_keep_their_declared_types(
    tmp_path, path, replacement
) -> None:
    config_path = tmp_path / "experiment.json"
    payload = default_config().to_dict()
    section, field = path
    payload[section][field] = replacement
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=rf"{section}\.{field}"):
        load_config(config_path)


def test_missing_fields_are_not_silently_filled_from_dataclass_defaults(tmp_path) -> None:
    path = tmp_path / "experiment.json"
    payload = default_config().to_dict()
    del payload["training"]["learning_rate"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"missing=\['learning_rate'\]"):
        load_config(path)


def test_unexpected_fields_are_rejected_instead_of_becoming_untracked_protocol(tmp_path) -> None:
    path = tmp_path / "experiment.json"
    payload = default_config().to_dict()
    payload["training"]["scheduler"] = "cosine"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"unexpected=\['scheduler'\]"):
        load_config(path)


def test_config_root_and_sections_must_be_json_objects(tmp_path) -> None:
    root_path = tmp_path / "root.json"
    root_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="experiment config must be a JSON object"):
        load_config(root_path)

    section_path = tmp_path / "section.json"
    payload = default_config().to_dict()
    payload["training"] = []
    section_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="training must be a JSON object"):
        load_config(section_path)


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


@pytest.mark.parametrize(
    "row",
    [
        {"text": None, "label": 0},
        {"text": "", "label": 0},
        {"text": "article", "label": True},
        {"text": "article", "label": 1.5},
        {"text": "article", "label": 4},
        {"text": "article"},
    ],
)
def test_upstream_rows_are_validated_before_conversion(row) -> None:
    with pytest.raises(ValueError):
        data_module._to_examples("train", [row])


def test_upstream_row_ids_keep_raw_provenance_while_content_is_normalized() -> None:
    first = data_module._to_examples("train", [{"text": "same\narticle", "label": 0}])[0]
    second = data_module._to_examples("test", [{"text": "same article", "label": 0}])[0]
    assert first.row_id != second.row_id
    with pytest.raises(data_module.SplitLeakError):
        data_module.assert_disjoint(Split("train", (first,)), Split("test", (second,)))


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
