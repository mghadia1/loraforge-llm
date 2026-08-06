from __future__ import annotations

import pytest

from loraforge.config import DataConfig, ExperimentConfig, default_config
from loraforge.data import (
    CLASS_NAMES,
    DatasetBundle,
    LockedTestSplitError,
    Split,
    deterministic_development_split,
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
    with pytest.raises(ValueError, match="exactly one"):
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
