from __future__ import annotations

import pytest

from loraforge.config import default_config
from loraforge.provenance import EvidenceError
from loraforge.training import training_argument_kwargs


MODERN = {
    "output_dir", "num_train_epochs", "learning_rate", "per_device_train_batch_size",
    "gradient_accumulation_steps", "gradient_checkpointing", "optim", "fp16", "seed",
    "logging_steps", "save_strategy", "report_to", "remove_unused_columns", "warmup_ratio",
}


def build(supported, steps=1_000):
    return training_argument_kwargs(
        default_config(), set(supported), output_dir="/tmp/out", total_optimizer_steps=steps
    )


def test_the_frozen_protocol_maps_straight_through_when_everything_is_supported() -> None:
    kwargs = build(MODERN)
    config = default_config()
    assert kwargs["warmup_ratio"] == config.training.warmup_ratio
    assert kwargs["learning_rate"] == config.training.learning_rate
    assert kwargs["optim"] == config.training.optimizer
    assert kwargs["seed"] == config.data.seed
    assert kwargs["fp16"] is True


def test_warmup_ratio_becomes_warmup_steps_when_the_ratio_is_unavailable() -> None:
    supported = (MODERN - {"warmup_ratio"}) | {"warmup_steps"}
    kwargs = build(supported, steps=1_000)
    assert "warmup_ratio" not in kwargs
    assert kwargs["warmup_steps"] == 30  # 3% of 1,000 optimizer steps


def test_warmup_steps_never_rounds_down_to_zero_on_a_short_run() -> None:
    supported = (MODERN - {"warmup_ratio"}) | {"warmup_steps"}
    assert build(supported, steps=5)["warmup_steps"] == 1


def test_losing_warmup_entirely_is_refused() -> None:
    with pytest.raises(EvidenceError, match="warmup"):
        build(MODERN - {"warmup_ratio"})


@pytest.mark.parametrize(
    "missing", ["learning_rate", "optim", "gradient_checkpointing", "seed", "fp16"]
)
def test_dropping_an_argument_the_protocol_depends_on_is_refused(missing) -> None:
    with pytest.raises(EvidenceError, match=missing):
        build(MODERN - {missing})


def test_cosmetic_arguments_are_dropped_quietly() -> None:
    kwargs = build(MODERN - {"logging_steps", "report_to"})
    assert "logging_steps" not in kwargs and "report_to" not in kwargs
    assert kwargs["learning_rate"] == default_config().training.learning_rate


def test_no_unsupported_argument_ever_reaches_TrainingArguments() -> None:
    supported = MODERN - {"remove_unused_columns", "save_strategy"}
    assert set(build(supported)).issubset(supported)
