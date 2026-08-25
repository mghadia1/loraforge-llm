"""Frozen experiment configuration and validation."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DATASET_NAME = "fancyzhx/ag_news"
DATASET_REVISION = "eb185aade064a813bc0b7f42de02595523103ca4"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_REVISION = "c170c708c41dac9275d15a8fff4eca08d52bab71"


@dataclass(frozen=True)
class DataConfig:
    dataset_name: str = DATASET_NAME
    dataset_revision: str = DATASET_REVISION
    seed: int = 73
    train_per_class: int = 2_000
    validation_per_class: int = 500
    validation_start_per_class: int | None = None
    publisher_test_rows: int = 7_600


@dataclass(frozen=True)
class QuantizationConfig:
    load_in_4bit: bool = True
    quant_type: str = "nf4"
    compute_dtype: str = "float16"
    use_double_quant: bool = True


@dataclass(frozen=True)
class LoraConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 2
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.03
    max_sequence_length: int = 512
    gradient_checkpointing: bool = True
    optimizer: str = "paged_adamw_8bit"
    select_metric: str = "macro_f1"


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int = 1
    model_name: str = MODEL_NAME
    model_revision: str = MODEL_REVISION
    data: DataConfig = field(default_factory=DataConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    resume_eligible: bool = False
    test_evaluations_allowed: int = 1

    def validate(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported experiment schema")
        _require_string("model_name", self.model_name)
        _require_string("model_revision", self.model_revision)
        _require_string("data.dataset_name", self.data.dataset_name)
        _require_string("data.dataset_revision", self.data.dataset_revision)
        _require_integer("data.seed", self.data.seed, minimum=0)
        _require_integer("data.train_per_class", self.data.train_per_class, minimum=1)
        _require_integer(
            "data.validation_per_class", self.data.validation_per_class, minimum=1
        )
        if self.data.validation_start_per_class is not None:
            _require_integer(
                "data.validation_start_per_class",
                self.data.validation_start_per_class,
                minimum=0,
            )
        _require_integer(
            "data.publisher_test_rows", self.data.publisher_test_rows, minimum=1
        )

        _require_boolean("quantization.load_in_4bit", self.quantization.load_in_4bit)
        _require_string("quantization.quant_type", self.quantization.quant_type)
        _require_string("quantization.compute_dtype", self.quantization.compute_dtype)
        _require_boolean(
            "quantization.use_double_quant", self.quantization.use_double_quant
        )
        if not self.quantization.load_in_4bit or self.quantization.quant_type != "nf4":
            raise ValueError("the frozen QLoRA protocol requires 4-bit NF4")

        _require_integer("lora.rank", self.lora.rank, minimum=1)
        _require_integer("lora.alpha", self.lora.alpha, minimum=1)
        _require_number("lora.dropout", self.lora.dropout)
        if not 0 <= self.lora.dropout < 1:
            raise ValueError("LoRA dropout must be in [0, 1)")
        _require_string("lora.bias", self.lora.bias)
        if type(self.lora.target_modules) is not tuple or not self.lora.target_modules:
            raise ValueError("lora.target_modules must be a nonempty JSON array of strings")
        for index, module in enumerate(self.lora.target_modules):
            _require_string(f"lora.target_modules[{index}]", module)
        if len(set(self.lora.target_modules)) != len(self.lora.target_modules):
            raise ValueError("lora.target_modules must not contain duplicates")

        _require_integer("training.epochs", self.training.epochs, minimum=1)
        _require_number("training.learning_rate", self.training.learning_rate, minimum=0.0)
        if self.training.learning_rate == 0:
            raise ValueError("training.learning_rate must be greater than zero")
        _require_integer(
            "training.per_device_train_batch_size",
            self.training.per_device_train_batch_size,
            minimum=1,
        )
        _require_integer(
            "training.per_device_eval_batch_size",
            self.training.per_device_eval_batch_size,
            minimum=1,
        )
        _require_integer(
            "training.gradient_accumulation_steps",
            self.training.gradient_accumulation_steps,
            minimum=1,
        )
        _require_number("training.warmup_ratio", self.training.warmup_ratio)
        if not 0 <= self.training.warmup_ratio <= 1:
            raise ValueError("training.warmup_ratio must be in [0, 1]")
        _require_integer(
            "training.max_sequence_length",
            self.training.max_sequence_length,
            minimum=1,
        )
        _require_boolean(
            "training.gradient_checkpointing", self.training.gradient_checkpointing
        )
        _require_string("training.optimizer", self.training.optimizer)
        _require_string("training.select_metric", self.training.select_metric)
        if self.training.select_metric != "macro_f1":
            raise ValueError("adapter selection must use validation macro-F1")
        _require_boolean("resume_eligible", self.resume_eligible)
        if (
            type(self.test_evaluations_allowed) is not int
            or self.test_evaluations_allowed not in (0, 1)
        ):
            raise ValueError(
                "test_evaluations_allowed must be the integer zero or one; "
                "JSON booleans are not a test budget"
            )
        if self.resume_eligible is not False:
            raise ValueError(
                "resume_eligible must remain false until the separate explanation gate passes"
            )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if payload["data"]["validation_start_per_class"] is None:
            # Preserve the schema and hashes of the completed experiment. Older
            # configs imply that validation starts after the training prefix.
            payload["data"].pop("validation_start_per_class")
        return payload

    def write(self, path: Path) -> None:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


def default_config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.validate()
    return config


def _require_string(name: str, value: Any) -> None:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")


def _require_boolean(name: str, value: Any) -> None:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a JSON boolean")


def _require_integer(name: str, value: Any, *, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")


def _require_number(name: str, value: Any, *, minimum: float | None = None) -> None:
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{name} must be a finite JSON number")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be greater than or equal to {minimum}")


def _require_object(name: str, value: Any) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{name} must be a JSON object")
    return dict(value)


def _require_fields(
    name: str,
    payload: dict[str, Any],
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> None:
    missing = required - payload.keys()
    unexpected = payload.keys() - required - optional
    if missing or unexpected:
        raise ValueError(
            f"{name} fields differ from schema: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )


def config_from_dict(value: Any) -> ExperimentConfig:
    """Parse an exact, fully typed experiment-config object."""
    payload = _require_object("experiment config", value)
    _require_fields(
        "experiment config",
        payload,
        required=frozenset(
            {
                "schema_version",
                "model_name",
                "model_revision",
                "data",
                "quantization",
                "lora",
                "training",
                "resume_eligible",
                "test_evaluations_allowed",
            }
        ),
    )

    data_payload = _require_object("data", payload.pop("data"))
    _require_fields(
        "data",
        data_payload,
        required=frozenset(
            {
                "dataset_name",
                "dataset_revision",
                "seed",
                "train_per_class",
                "validation_per_class",
                "publisher_test_rows",
            }
        ),
        optional=frozenset({"validation_start_per_class"}),
    )
    quantization_payload = _require_object(
        "quantization", payload.pop("quantization")
    )
    _require_fields(
        "quantization",
        quantization_payload,
        required=frozenset(
            {"load_in_4bit", "quant_type", "compute_dtype", "use_double_quant"}
        ),
    )
    lora_payload = _require_object("lora", payload.pop("lora"))
    _require_fields(
        "lora",
        lora_payload,
        required=frozenset(
            {"rank", "alpha", "dropout", "bias", "target_modules"}
        ),
    )
    target_modules = lora_payload["target_modules"]
    if type(target_modules) is not list:
        raise ValueError("lora.target_modules must be a nonempty JSON array of strings")
    lora_payload["target_modules"] = tuple(target_modules)
    training_payload = _require_object("training", payload.pop("training"))
    _require_fields(
        "training",
        training_payload,
        required=frozenset(
            {
                "epochs",
                "learning_rate",
                "per_device_train_batch_size",
                "per_device_eval_batch_size",
                "gradient_accumulation_steps",
                "warmup_ratio",
                "max_sequence_length",
                "gradient_checkpointing",
                "optimizer",
                "select_metric",
            }
        ),
    )

    data = DataConfig(**data_payload)
    quantization = QuantizationConfig(**quantization_payload)
    lora = LoraConfig(**lora_payload)
    training = TrainingConfig(**training_payload)
    config = ExperimentConfig(
        **payload,
        data=data,
        quantization=quantization,
        lora=lora,
        training=training,
    )
    config.validate()
    return config


def load_config(path: Path) -> ExperimentConfig:
    """Load and validate an exact, fully typed JSON experiment config."""
    return config_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
