"""Frozen experiment configuration and validation."""

from __future__ import annotations

import json
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
        if not self.quantization.load_in_4bit or self.quantization.quant_type != "nf4":
            raise ValueError("the frozen QLoRA protocol requires 4-bit NF4")
        if self.lora.rank <= 0 or self.lora.alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0 <= self.lora.dropout < 1:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if self.training.select_metric != "macro_f1":
            raise ValueError("adapter selection must use validation macro-F1")
        if self.data.train_per_class <= 0 or self.data.validation_per_class <= 0:
            raise ValueError("development split counts must be positive")
        if (
            self.data.validation_start_per_class is not None
            and self.data.validation_start_per_class < 0
        ):
            raise ValueError("validation_start_per_class must be non-negative")
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


def load_config(path: Path) -> ExperimentConfig:
    """Load and validate a JSON experiment config with typed nested sections."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    data = DataConfig(**payload.pop("data"))
    quantization = QuantizationConfig(**payload.pop("quantization"))
    lora_payload = payload.pop("lora")
    if "target_modules" in lora_payload:
        lora_payload["target_modules"] = tuple(lora_payload["target_modules"])
    lora = LoraConfig(**lora_payload)
    training = TrainingConfig(**payload.pop("training"))
    config = ExperimentConfig(
        **payload,
        data=data,
        quantization=quantization,
        lora=lora,
        training=training,
    )
    config.validate()
    return config
