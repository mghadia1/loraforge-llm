"""Validation-first baseline/evaluation artifacts; test requires an explicit unlock."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import ExperimentConfig
from .data import DatasetBundle
from .metrics import evaluation_block, fit_temperature
from .modeling import peak_cuda_memory_gib, score_class_codes


def evaluate_split(model, tokenizer, split, config: ExperimentConfig) -> tuple[np.ndarray, dict[str, Any]]:
    logits = score_class_codes(
        model,
        tokenizer,
        split.texts,
        batch_size=config.training.per_device_eval_batch_size,
        max_length=config.training.max_sequence_length,
    )
    return logits, evaluation_block(logits, split.labels)


def validation_baseline(
    model, tokenizer, bundle: DatasetBundle, config: ExperimentConfig
) -> dict[str, Any]:
    logits, metrics = evaluate_split(model, tokenizer, bundle.validation, config)
    temperature = fit_temperature(logits, bundle.validation.labels)
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "model": config.model_name,
        "model_revision": config.model_revision,
        "split": "validation",
        "test_evaluated": False,
        "metrics_before_temperature": metrics,
        "validation_temperature": temperature,
        "metrics_after_temperature": evaluation_block(
            logits, bundle.validation.labels, temperature=temperature
        ),
        "peak_cuda_memory_gib": peak_cuda_memory_gib(),
    }


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
