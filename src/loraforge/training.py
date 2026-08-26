"""Step 4: two frozen epochs, per-epoch validation, and validation-only selection."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from .collate import SupervisedCollator, build_supervised_features
from .config import ExperimentConfig
from .data import DatasetBundle
from .metrics import evaluation_block
from .provenance import (
    EvidenceError,
    environment,
    read_json,
    save_logits,
    sha256_directory,
    sha256_labels,
    utc_now,
    write_json,
)
from .qlora import parameter_report
from .selection import SELECTION_RULE


# Loose enough to survive a few rows flipping on non-deterministic 4-bit kernels
# across GPU sessions, tight enough that a changed prompt, tokenizer, or model
# revision cannot hide inside it.
BASELINE_AGREEMENT_TOLERANCE = 0.005


def select_checkpoint(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Highest validation macro-F1 wins; an exact tie goes to the earlier epoch."""
    if not records:
        raise ValueError("no checkpoint records to select from")
    epochs = [record["epoch"] for record in records]
    if len(set(epochs)) != len(epochs):
        raise ValueError(f"duplicate epoch numbers in checkpoint records: {epochs}")
    return min(records, key=lambda record: (-record["validation"]["macro_f1"], record["epoch"]))


class _EpochRecorder:
    """Save the adapter and score validation at the end of every epoch."""

    def __init__(self, model, tokenizer, bundle: DatasetBundle, config: ExperimentConfig, root: Path):
        from transformers import TrainerCallback

        self.model = model
        self.tokenizer = tokenizer
        self.bundle = bundle
        self.config = config
        self.root = root
        self.records: list[dict[str, Any]] = []
        self.logits: dict[int, np.ndarray] = {}
        recorder = self

        class Callback(TrainerCallback):
            def on_epoch_end(self, args, state, control, **kwargs):
                recorder.capture(round(state.epoch))

        self.callback = Callback()

    def capture(self, epoch: int) -> None:
        import torch

        from .modeling import score_class_codes

        relative_dir = f"adapters/epoch-{epoch}"
        adapter_dir = self.root / relative_dir
        self.model.save_pretrained(adapter_dir)
        was_training = self.model.training
        # Scoring alone peaked at ~11.9 GiB of a 15.4 GiB T4 in phase one, so give
        # the evaluation every byte the training step is no longer using.
        torch.cuda.empty_cache()
        started = time.perf_counter()
        logits = score_class_codes(
            self.model,
            self.tokenizer,
            self.bundle.validation.texts,
            batch_size=self.config.training.per_device_eval_batch_size,
            max_length=self.config.training.max_sequence_length,
        )
        elapsed = time.perf_counter() - started
        torch.cuda.empty_cache()
        if was_training:
            self.model.train()
        self.logits[epoch] = logits
        self.records.append(
            {
                "epoch": epoch,
                "adapter_dir": relative_dir,
                "adapter_hashes": sha256_directory(adapter_dir),
                "validation": evaluation_block(logits, self.bundle.validation.labels),
                "validation_seconds": elapsed,
            }
        )


def _base_validation_logits(model, tokenizer, bundle: DatasetBundle, config: ExperimentConfig) -> np.ndarray:
    """Score validation with the adapter switched off, so base and tuned share prompts."""
    from .modeling import score_class_codes

    with model.disable_adapter():
        return score_class_codes(
            model,
            tokenizer,
            bundle.validation.texts,
            batch_size=config.training.per_device_eval_batch_size,
            max_length=config.training.max_sequence_length,
        )


def check_baseline_agreement(recomputed: dict[str, Any], phase_one_path: Path) -> dict[str, Any]:
    """The re-scored base must reproduce the T4 phase-one baseline, or something moved."""
    recorded = read_json(phase_one_path)
    expected = recorded["metrics_before_temperature"]["macro_f1"]
    actual = recomputed["macro_f1"]
    if abs(expected - actual) > BASELINE_AGREEMENT_TOLERANCE:
        raise EvidenceError(
            f"re-scored base macro-F1 {actual} disagrees with {phase_one_path} value {expected}; "
            "the prompt, tokenizer, or model changed between phases"
        )
    return {
        "phase_one_macro_f1": expected,
        "rescored_macro_f1": actual,
        "absolute_difference": abs(expected - actual),
        "tolerance": BASELINE_AGREEMENT_TOLERANCE,
        "agrees": True,
    }


# TrainingArguments has been reshaped across transformers majors. Building the
# kwargs explicitly, then checking them against the installed signature, keeps a
# missing knob from either vanishing silently or surfacing as a bare TypeError
# after the base model has already been downloaded and loaded.
COSMETIC_ARGUMENTS = frozenset({"logging_steps", "report_to", "save_strategy", "remove_unused_columns"})


def training_argument_kwargs(
    config: ExperimentConfig,
    supported: set[str],
    *,
    output_dir: str,
    total_optimizer_steps: int,
) -> dict[str, Any]:
    """Map the frozen protocol onto whatever TrainingArguments accepts here."""
    requested: dict[str, Any] = {
        "output_dir": output_dir,
        "num_train_epochs": config.training.epochs,
        "learning_rate": config.training.learning_rate,
        "per_device_train_batch_size": config.training.per_device_train_batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "gradient_checkpointing": config.training.gradient_checkpointing,
        "optim": config.training.optimizer,
        "fp16": config.quantization.compute_dtype == "float16",
        "seed": config.data.seed,
        "logging_steps": 10,
        "save_strategy": "no",
        "report_to": [],
        "remove_unused_columns": False,
    }

    # Warmup is part of the protocol, so express it however this version allows.
    if "warmup_ratio" in supported:
        requested["warmup_ratio"] = config.training.warmup_ratio
    elif "warmup_steps" in supported:
        requested["warmup_steps"] = max(1, round(config.training.warmup_ratio * total_optimizer_steps))
    else:
        raise EvidenceError(
            "TrainingArguments accepts neither warmup_ratio nor warmup_steps; the frozen "
            f"{config.training.warmup_ratio:.0%} warmup cannot be reproduced on this version"
        )

    dropped = sorted(set(requested) - supported)
    essential = [name for name in dropped if name not in COSMETIC_ARGUMENTS]
    if essential:
        raise EvidenceError(
            f"this transformers version does not accept {essential}, which the frozen "
            "protocol depends on; pin the versions recorded in the training report "
            "rather than training with different semantics"
        )
    return {name: value for name, value in requested.items() if name in supported}


def train_qlora(
    model,
    tokenizer,
    bundle: DatasetBundle,
    config: ExperimentConfig,
    *,
    root: Path = Path("."),
) -> dict[str, Any]:
    """Run the two frozen epochs and write ``outputs/training-report.json``."""
    import torch
    from transformers import Trainer, TrainingArguments

    root = Path(root)
    if (root / "outputs" / "training-report.json").exists():
        raise EvidenceError(
            "outputs/training-report.json already exists; move it aside before retraining "
            "so a failed run is never silently overwritten"
        )

    torch.manual_seed(config.data.seed)
    # Audit first: this raises if any non-adapter weight is trainable, which is
    # worth catching in seconds rather than after hours of training.
    parameters = parameter_report(model)
    features = build_supervised_features(
        tokenizer, bundle.train, max_length=config.training.max_sequence_length
    )
    base_logits = _base_validation_logits(model, tokenizer, bundle, config)
    base_metrics = evaluation_block(base_logits, bundle.validation.labels)
    baseline_agreement = check_baseline_agreement(
        base_metrics, root / "outputs" / "base-validation.json"
    )

    recorder = _EpochRecorder(model, tokenizer, bundle, config, root)
    import inspect
    import math

    effective_batch = (
        config.training.per_device_train_batch_size * config.training.gradient_accumulation_steps
    )
    total_optimizer_steps = math.ceil(len(features) / effective_batch) * config.training.epochs
    argument_kwargs = training_argument_kwargs(
        config,
        set(inspect.signature(TrainingArguments.__init__).parameters),
        output_dir=str(root / "outputs" / "trainer"),
        total_optimizer_steps=total_optimizer_steps,
    )
    arguments = TrainingArguments(**argument_kwargs)
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=features,
        data_collator=SupervisedCollator(tokenizer.pad_token_id),
        callbacks=[recorder.callback],
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    trainer.train()
    wall_time = time.perf_counter() - started

    if len(recorder.records) != config.training.epochs:
        raise EvidenceError(
            f"expected {config.training.epochs} epoch checkpoints, captured {len(recorder.records)}"
        )
    selected = select_checkpoint(recorder.records)
    selected_relative = "adapters/selected"
    selected_dir = root / selected_relative
    if selected_dir.exists():
        shutil.rmtree(selected_dir)
    shutil.copytree(root / selected["adapter_dir"], selected_dir)

    report = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "model": config.model_name,
        "model_revision": config.model_revision,
        "config": config.to_dict(),
        "environment": environment(),
        "training_arguments": argument_kwargs,
        "parameters": parameters,
        "test_evaluated": False,
        "train_rows": len(bundle.train),
        "validation_rows": len(bundle.validation),
        "validation_label_sha256": sha256_labels(bundle.validation.labels),
        "baseline_agreement": baseline_agreement,
        "base_validation_metrics": base_metrics,
        "base_validation_logits": save_logits(
            base_logits, root, "outputs/logits/base-validation.npy"
        ),
        "epochs": [
            {
                **record,
                "validation_logits": save_logits(
                    recorder.logits[record["epoch"]],
                    root,
                    f"outputs/logits/tuned-validation-epoch-{record['epoch']}.npy",
                ),
            }
            for record in recorder.records
        ],
        "selection": {
            "rule": SELECTION_RULE,
            "selected_epoch": selected["epoch"],
            "selected_adapter_dir": selected_relative,
            "selected_adapter_hashes": sha256_directory(selected_dir),
        },
        "loss_curve": [
            entry for entry in trainer.state.log_history if "loss" in entry or "train_loss" in entry
        ],
        "wall_time_seconds": wall_time,
        "peak_cuda_memory_gib": float(torch.cuda.max_memory_allocated() / 1024**3),
    }
    write_json(report, root / "outputs" / "training-report.json")
    return report
