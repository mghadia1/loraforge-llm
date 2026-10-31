"""Step 4: two frozen epochs, per-epoch validation, and validation-only selection."""

from __future__ import annotations

import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from .collate import SupervisedCollator, build_supervised_features
from .config import ExperimentConfig
from .data import CLASS_NAMES, DatasetBundle
from .metrics import evaluation_block
from .provenance import (
    EvidenceError,
    PRETRAINING_MANIFEST_PATH,
    environment,
    read_json,
    save_logits,
    sha256_directory,
    sha256_file,
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
TRAINING_REPORT_SCHEMA_VERSION = 2


def capture_pretraining_manifest(
    config: ExperimentConfig,
    training_arguments: dict[str, Any],
    parameters: dict[str, Any],
    *,
    root: Path = Path("."),
) -> tuple[dict[str, Any], dict[str, str]]:
    """Write the immutable controls captured before optimizer training starts."""
    root = Path(root)
    target = root / PRETRAINING_MANIFEST_PATH
    if target.exists():
        raise EvidenceError(
            f"{target} already exists; move it aside before retraining so a prior "
            "attempt's pre-training controls are never silently overwritten"
        )
    manifest = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "stage": "before_optimizer_training",
        "test_loaded": False,
        "config": config.to_dict(),
        "environment": environment(),
        "training_arguments": training_arguments,
        "parameters": parameters,
    }
    write_json(manifest, target)
    # JSON normalizes tuples such as target_modules to arrays. Keep the in-memory
    # values byte-semantically aligned with the artifact that will be verified.
    manifest = read_json(target)
    reference = {
        "path": PRETRAINING_MANIFEST_PATH,
        "sha256": sha256_file(target),
    }
    return manifest, reference


def refuse_existing_training_outputs(root: Path) -> None:
    """Require a fresh phase-two output area before any costly training setup.

    A failed GPU run can leave checkpoints, logits, or Trainer state without ever
    reaching ``training-report.json``. Reusing those paths would overwrite the
    evidence from that attempt and make a later report ambiguous, so callers must
    choose a new run root instead.
    """
    root = Path(root)
    known_paths = {
        root / "outputs" / "training-report.json",
        root / PRETRAINING_MANIFEST_PATH,
        root / "outputs" / "trainer",
        root / "outputs" / "logits" / "base-validation.npy",
        root / "outputs" / "frozen-selection.json",
        root / "outputs" / "final-test-attempt.json",
        root / "outputs" / "final-test-report.json",
        root / "adapters" / "selected",
    }
    known_paths.update((root / "adapters").glob("epoch-*"))
    known_paths.update(
        (root / "outputs" / "logits").glob("tuned-validation-epoch-*.npy")
    )
    conflicts = sorted(
        path.relative_to(root).as_posix()
        for path in known_paths
        if path.exists() or path.is_symlink()
    )
    if conflicts:
        raise EvidenceError(
            "training outputs already exist and must not be overwritten: "
            f"{conflicts}; use a fresh run root and preserve the prior attempt"
        )


def select_checkpoint(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Highest validation macro-F1 wins; an exact tie goes to the earlier epoch."""
    if not records:
        raise ValueError("no checkpoint records to select from")
    epochs = []
    for index, record in enumerate(records):
        if type(record) is not dict:
            raise ValueError(f"checkpoint record {index} must be an object")
        epoch = record.get("epoch")
        if type(epoch) is not int or epoch < 1:
            raise ValueError(
                f"checkpoint record {index} epoch must be a positive integer"
            )
        validation = record.get("validation")
        if type(validation) is not dict:
            raise ValueError(
                f"checkpoint record {index} validation must be an object"
            )
        macro_f1 = validation.get("macro_f1")
        if (
            isinstance(macro_f1, bool)
            or type(macro_f1) not in (int, float)
            or not math.isfinite(macro_f1)
            or not 0.0 <= macro_f1 <= 1.0
        ):
            raise ValueError(
                f"checkpoint record {index} validation macro-F1 must be a "
                "finite JSON number in [0, 1]"
            )
        epochs.append(epoch)
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


def check_baseline_agreement(
    recomputed: dict[str, Any],
    phase_one_path: Path,
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Bind the re-scored base to the compatible phase-one validation artifact."""
    recorded = read_json(phase_one_path)
    if not isinstance(recorded, dict):
        raise EvidenceError("phase-one baseline must be a JSON object")
    expected_provenance = {
        "schema_version": 1,
        "model": config.model_name,
        "model_revision": config.model_revision,
        "split": "validation",
        "test_evaluated": False,
    }
    for field, expected_value in expected_provenance.items():
        actual_value = recorded.get(field)
        if type(actual_value) is not type(expected_value) or actual_value != expected_value:
            raise EvidenceError(
                f"phase-one baseline {field} does not match the training protocol"
            )

    metrics = recorded.get("metrics_before_temperature")
    if not isinstance(metrics, dict):
        raise EvidenceError("phase-one baseline metrics must be an object")
    expected_rows = len(CLASS_NAMES) * config.data.validation_per_class
    if type(metrics.get("rows")) is not int or metrics["rows"] != expected_rows:
        raise EvidenceError(
            "phase-one baseline row count does not match the validation protocol"
        )
    expected = metrics.get("macro_f1")
    actual = recomputed.get("macro_f1")
    if (
        isinstance(expected, bool)
        or not isinstance(expected, (int, float))
        or not np.isfinite(expected)
        or not 0.0 <= expected <= 1.0
    ):
        raise EvidenceError("phase-one baseline macro-F1 must be a finite value in [0, 1]")
    if (
        isinstance(actual, bool)
        or not isinstance(actual, (int, float))
        or not np.isfinite(actual)
        or not 0.0 <= actual <= 1.0
    ):
        raise EvidenceError("re-scored base macro-F1 must be a finite value in [0, 1]")
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


def optimizer_step_count(config: ExperimentConfig, train_rows: int) -> int:
    """Optimizer steps over the frozen epochs; a final partial batch still steps."""
    effective_batch = (
        config.training.per_device_train_batch_size * config.training.gradient_accumulation_steps
    )
    return math.ceil(train_rows / effective_batch) * config.training.epochs


def resolved_warmup_steps(config: ExperimentConfig, total_optimizer_steps: int) -> int:
    """The integer warmup that matches ``warmup_ratio``, rounded up as Transformers does.

    Training writes this value when TrainingArguments has no ``warmup_ratio``, and
    verification recomputes it here, so the two can never disagree.
    """
    return math.ceil(config.training.warmup_ratio * total_optimizer_steps)


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
        requested["warmup_steps"] = resolved_warmup_steps(config, total_optimizer_steps)
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
    refuse_existing_training_outputs(root)

    import torch
    from transformers import Trainer, TrainingArguments

    root = Path(root)

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
        base_metrics, root / "outputs" / "base-validation.json", config
    )

    recorder = _EpochRecorder(model, tokenizer, bundle, config, root)
    import inspect

    total_optimizer_steps = optimizer_step_count(config, len(features))
    argument_kwargs = training_argument_kwargs(
        config,
        set(inspect.signature(TrainingArguments.__init__).parameters),
        output_dir=str(root / "outputs" / "trainer"),
        total_optimizer_steps=total_optimizer_steps,
    )
    pretraining, pretraining_reference = capture_pretraining_manifest(
        config,
        argument_kwargs,
        parameters,
        root=root,
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
    shutil.copytree(root / selected["adapter_dir"], selected_dir)

    report = {
        "schema_version": TRAINING_REPORT_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "model": config.model_name,
        "model_revision": config.model_revision,
        "config": pretraining["config"],
        "pretraining_manifest": pretraining_reference,
        "environment": pretraining["environment"],
        "training_arguments": pretraining["training_arguments"],
        "parameters": pretraining["parameters"],
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
